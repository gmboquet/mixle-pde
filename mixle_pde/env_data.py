"""Ocean/atmosphere environmental-medium assembler: real-world profiles/rasters onto the (nz, nr) coefficient
fields the sonar/radar solvers eat.

This is not a mineral/geoscience field-data ingest path; SEG-Y/LAS/potential-field/assay ingest is workstream B
(absent today). What lives here is the ocean/atmosphere propagation-medium pipeline: GEBCO bathymetry, WOA/Argo
sound-speed climatology, DEM terrain, and ERA5 refractivity profiles (via the ``load_*`` functions below),
assembled onto the solver grids that ``parabolic_equation``/``helmholtz_pml`` consume.

The parabolic-equation and Helmholtz forwards in this package (``parabolic_equation``, ``helmholtz_pml``) run on
a regular range-depth grid and want a single flat per-node coefficient field: a sound speed ``c(z, r)`` for
sonar, a modified refractivity ``M(z, r)`` for radar, or the squared slowness ``m = 1/c^2`` that the Helmholtz
operator assembles from. The world does not hand you that grid. It hands you a vertical profile measured at a
few depths, maybe a handful of such profiles along the track (range-varying columns), and a bathymetry (ocean)
or terrain (atmosphere) depth ``D(r)`` that bounds the medium. This module lands those samples on the solver
grid by differentiable linear interpolation and flags the nodes outside the medium.

The core :func:`assemble_field` is dependency-light (numpy, and torch only when a torch profile is passed) and
differentiable: gradients flow from a solver-side scalar back through the interpolation to the profile control
points, so the profile itself is an invertible latent driver behind a sonar/radar observation. The bathymetry
helper :func:`seabed_mask` marks below-seabed / above-terrain nodes so they can be flagged or filled.

The optional loaders (:func:`load_gebco`, :func:`load_woa_argo`, :func:`load_dem`, :func:`load_era5_profile`)
are thin interfaces onto real ocean/atmosphere datasets (GEBCO, WOA-Argo, DEM, ERA5). They import their heavy
backend (xarray / rasterio / cfgrib) *inside* the function and raise a clear ImportError naming the extra to
install if it is absent, so importing this module and running the core never needs any of them. They read a
local file plus a lon/lat/time subset and return plain numpy arrays that feed :func:`assemble_field`; they never
download or vendor data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "assemble_field",
    "seabed_mask",
    "apply_mask",
    "load_gebco",
    "load_woa_argo",
    "load_dem",
    "load_era5_profile",
]


def _is_torch(x: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.is_tensor(x)


def _interp1d(x_new, xp, fp):
    """Differentiable 1-D linear interpolation ``fp(xp) -> value at x_new``, clamped to the endpoints.

    ``xp`` must be ascending. Works on numpy arrays and (when ``fp`` is a torch tensor) autograd tensors, so
    gradients flow from the result back through ``fp``. For a torch ``fp`` the interpolation weights come from
    numpy ``xp``/``x_new`` (grid coordinates are data, not parameters) and the gather/blend stays in torch.
    """
    xp = np.asarray(xp, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if xp.ndim != 1 or xp.size < 2:
        raise ValueError("interpolation nodes xp must be 1-D with at least two points.")
    if np.any(np.diff(xp) <= 0.0):
        raise ValueError("interpolation nodes xp must be strictly ascending.")

    # locate each query in [xp[lo], xp[lo+1]] and form the linear blend weight
    xc = np.clip(x_new, xp[0], xp[-1])
    hi = np.clip(np.searchsorted(xp, xc, side="right"), 1, xp.size - 1)
    lo = hi - 1
    denom = xp[hi] - xp[lo]
    w = (xc - xp[lo]) / denom  # weight on the upper node

    if _is_torch(fp):
        import torch

        lo_t = torch.as_tensor(lo, dtype=torch.long)
        hi_t = torch.as_tensor(hi, dtype=torch.long)
        w_t = torch.as_tensor(w, dtype=fp.dtype if fp.is_floating_point() else torch.float64)
        return fp[lo_t] * (1.0 - w_t) + fp[hi_t] * w_t
    fp = np.asarray(fp, dtype=float)
    return fp[lo] * (1.0 - w) + fp[hi] * w


def assemble_field(profile_depths, profile_values, z_grid, r_grid, *, ranges=None):
    """Land a vertical profile (optionally range-varying) onto a regular ``(nz, nr)`` grid, returned flat.

    The solver grid is C-order flattened with depth as axis 0 and range as axis 1, so node ``(iz, ir)`` is at
    flat index ``iz * nr + ir`` -- the convention :mod:`mixle_pde.helmholtz_pml` and
    :mod:`mixle_pde.parabolic_equation` assemble against. Each grid depth is filled by differentiable linear
    interpolation of the profile in depth (clamped to the profile's endpoints), and if range-varying columns are
    given, by a second linear interpolation across range.

    Two modes:

    * Range-independent: ``profile_values`` is 1-D of length ``len(profile_depths)``. Every range column gets the
      same interpolated profile.
    * Range-varying: ``profile_values`` is 2-D ``(n_depths, n_cols)`` with a strictly-ascending ``ranges`` of the
      ``n_cols`` sample ranges. Each column is interpolated in depth onto ``z_grid``, then the columns are
      linearly interpolated in range onto ``r_grid`` (clamped to the sampled range span).

    Differentiable: if ``profile_values`` is a torch tensor the whole map is torch and gradients flow back to the
    profile control points (the interpolation weights are numpy, being grid geometry, not parameters).

    Args:
        profile_depths: 1-D ascending depths of the profile samples (metres), length ``n_depths``.
        profile_values: profile field at those depths -- 1-D ``(n_depths,)`` or 2-D ``(n_depths, n_cols)``.
        z_grid: 1-D grid depths (metres), length ``nz`` (axis 0 of the field).
        r_grid: 1-D grid ranges (metres), length ``nr`` (axis 1 of the field).
        ranges: sample ranges for the range-varying case, ascending, length ``n_cols``; ``None`` otherwise.

    Returns:
        flat field of length ``nz * nr`` (numpy array, or torch tensor if ``profile_values`` is torch),
        C-order over ``(iz, ir)``.
    """
    profile_depths = np.asarray(profile_depths, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    r_grid = np.asarray(r_grid, dtype=float)
    nz, nr = z_grid.size, r_grid.size
    torch_mode = _is_torch(profile_values)

    if torch_mode:
        import torch

        ndim = profile_values.dim()
    else:
        profile_values = np.asarray(profile_values, dtype=float)
        ndim = profile_values.ndim

    if ndim == 1:
        # range-independent: interpolate once in depth, broadcast across range
        col = _interp1d(z_grid, profile_depths, profile_values)  # length nz
        if torch_mode:
            return col.reshape(nz, 1).expand(nz, nr).reshape(-1).contiguous()
        return np.broadcast_to(col.reshape(nz, 1), (nz, nr)).reshape(-1).copy()

    if ndim != 2:
        raise ValueError("profile_values must be 1-D (range-independent) or 2-D (n_depths, n_cols).")
    if ranges is None:
        raise ValueError("range-varying profile_values (2-D) requires ascending 'ranges' for the columns.")
    ranges = np.asarray(ranges, dtype=float)
    n_cols = profile_values.shape[1]
    if ranges.size != n_cols:
        raise ValueError(f"ranges has {ranges.size} entries but profile_values has {n_cols} columns.")

    # interpolate every sampled column onto z_grid, then blend across range onto r_grid
    if torch_mode:
        cols = torch.stack([_interp1d(z_grid, profile_depths, profile_values[:, c]) for c in range(n_cols)], dim=1)
        field = _interp1d_rows(r_grid, ranges, cols)  # (nz, nr)
        return field.reshape(-1).contiguous()
    cols = np.stack([_interp1d(z_grid, profile_depths, profile_values[:, c]) for c in range(n_cols)], axis=1)
    field = _interp1d_rows(r_grid, ranges, cols)
    return field.reshape(-1).copy()


def _interp1d_rows(x_new, xp, fp_rows):
    """Linear interpolation across columns for every row of ``fp_rows`` (shape ``(nz, n_cols)``).

    Interpolates each row independently at the query points ``x_new`` over ascending nodes ``xp``, returning
    ``(nz, len(x_new))``. Differentiable through a torch ``fp_rows``.
    """
    xp = np.asarray(xp, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    xc = np.clip(x_new, xp[0], xp[-1])
    hi = np.clip(np.searchsorted(xp, xc, side="right"), 1, xp.size - 1)
    lo = hi - 1
    w = (xc - xp[lo]) / (xp[hi] - xp[lo])
    if _is_torch(fp_rows):
        import torch

        lo_t = torch.as_tensor(lo, dtype=torch.long)
        hi_t = torch.as_tensor(hi, dtype=torch.long)
        w_t = torch.as_tensor(w, dtype=fp_rows.dtype if fp_rows.is_floating_point() else torch.float64)
        return fp_rows[:, lo_t] * (1.0 - w_t) + fp_rows[:, hi_t] * w_t
    return fp_rows[:, lo] * (1.0 - w) + fp_rows[:, hi] * w


def seabed_mask(depth_profile, z_grid, r_grid, *, terrain=False):
    """Boolean ``(nz, nr)`` mask (flattened) of nodes outside the propagating medium, given a boundary ``D(r)``.

    ``depth_profile`` is the seabed depth (ocean) or terrain elevation (atmosphere) at each grid range,
    length ``nr``. In the default ocean mode a node is *outside* (masked True) when its depth exceeds the local
    seabed depth, ``z_grid[iz] > D(r_grid[ir])``: the water column is above the seabed. With ``terrain=True`` the
    sense flips for a bottom-referenced atmosphere grid: a node is outside when it sits *below* the terrain,
    ``z_grid[iz] < D(r_grid[ir])``.

    Args:
        depth_profile: boundary depth/elevation per grid range (metres), length ``nr``.
        z_grid: grid depths, length ``nz``.
        r_grid: grid ranges, length ``nr`` (used only for its length / shape agreement).
        terrain: if True treat the boundary as terrain below the domain (flip the inequality).

    Returns:
        flat boolean numpy array of length ``nz * nr``, True where the node is outside the medium.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    r_grid = np.asarray(r_grid, dtype=float)
    D = np.asarray(depth_profile, dtype=float)
    nz, nr = z_grid.size, r_grid.size
    if D.size != nr:
        raise ValueError(f"depth_profile has {D.size} entries but r_grid has {nr}.")
    zz = z_grid.reshape(nz, 1)
    Dr = D.reshape(1, nr)
    outside = zz < Dr if terrain else zz > Dr
    return outside.reshape(-1).copy()


def apply_mask(field, mask, fill):
    """Fill the masked (outside-medium) nodes of a flat field with ``fill``, differentiable through the rest.

    ``field`` and ``mask`` are flat length ``nz*nr``; ``mask`` True marks a node to overwrite with the scalar
    ``fill`` (e.g. a seabed sound speed, or NaN to flag). The unmasked nodes pass through untouched, so a torch
    ``field`` keeps its gradient there.
    """
    m = np.asarray(mask, dtype=bool)
    if _is_torch(field):
        import torch

        keep = torch.as_tensor(~m)
        fill_t = torch.as_tensor(float(fill), dtype=field.dtype)
        return torch.where(keep, field, fill_t)
    out = np.asarray(field, dtype=float).copy()
    out[m] = float(fill)
    return out


# --- optional dataset loaders (heavy import guarded inside each function; never download/vendor data) --------
def _require(module: str, extra: str, dataset: str):
    """Import an optional backend or raise a clear ImportError naming the pip extra to install."""
    try:
        return __import__(module)
    except ImportError as e:
        raise ImportError(
            f"reading {dataset} needs the optional dependency '{module}'. "
            f"Install it with: pip install 'mixle-pde[{extra}]'  (or: pip install {module})."
        ) from e


def load_gebco(path, *, lon, lat, var="elevation"):
    """GEBCO bathymetry (netCDF) -> a 2-D elevation raster subset, as plain numpy arrays.

    Requires ``xarray`` (extra ``netcdf``). Opens the local GEBCO grid at ``path`` and selects the lon/lat
    window (each a ``(min, max)`` pair), returning ``(lon, lat, elevation)`` numpy arrays where ``elevation``
    is metres (negative below sea level, the GEBCO convention). Never downloads; ``path`` must be a local file.

    Args:
        path: local netCDF path.
        lon: ``(lon_min, lon_max)`` selection window (degrees east).
        lat: ``(lat_min, lat_max)`` selection window (degrees north).
        var: elevation variable name in the file (default ``"elevation"``).

    Returns:
        ``(lon, lat, elevation)`` numpy arrays; ``elevation`` shape ``(len(lat), len(lon))``.
    """
    xr = _require("xarray", "netcdf", "GEBCO bathymetry")
    ds = xr.open_dataset(path)
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    sub = ds[var].sel({lon_name: slice(lon[0], lon[1]), lat_name: slice(lat[0], lat[1])})
    return np.asarray(sub[lon_name]), np.asarray(sub[lat_name]), np.asarray(sub.values, dtype=float)


def load_woa_argo(path, *, lon, lat, var="temperature", depth_name="depth", time=None):
    """World Ocean Atlas / Argo T-S (netCDF) -> a vertical profile at a lon/lat (time optional), as numpy arrays.

    Requires ``xarray`` (extra ``netcdf``). Opens ``path`` and selects the nearest lon/lat (and ``time`` if
    given), returning ``(depth, values)`` numpy arrays: the vertical profile of ``var`` (temperature or
    salinity) that feeds :func:`assemble_field` as ``profile_depths`` / ``profile_values``.

    Args:
        path: local netCDF path.
        lon: longitude (degrees east) of the column.
        lat: latitude (degrees north) of the column.
        var: variable name (default ``"temperature"``; e.g. ``"salinity"``).
        depth_name: name of the depth coordinate (default ``"depth"``).
        time: optional time selector passed to ``.sel(method="nearest")``.

    Returns:
        ``(depth, values)`` numpy arrays, ascending in depth.
    """
    xr = _require("xarray", "netcdf", "World Ocean Atlas / Argo profiles")
    ds = xr.open_dataset(path)
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    sel = {lon_name: lon, lat_name: lat}
    if time is not None:
        sel["time"] = time
    prof = ds[var].sel(sel, method="nearest")
    depth = np.asarray(prof[depth_name], dtype=float)
    order = np.argsort(depth)
    return depth[order], np.asarray(prof.values, dtype=float)[order]


def load_dem(path, *, window=None):
    """SRTM / Copernicus DEM (GeoTIFF/raster) -> a terrain-elevation array + transform, as plain numpy arrays.

    Requires ``rasterio`` (extra ``raster``). Reads band 1 of the local raster at ``path`` (optionally a pixel
    ``window=(row_off, col_off, height, width)``), returning ``(elevation, transform)`` where ``elevation`` is
    metres and ``transform`` is the six affine coefficients (pixel -> world) as a numpy array.

    Args:
        path: local GeoTIFF / raster path.
        window: optional ``(row_off, col_off, height, width)`` pixel window; ``None`` reads the full raster.

    Returns:
        ``(elevation, transform)``: elevation ``(H, W)`` numpy array, transform length-6 numpy array.
    """
    rio = _require("rasterio", "raster", "SRTM / Copernicus DEM rasters")
    with rio.open(path) as src:
        if window is None:
            arr = src.read(1)
        else:
            from rasterio.windows import Window

            arr = src.read(1, window=Window(window[1], window[0], window[3], window[2]))
        t = src.transform
        transform = np.array([t.a, t.b, t.c, t.d, t.e, t.f], dtype=float)
    return np.asarray(arr, dtype=float), transform


def load_era5_profile(path, *, lon, lat, time=None, var="t"):
    """ERA5 / radiosonde atmospheric profile (GRIB/NetCDF) -> a vertical profile at a lon/lat, as numpy arrays.

    Requires ``cfgrib`` (and ``xarray``) for GRIB, or just ``xarray`` for NetCDF (extra ``grib``). Opens ``path``
    with the cfgrib engine when the suffix is GRIB, selects the nearest lon/lat (and ``time`` if given), and
    returns ``(pressure_level, values)`` numpy arrays ordered by ascending pressure level (hPa), i.e. top of
    atmosphere first. Feed with the modified refractivity or temperature that drives the radar PE.

    Args:
        path: local GRIB or NetCDF path.
        lon: longitude (degrees east).
        lat: latitude (degrees north).
        time: optional time selector (nearest).
        var: variable name (default ``"t"`` temperature).

    Returns:
        ``(level, values)`` numpy arrays, ascending in pressure level.
    """
    xr = _require("xarray", "grib", "ERA5 / radiosonde atmospheric profiles")
    if str(path).lower().endswith((".grib", ".grib2", ".grb")):
        _require("cfgrib", "grib", "ERA5 GRIB files")
        ds = xr.open_dataset(path, engine="cfgrib")
    else:
        ds = xr.open_dataset(path)
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    level_name = "isobaricInhPa" if "isobaricInhPa" in ds.coords else "level"
    sel = {lon_name: lon, lat_name: lat}
    if time is not None:
        sel["time"] = time
    prof = ds[var].sel(sel, method="nearest")
    level = np.asarray(prof[level_name], dtype=float)
    order = np.argsort(level)
    return level[order], np.asarray(prof.values, dtype=float)[order]
