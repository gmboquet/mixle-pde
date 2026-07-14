"""InSAR line-of-sight displacement ingest (workstream G4): a subsidence GeoTIFF -> IC-4 Observations.

:func:`load_insar` reads an already-unwrapped, already-phase-to-displacement-converted line-of-sight
(LOS) displacement raster (the standard delivered product of an InSAR processing chain -- phase
unwrapping and atmospheric correction of the raw interferogram are explicitly out of scope, see the
workstream's non-goals) into ONE :class:`~mixle_pde.observations.Observation` of ``kind="insar_los"``.
An ``Observation`` already carries an arbitrary batch of points (``location`` is ``(n, 3)``, ``value``
is ``(n,)``) -- the same convention every other ingest card in :mod:`mixle_pde.io` and
:mod:`mixle_pde.capabilities` uses for a whole grid's worth of measurements -- so one raster becomes
one ``Observation`` carrying every valid pixel, not one ``Observation`` per pixel.

``rasterio`` is a heavy, optional dependency and is only imported inside :func:`load_insar` (the
``env_data.py`` / :mod:`mixle_pde.io.potfield` ``_require`` convention), so importing this module never
requires it to be installed.

The companion forward/inverse (:func:`mixle_pde.poroelastic.poroelastic_subsidence` and
:func:`mixle_pde.poroelastic.invert_deformation`) reads the LOS look vector this module attaches to
``Observation.provenance["los_vector"]``, so a posterior is fit against the SAME projection the data was
read with.
"""

from __future__ import annotations

import numpy as np

from mixle_pde.observations import Observation

#: Default line-of-sight unit vector (east, north, up) when the caller supplies none: straight up, i.e.
#: treats the raster as an already-vertical (not slant-range) displacement field.
DEFAULT_LOS_VECTOR = (0.0, 0.0, 1.0)


def _require_rasterio():
    """Import rasterio or raise a clear ImportError naming the pip extra to install."""
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "reading InSAR LOS-displacement rasters needs the optional dependency 'rasterio'. "
            "Install it with: pip install 'mixle-pde[raster]'  (or: pip install rasterio)."
        ) from e
    return rasterio


def load_insar(
    path: str,
    *,
    crs: str | None = None,
    los_vector: tuple[float, float, float] | np.ndarray | None = None,
    noise_std: float = 0.01,
) -> list[Observation]:
    """Read a single-band unwrapped LOS-displacement GeoTIFF at ``path`` into one ``insar_los`` Observation.

    Args:
        path: local GeoTIFF (or any raster ``rasterio`` can open); band 1 is the LOS displacement in
            metres (already unwrapped and phase-converted -- see module docstring).
        crs: destination CRS (an EPSG string, e.g. ``"EPSG:32611"``) to reproject the raster's
            pixel-centre coordinates into via :func:`mixle_pde.geospatial.crs.transform_points` (B1),
            when it differs from the raster's own CRS. ``None`` keeps the raster's native CRS
            (``Observation.crs`` is then that native CRS, or ``None`` if the raster has none set).
        los_vector: unit line-of-sight look vector, ``(east, north, up)`` convention, positive = ground
            motion AWAY from the satellite (the standard InSAR range-increase sign, i.e. subsidence
            under a near-vertical look reads positive). Normalized internally; defaults to straight up
            ``(0, 0, 1)``. Recorded in ``Observation.provenance["los_vector"]`` so
            :func:`mixle_pde.poroelastic.invert_deformation` projects onto the same direction the data
            was read with.
        noise_std: per-pixel LOS noise standard deviation, metres. A raw GeoTIFF carries no noise
            metadata, so every valid pixel is assigned this one typical unwrapped-InSAR precision.

    Returns:
        A single-element list holding one ``Observation(kind="insar_los", ...)`` over every finite
        (non-NaN / non-masked) pixel -- NaN pixels (the usual low-coherence gaps in a real unwrapped
        interferogram) are dropped so the observation only ever carries valid measurements.
    """
    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive.")
    rasterio = _require_rasterio()
    los = np.asarray(DEFAULT_LOS_VECTOR if los_vector is None else los_vector, dtype=float)
    norm = np.linalg.norm(los)
    if norm <= 0.0:
        raise ValueError("los_vector must be a non-zero 3-vector.")
    los = los / norm

    with rasterio.open(path) as ds:
        band = ds.read(1).astype(float)
        transform = tuple(ds.transform)[:6]
        native_crs = str(ds.crs) if ds.crs is not None else None

    ny, nx = band.shape
    a, b, c, d, e, f = transform
    x = c + a * (np.arange(nx) + 0.5)
    y = f + e * (np.arange(ny) + 0.5)
    xx, yy = np.meshgrid(x, y)  # xx[i, j] = x[j], yy[i, j] = y[i], matching band[i, j]

    valid = np.isfinite(band)
    if not np.any(valid):
        raise ValueError(f"no finite pixels in InSAR raster {path!r}.")
    n = int(valid.sum())
    xyz = np.column_stack([xx[valid], yy[valid], np.zeros(n)])
    disp = band[valid]

    out_crs = native_crs
    if crs is not None and native_crs is not None and crs != native_crs:
        from mixle_pde.geospatial.crs import transform_points

        xyz = transform_points(xyz, src_crs=native_crs, dst_crs=crs)
        out_crs = crs
    elif crs is not None:
        out_crs = crs  # no source CRS on the raster to reproject from; trust the caller's declared CRS

    noise_cov = np.full(n, float(noise_std) ** 2)
    obs = Observation(
        kind="insar_los",
        location=xyz,
        value=disp,
        noise_cov=noise_cov,
        crs=out_crs,
        modality="insar",
        provenance={"los_vector": los.tolist(), "source_path": path},
    )
    return [obs]
