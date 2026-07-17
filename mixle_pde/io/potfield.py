"""Potential-field grid ingest (workstream B4): read a gridded gravity/magnetic raster off disk.

:mod:`mixle_pde.reductions` (the other half of this card) turns a loaded grid into a physically
useful one -- reduced-to-pole, upward-continued. This module is only the ingest half: a
:class:`PotentialFieldGrid` fixes ONE shape (a ``(ny, nx)`` band, its 1-D axis coordinates, CRS,
and affine transform) for any single-band gridded potential-field product (aeromagnetic TMI,
Bouguer gravity, ...), read via :func:`load_grid`. The CRS is carried as a plain string (the
convention IC-4 also uses for :class:`~mixle_pde.observations.Observation.crs`) rather than a
live CRS object, so this module has no import-time dependency on any particular CRS library.

``rasterio`` is a heavy, optional dependency and is only imported inside :func:`load_grid`
(``env_data.py``'s established ``_require`` convention), so importing this module never requires
it to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PotentialFieldGrid:
    """A single-band, regularly gridded potential-field raster and its geospatial framing.

    ``values`` is the ``(ny, nx)`` band exactly as read off disk (row 0 = the raster's first row,
    conventionally its north edge). ``x``/``y`` are the 1-D pixel-centre axis coordinates
    (easting/northing, or lon/lat) derived from ``transform``, so ``values[i, j]`` sits at
    ``(x[j], y[i])``. ``crs`` is the PROJ/EPSG string the raster is stored in (``None`` if the
    raster has no CRS set). ``transform`` is the 6-tuple affine ``(a, b, c, d, e, f)`` mapping
    pixel ``(col, row)`` -> world ``(x, y)`` (rasterio's own ``Affine`` ordering), assumed
    axis-aligned (``b == d == 0``, no shear/rotation) -- the common case for a delivered survey
    grid and the only case this MVP resolves ``x``/``y`` for.
    """

    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    crs: str | None
    transform: tuple


def _require_rasterio():
    """Import rasterio or raise a clear ImportError naming the pip extra to install."""
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "reading potential-field grids needs the optional dependency 'rasterio'. "
            "Install it with: pip install 'mixle-pde[raster]'  (or: pip install rasterio)."
        ) from e
    return rasterio


def load_grid(path: str) -> PotentialFieldGrid:
    """Read band 1 of a single-band GeoTIFF potential-field grid into a `PotentialFieldGrid`.

    Args:
        path: local GeoTIFF path (or any raster ``rasterio`` can open).

    Returns:
        A `PotentialFieldGrid` with ``values`` cast to ``float``, and ``x``/``y`` the pixel-centre
        axis coordinates implied by the raster's affine transform and shape.
    """
    rasterio = _require_rasterio()
    with rasterio.open(path) as ds:
        values = ds.read(1).astype(float)
        transform = tuple(ds.transform)[:6]
        crs = str(ds.crs) if ds.crs is not None else None

    ny, nx = values.shape
    a, b, c, d, e, f = transform
    x = c + a * (np.arange(nx) + 0.5)
    y = f + e * (np.arange(ny) + 0.5)
    return PotentialFieldGrid(values=values, x=x, y=y, crs=crs, transform=transform)
