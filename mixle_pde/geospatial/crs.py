"""IC-4 keystone: CRS / projection primitives (workstream B1).

Every ingest card downstream (SEG-Y, LAS, potential-field grids, EM/ERT/MT soundings, geochem assays,
raster/vector GIS) reads acquisition geometry in whatever CRS the survey was shot in -- almost always a
local UTM zone -- and needs it reconciled into one common frame before it can sit next to another
survey, a mesh, or a basemap. This module supplies exactly that: :func:`utm_epsg_for` picks the UTM
zone for a site from its lon/lat, :func:`transform_points` moves an ``(n, 3)`` point cloud between any
two EPSG-coded CRSes via :mod:`pyproj`, and :func:`to_geographic` is the common special case of
transforming into ``EPSG:4326`` (lon, lat, elevation).

Deliberately out of scope (see the work order's non-goals): no datum-grid shifts (NADCON/HARN), no
vertical-datum transforms (orthometric vs. ellipsoidal height), no gridding/resampling. ``z`` (elevation)
passes through untouched -- only the horizontal (x, y) / (lon, lat) pair is reprojected.
"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer


def utm_epsg_for(lon: float, lat: float) -> int:
    """The EPSG code of the UTM zone containing ``(lon, lat)``.

    Northern-hemisphere zones are ``326zz`` (WGS84 / UTM zone ``zz`` N); southern-hemisphere zones are
    ``327zz``. ``lat >= 0`` (including the equator) maps to the northern-hemisphere family, matching the
    EPSG convention.
    """
    zone = int((lon + 180.0) // 6.0) + 1
    base = 32600 if lat >= 0 else 32700
    return base + zone


def transform_points(xyz: np.ndarray, *, src_crs: str, dst_crs: str) -> np.ndarray:
    """Reproject an ``(n, 3)`` array of ``(x, y, z)`` points from ``src_crs`` to ``dst_crs``.

    ``src_crs``/``dst_crs`` are EPSG strings (e.g. ``"EPSG:32613"``). Uses
    ``pyproj.Transformer.from_crs(..., always_xy=True)`` so the horizontal pair is always consumed and
    returned as ``(x, y)`` (or ``(lon, lat)``) regardless of a CRS's native axis order; ``z`` passes
    through unchanged (no vertical-datum transform -- see module docstring).
    """
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be an (n, 3) array of (x, y, z) points, got shape {xyz.shape}.")
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(xyz[:, 0], xyz[:, 1])
    return np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float), xyz[:, 2]])


def to_geographic(xyz: np.ndarray, *, src_crs: str) -> np.ndarray:
    """Reproject ``(n, 3)`` points from ``src_crs`` into geographic ``EPSG:4326`` (lon, lat, z)."""
    return transform_points(xyz, src_crs=src_crs, dst_crs="EPSG:4326")
