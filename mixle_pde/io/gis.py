"""GIS vector I/O (workstream B8): geopandas-backed layers feeding drillhole/survey geometry.

Real-world drillhole collars, claim boundaries, and survey line work arrive as ordinary GIS vector
files -- a shapefile, a GeoPackage, a GeoJSON export from a mine-planning tool. :func:`load_vector`
is the one entry point onto that world: it hands back a :class:`GISLayer` (the ``geopandas``
frame plus its resolved CRS string) so every other loader in this module, and eventually the CRS
layer (B1) and :class:`~mixle_pde.observations.SurveyGeometry` (B7), can key off the same string
instead of re-deriving it. :func:`drillhole_collars` is the first concrete consumer: a point layer
with an elevation column becomes the ``(n, 3)`` collar XYZ array a downhole-survey / mesh-mapping
step expects.

``geopandas`` (and its optional ``fiona``/``pyogrio`` vector drivers) is imported lazily inside the
functions below, never at module import time, so importing :mod:`mixle_pde` or this module never
requires the GIS stack -- only calling a loader does. A missing dependency raises a plain
``ImportError`` naming the pip extra to install (see ``pyproject.toml``'s ``data`` extra), the same
convention :func:`mixle_pde.env_data._require` establishes for the other optional dataset loaders.

Reprojecting whole rasters and repairing invalid vector topology are explicitly out of scope here
(B1 handles point-level CRS conversion; map digitization is workstream I).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_geopandas():
    """Import ``geopandas``, or raise a clear ImportError naming the extra to install."""
    try:
        import geopandas
    except ImportError as e:
        raise ImportError(
            "reading GIS vector layers needs the optional dependency 'geopandas'. "
            "Install it with: pip install 'mixle-pde[data]'  (or: pip install geopandas)."
        ) from e
    return geopandas


@dataclass
class GISLayer:
    """A vector layer loaded from disk: the geopandas frame plus its resolved CRS string.

    ``crs`` is ``None`` only when the source file itself carries no CRS metadata; callers that need
    a CRS to relate this layer to other survey data (B1) should treat ``None`` as "unknown, not
    mesh-local" and ask the data provider rather than assume a default.
    """

    frame: Any  # geopandas.GeoDataFrame
    crs: str | None


def load_vector(path: str) -> GISLayer:
    """Load a vector file (shapefile, GeoPackage, GeoJSON, ...) into a :class:`GISLayer`.

    Args:
        path: local path to any OGR-readable vector format; the driver is inferred from the file
            itself (geopandas/pyogrio convention), never downloaded or vendored.

    Returns:
        A :class:`GISLayer` wrapping the ``geopandas.GeoDataFrame`` and its CRS as a plain string
        (``None`` if the file carries no CRS).
    """
    geopandas = _require_geopandas()
    gdf = geopandas.read_file(path)
    crs = str(gdf.crs) if gdf.crs is not None else None
    return GISLayer(frame=gdf, crs=crs)


def drillhole_collars(path: str, *, elevation_field: str = "Z") -> np.ndarray:
    """Drillhole collar XYZ from a point vector layer (e.g. a collars shapefile/GeoJSON).

    Stacks the point geometry's ``x``/``y`` with an elevation column into the ``(n, 3)`` collar
    array a downhole-survey step (dip/azimuth desurveying) or a :class:`~mixle_pde.observations.
    SurveyGeometry` mesh-mapping consumes; it does not itself resolve CRS or map to mesh nodes.

    Args:
        path: local vector file with point geometry and an elevation column.
        elevation_field: column holding the collar elevation (default ``"Z"``); falls back to
            ``"elevation"`` if ``elevation_field`` is absent from the layer.

    Returns:
        ``(n, 3)`` float array of ``[x, y, z]`` collar coordinates in the layer's native CRS.

    Raises:
        KeyError: neither ``elevation_field`` nor ``"elevation"`` is a column on the layer.
    """
    layer = load_vector(path)
    gdf = layer.frame
    field = elevation_field if elevation_field in gdf.columns else "elevation"
    if field not in gdf.columns:
        raise KeyError(f"{path!r} has no {elevation_field!r} or 'elevation' column to use as collar Z.")
    x = gdf.geometry.x.to_numpy(dtype=float)
    y = gdf.geometry.y.to_numpy(dtype=float)
    z = gdf[field].to_numpy(dtype=float)
    return np.stack([x, y, z], axis=1)
