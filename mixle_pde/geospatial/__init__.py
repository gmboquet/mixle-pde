"""CRS / projection utilities (workstream B1).

The keystone geospatial layer every real-world ingest card (B2-B8) builds on: acquisition geometry
arrives in whatever coordinate reference system the survey was shot in (a local UTM zone, a state
plane system, ...) and has to be reconciled -- against other surveys, against a mesh, against a basemap
-- in one common frame. This module fixes that frame to ``EPSG:4326`` (geographic lon/lat/elevation)
and provides the two primitives everything else composes: picking the right UTM zone for a site
(:func:`~mixle_pde.geospatial.crs.utm_epsg_for`), and transforming points between any two EPSG-coded
CRSes (:func:`~mixle_pde.geospatial.crs.transform_points`, :func:`~mixle_pde.geospatial.crs.to_geographic`).
"""

from __future__ import annotations

from mixle_pde.geospatial.crs import to_geographic, transform_points, utm_epsg_for

__all__ = ["utm_epsg_for", "transform_points", "to_geographic"]
