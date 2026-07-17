"""IC-4 keystone conformance: CRS / projection layer (workstream B1): mixle_pde.geospatial.crs.

Two synthetic surveys of the SAME physical control points are shot in adjacent UTM zones -- one in
``EPSG:32613`` (zone 13N), one in ``EPSG:32614`` (zone 14N) -- exactly as two real acquisition crews
working near a zone boundary might each default to "their" zone. Reprojecting both into a common frame
(``EPSG:4326``) must co-register the shared control points to sub-metre precision even though the two
surveys never shared a coordinate system on disk.
"""

import csv
import os
import unittest

import numpy as np
from pyproj import Geod

from mixle_pde.geospatial.crs import to_geographic, transform_points, utm_epsg_for
from mixle_pde.observations import Observation

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "crs_control_points.csv")


def _read_control_points():
    with open(FIXTURE, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


class UtmEpsgForTest(unittest.TestCase):
    def test_zone_13n_northern_hemisphere(self):
        # -102.35 -> zone int((-102.35+180)//6)+1 = 13, northern hemisphere -> 32613.
        self.assertEqual(utm_epsg_for(-102.35, 39.10), 32613)

    def test_zone_14n_northern_hemisphere(self):
        self.assertEqual(utm_epsg_for(-101.90, 39.55), 32614)

    def test_southern_hemisphere_uses_327xx(self):
        self.assertEqual(utm_epsg_for(-70.0, -33.0), 32719)

    def test_zone_boundary_is_half_open(self):
        # exactly on a 6-degree meridian: zone int((-102+180)//6)+1 = 14 (falls in the zone to the east).
        self.assertEqual(utm_epsg_for(-102.0, 40.0), 32614)


class TransformPointsTest(unittest.TestCase):
    def test_round_trip_identity_within_millimetres(self):
        rng = np.random.default_rng(0)
        xyz = np.column_stack(
            [500000 + rng.uniform(-1e4, 1e4, 5), 4400000 + rng.uniform(-1e4, 1e4, 5), rng.uniform(0, 500, 5)]
        )
        geo = transform_points(xyz, src_crs="EPSG:32613", dst_crs="EPSG:4326")
        back = transform_points(geo, src_crs="EPSG:4326", dst_crs="EPSG:32613")
        np.testing.assert_allclose(back, xyz, atol=1e-3)

    def test_z_passes_through_unchanged(self):
        xyz = np.array([[500000.0, 4400000.0, 123.456]])
        out = transform_points(xyz, src_crs="EPSG:32613", dst_crs="EPSG:4326")
        self.assertEqual(out.shape, (1, 3))
        self.assertAlmostEqual(out[0, 2], 123.456, places=9)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            transform_points(np.zeros((3, 2)), src_crs="EPSG:32613", dst_crs="EPSG:4326")


class CoRegistrationTest(unittest.TestCase):
    """The DoD: two synthetic surveys in different UTM zones co-register to < 1 m in EPSG:4326."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _read_control_points()
        cls.geod = Geod(ellps="WGS84")

        n = len(cls.rows)
        z = np.linspace(1200.0, 1450.0, n)  # arbitrary synthetic elevations, metres

        xyz_13 = np.column_stack(
            [[float(r["utm13n_easting"]) for r in cls.rows], [float(r["utm13n_northing"]) for r in cls.rows], z]
        )
        xyz_14 = np.column_stack(
            [[float(r["utm14n_easting"]) for r in cls.rows], [float(r["utm14n_northing"]) for r in cls.rows], z]
        )

        cls.survey_a = Observation(
            kind="gravity",
            location=xyz_13,
            value=np.zeros(n),
            noise_cov=np.ones(n),
            crs="EPSG:32613",
            modality="gravity",
        )
        cls.survey_b = Observation(
            kind="gravity",
            location=xyz_14,
            value=np.zeros(n),
            noise_cov=np.ones(n),
            crs="EPSG:32614",
            modality="gravity",
        )

        cls.geo_a = to_geographic(cls.survey_a.location, src_crs=cls.survey_a.crs)
        cls.geo_b = to_geographic(cls.survey_b.location, src_crs=cls.survey_b.crs)

    def test_observation_carries_crs_and_modality(self):
        self.assertEqual(self.survey_a.crs, "EPSG:32613")
        self.assertEqual(self.survey_b.modality, "gravity")

    def test_recovers_reference_lon_lat(self):
        ref_lon = np.array([float(r["lon"]) for r in self.rows])
        ref_lat = np.array([float(r["lat"]) for r in self.rows])
        np.testing.assert_allclose(self.geo_a[:, 0], ref_lon, atol=1e-6)
        np.testing.assert_allclose(self.geo_a[:, 1], ref_lat, atol=1e-6)
        np.testing.assert_allclose(self.geo_b[:, 0], ref_lon, atol=1e-6)
        np.testing.assert_allclose(self.geo_b[:, 1], ref_lat, atol=1e-6)

    def test_elevation_untouched_by_reprojection(self):
        np.testing.assert_allclose(self.geo_a[:, 2], self.survey_a.location[:, 2])
        np.testing.assert_allclose(self.geo_b[:, 2], self.survey_b.location[:, 2])

    def test_shared_control_points_co_register_under_one_metre(self):
        _, _, dist_m = self.geod.inv(self.geo_a[:, 0], self.geo_a[:, 1], self.geo_b[:, 0], self.geo_b[:, 1])
        dist_m = np.asarray(dist_m)
        self.assertTrue(np.all(dist_m < 1.0), f"max co-registration error {dist_m.max():.4f} m >= 1 m")


if __name__ == "__main__":
    unittest.main()
