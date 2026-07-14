"""SEG-Y seismic gather ingest (mixle_pde.io.segy)."""

import unittest

import numpy as np

from mixle_pde.io.segy import SeismicGather, load_segy


class LoadSegyTest(unittest.TestCase):
    def test_tiny_fixture_shapes_and_dt(self):
        g = load_segy("tests/fixtures/tiny.sgy")

        self.assertIsInstance(g, SeismicGather)
        self.assertEqual(g.traces.shape, (5, 50))
        self.assertAlmostEqual(g.dt, 0.004, delta=1e-6)
        self.assertEqual(g.source_xyz.shape, (5, 3))
        self.assertTrue(np.all(np.isfinite(g.traces)))
        self.assertTrue(np.all(np.isfinite(g.source_xyz)))

    def test_receiver_xyz_and_cdp_shapes(self):
        g = load_segy("tests/fixtures/tiny.sgy")

        self.assertEqual(g.receiver_xyz.shape, (5, 3))
        self.assertEqual(g.cdp.shape, (5,))
        self.assertTrue(np.all(np.isfinite(g.receiver_xyz)))
        self.assertTrue(np.all(np.isfinite(g.cdp)))
        # fixture's CDP header is 100..104
        self.assertTrue(np.array_equal(g.cdp, np.array([100, 101, 102, 103, 104])))

    def test_source_group_scalar_descaling(self):
        # fixture's SourceGroupScalar is -10 (divide by 10); source_x true values are 500000+10*i
        g = load_segy("tests/fixtures/tiny.sgy")
        expected_source_x = 500000.0 + 10.0 * np.arange(5)
        self.assertTrue(np.allclose(g.source_xyz[:, 0], expected_source_x))

    def test_crs_defaults_to_none_and_is_stamped_when_given(self):
        g = load_segy("tests/fixtures/tiny.sgy")
        self.assertIsNone(g.crs)

        g_crs = load_segy("tests/fixtures/tiny.sgy", crs="EPSG:32615")
        self.assertEqual(g_crs.crs, "EPSG:32615")


if __name__ == "__main__":
    unittest.main()
