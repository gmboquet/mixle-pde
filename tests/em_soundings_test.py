"""Tests for EM/ERT/MT/AEM sounding ingest (workstream B5, mixle_pde.io.em_soundings)."""

import os
import unittest

import numpy as np
import torch

from mixle_pde.geophysics import dc_resistivity
from mixle_pde.io.em_soundings import SoundingSet, load_ert, load_mt_edi
from mixle_pde.latent import Field3D

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _grid_5x5x5() -> Field3D:
    xs, ys, zs = np.meshgrid(np.arange(5), np.arange(5), np.arange(5), indexing="ij")
    coordinates = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(float)
    return Field3D(coordinates=coordinates, spacing=1.0, units="dimensionless", property_name="log_conductivity")


class LoadErtTest(unittest.TestCase):
    def test_schedule_is_integer_array_with_valid_node_indices(self):
        grid = _grid_5x5x5()
        soundings = load_ert(os.path.join(FIXTURES, "ert_survey.csv"), grid)
        self.assertIsInstance(soundings, SoundingSet)
        self.assertTrue(np.issubdtype(soundings.schedule.dtype, np.integer))
        self.assertEqual(soundings.schedule.ndim, 2)
        self.assertEqual(soundings.schedule.shape[1], 4)
        self.assertTrue(np.all(soundings.schedule >= 0))
        self.assertTrue(np.all(soundings.schedule < grid.n))

    def test_dc_resistivity_forward_runs_on_the_mapped_schedule(self):
        # This is the Definition-of-Done assertion: no hand-built indices anywhere in this test --
        # the schedule comes entirely from load_ert's electrode -> nearest-node mapping.
        grid = _grid_5x5x5()
        soundings = load_ert(os.path.join(FIXTURES, "ert_survey.csv"), grid)
        torch.set_default_dtype(torch.float64)
        log_sigma0 = torch.zeros(grid.n, dtype=torch.float64)
        out = dc_resistivity(log_sigma0, (5, 5, 5), soundings.schedule)
        self.assertEqual(len(out), len(soundings.schedule))
        self.assertTrue(torch.isfinite(out).all())

    def test_stations_and_data_and_crs_passthrough(self):
        grid = _grid_5x5x5()
        soundings = load_ert(os.path.join(FIXTURES, "ert_survey.csv"), grid, crs="EPSG:32612")
        self.assertEqual(soundings.stations.shape[1], 3)
        self.assertEqual(soundings.stations.shape[0], 6)  # 6 electrodes in the fixture
        self.assertEqual(soundings.crs, "EPSG:32612")
        self.assertEqual(soundings.data.shape, (soundings.schedule.shape[0],))
        self.assertIsNone(soundings.frequencies)


class LoadMtEdiTest(unittest.TestCase):
    def test_frequencies_and_data_shapes(self):
        soundings = load_mt_edi(os.path.join(FIXTURES, "mt_station.edi"))
        self.assertIsInstance(soundings, SoundingSet)
        self.assertEqual(soundings.stations.shape, (1, 3))
        self.assertIsNone(soundings.schedule)
        self.assertEqual(soundings.frequencies.shape, (5,))
        self.assertEqual(soundings.data.shape, (5, 2))
        self.assertTrue(np.all(soundings.frequencies > 0))
        self.assertTrue(np.all(np.diff(soundings.frequencies) < 0))  # fixture is high-to-low frequency
        self.assertTrue(np.all(soundings.data[:, 0] > 0))  # apparent resistivity, Ohm-m
        self.assertTrue(np.all(soundings.data[:, 1] > 0))  # phase, degrees

    def test_station_xyz_uses_b1_utm_projection_by_default(self):
        # fixture LAT/LONG (32.26N, -110.94W) falls in UTM zone 12N -> EPSG:32612 (workstream B1).
        soundings = load_mt_edi(os.path.join(FIXTURES, "mt_station.edi"))
        self.assertTrue(np.all(np.isfinite(soundings.stations)))
        self.assertEqual(soundings.crs, "EPSG:32612")
        # a real UTM easting/northing, not raw lon/lat degrees
        self.assertGreater(abs(soundings.stations[0, 0]), 1000.0)
        self.assertGreater(abs(soundings.stations[0, 1]), 1000.0)
        self.assertEqual(soundings.stations[0, 2], 1200.0)  # ELEV passes through unchanged

    def test_explicit_crs_overrides_the_default_label(self):
        soundings = load_mt_edi(os.path.join(FIXTURES, "mt_station.edi"), crs="EPSG:32612")
        self.assertEqual(soundings.crs, "EPSG:32612")


if __name__ == "__main__":
    unittest.main()
