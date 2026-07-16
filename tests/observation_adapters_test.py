"""Round-trip tests for mixle_pde.io.observation_adapters (workstream MP-I2 remainder).

Each test loads a real io-module fixture through that module's own existing loader (the same
loader + fixture combination its own conformance test already exercises), adapts the result, and
checks the resulting Observation's location/value/crs fields against the source data by direct
computation -- not against a second, independently-invented expectation.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from mixle_pde.io.em_soundings import load_ert, load_mt_edi
from mixle_pde.io.las import load_las
from mixle_pde.io.observation_adapters import (
    ert_soundings_to_observation,
    mt_sounding_to_observation,
    potfield_grid_to_observation,
    well_log_curve_to_observation,
)
from mixle_pde.io.potfield import load_grid
from mixle_pde.latent import Field3D
from mixle_pde.observations import Observation

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _grid_5x5x5() -> Field3D:
    xs, ys, zs = np.meshgrid(np.arange(5), np.arange(5), np.arange(5), indexing="ij")
    coordinates = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(float)
    return Field3D(coordinates=coordinates, spacing=1.0, units="dimensionless", property_name="log_conductivity")


class PotfieldGridAdapterTest(unittest.TestCase):
    FIXTURE = os.path.join(FIXTURES, "mag_grid.tif")

    def test_adapts_shape_kind_and_crs(self):
        grid = load_grid(self.FIXTURE)
        obs = potfield_grid_to_observation(grid, kind="magnetics", noise_std=2.0)

        self.assertIsInstance(obs, Observation)
        self.assertEqual(obs.kind, "magnetics")
        self.assertEqual(obs.n, grid.values.size)
        self.assertEqual(obs.location.shape, (grid.values.size, 3))
        self.assertIsNotNone(obs.crs)
        self.assertIn("32611", obs.crs)
        self.assertTrue(obs.is_diagonal)
        np.testing.assert_allclose(obs.noise_cov, np.full(obs.n, 4.0))

    def test_pixel_location_and_value_land_at_the_right_index(self):
        grid = load_grid(self.FIXTURE)
        obs = potfield_grid_to_observation(grid, kind="gravity", noise_std=1.0, elevation=250.0)
        ny, nx = grid.values.shape
        i, j = 5, 17  # an arbitrary interior pixel
        k = i * nx + j  # row-major raveling matches the adapter's default (C-order) ravel()

        self.assertAlmostEqual(obs.location[k, 0], grid.x[j])
        self.assertAlmostEqual(obs.location[k, 1], grid.y[i])
        self.assertAlmostEqual(obs.location[k, 2], 250.0)
        self.assertAlmostEqual(obs.value[k], grid.values[i, j])

    def test_bad_kind_rejected(self):
        grid = load_grid(self.FIXTURE)
        with self.assertRaises(ValueError):
            potfield_grid_to_observation(grid, kind="assay", noise_std=1.0)


class MtSoundingAdapterTest(unittest.TestCase):
    FIXTURE = os.path.join(FIXTURES, "mt_station.edi")

    def test_apparent_resistivity_component(self):
        soundings = load_mt_edi(self.FIXTURE)
        obs = mt_sounding_to_observation(soundings, noise_std=0.5)

        self.assertEqual(obs.kind, "layered_mt_apparent_resistivity")
        self.assertEqual(obs.n, soundings.frequencies.shape[0])
        np.testing.assert_allclose(obs.value, soundings.data[:, 0])
        np.testing.assert_array_equal(obs.location, np.repeat(soundings.stations, obs.n, axis=0))
        self.assertEqual(obs.crs, soundings.crs)

    def test_log_apparent_resistivity_and_phase_components(self):
        soundings = load_mt_edi(self.FIXTURE)

        log_obs = mt_sounding_to_observation(soundings, component="log_apparent_resistivity", noise_std=0.1)
        np.testing.assert_allclose(log_obs.value, np.log(soundings.data[:, 0]))
        self.assertEqual(log_obs.kind, "layered_mt_log_apparent_resistivity")

        phase_obs = mt_sounding_to_observation(soundings, component="phase", noise_std=1.0)
        np.testing.assert_allclose(phase_obs.value, soundings.data[:, 1])
        self.assertEqual(phase_obs.kind, "layered_mt_phase")

    def test_rejects_an_ert_shaped_sounding(self):
        grid = _grid_5x5x5()
        ert = load_ert(os.path.join(FIXTURES, "ert_survey.csv"), grid)
        with self.assertRaises(ValueError):
            mt_sounding_to_observation(ert, noise_std=1.0)


class ErtSoundingAdapterTest(unittest.TestCase):
    def test_location_is_the_grid_resolved_quadrupole_centroid(self):
        grid = _grid_5x5x5()
        soundings = load_ert(os.path.join(FIXTURES, "ert_survey.csv"), grid)
        obs = ert_soundings_to_observation(soundings, grid, noise_std=5.0)

        self.assertEqual(obs.kind, "dc_resistivity")
        self.assertEqual(obs.n, soundings.schedule.shape[0])
        np.testing.assert_allclose(obs.value, soundings.data)
        expected_location = grid.coordinates[soundings.schedule].mean(axis=1)
        np.testing.assert_allclose(obs.location, expected_location)

    def test_rejects_an_mt_shaped_sounding(self):
        soundings = load_mt_edi(os.path.join(FIXTURES, "mt_station.edi"))
        grid = _grid_5x5x5()
        with self.assertRaises(ValueError):
            ert_soundings_to_observation(soundings, grid, noise_std=1.0)


class WellLogCurveAdapterTest(unittest.TestCase):
    FIXTURE = os.path.join(FIXTURES, "tiny.las")

    def test_rhob_curve_round_trips_into_a_borehole_observation(self):
        log = load_las(self.FIXTURE)
        obs = well_log_curve_to_observation(log, "RHOB", noise_std=0.02, wellhead_xy=(500.0, 1000.0))

        self.assertEqual(obs.kind, "borehole")
        self.assertEqual(obs.n, log.depth.shape[0])
        np.testing.assert_allclose(obs.value, log.curves["RHOB"])
        np.testing.assert_allclose(obs.location[:, 0], np.full(obs.n, 500.0))
        np.testing.assert_allclose(obs.location[:, 1], np.full(obs.n, 1000.0))
        np.testing.assert_allclose(obs.location[:, 2], -log.depth)

    def test_default_wellhead_is_the_origin(self):
        log = load_las(self.FIXTURE)
        obs = well_log_curve_to_observation(log, "DT", noise_std=1.0)
        np.testing.assert_allclose(obs.location[:, :2], np.zeros((obs.n, 2)))

    def test_unknown_curve_raises(self):
        log = load_las(self.FIXTURE)
        with self.assertRaises(KeyError):
            well_log_curve_to_observation(log, "NOPE", noise_std=1.0)


if __name__ == "__main__":
    unittest.main()
