"""Reusable synthetic Earth inversion harnesses."""

import unittest

import numpy as np

from mixle_pde.earth_scenarios import (
    Synthetic3DInversionResult,
    Synthetic4DAssimilationResult,
    run_synthetic_3d_geochem_geophysics_inversion,
    run_synthetic_4d_biostrat_assimilation,
)


class SyntheticEarthScenarioTest(unittest.TestCase):
    def test_3d_geophysics_geochem_harness_returns_improved_posterior(self):
        result = run_synthetic_3d_geochem_geophysics_inversion(n_samples=512, rng=np.random.default_rng(11))

        self.assertIsInstance(result, Synthetic3DInversionResult)
        self.assertEqual(result.truth.shape, (result.grid.n,))
        self.assertEqual(result.sampled_posterior.samples.shape, (512, result.grid.n))
        self.assertEqual(result.geochem_updated_posterior.samples.shape, (512, result.grid.n))
        self.assertLess(
            result.metrics["assay_cell_updated_error"],
            result.metrics["assay_cell_geophysical_error"],
        )
        self.assertLess(result.metrics["geochem_effective_sample_size"], 512)

    def test_4d_biostrat_harness_returns_improved_time_lapse_posterior(self):
        result = run_synthetic_4d_biostrat_assimilation(n_samples=512, rng=np.random.default_rng(12))

        self.assertIsInstance(result, Synthetic4DAssimilationResult)
        self.assertEqual(result.truth.shape, (2, result.grid.n))
        self.assertEqual(result.sampled_posterior.samples.shape, (512, 2, result.grid.n))
        self.assertEqual(result.biostrat_updated_posterior.samples.shape, (512, 2, result.grid.n))
        self.assertLess(
            result.metrics["biostrat_updated_final_error"],
            result.metrics["dynamics_final_error"],
        )
        self.assertGreater(result.metrics["joint_start_final_covariance"], 0.0)


if __name__ == "__main__":
    unittest.main()
