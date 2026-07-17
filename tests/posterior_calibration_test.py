"""Posterior calibration diagnostics for Workstream G acceptance."""

import unittest

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator
from mixle_pde.posterior_calibration import (
    heldout_observation_check,
    identifiability_diagnostic,
    observation_sensitivity,
    truth_coverage,
    uncertainty_inflation,
)


def _grid():
    coords = np.array([[float(i), 0.0, -10.0] for i in range(6)])
    return Field3D(coords, spacing=1.0, units="ppm", property_name="cu_ppm")


class PosteriorCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.grid = _grid()
        self.registry = ForwardOperatorRegistry()
        self.registry.register(borehole_forward_operator())
        self.truth = np.array([2.0, 3.0, 4.0, 8.0, 12.0, 16.0])

    def _posterior(self):
        mean = self.truth + np.array([0.1, -0.1, 0.0, 0.2, -0.2, 0.0])
        cov = np.diag(np.array([0.25, 0.25, 0.25, 4.0, 9.0, 16.0]))
        return PosteriorField3D(self.grid, mean=mean, dense_cov=cov)

    def test_truth_coverage_counts_cells_inside_credible_intervals(self):
        report = truth_coverage(self._posterior(), self.truth, alpha=0.1)
        self.assertEqual(report.total, self.grid.n)
        self.assertGreaterEqual(report.coverage, 0.9)
        self.assertAlmostEqual(report.expected_coverage, 0.9)
        self.assertLess(report.mean_abs_error, 0.2)

    def test_heldout_observation_check_scores_linear_predictive_distribution(self):
        posterior = self._posterior()
        held = Observation(
            "borehole",
            self.grid.coordinates[[0, 3, 5]],
            self.truth[[0, 3, 5]],
            np.full(3, 0.25),
        )
        fit = heldout_observation_check(posterior, self.registry, [held], alpha=0.1)
        self.assertEqual(fit.n_observations, 3)
        self.assertTrue(np.isfinite(fit.log_likelihood))
        self.assertLess(fit.standardized_rmse, 1.0)
        self.assertGreaterEqual(fit.coverage, 2.0 / 3.0)

    def test_uncertainty_inflates_away_from_sensitive_cells(self):
        posterior = self._posterior()
        obs = Observation("borehole", self.grid.coordinates[[0, 1, 2]], self.truth[[0, 1, 2]], np.full(3, 0.25))
        sensitivity = observation_sensitivity(posterior, self.registry, [obs])
        inflation = uncertainty_inflation(posterior, sensitive_mask=sensitivity > 0.0)
        self.assertEqual(inflation.near_count, 3)
        self.assertEqual(inflation.far_count, 3)
        self.assertGreater(inflation.ratio, 3.0)

    def test_identifiability_flags_sparse_observations_and_accepts_dense_coverage(self):
        posterior = self._posterior()
        sparse_obs = [Observation("borehole", self.grid.coordinates[[0]], self.truth[[0]], [0.25])]
        sparse = identifiability_diagnostic(
            posterior, self.registry, sparse_obs, sensitivity_threshold=1.0e-12, min_sensitive_fraction=0.5
        )
        self.assertTrue(sparse.insufficient_observations)
        self.assertLess(sparse.sensitive_fraction, 0.5)

        dense_obs = [
            Observation("borehole", self.grid.coordinates[:4], self.truth[:4], np.full(4, 0.25)),
        ]
        dense = identifiability_diagnostic(
            posterior, self.registry, dense_obs, sensitivity_threshold=1.0e-12, min_sensitive_fraction=0.5
        )
        self.assertFalse(dense.insufficient_observations)
        self.assertGreaterEqual(dense.sensitive_fraction, 0.5)

    def test_diagnostics_on_real_linear_inversion(self):
        observed = np.array([0, 1, 2])
        obs = Observation("borehole", self.grid.coordinates[observed], self.truth[observed], np.full(3, 0.25))
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.2, marginal_precision=0.01, length_scale=2.0)
        posterior = linear_gaussian_invert(self.grid, [obs], self.registry, prior)

        coverage = truth_coverage(posterior, self.truth, alpha=0.1)
        diagnostic = identifiability_diagnostic(
            posterior, self.registry, [obs], sensitivity_threshold=1.0e-12, min_sensitive_fraction=0.75
        )
        self.assertTrue(np.isfinite(coverage.coverage))
        self.assertTrue(diagnostic.insufficient_observations)
        self.assertIn("below required", diagnostic.reason)


if __name__ == "__main__":
    unittest.main()
