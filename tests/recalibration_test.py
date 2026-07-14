"""Definition-of-Done test for A3: `posterior_calibration.recalibrate` actually recalibrates."""

import unittest
import warnings

import numpy as np

from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator
from mixle_pde.posterior_calibration import (
    Recalibration,
    heldout_observation_check,
    recalibrate,
)


class RecalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 300
        coords = np.array([[float(i), 0.0, -10.0] for i in range(n)])
        self.grid = Field3D(coords, spacing=1.0, units="ppm", property_name="cu_ppm")
        self.registry = ForwardOperatorRegistry()
        self.registry.register(borehole_forward_operator())

        truth = rng.normal(loc=5.0, scale=2.0, size=n)
        sigma = 1.0
        # A well-fit linear-Gaussian posterior: the posterior mean's own error against truth is drawn
        # from exactly the variance the posterior claims (sigma**2), so standardized held-out residuals
        # are unit-normal by construction and the chi-square inflation is ~1.
        mean = truth + rng.standard_normal(n) * sigma
        cov = np.eye(n) * sigma**2
        self.posterior = PosteriorField3D(self.grid, mean=mean, dense_cov=cov)

        # Held-out observations are the (near-noiseless) synthetic truth -- scoring them against the
        # posterior mean/covariance measures exactly the posterior's own claimed-vs-actual error.
        self.held_out = [Observation("borehole", coords, truth, np.full(n, 1.0e-8))]

    def test_recalibrate_restores_nominal_coverage_from_an_overconfident_posterior(self):
        alpha = 0.1
        # `heldout_observation_check`'s slogdet on a large dense predictive covariance triggers a
        # spurious (still numerically correct) RuntimeWarning on some platform/numpy/LAPACK combos at
        # this observation count; irrelevant to what this test measures.
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

        # A well-fit posterior should already be close to nominal coverage.
        well_fit_fit = heldout_observation_check(self.posterior, self.registry, self.held_out, alpha=alpha)
        self.assertGreaterEqual(well_fit_fit.coverage, 0.85)

        # Inject overconfidence: shrink the claimed covariance by 9x (std by 3x).
        overconfident = PosteriorField3D(self.grid, mean=self.posterior.mean, dense_cov=self.posterior.dense_cov / 9.0)
        before_fit = heldout_observation_check(overconfident, self.registry, self.held_out, alpha=alpha)
        self.assertLess(before_fit.coverage, 0.5)

        rescaled, info = recalibrate(overconfident, self.registry, self.held_out, alpha=alpha)
        self.assertIsInstance(info, Recalibration)
        self.assertAlmostEqual(info.inflation, 3.0, delta=0.5)
        self.assertEqual(info.alpha, alpha)
        self.assertGreater(info.conformal_quantile, 0.0)

        after_fit = heldout_observation_check(rescaled, self.registry, self.held_out, alpha=alpha)
        self.assertGreaterEqual(after_fit.coverage, 0.88)

        # Recalibration widens covariance without touching the point estimate.
        np.testing.assert_allclose(rescaled.mean, overconfident.mean)
        np.testing.assert_allclose(np.diag(rescaled.cov), np.diag(overconfident.cov) * info.inflation**2)


if __name__ == "__main__":
    unittest.main()
