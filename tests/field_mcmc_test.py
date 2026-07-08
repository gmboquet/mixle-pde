"""Small-reference MCMC field inversion.

This covers the Workstream G validation rung: sampled posterior artifacts for nonlinear/non-Gaussian
checks, not production-scale inference.
"""

import unittest

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import field_log_posterior_kernel, metropolis_field_invert
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator


class MetropolisFieldInversionTest(unittest.TestCase):
    def test_one_cell_gaussian_matches_closed_form_reference(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, 0.0]]),
            spacing=1.0,
            units="kg/m^3",
            property_name="density",
        )
        registry = ForwardOperatorRegistry()
        registry.register(borehole_forward_operator())
        observation = Observation(
            "borehole",
            grid.coordinates,
            value=np.array([2.0]),
            noise_cov=np.array([0.25]),
        )
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)

        posterior, report = metropolis_field_invert(
            grid,
            [observation],
            registry,
            prior,
            n_samples=3000,
            burn_in=500,
            thin=1,
            step_scale=0.7,
            rng=np.random.default_rng(3),
        )

        expected_var = 1.0 / (1.0 + 1.0 / 0.25)
        expected_mean = expected_var * (2.0 / 0.25)
        self.assertIsInstance(posterior, PosteriorFieldSamples3D)
        self.assertEqual(report.stored_samples, 3000)
        self.assertGreater(report.acceptance_rate, 0.2)
        self.assertLess(report.acceptance_rate, 0.9)
        self.assertAlmostEqual(float(posterior.mean[0]), expected_mean, delta=0.08)
        self.assertAlmostEqual(float(posterior.marginal_variance[0]), expected_var, delta=0.05)
        lo, hi = posterior.credible_interval(alpha=0.1)
        self.assertLess(lo[0], expected_mean)
        self.assertGreater(hi[0], expected_mean)

    def test_bounded_field_samples_stay_physical(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            spacing=1.0,
            units="fraction",
            property_name="porosity",
            bounds=(0.0, 1.0),
        )
        registry = ForwardOperatorRegistry()
        registry.register(borehole_forward_operator())
        truth = np.array([0.25, 0.75])
        observation = Observation("borehole", grid.coordinates, truth, np.full(2, 0.03**2))
        prior = FieldGaussianPrior(
            mean=grid.to_unconstrained(np.full(grid.n, 0.5)),
            smoothness_precision=0.02,
            marginal_precision=0.1,
            length_scale=1.0,
        )

        posterior, report = metropolis_field_invert(
            grid,
            [observation],
            registry,
            prior,
            n_samples=1200,
            burn_in=300,
            thin=2,
            step_scale=np.full(grid.n, 0.25),
            rng=np.random.default_rng(9),
        )

        physical = posterior.physical_samples
        self.assertEqual(physical.shape, (1200, grid.n))
        self.assertTrue(np.all(physical > 0.0))
        self.assertTrue(np.all(physical < 1.0))
        np.testing.assert_allclose(grid.from_unconstrained(posterior.map), truth, atol=0.08)
        self.assertEqual(report.proposed, 2700)

    def test_log_posterior_rejects_bad_shape(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, 0.0]]),
            spacing=1.0,
            units="kg/m^3",
            property_name="density",
        )
        registry = ForwardOperatorRegistry()
        registry.register(borehole_forward_operator())
        observation = Observation("borehole", grid.coordinates, np.array([1.0]), np.array([1.0]))
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)

        with self.assertRaises(ValueError):
            field_log_posterior_kernel(grid, [observation], registry, prior, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
