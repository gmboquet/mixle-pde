"""MAP/Laplace posterior fit (workstream G, first inference-ladder rung): mixle_pde.posterior_fit."""

import unittest

import numpy as np

from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    gravity_forward_operator,
)
from mixle_pde.posterior_fit import fit_map_posterior, posterior_predictive_log_likelihood


def _grid(n_per_axis=3, z=-50.0, spacing=10.0):
    xs = np.arange(n_per_axis, dtype=float) * spacing
    coords = np.array([[x, y, z] for x in xs for y in xs])
    return coords


class ClosedFormPosteriorRecoversTruthTest(unittest.TestCase):
    def setUp(self):
        self.cells = _grid()
        self.volumes = np.full(len(self.cells), 1000.0)
        self.grid = Field3D(coordinates=self.cells, spacing=10.0, units="kg/m^3", property_name="density_contrast")
        self.true_field = np.zeros(len(self.cells))
        self.true_field[4] = 500.0

        self.registry = ForwardOperatorRegistry()
        self.gravity_op = gravity_forward_operator(self.cells, self.volumes)
        self.registry.register(self.gravity_op)
        self.registry.register(borehole_forward_operator())

    def _dense_gravity_observations(self, noise_std=0.02, seed=0):
        rng = np.random.RandomState(seed)
        obs_locations = np.array([[x, y, 0.0] for x in (0.0, 10.0, 20.0) for y in (0.0, 10.0, 20.0)])
        signal = self.gravity_op.predict(self.grid, self.true_field, obs_locations)
        noisy = signal + rng.normal(0, noise_std, size=len(signal))
        return Observation(
            kind="gravity", location=obs_locations, value=noisy, noise_cov=np.full(len(noisy), noise_std**2)
        )

    def test_posterior_mean_is_close_to_the_true_field_with_a_direct_hit(self):
        # gravity alone is famously non-unique (many density distributions fit the same surface
        # anomaly almost equally well, especially this collinear a station/cell layout); a direct
        # borehole hit at the anomalous cell plus a REALISTIC (not uninformative) prior -- density
        # contrasts of this magnitude are not expected everywhere -- together resolve it.
        gravity_obs = self._dense_gravity_observations()
        borehole_obs = Observation(kind="borehole", location=self.cells[[4]], value=[500.0], noise_cov=[5.0**2])
        posterior = fit_map_posterior(self.grid, [gravity_obs, borehole_obs], self.registry, prior_cov=200.0**2)
        np.testing.assert_allclose(posterior.mean, self.true_field, atol=60.0)

    def test_posterior_variance_shrinks_relative_to_the_prior(self):
        obs = self._dense_gravity_observations()
        prior_cov = 1000.0**2
        posterior = fit_map_posterior(self.grid, [obs], self.registry, prior_cov=prior_cov)
        # every cell's posterior variance must be no larger than the prior's -- evidence only
        # sharpens a Gaussian conjugate update, never widens it.
        self.assertTrue(np.all(posterior.marginal_variance <= prior_cov + 1e-6))
        # adding a direct, low-noise borehole hit at cell 4 shrinks that cell's variance sharply --
        # gravity's own (nearly degenerate, non-unique) sensitivity barely constrains it alone.
        borehole_obs = Observation(kind="borehole", location=self.cells[[4]], value=[500.0], noise_cov=[5.0**2])
        posterior_with_borehole = fit_map_posterior(self.grid, [obs, borehole_obs], self.registry, prior_cov=prior_cov)
        self.assertLess(posterior_with_borehole.marginal_variance[4], prior_cov * 0.01)
        self.assertLess(posterior_with_borehole.marginal_variance[4], posterior.marginal_variance[4])

    def test_credible_interval_covers_the_truth_on_repeated_noise_draws(self):
        covered = 0
        n_trials = 40
        for seed in range(n_trials):
            obs = self._dense_gravity_observations(noise_std=0.02, seed=seed)
            posterior = fit_map_posterior(self.grid, [obs], self.registry, prior_cov=1000.0**2)
            lo, hi = posterior.credible_interval(alpha=0.1)
            covered += int(lo[4] <= self.true_field[4] <= hi[4])
        # a proper 90% credible interval should cover comfortably more than half the trials --
        # a loose sanity bound (not a tight calibration claim, n_trials=40 is small).
        self.assertGreater(covered / n_trials, 0.7)

    def test_mixed_kind_observations_improve_over_gravity_alone(self):
        gravity_obs = self._dense_gravity_observations()
        borehole_obs = Observation(kind="borehole", location=self.cells[[4]], value=[500.0], noise_cov=[5.0**2])

        posterior_gravity_only = fit_map_posterior(self.grid, [gravity_obs], self.registry, prior_cov=1000.0**2)
        posterior_mixed = fit_map_posterior(self.grid, [gravity_obs, borehole_obs], self.registry, prior_cov=1000.0**2)
        # the direct, low-noise borehole hit at the anomalous cell sharpens that cell's posterior
        # substantially beyond what gravity alone (an indirect, blurred sensitivity) achieves.
        self.assertLess(posterior_mixed.marginal_variance[4], posterior_gravity_only.marginal_variance[4] * 0.5)
        self.assertLess(abs(posterior_mixed.mean[4] - 500.0), abs(posterior_gravity_only.mean[4] - 500.0) + 1e-6)

    def test_posterior_predictive_likelihood_prefers_the_fitted_posterior_over_the_bare_prior(self):
        obs = self._dense_gravity_observations()
        posterior = fit_map_posterior(self.grid, [obs], self.registry, prior_cov=1000.0**2)
        ll_fitted = posterior_predictive_log_likelihood(self.grid, posterior, [obs], self.registry)

        prior_field = np.zeros(len(self.cells))
        ll_prior = self.registry.total_log_likelihood(self.grid, prior_field, [obs])
        self.assertGreater(ll_fitted, ll_prior)

    def test_nonlinear_operator_without_jacobian_raises(self):
        from mixle_pde.observations import ForwardOperator

        registry = ForwardOperatorRegistry()
        registry.register(ForwardOperator("mystery", predict=lambda grid, fv, loc: np.zeros(len(loc))))
        obs = Observation(kind="mystery", location=[[0.0, 0.0, 0.0]], value=[1.0], noise_cov=[0.1])
        with self.assertRaises(ValueError):
            fit_map_posterior(self.grid, [obs], registry)

    def test_bounded_property_raises(self):
        bounded_grid = Field3D(
            coordinates=self.cells, spacing=10.0, units="frac", property_name="porosity", bounds=(0.0, 1.0)
        )
        obs = Observation(kind="borehole", location=self.cells[[0]], value=[0.2], noise_cov=[0.01])
        with self.assertRaises(ValueError):
            fit_map_posterior(bounded_grid, [obs], self.registry)


if __name__ == "__main__":
    unittest.main()
