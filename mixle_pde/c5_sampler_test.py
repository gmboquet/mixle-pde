"""C5 -- preconditioned / gradient field samplers, EnKF localization, particle rejuvenation.

The headline regression this module guards is the one :func:`metropolis_field_invert` cannot pass: a
posterior with two well-separated likelihood modes. An isotropic random-walk proposal essentially never
crosses the low-likelihood gap between the modes in a practical iteration budget, so it collapses onto
whichever side it starts near. :func:`~mixle_pde.field_mcmc.pcn_field_invert`'s proposal injects a fresh
prior-scaled draw every step (not just a small local perturbation), which is what lets it occasionally
land near the other mode and mix between both -- this is the multimodal fallback DR-ALG C5 exists for.

The remaining tests cover the smaller "+ ensemble fixes" half of this card: the gradient used by
``mala_field_invert``/``hmc_field_invert`` against a linear-Gaussian problem with a known closed-form
posterior, the Gaspari-Cohn localization taper, ``assimilate_4d_ensemble``'s new (default-off)
``localization_radius`` kwarg, and the post-resample rejuvenation added to ``particle_assimilate_4d``
and ``sample_update._update_samples``.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.cluster.vq import kmeans2

from mixle_pde.field_assimilation import (
    assimilate_4d_ensemble,
    gaspari_cohn_localization,
    particle_assimilate_4d,
)
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import (
    field_log_posterior_grad_kernel,
    hmc_field_invert,
    mala_field_invert,
    metropolis_field_invert,
    pcn_field_invert,
)
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperator, ForwardOperatorRegistry, Observation
from mixle_pde.sample_update import _update_samples, update_sampled_field_posterior


def _bimodal_problem(mode: float = 4.0, marginal_precision: float = 1.0 / 64.0, noise_var: float = 0.25):
    """A single-cell field whose likelihood is a double well: ``predict = field**2``.

    With ``observation.value == mode**2`` and a tight ``noise_var``, both ``+mode`` and ``-mode`` fit
    the data almost perfectly while the region between them (near ``field == 0``) is a deep likelihood
    trough -- a minimal, exactly-controlled bimodal posterior with a wide, symmetric prior so neither
    mode is favoured a priori.
    """
    grid = Field3D(coordinates=np.zeros((1, 3)), spacing=1.0, units="", property_name="x")
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=marginal_precision, neighbors=1)

    def predict(grid, field_values, obs_locations):
        return field_values**2

    def jacobian_at_values(grid, field_values, obs_locations):
        return np.array([[2.0 * field_values[0]]])

    op = ForwardOperator("quadratic", predict, jacobian_at_values=jacobian_at_values, differentiable=True)
    registry = ForwardOperatorRegistry()
    registry.register(op)
    observation = Observation(
        kind="quadratic",
        location=np.zeros((1, 3)),
        value=np.array([mode**2]),
        noise_cov=np.array([noise_var]),
    )
    return grid, registry, prior, [observation]


def _mode_occupancy(samples: np.ndarray, mode: float) -> tuple[np.ndarray, np.ndarray]:
    """K-means (k=2), seeded at the two known mode locations, into (sorted centers, occupancy)."""
    flat = np.asarray(samples, dtype=float).reshape(-1, 1)
    init = np.array([[-mode], [mode]])
    centers, labels = kmeans2(flat, k=init, minit="matrix")
    counts = np.bincount(labels, minlength=2)
    occupancy = counts / counts.sum()
    return centers.ravel(), occupancy


class BimodalPosteriorSamplerTest(unittest.TestCase):
    """Definition of Done: pcn_field_invert finds both modes; metropolis_field_invert collapses to one."""

    def test_pcn_finds_both_modes_while_metropolis_collapses(self):
        mode = 4.0
        grid, registry, prior, observations = _bimodal_problem(mode=mode)

        pcn_posterior, pcn_report = pcn_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=8000,
            burn_in=2000,
            thin=1,
            beta_pcn=0.5,
            rng=np.random.default_rng(0),
        )
        pcn_centers, pcn_occupancy = _mode_occupancy(pcn_posterior.samples[:, 0], mode)

        rwm_posterior, rwm_report = metropolis_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=4000,
            burn_in=2000,
            thin=1,
            step_scale=1.0,
            rng=np.random.default_rng(0),
        )
        rwm_centers, rwm_occupancy = _mode_occupancy(rwm_posterior.samples[:, 0], mode)

        # pCN: both modes recovered at +-mode, each occupying at least 20% of the stored draws.
        np.testing.assert_allclose(sorted(pcn_centers), [-mode, mode], atol=0.5)
        self.assertGreaterEqual(pcn_occupancy.min(), 0.20)
        self.assertEqual(pcn_report.stored_samples, 8000)

        # Baseline random-walk Metropolis: collapses onto one side, the other mode under 5% occupied.
        self.assertLess(rwm_occupancy.min(), 0.05)
        self.assertEqual(rwm_report.stored_samples, 4000)


class GradientFieldSamplerTest(unittest.TestCase):
    """MALA/HMC recover a known linear-Gaussian posterior using field_log_posterior_grad_kernel."""

    def setUp(self):
        self.coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.grid = Field3D(coordinates=self.coords, spacing=1.0, units="", property_name="x")
        self.prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0, marginal_precision=0.1, neighbors=2)
        self.J = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3]])
        self.noise_var = np.array([0.3, 0.3])
        true_field = np.array([1.0, -0.5, 0.8])
        self.y = self.J @ true_field

        def predict(grid, field_values, obs_locations):
            return self.J @ field_values

        def jacobian(grid, obs_locations):
            return self.J

        self.registry = ForwardOperatorRegistry()
        self.registry.register(ForwardOperator("linear", predict, jacobian=jacobian))
        self.observation = Observation(kind="linear", location=np.zeros((2, 3)), value=self.y, noise_cov=self.noise_var)

        precision = self.prior.precision(self.grid)
        prior_mean = self.prior.mean_vector(self.grid)
        Rinv = np.diag(1.0 / self.noise_var)
        post_precision = precision + self.J.T @ Rinv @ self.J
        self.post_cov = np.linalg.inv(post_precision)
        self.post_mean = self.post_cov @ (precision @ prior_mean + self.J.T @ Rinv @ self.y)
        self.post_std = np.sqrt(np.diag(self.post_cov))

    def test_grad_kernel_matches_finite_difference(self):
        u = np.array([0.3, -0.2, 0.6])
        grad = field_log_posterior_grad_kernel(self.grid, [self.observation], self.registry, self.prior, u)
        eps = 1.0e-6
        numeric = np.empty(3)
        from mixle_pde.field_mcmc import field_log_posterior_kernel

        for i in range(3):
            up = u.copy()
            up[i] += eps
            um = u.copy()
            um[i] -= eps
            numeric[i] = (
                field_log_posterior_kernel(self.grid, [self.observation], self.registry, self.prior, up)
                - field_log_posterior_kernel(self.grid, [self.observation], self.registry, self.prior, um)
            ) / (2.0 * eps)
        np.testing.assert_allclose(grad, numeric, atol=1.0e-3)

    def test_mala_recovers_linear_gaussian_posterior(self):
        posterior, report = mala_field_invert(
            self.grid,
            [self.observation],
            self.registry,
            self.prior,
            n_samples=4000,
            burn_in=1500,
            thin=1,
            step_size=0.1,
            rng=np.random.default_rng(11),
        )
        self.assertTrue(np.all(np.isfinite(posterior.samples)))
        self.assertGreater(report.acceptance_rate, 0.1)
        deviation = np.abs(posterior.samples.mean(0) - self.post_mean)
        self.assertTrue(np.all(deviation <= 3.0 * self.post_std), deviation)

    def test_hmc_recovers_linear_gaussian_posterior(self):
        posterior, report = hmc_field_invert(
            self.grid,
            [self.observation],
            self.registry,
            self.prior,
            n_samples=2000,
            burn_in=1000,
            thin=1,
            step_size=0.1,
            n_leapfrog=10,
            rng=np.random.default_rng(11),
        )
        self.assertTrue(np.all(np.isfinite(posterior.samples)))
        self.assertGreater(report.acceptance_rate, 0.1)
        deviation = np.abs(posterior.samples.mean(0) - self.post_mean)
        self.assertTrue(np.all(deviation <= 3.0 * self.post_std), deviation)

    def test_gradient_sampler_requires_a_jacobian(self):
        registry = ForwardOperatorRegistry()
        registry.register(ForwardOperator("no_jacobian", predict=lambda grid, f, loc: self.J @ f))
        obs = Observation(kind="no_jacobian", location=np.zeros((2, 3)), value=self.y, noise_cov=self.noise_var)
        with self.assertRaises(ValueError):
            mala_field_invert(self.grid, [obs], registry, self.prior, n_samples=10, burn_in=0, thin=1, step_size=0.1)


class GaspariCohnLocalizationTest(unittest.TestCase):
    def test_taper_is_one_at_zero_distance_and_decays_to_zero_at_the_radius(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        taper = gaspari_cohn_localization(coords, radius=2.0).toarray()
        np.testing.assert_allclose(np.diag(taper), 1.0)
        np.testing.assert_allclose(taper, taper.T)
        self.assertTrue(np.all((taper >= 0.0) & (taper <= 1.0)))
        self.assertGreater(taper[0, 1], 0.0)  # distance 1.0 < radius 2.0: nonzero taper
        self.assertEqual(taper[0, 2], 0.0)  # distance 5.0 > radius 2.0: fully localized out

    def test_rejects_non_positive_radius(self):
        with self.assertRaises(ValueError):
            gaspari_cohn_localization(np.zeros((2, 3)), radius=0.0)


class EnsembleLocalizationTest(unittest.TestCase):
    def setUp(self):
        self.grid = Field3D(
            coordinates=np.array([[0.0, 0.0, -10.0]]),
            spacing=1.0,
            units="state",
            property_name="nonlinear_state",
        )
        self.times = np.array([0.0, 1.0])
        self.registry = ForwardOperatorRegistry()

        def predict(grid, field_values, obs_locations):
            return np.full(obs_locations.shape[0], float(field_values[0] ** 2))

        self.registry.register(ForwardOperator("square_sensor", predict=predict, differentiable=False))
        self.observations = [
            [
                Observation(
                    "square_sensor",
                    np.array([[0.0, 0.0, -10.0]]),
                    np.array([value**2]),
                    np.array([0.03**2]),
                    time=time,
                )
            ]
            for time, value in zip(self.times, [1.0, 1.35], strict=True)
        ]
        self.prior = FieldGaussianPrior(mean=0.8, smoothness_precision=0.0, marginal_precision=4.0, length_scale=1.0)

    def test_localization_radius_defaults_to_none_and_is_unchanged(self):
        explicit = assimilate_4d_ensemble(
            self.grid,
            self.times,
            self.observations,
            self.registry,
            self.prior,
            process_var=0.08,
            ensemble_size=64,
            rng=np.random.default_rng(5),
            localization_radius=None,
        )
        omitted = assimilate_4d_ensemble(
            self.grid,
            self.times,
            self.observations,
            self.registry,
            self.prior,
            process_var=0.08,
            ensemble_size=64,
            rng=np.random.default_rng(5),
        )
        for a, b in zip(explicit.means, omitted.means, strict=True):
            np.testing.assert_allclose(a, b)
        for a, b in zip(explicit.covs, omitted.covs, strict=True):
            np.testing.assert_allclose(a, b)

    def test_localization_radius_much_larger_than_the_grid_barely_perturbs_the_result(self):
        baseline = assimilate_4d_ensemble(
            self.grid,
            self.times,
            self.observations,
            self.registry,
            self.prior,
            process_var=0.08,
            ensemble_size=64,
            rng=np.random.default_rng(5),
        )
        localized = assimilate_4d_ensemble(
            self.grid,
            self.times,
            self.observations,
            self.registry,
            self.prior,
            process_var=0.08,
            ensemble_size=64,
            rng=np.random.default_rng(5),
            localization_radius=1.0e9,
        )
        for a, b in zip(baseline.means, localized.means, strict=True):
            np.testing.assert_allclose(a, b, rtol=1.0e-6, atol=1.0e-9)

    def test_localization_rejects_non_positive_radius(self):
        with self.assertRaises(ValueError):
            assimilate_4d_ensemble(
                self.grid,
                self.times,
                self.observations,
                self.registry,
                self.prior,
                process_var=0.08,
                localization_radius=0.0,
            )


class ParticleRejuvenationTest(unittest.TestCase):
    def test_rejuvenation_restores_diversity_after_heavy_resampling(self):
        grid = Field3D(coordinates=np.zeros((1, 3)), spacing=1.0, units="", property_name="x")
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)
        registry = ForwardOperatorRegistry()

        def predict(grid, field_values, obs_locations):
            return field_values.copy()

        def jacobian(grid, obs_locations):
            return np.eye(1)

        registry.register(ForwardOperator("direct", predict, jacobian=jacobian))
        times = np.array([0.0, 1.0, 2.0])
        observations_by_time = [
            [Observation("direct", np.zeros((1, 3)), np.array([2.0]), np.array([0.01]), time=t)] for t in times
        ]

        baseline, _ = particle_assimilate_4d(
            grid,
            times,
            observations_by_time,
            registry,
            prior,
            process_var=0.05,
            n_particles=200,
            resample_threshold=1.0,
            rejuvenate_steps=0,
            rng=np.random.default_rng(1),
        )
        rejuvenated, _ = particle_assimilate_4d(
            grid,
            times,
            observations_by_time,
            registry,
            prior,
            process_var=0.05,
            n_particles=200,
            resample_threshold=1.0,
            rejuvenate_steps=5,
            rng=np.random.default_rng(1),
        )

        baseline_unique = np.unique(baseline.samples[:, -1, 0]).size
        rejuvenated_unique = np.unique(rejuvenated.samples[:, -1, 0]).size
        self.assertGreater(rejuvenated_unique, baseline_unique)

    def test_rejects_negative_rejuvenate_steps(self):
        grid = Field3D(coordinates=np.zeros((1, 3)), spacing=1.0, units="", property_name="x")
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)
        registry = ForwardOperatorRegistry()
        registry.register(
            ForwardOperator("direct", lambda grid, f, loc: f.copy(), jacobian=lambda grid, loc: np.eye(1))
        )
        times = np.array([0.0])
        observations_by_time = [[Observation("direct", np.zeros((1, 3)), np.array([2.0]), np.array([0.01]))]]
        with self.assertRaises(ValueError):
            particle_assimilate_4d(
                grid, times, observations_by_time, registry, prior, process_var=0.05, rejuvenate_steps=-1
            )


class SampleUpdateRejuvenationTest(unittest.TestCase):
    def test_rejuvenation_restores_diversity_in_update_sampled_field_posterior(self):
        grid = Field3D(coordinates=np.zeros((1, 3)), spacing=1.0, units="", property_name="x")
        rng = np.random.default_rng(2)
        posterior = PosteriorFieldSamples3D(grid=grid, samples=rng.uniform(-5.0, 5.0, size=(60, 1)))

        def sharp_likelihood(field_values: np.ndarray) -> float:
            return float(-0.5 * ((field_values[0] - 3.0) / 0.05) ** 2)

        baseline, _ = update_sampled_field_posterior(
            posterior, [sharp_likelihood], n_samples=300, rng=np.random.default_rng(9), rejuvenate_steps=0
        )
        rejuvenated, _ = update_sampled_field_posterior(
            posterior, [sharp_likelihood], n_samples=300, rng=np.random.default_rng(9), rejuvenate_steps=8
        )

        baseline_unique = np.unique(baseline.samples[:, 0]).size
        rejuvenated_unique = np.unique(rejuvenated.samples[:, 0]).size
        self.assertGreater(rejuvenated_unique, baseline_unique)

    def test_update_samples_rejects_negative_rejuvenate_steps(self):
        samples = np.array([[0.0], [1.0], [2.0]])
        log_likelihood = np.array([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            _update_samples(
                samples,
                log_likelihood,
                old_log_posterior=None,
                n_samples=None,
                rng=np.random.default_rng(0),
                resample=True,
                likelihood_count=1,
                rejuvenate_steps=-1,
            )


if __name__ == "__main__":
    unittest.main()
