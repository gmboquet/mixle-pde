"""Sequential Monte Carlo with tempering for latent 3D field posteriors (MP-I5).

Acceptance evidence: the headline test in this module is the one no single-chain sampler in
``mixle_pde.field_mcmc`` can pass reliably -- a genuinely multimodal posterior (see
``mixle_pde/c5_sampler_test.py``'s ``BimodalPosteriorSamplerTest`` for the established single-cell
double-well construction this reuses) where SMC-tempering must recover BOTH modes with occupancy
fractions matching the TRUE relative mode weights (an exact 1-D numerical quadrature of the closed-form
unnormalized posterior density, not an approximation), while a vanilla Metropolis chain on the identical
problem collapses onto one side. The remaining tests cover closed-form correctness (a linear-Gaussian
posterior with a known mean/covariance, the same acceptance bar every solver in this repo is held to per
``CONTRIBUTING.md``), the fixed/adaptive/explicit temperature-schedule modes, and input validation.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy import integrate, optimize
from scipy.cluster.vq import kmeans2

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import metropolis_field_invert
from mixle_pde.field_smc import SMCReport, smc_tempering_field_invert
from mixle_pde.latent import Field3D
from mixle_pde.observations import ForwardOperator, ForwardOperatorRegistry, Observation


def _asymmetric_bimodal_problem(mode: float, noise_var: float, marginal_precision: float, prior_mean: float):
    """A single-cell double-well posterior: ``predict(u) = u**2``, so ``u = +-mode`` both fit
    ``observation.value == mode**2`` almost perfectly (a deep likelihood trough separates them), exactly
    ``mixle_pde/c5_sampler_test.py``'s ``_bimodal_problem`` construction. Unlike that fixture (prior
    mean 0, so the two modes tie at 50/50 by symmetry), ``prior_mean`` is shifted away from 0 here: the
    quadratic ``predict`` is an even function, so the likelihood alone contributes EXACTLY equal peak
    height and curvature to both modes (verified analytically in this module's docstring reasoning and
    empirically by :func:`_true_mode_weights` below) -- any weight asymmetry between the two modes is
    due entirely to the prior favouring one side, by a precisely known amount.
    """
    grid = Field3D(coordinates=np.zeros((1, 3)), spacing=1.0, units="", property_name="x")
    prior = FieldGaussianPrior(
        mean=prior_mean, smoothness_precision=0.0, marginal_precision=marginal_precision, neighbors=1
    )

    def predict(grid, field_values, obs_locations):
        return field_values**2

    def jacobian_at_values(grid, field_values, obs_locations):
        return np.array([[2.0 * field_values[0]]])

    op = ForwardOperator("quadratic", predict, jacobian_at_values=jacobian_at_values, differentiable=True)
    registry = ForwardOperatorRegistry()
    registry.register(op)
    observation = Observation(
        kind="quadratic", location=np.zeros((1, 3)), value=np.array([mode**2]), noise_cov=np.array([noise_var])
    )
    return grid, registry, prior, [observation]


def _true_mode_weights(
    mode: float, noise_var: float, prior_mean: float, marginal_precision: float
) -> tuple[float, float]:
    """Ground truth relative mode weight, by exact 1-D numerical quadrature of the closed-form
    unnormalized posterior density ``prior(u) * likelihood(u)`` (both known in closed form for this
    fixture), split at the trough between the two modes. This is not an approximation of the true
    weights -- ``scipy.integrate.quad`` over a smooth, rapidly-decaying 1-D integrand converges to
    machine precision -- so it is the honest reference SMC's (and Metropolis's) recovered occupancy is
    checked against. Returns ``(weight_at_minus_mode, weight_at_plus_mode)``, summing to 1.
    """

    def log_target(u: float) -> float:
        return -0.5 * marginal_precision * (u - prior_mean) ** 2 - 0.5 * (mode**2 - u**2) ** 2 / noise_var

    trough = optimize.minimize_scalar(log_target, bounds=(-mode + 0.05, mode - 0.05), method="bounded").x
    peak = max(log_target(mode), log_target(-mode))

    def shifted_density(u: float) -> float:
        return float(np.exp(log_target(u) - peak))

    left, _ = integrate.quad(shifted_density, -np.inf, trough, limit=200)
    right, _ = integrate.quad(shifted_density, trough, np.inf, limit=200)
    total = left + right
    return left / total, right / total


def _mode_occupancy(samples: np.ndarray, mode: float) -> tuple[np.ndarray, np.ndarray]:
    """K-means (k=2), seeded at the two known mode locations, into (sorted centers, occupancy)."""
    flat = np.asarray(samples, dtype=float).reshape(-1, 1)
    init = np.array([[-mode], [mode]])
    centers, labels = kmeans2(flat, k=init, minit="matrix")
    counts = np.bincount(labels, minlength=2)
    occupancy = counts / counts.sum()
    order = np.argsort(centers.ravel())
    return centers.ravel()[order], occupancy[order]


class SMCTemperingMultimodalRecoveryTest(unittest.TestCase):
    """Definition of Done: SMC-tempering recovers BOTH modes with occupancy matching the true relative
    weights; plain Metropolis on the identical problem collapses onto one side."""

    def setUp(self):
        self.mode = 4.0
        self.noise_var = 0.25
        self.marginal_precision = 0.25
        self.prior_mean = 0.3
        self.grid, self.registry, self.prior, self.observations = _asymmetric_bimodal_problem(
            mode=self.mode,
            noise_var=self.noise_var,
            marginal_precision=self.marginal_precision,
            prior_mean=self.prior_mean,
        )
        self.true_w_minus, self.true_w_plus = _true_mode_weights(
            self.mode, self.noise_var, self.prior_mean, self.marginal_precision
        )

    def test_fixture_true_weights_are_meaningfully_unequal(self):
        # Sanity check on the fixture itself: without this, "recovers relative weights" would not
        # actually be exercised by a coincidental 50/50 split.
        self.assertGreater(abs(self.true_w_minus - self.true_w_plus), 0.15)
        self.assertAlmostEqual(self.true_w_minus + self.true_w_plus, 1.0, places=6)

    def test_smc_tempering_recovers_both_modes_with_correct_relative_weights(self):
        posterior, report = smc_tempering_field_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            n_particles=2000,
            adaptive=True,
            resample_threshold=0.5,
            rejuvenate_steps=8,
            rejuvenate_beta=0.3,
            rng=np.random.default_rng(0),
        )
        centers, occupancy = _mode_occupancy(posterior.samples[:, 0], self.mode)
        np.testing.assert_allclose(centers, [-self.mode, self.mode], atol=0.5)
        recovered_w_minus, recovered_w_plus = occupancy

        self.assertLess(abs(recovered_w_minus - self.true_w_minus), 0.08)
        self.assertLess(abs(recovered_w_plus - self.true_w_plus), 0.08)
        # Both modes genuinely present -- not a collapse to (near) one side.
        self.assertGreater(occupancy.min(), 0.15)

        self.assertTrue(np.isfinite(report.log_evidence))
        self.assertEqual(report.effective_sample_size[0], 2000.0)
        self.assertEqual(report.n_particles, 2000)
        self.assertEqual(posterior.provenance["method"], "smc_tempering")

    def test_metropolis_collapses_to_one_mode_on_the_identical_problem(self):
        posterior, _report = metropolis_field_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            n_samples=4000,
            burn_in=2000,
            thin=1,
            step_scale=1.0,
            rng=np.random.default_rng(0),
        )
        _, occupancy = _mode_occupancy(posterior.samples[:, 0], self.mode)
        self.assertLess(occupancy.min(), 0.02)

    def test_fixed_schedule_also_recovers_both_modes(self):
        posterior, report = smc_tempering_field_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            n_particles=1500,
            adaptive=False,
            n_temperatures=40,
            resample_threshold=0.5,
            rejuvenate_steps=6,
            rejuvenate_beta=0.3,
            rng=np.random.default_rng(3),
        )
        self.assertEqual(report.n_steps, 40)
        np.testing.assert_allclose(report.temperature_schedule, np.linspace(0.0, 1.0, 41))
        _, occupancy = _mode_occupancy(posterior.samples[:, 0], self.mode)
        self.assertGreater(occupancy.min(), 0.15)


class GaussianReferenceRecoveryTest(unittest.TestCase):
    """SMC-tempering recovers a known closed-form linear-Gaussian posterior (the acceptance bar
    ``CONTRIBUTING.md`` sets for every solver: agreement with an exact reference, not just "it runs"),
    reusing ``mixle_pde/c5_sampler_test.py``'s ``GradientFieldSamplerTest`` fixture."""

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
        r_inv = np.diag(1.0 / self.noise_var)
        post_precision = precision + self.J.T @ r_inv @ self.J
        self.post_cov = np.linalg.inv(post_precision)
        self.post_mean = self.post_cov @ (precision @ prior_mean + self.J.T @ r_inv @ self.y)
        self.post_std = np.sqrt(np.diag(self.post_cov))

    def test_smc_tempering_recovers_linear_gaussian_posterior(self):
        posterior, report = smc_tempering_field_invert(
            self.grid,
            [self.observation],
            self.registry,
            self.prior,
            n_particles=3000,
            adaptive=True,
            resample_threshold=0.5,
            rejuvenate_steps=6,
            rejuvenate_beta=0.3,
            rng=np.random.default_rng(11),
        )
        self.assertTrue(np.all(np.isfinite(posterior.samples)))
        deviation = np.abs(posterior.samples.mean(0) - self.post_mean)
        self.assertTrue(np.all(deviation <= 0.15), deviation)
        std_ratio = posterior.samples.std(0) / self.post_std
        self.assertTrue(np.all((std_ratio > 0.7) & (std_ratio < 1.3)), std_ratio)
        self.assertGreaterEqual(report.resample_count, 1)


class SMCTemperingScheduleAndValidationTest(unittest.TestCase):
    def setUp(self):
        self.grid, self.registry, self.prior, self.observations = _asymmetric_bimodal_problem(
            mode=4.0, noise_var=0.25, marginal_precision=0.25, prior_mean=0.3
        )

    def test_explicit_temperature_schedule_is_honored(self):
        schedule = np.linspace(0.0, 1.0, 9) ** 2
        posterior, report = smc_tempering_field_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            n_particles=600,
            temperature_schedule=schedule,
            resample_threshold=0.5,
            rejuvenate_steps=3,
            rng=np.random.default_rng(5),
        )
        np.testing.assert_allclose(report.temperature_schedule, schedule)
        self.assertEqual(report.n_steps, 8)
        self.assertTrue(np.all(np.isfinite(posterior.samples)))

    def test_explicit_schedule_must_start_at_zero(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid,
                self.observations,
                self.registry,
                self.prior,
                n_particles=100,
                temperature_schedule=np.array([0.1, 0.5, 1.0]),
            )

    def test_explicit_schedule_must_end_at_one(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid,
                self.observations,
                self.registry,
                self.prior,
                n_particles=100,
                temperature_schedule=np.array([0.0, 0.5, 0.9]),
            )

    def test_explicit_schedule_must_be_strictly_increasing(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid,
                self.observations,
                self.registry,
                self.prior,
                n_particles=100,
                temperature_schedule=np.array([0.0, 0.5, 0.4, 1.0]),
            )

    def test_rejects_too_few_particles(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(self.grid, self.observations, self.registry, self.prior, n_particles=1)

    def test_rejects_bad_resample_threshold(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid, self.observations, self.registry, self.prior, n_particles=100, resample_threshold=0.0
            )
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid, self.observations, self.registry, self.prior, n_particles=100, resample_threshold=1.5
            )

    def test_rejects_negative_rejuvenate_steps(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid, self.observations, self.registry, self.prior, n_particles=100, rejuvenate_steps=-1
            )

    def test_rejects_bad_rejuvenate_beta(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid, self.observations, self.registry, self.prior, n_particles=100, rejuvenate_beta=0.0
            )
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid, self.observations, self.registry, self.prior, n_particles=100, rejuvenate_beta=1.5
            )

    def test_rejects_non_positive_n_temperatures(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(
                self.grid, self.observations, self.registry, self.prior, n_particles=100, n_temperatures=0
            )

    def test_requires_at_least_one_observation(self):
        with self.assertRaises(ValueError):
            smc_tempering_field_invert(self.grid, [], self.registry, self.prior, n_particles=100)

    def test_adaptive_schedule_raising_when_capped_too_low(self):
        with self.assertRaises(RuntimeError):
            smc_tempering_field_invert(
                self.grid,
                self.observations,
                self.registry,
                self.prior,
                n_particles=200,
                adaptive=True,
                n_temperatures=1,
                resample_threshold=0.9,
                rng=np.random.default_rng(0),
            )


class SMCReportValidationTest(unittest.TestCase):
    def _valid_kwargs(self):
        return dict(
            n_particles=10,
            n_steps=2,
            temperature_schedule=np.array([0.0, 0.5, 1.0]),
            resample_count=1,
            effective_sample_size=[10.0, 8.0, 9.0],
            log_evidence=-1.0,
            mean_rejuvenation_acceptance_rate=0.5,
        )

    def test_valid_report_round_trips(self):
        report = SMCReport(**self._valid_kwargs())
        self.assertEqual(report.n_steps, 2)

    def test_nan_acceptance_rate_is_allowed(self):
        kwargs = self._valid_kwargs()
        kwargs["mean_rejuvenation_acceptance_rate"] = float("nan")
        report = SMCReport(**kwargs)
        self.assertTrue(np.isnan(report.mean_rejuvenation_acceptance_rate))

    def test_schedule_must_start_at_zero_end_at_one(self):
        kwargs = self._valid_kwargs()
        kwargs["temperature_schedule"] = np.array([0.1, 0.5, 1.0])
        with self.assertRaises(ValueError):
            SMCReport(**kwargs)

    def test_schedule_length_must_match_n_steps(self):
        kwargs = self._valid_kwargs()
        kwargs["n_steps"] = 5
        with self.assertRaises(ValueError):
            SMCReport(**kwargs)

    def test_effective_sample_size_length_must_match(self):
        kwargs = self._valid_kwargs()
        kwargs["effective_sample_size"] = [10.0, 8.0]
        with self.assertRaises(ValueError):
            SMCReport(**kwargs)


if __name__ == "__main__":
    unittest.main()
