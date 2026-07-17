"""No-U-Turn Sampler for latent 3D fields (MP-I5).

:mod:`mixle_pde.field_mcmc` already shipped Random-Walk Metropolis, pCN, MALA, and HMC;
:mod:`mixle_pde.verification.mcmc_diagnostics`'s own module docstring recorded the resulting gap
explicitly: "This repo's registered samplers ... are not NUTS." :func:`~mixle_pde.field_mcmc.nuts_field_invert`
closes it by adding the two things that distinguish NUTS from :func:`~mixle_pde.field_mcmc.hmc_field_invert`'s
fixed-step-count trajectory (Hoffman & Gelman, 2014): the no-U-turn binary-tree termination criterion and
dual-averaging step-size adaptation. Both reuse the same leapfrog integrator and prior-precision mass
matrix HMC already uses -- this file does not re-derive that machinery, only exercises the new sampler.

Coverage:

* :class:`NUTSAnalyticalPosteriorTest` -- the headline proof: on a linear-Gaussian problem with a
  closed-form posterior mean/covariance, NUTS's empirical mean and covariance match the analytical
  answer within Monte Carlo error, using the chain's *own* reported effective sample size
  (:func:`mixle_pde.verification.mcmc_diagnostics.multichain_ess`) to size the tolerance rather than an
  arbitrary fixed number.
* :class:`NUTSVersusHMCEfficiencyTest` -- NUTS's effective-sample-size-per-gradient-evaluation should be
  at least comparable to (this problem: measurably better than) HMC's, on the identical problem with a
  matched sample budget -- adaptive trajectory length is NUTS's whole reason to exist.
* :class:`NUTSDualAveragingTest` -- step-size adaptation actually adapts (converges to the same place
  from a deliberately too-large and too-small starting point, and lands near ``target_accept``).
* :class:`NUTSTreeBuildingTest` -- the no-U-turn criterion is doing real work (typical termination well
  short of ``max_tree_depth``, not truncated by the safety cap; no spurious divergences on a
  well-behaved posterior).
* :class:`NUTSValidationTest` -- argument validation and the same "needs a Jacobian" requirement
  MALA/HMC already enforce.
"""

from __future__ import annotations

import unittest

import numpy as np

import mixle_pde.field_mcmc as field_mcmc
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_mcmc import NUTSReport, hmc_field_invert, nuts_field_invert
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.observations import ForwardOperator, ForwardOperatorRegistry, Observation
from mixle_pde.verification.mcmc_diagnostics import multichain_ess


def _linear_gaussian_problem():
    """3-cell field, 2-observation linear operator, Gaussian prior: a conjugate normal-normal update
    with a closed-form posterior mean and covariance. Mirrors the fixture
    ``mixle_pde/c5_sampler_test.py::GradientFieldSamplerTest`` already uses as MALA/HMC's own
    known-answer check, so NUTS is held to the identical reference problem.
    """
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    grid = Field3D(coordinates=coords, spacing=1.0, units="", property_name="x")
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0, marginal_precision=0.1, neighbors=2)
    jac = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3]])
    noise_var = np.array([0.3, 0.3])
    true_field = np.array([1.0, -0.5, 0.8])
    y = jac @ true_field

    def predict(grid, field_values, obs_locations):
        return jac @ field_values

    def jacobian(grid, obs_locations):
        return jac

    registry = ForwardOperatorRegistry()
    registry.register(ForwardOperator("linear", predict, jacobian=jacobian))
    observation = Observation(kind="linear", location=np.zeros((2, 3)), value=y, noise_cov=noise_var)

    precision = prior.precision(grid)
    prior_mean = prior.mean_vector(grid)
    r_inv = np.diag(1.0 / noise_var)
    post_precision = precision + jac.T @ r_inv @ jac
    post_cov = np.linalg.inv(post_precision)
    post_mean = post_cov @ (precision @ prior_mean + jac.T @ r_inv @ y)
    return grid, registry, prior, [observation], post_mean, post_cov


def _count_gradient_evaluations(fn, *args, **kwargs):
    """Monkeypatch ``mixle_pde.field_mcmc.field_log_posterior_grad_kernel`` to count real gradient calls.

    ``hmc_field_invert`` and ``nuts_field_invert`` each call this name as a bare module-global inside a
    local closure, which Python resolves dynamically from the module's ``__dict__`` at call time; patching
    the module attribute therefore counts every call made from inside either function without touching
    either one's source. Used both to give HMC (which reports no such count itself) a real,
    independently-measured gradient-evaluation total for the efficiency comparison, and as a cross-check
    of :class:`NUTSReport`'s own internal ``gradient_evaluations`` tally.
    """
    original = field_mcmc.field_log_posterior_grad_kernel
    counter = {"n": 0}

    def counting_wrapper(*a, **kw):
        counter["n"] += 1
        return original(*a, **kw)

    field_mcmc.field_log_posterior_grad_kernel = counting_wrapper
    try:
        result = fn(*args, **kwargs)
    finally:
        field_mcmc.field_log_posterior_grad_kernel = original
    return result, counter["n"]


class NUTSAnalyticalPosteriorTest(unittest.TestCase):
    """Definition of Done: NUTS's empirical mean/covariance from a real chain matches the analytical
    posterior within Monte Carlo error."""

    def test_nuts_recovers_linear_gaussian_mean_and_covariance(self):
        grid, registry, prior, observations, post_mean, post_cov = _linear_gaussian_problem()

        posterior, report = nuts_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=3000,
            warmup=1000,
            thin=1,
            rng=np.random.default_rng(11),
        )
        self.assertIsInstance(posterior, PosteriorFieldSamples3D)
        self.assertIsInstance(report, NUTSReport)
        self.assertEqual(report.stored_samples, 3000)
        self.assertTrue(np.all(np.isfinite(posterior.samples)))
        self.assertEqual(report.n_divergences, 0)

        ess = multichain_ess(posterior.samples[None, :, :])
        self.assertTrue(np.all(ess > 200.0), ess)  # a real, usable number of independent-equivalent draws

        # Mean: Monte Carlo standard error of a chain mean is sigma / sqrt(ess), per parameter.
        empirical_mean = posterior.mean
        se_mean = np.sqrt(np.diag(post_cov) / ess)
        deviation = np.abs(empirical_mean - post_mean)
        self.assertTrue(np.all(deviation < 6.0 * se_mean), (deviation, se_mean))

        # Covariance: asymptotic sampling SE of an empirical covariance entry under joint normality
        # (Isserlis' theorem / multivariate delta method): Var(S_ij) ~= (Sigma_ii*Sigma_jj + Sigma_ij^2)
        # / n_eff. Uses the worst (smallest) per-parameter ess for a conservative, single shared scale.
        empirical_cov = np.cov(posterior.samples, rowvar=False)
        ess_min = float(ess.min())
        se_cov = np.sqrt((np.outer(np.diag(post_cov), np.diag(post_cov)) + post_cov**2) / ess_min)
        cov_deviation = np.abs(empirical_cov - post_cov)
        self.assertTrue(np.all(cov_deviation < 6.0 * se_cov), (cov_deviation, se_cov))


class NUTSVersusHMCEfficiencyTest(unittest.TestCase):
    """NUTS's whole value proposition over hmc_field_invert is adaptive trajectory length; its
    effective-sample-size-per-gradient-evaluation should be at least comparable, on the same problem
    with a matched sample budget."""

    def test_ess_per_gradient_evaluation_at_least_matches_hmc(self):
        grid, registry, prior, observations, _post_mean, _post_cov = _linear_gaussian_problem()

        (hmc_posterior, _hmc_report), hmc_grad_evals = _count_gradient_evaluations(
            hmc_field_invert,
            grid,
            observations,
            registry,
            prior,
            n_samples=2000,
            burn_in=1000,
            thin=1,
            step_size=0.1,
            n_leapfrog=10,
            rng=np.random.default_rng(11),
        )
        (nuts_posterior, nuts_report), nuts_grad_evals = _count_gradient_evaluations(
            nuts_field_invert,
            grid,
            observations,
            registry,
            prior,
            n_samples=2000,
            warmup=1000,
            thin=1,
            rng=np.random.default_rng(11),
        )

        # NUTSReport's own internal gradient-evaluation tally must match the independently
        # monkeypatch-counted total exactly -- both count the same real event two different ways.
        self.assertEqual(nuts_report.gradient_evaluations, nuts_grad_evals)

        hmc_ess = multichain_ess(hmc_posterior.samples[None, :, :])
        nuts_ess = multichain_ess(nuts_posterior.samples[None, :, :])
        hmc_ess_per_grad = float(hmc_ess.min()) / hmc_grad_evals
        nuts_ess_per_grad = float(nuts_ess.min()) / nuts_grad_evals

        self.assertGreaterEqual(
            nuts_ess_per_grad,
            0.8 * hmc_ess_per_grad,
            f"NUTS ESS/gradient-eval ({nuts_ess_per_grad:.5f}) should be at least comparable to "
            f"HMC's ({hmc_ess_per_grad:.5f}) on the same problem.",
        )


class NUTSDualAveragingTest(unittest.TestCase):
    """Dual-averaging step-size adaptation (Hoffman & Gelman 2014, Algorithm 6) should converge to
    roughly the same final step size regardless of a deliberately too-large or too-small starting
    point, and land the realized acceptance statistic near ``target_accept``."""

    def test_adapts_toward_target_accept_from_either_direction(self):
        grid, registry, prior, observations, _post_mean, _post_cov = _linear_gaussian_problem()

        _too_large_posterior, too_large_report = nuts_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=200,
            warmup=500,
            thin=1,
            initial_step_size=2.0,
            target_accept=0.8,
            rng=np.random.default_rng(3),
        )
        _too_small_posterior, too_small_report = nuts_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=200,
            warmup=500,
            thin=1,
            initial_step_size=0.001,
            target_accept=0.8,
            rng=np.random.default_rng(3),
        )

        self.assertAlmostEqual(too_large_report.step_size, too_small_report.step_size, delta=0.05)
        self.assertAlmostEqual(too_large_report.mean_accept_stat, 0.8, delta=0.15)
        self.assertAlmostEqual(too_small_report.mean_accept_stat, 0.8, delta=0.15)

    def test_warmup_zero_leaves_step_size_unadapted(self):
        grid, registry, prior, observations, _post_mean, _post_cov = _linear_gaussian_problem()
        _posterior, report = nuts_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=50,
            warmup=0,
            thin=1,
            initial_step_size=0.15,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(report.step_size, 0.15)
        self.assertEqual(report.stored_samples, 50)
        self.assertEqual(report.warmup, 0)


class NUTSTreeBuildingTest(unittest.TestCase):
    """The no-U-turn criterion should be doing real work: the trajectory tree typically self-terminates
    well short of ``max_tree_depth`` rather than being truncated by the safety cap every iteration, and
    this well-behaved posterior should not be flagged divergent."""

    def test_terminates_before_max_tree_depth_with_no_divergences(self):
        grid, registry, prior, observations, _post_mean, _post_cov = _linear_gaussian_problem()
        _posterior, report = nuts_field_invert(
            grid,
            observations,
            registry,
            prior,
            n_samples=1000,
            warmup=500,
            thin=1,
            max_tree_depth=10,
            rng=np.random.default_rng(7),
        )
        self.assertLess(report.mean_tree_depth, 6.0)
        self.assertEqual(report.max_tree_depth_hits, 0)
        self.assertEqual(report.n_divergences, 0)


class NUTSValidationTest(unittest.TestCase):
    def setUp(self):
        (
            self.grid,
            self.registry,
            self.prior,
            self.observations,
            self._post_mean,
            self._post_cov,
        ) = _linear_gaussian_problem()

    def test_requires_a_jacobian(self):
        registry = ForwardOperatorRegistry()
        registry.register(ForwardOperator("no_jacobian", predict=lambda grid, f, loc: f[:2].copy()))
        obs = Observation(
            kind="no_jacobian", location=np.zeros((2, 3)), value=np.array([0.5, 0.5]), noise_cov=np.array([0.3, 0.3])
        )
        with self.assertRaises(ValueError):
            nuts_field_invert(self.grid, [obs], registry, self.prior, n_samples=10, warmup=10, thin=1)

    def test_rejects_non_positive_n_samples(self):
        with self.assertRaises(ValueError):
            nuts_field_invert(self.grid, self.observations, self.registry, self.prior, n_samples=0, warmup=10, thin=1)

    def test_rejects_negative_warmup(self):
        with self.assertRaises(ValueError):
            nuts_field_invert(self.grid, self.observations, self.registry, self.prior, n_samples=10, warmup=-1, thin=1)

    def test_rejects_non_positive_thin(self):
        with self.assertRaises(ValueError):
            nuts_field_invert(self.grid, self.observations, self.registry, self.prior, n_samples=10, warmup=10, thin=0)

    def test_rejects_target_accept_out_of_range(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                nuts_field_invert(
                    self.grid,
                    self.observations,
                    self.registry,
                    self.prior,
                    n_samples=10,
                    warmup=10,
                    thin=1,
                    target_accept=bad,
                )

    def test_rejects_non_positive_max_tree_depth(self):
        with self.assertRaises(ValueError):
            nuts_field_invert(
                self.grid,
                self.observations,
                self.registry,
                self.prior,
                n_samples=10,
                warmup=10,
                thin=1,
                max_tree_depth=0,
            )

    def test_rejects_non_positive_initial_step_size(self):
        with self.assertRaises(ValueError):
            nuts_field_invert(
                self.grid,
                self.observations,
                self.registry,
                self.prior,
                n_samples=10,
                warmup=10,
                thin=1,
                initial_step_size=-1.0,
            )

    def test_rejects_wrong_shape_initial_unconstrained(self):
        with self.assertRaises(ValueError):
            nuts_field_invert(
                self.grid,
                self.observations,
                self.registry,
                self.prior,
                n_samples=10,
                warmup=10,
                thin=1,
                initial_unconstrained=np.zeros(2),
            )

    def test_rejects_empty_observations(self):
        with self.assertRaises(ValueError):
            nuts_field_invert(self.grid, [], self.registry, self.prior, n_samples=10, warmup=10, thin=1)


if __name__ == "__main__":
    unittest.main()
