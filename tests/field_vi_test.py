"""Mean-field variational inference for latent 3D fields (MP-I5).

Verifies :func:`mean_field_vi_invert` against known closed-form Gaussian posteriors: an uncorrelated
single-cell case (basic correctness -- necessary, not sufficient) and a correlated multi-cell case (the
honest test) where mean-field VI's variance-underestimation bias is real, quantified, and shown
alongside a contrasting case where it is negligible -- rather than only reporting an easy example that
would hide it.
"""

import unittest

import numpy as np
from scipy.stats import multivariate_normal

from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.field_vi import VIReport, elbo_estimate, mean_field_vi_invert
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperator, ForwardOperatorRegistry, Observation, borehole_forward_operator


class MeanFieldVIInputValidationTest(unittest.TestCase):
    def setUp(self):
        self.grid = Field3D(coordinates=np.zeros((1, 3)), spacing=1.0, units="", property_name="x")
        self.prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)
        self.registry = ForwardOperatorRegistry()
        self.registry.register(borehole_forward_operator())
        self.observation = Observation("borehole", self.grid.coordinates, np.array([1.0]), np.array([1.0]))

    def test_rejects_empty_observations(self):
        with self.assertRaises(ValueError):
            mean_field_vi_invert(self.grid, [], self.registry, self.prior, n_iterations=10)

    def test_rejects_non_positive_n_iterations(self):
        with self.assertRaises(ValueError):
            mean_field_vi_invert(self.grid, [self.observation], self.registry, self.prior, n_iterations=0)

    def test_rejects_non_positive_n_mc_samples(self):
        with self.assertRaises(ValueError):
            mean_field_vi_invert(
                self.grid, [self.observation], self.registry, self.prior, n_iterations=10, n_mc_samples=0
            )

    def test_rejects_bad_initial_unconstrained_shape(self):
        with self.assertRaises(ValueError):
            mean_field_vi_invert(
                self.grid,
                [self.observation],
                self.registry,
                self.prior,
                n_iterations=10,
                initial_unconstrained=np.zeros(2),
            )

    def test_requires_a_jacobian(self):
        """Mirrors mixle_pde/c5_sampler_test.py's identical precondition for mala/hmc_field_invert: the
        reparameterization gradient chains through field_log_posterior_grad_kernel, which needs a
        registered Jacobian exactly like the other gradient-based samplers."""
        registry = ForwardOperatorRegistry()
        registry.register(ForwardOperator("no_jacobian", predict=lambda grid, f, loc: f.copy()))
        obs = Observation(
            kind="no_jacobian", location=np.zeros((1, 3)), value=np.array([1.0]), noise_cov=np.array([1.0])
        )
        with self.assertRaises(ValueError):
            mean_field_vi_invert(self.grid, [obs], registry, self.prior, n_iterations=10)


class MeanFieldVISingleCellClosedFormTest(unittest.TestCase):
    """Basic correctness on an UNCORRELATED (single-cell) conjugate posterior -- necessary, not
    sufficient; see MeanFieldVICorrelatedBiasTest below for the case that actually stresses the
    mean-field assumption this easy case cannot."""

    def test_matches_closed_form_mean_and_variance(self):
        grid = Field3D(coordinates=np.array([[0.0, 0.0, 0.0]]), spacing=1.0, units="kg/m^3", property_name="density")
        registry = ForwardOperatorRegistry()
        registry.register(borehole_forward_operator())
        observation = Observation("borehole", grid.coordinates, value=np.array([2.0]), noise_cov=np.array([0.25]))
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)

        # Conjugate Gaussian-Gaussian update: prior N(0, 1), likelihood N(field, 0.25).
        expected_var = 1.0 / (1.0 + 1.0 / 0.25)
        expected_mean = expected_var * (2.0 / 0.25)

        posterior, report = mean_field_vi_invert(
            grid,
            [observation],
            registry,
            prior,
            n_iterations=1500,
            n_mc_samples=8,
            learning_rate=0.2,
            rng=np.random.default_rng(0),
        )

        self.assertIsInstance(posterior, PosteriorField3D)
        self.assertIsNone(posterior.dense_cov)  # mean-field (diag_var) mode, never a correlated posterior
        self.assertIsNotNone(posterior.diag_var)
        self.assertAlmostEqual(float(posterior.mean[0]), expected_mean, delta=0.05)
        self.assertAlmostEqual(float(posterior.marginal_variance[0]), expected_var, delta=0.03)

        lo, hi = posterior.credible_interval(0.9)
        self.assertLess(lo[0], expected_mean)
        self.assertGreater(hi[0], expected_mean)

        self.assertIsInstance(report, VIReport)
        self.assertEqual(report.iterations, 1500)
        self.assertEqual(len(report.elbo_history), 1500)
        self.assertIsInstance(report.converged, bool)
        self.assertLessEqual(report.final_elbo, report.best_elbo)


class MeanFieldVICorrelatedBiasTest(unittest.TestCase):
    """The honest test: a genuinely CORRELATED conjugate posterior. A chain prior with strong
    neighbour-to-neighbour smoothness coupling relative to its marginal anchor makes cells strongly
    correlated a priori; observing only the two end cells leaves the three interior cells informed
    mostly BY that correlation rather than by their own direct data -- exactly where mean-field's
    diagonal-covariance restriction has real correlation to get wrong.

    Fits VI once for the whole class (an expensive stochastic optimization) in ``setUpClass`` and checks
    several independent properties of that one fit, rather than re-fitting per assertion.
    """

    @classmethod
    def setUpClass(cls):
        n = 5
        coords = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
        cls.grid = Field3D(coordinates=coords, spacing=1.0, units="", property_name="x")
        cls.prior = FieldGaussianPrior(
            mean=0.0, smoothness_precision=20.0, marginal_precision=0.05, neighbors=2, length_scale=1.0
        )
        cls.registry = ForwardOperatorRegistry()
        cls.registry.register(borehole_forward_operator())
        obs_loc = coords[[0, n - 1]]
        cls.obs_val = np.array([1.4, -0.9])
        cls.observation = Observation("borehole", obs_loc, cls.obs_val, noise_cov=np.array([0.05**2, 0.05**2]))
        cls.observations = [cls.observation]

        # Ground truth #1: the exact closed-form posterior (a completely independent code path from
        # mean_field_vi_invert -- linear_gaussian_invert solves the normal equations directly).
        cls.exact = linear_gaussian_invert(cls.grid, cls.observations, cls.registry, cls.prior)
        cls.true_std = cls.exact.marginal_std
        # Ground truth #2: the KL-optimal mean-field covariance for THIS Gaussian target has a closed
        # form too -- marginal variances equal to the conditional variances 1 / diag(posterior precision)
        # -- letting the optimizer's result be checked against the right answer for its own restricted
        # family, decoupled from how good that restricted family is.
        posterior_precision = np.linalg.inv(cls.exact.dense_cov)
        cls.mean_field_theory_std = 1.0 / np.sqrt(np.diag(posterior_precision))

        cls.posterior, cls.report = mean_field_vi_invert(
            cls.grid,
            cls.observations,
            cls.registry,
            cls.prior,
            n_iterations=1000,
            n_mc_samples=8,
            learning_rate=0.25,
            rng=np.random.default_rng(7),
        )

    def test_mean_matches_exact_posterior_mean(self):
        # Mean-field VI is exact for the MEAN of a Gaussian target (the mean update does not depend on
        # the assumed covariance structure), so this should match tightly despite the diagonal
        # restriction -- the restriction's cost shows up only in the variance, checked below.
        np.testing.assert_allclose(self.posterior.mean, self.exact.mean, atol=0.02)

    def test_variance_matches_the_theoretical_mean_field_optimum(self):
        # Confirms the optimizer actually reaches the KL-optimal mean-field solution, not merely "some
        # biased number" -- an independent correctness check on the ELBO/gradient machinery, decoupled
        # from how close that restricted optimum is to the true posterior (checked next).
        ratio = self.posterior.marginal_std / self.mean_field_theory_std
        np.testing.assert_allclose(ratio, 1.0, atol=0.08)

    def test_variance_underestimates_the_true_posterior_for_correlated_cells(self):
        # The honest result: the three interior cells (inferred mostly through prior correlation with
        # their observed neighbours, never directly) show mean-field's known variance-underestimation
        # bias clearly -- not an edge-of-noise effect that a looser tolerance would hide.
        ratio_true = self.posterior.marginal_std / self.true_std
        interior = ratio_true[1:4]
        self.assertTrue(np.all(interior < 0.92), f"expected clear underestimation, got ratios {interior}")
        self.assertTrue(np.all(interior > 0.5), f"underestimation should be real but not extreme, got {interior}")

    def test_directly_observed_cells_show_negligible_bias_by_contrast(self):
        # The two end cells are pinned almost entirely by their own tight direct observation (noise std
        # 0.05); little posterior correlation is left for the diagonal restriction to mis-model, so
        # mean-field recovers their variance almost exactly -- the honest contrast case, demonstrating
        # the bias is a real correlation effect and not a general, blanket underestimation.
        ratio_true = self.posterior.marginal_std / self.true_std
        endpoints = ratio_true[[0, 4]]
        np.testing.assert_allclose(endpoints, 1.0, atol=0.05)

    def test_elbo_is_a_valid_lower_bound_on_the_closed_form_log_evidence(self):
        # Cross-check independent of the mean/variance comparisons above: for this linear-Gaussian model
        # the marginal log evidence log p(D) has a closed form (the observed values are jointly Gaussian
        # once the field is marginalized out). A fresh, high-sample-count ELBO estimate at the fitted
        # mean-field posterior must sit at or below it -- the defining property of a variational lower
        # bound -- with the gap a direct, honest measure of the approximation's KL cost.
        precision_prior = self.prior.precision(self.grid)
        cov_prior = np.linalg.inv(precision_prior)
        mean_prior = self.prior.mean_vector(self.grid)
        n = self.grid.n
        J = np.zeros((2, n))
        J[0, 0] = 1.0
        J[1, -1] = 1.0
        R = np.diag([0.05**2, 0.05**2])
        log_evidence = multivariate_normal.logpdf(self.obs_val, mean=J @ mean_prior, cov=J @ cov_prior @ J.T + R)

        fresh_elbo = elbo_estimate(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            self.posterior,
            n_mc_samples=5000,
            rng=np.random.default_rng(123),
        )
        # A small slack (0.1 nat) absorbs residual Monte Carlo noise in the fresh ELBO estimate itself;
        # the bound must hold up to that noise floor, not exactly bit-for-bit.
        self.assertLessEqual(fresh_elbo, log_evidence + 0.1)
        gap = log_evidence - fresh_elbo
        self.assertGreater(gap, 0.0)  # a real, nonzero KL cost from the mean-field restriction
        self.assertLess(gap, 2.0)  # ... but not a wildly loose bound either

    def test_elbo_estimate_rejects_a_non_mean_field_posterior(self):
        with self.assertRaises(ValueError):
            elbo_estimate(self.grid, self.observations, self.registry, self.prior, self.exact)


class MeanFieldVIBoundedFieldTest(unittest.TestCase):
    def test_bounded_field_stays_physical(self):
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

        posterior, report = mean_field_vi_invert(
            grid, [observation], registry, prior, n_iterations=800, learning_rate=0.15, rng=np.random.default_rng(9)
        )

        samples = posterior.sample(2000, np.random.default_rng(1))
        self.assertTrue(np.all(samples > 0.0))
        self.assertTrue(np.all(samples < 1.0))
        np.testing.assert_allclose(grid.from_unconstrained(posterior.map), truth, atol=0.08)

        lo, hi = posterior.credible_interval(0.9)
        self.assertTrue(np.all(lo > 0.0) and np.all(hi < 1.0))


if __name__ == "__main__":
    unittest.main()
