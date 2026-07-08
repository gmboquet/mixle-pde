"""Linear-Gaussian inversion of a latent 3D field (workstream G6, core acceptance).

Satisfies G's first acceptance criterion: create a 3D latent subsurface object, attach MULTIMODAL
observations (gravity + borehole), run an inversion, and extract posterior mean, variance, credible
intervals, and samples -- plus a coverage-on-synthetic-truth calibration check and a held-out
posterior-predictive check.
"""

import unittest

import numpy as np

from mixle_pde.field_inversion import (
    FieldGaussianPrior,
    linear_gaussian_invert,
    posterior_predictive_check,
    sparse_linear_gaussian_invert,
)
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    gravity_forward_operator,
)


def _subsurface_grid():
    """A small horizontal slab of subsurface cells (z up; cells below the surface z=0)."""
    xs = np.linspace(0.0, 100.0, 6)
    ys = np.linspace(0.0, 100.0, 6)
    zs = np.array([-30.0, -50.0])
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    grid = Field3D(coordinates=pts, spacing=20.0, units="kg/m^3", property_name="density_contrast", bounds=None)
    return grid


def _true_field(grid):
    """A compact positive density-contrast blob near the slab centre, zero elsewhere."""
    centre = np.array([50.0, 50.0, -40.0])
    d2 = np.sum((grid.coordinates - centre) ** 2, axis=1)
    return 500.0 * np.exp(-d2 / (2.0 * 35.0**2))


class LinearGaussianInversionTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.grid = _subsurface_grid()
        self.truth = _true_field(self.grid)
        self.volumes = np.full(self.grid.n, 20.0**3, dtype=float)

        self.registry = ForwardOperatorRegistry()
        self.registry.register(gravity_forward_operator(self.grid.coordinates, self.volumes))
        self.registry.register(borehole_forward_operator())

        # gravity stations on a surface grid above the slab
        gx, gy = np.meshgrid(np.linspace(0, 100, 5), np.linspace(0, 100, 5))
        self.grav_loc = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
        G = self.registry.get("gravity").jacobian(self.grid, self.grav_loc)
        grav_noise = 2.0e-4
        grav_clean = G @ self.truth
        grav_val = grav_clean + self.rng.normal(0, grav_noise, size=grav_clean.shape)
        self.gravity = Observation(
            kind="gravity", location=self.grav_loc, value=grav_val, noise_cov=np.full(grav_clean.shape, grav_noise**2)
        )

        # boreholes: direct samples at a subset (~40%) of the cells
        idx = self.rng.choice(self.grid.n, size=int(0.4 * self.grid.n), replace=False)
        bore_loc = self.grid.coordinates[idx]
        bore_noise = 5.0
        bore_val = self.truth[idx] + self.rng.normal(0, bore_noise, size=idx.shape)
        self.borehole = Observation(
            kind="borehole", location=bore_loc, value=bore_val, noise_cov=np.full(idx.shape, bore_noise**2)
        )

        self.prior = FieldGaussianPrior(
            mean=0.0, smoothness_precision=2.0e-3, marginal_precision=1.0e-5, length_scale=25.0, neighbors=6
        )

    def test_inversion_recovers_the_blob_and_exposes_a_full_posterior(self):
        post = linear_gaussian_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)

        self.assertIsInstance(post, PosteriorField3D)
        # posterior mean tracks the synthetic truth
        corr = np.corrcoef(post.mean, self.truth)[0, 1]
        self.assertGreater(corr, 0.9)
        # every posterior artifact the acceptance criterion names is available and well-formed
        self.assertEqual(post.mean.shape, (self.grid.n,))
        self.assertTrue(np.all(post.marginal_variance > 0))
        lo, hi = post.credible_interval(alpha=0.1)
        self.assertTrue(np.all(hi >= lo))
        samples = post.sample(64, self.rng)
        self.assertEqual(samples.shape, (64, self.grid.n))
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_credible_interval_covers_synthetic_truth_near_nominal(self):
        # calibration is a repeated-sampling property: average coverage over independent noise
        # realizations rather than trusting a single draw (which has high per-draw variance).
        coverages = []
        for seed in range(8):
            rng = np.random.default_rng(100 + seed)
            G = self.registry.get("gravity").jacobian(self.grid, self.grav_loc)
            gval = G @ self.truth + rng.normal(0, 2.0e-4, size=G.shape[0])
            gravity = Observation(
                kind="gravity", location=self.grav_loc, value=gval, noise_cov=np.full(G.shape[0], (2.0e-4) ** 2)
            )
            # dense boreholes: under adequate data the posterior is data-dominated, not prior-dominated
            # (gravity alone is ill-posed at depth, which leaves un-observed cells overconfident -- see
            # test_uncertainty_is_higher_where_there_is_no_data for that regime).
            idx = rng.choice(self.grid.n, size=int(0.75 * self.grid.n), replace=False)
            bval = self.truth[idx] + rng.normal(0, 5.0, size=idx.shape)
            bore = Observation(
                kind="borehole", location=self.grid.coordinates[idx], value=bval, noise_cov=np.full(idx.shape, 25.0)
            )
            post = linear_gaussian_invert(self.grid, [gravity, bore], self.registry, self.prior)
            lo, hi = post.credible_interval(alpha=0.1)
            coverages.append(float(np.mean((self.truth >= lo) & (self.truth <= hi))))
        # 90% nominal; mean coverage over realizations should land in a sensible calibrated band
        self.assertGreaterEqual(float(np.mean(coverages)), 0.8)

    def test_uncertainty_is_higher_where_there_is_no_data(self):
        # borehole-sampled cells should end up more certain than un-sampled ones on average
        post = linear_gaussian_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)
        diffs = self.grid.coordinates[:, None, :] - self.borehole.location[None, :, :]
        sampled = np.min(np.sum(diffs**2, axis=2), axis=1) < 1e-6
        self.assertLess(post.marginal_std[sampled].mean(), post.marginal_std[~sampled].mean())

    def test_posterior_predictive_check_on_held_out_gravity(self):
        post = linear_gaussian_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)
        # held-out gravity stations at fresh locations
        hx, hy = np.meshgrid(np.linspace(10, 90, 4), np.linspace(10, 90, 4))
        hloc = np.column_stack([hx.ravel(), hy.ravel(), np.full(hx.size, 5.0)])
        Gh = self.registry.get("gravity").jacobian(self.grid, hloc)
        noise = 2.0e-4
        val = Gh @ self.truth + self.rng.normal(0, noise, size=Gh.shape[0])
        held = Observation(kind="gravity", location=hloc, value=val, noise_cov=np.full(val.shape, noise**2))

        check = posterior_predictive_check(post, self.registry, [held], alpha=0.1)
        self.assertGreaterEqual(check.coverage, 0.7)
        self.assertLess(np.abs(check.standardized).mean(), 3.0)  # residuals are O(1) in std units

    def test_sparse_inversion_matches_dense_reference_without_dense_covariance_storage(self):
        dense = linear_gaussian_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)
        sparse = sparse_linear_gaussian_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)

        self.assertIsNone(sparse.cov)
        self.assertIsNotNone(sparse.precision_factor)
        np.testing.assert_allclose(sparse.mean, dense.mean, rtol=1.0e-8, atol=1.0e-8)
        np.testing.assert_allclose(sparse.marginal_variance, dense.marginal_variance, rtol=1.0e-6, atol=1.0e-6)

    def test_sparse_posterior_predictive_uses_precision_covariance_actions(self):
        sparse = sparse_linear_gaussian_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)
        dense = linear_gaussian_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)
        held = Observation(
            kind="borehole",
            location=self.grid.coordinates[[0, 10, 20]],
            value=self.truth[[0, 10, 20]],
            noise_cov=np.full(3, 25.0),
        )

        sparse_check = posterior_predictive_check(sparse, self.registry, [held], alpha=0.1)
        dense_check = posterior_predictive_check(dense, self.registry, [held], alpha=0.1)
        np.testing.assert_allclose(sparse_check.standardized, dense_check.standardized, rtol=1.0e-6, atol=1.0e-6)

    def test_bounded_field_is_rejected_not_silently_linearized(self):
        bounded = Field3D(
            coordinates=self.grid.coordinates, spacing=20.0, units="", property_name="p", bounds=(0.0, 1.0)
        )
        with self.assertRaises(ValueError):
            linear_gaussian_invert(bounded, [self.borehole], self.registry, self.prior)

    def test_nonlinear_operator_without_jacobian_is_rejected(self):
        reg = ForwardOperatorRegistry()
        from mixle_pde.observations import ForwardOperator

        reg.register(ForwardOperator("borehole", predict=lambda g, f, loc: f[:1], jacobian=None))
        with self.assertRaises(ValueError):
            linear_gaussian_invert(self.grid, [self.borehole], reg, self.prior)


if __name__ == "__main__":
    unittest.main()
