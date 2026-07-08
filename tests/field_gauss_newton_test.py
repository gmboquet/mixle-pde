"""Gauss-Newton inversion for bounded (nonlinear-transform) fields (workstream G6 ladder rung 2).

The linear-Gaussian engine handles only identity-transform fields; this closes the positivity-/bounds-
constrained case, so a recovered density-contrast field and all its posterior samples stay non-negative.
"""

import unittest

import numpy as np

from mixle_pde.field_gauss_newton import gauss_newton_invert
from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    dc_resistivity_forward_operator,
    gravity_forward_operator,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _grid(bounds):
    xs = np.linspace(0.0, 100.0, 5)
    ys = np.linspace(0.0, 100.0, 5)
    zs = np.array([-30.0, -50.0])
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    return Field3D(coordinates=pts, spacing=25.0, units="kg/m^3", property_name="density", bounds=bounds)


def _blob(grid, amp):
    d2 = np.sum((grid.coordinates - np.array([50.0, 50.0, -40.0])) ** 2, axis=1)
    return amp * np.exp(-d2 / (2.0 * 35.0**2))


class GaussNewtonInversionTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        # non-negative density contrast: lower bound 0, no upper bound (log transform)
        self.grid = _grid(bounds=(0.0, None))
        self.truth = _blob(self.grid, 400.0)  # strictly positive body
        self.volumes = np.full(self.grid.n, 25.0**3, dtype=float)
        self.registry = ForwardOperatorRegistry()
        self.registry.register(gravity_forward_operator(self.grid.coordinates, self.volumes))
        self.registry.register(borehole_forward_operator())

        gx, gy = np.meshgrid(np.linspace(0, 100, 5), np.linspace(0, 100, 5))
        self.grav_loc = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
        G = self.registry.get("gravity").jacobian(self.grid, self.grav_loc)
        self.gravity = Observation(
            kind="gravity",
            location=self.grav_loc,
            value=G @ self.truth + self.rng.normal(0, 2.0e-4, size=G.shape[0]),
            noise_cov=np.full(G.shape[0], (2.0e-4) ** 2),
        )
        idx = self.rng.choice(self.grid.n, size=int(0.5 * self.grid.n), replace=False)
        self.borehole = Observation(
            kind="borehole",
            location=self.grid.coordinates[idx],
            value=self.truth[idx] + self.rng.normal(0, 5.0, size=idx.shape),
            noise_cov=np.full(idx.shape, 25.0),
        )
        # prior mean in unconstrained (log) space: log(50) ~ a modest positive background
        self.prior = FieldGaussianPrior(
            mean=float(np.log(50.0)),
            smoothness_precision=5.0e-3,
            marginal_precision=1.0e-3,
            length_scale=25.0,
            neighbors=6,
        )

    def test_recovers_a_positive_body_and_every_sample_stays_nonnegative(self):
        post, report = gauss_newton_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)
        self.assertTrue(report.converged)
        recovered = self.grid.from_unconstrained(post.mean)  # physical density
        self.assertTrue(np.all(recovered >= 0.0))  # positivity by construction
        corr = np.corrcoef(recovered, self.truth)[0, 1]
        self.assertGreater(corr, 0.85)
        # every posterior sample also respects the bound
        samples = post.sample(64, self.rng)
        self.assertTrue(np.all(samples >= 0.0))
        lo, hi = post.credible_interval(0.1)
        self.assertTrue(np.all(lo >= 0.0))

    def test_converges_and_reduces_the_data_misfit(self):
        post, report = gauss_newton_invert(self.grid, [self.gravity, self.borehole], self.registry, self.prior)
        self.assertTrue(report.converged)
        self.assertLessEqual(report.iterations, 100)
        # the log-transform inversion is stiff (damped GN takes many small steps), but the first big
        # step dwarfs the last, and the data misfit is finite and small
        self.assertTrue(report.step_norms[-1] < report.step_norms[0])
        self.assertTrue(np.isfinite(report.final_data_misfit))

    def test_identity_transform_matches_the_linear_gaussian_solve(self):
        # with bounds=None GN reduces to the exact linear-Gaussian solution (one Newton step is exact).
        grid = _grid(bounds=None)
        truth = _blob(grid, 400.0)
        vols = np.full(grid.n, 25.0**3)
        reg = ForwardOperatorRegistry()
        reg.register(gravity_forward_operator(grid.coordinates, vols))
        reg.register(borehole_forward_operator())
        idx = np.arange(0, grid.n, 2)
        bore = Observation(
            kind="borehole", location=grid.coordinates[idx], value=truth[idx], noise_cov=np.full(idx.shape, 25.0)
        )
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=2e-3, marginal_precision=1e-4, length_scale=25.0)

        gn_post, _ = gauss_newton_invert(grid, [bore], reg, prior)
        lg_post = linear_gaussian_invert(grid, [bore], reg, prior)
        np.testing.assert_allclose(gn_post.mean, lg_post.mean, atol=1e-6)

    def test_no_observations_and_missing_jacobian_are_rejected(self):
        with self.assertRaises(ValueError):
            gauss_newton_invert(self.grid, [], self.registry, self.prior)
        reg = ForwardOperatorRegistry()
        from mixle_pde.observations import ForwardOperator

        reg.register(ForwardOperator("borehole", predict=lambda g, f, loc: f[:1], jacobian=None))
        with self.assertRaises(ValueError):
            gauss_newton_invert(self.grid, [self.borehole], reg, self.prior)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DCResistivityGaussNewtonTest(unittest.TestCase):
    def test_nonlinear_dc_resistivity_operator_reduces_data_residual(self):
        shape = (4, 4, 4)
        idx = np.arange(np.prod(shape)).reshape(shape)
        coords = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)], dtype=float)
        grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity")
        schedule = [
            (int(idx[1, 1, 1]), int(idx[1, 1, 2]), int(idx[2, 1, 1]), int(idx[2, 1, 2])),
            (int(idx[1, 2, 1]), int(idx[1, 2, 2]), int(idx[2, 2, 1]), int(idx[2, 2, 2])),
            (int(idx[1, 1, 1]), int(idx[1, 2, 1]), int(idx[2, 1, 2]), int(idx[2, 2, 2])),
        ]
        locations = np.array([[1.5, 1.0, 1.5], [1.5, 2.0, 1.5], [1.5, 1.5, 1.5]])
        truth = np.zeros(grid.n)
        truth[int(idx[2, 1, 1])] = 0.25
        truth[int(idx[2, 2, 2])] = -0.15
        op = dc_resistivity_forward_operator(
            shape,
            schedule,
            sigma_ref=0.02,
            log_data=True,
            finite_difference_step=3.0e-5,
        )
        registry = ForwardOperatorRegistry()
        registry.register(op)
        observation = Observation(
            "dc_resistivity",
            locations,
            op.predict(grid, truth, locations),
            np.full(len(schedule), 0.02**2),
        )
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0e-3, marginal_precision=5.0e-2, length_scale=2.0)
        posterior, report = gauss_newton_invert(grid, [observation], registry, prior, max_iter=6, tol=1.0e-8)

        prior_residual = np.linalg.norm(observation.value - op.predict(grid, np.zeros(grid.n), locations))
        posterior_residual = np.linalg.norm(observation.value - op.predict(grid, posterior.mean, locations))
        self.assertLess(posterior_residual, prior_residual)
        self.assertTrue(np.isfinite(report.final_data_misfit))


if __name__ == "__main__":
    unittest.main()
