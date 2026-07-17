"""MP-N7 -- hybrid surrogate/full-order inversion: DoD conformance.

``hybrid_gauss_newton_invert`` wraps the same MAP objective ``gauss_newton_invert`` solves, letting
early/exploratory iterations spend a cheap ``Surrogate.predict`` call instead of the real forward, while
guaranteeing (a) the calibrated ``defer`` gate forces a fallback whenever it fires, (b) the last
``n_final_full_order`` scheduled iterations are always full-order, and (c) the returned posterior is
always the literal output of a full-order call, never the surrogate's own guess. Most cases below drive
that contract directly with a fast, duck-typed stub surrogate (no NN fit needed to prove the
orchestration logic); one integration test wires a real ``distill_forward``-built ``Surrogate`` through
it end to end.
"""

from __future__ import annotations

import unittest

import numpy as np

from mixle_pde.field_gauss_newton import gauss_newton_invert
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.hybrid_inversion import HybridInversionReport, HybridIterationRecord, hybrid_gauss_newton_invert
from mixle_pde.latent import Field3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator
from mixle_pde.surrogate import distill_forward


def _grid():
    xs = np.linspace(0.0, 100.0, 5)
    ys = np.linspace(0.0, 100.0, 5)
    zs = np.array([-30.0, -50.0])
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    return Field3D(coordinates=pts, spacing=25.0, units="kg/m^3", property_name="density", bounds=(0.0, None))


def _blob(grid, amp):
    d2 = np.sum((grid.coordinates - np.array([50.0, 50.0, -40.0])) ** 2, axis=1)
    return amp * np.exp(-d2 / (2.0 * 35.0**2))


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    grid = _grid()
    truth = _blob(grid, 400.0)
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    observation = Observation(
        kind="borehole",
        location=grid.coordinates,
        value=truth + rng.normal(0, 5.0, size=grid.n),
        noise_cov=np.full(grid.n, 25.0),
    )
    prior = FieldGaussianPrior(
        mean=float(np.log(50.0)),
        smoothness_precision=5.0e-3,
        marginal_precision=1.0e-3,
        length_scale=25.0,
        neighbors=6,
    )
    return grid, [observation], registry, prior


def _small_fixture(seed=1):
    """A low-dimensional fixture (grid.n=6): distill_forward fits one small net per output
    dimension, so a real end-to-end surrogate stays cheap only when grid.n is kept small."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 50.0, 3)
    ys = np.linspace(0.0, 50.0, 2)
    zs = np.array([-20.0])
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    grid = Field3D(coordinates=pts, spacing=25.0, units="kg/m^3", property_name="density", bounds=(0.0, None))
    truth = 40.0 * np.exp(-np.sum((grid.coordinates - np.array([25.0, 25.0, -20.0])) ** 2, axis=1) / (2.0 * 20.0**2))
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    observation = Observation(
        kind="borehole",
        location=grid.coordinates,
        value=truth + rng.normal(0, 2.0, size=grid.n),
        noise_cov=np.full(grid.n, 4.0),
    )
    prior = FieldGaussianPrior(
        mean=float(np.log(5.0)),
        smoothness_precision=1.0e-2,
        marginal_precision=1.0e-2,
        length_scale=20.0,
        neighbors=4,
    )
    return grid, [observation], registry, prior


class _StubSurrogate:
    """A minimal duck-typed stand-in for :class:`mixle_pde.surrogate.Surrogate` -- exercises the
    wrapper's orchestration logic without paying for an NN fit."""

    def __init__(self, *, always_defer: bool, pull: float = 0.5):
        self.always_defer = always_defer
        self.pull = pull
        self.predict_calls = 0
        self.defer_calls = 0

    def predict(self, u: np.ndarray) -> np.ndarray:
        self.predict_calls += 1
        return np.asarray(u, dtype=float) * (1.0 - self.pull)

    def defer(self, u: np.ndarray) -> bool:
        self.defer_calls += 1
        return self.always_defer


class HybridGaussNewtonInvertTest(unittest.TestCase):
    def setUp(self):
        self.grid, self.observations, self.registry, self.prior = _fixture()

    def test_final_scheduled_iteration_is_always_full_order(self):
        surrogate = _StubSurrogate(always_defer=False)
        _, report = hybrid_gauss_newton_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            surrogate,
            max_iterations=5,
            n_final_full_order=2,
        )
        self.assertFalse(report.iterations[-1].used_surrogate)
        self.assertTrue(report.verified_against_full_order)
        # a trusting surrogate that never defers should still get used for the non-reserved iterations
        self.assertGreaterEqual(report.n_surrogate_iterations, 1)
        self.assertGreaterEqual(surrogate.predict_calls, 1)

    def test_defer_forces_full_order_on_every_iteration(self):
        surrogate = _StubSurrogate(always_defer=True)
        _, report = hybrid_gauss_newton_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            surrogate,
            max_iterations=4,
            n_final_full_order=1,
        )
        self.assertEqual(report.n_surrogate_iterations, 0)
        self.assertTrue(all(not record.used_surrogate for record in report.iterations))
        self.assertTrue(all(record.reason == "surrogate_deferred" for record in report.iterations[:-1]))
        self.assertGreater(surrogate.defer_calls, 0)
        self.assertEqual(surrogate.predict_calls, 0)  # never trusted, so predict is never even called

    def test_missing_surrogate_runs_full_order_throughout(self):
        _, report = hybrid_gauss_newton_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            None,
            max_iterations=3,
            n_final_full_order=1,
        )
        self.assertEqual(report.n_surrogate_iterations, 0)
        self.assertTrue(all(record.reason == "surrogate_unavailable" for record in report.iterations))

    def test_iteration_counts_are_internally_consistent(self):
        surrogate = _StubSurrogate(always_defer=False)
        _, report = hybrid_gauss_newton_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            surrogate,
            max_iterations=5,
            n_final_full_order=2,
        )
        self.assertIsInstance(report, HybridInversionReport)
        self.assertIsInstance(report.iterations[0], HybridIterationRecord)
        self.assertEqual(report.n_surrogate_iterations + report.n_full_order_iterations, len(report.iterations))
        self.assertEqual(report.converged, report.final_report.converged)
        full_order_records = [record for record in report.iterations if not record.used_surrogate]
        self.assertEqual(full_order_records[-1].inner_report, report.final_report)

    def test_surrogate_shape_mismatch_is_rejected(self):
        class _WrongShape:
            def predict(self, u):
                return np.asarray(u, dtype=float)[:-1]  # one element short

            def defer(self, u):
                return False

        with self.assertRaises(ValueError):
            # n_final_full_order=1 (< max_iterations) so iteration 0 is not in the reserved tail and
            # actually reaches the surrogate -- otherwise the mismatch would never be exercised.
            hybrid_gauss_newton_invert(
                self.grid,
                self.observations,
                self.registry,
                self.prior,
                _WrongShape(),
                max_iterations=2,
                n_final_full_order=1,
            )

    def test_invalid_iteration_budgets_are_rejected(self):
        with self.assertRaises(ValueError):
            hybrid_gauss_newton_invert(self.grid, self.observations, self.registry, self.prior, max_iterations=0)
        with self.assertRaises(ValueError):
            hybrid_gauss_newton_invert(self.grid, self.observations, self.registry, self.prior, n_final_full_order=0)

    def test_result_is_close_to_a_direct_full_order_solve(self):
        surrogate = _StubSurrogate(always_defer=False, pull=0.2)
        hybrid_post, hybrid_report = hybrid_gauss_newton_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            surrogate,
            max_iterations=6,
            n_final_full_order=2,
            inner_max_iter=100,
        )
        direct_post, direct_report = gauss_newton_invert(self.grid, self.observations, self.registry, self.prior)
        self.assertTrue(hybrid_report.final_report.converged)
        self.assertTrue(direct_report.converged)
        # both chase the same MAP; the hybrid trajectory should land close to the from-scratch solve
        np.testing.assert_allclose(hybrid_post.mean, direct_post.mean, atol=0.5)


class GaussNewtonWarmStartTest(unittest.TestCase):
    """Direct coverage of the small ``u_init`` addition ``hybrid_gauss_newton_invert`` relies on."""

    def setUp(self):
        self.grid, self.observations, self.registry, self.prior = _fixture()

    def test_default_u_init_matches_no_u_init(self):
        post_a, report_a = gauss_newton_invert(self.grid, self.observations, self.registry, self.prior, max_iter=3)
        post_b, report_b = gauss_newton_invert(
            self.grid, self.observations, self.registry, self.prior, max_iter=3, u_init=None
        )
        np.testing.assert_array_equal(post_a.mean, post_b.mean)
        self.assertEqual(report_a, report_b)

    def test_explicit_u_init_changes_the_first_step(self):
        m0 = self.prior.mean_vector(self.grid)
        warm = m0 + 1.0
        post_cold, _ = gauss_newton_invert(self.grid, self.observations, self.registry, self.prior, max_iter=1)
        post_warm, _ = gauss_newton_invert(
            self.grid, self.observations, self.registry, self.prior, max_iter=1, u_init=warm
        )
        self.assertFalse(np.allclose(post_cold.mean, post_warm.mean))

    def test_u_init_wrong_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            gauss_newton_invert(self.grid, self.observations, self.registry, self.prior, u_init=np.zeros(3))


class HybridSurrogateIntegrationTest(unittest.TestCase):
    """One slower, real ``distill_forward``-built surrogate, proving genuine (not duck-typed) wiring."""

    def test_real_distilled_surrogate_is_accepted_and_used(self):
        grid, observations, registry, prior = _small_fixture()
        m0 = prior.mean_vector(grid)

        def _cheap_teacher(u):
            # a fast stand-in "one refinement step" -- pulls halfway toward the prior mean;
            # deliberately NOT calling the real (expensive) forward, so distillation stays cheap.
            return 0.5 * (np.asarray(u, dtype=float) + m0)

        def _sampler(n, rng):
            return m0[None, :] + rng.normal(0.0, 0.5, size=(n, m0.shape[0]))

        surrogate = distill_forward(_cheap_teacher, _sampler, budget=16, seed=0, hidden=(8,), epochs=20)

        _, report = hybrid_gauss_newton_invert(
            grid, observations, registry, prior, surrogate, max_iterations=4, n_final_full_order=2
        )
        self.assertTrue(report.verified_against_full_order)
        self.assertEqual(report.n_surrogate_iterations + report.n_full_order_iterations, len(report.iterations))
        self.assertFalse(report.iterations[-1].used_surrogate)


if __name__ == "__main__":
    unittest.main()
