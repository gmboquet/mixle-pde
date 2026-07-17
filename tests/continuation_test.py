"""Tests for mixle_pde.continuation: Newton iteration, natural-parameter continuation, and
pseudo-arclength continuation.

The acceptance bar here is the one the work plan names explicitly for this task: continuation must
actually cross a fold on the classic 1-D Bratu problem ``u'' + lambda * exp(u) = 0``, ``u(0) = u(1) =
0``, "with the appropriate method." That has two parts this file checks directly rather than merely
running the solver and looking at exit codes:

1. Natural-parameter continuation (step ``lambda``, re-solve by Newton) structurally *cannot* cross the
   fold -- there is no root beyond the critical ``lambda`` at fixed ``lambda`` -- and must say so
   honestly (a concrete failure reason, not a fabricated convergence).
2. Pseudo-arclength continuation *can* cross it, and the discovered fold location is checked against an
   independent closed-form reference (:func:`mixle_pde.continuation.bratu_reference_fold`), not just
   against the solver's own internal consistency.

Every accepted continuation step is additionally re-verified to be a genuine root (residual below
``tol``), not merely a point the stepper happened to land on -- the same "receipt, not a fabricated
converged" bar the module's docstrings commit to.

MP-F2's Jacobian-free Newton-Krylov corrector (``corrector="jfnk"``) is held to the same fold-crossing
bar as the default dense corrector (``test_arclength_continuation_jfnk_crosses_the_fold_and_matches_reference``),
plus a direct step-by-step cross-check against the dense corrector on the identical scenario
(``test_arclength_continuation_jfnk_matches_dense_corrector_pointwise``) -- the two are alternative
linear-solve strategies for the same augmented nonlinear system and must land on the same branch, not
merely agree on the headline fold number.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.continuation import (
    ParametricProblem,
    arclength_continuation,
    bratu_problem,
    bratu_reference_fold,
    natural_continuation,
    newton_solve,
)

_N = 99  # interior finite-difference nodes for the discretized Bratu problem


def test_newton_solve_converges_quadratically_on_bratu():
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)
    receipt = newton_solve(problem, u0, lam=1.0)

    assert receipt.converged
    assert receipt.failure_reason is None
    assert receipt.residual_history[-1] < 1e-9
    # Newton on a smooth residual from a reasonable start converges quadratically: each residual should
    # shrink by a much larger factor than the last once it is in the basin of attraction.
    ratios = [
        receipt.residual_history[i] / receipt.residual_history[i + 1]
        for i in range(len(receipt.residual_history) - 1)
        if receipt.residual_history[i + 1] > 0
    ]
    assert ratios[-1] > 1e3, f"final convergence step was not quadratic-like: ratios={ratios}"


def test_bratu_problem_rejects_too_few_nodes():
    with pytest.raises(ValueError):
        bratu_problem(2)


def test_natural_continuation_reaches_target_below_the_fold():
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)
    target = 2.0  # comfortably below the ~3.5138 fold

    receipt = natural_continuation(problem, u0, lam0=0.0, lam_target=target, n_steps=20)

    assert receipt.stopped_reason == "target_reached"
    assert receipt.crossed_fold is False
    assert all(step.accepted for step in receipt.steps)
    for step in receipt.steps:
        assert step.newton.residual_history[-1] < 1e-9

    # Cross-check: solving directly at the target lambda from scratch agrees with the continuation
    # path's own final point, to solver tolerance -- two independent routes to the same root.
    direct = newton_solve(problem, u0, lam=target)
    assert direct.converged
    assert np.max(np.abs(receipt.steps[-1].u - direct.u)) < 1e-6


def test_natural_continuation_cannot_cross_the_fold():
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)
    _, lambda_c = bratu_reference_fold()

    receipt = natural_continuation(problem, u0, lam0=0.0, lam_target=lambda_c + 1.0, n_steps=60)

    assert receipt.stopped_reason == "newton_failed"
    assert receipt.crossed_fold is False

    accepted = [step for step in receipt.steps if step.accepted]
    rejected = [step for step in receipt.steps if not step.accepted]
    assert accepted and rejected
    # Every accepted point is a genuine root -- natural continuation never fabricates a converged step.
    for step in accepted:
        assert step.newton.residual_history[-1] < 1e-9
    # The stepper gives up strictly below the fold; it must not report having reached or passed it.
    assert max(step.parameter for step in accepted) < lambda_c
    # The rejected step honestly carries a concrete failure reason, never a silent or fabricated one.
    assert rejected[0].newton.converged is False
    assert rejected[0].newton.failure_reason is not None


def test_arclength_continuation_crosses_the_fold_and_matches_reference():
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)
    theta_c, lambda_c = bratu_reference_fold()
    assert lambda_c == pytest.approx(3.513830719, abs=1e-6)  # sanity-check the reference itself

    receipt = arclength_continuation(problem, u0, lam0=0.0, ds=0.05, n_steps=400)

    assert receipt.crossed_fold is True
    accepted = [step for step in receipt.steps if step.accepted]
    assert len(accepted) == len(receipt.steps), "arclength continuation should not have rejected steps here"
    # Every single accepted step -- both branches, not just the endpoints -- is a genuine root.
    for step in accepted:
        assert step.newton.residual_history[-1] < 1e-9

    lambda_values = [step.parameter for step in accepted]
    lambda_max = max(lambda_values)
    fold_index = lambda_values.index(lambda_max)

    # The numerically discovered fold agrees with the independent closed-form reference to well within
    # discretization (O(h^2), h = 1/100 here) and step-size (ds = 0.05) error.
    assert lambda_max == pytest.approx(lambda_c, abs=1e-2)
    # A genuine fold: parameter direction of travel reverses (later lambdas trend back down).
    assert lambda_values[-1] < lambda_max
    # The classic Bratu signature of the upper branch: past the fold, amplitude keeps growing while
    # lambda falls back -- not merely a stall at the turning point.
    fold_amplitude = float(np.max(np.abs(accepted[fold_index].u)))
    end_amplitude = float(np.max(np.abs(accepted[-1].u)))
    assert end_amplitude > fold_amplitude * 1.5


def test_arclength_continuation_jfnk_crosses_the_fold_and_matches_reference():
    # Same scenario, tolerances, and assertions as
    # test_arclength_continuation_crosses_the_fold_and_matches_reference above, with corrector="jfnk"
    # in place of the default "dense": the matrix-free Jacobian-free Newton-Krylov corrector must clear
    # the identical acceptance bar the dense bordered-Jacobian corrector already clears (MP-F2).
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)
    _theta_c, lambda_c = bratu_reference_fold()

    receipt = arclength_continuation(problem, u0, lam0=0.0, ds=0.05, n_steps=400, corrector="jfnk")

    assert receipt.crossed_fold is True
    accepted = [step for step in receipt.steps if step.accepted]
    assert len(accepted) == len(receipt.steps), "jfnk arclength continuation should not have rejected steps here"
    for step in accepted:
        assert step.newton.residual_history[-1] < 1e-9

    lambda_values = [step.parameter for step in accepted]
    lambda_max = max(lambda_values)

    assert lambda_max == pytest.approx(lambda_c, abs=1e-2)
    assert lambda_values[-1] < lambda_max
    fold_index = lambda_values.index(lambda_max)
    fold_amplitude = float(np.max(np.abs(accepted[fold_index].u)))
    end_amplitude = float(np.max(np.abs(accepted[-1].u)))
    assert end_amplitude > fold_amplitude * 1.5


def test_arclength_continuation_jfnk_matches_dense_corrector_pointwise():
    # The direct MP-F2 cross-check the task requires: the matrix-free JFNK corrector must trace the
    # same branch as the full-Jacobian dense corrector it is an alternative to -- not merely agree on
    # the final fold location, but land on the same (u, lambda) at every single accepted step, since
    # both are Newton-correcting the identical augmented system from the identical predictor point.
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)

    dense = arclength_continuation(problem, u0, lam0=0.0, ds=0.05, n_steps=400, corrector="dense")
    jfnk = arclength_continuation(problem, u0, lam0=0.0, ds=0.05, n_steps=400, corrector="jfnk")

    assert dense.crossed_fold is True
    assert jfnk.crossed_fold is True
    assert dense.stopped_reason == jfnk.stopped_reason

    dense_accepted = [step for step in dense.steps if step.accepted]
    jfnk_accepted = [step for step in jfnk.steps if step.accepted]
    assert len(dense_accepted) == len(jfnk_accepted)

    max_u_diff = max(float(np.max(np.abs(sd.u - sj.u))) for sd, sj in zip(dense_accepted, jfnk_accepted))
    max_lambda_diff = max(abs(sd.parameter - sj.parameter) for sd, sj in zip(dense_accepted, jfnk_accepted))
    # Both correctors converge the same augmented residual to the same 1e-9 tolerance from the same
    # predictor point every step, so they should agree far tighter than that -- 1e-6 leaves comfortable
    # headroom above the ~1e-12 agreement actually observed while still being a real, meaningful bound.
    assert max_u_diff < 1e-6
    assert max_lambda_diff < 1e-6

    dense_lambda_max = max(step.parameter for step in dense_accepted)
    jfnk_lambda_max = max(step.parameter for step in jfnk_accepted)
    assert jfnk_lambda_max == pytest.approx(dense_lambda_max, abs=1e-6)


def test_arclength_continuation_rejects_unknown_corrector():
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)
    with pytest.raises(ValueError):
        arclength_continuation(problem, u0, lam0=0.0, ds=0.05, n_steps=1, corrector="bogus")


def test_arclength_continuation_jfnk_reports_max_iterations_exceeded():
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)  # u=0 is an exact root at lambda=0, so the initial point converges even with
    # max_iterations=0; the first predicted step off that root is not itself a root, so the jfnk
    # corrector's Newton loop -- capped at zero iterations -- must give up honestly rather than
    # fabricate convergence.

    receipt = arclength_continuation(problem, u0, lam0=0.0, ds=0.05, n_steps=1, max_iterations=0, corrector="jfnk")

    assert receipt.stopped_reason == "newton_failed"
    rejected = [step for step in receipt.steps if not step.accepted]
    assert rejected
    assert rejected[0].newton.converged is False
    assert rejected[0].newton.failure_reason == "max_iterations_exceeded"


def test_arclength_continuation_jfnk_reports_non_finite_residual():
    # log(u) is non-finite over the entire u <= 0 half-line, not only at a single point, so a
    # deliberately oversized first step reliably lands the jfnk corrector's very first residual
    # evaluation in that region without needing to hit an exact singularity.
    def residual(u, lam):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.array([np.log(u[0]) - lam])

    def jac_u(u, lam):
        return np.array([[1.0 / u[0]]])

    problem = ParametricProblem(
        residual=residual,
        jac_u=jac_u,
        jac_lambda=lambda u, lam: np.array([-1.0]),
        n=1,
        name="log_blowup",
    )
    u0 = np.array([1.0])  # log(1) - 0 == 0: an exact root at lambda=0, so the tangent is well-defined.

    receipt = arclength_continuation(problem, u0, lam0=0.0, ds=2.0, n_steps=1, direction=-1.0, corrector="jfnk")

    rejected = [step for step in receipt.steps if not step.accepted]
    assert rejected
    assert rejected[0].newton.converged is False
    assert rejected[0].newton.failure_reason == "non_finite_residual"


def test_newton_solve_reports_singular_jacobian_not_fabricated_success():
    # A residual whose two rows are always identical has an everywhere-singular Jacobian by
    # construction, regardless of lambda -- a controlled, deterministic check of the failure path.
    problem = ParametricProblem(
        residual=lambda u, lam: np.array([u[0] - u[1], u[0] - u[1]]),
        jac_u=lambda u, lam: np.array([[1.0, -1.0], [1.0, -1.0]]),
        jac_lambda=lambda u, lam: np.zeros(2),
        n=2,
        name="structurally_singular",
    )
    receipt = newton_solve(problem, u0=np.array([1.0, 0.0]), lam=0.0)

    assert receipt.converged is False
    assert receipt.failure_reason == "singular_jacobian"


def test_newton_solve_reports_max_iterations_exceeded():
    problem = bratu_problem(_N)
    u0 = np.zeros(_N)  # nonzero residual at lambda=1 (u=0 only solves lambda=0)

    receipt = newton_solve(problem, u0, lam=1.0, max_iterations=0)

    assert receipt.converged is False
    assert receipt.failure_reason == "max_iterations_exceeded"
    assert receipt.iterations == 0


def test_newton_solve_reports_non_finite_residual():
    # 1/u[0] - lam blows up to +/-inf exactly at u[0] = 0, the chosen initial guess -- deterministic.
    def residual(u, lam):
        with np.errstate(divide="ignore"):
            return np.array([1.0 / u[0] - lam])

    def jac_u(u, lam):
        with np.errstate(divide="ignore"):
            return np.array([[-1.0 / u[0] ** 2]])

    problem = ParametricProblem(
        residual=residual,
        jac_u=jac_u,
        jac_lambda=lambda u, lam: np.array([-1.0]),
        n=1,
        name="blowup_at_origin",
    )
    with np.errstate(divide="ignore"):
        receipt = newton_solve(problem, u0=np.array([0.0]), lam=1.0)

    assert receipt.converged is False
    assert receipt.failure_reason == "non_finite_residual"
