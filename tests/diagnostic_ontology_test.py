"""Tests for the MP-J3 diagnostic ontology and repair-recipe catalog
(mixle_pde.verification.diagnostic_ontology).

Every classification test drives a real, unmodified call into mixle_pde.continuation or
mixle_pde.pde_backend_registry (never a mock of either) and checks this module's classifier against
what that real call actually returned or raised -- the same "not fabricated" discipline
tests/continuation_test.py and tests/modeling_traces_test.py already hold themselves to. The one
kernel-backed cross-check (CFL_VIOLATION against the real elastic-fd-leapfrog kernel via
mixle_pde.verification.modeling_traces) is import-gated on torch, matching every other test file in
this repository that touches a torch-backed kernel.
"""

from __future__ import annotations

import math

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
from mixle_pde.pde_backend_registry import get_kernel_registration, run_math_problem
from mixle_pde.verification.diagnostic_ontology import (
    REPAIR_CATALOG,
    DiagnosticCode,
    RepairActionKind,
    RepairOutcome,
    apply_repair_attempt,
    classify_backend_dispatch_error,
    classify_continuation_receipt,
    classify_failure_class,
    classify_newton_receipt,
    get_repair_action,
    list_diagnostic_codes,
    run_bounded_repair,
    seeded_failure_corpus,
)

_COURANT_LIMIT = 1.0 / math.sqrt(3.0)


def _oob_problem(*, problem_id: str, operator_kind: str, discretization: str, grid_shape: tuple[int, ...]) -> dict:
    """Minimal CON-MATH-PROBLEM-V1 study, shaped like tests/pde_backend_registry_kernels2_test.py's own
    ``_problem`` builder, carrying only what ``require_compatible`` needs plus an out-of-range
    ``grid_shape``."""
    return {
        "id": problem_id,
        "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": "structured_grid"}}],
        "unknowns": [{"id": "field", "domain_id": "domain"}],
        "operators": [
            {
                "id": f"{problem_id}-operator",
                "kind": operator_kind,
                "input_ids": ["field"],
                "output_ids": ["field"],
                "discretization": discretization,
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [],
        "solve_plan": {"parameters": {"grid_shape": grid_shape}},
    }


# ---------------------------------------------------------------------------
# DiagnosticCode / classify_failure_class
# ---------------------------------------------------------------------------
def test_all_diagnostic_codes_are_listed():
    assert set(list_diagnostic_codes()) == set(DiagnosticCode)
    assert len(list_diagnostic_codes()) == 6


@pytest.mark.parametrize(
    "name,expected",
    [
        ("singular_jacobian", DiagnosticCode.SINGULAR_JACOBIAN),
        ("singular_tangent_jacobian", DiagnosticCode.SINGULAR_JACOBIAN),
        ("non_finite_residual", DiagnosticCode.NON_FINITE_RESULT),
        ("non_finite_jacobian", DiagnosticCode.NON_FINITE_RESULT),
        ("non_finite_step", DiagnosticCode.NON_FINITE_RESULT),
        ("max_iterations_exceeded", DiagnosticCode.MAX_ITERATIONS_EXCEEDED),
        ("cfl_violation_diverged", DiagnosticCode.CFL_VIOLATION),
        ("cfl_violation_unstable_growth", DiagnosticCode.CFL_VIOLATION),
    ],
)
def test_classify_failure_class_covers_every_real_literal(name, expected):
    assert classify_failure_class(name) is expected


def test_classify_failure_class_returns_none_for_unrecognized_string():
    assert classify_failure_class("not_a_real_failure_class") is None


def test_classify_failure_class_does_not_conflate_repair_outcome_labels():
    # modeling_traces.py's own repair-attempt-outcome vocabulary is a different, already-solved
    # problem (whether ITS repair worked), not a diagnostic code -- must not silently classify.
    assert classify_failure_class("repair_still_violates_precondition") is None
    assert classify_failure_class("verification_disagreement") is None


# ---------------------------------------------------------------------------
# classify_newton_receipt -- every branch driven by a real newton_solve call
# ---------------------------------------------------------------------------
def test_classify_newton_receipt_is_none_when_converged():
    problem = bratu_problem(9)
    receipt = newton_solve(problem, u0=np.zeros(9), lam=1.0)
    assert receipt.converged
    assert classify_newton_receipt(receipt) is None


def test_classify_newton_receipt_singular_jacobian():
    # Identical to tests/continuation_test.py's own construction: two rows always equal -> exactly
    # singular for every u/lambda.
    problem = ParametricProblem(
        residual=lambda u, lam: np.array([u[0] - u[1], u[0] - u[1]]),
        jac_u=lambda u, lam: np.array([[1.0, -1.0], [1.0, -1.0]]),
        jac_lambda=lambda u, lam: np.zeros(2),
        n=2,
        name="structurally_singular",
    )
    receipt = newton_solve(problem, u0=np.array([1.0, 0.0]), lam=0.0)
    assert receipt.failure_reason == "singular_jacobian"
    assert classify_newton_receipt(receipt) is DiagnosticCode.SINGULAR_JACOBIAN


def test_classify_newton_receipt_non_finite_residual():
    def residual(u, lam):
        with np.errstate(divide="ignore"):
            return np.array([1.0 / u[0] - lam])

    def jac_u(u, lam):
        with np.errstate(divide="ignore"):
            return np.array([[-1.0 / u[0] ** 2]])

    problem = ParametricProblem(
        residual=residual, jac_u=jac_u, jac_lambda=lambda u, lam: np.array([-1.0]), n=1, name="blowup"
    )
    with np.errstate(divide="ignore"):
        receipt = newton_solve(problem, u0=np.array([0.0]), lam=1.0)
    assert receipt.failure_reason == "non_finite_residual"
    assert classify_newton_receipt(receipt) is DiagnosticCode.NON_FINITE_RESULT


def test_classify_newton_receipt_non_finite_jacobian():
    def jac_u(u, lam):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.array([[1.0 / u[0]]])

    problem = ParametricProblem(
        residual=lambda u, lam: np.array([u[0] - 3.0]),
        jac_u=jac_u,
        jac_lambda=lambda u, lam: np.array([0.0]),
        n=1,
        name="jac_blowup",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        receipt = newton_solve(problem, u0=np.array([0.0]), lam=0.0, max_iterations=5)
    assert receipt.failure_reason == "non_finite_jacobian"
    assert classify_newton_receipt(receipt) is DiagnosticCode.NON_FINITE_RESULT


def test_classify_newton_receipt_non_finite_step():
    # Finite, nonzero (subnormal) Jacobian -- the linear solve itself does not raise LinAlgError -- but
    # dividing a normal residual by it overflows the Newton step to a non-finite delta.
    problem = ParametricProblem(
        residual=lambda u, lam: np.array([2.5]),
        jac_u=lambda u, lam: np.array([[5.0e-320]]),
        jac_lambda=lambda u, lam: np.array([0.0]),
        n=1,
        name="step_overflow",
    )
    with np.errstate(over="ignore"):
        receipt = newton_solve(problem, u0=np.array([1.0]), lam=0.0, max_iterations=5)
    assert receipt.failure_reason == "non_finite_step"
    assert classify_newton_receipt(receipt) is DiagnosticCode.NON_FINITE_RESULT


def test_classify_newton_receipt_max_iterations_exceeded():
    problem = bratu_problem(99)
    receipt = newton_solve(problem, u0=np.zeros(99), lam=1.0, max_iterations=0)
    assert receipt.failure_reason == "max_iterations_exceeded"
    assert classify_newton_receipt(receipt) is DiagnosticCode.MAX_ITERATIONS_EXCEEDED


# ---------------------------------------------------------------------------
# classify_continuation_receipt -- every branch driven by a real continuation call
# ---------------------------------------------------------------------------
def test_classify_continuation_receipt_none_for_target_reached():
    problem = bratu_problem(99)
    receipt = natural_continuation(problem, u0=np.zeros(99), lam0=0.0, lam_target=2.0, n_steps=20)
    assert receipt.stopped_reason == "target_reached"
    assert classify_continuation_receipt(receipt) is None


def test_classify_continuation_receipt_none_for_max_steps_reached():
    problem = bratu_problem(99)
    # A handful of tiny arclength steps well away from the fold: nominal termination, not a failure.
    receipt = arclength_continuation(problem, u0=np.zeros(99), lam0=0.0, ds=0.01, n_steps=3)
    assert receipt.stopped_reason == "max_steps_reached"
    assert classify_continuation_receipt(receipt) is None


def test_classify_continuation_receipt_drills_into_newton_failed():
    # Same configuration as tests/continuation_test.py::test_natural_continuation_cannot_cross_the_fold:
    # natural continuation aimed past the closed-form fold cannot find a root and gives up honestly.
    problem = bratu_problem(99)
    _, lambda_c = bratu_reference_fold()
    receipt = natural_continuation(problem, u0=np.zeros(99), lam0=0.0, lam_target=lambda_c + 1.0, n_steps=60)
    assert receipt.stopped_reason == "newton_failed"
    rejected = next(step for step in receipt.steps if not step.accepted)
    code = classify_continuation_receipt(receipt)
    assert code is classify_newton_receipt(rejected.newton)
    assert code is not None


def test_classify_continuation_receipt_drills_into_initial_point_not_converged():
    problem = bratu_problem(99)
    # lam0=1.0 (not 0): u0=0 does not trivially solve it, so max_iterations=0 genuinely fails to converge.
    receipt = natural_continuation(problem, u0=np.zeros(99), lam0=1.0, lam_target=2.0, n_steps=5, max_iterations=0)
    assert receipt.stopped_reason == "initial_point_not_converged"
    assert receipt.steps[0].newton.failure_reason == "max_iterations_exceeded"
    assert classify_continuation_receipt(receipt) is DiagnosticCode.MAX_ITERATIONS_EXCEEDED


def test_classify_continuation_receipt_singular_tangent_jacobian():
    # Residual identically zero (trivially converged start); Jacobian identically zero (the tangent
    # solve at that converged point is exactly singular) -- the predictor's own failure path, distinct
    # from a Newton corrector's singular_jacobian.
    problem = ParametricProblem(
        residual=lambda u, lam: np.array([0.0]),
        jac_u=lambda u, lam: np.array([[0.0]]),
        jac_lambda=lambda u, lam: np.array([1.0]),
        n=1,
        name="zero_residual_singular_tangent",
    )
    receipt = arclength_continuation(problem, u0=np.array([0.0]), lam0=0.5, ds=0.1, n_steps=5)
    assert receipt.stopped_reason == "singular_tangent_jacobian"
    assert classify_continuation_receipt(receipt) is DiagnosticCode.SINGULAR_JACOBIAN


# ---------------------------------------------------------------------------
# classify_backend_dispatch_error -- real KeyError/ValueError from pde_backend_registry's own boundary
# ---------------------------------------------------------------------------
def test_classify_backend_dispatch_error_unknown_backend():
    with pytest.raises(KeyError) as excinfo:
        get_kernel_registration("not-a-real-backend-id")
    assert classify_backend_dispatch_error(excinfo.value) is DiagnosticCode.UNKNOWN_BACKEND


def test_classify_backend_dispatch_error_out_of_applicability_domain_groundwater():
    # groundwater-fd-transport supports 1-D/2-D grids only; this invoker never imports torch.
    problem = _oob_problem(
        problem_id="oob-groundwater",
        operator_kind="linear_operator",
        discretization="FD-implicit",
        grid_shape=(4, 4, 4),
    )
    with pytest.raises(ValueError) as excinfo:
        run_math_problem(problem, "groundwater-fd-transport")
    assert classify_backend_dispatch_error(excinfo.value) is DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN


def test_classify_backend_dispatch_error_out_of_applicability_domain_helmholtz():
    torch = pytest.importorskip("torch", reason="helmholtz-pml-fd uses the differentiable ops backend")
    del torch
    # helmholtz-pml-fd is 2-D only.
    problem = _oob_problem(
        problem_id="oob-helmholtz",
        operator_kind="linear_operator",
        discretization="FD-helmholtz-pml",
        grid_shape=(8, 8, 8),
    )
    with pytest.raises(ValueError) as excinfo:
        run_math_problem(problem, "helmholtz-pml-fd")
    assert classify_backend_dispatch_error(excinfo.value) is DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN


def test_classify_backend_dispatch_error_none_for_unrelated_exception_type():
    assert classify_backend_dispatch_error(RuntimeError("unrelated")) is None


# ---------------------------------------------------------------------------
# REPAIR_CATALOG structure
# ---------------------------------------------------------------------------
def test_repair_catalog_has_exactly_one_entry_per_code():
    assert set(REPAIR_CATALOG) == set(DiagnosticCode)
    for code, action in REPAIR_CATALOG.items():
        assert action.code is code
        assert get_repair_action(code) is action


def test_repair_catalog_scale_parameter_entries_are_internally_consistent():
    for code in (
        DiagnosticCode.CFL_VIOLATION,
        DiagnosticCode.MAX_ITERATIONS_EXCEEDED,
        DiagnosticCode.NON_FINITE_RESULT,
    ):
        action = REPAIR_CATALOG[code]
        assert action.kind is RepairActionKind.SCALE_PARAMETER
        assert isinstance(action.parameter_name, str) and action.parameter_name
        assert action.retry_policy.max_attempts > 0
        assert action.retry_policy.scale_factor is not None and action.retry_policy.scale_factor > 0


def test_repair_catalog_requires_human_entries_take_no_action():
    for code in (
        DiagnosticCode.SINGULAR_JACOBIAN,
        DiagnosticCode.UNKNOWN_BACKEND,
        DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN,
    ):
        action = REPAIR_CATALOG[code]
        assert action.kind is RepairActionKind.REQUIRES_HUMAN
        assert action.parameter_name is None
        assert action.retry_policy.max_attempts == 0
        assert action.retry_policy.scale_factor is None


def test_repair_catalog_entries_are_frozen():
    action = REPAIR_CATALOG[DiagnosticCode.CFL_VIOLATION]
    with pytest.raises(AttributeError):
        action.parameter_name = "not_dt"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# apply_repair_attempt -- pure, deterministic, bounded
# ---------------------------------------------------------------------------
def test_apply_repair_attempt_scales_from_the_original_value_each_time():
    action = REPAIR_CATALOG[DiagnosticCode.CFL_VIOLATION]
    params = {"dt": 0.08}
    first = apply_repair_attempt(action, params, attempt=1)
    second = apply_repair_attempt(action, params, attempt=2)
    assert first["dt"] == pytest.approx(0.08 * 0.5)
    assert second["dt"] == pytest.approx(0.08 * 0.25)  # from the ORIGINAL 0.08, not compounded on `first`
    # purity: original params untouched, repeated calls agree
    assert params["dt"] == 0.08
    assert apply_repair_attempt(action, params, attempt=1) == first


def test_apply_repair_attempt_none_outside_bounds():
    action = REPAIR_CATALOG[DiagnosticCode.MAX_ITERATIONS_EXCEEDED]
    assert apply_repair_attempt(action, {"max_iterations": 10}, attempt=0) is None
    assert apply_repair_attempt(action, {"max_iterations": 10}, attempt=action.retry_policy.max_attempts + 1) is None


def test_apply_repair_attempt_none_for_requires_human():
    action = REPAIR_CATALOG[DiagnosticCode.SINGULAR_JACOBIAN]
    assert apply_repair_attempt(action, {}, attempt=1) is None


def test_apply_repair_attempt_raises_for_missing_parameter():
    action = REPAIR_CATALOG[DiagnosticCode.CFL_VIOLATION]
    with pytest.raises(KeyError):
        apply_repair_attempt(action, {"not_dt": 1.0}, attempt=1)


# ---------------------------------------------------------------------------
# run_bounded_repair -- real, honest resolution against genuinely re-verified conditions
# ---------------------------------------------------------------------------
def test_run_bounded_repair_resolves_a_real_cfl_violation():
    vp, spacing, violation_factor = 2.0, 0.1, 3.0
    dt = violation_factor * _COURANT_LIMIT * spacing / vp
    assert (vp * dt / spacing) > _COURANT_LIMIT  # genuine violation by construction

    action = REPAIR_CATALOG[DiagnosticCode.CFL_VIOLATION]
    run = run_bounded_repair(
        action,
        {"dt": dt},
        is_resolved=lambda params: (vp * params["dt"] / spacing) <= _COURANT_LIMIT + 1.0e-12,
    )
    assert run.outcome is RepairOutcome.RESOLVED
    assert 1 <= run.attempts_used <= action.retry_policy.max_attempts
    # independently re-verify the repair's own claim, not just trust the outcome label
    assert (vp * run.final_params["dt"] / spacing) <= _COURANT_LIMIT + 1.0e-12


def test_run_bounded_repair_resolves_a_real_max_iterations_exceeded():
    # Empirically confirmed: bratu_problem(5) at lam=1.0 from u0=0 needs 3 Newton iterations;
    # max_iterations=1 and 2 both genuinely fail, 4 genuinely succeeds.
    problem = bratu_problem(5)
    u0 = np.zeros(5)
    receipt = newton_solve(problem, u0, lam=1.0, max_iterations=1)
    assert receipt.failure_reason == "max_iterations_exceeded"

    action = REPAIR_CATALOG[DiagnosticCode.MAX_ITERATIONS_EXCEEDED]

    def is_resolved(params: dict) -> bool:
        return newton_solve(problem, u0, lam=1.0, max_iterations=int(params["max_iterations"])).converged

    run = run_bounded_repair(action, {"max_iterations": 1}, is_resolved=is_resolved)
    assert run.outcome is RepairOutcome.RESOLVED
    assert run.attempts_used == 2  # attempt 1 -> 2 (still fails), attempt 2 -> 4 (converges)
    # independently re-verify: re-running Newton with the repaired budget genuinely converges
    final_receipt = newton_solve(problem, u0, lam=1.0, max_iterations=int(run.final_params["max_iterations"]))
    assert final_receipt.converged


def test_run_bounded_repair_bounded_retry_machinery_resolves_within_budget():
    action = REPAIR_CATALOG[DiagnosticCode.NON_FINITE_RESULT]
    run = run_bounded_repair(action, {"damping": 1.0}, is_resolved=lambda params: params["damping"] <= 0.3)
    assert run.outcome is RepairOutcome.RESOLVED
    assert run.attempts_used == 2  # 1.0 * 0.5**1 = 0.5 (no); 1.0 * 0.5**2 = 0.25 (yes)
    assert run.final_params["damping"] == pytest.approx(0.25)


def test_run_bounded_repair_bounded_retry_machinery_exhausts_without_looping_forever():
    action = REPAIR_CATALOG[DiagnosticCode.NON_FINITE_RESULT]
    calls = []

    def is_resolved(params: dict) -> bool:
        calls.append(params["damping"])
        return False  # never resolves -- must terminate at max_attempts, not hang

    run = run_bounded_repair(action, {"damping": 1.0}, is_resolved=is_resolved)
    assert run.outcome is RepairOutcome.ATTEMPTS_EXHAUSTED
    assert run.attempts_used == action.retry_policy.max_attempts
    assert len(calls) == action.retry_policy.max_attempts  # exactly bounded, no extra/infinite calls


def test_run_bounded_repair_requires_human_never_calls_is_resolved():
    action = REPAIR_CATALOG[DiagnosticCode.SINGULAR_JACOBIAN]

    def is_resolved(params: dict) -> bool:
        raise AssertionError("is_resolved must never be called for a REQUIRES_HUMAN action")

    run = run_bounded_repair(action, {}, is_resolved=is_resolved)
    assert run.outcome is RepairOutcome.REQUIRES_HUMAN
    assert run.attempts_used == 0


# ---------------------------------------------------------------------------
# seeded_failure_corpus -- the MP-J3 accept bar ("at least 100 seeded failures map to stable codes")
# ---------------------------------------------------------------------------
def test_seeded_failure_corpus_produces_at_least_100_items_all_mapped_to_stable_codes():
    corpus = seeded_failure_corpus(seed=20260716, count=100)
    assert len(corpus) == 100
    for item in corpus:
        assert isinstance(item.code, DiagnosticCode)
    # every one of the six codes is genuinely reachable, not just a subset
    assert {item.code for item in corpus} == set(DiagnosticCode)


def test_seeded_failure_corpus_is_deterministic():
    first = seeded_failure_corpus(seed=7, count=27)
    second = seeded_failure_corpus(seed=7, count=27)
    assert first == second


def test_seeded_failure_corpus_rejects_non_positive_count():
    with pytest.raises(ValueError):
        seeded_failure_corpus(seed=1, count=0)


def test_seeded_failure_corpus_items_are_traceable_to_a_real_source():
    corpus = seeded_failure_corpus(seed=3, count=18)
    for item in corpus:
        assert item.source.startswith("mixle_pde.")
        assert item.detail  # never an empty, unexplained record


# ---------------------------------------------------------------------------
# Cross-check against the real, kernel-verified CFL trace generator (modeling_traces.py, MP-M1) --
# proof this taxonomy actually classifies what that already-shipped, kernel-backed module produces.
# ---------------------------------------------------------------------------
def test_cfl_violation_classifies_the_real_kernel_verified_modeling_trace():
    pytest.importorskip("torch", reason="elastic-fd-leapfrog uses the differentiable ops backend")
    from mixle_pde.verification.modeling_traces import generate_elastic_cfl_trace, generate_elastic_cfl_unresolved_trace

    repaired_trace = generate_elastic_cfl_trace(seed=11)
    unresolved_trace = generate_elastic_cfl_unresolved_trace(seed=11)
    assert classify_failure_class(repaired_trace.detected_failure_class) is DiagnosticCode.CFL_VIOLATION
    assert classify_failure_class(unresolved_trace.detected_failure_class) is DiagnosticCode.CFL_VIOLATION
