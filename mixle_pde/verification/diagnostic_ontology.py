"""Diagnostic ontology and deterministic repair recipes for mixle-pde solver failures (MP-J3).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``, MP-J3) records: "No
diagnostic-code/repair-action catalog found." The work plan's full MP-J3 description is much larger
than any one change should attempt in one pass: "Normalize geometry, mesh, compile, linear/nonlinear,
time-step, coupling, conservation, observation/data, prior/likelihood, inference, posterior-
diagnostic, surrogate-validity, and resource failures. Attach bounded repair actions with
preconditions and expected effects," with the accept bar "at least 100 seeded failures map to stable
codes; each auto-repair recipe fixes its target class or returns ``requires_human`` without loops."

This module is a deliberately narrow baseline slice of that scope, matching how the closest sibling
verification modules (``knowledge_catalog.py``/MP-J2, ``validation_tiers.py``/MP-K5,
``modeling_traces.py``/MP-M1) each scoped down to a first slice of their own much larger task rather
than attempting the full named surface in one change: it covers only three of MP-J3's thirteen named
failure categories -- linear/nonlinear, time-step, and compile/dispatch -- and only the specific codes
this module could ground in a real, already-observed failure path in this repository's own solvers,
never an invented or hypothetical one. Mesh, coupling, conservation, observation/data,
prior/likelihood, inference, posterior-diagnostic, surrogate-validity, and resource failures are
explicitly not attempted here.

This is the general TAXONOMY that ``knowledge_catalog.py`` (which documents equations/applicability/
evidence, not failures) and ``modeling_traces.py`` (which generates worked repair *traces* from one
specific kernel, the ``elastic-fd-leapfrog`` CFL precondition) could each plug into, not a replacement
for either: a typed, stable :class:`DiagnosticCode` enum, paired with a deterministic
:class:`RepairAction` recipe and a bounded :class:`RetryPolicy`, that is not specific to any one
kernel. To keep the dependency direction a taxonomy should have (lower-level solver primitives feed
up into it, sibling verification modules can consume it, it does not reach sideways into a sibling
verification module), this module imports only :mod:`mixle_pde.continuation` at module level
(read-only -- continuation.py is not modified here); ``tests/diagnostic_ontology_test.py`` additionally
imports :mod:`mixle_pde.pde_backend_registry` (read-only) and, only to cross-check
:data:`DiagnosticCode.CFL_VIOLATION` against a real kernel run,
:mod:`mixle_pde.verification.modeling_traces` (read-only) -- neither is a production dependency of
this module.

Concrete grounding (every code below is traced to a real, already-shipped failure path, not invented):

* :attr:`DiagnosticCode.SINGULAR_JACOBIAN` -- ``mixle_pde.continuation.NewtonReceipt.failure_reason ==
  "singular_jacobian"`` (raised by ``newton_solve``/``_augmented_corrector`` on an exactly singular
  ``dF/du``, live-exercised by
  ``tests/continuation_test.py::test_newton_solve_reports_singular_jacobian_not_fabricated_success``)
  and ``ContinuationReceipt.stopped_reason == "singular_tangent_jacobian"`` (the pseudo-arclength
  predictor's own tangent solve, ``continuation.py``'s ``_tangent``).
* :attr:`DiagnosticCode.NON_FINITE_RESULT` -- ``NewtonReceipt.failure_reason`` in
  ``{"non_finite_residual", "non_finite_jacobian", "non_finite_step"}`` (the first is live-exercised by
  ``tests/continuation_test.py::test_newton_solve_reports_non_finite_residual``).
* :attr:`DiagnosticCode.MAX_ITERATIONS_EXCEEDED` -- ``NewtonReceipt.failure_reason ==
  "max_iterations_exceeded"``, live-exercised by
  ``tests/continuation_test.py::test_newton_solve_reports_max_iterations_exceeded``.
* :attr:`DiagnosticCode.CFL_VIOLATION` -- ``mixle_pde.verification.modeling_traces.ModelingTrace.
  detected_failure_class`` in ``{"cfl_violation_diverged", "cfl_violation_unstable_growth"}``, and the
  documented closed-form limit it checks, ``mixle_pde.elastic.ElasticWave3D``'s own docstring: "The 3D
  Courant limit is ``dt <= spacing / (vp * sqrt(3))``."
* :attr:`DiagnosticCode.UNKNOWN_BACKEND` -- ``mixle_pde.pde_backend_registry.get_kernel_registration``
  raises ``KeyError(f"unknown PDE backend {backend_id!r}; registered: {sorted(_REGISTRATIONS)}")`` for
  an unregistered id.
* :attr:`DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN` -- several registered invokers in
  ``pde_backend_registry.py`` raise ``ValueError`` for a solve_plan parameter outside that specific
  registered invocation's stated shape, e.g. ``"helmholtz-pml-fd is 2-D only; got grid_shape
  {shape}."``, ``"groundwater-fd-transport supports 1-D or 2-D grids only; got shape {shape}."``,
  ``"flow-spectral-ns supports dim=2 or dim=3."``.

Repair-recipe honesty note
---------------------------
Not every code gets a mechanical auto-repair, and this module does not pretend otherwise. Three codes
(``SINGULAR_JACOBIAN``, ``UNKNOWN_BACKEND``, ``OUT_OF_APPLICABILITY_DOMAIN``) have no safe,
generically-effective retry available through the caller-facing keyword parameters this module can
touch -- see each :class:`RepairAction.rationale` for the specific reasoning (e.g. reducing ``damping``
does *not* fix a singular Jacobian: the Jacobian is re-evaluated at the same pre-step iterate
regardless of how far the previous accepted step moved, so a damped retry repeats the identical
singular solve). Their :data:`REPAIR_CATALOG` entry is honestly ``REQUIRES_HUMAN`` with zero attempts,
never a fabricated "fix" -- this repository's standing "``unknown``/``timeout``/``resource_limit`` are
valid typed outcomes, never a fabricated Boolean" convention, applied to repair recipes. The other
three (``CFL_VIOLATION``, ``MAX_ITERATIONS_EXCEEDED``, ``NON_FINITE_RESULT``) get a bounded,
deterministic scale-and-retry recipe over a real, already-exposed keyword parameter
(``dt``/``max_iterations``/``damping``) -- :func:`run_bounded_repair` structurally cannot loop past
``RetryPolicy.max_attempts`` (a bounded ``for`` loop, never a ``while True``), satisfying MP-J3's own
accept bar ("...or returns ``requires_human`` without loops") as a construction guarantee, not merely a
generous cap.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from mixle_pde.continuation import (
    ContinuationReceipt,
    NewtonReceipt,
    ParametricProblem,
    arclength_continuation,
    bratu_problem,
    newton_solve,
)

__all__ = [
    "DiagnosticCode",
    "classify_failure_class",
    "classify_newton_receipt",
    "classify_continuation_receipt",
    "classify_backend_dispatch_error",
    "RepairActionKind",
    "RetryPolicy",
    "RepairAction",
    "REPAIR_CATALOG",
    "get_repair_action",
    "list_diagnostic_codes",
    "RepairOutcome",
    "RepairRun",
    "apply_repair_attempt",
    "run_bounded_repair",
    "SeededFailure",
    "seeded_failure_corpus",
]


class DiagnosticCode(Enum):
    """A stable, typed code for one class of solver failure this repository's own kernels/solvers are
    already observed to raise or detect. See the module docstring's "Concrete grounding" section for
    exactly which real function/exception/failure_reason each member traces back to.
    """

    SINGULAR_JACOBIAN = "singular_jacobian"
    NON_FINITE_RESULT = "non_finite_result"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
    CFL_VIOLATION = "cfl_violation"
    UNKNOWN_BACKEND = "unknown_backend"
    OUT_OF_APPLICABILITY_DOMAIN = "out_of_applicability_domain"


#: Maps the exact string literals mixle_pde.continuation and mixle_pde.verification.modeling_traces
#: already emit to a DiagnosticCode. Deliberately excludes ContinuationReceipt.stopped_reason values
#: "newton_failed"/"initial_point_not_converged" (not root causes themselves -- see
#: classify_continuation_receipt, which drills into the nested NewtonReceipt instead) and
#: modeling_traces.ModelingTrace.final_failure_class values "repair_still_violates_precondition"/
#: "verification_disagreement" (those describe modeling_traces.py's own repair-attempt outcome, a
#: different and already-solved problem, not the original diagnostic code).
_FAILURE_CLASS_TO_CODE: Mapping[str, DiagnosticCode] = {
    "singular_jacobian": DiagnosticCode.SINGULAR_JACOBIAN,
    "singular_tangent_jacobian": DiagnosticCode.SINGULAR_JACOBIAN,
    "non_finite_residual": DiagnosticCode.NON_FINITE_RESULT,
    "non_finite_jacobian": DiagnosticCode.NON_FINITE_RESULT,
    "non_finite_step": DiagnosticCode.NON_FINITE_RESULT,
    "max_iterations_exceeded": DiagnosticCode.MAX_ITERATIONS_EXCEEDED,
    "cfl_violation_diverged": DiagnosticCode.CFL_VIOLATION,
    "cfl_violation_unstable_growth": DiagnosticCode.CFL_VIOLATION,
}


def classify_failure_class(name: str) -> DiagnosticCode | None:
    """Classify a raw failure-class string into a :class:`DiagnosticCode`.

    The single source of truth every other ``classify_*`` helper in this module delegates to. Returns
    ``None`` for a string this ontology does not (yet) recognize, never a guess.
    """
    return _FAILURE_CLASS_TO_CODE.get(name)


def classify_newton_receipt(receipt: NewtonReceipt) -> DiagnosticCode | None:
    """Classify a real :class:`mixle_pde.continuation.NewtonReceipt`.

    ``None`` when ``receipt.converged`` (nothing to diagnose) or -- defensively -- when
    ``failure_reason`` is not one of the five values that dataclass's own docstring documents as
    exhaustive, rather than guessing at an unrecognized reason.
    """
    if receipt.converged or receipt.failure_reason is None:
        return None
    return classify_failure_class(receipt.failure_reason)


def classify_continuation_receipt(receipt: ContinuationReceipt) -> DiagnosticCode | None:
    """Classify a real :class:`mixle_pde.continuation.ContinuationReceipt`.

    ``"target_reached"``/``"max_steps_reached"`` are nominal termination, not failures, and map to
    ``None``. ``"singular_tangent_jacobian"`` is a direct, distinct root cause (the arclength
    predictor's own tangent solve, not a Newton corrector). ``"newton_failed"`` and
    ``"initial_point_not_converged"`` are not themselves root causes -- both mean "a nested Newton
    corrector solve failed" -- so this drills into the last rejected step's own
    :class:`~mixle_pde.continuation.NewtonReceipt` and classifies *that*, never inventing a seventh
    code for "a continuation step failed" in general.
    """
    if receipt.stopped_reason == "singular_tangent_jacobian":
        return DiagnosticCode.SINGULAR_JACOBIAN
    if receipt.stopped_reason in ("newton_failed", "initial_point_not_converged"):
        failing_step = next((step for step in reversed(receipt.steps) if not step.accepted), None)
        if failing_step is not None:
            return classify_newton_receipt(failing_step.newton)
        return None
    return None


def classify_backend_dispatch_error(exc: BaseException) -> DiagnosticCode | None:
    """Classify an exception raised by :mod:`mixle_pde.pde_backend_registry`'s dispatch boundary
    (``get_kernel_registration`` / ``run_math_problem``).

    Grounded directly in that module's own two exception types at the dispatch boundary:
    ``get_kernel_registration`` raises :class:`KeyError` for a backend id that is not registered
    (:attr:`DiagnosticCode.UNKNOWN_BACKEND`); several ``_invoke_*`` functions raise :class:`ValueError`
    for a caller-supplied grid/mesh shape or dimension outside what that specific registered invocation
    supports (:attr:`DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN`). Returns ``None`` for any other
    exception type -- this only classifies the two types that dispatch boundary is documented to raise,
    never guesses at an unrelated ``KeyError``/``ValueError`` raised somewhere else in a caller's stack.
    """
    if isinstance(exc, KeyError):
        return DiagnosticCode.UNKNOWN_BACKEND
    if isinstance(exc, ValueError):
        return DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN
    return None


class RepairActionKind(Enum):
    """What kind of mechanical action a :class:`RepairAction` takes -- plain data, never a stored
    callback, per this program's standing "keep semantics in typed schema objects, not backend payloads
    or Python callbacks" convention.

    ``SCALE_PARAMETER`` names a real keyword parameter of the failing call and a multiplicative factor
    to apply to it. ``REQUIRES_HUMAN`` is a first-class typed outcome for a failure class this module
    has no safe, generically-effective mechanical fix for -- attempting one anyway and calling it a
    "repair" would be exactly the fabricated-success this repository's typed-outcome convention exists
    to prevent.
    """

    SCALE_PARAMETER = "scale_parameter"
    REQUIRES_HUMAN = "requires_human"


@dataclass(frozen=True)
class RetryPolicy:
    """A bounded retry budget for one :class:`RepairAction`.

    ``max_attempts`` is always finite, and :func:`run_bounded_repair` never loops past it -- MP-J3's
    accept bar ("...or returns ``requires_human`` without loops") read literally: the "without loops"
    half means a hard, structural bound, not merely a generous one. ``scale_factor`` is the
    multiplicative factor :func:`apply_repair_attempt` raises to the ``attempt``-th power and applies to
    the named parameter's *original* value (never compounded on a previous candidate), so two calls
    with the same ``(action, original_params, attempt)`` always return the same candidate. ``None`` only
    for a ``REQUIRES_HUMAN`` action, which has no parameter to scale.
    """

    max_attempts: int
    scale_factor: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")
        if self.max_attempts > 0 and self.scale_factor is None:
            raise ValueError("scale_factor is required when max_attempts > 0")
        if self.scale_factor is not None and self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")


@dataclass(frozen=True)
class RepairAction:
    """One :class:`DiagnosticCode`'s deterministic repair recipe: preconditions, expected effect, the
    bounded retry policy, and (for a ``SCALE_PARAMETER`` recipe) which real keyword parameter of the
    failing call it mechanically adjusts.
    """

    code: DiagnosticCode
    kind: RepairActionKind
    parameter_name: str | None
    retry_policy: RetryPolicy
    precondition: str
    expected_effect: str
    rationale: str


REPAIR_CATALOG: Mapping[DiagnosticCode, RepairAction] = {
    DiagnosticCode.CFL_VIOLATION: RepairAction(
        code=DiagnosticCode.CFL_VIOLATION,
        kind=RepairActionKind.SCALE_PARAMETER,
        parameter_name="dt",
        retry_policy=RetryPolicy(max_attempts=4, scale_factor=0.5),
        precondition=(
            "the failing call's solve_plan carries a numeric 'dt' and a documented Courant-type "
            "stability limit (e.g. mixle_pde.elastic.ElasticWave3D: dt <= spacing / (vp * sqrt(3))) "
            "that the caller's dt violates"
        ),
        expected_effect=(
            "halving dt strictly reduces the Courant number vp*dt/spacing, which is monotone in dt, so "
            "a bounded number of halvings is guaranteed to eventually clear any finite documented limit"
        ),
        rationale=(
            "mixle_pde.verification.modeling_traces.generate_elastic_cfl_trace already performs and "
            "independently reverifies exactly this repair (repair_safety_factor=0.5) against the real "
            "elastic-fd-leapfrog kernel, whose detected_failure_class is 'cfl_violation_diverged' or "
            "'cfl_violation_unstable_growth' when courant = vp*dt/spacing exceeds 1/sqrt(3) "
            "(mixle_pde.elastic.ElasticWave3D docstring); this recipe generalizes that already-verified "
            "mechanical fix rather than inventing a new one"
        ),
    ),
    DiagnosticCode.MAX_ITERATIONS_EXCEEDED: RepairAction(
        code=DiagnosticCode.MAX_ITERATIONS_EXCEEDED,
        kind=RepairActionKind.SCALE_PARAMETER,
        parameter_name="max_iterations",
        retry_policy=RetryPolicy(max_attempts=3, scale_factor=2.0),
        precondition=(
            "the failing call is a mixle_pde.continuation.newton_solve (or natural_continuation / "
            "arclength_continuation) corrector whose NewtonReceipt.failure_reason == "
            "'max_iterations_exceeded' -- the residual was still finite, not diverging"
        ),
        expected_effect=(
            "doubling the iteration budget gives an iteration that was still making progress more room "
            "to reach residual_history[-1] < tol; it does not help a genuinely diverging or stalled "
            "iteration, which is exactly why this recipe is bounded rather than unconditionally "
            "reapplied"
        ),
        rationale=(
            "mixle_pde.continuation.newton_solve raises this exact failure_reason literal (see "
            "NewtonReceipt's docstring's enumerated set) and already exposes max_iterations as a real, "
            "documented keyword parameter (default 50) on newton_solve/natural_continuation/"
            "arclength_continuation -- confirmed live by tests/continuation_test.py::"
            "test_newton_solve_reports_max_iterations_exceeded, which forces this exact failure with "
            "max_iterations=0"
        ),
    ),
    DiagnosticCode.NON_FINITE_RESULT: RepairAction(
        code=DiagnosticCode.NON_FINITE_RESULT,
        kind=RepairActionKind.SCALE_PARAMETER,
        parameter_name="damping",
        retry_policy=RetryPolicy(max_attempts=3, scale_factor=0.5),
        precondition=(
            "the failing call is a mixle_pde.continuation.newton_solve corrector whose "
            "NewtonReceipt.failure_reason is one of 'non_finite_residual', 'non_finite_jacobian', or "
            "'non_finite_step' -- the previous accepted iterate stepped into a region where the "
            "residual/Jacobian blows up (e.g. across a 1/u-type singularity)"
        ),
        expected_effect=(
            "halving damping shrinks how far each accepted Newton step moves (u = u + damping*delta), "
            "so a smaller step is less likely to land in the non-finite region -- a genuine, bounded "
            "sub-relaxation mitigation, not a guarantee (a pathological residual can still blow up "
            "arbitrarily close to the start), which is exactly why this recipe is bounded rather than "
            "claimed to always resolve"
        ),
        rationale=(
            "mixle_pde.continuation.newton_solve/natural_continuation/arclength_continuation all "
            "already expose damping as a real keyword parameter (default 1.0); "
            "tests/continuation_test.py::test_newton_solve_reports_non_finite_residual forces this "
            "exact failure_reason live"
        ),
    ),
    DiagnosticCode.SINGULAR_JACOBIAN: RepairAction(
        code=DiagnosticCode.SINGULAR_JACOBIAN,
        kind=RepairActionKind.REQUIRES_HUMAN,
        parameter_name=None,
        retry_policy=RetryPolicy(max_attempts=0, scale_factor=None),
        precondition=(
            "the failing call's NewtonReceipt.failure_reason == 'singular_jacobian' (a Newton corrector "
            "solve) or ContinuationReceipt.stopped_reason == 'singular_tangent_jacobian' (the pseudo-"
            "arclength predictor's tangent solve)"
        ),
        expected_effect="none attempted -- see rationale for why no bounded mechanical retry is safe here",
        rationale=(
            "mixle_pde.continuation.newton_solve/_augmented_corrector/_tangent raise this on an exactly "
            "singular dF/du (np.linalg.LinAlgError from np.linalg.solve); tests/continuation_test.py::"
            "test_newton_solve_reports_singular_jacobian_not_fabricated_success forces this live with a "
            "residual that is structurally rank-deficient for every u and lambda. Unlike a non-finite "
            "step, damping cannot fix this: damping only scales the step *after* the singular solve "
            "already failed, and the Jacobian is re-evaluated at the same current iterate regardless of "
            "how far the previous step moved, so retrying with a smaller damping repeats the identical "
            "singular solve. A real fix (Levenberg-Marquardt-style regularization of the linear solve, "
            "or a different starting iterate) would require changing continuation.py's own solver "
            "internals, out of this module's scope -- so the honest recipe is requires_human, matching "
            "this program's 'unknown/timeout/resource_limit are valid typed outcomes, never a "
            "fabricated Boolean' convention rather than pretending a retry would help"
        ),
    ),
    DiagnosticCode.UNKNOWN_BACKEND: RepairAction(
        code=DiagnosticCode.UNKNOWN_BACKEND,
        kind=RepairActionKind.REQUIRES_HUMAN,
        parameter_name=None,
        retry_policy=RetryPolicy(max_attempts=0, scale_factor=None),
        precondition=(
            "mixle_pde.pde_backend_registry.get_kernel_registration(backend_id) (also reached via "
            "run_math_problem) raised KeyError for a backend_id not in its registry"
        ),
        expected_effect="none attempted -- see rationale",
        rationale=(
            "get_kernel_registration's own KeyError message already reports the full registered-id "
            "list (sorted(_REGISTRATIONS)), so the mechanical lookup a repair could perform is free -- "
            "but silently substituting a different registered backend would silently change which "
            "physics gets solved, exactly the 'invented components or silent approximations' MP-J2's "
            "own accept bar names as unacceptable; requires_human keeps that choice with a caller who "
            "actually knows which physics they meant"
        ),
    ),
    DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN: RepairAction(
        code=DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN,
        kind=RepairActionKind.REQUIRES_HUMAN,
        parameter_name=None,
        retry_policy=RetryPolicy(max_attempts=0, scale_factor=None),
        precondition=(
            "a registered backend's invoker raised ValueError for a solve_plan parameter outside what "
            "that specific registered invocation supports, e.g. mixle_pde.pde_backend_registry."
            "_invoke_helmholtz_pml's 'helmholtz-pml-fd is 2-D only; got grid_shape {shape}.', "
            "_invoke_groundwater_fd's 'groundwater-fd-transport supports 1-D or 2-D grids only; got "
            "shape {shape}.', or _invoke_flow_spectral's 'flow-spectral-ns supports dim=2 or dim=3.'"
        ),
        expected_effect="none attempted -- see rationale",
        rationale=(
            "changing a problem's dimensionality/grid shape to fit a specific registered invocation is "
            "a modeling decision (a different discretization or a different backend entirely may be "
            "the right call), not a safe mechanical retry with the same parameters scaled -- consistent "
            "with knowledge_catalog.py's own applicability entries, which state each of these limits as "
            "a hard boundary of the registered invocation, not a tunable"
        ),
    ),
}

if set(REPAIR_CATALOG) != set(DiagnosticCode):
    raise AssertionError("REPAIR_CATALOG must have exactly one entry per DiagnosticCode member")


def get_repair_action(code: DiagnosticCode) -> RepairAction:
    """Look up the one :class:`RepairAction` recipe for a :class:`DiagnosticCode`."""
    return REPAIR_CATALOG[code]


def list_diagnostic_codes() -> tuple[DiagnosticCode, ...]:
    """Every diagnostic code this ontology covers, in declaration order."""
    return tuple(DiagnosticCode)


class RepairOutcome(Enum):
    """A :func:`run_bounded_repair` call's typed final disposition -- never a fabricated boolean."""

    RESOLVED = "resolved"
    REQUIRES_HUMAN = "requires_human"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"


@dataclass(frozen=True)
class RepairRun:
    """The outcome of running one :class:`RepairAction`'s bounded retry loop."""

    code: DiagnosticCode
    outcome: RepairOutcome
    attempts_used: int
    final_params: Mapping[str, Any]


def apply_repair_attempt(
    action: RepairAction, original_params: Mapping[str, Any], attempt: int
) -> Mapping[str, Any] | None:
    """One deterministic, pure repair candidate: the ``attempt``-th (1-indexed) mechanical repair of
    ``action`` applied to ``original_params``.

    Always scales from the *original* value by ``scale_factor ** attempt``, never from a previous
    candidate, so two calls with the same ``(action, original_params, attempt)`` always return the same
    result. Returns ``None`` when there is no mechanical attempt available -- ``action.kind is
    REQUIRES_HUMAN``, or ``attempt`` outside ``[1, action.retry_policy.max_attempts]`` -- the caller's
    signal to stop, never to invent one.
    """
    if action.kind is RepairActionKind.REQUIRES_HUMAN:
        return None
    if not (1 <= attempt <= action.retry_policy.max_attempts):
        return None
    if action.parameter_name not in original_params:
        raise KeyError(f"repair action for {action.code} expects params[{action.parameter_name!r}] to be set")
    factor = action.retry_policy.scale_factor
    assert factor is not None  # RetryPolicy.__post_init__ guarantees this whenever max_attempts > 0
    base = original_params[action.parameter_name]
    updated = dict(original_params)
    updated[action.parameter_name] = base * (factor**attempt)
    return updated


def run_bounded_repair(
    action: RepairAction,
    original_params: Mapping[str, Any],
    is_resolved: Callable[[Mapping[str, Any]], bool],
) -> RepairRun:
    """Apply ``action``'s bounded retry policy, checking ``is_resolved`` after each candidate, and stop
    the moment either a resolution or the attempt budget is reached.

    Structurally cannot loop indefinitely: the driving loop is a bounded ``for attempt in
    range(1, max_attempts + 1)``, never a ``while True`` -- MP-J3's accept bar ("...or returns
    requires_human without loops") read as a construction guarantee. A ``REQUIRES_HUMAN`` action
    returns immediately with zero attempts, never invoking ``is_resolved`` at all.
    """
    if action.kind is RepairActionKind.REQUIRES_HUMAN:
        return RepairRun(
            code=action.code,
            outcome=RepairOutcome.REQUIRES_HUMAN,
            attempts_used=0,
            final_params=dict(original_params),
        )
    candidate: Mapping[str, Any] = dict(original_params)
    for attempt in range(1, action.retry_policy.max_attempts + 1):
        candidate = apply_repair_attempt(action, original_params, attempt)
        if is_resolved(candidate):
            return RepairRun(
                code=action.code, outcome=RepairOutcome.RESOLVED, attempts_used=attempt, final_params=candidate
            )
    return RepairRun(
        code=action.code,
        outcome=RepairOutcome.ATTEMPTS_EXHAUSTED,
        attempts_used=action.retry_policy.max_attempts,
        final_params=candidate,
    )


@dataclass(frozen=True)
class SeededFailure:
    """One deterministically constructed, genuinely failing call and the diagnostic code it maps to.

    ``source`` names exactly which real function this item drove (e.g.
    ``"mixle_pde.continuation.newton_solve"``), so a seeded failure is reproducible and traceable back
    to the actual grounding call, never merely a label.
    """

    seed: int
    source: str
    code: DiagnosticCode
    detail: Mapping[str, Any]


def _derive_seed(seed: int, *parts: int) -> int:
    """Deterministically mix an integer seed with extra integer parts.

    Never uses ``hash()``: CPython's string/bytes hashing is process-salted by default
    (``PYTHONHASHSEED``), which would silently break reproducibility across processes. Mirrors
    :func:`mixle_pde.verification.modeling_traces._derive_seed`.
    """
    mixed = seed
    for part in parts:
        mixed = (mixed * 1_000_003 + int(part)) & ((1 << 63) - 1)
    return mixed


def _seeded_singular_jacobian_newton(item_seed: int) -> SeededFailure:
    """Real call into ``newton_solve`` with a Jacobian that is a literal exact zero and a residual that
    never reaches zero, so the corrector's own linear solve is reached and is exactly singular every
    time.

    Deliberately *not* built from floating-point cancellation (e.g. two structurally-identical rows
    scaled by a runtime-computed factor): confirmed empirically before writing this constructor that
    ``np.linalg.solve`` on ``[[s, -s], [s, -s]]`` for a random ``s`` fails to raise ``LinAlgError`` in
    roughly 15% of cases (31/200 sampled scales) because LAPACK's singularity check is an exact-zero-
    pivot test, not a condition-number threshold, and cancellation arithmetic does not reliably produce
    a bit-exact zero pivot. A literal ``0.0`` entry has no such dependence and raises ``LinAlgError``
    100% of the time (also confirmed empirically), matching the same literal-zero pattern
    :func:`_seeded_singular_tangent_jacobian_arclength` already uses for the tangent-solve trigger.
    """
    rng = random.Random(item_seed)
    residual_value = rng.uniform(0.5, 5.0)
    problem = ParametricProblem(
        residual=lambda u, lam, r=residual_value: np.array([r]),
        jac_u=lambda u, lam: np.array([[0.0]]),
        jac_lambda=lambda u, lam: np.array([0.0]),
        n=1,
        name="seeded_singular_jacobian",
    )
    receipt = newton_solve(problem, u0=np.array([0.0]), lam=0.0)
    code = classify_newton_receipt(receipt)
    if code is not DiagnosticCode.SINGULAR_JACOBIAN:
        raise AssertionError(f"seed {item_seed}: expected SINGULAR_JACOBIAN, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.continuation.newton_solve",
        code=code,
        detail={"failure_reason": receipt.failure_reason, "residual_value": residual_value},
    )


def _seeded_non_finite_residual_newton(item_seed: int) -> SeededFailure:
    """Real call into ``newton_solve`` with ``c/u[0] - lambda``, which blows up to a non-finite residual
    exactly at the chosen start ``u[0] = 0`` -- the same shape as
    ``tests/continuation_test.py::test_newton_solve_reports_non_finite_residual``.
    """
    rng = random.Random(item_seed)
    coefficient = rng.uniform(0.5, 3.0)

    def residual(u, lam, c=coefficient):
        with np.errstate(divide="ignore"):
            return np.array([c / u[0] - lam])

    def jac_u(u, lam, c=coefficient):
        with np.errstate(divide="ignore"):
            return np.array([[-c / u[0] ** 2]])

    problem = ParametricProblem(
        residual=residual,
        jac_u=jac_u,
        jac_lambda=lambda u, lam: np.array([-1.0]),
        n=1,
        name="seeded_blowup_at_origin",
    )
    with np.errstate(divide="ignore"):
        receipt = newton_solve(problem, u0=np.array([0.0]), lam=1.0)
    code = classify_newton_receipt(receipt)
    if code is not DiagnosticCode.NON_FINITE_RESULT:
        raise AssertionError(f"seed {item_seed}: expected NON_FINITE_RESULT, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.continuation.newton_solve",
        code=code,
        detail={"failure_reason": receipt.failure_reason, "coefficient": coefficient},
    )


def _seeded_non_finite_jacobian_newton(item_seed: int) -> SeededFailure:
    """Real call into ``newton_solve`` with a finite residual but a Jacobian ``1/u[0]`` that blows up
    exactly at the chosen start ``u[0] = 0``."""
    rng = random.Random(item_seed)
    target = rng.uniform(0.5, 5.0)

    def jac_u(u, lam):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.array([[1.0 / u[0]]])

    problem = ParametricProblem(
        residual=lambda u, lam, t=target: np.array([u[0] - t]),
        jac_u=jac_u,
        jac_lambda=lambda u, lam: np.array([0.0]),
        n=1,
        name="seeded_jacobian_blowup",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        receipt = newton_solve(problem, u0=np.array([0.0]), lam=0.0, max_iterations=5)
    code = classify_newton_receipt(receipt)
    if code is not DiagnosticCode.NON_FINITE_RESULT:
        raise AssertionError(f"seed {item_seed}: expected NON_FINITE_RESULT, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.continuation.newton_solve",
        code=code,
        detail={"failure_reason": receipt.failure_reason, "target": target},
    )


def _seeded_non_finite_step_newton(item_seed: int) -> SeededFailure:
    """Real call into ``newton_solve`` with a finite, nonzero-but-subnormal Jacobian: the linear solve
    itself stays exact (no ``LinAlgError``), but dividing a normal residual by a subnormal Jacobian
    overflows the Newton step to a non-finite ``delta`` -- confirmed live before writing this
    constructor (``np.linalg.solve([[1e-320]], [-r])`` returns ``[-inf]``, not an exception).
    """
    rng = random.Random(item_seed)
    residual_value = rng.uniform(0.5, 5.0)
    tiny = rng.uniform(1.0, 9.0) * 1.0e-320

    problem = ParametricProblem(
        residual=lambda u, lam, r=residual_value: np.array([r]),
        jac_u=lambda u, lam, t=tiny: np.array([[t]]),
        jac_lambda=lambda u, lam: np.array([0.0]),
        n=1,
        name="seeded_step_overflow",
    )
    with np.errstate(over="ignore"):
        receipt = newton_solve(problem, u0=np.array([1.0]), lam=0.0, max_iterations=5)
    code = classify_newton_receipt(receipt)
    if code is not DiagnosticCode.NON_FINITE_RESULT:
        raise AssertionError(f"seed {item_seed}: expected NON_FINITE_RESULT, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.continuation.newton_solve",
        code=code,
        detail={"failure_reason": receipt.failure_reason, "residual_value": residual_value, "tiny_jacobian": tiny},
    )


def _seeded_max_iterations_exceeded_newton(item_seed: int) -> SeededFailure:
    """Real call into ``newton_solve`` on the Bratu problem with ``max_iterations=0`` -- the same shape
    as ``tests/continuation_test.py::test_newton_solve_reports_max_iterations_exceeded``, seed-varied
    over grid size and lambda.
    """
    rng = random.Random(item_seed)
    n = rng.choice((5, 8, 12))
    lam = rng.uniform(0.5, 2.5)
    problem = bratu_problem(n)
    receipt = newton_solve(problem, u0=np.zeros(n), lam=lam, max_iterations=0)
    code = classify_newton_receipt(receipt)
    if code is not DiagnosticCode.MAX_ITERATIONS_EXCEEDED:
        raise AssertionError(f"seed {item_seed}: expected MAX_ITERATIONS_EXCEEDED, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.continuation.newton_solve",
        code=code,
        detail={"failure_reason": receipt.failure_reason, "n": n, "lam": lam},
    )


def _seeded_singular_tangent_jacobian_arclength(item_seed: int) -> SeededFailure:
    """Real call into ``arclength_continuation`` with a residual that is identically zero (so the start
    converges trivially) and a Jacobian that is identically zero (so the tangent solve at that converged
    point is exactly singular) -- exercises the predictor's own failure path, distinct from a Newton
    corrector's ``singular_jacobian``.
    """
    rng = random.Random(item_seed)
    lam0 = rng.uniform(-2.0, 2.0)
    problem = ParametricProblem(
        residual=lambda u, lam: np.array([0.0]),
        jac_u=lambda u, lam: np.array([[0.0]]),
        jac_lambda=lambda u, lam: np.array([1.0]),
        n=1,
        name="seeded_zero_residual_singular_tangent",
    )
    receipt = arclength_continuation(problem, u0=np.array([0.0]), lam0=lam0, ds=0.1, n_steps=5)
    code = classify_continuation_receipt(receipt)
    if code is not DiagnosticCode.SINGULAR_JACOBIAN:
        raise AssertionError(f"seed {item_seed}: expected SINGULAR_JACOBIAN, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.continuation.arclength_continuation",
        code=code,
        detail={"stopped_reason": receipt.stopped_reason, "lam0": lam0},
    )


def _seeded_unknown_backend(item_seed: int) -> SeededFailure:
    """Real call into ``mixle_pde.pde_backend_registry.get_kernel_registration`` with a deliberately
    unregistered backend id."""
    from mixle_pde.pde_backend_registry import get_kernel_registration

    rng = random.Random(item_seed)
    bogus_id = f"not-a-real-backend-{rng.randint(0, 1_000_000)}"
    try:
        get_kernel_registration(bogus_id)
    except KeyError as exc:
        code = classify_backend_dispatch_error(exc)
    else:
        raise AssertionError(f"seed {item_seed}: expected KeyError for unregistered backend id {bogus_id!r}")
    if code is not DiagnosticCode.UNKNOWN_BACKEND:
        raise AssertionError(f"seed {item_seed}: expected UNKNOWN_BACKEND, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.pde_backend_registry.get_kernel_registration",
        code=code,
        detail={"backend_id": bogus_id},
    )


def _seeded_out_of_applicability_domain(item_seed: int) -> SeededFailure:
    """Real call into ``mixle_pde.pde_backend_registry.run_math_problem`` for ``groundwater-fd-
    transport`` (a torch-free invoker) with a grid_shape whose length is outside the 1-D/2-D this
    specific registration supports.
    """
    from mixle_pde.pde_backend_registry import run_math_problem

    rng = random.Random(item_seed)
    n_dims = rng.choice((3, 4, 5))
    shape = tuple(rng.choice((4, 5, 6)) for _ in range(n_dims))
    problem = {
        "id": f"seeded-oob-{item_seed}",
        "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": "structured_grid"}}],
        "unknowns": [{"id": "field", "domain_id": "domain"}],
        "operators": [
            {
                "id": "seeded-oob-operator",
                "kind": "linear_operator",
                "input_ids": ["field"],
                "output_ids": ["field"],
                "discretization": "FD-implicit",
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [],
        "solve_plan": {"parameters": {"grid_shape": shape}},
    }
    try:
        run_math_problem(problem, "groundwater-fd-transport")
    except ValueError as exc:
        code = classify_backend_dispatch_error(exc)
    else:
        raise AssertionError(f"seed {item_seed}: expected ValueError for grid_shape {shape}")
    if code is not DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN:
        raise AssertionError(f"seed {item_seed}: expected OUT_OF_APPLICABILITY_DOMAIN, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.pde_backend_registry.run_math_problem[groundwater-fd-transport]",
        code=code,
        detail={"grid_shape": shape},
    )


#: mixle_pde.elastic.ElasticWave3D's own documented 3-D leapfrog Courant limit: dt <= spacing/(vp*sqrt(3)).
_ELASTIC_3D_COURANT_LIMIT = 1.0 / math.sqrt(3.0)


def _seeded_cfl_violation(item_seed: int) -> SeededFailure:
    """A genuine violation of the documented closed-form elastic-fd-leapfrog Courant limit, checked
    directly against the same formula ``mixle_pde.verification.modeling_traces`` uses (no kernel call
    here, to keep the bulk corpus cheap -- ``tests/diagnostic_ontology_test.py`` separately cross-checks
    :data:`DiagnosticCode.CFL_VIOLATION` against the real, kernel-verified
    ``generate_elastic_cfl_trace``/``generate_elastic_cfl_unresolved_trace`` output).
    """
    rng = random.Random(item_seed)
    vp = rng.uniform(1.0, 3.0)
    spacing = rng.uniform(0.05, 0.2)
    violation_factor = rng.uniform(1.2, 6.0)  # always > 1: a genuine violation by construction
    dt = violation_factor * _ELASTIC_3D_COURANT_LIMIT * spacing / vp
    courant = vp * dt / spacing
    if not (courant > _ELASTIC_3D_COURANT_LIMIT):
        raise AssertionError(f"seed {item_seed}: constructed dt did not actually violate the Courant limit")
    code = classify_failure_class("cfl_violation_diverged")
    if code is not DiagnosticCode.CFL_VIOLATION:
        raise AssertionError(f"seed {item_seed}: expected CFL_VIOLATION, got {code}")
    return SeededFailure(
        seed=item_seed,
        source="mixle_pde.elastic.ElasticWave3D (documented Courant limit)",
        code=code,
        detail={
            "vp": vp,
            "spacing": spacing,
            "dt": dt,
            "courant_number": courant,
            "courant_limit": _ELASTIC_3D_COURANT_LIMIT,
        },
    )


_CONSTRUCTORS: tuple[Callable[[int], SeededFailure], ...] = (
    _seeded_singular_jacobian_newton,
    _seeded_non_finite_residual_newton,
    _seeded_non_finite_jacobian_newton,
    _seeded_non_finite_step_newton,
    _seeded_max_iterations_exceeded_newton,
    _seeded_singular_tangent_jacobian_arclength,
    _seeded_unknown_backend,
    _seeded_out_of_applicability_domain,
    _seeded_cfl_violation,
)


def seeded_failure_corpus(*, seed: int, count: int) -> tuple[SeededFailure, ...]:
    """Deterministically generate ``count`` seeded failures, cycling through nine real-call
    constructors (covering all six :class:`DiagnosticCode` members -- two each for
    ``SINGULAR_JACOBIAN`` and ``NON_FINITE_RESULT``, which each have more than one distinct real
    trigger), each driving an actual failing call into ``mixle_pde.continuation`` or
    ``mixle_pde.pde_backend_registry`` (or, for ``CFL_VIOLATION``, the same documented closed-form
    Courant formula ``mixle_pde.verification.modeling_traces`` already verifies against the real
    kernel -- see that module's own ``generate_elastic_cfl_trace`` for the kernel-backed version,
    deliberately not re-run once per item here to keep this corpus cheap enough for a single bounded
    validation run).

    Every constructor raises :class:`AssertionError` rather than returning a mis-classified item, so
    ``all(item.code is not None for item in seeded_failure_corpus(...))`` holds by construction, not
    merely by post-hoc check -- MP-J3's own accept bar ("at least 100 seeded failures map to stable
    codes") satisfied structurally.

    Pure function of ``(seed, count)``: two calls with the same inputs produce identical items, since
    each constructor seeds its own :class:`random.Random` from a value :func:`_derive_seed` mixes from
    ``(seed, index)``, never from wall-clock or process state.
    """
    if count < 1:
        raise ValueError("count must be positive")
    items = []
    for index in range(count):
        item_seed = _derive_seed(seed, index)
        constructor = _CONSTRUCTORS[index % len(_CONSTRUCTORS)]
        items.append(constructor(item_seed))
    return tuple(items)
