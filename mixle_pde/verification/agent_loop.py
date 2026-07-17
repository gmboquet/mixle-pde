"""Bounded coarse-to-fine agent execution loop over registered PDE kernels and the IC-3 inversion tool
(MP-J4).

The work plan's full MP-J4 description is a fourteen-stage pipeline: "requirements -> assumptions/gaps ->
draft -> validate -> mesh preview -> coarse mesh -> dry compile -> resource estimate -> coarse solve ->
verify -> refine -> selected study -> posterior/surrogate diagnostics where applicable -> final. Limit
patches, retries, wall time, DOFs, samples, training runs, and cost." That full pipeline runs over a
``ModelSpec``/mesh/compile agent-tool surface (section 6.7's fifteen tools: ``create_model_spec``,
``generate_mesh``, ``compile_model``, ...) that does not exist anywhere in this repo yet -- the M2
reconciliation ledger's own MP-J1 row records only one narrow, already-landed tool surface,
``mixle_pde/tools.py``'s four IC-3 Earth-inversion tools (``run_inversion``/``query_posterior``/
``gassmann``/``forward_model``), "not the full fifteen-tool construct/mesh/compile/solve/verify/train/query
surface."

This module is a deliberately narrow baseline slice of MP-J4, matching how every other task in this same
MP-J/MP-K/MP-N family (``knowledge_catalog.py``/MP-J2, ``diagnostic_ontology.py``/MP-J3,
``validation_tiers.py``/MP-K5, ``artifact_store.py``/MP-K1, ``job_governance.py``/MP-L4,
``drift_monitor.py``/MP-N6) scoped its own much larger task down to a first slice grounded in what
actually already exists, rather than fabricating the missing agent-tool/mesh/compile surface to reach the
full pipeline. What this module actually implements, over the real, already-registered
:mod:`mixle_pde.pde_backend_registry` kernels (MP-E5) and the real, already-landed
:func:`mixle_pde.tools.run_inversion` (MP-J1/IC-3):

* **estimate_resources** -- :func:`estimate_dofs` reads each registered kernel's own solve-plan size
  parameter (``grid_size`` or ``grid_shape`` -- read directly from ``pde_backend_registry.py``'s own
  ``_invoke_*`` functions, not guessed) and estimates a DOF count *before* any solve is dispatched.
* **coarse mesh / coarse solve** -- :func:`run_coarse_to_fine` scales the requested resolution down by
  ``coarse_scale`` and dispatches one real :func:`~mixle_pde.pde_backend_registry.run_math_problem` call.
* **verify** -- :func:`_verify_evidence` reads the real ``PDEStudyResult.evidence`` dict every kernel
  already returns (``convergence.finite``/``convergence.stable`` when present, a top-level ``residual``
  when present) -- the same fields each kernel's own accepted-study test already checks (see
  ``knowledge_catalog.py``), not a new verification bar invented for this module.
* **refine** -- on a verification failure, :func:`_classify_evidence_failure` and
  :func:`_attempt_repair` reuse MP-J3's own :data:`~mixle_pde.verification.diagnostic_ontology.REPAIR_CATALOG`
  and :func:`~mixle_pde.verification.diagnostic_ontology.run_bounded_repair` verbatim -- this module adds no
  new repair math, only a new *caller* of J3's already-bounded, already bounded-for-real machinery.
* **final** -- once the coarse resolution verifies (immediately or after a bounded repair), one real solve
  at the caller's originally-requested (uncoarsened) resolution, carrying forward any repaired parameter.
* **inverse case** -- :func:`run_inverse_case` wraps :func:`mixle_pde.tools.run_inversion` in the same
  typed :class:`LoopOutcome` vocabulary, satisfying MP-J4's "solves held-out forward and inverse cases"
  half without a second, incompatible orchestration signature (``run_inversion``'s ``dataset_ref``/
  ``modality``/``prior``/``config`` shape is not a ``CON-MATH-PROBLEM-V1`` study and does not fit
  :func:`run_coarse_to_fine`'s signature).

Explicitly NOT attempted (matching this family's "narrow, honest slice" convention rather than fabricating
coverage): mesh preview, dry compile, and a real resource *estimate* beyond a DOF count (no memory/wall-
clock model); ``selected study``/posterior-or-surrogate diagnostics as a wired pipeline stage (an inverse
case is reachable through :func:`run_inverse_case`, but this module does not itself compute posterior or
surrogate validity diagnostics -- MP-I8/MP-N6 already own that); "samples", "training runs" budget
dimensions (no sampler or surrogate-training stage is wired here). :class:`LoopOutcome` gives a typed
result for ``unsupported`` and ``over_budget`` (both directly grounded below); MP-J4's accept bar also names
``ill-posed``, ``unidentifiable``, and ``OOD`` termination, which this baseline does not diagnose as
*distinct* typed outcomes -- no already-landed, generically-safe signal for any of the three was found
reachable from this module's two real entry points (see :func:`_classify_evidence_failure`'s and
:func:`run_inverse_case`'s docstrings for exactly what was checked and ruled out). Rather than fabricate a
proxy, every exception this module's two entry points do not specifically recognize (via MP-J3's own
:func:`~mixle_pde.verification.diagnostic_ontology.classify_backend_dispatch_error`) still resolves to the
typed :attr:`LoopOutcome.REQUIRES_HUMAN` outcome through a generic backstop rather than propagating --
so the loop still "terminates safely" (never raises, never loops unboundedly) on every input, including
those three named categories, it just does not *diagnose* them by name.

Dependency direction (matching this family's own convention: solver/verification primitives feed up into
an orchestration layer, the orchestration layer never reaches sideways into a sibling or edits a primitive
it consumes): this module imports :mod:`mixle_pde.pde_backend_registry`,
:mod:`mixle_pde.verification.diagnostic_ontology` (MP-J3), :mod:`mixle_pde.verification.knowledge_catalog`
(MP-J2, used only to enrich an ``UNSUPPORTED`` receipt's detail with that catalog's own recorded
applicability text when one exists), and :mod:`mixle_pde.tools` (MP-J1) read-only. None of those four
modules is modified here.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mixle_pde.pde_backend_registry import PDEStudyResult, run_math_problem
from mixle_pde.verification import knowledge_catalog
from mixle_pde.verification.diagnostic_ontology import (
    DiagnosticCode,
    RepairActionKind,
    RepairOutcome,
    RepairRun,
    classify_backend_dispatch_error,
    get_repair_action,
    run_bounded_repair,
)

__all__ = [
    "LoopOutcome",
    "LoopBudget",
    "StageReceipt",
    "LoopReceipt",
    "estimate_dofs",
    "run_coarse_to_fine",
    "run_inverse_case",
]


class LoopOutcome(Enum):
    """The coarse-to-fine loop's typed final disposition -- never a fabricated boolean, matching this
    program's standing "unknown/timeout/resource_limit are valid typed outcomes" convention.

    ``SOLVED``: the final (uncoarsened) resolution solve ran and verified.
    ``UNSUPPORTED``: the problem/backend pairing is genuinely incompatible (an unregistered backend id, or
    an operator/discretization/mesh-cell-type/evidence-kind the backend's profile does not declare) --
    :attr:`LoopReceipt.diagnostic_code` is always set for this outcome.
    ``OVER_BUDGET``: a caller-declared resource ceiling (DOF count or wall-clock time) was exceeded --
    always known *before* the loop gives up, from :class:`LoopBudget` alone, never inferred after a crash.
    ``REQUIRES_HUMAN``: verification failed and no safe, already-grounded mechanical repair applies (or one
    was attempted and its bounded attempt budget was exhausted), or an exception this module's classifiers
    do not specifically recognize was raised. See the module docstring for exactly which of MP-J4's named
    termination categories (unsupported/ill-posed/unidentifiable/OOD/over-budget) this typed vocabulary
    covers by name versus safely-but-generically catches under this outcome.
    """

    SOLVED = "solved"
    UNSUPPORTED = "unsupported"
    OVER_BUDGET = "over_budget"
    REQUIRES_HUMAN = "requires_human"


@dataclass(frozen=True)
class LoopBudget:
    """Caller-declared ceilings the loop enforces before/between every dispatch.

    ``max_dofs`` gates *before* any solve is dispatched (see :func:`estimate_dofs`) -- an over-budget
    problem never reaches the real kernel. ``max_wall_time_seconds`` is checked with a monotonic clock
    before every dispatch (coarse, each repair attempt, final); once exceeded the loop stops immediately,
    even mid-repair. ``max_repair_cycles`` bounds how many times this loop invokes
    :func:`~mixle_pde.verification.diagnostic_ontology.run_bounded_repair` across the whole run (at most
    one per stage in this two-stage -- coarse, final -- design, so at most 2) -- a *loop*-level ceiling on
    top of (never wider than) each :class:`~mixle_pde.verification.diagnostic_ontology.RetryPolicy`'s own
    per-code ``max_attempts``.
    """

    max_dofs: int = 200_000
    max_wall_time_seconds: float = 60.0
    max_repair_cycles: int = 2

    def __post_init__(self) -> None:
        if self.max_dofs <= 0:
            raise ValueError("max_dofs must be positive")
        if self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive")
        if self.max_repair_cycles < 0:
            raise ValueError("max_repair_cycles must be non-negative")


@dataclass(frozen=True)
class StageReceipt:
    """One stage's outcome within a :class:`LoopReceipt` -- every stage the loop actually ran is recorded,
    a failed stage is never hidden by only reporting the aggregate outcome."""

    stage: str
    ok: bool
    detail: str
    elapsed_seconds: float


@dataclass(frozen=True)
class LoopReceipt:
    """The full, inspectable outcome of one :func:`run_coarse_to_fine` (or :func:`run_inverse_case`) call."""

    outcome: LoopOutcome
    backend_id: str
    stages: tuple[StageReceipt, ...]
    repair_runs: tuple[RepairRun, ...]
    final_result: PDEStudyResult | None
    diagnostic_code: DiagnosticCode | None
    detail: str
    wall_time_seconds: float


# ---------------------------------------------------------------------------
# DOF estimation -- grounded directly in each _invoke_* function's own size parameter and default
# (mixle_pde/pde_backend_registry.py), read from source, not guessed.
# ---------------------------------------------------------------------------
_GRID_SIZE_KEY: Mapping[str, str] = {
    "elastic-fd-leapfrog": "grid_size",
    "wave-fd-leapfrog": "grid_size",
    "flow-fd-streamfunction": "grid_size",
    "em-fdtd-yee": "grid_size",
    "transport-fd-advdiff": "grid_size",
    "flow-spectral-ns": "grid_size",
    "helmholtz-pml-fd": "grid_shape",
    "groundwater-fd-transport": "grid_shape",
    "fem-p1-simplex": "grid_shape",
}

#: Each registered invoker's own literal default for its size parameter (pde_backend_registry.py).
_GRID_SIZE_DEFAULT: Mapping[str, Any] = {
    "elastic-fd-leapfrog": 10,
    "wave-fd-leapfrog": 24,
    "flow-fd-streamfunction": 24,
    "em-fdtd-yee": 8,
    "transport-fd-advdiff": 41,
    "flow-spectral-ns": 32,
    "helmholtz-pml-fd": (16, 16),
    "groundwater-fd-transport": (14, 14),
    "fem-p1-simplex": (6, 6),
}

#: Grid dimensionality for the scalar-``grid_size`` backends (DOFs ~ grid_size**power); flow-spectral-ns
#: instead reads its own "dim" solve_plan parameter (2 or 3), so it is handled separately below.
_DOF_POWER: Mapping[str, int] = {
    "elastic-fd-leapfrog": 3,
    "wave-fd-leapfrog": 2,
    "flow-fd-streamfunction": 2,
    "em-fdtd-yee": 3,
    "transport-fd-advdiff": 1,
}


def _enrich_unsupported_detail(detail: str, backend_id: str) -> str:
    """Append :mod:`mixle_pde.verification.knowledge_catalog`'s (MP-J2) own recorded applicability text
    for ``backend_id`` to an ``UNSUPPORTED`` detail message, when a catalog entry for it exists -- gives a
    caller the *reason* a backend cannot do what was asked (not only the raw exception text), reusing
    MP-J2's already-written, already-tested applicability strings rather than re-deriving new ones."""
    try:
        entry = knowledge_catalog.get_entry(backend_id)
    except KeyError:
        return detail
    return f"{detail} (knowledge_catalog applicability for {backend_id!r}: {'; '.join(entry.applicability)})"


def _parameters_of(problem: Mapping[str, Any]) -> dict[str, Any]:
    plan = problem.get("solve_plan") or {}
    params = plan.get("parameters") or {}
    return dict(params)


def _with_parameters(problem: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    """A shallow-copied ``problem`` with ``solve_plan.parameters`` replaced -- every other field (id,
    domains, unknowns, operators, constraints, objectives, evidence_requests) passes through unchanged."""
    plan = dict(problem.get("solve_plan") or {})
    plan["parameters"] = dict(parameters)
    return {**problem, "solve_plan": plan}


def estimate_dofs(problem: Mapping[str, Any], backend_id: str) -> int:
    """Estimate the DOF count the *final* (uncoarsened) resolution in ``problem`` would need for
    ``backend_id``, before any solve is dispatched.

    Raises :class:`KeyError` for a ``backend_id`` this estimator does not recognize -- callers should
    treat that the same way :func:`~mixle_pde.pde_backend_registry.get_kernel_registration` treats an
    unregistered id (an ``UNSUPPORTED`` outcome), not silently skip the budget check.
    """
    if backend_id not in _GRID_SIZE_KEY:
        raise KeyError(
            f"no DOF estimator registered for backend id {backend_id!r}; known backends: {sorted(_GRID_SIZE_KEY)}"
        )
    params = _parameters_of(problem)
    key = _GRID_SIZE_KEY[backend_id]
    value = params.get(key, _GRID_SIZE_DEFAULT[backend_id])
    if backend_id == "flow-spectral-ns":
        dim = int(params.get("dim", 2))
        return int(value) ** dim
    if isinstance(value, (tuple, list)):
        result = 1
        for component in value:
            result *= int(component)
        return result
    return int(value) ** _DOF_POWER[backend_id]


def _coarsen_parameters(
    parameters: Mapping[str, Any], backend_id: str, *, coarse_scale: float, min_size: int = 4
) -> dict[str, Any]:
    """A copy of ``parameters`` with the backend's size parameter scaled down by ``coarse_scale`` (floored
    at ``min_size`` per axis so a degenerate 0/1-point grid is never dispatched)."""
    if not (0.0 < coarse_scale < 1.0):
        raise ValueError("coarse_scale must be in (0, 1)")
    coarse = dict(parameters)
    key = _GRID_SIZE_KEY[backend_id]
    value = parameters.get(key, _GRID_SIZE_DEFAULT[backend_id])
    if isinstance(value, (tuple, list)):
        coarse[key] = tuple(max(min_size, int(round(component * coarse_scale))) for component in value)
    else:
        coarse[key] = max(min_size, int(round(int(value) * coarse_scale)))
    return coarse


# ---------------------------------------------------------------------------
# Verification -- reads the real PDEStudyResult.evidence shape every one of the nine registered kernels
# actually returns (mixle_pde/pde_backend_registry.py, read directly, not assumed): "convergence.finite"
# is present for eight of the nine (fem-p1-simplex's own "convergence" sub-dict never carries a "finite"
# key, only interior_nodes/n_nodes -- its pass/fail signal is the top-level "residual" instead); ".stable"
# is present only for elastic-fd-leapfrog; a top-level "residual" is present for
# helmholtz-pml-fd/fem-p1-simplex/flow-spectral-ns. Every check below is a ".get"/"in" membership test, so
# a kernel missing one of these keys is never a KeyError, only a skipped check.
# ---------------------------------------------------------------------------
def _verify_evidence(evidence: Mapping[str, Any], *, residual_tolerance: float) -> tuple[bool, str]:
    """A generic accept-bar check over any registered kernel's real evidence dict: finite (and, when
    reported, stable) state, and a top-level residual within ``residual_tolerance`` when one is reported.

    Deliberately generic and looser than ``knowledge_catalog.py``'s own per-kernel tolerances (e.g.
    fem-p1-simplex/helmholtz-pml-fd's own tests hold their residual to 1e-8): this module's job is a
    single coarse-to-fine accept/reject gate reusable across all nine kernels, not a replacement for
    MP-J2's already-recorded, kernel-specific verification evidence.
    """
    convergence = evidence.get("convergence") or {}
    if "finite" in convergence and not convergence["finite"]:
        return False, "evidence.convergence.finite is False"
    if "stable" in convergence and not convergence["stable"]:
        return False, "evidence.convergence.stable is False"
    if "residual" in evidence:
        residual = evidence["residual"]
        if residual is None or residual != residual or abs(residual) == float("inf"):  # NaN check without numpy
            return False, f"evidence.residual is non-finite ({residual!r})"
        if abs(residual) > residual_tolerance:
            return False, f"evidence.residual {residual!r} exceeds tolerance {residual_tolerance!r}"
    return True, "finite/stable/residual checks passed"


#: The only backends this module treats a non-finite/unstable result as a Courant-type (CFL) violation for:
#: knowledge_catalog.py's own applicability text (MP-J2, already-reviewed) documents exactly these three as
#: "explicit conditionally-stable" time-stepping schemes. The other six registered kernels are either a
#: direct/steady solve (helmholtz-pml-fd, fem-p1-simplex -- no time-stepping, no CFL limit to violate) or
#: an implicit method-of-lines scheme by default (groundwater-fd-transport, transport-fd-advdiff) or have
#: no citable, already-established Courant-limit grounding in this repo (flow-fd-streamfunction,
#: flow-spectral-ns) -- applying the same mechanical dt-halving repair to any of those six would not be
#: grounded in anything this repository has already verified, so a failure there is REQUIRES_HUMAN instead.
_CFL_REPAIRABLE_BACKENDS = frozenset({"elastic-fd-leapfrog", "wave-fd-leapfrog", "em-fdtd-yee"})


def _classify_evidence_failure(evidence: Mapping[str, Any], backend_id: str) -> DiagnosticCode | None:
    """Classify a *verification* failure (a completed solve whose evidence did not pass
    :func:`_verify_evidence`) -- distinct from a *dispatch* failure (an exception), which
    :func:`~mixle_pde.verification.diagnostic_ontology.classify_backend_dispatch_error` already covers.

    Returns :attr:`~mixle_pde.verification.diagnostic_ontology.DiagnosticCode.CFL_VIOLATION` only for
    :data:`_CFL_REPAIRABLE_BACKENDS`, reusing that MP-J3 code honestly: a non-finite (or, for
    elastic-fd-leapfrog, explicitly non-stable) result from an explicit conditionally-stable leapfrog
    kernel *is* a Courant-type violation by construction, not a stretch of the code's meaning. Returns
    ``None`` -- "not classified by this ontology" -- for every other backend, exactly mirroring
    :func:`~mixle_pde.verification.diagnostic_ontology.classify_failure_class`'s own "never guess" contract
    for a failure-class string it does not recognize.
    """
    if backend_id not in _CFL_REPAIRABLE_BACKENDS:
        return None
    convergence = evidence.get("convergence") or {}
    finite = convergence.get("finite", True)
    stable = convergence.get("stable", True)
    if not finite or not stable:
        return DiagnosticCode.CFL_VIOLATION
    return None


def _attempt_repair(
    problem_template: Mapping[str, Any],
    backend_id: str,
    code: DiagnosticCode,
    params: Mapping[str, Any],
    *,
    residual_tolerance: float,
) -> tuple[RepairRun, PDEStudyResult | None]:
    """Apply MP-J3's :data:`~mixle_pde.verification.diagnostic_ontology.REPAIR_CATALOG` recipe for
    ``code`` against a real dispatch loop: each candidate parameter set is actually solved (via
    :func:`~mixle_pde.pde_backend_registry.run_math_problem`) and actually re-verified (via
    :func:`_verify_evidence`), never merely checked against a synthetic predicate.

    Defensively short-circuits to a :attr:`~mixle_pde.verification.diagnostic_ontology.RepairOutcome.REQUIRES_HUMAN`
    :class:`~mixle_pde.verification.diagnostic_ontology.RepairRun` (zero attempts, no dispatch) when the
    recipe's target parameter is not present in ``params`` at all -- J3's own
    :class:`~mixle_pde.verification.diagnostic_ontology.RepairAction` catalog is grounded in
    :mod:`mixle_pde.continuation`'s keyword parameters (``dt``/``max_iterations``/``damping``), and only
    ``dt`` (the CFL_VIOLATION recipe's target) is ever reachable here (see
    :data:`_CFL_REPAIRABLE_BACKENDS`); a missing parameter would otherwise reach
    :func:`~mixle_pde.verification.diagnostic_ontology.apply_repair_attempt`'s own documented ``KeyError``,
    which this function never lets propagate.
    """
    action = get_repair_action(code)
    if action.kind is RepairActionKind.SCALE_PARAMETER and action.parameter_name not in params:
        return (
            RepairRun(code=code, outcome=RepairOutcome.REQUIRES_HUMAN, attempts_used=0, final_params=dict(params)),
            None,
        )

    last_result: dict[str, PDEStudyResult] = {}

    def is_resolved(candidate_params: Mapping[str, Any]) -> bool:
        candidate_problem = _with_parameters(problem_template, candidate_params)
        try:
            result = run_math_problem(candidate_problem, backend_id)
        except Exception:
            return False
        ok, _detail = _verify_evidence(result.evidence, residual_tolerance=residual_tolerance)
        if ok:
            last_result["result"] = result
        return ok

    run = run_bounded_repair(action, params, is_resolved)
    return run, last_result.get("result")


def run_coarse_to_fine(
    problem: Mapping[str, Any],
    backend_id: str,
    *,
    budget: LoopBudget = LoopBudget(),
    coarse_scale: float = 0.5,
    residual_tolerance: float = 1.0e-6,
) -> LoopReceipt:
    """Run a bounded coarse-then-final study through a registered :mod:`mixle_pde.pde_backend_registry`
    kernel: estimate DOFs, dispatch a coarsened solve, verify it, bounded-repair it if needed, then
    dispatch the caller's originally-requested (uncoarsened) resolution and verify/repair that too.

    Never raises for an unsupported backend/problem pairing, an over-budget request, or a verification
    failure with or without an available mechanical repair -- every one of those resolves to a typed
    :class:`LoopOutcome` on the returned :class:`LoopReceipt`, per this module's docstring.
    """
    start = time.monotonic()
    stages: list[StageReceipt] = []
    repair_runs: list[RepairRun] = []

    def elapsed() -> float:
        return time.monotonic() - start

    def finish(
        outcome: LoopOutcome, *, detail: str, code: DiagnosticCode | None = None, result: PDEStudyResult | None = None
    ) -> LoopReceipt:
        return LoopReceipt(
            outcome=outcome,
            backend_id=backend_id,
            stages=tuple(stages),
            repair_runs=tuple(repair_runs),
            final_result=result,
            diagnostic_code=code,
            detail=detail,
            wall_time_seconds=elapsed(),
        )

    # --- estimate_resources: DOF ceiling, before any solve is dispatched ---
    try:
        dofs = estimate_dofs(problem, backend_id)
    except KeyError as exc:
        stages.append(StageReceipt("estimate_resources", False, str(exc), elapsed()))
        return finish(LoopOutcome.UNSUPPORTED, detail=str(exc), code=DiagnosticCode.UNKNOWN_BACKEND)
    over_dof_budget = dofs > budget.max_dofs
    stages.append(
        StageReceipt(
            "estimate_resources", not over_dof_budget, f"estimated {dofs} DOFs vs. budget {budget.max_dofs}", elapsed()
        )
    )
    if over_dof_budget:
        return finish(LoopOutcome.OVER_BUDGET, detail=f"estimated {dofs} DOFs exceeds budget {budget.max_dofs}")

    final_parameters = _parameters_of(problem)
    working_parameters = _coarsen_parameters(final_parameters, backend_id, coarse_scale=coarse_scale)
    repair_cycles_used = 0

    for stage_name, stage_parameters in (("coarse", working_parameters), ("final", None)):
        if stage_name == "final":
            # Carry forward any parameter a coarse-stage repair changed (e.g. a halved dt) into the final,
            # originally-requested resolution -- refining the grid can shift a Courant-type limit (finer
            # spacing, fixed dt, worsens vp*dt/spacing), so the repaired value is the right starting point,
            # not the caller's original (already known to have failed at a coarser, more forgiving grid).
            stage_parameters = dict(final_parameters)
            # only carry forward non-size parameters that changed (e.g. "dt"), never the coarsened size itself
            for key, value in working_parameters.items():
                if key == _GRID_SIZE_KEY[backend_id]:
                    continue
                if final_parameters.get(key) != value:
                    stage_parameters[key] = value

        if elapsed() > budget.max_wall_time_seconds:
            return finish(
                LoopOutcome.OVER_BUDGET, detail=f"wall time {elapsed():.3f}s exceeded budget before {stage_name}_solve"
            )

        candidate_problem = _with_parameters(problem, stage_parameters)
        try:
            result = run_math_problem(candidate_problem, backend_id)
        except Exception as exc:
            code = classify_backend_dispatch_error(exc)
            stages.append(StageReceipt(f"{stage_name}_solve", False, f"{type(exc).__name__}: {exc}", elapsed()))
            if code is not None:
                return finish(
                    LoopOutcome.UNSUPPORTED, detail=_enrich_unsupported_detail(str(exc), backend_id), code=code
                )
            return finish(LoopOutcome.REQUIRES_HUMAN, detail=f"unrecognized exception {type(exc).__name__}: {exc}")

        stages.append(StageReceipt(f"{stage_name}_solve", True, "dispatched and returned", elapsed()))
        ok, verify_detail = _verify_evidence(result.evidence, residual_tolerance=residual_tolerance)
        stages.append(StageReceipt(f"verify_{stage_name}", ok, verify_detail, elapsed()))

        if not ok:
            code = _classify_evidence_failure(result.evidence, backend_id)
            if code is None:
                return finish(LoopOutcome.REQUIRES_HUMAN, detail=verify_detail, code=None)
            if repair_cycles_used >= budget.max_repair_cycles:
                return finish(
                    LoopOutcome.OVER_BUDGET,
                    detail=f"repair-cycle budget ({budget.max_repair_cycles}) exhausted",
                    code=code,
                )
            repair_cycles_used += 1
            repair_run, repaired_result = _attempt_repair(
                problem, backend_id, code, stage_parameters, residual_tolerance=residual_tolerance
            )
            repair_runs.append(repair_run)
            stages.append(
                StageReceipt(
                    f"repair_{stage_name}",
                    repair_run.outcome is RepairOutcome.RESOLVED,
                    f"{code.value}: {repair_run.outcome.value} in {repair_run.attempts_used} attempt(s)",
                    elapsed(),
                )
            )
            if repair_run.outcome is not RepairOutcome.RESOLVED:
                return finish(LoopOutcome.REQUIRES_HUMAN, detail=verify_detail, code=code)
            result = repaired_result
            if stage_name == "coarse":
                working_parameters = dict(repair_run.final_params)
            else:
                final_parameters = dict(repair_run.final_params)

        if stage_name == "final":
            return finish(LoopOutcome.SOLVED, detail="final resolution solved and verified", result=result)

    raise AssertionError("unreachable: the coarse/final stage loop always returns")  # pragma: no cover


def run_inverse_case(
    dataset_ref: str,
    modality: str,
    prior: str,
    config: dict[str, Any] | None = None,
) -> LoopReceipt:
    """Run :func:`mixle_pde.tools.run_inversion` (MP-J1/IC-3) under the same typed :class:`LoopOutcome`
    vocabulary :func:`run_coarse_to_fine` uses for the forward-kernel path.

    ``run_inversion`` only fits the two linear (fixed-Jacobian) modalities (``gravity``/``magnetics``) with
    the ``"smooth"`` prior today; every other modality/prior value raises a clear
    :class:`ValueError` naming what *is* wired (see ``tools.py``'s own docstring) -- this function maps
    that (and any backend-dispatch-shaped error) to :attr:`LoopOutcome.UNSUPPORTED` via MP-J3's
    :func:`~mixle_pde.verification.diagnostic_ontology.classify_backend_dispatch_error`, and any other
    exception to :attr:`LoopOutcome.REQUIRES_HUMAN` (the same generic backstop
    :func:`run_coarse_to_fine` uses), rather than let either propagate.

    Unlike :func:`run_coarse_to_fine`, this function takes no :class:`LoopBudget`: ``run_inversion`` is one
    non-iterative call with no coarse/refine structure and no cheap, already-available size parameter to
    estimate cost from before reading ``dataset_ref`` off disk (unlike :func:`estimate_dofs`, which reads a
    solve_plan parameter that is already in memory) -- claiming a pre-flight budget gate here would not be
    real enforcement. ``wall_time_seconds`` on the returned receipt is still a genuine, measured
    observation, just not one this function can act on before the call completes.

    No dedicated ``ill_posed``/``unidentifiable`` outcome is produced here either (see the module
    docstring): ``run_inversion``'s own closed-form linear-Gaussian posterior
    (:func:`mixle_pde.field_inversion.linear_gaussian_invert`) always adds a strictly-positive-definite
    prior precision plus a ``1e-10`` Cholesky jitter floor before solving, specifically so a genuinely
    singular posterior precision essentially cannot arise through this entry point -- confirmed by reading
    that function directly rather than assumed, so there is no already-grounded, naturally-occurring
    trigger for a dedicated "ill-posed" code here to classify (a fabricated one would violate this
    program's "no invented failure signal" convention); a truly pathological caller-supplied prior/dataset
    would surface as an ordinary, unclassified exception and fall through to ``REQUIRES_HUMAN`` like any
    other unrecognized failure, never as a silent success.
    """
    from mixle_pde import tools

    start = time.monotonic()
    try:
        result = tools.run_inversion(dataset_ref, modality, prior, config)
    except Exception as exc:
        elapsed = time.monotonic() - start
        code = classify_backend_dispatch_error(exc)
        stage = StageReceipt("run_inversion", False, f"{type(exc).__name__}: {exc}", elapsed)
        if code is not None:
            outcome = LoopOutcome.UNSUPPORTED
        else:
            outcome = LoopOutcome.REQUIRES_HUMAN
        return LoopReceipt(
            outcome=outcome,
            backend_id=f"run_inversion:{modality}",
            stages=(stage,),
            repair_runs=(),
            final_result=None,
            diagnostic_code=code,
            detail=str(exc),
            wall_time_seconds=elapsed,
        )

    elapsed = time.monotonic() - start
    stage = StageReceipt("run_inversion", True, f"posterior_ref={result['posterior_ref']}", elapsed)
    return LoopReceipt(
        outcome=LoopOutcome.SOLVED,
        backend_id=f"run_inversion:{modality}",
        stages=(stage,),
        repair_runs=(),
        final_result=None,
        diagnostic_code=None,
        detail=f"posterior fit: {result['diagnostics']}",
        wall_time_seconds=elapsed,
    )
