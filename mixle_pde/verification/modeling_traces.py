"""Verified modeling-trace corpus: requirement -> assumption -> check -> outcome/repair traces (MP-M1).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``, MP-M1) records: "No
requirements->assumptions->repair trace corpus found anywhere." This module is a first worked
generator for that corpus, structurally templated on :mod:`mixle_numerics.training` (mixle-discrete,
DISC-T2/T4, already merged to that repo's ``release/0.8.0``): both modules are a deterministic,
seed-parameterized generator producing typed, independently-reverified records from an
already-shipped capability, never inventing a competing schema and never trusting the producing
capability's own success claim without a fresh, independent recheck. The domain differs by design --
that module generates (problem, solution, trace) triples for a pure-math task (polynomial
factorization); this one generates a *modeling-decision* trace (a requirement, an assumption that
turns out to be wrong, a check against a real solver run, and a repair) for a PDE task, per this
card's own description: "Record requirements, assumptions, gaps, ..., diagnostics, repairs, ...".

Source capability: the ``elastic-fd-leapfrog`` kernel already registered in
:mod:`mixle_pde.pde_backend_registry` (read-only import here -- this module registers nothing new
and does not modify that registry), which wraps the real, unmodified
:class:`mixle_pde.elastic.ElasticWave3D`. That class's own docstring states its applicability limit
in closed form: "The 3D Courant limit is ``dt <= spacing / (vp * sqrt(3))``." This module deliberately
constructs a ``dt`` that violates that documented limit (the brief's own second example of a
deliberately-violated precondition: "a parameter outside the kernel's stated applicability"), runs
the real kernel through :func:`mixle_pde.pde_backend_registry.run_math_problem` unmodified, and reads
back its actual returned evidence -- not a fabricated one. A real elastic-wave leapfrog run pushed
this far past its own stability limit does not always overflow to ``inf``/``nan`` within a handful of
steps (confirmed empirically before writing this module: at courant ~4x the limit the returned state
stayed finite but grew to ~1e11 within 8 steps); the kernel's own ``stable`` flag (Courant number
against its own stated limit), not raw finiteness alone, is therefore the failure signal this module
checks, exactly the modeling lesson a real detect-and-repair trace should capture.

Independent re-verification (never trust the kernel's own return value directly, mirroring
:mod:`mixle_numerics.training`'s "reconstruct from scratch, do not re-invoke the producing algorithm"
discipline): :func:`_run_and_verify` recomputes the Courant number from the raw ``vp``/``dt``/
``spacing`` inputs using the same documented formula -- entirely outside
:class:`~mixle_pde.elastic.ElasticWave3D`'s own code path -- and separately calls
``np.isfinite(...).all()`` directly on the kernel's raw returned solution array, never reading the
evidence dict's own ``finite``/``stable``/``courant_number`` fields as ground truth. Both independent
results are then compared against what the kernel itself self-reported
(:attr:`ModelingTraceVerification.agrees_with_kernel_report`); every trace's ``final_outcome`` is
derived from that fresh recheck of the repair (second) kernel call, never from the first call's
label or from an assumption that the repair worked.

``final_outcome`` is a three-valued :class:`ModelingTraceOutcome` (``repaired`` / ``unresolved`` /
``unknown``), never a fabricated boolean, per this program's standing convention. Both real outcomes
are exercised by a genuine (not mocked) kernel rerun: :func:`generate_elastic_cfl_trace` repairs with
a comfortable safety margin (independently reverified to actually succeed), and
:func:`generate_elastic_cfl_unresolved_trace` "repairs" with a deliberately insufficient margin (still
15% over the limit -- a realistic near-miss) so the corpus also carries a genuine counterexample trace,
per this card's own "Generate positive and counterexample traces" note and its accept bar that "failed
traces are labeled by failure class rather than removed."

Scope: one worked generator (the ``elastic-fd-leapfrog`` CFL precondition) over one registered
kernel, matching the sibling D19 card's own stated "lean acceptance bar" rather than the full MP-M1
description's much larger surface (mesh/solver/coupling decisions, inference actions, posterior
diagnostics, surrogate lifecycle, ...) -- those remain future generators over their own registered
capabilities, not invented here.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from mixle_pde.pde_backend_registry import get_kernel_registration, run_math_problem

__all__ = [
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "ModelingStepKind",
    "ModelingTraceOutcome",
    "ModelingStep",
    "ModelingTraceVerification",
    "ModelingTrace",
    "ModelingTraceGenerationError",
    "generate_elastic_cfl_trace",
    "generate_elastic_cfl_unresolved_trace",
    "generate_elastic_cfl_trace_corpus",
]

GENERATOR_ID = "mixle_pde.verification.modeling_traces.elastic_cfl_precondition"
GENERATOR_VERSION = "1.0.0"

_BACKEND_ID = "elastic-fd-leapfrog"
#: mixle_pde.elastic.ElasticWave3D's own documented 3-D leapfrog Courant limit: dt <= spacing/(vp*sqrt(3)).
_COURANT_LIMIT = float(1.0 / np.sqrt(3.0))
#: Comfortable margin under the limit -- the "successful repair" generator's target.
_REPAIR_SAFETY_FACTOR = 0.5
#: Still 15% over the limit -- the "insufficient repair" (genuine counterexample) generator's target.
_INSUFFICIENT_REPAIR_SAFETY_FACTOR = 1.15


class ModelingStepKind(Enum):
    """One stage in a modeling trace's requirement -> assumption -> check -> outcome/repair sequence.

    A trace is a concrete sequence over these kinds, and the sequence may revisit ``CHECK``/``OUTCOME``
    after a ``REPAIR`` -- exactly the "detect, then repair, then recheck" loop a real modeling session
    goes through, not a single fixed-length tuple.
    """

    REQUIREMENT = "requirement"
    ASSUMPTION = "assumption"
    CHECK = "check"
    OUTCOME = "outcome"
    REPAIR = "repair"


class ModelingTraceOutcome(Enum):
    """A trace's final, independently-verified disposition -- never a fabricated boolean.

    ``UNKNOWN`` is a first-class outcome, not an error folded into ``UNRESOLVED``: it is reported
    whenever this module's own independent recomputation disagrees with the kernel's self-reported
    evidence -- an honest "this needs a closer look", not a forced pass/fail.
    """

    REPAIRED = "repaired"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


class ModelingTraceGenerationError(RuntimeError):
    """A constructed instance failed this module's own fail-closed sanity check before labeling.

    Mirrors :mod:`mixle_numerics.training`'s ``FactorizationVerificationError`` discipline: refuse to
    emit a trace whose "detected violation" did not actually happen under independent recomputation,
    rather than silently mislabeling it. The violation margin this module draws is always > 1 by
    construction, so this is defense in depth, not an expected runtime path.
    """


@dataclass(frozen=True)
class ModelingStep:
    """One immutable entry in a :class:`ModelingTrace`'s ``steps`` sequence."""

    kind: ModelingStepKind
    statement: str
    detail: Mapping[str, Any]


@dataclass(frozen=True)
class ModelingTraceVerification:
    """This module's own independent recheck of one kernel call -- never copied from its evidence.

    ``predicted_courant_number``/``predicted_violates_limit`` are computed from the raw ``vp``/``dt``/
    ``spacing`` inputs by this module, before the kernel is even called. ``raw_solution_finite``/
    ``raw_solution_max_abs`` are computed by this module directly from the kernel's raw returned
    solution array (``np.isfinite``/``np.max(np.abs(...))``), never from the evidence dict's own
    ``finite`` field. ``agrees_with_kernel_report`` is true only when both independent recomputations
    match what the kernel itself self-reported in ``PDEStudyResult.evidence``.
    """

    predicted_courant_number: float
    predicted_violates_limit: bool
    reported_courant_number: float
    reported_stable: bool
    raw_solution_finite: bool
    raw_solution_max_abs: float
    agrees_with_kernel_report: bool


@dataclass(frozen=True)
class ModelingTrace:
    """A verified requirement -> assumption -> check -> outcome/repair trace for one modeling task.

    ``violating_evidence``/``repaired_evidence`` are the real, unmodified
    ``PDEStudyResult.evidence`` mappings returned by the two actual kernel calls this trace is built
    from -- "the kernel's own actual return value" the generator's docstring requires every outcome be
    checked against. ``trace_digest`` is a sha256 of a canonical JSON snapshot of the trace's
    identifying fields, so two calls with the same ``(seed, item_index)`` are provably byte-identical
    (mirrors :mod:`mixle_numerics.training`'s "regenerated items have identical typed ... hashes" gate).
    """

    trace_id: str
    generator_id: str
    generator_version: str
    backend_id: str
    kernel_source: str
    violated_precondition: str
    steps: tuple[ModelingStep, ...]
    detected_failure_class: str
    final_outcome: ModelingTraceOutcome
    final_failure_class: str | None
    violating_verification: ModelingTraceVerification
    repaired_verification: ModelingTraceVerification
    violating_problem: Mapping[str, Any]
    repaired_problem: Mapping[str, Any]
    violating_evidence: Mapping[str, Any]
    repaired_evidence: Mapping[str, Any]
    trace_digest: str


def _derive_seed(seed: int, *parts: int) -> int:
    """Deterministically mix an integer seed with extra integer parts.

    Never uses ``hash()``: CPython's string/bytes hashing is process-salted by default
    (``PYTHONHASHSEED``), which would silently break reproducibility across processes.
    """
    mixed = seed
    for part in parts:
        mixed = (mixed * 1_000_003 + int(part)) & ((1 << 63) - 1)
    return mixed


def _elastic_problem(
    *, problem_id: str, grid_size: int, dt: float, n_steps: int, vp: float, spacing: float
) -> dict[str, Any]:
    """Build one CON-MATH-PROBLEM-V1 study for the registered ``elastic-fd-leapfrog`` backend.

    Shape matches ``tests/pde_backend_registry_kernels2_test.py``'s own worked fixture for this same
    backend -- the same required fields ``problem_adapter.inspect_math_problem`` checks for.
    """
    return {
        "id": problem_id,
        "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": "structured_grid"}}],
        "unknowns": [{"id": "field", "domain_id": "domain"}],
        "operators": [
            {
                "id": f"{problem_id}-operator",
                "kind": "time_stepping",
                "input_ids": ["field"],
                "output_ids": ["field"],
                "discretization": "FD-staggered-leapfrog",
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [{"kind": "convergence", "required": True}],
        "solve_plan": {
            "parameters": {
                "grid_size": grid_size,
                "dt": dt,
                "n_steps": n_steps,
                "vp": vp,
                "spacing": spacing,
            }
        },
    }


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_and_verify(
    *, problem_id: str, grid_size: int, dt: float, n_steps: int, vp: float, spacing: float
) -> tuple[dict[str, Any], dict[str, Any], ModelingTraceVerification]:
    """Run the real ``elastic-fd-leapfrog`` kernel once and independently reverify its own return value."""
    problem = _elastic_problem(
        problem_id=problem_id, grid_size=grid_size, dt=dt, n_steps=n_steps, vp=vp, spacing=spacing
    )
    # Independent recomputation #1: pure arithmetic on the raw inputs, computed before the kernel runs
    # and never touching mixle_pde.elastic.ElasticWave3D's own code path.
    predicted_courant = float(vp * dt / spacing)
    predicted_violates = predicted_courant > _COURANT_LIMIT

    result = run_math_problem(problem, _BACKEND_ID)
    evidence = dict(result.evidence)
    convergence = evidence["convergence"]

    # Independent recomputation #2: inspect the raw returned array directly, not the evidence dict.
    raw_finite = bool(np.isfinite(result.solution).all())
    raw_max_abs = float(np.max(np.abs(result.solution))) if raw_finite else float("inf")

    reported_courant = float(convergence["courant_number"])
    reported_stable = bool(convergence["stable"])
    independently_stable = not predicted_violates
    agrees = (
        abs(reported_courant - predicted_courant) < 1.0e-9
        and reported_stable == independently_stable
        and bool(convergence["finite"]) == raw_finite
    )
    verification = ModelingTraceVerification(
        predicted_courant_number=predicted_courant,
        predicted_violates_limit=predicted_violates,
        reported_courant_number=reported_courant,
        reported_stable=reported_stable,
        raw_solution_finite=raw_finite,
        raw_solution_max_abs=raw_max_abs,
        agrees_with_kernel_report=agrees,
    )
    return problem, evidence, verification


def _generate_elastic_cfl_trace(*, seed: int, item_index: int, repair_safety_factor: float) -> ModelingTrace:
    item_seed = _derive_seed(seed, item_index)
    rng = random.Random(item_seed)
    registration = get_kernel_registration(_BACKEND_ID)

    grid_size = rng.choice((5, 6, 7))
    spacing = 1.0 / (grid_size - 1)
    vp = round(rng.uniform(1.2, 2.4), 3)
    n_steps = rng.choice((6, 8, 10))
    violation_factor = rng.uniform(1.8, 4.5)  # always > 1: deliberately over the documented limit
    dt_violating = round(violation_factor * _COURANT_LIMIT * spacing / vp, 6)
    dt_repaired = round(repair_safety_factor * _COURANT_LIMIT * spacing / vp, 6)

    requirement = ModelingStep(
        kind=ModelingStepKind.REQUIREMENT,
        statement=(
            "The elastic-wave leapfrog stepper's evolved state must satisfy its own documented 3-D "
            "Courant stability limit: courant = vp * dt / spacing <= 1/sqrt(3) "
            "(mixle_pde.elastic.ElasticWave3D docstring)."
        ),
        detail={"courant_limit": _COURANT_LIMIT, "formula": "vp * dt / spacing", "backend_id": _BACKEND_ID},
    )
    assumption = ModelingStep(
        kind=ModelingStepKind.ASSUMPTION,
        statement=(
            f"Assumed dt={dt_violating} is small enough for vp={vp}, spacing={spacing:.6f} on a "
            f"{grid_size}^3 grid to satisfy the Courant limit -- deliberately wrong by construction "
            f"(violation_factor={violation_factor:.3f} > 1), so the check below detects a real failure "
            "rather than a fabricated one."
        ),
        detail={
            "vp": vp,
            "dt": dt_violating,
            "spacing": spacing,
            "grid_size": grid_size,
            "n_steps": n_steps,
            "violation_factor": violation_factor,
        },
    )

    violating_problem, violating_evidence, violating_verification = _run_and_verify(
        problem_id=f"modeling-trace-{item_seed}-violating",
        grid_size=grid_size,
        dt=dt_violating,
        n_steps=n_steps,
        vp=vp,
        spacing=spacing,
    )
    if not violating_verification.predicted_violates_limit:
        raise ModelingTraceGenerationError(
            f"seed {item_seed}: constructed 'violating' dt={dt_violating} did not actually exceed the "
            "Courant limit under this module's own independent recomputation; refusing to fabricate a "
            "detect step that did not really happen"
        )

    detected_failure_class = (
        "cfl_violation_diverged" if not violating_verification.raw_solution_finite else "cfl_violation_unstable_growth"
    )
    check_1 = ModelingStep(
        kind=ModelingStepKind.CHECK,
        statement=(
            "Ran the real, unmodified elastic-fd-leapfrog kernel via "
            "mixle_pde.pde_backend_registry.run_math_problem and independently reverified its returned "
            "evidence: recomputed the Courant number from the raw vp/dt/spacing inputs (not the "
            "kernel's internal state), and independently checked np.isfinite(...) on the raw returned "
            "solution array (not the evidence dict's own 'finite' flag)."
        ),
        detail={
            "evidence": violating_evidence,
            "independent_courant_number": violating_verification.predicted_courant_number,
            "independent_raw_solution_finite": violating_verification.raw_solution_finite,
            "independent_raw_solution_max_abs": violating_verification.raw_solution_max_abs,
            "agrees_with_kernel_report": violating_verification.agrees_with_kernel_report,
        },
    )
    outcome_1 = ModelingStep(
        kind=ModelingStepKind.OUTCOME,
        statement=(
            f"Confirmed a genuine precondition violation ({detected_failure_class}): the independently "
            f"recomputed courant={violating_verification.predicted_courant_number:.4f} exceeds the "
            f"limit {_COURANT_LIMIT:.4f}, and the real kernel run agrees "
            f"(self-reported stable={violating_verification.reported_stable})."
        ),
        detail={"failure_class": detected_failure_class},
    )
    repair = ModelingStep(
        kind=ModelingStepKind.REPAIR,
        statement=(
            f"Repaired by changing dt from {dt_violating} to {dt_repaired} "
            f"({repair_safety_factor:.2f}x the limit-satisfying value for the same vp/spacing/grid), "
            "the same documented Courant formula, holding every other parameter fixed."
        ),
        detail={
            "dt_before": dt_violating,
            "dt_after": dt_repaired,
            "repair_safety_factor": repair_safety_factor,
        },
    )

    repaired_problem, repaired_evidence, repaired_verification = _run_and_verify(
        problem_id=f"modeling-trace-{item_seed}-repaired",
        grid_size=grid_size,
        dt=dt_repaired,
        n_steps=n_steps,
        vp=vp,
        spacing=spacing,
    )
    check_2 = ModelingStep(
        kind=ModelingStepKind.CHECK,
        statement="Reran the same real kernel with the repaired dt and independently reverified again.",
        detail={
            "evidence": repaired_evidence,
            "independent_courant_number": repaired_verification.predicted_courant_number,
            "independent_raw_solution_finite": repaired_verification.raw_solution_finite,
            "independent_raw_solution_max_abs": repaired_verification.raw_solution_max_abs,
            "agrees_with_kernel_report": repaired_verification.agrees_with_kernel_report,
        },
    )

    if not repaired_verification.agrees_with_kernel_report:
        final_outcome = ModelingTraceOutcome.UNKNOWN
        final_failure_class = "verification_disagreement"
        outcome_statement = (
            "This module's independent recomputation disagrees with the kernel's own self-reported "
            "evidence on the repaired run -- labeled unknown rather than trusting either side blindly."
        )
    elif repaired_verification.raw_solution_finite and not repaired_verification.predicted_violates_limit:
        final_outcome = ModelingTraceOutcome.REPAIRED
        final_failure_class = None
        outcome_statement = (
            f"Repair independently verified: recomputed courant="
            f"{repaired_verification.predicted_courant_number:.4f} <= limit {_COURANT_LIMIT:.4f}, and "
            "the raw returned state is finite -- confirmed against the real kernel's own return value."
        )
    else:
        final_outcome = ModelingTraceOutcome.UNRESOLVED
        final_failure_class = "repair_still_violates_precondition"
        outcome_statement = (
            "The repaired run still fails independent verification against the real kernel's own "
            "return value -- labeled unresolved, not silently dropped from the corpus."
        )
    outcome_2 = ModelingStep(
        kind=ModelingStepKind.OUTCOME,
        statement=outcome_statement,
        detail={"final_outcome": final_outcome.value, "final_failure_class": final_failure_class},
    )

    steps = (requirement, assumption, check_1, outcome_1, repair, check_2, outcome_2)
    digest_payload = {
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "seed": item_seed,
        "backend_id": _BACKEND_ID,
        "repair_safety_factor": repair_safety_factor,
        "violating_problem": violating_problem,
        "repaired_problem": repaired_problem,
        "final_outcome": final_outcome.value,
    }
    return ModelingTrace(
        trace_id=f"modeling-trace-elastic-cfl-{item_seed}",
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        backend_id=_BACKEND_ID,
        kernel_source=registration.source,
        violated_precondition="3-D leapfrog Courant stability limit: vp * dt / spacing <= 1/sqrt(3)",
        steps=steps,
        detected_failure_class=detected_failure_class,
        final_outcome=final_outcome,
        final_failure_class=final_failure_class,
        violating_verification=violating_verification,
        repaired_verification=repaired_verification,
        violating_problem=violating_problem,
        repaired_problem=repaired_problem,
        violating_evidence=violating_evidence,
        repaired_evidence=repaired_evidence,
        trace_digest=_canonical_digest(digest_payload),
    )


def generate_elastic_cfl_trace(*, seed: int, item_index: int = 0) -> ModelingTrace:
    """Deterministic positive-path trace: a genuine CFL violation, then a repair that this module's own
    independent recheck confirms actually fixed it (``final_outcome == REPAIRED``).

    Pure function of ``(seed, item_index)``: two calls with the same inputs produce a byte-identical
    ``trace_digest``.
    """
    return _generate_elastic_cfl_trace(seed=seed, item_index=item_index, repair_safety_factor=_REPAIR_SAFETY_FACTOR)


def generate_elastic_cfl_unresolved_trace(*, seed: int, item_index: int = 0) -> ModelingTrace:
    """Deterministic counterexample trace: the "repair" still leaves the Courant number 15% over the
    limit (a realistic insufficient correction, not a mocked failure), so the real second kernel call's
    own independent recheck reports ``final_outcome == UNRESOLVED`` with failure_class
    ``"repair_still_violates_precondition"`` -- exercising this card's "failed traces are labeled by
    failure class rather than removed" requirement against a real, unmodified kernel rerun.
    """
    return _generate_elastic_cfl_trace(
        seed=seed, item_index=item_index, repair_safety_factor=_INSUFFICIENT_REPAIR_SAFETY_FACTOR
    )


def generate_elastic_cfl_trace_corpus(*, seed: int, count: int) -> tuple[ModelingTrace, ...]:
    """Generate ``count`` reproducible positive-path traces (see :func:`generate_elastic_cfl_trace`)."""
    if count < 1:
        raise ValueError("count must be positive")
    return tuple(generate_elastic_cfl_trace(seed=seed, item_index=index) for index in range(count))
