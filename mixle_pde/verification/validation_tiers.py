"""Typed validation-evidence tiers for mixle-pde capabilities (MP-K5).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``, MP-K5) records that
mixle-pde's ``io/`` modules ingest real field data as *inputs* (SEG-Y, LAS, InSAR, assays), and that
several modules already produce individual pieces of validation evidence informally --
:mod:`mixle_pde.verification.mms` and :func:`mixle_pde.multiphysics_reference.manufactured_solution_
refinement` both run the method of manufactured solutions; ``canonical_adapter.py`` stamps an ad hoc
``evidence_strength`` string on every receipt; ``capability_inventory.py`` stamps a free-text
``verification_level`` per module -- but no shared, typed scheme exists anywhere for *which kind* of
validation evidence a result is, or for comparing across kinds. This module is that scheme.

Five tiers, ordered by how controlled the comparison is (not by a collapsible "strength" score --
see the honesty note below):

1. ``manufactured`` -- Method of Manufactured Solutions: an exact solution chosen for algebraic
   convenience (it need not satisfy any physical law on its own); the PDE's forcing/source/boundary
   terms are derived so that solution becomes exact, and the discretization is checked against it.
2. ``analytic`` -- a closed-form solution of the real physical problem itself (not an invented
   forcing function), valid in some idealized regime (e.g. 1-D steady conduction, Poiseuille flow).
3. ``code_to_code`` -- agreement against a second, independently implemented solver or backend on
   the same problem.
4. ``experimental`` -- agreement against a controlled laboratory/bench measurement, with documented
   instrumentation and uncertainty.
5. ``field`` -- comparison against real-world, uncontrolled deployment data, with no independent
   ground truth available at all.

Methodology and honesty notes
------------------------------
* Tier order above is the conventional MMS -> analytic -> code-to-code -> experimental -> field
  progression this repo's own docs (``docs/validation.rst``, ``docs/release-readiness.rst``,
  ``docs/solver-selection-and-inversion-guide.rst``) already use informally. It is **not** a scalar
  strength ranking that collapses to "higher tier = more trustworthy": each tier answers a different
  question and has its own confounds (see :data:`TIER_SEMANTICS`). A ``field``-tier plausibility
  check on real deployment data is not "worse evidence" than a ``manufactured``-tier check in some
  universal sense -- it is evidence of a different thing (real-world plausibility vs. discretization
  correctness), and neither substitutes for the other.
* :data:`STATIC_EVIDENCE_CATALOG` entries are citations of already-merged evidence living in *other*
  modules' own tests, verified by reading the cited module/test when this catalog was written. This
  module does not re-execute them and their continued truth is not re-checked automatically --
  ``tests/validation_tiers_test.py`` asserts every cited source path actually exists on disk, but a
  citation going stale in *content* (not just path) would not be caught until someone re-reads it.
  Records produced by the :func:`validation_evidence` decorator, by contrast, carry
  ``origin="dynamic_run"``: they are appended only after the wrapped function actually returns
  without raising, in the current process -- a failing or skipped test cannot silently claim
  evidence. The two kinds are distinguished by :attr:`ValidationEvidenceRecord.origin`, not merely by
  docstring prose, so a caller can filter on it.
* No ``experimental`` or ``field`` tier evidence is registered anywhere in this module as of this
  writing -- consistent with the MP-K5 ledger finding that drove this module's creation. Defining
  :class:`ValidationTier` members and :data:`TIER_SEMANTICS` for those two tiers is scope (the
  taxonomy has to be able to name them so a future field-validation card has somewhere to attach
  its evidence); populating them with a real record is not, and this module does not fabricate one.
* This module only classifies and reports evidence that already exists elsewhere; it does not itself
  establish correctness, run a solver, or grant a capability any tier it has not earned.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "ValidationTier",
    "TIER_ORDER",
    "TierSemantics",
    "TIER_SEMANTICS",
    "semantics_for",
    "ValidationEvidenceRecord",
    "STATIC_EVIDENCE_CATALOG",
    "register_evidence",
    "clear_registry",
    "registered_evidence",
    "validation_evidence",
    "evidence_for_capability",
    "tiers_for_capability",
    "known_capabilities",
    "CapabilityEvidenceReport",
    "evidence_report",
]


class ValidationTier(Enum):
    """One kind of validation comparison a result can be checked against."""

    MANUFACTURED = "manufactured"
    ANALYTIC = "analytic"
    CODE_TO_CODE = "code_to_code"
    EXPERIMENTAL = "experimental"
    FIELD = "field"


#: Canonical display/iteration order for the five tiers (see module docstring: not a strength rank).
TIER_ORDER: tuple[ValidationTier, ...] = (
    ValidationTier.MANUFACTURED,
    ValidationTier.ANALYTIC,
    ValidationTier.CODE_TO_CODE,
    ValidationTier.EXPERIMENTAL,
    ValidationTier.FIELD,
)


@dataclass(frozen=True)
class TierSemantics:
    """What a tier's evidence does and does not establish.

    ``gives_exact_error`` is true only when the comparison target is known in closed form, so the
    measured discrepancy *is* the error (not an estimate of it bounded by some other uncertainty).
    ``establishes_physical_validity`` is true only when the comparison target is drawn from physical
    reality (a real solution, a real measurement) rather than another model or an invented function --
    a manufactured solution deliberately need not be physical, so it is false there even though the
    error is exact.
    """

    tier: ValidationTier
    ground_truth: str
    gives_exact_error: bool
    establishes_physical_validity: bool
    typical_confounds: tuple[str, ...]
    evidence_strength: str

    def __post_init__(self) -> None:
        if not isinstance(self.tier, ValidationTier):
            raise TypeError("tier must be a ValidationTier member")
        for value, label in (
            (self.ground_truth, "ground_truth"),
            (self.evidence_strength, "evidence_strength"),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        if not self.typical_confounds:
            raise ValueError("typical_confounds must list at least one known confound")


TIER_SEMANTICS: tuple[TierSemantics, ...] = (
    TierSemantics(
        tier=ValidationTier.MANUFACTURED,
        ground_truth=(
            "An exact solution chosen for algebraic convenience -- it need not satisfy any physical "
            "law on its own. The PDE's forcing/source/boundary terms are derived so that solution "
            "becomes exact, then the real discretized solver is driven with that forcing."
        ),
        gives_exact_error=True,
        establishes_physical_validity=False,
        typical_confounds=(
            "only exercises what the chosen manufactured solution touches -- an untested branch or "
            "boundary case is invisible to a single manufactured solution",
            "confirms the implemented discretization converges at its claimed order; says nothing "
            "about whether the PDE model itself is the right physics for a given deployment",
        ),
        evidence_strength="exact-error, code-verification-only",
    ),
    TierSemantics(
        tier=ValidationTier.ANALYTIC,
        ground_truth=(
            "A closed-form solution that itself solves the real physical problem being modeled "
            "(e.g. 1-D steady conduction, Poiseuille flow), not an invented forcing function."
        ),
        gives_exact_error=True,
        establishes_physical_validity=True,
        typical_confounds=(
            "closed forms exist only for simplified geometry/boundary conditions/coefficients -- "
            "the covered regime is rarely the one actually deployed",
            "agreement in the linear/idealized limit does not bound error in the nonlinear or "
            "heterogeneous regime a production run actually exercises",
        ),
        evidence_strength="exact-error, narrow-regime",
    ),
    TierSemantics(
        tier=ValidationTier.CODE_TO_CODE,
        ground_truth=(
            "Output of a second, independently implemented solver or backend (an external package, "
            "or an independent internal code path) on the same problem."
        ),
        gives_exact_error=False,
        establishes_physical_validity=False,
        typical_confounds=(
            "agreement does not imply correctness if both codes share a modeling assumption, a "
            "discretization family, or a common-ancestor bug",
            "disagreement alone does not localize which code is wrong without a third, more-trusted reference",
        ),
        evidence_strength="relative-agreement-only",
    ),
    TierSemantics(
        tier=ValidationTier.EXPERIMENTAL,
        ground_truth=(
            "A physical measurement from a controlled laboratory or bench experiment, with "
            "documented instrumentation, uncertainty, and boundary/initial conditions."
        ),
        gives_exact_error=False,
        establishes_physical_validity=True,
        typical_confounds=(
            "instrument uncertainty, calibration drift, and idealized lab boundary conditions "
            "versus field deployment conditions",
            "small-scale or short-duration lab regimes may not extrapolate to full-scale or long-duration behavior",
        ),
        evidence_strength="physical, bounded-uncertainty",
    ),
    TierSemantics(
        tier=ValidationTier.FIELD,
        ground_truth=(
            "Real-world field observation (e.g. SEG-Y, LAS, InSAR, assay data) with no controlled "
            "experiment design and no independent ground truth."
        ),
        gives_exact_error=False,
        establishes_physical_validity=False,
        typical_confounds=(
            "no independent ground truth exists: only self-consistency or plausibility can be "
            "checked, never a true error against a known-correct answer",
            "unmodeled confounding processes, incomplete metadata, and non-uniform data quality are "
            "the norm rather than the exception",
        ),
        evidence_strength="plausibility-only, no ground truth",
    ),
)

_SEMANTICS_BY_TIER: dict[ValidationTier, TierSemantics] = {entry.tier: entry for entry in TIER_SEMANTICS}
if frozenset(_SEMANTICS_BY_TIER) != frozenset(ValidationTier):
    raise AssertionError("TIER_SEMANTICS must define exactly one entry per ValidationTier member")


def semantics_for(tier: ValidationTier) -> TierSemantics:
    """Return the evidence-strength semantics for ``tier``."""
    return _SEMANTICS_BY_TIER[tier]


_VALID_ORIGINS = frozenset({"static_catalog", "dynamic_run"})


@dataclass(frozen=True)
class ValidationEvidenceRecord:
    """One tagged piece of evidence: ``capability`` was checked at ``tier`` by ``source``.

    ``capability`` is a free-text identifier (module path, optionally with a registered-kernel id in
    parentheses) -- the same informal-but-consistent-key convention
    :mod:`mixle_pde.verification.capability_inventory` uses for ``SolverProfile.module``.
    ``origin`` distinguishes a human-verified citation of evidence produced elsewhere
    (``"static_catalog"``) from a record appended by :func:`validation_evidence` after its wrapped
    call actually completed in this process (``"dynamic_run"``); see the module docstring.
    """

    capability: str
    tier: ValidationTier
    source: str
    summary: str
    origin: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tier, ValidationTier):
            raise TypeError("tier must be a ValidationTier member")
        for value, label in (
            (self.capability, "capability"),
            (self.source, "source"),
            (self.summary, "summary"),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        if self.origin not in _VALID_ORIGINS:
            raise ValueError(f"origin must be one of {sorted(_VALID_ORIGINS)}, got {self.origin!r}")


# Human-verified citations of validation evidence already merged elsewhere in this repo (read
# directly from the cited source at the time this catalog was written -- see the module docstring's
# honesty note on how this differs from a `validation_evidence`-tagged dynamic record).
STATIC_EVIDENCE_CATALOG: tuple[ValidationEvidenceRecord, ...] = (
    ValidationEvidenceRecord(
        capability="mixle_pde.dynamics.AdvectionDiffusionOperator (transport-fd-advdiff)",
        tier=ValidationTier.MANUFACTURED,
        source=("tests/mms_test.py::test_transport_fd_advdiff_mms_convergence_receipt_passes_at_expected_first_order"),
        summary=(
            "MP-K3: drives the registered transport-fd-advdiff kernel (via mixle_pde.verification."
            "mms.transport_fd_advdiff_convergence_receipt) with a manufactured "
            "sin(kx)cos(wt) solution (independent of the physical dispersion relation) at four grid "
            "resolutions and asserts the measured order of convergence lands within tolerance of the "
            "documented theoretical order (1, first-order upwind advection)."
        ),
        origin="static_catalog",
        limitations=(
            "exercises only the transport-fd-advdiff kernel's 1-D periodic-grid path",
            "confirms discretization order, not that advection-diffusion is the right physical model "
            "for any given deployment",
        ),
    ),
    ValidationEvidenceRecord(
        capability="mixle_pde.multiphysics_reference (composite-heat P1 FEM reference)",
        tier=ValidationTier.MANUFACTURED,
        source=(
            "tests/multiphysics_reference_test.py::test_p1_fem_and_manufactured_refinement_produce_independent_evidence"
        ),
        summary=(
            "Verifies the composite-heat reference's P1 FEM assembly (via mixle_pde."
            "multiphysics_reference.manufactured_solution_refinement) against the manufactured "
            "solution u(x)=x(1-x) (satisfying -u''=2 on [0,1]); the merged test asserts the observed "
            "L2 order of convergence is >= 1.99, matching P1 FEM's theoretical second order."
        ),
        origin="static_catalog",
        limitations=("1-D, single-element-type reference problem only",),
    ),
    ValidationEvidenceRecord(
        capability="mixle_pde.verification.result_queries (point/integral/extrema queries)",
        tier=ValidationTier.CODE_TO_CODE,
        source=("tests/result_queries_test.py::test_native_and_fem_backends_agree_on_a_known_analytic_ramp_solution"),
        summary=(
            "MP-K2: runs the same P1 mesh and Poisson problem through two independently implemented "
            "backends -- the native rational-linear adapter (mixle_pde.canonical_adapter) and the "
            "fem-p1-simplex kernel (mixle_pde.pde_backend_registry) -- and asserts point probe, "
            "integral, and extrema queries agree within a documented tolerance across both."
        ),
        origin="static_catalog",
        limitations=(
            "both backends share this repo's own P1 assembly lineage; cross-backend agreement does "
            "not rule out a modeling assumption shared by both",
        ),
    ),
    ValidationEvidenceRecord(
        capability="mixle_pde.verification.result_queries (point/integral/extrema queries)",
        tier=ValidationTier.ANALYTIC,
        source=("tests/result_queries_test.py::test_native_and_fem_backends_agree_on_a_known_analytic_ramp_solution"),
        summary=(
            "The same MP-K2 test also compares both backends against the closed-form analytic ramp "
            "solution u(x, y) = x for a custom Dirichlet condition on the real Poisson problem being "
            "modeled (not an invented forcing function)."
        ),
        origin="static_catalog",
        limitations=(
            "the analytic ramp is a linear, boundary-driven special case; it does not cover "
            "nonzero-source or nonlinear regimes",
        ),
    ),
)

# Records appended by `validation_evidence` after its wrapped call completes without raising, in
# this process. Never mutated directly by callers -- use `register_evidence`/`clear_registry`.
_REGISTRY: list[ValidationEvidenceRecord] = []


def register_evidence(record: ValidationEvidenceRecord) -> ValidationEvidenceRecord:
    """Append ``record`` to the in-process dynamic registry and return it unchanged."""
    _REGISTRY.append(record)
    return record


def clear_registry() -> None:
    """Empty the in-process dynamic registry (``STATIC_EVIDENCE_CATALOG`` is unaffected)."""
    _REGISTRY.clear()


def registered_evidence() -> tuple[ValidationEvidenceRecord, ...]:
    """Return every dynamically registered record, in registration order."""
    return tuple(_REGISTRY)


def validation_evidence(
    *,
    tier: ValidationTier,
    capability: str,
    summary: str,
    limitations: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a test/benchmark function: record tier evidence only once it actually passes.

    The wrapped function still runs (and its return value still passes through) exactly as before --
    this only adds a side effect. Evidence is appended to the dynamic registry (``origin=
    "dynamic_run"``) after the call returns; an exception propagates unchanged and nothing is
    registered, so a failing or skipped test cannot silently claim evidence.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            register_evidence(
                ValidationEvidenceRecord(
                    capability=capability,
                    tier=tier,
                    source=f"{func.__module__}.{func.__qualname__}",
                    summary=summary,
                    origin="dynamic_run",
                    limitations=limitations,
                )
            )
            return result

        return wrapper

    return decorator


def _all_records() -> tuple[ValidationEvidenceRecord, ...]:
    return STATIC_EVIDENCE_CATALOG + registered_evidence()


def evidence_for_capability(capability: str) -> tuple[ValidationEvidenceRecord, ...]:
    """Every record (static or dynamic) tagged against ``capability``, in catalog-then-run order."""
    return tuple(record for record in _all_records() if record.capability == capability)


def tiers_for_capability(capability: str) -> frozenset[ValidationTier]:
    """Which tier(s) of validation evidence exist for ``capability``. Empty if none are recorded."""
    return frozenset(record.tier for record in evidence_for_capability(capability))


def known_capabilities() -> tuple[str, ...]:
    """Every capability identifier with at least one recorded record, sorted for stable output."""
    return tuple(sorted({record.capability for record in _all_records()}))


@dataclass(frozen=True)
class CapabilityEvidenceReport:
    """What tier(s) of validation evidence a capability has, and every record behind them."""

    capability: str
    tiers: frozenset[ValidationTier]
    records: tuple[ValidationEvidenceRecord, ...]

    def semantics(self) -> tuple[TierSemantics, ...]:
        """``semantics_for`` of every tier present, in :data:`TIER_ORDER`."""
        return tuple(semantics_for(tier) for tier in TIER_ORDER if tier in self.tiers)


def evidence_report(capability: str) -> CapabilityEvidenceReport:
    """Build a :class:`CapabilityEvidenceReport` for ``capability`` from all known evidence."""
    records = evidence_for_capability(capability)
    return CapabilityEvidenceReport(
        capability=capability,
        tiers=frozenset(record.tier for record in records),
        records=records,
    )
