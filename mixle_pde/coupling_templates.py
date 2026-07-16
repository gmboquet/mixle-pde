"""Coupling-template catalog and interface-balance diagnostics (MP-G8).

Source: notes/mixle-pde-ai-native-multiphysics-work-plan.md workstream G ("Multiphysics coupling and
field exchange"), MP-G8 ("Coupling template planner and interface diagnostics"). The M2 reconciliation
ledger (docs/reconciliation/mp-task-ledger.md) records MP-G8 as not-started: "No coupling-template
catalog or interface-balance-receipt planner found anywhere." This module fills exactly that gap.

Scope and what this deliberately is not
----------------------------------------
This is a planning/diagnostic layer, not a solver -- it never assembles a system, never integrates a
PDE, and never runs an iteration loop. Two independent pieces:

1. :func:`coupling_template_catalog` / :func:`get_coupling_template` -- a small, typed lookup of named
   coupling *patterns* (one-way, monolithic, partitioned-Dirichlet-Neumann, partitioned-quasi-Newton).
   Each :class:`CouplingTemplate` declares the port roles it expects participants to expose at a shared
   interface and the convergence policy that pattern requires -- a structural description a scenario
   can be checked against (:func:`check_scenario_against_template`), not an executable coupling engine.
2. :func:`evaluate_interface_balance` -- given two participants' already-computed states sampled at a
   shared interface, compute a typed pass/fail/unknown :class:`InterfaceBalanceReceipt`: a continuity
   (field-jump) residual and a conservation (flux-mismatch) residual. This is deliberately
   physics-agnostic -- it takes plain arrays and never imports a solver, mesh, or specific-physics
   module, so it applies equally to a thermal interface, a fluid-structure interface, or an
   electromagnetic one.

Distinct from adjacent work, so as not to duplicate it:

- Not :mod:`mixle_pde.multiphysics_reference` (PR #86, formerly #71) -- that module executes one
  specific coupled scenario (monolithic/partitioned composite-heat, two conducting bars) end to end,
  including FEM assembly, a Bayesian inversion, and a surrogate. This module has no scenario, no mesh,
  and no solver; it is the general-purpose planning layer a scenario like that could be checked
  against, not another instance of it.
- Not MP-G7 (global equations / 0D-1D / network coupling) -- lumped-parameter and network topology are
  out of scope here; the templates below describe field-exchange patterns between already-discretized
  continuum participants.
- Not MP-G2 (mesh-to-mesh field mapping) -- :func:`evaluate_interface_balance` assumes its two inputs
  already sample the identical set of interface points (positional correspondence); it performs no
  interpolation or projection between non-matching interface traces.
- Not MP-G6 (typed conjugate ports / interface ontology, owned by mixle-physics' semantics.py) -- the
  port roles here are a minimal local vocabulary (dirichlet/neumann) sized to what the catalog and the
  balance receipt need, not a full physical-quantity ontology.

Neither half of this module reads or writes mixle_pde/io/artifacts.py,
mixle_pde/verification/capability_inventory.py, or mixle_pde/pde_backend_registry.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "ConvergencePolicy",
    "CouplingPlanCheck",
    "CouplingTemplate",
    "CouplingTemplateError",
    "InterfaceBalanceReceipt",
    "InterfaceVerdict",
    "ParticipantInterfaceState",
    "PortRequirement",
    "PortRole",
    "check_scenario_against_template",
    "coupling_template_catalog",
    "evaluate_interface_balance",
    "get_coupling_template",
]


class PortRole(str, Enum):
    """The two conjugate roles a field can play at a shared coupling interface.

    A well-posed interface pairs exactly one DIRICHLET role (a prescribed value one participant hands
    to the other) with one NEUMANN role (the conjugate flux/reaction the receiving participant
    computes and returns) -- this is the minimal vocabulary the templates and the balance receipt in
    this module need, not a general physical-quantity ontology.
    """

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class ConvergencePolicy(str, Enum):
    """What a coupling template requires of the (external) execution loop, if anything."""

    NONE = "none"
    EXACT = "exact"
    FIXED_POINT = "fixed-point"
    QUASI_NEWTON = "quasi-newton"


class InterfaceVerdict(str, Enum):
    """A typed pass/fail/unknown verdict -- never a fabricated boolean when a check could not run."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class CouplingTemplateError(KeyError):
    """Raised by :func:`get_coupling_template` when the requested template name is not catalogued."""


@dataclass(frozen=True)
class PortRequirement:
    """One field a coupling template expects a participant to expose at the shared interface.

    ``field_name`` is the physical quantity being exchanged (e.g. "temperature", "displacement",
    "potential"); ``role`` says whether the participant supplies it as a prescribed (Dirichlet) value
    or returns it as a computed (Neumann) flux/reaction -- see :class:`PortRole`. ``unit`` is optional
    and purely informational: this module never converts or validates units against it.
    """

    field_name: str
    role: PortRole
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("PortRequirement.field_name must be non-empty.")
        object.__setattr__(self, "role", PortRole(self.role))


@dataclass(frozen=True)
class CouplingTemplate:
    """A named, reusable coupling pattern: the ports it needs and the convergence policy it requires.

    This is a typed catalog entry, not a solver -- it declares the *shape* of a valid coupling (how
    many participants, what each exposes at the interface, what convergence discipline the pattern
    implies) so a proposed scenario can be checked against it before any physics is wired up.
    ``requires_relaxation`` flags patterns whose fixed-point iteration is known to need relaxation or
    acceleration to converge in practice (e.g. strong added-mass effects), independent of
    ``convergence_policy`` itself.
    """

    name: str
    description: str
    participant_count: int
    ports: tuple[PortRequirement, ...]
    convergence_policy: ConvergencePolicy
    requires_relaxation: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CouplingTemplate.name must be non-empty.")
        if self.participant_count < 2:
            raise ValueError("CouplingTemplate.participant_count must be >= 2 (coupling needs at least two).")
        if not self.ports:
            raise ValueError("CouplingTemplate must declare at least one port requirement.")
        object.__setattr__(self, "convergence_policy", ConvergencePolicy(self.convergence_policy))


def coupling_template_catalog() -> dict[str, CouplingTemplate]:
    """The full named catalog of coupling templates, keyed by template name.

    Four baseline patterns, matching the taxonomy already declared as schema -- but never catalogued
    together with concrete port/convergence requirements -- by mixle-sim's
    ``programs.py::CouplingStrategy`` (MONOLITHIC/PARTITIONED enum tags, MP-G3/MP-G4): one-way,
    monolithic, and two partitioned variants distinguished by their fixed-point acceleration. Not
    exhaustive -- multirate and network/global-equation coupling are separate, not-yet-catalogued
    patterns (MP-G5, MP-G7).
    """

    templates = (
        CouplingTemplate(
            name="one-way",
            description=(
                "One participant drives the other with no feedback (e.g. a temperature field driving "
                "thermal stress); the driven participant's response never influences the driver."
            ),
            participant_count=2,
            ports=(
                PortRequirement(field_name="driving_field", role=PortRole.DIRICHLET),
                PortRequirement(field_name="driven_response", role=PortRole.NEUMANN),
            ),
            convergence_policy=ConvergencePolicy.NONE,
        ),
        CouplingTemplate(
            name="monolithic",
            description=(
                "All participants' unknowns are assembled into one block system and solved "
                "simultaneously; the interface is satisfied exactly by construction, not iterated."
            ),
            participant_count=2,
            ports=(
                PortRequirement(field_name="interface_state", role=PortRole.DIRICHLET),
                PortRequirement(field_name="interface_state", role=PortRole.NEUMANN),
            ),
            convergence_policy=ConvergencePolicy.EXACT,
        ),
        CouplingTemplate(
            name="partitioned-dirichlet-neumann",
            description=(
                "Two participants solved separately per iteration: one takes the shared field as a "
                "prescribed Dirichlet condition, the other returns the conjugate flux as a Neumann "
                "condition; iterate (typically under-relaxed) to a fixed point."
            ),
            participant_count=2,
            ports=(
                PortRequirement(field_name="interface_value", role=PortRole.DIRICHLET),
                PortRequirement(field_name="interface_flux", role=PortRole.NEUMANN),
            ),
            convergence_policy=ConvergencePolicy.FIXED_POINT,
            requires_relaxation=True,
        ),
        CouplingTemplate(
            name="partitioned-quasi-newton",
            description=(
                "Partitioned Dirichlet-Neumann coupling accelerated with a quasi-Newton update (e.g. "
                "IQN-ILS) instead of fixed relaxation, for stiff couplings (e.g. strong added-mass) "
                "where plain fixed-point iteration converges slowly or not at all."
            ),
            participant_count=2,
            ports=(
                PortRequirement(field_name="interface_value", role=PortRole.DIRICHLET),
                PortRequirement(field_name="interface_flux", role=PortRole.NEUMANN),
            ),
            convergence_policy=ConvergencePolicy.QUASI_NEWTON,
        ),
    )
    return {template.name: template for template in templates}


def get_coupling_template(name: str) -> CouplingTemplate:
    """Look up a named coupling template. Raises :class:`CouplingTemplateError` if the name is unknown."""

    catalog = coupling_template_catalog()
    try:
        return catalog[name]
    except KeyError:
        raise CouplingTemplateError(
            f"no coupling template named {name!r}; known templates: {sorted(catalog)}"
        ) from None


@dataclass(frozen=True)
class CouplingPlanCheck:
    """Whether a proposed scenario's declared interface roles structurally satisfy a named template.

    A structural check only: it compares multisets of :class:`PortRole` values and knows nothing about
    what specific physics (heat, elasticity, electromagnetics, ...) the fields carry.
    """

    template_name: str
    satisfied: bool
    required_roles: tuple[str, ...]
    declared_roles: tuple[str, ...]
    detail: str


def check_scenario_against_template(
    template_name: str,
    declared_field_roles: Sequence[PortRole | str],
) -> CouplingPlanCheck:
    """Check whether a scenario's declared field roles satisfy a named coupling template's ports.

    ``declared_field_roles`` is what a scenario author states they will exchange at the interface
    (e.g. ``["dirichlet", "neumann"]``); order does not matter but multiplicity does -- every template
    catalogued today requires exactly one dirichlet and one neumann role.
    """

    template = get_coupling_template(template_name)
    required = tuple(sorted(port.role.value for port in template.ports))
    declared = tuple(sorted(PortRole(role).value for role in declared_field_roles))
    satisfied = required == declared
    detail = (
        f"scenario satisfies template {template_name!r}."
        if satisfied
        else f"template {template_name!r} requires roles {list(required)}, scenario declared {list(declared)}."
    )
    return CouplingPlanCheck(
        template_name=template_name,
        satisfied=satisfied,
        required_roles=required,
        declared_roles=declared,
        detail=detail,
    )


@dataclass(frozen=True)
class ParticipantInterfaceState:
    """One participant's already-computed state sampled at a shared coupling interface.

    ``value`` is the participant's own primary field at the interface points (e.g. its temperature,
    displacement, potential); ``flux`` is the conjugate quantity it computes there (e.g. heat flux,
    traction, current density), reported using each participant's own outward-normal sign convention
    (flux leaving a participant is positive) -- a well-posed interface then requires opposite-signed,
    equal-magnitude flux between the two participants, not equal-signed. Point correspondence with a
    second state is purely positional: ``value[i]``/``flux[i]`` must sample the same physical
    interface point as the other participant's ``value[i]``/``flux[i]``; this dataclass does not
    itself verify that (only :func:`evaluate_interface_balance`, comparing two states, can).
    """

    participant_id: str
    value: np.ndarray
    flux: np.ndarray
    unit_value: str | None = None
    unit_flux: str | None = None

    def __post_init__(self) -> None:
        if not self.participant_id:
            raise ValueError("ParticipantInterfaceState.participant_id must be non-empty.")
        value = np.asarray(self.value, dtype=float).reshape(-1)
        flux = np.asarray(self.flux, dtype=float).reshape(-1)
        if value.shape != flux.shape:
            raise ValueError("ParticipantInterfaceState.value and .flux must have the same shape.")
        if value.size == 0:
            raise ValueError("ParticipantInterfaceState must sample at least one interface point.")
        if not (np.all(np.isfinite(value)) and np.all(np.isfinite(flux))):
            raise ValueError("ParticipantInterfaceState.value and .flux must be finite.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "flux", flux)


@dataclass(frozen=True)
class InterfaceBalanceReceipt:
    """A typed pass/fail/unknown verdict for conservation/continuity at a shared coupling interface.

    Two residuals, independent of which specific physics is coupled:

    - ``jump_residual``: max absolute difference between the two participants' interface ``value`` --
      nonzero means the two disagree on the shared field (a continuity/jump violation).
    - ``flux_mismatch_residual``: max absolute value of the two participants' interface ``flux``
      *summed* -- nonzero means the conjugate fluxes do not cancel under the outward-normal
      convention (a conservation violation).

    ``verdict`` is PASS when both residuals are within their tolerances, FAIL when either tolerance is
    exceeded, and UNKNOWN when the two states cannot be meaningfully compared at all (mismatched point
    counts, or incompatible declared units) -- never silently coerced to FAIL, so a caller can tell
    "checked and it is broken" apart from "could not check."
    """

    participant_a: str
    participant_b: str
    jump_residual: float
    flux_mismatch_residual: float
    jump_tolerance: float
    flux_tolerance: float
    verdict: InterfaceVerdict
    detail: str


def evaluate_interface_balance(
    state_a: ParticipantInterfaceState,
    state_b: ParticipantInterfaceState,
    *,
    jump_tolerance: float = 1e-8,
    flux_tolerance: float = 1e-8,
) -> InterfaceBalanceReceipt:
    """Compute a typed continuity/conservation verdict between two participants' interface states.

    Point correspondence is assumed positional -- this function performs no mesh-to-mesh
    interpolation or projection (a separate concern, MP-G2); resample both states onto a common
    interface trace before calling this. When the two states have different point counts, or declare
    different non-``None`` units for the same channel, no comparison is possible, so the verdict is
    UNKNOWN rather than a guessed FAIL or PASS.
    """

    if jump_tolerance < 0 or flux_tolerance < 0:
        raise ValueError("jump_tolerance and flux_tolerance must both be >= 0.")
    if state_a.participant_id == state_b.participant_id:
        raise ValueError("evaluate_interface_balance requires two distinctly identified participants.")

    def _unknown(detail: str) -> InterfaceBalanceReceipt:
        return InterfaceBalanceReceipt(
            participant_a=state_a.participant_id,
            participant_b=state_b.participant_id,
            jump_residual=float("nan"),
            flux_mismatch_residual=float("nan"),
            jump_tolerance=jump_tolerance,
            flux_tolerance=flux_tolerance,
            verdict=InterfaceVerdict.UNKNOWN,
            detail=detail,
        )

    if state_a.value.shape != state_b.value.shape:
        return _unknown(
            f"cannot compare: {state_a.participant_id!r} samples {state_a.value.size} interface "
            f"point(s), {state_b.participant_id!r} samples {state_b.value.size}; resample onto a "
            "common interface trace before evaluating balance."
        )
    for channel, unit_a, unit_b in (
        ("value", state_a.unit_value, state_b.unit_value),
        ("flux", state_a.unit_flux, state_b.unit_flux),
    ):
        if unit_a is not None and unit_b is not None and unit_a != unit_b:
            return _unknown(
                f"cannot compare {channel}: {state_a.participant_id!r} reports unit {unit_a!r}, "
                f"{state_b.participant_id!r} reports unit {unit_b!r}; this module performs no unit "
                "conversion."
            )

    jump_residual = float(np.max(np.abs(state_a.value - state_b.value)))
    flux_mismatch_residual = float(np.max(np.abs(state_a.flux + state_b.flux)))
    passed = jump_residual <= jump_tolerance and flux_mismatch_residual <= flux_tolerance
    detail = (
        f"jump_residual={jump_residual:.3e} (tolerance {jump_tolerance:.3e}), "
        f"flux_mismatch_residual={flux_mismatch_residual:.3e} (tolerance {flux_tolerance:.3e})"
    )
    return InterfaceBalanceReceipt(
        participant_a=state_a.participant_id,
        participant_b=state_b.participant_id,
        jump_residual=jump_residual,
        flux_mismatch_residual=flux_mismatch_residual,
        jump_tolerance=jump_tolerance,
        flux_tolerance=flux_tolerance,
        verdict=InterfaceVerdict.PASS if passed else InterfaceVerdict.FAIL,
        detail=detail,
    )
