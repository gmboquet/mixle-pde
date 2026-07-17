"""Characteristic scales, nondimensionalization, and automatic solver-hint advisories (MP-F6).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``) records MP-F6 ("Scaling,
nondimensionalization, and automatic solver hints") as ``not-started``: "No nondimensionalization/
scaling module in mixle-pde, mixle-sim, or mixle-discrete." This module is a first, baseline slice of
that capability -- matching the scope other MP-* baseline modules in this repo have shipped (e.g.
``global_sensitivity.py``/MP-I11, ``reduced_basis.py``), not the full multi-week work-plan item (which
also covers residual-block scaling and live solver-family selection; see
``notes/mixle-pde-ai-native-multiphysics-work-plan.md`` MP-F6).

Three pieces, matching the work-plan's own description ("Compute characteristic scales and
dimensionless groups where declared; scale variables/residual blocks and recommend solver families
without changing model semantics"):

* :class:`CharacteristicScales` -- the modeller's declared reference magnitudes (``length`` is the
  only required field; ``time``/``velocity``/``pressure``/``diffusivity``/``reaction_rate``/
  ``viscosity``/``density`` are declared only when the corresponding physical term is actually present
  in the model being scaled). Declaring a scale never changes model semantics by itself -- it only sets
  the yardstick the rest of this module measures against. An undeclared scale is never silently assumed
  to be ``1.0`` or any other fabricated magnitude: functions that need one raise a clear ``ValueError``.
* :func:`nondimensionalize` / :func:`redimensionalize` -- a lossless round-trip pair that rescales a
  :class:`PDEParameterSet` (a named, unit-tagged bag of dimensional PDE coefficients) into and back out
  of dimensionless form.
* :func:`recommend_solver_hints` -- computes named dimensionless groups (Peclet/Reynolds/Damkohler-style
  ratios) from whichever scales were declared, and turns an extreme ratio into a typed
  :class:`SolverAdvisory` flagging that one term is likely negligible or stiff relative to another.
  Advisories are strictly informational: nothing in this module calls a solver, selects or mutates a
  :class:`mixle_pde.pde_backend_registry.PDEKernelRegistration`/``PDEBackendProfile``, or edits a
  ``solve_plan`` -- see :class:`SolverAdvisory` for the same note repeated at the type that matters.

This module is solver-agnostic (it never imports a PDE kernel itself), the same shape
``reduced_basis.py``/``global_sensitivity.py`` already use for a caller-supplied-data capability.
``tests/scaling_test.py`` is what grounds it against a real kernel: it reads
:mod:`mixle_pde.pde_backend_registry`'s registered ``"transport-fd-advdiff"`` profile (which wraps
:class:`mixle_pde.dynamics.AdvectionDiffusionOperator`, ports ``diffusivity`` in ``m^2/s`` and
``velocity`` in ``m/s``) and that kernel's own invoker defaults (``diffusivity=0.01``, ``velocity=0.5``,
``length=1.0``) to build a real :class:`PDEParameterSet`. Those numbers give a domain-scale Peclet
number of ``0.5 * 1.0 / 0.01 = 50`` -- strongly advection-dominated, so :func:`recommend_solver_hints`
reports the ``advection_dominated`` advisory. This repo already treats that regime as meaningful for the
same operator family (``mixle_pde/reactive_transport.py``'s docstring mentions "Peclet/CFL numbers", and
``tests/test_amd.py`` documents its own ``AdvectionDiffusionOperator`` column as "strongly
advection-dominated (grid Peclet ~ 1)"); those are a *grid* (cell-spacing) Peclet number at a specific
mesh resolution, a different, generally much smaller quantity than this module's *domain-scale* Peclet
number, so the two are not numerically the same value -- only evidence for the same underlying regime
being one this codebase already reasons about.

Threshold note: the Peclet/Reynolds/Damkohler cutoffs :func:`recommend_solver_hints` uses are
illustrative order-of-magnitude flags ("this term looks negligible or dominant relative to the other"),
not calibrated physical transition points (a real inertial-to-viscous transition Reynolds number is
geometry-dependent and typically far larger than the round number used here). Per this project's "no
100%-realistic physics claim" convention, this module never claims otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CharacteristicScales",
    "PDEParameter",
    "PDEParameterSet",
    "DimensionlessParameter",
    "DimensionlessParameterSet",
    "DimensionlessGroup",
    "SolverAdvisory",
    "SolverHintReport",
    "nondimensionalize",
    "redimensionalize",
    "peclet_number",
    "reynolds_number",
    "damkohler_number",
    "dimensionless_groups",
    "recommend_solver_hints",
]

# Every dimension a PDEParameter may be tagged with, other than "1" (already dimensionless) -- each
# name is exactly a CharacteristicScales field. Adding a new supported dimension means adding one field
# there and one entry here; nondimensionalize/redimensionalize stay generic over whatever is declared.
_DIMENSIONLESS = "1"
_SCALE_FIELDS: tuple[str, ...] = (
    "length",
    "time",
    "velocity",
    "pressure",
    "diffusivity",
    "reaction_rate",
    "viscosity",
    "density",
)


def _validate_scale(name: str, value: float | None, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"CharacteristicScales.{name} is required.")
        return
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"CharacteristicScales.{name} must be a finite positive value; got {value!r}.")


@dataclass(frozen=True)
class CharacteristicScales:
    """The modeller's declared reference magnitudes (SI units) for one PDE study.

    ``length`` is required -- every registered kernel in :mod:`mixle_pde.pde_backend_registry` has a
    spatial extent, so it is the one scale every :class:`PDEParameter` unit below can always be measured
    against. Every other field defaults to ``None`` ("not declared"), which is not the same as
    declaring it ``1.0``: :func:`nondimensionalize` and :func:`dimensionless_groups` raise a clear error
    for a parameter or group that needs an undeclared scale, rather than assuming a magnitude.
    """

    length: float
    time: float | None = None
    velocity: float | None = None
    pressure: float | None = None
    diffusivity: float | None = None
    reaction_rate: float | None = None
    viscosity: float | None = None
    density: float | None = None

    def __post_init__(self) -> None:
        _validate_scale("length", self.length, required=True)
        for name in _SCALE_FIELDS[1:]:
            _validate_scale(name, getattr(self, name), required=False)


def _scale_factor(dimension: str, label: str, scales: CharacteristicScales) -> float:
    """The characteristic magnitude that divides a value tagged ``dimension`` to make it dimensionless."""
    if dimension == _DIMENSIONLESS:
        return 1.0
    value = getattr(scales, dimension)
    if value is None:
        raise ValueError(f"{label!r} needs CharacteristicScales.{dimension} to be declared (currently unset).")
    return value


def _validate_dimension(dimension: str, label: str) -> None:
    if dimension != _DIMENSIONLESS and dimension not in _SCALE_FIELDS:
        raise ValueError(
            f"{label!r} has unknown dimension {dimension!r}; expected one of {_SCALE_FIELDS!r} or {_DIMENSIONLESS!r}."
        )


@dataclass(frozen=True)
class PDEParameter:
    """One named, dimensional PDE coefficient.

    ``units`` is a display/provenance string using the same SI (or ``"1"`` for dimensionless)
    convention :class:`mixle_pde.pde_backend_registry.PDEPort` already uses, so a parameter can be built
    directly from a registered kernel's declared port. ``dimension`` is the field of
    :class:`CharacteristicScales` that nondimensionalizes this value (or ``"1"`` if it already is
    dimensionless) -- kept separate from ``units`` because a unit string alone is ambiguous (e.g.
    diffusivity and kinematic viscosity are both ``m^2/s``); ``dimension`` resolves that explicitly
    rather than by guessing from the parameter's name.
    """

    name: str
    value: float
    units: str
    dimension: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"PDEParameter {self.name!r} value must be finite; got {self.value!r}.")
        _validate_dimension(self.dimension, self.name)


@dataclass(frozen=True)
class PDEParameterSet:
    """A named bag of dimensional PDE coefficients for one study."""

    parameters: tuple[PDEParameter, ...]

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"PDEParameterSet has duplicate parameter names: {names!r}.")

    def as_mapping(self) -> dict[str, float]:
        """Return ``{name: value}``, discarding units/dimension -- for plain-number call sites."""
        return {p.name: p.value for p in self.parameters}

    def get(self, name: str) -> PDEParameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"PDEParameterSet has no parameter named {name!r}.")


@dataclass(frozen=True)
class DimensionlessParameter:
    """One :class:`PDEParameter` after :func:`nondimensionalize`, with enough provenance to invert it."""

    name: str
    value: float
    source_units: str
    source_dimension: str


@dataclass(frozen=True)
class DimensionlessParameterSet:
    """The output of :func:`nondimensionalize`; feed straight back into :func:`redimensionalize`."""

    parameters: tuple[DimensionlessParameter, ...]

    def as_mapping(self) -> dict[str, float]:
        return {p.name: p.value for p in self.parameters}


def nondimensionalize(params: PDEParameterSet, scales: CharacteristicScales) -> DimensionlessParameterSet:
    """Rescale every parameter in ``params`` by its declared dimension's magnitude in ``scales``.

    Never mutates ``params`` (both are frozen dataclasses); returns a fresh
    :class:`DimensionlessParameterSet`. Raises ``ValueError`` if a parameter needs a scale ``scales``
    did not declare -- there is no silent fallback to an assumed magnitude.
    """
    return DimensionlessParameterSet(
        parameters=tuple(
            DimensionlessParameter(
                name=p.name,
                value=p.value / _scale_factor(p.dimension, p.name, scales),
                source_units=p.units,
                source_dimension=p.dimension,
            )
            for p in params.parameters
        )
    )


def redimensionalize(dimensionless: DimensionlessParameterSet, scales: CharacteristicScales) -> PDEParameterSet:
    """Invert :func:`nondimensionalize`: multiply each dimensionless value back by its stored scale.

    ``scales`` should be the same declaration used to nondimensionalize (or an equivalent one) -- this
    function trusts the caller; it does not store or compare a fingerprint of the original scales.
    """
    return PDEParameterSet(
        parameters=tuple(
            PDEParameter(
                name=p.name,
                value=p.value * _scale_factor(p.source_dimension, p.name, scales),
                units=p.source_units,
                dimension=p.source_dimension,
            )
            for p in dimensionless.parameters
        )
    )


# ---------------------------------------------------------------------------
# Named dimensionless groups (Peclet / Reynolds / Damkohler) and solver-hint advisories
# ---------------------------------------------------------------------------


def peclet_number(*, velocity: float, length: float, diffusivity: float) -> float:
    """Advective-to-diffusive transport ratio ``Pe = |velocity| * length / diffusivity``.

    ``Pe >> 1`` means advection dominates diffusion over ``length`` -- the regime
    :mod:`mixle_pde.dynamics`'s ``AdvectionDiffusionOperator`` discretizes its advective term with an
    upwind (not central) difference for; ``Pe << 1`` means diffusion dominates and advection is likely
    negligible over that length.
    """
    if not math.isfinite(diffusivity) or diffusivity <= 0.0:
        raise ValueError(f"peclet_number needs a finite positive diffusivity; got {diffusivity!r}.")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"peclet_number needs a finite positive length; got {length!r}.")
    if not math.isfinite(velocity):
        raise ValueError(f"peclet_number needs a finite velocity; got {velocity!r}.")
    return abs(velocity) * length / diffusivity


def reynolds_number(*, velocity: float, length: float, viscosity: float) -> float:
    """Inertial-to-viscous ratio ``Re = |velocity| * length / viscosity`` (kinematic viscosity, m^2/s)."""
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError(f"reynolds_number needs a finite positive (kinematic) viscosity; got {viscosity!r}.")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"reynolds_number needs a finite positive length; got {length!r}.")
    if not math.isfinite(velocity):
        raise ValueError(f"reynolds_number needs a finite velocity; got {velocity!r}.")
    return abs(velocity) * length / viscosity


def damkohler_number(*, reaction_rate: float, length: float, velocity: float) -> float:
    """Reaction-to-advective-transport ratio ``Da = reaction_rate * length / |velocity|``.

    ``Da >> 1`` means the reaction equilibrates far faster than material is transported across
    ``length`` (the reaction step is stiff relative to the transport step -- compare
    :func:`mixle_pde.dynamics.integrate_stiff`, already used for exactly this kind of regime elsewhere
    in this repo); ``Da << 1`` means the reaction term is likely negligible over one transport time.
    """
    if not math.isfinite(velocity) or velocity == 0.0:
        raise ValueError(f"damkohler_number needs a finite nonzero advective velocity; got {velocity!r}.")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"damkohler_number needs a finite positive length; got {length!r}.")
    if not math.isfinite(reaction_rate) or reaction_rate < 0.0:
        raise ValueError(f"damkohler_number needs a finite non-negative reaction_rate; got {reaction_rate!r}.")
    return reaction_rate * length / abs(velocity)


@dataclass(frozen=True)
class DimensionlessGroup:
    """One named dimensionless ratio derived from a :class:`CharacteristicScales` declaration."""

    name: str
    value: float
    definition: str


def dimensionless_groups(scales: CharacteristicScales) -> tuple[DimensionlessGroup, ...]:
    """Compute whichever named ratios (Peclet/Reynolds/Damkohler) the declared scales support.

    Each group needs ``length`` plus specific optional scales (Peclet: ``velocity``+``diffusivity``;
    Reynolds: ``velocity``+``viscosity``; Damkohler: ``velocity``+``reaction_rate``). A group whose
    required scales were not all declared is simply omitted from the result -- never computed from a
    fabricated default.
    """
    groups: list[DimensionlessGroup] = []
    if scales.velocity is not None and scales.diffusivity is not None:
        groups.append(
            DimensionlessGroup(
                name="peclet",
                value=peclet_number(velocity=scales.velocity, length=scales.length, diffusivity=scales.diffusivity),
                definition="abs(velocity) * length / diffusivity",
            )
        )
    if scales.velocity is not None and scales.viscosity is not None:
        groups.append(
            DimensionlessGroup(
                name="reynolds",
                value=reynolds_number(velocity=scales.velocity, length=scales.length, viscosity=scales.viscosity),
                definition="abs(velocity) * length / viscosity",
            )
        )
    if scales.velocity is not None and scales.reaction_rate is not None:
        groups.append(
            DimensionlessGroup(
                name="damkohler",
                value=damkohler_number(
                    reaction_rate=scales.reaction_rate, length=scales.length, velocity=scales.velocity
                ),
                definition="reaction_rate * length / abs(velocity)",
            )
        )
    return tuple(groups)


@dataclass(frozen=True)
class SolverAdvisory:
    """A typed, non-binding scaling advisory -- never a silent solver-behavior change.

    Nothing in :mod:`mixle_pde.scaling` calls a solver, selects a
    :mod:`mixle_pde.pde_backend_registry` backend, or edits a ``solve_plan``;
    :func:`recommend_solver_hints` only ever returns data for a caller to act on (or ignore).
    ``severity`` is one of ``"info"``/``"advisory"`` (a typed outcome, never a bare boolean, per this
    project's convention) -- there is no ``"critical"`` level, because this module has no way to know
    whether a given caller's solver can actually handle the flagged regime.
    """

    code: str
    severity: str
    group_name: str
    group_value: float
    message: str

    def __post_init__(self) -> None:
        if self.severity not in ("info", "advisory"):
            raise ValueError(f"SolverAdvisory.severity must be 'info' or 'advisory'; got {self.severity!r}.")


@dataclass(frozen=True)
class SolverHintReport:
    """The full output of :func:`recommend_solver_hints`: every computed group plus any advisories."""

    groups: tuple[DimensionlessGroup, ...]
    advisories: tuple[SolverAdvisory, ...]

    def groups_by_name(self) -> dict[str, float]:
        return {g.name: g.value for g in self.groups}


# Illustrative order-of-magnitude thresholds -- not calibrated physical transition points. See the
# module docstring's "Threshold note".
_PECLET_ADVECTION_DOMINATED = 10.0
_PECLET_DIFFUSION_DOMINATED = 0.1
_REYNOLDS_INERTIA_DOMINATED = 1.0
_DAMKOHLER_REACTION_STIFF = 10.0
_DAMKOHLER_REACTION_NEGLIGIBLE = 0.1


def recommend_solver_hints(scales: CharacteristicScales) -> SolverHintReport:
    """Compute dimensionless groups and turn an extreme ratio into a typed, informational advisory.

    This never selects a backend, mutates a
    :class:`~mixle_pde.pde_backend_registry.PDEKernelRegistration`, or edits a ``solve_plan`` --
    :class:`SolverAdvisory` is read-only guidance.
    """
    groups = dimensionless_groups(scales)
    advisories: list[SolverAdvisory] = []
    for group in groups:
        if group.name == "peclet":
            if group.value >= _PECLET_ADVECTION_DOMINATED:
                advisories.append(
                    SolverAdvisory(
                        code="advection_dominated",
                        severity="advisory",
                        group_name="peclet",
                        group_value=group.value,
                        message=(
                            f"Peclet={group.value:.3g} >= {_PECLET_ADVECTION_DOMINATED:g}: advection "
                            "dominates diffusion over the declared length scale. A central-difference "
                            "diffusion discretization can be unstable/oscillatory here; consider an "
                            "upwind or flux-limited scheme and an implicit (not explicit-Euler) time "
                            "integrator."
                        ),
                    )
                )
            elif group.value <= _PECLET_DIFFUSION_DOMINATED:
                advisories.append(
                    SolverAdvisory(
                        code="diffusion_dominated",
                        severity="info",
                        group_name="peclet",
                        group_value=group.value,
                        message=(
                            f"Peclet={group.value:.3g} <= {_PECLET_DIFFUSION_DOMINATED:g}: the advective "
                            "term is likely negligible relative to diffusion over the declared length "
                            "scale."
                        ),
                    )
                )
        elif group.name == "reynolds" and group.value >= _REYNOLDS_INERTIA_DOMINATED:
            advisories.append(
                SolverAdvisory(
                    code="inertia_dominated",
                    severity="advisory",
                    group_name="reynolds",
                    group_value=group.value,
                    message=(
                        f"Reynolds={group.value:.3g}: inertial transport is not negligible relative to "
                        "viscous diffusion over the declared length scale; an explicit viscous step may "
                        "need a restrictively small time step."
                    ),
                )
            )
        elif group.name == "damkohler":
            if group.value >= _DAMKOHLER_REACTION_STIFF:
                advisories.append(
                    SolverAdvisory(
                        code="reaction_stiff",
                        severity="advisory",
                        group_name="damkohler",
                        group_value=group.value,
                        message=(
                            f"Damkohler={group.value:.3g} >= {_DAMKOHLER_REACTION_STIFF:g}: the reaction "
                            "term equilibrates much faster than transport over the declared length scale "
                            "(the reaction step is stiff relative to transport); consider an "
                            "implicit/stiff integrator for that term (see "
                            "mixle_pde.dynamics.integrate_stiff) rather than an explicit scheme sized for "
                            "the transport timescale alone."
                        ),
                    )
                )
            elif group.value <= _DAMKOHLER_REACTION_NEGLIGIBLE:
                advisories.append(
                    SolverAdvisory(
                        code="reaction_negligible",
                        severity="info",
                        group_name="damkohler",
                        group_value=group.value,
                        message=(
                            f"Damkohler={group.value:.3g} <= {_DAMKOHLER_REACTION_NEGLIGIBLE:g}: the "
                            "reaction term is likely negligible relative to transport over the declared "
                            "length scale."
                        ),
                    )
                )
    return SolverHintReport(groups=groups, advisories=tuple(advisories))
