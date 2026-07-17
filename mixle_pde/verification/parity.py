"""Cross-kernel and reference parity harness for registered mixle-pde backends (MP-K4).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``, MP-K4) records: "No FEniCSx
or external-reference comparison harness found anywhere." A real external-tool comparison (FEniCSx,
OpenFOAM, ...) needs heavy dependencies this repo does not install, and standing this repo up against
one is a separate, much larger card. This module is the honestly-scoped internal version: a generic,
reusable harness (:class:`ParityCheck`) that runs any set of named solve callables against each other
and against a shared closed-form/manufactured reference, reporting two *typed, never-conflated* kinds
of record --

* :class:`KernelDiscrepancy` -- how much two independently-invoked solves disagree with each other
  (:data:`~mixle_pde.verification.validation_tiers.ValidationTier.CODE_TO_CODE`: relative agreement
  only, does not by itself establish either side is correct -- see that tier's documented confounds).
* :class:`KernelReferenceError` -- how much one solve differs from a closed-form/manufactured exact
  solution (:data:`~mixle_pde.verification.validation_tiers.ValidationTier.MANUFACTURED` or
  ``ANALYTIC``: a genuine error against known ground truth).

Real overlapping registered kernels. :mod:`mixle_pde.pde_backend_registry` registers nine legacy
kernels; reading it (read-only -- this module never edits it) for two that solve the same class of
problem surfaces exactly one genuine pair: ``transport-fd-advdiff``
(:class:`mixle_pde.dynamics.AdvectionDiffusionOperator`) and ``groundwater-fd-transport``
(:class:`mixle_pde.groundwater.GroundwaterTransportOperator`), both linear advection-diffusion
transport operators sharing the same ``mixle_pde.dynamics.laplacian_matrix``/
``upwind_gradient_matrix`` stencils -- the ``groundwater-fd-transport`` module's own docstring says so
directly: it "generalizes AdvectionDiffusionOperator's single scalar velocity to a per-cell velocity
field ... and adds velocity-dependent dispersion, first-order decay, and linear retardation on top."
Every other registered kernel models distinct physics (elastic vs. acoustic vs. electromagnetic wave
fields, a steady FEM Poisson solve with no FD counterpart, a walled-cavity FD flow solver whose
Dirichlet streamfunction boundary is structurally incompatible with the periodic-only spectral flow
kernel) -- see this module's PR description for the full pairwise scan. No other pair overlaps.

Because both registered kernels reduce to literally the same discrete operator in the parameter regime
this module drives them in (see :func:`groundwater_uniform_velocity_subject`), the honest expectation
is that :func:`transport_groundwater_parity_check` measures a kernel-vs-kernel discrepancy of (up to
floating-point round-off) exactly zero -- this is still a real, valuable regression guard: it fails
loudly the moment either registered invoker's parameter wiring, or the generalization's specialization
behavior, silently drifts apart. :func:`transport_fd_advdiff_subject` runs through the real,
compatibility-checked ``run_math_problem`` boundary; :func:`groundwater_uniform_velocity_subject`
drives the unmodified ``GroundwaterTransportOperator`` class directly, because the registered
``groundwater-fd-transport`` invoker only ever derives its velocity field from
:func:`mixle_pde.groundwater.darcy_velocity` (a Poisson-solve-driven field that is never exactly
spatially uniform) and exposes no parameter to request the class's own documented uniform-velocity
shorthand instead -- so it cannot be driven, through its registered entry point alone, to solve
literally the same problem as ``transport-fd-advdiff``. This is the same bypass
:mod:`mixle_pde.verification.mms` already uses for MP-K3, for the same class of reason (that module's
own docstring: "drives the same unmodified public building blocks the registered kernel wraps"), and
this module uses the same provenance-drift guard that one does (asserting the registration's ``source``
still points at the expected class) so the comparison stays honestly tied to the kernel it claims to
exercise.

Framework generality. :class:`ParityCheck` is not specific to this one pair -- ``subjects`` is any
mapping of label to a zero-argument :class:`ParitySubject` callable, and a single subject compared
only against ``reference`` (no second kernel at all) is a supported, tested degenerate case: exactly
the "framework-only, one kernel against a manufactured reference" fallback this card's brief allows
when no second overlapping kernel exists. Nothing here depends on which two kernels (or how many) are
supplied.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from mixle_pde.groundwater import GroundwaterTransportOperator
from mixle_pde.pde_backend_registry import get_kernel_registration, run_math_problem
from mixle_pde.verification.validation_tiers import ValidationTier

__all__ = [
    "ParitySample",
    "ParitySubject",
    "ReferenceFn",
    "KernelDiscrepancy",
    "KernelReferenceError",
    "ParityReport",
    "ParityCheck",
    "periodic_advection_diffusion_mode",
    "transport_fd_advdiff_subject",
    "groundwater_uniform_velocity_subject",
    "transport_groundwater_parity_check",
]

# The two registered kernels this module's worked example exercises, and the underlying class each
# one's registration must still point at -- a provenance-drift guard, exactly mirroring the one
# mixle_pde.verification.mms uses for the same reason (see the module docstring).
_TRANSPORT_BACKEND_ID = "transport-fd-advdiff"
_GROUNDWATER_BACKEND_ID = "groundwater-fd-transport"
_TRANSPORT_SOURCE = "mixle_pde.dynamics.AdvectionDiffusionOperator"
_GROUNDWATER_SOURCE = "mixle_pde.groundwater.GroundwaterTransportOperator"

_SCHEME_DISCRETIZATION = {"implicit": "FD-implicit", "explicit": "FD-explicit", "exact": "FD-exact"}


# ---------------------------------------------------------------------------
# Generic harness types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParitySample:
    """One field a :class:`ParitySubject` produced: named values at explicit coordinates.

    Coordinates are carried alongside the values (never assumed positional) so a
    :class:`ParityCheck` can refuse to compare two samples that were not actually evaluated at the
    same points, rather than silently diffing misaligned arrays.
    """

    label: str
    values: np.ndarray
    coordinates: np.ndarray

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("ParitySample.label must be non-empty.")
        if self.values.shape != self.coordinates.shape:
            raise ValueError(
                f"{self.label}: values shape {self.values.shape} must match coordinates shape {self.coordinates.shape}."
            )


class ParitySubject(Protocol):
    """A zero-argument solve entry point a :class:`ParityCheck` can run.

    Deliberately just a callable shape -- any function/closure/``functools.partial`` that returns a
    :class:`ParitySample` works, so a caller can wrap a registered mixle-pde kernel (see
    :func:`transport_fd_advdiff_subject`), a bare solver class, or (for the single-subject fallback
    this module also supports) anything else. Never stored on a frozen/serializable dataclass here --
    only the :class:`ParitySample` a subject *returns* ever becomes part of a typed record.
    """

    def __call__(self) -> ParitySample: ...


class ReferenceFn(Protocol):
    """A closed-form/manufactured reference: field values at arbitrary coordinates, no solver involved."""

    def __call__(self, coordinates: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class KernelDiscrepancy:
    """Kernel-vs-kernel: how much two independently-run subjects disagree on what the caller asserts
    is the same problem instance. This is :data:`ValidationTier.CODE_TO_CODE` evidence -- relative
    agreement only; it does not by itself establish that either subject is numerically correct (both
    could share a modeling assumption, a discretization family, or a common-ancestor bug -- see
    :func:`mixle_pde.verification.validation_tiers.semantics_for`).
    """

    label_a: str
    label_b: str
    coordinates_shape: tuple[int, ...]
    max_abs_discrepancy: float
    rms_discrepancy: float
    tier: ValidationTier = ValidationTier.CODE_TO_CODE

    def __post_init__(self) -> None:
        if self.tier is not ValidationTier.CODE_TO_CODE:
            raise ValueError(f"KernelDiscrepancy.tier must be ValidationTier.CODE_TO_CODE; got {self.tier}.")
        for value, name in (
            (self.max_abs_discrepancy, "max_abs_discrepancy"),
            (self.rms_discrepancy, "rms_discrepancy"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative; got {value}.")


@dataclass(frozen=True)
class KernelReferenceError:
    """Kernel-vs-reference: how much one subject's output differs from a closed-form/manufactured
    exact solution -- a genuine error against known ground truth, never conflated with
    :class:`KernelDiscrepancy` (two subjects can agree with each other while both disagreeing with
    the reference, or vice versa; this record and that one are always reported separately).
    """

    label: str
    reference_label: str
    coordinates_shape: tuple[int, ...]
    max_abs_error: float
    rms_error: float
    tier: ValidationTier

    def __post_init__(self) -> None:
        if self.tier not in (ValidationTier.MANUFACTURED, ValidationTier.ANALYTIC):
            raise ValueError(
                f"KernelReferenceError.tier must be MANUFACTURED or ANALYTIC (a closed-form reference "
                f"gives an exact error, not a code-to-code comparison); got {self.tier}."
            )
        for value, name in ((self.max_abs_error, "max_abs_error"), (self.rms_error, "rms_error")):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative; got {value}.")


@dataclass(frozen=True)
class ParityReport:
    """The full typed result of one :class:`ParityCheck` run.

    ``kernel_discrepancies`` holds one :class:`KernelDiscrepancy` per pair of subjects (empty if
    fewer than two subjects were supplied -- the single-kernel-vs-reference fallback case).
    ``reference_errors`` holds one :class:`KernelReferenceError` per subject (empty if no ``reference``
    was supplied). The two are always separate fields; nothing in this module ever averages or
    combines them into one number.
    """

    problem_label: str
    kernel_discrepancies: tuple[KernelDiscrepancy, ...]
    reference_errors: tuple[KernelReferenceError, ...]

    def __post_init__(self) -> None:
        if not self.problem_label.strip():
            raise ValueError("ParityReport.problem_label must be non-empty.")


def _compare_kernels(label_a: str, sample_a: ParitySample, label_b: str, sample_b: ParitySample) -> KernelDiscrepancy:
    if sample_a.values.shape != sample_b.values.shape:
        raise ValueError(
            f"cannot compare {label_a!r} and {label_b!r}: value shapes differ "
            f"({sample_a.values.shape} vs {sample_b.values.shape}) -- they are not the same problem instance."
        )
    if not np.array_equal(sample_a.coordinates, sample_b.coordinates):
        raise ValueError(
            f"cannot compare {label_a!r} and {label_b!r}: sampled at different coordinates. A "
            "kernel-vs-kernel comparison requires both subjects to solve on literally the same grid; "
            "this harness never interpolates to force an alignment."
        )
    diff = np.asarray(sample_a.values, dtype=float) - np.asarray(sample_b.values, dtype=float)
    return KernelDiscrepancy(
        label_a=label_a,
        label_b=label_b,
        coordinates_shape=tuple(sample_a.coordinates.shape),
        max_abs_discrepancy=float(np.max(np.abs(diff))),
        rms_discrepancy=float(np.sqrt(np.mean(diff**2))),
    )


def _compare_to_reference(
    label: str, sample: ParitySample, reference: ReferenceFn, *, reference_label: str, tier: ValidationTier
) -> KernelReferenceError:
    exact = np.asarray(reference(sample.coordinates), dtype=float)
    if exact.shape != sample.values.shape:
        raise ValueError(
            f"reference {reference_label!r} returned shape {exact.shape} for {label!r}'s coordinates "
            f"{sample.coordinates.shape}; expected the subject's values shape {sample.values.shape}."
        )
    diff = np.asarray(sample.values, dtype=float) - exact
    return KernelReferenceError(
        label=label,
        reference_label=reference_label,
        coordinates_shape=tuple(sample.coordinates.shape),
        max_abs_error=float(np.max(np.abs(diff))),
        rms_error=float(np.sqrt(np.mean(diff**2))),
        tier=tier,
    )


@dataclass(frozen=True)
class ParityCheck:
    """Run a set of named solve subjects on what the caller asserts is the same problem instance, and
    optionally against a shared closed-form/manufactured reference.

    Generic by construction: ``subjects`` is any mapping of label to zero-argument callable returning
    a :class:`ParitySample` (see :class:`ParitySubject`) -- nothing here is specific to mixle-pde's own
    kernels. This module cannot verify from the outputs alone that two subjects genuinely solve the
    same problem (same domain, coefficients, boundary/initial conditions); that is the caller's
    responsibility (see :func:`transport_groundwater_parity_check` for a worked, verified example). A
    single subject with no peer is a fully supported case -- ``kernel_discrepancies`` is simply empty.
    """

    problem_label: str
    subjects: Mapping[str, ParitySubject]
    reference: ReferenceFn | None = None
    reference_label: str = "manufactured_reference"
    reference_tier: ValidationTier = ValidationTier.MANUFACTURED

    def __post_init__(self) -> None:
        if not self.problem_label.strip():
            raise ValueError("ParityCheck.problem_label must be non-empty.")
        if not self.subjects:
            raise ValueError("ParityCheck needs at least one subject.")
        if self.reference_tier not in (ValidationTier.MANUFACTURED, ValidationTier.ANALYTIC):
            raise ValueError(
                "reference_tier must be MANUFACTURED or ANALYTIC (a closed-form/manufactured solution "
                f"gives an exact error, not a code-to-code comparison); got {self.reference_tier}."
            )

    def run(self) -> ParityReport:
        """Execute every subject once, then build the typed discrepancy/error records.

        Subjects are executed exactly once each (their results are reused for every pairwise
        comparison and the reference comparison), so a subject with a real (slow, or side-effecting)
        solve behind it is never re-run.
        """
        samples: dict[str, ParitySample] = {label: subject() for label, subject in self.subjects.items()}

        kernel_discrepancies = tuple(
            _compare_kernels(label_a, sample_a, label_b, sample_b)
            for (label_a, sample_a), (label_b, sample_b) in itertools.combinations(samples.items(), 2)
        )

        reference_errors: tuple[KernelReferenceError, ...] = ()
        if self.reference is not None:
            reference_errors = tuple(
                _compare_to_reference(
                    label, sample, self.reference, reference_label=self.reference_label, tier=self.reference_tier
                )
                for label, sample in samples.items()
            )

        return ParityReport(
            problem_label=self.problem_label,
            kernel_discrepancies=kernel_discrepancies,
            reference_errors=reference_errors,
        )


# ---------------------------------------------------------------------------
# Manufactured reference: exact periodic single-Fourier-mode solution of linear advection-diffusion.
# ---------------------------------------------------------------------------
def periodic_advection_diffusion_mode(
    x: np.ndarray,
    t: float,
    *,
    amplitude: float = 1.0,
    wavenumber: float,
    diffusivity: float,
    velocity: float,
) -> np.ndarray:
    """The exact solution of ``du/dt = D d^2u/dx^2 - c du/dx`` (periodic boundary) for the
    single-Fourier-mode initial condition ``u(x, 0) = amplitude * cos(wavenumber * x)``:

        u(x, t) = amplitude * exp(-D k^2 t) * cos(k (x - c t))

    By direct substitution: with ``u`` as above, ``u_xx = -k^2 u`` and
    ``u_t = -D k^2 u + c k amplitude exp(-D k^2 t) sin(k(x - c t)) = -D k^2 u - c u_x``, i.e.
    ``u_t = D u_xx - c u_x`` exactly for every ``x`` and ``t`` -- advection rigidly translates the mode
    at speed ``velocity`` while diffusion decays its amplitude at rate ``D k^2``, with no other
    approximation. This is the same continuous PDE
    :class:`mixle_pde.dynamics.AdvectionDiffusionOperator` discretizes, so it is a genuine
    :data:`ValidationTier.MANUFACTURED` reference for that kernel (and, configured per
    :func:`groundwater_uniform_velocity_subject`, for ``GroundwaterTransportOperator`` too) -- not an
    invented forcing function requiring a source-term correction.
    """
    k = float(wavenumber)
    decay = math.exp(-float(diffusivity) * k * k * float(t))
    return float(amplitude) * decay * np.cos(k * (np.asarray(x, dtype=float) - float(velocity) * float(t)))


# ---------------------------------------------------------------------------
# Concrete subjects: the two genuinely overlapping registered kernels.
# ---------------------------------------------------------------------------
def _transport_problem(*, discretization: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": "parity-transport-fd-advdiff",
        "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": "structured_grid"}}],
        "unknowns": [{"id": "field", "domain_id": "domain"}],
        "operators": [
            {
                "id": "parity-transport-fd-advdiff-operator",
                "kind": "linear_operator",
                "input_ids": ["field"],
                "output_ids": ["field"],
                "discretization": discretization,
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [{"kind": "convergence", "required": True}],
        "solve_plan": {"parameters": dict(parameters)},
    }


def transport_fd_advdiff_subject(
    *,
    diffusivity: float,
    velocity: float,
    n: int,
    length: float = 1.0,
    bc: str = "periodic",
    scheme: str = "exact",
    dt: float,
    n_steps: int,
    initial: Callable[[np.ndarray], np.ndarray],
) -> ParitySubject:
    """A :class:`ParitySubject` running the registered ``transport-fd-advdiff`` kernel through the
    real ``run_math_problem`` compatibility-checked boundary.

    Its registered invoker's ``solve_plan`` parameter surface (``grid_size``/``diffusivity``/
    ``velocity``/``dt``/``n_steps``/``boundary``/``scheme``/``initial``) covers exactly what this
    needs, so -- unlike :func:`groundwater_uniform_velocity_subject` -- no bypass of the registered
    entry point is required on this side.
    """
    registration = get_kernel_registration(_TRANSPORT_BACKEND_ID)
    if registration.source != _TRANSPORT_SOURCE:
        raise AssertionError(
            f"{_TRANSPORT_BACKEND_ID} registration source changed to {registration.source!r}; "
            "re-check mixle_pde.verification.parity's binding to the registered kernel."
        )
    if scheme not in _SCHEME_DISCRETIZATION:
        raise ValueError(f"scheme must be one of {sorted(_SCHEME_DISCRETIZATION)}; got {scheme!r}.")
    discretization = _SCHEME_DISCRETIZATION[scheme]

    def _solve() -> ParitySample:
        grid = np.linspace(0.0, length, n)  # matches mixle_pde.dynamics.DynamicsOperator's own grid exactly
        problem = _transport_problem(
            discretization=discretization,
            parameters={
                "grid_size": n,
                "length": length,
                "diffusivity": diffusivity,
                "velocity": velocity,
                "dt": dt,
                "n_steps": n_steps,
                "boundary": bc,
                "scheme": scheme,
                "initial": initial(grid),
            },
        )
        result = run_math_problem(problem, _TRANSPORT_BACKEND_ID)
        return ParitySample(
            label=_TRANSPORT_BACKEND_ID, values=np.asarray(result.solution, dtype=float), coordinates=grid
        )

    return _solve


def groundwater_uniform_velocity_subject(
    *,
    diffusivity: float,
    velocity: float,
    n: int,
    length: float = 1.0,
    bc: str = "periodic",
    scheme: str = "exact",
    dt: float,
    n_steps: int,
    initial: Callable[[np.ndarray], np.ndarray],
) -> ParitySubject:
    """A :class:`ParitySubject` running the registered ``groundwater-fd-transport`` kernel's
    underlying :class:`~mixle_pde.groundwater.GroundwaterTransportOperator` class directly,
    configured with its documented "uniform per-axis velocity" (G2) shorthand instead of a
    Darcy-derived field.

    The registered ``groundwater-fd-transport`` invoker (``_invoke_groundwater_fd`` in
    :mod:`mixle_pde.pde_backend_registry`) always calls
    :func:`mixle_pde.groundwater.darcy_velocity` to build its velocity field, and exposes no
    ``solve_plan`` parameter to request the class's own uniform-velocity shorthand instead. A
    Poisson-solve-derived Darcy field driven by a source/sink pair is never exactly spatially uniform
    (see that function's own docstring), so the registered entry point alone cannot be driven to solve
    literally the same advection-diffusion problem as ``transport-fd-advdiff``. This mirrors the exact
    bypass :mod:`mixle_pde.verification.mms` already uses for MP-K3, for the same reason (its own
    docstring: "drives the same unmodified public building blocks the registered kernel wraps") -- the
    class is real and unmodified, only the registered invoker's parameter surface is too narrow for
    this particular comparison. The provenance-drift guard below ties this back to the actual
    registered kernel it claims to exercise, exactly as MP-K3's guard does.

    Configured with ``dispersivity=0`` and ``molecular_diffusion=diffusivity``, ``decay=0``, and
    ``retardation=1``, :meth:`~mixle_pde.groundwater.GroundwaterTransportOperator.operator_matrix`
    reduces algebraically to exactly ``diffusivity * laplacian_matrix(...) - velocity *
    upwind_gradient_matrix(...)`` -- ``AdvectionDiffusionOperator.operator_matrix``'s own formula --
    since both classes are built from the same ``mixle_pde.dynamics.laplacian_matrix``/
    ``upwind_gradient_matrix`` stencils. The honest expectation is therefore that this subject and
    :func:`transport_fd_advdiff_subject`, driven on the same grid/coefficients/schedule, agree to
    floating-point round-off -- a real regression guard on that reduction, not a coincidence.
    """
    registration = get_kernel_registration(_GROUNDWATER_BACKEND_ID)
    if registration.source != _GROUNDWATER_SOURCE:
        raise AssertionError(
            f"{_GROUNDWATER_BACKEND_ID} registration source changed to {registration.source!r}; "
            "re-check mixle_pde.verification.parity's binding to the registered kernel."
        )

    def _solve() -> ParitySample:
        h = length / (n - 1)
        operator = GroundwaterTransportOperator(
            velocity_field=[velocity],
            dispersivity=0.0,
            shape=(n,),
            retardation=1.0,
            decay=0.0,
            spacing=h,
            bc=bc,
            scheme=scheme,
            molecular_diffusion=diffusivity,
        )
        transition = operator.transition_matrix(dt)
        state = np.asarray(initial(operator.grid), dtype=float)
        for _ in range(n_steps):
            state = transition @ state
        return ParitySample(label=_GROUNDWATER_BACKEND_ID, values=state, coordinates=operator.grid)

    return _solve


def transport_groundwater_parity_check(
    *,
    diffusivity: float = 0.02,
    velocity: float = 0.5,
    n: int = 121,
    length: float = 1.0,
    wavenumber_periods: int = 2,
    amplitude: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 5,
) -> ParityCheck:
    """Build the worked MP-K4 demonstration: ``transport-fd-advdiff`` and ``groundwater-fd-transport``
    (the latter via :func:`groundwater_uniform_velocity_subject`) driven on the identical 1-D periodic
    linear advection-diffusion problem, checked against each other and against the exact
    :func:`periodic_advection_diffusion_mode` manufactured solution.

    ``wavenumber_periods`` full cosine periods are placed across ``[0, length)``; the same callable
    seeds both subjects' initial condition and (evaluated at ``t = dt * n_steps``) the reference.
    """
    wavenumber = 2.0 * math.pi * wavenumber_periods / length

    def _initial(x: np.ndarray) -> np.ndarray:
        return amplitude * np.cos(wavenumber * x)

    def _reference(x: np.ndarray) -> np.ndarray:
        return periodic_advection_diffusion_mode(
            x,
            dt * n_steps,
            amplitude=amplitude,
            wavenumber=wavenumber,
            diffusivity=diffusivity,
            velocity=velocity,
        )

    subjects = {
        _TRANSPORT_BACKEND_ID: transport_fd_advdiff_subject(
            diffusivity=diffusivity,
            velocity=velocity,
            n=n,
            length=length,
            bc="periodic",
            scheme="exact",
            dt=dt,
            n_steps=n_steps,
            initial=_initial,
        ),
        _GROUNDWATER_BACKEND_ID: groundwater_uniform_velocity_subject(
            diffusivity=diffusivity,
            velocity=velocity,
            n=n,
            length=length,
            bc="periodic",
            scheme="exact",
            dt=dt,
            n_steps=n_steps,
            initial=_initial,
        ),
    }
    return ParityCheck(
        problem_label="1-D periodic linear advection-diffusion (uniform velocity, single Fourier mode)",
        subjects=subjects,
        reference=_reference,
        reference_label="periodic_advection_diffusion_mode",
        reference_tier=ValidationTier.MANUFACTURED,
    )
