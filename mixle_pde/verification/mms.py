"""Method-of-manufactured-solutions (MMS) convergence verification (MP-K3).

Method of manufactured solutions: pick a smooth analytic function ``u_exact(x, t)`` (it need not
solve any physical problem on its own), plug it into the continuous PDE operator to derive the
forcing/source term ``S = du/dt - L[u]`` that *would* make ``u_exact`` an exact solution, then
drive the real discretized solver with that source and compare its numerical output against
``u_exact`` at increasing mesh/grid resolution. Because ``u_exact`` and ``S`` are both known in
closed form, the only error left is the solver's own discretization error, so refining the mesh
must shrink that error at the discretization's known theoretical rate -- if it does not, the
solver (or its wiring) is broken.

This module is a standalone verification utility layered *on top of* the already-registered legacy
kernels in :mod:`mixle_pde.pde_backend_registry` -- it imports that module read-only (to confirm it
is exercising a real registered backend) and never edits it. The caller supplies the exact solution
and source term as plain callables; this module does not perform any symbolic differentiation.

Kernel exercised: ``transport-fd-advdiff`` (:class:`mixle_pde.dynamics.AdvectionDiffusionOperator`),
the method-of-lines advection-diffusion operator ``du/dt = D d^2u/dx^2 - c du/dx`` with a periodic
1-D grid. :func:`mixle_pde.pde_backend_registry.run_math_problem`'s own ``transport-fd-advdiff``
invocation only steps the homogeneous transition (``state = transition @ state``, no forcing input),
so it cannot carry a manufactured source term. This module instead drives the same unmodified public
building blocks the registered kernel wraps --
:meth:`~mixle_pde.dynamics.AdvectionDiffusionOperator.operator_matrix` and
:meth:`~mixle_pde.dynamics.AdvectionDiffusionOperator.transition_matrix` -- with the standard
backward-Euler-with-source extension of its own ``scheme="implicit"`` update
(``(I - dt G) u_{n+1} = u_n + dt S_{n+1}``, i.e. ``u_{n+1} = transition_matrix(dt) @ (u_n + dt *
S_{n+1})``): a textbook composition, not a modification of the class.

Theoretical order: :func:`mixle_pde.dynamics.upwind_gradient_matrix` is a first-order upwind
difference (its own docstring: "First-order upwind difference for d/dx"), while
:func:`mixle_pde.dynamics.laplacian_matrix` is a second-order central difference. For a manufactured
solution with nonzero advection velocity, the first-order upwind advection term dominates the
truncation error asymptotically, so the theoretical spatial order of convergence for this
solver/discretization is 1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from mixle_pde.dynamics import AdvectionDiffusionOperator
from mixle_pde.pde_backend_registry import get_kernel_registration

__all__ = [
    "MMSResolutionResult",
    "ConvergenceReceipt",
    "estimate_convergence_order",
    "evaluate_convergence",
    "run_transport_fd_advdiff_mms",
    "transport_fd_advdiff_convergence_receipt",
]

# The registered legacy kernel this module exercises, and its documented theoretical spatial order
# (first-order upwind advection dominates the second-order central-difference diffusion term).
_TRANSPORT_BACKEND_ID = "transport-fd-advdiff"
_TRANSPORT_THEORETICAL_ORDER = 1.0


@dataclass(frozen=True)
class MMSResolutionResult:
    """The numerical error measured at one mesh/grid resolution of an MMS run.

    ``resolution`` is the grid point count; ``grid_spacing`` is the corresponding uniform spacing
    ``h``; ``error`` is the ``norm``-named error between the numerical and exact solution fields at
    the run's final simulated time.
    """

    resolution: int
    grid_spacing: float
    error: float
    norm: str


@dataclass(frozen=True)
class ConvergenceReceipt:
    """A pass/fail verdict for a solver's observed order of mesh/grid convergence.

    ``measured_order`` is the log-log slope of ``error`` vs ``grid_spacing`` fit across
    ``resolutions`` (see :func:`estimate_convergence_order`); ``expected_order`` is the solver's own
    documented theoretical order for the discretization exercised; ``passed`` is true when the two
    agree within ``order_tolerance``. A failed dimension is never hidden: every field that went into
    the verdict is recorded on the receipt itself.
    """

    solver_id: str
    discretization: str
    norm: str
    resolutions: tuple[int, ...]
    grid_spacings: tuple[float, ...]
    errors: tuple[float, ...]
    measured_order: float
    expected_order: float
    order_tolerance: float
    passed: bool
    detail: str


def estimate_convergence_order(grid_spacings: Sequence[float], errors: Sequence[float]) -> float:
    """Estimate the observed order of convergence from error-vs-resolution pairs.

    Assumes the asymptotic error model ``error ~= C * h**p``, so ``log(error) = log(C) + p *
    log(h)``; returns ``p`` as the least-squares slope of ``log(error)`` against ``log(h)`` across
    every supplied resolution (not just the first/last pair), which is more robust to noise or
    pre-asymptotic behavior at any single refinement step than a two-point slope.
    """
    if len(grid_spacings) != len(errors):
        raise ValueError("grid_spacings and errors must have the same length.")
    if len(grid_spacings) < 2:
        raise ValueError("need at least two resolutions to estimate an order of convergence.")
    log_h = np.log(np.asarray(grid_spacings, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    if not (np.all(np.isfinite(log_h)) and np.all(np.isfinite(log_e))):
        raise ValueError("cannot estimate convergence order: a grid spacing or error is zero, negative, or non-finite.")
    slope, _intercept = np.polyfit(log_h, log_e, 1)
    return float(slope)


def evaluate_convergence(
    results: Sequence[MMSResolutionResult],
    *,
    solver_id: str,
    discretization: str,
    expected_order: float,
    order_tolerance: float = 0.3,
) -> ConvergenceReceipt:
    """Build a :class:`ConvergenceReceipt` verdict from a sequence of per-resolution MMS results."""
    if len(results) < 2:
        raise ValueError("need at least two resolutions to build a convergence receipt.")
    norms = {result.norm for result in results}
    if len(norms) != 1:
        raise ValueError(f"all resolution results must share one error norm; got {sorted(norms)}")
    resolutions = tuple(result.resolution for result in results)
    grid_spacings = tuple(result.grid_spacing for result in results)
    errors = tuple(result.error for result in results)
    measured_order = estimate_convergence_order(grid_spacings, errors)
    passed = abs(measured_order - expected_order) <= order_tolerance
    detail = (
        f"{solver_id} ({discretization}): measured order {measured_order:.3f} vs expected "
        f"{expected_order:.3f} (tolerance +/-{order_tolerance:.2f}) over resolutions {resolutions} "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return ConvergenceReceipt(
        solver_id=solver_id,
        discretization=discretization,
        norm=next(iter(norms)),
        resolutions=resolutions,
        grid_spacings=grid_spacings,
        errors=errors,
        measured_order=measured_order,
        expected_order=expected_order,
        order_tolerance=order_tolerance,
        passed=passed,
        detail=detail,
    )


def run_transport_fd_advdiff_mms(
    *,
    exact_solution: Callable[[np.ndarray, float], np.ndarray],
    source_term: Callable[[np.ndarray, float], np.ndarray],
    resolutions: Sequence[int],
    diffusivity: float,
    velocity: float,
    length: float = 1.0,
    dt: float,
    n_steps: int,
    norm: str = "max",
) -> tuple[MMSResolutionResult, ...]:
    """Run the registered ``transport-fd-advdiff`` legacy kernel against a manufactured solution.

    For each grid point count in ``resolutions``, builds the real, unmodified
    :class:`mixle_pde.dynamics.AdvectionDiffusionOperator` (``scheme="implicit"``, periodic boundary
    -- the same class :mod:`mixle_pde.pde_backend_registry`'s ``transport-fd-advdiff`` entry wraps),
    seeds it with ``exact_solution`` at ``t=0``, and advances it ``n_steps`` of size ``dt`` with the
    standard backward-Euler-with-source update ``u_{n+1} = transition_matrix(dt) @ (u_n + dt *
    source_term(x, t_{n+1}))``. Returns one :class:`MMSResolutionResult` per resolution, holding the
    ``norm``-named error (``"max"`` for max-norm, ``"l2"`` for RMS L2-norm) between the numerical and
    exact fields at the run's final simulated time ``n_steps * dt``.

    Raises :class:`AssertionError` if the ``transport-fd-advdiff`` registration in
    :mod:`mixle_pde.pde_backend_registry` no longer points at
    ``mixle_pde.dynamics.AdvectionDiffusionOperator`` -- this module drives that class directly, so a
    provenance drift there would silently decouple this verification from the registered kernel it
    claims to exercise.
    """
    if norm not in ("max", "l2"):
        raise ValueError("norm must be 'max' or 'l2'.")
    if len(resolutions) < 2:
        raise ValueError("need at least two resolutions to run an MMS convergence study.")

    registration = get_kernel_registration(_TRANSPORT_BACKEND_ID)
    if registration.source != "mixle_pde.dynamics.AdvectionDiffusionOperator":
        raise AssertionError(
            f"{_TRANSPORT_BACKEND_ID} registration source changed to {registration.source!r}; "
            "re-check mixle_pde.verification.mms's binding to the registered kernel."
        )

    results: list[MMSResolutionResult] = []
    for resolution in resolutions:
        operator = AdvectionDiffusionOperator(
            diffusivity=diffusivity,
            velocity=velocity,
            n=resolution,
            length=length,
            bc="periodic",
            scheme="implicit",
        )
        x = operator.grid
        transition = operator.transition_matrix(dt)
        state = np.asarray(exact_solution(x, 0.0), dtype=float)
        for step in range(n_steps):
            t_next = (step + 1) * dt
            forcing = np.asarray(source_term(x, t_next), dtype=float)
            state = transition @ (state + dt * forcing)

        final_time = n_steps * dt
        exact_final = np.asarray(exact_solution(x, final_time), dtype=float)
        error_field = state - exact_final
        if norm == "max":
            error = float(np.max(np.abs(error_field)))
        else:
            error = float(np.sqrt(np.mean(error_field**2)))

        results.append(MMSResolutionResult(resolution=resolution, grid_spacing=operator.h, error=error, norm=norm))
    return tuple(results)


def transport_fd_advdiff_convergence_receipt(
    *,
    exact_solution: Callable[[np.ndarray, float], np.ndarray],
    source_term: Callable[[np.ndarray, float], np.ndarray],
    resolutions: Sequence[int],
    diffusivity: float,
    velocity: float,
    length: float = 1.0,
    dt: float,
    n_steps: int,
    norm: str = "max",
    expected_order: float = _TRANSPORT_THEORETICAL_ORDER,
    order_tolerance: float = 0.3,
) -> ConvergenceReceipt:
    """Run :func:`run_transport_fd_advdiff_mms` and fold the result into a :class:`ConvergenceReceipt`.

    ``expected_order`` defaults to 1.0, the documented theoretical spatial order of the
    ``transport-fd-advdiff`` kernel's first-order upwind advection discretization (see this module's
    docstring); pass an explicit value to test against a different (e.g. deliberately wrong) claim.
    """
    results = run_transport_fd_advdiff_mms(
        exact_solution=exact_solution,
        source_term=source_term,
        resolutions=resolutions,
        diffusivity=diffusivity,
        velocity=velocity,
        length=length,
        dt=dt,
        n_steps=n_steps,
        norm=norm,
    )
    registration = get_kernel_registration(_TRANSPORT_BACKEND_ID)
    return evaluate_convergence(
        results,
        solver_id=registration.profile.id,
        discretization="FD-implicit (backward Euler, upwind advection, central-difference diffusion)",
        expected_order=expected_order,
        order_tolerance=order_tolerance,
    )
