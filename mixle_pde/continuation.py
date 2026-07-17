"""Newton solves with parameter continuation: natural-parameter stepping and pseudo-arclength
continuation for a residual ``F(u; lambda) = 0`` with one scalar continuation parameter ``lambda``.

This is the continuation half of the nonlinear-solves workstream. :mod:`mixle_pde.nonlinear` already
covers a *differentiable* Newton solve at a fixed parameter value (Newton forward, implicit-function-
theorem adjoint backward, for PDE-constrained gradient-based inversion) -- a different job from tracing
how a solution *branch* moves as a parameter varies. This module does that tracing:

- :func:`newton_solve` is a plain residual/Jacobian Newton iteration in NumPy, with a typed receipt of
  the convergence history and, on failure, a concrete machine-checkable reason (never a fabricated
  ``converged=True``).
- :func:`natural_continuation` steps the parameter directly and re-solves by Newton at each step,
  warm-started from the previous point. This is the simplest continuation strategy and it has a sharp,
  well-known limitation: it cannot follow a branch past a *fold* (turning point), because at fixed
  ``lambda`` beyond the fold there is no root to find at all, and near the fold ``dF/du`` becomes
  ill-conditioned and Newton stops converging.
- :func:`arclength_continuation` implements pseudo-arclength continuation (Keller 1977): it treats
  ``lambda`` as an extra unknown alongside ``u`` and parametrizes the branch by arc length ``s`` instead
  of by ``lambda`` directly, solving the augmented system ``[F(u, lambda); N(u, lambda)] = 0`` where
  ``N`` linearizes the arclength constraint against the branch's local tangent. This *can* cross a fold
  -- the tangent's ``dlambda/ds`` component rotates through zero at the turning point rather than the
  solve failing. Its corrector step (``corrector="dense"``, the default) forms and factors the dense
  bordered Jacobian ``[[dF/du, dF/dlambda], [du_ds^T, dlambda_ds]]`` on every Newton iteration, which
  does not scale to large discretizations; ``corrector="jfnk"`` selects a Jacobian-free Newton-Krylov
  alternative (:func:`_augmented_corrector_jfnk`) that never assembles that matrix, instead solving
  each Newton step with GMRES against a matrix-free operator built from finite-difference
  directional derivatives of the augmented residual (Knoll & Keyes 2004). Both correctors share the
  same :class:`NewtonReceipt` contract and, on the Bratu problem below, trace the same branch to
  solver tolerance -- see ``tests/continuation_test.py``'s JFNK-vs-dense cross-check. The once-per-step
  predictor tangent (:func:`_tangent`) still solves a single dense linear system regardless of
  ``corrector``; making that matrix-free too remains open scope. The motivation is asymptotic, not a
  claim that ``"jfnk"`` is faster at every scale: at the Bratu problem's tested size, a single dense
  factorization is cheap enough that ``"dense"`` is likely faster in wall-clock terms; see
  :func:`_augmented_corrector_jfnk` for the honest trade-off this does not overclaim.
- :func:`bratu_problem` builds the classic test case named for exactly this purpose in the work plan:
  the 1-D Bratu equation ``u'' + lambda * exp(u) = 0``, which has a well-characterized fold. See its
  docstring for the closed-form reference solution and :func:`bratu_reference_fold` for computing the
  reference fold location independently of any continuation run.

Scope note: this covers Newton iteration, natural continuation, and arclength continuation (both the
dense and Jacobian-free Newton-Krylov correctors) with singular/non-finite Jacobian detection -- the
baseline needed to actually cross a fold, which is the task's headline acceptance bar ("Bratu
continuation crosses the fold with the appropriate method"), plus MP-F2's follow-on JFNK-scaling gap.
Line search / trust-region globalization, Picard iteration, a matrix-free predictor tangent, bounds and
variational-inequality constraints, branch-switching at bifurcation points (as opposed to folds), and
automatic scaling are explicitly out of scope for this module and remain open against the wider
nonlinear-solves task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

__all__ = [
    "ParametricProblem",
    "NewtonReceipt",
    "ContinuationStep",
    "ContinuationReceipt",
    "newton_solve",
    "natural_continuation",
    "arclength_continuation",
    "bratu_problem",
    "bratu_reference_fold",
]

# Standard Jacobian-free Newton-Krylov finite-difference scaling (Knoll & Keyes 2004, eq. 14): the
# perturbation is scaled by sqrt(machine epsilon) so the finite-difference truncation error and
# floating-point cancellation error in the directional derivative are balanced.
_JFNK_FD_EPS = float(np.sqrt(np.finfo(float).eps))


@dataclass(frozen=True)
class ParametricProblem:
    """A parametrized nonlinear residual ``F(u; lambda) = 0`` for Newton and continuation.

    ``residual(u, lambda) -> R`` returns the residual (shape ``(n,)``); ``jac_u(u, lambda) -> dR/du``
    returns the dense Jacobian with respect to the state (shape ``(n, n)``); ``jac_lambda(u, lambda) ->
    dR/dlambda`` returns the partial derivative with respect to the scalar continuation parameter
    (shape ``(n,)``). ``n`` is the unknown-vector length and ``name`` is a free-text label used only for
    reporting.
    """

    residual: Callable[[np.ndarray, float], np.ndarray]
    jac_u: Callable[[np.ndarray, float], np.ndarray]
    jac_lambda: Callable[[np.ndarray, float], np.ndarray]
    n: int
    name: str = ""


@dataclass(frozen=True)
class NewtonReceipt:
    """Typed outcome of one Newton solve at a fixed parameter value.

    ``residual_history`` is the infinity-norm residual measured *before* each Newton step, so
    ``residual_history[0]`` is the initial guess's residual and, when converged,
    ``residual_history[-1]`` is the one that cleared ``tol``. ``converged`` is only ever ``True`` when
    that final residual actually cleared the tolerance; every failed solve carries a concrete
    ``failure_reason`` (``"max_iterations_exceeded"``, ``"singular_jacobian"``,
    ``"non_finite_residual"``, ``"non_finite_jacobian"``, or ``"non_finite_step"``) rather than a
    fabricated success or a silent crash.
    """

    converged: bool
    iterations: int
    residual_history: tuple[float, ...]
    failure_reason: str | None
    u: np.ndarray


@dataclass(frozen=True)
class ContinuationStep:
    """One point produced while tracing a solution branch.

    ``arclength`` is the cumulative pseudo-arclength ``s`` for :func:`arclength_continuation` steps and
    ``None`` for :func:`natural_continuation` steps, which have no arclength parametrization.
    ``accepted`` is ``False`` exactly when ``newton.converged`` is ``False``; a rejected step is still
    recorded, never dropped, so a receipt's step count always matches how many corrector solves were
    actually attempted.
    """

    parameter: float
    arclength: float | None
    u: np.ndarray
    newton: NewtonReceipt
    accepted: bool


@dataclass(frozen=True)
class ContinuationReceipt:
    """The full record of a continuation run.

    ``stopped_reason`` is one of ``"target_reached"`` / ``"max_steps_reached"`` (nominal termination),
    ``"newton_failed"`` (a corrector step did not converge, so continuation stopped rather than
    fabricate a point), ``"singular_tangent_jacobian"`` (the predictor's tangent solve hit an exactly
    singular matrix), or ``"initial_point_not_converged"`` (the starting guess never became a genuine
    root). ``crossed_fold`` is ``True`` when the parameter's direction of travel reversed at least once
    across the accepted steps -- the structural signature of passing a turning point -- and is always
    ``False`` for :func:`natural_continuation`, which cannot detect or cross one.
    """

    steps: tuple[ContinuationStep, ...]
    stopped_reason: str
    crossed_fold: bool


def newton_solve(
    problem: ParametricProblem,
    u0,
    lam: float,
    *,
    max_iterations: int = 50,
    tol: float = 1e-9,
    damping: float = 1.0,
) -> NewtonReceipt:
    """Solve ``F(u; lambda) = 0`` for fixed ``lambda`` by damped Newton iteration.

    ``u0`` is the initial guess. Returns a :class:`NewtonReceipt` whose ``u`` is the last iterate
    produced -- the converged root when ``converged`` is ``True``, otherwise the point the iteration
    was at when it gave up, never advanced further on a residual that was not actually verified to be
    small.
    """
    u = np.array(u0, dtype=float, copy=True)
    history: list[float] = []
    for iteration in range(int(max_iterations) + 1):
        r = np.asarray(problem.residual(u, lam), dtype=float)
        if not np.all(np.isfinite(r)):
            history.append(float("inf"))
            return NewtonReceipt(
                converged=False,
                iterations=iteration,
                residual_history=tuple(history),
                failure_reason="non_finite_residual",
                u=u,
            )
        rnorm = float(np.max(np.abs(r)))
        history.append(rnorm)
        if rnorm < tol:
            return NewtonReceipt(
                converged=True,
                iterations=iteration,
                residual_history=tuple(history),
                failure_reason=None,
                u=u,
            )
        if iteration == max_iterations:
            break
        j = np.asarray(problem.jac_u(u, lam), dtype=float)
        if not np.all(np.isfinite(j)):
            return NewtonReceipt(
                converged=False,
                iterations=iteration + 1,
                residual_history=tuple(history),
                failure_reason="non_finite_jacobian",
                u=u,
            )
        try:
            delta = np.linalg.solve(j, -r)
        except np.linalg.LinAlgError:
            return NewtonReceipt(
                converged=False,
                iterations=iteration + 1,
                residual_history=tuple(history),
                failure_reason="singular_jacobian",
                u=u,
            )
        if not np.all(np.isfinite(delta)):
            return NewtonReceipt(
                converged=False,
                iterations=iteration + 1,
                residual_history=tuple(history),
                failure_reason="non_finite_step",
                u=u,
            )
        u = u + damping * delta
    return NewtonReceipt(
        converged=False,
        iterations=int(max_iterations),
        residual_history=tuple(history),
        failure_reason="max_iterations_exceeded",
        u=u,
    )


def _tangent(
    problem: ParametricProblem,
    u: np.ndarray,
    lam: float,
    prev_du_ds: np.ndarray | None = None,
    prev_dlambda_ds: float | None = None,
):
    """Unit tangent ``(du/ds, dlambda/ds)`` to the solution branch at a converged ``(u, lambda)``.

    Differentiating ``F(u(s), lambda(s)) = 0`` gives ``dF/du * du_ds + dF/dlambda * dlambda_ds = 0``;
    solving with the normalization ``dlambda_ds = 1`` and then rescaling to unit length is the standard
    bordering construction (Keller 1977). Near a fold, ``dF/du`` is ill-conditioned, which correctly
    drives ``dlambda_ds`` toward zero (the branch momentarily moves purely in ``u``) rather than
    blowing up. When a previous tangent is given, the sign is chosen so the branch continues in the same
    direction instead of reversing on itself.
    """
    j = np.asarray(problem.jac_u(u, lam), dtype=float)
    jl = np.asarray(problem.jac_lambda(u, lam), dtype=float)
    t = np.linalg.solve(j, -jl)
    dlambda_ds = 1.0 / np.sqrt(1.0 + float(t @ t))
    du_ds = t * dlambda_ds
    if prev_du_ds is not None and prev_dlambda_ds is not None:
        if float(du_ds @ prev_du_ds) + dlambda_ds * prev_dlambda_ds < 0.0:
            du_ds, dlambda_ds = -du_ds, -dlambda_ds
    return du_ds, dlambda_ds


def _augmented_corrector(
    problem: ParametricProblem,
    u_guess: np.ndarray,
    lam_guess: float,
    u_prev: np.ndarray,
    lam_prev: float,
    du_ds: np.ndarray,
    dlambda_ds: float,
    ds: float,
    *,
    max_iterations: int,
    tol: float,
    damping: float,
) -> tuple[NewtonReceipt, float]:
    """Newton-correct the augmented pseudo-arclength system for one continuation step.

    Solves ``F(u, lambda) = 0`` jointly with the linearized arclength constraint
    ``N(u, lambda) = du_ds . (u - u_prev) + dlambda_ds * (lambda - lambda_prev) - ds = 0`` for the
    ``(n + 1)``-vector ``(u, lambda)``, via the bordered Jacobian
    ``[[dF/du, dF/dlambda], [du_ds^T, dlambda_ds]]``. Returns the :class:`NewtonReceipt` for the
    ``u``-part plus the corrected ``lambda``.
    """
    u = np.array(u_guess, dtype=float, copy=True)
    lam = float(lam_guess)
    n = problem.n
    history: list[float] = []
    for iteration in range(int(max_iterations) + 1):
        r = np.asarray(problem.residual(u, lam), dtype=float)
        arc = float(du_ds @ (u - u_prev) + dlambda_ds * (lam - lam_prev) - ds)
        if not (np.all(np.isfinite(r)) and np.isfinite(arc)):
            history.append(float("inf"))
            return (
                NewtonReceipt(
                    converged=False,
                    iterations=iteration,
                    residual_history=tuple(history),
                    failure_reason="non_finite_residual",
                    u=u,
                ),
                lam,
            )
        rnorm = float(max(np.max(np.abs(r)), abs(arc)))
        history.append(rnorm)
        if rnorm < tol:
            return (
                NewtonReceipt(
                    converged=True,
                    iterations=iteration,
                    residual_history=tuple(history),
                    failure_reason=None,
                    u=u,
                ),
                lam,
            )
        if iteration == max_iterations:
            break
        ju = np.asarray(problem.jac_u(u, lam), dtype=float)
        jl = np.asarray(problem.jac_lambda(u, lam), dtype=float)
        if not (np.all(np.isfinite(ju)) and np.all(np.isfinite(jl))):
            return (
                NewtonReceipt(
                    converged=False,
                    iterations=iteration + 1,
                    residual_history=tuple(history),
                    failure_reason="non_finite_jacobian",
                    u=u,
                ),
                lam,
            )
        augmented = np.zeros((n + 1, n + 1))
        augmented[:n, :n] = ju
        augmented[:n, n] = jl
        augmented[n, :n] = du_ds
        augmented[n, n] = dlambda_ds
        rhs = np.concatenate([-r, [-arc]])
        try:
            delta = np.linalg.solve(augmented, rhs)
        except np.linalg.LinAlgError:
            return (
                NewtonReceipt(
                    converged=False,
                    iterations=iteration + 1,
                    residual_history=tuple(history),
                    failure_reason="singular_jacobian",
                    u=u,
                ),
                lam,
            )
        if not np.all(np.isfinite(delta)):
            return (
                NewtonReceipt(
                    converged=False,
                    iterations=iteration + 1,
                    residual_history=tuple(history),
                    failure_reason="non_finite_step",
                    u=u,
                ),
                lam,
            )
        u = u + damping * delta[:n]
        lam = lam + float(damping * delta[n])
    return (
        NewtonReceipt(
            converged=False,
            iterations=int(max_iterations),
            residual_history=tuple(history),
            failure_reason="max_iterations_exceeded",
            u=u,
        ),
        lam,
    )


def _augmented_residual(
    problem: ParametricProblem,
    z: np.ndarray,
    u_prev: np.ndarray,
    lam_prev: float,
    du_ds: np.ndarray,
    dlambda_ds: float,
    ds: float,
) -> np.ndarray | None:
    """Evaluate the augmented pseudo-arclength residual ``G(z) = [F(u, lambda); N(u, lambda)]`` at the
    stacked unknown ``z = [u; lambda]`` (shape ``(n + 1,)``), returning ``None`` in place of a
    non-finite image so every caller handles that breakdown the same way. This is the one piece of the
    augmented system genuinely shared between :func:`_augmented_corrector`'s dense path and
    :func:`_augmented_corrector_jfnk`'s matrix-free one: both correct the same ``G(z) = 0``, they only
    differ in how the linear step is solved.
    """
    n = problem.n
    u = z[:n]
    lam = float(z[n])
    r = np.asarray(problem.residual(u, lam), dtype=float)
    arc = float(du_ds @ (u - u_prev) + dlambda_ds * (lam - lam_prev) - ds)
    g = np.concatenate([r, [arc]])
    if not np.all(np.isfinite(g)):
        return None
    return g


def _augmented_jvp(
    problem: ParametricProblem,
    z: np.ndarray,
    g_z: np.ndarray,
    v: np.ndarray,
    u_prev: np.ndarray,
    lam_prev: float,
    du_ds: np.ndarray,
    dlambda_ds: float,
    ds: float,
    fd_eps: float,
) -> np.ndarray | None:
    """Forward-difference directional derivative ``dG/dz @ v ~ (G(z + eps*v) - G(z)) / eps`` of the
    augmented residual -- the Jacobian-vector product a matrix-free Krylov solve needs without ever
    assembling ``dG/dz`` (Knoll & Keyes 2004). ``eps`` is scaled by both ``fd_eps`` and the current
    iterate/direction magnitudes (their eq. 14), and ``g_z``, the already-evaluated ``G(z)``, is passed
    in rather than recomputed so each of GMRES's ``matvec`` calls within one Newton iteration costs one
    extra residual evaluation, not two. Returns ``None`` in place of a non-finite image, exactly like
    :func:`_augmented_residual`.
    """
    v_norm = float(np.linalg.norm(v))
    if v_norm == 0.0:
        return np.zeros_like(g_z)
    eps = fd_eps * (1.0 + float(np.linalg.norm(z))) / v_norm
    g_pert = _augmented_residual(problem, z + eps * v, u_prev, lam_prev, du_ds, dlambda_ds, ds)
    if g_pert is None:
        return None
    return (g_pert - g_z) / eps


def _augmented_corrector_jfnk(
    problem: ParametricProblem,
    u_guess: np.ndarray,
    lam_guess: float,
    u_prev: np.ndarray,
    lam_prev: float,
    du_ds: np.ndarray,
    dlambda_ds: float,
    ds: float,
    *,
    max_iterations: int,
    tol: float,
    damping: float,
    krylov_rtol: float,
    krylov_maxiter: int | None,
    fd_eps: float,
) -> tuple[NewtonReceipt, float]:
    """Jacobian-free Newton-Krylov correction of the augmented pseudo-arclength system for one
    continuation step.

    Same contract as :func:`_augmented_corrector`: identical :class:`NewtonReceipt` shape, the same
    ``max(||F||_inf, |N|)`` convergence criterion, and the same failure-reason vocabulary -- but this
    never forms or factors the bordered Jacobian ``[[dF/du, dF/dlambda], [du_ds^T, dlambda_ds]]``. Each
    Newton step's linear solve instead runs GMRES (:func:`scipy.sparse.linalg.gmres`) against a
    matrix-free operator whose action is :func:`_augmented_jvp`, so the corrector's per-iteration cost
    is residual evaluations and matrix-vector products only, never an assembled or factored
    ``(n + 1, n + 1)`` matrix. That repeated per-iteration dense solve is exactly what does not scale to
    large discretizations, and it is the corrector -- not the once-per-step predictor tangent solved by
    :func:`_tangent`, left dense and untouched here -- that pays for it more than once within a single
    continuation step whenever the corrector needs more than one Newton iteration.

    ``krylov_rtol``/``krylov_maxiter`` bound the inner GMRES solve. An inner solve that has not reached
    ``krylov_rtol`` after ``krylov_maxiter`` iterations is not itself a failure: inexact Newton-Krylov
    methods are expected to run under-converged inner solves far from the root (Knoll & Keyes 2004,
    section 2.2's discussion of forcing terms), so the outer criteria -- the residual tolerance,
    non-finite detection, and the iteration budget -- remain the only ways this reports ``converged`` or
    a failure, exactly as for the dense corrector. One taxonomy difference is real, not an oversight: a
    structurally singular augmented system has no ``LinAlgError`` analogue under GMRES (it stalls
    instead of raising), so this corrector can report ``non_finite_step`` or
    ``max_iterations_exceeded`` where the dense corrector would report ``singular_jacobian``, and never
    reports ``singular_jacobian`` or ``non_finite_jacobian`` itself.

    ``krylov_rtol``'s default (``1e-6``, not GMRES's own far tighter usual defaults) is deliberate, not
    a shortcut: GMRES's ``rtol`` is relative to the *starting* linear residual, which near a
    well-converging branch is already close to the outer ``tol``, so demanding several more orders of
    relative reduction buys negligible outer-residual improvement for a real cost in Krylov iterations
    (confirmed directly while tuning this default: on this module's own Bratu scenario, ``rtol=1e-10``
    needs hundreds of matrix-vector products per Newton iteration where ``rtol=1e-6`` needs a small
    fraction of that, for the same converged outer result to the same ``tol``).

    Honest trade-off, not overclaimed: at the Bratu problem's tested scale (``n`` in the low hundreds),
    a single dense ``(n + 1, n + 1)`` factorization is cheap enough that the dense corrector is likely
    faster in wall-clock terms than this one, since restarted GMRES needs many matrix-vector products
    (each a full residual evaluation) to reach the same tolerance a direct solve reaches in one
    factorization. The motivation is asymptotic: dense factorization cost grows like ``O(n^3)`` (and its
    memory like ``O(n^2)``) per Newton iteration, while this corrector's cost per iteration stays linear
    in the cost of one residual evaluation times the number of Krylov iterations GMRES actually needs --
    a crossover this module does not claim a specific ``n`` for, only that one exists.
    """
    n = problem.n
    z = np.concatenate([np.array(u_guess, dtype=float, copy=True), [float(lam_guess)]])
    history: list[float] = []
    for iteration in range(int(max_iterations) + 1):
        g = _augmented_residual(problem, z, u_prev, lam_prev, du_ds, dlambda_ds, ds)
        if g is None:
            history.append(float("inf"))
            return (
                NewtonReceipt(
                    converged=False,
                    iterations=iteration,
                    residual_history=tuple(history),
                    failure_reason="non_finite_residual",
                    u=z[:n],
                ),
                float(z[n]),
            )
        rnorm = float(np.max(np.abs(g)))
        history.append(rnorm)
        if rnorm < tol:
            return (
                NewtonReceipt(
                    converged=True,
                    iterations=iteration,
                    residual_history=tuple(history),
                    failure_reason=None,
                    u=z[:n],
                ),
                float(z[n]),
            )
        if iteration == max_iterations:
            break

        def _matvec(v: np.ndarray, _z: np.ndarray = z, _g: np.ndarray = g) -> np.ndarray:
            jv = _augmented_jvp(problem, _z, _g, v, u_prev, lam_prev, du_ds, dlambda_ds, ds, fd_eps)
            return jv if jv is not None else np.full(n + 1, np.nan)

        operator = LinearOperator((n + 1, n + 1), matvec=_matvec, dtype=float)
        delta, _info = gmres(
            operator,
            -g,
            rtol=krylov_rtol,
            atol=0.0,
            restart=min(n + 1, 50),
            maxiter=krylov_maxiter,
        )
        if not np.all(np.isfinite(delta)):
            return (
                NewtonReceipt(
                    converged=False,
                    iterations=iteration + 1,
                    residual_history=tuple(history),
                    failure_reason="non_finite_step",
                    u=z[:n],
                ),
                float(z[n]),
            )
        z = z + damping * delta
    return (
        NewtonReceipt(
            converged=False,
            iterations=int(max_iterations),
            residual_history=tuple(history),
            failure_reason="max_iterations_exceeded",
            u=z[:n],
        ),
        float(z[n]),
    )


def natural_continuation(
    problem: ParametricProblem,
    u0,
    lam0: float,
    lam_target: float,
    *,
    n_steps: int,
    max_iterations: int = 50,
    tol: float = 1e-9,
    damping: float = 1.0,
) -> ContinuationReceipt:
    """Step ``lambda`` directly from ``lam0`` to ``lam_target`` in ``n_steps`` equal increments,
    re-solving by Newton at each step and warm-starting from the previous point.

    This is the simplest continuation strategy and it cannot cross a fold: beyond the critical
    parameter there is no root at fixed ``lambda`` to find, so the corresponding Newton solve fails and
    the receipt stops honestly with ``stopped_reason="newton_failed"`` rather than reporting a point
    that was never actually converged.
    """
    start = newton_solve(problem, u0, float(lam0), max_iterations=max_iterations, tol=tol, damping=damping)
    steps = [
        ContinuationStep(
            parameter=float(lam0),
            arclength=None,
            u=start.u,
            newton=start,
            accepted=start.converged,
        )
    ]
    if not start.converged:
        return ContinuationReceipt(steps=tuple(steps), stopped_reason="initial_point_not_converged", crossed_fold=False)

    u = start.u
    stopped_reason = "target_reached"
    for lam in np.linspace(float(lam0), float(lam_target), int(n_steps) + 1)[1:]:
        receipt = newton_solve(problem, u, float(lam), max_iterations=max_iterations, tol=tol, damping=damping)
        steps.append(
            ContinuationStep(
                parameter=float(lam),
                arclength=None,
                u=receipt.u,
                newton=receipt,
                accepted=receipt.converged,
            )
        )
        if not receipt.converged:
            stopped_reason = "newton_failed"
            break
        u = receipt.u
    return ContinuationReceipt(steps=tuple(steps), stopped_reason=stopped_reason, crossed_fold=False)


def arclength_continuation(
    problem: ParametricProblem,
    u0,
    lam0: float,
    *,
    ds: float,
    n_steps: int,
    max_iterations: int = 20,
    tol: float = 1e-9,
    damping: float = 1.0,
    direction: float = 1.0,
    corrector: str = "dense",
    krylov_rtol: float = 1e-6,
    krylov_maxiter: int | None = 200,
    fd_eps: float = _JFNK_FD_EPS,
) -> ContinuationReceipt:
    """Trace the solution branch through ``origin (u0, lam0)`` by pseudo-arclength continuation.

    Each step predicts along the local tangent and Newton-corrects the augmented ``(u, lambda)`` system
    against the linearized arclength constraint. Unlike :func:`natural_continuation`, this can cross a
    fold: the tangent's ``dlambda/ds`` component passes through zero and changes sign there instead of
    the corrector failing, which is recorded in the returned receipt's ``crossed_fold`` flag.
    ``direction`` (its sign only) picks which way along the branch the first step goes.

    ``corrector`` selects how each step's Newton correction is solved: ``"dense"`` (default) forms and
    factors the bordered Jacobian every iteration via :func:`_augmented_corrector`, exactly as before;
    ``"jfnk"`` never assembles that matrix, instead solving each iteration with GMRES against a
    matrix-free finite-difference operator via :func:`_augmented_corrector_jfnk` -- the alternative that
    scales to large discretizations where forming/factoring a dense Jacobian per iteration does not.
    ``krylov_rtol``, ``krylov_maxiter``, and ``fd_eps`` tune the inner GMRES solve and are ignored when
    ``corrector="dense"``.
    """
    if corrector not in ("dense", "jfnk"):
        raise ValueError(f"corrector must be 'dense' or 'jfnk', got {corrector!r}")
    start = newton_solve(problem, u0, float(lam0), max_iterations=max_iterations, tol=tol, damping=damping)
    if not start.converged:
        step = ContinuationStep(parameter=float(lam0), arclength=0.0, u=start.u, newton=start, accepted=False)
        return ContinuationReceipt(steps=(step,), stopped_reason="initial_point_not_converged", crossed_fold=False)

    u_prev, lam_prev = start.u, float(lam0)
    steps = [ContinuationStep(parameter=lam_prev, arclength=0.0, u=u_prev, newton=start, accepted=True)]

    try:
        du_ds, dlambda_ds = _tangent(problem, u_prev, lam_prev)
    except np.linalg.LinAlgError:
        return ContinuationReceipt(steps=tuple(steps), stopped_reason="singular_tangent_jacobian", crossed_fold=False)
    sign = 1.0 if direction >= 0.0 else -1.0
    du_ds, dlambda_ds = sign * du_ds, sign * dlambda_ds

    crossed_fold = False
    s = 0.0
    stopped_reason = "max_steps_reached"
    for _ in range(int(n_steps)):
        u_pred = u_prev + ds * du_ds
        lam_pred = lam_prev + ds * dlambda_ds
        if corrector == "dense":
            receipt, lam_new = _augmented_corrector(
                problem,
                u_pred,
                lam_pred,
                u_prev,
                lam_prev,
                du_ds,
                dlambda_ds,
                ds,
                max_iterations=max_iterations,
                tol=tol,
                damping=damping,
            )
        else:
            receipt, lam_new = _augmented_corrector_jfnk(
                problem,
                u_pred,
                lam_pred,
                u_prev,
                lam_prev,
                du_ds,
                dlambda_ds,
                ds,
                max_iterations=max_iterations,
                tol=tol,
                damping=damping,
                krylov_rtol=krylov_rtol,
                krylov_maxiter=krylov_maxiter,
                fd_eps=fd_eps,
            )
        if not receipt.converged:
            steps.append(
                ContinuationStep(parameter=lam_new, arclength=s + ds, u=receipt.u, newton=receipt, accepted=False)
            )
            stopped_reason = "newton_failed"
            break
        s += ds
        try:
            du_ds_new, dlambda_ds_new = _tangent(problem, receipt.u, lam_new, du_ds, dlambda_ds)
        except np.linalg.LinAlgError:
            steps.append(ContinuationStep(parameter=lam_new, arclength=s, u=receipt.u, newton=receipt, accepted=True))
            stopped_reason = "singular_tangent_jacobian"
            break
        if dlambda_ds_new != 0.0 and dlambda_ds != 0.0 and np.sign(dlambda_ds_new) != np.sign(dlambda_ds):
            crossed_fold = True
        steps.append(ContinuationStep(parameter=lam_new, arclength=s, u=receipt.u, newton=receipt, accepted=True))
        u_prev, lam_prev, du_ds, dlambda_ds = receipt.u, lam_new, du_ds_new, dlambda_ds_new

    return ContinuationReceipt(steps=tuple(steps), stopped_reason=stopped_reason, crossed_fold=crossed_fold)


def bratu_problem(n: int) -> ParametricProblem:
    """The 1-D Bratu problem ``u'' + lambda * exp(u) = 0`` on ``[0, 1]`` with homogeneous Dirichlet
    boundaries ``u(0) = u(1) = 0``, discretized by second-order central finite differences on ``n``
    interior nodes (spacing ``h = 1 / (n + 1)``).

    The classic continuation test case (Glowinski, Keller & Reinhart 1985): for ``lambda`` below a
    critical fold ``lambda_c`` there are two solution branches (a low- and a high-amplitude branch),
    exactly one at ``lambda = lambda_c``, and none above it. :func:`natural_continuation` cannot follow
    a branch past ``lambda_c`` -- there is no root to find at fixed ``lambda`` beyond the fold --  while
    :func:`arclength_continuation` traces through the turning point onto the second branch.

    The continuous problem has a closed-form solution family
    ``u(x) = -2 log(cosh((x - 1/2) * theta / 2) / cosh(theta / 4))`` with
    ``lambda(theta) = theta**2 / (2 * cosh(theta / 4)**2)``; see :func:`bratu_reference_fold` for the
    corresponding independent reference fold location.
    """
    n = int(n)
    if n < 3:
        raise ValueError(f"bratu_problem requires n >= 3 interior nodes, got {n}")
    h = 1.0 / (n + 1)
    h2 = h * h

    def residual(u, lam):
        u = np.asarray(u, dtype=float)
        left = np.concatenate(([0.0], u[:-1]))
        right = np.concatenate((u[1:], [0.0]))
        return left - 2.0 * u + right + h2 * float(lam) * np.exp(u)

    def jac_u(u, lam):
        u = np.asarray(u, dtype=float)
        diag = -2.0 + h2 * float(lam) * np.exp(u)
        return np.diag(diag) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)

    def jac_lambda(u, lam):
        u = np.asarray(u, dtype=float)
        return h2 * np.exp(u)

    return ParametricProblem(residual=residual, jac_u=jac_u, jac_lambda=jac_lambda, n=n, name=f"bratu_1d(n={n})")


def bratu_reference_fold() -> tuple[float, float]:
    """Independent closed-form fold location for the *continuous* 1-D Bratu problem (not the
    discretized one :func:`bratu_problem` returns), for validating continuation output against a
    reference that was not produced by this module's own solver.

    Returns ``(theta_c, lambda_c)`` where ``lambda_c = max_theta theta**2 / (2 * cosh(theta / 4)**2)``
    is the critical parameter beyond which the exact boundary value problem has no solution.
    """
    from scipy.optimize import minimize_scalar

    def neg_lambda(theta):
        return -(theta**2) / (2.0 * np.cosh(theta / 4.0) ** 2)

    result = minimize_scalar(neg_lambda, bounds=(1e-6, 12.0), method="bounded", options={"xatol": 1e-12})
    theta_c = float(result.x)
    lambda_c = -float(result.fun)
    return theta_c, lambda_c
