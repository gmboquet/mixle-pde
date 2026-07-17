"""Gauss-Newton inversion for BOUNDED (nonlinear-transform) latent fields (workstream G6, ladder rung 2).

The linear-Gaussian engine (:func:`mixle_pde.field_inversion.linear_gaussian_invert`) is exact but only
for an identity-transform field. A physical property is often bounded -- a density contrast or
susceptibility that must stay non-negative, a porosity in ``[0, 1]`` -- and a :class:`Field3D` encodes
that with ``bounds``, modelling the Gaussian in an UNCONSTRAINED space ``u`` with a monotone map
``phi(u)`` to physical units (:meth:`Field3D.from_unconstrained`). A linear forward operator is then
linear in ``phi(u)`` but NONLINEAR in the posterior variable ``u``, so the posterior is no longer
closed-form.

This module climbs the inference ladder to Gauss-Newton: iterate the linearized normal equations at the
current estimate to find the MAP in ``u``-space, and return the Laplace (inverse-Hessian) covariance
there as a :class:`PosteriorField3D`. Because the posterior lives in unconstrained space and
:class:`PosteriorField3D` already maps samples/intervals back through ``phi``, every recovered sample and
credible-interval endpoint respects the bounds by construction -- a positivity-constrained inversion can
never report a negative density.

Each Gauss-Newton step (prior on ``u``, precision ``Q``; observations ``d_i = J_i phi(u) + e_i``):

    r_i   = d_i - J_i phi(u_k)                          (residual at the current estimate)
    B     = diag(phi'(u_k))                             (transform Jacobian, per cell)
    A_i   = J_i B                                       (forward Jacobian w.r.t. u)
    Lambda = Q + sum_i A_i^T R_i^-1 A_i                 (Gauss-Newton Hessian)
    g      = Q (u_k - m0) - sum_i A_i^T R_i^-1 r_i      (gradient)
    u_{k+1} = u_k - Lambda^-1 g

converged when the step is small; the posterior covariance is ``Lambda^-1`` at the optimum.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, _noise_precision, linear_gaussian_invert
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.linear_solve import dense_spd_solve
from mixle_pde.observations import ForwardOperatorRegistry, Observation


def _transform_jacobian_diag(grid: Field3D, u: np.ndarray) -> np.ndarray:
    """``d phi / d u`` per cell, where ``phi = grid.from_unconstrained`` -- the monotone map's slope.

    Identity (no bounds): 1. Lower bound only (``phi = lo + exp(u)``): ``exp(u) = phi - lo``. Both bounds
    (``phi = lo + (hi-lo) sigmoid(u)``): ``(phi - lo)(hi - phi)/(hi - lo)``. Upper bound only
    (``phi = hi - exp(u)``): ``exp(u) = hi - phi``.
    """
    phi = grid.from_unconstrained(u)
    if grid.bounds is None:
        return np.ones_like(u)
    lo, hi = grid.bounds
    if lo is not None and hi is not None:
        return (phi - lo) * (hi - phi) / (hi - lo)
    if lo is not None:
        return phi - lo
    return hi - phi


@dataclass
class GaussNewtonReport:
    """Convergence diagnostics for the Gauss-Newton solve."""

    iterations: int
    converged: bool
    step_norms: list[float]
    final_data_misfit: float


def gauss_newton_invert(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    max_iter: int = 100,
    tol: float = 1e-5,
    jitter: float = 1.0e-10,
    factor_cache: dict | None = None,
    u_init: np.ndarray | None = None,
) -> tuple[PosteriorField3D, GaussNewtonReport]:
    """Gauss-Newton MAP + Laplace posterior for a bounded-field linear-observation inversion.

    Every observation's operator must declare a Jacobian (linear in the PHYSICAL field). The field's
    ``bounds`` drive the nonlinear transform; an identity-transform field is allowed too (then this
    reduces to the exact linear-Gaussian solve in one step). ``factor_cache`` (see
    :func:`mixle_pde.linear_solve.make_factor_cache`) is threaded through to the final Laplace-covariance
    solve so a caller iterating this function (e.g. an outer EM loop refitting the prior) can reuse the
    Cholesky factor whenever it calls back in with the same Hessian object. Returns the posterior and a
    convergence report.

    ``u_init``, when given, seeds the Newton iterate ``u`` instead of starting from the prior mean
    ``m0`` -- a warm start. The regularization anchor stays exactly ``m0``/``Q`` (the prior term in the
    objective is untouched), so a warm-started solve still converges to a MAP of the same stated
    problem; only the number of damped steps needed to get there changes. This is what lets
    :func:`mixle_pde.hybrid_inversion.hybrid_gauss_newton_invert` chain repeated calls into one
    trajectory instead of restarting from ``m0`` every time. Ignored by the linear fast path (a single
    exact solve has no starting point to warm).
    """
    if not observations:
        raise ValueError("need at least one observation to invert.")
    n = grid.n
    Q = prior.precision(grid)
    m0 = prior.mean_vector(grid)
    if u_init is not None:
        u_init = np.asarray(u_init, dtype=float)
        if u_init.shape != m0.shape:
            raise ValueError(f"u_init must have shape {m0.shape}, got {u_init.shape}.")

    ops = {}
    for obs in observations:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
            raise ValueError(f"observation kind {obs.kind!r} needs a Jacobian for Gauss-Newton inversion.")
        ops[id(obs)] = op

    rinvs = {id(obs): _noise_precision(obs) for obs in observations}

    if grid.bounds is None and all(op.is_linear for op in ops.values()):
        posterior = linear_gaussian_invert(grid, observations, registry, prior, jitter=jitter)
        misfit = 0.0
        for obs in observations:
            resid = obs.value - ops[id(obs)].predict_observation(grid, posterior.mean, obs)
            misfit += float(resid @ rinvs[id(obs)] @ resid)
        step_norm = float(np.linalg.norm(posterior.mean - m0))
        return posterior, GaussNewtonReport(
            iterations=1,
            converged=True,
            step_norms=[step_norm],
            final_data_misfit=misfit,
        )

    def objective(u_vec: np.ndarray) -> float:
        phi = grid.from_unconstrained(u_vec)
        val = 0.5 * float((u_vec - m0) @ Q @ (u_vec - m0))
        for obs in observations:
            predicted = ops[id(obs)].predict_observation(grid, phi, obs)
            resid = obs.value - predicted
            val += 0.5 * float(resid @ rinvs[id(obs)] @ resid)
        return val

    u = m0.copy() if u_init is None else u_init.copy()
    step_norms: list[float] = []
    converged = False
    lam = Q
    mu = 1.0e-3  # Levenberg-Marquardt damping: scales toward gradient descent when GN overshoots
    obj = objective(u)
    step_tol = max(np.sqrt(tol), 1.0e-4)
    for _ in range(max_iter):
        phi = grid.from_unconstrained(u)
        B = _transform_jacobian_diag(grid, u)
        lam = Q.copy()
        grad = Q @ (u - m0)
        for obs in observations:
            op = ops[id(obs)]
            predicted = op.predict_observation(grid, phi, obs)
            J = op.local_jacobian(grid, phi, obs)
            A = J * B[None, :]  # J @ diag(B)
            at_rinv = A.T @ rinvs[id(obs)]
            lam = lam + at_rinv @ A
            grad = grad - at_rinv @ (obs.value - predicted)

        # damped step; grow damping until the penalized objective actually decreases (or give up this iter)
        accepted = False
        for _ls in range(40):
            damped = lam + (mu * np.diag(lam) + jitter) * np.eye(n)
            step = np.linalg.solve(damped, grad)
            u_try = u - step
            obj_try = objective(u_try)
            if obj_try < obj:
                rel_improve = (obj - obj_try) / max(abs(obj), 1.0)
                step_norm = float(np.linalg.norm(step))
                scaled_step = step_norm / max(1.0, float(np.linalg.norm(u_try)))
                u, obj = u_try, obj_try
                mu = max(mu * 0.5, 1.0e-9)
                step_norms.append(step_norm)
                accepted = True
                # scale-free convergence: the penalized objective has stopped improving meaningfully
                if rel_improve < tol or scaled_step < step_tol:
                    converged = True
                break
            mu *= 3.0
        if not accepted:
            step_norms.append(0.0)
            converged = True  # no downhill step remains: at a (local) optimum
            break
        if converged:
            break

    phi = grid.from_unconstrained(u)
    misfit = 0.0
    for obs in observations:
        resid = obs.value - ops[id(obs)].predict_observation(grid, phi, obs)
        misfit += float(resid @ rinvs[id(obs)] @ resid)

    cov = dense_spd_solve(lam + jitter * np.eye(n), np.eye(n), factor_cache=factor_cache)
    posterior = PosteriorField3D(grid=grid, mean=u, map=u.copy(), dense_cov=cov)
    report = GaussNewtonReport(
        iterations=len(step_norms), converged=converged, step_norms=step_norms, final_data_misfit=misfit
    )
    return posterior, report


def gauss_newton_hessian_hvp(
    grid: Field3D,
    observations: list[Observation],
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    u: np.ndarray,
    *,
    jitter: float = 1.0e-10,
) -> Callable[[np.ndarray], np.ndarray]:
    """Matrix-free Hessian-vector product of the Gauss-Newton Hessian ``Lambda`` at ``u``.

    ``Lambda = Q + sum_i A_i^T R_i^-1 A_i`` (the same linearized normal-equations Hessian
    :func:`gauss_newton_invert` assembles densely each iteration) -- but this returns a closure computing
    ``Lambda @ v`` by looping over each observation's own (small) local Jacobian, without ever summing
    into a dense ``(n, n)`` matrix. This is the HVP :func:`mixle_pde.uq_lowrank.randomized_lowrank_hessian`
    needs to get a low-rank Laplace-approximation posterior at survey scale, where forming ``Lambda``
    densely (as `gauss_newton_invert` does for its exact small-problem posterior) is infeasible.
    """
    n = grid.n
    Q = prior.precision(grid)
    phi = grid.from_unconstrained(u)
    B = _transform_jacobian_diag(grid, u)

    local_ops: list[tuple[np.ndarray, np.ndarray]] = []
    for obs in observations:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
            raise ValueError(f"observation kind {obs.kind!r} needs a Jacobian for the Gauss-Newton Hessian.")
        jac = op.local_jacobian(grid, phi, obs)
        A = jac * B[None, :]  # J @ diag(B), same linearization gauss_newton_invert uses
        rinv = _noise_precision(obs)
        local_ops.append((A, rinv))

    def hvp(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        out = Q @ v + jitter * v
        for A, rinv in local_ops:
            out = out + A.T @ (rinv @ (A @ v))
        return out

    return hvp
