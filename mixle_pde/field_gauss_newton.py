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

from dataclasses import dataclass

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, _noise_precision
from mixle_pde.latent import Field3D, PosteriorField3D
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
) -> tuple[PosteriorField3D, GaussNewtonReport]:
    """Gauss-Newton MAP + Laplace posterior for a bounded-field linear-observation inversion.

    Every observation's operator must declare a Jacobian (linear in the PHYSICAL field). The field's
    ``bounds`` drive the nonlinear transform; an identity-transform field is allowed too (then this
    reduces to the exact linear-Gaussian solve in one step). Returns the posterior and a convergence
    report.
    """
    if not observations:
        raise ValueError("need at least one observation to invert.")
    n = grid.n
    Q = prior.precision(grid)
    m0 = prior.mean_vector(grid)

    jacs = {}
    for obs in observations:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
            raise ValueError(f"observation kind {obs.kind!r} needs a Jacobian for Gauss-Newton inversion.")
        J = np.atleast_2d(np.asarray(op.jacobian(grid, obs.location), dtype=float))
        if J.shape != (obs.n, n):
            raise ValueError(f"operator {obs.kind!r} Jacobian shape {J.shape} != ({obs.n}, {n}).")
        jacs[id(obs)] = J

    rinvs = {id(obs): _noise_precision(obs) for obs in observations}

    def objective(u_vec: np.ndarray) -> float:
        phi = grid.from_unconstrained(u_vec)
        val = 0.5 * float((u_vec - m0) @ Q @ (u_vec - m0))
        for obs in observations:
            resid = obs.value - jacs[id(obs)] @ phi
            val += 0.5 * float(resid @ rinvs[id(obs)] @ resid)
        return val

    u = m0.copy()
    step_norms: list[float] = []
    converged = False
    lam = Q
    mu = 1.0e-3  # Levenberg-Marquardt damping: scales toward gradient descent when GN overshoots
    obj = objective(u)
    for _ in range(max_iter):
        phi = grid.from_unconstrained(u)
        B = _transform_jacobian_diag(grid, u)
        lam = Q.copy()
        grad = Q @ (u - m0)
        for obs in observations:
            J = jacs[id(obs)]
            A = J * B[None, :]  # J @ diag(B)
            at_rinv = A.T @ rinvs[id(obs)]
            lam = lam + at_rinv @ A
            grad = grad - at_rinv @ (obs.value - J @ phi)

        # damped step; grow damping until the penalized objective actually decreases (or give up this iter)
        accepted = False
        for _ls in range(40):
            damped = lam + (mu * np.diag(lam) + jitter) * np.eye(n)
            step = np.linalg.solve(damped, grad)
            u_try = u - step
            obj_try = objective(u_try)
            if obj_try < obj:
                rel_improve = (obj - obj_try) / max(abs(obj), 1.0)
                u, obj = u_try, obj_try
                mu = max(mu * 0.5, 1.0e-9)
                step_norms.append(float(np.linalg.norm(step)))
                accepted = True
                # scale-free convergence: the penalized objective has stopped improving meaningfully
                if rel_improve < tol:
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
        resid = obs.value - jacs[id(obs)] @ phi
        misfit += float(resid @ rinvs[id(obs)] @ resid)

    cov = np.linalg.inv(lam + jitter * np.eye(n))
    posterior = PosteriorField3D(grid=grid, mean=u, map=u.copy(), cov=cov)
    report = GaussNewtonReport(
        iterations=len(step_norms), converged=converged, step_norms=step_norms, final_data_misfit=misfit
    )
    return posterior, report
