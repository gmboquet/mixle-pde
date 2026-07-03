"""Differentiable nonlinear steady solver: Newton for ``F(u; theta) = 0`` with an implicit-function-theorem
backward.

This is the nonlinear keystone of the PDE-inverse stack. :func:`mixle_pde.pde_solve.sparse_solve` handles a
*linear* steady state ``A(theta) u = b(theta)`` with an adjoint backward; :func:`nonlinear_solve` generalizes
it to any residual ``F(u; theta) = 0``. The forward pass runs Newton (a sparse LU per iterate); the backward
pass does NOT differentiate through the iterations. Instead it uses the implicit function theorem at the
converged root: ``F(u*(theta), theta) = 0`` implies

    du*/dtheta = -(dF/du)^{-1} (dF/dtheta),

so for a scalar loss ``L(u*)`` the gradient is ``dL/dtheta = -lambda^T (dF/dtheta)`` where the adjoint state
``lambda`` solves ``(dF/du)^T lambda = dL/du``. That is one extra linear solve, reusing the converged Jacobian
factorization, no matter how many entries ``theta`` has -- the same win the linear adjoint gives, now for a
nonlinear steady problem. Reusable for Poisson-Boltzmann, Bratu, and PNP.

The residual and Jacobian are user callbacks so the physics is not baked in:

    residual_fn(u, theta) -> F          # a torch tensor of length n
    jac_fn(u, theta)      -> (rows, cols, vals)   # the sparse dF/du at (u, theta)

:func:`reaction_diffusion_residual` builds the common case ``-div(kappa grad u) + g(u; theta) = f`` (Dirichlet)
by reusing :func:`mixle_pde.pde_solve.divergence_form` plus a nodal reaction term, and its analytic Jacobian.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from mixle.ppl._grid import _grid_faces

__all__ = ["nonlinear_solve", "reaction_diffusion_residual"]


def _torch():
    import torch

    return torch


def _assemble_csc(rows, cols, vals, n, sp):
    r = rows.detach().cpu().numpy()
    c = cols.detach().cpu().numpy()
    return sp.csc_matrix((vals.detach().cpu().numpy(), (r, c)), shape=(int(n), int(n)))


def _newton(residual_fn, jac_fn, u0, theta, n, *, max_its, tol, damping):
    """Newton iteration on ``F(u; theta) = 0`` in NumPy space (no autograd graph). Returns the converged
    ``u`` (detached torch tensor) and the LU factorization of the converged Jacobian ``dF/du``."""
    import scipy.sparse.linalg as spla

    torch = _torch()
    import scipy.sparse as sp

    u = u0.detach().clone()
    lu = None
    for _ in range(int(max_its)):
        with torch.no_grad():
            f = residual_fn(u, theta)
        fnp = f.detach().cpu().numpy()
        if np.linalg.norm(fnp, ord=np.inf) < tol:
            break
        with torch.no_grad():
            rows, cols, vals = jac_fn(u, theta)
        A = _assemble_csc(rows, cols, vals, n, sp)
        lu = spla.splu(A)
        delta = lu.solve(fnp)
        u = u - damping * torch.as_tensor(delta, dtype=u.dtype)
    # ensure the returned factorization is the Jacobian at the final iterate
    with torch.no_grad():
        rows, cols, vals = jac_fn(u, theta)
    lu = spla.splu(_assemble_csc(rows, cols, vals, n, sp))
    return u.detach(), lu


def _nonlinear_solve_function(torch):
    """Build (once) the autograd Function bound to this torch import."""

    class _NonlinearSolve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, residual_fn, jac_fn, u0, n, max_its, tol, damping, theta):
            u_star, lu = _newton(residual_fn, jac_fn, u0, theta, n, max_its=max_its, tol=tol, damping=damping)
            ctx.residual_fn = residual_fn
            ctx.lu = lu
            ctx.n = int(n)
            ctx.save_for_backward(u_star, theta)
            return u_star

        @staticmethod
        def backward(ctx, grad_u):
            u_star, theta = ctx.saved_tensors
            # adjoint state: (dF/du)^H lambda = dL/du  -- one solve reusing the converged factorization.
            lam = ctx.lu.solve(grad_u.detach().cpu().numpy(), trans="H")
            lam_t = torch.as_tensor(lam, dtype=u_star.dtype)
            # dL/dtheta = -lambda^T (dF/dtheta): a single VJP through the residual at the frozen root, with
            # theta the differentiated input and u* held constant. Handles scalar or vector theta.
            theta_g = None
            if theta.requires_grad or ctx.needs_input_grad[7]:
                with torch.enable_grad():
                    theta_v = theta.detach().clone().requires_grad_(True)
                    f = ctx.residual_fn(u_star.detach(), theta_v)
                    (theta_g,) = torch.autograd.grad(f, theta_v, grad_outputs=-lam_t)
            # residual_fn, jac_fn, u0, n, max_its, tol, damping, theta
            return None, None, None, None, None, None, None, theta_g

    return _NonlinearSolve


_CACHE: dict = {}


def nonlinear_solve(
    residual_fn: Callable,
    jac_fn: Callable,
    u0,
    theta,
    *,
    max_its: int = 50,
    tol: float = 1e-10,
    damping: float = 1.0,
):
    """Solve ``F(u; theta) = 0`` by Newton with an implicit-function-theorem backward.

    ``residual_fn(u, theta) -> F`` returns the residual (torch tensor, length ``n``); ``jac_fn(u, theta) ->
    (rows, cols, vals)`` returns the sparse Jacobian ``dF/du`` at the current iterate. ``u0`` is the initial
    guess, ``theta`` the differentiable parameter tensor (scalar or vector). Returns the converged root ``u``
    with gradients to ``theta`` computed by the adjoint / IFT (one extra linear solve reusing the converged
    factorization), never by differentiating through the Newton iterations.
    """
    torch = _torch()
    fn = _CACHE.get("fn")
    if fn is None:
        fn = _CACHE["fn"] = _nonlinear_solve_function(torch)
    u0 = torch.as_tensor(u0, dtype=torch.float64)
    n = u0.shape[0]
    return fn.apply(residual_fn, jac_fn, u0, int(n), int(max_its), float(tol), float(damping), theta)


def reaction_diffusion_residual(
    shape,
    f,
    g: Callable,
    dg_du: Callable,
    *,
    kappa=None,
    spacing: float = 1.0,
):
    """Build ``(residual_fn, jac_fn)`` for the steady nonlinear reaction-diffusion ``-div(kappa grad u) +
    g(u; theta) = f`` with Dirichlet boundaries, reusing :func:`divergence_form` for the diffusion operator.

    ``g(u, theta) -> nodal reaction`` and ``dg_du(u, theta) -> nodal d g / d u`` are elementwise on the field
    (both torch tensors of length ``n``). ``f`` is the source on interior nodes; on boundary nodes the residual
    is ``u - f`` so the source doubles as the Dirichlet data there (identity boundary rows, matching
    :func:`divergence_form`). ``kappa`` is a constant-1 field by default (pure Laplacian diffusion).

    Returns ``(residual_fn, jac_fn)`` ready for :func:`nonlinear_solve`. The reaction is applied on interior
    nodes only, so a boundary node's row stays the identity and its Jacobian diagonal is exactly 1.
    """
    torch = _torch()
    from mixle_pde.pde_solve import divergence_form

    grid = _grid_faces(shape, spacing)
    n = int(grid["n"])
    interior = torch.as_tensor(grid["interior"], dtype=torch.long)
    f_t = torch.as_tensor(np.asarray(f, dtype=float).reshape(-1))
    kap = torch.ones(n, dtype=torch.float64) if kappa is None else torch.as_tensor(kappa, dtype=torch.float64)

    def _matvec(u):
        rows, cols, vals = divergence_form(kap, shape, spacing=spacing, torch=torch)[:3]
        out = torch.zeros(n, dtype=u.dtype)
        return out.index_add(0, rows, vals * u[cols])

    def residual_fn(u, theta):
        # (-div kappa grad u) row already equals u on boundary nodes (identity), so subtract f everywhere and
        # add the interior reaction; on the boundary the residual is (u - f) -> Dirichlet u = f.
        r = _matvec(u) - f_t
        reaction = torch.zeros(n, dtype=u.dtype)
        reaction = reaction.index_add(0, interior, g(u, theta)[interior])
        return r + reaction

    def jac_fn(u, theta):
        rows, cols, vals = divergence_form(kap, shape, spacing=spacing, torch=torch)[:3]
        # add d g / d u on interior diagonals (boundary diagonal stays the identity 1)
        d = dg_du(u, theta)[interior]
        rows = torch.cat([rows, interior])
        cols = torch.cat([cols, interior])
        vals = torch.cat([vals, d.to(vals.dtype)])
        return rows, cols, vals

    return residual_fn, jac_fn
