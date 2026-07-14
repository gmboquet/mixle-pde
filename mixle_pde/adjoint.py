"""Discrete adjoint sensitivities for the sparse-PDE forward operators (workstream C1).

The registry's nonlinear EM/DC/MT/CSEM operators (:mod:`mixle_pde.observations`) all route their
physics through :func:`mixle_pde.pde_solve.sparse_solve`: a sparse linear system ``A(theta) u =
b(theta)`` factorized once (``splu``) with a custom autograd backward that solves a single adjoint
system ``A^H lambda = dL/du`` reusing that cached factorization. Today's registry Jacobians ignore
this and fall back to a central finite-difference loop over every model parameter -- ``2 * n_model``
extra forward factorizations for an ``n_model``-cell field. This module gives the registry a proper
reverse-mode Jacobian instead: one differentiable forward pass (one factorization, or one per unique
right-hand side, e.g. one per current injection or per sounding frequency) followed by ``n_obs``
seeded backward passes, each an O(1) adjoint solve against the cached factor. Cost drops from
``O(n_model)`` factorizations to ``O(1)`` -- a win whenever ``n_obs << n_model`` (the common case: a
handful of electrodes/frequencies/edges sensing a field of thousands of cells).

``torch_adjoint_jvp`` computes the transposed quantity, ``J @ v`` for a single model-space direction
``v``, needed by Hessian-vector-product machinery (workstream C2) without ever materializing ``J``.
It attempts the standard "forward-over-reverse" (double-vjp) construction, which is exact whenever
the underlying backward is itself autograd-differentiable; :func:`mixle_pde.pde_solve.sparse_solve`'s
backward computes its adjoint solve through a detached numpy factorization (by design -- see that
module's docstring: it is deliberately first-order only, so a second-order Laplace fit is refused
rather than silently wrong), so the double-backward graph is severed and this transparently falls
back to a one-sided directional derivative on the same ``predict_torch`` (one extra evaluation, not a
central difference) -- still first-order exact and still "one linearized solve" in cost.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

__all__ = ["torch_adjoint_jacobian", "torch_adjoint_jvp"]


def _torch():
    import torch

    return torch


def torch_adjoint_jacobian(predict_torch: Callable[[Tensor], Tensor], x: np.ndarray, *, n_obs: int) -> np.ndarray:
    """Reverse-mode Jacobian of ``predict_torch`` at ``x``, shape ``(n_obs, n_model)``.

    ``predict_torch`` must be a grad-enabled forward (no ``torch.no_grad()``) built on the
    differentiable core of a registry operator -- e.g. :func:`mixle_pde.geophysics.dc_resistivity`
    directly, which routes through :func:`mixle_pde.pde_solve.sparse_solve`. That solve's custom
    autograd Function factorizes the system once per forward call and caches the LU factor on its
    context; each of the ``n_obs`` seeded ``.backward()`` calls below reuses that cached factor to
    solve a single adjoint system, so the total linear-algebra cost is one forward factorization
    (per unique right-hand side the operator's physics needs) plus ``n_obs`` O(1) adjoint solves --
    not the ``2 * n_model`` forward factorizations a central-difference Jacobian costs.

    Args:
        predict_torch: differentiable ``x_tensor -> y_tensor`` map, ``y`` real-valued length ``n_obs``.
        x: ``(n_model,)`` point to linearize at.
        n_obs: expected length of ``predict_torch``'s output (validated against the actual output).

    Returns:
        ``(n_obs, n_model)`` ndarray, dtype float64.
    """
    torch = _torch()
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    n_model = x_arr.shape[0]
    x_t = torch.as_tensor(x_arr, dtype=torch.float64).requires_grad_(True)
    y = torch.atleast_1d(predict_torch(x_t))
    if y.shape[0] != int(n_obs):
        raise ValueError(f"predict_torch returned {y.shape[0]} outputs, expected n_obs={n_obs}.")
    jac = np.empty((int(n_obs), n_model), dtype=float)
    for j in range(int(n_obs)):
        seed = torch.zeros(int(n_obs), dtype=y.dtype)
        seed[j] = 1.0
        (grad_x,) = torch.autograd.grad(y, x_t, grad_outputs=seed, retain_graph=True)
        row = grad_x.detach().cpu().numpy()
        jac[j] = row.real if np.iscomplexobj(row) else row
    return jac


def torch_adjoint_jvp(predict_torch: Callable[[Tensor], Tensor], x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Jacobian-vector product ``J @ v`` of ``predict_torch`` at ``x``, in one linearized pass.

    Tries the exact forward-over-reverse construction first: seed a dummy cotangent ``w`` through one
    reverse pass (``create_graph=True``) to get the symbolic map ``w -> w^T J``, then differentiate the
    scalar ``(w^T J) . v`` back through ``w`` to read off ``J v`` -- exact whenever the reverse pass is
    itself twice-differentiable. When it is not (see module docstring), falls back to a one-sided
    directional derivative on ``predict_torch`` -- one extra forward solve, first-order exact.
    """
    torch = _torch()
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    v_arr = np.asarray(v, dtype=float).reshape(-1)
    x_t = torch.as_tensor(x_arr, dtype=torch.float64).requires_grad_(True)
    y = torch.atleast_1d(predict_torch(x_t))
    w = torch.zeros_like(y).requires_grad_(True)
    (g,) = torch.autograd.grad(y, x_t, grad_outputs=w, create_graph=True, allow_unused=True)
    jvp = None
    if g is not None and g.requires_grad:
        v_t = torch.as_tensor(v_arr, dtype=g.real.dtype if g.is_complex() else g.dtype)
        h = (g * v_t).sum() if not g.is_complex() else (g.real * v_t).sum()
        (jvp,) = torch.autograd.grad(h, w, allow_unused=True)
    if jvp is None:
        y0 = y.detach().cpu().numpy()
        v_norm = float(np.linalg.norm(v_arr))
        step = 1.0e-6 * (1.0 + float(np.linalg.norm(x_arr))) / max(v_norm, 1.0e-30)
        with torch.no_grad():
            x_pert = torch.as_tensor(x_arr + step * v_arr, dtype=torch.float64)
            y1 = torch.atleast_1d(predict_torch(x_pert)).detach().cpu().numpy()
        result = (y1 - y0) / step
        return result.real if np.iscomplexobj(result) else result
    row = jvp.detach().cpu().numpy()
    return row.real if np.iscomplexobj(row) else row
