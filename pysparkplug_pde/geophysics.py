"""Near-surface geophysical forward operators and a regularized inversion engine.

The adjoint sparse stack (:mod:`pysparkplug_pde.pde_solve`) gives differentiable PDE *forwards*, and
:class:`pysparkplug_pde.inverse.Differential` wraps a forward as a likelihood for ``pysp.ppl``'s
Gaussian-field MAP/Gauss-Newton. That pairing is excellent for mildly ill-posed problems (radar/sonar/seismic
full-waveform, where the data strongly constrain the field), but it struggles on the *severely* ill-posed
potential-field problems of exploration geophysics -- DC resistivity (ERT) above all, where the sensitivity
decays by orders of magnitude with depth and the generic Gaussian-field prior either leaves deep cells pinned
to the reference or has to be hand-tuned per problem.

This module adds the machinery those problems actually need:

* :func:`dc_resistivity` -- the DC (Poisson) resistivity forward: transfer resistances for a quadrupole
  schedule, one factorization reused across all measurements sharing a current injection, differentiable in
  log-conductivity.
* :func:`straight_ray_operator` -- the ray-length matrix for first-arrival traveltime tomography (crosshole /
  surface GPR and seismic), a sparse linear forward.
* :func:`roughness_operator` -- the grid first-difference (smoothness) operator used to regularize.
* :func:`regularized_gauss_newton` -- an Occam-style regularized Gauss-Newton inversion with smoothness
  (and optional reference/bounds), a line search, and a linearized posterior standard deviation. This is the
  workhorse: it inverts ERT, gravity, magnetics, or any differentiable forward, robustly and without
  per-problem prior tuning.
* :func:`cross_gradient` and :func:`joint_inversion` -- structural (cross-gradient) coupling of several
  property models that share boundaries but not a petrophysical law, and the joint inverter that uses it.

Everything is torch-differentiable and uses the package's existing ``divergence_form`` / ``sparse_solve``;
nothing here patches pysp.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp

__all__ = [
    "dc_resistivity",
    "straight_ray_operator",
    "roughness_operator",
    "regularized_gauss_newton",
    "cross_gradient",
    "joint_inversion",
]


def _torch():
    import torch

    return torch


# --------------------------------------------------------------------------------------------------
# Forward operators
# --------------------------------------------------------------------------------------------------
def dc_resistivity(log_sigma, shape, schedule, *, spacing=1.0, sigma_ref=1.0, log_data=True, clamp=12.0):
    """DC-resistivity (ERT) forward: transfer resistances for a quadrupole measurement schedule.

    Solves the steady current-flow equation ``-div(sigma grad phi) = I`` (Poisson) on the structured grid,
    with the package's Dirichlet box acting as the far-field ground -- so electrodes must be *interior*
    nodes (surface or borehole), which is the physical setup. ``sigma = sigma_ref * exp(log_sigma)`` keeps
    conductivity positive and makes ``log_sigma`` the natural (log-) inversion parameter.

    Args:
        log_sigma: torch tensor, per-node log-conductivity contrast (length ``prod(shape)``). Gradients flow.
        shape: grid shape ``(nx, ny[, nz])``.
        schedule: sequence of quadrupoles ``(a, b, m, n)`` -- current injected at nodes ``a`` (+) and ``b``
            (-), potential measured between ``m`` and ``n``. ``b`` and/or ``n`` may be ``None`` for
            pole-pole / pole-dipole arrays.
        spacing: grid spacing (scalar or per-axis).
        sigma_ref: reference conductivity multiplying ``exp(log_sigma)``.
        log_data: if True (default) return ``log|R|`` (the stable, well-scaled data for inversion); else ``R``.
        clamp: bound ``|log_sigma|`` before exponentiating, so an extreme iterate during a Gauss-Newton line
            search cannot produce an infinite/zero conductivity and a singular factor. Gradients still flow on
            the unclamped interior; set to ``None`` to disable.

    Returns:
        torch tensor of length ``len(schedule)`` -- the (log) transfer resistances ``R = (phi_m - phi_n)/I``.
    """
    from pysparkplug_pde.pde_solve import divergence_form, sparse_solve

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    cell = float(np.prod(np.atleast_1d(spacing)))
    if clamp is not None:
        log_sigma = torch.clamp(log_sigma, -float(clamp), float(clamp))
    sigma = sigma_ref * torch.exp(log_sigma)
    rows, cols, vals, nn = divergence_form(sigma, shape, spacing=spacing)
    # group measurements by current injection so we factorize/solve once per unique (a, b)
    inj_of = {}
    for q in schedule:
        a, b = q[0], q[1]
        inj_of.setdefault((a, b), None)
    pot = {}
    for (a, b) in inj_of:
        rhs = torch.zeros(n, dtype=sigma.dtype)
        rhs[a] = 1.0 / cell
        if b is not None:
            rhs[b] = -1.0 / cell
        pot[(a, b)] = sparse_solve(vals, rows, cols, nn, rhs)
    out = []
    for q in schedule:
        a, b, m = q[0], q[1], q[2]
        nn_e = q[3] if len(q) > 3 else None
        phi = pot[(a, b)]
        r = phi[m] - (phi[nn_e] if nn_e is not None else 0.0)
        out.append(r)
    r = torch.stack(out)
    if log_data:
        return torch.log(torch.abs(r) + 1e-12)
    return r


def straight_ray_operator(shape, sources, receivers, *, spacing=1.0, n_seg=64, pairs=None):
    """Sparse ray-length matrix ``L`` (n_rays x n_cells) for straight-ray traveltime tomography.

    Each row is a source->receiver ray discretized into ``n_seg`` segments and accumulated into the grid
    cells it crosses, so ``traveltime = L @ slowness``. This is the linear forward for first-arrival
    crosshole / surface GPR and seismic tomography. By default every (source, receiver) pair is used; pass
    ``pairs`` (a list of ``(i, j)`` index pairs into ``sources``/``receivers``) to use a subset.

    Args:
        shape: grid shape ``(nx, ny[, nz])``.
        sources, receivers: ``(ns, d)`` and ``(nr, d)`` physical coordinates.
        spacing: grid spacing (scalar or per-axis).
        n_seg: segments per ray.
        pairs: optional explicit list of ``(src_index, rcv_index)`` pairs.

    Returns:
        scipy.sparse.csr_matrix of shape ``(n_rays, prod(shape))``.
    """
    shape = tuple(int(s) for s in shape)
    d = len(shape)
    sp_ = np.atleast_1d(spacing).astype(float)
    if sp_.size == 1:
        sp_ = np.full(d, sp_[0])
    src = np.asarray(sources, float)
    rcv = np.asarray(receivers, float)
    strides = np.array([int(np.prod(shape[k + 1:])) for k in range(d)])
    if pairs is None:
        pairs = [(i, j) for i in range(len(src)) for j in range(len(rcv))]
    t = np.linspace(0.0, 1.0, n_seg)[:, None]
    rows, cols, data = [], [], []
    for ridx, (i, j) in enumerate(pairs):
        a, b = src[i], rcv[j]
        pts = a + (b - a) * t
        seg_len = np.linalg.norm(b - a) / (n_seg - 1)
        idx = np.clip(np.round(pts / sp_).astype(int), 0, np.array(shape) - 1)
        flat = idx @ strides
        binc = np.bincount(flat, minlength=int(np.prod(shape)))
        nz = np.nonzero(binc)[0]
        rows.extend([ridx] * len(nz))
        cols.extend(nz.tolist())
        data.extend((binc[nz] * seg_len).tolist())
    return sp.csr_matrix((data, (rows, cols)), shape=(len(pairs), int(np.prod(shape))))


def roughness_operator(shape, *, spacing=1.0):
    """First-difference (gradient) operator ``R`` on a structured grid: one row per interior face, with
    ``+1/h`` and ``-1/h`` on the two adjacent cells. ``||R m||^2`` is the standard smoothness penalty.

    Returns:
        scipy.sparse.csr_matrix of shape ``(n_faces, prod(shape))``.
    """
    from pysp.ppl._grid import _grid_faces

    shape = tuple(int(s) for s in shape)
    g = _grid_faces(shape, spacing)
    fa, fb, fw = g["face_a"], g["face_b"], np.sqrt(np.asarray(g["face_w"], float))
    nf = len(fa)
    n = g["n"]
    rows = np.concatenate([np.arange(nf), np.arange(nf)])
    cols = np.concatenate([fa, fb])
    data = np.concatenate([fw, -fw])
    return sp.csr_matrix((data, (rows, cols)), shape=(nf, n))


# --------------------------------------------------------------------------------------------------
# Regularized Gauss-Newton inversion (the workhorse)
# --------------------------------------------------------------------------------------------------
def regularized_gauss_newton(
    forward: Callable,
    data,
    x0,
    *,
    noise=1.0,
    beta: float = 1.0,
    roughness=None,
    ref=None,
    lower: float | None = None,
    upper: float | None = None,
    n_iter: int = 12,
    jac_every: int = 1,
    line_search: int = 25,
    rtol: float = 1e-4,
    verbose: bool = False,
):
    r"""Occam-style regularized Gauss-Newton for any differentiable forward ``forward(x) -> data``.

    Minimizes :math:`\tfrac12\lVert (forward(x)-d)/\sigma\rVert^2 + \tfrac{\beta}{2}\lVert R(x-x_\mathrm{ref})\rVert^2`
    by Gauss-Newton with a backtracking line search. The smoothness operator ``R`` (default the grid
    first-difference, via :func:`roughness_operator`, when ``shape`` can be inferred -- otherwise damping)
    fills the data null space, which is exactly what makes severely ill-posed potential-field problems
    (DC resistivity, gravity, magnetics) invert robustly without per-problem prior tuning. The data
    sensitivity ``J^T J`` supplies depth/spatial weighting automatically.

    Args:
        forward: differentiable ``forward(x_torch) -> torch tensor`` of length ``len(data)``.
        data: observed data (array).
        x0: initial model (array); also sets the model length ``n``.
        noise: data standard deviation (scalar or per-datum array).
        beta: regularization weight.
        roughness: scipy-sparse ``(k, n)`` smoothness operator; if ``None`` uses identity damping.
        ref: reference model the smoothness pulls toward (default zeros).
        lower, upper: optional box bounds on the model.
        n_iter: max Gauss-Newton iterations.
        line_search: max backtracking halvings per iteration.
        rtol: stop when the relative objective decrease falls below this.
        verbose: print per-iteration objective.

    Returns:
        ``(x, std)`` -- the inverted model (numpy) and a linearized posterior standard deviation per cell
        (``sqrt(diag((J^T W J + beta R^T R)^{-1}))``).
    """
    import torch

    x = torch.as_tensor(np.asarray(x0, float)).clone()
    n = x.numel()
    data_t = torch.as_tensor(np.asarray(data, float))
    w = np.broadcast_to(1.0 / np.asarray(noise, float), (len(data_t),)).astype(float)
    w_t = torch.as_tensor(w)
    ref_np = np.zeros(n) if ref is None else np.asarray(ref, float)
    R = sp.eye(n, format="csr") if roughness is None else roughness.tocsr()
    RtR = (R.T @ R).tocsc()
    RtR_dense = np.asarray(RtR.todense())

    def objective(xv):
        r = ((forward(xv) - data_t) * w_t).detach().cpu().numpy()
        rr = R @ (xv.detach().cpu().numpy() - ref_np)
        return 0.5 * float(r @ r) + 0.5 * beta * float(rr @ rr)

    f_prev = objective(x)
    Jw = H = None
    for it in range(n_iter):
        x0d = x.detach()
        pred = forward(x0d)
        # Gauss-Newton; reuse the Jacobian for `jac_every` iterations (quasi-Newton) -- the forward is
        # mildly nonlinear, so recomputing the expensive sensitivity every step is wasteful.
        if Jw is None or it % jac_every == 0:
            J = torch.autograd.functional.jacobian(forward, x0d, vectorize=False).detach().cpu().numpy()
            Jw = J * w[:, None]
            H = Jw.T @ Jw + beta * RtR_dense
        rw = ((pred - data_t) * w_t).detach().cpu().numpy()
        g = Jw.T @ rw + beta * (RtR @ (x0d.cpu().numpy() - ref_np))
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(H, -g, rcond=None)[0]
        step = 1.0
        improved = False
        for _ in range(line_search):
            xn = x0d + step * torch.as_tensor(dx)
            if lower is not None or upper is not None:
                xn = torch.clamp(xn, lower, upper)
            f_new = objective(xn)
            if f_new < f_prev:
                x = xn
                improved = True
                break
            step *= 0.5
        if verbose:
            print(f"  GN iter {it:2d}  objective {f_prev:.4e} -> {objective(x):.4e}  step {step:.3g}")
        if not improved:
            break
        f_now = objective(x)
        if abs(f_prev - f_now) < rtol * abs(f_prev):
            f_prev = f_now
            break
        f_prev = f_now
    # linearized posterior std from the final Gauss-Newton Hessian
    x0d = x.detach()
    J = torch.autograd.functional.jacobian(forward, x0d, vectorize=False).detach().cpu().numpy()
    Jw = J * w[:, None]
    H = Jw.T @ Jw + beta * RtR_dense
    try:
        std = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0.0, None))
    except np.linalg.LinAlgError:
        std = np.full(n, np.nan)
    return x.detach().cpu().numpy(), std


# --------------------------------------------------------------------------------------------------
# Structural (cross-gradient) joint inversion
# --------------------------------------------------------------------------------------------------
def cross_gradient(m1, m2, shape, *, spacing=1.0):
    """Cross-gradient vector ``t = grad(m1) x grad(m2)`` (torch), the structural-similarity measure of
    Gallardo & Meju (2003): it vanishes where the two property models have parallel (or zero) gradients, so
    penalizing ``||t||^2`` forces them to share boundaries without assuming any petrophysical relation
    between their values. Returns the stacked components (one in 2-D, three in 3-D)."""
    from pysparkplug_pde.ops import make_ops

    ops = make_ops()
    shape = tuple(int(s) for s in shape)
    d = len(shape)
    g1 = [ops.grad(m1, shape, k, spacing=spacing) for k in range(d)]
    g2 = [ops.grad(m2, shape, k, spacing=spacing) for k in range(d)]
    if d == 2:
        return (g1[0] * g2[1] - g1[1] * g2[0]).reshape(-1)
    comps = [
        g1[1] * g2[2] - g1[2] * g2[1],
        g1[2] * g2[0] - g1[0] * g2[2],
        g1[0] * g2[1] - g1[1] * g2[0],
    ]
    import torch

    return torch.cat(comps)


def joint_inversion(
    forwards: Sequence[Callable],
    datas: Sequence,
    x0s: Sequence,
    shape,
    *,
    noises: Sequence | None = None,
    betas: Sequence | None = None,
    roughness=None,
    spacing=1.0,
    cross_gradient_weight: float = 0.0,
    bounds: tuple | None = None,
    n_iter: int = 10,
    jac_every: int = 1,
    line_search: int = 25,
    verbose: bool = False,
):
    r"""Joint inversion of several property models, optionally coupled by the cross-gradient.

    Each model ``x_i`` has its own differentiable forward ``forwards[i]`` and data ``datas[i]``; all share the
    grid ``shape``. With ``cross_gradient_weight > 0`` a structural penalty
    :math:`\tfrac{\lambda}{2}\sum_{i<j}\lVert \nabla x_i \times \nabla x_j\rVert^2` couples every pair, so the
    models are driven to share boundaries while each keeps its own value scale (no petrophysical law assumed).
    A block Gauss-Newton step over the stacked model updates all of them together.

    Returns:
        list of inverted models (numpy arrays), one per forward.
    """
    import torch

    P = len(forwards)
    xs = [torch.as_tensor(np.asarray(x0, float)).clone() for x0 in x0s]
    n = xs[0].numel()
    datas_t = [torch.as_tensor(np.asarray(d, float)) for d in datas]
    noises = noises or [1.0] * P
    betas = betas or [1.0] * P
    ws = [np.broadcast_to(1.0 / np.asarray(noises[i], float), (len(datas_t[i]),)).astype(float) for i in range(P)]
    R = sp.eye(n, format="csr") if roughness is None else roughness.tocsr()
    RtR = np.asarray((R.T @ R).todense())
    lam = float(cross_gradient_weight)
    lo, hi = (bounds if bounds is not None else (None, None))

    def xg_pen(xlist):
        if lam <= 0:
            return 0.0
        s = 0.0
        for i in range(P):
            for j in range(i + 1, P):
                t = cross_gradient(xlist[i], xlist[j], shape, spacing=spacing)
                s += float((t * t).sum())
        return 0.5 * lam * s

    def objective(xlist):
        tot = 0.0
        for i in range(P):
            r = ((forwards[i](xlist[i]) - datas_t[i]) * torch.as_tensor(ws[i])).detach().cpu().numpy()
            reg = R @ xlist[i].detach().cpu().numpy()
            tot += 0.5 * float(r @ r) + 0.5 * betas[i] * float(reg @ reg)
        return tot + xg_pen(xlist)

    f_prev = objective(xs)
    Hcache = [None] * P
    Jwcache = [None] * P
    for it in range(n_iter):
        # per-model Gauss-Newton block (data + smoothness), with the cross-gradient handled by a gradient step
        new = []
        # cross-gradient gradient wrt each model (autograd through the coupling)
        xg_grads = [np.zeros(n) for _ in range(P)]
        if lam > 0:
            xv = [x.detach().clone().requires_grad_(True) for x in xs]
            pen = torch.as_tensor(0.0)
            for i in range(P):
                for j in range(i + 1, P):
                    t = cross_gradient(xv[i], xv[j], shape, spacing=spacing)
                    pen = pen + 0.5 * lam * (t * t).sum()
            pen.backward()
            xg_grads = [xv[i].grad.detach().cpu().numpy() for i in range(P)]
        for i in range(P):
            x0d = xs[i].detach()
            pred = forwards[i](x0d)
            if Jwcache[i] is None or it % jac_every == 0:
                J = torch.autograd.functional.jacobian(forwards[i], x0d, vectorize=False).detach().cpu().numpy()
                Jwcache[i] = J * ws[i][:, None]
                Hcache[i] = Jwcache[i].T @ Jwcache[i] + betas[i] * RtR + lam * np.eye(n) * 1e-6
            Jw, H = Jwcache[i], Hcache[i]
            rw = ((pred - datas_t[i]) * torch.as_tensor(ws[i])).detach().cpu().numpy()
            g = Jw.T @ rw + betas[i] * (RtR @ x0d.cpu().numpy()) + xg_grads[i]
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(H, -g, rcond=None)[0]
            new.append(dx)
        step = 1.0
        improved = False
        for _ in range(line_search):
            cand = []
            for i in range(P):
                xc = xs[i].detach() + step * torch.as_tensor(new[i])
                if lo is not None or hi is not None:
                    xc = torch.clamp(xc, lo, hi)
                cand.append(xc)
            if objective(cand) < f_prev:
                xs = cand
                improved = True
                break
            step *= 0.5
        if verbose:
            print(f"  joint GN iter {it:2d}  objective {f_prev:.4e} -> {objective(xs):.4e}")
        if not improved:
            break
        f_prev = objective(xs)
    return [x.detach().cpu().numpy() for x in xs]
