"""Near-surface geophysical forward operators and a regularized inversion engine.

The adjoint sparse stack (:mod:`mixle_pde.pde_solve`) gives differentiable PDE *forwards*, and
:class:`mixle_pde.inverse.Differential` wraps a forward as a likelihood for ``mixle.ppl``'s
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
nothing here patches mixle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp

__all__ = [
    "dc_resistivity",
    "straight_ray_operator",
    "eikonal_traveltime",
    "traveltime_tomography",
    "magnetic_dipole_sensitivity",
    "gravity_point_sensitivity",
    "depth_weighting",
    "roughness_operator",
    "regularized_gauss_newton",
    "cross_gradient",
    "joint_inversion",
]


def _torch():
    import torch

    return torch


def field_direction(inclination_deg, declination_deg):
    """Unit vector of the geomagnetic field in an (east, north, up) frame from inclination/declination
    (degrees; inclination positive *down*, declination clockwise from north)."""
    inc, dec = np.radians(inclination_deg), np.radians(declination_deg)
    return np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])


def magnetic_dipole_sensitivity(obs, cells, volumes, *, inclination, declination, field_nt=50000.0):
    r"""Linear sensitivity matrix ``G`` (n_obs x n_cells) of the magnetic **total-field anomaly** to cell
    susceptibility, under the induced-magnetization dipole approximation (each cell a point dipole along the
    ambient field). With susceptibility ``kappa`` (SI, dimensionless), ``d = G @ kappa`` gives the anomaly in nT.

    For a cell of volume ``V`` at displacement ``r`` from an observation point, with field unit vector ``b``
    and strength ``T0``: ``dT = (T0 V / 4pi) (3 (b.r_hat)^2 - 1) / |r|^3 * kappa`` -- the standard dipole
    total-field kernel. This is the cheap, robust approximation good for coarse meshes and observations above
    the cells; the exact rectangular-prism formula (Bhattacharyya 1964) is the rigorous alternative.

    Args:
        obs: (n_obs, 3) observation coordinates (east, north, up), metres.
        cells: (n_cells, 3) cell-centre coordinates, metres.
        volumes: (n_cells,) cell volumes, m^3 (scalar broadcast allowed).
        inclination, declination: geomagnetic field inclination/declination, degrees.
        field_nt: ambient field strength T0, nT.

    Returns:
        ``G`` (n_obs, n_cells) such that ``G @ kappa`` is the total-field anomaly (nT).
    """
    b = field_direction(inclination, declination)
    obs = np.asarray(obs, float)
    cells = np.asarray(cells, float)
    V = np.broadcast_to(np.asarray(volumes, float), (len(cells),))
    d = obs[:, None, :] - cells[None, :, :]  # (n_obs, n_cells, 3) displacement obs<-cell
    r = np.maximum(np.linalg.norm(d, axis=2), 1e-6)
    bdotr = (d / r[:, :, None]) @ b  # (n_obs, n_cells)
    return (field_nt / (4.0 * np.pi)) * V[None, :] * (3.0 * bdotr**2 - 1.0) / r**3


def gravity_point_sensitivity(obs, cells, volumes):
    r"""Linear sensitivity matrix ``G`` (n_obs x n_cells) of the **vertical gravity anomaly** to cell density
    contrast, under the point-mass approximation (each cell a point mass at its centre). With density contrast
    ``rho`` (kg/m^3), ``d = G @ rho`` gives the anomaly in **mGal**.

    For a cell of volume ``V`` at displacement ``r`` from an observation point (vertical offset ``dz``, with
    ``z`` up so the cell is below): ``g_z = 1e5 * G_grav * V * dz / |r|^3 * rho`` (the ``1e5`` converts
    m/s^2 to mGal). Linear in ``rho``. The exact rectangular-prism formula (Nagy 1966 / Plouff 1976) is the
    rigorous alternative for coarse meshes near observations.

    Args:
        obs: (n_obs, 3) observation coordinates (east, north, up), metres.
        cells: (n_cells, 3) cell-centre coordinates, metres.
        volumes: (n_cells,) cell volumes, m^3 (scalar broadcast allowed).

    Returns:
        ``G`` (n_obs, n_cells) such that ``G @ rho`` is the vertical gravity anomaly (mGal).
    """
    G_GRAV = 6.674e-11
    obs = np.asarray(obs, float)
    cells = np.asarray(cells, float)
    V = np.broadcast_to(np.asarray(volumes, float), (len(cells),))
    d = obs[:, None, :] - cells[None, :, :]
    r = np.maximum(np.linalg.norm(d, axis=2), 1e-6)
    return 1.0e5 * G_GRAV * V[None, :] * d[:, :, 2] / r**3


def depth_weighting(cell_z, z0, *, nu=3.0, eps=None):
    r"""Li & Oldenburg (1996, 1998) depth weighting ``w(z) = (|z - z0| + eps)^{-nu/2}``, normalized by its
    maximum. Potential-field kernels decay with depth, so an unweighted inversion piles all structure at the
    surface; folding ``w`` into the model regularization compensates. Use ``nu=3`` for magnetics, ``nu=2`` for
    gravity (the Li & Oldenburg / SimPEG values).

    Args:
        cell_z: (n_cells,) vertical coordinate of each cell centre (same sign convention as ``z0``).
        z0: reference level (e.g. the observation height).
        nu: exponent (3 magnetics, 2 gravity).
        eps: offset preventing a singularity at ``z = z0``; defaults to half the smallest cell spacing.

    Returns:
        (n_cells,) weights in (0, 1], largest at depth.
    """
    z = np.asarray(cell_z, float)
    if eps is None:
        dz = np.abs(np.diff(np.unique(np.round(z, 6))))
        eps = 0.5 * (dz.min() if len(dz) else 1.0)
    w = (np.abs(z - z0) + eps) ** (-nu / 2.0)
    return w / w.max()


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
    from mixle_pde.pde_solve import divergence_form, sparse_solve

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
    for a, b in inj_of:
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
    strides = np.array([int(np.prod(shape[k + 1 :])) for k in range(d)])
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


def _fsm_2d(s, T, si, sj, h, n_cycles):
    """Fast-sweeping eikonal solve |grad T| = s on a 2-D grid (Godunov upwind, alternating sweeps)."""
    nx, nz = s.shape
    for _ in range(n_cycles):
        for di in range(2):
            for dj in range(2):
                for ii in range(nx):
                    i = ii if di == 0 else nx - 1 - ii
                    for jj in range(nz):
                        j = jj if dj == 0 else nz - 1 - jj
                        if i == si and j == sj:
                            continue
                        if i == 0:
                            a = T[1, j]
                        elif i == nx - 1:
                            a = T[nx - 2, j]
                        else:
                            a = min(T[i - 1, j], T[i + 1, j])
                        if j == 0:
                            b = T[i, 1]
                        elif j == nz - 1:
                            b = T[i, nz - 2]
                        else:
                            b = min(T[i, j - 1], T[i, j + 1])
                        f = s[i, j] * h
                        if abs(a - b) >= f:
                            tn = min(a, b) + f
                        else:
                            tn = 0.5 * (a + b + np.sqrt(2.0 * f * f - (a - b) ** 2))
                        if tn < T[i, j]:
                            T[i, j] = tn
    return T


try:
    from numba import njit as _njit

    _fsm_2d_fast = _njit(cache=True, fastmath=True)(_fsm_2d)
except Exception:  # pragma: no cover - numba optional
    _fsm_2d_fast = _fsm_2d


def eikonal_traveltime(slowness, shape, source, *, spacing=1.0, n_cycles=3):
    """First-arrival traveltimes from one source by the **fast-sweeping eikonal** solver (the non-trivial,
    ray-bending forward): solves ``|grad T| = slowness`` on a 2-D grid with ``T=0`` at the source node.

    Args:
        slowness: (n_cells,) per-cell slowness (1/velocity), grid flattened C-order over ``shape``.
        shape: ``(nx, nz)`` grid shape.
        source: flat index of the source node.
        spacing: grid spacing (scalar).
        n_cycles: fast-sweeping cycles (each = 4 directional sweeps); 2-4 suffice.

    Returns:
        (n_cells,) traveltime field, flattened.
    """
    nx, nz = (int(shape[0]), int(shape[1]))
    s = np.ascontiguousarray(np.asarray(slowness, float).reshape(nx, nz))
    T = np.full((nx, nz), 1.0e9)
    si, sj = int(source) // nz, int(source) % nz
    T[si, sj] = 0.0
    T = _fsm_2d_fast(s, T, si, sj, float(spacing), int(n_cycles))
    return T.ravel()


def _backtrace_ray(Tfield, shape, h, recv, src, n_cells, max_steps=4000):
    """Ray path receiver->source as cell path-lengths, following the descent of the traveltime field
    (the Fréchet kernel ``dt/ds_cell = path length in cell``)."""
    nx, nz = shape
    T = Tfield.reshape(nx, nz)
    pos = np.array([recv // nz, recv % nz], float)
    tgt = np.array([src // nz, src % nz], float)
    row = np.zeros(n_cells)
    step = 0.4
    for _ in range(max_steps):
        i, j = int(round(pos[0])), int(round(pos[1]))
        i = min(max(i, 0), nx - 1)
        j = min(max(j, 0), nz - 1)
        gi = (T[min(i + 1, nx - 1), j] - T[max(i - 1, 0), j]) / 2.0
        gj = (T[i, min(j + 1, nz - 1)] - T[i, max(j - 1, 0)]) / 2.0
        g = np.hypot(gi, gj)
        if g < 1e-12:
            break
        pos = pos - step * np.array([gi, gj]) / g
        row[i * nz + j] += step * h
        if np.hypot(pos[0] - tgt[0], pos[1] - tgt[1]) < 1.0:
            break
    return row


def traveltime_tomography(
    times,
    sources,
    receivers,
    shape,
    *,
    spacing=1.0,
    slowness0=None,
    noise=1.0,
    beta=1.0,
    n_iter=8,
    line_search=20,
    n_cycles=3,
    bounds=None,
    verbose=False,
):
    r"""Crosshole/surface first-arrival **traveltime tomography** with the bending-ray eikonal forward.

    Inverts observed traveltimes for a 2-D slowness (1/velocity) field by regularized Gauss-Newton: each
    iteration solves the eikonal once per source (``eikonal_traveltime``), predicts the traveltimes, builds the
    ray-path Jacobian by back-tracing the traveltime gradient (``_backtrace_ray``), and takes a smoothness-
    regularized step. Because the rays *bend* with the updated model, this is genuinely nonlinear -- the
    non-trivial forward, unlike straight-ray tomography.

    Args:
        times: (n_rays,) observed first-arrival traveltimes.
        sources, receivers: (n_rays,) flat grid indices of each ray's source and receiver node.
        shape: ``(nx, nz)`` grid.
        spacing: grid spacing.
        slowness0: initial slowness (scalar or (n_cells,)); default = median apparent slowness.
        noise: per-datum traveltime std (scalar or array).
        beta: smoothness weight.
        n_iter: Gauss-Newton iterations.
        bounds: optional ``(lo, hi)`` slowness bounds.
        verbose: print misfit per iteration.

    Returns:
        ``(slowness, velocity, predicted_times)`` -- the inverted slowness (n_cells,), its reciprocal, and the
        final predicted traveltimes.
    """
    nx, nz = int(shape[0]), int(shape[1])
    N = nx * nz
    times = np.asarray(times, float)
    sources = np.asarray(sources, int)
    receivers = np.asarray(receivers, int)
    w = np.broadcast_to(1.0 / np.asarray(noise, float), times.shape).astype(float)
    if slowness0 is None:
        slowness0 = np.median(
            times
            / np.maximum(
                1e-6,
                np.abs(
                    np.array(
                        [
                            np.hypot((sources[k] // nz - receivers[k] // nz), (sources[k] % nz - receivers[k] % nz))
                            * spacing
                            for k in range(len(times))
                        ]
                    )
                ),
            )
        )
    s = np.full(N, float(slowness0)) if np.isscalar(slowness0) else np.asarray(slowness0, float).copy()
    R = roughness_operator(shape, spacing=spacing)
    RtR = np.asarray((R.T @ R).todense())
    uniq = np.unique(sources)

    def forward_and_jac(sv):
        tpred = np.zeros(len(times))
        G = np.zeros((len(times), N))
        for src in uniq:
            T = eikonal_traveltime(sv, shape, int(src), spacing=spacing, n_cycles=n_cycles)
            idx = np.where(sources == src)[0]
            for k in idx:
                tpred[k] = T[receivers[k]]
                G[k] = _backtrace_ray(T, (nx, nz), spacing, int(receivers[k]), int(src), N)
        return tpred, G

    def misfit(sv):
        tp = np.zeros(len(times))
        for src in uniq:
            T = eikonal_traveltime(sv, shape, int(src), spacing=spacing, n_cycles=n_cycles)
            idx = np.where(sources == src)[0]
            tp[idx] = T[receivers[idx]]
        r = (tp - times) * w
        reg = R @ sv
        return 0.5 * float(r @ r) + 0.5 * beta * float(reg @ reg), tp

    f_prev, _ = misfit(s)
    for it in range(n_iter):
        tpred, G = forward_and_jac(s)
        Gw = G * w[:, None]
        H = Gw.T @ Gw + beta * RtR
        g = Gw.T @ ((tpred - times) * w) + beta * (RtR @ s)
        try:
            ds = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            ds = np.linalg.lstsq(H, -g, rcond=None)[0]
        step = 1.0
        improved = False
        for _ in range(line_search):
            cand = s + step * ds
            if bounds is not None:
                cand = np.clip(cand, bounds[0], bounds[1])
            f_new, _ = misfit(cand)
            if f_new < f_prev:
                s = cand
                f_prev = f_new
                improved = True
                break
            step *= 0.5
        if verbose:
            print(f"  tomography iter {it:2d}  misfit {f_prev:.4e}  step {step:.3g}")
        if not improved:
            break
    tpred, _ = forward_and_jac(s)
    return s, 1.0 / s, tpred


def roughness_operator(shape, *, spacing=1.0):
    """First-difference (gradient) operator ``R`` on a structured grid: one row per interior face, with
    ``+1/h`` and ``-1/h`` on the two adjacent cells. ``||R m||^2`` is the standard smoothness penalty.

    Returns:
        scipy.sparse.csr_matrix of shape ``(n_faces, prod(shape))``.
    """
    from mixle.ppl._grid import _grid_faces

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
    from mixle_pde.ops import make_ops

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


class JointInversionResult(list):
    """Return type of :func:`joint_inversion`: a list of the inverted per-model arrays, unpacked/indexed
    exactly like the historical bare-``list`` return (``m1, m2 = joint_inversion(...)`` is unaffected), plus
    ``objective_history`` -- the total objective after each accepted outer iteration (index 0 is the initial,
    pre-iteration objective) -- so a caller can compare how many iterations two configurations needed to reach
    a given objective value (C7: ``coupling_in_hessian`` convergence-speed check)."""

    objective_history: list


def _cross_gradient_gn_blocks(xs_detached: Sequence, shape, spacing: float, n: int) -> list:
    """Gauss-Newton curvature blocks for the cross-gradient penalty (DR-ALG C7, step 1).

    For every pair ``(i, j)`` linearizes ``t = cross_gradient(x_i, x_j)`` about the current ``xs_detached``:
    ``B_i = dt/dx_i`` holding ``x_j`` fixed, ``B_j = dt/dx_j`` holding ``x_i`` fixed. Returns, per model, the
    accumulated ``B^T B`` curvature (unscaled by ``lam`` -- the caller applies the penalty weight), so adding
    ``lam * blocks[i]`` into ``Hcache[i]`` is the second-order term the RHS-only path only pushes into the
    gradient.
    """
    import torch

    P = len(xs_detached)
    blocks = [np.zeros((n, n)) for _ in range(P)]
    for i in range(P):
        for j in range(i + 1, P):
            xi_d, xj_d = xs_detached[i], xs_detached[j]

            def f_i(a, _xj=xj_d):
                return cross_gradient(a, _xj, shape, spacing=spacing)

            def f_j(b, _xi=xi_d):
                return cross_gradient(_xi, b, shape, spacing=spacing)

            Bi = torch.autograd.functional.jacobian(f_i, xi_d, vectorize=False).detach().cpu().numpy()
            Bj = torch.autograd.functional.jacobian(f_j, xj_d, vectorize=False).detach().cpu().numpy()
            blocks[i] += Bi.T @ Bi
            blocks[j] += Bj.T @ Bj
    return blocks


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
    coupling_in_hessian: bool = True,
):
    r"""Joint inversion of several property models, optionally coupled by the cross-gradient.

    Each model ``x_i`` has its own differentiable forward ``forwards[i]`` and data ``datas[i]``; all share the
    grid ``shape``. With ``cross_gradient_weight > 0`` a structural penalty
    :math:`\tfrac{\lambda}{2}\sum_{i<j}\lVert \nabla x_i \times \nabla x_j\rVert^2` couples every pair, so the
    models are driven to share boundaries while each keeps its own value scale (no petrophysical law assumed).
    A block Gauss-Newton step over the stacked model updates all of them together.

    ``coupling_in_hessian`` (C7): when ``True`` (the default), the Gauss-Newton curvature of the
    cross-gradient penalty (:func:`_cross_gradient_gn_blocks`) is added into each model's Hessian block,
    on top of the existing gradient-only coupling -- second-order structural coupling, which converges in
    fewer outer iterations than pushing the coupling through the gradient alone. Passing ``False`` reproduces
    the previous (gradient-only / ``lam * eye(n) * 1e-6`` Hessian jitter) behaviour bit-for-bit.

    ``bounds`` is ``None``, a single ``(lo, hi)`` applied to every model, or a length-P sequence of per-model
    ``(lo, hi)`` tuples -- use the per-model form when the models live on different scales (e.g. ERT
    log-conductivity vs seismic slowness), so each is clamped to its own physical range.

    Returns:
        `JointInversionResult` -- a list of inverted models (numpy arrays), one per forward, plus an
        `objective_history` attribute.
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
    # bounds: None, a single (lo, hi) applied to every model, or a length-P sequence of per-model (lo, hi)
    # tuples -- the latter is needed when the models live on different scales (e.g. log-conductivity vs slowness).
    if bounds is None:
        bnds = [(None, None)] * P
    elif len(bounds) == P and all(isinstance(b, (tuple, list)) for b in bounds):
        bnds = [tuple(b) for b in bounds]
    else:
        bnds = [tuple(bounds)] * P

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
    history = [f_prev]
    Hcache = [None] * P
    Jwcache = [None] * P
    xg_hess_blocks = None
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
        # cross-gradient GN curvature (C7): recomputed on the same cadence as the per-model Jacobian cache
        refresh_jac = (it % jac_every == 0) or any(J is None for J in Jwcache)
        if coupling_in_hessian and lam > 0 and refresh_jac:
            xg_hess_blocks = _cross_gradient_gn_blocks([x.detach() for x in xs], shape, spacing, n)
        for i in range(P):
            x0d = xs[i].detach()
            pred = forwards[i](x0d)
            if Jwcache[i] is None or it % jac_every == 0:
                J = torch.autograd.functional.jacobian(forwards[i], x0d, vectorize=False).detach().cpu().numpy()
                Jwcache[i] = J * ws[i][:, None]
                Hcache[i] = Jwcache[i].T @ Jwcache[i] + betas[i] * RtR + lam * np.eye(n) * 1e-6
                if coupling_in_hessian and lam > 0 and xg_hess_blocks is not None:
                    Hcache[i] = Hcache[i] + lam * xg_hess_blocks[i]
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
                lo_i, hi_i = bnds[i]
                if lo_i is not None or hi_i is not None:
                    xc = torch.clamp(xc, lo_i, hi_i)
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
        history.append(f_prev)
    result = JointInversionResult(x.detach().cpu().numpy() for x in xs)
    result.objective_history = history
    return result
