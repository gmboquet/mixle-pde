"""Steady Smoluchowski diffusion-limited association rates for protein/ligand binding.

The diffusion-controlled on-rate of a ligand reaching a reactive target obeys the steady Smoluchowski
equation with a potential of mean force ``W(x)`` (in units of ``k_B T``). Written for the survival
probability ``p`` (fraction of trajectories not yet absorbed), the drift-diffusion operator is
self-adjoint after the change of variables ``q = e^{W} p``:

    div[ D e^{-W} grad( e^{W} p ) ] = 0,

with an absorbing (reactive) boundary ``p = 0`` on the target surface and ``p -> 1`` in the bulk. The
substitution collapses the drift term into a symmetric variable-coefficient divergence operator, so the
same ``-div(kappa grad)`` assembly used for conduction/diffusion applies with the reweighted conductivity

    kappa = D e^{-W}.

Solving for ``q`` and reporting the total diffusive flux into the absorbing boundary (divided by the bulk
concentration) gives the association rate ``k_on``. Two forms are provided:

* :func:`smoluchowski_rate_radial` -- a spherically symmetric target: an exact 1-D finite-volume solve on
  ``r in [a, R]`` (spherical shell conductances ``r^2 D e^{-W}``). Cleanest verification; recovers the
  Debye-Smoluchowski limit ``4 pi D a`` and the Debye interaction factor to ~1 percent.
* :func:`smoluchowski_rate_box` -- an absorbing sphere embedded in a 3-D Cartesian box, reusing
  :func:`mixle_pde.pde_solve.divergence_form` with ``kappa = D e^{-W}`` directly. Staircased geometry, so
  it recovers ``4 pi D a`` to ~5-10 percent.

Both are differentiable in ``D`` and the ``W`` parameters via :func:`mixle_pde.pde_solve.sparse_solve`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "smoluchowski_rate_radial",
    "smoluchowski_rate_box",
    "smoluchowski_debye_factor",
    "smoluchowski_rate_free",
]


def _torch():
    import torch

    return torch


def _as_field(x, n, torch):
    """Broadcast a scalar or array to a length-``n`` torch float64 field (gradients flow through)."""
    if torch.is_tensor(x):
        t = x.to(torch.float64).reshape(-1)
        if t.shape[0] == 1:
            return t.expand(int(n)).clone()
        return t
    if np.isscalar(x) or (hasattr(x, "ndim") and getattr(x, "ndim", 1) == 0):
        return torch.full((int(n),), float(x), dtype=torch.float64)
    t = torch.as_tensor(np.asarray(x, dtype=float).reshape(-1), dtype=torch.float64)
    if t.shape[0] != int(n):
        raise ValueError(f"field length {t.shape[0]} != number of nodes {int(n)}.")
    return t


def smoluchowski_rate_radial(D, a, R, *, n=400, W=None):
    """Steady Smoluchowski on-rate for a spherically symmetric absorbing target of radius ``a``.

    Solves ``d/dr[ r^2 D e^{-W} dq/dr ] = 0`` for ``q = e^{W} p`` on ``[a, R]`` by a spherical
    finite-volume scheme (face conductance ``r_face^2 D e^{-W(r_face)} / h``), with ``q(a) = 0`` (absorbing)
    and ``p(R) = 1`` (bulk, so ``q(R) = e^{W(R)}``). The rate is the total diffusive flux into the absorbing
    inner boundary, ``k_on = 4 pi r^2 D e^{-W} dq/dr`` evaluated at the innermost face -- a constant of the
    steady problem.

    Args:
        D: diffusion constant (scalar torch tensor or float; gradients flow if a tensor).
        a: absorbing radius.
        R: outer (bulk) radius; take ``R >> a`` to approach the infinite-domain limit.
        n: number of radial nodes.
        W: potential of mean force. ``None`` (free diffusion), a scalar, a length-``n`` array/tensor of
            nodal values, or a callable ``W(r) -> array`` (evaluated on the radial nodes). In ``k_B T``.

    Returns:
        scalar torch tensor ``k_on`` (differentiable in ``D`` and any tensor entries of ``W``).
    """
    from mixle_pde.pde_solve import sparse_solve

    torch = _torch()
    a = float(a)
    R = float(R)
    if not R > a:
        raise ValueError(f"outer radius R={R} must exceed absorbing radius a={a}.")
    n = int(n)
    r = torch.linspace(a, R, n, dtype=torch.float64)
    h = (R - a) / (n - 1)
    D_t = D if torch.is_tensor(D) else torch.as_tensor(float(D), dtype=torch.float64)

    if W is None:
        W_nodes = torch.zeros(n, dtype=torch.float64)
    elif callable(W):
        W_nodes = _as_field(W(r), n, torch)
    else:
        W_nodes = _as_field(W, n, torch)

    # Face-centred spherical conductances C_{i+1/2} = r_face^2 D e^{-W(r_face)} / h (nodes i, i+1).
    r_face = 0.5 * (r[:-1] + r[1:])
    W_face = 0.5 * (W_nodes[:-1] + W_nodes[1:])
    cond = (r_face**2) * D_t * torch.exp(-W_face) / h  # (n-1,)

    # Assemble the symmetric tridiagonal system for q. Interior node i balances the two adjacent faces;
    # rows 0 and n-1 are Dirichlet identities carrying q(a)=0 and q(R)=e^{W(R)}.
    idx = torch.arange(n)
    diag = torch.zeros(n, dtype=torch.float64)
    diag = diag.index_add(0, idx[:-1], cond).index_add(0, idx[1:], cond)
    # zero out the two boundary diagonals, then overwrite with 1 (identity rows)
    keep = torch.ones(n, dtype=torch.float64)
    keep[0] = 0.0
    keep[-1] = 0.0
    diag = diag * keep

    left = idx[1:-1]  # interior nodes i = 1..n-2 carry the two off-diagonal couplings
    off_lo = cond[: n - 2]  # coupling of interior node i to i-1 (face i-1/2)
    off_hi = cond[1:]  # coupling of interior node i to i+1 (face i+1/2)

    rows = torch.cat([idx, left, left, torch.tensor([0, n - 1])])
    cols = torch.cat([idx, left - 1, left + 1, torch.tensor([0, n - 1])])
    vals = torch.cat(
        [
            diag,
            -off_lo,
            -off_hi,
            torch.ones(2, dtype=torch.float64),  # boundary identity rows
        ]
    )

    b = torch.zeros(n, dtype=torch.float64)
    b[-1] = torch.exp(W_nodes[-1])  # q(R) = e^{W(R)} p(R), p(R)=1
    # b[0] stays 0: q(a) = e^{W(a)} * 0

    q = sparse_solve(vals, rows, cols, n, b)

    # k_on = total flux into the absorbing boundary = 4 pi * innermost-face conductance * (q_0 - q_1) / (-1)
    # Flux (outward from target) = C_{1/2} (q_1 - q_0); influx magnitude 4 pi C_{1/2} (q_1 - q_0).
    flux = cond[0] * (q[1] - q[0])
    return 4.0 * np.pi * flux


def smoluchowski_rate_box(D, a, shape, *, spacing=1.0, W=None, center=None):
    """Steady Smoluchowski on-rate for an absorbing sphere embedded in a 3-D Cartesian box.

    Reuses :func:`mixle_pde.pde_solve.divergence_form` with the reweighted conductivity ``kappa = D e^{-W}``
    (the drift term folded into the symmetric operator), pins ``q = 0`` inside the absorbing sphere and
    ``q = e^{W}`` on the box boundary (bulk ``p = 1``), and solves with
    :func:`mixle_pde.pde_solve.sparse_solve`. The rate is the diffusive flux across the faces bounding the
    absorbing region. Staircased geometry, so expect ~5-10 percent agreement with ``4 pi D a``.

    Args:
        D: diffusion constant (scalar torch tensor or float).
        a: absorbing-sphere radius (in the same length units as ``spacing``).
        shape: grid shape ``(nx, ny, nz)``.
        spacing: grid spacing (scalar or per-axis).
        W: potential of mean force -- ``None``/scalar/per-node array/tensor, or a callable ``W(radii)``
            of the node radii from ``center``. In ``k_B T``.
        center: sphere centre as a coordinate ``(x, y, z)``; defaults to the box centre.

    Returns:
        scalar torch tensor ``k_on`` (differentiable in ``D`` and tensor entries of ``W``).
    """
    from mixle.ppl._grid import _grid_faces

    from mixle_pde.pde_solve import divergence_form, sparse_solve

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3:
        raise ValueError(f"smoluchowski_rate_box is 3-D; got shape {shape}.")
    n = int(np.prod(shape))
    sp = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    D_t = D if torch.is_tensor(D) else torch.as_tensor(float(D), dtype=torch.float64)

    coords = [np.arange(s) * sp[ax] for ax, s in enumerate(shape)]
    X, Y, Z = np.meshgrid(*coords, indexing="ij")
    if center is None:
        center = [(s - 1) * 0.5 * sp[ax] for ax, s in enumerate(shape)]
    rad = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2).ravel()

    if W is None:
        W_nodes = torch.zeros(n, dtype=torch.float64)
    elif callable(W):
        W_nodes = _as_field(W(rad), n, torch)
    else:
        W_nodes = _as_field(W, n, torch)

    kappa = D_t * torch.exp(-W_nodes)
    rows, cols, vals, nn = divergence_form(kappa, shape, spacing=spacing)

    g = _grid_faces(shape, spacing)
    boundary_mask = g["boundary_mask"]
    absorb_mask = rad <= a  # nodes inside the reactive sphere: pinned q = 0
    if not absorb_mask.any():
        raise ValueError(f"absorbing radius a={a} encloses no grid node; refine the grid or enlarge a.")

    # Pin the reactive-sphere nodes (q=0) and the box-boundary nodes (bulk) to Dirichlet identity rows.
    # divergence_form already made the box-boundary rows identity; here we drop every assembled entry whose
    # ROW is pinned and re-add a single identity entry per pinned node, so absorbing nodes become q=0 rows.
    pinned = absorb_mask | boundary_mask
    row_pinned = pinned[rows.detach().cpu().numpy()]
    keep = torch.as_tensor(~row_pinned)
    pinned_idx = torch.as_tensor(np.where(pinned)[0], dtype=torch.long)
    rows_f = torch.cat([rows[keep], pinned_idx])
    cols_f = torch.cat([cols[keep], pinned_idx])
    vals_f = torch.cat([vals[keep], torch.ones(len(pinned_idx), dtype=vals.dtype)])

    b = torch.zeros(n, dtype=torch.float64)
    box_idx = torch.as_tensor(np.where(boundary_mask & ~absorb_mask)[0], dtype=torch.long)
    b[box_idx] = torch.exp(W_nodes[box_idx])  # bulk p=1 -> q=e^{W}; absorbing nodes keep q=0
    q = sparse_solve(vals_f, rows_f, cols_f, nn, b)

    # k_on = total diffusive flux out of the reactive region across the faces straddling its boundary.
    fa = g["face_a"]
    fb = g["face_b"]
    fw = g["face_w"]
    ci = np.where(absorb_mask[fa] ^ absorb_mask[fb])[0]  # faces with exactly one absorbing endpoint
    fa_t = torch.as_tensor(fa[ci], dtype=torch.long)
    fb_t = torch.as_tensor(fb[ci], dtype=torch.long)
    face_cond = 0.5 * (kappa[fa_t] + kappa[fb_t]) * torch.as_tensor(fw[ci], dtype=torch.float64)
    absorb_is_a = torch.as_tensor(absorb_mask[fa[ci]])
    q_out = torch.where(absorb_is_a, q[fb_t], q[fa_t])
    q_in = torch.where(absorb_is_a, q[fa_t], q[fb_t])  # = 0 at convergence
    return (face_cond * (q_out - q_in)).sum()


def smoluchowski_debye_factor(a, R, W):
    """Closed-form Debye interaction factor ``f = [a * integral_a^R e^{W(r)} / r^2 dr]^{-1}``.

    The steady Smoluchowski rate with a centrally symmetric ``W(r)`` is ``k = 4 pi D a f``, i.e. ``f`` is
    the rate relative to the free Debye-Smoluchowski value ``4 pi D a``. ``W`` is a callable ``W(r)`` (in
    ``k_B T``); the integral is done by high-resolution quadrature on ``[a, R]``.
    """
    a = float(a)
    R = float(R)
    rr = np.linspace(a, R, 200001)
    integrand = np.exp(np.asarray(W(rr), dtype=float)) / rr**2
    trap = getattr(np, "trapezoid", None) or np.trapz
    integral = trap(integrand, rr)
    return 1.0 / (a * integral)


def smoluchowski_rate_free(D, a):
    """Debye-Smoluchowski free-diffusion rate ``k = 4 pi D a`` (perfectly absorbing sphere, ``W = 0``)."""
    return 4.0 * np.pi * float(D) * float(a)
