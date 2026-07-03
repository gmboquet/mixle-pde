"""Frequency-domain Helmholtz with a perfectly-matched-layer boundary and complex (attenuating) modulus.

The existing :func:`mixle_pde.pde_solve.helmholtz_operator` is real and Dirichlet-box: an outgoing wave
reflects off the wall and comes back, so a frequency-domain full-waveform inversion sees a domain full of
spurious echoes and its gradients are corrupted. This module replaces the hard wall with a perfectly-matched
layer (PML): near each edge the coordinate is analytically continued into the complex plane,
``s_x(x) = 1 + i d_x(x) / omega``, which leaves the interior equation untouched but turns an outgoing wave in
the layer into an exponentially decaying one, with (in the continuum) zero reflection at any angle.

Multiplying the stretched Helmholtz equation ``(1/s_x d_x(1/s_x d_x u) + 1/s_z d_z(1/s_z d_z u)) + omega^2 m u
= -f`` through by ``s_x s_z`` gives the complex-symmetric divergence form

    -div(A grad u) - s_x s_z omega^2 m u = s_x s_z f,   A = diag(s_z / s_x, s_x / s_z),

which is assembled node-by-node with the anisotropic face coefficients (``s_z/s_x`` on x-faces, ``s_x/s_z`` on
z-faces evaluated at the face midpoint) and the ``s_x s_z omega^2 m`` mass term on the diagonal. The result is
returned as ``(rows, cols, vals, n)`` with complex ``vals`` for the complex-correct adjoint
:func:`mixle_pde.pde_solve.sparse_solve`, so the whole map is differentiable in the squared-slowness /
modulus field ``m``.

Attenuation rides on ``m``: passing a finite quality factor ``Q`` makes the modulus complex,
``m -> m (1 + i / Q)``, which is the standard viscoacoustic loss and produces the amplitude decay
``exp(-omega r / (2 Q c))`` along the propagation direction.
"""

from __future__ import annotations

import numpy as np

__all__ = ["helmholtz_pml_operator", "solve_helmholtz_pml", "pml_profile"]


def _torch():
    import torch

    return torch


def pml_profile(shape, spacing, pml_width, pml_strength, *, torch=None):
    """Per-axis damping profiles ``d_ax(x)`` of the PML, one flat array per axis (length ``prod(shape)``).

    A polynomial (quadratic) ramp that is zero in the interior and rises to ``pml_strength`` at the outer
    edge, over ``pml_width`` nodes on each side of each axis: ``d(t) = pml_strength * t^2`` with ``t`` the
    fractional depth into the layer. Returned as real torch tensors so the stretch ``s = 1 + i d / omega`` is
    differentiable-through (the profile itself is data, not a parameter).

    Args:
        shape: grid shape ``(nx, nz)``.
        spacing: grid spacing (scalar or per-axis); unused here but kept for signature parity.
        pml_width: layer thickness in nodes, a scalar or per-axis sequence.
        pml_strength: peak damping ``d`` at the outer edge, scalar or per-axis.
        torch: backend module (defaults to the package torch import).

    Returns:
        list of ``ndim`` flat real tensors ``d_ax`` giving the damping at every node along that axis.
    """
    torch = torch or _torch()
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    widths = np.broadcast_to(np.asarray(pml_width, dtype=float), (ndim,))
    strengths = np.broadcast_to(np.asarray(pml_strength, dtype=float), (ndim,))
    profiles = []
    for ax in range(ndim):
        w = float(widths[ax])
        s0 = float(strengths[ax])
        na = shape[ax]
        coord = np.arange(na, dtype=float)
        d1 = np.zeros(na)
        if w > 0.0:
            # distance into the left / right layer (in nodes), zero in the interior
            left = np.clip((w - coord) / w, 0.0, 1.0)
            right = np.clip((coord - (na - 1 - w)) / w, 0.0, 1.0)
            t = np.maximum(left, right)
            d1 = s0 * t**2
        # broadcast the 1-D axis profile to the full grid, C-order flatten
        bshape = [1] * ndim
        bshape[ax] = na
        dfull = np.broadcast_to(d1.reshape(bshape), shape).reshape(-1).copy()
        profiles.append(torch.as_tensor(dfull, dtype=torch.float64))
    return profiles


def helmholtz_pml_operator(m, shape, *, omega, spacing=1.0, pml_width, pml_strength, Q=None, torch=None):
    """Assemble the PML Helmholtz operator ``-div(A grad u) - s_x s_z omega^2 m u`` as ``(rows, cols, vals, n)``.

    ``m`` is the squared-slowness / modulus field ``1/c(x)^2`` (per node, length ``prod(shape)``); ``omega`` the
    angular frequency. The complex coordinate stretch ``s_ax = 1 + i d_ax / omega`` (with the polynomial PML
    profile :func:`pml_profile`) makes the operator absorbing on all four edges. The anisotropic face
    coefficient is ``s_z/s_x`` on x-faces and ``s_x/s_z`` on z-faces (each evaluated at the face midpoint by the
    arithmetic mean of the two node stretches), and the mass term ``s_x s_z omega^2 m`` sits on the diagonal.
    The outer ring of nodes is a Dirichlet identity row (the field is fully absorbed there), so the source sets
    those to zero.

    With ``Q`` finite the modulus is made complex, ``m -> m (1 + i/Q)``, the standard viscoacoustic loss.

    Args:
        m: per-node squared slowness ``1/c^2`` (torch tensor, gradients flow, or array-like).
        shape: grid shape ``(nx, nz)``.
        omega: angular frequency ``2 pi f`` (rad/s).
        spacing: grid spacing (scalar or per-axis, m).
        pml_width: PML thickness in nodes (scalar or per-axis).
        pml_strength: peak PML damping (scalar or per-axis); larger absorbs more per node.
        Q: quality factor for viscoacoustic attenuation; ``None`` for lossless.
        torch: backend module (defaults to the package torch import).

    Returns:
        ``(rows, cols, vals, n)`` for :func:`mixle_pde.pde_solve.sparse_solve`; ``vals`` complex, differentiable in ``m``.
    """
    torch = torch or _torch()
    shape = tuple(int(s) for s in shape)
    if len(shape) != 2:
        raise ValueError(f"helmholtz_pml_operator is 2-D; got shape {shape}.")
    nx, nz = shape
    n = nx * nz
    sp = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
    hx, hz = float(sp[0]), float(sp[1])

    m = m if torch.is_tensor(m) else torch.as_tensor(np.asarray(m, dtype=float))
    m = m.to(torch.complex128)
    if Q is not None:
        m = m * (1.0 + 1j / float(Q))

    d = pml_profile(shape, spacing, pml_width, pml_strength, torch=torch)
    j = torch.tensor(1j, dtype=torch.complex128)
    s = [1.0 + j * dax.to(torch.complex128) / float(omega) for dax in d]  # s_x, s_z per node
    sx, sz = s[0], s[1]

    idx = np.arange(n).reshape(nx, nz)
    # boundary ring: Dirichlet identity; interior: the stretched divergence + mass
    on_boundary = np.zeros((nx, nz), dtype=bool)
    on_boundary[0, :] = on_boundary[-1, :] = True
    on_boundary[:, 0] = on_boundary[:, -1] = True
    interior_mask = ~on_boundary
    interior = torch.as_tensor(idx[interior_mask].ravel(), dtype=torch.long)
    boundary = torch.as_tensor(idx[on_boundary].ravel(), dtype=torch.long)

    rows_l, cols_l, vals_l = [], [], []
    diag = torch.zeros(n, dtype=torch.complex128)

    # x-faces (between (i, k) and (i+1, k)); coefficient s_z / s_x at the face midpoint / hx^2
    fa = torch.as_tensor(idx[:-1, :].ravel(), dtype=torch.long)
    fb = torch.as_tensor(idx[1:, :].ravel(), dtype=torch.long)
    cx = 0.5 * (sz[fa] + sz[fb]) / (0.5 * (sx[fa] + sx[fb])) / (hx * hx)
    # z-faces (between (i, k) and (i, k+1)); coefficient s_x / s_z at the face midpoint / hz^2
    ga = torch.as_tensor(idx[:, :-1].ravel(), dtype=torch.long)
    gb = torch.as_tensor(idx[:, 1:].ravel(), dtype=torch.long)
    cz = 0.5 * (sx[ga] + sx[gb]) / (0.5 * (sz[ga] + sz[gb])) / (hz * hz)

    bmask = torch.as_tensor(on_boundary.ravel())
    for a, b, c in ((fa, fb, cx), (ga, gb, cz)):
        a_int = ~bmask[a]
        b_int = ~bmask[b]
        # off-diagonal coupling into interior rows only (boundary rows stay identity)
        rows_l.append(a[a_int])
        cols_l.append(b[a_int])
        vals_l.append(-c[a_int])
        rows_l.append(b[b_int])
        cols_l.append(a[b_int])
        vals_l.append(-c[b_int])
        diag = diag.index_add(0, a[a_int], c[a_int]).index_add(0, b[b_int], c[b_int])

    # mass term -s_x s_z omega^2 m on interior diagonal
    mass = -(omega**2) * (sx * sz) * m
    diag = diag + torch.zeros(n, dtype=torch.complex128).index_add(0, interior, mass[interior])

    rows_l.append(interior)
    cols_l.append(interior)
    vals_l.append(diag[interior])
    rows_l.append(boundary)
    cols_l.append(boundary)
    vals_l.append(torch.ones(len(boundary), dtype=torch.complex128))

    rows = torch.cat(rows_l)
    cols = torch.cat(cols_l)
    vals = torch.cat(vals_l)
    return rows, cols, vals, n


def solve_helmholtz_pml(m, shape, source, *, omega, spacing=1.0, pml_width, pml_strength, Q=None, torch=None):
    """Solve the PML Helmholtz problem for a source ``f`` and return the complex field ``u`` (flat, length n).

    Assembles :func:`helmholtz_pml_operator` and solves ``L u = s_x s_z f`` with the adjoint-capable
    :func:`mixle_pde.pde_solve.sparse_solve`. The right-hand side is pre-multiplied by ``s_x s_z`` to match the
    operator's ``-div(A grad) - s_x s_z omega^2 m`` scaling (in the interior ``s_x s_z = 1``, so a point source
    is unchanged there). Boundary source entries are ignored (those rows are the absorbing Dirichlet identity).

    Args:
        m: per-node squared slowness ``1/c^2`` (torch tensor or array-like); gradients flow.
        shape: grid shape ``(nx, nz)``.
        source: right-hand side ``f`` -- an ``int`` flat index for a unit point source, or a full ``(n,)`` field.
        omega: angular frequency (rad/s).
        spacing: grid spacing (scalar or per-axis).
        pml_width, pml_strength: PML thickness (nodes) and peak damping.
        Q: quality factor for viscoacoustic attenuation; ``None`` for lossless.
        torch: backend module (defaults to the package torch import).

    Returns:
        complex torch tensor ``u`` of length ``prod(shape)``, differentiable in ``m``.
    """
    from mixle_pde.pde_solve import sparse_solve

    torch = torch or _torch()
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    rows, cols, vals, n = helmholtz_pml_operator(
        m, shape, omega=omega, spacing=spacing, pml_width=pml_width, pml_strength=pml_strength, Q=Q, torch=torch
    )
    if isinstance(source, (int, np.integer)):
        b = torch.zeros(n, dtype=torch.complex128)
        b[int(source)] = 1.0
    else:
        b = source if torch.is_tensor(source) else torch.as_tensor(np.asarray(source))
        b = b.to(torch.complex128)
    # scale the rhs by s_x s_z (matches the operator; identity in the interior where the source lives)
    d = pml_profile(shape, spacing, pml_width, pml_strength, torch=torch)
    jj = torch.tensor(1j, dtype=torch.complex128)
    sx = 1.0 + jj * d[0].to(torch.complex128) / float(omega)
    sz = 1.0 + jj * d[1].to(torch.complex128) / float(omega)
    b = b * (sx * sz)
    return sparse_solve(vals, rows, cols, n, b)
