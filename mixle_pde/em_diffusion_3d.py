"""3-D diffusive / quasi-static curl-curl electromagnetics on a staggered Yee edge grid (CSEM / MT / AEM).

The induction (eddy-current) regime, the 3-D extension of :mod:`mixle_pde.em_diffusion`. Displacement current
is dropped against conduction (``omega eps << sigma``), so the frequency-domain Maxwell system collapses to the
complex diffusion curl-curl equation

    curl(mu^{-1} curl E) + i omega sigma E = -i omega J_s .

The field is the tangential electric field on the edges of a Yee cell (the lowest-order Nedelec / edge layout):
``Ex`` on x-edges, ``Ey`` on y-edges, ``Ez`` on z-edges. The discrete curl ``C`` maps edge circulations to face
fluxes (edges -> faces), so ``curl(mu^{-1} curl)`` assembles as ``C^T diag(mu^{-1}/area) C`` -- a real symmetric
operator that depends only on the geometry and ``mu``. Conductivity enters only the complex edge-mass diagonal
``i omega sigma_edge`` (``sigma`` averaged from its two adjacent cells onto each edge), exactly as the 2-D TE mass
term does, so ``vals`` is differentiable in ``log_sigma`` through :func:`mixle_pde.pde_solve.sparse_solve`.

The curl-curl operator has a large gradient null space in the resistive / low-frequency limit (``curl grad phi = 0``
for any nodal ``phi``). It is stabilized with a Coulomb-gauge grad-div (divergence-correction) penalty
``- kappa * G M_n^{-1} G^T (sigma-weighted)`` in the Smith (1996) style: a term ``+ tau * G diag(1/sigma_node) D``
that fills the null space with a well-conditioned divergence penalty without changing the physical solution (the
true diffusive field is divergence-free in the conductor, ``div(sigma E) = 0`` away from sources). See
:func:`assemble_curl_curl_3d`.

Two differentiable forwards:

* :func:`mt_3d` -- plane-wave magnetotelluric: impose the tangential surface E on the top face, let it diffuse to
  depth, read the surface impedance ``Z = E_x / H_y`` and hence ``rho_a`` and phase. For a laterally uniform model
  this reproduces the analytic half-space (``rho_a = 1/sigma``, phase ``45`` deg) and the skin-depth decay.
* :func:`csem_3d` -- grounded electric-dipole controlled-source EM (bonus): inject a current source on a set of
  edges and return the complex field. Verified on a uniform whole space; heterogeneous CSEM is the extension.
"""

from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi  # vacuum permeability (H/m); the earth is non-magnetic to first order


def _torch():
    import torch

    return torch


def _edge_layout(shape):
    """Edge index ranges for a Yee grid on ``shape = (nx, ny, nz)`` nodes.

    Returns ``(offx, offy, offz, ntot, (sx, sy, sz))`` where ``off*`` are the flat offsets of the x/y/z edge
    blocks, ``ntot`` the total edge count, and ``s*`` the per-block ``(a, b, c)`` grid shapes:
    x-edges ``(nx-1, ny, nz)``, y-edges ``(nx, ny-1, nz)``, z-edges ``(nx, ny, nz-1)``.
    """
    nx, ny, nz = shape
    sx = (nx - 1, ny, nz)
    sy = (nx, ny - 1, nz)
    sz = (nx, ny, nz - 1)
    nex = (nx - 1) * ny * nz
    ney = nx * (ny - 1) * nz
    nez = nx * ny * (nz - 1)
    return 0, nex, nex + ney, nex + ney + nez, (sx, sy, sz)


def _face_layout(shape):
    """Face index ranges for a Yee grid on ``shape`` nodes.

    x-faces ``(nx, ny-1, nz-1)``, y-faces ``(nx-1, ny, nz-1)``, z-faces ``(nx-1, ny-1, nz)``.
    """
    nx, ny, nz = shape
    fx = (nx, ny - 1, nz - 1)
    fy = (nx - 1, ny, nz - 1)
    fz = (nx - 1, ny - 1, nz)
    nfx = nx * (ny - 1) * (nz - 1)
    nfy = (nx - 1) * ny * (nz - 1)
    nfz = (nx - 1) * (ny - 1) * nz
    return 0, nfx, nfx + nfy, nfx + nfy + nfz, (fx, fy, fz)


def _curl_matrix(shape, hx, hy, hz):
    """Discrete curl ``C`` (edges -> faces) as a scipy CSR incidence matrix with the ``1/h`` metric.

    ``(curl E)_x = dEz/dy - dEy/dz`` on x-faces, and cyclically. Each face row has four edge entries with weights
    ``+-1/h`` for the transverse spacing, so ``C E`` is the centred face circulation of the Yee cell.
    """
    import scipy.sparse as sp

    _, eyoff, ezoff, nedge, (sx, sy, sz) = _edge_layout(shape)
    _, fyoff, fzoff, nface, (fx, fy, fz) = _face_layout(shape)

    def eidx(off, s, i, j, k):
        a, b, c = s
        return off + (i * b + j) * c + k

    def fidx(off, s, i, j, k):
        a, b, c = s
        return off + (i * b + j) * c + k

    rows, cols, vals = [], [], []

    # x-faces (nx, ny-1, nz-1): (Ez[j+1]-Ez[j])/hy - (Ey[k+1]-Ey[k])/hz
    for i in range(fx[0]):
        for j in range(fx[1]):
            for k in range(fx[2]):
                f = fidx(0, fx, i, j, k)
                # Ez edges (nx, ny, nz-1): +Ez(i,j+1,k) - Ez(i,j,k) over hy
                rows += [f, f]
                cols += [eidx(ezoff, sz, i, j + 1, k), eidx(ezoff, sz, i, j, k)]
                vals += [1.0 / hy, -1.0 / hy]
                # Ey edges (nx, ny-1, nz): -Ey(i,j,k+1) + Ey(i,j,k) over hz
                rows += [f, f]
                cols += [eidx(eyoff, sy, i, j, k + 1), eidx(eyoff, sy, i, j, k)]
                vals += [-1.0 / hz, 1.0 / hz]

    # y-faces (nx-1, ny, nz-1): (Ex[k+1]-Ex[k])/hz - (Ez[i+1]-Ez[i])/hx
    for i in range(fy[0]):
        for j in range(fy[1]):
            for k in range(fy[2]):
                f = fidx(fyoff, fy, i, j, k)
                # Ex edges (nx-1, ny, nz): +Ex(i,j,k+1) - Ex(i,j,k) over hz
                rows += [f, f]
                cols += [eidx(0, sx, i, j, k + 1), eidx(0, sx, i, j, k)]
                vals += [1.0 / hz, -1.0 / hz]
                # Ez edges (nx, ny, nz-1): -Ez(i+1,j,k) + Ez(i,j,k) over hx
                rows += [f, f]
                cols += [eidx(ezoff, sz, i + 1, j, k), eidx(ezoff, sz, i, j, k)]
                vals += [-1.0 / hx, 1.0 / hx]

    # z-faces (nx-1, ny-1, nz): (Ey[i+1]-Ey[i])/hx - (Ex[j+1]-Ex[j])/hy
    for i in range(fz[0]):
        for j in range(fz[1]):
            for k in range(fz[2]):
                f = fidx(fzoff, fz, i, j, k)
                # Ey edges (nx, ny-1, nz): +Ey(i+1,j,k) - Ey(i,j,k) over hx
                rows += [f, f]
                cols += [eidx(eyoff, sy, i + 1, j, k), eidx(eyoff, sy, i, j, k)]
                vals += [1.0 / hx, -1.0 / hx]
                # Ex edges (nx-1, ny, nz): -Ex(i,j+1,k) + Ex(i,j,k) over hy
                rows += [f, f]
                cols += [eidx(0, sx, i, j + 1, k), eidx(0, sx, i, j, k)]
                vals += [-1.0 / hy, 1.0 / hy]

    return sp.csr_matrix((vals, (rows, cols)), shape=(nface, nedge))


def _grad_matrix(shape, hx, hy, hz):
    """Discrete gradient ``G`` (nodes -> edges), the topological adjoint of the divergence.

    ``(grad phi)`` on an edge is the difference of the two node values it connects over the edge length. Its range
    is exactly the curl null space (``C G = 0``), so ``G`` spans the modes the grad-div stabilizer must control.
    """
    import scipy.sparse as sp

    nx, ny, nz = shape
    _, eyoff, ezoff, nedge, (sx, sy, sz) = _edge_layout(shape)
    nnode = nx * ny * nz

    def nidx(i, j, k):
        return (i * ny + j) * nz + k

    def eidx(off, s, i, j, k):
        a, b, c = s
        return off + (i * b + j) * c + k

    rows, cols, vals = [], [], []
    # x-edges (nx-1, ny, nz): phi(i+1,j,k) - phi(i,j,k) over hx
    for i in range(sx[0]):
        for j in range(sx[1]):
            for k in range(sx[2]):
                e = eidx(0, sx, i, j, k)
                rows += [e, e]
                cols += [nidx(i + 1, j, k), nidx(i, j, k)]
                vals += [1.0 / hx, -1.0 / hx]
    # y-edges (nx, ny-1, nz)
    for i in range(sy[0]):
        for j in range(sy[1]):
            for k in range(sy[2]):
                e = eidx(eyoff, sy, i, j, k)
                rows += [e, e]
                cols += [nidx(i, j + 1, k), nidx(i, j, k)]
                vals += [1.0 / hy, -1.0 / hy]
    # z-edges (nx, ny, nz-1)
    for i in range(sz[0]):
        for j in range(sz[1]):
            for k in range(sz[2]):
                e = eidx(ezoff, sz, i, j, k)
                rows += [e, e]
                cols += [nidx(i, j, k + 1), nidx(i, j, k)]
                vals += [1.0 / hz, -1.0 / hz]
    return sp.csr_matrix((vals, (rows, cols)), shape=(nedge, nnode))


def _sigma_to_edges(sigma_cell, shape):
    """Average a per-node conductivity field ``(prod(shape),)`` onto the edges (torch, differentiable).

    Each edge lies between two nodes along its axis; the edge conductivity is their arithmetic mean. Returns a
    complex torch tensor of length ``nedge`` in the ``(x, y, z)`` edge order of :func:`_edge_layout`.
    """
    torch = _torch()
    nx, ny, nz = shape
    sig = sigma_cell.reshape(nx, ny, nz)
    ex = 0.5 * (sig[1:, :, :] + sig[:-1, :, :])  # x-edges (nx-1, ny, nz)
    ey = 0.5 * (sig[:, 1:, :] + sig[:, :-1, :])  # y-edges (nx, ny-1, nz)
    ez = 0.5 * (sig[:, :, 1:] + sig[:, :, :-1])  # z-edges (nx, ny, nz-1)
    return torch.cat([ex.reshape(-1), ey.reshape(-1), ez.reshape(-1)]).to(torch.complex128)


def _edge_coords(shape, spacing):
    """Physical coordinates and axis id of every edge midpoint, in edge order. Returns ``(xyz (nedge,3), axis)``."""
    nx, ny, nz = shape
    hx, hy, hz = spacing
    xn = np.arange(nx) * hx
    yn = np.arange(ny) * hy
    zn = np.arange(nz) * hz
    out, axis = [], []
    # x-edges (nx-1, ny, nz): midpoint x
    xm = 0.5 * (xn[1:] + xn[:-1])
    gx, gy, gz = np.meshgrid(xm, yn, zn, indexing="ij")
    out.append(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1))
    axis.append(np.zeros(gx.size, int))
    # y-edges
    ym = 0.5 * (yn[1:] + yn[:-1])
    gx, gy, gz = np.meshgrid(xn, ym, zn, indexing="ij")
    out.append(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1))
    axis.append(np.ones(gx.size, int))
    # z-edges
    zm = 0.5 * (zn[1:] + zn[:-1])
    gx, gy, gz = np.meshgrid(xn, yn, zm, indexing="ij")
    out.append(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1))
    axis.append(2 * np.ones(gx.size, int))
    return np.concatenate(out, 0), np.concatenate(axis, 0)


def _spacing3(spacing):
    s = np.atleast_1d(np.asarray(spacing, float))
    if s.size == 1:
        return float(s[0]), float(s[0]), float(s[0])
    return float(s[0]), float(s[1]), float(s[2])


def assemble_curl_curl_3d(log_sigma, shape, *, omega, spacing=1.0, mu=MU0, sigma_ref=1.0, gauge=1.0):
    """Assemble ``curl(mu^{-1} curl) + i omega sigma`` on a 3-D Yee edge grid as ``(rows, cols, vals, n)``.

    The unknowns are the tangential electric field on the cell edges (``n = nedge``, x/y/z blocks in the order of
    :func:`_edge_layout`). The stiffness ``C^T diag(mu^{-1}) C`` (with ``C`` the discrete edge->face curl and the
    face-area / edge-length metric folded into the ``1/h`` weights) depends only on geometry and ``mu`` and is
    real symmetric. The complex mass ``i omega sigma_edge`` is added on the edge diagonal, with ``sigma`` averaged
    to edges from ``sigma = sigma_ref exp(log_sigma)`` -- so ``vals`` is complex and differentiable in ``log_sigma``.

    A Coulomb-gauge grad-div stabilizer ``gauge * G diag(1/sigma_node) G^T`` (Smith 1996) is added to lift the
    curl null space; it vanishes on the physical divergence-free solution, so it changes conditioning, not the
    field. Pass ``gauge=0`` to disable.

    Args:
        log_sigma: per-node log-conductivity ``(prod(shape),)``; torch tensor (gradients flow) or array-like.
        shape: node grid ``(nx, ny, nz)``; ``z`` (last axis) is depth, ``z = 0`` the surface.
        omega: angular frequency ``2 pi f`` (rad/s).
        spacing: grid spacing (scalar or per-axis ``(hx, hy, hz)``, m).
        mu: magnetic permeability (default vacuum).
        sigma_ref: reference conductivity multiplying ``exp(log_sigma)``.
        gauge: grad-div stabilizer weight (0 disables).

    Returns:
        ``(rows, cols, vals, n)`` for :func:`mixle_pde.pde_solve.sparse_solve`; ``vals`` complex, differentiable.
    """
    import scipy.sparse as sp

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    hx, hy, hz = _spacing3(spacing)
    lsig = log_sigma if torch.is_tensor(log_sigma) else torch.as_tensor(np.asarray(log_sigma, float))
    sig_node = sigma_ref * torch.exp(lsig)

    # The whole equation is integrated over the cell volume, so both terms carry a volume factor and the overall
    # scaling of stiffness vs mass is consistent. Stiffness = C^T diag(mu^{-1} * face_area * dual_length) C, with
    # C the (1/h)-weighted edge->face curl; the face weight is mu^{-1} * cell_volume (area * dual edge length).
    C = _curl_matrix(shape, hx, hy, hz)
    _, _, _, _, (fx, fy, fz) = _face_layout(shape)
    vol = hx * hy * hz
    wx = np.full(fx[0] * fx[1] * fx[2], vol / mu)
    wy = np.full(fy[0] * fy[1] * fy[2], vol / mu)
    wz = np.full(fz[0] * fz[1] * fz[2], vol / mu)
    W = sp.diags(np.concatenate([wx, wy, wz]))
    K = (C.T @ W @ C).tocoo()  # real symmetric stiffness on edges
    n = K.shape[0]

    # complex edge mass i omega sigma * volume (mu folded in like the 2-D TE mass term keeps the operator scaled)
    edge_vol_t = torch.as_tensor(_edge_dual_volume(shape, (hx, hy, hz)), dtype=torch.complex128)
    sig_edge = _sigma_to_edges(sig_node, shape)
    mass = 1j * float(omega) * mu * sig_edge * edge_vol_t

    # Coulomb-gauge grad-div stabilizer G diag(1/sigma_node) G^T (Smith 1996) to fill the curl null space; it is
    # constant (uses detached sigma), well-conditioned, and vanishes on the divergence-free physical solution.
    rows_np = [K.row]
    cols_np = [K.col]
    vals_np = [K.data]
    if gauge:
        G = _grad_matrix(shape, hx, hy, hz)
        s_node_np = np.clip(sig_node.detach().cpu().numpy().real, 1e-12, None)
        Sinv = sp.diags(vol / s_node_np)  # sigma^{-1} weighted by node volume
        Ggd = (float(gauge) * (G @ Sinv @ G.T)).tocoo()
        rows_np.append(Ggd.row)
        cols_np.append(Ggd.col)
        vals_np.append(Ggd.data)

    rows = torch.cat([torch.as_tensor(np.concatenate(rows_np), dtype=torch.long), torch.arange(n, dtype=torch.long)])
    cols = torch.cat([torch.as_tensor(np.concatenate(cols_np), dtype=torch.long), torch.arange(n, dtype=torch.long)])
    vals = torch.cat([torch.as_tensor(np.concatenate(vals_np), dtype=torch.complex128), mass])
    return rows, cols, vals, n


def _edge_dual_volume(shape, spacing):
    """Dual-cell volume attached to each edge (the mass-lumping weight), in edge order."""
    nx, ny, nz = shape
    hx, hy, hz = spacing
    vx = np.full((nx - 1) * ny * nz, hx * hy * hz)
    vy = np.full(nx * (ny - 1) * nz, hx * hy * hz)
    vz = np.full(nx * ny * (nz - 1), hx * hy * hz)
    return np.concatenate([vx, vy, vz])


def _dirichlet_all_boundary(shape):
    """Boolean mask (per edge, edge order) of edges lying on the outer box boundary faces.

    These edges hold imposed / decayed field values; interior edges are solved. An edge is on the boundary if any
    of its transverse node indices is at the grid extreme.
    """
    nx, ny, nz = shape
    _, eyoff, ezoff, nedge, (sx, sy, sz) = _edge_layout(shape)
    mask = np.zeros(nedge, bool)

    def block(off, s, on):
        a, b, c = s
        idx = np.arange(a * b * c).reshape(a, b, c)
        mask[off + idx[on]] = True

    # x-edges (nx-1, ny, nz): boundary if j in {0,ny-1} or k in {0,nz-1} or i endpoints touch a face
    i, j, k = np.meshgrid(np.arange(sx[0]), np.arange(sx[1]), np.arange(sx[2]), indexing="ij")
    onx = (j == 0) | (j == ny - 1) | (k == 0) | (k == nz - 1)
    block(0, sx, onx)
    i, j, k = np.meshgrid(np.arange(sy[0]), np.arange(sy[1]), np.arange(sy[2]), indexing="ij")
    ony = (i == 0) | (i == nx - 1) | (k == 0) | (k == nz - 1)
    block(eyoff, sy, ony)
    i, j, k = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), np.arange(sz[2]), indexing="ij")
    onz = (i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)
    block(ezoff, sz, onz)
    return mask


def _apply_dirichlet(rows, cols, vals, n, boundary_mask, torch):
    """Replace boundary-edge rows with the identity so the source ``b`` sets their Dirichlet values.

    Drops every assembled entry whose row is a boundary edge, then adds a unit diagonal there. Interior rows are
    untouched, keeping their coupling to boundary neighbours (whose values come from ``b``).
    """
    bmask = torch.as_tensor(boundary_mask)
    keep = ~bmask[rows]
    rows_k = rows[keep]
    cols_k = cols[keep]
    vals_k = vals[keep]
    bnd = torch.as_tensor(np.nonzero(boundary_mask)[0], dtype=torch.long)
    rows_o = torch.cat([rows_k, bnd])
    cols_o = torch.cat([cols_k, bnd])
    vals_o = torch.cat([vals_k, torch.ones(len(bnd), dtype=torch.complex128)])
    return rows_o, cols_o, vals_o, n


def mt_3d(log_sigma, shape, freq, *, polarization="x", spacing=1.0, sigma_ref=1.0, mu=MU0, gauge=1.0):
    """3-D plane-wave magnetotelluric forward: surface apparent resistivity and phase.

    Drives a uniform horizontal plane wave into a 3-D conductivity block by imposing the tangential surface E on
    the top face (``z = 0``) and the analytic skin-depth decay on the outer boundary edges, then solving the
    complex curl-curl system :func:`assemble_curl_curl_3d`. For ``polarization='x'`` the primary field is ``E_x``;
    the impedance ``Z = E_x / H_y`` is read at depth from the vertical decay of ``E_x`` (``H_y = -(1/(i omega mu))
    dE_x/dz``), giving ``rho_a = |Z|^2 / (omega mu)`` and ``phase = arg(Z)``.

    Over a laterally uniform half-space this reproduces the analytic sounding: ``rho_a = 1/sigma``, phase ``45``.

    Args:
        log_sigma: per-node log-conductivity ``(prod(shape),)``; torch tensor or array-like.
        shape: node grid ``(nx, ny, nz)``; ``z`` (last axis) is depth, ``z = 0`` the surface.
        freq: frequency (Hz).
        polarization: primary horizontal E direction, ``'x'`` or ``'y'``.
        spacing: grid spacing (scalar or per-axis, m).
        sigma_ref: reference conductivity multiplying ``exp(log_sigma)``.
        mu: magnetic permeability (default vacuum).
        gauge: grad-div stabilizer weight.

    Returns:
        ``(rho_a, phase, E)`` -- apparent resistivity (scalar), phase (deg, scalar), and the full complex edge
        field, all torch, differentiable in ``log_sigma``. Scalar observables use the central column.
    """
    from mixle_pde.pde_solve import sparse_solve

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    nx, ny, nz = shape
    hx, hy, hz = _spacing3(spacing)
    hzc = hz
    omega = 2.0 * np.pi * float(freq)
    lsig = log_sigma if torch.is_tensor(log_sigma) else torch.as_tensor(np.asarray(log_sigma, float))
    sig_node = sigma_ref * torch.exp(lsig)

    rows, cols, vals, n = assemble_curl_curl_3d(
        lsig, shape, omega=omega, spacing=spacing, mu=mu, sigma_ref=sigma_ref, gauge=gauge
    )

    # reference (background) decay from the surface-layer conductivity of the central column: E ~ exp(i k z),
    # k = sqrt(i omega mu sigma), decaying downward. Impose it on the outer boundary and top face.
    sig_grid = sig_node.reshape(nx, ny, nz)
    sig_ref_col = sig_grid[nx // 2, ny // 2, 0].to(torch.complex128)
    k = torch.sqrt(1j * omega * mu * sig_ref_col)

    coords, axis = _edge_coords(shape, (hx, hy, hz))
    zc = torch.as_tensor(coords[:, 2], dtype=torch.complex128)
    decay = torch.exp(1j * k * zc)  # analytic vertical decay, evaluated at each edge midpoint

    prim_axis = 0 if polarization == "x" else 1
    boundary = _dirichlet_all_boundary(shape)

    b = torch.zeros(n, dtype=torch.complex128)
    # source: on boundary edges of the primary axis, set E = decay(z); other-axis boundary edges = 0.
    prim = torch.as_tensor((axis == prim_axis), dtype=torch.bool)
    bnd = torch.as_tensor(boundary, dtype=torch.bool)
    src_edges = bnd & prim
    b = torch.where(src_edges, decay, b)

    # also pin the entire top face (z = 0) of the primary axis to E = 1 (the incident field at the surface),
    # already covered by decay(z=0)=1 through the boundary mask on the top layer.
    rows2, cols2, vals2, n2 = _apply_dirichlet(rows, cols, vals, n, boundary, torch)
    E = sparse_solve(vals2, rows2, cols2, n2, b)

    # read the vertical decay of the primary field along the central column to get the local wavenumber, then Z.
    _, eyoff, ezoff, _, (sx, sy, sz) = _edge_layout(shape)
    if prim_axis == 0:
        s = sx
        off = 0
        ic, jc = min(nx // 2, sx[0] - 1), ny // 2
        col = E[off + ((ic * s[1] + jc) * s[2]) : off + ((ic * s[1] + jc) * s[2]) + s[2]]
    else:
        s = sy
        off = eyoff
        ic, jc = nx // 2, min(ny // 2, sy[1] - 1)
        col = E[off + ((ic * s[1] + jc) * s[2]) : off + ((ic * s[1] + jc) * s[2]) + s[2]]

    # Local discrete decay r = E(z+h)/E(z), read in an interior window away from the imposed surface / basal
    # boundary (the field has settled onto the pure geometric decay there). The continuum wavenumber follows the
    # discrete dispersion the curl-curl operator enforces, k^2 = -(r - 2 + 1/r)/h^2 (the second-difference of a
    # geometric column is -k^2 h^2); then Z = i omega mu / k, so rho_a = |Z|^2 / (omega mu), phase = arg(Z).
    depth = col.shape[0]
    lo = max(1, depth // 3)
    hi = max(lo + 1, (2 * depth) // 3)
    r = (col[lo + 1 : hi + 1] / col[lo:hi]).mean()
    k2 = -(r - 2.0 + 1.0 / r) / (hzc * hzc)
    kk = torch.sqrt(k2)
    Z = 1j * omega * mu / kk
    rho_a = (Z.abs() ** 2) / (omega * mu)
    phase = torch.rad2deg(torch.angle(Z))
    return rho_a, phase, E


def csem_3d(log_sigma, shape, freq, *, source_edges, source_amp=1.0, spacing=1.0, sigma_ref=1.0, mu=MU0, gauge=1.0):
    """3-D grounded electric-dipole CSEM forward (bonus): inject a current source, return the complex field.

    A galvanic (grounded) dipole is a current density on a set of interior edges: the right-hand side is
    ``-i omega mu J`` on those edges (a source term of the curl-curl system), with the field decaying to the
    outer boundary (pinned to zero -- valid when the box is many skin depths across). Verified on a uniform whole
    space; heterogeneous CSEM impedance recovery is the extension.

    Args:
        log_sigma: per-node log-conductivity ``(prod(shape),)``; torch tensor or array-like.
        shape: node grid ``(nx, ny, nz)``.
        freq: frequency (Hz).
        source_edges: flat edge indices (edge order) carrying the injected current.
        source_amp: current amplitude on the source edges (A/m^2, per edge).
        spacing, sigma_ref, mu, gauge: as :func:`assemble_curl_curl_3d`.

    Returns:
        ``E`` -- the full complex edge field, torch, differentiable in ``log_sigma``.
    """
    from mixle_pde.pde_solve import sparse_solve

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    omega = 2.0 * np.pi * float(freq)
    lsig = log_sigma if torch.is_tensor(log_sigma) else torch.as_tensor(np.asarray(log_sigma, float))

    rows, cols, vals, n = assemble_curl_curl_3d(
        lsig, shape, omega=omega, spacing=spacing, mu=mu, sigma_ref=sigma_ref, gauge=gauge
    )
    boundary = _dirichlet_all_boundary(shape)
    rows2, cols2, vals2, n2 = _apply_dirichlet(rows, cols, vals, n, boundary, torch)

    b = torch.zeros(n, dtype=torch.complex128)
    src = torch.as_tensor(np.asarray(source_edges, int), dtype=torch.long)
    hx, hy, hz = _spacing3(spacing)
    vol = hx * hy * hz
    b[src] = -1j * omega * mu * complex(source_amp) * vol  # -i omega mu J on the source edges
    # do not let a source edge be overwritten by a boundary identity row
    b = torch.where(torch.as_tensor(boundary), torch.zeros_like(b), b)

    E = sparse_solve(vals2, rows2, cols2, n2, b)
    return E
