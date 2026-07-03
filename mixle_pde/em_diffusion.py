"""Diffusive / quasi-static electromagnetic forwards for the conductive earth (CSEM / MT / AEM, eddy-current NDE).

This is the *induction* regime, not the lossless-wave regime that :mod:`mixle_pde.maxwell` covers. When
displacement currents are negligible against conduction currents (``omega eps << sigma``), the source-free
curl-curl equation

    curl(mu^{-1} curl E) + i omega sigma E = -i omega J_s

is a complex diffusion equation: the ``i omega sigma`` mass term is the eddy-current loss, and the field decays
into a conductor over the skin depth ``delta = sqrt(2 / (omega mu sigma))`` instead of propagating. Recovering
``sigma(x)`` (as log-conductivity) from surface / airborne / borehole measurements of that decaying field is the
inverse problem of controlled-source EM, magnetotellurics, airborne EM, and eddy-current flaw detection.

Two differentiable forwards, in increasing dimensionality:

* :func:`layered_mt_impedance` -- the 1-D layered magnetotelluric / AEM forward by the Wait impedance recursion.
  Closed form, complex, differentiable in the layer conductivities; the natural forward for a 1-D sounding.
* :func:`mt_2d_te` / :func:`assemble_mt_te` -- the 2-D magnetotelluric TE-mode finite-difference forward: the
  complex diffusion equation for the along-strike electric field, assembled as ``(rows, cols, vals, n)`` for the
  adjoint-capable :func:`mixle_pde.pde_solve.sparse_solve` with the ``i omega mu sigma`` mass term.

Everything runs through the ``ops`` complex arithmetic and the package's ``divergence_form`` / ``sparse_solve``,
so the whole map is differentiable end to end and never imports torch in the stepper; the 3-D edge-element
curl-curl forward is the natural extension.
"""

from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi  # vacuum permeability (H/m); the earth is non-magnetic to first order


def _torch():
    import torch

    return torch


def layered_mt_impedance(sigma, thicknesses, freqs, *, mu=MU0):
    """1-D layered magnetotelluric / AEM forward by the Wait impedance recursion.

    A stack of ``L`` homogeneous layers over a terminating half-space, illuminated by a vertically-incident plane
    wave. In layer ``j`` the vertical wavenumber is ``k_j = sqrt(i omega mu sigma_j)`` (root with positive real and
    imaginary parts, so the field decays downward) and the intrinsic impedance is ``Zi_j = i omega mu / k_j``. The
    surface impedance is built from the bottom up by the recursion (Wait 1954)

        Z_{j} = Zi_j (Z_{j+1} + Zi_j tanh(k_j d_j)) / (Zi_j + Z_{j+1} tanh(k_j d_j)),

    started at the basal half-space ``Z_L = Zi_L``. From the surface impedance ``Z`` the observables are the
    apparent resistivity ``rho_a = |Z|^2 / (omega mu)`` and the phase ``arg(Z)``.

    For a uniform half-space this is exact: ``rho_a == 1 / sigma`` at every frequency and ``phase == 45 deg``.

    Args:
        sigma: per-layer conductivity ``(L,)`` (S/m), top to bottom; the last entry is the basal half-space.
            A torch tensor (gradients flow) or array-like.
        thicknesses: layer thicknesses ``(L-1,)`` (m) for the layers *above* the half-space; the half-space has
            no thickness. A length-``L`` sequence is accepted and its last entry ignored. Empty for a half-space.
        freqs: frequencies ``(F,)`` in Hz.
        mu: magnetic permeability (default vacuum ``mu0``).

    Returns:
        ``(rho_a, phase, Z)`` -- apparent resistivity ``(F,)`` (ohm-m), phase ``(F,)`` (degrees), and the complex
        surface impedance ``(F,)``. All torch tensors, differentiable in ``sigma``.
    """
    torch = _torch()
    sig = sigma if torch.is_tensor(sigma) else torch.as_tensor(np.asarray(sigma, float))
    sig = sig.to(torch.complex128)
    thick = torch.as_tensor(np.asarray(thicknesses, float)[: sig.numel() - 1], dtype=torch.float64)
    f = np.asarray(freqs, float).reshape(-1)
    two_pi = 2.0 * np.pi
    Zs = []
    for fk in f:
        omega = two_pi * float(fk)
        wmu = omega * mu
        # basal half-space impedance
        k = torch.sqrt(1j * wmu * sig[-1])
        Z = 1j * wmu / k
        # recurse upward through the overlying layers (bottom to top)
        for j in range(sig.numel() - 2, -1, -1):
            kj = torch.sqrt(1j * wmu * sig[j])
            Zi = 1j * wmu / kj
            th = torch.tanh(kj * thick[j])
            Z = Zi * (Z + Zi * th) / (Zi + Z * th)
        Zs.append(Z)
    Z = torch.stack(Zs)
    omega_t = torch.as_tensor(two_pi * f, dtype=torch.float64)
    rho_a = (Z.abs() ** 2) / (omega_t * mu)
    phase = torch.rad2deg(torch.angle(Z))
    return rho_a, phase, Z


def assemble_mt_te(sigma, shape, *, omega, spacing=1.0, mu=MU0):
    """Assemble the 2-D magnetotelluric TE-mode operator ``-lap E + i omega mu sigma E`` as ``(rows, cols, vals, n)``.

    In the TE (E-parallel) mode the along-strike electric field ``E`` (out of the ``x-z`` plane) satisfies the
    scalar complex diffusion equation ``-lap E + i omega mu sigma(x, z) E = 0`` -- the quasi-static reduction of
    the curl-curl equation for a 2-D conductivity structure. The constant-coefficient Laplacian comes from
    :func:`mixle_pde.pde_solve.divergence_form` (Dirichlet identity rows on the box, so ``b`` sets the boundary
    field); the ``i omega mu sigma`` eddy-current mass term is added on the interior diagonal, so ``vals`` is
    complex and differentiable in ``sigma``.

    Args:
        sigma: per-node conductivity ``(prod(shape),)`` (S/m); a torch tensor (gradients flow) or array-like.
        shape: grid shape ``(nx, nz)`` with ``z`` (depth) the last axis.
        omega: angular frequency ``2 pi f`` (rad/s).
        spacing: grid spacing (scalar or per-axis, m).
        mu: magnetic permeability (default vacuum).

    Returns:
        ``(rows, cols, vals, n)`` for :func:`mixle_pde.pde_solve.sparse_solve`; ``vals`` complex, differentiable.
    """
    from mixle.ppl._grid import _grid_faces

    from mixle_pde.pde_solve import divergence_form

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    sig = sigma if torch.is_tensor(sigma) else torch.as_tensor(np.asarray(sigma, float))
    sig = sig.to(torch.complex128)
    rows, cols, vals, n = divergence_form(torch.ones(sig.numel(), dtype=torch.complex128), shape, spacing=spacing)
    g = _grid_faces(shape, spacing)
    interior = torch.as_tensor(g["interior"], dtype=torch.long)
    mass = 1j * float(omega) * mu * sig[interior]  # eddy-current loss on interior nodes
    rows = torch.cat([rows, interior])
    cols = torch.cat([cols, interior])
    vals = torch.cat([vals, mass])
    return rows, cols, vals, n


def _discrete_root(k2h2):
    """The decaying root ``r`` (``|r| < 1``) of the 1-D discrete dispersion ``r - 2 + 1/r = k^2 h^2``.

    For a laterally uniform column the finite-difference solution of ``-lap E + i omega mu sigma E = 0`` is
    exactly geometric, ``E_j = r^j``, so pinning the far boundary to ``r^j`` gives the exact 1-D decay with no
    reflection. ``k2h2 = i omega mu sigma h^2`` is the (complex) squared wavenumber times the spacing squared.
    """
    torch = _torch()
    two_plus = 2.0 + k2h2
    disc = torch.sqrt(two_plus * two_plus - 4.0)
    r1 = 0.5 * (two_plus + disc)
    r2 = 0.5 * (two_plus - disc)
    return torch.where(r1.abs() < 1.0, r1, r2)


def mt_2d_te(log_sigma, shape, freq, *, spacing=1.0, sigma_ref=1.0, mu=MU0):
    """2-D magnetotelluric TE-mode finite-difference forward: surface apparent resistivity and phase.

    Drives a plane wave into a 2-D conductivity section by holding ``E = 1`` on the top (air-earth) boundary and
    solving the complex diffusion equation :func:`assemble_mt_te`, with the side and basal boundaries pinned to
    the *discrete* geometric decay ``E_j = r^j`` of the local column (``r`` from :func:`_discrete_root`) so a
    laterally uniform section is reproduced with no reflection artefact. The MT impedance at each surface site is
    read from the field's local (discrete) squared wavenumber ``k^2 = (E_1/E_0 - 2 + E_0/E_1) / h^2`` -- the same
    dispersion relation the operator enforces -- giving ``Z = i omega mu / sqrt(k^2)``, then
    ``rho_a = |Z|^2 / (omega mu)`` and ``phase = arg(Z)``. Conductivity is ``sigma = sigma_ref exp(log_sigma)`` so
    ``log_sigma`` is the natural (log-) inversion parameter and stays positive.

    Over a laterally uniform half-space this reproduces the exact sounding: ``rho_a == 1 / sigma`` and ``phase == 45``.

    Args:
        log_sigma: per-node log-conductivity ``(prod(shape),)``; a torch tensor (gradients flow) or array-like.
        shape: grid shape ``(nx, nz)`` with ``z`` (depth) the last axis; row ``z = 0`` is the surface.
        freq: frequency (Hz).
        spacing: grid spacing (scalar or per-axis, m).
        sigma_ref: reference conductivity multiplying ``exp(log_sigma)``.
        mu: magnetic permeability (default vacuum).

    Returns:
        ``(rho_a, phase)`` -- apparent resistivity ``(nx,)`` and phase ``(nx,)`` (degrees) at the ``nx`` surface
        sites, torch tensors differentiable in ``log_sigma``.
    """
    from mixle_pde.pde_solve import sparse_solve

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    nx, nz = shape
    hz = float(np.atleast_1d(spacing)[-1])
    omega = 2.0 * np.pi * float(freq)
    lsig = log_sigma if torch.is_tensor(log_sigma) else torch.as_tensor(np.asarray(log_sigma, float))
    sig = sigma_ref * torch.exp(lsig)
    rows, cols, vals, n = assemble_mt_te(sig, shape, omega=omega, spacing=spacing, mu=mu)

    # per-column discrete decay r^j from each column's surface conductivity (exact for a uniform column)
    sig_surf = sig.reshape(nx, nz)[:, 0].to(torch.complex128)
    r = _discrete_root(1j * omega * mu * sig_surf * hz * hz)  # (nx,)
    powers = r[:, None] ** torch.arange(nz, dtype=torch.float64).to(torch.complex128)[None, :]  # (nx, nz)

    idx = np.arange(nx * nz).reshape(nx, nz)
    b = torch.zeros(nx * nz, dtype=torch.complex128)
    for i in range(nx):
        b[idx[i, 0]] = 1.0  # surface source E = 1
        b[idx[i, nz - 1]] = powers[i, nz - 1]  # basal boundary: discrete decay
    for j in range(nz):
        b[idx[0, j]] = powers[0, j]  # side boundaries: discrete decay of the edge columns
        b[idx[nx - 1, j]] = powers[nx - 1, j]

    E = sparse_solve(vals, rows, cols, n, b).reshape(nx, nz)
    # local squared wavenumber from the discrete dispersion the operator enforces; Z = i omega mu / k
    rr = E[:, 1] / E[:, 0]
    k2 = (rr - 2.0 + 1.0 / rr) / (hz * hz)
    k = torch.sqrt(k2)
    Z = 1j * omega * mu / k
    rho_a = (Z.abs() ** 2) / (omega * mu)
    phase = torch.rad2deg(torch.angle(Z))
    return rho_a, phase
