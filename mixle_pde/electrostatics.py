"""Poisson-Boltzmann electrostatics for biomolecular / continuum-solvent problems.

The implicit-solvent electrostatic potential ``phi`` around fixed charges obeys the Poisson-Boltzmann
equation (PBE)

    -div(eps(r) grad phi) + kappabar^2(r) sinh(phi) = sum_k q_k delta(r - r_k),

with a spatially varying permittivity ``eps(r)`` (low inside a solute cavity, high in water) and a
modified Debye-Huckel screening factor ``kappabar^2(r) = eps kappa^2`` that is nonzero only where mobile
salt ions can go. Two regimes matter:

* The **linearized PBE** (Debye-Huckel), valid at low potential, replaces ``sinh(phi)`` by ``phi``:

      -div(eps grad phi) + kappabar^2 phi = sum_k q_k delta(r - r_k).

  This is a linear elliptic system. It is assembled by reusing :func:`mixle_pde.pde_solve.divergence_form`
  for the variable-coefficient ``-div(eps grad)`` operator, adding the screening term ``kappabar^2`` on the
  diagonal of interior nodes, injecting each point charge as a finite-volume delta on the right-hand side
  (exactly as :func:`mixle_pde.geophysics.dc_resistivity` injects a current electrode), and solving with the
  adjoint-differentiable :func:`mixle_pde.pde_solve.sparse_solve`. In a homogeneous electrolyte it recovers
  the screened-Coulomb (Yukawa) potential ``phi(r) = q e^{-kappabar r} / (4 pi eps r)``.

* The **full nonlinear PBE** keeps ``sinh(phi)``, needed at high potential (near a highly charged
  biomolecule). It is a nonlinear elliptic problem, solved with the Newton + implicit-adjoint steady solver
  :func:`mixle_pde.nonlinear.nonlinear_solve`, with reaction ``g(phi) = kappabar^2 sinh(phi)`` and Jacobian
  diagonal ``dg/dphi = kappabar^2 cosh(phi)``. As the source shrinks it reduces to the linearized PBE.

The observable of interest for solvation and binding is the **reaction-field energy**
``DeltaG = 0.5 sum_k q_k (phi(r_k) - phi_vacuum(r_k))``, the electrostatic free energy of moving the fixed
charges from a reference (homogeneous / vacuum) medium into the solvated dielectric. For a single ion in a
sphere it is the Born solvation energy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "linearized_pbe",
    "nonlinear_pbe",
    "reaction_field_energy",
    "yukawa_potential",
    "born_solvation_energy",
]


def _torch():
    import torch

    return torch


def _as_field(x, n, torch):
    """Broadcast a scalar or array to a length-``n`` torch float64 field."""
    if np.isscalar(x) or (hasattr(x, "ndim") and getattr(x, "ndim", 1) == 0):
        return torch.full((int(n),), float(x), dtype=torch.float64)
    t = torch.as_tensor(np.asarray(x, dtype=float).reshape(-1), dtype=torch.float64)
    if t.shape[0] != int(n):
        raise ValueError(f"field length {t.shape[0]} != number of nodes {int(n)}.")
    return t


def _charge_rhs(charges, shape, spacing, n, torch, dtype):
    """Finite-volume point-charge source: ``b[k] = q_k / cell`` at each charge node (a discrete delta),
    so the assembled system approximates ``... = sum_k q_k delta(r - r_k)`` (see ``dc_resistivity``)."""
    cell = float(np.prod(np.atleast_1d(spacing)))
    b = torch.zeros(int(n), dtype=dtype)
    for node, q in charges:
        b[int(node)] = b[int(node)] + float(q) / cell
    return b


def _normalize_charges(charges):
    """Accept ``[(node, q), ...]`` or a dict ``{node: q}``; return a list of ``(int node, float q)``."""
    if isinstance(charges, dict):
        return [(int(k), float(v)) for k, v in charges.items()]
    return [(int(node), float(q)) for node, q in charges]


def linearized_pbe(eps, kappabar2, charges, shape, *, spacing=1.0):
    """Solve the linearized (Debye-Huckel) PBE ``-div(eps grad phi) + kappabar^2 phi = sum_k q_k delta``.

    Assembles the variable-coefficient operator with :func:`mixle_pde.pde_solve.divergence_form`, adds the
    screening term ``kappabar^2`` on interior diagonals, injects the point charges as finite-volume deltas
    on the right-hand side, and solves with the adjoint-differentiable
    :func:`mixle_pde.pde_solve.sparse_solve`. Dirichlet-zero on the box boundary (the far-field ground).

    Args:
        eps: permittivity field ``eps(r)`` -- scalar or per-node array (length ``prod(shape)``). Gradients
            flow through a torch tensor.
        kappabar2: screening field ``kappabar^2(r) = eps kappa^2`` -- scalar or per-node array; zero where
            there is no mobile salt (e.g. inside a solute cavity).
        charges: point charges as ``[(node, q), ...]`` (flat node indices, interior only) or ``{node: q}``.
        shape: grid shape ``(nx, ny[, nz])``.
        spacing: grid spacing (scalar or per-axis).

    Returns:
        torch tensor ``phi`` (length ``prod(shape)``), differentiable in ``eps`` / ``kappabar2``.
    """
    from mixle.ppl._grid import _grid_faces

    from mixle_pde.pde_solve import divergence_form, sparse_solve

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    eps_t = _as_field(eps, n, torch)
    kbar2_t = _as_field(kappabar2, n, torch)

    rows, cols, vals, nn = divergence_form(eps_t, shape, spacing=spacing)
    g = _grid_faces(shape, spacing)
    interior = torch.as_tensor(g["interior"], dtype=torch.long)
    # + kappabar^2 phi on interior diagonals (boundary rows stay the Dirichlet identity)
    rows = torch.cat([rows, interior])
    cols = torch.cat([cols, interior])
    vals = torch.cat([vals, kbar2_t[interior].to(vals.dtype)])

    b = _charge_rhs(_normalize_charges(charges), shape, spacing, n, torch, vals.dtype)
    return sparse_solve(vals, rows, cols, nn, b)


def nonlinear_pbe(eps, kappabar2, charges, shape, *, spacing=1.0, u0=None, max_its=50, tol=1e-10, damping=1.0):
    """Solve the full nonlinear PBE ``-div(eps grad phi) + kappabar^2 sinh(phi) = sum_k q_k delta``.

    Reuses :func:`mixle_pde.nonlinear.nonlinear_solve` (Newton + implicit-adjoint backward) with the
    reaction ``g(phi) = kappabar^2 sinh(phi)`` and diagonal Jacobian ``dg/dphi = kappabar^2 cosh(phi)``.
    The diffusion and point-charge assembly match :func:`linearized_pbe`, so as the charges shrink the
    solution reduces to the linearized PBE.

    Args:
        eps, kappabar2, charges, shape, spacing: as in :func:`linearized_pbe`.
        u0: optional initial guess for ``phi`` (default zeros).
        max_its, tol, damping: Newton controls forwarded to :func:`nonlinear_solve`.

    Returns:
        torch tensor ``phi`` (length ``prod(shape)``).
    """
    from mixle_pde.nonlinear import nonlinear_solve, reaction_diffusion_residual

    torch = _torch()
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    eps_t = _as_field(eps, n, torch)
    kbar2_t = _as_field(kappabar2, n, torch)

    # f is the interior source and doubles as the boundary Dirichlet data (zero here). reaction_diffusion_
    # residual builds -div(kappa grad u) + g(u) = f with kappa = eps.
    f = _charge_rhs(_normalize_charges(charges), shape, spacing, n, torch, torch.float64)

    def g(u, theta):
        return kbar2_t * torch.sinh(u)

    def dg_du(u, theta):
        return kbar2_t * torch.cosh(u)

    residual_fn, jac_fn = reaction_diffusion_residual(shape, f, g, dg_du, kappa=eps_t, spacing=spacing)
    if u0 is None:
        u0 = torch.zeros(n, dtype=torch.float64)
    theta = torch.zeros(1, dtype=torch.float64)  # no external parameter; the physics is fixed
    return nonlinear_solve(residual_fn, jac_fn, u0, theta, max_its=max_its, tol=tol, damping=damping)


def reaction_field_energy(phi, phi_reference, charges):
    """Reaction-field electrostatic energy ``DeltaG = 0.5 sum_k q_k (phi(r_k) - phi_reference(r_k))``.

    ``phi`` is the solvated potential and ``phi_reference`` the potential of the same charges in the
    reference medium (a homogeneous / vacuum solve on the same grid), so their difference is the reaction
    field and its self-energy divergence cancels. For a single ion in a sphere this is the Born solvation
    energy.

    Args:
        phi: solvated potential (torch tensor or array), length ``prod(shape)``.
        phi_reference: reference-medium potential, same length.
        charges: ``[(node, q), ...]`` or ``{node: q}``.

    Returns:
        scalar ``DeltaG`` (torch tensor if ``phi`` is a torch tensor, else float).
    """
    torch = _torch()
    charges = _normalize_charges(charges)
    is_t = torch.is_tensor(phi)
    dphi = phi - phi_reference
    total = 0.0
    for node, q in charges:
        total = total + 0.5 * q * dphi[int(node)]
    if is_t and not torch.is_tensor(total):
        total = torch.as_tensor(total, dtype=torch.float64)
    return total


def yukawa_potential(r, q, eps, kappabar):
    """Analytical screened-Coulomb (Yukawa) potential ``q e^{-kappabar r} / (4 pi eps r)`` of a point charge
    in a homogeneous electrolyte -- the closed form the linearized PBE reproduces away from the source."""
    r = np.asarray(r, float)
    return q * np.exp(-kappabar * r) / (4.0 * np.pi * eps * r)


def born_solvation_energy(q, R, eps_in, eps_out, *, eps0=1.0):
    """Born ion solvation energy ``-(q^2 / (8 pi eps0 R))(1/eps_in - 1/eps_out)`` -- the reaction-field free
    energy of a charge ``q`` at the centre of a cavity of radius ``R`` (interior permittivity ``eps_in``)
    embedded in solvent ``eps_out``. ``eps0`` is the vacuum permittivity in the working unit system."""
    return -(q**2 / (8.0 * np.pi * eps0 * R)) * (1.0 / eps_in - 1.0 / eps_out)
