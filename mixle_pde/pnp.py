"""Poisson-Nernst-Planck coupled electro-diffusion for ion channels and nanopores (1-D, equilibrium).

The PNP system couples, per ionic species ``i``, the steady Nernst-Planck flux and its conservation

    J_i = -D_i (grad c_i + z_i c_i grad phi),      div J_i = 0

to Poisson's equation for the electrostatic potential driven by the mobile plus fixed charge

    -div(eps grad phi) = sum_i z_i c_i + rho_fixed.

Units are nondimensional with the thermal voltage absorbed into ``phi`` (``e = kT = 1``), so ``phi`` is in
units of ``kT/e`` and concentrations in units of a reference bulk. This is the ion-channel / nanopore
electrostatics setting: a charged pore wall (``rho_fixed`` or a wall potential) screened by a bathing
electrolyte.

At zero net flux / zero current every species is in thermodynamic equilibrium, so ``div J_i = 0`` forces the
Boltzmann distribution ``c_i(x) = c_i^bulk exp(-z_i phi(x))`` exactly. Substituting it into Poisson gives the
Poisson-Boltzmann equation

    -div(eps grad phi) = sum_i z_i c_i^bulk exp(-z_i phi) + rho_fixed,

a single nonlinear elliptic PDE for ``phi``. :func:`pnp_equilibrium` solves it with the differentiable Newton
solver :func:`mixle_pde.nonlinear.nonlinear_solve` (an implicit-function-theorem backward, one extra solve),
then recovers each ``c_i`` from the converged ``phi`` by Boltzmann. The forward is differentiable in
``rho_fixed`` (the fixed pore charge, which enters the source) and accepts ``D_i`` in the signature; at
equilibrium the concentrations are diffusivity-independent, so the ``D_i`` gradient is exactly zero, which is
the correct physics rather than a missing path. The driven finite-current I-V (a Gummel or fully-coupled
Newton over the Nernst-Planck drift-diffusion balance) is the natural extension.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["pnp_equilibrium", "debye_length"]


def _torch():
    import torch

    return torch


def debye_length(z: Sequence[float], c_bulk: Sequence[float], *, eps: float = 1.0) -> float:
    """The linearized (Debye) screening length ``lambda_D = sqrt(eps / sum_i z_i^2 c_i^bulk)``.

    In the nondimensional units used here (``e = kT = 1``) the ionic strength ``sum_i z_i^2 c_i^bulk`` sets the
    inverse-square screening length; near a weakly charged wall the potential decays as ``exp(-x / lambda_D)``.
    ``eps`` is the (constant) permittivity.
    """
    z = np.asarray(z, dtype=float)
    c = np.asarray(c_bulk, dtype=float)
    ionic_strength = float(np.sum(z**2 * c))
    return float(np.sqrt(eps / ionic_strength))


def pnp_equilibrium(
    z: Sequence[float],
    c_bulk: Sequence[float],
    shape,
    *,
    spacing: float = 1.0,
    eps=1.0,
    rho_fixed=None,
    diffusivity: Sequence[float] | None = None,
    phi_left: float = 0.0,
    phi_right: float = 0.0,
    max_its: int = 100,
    phi0=None,
):
    """Zero-current Poisson-Nernst-Planck forward: potential ``phi`` and per-species concentrations ``c_i``.

    Solves the Poisson-Boltzmann equation ``-div(eps grad phi) = sum_i z_i c_i^bulk exp(-z_i phi) + rho_fixed``
    with Dirichlet potential on the two ends (a wall/bath potential) via the differentiable Newton solver, then
    sets ``c_i = c_i^bulk exp(-z_i phi)`` (the exact zero-flux equilibrium of each Nernst-Planck species). The
    map is differentiable in ``rho_fixed``; ``diffusivity`` is accepted but does not affect the equilibrium
    solution (a physical fact at zero current), so its gradient is zero.

    Args:
        z: ionic valences ``(S,)`` (e.g. ``[+1, -1]`` for a 1:1 salt).
        c_bulk: bulk (reservoir) concentrations ``(S,)`` in reference units, the Boltzmann prefactors.
        shape: 1-D grid shape ``(m,)`` (a length-1 tuple or an int).
        spacing: node spacing ``h``.
        eps: permittivity; a scalar or a per-node torch tensor / array (variable-coefficient Poisson).
        rho_fixed: fixed-charge density on interior nodes ``(m,)`` (the pore/wall charge); a torch tensor
            (gradients flow) or array-like. ``None`` means no fixed charge.
        diffusivity: per-species ``D_i`` ``(S,)``; accepted for the PNP signature. Zero-current equilibrium is
            independent of ``D_i``, so this only documents the physics and carries a zero gradient.
        phi_left, phi_right: Dirichlet potential at the two ends (wall and bath potentials).
        max_its: Newton iterations.
        phi0: optional initial guess ``(m,)`` for ``phi`` (defaults to a linear ramp between the end values).

    Returns:
        ``(phi, c)`` where ``phi`` is ``(m,)`` and ``c`` is ``(S, m)`` (one row per species), torch tensors.
        ``phi`` (and hence ``c``) is differentiable in ``rho_fixed``.
    """
    from mixle_pde.nonlinear import nonlinear_solve, reaction_diffusion_residual

    torch = _torch()
    shape = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)
    if len(shape) != 1:
        raise ValueError(f"pnp_equilibrium is 1-D; got shape {shape}.")
    m = shape[0]
    zt = torch.as_tensor(np.asarray(z, dtype=float))
    cb = torch.as_tensor(np.asarray(c_bulk, dtype=float))
    if zt.numel() != cb.numel():
        raise ValueError("z and c_bulk must have the same length (one per species).")
    kappa = eps if torch.is_tensor(eps) else torch.full((m,), float(eps), dtype=torch.float64)

    # The mobile-charge reaction on the LHS: g(phi; rho) = -(sum_i z_i c_i^bulk exp(-z_i phi)) - rho_fixed, so
    # the assembled residual -div(eps grad phi) + g = 0 reproduces Poisson-Boltzmann with the fixed charge.
    def _mobile(u):
        # sum_i z_i c_i^bulk exp(-z_i phi), broadcast over the (S, m) exponentials
        e = cb[:, None] * torch.exp(-zt[:, None] * u[None, :])  # (S, m)
        return (zt[:, None] * e).sum(dim=0)  # (m,)

    def g(u, rho):
        r = -_mobile(u)
        if rho is not None:
            r = r - rho
        return r

    def dg(u, rho):
        # d/dphi of -sum_i z_i c_i exp(-z_i phi) = sum_i z_i^2 c_i exp(-z_i phi)
        e = cb[:, None] * torch.exp(-zt[:, None] * u[None, :])
        return (zt[:, None] ** 2 * e).sum(dim=0)

    # Dirichlet potential enters through the source on the boundary nodes (identity rows in divergence_form).
    f_full = np.zeros(m, dtype=float)
    f_full[0] = float(phi_left)
    f_full[-1] = float(phi_right)

    residual_fn, jac_fn = reaction_diffusion_residual(shape, f_full, g, dg, kappa=kappa, spacing=spacing)

    if phi0 is None:
        u0 = torch.as_tensor(np.linspace(phi_left, phi_right, m))
    else:
        u0 = torch.as_tensor(np.asarray(phi0, dtype=float).reshape(-1))

    theta = torch.zeros(m, dtype=torch.float64) if rho_fixed is None else torch.as_tensor(rho_fixed)
    phi = nonlinear_solve(residual_fn, jac_fn, u0, theta, max_its=max_its)

    # Boltzmann equilibrium concentrations from the converged potential.
    c = cb[:, None] * torch.exp(-zt[:, None] * phi[None, :])  # (S, m)

    # diffusivity is part of the PNP contract but zero-current equilibrium is independent of it; touch it in a
    # differentiable no-op so a caller passing a requires_grad D_i gets a defined (zero) gradient path.
    if diffusivity is not None:
        d = torch.as_tensor(np.asarray(diffusivity, dtype=float)) if not torch.is_tensor(diffusivity) else diffusivity
        phi = phi + 0.0 * d.sum()
        c = c + 0.0 * d.sum()

    return phi, c
