"""Spectral induced polarization (SIP) forward for disseminated-sulphide mineral exploration.

Induced polarization measures the *frequency-dependent complex* conductivity of the ground. Disseminated
sulphides (and clays) polarize under an applied current, so their conductivity acquires a phase lag that the
Cole-Cole relaxation model captures:

    sigma(omega) = sigma_inf * (1 - m / (1 + (i omega tau)^c)),

with ``sigma_inf`` the high-frequency conductivity, ``m`` the chargeability (0..1), ``tau`` the time constant
(s), and ``c`` the frequency exponent (0..1). The DC (omega -> 0) limit is ``sigma_inf (1 - m)`` and the
high-frequency limit is ``sigma_inf``; between them the imaginary part peaks near ``omega tau ~ 1``.

The SIP forward is just the DC-resistivity Poisson operator ``-div(sigma grad phi) = I`` run with this
COMPLEX, frequency-dependent conductivity. ``divergence_form`` assembles the operator for a complex
coefficient field and ``sparse_solve`` has a complex-correct adjoint backward, so the whole spectrum is
differentiable in the Cole-Cole parameters. For a homogeneous model the transfer resistance is
``R(omega) = K_geom / sigma(omega)`` with a purely geometric factor ``K_geom``, so the recovered apparent
complex conductivity reproduces the input Cole-Cole spectrum exactly.

Nothing here patches mixle; it reuses ``mixle_pde.pde_solve`` and mirrors ``geophysics.dc_resistivity``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cole_cole_conductivity",
    "sip_forward",
    "apparent_conductivity",
    "geometric_factor",
]


def _torch():
    import torch

    return torch


def cole_cole_conductivity(omega, sigma_inf, m, tau, c):
    r"""Cole-Cole complex conductivity ``sigma_inf (1 - m / (1 + (i omega tau)^c))``.

    Args:
        omega: angular frequency (scalar or torch tensor of frequencies), rad/s.
        sigma_inf: high-frequency conductivity (positive).
        m: chargeability in [0, 1).
        tau: relaxation time constant (s).
        c: frequency exponent in (0, 1].

    All parameters may be torch tensors (gradients flow) or python floats. Returns a complex torch tensor
    broadcast over ``omega``.
    """
    torch = _torch()
    omega = torch.as_tensor(omega, dtype=torch.float64)
    sigma_inf = torch.as_tensor(sigma_inf, dtype=torch.float64)
    m = torch.as_tensor(m, dtype=torch.float64)
    tau = torch.as_tensor(tau, dtype=torch.float64)
    c = torch.as_tensor(c, dtype=torch.float64)
    iwt = 1j * (omega * tau).to(torch.complex128)
    # (i omega tau)^c via complex log; at omega=0 the term is 0 (DC limit) so guard the branch point.
    powered = torch.where(
        omega.to(torch.complex128) == 0,
        torch.zeros_like(iwt),
        torch.exp(c.to(torch.complex128) * torch.log(iwt + 0j)),
    )
    return sigma_inf.to(torch.complex128) * (1.0 - m.to(torch.complex128) / (1.0 + powered))


def _homogeneous_field(sigma_node, shape, schedule, spacing, cell):
    """Solve the Poisson operator once per unique current injection for a (possibly complex) node field."""
    from mixle_pde.pde_solve import divergence_form, sparse_solve

    torch = _torch()
    n = int(np.prod(shape))
    rows, cols, vals, nn = divergence_form(sigma_node, shape, spacing=spacing)
    pot = {}
    for q in schedule:
        a, b = q[0], q[1]
        if (a, b) in pot:
            continue
        rhs = torch.zeros(n, dtype=sigma_node.dtype)
        rhs[a] = 1.0 / cell
        if b is not None:
            rhs[b] = -1.0 / cell
        pot[(a, b)] = sparse_solve(vals, rows, cols, nn, rhs)
    out = []
    for q in schedule:
        a, b, mm = q[0], q[1], q[2]
        ne = q[3] if len(q) > 3 else None
        phi = pot[(a, b)]
        out.append(phi[mm] - (phi[ne] if ne is not None else 0.0))
    return torch.stack(out)


def sip_forward(
    log_sigma_inf,
    m,
    tau,
    c,
    shape,
    schedule,
    omega,
    *,
    spacing=1.0,
    sigma_ref=1.0,
    clamp=12.0,
):
    r"""Spectral IP forward: complex transfer resistances for a quadrupole schedule at each frequency.

    Assembles the Cole-Cole complex conductivity ``sigma(omega) = sigma_inf (1 - m/(1 + (i omega tau)^c))``
    with ``sigma_inf = sigma_ref * exp(log_sigma_inf)`` and runs the DC Poisson operator
    ``-div(sigma grad phi) = I`` (the same machinery as ``geophysics.dc_resistivity``) at every frequency.
    The Dirichlet box acts as the far-field ground, so electrodes must be interior nodes.

    Args:
        log_sigma_inf: torch tensor, per-node log high-frequency conductivity contrast (length ``prod(shape)``).
        m, tau, c: Cole-Cole parameters. Scalars (whole-space) or per-node torch tensors (length ``prod(shape)``).
        shape: grid shape ``(nx, ny[, nz])``.
        schedule: sequence of quadrupoles ``(a, b, m_elec, n_elec)``; ``b`` and/or ``n_elec`` may be ``None``.
        omega: iterable of angular frequencies (rad/s).
        spacing: grid spacing (scalar or per-axis).
        sigma_ref: reference conductivity multiplying ``exp(log_sigma_inf)``.
        clamp: bound ``|log_sigma_inf|`` before exponentiating; ``None`` disables.

    Returns:
        complex torch tensor of shape ``(len(omega), len(schedule))`` -- the complex transfer resistances
        ``R(omega) = (phi_m - phi_n) / I``. Differentiable in ``(log_sigma_inf, m, tau, c)``.
    """
    torch = _torch()
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    cell = float(np.prod(np.atleast_1d(spacing)))
    if clamp is not None:
        log_sigma_inf = torch.clamp(log_sigma_inf, -float(clamp), float(clamp))
    if not torch.is_tensor(log_sigma_inf):
        log_sigma_inf = torch.as_tensor(log_sigma_inf, dtype=torch.float64)
    if log_sigma_inf.ndim == 0:
        log_sigma_inf = log_sigma_inf.expand(n)
    sigma_inf = sigma_ref * torch.exp(log_sigma_inf)

    def _expand(x):
        x = torch.as_tensor(x, dtype=torch.float64)
        return x.expand(n) if x.ndim == 0 else x

    m_n, tau_n, c_n = _expand(m), _expand(tau), _expand(c)

    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    rows_out = []
    for w in omega:
        sigma_node = cole_cole_conductivity(float(w), sigma_inf, m_n, tau_n, c_n)
        rows_out.append(_homogeneous_field(sigma_node, shape, schedule, spacing, cell))
    return torch.stack(rows_out)


def apparent_conductivity(resistances, geometric_factor):
    """Apparent complex conductivity ``sigma_app = geometric_factor / R`` from transfer resistances.

    ``geometric_factor`` is ``K_geom = R_ref * sigma_ref_scalar`` measured on a homogeneous reference (any
    conductivity), where for a homogeneous medium ``R = K_geom / sigma`` exactly. Pass the per-schedule
    ``K_geom`` (broadcast over frequency) to invert each frequency's resistance to apparent conductivity.
    """
    torch = _torch()
    geometric_factor = torch.as_tensor(geometric_factor, dtype=resistances.dtype)
    return geometric_factor / resistances


def geometric_factor(shape, schedule, *, spacing=1.0):
    """The purely geometric factor ``K_geom`` per quadrupole: ``R = K_geom / sigma`` in a homogeneous medium.

    Solves the unit-conductivity DC problem once; the resulting resistances equal ``K_geom`` (since
    ``sigma = 1``). Real-valued and independent of the Cole-Cole parameters.
    """
    torch = _torch()
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    cell = float(np.prod(np.atleast_1d(spacing)))
    sigma_node = torch.ones(n, dtype=torch.float64)
    return _homogeneous_field(sigma_node, shape, schedule, spacing, cell).real
