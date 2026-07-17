"""1-D lumped material-transport physics for mineral-processing plant design (slurry / conveyor / floc).

This is the "back of the envelope, but load-bearing" layer that sits between the full PDE solvers in
this package and the network-optimization layer in ``mixle.relations``: it turns a handful of design
parameters (pipe diameter, belt geometry, reagent dose) into the scalar capacities and pressure/rate
curves that a production-network model consumes as plain ``cap``/cost numbers. None of the three
capabilities here run a mesh -- that is the deliberate scope (see the module docstrings of
:mod:`mixle_pde.flow`, :mod:`mixle_pde.two_phase`, :mod:`mixle_pde.smoluchowski`, which *do* run full
PDE/steady-state solves for the underlying physics regimes and are cross-referenced below).

Slurry hydraulics (:func:`slurry_pressure_drop`)
    Homogeneous-model Darcy-Weisbach base loss with the mixture density
    ``rho_m = rho_f (1 - phi) + rho_s phi`` in the ``rho v^2 / 2`` term (the same linear density blend
    :meth:`mixle_pde.two_phase.TwoPhaseFlow2D.rho` uses for its diffuse-interface phase field, specialized
    here to a single bulk solids fraction rather than a spatial field), then a multiplicative Durand-type
    heterogeneous-flow correction for the extra friction the settling/re-suspension of solids adds on top
    of the density effect. The single-phase friction factor is the explicit Swamee-Jain approximation to
    Colebrook-White (laminar ``64/Re`` below transition) -- the standard closed-form stand-in for the
    friction-factor root-find a full incompressible solve (:mod:`mixle_pde.flow`'s streamfunction-vorticity
    Navier-Stokes) would otherwise require; a 1-D correlation has no mesh to converge, so the lumped
    closed form is used directly rather than invoking the 2-D PDE solver for a bulk pipe-friction number.

Conveyor throughput (:func:`conveyor_throughput`)
    The textbook ``belt_speed * cross_section * bulk_density`` mass-rate relation, derated by a
    load-shape factor for the non-rectangular, non-fully-loaded trough profile real belts carry.

Flocculation kinetics (:func:`flocculation_kinetics`)
    The discrete Smoluchowski population-balance (coagulation) equation
    ``dc_k/dt = 1/2 sum_{i+j=k} K(i,j) c_i c_j - c_k sum_i K(k,i) c_i`` integrated over aggregate-size
    bins. The default ("brownian") kernel is the perikinetic Smoluchowski encounter rate for two
    diffusing spheres of combined radius ``a_i + a_j`` and combined diffusivity ``D_i + D_j`` -- exactly
    :func:`mixle_pde.smoluchowski.smoluchowski_rate_free` evaluated pairwise (that function's ``4 pi D a``
    steady absorbing-sphere rate *is* the two-body Brownian coagulation kernel once ``D`` and ``a`` are
    each replaced by the pairwise sums), which is the genuine point of reuse: this module does not
    re-derive the diffusion-limited encounter rate, it just sums it over an aggregate-size population
    instead of a single ligand/target pair.

Every public function here works on plain floats/arrays (no ``torch``/``ops`` backend, no latent
parameters) -- the outputs are meant to be dropped straight into ``cap``/``cost`` arrays for
:func:`mixle.relations.min_cost_flow` (or a residence-time bound for a scheduling constraint), not
differentiated through.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mixle_pde.smoluchowski import smoluchowski_rate_free

__all__ = ["slurry_pressure_drop", "conveyor_throughput", "flocculation_kinetics"]

# Durand-type heterogeneous-flow constant. Both are literature values for the same power-law form
# (Phi = K * [g D (Ss - 1) / (V^2 sqrt(Cd))]^1.5); "durand" is the original Durand & Condolios (1952)
# coefficient commonly cited as ~121, "wilson" is the somewhat more conservative ~150 used in later
# four-component (Wilson/Addie/Sellgren) design practice for coarser, more heterogeneous mixtures.
_DURAND_K = {"durand": 121.0, "wilson": 150.0}


# ---------------------------------------------------------------------------
# Slurry hydraulics
# ---------------------------------------------------------------------------
def _darcy_friction_factor(re: np.ndarray, relative_roughness: float) -> np.ndarray:
    """Darcy friction factor: laminar ``64/Re`` below transition, else the explicit Swamee-Jain
    approximation to Colebrook-White (no iterative root-find needed for a 1-D lumped model)."""
    re_safe = np.maximum(re, 1.0e-6)
    laminar = 64.0 / re_safe
    with np.errstate(divide="ignore", invalid="ignore"):
        turbulent = 0.25 / (np.log10(relative_roughness / 3.7 + 5.74 / re_safe**0.9)) ** 2
    return np.where(re_safe < 2300.0, laminar, turbulent)


def slurry_pressure_drop(
    flow_rate: Any,
    diameter: float,
    length: float,
    solids_fraction: float,
    *,
    rheology: str = "durand",
    rho_fluid: float = 1000.0,
    rho_solid: float = 2650.0,
    viscosity: float = 1.0e-3,
    roughness: float = 4.5e-5,
    drag_coefficient: float = 0.44,
    g: float = 9.80665,
) -> np.ndarray:
    """Pipe pressure drop (Pa) for a settling slurry: mixture-density Darcy-Weisbach plus a Durand
    heterogeneous-flow correction.

    ``flow_rate`` (m^3/s, scalar or array -- the return shape matches it) drives the mean velocity
    ``v = flow_rate / (pi/4 diameter^2)`` through a pipe of ``diameter`` (m) and ``length`` (m) carrying
    solids at volumetric ``solids_fraction`` (``phi`` in ``[0, 1)``). The base loss uses the mixture
    density ``rho_m = rho_fluid (1 - phi) + rho_solid phi`` in the usual ``f (L/D) (rho v^2 / 2)`` form
    (the homogeneous-model first-order density correction); ``rheology`` then selects the Durand-type
    multiplicative correction ``(1 + phi * Phi)`` for the extra friction from imperfectly-suspended
    solids, with ``Phi = K [g D (Ss - 1) / (v^2 sqrt(Cd))]^1.5``, ``Ss = rho_solid / rho_fluid`` and a
    representative particle ``drag_coefficient`` (default 0.44, Newton's-regime drag of the
    coarse/settling fraction that historically calibrated the Durand correlation). ``rheology`` is one of
    ``"durand"`` (K=121) or ``"wilson"`` (K=150, a more conservative heterogeneous-flow variant).

    At the deposition/limiting velocity the correction term diverges (as it should -- the correlation is
    only meant for flows kept above the deposition velocity); very low nonzero ``flow_rate`` will
    therefore blow this up, which is the physically correct warning sign rather than a bug.
    """
    if rheology not in _DURAND_K:
        raise ValueError(f"unknown rheology {rheology!r}; expected one of {sorted(_DURAND_K)}")
    q = np.asarray(flow_rate, dtype=float)
    d = float(diameter)
    length_m = float(length)
    phi = float(solids_fraction)
    if not (0.0 <= phi < 1.0):
        raise ValueError(f"solids_fraction must be in [0, 1); got {phi}")
    if d <= 0.0 or length_m <= 0.0:
        raise ValueError("diameter and length must be positive")

    area = 0.25 * np.pi * d**2
    v = q / area
    rho_m = rho_fluid * (1.0 - phi) + rho_solid * phi

    re = rho_m * np.abs(v) * d / viscosity
    f = _darcy_friction_factor(re, roughness / d)
    dp_base = f * (length_m / d) * (rho_m * v**2 / 2.0)

    ss = rho_solid / rho_fluid
    v_safe = np.maximum(np.abs(v), 1.0e-9)
    froude_term = g * d * (ss - 1.0) / (v_safe**2 * np.sqrt(drag_coefficient))
    correction = 1.0 + phi * _DURAND_K[rheology] * froude_term**1.5
    return dp_base * correction


# ---------------------------------------------------------------------------
# Conveyor throughput
# ---------------------------------------------------------------------------
def conveyor_throughput(
    belt_speed: float,
    cross_section: float,
    bulk_density: float,
    *,
    load_shape_factor: float = 0.9,
) -> float:
    """Mass throughput (t/h) of a belt conveyor: ``belt_speed * cross_section * bulk_density``, derated.

    ``belt_speed`` (m/s), ``cross_section`` (m^2, the loaded material's cross-sectional area on the
    belt), ``bulk_density`` (kg/m^3) give the ideal mass rate; ``load_shape_factor`` (default 0.9) derates
    it for the real, non-rectangular trough/surcharge load profile and the operating margin below the
    fully-loaded design capacity that plant design practice keeps for feed-rate variability.
    """
    if belt_speed < 0.0 or cross_section < 0.0 or bulk_density < 0.0:
        raise ValueError("belt_speed, cross_section, and bulk_density must be non-negative")
    if not (0.0 < load_shape_factor <= 1.0):
        raise ValueError(f"load_shape_factor must be in (0, 1]; got {load_shape_factor}")
    mass_rate_kg_s = belt_speed * cross_section * bulk_density * load_shape_factor
    return mass_rate_kg_s * 3.6  # kg/s -> t/h


# ---------------------------------------------------------------------------
# Flocculation / aggregation kinetics
# ---------------------------------------------------------------------------
def _brownian_kernel(sizes: np.ndarray, *, temperature: float, viscosity: float, monomer_radius: float) -> np.ndarray:
    """Perikinetic Smoluchowski coagulation kernel ``K(i, j) = smoluchowski_rate_free(D_i + D_j, a_i + a_j)``.

    Aggregate ``i`` is modeled as a sphere of radius ``a_i = monomer_radius * i**(1/3)`` (constant
    aggregate density, volume additive in the bin index) with Stokes-Einstein diffusivity
    ``D_i = k_B T / (6 pi viscosity a_i)``. Reusing :func:`mixle_pde.smoluchowski.smoluchowski_rate_free`
    pairwise is exact: that function's ``4 pi D a`` steady absorbing-sphere rate is precisely the
    diffusion-limited two-body encounter rate once ``D`` and ``a`` are the pairwise sums.
    """
    k_b = 1.380649e-23
    a = monomer_radius * sizes ** (1.0 / 3.0)
    dcoef = k_b * temperature / (6.0 * np.pi * viscosity * a)
    n = sizes.size
    # smoluchowski_rate_free is a scalar closed form (4 pi D a); called pairwise rather than vectorized
    # since that is literally the reused function's signature (see mixle_pde.smoluchowski).
    return np.array(
        [[smoluchowski_rate_free(dcoef[i] + dcoef[j], a[i] + a[j]) for j in range(n)] for i in range(n)],
        dtype=float,
    )


_BUILTIN_KERNELS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "constant": lambda sizes: np.ones((sizes.size, sizes.size)),
    "sum": lambda sizes: sizes[:, None] + sizes[None, :],
    "product": lambda sizes: np.outer(sizes, sizes),
}


def _resolve_kernel(kernel: Any, n: int, kernel_kwargs: dict) -> np.ndarray:
    sizes = np.arange(1, n + 1, dtype=float)
    if isinstance(kernel, str):
        if kernel == "brownian":
            return _brownian_kernel(
                sizes,
                temperature=kernel_kwargs.get("temperature", 293.15),
                viscosity=kernel_kwargs.get("viscosity", 1.0e-3),
                monomer_radius=kernel_kwargs.get("monomer_radius", 1.0e-6),
            )
        if kernel in _BUILTIN_KERNELS:
            return np.asarray(_BUILTIN_KERNELS[kernel](sizes), dtype=float)
        raise ValueError(f"unknown kernel {kernel!r}; expected one of 'brownian', {sorted(_BUILTIN_KERNELS)}")
    if callable(kernel):
        return np.asarray([[kernel(i, j) for j in sizes] for i in sizes], dtype=float)
    kmat = np.asarray(kernel, dtype=float)
    if kmat.shape != (n, n):
        raise ValueError(f"kernel matrix must be ({n}, {n}); got {kmat.shape}")
    return kmat


def _coagulation_rhs(c: np.ndarray, kmat: np.ndarray) -> np.ndarray:
    """The discrete Smoluchowski coagulation ODE right-hand side, ``dc/dt`` on bins ``1..N``."""
    n = c.shape[0]
    gain = np.zeros(n, dtype=float)
    for k in range(2, n + 1):  # target size k (1-indexed); pairs (i, k-i), i = 1..k-1
        i = np.arange(1, k)
        j = k - i
        gain[k - 1] = 0.5 * np.sum(kmat[i - 1, j - 1] * c[i - 1] * c[j - 1])
    loss = c * (kmat @ c)
    return gain - loss


def flocculation_kinetics(c0: Any, kernel: Any, t: Any, **kernel_kwargs: Any) -> np.ndarray:
    """Smoluchowski aggregate-size-distribution evolution ``c(t)`` from initial concentrations ``c0``.

    ``c0`` is a length-``N`` array of number concentrations, bin ``k`` (1-indexed) holding aggregates made
    of ``k`` monomers. ``kernel`` selects the pairwise aggregation-rate matrix ``K(i, j)``: the string
    ``"brownian"`` (default physical kernel, the diffusion-limited perikinetic rate reusing
    :func:`mixle_pde.smoluchowski.smoluchowski_rate_free`; accepts ``temperature``, ``viscosity``,
    ``monomer_radius`` keywords), one of the classic solvable test kernels ``"constant"``, ``"sum"``,
    ``"product"``, a callable ``kernel(i, j) -> rate`` evaluated over the ``1..N`` size grid, or an
    ``(N, N)`` rate matrix directly. ``t`` is a scalar or array of output times (seconds).

    Returns an ``(N,)`` array if ``t`` is scalar, else ``(len(t), N)`` -- the concentration in each
    size bin at each requested time, integrated from the coagulation ODE
    ``dc_k/dt = 1/2 sum_{i+j=k} K(i,j) c_i c_j - c_k sum_i K(k,i) c_i``. Useful downstream as a
    residence-time-to-target-size curve: invert the mean-size trajectory for the residence time a
    flocculation tank/thickener needs to reach a target floc size.
    """
    from scipy.integrate import solve_ivp

    c0_arr = np.asarray(c0, dtype=float)
    n = c0_arr.shape[0]
    kmat = _resolve_kernel(kernel, n, kernel_kwargs)

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    scalar_t = np.ndim(t) == 0
    order = np.argsort(t_arr)
    t_sorted = t_arr[order]
    t_span = (0.0, float(max(t_sorted[-1], 1.0e-12)))

    def rhs(_time: float, c: np.ndarray) -> np.ndarray:
        return _coagulation_rhs(np.maximum(c, 0.0), kmat)

    # atol scales with the problem's own concentration magnitude (raw SI number densities can be ~1e17,
    # normalized textbook concentrations ~1) rather than a fixed absolute floor, so the adaptive step
    # size is set by the requested relative accuracy rather than by chasing an irrelevantly tiny floor.
    scale = float(np.max(np.abs(c0_arr))) or 1.0
    sol = solve_ivp(rhs, t_span, c0_arr, t_eval=t_sorted, method="RK45", rtol=1.0e-6, atol=1.0e-9 * scale)
    if not sol.success:
        raise RuntimeError(f"flocculation_kinetics: integration failed ({sol.message})")

    out_sorted = np.maximum(sol.y.T, 0.0)  # (len(t_sorted), N)
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out[0] if scalar_t else out
