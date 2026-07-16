"""1-D immiscible two-phase (water-oil) displacement: the Buckley-Leverett problem (workstream H, MP-H6).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``, MP-H6) records that
:mod:`mixle_pde.groundwater` and :mod:`mixle_pde.poroelastic` cover Darcy/Biot-class single-phase
(and poroelastic) flow only, and that "no ... two-phase (Buckley-Leverett-class) evidence" exists
anywhere in the package. :mod:`mixle_pde.two_phase` is a different thing entirely -- a diffuse-
interface (phase-field) Navier-Stokes solver for two *immiscible fluids sharing one velocity field*
(core-annular pipe flow); it has no relative permeability, no fractional flow, and no porous medium.
This module is the missing piece: the classic reservoir-engineering displacement problem, where two
fluids move through a *fixed porous medium* at *different* velocities set by their relative mobilities.

**Physics.** Water (subscript ``w``) displaces oil (subscript ``o``) through a 1-D homogeneous porous
column at a constant total Darcy (superficial) velocity ``v`` (incompressible flow: no sources or
accumulation of total volume, so ``v`` is uniform in ``x``). Neglecting gravity and capillary pressure
(the classical Buckley & Leverett 1942 assumptions), conservation of the water phase gives

    phi * dS/dt + v * df(S)/dx = 0,

the saturation ``S = S_w`` transported by the *fractional-flow function* ``f(S) = lambda_w / (lambda_w
+ lambda_o)`` (the fraction of total flow that is water), with phase mobilities ``lambda_p =
k_rp(S) / mu_p`` set by relative permeability curves ``k_rp`` and viscosities ``mu_p``.
:class:`CoreyFractionalFlow` supplies the standard power-law (Corey) relative-permeability closure.

``f`` is monotonically increasing from 0 (at the connate water saturation ``S_wc``, no mobile water)
to 1 (at ``1 - S_or``, the residual oil saturation, no mobile oil), and -- for the usual Corey exponents
``>= 2`` on both phases -- S-shaped: convex near ``S_wc``, concave near ``1 - S_or``, with a single
inflection point. Because ``f`` is monotone, every characteristic of the quasilinear form ``dS/dt +
(v/phi) f'(S) dS/dx = 0`` moves in the same (downstream) direction, but ``f'`` is *not* monotone in
``S`` (it rises from 0, peaks at the inflection point, falls back to 0), so a naive rarefaction fan
from the injected saturation down to the initial saturation would be multi-valued. The physically
correct (entropy) solution instead develops a shock: the classic S-shaped saturation profile with a
smooth rarefaction behind a sharp displacement front.

**Two independent pieces of evidence, both against known closed-form theory, not each other:**

* :func:`welge_shock_front` locates the front (shock) saturation and speed via the Welge (1952)
  tangent construction -- pure algebra/calculus on ``f``, no PDE solve involved. It is checked against
  its own defining property, the Rankine-Hugoniot condition, to machine precision (see
  ``tests/buckley_leverett_test.py``).
* :func:`solve_buckley_leverett_upwind` marches the saturation PDE forward with a first-order explicit
  upwind (Godunov) finite-difference scheme. Because ``f' >= 0`` everywhere on the admissible saturation
  range, the exact Godunov numerical flux at every cell face reduces to plain single-point upwinding
  (``F = f(S_upwind)``), so this "simple upwind" scheme is not an ad hoc approximation -- it is the
  entropy-consistent finite-volume method for this particular (monotone-flux) conservation law.
* :func:`buckley_leverett_analytic` assembles the exact self-similar solution (rarefaction fan + shock)
  from :func:`welge_shock_front` via the method of characteristics, giving an independently-computable
  reference profile the numerical march is checked against.

References: S. Buckley & M. Leverett, "Mechanism of Fluid Displacement in Sands", Trans. AIME 146
(1942) 107-116; H. Welge, "A Simplified Method for Computing Oil Recovery by Gas or Water Drive",
Trans. AIME 195 (1952) 91-98.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "CoreyFractionalFlow",
    "WelgeShock",
    "welge_shock_front",
    "BuckleyLeverettAnalytic",
    "buckley_leverett_analytic",
    "BuckleyLeverettResult",
    "solve_buckley_leverett_upwind",
]


@dataclass(frozen=True)
class CoreyFractionalFlow:
    """Corey power-law relative permeabilities and the water fractional-flow function ``f(S)``.

    ``k_rw(S) = water_endpoint * Se^water_exponent``, ``k_ro(S) = oil_endpoint * (1 - Se)^oil_exponent``,
    with the normalized (effective) saturation ``Se = (S - S_wc) / (1 - S_wc - S_or)`` clamped to
    ``[0, 1]``. ``f(S) = (k_rw/mu_w) / (k_rw/mu_w + k_ro/mu_o)``. The default exponents ``2.0`` are the
    standard "quadratic Corey" textbook choice that gives a genuine S-shaped (inflected) curve; both
    exponents must be ``>= 1`` so the endpoint slopes stay finite.
    """

    water_viscosity: float
    oil_viscosity: float
    connate_water_saturation: float = 0.0
    residual_oil_saturation: float = 0.0
    water_exponent: float = 2.0
    oil_exponent: float = 2.0
    water_endpoint: float = 1.0
    oil_endpoint: float = 1.0

    def __post_init__(self) -> None:
        if self.water_viscosity <= 0.0 or self.oil_viscosity <= 0.0:
            raise ValueError("water_viscosity and oil_viscosity must be positive.")
        if self.water_endpoint <= 0.0 or self.oil_endpoint <= 0.0:
            raise ValueError("water_endpoint and oil_endpoint must be positive.")
        if self.water_exponent < 1.0 or self.oil_exponent < 1.0:
            raise ValueError("water_exponent and oil_exponent must be >= 1 (a sub-linear endpoint is singular).")
        if not (0.0 <= self.connate_water_saturation < 1.0):
            raise ValueError("connate_water_saturation must lie in [0, 1).")
        if not (0.0 <= self.residual_oil_saturation < 1.0):
            raise ValueError("residual_oil_saturation must lie in [0, 1).")
        if self.connate_water_saturation + self.residual_oil_saturation >= 1.0:
            raise ValueError("connate_water_saturation + residual_oil_saturation must be < 1.")

    def _span(self) -> float:
        return 1.0 - self.connate_water_saturation - self.residual_oil_saturation

    def _effective_saturation(self, saturation: Any) -> np.ndarray:
        s = np.asarray(saturation, dtype=float)
        se = (s - self.connate_water_saturation) / self._span()
        return np.clip(se, 0.0, 1.0)

    def relative_permeabilities(self, saturation: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(k_rw, k_ro)`` at the given water saturation(s)."""
        se = self._effective_saturation(saturation)
        krw = self.water_endpoint * se**self.water_exponent
        kro = self.oil_endpoint * (1.0 - se) ** self.oil_exponent
        return krw, kro

    def fractional_flow(self, saturation: Any) -> np.ndarray:
        """The water fractional-flow function ``f(S) = lambda_w / (lambda_w + lambda_o)``."""
        krw, kro = self.relative_permeabilities(saturation)
        lam_w = krw / self.water_viscosity
        lam_o = kro / self.oil_viscosity
        return lam_w / (lam_w + lam_o)

    def fractional_flow_derivative(self, saturation: Any) -> np.ndarray:
        """The analytic derivative ``f'(S)`` (quotient rule on the mobilities, not a finite difference)."""
        se = self._effective_saturation(saturation)
        span = self._span()
        krw = self.water_endpoint * se**self.water_exponent
        kro = self.oil_endpoint * (1.0 - se) ** self.oil_exponent
        dkrw = self.water_endpoint * self.water_exponent * se ** (self.water_exponent - 1.0) / span
        dkro = -self.oil_endpoint * self.oil_exponent * (1.0 - se) ** (self.oil_exponent - 1.0) / span
        lam_w, lam_o = krw / self.water_viscosity, kro / self.oil_viscosity
        dlam_w, dlam_o = dkrw / self.water_viscosity, dkro / self.oil_viscosity
        total = lam_w + lam_o
        return (dlam_w * lam_o - lam_w * dlam_o) / total**2


@dataclass(frozen=True)
class WelgeShock:
    """The Buckley-Leverett flood-front (shock) saturation and dimensionless speed.

    ``saturation`` is the point where the straight line from ``(initial_saturation,
    f(initial_saturation))`` is tangent to the ``f(S)`` curve (Welge's 1952 construction). By
    construction the tangent slope equals both ``f'(saturation)`` and the Rankine-Hugoniot shock speed
    ``(f(saturation) - f(initial_saturation)) / (saturation - initial_saturation)`` -- the two must agree
    to within solver tolerance, which is exactly what makes ``saturation`` the shock front rather than an
    arbitrary point on the curve. ``speed`` is that common value, in units of ``f'`` (multiply by
    ``velocity / porosity`` for a physical front velocity).
    """

    saturation: float
    speed: float


def _max_wave_speed(model: CoreyFractionalFlow, s_lo: float, s_hi: float, *, n_probe: int = 4000) -> float:
    """``max |f'(S)|`` over ``[min(s_lo, s_hi), max(s_lo, s_hi)]``, sampled on a dense grid.

    By the mean value theorem this bounds not only every pointwise characteristic speed the march ever
    evaluates but also every discrete secant slope ``(f(a) - f(b)) / (a - b)`` between two saturations in
    the interval -- i.e. it bounds any shock speed the scheme could ever produce as well, so it is a
    valid (if slightly conservative) CFL speed bound for the whole simulation, computed once up front.
    """
    lo, hi = (s_lo, s_hi) if s_lo <= s_hi else (s_hi, s_lo)
    probes = np.linspace(lo, hi, n_probe)
    return float(np.max(np.abs(model.fractional_flow_derivative(probes))))


def welge_shock_front(model: CoreyFractionalFlow, *, initial_saturation: float | None = None) -> WelgeShock:
    """Solve for the Welge (1952) tangent point: the shock front saturation and speed.

    ``initial_saturation`` (default :attr:`CoreyFractionalFlow.connate_water_saturation`) is the
    saturation ahead of the flood front. The tangent condition is
    ``f'(S) = (f(S) - f(initial_saturation)) / (S - initial_saturation)``; the left side falls from a
    positive value near ``initial_saturation`` to 0 at ``1 - residual_oil_saturation`` while the secant
    slope on the right stays positive throughout, so their difference changes sign exactly once on a
    genuinely S-shaped ``f`` (Corey exponents ``> 1``) -- located by a coarse bracketing scan (robust to
    any admissible exponents) followed by Brent's method for a tight root.
    """
    s_init = model.connate_water_saturation if initial_saturation is None else float(initial_saturation)
    s_max = 1.0 - model.residual_oil_saturation
    if not (0.0 <= s_init < s_max):
        raise ValueError("initial_saturation must lie in [0, 1 - residual_oil_saturation).")
    f_init = float(model.fractional_flow(s_init))

    def gap(s: float) -> float:
        return float(model.fractional_flow_derivative(s)) - (float(model.fractional_flow(s)) - f_init) / (s - s_init)

    span = s_max - s_init
    probes = np.linspace(s_init + 1.0e-9 * span, s_max - 1.0e-9 * span, 4000)
    gaps = np.array([gap(s) for s in probes])
    sign_changes = np.where(np.diff(np.sign(gaps)) < 0)[0]  # expect a single + -> - crossing (see docstring)
    if sign_changes.size == 0:
        raise ValueError(
            "no Welge tangent point found; fractional_flow(model) may not be S-shaped "
            "(check that both Corey exponents are > 1)."
        )
    i = int(sign_changes[0])
    s_f = brentq(gap, probes[i], probes[i + 1], xtol=1.0e-13, rtol=1.0e-13)
    return WelgeShock(saturation=float(s_f), speed=float(model.fractional_flow_derivative(s_f)))


@dataclass(frozen=True)
class BuckleyLeverettAnalytic:
    """The exact self-similar Buckley-Leverett saturation profile from :func:`buckley_leverett_analytic`."""

    positions: np.ndarray
    saturation: np.ndarray
    shock: WelgeShock
    shock_position: float


def buckley_leverett_analytic(
    model: CoreyFractionalFlow,
    positions: Any,
    time: float,
    *,
    velocity: float,
    porosity: float,
    initial_saturation: float | None = None,
    injection_saturation: float | None = None,
) -> BuckleyLeverettAnalytic:
    """The exact Buckley-Leverett saturation profile ``S(x, t)`` from the method of characteristics.

    A column initially at ``initial_saturation`` (default the connate water saturation) is flooded from
    ``x=0`` at a fixed ``injection_saturation`` (default ``1 - residual_oil_saturation``, the standard
    "waterflood at the maximum water saturation" assumption) held for all ``t > 0``. With ``u = velocity /
    porosity``, the closed-form solution at ``positions >= 0`` and the given ``time`` is:

    * ``0 <= x < x_shock``: the rarefaction fan, ``S`` implicitly given by ``x = u * time * f'(S)``, valid
      because :func:`welge_shock_front` places the shock exactly where this range's characteristic speed
      ``f'`` is monotonically decreasing in ``S`` (the curve's concave part) -- so the map is invertible
      and is evaluated here by interpolation on a dense monotone table.
    * ``x_shock <= x``: the undisturbed ``initial_saturation``, where ``x_shock = u * time *
      shock.speed`` is the Rankine-Hugoniot front position.

    Requires ``injection_saturation`` to be at or above the shock saturation (the standard "simple
    waterflood" regime this closed form covers); a lower injection saturation would need a different wave
    structure and is rejected rather than silently mis-evaluated.
    """
    x = np.asarray(positions, dtype=float)
    t = float(time)
    if t <= 0.0:
        raise ValueError("time must be positive.")
    if velocity <= 0.0 or porosity <= 0.0:
        raise ValueError("velocity and porosity must be positive.")
    s_init = model.connate_water_saturation if initial_saturation is None else float(initial_saturation)
    s_inj = (1.0 - model.residual_oil_saturation) if injection_saturation is None else float(injection_saturation)
    shock = welge_shock_front(model, initial_saturation=s_init)
    if s_inj < shock.saturation:
        raise ValueError(
            "injection_saturation is below the Welge shock saturation; this closed form only covers the "
            "standard waterflood regime where injection_saturation >= the shock saturation."
        )
    u = float(velocity) / float(porosity)
    x_shock = u * shock.speed * t

    s_grid = np.linspace(shock.saturation, s_inj, 4000)
    x_grid = u * t * np.asarray(model.fractional_flow_derivative(s_grid), dtype=float)
    order = np.argsort(x_grid)  # x_grid is monotonically decreasing in s_grid; sort ascending for interp

    saturation = np.where(
        x < x_shock,
        np.interp(x, x_grid[order], s_grid[order], left=s_inj, right=shock.saturation),
        s_init,
    )
    return BuckleyLeverettAnalytic(positions=x, saturation=saturation, shock=shock, shock_position=x_shock)


@dataclass(frozen=True)
class BuckleyLeverettResult:
    """Grid and saturation profile from :func:`solve_buckley_leverett_upwind`."""

    positions: np.ndarray
    saturation: np.ndarray
    time: float
    n_steps: int
    dt: float


def solve_buckley_leverett_upwind(
    model: CoreyFractionalFlow,
    *,
    length: float,
    n_cells: int,
    velocity: float,
    porosity: float,
    time: float,
    initial_saturation: float | None = None,
    injection_saturation: float | None = None,
    cfl: float = 0.9,
) -> BuckleyLeverettResult:
    """March ``phi dS/dt + v df(S)/dx = 0`` forward with a first-order explicit upwind finite-difference scheme.

    The domain ``[0, length]`` is split into ``n_cells`` uniform finite-volume cells; the boundary
    condition is a fixed ``injection_saturation`` (default ``1 - residual_oil_saturation``) held at
    ``x=0`` for all ``t > 0``, the initial condition a uniform ``initial_saturation`` (default the connate
    water saturation). Because ``f' >= 0`` everywhere on the admissible range, the update at every cell is
    simply

        S_i <- S_i - (v dt / (phi dx)) * (f(S_i) - f(S_{i-1})),

    i.e. always differencing against the upstream (lower-``x``) neighbour -- the classic single-point
    "upwind" scheme, applied uniformly since flow moves in one direction throughout. No boundary condition
    is needed at ``x=length``: the last cell's own outgoing flux uses only its own (upwind) value, so the
    scheme is automatically a zero-inflow ("outflow") condition there. ``dt`` is set once from a global
    CFL bound (see :func:`_max_wave_speed`) so ``cfl in (0, 1]`` is a genuine Courant number for the whole
    march, then shrunk slightly so an integer number of equal steps lands exactly on ``time``.
    """
    if not (0.0 < cfl <= 1.0):
        raise ValueError("cfl must lie in (0, 1].")
    n = int(n_cells)
    if n < 2:
        raise ValueError("n_cells must be at least 2.")
    length, time = float(length), float(time)
    if length <= 0.0 or time <= 0.0:
        raise ValueError("length and time must be positive.")
    if velocity <= 0.0 or porosity <= 0.0:
        raise ValueError("velocity and porosity must be positive (this solver marches flow in +x only).")

    dx = length / n
    positions = (np.arange(n) + 0.5) * dx
    s_init = model.connate_water_saturation if initial_saturation is None else float(initial_saturation)
    s_inj = (1.0 - model.residual_oil_saturation) if injection_saturation is None else float(injection_saturation)
    if s_inj <= s_init:
        raise ValueError("injection_saturation must exceed initial_saturation (a water-displacing-oil drive).")
    u = velocity / porosity

    max_speed = _max_wave_speed(model, s_init, s_inj)
    if max_speed <= 0.0:
        raise ValueError("fractional_flow_derivative vanishes over [initial_saturation, injection_saturation].")
    dt_limit = cfl * dx / (u * max_speed)
    n_steps = max(1, int(np.ceil(time / dt_limit)))
    dt = time / n_steps

    state = np.full(n, s_init, dtype=float)
    courant = u * dt / dx
    for _ in range(n_steps):
        flux = np.asarray(model.fractional_flow(state), dtype=float)
        flux_upstream = np.concatenate(([float(model.fractional_flow(s_inj))], flux[:-1]))
        state = state - courant * (flux - flux_upstream)

    return BuckleyLeverettResult(positions=positions, saturation=state, time=time, n_steps=n_steps, dt=dt)
