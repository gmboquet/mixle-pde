"""Compressible gas dynamics -- the 1-D Euler equations with an exact Riemann solver.

The compressible Euler equations govern inviscid high-speed / high-pressure / high-temperature gas flow:

    d/dt [rho, rho u, E] + d/dx [rho u, rho u^2 + p, u (E + p)] = 0,   p = (gamma - 1)(E - rho u^2 / 2),

for density ``rho``, velocity ``u``, pressure ``p`` and total energy ``E`` of an ideal gas with ratio of
specific heats ``gamma``. This module provides two complementary pieces:

* :func:`exact_riemann_solution` -- the *exact* self-similar solution of the Riemann problem (arbitrary
  left/right states separated by a discontinuity), via Toro's pressure-function Newton iteration for the
  star region followed by exact sampling of the left/right shocks, rarefactions and contact. This is the
  analytic reference for shock-capturing schemes (e.g. the Sod shock tube).
* :func:`solve_euler_1d` -- a conservative finite-volume solver with the HLL approximate-Riemann flux and
  adaptive CFL time stepping, which captures shocks and contacts at the correct speeds and converges to
  the exact Riemann solution as the grid is refined.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics* (2009), ch. 3-4; Sod, *J.
Computational Physics* 27 (1978).
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

State = tuple[float, float, float]  # (rho, u, p)


@dataclass(frozen=True)
class CombustionResult:
    """Time history from a zero-dimensional ideal-gas combustion chamber."""

    time: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    fuel_fraction: np.ndarray
    volume: np.ndarray
    heat_release_rate: np.ndarray


def _wave_function(p: float, rho_k: float, p_k: float, a_k: float, gamma: float) -> float:
    """Toro's ``f_K(p)``: the velocity change across the left/right wave (shock if ``p>p_k`` else fan)."""
    if p > p_k:  # shock
        a = 2.0 / ((gamma + 1.0) * rho_k)
        b = (gamma - 1.0) / (gamma + 1.0) * p_k
        return (p - p_k) * math.sqrt(a / (p + b))
    return 2.0 * a_k / (gamma - 1.0) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


def _star_region(left: State, right: State, gamma: float) -> tuple[float, float]:
    """Solve for the star-region pressure and velocity ``(p*, u*)`` by Newton iteration on the pressure."""
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)

    def total(p: float) -> float:
        return _wave_function(p, rho_l, p_l, a_l, gamma) + _wave_function(p, rho_r, p_r, a_r, gamma) + (u_r - u_l)

    p = 0.5 * (p_l + p_r)
    for _ in range(100):
        dp = 1.0e-7 * p
        deriv = (total(p + dp) - total(p - dp)) / (2.0 * dp)
        p_new = max(p - total(p) / deriv, 1.0e-9)
        if abs(p_new - p) < 1.0e-13 * p:
            p = p_new
            break
        p = p_new
    u_star = 0.5 * (u_l + u_r) + 0.5 * (
        _wave_function(p, rho_r, p_r, a_r, gamma) - _wave_function(p, rho_l, p_l, a_l, gamma)
    )
    return p, u_star


def exact_riemann_solution(left: State, right: State, x: Any, t: float, *, gamma: float = 1.4) -> np.ndarray:
    """Return the exact solution ``(rho, u, p)`` of the Euler Riemann problem at positions ``x``, time ``t``.

    Args:
        left: the left state ``(rho, u, p)`` (for ``x < 0`` at ``t = 0``).
        right: the right state ``(rho, u, p)`` (for ``x > 0``).
        x: positions relative to the initial discontinuity (the solution is self-similar in ``x/t``).
        t: time (> 0).
        gamma: ratio of specific heats.

    Returns:
        Array of shape ``(len(x), 3)`` with columns ``rho, u, p``.
    """
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)
    p_star, u_star = _star_region(left, right, gamma)
    g1 = (gamma - 1.0) / (gamma + 1.0)
    out = np.empty((len(np.asarray(x)), 3))
    for i, xi in enumerate(np.asarray(x, dtype=np.float64)):
        s = xi / t
        if s < u_star:  # left of the contact
            if p_star > p_l:  # left shock
                s_shock = u_l - a_l * math.sqrt(
                    (gamma + 1.0) / (2.0 * gamma) * p_star / p_l + (gamma - 1.0) / (2.0 * gamma)
                )
                if s < s_shock:
                    rho, u, p = rho_l, u_l, p_l
                else:
                    rho = rho_l * (p_star / p_l + g1) / (g1 * p_star / p_l + 1.0)
                    u, p = u_star, p_star
            else:  # left rarefaction
                a_star_l = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
                if s < u_l - a_l:
                    rho, u, p = rho_l, u_l, p_l
                elif s > u_star - a_star_l:
                    rho = rho_l * (p_star / p_l) ** (1.0 / gamma)
                    u, p = u_star, p_star
                else:  # inside the fan
                    u = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * u_l + s)
                    c = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * (u_l - s))
                    rho = rho_l * (c / a_l) ** (2.0 / (gamma - 1.0))
                    p = p_l * (c / a_l) ** (2.0 * gamma / (gamma - 1.0))
        else:  # right of the contact
            if p_star > p_r:  # right shock
                s_shock = u_r + a_r * math.sqrt(
                    (gamma + 1.0) / (2.0 * gamma) * p_star / p_r + (gamma - 1.0) / (2.0 * gamma)
                )
                if s > s_shock:
                    rho, u, p = rho_r, u_r, p_r
                else:
                    rho = rho_r * (p_star / p_r + g1) / (g1 * p_star / p_r + 1.0)
                    u, p = u_star, p_star
            else:  # right rarefaction
                a_star_r = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
                if s > u_r + a_r:
                    rho, u, p = rho_r, u_r, p_r
                elif s < u_star + a_star_r:
                    rho = rho_r * (p_star / p_r) ** (1.0 / gamma)
                    u, p = u_star, p_star
                else:  # inside the fan
                    u = 2.0 / (gamma + 1.0) * (-a_r + (gamma - 1.0) / 2.0 * u_r + s)
                    c = 2.0 / (gamma + 1.0) * (a_r - (gamma - 1.0) / 2.0 * (u_r - s))
                    rho = rho_r * (c / a_r) ** (2.0 / (gamma - 1.0))
                    p = p_r * (c / a_r) ** (2.0 * gamma / (gamma - 1.0))
        out[i] = (rho, u, p)
    return out


def _to_conserved(prim: np.ndarray, gamma: float) -> np.ndarray:
    rho, u, p = prim[0], prim[1], prim[2]
    return np.array([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u * u])


def _to_primitive(cons: np.ndarray, gamma: float) -> np.ndarray:
    rho = cons[0]
    u = cons[1] / rho
    p = (gamma - 1.0) * (cons[2] - 0.5 * rho * u * u)
    return np.array([rho, u, p])


def _euler_flux(cons: np.ndarray, gamma: float) -> np.ndarray:
    rho, u, p = _to_primitive(cons, gamma)
    return np.array([rho * u, rho * u * u + p, u * (cons[2] + p)])


def _hll_flux(ul: np.ndarray, ur: np.ndarray, gamma: float) -> np.ndarray:
    rho_l, u_l, p_l = _to_primitive(ul, gamma)
    rho_r, u_r, p_r = _to_primitive(ur, gamma)
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)
    sl = min(u_l - a_l, u_r - a_r)
    sr = max(u_l + a_l, u_r + a_r)
    if sl >= 0.0:
        return _euler_flux(ul, gamma)
    if sr <= 0.0:
        return _euler_flux(ur, gamma)
    return (sr * _euler_flux(ul, gamma) - sl * _euler_flux(ur, gamma) + sl * sr * (ur - ul)) / (sr - sl)


def solve_euler_1d(
    rho0: Any, u0: Any, p0: Any, dx: float, t_final: float, *, gamma: float = 1.4, cfl: float = 0.4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evolve the 1-D Euler equations with a conservative HLL finite-volume scheme (outflow boundaries).

    Args:
        rho0, u0, p0: initial density, velocity and pressure arrays on a uniform grid of spacing ``dx``.
        dx: cell width.
        t_final: final time.
        gamma: ratio of specific heats.
        cfl: CFL number for the adaptive time step.

    Returns:
        The final ``(rho, u, p)`` arrays. Captures shocks/contacts at the correct speeds and converges to
        :func:`exact_riemann_solution` under grid refinement.
    """
    prim = np.vstack(
        [np.asarray(rho0, dtype=np.float64), np.asarray(u0, dtype=np.float64), np.asarray(p0, dtype=np.float64)]
    )
    n = prim.shape[1]
    u = np.array([_to_conserved(prim[:, i], gamma) for i in range(n)]).T
    t = 0.0
    while t < t_final:
        pr = np.array([_to_primitive(u[:, i], gamma) for i in range(n)])
        smax = float(np.max(np.abs(pr[:, 1]) + np.sqrt(gamma * pr[:, 2] / pr[:, 0])))
        dt = min(cfl * dx / smax, t_final - t)
        f = np.zeros((3, n + 1))
        for i in range(1, n):
            f[:, i] = _hll_flux(u[:, i - 1], u[:, i], gamma)
        f[:, 0] = _euler_flux(u[:, 0], gamma)  # zero-gradient (outflow) boundaries
        f[:, n] = _euler_flux(u[:, n - 1], gamma)
        u = u - dt / dx * (f[:, 1:] - f[:, :-1])
        t += dt
    pr = np.array([_to_primitive(u[:, i], gamma) for i in range(n)])
    return pr[:, 0], pr[:, 1], pr[:, 2]


def solve_reactive_euler_1d(
    rho0: Any,
    u0: Any,
    p0: Any,
    fuel0: Any,
    dx: float,
    t_final: float,
    *,
    gamma: float = 1.35,
    cfl: float = 0.35,
    gas_constant: float = 287.0,
    heat_release: float = 4.0e7,
    pre_exponential: float = 1000.0,
    activation_temperature: float = 5000.0,
    reaction_order: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evolve 1-D Euler flow with a passively advected fuel fraction and heat release."""
    rho = np.asarray(rho0, dtype=np.float64).reshape(-1)
    vel = np.asarray(u0, dtype=np.float64).reshape(-1)
    pressure = np.asarray(p0, dtype=np.float64).reshape(-1)
    fuel = np.broadcast_to(np.asarray(fuel0, dtype=np.float64), rho.shape).copy()
    if rho.shape != vel.shape or rho.shape != pressure.shape:
        raise ValueError("rho0, u0, and p0 must have the same shape.")
    if np.any(rho <= 0.0) or np.any(pressure <= 0.0) or np.any(fuel < 0.0):
        raise ValueError("density and pressure must be positive and fuel must be nonnegative.")
    if dx <= 0.0 or t_final < 0.0:
        raise ValueError("dx must be positive and t_final must be nonnegative.")
    if gamma <= 1.0 or gas_constant <= 0.0 or reaction_order <= 0.0:
        raise ValueError("gamma, gas_constant, and reaction_order are invalid.")

    n = int(rho.size)
    conserved = np.empty((4, n), dtype=np.float64)
    for i in range(n):
        conserved[:3, i] = _to_conserved(np.array([rho[i], vel[i], pressure[i]]), gamma)
        conserved[3, i] = rho[i] * fuel[i]

    t = 0.0
    while t < t_final:
        prim = np.array([_to_reactive_primitive(conserved[:, i], gamma) for i in range(n)])
        sound = np.sqrt(gamma * prim[:, 2] / prim[:, 0])
        smax = float(np.max(np.abs(prim[:, 1]) + sound))
        dt = min(float(cfl) * float(dx) / max(smax, 1.0e-12), float(t_final) - t)
        flux = np.zeros((4, n + 1), dtype=np.float64)
        for i in range(1, n):
            flux[:, i] = _hll_reactive_flux(conserved[:, i - 1], conserved[:, i], gamma)
        flux[:, 0] = _reactive_flux(conserved[:, 0], gamma)
        flux[:, n] = _reactive_flux(conserved[:, n - 1], gamma)
        conserved = conserved - dt / float(dx) * (flux[:, 1:] - flux[:, :-1])
        _apply_arrhenius_heat_release(
            conserved,
            dt,
            gamma=gamma,
            gas_constant=gas_constant,
            heat_release=heat_release,
            pre_exponential=pre_exponential,
            activation_temperature=activation_temperature,
            reaction_order=reaction_order,
        )
        t += dt

    prim = np.array([_to_reactive_primitive(conserved[:, i], gamma) for i in range(n)])
    return prim[:, 0], prim[:, 1], prim[:, 2], prim[:, 3]


def engine_cylinder_volume(
    time: Any,
    *,
    clearance_volume: float,
    swept_volume: float,
    rpm: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Approximate piston/cylinder volume history from a sinusoidal crank motion.

    ``clearance_volume`` is the top-dead-centre volume and ``swept_volume`` is the displacement volume.
    The returned volume oscillates between ``clearance_volume`` and
    ``clearance_volume + swept_volume``. This is the lightweight moving-volume input needed by the
    zero-dimensional combustion model; detailed crank/rod kinematics can replace it later without
    changing the combustion API.
    """

    t = np.asarray(time, dtype=float)
    theta = 2.0 * math.pi * float(rpm) / 60.0 * t + float(phase)
    return float(clearance_volume) + 0.5 * float(swept_volume) * (1.0 - np.cos(theta))


def simulate_zero_d_combustion(
    time: Any,
    *,
    initial_temperature: float,
    initial_pressure: float,
    initial_fuel_fraction: float = 1.0,
    volume: Any = 1.0,
    gamma: float = 1.35,
    gas_constant: float = 287.0,
    heat_release: float = 4.4e7,
    pre_exponential: float = 50.0,
    activation_temperature: float = 8000.0,
    reaction_order: float = 1.0,
    wall_temperature: float | None = None,
    heat_loss_coefficient: float = 0.0,
    surface_area: float = 0.0,
) -> CombustionResult:
    """Integrate a zero-dimensional reactive ideal-gas chamber.

    The state is temperature ``T`` and fuel mass fraction ``Y`` in a closed chamber. Pressure is computed
    from the ideal-gas law using the initial mass. The model includes Arrhenius one-step fuel depletion,
    heat release, optional wall heat loss, and ``-p dV`` work from a prescribed volume history. It is a
    fast simulator kernel for engine-cylinder and explosion studies, not a substitute for turbulent CFD or
    detailed chemistry.
    """

    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("time must contain at least two entries.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    if initial_temperature <= 0.0 or initial_pressure <= 0.0:
        raise ValueError("initial_temperature and initial_pressure must be positive.")
    if initial_fuel_fraction < 0.0:
        raise ValueError("initial_fuel_fraction must be nonnegative.")
    if gamma <= 1.0 or gas_constant <= 0.0:
        raise ValueError("gamma must exceed one and gas_constant must be positive.")
    if pre_exponential < 0.0 or activation_temperature < 0.0 or reaction_order <= 0.0:
        raise ValueError("reaction parameters must be nonnegative and reaction_order must be positive.")

    vol = _volume_history(volume, t)
    if np.any(vol <= 0.0):
        raise ValueError("volume must be positive at every time.")
    dvol_dt = np.gradient(vol, t)
    cv = float(gas_constant) / (float(gamma) - 1.0)
    mass = float(initial_pressure) * vol[0] / (float(gas_constant) * float(initial_temperature))
    wall_t = float(initial_temperature) if wall_temperature is None else float(wall_temperature)
    h_area = float(heat_loss_coefficient) * float(surface_area)

    temp = np.empty_like(t)
    fuel = np.empty_like(t)
    temp[0] = float(initial_temperature)
    fuel[0] = float(initial_fuel_fraction)

    def rhs(time_i: float, state: np.ndarray) -> np.ndarray:
        temperature = max(float(state[0]), 1.0)
        fuel_fraction = max(float(state[1]), 0.0)
        volume_i = float(np.interp(time_i, t, vol))
        dvolume_i = float(np.interp(time_i, t, dvol_dt))
        pressure_i = mass * float(gas_constant) * temperature / volume_i
        rate = float(pre_exponential) * math.exp(-float(activation_temperature) / temperature)
        burn = rate * (fuel_fraction ** float(reaction_order))
        heat_loss = h_area * (temperature - wall_t) / mass
        dtemperature = (float(heat_release) * burn - pressure_i * dvolume_i / mass - heat_loss) / cv
        return np.array([dtemperature, -burn], dtype=float)

    for i in range(t.size - 1):
        dt = float(t[i + 1] - t[i])
        state = np.array([temp[i], fuel[i]], dtype=float)
        k1 = rhs(float(t[i]), state)
        k2 = rhs(float(t[i] + 0.5 * dt), state + 0.5 * dt * k1)
        k3 = rhs(float(t[i] + 0.5 * dt), state + 0.5 * dt * k2)
        k4 = rhs(float(t[i + 1]), state + dt * k3)
        next_state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        temp[i + 1] = max(float(next_state[0]), 1.0)
        fuel[i + 1] = min(max(float(next_state[1]), 0.0), float(initial_fuel_fraction))

    pressure = mass * float(gas_constant) * temp / vol
    burn_rate = np.asarray(
        [
            float(pre_exponential)
            * math.exp(-float(activation_temperature) / max(float(temp_i), 1.0))
            * (max(float(fuel_i), 0.0) ** float(reaction_order))
            for temp_i, fuel_i in zip(temp, fuel, strict=True)
        ],
        dtype=float,
    )
    heat_release_rate = mass * float(heat_release) * burn_rate
    return CombustionResult(
        time=t,
        temperature=temp,
        pressure=pressure,
        fuel_fraction=fuel,
        volume=vol,
        heat_release_rate=heat_release_rate,
    )


def _volume_history(volume: Any, time: np.ndarray) -> np.ndarray:
    if callable(volume):
        return np.asarray([volume(float(ti)) for ti in time], dtype=float)
    vol = np.asarray(volume, dtype=float)
    if vol.ndim == 0:
        return np.full_like(time, float(vol), dtype=float)
    if vol.shape != time.shape:
        raise ValueError("volume must be a scalar, callable, or array with the same shape as time.")
    return vol


def _to_reactive_primitive(cons: np.ndarray, gamma: float) -> np.ndarray:
    rho = max(float(cons[0]), 1.0e-14)
    velocity = float(cons[1]) / rho
    pressure = max((float(gamma) - 1.0) * (float(cons[2]) - 0.5 * rho * velocity * velocity), 1.0e-14)
    fuel = max(float(cons[3]) / rho, 0.0)
    return np.array([rho, velocity, pressure, fuel], dtype=np.float64)


def _reactive_flux(cons: np.ndarray, gamma: float) -> np.ndarray:
    rho, velocity, pressure, fuel = _to_reactive_primitive(cons, gamma)
    return np.array(
        [
            rho * velocity,
            rho * velocity * velocity + pressure,
            velocity * (float(cons[2]) + pressure),
            rho * fuel * velocity,
        ],
        dtype=np.float64,
    )


def _hll_reactive_flux(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    rho_l, u_l, p_l, _ = _to_reactive_primitive(left, gamma)
    rho_r, u_r, p_r, _ = _to_reactive_primitive(right, gamma)
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)
    sl = min(u_l - a_l, u_r - a_r)
    sr = max(u_l + a_l, u_r + a_r)
    if sl >= 0.0:
        return _reactive_flux(left, gamma)
    if sr <= 0.0:
        return _reactive_flux(right, gamma)
    return (sr * _reactive_flux(left, gamma) - sl * _reactive_flux(right, gamma) + sl * sr * (right - left)) / (sr - sl)


def _apply_arrhenius_heat_release(
    conserved: np.ndarray,
    dt: float,
    *,
    gamma: float,
    gas_constant: float,
    heat_release: float,
    pre_exponential: float,
    activation_temperature: float,
    reaction_order: float,
) -> None:
    for i in range(conserved.shape[1]):
        rho, _, pressure, fuel = _to_reactive_primitive(conserved[:, i], gamma)
        if fuel <= 0.0:
            conserved[3, i] = 0.0
            continue
        temperature = max(pressure / (rho * float(gas_constant)), 1.0)
        rate = float(pre_exponential) * math.exp(-float(activation_temperature) / temperature)
        burned = min(fuel, float(dt) * rate * (fuel ** float(reaction_order)))
        conserved[3, i] = rho * (fuel - burned)
        conserved[2, i] += rho * float(heat_release) * burned
