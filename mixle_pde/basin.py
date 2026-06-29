"""Petroleum-systems / thermal-history forwards: the conductive geotherm and organic maturation.

Basin thermal modelling asks the highest-stakes question in frontier exploration — *did the source rock reach
the oil window, and when, relative to trap formation?* — and answers it from two ingredients: the temperature a
rock has experienced through its burial history, and a kinetic model that converts that temperature history
into a measurable maturity (vitrinite reflectance, %Ro). This module supplies both as differentiable forwards
that pair with the package's inverse machinery (:func:`mixle_pde.geophysics.regularized_gauss_newton`,
or ``mixle.ppl`` for a Bayesian posterior) so that *measured* present-day temperatures and %Ro can be inverted
for the unknown — typically the basal heat-flow history — with calibrated uncertainty.

* :func:`geotherm` — steady-state conductive temperature with depth for a layered column (depth-varying thermal
  conductivity and radiogenic heat production), the present-day thermal state and the forward for a
  heat-flow / geothermal-gradient inversion.
* :func:`easy_ro` / :func:`easy_ro_profile` — the **EASY%Ro** vitrinite-reflectance maturation model of Sweeney
  & Burnham (1990): twenty parallel first-order Arrhenius reactions integrated over a time–temperature history.
  This is the de-facto standard maturity model for calibrating paleothermal histories. The implementation is
  validated against its published calibration in ``tests/basin_test.py`` (Ro 0.20 % immature → 4.69 %
  overmature; oil-window onset Ro≈0.6 % near 100–115 °C; with the correct heating-rate dependence).

The thermal history that feeds maturation is assembled from a burial history (when each layer was at what
depth) and a geotherm per time step — that assembly is problem-specific and lives in the application, while the
two physical kernels here are general.

References: Sweeney & Burnham (1990), *AAPG Bull.* 74, 1559–1570 (EASY%Ro); Allen & Allen, *Basin Analysis*
(conductive geotherms, heat production); Lerche (1990), *Basin Analysis: Quantitative Methods*.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EASYRO_FREQUENCY_FACTOR",
    "EASYRO_ACTIVATION_ENERGIES",
    "EASYRO_WEIGHTS",
    "GAS_CONSTANT_KCAL",
    "geotherm",
    "easy_ro",
    "easy_ro_profile",
]

# --- EASY%Ro kinetics (Sweeney & Burnham 1990) ---------------------------------------------------------------
EASYRO_FREQUENCY_FACTOR = 1.0e13  # Arrhenius pre-exponential A, s^-1
GAS_CONSTANT_KCAL = 0.0019872041  # gas constant R, kcal / (mol K)
# twenty parallel reactions: activation energies (kcal/mol) and their stoichiometric weights (sum = 0.85)
EASYRO_ACTIVATION_ENERGIES = np.array(
    [34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0,
     54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0]
)
EASYRO_WEIGHTS = np.array(
    [0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07,
     0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
)

_SEC_PER_MYR = 1.0e6 * 365.25 * 24.0 * 3600.0


def geotherm(thickness, conductivity, *, surface_temp=10.0, surface_heat_flow=0.060, heat_production=0.0):
    """Steady-state conductive geotherm for a layered 1-D column.

    Solves $\\,dT/dz = q(z)/k(z)\\,$ with the upward heat flow drawn down by radiogenic production,
    $\\,q(z) = q_0 - \\int_0^z H\\,dz'\\,$, marching down from the surface. Heat production makes the geotherm
    concave (steeper near surface) — the correct first-order crustal behaviour.

    Parameters
    ----------
    thickness : array (n_layer,)
        Layer thicknesses from the surface downward, in metres.
    conductivity : float or array (n_layer,)
        Thermal conductivity per layer, W m^-1 K^-1 (e.g. ~1.5 shale, ~3.5 sandstone, ~6 salt).
    surface_temp : float
        Surface (seabed / ground) temperature, °C.
    surface_heat_flow : float
        Surface heat flow $q_0$, W m^-2 (e.g. 0.060 = 60 mW m^-2). The natural quantity to invert for.
    heat_production : float or array (n_layer,)
        Volumetric radiogenic heat production per layer, W m^-3 (~1e-6 typical upper crust; 0 to ignore).

    Returns
    -------
    depth : array (n_layer,)
        Depth to the bottom of each layer, metres.
    temperature : array (n_layer,)
        Temperature at the bottom of each layer, °C.
    """
    dz = np.asarray(thickness, float)
    k = np.broadcast_to(np.asarray(conductivity, float), dz.shape)
    H = np.broadcast_to(np.asarray(heat_production, float), dz.shape)
    T = float(surface_temp)
    q = float(surface_heat_flow)
    depth, temp = [], []
    z = 0.0
    for j in range(len(dz)):
        # within layer j the upward flux falls linearly; integrate dT = q/k dz with q(z)=q - H*(z-z_top)
        T = T + (q * dz[j] - 0.5 * H[j] * dz[j] ** 2) / k[j]
        q = q - H[j] * dz[j]
        z = z + dz[j]
        depth.append(z); temp.append(T)
    return np.array(depth), np.array(temp)


def easy_ro(time_ma, temperature_c):
    """Vitrinite reflectance %Ro from a time–temperature history by the EASY%Ro model.

    Parameters
    ----------
    time_ma : array (n_t,)
        Time in millions of years, monotonically increasing toward the present (the spacing sets the
        integration step; finer is more accurate).
    temperature_c : array (n_t,)
        Temperature of the horizon at each time, °C.

    Returns
    -------
    float
        Vitrinite reflectance, %Ro (0.20 % immature → 4.69 % at full maturation).
    """
    t = np.asarray(time_ma, float) * _SEC_PER_MYR
    Tk = np.asarray(temperature_c, float) + 273.15
    if t.shape != Tk.shape or t.ndim != 1 or t.size < 2:
        raise ValueError("time_ma and temperature_c must be 1-D arrays of equal length >= 2")
    integral = np.zeros_like(EASYRO_ACTIVATION_ENERGIES)
    for j in range(1, len(t)):
        dt = t[j] - t[j - 1]
        t_mid = 0.5 * (Tk[j] + Tk[j - 1])  # midpoint temperature over the step
        integral += EASYRO_FREQUENCY_FACTOR * np.exp(-EASYRO_ACTIVATION_ENERGIES / (GAS_CONSTANT_KCAL * t_mid)) * dt
    reacted = EASYRO_WEIGHTS * (1.0 - np.exp(-integral))
    F = float(np.sum(reacted))
    return float(np.exp(-1.6 + 3.7 * F))


def easy_ro_profile(time_ma, temperatures):
    """%Ro for several horizons at once. ``temperatures`` is (n_horizon, n_t) sharing the ``time_ma`` axis."""
    temperatures = np.atleast_2d(np.asarray(temperatures, float))
    return np.array([easy_ro(time_ma, row) for row in temperatures])
