"""Atmospheric radio refractivity as closed-form differentiable transforms for radar tropospheric propagation.

A radar parabolic-equation (PE) propagator does not march on temperature and pressure; it marches on the
*modified refractivity* ``M(z)``, whose vertical gradient sets whether energy is trapped near the surface
(an anomalous "duct" that lets a radar see far past the geometric horizon) or refracted upward into the
standard atmosphere. This module is the environmental front end: it turns the meteorological state
(pressure, temperature, humidity) into ``N`` and ``M`` by the ITU-R P.453 recommendation, as elementwise
differentiable transforms in the ``rock_physics.py`` mold, so a sounding profile can be a latent driver
behind a propagation observation and inverted for with autograd.

The refractivity of moist air (ITU-R P.453) is

    N = 77.6/T * (P + 4810 * e / T)                            [N-units]

with ``T`` absolute temperature (K), ``P`` total (dry + wet) pressure (hPa), and ``e`` the water-vapour
partial pressure (hPa). The first term is the dry-air (density) contribution; the ``4810 e / T`` term is
the wet contribution and dominates the near-surface variability that forms ducts. Water-vapour pressure is
obtained from relative humidity through the saturation vapour pressure ``e_s(T)`` (ITU-R P.453 water form).

Because the Earth is curved but a PE solver runs on a flat grid, the ray-curvature is folded into the
refractivity via the Earth-flattening transform, giving the modified refractivity

    M(z) = N(z) + 1e6 * z / a_earth = N(z) + 0.157 * z         (z in metres, a_earth = 6371 km)

A layer where ``dM/dz < 0`` is a *trapping duct*: rays bend downward faster than the Earth curves away, so
energy is guided. The standard atmosphere has ``dN/dz ~ -0.039`` N-units/m, hence ``dM/dz ~ +0.118`` (no
trapping); a surface/evaporation duct inverts the sign of ``dM/dz`` over some height interval.

All math goes through the ``ops`` namespace (or plain array arithmetic), so every transform is
backend-agnostic and differentiable end to end.

References: ITU-R P.453 (*The radio refractive index: its formula and refractivity data*); Bean & Dutton,
*Radio Meteorology* (1966); Barrios (1994), *A terrain parabolic equation model* (M-units and ducting).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "EARTH_RADIUS_M",
    "M_CURVATURE_GRADIENT",
    "refractivity",
    "saturation_vapour_pressure",
    "vapour_pressure_from_humidity",
    "modified_refractivity",
    "standard_refractivity_profile",
    "duct_layers",
]

# Earth radius (km 6371) in metres; the modified-refractivity curvature term is 1e6 / a_earth per metre.
EARTH_RADIUS_M = 6.371e6
M_CURVATURE_GRADIENT = 1.0e6 / EARTH_RADIUS_M  # = 0.156961... M-units per metre (~0.157)

# ITU-R P.453 saturation-vapour-pressure coefficients for water (e_s in hPa, t in degrees Celsius).
_ES_A = 6.1121
_ES_B = 17.502
_ES_C = 240.97


def _exp(x, ops):
    """Backend-agnostic exp: the passed differentiable ``ops`` on autograd tensors, else numpy."""
    if ops is not None:
        try:
            import torch

            if torch.is_tensor(x):
                return ops.exp(x)
        except ImportError:
            pass
    return np.exp(x)


def refractivity(pressure_hpa: Any, temperature_k: Any, vapour_pressure_hpa: Any) -> Any:
    """Radio refractivity ``N`` of moist air by ITU-R P.453.

    ``N = 77.6/T * (P + 4810 * e / T)`` in N-units (dimensionless x 1e-6). ``pressure_hpa`` is the total
    pressure ``P`` (hPa), ``temperature_k`` the absolute temperature ``T`` (K), ``vapour_pressure_hpa`` the
    water-vapour partial pressure ``e`` (hPa, from :func:`vapour_pressure_from_humidity`). Scalars or
    fields; pure elementwise arithmetic, so differentiable in every argument.
    """
    dry = pressure_hpa / temperature_k
    wet = 4810.0 * vapour_pressure_hpa / temperature_k**2
    return 77.6 * (dry + wet)


def saturation_vapour_pressure(temperature_c: Any, *, ops=None) -> Any:
    """Saturation water-vapour pressure ``e_s`` (hPa) over water by the ITU-R P.453 form.

    ``e_s = 6.1121 * exp(17.502 t / (t + 240.97))`` with ``t`` in degrees Celsius. The enhancement factor
    (a <0.5% pressure correction) is omitted; it is well below the humidity uncertainty that drives ducting.
    Differentiable via ``ops.exp`` on autograd tensors.
    """
    return _ES_A * _exp(_ES_B * temperature_c / (temperature_c + _ES_C), ops)


def vapour_pressure_from_humidity(relative_humidity: Any, temperature_c: Any, *, ops=None) -> Any:
    """Water-vapour partial pressure ``e`` (hPa) from relative humidity and temperature.

    ``e = RH * e_s(t)`` with ``relative_humidity`` a fraction in [0, 1] and ``temperature_c`` in degrees
    Celsius. Feeds the ``e`` argument of :func:`refractivity`. Differentiable.
    """
    return relative_humidity * saturation_vapour_pressure(temperature_c, ops=ops)


def modified_refractivity(refractivity_n: Any, height_m: Any) -> Any:
    """Modified refractivity ``M`` (M-units) from ``N`` and height by the Earth-flattening transform.

    ``M(z) = N(z) + 1e6 * z / a_earth = N(z) + 0.157 * z`` (``height_m`` in metres, ``a_earth`` = 6371 km).
    This is the quantity a parabolic-equation propagator marches on: ``dM/dz < 0`` over a height interval is
    a trapping duct. Elementwise and differentiable.
    """
    return refractivity_n + M_CURVATURE_GRADIENT * height_m


def standard_refractivity_profile(height_m: Any, *, n0: float = 315.0, scale_height_m: float = 8077.0, ops=None):
    """The standard exponential-atmosphere refractivity profile ``N(z) = N0 * exp(-z / H)``.

    A closed-form reference atmosphere (Bean & Dutton). ``height_m`` in metres, ``n0`` the surface
    refractivity (N-units), ``scale_height_m`` the refractivity scale height ``H``. The default
    ``N0 = 315``, ``H = 8077 m`` reproduces the canonical near-surface gradient ``dN/dz ~ -0.039`` N-units/m
    (hence ``dM/dz ~ +0.118``, no ducting). Differentiable; use it as the no-duct baseline.
    """
    return n0 * _exp(-height_m / scale_height_m, ops)


def duct_layers(height_m: Any, m_profile: Any, *, tol: float = 0.0):
    """Detect trapping layers where the modified refractivity ``M`` decreases with height.

    Given co-ordinate array ``height_m`` and modified-refractivity array ``m_profile`` (same length,
    monotonically increasing height), returns a boolean mask over the ``n-1`` intervals that is ``True``
    where ``dM/dz < -tol`` (a trapping duct). Also usable as a simple duct detector: ``duct_layers(...).any()``
    is ``True`` iff any trapping layer exists. Pure numpy on the sampled profile (not differentiated).
    """
    z = np.asarray(height_m, float)
    m = np.asarray(m_profile, float)
    if z.shape != m.shape or z.ndim != 1 or z.size < 2:
        raise ValueError("height_m and m_profile must be 1-D arrays of equal length >= 2")
    dm_dz = np.diff(m) / np.diff(z)
    return dm_dz < -tol
