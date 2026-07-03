"""Closed-form differentiable absorption / attenuation models for the wave and Helmholtz solvers.

An outgoing acoustic or electromagnetic wave loses amplitude to the medium it crosses. In a
frequency-domain viscoacoustic solver that loss is carried by a complex modulus,
``m -> m (1 + i/Q)`` (see :mod:`mixle_pde.helmholtz_pml`), where the quality factor ``Q`` sets the
decay ``exp(-omega r / (2 Q c))`` along the ray. This module supplies the physical attenuation
coefficient ``alpha`` (dB per length) for the two media the sonar+radar stack propagates through, plus
the closed-form bridge from ``alpha`` to that ``Q`` slot.

Everything is a chain of elementwise transforms in the mould of :mod:`mixle_pde.rock_physics`: scalar or
field in, scalar or field out, differentiable in frequency / rain-rate / environment through the passed
``ops`` namespace (torch autograd) and falling back to numpy for plain arrays.

Seawater (``f`` in kHz, ``alpha`` in dB/km):
  * :func:`thorp_seawater` -- the classic Thorp (1967) empirical curve, chemistry folded into constants.
  * :func:`francois_garrison_seawater` -- Francois & Garrison (1982) with the boric-acid, MgSO4, and
    pure-water relaxation terms resolved from ``T, S, pH, depth``.

Atmosphere (``f`` in GHz, ``alpha`` in dB/km):
  * :func:`itu_gaseous_oxygen` / :func:`itu_gaseous_water_vapour` / :func:`itu_gaseous` -- ITU-R P.676
    line-by-line gaseous absorption (the oxygen 60 GHz complex + water-vapour lines + dry continuum).
  * :func:`itu_rain_specific` -- ITU-R P.838 rain specific attenuation ``gamma_R = k R^alpha``, with the
    tabulated ``k, alpha`` log-log interpolated in frequency for horizontal / vertical polarisation.

Bridge to the solver:
  * :func:`db_per_length_to_nepers`, :func:`quality_factor`, :func:`complex_modulus_fraction` turn an
    ``alpha`` (dB per length) into the dimensionless ``Q`` (or the imaginary modulus fraction ``1/Q``)
    that :func:`mixle_pde.helmholtz_pml.helmholtz_pml_operator` consumes.

References: Thorp (1967) *J. Acoust. Soc. Am.*; Francois & Garrison (1982) *JASA* 72, 896 & 1879;
Rec. ITU-R P.676 (gaseous attenuation); Rec. ITU-R P.838-3 (rain specific attenuation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "thorp_seawater",
    "francois_garrison_seawater",
    "itu_gaseous_oxygen",
    "itu_gaseous_water_vapour",
    "itu_gaseous",
    "itu_rain_specific",
    "P838_FREQ_GHZ",
    "P838_KH",
    "P838_AH",
    "P838_KV",
    "P838_AV",
    "db_per_length_to_nepers",
    "quality_factor",
    "complex_modulus_fraction",
]

_LN10 = float(np.log(10.0))
_PI = float(np.pi)


def _exp(x, ops):
    if ops is not None:
        try:
            import torch

            if torch.is_tensor(x):
                return ops.exp(x)
        except ImportError:
            pass
    return np.exp(x)


def _sqrt(x, ops):
    if ops is not None:
        try:
            import torch

            if torch.is_tensor(x):
                return ops.sqrt(x)
        except ImportError:
            pass
    return np.sqrt(x)


def _pow10(x, ops):
    return _exp(x * _LN10, ops)


# --- seawater ------------------------------------------------------------------------------------------------
def thorp_seawater(f_khz: Any, *, ops=None) -> Any:
    """Thorp (1967) seawater absorption ``alpha(f)`` in dB/km, ``f`` in kHz.

    The classic form (deep-ocean chemistry folded into the constants):

        alpha = 0.1 f^2/(1 + f^2) + 40 f^2/(4100 + f^2) + 2.75e-4 f^2 + 0.003   [dB/km, f in kHz]

    The two Lorentzian terms are the boric-acid (~1 kHz) and magnesium-sulphate (~40 kHz) relaxations; the
    ``f^2`` term is pure-water viscous absorption and the constant is the low-frequency floor. Differentiable
    in ``f`` (plain arithmetic, works on autograd tensors). ``ops`` is accepted for signature parity.
    """
    del ops
    f2 = f_khz * f_khz
    return 0.1 * f2 / (1.0 + f2) + 40.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003


def francois_garrison_seawater(
    f_khz: Any,
    *,
    temperature_c: Any = 4.0,
    salinity_ppt: Any = 35.0,
    ph: Any = 8.0,
    depth_m: Any = 0.0,
    ops=None,
) -> Any:
    """Francois & Garrison (1982) seawater absorption ``alpha(f, T, S, pH, D)`` in dB/km, ``f`` in kHz.

    Three relaxation contributions summed:

        alpha = A1 P1 f1 f^2/(f1^2 + f^2)   (boric acid, pH- and T-dependent, ~1 kHz)
              + A2 P2 f2 f^2/(f2^2 + f^2)   (magnesium sulphate, S- and depth-dependent, ~100 kHz)
              + A3 P3 f^2                    (pure-water viscosity)

    with the sound speed ``c = 1412 + 3.21 T + 1.19 S + 0.0167 D`` and the published coefficient formulas
    for the relaxation frequencies ``f1, f2``, magnitudes ``A1, A2, A3`` and pressure factors ``P1, P2, P3``.
    ``T`` in deg C, ``S`` in ppt, ``D`` in m. Differentiable in every argument through ``ops`` (torch).

    The pure-water magnitude ``A3`` uses the published low-temperature polynomial (valid ``T <= 20 C``); the
    seawater use here is the cold deep ocean, so that branch is taken. It reduces to the same order of
    magnitude as :func:`thorp_seawater` and matches it to a few percent through the sonar band.
    """
    T = temperature_c
    S = salinity_ppt
    D = depth_m
    f2v = f_khz * f_khz
    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * D
    Tk = T + 273.0

    # boric acid
    A1 = (8.686 / c) * _pow10(0.78 * ph - 5.0, ops)
    P1 = 1.0
    f1 = 2.8 * _sqrt(S / 35.0, ops) * _pow10(4.0 - 1245.0 / Tk, ops)
    term1 = A1 * P1 * f1 * f2v / (f1 * f1 + f2v)

    # magnesium sulphate
    A2 = 21.44 * (S / c) * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * D * D
    f2r = (8.17 * _pow10(8.0 - 1990.0 / Tk, ops)) / (1.0 + 0.0018 * (S - 35.0))
    term2 = A2 * P2 * f2r * f2v / (f2r * f2r + f2v)

    # pure water (low-temperature branch, T <= 20 C)
    A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T * T - 1.50e-8 * T * T * T
    P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * D * D
    term3 = A3 * P3 * f2v

    return term1 + term2 + term3


# --- atmosphere: ITU-R P.676 gaseous absorption --------------------------------------------------------------
# Oxygen spectroscopic table (Rec. ITU-R P.676 Annex 1, Table 1): f0 (GHz), a1..a6.
_OXY = np.array(
    [
        [50.474214, 0.975, 9.651, 6.690, 0.0, 2.566, 6.850],
        [50.987745, 2.529, 8.653, 7.170, 0.0, 2.246, 6.800],
        [51.503360, 6.193, 7.709, 7.640, 0.0, 1.947, 6.729],
        [52.021429, 14.320, 6.819, 8.110, 0.0, 1.667, 6.640],
        [52.542418, 31.240, 5.983, 8.580, 0.0, 1.388, 6.526],
        [53.066934, 64.290, 5.201, 9.060, 0.0, 1.349, 6.206],
        [53.595775, 124.600, 4.474, 9.550, 0.0, 2.227, 5.085],
        [54.130025, 227.300, 3.800, 9.960, 0.0, 3.170, 3.750],
        [54.671180, 389.700, 3.182, 10.370, 0.0, 3.558, 2.654],
        [55.221384, 627.100, 2.618, 10.890, 0.0, 2.560, 2.952],
        [55.783815, 945.300, 2.109, 11.340, 0.0, -1.172, 6.135],
        [56.264774, 543.400, 0.014, 17.030, 0.0, 3.525, -0.978],
        [56.363399, 1331.800, 1.654, 11.890, 0.0, -2.378, 6.547],
        [56.968211, 1746.600, 1.255, 12.230, 0.0, -3.545, 6.451],
        [57.612486, 2120.100, 0.910, 12.620, 0.0, -5.416, 6.056],
        [58.323877, 2363.700, 0.621, 12.950, 0.0, -1.932, 0.436],
        [58.446588, 1442.100, 0.083, 14.910, 0.0, 6.768, -1.273],
        [59.164204, 2379.900, 0.387, 13.530, 0.0, -6.561, 2.309],
        [59.590983, 2090.700, 0.207, 14.080, 0.0, 6.957, -0.776],
        [60.306056, 2103.400, 0.207, 14.150, 0.0, -6.395, 0.699],
        [60.434778, 2438.000, 0.386, 13.390, 0.0, 6.342, -2.825],
        [61.150562, 2479.500, 0.621, 12.920, 0.0, 1.014, -0.584],
        [61.800158, 2275.900, 0.910, 12.630, 0.0, 5.014, -6.619],
        [62.411220, 1915.400, 1.255, 12.170, 0.0, 3.029, -6.759],
        [62.486253, 1503.000, 0.083, 15.130, 0.0, -4.499, 0.844],
        [62.997984, 1490.200, 1.654, 11.740, 0.0, 1.856, -6.675],
        [63.568526, 1078.000, 2.108, 11.340, 0.0, 0.658, -6.139],
        [64.127675, 728.700, 2.617, 10.880, 0.0, -3.036, -2.895],
        [64.678910, 461.300, 3.181, 10.380, 0.0, -3.968, -2.590],
        [65.224078, 274.000, 3.800, 9.960, 0.0, -3.528, -3.680],
        [65.764779, 153.000, 4.473, 9.550, 0.0, -2.548, -5.002],
        [66.302096, 80.400, 5.200, 9.060, 0.0, -1.660, -6.091],
        [66.836834, 39.800, 5.982, 8.580, 0.0, -1.680, -6.393],
        [67.369601, 18.560, 6.818, 8.110, 0.0, -1.956, -6.475],
        [67.900868, 8.172, 7.708, 7.640, 0.0, -2.216, -6.545],
        [68.431006, 3.397, 8.652, 7.170, 0.0, -2.492, -6.600],
        [68.960312, 1.334, 9.650, 6.690, 0.0, -2.773, -6.650],
        [118.750334, 940.300, 0.010, 16.640, 0.0, -0.439, 0.079],
        [368.498246, 67.400, 0.048, 16.400, 0.0, 0.000, 0.000],
        [424.763020, 637.700, 0.044, 16.400, 0.0, 0.000, 0.000],
        [487.249273, 237.400, 0.049, 16.000, 0.0, 0.000, 0.000],
        [715.392902, 98.100, 0.145, 16.000, 0.0, 0.000, 0.000],
        [773.839490, 572.300, 0.141, 16.200, 0.0, 0.000, 0.000],
        [834.145546, 183.100, 0.145, 14.700, 0.0, 0.000, 0.000],
    ]
)

# Water-vapour spectroscopic table (Rec. ITU-R P.676 Annex 1, Table 2): f0 (GHz), b1..b6.
_WV = np.array(
    [
        [22.235080, 0.1079, 2.144, 26.38, 0.76, 5.087, 1.00],
        [67.803960, 0.0011, 8.732, 28.58, 0.69, 4.930, 0.82],
        [119.995940, 0.0007, 8.353, 29.48, 0.70, 4.780, 0.79],
        [183.310087, 2.2730, 0.668, 29.06, 0.77, 5.022, 0.85],
        [321.225630, 0.0470, 6.179, 24.04, 0.67, 4.398, 0.54],
        [325.152888, 1.5140, 1.541, 28.23, 0.64, 4.893, 0.74],
        [336.227764, 0.0010, 9.825, 26.93, 0.69, 4.740, 0.61],
        [380.197353, 11.6700, 1.048, 28.11, 0.54, 5.063, 0.89],
        [390.134508, 0.0045, 7.347, 21.52, 0.63, 4.810, 0.55],
        [437.346667, 0.0632, 5.048, 18.45, 0.60, 4.230, 0.48],
        [439.150807, 0.9098, 3.595, 20.07, 0.63, 4.483, 0.52],
        [443.018343, 0.1920, 5.048, 15.55, 0.60, 5.083, 0.50],
        [448.001085, 10.4100, 1.405, 25.64, 0.66, 5.028, 0.67],
        [470.888999, 0.3254, 3.597, 21.34, 0.66, 4.506, 0.65],
        [474.689092, 1.2600, 2.379, 23.20, 0.65, 4.804, 0.64],
        [488.490108, 0.2529, 2.852, 25.86, 0.69, 5.201, 0.72],
        [503.568532, 0.0372, 6.731, 16.12, 0.61, 3.980, 0.43],
        [504.482692, 0.0124, 6.731, 16.12, 0.61, 4.010, 0.45],
        [547.676440, 0.9785, 0.158, 26.00, 0.70, 4.500, 1.00],
        [552.020960, 0.1840, 0.158, 26.00, 0.70, 4.500, 1.00],
        [556.935985, 497.0000, 0.159, 30.86, 0.69, 4.552, 1.00],
        [620.700807, 5.0150, 2.391, 24.38, 0.71, 4.856, 0.68],
        [645.766085, 0.0067, 8.633, 18.00, 0.60, 4.000, 0.50],
        [658.005280, 0.2732, 7.816, 32.10, 0.69, 4.140, 1.00],
        [752.033113, 243.4000, 0.396, 30.86, 0.68, 4.352, 0.84],
        [841.051732, 0.0134, 8.177, 15.90, 0.33, 5.760, 0.45],
        [859.965698, 0.1325, 8.055, 30.60, 0.68, 4.090, 0.84],
        [899.303175, 0.0547, 7.914, 29.85, 0.68, 4.530, 0.90],
        [902.611085, 0.0386, 8.429, 28.65, 0.70, 5.100, 0.95],
        [906.205957, 0.1836, 5.110, 24.08, 0.70, 4.700, 0.53],
        [916.171582, 8.4000, 1.441, 26.73, 0.70, 5.150, 0.78],
        [923.113555, 0.0079, 10.293, 29.00, 0.70, 5.000, 0.80],
        [970.315022, 9.0090, 1.919, 25.50, 0.64, 4.940, 0.67],
        [987.926764, 134.6000, 0.257, 29.85, 0.68, 4.550, 0.90],
    ]
)


def _as_backend(cols, ref):
    """Return the constant table columns as tensors matching ``ref`` if it is a torch tensor, else numpy.

    The spectroscopic tables are numpy constants; when the frequency (or an environment arg) is a torch
    tensor we must lift the constants to torch so the arithmetic (and its gradient) stays on the backend."""
    try:
        import torch

        if torch.is_tensor(ref):
            return [torch.as_tensor(c, dtype=ref.dtype) for c in cols]
    except ImportError:
        pass
    return list(cols)


def _oxygen_npp(f, p, e, theta, ops):
    """Imaginary refractivity ``N''`` from the oxygen lines + dry continuum (ITU-R P.676)."""
    f0, a1, a2, a3, a4, a5, a6 = _as_backend([_OXY[:, i] for i in range(7)], f)
    si = a1 * 1e-7 * p * theta**3 * _exp(a2 * (1.0 - theta), ops)
    df = a3 * 1e-4 * (p * theta ** (0.8 - a4) + 1.1 * e * theta)
    df = _sqrt(df * df + 2.25e-6, ops)
    delta = (a5 + a6 * theta) * 1e-4 * (p + e) * theta**0.8
    fi = (f / f0) * (
        (df - delta * (f0 - f)) / ((f0 - f) ** 2 + df * df) + (df - delta * (f0 + f)) / ((f0 + f) ** 2 + df * df)
    )
    resonant = sum(si[k] * fi[k] for k in range(len(f0)))
    # Debye dry continuum + nonresonant N2 term
    d = 5.6e-4 * (p + e) * theta**0.8
    nd = f * p * theta**2 * (6.14e-5 / (d * (1.0 + (f / d) ** 2)) + 1.4e-12 * p * theta**1.5 / (1.0 + 1.9e-5 * f**1.5))
    return resonant + nd


def _water_npp(f, p, e, theta, ops):
    """Imaginary refractivity ``N''`` from the water-vapour lines (ITU-R P.676)."""
    f0, b1, b2, b3, b4, b5, b6 = _as_backend([_WV[:, i] for i in range(7)], f)
    si = b1 * 1e-1 * e * theta**3.5 * _exp(b2 * (1.0 - theta), ops)
    df = b3 * 1e-4 * (p * theta**b4 + b5 * e * theta**b6)
    df = 0.535 * df + _sqrt(0.217 * df * df + (2.1316e-12 * f0 * f0) / theta, ops)
    fi = (f / f0) * (df / ((f0 - f) ** 2 + df * df) + df / ((f0 + f) ** 2 + df * df))
    return sum(si[k] * fi[k] for k in range(len(f0)))


def itu_gaseous_oxygen(
    f_ghz: Any, *, pressure_hpa: Any = 1013.0, vapour_hpa: Any = 0.0, temperature_k: Any = 288.15, ops=None
) -> Any:
    """ITU-R P.676 oxygen (dry-air) specific attenuation, dB/km, ``f`` in GHz.

    Line-by-line over the oxygen spectroscopic table (the 60 GHz complex, the 118.75 GHz line and higher)
    plus the Debye dry-air continuum. ``gamma = 0.1820 f N''_oxygen``. Differentiable in frequency.
    Reference behaviour: ~15 dB/km in the 60 GHz band, ~0.008 dB/km at 10 GHz (sea-level dry air).
    """
    theta = 300.0 / temperature_k
    return 0.1820 * f_ghz * _oxygen_npp(f_ghz, pressure_hpa, vapour_hpa, theta, ops)


def itu_gaseous_water_vapour(
    f_ghz: Any, *, pressure_hpa: Any = 1013.0, vapour_hpa: Any = 7.5, temperature_k: Any = 288.15, ops=None
) -> Any:
    """ITU-R P.676 water-vapour specific attenuation, dB/km, ``f`` in GHz.

    Line-by-line over the water-vapour spectroscopic table. ``gamma = 0.1820 f N''_vapour``. The default
    ``vapour_hpa = 7.5`` hPa is the standard 7.5 g/m^3 surface humidity. Differentiable in frequency.
    """
    theta = 300.0 / temperature_k
    return 0.1820 * f_ghz * _water_npp(f_ghz, pressure_hpa, vapour_hpa, theta, ops)


def itu_gaseous(
    f_ghz: Any, *, pressure_hpa: Any = 1013.0, vapour_hpa: Any = 7.5, temperature_k: Any = 288.15, ops=None
) -> Any:
    """Total ITU-R P.676 gaseous specific attenuation (oxygen + water vapour), dB/km, ``f`` in GHz."""
    o2 = itu_gaseous_oxygen(
        f_ghz, pressure_hpa=pressure_hpa, vapour_hpa=vapour_hpa, temperature_k=temperature_k, ops=ops
    )
    wv = itu_gaseous_water_vapour(
        f_ghz, pressure_hpa=pressure_hpa, vapour_hpa=vapour_hpa, temperature_k=temperature_k, ops=ops
    )
    return o2 + wv


# --- atmosphere: ITU-R P.838 rain specific attenuation -------------------------------------------------------
# Rec. ITU-R P.838-3 coefficients (freq GHz, kH, aH, kV, aV); log-log interpolated in frequency.
_P838 = np.array(
    [
        [1.0, 0.0000387, 0.9120, 0.0000352, 0.8800],
        [2.0, 0.0002000, 0.9630, 0.0001380, 0.9230],
        [4.0, 0.0006500, 1.1210, 0.0005910, 1.0750],
        [6.0, 0.0017500, 1.3080, 0.0015500, 1.2650],
        [8.0, 0.0045000, 1.3270, 0.0039500, 1.3100],
        [10.0, 0.0121700, 1.2571, 0.0112900, 1.2156],
        [12.0, 0.0218600, 1.2000, 0.0202800, 1.1825],
        [15.0, 0.0367000, 1.1540, 0.0335000, 1.1280],
        [20.0, 0.0751000, 1.0990, 0.0691000, 1.0651],
        [25.0, 0.1240000, 1.0610, 0.1130000, 1.0300],
        [30.0, 0.1870000, 1.0210, 0.1670000, 1.0000],
        [40.0, 0.3500000, 0.9390, 0.3100000, 0.9290],
        [50.0, 0.5360000, 0.8730, 0.4790000, 0.8680],
        [60.0, 0.7070000, 0.8260, 0.6420000, 0.8240],
        [70.0, 0.8510000, 0.7930, 0.7840000, 0.7930],
        [80.0, 0.9750000, 0.7690, 0.9060000, 0.7690],
        [90.0, 1.0600000, 0.7530, 0.9990000, 0.7540],
        [100.0, 1.1200000, 0.7430, 1.0600000, 0.7440],
    ]
)
P838_FREQ_GHZ = _P838[:, 0].copy()
P838_KH = _P838[:, 1].copy()
P838_AH = _P838[:, 2].copy()
P838_KV = _P838[:, 3].copy()
P838_AV = _P838[:, 4].copy()


def _p838_coeffs(f_ghz, polarisation, ops):
    """``(k, alpha)`` at frequency ``f`` by log-log interpolation of the P.838 table.

    ``k`` is interpolated in ``log k`` vs ``log f`` and ``alpha`` linearly in ``log f`` (the standard
    P.838 interpolation). Differentiable in ``f`` when ``f`` is a torch scalar (numpy interp otherwise)."""
    pol = polarisation.lower()
    if pol in ("h", "horizontal"):
        kcol, acol = P838_KH, P838_AH
    elif pol in ("v", "vertical"):
        kcol, acol = P838_KV, P838_AV
    else:
        raise ValueError(f"polarisation must be 'h' or 'v'; got {polarisation!r}.")
    lf = np.log(P838_FREQ_GHZ)
    logk = np.log(kcol)

    tensor = False
    if ops is not None:
        try:
            import torch

            tensor = torch.is_tensor(f_ghz)
        except ImportError:
            tensor = False
    if not tensor:
        lg = np.log(np.asarray(f_ghz, dtype=float))
        return float(np.exp(np.interp(lg, lf, logk))), float(np.interp(lg, lf, acol))

    import torch

    lg = torch.log(f_ghz)
    lft = torch.as_tensor(lf, dtype=lg.dtype)
    logkt = torch.as_tensor(logk, dtype=lg.dtype)
    act = torch.as_tensor(acol, dtype=lg.dtype)
    # differentiable piecewise-linear interpolation on the sorted knot grid
    idx = torch.clamp(torch.searchsorted(lft, lg), 1, len(lft) - 1)
    x0, x1 = lft[idx - 1], lft[idx]
    w = (lg - x0) / (x1 - x0)
    logk_i = logkt[idx - 1] + w * (logkt[idx] - logkt[idx - 1])
    a_i = act[idx - 1] + w * (act[idx] - act[idx - 1])
    return torch.exp(logk_i), a_i


def itu_rain_specific(f_ghz: Any, rain_rate_mm_h: Any, *, polarisation: str = "h", ops=None) -> Any:
    """ITU-R P.838 rain specific attenuation ``gamma_R = k(f) R^alpha(f)``, dB/km.

    ``f`` in GHz, ``R`` rain rate in mm/h. ``polarisation`` is ``'h'`` (horizontal) or ``'v'`` (vertical);
    ``k, alpha`` are log-log interpolated from the P.838-3 coefficient table. Differentiable in the rain
    rate (and in frequency when ``f`` is a torch scalar). Reference: ~1.7 dB/km at 10 GHz, 50 mm/h (H).
    """
    k, a = _p838_coeffs(f_ghz, polarisation, ops)
    if ops is not None:
        try:
            import torch

            if torch.is_tensor(rain_rate_mm_h) or torch.is_tensor(k):
                r = rain_rate_mm_h if torch.is_tensor(rain_rate_mm_h) else torch.as_tensor(float(rain_rate_mm_h))
                return k * torch.exp(a * torch.log(r))
        except ImportError:
            pass
    return k * rain_rate_mm_h**a


# --- bridge to the complex-modulus Q slot --------------------------------------------------------------------
def db_per_length_to_nepers(alpha_db: Any) -> Any:
    """Convert an attenuation from dB per length to nepers per (the same) length: ``alpha_np = alpha_dB ln10/20``.

    Amplitude decays as ``exp(-alpha_np x)``; a field falling by ``alpha_dB`` decibels per unit length falls by
    ``ln(10)/20`` nepers over that same unit. Keep ``f`` and ``alpha_dB`` on matching length units (see
    :func:`quality_factor`)."""
    return alpha_db * (_LN10 / 20.0)


def quality_factor(alpha_db: Any, f_hz: Any, c: Any, *, length_scale: float = 1000.0) -> Any:
    """Viscoacoustic quality factor ``Q`` from an attenuation coefficient, for the solver's ``m(1 + i/Q)`` slot.

    The complex-modulus wave decays as ``exp(-omega r / (2 Q c))``, i.e. the amplitude attenuation in nepers
    per metre is ``alpha_np = omega / (2 Q c)``, so

        Q = omega / (2 c alpha_np) = (2 pi f) / (2 c alpha_np) = pi f / (c alpha_np).

    ``alpha_db`` is the specific attenuation from the seawater / atmosphere models, in dB per ``length_scale``
    metres (``length_scale = 1000`` because those return dB/km); it is converted to nepers/metre internally.
    ``f_hz`` is the frequency in Hz and ``c`` the wave speed in m/s (both SI, unlike the models' kHz/GHz).
    Returns the dimensionless ``Q`` (larger = less lossy). Differentiable in ``alpha`` and ``f``.
    """
    alpha_np_per_m = db_per_length_to_nepers(alpha_db) / float(length_scale)
    return _PI * f_hz / (c * alpha_np_per_m)


def complex_modulus_fraction(alpha_db: Any, f_hz: Any, c: Any, *, length_scale: float = 1000.0) -> Any:
    """Imaginary modulus fraction ``1/Q`` for ``m -> m (1 + i/Q)`` (the reciprocal of :func:`quality_factor`).

    Some callers want the loss tangent ``1/Q`` directly (it is what multiplies the modulus); this returns it
    without forming ``Q`` first: ``1/Q = c alpha_np / (pi f)``. Differentiable in ``alpha`` and ``f``."""
    alpha_np_per_m = db_per_length_to_nepers(alpha_db) / float(length_scale)
    return (c * alpha_np_per_m) / (_PI * f_hz)
