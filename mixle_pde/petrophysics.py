"""Closed-form petrophysical transforms over well-log curves (workstream B3).

A wireline log (:mod:`mixle_pde.io.las`) records instrument responses -- sonic slowness, bulk
density, deep resistivity -- not the physical quantities an inversion or a rock-physics model wants.
These three elementwise transforms bridge that gap, each a standard closed-form petrophysics
relation with no fitting involved:

* :func:`vp_from_dt` turns sonic slowness (microseconds per foot) into compressional velocity.
* :func:`gardner_density` estimates bulk density from velocity (Gardner's relation) when a density
  curve is missing or needs a consistency check against the recorded ``RHOB``.
* :func:`archie_sw` turns resistivity and porosity into water saturation (Archie's equation), the
  classic reservoir-engineering read of "how much of the pore space is water versus hydrocarbon".

``Vp``/``rho`` computed here feed :func:`mixle_pde.rock_physics.moduli_from_velocity` directly, so a
well log becomes elastic moduli with no PDE involved -- the log-domain analogue of the seismic
Vp/Vs/rho -> (K, mu) map.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def vp_from_dt(dt_us_per_ft: Any) -> np.ndarray:
    """Compressional velocity (km/s) from sonic slowness (microseconds per foot).

    ``Vp = 304.8 / dt_us_per_ft`` -- 304.8 mm/ft converts the reciprocal slowness to km/s directly
    (1 ft/us = 304.8 km/s; a slower, higher-``DT`` rock gives a lower velocity).
    """
    dt = np.asarray(dt_us_per_ft, dtype=float)
    return 304.8 / dt


def gardner_density(vp_km_s: Any, *, alpha: float = 1.741, beta: float = 0.25) -> np.ndarray:
    """Bulk density (g/cm^3) from compressional velocity via Gardner's relation ``rho = alpha * Vp^beta``.

    ``vp_km_s`` is compressional velocity in km/s; the default ``(alpha, beta) = (1.741, 0.25)`` is
    Gardner's original coefficients for that unit convention. Useful as a density estimate when no
    ``RHOB`` curve was recorded, or as a sanity check against one that was.
    """
    vp = np.asarray(vp_km_s, dtype=float)
    return alpha * vp**beta


def archie_sw(
    rt: Any,
    phi: Any,
    *,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    rw: float = 0.1,
) -> np.ndarray:
    """Water saturation from Archie's equation: ``Sw = ((a * rw) / (phi^m * rt)) ** (1/n)``, clipped to [0, 1].

    ``rt`` is true (deep) resistivity (ohm-m), ``phi`` is porosity (fraction), ``rw`` is formation-water
    resistivity (ohm-m), and ``a``/``m``/``n`` are the tortuosity, cementation, and saturation exponents.
    The result is clipped to ``[0, 1]`` because Archie's equation is not itself bounded -- noisy or
    out-of-range inputs (e.g. very low porosity) can push the raw expression outside the physical range.
    """
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    sw = ((a * rw) / (phi**m * rt)) ** (1.0 / n)
    return np.clip(sw, 0.0, 1.0)
