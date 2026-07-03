"""Ocean sound-speed as closed-form differentiable transforms c(T, S, D) -- the acoustic refractive driver.

Underwater-acoustic propagation (sonar, the parabolic-equation and ray forwards elsewhere in this package)
is driven by the sound-speed field c(T, S, D): temperature, salinity, and depth (or pressure) set the local
phase speed, and the acoustic refractive index is n = c_ref / c. The ocean does not hand you c directly; it
hands you a temperature/salinity structure, and c is recovered through an empirical equation of state. Two
are standard:

* :func:`mackenzie` -- the nine-term equation of Mackenzie (1981), c(T, S, D) in depth (metres) directly.
  Compact, accurate to a few tenths of m/s over 2-30 C, 25-40 ppt, 0-8000 m; the workhorse for ray tracing.
* :func:`unesco` -- the UNESCO / Chen-Millero(-Li) international-standard equation, c(T, S, P) in pressure
  (bar). Split into pure-water, salinity, and pressure blocks; valid 0-40 C, 5-40 ppt, 0-1000 bar.

Because both are pure elementwise polynomials they are differentiable end to end, so T, S, D can be latent
drivers behind a sonar observation and autograd delivers dc/dT, dc/dS, dc/dD for a Gauss-Newton or Bayesian
inverse. All math is plain array/tensor arithmetic, so the transforms run identically on numpy arrays and on
autograd tensors from the ``ops`` namespace, in the :mod:`mixle_pde.rock_physics` closed-form mold.

:func:`depth_to_pressure` / :func:`pressure_to_depth` bridge the two (Leroy & Parthiot 1998), so a Mackenzie
depth field and a UNESCO pressure field describe the same water column.

References: Mackenzie (1981), *JASA* 70, 807-812 (nine-term equation); Chen & Millero (1977), *JASA* 62,
1129-1135; Wong & Zhu / UNESCO (1995); Millero & Li (1994) correction; Leroy & Parthiot (1998), *JASA* 103,
1346-1352 (depth-pressure). Check values are asserted in ``tests/sound_speed_test.py``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "mackenzie",
    "unesco",
    "depth_to_pressure",
    "pressure_to_depth",
]


def mackenzie(T: Any, S: Any, D: Any) -> Any:
    """Sound speed c(T, S, D) by the nine-term equation of Mackenzie (1981).

    ``T`` temperature (deg C), ``S`` salinity (ppt / PSU), ``D`` depth (metres). Scalars or broadcastable
    fields (numpy arrays or autograd tensors). Returns c in m/s. Valid roughly 2-30 C, 25-40 ppt, 0-8000 m.

    The nine terms are the published form
    ``c = 1448.96 + 4.591 T - 5.304e-2 T^2 + 2.374e-4 T^3 + 1.340 (S-35) + 1.630e-2 D + 1.675e-7 D^2
    - 1.025e-2 T (S-35) - 7.139e-13 T D^3``. Pure elementwise arithmetic, so it is differentiable in every
    argument (autograd dc/dT, dc/dS, dc/dD).

    Published check value: ``mackenzie(25, 35, 1000) = 1550.744`` m/s.
    """
    Sm = S - 35.0
    return (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T**2
        + 2.374e-4 * T**3
        + 1.340 * Sm
        + 1.630e-2 * D
        + 1.675e-7 * D**2
        - 1.025e-2 * T * Sm
        - 7.139e-13 * T * D**3
    )


def unesco(T: Any, S: Any, P: Any) -> Any:
    """Sound speed c(T, S, P) by the UNESCO / Chen-Millero(-Li) international-standard equation.

    ``T`` temperature (deg C), ``S`` salinity (ppt / PSU), ``P`` gauge pressure (bar; 0 at the surface).
    Scalars or broadcastable fields (numpy arrays or autograd tensors). Returns c in m/s. Valid 0-40 C,
    5-40 ppt, 0-1000 bar.

    The equation is a sum of a pure-water term and salinity contributions,
    ``c = Cw(T, P) + A(T, P) S + B(T, P) S^{3/2} + D(T, P) S^2``, each block a polynomial in ``T`` and ``P``
    with the Chen-Millero coefficients (Wong & Zhu 1995). Pure elementwise arithmetic, so it is
    differentiable in every argument.

    Use :func:`depth_to_pressure` to feed a depth field. Published check value:
    ``unesco(0, 35, 0) = 1449.14`` m/s (surface, standard salinity).
    """
    # pure-water speed Cw(T, P)
    Cw = (
        1402.388
        + 5.03830 * T
        - 5.81090e-2 * T**2
        + 3.3432e-4 * T**3
        - 1.47797e-6 * T**4
        + 3.1419e-9 * T**5
        + (0.153563 + 6.8999e-4 * T - 8.1829e-6 * T**2 + 1.3632e-7 * T**3 - 6.1260e-10 * T**4) * P
        + (3.1260e-5 - 1.7111e-6 * T + 2.5986e-8 * T**2 - 2.5353e-10 * T**3 + 1.0415e-12 * T**4) * P**2
        + (-9.7729e-9 + 3.8513e-10 * T - 2.3654e-12 * T**2) * P**3
    )
    # salinity-linear block A(T, P)
    A = (
        1.389
        - 1.262e-2 * T
        + 7.166e-5 * T**2
        + 2.008e-6 * T**3
        - 3.21e-8 * T**4
        + (9.4742e-5 - 1.2583e-5 * T - 6.4928e-8 * T**2 + 1.0515e-8 * T**3 - 2.0142e-10 * T**4) * P
        + (-3.9064e-7 + 9.1061e-9 * T - 1.6009e-10 * T**2 + 7.994e-12 * T**3) * P**2
        + (1.100e-10 + 6.651e-12 * T - 3.391e-13 * T**2) * P**3
    )
    # salinity-3/2 block B(T, P) and salinity-squared block D(T, P)
    B = -1.922e-2 - 4.42e-5 * T + (7.3637e-5 + 1.7950e-7 * T) * P
    Dblk = 1.727e-3 - 7.9836e-6 * P
    return Cw + A * S + B * S**1.5 + Dblk * S**2


# --- depth <-> pressure (Leroy & Parthiot 1998) --------------------------------------------------------------
def _gravity(lat: Any, ops=None) -> Any:
    """Latitude-dependent surface gravity (m/s^2), international gravity formula."""
    s2 = _sin(0.0174532925199433 * lat, ops) ** 2  # sin(lat in radians)^2
    return 9.780318 * (1.0 + 5.2788e-3 * s2 - 2.36e-5 * s2**2)


def _sin(x, ops):
    if ops is not None:
        try:
            import torch

            if torch.is_tensor(x):
                return ops.sin(x)
        except ImportError:
            pass
    import numpy as np

    return np.sin(x)


def depth_to_pressure(depth: Any, *, latitude: float = 45.0, ops=None) -> Any:
    """Gauge pressure (bar) at ocean ``depth`` (metres), Leroy & Parthiot (1998) for standard seawater.

    ``depth`` scalar or field (metres). ``latitude`` in degrees sets the gravity correction. Returns gauge
    pressure in bar (the argument :func:`unesco` expects), differentiable in ``depth``. The bridge that lets
    a Mackenzie depth field and a UNESCO pressure field describe the same water column.

    Check value: ``depth_to_pressure(1000, latitude=45)`` ~ 101.0 bar.
    """
    z = depth
    g = _gravity(latitude, ops)
    # standard-ocean pressure profile at 45 deg, then latitude/gravity scaling (pressure returned in MPa)
    h45 = 1.00818e-2 * z + 2.465e-8 * z**2 - 1.25e-13 * z**3 + 2.8e-19 * z**4
    k = (g - 2.0e-5 * z) / (9.80612 - 2.0e-5 * z)
    p_mpa = h45 * k
    return 10.0 * p_mpa  # MPa -> bar


def pressure_to_depth(pressure: Any, *, latitude: float = 45.0) -> Any:
    """Ocean depth (metres) from gauge ``pressure`` (bar), the inverse of :func:`depth_to_pressure`.

    ``pressure`` scalar or field (bar). Uses the UNESCO/Saunders standard-ocean inversion (Leroy & Parthiot
    1998); accurate to a metre or so over the ocean column. Round-trips :func:`depth_to_pressure` closely.
    """
    import numpy as np

    p_mpa = np.asarray(pressure, float) / 10.0  # bar -> MPa
    s2 = np.sin(np.radians(latitude)) ** 2
    g = 9.7803 * (1.0 + 5.3e-3 * s2)
    num = 9.72659e2 * p_mpa - 2.2512e-1 * p_mpa**2 + 2.279e-4 * p_mpa**3 - 1.82e-7 * p_mpa**4
    return num / (g + 1.092e-4 * p_mpa)
