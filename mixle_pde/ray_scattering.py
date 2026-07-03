"""High-frequency (asymptotic) EM/acoustic scattering: PO radar cross section, ray multipath, edge diffraction.

Where the full-wave Helmholtz solvers (:mod:`mixle_pde.helmholtz_pml`) resolve every wavelength on a grid, an
urban/target scene at radar frequencies is thousands of wavelengths across and only the *asymptotic* (ray)
limit is tractable. This module is that realism layer: it treats each smooth facet as a locally flat mirror
(physical optics), traces a small fixed set of direct / single-bounce / double-bounce rays through a
ground-plus-building scene (image method), and handles the one place geometry fails -- the shadow behind an
obstruction -- with a scalar knife-edge diffraction coefficient.

Three pieces:

* :func:`po_rcs` -- physical-optics / geometric-optics monostatic radar cross section (m^2) of the canonical
  calibration shapes: a flat plate, a dihedral, a triangular trihedral corner reflector, and a sphere. At
  normal incidence these reduce to the textbook closed forms (see the coefficient table in the function).
* :func:`two_ray_pattern` and :func:`multipath_power` -- a coherent sum of direct + ground-reflected (+ optional
  building single/double-bounce) paths giving received power versus receiver position. The classic two-ray
  ground-reflection lobing (height-gain nulls where direct and reflected paths are ``pi`` out of phase) falls
  straight out.
* :func:`knife_edge_diffraction` -- the Fresnel-integral single-knife-edge loss versus the diffraction
  parameter ``v``: ~6 dB right at grazing (``v = 0``) rising through the deep-shadow asymptote.

Differentiability caveat: unlike the PE / Helmholtz forwards, this is only *partially* differentiable. The PO
amplitudes and the diffraction loss are smooth in geometry, but ray topology (which facets are lit, which
bounce paths exist, whether a receiver is shadowed) is piecewise constant, so the map has jumps and does NOT
drop into the differentiable inverse spine (:class:`mixle_pde.inverse.Differential`). It is a numpy forward for
scene realism and calibration, not for gradient-based inversion. Everything is plain :mod:`numpy`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "wavelength",
    "po_rcs",
    "fresnel_integral",
    "knife_edge_diffraction",
    "two_ray_pattern",
    "multipath_power",
]

# speed of light (m/s); the default carrier medium for radar RCS
_C_LIGHT = 299_792_458.0


def wavelength(frequency: float, *, speed: float = _C_LIGHT) -> float:
    """Wavelength ``lambda = speed / frequency`` (m). ``speed`` defaults to the speed of light."""
    return float(speed) / float(frequency)


def po_rcs(shape: str, lam: float, *, incidence: float = 0.0, **dims: float) -> float:
    """Monostatic radar cross section (m^2) of a canonical shape in the physical/geometric-optics limit.

    High-frequency backscatter of a perfectly conducting calibration target. Each result is the standard
    asymptotic closed form; the coefficient conventions are:

    ======================  =========================================  ================================
    ``shape``               peak (normal-incidence) RCS                required ``dims``
    ======================  =========================================  ================================
    ``"plate"``             ``4 pi A^2 / lambda^2``, ``A = a b``       ``a``, ``b`` (rectangular side, m)
    ``"dihedral"``          ``8 pi a^2 b^2 / lambda^2``                ``a``, ``b`` (plate sides, m)
    ``"trihedral"``         ``4 pi a^4 / (3 lambda^2)``                ``a`` (triangular edge, m)
    ``"sphere"``            ``pi a^2``                                 ``a`` (radius, m)
    ======================  =========================================  ================================

    The trihedral coefficient ``4 pi a^4 / (3 lambda^2)`` is the *triangular* corner reflector (three right
    isoceles triangles of leg ``a``); the square-plate trihedral would instead be ``12 pi a^4 / lambda^2``.
    The sphere is the optical (geometric) cross section ``pi a^2``, valid for ``a >> lambda`` (above the Mie
    ripple); it is frequency-independent, so ``lam`` is ignored for a sphere.

    For the plate a physical-optics angular taper is applied: ``sigma(theta) = sigma_0 cos^2(theta)
    sinc^2((2 a / lambda) sin theta)`` with ``theta`` the incidence off broadside (``sinc(x) = sin(pi x)/(pi
    x)``), so ``incidence = 0`` recovers the peak. The dihedral gets the leading ``cos^2`` roll-off of its
    doubly-reflecting response; the trihedral and sphere return their peak (their broad flat lobe / isotropy
    makes a single scalar the useful value).

    Args:
        shape: one of ``"plate"``, ``"dihedral"``, ``"trihedral"``, ``"sphere"``.
        lam: wavelength (m).
        incidence: incidence angle off the boresight/symmetry axis (rad); default normal incidence.
        **dims: shape dimensions (see table): ``a``/``b`` sides or ``a`` radius, in metres.

    Returns:
        the monostatic RCS in m^2.
    """
    lam = float(lam)
    theta = float(incidence)
    shape = shape.lower()

    if shape == "sphere":
        a = float(dims["a"])
        return float(np.pi * a * a)

    if shape == "plate":
        a = float(dims["a"])
        b = float(dims["b"])
        area = a * b
        sigma0 = 4.0 * np.pi * area * area / (lam * lam)
        # PO taper along the a-dimension; sinc(x) = sin(pi x)/(pi x)
        u = (2.0 * a / lam) * np.sin(theta)
        taper = np.cos(theta) ** 2 * _sinc(u) ** 2
        return float(sigma0 * taper)

    if shape == "dihedral":
        a = float(dims["a"])
        b = float(dims["b"])
        sigma0 = 8.0 * np.pi * a * a * b * b / (lam * lam)
        return float(sigma0 * np.cos(theta) ** 2)

    if shape == "trihedral":
        a = float(dims["a"])
        return float(4.0 * np.pi * a**4 / (3.0 * lam * lam))

    raise ValueError(f"unknown shape {shape!r}; expected plate/dihedral/trihedral/sphere.")


def _sinc(x: ArrayLike) -> np.ndarray:
    """Normalized sinc ``sin(pi x) / (pi x)`` (``= 1`` at ``x = 0``), matching :func:`numpy.sinc`."""
    return np.sinc(np.asarray(x, dtype=float))


def fresnel_integral(v: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The Fresnel cosine/sine integrals ``C(v) = int_0^v cos(pi t^2 / 2) dt`` and ``S(v)`` (sine).

    Evaluated by direct quadrature (Simpson) so the knife-edge loss can be asserted against these values
    without depending on :mod:`scipy`. As ``v -> +inf`` both approach ``1/2``.

    Args:
        v: upper limit(s) of the integral (scalar or array).

    Returns:
        ``(C, S)`` arrays matching the shape of ``v``.
    """
    v = np.asarray(v, dtype=float)
    scalar = v.ndim == 0
    vv = np.atleast_1d(v)
    n = 2000  # even -> Simpson
    out_c = np.empty_like(vv)
    out_s = np.empty_like(vv)
    for i, top in enumerate(vv):
        t = np.linspace(0.0, top, n + 1)
        arg = 0.5 * np.pi * t * t
        out_c[i] = _simpson(np.cos(arg), t)
        out_s[i] = _simpson(np.sin(arg), t)
    if scalar:
        return out_c[0], out_s[0]
    return out_c, out_s


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    """Composite Simpson quadrature of ``y`` over a uniform grid ``x`` (even number of intervals)."""
    if x[-1] == x[0]:
        return 0.0
    h = (x[-1] - x[0]) / (len(x) - 1)
    s = y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()
    return float(h * s / 3.0)


def knife_edge_diffraction(v: ArrayLike) -> np.ndarray:
    """Single knife-edge diffraction loss (dB, negative = loss) versus the Fresnel parameter ``v``.

    The scalar-field amplitude behind a half-plane edge is ``F(v) = (1 + j) / 2 * [ (1/2 - C(v)) - j (1/2 -
    S(v)) ]`` with ``C, S`` the Fresnel integrals (:func:`fresnel_integral`); the loss relative to free space
    is ``L(v) = 20 log10 |F(v)|``. The diffraction parameter is the usual
    ``v = h sqrt(2 (d1 + d2) / (lambda d1 d2))`` (``h`` the edge intrusion above the line-of-sight, positive
    into shadow). Reference values: ``L(0) ~ -6 dB`` (edge exactly on the line of sight) and
    ``L(1) ~ -14 to -16 dB``; deep in the shadow ``L(v) ~ 20 log10(0.225 / v)``.

    Args:
        v: the dimensionless diffraction parameter (scalar or array); ``v > 0`` is into the shadow.

    Returns:
        the diffraction loss in dB (``<= 0``), matching the shape of ``v``.
    """
    c, s = fresnel_integral(v)
    # F(v) = (1+j)/2 * [(1/2 - C) - j(1/2 - S)]
    f = 0.5 * (1.0 + 1j) * ((0.5 - c) - 1j * (0.5 - s))
    amp = np.abs(f)
    return 20.0 * np.log10(np.clip(amp, 1e-30, None))


def two_ray_pattern(
    ht: float,
    hr: ArrayLike,
    d: float,
    lam: float,
    *,
    reflection: complex = -1.0,
) -> np.ndarray:
    """Received power (linear, relative) of the two-ray ground-reflection model versus receiver height.

    A transmitter at height ``ht`` and a receiver at height ``hr`` separated by horizontal range ``d`` over a
    flat ground plane. The receiver sees a direct ray of length ``r_d = sqrt(d^2 + (ht - hr)^2)`` and a
    ground-reflected ray of length ``r_r = sqrt(d^2 + (ht + hr)^2)`` carrying the ground reflection coefficient
    ``Gamma`` (``-1`` for a perfect conductor at grazing). Summing the two complex ``exp(-j k r)/r`` amplitudes,

        P(hr) = | e^{-j k r_d}/r_d + Gamma e^{-j k r_r}/r_r |^2.

    Nulls occur where the two paths are ``pi`` out of phase. For ``d >> ht, hr`` the path difference is
    ``r_r - r_d ~ 2 ht hr / d``, so with ``Gamma = -1`` the pattern ``~ sin^2(2 pi ht hr / (lambda d))`` has
    nulls at ``hr = m lambda d / (2 ht)`` and lobe maxima halfway between.

    Args:
        ht: transmitter height (m).
        hr: receiver height(s) (m), scalar or array.
        d: horizontal range (m).
        lam: wavelength (m).
        reflection: ground reflection coefficient ``Gamma`` (complex); ``-1`` = perfect grazing conductor.

    Returns:
        received power (linear, in units where a single unit-amplitude direct ray at range 1 has power 1),
        matching the shape of ``hr``.
    """
    hr = np.asarray(hr, dtype=float)
    k = 2.0 * np.pi / float(lam)
    r_d = np.sqrt(d * d + (ht - hr) ** 2)
    r_r = np.sqrt(d * d + (ht + hr) ** 2)
    field = np.exp(-1j * k * r_d) / r_d + reflection * np.exp(-1j * k * r_r) / r_r
    return np.abs(field) ** 2


def multipath_power(
    tx: ArrayLike,
    rx: ArrayLike,
    lam: float,
    *,
    reflection: complex = -1.0,
    building: dict | None = None,
) -> float:
    """Coherent multipath received power for a 2-D ground + optional box-building scene (image method).

    A transmitter ``tx = (x, z)`` and receiver ``rx = (x, z)`` in the vertical plane above a flat ground at
    ``z = 0``. Paths summed coherently as ``sum_p Gamma_p exp(-j k L_p) / L_p``:

    * the direct line-of-sight path;
    * the single ground bounce (image of the source in ``z = 0``, reflection ``Gamma``);
    * if a ``building`` wall is given, the single bounce off that vertical wall and the ground->wall
      double bounce (image method, each bounce carrying ``Gamma``).

    ``building`` is ``{"x": wall_x, "zmax": height, "gamma": wall_reflection}`` describing a vertical wall face
    at ``x = wall_x`` extending from the ground to ``zmax``; ``gamma`` defaults to ``reflection``. Rays whose
    reflection point falls off the finite wall are dropped (this drop is the discontinuous ray-topology this
    module documents as non-differentiable).

    Args:
        tx: transmitter position ``(x, z)`` (m).
        rx: receiver position ``(x, z)`` (m).
        lam: wavelength (m).
        reflection: ground reflection coefficient ``Gamma``.
        building: optional vertical-wall descriptor (see above); ``None`` for ground-only.

    Returns:
        received power (linear, relative), the squared magnitude of the coherent path sum.
    """
    tx = np.asarray(tx, dtype=float)
    rx = np.asarray(rx, dtype=float)
    k = 2.0 * np.pi / float(lam)

    def leg(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.hypot(q[0] - p[0], q[1] - p[1]))

    field = 0.0 + 0.0j

    # direct
    ld = leg(tx, rx)
    field += np.exp(-1j * k * ld) / ld

    # ground single bounce: image of tx in z = 0
    tx_g = np.array([tx[0], -tx[1]])
    lg = leg(tx_g, rx)
    field += reflection * np.exp(-1j * k * lg) / lg

    if building is not None:
        wx = float(building["x"])
        zmax = float(building["zmax"])
        gamma_w = complex(building.get("gamma", reflection))

        # single wall bounce: image of tx in the vertical plane x = wx
        tx_w = np.array([2.0 * wx - tx[0], tx[1]])
        # reflection point z where the straight image->rx line crosses x = wx
        if abs(rx[0] - tx_w[0]) > 1e-12:
            frac = (wx - tx_w[0]) / (rx[0] - tx_w[0])
            z_hit = tx_w[1] + frac * (rx[1] - tx_w[1])
            if 0.0 <= frac <= 1.0 and 0.0 <= z_hit <= zmax:
                lw = leg(tx_w, rx)
                field += gamma_w * np.exp(-1j * k * lw) / lw

        # ground -> wall double bounce: image in z=0 then in x=wx
        tx_gw = np.array([2.0 * wx - tx[0], -tx[1]])
        if abs(rx[0] - tx_gw[0]) > 1e-12:
            frac = (wx - tx_gw[0]) / (rx[0] - tx_gw[0])
            z_hit = tx_gw[1] + frac * (rx[1] - tx_gw[1])
            if 0.0 <= frac <= 1.0 and 0.0 <= z_hit <= zmax:
                lgw = leg(tx_gw, rx)
                field += reflection * gamma_w * np.exp(-1j * k * lgw) / lgw

    return float(np.abs(field) ** 2)
