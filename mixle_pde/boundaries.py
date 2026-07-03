"""Closed-form differentiable boundary / interaction models that make the PE a real waveguide.

A parabolic-equation propagator is only a channel once its boundaries are physics. An ocean-acoustic PE
is a waveguide bounded above by the sea surface and below by the seabed; a tropospheric radar PE reflects
off the sea/land surface. This module supplies the elementwise, torch-differentiable reflection models the
marcher applies at those boundaries, in the mould of :mod:`mixle_pde.rock_physics`.

Seabed (fluid-fluid). A plane wave in the water (sound speed ``c1``, density ``rho1``) striking a fluid
half-space seabed (``c2``, ``rho2``, optional volume attenuation) reflects with the Rayleigh coefficient

    R(theta) = (Z2 cos theta_i - Z1 cos theta_t) / (Z2 cos theta_i + Z1 cos theta_t)

where ``theta_i`` is the grazing angle, ``theta_t`` the transmitted grazing angle from Snell's law
``cos theta_t = (c2/c1) cos theta_i``, and ``Z = rho c`` is the plane-wave impedance. For a faster seabed
(``c2 > c1``) there is a critical grazing angle ``theta_c = arccos(c1/c2)``: below it ``cos theta_t`` turns
imaginary, the wave is evanescent in the bottom, and ``|R| = 1`` (total internal reflection). The bottom
loss in dB is ``-20 log10 |R|``.

Sea surface. The air/water contrast makes the surface a near-perfect pressure-release boundary, ``R = -1``.
A rough surface scatters energy out of the coherent (specular) field; the coherent reflection coefficient
is reduced by the Rayleigh roughness factor ``rho = exp(-2 (k sigma sin theta)^2)`` (Eckart), with the
optional Miller-Brown-Vegh correction that restores the incoherent floor at large roughness.

Radar surface. A grazing radar reflects off the sea/land with the Fresnel coefficients set by the surface
complex relative permittivity ``eps_r`` for horizontal or vertical polarisation.

All math goes through the ``ops`` namespace (or plain array / numpy arithmetic), so every transform is
backend-agnostic and differentiable end to end. The seabed coefficient uses complex arithmetic below the
critical angle, so gradients flow through the evanescent branch too.

References: Brekhovskikh & Lysanov, *Fundamentals of Ocean Acoustics*; Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics* (2011); Eckart (1953) *JASA*; Miller-Brown-Vegh coherent-loss model;
Fresnel equations for a lossy dielectric half-space.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "impedance",
    "critical_grazing_angle",
    "seabed_reflection",
    "bottom_loss_db",
    "surface_reflection",
    "rayleigh_roughness",
    "coherent_roughness_factor",
    "radar_surface_reflection",
]


def _is_torch(x) -> bool:
    try:
        import torch

        return torch.is_tensor(x)
    except ImportError:
        return False


def _sqrt(x, ops):
    """Backend-agnostic sqrt: ``ops.sqrt`` on autograd tensors, else numpy (complex-safe)."""
    if ops is not None and _is_torch(x):
        return ops.sqrt(x)
    return np.sqrt(x)


def impedance(rho: Any, c: Any) -> Any:
    """Plane-wave (characteristic) acoustic impedance ``Z = rho c``.

    ``rho`` medium density, ``c`` sound speed, in consistent units (e.g. kg/m^3 and m/s give Z in Rayl).
    Elementwise and differentiable.
    """
    return rho * c


def critical_grazing_angle(c1: Any, c2: Any) -> Any:
    """Critical grazing angle ``theta_c = arccos(c1/c2)`` (radians) for a faster lower medium.

    ``c1`` water sound speed, ``c2`` seabed sound speed. Defined only for ``c2 > c1`` (a faster bottom);
    for grazing angles below ``theta_c`` the seabed reflection is total (``|R| = 1``). For ``c2 <= c1`` the
    ratio ``c1/c2 >= 1`` and there is no critical angle (``arccos`` clips to 0). Scalars or fields.
    """
    ratio = c1 / c2
    if _is_torch(ratio):
        import torch

        return torch.arccos(torch.clamp(ratio, max=1.0))
    return np.arccos(np.clip(ratio, None, 1.0))


def seabed_reflection(
    grazing_angle: Any,
    c1: Any,
    rho1: Any,
    c2: Any,
    rho2: Any,
    *,
    attenuation_db_per_wavelength: Any = 0.0,
    ops=None,
) -> Any:
    """Rayleigh plane-wave reflection coefficient ``R(theta)`` at a fluid-fluid water/seabed interface.

    ``grazing_angle`` the grazing angle ``theta_i`` (radians, measured from the interface); ``c1, rho1`` the
    water sound speed and density; ``c2, rho2`` the seabed sound speed and density. An optional volume
    attenuation ``attenuation_db_per_wavelength`` (dB per wavelength in the bottom) makes ``c2`` complex,
    ``c2 -> c2 / (1 + i delta)`` with ``delta = (alpha ln10)/(40 pi)``, so ``|R| < 1`` even below the
    critical angle (a lossy bottom).

    The transmitted grazing angle follows Snell's law ``cos theta_t = (c2/c1) cos theta_i``; below the
    critical angle its cosine is imaginary (evanescent), handled with complex arithmetic so gradients still
    flow. Returns ``R = (Z2 cos theta_i - Z1 cos theta_t) / (Z2 cos theta_i + Z1 cos theta_t)`` with
    ``Z = rho c``. Complex in general; ``|R|`` is the reflected amplitude fraction. Differentiable in the
    seabed parameters ``c2, rho2`` (and the water parameters).
    """
    # A lossy bottom folds the attenuation into a complex sound speed (small-loss approximation).
    delta = attenuation_db_per_wavelength * np.log(10.0) / (40.0 * np.pi)
    lossy = not (np.isscalar(delta) and float(np.asarray(delta)) == 0.0)

    torchy = any(_is_torch(v) for v in (grazing_angle, c1, rho1, c2, rho2, attenuation_db_per_wavelength))

    if torchy:
        import torch

        cos_i = torch.cos(torch.as_tensor(grazing_angle, dtype=torch.float64)).to(torch.complex128)
        c1c = torch.as_tensor(c1, dtype=torch.complex128)
        c2c = torch.as_tensor(c2, dtype=torch.complex128)
        if lossy:
            c2c = c2c / (1.0 + 1j * torch.as_tensor(delta, dtype=torch.complex128))
        rho1c = torch.as_tensor(rho1, dtype=torch.complex128)
        rho2c = torch.as_tensor(rho2, dtype=torch.complex128)
        sqrt = ops.sqrt if ops is not None else torch.sqrt
    else:
        cos_i = np.asarray(np.cos(grazing_angle), dtype=np.complex128)
        c1c = np.asarray(c1, dtype=np.complex128)
        c2c = np.asarray(c2, dtype=np.complex128)
        if lossy:
            c2c = c2c / (1.0 + 1j * np.asarray(delta, dtype=np.complex128))
        rho1c = np.asarray(rho1, dtype=np.complex128)
        rho2c = np.asarray(rho2, dtype=np.complex128)
        sqrt = np.sqrt
        del ops

    # Snell: cos theta_t = (c2/c1) cos theta_i; sin theta_t via the branch-consistent complex sqrt.
    cos_t_ratio = (c2c / c1c) * cos_i
    sin_t = sqrt(1.0 - cos_t_ratio * cos_t_ratio)  # = cos(grazing_t) complement; the vertical wavenumber factor

    z1 = rho1c * c1c
    z2 = rho2c * c2c
    # Grazing-angle Rayleigh form: numerator uses sin theta_i (= vertical component) on each side.
    sin_i = sqrt(1.0 - cos_i * cos_i)
    num = z2 * sin_i - z1 * sin_t
    den = z2 * sin_i + z1 * sin_t
    return num / den


def bottom_loss_db(reflection: Any, *, ops=None) -> Any:
    """Bottom loss in dB from a (possibly complex) reflection coefficient: ``BL = -20 log10 |R|``.

    ``reflection`` the coefficient from :func:`seabed_reflection`. Zero for total reflection (``|R| = 1``),
    positive for a lossy / sub-unity reflection. Differentiable via ``ops``.
    """
    if ops is not None and _is_torch(reflection):
        mag = ops.abs(reflection)
        return -20.0 * ops.log(mag) / np.log(10.0)
    mag = np.abs(reflection)
    return -20.0 * np.log10(mag)


def surface_reflection() -> float:
    """Sea-surface (pressure-release) reflection coefficient ``R = -1``.

    The air/water impedance contrast is so large that a downward-propagating acoustic wave reflects almost
    perfectly with a sign flip (a pressure node at the surface). A constant, the ideal specular value before
    any roughness reduction from :func:`coherent_roughness_factor`.
    """
    return -1.0


def rayleigh_roughness(k: Any, sigma: Any, grazing_angle: Any, *, ops=None) -> Any:
    """Rayleigh roughness parameter ``g = k sigma sin theta`` (the phase spread across the rough surface).

    ``k`` acoustic wavenumber ``2 pi f / c``, ``sigma`` the rms surface height, ``grazing_angle`` theta
    (radians). Roughness scatters energy out of the coherent field once ``g`` approaches 1. Differentiable.
    """
    if ops is not None and any(_is_torch(v) for v in (k, sigma, grazing_angle)):
        import torch

        return k * sigma * ops.sin(torch.as_tensor(grazing_angle, dtype=torch.float64))
    return k * sigma * np.sin(grazing_angle)


def coherent_roughness_factor(
    k: Any,
    sigma: Any,
    grazing_angle: Any,
    *,
    miller_brown_vegh: bool = False,
    ops=None,
) -> Any:
    """Coherent (specular) reflection reduction from surface roughness -- the Eckart factor.

    ``rho = exp(-2 (k sigma sin theta)^2)`` reduces the coherent reflection coefficient (Eckart); at
    ``sigma -> 0`` it is 1 (smooth surface, full specular return) and it falls monotonically toward 0 as the
    Rayleigh parameter ``g = k sigma sin theta`` grows. The full coherent coefficient off the sea surface is
    ``R_coh = surface_reflection() * rho = -rho``.

    With ``miller_brown_vegh=True`` the Miller-Brown-Vegh correction ``rho -> rho * I0(2 g^2)`` (via the
    modified Bessel ``I0``) restores the incoherent floor at large roughness, where the plain Eckart factor
    under-predicts the returned power. Differentiable in ``k, sigma, theta``.
    """
    g = rayleigh_roughness(k, sigma, grazing_angle, ops=ops)
    g2 = g * g
    if ops is not None and _is_torch(g):
        rho = ops.exp(-2.0 * g2)
        if miller_brown_vegh:
            import torch

            rho = rho * torch.special.i0(2.0 * g2)
        return rho
    rho = np.exp(-2.0 * g2)
    if miller_brown_vegh:
        from scipy.special import i0

        rho = rho * i0(2.0 * np.asarray(g2))
    return rho


def radar_surface_reflection(
    grazing_angle: Any,
    eps_r: Any,
    *,
    polarisation: str = "horizontal",
    ops=None,
) -> Any:
    """Fresnel surface-impedance reflection coefficient of a lossy half-space for grazing radar.

    ``grazing_angle`` the grazing angle psi (radians, from the surface); ``eps_r`` the complex relative
    permittivity of the surface (sea or land), ``eps_r = eps' - i 60 lambda sigma`` for a conducting medium.
    ``polarisation`` is ``"horizontal"`` (E parallel to the surface) or ``"vertical"``.

    With ``s = sin psi`` and ``w = sqrt(eps_r - cos^2 psi)`` the Fresnel coefficients are

        R_h = (s - w) / (s + w),   R_v = (eps_r s - w) / (eps_r s + w).

    At grazing incidence (psi -> 0) both tend to ``-1`` (pressure-release-like), the pseudo-Brewster dip in
    ``|R_v|`` sitting just above. Returns the complex coefficient; ``|R|`` is the reflected amplitude
    fraction. Differentiable in ``eps_r`` and the angle.
    """
    torchy = any(_is_torch(v) for v in (grazing_angle, eps_r))
    if torchy:
        import torch

        psi = torch.as_tensor(grazing_angle, dtype=torch.float64)
        s = torch.sin(psi).to(torch.complex128)
        cos2 = (torch.cos(psi).to(torch.complex128)) ** 2
        eps = torch.as_tensor(eps_r, dtype=torch.complex128)
        sqrt = ops.sqrt if ops is not None else torch.sqrt
    else:
        s = np.asarray(np.sin(grazing_angle), dtype=np.complex128)
        cos2 = np.asarray(np.cos(grazing_angle), dtype=np.complex128) ** 2
        eps = np.asarray(eps_r, dtype=np.complex128)
        sqrt = np.sqrt
        del ops

    w = sqrt(eps - cos2)
    pol = polarisation.lower()
    if pol in ("horizontal", "h"):
        return (s - w) / (s + w)
    if pol in ("vertical", "v"):
        return (eps * s - w) / (eps * s + w)
    raise ValueError(f"polarisation must be 'horizontal' or 'vertical', got {polarisation!r}")
