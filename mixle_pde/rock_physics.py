"""Biot-Gassmann fluid substitution as closed-form differentiable rock-physics transforms.

Elastic full-waveform inversion recovers the seismic observables ``Vp, Vs, rho`` (compressional and
shear velocity, bulk density). Those are not the reservoir quantities an interpreter wants -- porosity,
mineralogy, and the pore fluid. Gassmann's equation is the bridge: it relates the saturated-rock bulk
modulus to the modulus of the dry rock frame, the mineral, and the pore fluid, under the low-frequency
assumption that pore pressure equilibrates across the sample. Because it is closed form (no PDE), the
whole map is a chain of elementwise transforms, so it is differentiable and can be composed after an
elastic-FWI forward to reparameterize ``(Vp, Vs, rho)`` into ``(phi, saturation)`` or to predict the
elastic response of a fluid change (brine -> gas, the "what if we produce it" question).

The forward saturated modulus is

    K_sat = K_dry + (1 - K_dry/K_min)^2 / ( phi/K_fl + (1 - phi)/K_min - K_dry/K_min^2 )

with the shear modulus fluid-independent, ``mu_sat = mu_dry``. Inverting for the dry frame is algebraic,

    K_dry = ( K_sat (phi K_min/K_fl + 1 - phi) - K_min )
            / ( phi K_min/K_fl + K_sat/K_min - 1 - phi )

so a fluid substitution is: measured moduli -> dry frame (inverse Gassmann with the in-situ fluid) ->
new saturated moduli (forward Gassmann with the target fluid), with the density updated for the fluid
mass swapped into the pore space. All math goes through the ``ops`` namespace (or plain array
arithmetic), so the transforms are backend-agnostic and differentiable end to end.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _sqrt(x, ops):
    """Backend-agnostic sqrt: the passed ``ops`` (differentiable, autograd tensors) or numpy for plain
    scalars/arrays. Keeps the transforms usable on both raw arrays and the ops-namespace tensor path."""
    if ops is not None:
        try:
            import torch

            if torch.is_tensor(x):
                return ops.sqrt(x)
        except ImportError:
            pass
    return np.sqrt(x)


def gassmann_ksat(K_dry: Any, K_min: Any, K_fl: Any, phi: Any, *, ops=None) -> Any:
    """Saturated-rock bulk modulus from the dry frame via Gassmann's equation.

    ``K_dry`` dry-frame bulk modulus, ``K_min`` mineral (solid grain) bulk modulus, ``K_fl`` pore-fluid
    bulk modulus, ``phi`` porosity (fraction). Scalars or fields; all in consistent units (e.g. GPa).
    Returns ``K_sat``. The shear modulus is unchanged by the fluid (``mu_sat = mu_dry``), so it is not a
    part of this transform.
    """
    del ops  # pure elementwise arithmetic; broadcast handles scalars, arrays, and autograd tensors
    dry_ratio = K_dry / K_min
    num = (1.0 - dry_ratio) ** 2
    den = phi / K_fl + (1.0 - phi) / K_min - K_dry / K_min**2
    return K_dry + num / den


def gassmann_kdry(K_sat: Any, K_min: Any, K_fl: Any, phi: Any, *, ops=None) -> Any:
    """Dry-frame bulk modulus from the saturated modulus (the exact algebraic inverse of Gassmann).

    Recovers ``K_dry`` given the fluid it was saturated with, so that a different fluid can be substituted
    into the same rock frame. Same arguments and units as :func:`gassmann_ksat`; it is its exact inverse
    (round trips to machine precision).
    """
    del ops
    a = K_sat * (phi * K_min / K_fl + 1.0 - phi) - K_min
    b = phi * K_min / K_fl + K_sat / K_min - 1.0 - phi
    return a / b


def moduli_from_velocity(Vp: Any, Vs: Any, rho: Any) -> tuple[Any, Any]:
    """Bulk and shear modulus from velocities and density: ``K = rho(Vp^2 - 4/3 Vs^2)``, ``mu = rho Vs^2``.

    Returns ``(K, mu)``. Units follow the inputs: with ``Vp, Vs`` in km/s and ``rho`` in g/cm^3 the moduli
    come out in GPa, the natural unit for Gassmann's mineral/fluid moduli.
    """
    K = rho * (Vp**2 - (4.0 / 3.0) * Vs**2)
    mu = rho * Vs**2
    return K, mu


def velocity_from_moduli(K: Any, mu: Any, rho: Any, *, ops=None) -> tuple[Any, Any]:
    """Velocities from moduli and density: ``Vp = sqrt((K + 4/3 mu)/rho)``, ``Vs = sqrt(mu/rho)``.

    Inverse of :func:`moduli_from_velocity`; returns ``(Vp, Vs)``. Uses ``ops.sqrt`` so it stays
    differentiable on autograd tensors (falls back to the numpy-style ``sqrt`` for plain arrays).
    """
    Vp = _sqrt((K + (4.0 / 3.0) * mu) / rho, ops)
    Vs = _sqrt(mu / rho, ops)
    return Vp, Vs


def fluid_substitute(
    Vp: Any,
    Vs: Any,
    rho: Any,
    *,
    phi: Any,
    K_min: Any,
    rho_min: Any,
    K_fl_in: Any,
    rho_fl_in: Any,
    K_fl_out: Any,
    rho_fl_out: Any,
    ops=None,
) -> tuple[Any, Any, Any]:
    """Predict ``(Vp, Vs, rho)`` after replacing the in-situ pore fluid with a new one (Gassmann).

    The full closed-form chain: velocities -> saturated moduli -> dry frame (inverse Gassmann with the
    in-situ fluid ``K_fl_in``) -> new saturated bulk modulus (forward Gassmann with the target fluid
    ``K_fl_out``) -> new density -> new velocities. The shear modulus is fluid-independent, so ``Vs`` moves
    only through the density change.

    Arguments (scalars or fields, consistent units -- e.g. velocities km/s, densities g/cm^3, moduli GPa):
      ``Vp, Vs, rho``      measured in-situ velocities and bulk density.
      ``phi``              porosity (fraction).
      ``K_min, rho_min``   mineral (solid grain) bulk modulus and density.
      ``K_fl_in, rho_fl_in``   bulk modulus and density of the fluid currently in the pores.
      ``K_fl_out, rho_fl_out`` bulk modulus and density of the fluid to substitute in.

    Returns ``(Vp_out, Vs_out, rho_out)``. Differentiable in every argument, so a saturation or porosity
    can be a latent driver behind an FWI observation. Round trips: substituting A->B then B->A recovers the
    inputs. The density update ``rho_out = rho + phi (rho_fl_out - rho_fl_in)`` swaps only the pore-fluid
    mass and leaves the frame (``(1 - phi) rho_min``) fixed, consistent with the Gassmann assumption that
    only the fluid changes.
    """
    K_sat_in, mu = moduli_from_velocity(Vp, Vs, rho)
    K_dry = gassmann_kdry(K_sat_in, K_min, K_fl_in, phi)
    K_sat_out = gassmann_ksat(K_dry, K_min, K_fl_out, phi)
    rho_out = rho + phi * (rho_fl_out - rho_fl_in)
    Vp_out, Vs_out = velocity_from_moduli(K_sat_out, mu, rho_out, ops=ops)
    return Vp_out, Vs_out, rho_out
