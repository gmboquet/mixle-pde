"""pysparkplug-pde — PDE/ODE-constrained Bayesian inverse problems, a mixle.ppl plugin.

Importing this package wires the PDE stack into pysparkplug's PPL surface through mixle's extension
hooks (it does not patch mixle): importing :mod:`mixle_pde.pde` fires its
``register_composite("PDEStateSpace", ..., fit_fn=pde_fit)`` so ``PDE(operator).fit(data)`` works, and
the dynamics operators register through ``register_dynamics_operator``. mixle core stays PDE-free; this
package depends on mixle, never the reverse.

    import mixle_pde as pde
    from mixle_pde import PDE, DiffusionOperator
    model = PDE(DiffusionOperator(0.1, n)).fit(field, dt=0.1)

The forward-model infrastructure (``_operator``, ``ops``, ``dynamics``, ``pde_solve``, ``inverse``,
``multiphysics``, ``pde``) and the concrete equations (``wave``/``wave_pml``, ``flow``/``spectral_flow``,
``gas_dynamics``, ``schrodinger``, ``fem``, ``shape``) are the modules; the headline solvers are
re-exported here.
"""

from __future__ import annotations

from typing import Any

from mixle.ppl.core import RandomVariable

# Register the sparse-solve detector so mixle.ppl.field can guard how='laplace' on a sparse PDE forward
# (its dense double-backward Hessian would be silently wrong). mixle has no PDE dependency; this plugs in.
from mixle.ppl.field import register_sparse_solve_detector as _register_sparse_solve_detector

# Importing `pde` fires register_composite("PDEStateSpace", ..., fit_fn=pde_fit) into mixle's registry.
from mixle_pde import pde  # noqa: F401

# realistic sonar/radar propagation (part 1): environmental transforms + the parabolic-equation keystone
from mixle_pde.attenuation import (
    francois_garrison_seawater,
    itu_gaseous,
    itu_rain_specific,
    quality_factor,
    thorp_seawater,
)
from mixle_pde.basin import easy_ro, easy_ro_profile, geotherm
from mixle_pde.beam import EulerBernoulliBeam

# realistic sonar/radar propagation (part 2): boundaries, KRAKEN normal modes, and the propagation inverse problems
from mixle_pde.boundaries import (
    bottom_loss_db,
    coherent_roughness_factor,
    critical_grazing_angle,
    radar_surface_reflection,
    seabed_reflection,
)
from mixle_pde.dynamics import (
    AdvectionDiffusionOperator,
    AdvectionOperator,
    DiffusionOperator,
    DynamicsOperator,
    available_dynamics_operators,
    make_operator,
    register_dynamics_operator,
)
from mixle_pde.elastic import ElasticWave3D

# new inverse-PDE families (wave 1): nonlinear-steady primitive, diffusive EM, transient heat, rock physics,
# migration, anisotropic elasticity, guided waves, poroelasticity
from mixle_pde.elastic_aniso import AnisotropicElasticWave3D, thomsen_to_cij

# new inverse-PDE families (wave 2): Poisson-Boltzmann, Poisson-Nernst-Planck, induced polarization,
# full-tensor potential fields, PML/complex-modulus Helmholtz, cycle-skip FWI misfits
from mixle_pde.electrostatics import linearized_pbe, nonlinear_pbe, reaction_field_energy
from mixle_pde.em_diffusion import layered_mt_impedance, mt_2d_te
from mixle_pde.em_diffusion_3d import assemble_curl_curl_3d, csem_3d, mt_3d

# realistic sonar/radar propagation (part 3): full-wave reference, asymptotic ray/PO scattering, real-data layer
from mixle_pde.env_data import (
    apply_mask,
    assemble_field,
    load_dem,
    load_era5_profile,
    load_gebco,
    load_woa_argo,
    seabed_mask,
)
from mixle_pde.flow import NavierStokes2D
from mixle_pde.flow3d import NavierStokes3D
from mixle_pde.geophysics import (
    cross_gradient,
    dc_resistivity,
    depth_weighting,
    gravity_point_sensitivity,
    joint_inversion,
    magnetic_dipole_sensitivity,
    regularized_gauss_newton,
    roughness_operator,
    straight_ray_operator,
)
from mixle_pde.guided_wave import SAFEPlate, safe_dispersion
from mixle_pde.heat import TransientHeat
from mixle_pde.helmholtz_pml import helmholtz_pml_operator, solve_helmholtz_pml
from mixle_pde.induced_polarization import apparent_conductivity, cole_cole_conductivity, sip_forward
from mixle_pde.inverse import Differential
from mixle_pde.maxwell import Maxwell3D
from mixle_pde.migration import born_modeling, lsrtm_step, rtm_image
from mixle_pde.misfit import envelope_misfit, hilbert_envelope, misfit, wasserstein1d_misfit, xcorr_traveltime_misfit
from mixle_pde.multiphysics import CoupledPDESystem, solve_poisson
from mixle_pde.nonlinear import nonlinear_solve, reaction_diffusion_residual
from mixle_pde.normal_modes import NormalModes1D
from mixle_pde.parabolic_equation import ParabolicEquation2D
from mixle_pde.pde_solve import sparse_used_since as _sparse_used_since
from mixle_pde.plate import KirchhoffPlate
from mixle_pde.pnp import debye_length, pnp_equilibrium
from mixle_pde.poroelastic import BiotPoroelastic1D, biot_gassmann_velocity
from mixle_pde.potential_fields import (
    gravity_gradient_tensor,
    magnetic_gradient_tensor,
    magnetic_vector_sensitivity,
)
from mixle_pde.propagation_inverse import ocean_sound_speed_inversion, refractivity_from_clutter
from mixle_pde.ray_scattering import knife_edge_diffraction, multipath_power, po_rcs, two_ray_pattern

# cross-modal subsurface reasoning: geophysical forward models -> mixle.reason evidence (belief + UQ)
from mixle_pde.reasoning import JointPotentialField, MechanisticFieldReasoner, SpatialFieldStore
from mixle_pde.refractivity import duct_layers, modified_refractivity, standard_refractivity_profile
from mixle_pde.rock_physics import fluid_substitute, gassmann_kdry, gassmann_ksat
from mixle_pde.shape import level_set_material, shape_optimize

# new inverse-PDE families (wave 3): Smoluchowski diffusion-limited on-rates, time-domain constant-Q viscoacoustic
from mixle_pde.smoluchowski import smoluchowski_debye_factor, smoluchowski_rate_box, smoluchowski_rate_radial
from mixle_pde.sound_speed import mackenzie, unesco
from mixle_pde.two_phase import TwoPhaseFlow2D
from mixle_pde.viscoelastic import ViscoacousticWave1D, q_of_omega, tau_fit
from mixle_pde.wave import WaveEquation2D
from mixle_pde.wave3d import WaveEquation3D
from mixle_pde.wavenumber_integration import WavenumberIntegration1D

_register_sparse_solve_detector(_sparse_used_since)


def PDE(operator: Any, *, name: str | None = None) -> RandomVariable:
    """PDE-constrained latent-field model for spatiotemporal data.

    ``operator`` is a :class:`~mixle_pde.dynamics.DynamicsOperator` (e.g. ``DiffusionOperator``,
    ``AdvectionOperator``) whose method-of-lines discretization fixes the linear state transition. Fit
    on a ``(T, m)`` array of noisy field observations: the Kalman/RTS smoother recovers the latent field
    and EM estimates the process/observation noise while the physics-derived dynamics are held fixed.
    Lowers to the ``PDEStateSpace`` family registered (with its fit_fn) when this package is imported.
    """
    return RandomVariable._sample("PDEStateSpace", (operator,), name=name)


__all__ = [
    "PDE",
    "DiffusionOperator",
    "AdvectionOperator",
    "AdvectionDiffusionOperator",
    "DynamicsOperator",
    "make_operator",
    "register_dynamics_operator",
    "available_dynamics_operators",
    "NavierStokes2D",
    "WaveEquation2D",
    "WaveEquation3D",
    "NavierStokes3D",
    "TwoPhaseFlow2D",
    "Maxwell3D",
    "ElasticWave3D",
    "EulerBernoulliBeam",
    "KirchhoffPlate",
    "Differential",
    "CoupledPDESystem",
    "solve_poisson",
    "gravity_point_sensitivity",
    "magnetic_dipole_sensitivity",
    "depth_weighting",
    "cross_gradient",
    "dc_resistivity",
    "joint_inversion",
    "regularized_gauss_newton",
    "roughness_operator",
    "straight_ray_operator",
    "JointPotentialField",
    "SpatialFieldStore",
    "MechanisticFieldReasoner",
    "shape_optimize",
    "level_set_material",
    "geotherm",
    "easy_ro",
    "easy_ro_profile",
    "nonlinear_solve",
    "reaction_diffusion_residual",
    "layered_mt_impedance",
    "mt_2d_te",
    "mt_3d",
    "csem_3d",
    "assemble_curl_curl_3d",
    "TransientHeat",
    "gassmann_ksat",
    "gassmann_kdry",
    "fluid_substitute",
    "rtm_image",
    "born_modeling",
    "lsrtm_step",
    "AnisotropicElasticWave3D",
    "thomsen_to_cij",
    "safe_dispersion",
    "SAFEPlate",
    "BiotPoroelastic1D",
    "biot_gassmann_velocity",
    "linearized_pbe",
    "nonlinear_pbe",
    "reaction_field_energy",
    "pnp_equilibrium",
    "debye_length",
    "cole_cole_conductivity",
    "sip_forward",
    "apparent_conductivity",
    "gravity_gradient_tensor",
    "magnetic_vector_sensitivity",
    "magnetic_gradient_tensor",
    "helmholtz_pml_operator",
    "solve_helmholtz_pml",
    "misfit",
    "envelope_misfit",
    "xcorr_traveltime_misfit",
    "wasserstein1d_misfit",
    "hilbert_envelope",
    "smoluchowski_rate_radial",
    "smoluchowski_rate_box",
    "smoluchowski_debye_factor",
    "ViscoacousticWave1D",
    "tau_fit",
    "q_of_omega",
    "mackenzie",
    "unesco",
    "modified_refractivity",
    "standard_refractivity_profile",
    "duct_layers",
    "thorp_seawater",
    "francois_garrison_seawater",
    "itu_gaseous",
    "itu_rain_specific",
    "quality_factor",
    "ParabolicEquation2D",
    "WavenumberIntegration1D",
    "po_rcs",
    "knife_edge_diffraction",
    "two_ray_pattern",
    "multipath_power",
    "assemble_field",
    "seabed_mask",
    "apply_mask",
    "load_gebco",
    "load_woa_argo",
    "load_dem",
    "load_era5_profile",
    "seabed_reflection",
    "critical_grazing_angle",
    "bottom_loss_db",
    "coherent_roughness_factor",
    "radar_surface_reflection",
    "NormalModes1D",
    "refractivity_from_clutter",
    "ocean_sound_speed_inversion",
]
