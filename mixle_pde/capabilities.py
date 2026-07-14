"""PDE modeling capability catalog and deterministic readiness checks.

The catalog is intentionally concrete: each required capability points at importable symbols and one or
more cheap scenarios that exercise the implementation. The Earth/subsurface checks use the modular
workstream-G surface (`latent`, `observations`, `field_inversion`, `field_assimilation`,
`field_priors`, `geo_observations`, and `posterior_query`) rather than a separate Earth-specific facade.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelingCapability:
    """A solver or inverse-modeling family Mixle can expose to applications."""

    id: str
    name: str
    category: str
    description: str
    equations: tuple[str, ...]
    solver_symbols: tuple[str, ...]
    required_dependencies: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    differentiable: bool = False
    inverse_ready: bool = False
    scenario_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def dependency_status(self) -> dict[str, bool]:
        """Return availability for required and optional import packages."""
        return {
            dep: importlib.util.find_spec(dep) is not None
            for dep in (*self.required_dependencies, *self.optional_dependencies)
        }

    @property
    def available(self) -> bool:
        """Whether every required dependency is importable in this environment."""
        status = self.dependency_status()
        return all(status.get(dep, False) for dep in self.required_dependencies)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["available"] = self.available
        data["dependencies"] = self.dependency_status()
        return data


@dataclass(frozen=True)
class ScenarioResult:
    """Result of a modeling readiness scenario."""

    id: str
    capability_id: str
    passed: bool
    metrics: dict[str, float]
    tolerance: dict[str, float]
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationScenario:
    """A deterministic check that proves a capability is runnable."""

    id: str
    capability_id: str
    name: str
    description: str
    runner: Callable[[], ScenarioResult]


DEFAULT_REQUIRED_CAPABILITIES = (
    "mesh.simplicial_3d_4d",
    "earth.geochem_biostrat_likelihoods",
    "pde.state_space",
    "pde.transient_heat",
    "gas.reactive_combustion",
    "geophysics.potential_fields",
    "reasoning.mechanistic_field",
    "wave.acoustic_2d",
)


def capability_catalog() -> tuple[ModelingCapability, ...]:
    """Return the built-in PDE modeling capabilities."""
    return (
        ModelingCapability(
            id="mesh.simplicial_3d_4d",
            name="3D/4D simplex meshes",
            category="mesh",
            description=(
                "Tetrahedral 3D meshes, direct 4D simplex meshes, static and moving 3D-to-4D "
                "space-time extrusion, and pipe/cylinder deformation helpers."
            ),
            equations=(
                "3D tetrahedral finite elements",
                "4D space-time simplex meshes",
                "moving-domain simplicial meshes",
            ),
            solver_symbols=(
                "SimplexMesh",
                "MovingSimplexMesh",
                "SimplexMesh.simplex_quality",
                "SimplexMesh.simplex_edge_lengths",
                "MovingSimplexMesh.simplex_quality_series",
                "MovingSimplexMesh.min_quality_series",
                "SimplexMesh.locate_points",
                "SimplexMesh.interpolate",
                "SimplexMesh.refined",
                "MovingSimplexMesh.interpolate_values",
                "assemble_simplex_mass_matrix",
                "assemble_simplex_stiffness_matrix",
                "assemble_simplex_fem_matrices",
                "assemble_simplex_load_vector",
                "solve_simplex_poisson",
                "step_simplex_diffusion",
                "simulate_simplex_diffusion",
                "box_simplex_mesh",
                "delaunay_mesh",
                "interpolate_simplex_field",
                "refine_simplex_mesh",
                "moving_mesh",
                "pipe_boundary_nodes",
                "pipe_boundary_facets",
                "pipe_radial_deformation",
                "pipe_simplex_mesh",
                "space_time_mesh",
            ),
            required_dependencies=("numpy", "scipy"),
            scenario_ids=("mesh_3d_4d_measure", "mesh_moving_pipe_deformation"),
            limitations=(
                "centroid refinement and mesh quality diagnostics are available; adaptive optimization remains future work",
                "no ALE, FSI, or curved/high-order formulation yet",
            ),
        ),
        ModelingCapability(
            id="earth.geochem_biostrat_likelihoods",
            name="3D/4D Earth posterior modeling",
            category="earth-inverse",
            description=(
                "Latent 3D fields, typed observation contracts, geochemistry/biostratigraphy likelihoods, "
                "linear and bounded MAP/Laplace inversion, 4D assimilation, cross-property priors, and "
                "posterior extraction/compression."
            ),
            equations=(
                "linear-Gaussian field inversion",
                "bounded-field Gauss-Newton MAP/Laplace inversion",
                "small-reference MCMC empirical posterior validation",
                "Kalman/RTS random-walk 4D assimilation",
                "Kalman/RTS linear-dynamics 4D assimilation",
                "ensemble Kalman nonlinear 4D assimilation",
                "particle-filter 4D sampled posterior assimilation",
                "geochemical censored-assay likelihoods",
                "multi-element assay covariance and batch-offset likelihoods",
                "biostratigraphic range-zone likelihoods",
                "palynology/microfossil assemblage likelihoods with reworking",
                "geochronology age likelihoods",
                "stratigraphic age-difference likelihoods",
                "facies/environment interval likelihoods",
                "nonlinear DC/ERT log-conductivity likelihoods",
                "layered MT log-conductivity likelihoods",
                "layered AEM apparent-conductivity likelihoods",
                "2D MT TE log-conductivity likelihoods",
                "3D MT curl-curl log-conductivity likelihoods",
                "3D CSEM curl-curl log-conductivity likelihoods",
                "dense and sparse graph-Matern smoothness priors",
                "sparse spatiotemporal random-walk priors",
                "sparse posterior precision factorization and covariance actions",
                "cross-property Gaussian priors",
                "posterior calibration diagnostics",
            ),
            solver_symbols=(
                "Field3D",
                "Field4D",
                "PosteriorField3D",
                "PosteriorField3D.posterior_predictive_draws",
                "PosteriorField4D",
                "PosteriorField4D.credible_interval",
                "PosteriorField4D.sample",
                "PosteriorField4D.at_time(interpolate=True)",
                "PosteriorField4D.cross_covariance",
                "PosteriorField4D.posterior_predictive_draws",
                "PosteriorFieldSamples4D",
                "PosteriorFieldSamples4D.at_time",
                "PosteriorFieldSamples4D.posterior_predictive_draws",
                "SampleUpdateReport",
                "Synthetic3DInversionResult",
                "Synthetic4DAssimilationResult",
                "run_synthetic_3d_geochem_geophysics_inversion",
                "run_synthetic_4d_biostrat_assimilation",
                "geochem_assay_likelihood",
                "biostrat_constraint_likelihood",
                "geochronology_age_likelihood",
                "stratigraphic_correlation_likelihood",
                "facies_interval_likelihood",
                "timed_likelihood",
                "update_sampled_field_posterior",
                "update_sampled_field_posterior_4d",
                "update_sampled_field_posterior_with_observations",
                "Observation",
                "ForwardOperatorRegistry",
                "ForwardOperator.capability_report",
                "ForwardOperatorRegistry.capability_report",
                "GeochemAssay",
                "MultiElementAssay",
                "multi_element_assay_log_likelihood",
                "multi_element_assay_posterior_predictive",
                "BiostratConstraint",
                "FossilAssemblage",
                "fossil_assemblage_log_likelihood",
                "fossil_assemblage_posterior_predictive",
                "GeochronologyAge",
                "StratigraphicCorrelation",
                "FaciesIntervalConstraint",
                "FieldGaussianPrior",
                "FieldGaussianPrior.precision_sparse",
                "linear_gaussian_invert",
                "sparse_linear_gaussian_invert",
                "gauss_newton_invert",
                "PosteriorFieldSamples3D",
                "PosteriorFieldSamples3D.posterior_predictive_draws",
                "metropolis_field_invert",
                "MCMCReport",
                "assimilate_4d",
                "assimilate_4d_linear_dynamics",
                "assimilate_4d_joint_linear_dynamics",
                "assimilate_4d_ensemble",
                "particle_assimilate_4d",
                "ParticleAssimilationReport",
                "dc_resistivity_forward_operator",
                "aem_layered_forward_operator",
                "layered_mt_forward_operator",
                "mt_2d_te_forward_operator",
                "mt_3d_forward_operator",
                "csem_3d_forward_operator",
                "CrossPropertyPrior",
                "CrossPropertyPrior.precision_sparse",
                "SpatioTemporalGaussianPrior",
                "SpatioTemporalGaussianPrior.precision_sparse",
                "depth_weighted_marginal_precision_sparse",
                "joint_linear_gaussian_invert",
                "marginal_at_points",
                "marginal_time_series",
                "region_summary",
                "region_time_summary",
                "region_mass",
                "region_mass_time_series",
                "time_slice",
                "SampledDerivedQuantity",
                "DerivedTimeSeries",
                "sampled_derived_quantity",
                "compress_to_low_rank",
                "to_ensemble",
                "truth_coverage",
                "heldout_observation_check",
                "uncertainty_inflation",
                "identifiability_diagnostic",
            ),
            required_dependencies=("numpy", "scipy", "torch"),
            inverse_ready=True,
            scenario_ids=(
                "earth_observation_likelihoods",
                "earth_forward_operator_contract",
                "earth_static_linear_inversion",
                "earth_bounded_gauss_newton",
                "earth_mcmc_reference_posterior",
                "earth_sampled_domain_likelihood_update",
                "earth_dc_resistivity_nonlinear_inversion",
                "earth_aem_layered_nonlinear_observation",
                "earth_layered_mt_nonlinear_inversion",
                "earth_mt_2d_te_nonlinear_observation",
                "earth_mt_3d_nonlinear_observation",
                "earth_csem_3d_nonlinear_observation",
                "earth_4d_assimilation",
                "earth_linear_dynamics_4d_assimilation",
                "earth_joint_4d_posterior",
                "earth_ensemble_4d_nonlinear_assimilation",
                "earth_particle_4d_nonlinear_assimilation",
                "earth_prior_coupling",
                "earth_sparse_prior_precision",
                "earth_sparse_posterior_factorization",
                "earth_posterior_extraction",
                "earth_posterior_calibration",
                "earth_synthetic_inversion_harnesses",
            ),
            limitations=(
                "geochemistry and biostratigraphy are likelihoods, not full reaction-path or paleoecology simulators",
                "multi-element censored assays use marginal censoring terms rather than a full truncated multivariate normal integral",
                "sparse posterior factorization stores precision-form Gaussian posteriors; production-scale preconditioned iterative solvers remain future work",
                "ensemble 4D assimilation is a stochastic Gaussian-summary reference, not a particle/MCMC smoother",
                "particle 4D assimilation is a sampled reference path for small nonlinear problems, not a production smoother",
                "Random-Walk Metropolis field inversion is a small-reference validation path, not a production-scale sampler",
                "DC/ERT posterior observations use finite-difference local sensitivities; production-scale adjoints remain future work",
                "MT, AEM, and CSEM observations use finite-difference local sensitivities; production-scale adjoints remain future work",
                "full airborne loop/flight-line AEM geometry remains future work",
            ),
        ),
        ModelingCapability(
            id="pde.state_space",
            name="PDE-constrained state space",
            category="inverse",
            description="Kalman/RTS smoothing and EM noise estimation for linear PDE trajectories.",
            equations=("diffusion", "advection", "advection-diffusion"),
            solver_symbols=("PDE", "DiffusionOperator", "AdvectionOperator", "AdvectionDiffusionOperator"),
            required_dependencies=("numpy", "scipy", "mixle"),
            inverse_ready=True,
            scenario_ids=("state_space_diffusion_forecast",),
            limitations=("linear transition operators only", "scalar process and observation variances"),
        ),
        ModelingCapability(
            id="pde.transient_heat",
            name="Transient heat conduction",
            category="forward-inverse",
            description="Finite-volume transient conduction for thermography and conductivity inversion.",
            equations=("rho c dT/dt = div(k grad T) + q",),
            solver_symbols=(
                "TransientHeat",
                "Differential",
                "DiffusionOperator",
                "step_simplex_diffusion",
                "simulate_simplex_diffusion",
            ),
            required_dependencies=("numpy", "scipy", "torch", "mixle"),
            differentiable=True,
            inverse_ready=True,
            scenario_ids=("heat_fourier_decay",),
            limitations=("explicit time step requires conduction CFL stability"),
        ),
        ModelingCapability(
            id="gas.reactive_combustion",
            name="Reactive gas and engine-cylinder combustion",
            category="forward-inverse",
            description=(
                "Compressible Euler reference solvers plus a zero-dimensional ideal-gas combustion "
                "chamber with fuel depletion, heat release, wall loss, and moving piston volume."
            ),
            equations=(
                "1D compressible Euler",
                "closed ideal-gas energy balance",
                "Arrhenius one-step fuel burn",
                "engine-cylinder p dV work",
            ),
            solver_symbols=(
                "exact_riemann_solution",
                "solve_euler_1d",
                "solve_reactive_euler_1d",
                "engine_cylinder_volume",
                "simulate_zero_d_combustion",
                "CombustionResult",
            ),
            required_dependencies=("numpy",),
            differentiable=False,
            inverse_ready=False,
            scenario_ids=("gas_zero_d_combustion_pressure_rise",),
            limitations=(
                "combustion uses one-step Arrhenius chemistry; no turbulent flame fronts, detonation, "
                "or detailed reaction mechanisms yet",
                "moving-volume coupling is prescribed rather than solved as fluid-structure interaction",
            ),
        ),
        ModelingCapability(
            id="geophysics.potential_fields",
            name="Potential-field geophysics",
            category="forward-inverse",
            description="Gravity, magnetic, DC, and ray-tomography operators for subsurface inference.",
            equations=("gravity point-mass kernel", "magnetic dipole kernel", "DC Poisson", "eikonal/ray tomography"),
            solver_symbols=(
                "gravity_point_sensitivity",
                "magnetic_dipole_sensitivity",
                "dc_resistivity",
                "regularized_gauss_newton",
            ),
            required_dependencies=("numpy", "scipy", "torch"),
            differentiable=True,
            inverse_ready=True,
            scenario_ids=("gravity_linearity",),
            limitations=("gravity and magnetic kernels use point-source cell approximations"),
        ),
        ModelingCapability(
            id="reasoning.mechanistic_field",
            name="Mechanistic field reasoning",
            category="reasoning",
            description="Sparse sensor fusion over a PDE trajectory prior with posterior uncertainty.",
            equations=("linear Gaussian PDE trajectory",),
            solver_symbols=("MechanisticFieldReasoner", "JointPotentialField", "SpatialFieldStore"),
            required_dependencies=("numpy", "scipy", "mixle"),
            inverse_ready=True,
            scenario_ids=("mechanistic_diffusion_reconstruction",),
            limitations=("requires linear dynamics for exact Gaussian smoothing"),
        ),
        ModelingCapability(
            id="wave.acoustic_2d",
            name="2D acoustic wave propagation",
            category="forward-inverse",
            description="Leapfrog acoustic wave propagation with absorbing sponge boundaries for FWI-style models.",
            equations=("u_tt = c(x)^2 laplacian(u) + source",),
            solver_symbols=("WaveEquation2D", "Differential", "rtm_image", "born_modeling", "lsrtm_step"),
            required_dependencies=("numpy", "scipy", "torch", "mixle"),
            differentiable=True,
            inverse_ready=True,
            scenario_ids=("wave_zero_state_stability",),
            limitations=("explicit step must satisfy the acoustic CFL condition"),
        ),
    )


def get_capability(capability_id: str) -> ModelingCapability:
    """Look up a capability by id."""
    by_id = {cap.id: cap for cap in capability_catalog()}
    try:
        return by_id[capability_id]
    except KeyError as exc:
        raise KeyError(f"unknown PDE modeling capability {capability_id!r}") from exc


def missing_required_dependencies(
    required: tuple[str, ...] = DEFAULT_REQUIRED_CAPABILITIES,
) -> dict[str, tuple[str, ...]]:
    """Return missing required import dependencies by capability id."""
    missing: dict[str, tuple[str, ...]] = {}
    for capability_id in required:
        cap = get_capability(capability_id)
        status = cap.dependency_status()
        deps = tuple(dep for dep in cap.required_dependencies if not status.get(dep, False))
        if deps:
            missing[capability_id] = deps
    return missing


def readiness_report(required: tuple[str, ...] = DEFAULT_REQUIRED_CAPABILITIES) -> dict[str, Any]:
    """Structured summary of available modeling capabilities and missing dependencies."""
    required_set = set(required)
    caps = [cap.as_dict() | {"required_for_release": cap.id in required_set} for cap in capability_catalog()]
    return {
        "required": list(required),
        "missing_dependencies": missing_required_dependencies(required),
        "capabilities": caps,
    }


def verification_scenarios() -> tuple[VerificationScenario, ...]:
    """Return deterministic modeling scenarios keyed by id."""
    return (
        VerificationScenario(
            id="mesh_3d_4d_measure",
            capability_id="mesh.simplicial_3d_4d",
            name="3D/4D simplex measure",
            description="Box tetrahedralization, 4D simplex meshing, and 3D-to-4D extrusion preserve measure.",
            runner=_run_mesh_3d_4d_measure,
        ),
        VerificationScenario(
            id="mesh_moving_pipe_deformation",
            capability_id="mesh.simplicial_3d_4d",
            name="Moving pipe deformation",
            description="A radially deforming 3D pipe/cylinder mesh remains positive and extrudes to 4D.",
            runner=_run_mesh_moving_pipe_deformation,
        ),
        VerificationScenario(
            id="earth_observation_likelihoods",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Geochemistry and biostratigraphy likelihoods",
            description="Censored assays and fossil range-zone likelihoods score consistent fields higher.",
            runner=_run_earth_observation_likelihoods,
        ),
        VerificationScenario(
            id="earth_forward_operator_contract",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Observation forward-operator contract",
            description="Gravity, magnetics, and borehole observations share the registry likelihood interface.",
            runner=_run_earth_forward_operator_contract,
        ),
        VerificationScenario(
            id="earth_static_linear_inversion",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="3D static field inversion",
            description="Linear-Gaussian inversion combines surface geophysics and borehole samples.",
            runner=_run_earth_static_linear_inversion,
        ),
        VerificationScenario(
            id="earth_bounded_gauss_newton",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Bounded-field MAP/Laplace inversion",
            description="Gauss-Newton inversion respects physical bounds through the latent-field transform.",
            runner=_run_earth_bounded_gauss_newton,
        ),
        VerificationScenario(
            id="earth_mcmc_reference_posterior",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Small-reference sampled posterior",
            description="Random-Walk Metropolis samples a one-cell posterior against a closed-form reference.",
            runner=_run_earth_mcmc_reference_posterior,
        ),
        VerificationScenario(
            id="earth_sampled_domain_likelihood_update",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Sampled domain-likelihood posterior update",
            description="Geochemical and biostratigraphic likelihoods reweight sampled 3D/4D posteriors.",
            runner=_run_earth_sampled_domain_likelihood_update,
        ),
        VerificationScenario(
            id="earth_dc_resistivity_nonlinear_inversion",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="DC/ERT nonlinear posterior observation",
            description="A DC resistivity observation exposes a local Jacobian and reduces nonlinear data residual.",
            runner=_run_earth_dc_resistivity_nonlinear_inversion,
        ),
        VerificationScenario(
            id="earth_layered_mt_nonlinear_inversion",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Layered MT nonlinear posterior observation",
            description="A layered MT observation exposes a local Jacobian and reduces nonlinear data residual.",
            runner=_run_earth_layered_mt_nonlinear_inversion,
        ),
        VerificationScenario(
            id="earth_aem_layered_nonlinear_observation",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Layered AEM nonlinear posterior observation",
            description="A layered AEM observation exposes a local Jacobian that linearizes perturbations.",
            runner=_run_earth_aem_layered_nonlinear_observation,
        ),
        VerificationScenario(
            id="earth_mt_2d_te_nonlinear_observation",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="2D MT TE nonlinear posterior observation",
            description="A 2D MT TE observation exposes a local Jacobian that linearizes perturbations.",
            runner=_run_earth_mt_2d_te_nonlinear_observation,
        ),
        VerificationScenario(
            id="earth_mt_3d_nonlinear_observation",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="3D MT curl-curl nonlinear posterior observation",
            description="A 3D MT observation exposes a local Jacobian that linearizes perturbations.",
            runner=_run_earth_mt_3d_nonlinear_observation,
        ),
        VerificationScenario(
            id="earth_csem_3d_nonlinear_observation",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="3D CSEM curl-curl nonlinear posterior observation",
            description="A 3D CSEM observation exposes a local Jacobian that linearizes perturbations.",
            runner=_run_earth_csem_3d_nonlinear_observation,
        ),
        VerificationScenario(
            id="earth_4d_assimilation",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="4D assimilation and smoothing",
            description="Time-ordered observations produce posterior slices at observed and unobserved times.",
            runner=_run_earth_4d_assimilation,
        ),
        VerificationScenario(
            id="earth_linear_dynamics_4d_assimilation",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Linear-dynamics 4D assimilation",
            description="Supplied state-transition matrices propagate observations through time.",
            runner=_run_earth_linear_dynamics_4d_assimilation,
        ),
        VerificationScenario(
            id="earth_joint_4d_posterior",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Joint 4D posterior covariance",
            description="Joint linear-dynamics assimilation stores full cross-time covariance.",
            runner=_run_earth_joint_4d_posterior,
        ),
        VerificationScenario(
            id="earth_ensemble_4d_nonlinear_assimilation",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Ensemble nonlinear 4D assimilation",
            description="Ensemble assimilation tracks nonlinear observations without a fixed Jacobian.",
            runner=_run_earth_ensemble_4d_nonlinear_assimilation,
        ),
        VerificationScenario(
            id="earth_particle_4d_nonlinear_assimilation",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Particle nonlinear 4D sampled assimilation",
            description="Particle assimilation returns sampled 4D trajectories for nonlinear observations.",
            runner=_run_earth_particle_4d_nonlinear_assimilation,
        ),
        VerificationScenario(
            id="earth_prior_coupling",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Depth and cross-property priors",
            description="Depth weighting and joint Gaussian coupling propagate information between properties.",
            runner=_run_earth_prior_coupling,
        ),
        VerificationScenario(
            id="earth_sparse_prior_precision",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Sparse graph prior precision",
            description="Sparse prior precision assembly matches dense references without dense storage.",
            runner=_run_earth_sparse_prior_precision,
        ),
        VerificationScenario(
            id="earth_sparse_posterior_factorization",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Sparse posterior precision factorization",
            description="Sparse linear-Gaussian inversion matches dense posterior means and covariance actions.",
            runner=_run_earth_sparse_posterior_factorization,
        ),
        VerificationScenario(
            id="earth_posterior_extraction",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Posterior extraction and compression",
            description="Point, section, region, derived, low-rank, diagonal, and ensemble artifacts are available.",
            runner=_run_earth_posterior_extraction,
        ),
        VerificationScenario(
            id="earth_posterior_calibration",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Posterior calibration diagnostics",
            description="Truth coverage, held-out fit, uncertainty inflation, and insufficient-data flags are measured.",
            runner=_run_earth_posterior_calibration,
        ),
        VerificationScenario(
            id="earth_synthetic_inversion_harnesses",
            capability_id="earth.geochem_biostrat_likelihoods",
            name="Synthetic 3D/4D Earth inversion harnesses",
            description="Reusable 3D geophysics+geochem and 4D biostrat assimilation harnesses return posteriors.",
            runner=_run_earth_synthetic_inversion_harnesses,
        ),
        VerificationScenario(
            id="heat_fourier_decay",
            capability_id="pde.transient_heat",
            name="Diffusion Fourier decay",
            description="Exact-transition diffusion decays a Fourier mode at the discrete analytical rate.",
            runner=_run_heat_fourier_decay,
        ),
        VerificationScenario(
            id="gas_zero_d_combustion_pressure_rise",
            capability_id="gas.reactive_combustion",
            name="Zero-dimensional combustion pressure rise",
            description="Constant-volume fuel burn increases temperature and pressure while depleting fuel.",
            runner=_run_gas_zero_d_combustion_pressure_rise,
        ),
        VerificationScenario(
            id="state_space_diffusion_forecast",
            capability_id="pde.state_space",
            name="PDE state-space forecast",
            description="A fitted diffusion state-space model forecasts the next latent field.",
            runner=_run_state_space_diffusion_forecast,
        ),
        VerificationScenario(
            id="gravity_linearity",
            capability_id="geophysics.potential_fields",
            name="Gravity linearity",
            description="The gravity forward has the expected sign and scales linearly with cell volume.",
            runner=_run_gravity_linearity,
        ),
        VerificationScenario(
            id="mechanistic_diffusion_reconstruction",
            capability_id="reasoning.mechanistic_field",
            name="Mechanistic sparse-sensor reconstruction",
            description="Sparse diffusion sensors reconstruct an unobserved space-time field.",
            runner=_run_mechanistic_diffusion_reconstruction,
        ),
        VerificationScenario(
            id="wave_zero_state_stability",
            capability_id="wave.acoustic_2d",
            name="Wave zero-state stability",
            description="A zero acoustic state with no source remains exactly zero and finite.",
            runner=_run_wave_zero_state_stability,
        ),
    )


def run_verification_scenario(scenario_id: str) -> ScenarioResult:
    """Run one named modeling readiness scenario."""
    scenarios = {scenario.id: scenario for scenario in verification_scenarios()}
    try:
        scenario = scenarios[scenario_id]
    except KeyError as exc:
        raise KeyError(f"unknown PDE verification scenario {scenario_id!r}") from exc
    return scenario.runner()


def run_required_modeling_checks(required: tuple[str, ...] = DEFAULT_REQUIRED_CAPABILITIES) -> list[ScenarioResult]:
    """Run all scenarios associated with the required capability ids."""
    missing = missing_required_dependencies(required)
    if missing:
        details = ", ".join(f"{cap}: {', '.join(deps)}" for cap, deps in sorted(missing.items()))
        raise RuntimeError(f"missing dependencies for required PDE modeling capabilities: {details}")

    scenario_ids: list[str] = []
    for capability_id in required:
        scenario_ids.extend(get_capability(capability_id).scenario_ids)
    return [run_verification_scenario(scenario_id) for scenario_id in scenario_ids]


def assert_required_modeling(
    required: tuple[str, ...] = DEFAULT_REQUIRED_CAPABILITIES,
) -> list[ScenarioResult]:
    """Run required checks and raise if any required capability fails its scenario."""
    results = run_required_modeling_checks(required)
    failed = [result for result in results if not result.passed]
    if failed:
        details = "; ".join(f"{result.id}: {result.message}" for result in failed)
        raise AssertionError(f"PDE modeling readiness checks failed: {details}")
    return results


def _result(
    scenario_id: str,
    capability_id: str,
    *,
    passed: bool,
    metrics: dict[str, float],
    tolerance: dict[str, float],
    message: str,
) -> ScenarioResult:
    return ScenarioResult(
        id=scenario_id,
        capability_id=capability_id,
        passed=bool(passed),
        metrics={key: float(value) for key, value in metrics.items()},
        tolerance={key: float(value) for key, value in tolerance.items()},
        message=message,
    )


def _small_grid(property_name: str = "density_contrast", units: str = "kg/m^3", bounds=None):
    from mixle_pde.latent import Field3D

    coords = np.array(
        [
            [0.0, 0.0, -30.0],
            [40.0, 0.0, -40.0],
            [0.0, 40.0, -50.0],
            [40.0, 40.0, -60.0],
        ],
        dtype=float,
    )
    return Field3D(coordinates=coords, spacing=40.0, units=units, property_name=property_name, bounds=bounds)


def _small_registry(grid, volumes):
    from mixle_pde.observations import ForwardOperatorRegistry, borehole_forward_operator, gravity_forward_operator

    registry = ForwardOperatorRegistry()
    registry.register(gravity_forward_operator(grid.coordinates, volumes))
    registry.register(borehole_forward_operator())
    return registry


def _run_mesh_3d_4d_measure() -> ScenarioResult:
    from mixle_pde.fem import (
        assemble_simplex_load_vector,
        assemble_simplex_mass_matrix,
        assemble_simplex_stiffness_matrix,
        solve_simplex_poisson,
        step_simplex_diffusion,
    )
    from mixle_pde.mesh import box_simplex_mesh, refine_simplex_mesh, space_time_mesh

    tol = 1.0e-12
    mesh3 = box_simplex_mesh((2, 2, 2), lengths=(2.0, 3.0, 4.0))
    mesh4 = box_simplex_mesh((2, 2, 2, 2), lengths=(2.0, 3.0, 4.0, 5.0))
    space_time = space_time_mesh(box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0)), [0.0, 0.25, 1.0])
    refined3 = refine_simplex_mesh(mesh3)
    refined4 = refine_simplex_mesh(mesh4)
    quality_ok = (
        mesh3.validate()["min_quality"] > 0.0
        and mesh4.validate()["min_quality"] > 0.0
        and space_time.validate()["min_quality"] > 0.0
        and refined3.validate()["min_quality"] > 0.0
        and refined4.validate()["min_quality"] > 0.0
    )
    err3 = abs(mesh3.total_measure() - 24.0) / 24.0
    err4 = abs(mesh4.total_measure() - 120.0) / 120.0
    err_st = abs(space_time.total_measure() - 1.0)
    err_refined3 = abs(refined3.total_measure() - mesh3.total_measure()) / mesh3.total_measure()
    err_refined4 = abs(refined4.total_measure() - mesh4.total_measure()) / mesh4.total_measure()
    mass3 = assemble_simplex_mass_matrix(mesh3)
    mass4 = assemble_simplex_mass_matrix(mesh4)
    stiffness3 = assemble_simplex_stiffness_matrix(mesh3)
    mass3_err = abs(float(mass3.sum()) - mesh3.total_measure()) / mesh3.total_measure()
    mass4_err = abs(float(mass4.sum()) - mesh4.total_measure()) / mesh4.total_measure()
    constant_residual = float(np.max(np.abs(stiffness3 @ np.ones(mesh3.n_nodes))))
    linear_x = mesh3.nodes[:, 0]
    linear_energy = float(linear_x @ (stiffness3 @ linear_x))
    linear_energy_err = abs(linear_energy - mesh3.total_measure()) / mesh3.total_measure()
    load4 = assemble_simplex_load_vector(mesh4, 2.0)
    load4_err = abs(float(load4.sum()) - 2.0 * mesh4.total_measure()) / (2.0 * mesh4.total_measure())
    poisson_mesh = box_simplex_mesh((3, 3, 3), lengths=(1.0, 1.0, 1.0))
    poisson_solution = solve_simplex_poisson(poisson_mesh, 1.0)
    poisson_boundary = poisson_mesh.boundary_nodes()
    poisson_interior = np.setdiff1d(np.arange(poisson_mesh.n_nodes), poisson_boundary)
    poisson_ok = (
        bool(np.isfinite(poisson_solution).all())
        and float(np.max(np.abs(poisson_solution[poisson_boundary]))) <= tol
        and float(np.max(poisson_solution[poisson_interior])) > 0.0
    )
    impulse = np.zeros(poisson_mesh.n_nodes)
    center = int(np.argmin(np.linalg.norm(poisson_mesh.nodes - np.array([0.5, 0.5, 0.5]), axis=1)))
    impulse[center] = 1.0
    diffused = step_simplex_diffusion(poisson_mesh, impulse, 0.05)
    diffusion_ok = (
        bool(np.isfinite(diffused).all())
        and float(diffused[center]) < 1.0
        and float(np.max(np.abs(diffused[poisson_boundary]))) <= tol
    )
    counts_ok = (
        mesh3.dim == 3
        and mesh3.n_simplices == 6
        and mesh4.dim == 4
        and mesh4.n_simplices == 24
        and space_time.dim == 4
        and space_time.n_simplices == 48
        and refined3.n_simplices == mesh3.n_simplices * 4
        and refined4.n_simplices == mesh4.n_simplices * 5
    )
    passed = (
        counts_ok
        and quality_ok
        and err3 <= tol
        and err4 <= tol
        and err_st <= tol
        and err_refined3 <= tol
        and err_refined4 <= tol
        and mass3_err <= tol
        and mass4_err <= tol
        and constant_residual <= tol
        and linear_energy_err <= tol
        and load4_err <= tol
        and poisson_ok
        and diffusion_ok
    )
    return _result(
        "mesh_3d_4d_measure",
        "mesh.simplicial_3d_4d",
        passed=passed,
        metrics={
            "box_3d_relative_measure_error": err3,
            "box_4d_relative_measure_error": err4,
            "space_time_relative_measure_error": err_st,
            "refined_3d_relative_measure_error": err_refined3,
            "refined_4d_relative_measure_error": err_refined4,
            "space_time_simplices": space_time.n_simplices,
            "refined_3d_simplices": refined3.n_simplices,
            "refined_4d_simplices": refined4.n_simplices,
            "box_3d_min_quality": mesh3.validate()["min_quality"],
            "box_4d_min_quality": mesh4.validate()["min_quality"],
            "space_time_min_quality": space_time.validate()["min_quality"],
            "fem_3d_mass_relative_error": mass3_err,
            "fem_4d_mass_relative_error": mass4_err,
            "fem_constant_stiffness_residual": constant_residual,
            "fem_linear_energy_relative_error": linear_energy_err,
            "fem_4d_load_relative_error": load4_err,
            "fem_poisson_center_value": float(np.max(poisson_solution[poisson_interior])),
            "fem_diffusion_center_value": float(diffused[center]),
        },
        tolerance={"relative_measure_error": tol},
        message="3D/4D simplex measures matched" if passed else "3D/4D simplex measure mismatch",
    )


def _run_mesh_moving_pipe_deformation() -> ScenarioResult:
    from mixle_pde.mesh import (
        moving_mesh,
        pipe_boundary_facets,
        pipe_boundary_nodes,
        pipe_radial_deformation,
        pipe_simplex_mesh,
    )

    base = pipe_simplex_mesh(inner_radius=0.5, outer_radius=1.0, length=2.0, n_theta=24, n_axial=3)
    boundary_nodes = pipe_boundary_nodes(base, inner_radius=0.5, outer_radius=1.0, length=2.0)
    boundary_facets = pipe_boundary_facets(base, inner_radius=0.5, outer_radius=1.0, length=2.0)
    motion = moving_mesh(
        base,
        [0.0, 0.5, 1.0],
        pipe_radial_deformation(axis="z", radial_strain=lambda t: 0.1 * t),
    )
    report = motion.validate()
    final_ratio = float(motion.measure_series()[-1] / motion.measure_series()[0])
    expected_ratio = 1.1**2
    space_time = motion.to_space_time_mesh()
    expected_pipe_volume = np.pi * (1.0**2 - 0.5**2) * 2.0
    pipe_volume_error = abs(base.total_measure() - expected_pipe_volume) / expected_pipe_volume
    rel_err = abs(final_ratio - expected_ratio) / expected_ratio
    counts_ok = motion.dim == 3 and space_time.dim == 4 and space_time.n_simplices == base.n_simplices * 4 * 2
    health_ok = report["positive_measure_all_steps"] and report["n_inverted_or_degenerate_relative_to_reference"] == 0
    quality_ok = report["min_quality"] > 0.0 and report["n_low_quality"] == 0
    boundary_ok = all(nodes.size > 0 for nodes in boundary_nodes.values()) and all(
        facets.size > 0 for facets in boundary_facets.values()
    )
    tol = 1.0e-12
    passed = counts_ok and health_ok and quality_ok and boundary_ok and rel_err <= tol and pipe_volume_error <= 0.02
    return _result(
        "mesh_moving_pipe_deformation",
        "mesh.simplicial_3d_4d",
        passed=passed,
        metrics={
            "final_volume_ratio": final_ratio,
            "expected_volume_ratio": expected_ratio,
            "relative_volume_ratio_error": rel_err,
            "space_time_simplices": space_time.n_simplices,
            "min_signed_measure_ratio": float(report["min_signed_measure_ratio"]),
            "min_quality": float(report["min_quality"]),
            "pipe_volume_relative_error": pipe_volume_error,
            "inner_wall_nodes": float(boundary_nodes["inner_wall"].size),
            "outer_wall_facets": float(boundary_facets["outer_wall"].shape[0]),
        },
        tolerance={"relative_volume_ratio_error": tol, "pipe_volume_relative_error": 0.02},
        message="moving pipe mesh remained valid" if passed else "moving pipe mesh deformation failed",
    )


def _run_earth_observation_likelihoods() -> ScenarioResult:
    from mixle_pde.geo_observations import (
        BiostratConstraint,
        FaciesIntervalConstraint,
        FossilAssemblage,
        GeochemAssay,
        GeochronologyAge,
        MultiElementAssay,
        StratigraphicCorrelation,
        additive_log_ratio,
        assay_log_likelihood,
        biostrat_log_likelihood,
        facies_interval_log_likelihood,
        fossil_assemblage_log_likelihood,
        geochronology_log_likelihood,
        inverse_additive_log_ratio,
        multi_element_assay_log_likelihood,
        stratigraphic_correlation_log_likelihood,
    )

    assay = GeochemAssay(
        element="Cu",
        location=np.array([[0.0, 0.0, -30.0], [40.0, 0.0, -40.0]]),
        value=np.array([120.0, 15.0]),
        noise_std=np.array([8.0, 4.0]),
        detection_limit=np.array([0.0, 15.0]),
        censored=np.array([False, True]),
        units="ppm",
        provenance={"lab": "synthetic-xrf"},
    )
    assay_gap = assay_log_likelihood(assay, np.array([120.0, 5.0])) - assay_log_likelihood(
        assay, np.array([30.0, 80.0])
    )
    multi_assay = MultiElementAssay(
        elements=("Cu", "Mo"),
        location=np.array([[0.0, 0.0, -30.0], [40.0, 0.0, -40.0]]),
        value=np.array([[120.0, 1.5], [80.0, 0.8]]),
        noise_cov=np.array([[16.0, 1.0], [1.0, 0.25]]),
        detection_limit=np.array([[0.0, 1.0], [0.0, 1.0]]),
        censored=np.array([[False, True], [False, True]]),
        batch_offset=np.array([5.0, -0.2]),
        units="ppm",
        provenance={"lab": "synthetic-icp-ms", "batch": "B1"},
    )
    multi_assay_gap = multi_element_assay_log_likelihood(
        multi_assay, np.array([[115.0, 0.3], [75.0, 0.2]])
    ) - multi_element_assay_log_likelihood(multi_assay, np.array([[40.0, 4.0], [20.0, 5.0]]))
    bio = BiostratConstraint(
        location=np.array([[0.0, 0.0, -100.0]]),
        taxon="range-zone-alpha",
        present=True,
        first_appearance=120.0,
        last_appearance=95.0,
        tolerance=3.0,
        provenance={"core": "synthetic-1"},
    )
    bio_gap = biostrat_log_likelihood(bio, 108.0) - biostrat_log_likelihood(bio, 70.0)
    assemblage = FossilAssemblage(
        taxa=("A", "B", "C"),
        location=np.array([[0.0, 0.0, -100.0]]),
        counts=np.array([18, 2, 0]),
        detection_probability=np.array([1.0, 0.8, 0.2]),
        reworking_probability=0.1,
        background_probability=np.array([0.6, 0.2, 0.2]),
        provenance={"core": "synthetic-1", "kind": "palynology"},
    )
    assemblage_gap = fossil_assemblage_log_likelihood(
        assemblage, np.array([0.8, 0.15, 0.05])
    ) - fossil_assemblage_log_likelihood(assemblage, np.array([0.05, 0.15, 0.8]))
    geochron = GeochronologyAge(
        location=np.array([[0.0, 0.0, -120.0]]),
        age=118.0,
        analytical_std=1.5,
        systematic_std=0.5,
        method="U-Pb zircon",
    )
    geochron_gap = geochronology_log_likelihood(geochron, 118.0) - geochronology_log_likelihood(geochron, 126.0)
    strat = StratigraphicCorrelation(
        location_a=np.array([[0.0, 0.0, -120.0]]),
        location_b=np.array([[20.0, 0.0, -95.0]]),
        age_difference=8.0,
        std=1.0,
    )
    strat_gap = stratigraphic_correlation_log_likelihood(
        strat, 118.0, 110.0
    ) - stratigraphic_correlation_log_likelihood(strat, 118.0, 95.0)
    facies = FaciesIntervalConstraint(
        location=np.array([[0.0, 0.0, -50.0]]),
        label="deltaic",
        property_name="facies_score",
        lower=3.0,
        upper=5.0,
        tolerance=0.5,
    )
    facies_gap = facies_interval_log_likelihood(facies, 4.0) - facies_interval_log_likelihood(facies, 8.0)
    comp = np.array([0.2, 0.5, 0.3])
    roundtrip = inverse_additive_log_ratio(additive_log_ratio(comp), total=1.0)
    alr_error = float(np.linalg.norm(roundtrip - comp))
    passed = (
        assay_gap > 0.0
        and multi_assay_gap > 0.0
        and bio_gap > 0.0
        and assemblage_gap > 0.0
        and geochron_gap > 0.0
        and strat_gap > 0.0
        and facies_gap > 0.0
        and alr_error <= 1.0e-10
    )
    return _result(
        "earth_observation_likelihoods",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "assay_log_likelihood_gap": assay_gap,
            "multi_element_assay_log_likelihood_gap": multi_assay_gap,
            "biostrat_log_likelihood_gap": bio_gap,
            "fossil_assemblage_log_likelihood_gap": assemblage_gap,
            "geochronology_log_likelihood_gap": geochron_gap,
            "stratigraphic_log_likelihood_gap": strat_gap,
            "facies_log_likelihood_gap": facies_gap,
            "alr_error": alr_error,
        },
        tolerance={"alr_error": 1.0e-10},
        message="geoscience observation likelihoods matched" if passed else "earth likelihood check failed",
    )


def _run_earth_forward_operator_contract() -> ScenarioResult:
    from mixle_pde.observations import (
        ForwardOperatorRegistry,
        Observation,
        borehole_forward_operator,
        gaussian_log_likelihood,
        gravity_forward_operator,
        magnetics_forward_operator,
    )

    grid = _small_grid()
    values = np.array([80.0, 420.0, 120.0, 300.0])
    volumes = np.full(grid.n, 40.0**3)
    stations = np.array([[0.0, 0.0, 5.0], [40.0, 0.0, 5.0], [20.0, 20.0, 5.0]])
    registry = ForwardOperatorRegistry()
    registry.register(gravity_forward_operator(grid.coordinates, volumes))
    registry.register(magnetics_forward_operator(grid.coordinates, volumes, inclination=60.0, declination=10.0))
    registry.register(borehole_forward_operator())
    gravity = Observation(
        kind="gravity",
        location=stations,
        value=registry.get("gravity").jacobian(grid, stations) @ values,
        noise_cov=np.full(stations.shape[0], 1.0e-6),
    )
    magnetic = Observation(
        kind="magnetics",
        location=stations[:2],
        value=registry.get("magnetics").jacobian(grid, stations[:2]) @ (values / 10000.0),
        noise_cov=np.full(2, 1.0e-4),
    )
    borehole = Observation(
        kind="borehole",
        location=grid.coordinates[[1, 3]],
        value=values[[1, 3]],
        noise_cov=np.full(2, 25.0),
    )
    ll = sum(registry.log_likelihood(grid, values, obs) for obs in (gravity, borehole))
    ll += gaussian_log_likelihood(
        magnetic, registry.get("magnetics").predict_observation(grid, values / 10000.0, magnetic)
    )
    report = registry.capability_report()
    shapes_ok = all(registry.get(kind).has_adjoint() for kind in ("gravity", "magnetics", "borehole"))
    metadata_ok = (
        report["gravity"]["jacobian_kind"] == "fixed"
        and report["magnetics"]["has_fixed_jacobian"]
        and report["borehole"]["differentiable"]
    )
    passed = shapes_ok and metadata_ok and np.isfinite(ll)
    return _result(
        "earth_forward_operator_contract",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={"joint_log_likelihood": ll, "registered_operator_count": float(len(report))},
        tolerance={},
        message="forward operators shared the registry contract" if passed else "forward operator contract failed",
    )


def _run_earth_static_linear_inversion() -> ScenarioResult:
    from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert, posterior_predictive_check
    from mixle_pde.observations import Observation

    grid = _small_grid()
    truth = np.array([80.0, 420.0, 120.0, 300.0])
    volumes = np.full(grid.n, 40.0**3)
    registry = _small_registry(grid, volumes)
    stations = np.array([[0.0, 0.0, 5.0], [40.0, 0.0, 5.0], [0.0, 40.0, 5.0], [40.0, 40.0, 5.0]])
    gravity_j = registry.get("gravity").jacobian(grid, stations)
    gravity = Observation(
        kind="gravity",
        location=stations,
        value=gravity_j @ truth,
        noise_cov=np.full(stations.shape[0], 1.0e-8),
    )
    borehole = Observation(
        kind="borehole",
        location=grid.coordinates[[0, 1, 3]],
        value=truth[[0, 1, 3]],
        noise_cov=np.full(3, 9.0),
    )
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0e-4, marginal_precision=1.0e-6, length_scale=60.0)
    posterior = linear_gaussian_invert(grid, [gravity, borehole], registry, prior)
    holdout_loc = np.array([[20.0, 20.0, 5.0]])
    holdout = Observation(
        kind="gravity",
        location=holdout_loc,
        value=registry.get("gravity").jacobian(grid, holdout_loc) @ truth,
        noise_cov=np.array([1.0e-8]),
    )
    prior_resid = float(np.linalg.norm(holdout.value))
    post_pred = registry.get("gravity").jacobian(grid, holdout.location) @ posterior.mean
    post_resid = float(np.linalg.norm(post_pred - holdout.value))
    check = posterior_predictive_check(posterior, registry, [holdout], alpha=0.1)
    var_ok = bool(np.all(posterior.marginal_variance > 0.0))
    passed = var_ok and post_resid < prior_resid and np.isfinite(check.coverage)
    return _result(
        "earth_static_linear_inversion",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "prior_holdout_residual": prior_resid,
            "posterior_holdout_residual": post_resid,
            "heldout_coverage": check.coverage,
            "mean_rmse": float(np.sqrt(np.mean((posterior.mean - truth) ** 2))),
        },
        tolerance={"posterior_residual_lt_prior": 1.0},
        message="3D linear-Gaussian inversion improved held-out fit" if passed else "3D inversion check failed",
    )


def _run_earth_bounded_gauss_newton() -> ScenarioResult:
    from mixle_pde.field_gauss_newton import gauss_newton_invert
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator

    grid = _small_grid("porosity", "fraction", bounds=(0.0, 1.0))
    truth = np.array([0.08, 0.32, 0.18, 0.45])
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    obs = Observation(kind="borehole", location=grid.coordinates, value=truth, noise_cov=np.full(grid.n, 0.02**2))
    prior = FieldGaussianPrior(
        mean=grid.to_unconstrained(np.full(grid.n, 0.2)),
        smoothness_precision=0.05,
        marginal_precision=0.2,
        length_scale=80.0,
    )
    posterior, report = gauss_newton_invert(grid, [obs], registry, prior, max_iter=80, tol=1.0e-8)
    physical = grid.from_unconstrained(posterior.mean)
    rmse = float(np.sqrt(np.mean((physical - truth) ** 2)))
    passed = report.converged and rmse <= 0.05 and bool(np.all((physical > 0.0) & (physical < 1.0)))
    return _result(
        "earth_bounded_gauss_newton",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={"rmse": rmse, "iterations": report.iterations, "final_data_misfit": report.final_data_misfit},
        tolerance={"rmse": 0.05},
        message="bounded MAP/Laplace inversion matched" if passed else "bounded inversion check failed",
    )


def _run_earth_mcmc_reference_posterior() -> ScenarioResult:
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.field_mcmc import metropolis_field_invert
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator

    grid = Field3D(
        coordinates=np.array([[0.0, 0.0, -10.0]]),
        spacing=1.0,
        units="kg/m^3",
        property_name="density_contrast",
    )
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    observation = Observation("borehole", grid.coordinates, np.array([2.0]), np.array([0.25]))
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)
    posterior, report = metropolis_field_invert(
        grid,
        [observation],
        registry,
        prior,
        n_samples=500,
        burn_in=150,
        thin=1,
        step_scale=0.7,
        rng=np.random.default_rng(17),
    )
    expected_var = 1.0 / (1.0 + 1.0 / 0.25)
    expected_mean = expected_var * (2.0 / 0.25)
    mean_error = abs(float(posterior.mean[0]) - expected_mean)
    variance_error = abs(float(posterior.marginal_variance[0]) - expected_var)
    lo, hi = posterior.credible_interval(alpha=0.1)
    interval_ok = bool(lo[0] < expected_mean < hi[0])
    passed = (
        report.stored_samples == 500
        and 0.15 < report.acceptance_rate < 0.95
        and mean_error <= 0.2
        and variance_error <= 0.15
        and interval_ok
    )
    return _result(
        "earth_mcmc_reference_posterior",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "acceptance_rate": report.acceptance_rate,
            "posterior_mean_error": mean_error,
            "posterior_variance_error": variance_error,
            "stored_samples": float(report.stored_samples),
        },
        tolerance={"posterior_mean_error": 0.2, "posterior_variance_error": 0.15},
        message="MCMC reference posterior matched" if passed else "MCMC reference posterior check failed",
    )


def _run_earth_sampled_domain_likelihood_update() -> ScenarioResult:
    from mixle_pde.field_assimilation import PosteriorFieldSamples4D
    from mixle_pde.geo_observations import (
        BiostratConstraint,
        GeochemAssay,
    )
    from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
    from mixle_pde.sample_update import (
        biostrat_constraint_likelihood,
        geochem_assay_likelihood,
        timed_likelihood,
        update_sampled_field_posterior_4d,
        update_sampled_field_posterior_with_observations,
    )

    assay_grid = Field3D(
        coordinates=np.array([[0.0, 0.0, 0.0]]),
        spacing=1.0,
        units="ppm",
        property_name="Cu",
    )
    assay_posterior = PosteriorFieldSamples3D(
        grid=assay_grid,
        samples=np.array([[1.0], [4.0], [8.0], [11.5], [12.0], [12.5]]),
    )
    assay = GeochemAssay(
        element="Cu",
        location=np.array([[0.0, 0.0, 0.0]]),
        value=np.array([12.0]),
        noise_std=np.array([0.25]),
        units="ppm",
    )

    updated_3d, report_3d = update_sampled_field_posterior_with_observations(
        assay_posterior,
        [geochem_assay_likelihood(assay, assay_grid)],
        n_samples=64,
        rng=np.random.default_rng(5),
    )

    age_grid = Field3D(
        coordinates=np.array([[0.0, 0.0, -100.0]]),
        spacing=1.0,
        units="Ma",
        property_name="age",
    )
    age_posterior = PosteriorFieldSamples4D(
        grid=age_grid,
        times=np.array([0.0, 1.0]),
        samples=np.array([[[20.0], [35.0]], [[20.0], [55.0]], [[20.0], [65.0]], [[20.0], [90.0]]]),
    )
    bio = BiostratConstraint(
        location=np.array([[0.0, 0.0, -100.0]]),
        taxon="Zone-A",
        present=True,
        first_appearance=60.0,
        last_appearance=50.0,
        tolerance=1.0,
    )

    updated_4d, report_4d = update_sampled_field_posterior_4d(
        age_posterior,
        [[], [timed_likelihood(biostrat_constraint_likelihood(bio, age_grid), 1.0)]],
        n_samples=32,
        rng=np.random.default_rng(7),
    )
    age_samples = updated_4d.physical_samples[:, 1, 0]
    passed = (
        updated_3d.samples.shape == (64, 1)
        and float(updated_3d.physical_samples.mean()) > 10.0
        and np.isclose(updated_3d.map[0], 12.0)
        and report_3d.effective_sample_size < assay_posterior.samples.shape[0]
        and updated_4d.samples.shape == (32, 2, 1)
        and report_4d.effective_sample_size < age_posterior.n_samples
        and bool(np.all((age_samples >= 50.0) & (age_samples <= 60.0)))
    )
    return _result(
        "earth_sampled_domain_likelihood_update",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "geochem_updated_mean": float(updated_3d.physical_samples.mean()),
            "geochem_effective_sample_size": report_3d.effective_sample_size,
            "biostrat_effective_sample_size": report_4d.effective_sample_size,
            "biostrat_mean_age": float(age_samples.mean()),
        },
        tolerance={
            "geochem_updated_mean_gt": 10.0,
            "biostrat_age_range_lo": 50.0,
            "biostrat_age_range_hi": 60.0,
        },
        message="sampled domain likelihood update moved posteriors"
        if passed
        else "sampled domain likelihood update failed",
    )


def _run_earth_dc_resistivity_nonlinear_inversion() -> ScenarioResult:
    from mixle_pde.field_gauss_newton import gauss_newton_invert
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import (
        ForwardOperatorRegistry,
        Observation,
        dc_resistivity_forward_operator,
    )

    shape = (4, 4, 4)
    node = np.arange(np.prod(shape)).reshape(shape)
    coords = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)], dtype=float)
    grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity")
    schedule = (
        (int(node[1, 1, 1]), int(node[1, 1, 2]), int(node[2, 1, 1]), int(node[2, 1, 2])),
        (int(node[1, 2, 1]), int(node[1, 2, 2]), int(node[2, 2, 1]), int(node[2, 2, 2])),
        (int(node[1, 1, 1]), int(node[1, 2, 1]), int(node[2, 1, 2]), int(node[2, 2, 2])),
    )
    locations = np.array([[1.5, 1.0, 1.5], [1.5, 2.0, 1.5], [1.5, 1.5, 1.5]])
    truth = np.zeros(grid.n)
    truth[int(node[2, 1, 1])] = 0.25
    truth[int(node[2, 2, 2])] = -0.15
    op = dc_resistivity_forward_operator(
        shape,
        schedule,
        sigma_ref=0.02,
        log_data=True,
        finite_difference_step=3.0e-5,
    )
    registry = ForwardOperatorRegistry()
    registry.register(op)
    observation = Observation(
        "dc_resistivity",
        locations,
        op.predict(grid, truth, locations),
        np.full(len(schedule), 0.02**2),
    )
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0e-3, marginal_precision=5.0e-2, length_scale=2.0)
    posterior, report = gauss_newton_invert(grid, [observation], registry, prior, max_iter=6, tol=1.0e-8)
    prior_residual = float(np.linalg.norm(observation.value - op.predict(grid, np.zeros(grid.n), locations)))
    posterior_residual = float(np.linalg.norm(observation.value - op.predict(grid, posterior.mean, locations)))
    passed = posterior_residual < prior_residual and np.isfinite(report.final_data_misfit)
    return _result(
        "earth_dc_resistivity_nonlinear_inversion",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "prior_residual": prior_residual,
            "posterior_residual": posterior_residual,
            "final_data_misfit": report.final_data_misfit,
        },
        tolerance={"posterior_residual_lt_prior": 1.0},
        message="DC/ERT nonlinear posterior observation reduced residual"
        if passed
        else "DC/ERT posterior check failed",
    )


def _run_earth_layered_mt_nonlinear_inversion() -> ScenarioResult:
    from mixle_pde.field_gauss_newton import gauss_newton_invert
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import (
        ForwardOperatorRegistry,
        Observation,
        layered_mt_forward_operator,
    )

    freqs = np.array([0.5, 2.0, 10.0, 50.0])
    thicknesses = (300.0,)
    locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
    grid = Field3D(
        coordinates=np.array([[0.0, 0.0, -100.0], [0.0, 0.0, -500.0]]),
        spacing=400.0,
        units="log(S/m)",
        property_name="log_conductivity",
    )
    truth = np.log(np.array([2.0, 15.0]))
    op = layered_mt_forward_operator(freqs, thicknesses, sigma_ref=0.01, finite_difference_step=1.0e-5)
    registry = ForwardOperatorRegistry()
    registry.register(op)
    observation = Observation(
        "layered_mt_log_apparent_resistivity",
        locations,
        op.predict(grid, truth, locations),
        np.full(freqs.size, 0.03**2),
    )
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0e-3, marginal_precision=1.0e-2, length_scale=500.0)
    posterior, report = gauss_newton_invert(grid, [observation], registry, prior, max_iter=12, tol=1.0e-8)
    prior_residual = float(np.linalg.norm(observation.value - op.predict(grid, np.zeros(grid.n), locations)))
    posterior_residual = float(np.linalg.norm(observation.value - op.predict(grid, posterior.mean, locations)))
    passed = posterior_residual < prior_residual and np.isfinite(report.final_data_misfit)
    return _result(
        "earth_layered_mt_nonlinear_inversion",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "prior_residual": prior_residual,
            "posterior_residual": posterior_residual,
            "final_data_misfit": report.final_data_misfit,
        },
        tolerance={"posterior_residual_lt_prior": 1.0},
        message="layered MT nonlinear posterior observation reduced residual"
        if passed
        else "layered MT posterior check failed",
    )


def _run_earth_aem_layered_nonlinear_observation() -> ScenarioResult:
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import Observation, aem_layered_forward_operator

    freqs = np.array([1.0, 10.0, 100.0])
    thicknesses = (300.0,)
    locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
    grid = Field3D(
        coordinates=np.array([[0.0, 0.0, -100.0], [0.0, 0.0, -500.0]]),
        spacing=400.0,
        units="log(S/m)",
        property_name="log_conductivity",
    )
    values = np.log(np.array([2.0, 20.0]))
    op = aem_layered_forward_operator(freqs, thicknesses, sigma_ref=0.01, finite_difference_step=1.0e-5)
    observation = Observation(
        "aem_layered_log_apparent_conductivity",
        locations,
        op.predict(grid, values, locations),
        np.full(freqs.size, 0.03**2),
    )
    jac = op.local_jacobian(grid, values, observation)
    perturb = np.array([1.0e-4, -2.0e-4])
    base = op.predict_observation(grid, values, observation)
    moved = op.predict_observation(grid, values + perturb, observation)
    linearized = jac @ perturb
    error = float(np.linalg.norm((moved - base) - linearized))
    signal = float(np.linalg.norm(moved - base))
    rel_error = error / max(signal, 1.0e-12)
    passed = np.all(np.isfinite(base)) and jac.shape == (freqs.size, grid.n) and rel_error <= 2.0e-3
    return _result(
        "earth_aem_layered_nonlinear_observation",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={"linearization_relative_error": rel_error, "signal_norm": signal},
        tolerance={"linearization_relative_error": 2.0e-3},
        message="layered AEM nonlinear posterior observation linearized"
        if passed
        else "layered AEM observation check failed",
    )


def _run_earth_mt_2d_te_nonlinear_observation() -> ScenarioResult:
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import Observation, mt_2d_te_forward_operator

    shape = (3, 8)
    freq = 10.0
    spacing = 100.0
    coords = np.array([[float(i), 0.0, -float(j)] for i in range(shape[0]) for j in range(shape[1])])
    grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity_2d")
    values = np.zeros(grid.n)
    values[shape[1] + 2] = 0.15
    locations = np.column_stack([np.arange(shape[0], dtype=float), np.zeros(shape[0]), np.zeros(shape[0])])
    op = mt_2d_te_forward_operator(shape, freq, spacing=spacing, sigma_ref=0.02, finite_difference_step=1.0e-5)
    observation = Observation(
        "mt_2d_te_log_apparent_resistivity",
        locations,
        op.predict(grid, values, locations),
        np.full(shape[0], 0.05**2),
    )
    jac = op.local_jacobian(grid, values, observation)
    perturb = np.zeros(grid.n)
    perturb[shape[1] + 2] = 1.0e-4
    base = op.predict_observation(grid, values, observation)
    moved = op.predict_observation(grid, values + perturb, observation)
    linearized = jac @ perturb
    error = float(np.linalg.norm((moved - base) - linearized))
    signal = float(np.linalg.norm(moved - base))
    rel_error = error / max(signal, 1.0e-12)
    passed = np.all(np.isfinite(base)) and jac.shape == (shape[0], grid.n) and rel_error <= 2.0e-3
    return _result(
        "earth_mt_2d_te_nonlinear_observation",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={"linearization_relative_error": rel_error, "signal_norm": signal},
        tolerance={"linearization_relative_error": 2.0e-3},
        message="2D MT TE nonlinear posterior observation linearized"
        if passed
        else "2D MT TE observation check failed",
    )


def _run_earth_mt_3d_nonlinear_observation() -> ScenarioResult:
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import Observation, mt_3d_forward_operator

    shape = (3, 3, 6)
    freqs = np.array([5.0, 20.0])
    spacing = 50.0
    coords = np.array(
        [[float(i), float(j), -float(k)] for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])]
    )
    grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity_3d")
    values = np.zeros(grid.n)
    param = shape[2] * (shape[1] + 1) + 2
    values[param] = 0.1
    locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
    op = mt_3d_forward_operator(shape, freqs, spacing=spacing, sigma_ref=0.05, finite_difference_step=1.0e-5)
    observation = Observation(
        "mt_3d_log_apparent_resistivity",
        locations,
        op.predict(grid, values, locations),
        np.full(freqs.size, 0.05**2),
    )
    jac = op.local_jacobian(grid, values, observation)
    perturb = np.zeros(grid.n)
    perturb[int(np.argmax(np.linalg.norm(jac, axis=0)))] = 1.0e-4
    base = op.predict_observation(grid, values, observation)
    moved = op.predict_observation(grid, values + perturb, observation)
    linearized = jac @ perturb
    error = float(np.linalg.norm((moved - base) - linearized))
    signal = float(np.linalg.norm(moved - base))
    rel_error = error / max(signal, 1.0e-12)
    passed = np.all(np.isfinite(base)) and jac.shape == (freqs.size, grid.n) and rel_error <= 3.0e-3
    return _result(
        "earth_mt_3d_nonlinear_observation",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={"linearization_relative_error": rel_error, "signal_norm": signal},
        tolerance={"linearization_relative_error": 3.0e-3},
        message="3D MT nonlinear posterior observation linearized" if passed else "3D MT observation check failed",
    )


def _run_earth_csem_3d_nonlinear_observation() -> ScenarioResult:
    from mixle_pde.em_diffusion_3d import _edge_coords, _edge_layout
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import Observation, csem_3d_forward_operator

    shape = (4, 4, 4)
    freq = 1.0
    spacing = 50.0
    coords = np.array(
        [[float(i), float(j), -float(k)] for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])]
    )
    grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity_3d")
    _, _, _, _, (sx, _, _) = _edge_layout(shape)
    source = (min(shape[0] // 2, sx[0] - 1) * sx[1] + shape[1] // 2) * sx[2] + shape[2] // 2
    receivers = np.array([source, source - sx[1] * sx[2], source - sx[2]], dtype=int)
    edge_coords, _ = _edge_coords(shape, (spacing, spacing, spacing))
    locations = edge_coords[receivers]
    values = np.zeros(grid.n)
    op = csem_3d_forward_operator(
        shape,
        freq,
        [int(source)],
        receivers,
        spacing=spacing,
        sigma_ref=0.1,
        finite_difference_step=1.0e-5,
    )
    observation = Observation(
        "csem_3d_log_amplitude",
        locations,
        op.predict(grid, values, locations),
        np.full(len(receivers), 0.05**2),
    )
    jac = op.local_jacobian(grid, values, observation)
    perturb = np.zeros(grid.n)
    perturb[int(np.argmax(np.linalg.norm(jac, axis=0)))] = 1.0e-4
    base = op.predict_observation(grid, values, observation)
    moved = op.predict_observation(grid, values + perturb, observation)
    linearized = jac @ perturb
    error = float(np.linalg.norm((moved - base) - linearized))
    signal = float(np.linalg.norm(moved - base))
    rel_error = error / max(signal, 1.0e-12)
    passed = np.all(np.isfinite(base)) and jac.shape == (len(receivers), grid.n) and rel_error <= 3.0e-3
    return _result(
        "earth_csem_3d_nonlinear_observation",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={"linearization_relative_error": rel_error, "signal_norm": signal},
        tolerance={"linearization_relative_error": 3.0e-3},
        message="3D CSEM nonlinear posterior observation linearized" if passed else "3D CSEM observation check failed",
    )


def _run_earth_4d_assimilation() -> ScenarioResult:
    from mixle_pde.field_assimilation import assimilate_4d
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.latent import Field4D
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator

    grid = _small_grid("temperature_c", "C")
    times = np.array([0.0, 1.0, 2.0])
    field4d = Field4D(grid, times=times, provenance={"scenario": "earth_4d_assimilation"})
    truth = [
        np.array([40.0, 55.0, 65.0, 80.0]),
        np.array([45.0, 63.0, 73.0, 91.0]),
        np.array([52.0, 72.0, 84.0, 105.0]),
    ]
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    obs_by_time = [
        [Observation("borehole", grid.coordinates, truth[0], np.full(grid.n, 4.0), time=times[0])],
        [],
        [Observation("borehole", grid.coordinates[[1, 3]], truth[2][[1, 3]], np.full(2, 4.0), time=times[2])],
    ]
    prior = FieldGaussianPrior(mean=45.0, smoothness_precision=0.02, marginal_precision=0.05, length_scale=80.0)
    posterior = assimilate_4d(grid, times, obs_by_time, registry, prior, process_var=30.0)
    start_err = float(np.linalg.norm(prior.mean_vector(grid) - truth[2]))
    final_err = float(np.linalg.norm(posterior.at_time(2.0).mean - truth[2]))
    mid = posterior.at_time(1.0).mean
    lower = np.minimum(posterior.at_time(0.0).mean, posterior.at_time(2.0).mean) - 1.0e-6
    upper = np.maximum(posterior.at_time(0.0).mean, posterior.at_time(2.0).mean) + 1.0e-6
    bracketed = float(np.mean((mid >= lower) & (mid <= upper)))
    ci_lo, ci_hi = posterior.credible_interval(0.1)
    samples = posterior.sample(4, np.random.default_rng(42))
    interpolated = posterior.at_time(0.5, interpolate=True).mean
    artifacts_ok = (
        field4d.n == times.size * grid.n
        and field4d.coordinates.shape == (times.size * grid.n, 4)
        and posterior.mean_array.shape == (times.size, grid.n)
        and posterior.marginal_std.shape == (times.size, grid.n)
        and ci_lo.shape == (times.size, grid.n)
        and ci_hi.shape == (times.size, grid.n)
        and samples.shape == (4, times.size, grid.n)
        and interpolated.shape == (grid.n,)
    )
    passed = final_err < start_err and bracketed >= 0.75 and len(posterior.means) == len(times) and artifacts_ok
    return _result(
        "earth_4d_assimilation",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "prior_final_error": start_err,
            "posterior_final_error": final_err,
            "mid_bracket_fraction": bracketed,
            "sample_count": float(samples.shape[0]),
            "time_count": float(posterior.mean_array.shape[0]),
            "field4d_node_count": float(field4d.n),
        },
        tolerance={"posterior_error_lt_prior": 1.0},
        message="4D assimilation produced posterior artifacts" if passed else "4D assimilation check failed",
    )


def _run_earth_linear_dynamics_4d_assimilation() -> ScenarioResult:
    from mixle_pde.field_assimilation import assimilate_4d_linear_dynamics
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator

    grid = Field3D(
        coordinates=np.array([[0.0, 0.0, 0.0]]),
        spacing=1.0,
        units="state",
        property_name="growth_state",
    )
    times = np.array([0.0, 1.0, 2.0])
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    observations = [
        [],
        [],
        [Observation("borehole", grid.coordinates, np.array([4.0]), np.array([0.01]), time=2.0)],
    ]
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)
    posterior = assimilate_4d_linear_dynamics(
        grid,
        times,
        observations,
        registry,
        prior,
        transitions=np.array([[[2.0]], [[2.0]]]),
        process_cov=1.0e-4,
    )
    means = posterior.mean_array[:, 0]
    expected = np.array([1.0, 2.0, 4.0])
    max_error = float(np.max(np.abs(means - expected)))
    passed = max_error <= 0.08 and posterior.marginal_std[-1, 0] < posterior.marginal_std[0, 0]
    return _result(
        "earth_linear_dynamics_4d_assimilation",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "max_trajectory_error": max_error,
            "start_mean": float(means[0]),
            "mid_mean": float(means[1]),
            "final_mean": float(means[2]),
        },
        tolerance={"max_trajectory_error": 0.08},
        message="linear-dynamics 4D assimilation propagated observations"
        if passed
        else "linear-dynamics 4D assimilation failed",
    )


def _run_earth_joint_4d_posterior() -> ScenarioResult:
    from mixle_pde.field_assimilation import assimilate_4d_joint_linear_dynamics
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator

    grid = Field3D(np.array([[0.0, 0.0, 0.0]]), spacing=1.0, units="state", property_name="growth_state")
    times = np.array([0.0, 1.0, 2.0])
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    observations = [
        [],
        [],
        [Observation("borehole", grid.coordinates, np.array([4.0]), np.array([0.01]), time=2.0)],
    ]
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0)
    posterior = assimilate_4d_joint_linear_dynamics(
        grid,
        times,
        observations,
        registry,
        prior,
        transitions=np.array([[[2.0]], [[2.0]]]),
        process_cov=1.0e-4,
    )
    draws = posterior.sample(8, np.random.default_rng(13))
    means = posterior.mean_array[:, 0]
    expected = np.array([1.0, 2.0, 4.0])
    max_error = float(np.max(np.abs(means - expected)))
    cross_cov = float(posterior.cross_covariance(0.0, 2.0)[0, 0])
    passed = (
        posterior.joint_cov is not None
        and posterior.joint_cov.shape == (3, 3)
        and cross_cov > 0.0
        and draws.shape == (8, 3, 1)
        and max_error <= 0.08
    )
    return _result(
        "earth_joint_4d_posterior",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={"max_trajectory_error": max_error, "start_final_cross_covariance": cross_cov},
        tolerance={"max_trajectory_error": 0.08},
        message="joint 4D posterior covariance stored" if passed else "joint 4D posterior check failed",
    )


def _run_earth_ensemble_4d_nonlinear_assimilation() -> ScenarioResult:
    from mixle_pde.field_assimilation import assimilate_4d_ensemble
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import ForwardOperator, ForwardOperatorRegistry, Observation

    grid = Field3D(np.array([[0.0, 0.0, -10.0]]), spacing=1.0, units="state", property_name="nonlinear_state")
    times = np.array([0.0, 1.0, 2.0])
    truth = np.array([1.0, 1.35, 1.7])
    registry = ForwardOperatorRegistry()

    def predict(grid, field_values, obs_locations):
        return np.full(obs_locations.shape[0], float(field_values[0] ** 2))

    registry.register(ForwardOperator("square_sensor", predict=predict))
    observations = [
        [
            Observation(
                "square_sensor",
                np.array([[0.0, 0.0, -10.0]]),
                np.array([value**2]),
                np.array([0.03**2]),
                time=time,
            )
        ]
        for time, value in zip(times, truth, strict=True)
    ]
    prior = FieldGaussianPrior(mean=0.8, smoothness_precision=0.0, marginal_precision=4.0, length_scale=1.0)
    posterior = assimilate_4d_ensemble(
        grid,
        times,
        observations,
        registry,
        prior,
        process_var=0.08,
        ensemble_size=256,
        rng=np.random.default_rng(123),
    )
    final_mean = float(posterior.at_time(2.0).mean[0])
    prior_error = abs(0.8 - float(truth[-1]))
    posterior_error = abs(final_mean - float(truth[-1]))
    held = Observation(
        "square_sensor",
        np.array([[0.0, 0.0, -10.0]]),
        np.array([0.0]),
        np.array([0.03**2]),
        time=2.0,
    )
    pred_error = float(abs(posterior.predict_observation(registry, held)[0] - truth[-1] ** 2))
    passed = posterior_error < prior_error and posterior_error < 0.25 and pred_error < 0.7
    return _result(
        "earth_ensemble_4d_nonlinear_assimilation",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "prior_final_error": prior_error,
            "posterior_final_error": posterior_error,
            "nonlinear_predictive_error": pred_error,
        },
        tolerance={"posterior_final_error": 0.25, "nonlinear_predictive_error": 0.7},
        message="ensemble nonlinear 4D assimilation reduced residual"
        if passed
        else "ensemble nonlinear assimilation check failed",
    )


def _run_earth_particle_4d_nonlinear_assimilation() -> ScenarioResult:
    from mixle_pde.field_assimilation import particle_assimilate_4d
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import ForwardOperator, ForwardOperatorRegistry, Observation

    grid = Field3D(np.array([[0.0, 0.0, -10.0]]), spacing=1.0, units="state", property_name="nonlinear_state")
    times = np.array([0.0, 1.0, 2.0])
    truth = np.array([1.0, 1.35, 1.7])
    registry = ForwardOperatorRegistry()

    def predict(grid, field_values, obs_locations):
        return np.full(obs_locations.shape[0], float(field_values[0] ** 2))

    registry.register(ForwardOperator("square_sensor", predict=predict))
    observations = [
        [
            Observation(
                "square_sensor",
                np.array([[0.0, 0.0, -10.0]]),
                np.array([value**2]),
                np.array([0.04**2]),
                time=time,
            )
        ]
        for time, value in zip(times, truth, strict=True)
    ]
    prior = FieldGaussianPrior(mean=0.9, smoothness_precision=0.0, marginal_precision=9.0, length_scale=1.0)
    posterior, report = particle_assimilate_4d(
        grid,
        times,
        observations,
        registry,
        prior,
        process_var=0.08,
        n_particles=500,
        rng=np.random.default_rng(1234),
    )
    held = Observation(
        "square_sensor",
        np.array([[0.0, 0.0, -10.0]]),
        np.array([0.0]),
        np.array([0.04**2]),
        time=2.0,
    )
    prior_error = abs(0.9 - float(truth[-1]))
    posterior_error = abs(float(posterior.mean_array[-1, 0]) - float(truth[-1]))
    pred_error = float(abs(posterior.predict_observation(registry, held)[0] - truth[-1] ** 2))
    draws = posterior.posterior_predictive_draws(registry, held, n=16, rng=np.random.default_rng(4321))
    passed = (
        posterior_error < prior_error
        and posterior_error < 0.3
        and pred_error < 0.7
        and posterior.samples.shape == (500, times.size, grid.n)
        and draws.shape == (16, 1)
        and all(ess > 0.0 for ess in report.effective_sample_size)
    )
    return _result(
        "earth_particle_4d_nonlinear_assimilation",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "prior_final_error": prior_error,
            "posterior_final_error": posterior_error,
            "nonlinear_predictive_error": pred_error,
            "min_effective_sample_size": float(min(report.effective_sample_size)),
            "posterior_predictive_draws": float(draws.shape[0]),
        },
        tolerance={"posterior_final_error": 0.3, "nonlinear_predictive_error": 0.7},
        message="particle nonlinear 4D assimilation produced sampled trajectories"
        if passed
        else "particle nonlinear assimilation check failed",
    )


def _run_earth_prior_coupling() -> ScenarioResult:
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.field_priors import CrossPropertyPrior, depth_weights, joint_linear_gaussian_invert
    from mixle_pde.latent import Field3D
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator

    grid_a = _small_grid("density_contrast", "kg/m^3")
    grid_b = Field3D(
        coordinates=grid_a.coordinates.copy(),
        spacing=grid_a.spacing,
        units="ppm",
        property_name="cu_ppm",
        bounds=None,
    )
    density = np.array([80.0, 420.0, 120.0, 300.0])
    cu_truth = 0.5 * density
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    obs_a = [Observation("borehole", grid_a.coordinates, density, np.full(grid_a.n, 16.0))]
    prior = CrossPropertyPrior(
        FieldGaussianPrior(mean=0.0, smoothness_precision=1.0e-4, marginal_precision=1.0e-5, length_scale=80.0),
        FieldGaussianPrior(mean=0.0, smoothness_precision=1.0e-4, marginal_precision=1.0e-5, length_scale=80.0),
        coupling=0.5,
        slope=0.5,
    )
    post_a, post_b = joint_linear_gaussian_invert(grid_a, grid_b, prior, obs_a, registry, [], registry)
    corr = float(np.corrcoef(post_b.mean, cu_truth)[0, 1])
    weights = depth_weights(grid_a)
    depth_order_ok = float(weights[-1] < weights[0])
    passed = corr >= 0.95 and depth_order_ok > 0.0 and post_a.marginal_std.mean() > 0.0
    return _result(
        "earth_prior_coupling",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={"coupled_property_correlation": corr, "deep_weight_lt_shallow": depth_order_ok},
        tolerance={"min_correlation": 0.95},
        message="depth weighting and cross-property coupling matched" if passed else "prior coupling check failed",
    )


def _run_earth_sparse_prior_precision() -> ScenarioResult:
    from mixle_pde.field_inversion import FieldGaussianPrior
    from mixle_pde.field_priors import (
        CrossPropertyPrior,
        SpatioTemporalGaussianPrior,
        depth_weighted_marginal_precision,
        depth_weighted_marginal_precision_sparse,
    )
    from mixle_pde.latent import Field3D, Field4D

    xs = np.linspace(0.0, 80.0, 4)
    ys = np.linspace(0.0, 80.0, 4)
    zs = np.array([-20.0, -60.0, -100.0])
    coords = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    grid = Field3D(coords, spacing=40.0, units="log(S/m)", property_name="log_conductivity")
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=2.0e-3, marginal_precision=1.0e-4, length_scale=40.0)
    sparse = prior.precision_sparse(grid)
    dense_error = float(np.linalg.norm(sparse.toarray() - prior.precision(grid)))
    weighted_sparse = depth_weighted_marginal_precision_sparse(prior, grid, beta=3.0, z0=10.0)
    weighted_error = float(
        np.linalg.norm(weighted_sparse.toarray() - depth_weighted_marginal_precision(prior, grid, beta=3.0, z0=10.0))
    )
    joint = CrossPropertyPrior(prior, prior, coupling=0.2, slope=1.5)
    joint_sparse = joint.precision_sparse(grid)
    joint_error = float(np.linalg.norm(joint_sparse.toarray() - joint.precision(grid)))
    field4d = Field4D(grid, times=np.array([0.0, 1.0, 3.0]))
    st_prior = SpatioTemporalGaussianPrior(prior, temporal_precision=0.5)
    st_sparse = st_prior.precision_sparse(field4d)
    st_error = float(np.linalg.norm(st_sparse.toarray() - st_prior.precision(field4d)))
    density = float(sparse.nnz / (grid.n * grid.n))
    joint_density = float(joint_sparse.nnz / ((2 * grid.n) * (2 * grid.n)))
    st_density = float(st_sparse.nnz / (field4d.n * field4d.n))
    passed = (
        dense_error <= 1.0e-12
        and weighted_error <= 1.0e-12
        and joint_error <= 1.0e-12
        and st_error <= 1.0e-12
        and density < 0.35
        and joint_density < 0.35
        and st_density < 0.35
    )
    return _result(
        "earth_sparse_prior_precision",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "single_precision_density": density,
            "joint_precision_density": joint_density,
            "dense_sparse_error": dense_error,
            "depth_weighted_sparse_error": weighted_error,
            "joint_sparse_error": joint_error,
            "spatiotemporal_sparse_error": st_error,
            "spatiotemporal_precision_density": st_density,
        },
        tolerance={"max_sparse_dense_error": 1.0e-12, "max_density": 0.35},
        message="sparse graph prior precision matched dense references" if passed else "sparse prior check failed",
    )


def _run_earth_sparse_posterior_factorization() -> ScenarioResult:
    from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert, sparse_linear_gaussian_invert
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator
    from mixle_pde.posterior_query import derived_quantity

    grid = _small_grid("density_contrast", "kg/m^3")
    truth = np.array([80.0, 420.0, 120.0, 300.0])
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    obs = Observation(
        kind="borehole",
        location=grid.coordinates[[0, 1, 3]],
        value=truth[[0, 1, 3]],
        noise_cov=np.full(3, 16.0),
    )
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1.0e-3, marginal_precision=1.0e-4, length_scale=80.0)
    dense = linear_gaussian_invert(grid, [obs], registry, prior)
    sparse = sparse_linear_gaussian_invert(grid, [obs], registry, prior)
    weights = np.array([1.0, -2.0, 0.5, 3.0])
    dense_q = derived_quantity(dense, weights)
    sparse_q = derived_quantity(sparse, weights)
    mean_error = float(np.linalg.norm(sparse.mean - dense.mean))
    marginal_error = float(np.linalg.norm(sparse.marginal_variance - dense.marginal_variance))
    derived_std_error = abs(float(sparse_q.std - dense_q.std))
    storage_ok = sparse.cov is None and sparse.precision_factor is not None
    tol = 1.0e-7
    passed = storage_ok and mean_error <= tol and marginal_error <= tol and derived_std_error <= tol
    return _result(
        "earth_sparse_posterior_factorization",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "mean_l2_error": mean_error,
            "marginal_variance_l2_error": marginal_error,
            "derived_std_error": derived_std_error,
        },
        tolerance={"max_error": tol},
        message="sparse posterior factorization matched dense covariance actions"
        if passed
        else "sparse posterior factorization check failed",
    )


def _run_earth_posterior_extraction() -> ScenarioResult:
    from mixle_pde.field_assimilation import PosteriorField4D, PosteriorFieldSamples4D
    from mixle_pde.latent import PosteriorField3D, PosteriorFieldSamples3D
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator
    from mixle_pde.posterior_query import (
        compress_to_low_rank,
        marginal_at_points,
        marginal_time_series,
        region_mass,
        region_mass_time_series,
        region_summary,
        region_time_summary,
        sampled_derived_quantity,
        section,
        time_slice,
        to_diagonal,
        to_ensemble,
    )

    grid = _small_grid()
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    obs = Observation("borehole", grid.coordinates[[1, 3]], np.zeros(2), np.full(2, 1.0))
    mean = np.array([80.0, 420.0, 120.0, 300.0])
    cov = np.array(
        [
            [25.0, 5.0, 1.0, 0.0],
            [5.0, 36.0, 2.0, 1.0],
            [1.0, 2.0, 16.0, 3.0],
            [0.0, 1.0, 3.0, 49.0],
        ]
    )
    posterior = PosteriorField3D(grid=grid, mean=mean, cov=cov)
    point = marginal_at_points(posterior, [1, 3])
    sec = section(posterior, z=-30.0)
    region = region_summary(posterior, np.array([False, True, False, True]))
    mass = region_mass(posterior, np.array([False, True, False, True]), np.full(grid.n, 40.0**3))
    predictive = posterior.posterior_predictive_draws(registry, obs, n=4, rng=np.random.default_rng(99))
    sampled = PosteriorFieldSamples3D(
        grid=grid,
        samples=np.array(
            [
                [70.0, 410.0, 110.0, 290.0],
                [85.0, 425.0, 125.0, 305.0],
                [95.0, 440.0, 130.0, 320.0],
            ]
        ),
    )
    sampled_region = region_summary(sampled, np.array([False, True, False, True]))
    sampled_mass = region_mass(sampled, np.array([False, True, False, True]), np.full(grid.n, 40.0**3))
    sampled_q = sampled_derived_quantity(sampled, np.array([1.0, 0.0, -1.0, 0.0]))
    sampled_predictive = sampled.posterior_predictive_draws(registry, obs, n=4, rng=np.random.default_rng(98))
    posterior4d = PosteriorField4D(grid=grid, times=np.array([0.0, 1.0]), means=[mean, mean + 10.0], covs=[cov, cov])
    sampled4d = PosteriorFieldSamples4D(
        grid=grid,
        times=np.array([0.0, 1.0]),
        samples=np.stack([sampled.samples, sampled.samples + 10.0], axis=1),
    )
    point4d = marginal_time_series(posterior4d, [1, 3])
    sampled_region4d = region_time_summary(sampled4d, np.array([False, True, False, True]))
    mass4d = region_mass_time_series(posterior4d, np.array([False, True, False, True]), np.full(grid.n, 40.0**3))
    sampled_mass4d = region_mass_time_series(sampled4d, np.array([False, True, False, True]), np.full(grid.n, 40.0**3))
    slice4d = time_slice(posterior4d, 0.5, interpolate=True)
    compact = compress_to_low_rank(posterior, rank=2)
    diagonal = to_diagonal(posterior)
    ensemble = to_ensemble(compact, 8, np.random.default_rng(123))
    sampled_ensemble = to_ensemble(sampled, 8, np.random.default_rng(321))
    passed = (
        point.mean.shape == (2,)
        and len(sec["mean"]) == 1
        and region["n_cells"] == 2
        and mass.std > 0.0
        and predictive.shape == (4, 2)
        and sampled_region["n_cells"] == 2
        and sampled_mass.std > 0.0
        and sampled_q.samples.shape == (3,)
        and sampled_predictive.shape == (4, 2)
        and point4d.mean.shape == (2, 2)
        and sampled_region4d["mean"].shape == (2, 2)
        and mass4d.mean.shape == (2,)
        and sampled_mass4d.samples.shape == (3, 2)
        and slice4d.mean.shape == (grid.n,)
        and compact.low_rank.shape == (grid.n, 2)
        and diagonal.diag_var.shape == (grid.n,)
        and ensemble.samples.shape == (8, grid.n)
        and sampled_ensemble.samples.shape == (8, grid.n)
    )
    return _result(
        "earth_posterior_extraction",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "low_rank_columns": 2.0,
            "ensemble_samples": 8.0,
            "sampled_derived_draws": float(sampled_q.samples.size),
            "time_series_count": float(point4d.mean.shape[0]),
            "posterior_predictive_draws": float(predictive.shape[0] + sampled_predictive.shape[0]),
            "region_cells": region["n_cells"],
        },
        tolerance={},
        message="posterior extraction and compression matched" if passed else "posterior extraction check failed",
    )


def _run_earth_posterior_calibration() -> ScenarioResult:
    from mixle_pde.latent import Field3D, PosteriorField3D
    from mixle_pde.observations import ForwardOperatorRegistry, Observation, borehole_forward_operator
    from mixle_pde.posterior_calibration import (
        heldout_observation_check,
        identifiability_diagnostic,
        observation_sensitivity,
        truth_coverage,
        uncertainty_inflation,
    )

    coords = np.array([[float(i), 0.0, -10.0] for i in range(6)])
    grid = Field3D(coords, spacing=1.0, units="ppm", property_name="cu_ppm")
    truth = np.array([2.0, 3.0, 4.0, 8.0, 12.0, 16.0])
    cov = np.diag([0.25, 0.25, 0.25, 4.0, 9.0, 16.0])
    posterior = PosteriorField3D(grid, mean=truth.copy(), cov=cov)
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    observed = Observation("borehole", grid.coordinates[:3], truth[:3], np.full(3, 0.25))
    heldout = Observation("borehole", grid.coordinates[[0, 3, 5]], truth[[0, 3, 5]], np.full(3, 0.25))
    coverage = truth_coverage(posterior, truth, alpha=0.1)
    heldout_fit = heldout_observation_check(posterior, registry, [heldout], alpha=0.1)
    sensitivity = observation_sensitivity(posterior, registry, [observed])
    inflation = uncertainty_inflation(posterior, sensitive_mask=sensitivity > 0.0)
    diagnostic = identifiability_diagnostic(posterior, registry, [observed], min_sensitive_fraction=0.75)
    passed = (
        coverage.coverage >= 0.9
        and np.isfinite(heldout_fit.log_likelihood)
        and heldout_fit.standardized_rmse <= 1.0e-12
        and inflation.ratio > 3.0
        and diagnostic.insufficient_observations
    )
    return _result(
        "earth_posterior_calibration",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "truth_coverage": coverage.coverage,
            "heldout_log_likelihood": heldout_fit.log_likelihood,
            "heldout_standardized_rmse": heldout_fit.standardized_rmse,
            "uncertainty_inflation_ratio": inflation.ratio,
            "sensitive_fraction": diagnostic.sensitive_fraction,
        },
        tolerance={"min_truth_coverage": 0.9, "min_uncertainty_inflation_ratio": 3.0},
        message="posterior calibration diagnostics matched" if passed else "posterior calibration check failed",
    )


def _run_earth_synthetic_inversion_harnesses() -> ScenarioResult:
    from mixle_pde.earth_scenarios import (
        run_synthetic_3d_geochem_geophysics_inversion,
        run_synthetic_4d_biostrat_assimilation,
    )

    result3 = run_synthetic_3d_geochem_geophysics_inversion(n_samples=512, rng=np.random.default_rng(11))
    result4 = run_synthetic_4d_biostrat_assimilation(n_samples=512, rng=np.random.default_rng(12))
    assay_improved = result3.metrics["assay_cell_updated_error"] < result3.metrics["assay_cell_geophysical_error"]
    time_improved = result4.metrics["biostrat_updated_final_error"] < result4.metrics["dynamics_final_error"]
    artifacts_ok = (
        result3.geochem_updated_posterior.samples.shape == (512, result3.grid.n)
        and result4.biostrat_updated_posterior.samples.shape == (512, result4.times.size, result4.grid.n)
        and result4.metrics["joint_start_final_covariance"] > 0.0
    )
    passed = assay_improved and time_improved and artifacts_ok
    return _result(
        "earth_synthetic_inversion_harnesses",
        "earth.geochem_biostrat_likelihoods",
        passed=passed,
        metrics={
            "assay_cell_geophysical_error": result3.metrics["assay_cell_geophysical_error"],
            "assay_cell_updated_error": result3.metrics["assay_cell_updated_error"],
            "dynamics_final_error": result4.metrics["dynamics_final_error"],
            "biostrat_updated_final_error": result4.metrics["biostrat_updated_final_error"],
            "joint_start_final_covariance": result4.metrics["joint_start_final_covariance"],
        },
        tolerance={"improvement_required": 1.0},
        message="synthetic 3D/4D Earth inversion harnesses improved posterior metrics"
        if passed
        else "synthetic Earth inversion harness check failed",
    )


def _run_heat_fourier_decay() -> ScenarioResult:
    from mixle_pde.dynamics import DiffusionOperator

    diffusivity = 0.23
    n = 48
    length = 1.0
    dt = 0.07
    steps = 6
    op = DiffusionOperator(diffusivity, n, length=length, bc="periodic", scheme="exact")
    x = np.arange(n, dtype=float)
    u = np.sin(2.0 * np.pi * x / n)
    transition = op.transition_matrix(dt)
    for _ in range(steps):
        u = transition @ u
    discrete_lambda = -4.0 * diffusivity * math.sin(math.pi / n) ** 2 / (op.h * op.h)
    expected_amp = math.exp(discrete_lambda * dt * steps)
    initial = np.sin(2.0 * np.pi * x / n)
    measured_amp = float(np.dot(u, initial) / np.dot(initial, initial))
    rel_error = abs(measured_amp - expected_amp) / abs(expected_amp)
    tol = 1.0e-10
    return _result(
        "heat_fourier_decay",
        "pde.transient_heat",
        passed=rel_error <= tol,
        metrics={"relative_error": rel_error, "measured_amplitude": measured_amp, "expected_amplitude": expected_amp},
        tolerance={"relative_error": tol},
        message="diffusion Fourier decay matched" if rel_error <= tol else "diffusion decay mismatch",
    )


def _run_gas_zero_d_combustion_pressure_rise() -> ScenarioResult:
    from mixle_pde.gas_dynamics import simulate_zero_d_combustion, solve_reactive_euler_1d

    time = np.linspace(0.0, 0.05, 101)
    result = simulate_zero_d_combustion(
        time,
        initial_temperature=1200.0,
        initial_pressure=101325.0,
        initial_fuel_fraction=0.02,
        volume=1.0e-3,
        heat_release=4.0e7,
        pre_exponential=800.0,
        activation_temperature=6000.0,
    )
    pressure_ratio = float(result.pressure[-1] / result.pressure[0])
    fuel_burned = float(result.fuel_fraction[0] - result.fuel_fraction[-1])
    finite = bool(
        np.isfinite(result.temperature).all()
        and np.isfinite(result.pressure).all()
        and np.isfinite(result.fuel_fraction).all()
    )
    n = 48
    p0 = np.full(n, 5.0e5)
    fuel0 = np.full(n, 0.02)
    _, _, reactive_pressure, reactive_fuel = solve_reactive_euler_1d(
        np.ones(n),
        np.zeros(n),
        p0,
        fuel0,
        dx=1.0 / n,
        t_final=0.002,
        heat_release=4.0e7,
        pre_exponential=1000.0,
        activation_temperature=3000.0,
    )
    reactive_pressure_ratio = float(np.mean(reactive_pressure) / np.mean(p0))
    reactive_fuel_burned = float(np.mean(fuel0) - np.mean(reactive_fuel))
    reactive_ok = (
        bool(np.isfinite(reactive_pressure).all())
        and bool(np.isfinite(reactive_fuel).all())
        and reactive_pressure_ratio > 1.01
        and reactive_fuel_burned > 0.0
        and bool(np.all(reactive_fuel >= 0.0))
    )
    passed = (
        finite
        and pressure_ratio > 1.05
        and fuel_burned > 0.0
        and np.all(result.fuel_fraction >= 0.0)
        and reactive_ok
    )
    return _result(
        "gas_zero_d_combustion_pressure_rise",
        "gas.reactive_combustion",
        passed=passed,
        metrics={
            "pressure_ratio": pressure_ratio,
            "fuel_burned": fuel_burned,
            "reactive_euler_pressure_ratio": reactive_pressure_ratio,
            "reactive_euler_fuel_burned": reactive_fuel_burned,
        },
        tolerance={"min_pressure_ratio": 1.05, "min_fuel_burned": 0.0},
        message="zero-dimensional combustion pressure rose" if passed else "zero-dimensional combustion failed",
    )


def _run_state_space_diffusion_forecast() -> ScenarioResult:
    from mixle_pde.dynamics import DiffusionOperator
    from mixle_pde.pde import kalman_rts_em

    n = 10
    steps = 7
    dt = 0.4
    op = DiffusionOperator(0.12, n, length=float(n), bc="neumann", scheme="exact")
    transition = op.transition_matrix(dt)
    x0 = np.exp(-((np.arange(n) - 4.5) ** 2) / 3.0)
    states = [x0]
    for _ in range(steps):
        states.append(transition @ states[-1])
    observations = np.asarray(states[:-1])
    result = kalman_rts_em(observations, op, dt=dt, max_its=40, tol=1.0e-9)
    forecast = result.forecast(1)[0]
    target = states[-1]
    rel_error = float(np.linalg.norm(forecast - target) / max(np.linalg.norm(target), 1.0e-12))
    tol = 0.08
    return _result(
        "state_space_diffusion_forecast",
        "pde.state_space",
        passed=rel_error <= tol and np.isfinite(result.loglik),
        metrics={"relative_forecast_error": rel_error, "loglik": result.loglik},
        tolerance={"relative_forecast_error": tol},
        message="state-space diffusion forecast matched" if rel_error <= tol else "state-space forecast mismatch",
    )


def _run_gravity_linearity() -> ScenarioResult:
    from mixle_pde.geophysics import gravity_point_sensitivity

    obs = np.array([[0.0, 0.0, 0.0]])
    cells = np.array([[0.0, 0.0, -100.0]])
    base = gravity_point_sensitivity(obs, cells, 1.0e6)[0, 0]
    doubled = gravity_point_sensitivity(obs, cells, 2.0e6)[0, 0]
    rel_error = abs(float(doubled) - 2.0 * float(base)) / max(abs(2.0 * float(base)), 1.0e-30)
    tol = 1.0e-12
    passed = base > 0.0 and rel_error <= tol
    return _result(
        "gravity_linearity",
        "geophysics.potential_fields",
        passed=passed,
        metrics={"base_response_mgal_per_kg_m3": base, "linearity_relative_error": rel_error},
        tolerance={"linearity_relative_error": tol},
        message="gravity forward sign and linearity matched" if passed else "gravity forward check failed",
    )


def _run_mechanistic_diffusion_reconstruction() -> ScenarioResult:
    from mixle_pde.dynamics import DiffusionOperator
    from mixle_pde.reasoning import MechanisticFieldReasoner

    n = 12
    steps = 8
    dt = 0.5
    op = DiffusionOperator(0.15, n, length=float(n), bc="neumann", scheme="implicit")
    transition = op.transition_matrix(dt)
    x0 = np.exp(-((np.arange(n) - 6.0) ** 2) / 2.0)
    truth = [x0]
    for _ in range(1, steps):
        truth.append(transition @ truth[-1])
    truth_arr = np.asarray(truth)
    reasoner = MechanisticFieldReasoner(op, dt=dt, steps=steps, x0_sd=2.0, process_sd=0.01)
    sensors = [
        reasoner.sensor(cell=cell, step=step, value=float(truth_arr[step, cell]), noise_sd=0.02)
        for step in (0, 4)
        for cell in (2, 4, 6, 8, 10)
    ]
    answer = reasoner.reason(sensors)
    reconstruction = reasoner.field(answer)
    corr = float(np.corrcoef(reconstruction.ravel(), truth_arr.ravel())[0, 1])
    unobserved_error = float(np.linalg.norm(reconstruction[5] - truth_arr[5]) / n)
    passed = corr >= 0.9 and unobserved_error <= 0.05
    return _result(
        "mechanistic_diffusion_reconstruction",
        "reasoning.mechanistic_field",
        passed=passed,
        metrics={"correlation": corr, "unobserved_step_l2_per_cell": unobserved_error},
        tolerance={"min_correlation": 0.9, "max_unobserved_step_l2_per_cell": 0.05},
        message="mechanistic field reconstruction matched" if passed else "mechanistic field reconstruction failed",
    )


def _run_wave_zero_state_stability() -> ScenarioResult:
    import torch

    from mixle_pde.ops import make_ops
    from mixle_pde.wave import WaveEquation2D

    torch.set_default_dtype(torch.float64)
    n = 14
    solver = WaveEquation2D(n, dt=0.2 / (n - 1), absorb_width=2)
    ops = make_ops()
    state = solver.pack(np.zeros(n * n), np.zeros(n * n))
    c2 = torch.ones(n * n)
    for _ in range(12):
        state = solver.step(state, c2, ops)
    max_abs = float(torch.max(torch.abs(state)).detach())
    passed = max_abs <= 1.0e-12 and bool(torch.isfinite(state).all())
    return _result(
        "wave_zero_state_stability",
        "wave.acoustic_2d",
        passed=passed,
        metrics={"max_abs_state": max_abs},
        tolerance={"max_abs_state": 1.0e-12},
        message="zero wave state remained stable" if passed else "zero wave state drifted",
    )
