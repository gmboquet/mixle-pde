"""Executable, narrowly-declared multiphysics inversion reference backend.

The backend consumes ``CON-MATH-PROBLEM-V1`` dictionaries and is intentionally
limited to the two-region composite heat benchmark. It proves the full mechanics of
FEM execution, conservative monolithic/partitioned coupling, independent refinement
evidence, bounded Bayesian inversion, and validated surrogate fallback without
claiming general multiphysics coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from mixle_pde.problem_adapter import PDEBackendProfile, PDECompatibilityReport, require_compatible

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REFERENCE_CODE_SCHEMA = "mixle-pde-composite-heat-reference/1"

REFERENCE_PROFILE = PDEBackendProfile(
    id="composite-heat-p1-bayesian-reference",
    operator_kinds=frozenset({"weak_form", "coupling"}),
    discretizations=frozenset({"P1", "monolithic", "partitioned"}),
    objective_senses=frozenset({"infer"}),
    mesh_cell_types=frozenset({"line"}),
    evidence_kinds=frozenset({"residual", "convergence", "conservation"}),
)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_digest(value: Any) -> str:
    """Digest plain artifacts using the shared canonical-JSON interchange rules."""

    encoded = json.dumps(
        _plain(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class ReferenceExecutionError(RuntimeError):
    category = "execution"

    def __init__(self, code: str, message: str, *, diagnostics: Mapping[str, Any] | None = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.diagnostics = dict(diagnostics or {})


class DomainFailure(ReferenceExecutionError):
    category = "domain"


class CompileFailure(ReferenceExecutionError):
    category = "compile"


class SolverFailure(ReferenceExecutionError):
    category = "solver"


class InferenceFailure(ReferenceExecutionError):
    category = "inference"


class ResourceFailure(ReferenceExecutionError):
    category = "resource"


class CancellationFailure(ReferenceExecutionError):
    category = "cancellation"


class EvidenceFailure(ReferenceExecutionError):
    category = "evidence"


class CouplingMode(str, Enum):
    MONOLITHIC = "monolithic"
    PARTITIONED = "partitioned"


@dataclass(frozen=True)
class ObservationValue:
    position_m: float
    temperature_K: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.position_m) and 0.5 < self.position_m < 1.0):
            raise DomainFailure("E_OBSERVATION_DOMAIN", "reference observations must lie inside the right region")
        if not math.isfinite(self.temperature_K):
            raise DomainFailure("E_OBSERVATION_VALUE", "observation temperature must be finite")


@dataclass(frozen=True)
class ReferenceData:
    observations: tuple[ObservationValue, ...]
    truth_conductivity: float | None = None

    def __post_init__(self) -> None:
        if len(self.observations) < 2 or len({item.position_m for item in self.observations}) < 2:
            raise InferenceFailure(
                "E_UNIDENTIFIABLE_CONFIGURATION",
                "at least two distinct right-region observations are required for the reference inversion",
            )
        if self.truth_conductivity is not None and (
            not math.isfinite(self.truth_conductivity) or self.truth_conductivity <= 0
        ):
            raise DomainFailure("E_TRUTH_VALUE", "synthetic truth conductivity must be positive and finite")


@dataclass(frozen=True)
class ExecutionIdentities:
    problem: str
    data: str
    solver: str
    mesh: str
    prior: str
    likelihood: str
    code: str

    def __post_init__(self) -> None:
        for label, value in asdict(self).items():
            if not _SHA256.fullmatch(value):
                raise CompileFailure("E_IDENTITY", f"{label} identity must be a SHA-256 digest")

    @property
    def identity(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class ReferenceStudy:
    problem: Mapping[str, Any]
    compatibility: PDECompatibilityReport
    mode: CouplingMode
    data: ReferenceData
    conductivity_left: float
    temperature_left_K: float
    temperature_right_K: float
    conductivity_bounds: tuple[float, float]
    conductivity_prior: Mapping[str, Any]
    noise_sigma_K: float
    identities: ExecutionIdentities


@dataclass(frozen=True)
class CouplingIteration:
    iteration: int
    interface_temperature_K: float
    left_flux_W_per_m2: float
    right_flux_W_per_m2: float
    flux_residual_W_per_m2: float


@dataclass(frozen=True)
class ForwardResult:
    backend: str
    mode: CouplingMode
    conductivity_right: float
    heat_flux_W_per_m2: float
    interface_temperatures_K: tuple[float, float]
    observation_predictions_K: tuple[float, ...]
    residual_vector: tuple[float, ...]
    oriented_interface_fluxes: tuple[float, float]
    coupling_history: tuple[CouplingIteration, ...]
    mesh_scale: float
    solution_identity: str


@dataclass(frozen=True)
class FEMResult:
    forward: ForwardResult
    nodes_m: tuple[float, ...]
    temperatures_K: tuple[float, ...]
    algebraic_residual: float


@dataclass(frozen=True)
class RefinementEvidence:
    scales: tuple[float, ...]
    l2_errors: tuple[float, ...]
    observed_orders: tuple[float, ...]
    manufactured_solution: str
    source: str


@dataclass(frozen=True)
class ValidatedSurrogate:
    conductivity_grid: tuple[float, ...]
    prediction_grid: tuple[tuple[float, ...], ...]
    training_domain: tuple[float, float]
    held_out_max_error_K: float
    truth_solver_identity: str
    validation_identity: str


@dataclass(frozen=True)
class SurrogateDecision:
    predictions_K: tuple[float, ...]
    used_surrogate: bool
    reason: str
    held_out_error_K: float
    authoritative_fallback: bool
    receipt_identity: str


@dataclass(frozen=True)
class PosteriorPointResult:
    value: float
    weight: float
    provenance_digest: str


@dataclass(frozen=True)
class PosteriorSummary:
    mean: float
    map_value: float
    credible_interval: tuple[float, float]
    credible_mass: float
    truth_recovered: bool | None
    information_gain_nats: float
    prior_boundary_mass: float
    identifiability_warning: str | None
    samples: tuple[PosteriorPointResult, ...]
    surrogate_uses: int
    authoritative_fallbacks: int
    identities: ExecutionIdentities
    summary_identity: str


def _required_by_id(values: list[Mapping[str, Any]], item_id: str, kind: str) -> Mapping[str, Any]:
    found = next((item for item in values if item.get("id") == item_id), None)
    if found is None:
        raise CompileFailure("E_REFERENCE_SHAPE", f"missing required {kind} {item_id!r}")
    return found


def prepare_reference_study(
    problem: Mapping[str, Any],
    problem_hash: str,
    data: ReferenceData,
    *,
    code_revision: str,
) -> ReferenceStudy:
    """Negotiate and bind an exact reference study before numerical execution."""

    if not _SHA256.fullmatch(problem_hash):
        raise CompileFailure("E_PROBLEM_IDENTITY", "problem hash must come from the Discrete contract parser")
    try:
        compatibility = require_compatible(problem, REFERENCE_PROFILE)
    except ValueError as error:
        raise CompileFailure("E_UNSUPPORTED_PROBLEM", str(error)) from error
    operators = list(problem.get("operators", ()))
    weak_forms = [item for item in operators if item.get("kind") == "weak_form"]
    couplings = [item for item in operators if item.get("kind") == "coupling"]
    if len(weak_forms) != 2 or len(couplings) != 1:
        raise CompileFailure("E_REFERENCE_SHAPE", "reference backend requires two weak forms and one coupling")
    coupling_mode = couplings[0].get("discretization")
    try:
        mode = CouplingMode(coupling_mode)
    except ValueError as error:
        raise CompileFailure("E_COUPLING_MODE", f"unsupported coupling mode {coupling_mode!r}") from error
    unknowns = list(problem.get("unknowns", ()))
    conductivity = _required_by_id(unknowns, "conductivity_right", "unknown")
    if conductivity.get("value_kind") != "parameter" or not conductivity.get("prior"):
        raise CompileFailure("E_PRIOR", "right conductivity must be a prior-governed parameter")
    bounds = conductivity.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise CompileFailure("E_PRIOR_BOUNDS", "right conductivity requires finite bounds")
    low, high = float(bounds[0]), float(bounds[1])
    if not (0 < low < high and math.isfinite(high)):
        raise CompileFailure("E_PRIOR_BOUNDS", "right conductivity bounds must be positive and finite")
    fixed = dict(problem.get("metadata", {}).get("fixed_parameters", {}))
    for parameter_id in ("conductivity_left", "temperature_at_x0", "temperature_at_x1"):
        if parameter_id not in fixed:
            raise CompileFailure("E_FIXED_PARAMETER", f"missing fixed parameter {parameter_id!r}")
    k_left = float(fixed["conductivity_left"])
    t_left = float(fixed["temperature_at_x0"])
    t_right = float(fixed["temperature_at_x1"])
    if not (k_left > 0 and t_left > t_right and all(math.isfinite(x) for x in (k_left, t_left, t_right))):
        raise DomainFailure("E_FIXED_PARAMETER", "reference fixed values are outside their validity domain")
    models = problem.get("metadata", {}).get("observation_models", ())
    if len(models) != 1:
        raise CompileFailure("E_LIKELIHOOD", "reference study requires exactly one observation model")
    noise = models[0].get("noise", {})
    if noise.get("family") != "independent_gaussian":
        raise CompileFailure("E_LIKELIHOOD", "only the declared independent Gaussian likelihood is supported")
    sigma = float(noise.get("sigma", 0.0))
    if not math.isfinite(sigma) or sigma <= 0:
        raise CompileFailure("E_LIKELIHOOD", "likelihood sigma must be positive and finite")
    prior = dict(conductivity["prior"])
    if prior.get("family") != "lognormal" or float(prior.get("log_std", 0.0)) <= 0:
        raise CompileFailure("E_PRIOR", "reference backend supports a bounded lognormal prior")
    mesh_payload = [domain.get("properties", {}) for domain in problem.get("domains", ())]
    identities = ExecutionIdentities(
        problem=problem_hash,
        data=canonical_digest(asdict(data)),
        solver=canonical_digest(
            {
                "profile": asdict(REFERENCE_PROFILE),
                "mode": mode.value,
                "coupling_tolerance": 1.0e-10,
                "inference": "bounded-grid-quadrature",
            }
        ),
        mesh=canonical_digest(mesh_payload),
        prior=canonical_digest(prior),
        likelihood=canonical_digest(noise),
        code=canonical_digest({"schema": REFERENCE_CODE_SCHEMA, "revision": code_revision}),
    )
    return ReferenceStudy(
        problem,
        compatibility,
        mode,
        data,
        k_left,
        t_left,
        t_right,
        (low, high),
        prior,
        sigma,
        identities,
    )


def _temperature_at(study: ReferenceStudy, conductivity_right: float, interface_temperature: float, x: float) -> float:
    if x <= 0.5:
        return study.temperature_left_K + (interface_temperature - study.temperature_left_K) * (x / 0.5)
    return interface_temperature + (study.temperature_right_K - interface_temperature) * ((x - 0.5) / 0.5)


def solve_reference_forward(
    study: ReferenceStudy,
    conductivity_right: float,
    *,
    maximum_iterations: int = 80,
    tolerance: float = 1.0e-10,
    relaxation: float = 0.6,
) -> ForwardResult:
    """Solve the coupled two-region system using the negotiated coupling mode."""

    low, high = study.conductivity_bounds
    if not math.isfinite(conductivity_right) or not low <= conductivity_right <= high:
        raise DomainFailure("E_CONDUCTIVITY_DOMAIN", "right conductivity is outside its declared bounds")
    if maximum_iterations <= 0:
        raise ResourceFailure("E_ITERATION_BUDGET", "maximum coupling iterations must be positive")
    if not 0 < relaxation <= 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise SolverFailure("E_COUPLING_POLICY", "invalid partitioned coupling policy")
    left_conductance = study.conductivity_left / 0.5
    right_conductance = conductivity_right / 0.5
    history: list[CouplingIteration] = []
    if study.mode is CouplingMode.MONOLITHIC:
        heat_flux = (study.temperature_left_K - study.temperature_right_K) / (
            0.5 / study.conductivity_left + 0.5 / conductivity_right
        )
        interface_temperature = study.temperature_left_K - heat_flux * 0.5 / study.conductivity_left
        left_flux = right_flux = heat_flux
    else:
        interface_temperature = 0.5 * (study.temperature_left_K + study.temperature_right_K)
        for iteration in range(1, maximum_iterations + 1):
            left_flux = left_conductance * (study.temperature_left_K - interface_temperature)
            right_flux = right_conductance * (interface_temperature - study.temperature_right_K)
            residual = left_flux - right_flux
            history.append(CouplingIteration(iteration, interface_temperature, left_flux, right_flux, abs(residual)))
            if abs(residual) <= tolerance:
                break
            interface_temperature += relaxation * residual / (left_conductance + right_conductance)
        else:
            raise SolverFailure(
                "E_COUPLING_NONCONVERGENCE",
                "partitioned interface iteration exhausted its bounded budget",
                diagnostics={"iterations": maximum_iterations, "last_residual": history[-1].flux_residual_W_per_m2},
            )
        heat_flux = 0.5 * (left_flux + right_flux)
    flux_residual = left_flux - right_flux
    predictions = tuple(
        _temperature_at(study, conductivity_right, interface_temperature, item.position_m)
        for item in study.data.observations
    )
    payload = {
        "backend": "analytic-conservative-reference",
        "mode": study.mode.value,
        "conductivity_right": conductivity_right,
        "heat_flux": heat_flux,
        "interface_temperature": interface_temperature,
        "predictions": predictions,
        "history": history,
        "identities": study.identities.identity,
    }
    return ForwardResult(
        "analytic-conservative-reference",
        study.mode,
        conductivity_right,
        heat_flux,
        (interface_temperature, interface_temperature),
        predictions,
        (flux_residual, 0.0, 0.0),
        (left_flux, -right_flux),
        tuple(history),
        0.5,
        canonical_digest(payload),
    )


def solve_reference_fem(
    study: ReferenceStudy,
    conductivity_right: float,
    *,
    cells_per_region: int = 4,
) -> FEMResult:
    """Assemble and solve the actual P1 finite-element system on the tagged line domain."""

    if isinstance(cells_per_region, bool) or cells_per_region < 1:
        raise ResourceFailure("E_MESH_BUDGET", "cells per region must be a positive integer")
    low, high = study.conductivity_bounds
    if not low <= conductivity_right <= high:
        raise DomainFailure("E_CONDUCTIVITY_DOMAIN", "right conductivity is outside its declared bounds")
    n_cells = 2 * cells_per_region
    nodes = np.linspace(0.0, 1.0, n_cells + 1)
    stiffness = np.zeros((n_cells + 1, n_cells + 1), dtype=float)
    for cell in range(n_cells):
        h = nodes[cell + 1] - nodes[cell]
        conductivity = study.conductivity_left if cell < cells_per_region else conductivity_right
        local = conductivity / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
        stiffness[cell : cell + 2, cell : cell + 2] += local
    interior = np.arange(1, n_cells)
    boundary_values = np.array([study.temperature_left_K, study.temperature_right_K])
    rhs = -stiffness[np.ix_(interior, np.array([0, n_cells]))] @ boundary_values
    temperatures = np.empty(n_cells + 1, dtype=float)
    temperatures[[0, n_cells]] = boundary_values
    temperatures[interior] = np.linalg.solve(stiffness[np.ix_(interior, interior)], rhs)
    algebraic = stiffness @ temperatures
    residual = float(np.linalg.norm(algebraic[interior], ord=np.inf))
    interface = cells_per_region
    h_left = nodes[interface] - nodes[interface - 1]
    h_right = nodes[interface + 1] - nodes[interface]
    left_flux = -study.conductivity_left * (temperatures[interface] - temperatures[interface - 1]) / h_left
    right_flux = -conductivity_right * (temperatures[interface + 1] - temperatures[interface]) / h_right
    predictions = tuple(float(np.interp(item.position_m, nodes, temperatures)) for item in study.data.observations)
    forward = ForwardResult(
        "p1-line-fem",
        study.mode,
        conductivity_right,
        0.5 * (left_flux + right_flux),
        (float(temperatures[interface]), float(temperatures[interface])),
        predictions,
        (residual, float(left_flux - right_flux), 0.0),
        (float(left_flux), float(-right_flux)),
        (),
        1.0 / n_cells,
        canonical_digest(
            {
                "backend": "p1-line-fem",
                "nodes": nodes,
                "temperatures": temperatures,
                "conductivity_right": conductivity_right,
                "identities": study.identities.identity,
            }
        ),
    )
    return FEMResult(forward, tuple(nodes), tuple(float(item) for item in temperatures), residual)


def manufactured_solution_refinement(levels: tuple[int, ...] = (4, 8, 16, 32)) -> RefinementEvidence:
    """Verify P1 FEM against ``u=x(1-x)``, satisfying ``-u''=2`` on ``[0,1]``."""

    if len(levels) < 2 or any(level < 2 for level in levels) or any(b <= a for a, b in zip(levels, levels[1:])):
        raise EvidenceFailure("E_REFINEMENT_LEVELS", "refinement levels must be strictly increasing")
    errors: list[float] = []
    scales: list[float] = []
    gauss_points = np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)])
    gauss_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    for n_cells in levels:
        nodes = np.linspace(0.0, 1.0, n_cells + 1)
        h = 1.0 / n_cells
        stiffness = np.zeros((n_cells + 1, n_cells + 1))
        load = np.zeros(n_cells + 1)
        local_stiffness = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        local_load = np.array([h, h])  # integral of 2*N_i over a cell
        for cell in range(n_cells):
            stiffness[cell : cell + 2, cell : cell + 2] += local_stiffness
            load[cell : cell + 2] += local_load
        solution = np.zeros(n_cells + 1)
        interior = np.arange(1, n_cells)
        solution[interior] = np.linalg.solve(stiffness[np.ix_(interior, interior)], load[interior])
        squared_error = 0.0
        for cell in range(n_cells):
            x0, x1 = nodes[cell], nodes[cell + 1]
            for point, weight in zip(gauss_points, gauss_weights, strict=True):
                x = 0.5 * ((1.0 - point) * x0 + (1.0 + point) * x1)
                numerical = 0.5 * ((1.0 - point) * solution[cell] + (1.0 + point) * solution[cell + 1])
                exact = x * (1.0 - x)
                squared_error += weight * (numerical - exact) ** 2 * h / 2.0
        scales.append(h)
        errors.append(math.sqrt(squared_error))
    orders = tuple(
        math.log(errors[index] / errors[index + 1]) / math.log(scales[index] / scales[index + 1])
        for index in range(len(errors) - 1)
    )
    return RefinementEvidence(
        tuple(scales),
        tuple(errors),
        orders,
        "u(x)=x(1-x)",
        "P1 Galerkin solve of -u''=2 with homogeneous Dirichlet boundaries",
    )


def build_validated_surrogate(study: ReferenceStudy, *, training_points: int = 65) -> ValidatedSurrogate:
    if training_points < 9:
        raise ResourceFailure("E_SURROGATE_BUDGET", "at least nine training points are required")
    low, high = study.conductivity_bounds
    grid = np.linspace(low, high, training_points)
    predictions = np.array(
        [solve_reference_forward(study, float(value)).observation_predictions_K for value in grid],
        dtype=float,
    )
    held_out = 0.5 * (grid[:-1] + grid[1:])
    maximum_error = 0.0
    for value in held_out:
        truth = solve_reference_forward(study, float(value)).observation_predictions_K
        estimate = tuple(
            float(np.interp(value, grid, predictions[:, column])) for column in range(predictions.shape[1])
        )
        maximum_error = max(maximum_error, max(abs(a - b) for a, b in zip(truth, estimate, strict=True)))
    validation_payload = {
        "training_grid": grid,
        "predictions": predictions,
        "held_out_grid": held_out,
        "held_out_max_error_K": maximum_error,
        "truth_solver": study.identities.solver,
        "data": study.identities.data,
    }
    return ValidatedSurrogate(
        tuple(float(item) for item in grid),
        tuple(tuple(float(value) for value in row) for row in predictions),
        (low, high),
        maximum_error,
        study.identities.solver,
        canonical_digest(validation_payload),
    )


def evaluate_with_surrogate(
    study: ReferenceStudy,
    conductivity_right: float,
    surrogate: ValidatedSurrogate,
    *,
    maximum_error_K: float = 0.25,
    drift_detected: bool = False,
    uncertainty_limit_K: float = 0.25,
) -> SurrogateDecision:
    reasons = []
    if surrogate.truth_solver_identity != study.identities.solver:
        reasons.append("truth-solver-identity-mismatch")
    if surrogate.held_out_max_error_K > maximum_error_K:
        reasons.append("held-out-error-exceeds-threshold")
    if not surrogate.training_domain[0] <= conductivity_right <= surrogate.training_domain[1]:
        reasons.append("outside-training-domain")
    if drift_detected:
        reasons.append("drift-detected")
    if surrogate.held_out_max_error_K > uncertainty_limit_K:
        reasons.append("uncertainty-bound-exceeds-limit")
    if reasons:
        predictions = solve_reference_forward(study, conductivity_right).observation_predictions_K
        used = False
        reason = ";".join(reasons)
    else:
        grid = np.asarray(surrogate.conductivity_grid)
        values = np.asarray(surrogate.prediction_grid)
        predictions = tuple(
            float(np.interp(conductivity_right, grid, values[:, column])) for column in range(values.shape[1])
        )
        used = True
        reason = "validated-within-domain"
    receipt = canonical_digest(
        {
            "study": study.identities.identity,
            "surrogate_validation": surrogate.validation_identity,
            "conductivity_right": conductivity_right,
            "used_surrogate": used,
            "reason": reason,
            "predictions": predictions,
        }
    )
    return SurrogateDecision(predictions, used, reason, surrogate.held_out_max_error_K, not used, receipt)


def _weighted_quantile(grid: np.ndarray, weights: np.ndarray, probability: float) -> float:
    index = int(np.searchsorted(np.cumsum(weights), probability, side="left"))
    return float(grid[min(index, len(grid) - 1)])


def posterior_point_digest(
    identities: ExecutionIdentities,
    index: int,
    value: float,
    weight: float,
) -> str:
    return canonical_digest({"identity_bundle": identities.identity, "index": index, "value": value, "weight": weight})


def run_bayesian_inversion(
    study: ReferenceStudy,
    *,
    grid_points: int = 381,
    credible_mass: float = 0.95,
    surrogate: ValidatedSurrogate | None = None,
) -> PosteriorSummary:
    """Run bounded deterministic posterior quadrature with fully bound samples."""

    if grid_points < 51 or grid_points > 10001:
        raise ResourceFailure("E_INFERENCE_BUDGET", "grid points must be between 51 and 10001")
    if not 0 < credible_mass < 1:
        raise InferenceFailure("E_CREDIBLE_MASS", "credible mass must lie strictly between zero and one")
    low, high = study.conductivity_bounds
    grid = np.linspace(low, high, grid_points)
    log_mean = float(study.conductivity_prior["log_mean"])
    log_std = float(study.conductivity_prior["log_std"])
    log_prior = -np.log(grid * log_std * math.sqrt(2.0 * math.pi)) - 0.5 * ((np.log(grid) - log_mean) / log_std) ** 2
    log_likelihood = np.empty(grid_points)
    surrogate_uses = 0
    fallbacks = 0
    observed = np.array([item.temperature_K for item in study.data.observations])
    for index, conductivity in enumerate(grid):
        if surrogate is None:
            prediction = solve_reference_forward(study, float(conductivity)).observation_predictions_K
        else:
            decision = evaluate_with_surrogate(study, float(conductivity), surrogate)
            prediction = decision.predictions_K
            surrogate_uses += int(decision.used_surrogate)
            fallbacks += int(decision.authoritative_fallback)
        residual = (observed - np.asarray(prediction)) / study.noise_sigma_K
        log_likelihood[index] = -0.5 * float(residual @ residual) - len(residual) * math.log(
            study.noise_sigma_K * math.sqrt(2.0 * math.pi)
        )
    log_posterior = log_prior + log_likelihood
    weights = np.exp(log_posterior - float(np.max(log_posterior)))
    weights /= float(np.sum(weights))
    prior_weights = np.exp(log_prior - float(np.max(log_prior)))
    prior_weights /= float(np.sum(prior_weights))
    tail = (1.0 - credible_mass) / 2.0
    interval = (_weighted_quantile(grid, weights, tail), _weighted_quantile(grid, weights, 1.0 - tail))
    mean = float(weights @ grid)
    map_value = float(grid[int(np.argmax(weights))])
    information_gain = float(np.sum(weights * np.log(np.maximum(weights, 1.0e-300) / prior_weights)))
    edge_count = max(1, grid_points // 50)
    boundary_mass = float(np.sum(weights[:edge_count]) + np.sum(weights[-edge_count:]))
    width_ratio = (interval[1] - interval[0]) / (high - low)
    warning = None
    if width_ratio >= 0.5:
        warning = "posterior remains broad relative to the declared prior domain"
    elif boundary_mass >= 0.1:
        warning = "posterior mass is concentrated near a prior boundary"
    samples = tuple(
        PosteriorPointResult(
            float(value),
            float(weight),
            posterior_point_digest(study.identities, index, float(value), float(weight)),
        )
        for index, (value, weight) in enumerate(zip(grid, weights, strict=True))
    )
    truth_recovered = None
    if study.data.truth_conductivity is not None:
        truth_recovered = interval[0] <= study.data.truth_conductivity <= interval[1]
    summary_payload = {
        "mean": mean,
        "map_value": map_value,
        "credible_interval": interval,
        "credible_mass": credible_mass,
        "truth_recovered": truth_recovered,
        "information_gain_nats": information_gain,
        "prior_boundary_mass": boundary_mass,
        "identifiability_warning": warning,
        "samples": samples,
        "surrogate_uses": surrogate_uses,
        "authoritative_fallbacks": fallbacks,
        "identities": study.identities,
    }
    return PosteriorSummary(
        mean,
        map_value,
        interval,
        credible_mass,
        truth_recovered,
        information_gain,
        boundary_mass,
        warning,
        samples,
        surrogate_uses,
        fallbacks,
        study.identities,
        canonical_digest(summary_payload),
    )


def with_coupling_mode(problem: Mapping[str, Any], mode: CouplingMode | str) -> dict[str, Any]:
    """Return a plain problem copy with only the declared coupling strategy changed."""

    mode = CouplingMode(mode)
    copied = json.loads(json.dumps(problem))
    couplings = [item for item in copied.get("operators", ()) if item.get("kind") == "coupling"]
    if len(couplings) != 1:
        raise CompileFailure("E_REFERENCE_SHAPE", "reference problem requires one coupling operator")
    couplings[0]["discretization"] = mode.value
    return copied


__all__ = [
    "CancellationFailure",
    "CompileFailure",
    "CouplingIteration",
    "CouplingMode",
    "DomainFailure",
    "EvidenceFailure",
    "ExecutionIdentities",
    "FEMResult",
    "ForwardResult",
    "InferenceFailure",
    "ObservationValue",
    "PosteriorPointResult",
    "PosteriorSummary",
    "REFERENCE_PROFILE",
    "ReferenceData",
    "ReferenceExecutionError",
    "ReferenceStudy",
    "RefinementEvidence",
    "ResourceFailure",
    "SolverFailure",
    "SurrogateDecision",
    "ValidatedSurrogate",
    "build_validated_surrogate",
    "canonical_digest",
    "evaluate_with_surrogate",
    "manufactured_solution_refinement",
    "posterior_point_digest",
    "prepare_reference_study",
    "run_bayesian_inversion",
    "solve_reference_fem",
    "solve_reference_forward",
    "with_coupling_mode",
]
