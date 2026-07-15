"""Portable Sim-record execution and the first receipt-bearing legacy FEM migration path."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from mixle_pde.fem import assemble_simplex_load_vector, assemble_simplex_stiffness_matrix, fem_poisson
from mixle_pde.mesh import SimplexMesh

SIM_LINEAR_SCHEMA = "mixle.sim.finite-linear-system/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, np.ndarray):
        return _normalize(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return float(value)
    raise TypeError(f"value is not finite canonical JSON: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rational(value: Any, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be an integer or rational string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} is not a valid rational") from error


@dataclass(frozen=True)
class NativeBackendCapability:
    id: str
    version: str
    record_schemas: tuple[str, ...]
    problem_kinds: tuple[str, ...]
    coefficient_domains: tuple[str, ...]
    mesh_cell_types: tuple[str, ...]
    discretizations: tuple[str, ...]
    evidence_kinds: tuple[str, ...]
    maturity: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.maturity not in {"experimental", "prototype", "implemented", "verified"}:
            raise ValueError("unsupported backend maturity")
        for label, values in (
            ("record schema", self.record_schemas),
            ("problem kind", self.problem_kinds),
            ("coefficient domain", self.coefficient_domains),
            ("evidence kind", self.evidence_kinds),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"backend {label} values must be nonempty and unique")

    @property
    def identity(self) -> str:
        return _digest(asdict(self))


def native_backend_manifest() -> tuple[NativeBackendCapability, ...]:
    """Advertise only canonical paths exercised by the adapter conformance tests."""

    return (
        NativeBackendCapability(
            id="mixle-pde.native-rational-linear",
            version="0.8.0.dev0",
            record_schemas=(SIM_LINEAR_SCHEMA,),
            problem_kinds=("rational_linear_system",),
            coefficient_domains=("rational",),
            mesh_cell_types=("triangle", "tetrahedron", "simplex"),
            discretizations=("P1",),
            evidence_kinds=("residual", "legacy-parity"),
            maturity="implemented",
            limitations=(
                "dense floating-point execution of exact-rational input",
                "checked residual rather than an exact independent certificate",
                "intended as an integrated/reference path; general solve planning belongs to Mixle Discrete",
            ),
        ),
    )


@dataclass(frozen=True)
class CanonicalSolveReceipt:
    record_digest: str
    source_problem_digest: str
    backend_capability_digest: str
    outcome: str
    residual_norm: float
    relative_residual: float
    evidence_strength: str
    coefficient_conversion: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.record_digest, "record digest"),
            (self.source_problem_digest, "source problem digest"),
            (self.backend_capability_digest, "backend capability digest"),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be SHA-256")
        if self.outcome not in {"approximate", "failed"}:
            raise ValueError("native rational adapter reports only approximate or failed outcomes")
        if any(not math.isfinite(value) or value < 0 for value in (self.residual_norm, self.relative_residual)):
            raise ValueError("residual evidence must be nonnegative and finite")


@dataclass(frozen=True)
class CanonicalLinearSolution:
    unknowns: tuple[str, ...]
    values: tuple[float, ...]
    receipt: CanonicalSolveReceipt

    def __post_init__(self) -> None:
        if not self.unknowns or len(self.unknowns) != len(self.values):
            raise ValueError("solution unknown and value dimensions must agree")
        if len(self.unknowns) != len(set(self.unknowns)) or any(not math.isfinite(value) for value in self.values):
            raise ValueError("solution unknowns must be unique and values finite")


def _validated_linear_record(record: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, str]:
    _normalize(record)
    if record.get("schema") != SIM_LINEAR_SCHEMA:
        raise ValueError(f"unsupported simulation record schema {record.get('schema')!r}")
    if record.get("coefficient_domain") != "rational":
        raise ValueError("native adapter requires exact rational input coefficients")
    source_digest = record.get("source_problem_digest")
    if not isinstance(source_digest, str) or not _SHA256.fullmatch(source_digest):
        raise ValueError("source_problem_digest must be SHA-256")
    unknown_value = record.get("unknowns")
    if (
        not isinstance(unknown_value, (list, tuple))
        or not unknown_value
        or not all(isinstance(item, str) and item for item in unknown_value)
    ):
        raise ValueError("record requires nonempty string unknown ids")
    unknowns = tuple(unknown_value)
    if len(unknowns) != len(set(unknowns)):
        raise ValueError("record unknown ids must be unique")
    matrix_value = record.get("matrix")
    rhs_value = record.get("rhs")
    if not isinstance(matrix_value, (list, tuple)) or not matrix_value or not isinstance(rhs_value, (list, tuple)):
        raise ValueError("record requires matrix and rhs arrays")
    size = len(unknowns)
    if len(matrix_value) != size or len(rhs_value) != size:
        raise ValueError("matrix, rhs, and unknown dimensions do not agree")
    rows: list[list[float]] = []
    for row_index, row in enumerate(matrix_value):
        if not isinstance(row, (list, tuple)) or len(row) != size:
            raise ValueError("simulation matrix must be square")
        rows.append([float(_rational(value, f"matrix[{row_index}][{column}]")) for column, value in enumerate(row)])
    rhs = np.asarray([float(_rational(value, f"rhs[{index}]")) for index, value in enumerate(rhs_value)], dtype=float)
    matrix = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("rational conversion overflowed finite floating-point execution")
    return unknowns, matrix, rhs, source_digest


def solve_sim_linear_system(record: Mapping[str, Any]) -> CanonicalLinearSolution:
    unknowns, matrix, rhs, source_digest = _validated_linear_record(record)
    try:
        values = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("native adapter could not solve singular or invalid system") from error
    residual = matrix @ values - rhs
    residual_norm = float(np.linalg.norm(residual))
    relative = residual_norm / max(float(np.linalg.norm(rhs)), np.finfo(float).eps)
    capability = native_backend_manifest()[0]
    receipt = CanonicalSolveReceipt(
        record_digest=_digest(record),
        source_problem_digest=source_digest,
        backend_capability_digest=capability.identity,
        outcome="approximate",
        residual_norm=residual_norm,
        relative_residual=relative,
        evidence_strength="checked-residual",
        coefficient_conversion="exact rational parsed then converted to IEEE-754 float64",
        limitations=capability.limitations,
    )
    return CanonicalLinearSolution(unknowns, tuple(float(value) for value in values), receipt)


@dataclass(frozen=True)
class LegacyCanonicalPoissonResult:
    solution: np.ndarray
    mesh_digest: str
    system_record: Mapping[str, Any]
    solve_receipt: CanonicalSolveReceipt
    legacy_parity_max_abs: float

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.mesh_digest):
            raise ValueError("mesh digest must be SHA-256")
        if not math.isfinite(self.legacy_parity_max_abs) or self.legacy_parity_max_abs < 0:
            raise ValueError("legacy parity error must be nonnegative and finite")


def _fraction_string(value: float, maximum_denominator: int) -> str:
    fraction = Fraction(str(float(value))).limit_denominator(maximum_denominator)
    return f"{fraction.numerator}/{fraction.denominator}"


def solve_p1_poisson_canonical(
    nodes: Any,
    triangles: Any,
    source: float,
    *,
    conductivity: float = 1.0,
    dirichlet: Mapping[int, float] | None = None,
    maximum_denominator: int = 10**9,
) -> LegacyCanonicalPoissonResult:
    """Preserve the legacy P1 result while exposing portable canonical execution receipts."""

    if maximum_denominator <= 0:
        raise ValueError("maximum denominator must be positive")
    if not np.isscalar(source) or not np.isscalar(conductivity):
        raise ValueError("initial canonical P1 wrapper supports scalar source and conductivity")
    mesh = SimplexMesh(nodes, triangles)
    stiffness = assemble_simplex_stiffness_matrix(mesh, diffusion=float(conductivity)).tolil()
    rhs = assemble_simplex_load_vector(mesh, float(source))
    boundary = mesh.boundary_nodes()
    conditions = (
        {int(node): 0.0 for node in boundary}
        if dirichlet is None
        else {int(node): float(value) for node, value in dirichlet.items()}
    )
    for node, value in conditions.items():
        if node < 0 or node >= mesh.n_nodes or not math.isfinite(value):
            raise ValueError("Dirichlet condition contains invalid node or value")
        stiffness.rows[node] = [node]
        stiffness.data[node] = [1.0]
        rhs[node] = value
    matrix = stiffness.toarray()
    mesh_digest = _digest({"nodes": mesh.nodes, "simplices": mesh.simplices})
    source_problem_digest = _digest(
        {
            "mesh_digest": mesh_digest,
            "form": "scalar-P1-Poisson",
            "source": float(source),
            "conductivity": float(conductivity),
            "dirichlet": {str(key): value for key, value in sorted(conditions.items())},
        }
    )
    record = {
        "schema": SIM_LINEAR_SCHEMA,
        "id": "legacy-p1-poisson",
        "coefficient_domain": "rational",
        "source_problem_digest": source_problem_digest,
        "unknowns": [f"u{index}" for index in range(mesh.n_nodes)],
        "matrix": [[_fraction_string(value, maximum_denominator) for value in row] for row in matrix],
        "rhs": [_fraction_string(value, maximum_denominator) for value in rhs],
        "conversion": {
            "source": "mixle-pde legacy scalar P1 assembly",
            "maximum_denominator": maximum_denominator,
            "mesh_digest": mesh_digest,
        },
    }
    canonical = solve_sim_linear_system(record)
    solution = np.asarray(canonical.values)
    legacy = fem_poisson(nodes, triangles, source, conductivity=conductivity, dirichlet=dirichlet)
    parity = float(np.max(np.abs(solution - legacy)))
    return LegacyCanonicalPoissonResult(solution, mesh_digest, record, canonical.receipt, parity)


__all__ = [
    "CanonicalLinearSolution",
    "CanonicalSolveReceipt",
    "LegacyCanonicalPoissonResult",
    "NativeBackendCapability",
    "SIM_LINEAR_SCHEMA",
    "native_backend_manifest",
    "solve_p1_poisson_canonical",
    "solve_sim_linear_system",
]
