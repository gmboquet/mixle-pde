"""Capability negotiation for solver-neutral mathematical PDE problems.

Canonical physics and simulation specifications belong to mixle-physics and
mixle-sim. This module is deliberately only a compatibility boundary for
CON-MATH-PROBLEM-V1 dictionaries presented to a mixle-pde backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PDEBackendProfile:
    id: str
    operator_kinds: frozenset[str]
    discretizations: frozenset[str]
    objective_senses: frozenset[str]
    mesh_cell_types: frozenset[str] = frozenset()
    evidence_kinds: frozenset[str] = frozenset({"residual", "convergence"})


@dataclass(frozen=True)
class PDECompatibilityReport:
    backend_id: str
    problem_id: str
    unsupported_features: tuple[str, ...]
    required_evidence: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.unsupported_features


class UnsupportedPDEProblem(ValueError):
    def __init__(self, report: PDECompatibilityReport) -> None:
        self.report = report
        super().__init__(f"{report.backend_id} does not support {list(report.unsupported_features)}")


def inspect_math_problem(problem: Mapping[str, Any], profile: PDEBackendProfile) -> PDECompatibilityReport:
    """Report unsupported semantics without selecting or claiming a backend implicitly."""

    for field in ("id", "domains", "unknowns", "operators", "constraints", "solve_plan"):
        if field not in problem:
            raise ValueError(f"CON-MATH-PROBLEM-V1 missing required field {field!r}")
    unsupported: set[str] = set()
    for operator in problem.get("operators", []):
        kind = str(operator.get("kind", ""))
        discretization = operator.get("discretization")
        if kind not in profile.operator_kinds:
            unsupported.add(f"operator:{kind or '<missing>'}")
        if discretization and discretization not in profile.discretizations:
            unsupported.add(f"discretization:{discretization}")
    for objective in problem.get("objectives", []):
        sense = str(objective.get("sense", ""))
        if sense not in profile.objective_senses:
            unsupported.add(f"objective:{sense or '<missing>'}")
    for domain in problem.get("domains", []):
        cell_type = domain.get("properties", {}).get("mesh_cell_type")
        if cell_type and profile.mesh_cell_types and cell_type not in profile.mesh_cell_types:
            unsupported.add(f"mesh_cell_type:{cell_type}")

    requested_evidence = tuple(
        str(item["kind"]) for item in problem.get("evidence_requests", []) if item.get("required", True)
    )
    for kind in requested_evidence:
        if kind not in profile.evidence_kinds:
            unsupported.add(f"evidence:{kind}")
    return PDECompatibilityReport(
        backend_id=profile.id,
        problem_id=str(problem["id"]),
        unsupported_features=tuple(sorted(unsupported)),
        required_evidence=requested_evidence,
    )


def require_compatible(problem: Mapping[str, Any], profile: PDEBackendProfile) -> PDECompatibilityReport:
    report = inspect_math_problem(problem, profile)
    if not report.supported:
        raise UnsupportedPDEProblem(report)
    return report
