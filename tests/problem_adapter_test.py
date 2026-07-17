from __future__ import annotations

import pytest

from mixle_pde.problem_adapter import PDEBackendProfile, UnsupportedPDEProblem, require_compatible


def _problem(discretization: str = "P1") -> dict:
    return {
        "id": "heat",
        "domains": [{"id": "plate", "kind": "mesh", "properties": {"mesh_cell_type": "triangle"}}],
        "unknowns": [{"id": "temperature", "domain_id": "plate"}],
        "operators": [
            {
                "id": "heat-weak-form",
                "kind": "weak_form",
                "input_ids": ["temperature"],
                "output_ids": ["temperature"],
                "discretization": discretization,
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [{"kind": "residual", "required": True}],
        "solve_plan": {},
    }


def _profile() -> PDEBackendProfile:
    return PDEBackendProfile(
        id="fem-profile",
        operator_kinds=frozenset({"weak_form"}),
        discretizations=frozenset({"P1", "P2"}),
        objective_senses=frozenset({"satisfy", "infer"}),
        mesh_cell_types=frozenset({"triangle", "tetrahedron"}),
    )


def test_accepts_only_explicitly_advertised_problem_features():
    report = require_compatible(_problem(), _profile())
    assert report.supported
    assert report.required_evidence == ("residual",)


def test_rejects_unsupported_discretization_without_degrading_it():
    with pytest.raises(UnsupportedPDEProblem) as error:
        require_compatible(_problem("spectral-element"), _profile())
    assert error.value.report.unsupported_features == ("discretization:spectral-element",)
