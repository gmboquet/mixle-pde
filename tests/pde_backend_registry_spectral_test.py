"""Tests wiring mixle-pde's pseudo-spectral Navier-Stokes kernel through the problem_adapter boundary
(MP-E5/PDE-E5, spectral-method family).

Mirrors the acceptance shape of tests/pde_backend_registry_test.py: one accepted study that actually runs
the underlying legacy solver (mixle_pde.spectral_flow.incompressible_ns_spectral) and checks its result
against the exact closed-form Taylor-Green vortex decay, and rejected studies proving require_compatible()
raises UnsupportedPDEProblem rather than the study being silently solved by a mismatched backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.pde_backend_registry import (
    PDEKernelRegistration,
    get_kernel_registration,
    list_kernel_registrations,
    run_math_problem,
)
from mixle_pde.problem_adapter import UnsupportedPDEProblem, require_compatible


def _problem(
    *,
    problem_id: str,
    operator_kind: str,
    discretization: str,
    mesh_cell_type: str,
    parameters: dict,
    objective_sense: str = "satisfy",
    evidence_kinds: tuple[str, ...] = (),
) -> dict:
    return {
        "id": problem_id,
        "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": mesh_cell_type}}],
        "unknowns": [{"id": "velocity", "domain_id": "domain"}],
        "operators": [
            {
                "id": f"{problem_id}-operator",
                "kind": operator_kind,
                "input_ids": ["velocity"],
                "output_ids": ["velocity"],
                "discretization": discretization,
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": objective_sense, "expression": {}}],
        "evidence_requests": [{"kind": kind, "required": True} for kind in evidence_kinds],
        "solve_plan": {"parameters": parameters},
    }


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------
def test_flow_spectral_ns_is_registered_alongside_the_existing_five_kernels():
    registrations = {reg.profile.id: reg for reg in list_kernel_registrations()}
    assert "flow-spectral-ns" in registrations
    # the five kernels wired under PDE-E5 stay untouched
    for existing_id in (
        "fem-p1-simplex",
        "wave-fd-leapfrog",
        "flow-fd-streamfunction",
        "em-fdtd-yee",
        "transport-fd-advdiff",
    ):
        assert existing_id in registrations


def test_flow_spectral_ns_declares_typed_ports_artifact_and_string_keyed_invocation():
    registration = get_kernel_registration("flow-spectral-ns")
    assert isinstance(registration, PDEKernelRegistration)
    assert registration.ports
    assert any(port.role == "input" for port in registration.ports)
    assert any(port.role == "output" for port in registration.ports)
    for port in registration.ports:
        assert port.units
    assert registration.artifact.kind
    assert registration.artifact.units
    assert isinstance(registration.invoke_key, str)
    assert get_kernel_registration(registration.profile.id) is registration


# ---------------------------------------------------------------------------
# flow-spectral-ns
# ---------------------------------------------------------------------------
def test_spectral_ns_accepted_study_matches_exact_taylor_green_decay():
    problem = _problem(
        problem_id="spectral-ns-study",
        operator_kind="time_stepping",
        discretization="spectral-fourier-rk4",
        mesh_cell_type="periodic_grid",
        parameters={"grid_size": 48, "dim": 2, "viscosity": 0.05, "dt": 0.001, "n_steps": 300},
        evidence_kinds=("residual", "convergence"),
    )
    result = run_math_problem(problem, "flow-spectral-ns")
    assert result.compatibility_report.supported
    assert isinstance(result.solution, np.ndarray)
    assert result.solution.shape == (2, 48, 48)
    assert np.all(np.isfinite(result.solution))
    assert result.evidence["convergence"]["finite"] is True
    # agreement with the exact closed-form Taylor-Green decay, not merely "it ran"
    assert result.evidence["residual"] < 1e-8


def test_spectral_ns_3d_accepted_study_matches_exact_abc_beltrami_decay():
    problem = _problem(
        problem_id="spectral-ns-3d-study",
        operator_kind="time_stepping",
        discretization="spectral-fourier-rk4",
        mesh_cell_type="periodic_grid",
        parameters={"grid_size": 24, "dim": 3, "viscosity": 0.1, "dt": 0.002, "n_steps": 100},
        evidence_kinds=("residual", "convergence"),
    )
    result = run_math_problem(problem, "flow-spectral-ns")
    assert result.compatibility_report.supported
    assert result.solution.shape == (3, 24, 24, 24)
    assert result.evidence["convergence"]["finite"] is True
    assert result.evidence["residual"] < 1e-8


def test_spectral_ns_rejects_unsupported_discretization_without_silently_downgrading():
    problem = _problem(
        problem_id="spectral-ns-study",
        operator_kind="time_stepping",
        discretization="FD-streamfunction-vorticity",  # belongs to flow-fd-streamfunction, not this backend
        mesh_cell_type="periodic_grid",
        parameters={"grid_size": 16},
        evidence_kinds=(),
    )
    registration = get_kernel_registration("flow-spectral-ns")
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "flow-spectral-ns")
    assert excinfo.value.report.unsupported_features == ("discretization:FD-streamfunction-vorticity",)
    # require_compatible alone rejects the same way, independent of invocation
    with pytest.raises(UnsupportedPDEProblem):
        require_compatible(problem, registration.profile)


def test_spectral_ns_rejects_unsupported_mesh_cell_type():
    problem = _problem(
        problem_id="spectral-ns-study",
        operator_kind="time_stepping",
        discretization="spectral-fourier-rk4",
        mesh_cell_type="structured_grid",  # not periodic_grid
        parameters={"grid_size": 16},
        evidence_kinds=(),
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "flow-spectral-ns")
    assert excinfo.value.report.unsupported_features == ("mesh_cell_type:structured_grid",)


def test_spectral_ns_rejects_evidence_kind_it_does_not_declare():
    problem = _problem(
        problem_id="spectral-ns-study",
        operator_kind="time_stepping",
        discretization="spectral-fourier-rk4",
        mesh_cell_type="periodic_grid",
        parameters={"grid_size": 16},
        evidence_kinds=("conservation",),  # flow-spectral-ns only declares residual/convergence
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "flow-spectral-ns")
    assert excinfo.value.report.unsupported_features == ("evidence:conservation",)


def test_spectral_ns_rejects_unsupported_operator_kind():
    problem = _problem(
        problem_id="spectral-ns-study",
        operator_kind="weak_form",  # flow-spectral-ns only declares "time_stepping"
        discretization="spectral-fourier-rk4",
        mesh_cell_type="periodic_grid",
        parameters={"grid_size": 16},
        evidence_kinds=(),
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "flow-spectral-ns")
    assert excinfo.value.report.unsupported_features == ("operator:weak_form",)
