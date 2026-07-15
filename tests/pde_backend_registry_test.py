"""Tests wiring mixle-pde's legacy FD/FDTD/FEM kernels through the problem_adapter boundary (PDE-E5).

Each registered backend gets an equivalent CON-MATH-PROBLEM-V1-shaped study: one accepted invocation that
actually runs the underlying legacy solver and returns finite, internally-consistent evidence, and at least
one rejected invocation proving require_compatible() raises UnsupportedPDEProblem rather than the study
being silently solved by whichever backend happens to be registered.
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

torch = pytest.importorskip("torch", reason="wave/flow/EM kernels use the differentiable ops backend")


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
        "unknowns": [{"id": "field", "domain_id": "domain"}],
        "operators": [
            {
                "id": f"{problem_id}-operator",
                "kind": operator_kind,
                "input_ids": ["field"],
                "output_ids": ["field"],
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
def test_registers_at_least_one_wave_flow_em_transport_and_fem_kernel():
    registrations = {reg.profile.id: reg for reg in list_kernel_registrations()}
    assert "fem-p1-simplex" in registrations
    assert any("wave" in backend_id for backend_id in registrations)
    assert any("flow" in backend_id for backend_id in registrations)
    assert any(backend_id.startswith("em-") for backend_id in registrations)
    assert any("transport" in backend_id for backend_id in registrations)


def test_every_registration_declares_ports_artifact_and_string_keyed_invocation():
    for registration in list_kernel_registrations():
        assert isinstance(registration, PDEKernelRegistration)
        assert registration.ports, f"{registration.profile.id} declares no ports"
        assert any(port.role == "input" for port in registration.ports)
        assert any(port.role == "output" for port in registration.ports)
        for port in registration.ports:
            assert port.units  # every port declares units, even "1" for dimensionless
        assert registration.artifact.kind
        assert registration.artifact.units
        # the invocation is a registered string key, never a live callable stored on the record
        assert isinstance(registration.invoke_key, str)
        assert get_kernel_registration(registration.profile.id) is registration


# ---------------------------------------------------------------------------
# fem-p1-simplex
# ---------------------------------------------------------------------------
def test_fem_p1_accepted_study_solves_and_returns_small_residual():
    problem = _problem(
        problem_id="fem-poisson-study",
        operator_kind="weak_form",
        discretization="P1",
        mesh_cell_type="triangle",
        parameters={"grid_shape": (6, 6), "lengths": (1.0, 1.0), "diffusion": 1.0, "source": 1.0},
        evidence_kinds=("residual", "convergence"),
    )
    result = run_math_problem(problem, "fem-p1-simplex")
    assert result.compatibility_report.supported
    assert isinstance(result.solution, np.ndarray)
    assert np.all(np.isfinite(result.solution))
    assert result.evidence["residual"] < 1e-8
    assert result.evidence["convergence"]["interior_nodes"] > 0


def test_fem_p1_rejects_unsupported_discretization_without_silently_downgrading():
    problem = _problem(
        problem_id="fem-poisson-study",
        operator_kind="weak_form",
        discretization="spectral-element",
        mesh_cell_type="triangle",
        parameters={"grid_shape": (6, 6)},
        evidence_kinds=("residual",),
    )
    registration = get_kernel_registration("fem-p1-simplex")
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "fem-p1-simplex")
    assert excinfo.value.report.unsupported_features == ("discretization:spectral-element",)
    # require_compatible alone (the E6-reconciled boundary) rejects the same way, independent of invocation
    with pytest.raises(UnsupportedPDEProblem):
        require_compatible(problem, registration.profile)


def test_fem_p1_rejects_unsupported_mesh_cell_type():
    problem = _problem(
        problem_id="fem-poisson-study",
        operator_kind="weak_form",
        discretization="P1",
        mesh_cell_type="hexahedron",
        parameters={"grid_shape": (6, 6)},
        evidence_kinds=(),
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "fem-p1-simplex")
    assert excinfo.value.report.unsupported_features == ("mesh_cell_type:hexahedron",)


# ---------------------------------------------------------------------------
# wave-fd-leapfrog
# ---------------------------------------------------------------------------
def test_wave_accepted_study_steps_and_stays_finite():
    problem = _problem(
        problem_id="wave-fwd-study",
        operator_kind="time_stepping",
        discretization="FD-leapfrog",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 16, "dt": 0.02, "n_steps": 10, "wave_speed": 1.0},
        evidence_kinds=("convergence",),
    )
    result = run_math_problem(problem, "wave-fd-leapfrog")
    assert result.compatibility_report.supported
    assert result.solution.shape == (16 * 16,)
    assert result.evidence["convergence"]["finite"] is True


def test_wave_rejects_evidence_kind_it_does_not_declare():
    problem = _problem(
        problem_id="wave-fwd-study",
        operator_kind="time_stepping",
        discretization="FD-leapfrog",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 16},
        evidence_kinds=("residual",),  # wave-fd-leapfrog only declares "convergence"
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "wave-fd-leapfrog")
    assert excinfo.value.report.unsupported_features == ("evidence:residual",)


# ---------------------------------------------------------------------------
# flow-fd-streamfunction
# ---------------------------------------------------------------------------
def test_flow_accepted_study_steps_and_stays_finite():
    problem = _problem(
        problem_id="flow-fwd-study",
        operator_kind="time_stepping",
        discretization="FD-streamfunction-vorticity",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 16, "viscosity": 0.1, "dt": 0.01, "n_steps": 5},
        evidence_kinds=("convergence",),
    )
    result = run_math_problem(problem, "flow-fd-streamfunction")
    assert result.compatibility_report.supported
    assert result.solution.shape == (16 * 16,)
    assert result.evidence["convergence"]["finite"] is True


def test_flow_rejects_unsupported_operator_kind():
    problem = _problem(
        problem_id="flow-fwd-study",
        operator_kind="weak_form",  # flow-fd-streamfunction only declares "time_stepping"
        discretization="FD-streamfunction-vorticity",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 16},
        evidence_kinds=(),
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "flow-fd-streamfunction")
    assert excinfo.value.report.unsupported_features == ("operator:weak_form",)


# ---------------------------------------------------------------------------
# em-fdtd-yee
# ---------------------------------------------------------------------------
def test_em_accepted_study_steps_and_preserves_div_h():
    problem = _problem(
        problem_id="em-cavity-study",
        operator_kind="time_stepping",
        discretization="FDTD-yee",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 6, "n_steps": 4},
        evidence_kinds=("convergence", "conservation"),
    )
    result = run_math_problem(problem, "em-fdtd-yee")
    assert result.compatibility_report.supported
    assert result.evidence["convergence"]["finite"] is True
    # div(mu H) starts and stays at machine precision under the Yee scheme
    assert result.evidence["conservation"]["div_h_after"] < 1e-8


def test_em_rejects_unsupported_discretization():
    problem = _problem(
        problem_id="em-cavity-study",
        operator_kind="time_stepping",
        discretization="spectral",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 6},
        evidence_kinds=(),
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "em-fdtd-yee")
    assert excinfo.value.report.unsupported_features == ("discretization:spectral",)


# ---------------------------------------------------------------------------
# transport-fd-advdiff
# ---------------------------------------------------------------------------
def test_transport_accepted_study_conserves_mass_under_periodic_bc():
    problem = _problem(
        problem_id="transport-study",
        operator_kind="linear_operator",
        discretization="FD-implicit",
        mesh_cell_type="structured_grid",
        parameters={
            "grid_size": 41,
            "diffusivity": 0.01,
            "velocity": 0.5,
            "dt": 0.01,
            "n_steps": 20,
            "boundary": "periodic",
        },
        evidence_kinds=("convergence", "conservation"),
    )
    result = run_math_problem(problem, "transport-fd-advdiff")
    assert result.compatibility_report.supported
    assert result.evidence["convergence"]["finite"] is True
    mass_before = result.evidence["conservation"]["mass_before"]
    mass_after = result.evidence["conservation"]["mass_after"]
    assert mass_after == pytest.approx(mass_before, rel=1e-6)


def test_transport_rejects_unsupported_scheme_discretization():
    problem = _problem(
        problem_id="transport-study",
        operator_kind="linear_operator",
        discretization="finite-volume-weno",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 41},
        evidence_kinds=(),
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "transport-fd-advdiff")
    assert excinfo.value.report.unsupported_features == ("discretization:finite-volume-weno",)


# ---------------------------------------------------------------------------
# Existing public API stays intact: require_compatible/UnsupportedPDEProblem behavior is untouched
# ---------------------------------------------------------------------------
def test_require_compatible_still_accepts_a_fully_supported_problem_directly():
    registration = get_kernel_registration("fem-p1-simplex")
    problem = _problem(
        problem_id="fem-poisson-study",
        operator_kind="weak_form",
        discretization="P1",
        mesh_cell_type="triangle",
        parameters={},
        evidence_kinds=("residual",),
    )
    report = require_compatible(problem, registration.profile)
    assert report.supported
    assert report.required_evidence == ("residual",)
