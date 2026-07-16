"""Tests wiring three more legacy kernels through the problem_adapter boundary (PDE-E5, second wave).

Mirrors the acceptance shape of tests/pde_backend_registry_test.py and tests/pde_backend_registry_spectral_test.py:
for each newly-registered backend, one accepted study that actually runs the underlying legacy solver
(mixle_pde.elastic.ElasticWave3D, mixle_pde.helmholtz_pml.solve_helmholtz_pml,
mixle_pde.groundwater.GroundwaterTransportOperator) and returns finite, internally-consistent evidence, and
one rejected study proving require_compatible() raises UnsupportedPDEProblem rather than the study being
silently solved by whichever backend happens to be registered.
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

torch = pytest.importorskip("torch", reason="elastic/helmholtz kernels use the differentiable ops backend")


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
def test_new_kernels_registered_alongside_the_existing_five():
    registrations = {reg.profile.id: reg for reg in list_kernel_registrations()}
    for new_id in ("elastic-fd-leapfrog", "helmholtz-pml-fd", "groundwater-fd-transport"):
        assert new_id in registrations
    # the five kernels wired under PDE-E5's first wave stay untouched
    for existing_id in (
        "fem-p1-simplex",
        "wave-fd-leapfrog",
        "flow-fd-streamfunction",
        "em-fdtd-yee",
        "transport-fd-advdiff",
    ):
        assert existing_id in registrations


def test_new_kernels_declare_typed_ports_artifact_and_string_keyed_invocation():
    for backend_id in ("elastic-fd-leapfrog", "helmholtz-pml-fd", "groundwater-fd-transport"):
        registration = get_kernel_registration(backend_id)
        assert isinstance(registration, PDEKernelRegistration)
        assert registration.ports, f"{backend_id} declares no ports"
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
# elastic-fd-leapfrog
# ---------------------------------------------------------------------------
def test_elastic_accepted_study_steps_and_stays_finite_and_stable():
    problem = _problem(
        problem_id="elastic-leapfrog-study",
        operator_kind="time_stepping",
        discretization="FD-staggered-leapfrog",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 10, "dt": 0.01, "n_steps": 8},
        evidence_kinds=("convergence",),
    )
    result = run_math_problem(problem, "elastic-fd-leapfrog")
    assert result.compatibility_report.supported
    assert isinstance(result.solution, np.ndarray)
    assert result.solution.shape == (9 * 10**3,)
    assert np.all(np.isfinite(result.solution))
    assert result.evidence["convergence"]["finite"] is True
    assert result.evidence["convergence"]["stable"] is True
    assert result.evidence["convergence"]["courant_number"] <= result.evidence["convergence"]["courant_limit"]


def test_elastic_rejects_evidence_kind_it_does_not_declare():
    problem = _problem(
        problem_id="elastic-leapfrog-study",
        operator_kind="time_stepping",
        discretization="FD-staggered-leapfrog",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 8},
        evidence_kinds=("residual",),  # elastic-fd-leapfrog only declares "convergence"
    )
    registration = get_kernel_registration("elastic-fd-leapfrog")
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "elastic-fd-leapfrog")
    assert excinfo.value.report.unsupported_features == ("evidence:residual",)
    # require_compatible alone (the E6-reconciled boundary) rejects the same way, independent of invocation
    with pytest.raises(UnsupportedPDEProblem):
        require_compatible(problem, registration.profile)


# ---------------------------------------------------------------------------
# helmholtz-pml-fd
# ---------------------------------------------------------------------------
def test_helmholtz_accepted_study_solves_and_returns_small_residual():
    problem = _problem(
        problem_id="helmholtz-pml-study",
        operator_kind="linear_operator",
        discretization="FD-helmholtz-pml",
        mesh_cell_type="structured_grid",
        parameters={"grid_shape": (16, 16), "omega": 6.0},
        evidence_kinds=("residual", "convergence"),
    )
    result = run_math_problem(problem, "helmholtz-pml-fd")
    assert result.compatibility_report.supported
    assert isinstance(result.solution, np.ndarray)
    assert np.iscomplexobj(result.solution)
    assert result.solution.shape == (256,)
    assert np.all(np.isfinite(result.solution))
    assert result.evidence["convergence"]["finite"] is True
    assert result.evidence["convergence"]["interior_nodes"] > 0
    assert result.evidence["residual"] < 1e-8


def test_helmholtz_rejects_unsupported_mesh_cell_type():
    problem = _problem(
        problem_id="helmholtz-pml-study",
        operator_kind="linear_operator",
        discretization="FD-helmholtz-pml",
        mesh_cell_type="triangle",  # helmholtz-pml-fd only declares "structured_grid"
        parameters={"grid_shape": (16, 16)},
        evidence_kinds=(),
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "helmholtz-pml-fd")
    assert excinfo.value.report.unsupported_features == ("mesh_cell_type:triangle",)


# ---------------------------------------------------------------------------
# groundwater-fd-transport
# ---------------------------------------------------------------------------
def test_groundwater_accepted_study_steps_and_stays_finite_with_near_conserved_mass():
    problem = _problem(
        problem_id="groundwater-transport-study",
        operator_kind="linear_operator",
        discretization="FD-implicit",
        mesh_cell_type="structured_grid",
        parameters={"grid_shape": (14, 14), "dt": 0.01, "n_steps": 20},
        evidence_kinds=("convergence", "conservation"),
    )
    result = run_math_problem(problem, "groundwater-fd-transport")
    assert result.compatibility_report.supported
    assert isinstance(result.solution, np.ndarray)
    assert result.solution.shape == (14 * 14,)
    assert result.evidence["convergence"]["finite"] is True
    mass_before = result.evidence["conservation"]["mass_before"]
    mass_after = result.evidence["conservation"]["mass_after"]
    # the default recharge is a zero-net injection/extraction doublet, so mass stays close to (not
    # exactly, since the advective term is non-conservative-form) conserved -- see the invoker's docstring
    assert mass_after == pytest.approx(mass_before, rel=0.05)


def test_groundwater_rejects_unsupported_discretization():
    problem = _problem(
        problem_id="groundwater-transport-study",
        operator_kind="linear_operator",
        discretization="finite-volume-weno",
        mesh_cell_type="structured_grid",
        parameters={"grid_shape": (14, 14)},
        evidence_kinds=(),
    )
    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "groundwater-fd-transport")
    assert excinfo.value.report.unsupported_features == ("discretization:finite-volume-weno",)
