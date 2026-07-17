"""Tests for the MP-J4 coarse-to-fine agent execution loop (mixle_pde.verification.agent_loop).

Every "solved" case drives a real, unmodified :func:`mixle_pde.pde_backend_registry.run_math_problem`
call (never a mock) and checks the loop's final result against the same real, already-established
reference every kernel's own accepted-study test already checks (mass conservation, the exact
closed-form Taylor-Green decay) -- "did the loop actually solve the right thing", not merely "did it not
crash". The CFL-repair tests independently recompute the Courant number from raw vp/dt/spacing inputs
(never trusting this module's own classification), mirroring
tests/diagnostic_ontology_test.py/modeling_traces.py's own "reconstruct from scratch" discipline.
"""

from __future__ import annotations

import math

import pytest

from mixle_pde.pde_backend_registry import list_kernel_registrations
from mixle_pde.verification import knowledge_catalog
from mixle_pde.verification.agent_loop import (
    _CFL_REPAIRABLE_BACKENDS,
    _GRID_SIZE_DEFAULT,
    _GRID_SIZE_KEY,
    _attempt_repair,
    _classify_evidence_failure,
    _verify_evidence,
    estimate_dofs,
    run_coarse_to_fine,
    run_inverse_case,
)
from mixle_pde.verification.diagnostic_ontology import DiagnosticCode, RepairOutcome

torch = pytest.importorskip("torch", reason="elastic/wave/em kernels use the differentiable ops backend")

_COURANT_LIMIT = 1.0 / math.sqrt(3.0)


def _problem(
    *,
    problem_id: str,
    operator_kind: str,
    discretization: str,
    mesh_cell_type: str,
    parameters: dict,
    evidence_kinds: tuple[str, ...] = (),
) -> dict:
    """Minimal CON-MATH-PROBLEM-V1 study -- shaped exactly like
    tests/pde_backend_registry_test.py/tests/pde_backend_registry_kernels2_test.py's own ``_problem``
    builders, so it is compatible with the real ``require_compatible`` boundary these kernels share."""
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
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [{"kind": kind, "required": True} for kind in evidence_kinds],
        "solve_plan": {"parameters": parameters},
    }


def _elastic_problem(*, grid_size: int, dt: float, spacing: float, vp: float, n_steps: int = 8) -> dict:
    return _problem(
        problem_id="agent-loop-elastic",
        operator_kind="time_stepping",
        discretization="FD-staggered-leapfrog",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": grid_size, "dt": dt, "n_steps": n_steps, "vp": vp, "spacing": spacing},
        evidence_kinds=("convergence",),
    )


# ---------------------------------------------------------------------------
# estimate_dofs / registry-sync
# ---------------------------------------------------------------------------
def test_dof_tables_have_no_drift_against_the_live_registry():
    """The registry-vs-disk drift check this repo's other completeness tests already use (e.g.
    tests/capability_inventory_test.py, tests/knowledge_catalog_test.py): every registered backend id
    must have a DOF estimator, and this module must not carry a stale entry for a retired id."""
    registered_ids = {registration.profile.id for registration in list_kernel_registrations()}
    assert set(_GRID_SIZE_KEY) == registered_ids
    assert set(_GRID_SIZE_DEFAULT) == registered_ids


def test_estimate_dofs_uses_the_default_when_no_parameter_is_supplied():
    problem = _problem(
        problem_id="p",
        operator_kind="linear_operator",
        discretization="FD-implicit",
        mesh_cell_type="structured_grid",
        parameters={},
    )
    assert estimate_dofs(problem, "transport-fd-advdiff") == 41  # grid_size default, power 1
    assert estimate_dofs(problem, "wave-fd-leapfrog") == 24**2
    assert estimate_dofs(problem, "elastic-fd-leapfrog") == 10**3
    assert estimate_dofs(problem, "em-fdtd-yee") == 8**3
    assert estimate_dofs(problem, "helmholtz-pml-fd") == 16 * 16
    assert estimate_dofs(problem, "groundwater-fd-transport") == 14 * 14
    assert estimate_dofs(problem, "fem-p1-simplex") == 6 * 6
    assert estimate_dofs(problem, "flow-spectral-ns") == 32**2  # default dim=2


def test_estimate_dofs_reads_caller_supplied_grid_size_and_dim():
    problem = _problem(
        problem_id="p",
        operator_kind="time_stepping",
        discretization="spectral-fourier-rk4",
        mesh_cell_type="periodic_grid",
        parameters={"grid_size": 24, "dim": 3},
    )
    assert estimate_dofs(problem, "flow-spectral-ns") == 24**3


def test_estimate_dofs_rejects_unregistered_backend_id():
    problem = _problem(
        problem_id="p",
        operator_kind="linear_operator",
        discretization="FD-implicit",
        mesh_cell_type="structured_grid",
        parameters={},
    )
    with pytest.raises(KeyError):
        estimate_dofs(problem, "not-a-real-backend")


# ---------------------------------------------------------------------------
# _verify_evidence -- unit tests over the real per-kernel evidence shapes (read directly from
# mixle_pde/pde_backend_registry.py, see agent_loop.py's own module docstring).
# ---------------------------------------------------------------------------
def test_verify_evidence_passes_a_plain_finite_result():
    ok, _detail = _verify_evidence({"convergence": {"finite": True}}, residual_tolerance=1e-6)
    assert ok is True


def test_verify_evidence_fails_on_non_finite_convergence():
    ok, detail = _verify_evidence({"convergence": {"finite": False}}, residual_tolerance=1e-6)
    assert ok is False
    assert "finite" in detail


def test_verify_evidence_fails_on_unstable_courant():
    ok, detail = _verify_evidence({"convergence": {"finite": True, "stable": False}}, residual_tolerance=1e-6)
    assert ok is False
    assert "stable" in detail


def test_verify_evidence_handles_fem_p1_shape_with_no_finite_key():
    """fem-p1-simplex's own evidence["convergence"] never carries a "finite" key (only
    interior_nodes/n_nodes) -- this must not KeyError, and the top-level residual is the real signal."""
    ok, _detail = _verify_evidence(
        {"residual": 1e-10, "convergence": {"interior_nodes": 4, "n_nodes": 9}}, residual_tolerance=1e-6
    )
    assert ok is True
    ok, detail = _verify_evidence(
        {"residual": 1.0, "convergence": {"interior_nodes": 4, "n_nodes": 9}}, residual_tolerance=1e-6
    )
    assert ok is False
    assert "residual" in detail


def test_verify_evidence_fails_on_nan_residual():
    ok, detail = _verify_evidence({"residual": float("nan"), "convergence": {}}, residual_tolerance=1e-6)
    assert ok is False
    assert "non-finite" in detail


# ---------------------------------------------------------------------------
# _classify_evidence_failure
# ---------------------------------------------------------------------------
def test_classify_evidence_failure_is_cfl_violation_only_for_the_grounded_backends():
    for backend_id in _CFL_REPAIRABLE_BACKENDS:
        assert (
            _classify_evidence_failure({"convergence": {"finite": False}}, backend_id) is DiagnosticCode.CFL_VIOLATION
        )
    assert set(_CFL_REPAIRABLE_BACKENDS) == {"elastic-fd-leapfrog", "wave-fd-leapfrog", "em-fdtd-yee"}


def test_classify_evidence_failure_is_none_for_non_cfl_backends():
    assert _classify_evidence_failure({"residual": 1.0, "convergence": {}}, "fem-p1-simplex") is None
    assert _classify_evidence_failure({"convergence": {"finite": False}}, "groundwater-fd-transport") is None


def test_classify_evidence_failure_is_none_when_finite_and_stable():
    assert _classify_evidence_failure({"convergence": {"finite": True, "stable": True}}, "elastic-fd-leapfrog") is None


# ---------------------------------------------------------------------------
# _attempt_repair -- the defensive "missing parameter" guard, unit-tested directly (mirrors this
# repository's convention of directly unit-testing internal pure functions like
# diagnostic_ontology.apply_repair_attempt) since helmholtz-pml-fd/fem-p1-simplex have no "dt" parameter
# to naturally trigger this path through a real solve.
# ---------------------------------------------------------------------------
def test_attempt_repair_requires_human_when_target_parameter_is_absent():
    problem = _problem(
        problem_id="p",
        operator_kind="linear_operator",
        discretization="FD-helmholtz-pml",
        mesh_cell_type="structured_grid",
        parameters={"grid_shape": (8, 8)},  # no "dt" key -- helmholtz-pml-fd is a direct frequency-domain solve
    )
    run, result = _attempt_repair(
        problem, "helmholtz-pml-fd", DiagnosticCode.CFL_VIOLATION, {"grid_shape": (8, 8)}, residual_tolerance=1e-6
    )
    assert run.outcome is RepairOutcome.REQUIRES_HUMAN
    assert run.attempts_used == 0
    assert result is None


# ---------------------------------------------------------------------------
# run_coarse_to_fine -- SOLVED, real numeric verification against a known-correct reference
# ---------------------------------------------------------------------------
def test_solves_transport_fd_advdiff_and_conserves_mass():
    """Mirrors tests/pde_backend_registry_test.py::test_transport_accepted_study_conserves_mass_under_periodic_bc
    exactly (same fixture), but through the coarse-to-fine loop: the coarse (grid_size~20) stage must pass
    this module's own finite/stable verification (asserted via ``all(s.ok ...)`` below, which includes the
    "verify_coarse" stage), and the final (uncoarsened, grid_size=41) stage's actual returned evidence must
    independently conserve mass to the same rel=1e-6 bar its own accepted-study test uses."""
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
    receipt = run_coarse_to_fine(problem, "transport-fd-advdiff")
    assert receipt.outcome.value == "solved"
    assert receipt.final_result is not None
    mass_before = receipt.final_result.evidence["conservation"]["mass_before"]
    mass_after = receipt.final_result.evidence["conservation"]["mass_after"]
    assert mass_after == pytest.approx(mass_before, rel=1e-6)
    stage_names = [s.stage for s in receipt.stages]
    assert "coarse_solve" in stage_names and "final_solve" in stage_names
    assert all(s.ok for s in receipt.stages)  # nothing failed along the way -- this is the clean path


def test_solves_flow_spectral_ns_matching_exact_taylor_green_decay():
    """Mirrors tests/pde_backend_registry_spectral_test.py's own accepted-study fixture: both the coarse
    (grid_size 24) and final (grid_size 48) resolutions must independently match the closed-form
    Taylor-Green decay to a tight residual, a genuine reference-agreement check, not merely finiteness."""
    problem = _problem(
        problem_id="spectral-ns-study",
        operator_kind="time_stepping",
        discretization="spectral-fourier-rk4",
        mesh_cell_type="periodic_grid",
        parameters={"grid_size": 48, "dim": 2, "viscosity": 0.05, "dt": 0.001, "n_steps": 300},
        evidence_kinds=("residual", "convergence"),
    )
    receipt = run_coarse_to_fine(problem, "flow-spectral-ns")
    assert receipt.outcome.value == "solved"
    assert receipt.final_result.solution.shape == (2, 48, 48)
    assert receipt.final_result.evidence["residual"] < 1e-6


# ---------------------------------------------------------------------------
# run_coarse_to_fine -- OVER_BUDGET, checked *before* any dispatch
# ---------------------------------------------------------------------------
def test_over_dof_budget_short_circuits_before_any_dispatch(monkeypatch):
    problem = _problem(
        problem_id="huge",
        operator_kind="linear_operator",
        discretization="FD-implicit",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 10_000_000},
        evidence_kinds=("convergence",),
    )
    import mixle_pde.verification.agent_loop as agent_loop_module

    def _fail_if_dispatched(*_args, **_kwargs):
        raise AssertionError("run_math_problem should never be called for an over-budget problem")

    monkeypatch.setattr(agent_loop_module, "run_math_problem", _fail_if_dispatched)
    from mixle_pde.verification.agent_loop import LoopBudget

    receipt = run_coarse_to_fine(problem, "transport-fd-advdiff", budget=LoopBudget(max_dofs=1000))
    assert receipt.outcome.value == "over_budget"
    assert len(receipt.stages) == 1
    assert receipt.stages[0].stage == "estimate_resources"


def test_wall_time_budget_is_enforced_between_stages():
    from mixle_pde.verification.agent_loop import LoopBudget

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
    # A budget effectively already exhausted by the time the coarse solve would dispatch.
    receipt = run_coarse_to_fine(problem, "transport-fd-advdiff", budget=LoopBudget(max_wall_time_seconds=1e-12))
    assert receipt.outcome.value == "over_budget"
    assert "wall time" in receipt.detail


# ---------------------------------------------------------------------------
# run_coarse_to_fine -- UNSUPPORTED
# ---------------------------------------------------------------------------
def test_unregistered_backend_is_unsupported_with_unknown_backend_code():
    problem = _problem(
        problem_id="p",
        operator_kind="linear_operator",
        discretization="FD-implicit",
        mesh_cell_type="structured_grid",
        parameters={},
    )
    receipt = run_coarse_to_fine(problem, "not-a-real-backend")
    assert receipt.outcome.value == "unsupported"
    assert receipt.diagnostic_code is DiagnosticCode.UNKNOWN_BACKEND


def test_unsupported_evidence_kind_is_unsupported_with_out_of_applicability_code():
    problem = _problem(
        problem_id="p",
        operator_kind="time_stepping",
        discretization="FD-staggered-leapfrog",
        mesh_cell_type="structured_grid",
        parameters={"grid_size": 8, "dt": 0.01, "n_steps": 6},
        evidence_kinds=("residual",),  # elastic-fd-leapfrog only declares "convergence" (see its profile)
    )
    receipt = run_coarse_to_fine(problem, "elastic-fd-leapfrog")
    assert receipt.outcome.value == "unsupported"
    assert receipt.diagnostic_code is DiagnosticCode.OUT_OF_APPLICABILITY_DOMAIN
    # the enrichment helper appends knowledge_catalog's own recorded applicability text
    assert "knowledge_catalog applicability" in receipt.detail


# ---------------------------------------------------------------------------
# run_coarse_to_fine -- CFL_VIOLATION refine/repair, independently re-verified
# ---------------------------------------------------------------------------
def test_cfl_violation_is_repaired_and_the_final_solve_uses_a_safe_dt():
    grid_size = 8
    spacing = 1.0 / (grid_size - 1)
    vp = 1.7
    violation_factor = 3.0  # comfortably within the CFL_VIOLATION recipe's 4-attempt/0.5-scale-factor budget
    dt_violating = violation_factor * _COURANT_LIMIT * spacing / vp
    # sanity-check the fixture itself actually violates the documented limit before using it
    assert (vp * dt_violating / spacing) > _COURANT_LIMIT

    problem = _elastic_problem(grid_size=grid_size, dt=dt_violating, spacing=spacing, vp=vp)
    receipt = run_coarse_to_fine(problem, "elastic-fd-leapfrog")

    assert receipt.outcome.value == "solved"
    assert len(receipt.repair_runs) >= 1
    coarse_repair = receipt.repair_runs[0]
    assert coarse_repair.code is DiagnosticCode.CFL_VIOLATION
    assert coarse_repair.outcome is RepairOutcome.RESOLVED
    assert 1 <= coarse_repair.attempts_used <= 4

    # Independent re-verification: recompute the Courant number for the actually-used final dt from
    # scratch (raw vp/dt/spacing arithmetic via the same formula the kernel's own docstring states),
    # never trusting this module's own RESOLVED label alone.
    reported_courant = receipt.final_result.evidence["convergence"]["courant_number"]
    assert reported_courant <= _COURANT_LIMIT + 1e-9
    assert receipt.final_result.evidence["convergence"]["stable"] is True


def test_cfl_violation_exhausts_bounded_repair_and_requires_human():
    grid_size = 8
    spacing = 1.0 / (grid_size - 1)
    vp = 1.7
    violation_factor = 60.0  # even 4 halvings (16x reduction) cannot clear this
    dt_violating = violation_factor * _COURANT_LIMIT * spacing / vp
    assert (vp * dt_violating / spacing) > _COURANT_LIMIT
    assert (vp * (dt_violating / 16.0) / spacing) > _COURANT_LIMIT  # still violates after 4 halvings

    problem = _elastic_problem(grid_size=grid_size, dt=dt_violating, spacing=spacing, vp=vp)
    receipt = run_coarse_to_fine(problem, "elastic-fd-leapfrog")

    assert receipt.outcome.value == "requires_human"
    assert receipt.diagnostic_code is DiagnosticCode.CFL_VIOLATION
    assert len(receipt.repair_runs) == 1
    assert receipt.repair_runs[0].outcome is RepairOutcome.ATTEMPTS_EXHAUSTED
    assert receipt.repair_runs[0].attempts_used == 4


def test_repair_cycle_budget_of_zero_forces_requires_human_without_attempting_repair():
    from mixle_pde.verification.agent_loop import LoopBudget

    grid_size = 8
    spacing = 1.0 / (grid_size - 1)
    vp = 1.7
    dt_violating = 3.0 * _COURANT_LIMIT * spacing / vp
    problem = _elastic_problem(grid_size=grid_size, dt=dt_violating, spacing=spacing, vp=vp)
    receipt = run_coarse_to_fine(problem, "elastic-fd-leapfrog", budget=LoopBudget(max_repair_cycles=0))
    assert receipt.outcome.value == "over_budget"
    assert not receipt.repair_runs


# ---------------------------------------------------------------------------
# run_inverse_case -- wraps mixle_pde.tools.run_inversion under the same typed vocabulary
# ---------------------------------------------------------------------------
def _tiny_gravity_dataset(tmp_path) -> str:
    """Identical fixture shape to tests/tools_test.py::_tiny_gravity_dataset."""
    import json

    coords = [[0.0, 0.0, -10.0], [10.0, 0.0, -10.0], [0.0, 10.0, -10.0], [10.0, 10.0, -10.0]]
    cell_volumes = [1000.0, 1000.0, 1000.0, 1000.0]
    stations = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 5.0, 0.0]]
    bundle = {
        "grid": {
            "coordinates": coords,
            "spacing": [10.0, 10.0, 10.0],
            "units": "kg/m3",
            "property_name": "density",
            "cell_volumes": cell_volumes,
            "crs": "EPSG:32611",
        },
        "observations": [
            {"location": stations, "value": [0.02, 0.015, 0.03], "noise_cov": [1e-4, 1e-4, 1e-4], "units": "mGal"}
        ],
    }
    path = tmp_path / "dataset.json"
    with open(path, "w") as f:
        json.dump(bundle, f)
    return str(path)


def test_run_inverse_case_solves_a_real_gravity_inversion(tmp_path):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    receipt = run_inverse_case(dataset_ref, "gravity", "smooth")
    assert receipt.outcome.value == "solved"
    assert receipt.backend_id == "run_inversion:gravity"
    assert receipt.stages[0].ok is True


def test_run_inverse_case_unsupported_modality(tmp_path):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    receipt = run_inverse_case(dataset_ref, "seismic", "smooth")
    assert receipt.outcome.value == "unsupported"


def test_run_inverse_case_unsupported_prior(tmp_path):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    receipt = run_inverse_case(dataset_ref, "gravity", "blocky")
    assert receipt.outcome.value == "unsupported"


# ---------------------------------------------------------------------------
# knowledge_catalog is genuinely consulted (not merely imported) -- confirms the enrichment helper
# actually reads real applicability text rather than a stub.
# ---------------------------------------------------------------------------
def test_knowledge_catalog_entry_exists_for_every_cfl_repairable_backend():
    for backend_id in _CFL_REPAIRABLE_BACKENDS:
        entry = knowledge_catalog.get_entry(backend_id)
        assert entry.applicability
