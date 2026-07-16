from dataclasses import replace

import pytest

from mixle_pde.multiphysics_reference import (
    CompileFailure,
    ObservationValue,
    ReferenceData,
    SolverFailure,
    build_validated_surrogate,
    canonical_digest,
    evaluate_with_surrogate,
    manufactured_solution_refinement,
    prepare_reference_study,
    run_bayesian_inversion,
    solve_reference_fem,
    solve_reference_forward,
)
from mixle_pde.reference_lifecycle import ReferenceStudyStore, ResourceBudget


def _problem(mode: str = "monolithic", weak_discretization: str = "P1") -> dict:
    return {
        "id": f"composite-heat::{mode}",
        "domains": [
            {"id": "left-material", "kind": "mesh", "properties": {"mesh_cell_type": "line", "mesh_ref": "m"}},
            {"id": "right-material", "kind": "mesh", "properties": {"mesh_cell_type": "line", "mesh_ref": "m"}},
        ],
        "unknowns": [
            {"id": "temperature_left_h", "domain_id": "left-material", "value_kind": "field"},
            {"id": "temperature_right_h", "domain_id": "right-material", "value_kind": "field"},
            {
                "id": "conductivity_right",
                "domain_id": "right-material",
                "value_kind": "parameter",
                "bounds": [0.1, 2.0],
                "prior": {"family": "lognormal", "log_mean": -0.5108256238, "log_std": 0.45},
            },
        ],
        "operators": [
            {
                "id": "left-weak-heat",
                "kind": "weak_form",
                "input_ids": ["temperature_left_h", "conductivity_right"],
                "output_ids": ["temperature_left_h"],
                "discretization": weak_discretization,
            },
            {
                "id": "right-weak-heat",
                "kind": "weak_form",
                "input_ids": ["temperature_right_h", "conductivity_right"],
                "output_ids": ["temperature_right_h"],
                "discretization": weak_discretization,
            },
            {
                "id": "thermal-interface",
                "kind": "coupling",
                "input_ids": ["temperature_left_h"],
                "output_ids": ["temperature_right_h"],
                "discretization": mode,
            },
        ],
        "constraints": [],
        "objectives": [{"id": "posterior", "sense": "infer", "expression": {}}],
        "evidence_requests": [
            {"kind": "residual", "required": True},
            {"kind": "convergence", "required": True},
            {"kind": "conservation", "required": True},
        ],
        "solve_plan": {},
        "schema_version": "1.0.0",
        "metadata": {
            "fixed_parameters": {
                "conductivity_left": 2.0,
                "temperature_at_x0": 400.0,
                "temperature_at_x1": 300.0,
            },
            "observation_models": [
                {"id": "temperature-probes", "noise": {"family": "independent_gaussian", "sigma": 0.5}}
            ],
        },
    }


def _data() -> ReferenceData:
    return ReferenceData((ObservationValue(0.7, 348.2), ObservationValue(0.9, 315.8)), 0.5)


def _study(mode: str = "monolithic"):
    problem = _problem(mode)
    return prepare_reference_study(problem, canonical_digest(problem), _data(), code_revision="test-revision")


def test_negotiates_only_the_exact_reference_envelope() -> None:
    assert _study().compatibility.supported
    with pytest.raises(CompileFailure, match="E_UNSUPPORTED_PROBLEM"):
        problem = _problem(weak_discretization="spectral-element")
        prepare_reference_study(problem, canonical_digest(problem), _data(), code_revision="test-revision")


def test_monolithic_and_partitioned_paths_conserve_and_agree() -> None:
    monolithic = solve_reference_forward(_study("monolithic"), 0.5)
    partitioned = solve_reference_forward(_study("partitioned"), 0.5)

    assert monolithic.heat_flux_W_per_m2 == pytest.approx(80.0)
    assert monolithic.interface_temperatures_K == pytest.approx((380.0, 380.0))
    assert partitioned.heat_flux_W_per_m2 == pytest.approx(monolithic.heat_flux_W_per_m2, abs=1e-9)
    assert abs(sum(partitioned.oriented_interface_fluxes)) <= 1e-9
    assert len(partitioned.coupling_history) > 1
    assert partitioned.coupling_history[-1].flux_residual_W_per_m2 <= 1e-10
    with pytest.raises(SolverFailure, match="E_COUPLING_NONCONVERGENCE"):
        solve_reference_forward(_study("partitioned"), 0.5, maximum_iterations=1)


def test_p1_fem_and_manufactured_refinement_produce_independent_evidence() -> None:
    study = _study()
    fem = solve_reference_fem(study, 0.5, cells_per_region=4)
    exact = solve_reference_forward(study, 0.5)
    refinement = manufactured_solution_refinement()

    assert fem.algebraic_residual <= 1e-9
    assert fem.forward.observation_predictions_K == pytest.approx(exact.observation_predictions_K, abs=1e-10)
    assert max(fem.forward.residual_vector) <= 1e-9
    assert all(b < a for a, b in zip(refinement.l2_errors[:-1], refinement.l2_errors[1:], strict=True))
    assert min(refinement.observed_orders) >= 1.99


def test_bayesian_inversion_recovers_truth_and_binds_every_point() -> None:
    study = _study()
    surrogate = build_validated_surrogate(study)
    posterior = run_bayesian_inversion(study, grid_points=201, surrogate=surrogate)

    assert surrogate.held_out_max_error_K <= 0.25
    assert posterior.truth_recovered
    assert posterior.credible_interval[0] <= 0.5 <= posterior.credible_interval[1]
    assert posterior.surrogate_uses == 201
    assert all(len(item.provenance_digest) == 64 for item in posterior.samples)


def test_surrogate_falls_back_for_invalid_domain_drift_and_validation() -> None:
    study = _study()
    surrogate = build_validated_surrogate(study)
    valid = evaluate_with_surrogate(study, 0.5, surrogate)
    narrowed = replace(surrogate, training_domain=(0.2, 1.8))
    outside = evaluate_with_surrogate(study, 0.15, narrowed)
    drifted = evaluate_with_surrogate(study, 0.5, surrogate, drift_detected=True)
    invalid = evaluate_with_surrogate(study, 0.5, replace(surrogate, held_out_max_error_K=1.0))

    assert valid.used_surrogate
    assert all(item.authoritative_fallback for item in (outside, drifted, invalid))
    assert "outside-training-domain" in outside.reason


def test_durable_lifecycle_is_idempotent_cancellable_resumable_and_typed(tmp_path) -> None:
    store = ReferenceStudyStore(tmp_path)
    problem = _problem("partitioned")
    problem_hash = canonical_digest(problem)
    study_id = store.create(
        problem,
        problem_hash,
        _data(),
        code_revision="test-revision",
        idempotency_key="same-request",
        grid_points=101,
    )
    assert (
        store.create(
            problem,
            problem_hash,
            _data(),
            code_revision="test-revision",
            idempotency_key="same-request",
            grid_points=101,
        )
        == study_id
    )
    assert store.cancel(study_id)["status"] == "cancelled"
    assert store.resume(study_id)["status"] == "created"
    assert store.run(study_id)["status"] == "complete"
    result = store.retrieve(study_id)
    assert result["posterior"]["truth_recovered"]
    assert result["discrete_evidence"]["posterior"]

    failed_id = store.create(
        problem,
        problem_hash,
        _data(),
        code_revision="test-revision",
        idempotency_key="over-budget",
        grid_points=301,
        budget=ResourceBudget(maximum_grid_points=101),
    )
    assert store.run(failed_id)["failure"]["category"] == "resource"
    store.resume(failed_id, grid_points=101)
    assert store.run(failed_id)["status"] == "complete"
