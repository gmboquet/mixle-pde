"""Real, executable cross-repo integration test for the composite-heat reference pipeline.

Four sibling packages each ship a module whose own docstring claims to be one layer of the
same "multiphysics inversion reference" scenario:

- ``mixle_physics.build_composite_heat_inversion`` -- solver-free physical spec (no mesh,
  weak form, solver, or sampler), emitting a ``CON-PHYSICS-PROBLEM-V1`` contract.
- ``mixle_sim.compile_reference_problem`` -- geometry/mesh/weak-form/coupling compilation
  (no numerical solver or inference), consuming that physics contract and emitting a
  ``CON-MATH-PROBLEM-V1`` math problem.
- ``mixle_pde.multiphysics_reference``/``reference_lifecycle`` -- actual P1 FEM execution,
  conservative monolithic/partitioned coupling, manufactured-solution refinement evidence, and
  bounded Bayesian inversion, consuming that math problem.
- ``mixle_numerics.multiphysics_reference`` (mixle-discrete) -- independent evidence
  verification that never imports a physical law, a mesh, or a PDE solver, re-deriving residual/
  conservation/interface/refinement/coupling/posterior-recovery/identifiability checks straight
  from the raw finite evidence mixle-pde's own execution produced.

Despite the matching contract tags, matching entity ids ("conductivity_right",
"left-right-thermal-interface", ...), and matching dataclass field names
(``mixle_pde.multiphysics_reference.ExecutionIdentities`` vs.
``mixle_numerics.multiphysics_reference.IdentityBundle``; the ``discrete_evidence`` dict
``ReferenceStudyStore.run`` assembles vs. ``mixle_numerics.multiphysics_reference.
ReferenceEvidence``'s constructor kwargs, field-for-field identical), none of the four repos
ever actually imports another: each repo's own test file hand-copies the adjacent contract as a
literal dict/fixture instead of calling the producing package, so a real field rename on either
side of a boundary would silently desync two hand-maintained copies rather than fail a test.

This file is the missing wiring: it runs physics -> sim -> pde -> discrete in one process, for
both coupling strategies, and asserts real numerical/structural invariants at every handoff
boundary -- not merely that the chain completes without raising.

Requires the optional ``mixle-pde[integration]`` extra (``mixle-physics``, ``mixle-sim
[integration]``, ``mixle-numerics``); skipped entirely when those siblings are not installed,
matching the precedent ``mixle_sim.benchmarks``/``tests/test_package_boundaries.py`` already
established for the same lazy, extras-gated cross-repo pattern.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest

mixle_physics = pytest.importorskip("mixle_physics", reason="requires the mixle-pde[integration] extra")
mixle_sim = pytest.importorskip("mixle_sim", reason="requires the mixle-pde[integration] extra")
mixle_numerics_problem = pytest.importorskip(
    "mixle_numerics.problem", reason="requires the mixle-pde[integration] extra"
)
mixle_numerics_reference = pytest.importorskip(
    "mixle_numerics.multiphysics_reference", reason="requires the mixle-pde[integration] extra"
)

from mixle_pde.multiphysics_reference import (  # noqa: E402
    ObservationValue,
    ReferenceData,
    build_validated_surrogate,
    manufactured_solution_refinement,
    prepare_reference_study,
    run_bayesian_inversion,
    solve_reference_fem,
)
from mixle_pde.reference_lifecycle import ReferenceStudyStore  # noqa: E402

STRATEGIES = ("monolithic", "partitioned")
TRUTH_CONDUCTIVITY = 0.5


def _observations() -> ReferenceData:
    return ReferenceData(
        observations=(
            ObservationValue(position_m=0.7, temperature_K=348.2),
            ObservationValue(position_m=0.9, temperature_K=315.8),
        ),
        truth_conductivity=TRUTH_CONDUCTIVITY,
    )


def _compile(strategy: str) -> tuple[dict[str, Any], str]:
    """Stage 1 -> 2: physics spec compiled by sim into a math problem, plus its discrete hash."""

    specification = mixle_physics.build_composite_heat_inversion()
    physics_contract = specification.theory.to_contract_dict(specification.need)
    assert physics_contract["contract"] == "CON-PHYSICS-PROBLEM-V1"

    compilation = mixle_sim.compile_reference_problem(physics_contract, strategy)
    # ReferenceCompilation.__post_init__ already refuses a lossy lowering or missing
    # conservation evidence; re-asserted explicitly here as a boundary contract, not just an
    # absence of an exception.
    assert compilation.compilation.receipt.unsupported_or_dropped == ()
    assert "conservation" in {item["kind"] for item in compilation.compilation.problem["evidence_requests"]}

    math_problem = compilation.compilation.problem
    # mixle-pde's own prepare_reference_study requires this hash to come "from the Discrete
    # contract parser" -- i.e. produced here, by mixle-numerics, not invented by the caller.
    problem_hash = mixle_numerics_problem.MathematicalProblem.from_dict(math_problem).semantic_hash()
    return math_problem, problem_hash


def _verify_with_discrete(math_problem: dict[str, Any], result: dict[str, Any]) -> Any:
    """Stage 3 -> 4: mixle-pde's result handed to mixle-discrete's independent verifier."""

    negotiated = mixle_numerics_reference.negotiate_problem(
        math_problem, mixle_numerics_reference.reference_problem_support()
    )
    negotiated.require_supported()

    evidence_payload = result["discrete_evidence"]
    evidence = mixle_numerics_reference.ReferenceEvidence(
        residual_vector=tuple(evidence_payload["residual_vector"]),
        oriented_interface_fluxes=tuple(evidence_payload["oriented_interface_fluxes"]),
        interface_values=tuple(evidence_payload["interface_values"]),
        refinement_scales=tuple(evidence_payload["refinement_scales"]),
        refinement_errors=tuple(evidence_payload["refinement_errors"]),
        coupling_residual_history=tuple(evidence_payload["coupling_residual_history"]),
        posterior=tuple(mixle_numerics_reference.PosteriorPoint(**item) for item in evidence_payload["posterior"]),
        truth_value=evidence_payload["truth_value"],
        prior_bounds=tuple(evidence_payload["prior_bounds"]),
        identifiability_warning=evidence_payload["identifiability_warning"],
        diagnostics=evidence_payload["diagnostics"],
    )
    identities = mixle_numerics_reference.IdentityBundle(**result["identities"])
    assert identities.problem == negotiated.problem.semantic_hash()
    return mixle_numerics_reference.verify_reference_evidence(negotiated.problem, evidence, identities)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_physics_spec_compiles_and_solves_and_is_independently_verified(strategy: str) -> None:
    """Run the full chain and check a real numerical claim at every boundary."""

    math_problem, problem_hash = _compile(strategy)
    data = _observations()

    # Stage 2 -> 3: sim's compiled math problem accepted and executed by pde.
    study = prepare_reference_study(math_problem, problem_hash, data, code_revision="mp0045-integration-test")
    assert study.mode.value == strategy

    fem = solve_reference_fem(study, TRUTH_CONDUCTIVITY, cells_per_region=8)
    assert fem.algebraic_residual < 1e-8
    left_interface_temperature, right_interface_temperature = fem.forward.interface_temperatures_K
    assert left_interface_temperature == pytest.approx(right_interface_temperature, abs=1e-9)

    refinement = manufactured_solution_refinement()
    assert all(1.8 <= order <= 2.2 for order in refinement.observed_orders), refinement.observed_orders

    surrogate = build_validated_surrogate(study)
    assert surrogate.held_out_max_error_K < 0.05

    posterior = run_bayesian_inversion(study, grid_points=201, surrogate=surrogate)
    assert posterior.truth_recovered
    assert posterior.credible_interval[0] < TRUTH_CONDUCTIVITY < posterior.credible_interval[1]
    assert posterior.mean == pytest.approx(TRUTH_CONDUCTIVITY, abs=0.05)

    # Stage 3 -> 4: pde's own raw evidence, independently re-verified by discrete.
    coupling_history = tuple(item.flux_residual_W_per_m2 for item in fem.forward.coupling_history) or (0.0,)
    result = {
        "identities": asdict(study.identities),
        "discrete_evidence": {
            "residual_vector": list(fem.forward.residual_vector),
            "oriented_interface_fluxes": list(fem.forward.oriented_interface_fluxes),
            "interface_values": list(fem.forward.interface_temperatures_K),
            "refinement_scales": list(refinement.scales),
            "refinement_errors": list(refinement.l2_errors),
            "coupling_residual_history": list(coupling_history),
            "posterior": [
                {"value": item.value, "weight": item.weight, "provenance_digest": item.provenance_digest}
                for item in posterior.samples
            ],
            "truth_value": data.truth_conductivity,
            "prior_bounds": list(study.conductivity_bounds),
            "identifiability_warning": posterior.identifiability_warning,
            "diagnostics": {
                "posterior_summary_identity": posterior.summary_identity,
                "forward_solution_identity": fem.forward.solution_identity,
            },
        },
    }
    report = _verify_with_discrete(math_problem, result)
    assert report.accepted, report.failures
    # Every named check, not only the aggregate -- a verifier that happened to accept for the
    # wrong reason is exactly what an aggregate-only assertion would hide.
    assert all(report.checks.values()), report.checks
    assert report.metrics["credible_low"] < TRUTH_CONDUCTIVITY < report.metrics["credible_high"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_reference_study_store_lifecycle_output_is_accepted_by_discrete(strategy: str, tmp_path: Any) -> None:
    """The durable create/run/retrieve lifecycle -- the realistic production path -- end to end."""

    math_problem, problem_hash = _compile(strategy)
    data = _observations()

    store = ReferenceStudyStore(tmp_path)
    study_id = store.create(
        math_problem,
        problem_hash,
        data,
        code_revision="mp0045-integration-test",
        idempotency_key=f"mp0045-{strategy}",
    )
    outcome = store.run(study_id)
    assert outcome["status"] == "complete", outcome.get("failure")
    result = store.retrieve(study_id)

    report = _verify_with_discrete(math_problem, result)
    assert report.accepted, report.failures
    assert result["posterior"]["truth_recovered"] is True


def test_monolithic_and_partitioned_strategies_agree_on_the_recovered_posterior() -> None:
    """Two independent numerical paths (direct analytic solve vs. partitioned fixed-point
    iteration) solving the same physical system should land on the same inference -- a real
    physical consistency check only visible when both strategies are run and compared."""

    means = {}
    for strategy in STRATEGIES:
        math_problem, problem_hash = _compile(strategy)
        data = _observations()
        study = prepare_reference_study(math_problem, problem_hash, data, code_revision="mp0045-integration-test")
        surrogate = build_validated_surrogate(study)
        posterior = run_bayesian_inversion(study, grid_points=201, surrogate=surrogate)
        means[strategy] = posterior.mean

    assert means["monolithic"] == pytest.approx(means["partitioned"], rel=1e-4)


def test_physics_contract_rejects_the_pipeline_before_sim_on_a_non_conservative_interface() -> None:
    """A negative control: the physics layer's own validity gate, not sim silently accepting a
    physically-invalid contract it was never asked to check."""

    with pytest.raises(mixle_physics.ReferenceSpecificationError, match="E_INTERFACE_MISSING"):
        mixle_physics.build_composite_heat_inversion(include_interface=False)
