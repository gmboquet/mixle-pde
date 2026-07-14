"""P2 -- coupled multiphysics scenario runner: DoD test.

A 3-step scenario (dewatering head -> contaminant advection -> subsidence) wired with two coupling
edges. `simulate` must walk the DAG in topological order, rewrite each child's `inputs_ref` onto its
parent's content-hashed `result_ref`, record `{"parents": [...], "coupling": field_name}` in every
downstream artifact's provenance, and be byte-reproducible on re-run.
"""

from __future__ import annotations

import os

from mixle_pde.simulation_service import STORE_DIR_ENV, Scenario, ScenarioStep, read_result_header, simulate


def _dewatering_scenario() -> Scenario:
    flow_step = ScenarioStep(
        op="flow",
        inputs_ref="source:constant-recharge",
        params={"shape": (33,), "source": 1.0, "conductivity": 1.0, "dirichlet": 0.0, "spacing": 1.0},
    )
    transport_step = ScenarioStep(
        op="transport",
        inputs_ref="unused-rewritten-by-coupling",
        params={"n": 33, "diffusivity": 1e-3, "velocity": 0.2, "dt": 1.0, "steps": 20, "length": 1.0},
    )
    poroelastic_step = ScenarioStep(
        op="poroelastic",
        inputs_ref="unused-rewritten-by-coupling",
        params={"compressibility": 1.0, "storage": 1.0, "dirichlet": 0.0, "spacing": 1.0},
    )
    return Scenario(
        steps=[flow_step, transport_step, poroelastic_step],
        couplings=[(0, 1, "head"), (1, 2, "load")],
        provenance={"scenario": "dewatering-subsidence"},
    )


def test_coupled_dag_wires_provenance_and_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))

    scenario = _dewatering_scenario()
    result = simulate(scenario)

    stage_refs = result.provenance["steps"]
    assert len(stage_refs) == 3

    for ref in stage_refs:
        assert os.path.exists(tmp_path / f"{ref}.npz")
        assert os.path.exists(tmp_path / f"{ref}.json")

    transport_header = read_result_header(stage_refs[1], store_dir=str(tmp_path))
    poroelastic_header = read_result_header(stage_refs[2], store_dir=str(tmp_path))

    assert transport_header["provenance"]["parents"] == [stage_refs[0]]
    assert transport_header["provenance"]["coupling"] == "head"
    assert poroelastic_header["provenance"]["parents"] == [stage_refs[1]]
    assert poroelastic_header["provenance"]["coupling"] == "load"

    assert result.result_ref == stage_refs[2]

    # a second run of the identical scenario must be byte-reproducible: same final content hash.
    result_again = simulate(scenario)
    assert result_again.result_ref == result.result_ref
    assert result_again.provenance["steps"] == stage_refs


def test_cyclic_couplings_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))
    scenario = _dewatering_scenario()
    scenario.couplings = [(0, 1, "head"), (1, 2, "load"), (2, 0, "back-edge")]

    import pytest

    with pytest.raises(ValueError):
        simulate(scenario)
