"""P2 -- coupled multiphysics scenario runner: DoD test.

A 3-step scenario (vorticity flow -> advected field -> poroelastic subsidence) wired with two coupling
edges. `simulate` must walk the DAG in topological order, rewrite each child's `inputs_ref` onto its
parent's content-hashed `result_ref`, record `{"parents": [...], "coupling": field_name}` in every
downstream artifact's provenance, and be byte-reproducible on re-run.

The scenario's physics were rewritten when P1 replaced this module's temporary placeholder
flow/poroelastic forwards (steady Poisson diffusion) with the real ``flow.NavierStokes2D`` /
``poroelastic.BiotPoroelastic1D`` solvers the spec always called for -- the placeholder's parameter
convention (``source``/``conductivity``/``compressibility``/``storage``) no longer applies. This test
only asserts the DAG-wiring/provenance/reproducibility properties (P2's own contribution), not specific
physical values, so the substitution does not change what is actually being verified.
"""

from __future__ import annotations

import os

import numpy as np

from mixle_pde.simulation_service import (
    STORE_DIR_ENV,
    Scenario,
    ScenarioStep,
    read_result_header,
    simulate,
    write_result_artifact,
)


def _dewatering_scenario(store_dir: str) -> Scenario:
    n_grid = 5  # NavierStokes2D grid side; nn = n_grid**2 = 25 is the flow field's flat length
    omega0_ref = write_result_artifact(
        {"omega": np.zeros(n_grid * n_grid)},
        grid={"shape": [n_grid, n_grid]},
        units="",
        provenance={"op": "test-fixture:initial-vorticity"},
        store_dir=store_dir,
    )
    flow_step = ScenarioStep(
        op="flow",
        inputs_ref=omega0_ref,
        params={"n": n_grid, "viscosity": 0.1, "dt": 0.01, "steps": 3},
    )
    transport_step = ScenarioStep(
        op="transport",
        inputs_ref="unused-rewritten-by-coupling",
        params={"n": n_grid * n_grid, "diffusivity": 1e-3, "velocity": 0.2, "dt": 1.0, "steps": 20, "length": 1.0},
    )
    poroelastic_step = ScenarioStep(
        op="poroelastic",
        inputs_ref="unused-rewritten-by-coupling",
        params={"n": n_grid * n_grid, "dt": 1e-4, "steps": 5},
    )
    return Scenario(
        steps=[flow_step, transport_step, poroelastic_step],
        couplings=[(0, 1, "psi"), (1, 2, "field")],
        provenance={"scenario": "flow-transport-subsidence"},
    )


def test_coupled_dag_wires_provenance_and_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))

    scenario = _dewatering_scenario(str(tmp_path))
    result = simulate(scenario)

    stage_refs = result.provenance["steps"]
    assert len(stage_refs) == 3

    for ref in stage_refs:
        assert os.path.exists(tmp_path / f"{ref}.npz")
        assert os.path.exists(tmp_path / f"{ref}.json")

    transport_header = read_result_header(stage_refs[1], store_dir=str(tmp_path))
    poroelastic_header = read_result_header(stage_refs[2], store_dir=str(tmp_path))

    assert transport_header["provenance"]["parents"] == [stage_refs[0]]
    assert transport_header["provenance"]["coupling"] == "psi"
    assert poroelastic_header["provenance"]["parents"] == [stage_refs[1]]
    assert poroelastic_header["provenance"]["coupling"] == "field"

    assert result.result_ref == stage_refs[2]

    # a second run of the identical scenario must be byte-reproducible: same final content hash.
    result_again = simulate(scenario)
    assert result_again.result_ref == result.result_ref
    assert result_again.provenance["steps"] == stage_refs


def test_cyclic_couplings_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))
    scenario = _dewatering_scenario(str(tmp_path))
    scenario.couplings = [(0, 1, "psi"), (1, 2, "field"), (2, 0, "back-edge")]

    import pytest

    with pytest.raises(ValueError):
        simulate(scenario)
