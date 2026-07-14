"""P1: the IC-11 forward-simulation service (`mixle_pde.simulation_service`).

Covers the frozen IC-11 signature, the "register, don't branch" forward registry, the single-step
coupling-free `simulate` path (content-hashed, idempotent artifact round-trip), and the guard rails that
route multi-step / unregistered-op scenarios to a clear error rather than silently doing the wrong thing.
"""

import inspect

import numpy as np
import pytest

import mixle_pde.simulation_service as sim_service
from mixle_pde.simulation_service import (
    Scenario,
    ScenarioStep,
    SimResult,
    available_forwards,
    get_tool_catalog,
    read_result_artifact,
    register_forward,
    simulate,
    write_result_artifact,
)


def test_ic11_signature():
    for name in ("simulate", "register_forward", "Scenario", "ScenarioStep", "SimResult"):
        assert hasattr(sim_service, name), name
    assert list(inspect.signature(simulate).parameters) == ["scenario"]


def test_builtin_forwards_registered():
    forwards = available_forwards()
    for op in ("transport", "dispersion", "wave", "flow", "em", "poroelastic", "model"):
        assert op in forwards
    # climate_rcm/exposure/habitat are open slots for L7/K/N -- not registered here.
    for op in ("climate_rcm", "exposure", "habitat"):
        assert op not in forwards


def test_register_forward_is_additive_not_branching():
    calls = []

    def toy(inputs_ref, params):
        calls.append((inputs_ref, params))
        return {"y": np.array([1.0, 2.0])}

    register_forward("__test_toy__", toy)
    try:
        assert "__test_toy__" in available_forwards()
        out = sim_service._FORWARDS["__test_toy__"]("ref", {"a": 1})
        assert list(out["y"]) == [1.0, 2.0]
        assert calls == [("ref", {"a": 1})]
    finally:
        del sim_service._FORWARDS["__test_toy__"]


def test_transport_simulate_round_trip(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MIXLE_PDE_SIM_STORE_DIR", str(store))

    n = 32
    plume = tmp_path / "plume.npz"
    field0 = np.exp(-((np.arange(n) - n / 2.0) ** 2) / 10.0)
    np.savez(plume, field=field0)

    scenario = Scenario(
        steps=[
            ScenarioStep(
                op="transport", inputs_ref=str(plume), params={"diffusivity": 1e-3, "velocity": 0.2, "n": n, "steps": 5}
            )
        ],
        couplings=[],
        provenance={"scenario_name": "plume-test"},
    )
    result = simulate(scenario)
    assert isinstance(result, SimResult)
    assert len(result.result_ref) == 64
    assert result.provenance == {"op": "transport", "content_hash": result.result_ref}

    from mixle_pde.io.artifacts import sha256_of_arrays

    arrays = read_result_artifact(result.result_ref, store_dir=str(store))
    assert sha256_of_arrays(arrays) == result.result_ref
    assert "field" in arrays
    assert arrays["field"].shape == (n,)

    # re-running the identical scenario is idempotent (same content hash, no duplicate growth of intent)
    result2 = simulate(scenario)
    assert result2.result_ref == result.result_ref


def test_simulate_without_couplings_requires_one_step():
    scenario = Scenario(steps=[], couplings=[], provenance={})
    with pytest.raises(ValueError):
        simulate(scenario)


def test_simulate_unknown_op_raises_keyerror(tmp_path):
    scenario = Scenario(
        steps=[ScenarioStep(op="__not_registered__", inputs_ref="x", params={})],
        couplings=[],
        provenance={},
    )
    with pytest.raises(KeyError):
        simulate(scenario)


def test_write_and_read_result_artifact_are_content_addressed(tmp_path):
    from mixle_pde.io.artifacts import sha256_of_arrays

    arrays = {"a": np.arange(5.0), "b": np.eye(2)}
    ref = write_result_artifact(arrays, grid={"n": 5}, units="m", provenance={"op": "test"}, store_dir=str(tmp_path))
    assert ref == sha256_of_arrays(arrays)
    assert (tmp_path / f"{ref}.npz").exists()
    assert (tmp_path / f"{ref}.json").exists()

    loaded = read_result_artifact(ref, store_dir=str(tmp_path))
    assert set(loaded) == {"a", "b"}
    np.testing.assert_array_equal(loaded["a"], arrays["a"])
    np.testing.assert_array_equal(loaded["b"], arrays["b"])


def test_ic10_catalog_entry_registered():
    catalog = get_tool_catalog()
    entry = catalog.get("simulate")
    assert entry is not None
    assert entry.owner == "physics"
    assert entry.verifier == "physical"
