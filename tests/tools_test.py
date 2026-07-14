"""Tests for the real IC-3 physics-tool handlers (work-plan E4).

``mixle_pde.io.artifacts`` (IC-2, workstream E2) is a sibling PR that has not merged as of this ticket,
so these tests install a minimal in-memory stand-in for it -- the same technique
``mixle-mlops/tests/test_e4_physics_tools.py`` already uses to test its own (mlops-side) wiring against a
not-yet-landed ``mixle_pde.tools``. Once E2 lands, ``tools.py``'s ``_artifacts_module`` (a plain
``from mixle_pde.io import artifacts``) picks up the real module unchanged and every test below still
holds -- only ``test_artifacts_missing_raises_clear_import_error`` (which relies on the real module being
absent) will need E2's own test suite to keep covering that path once it lands.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import types

import pytest

from mixle_pde import tools


class _FakeArtifacts:
    """In-memory stand-in for `mixle_pde.io.artifacts` (IC-2), keyed by the same `path` string
    `save_posterior`/`load_posterior`/`content_hash` all take."""

    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    def save_posterior(self, posterior, path):
        self._store[path] = posterior

    def load_posterior(self, path):
        if path not in self._store:
            raise FileNotFoundError(path)
        return self._store[path]

    def content_hash(self, path):
        posterior = self._store[path]
        return hashlib.sha256(posterior.mean.tobytes()).hexdigest()


@pytest.fixture
def fake_artifacts(monkeypatch):
    fake = _FakeArtifacts()
    module = types.ModuleType("mixle_pde.io.artifacts")
    module.save_posterior = fake.save_posterior
    module.load_posterior = fake.load_posterior
    module.content_hash = fake.content_hash
    monkeypatch.setitem(sys.modules, "mixle_pde.io.artifacts", module)
    return fake


def _write_json(path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def _tiny_gravity_dataset(tmp_path) -> str:
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
            {
                "location": stations,
                "value": [0.02, 0.015, 0.03],
                "noise_cov": [1e-4, 1e-4, 1e-4],
                "units": "mGal",
            }
        ],
    }
    path = tmp_path / "dataset.json"
    _write_json(path, bundle)
    return str(path)


# --- frozen IC-3 schema / signature conformance ---


def test_frozen_tool_names():
    assert set(tools.PHYSICS_TOOL_SCHEMAS) == {"run_inversion", "query_posterior", "gassmann", "forward_model"}


def test_query_enum_is_frozen():
    q = tools.PHYSICS_TOOL_SCHEMAS["query_posterior"]["properties"]["query"]["enum"]
    assert q == ["region_mass", "prob_exceed", "net_pay", "drill_target", "marginal", "section"]


def test_run_inversion_required_args_and_signature():
    assert tools.PHYSICS_TOOL_SCHEMAS["run_inversion"]["required"] == ["dataset_ref", "modality", "prior"]
    assert list(inspect.signature(tools.run_inversion).parameters) == ["dataset_ref", "modality", "prior", "config"]


def test_query_posterior_signature():
    assert tools.PHYSICS_TOOL_SCHEMAS["query_posterior"]["required"] == ["posterior_ref", "query"]
    assert list(inspect.signature(tools.query_posterior).parameters) == ["posterior_ref", "query", "params"]


def test_gassmann_and_forward_model_signatures():
    assert list(inspect.signature(tools.gassmann).parameters) == ["inputs"]
    assert list(inspect.signature(tools.forward_model).parameters) == ["modality", "model_ref", "geometry_ref"]


# --- run_inversion + query_posterior real round trip (gravity, "smooth" prior) ---


def test_run_inversion_then_query_posterior_region_mass(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    result = tools.run_inversion(dataset_ref, "gravity", "smooth")
    assert result["diagnostics"]["modality"] == "gravity"
    assert result["diagnostics"]["n_cells"] == 4
    assert len(result["diagnostics"]["content_hash"]) == 64
    posterior_ref = result["posterior_ref"]

    out = tools.query_posterior(posterior_ref, "region_mass", {"region": [True, True, False, False]})
    assert "prior_dominated" in out
    assert out["prior_dominated"] is False
    assert isinstance(out["value"], float)
    assert (
        out["distribution"]["credible_interval_90"][0] <= out["value"] <= out["distribution"]["credible_interval_90"][1]
    )


def test_query_posterior_prob_exceed_and_marginal(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    posterior_ref = tools.run_inversion(dataset_ref, "gravity", "smooth")["posterior_ref"]

    exceed = tools.query_posterior(
        posterior_ref, "prob_exceed", {"region_index": [0, 1], "threshold": 0.0, "n": 256, "seed": 0}
    )
    assert "prior_dominated" in exceed
    assert 0.0 <= exceed["value"] <= 1.0

    marginal = tools.query_posterior(posterior_ref, "marginal", {"index": [0]})
    assert "prior_dominated" in marginal
    assert isinstance(marginal["value"], float)


def test_query_posterior_section(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    posterior_ref = tools.run_inversion(dataset_ref, "gravity", "smooth")["posterior_ref"]

    out = tools.query_posterior(posterior_ref, "section", {"z": -10.0})
    assert "prior_dominated" in out
    assert len(out["value"]) == 4


def test_query_posterior_drill_target(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    posterior_ref = tools.run_inversion(dataset_ref, "gravity", "smooth")["posterior_ref"]

    out = tools.query_posterior(
        posterior_ref,
        "drill_target",
        {"region": [True, True, True, True], "criteria": {"op": ">", "threshold": -1.0}, "n": 128, "seed": 1},
    )
    assert "prior_dominated" in out
    assert 0.0 <= out["value"] <= 1.0


def test_query_posterior_net_pay(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    posterior_ref = tools.run_inversion(dataset_ref, "gravity", "smooth")["posterior_ref"]

    out = tools.query_posterior(
        posterior_ref,
        "net_pay",
        {"column_index": [0, 1], "sat_cut": -1.0, "thickness": [1.0, 1.0], "n": 128, "seed": 2},
    )
    assert "prior_dominated" in out


def test_query_posterior_prior_dominated_flag_uses_supplied_variances(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    posterior_ref = tools.run_inversion(dataset_ref, "gravity", "smooth")["posterior_ref"]

    out = tools.query_posterior(
        posterior_ref,
        "region_mass",
        {
            "region": [True, True, False, False],
            "prior_var": [1.0, 1.0, 1.0, 1.0],
            "posterior_var": [0.99, 0.99, 0.99, 0.99],
        },
    )
    assert out["prior_dominated"] is True


def test_run_inversion_unsupported_modality_is_a_clear_error(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    with pytest.raises(ValueError, match="not wired"):
        tools.run_inversion(dataset_ref, "seismic", "smooth")


def test_run_inversion_unsupported_prior_is_a_clear_error(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    with pytest.raises(ValueError, match="not wired"):
        tools.run_inversion(dataset_ref, "gravity", "blocky")


def test_query_posterior_unknown_query(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    posterior_ref = tools.run_inversion(dataset_ref, "gravity", "smooth")["posterior_ref"]
    with pytest.raises(ValueError):
        tools.query_posterior(posterior_ref, "bogus")


# --- gassmann ---


def test_gassmann_deterministic_round_trip():
    inputs = {
        "Vp": 3.0,
        "Vs": 1.6,
        "rho": 2.3,
        "phi": 0.2,
        "K_min": 37.0,
        "rho_min": 2.65,
        "K_fl_in": 2.5,
        "rho_fl_in": 1.0,
        "K_fl_out": 0.02,
        "rho_fl_out": 0.1,
    }
    out = tools.gassmann(inputs)
    assert set(out) == {"vp", "vs", "rho", "uncertainty"}
    assert out["uncertainty"] == {"vp": 0.0, "vs": 0.0, "rho": 0.0}
    assert out["vp"] > 0.0 and out["vs"] > 0.0 and out["rho"] > 0.0


def test_gassmann_propagates_uncertainty():
    inputs = {
        "Vp": 3.0,
        "Vs": 1.6,
        "rho": 2.3,
        "phi": 0.2,
        "K_min": 37.0,
        "rho_min": 2.65,
        "K_fl_in": 2.5,
        "rho_fl_in": 1.0,
        "K_fl_out": 0.02,
        "rho_fl_out": 0.1,
        "uncertainty": {"phi": 0.02},
        "n_samples": 4000,
        "seed": 0,
    }
    out = tools.gassmann(inputs)
    assert out["uncertainty"]["vp"] > 0.0


def test_gassmann_missing_field():
    with pytest.raises(ValueError, match="missing required field"):
        tools.gassmann({"Vp": 1.0})


# --- forward_model ---


def test_forward_model_gravity_round_trip(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    posterior_ref = tools.run_inversion(dataset_ref, "gravity", "smooth")["posterior_ref"]

    geometry_path = tmp_path / "geometry.json"
    _write_json(geometry_path, {"points": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], "crs": "EPSG:32611"})

    out = tools.forward_model("gravity", posterior_ref, str(geometry_path))
    assert "data_ref" in out
    with open(out["data_ref"]) as f:
        payload = json.load(f)
    assert len(payload["data"]) == 2
    assert payload["modality"] == "gravity"


def test_forward_model_unsupported_modality(tmp_path, fake_artifacts):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    posterior_ref = tools.run_inversion(dataset_ref, "gravity", "smooth")["posterior_ref"]
    with pytest.raises(ValueError, match="not wired"):
        tools.forward_model("seismic", posterior_ref, "unused")


def test_artifacts_missing_raises_clear_import_error(tmp_path):
    dataset_ref = _tiny_gravity_dataset(tmp_path)
    with pytest.raises(ImportError, match="mixle_pde.io.artifacts"):
        tools.run_inversion(dataset_ref, "gravity", "smooth")
