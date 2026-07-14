"""Definition-of-Done + coverage tests for L7 -- regional climate downscaling + site extremes."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from mixle_pde.climate_downscale import (
    DownscaledField,
    QuantileMap,
    apply_quantile_map,
    downscale,
    downscaled_return_level,
    fit_quantile_map,
    site_spatial_extremes,
)
from mixle_pde.mesh import SimplexMesh

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "gcm_boundary_stub.npz")


def _load_fixture():
    with np.load(FIXTURE_PATH) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _site_mesh(lat=40.0, lon=-105.0):
    # A single-site "mesh": one node, no cells -- valid, minimal site geometry for the statistical path.
    return SimplexMesh(nodes=np.array([[lat, lon]]), simplices=np.empty((0, 3), dtype=int))


def test_downscaled_return_level_matches_reference():
    fx = _load_fixture()
    model_hist, obs_hist, model_future = fx["model_hist"], fx["obs_hist"], fx["model_future"]
    a, b = float(fx["a"]), float(fx["b"])

    content_hash = hashlib.sha256(model_future.tobytes()).hexdigest()
    gcm = {
        "value": {"model": model_hist, "projection": model_future},
        "content_hash": content_hash,
        "model_id": "climate-projection-stub",
        "version": "synthetic-v1",
    }

    field = downscale(gcm, _site_mesh(), method="statistical", obs_baseline=obs_hist, variable="tmax")

    assert isinstance(field, DownscaledField)
    assert field.climate_hash == content_hash
    assert len(field.climate_hash) == 64
    int(field.climate_hash, 16)  # is genuinely hex

    threshold = float(np.quantile(obs_hist, 0.9))
    true_future = a * model_future + b  # the known site relationship -- ground truth for the projection
    reference = downscaled_return_level(true_future, threshold, 20.0)
    got = field.return_levels[20.0]

    assert abs(got - reference) / abs(reference) < 0.05


def test_downscale_requires_obs_baseline_for_statistical_method():
    gcm = {"value": {"model": np.arange(10.0), "projection": np.arange(10.0)}, "content_hash": "0" * 64}
    with pytest.raises(ValueError):
        downscale(gcm, _site_mesh(), method="statistical")


def test_downscale_rejects_unknown_method():
    gcm = {"value": np.arange(10.0), "content_hash": "0" * 64}
    with pytest.raises(ValueError):
        downscale(gcm, _site_mesh(), method="bogus", obs_baseline=np.arange(10.0))


def test_downscale_accepts_provenanced_result_object():
    class _PR:
        value = {"model": np.linspace(0, 1, 50), "projection": np.linspace(0, 1, 50)}
        content_hash = "a" * 64
        model_id = "climate-projection-stub"
        version = "synthetic-v1"

    field = downscale(_PR(), _site_mesh(), obs_baseline=np.linspace(0, 1, 50))
    assert field.climate_hash == "a" * 64
    assert field.provenance["gcm"]["model_id"] == "climate-projection-stub"


def test_dynamical_method_raises_clear_error_when_p1_unavailable():
    gcm = {"value": np.linspace(0, 1, 20), "content_hash": "b" * 64}
    with pytest.raises(RuntimeError, match="simulation_service"):
        downscale(gcm, _site_mesh(), method="dynamical")


def test_fit_and_apply_quantile_map_roundtrip_affine_bias():
    rng = np.random.default_rng(1)
    obs = rng.normal(10.0, 2.0, 2000)
    model = (obs - 5.0) / 2.0  # obs = 2*model + 5
    qm = fit_quantile_map(model, obs, n_q=50)
    assert isinstance(qm, QuantileMap)

    probe = model[:100]
    mapped = apply_quantile_map(qm, probe)
    np.testing.assert_allclose(mapped, 2.0 * probe + 5.0, atol=0.2)


def test_site_spatial_extremes_fits_smith_maxstable():
    rng = np.random.default_rng(2)
    site = SimplexMesh(nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), simplices=np.array([[0, 1, 2]]))
    block_maxima = 1.0 / rng.uniform(0.01, 1.0, size=(200, 3))  # crude unit-Frechet-ish replicates

    fitted = site_spatial_extremes(site, block_maxima)
    theta_far = fitted.extremal_coefficient(np.array([10.0, 10.0]))
    theta_near = fitted.extremal_coefficient(np.array([1e-6, 0.0]))
    assert 1.0 <= theta_near <= theta_far <= 2.0
