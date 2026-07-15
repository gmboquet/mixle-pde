"""DoD test for E2 -- PDE artifact I/O + registrable model kind.

Named ``test_e2_artifacts.py`` per the work order's frozen DoD command (the repo's own convention is
``*_test.py``, but this exact path is what the DoD invokes).
"""

import json

import numpy as np

from mixle_pde.io import artifacts
from mixle_pde.latent import Field3D, PosteriorField3D


def _small_posterior() -> PosteriorField3D:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    grid = Field3D(
        coordinates=coords,
        spacing=1.0,
        units="kg/m3",
        property_name="density",
        provenance={"source": "synthetic-e2-test", "crs": "EPSG:32611"},
    )
    rng = np.random.default_rng(0)
    a = rng.standard_normal((grid.n, grid.n))
    cov = a @ a.T + 0.5 * np.eye(grid.n)
    mean = rng.standard_normal(grid.n)
    return PosteriorField3D(grid=grid, mean=mean, dense_cov=cov)


def test_round_trip_mean_and_covariance(tmp_path):
    posterior = _small_posterior()
    path = str(tmp_path / "posterior")

    artifacts.save_posterior(posterior, path)
    loaded = artifacts.load_posterior(path)

    np.testing.assert_allclose(loaded.mean, posterior.mean, atol=1e-10)
    np.testing.assert_allclose(loaded.cov, posterior.cov, atol=1e-10)


def test_json_header_has_frozen_keys(tmp_path):
    posterior = _small_posterior()
    path = str(tmp_path / "posterior")
    artifacts.save_posterior(posterior, path)

    with open(f"{path}.json") as f:
        header = json.load(f)

    for key in artifacts.HEADER_KEYS:
        assert key in header
    assert header["crs"] == "EPSG:32611"
    assert header["units"] == "kg/m3"


def test_resave_of_the_loaded_posterior_yields_the_same_content_hash(tmp_path):
    posterior = _small_posterior()
    path_a = str(tmp_path / "a")
    artifacts.save_posterior(posterior, path_a)
    hash_a = artifacts.content_hash(path_a)

    loaded = artifacts.load_posterior(path_a)
    path_b = str(tmp_path / "b")
    artifacts.save_posterior(loaded, path_b)
    hash_b = artifacts.content_hash(path_b)

    assert hash_a == hash_b
    assert len(hash_a) == 64


def test_registry_register_then_load_round_trips(tmp_path):
    from mixle.registry import Registry

    posterior = _small_posterior()
    registry = Registry(dir=str(tmp_path / "registry"))
    entry = registry.register(posterior, capabilities=["field_posterior:density"])
    assert entry.kind == "field_posterior"

    loaded = registry.load(entry.entry_id)
    np.testing.assert_allclose(loaded.mean, posterior.mean, atol=1e-10)
    np.testing.assert_allclose(loaded.cov, posterior.cov, atol=1e-10)
