from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from mixle_pde.io.artifacts import Field3D, load_posterior, save_posterior
from mixle_pde.latent import PosteriorField3D, SparsePosteriorPrecision


def _grid() -> Field3D:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    return Field3D(coordinates=coordinates, spacing=(1.0, 1.0, 1.0), units="kg/m3", property_name="density")


def _cov_mode_on_disk(path: str) -> str:
    with open(f"{path}.json") as f:
        header = json.load(f)
    return header["grid"]["cov_mode"]


def test_dense_covariance_round_trips_losslessly(tmp_path) -> None:
    grid = _grid()
    mean = np.array([1.0, 2.0, 3.0, 4.0])
    dense_cov = np.eye(grid.n) * 0.5
    posterior = PosteriorField3D(grid=grid, mean=mean, dense_cov=dense_cov)

    path = str(tmp_path / "dense")
    save_posterior(posterior, path)
    assert _cov_mode_on_disk(path) == "dense"
    loaded = load_posterior(path)

    # Storage mode itself must round-trip, not just the derived .cov values -- a loader that
    # silently degrades every mode to a dense reconstruction would still pass a values-only check.
    assert loaded.dense_cov is not None
    assert loaded.precision_factor is None
    assert loaded.low_rank is None
    np.testing.assert_allclose(loaded.mean, posterior.mean)
    np.testing.assert_allclose(loaded.cov, posterior.cov)


def test_sparse_precision_covariance_round_trips_losslessly(tmp_path) -> None:
    grid = _grid()
    mean = np.zeros(grid.n)
    precision = sp.identity(grid.n, format="csc") * 2.0
    posterior = PosteriorField3D(grid=grid, mean=mean, precision_factor=SparsePosteriorPrecision(precision=precision))

    path = str(tmp_path / "precision")
    save_posterior(posterior, path)
    assert _cov_mode_on_disk(path) == "precision"
    loaded = load_posterior(path)

    assert loaded.dense_cov is None
    assert loaded.precision_factor is not None
    assert loaded.low_rank is None
    np.testing.assert_allclose(loaded.mean, posterior.mean)
    np.testing.assert_allclose(loaded.cov, posterior.cov, atol=1e-10)


def test_low_rank_covariance_round_trips_losslessly(tmp_path) -> None:
    grid = _grid()
    mean = np.zeros(grid.n)
    rng = np.random.default_rng(0)
    low_rank = rng.normal(size=(grid.n, 2))
    diag_var = np.full(grid.n, 0.1)
    posterior = PosteriorField3D(grid=grid, mean=mean, low_rank=low_rank, diag_var=diag_var)

    path = str(tmp_path / "low_rank")
    save_posterior(posterior, path)
    assert _cov_mode_on_disk(path) == "low_rank"
    loaded = load_posterior(path)

    assert loaded.dense_cov is None
    assert loaded.precision_factor is None
    assert loaded.low_rank is not None
    np.testing.assert_allclose(loaded.mean, posterior.mean)
    np.testing.assert_allclose(loaded.cov, posterior.cov)


def test_diagonal_covariance_round_trips_losslessly(tmp_path) -> None:
    grid = _grid()
    mean = np.zeros(grid.n)
    diag_var = np.array([0.1, 0.2, 0.3, 0.4])
    posterior = PosteriorField3D(grid=grid, mean=mean, diag_var=diag_var)

    path = str(tmp_path / "diag")
    save_posterior(posterior, path)
    assert _cov_mode_on_disk(path) == "diag"
    loaded = load_posterior(path)

    assert loaded.dense_cov is None
    assert loaded.precision_factor is None
    assert loaded.low_rank is None
    np.testing.assert_allclose(loaded.mean, posterior.mean)
    np.testing.assert_allclose(loaded.cov, posterior.cov)


def test_save_posterior_rejects_non_posterior_field_3d(tmp_path) -> None:
    with pytest.raises(TypeError, match="PosteriorField3D"):
        save_posterior(object(), str(tmp_path / "bad"))
