"""P3 -- digital twin of the physical environment: DoD test.

Synthesises a truth field and a noisy batch monitoring sweep, builds an un-updated baseline twin and
an assimilated twin, and checks that the assimilated twin's next forecast tracks the synthetic truth
strictly better than the un-updated baseline's forecast -- the twin's whole point: monitoring in via
data assimilation, what-ifs out via `simulate`, and the loop actually helps.
"""

from __future__ import annotations

import numpy as np

from mixle_pde import simulation_service
from mixle_pde.digital_twin import build_twin
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
)
from mixle_pde.simulation_service import STORE_DIR_ENV, Scenario, ScenarioStep


def _persist_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """A trivial persistence forecast: return the seeded field unchanged.

    Registered here (test-only) purely to exercise `DigitalTwin.forecast`'s seed -> `simulate` wiring
    with real, honest physics-free "physics": what the twin's current calibrated estimate says the
    field looks like, projected forward with no assumed change. Real evolution forwards ("flow",
    "transport", ...) already live in `simulation_service`; this DoD does not add one there.
    """
    store_dir = params["_store_dir"]
    upstream = simulation_service.read_result_artifact(inputs_ref, store_dir=store_dir)
    return {"field": upstream["field"]}


simulation_service.register_forward("persist", _persist_forward)


def _grid() -> Field3D:
    xs = np.linspace(0.0, 40.0, 9)
    coords = np.array([[x, 0.0, 0.0] for x in xs], dtype=float)
    return Field3D(coordinates=coords, spacing=5.0, units="m", property_name="head", bounds=None)


def _truth(grid: Field3D) -> np.ndarray:
    return 12.0 + 0.75 * grid.coordinates[:, 0]


def _monitoring_sweep(grid: Field3D, truth: np.ndarray, rng: np.random.Generator):
    times = np.array([0.0, 1.0, 2.0])
    obs_by_time = []
    for _ in times:
        value = truth + rng.normal(0.0, 0.6, size=grid.n)
        obs_by_time.append(
            [
                Observation(
                    kind="borehole",
                    location=grid.coordinates,
                    value=value,
                    noise_cov=np.full(grid.n, 0.6**2),
                )
            ]
        )
    return obs_by_time, times


def _forecast_rmse(twin, scenario: Scenario, truth: np.ndarray, store_dir: str) -> float:
    result = twin.forecast(scenario, n=64, rng=np.random.default_rng(11))
    arrays = simulation_service.read_result_artifact(result.result_ref, store_dir=store_dir)
    return float(np.sqrt(np.mean((arrays["field"] - truth) ** 2)))


def test_assimilated_twin_forecasts_closer_to_truth_than_the_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))

    rng = np.random.default_rng(7)
    grid = _grid()
    truth = _truth(grid)

    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0e-3, length_scale=5.0)

    obs_by_time, times = _monitoring_sweep(grid, truth, rng)

    t0 = build_twin(grid, registry, prior, process_var=1.0e-2)
    t1 = t0.assimilate(obs_by_time, times)

    scenario = Scenario(steps=[ScenarioStep(op="persist", inputs_ref="unused", params={})])

    rmse_before = _forecast_rmse(t0, scenario, truth, str(tmp_path))
    rmse_after = _forecast_rmse(t1, scenario, truth, str(tmp_path))

    assert rmse_after < rmse_before


def test_assimilate_returns_a_new_twin_and_does_not_mutate_the_original(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))

    rng = np.random.default_rng(3)
    grid = _grid()
    truth = _truth(grid)

    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0e-3, length_scale=5.0)

    obs_by_time, times = _monitoring_sweep(grid, truth, rng)

    t0 = build_twin(grid, registry, prior, process_var=1.0e-2)
    baseline_mean = t0.state.mean.copy()
    t1 = t0.assimilate(obs_by_time, times)

    assert t1 is not t0
    assert np.array_equal(t0.state.mean, baseline_mean)
    assert not np.array_equal(t1.state.mean, t0.state.mean)
    assert t1.grid.provenance.get("content_hash") is not None


def test_forecast_result_carries_posterior_spread_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))

    grid = _grid()
    registry = ForwardOperatorRegistry()
    registry.register(borehole_forward_operator())
    prior = FieldGaussianPrior(mean=0.0, smoothness_precision=0.0, marginal_precision=1.0e-2, length_scale=5.0)

    twin = build_twin(grid, registry, prior, process_var=1.0e-2)
    scenario = Scenario(steps=[ScenarioStep(op="persist", inputs_ref="unused", params={})])

    result = twin.forecast(scenario, n=32, rng=np.random.default_rng(0))

    assert result.uncertainty is not None
    assert result.uncertainty["mean"].shape == (grid.n,)
    assert result.uncertainty["std"].shape == (grid.n,)
    assert np.all(result.uncertainty["std"] >= 0.0)
