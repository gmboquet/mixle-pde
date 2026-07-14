"""P4 -- simulation UQ + surrogate acceleration: DoD conformance.

Two behaviours are under test, both appended to `simulation_service.py` on top of the P1/P2
substrate: (1) `register_surrogate` distills a fast E6 `Surrogate` for a registered `op` and wires it
behind a calibrated escalation gate, so easy in-distribution what-ifs are answered by the surrogate
and hard out-of-distribution ones are escalated to the real teacher forward; (2) `propagate_uq` pushes
`n` IC-1 posterior draws of a scenario's leading input through the (possibly coupled) forwards and
reports the output ensemble's per-node credible interval as honest Monte-Carlo forward UQ.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mixle_pde.simulation_service import (
    _FORWARDS,
    _SURROGATES,
    STORE_DIR_ENV,
    Scenario,
    ScenarioStep,
    propagate_uq,
    register_surrogate,
    write_result_artifact,
)


def _analytic_forward(x):
    """Cheap stand-in "teacher": a smooth nonlinear scalar function of two physical parameters."""
    a, b = x
    return math.sin(a) + 0.5 * b * b


def _uniform_sampler(lo, hi):
    def sampler(n, rng):
        return rng.uniform(lo, hi, size=(n, 2))

    return sampler


class _GaussianPosterior:
    """Minimal IC-1 `Posterior` conformer: an isotropic Gaussian over the two toy inputs."""

    def __init__(self, loc: np.ndarray, scale: float):
        self._loc = np.asarray(loc, dtype=float)
        self._scale = float(scale)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self._loc + self._scale * rng.standard_normal((n, self._loc.shape[0]))

    @property
    def mean(self) -> np.ndarray:
        return self._loc

    @property
    def cov(self):
        return (self._scale**2) * np.eye(self._loc.shape[0])

    def credible_interval(self, level: float):
        z = self._scale * 1.645  # ~90% for a standard normal, good enough for a test double
        return self._loc - z, self._loc + z

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError("unused by propagate_uq")


def _x_artifact(store_dir: str, x) -> str:
    return write_result_artifact(
        {"x": np.asarray(x, dtype=float)},
        grid={"shape": [len(x)]},
        units="",
        provenance={"op": "test-input-draw"},
        store_dir=store_dir,
    )


def test_register_surrogate_wires_predict_and_defer_behind_the_op(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))
    register_surrogate(
        "toy",
        teacher=_analytic_forward,
        sampler=_uniform_sampler(-2.0, 2.0),
        budget=256,
        seed=0,
    )
    assert "toy" in _FORWARDS
    surrogate = _SURROGATES["toy"]

    rng = np.random.default_rng(1)
    easy = rng.uniform(-1.5, 1.5, size=(30, 2))
    hard = rng.uniform(7.0, 9.0, size=(30, 2))

    # easy, in-distribution split: the wrapped forward should match the teacher within tolerance,
    # and rarely need to escalate.
    easy_errors = []
    easy_defers = 0
    for x in easy:
        ref = _x_artifact(str(tmp_path), x)
        out = _FORWARDS["toy"](ref, {"_store_dir": str(tmp_path)})["value"]
        truth = _analytic_forward(x)
        easy_errors.append(abs(float(out[0]) - truth))
        if surrogate.defer(x):
            easy_defers += 1
    assert np.mean(easy_errors) < 0.3
    easy_rate = easy_defers / len(easy)
    assert easy_rate <= 0.2

    # hard, out-of-distribution split: the deferral rate must rise, and the service must fall back
    # to the exact teacher value (escalation), not a stale extrapolated surrogate guess.
    hard_defers = 0
    for x in hard:
        ref = _x_artifact(str(tmp_path), x)
        out = _FORWARDS["toy"](ref, {"_store_dir": str(tmp_path)})["value"]
        truth = _analytic_forward(x)
        assert abs(float(out[0]) - truth) < 1e-9  # escalated: exact teacher value, not a guess
        if surrogate.defer(x):
            hard_defers += 1
    hard_rate = hard_defers / len(hard)
    assert hard_rate > easy_rate
    assert hard_rate >= 0.6


def test_propagate_uq_returns_a_non_degenerate_credible_interval(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))
    register_surrogate(
        "toy-uq",
        teacher=_analytic_forward,
        sampler=_uniform_sampler(-2.0, 2.0),
        budget=256,
        seed=2,
    )

    leading = ScenarioStep(op="toy-uq", inputs_ref="unused-rewritten-by-propagate_uq", params={})
    scenario = Scenario(steps=[leading], provenance={"scenario": "toy-uq-forecast"})

    posterior = _GaussianPosterior(loc=np.array([0.0, 0.0]), scale=1.0)
    result = propagate_uq(scenario, posterior, n=128, rng=np.random.default_rng(7))

    assert result.result_ref
    assert result.uncertainty is not None
    lo, hi = result.uncertainty["value"]
    assert np.all(hi >= lo)
    assert np.any(hi > lo)  # non-degenerate: real spread from the pushforward, not a point mass


def test_propagate_uq_rejects_a_prior_dominated_zero_draw_count(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_DIR_ENV, str(tmp_path))
    register_surrogate(
        "toy-uq2",
        teacher=_analytic_forward,
        sampler=_uniform_sampler(-2.0, 2.0),
        budget=256,
        seed=3,
    )
    leading = ScenarioStep(op="toy-uq2", inputs_ref="unused", params={})
    scenario = Scenario(steps=[leading])
    posterior = _GaussianPosterior(loc=np.array([0.0, 0.0]), scale=1.0)
    with pytest.raises(ValueError):
        propagate_uq(scenario, posterior, n=0, rng=np.random.default_rng(0))
