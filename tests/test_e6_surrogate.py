"""E6 -- physics-surrogate distillation + cascade: DoD conformance.

A cheap analytic function stands in for an expensive PDE forward. The surrogate distilled from it
should (1) match the real forward within tolerance on inputs like the ones it trained on, and (2)
honestly defer inputs that are unlike anything it trained on -- with the deferral rate rising on a
clearly out-of-distribution ("hard") split relative to the in-distribution ("easy") one.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from mixle_pde.surrogate import Surrogate, distill_forward


def _analytic_forward(x):
    """Cheap stand-in "teacher": a smooth nonlinear scalar function of two physical parameters."""
    a, b = x
    return math.sin(a) + 0.5 * b * b


def _uniform_sampler(lo, hi):
    def sampler(n, rng):
        return rng.uniform(lo, hi, size=(n, 2))

    return sampler


@pytest.fixture(scope="module")
def surrogate() -> Surrogate:
    return distill_forward(
        _analytic_forward,
        _uniform_sampler(-2.0, 2.0),
        budget=80,
        seed=0,
        holdout=0.3,
    )


def test_frozen_signature():
    params = list(inspect.signature(distill_forward).parameters)
    # the frozen call surface: positional teacher/sampler, keyword-only budget (required) and seed=0
    assert params[:2] == ["teacher", "sampler"]
    assert "budget" in params and "seed" in params
    sig = inspect.signature(distill_forward)
    assert sig.parameters["seed"].default == 0
    assert sig.parameters["budget"].kind == inspect.Parameter.KEYWORD_ONLY
    assert hasattr(Surrogate, "predict") and hasattr(Surrogate, "defer")


def test_surrogate_matches_teacher_within_tolerance_on_easy_inputs(surrogate):
    rng = np.random.default_rng(1)
    easy = rng.uniform(-1.5, 1.5, size=(30, 2))

    errors = [abs(surrogate.predict(x) - _analytic_forward(x)) for x in easy]
    assert np.mean(errors) < 0.3
    assert np.max(errors) < 0.75

    easy_deferrals = [surrogate.defer(x) for x in easy]
    assert np.mean(easy_deferrals) <= 0.2  # in-distribution: mostly trusted, not escalated


def test_deferral_rate_rises_on_the_hard_out_of_distribution_split(surrogate):
    rng = np.random.default_rng(2)
    easy = rng.uniform(-1.5, 1.5, size=(30, 2))
    hard = rng.uniform(7.0, 9.0, size=(30, 2))  # far outside the [-2, 2]^2 training range

    easy_rate = float(np.mean([surrogate.defer(x) for x in easy]))
    hard_rate = float(np.mean([surrogate.defer(x) for x in hard]))

    assert hard_rate > easy_rate
    assert hard_rate >= 0.6


def test_surrogate_is_not_globally_imprecise(surrogate):
    # the calibrated interval should be well inside the precision floor -- otherwise everything
    # would defer regardless of distribution, which would make the OOD-rise assertion meaningless
    assert surrogate.imprecise is False
    assert surrogate.qhat[0] < surrogate.tol[0]


def test_evaluate_reports_error_and_deferral_rate(surrogate):
    rng = np.random.default_rng(3)
    xs = list(rng.uniform(-1.5, 1.5, size=(10, 2)))
    report = surrogate.evaluate(xs)
    assert report["n"] == 10
    assert report["mae"][0] < 0.5
    assert 0.0 <= report["deferral_rate"] <= 1.0


def test_distill_forward_rejects_too_small_a_budget():
    with pytest.raises(ValueError):
        distill_forward(_analytic_forward, _uniform_sampler(-1.0, 1.0), budget=4, seed=0)


def test_to_task_cascade_adapter_hosts_the_surrogate_behind_the_platform_surface(surrogate):
    mlops = pytest.importorskip("mixle_mlops.models.task_cascade")
    from mixle_pde.surrogate import to_task_cascade_adapter

    adapter = to_task_cascade_adapter(surrogate, "physics-surrogate-demo")
    assert isinstance(adapter, mlops.TaskCascadeAdapter)
    assert adapter.capabilities() >= {"chat", "predict"}
    assert "score" not in adapter.capabilities()  # no label distribution behind a continuous forward
