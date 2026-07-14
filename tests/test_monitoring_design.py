"""G6 DoD: an EIG-designed monitoring network beats a naive uniform grid on detection time.

Builds a synthetic plume ensemble around a fairly localized source prior, lets
``design_monitoring_network`` greedily pick ``k`` sites out of a coarse candidate grid by
expected information gain about the source (location, rate, onset), and compares its
``expected_detection_time`` against an evenly-spaced ("naive") subset of the same candidate grid
of equal size. An EIG-informed network should concentrate sensors where they are both
informative about the source and close enough to actually see it, so it should detect a
random draw from the same prior sooner, on average, than sensors spread blindly across the
whole domain (several of which sit far from where the plume ever reaches).
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.monitoring_design import (
    GaussianSourcePosterior,
    design_monitoring_network,
    expected_detection_time,
)


def _plume_prior() -> GaussianSourcePosterior:
    # theta = (x0, y0, log_rate, onset); source lives near (80, 0) with modest spread.
    mean = np.array([80.0, 0.0, np.log(40.0), 0.0])
    cov = np.diag([400.0, 900.0, 0.09, 4.0])
    return GaussianSourcePosterior(mean, cov)


def _candidate_grid() -> np.ndarray:
    xs = np.linspace(20.0, 400.0, 6)
    ys = np.linspace(-150.0, 150.0, 6)
    return np.array([[x, y] for x in xs for y in ys])


def _naive_uniform_subset(candidates: np.ndarray, k: int) -> list[int]:
    """An evenly-spaced subset of the candidate grid -- a design that ignores the prior."""
    n = candidates.shape[0]
    idx = sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))
    i = 0
    while len(idx) < k:
        if i not in idx:
            idx.append(i)
        i += 1
    return sorted(idx)[:k]


def _sample_scenarios(prior: GaussianSourcePosterior, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    draws = prior.samples(n, rng)
    return [{"location": (float(d[0]), float(d[1])), "rate": float(np.exp(d[2])), "onset": float(d[3])} for d in draws]


def test_design_beats_naive_uniform_grid_on_detection_time():
    prior = _plume_prior()
    candidates = _candidate_grid()
    k = 6

    chosen = design_monitoring_network(candidates, prior, budget=k, k=k, criterion="eig")
    assert len(chosen) == k
    assert len(set(chosen)) == k
    assert all(0 <= i < candidates.shape[0] for i in chosen)

    naive_idx = _naive_uniform_subset(candidates, k)
    scenarios = _sample_scenarios(prior, 200, seed=0)

    designed_time = expected_detection_time(candidates[chosen], scenarios)
    naive_time = expected_detection_time(candidates[naive_idx], scenarios)

    assert designed_time < naive_time


def test_budget_and_k_cap_the_selection():
    prior = _plume_prior()
    candidates = _candidate_grid()

    assert len(design_monitoring_network(candidates, prior, budget=3, k=6, criterion="eig")) == 3
    assert len(design_monitoring_network(candidates, prior, budget=6, k=3, criterion="eig")) == 3
    assert design_monitoring_network(candidates, prior, budget=0, k=6, criterion="eig") == []


def test_invalid_criterion_rejected():
    prior = _plume_prior()
    candidates = _candidate_grid()
    with pytest.raises(ValueError):
        design_monitoring_network(candidates, prior, budget=3, k=3, criterion="bogus")


def test_nmc_criterion_produces_a_valid_design():
    prior = _plume_prior()
    candidates = _candidate_grid()[:9]  # keep the nested-Monte-Carlo path cheap
    chosen = design_monitoring_network(candidates, prior, budget=3, k=3, criterion="nmc")
    assert len(chosen) == 3
    assert len(set(chosen)) == 3


def test_expected_detection_time_rejects_empty_sites():
    with pytest.raises(ValueError):
        expected_detection_time(np.empty((0, 2)), [{"location": (0.0, 0.0), "rate": 1.0, "onset": 0.0}])
