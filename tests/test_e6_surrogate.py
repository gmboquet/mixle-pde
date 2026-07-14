"""DoD conformance test for E6 -- physics-surrogate distillation + cascade.

Uses a cheap analytic function as a stand-in "teacher" for an expensive PDE forward: distills a fast
student from it, and checks (a) the surrogate tracks the full forward within tolerance on in-distribution
("easy") inputs and (b) the calibrated gate defers out-of-distribution ("hard") inputs back to the full
forward at a materially higher rate than it does on the easy split.
"""

from __future__ import annotations

import numpy as np

from mixle_pde.surrogate import Surrogate, distill_forward


def _analytic_forward(x):
    """A cheap stand-in for an expensive PDE forward: params -> a 2-channel synthetic observation."""
    a, b = float(x[0]), float(x[1])
    return (np.sin(a) + 0.3 * b, a * a - 0.2 * b)


def _in_distribution_sampler(n, rng):
    """Draw ``n`` record inputs (tuples) from the training domain the surrogate is distilled over."""
    a = rng.uniform(-1.0, 1.0, size=n)
    b = rng.uniform(-1.0, 1.0, size=n)
    return [(float(ai), float(bi)) for ai, bi in zip(a, b)]


def test_e6_surrogate_matches_teacher_on_easy_inputs_and_defers_hard_inputs():
    surrogate = distill_forward(_analytic_forward, _in_distribution_sampler, budget=96, seed=0)
    assert isinstance(surrogate, Surrogate)

    rng = np.random.RandomState(123)

    # Easy split: fresh in-distribution draws -- the surrogate should track the full forward closely
    # and rarely defer.
    easy_inputs = _in_distribution_sampler(40, rng)
    errs = []
    for x in easy_inputs:
        yhat = np.asarray(surrogate.predict(x), dtype=float)
        ytrue = np.asarray(_analytic_forward(x), dtype=float)
        errs.append(np.max(np.abs(yhat - ytrue)))
    assert np.mean(errs) < 0.25, f"surrogate should track the full solve within tolerance, got {np.mean(errs)}"

    easy_deferred = np.mean([surrogate.defer(x) for x in easy_inputs])

    # Hard split: far outside the training domain -- the calibrated gate should escalate these.
    hard_inputs = [(float(a), float(b)) for a, b in zip(rng.uniform(8, 12, size=40), rng.uniform(8, 12, size=40))]
    hard_deferred = np.mean([surrogate.defer(x) for x in hard_inputs])

    assert hard_deferred > easy_deferred, (
        f"deferral rate should rise on the out-of-distribution split: easy={easy_deferred}, hard={hard_deferred}"
    )
    assert hard_deferred >= 0.8
    assert easy_deferred <= 0.2


def test_e6_scalar_teacher_output_is_supported():
    def scalar_forward(x):
        return float(x[0]) ** 2 + float(x[1])

    def sampler(n, rng):
        return [(float(v), float(w)) for v, w in zip(rng.uniform(-1, 1, size=n), rng.uniform(-1, 1, size=n))]

    surrogate = distill_forward(scalar_forward, sampler, budget=48, seed=1)
    x = (0.1, 0.2)
    yhat = surrogate.predict(x)
    assert isinstance(yhat, float)
    assert abs(yhat - scalar_forward(x)) < 0.3


def test_e6_solve_falls_back_to_teacher_when_deferred():
    calls = {"n": 0}

    def counting_forward(x):
        calls["n"] += 1
        return (np.sin(float(x[0])),)

    def sampler(n, rng):
        return [(float(v),) for v in rng.uniform(-1, 1, size=n)]

    surrogate = distill_forward(counting_forward, sampler, budget=48, seed=2)
    baseline = calls["n"]
    far_ood = (50.0,)
    assert surrogate.defer(far_ood)
    out = surrogate.solve(far_ood)
    assert calls["n"] == baseline + 1  # escalated to the real teacher exactly once
    assert out == counting_forward(far_ood) or np.allclose(out, counting_forward(far_ood))
