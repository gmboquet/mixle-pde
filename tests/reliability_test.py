"""Tests for mixle_pde.reliability: FORM (HLRF) design-point search + importance-sampling cross-check.

The acceptance bar mirrors tests/global_sensitivity_test.py's own standard: closed-form analytic
references, not just "it runs". Three independent references are used:

* A **linear** limit state, for which FORM is mathematically exact (the tangent-hyperplane
  approximation coincides with the true limit-state surface when that surface already is a hyperplane).
  ``beta``, ``design_point``, ``alpha``, and ``probability`` are all checked against values derived
  directly from the closed-form geometry, not merely "runs without error".
* A **radially symmetric quadratic** limit state (``c - ||u||^2``), for which the true failure
  probability has an exact closed form: ``||U||^2`` for standard-normal ``U`` in 2 dimensions is
  chi-squared with 2 degrees of freedom, whose survival function is ``exp(-c/2)`` (a standard,
  independently-checkable fact -- chi-squared-2 is exponential with rate 1/2). This is used to
  demonstrate FORM's genuine linearization bias: ``beta`` itself is found exactly (the closest point on
  a circle to the origin is an elementary, curvature-independent computation), but converting that
  ``beta`` into a probability via ``Phi(-beta)`` treats the curved circular boundary as a flat tangent
  line and is measurably, substantially wrong here -- the point the module's docstring makes explicit
  and the reason :func:`importance_sampling_probability` exists at all.
* A **rare-event linear** limit state (large beta, small true probability from the same closed form
  ``scipy.stats.norm.cdf(-beta)`` used above), used to validate
  :func:`importance_sampling_probability` both for accuracy (against the closed form) and for honest
  self-calibration (its reported ``standard_error`` is checked against the *empirical* spread of many
  independently-seeded repeated estimates, not merely asserted to look small) and for its practical
  purpose (variance reduction: plain crude Monte Carlo at the identical sample budget observes zero
  failure draws at all, i.e. reports nothing).

The radially-symmetric quadratic case is deliberately *not* also used to validate
:func:`importance_sampling_probability`: a circle has no unique closest point (every point on it is
equally close to the origin), so centering an importance density on any single one of them is a poor
proposal for that specific integrand -- a fact about that test problem's angular symmetry, not about
the estimator, which is why the IS accuracy evidence instead uses the linear case, where the design
point is genuinely unique and dominant.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mixle_pde.reliability import (
    FORMResult,
    ImportanceSamplingResult,
    form,
    importance_sampling_probability,
    physical_to_standard_normal,
    standard_normal_to_physical,
)

# =============================================================================
# FORM: exact on a linear limit state
# =============================================================================


def test_form_linear_limit_state_is_exact():
    rng = np.random.default_rng(0)
    a = rng.normal(size=3)
    a = a / np.linalg.norm(a)
    b = 2.5

    def g(u: np.ndarray) -> float:
        return b - float(np.dot(a, u))

    result = form(g, dim=3)

    assert isinstance(result, FORMResult)
    assert result.converged
    assert result.n_iterations <= 3
    assert result.beta == pytest.approx(b, abs=1.0e-8)
    assert result.probability == pytest.approx(float(stats.norm.cdf(-b)), abs=1.0e-8)
    assert np.allclose(result.design_point, b * a, atol=1.0e-8)
    assert np.allclose(result.alpha, a, atol=1.0e-8)
    # alpha is a unit vector by construction (design_point / beta).
    assert np.linalg.norm(result.alpha) == pytest.approx(1.0, abs=1.0e-8)


def test_form_gradient_callable_matches_finite_difference_and_uses_fewer_evaluations():
    a = np.array([0.6, 0.8])  # already a unit vector
    b = 2.5

    def g(u: np.ndarray) -> float:
        return b - float(np.dot(a, u))

    def gradient(u: np.ndarray) -> np.ndarray:
        return -a

    result_fd = form(g, dim=2)
    result_analytic = form(g, dim=2, gradient=gradient)

    assert result_analytic.beta == pytest.approx(result_fd.beta, abs=1.0e-8)
    assert np.allclose(result_analytic.design_point, result_fd.design_point, atol=1.0e-8)
    # Supplying an analytic gradient must skip the 2*dim finite-difference evaluations per iteration.
    assert result_analytic.n_evaluations < result_fd.n_evaluations


# =============================================================================
# FORM: genuine linearization bias on a curved (quadratic) limit state
# =============================================================================


def test_form_quadratic_limit_state_shows_real_linearization_bias():
    # g(u) = c - ||u||^2: failure is {||u|| >= sqrt(c)}, a circle of radius sqrt(c). The closest point on
    # a circle to the origin is at distance exactly sqrt(c), found exactly regardless of curvature; the
    # true failure probability, P(chi2_2 >= c) = exp(-c/2), is NOT Phi(-sqrt(c)) -- that equality only
    # holds for a flat (hyperplane) boundary. Starting from u0=0 is invalid here (g is stationary at the
    # origin -- see test_form_vanishing_gradient_raises_a_clear_error), so start from an offset point.
    c = 9.0

    def g(u: np.ndarray) -> float:
        return float(c - np.dot(u, u))

    result = form(g, dim=2, u0=np.array([0.3, 0.2]))

    assert result.converged
    assert result.beta == pytest.approx(np.sqrt(c), abs=1.0e-6)

    true_probability = float(np.exp(-c / 2.0))
    assert result.probability == pytest.approx(0.0013498980, abs=1.0e-6)
    assert true_probability == pytest.approx(0.0111089965, abs=1.0e-6)

    # The bias is real and large (roughly 8x here), and directional: the safe region {||u|| <= sqrt(c)}
    # is convex, which is the standard textbook setup where FORM's tangent-plane approximation
    # *underestimates* the true (curved) failure probability.
    assert result.probability < true_probability
    assert true_probability / result.probability > 5.0


def test_form_vanishing_gradient_raises_a_clear_error():
    # The quadratic case above is stationary (grad = -2u = 0) exactly at the default u0=0 starting point.
    def g(u: np.ndarray) -> float:
        return float(1.0 - np.dot(u, u))

    with pytest.raises(ValueError, match="gradient vanished"):
        form(g, dim=2)

    # Also reachable via an explicitly supplied degenerate gradient callable, independent of geometry.
    def g_any(u: np.ndarray) -> float:
        return float(1.0 - np.sum(u**2))

    with pytest.raises(ValueError, match="gradient vanished"):
        form(g_any, dim=2, gradient=lambda u: np.zeros(2), u0=np.array([0.1, 0.1]))


def test_form_requires_a_safe_starting_point():
    def g(u: np.ndarray) -> float:
        return -1.0 - float(np.sum(u))  # g(0) = -1 <= 0: origin already fails.

    with pytest.raises(ValueError, match="requires g\\(u0\\) > 0"):
        form(g, dim=2)


# =============================================================================
# FORM: input and result validation
# =============================================================================


def test_form_input_validation():
    def g(u: np.ndarray) -> float:
        return 1.0 - float(np.sum(u))

    with pytest.raises(ValueError, match="positive integer"):
        form(g, dim=0)
    with pytest.raises(ValueError, match="shape"):
        form(g, dim=2, u0=np.zeros(3))
    with pytest.raises(ValueError, match="positive integer"):
        form(g, dim=2, max_iter=0)
    with pytest.raises(ValueError, match="tol must be positive"):
        form(g, dim=2, tol=0.0)
    with pytest.raises(ValueError, match="finite_diff_step must be positive"):
        form(g, dim=2, finite_diff_step=-1.0e-6)


def test_form_result_shape_and_range_validation():
    with pytest.raises(ValueError, match="matching shape"):
        FORMResult(
            beta=1.0,
            probability=0.1,
            design_point=np.zeros(3),
            alpha=np.zeros(2),
            converged=True,
            n_iterations=1,
            n_evaluations=1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        FORMResult(
            beta=-1.0,
            probability=0.1,
            design_point=np.zeros(2),
            alpha=np.zeros(2),
            converged=True,
            n_iterations=1,
            n_evaluations=1,
        )
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        FORMResult(
            beta=1.0,
            probability=1.5,
            design_point=np.zeros(2),
            alpha=np.zeros(2),
            converged=True,
            n_iterations=1,
            n_evaluations=1,
        )


# =============================================================================
# Importance sampling: accuracy, honest self-calibration, and variance reduction
# =============================================================================


def test_importance_sampling_matches_closed_form_on_a_rare_linear_event():
    rng = np.random.default_rng(3)
    a = rng.normal(size=4)
    a = a / np.linalg.norm(a)
    b = 4.0  # true probability ~ 3.17e-5: a genuine rare event.

    def g(u: np.ndarray) -> float:
        return b - float(np.dot(a, u))

    def g_batch(points: np.ndarray) -> np.ndarray:
        return b - points @ a

    design = form(g, dim=4)
    true_probability = float(stats.norm.cdf(-b))

    result = importance_sampling_probability(
        g_batch, dim=4, shift=design.design_point, n_samples=20_000, rng=np.random.default_rng(11)
    )

    assert isinstance(result, ImportanceSamplingResult)
    assert abs(result.probability - true_probability) < 3.0 * result.standard_error
    assert result.coefficient_of_variation < 0.1  # a trustworthy-by-convention estimate
    assert 0.0 < result.n_effective <= result.n_samples


def test_importance_sampling_standard_error_is_honestly_calibrated():
    # The single-run standard_error should agree with the *empirical* spread of many independently
    # seeded repeats -- the real test of whether the reported diagnostic can be trusted, not just
    # whether it looks like a plausible small number.
    a = np.array([1.0, 0.0, 0.0])
    b = 3.5

    def g_batch(points: np.ndarray) -> np.ndarray:
        return b - points @ a

    shift = b * a
    single = importance_sampling_probability(g_batch, dim=3, shift=shift, n_samples=8_000, rng=np.random.default_rng(0))

    repeats = np.array(
        [
            importance_sampling_probability(
                g_batch, dim=3, shift=shift, n_samples=8_000, rng=np.random.default_rng(seed)
            ).probability
            for seed in range(30)
        ]
    )
    empirical_std = float(np.std(repeats, ddof=1))

    # Within a factor of 2 of each other -- a loose but genuine cross-check (both quantities are
    # themselves statistical estimates from finite samples), not an exact-equality assertion.
    assert 0.5 * single.standard_error < empirical_std < 2.0 * single.standard_error


def test_importance_sampling_beats_crude_monte_carlo_at_the_same_sample_budget():
    a = np.array([1.0, 0.0])
    b = 4.5  # true probability ~ 3.4e-6

    def g_batch(points: np.ndarray) -> np.ndarray:
        return b - points @ a

    n_samples = 5_000
    result = importance_sampling_probability(
        g_batch, dim=2, shift=b * a, n_samples=n_samples, rng=np.random.default_rng(5)
    )
    assert result.probability > 0.0  # a genuine, nonzero, precise estimate

    crude = np.random.default_rng(6).normal(size=(n_samples, 2))
    crude_hits = np.sum(g_batch(crude) <= 0.0)
    assert crude_hits == 0  # crude MC at the identical budget sees the rare event zero times


def test_importance_sampling_input_validation():
    def g_batch(points: np.ndarray) -> np.ndarray:
        return 1.0 - points[:, 0]

    with pytest.raises(ValueError, match="positive integer"):
        importance_sampling_probability(g_batch, dim=0, shift=np.zeros(0), n_samples=100)
    with pytest.raises(ValueError, match="shape"):
        importance_sampling_probability(g_batch, dim=2, shift=np.zeros(3), n_samples=100)
    with pytest.raises(ValueError, match="at least 2"):
        importance_sampling_probability(g_batch, dim=2, shift=np.zeros(2), n_samples=1)

    def g_wrong_shape(points: np.ndarray) -> np.ndarray:
        return np.zeros(len(points) + 1)

    with pytest.raises(ValueError, match="shape"):
        importance_sampling_probability(g_wrong_shape, dim=2, shift=np.zeros(2), n_samples=10)


def test_importance_sampling_result_validation():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ImportanceSamplingResult(
            probability=1.5, standard_error=0.01, coefficient_of_variation=0.1, n_samples=10, n_effective=5.0
        )
    with pytest.raises(ValueError, match="non-negative"):
        ImportanceSamplingResult(
            probability=0.1, standard_error=-0.01, coefficient_of_variation=0.1, n_samples=10, n_effective=5.0
        )
    with pytest.raises(ValueError, match="positive integer"):
        ImportanceSamplingResult(
            probability=0.1, standard_error=0.01, coefficient_of_variation=0.1, n_samples=0, n_effective=5.0
        )
    with pytest.raises(ValueError, match="non-negative"):
        ImportanceSamplingResult(
            probability=0.1, standard_error=0.01, coefficient_of_variation=0.1, n_samples=10, n_effective=-1.0
        )


# =============================================================================
# Isoprobabilistic transform
# =============================================================================


def test_standard_normal_physical_transform_round_trip():
    marginals = [
        stats.norm(loc=5.0, scale=2.0),
        stats.lognorm(s=0.5, scale=np.exp(1.0)),
        stats.uniform(loc=-3.0, scale=6.0),
    ]
    rng = np.random.default_rng(1)
    u = rng.normal(size=(50, 3))

    x = standard_normal_to_physical(u, marginals)
    u_roundtrip = physical_to_standard_normal(x, marginals)

    assert np.allclose(u, u_roundtrip, atol=1.0e-8)


def test_transform_shape_validation():
    marginals = [stats.norm(), stats.norm()]
    with pytest.raises(ValueError, match="length"):
        standard_normal_to_physical(np.zeros((5, 3)), marginals)
    with pytest.raises(ValueError, match="length"):
        physical_to_standard_normal(np.zeros((5, 3)), marginals)
    with pytest.raises(ValueError, match="at least one"):
        standard_normal_to_physical(np.zeros((5, 0)), [])


# =============================================================================
# End-to-end workflow: FORM design point feeding the importance-sampling cross-check
# =============================================================================


def test_form_and_importance_sampling_agree_closely_when_the_limit_state_is_linear():
    # The documented workflow (form() then importance_sampling_probability() at its design point) on a
    # limit state where FORM itself is exact: the two independently-derived estimates should land close
    # to each other and to the shared closed-form truth, not just each individually close to the truth.
    a = np.array([0.28, 0.96])
    b = 3.0

    def g(u: np.ndarray) -> float:
        return b - float(np.dot(a, u))

    def g_batch(points: np.ndarray) -> np.ndarray:
        return b - points @ a

    design = form(g, dim=2)
    refined = importance_sampling_probability(
        g_batch, dim=2, shift=design.design_point, n_samples=50_000, rng=np.random.default_rng(21)
    )

    true_probability = float(stats.norm.cdf(-b))
    assert design.probability == pytest.approx(true_probability, abs=1.0e-8)
    assert abs(refined.probability - true_probability) < 4.0 * refined.standard_error
    assert abs(refined.probability - design.probability) < 4.0 * refined.standard_error
