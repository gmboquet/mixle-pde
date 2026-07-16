"""Tests for mixle_pde.global_sensitivity: Saltelli pick-freeze Sobol' indices and Morris screening.

The acceptance bar for the Sobol' estimator is quantitative, not "it runs": both the Ishigami function and
the Sobol-G ("G-function") test problem have closed-form true first-order and total-order indices (Saltelli
et al. 2008, "Global Sensitivity Analysis: The Primer"), so :func:`sobol_indices` is checked against those
known values directly, at two sample sizes, asserting both a documented tight tolerance at the larger size
and that the error measurably shrinks going from the smaller size to the larger one. The reference formulas
themselves are re-derived from the closed forms in the docstrings below rather than pasted as bare numbers,
so the expected values are traceable to the math, not "trust me."

Morris screening has no closed-form index to check against (it is a *ranking* method, not a variance
decomposition), so it is validated two ways: exactly, on a linear function (where the true elementary effect
is the constant coefficient itself, everywhere -- no sampling noise possible), and qualitatively, on a
function with one obviously-dominant and one obviously-negligible parameter, checking ``mu_star`` ranks them
correctly by a wide margin.

A final test builds a QoI from an actual solver (:func:`mixle_pde.dynamics.integrate_adaptive`, a
Lotka-Volterra ODE) rather than a closed-form analytic function, demonstrating the "solver-agnostic, caller
evaluates the QoI" design end to end; since Lotka-Volterra has no known analytic Sobol' indices, that test
only checks structural sanity (finite, plausible range), not accuracy -- the analytic tests above are the
real accuracy evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.dynamics import integrate_adaptive
from mixle_pde.global_sensitivity import (
    MorrisDesign,
    MorrisResult,
    SaltelliDesign,
    SobolIndices,
    morris_design,
    morris_indices,
    saltelli_design,
    saltelli_sample,
    sobol_indices,
)

# =============================================================================
# Analytic reference problems
# =============================================================================


def _ishigami(points: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """Ishigami & Homma (1990): ``sin(x1) + a*sin(x2)**2 + b*x3**4*sin(x1)``, x_i ~ Uniform(-pi, pi) iid."""
    x1, x2, x3 = points[:, 0], points[:, 1], points[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * (x3**4) * np.sin(x1)


def _ishigami_true_indices(a: float = 7.0, b: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Ishigami Sobol' indices (Saltelli et al. 2008, ch. 4): only X1/X3 interact (V13); X2 is
    purely additive; X3 has zero first-order effect but a nonzero total-order effect entirely via V13.
    """
    v1 = 0.5 * (1.0 + b * np.pi**4 / 5.0) ** 2
    v2 = a**2 / 8.0
    v3 = 0.0
    v13 = 8.0 * b**2 * np.pi**8 / 225.0
    total_variance = v1 + v2 + v3 + v13
    first_order = np.array([v1, v2, v3]) / total_variance
    total_order = np.array([v1 + v13, v2, v13]) / total_variance
    return first_order, total_order


def _sobol_g(points: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Sobol' G-function: ``prod_i (|4*x_i - 2| + a_i) / (1 + a_i)``, x_i ~ Uniform(0, 1) iid.

    Larger ``a_i`` makes parameter ``i`` progressively less influential (``a_i = 0`` is the most influential
    possible setting, ``a_i -> inf`` drives that factor's variance to zero).
    """
    return np.prod((np.abs(4.0 * points - 2.0) + a[None, :]) / (1.0 + a[None, :]), axis=1)


def _sobol_g_true_indices(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Sobol'-G indices: each factor's own variance is ``V_i = 1/(3*(1+a_i)**2)``; because G is a
    product of independent-argument, mean-1 factors, ``Var(G) = prod_i(1 + V_i) - 1`` and the first-/total-
    order indices follow directly (Saltelli et al. 2008, ch. 4).
    """
    v_i = 1.0 / (3.0 * (1.0 + a) ** 2)
    total_variance = np.prod(1.0 + v_i) - 1.0
    first_order = v_i / total_variance
    total_order = v_i * (total_variance + 1.0) / ((1.0 + v_i) * total_variance)
    return first_order, total_order


# =============================================================================
# Sobol' indices: accuracy against known analytic references
# =============================================================================


def test_sobol_indices_ishigami_converges_to_analytic_values():
    s_true, st_true = _ishigami_true_indices()
    bounds = [(-np.pi, np.pi)] * 3

    errors = {}
    for n_base in (256, 16384):
        design = saltelli_sample(bounds, n_base, rng=np.random.default_rng(12345))
        result = sobol_indices(design, _ishigami(design.points))
        errors[n_base] = (
            np.max(np.abs(result.first_order - s_true)),
            np.max(np.abs(result.total_order - st_true)),
        )
        if n_base == 16384:
            large_result = result

    # Documented tolerance at the larger sample size -- the actual accuracy bar.
    max_err_s, max_err_st = errors[16384]
    assert max_err_s < 0.01, f"first-order error {max_err_s} exceeds documented 0.01 tolerance at n_base=16384"
    assert max_err_st < 0.01, f"total-order error {max_err_st} exceeds documented 0.01 tolerance at n_base=16384"

    # Convergence trend: error at the larger sample size is a fraction of the error at the smaller one.
    assert errors[16384][0] < 0.25 * errors[256][0]
    assert errors[16384][1] < 0.25 * errors[256][1]

    # Structural sanity that holds regardless of the specific function: total-order can never be smaller
    # than first-order (the total effect always includes the individual effect), up to estimator noise.
    assert np.all(large_result.total_order >= large_result.first_order - 1.0e-3)

    # X3 has an analytically zero first-order index but a large total-order index (pure interaction with
    # X1, via the x3**4 term) -- the qualitative signature the Ishigami function is famous for.
    assert abs(large_result.first_order[2]) < 0.01
    assert large_result.total_order[2] > 0.2


def test_sobol_indices_sobol_g_converges_to_analytic_values():
    a_coeffs = np.array([0.0, 0.5, 3.0, 9.0, 99.0, 99.0])
    s_true, st_true = _sobol_g_true_indices(a_coeffs)
    bounds = [(0.0, 1.0)] * 6

    errors = {}
    for n_base in (256, 16384):
        design = saltelli_sample(bounds, n_base, rng=np.random.default_rng(777))
        result = sobol_indices(design, _sobol_g(design.points, a_coeffs))
        errors[n_base] = (
            np.max(np.abs(result.first_order - s_true)),
            np.max(np.abs(result.total_order - st_true)),
        )
        if n_base == 16384:
            large_result = result

    max_err_s, max_err_st = errors[16384]
    assert max_err_s < 0.01, f"first-order error {max_err_s} exceeds documented 0.01 tolerance at n_base=16384"
    assert max_err_st < 0.01, f"total-order error {max_err_st} exceeds documented 0.01 tolerance at n_base=16384"

    assert errors[16384][0] < 0.25 * errors[256][0]
    assert errors[16384][1] < 0.25 * errors[256][1]

    assert np.all(large_result.total_order >= large_result.first_order - 1.0e-3)

    # a=[0, 0.5, 3, 9, 99, 99]: strictly decreasing influence as a_i grows, both orderings should agree.
    assert np.all(np.diff(large_result.first_order) < 0.0)
    assert np.all(np.diff(large_result.total_order) < 0.0)
    # The two largest-a_i (near-inert) parameters should read as essentially unimportant.
    assert large_result.total_order[4] < 0.01
    assert large_result.total_order[5] < 0.01


# =============================================================================
# Saltelli design construction: structure and validation
# =============================================================================


def test_saltelli_design_recombination_structure():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=(5, 3))
    b = rng.uniform(size=(5, 3))
    design = saltelli_design(a, b)

    assert isinstance(design, SaltelliDesign)
    assert design.n_base == 5
    assert design.n_params == 3
    assert design.n_evaluations == 5 * (3 + 2)
    assert design.points.shape == (25, 3)

    assert np.array_equal(design.points[0:5], a)
    assert np.array_equal(design.points[5:10], b)
    for i in range(3):
        ab_i = design.points[(2 + i) * 5 : (3 + i) * 5]
        expected = a.copy()
        expected[:, i] = b[:, i]
        assert np.array_equal(ab_i, expected)


def test_saltelli_design_shape_mismatch_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="identical shape"):
        saltelli_design(rng.uniform(size=(5, 3)), rng.uniform(size=(4, 3)))
    with pytest.raises(ValueError, match="2-D"):
        saltelli_design(rng.uniform(size=5), rng.uniform(size=5))


def test_saltelli_design_direct_construction_shape_validation():
    with pytest.raises(ValueError, match="shape"):
        SaltelliDesign(points=np.zeros((10, 3)), n_base=5, n_params=3)  # expects 25 rows, not 10


def test_saltelli_sample_requires_power_of_two_n_base():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="power of two"):
        saltelli_sample(bounds, 100, rng=np.random.default_rng(0))
    # a power of two succeeds.
    design = saltelli_sample(bounds, 128, rng=np.random.default_rng(0))
    assert design.n_base == 128


def test_saltelli_sample_bounds_validation():
    with pytest.raises(ValueError, match="hi > lo"):
        saltelli_sample([(1.0, 0.0)], 8, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least one parameter"):
        saltelli_sample(np.zeros((0, 2)), 8, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="\\(lo, hi\\) pairs"):
        saltelli_sample([(0.0, 1.0, 2.0)], 8, rng=np.random.default_rng(0))


def test_sobol_indices_qoi_shape_mismatch_raises():
    design = saltelli_design(np.zeros((4, 2)), np.ones((4, 2)))  # n_evaluations = 4*4 = 16
    with pytest.raises(ValueError, match="n_evaluations"):
        sobol_indices(design, np.zeros(10))
    with pytest.raises(ValueError, match="1-D"):
        sobol_indices(design, np.zeros((16, 1)))


def test_sobol_indices_constant_qoi_raises():
    design = saltelli_design(np.zeros((4, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError, match="zero"):
        sobol_indices(design, np.full(design.n_evaluations, 3.0))


def test_sobol_indices_result_shape_validation():
    with pytest.raises(ValueError, match="matching shape"):
        SobolIndices(first_order=np.zeros(3), total_order=np.zeros(2), variance=1.0, n_base=4)


# =============================================================================
# Morris elementary-effects screening
# =============================================================================


def test_morris_recovers_exact_coefficients_on_a_linear_function():
    # For f(x) = c . x, the elementary effect of every parameter is exactly c_i at EVERY point (no
    # sampling noise possible), so this is checked to (near) machine precision, not a statistical bound.
    coeffs = np.array([10.0, -3.0, 0.001, 5.0])

    def linear_fn(points: np.ndarray) -> np.ndarray:
        return points @ coeffs

    bounds = [(-2.0, 2.0)] * 4
    design = morris_design(bounds, n_trajectories=1, rng=np.random.default_rng(2))
    result = morris_indices(design, linear_fn(design.points))

    assert np.allclose(result.mu, coeffs, atol=1.0e-9)
    assert np.allclose(result.mu_star, np.abs(coeffs), atol=1.0e-9)
    assert np.allclose(result.sigma, 0.0, atol=1.0e-9)

    # Still exact with several trajectories (a linear function has zero elementary-effect variance).
    design5 = morris_design(bounds, n_trajectories=5, rng=np.random.default_rng(3))
    result5 = morris_indices(design5, linear_fn(design5.points))
    assert np.allclose(result5.mu, coeffs, atol=1.0e-9)
    assert np.allclose(result5.sigma, 0.0, atol=1.0e-9)


def test_morris_ranks_dominant_and_negligible_parameters_correctly():
    # Sobol'-G with a=[0, 99]: parameter 0 is maximally influential, parameter 1 is nearly inert -- an
    # unambiguous "obviously dominant" vs "obviously negligible" pair to screen.
    a_coeffs = np.array([0.0, 99.0])

    def g(points: np.ndarray) -> np.ndarray:
        return _sobol_g(points, a_coeffs)

    bounds = [(0.0, 1.0)] * 2
    design = morris_design(bounds, n_trajectories=32, n_levels=4, rng=np.random.default_rng(4))
    result = morris_indices(design, g(design.points))

    assert result.mu_star[0] > 10.0 * result.mu_star[1]
    assert result.mu_star[1] < 0.1


def test_morris_design_bounds_and_level_validation():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError, match="even integer"):
        morris_design(bounds, n_trajectories=4, n_levels=5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="even integer"):
        morris_design(bounds, n_trajectories=4, n_levels=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="positive integer"):
        morris_design(bounds, n_trajectories=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="hi > lo"):
        morris_design([(1.0, 1.0)], n_trajectories=4, rng=np.random.default_rng(0))


def test_morris_design_structure():
    bounds = [(0.0, 10.0), (-5.0, 5.0), (0.0, 1.0)]
    design = morris_design(bounds, n_trajectories=6, n_levels=4, rng=np.random.default_rng(1))

    assert isinstance(design, MorrisDesign)
    assert design.n_trajectories == 6
    assert design.n_params == 3
    assert design.n_evaluations == 6 * 4
    assert design.points.shape == (24, 3)
    assert design.changed_param.shape == (6, 3)

    # every point stays within bounds.
    lo = np.array([0.0, -5.0, 0.0])
    hi = np.array([10.0, 5.0, 1.0])
    assert np.all(design.points >= lo - 1.0e-9)
    assert np.all(design.points <= hi + 1.0e-9)

    # within each trajectory, each step changes exactly one coordinate, and every parameter is visited
    # exactly once (changed_param's row is a permutation of range(n_params)).
    traj_points = design.points.reshape(6, 4, 3)
    for t in range(6):
        assert sorted(design.changed_param[t].tolist()) == [0, 1, 2]
        for k in range(3):
            diff = traj_points[t, k + 1] - traj_points[t, k]
            changed = np.flatnonzero(np.abs(diff) > 1.0e-12)
            assert list(changed) == [design.changed_param[t, k]]


def test_morris_indices_qoi_shape_mismatch_raises():
    design = morris_design([(0.0, 1.0), (0.0, 1.0)], n_trajectories=3, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="n_evaluations"):
        morris_indices(design, np.zeros(5))
    with pytest.raises(ValueError, match="1-D"):
        morris_indices(design, np.zeros((design.n_evaluations, 1)))


def test_morris_design_direct_construction_shape_validation():
    with pytest.raises(ValueError, match="shape"):
        MorrisDesign(points=np.zeros((5, 2)), changed_param=np.zeros((3, 2), dtype=int), n_trajectories=3, n_params=2)


def test_morris_result_shape_validation():
    with pytest.raises(ValueError, match="matching shape"):
        MorrisResult(mu=np.zeros(3), mu_star=np.zeros(2), sigma=np.zeros(3), n_trajectories=4)


# =============================================================================
# Solver-agnostic design point: a real ODE integrator as the QoI source
# =============================================================================


def test_sobol_indices_on_an_actual_ode_solver_qoi():
    # global_sensitivity itself never imports a solver -- this test is the demonstration that the
    # caller-evaluates-the-QoI contract works end to end with a real one (a Lotka-Volterra predator-prey
    # ODE integrated by mixle_pde.dynamics.integrate_adaptive). Lotka-Volterra has no known closed-form
    # Sobol' indices, so this checks structural sanity (finite, plausible range), not accuracy -- the
    # analytic Ishigami/Sobol-G tests above are the accuracy evidence.
    def final_prey_population(params_row: np.ndarray) -> float:
        growth, predation, decay, conversion = params_row

        def rhs(t, y):
            return [growth * y[0] - predation * y[0] * y[1], -decay * y[1] + conversion * y[0] * y[1]]

        trajectory = integrate_adaptive(rhs, [1.0, 1.0], [5.0], rtol=1.0e-8, atol=1.0e-10)
        return float(trajectory[-1, 0])

    bounds = [(1.0, 2.0), (0.8, 1.2), (2.5, 3.5), (0.8, 1.2)]
    design = saltelli_sample(bounds, n_base=32, rng=np.random.default_rng(5))
    qoi = np.array([final_prey_population(row) for row in design.points])

    result = sobol_indices(design, qoi)

    assert np.all(np.isfinite(result.first_order))
    assert np.all(np.isfinite(result.total_order))
    # generous plausible-range check (not an accuracy check -- see docstring above).
    assert np.all(result.first_order > -0.5) and np.all(result.first_order < 1.5)
    assert np.all(result.total_order > -0.5) and np.all(result.total_order < 1.5)
