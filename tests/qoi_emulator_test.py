"""Smoke tests for mixle_pde.qoi_emulator (MP-N4): the shared QoIEmulator contract, the Gaussian-
process and Bayesian-polynomial emulator families built on it, and the generic calibrate() routine
both families share.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mixle_pde.design_of_experiments import latin_hypercube_design
from mixle_pde.qoi_emulator import (
    BayesianPolynomialEmulator,
    CalibrationReport,
    EmulatorPrediction,
    GaussianProcessEmulator,
    QoIEmulator,
    calibrate,
    fit_bayesian_polynomial,
    fit_gaussian_process,
)

_BOUNDS = [(-2.0, 2.0), (-2.0, 2.0)]


def _smooth_teacher(points: np.ndarray) -> np.ndarray:
    """A smooth, deterministic 2-D scalar function -- well within a GP/low-degree-polynomial's reach."""
    a, b = points[:, 0], points[:, 1]
    return np.sin(a) + 0.3 * b**2 - 0.5 * a * b


def _fit_xy(bounds, n, *, seed):
    points, _ = latin_hypercube_design(bounds, n, seed=seed)
    return points, _smooth_teacher(points)


# ---------------------------------------------------------------------------
# EmulatorPrediction / CalibrationReport self-validation
# ---------------------------------------------------------------------------


def test_emulator_prediction_rejects_malformed_fields():
    valid = EmulatorPrediction(mean=np.array([0.0, 1.0]), std=np.array([0.1, 0.2]), in_domain=np.array([True, False]))
    assert valid.mean.shape == (2,)

    with pytest.raises(ValueError):
        EmulatorPrediction(mean=np.array([0.0, 1.0]), std=np.array([0.1]), in_domain=np.array([True, False]))
    with pytest.raises(ValueError):
        EmulatorPrediction(mean=np.array([0.0, np.nan]), std=np.array([0.1, 0.2]), in_domain=np.array([True, False]))
    with pytest.raises(ValueError):
        EmulatorPrediction(mean=np.array([0.0, 1.0]), std=np.array([0.1, 0.0]), in_domain=np.array([True, False]))
    with pytest.raises(ValueError):
        EmulatorPrediction(mean=np.array([0.0, 1.0]), std=np.array([0.1, -0.2]), in_domain=np.array([True, False]))


def test_calibration_report_rejects_out_of_range_fields():
    report = CalibrationReport(
        n=5,
        mae=0.1,
        rmse=0.2,
        coverage_1sigma=0.7,
        coverage_2sigma=0.95,
        mean_standardized_residual=0.0,
        ood_fraction=0.1,
    )
    assert report.n == 5

    with pytest.raises(ValueError):
        CalibrationReport(
            n=0,
            mae=0.1,
            rmse=0.2,
            coverage_1sigma=0.7,
            coverage_2sigma=0.95,
            mean_standardized_residual=0.0,
            ood_fraction=0.1,
        )
    with pytest.raises(ValueError):
        CalibrationReport(
            n=5,
            mae=0.1,
            rmse=0.2,
            coverage_1sigma=1.5,
            coverage_2sigma=0.95,
            mean_standardized_residual=0.0,
            ood_fraction=0.1,
        )


# ---------------------------------------------------------------------------
# fit_gaussian_process
# ---------------------------------------------------------------------------


def test_gaussian_process_fits_a_smooth_function_with_low_error_and_reasonable_coverage():
    x_train, y_train = _fit_xy(_BOUNDS, 40, seed=1)
    emulator = fit_gaussian_process(x_train, y_train, noise_variance=1e-4)
    assert isinstance(emulator, GaussianProcessEmulator)

    x_holdout, y_holdout = _fit_xy(_BOUNDS, 200, seed=2)
    report = calibrate(emulator, x_holdout, y_holdout)
    assert isinstance(report, CalibrationReport)
    assert report.n == 200
    assert report.rmse < 0.3
    assert 0.4 <= report.coverage_1sigma <= 1.0
    assert report.coverage_2sigma >= report.coverage_1sigma


def test_gaussian_process_is_deterministic_given_fixed_data():
    x_train, y_train = _fit_xy(_BOUNDS, 24, seed=3)
    a = fit_gaussian_process(x_train, y_train)
    b = fit_gaussian_process(x_train, y_train)
    probe = np.array([[0.3, -0.4], [1.1, 1.2]])
    pred_a = a.predict(probe)
    pred_b = b.predict(probe)
    np.testing.assert_array_equal(pred_a.mean, pred_b.mean)
    np.testing.assert_array_equal(pred_a.std, pred_b.std)


def test_gaussian_process_flags_out_of_domain_far_from_training_data():
    x_train, y_train = _fit_xy(_BOUNDS, 30, seed=4)
    emulator = fit_gaussian_process(x_train, y_train)

    inside_pred = emulator.predict(np.array([[0.0, 0.0]]))
    outside_pred = emulator.predict(np.array([[500.0, -500.0]]))
    assert bool(inside_pred.in_domain[0]) is True
    assert bool(outside_pred.in_domain[0]) is False
    assert outside_pred.std[0] > inside_pred.std[0]


def test_fit_gaussian_process_rejects_non_positive_noise_variance_and_mismatched_lengths():
    x_train, y_train = _fit_xy(_BOUNDS, 10, seed=5)
    with pytest.raises(ValueError):
        fit_gaussian_process(x_train, y_train, noise_variance=0.0)
    with pytest.raises(ValueError):
        fit_gaussian_process(x_train, y_train[:-1])


# ---------------------------------------------------------------------------
# fit_bayesian_polynomial
# ---------------------------------------------------------------------------


def test_bayesian_polynomial_fits_a_smooth_function_and_shares_the_calibrate_contract():
    x_train, y_train = _fit_xy(_BOUNDS, 40, seed=6)
    emulator = fit_bayesian_polynomial(x_train, y_train, degree=3, noise_variance=1e-2)
    assert isinstance(emulator, BayesianPolynomialEmulator)

    x_holdout, y_holdout = _fit_xy(_BOUNDS, 200, seed=7)
    report = calibrate(emulator, x_holdout, y_holdout)  # the exact same calibrate() used for the GP above
    assert isinstance(report, CalibrationReport)
    assert report.n == 200
    assert np.isfinite(report.rmse)


def test_bayesian_polynomial_flags_out_of_domain_far_from_training_data():
    x_train, y_train = _fit_xy(_BOUNDS, 30, seed=8)
    emulator = fit_bayesian_polynomial(x_train, y_train, degree=2)

    inside_pred = emulator.predict(np.array([[0.0, 0.0]]))
    outside_pred = emulator.predict(np.array([[500.0, -500.0]]))
    assert bool(inside_pred.in_domain[0]) is True
    assert bool(outside_pred.in_domain[0]) is False


def test_fit_bayesian_polynomial_rejects_non_positive_hyperparameters_and_too_few_points():
    x_train, y_train = _fit_xy(_BOUNDS, 10, seed=9)
    with pytest.raises(ValueError):
        fit_bayesian_polynomial(x_train, y_train, noise_variance=0.0)
    with pytest.raises(ValueError):
        fit_bayesian_polynomial(x_train, y_train, prior_variance=-1.0)
    with pytest.raises(ValueError):
        fit_bayesian_polynomial(x_train[:2], y_train[:2], degree=5)  # 21 degree-5/2-D features, 2 points


def test_gp_and_polynomial_emulators_both_satisfy_the_shared_qoi_emulator_contract():
    x_train, y_train = _fit_xy(_BOUNDS, 20, seed=11)
    gp = fit_gaussian_process(x_train, y_train)
    poly = fit_bayesian_polynomial(x_train, y_train, degree=2)
    assert isinstance(gp, QoIEmulator)
    assert isinstance(poly, QoIEmulator)


# ---------------------------------------------------------------------------
# Worked example: a QoI emulator of a real registered PDE kernel (mixle_pde.fem.solve_simplex_poisson)
# ---------------------------------------------------------------------------


def test_worked_example_gaussian_process_emulates_a_registered_fem_kernel_qoi():
    """mixle_pde.fem.solve_simplex_poisson (already registered/tested elsewhere in this package) used
    only as a realistic teacher forward -- the same pattern tests/design_of_experiments_test.py uses
    for its own worked example. fem.py itself is untouched."""
    from mixle_pde.fem import solve_simplex_poisson
    from mixle_pde.mesh import box_simplex_mesh

    mesh = box_simplex_mesh((6, 6), lengths=(1.0, 1.0))

    def teacher_mean_field(diffusion: float, magnitude: float) -> float:
        u = solve_simplex_poisson(mesh, source=float(magnitude), diffusion=float(diffusion))
        return float(u.mean())

    bounds = [(0.5, 3.0), (0.5, 3.0)]
    train_points, _ = latin_hypercube_design(bounds, 24, seed=10)
    train_values = np.array([teacher_mean_field(d, m) for d, m in train_points])

    emulator = fit_gaussian_process(train_points, train_values, noise_variance=1e-6)

    low_regime = np.array([[3.0, 0.5]])  # low source, high diffusion -> small mean solution
    high_regime = np.array([[0.5, 3.0]])  # high source, low diffusion -> large mean solution

    low_pred = emulator.predict(low_regime)
    high_pred = emulator.predict(high_regime)

    assert math.isfinite(low_pred.mean[0])
    assert abs(high_pred.mean[0] - teacher_mean_field(0.5, 3.0)) < 0.05
    assert high_pred.mean[0] > low_pred.mean[0]  # learned real input sensitivity, not a constant
    assert bool(low_pred.in_domain[0]) and bool(high_pred.in_domain[0])
