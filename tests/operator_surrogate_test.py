"""Tests for mixle_pde.operator_surrogate: the linear/low-rank field-to-field operator surrogate.

Three groups, mirroring tests/reduced_basis_test.py's own structure:

1. A hand-built analytic case with an *exactly known* low-rank linear map (and, deliberately,
   ``n_dof_in != n_dof_out`` -- different input/output field dimensions -- to exercise MP-N5's
   "geometry-varying" requirement structurally): the fitted surrogate must recover that map to near
   machine precision on fresh queries drawn from the same distribution as training.
2. Shape/argument validation error paths.
3. A realistic-kernel scenario built from an already-registered, unmodified kernel
   (:class:`mixle_pde.dynamics.AdvectionDiffusionOperator`) swept over several diffusivities with several
   randomly-shaped initial fields each, split into disjoint (interleaved) train/held-out diffusivity grids.
   Because the true input-to-output map genuinely differs per diffusivity, a single fitted linear map can
   only ever be an imperfect compromise across the family -- unlike group 1's exact-recovery case, this is
   the honest "does the calibration/OOD gate behave sensibly on a real, imperfectly-linear generalization
   problem" check: more rank measurably helps and flips the precision-floor gate, and the out-of-domain gate
   catches an input field unlike anything trained on while leaving in-distribution queries alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.dynamics import AdvectionDiffusionOperator
from mixle_pde.operator_surrogate import (
    LinearOperatorCalibrationReport,
    LinearOperatorSurrogate,
    OperatorPrediction,
    calibrate_linear_operator_surrogate,
    fit_linear_operator_surrogate,
)

# --- group 1: hand-built exact low-rank recovery, n_dof_in != n_dof_out --------------------------------

_N_DOF_IN = 14
_N_DOF_OUT = 22
_TRUE_RANK = 3


@pytest.fixture(scope="module")
def exact_low_rank_problem():
    rng = np.random.default_rng(0)
    basis_true_in, _ = np.linalg.qr(rng.standard_normal((_N_DOF_IN, _TRUE_RANK)))
    basis_true_out, _ = np.linalg.qr(rng.standard_normal((_N_DOF_OUT, _TRUE_RANK)))
    core = rng.standard_normal((_TRUE_RANK, _TRUE_RANK))

    def true_operator(u_column: np.ndarray) -> np.ndarray:
        return basis_true_out @ (core @ (basis_true_in.T @ u_column))

    n_train = 60
    latent_train = rng.standard_normal((_TRUE_RANK, n_train))
    inputs_train = basis_true_in @ latent_train
    outputs_train = np.stack([true_operator(inputs_train[:, i]) for i in range(n_train)], axis=1)

    n_test = 20
    latent_test = rng.standard_normal((_TRUE_RANK, n_test))
    inputs_test = basis_true_in @ latent_test
    outputs_test = np.stack([true_operator(inputs_test[:, i]) for i in range(n_test)], axis=1)

    return inputs_train, outputs_train, inputs_test, outputs_test


def test_fit_recovers_exact_low_rank_map_on_fresh_queries(exact_low_rank_problem):
    inputs_train, outputs_train, inputs_test, outputs_test = exact_low_rank_problem
    surrogate = fit_linear_operator_surrogate(
        inputs_train, outputs_train, rank_in=_TRUE_RANK, rank_out=_TRUE_RANK, ridge=1e-10
    )

    assert isinstance(surrogate, LinearOperatorSurrogate)
    assert surrogate.basis_in.n_dof == _N_DOF_IN
    assert surrogate.basis_out.n_dof == _N_DOF_OUT
    assert surrogate.coefficient_map.shape == (_TRUE_RANK, _TRUE_RANK)

    prediction = surrogate.predict(inputs_test)
    assert isinstance(prediction, OperatorPrediction)
    assert prediction.field.shape == outputs_test.shape  # (n_dof_out, n_test) -- geometry-varying shape

    # a rank-matched linear regression against an exactly rank-3 map, fit from 60 noiseless pairs, must
    # recover it to near machine precision on fresh same-distribution queries -- not just fit training data.
    relative_error = np.linalg.norm(prediction.field - outputs_test) / np.linalg.norm(outputs_test)
    assert relative_error < 1e-6


def test_predict_single_query_returns_1d_field(exact_low_rank_problem):
    inputs_train, outputs_train, inputs_test, _ = exact_low_rank_problem
    surrogate = fit_linear_operator_surrogate(
        inputs_train, outputs_train, rank_in=_TRUE_RANK, rank_out=_TRUE_RANK, ridge=1e-10
    )
    prediction = surrogate.predict(inputs_test[:, 0])
    assert prediction.field.shape == (_N_DOF_OUT,)
    assert prediction.in_domain.shape == (1,)
    assert prediction.input_relative_residual.shape == (1,)


def test_calibration_report_on_exact_recovery_is_informative(exact_low_rank_problem):
    inputs_train, outputs_train, inputs_test, outputs_test = exact_low_rank_problem
    surrogate = fit_linear_operator_surrogate(
        inputs_train, outputs_train, rank_in=_TRUE_RANK, rank_out=_TRUE_RANK, ridge=1e-10
    )
    report = calibrate_linear_operator_surrogate(surrogate, inputs_test, outputs_test)

    assert isinstance(report, LinearOperatorCalibrationReport)
    assert report.n == inputs_test.shape[1]
    # essentially exact recovery: the calibrated bound is far tighter than the trivial mean-output baseline.
    assert report.qhat_relative_l2_error < 1e-4
    assert report.baseline_relative_l2_error > 0.1
    assert report.imprecise is False


def test_far_out_of_distribution_query_is_flagged(exact_low_rank_problem):
    inputs_train, outputs_train, inputs_test, _ = exact_low_rank_problem
    surrogate = fit_linear_operator_surrogate(
        inputs_train, outputs_train, rank_in=_TRUE_RANK, rank_out=_TRUE_RANK, ridge=1e-10
    )
    far_query = inputs_test[:, 0] * 50.0  # far outside the training envelope in reduced coordinates
    prediction = surrogate.predict(far_query)
    assert not bool(prediction.in_domain[0])
    # predict() never refuses to answer -- a field is still returned alongside the honesty flag.
    assert prediction.field.shape == (_N_DOF_OUT,)


# --- group 2: shape/argument validation ----------------------------------------------------------------


def test_fit_rejects_mismatched_snapshot_columns():
    rng = np.random.default_rng(1)
    inputs = rng.standard_normal((5, 10))
    outputs = rng.standard_normal((6, 9))
    with pytest.raises(ValueError, match="same number of paired"):
        fit_linear_operator_surrogate(inputs, outputs, rank_in=2, rank_out=2)


def test_fit_rejects_non_positive_ridge():
    rng = np.random.default_rng(1)
    inputs = rng.standard_normal((5, 10))
    outputs = rng.standard_normal((6, 10))
    with pytest.raises(ValueError, match="ridge"):
        fit_linear_operator_surrogate(inputs, outputs, rank_in=2, rank_out=2, ridge=0.0)
    with pytest.raises(ValueError, match="ridge"):
        fit_linear_operator_surrogate(inputs, outputs, rank_in=2, rank_out=2, ridge=-1.0)


def test_fit_rejects_negative_margins():
    rng = np.random.default_rng(1)
    inputs = rng.standard_normal((5, 10))
    outputs = rng.standard_normal((6, 10))
    with pytest.raises(ValueError, match="ood_margin"):
        fit_linear_operator_surrogate(inputs, outputs, rank_in=2, rank_out=2, ood_margin=-0.1)
    with pytest.raises(ValueError, match="input_residual_margin"):
        fit_linear_operator_surrogate(inputs, outputs, rank_in=2, rank_out=2, input_residual_margin=-0.1)


def test_fit_requires_at_least_two_snapshots():
    rng = np.random.default_rng(1)
    inputs = rng.standard_normal((5, 1))
    outputs = rng.standard_normal((6, 1))
    with pytest.raises(ValueError, match="at least two"):
        fit_linear_operator_surrogate(inputs, outputs, rank_in=1, rank_out=1)


def test_predict_rejects_wrong_input_dimension(exact_low_rank_problem):
    inputs_train, outputs_train, _, _ = exact_low_rank_problem
    surrogate = fit_linear_operator_surrogate(inputs_train, outputs_train, rank_in=2, rank_out=2, ridge=1e-6)
    with pytest.raises(ValueError, match="basis_in.n_dof"):
        surrogate.predict(np.zeros(_N_DOF_IN + 1))


def test_calibrate_rejects_mismatched_holdout_columns(exact_low_rank_problem):
    inputs_train, outputs_train, inputs_test, outputs_test = exact_low_rank_problem
    surrogate = fit_linear_operator_surrogate(inputs_train, outputs_train, rank_in=2, rank_out=2, ridge=1e-6)
    with pytest.raises(ValueError, match="same number of paired"):
        calibrate_linear_operator_surrogate(surrogate, inputs_test, outputs_test[:, :-1])


def test_calibrate_rejects_alpha_out_of_range(exact_low_rank_problem):
    inputs_train, outputs_train, inputs_test, outputs_test = exact_low_rank_problem
    surrogate = fit_linear_operator_surrogate(inputs_train, outputs_train, rank_in=2, rank_out=2, ridge=1e-6)
    with pytest.raises(ValueError, match="alpha"):
        calibrate_linear_operator_surrogate(surrogate, inputs_test, outputs_test, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        calibrate_linear_operator_surrogate(surrogate, inputs_test, outputs_test, alpha=1.0)


# --- group 3: realistic kernel (AdvectionDiffusionOperator), genuine imperfect generalization -----------

_GRID_N = 32
_LENGTH = 1.0
_VELOCITY = 0.3
_DT = 0.01
_STEPS_FORWARD = 5
_N_FIELDS_PER_DIFFUSIVITY = 6
_TRAIN_DIFFUSIVITIES = np.linspace(0.02, 0.10, 6)
_HELDOUT_DIFFUSIVITIES = _TRAIN_DIFFUSIVITIES[:-1] + np.diff(_TRAIN_DIFFUSIVITIES) / 2.0  # interleaved, disjoint


def _varied_initial_fields(rng: np.random.Generator, n_fields: int) -> np.ndarray:
    """A batch of differently-shaped bump-plus-sinusoid fields (random center/width/amplitude/phase) --
    varied *input* fields, not one fixed field swept only through a hidden parameter."""
    x = np.linspace(0.0, _LENGTH, _GRID_N)
    fields = []
    for _ in range(n_fields):
        center = rng.uniform(0.1, 0.9) * _LENGTH
        width = rng.uniform(0.03, 0.08) * _LENGTH
        amplitude = rng.uniform(0.5, 1.5)
        phase = rng.uniform(0.0, 2 * np.pi)
        field = amplitude * np.exp(-((x - center) ** 2) / (2.0 * width**2)) + 0.2 * np.sin(
            2.0 * np.pi * x + phase
        )
        fields.append(field)
    return np.stack(fields, axis=1)


def _advection_diffusion_pairs(diffusivities, rng: np.random.Generator, n_fields: int = _N_FIELDS_PER_DIFFUSIVITY):
    """Paired (state at t, state _STEPS_FORWARD steps later) snapshots, pooled across every diffusivity in
    ``diffusivities`` -- the TRUE input->output map genuinely differs per diffusivity, so a single fitted
    linear map is only ever an imperfect compromise (unlike the exact-recovery fixture above)."""
    ins, outs = [], []
    for diffusivity in diffusivities:
        operator = AdvectionDiffusionOperator(
            diffusivity=float(diffusivity),
            velocity=_VELOCITY,
            n=_GRID_N,
            length=_LENGTH,
            bc="periodic",
            scheme="implicit",
        )
        step_operator = np.linalg.matrix_power(operator.transition_matrix(_DT), _STEPS_FORWARD)
        fields = _varied_initial_fields(rng, n_fields)
        for j in range(fields.shape[1]):
            u0 = fields[:, j]
            ins.append(u0)
            outs.append(step_operator @ u0)
    return np.stack(ins, axis=1), np.stack(outs, axis=1)


@pytest.fixture(scope="module")
def advection_diffusion_split():
    rng = np.random.default_rng(0)
    train_in, train_out = _advection_diffusion_pairs(_TRAIN_DIFFUSIVITIES, rng)
    heldout_in, heldout_out = _advection_diffusion_pairs(_HELDOUT_DIFFUSIVITIES, rng)
    return train_in, train_out, heldout_in, heldout_out


def test_train_and_heldout_diffusivities_are_disjoint():
    assert set(_TRAIN_DIFFUSIVITIES.tolist()).isdisjoint(set(_HELDOUT_DIFFUSIVITIES.tolist()))


def test_more_rank_measurably_helps_and_flips_the_precision_floor(advection_diffusion_split):
    train_in, train_out, heldout_in, heldout_out = advection_diffusion_split

    low_rank_surrogate = fit_linear_operator_surrogate(train_in, train_out, rank_in=1, rank_out=1, ridge=1e-6)
    low_report = calibrate_linear_operator_surrogate(low_rank_surrogate, heldout_in, heldout_out)

    mid_rank_surrogate = fit_linear_operator_surrogate(train_in, train_out, rank_in=6, rank_out=6, ridge=1e-6)
    mid_report = calibrate_linear_operator_surrogate(mid_rank_surrogate, heldout_in, heldout_out)

    # rank=1 carries essentially no more information than guessing the training-mean output field.
    assert low_report.imprecise is True
    # rank=6 is a measurably, substantially better fit on the SAME held-out set...
    assert mid_report.mean_relative_l2_error < 0.5 * low_report.mean_relative_l2_error
    # ...and clears the precision floor the rank=1 fit failed.
    assert mid_report.imprecise is False
    assert mid_report.qhat_relative_l2_error < mid_report.baseline_relative_l2_error


def test_far_out_of_distribution_input_field_is_flagged(advection_diffusion_split):
    train_in, train_out, _, _ = advection_diffusion_split
    surrogate = fit_linear_operator_surrogate(train_in, train_out, rank_in=6, rank_out=6, ridge=1e-6)

    # an input field far outside the training amplitude/shape envelope -- the honest thing this gate can
    # detect (it has no way to observe a hidden simulation parameter like diffusivity that never shows up
    # in the input field itself; see the module docstring's "Honesty gate" section).
    far_field = train_in[:, 0] * 40.0 + 5.0
    prediction = surrogate.predict(far_field)
    assert not bool(prediction.in_domain[0])


def test_in_distribution_input_fields_stay_in_domain(advection_diffusion_split):
    train_in, train_out, _, _ = advection_diffusion_split
    surrogate = fit_linear_operator_surrogate(train_in, train_out, rank_in=6, rank_out=6, ridge=1e-6)

    rng = np.random.default_rng(0)
    fresh_in, _ = _advection_diffusion_pairs(_TRAIN_DIFFUSIVITIES[:2], rng, n_fields=3)
    prediction = surrogate.predict(fresh_in)
    assert prediction.in_domain.all()
