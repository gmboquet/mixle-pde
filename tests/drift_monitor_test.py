"""Tests for mixle_pde.drift_monitor: the generic streaming drift monitor (MP-N6 remainder).

Four groups:

1. Core :class:`DriftMonitor` mechanics on a hand-crafted, exactly-hand-verifiable error sequence
   (no surrogate involved) -- pins down the rolling-window / persistence / threshold arithmetic
   precisely, plus :meth:`~mixle_pde.drift_monitor.DriftMonitor.reset` and argument validation.
2. :class:`~mixle_pde.drift_monitor.DriftAlert` construction validation.
3. The three ``*CalibrationView`` adapters, each built from a real, already-fitted instance of the
   corresponding surrogate/emulator family (:mod:`mixle_pde.surrogate`, :mod:`mixle_pde.qoi_emulator`,
   :mod:`mixle_pde.operator_surrogate`) -- confirms each reuses that family's own calibration bound
   verbatim and satisfies the :class:`~mixle_pde.drift_monitor.Predictor` protocol structurally.
4. The flagship synthetic-stream scenario: a real :class:`~mixle_pde.qoi_emulator.GaussianProcessEmulator`
   is fit and calibrated once, then monitored over a stream that starts well-calibrated (same function
   the emulator was fit on) and genuinely drifts partway through (the true function changes, the
   emulator's own predictions are never updated) -- confirms zero false positives before drift onset and
   detection within a bounded window after onset, not immediately and not never.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mixle_pde.drift_monitor import (
    DriftAlert,
    DriftMonitor,
    OperatorSurrogateCalibrationView,
    Predictor,
    QoIEmulatorCalibrationView,
    SurrogateCalibrationView,
)
from mixle_pde.operator_surrogate import calibrate_linear_operator_surrogate, fit_linear_operator_surrogate
from mixle_pde.qoi_emulator import calibrate, fit_gaussian_process
from mixle_pde.surrogate import Surrogate

# --- group 1: core DriftMonitor mechanics, hand-verifiable arithmetic -----------------------------------


def test_alert_fires_exactly_when_persistence_is_met():
    # window=4, persistence=2, threshold=1.0, calibration_bound=1.0; prediction pinned at 0.0 so the
    # per-step residual error equals true_value exactly (see module docstring's shape-agnostic L2 rule).
    monitor = DriftMonitor(calibration_bound=1.0, window=4, persistence=2, threshold=1.0)

    stable_errors = [0.1, 0.1, 0.1, 0.1]
    for i, err in enumerate(stable_errors, start=1):
        alert = monitor.update(f"x{i}", 0.0, err)
        assert alert is None
    assert monitor.alerts == []

    # deque now [0.1, 0.1, 0.1, 5.0]; mean=1.325, ratio=1.325 > 1.0 -> consecutive=1 (< persistence=2)
    alert5 = monitor.update("x5", 0.0, 5.0)
    assert alert5 is None
    assert monitor.alerts == []

    # deque now [0.1, 0.1, 5.0, 5.0]; mean=2.55, ratio=2.55 > 1.0 -> consecutive=2 == persistence -> fires
    alert6 = monitor.update("x6", 0.0, 5.0)
    assert isinstance(alert6, DriftAlert)
    assert alert6.step == 6
    assert alert6.input == "x6"
    assert alert6.window_size == 4
    assert math.isclose(alert6.rolling_error, 2.55, rel_tol=1e-9)
    assert math.isclose(alert6.exceedance_ratio, 2.55, rel_tol=1e-9)
    assert alert6.consecutive_exceedances == 2
    assert monitor.alerts == [alert6]
    assert monitor.n_observed == 6

    # drift persists -> the monitor keeps alerting on every subsequent qualifying update, not just once.
    alert7 = monitor.update("x7", 0.0, 5.0)
    assert isinstance(alert7, DriftAlert)
    assert alert7.consecutive_exceedances == 3
    assert len(monitor.alerts) == 2


def test_single_moderate_outlier_diluted_by_the_window_does_not_trigger_a_false_alarm():
    # A single point at ~2x the calibration bound, surrounded by well-calibrated neighbors, is diluted
    # by the window average below the ratio*threshold trip point (mean=(0.1*3+2.0)/4=0.575 < 1.0) -- it
    # never even crosses once, let alone for `persistence` updates in a row. (A single point severe
    # enough to push the *window mean itself* over threshold is different: it legitimately perturbs a
    # mean-based statistic for multiple consecutive window-slides by construction, exactly like any
    # rolling average -- that is expected behavior, not a false positive, and is deliberately not what
    # this test exercises. The flagship end-to-end test below is the real "no false positives under
    # ordinary calibrated noise" evidence, using a genuine fitted emulator's own held-out residuals.)
    monitor = DriftMonitor(calibration_bound=1.0, window=4, persistence=2, threshold=1.0)
    sequence = [0.1, 0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1]  # single moderate spike, then back to normal
    alerts = [monitor.update(i, 0.0, err) for i, err in enumerate(sequence)]
    assert all(a is None for a in alerts)
    assert monitor.alerts == []


def test_reset_clears_state_but_keeps_configuration():
    monitor = DriftMonitor(calibration_bound=1.0, window=4, persistence=2, threshold=1.0)
    for err in (0.1, 0.1, 0.1, 0.1, 5.0, 5.0):
        monitor.update("x", 0.0, err)
    assert len(monitor.alerts) == 1
    assert monitor.n_observed == 6

    monitor.reset()
    assert monitor.alerts == []
    assert monitor.n_observed == 0

    # replaying the stable-only prefix now raises no alert, proving the rolling window was really cleared.
    for err in (0.1, 0.1, 0.1, 0.1):
        assert monitor.update("x", 0.0, err) is None
    assert monitor.alerts == []
    assert monitor.calibration_bound == 1.0  # configuration itself is untouched by reset()


def test_run_consumes_a_full_stream_and_returns_every_alert():
    monitor = DriftMonitor(calibration_bound=1.0, window=4, persistence=2, threshold=1.0)
    stream = [(i, 0.0, err) for i, err in enumerate([0.1, 0.1, 0.1, 0.1, 5.0, 5.0, 5.0])]
    alerts = monitor.run(stream)
    assert len(alerts) == 2  # fires at step 6 and step 7, per test_alert_fires_exactly_when_persistence_is_met
    assert alerts == monitor.alerts


def test_relative_mode_normalizes_by_true_value_norm():
    monitor = DriftMonitor(calibration_bound=0.5, window=2, persistence=1, threshold=1.0, relative=True)
    # |4 - 5| / |5| = 0.2 each -- window not yet full, then rolling mean 0.2 well under the 0.5 bound.
    assert monitor.update("x", 4.0, 5.0) is None
    assert monitor.update("x", 4.0, 5.0) is None
    # |0 - 5| / |5| = 1.0 -- deque becomes [0.2, 1.0], rolling mean 0.6, ratio 0.6/0.5=1.2 > threshold,
    # persistence=1 fires on this first qualifying window.
    alert = monitor.update("x", 0.0, 5.0)
    assert alert is not None
    assert math.isclose(alert.rolling_error, (0.2 + 1.0) / 2.0)
    assert math.isclose(alert.exceedance_ratio, 1.2)


def test_update_rejects_mismatched_prediction_and_truth_shapes():
    monitor = DriftMonitor(calibration_bound=1.0)
    with pytest.raises(ValueError, match="shape"):
        monitor.update("x", [1.0, 2.0], [1.0, 2.0, 3.0])


def test_update_rejects_non_finite_values():
    monitor = DriftMonitor(calibration_bound=1.0)
    with pytest.raises(ValueError, match="finite"):
        monitor.update("x", float("nan"), 1.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"calibration_bound": 0.0}, "calibration_bound"),
        ({"calibration_bound": -1.0}, "calibration_bound"),
        ({"calibration_bound": float("nan")}, "calibration_bound"),
        ({"calibration_bound": 1.0, "window": 1}, "window"),
        ({"calibration_bound": 1.0, "threshold": 0.0}, "threshold"),
        ({"calibration_bound": 1.0, "persistence": 0}, "persistence"),
    ],
)
def test_construction_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DriftMonitor(**kwargs)


# --- group 2: DriftAlert direct-construction validation -------------------------------------------------


def test_drift_alert_accepts_a_valid_record():
    alert = DriftAlert(
        step=1,
        input="x",
        window_size=5,
        rolling_error=0.5,
        calibration_bound=0.2,
        exceedance_ratio=2.5,
        consecutive_exceedances=3,
    )
    assert alert.step == 1
    assert alert.consecutive_exceedances == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("step", 0),
        ("window_size", 0),
        ("rolling_error", -0.1),
        ("calibration_bound", 0.0),
        ("exceedance_ratio", 0.0),
        ("consecutive_exceedances", 0),
    ],
)
def test_drift_alert_rejects_invalid_fields(field, value):
    base = dict(
        step=1,
        input="x",
        window_size=5,
        rolling_error=0.5,
        calibration_bound=0.2,
        exceedance_ratio=2.5,
        consecutive_exceedances=3,
    )
    base[field] = value
    with pytest.raises(ValueError):
        DriftAlert(**base)


# --- group 3: the three *CalibrationView adapters, each against a real fitted instance -------------------


def test_surrogate_calibration_view_reuses_qhat_verbatim():
    # constructed directly via keyword args (the E6 distillation fit itself is a separate, already-shipped
    # concern this task does not re-exercise) -- see mixle_pde.surrogate.Surrogate's own module docstring.
    surrogate = Surrogate(
        nets=[],
        featurizer=None,
        gate=None,
        teacher=lambda x: 0.0,
        qhat=np.array([0.05, 0.08]),
        tol=np.array([0.2, 0.2]),
        y_mean=np.array([0.0, 0.0]),
        y_scale=np.array([1.0, 1.0]),
        holdout_mae=np.array([0.03, 0.03]),
        scalar_output=False,
        alpha=0.1,
        seed=0,
        budget=32,
    )
    view = SurrogateCalibrationView(surrogate=surrogate)
    assert isinstance(view, Predictor)
    assert view.calibration_bound() == pytest.approx(0.08)  # max over the per-output qhat array


def test_surrogate_calibration_view_rejects_non_positive_qhat():
    surrogate = Surrogate(
        nets=[],
        featurizer=None,
        gate=None,
        teacher=lambda x: 0.0,
        qhat=np.array([0.0, 0.0]),
        tol=np.array([0.2, 0.2]),
        y_mean=np.array([0.0, 0.0]),
        y_scale=np.array([1.0, 1.0]),
        holdout_mae=np.array([0.03, 0.03]),
    )
    with pytest.raises(ValueError, match="qhat"):
        SurrogateCalibrationView(surrogate=surrogate).calibration_bound()


def test_qoi_emulator_calibration_view_reuses_rmse_verbatim():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(40, 2))
    y = np.sin(3 * x[:, 0]) + 0.1 * x[:, 1]
    gp = fit_gaussian_process(x, y)
    xh = rng.uniform(-1, 1, size=(20, 2))
    yh = np.sin(3 * xh[:, 0]) + 0.1 * xh[:, 1]
    report = calibrate(gp, xh, yh)

    view = QoIEmulatorCalibrationView(report=report)
    assert isinstance(view, Predictor)
    assert view.calibration_bound() == pytest.approx(report.rmse)


def test_operator_surrogate_calibration_view_reuses_qhat_relative_l2_error_verbatim():
    rng = np.random.default_rng(0)
    n_in, n_out, n_train = 10, 12, 30
    basis_in, _ = np.linalg.qr(rng.standard_normal((n_in, 3)))
    basis_out, _ = np.linalg.qr(rng.standard_normal((n_out, 3)))
    core = rng.standard_normal((3, 3))

    def true_operator(u_column: np.ndarray) -> np.ndarray:
        return basis_out @ (core @ (basis_in.T @ u_column))

    latent = rng.standard_normal((3, n_train))
    inputs = basis_in @ latent
    outputs = np.stack([true_operator(inputs[:, i]) for i in range(n_train)], axis=1)
    surrogate = fit_linear_operator_surrogate(inputs, outputs, rank_in=3, rank_out=3, ridge=1e-8)

    latent_h = rng.standard_normal((3, 10))
    inputs_h = basis_in @ latent_h
    outputs_h = np.stack([true_operator(inputs_h[:, i]) for i in range(10)], axis=1)
    report = calibrate_linear_operator_surrogate(surrogate, inputs_h, outputs_h)

    view = OperatorSurrogateCalibrationView(report=report)
    assert isinstance(view, Predictor)
    assert view.calibration_bound() == pytest.approx(report.qhat_relative_l2_error)

    # a short relative-error stream against near-exact recovery: essentially zero error throughout, so
    # the monitor built from this real surrogate's own bound must never alert.
    monitor = DriftMonitor.from_predictor(view, window=5, persistence=2, relative=True)
    for i in range(10):
        prediction = surrogate.predict(inputs_h[:, i]).field
        assert monitor.update(inputs_h[:, i], prediction, outputs_h[:, i]) is None
    assert monitor.alerts == []


# --- group 4: flagship scenario -- well-calibrated stream, then a genuine mid-stream function change -----

_NOISE = 0.05


def _true_function_before(x: np.ndarray) -> np.ndarray:
    return np.sin(3.0 * x[:, 0]) + 0.2 * x[:, 1]


def _true_function_after(x: np.ndarray) -> np.ndarray:
    # a genuine, systematic change to the underlying function being modeled -- not just noisier
    # observations of the same function -- while the emulator itself is never refit or retrained.
    return _true_function_before(x) + 0.15


def test_drift_monitor_detects_a_genuine_mid_stream_function_change_within_a_bounded_window():
    rng = np.random.default_rng(0)

    x_train = rng.uniform(-1, 1, size=(80, 2))
    y_train = _true_function_before(x_train) + rng.normal(0.0, _NOISE, size=80)
    emulator = fit_gaussian_process(x_train, y_train)

    x_cal = rng.uniform(-1, 1, size=(100, 2))
    y_cal = _true_function_before(x_cal) + rng.normal(0.0, _NOISE, size=100)
    report = calibrate(emulator, x_cal, y_cal)

    view = QoIEmulatorCalibrationView(report=report)
    monitor = DriftMonitor.from_predictor(view, window=20, persistence=3, threshold=1.2)

    n_stable = 150
    n_drift = 150
    drift_onset_step = n_stable + 1  # update() steps are 1-indexed; this is the first drifted observation

    alerts_before_onset = []
    first_alert_after_onset = None
    for i in range(n_stable + n_drift):
        x_row = rng.uniform(-1, 1, size=(1, 2))
        true_fn = _true_function_before if i < n_stable else _true_function_after
        y_true = float(true_fn(x_row)[0] + rng.normal(0.0, _NOISE))
        y_pred = float(emulator.predict(x_row).mean[0])

        alert = monitor.update(x_row[0], y_pred, y_true)
        if alert is None:
            continue
        if alert.step < drift_onset_step:
            alerts_before_onset.append(alert)
        elif first_alert_after_onset is None:
            first_alert_after_onset = alert

    # avoid false positives: nothing fires while the stream is still well-calibrated.
    assert alerts_before_onset == []

    # avoid missing it: drift is genuinely caught somewhere in the drift phase...
    assert first_alert_after_onset is not None
    delay = first_alert_after_onset.step - drift_onset_step

    # ...and not immediately at the very first drifted observation (persistence + a rolling window need
    # to actually accumulate evidence)...
    assert delay >= 2
    # ...nor anywhere close to "never" -- comfortably inside the 150-observation drift phase, well short
    # of it, i.e. detection is a genuine early-warning signal, not a coincidence at the very end.
    assert delay <= 75

    assert first_alert_after_onset.calibration_bound == pytest.approx(report.rmse)
    assert first_alert_after_onset.exceedance_ratio > 1.2
