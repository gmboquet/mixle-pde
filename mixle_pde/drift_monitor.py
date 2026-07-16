"""Streaming drift monitor for fitted surrogates/emulators, reusing their own calibration bound (MP-N6).

Source: `notes/mixle-pde-ai-native-multiphysics-work-plan.md` workstream N, **MP-N6** ("Certification,
calibration, validity, error budgets (surrogate-specific)"). The reconciliation ledger
(`docs/reconciliation/mp-task-ledger.md`) records MP-N6 as `owned-elsewhere (partial)`: "`surrogate.py`'s
conformal calibration (`qhat` precision floor) and OOD `defer()` gate cover part of this; no independent
third-party validation, drift monitoring, or stability/conservation test suite for surrogates found." This
module closes exactly the "drift monitoring" gap named there -- it does not attempt independent
third-party validation or a stability/conservation test suite, which remain separate, unclaimed work.

Three surrogate/emulator families already ship in this package, each with its own honest calibration
step, but none sharing a common predictive return type or a common field name for "the error this fit
is calibrated to stay within":

- :mod:`mixle_pde.surrogate` (E6/MP-N1/MP-N4): :class:`~mixle_pde.surrogate.Surrogate` stores a
  split-conformal interval half-width directly as ``qhat`` (an ``(k,)`` array, one per output
  dimension) alongside a precision floor ``tol``.
- :mod:`mixle_pde.qoi_emulator` (MP-N4): :class:`~mixle_pde.qoi_emulator.GaussianProcessEmulator` and
  :class:`~mixle_pde.qoi_emulator.BayesianPolynomialEmulator` report a genuine per-query posterior
  predictive ``std`` from :meth:`~mixle_pde.qoi_emulator.QoIEmulator.predict`, but the *held-out*
  calibration number -- :func:`~mixle_pde.qoi_emulator.calibrate`'s ``CalibrationReport.rmse`` -- lives
  on a separately returned report, not the emulator itself.
- :mod:`mixle_pde.operator_surrogate` (MP-N5): :class:`~mixle_pde.operator_surrogate.LinearOperatorSurrogate`
  likewise keeps its split-conformal bound (``LinearOperatorCalibrationReport.qhat_relative_l2_error``,
  a *relative* L2 error) on a report returned by
  :func:`~mixle_pde.operator_surrogate.calibrate_linear_operator_surrogate`, not on the surrogate.

Rather than force a new shared predictive interface onto three already-shipped modules this task does not
own (and must not modify -- see each module's own docstring for why its scope is deliberately frozen),
this module defines the one minimal :class:`Predictor` protocol :class:`DriftMonitor` actually needs --
a single positive scalar, ``calibration_bound()`` -- and three small ``*CalibrationView`` adapters, one
per existing family, each reading that number straight off the real, already-fitted/calibrated object.
None of the three adapters fits, calibrates, or approximates anything; they are pure field reads.

:class:`DriftMonitor` itself is generic over all three: feed it a stream of ``(input, prediction,
true_value)`` triples -- the same shape every family already produces once a caller has called
``predict()`` and later observed the truth -- and it maintains a rolling mean residual error, comparing
it against ``calibration_bound`` once enough of the stream has accumulated. A :class:`DriftAlert` fires
only after that comparison holds for several consecutive updates in a row, so a single noisy observation
cannot trigger a false alarm, while genuine, sustained drift is always caught within a bounded number of
observations of its onset (see `tests/drift_monitor_test.py` for a synthetic stream that starts
well-calibrated and then genuinely drifts partway through).

Scope note (baseline only, matching the "one clean increment" precedent set by this workstream's
siblings): a single rolling-mean-vs-bound statistic with a consecutive-exceedance persistence rule (no
CUSUM/Page-Hinkley/ADWIN change-point machinery, no multivariate drift localization, no automatic
retraining/rollback action). This module only detects and reports; MP-N8 ("registry, promotion,
monitoring, retraining, rollback") remains separate, unclaimed work.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle_pde.operator_surrogate import LinearOperatorCalibrationReport
from mixle_pde.qoi_emulator import CalibrationReport
from mixle_pde.surrogate import Surrogate

__all__ = [
    "Predictor",
    "DriftAlert",
    "DriftMonitor",
    "SurrogateCalibrationView",
    "QoIEmulatorCalibrationView",
    "OperatorSurrogateCalibrationView",
]


@runtime_checkable
class Predictor(Protocol):
    """The minimal contract :class:`DriftMonitor` needs from any surrogate/emulator family.

    Not a general predictive interface (this package's three existing surrogate families return three
    different, incompatible shapes from ``predict()`` -- a bare float/array, an ``EmulatorPrediction``,
    an ``OperatorPrediction`` -- and unifying that is out of this task's scope and not needed for drift
    monitoring). This protocol asks only for the one number a generic monitor actually needs: a positive
    scalar ``calibration_bound()``, the same error magnitude each family's own calibration step already
    established as "how wrong this fit is allowed to be." See the ``*CalibrationView`` classes below for
    the three adapters that satisfy this protocol for this package's existing families.
    """

    def calibration_bound(self) -> float:
        """A positive error magnitude such that a fresh, exchangeable prediction error was not
        expected (at whatever confidence level the family's own calibration step already fixed) to
        exceed it -- reused verbatim from the relevant surrogate/emulator, never recomputed here."""
        ...


def _positive_finite(value: float, *, name: str) -> float:
    value = float(value)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return value


@dataclass(frozen=True)
class SurrogateCalibrationView:
    """Adapts a fitted :class:`mixle_pde.surrogate.Surrogate` (E6) to :class:`Predictor`.

    Reuses ``surrogate.qhat`` -- the split-conformal interval half-width ``distill_forward``'s own
    calibration split already computed -- taking the largest per-output-dimension entry so a
    multi-output surrogate is monitored against its loosest calibrated guarantee. No new fitting or
    statistics.
    """

    surrogate: Surrogate

    def calibration_bound(self) -> float:
        bound = float(np.max(np.asarray(self.surrogate.qhat, dtype=np.float64)))
        return _positive_finite(bound, name="Surrogate.qhat")


@dataclass(frozen=True)
class QoIEmulatorCalibrationView:
    """Adapts a :class:`mixle_pde.qoi_emulator.CalibrationReport` to :class:`Predictor`.

    Build the report first via :func:`mixle_pde.qoi_emulator.calibrate` (shared by
    :class:`~mixle_pde.qoi_emulator.GaussianProcessEmulator` and
    :class:`~mixle_pde.qoi_emulator.BayesianPolynomialEmulator` alike), then wrap it here. Reuses
    ``report.rmse`` -- the family's own held-out measurement of how wrong ``predict()`` actually was on
    data it was not fit from. No new fitting or statistics.
    """

    report: CalibrationReport

    def calibration_bound(self) -> float:
        return _positive_finite(self.report.rmse, name="CalibrationReport.rmse")


@dataclass(frozen=True)
class OperatorSurrogateCalibrationView:
    """Adapts a :class:`mixle_pde.operator_surrogate.LinearOperatorCalibrationReport` to
    :class:`Predictor`.

    Build the report first via :func:`mixle_pde.operator_surrogate.calibrate_linear_operator_surrogate`,
    then wrap it here. Reuses ``report.qhat_relative_l2_error`` -- the family's own split-conformal
    relative-L2-error quantile. This bound is *relative* (a fraction of ``||true_value||``, not an
    absolute magnitude); construct :class:`DriftMonitor` with ``relative=True`` when using this view.
    No new fitting or statistics.
    """

    report: LinearOperatorCalibrationReport

    def calibration_bound(self) -> float:
        return _positive_finite(
            self.report.qhat_relative_l2_error, name="LinearOperatorCalibrationReport.qhat_relative_l2_error"
        )


def _residual_error(prediction: Any, true_value: Any, *, relative: bool) -> float:
    """L2 residual magnitude between one ``prediction`` and one ``true_value`` -- scalar or vector/field
    alike, matching the shape-agnostic convention this package's own calibration routines already use
    (:func:`mixle_pde.qoi_emulator.calibrate`, :func:`mixle_pde.operator_surrogate
    .calibrate_linear_operator_surrogate`). ``relative=True`` normalizes by ``||true_value||``, required
    when the calibration bound being compared against is itself a relative error (see
    :class:`OperatorSurrogateCalibrationView`).
    """
    p = np.atleast_1d(np.asarray(prediction, dtype=np.float64))
    t = np.atleast_1d(np.asarray(true_value, dtype=np.float64))
    if p.shape != t.shape:
        raise ValueError(f"prediction shape {p.shape} does not match true_value shape {t.shape}")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t))):
        raise ValueError("prediction and true_value must both be finite")
    residual = float(np.linalg.norm(p - t))
    if not relative:
        return residual
    true_norm = float(np.linalg.norm(t))
    return residual / true_norm if true_norm > 1e-12 else residual


@dataclass(frozen=True)
class DriftAlert:
    """Raised by :meth:`DriftMonitor.update` when the rolling calibration-error statistic has exceeded
    the surrogate's own stated calibration bound for ``consecutive_exceedances`` observations in a row --
    long enough to be very unlikely from one noisy point, without waiting so long that a real drift
    episode goes unflagged for an unreasonable stretch of the stream.
    """

    step: int
    input: Any
    window_size: int
    rolling_error: float
    calibration_bound: float
    exceedance_ratio: float
    consecutive_exceedances: int

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError(f"DriftAlert.step must be positive; got {self.step!r}")
        if self.window_size <= 0:
            raise ValueError(f"DriftAlert.window_size must be positive; got {self.window_size!r}")
        if not (np.isfinite(self.rolling_error) and self.rolling_error >= 0.0):
            raise ValueError(f"DriftAlert.rolling_error must be finite and non-negative; got {self.rolling_error!r}")
        _positive_finite(self.calibration_bound, name="DriftAlert.calibration_bound")
        _positive_finite(self.exceedance_ratio, name="DriftAlert.exceedance_ratio")
        if self.consecutive_exceedances <= 0:
            raise ValueError(
                f"DriftAlert.consecutive_exceedances must be positive; got {self.consecutive_exceedances!r}"
            )


@dataclass
class DriftMonitor:
    """Generic streaming drift monitor for a fitted surrogate/emulator's live prediction error.

    Feed one ``(input, prediction, true_value)`` triple at a time via :meth:`update` (or a whole stream
    via :meth:`run`); ``prediction``/``true_value`` may be scalars or vectors/fields. Once ``window``
    triples have accumulated, every subsequent call recomputes the rolling mean residual error over the
    last ``window`` triples and compares it against ``calibration_bound * threshold``. A
    :class:`DriftAlert` fires only once that comparison has held for ``persistence`` consecutive updates
    in a row: a single noisy triple cannot trigger a false alarm, while sustained drift whose steady-state
    rolling error exceeds the bound is caught within roughly ``window + persistence`` observations of its
    onset at the latest (often sooner, once enough drifted points have entered the window to move the
    mean past the threshold) -- never indefinitely. Drift too subtle to move the fully-saturated window's
    mean past ``calibration_bound * threshold`` is, by construction, drift the calibration step's own
    error bound already tolerates, and is not expected to be flagged.

    ``threshold=1.0`` (the default) compares the rolling *mean* error against the calibration bound
    directly. For :class:`QoIEmulatorCalibrationView` that bound is itself a held-out RMSE -- the same
    kind of aggregate error statistic -- so the comparison is direct. For :class:`SurrogateCalibrationView`
    and :class:`OperatorSurrogateCalibrationView` the bound is a split-conformal *quantile* (an
    upper-tail guarantee, not a mean), so a stable, non-drifting stream's rolling mean sits comfortably
    below it by construction (root-mean-square/quantile bounds dominate the mean for the same
    distribution) -- callers who want a more sensitive alarm against a quantile-style bound can lower
    ``threshold`` below 1.0.

    ``calibration_bound`` must be read from an existing, already-fitted surrogate/emulator (typically via
    :meth:`from_predictor` and one of the ``*CalibrationView`` adapters) -- this module never fits or
    calibrates anything itself.
    """

    calibration_bound: float
    window: int = 20
    threshold: float = 1.0
    persistence: int = 3
    relative: bool = False
    alerts: list[DriftAlert] = field(default_factory=list, init=False)
    _errors: deque = field(default_factory=deque, init=False, repr=False)
    _consecutive_exceedances: int = field(default=0, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        _positive_finite(self.calibration_bound, name="calibration_bound")
        if self.window < 2:
            raise ValueError(f"window must be at least 2; got {self.window!r}")
        if self.threshold <= 0.0:
            raise ValueError(f"threshold must be positive; got {self.threshold!r}")
        if self.persistence < 1:
            raise ValueError(f"persistence must be at least 1; got {self.persistence!r}")
        self._errors = deque(maxlen=self.window)

    @classmethod
    def from_predictor(
        cls,
        predictor: Predictor,
        *,
        window: int = 20,
        threshold: float = 1.0,
        persistence: int = 3,
        relative: bool = False,
    ) -> DriftMonitor:
        """Build a :class:`DriftMonitor` whose ``calibration_bound`` is read directly off
        ``predictor.calibration_bound()`` -- the standard way to start monitoring any of this package's
        existing surrogate/emulator families through the matching ``*CalibrationView`` adapter above.
        """
        return cls(
            calibration_bound=predictor.calibration_bound(),
            window=window,
            threshold=threshold,
            persistence=persistence,
            relative=relative,
        )

    @property
    def n_observed(self) -> int:
        """Total number of triples seen so far via :meth:`update`."""
        return self._step

    def update(self, x: Any, prediction: Any, true_value: Any) -> DriftAlert | None:
        """Ingest one ``(input, prediction, true_value)`` triple; return a fresh :class:`DriftAlert` if
        this update is the one that completes ``persistence`` consecutive exceedances, else ``None``.
        """
        self._step += 1
        error = _residual_error(prediction, true_value, relative=self.relative)
        self._errors.append(error)

        if len(self._errors) < self.window:
            self._consecutive_exceedances = 0
            return None

        rolling_error = float(np.mean(self._errors))
        ratio = rolling_error / self.calibration_bound
        self._consecutive_exceedances = self._consecutive_exceedances + 1 if ratio > self.threshold else 0

        if self._consecutive_exceedances < self.persistence:
            return None

        alert = DriftAlert(
            step=self._step,
            input=x,
            window_size=self.window,
            rolling_error=rolling_error,
            calibration_bound=self.calibration_bound,
            exceedance_ratio=ratio,
            consecutive_exceedances=self._consecutive_exceedances,
        )
        self.alerts.append(alert)
        return alert

    def run(self, stream: Iterable[tuple[Any, Any, Any]]) -> list[DriftAlert]:
        """Consume an entire ``(input, prediction, true_value)`` stream via repeated :meth:`update`
        calls; returns every :class:`DriftAlert` raised, in stream order (also accumulated in
        :attr:`alerts`)."""
        raised: list[DriftAlert] = []
        for x, prediction, true_value in stream:
            alert = self.update(x, prediction, true_value)
            if alert is not None:
                raised.append(alert)
        return raised

    def reset(self) -> None:
        """Clear all accumulated stream state (rolling window, consecutive-exceedance count, alert
        history, step counter) without changing the calibration bound or thresholds -- e.g. after the
        underlying surrogate has been refit/recalibrated in response to a raised :class:`DriftAlert`."""
        self._errors.clear()
        self._consecutive_exceedances = 0
        self._step = 0
        self.alerts.clear()
