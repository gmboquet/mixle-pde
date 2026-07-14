"""E6 -- physics-surrogate distillation + cascade.

An expensive forward (a PDE solve, a rock-physics simulator, anything shaped ``teacher(x) -> y``)
is often orders of magnitude slower than a caller needs for interactive what-if exploration.
``distill_forward`` samples the input space, labels the sample with the real ``teacher``, and fits
a fast local student on those ``(input, output)`` pairs -- exactly the numeric-regression shape of
the same distill/calibrate/escalate loop ``mixle.task.solve.solve`` runs for text/record
classification, here run through its regression sibling :mod:`mixle.task.regress`
(``solve_regression``'s ``_fit_scaled``/``_calibrate`` split-conformal machinery) so the returned
``Surrogate`` carries the same kind of honest, calibrated escalation signal.

The returned :class:`Surrogate` answers two questions per input: ``predict(x)`` is the student's
fast point estimate; ``defer(x)`` is the calibrated gate saying whether that estimate should be
trusted or the caller should re-run the real ``teacher``. Two independent conditions can trigger a
defer, mirroring how ``mixle_mlops.models.task_cascade.TaskCascadeAdapter`` frames its own
escalate-or-answer decision for a classifier cascade:

1. **Globally imprecise.** The held-out (never-trained-on) split calibrates a per-output-dimension
   interval half-width ``qhat`` at level ``1 - alpha`` (split conformal, distribution-free). If
   ``qhat`` is no tighter than the teacher's own held-out spread, the interval is telling you
   nothing the raw output distribution didn't already -- the same "prior, not data, sets the
   width" honesty flag :mod:`mixle.reason.posterior_protocol` calls ``prior_dominated``. When that
   happens every input defers; the surrogate isn't fit for purpose yet.
2. **Locally out-of-distribution.** A :class:`mixle.task.density.DensityGate` (the same
   density-escalation primitive ``mixle.task.solve.solve``'s own ``ood=`` knob uses) is fit on the
   training inputs; an input whose ``log p(x)`` falls below the calibrated floor is unlike
   anything the student trained on, regardless of how confident its point estimate looks.

``mixle_mlops.models.task_cascade.TaskCascadeAdapter`` is wired for a label-valued
``CalibratedTaskModel`` (it calls ``.batch``, ``.predict_sets``, ``.task.adapter.labels`` on the
model it wraps) -- there is no regression path there today. Rather than redesign that adapter (it
is shared, unfrozen, general-purpose code, not something this task owns), ``to_task_cascade_adapter``
below builds the minimal duck-typed facade ``TaskCascadeAdapter`` needs so a continuous-valued
``Surrogate`` can still be served through the platform's uniform model surface unmodified; see the
docstring on that function and the PR notes for the specifics of the shim.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from mixle.task.density import DensityGate
from mixle.task.regress import RecordRegressionFeaturizer, _calibrate, _fit_scaled

__all__ = ["Surrogate", "distill_forward", "to_task_cascade_adapter"]


def _as_record(x: Any) -> Any:
    """Normalize one sampled input into the record shape mixle's task featurizers understand.

    ``dict`` records pass through untouched; anything array-like (a numpy row, a tuple/list of
    physical parameters) becomes a plain tuple of floats so :class:`RecordRegressionFeaturizer`
    sees real numeric columns instead of a single opaque, hashed-by-``repr`` value.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, (tuple, list, np.ndarray)):
        return tuple(float(v) for v in np.atleast_1d(np.asarray(x, dtype=float)).tolist())
    return x


def _as_matrix(ys: Sequence[Any]) -> np.ndarray:
    """Stack teacher outputs into an ``(n, k)`` float matrix, whether ``teacher`` returns a scalar or a vector."""
    arr = np.asarray(ys, dtype=np.float64)
    return arr.reshape(len(ys), -1) if arr.ndim <= 1 else arr


@dataclass
class Surrogate:
    """A distilled, conformally-calibrated stand-in for an expensive physics forward.

    ``predict(x)`` always returns the fast student's estimate (a float when the teacher returns a
    scalar, an ``(k,)`` array when it returns a vector). ``defer(x)`` is the honest gate: True when
    the calibrated interval is not tight enough to trust (a global precision floor, ``qhat >
    tol``) or ``x`` is out-of-distribution relative to the training sample (a per-input density
    check) -- either way, re-run ``teacher`` instead of trusting this estimate.
    """

    nets: list[Any]
    featurizer: RecordRegressionFeaturizer
    gate: DensityGate
    teacher: Callable[..., Any]
    qhat: np.ndarray  # (k,) calibrated interval half-width per output dimension
    tol: np.ndarray  # (k,) precision floor per output dimension (see module docstring, condition 1)
    y_mean: np.ndarray
    y_scale: np.ndarray
    holdout_mae: np.ndarray
    scalar_output: bool = True
    alpha: float = 0.1
    seed: int = 0
    budget: int = 0
    n_requests: int = 0
    n_deferred: int = 0

    def _predict_raw(self, records: list[Any]) -> np.ndarray:
        import torch

        feats = np.asarray(self.featurizer.transform(records), dtype=np.float32)
        cols = []
        with torch.no_grad():
            for j, net in enumerate(self.nets):
                out = net(torch.as_tensor(feats)).numpy()[:, 0]
                cols.append(out * self.y_scale[j] + self.y_mean[j])
        return np.stack(cols, axis=1)  # (n, k)

    def predict(self, x: Any) -> Any:
        """The surrogate's fast point estimate for ``x`` -- a float if ``teacher`` is scalar-valued."""
        row = self._predict_raw([_as_record(x)])[0]
        return float(row[0]) if self.scalar_output else row

    @property
    def imprecise(self) -> bool:
        """Whether the calibrated interval fails the precision floor for *every* input (a global flag)."""
        return bool(np.any(self.qhat > self.tol))

    def is_ood(self, x: Any) -> bool:
        """Whether ``x`` falls below the density gate's calibrated in-distribution floor."""
        return bool(self.gate.ood_mask([_as_record(x)])[0])

    def defer(self, x: Any) -> bool:
        """True when ``x`` should be escalated to the real ``teacher`` instead of trusting ``predict(x)``."""
        self.n_requests += 1
        out = self.imprecise or self.is_ood(x)
        if out:
            self.n_deferred += 1
        return out

    def evaluate(self, xs: Sequence[Any]) -> dict[str, Any]:
        """Measure surrogate-vs-teacher error over ``xs`` -- a diagnostic, independent of the fit's own
        calibration split (step 4 of the E6 recipe: "measure surrogate-vs-full error on a held-out set")."""
        records = [_as_record(x) for x in xs]
        pred = self._predict_raw(records)
        truth = _as_matrix([self.teacher(x) for x in records])
        err = np.abs(pred - truth)
        return {
            "n": len(records),
            "mae": err.mean(axis=0).tolist(),
            "max_abs_error": err.max(axis=0).tolist(),
            "deferral_rate": float(np.mean([self.defer(x) for x in records])) if records else float("nan"),
        }

    @property
    def deferral_rate(self) -> float:
        """Live deferral rate across every :meth:`defer` call so far (requests are cumulative)."""
        return (self.n_deferred / self.n_requests) if self.n_requests else 0.0

    def report(self) -> dict[str, Any]:
        """Calibration + live deferral summary -- what a dashboard would want."""
        return {
            "budget": self.budget,
            "alpha": self.alpha,
            "qhat": self.qhat.tolist(),
            "tol": self.tol.tolist(),
            "imprecise": self.imprecise,
            "holdout_mae": self.holdout_mae.tolist(),
            "requests": self.n_requests,
            "deferred": self.n_deferred,
            "deferral_rate": self.deferral_rate,
        }


def distill_forward(
    teacher: Callable[[Any], Any],
    sampler: Callable[[int, np.random.Generator], Any],
    *,
    budget: int,
    seed: int = 0,
    alpha: float = 0.1,
    holdout: float = 0.25,
    ood_alpha: float = 0.05,
    n_components: int = 3,
    hidden: Sequence[int] = (64,),
    epochs: int = 300,
    lr: float = 1e-2,
    dim: int = 32,
) -> Surrogate:
    """Distill a fast, calibrated :class:`Surrogate` of an expensive ``teacher`` forward (E6 recipe).

    Args:
        teacher: the expensive forward being replaced, ``teacher(x) -> float`` or ``teacher(x) ->
            array-like``. Called exactly ``budget`` times (once per sampled input) -- the cost this
            whole exercise is trying to amortize.
        sampler: draws candidate inputs to label, ``sampler(n, rng) -> Sequence`` of length ``n``
            (the same ``samples(n, rng)`` shape as :mod:`mixle.reason.posterior_protocol`'s
            ``Posterior``, e.g. a design-of-experiments draw or a posterior pushforward).
        budget: total number of expensive ``teacher`` calls to spend on labeling (train + held-out
            calibration together; nothing here calls ``teacher`` a second time).
        seed: determinism for sampling, the train/calibration split, student init, and the density
            fit.
        alpha: split-conformal miscoverage (``1 - alpha`` interval coverage of the teacher's value).
        holdout: fraction of the budget reserved for calibration (never trained on).
        ood_alpha: density quantile the out-of-distribution floor is set at (see
            :meth:`mixle.task.density.DensityGate.fit`).
        n_components: number of mixture components in the input density estimate.
        hidden, epochs, lr, dim: student network width, training length, learning rate, and the
            featurizer's hashed-categorical width (see :class:`mixle.task.regress
            .RecordRegressionFeaturizer` -- physics inputs are almost always plain numeric tuples,
            so this only matters when a sampled input carries non-numeric fields).

    Returns:
        A calibrated :class:`Surrogate`. Step 4 of the recipe (measuring surrogate-vs-full error to
        set the gate) is exactly the split-conformal calibration below: ``qhat`` is derived from
        held-out surrogate-vs-teacher residuals, and the precision floor ``tol`` is the teacher's
        own held-out spread, so an uninformative student can never masquerade as confident.
    """
    if budget < 16:
        raise ValueError("distill_forward needs a budget of at least 16 teacher calls")

    rng = np.random.default_rng(seed)
    raw_inputs = list(sampler(budget, rng))
    if len(raw_inputs) != budget:
        raise ValueError(f"sampler({budget}, rng) returned {len(raw_inputs)} inputs, expected {budget}")
    records = [_as_record(x) for x in raw_inputs]
    raw_ys = [teacher(x) for x in records]
    scalar_output = all(np.ndim(y) == 0 for y in raw_ys)
    ys = _as_matrix(raw_ys)

    split_rng = np.random.RandomState(seed)
    order = split_rng.permutation(len(records))
    n_cal = max(4, int(round(len(records) * holdout)))
    cal_idx, train_idx = order[:n_cal], order[n_cal:]
    train_records = [records[i] for i in train_idx]
    cal_records = [records[i] for i in cal_idx]
    train_ys, cal_ys = ys[train_idx], ys[cal_idx]

    featurizer = RecordRegressionFeaturizer(dim=dim, seed=seed).fit(train_records)

    nets, y_means, y_scales, qhats, maes = [], [], [], [], []
    for k in range(ys.shape[1]):
        cand = _fit_scaled(
            train_records, train_ys[:, k].tolist(), featurizer, tuple(hidden), int(epochs), float(lr), int(seed)
        )
        qhat, mae = _calibrate(cand, featurizer, cal_records, cal_ys[:, k].tolist(), float(alpha))
        nets.append(cand[0])
        y_means.append(cand[1][0])
        y_scales.append(cand[1][1])
        qhats.append(qhat)
        maes.append(mae)

    # the precision floor: an interval no tighter than the teacher's own held-out spread carries no
    # information (mirrors IC-1 DerivedQuantity's `prior_dominated` honesty flag, transplanted here)
    tol = np.array([max(1e-9, float(np.std(cal_ys[:, k]))) for k in range(ys.shape[1])])

    gate = DensityGate(featurizer).fit(train_records, n_components=n_components, alpha=ood_alpha, seed=seed)

    return Surrogate(
        nets=nets,
        featurizer=featurizer,
        gate=gate,
        teacher=teacher,
        qhat=np.asarray(qhats),
        tol=tol,
        y_mean=np.asarray(y_means),
        y_scale=np.asarray(y_scales),
        holdout_mae=np.asarray(maes),
        scalar_output=scalar_output,
        alpha=alpha,
        seed=seed,
        budget=budget,
    )


class _CascadeTaskFacade:
    """Minimal ``.batch`` / ``.adapter`` surface a :class:`Surrogate` needs to sit behind
    ``TaskCascadeAdapter``'s classifier-shaped ``_task`` slot (see module docstring)."""

    class _Adapter:
        labels: list[str] = []

    def __init__(self, surrogate: Surrogate) -> None:
        self._surrogate = surrogate
        self.adapter = self._Adapter()

    def batch(self, records: list[Any]) -> list[str]:
        return [repr(self._surrogate.predict(r)) for r in records]


class _CascadeModelFacade:
    """Minimal ``.decide`` / ``.task`` / ``predict_sets`` / ``_escalate_flags`` surface
    ``TaskCascadeAdapter`` needs to treat a :class:`Surrogate` as a "calibrated" model."""

    def __init__(self, surrogate: Surrogate) -> None:
        self.task = _CascadeTaskFacade(surrogate)
        self._surrogate = surrogate

    def decide(self, x: Any) -> Any:
        return None if self._surrogate.defer(x) else self._surrogate.predict(x)

    def predict_set(self, x: Any) -> list[str]:
        return [] if self._surrogate.defer(x) else [repr(self._surrogate.predict(x))]

    def predict_sets(self, records: list[Any]) -> list[list[str]]:
        return [self.predict_set(r) for r in records]

    def _escalate_flags(self, records: list[Any], sets: list[list[str]]) -> list[bool]:
        return [len(s) != 1 for s in sets]


def to_task_cascade_adapter(surrogate: Surrogate, name: str) -> Any:
    """Host ``surrogate`` behind the platform's uniform ``/v1`` model surface via ``TaskCascadeAdapter``.

    ``TaskCascadeAdapter`` (``mixle_mlops.models.task_cascade``) is wired for a label-valued
    ``CalibratedTaskModel``: it duck-types its wrapped model's ``.batch``, ``.decide``,
    ``.predict_sets``, ``._escalate_flags``, and ``.task.adapter.labels``. There is no continuous
    regression path in that adapter today, and it is shared, unfrozen, general-purpose serving code
    this task does not own -- so rather than redesign it, this builds the minimal facade above that
    satisfies exactly that surface, translating ``Surrogate.predict``/``Surrogate.defer`` into the
    adapter's escalate-or-answer shape (a defer becomes an empty conformal "set"; an answer becomes
    a length-one set holding the surrogate's stringified point estimate). ``score`` and the
    class-probability surface stay unavailable, correctly, since there is no label distribution
    behind a continuous forward.
    """
    from mixle_mlops.models.task_cascade import TaskCascadeAdapter

    return TaskCascadeAdapter(name, _CascadeModelFacade(surrogate))
