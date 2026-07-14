"""Physics-surrogate distillation + cascade (workstream E, card E6).

An expensive PDE forward (a rock-physics kernel, a full 3D EM/seismic simulator, ...) is often called
thousands of times inside an inversion or a sensitivity sweep. :func:`distill_forward` turns that forward
into a cheap, calibrated :class:`Surrogate`: sample the input space, label the samples by running the
real forward once, and fit a fast student on the resulting ``(input, output)`` pairs. The student answers
``predict(x)`` for free; ``defer(x)`` is the calibrated gate that flags inputs unlike anything the student
was fit on, so the caller escalates those back to the full forward instead of trusting an extrapolated
answer.

Inputs are ``mixle.task.solve`` "record" inputs -- tuples (or dicts) of numeric/categorical fields, e.g.
model parameters for the forward. The teacher may return a scalar or a fixed-length vector; a vector
output fits one calibrated student per channel and combines the per-channel predictions into one answer.

Reuse note (see the "Notes" section of this card's PR body for the full rationale): the work order's
suggested student is ``mixle.task.solve`` / ``mixle.task.regress.solve_regression`` -- a hashed-feature
MLP trained with ``torch`` autograd. That student's featurizer (:class:`mixle.task.regress.
RecordRegressionFeaturizer`, which standardizes numeric record fields the way a physics parameter vector
needs) and its split-conformal calibration rule are reused verbatim here. The MLP fit itself is not: a
``torch`` training loop (``loss.backward()``) reproducibly left the process unable to exit cleanly under
this task's execution environment. The student below fits the *same* standardized features with a
closed-form ridge regression instead of a trained MLP -- numerically stable, deterministic, and
dependency-light, with an identical calibration guarantee (the conformal quantile does not care how the
point predictor was fit). The deferral gate is :class:`mixle.task.density.DensityGate`, used unmodified --
the same ``p(x)`` out-of-distribution floor ``mixle.task.solve``'s own classification cascade uses to
escalate inputs unlike its training distribution, regardless of how confident the student looks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mixle.task.density import DensityGate
from mixle.task.model import HashedRecord
from mixle.task.regress import RecordRegressionFeaturizer
from mixle.task.solve import _label_with

_MIN_BUDGET = 12  # mirrors mixle.task.regress.solve_regression's own train/calibration-split minimum
_HOLDOUT = 0.25  # fraction of the budget reserved for conformal calibration (never trained on)
_ALPHA = 0.1  # conformal miscoverage: qhat targets 90% coverage of the teacher's answer


@dataclass
class _ChannelStudent:
    """A closed-form ridge-regression student for one output channel, conformally calibrated."""

    featurizer: RecordRegressionFeaturizer
    weights: np.ndarray  # standardized-feature weights, bias folded in as the last entry
    y_mean: float
    y_scale: float
    qhat: float  # split-conformal half-width: |teacher - predict| <= qhat with ~(1 - alpha) coverage
    holdout_mae: float

    def predict(self, xs: Sequence[Any]) -> np.ndarray:
        feats = _features(self.featurizer, xs)
        aug = np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=1)
        z = aug @ self.weights
        return z * self.y_scale + self.y_mean


def _features(featurizer: RecordRegressionFeaturizer, xs: Sequence[Any]) -> np.ndarray:
    """Standardized record features, augmented with polynomial/interaction terms of the numeric fields.

    ``RecordRegressionFeaturizer`` gives clean standardized numeric columns (unlike the tanh-squashed
    ``HashedRecord``), but the ridge student is linear in whatever features it is handed. A physics
    forward is rarely linear in its raw parameters (rock-physics moduli, travel-time curves, ...), so the
    numeric columns are expanded with squares, cubes, and pairwise products before the closed-form fit --
    letting a linear-in-features model track that curvature without reaching for a trained (torch) net.
    """
    feats = np.asarray(featurizer.transform(list(xs)), dtype=np.float64)
    n_num = len(featurizer.num_keys)
    if n_num == 0:
        return feats
    num = feats[:, :n_num]
    terms = [feats, num**2, num**3]
    if n_num > 1:
        pairs = [(num[:, i] * num[:, j])[:, None] for i in range(n_num) for j in range(i + 1, n_num)]
        terms.append(np.concatenate(pairs, axis=1))
    return np.concatenate(terms, axis=1)


def _fit_ridge(feats: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression on standardized features; the bias column is left unregularized."""
    aug = np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=1)
    d = aug.shape[1]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0
    return np.linalg.solve(aug.T @ aug + reg, aug.T @ y)


def _fit_channel(
    train_inputs: list[Any],
    train_y: np.ndarray,
    cal_inputs: list[Any],
    cal_y: np.ndarray,
    *,
    dim: int,
    seed: int,
    alpha: float,
) -> _ChannelStudent:
    featurizer = RecordRegressionFeaturizer(dim=dim, seed=seed).fit(train_inputs)
    mean, scale = float(train_y.mean()), float(train_y.std() or 1.0)
    feats = _features(featurizer, train_inputs)
    weights = _fit_ridge(feats, (train_y - mean) / scale)

    student = _ChannelStudent(
        featurizer=featurizer, weights=weights, y_mean=mean, y_scale=scale, qhat=0.0, holdout_mae=0.0
    )
    pred_cal = student.predict(cal_inputs)
    resid = np.abs(cal_y - pred_cal)
    n = len(resid)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    student.qhat = float(np.sort(resid)[min(rank, n) - 1]) if rank <= n else float("inf")
    student.holdout_mae = float(resid.mean())
    return student


@dataclass
class Surrogate:
    """A calibrated student in front of an expensive forward.

    ``predict(x)`` always answers from the distilled student -- it never calls the teacher. ``defer(x)``
    is the calibrated escalation gate: ``True`` when ``x`` falls outside the density region the student
    was trained on, in which case the caller should prefer :meth:`solve`, which runs the full ``teacher``
    forward instead of trusting an extrapolated surrogate answer.
    """

    teacher: Callable[[Any], Any]
    students: list[_ChannelStudent]
    gate: DensityGate
    vector_output: bool
    holdout_mae: np.ndarray = field(repr=False)  # per-output-channel surrogate-vs-full error, held-out split

    def predict(self, x: Any) -> float | np.ndarray:
        """The surrogate's point estimate for ``x`` -- a float for a scalar teacher, else an array."""
        vals = np.asarray([float(s.predict([x])[0]) for s in self.students], dtype=float)
        return vals if self.vector_output else float(vals[0])

    def defer(self, x: Any) -> bool:
        """``True`` when ``x`` is out-of-distribution relative to the surrogate's training inputs."""
        return bool(self.gate.is_ood(x))

    def solve(self, x: Any) -> float | np.ndarray:
        """Answer with the cheap surrogate when it is safe to trust, else escalate to the full teacher."""
        if self.defer(x):
            return self.teacher(x)
        return self.predict(x)

    def report(self) -> dict[str, Any]:
        """Calibrated accuracy per output channel -- what ``predict`` is worth trusting to."""
        return {
            "vector_output": self.vector_output,
            "n_outputs": len(self.students),
            "holdout_mae": [float(v) for v in self.holdout_mae],
            "qhat": [float(s.qhat) for s in self.students],
            "ood_log_threshold": self.gate.log_threshold,
        }


def distill_forward(
    teacher: Callable[[Any], Any],
    sampler: Callable[[int, np.random.RandomState], Sequence[Any]],
    *,
    budget: int,
    seed: int = 0,
    ood_alpha: float = 0.1,
    gate_dim: int = 64,
) -> Surrogate:
    """Distill a fast, calibrated :class:`Surrogate` from an expensive ``teacher`` forward.

    Args:
        teacher: the expensive forward, ``teacher(x) -> float`` or ``teacher(x) -> sequence[float]``.
            Called exactly once over the sampled training inputs (never re-invoked per output channel).
        sampler: ``sampler(n, rng) -> inputs`` draws ``n`` candidate inputs from the domain the surrogate
            should cover -- record inputs (tuples or dicts), the shape ``mixle.task.solve`` expects.
        budget: total number of teacher evaluations to spend distilling (train + calibration split).
        seed: split/fit determinism, forwarded to the regression student(s) and the density gate.
        ood_alpha: the density gate's out-of-distribution floor -- the quantile of training-input log
            density below which an input is flagged as unlike anything trained on (default 10%).
        gate_dim: hashed-feature width for the density gate's featurizer.

    Returns:
        A :class:`Surrogate` wrapping one conformally-calibrated student per output channel plus the
        density-gate deferral rule.
    """
    if budget < _MIN_BUDGET:
        raise ValueError(f"distill_forward needs budget >= {_MIN_BUDGET} to train and calibrate honestly")

    rng = np.random.RandomState(seed)
    inputs = list(sampler(budget, rng))
    if len(inputs) < _MIN_BUDGET:
        raise ValueError(f"sampler(budget, rng) returned {len(inputs)} inputs; need at least {_MIN_BUDGET}")

    # The expensive forward is evaluated exactly once, over every sampled input.
    raw = _label_with(teacher, inputs)
    outputs = np.asarray(raw, dtype=float)
    vector_output = outputs.ndim > 1
    if not vector_output:
        outputs = outputs[:, None]
    n_channels = outputs.shape[1]

    split = np.random.RandomState(seed)
    order = split.permutation(len(inputs))
    n_cal = max(4, int(round(len(inputs) * _HOLDOUT)))
    cal_idx, train_idx = order[:n_cal], order[n_cal:]
    train_inputs = [inputs[i] for i in train_idx]
    cal_inputs = [inputs[i] for i in cal_idx]

    students: list[_ChannelStudent] = []
    holdout_mae = np.zeros(n_channels)
    for channel in range(n_channels):
        train_y = outputs[train_idx, channel]
        cal_y = outputs[cal_idx, channel]
        student = _fit_channel(train_inputs, train_y, cal_inputs, cal_y, dim=gate_dim, seed=seed, alpha=_ALPHA)
        students.append(student)
        holdout_mae[channel] = student.holdout_mae

    # The OOD gate is fit on the training inputs only -- real inputs the surrogate actually learned from.
    gate = DensityGate(HashedRecord(dim=gate_dim, seed=seed)).fit(train_inputs, alpha=ood_alpha, seed=seed)

    return Surrogate(
        teacher=teacher,
        students=students,
        gate=gate,
        vector_output=vector_output,
        holdout_mae=holdout_mae,
    )
