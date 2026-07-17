"""Linear/low-rank operator surrogate: field-to-field prediction via a learned reduced linear map (MP-N5).

Source: `notes/mixle-pde-ai-native-multiphysics-work-plan.md` workstream N, **MP-N5** ("Spatial/temporal/
geometry-varying operator surrogates"). The reconciliation ledger (`docs/reconciliation/mp-task-ledger.md`)
records MP-N5 as `not-started`: "No FNO/DeepONet/graph-neural/geometry-conditioned operator surrogate
found."

**This module is not that.** A Fourier Neural Operator, a DeepONet, or a graph-neural operator surrogate
each need real deep-learning infrastructure (a trained nonlinear network, a meaningful training-data scale,
and a validated generalization story) that is out of scope for one bounded implementation task and would be
irresponsible to claim without it. What is implemented here is the honestly-scoped linear baseline one rung
above :mod:`mixle_pde.reduced_basis`'s single-field POD compression: given **paired** input/output field
snapshots ``(u_i, v_i)``, fit two independent POD bases (via :func:`mixle_pde.reduced_basis.build_pod_basis`
-- one for the input family, one for the output family, so ``u`` and ``v`` may live on entirely different
meshes/discretizations of different sizes, the "geometry-varying" half of MP-N5's title) and then fit a
single **linear regression map between the two bases' reduced coordinates** by ridge-regularized least
squares. Predicting a new field is: project the input onto its basis, apply the learned reduced linear map,
reconstruct the predicted output from the output basis. There is no neural network, no nonlinearity, and no
autoregressive time-stepping anywhere in this module.

This is deliberately **not** a Galerkin-projected reduced operator either: a true Galerkin ROM projects the
*governing weak form* (the actual discretized PDE operator) onto the reduced basis, which needs weak-form
machinery this stack does not expose (:mod:`mixle_pde.reduced_basis`'s own module docstring makes the same
disclaimer). What this module fits instead is a **non-intrusive, data-driven reduced-order regression**:
the linear map is learned purely from the paired snapshots' reduced coordinates, the same "POD + regression"
family of technique surveyed e.g. by Audouze/De Vuyst/Nair (2013) and Guo/Hesthaven (2019) as a common
non-intrusive-ROM baseline. Callers who need the true governing operator's Galerkin projection, hyper-
reduction (DEIM), or transient reduced-order state-space integration are not served by this module.

**Honesty gate.** A linear map fit from a finite snapshot set has no way to know whether a new query input
resembles its training data, so :meth:`LinearOperatorSurrogate.predict` reports an ``in_domain`` flag per
query built from two independent checks, mirroring the geometric envelope check in
:mod:`mixle_pde.qoi_emulator` and the held-out reconstruction-error certification in
:mod:`mixle_pde.reduced_basis`:

1. **Reduced-coordinate envelope.** The query's own projected reduced coordinates must fall within the
   training inputs' reduced-coordinate envelope (standardized per component, expanded by ``ood_margin``) --
   the same "is this near any training coordinate" geometric rule
   :mod:`mixle_pde.qoi_emulator` uses, applied here in the input POD basis' reduced space.
2. **Input reconstruction fidelity.** The query field's own POD-projection residual (via
   :func:`mixle_pde.reduced_basis.project`/:func:`~mixle_pde.reduced_basis.reconstruct`) must not exceed a
   calibrated multiple of the worst reconstruction residual observed among the *training* inputs -- a query
   whose own shape the input basis cannot represent well is not something the fitted reduced map was ever
   exposed to, regardless of where its reduced coordinates happen to land.

Separately, :func:`calibrate_linear_operator_surrogate` scores the fitted surrogate against **held-out**
input/output pairs it was not fit from (never the training pairs -- see
:func:`mixle_pde.reduced_basis.reduced_basis_error`'s identical caution) and reports a split-conformal
calibrated relative-error bound ``qhat_relative_l2_error`` (the finite-sample-corrected quantile used by
:mod:`mixle_pde.surrogate`'s own ``qhat`` calibration) alongside ``baseline_relative_l2_error`` -- the error
of the trivial "always predict the training output mean field" baseline on the same held-out set. When the
calibrated bound is no tighter than that trivial baseline (``imprecise=True``), the fitted map carries no
more information than guessing the mean output field, mirroring :mod:`mixle_pde.surrogate`'s own
``Surrogate.imprecise`` precision floor. Nothing in this module reports a prediction as trustworthy purely
because it looks numerically confident.

Scope note (baseline only, matching the "one clean increment" precedent of this workstream's siblings):
linear maps only (no nonlinear/neural operator family), single-shot field-to-field prediction only (no
autoregressive multi-step rollout), a fixed ridge-regularized least-squares fit only (no adaptive rank
selection, no cross-validated ridge search), and a point-estimate + calibrated-error-bound honesty report
only (no genuine Bayesian posterior over the output field -- this is a deterministic linear regression, not
a Gaussian process, so it must not be described as one).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle_pde.reduced_basis import PODBasis, build_pod_basis, project, reconstruct

__all__ = [
    "LinearOperatorSurrogate",
    "OperatorPrediction",
    "LinearOperatorCalibrationReport",
    "fit_linear_operator_surrogate",
    "calibrate_linear_operator_surrogate",
]


def _relative_norms(residual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Column-wise ``||residual|| / ||reference||``, with a safe floor for a near-zero reference column."""
    ref_norms = np.linalg.norm(reference, axis=0)
    ref_norms_safe = np.where(ref_norms > 1e-12, ref_norms, 1.0)
    return np.linalg.norm(residual, axis=0) / ref_norms_safe


@dataclass(frozen=True)
class OperatorPrediction:
    """A batch of predicted output fields plus this module's honesty gate.

    ``field`` is ``(n_dof_out,)`` for a single query or ``(n_dof_out, n_queries)`` for a batch, mirroring
    :func:`mixle_pde.reduced_basis.reconstruct`'s own single/batch convention. ``in_domain`` and
    ``input_relative_residual`` are always 1-D arrays of length ``n_queries`` (length 1 for a single query)
    -- see the module docstring's "Honesty gate" section for what ``in_domain`` combines.
    """

    field: np.ndarray
    in_domain: np.ndarray
    input_relative_residual: np.ndarray

    def __post_init__(self) -> None:
        in_domain = np.asarray(self.in_domain, dtype=bool).reshape(-1)
        residual = np.asarray(self.input_relative_residual, dtype=np.float64).reshape(-1)
        if in_domain.shape != residual.shape:
            raise ValueError(
                "OperatorPrediction.in_domain and input_relative_residual must have the same length; got "
                f"{in_domain.shape} vs {residual.shape}."
            )
        n_queries = in_domain.shape[0]
        if self.field.ndim == 1:
            if n_queries != 1:
                raise ValueError(
                    f"a 1-D OperatorPrediction.field is a single query; expected 1 in_domain entry, got {n_queries}."
                )
        elif self.field.ndim == 2:
            if self.field.shape[1] != n_queries:
                raise ValueError(
                    f"OperatorPrediction.field has {self.field.shape[1]} query columns but in_domain has "
                    f"{n_queries} entries."
                )
        else:
            raise ValueError(f"OperatorPrediction.field must be 1-D or 2-D; got shape {self.field.shape}.")
        object.__setattr__(self, "in_domain", in_domain)
        object.__setattr__(self, "input_relative_residual", residual)


@dataclass(frozen=True)
class LinearOperatorCalibrationReport:
    """Held-out generalization report produced by :func:`calibrate_linear_operator_surrogate`.

    ``qhat_relative_l2_error`` is the split-conformal (finite-sample-corrected) ``1 - alpha`` quantile of
    held-out relative-L2 reconstruction error -- at least a ``1 - alpha`` fraction of exchangeable future
    queries are expected to reconstruct at least this well, the same style of distribution-free guarantee
    :mod:`mixle_pde.surrogate`'s own conformal ``qhat`` reports. ``baseline_relative_l2_error`` is the same
    metric for the trivial "always predict the training output mean field" baseline on the identical
    held-out set; ``imprecise`` is true when the fitted operator is no tighter than that baseline.
    """

    n: int
    alpha: float
    mean_relative_l2_error: float
    max_relative_l2_error: float
    qhat_relative_l2_error: float
    baseline_relative_l2_error: float
    imprecise: bool
    ood_fraction: float

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"LinearOperatorCalibrationReport.n must be positive; got {self.n!r}.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha!r}.")
        for name in (
            "mean_relative_l2_error",
            "max_relative_l2_error",
            "qhat_relative_l2_error",
            "baseline_relative_l2_error",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
        if not (0.0 <= self.ood_fraction <= 1.0):
            raise ValueError(f"ood_fraction must be a fraction in [0, 1]; got {self.ood_fraction!r}.")


@dataclass(frozen=True)
class LinearOperatorSurrogate:
    """A fitted linear/low-rank field-to-field operator surrogate. Build with
    :func:`fit_linear_operator_surrogate`; every field here derives from the training snapshots together, so
    construct through that function rather than directly.

    ``basis_in``/``basis_out`` are independent POD bases (input and output fields may have different
    dimension and live on different meshes -- MP-N5's "geometry-varying" requirement). ``coefficient_map``
    (shape ``(basis_out.rank, basis_in.rank)``) is the fitted linear map between their reduced coordinates:
    ``reduced_out ~= coefficient_map @ reduced_in``.
    """

    basis_in: PODBasis
    basis_out: PODBasis
    coefficient_map: np.ndarray
    reduced_in_mean: np.ndarray
    reduced_in_scale: np.ndarray
    reduced_in_std_min: np.ndarray
    reduced_in_std_max: np.ndarray
    train_output_mean: np.ndarray
    max_train_input_relative_residual: float
    ridge: float = 1e-6
    ood_margin: float = 0.25
    input_residual_margin: float = 2.0

    def __post_init__(self) -> None:
        r_in, r_out = self.basis_in.rank, self.basis_out.rank
        if self.coefficient_map.shape != (r_out, r_in):
            raise ValueError(
                f"coefficient_map must have shape (basis_out.rank={r_out}, basis_in.rank={r_in}); "
                f"got {self.coefficient_map.shape}."
            )
        for name in ("reduced_in_mean", "reduced_in_scale", "reduced_in_std_min", "reduced_in_std_max"):
            shape = getattr(self, name).shape
            if shape != (r_in,):
                raise ValueError(f"{name} must have shape (basis_in.rank={r_in},); got {shape}.")
        if self.train_output_mean.shape != (self.basis_out.n_dof,):
            raise ValueError(
                f"train_output_mean must have shape (basis_out.n_dof={self.basis_out.n_dof},); "
                f"got {self.train_output_mean.shape}."
            )
        if self.max_train_input_relative_residual < 0.0:
            raise ValueError("max_train_input_relative_residual must be non-negative.")
        if self.ridge <= 0.0:
            raise ValueError(
                f"ridge must be strictly positive (well-posedness of the reduced fit); got {self.ridge!r}."
            )
        if self.ood_margin < 0.0:
            raise ValueError(f"ood_margin must be non-negative; got {self.ood_margin!r}.")
        if self.input_residual_margin < 0.0:
            raise ValueError(f"input_residual_margin must be non-negative; got {self.input_residual_margin!r}.")

    def predict(self, u) -> OperatorPrediction:
        """Predict the output field(s) for input field(s) ``u`` (``(n_dof_in,)`` or ``(n_dof_in, n_queries)``).

        Always returns a full point-estimate prediction -- this method never refuses to answer. Check
        ``.in_domain`` (see the module docstring's "Honesty gate" section) before trusting a query that
        strays from the training snapshots' span.
        """
        field = np.asarray(u, dtype=float)
        if field.ndim not in (1, 2):
            raise ValueError(f"u must be 1-D (n_dof_in,) or 2-D (n_dof_in, n_queries); got shape {field.shape}.")
        if field.shape[0] != self.basis_in.n_dof:
            raise ValueError(
                f"u's leading dimension must be basis_in.n_dof={self.basis_in.n_dof}; got shape {field.shape}."
            )
        single = field.ndim == 1
        field2 = field[:, None] if single else field

        c_in = project(field2, self.basis_in)  # (rank_in, n_queries)
        c_out = self.coefficient_map @ c_in  # (rank_out, n_queries)
        out_field = reconstruct(c_out, self.basis_out)  # (n_dof_out, n_queries)

        c_in_std = (c_in - self.reduced_in_mean[:, None]) / self.reduced_in_scale[:, None]
        lo = self.reduced_in_std_min[:, None] - self.ood_margin
        hi = self.reduced_in_std_max[:, None] + self.ood_margin
        envelope_ok = np.all((c_in_std >= lo) & (c_in_std <= hi), axis=0)  # (n_queries,)

        recon_in = reconstruct(c_in, self.basis_in)
        input_relative_residual = _relative_norms(field2 - recon_in, field2)
        fidelity_floor = max(1e-9, self.input_residual_margin * self.max_train_input_relative_residual)
        fidelity_ok = input_relative_residual <= fidelity_floor

        in_domain = envelope_ok & fidelity_ok

        return OperatorPrediction(
            field=out_field[:, 0] if single else out_field,
            in_domain=in_domain,
            input_relative_residual=input_relative_residual,
        )


def fit_linear_operator_surrogate(
    inputs,
    outputs,
    *,
    rank_in: int | None = None,
    energy_threshold_in: float | None = None,
    rank_out: int | None = None,
    energy_threshold_out: float | None = None,
    ridge: float = 1e-6,
    ood_margin: float = 0.25,
    input_residual_margin: float = 2.0,
) -> LinearOperatorSurrogate:
    """Fit a :class:`LinearOperatorSurrogate` from paired ``(n_dof_in, n_train)``/``(n_dof_out, n_train)``
    input/output snapshot matrices (matching columns: ``inputs[:, i]`` produced ``outputs[:, i]``).

    ``rank_in``/``energy_threshold_in`` (exactly one required) select the input POD basis via
    :func:`mixle_pde.reduced_basis.build_pod_basis`; ``rank_out``/``energy_threshold_out`` do the same,
    independently, for the output basis -- ``inputs`` and ``outputs`` may have different ``n_dof`` (input
    and output fields on different meshes/discretizations). The reduced linear map is then fit by
    ridge-regularized least squares in the two bases' reduced coordinates: ``coefficient_map`` minimizes
    ``||C_out - coefficient_map @ C_in||^2 + ridge * ||coefficient_map||^2`` in closed form.

    Raises:
        ValueError: mismatched/insufficient snapshot columns, non-finite input, a non-positive ``ridge``,
            a negative ``ood_margin``/``input_residual_margin``, or (propagated from
            :func:`~mixle_pde.reduced_basis.build_pod_basis`) an invalid rank/energy-threshold choice for
            either basis.
    """
    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    if inputs.ndim != 2 or outputs.ndim != 2:
        raise ValueError(
            f"inputs and outputs must both be 2-D (n_dof, n_train); got shapes {inputs.shape}, {outputs.shape}."
        )
    if inputs.shape[1] != outputs.shape[1]:
        raise ValueError(
            "inputs and outputs must have the same number of paired snapshot columns; got "
            f"{inputs.shape[1]} vs {outputs.shape[1]}."
        )
    if inputs.shape[1] < 2:
        raise ValueError("fit_linear_operator_surrogate needs at least two paired input/output snapshots.")
    if not (np.all(np.isfinite(inputs)) and np.all(np.isfinite(outputs))):
        raise ValueError("inputs and outputs must be finite.")
    if ridge <= 0.0:
        raise ValueError(f"ridge must be strictly positive (well-posedness of the reduced fit); got {ridge!r}.")
    if ood_margin < 0.0:
        raise ValueError(f"ood_margin must be non-negative; got {ood_margin!r}.")
    if input_residual_margin < 0.0:
        raise ValueError(f"input_residual_margin must be non-negative; got {input_residual_margin!r}.")

    basis_in = build_pod_basis(inputs, rank=rank_in, energy_threshold=energy_threshold_in)
    basis_out = build_pod_basis(outputs, rank=rank_out, energy_threshold=energy_threshold_out)

    c_in = project(inputs, basis_in)  # (rank_in, n_train)
    c_out = project(outputs, basis_out)  # (rank_out, n_train)

    gram = c_in @ c_in.T
    regularized = gram + ridge * np.eye(basis_in.rank)
    rhs = c_in @ c_out.T
    coefficient_map = np.linalg.solve(regularized, rhs).T  # (rank_out, rank_in)

    reduced_in_mean = c_in.mean(axis=1)
    reduced_in_scale = c_in.std(axis=1)
    reduced_in_scale = np.where(reduced_in_scale > 1e-12, reduced_in_scale, 1.0)
    c_in_std = (c_in - reduced_in_mean[:, None]) / reduced_in_scale[:, None]

    recon_in_train = reconstruct(c_in, basis_in)
    train_input_relative_residual = _relative_norms(inputs - recon_in_train, inputs)

    return LinearOperatorSurrogate(
        basis_in=basis_in,
        basis_out=basis_out,
        coefficient_map=coefficient_map,
        reduced_in_mean=reduced_in_mean,
        reduced_in_scale=reduced_in_scale,
        reduced_in_std_min=c_in_std.min(axis=1),
        reduced_in_std_max=c_in_std.max(axis=1),
        train_output_mean=outputs.mean(axis=1),
        max_train_input_relative_residual=float(np.max(train_input_relative_residual)),
        ridge=ridge,
        ood_margin=ood_margin,
        input_residual_margin=input_residual_margin,
    )


def calibrate_linear_operator_surrogate(
    surrogate: LinearOperatorSurrogate,
    inputs_holdout,
    outputs_holdout,
    *,
    alpha: float = 0.1,
) -> LinearOperatorCalibrationReport:
    """Score ``surrogate`` against held-out ``(inputs_holdout, outputs_holdout)`` pairs it was **not** fit
    from -- pass snapshots excluded from :func:`fit_linear_operator_surrogate`'s training columns, never
    training pairs themselves (see :func:`mixle_pde.reduced_basis.reduced_basis_error`'s identical
    caution: held-out error is the only honest way to know whether the fitted map generalizes).

    Raises:
        ValueError: mismatched/empty holdout columns, a holdout field whose dimension does not match
            ``surrogate``'s bases, or ``alpha`` outside ``(0, 1)``.
    """
    inputs_holdout = np.asarray(inputs_holdout, dtype=float)
    outputs_holdout = np.asarray(outputs_holdout, dtype=float)
    if inputs_holdout.ndim != 2 or outputs_holdout.ndim != 2:
        raise ValueError(
            f"inputs_holdout and outputs_holdout must both be 2-D (n_dof, n_heldout); got shapes "
            f"{inputs_holdout.shape}, {outputs_holdout.shape}."
        )
    if inputs_holdout.shape[1] != outputs_holdout.shape[1]:
        raise ValueError(
            "inputs_holdout and outputs_holdout must have the same number of paired columns; got "
            f"{inputs_holdout.shape[1]} vs {outputs_holdout.shape[1]}."
        )
    if inputs_holdout.shape[1] < 1:
        raise ValueError("calibrate_linear_operator_surrogate needs at least one held-out pair.")
    if inputs_holdout.shape[0] != surrogate.basis_in.n_dof:
        raise ValueError(
            f"inputs_holdout's leading dimension must be basis_in.n_dof={surrogate.basis_in.n_dof}; "
            f"got shape {inputs_holdout.shape}."
        )
    if outputs_holdout.shape[0] != surrogate.basis_out.n_dof:
        raise ValueError(
            f"outputs_holdout's leading dimension must be basis_out.n_dof={surrogate.basis_out.n_dof}; "
            f"got shape {outputs_holdout.shape}."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")

    prediction = surrogate.predict(inputs_holdout)
    relative_errors = _relative_norms(outputs_holdout - prediction.field, outputs_holdout)
    baseline_relative_errors = _relative_norms(outputs_holdout - surrogate.train_output_mean[:, None], outputs_holdout)

    n = int(inputs_holdout.shape[1])
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    qhat = float(np.quantile(relative_errors, level, method="higher"))
    baseline = float(np.mean(baseline_relative_errors))

    return LinearOperatorCalibrationReport(
        n=n,
        alpha=alpha,
        mean_relative_l2_error=float(np.mean(relative_errors)),
        max_relative_l2_error=float(np.max(relative_errors)),
        qhat_relative_l2_error=qhat,
        baseline_relative_l2_error=baseline,
        imprecise=bool(qhat >= baseline),
        ood_fraction=float(np.mean(~prediction.in_domain)),
    )
