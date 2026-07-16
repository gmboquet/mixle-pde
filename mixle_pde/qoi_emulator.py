"""Probabilistic scalar/quantity-of-interest emulators behind one predictive contract (MP-N4).

Source: notes/mixle-pde-ai-native-multiphysics-work-plan.md workstream N, MP-N4 ("Probabilistic
scalar and quantity-of-interest emulators"). The work-plan's own accept bar: "analytic and nonlinear
benchmarks meet calibration, coverage, held-out error, and out-of-domain rejection thresholds;
deterministic predictors cannot masquerade as posterior uncertainty."

:mod:`mixle_pde.surrogate` (workstream E6, and MP-N4's own "calibrated neural emulator adapters"
member) already gives one calibrated emulator: a neural student with a split-conformal precision floor
and a density-gate OOD check. This module adds the two members that module explicitly does not cover:
:class:`GaussianProcessEmulator` (a closed-form Gaussian-process posterior -- exact predictive mean and
covariance from a squared-exponential kernel) and :class:`BayesianPolynomialEmulator` (a closed-form
Bayesian-ridge posterior over polynomial features -- the "polynomial-chaos/regression" family member).
Both share one :class:`QoIEmulator` predictive contract along with one :func:`calibrate` routine, so a
caller can swap emulator families without touching calibration code -- the "behind one contract"
requirement in MP-N4's own task description.

Every emulator in this module reports a genuine closed-form Bayesian posterior predictive standard
deviation -- never a placeholder, a heuristic residual estimate, or an all-zero array -- and both fit
functions reject a non-positive ``noise_variance``, so :class:`EmulatorPrediction`'s ``std`` is always
strictly positive by construction. This is a direct, structural answer to MP-N4's explicit warning that
"deterministic predictors cannot masquerade as posterior uncertainty": nothing in this module can
report certainty it does not have.

Baseline scope only, matching the "one clean increment" precedent set by this workstream's siblings
(:mod:`mixle_pde.verification.result_queries`'s MP-K2 baseline, :mod:`mixle_pde.design_of_experiments`'s
MP-N2 baseline): a single isotropic RBF kernel (no automatic relevance determination, no
sparse/inducing-point approximation), homoscedastic noise (no heteroscedastic prediction), no
multifidelity input, no gradient/derivative prediction, and no batch-acquisition/active-learning
policy. Tree/ensemble emulator adapters are not covered either. :mod:`mixle_pde.surrogate` is
mentioned only for this docstring's cross-reference -- it is never imported or modified from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.linalg import solve_triangular

__all__ = [
    "EmulatorPrediction",
    "CalibrationReport",
    "QoIEmulator",
    "GaussianProcessEmulator",
    "BayesianPolynomialEmulator",
    "fit_gaussian_process",
    "fit_bayesian_polynomial",
    "calibrate",
]


@dataclass(frozen=True)
class EmulatorPrediction:
    """Posterior predictive summary for a batch of query points, common to every emulator family here.

    ``mean`` and ``std`` are ``(n,)`` arrays -- ``std`` is a genuine posterior predictive standard
    deviation for a *new observation* (latent-function uncertainty plus the fitted observation-noise
    floor), never a placeholder. ``in_domain`` is an ``(n,)`` boolean array flagging whether training
    data actually covers each query point (see :func:`_in_domain_from_standardized_distance` for the
    concrete rule every emulator family in this module uses) versus extrapolation beyond it. A caller
    should treat an ``in_domain=False`` prediction as low-confidence regardless of how small ``std``
    looks in isolation.
    """

    mean: np.ndarray
    std: np.ndarray
    in_domain: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        in_domain = np.asarray(self.in_domain, dtype=bool).reshape(-1)
        if not (mean.shape == std.shape == in_domain.shape):
            raise ValueError(
                f"mean, std, and in_domain must share one shape; got {mean.shape}, {std.shape}, {in_domain.shape}"
            )
        if mean.shape[0] == 0:
            raise ValueError("EmulatorPrediction must cover at least one query point")
        if not np.all(np.isfinite(mean)):
            raise ValueError("EmulatorPrediction.mean must be finite")
        if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
            raise ValueError("EmulatorPrediction.std must be finite and strictly positive everywhere")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "in_domain", in_domain)


@dataclass(frozen=True)
class CalibrationReport:
    """Held-out calibration/coverage summary -- MP-N4's "calibration, coverage, held-out error" accept
    bar, computed generically by :func:`calibrate` for any :class:`QoIEmulator`.

    ``coverage_1sigma``/``coverage_2sigma`` are the observed fraction of held-out points whose
    standardized residual ``(y_true - mean) / std`` falls within 1 (resp. 2) standard deviations -- a
    well-calibrated Gaussian posterior should land near 0.683 / 0.954. ``ood_fraction`` is the fraction
    of held-out points :attr:`EmulatorPrediction.in_domain` flagged as extrapolation.
    """

    n: int
    mae: float
    rmse: float
    coverage_1sigma: float
    coverage_2sigma: float
    mean_standardized_residual: float
    ood_fraction: float

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("CalibrationReport.n must be positive")
        for name in ("coverage_1sigma", "coverage_2sigma", "ood_fraction"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"CalibrationReport.{name} must be a fraction in [0, 1]; got {value!r}")
        for name in ("mae", "rmse", "mean_standardized_residual"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"CalibrationReport.{name} must be finite")


@runtime_checkable
class QoIEmulator(Protocol):
    """The one predictive contract every scalar/QoI emulator family in this module satisfies."""

    def predict(self, x: np.ndarray) -> EmulatorPrediction:
        """Return the posterior predictive mean/std/in-domain flag at each row of ``x``."""
        ...


def _as_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n, d) after atleast_2d; got shape {x.shape}")
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"x has {x.shape[0]} rows but y has {y.shape[0]} values")
    if x.shape[0] < 2:
        raise ValueError("at least two training points are required")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite")
    return x, y


def _as_query(x, *, dim: int) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"query x must have shape (n, {dim}); got {x.shape}")
    return x


def _in_domain_from_standardized_distance(
    xq_std: np.ndarray, x_train_std: np.ndarray, *, margin: float
) -> np.ndarray:
    """A query point (already in standardized-x coordinates) counts as in-domain when every coordinate
    falls within the training data's own per-dimension range, expanded by ``margin`` standardized
    units on each side. Purely geometric -- "is this near any training coordinate" -- and shared
    verbatim by every emulator family in this module.

    A variance-shrinkage-ratio rule (posterior functional variance vs. prior functional variance at the
    query point) was tried first and rejected: it works for the Gaussian process's bounded, stationary
    kernel (prior variance is the same everywhere, so shrinkage cleanly falls to zero far from data),
    but does not generalize to the polynomial family's unbounded feature basis -- a single high-order
    monomial coefficient can stay tightly constrained by the posterior arbitrarily far along its own
    axis, so the shrinkage ratio does not reliably return to "no information" as distance grows, and a
    wildly extrapolated point was observed to be misreported as in-domain. A direct coordinate-envelope
    check has no such family-specific failure mode, so both families use it instead.
    """
    lo = x_train_std.min(axis=0) - margin
    hi = x_train_std.max(axis=0) + margin
    return np.all((xq_std >= lo) & (xq_std <= hi), axis=1)


# ---------------------------------------------------------------------------
# Gaussian-process emulator
# ---------------------------------------------------------------------------


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, *, lengthscale: float, signal_variance: float) -> np.ndarray:
    sq1 = np.sum(x1 * x1, axis=1)
    sq2 = np.sum(x2 * x2, axis=1)
    sq_dists = np.maximum(sq1[:, None] + sq2[None, :] - 2.0 * x1 @ x2.T, 0.0)
    return signal_variance * np.exp(-0.5 * sq_dists / (lengthscale**2))


def _log_marginal_likelihood(cholesky_lower: np.ndarray, alpha: np.ndarray, y: np.ndarray) -> float:
    n = y.shape[0]
    fit_term = -0.5 * float(y @ alpha)
    complexity_term = -float(np.sum(np.log(np.diag(cholesky_lower))))
    normalizer = -0.5 * n * np.log(2.0 * np.pi)
    return fit_term + complexity_term + normalizer


@dataclass(frozen=True)
class GaussianProcessEmulator:
    """A fitted squared-exponential Gaussian-process posterior over a scalar QoI.

    Build with :func:`fit_gaussian_process`; every field here derives from the training data and the
    fitted lengthscale together, so construct through that function rather than directly.
    """

    x_train: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    lengthscale: float
    signal_variance: float
    noise_variance: float
    alpha: np.ndarray  # (K + noise*I)^-1 (standardized y), shape (n_train,)
    cholesky_lower: np.ndarray  # lower Cholesky factor of (K + noise*I), shape (n_train, n_train)
    ood_margin: float = 0.25

    def __post_init__(self) -> None:
        n = self.x_train.shape[0]
        if n < 2:
            raise ValueError("GaussianProcessEmulator requires at least two training points")
        if self.alpha.shape != (n,) or self.cholesky_lower.shape != (n, n):
            raise ValueError("GaussianProcessEmulator's fitted arrays are inconsistent with x_train")
        if self.lengthscale <= 0.0 or self.signal_variance <= 0.0 or self.noise_variance <= 0.0:
            raise ValueError("lengthscale, signal_variance, and noise_variance must all be positive")
        if self.ood_margin < 0.0:
            raise ValueError("ood_margin must be non-negative")

    def predict(self, x: np.ndarray) -> EmulatorPrediction:
        xq = _as_query(x, dim=self.x_train.shape[1])
        xq_std = (xq - self.x_mean) / self.x_scale
        xt_std = (self.x_train - self.x_mean) / self.x_scale

        k_star = _rbf_kernel(xq_std, xt_std, lengthscale=self.lengthscale, signal_variance=self.signal_variance)
        mean_std = k_star @ self.alpha

        v = solve_triangular(self.cholesky_lower, k_star.T, lower=True)  # (n_train, n_query)
        latent_var = np.maximum(self.signal_variance - np.sum(v * v, axis=0), 0.0)
        predictive_var = latent_var + self.noise_variance

        mean = mean_std * self.y_scale + self.y_mean
        std = np.sqrt(predictive_var) * self.y_scale

        in_domain = _in_domain_from_standardized_distance(xq_std, xt_std, margin=self.ood_margin)
        return EmulatorPrediction(mean=mean, std=std, in_domain=in_domain)


def fit_gaussian_process(
    x,
    y,
    *,
    lengthscale: float | None = None,
    lengthscale_candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    signal_variance: float = 1.0,
    noise_variance: float = 1e-3,
    ood_margin: float = 0.25,
) -> GaussianProcessEmulator:
    """Fit a squared-exponential :class:`GaussianProcessEmulator` to ``(x, y)`` by exact closed-form
    GP regression (Cholesky solve; Rasmussen & Williams Algorithm 2.1 -- no iterative optimizer).

    Inputs are standardized internally (``x`` per-dimension zero-mean/unit-std, ``y`` zero-mean/unit-
    variance) before the kernel is evaluated -- ``signal_variance``/``noise_variance`` are therefore in
    *standardized-y* units, and ``lengthscale``/``lengthscale_candidates`` are in *standardized-x*
    units. If ``lengthscale`` is ``None`` (the default), every value in ``lengthscale_candidates`` is
    scored by log marginal likelihood at the fixed ``signal_variance``/``noise_variance`` and the best
    is kept -- a small deterministic grid search, not a general hyperparameter optimizer.

    Raises:
        ValueError: non-finite/mismatched ``x``/``y``, fewer than two training points, a non-positive
            lengthscale candidate, or a non-positive ``signal_variance``/``noise_variance`` (the latter
            would let a query point's reported ``std`` reach zero, which :class:`EmulatorPrediction`
            forbids).
    """
    x, y = _as_xy(x, y)
    if signal_variance <= 0.0:
        raise ValueError("signal_variance must be positive")
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive")
    candidates = (lengthscale,) if lengthscale is not None else tuple(lengthscale_candidates)
    if not candidates:
        raise ValueError("lengthscale_candidates must be non-empty when lengthscale is None")

    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    x_std = (x - x_mean) / x_scale

    y_mean = float(y.mean())
    y_scale = float(y.std())
    y_scale = y_scale if y_scale > 1e-12 else 1.0
    y_std = (y - y_mean) / y_scale

    n = x_std.shape[0]
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for candidate in candidates:
        if candidate <= 0.0:
            raise ValueError(f"lengthscale candidates must be positive; got {candidate!r}")
        k = _rbf_kernel(x_std, x_std, lengthscale=candidate, signal_variance=signal_variance)
        k[np.diag_indices(n)] += noise_variance
        cholesky_lower = np.linalg.cholesky(k)
        w = solve_triangular(cholesky_lower, y_std, lower=True)
        alpha = solve_triangular(cholesky_lower.T, w, lower=False)
        lml = _log_marginal_likelihood(cholesky_lower, alpha, y_std)
        if best is None or lml > best[0]:
            best = (lml, candidate, cholesky_lower, alpha)

    assert best is not None  # candidates is non-empty, so the loop above ran at least once
    _, best_lengthscale, cholesky_lower, alpha = best
    return GaussianProcessEmulator(
        x_train=x,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        lengthscale=best_lengthscale,
        signal_variance=signal_variance,
        noise_variance=noise_variance,
        alpha=alpha,
        cholesky_lower=cholesky_lower,
        ood_margin=ood_margin,
    )


# ---------------------------------------------------------------------------
# Bayesian polynomial-regression emulator ("polynomial-chaos/regression" family member)
# ---------------------------------------------------------------------------


def _polynomial_exponents(dim: int, degree: int) -> list[tuple[int, ...]]:
    """Every exponent tuple of total degree <= ``degree`` over ``dim`` input dimensions, including the
    constant (all-zero) term -- a plain total-degree polynomial basis, not a full orthogonal-chaos
    (Hermite/Legendre) basis; see the module docstring's scope note. The stars-and-bars bijection
    between size-``total`` combinations-with-replacement of ``range(dim)`` and exponent vectors summing
    to ``total`` means this already produces each exponent tuple exactly once, in a stable order.
    """
    exponents: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        for combo in combinations_with_replacement(range(dim), total):
            exponent = [0] * dim
            for idx in combo:
                exponent[idx] += 1
            exponents.append(tuple(exponent))
    return exponents


def _polynomial_features(x: np.ndarray, exponents: list[tuple[int, ...]]) -> np.ndarray:
    return np.stack([np.prod(x ** np.array(exponent), axis=1) for exponent in exponents], axis=1)


@dataclass(frozen=True)
class BayesianPolynomialEmulator:
    """A fitted Bayesian-ridge posterior over a total-degree polynomial feature basis.

    Build with :func:`fit_bayesian_polynomial`; every field here derives from the training data and
    the declared degree together, so construct through that function rather than directly.
    """

    exponents: tuple[tuple[int, ...], ...]
    x_train: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    posterior_mean: np.ndarray  # (n_features,)
    posterior_cov: np.ndarray  # (n_features, n_features)
    prior_variance: float
    noise_variance: float
    ood_margin: float = 0.25

    def __post_init__(self) -> None:
        n_features = len(self.exponents)
        if n_features == 0:
            raise ValueError("BayesianPolynomialEmulator requires at least one polynomial feature")
        if self.posterior_mean.shape != (n_features,) or self.posterior_cov.shape != (n_features, n_features):
            raise ValueError("BayesianPolynomialEmulator's fitted arrays are inconsistent with its exponents")
        if self.x_train.ndim != 2 or self.x_train.shape[1] != self.x_mean.shape[0]:
            raise ValueError("BayesianPolynomialEmulator.x_train must have shape (n_train, dim) matching x_mean")
        if self.x_train.shape[0] < n_features:
            raise ValueError("BayesianPolynomialEmulator requires at least as many training points as features")
        if self.prior_variance <= 0.0 or self.noise_variance <= 0.0:
            raise ValueError("prior_variance and noise_variance must both be positive")
        if self.ood_margin < 0.0:
            raise ValueError("ood_margin must be non-negative")

    def predict(self, x: np.ndarray) -> EmulatorPrediction:
        xq = _as_query(x, dim=self.x_mean.shape[0])
        xq_std = (xq - self.x_mean) / self.x_scale
        xt_std = (self.x_train - self.x_mean) / self.x_scale
        phi = _polynomial_features(xq_std, list(self.exponents))

        mean_std = phi @ self.posterior_mean
        functional_var = np.maximum(np.einsum("ij,jk,ik->i", phi, self.posterior_cov, phi), 0.0)
        predictive_var = functional_var + self.noise_variance

        mean = mean_std * self.y_scale + self.y_mean
        std = np.sqrt(predictive_var) * self.y_scale

        in_domain = _in_domain_from_standardized_distance(xq_std, xt_std, margin=self.ood_margin)
        return EmulatorPrediction(mean=mean, std=std, in_domain=in_domain)


def fit_bayesian_polynomial(
    x,
    y,
    *,
    degree: int = 2,
    prior_variance: float = 10.0,
    noise_variance: float = 1e-2,
    ood_margin: float = 0.25,
) -> BayesianPolynomialEmulator:
    """Fit a :class:`BayesianPolynomialEmulator` -- Bayesian ridge regression (Gaussian prior
    ``N(0, prior_variance * I)`` on the weights, Gaussian noise ``N(0, noise_variance)``) over a
    total-degree-``degree`` polynomial feature basis -- to ``(x, y)`` in closed form.

    ``x`` is standardized per-dimension and ``y`` is standardized before the basis is built, exactly as
    in :func:`fit_gaussian_process`; ``prior_variance``/``noise_variance`` are therefore in
    standardized-``y`` units.

    Raises:
        ValueError: non-finite/mismatched ``x``/``y``, fewer training points than polynomial features,
            or a non-positive ``degree``/``prior_variance``/``noise_variance``.
    """
    x, y = _as_xy(x, y)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if prior_variance <= 0.0:
        raise ValueError("prior_variance must be positive")
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive")

    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    x_std = (x - x_mean) / x_scale

    y_mean = float(y.mean())
    y_scale = float(y.std())
    y_scale = y_scale if y_scale > 1e-12 else 1.0
    y_std = (y - y_mean) / y_scale

    exponents = _polynomial_exponents(x.shape[1], degree)
    if x.shape[0] < len(exponents):
        raise ValueError(
            f"need at least {len(exponents)} training points for a degree-{degree} basis "
            f"over {x.shape[1]} dimensions; got {x.shape[0]}"
        )
    phi = _polynomial_features(x_std, exponents)

    precision = (phi.T @ phi) / noise_variance + np.eye(len(exponents)) / prior_variance
    posterior_cov = np.linalg.inv(precision)
    posterior_mean = posterior_cov @ (phi.T @ y_std) / noise_variance

    return BayesianPolynomialEmulator(
        exponents=tuple(exponents),
        x_train=x,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        prior_variance=prior_variance,
        noise_variance=noise_variance,
        ood_margin=ood_margin,
    )


# ---------------------------------------------------------------------------
# Shared calibration routine -- MP-N4's "calibration, coverage, held-out error" bar
# ---------------------------------------------------------------------------


def calibrate(emulator: QoIEmulator, x_holdout, y_holdout) -> CalibrationReport:
    """Score any :class:`QoIEmulator` against held-out ``(x_holdout, y_holdout)`` -- the same routine
    for a :class:`GaussianProcessEmulator`, a :class:`BayesianPolynomialEmulator`, or any future
    ``QoIEmulator`` member, satisfying MP-N4's "behind one contract" requirement for calibration too.
    """
    y = np.asarray(y_holdout, dtype=np.float64).reshape(-1)
    if y.shape[0] == 0:
        raise ValueError("x_holdout/y_holdout must contain at least one point")
    prediction = emulator.predict(x_holdout)
    if prediction.mean.shape[0] != y.shape[0]:
        raise ValueError("emulator.predict returned a different point count than y_holdout")

    residual = y - prediction.mean
    z = residual / prediction.std
    return CalibrationReport(
        n=int(y.shape[0]),
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(residual**2))),
        coverage_1sigma=float(np.mean(np.abs(z) <= 1.0)),
        coverage_2sigma=float(np.mean(np.abs(z) <= 2.0)),
        mean_standardized_residual=float(np.mean(z)),
        ood_fraction=float(np.mean(~prediction.in_domain)),
    )
