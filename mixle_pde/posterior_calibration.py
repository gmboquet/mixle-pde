"""Posterior calibration diagnostics for latent field artifacts.

Workstream G requires posterior claims to be measured, not just returned: synthetic-truth coverage,
held-out observation fit, uncertainty inflation away from data, and an explicit insufficient-data
diagnostic. The functions here operate on the shared :class:`mixle_pde.latent.PosteriorField3D`
interface, so the same checks apply to static inversions and to individual 4D assimilation slices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from mixle_pde.latent import PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry, Observation


@dataclass(frozen=True)
class TruthCoverage:
    """Coverage of a known synthetic truth by marginal posterior credible intervals."""

    alpha: float
    expected_coverage: float
    coverage: float
    inside: int
    total: int
    mean_abs_error: float


@dataclass(frozen=True)
class HeldoutFit:
    """Posterior-predictive fit of held-out observations under a linear-Gaussian predictive law."""

    alpha: float
    log_likelihood: float
    rmse: float
    standardized_rmse: float
    coverage: float
    n_observations: int


@dataclass(frozen=True)
class UncertaintyInflation:
    """Whether posterior marginal uncertainty grows away from observed/sensitive cells."""

    near_std: float
    far_std: float
    ratio: float
    near_count: int
    far_count: int


@dataclass(frozen=True)
class IdentifiabilityDiagnostic:
    """Sensitivity and uncertainty summary that flags underdetermined observation configurations."""

    sensitive_fraction: float
    insensitive_mean_std: float
    sensitive_mean_std: float
    insensitive_to_sensitive_std_ratio: float
    insufficient_observations: bool
    reason: str


def truth_coverage(posterior: PosteriorField3D, truth: np.ndarray, *, alpha: float = 0.1) -> TruthCoverage:
    """Measure how often marginal credible intervals cover a synthetic truth field.

    ``truth`` is in the field's physical units. ``PosteriorField3D.credible_interval`` already maps the
    posterior through the field's bound transform, so this works for identity and bounded fields.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    truth = np.asarray(truth, dtype=float)
    if truth.shape != (posterior.grid.n,):
        raise ValueError(f"truth must have shape ({posterior.grid.n},).")
    lower, upper = posterior.credible_interval(alpha=alpha)
    inside_mask = (truth >= lower) & (truth <= upper)
    mean_phys = posterior.grid.from_unconstrained(posterior.mean)
    return TruthCoverage(
        alpha=alpha,
        expected_coverage=1.0 - alpha,
        coverage=float(np.mean(inside_mask)),
        inside=int(np.sum(inside_mask)),
        total=int(truth.size),
        mean_abs_error=float(np.mean(np.abs(mean_phys - truth))),
    )


def heldout_observation_check(
    posterior: PosteriorField3D,
    registry: ForwardOperatorRegistry,
    held_out: list[Observation],
    *,
    alpha: float = 0.1,
) -> HeldoutFit:
    """Score held-out observations against the posterior predictive distribution.

    This diagnostic currently requires fixed-linear forward operators, because it propagates posterior
    covariance as ``J Sigma J.T`` exactly. Nonlinear held-out checks should be added through local
    linearization or sampling when that posterior family is promoted.
    """
    if not held_out:
        raise ValueError("held_out must contain at least one observation.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    cov = posterior.cov if posterior.cov is not None else np.diag(posterior.marginal_variance)
    z = ndtri(1.0 - alpha / 2.0)
    log_likelihood = 0.0
    sq_resid: list[float] = []
    sq_standardized: list[float] = []
    inside = 0
    total = 0
    for obs in held_out:
        op = registry.get(obs.kind)
        if not op.is_linear:
            raise ValueError(f"heldout_observation_check needs fixed-linear operators; {obs.kind!r} is nonlinear.")
        jac = op.local_jacobian(posterior.grid, posterior.mean, obs)
        predicted = jac @ posterior.mean
        predictive_cov = jac @ cov @ jac.T
        noise_cov = obs.noise_cov if not obs.is_diagonal else np.diag(obs.noise_cov)
        total_cov = predictive_cov + noise_cov
        residual = obs.value - predicted
        precision = np.linalg.inv(total_cov)
        sign, logdet = np.linalg.slogdet(total_cov)
        if sign <= 0.0:
            raise ValueError("predictive covariance must be positive definite.")
        log_likelihood += float(-0.5 * (residual @ precision @ residual + logdet + obs.n * np.log(2.0 * np.pi)))
        std = np.sqrt(np.diag(total_cov))
        sq_resid.extend((residual**2).tolist())
        sq_standardized.extend(((residual / std) ** 2).tolist())
        inside += int(np.sum(np.abs(residual) <= z * std))
        total += obs.n
    return HeldoutFit(
        alpha=alpha,
        log_likelihood=log_likelihood,
        rmse=float(np.sqrt(np.mean(sq_resid))),
        standardized_rmse=float(np.sqrt(np.mean(sq_standardized))),
        coverage=float(inside / total),
        n_observations=total,
    )


def observation_sensitivity(
    posterior: PosteriorField3D,
    registry: ForwardOperatorRegistry,
    observations: list[Observation],
    *,
    sensitivity_floor: float = 1.0e-12,
) -> np.ndarray:
    """Aggregate absolute Jacobian sensitivity per field cell for an observation batch."""
    if sensitivity_floor < 0.0:
        raise ValueError("sensitivity_floor must be non-negative.")
    sensitivity = np.zeros(posterior.grid.n, dtype=float)
    field_values = posterior.grid.from_unconstrained(posterior.mean)
    for obs in observations:
        op = registry.get(obs.kind)
        if not op.has_adjoint():
            continue
        jac = op.local_jacobian(posterior.grid, field_values, obs)
        sensitivity += np.sum(np.abs(jac), axis=0)
    sensitivity[sensitivity < sensitivity_floor] = 0.0
    return sensitivity


def uncertainty_inflation(
    posterior: PosteriorField3D,
    *,
    sensitive_mask: np.ndarray,
) -> UncertaintyInflation:
    """Compare marginal posterior uncertainty in sensitive versus insensitive cells."""
    sensitive_mask = np.asarray(sensitive_mask, dtype=bool)
    if sensitive_mask.shape != (posterior.grid.n,):
        raise ValueError(f"sensitive_mask must have shape ({posterior.grid.n},).")
    if np.all(sensitive_mask) or not np.any(sensitive_mask):
        raise ValueError("sensitive_mask must contain both sensitive and insensitive cells.")
    near = posterior.marginal_std[sensitive_mask]
    far = posterior.marginal_std[~sensitive_mask]
    near_std = float(np.mean(near))
    far_std = float(np.mean(far))
    return UncertaintyInflation(
        near_std=near_std,
        far_std=far_std,
        ratio=float(far_std / near_std) if near_std > 0.0 else float("inf"),
        near_count=int(near.size),
        far_count=int(far.size),
    )


def identifiability_diagnostic(
    posterior: PosteriorField3D,
    registry: ForwardOperatorRegistry,
    observations: list[Observation],
    *,
    sensitivity_threshold: float = 1.0e-10,
    min_sensitive_fraction: float = 0.25,
) -> IdentifiabilityDiagnostic:
    """Flag whether the observation batch leaves too much of the field weakly constrained.

    The diagnostic is deliberately conservative: it uses declared operator Jacobians/local Jacobians as
    evidence of direct sensitivity and pairs that with posterior marginal uncertainty. A flagged result
    is not a proof that inference is wrong; it is a provenance-ready warning that the posterior is
    weakly identified by the supplied observations.
    """
    if sensitivity_threshold < 0.0:
        raise ValueError("sensitivity_threshold must be non-negative.")
    if not 0.0 <= min_sensitive_fraction <= 1.0:
        raise ValueError("min_sensitive_fraction must be in [0, 1].")
    sensitivity = observation_sensitivity(posterior, registry, observations, sensitivity_floor=sensitivity_threshold)
    sensitive = sensitivity > sensitivity_threshold
    fraction = float(np.mean(sensitive)) if sensitive.size else 0.0
    if not np.any(sensitive):
        return IdentifiabilityDiagnostic(
            sensitive_fraction=0.0,
            insensitive_mean_std=float(np.mean(posterior.marginal_std)),
            sensitive_mean_std=float("nan"),
            insensitive_to_sensitive_std_ratio=float("inf"),
            insufficient_observations=True,
            reason="no declared observation sensitivity touches the field",
        )
    insensitive_std = float(np.mean(posterior.marginal_std[~sensitive])) if np.any(~sensitive) else 0.0
    sensitive_std = float(np.mean(posterior.marginal_std[sensitive]))
    ratio = float(insensitive_std / sensitive_std) if sensitive_std > 0.0 else float("inf")
    insufficient = fraction < min_sensitive_fraction
    reason = (
        f"sensitive fraction {fraction:.3f} below required {min_sensitive_fraction:.3f}"
        if insufficient
        else "observation sensitivity covers the required fraction"
    )
    return IdentifiabilityDiagnostic(
        sensitive_fraction=fraction,
        insensitive_mean_std=insensitive_std,
        sensitive_mean_std=sensitive_std,
        insensitive_to_sensitive_std_ratio=ratio,
        insufficient_observations=bool(insufficient),
        reason=reason,
    )
