"""Bayesian model selection over competing subsurface hypotheses (workstream C8).

A field inversion always assumes a structural model: a prior (correlation length, smoothness), a
forward operator (which physics, which parameterization), sometimes a whole different geological
story. Two hypotheses can both produce a plausible-looking posterior; deciding between them needs a
number that trades data fit against model complexity -- the Bayesian model evidence ``Z = p(D | model)``,
not just a raw log-likelihood (which always prefers the more flexible model).

:func:`log_evidence_laplace` is the Laplace (saddle-point) approximation to that evidence: the MAP
log-likelihood plus the prior's own log-density at the MAP, minus the automatic Occam complexity
penalty ``1/2 log|H / 2 pi|`` read off the posterior's curvature ``H`` at the MAP point. The log-det
term is read off whichever covariance-storage mode the posterior already uses (dense, sparse-precision
via its cached LU factor, or low-rank + diagonal via the matrix-determinant lemma) -- it never
materializes a fresh ``(n, n)`` array beyond what the posterior itself already stores, so it degrades
gracefully to survey scale even before :mod:`mixle_pde.uq_lowrank` lands a dedicated selected-inversion
log-determinant.

:func:`bayes_factor` and :func:`rank_hypotheses` are the comparison layer: any log-evidence number can
be compared, whether it came from :func:`log_evidence_laplace` (linear-Gaussian / Laplace path) or from
accumulating the sequential ``log_evidence_increment`` fields already reported by
:class:`~mixle_pde.field_assimilation.ParticleAssimilationReport` and
:class:`~mixle_pde.sample_update.SampleUpdateReport` (the SMC/importance-sampling evidence estimate for
a nonlinear or non-Gaussian posterior family) -- both are just a running sum of log-evidence
contributions, so :func:`rank_hypotheses` treats them uniformly via :class:`InversionResult`.

Like :func:`~mixle_pde.posterior_calibration.heldout_observation_check`, :func:`log_evidence_laplace`
reads the posterior's own unconstrained-space Gaussian summary directly, so it is exact for the
identity-transform (``bounds=None``) fields ``linear_gaussian_invert``/``sparse_linear_gaussian_invert``
produce; a bounded field's evidence would need the same physical-space sampling correction workstream A
gave `region_mass`/`derived_quantity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InversionResult:
    """One scored hypothesis: a fitted structural model (prior / forward-operator variant) plus its
    log-evidence, ready for :func:`rank_hypotheses` to compare against its competitors."""

    name: str
    log_evidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _sparse_log_abs_det(matrix: Any) -> float:
    """``log|det(matrix)|`` for a sparse matrix via its LU factor's diagonal (never densifies)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    lu = splu(sp.csc_matrix(matrix))
    diag_u = np.abs(lu.U.diagonal())
    if np.any(diag_u <= 0.0):
        raise ValueError("matrix is singular; cannot take a log-determinant.")
    return float(np.sum(np.log(diag_u)))


def _posterior_log_abs_det_precision(posterior: Any) -> float:
    """``log|H|`` where ``H`` is the posterior precision, read off whichever covariance-storage mode
    ``posterior`` uses -- dense covariance, sparse precision factor, or low-rank + diagonal -- without
    ever materializing a dense ``(n, n)`` array beyond what is already stored."""
    if posterior.dense_cov is not None:
        # Some LAPACK backends (e.g. Apple Accelerate) raise spurious divide-by-zero/overflow
        # RuntimeWarnings from slogdet's internal LU factorization on well-conditioned matrices even
        # though the returned sign/logdet are correct; the warnings are cosmetic, not a correctness
        # signal, so they are suppressed rather than left to alarm callers.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sign, logdet_cov = np.linalg.slogdet(posterior.dense_cov)
        if sign <= 0.0:
            raise ValueError("posterior covariance must be positive definite.")
        return -logdet_cov
    if posterior.precision_factor is not None:
        return _sparse_log_abs_det(posterior.precision_factor.precision)
    if posterior.low_rank is not None:
        low_rank = posterior.low_rank
        diag_var = posterior.diag_var
        k = low_rank.shape[1]
        # matrix-determinant lemma: |diag(d) + U U^T| = |diag(d)| * |I_k + U^T diag(1/d) U|
        inner = np.eye(k) + (low_rank.T / diag_var) @ low_rank
        sign, logdet_inner = np.linalg.slogdet(inner)
        if sign <= 0.0:
            raise ValueError("low-rank posterior covariance must be positive definite.")
        logdet_cov = float(np.sum(np.log(diag_var)) + logdet_inner)
        return -logdet_cov
    diag_var = posterior.diag_var
    return float(-np.sum(np.log(diag_var)))


def log_evidence_laplace(log_likelihood_at_map: float, prior: Any, posterior: Any) -> float:
    """Laplace approximation to the log model evidence ``log Z`` (Bayesian Occam factor).

    ``log Z ~= log_likelihood_at_map + log p(m_MAP) - 1/2 log|H / 2 pi|`` where ``H`` is the posterior
    precision (negative Hessian of the log joint) at the MAP point. The prior's own log-density at the
    MAP supplies the regularization half of the data-fit term; the log-determinant term is the automatic
    complexity penalty -- a model whose posterior is sharply peaked by the data pays a bigger Occam
    penalty than one that stays broad and prior-dominated, so a more flexible model is not automatically
    favored just for fitting the data better at the MAP point.

    ``prior`` must expose ``mean_vector(grid)`` and ``precision_sparse(grid)`` (the
    :class:`~mixle_pde.field_inversion.FieldGaussianPrior` shape); ``posterior`` must expose ``grid``,
    ``map`` (or ``mean``), and one of the :class:`~mixle_pde.latent.PosteriorField3D` covariance-storage
    modes.
    """
    grid = posterior.grid
    map_point = np.asarray(posterior.map if posterior.map is not None else posterior.mean, dtype=float)
    prior_mean = prior.mean_vector(grid)
    prior_precision = prior.precision_sparse(grid)
    diff = map_point - prior_mean
    quad = float(diff @ (prior_precision @ diff))
    log_det_prior_precision = _sparse_log_abs_det(prior_precision)
    d = int(grid.n)
    log_prior_at_map = 0.5 * log_det_prior_precision - 0.5 * quad - 0.5 * d * np.log(2.0 * np.pi)
    log_det_h = _posterior_log_abs_det_precision(posterior)
    occam_penalty = 0.5 * (log_det_h - d * np.log(2.0 * np.pi))
    return float(log_likelihood_at_map) + float(log_prior_at_map) - float(occam_penalty)


def bayes_factor(log_evidence_a: float, log_evidence_b: float) -> float:
    """Bayes factor ``K = Z_a / Z_b``, computed stably in log-space then exponentiated.

    ``K > 1`` favors hypothesis ``a``; ``K < 1`` favors ``b``. Either evidence may come from
    :func:`log_evidence_laplace` or from a summed sequential ``log_evidence_increment``.
    """
    return float(np.exp(float(log_evidence_a) - float(log_evidence_b)))


def rank_hypotheses(hypotheses: list[InversionResult]) -> list[tuple[int, float]]:
    """Rank competing hypotheses by descending log-evidence.

    Returns ``(original_index, log_evidence)`` pairs, best hypothesis first, so a caller can recover
    which input hypothesis each rank corresponds to.
    """
    scored = [(i, float(hypothesis.log_evidence)) for i, hypothesis in enumerate(hypotheses)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
