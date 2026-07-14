"""Data-informativeness diagnostic: does the posterior reflect the data, or just the prior?

A closed-form Gaussian posterior always narrows relative to the prior somewhere, but "narrower" is not
the same as "informed by data" -- a cell far from every sensor, or weakly linked into the forward
operator, can shrink only because the smoothness prior correlates it with better-constrained neighbours.
Reporting a mean/credible-interval for such a cell without saying so is dishonest: the number is mostly
the modeller's regularisation choice, not the survey.

This module answers that question directly, per cell and per region:

* :func:`prior_marginal_variance` -- the marginal variance of the prior ALONE (no data), the yardstick
  everything else is measured against.
* :func:`variance_reduction` -- the fractional shrinkage ``1 - var_post / var_prior`` per cell; near 0
  means the data did essentially nothing there, near 1 means the data dominates.
* :func:`prior_dominated_mask` -- a boolean per-cell flag: reduction below ``threshold`` (default 0.1).
* :func:`region_prior_dominated` -- the region-level verdict a driller-facing decision quantity needs:
  the weighted-mean reduction over a region's support (e.g. `region_mass`'s cell-volume weights) compared
  to ``threshold``.

``posterior_query.derived_quantity``/``region_mass`` thread this through as the ``prior_dominated`` flag
on :class:`~mixle_pde.posterior_query.DerivedQuantity` (work-plan A2) -- a decision quantity is never
handed to a driller without it. This module only reports; it never recalibrates or corrects the posterior
(that is `posterior_calibration`, workstream A3).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def prior_marginal_variance(prior: Any, grid: Any) -> np.ndarray:
    """Marginal variance of the prior alone: ``diag(inv(prior.precision(grid)))``.

    ``prior.precision(grid)`` is the dense ``(n, n)`` prior precision (e.g.
    :meth:`mixle_pde.field_inversion.FieldGaussianPrior.precision`); inverting it once gives the
    no-data covariance, whose diagonal is the per-cell prior variance every posterior is measured
    against.
    """
    precision = np.asarray(prior.precision(grid), dtype=float)
    n = precision.shape[0]
    if precision.ndim != 2 or precision.shape != (n, n):
        raise ValueError("prior.precision(grid) must return a square (n, n) matrix.")
    cov = np.linalg.inv(precision)
    return np.diag(cov).copy()


def variance_reduction(prior_var: np.ndarray, posterior_var: np.ndarray) -> np.ndarray:
    """Per-cell fractional variance reduction: ``1 - posterior_var / prior_var``.

    A cell with ``reduction`` near 0 is essentially unconstrained by the data (the posterior variance
    equals the prior variance); near 1 the data has all but pinned the cell down. Cells with zero prior
    variance (a degenerate, already-certain prior) report a reduction of 1 rather than dividing by zero.
    """
    prior_var = np.asarray(prior_var, dtype=float).reshape(-1)
    posterior_var = np.asarray(posterior_var, dtype=float).reshape(-1)
    if prior_var.shape != posterior_var.shape:
        raise ValueError("prior_var and posterior_var must have the same shape.")
    reduction = np.ones_like(prior_var)
    positive = prior_var > 0.0
    reduction[positive] = 1.0 - posterior_var[positive] / prior_var[positive]
    return reduction


def prior_dominated_mask(prior_var: np.ndarray, posterior_var: np.ndarray, *, threshold: float = 0.1) -> np.ndarray:
    """Boolean per-cell mask: True where the data left the cell's uncertainty near its prior width."""
    return variance_reduction(prior_var, posterior_var) < float(threshold)


def region_prior_dominated(
    prior_var: np.ndarray, posterior_var: np.ndarray, weights: np.ndarray, *, threshold: float = 0.1
) -> bool:
    """True when the weighted-mean variance reduction over a region's support is below ``threshold``.

    ``weights`` is the same weighting a decision quantity uses over the region (e.g. `region_mass`'s
    zero-outside/volume-inside weights); the region's support is where ``weights`` is non-zero. A region
    is "prior dominated" when its data-weighted average informativeness never rose much above the prior
    baseline, i.e. the reported mean/interval is mostly the regulariser's doing.
    """
    reduction = variance_reduction(prior_var, posterior_var)
    w = np.abs(np.asarray(weights, dtype=float).reshape(-1))
    if w.shape != reduction.shape:
        raise ValueError("weights must have the same shape as prior_var/posterior_var.")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must carry positive total mass over the region's support.")
    weighted_mean = float(np.sum(w * reduction) / total)
    return bool(weighted_mean < float(threshold))
