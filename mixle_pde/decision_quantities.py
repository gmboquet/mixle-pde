"""Decision-quantity UQ: calibrated Monte Carlo pushforwards of a posterior (work-plan A5, IC-8 surface).

Every driller-facing number here is a *distribution*, not a point estimate: each function maps a
posterior (:class:`~mixle_pde.latent.PosteriorField3D` or
:class:`~mixle_pde.latent.PosteriorFieldSamples3D`, the concrete artifacts underneath IC-1's shared
``Posterior`` shape) plus a region/threshold/criteria into a
:class:`~mixle_pde.posterior_query.SampledDerivedQuantity` carrying the Monte Carlo draws, a credible
interval, and the ``prior_dominated`` honesty flag from work-plan A2 -- a number a driller could act on
never ships without it.

``region_mass`` is the one linear functional in the set and stays *exact*: it is a thin pass-through to
:func:`mixle_pde.posterior_query.region_mass` (closed-form Gaussian or empirical, whichever the posterior
already supports), with the honesty flag attached. Every other quantity here is nonlinear in the field
(an indicator, a threshold, a predicate), so each is estimated by drawing ``n`` physical-unit posterior
samples (:meth:`PosteriorField3D.sample` / :meth:`PosteriorFieldSamples3D.sample`) and evaluating the
quantity per draw:

* :func:`prob_exceed` -- per draw, the fraction of the region exceeding ``threshold``; the returned
  distribution is that per-draw fraction across draws.
* :func:`tonnage_above_cutoff` -- per draw, ``sum_region volume * field * 1[field > cutoff]``.
* :func:`net_pay` -- per draw, ``sum thickness * 1[saturation >= sat_cut]`` down a 1D ``column_index``.
* :func:`drill_target_prob` -- per draw, the fraction of the region for which a caller-supplied
  ``criteria`` predicate holds (``prob_exceed`` is the ``criteria = lambda s: s > threshold`` special
  case, spelled out separately because it is the common driller question).

``prior_dominated`` is computed from optional ``prior_var`` / ``posterior_var`` per-cell arrays (the
shape work-plan A2's ``informativeness.region_prior_dominated`` takes): when supplied, the
weighted-by-region mean variance reduction ``1 - posterior_var / prior_var`` is compared against a 0.1
threshold, exactly mirroring A2's planned algorithm. A2 has not landed in this branch yet, so the
reduction is computed locally in :func:`_prior_dominated`; once ``mixle_pde.informativeness`` exists this
local copy should be replaced by a direct call so the two never drift apart. When ``prior_var`` /
``posterior_var`` are not supplied, the flag conservatively defaults to ``False`` (unknown, not claimed
to be data-driven) -- the same default A2's own ``DerivedQuantity.prior_dominated`` field uses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mixle_pde import posterior_query as _pq
from mixle_pde.posterior_query import region_mass as _region_mass_exact

__all__ = [
    "DerivedQuantity",
    "SampledDerivedQuantity",
    "region_mass",
    "prob_exceed",
    "tonnage_above_cutoff",
    "net_pay",
    "drill_target_prob",
]


class DerivedQuantity(_pq.DerivedQuantity):
    """``posterior_query.DerivedQuantity`` plus the ``prior_dominated`` flag (work-plan A2).

    ``credible_interval`` takes IC-1's ``level`` (e.g. ``0.9`` for the central 90% interval) rather than
    ``posterior_query``'s ``alpha`` (e.g. ``0.1`` for the same interval) -- IC-8 quantities are typed
    against the frozen ``mixle.reason.posterior_protocol.DerivedQuantity`` shape, which is ``level``, so
    this module is internally consistent with IC-1 even though it is built on ``posterior_query``, whose
    own objects use the opposite convention. Mixing the two conventions on a driller-facing UQ number is
    exactly the kind of wrong-space bug work-plan A6 exists to catch, so it is fixed here rather than
    inherited silently.
    """

    def __init__(self, mean: float, std: float, prior_dominated: bool = False) -> None:
        super().__init__(mean=mean, std=std)
        self.prior_dominated = prior_dominated

    def credible_interval(self, level: float = 0.9) -> tuple[float, float]:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        return super().credible_interval(alpha=1.0 - level)


class SampledDerivedQuantity(_pq.SampledDerivedQuantity):
    """``posterior_query.SampledDerivedQuantity`` plus ``prior_dominated``; see :class:`DerivedQuantity`
    for why ``credible_interval`` takes ``level`` rather than ``alpha``."""

    def __init__(self, samples: np.ndarray, prior_dominated: bool = False) -> None:
        super().__init__(samples=samples)
        self.prior_dominated = prior_dominated

    def credible_interval(self, level: float = 0.9) -> tuple[float, float]:
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        return super().credible_interval(alpha=1.0 - level)


def _rng_or_default(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _weighted_variance_reduction(prior_var: np.ndarray, posterior_var: np.ndarray, weights: np.ndarray) -> float | None:
    """Weighted mean of ``1 - posterior_var / prior_var`` over cells with positive weight and prior_var.

    Returns ``None`` when there is no supported cell to average over (nothing to conclude).
    """
    prior_var = np.asarray(prior_var, dtype=float)
    posterior_var = np.asarray(posterior_var, dtype=float)
    weights = np.asarray(weights, dtype=float)
    support = (weights > 0.0) & (prior_var > 0.0)
    if not np.any(support):
        return None
    reduction = 1.0 - posterior_var[support] / prior_var[support]
    return float(np.average(reduction, weights=weights[support]))


def _prior_dominated(
    prior_var: np.ndarray | None,
    posterior_var: np.ndarray | None,
    weights: np.ndarray,
    *,
    threshold: float = 0.1,
) -> bool:
    """Work-plan A2's honesty flag: True when the region's mean variance reduction is below threshold.

    Self-contained mirror of A2's planned ``informativeness.region_prior_dominated`` -- A2 has not
    landed in this branch, so the formula is duplicated locally rather than imported; swap this body for
    a direct call once ``mixle_pde.informativeness`` exists.
    """
    if prior_var is None or posterior_var is None:
        return False
    reduction = _weighted_variance_reduction(prior_var, posterior_var, weights)
    if reduction is None:
        return False
    return bool(reduction < threshold)


def _finalize(samples: np.ndarray, prior_dominated: bool) -> SampledDerivedQuantity:
    return SampledDerivedQuantity(samples=samples, prior_dominated=prior_dominated)


def region_mass(
    posterior: Any,
    region: np.ndarray,
    cell_volumes: Any,
    *,
    prior_var: np.ndarray | None = None,
    posterior_var: np.ndarray | None = None,
) -> DerivedQuantity | SampledDerivedQuantity:
    """Posterior of ``sum_{i in region} field_i * volume_i`` -- exact via ``posterior_query.region_mass``.

    The one linear functional in this module; no Monte Carlo is used (work-plan A5 non-goal: region_mass
    stays exact). The underlying mean/std (or samples, for an empirical posterior) are computed exactly
    by ``posterior_query.region_mass`` and carried over unchanged; only the carrier type changes, to
    attach the ``prior_dominated`` honesty flag (work-plan A2, from optional ``prior_var`` /
    ``posterior_var`` per-cell arrays over the full grid) and IC-1's ``level``-based credible interval.
    """
    region = np.asarray(region, dtype=bool)
    exact = _region_mass_exact(posterior, region, cell_volumes)
    vols = np.broadcast_to(np.asarray(cell_volumes, dtype=float), region.shape)
    weights = np.where(region, vols, 0.0)
    flag = _prior_dominated(prior_var, posterior_var, weights)
    if isinstance(exact, _pq.SampledDerivedQuantity):
        return SampledDerivedQuantity(samples=exact.samples, prior_dominated=flag)
    return DerivedQuantity(mean=exact.mean, std=exact.std, prior_dominated=flag)


def prob_exceed(
    posterior: Any,
    region: np.ndarray,
    *,
    threshold: float,
    n: int = 4096,
    rng: np.random.Generator | None = None,
    prior_var: np.ndarray | None = None,
    posterior_var: np.ndarray | None = None,
) -> SampledDerivedQuantity:
    """Posterior distribution of the region fraction where ``field > threshold`` (Monte Carlo).

    Draws ``n`` physical-unit posterior samples; per draw computes the fraction of ``region`` cells
    exceeding ``threshold``. The returned quantity's samples are that per-draw fraction, so its mean,
    std, and credible interval describe the uncertainty in "what fraction of this region exceeds
    threshold" -- the common driller question.
    """
    region = np.asarray(region, dtype=bool)
    if not np.any(region):
        raise ValueError("region must select at least one cell.")
    rng = _rng_or_default(rng)
    draws = posterior.sample(int(n), rng)[:, region]
    fraction = np.mean(draws > threshold, axis=1)
    weights = region.astype(float)
    return _finalize(fraction, _prior_dominated(prior_var, posterior_var, weights))


def tonnage_above_cutoff(
    posterior: Any,
    region: np.ndarray,
    cell_volumes: Any,
    *,
    cutoff: float,
    n: int = 4096,
    rng: np.random.Generator | None = None,
    prior_var: np.ndarray | None = None,
    posterior_var: np.ndarray | None = None,
) -> SampledDerivedQuantity:
    """Posterior distribution of ``sum_region volume * field * 1[field > cutoff]`` (Monte Carlo).

    Unlike ``region_mass``, this is nonlinear (the cutoff indicator), so it is estimated by drawing ``n``
    physical-unit posterior samples and summing thresholded mass per draw.
    """
    region = np.asarray(region, dtype=bool)
    if not np.any(region):
        raise ValueError("region must select at least one cell.")
    vols = np.broadcast_to(np.asarray(cell_volumes, dtype=float), region.shape)
    rng = _rng_or_default(rng)
    draws = posterior.sample(int(n), rng)
    region_draws = draws[:, region]
    region_vols = vols[region]
    above = region_draws > cutoff
    tonnage = np.sum(np.where(above, region_draws, 0.0) * region_vols[None, :], axis=1)
    weights = np.where(region, vols, 0.0)
    return _finalize(tonnage, _prior_dominated(prior_var, posterior_var, weights))


def net_pay(
    posterior: Any,
    column_index: np.ndarray,
    *,
    sat_cut: float,
    thickness: np.ndarray,
    n: int = 4096,
    rng: np.random.Generator | None = None,
    prior_var: np.ndarray | None = None,
    posterior_var: np.ndarray | None = None,
) -> SampledDerivedQuantity:
    """Posterior distribution of net pay = thresholded saturation x thickness down a well column.

    ``column_index`` selects the grid cells that make up one 1D column (e.g. a wellbore's depth series);
    it is an index array, not a boolean region mask, because net pay is inherently a per-column quantity
    (deviates from IC-8's illustrative ``region`` parameter name -- see the PR notes). Per draw, sums
    ``thickness`` over cells whose saturation draw is at or above ``sat_cut``.
    """
    column_index = np.asarray(column_index, dtype=int)
    if column_index.size == 0:
        raise ValueError("column_index must select at least one cell.")
    thickness = np.broadcast_to(np.asarray(thickness, dtype=float), column_index.shape)
    rng = _rng_or_default(rng)
    draws = posterior.sample(int(n), rng)[:, column_index]
    above = draws >= sat_cut
    pay = np.sum(np.where(above, thickness[None, :], 0.0), axis=1)
    if prior_var is not None and posterior_var is not None:
        prior_var_col = np.asarray(prior_var, dtype=float)[column_index]
        posterior_var_col = np.asarray(posterior_var, dtype=float)[column_index]
    else:
        prior_var_col, posterior_var_col = prior_var, posterior_var
    return _finalize(pay, _prior_dominated(prior_var_col, posterior_var_col, thickness))


def drill_target_prob(
    posterior: Any,
    region: np.ndarray,
    *,
    criteria: Callable[[np.ndarray], np.ndarray],
    n: int = 4096,
    rng: np.random.Generator | None = None,
    prior_var: np.ndarray | None = None,
    posterior_var: np.ndarray | None = None,
) -> SampledDerivedQuantity:
    """Posterior probability a drill target in ``region`` meets ``criteria`` (Monte Carlo).

    ``criteria`` takes one draw's physical-unit region values (shape ``(|region|,)``, e.g. grade or
    saturation) and returns a boolean array over the same cells indicating which meet the target
    definition (grade/thickness/depth, or any combination). Per draw, the fraction of region cells
    meeting ``criteria`` is computed; ``prob_exceed`` is the ``criteria = lambda s: s > threshold``
    special case of this, spelled out separately since it is the common question.
    """
    region = np.asarray(region, dtype=bool)
    if not np.any(region):
        raise ValueError("region must select at least one cell.")
    rng = _rng_or_default(rng)
    draws = posterior.sample(int(n), rng)[:, region]
    per_draw = np.array([np.mean(np.asarray(criteria(draw), dtype=bool)) for draw in draws], dtype=float)
    weights = region.astype(float)
    return _finalize(per_draw, _prior_dominated(prior_var, posterior_var, weights))
