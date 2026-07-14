"""G9 -- environmental constraints feed production + reclamation planning (work-plan workstream G).

Turns the shared subsurface posterior (IC-1) into the two things a mine-planning optimizer needs in
order to respect environmental limits: which blocks are off-limits (a no-mine/buffer mask) and how much
reclaiming a disturbed block costs (an additive liability term). Both surfaces are derived through
IC-8's calibrated decision-quantity API (:mod:`mixle_pde.decision_quantities`) so a masked-out block
carries the same honesty guarantee (samples + credible interval + ``prior_dominated``) as every other
driller-facing number in the plan. The mask and the cost surface are then folded into an IC-9-shaped
network payload -- forbidden nodes / zero-capacity arcs, and an added cost term -- for H1's deterministic
``min_cost_flow``/``network_design`` (or H4's stochastic optimizer) to consume. Per the work-plan
non-goal, this module never imports or calls the network-flow solver itself (no dependency on
``mixle.relations``, and no edit to it); it only produces the payload the solver reads.

G1 (groundwater seepage transport) and G4 (InSAR subsidence inversion) have not landed on this branch
yet, and H1 (the IC-9 ``min_cost_flow``/``network_design`` fill-in) has not landed either -- the
work-plan's own algorithm text anticipates this ("reuse G4 subsidence + G1/G5 seepage where available").
Consequently:

* :func:`exclusion_mask` does not import G1/G4; it works directly off any IC-1-shaped posterior whose
  draws stack a seepage-risk field and a subsidence field along one flat block axis (see the function
  docstring for the exact layout). :class:`BlockFieldPosterior` is a small Gaussian stand-in for that
  shape (mirrors G6's ``GaussianSourcePosterior`` convenience) -- once G1/G4 land, their forward models
  only need to expose the same duck-typed surface (``sample``/``mean``/``cov``/``credible_interval``),
  there is no interface to migrate.
* :func:`apply_env_constraints` takes its own ``network`` payload shape (a plain mapping of
  ``cap``/``cost``/``supply`` node matrices, i.e. exactly ``min_cost_flow``'s frozen ``(cap, cost,
  supply)`` triple, plus an optional ``block_nodes`` index) rather than literally
  ``network_design``'s ``(nodes, arcs, fixed_costs, demands)`` positional shape -- IC-9's frozen stub
  for ``network_design`` does not expose an explicit per-arc cap/cost in its 4-argument signature, and
  with H1 not yet merged there is nothing in this branch that pins down how ``arcs`` internally encodes
  cap/cost. The returned dict still carries ``forbidden_nodes`` (and passes through any
  ``nodes``/``arcs``/``fixed_costs``/``demands`` the caller supplied) so an H1/H4 caller assembling a
  fixed-charge ``network_design`` model can exclude the same nodes from its own arc set.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import norm

from mixle_pde import decision_quantities as _dq

__all__ = [
    "exclusion_mask",
    "reclamation_cost",
    "apply_env_constraints",
    "BlockFieldPosterior",
]


class _SimpleDerivedQuantity:
    """Minimal IC-1 ``DerivedQuantity``: samples + credible interval + the honesty flag."""

    def __init__(self, samples: Any) -> None:
        self.samples = np.asarray(samples)
        self.prior_dominated = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)


class BlockFieldPosterior:
    """A diagonal-Gaussian `Posterior` (IC-1) over a stacked ``(seepage_field, subsidence_field)``
    vector, one entry per block per field.

    Stand-in for G1's seepage-transport / G4's subsidence-inversion output until they land on this
    branch (mirrors G6's ``GaussianSourcePosterior`` in ``monitoring_design.py``): any object exposing
    ``sample(n, rng)``/``mean``/``cov``/``credible_interval(level)`` -- including G1's/G4's eventual
    posteriors -- works as :func:`exclusion_mask`'s ``posterior`` argument unchanged, so there is no
    interface to migrate once they exist. Exposes both the singular ``sample`` (what
    ``decision_quantities.prob_exceed`` calls against the concrete ``PosteriorField3D``/
    ``PosteriorFieldSamples3D`` posteriors on this branch today) and the plural ``samples`` alias IC-1
    itself specifies, so it satisfies both call sites.
    """

    def __init__(
        self,
        seepage_mean: Any,
        seepage_std: Any,
        subsidence_mean: Any,
        subsidence_std: Any,
    ) -> None:
        seepage_mean = np.atleast_1d(np.asarray(seepage_mean, dtype=float))
        subsidence_mean = np.atleast_1d(np.asarray(subsidence_mean, dtype=float))
        if seepage_mean.shape != subsidence_mean.shape:
            raise ValueError("seepage_mean and subsidence_mean must have the same block count.")
        seepage_std = np.broadcast_to(np.asarray(seepage_std, dtype=float), seepage_mean.shape)
        subsidence_std = np.broadcast_to(np.asarray(subsidence_std, dtype=float), subsidence_mean.shape)
        self.n_blocks = int(seepage_mean.shape[0])
        self._mean = np.concatenate([seepage_mean, subsidence_mean])
        self._std = np.concatenate([seepage_std, subsidence_std])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((int(n), self._mean.shape[0]))
        return self._mean[None, :] + self._std[None, :] * z

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """IC-1's plural alias for :meth:`sample`."""
        return self.sample(n, rng)

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        return np.diag(self._std**2)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        z = float(norm.ppf(0.5 + level / 2.0))
        return self._mean - z * self._std, self._mean + z * self._std

    def derived_quantity(self, fn: Any, n: int, rng: np.random.Generator) -> _SimpleDerivedQuantity:
        return _SimpleDerivedQuantity(fn(self.sample(n, rng)))


def _block_regions(n_blocks: int, *, offset: int, total_dim: int) -> list[np.ndarray]:
    """One one-hot region mask per block, selecting position ``offset + i`` of a ``total_dim`` field --
    used to ask IC-8's ``prob_exceed`` a single-cell (i.e. exact-probability) question per block."""
    regions = []
    for i in range(n_blocks):
        region = np.zeros(total_dim, dtype=bool)
        region[offset + i] = True
        regions.append(region)
    return regions


def exclusion_mask(
    posterior: Any,
    *,
    seepage_prob_cut: float,
    subsidence_cut: float,
    n_blocks: int | None = None,
    seepage_threshold: float = 0.0,
    subsidence_threshold: float = 0.0,
    n: int = 2048,
    rng: np.random.Generator | None = None,
    prior_var: np.ndarray | None = None,
    posterior_var: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean no-mine/buffer mask over blocks, from IC-8 ``prob_exceed`` surfaces (work-plan algorithm
    steps 1-2).

    ``posterior`` draws a flat ``(2 * n_blocks,)`` vector per sample: the first ``n_blocks`` entries are
    a seepage-risk field (e.g. a Darcy-flux or plume-concentration proxy at each block -- G1's eventual
    output), the next ``n_blocks`` a subsidence-displacement field (G4's eventual output).
    :class:`BlockFieldPosterior` builds exactly this layout. ``n_blocks`` defaults to half the
    posterior's dimensionality when omitted.

    For each block, two IC-8 :func:`~mixle_pde.decision_quantities.prob_exceed` decision quantities are
    computed on a one-cell region: the posterior probability the seepage field there exceeds
    ``seepage_threshold``, and the posterior probability the subsidence field there exceeds
    ``subsidence_threshold``. A block is excluded when either probability clears its cutoff
    (``seepage_prob_cut`` / ``subsidence_cut``) -- or when ``prob_exceed`` reports ``prior_dominated``
    for that block, i.e. there is not yet enough data to clear it (the work-plan algorithm's "excluded
    conservatively"). ``prior_var``/``posterior_var``, when supplied, are the full ``(2 * n_blocks,)``
    per-cell variance arrays ``prob_exceed`` needs to compute that flag; without them the flag defaults
    to ``False`` for every block (IC-8's own conservative default when informativeness is unknown), so
    only the direct exceedance-probability path drives exclusion.
    """
    rng = rng if rng is not None else np.random.default_rng()
    total_dim = int(np.asarray(posterior.mean).shape[0])
    resolved_n_blocks = int(n_blocks) if n_blocks is not None else total_dim // 2
    if resolved_n_blocks <= 0 or 2 * resolved_n_blocks > total_dim:
        raise ValueError("n_blocks must be positive and fit twice within the posterior's dimensionality.")

    seepage_regions = _block_regions(resolved_n_blocks, offset=0, total_dim=total_dim)
    subsidence_regions = _block_regions(resolved_n_blocks, offset=resolved_n_blocks, total_dim=total_dim)

    mask = np.zeros(resolved_n_blocks, dtype=bool)
    for i in range(resolved_n_blocks):
        seep_dq = _dq.prob_exceed(
            posterior,
            seepage_regions[i],
            threshold=seepage_threshold,
            n=n,
            rng=rng,
            prior_var=prior_var,
            posterior_var=posterior_var,
        )
        sub_dq = _dq.prob_exceed(
            posterior,
            subsidence_regions[i],
            threshold=subsidence_threshold,
            n=n,
            rng=rng,
            prior_var=prior_var,
            posterior_var=posterior_var,
        )
        seep_p = float(np.mean(seep_dq.samples))
        sub_p = float(np.mean(sub_dq.samples))
        mask[i] = (
            seep_p > seepage_prob_cut
            or sub_p > subsidence_cut
            or bool(seep_dq.prior_dominated)
            or bool(sub_dq.prior_dominated)
        )
    return mask


def _block_categories(block_model: Any, n_blocks: int) -> np.ndarray:
    """Per-block category/lithology labels, looked up on ``block_model`` (a mapping with a
    ``"category"`` key, an object with a ``category`` attribute, or ``None`` for a single uniform
    category)."""
    if block_model is None:
        categories = None
    elif isinstance(block_model, Mapping):
        categories = block_model.get("category")
    else:
        categories = getattr(block_model, "category", None)
    if categories is None:
        return np.full(n_blocks, "default", dtype=object)
    categories = np.asarray(categories, dtype=object)
    if categories.shape[0] != n_blocks:
        raise ValueError("block_model's category count must match disturbance's block count.")
    return categories


def reclamation_cost(
    block_model: Any,
    disturbance: Any,
    *,
    unit_costs: Mapping[str, float] | float,
) -> np.ndarray:
    """Per-block reclamation liability = disturbance x unit cost (work-plan algorithm step 3).

    ``disturbance`` is a length-``n_blocks`` array of disturbed area/volume per block (e.g. surface
    footprint if the block is mined/stripped; zero for an untouched block). ``unit_costs`` is either a
    single reclamation rate applied uniformly, or a mapping from a per-block category/lithology label
    (looked up on ``block_model``: a mapping with a ``"category"`` key, an object with a ``category``
    attribute, or an array of labels the same length as ``disturbance``) to its own per-unit rate --
    so, e.g., a tailings block and a waste-rock block can carry different reclamation costs. The result
    is the additive liability term :func:`apply_env_constraints` folds into the extraction/arc cost
    vector ``min_cost_flow``/``network_design`` optimize over.
    """
    disturbance = np.atleast_1d(np.asarray(disturbance, dtype=float))
    if isinstance(unit_costs, Mapping):
        categories = _block_categories(block_model, disturbance.shape[0])
        try:
            rates = np.array([float(unit_costs[c]) for c in categories], dtype=float)
        except KeyError as exc:
            raise KeyError(f"unit_costs has no rate for block category {exc.args[0]!r}.") from exc
    else:
        rates = np.full(disturbance.shape[0], float(unit_costs), dtype=float)
    return disturbance * rates


def apply_env_constraints(
    network: Mapping[str, Any],
    exclusion_mask: np.ndarray,
    reclamation_cost: np.ndarray,
) -> dict[str, Any]:
    """Fold G9's mask + liability surface into an IC-9-shaped network payload (work-plan algorithm
    step 4); H1/H4 read this, this module never calls the solver (work-plan non-goal).

    ``network`` is a plain mapping describing the reference network over blocks (and, optionally, extra
    non-block nodes such as a plant or market sink): ``"cap"``/``"cost"`` are ``(n, n)`` arc matrices in
    exactly :func:`mixle.relations.min_cost_flow`'s frozen ``(cap, cost, supply)`` shape, ``"supply"``
    is the optional length-``n`` node supply vector, and ``"block_nodes"`` (optional, defaults to
    ``arange(len(exclusion_mask))``) maps each block index to its node index in that network.

    Excluded blocks become forbidden nodes: every arc touching one has its capacity zeroed (so no flow
    can originate from, or land back on, a no-mine block), and, if that node carried supply, the supply
    is zeroed too (there is nothing left to extract). Every remaining block's reclamation liability is
    added to the cost of its outgoing (extraction) arcs, so a block that is cheap to mine but expensive
    to reclaim no longer looks artificially attractive to the optimizer. Any ``nodes``/``arcs``/
    ``fixed_costs``/``demands`` the caller supplied (the shape ``network_design`` itself takes) are
    passed through unchanged alongside ``forbidden_nodes``, for a fixed-charge caller to exclude the
    same nodes from its own arc set.
    """
    exclusion_mask = np.asarray(exclusion_mask, dtype=bool)
    reclamation_cost = np.atleast_1d(np.asarray(reclamation_cost, dtype=float))
    if reclamation_cost.shape[0] != exclusion_mask.shape[0]:
        raise ValueError("exclusion_mask and reclamation_cost must have the same block count.")

    cap = np.array(network["cap"], dtype=float, copy=True)
    cost = np.array(network["cost"], dtype=float, copy=True)
    if cap.shape != cost.shape or cap.ndim != 2 or cap.shape[0] != cap.shape[1]:
        raise ValueError("network['cap'] and network['cost'] must be equal-shape square (n, n) arrays.")

    block_nodes = np.asarray(network.get("block_nodes", np.arange(exclusion_mask.shape[0])), dtype=int)
    if block_nodes.shape[0] != exclusion_mask.shape[0]:
        raise ValueError("network['block_nodes'] must have one entry per block.")

    forbidden_nodes = sorted(int(node) for node, excluded in zip(block_nodes, exclusion_mask) if excluded)
    for node in forbidden_nodes:
        cap[node, :] = 0.0
        cap[:, node] = 0.0

    for node, liability in zip(block_nodes, reclamation_cost):
        outgoing = cap[node, :] > 0.0
        cost[node, outgoing] = cost[node, outgoing] + liability

    result: dict[str, Any] = {
        "cap": cap,
        "cost": cost,
        "forbidden_nodes": forbidden_nodes,
        "reclamation_cost": reclamation_cost,
    }
    if "supply" in network:
        supply = np.array(network["supply"], dtype=float, copy=True)
        if forbidden_nodes:
            supply[forbidden_nodes] = 0.0
        result["supply"] = supply
    for passthrough in ("nodes", "arcs", "fixed_costs", "demands"):
        if passthrough in network:
            result[passthrough] = network[passthrough]

    return result
