"""G9 DoD: a seepage-risk exclusion mask forces a costlier-but-feasible re-optimized flow plan.

Builds a synthetic seepage-risk "polygon" (two blocks whose seepage-risk field posterior is
concentrated well above a regulatory threshold, versus four clean blocks), derives an
`exclusion_mask` for it, prices a per-block `reclamation_cost` liability, and folds both into a
small min-cost-flow reference network (blocks -> a single processing sink) via
`apply_env_constraints`. The unconstrained network routes the cheapest way straight through the
risky blocks; feeding the constrained payload to a reference `min_cost_flow` solver must remove
those blocks from the solution entirely and land on a strictly higher-cost (but still feasible)
plan than the unconstrained one -- exactly the "removes the flagged blocks, forces a
higher-cost feasible replan" property the work-plan's algorithm exists to guarantee.

`mixle.relations.min_cost_flow` (workstream H1, IC-9) has not landed on this branch yet, so this
test carries a small self-contained reference min-cost-flow solver (`_reference_min_cost_flow`,
an LP over per-arc flow variables) that implements the exact same `(cap, cost, supply) -> (value,
flow)` contract; the moment `mixle.relations.min_cost_flow` exists, `_solve_min_cost_flow` below
picks the real implementation up automatically and this reference path is never used.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from mixle_pde.reclamation import (
    BlockFieldPosterior,
    apply_env_constraints,
    exclusion_mask,
    reclamation_cost,
)

try:  # pragma: no cover - exercised automatically once H1 lands
    from mixle.relations import min_cost_flow as _core_min_cost_flow
except ImportError:  # pragma: no cover
    _core_min_cost_flow = None


def _reference_min_cost_flow(cap: np.ndarray, cost: np.ndarray, supply: np.ndarray) -> SimpleNamespace:
    """Self-contained reference implementation of IC-9's ``min_cost_flow(cap, cost, supply)``
    contract (successive-shortest-path in the real H1 fill-in; a plain LP here, same answer)."""
    cap = np.asarray(cap, dtype=float)
    cost = np.asarray(cost, dtype=float)
    supply = np.asarray(supply, dtype=float)
    n = cap.shape[0]
    arcs = [(u, v) for u in range(n) for v in range(n) if cap[u, v] > 0.0]
    if not arcs:
        return SimpleNamespace(value=0.0, flow=np.zeros((n, n)))
    c = np.array([cost[u, v] for u, v in arcs])
    bounds = [(0.0, cap[u, v]) for u, v in arcs]
    a_eq = np.zeros((n, len(arcs)))
    for k, (u, v) in enumerate(arcs):
        a_eq[u, k] += 1.0
        a_eq[v, k] -= 1.0
    res = linprog(c, A_eq=a_eq, b_eq=supply, bounds=bounds, method="highs")
    if not res.success:
        raise ValueError(f"reference min-cost flow infeasible: {res.message}")
    flow = np.zeros((n, n))
    for k, (u, v) in enumerate(arcs):
        flow[u, v] = res.x[k]
    return SimpleNamespace(value=float(res.fun), flow=flow)


def _solve_min_cost_flow(cap: np.ndarray, cost: np.ndarray, supply: np.ndarray) -> SimpleNamespace:
    if _core_min_cost_flow is not None:
        try:
            return _core_min_cost_flow(cap, cost, supply)
        except NotImplementedError:
            pass
    return _reference_min_cost_flow(cap, cost, supply)


N_BLOCKS = 6
RISKY_BLOCKS = (0, 1)  # the synthetic seepage-risk polygon
SAFE_BLOCKS = (2, 3, 4, 5)


def _risk_posterior() -> BlockFieldPosterior:
    # Seepage field: risky blocks sit far above the regulatory threshold (5.0), safe blocks far below.
    seepage_mean = np.array([8.0, 8.0, 1.0, 1.0, 1.0, 1.0])
    seepage_std = np.full(N_BLOCKS, 1.0)
    # Subsidence field: quiet everywhere (never exceeds its own threshold) so only seepage drives masking.
    subsidence_mean = np.zeros(N_BLOCKS)
    subsidence_std = np.full(N_BLOCKS, 0.1)
    return BlockFieldPosterior(seepage_mean, seepage_std, subsidence_mean, subsidence_std)


def _reference_network() -> dict:
    # nodes: 0 = source, 1..N_BLOCKS = blocks, N_BLOCKS + 1 = sink.
    n = N_BLOCKS + 2
    sink = N_BLOCKS + 1
    cap = np.zeros((n, n))
    cost = np.zeros((n, n))
    demand = 10.0
    cap[0, 1 : N_BLOCKS + 1] = 1e6  # source -> every block, effectively uncapacitated
    block_capacity = np.full(N_BLOCKS, 5.0)
    # risky blocks are artificially cheap to extract from -- an unconstrained optimizer loves them.
    extraction_cost = np.array([1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
    for i in range(N_BLOCKS):
        cap[1 + i, sink] = block_capacity[i]
        cost[1 + i, sink] = extraction_cost[i]
    supply = np.zeros(n)
    supply[0] = demand
    supply[sink] = -demand
    return {"cap": cap, "cost": cost, "supply": supply, "block_nodes": np.arange(1, N_BLOCKS + 1)}


def test_exclusion_mask_flags_only_the_risky_polygon():
    posterior = _risk_posterior()
    mask = exclusion_mask(
        posterior,
        seepage_prob_cut=0.5,
        subsidence_cut=0.5,
        n_blocks=N_BLOCKS,
        seepage_threshold=5.0,
        subsidence_threshold=1.0,
        rng=np.random.default_rng(0),
    )
    assert mask.dtype == bool
    assert mask.shape == (N_BLOCKS,)
    for i in RISKY_BLOCKS:
        assert mask[i], f"risky block {i} should be excluded"
    for i in SAFE_BLOCKS:
        assert not mask[i], f"safe block {i} should not be excluded"


def test_reclamation_cost_uses_category_rates():
    block_model = {"category": np.array(["ore", "ore", "waste", "waste", "waste", "waste"], dtype=object)}
    disturbance = np.array([1.0, 2.0, 1.0, 1.0, 0.0, 3.0])
    unit_costs = {"ore": 10.0, "waste": 4.0}
    cost = reclamation_cost(block_model, disturbance, unit_costs=unit_costs)
    np.testing.assert_allclose(cost, [10.0, 20.0, 4.0, 4.0, 0.0, 12.0])


def test_reclamation_cost_scalar_rate():
    disturbance = np.array([2.0, 0.0, 5.0])
    cost = reclamation_cost(None, disturbance, unit_costs=2.5)
    np.testing.assert_allclose(cost, [5.0, 0.0, 12.5])


def test_env_constraints_removes_flagged_blocks_and_raises_cost():
    posterior = _risk_posterior()
    mask = exclusion_mask(
        posterior,
        seepage_prob_cut=0.5,
        subsidence_cut=0.5,
        n_blocks=N_BLOCKS,
        seepage_threshold=5.0,
        subsidence_threshold=1.0,
        rng=np.random.default_rng(1),
    )
    assert [i for i in range(N_BLOCKS) if mask[i]] == list(RISKY_BLOCKS)

    block_model = {"category": np.array(["tailings"] * N_BLOCKS, dtype=object)}
    disturbance = np.ones(N_BLOCKS)
    liability = reclamation_cost(block_model, disturbance, unit_costs={"tailings": 0.5})

    network = _reference_network()

    unconstrained = _solve_min_cost_flow(network["cap"], network["cost"], network["supply"])
    # The unconstrained optimum should lean on the (cheap) risky blocks: some flow reaches the sink
    # through at least one of them.
    block_nodes = network["block_nodes"]
    sink = N_BLOCKS + 1
    risky_flow_before = sum(unconstrained.flow[block_nodes[i], sink] for i in RISKY_BLOCKS)
    assert risky_flow_before > 0.0

    payload = apply_env_constraints(network, mask, liability)
    assert payload["forbidden_nodes"] == [int(block_nodes[i]) for i in RISKY_BLOCKS]

    constrained = _solve_min_cost_flow(payload["cap"], payload["cost"], payload["supply"])

    # Feasible: the full demand still reaches the sink.
    total_in = constrained.flow[:, sink].sum()
    assert total_in == pytest.approx(network["supply"][0], rel=1e-6)

    # The flagged blocks carry no flow at all in the re-optimized plan.
    for i in RISKY_BLOCKS:
        assert constrained.flow[block_nodes[i], sink] == pytest.approx(0.0, abs=1e-8)
        assert constrained.flow[:, block_nodes[i]].sum() == pytest.approx(0.0, abs=1e-8)

    # And it costs strictly more than the unconstrained plan.
    assert constrained.value > unconstrained.value


def test_apply_env_constraints_requires_matching_block_counts():
    network = _reference_network()
    with pytest.raises(ValueError):
        apply_env_constraints(network, np.zeros(N_BLOCKS + 1, dtype=bool), np.zeros(N_BLOCKS))
