"""Tests for mixle_pde.network_coupling: lumped-parameter 0D/1D network coupling (MP-G7).

Two worked examples anchor the module against ground truth computed independently of its own assembly
code:

* a three-reservoir "star" junction, checked against Millman's theorem (the closed-form solution for a
  single free node fed by several linear conductances to fixed potentials) -- for both a linear
  (Ohm's-law) branch law and a nonlinear (turbulent/orifice, ``sign(dp) * sqrt(|dp|)``) branch law, the
  nonlinear case checked against an independent `scipy.optimize.brentq` root-find of the same scalar
  Kirchhoff equation;
* a small pipe/thermal-circuit network with two free junctions (one degree-2, one degree-3), checked
  against the classic nodal-admittance 2x2 linear system solved by hand (Cramer's rule) -- reused again,
  unchanged, as the t -> infinity ground truth for the transient DAE integrator, since a network's steady
  state does not depend on how much capacitance its nodes carry.

Every solved case also gets an independent per-node conservation check computed directly from the
Kirchhoff-balance definition (never by reading back the module's own residual), so an assembly bug that
happened to still solve *a* linear system would still be caught.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from mixle_pde.network_coupling import (
    NetworkEdge,
    NetworkNode,
    NetworkSolution,
    NetworkSpec,
    NetworkTrajectory,
    simulate_transient,
    solve_steady_state,
)


def _net_flow_into_free_node(spec: NetworkSpec, solution: NetworkSolution, node_id: str) -> float:
    """Independent re-derivation of "net flow into node_id" from the Kirchhoff-balance definition
    (source plus signed incident edge flows), built only from `solution.edge_flows` and the spec's own
    edge endpoints -- not by calling back into any private module helper."""
    node = next(n for n in spec.nodes if n.id == node_id)
    total = node.source
    for edge in spec.edges:
        if edge.node_from == node_id:
            total -= solution.edge_flows[edge.id]
        elif edge.node_to == node_id:
            total += solution.edge_flows[edge.id]
    return total


# --- worked example 1: three-reservoir star junction, linear branches (Millman's theorem) --------------


def _star_spec(flow_kind: str) -> NetworkSpec:
    reservoirs = {"R1": 100.0, "R2": 60.0, "R3": 20.0}
    coeffs = {"R1": 2.0, "R2": 1.0, "R3": 0.5}
    nodes = [NetworkNode(id=rid, kind="temperature", fixed_value=val) for rid, val in reservoirs.items()]
    nodes.append(NetworkNode(id="J", kind="temperature"))
    edges = []
    for rid in reservoirs:
        if flow_kind == "linear":
            edges.append(NetworkEdge(id=f"e_{rid}", node_from=rid, node_to="J", conductance=coeffs[rid]))
        else:
            k = coeffs[rid]
            edges.append(
                NetworkEdge(
                    id=f"e_{rid}",
                    node_from=rid,
                    node_to="J",
                    flow_fn=lambda va, vb, k=k: k * np.sign(va - vb) * np.sqrt(abs(va - vb)),
                )
            )
    return NetworkSpec(nodes=tuple(nodes), edges=tuple(edges))


def test_millman_star_linear_matches_closed_form():
    spec = _star_spec("linear")
    solution = solve_steady_state(spec)

    reservoirs = {"R1": 100.0, "R2": 60.0, "R3": 20.0}
    coeffs = {"R1": 2.0, "R2": 1.0, "R3": 0.5}
    expected_j = sum(coeffs[r] * reservoirs[r] for r in reservoirs) / sum(coeffs.values())

    assert isinstance(solution, NetworkSolution)
    assert solution.method == "linear"
    assert solution.converged
    assert solution.iterations == 0
    assert solution.values["J"] == pytest.approx(expected_j, abs=1e-9)
    # boundary nodes pass through unchanged
    for rid, val in reservoirs.items():
        assert solution.values[rid] == val

    # independent conservation check: J has no source, so net flow in must vanish.
    assert abs(_net_flow_into_free_node(spec, solution, "J")) < 1e-8


def test_millman_star_nonlinear_matches_independent_root_find():
    spec = _star_spec("nonlinear")
    solution = solve_steady_state(spec)

    reservoirs = {"R1": 100.0, "R2": 60.0, "R3": 20.0}
    coeffs = {"R1": 2.0, "R2": 1.0, "R3": 0.5}

    def kirchhoff_residual(v_j: float) -> float:
        return sum(coeffs[r] * np.sign(reservoirs[r] - v_j) * np.sqrt(abs(reservoirs[r] - v_j)) for r in reservoirs)

    expected_j = brentq(kirchhoff_residual, 0.0, 150.0, xtol=1e-12)

    assert solution.method == "picard-linearization"
    assert solution.converged
    assert solution.iterations >= 1
    assert solution.values["J"] == pytest.approx(expected_j, abs=1e-6)
    assert abs(_net_flow_into_free_node(spec, solution, "J")) < 1e-6


# --- worked example 2: two-junction branching network, hand-solved 2x2 nodal system --------------------

_G1, _G2, _G3, _G4 = 3.0, 2.0, 1.0, 4.0  # A-B, B-C, C-D, E-C conductances
_VA, _VD, _VE = 100.0, 0.0, 50.0  # fixed reservoir values


def _branching_spec(*, capacitance_b: float = 0.0, capacitance_c: float = 0.0, initial: float = 0.0) -> NetworkSpec:
    nodes = (
        NetworkNode(id="A", kind="head", fixed_value=_VA),
        NetworkNode(id="D", kind="head", fixed_value=_VD),
        NetworkNode(id="E", kind="head", fixed_value=_VE),
        NetworkNode(id="B", kind="head", capacitance=capacitance_b, initial_value=initial),
        NetworkNode(id="C", kind="head", capacitance=capacitance_c, initial_value=initial),
    )
    edges = (
        NetworkEdge(id="AB", node_from="A", node_to="B", conductance=_G1),
        NetworkEdge(id="BC", node_from="B", node_to="C", conductance=_G2),
        NetworkEdge(id="CD", node_from="C", node_to="D", conductance=_G3),
        NetworkEdge(id="EC", node_from="E", node_to="C", conductance=_G4),
    )
    return NetworkSpec(nodes=nodes, edges=edges)


def _branching_closed_form() -> tuple[float, float]:
    """Hand-solved (Cramer's rule) nodal-admittance system for _branching_spec, derived directly from
    Kirchhoff's law -- not by calling into mixle_pde.network_coupling at all."""
    a11, a12 = _G1 + _G2, -_G2
    a21, a22 = -_G2, _G2 + _G3 + _G4
    b1, b2 = _G1 * _VA, _G3 * _VD + _G4 * _VE
    det = a11 * a22 - a12 * a21
    v_b = (b1 * a22 - a12 * b2) / det
    v_c = (a11 * b2 - a21 * b1) / det
    return v_b, v_c


def test_branching_network_matches_hand_solved_cramers_rule():
    spec = _branching_spec()
    solution = solve_steady_state(spec)
    expected_b, expected_c = _branching_closed_form()

    assert solution.method == "linear"
    assert solution.converged
    assert solution.values["B"] == pytest.approx(expected_b, abs=1e-9)
    assert solution.values["C"] == pytest.approx(expected_c, abs=1e-9)

    # independent conservation check at both free junctions (B has degree 2, C has degree 3).
    assert abs(_net_flow_into_free_node(spec, solution, "B")) < 1e-8
    assert abs(_net_flow_into_free_node(spec, solution, "C")) < 1e-8


def test_branching_network_transient_relaxes_to_the_same_steady_state():
    # Same topology and conductances as the algebraic case, but B and C now carry generalized
    # capacitance (thermal mass / tank storage) and start away from equilibrium -- the DAE integrator
    # must relax to the identical closed-form values, since a network's steady state does not depend on
    # how much capacitance its nodes carry.
    spec = _branching_spec(capacitance_b=4.0, capacitance_c=2.0, initial=0.0)
    expected_b, expected_c = _branching_closed_form()

    t_eval = np.array([0.1, 1.0, 5.0, 15.0, 30.0])
    trajectory = simulate_transient(spec, t_eval)

    assert isinstance(trajectory, NetworkTrajectory)
    assert trajectory.free_node_ids == ("B", "C")
    assert trajectory.node_values["B"].shape == t_eval.shape
    assert trajectory.node_values["C"].shape == t_eval.shape
    # fixed nodes are broadcast as constant trajectories
    assert np.all(trajectory.node_values["A"] == _VA)
    assert np.all(trajectory.node_values["D"] == _VD)
    assert np.all(trajectory.node_values["E"] == _VE)

    # not yet at equilibrium at the first output time (started at 0, away from ~80/~52)...
    assert abs(trajectory.node_values["B"][0] - expected_b) > 1.0
    # ...but has relaxed to it well within the given horizon.
    assert trajectory.node_values["B"][-1] == pytest.approx(expected_b, abs=1e-3)
    assert trajectory.node_values["C"][-1] == pytest.approx(expected_c, abs=1e-3)


# --- validation paths ------------------------------------------------------------------------------


def test_duplicate_node_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        NetworkSpec(nodes=(NetworkNode(id="X", kind="k"), NetworkNode(id="X", kind="k")))


def test_edge_referencing_unknown_node_rejected():
    with pytest.raises(ValueError, match="unknown node_from"):
        NetworkSpec(
            nodes=(NetworkNode(id="X", kind="k", fixed_value=1.0), NetworkNode(id="Y", kind="k")),
            edges=(NetworkEdge(id="e", node_from="missing", node_to="Y", conductance=1.0),),
        )


def test_self_loop_edge_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        NetworkEdge(id="e", node_from="X", node_to="X", conductance=1.0)


def test_edge_requires_exactly_one_of_conductance_or_flow_fn():
    with pytest.raises(ValueError, match="exactly one"):
        NetworkEdge(id="e", node_from="X", node_to="Y")
    with pytest.raises(ValueError, match="exactly one"):
        NetworkEdge(id="e", node_from="X", node_to="Y", conductance=1.0, flow_fn=lambda a, b: a - b)


def test_edge_conductance_must_be_positive():
    with pytest.raises(ValueError, match="conductance"):
        NetworkEdge(id="e", node_from="X", node_to="Y", conductance=0.0)
    with pytest.raises(ValueError, match="conductance"):
        NetworkEdge(id="e", node_from="X", node_to="Y", conductance=-1.0)


def test_fixed_node_with_nonzero_capacitance_rejected():
    with pytest.raises(ValueError, match="capacitance"):
        NetworkNode(id="X", kind="k", fixed_value=1.0, capacitance=2.0)


def test_ungrounded_island_is_a_clear_error_not_a_silent_wrong_answer():
    # X and Y are only connected to each other -- no path to any fixed node -- so the assembled system
    # is exactly singular (the network has no reference/ground).
    spec = NetworkSpec(
        nodes=(NetworkNode(id="X", kind="k"), NetworkNode(id="Y", kind="k")),
        edges=(NetworkEdge(id="xy", node_from="X", node_to="Y", conductance=1.0),),
    )
    with pytest.raises(ValueError, match="singular"):
        solve_steady_state(spec)


def test_free_and_fixed_node_id_properties():
    spec = _branching_spec()
    assert spec.free_node_ids == ("B", "C")
    assert spec.fixed_node_ids == ("A", "D", "E")
