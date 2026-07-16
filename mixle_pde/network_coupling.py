"""Lumped-parameter / 0D-1D network coupling: typed nodes and edges assembled into a global
algebraic/DAE system (MP-G7).

Many multiphysics problems are not worth discretizing as a full field PDE everywhere: a well, a pipe
run, a thermal circuit board, or a reservoir tank is well captured by a handful of *lumped* state
variables connected by *conductive* relations, exactly the way SPICE treats an electrical circuit or a
bond graph treats a generalized network (Karnopp, Margolis & Rosenberg, *System Dynamics*). This module
is the reference kernel for that pattern, independent of any single physical domain:

* a :class:`NetworkNode` is one lumped state variable (a junction pressure, a thermal-mass temperature,
  a tank head, a circuit-node voltage, ...), either a free unknown or a prescribed boundary value
  (``fixed_value``), and optionally a *generalized capacitance* (thermal mass, electrical capacitance,
  tank cross-sectional area, ...) that turns its balance equation from algebraic into differential;
* a :class:`NetworkEdge` is a *generalized conductance* relation between two adjacent nodes -- either
  linear (``conductance``, Ohm's-law-like: flow proportional to the value difference) or a caller-
  supplied nonlinear constitutive law (``flow_fn``, e.g. turbulent/orifice pipe flow, flow proportional
  to ``sign(dp) * sqrt(|dp|)``);
* :func:`solve_steady_state` assembles the Kirchhoff-style nodal balance ("net flow into every free
  node is zero", the network generalization of Kirchhoff's current law -- Desoer & Kuh, *Basic Circuit
  Theory*) into a global system and solves it: a direct dense linear solve when every edge is linear, or
  a damped fixed-point iteration (successive linearization: each free node's neighboring edges are
  relinearized at the current iterate's secant conductance, then the resulting *linear* system is solved
  exactly and the whole cycle repeats -- the classical "linear theory method" for nonlinear resistor and
  pipe networks, e.g. Wood & Charles 1972) when any edge is nonlinear;
* :func:`simulate_transient` promotes the same assembled balance to a genuine differential-algebraic
  system -- differential rows for capacitive nodes, algebraic rows (mass-matrix zero) for the rest -- and
  integrates it with the existing semi-explicit index-1 DAE solver,
  :func:`mixle_pde.dynamics.integrate_dae` (Brenan, Campbell & Petzold, *Numerical Solution of Initial-
  Value Problems in DAEs*), rather than re-implementing a second time integrator here.

This fills MP-G7 in ``docs/reconciliation/mp-task-ledger.md`` ("Global equations and 0D/1D/network
coupling"), previously ``not-started``: the only prior "network" code
(``mixle_sim/geometry.py::Network``/``NetworkNode``/``NetworkEdge``) is 1-D *geometry* (well/fracture/
pipe placement in space), not lumped-parameter global-equation coupling, and in any case exists only on
an unmerged sibling-repo PR. This module is self-contained and imports nothing from mixle-sim.

Scope (baseline, matching this repo's other MP-* baselines): dynamics live on *nodes* only (a
generalized capacitance/storage term), not on edges -- there is no inertive/inductive branch state
(an edge's own flow has no independent dynamics). ``flow_fn`` is expected to be antisymmetric
(``flow_fn(a, b) == -flow_fn(b, a)``, ``flow_fn(a, a) == 0``) and, for the fixed-point iteration to
converge reliably, monotone non-decreasing in ``a - b``; neither property is verified generically.
Boundary (``fixed_value``) nodes are time-invariant constants, not time-varying forcing functions. There
is no graph-connectivity pre-check: a free node with no conductive path (direct or indirect) to any
fixed node makes the assembled system singular, surfaced as a clear :class:`ValueError` from the linear
solve rather than diagnosed structurally beforehand. :func:`simulate_transient`'s accuracy and step
control are exactly those of :func:`mixle_pde.dynamics.integrate_dae` (fixed-``h_max`` substeps, not
adaptive).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "NetworkNode",
    "NetworkEdge",
    "NetworkSpec",
    "NetworkSolution",
    "NetworkTrajectory",
    "solve_steady_state",
    "simulate_transient",
]

_SECANT_EPS = 1.0e-9
_MIN_CONDUCTANCE = 1.0e-12


@dataclass(frozen=True)
class NetworkNode:
    """One lumped state variable in the network.

    ``fixed_value`` (when not ``None``) makes this a prescribed boundary/reservoir node -- it is never
    solved for, and its ``capacitance`` must be ``0`` (a value that never changes has no storage
    equation). Otherwise this is a free node: ``capacitance == 0`` makes it *algebraic* (its balance
    equation is the Kirchhoff constraint "net flow in is zero" at every instant, no accumulation), and
    ``capacitance > 0`` makes it *differential* (``capacitance * d(value)/dt = net flow in``), the
    generalized-capacitance analogy shared by thermal mass, electrical capacitance, and tank
    cross-sectional area. ``source`` is an external flow injected at this node (a pump, a heat source, a
    current source); it is ignored for a fixed node. ``initial_value`` is the state at ``t0`` for
    :func:`simulate_transient` (differential nodes) and the fixed-point starting guess for
    :func:`solve_steady_state` (algebraic nodes); it is unused for a fixed node.
    """

    id: str
    kind: str
    initial_value: float = 0.0
    capacitance: float = 0.0
    fixed_value: float | None = None
    source: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("NetworkNode.id must be a non-empty string.")
        if self.capacitance < 0.0:
            raise ValueError(f"NetworkNode {self.id!r}.capacitance must be >= 0; got {self.capacitance!r}.")
        if self.fixed_value is not None and self.capacitance != 0.0:
            raise ValueError(
                f"NetworkNode {self.id!r} sets both fixed_value and a nonzero capacitance; a fixed "
                "(boundary) node has no storage equation, so capacitance must be 0 when fixed_value is set."
            )


@dataclass(frozen=True)
class NetworkEdge:
    """A generalized-conductance relation between two adjacent nodes.

    Exactly one of ``conductance`` (linear: the flow from ``node_from`` to ``node_to`` is
    ``conductance * (value_from - value_to)``) or ``flow_fn`` (nonlinear: the flow is
    ``flow_fn(value_from, value_to)``) must be given. ``flow_fn`` should be antisymmetric
    (``flow_fn(a, b) == -flow_fn(b, a)``) so the constitutive law does not depend on which endpoint was
    labeled ``node_from`` -- see the module Scope note.
    """

    id: str
    node_from: str
    node_to: str
    conductance: float | None = None
    flow_fn: Callable[[float, float], float] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("NetworkEdge.id must be a non-empty string.")
        if self.node_from == self.node_to:
            raise ValueError(
                f"NetworkEdge {self.id!r} connects node {self.node_from!r} to itself; self-loops are not supported."
            )
        if (self.conductance is None) == (self.flow_fn is None):
            raise ValueError(
                f"NetworkEdge {self.id!r} requires exactly one of `conductance` (linear) or `flow_fn` "
                f"(nonlinear), never both or neither; got conductance={self.conductance!r}, "
                f"flow_fn={self.flow_fn!r}."
            )
        if self.conductance is not None and self.conductance <= 0.0:
            raise ValueError(f"NetworkEdge {self.id!r}.conductance must be > 0; got {self.conductance!r}.")


@dataclass(frozen=True)
class NetworkSpec:
    """A complete lumped-parameter network: its nodes and the edges connecting them."""

    nodes: tuple[NetworkNode, ...]
    edges: tuple[NetworkEdge, ...] = ()

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("NetworkSpec.nodes must be non-empty.")
        node_ids = [n.id for n in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            dupes = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
            raise ValueError(f"NetworkSpec node ids must be unique; duplicated: {dupes}.")
        edge_ids = [e.id for e in self.edges]
        if len(set(edge_ids)) != len(edge_ids):
            dupes = sorted({edge_id for edge_id in edge_ids if edge_ids.count(edge_id) > 1})
            raise ValueError(f"NetworkSpec edge ids must be unique; duplicated: {dupes}.")
        node_id_set = set(node_ids)
        for edge in self.edges:
            if edge.node_from not in node_id_set:
                raise ValueError(f"NetworkEdge {edge.id!r} references unknown node_from {edge.node_from!r}.")
            if edge.node_to not in node_id_set:
                raise ValueError(f"NetworkEdge {edge.id!r} references unknown node_to {edge.node_to!r}.")

    @property
    def free_node_ids(self) -> tuple[str, ...]:
        """Ids of nodes whose value is solved for (``fixed_value is None``), in declaration order."""
        return tuple(n.id for n in self.nodes if n.fixed_value is None)

    @property
    def fixed_node_ids(self) -> tuple[str, ...]:
        """Ids of prescribed boundary nodes (``fixed_value is not None``), in declaration order."""
        return tuple(n.id for n in self.nodes if n.fixed_value is not None)


@dataclass(frozen=True)
class NetworkSolution:
    """The result of :func:`solve_steady_state`.

    ``values`` covers every node (solved free values plus the given fixed values); ``edge_flows`` gives
    the flow from ``node_from`` to ``node_to`` for every edge, evaluated at ``values``. ``method`` is
    ``"trivial"`` (no free nodes), ``"linear"`` (every edge linear: one direct dense solve), or
    ``"picard-linearization"`` (any nonlinear edge: damped fixed-point iteration). ``converged`` is
    ``residual_norm <= tol`` actually checked against the true (non-linearized) nodal balance -- never a
    fabricated success flag -- and ``iterations`` counts the fixed-point steps taken (``0`` for the
    direct linear solve, which needs no iterative refinement).
    """

    values: dict[str, float]
    edge_flows: dict[str, float]
    converged: bool
    iterations: int
    residual_norm: float
    method: str


@dataclass(frozen=True)
class NetworkTrajectory:
    """The result of :func:`simulate_transient`: every node's value at each requested time.

    ``node_values[node_id]`` has shape ``(len(times),)`` for every node (free and fixed alike -- fixed
    nodes are broadcast as a constant trajectory, so callers never need to special-case them).
    """

    times: np.ndarray
    node_values: dict[str, np.ndarray]
    free_node_ids: tuple[str, ...]


def _incidence(spec: NetworkSpec) -> dict[str, list[NetworkEdge]]:
    incidence: dict[str, list[NetworkEdge]] = {node.id: [] for node in spec.nodes}
    for edge in spec.edges:
        incidence[edge.node_from].append(edge)
        incidence[edge.node_to].append(edge)
    return incidence


def _edge_flow(edge: NetworkEdge, values: dict[str, float]) -> float:
    """The flow along ``edge`` in the ``node_from -> node_to`` direction, at ``values``."""
    v_from = values[edge.node_from]
    v_to = values[edge.node_to]
    if edge.conductance is not None:
        return edge.conductance * (v_from - v_to)
    return float(edge.flow_fn(v_from, v_to))


def _net_flow_into(
    node_id: str,
    values: dict[str, float],
    node_map: dict[str, NetworkNode],
    incidence: dict[str, list[NetworkEdge]],
) -> float:
    """External source plus the sum of incident edge flows directed into ``node_id`` -- the left-hand
    side of that node's Kirchhoff balance (``capacitance * dvalue/dt`` for a differential node, and the
    residual that must vanish for an algebraic one)."""
    total = node_map[node_id].source
    for edge in incidence[node_id]:
        flow = _edge_flow(edge, values)
        if edge.node_from == node_id:
            total -= flow
        else:
            total += flow
    return total


def _residual_norm(
    spec: NetworkSpec,
    values: dict[str, float],
    node_map: dict[str, NetworkNode],
    incidence: dict[str, list[NetworkEdge]],
    free_ids: tuple[str, ...],
) -> float:
    if not free_ids:
        return 0.0
    residuals = [_net_flow_into(node_id, values, node_map, incidence) for node_id in free_ids]
    return float(np.max(np.abs(residuals)))


def _secant_conductance(edge: NetworkEdge, values: dict[str, float]) -> float:
    """A local linear conductance for ``edge`` at ``values``: itself if linear, otherwise the secant
    slope of ``flow_fn`` (falling back to a numerical derivative when the value difference is too small
    for the secant to be well-conditioned -- exactly the physically common "no flow yet" state)."""
    if edge.conductance is not None:
        return edge.conductance
    v_from, v_to = values[edge.node_from], values[edge.node_to]
    dv = v_from - v_to
    flow = float(edge.flow_fn(v_from, v_to))
    if abs(dv) > _SECANT_EPS:
        g = flow / dv
    else:
        bumped = float(edge.flow_fn(v_from + _SECANT_EPS, v_to))
        g = (bumped - flow) / _SECANT_EPS
    return max(g, _MIN_CONDUCTANCE)


def _assemble_linear(
    spec: NetworkSpec,
    node_map: dict[str, NetworkNode],
    free_ids: tuple[str, ...],
    edge_conductance: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """The Dirichlet-eliminated nodal-conductance matrix and right-hand side for the free nodes: a
    direct generalization of the classic resistive-network nodal-admittance assembly (graph Laplacian
    restricted to the free nodes, plus boundary-injection terms from fixed neighbors)."""
    index = {node_id: k for k, node_id in enumerate(free_ids)}
    n = len(free_ids)
    matrix = np.zeros((n, n), dtype=float)
    rhs = np.array([node_map[node_id].source for node_id in free_ids], dtype=float)
    for edge in spec.edges:
        g = edge_conductance[edge.id]
        i_from = index.get(edge.node_from)
        i_to = index.get(edge.node_to)
        if i_from is not None:
            matrix[i_from, i_from] += g
        if i_to is not None:
            matrix[i_to, i_to] += g
        if i_from is not None and i_to is not None:
            matrix[i_from, i_to] -= g
            matrix[i_to, i_from] -= g
        elif i_from is not None:  # node_to is a fixed boundary value
            rhs[i_from] += g * node_map[edge.node_to].fixed_value
        elif i_to is not None:  # node_from is a fixed boundary value
            rhs[i_to] += g * node_map[edge.node_from].fixed_value
        # else: both endpoints fixed -- the edge carries a determined flow but constrains no unknown.
    return matrix, rhs


def _solve_linear_block(
    spec: NetworkSpec,
    node_map: dict[str, NetworkNode],
    free_ids: tuple[str, ...],
    edge_conductance: dict[str, float],
) -> np.ndarray:
    matrix, rhs = _assemble_linear(spec, node_map, free_ids, edge_conductance)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "network steady-state system is singular -- check for a free node with no conductive path "
            "(directly or indirectly) to any fixed/boundary node (an ungrounded island)."
        ) from exc


def solve_steady_state(
    spec: NetworkSpec,
    *,
    tol: float = 1.0e-8,
    max_iterations: int = 200,
    relaxation: float = 1.0,
) -> NetworkSolution:
    """Solve the network's steady-state Kirchhoff balance: ``net flow into every free node == 0``.

    Capacitance never enters this equation (steady state is exactly the state where nothing is changing,
    so ``capacitance * dvalue/dt == 0`` regardless of the capacitance value) -- it only matters for
    :func:`simulate_transient`. When every edge is linear this is one direct dense solve
    (``method="linear"``, via :func:`numpy.linalg.solve`) of the assembled nodal-conductance system --
    exact up to floating-point precision, no iteration needed. When any edge is nonlinear, a damped
    fixed-point iteration relinearizes every edge at the current iterate's secant conductance
    (:func:`_secant_conductance`), solves the resulting *linear* system exactly, and repeats
    (``method="picard-linearization"``) until the true (non-linearized) residual falls below ``tol`` or
    ``max_iterations`` is reached -- ``converged=False`` past that point rather than a fabricated
    success.
    """
    node_map = {n.id: n for n in spec.nodes}
    free_ids = spec.free_node_ids
    incidence = _incidence(spec)
    values = {n.id: (n.fixed_value if n.fixed_value is not None else n.initial_value) for n in spec.nodes}

    if not free_ids:
        edge_flows = {e.id: _edge_flow(e, values) for e in spec.edges}
        return NetworkSolution(
            values=values, edge_flows=edge_flows, converged=True, iterations=0, residual_norm=0.0, method="trivial"
        )

    is_linear = all(e.conductance is not None for e in spec.edges)
    iterations = 0
    if is_linear:
        method = "linear"
        conductances = {e.id: e.conductance for e in spec.edges}
        solved = _solve_linear_block(spec, node_map, free_ids, conductances)
        for node_id, value in zip(free_ids, solved):
            values[node_id] = float(value)
    else:
        method = "picard-linearization"
        for iterations in range(1, max_iterations + 1):
            conductances = {e.id: _secant_conductance(e, values) for e in spec.edges}
            solved = _solve_linear_block(spec, node_map, free_ids, conductances)
            for node_id, value in zip(free_ids, solved):
                values[node_id] += relaxation * (float(value) - values[node_id])
            if _residual_norm(spec, values, node_map, incidence, free_ids) <= tol:
                break

    residual_norm = _residual_norm(spec, values, node_map, incidence, free_ids)
    edge_flows = {e.id: _edge_flow(e, values) for e in spec.edges}
    return NetworkSolution(
        values=values,
        edge_flows=edge_flows,
        converged=residual_norm <= tol,
        iterations=iterations,
        residual_norm=residual_norm,
        method=method,
    )


def simulate_transient(
    spec: NetworkSpec,
    t_eval,
    *,
    t0: float = 0.0,
    h_max: float = 0.05,
    newton_tol: float = 1.0e-11,
    max_newton: int = 60,
) -> NetworkTrajectory:
    """Integrate the network forward in time as a semi-explicit index-1 DAE.

    Differential (``capacitance > 0``) free nodes get a genuine ODE row (``capacitance * dvalue/dt =
    net flow in``); algebraic (``capacitance == 0``) free nodes get a mass-matrix-zero row, i.e. the same
    Kirchhoff constraint :func:`solve_steady_state` solves, enforced at every instant rather than only at
    steady state. Both are handed as one system to :func:`mixle_pde.dynamics.integrate_dae`, which
    requires a *consistent* initial condition (the algebraic rows must already hold at ``t0``): this
    function builds one automatically by clamping every differential node at its declared
    ``initial_value`` and solving the remaining algebraic nodes to steady state against that (reusing
    :func:`solve_steady_state`, not a second solver).
    """
    # imported lazily so a caller that only needs the steady-state path never pays for this dependency
    from mixle_pde.dynamics import integrate_dae

    node_map = {n.id: n for n in spec.nodes}
    free_ids = spec.free_node_ids
    incidence = _incidence(spec)
    if not free_ids:
        raise ValueError("simulate_transient requires at least one free (non-fixed) node.")

    clamped_nodes = tuple(
        NetworkNode(
            id=n.id,
            kind=n.kind,
            initial_value=n.initial_value,
            capacitance=0.0,
            fixed_value=n.initial_value,
            source=n.source,
        )
        if (n.fixed_value is None and n.capacitance > 0.0)
        else n
        for n in spec.nodes
    )
    consistent = solve_steady_state(NetworkSpec(nodes=clamped_nodes, edges=spec.edges))

    y0 = np.array(
        [
            node_map[node_id].initial_value if node_map[node_id].capacitance > 0.0 else consistent.values[node_id]
            for node_id in free_ids
        ],
        dtype=float,
    )
    mass = np.diag([node_map[node_id].capacitance for node_id in free_ids]).astype(float)
    fixed_values = {n.id: n.fixed_value for n in spec.nodes if n.fixed_value is not None}

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        values = dict(fixed_values)
        values.update(zip(free_ids, y))
        return np.array([_net_flow_into(node_id, values, node_map, incidence) for node_id in free_ids], dtype=float)

    trajectory = np.asarray(
        integrate_dae(rhs, y0, t_eval, mass, t0=t0, h_max=h_max, newton_tol=newton_tol, max_newton=max_newton)
    )

    times = np.asarray(t_eval, dtype=float)
    node_values = {node_id: trajectory[:, i] for i, node_id in enumerate(free_ids)}
    for n in spec.nodes:
        if n.fixed_value is not None:
            node_values[n.id] = np.full(times.shape, n.fixed_value, dtype=float)

    return NetworkTrajectory(times=times, node_values=node_values, free_node_ids=free_ids)
