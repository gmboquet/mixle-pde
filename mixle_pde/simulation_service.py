"""IC-11 -- scenario & forward-simulation runner (work-plan workstream P).

``Scenario``/``ScenarioStep``/``SimResult`` are the frozen shapes a frontier model drives to run a
(possibly coupled) multiphysics what-if. ``simulate`` is the single entry point: a coupling-free
scenario runs its one step through the ``_FORWARDS`` registry and lands a content-hashed artifact; a
scenario with non-empty ``couplings`` delegates to ``_run_coupled_dag``, which topologically walks the
step graph, rewiring each child's ``inputs_ref`` onto its parent's ``result_ref`` so the DAG is
provenance-tracked by construction. ``register_forward`` is the "register, don't branch" seam every
mixle_pde forward operator dispatches through -- the same precedent as
``dynamics.register_dynamics_operator``.

P1 (the upstream owner of this module's single-step path) had not landed on release/0.8.0 at the time
the coupled-DAG runner (P2) was implemented, so this file also carries the minimal P1 substrate P2
builds on: ``register_forward``/``_FORWARDS``, the coupling-free half of ``simulate``, and
``write_result_artifact``/``read_result_artifact``. Three simple, honest forwards ("flow", "transport"/
"dispersion", "poroelastic") are registered so the coupled scenario has real physics to run -- steady
groundwater-head diffusion, advection-diffusion transport, and quasi-static consolidation subsidence --
short of the full torch-differentiable machinery (``flow.NavierStokes2D``, ``poroelastic.BiotPoroelastic1D``,
...) that is P1's own scope. See the PR notes for the reasoning.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from mixle_pde import multiphysics
from mixle_pde.dynamics import AdvectionDiffusionOperator
from mixle_pde.io.artifacts import sha256_of_arrays

RESULT_SCHEMA = "mixle_pde.sim_result/v1"
STORE_DIR_ENV = "MIXLE_PDE_SIM_STORE"


@dataclass
class ScenarioStep:
    op: str  # "wave|flow|em|transport|dispersion|poroelastic|climate_rcm|exposure|habitat|coupled"
    inputs_ref: str  # content-hashed artifact handle (IC-2 style) or a registered field id
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    steps: list[ScenarioStep]
    couplings: list[tuple[int, int, str]] = field(default_factory=list)  # (from_step, to_step, field_name)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimResult:
    result_ref: str  # content-hashed artifact handle
    uncertainty: Any | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


_FORWARDS: dict[str, Callable[[str, dict], dict[str, np.ndarray]]] = {}


def register_forward(op: str, fn: Callable[[str, dict], dict[str, np.ndarray]]) -> None:
    """Register a forward operator under an ``op`` name so ``simulate`` (and the MCP tool) can dispatch."""
    _FORWARDS[op] = fn


def _default_store_dir() -> str:
    """Where content-addressed result artifacts land; override with ``MIXLE_PDE_SIM_STORE`` (tests do)."""
    return os.environ.get(STORE_DIR_ENV) or os.path.join(tempfile.gettempdir(), "mixle_pde_sim_store")


def write_result_artifact(
    arrays: dict[str, np.ndarray], *, grid: dict, units: str, provenance: dict, store_dir: str
) -> str:
    """Content-hash ``arrays`` (the frozen ``sha256_of_arrays`` rule, IC-2) and write
    ``{store_dir}/{ref}.npz`` + ``{ref}.json`` (an IC-2-shaped header). Idempotent: identical arrays
    always re-derive the same ``ref`` and overwrite with identical bytes, so re-running a scenario is
    byte-reproducible."""
    os.makedirs(store_dir, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    ref = sha256_of_arrays(arrays)
    np.savez(os.path.join(store_dir, f"{ref}.npz"), **arrays)
    header = {
        "schema": RESULT_SCHEMA,
        "content_hash": ref,
        "crs": provenance.get("crs") if isinstance(provenance, dict) else None,
        "grid": grid,
        "units": units,
        "provenance": provenance,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(store_dir, f"{ref}.json"), "w") as f:
        json.dump(header, f, indent=2, sort_keys=True, default=str)
    return ref


def read_result_artifact(ref: str, *, store_dir: str) -> dict[str, np.ndarray]:
    """Read back the arrays written by ``write_result_artifact`` for content hash ``ref``."""
    with np.load(os.path.join(store_dir, f"{ref}.npz")) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def read_result_header(ref: str, *, store_dir: str) -> dict[str, Any]:
    """Read the sibling json header (schema/grid/units/provenance) for content hash ``ref``."""
    with open(os.path.join(store_dir, f"{ref}.json")) as f:
        return json.load(f)


def _dispatch(step: ScenarioStep, *, store_dir: str) -> dict[str, np.ndarray]:
    fn = _FORWARDS.get(step.op)
    if fn is None:
        raise ValueError(f"no forward registered for op {step.op!r}; registered: {sorted(_FORWARDS)}")
    return fn(step.inputs_ref, {**step.params, "_store_dir": store_dir})


def _artifact_header(step: ScenarioStep, arrays: dict[str, np.ndarray]) -> tuple[dict, str]:
    grid = step.params.get("grid") or {"shape": list(next(iter(arrays.values())).shape)}
    units = step.params.get("units", "")
    return grid, units


def simulate(scenario: Scenario) -> SimResult:
    """Run a (possibly coupled multiphysics) forward scenario; return a content-hashed result handle.

    Every mixle_pde forward registers an ``op`` here via ``register_forward``. A scenario with
    couplings delegates to ``_run_coupled_dag``; otherwise the single step runs directly. This is the
    one surface a frontier model drives for what-if.
    """
    if scenario.couplings:
        return _run_coupled_dag(scenario)
    store_dir = _default_store_dir()
    step = scenario.steps[0]
    arrays = _dispatch(step, store_dir=store_dir)
    grid, units = _artifact_header(step, arrays)
    ref = write_result_artifact(
        arrays,
        grid=grid,
        units=units,
        provenance={"op": step.op, **scenario.provenance},
        store_dir=store_dir,
    )
    return SimResult(result_ref=ref, uncertainty=None, provenance={"op": step.op, "content_hash": ref})


def _topological_order(n: int, edges: list[tuple[int, int, str]]) -> list[int]:
    """Kahn's algorithm over the ``(from, to)`` edges; raises if ``edges`` do not form a DAG."""
    indeg = [0] * n
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for frm, to, _ in edges:
        adjacency[frm].append(to)
        indeg[to] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    order: list[int] = []
    indeg_work = list(indeg)
    while ready:
        i = ready.pop(0)
        order.append(i)
        for to in adjacency[i]:
            indeg_work[to] -= 1
            if indeg_work[to] == 0:
                ready.append(to)
    if len(order) != n:
        raise ValueError("scenario.couplings form a cycle; a scenario DAG must be acyclic")
    return order


def _run_coupled_dag(scenario: Scenario) -> SimResult:
    """Topologically execute ``scenario.steps`` over ``scenario.couplings`` edges ``(from_step, to_step,
    field_name)``. Each child's ``inputs_ref`` is rewritten onto its parent's ``result_ref`` before the
    child runs, so the child literally consumes the parent's content hash; the produced artifact's
    provenance records ``{"parents": [...], "coupling": field_name}``. ``SimResult.result_ref`` is the
    sink step's (no outgoing edge) artifact; ``provenance`` carries the full ordered edge list plus
    every stage's ``result_ref``.
    """
    steps = scenario.steps
    n = len(steps)
    order = _topological_order(n, scenario.couplings)

    incoming: dict[int, list[tuple[int, str]]] = {i: [] for i in range(n)}
    outdeg = [0] * n
    for frm, to, fname in scenario.couplings:
        incoming[to].append((frm, fname))
        outdeg[frm] += 1

    store_dir = _default_store_dir()
    result_refs: list[str | None] = [None] * n
    for i in order:
        step = steps[i]
        parents = [result_refs[frm] for frm, _ in incoming[i]]
        coupling_field = incoming[i][-1][1] if incoming[i] else None
        inputs_ref = parents[-1] if parents else step.inputs_ref
        run_step = ScenarioStep(op=step.op, inputs_ref=inputs_ref, params=step.params)
        arrays = _dispatch(run_step, store_dir=store_dir)

        step_provenance: dict[str, Any] = {"op": step.op}
        if parents:
            step_provenance["parents"] = parents
            step_provenance["coupling"] = coupling_field

        grid, units = _artifact_header(step, arrays)
        ref = write_result_artifact(arrays, grid=grid, units=units, provenance=step_provenance, store_dir=store_dir)
        result_refs[i] = ref

    sinks = [i for i in range(n) if outdeg[i] == 0]
    sink = sinks[-1] if sinks else order[-1]
    return SimResult(
        result_ref=result_refs[sink],
        uncertainty=None,
        provenance={"steps": result_refs, "couplings": list(scenario.couplings), **scenario.provenance},
    )


# --- built-in forwards (the minimal P1 substrate the coupled DAG runs against) -----------------------


def _flow_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """Steady-state dewatering head: ``-div(K grad h) = source`` (``multiphysics.solve_poisson``)."""
    shape = tuple(params.get("shape", (33,)))
    source = params.get("source", 1.0)
    conductivity = params.get("conductivity", 1.0)
    dirichlet = params.get("dirichlet", 0.0)
    spacing = params.get("spacing", 1.0)
    head = multiphysics.solve_poisson(shape, source, conductivity=conductivity, dirichlet=dirichlet, spacing=spacing)
    return {"head": head}


def _transport_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """Contaminant advection-diffusion (``dynamics.AdvectionDiffusionOperator``), driven by the upstream
    head field's hydraulic gradient when no explicit ``velocity`` is given."""
    store_dir = params["_store_dir"]
    upstream = read_result_artifact(inputs_ref, store_dir=store_dir)
    head = np.asarray(upstream["head"], dtype=float).reshape(-1)
    n = int(params.get("n", head.shape[0]))
    length = float(params.get("length", 1.0))
    h = length / max(n - 1, 1)
    velocity = params.get("velocity")
    if velocity is None:
        grad = np.gradient(head[:n], h) if n > 1 else np.zeros(1)
        velocity = -float(np.mean(grad))
    diffusivity = float(params.get("diffusivity", 1e-3))
    dt = float(params.get("dt", 1.0))
    n_steps = int(params.get("steps", 1))
    operator = AdvectionDiffusionOperator(diffusivity=diffusivity, velocity=float(velocity), n=n, length=length)
    transition = operator.transition_matrix(dt)
    concentration = np.asarray(params.get("initial_concentration", np.zeros(n)), dtype=float)
    if concentration.shape[0] != n:
        concentration = np.resize(concentration, n)
    for _ in range(n_steps):
        concentration = transition @ concentration
    return {"concentration": concentration, "load": concentration}


def _poroelastic_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """Quasi-static consolidation subsidence -- the steady limit of Biot/Terzaghi consolidation,
    ``-div(S grad u) = compressibility * load`` -- driven by the upstream contaminant/load field."""
    store_dir = params["_store_dir"]
    upstream = read_result_artifact(inputs_ref, store_dir=store_dir)
    load = np.asarray(upstream.get("load", upstream.get("concentration")), dtype=float).reshape(-1)
    n = load.shape[0]
    compressibility = float(params.get("compressibility", 1.0))
    storage = params.get("storage", 1.0)
    dirichlet = params.get("dirichlet", 0.0)
    spacing = params.get("spacing", 1.0)
    subsidence = multiphysics.solve_poisson(
        (n,), compressibility * load, conductivity=storage, dirichlet=dirichlet, spacing=spacing
    )
    return {"subsidence": subsidence}


def _coupled_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """``op="coupled"``: dispatch to ``multiphysics.run_coupled`` for one tightly-coupled block-solve
    step (fields on one shared grid, e.g. thermo-elastic or reaction-diffusion exchange)."""
    return multiphysics.run_coupled(
        params["shape"],
        params["conductivities"],
        params["coupling"],
        params["sources"],
        dirichlet=params.get("dirichlet", 0.0),
        spacing=params.get("spacing", 1.0),
        provenance=params.get("provenance"),
    )


register_forward("flow", _flow_forward)
register_forward("transport", _transport_forward)
register_forward("dispersion", _transport_forward)
register_forward("poroelastic", _poroelastic_forward)
register_forward("coupled", _coupled_forward)
