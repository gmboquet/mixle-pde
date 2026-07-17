"""IC-11 -- scenario & forward-simulation runner (work-plan workstream P).

``Scenario``/``ScenarioStep``/``SimResult`` are the frozen shapes a frontier model drives to run a
(possibly coupled) multiphysics what-if. ``simulate`` is the single entry point: a coupling-free
scenario runs its one step through the ``_FORWARDS`` registry and lands a content-hashed artifact; a
scenario with non-empty ``couplings`` delegates to ``_run_coupled_dag``, which topologically walks the
step graph, rewiring each child's ``inputs_ref`` onto its parent's ``result_ref`` so the DAG is
provenance-tracked by construction. ``register_forward`` is the "register, don't branch" seam every
mixle_pde forward operator dispatches through -- the same precedent as
``dynamics.register_dynamics_operator``.

P1's own full-fidelity forwards are registered below: ``flow`` -> ``flow.NavierStokes2D`` (vorticity-
streamfunction), ``transport``/``dispersion`` -> ``dynamics.AdvectionDiffusionOperator``, ``wave`` ->
``wave.WaveEquation2D``, ``em`` -> ``em_diffusion.mt_2d_te`` (or ``maxwell.Maxwell3D`` FDTD), ``poroelastic``
-> ``poroelastic.BiotPoroelastic1D``, and ``model`` -> ``mixle.inference.simulate.simulate``. An earlier,
temporary substrate registered simplified stand-ins (steady Poisson diffusion) for ``flow``/``poroelastic``
under the same op names before P1 itself landed; those are replaced here by the real solvers the spec always
called for. ``_primary_field`` resolves a forward's input array across a small priority list of upstream key
names (``"field"``, or a producer-specific name like ``"psi"``) rather than hard-coding one producer's output
shape, so ``transport`` can consume either a raw ``"field"`` artifact (standalone) or an upstream ``flow``
step's streamfunction (coupled) without either caller needing to know about the other.
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
from mixle.reason.posterior_protocol import Posterior

from mixle_pde import multiphysics
from mixle_pde.dynamics import AdvectionDiffusionOperator
from mixle_pde.io.artifacts import sha256_of_arrays
from mixle_pde.surrogate import Surrogate, distill_forward

RESULT_SCHEMA = "mixle_pde.sim_result/v1"
STORE_DIR_ENV = "MIXLE_PDE_SIM_STORE"
UQ_LEVEL = 0.9  # central credible-interval mass propagate_uq reports (work-plan P4; not IC-11 frozen)


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
    if not callable(fn):
        raise TypeError("forward must be callable")
    _FORWARDS[op] = fn


def available_forwards() -> list[str]:
    """The sorted names of every registered forward operator."""
    return sorted(_FORWARDS)


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


def _resolve_artifact_path(ref: str, store_dir: str) -> str:
    """Resolve ``ref`` -- a bare content-hash living in ``store_dir``, or a filesystem path (absolute,
    relative with a path separator, or simply an existing ``<ref>.npz``) -- to the ``.npz``-less base
    path. Lets a scenario step point directly at an arbitrary artifact file (a fixture, a raw upstream
    dataset) without first having to run it through ``write_result_artifact``."""
    candidate = ref[: -len(".npz")] if ref.endswith(".npz") else ref
    if os.path.isabs(candidate) or os.sep in candidate or os.path.exists(candidate + ".npz"):
        return candidate
    return os.path.join(store_dir, candidate)


def read_result_artifact(ref: str, *, store_dir: str) -> dict[str, np.ndarray]:
    """Read back the arrays written by ``write_result_artifact`` for content hash ``ref``, or any raw
    ``.npz`` file ``ref`` resolves to (see ``_resolve_artifact_path``)."""
    path = _resolve_artifact_path(ref, store_dir)
    with np.load(f"{path}.npz") as data:
        return {k: np.asarray(data[k]) for k in data.files}


def read_result_header(ref: str, *, store_dir: str) -> dict[str, Any]:
    """Read the sibling json header (schema/grid/units/provenance) for content hash ``ref``."""
    with open(os.path.join(store_dir, f"{ref}.json")) as f:
        return json.load(f)


def _dispatch(step: ScenarioStep, *, store_dir: str) -> dict[str, np.ndarray]:
    fn = _FORWARDS.get(step.op)
    if fn is None:
        raise KeyError(f"no forward registered for op {step.op!r}; registered ops: {available_forwards()}")
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
    if len(scenario.steps) != 1:
        raise ValueError(
            "simulate() without couplings requires exactly one ScenarioStep; a multi-step scenario needs "
            "scenario.couplings to route through _run_coupled_dag."
        )
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


# --- built-in forwards ---------------------------------------------------------------------------


def _primary_field(arrays: dict[str, np.ndarray], *, keys: tuple[str, ...] = ("field",)) -> np.ndarray:
    """Pick the array a scalar-field-consuming forward should evolve: the first of ``keys`` present in
    ``arrays``, else the sole array if there is exactly one. ``keys`` lets a forward accept a specific
    upstream producer's output name (e.g. ``"psi"`` from ``flow``) ahead of the generic ``"field"``
    convention, without either the consumer or the producer needing to know about the other."""
    for key in keys:
        if key in arrays:
            return np.asarray(arrays[key], dtype=float)
    if len(arrays) == 1:
        return np.asarray(next(iter(arrays.values())), dtype=float)
    raise KeyError(f"expected one of {keys!r} (or exactly one array) in the input artifact; got {sorted(arrays)}")


def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)


def _flow_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The flow forward: step ``flow.NavierStokes2D`` (vorticity-streamfunction) ``params["steps"]``
    times from an initial vorticity field read from ``inputs_ref`` (key ``"omega"`` or ``"field"``, or
    the sole array)."""
    from mixle_pde.flow import NavierStokes2D
    from mixle_pde.ops import make_ops

    data = read_result_artifact(inputs_ref, store_dir=params["_store_dir"])
    omega0 = _primary_field(data, keys=("omega", "field"))
    n = int(params.get("n", int(round(omega0.shape[0] ** 0.5))))
    ns = NavierStokes2D(
        n,
        viscosity=float(params.get("viscosity", 0.1)),
        dt=float(params.get("dt", 0.01)),
        spacing=params.get("spacing"),
        implicit_diffusion=bool(params.get("implicit_diffusion", False)),
    )
    ops = make_ops()
    omega = ops.tensor(omega0)
    for _ in range(int(params.get("steps", 1))):
        omega = ns.step(omega, ops)
    psi = ns.streamfunction(omega, ops)
    return {"omega": _to_numpy(omega), "psi": _to_numpy(psi)}


def _transport_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The transport/dispersion forward: step ``dynamics.AdvectionDiffusionOperator``
    ``params["steps"]`` times. Loads the initial scalar field from ``inputs_ref`` (key ``"field"`` for
    a standalone/raw input, or ``"psi"``/``"head"`` from an upstream ``flow``-style producer, or the
    sole array), builds the one-step transition matrix (implicit Euler by default), and applies it."""
    data = read_result_artifact(inputs_ref, store_dir=params["_store_dir"])
    field0 = _primary_field(data, keys=("field", "psi", "head"))
    n = int(params.get("n", field0.shape[0]))
    if field0.shape[0] != n:
        field0 = np.resize(field0, n)
    operator = AdvectionDiffusionOperator(
        diffusivity=float(params["diffusivity"]),
        velocity=float(params["velocity"]),
        n=n,
        length=float(params.get("length", 1.0)),
        bc=params.get("bc", "periodic"),
        scheme=params.get("scheme", "implicit"),
    )
    dt = float(params.get("dt", 0.05))
    transition = operator.transition_matrix(dt)
    u = field0.copy()
    for _ in range(int(params.get("steps", 1))):
        u = transition @ u
    return {"field": u}


def _forward_wave(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The wave forward: step ``wave.WaveEquation2D`` ``params["steps"]`` times (zero source unless
    ``c2``/``u0``/``w0`` are given in the input artifact)."""
    from mixle_pde.ops import make_ops
    from mixle_pde.wave import WaveEquation2D

    data = read_result_artifact(inputs_ref, store_dir=params["_store_dir"])
    n = int(params["n"]) if "n" in params else int(round(len(next(iter(data.values()))) ** 0.5))
    nn = n * n
    c2 = data.get("c2")
    c2 = np.full(nn, float(params.get("velocity", 1.0)) ** 2) if c2 is None else np.asarray(c2, dtype=float).ravel()
    u0 = np.asarray(data.get("u0", np.zeros(nn)), dtype=float).ravel()
    w0 = np.asarray(data.get("w0", np.zeros(nn)), dtype=float).ravel()

    wave = WaveEquation2D(
        n,
        dt=float(params.get("dt", 0.1 / max(n - 1, 1))),
        spacing=params.get("spacing"),
        absorb_width=int(params.get("absorb_width", 0)),
        absorb_strength=float(params.get("absorb_strength", 2.0)),
    )
    ops = make_ops()
    state = wave.pack(u0, w0)
    c2_t = ops.tensor(c2)
    for _ in range(int(params.get("steps", 1))):
        state = wave.step(state, c2_t, ops)
    return {"u": _to_numpy(wave.displacement(state))}


def _forward_em(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The em forward: 2-D magnetotelluric TE-mode sounding (``em_diffusion.mt_2d_te``) by default;
    ``params["mode"] == "fdtd"`` steps a 3-D ``maxwell.Maxwell3D`` cavity instead, for the transient
    regime."""
    data = read_result_artifact(inputs_ref, store_dir=params["_store_dir"])
    if params.get("mode") == "fdtd":
        from mixle_pde.maxwell import Maxwell3D
        from mixle_pde.ops import make_ops

        n = int(params["n"])
        nc = n**3
        zeros = lambda: np.zeros(nc)  # noqa: E731
        Ex, Ey, Ez = data.get("Ex", zeros()), data.get("Ey", zeros()), data.get("Ez", zeros())
        Hx, Hy, Hz = data.get("Hx", zeros()), data.get("Hy", zeros()), data.get("Hz", zeros())
        mx = Maxwell3D(
            n,
            dt=float(params.get("dt", 0.1 / max(n - 1, 1))),
            spacing=params.get("spacing"),
            eps=float(params.get("eps", 1.0)),
            mu=float(params.get("mu", 1.0)),
        )
        ops = make_ops()
        state = mx.pack(Ex, Ey, Ez, Hx, Hy, Hz)
        for _ in range(int(params.get("steps", 1))):
            state = mx.step(state, ops)
        names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        return dict(zip(names, (_to_numpy(c) for c in mx.unpack(state))))

    from mixle_pde.em_diffusion import mt_2d_te

    log_sigma = _primary_field(data, keys=("log_sigma",))
    shape = tuple(int(s) for s in params["shape"])
    rho_a, phase = mt_2d_te(
        log_sigma,
        shape,
        freq=float(params["freq"]),
        spacing=params.get("spacing", 1.0),
        sigma_ref=float(params.get("sigma_ref", 1.0)),
    )
    return {"rho_a": _to_numpy(rho_a), "phase": _to_numpy(phase)}


def _poroelastic_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The poroelastic forward: step ``poroelastic.BiotPoroelastic1D`` ``params["steps"]`` times.

    Reads its initial ``(v, q, sigma, pf)`` state directly from the input artifact when those keys are
    present (the native convention); otherwise treats a single upstream scalar (e.g. an upstream
    ``transport`` step's evolved ``"field"``) as the initial pore-fluid pressure ``pf`` -- the natural
    "loading" analog for a coupled scenario feeding one field into subsidence -- with ``v``/``q``/
    ``sigma`` at rest."""
    from mixle_pde.ops import make_ops
    from mixle_pde.poroelastic import BiotPoroelastic1D

    data = read_result_artifact(inputs_ref, store_dir=params["_store_dir"])
    if any(k in data for k in ("v", "q", "sigma", "pf")):
        n = int(params["n"]) if "n" in params else len(next(iter(data.values())))
        zeros = lambda: np.zeros(n)  # noqa: E731
        v0, q0, sigma0 = data.get("v", zeros()), data.get("q", zeros()), data.get("sigma", zeros())
        pf0 = data.get("pf", zeros())
    else:
        pf0 = _primary_field(data, keys=("field", "load", "concentration"))
        n = int(params.get("n", pf0.shape[0]))
        v0 = q0 = sigma0 = np.zeros(n)

    kwarg_names = (
        "spacing",
        "k_solid",
        "k_fluid",
        "k_dry",
        "mu",
        "phi",
        "rho_solid",
        "rho_fluid",
        "eta",
        "permeability",
        "tortuosity",
        "absorb_width",
        "absorb_strength",
    )
    kwargs = {k: params[k] for k in kwarg_names if k in params}
    biot = BiotPoroelastic1D(n, dt=float(params.get("dt", 1e-4)), **kwargs)
    ops = make_ops()
    state = biot.pack(v0, q0, sigma0, pf0)
    for _ in range(int(params.get("steps", 1))):
        state = biot.step(state, ops)
    v, q, sigma, pf = biot.unpack(state)
    return {
        "v": _to_numpy(v),
        "q": _to_numpy(q),
        "sigma": _to_numpy(sigma),
        "pf": _to_numpy(pf),
        "subsidence": _to_numpy(pf),
    }


def _forward_model(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The model forward: sample ``n`` draws from a fitted core generative model via
    ``mixle.inference.simulate.simulate(model).run(n, seed=...)``.

    Unlike the mesh-based pde forwards, a fitted model is not a numpy-array artifact, so it is passed
    in-process as ``params["model"]`` (a live Python object) rather than resolved from ``inputs_ref``.
    """
    from mixle.inference.simulate import simulate as core_simulate

    model = params.get("model")
    if model is None:
        raise ValueError("op 'model' requires params['model'] (a fitted generative model object)")
    sim = core_simulate(model)
    records = sim.run(int(params.get("n", 100)), seed=int(params.get("seed", 0)))
    return {"samples": np.asarray(records, dtype=float)}


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
register_forward("wave", _forward_wave)
register_forward("em", _forward_em)
register_forward("poroelastic", _poroelastic_forward)
register_forward("model", _forward_model)
register_forward("coupled", _coupled_forward)
# "climate_rcm"/"exposure"/"habitat" are open slots L7/K/N register -- not this task's forwards.


# --- IC-10 catalog registration -- "so the router (M3) sees it uniformly" -------------------------
#
# `mixle.task.catalog` is IC-10's frozen module. If it has not landed in core mixle in a given
# environment, importing it unconditionally would make this whole module fail to import and block
# every P1/P2/P4 consumer on an unrelated contract. Rather than touch the frozen contract or block on
# it, degrade gracefully: use the real `CatalogEntry`/`ToolCatalog` when importable, and fall back to a
# private shim reproducing IC-10's exact frozen shape when it is not. Once `mixle.task.catalog` lands
# everywhere, this `try/except` collapses to the plain import -- no call site below needs to change.
try:  # pragma: no cover - exercised by whichever branch is importable in a given environment
    from mixle.task.catalog import CatalogEntry, ToolCatalog
except ImportError:  # pragma: no cover
    from dataclasses import dataclass as _dataclass

    @_dataclass(frozen=True)
    class CatalogEntry:  # type: ignore[no-redef]
        id: str
        schema: dict[str, Any]
        owner: str
        cost: float = 0.0
        reliability: float = 1.0
        verifier: str | None = None

    class ToolCatalog:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._entries: dict[str, CatalogEntry] = {}

        def register(self, entry: CatalogEntry) -> None:
            self._entries[entry.id] = entry

        def get(self, entry_id: str) -> CatalogEntry | None:
            return self._entries.get(entry_id)

        def all(self) -> list[CatalogEntry]:
            return list(self._entries.values())


SIMULATE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenario": {
            "type": "object",
            "description": "IC-11 Scenario: {steps: [{op, inputs_ref, params}], couplings, provenance}",
        }
    },
    "required": ["scenario"],
}

_TOOL_CATALOG = ToolCatalog()
_TOOL_CATALOG.register(
    CatalogEntry(
        id="simulate", schema=SIMULATE_TOOL_SCHEMA, owner="physics", cost=1.0, reliability=1.0, verifier="physical"
    )
)


def get_tool_catalog() -> ToolCatalog:
    """The IC-10 tool catalog with the ``simulate`` entry registered, for the router (M3) to enumerate."""
    return _TOOL_CATALOG


# --- P4: simulation UQ + surrogate acceleration --------------------------------------------------
#
# Two additive capabilities on top of the P1/P2 scenario runner, neither of which touches `simulate`:
#
# `propagate_uq` answers "how uncertain is this what-if", not just "what is the point estimate": it
# draws `n` samples from an IC-1 `Posterior` over the scenario's leading input, re-runs the whole
# (possibly coupled) scenario once per draw through the existing `simulate` entry point, and reports
# the output ensemble's per-node central credible interval -- honest Monte-Carlo forward UQ, no new
# numerics beyond that pushforward.
#
# `register_surrogate` answers "how do I make this what-if fast" without silently trading away
# correctness: it distills an E6 `Surrogate` for an expensive `teacher` forward and registers a
# wrapper under `op` that answers from the surrogate when the calibrated gate says the input is
# trustworthy, and transparently escalates to the real `teacher` otherwise -- the E6 cascade, wired
# into the `_FORWARDS` registry so `simulate`/`propagate_uq` dispatch to it exactly like any other op.

_SURROGATES: dict[str, Surrogate] = {}


def register_surrogate(op: str, *, teacher: Callable, sampler: Callable, budget: int, seed: int = 0) -> None:
    """Distil a fast surrogate of forward ``op`` (E6 recipe) and register it under ``op`` behind a
    calibrated deferral gate.

    ``teacher(x) -> y`` is the expensive forward being replaced (a scalar or array-valued function of
    a plain numeric input, the same shape ``distill_forward`` already expects); ``sampler(n, rng) ->
    Sequence`` draws candidate inputs to label. The registered wrapper forward reads its physical
    input ``x`` from the ``"x"`` array of the artifact at ``inputs_ref`` (the same content-addressed
    convention every other forward reads its state from) and returns ``{"value": ...}``: the fast
    ``surrogate.predict(x)`` when ``not surrogate.defer(x)``, or the exact ``teacher(x)`` when the
    calibrated gate defers -- interactive answers on the easy majority, honest escalation on the hard
    tail, never a silently-wrong number.
    """
    surrogate = distill_forward(teacher, sampler, budget=budget, seed=seed)
    _SURROGATES[op] = surrogate

    def _cascade_forward(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
        store_dir = params["_store_dir"]
        x = np.asarray(read_result_artifact(inputs_ref, store_dir=store_dir)["x"], dtype=float)
        y = teacher(x) if surrogate.defer(x) else surrogate.predict(x)
        return {"value": np.atleast_1d(np.asarray(y, dtype=float))}

    register_forward(op, _cascade_forward)


def propagate_uq(scenario: Scenario, input_posterior: Posterior, *, n: int, rng) -> SimResult:
    """Push ``n`` IC-1 posterior draws of the leading input through the scenario forwards.

    Draws ``input_posterior.samples(n, rng)`` (IC-1, a ``(n, d)`` matrix in physical units), writes
    each draw as an ``"x"`` artifact, and re-runs the *entire* scenario (coupled or not) once per draw
    through the existing ``simulate`` entry point with the leading step's ``inputs_ref`` rewritten
    onto that draw -- the same forward dispatch every other caller of this module goes through, so a
    surrogate registered via ``register_surrogate`` is exercised exactly as it would be for a single
    what-if. ``SimResult.result_ref`` is the ensemble mean (a genuine content-hashed artifact, so it
    can be read back like any other result); ``SimResult.uncertainty`` maps each output array key to
    its per-node ``(lo, hi)`` central credible interval at :data:`UQ_LEVEL` mass, computed empirically
    from the output ensemble -- no analytic UQ, just the Monte-Carlo pushforward the work order calls
    for.
    """
    if n <= 0:
        raise ValueError("propagate_uq needs n >= 1 posterior draws")
    draws = np.atleast_2d(np.asarray(input_posterior.samples(n, rng), dtype=float))
    store_dir = _default_store_dir()
    leading = scenario.steps[0]

    member_refs: list[str] = []
    member_arrays: list[dict[str, np.ndarray]] = []
    for draw in draws:
        x_ref = write_result_artifact(
            {"x": np.asarray(draw, dtype=float)},
            grid={"shape": [int(np.asarray(draw).shape[0])]},
            units="",
            provenance={"op": "propagate_uq:input-draw"},
            store_dir=store_dir,
        )
        draw_steps = [ScenarioStep(op=leading.op, inputs_ref=x_ref, params=leading.params), *scenario.steps[1:]]
        draw_scenario = Scenario(steps=draw_steps, couplings=scenario.couplings, provenance=scenario.provenance)
        member_result = simulate(draw_scenario)
        member_refs.append(member_result.result_ref)
        member_arrays.append(read_result_artifact(member_result.result_ref, store_dir=store_dir))

    keys = member_arrays[0].keys()
    stacked = {k: np.stack([m[k] for m in member_arrays], axis=0) for k in keys}
    alpha = (1.0 - UQ_LEVEL) / 2.0
    uncertainty = {k: (np.quantile(v, alpha, axis=0), np.quantile(v, 1.0 - alpha, axis=0)) for k, v in stacked.items()}
    mean_arrays = {k: v.mean(axis=0) for k, v in stacked.items()}

    ref = write_result_artifact(
        mean_arrays,
        grid={"shape": list(next(iter(mean_arrays.values())).shape)},
        units="",
        provenance={"op": "propagate_uq", "n": n, "member_refs": member_refs, **scenario.provenance},
        store_dir=store_dir,
    )
    return SimResult(
        result_ref=ref,
        uncertainty=uncertainty,
        provenance={"op": "propagate_uq", "n": n, "level": UQ_LEVEL, "member_refs": member_refs},
    )
