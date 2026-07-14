"""IC-11 — Scenario & forward-simulation service (workstream P1).

The single surface a frontier model drives for physics what-ifs: :func:`simulate` takes a
:class:`Scenario` (an ordered list of :class:`ScenarioStep`, plus any inter-step couplings) and returns a
content-hashed :class:`SimResult`. Every mixle_pde forward operator (dynamics, wave, flow, EM,
poroelastic, ...) registers itself under a short ``op`` name via :func:`register_forward` -- "register,
don't branch", the same precedent already used by :func:`mixle_pde.dynamics.register_dynamics_operator` --
so :func:`simulate` never special-cases a physics family; it only knows how to look an ``op`` up and
dispatch to it.

Results are persisted as content-addressable artifacts (:func:`write_result_artifact` /
:func:`read_result_artifact`), hashed with the IC-2 frozen rule (:func:`mixle_pde.io.artifacts.sha256_of_arrays`)
so a result is de-duplicable and independently verifiable, exactly like an IC-2 field-posterior artifact --
this module reuses that hashing rule directly rather than waiting on ``save_posterior``, which serialises a
different object (a fitted :class:`~mixle_pde.latent.PosteriorField3D`), not a plain forward's output arrays.

Coupled (multi-step, DAG) scenarios are out of scope here: ``scenario.couplings`` non-empty delegates to
:func:`_run_coupled_dag`, a stub P2 fills in. This module only drives the single-step, coupling-free path.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from mixle_pde.dynamics import AdvectionDiffusionOperator
from mixle_pde.io.artifacts import sha256_of_arrays

__all__ = [
    "Scenario",
    "ScenarioStep",
    "SimResult",
    "simulate",
    "register_forward",
    "write_result_artifact",
    "read_result_artifact",
    "get_tool_catalog",
]


# ---------------------------------------------------------------------------
# IC-11 frozen dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ScenarioStep:
    op: str  # "wave|flow|em|transport|dispersion|poroelastic|climate_rcm|exposure|habitat"
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


# ---------------------------------------------------------------------------
# Result-artifact store: content-addressed, IC-2-header-shaped, but for plain forward outputs
# (not a `Posterior`, so it does not go through `io.artifacts.save_posterior`/`load_posterior`).
# ---------------------------------------------------------------------------
RESULT_ARTIFACT_SCHEMA = "mixle_pde.sim_result/v1"
RESULT_HEADER_KEYS = ("schema", "content_hash", "crs", "grid", "units", "provenance", "created")

DEFAULT_STORE_DIR = os.path.join(tempfile.gettempdir(), "mixle_pde_sim_store")


def _store_dir() -> str:
    """The artifact store directory: ``$MIXLE_PDE_SIM_STORE_DIR`` if set, else :data:`DEFAULT_STORE_DIR`.

    `simulate`'s signature is frozen to a single `scenario` argument (IC-11), so the store location is a
    process-wide setting rather than a call parameter -- resolved fresh on every call (not cached at import
    time) so a test can point it at a tmp directory via the environment before dispatching.
    """
    return os.environ.get("MIXLE_PDE_SIM_STORE_DIR") or DEFAULT_STORE_DIR


def _resolve_artifact_path(ref: str, store_dir: str) -> str:
    """Resolve `ref` -- a bare content-hash living in `store_dir`, or a filesystem path (absolute, or
    relative with a path separator, or simply an existing `<ref>.npz`) -- to the `.npz`-less base path."""
    candidate = ref[: -len(".npz")] if ref.endswith(".npz") else ref
    if os.path.isabs(candidate) or os.sep in candidate or os.path.exists(candidate + ".npz"):
        return candidate
    return os.path.join(store_dir, candidate)


def write_result_artifact(
    arrays: dict[str, np.ndarray], *, grid: dict, units: str, provenance: dict, store_dir: str
) -> str:
    """Hash `arrays` with the frozen IC-2 rule and write ``{store_dir}/{ref}.npz`` + ``{ref}.json``
    (IC-2 header keys). Content-addressed: writing the same arrays twice reproduces the same `ref` and
    the same bytes (idempotent), so a re-run of an identical step never grows the store."""
    digest = sha256_of_arrays(arrays)
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, digest)
    np.savez(f"{path}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    header = {
        "schema": RESULT_ARTIFACT_SCHEMA,
        "content_hash": digest,
        "crs": grid.get("crs"),
        "grid": grid,
        "units": units,
        "provenance": provenance,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    import json

    with open(f"{path}.json", "w") as f:
        json.dump(header, f, indent=2, sort_keys=True, default=str)
    return digest


def read_result_artifact(ref: str, *, store_dir: str) -> dict[str, np.ndarray]:
    """Load the arrays of a result artifact written by :func:`write_result_artifact` (or any raw ``.npz``
    fixture) from `ref` -- a bare content-hash resolved against `store_dir`, or a filesystem path."""
    path = _resolve_artifact_path(ref, store_dir)
    npz_path = f"{path}.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"no result artifact at {npz_path!r} (ref={ref!r}, store_dir={store_dir!r})")
    with np.load(npz_path) as npz:
        return {k: npz[k] for k in npz.files}


def _primary_field(arrays: dict[str, np.ndarray], key: str = "field") -> np.ndarray:
    """Pick the array a scalar-field forward should evolve: `key` if present, else the sole array."""
    if key in arrays:
        return np.asarray(arrays[key], dtype=float)
    if len(arrays) == 1:
        return np.asarray(next(iter(arrays.values())), dtype=float)
    raise KeyError(f"expected a {key!r} array (or exactly one array) in the input artifact; got {sorted(arrays)}")


def _default_grid(params: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """A generic `grid` header for a result artifact: output array shapes plus any grid-shaped params."""
    grid: dict[str, Any] = {"shapes": {k: list(np.shape(v)) for k, v in arrays.items()}}
    for key in ("n", "shape", "spacing", "length"):
        if key in params:
            grid[key] = params[key]
    return grid


def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)


# ---------------------------------------------------------------------------
# Forward registry -- "register, don't branch" (dynamics.register_dynamics_operator precedent)
# ---------------------------------------------------------------------------
_FORWARDS: dict[str, Callable[[str, dict], dict[str, np.ndarray]]] = {}


def register_forward(op: str, fn: Callable[[str, dict], dict[str, np.ndarray]]) -> None:
    """Register a forward operator under an `op` name so `simulate` and the MCP tool can dispatch to it."""
    if not callable(fn):
        raise TypeError("forward must be callable")
    _FORWARDS[op] = fn


def available_forwards() -> list[str]:
    """The sorted names of every registered forward operator."""
    return sorted(_FORWARDS)


# ---------------------------------------------------------------------------
# Built-in pde forwards
# ---------------------------------------------------------------------------
def _forward_advection_diffusion(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The transport/dispersion forward: step `AdvectionDiffusionOperator` `params["steps"]` times.

    Loads the initial scalar field from `inputs_ref` (key "field", or the sole array), builds the
    one-step transition matrix (implicit Euler by default -- unconditionally stable), and applies it
    `steps` times.
    """
    data = read_result_artifact(inputs_ref, store_dir=_store_dir())
    field0 = _primary_field(data)
    n = int(params.get("n", field0.shape[0]))
    if field0.shape[0] != n:
        raise ValueError(f"input field length {field0.shape[0]} does not match params['n']={n}")
    op = AdvectionDiffusionOperator(
        diffusivity=float(params["diffusivity"]),
        velocity=float(params["velocity"]),
        n=n,
        length=float(params.get("length", 1.0)),
        bc=params.get("bc", "periodic"),
        scheme=params.get("scheme", "implicit"),
    )
    dt = float(params.get("dt", 0.05))
    transition = op.transition_matrix(dt)
    u = field0.copy()
    for _ in range(int(params.get("steps", 1))):
        u = transition @ u
    return {"field": u}


def _forward_wave(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The wave forward: step `WaveEquation2D` `params["steps"]` times (zero source unless `c2`/`u0`/`w0` given)."""
    from mixle_pde.ops import make_ops
    from mixle_pde.wave import WaveEquation2D

    data = read_result_artifact(inputs_ref, store_dir=_store_dir())
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


def _forward_flow(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The flow forward: step `NavierStokes2D` `params["steps"]` times from an initial vorticity field."""
    from mixle_pde.flow import NavierStokes2D
    from mixle_pde.ops import make_ops

    data = read_result_artifact(inputs_ref, store_dir=_store_dir())
    omega0 = _primary_field(data, key="omega")
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


def _forward_em(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The em forward: 2-D magnetotelluric TE-mode sounding (`mt_2d_te`) by default; `params["mode"] ==
    "fdtd"` steps a 3-D `Maxwell3D` cavity instead, for the transient-field regime."""
    data = read_result_artifact(inputs_ref, store_dir=_store_dir())
    if params.get("mode") == "fdtd":
        from mixle_pde.maxwell import Maxwell3D
        from mixle_pde.ops import make_ops

        n = int(params["n"])
        nc = n**3
        zeros = lambda: np.zeros(nc)  # noqa: E731
        Ex = data.get("Ex", zeros())
        Ey = data.get("Ey", zeros())
        Ez = data.get("Ez", zeros())
        Hx = data.get("Hx", zeros())
        Hy = data.get("Hy", zeros())
        Hz = data.get("Hz", zeros())
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

    log_sigma = _primary_field(data, key="log_sigma")
    shape = tuple(int(s) for s in params["shape"])
    rho_a, phase = mt_2d_te(
        log_sigma,
        shape,
        freq=float(params["freq"]),
        spacing=params.get("spacing", 1.0),
        sigma_ref=float(params.get("sigma_ref", 1.0)),
    )
    return {"rho_a": _to_numpy(rho_a), "phase": _to_numpy(phase)}


def _forward_poroelastic(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The poroelastic forward: step `BiotPoroelastic1D` `params["steps"]` times (dewatering / subsidence style)."""
    from mixle_pde.ops import make_ops
    from mixle_pde.poroelastic import BiotPoroelastic1D

    data = read_result_artifact(inputs_ref, store_dir=_store_dir())
    n = int(params["n"]) if "n" in params else len(next(iter(data.values())))
    zeros = lambda: np.zeros(n)  # noqa: E731
    v0 = data.get("v", zeros())
    q0 = data.get("q", zeros())
    sigma0 = data.get("sigma", zeros())
    pf0 = data.get("pf", zeros())

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
    return {"v": _to_numpy(v), "q": _to_numpy(q), "sigma": _to_numpy(sigma), "pf": _to_numpy(pf)}


def _forward_model(inputs_ref: str, params: dict) -> dict[str, np.ndarray]:
    """The model forward: sample `n` draws from a fitted core generative model via
    `mixle.inference.simulate.simulate(model).run(n, seed=...)`.

    Unlike the mesh-based pde forwards, a fitted model is not a numpy-array artifact, so it is passed
    in-process as `params["model"]` (a live Python object) rather than resolved from `inputs_ref`.
    """
    from mixle.inference.simulate import simulate as core_simulate

    model = params.get("model")
    if model is None:
        raise ValueError("op 'model' requires params['model'] (a fitted generative model object)")
    sim = core_simulate(model)
    records = sim.run(int(params.get("n", 100)), seed=int(params.get("seed", 0)))
    return {"samples": np.asarray(records, dtype=float)}


register_forward("transport", _forward_advection_diffusion)
register_forward("dispersion", _forward_advection_diffusion)
register_forward("wave", _forward_wave)
register_forward("flow", _forward_flow)
register_forward("em", _forward_em)
register_forward("poroelastic", _forward_poroelastic)
register_forward("model", _forward_model)
# "climate_rcm"/"exposure"/"habitat" are open slots L7/K/N register -- not this task's forwards.


# ---------------------------------------------------------------------------
# The IC-11 surface
# ---------------------------------------------------------------------------
def simulate(scenario: Scenario) -> SimResult:
    """Run a (possibly coupled multiphysics) forward scenario; return a content-hashed result handle.

    Every mixle_pde forward registers an `op` here via `register_forward(op, fn)`. This is the single
    surface a frontier model drives for what-if. A non-empty `scenario.couplings` delegates to
    `_run_coupled_dag` (P2); otherwise `scenario` must carry exactly one step, which is dispatched to its
    registered forward and its output persisted as a content-addressed artifact.
    """
    if scenario.couplings:
        return _run_coupled_dag(scenario)
    if len(scenario.steps) != 1:
        raise ValueError(
            "simulate() without couplings requires exactly one ScenarioStep; a multi-step scenario needs "
            "scenario.couplings to route through _run_coupled_dag (P2)."
        )
    step = scenario.steps[0]
    fn = _FORWARDS.get(step.op)
    if fn is None:
        raise KeyError(f"no forward registered for op {step.op!r}; registered ops: {available_forwards()}")

    arrays = fn(step.inputs_ref, step.params)

    provenance = dict(scenario.provenance)
    provenance.update({"op": step.op, "inputs_ref": step.inputs_ref, "params": dict(step.params)})
    result_ref = write_result_artifact(
        arrays,
        grid=_default_grid(step.params, arrays),
        units=str(step.params.get("units", "")),
        provenance=provenance,
        store_dir=_store_dir(),
    )
    return SimResult(result_ref=result_ref, uncertainty=None, provenance={"op": step.op, "content_hash": result_ref})


def _run_coupled_dag(scenario: Scenario) -> SimResult:
    raise NotImplementedError("P2")


# ---------------------------------------------------------------------------
# IC-10 catalog registration -- "so the router (M3) sees it uniformly"
# ---------------------------------------------------------------------------
# `mixle.task.catalog` is IC-10's frozen module (work-plan Wave 0). As of this change it has not yet
# landed in core mixle (no `mixle/task/catalog.py` on `release/0.8.0`), so importing it unconditionally
# would make this module fail to import and block every P1 consumer on an unrelated, not-yet-landed
# contract. Rather than touch the frozen contract or block on it, degrade gracefully: use the real
# `CatalogEntry`/`ToolCatalog` when importable, and fall back to a private shim reproducing IC-10's exact
# frozen shape (same dataclass fields, same `register`/`get`/`all` surface) when it is not. Once
# `mixle.task.catalog` lands, this whole `try/except` collapses to the plain import -- no call site below
# needs to change.
try:  # pragma: no cover - exercised by whichever branch is importable in a given environment
    from mixle.task.catalog import CatalogEntry, ToolCatalog
except ImportError:  # pragma: no cover - IC-10 not yet landed in core mixle
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
_TOOL_CATALOG.register(CatalogEntry(id="simulate", schema=SIMULATE_TOOL_SCHEMA, owner="physics", verifier="physical"))


def get_tool_catalog() -> ToolCatalog:
    """The IC-10 tool catalog with the `simulate` entry registered, for the router (M3) to enumerate."""
    return _TOOL_CATALOG
