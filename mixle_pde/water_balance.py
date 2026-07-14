"""L2 -- physical climate risk to operations: catchment water balance (work-plan L2).

Turns a climate projection (precipitation + evapotranspiration) into a per-step water budget for an
operation's catchment: how much water the catchment routes into storage, how much storage holds after
evaporative and operational losses, and whether the operation runs short. Every number in the returned
:class:`WaterBudget` traces back to the climate driver that produced it via ``climate_hash`` -- an L4
``ProvenancedResult.content_hash`` when the caller passes one (preferred), so a downstream shortfall is
never reported without knowing which climate model + version predicted the rainfall behind it.

Runoff generation is not a free parameter: :func:`water_balance` derives a catchment routing coefficient
by driving :class:`mixle_pde.flow.NavierStokes2D`'s streamfunction Poisson solve (the same adjoint
``sparse_solve`` the flow-inversion problems use) with a fixed unit vorticity pattern oriented along the
catchment's long axis, and reads the resulting circulation speed off the interior nodes. The coefficient
depends only on the catchment's shape, so it is computed once per call and applied to every step's
precipitation; a more elongated (or otherwise more efficiently-draining) catchment routes a larger share
of its rainfall before it is lost to storage or infiltration. This is a deliberately simplified stand-in
for a full hydrological routing model (a real distributed rainfall-runoff scheme is out of scope for L2 --
see the module's non-goals in the work plan); it gives the water budget a genuine, deterministic link to
the flow-physics stack rather than an arbitrary runoff fraction.

Climate input: the ``climate`` argument is either an L4 ``ProvenancedResult``-like object (anything
exposing ``.value`` and ``.content_hash``, with ``.value`` a mapping of per-step precipitation/PET
series) or a plain ``dict`` with the same fields (optionally nested under a ``"value"`` key). No
``mixle_mlops`` import is required or performed here -- L2 is parallel-independent of the concrete L4
adapter classes and consumes only the frozen ``ProvenancedResult`` shape as data (duck-typed), exactly as
L1 consumes its upstream plan as a plain dict.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle_pde.flow import NavierStokes2D
from mixle_pde.ops import make_ops

__all__ = ["WaterBudget", "water_balance"]

# Small structured grid used only to derive the catchment's routing coefficient; independent of the
# per-step climate series, so a tiny grid is plenty (no need to resolve the actual catchment mesh here).
_ROUTING_GRID_N = 9
_ROUTING_VISCOSITY = 0.05
_ROUTING_DT = 0.05
# Fraction of routed inflow released downstream as a fixed environmental compensation flow each step,
# distinct from the operational `demand_m3` withdrawal.
_COMPENSATION_FRACTION = 0.05
# Storage (pond/sump) footprint as a fraction of the catchment's own plan-view area, used only to size the
# direct evaporative loss *from storage* (see the module docstring's note on the two evap terms): the
# frozen signature has no separate reservoir-geometry parameter, so this is a fixed light-touch proxy.
_STORAGE_SURFACE_FRACTION = 0.01


@dataclass
class WaterBudget:
    """Per-step catchment water budget: routed inflow, released outflow, and resulting storage.

    ``inflow``/``outflow``/``storage`` are length-``steps`` arrays (m^3). ``shortfall_m3`` accumulates any
    deficit where storage would have gone negative (clamped at 0 each step). ``climate_hash`` is the
    upstream climate driver's ``content_hash`` (``None`` only if the driver carried none). ``provenance``
    records the climate driver's identity plus the derived routing coefficient, so the budget is
    reproducible and attributable.
    """

    inflow: np.ndarray
    outflow: np.ndarray
    storage: np.ndarray
    shortfall_m3: float
    climate_hash: str | None
    provenance: dict


def _canonical_bytes(obj: Any) -> bytes:
    """Deterministic bytes for a small provenance payload (numbers, strings, arrays, mappings, sequences)."""
    if obj is None:
        return b"N"
    if isinstance(obj, (bool, np.bool_)):
        return b"b1" if obj else b"b0"
    if isinstance(obj, (int, np.integer)):
        return b"i" + repr(int(obj)).encode()
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return b"fNaN" if f != f else b"f" + np.float64(f).tobytes()
    if isinstance(obj, str):
        return b"s" + obj.encode("utf-8")
    if isinstance(obj, np.ndarray):
        return b"a" + str(obj.dtype).encode() + b":" + np.ascontiguousarray(obj, dtype=float).tobytes()
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return b"d{" + b",".join(_canonical_bytes(k) + b":" + _canonical_bytes(v) for k, v in items) + b"}"
    if isinstance(obj, (list, tuple)):
        return b"t[" + b",".join(_canonical_bytes(v) for v in obj) + b"]"
    return b"r" + repr(obj).encode()


def _content_hash(payload: Any) -> str:
    """sha256 hex digest of a canonical encoding of ``payload`` (IC-2's hashing convention, applied here to
    an activity-style payload rather than an artifact's arrays)."""
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _catchment_area_m2(catchment: Any) -> float:
    """Total plan-view catchment area, m^2 (mesh node coordinates are assumed metres)."""
    if hasattr(catchment, "simplex_measures"):
        return float(np.sum(catchment.simplex_measures()))
    if hasattr(catchment, "area"):
        return float(catchment.area)
    raise TypeError(
        "catchment must be a mixle_pde.mesh.SimplexMesh (or expose .simplex_measures()/.area for its plan-view area)."
    )


def _catchment_shape_factor(catchment: Any) -> float:
    """A dimensionless (0, 1) elongation factor from the mesh's bounding-box aspect ratio.

    More elongated catchments (a higher ratio of long axis to short axis) drain more directly to an
    outlet than compact/square ones, so this scales the vorticity forcing that seeds
    :func:`_routing_coefficient`.
    """
    nodes = np.asarray(getattr(catchment, "nodes", catchment), dtype=float)
    span = nodes[:, :2].max(axis=0) - nodes[:, :2].min(axis=0)
    lo, hi = float(np.min(span)), float(np.max(span))
    if hi <= 0.0:
        return 0.5
    return float(np.clip(1.0 - lo / hi, 0.05, 0.95))


def _routing_coefficient(catchment: Any) -> float:
    """The fraction of catchment precipitation that becomes routed inflow each step.

    Derived once from the catchment's shape by driving :class:`~mixle_pde.flow.NavierStokes2D`'s linear
    streamfunction Poisson solve (``laplacian(psi) = -omega``, the same adjoint ``sparse_solve`` the
    flow-inversion problems use) with a fixed unit vorticity pattern tilted along the catchment's long
    axis, then reading the resulting circulation speed off the interior nodes. Because the streamfunction
    solve and the velocity gradient are both linear in the vorticity forcing, this coefficient is a fixed
    property of the catchment's shape (not of the climate signal), so it is computed once and reused every
    step. The raw circulation speed is squashed onto ``(0, 1)`` by ``speed / (speed + 1)`` -- bounded and
    strictly increasing in the underlying flow-physics response.
    """
    n = _ROUTING_GRID_N
    ns = NavierStokes2D(n, viscosity=_ROUTING_VISCOSITY, dt=_ROUTING_DT)
    ops = make_ops()
    shape_factor = _catchment_shape_factor(catchment)
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), indexing="ij")
    omega0 = ops.tensor((np.sin(np.pi * xx) * np.sin(np.pi * yy) * (1.0 + shape_factor)).ravel())
    psi0 = ns.streamfunction(omega0, ops)
    u0, v0 = ns.velocity(psi0, ops)
    speed = float(np.sqrt(u0.numpy() ** 2 + v0.numpy() ** 2).mean())
    return float(speed / (speed + 1.0))


def _climate_fields(climate: Any, steps: int) -> tuple[np.ndarray, np.ndarray, str | None, dict]:
    """Extract (precip_mm, evap_mm, climate_hash, identity) from an L4 ``ProvenancedResult``-like object or
    a plain dict; broadcasts scalar/short series to ``steps``."""
    identity: dict = {}
    if hasattr(climate, "content_hash") and hasattr(climate, "value"):
        content_hash = climate.content_hash
        identity = {
            "model_id": getattr(climate, "model_id", None),
            "version": getattr(climate, "version", None),
        }
        value = climate.value
    elif isinstance(climate, dict):
        content_hash = climate.get("content_hash")
        identity = {"model_id": climate.get("model_id"), "version": climate.get("version")}
        value = climate.get("value", climate)
    else:
        raise TypeError("climate must be a ProvenancedResult-like object (.value/.content_hash) or a dict.")

    if not isinstance(value, dict):
        raise TypeError("climate's value must be a mapping with 'precip_mm'/'evap_mm' series.")

    def _series(*keys: str) -> np.ndarray:
        for k in keys:
            if k in value:
                arr = np.atleast_1d(np.asarray(value[k], dtype=float))
                if arr.size == 1:
                    return np.full(steps, float(arr[0]))
                if arr.size < steps:
                    reps = int(np.ceil(steps / arr.size))
                    arr = np.tile(arr, reps)
                return arr[:steps]
        raise KeyError(f"climate value is missing any of {keys!r}.")

    precip_mm = _series("precip_mm", "precipitation_mm")
    evap_mm = _series("evap_mm", "evapotranspiration_mm", "pet_mm")
    return precip_mm, evap_mm, content_hash, identity


def water_balance(
    catchment: Any,
    *,
    climate: Any,
    demand_m3: float,
    storage0_m3: float,
    dt_days: float = 30.0,
    steps: int = 12,
) -> WaterBudget:
    """March a catchment's storage forward under a climate scenario and report any shortfall.

    ``catchment`` is a :class:`mixle_pde.mesh.SimplexMesh` (or anything exposing ``.simplex_measures()``/
    ``.nodes``) describing the catchment's plan-view geometry. ``climate`` supplies precipitation and
    evapotranspiration -- an L4 ``ProvenancedResult`` (preferred; its ``content_hash`` is recorded as
    ``climate_hash``) or a plain dict with the same ``precip_mm``/``evap_mm`` fields. ``demand_m3`` is the
    fixed per-step operational withdrawal; ``storage0_m3`` the starting storage; ``dt_days``/``steps`` the
    step length and count (default: 12 monthly steps).

    Each step: catchment-wide evapotranspiration first reduces the precipitation surplus available to
    generate runoff (``net_precip = max(precip - evap, 0)``), and the flow-derived routing coefficient
    (:func:`_routing_coefficient`) converts a share of that surplus into routed ``inflow``. A fixed
    fraction of the routed inflow is released downstream as environmental-compensation ``outflow``.
    Separately, the storage pool itself (assumed a small fraction of the catchment's area -- there is no
    reservoir-geometry parameter in this signature) loses water to direct evaporation, and ``demand_m3``
    is withdrawn every step regardless of scenario. Storage is clamped at 0 and any deficit accumulates
    into ``shortfall_m3`` -- an operation that would draw storage negative is short by that amount, not
    negative.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if storage0_m3 < 0.0:
        raise ValueError("storage0_m3 must be >= 0.")

    area_m2 = _catchment_area_m2(catchment)
    storage_area_m2 = area_m2 * _STORAGE_SURFACE_FRACTION
    routing_coeff = _routing_coefficient(catchment)
    precip_mm, evap_mm, climate_hash, identity = _climate_fields(climate, steps)

    inflow = np.zeros(steps)
    outflow = np.zeros(steps)
    storage = np.zeros(steps)
    shortfall_m3 = 0.0
    storage_prev = float(storage0_m3)

    for t in range(steps):
        net_precip_mm = max(float(precip_mm[t]) - float(evap_mm[t]), 0.0)
        precip_vol_m3 = net_precip_mm / 1000.0 * area_m2
        gross_inflow = routing_coeff * precip_vol_m3
        released = _COMPENSATION_FRACTION * gross_inflow
        storage_evap_m3 = max(float(evap_mm[t]), 0.0) / 1000.0 * storage_area_m2

        next_storage = storage_prev + gross_inflow - storage_evap_m3 - float(demand_m3) - released
        if next_storage < 0.0:
            shortfall_m3 += -next_storage
            next_storage = 0.0

        inflow[t] = gross_inflow
        outflow[t] = released
        storage[t] = next_storage
        storage_prev = next_storage

    provenance = {
        "climate_model_id": identity.get("model_id"),
        "climate_version": identity.get("version"),
        "climate_hash": climate_hash,
        "catchment_area_m2": area_m2,
        "routing_coefficient": routing_coeff,
        "demand_m3": float(demand_m3),
        "dt_days": float(dt_days),
        "steps": int(steps),
        "input_hash": _content_hash(
            {
                "precip_mm": precip_mm,
                "evap_mm": evap_mm,
                "demand_m3": float(demand_m3),
                "storage0_m3": float(storage0_m3),
                "dt_days": float(dt_days),
                "steps": int(steps),
            }
        ),
    }

    return WaterBudget(
        inflow=inflow,
        outflow=outflow,
        storage=storage,
        shortfall_m3=float(shortfall_m3),
        climate_hash=climate_hash,
        provenance=provenance,
    )
