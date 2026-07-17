"""IC-3 -- physics tool JSON-schemas and handlers (work-plan E4). The model-callable surface over the
inversion engine: four tools, each a frozen JSON-schema ``input_schema`` (``PHYSICS_TOOL_SCHEMAS``,
verbatim from the IC-3 contract -- never edit these shapes) plus a real handler wired to the existing
Earth-inversion machinery. ``mixle-mlops``' ``mcp/physics_tools.py`` wraps each into an
``mcp.server.Tool`` and registers it in ``build_model_tools``/``ToolRegistry`` (E4, mlops half).

Scope (work-plan E4, 2pw, "no new physics -- uses the existing inversions"):

* ``run_inversion``/``forward_model`` wire the two FIXED-JACOBIAN (linear) forward operators
  (:func:`mixle_pde.observations.gravity_forward_operator`,
  :func:`mixle_pde.observations.magnetics_forward_operator`) through
  :func:`mixle_pde.field_inversion.linear_gaussian_invert`, the exact closed-form linear-Gaussian
  posterior. The remaining IC-3 modality enum values (``dc``, ``ip``, ``mt``, ``csem``, ``aem``,
  ``seismic``) have STATE-DEPENDENT (nonlinear) forward operators today -- fitting those needs the
  separate Gauss-Newton/IRLS path (:mod:`mixle_pde.field_gauss_newton`,
  :mod:`mixle_pde.blocky_priors`) over a different grid abstraction, which is out of this ticket's
  no-new-physics, 2pw budget; both functions raise a clear, actionable error naming the modalities
  that *are* wired rather than silently no-op-ing.
* Only the ``"smooth"`` prior (:class:`mixle_pde.field_inversion.FieldGaussianPrior`) is wired for the
  same reason: ``"blocky"``/``"compact"``/``"anisotropic"`` live behind
  :func:`mixle_pde.blocky_priors.blocky_invert`, which runs over
  :func:`mixle_pde.geophysics.regularized_gauss_newton`'s regular-grid-shape model rather than
  :mod:`mixle_pde.field_inversion`'s ``Field3D``/``ForwardOperatorRegistry`` path this tool uses.
* ``query_posterior`` is fully general: every one of the six IC-3 ``query`` values works against any
  posterior a caller has saved, dispatching ``region_mass``/``prob_exceed``/``net_pay``/``drill_target``
  to the real :mod:`mixle_pde.decision_quantities` (A5, IC-8) surface, and ``marginal``/``section``
  directly against the IC-1 ``Posterior`` shape (:meth:`~mixle_pde.latent.PosteriorField3D.credible_interval`,
  :meth:`~mixle_pde.latent.PosteriorField3D.slice`). Every branch ALWAYS returns ``prior_dominated``
  (work-plan A2), defaulting to ``False`` (unknown, not claimed data-driven) when the caller supplies no
  ``prior_var``/``posterior_var``, mirroring :mod:`mixle_pde.decision_quantities`'s own honesty default.
* ``gassmann`` reuses :func:`mixle_pde.rock_physics.fluid_substitute` unchanged, adding Monte Carlo
  uncertainty propagation from caller-supplied per-input standard deviations -- no new rock-physics math.

``run_inversion``/``query_posterior``/``forward_model`` persist/reload posteriors through
``mixle_pde.io.artifacts`` (IC-2, workstream E2), lazily imported so this module still imports cleanly
before E2 lands (its PR is open, not yet merged, as of this ticket) -- the same lazy-import-with-a-clear-
message convention :mod:`mixle_pde.env_data` already uses for optional heavy dependencies. ``dataset_ref``
(an observation set) and ``geometry_ref`` (a survey's XYZ) have no artifact format of their own yet in any
frozen contract, so this module defines the minimal JSON bundle shapes ``_load_dataset_bundle``/
``_load_geometry_bundle`` read; the PR notes flag this as a stopgap for whichever ingest task lands a real
dataset-artifact convention.
"""

from __future__ import annotations

import importlib
import json
import operator as _operator
import os
import tempfile
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from mixle_pde import decision_quantities
from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    SurveyGeometry,
    gravity_forward_operator,
    magnetics_forward_operator,
)
from mixle_pde.rock_physics import fluid_substitute

PHYSICS_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "run_inversion": {
        "type": "object",
        "properties": {
            "dataset_ref": {"type": "string", "description": "content-hashed handle to the observation set"},
            "modality": {
                "type": "string",
                "enum": ["gravity", "magnetics", "dc", "ip", "mt", "csem", "aem", "seismic"],
            },
            "prior": {"type": "string", "enum": ["smooth", "blocky", "compact", "anisotropic"]},
            "config": {"type": "object", "description": "solver/regulariser knobs"},
        },
        "required": ["dataset_ref", "modality", "prior"],
    },
    "query_posterior": {
        "type": "object",
        "properties": {
            "posterior_ref": {"type": "string", "description": "content-hashed handle to a saved posterior (IC-2)"},
            "query": {
                "type": "string",
                "enum": ["region_mass", "prob_exceed", "net_pay", "drill_target", "marginal", "section"],
            },
            "params": {"type": "object"},
        },
        "required": ["posterior_ref", "query"],
    },
    "gassmann": {
        "type": "object",
        "properties": {"inputs": {"type": "object", "description": "moduli/porosity/saturation inputs"}},
        "required": ["inputs"],
    },
    "forward_model": {
        "type": "object",
        "properties": {
            "modality": {"type": "string"},
            "model_ref": {"type": "string"},
            "geometry_ref": {"type": "string"},
        },
        "required": ["modality", "model_ref", "geometry_ref"],
    },
}

# --- linear (fixed-Jacobian) modalities run_inversion/forward_model can actually fit/predict today ---

_LINEAR_MODALITY_BUILDERS: dict[str, Callable[[np.ndarray, np.ndarray, dict[str, Any]], Any]] = {
    "gravity": lambda cells, volumes, cfg: gravity_forward_operator(cells, volumes),
    "magnetics": lambda cells, volumes, cfg: magnetics_forward_operator(
        cells,
        volumes,
        inclination=float(cfg.get("inclination", 60.0)),
        declination=float(cfg.get("declination", 0.0)),
        field_nt=float(cfg.get("field_nt", 50000.0)),
    ),
}

_COMPARATORS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    ">": _operator.gt,
    ">=": _operator.ge,
    "<": _operator.lt,
    "<=": _operator.le,
    "==": _operator.eq,
}

_GASSMANN_FIELDS = (
    "Vp",
    "Vs",
    "rho",
    "phi",
    "K_min",
    "rho_min",
    "K_fl_in",
    "rho_fl_in",
    "K_fl_out",
    "rho_fl_out",
)


def _unsupported_modality(tool: str, modality: str) -> ValueError:
    return ValueError(
        f"modality {modality!r} is not wired through {tool} yet; only {sorted(_LINEAR_MODALITY_BUILDERS)} have a "
        "fixed-Jacobian forward operator today. dc/ip/mt/csem/aem/seismic need the nonlinear Gauss-Newton path "
        "(mixle_pde.field_gauss_newton) instead of the closed-form linear-Gaussian inversion this tool wires -- "
        "out of scope for this ticket's no-new-physics, 2pw budget."
    )


def _artifacts_module() -> Any:
    """Lazily import ``mixle_pde.io.artifacts`` (IC-2, E2); raise a clear, actionable error if it is not
    yet installed/merged rather than failing this whole module's import.

    Uses :func:`importlib.import_module` rather than ``from mixle_pde.io import artifacts``: once the
    real submodule has been imported anywhere in the process, ``mixle_pde.io`` caches it as an
    attribute, and a plain ``from`` import resolves that cached attribute directly -- bypassing a
    test's ``monkeypatch.setitem(sys.modules, "mixle_pde.io.artifacts", fake)`` entirely.
    ``import_module`` always consults ``sys.modules`` first, so it honors the patch.
    """
    try:
        artifacts = importlib.import_module("mixle_pde.io.artifacts")
    except ImportError as exc:
        raise ImportError(
            "run_inversion/query_posterior/forward_model need mixle_pde.io.artifacts (IC-2, workstream E2) for "
            "posterior persistence; install/merge the E2 artifact-I/O module to enable them."
        ) from exc
    return artifacts


def _default_artifact_path() -> str:
    directory = os.environ.get("MIXLE_PDE_TOOLS_ARTIFACT_DIR") or os.path.join(
        tempfile.gettempdir(), "mixle_pde_tool_artifacts"
    )
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, uuid.uuid4().hex)


def _write_json(path: str, obj: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def _load_dataset_bundle(dataset_ref: str, modality: str) -> tuple[Field3D, list[Observation], np.ndarray | None]:
    """Read the minimal JSON observation-set bundle ``dataset_ref`` points to.

    Not an IC (no frozen dataset-artifact contract exists yet); this is the smallest shape that lets
    ``run_inversion`` reconstruct a :class:`~mixle_pde.latent.Field3D` grid and its
    :class:`~mixle_pde.observations.Observation` batch from a single JSON handle. See the module
    docstring's "Notes" pointer.
    """
    with open(dataset_ref) as f:
        bundle = json.load(f)
    grid_spec = bundle["grid"]
    coordinates = np.asarray(grid_spec["coordinates"], dtype=float)
    spacing = grid_spec.get("spacing", 1.0)
    if isinstance(spacing, list):
        spacing = tuple(float(s) for s in spacing)
    cell_volumes = np.asarray(grid_spec["cell_volumes"], dtype=float) if "cell_volumes" in grid_spec else None
    provenance = dict(grid_spec.get("provenance") or {})
    if grid_spec.get("crs") is not None:
        provenance.setdefault("crs", grid_spec["crs"])
    if cell_volumes is not None:
        provenance["cell_volumes"] = cell_volumes.tolist()
    grid = Field3D(
        coordinates=coordinates,
        spacing=spacing,
        units=grid_spec.get("units", ""),
        property_name=grid_spec.get("property_name", modality),
        provenance=provenance,
    )
    observations = [
        Observation(
            kind=modality,
            location=np.asarray(obs_spec["location"], dtype=float),
            value=np.asarray(obs_spec["value"], dtype=float),
            noise_cov=np.asarray(obs_spec["noise_cov"], dtype=float),
            time=obs_spec.get("time"),
            units=obs_spec.get("units", ""),
            provenance=dict(obs_spec.get("provenance") or {}),
            crs=obs_spec.get("crs"),
            modality=modality,
        )
        for obs_spec in bundle.get("observations", [])
    ]
    return grid, observations, cell_volumes


def _load_geometry_bundle(geometry_ref: str) -> SurveyGeometry:
    """Read the minimal JSON survey-geometry bundle ``geometry_ref`` points to (see module docstring)."""
    with open(geometry_ref) as f:
        spec = json.load(f)
    return SurveyGeometry(points=np.asarray(spec["points"], dtype=float), crs=spec.get("crs"))


def _build_prior(prior: str, config: dict[str, Any]) -> FieldGaussianPrior:
    if prior != "smooth":
        raise ValueError(
            f"prior {prior!r} is not wired through run_inversion yet; only 'smooth' "
            "(mixle_pde.field_inversion.FieldGaussianPrior) is supported today. 'blocky'/'compact'/"
            "'anisotropic' need mixle_pde.blocky_priors.blocky_invert's IRLS loop over "
            "mixle_pde.geophysics.regularized_gauss_newton, a different (regular-grid-shape) model than "
            "field_inversion's Field3D/ForwardOperatorRegistry path -- out of scope for this ticket."
        )
    kwargs = {
        k: config[k] for k in ("smoothness_precision", "marginal_precision", "length_scale", "neighbors") if k in config
    }
    return FieldGaussianPrior(**kwargs)


def run_inversion(dataset_ref: str, modality: str, prior: str, config: dict | None = None) -> dict:
    """Fit a posterior; return ``{posterior_ref: str, diagnostics: dict}`` (``posterior_ref`` is content-hashed)."""
    config = dict(config or {})
    if modality not in _LINEAR_MODALITY_BUILDERS:
        raise _unsupported_modality("run_inversion", modality)
    grid, observations, cell_volumes = _load_dataset_bundle(dataset_ref, modality)
    if not observations:
        raise ValueError(f"dataset {dataset_ref!r} has no observations to invert.")
    if cell_volumes is None:
        raise ValueError(f"dataset {dataset_ref!r} is missing grid.cell_volumes, required for modality {modality!r}.")
    prior_obj = _build_prior(prior, config)
    registry = ForwardOperatorRegistry()
    registry.register(_LINEAR_MODALITY_BUILDERS[modality](grid.coordinates, cell_volumes, config))
    posterior = linear_gaussian_invert(grid, observations, registry, prior_obj)

    artifacts = _artifacts_module()
    path = config.get("artifact_path") or _default_artifact_path()
    artifacts.save_posterior(posterior, path)
    return {
        "posterior_ref": path,
        "diagnostics": {
            "modality": modality,
            "prior": prior,
            "n_observations": int(sum(obs.n for obs in observations)),
            "n_cells": int(grid.n),
            "content_hash": artifacts.content_hash(path),
        },
    }


def _optional_array(value: Any) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=float)


def _region_mask(params: dict[str, Any], n: int) -> np.ndarray:
    if "region" in params:
        mask = np.asarray(params["region"], dtype=bool)
        if mask.shape != (n,):
            raise ValueError(f"params.region must have length {n}, got {mask.shape}.")
        return mask
    if "region_index" in params:
        idx = np.asarray(params["region_index"], dtype=int)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask
    raise ValueError("params must include 'region' (boolean mask) or 'region_index' (index list).")


def _rng(params: dict[str, Any]) -> np.random.Generator:
    return np.random.default_rng(params.get("seed"))


def _cell_volumes_for(posterior: Any, params: dict[str, Any]) -> np.ndarray:
    if "cell_volumes" in params:
        return np.asarray(params["cell_volumes"], dtype=float)
    stored = (posterior.grid.provenance or {}).get("cell_volumes")
    if stored is not None:
        return np.asarray(stored, dtype=float)
    raise ValueError("params must include 'cell_volumes' (the saved posterior's grid has none stored).")


def _variance_reduction_flag(params: dict[str, Any], mask: np.ndarray) -> bool:
    """Work-plan A2's honesty flag, computed the same way :mod:`mixle_pde.decision_quantities` does for
    its own quantities: ``False`` (unknown, not claimed data-driven) unless the caller supplies
    ``prior_var``/``posterior_var`` over the full grid, in which case the region's weighted mean
    variance reduction is compared against the same 0.1 threshold."""
    prior_var = _optional_array(params.get("prior_var"))
    posterior_var = _optional_array(params.get("posterior_var"))
    if prior_var is None or posterior_var is None:
        return False
    weights = mask.astype(float)
    support = (weights > 0.0) & (prior_var > 0.0)
    if not np.any(support):
        return False
    reduction = 1.0 - posterior_var[support] / prior_var[support]
    return bool(np.average(reduction, weights=weights[support]) < 0.1)


def _summarize(derived: Any) -> dict[str, Any]:
    lo, hi = derived.credible_interval(0.9)
    return {
        "value": float(derived.mean),
        "distribution": {
            "mean": float(derived.mean),
            "std": float(derived.std),
            "credible_interval_90": [float(lo), float(hi)],
        },
        "prior_dominated": bool(derived.prior_dominated),
    }


def _query_region_mass(posterior: Any, params: dict[str, Any]) -> dict[str, Any]:
    mask = _region_mask(params, posterior.grid.n)
    cell_volumes = _cell_volumes_for(posterior, params)
    dq = decision_quantities.region_mass(
        posterior,
        mask,
        cell_volumes,
        prior_var=_optional_array(params.get("prior_var")),
        posterior_var=_optional_array(params.get("posterior_var")),
    )
    return _summarize(dq)


def _query_prob_exceed(posterior: Any, params: dict[str, Any]) -> dict[str, Any]:
    if "threshold" not in params:
        raise ValueError("query 'prob_exceed' requires params.threshold.")
    mask = _region_mask(params, posterior.grid.n)
    dq = decision_quantities.prob_exceed(
        posterior,
        mask,
        threshold=float(params["threshold"]),
        n=int(params.get("n", 4096)),
        rng=_rng(params),
        prior_var=_optional_array(params.get("prior_var")),
        posterior_var=_optional_array(params.get("posterior_var")),
    )
    return _summarize(dq)


def _query_net_pay(posterior: Any, params: dict[str, Any]) -> dict[str, Any]:
    for key in ("column_index", "sat_cut", "thickness"):
        if key not in params:
            raise ValueError(f"query 'net_pay' requires params.{key}.")
    dq = decision_quantities.net_pay(
        posterior,
        np.asarray(params["column_index"], dtype=int),
        sat_cut=float(params["sat_cut"]),
        thickness=np.asarray(params["thickness"], dtype=float),
        n=int(params.get("n", 4096)),
        rng=_rng(params),
        prior_var=_optional_array(params.get("prior_var")),
        posterior_var=_optional_array(params.get("posterior_var")),
    )
    return _summarize(dq)


def _criteria_predicate(spec: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt IC-3's free-form ``params.criteria`` dict into the callable
    :func:`mixle_pde.decision_quantities.drill_target_prob` needs (a thin shim -- A5's real
    ``criteria`` argument is a callable, not the dict IC-8's illustrative stub used)."""
    clauses = spec.get("clauses", [spec])

    def predicate(draw: np.ndarray) -> np.ndarray:
        mask = np.ones(draw.shape, dtype=bool)
        for clause in clauses:
            if "min" in clause or "max" in clause:
                lo, hi = clause.get("min"), clause.get("max")
                if lo is not None:
                    mask &= draw >= float(lo)
                if hi is not None:
                    mask &= draw <= float(hi)
                continue
            op_name = clause.get("op", ">=")
            if op_name not in _COMPARATORS:
                raise ValueError(f"unsupported criteria op {op_name!r}; expected one of {sorted(_COMPARATORS)}.")
            mask &= _COMPARATORS[op_name](draw, float(clause["threshold"]))
        return mask

    return predicate


def _query_drill_target(posterior: Any, params: dict[str, Any]) -> dict[str, Any]:
    if "criteria" not in params:
        raise ValueError("query 'drill_target' requires params.criteria.")
    mask = _region_mask(params, posterior.grid.n)
    dq = decision_quantities.drill_target_prob(
        posterior,
        mask,
        criteria=_criteria_predicate(params["criteria"]),
        n=int(params.get("n", 4096)),
        rng=_rng(params),
        prior_var=_optional_array(params.get("prior_var")),
        posterior_var=_optional_array(params.get("posterior_var")),
    )
    return _summarize(dq)


def _query_marginal(posterior: Any, params: dict[str, Any]) -> dict[str, Any]:
    n = posterior.grid.n
    if "index" in params:
        idx = np.atleast_1d(np.asarray(params["index"], dtype=int))
    elif "region" in params or "region_index" in params:
        idx = np.flatnonzero(_region_mask(params, n))
    else:
        raise ValueError("query 'marginal' requires params.index or params.region/region_index.")
    alpha = 1.0 - float(params.get("level", 0.9))
    lo, hi = posterior.credible_interval(alpha)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    mean = posterior.mean[idx]
    std = posterior.marginal_std[idx]
    return {
        "index": idx.tolist(),
        "value": mean.tolist() if mean.size > 1 else float(mean[0]),
        "distribution": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "credible_interval_90": [lo[idx].tolist(), hi[idx].tolist()],
        },
        "prior_dominated": _variance_reduction_flag(params, mask),
    }


def _query_section(posterior: Any, params: dict[str, Any]) -> dict[str, Any]:
    axes = {axis: params[axis] for axis in ("x", "y", "z") if axis in params}
    if not axes:
        raise ValueError("query 'section' requires at least one of params.x/y/z.")
    result = posterior.slice(**axes)
    return {
        "coordinates": result["coordinates"].tolist(),
        "value": result["mean"].tolist(),
        "distribution": {
            "mean": result["mean"].tolist(),
            "map": result["map"].tolist(),
            "marginal_std": result["marginal_std"].tolist(),
        },
        "prior_dominated": _variance_reduction_flag(params, result["index"]),
    }


_QUERY_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "region_mass": _query_region_mass,
    "prob_exceed": _query_prob_exceed,
    "net_pay": _query_net_pay,
    "drill_target": _query_drill_target,
    "marginal": _query_marginal,
    "section": _query_section,
}


def query_posterior(posterior_ref: str, query: str, params: dict | None = None) -> dict:
    """Read a decision quantity; return ``{value|distribution, prior_dominated: bool}`` -- flag ALWAYS present."""
    if query not in _QUERY_HANDLERS:
        raise ValueError(f"unknown query {query!r}; expected one of {sorted(_QUERY_HANDLERS)}.")
    artifacts = _artifacts_module()
    posterior = artifacts.load_posterior(posterior_ref)
    return _QUERY_HANDLERS[query](posterior, dict(params or {}))


def gassmann(inputs: dict) -> dict:
    """Rock physics; return ``{vp, vs, rho | phi, sat, uncertainty}``."""
    missing = [k for k in _GASSMANN_FIELDS if k not in inputs]
    if missing:
        raise ValueError(f"gassmann inputs missing required field(s): {missing}")
    base = {k: float(inputs[k]) for k in _GASSMANN_FIELDS}
    uncertainty = dict(inputs.get("uncertainty") or {})
    if not uncertainty:
        vp_out, vs_out, rho_out = fluid_substitute(**base)
        return {
            "vp": float(vp_out),
            "vs": float(vs_out),
            "rho": float(rho_out),
            "uncertainty": {"vp": 0.0, "vs": 0.0, "rho": 0.0},
        }
    n_samples = int(inputs.get("n_samples", 2000))
    rng = np.random.default_rng(inputs.get("seed"))
    draws = {k: base[k] + rng.standard_normal(n_samples) * float(uncertainty.get(k, 0.0)) for k in _GASSMANN_FIELDS}
    vp_out, vs_out, rho_out = fluid_substitute(**draws)
    ddof = 1 if n_samples > 1 else 0
    return {
        "vp": float(np.mean(vp_out)),
        "vs": float(np.mean(vs_out)),
        "rho": float(np.mean(rho_out)),
        "uncertainty": {
            "vp": float(np.std(vp_out, ddof=ddof)),
            "vs": float(np.std(vs_out, ddof=ddof)),
            "rho": float(np.std(rho_out, ddof=ddof)),
        },
    }


def forward_model(modality: str, model_ref: str, geometry_ref: str) -> dict:
    """Simulate data; return ``{data_ref: str}`` (content-hashed)."""
    if modality not in _LINEAR_MODALITY_BUILDERS:
        raise _unsupported_modality("forward_model", modality)
    artifacts = _artifacts_module()
    posterior = artifacts.load_posterior(model_ref)
    geometry = _load_geometry_bundle(geometry_ref)
    stored_volumes = (posterior.grid.provenance or {}).get("cell_volumes")
    if stored_volumes is None:
        raise ValueError(f"model {model_ref!r} grid has no stored cell_volumes; required for modality {modality!r}.")
    cell_volumes = np.asarray(stored_volumes, dtype=float)
    operator = _LINEAR_MODALITY_BUILDERS[modality](posterior.grid.coordinates, cell_volumes, {})
    predicted = operator.predict(posterior.grid, posterior.mean, geometry.points)

    path = _default_artifact_path()
    _write_json(
        f"{path}.json",
        {
            "schema": "mixle_pde.forward_model_result/v1",
            "modality": modality,
            "model_ref": model_ref,
            "geometry_ref": geometry_ref,
            "points": geometry.points.tolist(),
            "data": np.asarray(predicted, dtype=float).tolist(),
        },
    )
    return {"data_ref": f"{path}.json"}
