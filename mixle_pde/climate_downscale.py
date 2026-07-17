"""L7 -- regional climate downscaling (statistical + dynamical) and site extremes (work-plan L7).

A coarse GCM field is not a site-usable number: a global/regional climate model's grid cell sits at a
resolution and with a bias structure that has nothing to do with the point a facility actually occupies.
Downscaling closes that gap two ways -- statistically (bias-correct the coarse series against the site's
own observed history, then carry the correction forward onto the projection: the analog / quantile-mapping
downscaler) or dynamically (drive a regional physics forward run, boundary-forced by the coarse field,
through the platform's own scenario/simulation surface rather than a bespoke reimplementation). Either way
the result is a site-resolution series with its GCM provenance still attached (``climate_hash``), and the
site *extremes* the operation actually cares about -- an extreme-heat or extreme-precipitation return
level -- are read off it with the platform's own extreme-value machinery
(:mod:`mixle.analysis.extreme`), never a bespoke estimator.

Two halves, one module:

* the "core" statistical pieces (:class:`QuantileMap`, :func:`fit_quantile_map`, :func:`apply_quantile_map`,
  :func:`downscaled_return_level`) are generic bias-correction + POT-return-level helpers that only touch
  plain arrays; they live here rather than in a separate ``mixle.analysis.quantile_mapping`` module because
  this task's execution is scoped to a single ``mixle-pde`` worktree/PR (see the PR notes) -- exactly how
  the sibling L2 task folded its own "+ core aggregation" annotation entirely into ``mixle-pde``'s
  ``water_balance.py`` without a companion core-``mixle`` change.
* :func:`downscale` is the "pde" half: it pulls the coarse GCM field (an L4 ``ClimateProjectionStub``
  ``ProvenancedResult``, IC-7, or an equivalent plain dict -- duck-typed exactly as
  ``water_balance._climate_fields`` (L2) does, no ``mixle_mlops`` import required or performed), threads
  its ``content_hash`` through as ``climate_hash`` so every derived extreme traces to its GCM + version,
  and dispatches to the statistical or dynamical path.

Extreme-value estimation is reused, never reimplemented (work-plan non-goals): single-site return levels
go through :func:`mixle.analysis.extreme.peaks_over_threshold` / :func:`mixle.analysis.extreme.return_level`
(POT/GPD); multi-site spatial extremes (:func:`site_spatial_extremes`) go through
:func:`mixle.analysis.max_stable.fit_smith_maxstable`. Both modules are consumed read-only.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from mixle.analysis.extreme import peaks_over_threshold, return_level
from mixle.analysis.max_stable import SmithMaxStable, fit_smith_maxstable

from mixle_pde.mesh import SimplexMesh

# The work-plan's frozen signature spells the site-geometry parameter's type ``Mesh``; the codebase's
# concrete mesh type is `mixle_pde.mesh.SimplexMesh` (the same type L2's `water_balance.catchment` takes).
Mesh = SimplexMesh

__all__ = [
    "QuantileMap",
    "DownscaledField",
    "fit_quantile_map",
    "apply_quantile_map",
    "downscaled_return_level",
    "downscale",
    "site_spatial_extremes",
]

# Standard candidate return periods (in series-observation units -- see the module docstring's fixture
# convention of one observation per year) a downscaled field reports extremes for; periods with too few
# threshold exceedances to fit a GPD are silently omitted rather than raising.
_DEFAULT_RETURN_PERIODS: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0, 100.0)
# The POT threshold a `downscale()` call picks automatically: the site baseline's own 90th percentile --
# a standard, dataset-relative "hot day" / "heavy rain day" convention when no threshold is supplied
# (the frozen L7 signature has no threshold parameter).
_DEFAULT_THRESHOLD_QUANTILE = 0.9
# Scratch store for the dynamical path's boundary-forcing artifact; P1's `write_result_artifact` /
# `read_result_artifact` take a mandatory `store_dir` the frozen L7 signature has no slot for (see
# `_run_dynamical`), so a fixed local directory stands in.
_DYNAMICAL_STORE_DIR = os.path.join(tempfile.gettempdir(), "mixle_pde_climate_rcm")


@dataclass
class QuantileMap:
    """An empirical CDF-matching bias-correction map: ``mod_quantiles[i]`` (the coarse model's i-th
    quantile) maps onto ``obs_quantiles[i]`` (the site's i-th quantile), fit by :func:`fit_quantile_map` and
    applied by :func:`apply_quantile_map`."""

    obs_quantiles: np.ndarray
    mod_quantiles: np.ndarray


def fit_quantile_map(model: np.ndarray, obs: np.ndarray, *, n_q: int = 100) -> QuantileMap:
    """Fit an empirical quantile-quantile bias-correction map between an overlapping-period coarse-model
    series and the site's observed baseline -- the analog / quantile-mapping downscaler (work-plan L7
    step 2). ``n_q`` equally spaced quantile levels (``0`` to ``1``) are matched between the two series;
    :func:`apply_quantile_map` interpolates any further model value (e.g. a future projection) through the
    fitted mapping.
    """
    model = np.asarray(model, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()
    if model.size < 2 or obs.size < 2:
        raise ValueError("fit_quantile_map needs at least two points in each of model/obs.")
    if n_q < 2:
        raise ValueError("n_q must be at least 2.")
    q = np.linspace(0.0, 1.0, int(n_q))
    return QuantileMap(obs_quantiles=np.quantile(obs, q), mod_quantiles=np.quantile(model, q))


def apply_quantile_map(qm: QuantileMap, series: np.ndarray) -> np.ndarray:
    """Bias-correct ``series`` through a fitted :class:`QuantileMap`: piecewise-linear interpolation
    through the matched quantile pairs. Values outside the fitted model range hold at the nearest fitted
    quantile (flat extrapolation -- the standard quantile-mapping convention)."""
    x = np.asarray(series, dtype=float).ravel()
    return np.interp(x, qm.mod_quantiles, qm.obs_quantiles)


def downscaled_return_level(series: np.ndarray, threshold: float, period: float) -> float:
    """Peaks-over-threshold return level of a (downscaled) site series.

    Fits a Generalized Pareto Distribution to the threshold exceedances
    (:func:`mixle.analysis.extreme.peaks_over_threshold`) and reads off the ``period``-return level
    (:func:`mixle.analysis.extreme.return_level`) -- reused, not reimplemented, extreme-value machinery
    (work-plan L7 non-goals). ``period`` is expressed in the same observation units as ``series`` (e.g. a
    once-per-20-years extreme-heat level from an annual series is ``period=20``).
    """
    fit = peaks_over_threshold(np.asarray(series, dtype=float).ravel(), threshold)
    return return_level(fit, period)


@dataclass
class DownscaledField:
    """A site-resolution climate series produced by :func:`downscale`, the return-level extremes derived
    from it (feeding L2's water balance), and the provenance chain back to the driving GCM.

    ``climate_hash`` is the upstream GCM driver's ``content_hash`` (``None`` only if the driver carried
    none) -- every downstream extreme traces to its GCM + version through this field.
    """

    series: np.ndarray
    return_levels: dict[float, float]
    climate_hash: str | None
    provenance: dict


def _gcm_payload(gcm: Any) -> tuple[Any, str | None, dict]:
    """Duck-type an L4 IC-7 ``ProvenancedResult`` (or an equivalent plain dict) into
    ``(value, content_hash, identity)``.

    Mirrors ``water_balance._climate_fields`` (L2): anything exposing ``.value``/``.content_hash`` is
    accepted, or a plain dict with the same fields (optionally nested under a ``"value"`` key) -- no
    ``mixle_mlops`` import is required or performed here, L7 is parallel-independent of the concrete L4
    adapter classes and consumes only the frozen ``ProvenancedResult`` shape as data.
    """
    if hasattr(gcm, "content_hash") and hasattr(gcm, "value"):
        identity = {"model_id": getattr(gcm, "model_id", None), "version": getattr(gcm, "version", None)}
        return gcm.value, gcm.content_hash, identity
    if isinstance(gcm, dict):
        identity = {"model_id": gcm.get("model_id"), "version": gcm.get("version")}
        return gcm.get("value", gcm), gcm.get("content_hash"), identity
    raise TypeError("gcm must be a ProvenancedResult-like object (.value/.content_hash) or a dict.")


def _named_series(value: Any, variable: str, key: str) -> np.ndarray | None:
    """Pull the ``key`` (``"model"`` or ``"projection"``) series for ``variable`` out of a GCM payload.

    Accepts, in priority order: a per-variable nested dict (``value[variable][key]``), a flat dict keyed
    by ``key`` or ``f"{variable}_{key}"``, or (for ``key == "projection"`` only) a bare array-like payload
    used directly as the projection series. Returns ``None`` if no such series is present.
    """
    if isinstance(value, dict):
        per_var = value.get(variable)
        if isinstance(per_var, dict) and key in per_var:
            return np.asarray(per_var[key], dtype=float).ravel()
        if key in value:
            return np.asarray(value[key], dtype=float).ravel()
        combo = f"{variable}_{key}"
        if combo in value:
            return np.asarray(value[combo], dtype=float).ravel()
        if key == "projection" and isinstance(per_var, (list, tuple, np.ndarray)):
            return np.asarray(per_var, dtype=float).ravel()
        return None
    if key == "projection":
        return np.asarray(value, dtype=float).ravel()
    return None


def _run_dynamical(
    projection: np.ndarray, site: Mesh, variable: str, climate_hash: str | None
) -> tuple[np.ndarray, dict]:
    """Drive a regional atmosphere/hydrology forward through the P1 ``simulate`` service (work-plan P1,
    IC-11 ``op="climate_rcm"``), boundary-forced by the coarse GCM projection -- a provenanced scenario,
    not a reimplemented regional model (work-plan L7 non-goals).

    ``mixle_pde.simulation_service`` (P1) had not landed in this repo as of this change (verified against
    every branch of the target release at the time this was written). Rather than inline a bespoke RCM,
    this lazily imports the P1 surface and raises a clear, actionable error naming the missing dependency
    when it is absent -- the same "lazy-import, catch ImportError, raise a clear message" convention
    ``env_data.py``'s ``_require`` helper uses for an unavailable optional system library. Once P1 lands,
    this drives it for real; the frozen L7 signature has no parameter for P1's mandatory ``store_dir``, so
    a fixed local scratch directory stands in (a thin adapter shim, not a contract change).
    """
    try:
        from mixle_pde.simulation_service import (
            Scenario,
            ScenarioStep,
            read_result_artifact,
            simulate,
            write_result_artifact,
        )
    except ImportError as exc:
        raise RuntimeError(
            "method='dynamical' downscaling drives mixle_pde.simulation_service.simulate(op='climate_rcm') "
            "(work-plan P1, IC-11), which is not available in this build. Land P1 first, or call "
            "downscale(..., method='statistical')."
        ) from exc

    os.makedirs(_DYNAMICAL_STORE_DIR, exist_ok=True)
    proj = np.asarray(projection, dtype=float).ravel()
    inputs_ref = write_result_artifact(
        {variable: proj},
        grid={"shape": [int(proj.shape[0])], "origin": [0.0], "spacing": [1.0]},
        units="unknown",
        provenance={"climate_hash": climate_hash, "site_nodes": np.asarray(site.nodes).tolist()},
        store_dir=_DYNAMICAL_STORE_DIR,
    )
    scenario = Scenario(
        steps=[ScenarioStep(op="climate_rcm", inputs_ref=inputs_ref, params={"variable": variable})],
        couplings=[],
        provenance={"climate_hash": climate_hash},
    )
    result = simulate(scenario)
    arrays = read_result_artifact(result.result_ref, store_dir=_DYNAMICAL_STORE_DIR)
    series = arrays.get(variable, next(iter(arrays.values())))
    method_provenance = {
        "method": "dynamical",
        "op": "climate_rcm",
        "result_ref": result.result_ref,
        **dict(result.provenance or {}),
    }
    return np.asarray(series, dtype=float).ravel(), method_provenance


def downscale(
    gcm: Any,  # an L4 IC-7 ProvenancedResult (unimported -- duck-typed, see `_gcm_payload`) or a dict
    site: Mesh,
    *,
    method: str = "statistical",
    obs_baseline: np.ndarray | None = None,
    variable: str = "tmax",
) -> DownscaledField:
    """Downscale a coarse GCM field to ``site`` and derive its extreme-value return levels.

    1. Pull the coarse GCM field's ``variable`` projection series out of ``gcm`` -- an L4
       ``ClimateProjectionStub`` ``ProvenancedResult`` (IC-7) or an equivalent dict -- and record its
       ``content_hash`` as ``climate_hash``.
    2. ``method="statistical"`` (default): fit an empirical quantile map (:func:`fit_quantile_map`) between
       the GCM's overlapping-period series and ``obs_baseline`` (the site's observed history), then
       :func:`apply_quantile_map` it to the projection -- the analog / quantile-mapping downscaler.
    3. ``method="dynamical"``: drive a regional atmospheric/hydrology forward run through the P1
       ``simulate`` service (IC-11, ``op="climate_rcm"``), boundary-forced by the GCM field
       (:func:`_run_dynamical`) -- a provenanced scenario, not a reimplemented model.
    4. Site extremes: :func:`downscaled_return_level` (POT/GPD) over a standard set of return periods,
       thresholded at the 90th percentile of the site baseline (or of the downscaled series itself when no
       baseline is available, e.g. a pure-dynamical run) -- these feed L2's water balance.

    For spatial extremes across several site points, see :func:`site_spatial_extremes` (Smith max-stable).
    """
    value, climate_hash, identity = _gcm_payload(gcm)
    projection = _named_series(value, variable, "projection")
    if projection is None:
        raise ValueError(f"gcm payload carries no {variable!r} projection series to downscale.")

    if method == "statistical":
        if obs_baseline is None:
            raise ValueError("method='statistical' requires obs_baseline (the site's observed history).")
        obs_baseline = np.asarray(obs_baseline, dtype=float).ravel()
        model_hist = _named_series(value, variable, "model")
        if model_hist is None:
            model_hist = projection
        qm = fit_quantile_map(model_hist, obs_baseline)
        series = apply_quantile_map(qm, projection)
        threshold_source = obs_baseline
        method_provenance: dict = {"method": "statistical", "n_q": int(qm.mod_quantiles.shape[0])}
    elif method == "dynamical":
        series, method_provenance = _run_dynamical(projection, site, variable, climate_hash)
        threshold_source = obs_baseline if obs_baseline is not None else series
        threshold_source = np.asarray(threshold_source, dtype=float).ravel()
    else:
        raise ValueError(f"unknown downscaling method {method!r}; expected 'statistical' or 'dynamical'.")

    threshold = float(np.quantile(threshold_source, _DEFAULT_THRESHOLD_QUANTILE))
    return_levels: dict[float, float] = {}
    for period in _DEFAULT_RETURN_PERIODS:
        try:
            return_levels[period] = downscaled_return_level(series, threshold, period)
        except ValueError:
            continue  # too few exceedances to fit a GPD at this threshold -- omit, never fabricate.

    provenance = {
        "variable": variable,
        "gcm": identity,
        "threshold": threshold,
        "threshold_quantile": _DEFAULT_THRESHOLD_QUANTILE,
        **method_provenance,
    }
    return DownscaledField(series=series, return_levels=return_levels, climate_hash=climate_hash, provenance=provenance)


def site_spatial_extremes(site: Mesh, block_maxima: np.ndarray) -> SmithMaxStable:
    """Fit a Smith max-stable process (:mod:`mixle.analysis.max_stable`) across ``site``'s mesh nodes, for
    spatial extremes replicated over several site points (work-plan L7 step 4) -- reused, not
    reimplemented, extreme-value machinery (work-plan non-goals).

    ``block_maxima`` is ``(n_replicates, n_nodes)`` -- e.g. one row per year of site-wide block maxima.
    """
    return fit_smith_maxstable(site.nodes, np.asarray(block_maxima, dtype=float))
