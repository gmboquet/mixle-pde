"""IC-2 — content-addressable field-posterior artifacts (work-plan E2).

``save_posterior`` writes an ``*.npz`` of the posterior's arrays plus a sibling ``*.json`` header with
the frozen keys ``{schema, content_hash, crs, grid, units, provenance, created}`` where
``grid = {shape, origin, spacing, ...}`` and ``content_hash`` is the sha256 of the arrays' bytes taken
in sorted-key order (so an artifact is verifiable and de-duplicable). ``load_posterior`` reconstructs a
:class:`mixle_pde.latent.PosteriorField3D` (satisfies the IC-1 ``Posterior`` protocol).
``content_hash(path)`` re-derives the digest for provenance/verification (E7, E10) without loading the
whole posterior.

Only :class:`~mixle_pde.latent.PosteriorField3D` over a mesh-free ("point grid") :class:`~mixle_pde.latent.Field3D`
round-trips today -- a mesh-backed field's :attr:`~mixle_pde.latent.Field3D.mesh` is not yet
content-addressable, so ``save_posterior`` raises a clear error rather than silently dropping the mesh.
Whichever of the four covariance storage modes (dense ``cov``, sparse ``precision_factor``, ``low_rank``
+ ``diag_var``, or ``diag_var`` alone) the posterior uses is preserved losslessly; a sparse precision
factor is stored by its CSC components so a survey-scale posterior is never densified just to persist it.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import scipy.sparse as sp
from mixle.reason.posterior_protocol import Posterior

from mixle_pde.latent import Field3D, PosteriorField3D, SparsePosteriorPrecision

ARTIFACT_SCHEMA = "mixle_pde.field_posterior/v1"
HEADER_KEYS = ("schema", "content_hash", "crs", "grid", "units", "provenance", "created")

_COV_DENSE = "dense"
_COV_PRECISION = "precision"
_COV_LOW_RANK = "low_rank"
_COV_DIAG = "diag"


def save_posterior(p: Posterior, path: str) -> None:
    """Serialise ``p`` to ``{path}.npz`` + ``{path}.json`` (frozen header keys); ``content_hash`` = sha256 of the
    sorted-key array bytes. Idempotent: the same posterior writes the same bytes and the same hash."""
    if not isinstance(p, PosteriorField3D):
        raise TypeError(f"save_posterior only supports mixle_pde.latent.PosteriorField3D today, got {type(p)!r}")
    grid = p.grid
    if grid.mesh is not None:
        raise NotImplementedError(
            "save_posterior does not yet serialise simplex-mesh geometry; only mesh-free (point-grid) "
            "Field3D posteriors round-trip today."
        )

    cov_mode, cov_arrays = _covariance_arrays(p)
    coords = np.asarray(grid.coordinates, dtype=float)

    arrays: dict[str, np.ndarray] = {
        "mean": np.asarray(p.mean, dtype=float),
        "map": np.asarray(p.map, dtype=float),
        "grid_coordinates": coords,
    }
    if grid.mask is not None:
        arrays["grid_mask"] = np.asarray(grid.mask, dtype=bool)
    arrays.update(cov_arrays)

    digest = sha256_of_arrays(arrays)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(f"{path}.npz", **arrays)

    spacing = grid.spacing
    spacing_json = list(spacing) if isinstance(spacing, (tuple, list)) else spacing
    provenance = dict(grid.provenance or {})
    header = {
        "schema": ARTIFACT_SCHEMA,
        "content_hash": digest,
        "crs": provenance.get("crs"),
        "grid": {
            "shape": list(coords.shape),
            "origin": coords.min(axis=0).tolist() if coords.size else [],
            "spacing": spacing_json,
            "bounds": list(grid.bounds) if grid.bounds is not None else None,
            "property_name": grid.property_name,
            "has_mask": grid.mask is not None,
            "cov_mode": cov_mode,
        },
        "units": grid.units,
        "provenance": provenance,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(f"{path}.json", "w") as f:
        json.dump(header, f, indent=2, sort_keys=True)


def load_posterior(path: str) -> Posterior:
    """Reconstruct a `Posterior` (IC-1) from a ``{path}.npz`` + ``{path}.json`` pair written by `save_posterior`."""
    with open(f"{path}.json") as f:
        header = json.load(f)
    with np.load(f"{path}.npz") as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}

    grid_header = header["grid"]
    spacing = grid_header.get("spacing")
    if isinstance(spacing, list):
        spacing = tuple(spacing)
    bounds = tuple(grid_header["bounds"]) if grid_header.get("bounds") is not None else None

    grid = Field3D(
        coordinates=arrays["grid_coordinates"],
        spacing=spacing,
        units=header["units"],
        property_name=grid_header.get("property_name", ""),
        bounds=bounds,
        mask=arrays.get("grid_mask"),
        provenance=dict(header.get("provenance") or {}),
    )

    cov_mode = grid_header.get("cov_mode", _COV_DENSE)
    cov_kwargs = _reconstruct_covariance(cov_mode, arrays)

    return PosteriorField3D(grid=grid, mean=arrays["mean"], map=arrays["map"], **cov_kwargs)


def content_hash(path: str) -> str:
    """Return the sha256 content hash recorded in ``{path}.json`` (recomputing from the npz if absent)."""
    if os.path.exists(f"{path}.json"):
        with open(f"{path}.json") as f:
            header = json.load(f)
        recorded = header.get("content_hash")
        if recorded:
            return recorded
    with np.load(f"{path}.npz") as data:
        arrays = {k: data[k] for k in data.files}
    return sha256_of_arrays(arrays)


def sha256_of_arrays(arrays: dict[str, Any]) -> str:
    """The frozen hashing rule: sha256 over ``arrays[k].tobytes()`` for ``k`` in ``sorted(arrays)``."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode("utf-8"))
        h.update(memoryview(arrays[k]).tobytes() if hasattr(arrays[k], "tobytes") else bytes(arrays[k]))
    return h.hexdigest()


def _covariance_arrays(p: PosteriorField3D) -> tuple[str, dict[str, np.ndarray]]:
    """Return ``(mode, {array_name: array})`` for whichever covariance storage ``p`` uses."""
    if p.cov is not None:
        return _COV_DENSE, {"cov": np.asarray(p.cov, dtype=float)}
    if p.precision_factor is not None:
        mat = sp.csc_matrix(p.precision_factor.precision)
        return _COV_PRECISION, {
            "precision_data": mat.data.astype(float),
            "precision_indices": mat.indices.astype(np.int64),
            "precision_indptr": mat.indptr.astype(np.int64),
            "precision_shape": np.asarray(mat.shape, dtype=np.int64),
        }
    if p.low_rank is not None:
        return _COV_LOW_RANK, {
            "low_rank": np.asarray(p.low_rank, dtype=float),
            "diag_var": np.asarray(p.diag_var, dtype=float),
        }
    return _COV_DIAG, {"diag_var": np.asarray(p.diag_var, dtype=float)}


def _reconstruct_covariance(cov_mode: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rebuild the ``PosteriorField3D`` covariance-mode kwargs from the loaded npz arrays."""
    if cov_mode == _COV_DENSE:
        return {"dense_cov": arrays["cov"]}
    if cov_mode == _COV_PRECISION:
        shape = tuple(int(x) for x in arrays["precision_shape"])
        mat = sp.csc_matrix(
            (arrays["precision_data"], arrays["precision_indices"], arrays["precision_indptr"]),
            shape=shape,
        )
        return {"precision_factor": SparsePosteriorPrecision(precision=mat)}
    if cov_mode == _COV_LOW_RANK:
        return {"low_rank": arrays["low_rank"], "diag_var": arrays["diag_var"]}
    return {"diag_var": arrays["diag_var"]}
