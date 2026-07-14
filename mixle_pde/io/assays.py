"""Geochemical assay ingest + QA/QC (workstream B6).

Multi-element assay sheets arrive as CSV/XLSX exports from a lab: a sample id, a collar/sample XYZ, and
one column per element, in whatever unit the lab reports (ppm, g/t, %, ppb). Below-detection results are
written as a censored cell like ``"<0.5"`` rather than a number. :func:`load_assays` normalizes every
element column to ppm and turns that lab convention into an explicit ``(censored, detection_limit)``
pair per cell -- the same shape :mod:`mixle_pde.geo_observations` already expects for a left-censored
likelihood. :func:`to_multi_element_assay` hands the loaded table to
:class:`mixle_pde.geo_observations.MultiElementAssay` so it scores directly with
:func:`mixle_pde.geo_observations.multi_element_assay_log_likelihood`. :func:`qaqc_flags` runs a few
cheap, format-agnostic sanity checks (below-detection, robust outliers, non-physical values, duplicate
sample ids) a geologist would otherwise eyeball by hand before trusting a batch of assays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle_pde import geo_observations

# Multiplicative factor turning a value reported in `unit` into ppm (parts per million).
_UNIT_TO_PPM = {
    "ppm": 1.0,
    "g/t": 1.0,  # 1 gram per tonne == 1 ppm by mass.
    "%": 1.0e4,
    "pct": 1.0e4,
    "ppb": 1.0e-3,
}


def _require_pandas():
    """Lazy-import pandas; raise a clear ImportError naming the extra to install."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "reading geochemical assay tables needs the optional dependency 'pandas' (and "
            "'openpyxl' for .xlsx workbooks). Install with: pip install 'mixle-pde[data]'."
        ) from e
    return pd


@dataclass
class AssayTable:
    """A multi-element geochemical assay table: sample geometry, ppm values, and censoring state.

    ``values`` is ``(n, k)`` ppm-normalized concentrations. ``censored[i, j]`` marks a below-detection
    cell (parsed from a leading ``"<"`` in the source column); its ``values[i, j]`` is set to the
    detection limit (the conventional at-detection-limit convention) and ``detection_limit[i, j]``
    records that limit in ppm. ``detection_limit`` is ``NaN`` for detected (uncensored) cells.
    """

    sample_id: np.ndarray
    xyz: np.ndarray
    elements: list[str]
    values: np.ndarray  # (n, k) ppm
    censored: np.ndarray  # (n, k) bool
    detection_limit: np.ndarray  # (n, k) ppm
    crs: str | None = None


def _normalize_unit(value: float, unit: str) -> float:
    try:
        factor = _UNIT_TO_PPM[unit.strip().lower()]
    except KeyError as e:
        raise ValueError(f"unsupported assay unit {unit!r}; expected one of {sorted(_UNIT_TO_PPM)}") from e
    return value * factor


def _parse_cell(raw: Any, unit: str) -> tuple[float, bool, float]:
    """Parse one raw assay cell into ``(value_ppm, censored, detection_limit_ppm)``."""
    text = raw.strip() if isinstance(raw, str) else raw
    if isinstance(text, str) and text.startswith("<"):
        limit = _normalize_unit(float(text[1:]), unit)
        return limit, True, limit
    value = _normalize_unit(float(text), unit)
    return value, False, float("nan")


def load_assays(path: str, *, element_cols: list[str], unit: str = "ppm", crs: str | None = None) -> AssayTable:
    """Load a CSV/XLSX assay sheet into an `AssayTable`, normalizing units to ppm and flagging censoring.

    ``path`` must have columns ``sample_id, X, Y, Z`` plus one column per name in ``element_cols``,
    each reported in ``unit`` (one of ``ppm``/``g/t``/``%``/``ppb``). A cell of the form ``"<x"`` is a
    below-detection reading.
    """
    pd = _require_pandas()
    reader = pd.read_excel if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv
    frame = reader(path)

    sample_id = frame["sample_id"].to_numpy()
    xyz = frame[["X", "Y", "Z"]].to_numpy(dtype=float)

    n = len(frame)
    k = len(element_cols)
    values = np.zeros((n, k), dtype=float)
    censored = np.zeros((n, k), dtype=bool)
    detection_limit = np.full((n, k), np.nan, dtype=float)

    for j, col in enumerate(element_cols):
        for i, raw in enumerate(frame[col].to_numpy()):
            v, c, dl = _parse_cell(raw, unit)
            values[i, j] = v
            censored[i, j] = c
            detection_limit[i, j] = dl

    return AssayTable(
        sample_id=sample_id,
        xyz=xyz,
        elements=list(element_cols),
        values=values,
        censored=censored,
        detection_limit=detection_limit,
        crs=crs,
    )


def qaqc_flags(table: AssayTable) -> dict[str, np.ndarray]:
    """Cheap, format-agnostic QA/QC flags over an `AssayTable`.

    Returns a dict of boolean arrays:
      * ``below_detection`` -- ``(n, k)``, alias of ``table.censored``.
      * ``outlier`` -- ``(n, k)``, a detected value more than 5 robust MADs from that element's
        detected-sample median (a lab transcription error or a genuine high-grade outlier either way
        deserves a human look before it drives an inversion).
      * ``negative_or_zero`` -- ``(n, k)``, a non-positive concentration (never physical for an assay).
      * ``duplicate_sample_id`` -- ``(n,)``, ``sample_id`` repeated elsewhere in the table.
    """
    values = table.values
    censored = table.censored
    n, k = values.shape

    outlier = np.zeros((n, k), dtype=bool)
    for j in range(k):
        detected = ~censored[:, j]
        if np.count_nonzero(detected) < 2:
            continue
        idx = np.flatnonzero(detected)
        col = values[idx, j]
        median = np.median(col)
        mad = np.median(np.abs(col - median))
        scale = 1.4826 * mad
        if scale <= 0.0:
            continue
        outlier[idx, j] = np.abs(col - median) > 5.0 * scale

    negative_or_zero = values <= 0.0

    sample_id = np.asarray(table.sample_id)
    _, inverse, counts = np.unique(sample_id, return_inverse=True, return_counts=True)
    duplicate_sample_id = counts[inverse] > 1

    return {
        "below_detection": censored.copy(),
        "outlier": outlier,
        "negative_or_zero": negative_or_zero,
        "duplicate_sample_id": duplicate_sample_id,
    }


def to_multi_element_assay(
    table: AssayTable, *, relative_noise: float = 0.1, noise_floor: float = 1e-3
) -> geo_observations.MultiElementAssay:
    """Build a `geo_observations.MultiElementAssay` from an `AssayTable` for likelihood scoring.

    Assay sheets do not carry a lab-reported analytical covariance, so the per-element noise variance
    here is a simple, documented placeholder: ``relative_noise`` fraction of that element's median
    detected value, floored at ``noise_floor`` ppm so a near-zero element never yields a degenerate
    (zero-variance) noise model. A caller with a real lab covariance should build
    `geo_observations.MultiElementAssay` directly and pass it in instead of using this default.
    """
    k = len(table.elements)
    std = np.empty(k, dtype=float)
    for j in range(k):
        detected = table.values[~table.censored[:, j], j]
        sample = detected if detected.size else table.values[:, j]
        magnitude = np.median(np.abs(sample))
        std[j] = max(relative_noise * magnitude, noise_floor)
    noise_cov = std**2

    return geo_observations.MultiElementAssay(
        elements=list(table.elements),
        location=table.xyz,
        value=table.values,
        noise_cov=noise_cov,
        detection_limit=table.detection_limit,
        censored=table.censored,
        provenance={"source": "mixle_pde.io.assays.load_assays", "crs": table.crs},
    )
