"""EM / ERT / MT / AEM sounding ingest (workstream B5).

Loads the two families of electrical/electromagnetic soundings the geophysics forwards in
:mod:`mixle_pde.geophysics` (:func:`~mixle_pde.geophysics.dc_resistivity`) and
:mod:`mixle_pde.observations` (:func:`~mixle_pde.observations.layered_mt_forward_operator`) consume:

* :func:`load_ert` -- a plain-text ERT/DC-resistivity survey: an electrode position table plus a
  ``(a, b, m, n)`` quadrupole measurement schedule and the apparent resistivities. Electrode positions
  are real-world XYZ; this module snaps them onto the nearest node of a caller-supplied
  :class:`~mixle_pde.latent.Field3D` grid so the returned ``schedule`` is the exact node-index
  quadrupole array :func:`mixle_pde.geophysics.dc_resistivity` /
  :func:`mixle_pde.observations.dc_resistivity_forward_operator` expect -- no hand-authored indices.
* :func:`load_mt_edi` -- a (partial) SEG "EDI" magnetotelluric sounding: the ``>FREQ``, ``>RHOXY``, and
  ``>PHSXY`` data blocks plus the ``HEAD`` station location (``LAT``/``LONG``/``ELEV``), feeding
  :func:`mixle_pde.observations.layered_mt_forward_operator`.

Both loaders return a :class:`SoundingSet`, the one shared container for ERT and MT/AEM soundings so a
later inversion card fits either kind without a per-modality branch. :func:`load_mt_edi` turns its
EDI station's geodetic ``LAT``/``LONG`` into metric XYZ via workstream B1's CRS layer
(:mod:`mixle_pde.geospatial.crs`): :func:`~mixle_pde.geospatial.crs.utm_epsg_for` picks the station's
local UTM zone and :func:`~mixle_pde.geospatial.crs.transform_points` reprojects into it.

Workstream **B7 (survey-geometry -> mesh mapping)** had not landed on ``release/0.8.0`` yet when this
module was written, so :func:`load_ert`'s electrode-XYZ -> grid-node-index step (which B7 owns as
``geometry_to_mesh.nearest_node_indices``) carries a small local stand-in, ``_nearest_node_indices``
below, that implements exactly the algorithm B7 specifies (``cKDTree(grid.coordinates).query(...)``).
Swap the import for the real module once B7 lands; the public behaviour is unchanged.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SoundingSet", "load_ert", "load_mt_edi"]


@dataclass
class SoundingSet:
    """One EM/ERT/MT/AEM sounding survey: station geometry, measured data, and (for ERT) the electrode
    schedule already mapped onto mesh node indices.

    ``data`` is the measured quantity. For ERT it is ``(n_measurements,)`` apparent resistivity
    (Ohm-m), one row per ``schedule`` quadrupole. For MT/AEM it is ``(n_frequencies, 2)``, columns
    ``[apparent_resistivity, phase_deg]`` for the on-diagonal xy impedance component (see module
    docstring / Non-goals for what is deliberately not parsed).
    """

    stations: np.ndarray
    data: np.ndarray
    frequencies: np.ndarray | None
    schedule: np.ndarray | None
    crs: str | None = None

    def __post_init__(self) -> None:
        stations = np.atleast_2d(np.asarray(self.stations, dtype=float))
        if stations.ndim != 2 or stations.shape[1] != 3:
            raise ValueError("stations must be an (n_stations, 3) array of XYZ points.")
        self.stations = stations
        self.data = np.asarray(self.data, dtype=float)
        if self.frequencies is not None:
            self.frequencies = np.atleast_1d(np.asarray(self.frequencies, dtype=float))
        if self.schedule is not None:
            schedule = np.asarray(self.schedule, dtype=np.int64)
            if schedule.ndim != 2 or schedule.shape[1] != 4:
                raise ValueError("schedule must be an (n_measurements, 4) node-index quadrupole array.")
            self.schedule = schedule


def _nearest_node_indices(points: np.ndarray, grid: Any) -> np.ndarray:
    """Nearest :class:`~mixle_pde.latent.Field3D` node index for each real-world XYZ point.

    Local stand-in for workstream B7's ``geometry_to_mesh.nearest_node_indices`` (not yet landed on
    ``release/0.8.0``); implements the same algorithm B7 specifies: ``cKDTree(grid.coordinates).query``.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(grid.coordinates, dtype=float))
    _, idx = tree.query(np.atleast_2d(np.asarray(points, dtype=float)))
    return np.atleast_1d(idx).astype(np.int64)


def _split_csv_sections(path: str) -> dict[str, list[dict[str, str]]]:
    """Split an ``ert_survey.csv``-style file into ``# name`` blocks, each parsed as header + rows.

    Section markers are lines starting with ``#`` (e.g. ``# electrodes``); the first non-empty line
    after a marker is the column header, every following line up to the next marker is a data row.
    """
    sections: dict[str, list[dict[str, str]]] = {}
    current: str | None = None
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    with open(path, newline="") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if current is not None:
                    sections[current] = rows
                current = line.lstrip("#").strip().lower()
                header = None
                rows = []
                continue
            fields = next(csv.reader([line]))
            if header is None:
                header = [h.strip() for h in fields]
            else:
                rows.append({h: v.strip() for h, v in zip(header, fields)})
        if current is not None:
            sections[current] = rows
    return sections


def load_ert(path: str, grid: Any, *, crs: str | None = None) -> SoundingSet:
    """Load a plain-text ERT/DC-resistivity survey and map its electrodes onto ``grid`` node indices.

    The file has two ``#``-marked sections: ``# electrodes`` (columns ``id, x, y, z`` -- one real-world
    electrode position per row) and ``# schedule`` (columns ``a, b, m, n, rhoa`` -- quadrupole electrode
    ids referencing the electrode table, plus the measured apparent resistivity). Electrode ids are
    looked up by their raw id string, so they need not be numeric or contiguous.

    Args:
        path: path to the ERT survey CSV (see :func:`_split_csv_sections` for the exact layout).
        grid: a :class:`~mixle_pde.latent.Field3D` (or anything exposing ``.coordinates``) whose nodes
            the electrodes are snapped onto.
        crs: optional EPSG/PROJ string describing ``stations``' coordinate reference system (metadata
            only here; workstream B1 owns the actual transform).

    Returns:
        A :class:`SoundingSet` whose ``schedule`` is the ``(n_measurements, 4)`` integer node-index
        quadrupole array the DC-resistivity forwards consume directly -- no hand-built indices.
    """
    sections = _split_csv_sections(path)
    if "electrodes" not in sections or "schedule" not in sections:
        raise ValueError(f"{path}: expected '# electrodes' and '# schedule' sections.")

    electrode_rows = sections["electrodes"]
    if not electrode_rows:
        raise ValueError(f"{path}: '# electrodes' section has no rows.")
    electrode_ids = [row["id"] for row in electrode_rows]
    electrode_xyz = np.array(
        [[float(row["x"]), float(row["y"]), float(row["z"])] for row in electrode_rows], dtype=float
    )
    node_of_electrode = dict(zip(electrode_ids, _nearest_node_indices(electrode_xyz, grid)))

    schedule_rows = sections["schedule"]
    if not schedule_rows:
        raise ValueError(f"{path}: '# schedule' section has no rows.")
    schedule = np.array(
        [[int(node_of_electrode[row[key]]) for key in ("a", "b", "m", "n")] for row in schedule_rows],
        dtype=np.int64,
    )
    rhoa = np.array([float(row["rhoa"]) for row in schedule_rows], dtype=float)

    return SoundingSet(stations=electrode_xyz, data=rhoa, frequencies=None, schedule=schedule, crs=crs)


_EDI_SECTION_RE = re.compile(r"^>\S")
_EDI_KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"?(-?[0-9:.]+)"?')


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _parse_edi_dms_or_decimal(text: str) -> float:
    """Parse an EDI ``LAT``/``LONG`` value in either ``DD:MM:SS.SS`` or plain decimal-degree form."""
    text = text.strip()
    if ":" not in text:
        return float(text)
    sign = -1.0 if text.startswith("-") else 1.0
    deg, minute, second = text.lstrip("-").split(":")
    return sign * (float(deg) + float(minute) / 60.0 + float(second) / 3600.0)


def _station_xyz_from_geodetic(lat_deg: float, lon_deg: float, elev_m: float) -> tuple[np.ndarray, str]:
    """Station XYZ (metres) in its local UTM zone, from geodetic ``LAT``/``LONG``/``ELEV`` (workstream B1).

    Picks the UTM EPSG code for ``(lon_deg, lat_deg)`` via
    :func:`mixle_pde.geospatial.crs.utm_epsg_for` and reprojects the single point from ``EPSG:4326``
    into it via :func:`mixle_pde.geospatial.crs.transform_points`; elevation passes through unchanged.
    Returns ``(xyz, crs)`` where ``crs`` is the ``"EPSG:<code>"`` string of the zone used.
    """
    from mixle_pde.geospatial.crs import transform_points, utm_epsg_for

    dst_crs = f"EPSG:{utm_epsg_for(lon_deg, lat_deg)}"
    xyz = transform_points(np.array([[lon_deg, lat_deg, elev_m]]), src_crs="EPSG:4326", dst_crs=dst_crs)
    return xyz[0], dst_crs


def _edi_block_values(lines: list[str], tag: str) -> np.ndarray:
    """Collect the whitespace-separated numeric tokens of an EDI ``>TAG ...`` data block."""
    tag_re = re.compile(rf"^>\s*{re.escape(tag)}\b", re.IGNORECASE)
    values: list[float] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _EDI_SECTION_RE.match(stripped):
            if in_block:
                break
            in_block = bool(tag_re.match(stripped))
            continue
        if in_block:
            values.extend(float(tok) for tok in stripped.replace("//", " ").split() if _is_number(tok))
    return np.array(values, dtype=float)


def load_mt_edi(path: str, *, crs: str | None = None) -> SoundingSet:
    """Load a (partial) SEG "EDI" magnetotelluric sounding: frequencies + xy apparent resistivity/phase.

    Line-scans the file's header for ``LAT``/``LONG``/``ELEV`` (station location -- degrees, either
    ``DD:MM:SS.SS`` or decimal, plus metres) and its ``>FREQ``, ``>RHOXY``, ``>PHSXY`` data blocks. Only
    the on-diagonal xy component is read (see module Non-goals); this feeds
    :func:`mixle_pde.observations.layered_mt_forward_operator`.

    Args:
        path: path to the ``.edi`` file.
        crs: EPSG/PROJ string for the returned ``stations``. Defaults to the station's own UTM zone
            (as picked by :func:`mixle_pde.geospatial.crs.utm_epsg_for`), which is the CRS the station
            XYZ is actually computed in; pass an explicit value only to relabel that metadata.

    Returns:
        A :class:`SoundingSet` with ``stations`` shape ``(1, 3)``, ``frequencies`` shape ``(n_freq,)``,
        ``schedule=None``, and ``data`` shape ``(n_freq, 2)`` -- columns
        ``[apparent_resistivity, phase_deg]``.
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    lat = lon = elev = None
    for match in _EDI_KV_RE.finditer(text):
        key, value = match.group(1).upper(), match.group(2)
        if key == "LAT" and lat is None:
            lat = _parse_edi_dms_or_decimal(value)
        elif key == "LONG" and lon is None:
            lon = _parse_edi_dms_or_decimal(value)
        elif key == "ELEV" and elev is None:
            elev = float(value)
    if lat is None or lon is None:
        raise ValueError(f"{path}: EDI header is missing LAT/LONG.")
    station_xyz, station_crs = _station_xyz_from_geodetic(lat, lon, 0.0 if elev is None else elev)

    frequencies = _edi_block_values(lines, "FREQ")
    rho_xy = _edi_block_values(lines, "RHOXY")
    phase_xy = _edi_block_values(lines, "PHSXY")
    if frequencies.size == 0:
        raise ValueError(f"{path}: no >FREQ block found.")
    if rho_xy.size != frequencies.size:
        raise ValueError(f"{path}: RHOXY length {rho_xy.size} != FREQ length {frequencies.size}.")
    if phase_xy.size == 0:
        phase_xy = np.full(frequencies.size, np.nan)
    elif phase_xy.size != frequencies.size:
        raise ValueError(f"{path}: PHSXY length {phase_xy.size} != FREQ length {frequencies.size}.")

    data = np.column_stack([rho_xy, phase_xy])
    return SoundingSet(
        stations=station_xyz.reshape(1, 3),
        data=data,
        frequencies=frequencies,
        schedule=None,
        crs=station_crs if crs is None else crs,
    )
