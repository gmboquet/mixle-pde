"""LAS well-log ingest (workstream B3).

A wireline LAS file (Log ASCII Standard, the industry-standard well-log text format) is a depth axis
plus a handful of named curves -- gamma ray, bulk density, sonic travel time, resistivity, and so on.
:func:`load_las` reads one into a :class:`WellLog`: plain numpy arrays, no ``lasio`` object leaking
into the rest of the package. :mod:`mixle_pde.petrophysics` turns those raw curves into the physical
quantities (velocity, density, water saturation) that feed :mod:`mixle_pde.rock_physics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WellLog:
    """One wireline well log: a depth axis plus named curves, as plain numpy arrays.

    ``curves`` maps a LAS mnemonic (``"GR"``, ``"RHOB"``, ``"DT"``, ``"RT"``, ...) to a 1-D array the
    same length as ``depth``; the depth curve itself is also present under its own mnemonic (typically
    ``"DEPT"``), so no curve recorded in the file is silently dropped. ``crs`` is ``None`` when no
    georeferenced surface/borehole-trajectory location has been attached at this ingest step (B1 EPSG
    strings apply once a wellhead XYZ is available).
    """

    depth: np.ndarray
    curves: dict[str, np.ndarray]
    crs: str | None = None


def load_las(path: str, *, crs: str | None = None) -> WellLog:
    """Read a LAS (2.0 ASCII) well log at ``path`` into a :class:`WellLog`.

    Requires the optional dependency ``lasio`` (extra ``las``). Depth comes from the LAS index
    (``las.index``); every curve -- including the depth curve under its own mnemonic -- is copied out
    as a plain ``float64`` numpy array so a caller never has to touch a ``lasio`` object directly.
    """
    try:
        import lasio
    except ImportError as e:
        raise ImportError(
            "reading LAS well logs needs the optional dependency 'lasio'. "
            "Install it with: pip install 'mixle-pde[las]'  (or: pip install lasio)."
        ) from e

    las = lasio.read(path)
    depth = np.asarray(las.index, dtype=float)
    curves = {c.mnemonic: np.asarray(c.data, dtype=float) for c in las.curves}
    return WellLog(depth=depth, curves=curves, crs=crs)
