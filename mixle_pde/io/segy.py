"""SEG-Y seismic gather ingest (workstream B2).

Wraps ``segyio`` to load a trace gather with ``ignore_geometry=True`` -- no inline/crossline volume
assumed, just the traces plus their per-trace acquisition headers -- into a plain-array
:class:`SeismicGather`. This gives the migration/FWI step (workstream C9) one common shape to consume
regardless of survey vintage, and keeps the heavy ``segyio`` dependency behind a single import site
(the same lazy-import-and-raise convention as :mod:`mixle_pde.env_data`'s ``_require``, so a mixle-pde
install without the ``segy`` extra still imports cleanly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeismicGather:
    """One SEG-Y trace gather: waveform samples plus per-trace source/receiver acquisition geometry.

    ``traces`` is ``(n_traces, n_samples)`` amplitude. ``source_xyz``/``receiver_xyz`` are
    ``(n_traces, 3)`` real-world coordinates, descaled per the SEG-Y ``SourceGroupScalar`` convention;
    Z is 0 unless the file carries a nonzero elevation header. ``cdp`` is the ``(n_traces,)`` common
    depth-point number. ``crs`` is the EPSG/PROJ string the XYZ columns are expressed in, or ``None``
    for unspecified/mesh-local coordinates (the :mod:`mixle_pde.observations` ``Observation.crs``
    convention, IC-4).
    """

    traces: np.ndarray
    dt: float
    source_xyz: np.ndarray
    receiver_xyz: np.ndarray
    cdp: np.ndarray
    crs: str | None = None


def _require_segyio():
    """Import ``segyio``, or raise a clear ImportError naming the pip extra to install."""
    try:
        import segyio
    except ImportError as e:
        raise ImportError(
            "reading SEG-Y files needs the optional dependency 'segyio'. "
            "Install it with: pip install 'mixle-pde[segy]'  (or: pip install segyio)."
        ) from e
    return segyio


def _descale(raw, scalar) -> np.ndarray:
    """Apply the SEG-Y coordinate-scalar convention to a raw header column.

    Per the SEG-Y trace-header spec, a positive scalar multiplies the raw integer value, a negative
    scalar divides by its absolute value, and 0 means "no scaling" (left as-is).
    """
    out = np.asarray(raw, dtype=float).copy()
    s = np.asarray(scalar, dtype=float)
    pos = s > 0
    neg = s < 0
    out[pos] *= s[pos]
    out[neg] /= -s[neg]
    return out


def load_segy(path: str, *, crs: str | None = None) -> SeismicGather:
    """Load a SEG-Y file into a :class:`SeismicGather`, ignoring inline/crossline 3D geometry (B2).

    Reads every trace and its source/receiver/CDP headers with ``segyio`` in unstructured
    (``ignore_geometry=True``) mode, so this handles a plain 2D line or an unsorted gather -- assembling
    a full 3D volume from inline/crossline headers is out of scope here (see workstream C9 for
    migration proper).

    Args:
        path: local SEG-Y file path.
        crs: EPSG/PROJ string the acquisition coordinates are expressed in, or ``None`` for
            unspecified/mesh-local coordinates; stamped onto the returned gather's ``crs`` field.

    Returns:
        A :class:`SeismicGather` with ``traces`` shape ``(n_traces, n_samples)``.
    """
    segyio = _require_segyio()

    with segyio.open(path, "r", ignore_geometry=True) as f:
        samples = np.asarray(f.samples, dtype=float)
        dt = (samples[1] - samples[0]) / 1000.0  # SEG-Y sample axis is in ms -> seconds
        traces = np.asarray(f.trace.raw[:], dtype=float)

        coord_scalar = np.asarray(f.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=float)
        source_x = _descale(f.attributes(segyio.TraceField.SourceX)[:], coord_scalar)
        source_y = _descale(f.attributes(segyio.TraceField.SourceY)[:], coord_scalar)
        group_x = _descale(f.attributes(segyio.TraceField.GroupX)[:], coord_scalar)
        group_y = _descale(f.attributes(segyio.TraceField.GroupY)[:], coord_scalar)

        elev_scalar = np.asarray(f.attributes(segyio.TraceField.ElevationScalar)[:], dtype=float)
        source_z = _descale(f.attributes(segyio.TraceField.SourceSurfaceElevation)[:], elev_scalar)
        group_z = _descale(f.attributes(segyio.TraceField.ReceiverGroupElevation)[:], elev_scalar)

        cdp = np.asarray(f.attributes(segyio.TraceField.CDP)[:], dtype=float)

    source_xyz = np.stack([source_x, source_y, source_z], axis=1)
    receiver_xyz = np.stack([group_x, group_y, group_z], axis=1)

    return SeismicGather(
        traces=traces,
        dt=float(dt),
        source_xyz=source_xyz,
        receiver_xyz=receiver_xyz,
        cdp=cdp,
        crs=crs,
    )
