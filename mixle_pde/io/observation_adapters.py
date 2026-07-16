"""Canonical Observation adapters for io ingest modules (workstream MP-I2 remainder).

:mod:`mixle_pde.observations` fixes ONE typed shape -- :class:`~mixle_pde.observations.Observation`
(location, value, noise_cov, time, units, provenance, crs, modality) -- that every observation kind
scores against with the same :func:`~mixle_pde.observations.gaussian_log_likelihood` and dispatches
through the same :class:`~mixle_pde.observations.ForwardOperatorRegistry`. Part of :mod:`mixle_pde.io`
already reads real survey/log files straight into that shape (:mod:`mixle_pde.io.insar`); the rest
reads into its own format-specific carrier that nothing else in the package understands
(:class:`mixle_pde.io.potfield.PotentialFieldGrid`, :class:`mixle_pde.io.em_soundings.SoundingSet`,
:class:`mixle_pde.io.las.WellLog`, ...). This module is the thin, typed glue layer for that second
group: one function per format-specific carrier, converting its already-parsed fields into an
``Observation`` -- no re-parsing of the original file and no new format assumptions, only the
geometry/units bookkeeping (grid flattening, station repetition per frequency, depth -> elevation
sign, grid-node -> quadrupole-centroid resolution) an ``Observation`` needs that the carrier itself
does not carry.

Every carrier here is missing at least one thing an ``Observation`` requires by contract --
``noise_cov`` -- because no source file below ships a per-sample uncertainty, so every adapter takes
an explicit ``noise_std`` from the caller rather than inventing one. This is the same convention
:func:`mixle_pde.io.insar.load_insar` already established for its own ``noise_std`` parameter.

Deliberately not adapted here: :class:`mixle_pde.io.segy.SeismicGather` (a trace is a whole waveform,
not a scalar value -- collapsing it to one number per trace would be a new signal-processing step,
not glue); GIS vector layers / drillhole collars (geometry with no associated measured field value);
and geochemical assays, whose left-censored (detection-limit) likelihood already has its own typed
home in :mod:`mixle_pde.geo_observations` (see :mod:`mixle_pde.observations`'s own module docstring)
because ``Observation.noise_cov``'s plain Gaussian contract cannot represent a censored measurement.
"""

from __future__ import annotations

import numpy as np

from mixle_pde.io.em_soundings import SoundingSet
from mixle_pde.io.las import WellLog
from mixle_pde.io.potfield import PotentialFieldGrid
from mixle_pde.latent import Field3D
from mixle_pde.observations import Observation

__all__ = [
    "potfield_grid_to_observation",
    "mt_sounding_to_observation",
    "ert_soundings_to_observation",
    "well_log_curve_to_observation",
]


def _positive_noise_cov(noise_std: float | np.ndarray, n: int) -> np.ndarray:
    """Broadcast ``noise_std`` (scalar or per-sample) to an ``(n,)`` diagonal noise covariance."""
    noise = np.array(np.broadcast_to(np.asarray(noise_std, dtype=float), (n,)), dtype=float)
    if np.any(noise <= 0.0):
        raise ValueError("noise_std must be strictly positive.")
    return noise**2


_POTFIELD_KINDS = ("gravity", "magnetics")


def potfield_grid_to_observation(
    grid: PotentialFieldGrid,
    *,
    kind: str,
    noise_std: float,
    elevation: float | np.ndarray = 0.0,
    modality: str | None = None,
) -> Observation:
    """Flatten a :class:`~mixle_pde.io.potfield.PotentialFieldGrid` into one grid-wide ``Observation``.

    ``kind`` must be ``"gravity"`` or ``"magnetics"`` -- the two potential-field kinds already
    registered by :func:`mixle_pde.observations.gravity_forward_operator` /
    :func:`~mixle_pde.observations.magnetics_forward_operator` -- since a bare ``PotentialFieldGrid``
    does not itself record which physical quantity its single band holds (the module docstring names
    both aeromagnetic TMI and Bouguer gravity as the same on-disk shape).

    ``location`` is built from ``grid.x``/``grid.y`` via the same ``meshgrid(x, y)`` (default
    ``indexing="xy"``) the module's own docstring uses to relate ``values[i, j]`` to ``(x[j], y[i])``
    (see :func:`mixle_pde.io.potfield.load_grid`), raveled in the matching row-major order so
    ``location[k]`` and ``value[k]`` always describe the same pixel. ``elevation`` fills the ``z``
    column -- a scalar (flat survey datum, the default) or a ``(ny, nx)`` array (a real terrain/drape
    surface) -- since the raster itself carries no elevation band.
    """
    if kind not in _POTFIELD_KINDS:
        raise ValueError(f"kind must be one of {_POTFIELD_KINDS} for a potential-field grid, got {kind!r}.")
    values = np.asarray(grid.values, dtype=float)
    ny, nx = values.shape
    xx, yy = np.meshgrid(np.asarray(grid.x, dtype=float), np.asarray(grid.y, dtype=float))

    elev = np.asarray(elevation, dtype=float)
    if elev.ndim == 0:
        elev = np.full((ny, nx), float(elev))
    elif elev.shape != (ny, nx):
        raise ValueError(f"elevation must be a scalar or shape {(ny, nx)}, got {elev.shape}.")

    location = np.column_stack([xx.ravel(), yy.ravel(), elev.ravel()])
    value = values.ravel().copy()
    return Observation(
        kind=kind,
        location=location,
        value=value,
        noise_cov=_positive_noise_cov(noise_std, value.shape[0]),
        crs=grid.crs,
        modality=kind if modality is None else modality,
        provenance={"source": "mixle_pde.io.potfield.PotentialFieldGrid", "transform": list(grid.transform)},
    )


_MT_COMPONENTS = ("apparent_resistivity", "log_apparent_resistivity", "phase")


def mt_sounding_to_observation(
    soundings: SoundingSet,
    *,
    component: str = "apparent_resistivity",
    noise_std: float | np.ndarray,
    modality: str = "mt",
) -> Observation:
    """Adapt a single-station MT/AEM :class:`~mixle_pde.io.em_soundings.SoundingSet` into an ``Observation``.

    Requires the MT shape :func:`~mixle_pde.io.em_soundings.load_mt_edi` returns (``frequencies`` set,
    ``schedule`` ``None``, one station); pass an ERT sounding to :func:`ert_soundings_to_observation`
    instead. ``component`` selects which of ``soundings.data``'s two columns becomes ``value`` and
    fixes ``kind`` to ``f"layered_mt_{component}"`` -- exactly the dispatch key
    :func:`mixle_pde.observations.layered_mt_forward_operator` registers for the same ``component``,
    so the returned ``Observation`` scores directly against that operator with no relabeling.
    ``location`` repeats the one station XYZ once per frequency, the "one row per frequency" metadata
    contract :func:`~mixle_pde.observations.layered_mt_forward_operator`'s own docstring documents.
    """
    if component not in _MT_COMPONENTS:
        raise ValueError(f"component must be one of {_MT_COMPONENTS}, got {component!r}.")
    if soundings.frequencies is None or soundings.schedule is not None:
        raise ValueError(
            "mt_sounding_to_observation expects an MT/AEM SoundingSet (frequencies set, schedule None); "
            "got an ERT-shaped SoundingSet -- use ert_soundings_to_observation instead."
        )
    if soundings.stations.shape[0] != 1:
        raise ValueError(
            f"mt_sounding_to_observation expects a single-station SoundingSet, got {soundings.stations.shape[0]}."
        )
    n = soundings.frequencies.shape[0]
    if component == "phase":
        value = soundings.data[:, 1].copy()
    else:
        rho = soundings.data[:, 0]
        value = np.log(rho) if component == "log_apparent_resistivity" else rho.copy()
    location = np.repeat(soundings.stations, n, axis=0)
    return Observation(
        kind=f"layered_mt_{component}",
        location=location,
        value=value,
        noise_cov=_positive_noise_cov(noise_std, n),
        crs=soundings.crs,
        modality=modality,
        provenance={
            "source": "mixle_pde.io.em_soundings.SoundingSet",
            "frequencies": soundings.frequencies.tolist(),
        },
    )


def ert_soundings_to_observation(
    soundings: SoundingSet,
    grid: Field3D,
    *,
    noise_std: float | np.ndarray,
    modality: str = "ert",
) -> Observation:
    """Adapt an ERT :class:`~mixle_pde.io.em_soundings.SoundingSet` into a ``dc_resistivity`` ``Observation``.

    ``soundings.schedule`` (from :func:`~mixle_pde.io.em_soundings.load_ert`) stores ``grid``-node
    indices, not indices into ``soundings.stations`` -- ``load_ert`` already snapped each electrode
    onto its nearest node of the same ``grid`` -- so ``location`` is resolved back through
    ``grid.coordinates`` rather than ``soundings.stations``: each row is the centroid of its
    quadrupole's four (a, b, m, n) node positions, one instance of the "survey midpoint" metadata
    :func:`mixle_pde.observations.dc_resistivity_forward_operator`'s docstring names as the convention
    (deliberately unspecified further there -- any point associated with the measurement is valid
    metadata).

    ``value`` is ``soundings.data`` -- apparent resistivity in Ohm-m -- copied through unchanged.
    ``kind="dc_resistivity"`` matches that operator's registered dispatch key so the pair type-checks
    end to end; scoring this ``Observation`` against the operator's own prediction (a transfer
    resistance, optionally logged) is only physically meaningful once the survey's geometric factor
    has been applied on one side or the other -- this adapter performs the type/shape unification
    MP-I2 owns, not that separate geophysics calibration step.
    """
    if soundings.schedule is None:
        raise ValueError(
            "ert_soundings_to_observation expects an ERT SoundingSet with a schedule; "
            "got an MT/AEM-shaped SoundingSet -- use mt_sounding_to_observation instead."
        )
    coordinates = np.asarray(grid.coordinates, dtype=float)
    location = coordinates[soundings.schedule].mean(axis=1)
    value = np.asarray(soundings.data, dtype=float).copy()
    return Observation(
        kind="dc_resistivity",
        location=location,
        value=value,
        noise_cov=_positive_noise_cov(noise_std, value.shape[0]),
        crs=soundings.crs,
        modality=modality,
        provenance={
            "source": "mixle_pde.io.em_soundings.SoundingSet",
            "schedule": soundings.schedule.tolist(),
        },
    )


def well_log_curve_to_observation(
    log: WellLog,
    curve: str,
    *,
    noise_std: float | np.ndarray,
    wellhead_xy: tuple[float, float] = (0.0, 0.0),
    kind: str = "borehole",
    modality: str = "well_log",
) -> Observation:
    """Adapt one named curve of a :class:`~mixle_pde.io.las.WellLog` into a borehole ``Observation``.

    Assumes a vertical well (measured depth equals true vertical depth -- no deviation survey is
    attached at LAS ingest, see :mod:`mixle_pde.io.las`'s module docstring): ``location`` is
    ``(wellhead_x, wellhead_y, -depth)`` per sample, the sign flip putting ``z`` on the same
    "elevation, positive up" convention :mod:`mixle_pde.io.insar` and the gravity/magnetics forward
    operators use. ``wellhead_xy`` places the well at its real surface location; the default
    ``(0, 0)`` treats it as already mesh-local. ``kind`` defaults to ``"borehole"``, matching
    :func:`mixle_pde.observations.borehole_forward_operator` (a nearest-grid-node point sample) --
    appropriate when ``curve`` already is the physical property the latent field represents, with no
    petrophysical transform (:mod:`mixle_pde.petrophysics`) in between.
    """
    if curve not in log.curves:
        raise KeyError(f"{curve!r} not in this WellLog's curves: {sorted(log.curves)}.")
    value = np.asarray(log.curves[curve], dtype=float).copy()
    depth = np.asarray(log.depth, dtype=float)
    if depth.shape != value.shape:
        raise ValueError(f"depth shape {depth.shape} does not match curve {curve!r} shape {value.shape}.")
    x0, y0 = wellhead_xy
    n = value.shape[0]
    location = np.column_stack([np.full(n, float(x0)), np.full(n, float(y0)), -depth])
    return Observation(
        kind=kind,
        location=location,
        value=value,
        noise_cov=_positive_noise_cov(noise_std, n),
        crs=log.crs,
        modality=modality,
        provenance={"source": "mixle_pde.io.las.WellLog", "curve": curve},
    )
