"""Community exposure pathways: sampling transport fields at receptor locations (workstream K2).

This card is the last mile between a transport field -- a G3 air-dispersion plume
(:mod:`mixle_pde.dispersion`) or a G1 groundwater solute-transport state (:mod:`mixle_pde.groundwater`)
-- and a community exposure number at a named receptor (a house, a well, a school). It builds no new
physics: :func:`receptor_exposure` takes whichever field(s) an upstream transport module produced,
already packaged as a :class:`ConcentrationField`, interpolates them onto receptor coordinates, and (when
the field carries an IC-1 posterior) propagates that uncertainty into a per-receptor credible interval.
:func:`couple_pathways` combines an air field and a water field at the same receptors by intake weights
(breathing rate, ingestion rate) into one combined exposure. Dose-response / health-risk conversion is
explicitly out of scope here (K3); this module only answers "how much is at the receptor", with
provenance tracing that number back to its source field.

Note on G1/G3: at the time this module was written neither :mod:`mixle_pde.dispersion` (G3) nor
:mod:`mixle_pde.groundwater` (G1) had landed in this branch, so :class:`ConcentrationField` is a thin,
self-contained container rather than a type imported from either module -- when G1/G3 land, their forward
evaluations (a `gaussian_plume` grid, a `DispersionOperator`/`GroundwaterTransportOperator` state) need
only be wrapped in a `ConcentrationField` to be consumed here; no change to this module is implied.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mixle.reason.posterior_protocol import Posterior

from mixle_pde.mesh import SimplexMesh

__all__ = ["ConcentrationField", "ReceptorExposure", "receptor_exposure", "couple_pathways"]


def _content_hash(values: np.ndarray) -> str:
    """sha256 of the array bytes; the frozen provenance link back to a source field (K2 step 5)."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype=float).tobytes()).hexdigest()


def _active_axes(coordinates: np.ndarray, *, tol: float = 1.0e-9) -> np.ndarray:
    """Coordinate axes with nonzero spread; a ground-level (x, y, 0) plume has a degenerate z axis that
    would make a 3-D Delaunay triangulation singular, so the fallback sampler triangulates only the axes
    the field actually varies over."""
    span = coordinates.max(axis=0) - coordinates.min(axis=0)
    axes = np.flatnonzero(span > tol)
    return axes if axes.size else np.array([0])


@dataclass
class ConcentrationField:
    """A sampled transport field, ready for receptor sampling.

    ``coordinates`` is ``(n, d)`` -- the points the field is defined at: mesh nodes (a
    :class:`~mixle_pde.groundwater.GroundwaterTransportOperator` state on its mesh) or plume/grid sample
    points (a :func:`~mixle_pde.dispersion.gaussian_plume` evaluation), matching the codebase convention
    of :class:`~mixle_pde.observations.Observation.location` -- typically ``d = 3`` even for a
    ground-level field where every point shares ``z = 0``. ``values`` is the point-estimate concentration
    at each coordinate, in physical units named by ``units`` (e.g. ``"ug/m3"``, ``"mg/L"``). ``mesh`` is
    an optional :class:`~mixle_pde.mesh.SimplexMesh` sharing ``coordinates`` as its nodes for exact
    barycentric sampling; when absent, :func:`receptor_exposure`/:func:`couple_pathways` build a Delaunay
    triangulation on the fly. ``posterior`` is an optional IC-1
    :class:`~mixle.reason.posterior_protocol.Posterior` over ``values`` (draws are ``(n_draws, n)``
    matching ``coordinates``), attached when the upstream field carries uncertainty (e.g. a G3
    ``apportion_sources`` fit or a Bayesian G1 calibration). ``content_hash`` identifies this field for
    provenance and is derived from ``values`` when not supplied.
    """

    coordinates: np.ndarray
    values: np.ndarray
    mesh: SimplexMesh | None = None
    posterior: Posterior | None = None
    units: str = ""
    content_hash: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.atleast_2d(np.asarray(self.coordinates, dtype=float))
        if coords.ndim != 2:
            raise ValueError("coordinates must be an (n, d) array.")
        self.coordinates = coords

        values = np.atleast_1d(np.asarray(self.values, dtype=float))
        if values.shape != (coords.shape[0],):
            raise ValueError("values must have shape (n,) matching coordinates.")
        self.values = values

        if self.mesh is not None:
            if self.mesh.nodes.shape != coords.shape or not np.allclose(self.mesh.nodes, coords):
                raise ValueError("mesh.nodes must equal coordinates when a mesh is supplied.")

        if self.content_hash is None:
            self.content_hash = _content_hash(self.values)


@dataclass
class ReceptorExposure:
    """Per-receptor exposure: point-estimate ``concentration``, an optional credible interval ``ci``
    (``(lo, hi)``, each shape matching ``receptors``, present only when a source field carried a
    posterior), and ``provenance`` linking the result back to its source field(s) by content hash."""

    receptors: np.ndarray
    concentration: np.ndarray
    ci: tuple[np.ndarray, np.ndarray] | None
    provenance: dict[str, Any]


def _build_sampler(field_: ConcentrationField):
    """Return ``sample(values, points) -> interpolated values`` for ``field_`` -- a mesh-backed
    barycentric sampler when ``field_.mesh`` is given, else a Delaunay triangulation built once on
    ``field_.coordinates`` (K2 algorithm step 2, "the mesh sampler"). ``values`` may be ``(n,)`` (the
    field's point estimate) or ``(k, n)`` (a batch of ``k`` posterior draws); points outside the
    triangulation's convex hull fall back to nearest-neighbor extrapolation rather than NaN."""
    coordinates = field_.coordinates
    if field_.mesh is not None:
        mesh = field_.mesh
        axes = np.arange(mesh.dim)
    else:
        from mixle_pde.mesh import delaunay_mesh

        axes = _active_axes(coordinates)
        mesh = delaunay_mesh(coordinates[:, axes])

    def _sample(values: np.ndarray, points: np.ndarray) -> np.ndarray:
        single = np.ndim(values) == 1
        vals = np.atleast_2d(np.asarray(values, dtype=float))
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.shape[1] != coordinates.shape[1]:
            raise ValueError(f"receptors must be given in the field's {coordinates.shape[1]}-D coordinate space.")
        out = mesh.interpolate(vals.T, pts[:, axes]).T
        missing = np.any(np.isnan(out), axis=0)
        if np.any(missing):
            from scipy.spatial import cKDTree

            tree = cKDTree(coordinates)
            _, nearest = tree.query(pts[missing], k=1)
            out[:, missing] = vals[:, nearest]
        return out[0] if single else out

    return _sample


def receptor_exposure(
    field: ConcentrationField,
    receptors: np.ndarray,
    *,
    pathway: str = "air",
    n_samples: int = 512,
    rng: np.random.Generator | None = None,
    credible_level: float = 0.9,
) -> ReceptorExposure:
    """Sample ``field`` at community ``receptors`` (K2 algorithm steps 1-5).

    ``pathway`` names which exposure route ``field`` represents (``"air"`` for a G3 dispersion field,
    ``"water"`` for a G1 groundwater solute-transport field); it only tags the result's provenance, it
    does not change the sampling. When ``field.posterior`` is present (IC-1), ``n_samples`` posterior
    draws are pushed through the identical interpolation via ``Posterior.derived_quantity`` and a
    per-receptor ``credible_level`` credible interval is returned in ``ci``; otherwise ``ci`` is ``None``.
    """
    receptors_arr = np.atleast_2d(np.asarray(receptors, dtype=float))
    sample = _build_sampler(field)
    concentration = sample(field.values, receptors_arr)

    ci = None
    prior_dominated = None
    if field.posterior is not None:
        active_rng = rng if rng is not None else np.random.default_rng()

        def _pushforward(draws: np.ndarray) -> np.ndarray:
            return sample(draws, receptors_arr)

        dq = field.posterior.derived_quantity(_pushforward, n_samples, active_rng)
        ci = dq.credible_interval(credible_level)
        prior_dominated = bool(getattr(dq, "prior_dominated", False))

    provenance: dict[str, Any] = dict(field.provenance)
    provenance.update(
        {
            "pathway": pathway,
            "source_content_hash": field.content_hash,
            "units": field.units,
            "n_samples": n_samples if field.posterior is not None else 0,
            "credible_level": credible_level if field.posterior is not None else None,
        }
    )
    if prior_dominated is not None:
        provenance["prior_dominated"] = prior_dominated

    return ReceptorExposure(receptors=receptors_arr, concentration=concentration, ci=ci, provenance=provenance)


def couple_pathways(
    air: ConcentrationField,
    water: ConcentrationField,
    receptors: np.ndarray,
    *,
    weights: dict[str, float],
    n_samples: int = 512,
    rng: np.random.Generator | None = None,
    credible_level: float = 0.9,
) -> ReceptorExposure:
    """Combine a G3 air-dispersion field and a G1 groundwater-transport field into one per-receptor
    exposure by intake ``weights`` (K2 algorithm step 3): ``weights["air"]`` is an air intake rate (e.g. a
    breathing rate) and ``weights["water"]`` an ingestion rate, so the combined exposure at each receptor
    is ``weights["air"] * air_concentration + weights["water"] * water_concentration``.

    When either field carries a posterior (IC-1), its draws are propagated through the same
    interpolation and combined at the DRAW level -- not by summing the two intervals' endpoints -- so
    ``ci`` is the correct pushforward of the (assumed independent) upstream uncertainties.
    """
    w_air = float(weights.get("air", 0.0))
    w_water = float(weights.get("water", 0.0))
    receptors_arr = np.atleast_2d(np.asarray(receptors, dtype=float))

    sample_air = _build_sampler(air)
    sample_water = _build_sampler(water)

    air_point = sample_air(air.values, receptors_arr)
    water_point = sample_water(water.values, receptors_arr)
    concentration = w_air * air_point + w_water * water_point

    ci = None
    if air.posterior is not None or water.posterior is not None:
        active_rng = rng if rng is not None else np.random.default_rng()

        if air.posterior is not None:
            air_draws = sample_air(air.posterior.samples(n_samples, active_rng), receptors_arr)
        else:
            air_draws = np.broadcast_to(air_point, (n_samples, receptors_arr.shape[0]))

        if water.posterior is not None:
            water_draws = sample_water(water.posterior.samples(n_samples, active_rng), receptors_arr)
        else:
            water_draws = np.broadcast_to(water_point, (n_samples, receptors_arr.shape[0]))

        combined = w_air * air_draws + w_water * water_draws
        alpha = 1.0 - credible_level
        lo, hi = np.quantile(combined, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        ci = (lo, hi)

    provenance = {
        "pathway": "air+water",
        "weights": dict(weights),
        "air_content_hash": air.content_hash,
        "water_content_hash": water.content_hash,
        "n_samples": n_samples if ci is not None else 0,
        "credible_level": credible_level if ci is not None else None,
    }

    return ReceptorExposure(receptors=receptors_arr, concentration=concentration, ci=ci, provenance=provenance)
