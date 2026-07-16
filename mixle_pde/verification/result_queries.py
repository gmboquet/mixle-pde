"""Backend-neutral point/domain result queries over completed mixle-pde solves (MP-K2).

Source: notes/mixle-pde-ai-native-multiphysics-work-plan.md workstream K, MP-K2 ("Backend-neutral
result and postprocessing queries"). This module implements the baseline slice of that task: point
probes, domain integrals, extrema, and unit-aware reporting. Fluxes, reactions, forces, energies,
spectra, pathlines, and general interpolation/export are deliberately **not** covered here -- the
work-plan scope note calls those out as much larger later increments, not part of this baseline.

This is a *query layer*, not a new result format: it never re-implements a mesh, a quadrature rule,
or a unit registry. It reads whatever a completed solve already produced --
:class:`mixle_pde.canonical_adapter.LegacyCanonicalPoissonResult` (the native-rational-linear
backend) or :class:`mixle_pde.pde_backend_registry.PDEStudyResult` (any registered FD/FDTD/FEM
kernel) -- and wraps it in :class:`FieldSample`, a thin, immutable container pairing the solve's
own node/grid values with the exact geometry (:class:`mixle_pde.mesh.SimplexMesh` for P1 results,
or bare coordinates for structured-grid results) and unit string the originating backend already
declared. Neither ``mixle_pde/problem_adapter.py``, ``mixle_pde/canonical_adapter.py``, nor
``mixle_pde/pde_backend_registry.py`` is modified by this module.

Point probes on a :class:`mixle_pde.mesh.SimplexMesh`-backed sample use the mesh's own P1
barycentric interpolation (:meth:`mixle_pde.mesh.SimplexMesh.interpolate`); on a bare
structured-grid sample they fall back to nearest-node lookup. Domain integrals on a mesh-backed
sample use the exact P1 consistent-mass-matrix quadrature (``ones @ (mass @ values)``, exact for
any field the P1 space itself represents, since the P1 basis is a partition of unity); on a bare
1-D grid sample they fall back to trapezoidal quadrature. Every query result carries the field's
declared unit string read from the originating backend/receipt -- ``unit=None`` when (as for
:mod:`mixle_pde.canonical_adapter`'s records today) the underlying result genuinely carries no unit
metadata, rather than a guessed value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle_pde.canonical_adapter import LegacyCanonicalPoissonResult
from mixle_pde.fem import assemble_simplex_mass_matrix
from mixle_pde.mesh import SimplexMesh
from mixle_pde.pde_backend_registry import PDEStudyResult, get_kernel_registration

__all__ = [
    "ExtremaResult",
    "FieldSample",
    "IntegralResult",
    "PointProbeResult",
    "extrema",
    "field_from_canonical_poisson",
    "field_from_kernel_study",
    "integrate",
    "point_probe",
]


@dataclass(frozen=True)
class FieldSample:
    """A named scalar field over the mesh/grid a completed solve already discretized on.

    ``mesh`` is present for a :class:`~mixle_pde.mesh.SimplexMesh`-discretized (P1 nodal) result --
    ``coordinates`` must then equal ``mesh.nodes`` exactly, since barycentric point probes and the
    mass-matrix domain integral both operate against that mesh's connectivity. ``mesh=None`` marks a
    bare structured-grid/point-cloud sample: point probes there fall back to nearest-node lookup and
    domain integration falls back to 1-D trapezoidal quadrature (the baseline scope's coverage;
    general N-D grid quadrature is not implemented here).

    ``unit`` is read from the originating registration/receipt, never invented -- ``None`` is a
    faithful report that the underlying result carries no unit metadata, not a missing feature.
    """

    field_name: str
    values: np.ndarray
    coordinates: np.ndarray
    unit: str | None
    backend_id: str
    mesh: SimplexMesh | None = None

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("FieldSample.field_name must be non-empty.")
        if not self.backend_id:
            raise ValueError("FieldSample.backend_id must be non-empty.")
        values = np.asarray(self.values, dtype=float).reshape(-1)
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2:
            raise ValueError("FieldSample.coordinates must have shape (n_points, dim).")
        if coordinates.shape[0] != values.shape[0]:
            raise ValueError("FieldSample.coordinates row count must match FieldSample.values length.")
        if not np.all(np.isfinite(values)):
            raise ValueError("FieldSample.values must be finite.")
        if self.mesh is not None:
            if self.mesh.n_nodes != values.shape[0]:
                raise ValueError("FieldSample.mesh node count must match FieldSample.values length.")
            if not np.array_equal(np.asarray(self.mesh.nodes, dtype=float), coordinates):
                raise ValueError("FieldSample.coordinates must equal mesh.nodes for a mesh-backed sample.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True)
class PointProbeResult:
    """The value of one field at (or near) one spatial point."""

    field_name: str
    point: tuple[float, ...]
    value: float
    unit: str | None
    method: str
    backend_id: str


@dataclass(frozen=True)
class IntegralResult:
    """A domain integral of one field over the whole sampled domain."""

    field_name: str
    value: float
    unit: str | None
    method: str
    domain_measure: float
    backend_id: str


@dataclass(frozen=True)
class ExtremaResult:
    """The minimum and maximum of one field and where each occurs."""

    field_name: str
    min_value: float
    min_location: tuple[float, ...]
    max_value: float
    max_location: tuple[float, ...]
    unit: str | None
    backend_id: str


def point_probe(sample: FieldSample, coordinates: Any) -> PointProbeResult:
    """Return ``sample``'s field value at (or near) ``coordinates``.

    Mesh-backed samples use P1 barycentric interpolation and raise ``ValueError`` if the point falls
    outside every simplex (no enclosing element -- reported explicitly rather than extrapolated).
    Bare grid/point-cloud samples use nearest-node lookup.
    """
    point = np.asarray(coordinates, dtype=float).reshape(-1)
    dim = sample.coordinates.shape[1]
    if point.shape[0] != dim:
        raise ValueError(f"point must have {dim} coordinate(s); got {point.shape[0]}.")

    if sample.mesh is not None:
        interpolated = sample.mesh.interpolate(sample.values, point)
        value = float(interpolated[0])
        if not np.isfinite(value):
            raise ValueError(f"point {tuple(point.tolist())} lies outside the meshed domain.")
        method = "p1-barycentric-interpolation"
    else:
        distances = np.linalg.norm(sample.coordinates - point[None, :], axis=1)
        nearest = int(np.argmin(distances))
        value = float(sample.values[nearest])
        method = "nearest-node"

    return PointProbeResult(
        field_name=sample.field_name,
        point=tuple(point.tolist()),
        value=value,
        unit=sample.unit,
        method=method,
        backend_id=sample.backend_id,
    )


def integrate(sample: FieldSample) -> IntegralResult:
    """Return the domain integral of ``sample``'s field over the whole solved domain.

    Mesh-backed samples use the exact P1 consistent-mass-matrix quadrature; bare 1-D grid samples
    use trapezoidal quadrature. Higher-dimensional bare-grid samples are out of the baseline scope
    and raise ``ValueError`` rather than guessing a quadrature rule.
    """
    if sample.mesh is not None:
        mass = assemble_simplex_mass_matrix(sample.mesh)
        value = float(np.ones(sample.mesh.n_nodes) @ (mass @ sample.values))
        method = "p1-consistent-mass-matrix-quadrature"
        domain_measure = sample.mesh.total_measure()
    else:
        if sample.coordinates.shape[1] != 1:
            raise ValueError(
                "trapezoidal domain integration is only implemented for 1-D structured-grid samples "
                "in this baseline; a mesh-backed sample is required for 2-D/3-D domains."
            )
        order = np.argsort(sample.coordinates[:, 0])
        x = sample.coordinates[order, 0]
        y = sample.values[order]
        value = float(np.trapezoid(y, x))
        method = "trapezoidal"
        domain_measure = float(x[-1] - x[0])

    return IntegralResult(
        field_name=sample.field_name,
        value=value,
        unit=sample.unit,
        method=method,
        domain_measure=domain_measure,
        backend_id=sample.backend_id,
    )


def extrema(sample: FieldSample) -> ExtremaResult:
    """Return ``sample``'s field minimum and maximum and the coordinates where each occurs."""
    argmin = int(np.argmin(sample.values))
    argmax = int(np.argmax(sample.values))
    return ExtremaResult(
        field_name=sample.field_name,
        min_value=float(sample.values[argmin]),
        min_location=tuple(sample.coordinates[argmin].tolist()),
        max_value=float(sample.values[argmax]),
        max_location=tuple(sample.coordinates[argmax].tolist()),
        unit=sample.unit,
        backend_id=sample.backend_id,
    )


def field_from_canonical_poisson(
    result: LegacyCanonicalPoissonResult,
    mesh: SimplexMesh,
    *,
    field_name: str = "solution",
    unit: str | None = None,
) -> FieldSample:
    """Wrap a native-rational-linear Poisson solve (:func:`mixle_pde.canonical_adapter.solve_p1_poisson_canonical`).

    ``mesh`` must be the exact :class:`~mixle_pde.mesh.SimplexMesh` (or the ``nodes``/``triangles``
    it was built from) passed into ``solve_p1_poisson_canonical`` -- the result itself stores only
    the nodal solution and execution receipts, never the geometry, so the caller supplies the same
    mesh object it already built rather than this module reconstructing or guessing one.

    ``unit`` defaults to ``None``: :mod:`mixle_pde.canonical_adapter`'s records carry no unit
    metadata for the solved field, so the honest default is unlabeled rather than invented. Pass an
    explicit unit string when the caller's own problem setup declares one out-of-band.
    """
    solution = np.asarray(result.solution, dtype=float).reshape(-1)
    if mesh.n_nodes != solution.shape[0]:
        raise ValueError("mesh node count does not match the canonical Poisson result's solution length.")
    return FieldSample(
        field_name=field_name,
        values=solution,
        coordinates=np.asarray(mesh.nodes, dtype=float),
        unit=unit,
        backend_id="mixle-pde.native-rational-linear",
        mesh=mesh,
    )


def field_from_kernel_study(
    result: PDEStudyResult,
    coordinates: Any,
    *,
    field_name: str | None = None,
    mesh: SimplexMesh | None = None,
) -> FieldSample:
    """Wrap a :func:`mixle_pde.pde_backend_registry.run_math_problem` kernel result for querying.

    A :class:`~mixle_pde.pde_backend_registry.PDEStudyResult` carries only the raw solution array
    and evidence -- the registry never returns geometry -- so the caller supplies ``coordinates``
    (and, for a mesh-based kernel such as ``fem-p1-simplex``, the same ``mesh`` used to build the
    study's ``solve_plan`` parameters). The unit is read from the result's own registered backend
    output port (``mixle_pde.pde_backend_registry.get_kernel_registration(result.backend_id)``),
    never invented.
    """
    registration = get_kernel_registration(result.backend_id)
    output_ports = [port for port in registration.ports if port.role == "output"]
    if not output_ports:
        raise ValueError(f"backend {result.backend_id!r} declares no output port to attach a unit to.")
    port = output_ports[0]
    name = field_name if field_name is not None else port.id

    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    values = np.asarray(result.solution, dtype=float).reshape(-1)
    if coords.shape[0] != values.shape[0]:
        raise ValueError("coordinates row count must match the study result's solution length.")

    return FieldSample(
        field_name=name,
        values=values,
        coordinates=coords,
        unit=port.units,
        backend_id=result.backend_id,
        mesh=mesh,
    )
