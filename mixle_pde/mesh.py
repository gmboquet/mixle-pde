"""Simplex mesh generation and deformation helpers for PDE models.

The solver stack mostly works on structured grids today, while the FEM path accepts a supplied
triangulation. This module provides the missing reusable mesh surface: deterministic box meshes in any
dimension, Delaunay meshes for scattered points, and space-time extrusion. A 3-D spatial tetrahedral mesh
extruded through time becomes a 4-D space-time mesh of pentachora, which is the natural representation for
moving-domain and transient finite-element work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import numpy as np

__all__ = [
    "MovingSimplexMesh",
    "SimplexMesh",
    "box_simplex_mesh",
    "delaunay_mesh",
    "interpolate_simplex_field",
    "moving_mesh",
    "pipe_boundary_facets",
    "pipe_boundary_nodes",
    "pipe_radial_deformation",
    "pipe_simplex_mesh",
    "refine_simplex_mesh",
    "space_time_mesh",
]


@dataclass(frozen=True)
class SimplexMesh:
    """A simplex mesh in arbitrary dimension.

    ``nodes`` is ``(n_nodes, dim)``. ``simplices`` is ``(n_cells, dim + 1)`` and stores node indices for
    line segments (1-D), triangles (2-D), tetrahedra (3-D), pentachora (4-D), and so on.
    """

    nodes: Any
    simplices: Any

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        simplices = np.asarray(self.simplices, dtype=int)
        if nodes.ndim != 2:
            raise ValueError("nodes must be a 2-D array of shape (n_nodes, dim).")
        if simplices.ndim != 2:
            raise ValueError("simplices must be a 2-D integer array.")
        if nodes.shape[1] < 1:
            raise ValueError("mesh dimension must be at least 1.")
        if simplices.shape[1] != nodes.shape[1] + 1:
            raise ValueError("simplices must have dim + 1 vertices.")
        if simplices.size and (simplices.min() < 0 or simplices.max() >= len(nodes)):
            raise ValueError("simplex node index out of range.")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "simplices", simplices)

    @property
    def dim(self) -> int:
        """Topological and coordinate dimension of the mesh."""
        return int(self.nodes.shape[1])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_simplices(self) -> int:
        return int(self.simplices.shape[0])

    def simplex_measures(self) -> np.ndarray:
        """Length / area / volume / hypervolume of every simplex."""
        return np.abs(self.simplex_signed_measures())

    def simplex_edge_lengths(self) -> np.ndarray:
        """All edge lengths per simplex; shape ``(n_simplices, (dim + 1) * dim / 2)``."""
        n_edges = self.dim * (self.dim + 1) // 2
        lengths = np.empty((self.n_simplices, n_edges), dtype=float)
        for i, simplex in enumerate(self.simplices):
            coords = self.nodes[simplex]
            for j, (a, b) in enumerate(combinations(range(self.dim + 1), 2)):
                lengths[i, j] = float(np.linalg.norm(coords[a] - coords[b]))
        return lengths

    def simplex_quality(self) -> np.ndarray:
        """Dimensionless simplex quality from edge-length spread.

        ``1`` means all simplex edges have equal length. Values near ``0`` flag collapsed or strongly
        stretched elements that are unsafe for PDE assembly.
        """
        lengths = self.simplex_edge_lengths()
        if lengths.size == 0:
            return np.array([], dtype=float)
        min_edge = np.min(lengths, axis=1)
        max_edge = np.max(lengths, axis=1)
        return np.divide(min_edge, max_edge, out=np.zeros_like(min_edge), where=max_edge > 0.0)

    def simplex_signed_measures(self) -> np.ndarray:
        """Signed simplex measures using the stored vertex ordering."""
        measures = np.empty(self.n_simplices, dtype=float)
        scale = math.factorial(self.dim)
        for i, simplex in enumerate(self.simplices):
            coords = self.nodes[simplex]
            edges = coords[1:] - coords[0]
            measures[i] = float(np.linalg.det(edges)) / scale
        return measures

    def total_measure(self) -> float:
        """Total mesh measure: length in 1-D, area in 2-D, volume in 3-D, hypervolume in 4-D."""
        return float(self.simplex_measures().sum())

    def facets(self) -> np.ndarray:
        """All simplex facets as sorted node-index rows of width ``dim``."""
        out: list[tuple[int, ...]] = []
        for simplex in self.simplices:
            for facet in combinations(simplex, self.dim):
                out.append(tuple(sorted(int(v) for v in facet)))
        return np.asarray(out, dtype=int).reshape(-1, self.dim)

    def boundary_facets(self) -> np.ndarray:
        """Facets that belong to exactly one simplex."""
        facets = self.facets()
        if len(facets) == 0:
            return facets
        unique, counts = np.unique(facets, axis=0, return_counts=True)
        return unique[counts == 1]

    def boundary_nodes(self) -> np.ndarray:
        """Node indices that lie on the mesh boundary."""
        facets = self.boundary_facets()
        if facets.size == 0:
            return np.array([], dtype=int)
        return np.unique(facets.ravel())

    def locate_points(self, points: Any, *, tol: float = 1.0e-10) -> tuple[np.ndarray, np.ndarray]:
        """Locate points in mesh simplices and return simplex indices plus barycentric weights."""
        pts = _as_points(points, self.dim)
        simplex_indices = np.full(pts.shape[0], -1, dtype=int)
        barycentric = np.full((pts.shape[0], self.dim + 1), np.nan, dtype=float)
        if pts.size == 0 or self.n_simplices == 0:
            return simplex_indices, barycentric

        tol = float(tol)
        for simplex_id, simplex in enumerate(self.simplices):
            unresolved = np.flatnonzero(simplex_indices < 0)
            if unresolved.size == 0:
                break
            coords = self.nodes[simplex]
            matrix = (coords[1:] - coords[0]).T
            rhs = (pts[unresolved] - coords[0]).T
            try:
                tail = np.linalg.solve(matrix, rhs).T
            except np.linalg.LinAlgError:
                continue
            candidate = np.column_stack([1.0 - tail.sum(axis=1), tail])
            inside = np.all(candidate >= -tol, axis=1) & np.all(candidate <= 1.0 + tol, axis=1)
            if not np.any(inside):
                continue
            chosen = unresolved[inside]
            weights = np.clip(candidate[inside], 0.0, 1.0)
            totals = weights.sum(axis=1, keepdims=True)
            weights = np.divide(weights, totals, out=weights, where=totals > 0.0)
            simplex_indices[chosen] = simplex_id
            barycentric[chosen] = weights
        return simplex_indices, barycentric

    def interpolate(self, values: Any, points: Any, *, fill_value: Any = np.nan, tol: float = 1.0e-10) -> np.ndarray:
        """Interpolate node-based values to arbitrary points with simplex barycentric weights."""
        vals = np.asarray(values)
        if vals.shape[:1] != (self.n_nodes,):
            raise ValueError(f"values must have first dimension equal to n_nodes ({self.n_nodes}).")
        pts = _as_points(points, self.dim)
        simplex_indices, barycentric = self.locate_points(pts, tol=tol)

        dtype = np.result_type(vals.dtype, np.asarray(fill_value).dtype)
        out = np.empty((pts.shape[0], *vals.shape[1:]), dtype=dtype)
        out[...] = fill_value
        inside = simplex_indices >= 0
        if np.any(inside):
            simplex_values = vals[self.simplices[simplex_indices[inside]]]
            out[inside] = np.einsum("ij,ij...->i...", barycentric[inside], simplex_values)
        return out

    def validate(self, *, min_measure: float = 1.0e-14, min_quality: float = 1.0e-8) -> dict[str, Any]:
        """Return structural validation metrics for the mesh."""
        measures = self.simplex_measures()
        quality = self.simplex_quality()
        return {
            "dim": self.dim,
            "n_nodes": self.n_nodes,
            "n_simplices": self.n_simplices,
            "finite_nodes": bool(np.isfinite(self.nodes).all()),
            "positive_measure": bool(np.all(measures > float(min_measure))) if len(measures) else False,
            "min_measure": float(measures.min()) if len(measures) else 0.0,
            "max_measure": float(measures.max()) if len(measures) else 0.0,
            "total_measure": float(measures.sum()),
            "min_quality": float(quality.min()) if len(quality) else 0.0,
            "mean_quality": float(quality.mean()) if len(quality) else 0.0,
            "n_low_quality": int(np.count_nonzero(quality < float(min_quality))),
            "n_boundary_facets": int(len(self.boundary_facets())),
            "n_boundary_nodes": int(len(self.boundary_nodes())),
        }

    def transformed(self, displacement: Any = None, *, map_fn: Any = None) -> SimplexMesh:
        """Return the same connectivity on transformed nodes.

        Use ``displacement`` for a vector field ``(n_nodes, dim)`` or a callable returning one. Use
        ``map_fn(nodes)`` for a full coordinate map. This is mesh motion plumbing for moving-domain work;
        it deliberately does not claim to solve ALE/FSI by itself.
        """
        if displacement is not None and map_fn is not None:
            raise ValueError("provide either displacement or map_fn, not both.")
        if map_fn is not None:
            nodes = np.asarray(map_fn(self.nodes), dtype=float)
        elif displacement is not None:
            delta = displacement(self.nodes) if callable(displacement) else displacement
            nodes = self.nodes + np.asarray(delta, dtype=float)
        else:
            nodes = self.nodes.copy()
        return SimplexMesh(nodes, self.simplices.copy())

    def refined(self, *, levels: int = 1, mask: Any = None) -> SimplexMesh:
        """Return a centroid-refined mesh.

        Each selected simplex gets one centroid node and is split into ``dim + 1`` child simplices by
        replacing one original vertex at a time with the centroid. With ``mask=None`` every simplex is
        refined. With a boolean mask, only selected simplices are split and unselected simplices are
        carried through unchanged.
        """
        levels = int(levels)
        if levels < 0:
            raise ValueError("levels must be non-negative.")
        mesh: SimplexMesh = self
        for _ in range(levels):
            mesh = mesh._refined_once(mask=mask if mesh is self else None)
        return mesh

    def _refined_once(self, *, mask: Any = None) -> SimplexMesh:
        if mask is None:
            refine_mask = np.ones(self.n_simplices, dtype=bool)
        else:
            refine_mask = np.asarray(mask, dtype=bool)
            if refine_mask.shape != (self.n_simplices,):
                raise ValueError(f"mask must have shape ({self.n_simplices},).")
        if not np.any(refine_mask):
            return SimplexMesh(self.nodes.copy(), self.simplices.copy())

        nodes = self.nodes.tolist()
        simplices: list[list[int]] = []
        for do_refine, simplex in zip(refine_mask, self.simplices, strict=True):
            simplex_list = [int(v) for v in simplex]
            if not do_refine:
                simplices.append(simplex_list)
                continue
            centroid = np.mean(self.nodes[simplex], axis=0)
            centroid_idx = len(nodes)
            nodes.append(centroid.tolist())
            for replace in range(self.dim + 1):
                child = simplex_list.copy()
                child[replace] = centroid_idx
                simplices.append(child)
        return SimplexMesh(np.asarray(nodes, dtype=float), np.asarray(simplices, dtype=int))

    def extrude_time(self, times: Any) -> SimplexMesh:
        """Extrude a spatial mesh through time into a space-time mesh.

        A ``d``-dimensional simplex swept over one time interval is decomposed into ``d + 1`` simplices in
        ``d + 1`` dimensions. In particular, a 3-D tetrahedral mesh extruded over time becomes a 4-D mesh
        of pentachora.
        """
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size < 2:
            raise ValueError("times must contain at least two entries.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing.")

        layers = [np.column_stack([self.nodes, np.full(self.n_nodes, t)]) for t in times]
        nodes = np.vstack(layers)
        simplices: list[list[int]] = []
        for layer in range(len(times) - 1):
            lo = layer * self.n_nodes
            hi = (layer + 1) * self.n_nodes
            for simplex in self.simplices:
                verts = [int(v) for v in simplex]
                for split in range(self.dim + 1):
                    bottom = [lo + verts[i] for i in range(split + 1)]
                    top = [hi + verts[i] for i in range(split, self.dim + 1)]
                    simplices.append(bottom + top)
        return SimplexMesh(nodes, np.asarray(simplices, dtype=int))


@dataclass(frozen=True)
class MovingSimplexMesh:
    """A simplex mesh with fixed connectivity and time-varying node coordinates.

    This represents moving-domain geometry for ALE-style solvers, pipe/cylinder deformation, and
    space-time finite-element assembly. It is intentionally geometry-only: solvers still decide how to
    transport fields, impose boundary conditions, and remesh when quality becomes unacceptable.
    """

    times: Any
    nodes_over_time: Any
    simplices: Any

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        nodes = np.asarray(self.nodes_over_time, dtype=float)
        simplices = np.asarray(self.simplices, dtype=int)
        if times.size < 1:
            raise ValueError("times must contain at least one entry.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing.")
        if nodes.ndim != 3:
            raise ValueError("nodes_over_time must have shape (n_times, n_nodes, dim).")
        if nodes.shape[0] != times.size:
            raise ValueError("nodes_over_time first axis must match times.")
        SimplexMesh(nodes[0], simplices)
        for step in range(1, nodes.shape[0]):
            SimplexMesh(nodes[step], simplices)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "nodes_over_time", nodes)
        object.__setattr__(self, "simplices", simplices)

    @property
    def dim(self) -> int:
        return int(self.nodes_over_time.shape[2])

    @property
    def n_steps(self) -> int:
        return int(self.nodes_over_time.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes_over_time.shape[1])

    @property
    def n_simplices(self) -> int:
        return int(self.simplices.shape[0])

    def at_step(self, step: int) -> SimplexMesh:
        """Return the spatial mesh at an integer time step."""
        return SimplexMesh(self.nodes_over_time[int(step)], self.simplices.copy())

    def at_time(self, time: float, *, clamp: bool = False) -> SimplexMesh:
        """Linearly interpolate node coordinates and return the mesh at ``time``."""
        t = float(time)
        if clamp:
            t = min(max(t, float(self.times[0])), float(self.times[-1]))
        elif t < self.times[0] or t > self.times[-1]:
            raise ValueError("time is outside the moving mesh time range.")

        right = int(np.searchsorted(self.times, t, side="left"))
        if right < self.n_steps and np.isclose(t, self.times[right]):
            return self.at_step(right)
        if right == 0:
            return self.at_step(0)
        if right >= self.n_steps:
            return self.at_step(self.n_steps - 1)

        left = right - 1
        weight = (t - self.times[left]) / (self.times[right] - self.times[left])
        nodes = (1.0 - weight) * self.nodes_over_time[left] + weight * self.nodes_over_time[right]
        return SimplexMesh(nodes, self.simplices.copy())

    def measure_series(self) -> np.ndarray:
        """Total spatial measure at each time step."""
        return np.asarray([self.at_step(step).total_measure() for step in range(self.n_steps)], dtype=float)

    def simplex_measure_series(self) -> np.ndarray:
        """Per-simplex measures with shape ``(n_times, n_simplices)``."""
        return np.vstack([self.at_step(step).simplex_measures() for step in range(self.n_steps)])

    def simplex_measure_ratios(self, *, reference_step: int = 0) -> np.ndarray:
        """Per-simplex volume/area ratios relative to a reference step."""
        measures = self.simplex_measure_series()
        ref = measures[int(reference_step)]
        return np.divide(measures, ref, out=np.full_like(measures, np.nan), where=ref > 0.0)

    def simplex_quality_series(self) -> np.ndarray:
        """Per-simplex quality with shape ``(n_times, n_simplices)``."""
        return np.vstack([self.at_step(step).simplex_quality() for step in range(self.n_steps)])

    def min_quality_series(self) -> np.ndarray:
        """Minimum simplex quality at each time step."""
        quality = self.simplex_quality_series()
        if quality.size == 0:
            return np.zeros(self.n_steps)
        return np.min(quality, axis=1)

    def simplex_signed_measure_ratios(self, *, reference_step: int = 0) -> np.ndarray:
        """Signed measure ratios; negative values flag element inversion relative to the reference."""
        signed = np.vstack([self.at_step(step).simplex_signed_measures() for step in range(self.n_steps)])
        ref = signed[int(reference_step)]
        return np.divide(signed, ref, out=np.full_like(signed, np.nan), where=np.abs(ref) > 0.0)

    def interpolate_values(
        self,
        values: Any,
        source_time: float,
        target_time: float,
        *,
        fill_value: Any = np.nan,
        tol: float = 1.0e-10,
    ) -> np.ndarray:
        """Transfer node-based values from ``source_time`` onto the nodes at ``target_time``."""
        source = self.at_time(float(source_time))
        target = self.at_time(float(target_time))
        return source.interpolate(values, target.nodes, fill_value=fill_value, tol=tol)

    def to_space_time_mesh(self) -> SimplexMesh:
        """Extrude moving spatial meshes into a ``dim + 1`` space-time simplex mesh."""
        layers = [
            np.column_stack([self.nodes_over_time[step], np.full(self.n_nodes, self.times[step])])
            for step in range(self.n_steps)
        ]
        nodes = np.vstack(layers)
        simplices: list[list[int]] = []
        for layer in range(self.n_steps - 1):
            lo = layer * self.n_nodes
            hi = (layer + 1) * self.n_nodes
            for simplex in self.simplices:
                verts = [int(v) for v in simplex]
                for split in range(self.dim + 1):
                    bottom = [lo + verts[i] for i in range(split + 1)]
                    top = [hi + verts[i] for i in range(split, self.dim + 1)]
                    simplices.append(bottom + top)
        return SimplexMesh(nodes, np.asarray(simplices, dtype=int))

    def validate(self, *, min_measure: float = 1.0e-14, min_quality: float = 1.0e-8) -> dict[str, Any]:
        """Return moving-domain mesh health metrics across all time steps."""
        measures = self.simplex_measure_series()
        quality = self.simplex_quality_series()
        signed_ratios = self.simplex_signed_measure_ratios()
        finite = bool(np.isfinite(self.nodes_over_time).all())
        positive = bool(np.all(measures > float(min_measure))) if measures.size else False
        inverted = int(np.count_nonzero(signed_ratios <= 0.0))
        return {
            "dim": self.dim,
            "n_steps": self.n_steps,
            "n_nodes": self.n_nodes,
            "n_simplices": self.n_simplices,
            "finite_nodes": finite,
            "positive_measure_all_steps": positive,
            "min_measure": float(measures.min()) if measures.size else 0.0,
            "max_measure": float(measures.max()) if measures.size else 0.0,
            "measure_min": float(self.measure_series().min()),
            "measure_max": float(self.measure_series().max()),
            "min_quality": float(np.min(quality)) if quality.size else 0.0,
            "mean_quality": float(np.mean(quality)) if quality.size else 0.0,
            "n_low_quality": int(np.count_nonzero(quality < float(min_quality))),
            "min_signed_measure_ratio": float(np.nanmin(signed_ratios)) if signed_ratios.size else 0.0,
            "n_inverted_or_degenerate_relative_to_reference": inverted,
        }


def box_simplex_mesh(shape: Any, *, lengths: Any = None, origin: Any = None) -> SimplexMesh:
    """Create a Freudenthal simplex mesh of an axis-aligned box.

    ``shape`` gives node counts per axis. ``len(shape) == 3`` gives a tetrahedral 3-D mesh;
    ``len(shape) == 4`` gives a 4-D simplex mesh. Each hyper-rectangular cell is split into ``dim!``
    simplices, deterministically.
    """
    shape = tuple(int(s) for s in np.atleast_1d(shape))
    dim = len(shape)
    if dim < 1:
        raise ValueError("shape must have at least one axis.")
    if any(s < 2 for s in shape):
        raise ValueError("every mesh axis needs at least two nodes.")
    lengths = (
        np.ones(dim, dtype=float) if lengths is None else np.broadcast_to(np.asarray(lengths, dtype=float), (dim,))
    )
    origin = np.zeros(dim, dtype=float) if origin is None else np.broadcast_to(np.asarray(origin, dtype=float), (dim,))

    axes = [origin[i] + np.linspace(0.0, lengths[i], shape[i]) for i in range(dim)]
    grids = np.meshgrid(*axes, indexing="ij")
    nodes = np.column_stack([grid.ravel() for grid in grids])

    simplices: list[list[int]] = []
    for cell in np.ndindex(*(s - 1 for s in shape)):
        for perm in permutations(range(dim)):
            idx = list(cell)
            verts = [tuple(idx)]
            for axis in perm:
                idx = idx.copy()
                idx[axis] += 1
                verts.append(tuple(idx))
            simplices.append([int(np.ravel_multi_index(v, shape)) for v in verts])
    return SimplexMesh(nodes, np.asarray(simplices, dtype=int))


def delaunay_mesh(points: Any) -> SimplexMesh:
    """Create a simplex mesh from scattered points with SciPy Delaunay triangulation.

    SciPy's Delaunay supports N-dimensional points, so this works for 3-D tetrahedral point clouds and
    4-D scattered point clouds where Qhull can form a nondegenerate triangulation.
    """
    from scipy.spatial import Delaunay

    pts = np.asarray(points, dtype=float)
    tri = Delaunay(pts)
    return SimplexMesh(pts, tri.simplices)


def moving_mesh(
    base_mesh: SimplexMesh,
    times: Any,
    displacement: Any = None,
    *,
    map_fn: Any = None,
) -> MovingSimplexMesh:
    """Create a moving-domain mesh from a base mesh and time-dependent geometry.

    ``displacement`` may be a callable ``displacement(nodes, time)`` or an array with shape
    ``(n_nodes, dim)`` or ``(n_times, n_nodes, dim)``. ``map_fn(nodes, time)`` can be supplied instead for
    a full coordinate map. Connectivity is fixed, so this is suitable for moderate deformations and
    space-time assembly; severe distortion still needs adaptive remeshing outside this helper.
    """
    if displacement is not None and map_fn is not None:
        raise ValueError("provide either displacement or map_fn, not both.")

    times = np.asarray(times, dtype=float).reshape(-1)
    if times.size < 1:
        raise ValueError("times must contain at least one entry.")

    if map_fn is not None:
        nodes_over_time = [np.asarray(map_fn(base_mesh.nodes, float(t)), dtype=float) for t in times]
    elif displacement is None:
        nodes_over_time = [base_mesh.nodes.copy() for _ in times]
    elif callable(displacement):
        nodes_over_time = [
            base_mesh.nodes + np.asarray(displacement(base_mesh.nodes, float(t)), dtype=float) for t in times
        ]
    else:
        delta = np.asarray(displacement, dtype=float)
        if delta.shape == base_mesh.nodes.shape:
            nodes_over_time = [base_mesh.nodes + delta for _ in times]
        elif delta.shape == (times.size, *base_mesh.nodes.shape):
            nodes_over_time = [base_mesh.nodes + delta[step] for step in range(times.size)]
        else:
            raise ValueError(
                "displacement must have shape (n_nodes, dim) or (n_times, n_nodes, dim), "
                "or be callable as displacement(nodes, time)."
            )

    return MovingSimplexMesh(times, np.asarray(nodes_over_time, dtype=float), base_mesh.simplices.copy())


def pipe_simplex_mesh(
    *,
    inner_radius: float,
    outer_radius: float,
    length: float,
    n_radial: int = 2,
    n_theta: int = 32,
    n_axial: int = 2,
    center: Any = None,
    axial_origin: float = 0.0,
    axis: int | str = 2,
) -> SimplexMesh:
    """Create a tetrahedral simplex mesh for an annular pipe/cylinder segment."""
    inner = float(inner_radius)
    outer = float(outer_radius)
    if inner < 0.0 or outer <= inner:
        raise ValueError("outer_radius must be greater than a non-negative inner_radius.")
    if float(length) <= 0.0:
        raise ValueError("length must be positive.")
    n_radial = int(n_radial)
    n_theta = int(n_theta)
    n_axial = int(n_axial)
    if n_radial < 2 or n_theta < 3 or n_axial < 2:
        raise ValueError("n_radial >= 2, n_theta >= 3, and n_axial >= 2 are required.")

    axis_index = _axis_index(axis)
    if axis_index < 0 or axis_index > 2:
        raise ValueError("pipe_simplex_mesh axis must be x, y, z, or an integer in [0, 2].")
    radial_axes = [i for i in range(3) if i != axis_index]
    radial_center = (
        np.zeros(2, dtype=float) if center is None else np.broadcast_to(np.asarray(center, dtype=float), (2,))
    )
    radii = np.linspace(inner, outer, n_radial)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    axial = float(axial_origin) + np.linspace(0.0, float(length), n_axial)

    nodes = np.empty((n_radial * n_theta * n_axial, 3), dtype=float)

    def node_index(ir: int, it: int, iz: int) -> int:
        return (iz * n_radial + ir) * n_theta + (it % n_theta)

    for iz, z in enumerate(axial):
        for ir, radius in enumerate(radii):
            for it, theta in enumerate(thetas):
                point = np.zeros(3, dtype=float)
                point[axis_index] = z
                point[radial_axes[0]] = radial_center[0] + radius * np.cos(theta)
                point[radial_axes[1]] = radial_center[1] + radius * np.sin(theta)
                nodes[node_index(ir, it, iz)] = point

    simplices: list[list[int]] = []
    for iz in range(n_axial - 1):
        for ir in range(n_radial - 1):
            for it in range(n_theta):
                corners = {
                    (0, 0, 0): node_index(ir, it, iz),
                    (1, 0, 0): node_index(ir + 1, it, iz),
                    (0, 1, 0): node_index(ir, it + 1, iz),
                    (1, 1, 0): node_index(ir + 1, it + 1, iz),
                    (0, 0, 1): node_index(ir, it, iz + 1),
                    (1, 0, 1): node_index(ir + 1, it, iz + 1),
                    (0, 1, 1): node_index(ir, it + 1, iz + 1),
                    (1, 1, 1): node_index(ir + 1, it + 1, iz + 1),
                }
                for perm in permutations(range(3)):
                    local = [0, 0, 0]
                    verts = [tuple(local)]
                    for local_axis in perm:
                        local = local.copy()
                        local[local_axis] = 1
                        verts.append(tuple(local))
                    simplices.append([corners[vertex] for vertex in verts])
    return _orient_simplex_mesh(nodes, np.asarray(simplices, dtype=int))


def pipe_boundary_nodes(
    mesh: SimplexMesh,
    *,
    inner_radius: float,
    outer_radius: float,
    length: float,
    center: Any = None,
    axial_origin: float = 0.0,
    axis: int | str = 2,
    tol: float = 1.0e-8,
) -> dict[str, np.ndarray]:
    """Classify pipe mesh nodes into inner/outer wall and inlet/outlet boundary groups."""
    if mesh.dim != 3:
        raise ValueError("pipe boundary classification requires a 3D mesh.")
    axis_index = _axis_index(axis)
    if axis_index < 0 or axis_index > 2:
        raise ValueError("axis must be x, y, z, or an integer in [0, 2].")
    radial_axes = [i for i in range(3) if i != axis_index]
    radial_center = (
        np.zeros(2, dtype=float) if center is None else np.broadcast_to(np.asarray(center, dtype=float), (2,))
    )
    radial = np.linalg.norm(mesh.nodes[:, radial_axes] - radial_center[None, :], axis=1)
    axial = mesh.nodes[:, axis_index]
    tolerance = float(tol) * max(1.0, float(outer_radius), float(length))
    return {
        "inner_wall": np.flatnonzero(np.isclose(radial, float(inner_radius), atol=tolerance)),
        "outer_wall": np.flatnonzero(np.isclose(radial, float(outer_radius), atol=tolerance)),
        "inlet": np.flatnonzero(np.isclose(axial, float(axial_origin), atol=tolerance)),
        "outlet": np.flatnonzero(np.isclose(axial, float(axial_origin) + float(length), atol=tolerance)),
    }


def pipe_boundary_facets(
    mesh: SimplexMesh,
    *,
    inner_radius: float,
    outer_radius: float,
    length: float,
    center: Any = None,
    axial_origin: float = 0.0,
    axis: int | str = 2,
    tol: float = 1.0e-8,
) -> dict[str, np.ndarray]:
    """Classify pipe boundary facets into named wall and end-cap groups."""
    groups = pipe_boundary_nodes(
        mesh,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        length=length,
        center=center,
        axial_origin=axial_origin,
        axis=axis,
        tol=tol,
    )
    facets = mesh.boundary_facets()
    out: dict[str, np.ndarray] = {}
    for name, nodes in groups.items():
        node_set = set(int(node) for node in nodes)
        mask = np.array([all(int(node) in node_set for node in facet) for facet in facets], dtype=bool)
        out[name] = facets[mask]
    return out


def refine_simplex_mesh(mesh: SimplexMesh, *, levels: int = 1, mask: Any = None) -> SimplexMesh:
    """Convenience wrapper for :meth:`SimplexMesh.refined`."""
    return mesh.refined(levels=levels, mask=mask)


def interpolate_simplex_field(
    mesh: SimplexMesh,
    values: Any,
    points: Any,
    *,
    fill_value: Any = np.nan,
    tol: float = 1.0e-10,
) -> np.ndarray:
    """Convenience wrapper for :meth:`SimplexMesh.interpolate`."""
    return mesh.interpolate(values, points, fill_value=fill_value, tol=tol)


def pipe_radial_deformation(
    *,
    axis: int | str = 2,
    center: Any = None,
    radial_strain: Any = 0.0,
    axial_strain: Any = 0.0,
    axial_origin: float = 0.0,
) -> Any:
    """Return ``displacement(nodes, time)`` for pipe/cylinder radial deformation.

    The radial displacement is ``radial_strain(time) * (r - center)`` in the plane normal to ``axis``.
    The optional axial strain applies ``axial_strain(time) * (x_axis - axial_origin)`` along the pipe
    axis. Scalars are accepted for static deformation; callables make the deformation time-dependent.
    """

    axis_index = _axis_index(axis)

    def _value(value: Any, time: float) -> float:
        return float(value(time) if callable(value) else value)

    def _displacement(nodes: Any, time: float) -> np.ndarray:
        pts = np.asarray(nodes, dtype=float)
        if pts.ndim != 2:
            raise ValueError("nodes must be a 2-D array.")
        if axis_index < 0 or axis_index >= pts.shape[1]:
            raise ValueError("axis is out of range for node dimension.")

        radial_axes = [i for i in range(pts.shape[1]) if i != axis_index]
        radial_center = (
            np.zeros(len(radial_axes), dtype=float)
            if center is None
            else np.broadcast_to(np.asarray(center, dtype=float), (len(radial_axes),))
        )
        delta = np.zeros_like(pts)
        delta[:, radial_axes] = _value(radial_strain, float(time)) * (pts[:, radial_axes] - radial_center)
        delta[:, axis_index] = _value(axial_strain, float(time)) * (pts[:, axis_index] - float(axial_origin))
        return delta

    return _displacement


def space_time_mesh(spatial_mesh: SimplexMesh, times: Any) -> SimplexMesh:
    """Convenience wrapper for ``spatial_mesh.extrude_time(times)``."""
    return spatial_mesh.extrude_time(times)


def _as_points(points: Any, dim: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 0:
        if dim != 1:
            raise ValueError(f"points must have shape (n_points, {dim}) or ({dim},).")
        return pts.reshape(1, 1)
    if pts.ndim == 1:
        if dim == 1:
            return pts.reshape(-1, 1)
        if pts.shape[0] == dim:
            return pts.reshape(1, dim)
        raise ValueError(f"points must have shape (n_points, {dim}) or ({dim},).")
    if pts.ndim == 2 and pts.shape[1] == dim:
        return pts
    raise ValueError(f"points must have shape (n_points, {dim}) or ({dim},).")


def _orient_simplex_mesh(nodes: np.ndarray, simplices: np.ndarray) -> SimplexMesh:
    mesh = SimplexMesh(nodes, simplices)
    oriented = simplices.copy()
    signed = mesh.simplex_signed_measures()
    for i, measure in enumerate(signed):
        if measure < 0.0:
            oriented[i, [0, 1]] = oriented[i, [1, 0]]
    return SimplexMesh(nodes, oriented)


def _axis_index(axis: int | str) -> int:
    if isinstance(axis, str):
        axes = {"x": 0, "y": 1, "z": 2, "t": 3}
        try:
            return axes[axis.lower()]
        except KeyError as exc:
            raise ValueError("axis must be an integer or one of 'x', 'y', 'z', 't'.") from exc
    return int(axis)
