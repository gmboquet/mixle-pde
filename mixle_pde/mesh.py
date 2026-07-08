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

__all__ = ["SimplexMesh", "box_simplex_mesh", "delaunay_mesh", "space_time_mesh"]


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
        measures = np.empty(self.n_simplices, dtype=float)
        scale = math.factorial(self.dim)
        for i, simplex in enumerate(self.simplices):
            coords = self.nodes[simplex]
            edges = coords[1:] - coords[0]
            measures[i] = abs(float(np.linalg.det(edges))) / scale
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

    def validate(self, *, min_measure: float = 1.0e-14) -> dict[str, Any]:
        """Return structural validation metrics for the mesh."""
        measures = self.simplex_measures()
        return {
            "dim": self.dim,
            "n_nodes": self.n_nodes,
            "n_simplices": self.n_simplices,
            "finite_nodes": bool(np.isfinite(self.nodes).all()),
            "positive_measure": bool(np.all(measures > float(min_measure))) if len(measures) else False,
            "min_measure": float(measures.min()) if len(measures) else 0.0,
            "max_measure": float(measures.max()) if len(measures) else 0.0,
            "total_measure": float(measures.sum()),
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


def space_time_mesh(spatial_mesh: SimplexMesh, times: Any) -> SimplexMesh:
    """Convenience wrapper for ``spatial_mesh.extrude_time(times)``."""
    return spatial_mesh.extrude_time(times)
