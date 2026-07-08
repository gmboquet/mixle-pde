"""Unstructured finite elements: P1 (linear triangular) FEM for the Poisson equation.

Structured grids cannot conform to real geology -- faults, pinch-outs, irregular basin outlines, local
refinement near a well. Finite elements on an unstructured triangular mesh can. This is the canonical
piece: linear (P1) elements assembling the stiffness matrix from per-triangle contributions, solving
``-div(kappa grad u) = f`` with Dirichlet boundaries on an arbitrary triangulation (e.g. a Delaunay mesh
of scattered points). Part of the earth-science/multiphysics work (Phase 5).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "assemble_simplex_fem_matrices",
    "assemble_simplex_mass_matrix",
    "assemble_simplex_stiffness_matrix",
    "boundary_nodes",
    "fem_poisson",
    "simplex_p1_gradients",
]


def boundary_nodes(triangles: np.ndarray) -> np.ndarray:
    """The boundary node indices of a triangulation -- the vertices of edges that belong to one triangle."""
    edges: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = (min(a, b), max(a, b))
            edges[e] = edges.get(e, 0) + 1
    bnd = {v for e, cnt in edges.items() if cnt == 1 for v in e}
    return np.array(sorted(bnd), dtype=int)


def simplex_p1_gradients(coords) -> tuple[float, np.ndarray]:
    """Return simplex measure and gradients of the P1 basis functions."""
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != pts.shape[1] + 1:
        raise ValueError("coords must have shape (dim + 1, dim).")
    dim = int(pts.shape[1])
    vandermonde = np.column_stack([np.ones(dim + 1), pts])
    coefficients = np.linalg.solve(vandermonde, np.eye(dim + 1))
    gradients = coefficients[1:].T
    measure = abs(float(np.linalg.det(pts[1:] - pts[0]))) / math.factorial(dim)
    return measure, gradients


def assemble_simplex_stiffness_matrix(mesh, *, diffusion=1.0, min_measure: float = 1.0e-14) -> sp.csr_matrix:
    """Assemble the P1 stiffness matrix on an arbitrary-dimension simplex mesh."""
    nodes = np.asarray(mesh.nodes, dtype=float)
    simplices = np.asarray(mesh.simplices, dtype=int)
    n_nodes = int(nodes.shape[0])
    dim = int(nodes.shape[1])
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for element, simplex in enumerate(simplices):
        try:
            measure, gradients = simplex_p1_gradients(nodes[simplex])
        except np.linalg.LinAlgError:
            continue
        if measure <= float(min_measure):
            continue
        coeff = _element_diffusion(diffusion, element, dim, len(simplices))
        if np.ndim(coeff) == 0:
            local = float(coeff) * measure * (gradients @ gradients.T)
        else:
            local = measure * gradients @ np.asarray(coeff, dtype=float) @ gradients.T
        for i, row in enumerate(simplex):
            for j, col in enumerate(simplex):
                rows.append(int(row))
                cols.append(int(col))
                vals.append(float(local[i, j]))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))


def assemble_simplex_mass_matrix(mesh, *, lumped: bool = False, min_measure: float = 1.0e-14) -> sp.csr_matrix:
    """Assemble the P1 mass matrix on an arbitrary-dimension simplex mesh."""
    nodes = np.asarray(mesh.nodes, dtype=float)
    simplices = np.asarray(mesh.simplices, dtype=int)
    n_nodes = int(nodes.shape[0])
    dim = int(nodes.shape[1])
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    n_local = dim + 1
    consistent_template = np.ones((n_local, n_local), dtype=float)
    np.fill_diagonal(consistent_template, 2.0)
    for simplex in simplices:
        try:
            measure, _ = simplex_p1_gradients(nodes[simplex])
        except np.linalg.LinAlgError:
            continue
        if measure <= float(min_measure):
            continue
        if lumped:
            for node in simplex:
                rows.append(int(node))
                cols.append(int(node))
                vals.append(float(measure / n_local))
        else:
            local = measure * consistent_template / (n_local * (n_local + 1))
            for i, row in enumerate(simplex):
                for j, col in enumerate(simplex):
                    rows.append(int(row))
                    cols.append(int(col))
                    vals.append(float(local[i, j]))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))


def assemble_simplex_fem_matrices(
    mesh,
    *,
    diffusion=1.0,
    lumped_mass: bool = False,
    min_measure: float = 1.0e-14,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return ``(stiffness, mass)`` for P1 elements on a simplex mesh."""
    stiffness = assemble_simplex_stiffness_matrix(mesh, diffusion=diffusion, min_measure=min_measure)
    mass = assemble_simplex_mass_matrix(mesh, lumped=lumped_mass, min_measure=min_measure)
    return stiffness, mass


def fem_poisson(nodes, triangles, source, *, conductivity=1.0, dirichlet=None) -> np.ndarray:
    """Solve ``-div(kappa grad u) = source`` by P1 finite elements on a triangular mesh.

    Args:
        nodes: ``(N, 2)`` vertex coordinates.
        triangles: ``(M, 3)`` integer vertex indices per element.
        source: ``f`` -- scalar, per-node array, or a callable ``f(xy) -> value``.
        conductivity: ``kappa`` -- scalar or per-element.
        dirichlet: ``{node_index: value}`` boundary conditions; default pins every boundary node to 0.

    Returns:
        ``u`` of shape ``(N,)`` -- the FEM solution at the nodes.
    """
    nodes = np.asarray(nodes, dtype=float)
    tris = np.asarray(triangles, dtype=int)
    nn = len(nodes)
    kappa = np.full(len(tris), float(conductivity)) if np.isscalar(conductivity) else np.asarray(conductivity, float)
    if callable(source):
        fval = np.array([source(p) for p in nodes])
    elif np.isscalar(source):
        fval = np.full(nn, float(source))
    else:
        fval = np.asarray(source, dtype=float).ravel()

    rows, cols, vals = [], [], []
    f = np.zeros(nn)
    for e, tri in enumerate(tris):
        (x1, y1), (x2, y2), (x3, y3) = nodes[tri]
        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if abs(area) < 1e-14:
            continue
        b = np.array([y2 - y3, y3 - y1, y1 - y2])  # d(basis)/dx * 2A
        c = np.array([x3 - x2, x1 - x3, x2 - x1])  # d(basis)/dy * 2A
        ke = kappa[e] * (np.outer(b, b) + np.outer(c, c)) / (4.0 * abs(area))  # P1 element stiffness
        for i in range(3):
            f[tri[i]] += abs(area) / 3.0 * fval[tri].mean()  # lumped load
            for j in range(3):
                rows.append(tri[i])
                cols.append(tri[j])
                vals.append(ke[i, j])
    k = sp.csr_matrix((vals, (rows, cols)), shape=(nn, nn)).tolil()

    bc = (
        {int(v): 0.0 for v in boundary_nodes(tris)}
        if dirichlet is None
        else {int(k_): float(v) for k_, v in dirichlet.items()}
    )
    for node, val in bc.items():  # Dirichlet: identity row, fixed RHS
        k.rows[node] = [node]
        k.data[node] = [1.0]
        f[node] = val
    return spla.spsolve(k.tocsr(), f)


def _element_diffusion(diffusion, element: int, dim: int, n_elements: int):
    coeff = np.asarray(diffusion, dtype=float)
    if coeff.ndim == 0:
        return float(coeff)
    if coeff.shape == (n_elements,):
        return float(coeff[int(element)])
    if coeff.shape == (dim, dim):
        return coeff
    if coeff.shape == (n_elements, dim, dim):
        return coeff[int(element)]
    raise ValueError(
        "diffusion must be scalar, shape (n_elements,), shape (dim, dim), "
        "or shape (n_elements, dim, dim)."
    )
