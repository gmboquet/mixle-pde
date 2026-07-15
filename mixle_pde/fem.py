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
    "assemble_simplex_load_vector",
    "assemble_simplex_mass_matrix",
    "assemble_simplex_stiffness_matrix",
    "boundary_nodes",
    "fem_poisson",
    "simulate_simplex_diffusion",
    "solve_simplex_poisson",
    "step_simplex_diffusion",
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


def assemble_simplex_load_vector(mesh, source, *, min_measure: float = 1.0e-14) -> np.ndarray:
    """Assemble a P1 load vector for scalar, nodal, callable, or per-element source data."""
    nodes = np.asarray(mesh.nodes, dtype=float)
    simplices = np.asarray(mesh.simplices, dtype=int)
    n_nodes = int(nodes.shape[0])
    if callable(source):
        nodal = np.asarray([source(point) for point in nodes], dtype=float)
        return assemble_simplex_mass_matrix(mesh, min_measure=min_measure) @ nodal

    values = np.asarray(source, dtype=float)
    if values.ndim == 0:
        nodal = np.full(n_nodes, float(values))
        return assemble_simplex_mass_matrix(mesh, min_measure=min_measure) @ nodal
    if values.shape == (n_nodes,):
        return assemble_simplex_mass_matrix(mesh, min_measure=min_measure) @ values
    if values.shape == (len(simplices),):
        load = np.zeros(n_nodes, dtype=float)
        n_local = int(nodes.shape[1]) + 1
        for element, simplex in enumerate(simplices):
            try:
                measure, _ = simplex_p1_gradients(nodes[simplex])
            except np.linalg.LinAlgError:
                continue
            if measure <= float(min_measure):
                continue
            load[simplex] += float(values[element]) * measure / n_local
        return load
    raise ValueError("source must be scalar, callable, shape (n_nodes,), or shape (n_simplices,).")


def solve_simplex_poisson(
    mesh,
    source,
    *,
    diffusion=1.0,
    dirichlet: dict[int, float] | None = None,
    min_measure: float = 1.0e-14,
) -> np.ndarray:
    """Solve ``-div(diffusion grad u) = source`` with P1 elements on a simplex mesh."""
    stiffness = assemble_simplex_stiffness_matrix(mesh, diffusion=diffusion, min_measure=min_measure).tolil()
    rhs = assemble_simplex_load_vector(mesh, source, min_measure=min_measure)
    if dirichlet is None:
        if not hasattr(mesh, "boundary_nodes"):
            raise ValueError("dirichlet is required unless mesh has boundary_nodes().")
        bc = {int(node): 0.0 for node in mesh.boundary_nodes()}
    else:
        bc = {int(node): float(value) for node, value in dirichlet.items()}
    for node, value in bc.items():
        stiffness.rows[node] = [node]
        stiffness.data[node] = [1.0]
        rhs[node] = value
    return spla.spsolve(stiffness.tocsr(), rhs)


def step_simplex_diffusion(
    mesh,
    values,
    dt: float,
    *,
    diffusion=1.0,
    source=0.0,
    dirichlet: dict[int, float] | None = None,
    lumped_mass: bool = False,
    min_measure: float = 1.0e-14,
) -> np.ndarray:
    """Advance ``u_t - div(diffusion grad u) = source`` one implicit Euler step."""
    u = np.asarray(values, dtype=float).reshape(-1)
    n_nodes = int(np.asarray(mesh.nodes).shape[0])
    if u.shape != (n_nodes,):
        raise ValueError(f"values must have shape ({n_nodes},).")
    step = float(dt)
    if step <= 0.0:
        raise ValueError("dt must be positive.")

    mass = assemble_simplex_mass_matrix(mesh, lumped=lumped_mass, min_measure=min_measure)
    stiffness = assemble_simplex_stiffness_matrix(mesh, diffusion=diffusion, min_measure=min_measure)
    load = assemble_simplex_load_vector(mesh, source, min_measure=min_measure)
    lhs = (mass + step * stiffness).tolil()
    rhs = mass @ u + step * load
    bc = _dirichlet_map(mesh, dirichlet)
    for node, value in bc.items():
        lhs.rows[node] = [node]
        lhs.data[node] = [1.0]
        rhs[node] = value
    return spla.spsolve(lhs.tocsr(), rhs)


def simulate_simplex_diffusion(
    mesh,
    initial,
    times,
    *,
    diffusion=1.0,
    source=0.0,
    dirichlet: dict[int, float] | None = None,
    lumped_mass: bool = False,
    min_measure: float = 1.0e-14,
) -> np.ndarray:
    """Run implicit simplex diffusion and return states with shape ``(n_times, n_nodes)``."""
    ts = np.asarray(times, dtype=float).reshape(-1)
    if ts.size < 1:
        raise ValueError("times must contain at least one entry.")
    if np.any(np.diff(ts) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    states = np.empty((ts.size, int(np.asarray(mesh.nodes).shape[0])), dtype=float)
    initial_values = np.asarray(initial, dtype=float).reshape(-1)
    if initial_values.shape != (states.shape[1],):
        raise ValueError(f"initial must have shape ({states.shape[1]},).")
    states[0] = initial_values
    for step in range(ts.size - 1):
        states[step + 1] = step_simplex_diffusion(
            mesh,
            states[step],
            float(ts[step + 1] - ts[step]),
            diffusion=diffusion,
            source=source,
            dirichlet=dirichlet,
            lumped_mass=lumped_mass,
            min_measure=min_measure,
        )
    return states


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
        "diffusion must be scalar, shape (n_elements,), shape (dim, dim), or shape (n_elements, dim, dim)."
    )


def _dirichlet_map(mesh, dirichlet: dict[int, float] | None) -> dict[int, float]:
    if dirichlet is None:
        if not hasattr(mesh, "boundary_nodes"):
            raise ValueError("dirichlet is required unless mesh has boundary_nodes().")
        return {int(node): 0.0 for node in mesh.boundary_nodes()}
    return {int(node): float(value) for node, value in dirichlet.items()}
