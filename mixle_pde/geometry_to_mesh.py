"""Survey-geometry -> mesh mapping (workstream B7, IC-4 `SurveyGeometry`).

Every nonlinear forward operator in :mod:`mixle_pde.observations` that needs more than a nearest-cell
lookup -- DC/ERT (:func:`~mixle_pde.observations.dc_resistivity_forward_operator`, a node-index
quadrupole ``schedule``) and CSEM (:func:`~mixle_pde.observations.csem_3d_forward_operator`, flat
Yee-``source_edges``/``receiver_edges``) -- has, until now, required the caller to hand-build those
index arrays. This module is the one place that turns real acquisition XYZ (electrode positions, shot
and receiver locations, flight lines) into the mesh handles those operators actually consume, so an
ingest card (B5's EM/ERT/MT/AEM soundings, B2's SEG-Y geometry, ...) never re-derives node or edge
numbering by hand.

:func:`nearest_node_indices` is the KD-tree primitive every other function here builds on.
:func:`electrodes_to_schedule` turns electrode XYZ + ``(a, b, m, n)`` electrode-id quadrupoles into the
node-index quadrupoles :func:`mixle_pde.geophysics.dc_resistivity` consumes directly.
:func:`yee_edge_index` reproduces :mod:`mixle_pde.em_diffusion_3d`'s edge numbering (x-edges, then
y-edges, then z-edges, each block C-ordered) to convert one source/receiver XYZ + dipole axis into a
flat edge index for the CSEM/MT operators.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

from mixle_pde.latent import Field3D

__all__ = ["nearest_node_indices", "electrodes_to_schedule", "yee_edge_index"]


def nearest_node_indices(coords: np.ndarray, grid: Field3D) -> np.ndarray:
    """Snap each row of ``coords`` (real XYZ) onto the nearest node of ``grid`` via a KD-tree.

    Returns the flat node index (into ``grid.coordinates``, and so into any field array flattened the
    same way) of the closest grid node to each input point. This is the primitive
    :class:`mixle_pde.observations.SurveyGeometry.resolve` and :func:`electrodes_to_schedule` both use
    -- the "no hand-built indices" step.
    """
    pts = np.atleast_2d(np.asarray(coords, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coords must be an (n, 3) array of (x, y, z) points, got shape {pts.shape}.")
    tree = cKDTree(grid.coordinates)
    _, idx = tree.query(pts)
    return np.atleast_1d(np.asarray(idx, dtype=int))


def electrodes_to_schedule(electrode_xyz: np.ndarray, abmn: np.ndarray, grid: Field3D) -> np.ndarray:
    """Map electrode XYZ + ``(a, b, m, n)`` electrode-id quadrupoles onto grid node-index quadrupoles.

    ``electrode_xyz`` is ``(n_electrodes, 3)``; ``abmn`` is ``(n_meas, 4)`` integer electrode-id rows
    into ``electrode_xyz`` -- current injected at ``a`` (+) / ``b`` (-), potential measured between
    ``m`` / ``n`` -- with ``-1`` marking a missing pole (a pole-pole / pole-dipole array). Each distinct
    electrode is snapped onto its nearest grid node once (via :func:`nearest_node_indices`), and every
    ``abmn`` row is relabelled into the matching node-index quadrupole. The result is an ``(n_meas, 4)``
    object array -- a missing pole passes through as Python ``None`` -- that
    :func:`mixle_pde.geophysics.dc_resistivity` and
    :func:`mixle_pde.observations.dc_resistivity_forward_operator` accept directly as ``schedule``, with
    no hand-built index table.
    """
    electrode_xyz = np.atleast_2d(np.asarray(electrode_xyz, dtype=float))
    if electrode_xyz.ndim != 2 or electrode_xyz.shape[1] != 3:
        raise ValueError(
            f"electrode_xyz must be an (n_electrodes, 3) array of (x, y, z) points, got shape {electrode_xyz.shape}."
        )
    abmn = np.atleast_2d(np.asarray(abmn, dtype=int))
    if abmn.ndim != 2 or abmn.shape[1] != 4:
        raise ValueError(f"abmn must be an (n_meas, 4) array of electrode-id quadrupoles, got shape {abmn.shape}.")
    n_electrodes = electrode_xyz.shape[0]
    if abmn.size and (abmn[abmn >= 0].max(initial=-1) >= n_electrodes):
        raise ValueError(f"abmn references an electrode id outside [0, {n_electrodes}).")

    node_of_electrode = nearest_node_indices(electrode_xyz, grid)
    schedule = np.empty(abmn.shape, dtype=object)
    for row in range(abmn.shape[0]):
        for col in range(abmn.shape[1]):
            eid = int(abmn[row, col])
            schedule[row, col] = None if eid < 0 else int(node_of_electrode[eid])
    return schedule


def _spacing3(spacing) -> tuple[float, float, float]:
    s = np.atleast_1d(np.asarray(spacing, dtype=float))
    if s.size == 1:
        return float(s[0]), float(s[0]), float(s[0])
    if s.size != 3:
        raise ValueError("spacing must be a scalar or a length-3 (hx, hy, hz) sequence.")
    return float(s[0]), float(s[1]), float(s[2])


def yee_edge_index(point_xyz: np.ndarray, shape: Sequence[int], *, axis: int, spacing=1.0) -> int:
    """Flat Yee-edge index of the edge nearest ``point_xyz`` running along ``axis`` (0=x, 1=y, 2=z).

    Reproduces the edge numbering :mod:`mixle_pde.em_diffusion_3d` assembles its curl-curl operator
    with, and that :func:`mixle_pde.observations.csem_3d_forward_operator`'s ``source_edges``/
    ``receiver_edges`` are indices into: x-edges first (``(nx - 1) * ny * nz`` of them, block shape
    ``(nx - 1, ny, nz)``), then y-edges (``nx * (ny - 1) * nz``, block shape ``(nx, ny - 1, nz)``), then
    z-edges (block shape ``(nx, ny, nz - 1)``) -- each block itself C-ordered over its own ``(i, j, k)``.
    An edge running along ``axis`` sits at the half-integer cell midpoint along that axis and at whole
    grid-node coordinates along the other two, so a source/receiver location plus its dipole axis maps
    onto exactly one flat edge index with no hand-built table.
    """
    nx, ny, nz = (int(s) for s in shape)
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("shape must have every dimension >= 2 (at least one interior edge per axis).")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0 (x), 1 (y), or 2 (z).")
    hx, hy, hz = _spacing3(spacing)
    x, y, z = (float(v) for v in np.asarray(point_xyz, dtype=float).reshape(3))

    n_x_edges = (nx - 1) * ny * nz
    n_y_edges = nx * (ny - 1) * nz

    if axis == 0:
        i = int(round(x / hx - 0.5))
        j = int(round(y / hy))
        k = int(round(z / hz))
        i = min(max(i, 0), nx - 2)
        j = min(max(j, 0), ny - 1)
        k = min(max(k, 0), nz - 1)
        return i * ny * nz + j * nz + k

    if axis == 1:
        i = int(round(x / hx))
        j = int(round(y / hy - 0.5))
        k = int(round(z / hz))
        i = min(max(i, 0), nx - 1)
        j = min(max(j, 0), ny - 2)
        k = min(max(k, 0), nz - 1)
        return n_x_edges + i * (ny - 1) * nz + j * nz + k

    i = int(round(x / hx))
    j = int(round(y / hy))
    k = int(round(z / hz - 0.5))
    i = min(max(i, 0), nx - 1)
    j = min(max(j, 0), ny - 1)
    k = min(max(k, 0), nz - 2)
    return n_x_edges + n_y_edges + i * ny * (nz - 1) + j * (nz - 1) + k
