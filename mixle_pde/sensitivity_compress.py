"""Wavelet-thresholded sensitivity-matrix compression for large potential-field surveys (C4).

An airborne gravity/magnetic survey's dense sensitivity matrix ``G`` (``n_obs x n_cells``) is the single
biggest memory item in a potential-field inversion -- a modest 5,000-station x 200,000-cell mesh is already
8 GB of ``float64``. But ``G`` is smooth: nearby cells have nearly identical sensitivity to a given station
(the kernel decays like ``1/r^2`` -- ``1/r^3``), so it compresses hard under a wavelet transform, the same
way a photograph does. :func:`wavelet_compress` builds an octree/Morton ordering of the cells so spatially
adjacent columns of ``G`` are adjacent in the matrix (making the column-wise wavelet transform sparse in the
first place), applies a separable 2-D Haar wavelet transform, thresholds the coefficients to a relative
Frobenius-energy budget, and returns a :class:`scipy.sparse.linalg.LinearOperator` that reproduces ``G @ x``
from the thresholded coefficients alone -- without ever holding a second dense ``n_obs x n_cells`` array.

The wavelet machinery here (:func:`_haar_transform_axis` / :func:`_haar_inverse_axis`) is a small,
dependency-free orthonormal multilevel Haar DWT/IDWT (the package has no wavelet library dependency, and a
plain pairwise-sum/difference butterfly is exact, orthogonal, and trivial to validate by round-trip), not a
general biorthogonal wavelet family -- sufficient for the octree-cluster compression this module targets.
Point kernels (:func:`mixle_pde.geophysics.gravity_point_sensitivity` /
:func:`mixle_pde.geophysics.magnetic_dipole_sensitivity`) and the exact prism kernels
(:func:`mixle_pde.geophysics.gravity_prism_sensitivity` / :func:`mixle_pde.geophysics.magnetic_prism_sensitivity`)
both produce a dense ``G`` this module can compress; it does not change how ``G`` is built, only how it is
stored and applied.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

__all__ = ["wavelet_compress"]


def _next_pow2(n: int) -> int:
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _morton_order(coords: np.ndarray) -> np.ndarray:
    """Column permutation clustering spatially nearby cells together (a Z-order / Morton curve over
    ``coords``), so the wavelet transform sees smooth, slowly-varying columns instead of an arbitrary cell
    ordering -- an octree in spirit (each Morton code is exactly the bit-interleaved octree address), without
    building an explicit tree structure.
    """
    coords = np.asarray(coords, dtype=float)
    mins = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - mins, 1e-12)
    bits = 16  # 16 bits/axis -> 48-bit code, ample resolution for clustering purposes only
    levels = (1 << bits) - 1
    scaled = np.clip(((coords - mins) / span) * levels, 0, levels).astype(np.uint64)

    def _spread(v: np.ndarray) -> np.ndarray:
        # interleave-friendly bit spreading (insert two zero bits after each bit), 16 -> 48 bits
        v = v.astype(np.uint64)
        v &= np.uint64(0xFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0xFFFF00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x00FF0000FF0000FF)
        v = (v | (v << np.uint64(8))) & np.uint64(0xF00F00F00F00F00F)
        v = (v | (v << np.uint64(4))) & np.uint64(0x30C30C30C30C30C3)
        v = (v | (v << np.uint64(2))) & np.uint64(0x9249249249249249)
        return v

    codes = _spread(scaled[:, 0]) | (_spread(scaled[:, 1]) << np.uint64(1)) | (_spread(scaled[:, 2]) << np.uint64(2))
    return np.argsort(codes, kind="stable")


def _haar_transform_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """Forward multilevel orthonormal Haar DWT along ``axis`` (length must be a power of two).

    Coarse-to-fine layout: index 0 is the fully-averaged (coarsest) coefficient, followed by detail
    coefficients from the coarsest split down to the finest. The per-level butterfly
    ``[[1, 1], [1, -1]] / sqrt(2)`` is orthogonal (and self-inverse), so this and
    :func:`_haar_inverse_axis` are exact adjoints/inverses of one another for any input.
    """
    a = np.moveaxis(np.asarray(a, dtype=float), axis, 0)
    n = a.shape[0]
    coeffs = np.empty_like(a)
    approx = a
    length = n
    write_pos = n
    while length > 1:
        half = length // 2
        evens, odds = approx[0:length:2], approx[1:length:2]
        new_approx = (evens + odds) / np.sqrt(2.0)
        detail = (evens - odds) / np.sqrt(2.0)
        write_pos -= half
        coeffs[write_pos : write_pos + half] = detail
        approx, length = new_approx, half
    coeffs[0:1] = approx
    return np.moveaxis(coeffs, 0, axis)


def _haar_inverse_axis(c: np.ndarray, axis: int) -> np.ndarray:
    """Inverse of :func:`_haar_transform_axis` along ``axis``."""
    c = np.moveaxis(np.asarray(c, dtype=float), axis, 0)
    n = c.shape[0]
    approx = c[0:1].copy()
    length = 1
    read_pos = 1
    while length < n:
        detail = c[read_pos : read_pos + length]
        read_pos += length
        evens = (approx + detail) / np.sqrt(2.0)
        odds = (approx - detail) / np.sqrt(2.0)
        new_approx = np.empty((length * 2,) + approx.shape[1:], dtype=float)
        new_approx[0::2] = evens
        new_approx[1::2] = odds
        approx, length = new_approx, length * 2
    return np.moveaxis(approx, 0, axis)


def _threshold_by_energy(coeffs: np.ndarray, rtol: float, g_norm: float) -> np.ndarray:
    """Zero out the smallest-magnitude coefficients whose combined (dropped) energy stays within
    ``(rtol * g_norm)^2`` -- a best-M-term approximation. Because the Haar transform is orthonormal, the
    Frobenius norm is preserved (``||coeffs||_F == ||G||_F``), so this directly bounds the reconstruction
    error: ``||G - G_approx||_F <= rtol * ||G||_F``, and therefore ``||(G - G_approx) @ x|| <= rtol *
    ||G||_F * ||x||`` for any ``x`` (submultiplicativity of the induced operator norm by the Frobenius norm).
    """
    flat = coeffs.reshape(-1)
    order = np.argsort(np.abs(flat))  # ascending magnitude -> candidates to drop, smallest first
    budget = (rtol * max(g_norm, 1e-300)) ** 2
    cum_dropped = np.cumsum(flat[order] ** 2)
    n_drop = int(np.searchsorted(cum_dropped, budget, side="right"))
    thresh = coeffs.copy()
    thresh.reshape(-1)[order[:n_drop]] = 0.0
    return thresh


def wavelet_compress(G: np.ndarray, coords, *, rtol: float = 1e-3) -> LinearOperator:
    """Compress a dense sensitivity matrix ``G`` into a wavelet-thresholded :class:`LinearOperator`.

    ``coords`` are the ``(n_cells, 3)`` cell coordinates ``G``'s columns correspond to (used only to build
    the octree/Morton column ordering -- ``G`` itself is not assumed sparse or structured). The returned
    operator's ``matvec``/``@``/``.matmul`` reproduces ``G @ x`` from the thresholded coefficient array,
    which -- for a ``G`` that is smooth in the cell coordinate (any real potential-field kernel) -- typically
    stores far fewer than ``G.size`` nonzeros for a given ``rtol``.

    Args:
        G: (n_obs, n_cells) dense sensitivity matrix.
        coords: (n_cells, 3) cell coordinates (cell centres, or any consistent per-column location).
        rtol: relative Frobenius-energy budget for the coefficients dropped by thresholding (see
            :func:`_threshold_by_energy`); smaller keeps more coefficients / is more accurate.

    Returns:
        A ``scipy.sparse.linalg.LinearOperator`` of shape ``(n_obs, n_cells)`` with an extra ``.matmul``
        alias for ``matvec``, an ``.nnz`` attribute (number of stored coefficients), and a ``.coeffs``
        attribute (the thresholded sparse coefficient matrix, for introspection).
    """
    G = np.asarray(G, dtype=float)
    n_obs, n_cells = G.shape
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] != n_cells:
        raise ValueError(f"coords has {coords.shape[0]} rows but G has {n_cells} columns.")
    order = _morton_order(coords)
    n_obs_pad, n_cells_pad = _next_pow2(n_obs), _next_pow2(n_cells)

    g_pad = np.zeros((n_obs_pad, n_cells_pad), dtype=float)
    g_pad[:n_obs, :n_cells] = G[:, order]
    coeffs = _haar_transform_axis(_haar_transform_axis(g_pad, axis=0), axis=1)
    coeffs = _threshold_by_energy(coeffs, rtol, np.linalg.norm(G))
    coeffs_sparse = sp.csr_matrix(coeffs)
    coeffs_sparse.eliminate_zeros()

    def matvec(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        xp = np.zeros(n_cells_pad, dtype=float)
        xp[:n_cells] = x[order]
        u = _haar_transform_axis(xp, axis=0)
        v = coeffs_sparse @ u
        y_pad = _haar_inverse_axis(v, axis=0)
        return y_pad[:n_obs]

    def rmatvec(y):
        y = np.asarray(y, dtype=float).reshape(-1)
        yp = np.zeros(n_obs_pad, dtype=float)
        yp[:n_obs] = y
        w = _haar_transform_axis(yp, axis=0)
        z = coeffs_sparse.T @ w
        xp = _haar_inverse_axis(z, axis=0)
        out = np.zeros(n_cells, dtype=float)
        out[order] = xp[:n_cells]
        return out

    op = LinearOperator(shape=(n_obs, n_cells), matvec=matvec, rmatvec=rmatvec, dtype=float)
    op.matmul = op.matvec  # DoD-convenience alias; scipy's LinearOperator has no `.matmul` of its own
    op.nnz = int(coeffs_sparse.nnz)
    op.coeffs = coeffs_sparse
    return op
