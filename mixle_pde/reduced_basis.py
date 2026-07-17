"""Proper Orthogonal Decomposition (POD) reduced-basis projection + held-out error certification.

A parametrized PDE solved at many parameter values (a diffusivity, a wave speed, a material property, ...)
produces a family of full-order solution fields that, for a well-behaved problem, sweeps out a
low-dimensional manifold inside the (typically much larger) full-order space: nearby parameters give
nearby solutions, so the *snapshot matrix* ``S`` (each column one full-order solution, shape
``(n_dof, n_snapshots)``) is usually well approximated by a rank-``r`` subspace for ``r << n_dof``. POD
(a.k.a. the "method of snapshots", Sirovich 1987) finds that subspace directly from ``S`` via its SVD:
``S = U diag(s) V^T``, and the leading columns of ``U`` (the *POD modes*) are, by the Eckart-Young theorem,
the orthonormal basis that best approximates every column of ``S`` in a least-squares sense for a given
rank. :func:`build_pod_basis` fits that basis; :func:`project` / :func:`reconstruct` are the linear maps
into and out of the reduced coordinates; :func:`reduced_basis_error` certifies how well the fitted basis
generalizes to snapshots it was never fit from -- the only way to know whether a chosen rank is actually
enough, rather than merely fitting the training snapshots well.

This module is solver-agnostic: it operates purely on a caller-supplied snapshot matrix and never imports
a PDE kernel itself. A caller builds that matrix however is convenient -- for example by sweeping a
parameter of :class:`mixle_pde.dynamics.AdvectionDiffusionOperator` (via its
``.transition_matrix(dt)``) or by calling :func:`mixle_pde.fem.solve_simplex_poisson` at several parameter
values and stacking the resulting fields as columns.

Scope note: this module is deliberately limited to POD basis-fitting, projection/reconstruction, and
independent reconstruction-error certification. It does *not* build a Galerkin-projected reduced operator
(that needs the governing weak form, not just snapshots), does not implement hyper-reduction/DEIM for
nonlinear terms, and does not provide transient reduced-order state-space integration -- all three need
weak-form machinery this stack does not yet expose. What is here is the linear-algebra foundation those
would eventually sit on top of.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PODBasis",
    "ReducedBasisError",
    "build_pod_basis",
    "project",
    "reconstruct",
    "reduced_basis_error",
]


@dataclass(frozen=True)
class PODBasis:
    """A fitted proper-orthogonal-decomposition reduced basis.

    ``modes`` (shape ``(n_dof, rank)``) are orthonormal left singular vectors of the snapshot matrix
    :func:`build_pod_basis` was fit from, ordered by decreasing captured energy; ``singular_values``
    (shape ``(rank,)``) are their corresponding singular values. ``energy_fraction`` is the fraction of
    the fitting snapshot matrix's total singular-value energy (``sum(singular_values**2)``) these
    ``rank`` modes capture, and ``n_snapshots`` records how many snapshot columns the basis was fit
    from. Use :func:`project` / :func:`reconstruct` to move fields into and out of this basis, and
    :func:`reduced_basis_error` to certify accuracy on data the basis has not seen.
    """

    modes: np.ndarray
    singular_values: np.ndarray
    energy_fraction: float
    n_snapshots: int

    def __post_init__(self) -> None:
        if self.modes.ndim != 2:
            raise ValueError(f"PODBasis.modes must be 2-D (n_dof, rank); got shape {self.modes.shape}.")
        if self.singular_values.shape != (self.modes.shape[1],):
            raise ValueError(
                "PODBasis.singular_values must have shape (rank,) matching modes.shape[1]; got "
                f"modes {self.modes.shape}, singular_values {self.singular_values.shape}."
            )

    @property
    def n_dof(self) -> int:
        """Full-order dimension the basis' modes live in."""
        return int(self.modes.shape[0])

    @property
    def rank(self) -> int:
        """Number of retained POD modes (the reduced-order dimension)."""
        return int(self.modes.shape[1])


@dataclass(frozen=True)
class ReducedBasisError:
    """Held-out reconstruction-error certificate produced by :func:`reduced_basis_error`.

    ``max_error`` / ``l2_error`` are, respectively, the worst-case single-snapshot Euclidean
    reconstruction error and the aggregate (Frobenius-norm) reconstruction error over every held-out
    snapshot; ``relative_l2_error`` normalizes ``l2_error`` by the held-out snapshot matrix's own norm.
    ``per_snapshot_error`` (shape ``(n_heldout,)``) gives the individual Euclidean reconstruction error
    of each held-out snapshot, for introspection.
    """

    max_error: float
    l2_error: float
    relative_l2_error: float
    per_snapshot_error: np.ndarray
    n_heldout: int


def build_pod_basis(
    snapshots,
    *,
    rank: int | None = None,
    energy_threshold: float | None = None,
) -> PODBasis:
    """Fit a POD reduced basis from a ``(n_dof, n_snapshots)`` snapshot matrix via truncated SVD.

    Exactly one of ``rank`` or ``energy_threshold`` must be given:

    * ``rank``: keep exactly this many leading POD modes.
    * ``energy_threshold`` (e.g. ``0.999``): keep the smallest number of leading modes whose cumulative
      singular-value energy (``sum(s[:r]**2) / sum(s**2)``) is at least this fraction.

    This is the classic "method of snapshots": the economy SVD ``snapshots = U @ diag(s) @ Vt`` is
    computed once (via ``numpy.linalg.svd``, truncated to ``min(n_dof, n_snapshots)`` by
    ``full_matrices=False``), and the returned basis is the leading ``rank`` columns of ``U`` -- by the
    Eckart-Young theorem, the best possible rank-``rank`` orthonormal approximating subspace for the
    training snapshots in the least-squares sense. Use :func:`reduced_basis_error` against snapshots
    *not* passed here to check whether that rank also generalizes.
    """
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be a 2-D (n_dof, n_snapshots) array; got shape {snapshots.shape}.")
    n_dof, n_snapshots = snapshots.shape
    if n_dof < 1 or n_snapshots < 1:
        raise ValueError(f"snapshots must be non-empty; got shape {snapshots.shape}.")
    if (rank is None) == (energy_threshold is None):
        raise ValueError(
            "build_pod_basis requires exactly one of `rank` or `energy_threshold` (never both, never "
            f"neither); got rank={rank!r}, energy_threshold={energy_threshold!r}."
        )

    u, s, _vt = np.linalg.svd(snapshots, full_matrices=False)
    max_rank = int(s.shape[0])
    total_energy = float(np.sum(s * s))

    if rank is not None:
        rank = int(rank)
        if rank < 1:
            raise ValueError(f"rank must be a positive integer; got {rank!r}.")
        if rank > max_rank:
            raise ValueError(
                f"rank={rank} exceeds the maximum possible rank {max_rank} = min(n_dof={n_dof}, "
                f"n_snapshots={n_snapshots}) of the snapshot matrix."
            )
        chosen_rank = rank
    else:
        energy_threshold = float(energy_threshold)
        if not (0.0 < energy_threshold <= 1.0):
            raise ValueError(f"energy_threshold must be a number in (0, 1]; got {energy_threshold!r}.")
        if total_energy <= 0.0:
            chosen_rank = 1  # degenerate all-zero snapshot matrix: no rank captures any energy
        else:
            cumulative_energy = np.cumsum(s * s) / total_energy
            chosen_rank = int(np.searchsorted(cumulative_energy, energy_threshold, side="left")) + 1
            chosen_rank = min(chosen_rank, max_rank)

    modes = np.array(u[:, :chosen_rank], dtype=float)
    values = np.array(s[:chosen_rank], dtype=float)
    energy_fraction = float(np.sum(values * values) / total_energy) if total_energy > 0.0 else 1.0

    return PODBasis(
        modes=modes,
        singular_values=values,
        energy_fraction=energy_fraction,
        n_snapshots=n_snapshots,
    )


def project(field, basis: PODBasis) -> np.ndarray:
    """Project a full-order field onto ``basis``'s reduced coordinates: ``coeffs = basis.modes.T @ field``.

    ``field`` is ``(n_dof,)`` for a single field, or ``(n_dof, n_fields)`` for a batch of fields sharing
    ``basis``; the returned reduced coordinates have shape ``(rank,)`` or ``(rank, n_fields)``
    respectively. Pair with :func:`reconstruct` to go back to full order.
    """
    field = np.asarray(field, dtype=float)
    if field.ndim not in (1, 2):
        raise ValueError(f"field must be 1-D (n_dof,) or 2-D (n_dof, n_fields); got shape {field.shape}.")
    if field.shape[0] != basis.n_dof:
        raise ValueError(f"field's leading dimension must be basis.n_dof={basis.n_dof}; got shape {field.shape}.")
    return basis.modes.T @ field


def reconstruct(coeffs, basis: PODBasis) -> np.ndarray:
    """Reconstruct a full-order field from ``basis``'s reduced coordinates: ``basis.modes @ coeffs``.

    ``coeffs`` is ``(rank,)`` for a single field, or ``(rank, n_fields)`` for a batch (as produced by
    :func:`project`); the returned full-order field(s) have shape ``(n_dof,)`` or ``(n_dof, n_fields)``.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim not in (1, 2):
        raise ValueError(f"coeffs must be 1-D (rank,) or 2-D (rank, n_fields); got shape {coeffs.shape}.")
    if coeffs.shape[0] != basis.rank:
        raise ValueError(f"coeffs' leading dimension must be basis.rank={basis.rank}; got shape {coeffs.shape}.")
    return basis.modes @ coeffs


def reduced_basis_error(heldout_snapshots, basis: PODBasis) -> ReducedBasisError:
    """Certify ``basis``'s reconstruction accuracy against snapshots it was NOT fit from.

    ``heldout_snapshots`` is ``(n_dof,)`` for a single held-out field or ``(n_dof, n_heldout)`` for a
    batch. Every column is projected and reconstructed through ``basis`` (via :func:`project` /
    :func:`reconstruct`) and compared to the original -- this is a genuine generalization check: pass
    snapshots that were excluded from the matrix :func:`build_pod_basis` fit ``basis`` from, not
    snapshots the basis was trained on (which would trivially reconstruct well for any rank capturing
    the training data's full column space and understate the basis' real-world error).
    """
    heldout = np.asarray(heldout_snapshots, dtype=float)
    if heldout.ndim == 1:
        heldout = heldout[:, None]
    if heldout.ndim != 2:
        raise ValueError(
            f"heldout_snapshots must be 1-D (n_dof,) or 2-D (n_dof, n_heldout); got shape {heldout.shape}."
        )
    if heldout.shape[0] != basis.n_dof:
        raise ValueError(
            f"heldout_snapshots' leading dimension must be basis.n_dof={basis.n_dof}; got shape {heldout.shape}."
        )
    if heldout.shape[1] < 1:
        raise ValueError("heldout_snapshots must contain at least one snapshot column.")

    residual = heldout - reconstruct(project(heldout, basis), basis)
    per_snapshot_error = np.linalg.norm(residual, axis=0)
    heldout_norm = float(np.linalg.norm(heldout))
    l2_error = float(np.linalg.norm(residual))

    return ReducedBasisError(
        max_error=float(np.max(per_snapshot_error)),
        l2_error=l2_error,
        relative_l2_error=float(l2_error / heldout_norm) if heldout_norm > 0.0 else 0.0,
        per_snapshot_error=per_snapshot_error,
        n_heldout=int(heldout.shape[1]),
    )
