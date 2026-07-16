"""Tests for mixle_pde.reduced_basis: POD basis construction, projection round-trip, and the
independent held-out reconstruction-error certificate.

Two of the tests below (basis-by-rank, basis-by-energy-threshold, and the mutual-exclusivity /
shape-validation checks) exercise :mod:`mixle_pde.reduced_basis` as pure snapshot-matrix linear algebra
against small hand-built matrices with an exactly known singular-value spectrum, so the "smallest
sufficient rank" and orthonormality properties can be checked analytically rather than just eyeballed.

The other two build a *realistic* snapshot family from an already-registered, unmodified kernel --
:class:`mixle_pde.dynamics.AdvectionDiffusionOperator` -- by sweeping its diffusivity parameter over a
training grid and a disjoint (interleaved) held-out grid, taking several time snapshots per diffusivity
value via repeated application of ``.transition_matrix(dt)``. A basis fit only on the training family is
then checked two ways: it reconstructs a training snapshot (one it was fit from) essentially exactly, and
it reconstructs the held-out family (diffusivities it never saw) with error that is small but genuinely
nonzero -- and that error must measurably shrink as the basis is given more modes, which is the actual
point of a reduced-basis method (more modes buys more accuracy) rather than an artifact of any one rank
choice.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.dynamics import AdvectionDiffusionOperator
from mixle_pde.reduced_basis import (
    PODBasis,
    build_pod_basis,
    project,
    reconstruct,
    reduced_basis_error,
)

# --- realistic snapshot family: advection-diffusion swept over diffusivity, several times each -------
_GRID_N = 48
_LENGTH = 1.0
_VELOCITY = 0.3
_DT = 0.01
_N_STEPS = (5, 10, 20, 40)
_TRAIN_DIFFUSIVITIES = np.linspace(0.01, 0.20, 12)
_HELDOUT_DIFFUSIVITIES = _TRAIN_DIFFUSIVITIES[:-1] + np.diff(_TRAIN_DIFFUSIVITIES) / 2.0  # interleaved, disjoint


def _advection_diffusion_snapshots(diffusivities) -> np.ndarray:
    """Snapshot columns of :class:`AdvectionDiffusionOperator` state at several times, for each
    diffusivity in ``diffusivities`` -- a smooth localized bump plus a background sinusoid, advanced by
    repeated application of the operator's own (unmodified) ``transition_matrix(dt)``. Returns
    ``(n_dof, len(diffusivities) * len(_N_STEPS))``.
    """
    x = np.linspace(0.0, _LENGTH, _GRID_N)
    u0 = np.exp(-((x - 0.3 * _LENGTH) ** 2) / (2.0 * (0.05 * _LENGTH) ** 2)) + 0.3 * np.sin(2.0 * np.pi * x)

    columns = []
    for diffusivity in diffusivities:
        operator = AdvectionDiffusionOperator(
            diffusivity=float(diffusivity),
            velocity=_VELOCITY,
            n=_GRID_N,
            length=_LENGTH,
            bc="periodic",
            scheme="implicit",
        )
        transition = operator.transition_matrix(_DT)
        state = u0.copy()
        for step in range(1, max(_N_STEPS) + 1):
            state = transition @ state
            if step in _N_STEPS:
                columns.append(state.copy())
    return np.stack(columns, axis=1)


@pytest.fixture(scope="module")
def train_snapshots() -> np.ndarray:
    return _advection_diffusion_snapshots(_TRAIN_DIFFUSIVITIES)


@pytest.fixture(scope="module")
def heldout_snapshots() -> np.ndarray:
    return _advection_diffusion_snapshots(_HELDOUT_DIFFUSIVITIES)


def test_train_and_heldout_diffusivities_are_disjoint():
    # Sanity-check the fixture design itself: the held-out family must never have been seen in training.
    assert set(_TRAIN_DIFFUSIVITIES.tolist()).isdisjoint(set(_HELDOUT_DIFFUSIVITIES.tolist()))


# --- build_pod_basis: rank selection -------------------------------------------------------------------


def test_build_pod_basis_by_rank_returns_requested_orthonormal_modes(train_snapshots):
    basis = build_pod_basis(train_snapshots, rank=5)

    assert isinstance(basis, PODBasis)
    assert basis.rank == 5
    assert basis.n_dof == _GRID_N
    assert basis.modes.shape == (_GRID_N, 5)
    assert basis.singular_values.shape == (5,)
    assert basis.n_snapshots == train_snapshots.shape[1]

    # POD modes are left singular vectors of the snapshot matrix: orthonormal columns.
    gram = basis.modes.T @ basis.modes
    assert np.allclose(gram, np.eye(5), atol=1e-9)

    # singular values are reported in descending order.
    assert np.all(np.diff(basis.singular_values) <= 0.0)

    # a 5-mode basis cannot capture more energy than the full-rank basis, and less than a 1-mode basis.
    assert 0.0 < basis.energy_fraction <= 1.0


def test_build_pod_basis_rank_exceeding_max_rank_raises():
    snapshots = np.random.default_rng(0).standard_normal((10, 4))  # max_rank = min(10, 4) = 4
    with pytest.raises(ValueError, match="exceeds the maximum possible rank"):
        build_pod_basis(snapshots, rank=5)


# --- build_pod_basis: energy-threshold selection -------------------------------------------------------


def test_build_pod_basis_energy_threshold_picks_minimal_sufficient_rank():
    # Hand-built snapshot matrix with an EXACTLY known singular-value spectrum, so the "smallest rank
    # whose cumulative energy clears the threshold" property can be checked analytically rather than
    # just observed.
    rng = np.random.default_rng(42)
    n_dof, n_snapshots = 30, 10
    s_true = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
    u_true, _ = np.linalg.qr(rng.standard_normal((n_dof, s_true.size)))
    v_true, _ = np.linalg.qr(rng.standard_normal((n_snapshots, s_true.size)))
    snapshots = (u_true * s_true) @ v_true.T

    total_energy = float(np.sum(s_true**2))
    cumulative_energy = np.cumsum(s_true**2) / total_energy
    threshold = 0.995
    expected_rank = int(np.searchsorted(cumulative_energy, threshold, side="left")) + 1
    assert 1 < expected_rank < s_true.size  # sanity: the example is neither degenerate nor saturating

    basis = build_pod_basis(snapshots, energy_threshold=threshold)

    assert basis.rank == expected_rank
    assert basis.energy_fraction >= threshold
    # minimality: one fewer mode must NOT have met the threshold.
    assert cumulative_energy[expected_rank - 2] < threshold


def test_build_pod_basis_energy_threshold_out_of_range_raises(train_snapshots):
    with pytest.raises(ValueError, match="energy_threshold"):
        build_pod_basis(train_snapshots, energy_threshold=0.0)
    with pytest.raises(ValueError, match="energy_threshold"):
        build_pod_basis(train_snapshots, energy_threshold=1.5)


# --- rank / energy_threshold mutual exclusivity ----------------------------------------------------


def test_build_pod_basis_requires_exactly_one_of_rank_or_energy_threshold():
    snapshots = np.random.default_rng(1).standard_normal((6, 4))
    with pytest.raises(ValueError, match="exactly one"):
        build_pod_basis(snapshots)
    with pytest.raises(ValueError, match="exactly one"):
        build_pod_basis(snapshots, rank=2, energy_threshold=0.9)


# --- project / reconstruct round trip on a TRAINING snapshot --------------------------------------------


def test_project_reconstruct_round_trip_for_a_training_snapshot(train_snapshots):
    # A basis capturing (numerically) the full column rank of the training matrix must reconstruct any
    # snapshot that went into fitting it almost exactly -- that snapshot lies exactly in the modes' span.
    max_rank = min(train_snapshots.shape)
    basis = build_pod_basis(train_snapshots, rank=max_rank)

    for column in (0, train_snapshots.shape[1] // 2, train_snapshots.shape[1] - 1):
        field = train_snapshots[:, column]
        coeffs = project(field, basis)
        assert coeffs.shape == (max_rank,)
        recon = reconstruct(coeffs, basis)
        assert recon.shape == field.shape
        assert np.max(np.abs(recon - field)) < 1e-8


def test_project_reconstruct_round_trip_batched(train_snapshots):
    # project/reconstruct also accept a batch of fields (n_dof, n_fields) at once.
    max_rank = min(train_snapshots.shape)
    basis = build_pod_basis(train_snapshots, rank=max_rank)

    coeffs = project(train_snapshots, basis)
    assert coeffs.shape == (max_rank, train_snapshots.shape[1])
    recon = reconstruct(coeffs, basis)
    assert recon.shape == train_snapshots.shape
    assert np.max(np.abs(recon - train_snapshots)) < 1e-8


# --- reduced_basis_error: genuine held-out generalization check -----------------------------------------


def test_reduced_basis_error_on_holdout_is_small_but_nonzero(train_snapshots, heldout_snapshots):
    basis = build_pod_basis(train_snapshots, rank=8)  # a moderate rank, well short of max_rank=48
    error = reduced_basis_error(heldout_snapshots, basis)

    assert error.n_heldout == heldout_snapshots.shape[1]
    assert error.per_snapshot_error.shape == (heldout_snapshots.shape[1],)

    # genuinely nonzero: this is held-out data the basis never fit, not a training-error report.
    assert error.max_error > 1e-6
    assert error.l2_error > 1e-6
    # but still small: the diffusivity family is smooth, so a rank-8 basis generalizes well.
    assert error.max_error < 0.1
    assert error.relative_l2_error < 0.01


def test_reduced_basis_error_shape_validation(train_snapshots):
    basis = build_pod_basis(train_snapshots, rank=4)
    with pytest.raises(ValueError):
        reduced_basis_error(np.zeros(basis.n_dof + 1), basis)
    with pytest.raises(ValueError):
        project(np.zeros(basis.n_dof + 1), basis)
    with pytest.raises(ValueError):
        reconstruct(np.zeros(basis.rank + 1), basis)


# --- the actual point of a reduced basis: more modes measurably reduces held-out error ------------------


def test_reduced_basis_error_shrinks_monotonically_as_rank_grows(train_snapshots, heldout_snapshots):
    ranks = [1, 2, 4, 8, 16]
    max_errors = []
    l2_errors = []
    for rank in ranks:
        basis = build_pod_basis(train_snapshots, rank=rank)
        error = reduced_basis_error(heldout_snapshots, basis)
        max_errors.append(error.max_error)
        l2_errors.append(error.l2_error)

    # a basis built from fewer modes than the data's true rank must generalize measurably worse: each
    # successive (strictly larger-rank) basis is strictly more accurate on the SAME held-out set --
    # never just "below some threshold", but an actual monotonically-decreasing sequence.
    for i in range(len(ranks) - 1):
        assert max_errors[i + 1] < max_errors[i], (ranks, max_errors)
        assert l2_errors[i + 1] < l2_errors[i], (ranks, l2_errors)

    # and the effect is real, not floating-point noise: an order-of-magnitude improvement from the
    # sparsest basis (rank=1) to the richest one tested (rank=16).
    assert max_errors[0] > 10.0 * max_errors[-1]
    assert l2_errors[0] > 10.0 * l2_errors[-1]
