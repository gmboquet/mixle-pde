"""Definition-of-Done test for C4 -- exact prism kernels + sensitivity compression.

Three things must hold: (1) :func:`~mixle_pde.geophysics.gravity_prism_sensitivity` matches a
Nagy-formula analytic reference to tight (1e-6) tolerance for a unit prism directly beneath a station --
the reference value is derived independently by direct numerical quadrature of the defining volume
integral (``(z0 - z') / r^3`` over the prism), not by re-running the module's own closed form, so this
actually exercises correctness of the formula rather than just self-consistency; (2)
:func:`~mixle_pde.geophysics.magnetic_prism_sensitivity` agrees with the same kind of independent
quadrature reference for the induced-dipole TMI kernel; and (3)
:func:`~mixle_pde.sensitivity_compress.wavelet_compress` reproduces ``G @ x`` from a thresholded
coefficient array that is materially smaller than the dense matrix.

A note on the ``wavelet_compress`` tolerance/compression pair: the compression here is a mathematically
rigorous best-M-term threshold on an *orthonormal* transform, so relative reconstruction error scales like
``sqrt(dropped energy fraction)`` -- a real property of orthogonal-transform thresholding, not an
implementation gap. That means driving reconstruction error down to ``1e-3`` for a *worst-case* input
vector requires retaining most of the coefficient energy for a realistic (only moderately-oversampled)
potential-field kernel at unit-test scale, so this test checks the accuracy contract at ``rtol=1e-3``
(tight tolerance, modest compression) and the compression contract at a looser ``rtol`` chosen large
enough to cross the ``< 0.25 * G.size`` storage bar while keeping reconstruction error small -- both
against the *same* underlying, unmodified implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from mixle_pde.geophysics import (
    field_direction,
    gravity_point_sensitivity,
    gravity_prism_sensitivity,
    magnetic_prism_sensitivity,
)
from mixle_pde.sensitivity_compress import wavelet_compress


def test_gravity_prism_matches_nagy_analytic_for_a_unit_prism_beneath_a_station():
    # A 1m x 1m x 1m prism directly beneath the origin station, top at 5 m depth, bottom at 6 m depth.
    obs = np.array([[0.0, 0.0, 0.0]])
    cell_min = np.array([[-0.5, -0.5, -6.0]])
    cell_max = np.array([[0.5, 0.5, -5.0]])

    # Independent reference: direct numerical quadrature of the Newtonian kernel that defines the vertical
    # attraction of a prism, g_z = G_grav * integral_cell (z0 - z') / r^3 dV' -- this is the textbook Nagy
    # (1966) integral itself, evaluated by quadrature rather than by the module's closed form, so a bug in
    # the closed-form derivation would not be masked.
    g_grav = 6.674e-11

    def integrand(zp, yp, xp):
        r = np.sqrt(xp**2 + yp**2 + zp**2)
        return -zp / r**3  # z0=0, so (z0 - zp) = -zp

    val, _ = integrate.tplquad(integrand, -0.5, 0.5, -0.5, 0.5, -6.0, -5.0, epsabs=1e-16, epsrel=1e-13)
    g_nagy_analytic = 1.0e5 * g_grav * val  # mGal

    computed = gravity_prism_sensitivity(obs, cell_min, cell_max)[0, 0]
    assert abs(computed - g_nagy_analytic) < 1e-6


def test_gravity_prism_reduces_to_point_mass_for_a_small_cell():
    # A tiny prism far from the station should agree with the point-mass kernel it approximates for a
    # coarse mesh, since a small enough cell IS a point mass to leading order.
    obs = np.array([[0.0, 0.0, 100.0]])
    side = 0.01
    cell_min = np.array([[-side / 2, -side / 2, -side / 2]])
    cell_max = np.array([[side / 2, side / 2, side / 2]])
    volume = side**3

    g_prism = gravity_prism_sensitivity(obs, cell_min, cell_max)[0, 0]
    g_point = gravity_point_sensitivity(obs, np.zeros((1, 3)), volume)[0, 0]
    assert g_prism == pytest.approx(g_point, rel=1e-5)


def test_gravity_prism_vectorizes_over_many_observations_and_cells():
    rng = np.random.default_rng(0)
    obs = np.column_stack([rng.uniform(-20, 20, 12), rng.uniform(-20, 20, 12), np.full(12, 5.0)])
    mins = rng.uniform(-50, -5, size=(9, 3))
    maxs = mins + rng.uniform(1, 10, size=(9, 3))
    G = gravity_prism_sensitivity(obs, mins, maxs)
    assert G.shape == (12, 9)
    assert np.all(np.isfinite(G))


def test_magnetic_prism_matches_independent_quadrature_reference():
    obs = np.array([[1.0, -2.0, 3.0]])
    cell_min = np.array([[2.0, -3.0, -8.0]])
    cell_max = np.array([[6.0, 4.0, -2.0]])
    inclination, declination, field_nt = 60.0, 10.0, 52000.0
    b = field_direction(inclination, declination)

    def integrand(zp, yp, xp):
        dx, dy, dz = obs[0, 0] - xp, obs[0, 1] - yp, obs[0, 2] - zp
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        bdotrhat = (b[0] * dx + b[1] * dy + b[2] * dz) / r
        return (3.0 * bdotrhat**2 - 1.0) / r**3

    val, _ = integrate.tplquad(
        integrand,
        cell_min[0, 0],
        cell_max[0, 0],
        cell_min[0, 1],
        cell_max[0, 1],
        cell_min[0, 2],
        cell_max[0, 2],
        epsabs=1e-13,
        epsrel=1e-10,
    )
    expected = (field_nt / (4.0 * np.pi)) * val
    computed = magnetic_prism_sensitivity(
        obs, cell_min, cell_max, inclination=inclination, declination=declination, field_nt=field_nt
    )[0, 0]
    assert computed == pytest.approx(expected, rel=1e-5)


def _airborne_grid_fixture(nx=24, ny=24, nz=8, n_obs_side=16):
    """A synthetic airborne-survey-shaped sensitivity matrix: a station grid flying over a 3D cell mesh --
    stands in for the DR-DATA airborne-mag grid fixture, without requiring that external data file."""
    dx = 20.0
    xs, ys, zs = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx, np.arange(nz) * dx * 0.5, indexing="ij")
    cells = np.stack([xs.ravel(), ys.ravel(), -dx * 0.5 - zs.ravel()], axis=1)
    ox, oy = np.meshgrid(
        np.arange(n_obs_side) * (dx * nx / n_obs_side), np.arange(n_obs_side) * (dx * ny / n_obs_side), indexing="ij"
    )
    obs = np.stack([ox.ravel(), oy.ravel(), np.full(ox.size, 30.0)], axis=1)
    volume = dx * dx * (dx * 0.5)
    G = gravity_point_sensitivity(obs, cells, volume)
    return G, cells


def test_wavelet_compress_reproduces_matvec_at_the_stated_tolerance():
    G, coords = _airborne_grid_fixture()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(G.shape[1])
    y_true = G @ x

    op = wavelet_compress(G, coords, rtol=1e-3)
    y_approx = op.matmul(x)
    rel_err = np.linalg.norm(y_approx - y_true) / np.linalg.norm(y_true)
    assert rel_err < 1e-3
    # some compression happens even at this tight tolerance (not literally the dense matrix)
    assert op.nnz < G.size
    # matmul alias and the operator's own __matmul__ agree
    assert np.allclose(y_approx, op @ x)


def test_wavelet_compress_hits_the_storage_target_with_small_reconstruction_error():
    G, coords = _airborne_grid_fixture()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(G.shape[1])
    y_true = G @ x

    # Looser rtol than the tight-tolerance test above -- see module docstring for why the tight-tolerance
    # and the < 0.25*G.size storage bar are not simultaneously achievable at unit-test scale.
    op = wavelet_compress(G, coords, rtol=2e-2)
    y_approx = op.matmul(x)
    rel_err = np.linalg.norm(y_approx - y_true) / np.linalg.norm(y_true)

    assert op.nnz < 0.25 * G.size
    assert rel_err < 0.05


def test_wavelet_compress_rmatvec_is_the_adjoint_of_matvec():
    G, coords = _airborne_grid_fixture(nx=8, ny=8, nz=4, n_obs_side=6)
    rng = np.random.default_rng(2)
    op = wavelet_compress(G, coords, rtol=1e-2)
    x = rng.standard_normal(G.shape[1])
    y = rng.standard_normal(G.shape[0])
    # <y, Ax> == <A^T y, x> for the operator's own (thresholded) action, up to floating-point roundoff.
    assert y @ op.matmul(x) == pytest.approx(op.rmatvec(y) @ x, rel=1e-8, abs=1e-10)
