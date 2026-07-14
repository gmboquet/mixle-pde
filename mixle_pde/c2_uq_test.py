"""C2 -- matrix-free / low-rank Hessian UQ + selected inversion (workstream C, inversion realism & scale).

Definition of Done (verbatim from the work order): build a banded SPD precision at n=100_000 cells, call
``SparsePosteriorPrecision(P).marginal_variance()`` under ``tracemalloc``, assert peak memory < 300 MB (a
dense (n, n) float64 inverse would be 80 GB), and on a small n=500 cross-check assert the result matches
``np.diag(np.linalg.inv(P.toarray()))`` within ``rtol=1e-6``. The remaining test cases exercise the rest
of the C2 public surface: the randomized low-rank Hessian eigendecomposition, a general (non-banded)
selected-inversion cross-check, the optional low-rank UQ attachment on
``sparse_linear_gaussian_invert``, and the matrix-free Gauss-Newton Hessian HVP.
"""

import tracemalloc
import unittest

import numpy as np
import scipy.sparse as sp

from mixle_pde.field_gauss_newton import _transform_jacobian_diag, gauss_newton_hessian_hvp
from mixle_pde.field_inversion import FieldGaussianPrior, _noise_precision, sparse_linear_gaussian_invert
from mixle_pde.latent import Field3D, SparsePosteriorPrecision
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    gravity_forward_operator,
)
from mixle_pde.uq_lowrank import randomized_lowrank_hessian, takahashi_selected_inversion


def _banded_spd(n: int, bandwidth: int, *, diag_value: float = 8.0, off_value: float = 0.5) -> sp.csc_matrix:
    data = [np.full(n, diag_value)]
    offsets = [0]
    for d in range(1, bandwidth + 1):
        band = np.full(n - d, off_value)
        data.append(band)
        offsets.append(d)
        data.append(band)
        offsets.append(-d)
    return sp.diags(data, offsets, shape=(n, n), format="csc")


class MarginalVarianceMemoryAndCorrectnessTestCase(unittest.TestCase):
    """The literal Definition of Done."""

    def test_selected_inversion_is_memory_bounded_at_survey_scale(self):
        n = 100_000
        precision = _banded_spd(n, bandwidth=3)

        tracemalloc.start()
        try:
            result = SparsePosteriorPrecision(precision).marginal_variance()
        finally:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        self.assertEqual(result.shape, (n,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result > 0.0))
        peak_mb = peak / 1.0e6
        self.assertLess(peak_mb, 300.0, f"peak traced memory {peak_mb:.1f} MB exceeded the 300 MB budget")

    def test_selected_inversion_matches_dense_reference_at_small_n(self):
        n = 500
        precision = _banded_spd(n, bandwidth=3)
        result = SparsePosteriorPrecision(precision).marginal_variance()
        dense_diag = np.diag(np.linalg.inv(precision.toarray()))
        np.testing.assert_allclose(result, dense_diag, rtol=1.0e-6)

    def test_dense_method_is_still_available_as_an_explicit_opt_in(self):
        n = 40
        precision = _banded_spd(n, bandwidth=2)
        sparse_result = SparsePosteriorPrecision(precision).marginal_variance(method="selected_inversion")
        dense_result = SparsePosteriorPrecision(precision).marginal_variance(method="dense")
        np.testing.assert_allclose(sparse_result, dense_result, rtol=1.0e-8)

    def test_unknown_method_raises(self):
        precision = _banded_spd(10, bandwidth=1)
        with self.assertRaises(ValueError):
            SparsePosteriorPrecision(precision).marginal_variance(method="bogus")


class TakahashiSelectedInversionTestCase(unittest.TestCase):
    """`takahashi_selected_inversion` on its own, including a genuinely non-banded sparsity pattern."""

    def test_matches_dense_inverse_on_banded_precision(self):
        n = 60
        precision = _banded_spd(n, bandwidth=4)
        got = takahashi_selected_inversion(precision)
        expected = np.diag(np.linalg.inv(precision.toarray()))
        np.testing.assert_allclose(got, expected, rtol=1.0e-8)

    def test_matches_dense_inverse_on_a_general_sparse_pattern(self):
        rng = np.random.default_rng(3)
        n = 45
        m = rng.standard_normal((n, n))
        dense = 0.5 * (m + m.T) + n * np.eye(n)  # diagonally dominant -> SPD, not banded
        precision = sp.csc_matrix(dense)
        got = takahashi_selected_inversion(precision)
        expected = np.diag(np.linalg.inv(dense))
        np.testing.assert_allclose(got, expected, rtol=1.0e-8)

    def test_single_cell_precision(self):
        precision = sp.csc_matrix(np.array([[4.0]]))
        np.testing.assert_allclose(takahashi_selected_inversion(precision), [0.25])

    def test_rejects_non_square_input(self):
        with self.assertRaises(ValueError):
            takahashi_selected_inversion(sp.csc_matrix(np.zeros((3, 4))))


class RandomizedLowRankHessianTestCase(unittest.TestCase):
    """The randomized eigendecomposition on a symmetric PSD operator accessed only via HVP."""

    def test_recovers_top_eigenpairs_of_a_fast_decaying_spectrum(self):
        rng = np.random.default_rng(7)
        n, true_rank = 200, 5
        basis = np.linalg.qr(rng.standard_normal((n, true_rank)))[0]
        true_eigs = np.array([50.0, 30.0, 12.0, 4.0, 1.0])
        H = (basis * true_eigs) @ basis.T  # exactly rank-5 PSD, easy target for randomized SVD

        def hvp(v):
            return H @ v

        U, s = randomized_lowrank_hessian(hvp, n, k=true_rank, rng=np.random.default_rng(11), oversample=10)
        self.assertEqual(U.shape, (n, true_rank))
        self.assertEqual(s.shape, (true_rank,))
        np.testing.assert_allclose(np.sort(s)[::-1], true_eigs, rtol=1.0e-6, atol=1.0e-6)
        # U's columns span the same subspace H does act on: reconstruction matches H itself
        reconstructed = (U * s) @ U.T
        np.testing.assert_allclose(reconstructed, H, atol=1.0e-6)

    def test_never_assembles_the_dense_operator(self):
        calls = {"n": 0}
        n = 300

        rng = np.random.default_rng(1)
        basis = np.linalg.qr(rng.standard_normal((n, 3)))[0]
        eigs = np.array([9.0, 4.0, 1.0])
        H = (basis * eigs) @ basis.T

        def hvp(v):
            calls["n"] += 1
            return H @ v

        U, s = randomized_lowrank_hessian(hvp, n, k=3, rng=np.random.default_rng(2), oversample=5)
        # only ever touches H through matvecs (2 batches of k+oversample), never an (n, n) materialization
        self.assertGreater(calls["n"], 0)
        self.assertLessEqual(calls["n"], 2 * (3 + 5))
        self.assertTrue(np.all(s >= 0.0))
        self.assertEqual(U.shape, (n, 3))

    def test_k_larger_than_n_is_truncated_safely(self):
        rng = np.random.default_rng(4)
        n = 5
        m = rng.standard_normal((n, n))
        H = m @ m.T

        def hvp(v):
            return H @ v

        U, s = randomized_lowrank_hessian(hvp, n, k=20, rng=np.random.default_rng(5), oversample=10)
        self.assertEqual(U.shape[0], n)
        self.assertEqual(s.shape[0], 20)


def _subsurface_grid():
    xs = np.linspace(0.0, 100.0, 5)
    ys = np.linspace(0.0, 100.0, 5)
    zs = np.array([-30.0, -50.0])
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    return Field3D(coordinates=pts, spacing=20.0, units="kg/m^3", property_name="density_contrast", bounds=None)


def _blob(grid, amp=400.0):
    d2 = np.sum((grid.coordinates - np.array([50.0, 50.0, -40.0])) ** 2, axis=1)
    return amp * np.exp(-d2 / (2.0 * 35.0**2))


class _MultimodalSetup:
    """Shared gravity + borehole toy problem (mirrors the existing field_inversion_test fixture)."""

    def _build(self):
        rng = np.random.default_rng(0)
        grid = _subsurface_grid()
        truth = _blob(grid)
        volumes = np.full(grid.n, 20.0**3, dtype=float)
        registry = ForwardOperatorRegistry()
        registry.register(gravity_forward_operator(grid.coordinates, volumes))
        registry.register(borehole_forward_operator())

        gx, gy = np.meshgrid(np.linspace(0, 100, 5), np.linspace(0, 100, 5))
        grav_loc = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
        G = registry.get("gravity").jacobian(grid, grav_loc)
        gravity = Observation(
            kind="gravity",
            location=grav_loc,
            value=G @ truth + rng.normal(0, 2.0e-4, size=G.shape[0]),
            noise_cov=np.full(G.shape[0], (2.0e-4) ** 2),
        )
        idx = rng.choice(grid.n, size=int(0.5 * grid.n), replace=False)
        borehole = Observation(
            kind="borehole",
            location=grid.coordinates[idx],
            value=truth[idx] + rng.normal(0, 5.0, size=idx.shape),
            noise_cov=np.full(idx.shape, 25.0),
        )
        prior = FieldGaussianPrior(
            smoothness_precision=5.0e-3, marginal_precision=1.0e-3, length_scale=25.0, neighbors=6
        )
        return grid, registry, prior, [gravity, borehole]


class SparseLowRankUQAttachmentTestCase(unittest.TestCase, _MultimodalSetup):
    """`sparse_linear_gaussian_invert(..., low_rank_k=...)` -- the C2 attachment in field_inversion.py."""

    def setUp(self):
        self.grid, self.registry, self.prior, self.observations = self._build()

    def test_low_rank_attachment_returns_diag_var_mode_no_dense_precision_needed(self):
        exact = sparse_linear_gaussian_invert(self.grid, self.observations, self.registry, self.prior)
        low_rank = sparse_linear_gaussian_invert(
            self.grid,
            self.observations,
            self.registry,
            self.prior,
            low_rank_k=self.grid.n - 1,
            rng=np.random.default_rng(0),
        )
        self.assertIsNone(low_rank.cov)
        self.assertIsNone(low_rank.precision_factor)
        self.assertIsNotNone(low_rank.diag_var)
        np.testing.assert_allclose(low_rank.mean, exact.mean, rtol=1.0e-8, atol=1.0e-8)
        # a near-full-rank correction should recover marginal variances close to the exact posterior
        np.testing.assert_allclose(low_rank.marginal_variance, exact.marginal_variance, rtol=0.05, atol=1.0e-6)

    def test_low_rank_variance_never_exceeds_the_prior_variance(self):
        prior_precision = SparsePosteriorPrecision(sp.csc_matrix(self.prior.precision(self.grid)))
        prior_diag = prior_precision.marginal_variance()
        low_rank = sparse_linear_gaussian_invert(
            self.grid, self.observations, self.registry, self.prior, low_rank_k=5, rng=np.random.default_rng(1)
        )
        self.assertTrue(np.all(low_rank.marginal_variance <= prior_diag + 1.0e-9))


class GaussNewtonHessianHvpTestCase(unittest.TestCase, _MultimodalSetup):
    """`gauss_newton_hessian_hvp` -- matrix-free access to the GN Hessian `gauss_newton_invert` assembles densely."""

    def setUp(self):
        self.grid, self.registry, self.prior, self.observations = self._build()

    def _dense_reference_hessian(self, u, jitter=1.0e-10):
        Q = self.prior.precision(self.grid)
        phi = self.grid.from_unconstrained(u)
        B = _transform_jacobian_diag(self.grid, u)
        lam = Q.copy()
        for obs in self.observations:
            op = self.registry.get(obs.kind)
            J = op.local_jacobian(self.grid, phi, obs)
            A = J * B[None, :]
            rinv = _noise_precision(obs)
            lam = lam + A.T @ rinv @ A
        return lam + jitter * np.eye(self.grid.n)

    def test_hvp_matches_the_dense_gauss_newton_hessian(self):
        u = self.prior.mean_vector(self.grid)
        lam_dense = self._dense_reference_hessian(u)
        hvp = gauss_newton_hessian_hvp(self.grid, self.observations, self.registry, self.prior, u)

        rng = np.random.default_rng(9)
        probe = rng.standard_normal((5, self.grid.n))
        for v in probe:
            np.testing.assert_allclose(hvp(v), lam_dense @ v, rtol=1.0e-8, atol=1.0e-8)

    def test_hvp_feeds_randomized_lowrank_hessian_without_forming_the_dense_hessian(self):
        u = self.prior.mean_vector(self.grid)
        hvp = gauss_newton_hessian_hvp(self.grid, self.observations, self.registry, self.prior, u)
        n = self.grid.n
        U, s = randomized_lowrank_hessian(hvp, n, k=min(6, n - 1), rng=np.random.default_rng(3), oversample=8)
        self.assertEqual(U.shape[0], n)
        self.assertTrue(np.all(s >= 0.0))
        # cross-check the top eigenvalue's Rayleigh quotient against the dense reference
        lam_dense = self._dense_reference_hessian(u)
        top = U[:, 0]
        rayleigh = float(top @ (lam_dense @ top))
        self.assertAlmostEqual(rayleigh, float(s[0]), delta=max(1.0, 0.05 * abs(float(s[0]))))


if __name__ == "__main__":
    unittest.main()
