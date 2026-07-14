"""Tests for the blocky/compact/anisotropic priors (workstream A1): reweighted IRLS Gauss-Newton and
the structural (dip-rotated) gradient operator.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    torch.set_default_dtype(torch.float64)
    from mixle_pde.blocky_priors import (
        blocky_invert,
        dip_rotated_gradient_operator,
        minimum_support_weights,
        total_variation_weights,
    )
    from mixle_pde.field_priors import MinimumSupportPrior, TVFieldPrior
    from mixle_pde.geophysics import (
        gravity_point_sensitivity,
        regularized_gauss_newton,
        roughness_operator,
    )


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class WeightFunctionTestCase(unittest.TestCase):
    def test_minimum_support_weights_favor_small_cells(self):
        m = np.array([0.0, 1.0, 10.0])
        w = minimum_support_weights(m, eps=1e-3)
        self.assertTrue(np.all(np.diff(w) < 0.0))  # weight shrinks as the cell value grows
        self.assertAlmostEqual(w[0], 1.0 / 1e-6, places=3)

    def test_total_variation_weights_shrink_with_larger_jumps(self):
        R = roughness_operator((6, 1, 1))
        m = np.zeros(6)
        m[3:] = 5.0
        w = total_variation_weights(m, R, eps=1e-2)
        self.assertEqual(w.shape, (R.shape[0],))
        self.assertTrue(np.all(w > 0.0))
        # the one face straddling the jump gets a much smaller weight than a face inside a flat region
        rm = np.abs(R @ m)
        self.assertLess(w[np.argmax(rm)], w[np.argmin(rm)])


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DipRotatedGradientTestCase(unittest.TestCase):
    def test_zero_rotation_matches_axis_aligned_nullspace(self):
        # a flat model is still perfectly smooth under any rotation (rotating zero gradients is zero)
        shape = (6, 6, 4)
        R = dip_rotated_gradient_operator(shape, strike_deg=35.0, dip_deg=40.0)
        m = np.full(np.prod(shape), 3.0)
        self.assertLess(np.abs(R @ m).max(), 1e-8)

    def test_rotated_shape_and_nontrivial_response(self):
        shape = (8, 8, 5)
        R0 = dip_rotated_gradient_operator(shape, strike_deg=0.0, dip_deg=0.0)
        self.assertEqual(R0.shape, (3 * np.prod(shape), np.prod(shape)))
        rng = np.random.RandomState(0)
        m = rng.randn(np.prod(shape))
        R45 = dip_rotated_gradient_operator(shape, strike_deg=30.0, dip_deg=45.0)
        # a dipped basis responds differently to the same model than the axis-aligned one
        self.assertGreater(np.linalg.norm((R0 @ m) - np.zeros(R0.shape[0])), 0.0)
        self.assertFalse(np.allclose(R0 @ m, R45 @ m))

    def test_2d_profile_rotation(self):
        shape = (10, 6)
        R = dip_rotated_gradient_operator(shape, strike_deg=0.0, dip_deg=20.0)
        self.assertEqual(R.shape, (2 * np.prod(shape), np.prod(shape)))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ReweightGaussNewtonTestCase(unittest.TestCase):
    """`reweight=None` must reproduce today's plain-quadratic `regularized_gauss_newton` exactly."""

    def test_reweight_none_is_unchanged(self):
        nx, nz = 10, 10
        N = nx * nz
        rng = np.random.RandomState(3)
        A = torch.as_tensor(rng.randn(30, N))

        def fwd(u):
            return A @ u

        d = (A @ torch.as_tensor(rng.randn(N))).numpy() + 0.01 * rng.randn(30)
        R = roughness_operator((nx, nz))
        x1, s1 = regularized_gauss_newton(fwd, d, np.zeros(N), noise=0.05, beta=0.1, roughness=R, n_iter=5)
        x2, s2 = regularized_gauss_newton(
            fwd, d, np.zeros(N), noise=0.05, beta=0.1, roughness=R, n_iter=5, reweight=None
        )
        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(s1, s2)

    def test_reweight_is_applied_iteratively(self):
        # a reweight that zeroes out the roughness entirely should behave like beta=0 (pure damping-free
        # data fit, up to the identity `roughness=None` damping term) once the weights have decayed.
        nx, nz = 8, 8
        N = nx * nz
        rng = np.random.RandomState(1)
        A = torch.as_tensor(rng.randn(40, N))
        x_true = rng.randn(N)
        d = (A @ torch.as_tensor(x_true)).numpy()

        def fwd(u):
            return A @ u

        R = roughness_operator((nx, nz))

        def reweight(m):
            return np.full(R.shape[0], 1e-8)  # ~turn off the roughness penalty via IRLS weights

        x_rw, _ = regularized_gauss_newton(
            fwd, d, np.zeros(N), noise=1.0, beta=10.0, roughness=R, reweight=reweight, n_iter=8
        )
        x_plain, _ = regularized_gauss_newton(fwd, d, np.zeros(N), noise=1.0, beta=10.0, roughness=R, n_iter=8)
        # heavily reweighting down the roughness penalty should fit the (noiseless) data much better
        self.assertLess(np.linalg.norm(x_rw - x_true), np.linalg.norm(x_plain - x_true))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class FieldPriorDataclassTestCase(unittest.TestCase):
    def test_tv_field_prior_matches_function(self):
        R = roughness_operator((6, 6, 1))
        m = np.random.RandomState(0).randn(36)
        prior = TVFieldPrior(roughness=R, eps=0.02)
        np.testing.assert_allclose(prior.reweight(m), total_variation_weights(m, R, eps=0.02))

    def test_minimum_support_prior_matches_function(self):
        m = np.random.RandomState(0).randn(20)
        prior = MinimumSupportPrior(eps=0.02)
        np.testing.assert_allclose(prior.reweight(m), minimum_support_weights(m, eps=0.02))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class BlockyGravityRecoveryTestCase(unittest.TestCase):
    """The Definition-of-Done case: a compact 16-cell dense block, gravity forward, and the comparison
    between the plain-smoothness baseline and `blocky_invert(prior="tv")`.
    """

    def _problem(self, seed=0):
        nx, ny, h = 14, 14, 30.0
        depth = 150.0
        gx = (np.arange(nx) - nx // 2) * h
        gy = (np.arange(ny) - ny // 2) * h
        cx, cy = np.meshgrid(gx, gy, indexing="ij")
        cells = np.column_stack([cx.ravel(), cy.ravel(), np.full(cx.size, -depth)])
        n = len(cells)
        shape = (nx, ny, 1)

        ox = np.linspace(gx.min(), gx.max(), nx)
        ox_grid, oy_grid = np.meshgrid(ox, ox, indexing="ij")
        obs = np.column_stack([ox_grid.ravel(), oy_grid.ravel(), np.full(ox_grid.size, 5.0)])

        idx = np.arange(n).reshape(nx, ny)
        true_peak = 400.0
        truth = np.zeros(n)
        block_idx = idx[nx // 2 - 2 : nx // 2 + 2, ny // 2 - 2 : ny // 2 + 2].ravel()  # 4x4 = 16 cells
        truth[block_idx] = true_peak

        g = gravity_point_sensitivity(obs, cells, h * h * 40.0)
        d_true = g @ truth
        rng = np.random.RandomState(seed)
        noise_std = 0.05 * np.abs(d_true).std() + 1e-9
        d = d_true + noise_std * rng.randn(len(d_true))
        return dict(
            shape=shape, n=n, h=h, g=g, truth=truth, true_peak=true_peak, block_idx=block_idx, d=d, noise_std=noise_std
        )

    def test_blocky_beats_smoothness_baseline(self):
        p = self._problem(seed=0)
        gt = torch.as_tensor(p["g"])

        def fwd(u):
            return gt @ u

        R = roughness_operator(p["shape"], spacing=p["h"])
        x_smooth, _ = regularized_gauss_newton(
            fwd, p["d"], np.zeros(p["n"]), noise=p["noise_std"], beta=100.0, roughness=R, n_iter=15, jac_every=99
        )
        x_tv, _ = blocky_invert(
            fwd,
            p["d"],
            np.zeros(p["n"]),
            prior="tv",
            shape=p["shape"],
            beta=0.01,
            noise=p["noise_std"],
            n_iter=15,
            jac_every=99,
        )

        true_peak = p["true_peak"]
        block_idx = p["block_idx"]
        true_mass = p["truth"].sum()

        self.assertLess(x_smooth.max(), 0.4 * true_peak)
        self.assertGreaterEqual(x_tv.max(), 0.7 * true_peak)
        recovered_mass = x_tv[block_idx].sum()
        self.assertLess(abs(recovered_mass - true_mass), 0.2 * true_mass)


if __name__ == "__main__":
    unittest.main()
