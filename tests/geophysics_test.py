"""Tests for the near-surface geophysics module: forward operators + the regularized inversion engine."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    torch.set_default_dtype(torch.float64)
    from pysparkplug_pde.geophysics import (
        cross_gradient,
        dc_resistivity,
        regularized_gauss_newton,
        roughness_operator,
        straight_ray_operator,
    )


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class RayOperatorTestCase(unittest.TestCase):
    def test_ray_length_totals(self):
        # a single straight ray's accumulated length ~ the source-receiver distance
        shape = (20, 20)
        L = straight_ray_operator(shape, [[0.0, 0.0]], [[19.0, 0.0]], spacing=1.0, n_seg=200)
        self.assertEqual(L.shape, (1, 400))
        self.assertAlmostEqual(L.sum(), 19.0, delta=0.5)

    def test_all_pairs_shape(self):
        L = straight_ray_operator((10, 10), np.zeros((3, 2)), np.ones((4, 2)), spacing=1.0)
        self.assertEqual(L.shape, (12, 100))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class RoughnessTestCase(unittest.TestCase):
    def test_constant_in_nullspace(self):
        R = roughness_operator((8, 8, 6))
        m = np.ones(8 * 8 * 6)
        self.assertLess(np.abs(R @ m).max(), 1e-9)  # a flat model is perfectly smooth


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class CrossGradientTestCase(unittest.TestCase):
    def test_parallel_gradients_vanish(self):
        shape = (16, 16)
        x, y = np.meshgrid(np.arange(16.0), np.arange(16.0), indexing="ij")
        a = torch.as_tensor((x + 0.5 * y).ravel())  # same gradient direction
        b = torch.as_tensor((2 * x + 1.0 * y).ravel())
        t = cross_gradient(a, b, shape)
        self.assertLess(float(t.abs().max()), 1e-9)

    def test_orthogonal_gradients_do_not_vanish(self):
        shape = (16, 16)
        x, y = np.meshgrid(np.arange(16.0), np.arange(16.0), indexing="ij")
        a = torch.as_tensor(x.ravel())
        b = torch.as_tensor(y.ravel())
        t = cross_gradient(a, b, shape)
        self.assertGreater(float(t.abs().max()), 0.5)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class GaussNewtonTomographyTestCase(unittest.TestCase):
    """The regularized inverter recovers a well-posed crosshole traveltime tomography target."""

    def test_recovers_slowness_anomaly(self):
        nx, nz, h = 18, 18, 1.0
        N = nx * nz
        X, Z = np.meshgrid(np.arange(nx) * h, np.arange(nz) * h, indexing="ij")
        s_true = np.full(N, 0.10)
        blob = ((X.ravel() - nx * h / 2) ** 2 + (Z.ravel() - nz * h / 2) ** 2) < 16
        s_true[blob] = 0.16
        # crosshole rays: left column -> right column
        coords = np.stack([X.ravel(), Z.ravel()], 1)
        src = coords[X.ravel() == 0]
        rcv = coords[X.ravel() == (nx - 1) * h]
        L = straight_ray_operator((nx, nz), src, rcv, spacing=h)
        Lt = torch.as_tensor(L.toarray())
        rng = np.random.RandomState(0)
        y = L @ s_true
        y = y + 0.005 * y.std() * rng.randn(*y.shape)
        R = roughness_operator((nx, nz), spacing=h)
        s_est, std = regularized_gauss_newton(
            lambda s: Lt @ s, y, np.full(N, 0.10), noise=0.005 * y.std(),
            beta=2.0, roughness=R, n_iter=4, jac_every=99,
        )
        self.assertGreater(np.corrcoef(s_est, s_true)[0, 1], 0.6)
        self.assertGreater(s_est[blob].mean(), s_est[~blob].mean())  # anomaly is faster-slowness
        self.assertEqual(std.shape, (N,))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DCResistivityTestCase(unittest.TestCase):
    """The DC forward runs, is finite, and obeys reciprocity (swap current/potential dipoles)."""

    def test_forward_and_reciprocity(self):
        nx, ny, nz, h = 12, 8, 12, 2.0
        idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
        a, b = int(idx[2, ny // 2, 3]), int(idx[2, ny // 2, 8])
        m, n = int(idx[nx - 3, ny // 2, 3]), int(idx[nx - 3, ny // 2, 8])
        log_sigma = torch.zeros(nx * ny * nz)
        fwd = dc_resistivity(log_sigma, (nx, ny, nz), [(a, b, m, n)], spacing=h, sigma_ref=0.02, log_data=False)
        recip = dc_resistivity(log_sigma, (nx, ny, nz), [(m, n, a, b)], spacing=h, sigma_ref=0.02, log_data=False)
        self.assertTrue(np.isfinite(float(fwd[0])))
        # reciprocity: R(AB,MN) == R(MN,AB)
        self.assertLess(abs(float(fwd[0]) - float(recip[0])), 1e-6 + 0.02 * abs(float(fwd[0])))

    def test_localizes_conductor(self):
        # ERT is severely ill-posed: we only require it to LOCATE a conductor (positive correlation),
        # not to recover its amplitude (which the smoothness regularization biases low, as in practice).
        nx, ny, nz, h = 14, 8, 14, 3.0
        N = nx * ny * nz
        X, _, Z = np.meshgrid(np.arange(nx) * h, np.arange(ny) * h, np.arange(nz) * h, indexing="ij")
        idx = np.arange(N).reshape(nx, ny, nz)
        m_true = np.zeros(N)
        blk = (np.abs(X.ravel() - nx * h / 2) < 9) & (np.abs(Z.ravel() - nz * h * 0.45) < 6)
        m_true[blk] = np.log(0.1 / 0.02)
        elec = np.array(
            [int(idx[bx, ny // 2, k]) for bx in (2, nx - 3) for k in range(1, nz - 1)]
            + [int(idx[i, ny // 2, 1]) for i in range(2, nx - 1, 2)]
        )
        rng = np.random.RandomState(1)
        quad = []
        for _ in range(160):
            aa, bb = elec[rng.choice(len(elec), 2, replace=False)]
            mm, nn = elec[rng.choice(len(elec), 2, replace=False)]
            quad.append((int(aa), int(bb), int(mm), int(nn)))

        def fwd(mm):
            return dc_resistivity(mm, (nx, ny, nz), quad, spacing=h, sigma_ref=0.02)

        d_true = fwd(torch.as_tensor(m_true)).detach().numpy()
        nlev = 0.02 * d_true.std() + 1e-9
        y = d_true + nlev * rng.randn(*d_true.shape)
        R = roughness_operator((nx, ny, nz), spacing=h)
        m_est, _ = regularized_gauss_newton(
            fwd, y, np.zeros(N), noise=nlev, beta=0.03, roughness=R, lower=-5, upper=5, n_iter=12, jac_every=3
        )
        # the conductor is located (positive correlation) and recovered as more conductive than background
        self.assertGreater(np.corrcoef(m_est, m_true)[0, 1], 0.15)
        self.assertGreater(m_est[blk].mean(), m_est[~blk].mean())


if __name__ == "__main__":
    unittest.main()
