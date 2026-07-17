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
    from mixle_pde.geophysics import (
        cross_gradient,
        dc_resistivity,
        depth_weighting,
        eikonal_traveltime,
        gravity_point_sensitivity,
        gravity_prism_sensitivity,
        invert_potential_field,
        joint_inversion,
        magnetic_dipole_sensitivity,
        regularized_gauss_newton,
        roughness_operator,
        straight_ray_operator,
        traveltime_tomography,
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
            lambda s: Lt @ s,
            y,
            np.full(N, 0.10),
            noise=0.005 * y.std(),
            beta=2.0,
            roughness=R,
            n_iter=4,
            jac_every=99,
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


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class PotentialFieldTestCase(unittest.TestCase):
    def test_depth_weighting_monotone(self):
        z = np.linspace(0.0, -1000.0, 11)  # surface down
        w = depth_weighting(z, 50.0, nu=2.0)
        self.assertAlmostEqual(w.max(), 1.0)
        self.assertTrue(np.all(np.diff(w) < 0))  # decreases with depth

    def test_gravity_sign_and_linearity(self):
        obs = np.array([[0.0, 0.0, 0.0]])
        cells = np.array([[0.0, 0.0, -100.0]])  # mass directly below
        G = gravity_point_sensitivity(obs, cells, 1.0e6)
        self.assertGreater(G[0, 0], 0.0)  # positive density below -> positive g_z
        self.assertAlmostEqual(float(gravity_point_sensitivity(obs, cells, 2.0e6)[0, 0]), 2 * float(G[0, 0]))

    def test_gravity_inversion_recovers_blob(self):
        # forward a compact density blob through the point-mass operator, invert with depth weighting, recover it
        nx, ny, nz, h = 11, 11, 7, 200.0
        gx = (np.arange(nx) - nx // 2) * h
        gz = -np.arange(1, nz + 1) * h
        CX, CY, CZ = np.meshgrid(gx, gx, gz, indexing="ij")
        cells = np.column_stack([CX.ravel(), CY.ravel(), CZ.ravel()])
        N = len(cells)
        ox = (np.arange(nx) - nx // 2) * h
        OX, OY = np.meshgrid(ox, ox, indexing="ij")
        obs = np.column_stack([OX.ravel(), OY.ravel(), np.full(OX.size, 50.0)])
        truth = np.zeros(N)
        truth[np.linalg.norm(cells - np.array([0.0, 0.0, -700.0]), axis=1) < 350.0] = -300.0
        G = gravity_point_sensitivity(obs, cells, h**3)
        rng = np.random.RandomState(0)
        d = G @ truth
        d = d + 0.02 * np.abs(d).std() * rng.randn(len(d))
        w = depth_weighting(cells[:, 2], 50.0, nu=2.0)
        Gw = torch.as_tensor(G / w[None, :])
        R = roughness_operator((nx, ny, nz))
        s, _ = regularized_gauss_newton(
            lambda u: Gw @ u,
            d,
            np.zeros(N),
            noise=0.02 * np.abs(d).std() + 1e-3,
            beta=1e-2,
            roughness=R,
            n_iter=3,
            jac_every=99,
        )
        rho = s / w
        # recovered low-density region overlaps the truth (location resolved)
        self.assertLess(cells[np.argmin(rho), 2], -200.0)  # minimum is below the surface
        self.assertGreater(np.corrcoef(rho, truth)[0, 1], 0.3)  # positive structural correlation

    def test_magnetic_finite(self):
        obs = np.array([[0.0, 0.0, 0.0]])
        cells = np.array([[0.0, 0.0, -100.0]])
        G = magnetic_dipole_sensitivity(obs, cells, 1.0e6, inclination=-60.0, declination=3.0)
        self.assertTrue(np.isfinite(G).all())


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class EikonalTestCase(unittest.TestCase):
    def test_homogeneous_traveltime(self):
        # in a uniform-slowness medium the eikonal traveltime equals slowness * straight-line distance
        nx, nz, h, s = 31, 31, 1.0, 0.2
        T = eikonal_traveltime(np.full(nx * nz, s), (nx, nz), 0, spacing=h, n_cycles=4).reshape(nx, nz)
        corner = T[nx - 1, nz - 1]
        exact = s * np.hypot((nx - 1) * h, (nz - 1) * h)
        self.assertLess(abs(corner - exact) / exact, 0.05)  # FSM is accurate to a few %

    def test_crosshole_tomography_recovers_layer(self):
        # plant a fast horizontal layer between two boreholes, forward, invert -> recover the layer
        nx, nz, h = 11, 21, 1.0
        N = nx * nz
        s_true = np.full((nx, nz), 0.2)
        s_true[:, 8:12] = 0.14  # fast (low-slowness) layer
        s_true = s_true.ravel()
        src = np.array([0 * nz + j for j in range(1, nz - 1, 2)])
        rcv = np.array([(nx - 1) * nz + j for j in range(1, nz - 1, 2)])
        si, ri = np.meshgrid(src, rcv, indexing="ij")
        si, ri = si.ravel(), ri.ravel()
        times = np.array(
            [eikonal_traveltime(s_true, (nx, nz), int(s), spacing=h, n_cycles=4)[int(r)] for s, r in zip(si, ri)]
        )
        rng = np.random.RandomState(0)
        times = times + 0.01 * times.std() * rng.randn(len(times))
        s_inv, vel, _ = traveltime_tomography(
            times,
            si,
            ri,
            (nx, nz),
            spacing=h,
            slowness0=0.2,
            noise=0.01 * times.std() + 1e-6,
            beta=5.0,
            n_iter=6,
            bounds=(0.08, 0.3),
        )
        # the recovered slowness is lower (faster) in the planted layer than outside it
        s3 = s_inv.reshape(nx, nz)
        self.assertLess(s3[:, 8:12].mean(), s3[:, 2:6].mean())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class JointInversionBoundsTestCase(unittest.TestCase):
    """joint_inversion accepts per-model bounds, so models on different scales each stay in their own range."""

    def test_per_model_bounds(self):
        nx, nz = 8, 6
        N = nx * nz
        shape = (nx, nz)
        A = torch.eye(N)

        def f1(x):
            return A @ x

        def f2(x):
            return A @ x

        d1 = np.full(N, 5.0)  # wants ~5 but is bounded to [0, 1]
        d2 = np.full(N, 1.0)  # wants ~1 but is bounded to [1e-4, 3e-4]
        m1, m2 = joint_inversion(
            [f1, f2],
            [d1, d2],
            [np.zeros(N), np.full(N, 2e-4)],
            shape,
            bounds=[(0.0, 1.0), (1e-4, 3e-4)],
            n_iter=6,
        )
        self.assertLessEqual(m1.max(), 1.0 + 1e-9)
        self.assertGreaterEqual(m1.min(), -1e-9)
        self.assertLessEqual(m2.max(), 3e-4 + 1e-12)
        self.assertGreaterEqual(m2.min(), 1e-4 - 1e-12)
        # a single (lo, hi) still applies to every model (back-compatible)
        n1, n2 = joint_inversion([f1, f2], [d1, d2], [np.zeros(N), np.zeros(N)], shape, bounds=(0.0, 1.0), n_iter=4)
        self.assertLessEqual(max(n1.max(), n2.max()), 1.0 + 1e-9)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DepthWeightingTestCase(unittest.TestCase):
    """Regression test for a real, verified bug: `depth_weighting` existed but nothing wired it into
    an inversion, so recovered density/susceptibility systematically piled up near the surface
    regardless of true source depth. `invert_potential_field` fixes this at the source."""

    def test_weight_decays_monotonically_from_the_surface(self):
        # locks in the corrected docstring's claim: largest near z0, decaying with |z - z0|.
        cell_z = np.array([0.0, -5.0, -15.0, -30.0, -55.0])
        w = depth_weighting(cell_z, z0=2.0, nu=2.0)
        self.assertTrue(np.all(np.diff(w) < 0), "weight must strictly decrease with depth")
        self.assertAlmostEqual(w[0], 1.0)  # normalized to 1 at its max (nearest the surface)

    def _vms_lens_scenario(self):
        """The exact synthetic VMS-lens geometry that exposed the depth bias in practice: a
        20x3x12 grid, a 700 kg/m^3 density contrast lens at 25-45m depth (centre 35m)."""
        h = 5.0
        nx, ny, nz = 20, 3, 12
        body_x, body_z = (8, 12), (5, 9)
        xs = np.arange(nx) * h
        zs = -np.arange(nz) * h
        mins, maxs = [], []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    mins.append([xs[i], (j - ny / 2) * h, zs[k] - h])
                    maxs.append([xs[i] + h, (j - ny / 2) * h + h, zs[k]])
        cell_mins, cell_maxs = np.array(mins), np.array(maxs)
        true_mask = np.zeros((nx, ny, nz), dtype=bool)
        true_mask[body_x[0] : body_x[1], :, body_z[0] : body_z[1]] = True
        true_mask_flat = true_mask.reshape(-1)
        rho_true = np.where(true_mask_flat, 700.0, 0.0)
        obs = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 2.0)])
        g = gravity_prism_sensitivity(obs, cell_mins, cell_maxs)
        true_depth_center = (body_z[0] + body_z[1]) / 2.0 * h  # 35.0 m
        return cell_mins, cell_maxs, g, rho_true, true_mask_flat, true_depth_center, nx, ny, nz, h

    def _recovered_depth_centroid(self, est, true_mask_flat, nx, ny, nz, h):
        w = np.clip(est, 0.0, None).reshape(nx, ny, nz).sum(axis=1)  # sum over the thin y axis
        zs = -np.arange(nz) * h
        w_sum = w.sum()
        self.assertGreater(w_sum, 0.0, "recovered model has no positive mass to form a centroid")
        return -float(np.sum(w.sum(axis=0) * zs) / w_sum)  # positive-down depth

    def test_invert_potential_field_recovers_depth_far_better_than_unweighted(self):
        cell_mins, cell_maxs, g, rho_true, true_mask_flat, true_depth, nx, ny, nz, h = self._vms_lens_scenario()
        n_cells = nx * ny * nz
        clean = g @ rho_true
        rng = np.random.RandomState(0)
        noise_std = 0.02 * np.std(clean) + 1e-4
        y = clean + noise_std * rng.randn(*clean.shape)

        # unweighted (the bug this test locks in a fix for): plain roughness, no depth compensation.
        roughness_plain = roughness_operator((nx, ny, nz), spacing=h)
        est_unweighted, _ = regularized_gauss_newton(
            lambda m: torch.as_tensor(g) @ m,
            y,
            np.zeros(n_cells),
            noise=noise_std,
            beta=0.05,
            roughness=roughness_plain,
            lower=0.0,
            upper=2000.0,
            n_iter=8,
            jac_every=3,
        )
        depth_unweighted = self._recovered_depth_centroid(est_unweighted, true_mask_flat, nx, ny, nz, h)

        # depth-weighted (the fix): same data, same beta, only the roughness operator differs.
        est_weighted, _ = invert_potential_field(
            g,
            y,
            cell_mins,
            cell_maxs,
            noise=noise_std,
            beta=0.05,
            z0=2.0,
            nu=2.0,
            lower=0.0,
            upper=2000.0,
            n_iter=8,
            jac_every=3,
        )
        depth_weighted_result = self._recovered_depth_centroid(est_weighted, true_mask_flat, nx, ny, nz, h)

        error_unweighted = abs(depth_unweighted - true_depth)
        error_weighted = abs(depth_weighted_result - true_depth)
        # the unweighted baseline must itself show the real, previously-observed bias (recovered
        # shallower than truth by a wide margin) -- otherwise this isn't testing what it claims to.
        self.assertGreater(
            error_unweighted,
            4.0,
            f"expected the unweighted baseline to show the known shallow bias; got depth={depth_unweighted:.1f}m vs true={true_depth:.1f}m",
        )
        # the fix must recover depth much more accurately, not just nominally better.
        self.assertLess(
            error_weighted,
            0.5 * error_unweighted,
            f"depth-weighted error ({error_weighted:.1f}m) should be well under half the unweighted error ({error_unweighted:.1f}m)",
        )
        self.assertLess(
            error_weighted,
            3.0,
            f"depth-weighted recovery should land within 3m of the true {true_depth}m depth; got {depth_weighted_result:.1f}m",
        )
