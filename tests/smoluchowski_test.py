"""Steady Smoluchowski diffusion-limited association rates: analytical-benchmark tests.

1) Free diffusion (W=0), perfectly absorbing sphere of radius a: the rate is the Debye-Smoluchowski value
   k = 4 pi D a. The radial solve matches to ~1 percent; the 3-D Cartesian-box solve to ~5-10 percent
   (staircased geometry).
2) Debye factor: with a centrally symmetric interaction W(r), the rate scales by
   f = [a * integral_a^R e^{W(r)}/r^2 dr]^{-1} relative to 4 pi D a. The numerical ratio matches this
   closed form to ~1 percent for both an attractive and a repulsive well.
3) The rate is differentiable in D (dk/dD ~ 4 pi a) and in the W parameters.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.smoluchowski import (
        smoluchowski_debye_factor,
        smoluchowski_rate_box,
        smoluchowski_rate_free,
        smoluchowski_rate_radial,
    )


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class SmoluchowskiRadialTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_free_diffusion_recovers_debye_smoluchowski(self):
        # W = 0: k_on -> 4 pi D a. Radial finite-volume solve, ~1% with a far bulk and fine spacing.
        D, a, R = 1.3, 0.8, 300 * 0.8
        k = float(smoluchowski_rate_radial(D, a, R, n=4000))
        k_ref = smoluchowski_rate_free(D, a)  # 4 pi D a
        self.assertAlmostEqual(k / k_ref, 1.0, delta=0.02)

    def test_debye_factor_attractive_well(self):
        # Attractive screened well W(r) = A e^{-(r-a)/lam}, A < 0: rate enhanced by the Debye factor.
        D, a, R, n = 1.0, 1.0, 300.0, 4000
        A, lam = -1.5, 2.0

        def W(r):
            return A * np.exp(-(np.asarray(r, float) - a) / lam)

        k = float(smoluchowski_rate_radial(D, a, R, n=n, W=W))
        f_num = k / smoluchowski_rate_free(D, a)
        f_ana = smoluchowski_debye_factor(a, R, W)
        self.assertGreater(f_ana, 1.0)  # attraction speeds association
        self.assertAlmostEqual(f_num / f_ana, 1.0, delta=0.02)

    def test_debye_factor_repulsive_well(self):
        D, a, R, n = 1.0, 1.0, 300.0, 4000
        A, lam = 1.2, 2.0

        def W(r):
            return A * np.exp(-(np.asarray(r, float) - a) / lam)

        k = float(smoluchowski_rate_radial(D, a, R, n=n, W=W))
        f_num = k / smoluchowski_rate_free(D, a)
        f_ana = smoluchowski_debye_factor(a, R, W)
        self.assertLess(f_ana, 1.0)  # repulsion slows association
        self.assertAlmostEqual(f_num / f_ana, 1.0, delta=0.02)

    def test_constant_well_debye_factor(self):
        # A piecewise-constant well over [a, r0] has a closed-form Debye factor; check against it directly.
        D, a, R, n = 1.0, 1.0, 300.0, 6000
        A, r0 = -2.0, 3.0

        def W(r):
            r = np.asarray(r, float)
            return np.where(r < r0, A, 0.0)

        k = float(smoluchowski_rate_radial(D, a, R, n=n, W=W))
        f_num = k / smoluchowski_rate_free(D, a)
        # integral_a^R e^{W}/r^2 dr = e^{A}(1/a - 1/r0) + (1/r0 - 1/R)
        integ = np.exp(A) * (1.0 / a - 1.0 / r0) + (1.0 / r0 - 1.0 / R)
        f_closed = 1.0 / (a * integ)
        self.assertAlmostEqual(f_num / f_closed, 1.0, delta=0.02)

    def test_differentiable_in_D(self):
        D = torch.tensor(1.0, requires_grad=True)
        a, R = 1.0, 300.0
        k = smoluchowski_rate_radial(D, a, R, n=2000)
        k.backward()
        # k = 4 pi D a f(W); with W=0, f=1, so dk/dD = 4 pi a (to the ~1% discretization).
        self.assertAlmostEqual(float(D.grad) / (4.0 * np.pi * a), 1.0, delta=0.02)

    def test_differentiable_in_well_depth(self):
        # Gradient of the rate w.r.t. the well-depth parameter is finite, nonzero, and the right sign
        # (a deeper attractive well A>0 in W = -A e^{...} raises the rate).
        a, R, n, lam = 1.0, 300.0, 1500, 2.0
        depth = torch.tensor(1.0, requires_grad=True)
        rr = torch.linspace(a, R, n)
        W_nodes = -depth * torch.exp(-(rr - a) / lam)  # attractive; deeper well = larger rate
        k = smoluchowski_rate_radial(1.0, a, R, n=n, W=W_nodes)
        k.backward()
        self.assertTrue(np.isfinite(float(depth.grad)))
        self.assertGreater(float(depth.grad), 0.0)

    def test_bad_radius_raises(self):
        with self.assertRaises(ValueError):
            smoluchowski_rate_radial(1.0, 2.0, 1.0)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class SmoluchowskiBoxTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_box_free_diffusion_matches_debye_smoluchowski(self):
        # Absorbing sphere staircased into a 3-D Cartesian box (reuses divergence_form with kappa=D e^{-W}).
        # Staircasing gives ~5-10% agreement with 4 pi D a.
        D, a = 1.0, 3.0
        nx = 41
        k = float(smoluchowski_rate_box(D, a, (nx, nx, nx), spacing=1.0))
        k_ref = smoluchowski_rate_free(D, a)
        self.assertAlmostEqual(k / k_ref, 1.0, delta=0.10)

    def test_box_differentiable_in_D(self):
        D = torch.tensor(1.0, requires_grad=True)
        a, nx = 3.0, 31
        k = smoluchowski_rate_box(D, a, (nx, nx, nx), spacing=1.0)
        k.backward()
        self.assertTrue(np.isfinite(float(D.grad)))
        self.assertGreater(float(D.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
