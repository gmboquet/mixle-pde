"""Recovery tests for the sonar/radar propagation inverse builders (mixle_pde.propagation_inverse).

Self-consistency recovery, never coverage-of-target: synthesize the received field with the PE forward
from a KNOWN parameter, add small noise, invert with joint([...]).fit, and assert the recovered parameter
is within tolerance of the truth.

  1. Radar refractivity-from-clutter: recover a known surface-duct height from the propagated field.
  2. Ocean tomography: recover a known scalar sound-speed anomaly amplitude from the received field.

The received complex field is highly oscillatory in the parameter (cycle-skipping), so both tests observe
the field MAGNITUDE at a modest vertical receiver array at the final range, whose misfit is convex around
the truth; that keeps a single-scalar Gauss-Newton fit fast and robust on a modest PE grid.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle.ppl import free, joint

    from mixle_pde.ops import make_ops
    from mixle_pde.parabolic_equation import ParabolicEquation2D, modified_refractivity_index
    from mixle_pde.propagation_inverse import (
        _soft_surface_duct,
        ocean_sound_speed_inversion,
        refractivity_from_clutter,
    )


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class RefractivityFromClutterTest(unittest.TestCase):
    """Recover a radar surface-duct height from the propagated (clutter) field magnitude."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        # UHF radar (300 MHz, 1 m wavelength) resolved at dz = 1 m, so the ducted field survives to range.
        self.c0 = 3.0e8
        self.freq = 3.0e8
        self.k0 = 2.0 * np.pi * self.freq / self.c0
        self.nz = 256
        self.dz = 1.0
        self.dr = 25.0
        self.n_range = 160  # 4 km
        self.pe = ParabolicEquation2D(
            self.nz, dz=self.dz, dr=self.dr, k0=self.k0, c0=self.c0, surface="free", absorb=48, absorb_strength=2.0
        )
        self.source_depth = 15.0
        self.width = 8.0 / self.k0
        self.psi0 = self.pe.starter(self.source_depth, width=self.width)
        self.z = self.pe.depths()
        # a vertical receiver array spanning the ducted region, sampled at the final range
        self.rx = torch.arange(4, 200, 6)

    def _observe(self, f, o):
        return o.abs(f)[-1][self.rx]

    def _synthesize_abs(self, h_d):
        ops = make_ops()
        m = _soft_surface_duct(self.z, torch.as_tensor(float(h_d)), 350.0, 0.118, 1.0, ops)
        n = modified_refractivity_index(m)
        field = self.pe.march(self.psi0, n, self.n_range)
        return self._observe(field, ops).detach().numpy()

    def test_misfit_is_convex_near_truth(self):
        # Sanity: the |field| receiver-array misfit descends monotonically toward the true duct height.
        h_true = 120.0
        ref = self._synthesize_abs(h_true)
        prev = None
        for h in (80.0, 100.0, 120.0):  # ascending toward truth: misfit must decrease
            mis = float(((self._synthesize_abs(h) - ref) ** 2).sum())
            if prev is not None:
                self.assertLess(mis, prev)
            prev = mis
        self.assertAlmostEqual(prev, 0.0, places=8)  # exact at the truth

    def test_recovers_known_duct_height(self):
        h_true = 120.0
        y = self._synthesize_abs(h_true)
        rng = np.random.RandomState(0)
        scale = 0.01 * y.max()
        y_noisy = y + scale * rng.randn(*y.shape)

        # Coarse grid search to seed the fit in the convex basin. The nonconvex climb from h_d = 0 lands in a
        # platform-dependent basin, so we first localize the duct with a deterministic global scan (standard
        # practice for 1-D refractivity-from-clutter), then Gauss-Newton refines from that seed. The grid
        # excludes 120 exactly so the refine does real work.
        grid = np.arange(70.0, 176.0, 15.0)
        mis = [float(((self._synthesize_abs(h) - y_noisy) ** 2).sum()) for h in grid]
        h0 = float(grid[int(np.argmin(mis))])
        self.assertLess(abs(h0 - h_true), 15.0)  # the grid alone localizes the duct to within a cell

        h_d = free(1, name="h_d", support="real")  # now the OFFSET from h0; GN starts in the basin near truth
        obs = refractivity_from_clutter(
            y_noisy,
            h_d,
            pe=self.pe,
            source_depth=self.source_depth,
            m0=350.0,
            base_gradient=0.118,
            strength=1.0,
            h0=h0,
            starter_width=self.width,
            n_range=self.n_range,
            scale=scale,
            observe=lambda f, p, o: self._observe(f, o),
        )
        post = joint([obs]).fit(how="gauss_newton")
        hm, hs = post.posterior("h_d")
        self.assertLess(abs((h0 + float(hm)) - h_true), 5.0)  # h0 + offset within 5 m of the true 120 m duct
        self.assertGreater(hs, 0.0)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class OceanSoundSpeedInversionTest(unittest.TestCase):
    """Recover a known scalar sound-speed anomaly amplitude from the received acoustic field magnitude."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.c0 = 1500.0
        self.freq = 150.0
        self.nz = 128
        self.dz = 0.9
        self.dr = 2.0
        self.n_range = 100
        self.pe = ParabolicEquation2D(
            self.nz,
            dz=self.dz,
            dr=self.dr,
            freq=self.freq,
            c0=self.c0,
            surface="pressure_release",
            absorb=32,
            absorb_strength=3.0,
        )
        z = self.pe.depths().detach().numpy()
        self.c_bg = 1500.0 + 0.017 * z  # mild downward-refracting background profile
        self.anomaly_depth = 55.0
        self.anomaly_width = 25.0
        self.shape = np.exp(-(((z - self.anomaly_depth) / self.anomaly_width) ** 2))
        self.source_depth = 30.0
        self.width = 3.0 / self.pe.k0
        self.psi0 = self.pe.starter(self.source_depth, width=self.width)
        self.rx = torch.arange(4, 100, 4)  # vertical receiver array at the final range

    def _observe(self, f, o):
        return o.abs(f)[-1][self.rx]

    def _synthesize_abs(self, dc):
        ops = make_ops()
        c = torch.as_tensor(self.c_bg + float(dc) * self.shape)
        n = self.c0 / c
        field = self.pe.march(self.psi0, n, self.n_range)
        return self._observe(field, ops).detach().numpy()

    def test_misfit_is_convex_near_truth(self):
        dc_true = 8.0
        ref = self._synthesize_abs(dc_true)
        prev = None
        for dc in (2.0, 5.0, 8.0):  # ascending toward truth: misfit must decrease
            mis = float(((self._synthesize_abs(dc) - ref) ** 2).sum())
            if prev is not None:
                self.assertLess(mis, prev)
            prev = mis
        self.assertAlmostEqual(prev, 0.0, places=8)

    def test_recovers_known_sound_speed_anomaly(self):
        dc_true = 8.0
        y = self._synthesize_abs(dc_true)
        rng = np.random.RandomState(1)
        scale = 0.01 * y.max()
        y_noisy = y + scale * rng.randn(*y.shape)

        dc = free(1, name="dc", support="real")
        obs = ocean_sound_speed_inversion(
            y_noisy,
            dc,
            pe=self.pe,
            source_depth=self.source_depth,
            c_profile=self.c_bg,
            anomaly_depth=self.anomaly_depth,
            anomaly_width=self.anomaly_width,
            starter_width=self.width,
            n_range=self.n_range,
            scale=scale,
            observe=lambda f, p, o: self._observe(f, o),
        )
        post = joint([obs]).fit(how="gauss_newton")
        dm, ds = post.posterior("dc")
        self.assertLess(abs(dm - dc_true), 1.0)  # within 1 m/s of the true 8 m/s anomaly amplitude
        self.assertGreater(ds, 0.0)


if __name__ == "__main__":
    unittest.main()
