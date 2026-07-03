"""Ocean sound-speed transforms (mixle_pde.sound_speed): published check values + sensitivities + gradients."""

import unittest

import numpy as np

from mixle_pde.sound_speed import depth_to_pressure, mackenzie, pressure_to_depth, unesco

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MackenzieTest(unittest.TestCase):
    def test_published_check_value(self):
        # Mackenzie (1981) published check: c(25 C, 35 ppt, 1000 m) = 1550.744 m/s.
        self.assertAlmostEqual(mackenzie(25.0, 35.0, 1000.0), 1550.744, places=2)

    def test_surface_reference_points(self):
        # At S=35, D=0 the salinity/depth terms vanish and c reduces to the pure temperature polynomial,
        # so c(0,35,0) is exactly the constant 1448.96 m/s.
        self.assertAlmostEqual(mackenzie(0.0, 35.0, 0.0), 1448.96, places=6)
        # c(30,35,0): warm surface water, hand-evaluated from the temperature polynomial.
        c30 = 1448.96 + 4.591 * 30 - 5.304e-2 * 30**2 + 2.374e-4 * 30**3
        self.assertAlmostEqual(mackenzie(30.0, 35.0, 0.0), c30, places=6)
        self.assertAlmostEqual(mackenzie(30.0, 35.0, 0.0), 1545.3638, places=3)

    def test_sensitivity_signs(self):
        # Warmer, saltier, deeper water is all faster: dc/dT, dc/dS, dc/dD > 0 near the check point.
        h = 1e-4
        dc_dT = (mackenzie(25 + h, 35, 1000) - mackenzie(25 - h, 35, 1000)) / (2 * h)
        dc_dS = (mackenzie(25, 35 + h, 1000) - mackenzie(25, 35 - h, 1000)) / (2 * h)
        dc_dD = (mackenzie(25, 35, 1000 + h) - mackenzie(25, 35, 1000 - h)) / (2 * h)
        self.assertGreater(dc_dT, 0.0)
        self.assertGreater(dc_dS, 0.0)
        self.assertGreater(dc_dD, 0.0)
        # dc/dT is a couple m/s per deg C near 25 C (roughly +2.4 at 25, rising to ~+3 by 20 C).
        self.assertTrue(2.0 < dc_dT < 4.0, dc_dT)
        # dc/dS is ~+1.1 m/s per ppt, dc/dD is ~+0.017 m/s per m.
        self.assertAlmostEqual(dc_dS, 1.0837, places=2)
        self.assertAlmostEqual(dc_dD, 0.0166, places=3)

    def test_dc_dT_rises_toward_colder_water(self):
        # The temperature sensitivity grows as water cools: dc/dT at 15 C exceeds that at 25 C.
        h = 1e-4
        d15 = (mackenzie(15 + h, 35, 0) - mackenzie(15 - h, 35, 0)) / (2 * h)
        d25 = (mackenzie(25 + h, 35, 0) - mackenzie(25 - h, 35, 0)) / (2 * h)
        self.assertGreater(d15, d25)
        self.assertTrue(3.0 < d15 < 3.5, d15)  # ~+3.16 m/s/C at 15 C

    def test_vectorized_over_a_field(self):
        # Elementwise: a T/S/D profile runs in one call and matches scalar calls.
        T = np.linspace(2.0, 30.0, 6)
        S = np.linspace(30.0, 38.0, 6)
        D = np.linspace(0.0, 5000.0, 6)
        c = mackenzie(T, S, D)
        self.assertEqual(c.shape, (6,))
        for i in range(6):
            self.assertAlmostEqual(c[i], mackenzie(T[i], S[i], D[i]), places=10)


class UnescoTest(unittest.TestCase):
    def test_published_surface_check_value(self):
        # UNESCO / Chen-Millero standard check: c(0 C, 35 ppt, 0 bar) = 1449.14 m/s (Fofonoff & Millard).
        self.assertAlmostEqual(unesco(0.0, 35.0, 0.0), 1449.14, places=2)

    def test_high_pressure_regression_value(self):
        # Internally consistent regression at the top of the valid pressure range (P = 1000 bar), computed
        # from the standard Chen-Millero coefficients. Guards the pressure blocks against coefficient drift.
        self.assertAlmostEqual(unesco(25.0, 35.0, 1000.0), 1699.242, places=2)

    def test_pure_water_limit(self):
        # At S = 0 the salinity blocks vanish and unesco reduces to the pure-water speed Cw(T, P). At the
        # surface Cw(0,0) is the leading coefficient 1402.388 m/s.
        self.assertAlmostEqual(unesco(0.0, 0.0, 0.0), 1402.388, places=3)

    def test_sensitivity_signs(self):
        # Same monotonicity as Mackenzie: warmer, saltier, higher-pressure water is faster.
        h = 1e-4
        dc_dT = (unesco(10 + h, 35, 100) - unesco(10 - h, 35, 100)) / (2 * h)
        dc_dS = (unesco(10, 35 + h, 100) - unesco(10, 35 - h, 100)) / (2 * h)
        dc_dP = (unesco(10, 35, 100 + h) - unesco(10, 35, 100 - h)) / (2 * h)
        self.assertGreater(dc_dT, 0.0)
        self.assertGreater(dc_dS, 0.0)
        self.assertGreater(dc_dP, 0.0)

    def test_agrees_with_mackenzie_within_a_few_ms(self):
        # Two independent equations of state must agree to a few tenths of m/s in their common valid range.
        # Convert Mackenzie's depth to UNESCO's pressure so both describe the same water column.
        for T, S, D in [(10.0, 35.0, 500.0), (20.0, 36.0, 2000.0), (4.0, 34.0, 3000.0)]:
            P = float(depth_to_pressure(D))
            self.assertAlmostEqual(mackenzie(T, S, D), unesco(T, S, P), delta=1.5)


class DepthPressureTest(unittest.TestCase):
    def test_surface_is_zero(self):
        self.assertAlmostEqual(float(depth_to_pressure(0.0)), 0.0, places=10)

    def test_thousand_metre_check_value(self):
        # Leroy & Parthiot (1998): ~1.01 MPa (101 bar) per 1000 m at mid-latitude, close to hydrostatic.
        p = float(depth_to_pressure(1000.0, latitude=45.0))
        self.assertAlmostEqual(p, 101.06, places=1)
        self.assertTrue(100.0 < p < 102.0, p)

    def test_roundtrip(self):
        # depth -> pressure -> depth recovers the depth to well under a metre across the column.
        for z in [100.0, 1000.0, 4000.0, 8000.0]:
            p = depth_to_pressure(z, latitude=45.0)
            z_rec = float(pressure_to_depth(p, latitude=45.0))
            self.assertAlmostEqual(z_rec, z, delta=1.0)

    def test_monotone_increasing(self):
        z = np.linspace(0.0, 6000.0, 20)
        p = depth_to_pressure(z, latitude=45.0)
        self.assertTrue(np.all(np.diff(p) > 0.0))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class AutogradTest(unittest.TestCase):
    def test_mackenzie_gradients_match_finite_difference(self):
        # Autograd dc/dT, dc/dS, dc/dD are finite and match central differences.
        for name, base in (("T", 25.0), ("S", 35.0), ("D", 1000.0)):
            args = {"T": 25.0, "S": 35.0, "D": 1000.0}

            def c_of(x, name=name, args=args):
                a = dict(args)
                a[name] = x
                return mackenzie(a["T"], a["S"], a["D"])

            x = torch.tensor(base, dtype=torch.float64, requires_grad=True)
            c = c_of(x)
            c.backward()
            g_auto = float(x.grad)
            self.assertTrue(np.isfinite(g_auto))
            eps = 1e-4 * max(1.0, abs(base))
            g_fd = (
                float(c_of(torch.tensor(base + eps, dtype=torch.float64)))
                - float(c_of(torch.tensor(base - eps, dtype=torch.float64)))
            ) / (2 * eps)
            self.assertAlmostEqual(g_auto, g_fd, places=4)

    def test_unesco_gradients_match_finite_difference(self):
        for name, base in (("T", 10.0), ("S", 35.0), ("P", 300.0)):
            args = {"T": 10.0, "S": 35.0, "P": 300.0}

            def c_of(x, name=name, args=args):
                a = dict(args)
                a[name] = x
                return unesco(a["T"], a["S"], a["P"])

            x = torch.tensor(base, dtype=torch.float64, requires_grad=True)
            c = c_of(x)
            c.backward()
            g_auto = float(x.grad)
            self.assertTrue(np.isfinite(g_auto))
            eps = 1e-4 * max(1.0, abs(base))
            g_fd = (
                float(c_of(torch.tensor(base + eps, dtype=torch.float64)))
                - float(c_of(torch.tensor(base - eps, dtype=torch.float64)))
            ) / (2 * eps)
            self.assertAlmostEqual(g_auto, g_fd, places=4)

    def test_depth_to_pressure_is_differentiable(self):
        z = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
        p = depth_to_pressure(z, ops=_ops())
        p.backward()
        g = float(z.grad)
        self.assertTrue(np.isfinite(g))
        self.assertGreater(g, 0.0)  # deeper is higher pressure


def _ops():
    from mixle_pde.ops import make_ops

    return make_ops()


if __name__ == "__main__":
    unittest.main()
