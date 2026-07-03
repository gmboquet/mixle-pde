"""Atmospheric radio refractivity (mixle_pde.refractivity): ITU-R P.453 values + M-gradient + ducting + grad."""

import unittest

import numpy as np

from mixle_pde.refractivity import (
    M_CURVATURE_GRADIENT,
    duct_layers,
    modified_refractivity,
    refractivity,
    saturation_vapour_pressure,
    standard_refractivity_profile,
    vapour_pressure_from_humidity,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RefractivityValueTest(unittest.TestCase):
    def test_standard_sea_level_refractivity(self):
        # ITU-R P.453 N for standard sea-level air: P = 1013 hPa, T = 15 C, RH = 60%.
        # e = 0.60 * e_s(15C) = 0.60 * 17.0457 = 10.2274 hPa; N = 77.6/288.15*(1013 + 4810*e/288.15).
        # Published near-surface value is ~315 N-units; the exact inputs above give 318.78.
        t_c = 15.0
        T_k = t_c + 273.15
        P = 1013.0
        e = vapour_pressure_from_humidity(0.60, t_c)
        self.assertAlmostEqual(float(saturation_vapour_pressure(t_c)), 17.04570756, places=6)
        self.assertAlmostEqual(float(e), 10.22742454, places=6)
        N = refractivity(P, T_k, e)
        self.assertAlmostEqual(float(N), 318.781628, places=5)
        # Within a few N-units of the canonical standard value 315.
        self.assertAlmostEqual(float(N), 315.0, delta=5.0)

    def test_dry_term_only(self):
        # Zero humidity leaves the dry (density) term N_dry = 77.6 P / T.
        T_k = 288.15
        P = 1013.0
        N = refractivity(P, T_k, 0.0)
        self.assertAlmostEqual(float(N), 77.6 * P / T_k, places=10)

    def test_wet_term_raises_refractivity(self):
        # Adding water vapour raises N above the dry value (the wet term is positive).
        T_k = 288.15
        P = 1013.0
        N_dry = refractivity(P, T_k, 0.0)
        N_wet = refractivity(P, T_k, 10.0)
        self.assertGreater(float(N_wet), float(N_dry))

    def test_refractivity_vectorized_over_a_profile(self):
        # Elementwise over a sounding: arrays of P, T, e run in one call.
        P = np.array([1013.0, 1000.0, 950.0])
        T_k = np.array([288.15, 285.0, 280.0])
        e = np.array([10.0, 8.0, 5.0])
        N = refractivity(P, T_k, e)
        self.assertEqual(N.shape, (3,))
        for i in range(3):
            self.assertAlmostEqual(N[i], refractivity(P[i], T_k[i], e[i]), places=12)


class ModifiedRefractivityGradientTest(unittest.TestCase):
    def test_curvature_gradient_constant(self):
        # The Earth-flattening term is 1e6 / a_earth = 0.157 M-units per metre.
        self.assertAlmostEqual(M_CURVATURE_GRADIENT, 0.157, delta=5e-4)

    def test_standard_atmosphere_gradients(self):
        # Standard exponential atmosphere N(z) = 315 exp(-z/8077): published near-surface gradients are
        # dN/dz ~ -0.039 N-units/m and hence dM/dz = dN/dz + 0.157 ~ +0.118 M-units/m (NO ducting).
        z = np.linspace(0.0, 100.0, 11)
        N = standard_refractivity_profile(z)
        M = modified_refractivity(N, z)
        dN_dz = np.diff(N) / np.diff(z)
        dM_dz = np.diff(M) / np.diff(z)
        self.assertAlmostEqual(float(dN_dz.mean()), -0.039, delta=0.039 * 0.05)  # within ~5%
        self.assertAlmostEqual(float(dM_dz.mean()), +0.118, delta=0.118 * 0.05)  # within ~5%
        # dM/dz stays positive everywhere in the standard atmosphere.
        self.assertTrue(np.all(dM_dz > 0.0))

    def test_M_equals_N_plus_curvature(self):
        # M(z) = N(z) + 0.157 z exactly.
        z = np.array([0.0, 25.0, 50.0, 100.0])
        N = np.array([315.0, 314.0, 313.0, 311.0])
        M = modified_refractivity(N, z)
        expected = N + M_CURVATURE_GRADIENT * z
        for i in range(len(z)):
            self.assertAlmostEqual(M[i], expected[i], places=12)


class DuctingTest(unittest.TestCase):
    def test_standard_atmosphere_has_no_duct(self):
        # The standard atmosphere never has dM/dz < 0, so no trapping layer is detected.
        z = np.linspace(0.0, 500.0, 51)
        M = modified_refractivity(standard_refractivity_profile(z), z)
        self.assertFalse(duct_layers(z, M).any())

    def test_surface_duct_is_detected(self):
        # A trapping surface/evaporation duct: M decreases over the lowest ~20 m (dM/dz < 0), then recovers.
        z = np.array([0.0, 10.0, 20.0, 30.0, 50.0, 100.0])
        M = np.array([340.0, 337.0, 335.0, 336.0, 340.0, 348.0])
        mask = duct_layers(z, M)
        # The first two intervals (0-10, 10-20 m) are trapping; the rest are standard.
        self.assertTrue(mask.any())
        np.testing.assert_array_equal(mask, np.array([True, True, False, False, False]))

    def test_duct_layers_shape_validation(self):
        with self.assertRaises(ValueError):
            duct_layers(np.array([0.0]), np.array([340.0]))
        with self.assertRaises(ValueError):
            duct_layers(np.array([0.0, 1.0]), np.array([340.0, 335.0, 330.0]))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentiabilityTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        from mixle_pde.ops import make_ops

        self.ops = make_ops()

    def test_dN_dT_matches_finite_difference(self):
        # Autograd dN/dT (holding e fixed) finite and finite-difference-consistent.
        P = 1013.0
        e = 10.2274
        T0 = 288.15

        def N_of_T(T_k):
            return refractivity(P, T_k, e)

        T = torch.tensor(T0, requires_grad=True)
        N = N_of_T(T)
        N.backward()
        g_auto = float(T.grad)
        self.assertTrue(np.isfinite(g_auto))
        self.assertLess(g_auto, 0.0)  # N falls as T rises

        eps = 1e-4
        g_fd = (float(N_of_T(torch.tensor(T0 + eps))) - float(N_of_T(torch.tensor(T0 - eps)))) / (2 * eps)
        self.assertAlmostEqual(g_auto, g_fd, places=5)

    def test_dN_de_matches_finite_difference(self):
        # Autograd dN/de (holding T fixed) finite and finite-difference-consistent; the wet term slope is
        # d N / d e = 77.6 * 4810 / T^2, a positive constant.
        P = 1013.0
        T_k = 288.15
        e0 = 10.0

        def N_of_e(e):
            return refractivity(P, T_k, e)

        e = torch.tensor(e0, requires_grad=True)
        N = N_of_e(e)
        N.backward()
        g_auto = float(e.grad)
        self.assertTrue(np.isfinite(g_auto))
        self.assertAlmostEqual(g_auto, 77.6 * 4810.0 / T_k**2, places=8)

        eps = 1e-4
        g_fd = (float(N_of_e(torch.tensor(e0 + eps))) - float(N_of_e(torch.tensor(e0 - eps)))) / (2 * eps)
        self.assertAlmostEqual(g_auto, g_fd, places=5)

    def test_e_from_humidity_is_differentiable(self):
        # dvapour/dRH via autograd equals e_s(t) and matches a finite difference; keeps humidity a driver.
        t_c = 15.0
        rh0 = 0.60

        def e_of_rh(rh):
            return vapour_pressure_from_humidity(rh, t_c, ops=self.ops)

        rh = torch.tensor(rh0, requires_grad=True)
        e = e_of_rh(rh)
        e.backward()
        g_auto = float(rh.grad)
        self.assertTrue(np.isfinite(g_auto))
        self.assertAlmostEqual(g_auto, float(saturation_vapour_pressure(t_c)), places=6)


if __name__ == "__main__":
    unittest.main()
