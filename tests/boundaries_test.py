"""Boundary / interaction models (mixle_pde.boundaries): Rayleigh seabed R, critical angle, roughness, radar."""

import unittest

import numpy as np

from mixle_pde.boundaries import (
    bottom_loss_db,
    coherent_roughness_factor,
    critical_grazing_angle,
    impedance,
    radar_surface_reflection,
    rayleigh_roughness,
    seabed_reflection,
    surface_reflection,
)

try:
    import torch

    from mixle_pde.ops import make_ops

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CriticalAngleTest(unittest.TestCase):
    def test_critical_angle_value(self):
        # theta_c = arccos(c1/c2) for a couple of faster-bottom pairs.
        for c1, c2 in [(1500.0, 1800.0), (1480.0, 1650.0)]:
            self.assertAlmostEqual(float(critical_grazing_angle(c1, c2)), float(np.arccos(c1 / c2)), places=12)
        # 1500/1800 -> 33.557 deg.
        self.assertAlmostEqual(float(np.degrees(critical_grazing_angle(1500.0, 1800.0))), 33.55730976, places=6)

    def test_total_reflection_below_critical(self):
        # Faster lossless bottom: |R| = 1 below theta_c, |R| < 1 above.
        c1, rho1, c2, rho2 = 1500.0, 1000.0, 1800.0, 2000.0
        tc = float(critical_grazing_angle(c1, c2))
        for deg in [2.0, 10.0, 20.0, 33.0]:
            th = np.radians(deg)
            self.assertLess(th, tc)
            self.assertAlmostEqual(abs(complex(seabed_reflection(th, c1, rho1, c2, rho2))), 1.0, places=10)
        for deg in [34.0, 45.0, 60.0, 90.0]:
            th = np.radians(deg)
            self.assertGreater(th, tc)
            self.assertLess(abs(complex(seabed_reflection(th, c1, rho1, c2, rho2))), 1.0)

    def test_slower_bottom_has_no_critical_angle(self):
        # c2 < c1 -> arccos clips to 0 (no total-reflection regime).
        self.assertAlmostEqual(float(critical_grazing_angle(1500.0, 1400.0)), 0.0, places=12)


class NormalIncidenceTest(unittest.TestCase):
    def test_normal_incidence_impedance_contrast(self):
        # At normal incidence (grazing = pi/2) R = (Z2 - Z1)/(Z2 + Z1), Z = rho c. Sand/water pair.
        c1, rho1 = 1500.0, 1000.0  # water
        c2, rho2 = 1800.0, 2000.0  # sand
        z1, z2 = impedance(rho1, c1), impedance(rho2, c2)
        expected = (z2 - z1) / (z2 + z1)
        R = complex(seabed_reflection(np.pi / 2, c1, rho1, c2, rho2))
        self.assertAlmostEqual(R.imag, 0.0, places=12)
        self.assertAlmostEqual(R.real, expected, places=12)
        self.assertAlmostEqual(R.real, 0.41176470588, places=9)

    def test_bottom_loss_db(self):
        # BL = -20 log10 |R|; total reflection -> 0 dB; the normal-incidence pair -> a finite loss.
        R_tot = seabed_reflection(np.radians(10.0), 1500.0, 1000.0, 1800.0, 2000.0)
        self.assertAlmostEqual(float(bottom_loss_db(R_tot)), 0.0, places=8)
        R = seabed_reflection(np.pi / 2, 1500.0, 1000.0, 1800.0, 2000.0)
        self.assertAlmostEqual(float(bottom_loss_db(R)), -20.0 * np.log10(0.41176470588), places=8)

    def test_lossy_bottom_below_critical_leaks(self):
        # A volume attenuation makes |R| < 1 even below the critical angle (energy into the bottom).
        R = seabed_reflection(np.radians(10.0), 1500.0, 1000.0, 1800.0, 2000.0, attenuation_db_per_wavelength=0.5)
        self.assertLess(abs(complex(R)), 1.0)
        self.assertGreater(float(bottom_loss_db(R)), 0.0)


class SurfaceRoughnessTest(unittest.TestCase):
    def test_pressure_release(self):
        self.assertEqual(surface_reflection(), -1.0)

    def test_roughness_smooth_limit(self):
        # rho -> 1 as sigma -> 0 (smooth surface, full specular return).
        k = 2 * np.pi * 1000.0 / 1500.0
        self.assertAlmostEqual(float(coherent_roughness_factor(k, 0.0, np.pi / 4)), 1.0, places=12)

    def test_roughness_reference_value(self):
        # At the Rayleigh parameter g = k sigma sin theta = 0.5, rho = exp(-2 * 0.25).
        k = 2 * np.pi * 1000.0 / 1500.0
        theta = np.pi / 2
        sigma = 0.5 / (k * np.sin(theta))
        self.assertAlmostEqual(float(rayleigh_roughness(k, sigma, theta)), 0.5, places=12)
        self.assertAlmostEqual(float(coherent_roughness_factor(k, sigma, theta)), float(np.exp(-2.0 * 0.25)), places=12)
        self.assertAlmostEqual(float(coherent_roughness_factor(k, sigma, theta)), 0.60653065971, places=9)

    def test_roughness_monotone_decreasing(self):
        # rho decreases monotonically as g = k sigma sin theta grows.
        k = 2 * np.pi * 2000.0 / 1500.0
        theta = np.radians(30.0)
        vals = [float(coherent_roughness_factor(k, s, theta)) for s in [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(b, a)

    def test_miller_brown_vegh_floor(self):
        # MBV raises the coherent factor at large roughness (restores the incoherent floor).
        try:
            import scipy.special  # noqa: F401
        except ImportError:
            self.skipTest("scipy not available")
        k = 2 * np.pi * 2000.0 / 1500.0
        theta = np.radians(30.0)
        sigma = 0.3
        plain = float(coherent_roughness_factor(k, sigma, theta))
        mbv = float(coherent_roughness_factor(k, sigma, theta, miller_brown_vegh=True))
        self.assertGreater(mbv, plain)


class RadarSurfaceTest(unittest.TestCase):
    def test_grazing_tends_to_minus_one(self):
        # At near-grazing incidence both polarisations -> -1 (pressure-release-like). Vertical
        # approaches more slowly (pseudo-Brewster), so use a very small grazing angle.
        eps = 70.0 - 1j * 60.0 * 0.06 * 80.0  # sea water at ~X-band-ish
        for pol in ("horizontal", "vertical"):
            R = complex(radar_surface_reflection(np.radians(0.01), eps, polarisation=pol))
            self.assertAlmostEqual(R.real, -1.0, delta=1e-2)
            self.assertLess(abs(R), 1.0)

    def test_horizontal_matches_fresnel_formula(self):
        # R_h = (sin psi - w)/(sin psi + w), w = sqrt(eps - cos^2 psi).
        eps = 15.0 - 3.0j
        psi = np.radians(20.0)
        s = np.sin(psi)
        w = np.sqrt(eps - np.cos(psi) ** 2)
        expected = (s - w) / (s + w)
        R = complex(radar_surface_reflection(psi, eps, polarisation="horizontal"))
        self.assertAlmostEqual(R.real, expected.real, places=12)
        self.assertAlmostEqual(R.imag, expected.imag, places=12)

    def test_vertical_matches_fresnel_formula(self):
        eps = 15.0 - 3.0j
        psi = np.radians(20.0)
        s = np.sin(psi)
        w = np.sqrt(eps - np.cos(psi) ** 2)
        expected = (eps * s - w) / (eps * s + w)
        R = complex(radar_surface_reflection(psi, eps, polarisation="vertical"))
        self.assertAlmostEqual(R.real, expected.real, places=12)
        self.assertAlmostEqual(R.imag, expected.imag, places=12)

    def test_bad_polarisation_raises(self):
        with self.assertRaises(ValueError):
            radar_surface_reflection(np.radians(20.0), 15.0 - 3.0j, polarisation="circular")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class DifferentiabilityTest(unittest.TestCase):
    def test_seabed_reflection_grad_matches_fd(self):
        # d|R|/dc2 and d|R|/drho2 by autograd match finite differences at an above-critical angle.
        ops = make_ops()
        c1, rho1, c2v, rho2v = 1500.0, 1000.0, 1800.0, 2000.0
        theta = np.radians(45.0)
        c2 = torch.tensor(c2v, dtype=torch.float64, requires_grad=True)
        rho2 = torch.tensor(rho2v, dtype=torch.float64, requires_grad=True)
        R = seabed_reflection(
            torch.tensor(theta),
            torch.tensor(c1),
            torch.tensor(rho1),
            c2,
            rho2,
            ops=ops,
        )
        loss = torch.abs(R)
        loss.backward()
        g_c2 = float(c2.grad)
        g_rho2 = float(rho2.grad)
        self.assertTrue(np.isfinite(g_c2) and np.isfinite(g_rho2))

        h = 1e-3

        def mag(c2_, rho2_):
            return abs(complex(seabed_reflection(theta, c1, rho1, c2_, rho2_)))

        fd_c2 = (mag(c2v + h, rho2v) - mag(c2v - h, rho2v)) / (2 * h)
        fd_rho2 = (mag(c2v, rho2v + h) - mag(c2v, rho2v - h)) / (2 * h)
        self.assertAlmostEqual(g_c2, fd_c2, places=6)
        self.assertAlmostEqual(g_rho2, fd_rho2, places=6)

    def test_radar_reflection_grad_wrt_eps(self):
        # |R_v| differentiable w.r.t. the real permittivity.
        ops = make_ops()
        eps_re = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
        eps = eps_re + 1j * torch.tensor(-3.0, dtype=torch.complex128)
        R = radar_surface_reflection(torch.tensor(np.radians(20.0)), eps, polarisation="vertical", ops=ops)
        loss = torch.abs(R)
        loss.backward()
        self.assertTrue(np.isfinite(float(eps_re.grad)))


if __name__ == "__main__":
    unittest.main()
