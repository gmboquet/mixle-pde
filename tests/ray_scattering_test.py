"""Validation of the asymptotic scattering layer against classic high-frequency closed forms.

Reference values asserted:
  * Flat plate: sigma = 4 pi A^2 / lambda^2 (A = a b), normal incidence, matched to a few percent.
  * Dihedral (a x b plates): sigma = 8 pi a^2 b^2 / lambda^2.
  * Trihedral (triangular, edge a): sigma = 4 pi a^4 / (3 lambda^2) (triangular-corner coefficient).
  * Sphere (radius a, optical): sigma = pi a^2 (frequency-independent).
  * Knife-edge: L(0) ~ -6 dB, L(1) ~ -14..-16 dB, both checked against an in-test Fresnel integral.
  * Two-ray ground reflection: height-gain nulls at hr = m lambda d / (2 ht).
"""

import unittest

import numpy as np

from mixle_pde.ray_scattering import (
    fresnel_integral,
    knife_edge_diffraction,
    multipath_power,
    po_rcs,
    two_ray_pattern,
    wavelength,
)


def _fresnel_ref(v):
    """Independent Fresnel C(v), S(v) by dense trapezoid (cross-checks the module's Simpson version)."""
    t = np.linspace(0.0, v, 20001)
    c = np.trapezoid(np.cos(0.5 * np.pi * t * t), t)
    s = np.trapezoid(np.sin(0.5 * np.pi * t * t), t)
    return c, s


class TestPhysicalOpticsRCS(unittest.TestCase):
    def test_flat_plate(self):
        lam = 0.03  # 10 GHz
        a, b = 0.3, 0.2  # a few wavelengths on a side
        sigma = po_rcs("plate", lam, a=a, b=b)
        analytic = 4.0 * np.pi * (a * b) ** 2 / lam**2
        self.assertAlmostEqual(sigma / analytic, 1.0, delta=0.01)

    def test_dihedral(self):
        lam = 0.03
        a, b = 0.25, 0.25
        sigma = po_rcs("dihedral", lam, a=a, b=b)
        analytic = 8.0 * np.pi * a**2 * b**2 / lam**2
        self.assertAlmostEqual(sigma / analytic, 1.0, delta=0.01)

    def test_trihedral_triangular(self):
        lam = 0.03
        a = 0.25
        sigma = po_rcs("trihedral", lam, a=a)
        analytic = 4.0 * np.pi * a**4 / (3.0 * lam**2)
        self.assertAlmostEqual(sigma / analytic, 1.0, delta=0.01)

    def test_sphere_optical(self):
        lam = 0.03
        a = 0.5  # a >> lambda: optical regime
        sigma = po_rcs("sphere", lam, a=a)
        analytic = np.pi * a**2
        self.assertAlmostEqual(sigma / analytic, 1.0, delta=0.01)
        # frequency-independent
        self.assertAlmostEqual(po_rcs("sphere", 0.01, a=a), sigma, places=9)

    def test_plate_taper_peaks_at_normal(self):
        lam = 0.03
        a, b = 0.3, 0.2
        peak = po_rcs("plate", lam, a=a, b=b, incidence=0.0)
        off = po_rcs("plate", lam, a=a, b=b, incidence=0.05)
        self.assertLess(off, peak)

    def test_unknown_shape_raises(self):
        with self.assertRaises(ValueError):
            po_rcs("ogive", 0.03, a=1.0)


class TestKnifeEdge(unittest.TestCase):
    def test_fresnel_matches_reference(self):
        for v in (0.5, 1.0, 2.0):
            c, s = fresnel_integral(v)
            cr, sr = _fresnel_ref(v)
            self.assertAlmostEqual(c, cr, delta=1e-4)
            self.assertAlmostEqual(s, sr, delta=1e-4)
        # C, S -> 1/2 as v -> infinity
        # the Fresnel spiral converges slowly (as 1/(pi v)), so a modest v is only within ~0.05
        c_inf, s_inf = fresnel_integral(8.0)
        self.assertAlmostEqual(c_inf, 0.5, delta=0.05)
        self.assertAlmostEqual(s_inf, 0.5, delta=0.05)

    def test_loss_at_v0_is_6db(self):
        # edge on the line of sight: exactly half the field -> 20 log10(0.5) = -6.02 dB
        loss0 = float(knife_edge_diffraction(0.0))
        self.assertAlmostEqual(loss0, -6.02, delta=0.1)

    def test_loss_at_v1(self):
        # compute the reference straight from the in-test Fresnel integral
        c, s = _fresnel_ref(1.0)
        f = 0.5 * (1.0 + 1j) * ((0.5 - c) - 1j * (0.5 - s))
        ref = 20.0 * np.log10(abs(f))
        loss1 = float(knife_edge_diffraction(1.0))
        self.assertAlmostEqual(loss1, ref, delta=0.05)
        # and it lands in the classic -14..-16 dB band
        self.assertTrue(-16.5 <= loss1 <= -13.5, msg=f"L(1)={loss1} dB out of band")

    def test_loss_monotone_into_shadow(self):
        vs = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        losses = knife_edge_diffraction(vs)
        self.assertTrue(np.all(np.diff(losses) < 0.0))


class TestMultipath(unittest.TestCase):
    def test_two_ray_null_locations(self):
        # far-field two-ray: nulls at hr = m lambda d / (2 ht)
        lam = 0.1
        ht = 20.0
        d = 5000.0
        m = np.arange(1, 6)
        hr_nulls = m * lam * d / (2.0 * ht)
        p = two_ray_pattern(ht, hr_nulls, d, lam)
        # deep nulls: power far below the pattern's typical lobe level
        # a lobe maximum sits halfway between consecutive nulls
        hr_lobe = 0.5 * lam * d / (2.0 * ht)
        p_lobe = float(two_ray_pattern(ht, hr_lobe, d, lam))
        self.assertTrue(np.all(p / p_lobe < 1e-3), msg=f"nulls not deep: {p / p_lobe}")

    def test_two_ray_first_lobe_above_null(self):
        lam = 0.1
        ht, d = 20.0, 5000.0
        hr = np.linspace(1.0, 60.0, 4000)
        p = two_ray_pattern(ht, hr, d, lam)
        # find the analytic first null and confirm it is a local minimum of the sampled pattern
        first_null = 1.0 * lam * d / (2.0 * ht)
        i = int(np.argmin(np.abs(hr - first_null)))
        window = p[max(0, i - 50) : i + 50]
        self.assertAlmostEqual(p[i], window.min(), delta=window.max() * 1e-2 + 1e-12)

    def test_multipath_direct_only_inverse_square(self):
        # ground pushed irrelevant: direct-only power ~ 1/L^2
        lam = 0.05
        # perfectly-absorbing ground so only the direct ray survives
        p = multipath_power((0.0, 100.0), (300.0, 100.0), lam, reflection=0.0)
        ld = np.hypot(300.0, 0.0)
        self.assertAlmostEqual(p, 1.0 / ld**2, delta=1e-9)

    def test_multipath_building_adds_power_paths(self):
        # a reflecting wall to the side adds bounce paths -> different coherent sum than ground-only
        lam = 0.05
        tx = (0.0, 10.0)
        rx = (20.0, 15.0)
        p_ground = multipath_power(tx, rx, lam)
        wall = {"x": 25.0, "zmax": 30.0, "gamma": -1.0}
        p_wall = multipath_power(tx, rx, lam, building=wall)
        # the wall contributes at least one extra coherent path, changing the received power
        self.assertNotAlmostEqual(p_ground, p_wall, delta=1e-6)


class TestWavelength(unittest.TestCase):
    def test_wavelength(self):
        self.assertAlmostEqual(wavelength(10e9), 0.0299792458, places=9)


if __name__ == "__main__":
    unittest.main()
