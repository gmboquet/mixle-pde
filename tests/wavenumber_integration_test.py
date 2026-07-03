"""Analytic checks for the wavenumber-integration full-wave field.

1. Free-field Sommerfeld recovery: a single homogeneous halfspace must reproduce the point-source Green's
   function ``exp(i k R) / (4 pi R)`` to a few percent away from the source.
2. Pekeris waveguide (slow water over a faster fluid halfspace): the transmission loss shows cylindrical
   spreading (``TL ~ 10 log10 r``, slope ~1 not ~2) with modal interference, and the depth Green's function
   peaks at horizontal wavenumbers inside the trapped-mode band ``[omega / c2, omega / c1]``.
"""

from __future__ import annotations

import unittest

import numpy as np

from mixle_pde.wavenumber_integration import WavenumberIntegration1D


class SommerfeldRecoveryTest(unittest.TestCase):
    def test_free_field_matches_point_source_green(self):
        # single homogeneous halfspace (no surface): the wavenumber integral is the free-space Green's fn.
        # a small attenuation regularizes the branch-point pole for the real-axis quadrature.
        w = WavenumberIntegration1D(
            freq=50.0,
            depths=[],
            speeds=[1500.0],
            densities=[1000.0],
            zs=20.0,
            beta=0.0002,
            n_kr=250_000,
            kr_max_fac=8.0,
            surface=False,
        )
        for r in (40.0, 60.0, 80.0, 120.0):
            for z in (40.0, 60.0):
                num = complex(w.field(r, z))
                ana = complex(w.sommerfeld(r, z))
                # magnitude to about a percent
                self.assertAlmostEqual(abs(num) / abs(ana), 1.0, delta=0.02)
                # phase agrees (the outgoing spherical wave), within a couple of degrees
                dphase = np.angle(num / ana)
                self.assertLess(abs(dphase), 0.05)

    def test_vectorized_field_matches_scalar(self):
        w = WavenumberIntegration1D(
            freq=50.0,
            depths=[],
            speeds=[1500.0],
            densities=[1000.0],
            zs=20.0,
            beta=0.0002,
            n_kr=120_000,
            kr_max_fac=8.0,
            surface=False,
        )
        rs = np.array([50.0, 90.0, 150.0])
        pv = w.field(rs, 45.0).detach().numpy()
        for i, r in enumerate(rs):
            ps = complex(w.field(float(r), 45.0))
            self.assertAlmostEqual(pv[i].real, ps.real, places=10)
            self.assertAlmostEqual(pv[i].imag, ps.imag, places=10)


class PekerisWaveguideTest(unittest.TestCase):
    def setUp(self):
        # isovelocity water c1 = 1500 over a faster fluid halfspace c2 = 1800; 100 m deep.
        self.pek = WavenumberIntegration1D(
            freq=50.0,
            depths=[100.0],
            speeds=[1500.0, 1800.0],
            densities=[1000.0, 1500.0],
            zs=30.0,
            beta=[0.0001, 0.0001],
            n_kr=300_000,
            kr_max_fac=3.0,
        )

    def test_cylindrical_spreading_slope(self):
        # range-averaged intensity should follow cylindrical spreading TL ~ 10 log10 r (slope ~1),
        # NOT spherical 20 log10 r (slope ~2). Smooth out the modal interference before fitting.
        rs = np.linspace(1000.0, 8000.0, 240)
        p = self.pek.field(rs, 30.0).detach().numpy()
        intensity = np.abs(p) ** 2
        smooth = np.convolve(intensity, np.ones(21) / 21, mode="same")
        tl = -10.0 * np.log10(smooth + 1e-30)
        mask = (rs > 1500.0) & (rs < 7000.0)
        design = np.vstack([10.0 * np.log10(rs[mask]), np.ones(mask.sum())]).T
        slope = np.linalg.lstsq(design, tl[mask], rcond=None)[0][0]
        # cylindrical, well away from the spherical slope of 2
        self.assertAlmostEqual(slope, 1.0, delta=0.25)
        self.assertLess(slope, 1.5)

    def test_modal_interference_present(self):
        # a multimode waveguide beats: |p| has many extrema in range (mode interference), unlike the
        # monotone free-field decay.
        rs = np.linspace(500.0, 12000.0, 400)
        p = self.pek.field(rs, 30.0).detach().numpy()
        mag = np.abs(p)
        extrema = int(np.sum(np.diff(np.sign(np.diff(mag))) != 0))
        self.assertGreater(extrema, 10)

    def test_trapped_mode_wavenumbers_in_band(self):
        # the depth Green's function peaks at the trapped-mode horizontal wavenumbers, which must lie in
        # [omega/c2, omega/c1]. Take the dominant peaks of |g(kr)|.
        kr_lo, kr_hi = self.pek.mode_wavenumber_window()
        self.assertLess(kr_lo, kr_hi)
        g = self.pek.green(30.0).detach().numpy()
        kr = self.pek.kr
        mag = np.abs(g)
        # local maxima above a fraction of the global peak
        thresh = 0.2 * mag.max()
        peaks = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:]) & (mag[1:-1] > thresh))[0] + 1
        self.assertGreater(len(peaks), 0)
        kr_peaks = kr[peaks]
        # every strong pole sits inside the trapped-mode band (allow a hair of quadrature slack)
        self.assertTrue(np.all(kr_peaks > kr_lo - 1e-3))
        self.assertTrue(np.all(kr_peaks < kr_hi + 1e-3))
        # and the strongest peak is genuinely inside the band
        strongest = kr[int(np.argmax(mag))]
        self.assertGreater(strongest, kr_lo)
        self.assertLess(strongest, kr_hi)


if __name__ == "__main__":
    unittest.main()
