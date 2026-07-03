"""Tests for the KRAKEN-style depth normal-mode solver against the Pekeris waveguide.

The Pekeris waveguide (isovelocity water ``c1`` of depth ``D`` over a faster fluid halfspace ``c2``) has an
exact characteristic equation whose roots are the trapped-mode horizontal wavenumbers. The reference here is
obtained by root-finding that equation in-test; the solver's ``k_m`` are asserted to match it, to lie in the
trapped band ``(omega/c2, omega/c1)``, and to number the analytic count. Mode shapes are checked to be
sinusoidal in the water column and evanescent below the seabed, and the modal-sum transmission loss is
cross-checked against a ParabolicEquation2D run of the same waveguide (the standard PE-vs-mode validation).
"""

import math
import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.optimize import brentq

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if HAS_TORCH:
    from mixle_pde.normal_modes import NormalModes1D, pekeris_characteristic, pekeris_mode_count
    from mixle_pde.parabolic_equation import ParabolicEquation2D


# Pekeris waveguide reference parameters
FREQ = 50.0
DEPTH = 100.0
C1 = 1500.0
C2 = 1800.0
RHO1 = 1000.0
RHO2 = 1500.0


def _pekeris_reference_roots():
    """Trapped-mode wavenumbers of the Pekeris waveguide by root-finding the characteristic equation."""
    omega = 2.0 * math.pi * FREQ
    lo, hi = omega / C2, omega / C1
    grid = np.linspace(lo * (1 + 1e-7), hi * (1 - 1e-7), 60000)
    vals = pekeris_characteristic(grid, omega, DEPTH, C1, C2, RHO1, RHO2)
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            roots.append(
                brentq(
                    lambda k: pekeris_characteristic(k, omega, DEPTH, C1, C2, RHO1, RHO2),
                    grid[i],
                    grid[i + 1],
                )
            )
    return np.sort(np.array(roots))[::-1]  # descending, like the solver


@unittest.skipUnless(HAS_TORCH and HAS_SCIPY, "needs torch and scipy")
class PekerisModesTest(unittest.TestCase):
    def setUp(self):
        self.omega = 2.0 * math.pi * FREQ
        self.solver = NormalModes1D(DEPTH, C1, rho=RHO1, n_z=1000, bottom="halfspace", c_bottom=C2, rho_bottom=RHO2)
        self.modes = self.solver.solve(FREQ)
        self.ref = _pekeris_reference_roots()

    def test_wavenumbers_in_trapped_band(self):
        """Check 1: every trapped k_m lies strictly in (omega/c2, omega/c1)."""
        k_lo, k_hi = self.omega / C2, self.omega / C1
        self.assertGreater(self.modes.n_mode, 0)
        for k in self.modes.k:
            self.assertGreater(k, k_lo)
            self.assertLess(k, k_hi)
        # phase speeds correspondingly bracketed by the two layer speeds
        for c_ph in self.modes.c_ph:
            self.assertGreater(c_ph, C1)
            self.assertLess(c_ph, C2)

    def test_mode_count_matches_analytic(self):
        """Check 2a: trapped-mode count matches M = ceil((2 f D / c1) sqrt(1 - (c1/c2)^2))."""
        expected = pekeris_mode_count(FREQ, DEPTH, C1, C2)
        # the closed-form estimate stated in the task, made exact by the ceil
        g1_max = self.omega * math.sqrt(1.0 / C1**2 - 1.0 / C2**2)
        self.assertEqual(expected, int(math.ceil(g1_max * DEPTH / math.pi)))
        self.assertEqual(self.modes.n_mode, expected)
        self.assertEqual(self.modes.n_mode, len(self.ref))

    def test_wavenumbers_match_characteristic_roots(self):
        """Check 2b: solver k_m match the roots of the Pekeris characteristic equation."""
        self.assertEqual(len(self.modes.k), len(self.ref))
        rel = np.abs(self.modes.k - self.ref) / self.ref
        # second-order-interior / first-order-boundary FD: agreement well under 1%
        self.assertLess(rel.max(), 1e-2)
        # and each solver wavenumber is an actual near-zero of the characteristic residual
        for k in self.modes.k:
            g1 = math.sqrt((self.omega / C1) ** 2 - k**2)
            res = pekeris_characteristic(k, self.omega, DEPTH, C1, C2, RHO1, RHO2)
            self.assertLess(abs(res) / (g1 + 1.0), 5e-2)

    def test_mode_shapes_sinusoidal_and_evanescent(self):
        """Check 3: mode shapes are sinusoidal in the water column and evanescent below the seabed."""
        z = self.modes.z
        dz = z[1] - z[0]
        for m in range(self.modes.n_mode):
            phi = self.modes.phi[:, m]
            # (a) sinusoidal in water: mode m has exactly m interior zero crossings (m half-wavelengths)
            zc = int(np.sum(np.diff(np.sign(phi[1:-1])) != 0))
            self.assertEqual(zc, m)
            # (b) matches sin(gamma1 z) with gamma1 the in-water vertical wavenumber
            g1 = math.sqrt((self.omega / C1) ** 2 - self.modes.k[m] ** 2)
            fit = np.sin(g1 * z)
            corr = np.corrcoef(phi, fit)[0, 1]
            self.assertGreater(abs(corr), 0.999)
            # (c) evanescent below the seabed: phi'(D)/phi(D) = -gamma2 (downward decay exp(-gamma2 (z-D)))
            g2 = math.sqrt(self.modes.k[m] ** 2 - (self.omega / C2) ** 2)
            log_deriv = (phi[-1] - phi[-2]) / dz / phi[-1]
            # one-sided (first-order) slope at the seabed node; matches -gamma2 to a few percent
            self.assertLess(abs(log_deriv - (-g2)) / g2, 3e-2)

    def test_pe_cross_check_transmission_loss(self):
        """Check 4: modal-sum TL vs range agrees with a ParabolicEquation2D run of the same waveguide."""
        z_s, z_r = 36.0, 50.0
        dz = 0.5
        z_tot = 260.0
        nz = int(z_tot / dz)
        c_prof = np.where(np.arange(nz) * dz <= DEPTH, C1, C2)
        pe = ParabolicEquation2D(
            nz, dz=dz, dr=20.0, freq=FREQ, c0=C1, surface="pressure_release", absorb=60, absorb_strength=3.0
        )
        psi0 = pe.starter(z_s)
        n_range = int(6000.0 / 20.0)
        field = pe.march(psi0, torch.as_tensor(C1 / c_prof), n_range)
        tl_pe = pe.transmission_loss(field).detach().numpy()
        ranges = (np.arange(n_range) + 1) * 20.0
        i_zr = int(round(z_r / dz))
        tl_pe_zr = tl_pe[:, i_zr]

        tl_mode = self.solver.transmission_loss(self.modes, z_s, ranges, np.array([z_r]))
        tl_mode_zr = tl_mode.detach().numpy().reshape(-1)

        # community PE-vs-mode validation: range-smoothed TL agrees to a few dB after aligning the reference
        def smooth(x, w=15):
            return np.convolve(x, np.ones(w) / w, mode="same")

        sm, sp = smooth(tl_mode_zr), smooth(tl_pe_zr)
        win = (ranges >= 1500.0) & (ranges <= 4500.0)
        a, b = sm[win], sp[win]
        offset = np.mean(a - b)  # the two propagators use different source normalizations
        resid = a - (b + offset)
        rms = np.sqrt(np.mean(resid**2))
        self.assertLess(rms, 2.0)  # RMS agreement to under 2 dB
        self.assertLess(np.abs(resid).max(), 4.0)  # peak within a few dB
        self.assertGreater(np.corrcoef(a, b)[0, 1], 0.85)  # same decay-with-range shape


@unittest.skipUnless(HAS_TORCH, "needs torch")
class SolverBehaviorTest(unittest.TestCase):
    def test_differentiable_in_sound_speed(self):
        """The depth operator eigenvalues are differentiable in the c(z) field (autograd flows)."""
        n_z = 500
        c = torch.full((n_z,), C1, dtype=torch.float64, requires_grad=True)
        nm = NormalModes1D(DEPTH, c, rho=RHO1, n_z=n_z, bottom="halfspace", c_bottom=C2, rho_bottom=RHO2)
        omega = 2.0 * math.pi * FREQ
        B, _, _ = nm._generalized_operator(omega, 0.05)
        ev = torch.linalg.eigvalsh(B)
        ev.max().backward()
        self.assertIsNotNone(c.grad)
        self.assertTrue(torch.isfinite(c.grad).all())
        self.assertGreater(float(c.grad.abs().sum()), 0.0)

    def test_rigid_bottom_half_wavelength_modes(self):
        """A rigid-bottom isovelocity duct has the exact modes k_m = sqrt((omega/c)^2 - ((m-1/2) pi/D)^2)."""
        nm = NormalModes1D(DEPTH, C1, rho=RHO1, n_z=1200, bottom="rigid")
        modes = nm.solve(FREQ)
        omega = 2.0 * math.pi * FREQ
        # pressure-release surface + rigid bottom: vertical wavenumbers (m - 1/2) pi / D, m = 1, 2, ...
        gz = np.array([(m + 0.5) * math.pi / DEPTH for m in range(modes.n_mode)])
        k_exact = np.sqrt((omega / C1) ** 2 - gz**2)
        self.assertGreater(modes.n_mode, 0)
        rel = np.abs(modes.k - k_exact) / k_exact
        self.assertLess(rel.max(), 1e-2)


if __name__ == "__main__":
    unittest.main()
