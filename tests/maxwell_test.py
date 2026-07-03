"""Tests for the 3-D FDTD Maxwell stepper: the no-monopole invariant and a PEC-cavity resonant mode.

Both checks are against analytics. (a) The Yee scheme preserves ``div(mu H)`` exactly, so a divergence-free
initial magnetic field stays divergence-free to rounding for all time (no spurious magnetic monopoles).
(b) A PEC box cavity rings its TM_110 mode at the analytical angular frequency
``omega = c*pi*sqrt(l^2+m^2+n^2)/L``; we seed the mode, sample an interior antinode, and recover the
frequency from the time series.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.maxwell import Maxwell3D
    from mixle_pde.ops import make_ops


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MaxwellDivergenceTestCase(unittest.TestCase):
    """The Yee scheme preserves div(mu H) to machine precision for a div-free initial H."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_div_H_stays_zero(self):
        n = 24
        h = 1.0 / n
        c = 1.0
        dt = 0.5 * h / (c * np.sqrt(3))  # comfortably under the 3-D Courant limit h/(c*sqrt(3))
        m = Maxwell3D(n, dt=dt, spacing=h)
        ops = make_ops()

        # A divergence-free H by construction: H = curl(A) using the *same* forward-difference curl that the
        # H update uses, so the grid divergence div_H(curl A) = 0 to rounding at t = 0.
        rng = np.random.RandomState(1)
        Ax = torch.as_tensor(rng.randn(n, n, n))
        Ay = torch.as_tensor(rng.randn(n, n, n))
        Az = torch.as_tensor(rng.randn(n, n, n))
        Hx, Hy, Hz = m._curl_E(Ax, Ay, Az, h)
        z = torch.zeros(n, n, n)
        state = m.pack(z, z, z, Hx, Hy, Hz)

        d0 = float(m.div_H(state, ops).abs().max())
        self.assertLess(d0, 1e-10)  # div-free initial condition

        for _ in range(400):
            state = m.step(state, ops)

        d1 = float(m.div_H(state, ops).abs().max())
        self.assertTrue(torch.isfinite(state).all())
        self.assertLess(d1, 1e-10)  # still div-free after 400 leapfrog steps


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MaxwellCavityTestCase(unittest.TestCase):
    """A PEC box cavity rings its TM_110 mode at the analytical frequency."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_tm110_resonant_frequency(self):
        n = 24
        h = 1.0 / n
        L = 1.0  # cavity side length (n cells of size h)
        c = 1.0
        dt = 0.5 * h / (c * np.sqrt(3))
        m = Maxwell3D(n, dt=dt, spacing=h, eps=1.0, mu=1.0)
        ops = make_ops()
        self.assertAlmostEqual(m.c, 1.0)

        # TM_110 cavity mode: Ez = sin(pi x / L) sin(pi y / L), uniform in z; Ex = Ey = 0. The PEC walls hold
        # tangential E at zero, which sin(pi x / L) already satisfies at x = 0, L.
        x = np.arange(n) * h
        prof = np.sin(np.pi * x / L)
        Ez = torch.as_tensor(prof[:, None, None] * prof[None, :, None] * np.ones((1, 1, n)))
        z = torch.zeros(n, n, n)
        state = m.pack(z, z, Ez, z, z, z)

        omega_a = c * np.pi * np.sqrt(1**2 + 1**2 + 0**2) / L  # analytical TM_110 angular frequency
        period_a = 2 * np.pi / omega_a

        ic = jc = kc = n // 2  # interior antinode of the mode
        n_steps = int(8 * period_a / dt)  # several periods for FFT resolution
        samples = np.empty(n_steps)
        for i in range(n_steps):
            state = m.step(state, ops)
            samples[i] = float(m.fields(state, ops)[2][ic, jc, kc])

        # dominant frequency of the interior time series
        spectrum = np.fft.rfft(samples - samples.mean())
        freqs = np.fft.rfftfreq(n_steps, dt)
        peak = freqs[int(np.argmax(np.abs(spectrum)))]
        omega_num = 2 * np.pi * peak

        rel_err = abs(omega_num - omega_a) / omega_a
        self.assertLess(rel_err, 0.05)  # within 5% of analytics
        self.assertGreater(float(np.abs(samples).max()), 0.5)  # the mode actually oscillates


if __name__ == "__main__":
    unittest.main()
