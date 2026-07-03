"""Spectral induced polarization (SIP) forward: verified against the Cole-Cole analytic spectrum and the
DC-resistivity limit."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    torch.set_default_dtype(torch.float64)
    from mixle_pde.geophysics import dc_resistivity
    from mixle_pde.induced_polarization import (
        apparent_conductivity,
        cole_cole_conductivity,
        geometric_factor,
        sip_forward,
    )


def _flat(shape):
    nx, ny = shape
    return lambda i, j: i * ny + j


def _schedule(shape):
    """A single interior dipole-dipole quadrupole (current dipole + potential dipole along a row)."""
    f = _flat(shape)
    nx, ny = shape
    r = nx // 2
    return [(f(r, 2), f(r, ny - 3), f(r, 3), f(r, ny - 4))]


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ColeColeLimitsTestCase(unittest.TestCase):
    def test_dc_limit(self):
        # omega -> 0: sigma(0) = sigma_inf (1 - m)
        sigma_inf, m, tau, c = 0.8, 0.3, 0.1, 0.5
        s0 = cole_cole_conductivity(0.0, sigma_inf, m, tau, c)
        self.assertAlmostEqual(s0.real.item(), sigma_inf * (1 - m), places=12)
        self.assertAlmostEqual(s0.imag.item(), 0.0, places=12)

    def test_high_frequency_limit(self):
        # omega -> inf: sigma -> sigma_inf
        sigma_inf, m, tau, c = 0.8, 0.3, 0.1, 0.5
        s_hi = cole_cole_conductivity(1e16, sigma_inf, m, tau, c)
        self.assertAlmostEqual(s_hi.real.item(), sigma_inf, places=6)
        self.assertAlmostEqual(s_hi.imag.item(), 0.0, places=6)

    def test_resistivity_phase_peaks_near_omega_tau_one(self):
        # the classic SIP curve is the phase of complex RESISTIVITY rho = 1/sigma; its magnitude peaks
        # near omega tau ~ 1 (the Cole-Cole relaxation), which is what a field survey reports.
        sigma_inf, m, tau, c = 1.0, 0.4, 0.05, 0.6
        w = np.logspace(-3, 3, 400) / tau
        s = torch.stack([cole_cole_conductivity(float(wi), sigma_inf, m, tau, c) for wi in w])
        rho = 1.0 / s
        phase = torch.atan2(rho.imag, rho.real).detach().numpy()
        w_peak = w[np.argmax(np.abs(phase))] * tau
        self.assertGreater(w_peak, 0.1)
        self.assertLess(w_peak, 10.0)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class SIPForwardTestCase(unittest.TestCase):
    def test_dc_reduction(self):
        # omega -> 0: sigma(0) = sigma_inf (1 - m); the SIP forward must match dc_resistivity run at that
        # DC conductivity to ~1e-6.
        shape = (11, 11)
        n = shape[0] * shape[1]
        sched = _schedule(shape)
        sigma_inf, m, tau, c = 0.8, 0.3, 0.1, 0.5
        log_sigma_inf = torch.log(torch.full((n,), float(sigma_inf)))
        R = sip_forward(log_sigma_inf, m, tau, c, shape, sched, [0.0])[0]
        # DC conductivity field for the reference forward
        sigma_dc = sigma_inf * (1 - m)
        log_sigma_dc = torch.log(torch.full((n,), float(sigma_dc)))
        R_dc = dc_resistivity(log_sigma_dc, shape, sched, log_data=False)
        self.assertLess(abs(R[0].imag.item()), 1e-9)
        self.assertLess(abs(R[0].real.item() - R_dc[0].item()), 1e-6)

    def test_recovers_cole_cole_spectrum(self):
        # homogeneous whole-space: apparent complex conductivity == input Cole-Cole sigma(omega) to ~1e-4
        shape = (13, 13)
        n = shape[0] * shape[1]
        sched = _schedule(shape)
        sigma_inf, m, tau, c = 0.9, 0.35, 0.02, 0.55
        log_sigma_inf = torch.log(torch.full((n,), float(sigma_inf)))
        w = np.logspace(-2, 2, 9) / tau
        R = sip_forward(log_sigma_inf, m, tau, c, shape, sched, w)  # (n_freq, 1)
        Kgeom = geometric_factor(shape, sched)  # (1,)
        sigma_app = apparent_conductivity(R, Kgeom.to(R.dtype))  # (n_freq, 1)
        for k, wi in enumerate(w):
            s_true = cole_cole_conductivity(float(wi), sigma_inf, m, tau, c)
            self.assertLess(abs(sigma_app[k, 0].item() - s_true.item()), 1e-4)

    def test_differentiable_in_parameters(self):
        shape = (11, 11)
        n = shape[0] * shape[1]
        sched = _schedule(shape)
        log_sigma_inf = torch.log(torch.full((n,), 0.8)).requires_grad_(True)
        m = torch.tensor(0.3, requires_grad=True)
        tau = torch.tensor(0.05, requires_grad=True)
        c = torch.tensor(0.6, requires_grad=True)
        w = [1.0 / 0.05]  # near the phase peak, where m/tau/c bite
        R = sip_forward(log_sigma_inf, m, tau, c, shape, sched, w)
        # a real scalar loss from the complex response (magnitude of the phase-bearing part)
        loss = (R.real**2 + R.imag**2).sum()
        loss.backward()
        for name, g in [("m", m.grad), ("tau", tau.grad), ("c", c.grad), ("log_sigma_inf", log_sigma_inf.grad)]:
            self.assertIsNotNone(g, f"no gradient for {name}")
            self.assertTrue(torch.isfinite(g).all(), f"non-finite gradient for {name}")
        # imaginary part carries the IP information, so m/tau/c gradients are non-zero
        self.assertGreater(abs(m.grad.item()), 0.0)
        self.assertGreater(abs(tau.grad.item()), 0.0)
        self.assertGreater(abs(c.grad.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
