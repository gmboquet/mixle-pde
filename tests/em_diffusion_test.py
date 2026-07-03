"""Tests for the diffusive-EM forwards: 1-D layered MT (Wait recursion) + 2-D MT TE finite difference.

The acceptance bar is agreement with the analytical solution: a uniform half-space returns ``rho_a == 1/sigma``
exactly at every frequency with a 45-degree phase, and the plane wave decays into the earth over the skin depth
``delta = sqrt(2/(omega mu sigma))``.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    torch.set_default_dtype(torch.float64)
    from mixle_pde.em_diffusion import MU0, assemble_mt_te, layered_mt_impedance, mt_2d_te


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class LayeredMTTestCase(unittest.TestCase):
    """The 1-D Wait recursion reproduces the half-space sounding exactly and recurses stably for layers."""

    def test_halfspace_apparent_resistivity_and_phase(self):
        # uniform half-space: rho_a == 1/sigma EXACTLY at every frequency, phase == 45 deg.
        sigma = 0.01
        freqs = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        rho_a, phase, _ = layered_mt_impedance([sigma], [], freqs)
        self.assertTrue(np.allclose(rho_a.detach().numpy(), 1.0 / sigma, rtol=1e-12))
        self.assertTrue(np.allclose(phase.detach().numpy(), 45.0, atol=1e-10))

    def test_halfspace_matches_intrinsic_impedance(self):
        # Z of a half-space == sqrt(i omega mu / sigma)
        sigma, f = 0.05, 30.0
        _, _, Z = layered_mt_impedance([sigma], [], [f])
        omega = 2 * np.pi * f
        Z_exact = np.sqrt(1j * omega * MU0 / sigma)
        self.assertAlmostEqual(complex(Z[0].detach()).real, Z_exact.real, delta=1e-9)
        self.assertAlmostEqual(complex(Z[0].detach()).imag, Z_exact.imag, delta=1e-9)

    def test_two_layer_matches_closed_form_recursion(self):
        # a conductive layer over a resistive half-space; compare against an independent numpy Wait recursion.
        mu = MU0
        sig = np.array([0.1, 0.01])
        h1 = 300.0
        f = 5.0
        omega = 2 * np.pi * f
        k2 = np.sqrt(1j * omega * mu * sig[1])
        Zi2 = 1j * omega * mu / k2
        k1 = np.sqrt(1j * omega * mu * sig[0])
        Zi1 = 1j * omega * mu / k1
        Zref = Zi1 * (Zi2 + Zi1 * np.tanh(k1 * h1)) / (Zi1 + Zi2 * np.tanh(k1 * h1))
        rho_ref = np.abs(Zref) ** 2 / (omega * mu)
        phase_ref = np.degrees(np.angle(Zref))
        rho_a, phase, Z = layered_mt_impedance(sig, [h1], [f])
        self.assertAlmostEqual(float(rho_a[0]), float(rho_ref), delta=1e-6 * rho_ref)
        self.assertAlmostEqual(float(phase[0]), float(phase_ref), delta=1e-6)
        # conductive-over-resistive: apparent resistivity sits between the two layer resistivities; the phase
        # drops below 45 deg (the resistive basement makes the sounding look more resistive at low frequency)
        self.assertGreater(float(rho_a[0]), 1.0 / sig[0])
        self.assertLess(float(rho_a[0]), 1.0 / sig[1])
        self.assertLess(float(phase[0]), 45.0)

    def test_differentiable_in_conductivity(self):
        sig = torch.tensor([0.02, 0.2], requires_grad=True)
        rho_a, _, _ = layered_mt_impedance(sig, [200.0], [10.0])
        rho_a.sum().backward()
        self.assertIsNotNone(sig.grad)
        self.assertTrue(torch.isfinite(sig.grad).all())
        self.assertGreater(float(sig.grad.abs().sum()), 0.0)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MT2DTestCase(unittest.TestCase):
    """The 2-D TE finite-difference forward reproduces the half-space sounding and the skin-depth decay."""

    def test_halfspace_apparent_resistivity_and_phase(self):
        # laterally uniform half-space: the 2-D FD surface impedance matches rho_a == 1/sigma, phase ~ 45.
        sigma = 0.02
        f = 50.0
        delta = np.sqrt(2.0 / (2 * np.pi * f * MU0 * sigma))
        h = delta / 6.0
        nx, nz = 6, 240
        log_sigma = torch.log(torch.full((nx * nz,), sigma))
        rho_a, phase = mt_2d_te(log_sigma, (nx, nz), f, spacing=h)
        rho_np = rho_a.detach().numpy()
        phase_np = phase.detach().numpy()
        # discrete-consistent recovery: every surface site is 1/sigma and 45 deg to well under a percent
        self.assertTrue(np.allclose(rho_np, 1.0 / sigma, rtol=1e-3))
        self.assertTrue(np.allclose(phase_np, 45.0, atol=0.05))

    def test_skin_depth_decay_rate(self):
        # the plane wave decays into the conductor as exp(-z/delta): the fitted decay rate matches 1/delta.
        sigma = 0.05
        f = 100.0
        omega = 2 * np.pi * f
        delta = np.sqrt(2.0 / (omega * MU0 * sigma))
        h = delta / 8.0
        nz = 400
        nx = 5
        shape = (nx, nz)
        sig = torch.full((nx * nz,), sigma, dtype=torch.complex128)
        rows, cols, vals, n = assemble_mt_te(sig, shape, omega=omega, spacing=h)
        # drive E = 1 at the surface, analytic decay on the far boundary; solve, then fit the interior decay.
        from mixle_pde.pde_solve import sparse_solve

        idx = np.arange(nx * nz).reshape(nx, nz)
        zc = np.arange(nz) * h
        k = np.sqrt(1j * omega * MU0 * sigma)
        Eanalytic = np.exp(1j * k * zc)
        b = torch.zeros(nx * nz, dtype=torch.complex128)
        for i in range(nx):
            b[idx[i, 0]] = 1.0
            b[idx[i, nz - 1]] = complex(Eanalytic[nz - 1])
        for j in range(nz):
            b[idx[0, j]] = complex(Eanalytic[j])
            b[idx[nx - 1, j]] = complex(Eanalytic[j])
        E = sparse_solve(vals, rows, cols, n, b).detach().numpy().reshape(nx, nz)
        col = np.abs(E[nx // 2])
        mask = (zc > 0.5 * delta) & (zc < 3.0 * delta)
        rate = -np.polyfit(zc[mask], np.log(col[mask]), 1)[0]
        self.assertAlmostEqual(rate * delta, 1.0, delta=0.02)  # numerical decay rate == 1/delta within 2%

    def test_forward_is_differentiable(self):
        sigma = 0.02
        nx, nz = 5, 120
        f = 40.0
        delta = np.sqrt(2.0 / (2 * np.pi * f * MU0 * sigma))
        log_sigma = torch.log(torch.full((nx * nz,), sigma)).requires_grad_(True)
        rho_a, phase = mt_2d_te(log_sigma, (nx, nz), f, spacing=delta / 6.0)
        rho_a[nx // 2].backward()
        self.assertIsNotNone(log_sigma.grad)
        self.assertTrue(torch.isfinite(log_sigma.grad).all())
        self.assertGreater(float(log_sigma.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
