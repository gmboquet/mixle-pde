"""Tests for the 1-D equilibrium Poisson-Nernst-Planck forward.

Acceptance bar is agreement with the stated analytical limits: the Boltzmann equilibrium distribution
``c_i = c_i^bulk exp(-z_i phi)`` and the Debye screening length ``lambda_D = sqrt(eps / sum z_i^2 c_i^bulk)``,
plus a finite-difference check of the ``rho_fixed`` gradient through the implicit-adjoint backward.
"""

import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from mixle_pde.pnp import debye_length, pnp_equilibrium


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class BoltzmannEquilibriumTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_symmetric_electrolyte_is_boltzmann(self):
        # 1:1 electrolyte, wall potential phi_left; each species must obey c_i = c_bulk exp(-z_i phi).
        m = 401
        L = 20.0
        h = L / (m - 1)
        z = [1.0, -1.0]
        c_bulk = [1.0, 1.0]
        phi, c = pnp_equilibrium(z, c_bulk, (m,), spacing=h, eps=1.0, phi_left=0.2, phi_right=0.0)
        phi = phi.detach().numpy()
        c = c.detach().numpy()
        for i, (zi, cbi) in enumerate(zip(z, c_bulk)):
            c_boltz = cbi * np.exp(-zi * phi)
            rel = np.max(np.abs(c[i] - c_boltz) / c_boltz)
            self.assertLess(rel, 0.02, f"species {i} deviates from Boltzmann by {rel:.3%}")

    def test_asymmetric_valence_is_boltzmann(self):
        # 2:1 salt (a divalent cation, monovalent anion), different bulk concentrations.
        m = 301
        L = 15.0
        h = L / (m - 1)
        z = [2.0, -1.0]
        c_bulk = [0.5, 1.0]  # electroneutral bulk: sum z_i c_i = 0
        phi, c = pnp_equilibrium(z, c_bulk, (m,), spacing=h, eps=1.0, phi_left=0.15, phi_right=0.0)
        phi = phi.detach().numpy()
        c = c.detach().numpy()
        for i, (zi, cbi) in enumerate(zip(z, c_bulk)):
            c_boltz = cbi * np.exp(-zi * phi)
            rel = np.max(np.abs(c[i] - c_boltz) / c_boltz)
            self.assertLess(rel, 0.02, f"species {i} deviates from Boltzmann by {rel:.3%}")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DebyeScreeningTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def _decay_length(self, x, phi, lo, hi):
        mask = (x > lo) & (x < hi)
        slope = np.polyfit(x[mask], np.log(np.abs(phi[mask])), 1)[0]
        return -1.0 / slope

    def test_screening_length_matches_debye(self):
        # Weakly charged wall: the near-wall potential decays as exp(-x/lambda_D). Fit the numerical decay
        # length and compare with lambda_D = sqrt(eps / sum z_i^2 c_i^bulk).
        m = 801
        L = 30.0
        h = L / (m - 1)
        x = np.linspace(0, L, m)
        z = [1.0, -1.0]
        c_bulk = [1.0, 1.0]
        eps = 1.0
        lam_D = debye_length(z, c_bulk, eps=eps)
        # small wall potential so the linearized (Debye) decay is the right analytic limit
        phi, _ = pnp_equilibrium(z, c_bulk, (m,), spacing=h, eps=eps, phi_left=0.05, phi_right=0.0)
        phi = phi.detach().numpy()
        lam_fit = self._decay_length(x, phi, 1.0, 12.0)
        rel = abs(lam_fit - lam_D) / lam_D
        self.assertLess(rel, 0.05, f"decay length {lam_fit:.4f} vs Debye {lam_D:.4f} (rel {rel:.3%})")

    def test_higher_ionic_strength_screens_more(self):
        # Doubling the bulk concentration shortens lambda_D by sqrt(2); the numerical decay must follow.
        m = 801
        L = 30.0
        h = L / (m - 1)
        x = np.linspace(0, L, m)
        z = [1.0, -1.0]
        eps = 1.0
        for c0 in (1.0, 2.0):
            c_bulk = [c0, c0]
            lam_D = debye_length(z, c_bulk, eps=eps)
            phi, _ = pnp_equilibrium(z, c_bulk, (m,), spacing=h, eps=eps, phi_left=0.05, phi_right=0.0)
            phi = phi.detach().numpy()
            lam_fit = self._decay_length(x, phi, 1.0, 10.0)
            self.assertLess(abs(lam_fit - lam_D) / lam_D, 0.05)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentiabilityTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_rho_fixed_gradient_matches_finite_difference(self):
        # A fixed-charge slab (pore charge) drives phi; d(loss)/d(rho_fixed) from the implicit-adjoint backward
        # must match a central finite difference at an interior node.
        m = 201
        L = 10.0
        h = L / (m - 1)
        z = [1.0, -1.0]
        c_bulk = [1.0, 1.0]
        rho0 = np.zeros(m)
        rho0[90:110] = 0.3  # a charged slab in the middle
        rho_t = torch.tensor(rho0, requires_grad=True)
        phi, _ = pnp_equilibrium(z, c_bulk, (m,), spacing=h, eps=1.0, rho_fixed=rho_t)
        loss = (phi**2).sum()
        loss.backward()
        g_auto = rho_t.grad.detach().numpy().copy()

        def loss_at(rv):
            p, _ = pnp_equilibrium(z, c_bulk, (m,), spacing=h, eps=1.0, rho_fixed=torch.tensor(rv))
            return float((p**2).sum())

        eps_fd = 1e-6
        k = 100
        rp = rho0.copy()
        rp[k] += eps_fd
        rm = rho0.copy()
        rm[k] -= eps_fd
        g_fd = (loss_at(rp) - loss_at(rm)) / (2 * eps_fd)
        self.assertLess(abs(g_auto[k] - g_fd), 1e-4 * max(1.0, abs(g_fd)))

    def test_diffusivity_gradient_is_zero_at_equilibrium(self):
        # Zero-current equilibrium is independent of D_i, so its gradient is exactly zero (correct physics).
        m = 101
        L = 8.0
        h = L / (m - 1)
        z = [1.0, -1.0]
        c_bulk = [1.0, 1.0]
        D = torch.tensor([1.5, 0.8], requires_grad=True)
        phi, c = pnp_equilibrium(z, c_bulk, (m,), spacing=h, eps=1.0, phi_left=0.1, diffusivity=D)
        (phi.sum() + c.sum()).backward()
        self.assertIsNotNone(D.grad)
        self.assertLess(float(D.grad.abs().max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
