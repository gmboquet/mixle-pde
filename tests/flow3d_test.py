"""Physics tests for the 3D incompressible Navier-Stokes solver (Chorin projection).

Three checks on the ``NavierStokes3D`` stepper: (a) the pressure projection drives the discrete divergence
of the corrected velocity down by many orders of magnitude, (b) with no forcing the total kinetic energy
decreases monotonically (viscous dissipation), and (c) a single decaying divergence-free no-slip mode loses
kinetic energy at the analytical viscous rate ``exp(-2 nu k^2 t)``.

The mode is ``psi = sin^2(pi x) sin^2(pi y) sin(pi z)`` with velocity ``(u, v, w) = (psi_y, -psi_x, 0)``. It
is exactly divergence-free (``div = psi_yx - psi_xy = 0``) and vanishes on all six no-slip walls, so the mask
introduces no jump and the projection barely touches it. Its continuum Rayleigh quotient (the negative
Laplacian eigenvalue that fixes the viscous decay rate) is ``k^2 = 19 pi^2 / 3`` in closed form.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.flow3d import NavierStokes3D
    from mixle_pde.ops import make_ops

# analytical continuum Rayleigh quotient of the seeded mode (negative Laplacian eigenvalue): k^2 = 19 pi^2 / 3
_MODE_K2 = 19.0 * np.pi**2 / 3.0


def _grid(n):
    g = np.linspace(0, 1, n)
    return np.meshgrid(g, g, g, indexing="ij")


def _decaying_mode(n, amp, ops, ns):
    """The exactly-divergence-free, no-slip mode ``(psi_y, -psi_x, 0)`` for ``psi=sin^2 x sin^2 y sin z``."""
    xx, yy, zz = _grid(n)
    s = lambda a: np.sin(np.pi * a)  # noqa: E731
    c = lambda a: np.cos(np.pi * a)  # noqa: E731
    u = (s(xx) ** 2) * (2 * np.pi * s(yy) * c(yy)) * s(zz)  # d psi / dy
    v = -(2 * np.pi * s(xx) * c(xx)) * (s(yy) ** 2) * s(zz)  # -d psi / dx
    mask = torch.as_tensor(ns._mask)
    return (
        amp * torch.as_tensor(u.ravel()) * mask,
        amp * torch.as_tensor(v.ravel()) * mask,
        ops.zeros(n**3),
    )


def _kinetic_energy(state):
    return 0.5 * float(sum((f * f).sum() for f in state))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class NavierStokes3DProjectionTestCase(unittest.TestCase):
    """The pressure projection makes the corrected velocity discretely divergence-free."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_projection_drives_divergence_small(self):
        ops = make_ops()
        n = 24
        h = 1.0 / (n - 1)
        nu = 0.05
        dt = 0.15 * h
        ns = NavierStokes3D(n, viscosity=nu, dt=dt)
        xx, yy, zz = _grid(n)
        mask = torch.as_tensor(ns._mask)
        # a swirling, deliberately non-divergence-free initial field (a Gaussian blob times a rotation)
        blob = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2 + (zz - 0.5) ** 2) / 0.05)
        state = (
            torch.as_tensor((blob * (yy - 0.5)).ravel()) * mask,
            torch.as_tensor((blob * (0.5 - xx)).ravel()) * mask,
            torch.as_tensor((blob * (zz - 0.5) * 0.3).ravel()) * mask,
        )
        # form u* (advect + diffuse) and measure its divergence before the projection
        u, v, w = state
        us = (u + dt * (-ns._advect(u, v, w, u, ops) + nu * ns._lap(u, ops))) * mask
        vs = (v + dt * (-ns._advect(u, v, w, v, ops) + nu * ns._lap(v, ops))) * mask
        ws = (w + dt * (-ns._advect(u, v, w, w, ops) + nu * ns._lap(w, ops))) * mask
        pre = float(ns.divergence((us, vs, ws), ops).norm())
        # the full step includes the projection; measure the corrected velocity's divergence
        post = float(ns.divergence(ns.step(state, ops), ops).norm())
        self.assertGreater(pre, 1e-2)  # the pre-projection field really is compressible
        self.assertLess(post, pre * 1e-4)  # the projection kills it by >= 4 orders of magnitude
        self.assertLess(post, 1e-5)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class NavierStokes3DDissipationTestCase(unittest.TestCase):
    """No forcing: kinetic energy decays monotonically at the analytical viscous rate."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_kinetic_energy_decreases_monotonically(self):
        ops = make_ops()
        n = 24
        h = 1.0 / (n - 1)
        ns = NavierStokes3D(n, viscosity=0.05, dt=0.15 * h)
        state = _decaying_mode(n, 0.02, ops, ns)
        energies = [_kinetic_energy(state)]
        for _ in range(40):
            state = ns.step(state, ops)
            energies.append(_kinetic_energy(state))
        energies = np.array(energies)
        self.assertTrue(np.all(np.diff(energies) < 0.0))  # strictly decreasing (viscous dissipation)
        self.assertLess(energies[-1], energies[0])

    def test_mode_decays_at_analytical_viscous_rate(self):
        ops = make_ops()
        n = 24
        h = 1.0 / (n - 1)
        nu = 0.05
        dt = 0.15 * h
        ns = NavierStokes3D(n, viscosity=nu, dt=dt)
        state = _decaying_mode(n, 0.02, ops, ns)
        self.assertLess(float(ns.divergence(state, ops).norm()), 1e-10)  # starts divergence-free
        nt = 40
        energies = [_kinetic_energy(state)]
        for _ in range(nt):
            state = ns.step(state, ops)
            energies.append(_kinetic_energy(state))
        energies = np.array(energies)
        t_grid = dt * np.arange(nt + 1)
        # fit log(KE) vs t; the slope is -(2 nu k^2) since KE ~ KE0 exp(-2 nu k^2 t)
        measured_rate = -np.polyfit(t_grid, np.log(energies), 1)[0]
        analytical_rate = 2.0 * nu * _MODE_K2
        self.assertLess(abs(measured_rate - analytical_rate) / analytical_rate, 0.15)  # within 15%


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class NavierStokes3DDifferentiableTestCase(unittest.TestCase):
    """Gradients flow through the step and the pressure projection (the adjoint sparse solve)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_gradient_flows_through_projection(self):
        ops = make_ops()
        n = 12
        h = 1.0 / (n - 1)
        ns = NavierStokes3D(n, viscosity=0.05, dt=0.15 * h)
        xx, yy, zz = _grid(n)
        s = lambda a: np.sin(np.pi * a)  # noqa: E731
        c = lambda a: np.cos(np.pi * a)  # noqa: E731
        um = (s(xx) ** 2) * (2 * np.pi * s(yy) * c(yy)) * s(zz)
        vm = -(2 * np.pi * s(xx) * c(xx)) * (s(yy) ** 2) * s(zz)
        mask = torch.as_tensor(ns._mask)
        amp = torch.tensor(0.02, requires_grad=True)
        state = (amp * torch.as_tensor(um.ravel()) * mask, amp * torch.as_tensor(vm.ravel()) * mask, ops.zeros(n**3))
        for _ in range(3):
            state = ns.step(state, ops)
        loss = 0.5 * sum((f * f).sum() for f in state)
        loss.backward()
        self.assertTrue(torch.isfinite(amp.grad))
        self.assertNotEqual(float(amp.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
