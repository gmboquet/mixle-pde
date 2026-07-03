"""Tests for the 3D acoustic wave-equation forward: eigenmode frequency and energy conservation.

We seed a fixed-boundary standing wave (a Dirichlet eigenmode of the box), run the leapfrog with a constant
velocity and no absorption, and check the two things a correct wave solver must get right: the mode
oscillates at the analytical angular frequency ``omega = c*pi*sqrt(p^2+q^2+r^2)/L``, and the total energy
(kinetic + potential) is conserved to within a small drift.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.ops import make_ops
    from mixle_pde.wave3d import WaveEquation3D


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class SpongeTestCase(unittest.TestCase):
    def test_absorbing_layer_is_edge_localized(self):
        n = 16
        wave = WaveEquation3D(n, dt=0.01, absorb_width=4, absorb_strength=2.0)
        g = wave._gamma.reshape(n, n, n)
        c = n // 2
        self.assertGreater(g[0, c, c], 0.0)  # damping near a face
        self.assertGreater(g[c, 0, c], 0.0)
        self.assertGreater(g[c, c, 0], 0.0)
        self.assertEqual(g[c, c, c], 0.0)  # none in the interior
        self.assertEqual(WaveEquation3D(n, dt=0.01, absorb_width=0)._gamma.max(), 0.0)


def _eigenmode(n, p, q, r, c=1.0):
    """Grid, mode field, and analytical angular frequency for the (p,q,r) Dirichlet eigenmode of the box."""
    L = 1.0
    h = L / (n - 1)
    g = np.linspace(0.0, L, n)
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    u0 = np.sin(np.pi * p * xx / L) * np.sin(np.pi * q * yy / L) * np.sin(np.pi * r * zz / L)
    omega = c * np.pi * np.sqrt(p**2 + q**2 + r**2) / L
    return h, u0, omega


def _energy(u, w, wave, ops, dt, c):
    """Total discrete energy of the field.

    Kinetic ``0.5 w^2`` plus potential ``-0.5 c^2 u . lap(u)``, both built from the solver's own 7-point
    Laplacian so the energy is the quantity the stepper actually integrates. Because leapfrog stores ``u``
    and ``w`` half a step apart, a matching ``-0.5 dt c^2 w . lap(u)`` correction removes the O(dt)
    staggering wobble and recovers the exactly conserved (shadow) energy of the symplectic scheme.
    """
    h = wave.h
    lap = wave._lap(torch.as_tensor(u.ravel()), ops).detach().numpy()
    uf = u.ravel()
    wf = w.ravel()
    kinetic = 0.5 * np.sum(wf**2)
    potential = -0.5 * c**2 * np.sum(uf * lap)
    correction = -0.5 * dt * c**2 * np.sum(wf * lap)
    return (kinetic + potential + correction) * h**3


def _run_mode(n=24, p=1, q=1, r=1, c=1.0, periods=1.2):
    """Leapfrog the (p,q,r) mode from rest at max displacement; return the recorded probe series and dt."""
    h, u0, omega = _eigenmode(n, p, q, r, c=c)
    T = 2.0 * np.pi / omega
    # stable, accurate explicit step: CFL number well under the 3D limit 1/sqrt(3)
    dt = 0.25 * h / c
    n_steps = int(np.ceil(periods * T / dt))

    wave = WaveEquation3D(n, dt=dt, spacing=h, absorb_width=0)
    ops = make_ops()
    c2 = torch.as_tensor(np.full(n**3, c**2))
    state = wave.pack(torch.as_tensor(u0.ravel()), torch.zeros(n**3))

    probe = (n // 3, n // 3, n // 3)  # a generic interior node (nonzero for the fundamental)
    pidx = np.ravel_multi_index(probe, (n, n, n))

    us = [float(u0[probe])]
    energies = [_energy(u0, np.zeros((n, n, n)), wave, ops, dt, c)]
    for _ in range(n_steps):
        state = wave.step(state, c2, ops)
        u = wave.displacement(state).detach().numpy()
        w = state[n**3 :].detach().numpy()
        us.append(float(u[pidx]))
        energies.append(_energy(u, w, wave, ops, dt, c))
    return np.asarray(us), np.asarray(energies), dt, omega, T


def _measure_period(us, dt):
    """Numerical period from the mean spacing of zero crossings (sign changes) of the probe series."""
    s = np.sign(us)
    s[s == 0] = 1.0
    crossings = np.where(np.diff(s) != 0)[0]
    # linear-interpolate each crossing to sub-step resolution
    times = []
    for i in crossings:
        frac = us[i] / (us[i] - us[i + 1])
        times.append((i + frac) * dt)
    times = np.asarray(times)
    half_periods = np.diff(times)  # consecutive zero crossings are half a period apart
    return 2.0 * float(np.mean(half_periods))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class EigenmodeFrequencyTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_fundamental_oscillates_at_analytical_frequency(self):
        us, _, dt, omega, T = _run_mode(n=24, p=1, q=1, r=1)
        T_num = _measure_period(us, dt)
        omega_num = 2.0 * np.pi / T_num
        rel = abs(omega_num - omega) / omega
        self.assertLess(rel, 0.05, f"omega_num={omega_num:.4f} vs analytical {omega:.4f} (rel {rel:.3%})")

    def test_mixed_mode_oscillates_at_analytical_frequency(self):
        us, _, dt, omega, T = _run_mode(n=24, p=2, q=1, r=1)
        T_num = _measure_period(us, dt)
        omega_num = 2.0 * np.pi / T_num
        rel = abs(omega_num - omega) / omega
        self.assertLess(rel, 0.05, f"omega_num={omega_num:.4f} vs analytical {omega:.4f} (rel {rel:.3%})")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class EnergyConservationTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_energy_is_conserved_without_absorption(self):
        _, energies, _, _, _ = _run_mode(n=24, p=1, q=1, r=1, periods=1.2)
        e0 = energies[0]
        drift = np.abs(energies - e0).max() / e0
        # the symplectic scheme conserves this energy to O(round-off); a few percent is a generous bound
        self.assertLess(drift, 0.01, f"energy drift {drift:.3%} over the run")


if __name__ == "__main__":
    unittest.main()
