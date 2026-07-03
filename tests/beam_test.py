"""Tests for the Euler-Bernoulli beam solver against its closed-form results (simply-supported ends).

The static solve is checked against the textbook uniform-load center deflection ``w_max = 5 q L^4 / 384 EI``;
the dynamic leapfrog stepper is checked against the fundamental natural frequency
``omega_1 = (pi/L)^2 sqrt(EI / rho A)`` measured from the period of the first sine mode.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.beam import EulerBernoulliBeam
    from mixle_pde.ops import make_ops


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class StaticDeflectionTestCase(unittest.TestCase):
    """A simply-supported beam under a uniform load q has center deflection 5 q L^4 / 384 EI."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_uniform_load_center_deflection(self):
        n, L, EI, q = 101, 2.0, 3.0, 0.7
        beam = EulerBernoulliBeam(n, length=L, EI=EI)
        ops = make_ops()
        w = beam.static(q, ops).detach().numpy()
        w_num = float(w.max())
        w_exact = 5.0 * q * L**4 / (384.0 * EI)
        rel_err = abs(w_num - w_exact) / w_exact
        self.assertLess(rel_err, 0.02, f"numerical {w_num} vs analytical {w_exact} (rel err {rel_err})")
        # the peak is at the beam center for a symmetric load
        self.assertEqual(int(np.argmax(w)), n // 2)

    def test_scales_with_rigidity(self):
        # doubling EI halves the deflection (linear elasticity)
        ops = make_ops()
        b1 = EulerBernoulliBeam(81, length=1.0, EI=1.0)
        b2 = EulerBernoulliBeam(81, length=1.0, EI=2.0)
        w1 = b1.static(1.0, ops).detach().numpy().max()
        w2 = b2.static(1.0, ops).detach().numpy().max()
        self.assertAlmostEqual(w1 / w2, 2.0, places=6)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class NaturalFrequencyTestCase(unittest.TestCase):
    """The first sine mode of a simply-supported beam oscillates at omega_1 = (pi/L)^2 sqrt(EI/rho A)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_fundamental_frequency(self):
        n, L, EI, rho, A = 61, 1.0, 1.0, 1.0, 1.0
        omega1 = (np.pi / L) ** 2 * np.sqrt(EI / (rho * A))
        period = 2.0 * np.pi / omega1
        h = L / (n - 1)
        dt = 0.25 * h**2 / np.sqrt(EI / (rho * A))  # explicit 4th-order stability margin
        beam = EulerBernoulliBeam(n, length=L, EI=EI, rho=rho, A=A, dt=dt)
        x = np.linspace(0, L, n)
        w0 = np.sin(np.pi * x / L)  # the exact first mode shape
        ops = make_ops()
        mid = n // 2
        with torch.no_grad():
            state = beam.pack(torch.as_tensor(w0), torch.zeros(n))
            n_steps = int(1.5 * period / dt)  # long enough for two downward zero crossings of the center
            prev = float(beam.displacement(state)[mid])
            crossings = []
            for i in range(n_steps):
                state = beam.step(state, ops)
                cur = float(beam.displacement(state)[mid])
                if prev > 0.0 and cur <= 0.0:  # downward zero crossing, sub-step linear interpolation
                    crossings.append((i + prev / (prev - cur)) * dt)
                prev = cur
        self.assertGreaterEqual(len(crossings), 2)
        omega_meas = 2.0 * np.pi / (crossings[1] - crossings[0])
        rel_err = abs(omega_meas - omega1) / omega1
        self.assertLess(rel_err, 0.05, f"measured {omega_meas} vs analytical {omega1} (rel err {rel_err})")


if __name__ == "__main__":
    unittest.main()
