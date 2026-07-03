"""Tests for the SAFE guided-wave dispersion solver against exact plate references.

Two analytical checks. SH modes have the closed form ``c_n = c_s / sqrt(1 - (n pi c_s / (omega 2h))^2)``,
asserted directly. Lamb modes have no closed form, so the reference is obtained here by root-finding the
exact Rayleigh-Lamb transcendental equations, and the SAFE fundamental S0 / A0 phase velocities are asserted
to match those roots. Differentiability of the dispersion map w.r.t. plate thickness is also checked.
"""

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
    from mixle_pde.guided_wave import SAFEPlate, safe_dispersion
    from mixle_pde.ops import make_ops


def _rayleigh_lamb_residual(cph, omega, h, c_l, c_s, symmetric):
    """Residual of the exact Rayleigh-Lamb dispersion relation (h = half-thickness); its zeros are modes."""
    k = omega / cph
    p = np.sqrt((omega / c_l) ** 2 - k**2 + 0j)
    q = np.sqrt((omega / c_s) ** 2 - k**2 + 0j)
    tan_ph = np.tan(p * h)
    tan_qh = np.tan(q * h)
    if symmetric:
        # tan(q h) / tan(p h) = -4 k^2 p q / (q^2 - k^2)^2
        lhs = tan_qh / tan_ph
        rhs = -4.0 * k**2 * p * q / (q**2 - k**2) ** 2
    else:
        # tan(q h) / tan(p h) = -(q^2 - k^2)^2 / (4 k^2 p q)
        lhs = tan_qh / tan_ph
        rhs = -((q**2 - k**2) ** 2) / (4.0 * k**2 * p * q)
    return float(np.real(lhs - rhs))


def _lowest_lamb_root(omega, h, c_l, c_s, symmetric, c_lo=500.0, c_hi=6000.0, n=6000):
    """The lowest phase-velocity root of the Rayleigh-Lamb relation (the fundamental S0 or A0)."""
    grid = np.linspace(c_lo, c_hi, n)
    vals = np.array([_rayleigh_lamb_residual(c, omega, h, c_l, c_s, symmetric) for c in grid])
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            root = brentq(
                lambda c: _rayleigh_lamb_residual(c, omega, h, c_l, c_s, symmetric),
                grid[i],
                grid[i + 1],
            )
            return float(root)
    raise RuntimeError("no Rayleigh-Lamb root found in the scanned range")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class SHDispersionTestCase(unittest.TestCase):
    """SH modes match the exact closed form c_n = c_s / sqrt(1 - (n pi c_s / (omega 2h))^2)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_sh_phase_velocities(self):
        c_s, c_l, rho = 3200.0, 6300.0, 2700.0
        thickness = 1.0e-3  # total plate thickness (2h)
        freq = 5.0e6
        omega = 2.0 * np.pi * freq
        ops = make_ops()
        d = safe_dispersion(freq, thickness, ops, c_l=c_l, c_s=c_s, rho=rho, n_elem=60)

        sh = sorted(c for c, k in zip(d.cph, d.kind, strict=True) if k == "SH")
        # exact SH branches n = 0, 1, 2 (n=0 is the non-dispersive c = c_s)
        for n in range(3):
            arg = 1.0 - (n * np.pi * c_s / (omega * thickness)) ** 2
            self.assertGreater(arg, 0.0, f"SH mode {n} should be propagating at this frequency")
            c_exact = c_s / np.sqrt(arg)
            rel = abs(sh[n] - c_exact) / c_exact
            self.assertLess(rel, 0.01, f"SH{n}: SAFE {sh[n]:.2f} vs exact {c_exact:.2f} (rel {rel:.4f})")

    def test_sh0_is_nondispersive_shear_speed(self):
        c_s = 3200.0
        ops = make_ops()
        # SH0 sits at exactly c_s at every frequency
        for freq in (2.0e6, 4.0e6, 6.0e6):
            d = safe_dispersion(freq, 1.0e-3, ops, c_s=c_s, n_elem=40)
            self.assertAlmostEqual(d.phase_velocity("SH", 0) / c_s, 1.0, places=4)


@unittest.skipUnless(HAS_TORCH and HAS_SCIPY, "requires PyTorch and SciPy")
class LambDispersionTestCase(unittest.TestCase):
    """Fundamental Lamb modes match the exact Rayleigh-Lamb roots to ~1-2%."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_s0_a0_phase_velocities(self):
        c_s, c_l, rho = 3200.0, 6300.0, 2700.0
        thickness = 1.0e-3
        half = thickness / 2.0
        freq = 5.0e6  # fd = 5000 Hz*m
        omega = 2.0 * np.pi * freq
        ops = make_ops()
        d = safe_dispersion(freq, thickness, ops, c_l=c_l, c_s=c_s, rho=rho, n_elem=60)

        s0_safe = d.phase_velocity("S", 0)
        a0_safe = d.phase_velocity("A", 0)
        s0_exact = _lowest_lamb_root(omega, half, c_l, c_s, symmetric=True)
        a0_exact = _lowest_lamb_root(omega, half, c_l, c_s, symmetric=False)

        rel_s0 = abs(s0_safe - s0_exact) / s0_exact
        rel_a0 = abs(a0_safe - a0_exact) / a0_exact
        self.assertLess(rel_s0, 0.02, f"S0: SAFE {s0_safe:.2f} vs Rayleigh-Lamb {s0_exact:.2f} (rel {rel_s0:.4f})")
        self.assertLess(rel_a0, 0.02, f"A0: SAFE {a0_safe:.2f} vs Rayleigh-Lamb {a0_exact:.2f} (rel {rel_a0:.4f})")

    def test_a0_below_s0_at_low_fd(self):
        # at low frequency-thickness the flexural A0 is slower than the extensional S0
        ops = make_ops()
        d = safe_dispersion(2.0e6, 1.0e-3, ops, n_elem=50)
        self.assertLess(d.phase_velocity("A", 0), d.phase_velocity("S", 0))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentiabilityTestCase(unittest.TestCase):
    """The SAFE map is differentiable in the plate thickness (gradient flows through assembly)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_gradient_wrt_thickness(self):
        ops = make_ops()
        h = torch.tensor(1.0e-3, requires_grad=True)
        plate = SAFEPlate(h, c_s=3200.0, c_l=6300.0, rho=2700.0, n_elem=30)
        K1, K3, M = plate._sh_matrices(ops)
        # SH generalized eigenvalue k^2 for the fundamental (smallest positive) at a fixed omega
        omega = 2.0 * np.pi * 5.0e6
        op = torch.linalg.solve(K3, omega**2 * M - K1)
        k2 = torch.linalg.eigvals(op)
        # sum of real parts is a smooth differentiable scalar in h
        loss = k2.real.sum()
        loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertGreater(abs(float(h.grad)), 0.0)

    def test_thickness_tensor_matches_float(self):
        ops = make_ops()
        d_float = safe_dispersion(5.0e6, 1.0e-3, ops, n_elem=40)
        d_tensor = safe_dispersion(5.0e6, torch.tensor(1.0e-3), ops, n_elem=40)
        self.assertAlmostEqual(d_float.phase_velocity("A", 0), d_tensor.phase_velocity("A", 0), places=3)
        self.assertAlmostEqual(d_float.phase_velocity("S", 0), d_tensor.phase_velocity("S", 0), places=3)


if __name__ == "__main__":
    unittest.main()
