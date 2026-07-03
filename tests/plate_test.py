"""Tests for the Kirchhoff-Love thin-plate solver against the exact Navier (simply-supported) solution.

A simply-supported rectangular plate under a uniform load ``q`` has the closed-form double-sine-series
deflection

    w(x, y) = (16 q / (pi^6 D)) * sum_{m,n odd} sin(m pi x/a) sin(n pi y/b) / ( m n ((m/a)^2 + (n/b)^2)^2 ).

We solve the finite-difference biharmonic static problem under the same uniform load and check the max/center
deflection against the series, and (optionally) recover the fundamental frequency from the dynamic stepper.
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
    from mixle_pde.plate import KirchhoffPlate


def navier_series(plate, q, terms=15):
    """The exact simply-supported deflection field under uniform load ``q``, summed over odd m, n."""
    n, a, b, D = plate.n, plate.a, plate.b, plate.D
    x = np.linspace(0, a, n)
    y = np.linspace(0, b, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    w = np.zeros((n, n))
    for m in range(1, terms + 1, 2):
        for k in range(1, terms + 1, 2):
            w += np.sin(m * np.pi * xx / a) * np.sin(k * np.pi * yy / b) / (m * k * ((m / a) ** 2 + (k / b) ** 2) ** 2)
    return w * 16.0 * q / (np.pi**6 * D)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class StaticPlateTestCase(unittest.TestCase):
    """The static biharmonic solve matches the Navier double-sine series."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def _check(self, a, b):
        ops = make_ops()
        plate = KirchhoffPlate(41, E=2.0e11, nu=0.3, h=0.01, rho=7800.0, a=a, b=b)
        q = 1.0e4
        w = plate.static(q, ops).reshape(plate.n, plate.n).detach().numpy()
        w_ref = navier_series(plate, q, terms=15)
        c = plate.n // 2
        rel_max = abs(w.max() - w_ref.max()) / w_ref.max()
        rel_center = abs(w[c, c] - w_ref[c, c]) / w_ref[c, c]
        self.assertLess(rel_max, 0.05, f"a={a} b={b}: max deflection {w.max():.6e} vs series {w_ref.max():.6e}")
        self.assertLess(rel_center, 0.05, f"a={a} b={b}: center {w[c, c]:.6e} vs series {w_ref[c, c]:.6e}")
        return w.max(), w_ref.max()

    def test_square_plate_matches_series(self):
        num, ref = self._check(1.0, 1.0)
        self.assertGreater(num, 0.0)
        self.assertGreater(ref, 0.0)

    def test_rectangular_plate_matches_series(self):
        self._check(1.5, 1.0)

    def test_boundary_is_clamped_to_zero(self):
        ops = make_ops()
        plate = KirchhoffPlate(31, a=1.0, b=1.0)
        w = plate.static(1.0e4, ops).reshape(plate.n, plate.n).detach().numpy()
        edge = np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]])
        # simply supported: w = 0 on the edges (to the sparse-solve residual, vs a ~1e-3 interior deflection)
        self.assertLess(np.abs(edge).max(), 1e-6 * np.abs(w).max())


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DynamicPlateTestCase(unittest.TestCase):
    """The leapfrog stepper is stable and recovers the fundamental frequency."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_fundamental_frequency(self):
        ops = make_ops()
        plate = KirchhoffPlate(31, E=2.0e11, nu=0.3, h=0.01, rho=7800.0, a=1.0, b=1.0)
        dt = plate.dynamic_dt(safety=0.4)
        n = plate.n
        x = np.linspace(0, plate.a, n)
        y = np.linspace(0, plate.b, n)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        # start in the fundamental mode shape at rest: it oscillates at the fundamental frequency
        w = torch.as_tensor((np.sin(np.pi * xx / plate.a) * np.sin(np.pi * yy / plate.b)).ravel().copy())
        v = torch.zeros(n * n)

        omega_true = np.pi**2 * (1.0 / plate.a**2 + 1.0 / plate.b**2) * np.sqrt(plate.D / (plate.rho * plate.h))
        period = 2.0 * np.pi / omega_true
        ctr = n * (n // 2) + (n // 2)

        nsteps = int(3.0 * period / dt)
        hist = np.empty(nsteps)
        ts = np.empty(nsteps)
        for i in range(nsteps):
            w, v = plate.step((w, v), ops, load=0.0)
            hist[i] = float(w[ctr])
            ts[i] = (i + 1) * dt

        self.assertTrue(np.isfinite(hist).all())
        self.assertLess(np.abs(hist).max(), 2.0)  # stable: no blow-up from the unit-amplitude start

        # period from down-going zero crossings of the center displacement (interpolated)
        crossings = []
        for i in range(1, nsteps):
            if hist[i - 1] > 0 and hist[i] <= 0:
                t0 = ts[i - 1] + (ts[i] - ts[i - 1]) * hist[i - 1] / (hist[i - 1] - hist[i])
                crossings.append(t0)
        self.assertGreaterEqual(len(crossings), 2)
        omega_est = 2.0 * np.pi / np.diff(np.array(crossings)).mean()
        self.assertLess(abs(omega_est - omega_true) / omega_true, 0.05)


if __name__ == "__main__":
    unittest.main()
