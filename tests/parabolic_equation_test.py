"""Split-step Fourier parabolic-equation propagator (mixle_pde.parabolic_equation).

Validated against the canonical PE benchmarks: the Lloyd-mirror interference pattern under a
pressure-release surface, machine-precision energy conservation of the unitary free march, a trapping
radar duct, and autograd differentiability in the refractive-index field.
"""

import unittest

import numpy as np

from mixle_pde.parabolic_equation import (
    ParabolicEquation2D,
    lloyd_mirror_pressure,
    modified_refractivity_index,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _local_minima(y, order):
    """Indices of strict local minima of ``y`` over a +/- ``order`` window."""
    idx = []
    for i in range(order, len(y) - order):
        if y[i] == min(y[i - order : i + order + 1]) and y[i] < y[i - 1] and y[i] < y[i + 1]:
            idx.append(i)
    return idx


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class LloydMirrorTest(unittest.TestCase):
    """A point source under a pressure-release surface: |p| ~ (2/r) |sin(k z_s z / r)| in the far field."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.c0 = 1500.0
        self.freq = 300.0  # 5 m wavelength
        self.k0 = 2.0 * np.pi * self.freq / self.c0
        self.z_s = 20.0  # source depth (m)
        self.nz = 1024
        self.dz = 300.0 / self.nz  # 300 m deep domain
        self.dr = 1.0
        self.n_range = 600  # range 600 m
        self.pe = ParabolicEquation2D(
            self.nz,
            dz=self.dz,
            dr=self.dr,
            freq=self.freq,
            c0=self.c0,
            surface="pressure_release",
            absorb=120,
            absorb_strength=5.0,
        )
        self.psi0 = self.pe.starter(self.z_s, width=4.0 / self.k0)
        field = self.pe.march(self.psi0, np.ones(self.nz), self.n_range)
        self.z = self.pe.depths().detach().numpy()
        self.r = self.n_range * self.dr
        self.p_num = np.abs(self.pe.pressure(field).detach().numpy()[-1])

    def test_far_field_pattern_correlates_with_analytic(self):
        # Correlate the numerical depth profile with the analytic Lloyd-mirror pattern in the shallow,
        # small-angle band where the PE approximation is valid (away from the source and the absorber).
        p_ana = lloyd_mirror_pressure(self.k0, self.z_s, self.z, self.r)
        band = (self.z > 1.5 * self.z_s) & (self.z < 150.0)
        a = self.p_num[band] / self.p_num[band].max()
        b = p_ana[band] / p_ana[band].max()
        corr = float(np.corrcoef(a, b)[0, 1])
        self.assertGreater(corr, 0.94)  # measured ~0.96

    def test_first_interference_null_location(self):
        # Analytic nulls sit at z_m = m pi r / (k z_s); the first must match the numerical null to a few %.
        null_spacing = np.pi * self.r / (self.k0 * self.z_s)  # 75 m for this geometry
        self.assertAlmostEqual(null_spacing, 75.0, places=6)
        band = (self.z > 1.5 * self.z_s) & (self.z < 150.0)
        zb = self.z[band]
        pb = self.p_num[band]
        mins = _local_minima(pb, order=20)
        self.assertTrue(mins, "no interference null found")
        numeric_null = min((zb[i] for i in mins), key=lambda x: abs(x - null_spacing))
        rel_err = abs(numeric_null - null_spacing) / null_spacing
        self.assertLess(rel_err, 0.05)  # measured ~0% (75.0 m)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class EnergyConservationTest(unittest.TestCase):
    """The free split-step march is unitary: the vertical energy integral of |psi|^2 is conserved."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_lossless_homogeneous_march_conserves_energy(self):
        # No absorber, field kept clear of the boundaries: energy is conserved to machine precision.
        nz, dz, dr = 512, 0.4, 2.0
        pe = ParabolicEquation2D(nz, dz=dz, dr=dr, freq=150.0, c0=1500.0, surface="free", absorb=0)
        psi0 = pe.starter(nz * dz / 2.0, width=4.0)  # centred, narrow angular spectrum
        field = pe.march(psi0, np.ones(nz), 200)
        E = pe.energy(field).detach().numpy()
        rel_change = abs(E[-1] - E[0]) / E[0]
        self.assertLess(rel_change, 1e-2)  # measured ~1e-12

    def test_pressure_release_march_conserves_energy(self):
        # The image method keeps the surface a perfect node without leaking energy from the physical half.
        nz, dz, dr = 512, 0.4, 2.0
        pe = ParabolicEquation2D(nz, dz=dz, dr=dr, freq=150.0, c0=1500.0, surface="pressure_release", absorb=0)
        psi0 = pe.starter(nz * dz / 2.0, width=4.0)
        field = pe.march(psi0, np.ones(nz), 150)
        E = pe.energy(field).detach().numpy()
        rel_change = abs(E[-1] - E[0]) / E[0]
        self.assertLess(rel_change, 1e-2)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class RadarDuctTest(unittest.TestCase):
    """A trapping modified-refractivity profile ducts the field along the surface past the horizon."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.c0 = 3.0e8
        self.freq = 3.0e9  # S-band, 0.1 m wavelength
        self.k0 = 2.0 * np.pi * self.freq / self.c0
        self.nz = 1024
        self.dz = 400.0 / self.nz  # 400 m tall atmosphere column
        self.dr = 20.0
        self.n_range = 3000  # range 60 km
        self.h_d = 150.0  # duct height (m)
        self.z = np.arange(self.nz) * self.dz

    def _run(self, M):
        pe = ParabolicEquation2D(
            self.nz, dz=self.dz, dr=self.dr, k0=self.k0, c0=self.c0, surface="free", absorb=200, absorb_strength=3.0
        )
        psi0 = pe.starter(20.0, width=10.0 / self.k0)  # low antenna
        field = pe.march(psi0, modified_refractivity_index(M), self.n_range)
        return np.abs(pe.pressure(field).detach().numpy()), pe.depths().detach().numpy()

    def test_duct_traps_surface_field_beyond_horizon(self):
        # Standard atmosphere: M rises ~0.118 M-units/m. Surface duct: M falls through a trapping layer to
        # h_d, then rises. The trapped surface field must far exceed the standard-atmosphere field at range.
        M_std = 0.118 * self.z
        M_duct = np.where(self.z < self.h_d, 350.0 - 1.0 * self.z, 350.0 - self.h_d + 0.118 * (self.z - self.h_d))
        p_std, zz = self._run(M_std)
        p_duct, _ = self._run(M_duct)
        surf = zz < self.h_d
        s_std = p_std[-1][surf].mean()
        s_duct = p_duct[-1][surf].mean()
        self.assertGreater(s_duct / s_std, 1.8)  # measured ~2.2 at 60 km


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentiabilityTest(unittest.TestCase):
    """The march is differentiable in the index field, so it drops into inverse.Differential."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_dloss_dsoundspeed_matches_finite_difference(self):
        # A sound-speed anomaly perturbs n = c0/c; d(final energy)/d(anomaly) from autograd must match a
        # central finite difference (the gradient inverse machinery relies on).
        c0, freq, nz, dz, dr, n_range = 1500.0, 200.0, 256, 0.6, 2.0, 100
        pe = ParabolicEquation2D(nz, dz=dz, dr=dr, freq=freq, c0=c0, surface="pressure_release", absorb=40)
        z = pe.depths()

        def loss(dc):
            c = c0 + dc * torch.exp(-(((z - 60.0) / 20.0) ** 2))
            n = c0 / c
            field = pe.march(pe.starter(30.0, width=3.0 / pe.k0), n, n_range)
            return torch.sum(torch.abs(field[-1]) ** 2) * dz

        dc = torch.tensor(10.0, requires_grad=True)
        loss(dc).backward()
        g_auto = float(dc.grad)
        self.assertTrue(np.isfinite(g_auto))

        eps = 1e-3
        g_fd = (float(loss(torch.tensor(10.0 + eps))) - float(loss(torch.tensor(10.0 - eps)))) / (2 * eps)
        self.assertAlmostEqual(g_auto, g_fd, places=6)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ApiTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_freq_and_k0_agree(self):
        a = ParabolicEquation2D(64, dz=1.0, dr=1.0, freq=150.0, c0=1500.0)
        k0 = 2.0 * np.pi * 150.0 / 1500.0
        b = ParabolicEquation2D(64, dz=1.0, dr=1.0, k0=k0)
        self.assertAlmostEqual(a.k0, b.k0, places=12)

    def test_march_shape_and_range_dependent_index(self):
        pe = ParabolicEquation2D(128, dz=0.5, dr=1.0, freq=100.0, c0=1500.0, surface="free", absorb=16)
        psi0 = pe.starter(30.0, width=3.0)
        # a range-dependent index supplied as a (n_range, nz) array
        n_field = np.ones((50, 128))
        field = pe.march(psi0, n_field)
        self.assertEqual(tuple(field.shape), (50, 128))
        tl = pe.transmission_loss(field)
        self.assertEqual(tuple(tl.shape), (50, 128))
        self.assertTrue(np.all(np.isfinite(tl.detach().numpy())))

    def test_modified_refractivity_index(self):
        # n = 1 + M * 1e-6; a 350 M-unit level -> n = 1.00035.
        self.assertAlmostEqual(modified_refractivity_index(350.0), 1.00035, places=8)


if __name__ == "__main__":
    unittest.main()
