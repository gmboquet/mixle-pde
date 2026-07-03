"""Frequency-domain Helmholtz with a perfectly-matched layer and a complex (attenuating) modulus.

Verifies against the analytical solutions:
  1) the free-space 2-D Green's function ``G = (i/4) H_0^{(1)}(k r)`` in the interior, with the PML absorbing
     outgoing waves (small near-boundary residual vs a reflecting Dirichlet box);
  2) the viscoacoustic amplitude decay ``exp(-omega r / (2 Q c))`` for a finite quality factor ``Q``.
"""

import unittest

import numpy as np
import scipy.special as ss
import torch

from mixle_pde.helmholtz_pml import helmholtz_pml_operator, solve_helmholtz_pml


def _radial(shape, h):
    nx, nz = shape
    xs = (np.arange(nx) - nx // 2) * h
    zs = (np.arange(nz) - nz // 2) * h
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return X, Z, np.sqrt(X**2 + Z**2)


class HelmholtzPMLTest(unittest.TestCase):
    def test_greens_function_interior(self):
        # homogeneous medium c = 1 (m = 1/c^2 = 1); wavelength ~ 12.6 nodes
        c, nx, nz, h, omega = 1.0, 141, 141, 1.0, 0.5
        k = omega / c
        m = np.ones(nx * nz)
        src = (nx // 2) * nz + nz // 2
        u = solve_helmholtz_pml(m, (nx, nz), src, omega=omega, spacing=h, pml_width=25, pml_strength=50.0)
        U = u.detach().numpy().reshape(nx, nz)
        self.assertTrue(np.all(np.isfinite(U)))

        _, _, R = _radial((nx, nz), h)
        G = (1j / 4.0) * ss.hankel1(0, k * np.maximum(R, 1e-9))
        # interior annulus away from the source and the PML; best-fit complex scale (source amplitude convention)
        mask = (R > 10) & (R < 38)
        a = np.vdot(G[mask], U[mask]) / np.vdot(G[mask], G[mask])
        rel = np.abs(U[mask] - a * G[mask]) / np.abs(a * G[mask])
        self.assertLess(np.median(rel), 0.08)  # matches the analytic Hankel Green's function to <8%
        # the discrete point source reproduces the continuum amplitude: the scale is close to 1
        self.assertLess(abs(a - 1.0), 0.15)

    def test_pml_absorbs_vs_dirichlet_box(self):
        # PML (absorbing) vs a hard Dirichlet box (pml_width=0): the box reflects, the PML does not.
        c, nx, nz, h, omega = 1.0, 141, 141, 1.0, 0.5
        k = omega / c
        m = np.ones(nx * nz)
        src = (nx // 2) * nz + nz // 2
        u_pml = solve_helmholtz_pml(m, (nx, nz), src, omega=omega, spacing=h, pml_width=25, pml_strength=50.0)
        u_box = solve_helmholtz_pml(m, (nx, nz), src, omega=omega, spacing=h, pml_width=0, pml_strength=0.0)
        U_pml = u_pml.detach().numpy().reshape(nx, nz)
        U_box = u_box.detach().numpy().reshape(nx, nz)

        _, _, R = _radial((nx, nz), h)
        G = (1j / 4.0) * ss.hankel1(0, k * np.maximum(R, 1e-9))
        inner = (R > 10) & (R < 38)
        a = np.vdot(G[inner], U_pml[inner]) / np.vdot(G[inner], G[inner])
        # residual against the *outgoing* Green's function in the physical interior (inside the PML, whose
        # inner edge here is r ~ 45). A hard box fills the whole interior with standing-wave reflections;
        # the PML leaves a clean outgoing field.
        ring = (R > 20) & (R < 40)
        res_pml = np.median(np.abs(U_pml[ring] - a * G[ring]) / np.abs(a * G[ring]))
        res_box = np.median(np.abs(U_box[ring] - a * G[ring]) / np.abs(a * G[ring]))
        self.assertLess(res_pml, 0.10)  # PML interior is a clean outgoing wave
        self.assertGreater(res_box, 0.5)  # the Dirichlet box is corrupted by reflections
        self.assertGreater(res_box / res_pml, 10.0)  # PML reflection is >10x smaller

    def test_attenuation_decay_rate(self):
        # finite Q -> amplitude decays as exp(-omega r / (2 Q c)) along the propagation direction
        c, nx, nz, h, omega, Q = 1.0, 201, 201, 1.0, 0.6, 20.0
        m = np.ones(nx * nz)
        src = (nx // 2) * nz + nz // 2
        u = solve_helmholtz_pml(m, (nx, nz), src, omega=omega, spacing=h, pml_width=30, pml_strength=60.0, Q=Q)
        U = u.detach().numpy().reshape(nx, nz)
        i0, k0 = nx // 2, nz // 2
        r = np.arange(15, 70).astype(float) * h
        amp = np.abs(U[i0 + r.astype(int), k0])
        # remove the 2-D geometric 1/sqrt(r) spreading; the residual slope is the attenuation rate
        y = np.log(amp * np.sqrt(r))
        A = np.vstack([np.ones_like(r), r]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rate = -coef[1]
        expected = omega / (2.0 * Q * c)
        self.assertLess(abs(rate - expected) / expected, 0.10)  # decay rate matches analytic to <10%

    def test_operator_complex_and_differentiable(self):
        # the assembled vals are complex, and the field is differentiable in the modulus m
        nx, nz, omega = 41, 41, 0.5
        m = torch.ones(nx * nz, dtype=torch.float64, requires_grad=True)
        rows, cols, vals, n = helmholtz_pml_operator(
            m, (nx, nz), omega=omega, spacing=1.0, pml_width=8, pml_strength=30.0
        )
        self.assertEqual(n, nx * nz)
        self.assertTrue(vals.is_complex())
        self.assertGreater(float(vals.imag.abs().sum().detach()), 0.0)  # PML makes the operator genuinely complex
        u = solve_helmholtz_pml(
            m, (nx, nz), (nx // 2) * nz + nz // 2, omega=omega, spacing=1.0, pml_width=8, pml_strength=30.0
        )
        (u.abs() ** 2).sum().backward()
        self.assertTrue(torch.isfinite(m.grad).all())
        self.assertGreater(float(m.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
