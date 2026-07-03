"""Tests for the 3-D diffusive curl-curl EM forward (Yee edge grid): skin depth, MT impedance, adjoint grads.

The acceptance bar is agreement with the analytic conductor: a uniform half-space decays the plane wave over the
skin depth ``delta = sqrt(2/(omega mu sigma))`` and returns the MT sounding ``rho_a = 1/sigma`` at a 45-degree
phase. The grid is kept modest (the 3-D complex sparse solve is the cost).
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
    from mixle_pde.em_diffusion_3d import (
        MU0,
        _apply_dirichlet,
        _dirichlet_all_boundary,
        _edge_coords,
        _edge_layout,
        assemble_curl_curl_3d,
        csem_3d,
        mt_3d,
    )
    from mixle_pde.pde_solve import sparse_solve


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class CurlCurl3DTestCase(unittest.TestCase):
    """The 3-D edge curl-curl forward reproduces the half-space skin-depth decay, MT impedance, and is diff'able."""

    def test_skin_depth_decay_rate(self):
        # uniform half-space: the imposed plane wave decays into the conductor as exp(-z/delta); the fitted
        # decay rate of the central x-edge column matches 1/delta.
        sigma = 0.05
        f = 100.0
        omega = 2 * np.pi * f
        delta = np.sqrt(2.0 / (omega * MU0 * sigma))
        h = delta / 8.0
        nx, ny, nz = 8, 8, 48
        shape = (nx, ny, nz)
        log_sigma = torch.log(torch.full((nx * ny * nz,), sigma))

        rows, cols, vals, n = assemble_curl_curl_3d(log_sigma, shape, omega=omega, spacing=h)
        coords, axis = _edge_coords(shape, (h, h, h))
        zc = coords[:, 2]
        k = np.sqrt(1j * omega * MU0 * sigma)  # decaying root (positive real + imag parts)
        decay = np.exp(1j * k * zc)
        boundary = _dirichlet_all_boundary(shape)
        prim = axis == 0  # x-directed primary field

        b = torch.zeros(n, dtype=torch.complex128)
        src = boundary & prim
        b[torch.as_tensor(src)] = torch.as_tensor(decay[src], dtype=torch.complex128)
        r2, c2, v2, n2 = _apply_dirichlet(rows, cols, vals, n, boundary, torch)
        E = sparse_solve(v2, r2, c2, n2, b).detach().numpy()

        _, _, _, _, (sx, _, _) = _edge_layout(shape)
        ic, jc = nx // 2, ny // 2
        base = (ic * sx[1] + jc) * sx[2]
        col = np.abs(E[base : base + sx[2]])
        zcol = np.arange(sx[2]) * h
        mask = (zcol > 0.5 * delta) & (zcol < 3.0 * delta)
        rate = -np.polyfit(zcol[mask], np.log(col[mask]), 1)[0]
        self.assertAlmostEqual(rate * delta, 1.0, delta=0.05)  # numerical decay rate == 1/delta within 5%

    def test_halfspace_mt_impedance(self):
        # laterally uniform half-space: the 3-D MT surface impedance gives rho_a == 1/sigma and phase == 45 deg.
        sigma = 0.05
        f = 100.0
        delta = np.sqrt(2.0 / (2 * np.pi * f * MU0 * sigma))
        h = delta / 6.0
        nx, ny, nz = 8, 8, 48
        shape = (nx, ny, nz)
        log_sigma = torch.log(torch.full((nx * ny * nz,), sigma))
        rho_a, phase, _ = mt_3d(log_sigma, shape, f, spacing=h)
        self.assertAlmostEqual(float(rho_a), 1.0 / sigma, delta=0.03 * (1.0 / sigma))  # within 3%
        self.assertAlmostEqual(float(phase), 45.0, delta=1.0)

    def test_mt_matches_layered_reference(self):
        # cross-check the 3-D half-space impedance against the 1-D Wait recursion (em_diffusion.layered).
        from mixle_pde.em_diffusion import layered_mt_impedance

        sigma = 0.02
        f = 50.0
        delta = np.sqrt(2.0 / (2 * np.pi * f * MU0 * sigma))
        h = delta / 6.0
        nx, ny, nz = 8, 8, 48
        shape = (nx, ny, nz)
        log_sigma = torch.log(torch.full((nx * ny * nz,), sigma))
        rho_a, phase, _ = mt_3d(log_sigma, shape, f, spacing=h)
        rho_ref, phase_ref, _ = layered_mt_impedance([sigma], [], [f])
        self.assertAlmostEqual(float(rho_a), float(rho_ref[0]), delta=0.03 * float(rho_ref[0]))
        self.assertAlmostEqual(float(phase), float(phase_ref[0]), delta=1.0)

    def test_mt_forward_is_differentiable(self):
        # a finite, nonzero adjoint gradient of the apparent resistivity w.r.t. log-conductivity.
        sigma = 0.05
        f = 100.0
        delta = np.sqrt(2.0 / (2 * np.pi * f * MU0 * sigma))
        h = delta / 6.0
        nx, ny, nz = 8, 8, 40
        shape = (nx, ny, nz)
        log_sigma = torch.log(torch.full((nx * ny * nz,), sigma)).requires_grad_(True)
        rho_a, _, _ = mt_3d(log_sigma, shape, f, spacing=h)
        rho_a.backward()
        self.assertIsNotNone(log_sigma.grad)
        self.assertTrue(torch.isfinite(log_sigma.grad).all())
        self.assertGreater(float(log_sigma.grad.abs().sum()), 0.0)

    def test_csem_wholespace_decays_and_is_differentiable(self):
        # grounded-dipole CSEM in a uniform whole space: the field is largest at the source and decays outward,
        # and a scalar of it has a finite nonzero gradient in log-sigma.
        nx, ny, nz = 18, 18, 18
        shape = (nx, ny, nz)
        H = 50.0
        sigma = 0.1
        f = 1.0
        log_sigma = torch.log(torch.full((nx * ny * nz,), sigma)).requires_grad_(True)
        _, _, _, _, (sx, _, _) = _edge_layout(shape)
        ic, jc, kc = nx // 2, ny // 2, nz // 2
        src = [(ic * sx[1] + jc) * sx[2] + kc]  # a central x-edge
        E = csem_3d(log_sigma, shape, f, source_edges=src, spacing=H)

        coords, axis = _edge_coords(shape, (H, H, H))
        d = np.linalg.norm(coords - coords[src[0]], axis=1)
        mag = E.detach().numpy()
        xm = axis == 0
        near = np.abs(mag[xm & (d < 80.0)]).mean()
        far = np.abs(mag[xm & (d > 250.0) & (d < 350.0)]).mean()
        self.assertGreater(near, far)  # field decays away from the galvanic source
        self.assertTrue(torch.isfinite(E).all())

        (E.abs() ** 2).sum().backward()
        self.assertTrue(torch.isfinite(log_sigma.grad).all())
        self.assertGreater(float(log_sigma.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
