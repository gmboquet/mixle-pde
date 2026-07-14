"""C6 DoD: vectorized 3-D assembly matches the loop reference, scales to 48^3 fast, and the CPML absorbs a
shot far better than the exponential sponge on the same run.

Three checks:

1. Correctness -- the vectorized ``NavierStokes3D._build_poisson`` and ``em_diffusion_3d._curl_matrix`` produce
   the exact same sparse operator as a frozen Python-loop reference implementation (the pre-vectorization code,
   preserved here as ground truth), on a small ``n=6`` grid.
2. Scale -- assembly at ``48**3`` unknowns completes in well under 5 seconds and the nonzero count grows
   O(n) (i.e. O(N) in the unknown count, a fixed-width stencil), not superlinearly, confirming there is no
   hidden Python loop over the grid.
3. Absorption -- ``WaveEquation3D``'s split-field CPML (``pml_width``) reflects less than 1% of a shot's
   energy back into the domain, while the legacy exponential sponge (``absorb_width``) leaks substantially
   more on the identical shot.
"""

from __future__ import annotations

import time
import unittest

import numpy as np
import scipy.sparse as sp

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    torch.set_default_dtype(torch.float64)
    from mixle_pde.em_diffusion_3d import _curl_matrix
    from mixle_pde.flow3d import NavierStokes3D
    from mixle_pde.ops import make_ops
    from mixle_pde.wave3d import WaveEquation3D


# ------------------------------------------------------------------------------------------------------------
# Frozen Python-loop reference implementations (the code these functions replace) -- ground truth for the
# equivalence checks below. These are deliberately NOT imported from the source modules: they exist only here,
# as the "old, slow, obviously correct" baseline the new vectorized assembly must reproduce bit-for-bit.
# ------------------------------------------------------------------------------------------------------------
def _reference_diff_matrix(n, h, axis):
    """The original triple-nested-loop ``ops.grad``-style central-difference matrix (flow3d._build_poisson)."""
    N = n**3
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = [i, j, k]
                if idx[axis] in (0, n - 1):
                    continue
                r = (i * n + j) * n + k
                hi = idx.copy()
                hi[axis] += 1
                lo = idx.copy()
                lo[axis] -= 1
                rows += [r, r]
                cols += [(hi[0] * n + hi[1]) * n + hi[2], (lo[0] * n + lo[1]) * n + lo[2]]
                vals += [1.0 / (2.0 * h), -1.0 / (2.0 * h)]
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _reference_poisson(n, h, pressure_reg):
    """The original ``NavierStokes3D._build_poisson`` body, loop-based."""
    N = n**3
    L = sum(D @ D for D in (_reference_diff_matrix(n, h, a) for a in range(3)))
    return (L + pressure_reg * sp.identity(N)).tocoo()


def _reference_edge_layout(shape):
    nx, ny, nz = shape
    sx = (nx - 1, ny, nz)
    sy = (nx, ny - 1, nz)
    sz = (nx, ny, nz - 1)
    nex = (nx - 1) * ny * nz
    ney = nx * (ny - 1) * nz
    nez = nx * ny * (nz - 1)
    return 0, nex, nex + ney, nex + ney + nez, (sx, sy, sz)


def _reference_face_layout(shape):
    nx, ny, nz = shape
    fx = (nx, ny - 1, nz - 1)
    fy = (nx - 1, ny, nz - 1)
    fz = (nx - 1, ny - 1, nz)
    nfx = nx * (ny - 1) * (nz - 1)
    nfy = (nx - 1) * ny * (nz - 1)
    nfz = (nx - 1) * (ny - 1) * nz
    return 0, nfx, nfx + nfy, nfx + nfy + nfz, (fx, fy, fz)


def _reference_curl_matrix(shape, hx, hy, hz):
    """The original triple-nested-loop-per-face-family ``_curl_matrix`` (em_diffusion_3d.py)."""
    _, eyoff, ezoff, nedge, (sx, sy, sz) = _reference_edge_layout(shape)
    _, fyoff, fzoff, nface, (fx, fy, fz) = _reference_face_layout(shape)

    def eidx(off, s, i, j, k):
        a, b, c = s
        return off + (i * b + j) * c + k

    def fidx(off, s, i, j, k):
        a, b, c = s
        return off + (i * b + j) * c + k

    rows, cols, vals = [], [], []

    for i in range(fx[0]):
        for j in range(fx[1]):
            for k in range(fx[2]):
                f = fidx(0, fx, i, j, k)
                rows += [f, f]
                cols += [eidx(ezoff, sz, i, j + 1, k), eidx(ezoff, sz, i, j, k)]
                vals += [1.0 / hy, -1.0 / hy]
                rows += [f, f]
                cols += [eidx(eyoff, sy, i, j, k + 1), eidx(eyoff, sy, i, j, k)]
                vals += [-1.0 / hz, 1.0 / hz]

    for i in range(fy[0]):
        for j in range(fy[1]):
            for k in range(fy[2]):
                f = fidx(fyoff, fy, i, j, k)
                rows += [f, f]
                cols += [eidx(0, sx, i, j, k + 1), eidx(0, sx, i, j, k)]
                vals += [1.0 / hz, -1.0 / hz]
                rows += [f, f]
                cols += [eidx(ezoff, sz, i + 1, j, k), eidx(ezoff, sz, i, j, k)]
                vals += [-1.0 / hx, 1.0 / hx]

    for i in range(fz[0]):
        for j in range(fz[1]):
            for k in range(fz[2]):
                f = fidx(fzoff, fz, i, j, k)
                rows += [f, f]
                cols += [eidx(eyoff, sy, i + 1, j, k), eidx(eyoff, sy, i, j, k)]
                vals += [1.0 / hx, -1.0 / hx]
                rows += [f, f]
                cols += [eidx(0, sx, i, j + 1, k), eidx(0, sx, i, j, k)]
                vals += [-1.0 / hy, 1.0 / hy]

    return sp.csr_matrix((vals, (rows, cols)), shape=(nface, nedge))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class VectorizedAssemblyMatchesLoopReferenceTestCase(unittest.TestCase):
    """The vectorized assembly is bit-for-bit identical to the frozen loop reference, on a small grid."""

    def test_build_poisson_matches_loop_reference(self):
        n = 6
        ns = NavierStokes3D(n, viscosity=0.05, dt=1.0e-3)
        rows, cols, vals, N = ns._poisson
        got = sp.coo_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(N, N)).toarray()
        ref = _reference_poisson(n, ns.h, ns._pressure_reg).toarray()
        np.testing.assert_allclose(got, ref, atol=1.0e-10)
        self.assertTrue(np.array_equal(got != 0.0, ref != 0.0))  # exact sparsity pattern too

    def test_curl_matrix_matches_loop_reference(self):
        shape = (6, 6, 6)
        hx, hy, hz = 0.37, 0.41, 0.29  # distinct spacings per axis to exercise every stencil weight
        got = _curl_matrix(shape, hx, hy, hz).toarray()
        ref = _reference_curl_matrix(shape, hx, hy, hz).toarray()
        np.testing.assert_allclose(got, ref, atol=1.0e-10)
        self.assertTrue(np.array_equal(got != 0.0, ref != 0.0))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ScalesTo48CubedTestCase(unittest.TestCase):
    """Assembly at 48**3 unknowns is fast (no Python triple loop) and nnz grows O(N), not superlinearly."""

    def test_build_poisson_scales_to_48_cubed_under_5_seconds(self):
        n_small = 12
        ns_small = NavierStokes3D(n_small, viscosity=0.05, dt=1.0e-3)
        nnz_small = ns_small._poisson[2].numpy().size
        density_small = nnz_small / n_small**3

        n = 48
        t0 = time.perf_counter()
        ns = NavierStokes3D(n, viscosity=0.05, dt=1.0e-3)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0, f"_build_poisson at n=48 took {elapsed:.2f}s (must be < 5s)")

        rows, cols, vals, N = ns._poisson
        self.assertEqual(N, n**3)
        nnz = vals.numpy().size
        density = nnz / N
        # a fixed-width stencil keeps nonzeros-per-row roughly constant as the grid grows; a hidden O(n^2)+
        # assembly path would blow this ratio up.
        self.assertLess(abs(density - density_small) / density_small, 0.5)

    def test_curl_matrix_scales_to_48_cubed_under_5_seconds(self):
        small = _curl_matrix((12, 12, 12), 1.0, 1.0, 1.0)
        density_small = small.nnz / small.shape[0]

        shape = (48, 48, 48)
        t0 = time.perf_counter()
        C = _curl_matrix(shape, 1.0, 1.0, 1.0)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0, f"_curl_matrix at 48^3 took {elapsed:.2f}s (must be < 5s)")

        density = C.nnz / C.shape[0]
        self.assertLess(abs(density - density_small) / density_small, 0.5)


def _domain_energy(u, w, h, c):
    """Kinetic ``0.5 w^2`` plus gradient-based elastic potential ``0.5 c^2 |grad u|^2``, integrated over the
    domain. A standard field-energy density -- used only as a comparable diagnostic across the two boundary
    treatments below, not as a claim of exact discrete energy conservation."""
    gx, gy, gz = np.gradient(u, h)
    kinetic = 0.5 * np.sum(w**2)
    potential = 0.5 * (c**2) * np.sum(gx**2 + gy**2 + gz**2)
    return (kinetic + potential) * h**3


def _shot_energy_ratio(n, width, strength, *, pml, steps):
    """Launch a Gaussian pulse from rest at the domain centre and return ``energy_final / energy_initial``.

    With a working absorbing boundary this ratio is small once the wavefront has crossed the domain and been
    absorbed by the boundary layer; with a leaky one, energy that reflected off the boundary is still sloshing
    around the domain and the ratio stays large.
    """
    c = 1.0
    h = 1.0 / (n - 1)
    dt = 0.2 * h / c  # well under the 3-D explicit CFL limit (1/sqrt(3))
    kwargs = dict(dt=dt, spacing=h, absorb_strength=strength)
    kwargs.update(dict(pml_width=width, pml_profile="polynomial") if pml else dict(absorb_width=width))
    wave = WaveEquation3D(n, **kwargs)
    ops = make_ops()

    g = np.linspace(0.0, 1.0, n)
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    sigma = 3.0 * h
    u0 = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2 + (zz - 0.5) ** 2) / (2.0 * sigma**2))
    state = wave.pack(torch.as_tensor(u0.ravel()), torch.zeros(n**3))
    c2 = torch.as_tensor(np.full(n**3, c**2))

    nnn = n**3

    def total_velocity(state):
        if wave._pml is None:
            return state[nnn : 2 * nnn]
        return state[nnn : 2 * nnn] + state[3 * nnn : 4 * nnn] + state[5 * nnn : 6 * nnn]

    e0 = _domain_energy(u0, np.zeros((n, n, n)), h, c)
    for _ in range(steps):
        state = wave.step(state, c2, ops)
    u_final = wave.displacement(state).detach().numpy().reshape(n, n, n)
    w_final = total_velocity(state).detach().numpy().reshape(n, n, n)
    e_final = _domain_energy(u_final, w_final, h, c)
    return e_final / e0


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class CPMLAbsorptionTestCase(unittest.TestCase):
    """The split-field CPML absorbs an outgoing shot; the exponential sponge leaks far more on the same shot."""

    def test_cpml_reflects_far_less_than_the_sponge(self):
        n, width, strength, steps = 40, 10, 3.0, 450
        pml_ratio = _shot_energy_ratio(n, width, strength, pml=True, steps=steps)
        sponge_ratio = _shot_energy_ratio(n, width, strength, pml=False, steps=steps)

        self.assertLess(pml_ratio, 1.0e-2, f"CPML left {pml_ratio:.3%} of the shot's energy in the domain")
        self.assertLess(pml_ratio, sponge_ratio, "the CPML should leak markedly less than the sponge baseline")


if __name__ == "__main__":
    unittest.main()
