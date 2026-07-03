"""Poisson-Boltzmann electrostatics: analytical-benchmark tests.

1) Linearized PBE in a homogeneous electrolyte reproduces the screened-Coulomb (Yukawa) potential
   ``phi(r) = q e^{-kappabar r} / (4 pi eps r)`` to a few percent away from the singular source cell.
2) The full nonlinear PBE reduces to the linearized PBE as the charge -> 0 (small-potential limit).
3) The reaction-field energy of a charge in a dielectric sphere matches the Born solvation energy.
"""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mixle_pde.electrostatics import (
        born_solvation_energy,
        linearized_pbe,
        nonlinear_pbe,
        reaction_field_energy,
        yukawa_potential,
    )


def _center_index(nx):
    c = nx // 2
    return (c * nx + c) * nx + c, c


def _radii(nx, c, h):
    xs = (np.arange(nx) - c) * h
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.sqrt(X**2 + Y**2 + Z**2).ravel()


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class YukawaTestCase(unittest.TestCase):
    """Benchmark 1 (tight): homogeneous electrolyte -> screened-Coulomb potential."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_linearized_pbe_matches_yukawa(self):
        nx = 31
        h = 1.0
        eps = 1.0
        kappabar = 0.4  # Debye length 2.5 cells: fast decay so the Dirichlet box is a faithful far field
        q = 1.0
        n = nx**3
        shape = (nx, nx, nx)
        cflat, c = _center_index(nx)

        phi = linearized_pbe(eps, eps * kappabar**2, {cflat: q}, shape, spacing=h).detach().numpy()

        r = _radii(nx, c, h)
        # away from the singular cell (r >= 3 cells), the radial potential should track the Yukawa form.
        # compare each cell against Yukawa at that cell's own radius (no shell-averaging Jensen bias on 1/r).
        for rr in (3.0, 4.0, 5.0, 6.0):
            mask = (np.abs(r - rr) < 0.5 * h) & (r > 0)
            num = phi[mask]
            ana = yukawa_potential(r[mask], q, eps, kappabar)
            per_cell = np.mean(np.abs(num - ana) / ana)  # mean per-cell relative error in this shell
            shell = abs(num.mean() - ana.mean()) / ana.mean()  # shell-averaged relative error
            self.assertLess(per_cell, 0.04, f"r={rr}: mean per-cell Yukawa error {per_cell:.2%}")
            self.assertLess(shell, 0.02, f"r={rr}: shell-averaged Yukawa error {shell:.2%}")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class NonlinearReducesToLinearTestCase(unittest.TestCase):
    """Benchmark 2: the nonlinear PBE (sinh) collapses onto the linear PBE at small potential."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_small_charge_limit(self):
        nx = 25
        h = 1.0
        eps = 1.0
        kappabar2 = eps * 0.4**2
        shape = (nx, nx, nx)
        cflat, _ = _center_index(nx)

        prev_ratio = None
        for q in (0.5, 0.2, 0.05):
            phi_lin = linearized_pbe(eps, kappabar2, {cflat: q}, shape, spacing=h).detach().numpy()
            phi_nl = nonlinear_pbe(eps, kappabar2, {cflat: q}, shape, spacing=h).detach().numpy()
            diff = np.abs(phi_nl - phi_lin).max()
            scale = np.abs(phi_lin).max()
            ratio = diff / scale
            # the two agree ever more tightly as the charge (hence the potential) shrinks
            self.assertLess(ratio, 1e-3, f"q={q}: max|phi_nl - phi_lin|/|phi_lin| = {ratio:.2e}")
            if prev_ratio is not None:
                self.assertLess(ratio, prev_ratio, "the linear/nonlinear gap must shrink with the charge")
            prev_ratio = ratio

        # at the smallest charge the agreement is essentially exact (well below 1e-5)
        self.assertLess(prev_ratio, 1e-5)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class BornSolvationTestCase(unittest.TestCase):
    """Benchmark 3 (looser, staircased sphere): reaction-field energy = Born solvation energy."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_reaction_field_energy_matches_born(self):
        nx = 29
        h = 1.0
        R = 3.5
        eps_in = 1.0
        eps_out = 80.0
        q = 1.0
        shape = (nx, nx, nx)
        cflat, c = _center_index(nx)

        r = _radii(nx, c, h)
        eps_diel = np.where(r <= R, eps_in, eps_out)  # low-dielectric cavity in high-dielectric solvent
        eps_hom = np.full(nx**3, eps_in)  # the vacuum / homogeneous reference medium

        # no salt screening for the Born model (kappabar^2 = 0), so this is pure Poisson
        phi_solv = linearized_pbe(eps_diel, 0.0, {cflat: q}, shape, spacing=h)
        phi_ref = linearized_pbe(eps_hom, 0.0, {cflat: q}, shape, spacing=h)

        dG = float(reaction_field_energy(phi_solv, phi_ref, {cflat: q}))
        dG_ana = born_solvation_energy(q, R, eps_in, eps_out)

        rel = abs(dG - dG_ana) / abs(dG_ana)
        self.assertLess(rel, 0.10, f"reaction-field dG {dG:.5f} vs Born {dG_ana:.5f} (rel {rel:.2%})")
        self.assertLess(dG, 0.0, "solvation of an ion into water must lower the electrostatic energy")


if __name__ == "__main__":
    unittest.main()
