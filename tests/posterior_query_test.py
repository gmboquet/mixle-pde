"""Posterior extraction and compact storage (workstream G8).

Query marginals at points / sections / regions, exact derived-quantity posteriors (linear functionals),
and compact low-rank / diagonal / ensemble storage of a latent-field posterior.
"""

import unittest

import numpy as np

from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.posterior_query import (
    compress_to_low_rank,
    derived_quantity,
    marginal_at_points,
    region_mass,
    region_summary,
    section,
    to_diagonal,
    to_ensemble,
)


def _grid(n_side=4):
    xs = np.linspace(0.0, 30.0, n_side)
    ys = np.linspace(0.0, 30.0, n_side)
    pts = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    return Field3D(coordinates=pts, spacing=10.0, units="kg/m^3", property_name="rho", bounds=None)


def _dense_posterior(seed=0):
    grid = _grid()
    rng = np.random.default_rng(seed)
    n = grid.n
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n + np.eye(n)  # SPD dense covariance
    mean = rng.normal(size=n) * 5.0
    return grid, PosteriorField3D(grid=grid, mean=mean, cov=cov), cov


class ExtractionTest(unittest.TestCase):
    def test_marginal_at_points_matches_the_posterior(self):
        grid, post, cov = _dense_posterior()
        idx = np.array([0, 5, 10])
        s = marginal_at_points(post, idx, alpha=0.1)
        np.testing.assert_allclose(s.mean, post.mean[idx])  # identity transform -> physical == unconstrained
        np.testing.assert_allclose(s.std, np.sqrt(np.diag(cov))[idx], rtol=1e-9)
        self.assertTrue(np.all(s.upper > s.lower))

    def test_section_selects_a_plane(self):
        grid, post, _ = _dense_posterior()
        sec = section(post, y=0.0)
        # the first row of the grid (y == 0) is 4 cells
        self.assertEqual(int(sec["index"].sum()), 4)

    def test_region_summary_restricts_to_the_mask(self):
        grid, post, _ = _dense_posterior()
        mask = grid.coordinates[:, 0] < 15.0  # west half
        r = region_summary(post, mask)
        self.assertEqual(r["n_cells"], int(mask.sum()))
        self.assertEqual(r["mean"].shape, (int(mask.sum()),))


class DerivedQuantityTest(unittest.TestCase):
    def test_linear_functional_posterior_is_exact(self):
        grid, post, cov = _dense_posterior()
        w = np.random.default_rng(3).normal(size=grid.n)
        dq = derived_quantity(post, w)
        self.assertAlmostEqual(dq.mean, float(w @ post.mean), places=9)
        self.assertAlmostEqual(dq.std, float(np.sqrt(w @ cov @ w)), places=9)
        lo, hi = dq.credible_interval(0.1)
        self.assertLess(lo, dq.mean)
        self.assertGreater(hi, dq.mean)

    def test_region_mass_is_the_volume_weighted_sum_posterior(self):
        grid, post, cov = _dense_posterior()
        mask = grid.coordinates[:, 1] > 15.0
        vols = np.full(grid.n, 1000.0)
        dq = region_mass(post, mask, vols)
        w = np.where(mask, vols, 0.0)
        self.assertAlmostEqual(dq.mean, float(w @ post.mean), places=6)
        self.assertAlmostEqual(dq.std, float(np.sqrt(w @ cov @ w)), places=6)

    def test_derived_quantity_agrees_across_storage_modes(self):
        grid, post, cov = _dense_posterior()
        w = np.ones(grid.n)
        exact = derived_quantity(post, w)
        lowrank = derived_quantity(compress_to_low_rank(post, grid.n), w)  # full rank -> exact
        self.assertAlmostEqual(exact.std, lowrank.std, places=6)


class CompressionTest(unittest.TestCase):
    def test_low_rank_preserves_marginal_variance_exactly(self):
        grid, post, cov = _dense_posterior()
        compressed = compress_to_low_rank(post, rank=3)
        self.assertEqual(compressed.low_rank.shape, (grid.n, 3))
        np.testing.assert_allclose(compressed.marginal_variance, np.diag(cov), rtol=1e-9)
        # storage really is smaller: n*k floats vs n*n
        self.assertLess(compressed.low_rank.size + compressed.diag_var.size, cov.size)

    def test_full_rank_compression_reconstructs_the_covariance(self):
        grid, post, cov = _dense_posterior()
        full = compress_to_low_rank(post, rank=grid.n)
        recon = full.low_rank @ full.low_rank.T + np.diag(full.diag_var)
        np.testing.assert_allclose(recon, cov, atol=1e-8)

    def test_to_diagonal_keeps_marginals(self):
        grid, post, cov = _dense_posterior()
        diag = to_diagonal(post)
        self.assertIsNone(diag.cov)
        np.testing.assert_allclose(diag.marginal_variance, np.diag(cov), rtol=1e-9)

    def test_to_ensemble_draws_samples(self):
        grid, post, _ = _dense_posterior()
        ens = to_ensemble(post, 200, np.random.default_rng(0))
        self.assertEqual(ens.samples.shape, (200, grid.n))
        np.testing.assert_allclose(ens.mean(), post.mean, atol=1.0)  # ensemble mean ~ posterior mean

    def test_rank_bounds_are_validated(self):
        grid, post, _ = _dense_posterior()
        with self.assertRaises(ValueError):
            compress_to_low_rank(post, rank=0)
        with self.assertRaises(ValueError):
            compress_to_low_rank(post, rank=grid.n + 1)


if __name__ == "__main__":
    unittest.main()
