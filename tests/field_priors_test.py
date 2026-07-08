"""Depth weighting + cross-property coupling priors (workstream G5).

Depth weighting removes the shallow-smearing bias of potential-field inversion; cross-property coupling
lets observations of one property recover another through a petrophysical relation.
"""

import unittest

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, _stable_nearest_neighbor_rows
from mixle_pde.field_priors import (
    CrossPropertyPrior,
    depth_weighted_marginal_precision,
    depth_weighted_marginal_precision_sparse,
    depth_weights,
    joint_linear_gaussian_invert,
)
from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    gravity_forward_operator,
)


def _grid(name="rho"):
    xs = np.linspace(0.0, 100.0, 5)
    ys = np.linspace(0.0, 100.0, 5)
    zs = np.array([-20.0, -60.0])  # a shallow and a deep layer
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    return Field3D(coordinates=pts, spacing=25.0, units="", property_name=name, bounds=None)


def _blob(grid, amp, depth=-60.0):
    d2 = np.sum((grid.coordinates - np.array([50.0, 50.0, depth])) ** 2, axis=1)
    return amp * np.exp(-d2 / (2.0 * 30.0**2))


class DepthWeightingTest(unittest.TestCase):
    def test_depth_weights_decay_with_depth(self):
        grid = _grid()
        w = depth_weights(grid, beta=3.0, z0=10.0)
        shallow = grid.coordinates[:, 2] == -20.0
        deep = grid.coordinates[:, 2] == -60.0
        self.assertGreater(w[shallow].mean(), w[deep].mean())  # shallow cells weighted harder to prior

    def test_depth_weighted_precision_frees_deep_cells(self):
        grid = _grid()
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1e-3, marginal_precision=1e-2, length_scale=25.0)
        base = prior.precision(grid)
        weighted = depth_weighted_marginal_precision(prior, grid, beta=3.0, z0=10.0)
        deep = grid.coordinates[:, 2] == -60.0
        # deep cells get a SMALLER diagonal precision under depth weighting (freer to carry anomaly)
        self.assertLess(np.diag(weighted)[deep].mean(), np.diag(base)[deep].mean())

    def test_invalid_params_rejected(self):
        grid = _grid()
        with self.assertRaises(ValueError):
            depth_weights(grid, z0=0.0)
        with self.assertRaises(ValueError):
            depth_weights(grid, beta=-1.0)


class SparsePrecisionAssemblyTest(unittest.TestCase):
    def test_nearest_neighbor_ties_are_index_stable(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        rows = _stable_nearest_neighbor_rows(coords, neighbors=2)
        self.assertEqual(rows[0], [1, 2])

    def test_sparse_prior_precision_matches_dense_reference(self):
        grid = _grid()
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1e-3, marginal_precision=1e-2, length_scale=25.0)
        sparse = prior.precision_sparse(grid)
        dense = prior.precision(grid)
        self.assertEqual(sparse.shape, dense.shape)
        self.assertLess(sparse.nnz, dense.size)
        np.testing.assert_allclose(sparse.toarray(), dense)

    def test_sparse_depth_weighted_precision_matches_dense_reference(self):
        grid = _grid()
        prior = FieldGaussianPrior(mean=0.0, smoothness_precision=1e-3, marginal_precision=1e-2, length_scale=25.0)
        sparse = depth_weighted_marginal_precision_sparse(prior, grid, beta=3.0, z0=10.0)
        dense = depth_weighted_marginal_precision(prior, grid, beta=3.0, z0=10.0)
        self.assertLess(sparse.nnz, dense.size)
        np.testing.assert_allclose(sparse.toarray(), dense)

    def test_sparse_cross_property_precision_matches_dense_reference(self):
        grid = _grid()
        prior = CrossPropertyPrior(
            prior_a=FieldGaussianPrior(mean=0.0, smoothness_precision=2e-3, marginal_precision=1e-4, length_scale=25.0),
            prior_b=FieldGaussianPrior(mean=0.0, smoothness_precision=3e-3, marginal_precision=2e-4, length_scale=30.0),
            coupling=0.7,
            slope=1.5,
        )
        sparse = prior.precision_sparse(grid)
        dense = prior.precision(grid)
        self.assertEqual(sparse.shape, dense.shape)
        self.assertLess(sparse.nnz, dense.size)
        np.testing.assert_allclose(sparse.toarray(), dense)


class CrossPropertyCouplingTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.grid_a = _grid("density")
        self.grid_b = _grid("susceptibility")
        self.slope = 2.0
        self.truth_a = _blob(self.grid_a, 300.0)
        self.truth_b = self.slope * self.truth_a  # petrophysically coupled truth
        vols = np.full(self.grid_a.n, 25.0**3)

        self.reg_a = ForwardOperatorRegistry()
        self.reg_a.register(gravity_forward_operator(self.grid_a.coordinates, vols))
        self.reg_a.register(borehole_forward_operator())
        self.reg_b = ForwardOperatorRegistry()
        self.reg_b.register(borehole_forward_operator())

        # property A: rich borehole data
        idx = self.rng.choice(self.grid_a.n, size=int(0.7 * self.grid_a.n), replace=False)
        self.obs_a = [
            Observation(
                kind="borehole",
                location=self.grid_a.coordinates[idx],
                value=self.truth_a[idx] + self.rng.normal(0, 3.0, size=idx.shape),
                noise_cov=np.full(idx.shape, 9.0),
            )
        ]

    def _priors(self, coupling):
        pa = FieldGaussianPrior(mean=0.0, smoothness_precision=2e-3, marginal_precision=1e-4, length_scale=25.0)
        pb = FieldGaussianPrior(mean=0.0, smoothness_precision=2e-3, marginal_precision=1e-4, length_scale=25.0)
        return CrossPropertyPrior(prior_a=pa, prior_b=pb, coupling=coupling, slope=self.slope)

    def test_coupling_recovers_property_b_with_no_direct_data(self):
        # property B has NO observations at all -- only the coupling to A can inform it.
        prior = self._priors(coupling=1.0)
        post_a, post_b = joint_linear_gaussian_invert(
            self.grid_a, self.grid_b, prior, self.obs_a, self.reg_a, [], self.reg_b
        )
        # A is recovered from its own data
        self.assertGreater(np.corrcoef(post_a.mean, self.truth_a)[0, 1], 0.9)
        # B is recovered THROUGH the coupling, and tracks slope * truth_a
        self.assertGreater(np.corrcoef(post_b.mean, self.truth_b)[0, 1], 0.9)
        # the recovered B is ~ slope * recovered A (the petrophysical relation the prior encodes)
        ratio = np.median(post_b.mean[self.truth_a > 50] / post_a.mean[self.truth_a > 50])
        self.assertAlmostEqual(ratio, self.slope, delta=0.5)

    def test_zero_coupling_leaves_b_at_the_prior(self):
        prior = self._priors(coupling=0.0)
        _, post_b = joint_linear_gaussian_invert(
            self.grid_a, self.grid_b, prior, self.obs_a, self.reg_a, [], self.reg_b
        )
        # with no coupling and no data, B stays at its zero prior mean
        self.assertLess(np.max(np.abs(post_b.mean)), 1.0)

    def test_coupling_beats_independent_for_b(self):
        coupled_prior = self._priors(coupling=1.0)
        indep_prior = self._priors(coupling=0.0)
        _, b_coupled = joint_linear_gaussian_invert(
            self.grid_a, self.grid_b, coupled_prior, self.obs_a, self.reg_a, [], self.reg_b
        )
        _, b_indep = joint_linear_gaussian_invert(
            self.grid_a, self.grid_b, indep_prior, self.obs_a, self.reg_a, [], self.reg_b
        )
        err_coupled = np.linalg.norm(b_coupled.mean - self.truth_b)
        err_indep = np.linalg.norm(b_indep.mean - self.truth_b)
        self.assertLess(err_coupled, err_indep)

    def test_bounded_field_rejected(self):
        bounded = Field3D(
            coordinates=self.grid_b.coordinates, spacing=25.0, units="", property_name="s", bounds=(0.0, 1.0)
        )
        with self.assertRaises(ValueError):
            joint_linear_gaussian_invert(
                self.grid_a, bounded, self._priors(1.0), self.obs_a, self.reg_a, [], self.reg_b
            )


if __name__ == "__main__":
    unittest.main()
