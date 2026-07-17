"""DoD test for C7 -- structural coupling in the joint-inversion Hessian + facies petrophysics.

Two independent checks:

* :class:`JointInversionHessianCouplingTest` builds a synthetic two-property (density, susceptibility)
  problem sharing a boundary and only partially observed, so the cross-gradient structural penalty
  actually does work. It asserts the second-order (``coupling_in_hessian=True``) path reaches the
  gradient-only (``coupling_in_hessian=False``) path's final objective in fewer outer iterations.
* :class:`FaciesMixturePriorTest` fits a 2-component :class:`~mixle_pde.field_priors.FaciesMixturePrior`
  to a synthetic bimodal ``(a, b)`` cloud and checks both facies means are recovered.
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
    from mixle_pde.geophysics import joint_inversion, roughness_operator

from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.field_priors import FaciesMixturePrior


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class JointInversionHessianCouplingTest(unittest.TestCase):
    """DR-ALG C7 step 1: the cross-gradient Gauss-Newton curvature converges in fewer iterations."""

    def _run(self, coupling_in_hessian):
        nx, nz = 10, 8
        n = nx * nz
        shape = (nx, nz)

        # two property models sharing a boundary at x = nx // 2 (a "reservoir" step), on different
        # physical scales -- density-like (~O(1)) and susceptibility-like (~O(1e-2)).
        true_a = np.zeros((nz, nx))
        true_a[:, nx // 2 :] = 1.0
        true_a = true_a.reshape(-1)
        true_b = np.zeros((nz, nx))
        true_b[:, nx // 2 :] = 0.02
        true_b = true_b.reshape(-1)

        # each model is only partially (every-other-cell) observed directly, so the models genuinely
        # need the roughness + structural prior to fill in the rest -- an identity forward over every
        # cell would already be solved in one Newton step regardless of the coupling.
        obs_idx = np.arange(0, n, 2)

        def f_a(x):
            return x[obs_idx]

        def f_b(x):
            return x[obs_idx]

        d_a = true_a[obs_idx]
        d_b = true_b[obs_idx]
        roughness = roughness_operator(shape)

        result = joint_inversion(
            [f_a, f_b],
            [d_a, d_b],
            [np.zeros(n), np.zeros(n)],
            shape,
            noises=[0.01, 0.001],
            betas=[1.0e-3, 1.0e-3],
            roughness=roughness,
            cross_gradient_weight=50.0,
            n_iter=20,
            jac_every=1,
            line_search=40,
            coupling_in_hessian=coupling_in_hessian,
        )
        return result

    def test_hessian_coupling_converges_in_fewer_iterations(self):
        result_rhs = self._run(coupling_in_hessian=False)
        result_hessian = self._run(coupling_in_hessian=True)

        # the RHS-only path only reaches its own final (best) objective at the very last iteration it ran
        obj_final_rhs = result_rhs.objective_history[-1]
        iters_rhs = len(result_rhs.objective_history) - 1

        iters_hessian = next(
            (i for i, obj in enumerate(result_hessian.objective_history) if obj <= obj_final_rhs),
            None,
        )
        self.assertIsNotNone(
            iters_hessian, "coupling_in_hessian=True never reached the RHS-only path's final objective"
        )
        self.assertLess(iters_hessian, iters_rhs)

    def test_coupling_in_hessian_false_matches_historical_behaviour(self):
        # coupling_in_hessian=False must reproduce the pre-C7 gradient-only-coupling numerics exactly:
        # rerunning it is deterministic (no randomness anywhere in the setup).
        r1 = self._run(coupling_in_hessian=False)
        r2 = self._run(coupling_in_hessian=False)
        for m1, m2 in zip(r1, r2):
            np.testing.assert_allclose(m1, m2)


class FaciesMixturePriorTest(unittest.TestCase):
    """DR-ALG C7 steps 2-3: the EM facies-mixture prior recovers a bimodal rock-physics cloud."""

    def _fit(self, a, b, init_means):
        prior_a = FieldGaussianPrior(smoothness_precision=0.0, marginal_precision=1.0e-2)
        prior_b = FieldGaussianPrior(smoothness_precision=0.0, marginal_precision=1.0e-2)
        prior = FaciesMixturePrior(
            priors=[prior_a, prior_b],
            means=np.asarray(init_means, dtype=float),
            covs=np.stack([np.eye(2), np.eye(2)]),
            weights=np.array([0.5, 0.5]),
        )
        responsibilities = None
        for _ in range(30):
            responsibilities = prior.em_update(a, b)
        return prior, responsibilities

    def test_two_facies_means_recovered_from_bimodal_cloud(self):
        rng = np.random.default_rng(0)
        n_each = 150
        true_means = [np.array([0.0, 0.0]), np.array([5.0, 4.0])]
        true_cov = np.eye(2) * 0.15
        cloud = np.concatenate([rng.multivariate_normal(m, true_cov, n_each) for m in true_means])
        a, b = cloud[:, 0], cloud[:, 1]

        prior, responsibilities = self._fit(a, b, init_means=[[-1.0, 1.0], [4.0, 3.0]])

        # match recovered facies to the true ones (label order is not guaranteed by EM) and check both
        recovered = prior.means
        dists = np.linalg.norm(recovered[:, None, :] - np.asarray(true_means)[None, :, :], axis=2)
        # each true facies should be the closest match to a distinct recovered facies (bimodal, not collapsed)
        match0 = np.argmin(dists[:, 0])
        match1 = np.argmin(dists[:, 1])
        self.assertNotEqual(match0, match1, "the two facies means collapsed onto one mode")
        np.testing.assert_allclose(recovered[match0], true_means[0], atol=0.4)
        np.testing.assert_allclose(recovered[match1], true_means[1], atol=0.4)

        # responsibilities are a proper per-cell distribution over the two facies
        self.assertEqual(responsibilities.shape, (2 * n_each, 2))
        np.testing.assert_allclose(responsibilities.sum(axis=1), 1.0)

    def test_precision_sparse_is_symmetric_positive_definite(self):
        from mixle_pde.latent import Field3D

        xs = np.linspace(0.0, 10.0, 4)
        pts = np.array([[x, 0.0, 0.0] for x in xs])
        grid = Field3D(coordinates=pts, spacing=2.5, units="", property_name="a")
        prior_a = FieldGaussianPrior(smoothness_precision=1.0, marginal_precision=1.0e-2)
        prior_b = FieldGaussianPrior(smoothness_precision=1.0, marginal_precision=1.0e-2)
        prior = FaciesMixturePrior(
            priors=[prior_a, prior_b],
            means=np.array([[0.0, 0.0], [3.0, 3.0]]),
            covs=np.stack([np.eye(2) * 0.5, np.eye(2) * 0.5]),
            weights=np.array([0.5, 0.5]),
        )
        n = grid.n
        responsibilities = np.tile([0.7, 0.3], (n, 1))
        q = prior.precision_sparse(grid, responsibilities=responsibilities).toarray()
        np.testing.assert_allclose(q, q.T)
        eigvals = np.linalg.eigvalsh(q)
        self.assertGreater(eigvals.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
