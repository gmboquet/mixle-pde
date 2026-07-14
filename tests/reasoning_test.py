"""Cross-modal subsurface reasoning (mixle_pde.reasoning) built on mixle.reason."""

import unittest

import numpy as np

from mixle_pde.reasoning import JointPotentialField, describe_posterior, field_describer


def _mesh():
    # A single subsurface layer of cells at depth h below a co-located surface observation grid.
    # (One layer keeps the potential-field inverse well-posed -- multi-depth gravity/magnetic
    # inversion is famously non-unique; depth resolution is a regularization problem, out of scope
    # for a reasoning-layer demo.)
    nx, ny, h = 6, 6, 100.0
    X, Y = np.meshgrid(np.arange(nx) * h, np.arange(ny) * h, indexing="ij")
    cells = np.stack([X.ravel(), Y.ravel(), np.full(X.size, -h)], axis=1)  # one layer at depth h
    obs = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 1.0)], axis=1)  # just above surface
    return cells, obs, h


class JointPotentialFieldTest(unittest.TestCase):
    def setUp(self):
        self.cells, self.obs, self.h = _mesh()
        self.n = len(self.cells)
        self.model = JointPotentialField(self.cells, volumes=self.h**3, rho_sd=300.0, kappa_sd=0.06, correlation=0.7)
        # plant a correlated anomaly: one central cell dense AND magnetic.
        self.rho_true = np.zeros(self.n)
        self.kappa_true = np.zeros(self.n)
        blk = np.argmin(np.linalg.norm(self.cells - np.array([200.0, 200.0, -100.0]), axis=1))
        self.rho_true[blk] = 600.0
        self.kappa_true[blk] = 0.12
        # synthetic noisy data from the true fields
        Gg = self.model.gravity(self.obs, np.zeros(len(self.obs)), noise_sd=1.0).H
        Gm = self.model.magnetic(self.obs, np.zeros(len(self.obs)), inclination=60.0, declination=0.0, noise_sd=1.0).H
        z_true = np.concatenate([self.rho_true, self.kappa_true])
        rng = np.random.RandomState(0)
        self.g_data = Gg @ z_true + rng.normal(0, 0.02, len(self.obs))
        self.m_data = Gm @ z_true + rng.normal(0, 0.5, len(self.obs))

    def test_shapes_and_prior(self):
        self.assertEqual(self.model.prior.mean().shape, (2 * self.n,))
        self.assertEqual(len(self.model.rho_index), self.n)
        self.assertEqual(len(self.model.kappa_index), self.n)

    def test_fusion_recovers_planted_anomaly(self):
        grav = self.model.gravity(self.obs, self.g_data, noise_sd=0.02)
        mag = self.model.magnetic(self.obs, self.m_data, inclination=60.0, declination=0.0, noise_sd=0.5)
        ans = self.model.reason([grav, mag])
        rho = self.model.density(ans)
        kappa = self.model.susceptibility(ans)
        # posterior means track the truth
        self.assertGreater(np.corrcoef(rho.mean, self.rho_true)[0, 1], 0.5)
        self.assertGreater(np.corrcoef(kappa.mean, self.kappa_true)[0, 1], 0.5)
        # both modalities are credited
        attr = ans.attribution()
        self.assertEqual(set(attr), {"gravity", "magnetic"})
        self.assertTrue(all(v > 0 for v in attr.values()))

    def test_fusion_beats_single_modality(self):
        grav = self.model.gravity(self.obs, self.g_data, noise_sd=0.02)
        mag = self.model.magnetic(self.obs, self.m_data, inclination=60.0, declination=0.0, noise_sd=0.5)
        both = self.model.reason([grav, mag])
        grav_only = self.model.reason([grav])
        # fusing both leaves a tighter joint belief than gravity alone
        self.assertLess(both.entropy(), grav_only.entropy())

    def test_cross_modal_information_transfer(self):
        # With prior correlation > 0, a gravity survey (which physically reads only rho) must still
        # reduce uncertainty about kappa, through the petrophysical prior coupling.
        grav = self.model.gravity(self.obs, self.g_data, noise_sd=0.02)
        prior_kappa_sd = self.model.susceptibility(self.model.reason([])).sd()
        post_kappa_sd = self.model.susceptibility(self.model.reason([grav])).sd()
        self.assertLess(post_kappa_sd.mean(), prior_kappa_sd.mean())

    def test_no_correlation_no_transfer(self):
        # Sanity: with correlation 0, gravity leaves the susceptibility belief untouched.
        model = JointPotentialField(self.cells, volumes=self.h**3, rho_sd=300.0, kappa_sd=0.06, correlation=0.0)
        grav = model.gravity(self.obs, self.g_data, noise_sd=0.02)
        prior_k = model.susceptibility(model.reason([])).sd()
        post_k = model.susceptibility(model.reason([grav])).sd()
        np.testing.assert_allclose(prior_k, post_k, atol=1e-9)

    def test_predict_new_survey_uncertainty_split(self):
        grav = self.model.gravity(self.obs, self.g_data, noise_sd=0.02)
        ans = self.model.reason([grav])
        # predict a fresh gravity reading; its uncertainty splits into epistemic (model) + aleatoric (noise)
        H_new = self.model.gravity(self.obs[:3], np.zeros(3), noise_sd=1.0).H
        dec = ans.predict(H_new, R=0.02**2)
        self.assertEqual(dec.kind, "variance")
        self.assertTrue(np.all(dec.epistemic >= 0))
        np.testing.assert_allclose(dec.total, dec.epistemic + dec.aleatoric, atol=1e-12)


class _ScalarFieldPosterior:
    """A minimal IC-1-conforming posterior: one scalar field over ``d`` grid cells with a controlled
    mean/spread -- everything :func:`describe_posterior` needs (`samples(n, rng)`)."""

    def __init__(self, mean: float, sd: float, d: int = 4):
        self.mean_value = float(mean)
        self.sd = float(sd)
        self.d = d

    def samples(self, n, rng):
        return rng.normal(self.mean_value, self.sd, size=(n, self.d))

    @property
    def mean(self):
        return np.full(self.d, self.mean_value)


class DescribePosteriorTest(unittest.TestCase):
    """E5: wiring an IC-1 posterior onto ``mixle.reason.language_bridge`` through mixle_pde.reasoning."""

    def test_field_describer_returns_a_posterior_describer(self):
        from mixle.reason.language_bridge import PosteriorDescriber

        describer = field_describer("porosity", tol=0.01)
        self.assertIsInstance(describer, PosteriorDescriber)

    def test_sharp_posterior_yields_a_confident_claim(self):
        posterior = _ScalarFieldPosterior(mean=100.0, sd=0.05)
        claim = describe_posterior(posterior, "total", tol=0.5, alpha=0.1, seed=0)
        self.assertIsNotNone(claim)
        self.assertTrue(claim.contains(400.0))  # 4 cells summed, mean ~400

    def test_diffuse_posterior_abstains(self):
        posterior = _ScalarFieldPosterior(mean=100.0, sd=500.0)
        claim = describe_posterior(posterior, "total", tol=0.5, alpha=0.1, seed=0)
        self.assertIsNone(claim)

    def test_unresolvable_field_raises(self):
        posterior = _ScalarFieldPosterior(mean=1.0, sd=0.1)
        with self.assertRaises(ValueError):
            describe_posterior(posterior, "not_a_real_field", tol=0.1, seed=0)

    def test_named_block_field_uses_the_index_property(self):
        # JointPotentialField-style posteriors expose `<field>_index`; describe_posterior should sum
        # just that block, not the whole draw.
        class _Blocked:
            def __init__(self):
                self.a_index = np.array([0, 1])
                self.b_index = np.array([2, 3])

            def samples(self, n, rng):
                draws = np.zeros((n, 4))
                draws[:, :2] = rng.normal(10.0, 0.01, size=(n, 2))  # block "a": sharp, sums to ~20
                draws[:, 2:] = rng.normal(0.0, 1000.0, size=(n, 2))  # block "b": diffuse
                return draws

        posterior = _Blocked()
        claim_a = describe_posterior(posterior, "a", tol=0.5, alpha=0.1, seed=0)
        claim_b = describe_posterior(posterior, "b", tol=0.5, alpha=0.1, seed=0)
        self.assertIsNotNone(claim_a)
        self.assertTrue(claim_a.contains(20.0))
        self.assertIsNone(claim_b)


if __name__ == "__main__":
    unittest.main()
