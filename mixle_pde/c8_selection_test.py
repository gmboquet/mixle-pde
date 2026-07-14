"""C8 acceptance: model selection (Laplace evidence) + value-of-information (greedy VOI).

Two checks on a shared synthetic gravity survey over a known density-contrast blob:

1. Two competing hypotheses are inverted against the same synthetic data -- one with a prior whose
   correlation length matches the true structure, one with a badly mismatched (too-short, too-tight)
   prior that cannot represent the blob. ``bayes_factor`` must favor the true structural model.
2. Three candidate borehole sites are scored by ``next_best_observation``; the site nearest the target
   region must beat a distant "random pick", and every reduction must land in ``(0, 1]``.
"""

import unittest

import numpy as np

from mixle_pde.field_inversion import FieldGaussianPrior, linear_gaussian_invert
from mixle_pde.latent import Field3D
from mixle_pde.model_selection import InversionResult, bayes_factor, log_evidence_laplace, rank_hypotheses
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    gravity_forward_operator,
)
from mixle_pde.voi import expected_variance_reduction, next_best_observation


def _grid() -> Field3D:
    xs = np.linspace(0.0, 100.0, 6)
    ys = np.linspace(0.0, 100.0, 6)
    zs = np.array([-30.0, -50.0])
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    return Field3D(coordinates=pts, spacing=20.0, units="kg/m^3", property_name="density_contrast", bounds=None)


def _true_field(grid: Field3D) -> np.ndarray:
    """A compact positive density-contrast blob near the slab centre, zero elsewhere."""
    centre = np.array([50.0, 50.0, -40.0])
    d2 = np.sum((grid.coordinates - centre) ** 2, axis=1)
    return 500.0 * np.exp(-d2 / (2.0 * 15.0**2))


class ModelSelectionTest(unittest.TestCase):
    """``bayes_factor`` must favor the structurally-correct hypothesis over a mismatched one."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)
        self.grid = _grid()
        self.truth = _true_field(self.grid)
        self.volumes = np.full(self.grid.n, 20.0**3)

        self.registry = ForwardOperatorRegistry()
        self.registry.register(gravity_forward_operator(self.grid.coordinates, self.volumes))

        gx, gy = np.meshgrid(np.linspace(0, 100, 6), np.linspace(0, 100, 6))
        grav_loc = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
        G = self.registry.get("gravity").jacobian(self.grid, grav_loc)
        noise_var = 1.0e-5
        clean = G @ self.truth
        value = clean + self.rng.normal(0.0, np.sqrt(noise_var), size=clean.shape)
        self.gravity = Observation(
            kind="gravity", location=grav_loc, value=value, noise_cov=np.full(clean.shape, noise_var)
        )

        # True structural assumption: a smoothness length scale that can represent the compact blob.
        self.true_prior = FieldGaussianPrior(
            mean=0.0, smoothness_precision=5.0e-4, marginal_precision=1.0e-6, length_scale=15.0, neighbors=6
        )
        # Wrong structural assumption: an over-smoothed, over-confident prior with a correlation length
        # several times the true blob radius -- it cannot represent a localized anomaly.
        self.wrong_prior = FieldGaussianPrior(
            mean=0.0, smoothness_precision=1.0e-2, marginal_precision=1.0e-2, length_scale=50.0, neighbors=6
        )

    def _hypothesis(self, name: str, prior: FieldGaussianPrior) -> InversionResult:
        posterior = linear_gaussian_invert(self.grid, [self.gravity], self.registry, prior)
        field_values = posterior.grid.from_unconstrained(posterior.map)
        log_likelihood_at_map = self.registry.total_log_likelihood(self.grid, field_values, [self.gravity])
        log_evidence = log_evidence_laplace(log_likelihood_at_map, prior, posterior)
        return InversionResult(name=name, log_evidence=log_evidence, metadata={"posterior": posterior})

    def test_bayes_factor_favors_the_true_structural_model(self) -> None:
        true_hypothesis = self._hypothesis("true", self.true_prior)
        wrong_hypothesis = self._hypothesis("wrong", self.wrong_prior)

        factor = bayes_factor(true_hypothesis.log_evidence, wrong_hypothesis.log_evidence)
        self.assertGreater(factor, 1.0)

        ranked = rank_hypotheses([wrong_hypothesis, true_hypothesis])
        self.assertEqual(ranked[0][0], 1)  # index 1 (true_hypothesis) ranks first
        self.assertGreater(ranked[0][1], ranked[1][1])


class ValueOfInformationTest(unittest.TestCase):
    """``next_best_observation`` must prefer the candidate closest to the target region."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(11)
        self.grid = _grid()
        self.truth = _true_field(self.grid)
        self.volumes = np.full(self.grid.n, 20.0**3)

        self.registry = ForwardOperatorRegistry()
        self.registry.register(gravity_forward_operator(self.grid.coordinates, self.volumes))
        self.registry.register(borehole_forward_operator())

        gx, gy = np.meshgrid(np.linspace(0, 100, 6), np.linspace(0, 100, 6))
        grav_loc = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
        G = self.registry.get("gravity").jacobian(self.grid, grav_loc)
        noise_var = 1.0e-5
        clean = G @ self.truth
        value = clean + self.rng.normal(0.0, np.sqrt(noise_var), size=clean.shape)
        gravity = Observation(kind="gravity", location=grav_loc, value=value, noise_cov=np.full(clean.shape, noise_var))

        prior = FieldGaussianPrior(
            mean=0.0, smoothness_precision=5.0e-4, marginal_precision=1.0e-6, length_scale=15.0, neighbors=6
        )
        self.posterior = linear_gaussian_invert(self.grid, [gravity], self.registry, prior)

        # Target region: the cells nearest the true blob centre.
        centre = np.array([50.0, 50.0, -40.0])
        d2 = np.sum((self.grid.coordinates - centre) ** 2, axis=1)
        self.region = d2 < 20.0**2
        self.assertTrue(np.any(self.region))

        def _candidate(location) -> Observation:
            return Observation(kind="borehole", location=[location], value=[0.0], noise_cov=[25.0])

        # Candidate 0: right in the target region (most informative).
        # Candidate 1: far corner of the survey (least informative -- the "random pick").
        # Candidate 2: halfway between.
        self.candidates = [
            _candidate([50.0, 50.0, -40.0]),
            _candidate([0.0, 0.0, -30.0]),
            _candidate([25.0, 25.0, -35.0]),
        ]
        self.borehole_op = self.registry.get("borehole")

    def test_next_best_observation_beats_a_random_pick_and_stays_in_range(self) -> None:
        best_index, best_reduction = next_best_observation(
            self.posterior, self.candidates, self.borehole_op, region=self.region, cell_volumes=self.volumes
        )
        self.assertEqual(best_index, 0)

        random_pick_reduction = expected_variance_reduction(
            self.posterior, self.candidates[1], self.borehole_op, region=self.region, cell_volumes=self.volumes
        )
        self.assertGreater(best_reduction, random_pick_reduction)

        for candidate in self.candidates:
            reduction = expected_variance_reduction(
                self.posterior, candidate, self.borehole_op, region=self.region, cell_volumes=self.volumes
            )
            self.assertGreater(reduction, 0.0)
            self.assertLessEqual(reduction, 1.0)


if __name__ == "__main__":
    unittest.main()
