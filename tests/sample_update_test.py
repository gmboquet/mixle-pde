"""Sampled-posterior Bayesian updates for domain likelihoods."""

import unittest

import numpy as np

from mixle_pde.field_assimilation import PosteriorFieldSamples4D
from mixle_pde.geo_observations import (
    BiostratConstraint,
    FaciesIntervalConstraint,
    GeochemAssay,
    GeochronologyAge,
    StratigraphicCorrelation,
)
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.sample_update import (
    biostrat_constraint_likelihood,
    facies_interval_likelihood,
    geochem_assay_likelihood,
    geochronology_age_likelihood,
    stratigraphic_correlation_likelihood,
    timed_likelihood,
    update_sampled_field_posterior_4d,
    update_sampled_field_posterior_with_observations,
)


class SampledPosteriorUpdateTest(unittest.TestCase):
    def test_geochem_likelihood_reweights_3d_field_samples(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, 0.0]]),
            spacing=1.0,
            units="ppm",
            property_name="Cu",
        )
        posterior = PosteriorFieldSamples3D(
            grid=grid,
            samples=np.array([[1.0], [4.0], [8.0], [11.5], [12.0], [12.5]]),
        )
        assay = GeochemAssay(
            element="Cu",
            location=np.array([[0.0, 0.0, 0.0]]),
            value=np.array([12.0]),
            noise_std=np.array([0.25]),
            units="ppm",
        )

        updated, report = update_sampled_field_posterior_with_observations(
            posterior,
            [geochem_assay_likelihood(assay, grid)],
            n_samples=64,
            rng=np.random.default_rng(5),
        )

        self.assertEqual(updated.samples.shape, (64, 1))
        self.assertLess(report.effective_sample_size, posterior.samples.shape[0])
        self.assertGreater(float(updated.physical_samples.mean()), 10.0)
        self.assertAlmostEqual(float(updated.map[0]), 12.0)

    def test_biostrat_likelihood_reweights_4d_age_trajectories(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, -100.0]]),
            spacing=1.0,
            units="Ma",
            property_name="age",
        )
        posterior = PosteriorFieldSamples4D(
            grid=grid,
            times=np.array([0.0, 1.0]),
            samples=np.array(
                [
                    [[20.0], [35.0]],
                    [[20.0], [55.0]],
                    [[20.0], [65.0]],
                    [[20.0], [90.0]],
                ]
            ),
        )
        occurrence = BiostratConstraint(
            location=np.array([[0.0, 0.0, -100.0]]),
            taxon="Zone-A",
            present=True,
            first_appearance=60.0,
            last_appearance=50.0,
            tolerance=1.0,
        )

        updated, report = update_sampled_field_posterior_4d(
            posterior,
            [[], [timed_likelihood(biostrat_constraint_likelihood(occurrence, grid), 1.0)]],
            n_samples=32,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(updated.samples.shape, (32, 2, 1))
        self.assertEqual(report.likelihood_count, 1)
        self.assertLess(report.effective_sample_size, posterior.n_samples)
        self.assertTrue(
            np.all((updated.physical_samples[:, 1, 0] >= 50.0) & (updated.physical_samples[:, 1, 0] <= 60.0))
        )

    def test_typed_age_facies_and_stratigraphic_likelihood_factories(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, -10.0], [10.0, 0.0, -20.0]]),
            spacing=10.0,
            units="Ma",
            property_name="age",
        )
        good = np.array([40.0, 55.0])
        bad = np.array([70.0, 20.0])
        age = GeochronologyAge(location=grid.coordinates[[0]], age=40.0, analytical_std=1.0)
        facies = FaciesIntervalConstraint(
            location=grid.coordinates[[1]],
            label="marine",
            property_name="age",
            lower=50.0,
            upper=60.0,
            tolerance=2.0,
        )
        strat = StratigraphicCorrelation(
            location_a=grid.coordinates[[1]],
            location_b=grid.coordinates[[0]],
            age_difference=15.0,
            std=1.0,
        )
        likelihoods = [
            geochronology_age_likelihood(age, grid),
            facies_interval_likelihood(facies, grid),
            stratigraphic_correlation_likelihood(strat, grid),
        ]

        good_score = sum(fn(good) for fn in likelihoods)
        bad_score = sum(fn(bad) for fn in likelihoods)

        self.assertGreater(good_score, bad_score)


if __name__ == "__main__":
    unittest.main()
