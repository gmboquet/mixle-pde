"""Sampled-posterior Bayesian updates for domain likelihoods."""

import unittest

import numpy as np

from mixle_pde.field_assimilation import PosteriorFieldSamples4D
from mixle_pde.geo_observations import (
    BiostratConstraint,
    GeochemAssay,
    assay_log_likelihood,
    assay_posterior_predictive,
    biostrat_log_likelihood,
)
from mixle_pde.latent import Field3D, PosteriorFieldSamples3D
from mixle_pde.sample_update import update_sampled_field_posterior, update_sampled_field_posterior_4d


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

        def likelihood(field_values):
            predicted = assay_posterior_predictive(assay, grid, field_values)
            return assay_log_likelihood(assay, predicted)

        updated, report = update_sampled_field_posterior(
            posterior,
            [likelihood],
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

        def likelihood(field_values, time):
            if not np.isclose(time, 1.0):
                return 0.0
            return biostrat_log_likelihood(occurrence, float(field_values[0]))

        updated, report = update_sampled_field_posterior_4d(
            posterior,
            [[], [likelihood]],
            n_samples=32,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(updated.samples.shape, (32, 2, 1))
        self.assertEqual(report.likelihood_count, 1)
        self.assertLess(report.effective_sample_size, posterior.n_samples)
        self.assertTrue(np.all((updated.physical_samples[:, 1, 0] >= 50.0) & (updated.physical_samples[:, 1, 0] <= 60.0)))


if __name__ == "__main__":
    unittest.main()
