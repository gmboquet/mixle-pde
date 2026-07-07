"""Geochemistry + paleontology/biostratigraphy observation models (workstream G3 acceptance).

Satisfies G's third acceptance criterion: geochemical and paleontology/biostratigraphy observations
can be ingested with units, provenance, censoring/absence uncertainty, age/stratigraphic position,
likelihood parameters, and posterior-predictive checks.
"""

import unittest

import numpy as np

from mixle_pde.geo_observations import (
    BiostratConstraint,
    GeochemAssay,
    additive_log_ratio,
    assay_log_likelihood,
    assay_posterior_predictive,
    biostrat_log_likelihood,
    inverse_additive_log_ratio,
)
from mixle_pde.latent import Field3D


class GeochemAssayTest(unittest.TestCase):
    def _assay(self, value, noise=0.5, censored=None, dl=None):
        loc = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        return GeochemAssay(
            element="Cu",
            location=loc,
            value=np.asarray(value, float),
            noise_std=np.full(2, noise),
            censored=censored,
            detection_limit=dl,
            units="ppm",
            provenance={"lab": "XRF-1", "batch": 7},
        )

    def test_detected_likelihood_is_maximized_at_the_measured_value(self):
        assay = self._assay([12.0, 8.0])
        best = assay_log_likelihood(assay, np.array([12.0, 8.0]))
        worse = assay_log_likelihood(assay, np.array([20.0, 2.0]))
        self.assertGreater(best, worse)

    def test_below_detection_limit_rewards_low_predictions(self):
        # both points censored at DL=5; a model predicting low (truly below limit) should score higher
        # than one predicting high (near/above the limit).
        assay = self._assay([5.0, 5.0], censored=np.array([True, True]), dl=np.array([5.0, 5.0]))
        low_pred = assay_log_likelihood(assay, np.array([1.0, 1.0]))
        high_pred = assay_log_likelihood(assay, np.array([9.0, 9.0]))
        self.assertGreater(low_pred, high_pred)

    def test_censoring_requires_a_detection_limit(self):
        with self.assertRaises(ValueError):
            self._assay([5.0, 5.0], censored=np.array([True, False]), dl=None)

    def test_provenance_and_units_are_retained(self):
        assay = self._assay([12.0, 8.0])
        self.assertEqual(assay.units, "ppm")
        self.assertEqual(assay.provenance["lab"], "XRF-1")

    def test_posterior_predictive_reads_the_nearest_grid_cell(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            spacing=10.0,
            units="ppm",
            property_name="Cu",
            bounds=None,
        )
        field_values = np.array([3.0, 7.0, 11.0])
        assay = self._assay([3.0, 7.0])
        pred = assay_posterior_predictive(assay, grid, field_values)
        np.testing.assert_allclose(pred, [3.0, 7.0])


class CompositionalTransformTest(unittest.TestCase):
    def test_alr_round_trips_a_composition(self):
        comp = np.array([0.2, 0.5, 0.3])  # sums to 1
        alr = additive_log_ratio(comp, denom_index=-1)
        self.assertEqual(alr.shape, (2,))
        back = inverse_additive_log_ratio(alr, total=1.0)
        np.testing.assert_allclose(back, comp, atol=1e-9)

    def test_alr_is_unconstrained_real_valued(self):
        comp = np.array([0.01, 0.98, 0.01])
        alr = additive_log_ratio(comp)
        self.assertTrue(np.all(np.isfinite(alr)))


class BiostratConstraintTest(unittest.TestCase):
    def test_occurrence_scores_zero_inside_the_range_zone_and_decays_outside(self):
        c = BiostratConstraint(
            location=[0.0, 0.0, -100.0],
            taxon="Globigerina",
            present=True,
            first_appearance=60.0,  # older bound (Ma)
            last_appearance=50.0,  # younger bound
            tolerance=2.0,
            provenance={"core": "A", "depth_m": 100.0},
        )
        self.assertEqual(biostrat_log_likelihood(c, 55.0), 0.0)  # inside the zone
        outside = biostrat_log_likelihood(c, 65.0)  # older than first appearance
        self.assertLess(outside, 0.0)
        self.assertLess(biostrat_log_likelihood(c, 70.0), outside)  # further out, less likely

    def test_absence_is_a_one_sided_soft_bound(self):
        c = BiostratConstraint(
            location=[0.0, 0.0, -100.0], taxon="Nannofossil-X", present=False, absence_bound=40.0, tolerance=2.0
        )
        young = biostrat_log_likelihood(c, 30.0)  # comfortably younger than the bound -> consistent
        old = biostrat_log_likelihood(c, 50.0)  # older than the bound -> unlikely under absence
        self.assertGreater(young, old)

    def test_occurrence_requires_range_and_absence_requires_bound(self):
        with self.assertRaises(ValueError):
            BiostratConstraint(location=[0, 0, 0], taxon="T", present=True)
        with self.assertRaises(ValueError):
            BiostratConstraint(location=[0, 0, 0], taxon="T", present=False)


if __name__ == "__main__":
    unittest.main()
