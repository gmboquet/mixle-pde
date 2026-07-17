"""Geochemistry + paleontology/biostratigraphy observation models (workstream G3 acceptance).

Satisfies G's third acceptance criterion: geochemical and paleontology/biostratigraphy observations
can be ingested with units, provenance, censoring/absence uncertainty, age/stratigraphic position,
likelihood parameters, and posterior-predictive checks.
"""

import unittest

import numpy as np

from mixle_pde.geo_observations import (
    BiostratConstraint,
    FaciesIntervalConstraint,
    FossilAssemblage,
    GeochemAssay,
    GeochronologyAge,
    MultiElementAssay,
    StratigraphicCorrelation,
    additive_log_ratio,
    assay_log_likelihood,
    assay_posterior_predictive,
    biostrat_log_likelihood,
    facies_interval_log_likelihood,
    fossil_assemblage_log_likelihood,
    fossil_assemblage_posterior_predictive,
    geochronology_log_likelihood,
    inverse_additive_log_ratio,
    multi_element_assay_log_likelihood,
    multi_element_assay_posterior_predictive,
    stratigraphic_correlation_log_likelihood,
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


class MultiElementAssayTest(unittest.TestCase):
    def _assay(self, *, censored=None, detection_limit=None, batch_offset=None):
        loc = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        return MultiElementAssay(
            elements=("Cu", "Mo"),
            location=loc,
            value=np.array([[12.0, 1.5], [8.0, 0.8]]),
            noise_cov=np.array([[0.25, 0.04], [0.04, 0.09]]),
            censored=censored,
            detection_limit=detection_limit,
            batch_offset=batch_offset,
            units="ppm",
            provenance={"lab": "ICP-MS", "batch": "B7"},
        )

    def test_covariance_likelihood_is_maximized_at_the_measured_vector(self):
        assay = self._assay()
        best = multi_element_assay_log_likelihood(assay, np.array([[12.0, 1.5], [8.0, 0.8]]))
        worse = multi_element_assay_log_likelihood(assay, np.array([[20.0, 0.2], [2.0, 4.0]]))
        self.assertGreater(best, worse)
        self.assertEqual(assay.noise_cov.shape, (2, 2, 2))

    def test_censored_elements_score_by_detection_limit_probability(self):
        censored = np.array([[False, True], [False, True]])
        detection_limit = np.array([[0.0, 1.0], [0.0, 1.0]])
        assay = self._assay(censored=censored, detection_limit=detection_limit)
        low = multi_element_assay_log_likelihood(assay, np.array([[12.0, 0.1], [8.0, 0.1]]))
        high = multi_element_assay_log_likelihood(assay, np.array([[12.0, 3.0], [8.0, 3.0]]))
        self.assertGreater(low, high)

    def test_batch_offset_is_applied_to_expected_measurement(self):
        assay = MultiElementAssay(
            elements=("Cu", "Mo"),
            location=np.array([[0.0, 0.0, 0.0]]),
            value=np.array([[14.0, 0.5]]),
            noise_cov=np.array([0.25, 0.09]),
            batch_offset=np.array([2.0, -1.0]),
            units="ppm",
        )
        true_field = multi_element_assay_log_likelihood(assay, np.array([[12.0, 1.5]]))
        uncorrected = multi_element_assay_log_likelihood(assay, np.array([[14.0, 0.5]]))
        self.assertGreater(true_field, uncorrected)

    def test_posterior_predictive_accepts_arrays_and_element_mappings(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            spacing=10.0,
            units="ppm",
            property_name="multi-element",
            bounds=None,
        )
        assay = self._assay()
        dense = np.array([[12.0, 1.5], [8.0, 0.8], [4.0, 0.4]])
        np.testing.assert_allclose(multi_element_assay_posterior_predictive(assay, grid, dense), dense[:2])
        mapped = {"Cu": dense[:, 0], "Mo": dense[:, 1]}
        np.testing.assert_allclose(multi_element_assay_posterior_predictive(assay, grid, mapped), dense[:2])

    def test_validation_rejects_bad_covariance_and_missing_detection_limits(self):
        with self.assertRaises(ValueError):
            MultiElementAssay(
                elements=("Cu", "Mo"),
                location=np.array([[0.0, 0.0, 0.0]]),
                value=np.array([[1.0, 2.0]]),
                noise_cov=np.array([[1.0, 2.0], [2.0, 1.0]]),
            )
        with self.assertRaises(ValueError):
            MultiElementAssay(
                elements=("Cu", "Mo"),
                location=np.array([[0.0, 0.0, 0.0]]),
                value=np.array([[1.0, 2.0]]),
                noise_cov=np.array([1.0, 1.0]),
                censored=np.array([[False, True]]),
            )


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


class FossilAssemblageTest(unittest.TestCase):
    def _assemblage(self, **kwargs):
        counts = kwargs.pop("counts", np.array([18, 2, 0]))
        return FossilAssemblage(
            taxa=("A", "B", "C"),
            location=[0.0, 0.0, -100.0],
            counts=counts,
            provenance={"core": "A", "depth_m": 100.0},
            **kwargs,
        )

    def test_assemblage_likelihood_rewards_matching_taxon_probabilities(self):
        obs = self._assemblage()
        matching = fossil_assemblage_log_likelihood(obs, np.array([0.85, 0.10, 0.05]))
        wrong = fossil_assemblage_log_likelihood(obs, np.array([0.05, 0.10, 0.85]))

        self.assertGreater(matching, wrong)
        self.assertEqual(obs.units, "count")
        self.assertEqual(obs.provenance["core"], "A")

    def test_reworking_background_softens_conflicting_local_prediction(self):
        no_rework = self._assemblage(reworking_probability=0.0)
        reworked = self._assemblage(
            reworking_probability=0.4,
            background_probability=np.array([0.8, 0.1, 0.1]),
        )
        local_wrong = np.array([0.05, 0.10, 0.85])

        self.assertGreater(
            fossil_assemblage_log_likelihood(reworked, local_wrong),
            fossil_assemblage_log_likelihood(no_rework, local_wrong),
        )

    def test_detection_probability_reduces_penalty_for_missing_poorly_detected_taxon(self):
        normal_detection = self._assemblage(detection_probability=np.array([1.0, 1.0, 1.0]))
        poor_c_detection = self._assemblage(detection_probability=np.array([1.0, 1.0, 0.05]))
        predicts_c = np.array([0.1, 0.1, 0.8])

        self.assertGreater(
            fossil_assemblage_log_likelihood(poor_c_detection, predicts_c),
            fossil_assemblage_log_likelihood(normal_detection, predicts_c),
        )

    def test_dirichlet_multinomial_overdispersion_is_finite(self):
        obs = self._assemblage(concentration=20.0)
        score = fossil_assemblage_log_likelihood(obs, np.array([0.85, 0.10, 0.05]))

        self.assertTrue(np.isfinite(score))

    def test_assemblage_posterior_predictive_reads_nearest_grid_cell(self):
        grid = Field3D(
            coordinates=np.array([[0.0, 0.0, -100.0], [10.0, 0.0, -100.0]]),
            spacing=10.0,
            units="probability",
            property_name="taxon_probability",
        )
        obs = self._assemblage()
        dense = np.array([[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
        mapped = {"A": dense[:, 0], "B": dense[:, 1], "C": dense[:, 2]}

        np.testing.assert_allclose(fossil_assemblage_posterior_predictive(obs, grid, dense), dense[0])
        np.testing.assert_allclose(fossil_assemblage_posterior_predictive(obs, grid, mapped), dense[0])

    def test_assemblage_validates_counts_and_probabilities(self):
        with self.assertRaises(ValueError):
            self._assemblage(counts=np.array([1, -1, 0]))
        with self.assertRaises(ValueError):
            self._assemblage(detection_probability=np.array([1.0, 0.0, 1.0]))
        with self.assertRaises(ValueError):
            self._assemblage(reworking_probability=1.0)


class GeochronologyAgeTest(unittest.TestCase):
    def test_age_likelihood_is_maximized_at_measured_age(self):
        obs = GeochronologyAge(
            location=[0.0, 0.0, -200.0],
            age=118.0,
            analytical_std=1.5,
            systematic_std=0.5,
            method="U-Pb zircon",
            provenance={"sample": "Z-01"},
        )
        self.assertGreater(geochronology_log_likelihood(obs, 118.0), geochronology_log_likelihood(obs, 125.0))
        self.assertAlmostEqual(obs.total_std, np.hypot(1.5, 0.5))

    def test_age_requires_positive_uncertainty(self):
        with self.assertRaises(ValueError):
            GeochronologyAge(location=[0, 0, 0], age=10.0, analytical_std=0.0)
        with self.assertRaises(ValueError):
            GeochronologyAge(location=[0, 0, 0], age=10.0, analytical_std=1.0, systematic_std=-1.0)


class StratigraphicCorrelationTest(unittest.TestCase):
    def test_relative_age_constraint_scores_correct_difference(self):
        corr = StratigraphicCorrelation(
            location_a=[0.0, 0.0, -120.0],
            location_b=[20.0, 0.0, -95.0],
            age_difference=8.0,
            std=1.0,
            provenance={"horizon": "H2"},
        )
        correct = stratigraphic_correlation_log_likelihood(corr, 118.0, 110.0)
        wrong = stratigraphic_correlation_log_likelihood(corr, 118.0, 95.0)
        self.assertGreater(correct, wrong)

    def test_relative_age_constraint_requires_positive_std(self):
        with self.assertRaises(ValueError):
            StratigraphicCorrelation(location_a=[0, 0, 0], location_b=[1, 0, 0], std=0.0)


class FaciesIntervalConstraintTest(unittest.TestCase):
    def test_present_interval_scores_inside_values_best(self):
        facies = FaciesIntervalConstraint(
            location=[0.0, 0.0, -50.0],
            label="deltaic",
            property_name="facies_score",
            lower=3.0,
            upper=5.0,
            tolerance=0.5,
            provenance={"core": "A"},
        )
        self.assertEqual(facies_interval_log_likelihood(facies, 4.0), 0.0)
        self.assertLess(facies_interval_log_likelihood(facies, 8.0), 0.0)

    def test_absence_interval_penalizes_values_inside_excluded_range(self):
        absence = FaciesIntervalConstraint(
            location=[0.0, 0.0, -50.0],
            label="basinal_absent",
            property_name="facies_score",
            lower=7.0,
            upper=9.0,
            tolerance=0.5,
            present=False,
        )
        outside = facies_interval_log_likelihood(absence, 4.0)
        inside = facies_interval_log_likelihood(absence, 8.0)
        self.assertEqual(outside, 0.0)
        self.assertLess(inside, outside)

    def test_interval_constraint_validates_bounds(self):
        with self.assertRaises(ValueError):
            FaciesIntervalConstraint(location=[0, 0, 0], label="bad", property_name="x", lower=1.0, upper=1.0)


if __name__ == "__main__":
    unittest.main()
