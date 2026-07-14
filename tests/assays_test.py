"""Geochemical assay ingest + QA/QC (workstream B6 acceptance).

Loads a small multi-element assay sheet with a below-detection cell, converts it to a
`geo_observations.MultiElementAssay`, and checks the censoring survives the round trip and that the
assay scores under the shared multi-element likelihood.
"""

import os
import unittest

import numpy as np

from mixle_pde import geo_observations
from mixle_pde.io.assays import load_assays, qaqc_flags, to_multi_element_assay

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "assays.csv")


class LoadAssaysTest(unittest.TestCase):
    def _table(self):
        return load_assays(_FIXTURE, element_cols=["Cu", "Au", "Ag"], unit="ppm")

    def test_shapes_and_units(self):
        table = self._table()
        n, k = 8, 3
        self.assertEqual(table.values.shape, (n, k))
        self.assertEqual(table.censored.shape, (n, k))
        self.assertEqual(table.detection_limit.shape, (n, k))
        self.assertEqual(table.xyz.shape, (n, 3))
        self.assertEqual(list(table.elements), ["Cu", "Au", "Ag"])

    def test_censored_cell_is_flagged_with_detection_limit(self):
        table = self._table()
        i = list(table.sample_id).index("S3")
        j = table.elements.index("Au")
        self.assertTrue(bool(table.censored[i, j]))
        self.assertAlmostEqual(table.detection_limit[i, j], 0.5)
        self.assertAlmostEqual(table.values[i, j], 0.5)
        # every other cell is a detected measurement.
        self.assertEqual(int(np.sum(table.censored)), 1)

    def test_unit_normalization_to_ppm(self):
        table_ppm = self._table()
        table_pct = load_assays(_FIXTURE, element_cols=["Cu", "Au", "Ag"], unit="%")
        np.testing.assert_allclose(table_pct.values, table_ppm.values * 1.0e4)

    def test_to_multi_element_assay_round_trips_censoring(self):
        table = self._table()
        assay = to_multi_element_assay(table)
        self.assertIsInstance(assay, geo_observations.MultiElementAssay)
        np.testing.assert_array_equal(assay.censored, table.censored)
        np.testing.assert_allclose(assay.value, table.values)

        predicted = table.values.copy()
        predicted[~table.censored] *= 0.98  # a plausible near-fit forward prediction.
        log_p = geo_observations.multi_element_assay_log_likelihood(assay, predicted)
        self.assertTrue(np.isfinite(log_p))

    def test_qaqc_flags_catch_the_censored_cell_and_the_outlier(self):
        table = self._table()
        flags = qaqc_flags(table)
        i_censored = list(table.sample_id).index("S3")
        j_au = table.elements.index("Au")
        self.assertTrue(bool(flags["below_detection"][i_censored, j_au]))

        i_outlier = list(table.sample_id).index("S6")
        j_cu = table.elements.index("Cu")
        self.assertTrue(bool(flags["outlier"][i_outlier, j_cu]))
        self.assertFalse(np.any(flags["duplicate_sample_id"]))
        self.assertFalse(np.any(flags["negative_or_zero"]))


if __name__ == "__main__":
    unittest.main()
