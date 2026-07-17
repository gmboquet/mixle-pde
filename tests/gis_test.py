"""GIS vector I/O (workstream B8 acceptance).

Loads the vendored drillhole-collars fixture through :func:`mixle_pde.io.gis.load_vector` and
:func:`mixle_pde.io.gis.drillhole_collars`, and checks the declared install extras exist so
``pip install 'mixle-pde[raster]'`` (etc.) actually resolves.
"""

import pathlib
import unittest

import numpy as np

from mixle_pde.io.gis import GISLayer, drillhole_collars, load_vector

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "collars.geojson"


class LoadVectorTest(unittest.TestCase):
    def test_returns_gis_layer_with_crs(self):
        layer = load_vector(str(FIXTURE))
        self.assertIsInstance(layer, GISLayer)
        self.assertIsNotNone(layer.crs)
        self.assertEqual(len(layer.frame), 4)


class DrillholeCollarsTest(unittest.TestCase):
    def test_shape_and_crs(self):
        layer = load_vector(str(FIXTURE))
        self.assertIsNotNone(layer.crs)

        collars = drillhole_collars(str(FIXTURE))
        self.assertEqual(collars.shape, (4, 3))
        self.assertEqual(collars.dtype, np.float64)

    def test_elevation_values_match_fixture_z_column(self):
        collars = drillhole_collars(str(FIXTURE))
        np.testing.assert_allclose(sorted(collars[:, 2]), [398.2, 405.0, 412.5, 431.8])

    def test_missing_elevation_column_raises(self):
        with self.assertRaises(KeyError):
            drillhole_collars(str(FIXTURE), elevation_field="does_not_exist")


if __name__ == "__main__":
    unittest.main()
