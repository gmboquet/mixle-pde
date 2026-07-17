"""IC-4 `SurveyGeometry` conformance: survey-geometry -> mesh mapping (workstream B7).

A 5x5x5 structured grid stands in for a real DC/ERT survey mesh. Electrode positions come from the
committed fixture (``tests/fixtures/dc_electrodes.csv``); the DC quadrupole schedule and the Yee edge
index are both cross-checked against a manually-computed reference, so this test fails if
:mod:`mixle_pde.geometry_to_mesh` ever drifts from the node/edge numbering the forward operators in
:mod:`mixle_pde.observations` actually use.
"""

import csv
import dataclasses
import os
import unittest

import numpy as np

from mixle_pde.geometry_to_mesh import electrodes_to_schedule, nearest_node_indices, yee_edge_index
from mixle_pde.geophysics import dc_resistivity
from mixle_pde.latent import Field3D
from mixle_pde.observations import Observation, SurveyGeometry

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "dc_electrodes.csv")

NX = NY = NZ = 5


def _node_index(i: int, j: int, k: int, ny: int = NY, nz: int = NZ) -> int:
    """Hand-computed flat node index for a C-order (nx, ny, nz) grid -- the reference this test checks
    `nearest_node_indices`/`electrodes_to_schedule` against."""
    return i * ny * nz + j * nz + k


def _grid_5x5x5(spacing: float = 1.0) -> Field3D:
    pts = np.array(
        [[i * spacing, j * spacing, k * spacing] for i in range(NX) for j in range(NY) for k in range(NZ)],
        dtype=float,
    )
    return Field3D(coordinates=pts, spacing=spacing, units="S/m", property_name="log_sigma")


def _read_electrode_xyz():
    with open(FIXTURE, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)


class NearestNodeIndicesTest(unittest.TestCase):
    def test_matches_hand_computed_flat_index(self):
        grid = _grid_5x5x5()
        query = np.array([[2.0, 2.0, 2.0], [0.1, 0.9, 4.2]])
        idx = nearest_node_indices(query, grid)
        self.assertEqual(idx.tolist(), [_node_index(2, 2, 2), _node_index(0, 1, 4)])


class ElectrodesToScheduleTest(unittest.TestCase):
    def test_reproduces_hand_built_dc_quadrupole_schedule_exactly(self):
        grid = _grid_5x5x5()
        electrode_xyz = _read_electrode_xyz()

        # electrode ids index into the fixture rows: 0=(0,0,0) 1=(4,0,0) 2=(2,0,0) 3=(2,4,0) 4=(0,4,0).
        # Row 0 is a plain Wenner-style quadrupole; row 1 reuses the same m/n but has no "b" pole (-1),
        # exercising the pole-dipole passthrough.
        abmn = np.array([[0, 1, 2, 3], [4, -1, 2, 3]], dtype=int)
        schedule = electrodes_to_schedule(electrode_xyz, abmn, grid)

        expected = [
            [_node_index(0, 0, 0), _node_index(4, 0, 0), _node_index(2, 0, 0), _node_index(2, 4, 0)],
            [_node_index(0, 4, 0), None, _node_index(2, 0, 0), _node_index(2, 4, 0)],
        ]
        self.assertEqual(schedule.shape, (2, 4))
        for row in range(2):
            for col in range(4):
                self.assertEqual(schedule[row, col], expected[row][col])

    def test_schedule_feeds_dc_resistivity_directly_no_hand_built_indices(self):
        import torch

        grid = _grid_5x5x5()
        electrode_xyz = _read_electrode_xyz()
        abmn = np.array([[0, 1, 2, 3], [4, -1, 2, 3]], dtype=int)
        schedule = electrodes_to_schedule(electrode_xyz, abmn, grid)

        log_sigma0 = torch.zeros(grid.n, dtype=torch.float64)
        out = dc_resistivity(log_sigma0, (NX, NY, NZ), schedule)
        self.assertEqual(len(out), 2)
        self.assertTrue(torch.isfinite(out).all())


class YeeEdgeIndexTest(unittest.TestCase):
    def test_mid_grid_x_edge_matches_manual_flat_index(self):
        idx = yee_edge_index(np.array([2.5, 2.0, 2.0]), (NX, NY, NZ), axis=0, spacing=1.0)
        # x-edge block shape (nx-1, ny, nz) = (4, 5, 5), offset 0, C-order (i=2, j=2, k=2)
        expected = 2 * NY * NZ + 2 * NZ + 2
        self.assertEqual(idx, expected)

    def test_y_and_z_edge_offsets_match_manual_flat_index(self):
        n_x_edges = (NX - 1) * NY * NZ
        n_y_edges = NX * (NY - 1) * NZ

        idx_y = yee_edge_index(np.array([2.0, 1.5, 3.0]), (NX, NY, NZ), axis=1, spacing=1.0)
        expected_y = n_x_edges + (2 * (NY - 1) * NZ + 1 * NZ + 3)
        self.assertEqual(idx_y, expected_y)

        idx_z = yee_edge_index(np.array([1.0, 4.0, 0.5]), (NX, NY, NZ), axis=2, spacing=1.0)
        expected_z = n_x_edges + n_y_edges + (1 * NY * (NZ - 1) + 4 * (NZ - 1) + 0)
        self.assertEqual(idx_z, expected_z)


class Ic4ConformanceTest(unittest.TestCase):
    """IC-4 shape conformance for `Observation.crs`/`modality` and the `SurveyGeometry` dataclass.

    Frozen field names/defaults from ``notes/exec/contracts.md`` IC-4; kept alongside the B7 DoD test
    rather than a separate ``test_ic_observation.py`` since this repo's pytest config only collects
    ``*_test.py`` (``pyproject.toml``), not the ``test_*.py`` naming the contract stub used.
    """

    def test_observation_gains_crs_and_modality(self):
        names = {f.name for f in dataclasses.fields(Observation)}
        self.assertTrue({"crs", "modality", "provenance"} <= names)

    def test_new_fields_default_so_existing_callsites_unbroken(self):
        o = Observation(kind="gravity", location=np.zeros((1, 3)), value=np.zeros(1), noise_cov=np.ones(1))
        self.assertIsNone(o.crs)
        self.assertEqual(o.modality, "")

    def test_survey_geometry_shape_and_defaults(self):
        g = SurveyGeometry(points=np.zeros((4, 3)), crs="EPSG:32611")
        self.assertEqual(g.points.shape, (4, 3))
        self.assertIsNone(g.node_index)
        self.assertIsNone(g.edge_index)


class SurveyGeometryResolveTest(unittest.TestCase):
    def test_resolve_fills_node_index_via_nearest_node_indices(self):
        grid = _grid_5x5x5()
        electrode_xyz = _read_electrode_xyz()
        geom = SurveyGeometry(points=electrode_xyz, crs="EPSG:32613")
        self.assertIsNone(geom.node_index)

        resolved = geom.resolve(grid)
        self.assertIsInstance(resolved, SurveyGeometry)
        expected = [
            _node_index(0, 0, 0),
            _node_index(4, 0, 0),
            _node_index(2, 0, 0),
            _node_index(2, 4, 0),
            _node_index(0, 4, 0),
        ]
        self.assertEqual(resolved.node_index.tolist(), expected)
        # resolve() returns a copy; the original geometry is untouched
        self.assertIsNone(geom.node_index)


if __name__ == "__main__":
    unittest.main()
