"""3D/4D simplex mesh helpers."""

import unittest

import numpy as np

from mixle_pde.mesh import (
    box_simplex_mesh,
    delaunay_mesh,
    moving_mesh,
    pipe_radial_deformation,
    space_time_mesh,
)


class SimplexMeshTest(unittest.TestCase):
    def test_box_tetrahedral_mesh_3d_volume_and_boundary(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(2.0, 3.0, 4.0))
        report = mesh.validate()

        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.n_nodes, 8)
        self.assertEqual(mesh.n_simplices, 6)
        self.assertAlmostEqual(mesh.total_measure(), 24.0)
        self.assertEqual(report["n_boundary_facets"], 12)
        self.assertEqual(report["n_boundary_nodes"], 8)
        self.assertTrue(report["positive_measure"])

    def test_box_simplex_mesh_4d_hypervolume(self):
        mesh = box_simplex_mesh((2, 2, 2, 2), lengths=(2.0, 3.0, 4.0, 5.0))
        report = mesh.validate()

        self.assertEqual(mesh.dim, 4)
        self.assertEqual(mesh.n_nodes, 16)
        self.assertEqual(mesh.n_simplices, 24)
        self.assertAlmostEqual(mesh.total_measure(), 120.0)
        self.assertEqual(report["n_boundary_nodes"], 16)
        self.assertTrue(report["positive_measure"])

    def test_extrude_3d_mesh_to_4d_space_time(self):
        spatial = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        mesh = space_time_mesh(spatial, [0.0, 0.25, 1.0])

        self.assertEqual(mesh.dim, 4)
        self.assertEqual(mesh.n_nodes, spatial.n_nodes * 3)
        self.assertEqual(mesh.n_simplices, spatial.n_simplices * (spatial.dim + 1) * 2)
        self.assertAlmostEqual(mesh.total_measure(), 1.0)
        self.assertTrue(mesh.validate()["positive_measure"])

    def test_transformed_deforms_nodes_and_scales_measure(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        moved = mesh.transformed(map_fn=lambda nodes: nodes * np.array([2.0, 1.0, 1.0]))

        np.testing.assert_array_equal(mesh.simplices, moved.simplices)
        self.assertAlmostEqual(mesh.total_measure(), 1.0)
        self.assertAlmostEqual(moved.total_measure(), 2.0)

    def test_moving_mesh_interpolates_and_extrudes_deformed_geometry(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        moving = moving_mesh(
            mesh,
            [0.0, 1.0],
            lambda nodes, t: t * nodes * np.array([1.0, 0.0, 0.0]),
        )
        mid = moving.at_time(0.5)
        space_time = moving.to_space_time_mesh()

        self.assertEqual(moving.dim, 3)
        self.assertEqual(space_time.dim, 4)
        self.assertEqual(space_time.n_simplices, mesh.n_simplices * 4)
        self.assertAlmostEqual(mid.total_measure(), 1.5)
        self.assertTrue(moving.validate()["positive_measure_all_steps"])

    def test_pipe_radial_deformation_scales_cross_section_volume(self):
        mesh = box_simplex_mesh((2, 2, 3), lengths=(1.0, 1.0, 2.0), origin=(-0.5, -0.5, 0.0))
        moving = moving_mesh(
            mesh,
            [0.0, 1.0],
            pipe_radial_deformation(axis="z", radial_strain=lambda t: 0.2 * t),
        )
        ratios = moving.simplex_measure_ratios()
        report = moving.validate()

        self.assertAlmostEqual(moving.measure_series()[1] / moving.measure_series()[0], 1.2**2)
        np.testing.assert_allclose(ratios[1], np.full(mesh.n_simplices, 1.2**2))
        self.assertEqual(report["n_inverted_or_degenerate_relative_to_reference"], 0)

    def test_delaunay_mesh_3d(self):
        rng = np.random.RandomState(0)
        points = rng.uniform(0.0, 1.0, (30, 3))
        mesh = delaunay_mesh(points)

        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.simplices.shape[1], 4)
        self.assertGreater(mesh.n_simplices, 0)
        self.assertGreater(len(mesh.boundary_facets()), 0)
        self.assertTrue(mesh.validate()["positive_measure"])


if __name__ == "__main__":
    unittest.main()
