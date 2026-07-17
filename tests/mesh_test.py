"""3D/4D simplex mesh helpers."""

import unittest

import numpy as np

from mixle_pde.mesh import (
    SimplexMesh,
    box_simplex_mesh,
    delaunay_mesh,
    interpolate_simplex_field,
    moving_mesh,
    pipe_boundary_facets,
    pipe_boundary_nodes,
    pipe_radial_deformation,
    pipe_simplex_mesh,
    refine_simplex_mesh,
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
        self.assertGreater(report["min_quality"], 0.0)
        self.assertEqual(report["n_low_quality"], 0)

    def test_simplex_quality_detects_collapsed_elements(self):
        mesh = SimplexMesh(
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            simplices=np.array([[0, 1, 2, 3]]),
        )
        report = mesh.validate(min_quality=0.1)

        self.assertFalse(report["positive_measure"])
        self.assertEqual(report["min_quality"], 0.0)
        self.assertEqual(report["n_low_quality"], 1)

    def test_box_simplex_mesh_4d_hypervolume(self):
        mesh = box_simplex_mesh((2, 2, 2, 2), lengths=(2.0, 3.0, 4.0, 5.0))
        report = mesh.validate()

        self.assertEqual(mesh.dim, 4)
        self.assertEqual(mesh.n_nodes, 16)
        self.assertEqual(mesh.n_simplices, 24)
        self.assertAlmostEqual(mesh.total_measure(), 120.0)
        self.assertEqual(report["n_boundary_nodes"], 16)
        self.assertTrue(report["positive_measure"])

    def test_refined_3d_mesh_preserves_volume_and_splits_each_tetrahedron(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(2.0, 3.0, 4.0))
        refined = mesh.refined()

        self.assertEqual(refined.dim, 3)
        self.assertEqual(refined.n_nodes, mesh.n_nodes + mesh.n_simplices)
        self.assertEqual(refined.n_simplices, mesh.n_simplices * 4)
        self.assertAlmostEqual(refined.total_measure(), mesh.total_measure())
        self.assertTrue(refined.validate()["positive_measure"])

    def test_refined_4d_mesh_preserves_hypervolume(self):
        mesh = box_simplex_mesh((2, 2, 2, 2), lengths=(1.0, 2.0, 3.0, 4.0))
        refined = refine_simplex_mesh(mesh, levels=1)

        self.assertEqual(refined.dim, 4)
        self.assertEqual(refined.n_nodes, mesh.n_nodes + mesh.n_simplices)
        self.assertEqual(refined.n_simplices, mesh.n_simplices * 5)
        self.assertAlmostEqual(refined.total_measure(), mesh.total_measure())

    def test_selective_refinement_splits_only_marked_simplices(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        mask = np.zeros(mesh.n_simplices, dtype=bool)
        mask[[0, 2]] = True
        refined = mesh.refined(mask=mask)

        self.assertEqual(refined.n_nodes, mesh.n_nodes + int(mask.sum()))
        self.assertEqual(refined.n_simplices, mesh.n_simplices + int(mask.sum()) * mesh.dim)
        self.assertAlmostEqual(refined.total_measure(), mesh.total_measure())

    def test_simplex_interpolates_linear_field_inside_3d_mesh(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        values = mesh.nodes[:, 0] + 2.0 * mesh.nodes[:, 1] + 3.0 * mesh.nodes[:, 2]
        points = np.array([[0.25, 0.25, 0.25], [0.75, 0.5, 0.25]])

        simplex_indices, barycentric = mesh.locate_points(points)
        out = interpolate_simplex_field(mesh, values, points)

        self.assertTrue(np.all(simplex_indices >= 0))
        np.testing.assert_allclose(barycentric.sum(axis=1), np.ones(points.shape[0]))
        np.testing.assert_allclose(out, points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2])

    def test_simplex_interpolation_marks_outside_points(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        values = np.arange(mesh.n_nodes, dtype=float)
        points = np.array([[0.25, 0.25, 0.25], [1.25, 0.25, 0.25]])

        simplex_indices, _ = mesh.locate_points(points)
        out = mesh.interpolate(values, points, fill_value=-99.0)

        self.assertGreaterEqual(simplex_indices[0], 0)
        self.assertEqual(simplex_indices[1], -1)
        self.assertEqual(out[1], -99.0)

    def test_simplex_interpolates_linear_field_inside_4d_mesh(self):
        mesh = box_simplex_mesh((2, 2, 2, 2), lengths=(1.0, 1.0, 1.0, 1.0))
        values = mesh.nodes @ np.array([1.0, 2.0, 3.0, 4.0])
        points = np.array([[0.25, 0.25, 0.25, 0.25], [0.75, 0.5, 0.25, 0.5]])

        out = mesh.interpolate(values, points)

        np.testing.assert_allclose(out, points @ np.array([1.0, 2.0, 3.0, 4.0]))

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

    def test_moving_mesh_quality_series_flags_degenerate_step(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        moving = moving_mesh(
            mesh,
            [0.0, 1.0],
            map_fn=lambda nodes, t: nodes * np.array([1.0 - t, 1.0, 1.0]),
        )
        quality = moving.simplex_quality_series()
        report = moving.validate(min_quality=0.1)

        self.assertEqual(quality.shape, (2, mesh.n_simplices))
        self.assertGreater(moving.min_quality_series()[0], 0.0)
        self.assertEqual(moving.min_quality_series()[1], 0.0)
        self.assertFalse(report["positive_measure_all_steps"])
        self.assertGreater(report["n_low_quality"], 0)

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

    def test_pipe_simplex_mesh_approximates_annular_cylinder_volume(self):
        mesh = pipe_simplex_mesh(inner_radius=0.5, outer_radius=1.0, length=2.0, n_theta=32, n_axial=3)
        expected = np.pi * (1.0**2 - 0.5**2) * 2.0

        self.assertEqual(mesh.dim, 3)
        self.assertEqual(mesh.n_nodes, 2 * 32 * 3)
        self.assertEqual(mesh.n_simplices, 32 * 2 * 6)
        self.assertTrue(mesh.validate()["positive_measure"])
        self.assertLess(abs(mesh.total_measure() - expected) / expected, 0.01)

    def test_pipe_simplex_mesh_deforms_radially(self):
        mesh = pipe_simplex_mesh(inner_radius=0.5, outer_radius=1.0, length=2.0, n_theta=24, n_axial=2)
        moving = moving_mesh(
            mesh,
            [0.0, 1.0],
            pipe_radial_deformation(axis="z", radial_strain=lambda t: 0.2 * t),
        )
        ratios = moving.simplex_measure_ratios()

        self.assertTrue(moving.validate()["positive_measure_all_steps"])
        np.testing.assert_allclose(ratios[1], np.full(mesh.n_simplices, 1.2**2))

    def test_pipe_boundary_groups_classify_nodes_and_facets(self):
        mesh = pipe_simplex_mesh(inner_radius=0.5, outer_radius=1.0, length=2.0, n_theta=16, n_axial=3)
        nodes = pipe_boundary_nodes(mesh, inner_radius=0.5, outer_radius=1.0, length=2.0)
        facets = pipe_boundary_facets(mesh, inner_radius=0.5, outer_radius=1.0, length=2.0)

        self.assertEqual(nodes["inner_wall"].size, 16 * 3)
        self.assertEqual(nodes["outer_wall"].size, 16 * 3)
        self.assertEqual(nodes["inlet"].size, 2 * 16)
        self.assertEqual(nodes["outlet"].size, 2 * 16)
        self.assertGreater(facets["inner_wall"].shape[0], 0)
        self.assertGreater(facets["outer_wall"].shape[0], 0)
        self.assertGreater(facets["inlet"].shape[0], 0)
        self.assertGreater(facets["outlet"].shape[0], 0)

    def test_moving_mesh_transfers_values_between_deformed_domains(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        moving = moving_mesh(
            mesh,
            [0.0, 1.0],
            map_fn=lambda nodes, t: nodes * np.array([1.0 + t, 1.0, 1.0]),
        )
        source = moving.at_time(1.0)
        values = source.nodes[:, 0] + 2.0 * source.nodes[:, 1] + 3.0 * source.nodes[:, 2]

        transferred = moving.interpolate_values(values, 1.0, 0.0)

        np.testing.assert_allclose(transferred, mesh.nodes[:, 0] + 2.0 * mesh.nodes[:, 1] + 3.0 * mesh.nodes[:, 2])

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
