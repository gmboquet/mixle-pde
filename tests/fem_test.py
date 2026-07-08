"""Unstructured P1 finite-element Poisson solver (Phase 5)."""

import unittest

import numpy as np
from scipy.spatial import Delaunay

from mixle_pde.fem import (
    assemble_simplex_fem_matrices,
    assemble_simplex_load_vector,
    assemble_simplex_mass_matrix,
    assemble_simplex_stiffness_matrix,
    boundary_nodes,
    fem_poisson,
    solve_simplex_poisson,
)
from mixle_pde.mesh import box_simplex_mesh


def _mesh_square(n):
    xs = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            tris.append([a, a + 1, a + n + 1])
            tris.append([a, a + n + 1, a + n])
    return nodes, np.array(tris)


class FEMPoissonTest(unittest.TestCase):
    def test_manufactured_solution_converges(self):
        errs = []
        for n in (11, 21, 41):
            nodes, tris = _mesh_square(n)
            u_true = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
            u = fem_poisson(nodes, tris, 2 * np.pi**2 * u_true)  # boundary pinned to 0 (= u_true there)
            errs.append(np.max(np.abs(u - u_true)))
        self.assertLess(errs[-1], 0.005)
        self.assertGreater(errs[0] / errs[1], 2.5)  # ~O(h^2): error drops ~4x per halving
        self.assertGreater(errs[1] / errs[2], 2.5)

    def test_unstructured_delaunay_mesh(self):
        rng = np.random.RandomState(1)
        pts = np.vstack([rng.uniform(0, 1, (400, 2)), [[0, 0], [1, 0], [0, 1], [1, 1]]])
        tri = Delaunay(pts)
        u_true = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        bnd = boundary_nodes(tri.simplices)
        u = fem_poisson(pts, tri.simplices, 2 * np.pi**2 * u_true, dirichlet={int(i): float(u_true[i]) for i in bnd})
        interior = np.setdiff1d(np.arange(len(pts)), bnd)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertLess(np.max(np.abs(u[interior] - u_true[interior])), 0.05)

    def test_boundary_detection(self):
        nodes, tris = _mesh_square(11)
        bnd = set(boundary_nodes(tris))
        expected = set(np.where((nodes[:, 0] == 0) | (nodes[:, 0] == 1) | (nodes[:, 1] == 0) | (nodes[:, 1] == 1))[0])
        self.assertEqual(bnd, expected)

    def test_heterogeneous_conductivity_runs(self):
        nodes, tris = _mesh_square(15)
        u = fem_poisson(nodes, tris, 1.0, conductivity=np.linspace(0.5, 2.0, len(tris)))
        self.assertTrue(np.all(np.isfinite(u)))

    def test_simplex_mass_matrix_conserves_3d_measure(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(2.0, 3.0, 4.0))
        mass = assemble_simplex_mass_matrix(mesh)

        self.assertEqual(mass.shape, (mesh.n_nodes, mesh.n_nodes))
        self.assertAlmostEqual(float(mass.sum()), mesh.total_measure())
        np.testing.assert_allclose((mass - mass.T).toarray(), np.zeros(mass.shape))

    def test_simplex_stiffness_annihilates_4d_constants(self):
        mesh = box_simplex_mesh((2, 2, 2, 2), lengths=(1.0, 2.0, 3.0, 4.0))
        stiffness, mass = assemble_simplex_fem_matrices(mesh)

        self.assertEqual(stiffness.shape, (mesh.n_nodes, mesh.n_nodes))
        self.assertAlmostEqual(float(mass.sum()), mesh.total_measure())
        np.testing.assert_allclose(stiffness @ np.ones(mesh.n_nodes), np.zeros(mesh.n_nodes), atol=1.0e-12)

    def test_simplex_stiffness_exact_linear_energy_3d(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(2.0, 3.0, 4.0))
        stiffness = assemble_simplex_stiffness_matrix(mesh)
        linear_x = mesh.nodes[:, 0]

        energy = float(linear_x @ (stiffness @ linear_x))

        self.assertAlmostEqual(energy, mesh.total_measure())

    def test_simplex_load_vector_integrates_scalar_source(self):
        mesh = box_simplex_mesh((2, 2, 2, 2), lengths=(1.0, 2.0, 3.0, 4.0))
        load = assemble_simplex_load_vector(mesh, 2.0)

        self.assertEqual(load.shape, (mesh.n_nodes,))
        self.assertAlmostEqual(float(load.sum()), 2.0 * mesh.total_measure())

    def test_simplex_poisson_solves_3d_zero_boundary_problem(self):
        mesh = box_simplex_mesh((3, 3, 3), lengths=(1.0, 1.0, 1.0))
        solution = solve_simplex_poisson(mesh, 1.0)
        boundary = mesh.boundary_nodes()
        interior = np.setdiff1d(np.arange(mesh.n_nodes), boundary)

        self.assertTrue(np.all(np.isfinite(solution)))
        np.testing.assert_allclose(solution[boundary], np.zeros(boundary.shape))
        self.assertGreater(float(solution[interior].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
