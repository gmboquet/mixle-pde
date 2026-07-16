"""Unstructured P1 finite-element Poisson solver (Phase 5)."""

import math
import unittest

import numpy as np
from scipy.spatial import Delaunay

from mixle_pde.fem import (
    RobinBC,
    assemble_robin_boundary_terms,
    assemble_simplex_fem_matrices,
    assemble_simplex_load_vector,
    assemble_simplex_mass_matrix,
    assemble_simplex_stiffness_matrix,
    boundary_nodes,
    fem_poisson,
    simulate_simplex_diffusion,
    solve_simplex_poisson,
    step_simplex_diffusion,
)
from mixle_pde.mesh import box_simplex_mesh
from mixle_pde.verification.mms import estimate_convergence_order


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

    def test_simplex_diffusion_step_damps_3d_interior_impulse(self):
        mesh = box_simplex_mesh((3, 3, 3), lengths=(1.0, 1.0, 1.0))
        initial = np.zeros(mesh.n_nodes)
        center = np.where(np.all(np.isclose(mesh.nodes, np.array([0.5, 0.5, 0.5])), axis=1))[0][0]
        initial[center] = 1.0

        updated = step_simplex_diffusion(mesh, initial, 0.05)

        self.assertTrue(np.all(np.isfinite(updated)))
        self.assertLess(float(updated[center]), 1.0)
        np.testing.assert_allclose(updated[mesh.boundary_nodes()], np.zeros(len(mesh.boundary_nodes())))

    def test_simplex_diffusion_simulation_returns_time_series(self):
        mesh = box_simplex_mesh((3, 3, 3), lengths=(1.0, 1.0, 1.0))
        initial = np.sin(np.pi * mesh.nodes[:, 0]) * np.sin(np.pi * mesh.nodes[:, 1]) * np.sin(np.pi * mesh.nodes[:, 2])

        states = simulate_simplex_diffusion(mesh, initial, [0.0, 0.05, 0.1])

        self.assertEqual(states.shape, (3, mesh.n_nodes))
        self.assertLess(float(np.linalg.norm(states[-1])), float(np.linalg.norm(states[0])))

    def test_robin_beta_zero_is_rejected(self):
        mesh = box_simplex_mesh((3, 3))
        with self.assertRaises(ValueError):
            assemble_robin_boundary_terms(mesh, RobinBC(alpha=1.0, beta=0.0, g=0.0))

    def test_robin_zero_alpha_and_g_is_a_true_natural_noop(self):
        # alpha=0 and g=0 is a pure zero-flux (Neumann) edge: the boundary integral degenerates to
        # "do nothing" to the assembled system, even though facets are still visited (the stiffness
        # matrix's sparsity pattern gains explicit zero entries, but every value is exactly zero).
        mesh = box_simplex_mesh((4, 4))
        matrix, rhs = assemble_robin_boundary_terms(mesh, RobinBC(alpha=0.0, beta=1.0, g=0.0))
        np.testing.assert_allclose(matrix.toarray(), np.zeros((mesh.n_nodes, mesh.n_nodes)))
        np.testing.assert_allclose(rhs, np.zeros(mesh.n_nodes))

    def test_robin_manufactured_solution_converges(self):
        # u = sin(pi x) * (y + 1) on the unit square. -Laplacian(u) = pi^2 sin(pi x) (y+1).
        # Dirichlet (exact trace) on x=0, x=1, y=0. Robin on y=1 (outward normal +y):
        #   alpha*u + beta*du/dn = g, with alpha=2, beta=1 =>
        #   g = 2*(2 sin(pi x)) + 1*(sin(pi x)) = 5 sin(pi x)  (nonconstant along the Robin edge).
        alpha, beta = 2.0, 1.0

        def u_exact(xy):
            x, y = xy[0], xy[1]
            return math.sin(math.pi * x) * (y + 1.0)

        def source(xy):
            x, y = xy[0], xy[1]
            return (math.pi**2) * math.sin(math.pi * x) * (y + 1.0)

        def robin_g(xy):
            return 5.0 * math.sin(math.pi * xy[0])

        resolutions = (11, 21, 41)
        hs = []
        errs = []
        for n in resolutions:
            mesh = box_simplex_mesh((n, n))
            boundary = mesh.boundary_nodes()
            bx, by = mesh.nodes[boundary, 0], mesh.nodes[boundary, 1]
            is_dirichlet = np.isclose(bx, 0.0) | np.isclose(bx, 1.0) | np.isclose(by, 0.0)
            dirichlet = {int(node): float(u_exact(mesh.nodes[node])) for node in boundary[is_dirichlet]}

            facets = mesh.boundary_facets()
            on_top = np.all(np.isclose(mesh.nodes[facets][..., 1], 1.0), axis=1)
            robin = RobinBC(alpha=alpha, beta=beta, g=robin_g, facets=facets[on_top])

            src_nodal = np.array([source(p) for p in mesh.nodes])
            u = solve_simplex_poisson(mesh, src_nodal, dirichlet=dirichlet, robin=robin)
            exact = np.array([u_exact(p) for p in mesh.nodes])

            hs.append(1.0 / (n - 1))
            errs.append(float(np.max(np.abs(u - exact))))

        self.assertLess(errs[-1], 2.0e-3)
        self.assertGreater(errs[0] / errs[1], 3.0)  # ~O(h^2): error drops ~4x per halving
        self.assertGreater(errs[1] / errs[2], 3.0)
        measured_order = estimate_convergence_order(hs, errs)
        self.assertAlmostEqual(measured_order, 2.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
