"""Field3D / PosteriorField3D: construction, bound transforms, sampling, credible intervals, slicing."""

import unittest

import numpy as np

from mixle_pde.latent import Field3D, Field4D, PosteriorField3D, SparsePosteriorPrecision
from mixle_pde.mesh import box_simplex_mesh, moving_mesh, pipe_radial_deformation


def _grid(n_per_axis=3):
    xs = np.arange(n_per_axis, dtype=float)
    coords = np.array([[x, y, z] for x in xs for y in xs for z in xs])
    return coords


class Field3DConstructionTestCase(unittest.TestCase):
    def test_valid_construction(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="porosity")
        self.assertEqual(f.n, coords.shape[0])
        self.assertEqual(f.geometry_kind, "point_grid")

    def test_field_can_bind_to_static_simplex_mesh(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        field = Field3D.from_mesh(mesh, spacing=1.0, units="kg/m^3", property_name="density")

        self.assertEqual(field.n, mesh.n_nodes)
        self.assertEqual(field.geometry_kind, "simplex_mesh")
        self.assertEqual(field.cell_measures.shape, (mesh.n_simplices,))

    def test_bad_coordinate_shape_raises(self):
        with self.assertRaises(ValueError):
            Field3D(coordinates=np.zeros((5, 2)), spacing=1.0, units="m", property_name="x")

    def test_mask_shape_mismatch_raises(self):
        coords = _grid()
        with self.assertRaises(ValueError):
            Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x", mask=np.ones(3, dtype=bool))

    def test_bad_bounds_raises(self):
        coords = _grid()
        with self.assertRaises(ValueError):
            Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x", bounds=(1.0, 0.0))

    def test_mesh_coordinate_mismatch_raises(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            Field3D(
                coordinates=mesh.nodes + np.array([1.0, 0.0, 0.0]),
                spacing=1.0,
                units="m",
                property_name="x",
                mesh=mesh,
            )


class Field4DConstructionTestCase(unittest.TestCase):
    def test_static_4d_field_exposes_space_time_coordinates_and_transforms(self):
        spatial = Field3D(coordinates=_grid(n_per_axis=2), spacing=1.0, units="frac", property_name="porosity")
        field = Field4D(spatial, times=np.array([0.0, 1.0, 2.0]))
        values = np.arange(field.n, dtype=float).reshape(field.n_times, field.n_per_time)

        self.assertEqual(field.n, 24)
        self.assertEqual(field.coordinates.shape, (24, 4))
        np.testing.assert_allclose(field.values_at_time(values, 1.0), values[1])
        np.testing.assert_allclose(field.from_unconstrained(field.to_unconstrained(values)), values)

    def test_moving_4d_field_uses_deformed_coordinates(self):
        mesh = box_simplex_mesh((2, 2, 2), lengths=(1.0, 1.0, 1.0), origin=(-0.5, -0.5, 0.0))
        spatial = Field3D.from_mesh(mesh, spacing=1.0, units="m", property_name="deformation")
        motion = moving_mesh(mesh, [0.0, 1.0], pipe_radial_deformation(axis="z", radial_strain=0.25))
        field = Field4D(spatial, times=np.array([0.0, 1.0]), moving_mesh=motion)

        self.assertEqual(field.geometry_kind, "moving_simplex_mesh")
        self.assertEqual(field.coordinates_at_time(1.0).shape, spatial.coordinates.shape)
        self.assertAlmostEqual(field.mesh_at_time(1.0).total_measure() / mesh.total_measure(), 1.25**2)

    def test_bad_4d_times_raise(self):
        spatial = Field3D(coordinates=_grid(n_per_axis=2), spacing=1.0, units="m", property_name="x")
        with self.assertRaises(ValueError):
            Field4D(spatial, times=np.array([0.0, 0.0]))


class TransformRoundTripTestCase(unittest.TestCase):
    def test_two_sided_bounds_round_trip(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="frac", property_name="porosity", bounds=(0.0, 1.0))
        physical = np.linspace(0.05, 0.95, coords.shape[0])
        unconstrained = f.to_unconstrained(physical)
        recovered = f.from_unconstrained(unconstrained)
        np.testing.assert_allclose(recovered, physical, atol=1e-8)

    def test_lower_bound_only_round_trip(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="mD", property_name="permeability", bounds=(0.0, None))
        physical = np.linspace(0.1, 100.0, coords.shape[0])
        unconstrained = f.to_unconstrained(physical)
        recovered = f.from_unconstrained(unconstrained)
        np.testing.assert_allclose(recovered, physical, atol=1e-6)

    def test_upper_bound_only_round_trip(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="frac", property_name="saturation", bounds=(None, 1.0))
        physical = np.linspace(-5.0, 0.9, coords.shape[0])
        unconstrained = f.to_unconstrained(physical)
        recovered = f.from_unconstrained(unconstrained)
        np.testing.assert_allclose(recovered, physical, atol=1e-6)

    def test_unbounded_is_identity(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="kg/m3", property_name="density")
        physical = np.linspace(-3.0, 3.0, coords.shape[0])
        unconstrained = f.to_unconstrained(physical)
        np.testing.assert_allclose(unconstrained, physical)
        np.testing.assert_allclose(f.from_unconstrained(unconstrained), physical)


class PosteriorField3DConstructionTestCase(unittest.TestCase):
    def test_requires_a_covariance_mode(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        with self.assertRaises(ValueError):
            PosteriorField3D(grid=f, mean=np.zeros(f.n))

    def test_rejects_two_covariance_modes(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        with self.assertRaises(ValueError):
            PosteriorField3D(grid=f, mean=np.zeros(f.n), cov=np.eye(f.n), diag_var=np.ones(f.n))

    def test_sparse_precision_marginals_match_dense_inverse(self):
        import scipy.sparse as sp

        coords = _grid(n_per_axis=2)
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        precision = sp.diags(np.linspace(1.0, 2.0, f.n), format="csr")
        sparse_factor = SparsePosteriorPrecision(precision)
        post = PosteriorField3D(grid=f, mean=np.zeros(f.n), precision_factor=sparse_factor)
        np.testing.assert_allclose(post.marginal_variance, 1.0 / np.linspace(1.0, 2.0, f.n))

    def test_map_defaults_to_mean(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        post = PosteriorField3D(grid=f, mean=np.full(f.n, 2.0), diag_var=np.ones(f.n))
        np.testing.assert_allclose(post.map, post.mean)


class SamplingTestCase(unittest.TestCase):
    def test_diagonal_sampling_shape_and_finite(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        post = PosteriorField3D(grid=f, mean=np.zeros(f.n), diag_var=np.full(f.n, 0.25))
        samples = post.sample(50, np.random.default_rng(0))
        self.assertEqual(samples.shape, (50, f.n))
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_dense_and_low_rank_recover_matching_marginal_std(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        rng = np.random.default_rng(1)
        low_rank = rng.normal(size=(f.n, 2)) * 0.3
        diag_var = np.full(f.n, 0.1)
        cov = low_rank @ low_rank.T + np.diag(diag_var)

        post_dense = PosteriorField3D(grid=f, mean=np.zeros(f.n), cov=cov)
        post_lr = PosteriorField3D(grid=f, mean=np.zeros(f.n), low_rank=low_rank, diag_var=diag_var)
        np.testing.assert_allclose(post_dense.marginal_variance, post_lr.marginal_variance, atol=1e-10)

        samples = post_dense.sample(20000, np.random.default_rng(2))
        empirical_std = samples.std(axis=0)
        np.testing.assert_allclose(empirical_std, post_dense.marginal_std, atol=0.05)

    def test_sparse_precision_sampling_shape_and_finite(self):
        import scipy.sparse as sp

        coords = _grid(n_per_axis=2)
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        precision = sp.diags(np.full(f.n, 4.0), format="csr")
        post = PosteriorField3D(
            grid=f,
            mean=np.zeros(f.n),
            precision_factor=SparsePosteriorPrecision(precision),
        )
        samples = post.sample(20, np.random.default_rng(4))
        self.assertEqual(samples.shape, (20, f.n))
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_sampling_maps_through_bounds(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="frac", property_name="porosity", bounds=(0.0, 1.0))
        post = PosteriorField3D(grid=f, mean=np.zeros(f.n), diag_var=np.full(f.n, 4.0))
        samples = post.sample(200, np.random.default_rng(3))
        self.assertTrue(np.all(samples > 0.0))
        self.assertTrue(np.all(samples < 1.0))


class CredibleIntervalTestCase(unittest.TestCase):
    def test_unbounded_interval_matches_closed_form_gaussian(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        mean = np.full(f.n, 1.5)
        var = np.full(f.n, 4.0)
        post = PosteriorField3D(grid=f, mean=mean, diag_var=var)
        lo, hi = post.credible_interval(alpha=0.1)
        # 90% central interval of N(1.5, 2^2): 1.5 +/- 1.6448536 * 2
        z = 1.6448536269514722
        np.testing.assert_allclose(lo, mean - z * 2.0, atol=1e-6)
        np.testing.assert_allclose(hi, mean + z * 2.0, atol=1e-6)

    def test_interval_ordering_and_symmetric_coverage_bounded(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="frac", property_name="porosity", bounds=(0.0, 1.0))
        post = PosteriorField3D(grid=f, mean=np.zeros(f.n), diag_var=np.full(f.n, 1.0))
        lo, hi = post.credible_interval(alpha=0.2)
        self.assertTrue(np.all(lo < hi))
        self.assertTrue(np.all((lo > 0.0) & (hi < 1.0)))
        # symmetric in unconstrained space around mean=0 -> physical bounds symmetric around 0.5
        np.testing.assert_allclose(lo + hi, np.full(f.n, 1.0), atol=1e-8)

    def test_bad_alpha_raises(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        post = PosteriorField3D(grid=f, mean=np.zeros(f.n), diag_var=np.ones(f.n))
        with self.assertRaises(ValueError):
            post.credible_interval(alpha=0.0)
        with self.assertRaises(ValueError):
            post.credible_interval(alpha=1.0)


class SliceTestCase(unittest.TestCase):
    def test_slice_selects_expected_subset_and_values(self):
        coords = _grid(n_per_axis=3)
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        # mean varies only with z, closed-form known per point
        mean = coords[:, 2] * 10.0
        post = PosteriorField3D(grid=f, mean=mean, diag_var=np.full(f.n, 0.01))

        result = post.slice(z=1.0)
        expected_index = coords[:, 2] == 1.0
        np.testing.assert_array_equal(result["index"], expected_index)
        self.assertEqual(result["coordinates"].shape[0], expected_index.sum())
        np.testing.assert_allclose(result["mean"], mean[expected_index])
        self.assertTrue(np.all(result["coordinates"][:, 2] == 1.0))

    def test_slice_two_axes(self):
        coords = _grid(n_per_axis=3)
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        post = PosteriorField3D(grid=f, mean=np.zeros(f.n), diag_var=np.ones(f.n))
        result = post.slice(x=0.0, y=0.0)
        expected_index = (coords[:, 0] == 0.0) & (coords[:, 1] == 0.0)
        np.testing.assert_array_equal(result["index"], expected_index)
        self.assertEqual(result["coordinates"].shape[0], 3)  # one point per z level

    def test_slice_requires_an_axis(self):
        coords = _grid()
        f = Field3D(coordinates=coords, spacing=1.0, units="m", property_name="x")
        post = PosteriorField3D(grid=f, mean=np.zeros(f.n), diag_var=np.ones(f.n))
        with self.assertRaises(ValueError):
            post.slice()


if __name__ == "__main__":
    unittest.main()
