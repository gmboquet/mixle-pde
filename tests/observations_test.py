"""Typed observations + common likelihood interface (workstream G2): mixle_pde.observations."""

import unittest

import numpy as np

from mixle_pde.geophysics import gravity_point_sensitivity
from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    gaussian_log_likelihood,
    gravity_forward_operator,
    magnetics_forward_operator,
)


def _grid(n_per_axis=3, z=-50.0, spacing=10.0):
    xs = np.arange(n_per_axis, dtype=float) * spacing
    coords = np.array([[x, y, z] for x in xs for y in xs])
    return coords


class ObservationConstructionTest(unittest.TestCase):
    def test_valid_diagonal_noise(self):
        obs = Observation(kind="gravity", location=[[0.0, 0.0, 0.0]], value=[1.2], noise_cov=[0.01])
        self.assertEqual(obs.n, 1)
        self.assertTrue(obs.is_diagonal)

    def test_bad_location_shape_raises(self):
        with self.assertRaises(ValueError):
            Observation(kind="gravity", location=[[0.0, 0.0]], value=[1.0], noise_cov=[0.01])

    def test_value_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            Observation(kind="gravity", location=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], value=[1.0], noise_cov=[0.01])

    def test_asymmetric_full_covariance_raises(self):
        with self.assertRaises(ValueError):
            Observation(
                kind="gravity",
                location=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                value=[1.0, 2.0],
                noise_cov=[[1.0, 0.5], [0.1, 1.0]],
            )

    def test_nonpositive_diagonal_variance_raises(self):
        with self.assertRaises(ValueError):
            Observation(kind="gravity", location=[[0.0, 0.0, 0.0]], value=[1.0], noise_cov=[0.0])


class GaussianLogLikelihoodTest(unittest.TestCase):
    def test_matches_manual_diagonal_formula(self):
        obs = Observation(kind="gravity", location=[[0.0, 0.0, 0.0]], value=[2.0], noise_cov=[0.25])
        ll = gaussian_log_likelihood(obs, predicted=[1.5])
        expected = -0.5 * ((2.0 - 1.5) ** 2 / 0.25 + np.log(2 * np.pi * 0.25))
        self.assertAlmostEqual(ll, expected, places=10)

    def test_matches_manual_full_covariance_formula(self):
        cov = np.array([[1.0, 0.3], [0.3, 0.8]])
        obs = Observation(kind="gravity", location=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], value=[1.0, 2.0], noise_cov=cov)
        predicted = np.array([0.5, 2.5])
        residual = obs.value - predicted
        prec = np.linalg.inv(cov)
        _, logdet = np.linalg.slogdet(cov)
        expected = -0.5 * (residual @ prec @ residual + logdet + 2 * np.log(2 * np.pi))
        self.assertAlmostEqual(gaussian_log_likelihood(obs, predicted), expected, places=10)

    def test_likelihood_is_maximized_at_zero_residual(self):
        obs = Observation(kind="gravity", location=[[0.0, 0.0, 0.0]], value=[3.0], noise_cov=[0.1])
        at_truth = gaussian_log_likelihood(obs, predicted=[3.0])
        off = gaussian_log_likelihood(obs, predicted=[3.5])
        self.assertGreater(at_truth, off)

    def test_wrong_shape_predicted_raises(self):
        obs = Observation(kind="gravity", location=[[0.0, 0.0, 0.0]], value=[1.0], noise_cov=[0.1])
        with self.assertRaises(ValueError):
            gaussian_log_likelihood(obs, predicted=[1.0, 2.0])


class GravityOperatorTest(unittest.TestCase):
    def test_predict_matches_manual_sensitivity_matmul(self):
        cells = _grid()
        volumes = np.full(len(cells), 1000.0)
        grid = Field3D(coordinates=cells, spacing=10.0, units="kg/m^3", property_name="density_contrast")
        field_values = np.zeros(len(cells))
        field_values[4] = 500.0  # one anomalous cell

        op = gravity_forward_operator(cells, volumes)
        obs_locations = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]])
        predicted = op.predict(grid, field_values, obs_locations)

        expected = gravity_point_sensitivity(obs_locations, cells, volumes) @ field_values
        np.testing.assert_allclose(predicted, expected)
        self.assertTrue(op.has_adjoint())

    def test_likelihood_favors_the_true_field_over_a_wrong_one(self):
        cells = _grid()
        volumes = np.full(len(cells), 1000.0)
        grid = Field3D(coordinates=cells, spacing=10.0, units="kg/m^3", property_name="density_contrast")
        true_field = np.zeros(len(cells))
        true_field[4] = 500.0

        op = gravity_forward_operator(cells, volumes)
        obs_locations = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0], [0.0, 20.0, 0.0]])
        rng = np.random.RandomState(0)
        true_signal = op.predict(grid, true_field, obs_locations)
        noisy_value = true_signal + rng.normal(0, 0.01, size=len(true_signal))
        obs = Observation(kind="gravity", location=obs_locations, value=noisy_value, noise_cov=np.full(3, 0.01**2))

        wrong_field = np.zeros(len(cells))
        wrong_field[0] = 500.0

        registry = ForwardOperatorRegistry()
        registry.register(op)
        ll_true = registry.log_likelihood(grid, true_field, obs)
        ll_wrong = registry.log_likelihood(grid, wrong_field, obs)
        self.assertGreater(ll_true, ll_wrong)


class MagneticsOperatorTest(unittest.TestCase):
    def test_predict_matches_manual_sensitivity_matmul(self):
        from mixle_pde.geophysics import magnetic_dipole_sensitivity

        cells = _grid()
        volumes = np.full(len(cells), 1000.0)
        grid = Field3D(coordinates=cells, spacing=10.0, units="SI", property_name="susceptibility")
        field_values = np.zeros(len(cells))
        field_values[3] = 0.05

        op = magnetics_forward_operator(cells, volumes, inclination=60.0, declination=5.0)
        obs_locations = np.array([[10.0, 10.0, 0.0]])
        predicted = op.predict(grid, field_values, obs_locations)
        expected = (
            magnetic_dipole_sensitivity(obs_locations, cells, volumes, inclination=60.0, declination=5.0) @ field_values
        )
        np.testing.assert_allclose(predicted, expected)


class BoreholeOperatorTest(unittest.TestCase):
    def test_predict_recovers_exact_field_value_at_a_grid_point(self):
        cells = _grid()
        grid = Field3D(coordinates=cells, spacing=10.0, units="frac", property_name="porosity")
        field_values = np.linspace(0.1, 0.9, len(cells))

        op = borehole_forward_operator()
        obs_locations = cells[[2, 5]]
        predicted = op.predict(grid, field_values, obs_locations)
        np.testing.assert_allclose(predicted, field_values[[2, 5]])

    def test_jacobian_is_a_selection_matrix(self):
        cells = _grid()
        grid = Field3D(coordinates=cells, spacing=10.0, units="frac", property_name="porosity")
        op = borehole_forward_operator()
        J = op.jacobian(grid, cells[[0, 3]])
        self.assertEqual(J.shape, (2, len(cells)))
        np.testing.assert_allclose(J.sum(axis=1), [1.0, 1.0])
        np.testing.assert_allclose(J[0], np.eye(len(cells))[0])


class RegistryMultiKindFusionTest(unittest.TestCase):
    def test_unregistered_kind_raises_key_error(self):
        registry = ForwardOperatorRegistry()
        with self.assertRaises(KeyError):
            registry.get("gravity")
        self.assertNotIn("gravity", registry)

    def test_total_log_likelihood_favors_truth_across_mixed_observation_kinds(self):
        cells = _grid()
        volumes = np.full(len(cells), 1000.0)
        grid = Field3D(coordinates=cells, spacing=10.0, units="kg/m^3", property_name="density_contrast")
        true_field = np.zeros(len(cells))
        true_field[4] = 500.0

        registry = ForwardOperatorRegistry()
        gravity_op = gravity_forward_operator(cells, volumes)
        borehole_op = borehole_forward_operator()
        registry.register(gravity_op)
        registry.register(borehole_op)
        self.assertIn("gravity", registry)
        self.assertIn("borehole", registry)

        grav_obs_locations = np.array([[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]])
        grav_value = gravity_op.predict(grid, true_field, grav_obs_locations)
        grav_observation = Observation(
            kind="gravity", location=grav_obs_locations, value=grav_value, noise_cov=np.full(2, 0.05**2)
        )
        bh_location = cells[[4]]
        bh_observation = Observation(kind="borehole", location=bh_location, value=[500.0], noise_cov=[10.0**2])
        observations = [grav_observation, bh_observation]

        wrong_field = np.zeros(len(cells))
        wrong_field[0] = 500.0

        ll_true = registry.total_log_likelihood(grid, true_field, observations)
        ll_wrong = registry.total_log_likelihood(grid, wrong_field, observations)
        self.assertGreater(ll_true, ll_wrong)


if __name__ == "__main__":
    unittest.main()
