"""Typed observations + common likelihood interface (workstream G2): mixle_pde.observations."""

import unittest

import numpy as np

from mixle_pde.geophysics import gravity_point_sensitivity
from mixle_pde.latent import Field3D
from mixle_pde.observations import (
    ForwardOperatorRegistry,
    Observation,
    aem_layered_forward_operator,
    borehole_forward_operator,
    csem_3d_forward_operator,
    dc_resistivity_forward_operator,
    gaussian_log_likelihood,
    gravity_forward_operator,
    layered_mt_forward_operator,
    magnetics_forward_operator,
    mt_2d_te_forward_operator,
    mt_3d_forward_operator,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DCResistivityOperatorTest(unittest.TestCase):
    def test_predict_matches_geophysics_forward(self):
        from mixle_pde.geophysics import dc_resistivity

        shape = (4, 4, 4)
        idx = np.arange(np.prod(shape)).reshape(shape)
        schedule = [(int(idx[1, 1, 1]), int(idx[1, 1, 2]), int(idx[2, 1, 1]), int(idx[2, 1, 2]))]
        coords = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)], dtype=float)
        grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity")
        values = np.zeros(grid.n)
        values[int(idx[2, 1, 1])] = 0.1
        locations = np.array([[1.5, 1.0, 1.5]])
        op = dc_resistivity_forward_operator(shape, schedule, sigma_ref=0.02, log_data=True)

        expected = (
            dc_resistivity(
                torch.as_tensor(values, dtype=torch.float64),
                shape,
                schedule,
                sigma_ref=0.02,
                log_data=True,
            )
            .detach()
            .numpy()
        )
        np.testing.assert_allclose(op.predict(grid, values, locations), expected)
        self.assertTrue(op.has_adjoint())
        self.assertFalse(op.is_linear)

    def test_local_jacobian_linearizes_a_small_perturbation(self):
        shape = (4, 4, 4)
        idx = np.arange(np.prod(shape)).reshape(shape)
        schedule = [
            (int(idx[1, 1, 1]), int(idx[1, 1, 2]), int(idx[2, 1, 1]), int(idx[2, 1, 2])),
            (int(idx[1, 2, 1]), int(idx[1, 2, 2]), int(idx[2, 2, 1]), int(idx[2, 2, 2])),
        ]
        coords = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)], dtype=float)
        grid = Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity")
        values = np.zeros(grid.n)
        values[int(idx[2, 1, 1])] = 0.05
        locations = np.array([[1.5, 1.0, 1.5], [1.5, 2.0, 1.5]])
        observation = Observation("dc_resistivity", locations, [0.0, 0.0], [1.0, 1.0])
        op = dc_resistivity_forward_operator(shape, schedule, sigma_ref=0.02, finite_difference_step=1.0e-5)
        jac = op.local_jacobian(grid, values, observation)

        perturb = np.zeros(grid.n)
        perturb[int(idx[2, 1, 1])] = 1.0e-4
        base = op.predict_observation(grid, values, observation)
        moved = op.predict_observation(grid, values + perturb, observation)
        np.testing.assert_allclose(moved - base, jac @ perturb, rtol=2.0e-2, atol=1.0e-7)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class LayeredMTForwardOperatorTest(unittest.TestCase):
    def _grid(self):
        return Field3D(
            coordinates=np.array([[0.0, 0.0, -100.0], [0.0, 0.0, -500.0]]),
            spacing=400.0,
            units="log(S/m)",
            property_name="log_conductivity",
        )

    def test_predict_matches_layered_mt_forward(self):
        import torch

        from mixle_pde.em_diffusion import layered_mt_impedance

        freqs = np.array([1.0, 10.0, 100.0])
        thicknesses = [300.0]
        sigma_ref = 0.01
        values = np.log(np.array([2.0, 20.0]))
        grid = self._grid()
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        op = layered_mt_forward_operator(freqs, thicknesses, sigma_ref=sigma_ref)

        sigma = sigma_ref * torch.exp(torch.as_tensor(values, dtype=torch.float64))
        rho_a, _, _ = layered_mt_impedance(sigma, thicknesses, freqs)
        expected = np.log(rho_a.detach().numpy())
        np.testing.assert_allclose(op.predict(grid, values, locations), expected)
        self.assertTrue(op.has_adjoint())
        self.assertFalse(op.is_linear)
        self.assertEqual(op.kind, "layered_mt_log_apparent_resistivity")

    def test_local_jacobian_linearizes_a_small_perturbation(self):
        freqs = np.array([1.0, 10.0, 100.0])
        thicknesses = [300.0]
        grid = self._grid()
        values = np.log(np.array([2.0, 20.0]))
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        observation = Observation(
            "layered_mt_log_apparent_resistivity",
            locations,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        )
        op = layered_mt_forward_operator(freqs, thicknesses, sigma_ref=0.01, finite_difference_step=1.0e-5)
        jac = op.local_jacobian(grid, values, observation)

        perturb = np.array([1.0e-4, -2.0e-4])
        base = op.predict_observation(grid, values, observation)
        moved = op.predict_observation(grid, values + perturb, observation)
        np.testing.assert_allclose(moved - base, jac @ perturb, rtol=1.0e-3, atol=1.0e-8)

    def test_phase_component_and_validation(self):
        freqs = np.array([1.0, 10.0])
        grid = self._grid()
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        op = layered_mt_forward_operator(freqs, [300.0], component="phase", sigma_ref=0.01)
        out = op.predict(grid, np.log(np.array([2.0, 20.0])), locations)
        self.assertEqual(out.shape, (2,))
        self.assertEqual(op.kind, "layered_mt_phase")
        with self.assertRaises(ValueError):
            layered_mt_forward_operator(freqs, [300.0], component="bad")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class LayeredAEMForwardOperatorTest(unittest.TestCase):
    def _grid(self):
        return Field3D(
            coordinates=np.array([[0.0, 0.0, -100.0], [0.0, 0.0, -500.0]]),
            spacing=400.0,
            units="log(S/m)",
            property_name="log_conductivity",
        )

    def test_predict_matches_reciprocal_layered_mt_apparent_resistivity(self):
        import torch

        from mixle_pde.em_diffusion import layered_mt_impedance

        freqs = np.array([1.0, 10.0, 100.0])
        thicknesses = [300.0]
        sigma_ref = 0.01
        values = np.log(np.array([2.0, 20.0]))
        grid = self._grid()
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        op = aem_layered_forward_operator(freqs, thicknesses, sigma_ref=sigma_ref)

        sigma = sigma_ref * torch.exp(torch.as_tensor(values, dtype=torch.float64))
        rho_a, _, _ = layered_mt_impedance(sigma, thicknesses, freqs)
        expected = np.log((1.0 / rho_a).detach().numpy())
        np.testing.assert_allclose(op.predict(grid, values, locations), expected)
        self.assertTrue(op.has_adjoint())
        self.assertFalse(op.is_linear)
        self.assertEqual(op.kind, "aem_layered_log_apparent_conductivity")

    def test_local_jacobian_linearizes_a_small_perturbation(self):
        freqs = np.array([1.0, 10.0, 100.0])
        thicknesses = [300.0]
        grid = self._grid()
        values = np.log(np.array([2.0, 20.0]))
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        observation = Observation(
            "aem_layered_log_apparent_conductivity",
            locations,
            np.zeros(freqs.size),
            np.ones(freqs.size),
        )
        op = aem_layered_forward_operator(freqs, thicknesses, sigma_ref=0.01, finite_difference_step=1.0e-5)
        jac = op.local_jacobian(grid, values, observation)

        perturb = np.array([1.0e-4, -2.0e-4])
        base = op.predict_observation(grid, values, observation)
        moved = op.predict_observation(grid, values + perturb, observation)
        np.testing.assert_allclose(moved - base, jac @ perturb, rtol=1.0e-3, atol=1.0e-8)

    def test_phase_component_and_validation(self):
        freqs = np.array([1.0, 10.0])
        grid = self._grid()
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        op = aem_layered_forward_operator(freqs, [300.0], component="phase", sigma_ref=0.01)
        out = op.predict(grid, np.log(np.array([2.0, 20.0])), locations)
        self.assertEqual(out.shape, (2,))
        self.assertEqual(op.kind, "aem_layered_phase")
        with self.assertRaises(ValueError):
            aem_layered_forward_operator(freqs, [300.0], component="bad")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MT2DTEForwardOperatorTest(unittest.TestCase):
    def _grid(self, shape):
        nx, nz = shape
        coords = np.array([[float(i), 0.0, -float(j)] for i in range(nx) for j in range(nz)])
        return Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity_2d")

    def test_predict_matches_mt_2d_te_forward(self):
        import torch

        from mixle_pde.em_diffusion import mt_2d_te

        shape = (3, 8)
        freq = 10.0
        spacing = 100.0
        sigma_ref = 0.02
        grid = self._grid(shape)
        values = np.zeros(grid.n)
        values[shape[1] + 2] = 0.15
        locations = np.column_stack([np.arange(shape[0], dtype=float), np.zeros(shape[0]), np.zeros(shape[0])])
        op = mt_2d_te_forward_operator(shape, freq, spacing=spacing, sigma_ref=sigma_ref)

        rho_a, _ = mt_2d_te(
            torch.as_tensor(values, dtype=torch.float64),
            shape,
            freq,
            spacing=spacing,
            sigma_ref=sigma_ref,
        )
        expected = np.log(rho_a.detach().numpy())
        np.testing.assert_allclose(op.predict(grid, values, locations), expected)
        self.assertTrue(op.has_adjoint())
        self.assertFalse(op.is_linear)
        self.assertEqual(op.kind, "mt_2d_te_log_apparent_resistivity")

    def test_local_jacobian_linearizes_a_small_perturbation(self):
        shape = (3, 8)
        freq = 10.0
        spacing = 100.0
        grid = self._grid(shape)
        values = np.zeros(grid.n)
        values[shape[1] + 2] = 0.15
        locations = np.column_stack([np.arange(shape[0], dtype=float), np.zeros(shape[0]), np.zeros(shape[0])])
        observation = Observation(
            "mt_2d_te_log_apparent_resistivity",
            locations,
            np.zeros(shape[0]),
            np.ones(shape[0]),
        )
        op = mt_2d_te_forward_operator(shape, freq, spacing=spacing, sigma_ref=0.02, finite_difference_step=1.0e-5)
        jac = op.local_jacobian(grid, values, observation)

        perturb = np.zeros(grid.n)
        perturb[shape[1] + 2] = 1.0e-4
        base = op.predict_observation(grid, values, observation)
        moved = op.predict_observation(grid, values + perturb, observation)
        np.testing.assert_allclose(moved - base, jac @ perturb, rtol=1.0e-3, atol=1.0e-8)

    def test_phase_component_and_validation(self):
        shape = (3, 8)
        freq = 10.0
        grid = self._grid(shape)
        locations = np.column_stack([np.arange(shape[0], dtype=float), np.zeros(shape[0]), np.zeros(shape[0])])
        op = mt_2d_te_forward_operator(shape, freq, component="phase", spacing=100.0, sigma_ref=0.02)
        out = op.predict(grid, np.zeros(grid.n), locations)
        self.assertEqual(out.shape, (shape[0],))
        self.assertEqual(op.kind, "mt_2d_te_phase")
        with self.assertRaises(ValueError):
            mt_2d_te_forward_operator(shape, freq, component="bad")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MT3DForwardOperatorTest(unittest.TestCase):
    def _grid(self, shape):
        nx, ny, nz = shape
        coords = np.array([[float(i), float(j), -float(k)] for i in range(nx) for j in range(ny) for k in range(nz)])
        return Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity_3d")

    def test_predict_matches_mt_3d_forward(self):
        import torch

        from mixle_pde.em_diffusion_3d import mt_3d

        shape = (3, 3, 6)
        freqs = np.array([5.0, 20.0])
        spacing = 50.0
        sigma_ref = 0.05
        grid = self._grid(shape)
        values = np.zeros(grid.n)
        values[shape[2] * (shape[1] + 1) + 2] = 0.1
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        op = mt_3d_forward_operator(shape, freqs, spacing=spacing, sigma_ref=sigma_ref)

        expected = []
        for freq in freqs:
            rho_a, _, _ = mt_3d(
                torch.as_tensor(values, dtype=torch.float64),
                shape,
                float(freq),
                spacing=spacing,
                sigma_ref=sigma_ref,
            )
            expected.append(float(torch.log(rho_a)))
        np.testing.assert_allclose(op.predict(grid, values, locations), np.array(expected))
        self.assertTrue(op.has_adjoint())
        self.assertFalse(op.is_linear)
        self.assertEqual(op.kind, "mt_3d_log_apparent_resistivity")

    def test_local_jacobian_linearizes_a_small_perturbation(self):
        shape = (3, 3, 6)
        freqs = np.array([5.0, 20.0])
        spacing = 50.0
        grid = self._grid(shape)
        values = np.zeros(grid.n)
        param = shape[2] * (shape[1] + 1) + 2
        values[param] = 0.1
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        observation = Observation(
            "mt_3d_log_apparent_resistivity",
            locations,
            np.zeros(freqs.size),
            np.ones(freqs.size),
        )
        op = mt_3d_forward_operator(shape, freqs, spacing=spacing, sigma_ref=0.05, finite_difference_step=1.0e-5)
        jac = op.local_jacobian(grid, values, observation)

        perturb = np.zeros(grid.n)
        perturb[param] = 1.0e-4
        base = op.predict_observation(grid, values, observation)
        moved = op.predict_observation(grid, values + perturb, observation)
        np.testing.assert_allclose(moved - base, jac @ perturb, rtol=2.0e-3, atol=1.0e-8)

    def test_phase_component_and_validation(self):
        shape = (3, 3, 6)
        freqs = np.array([5.0])
        grid = self._grid(shape)
        locations = np.column_stack([freqs, np.zeros_like(freqs), np.zeros_like(freqs)])
        op = mt_3d_forward_operator(shape, freqs, component="phase", spacing=50.0, sigma_ref=0.05)
        out = op.predict(grid, np.zeros(grid.n), locations)
        self.assertEqual(out.shape, (freqs.size,))
        self.assertEqual(op.kind, "mt_3d_phase")
        with self.assertRaises(ValueError):
            mt_3d_forward_operator(shape, freqs, component="bad")


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class CSEM3DForwardOperatorTest(unittest.TestCase):
    def _grid(self, shape):
        nx, ny, nz = shape
        coords = np.array([[float(i), float(j), -float(k)] for i in range(nx) for j in range(ny) for k in range(nz)])
        return Field3D(coords, spacing=1.0, units="log(S/m)", property_name="log_conductivity_3d")

    def _survey(self, shape, spacing):
        from mixle_pde.em_diffusion_3d import _edge_coords, _edge_layout

        nx, ny, nz = shape
        _, _, _, _, (sx, _, _) = _edge_layout(shape)
        ic, jc, kc = min(nx // 2, sx[0] - 1), ny // 2, nz // 2
        source = (ic * sx[1] + jc) * sx[2] + kc
        receivers = np.array([source, source - sx[1] * sx[2], source - sx[2]], dtype=int)
        edge_coords, _ = _edge_coords(shape, (spacing, spacing, spacing))
        return [int(source)], receivers, edge_coords[receivers]

    def test_predict_matches_csem_3d_forward(self):
        from mixle_pde.em_diffusion_3d import csem_3d

        shape = (4, 4, 4)
        freq = 1.0
        spacing = 50.0
        sigma_ref = 0.1
        grid = self._grid(shape)
        source_edges, receiver_edges, locations = self._survey(shape, spacing)
        values = np.zeros(grid.n)
        op = csem_3d_forward_operator(shape, freq, source_edges, receiver_edges, spacing=spacing, sigma_ref=sigma_ref)

        field = csem_3d(
            torch.as_tensor(values, dtype=torch.float64),
            shape,
            freq,
            source_edges=source_edges,
            spacing=spacing,
            sigma_ref=sigma_ref,
        )[list(receiver_edges)]
        expected = np.log(field.detach().abs().numpy())
        np.testing.assert_allclose(op.predict(grid, values, locations), expected)
        self.assertTrue(op.has_adjoint())
        self.assertFalse(op.is_linear)
        self.assertEqual(op.kind, "csem_3d_log_amplitude")

    def test_local_jacobian_linearizes_a_small_perturbation(self):
        shape = (4, 4, 4)
        freq = 1.0
        spacing = 50.0
        grid = self._grid(shape)
        source_edges, receiver_edges, locations = self._survey(shape, spacing)
        values = np.zeros(grid.n)
        observation = Observation(
            "csem_3d_log_amplitude",
            locations,
            np.zeros(len(receiver_edges)),
            np.ones(len(receiver_edges)),
        )
        op = csem_3d_forward_operator(
            shape,
            freq,
            source_edges,
            receiver_edges,
            spacing=spacing,
            sigma_ref=0.1,
            finite_difference_step=1.0e-5,
        )
        jac = op.local_jacobian(grid, values, observation)

        perturb = np.zeros(grid.n)
        perturb[int(np.argmax(np.linalg.norm(jac, axis=0)))] = 1.0e-4
        base = op.predict_observation(grid, values, observation)
        moved = op.predict_observation(grid, values + perturb, observation)
        np.testing.assert_allclose(moved - base, jac @ perturb, rtol=2.0e-3, atol=1.0e-8)

    def test_imag_component_and_validation(self):
        shape = (4, 4, 4)
        freq = 1.0
        spacing = 50.0
        grid = self._grid(shape)
        source_edges, receiver_edges, locations = self._survey(shape, spacing)
        op = csem_3d_forward_operator(
            shape,
            freq,
            source_edges,
            receiver_edges,
            component="imag",
            spacing=spacing,
            sigma_ref=0.1,
        )
        out = op.predict(grid, np.zeros(grid.n), locations)
        self.assertEqual(out.shape, (len(receiver_edges),))
        self.assertEqual(op.kind, "csem_3d_imag")
        with self.assertRaises(ValueError):
            csem_3d_forward_operator(shape, freq, source_edges, receiver_edges, component="bad")


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
