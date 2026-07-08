"""4D assimilation and smoothing of an evolving latent field (workstream G7 acceptance).

Satisfies G's second acceptance criterion: create a 4D evolving Earth-state object, assimilate
observations over time (including a time with NO observations), and extract posterior slices and
posterior-predictive observations at multiple times.
"""

import unittest

import numpy as np

from mixle_pde.field_assimilation import PosteriorField4D, assimilate_4d, assimilate_4d_ensemble
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import (
    ForwardOperator,
    ForwardOperatorRegistry,
    Observation,
    borehole_forward_operator,
    gravity_forward_operator,
)


def _grid():
    xs = np.linspace(0.0, 100.0, 5)
    ys = np.linspace(0.0, 100.0, 5)
    zs = np.array([-30.0, -50.0])
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    return Field3D(coordinates=pts, spacing=25.0, units="kg/m^3", property_name="density_contrast", bounds=None)


def _blob(grid, amplitude):
    centre = np.array([50.0, 50.0, -40.0])
    d2 = np.sum((grid.coordinates - centre) ** 2, axis=1)
    return amplitude * np.exp(-d2 / (2.0 * 35.0**2))


class Assimilation4DTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.grid = _grid()
        self.times = np.array([0.0, 1.0, 2.0, 3.0])
        self.amps = [300.0, 400.0, 500.0, 600.0]  # density-contrast blob (kg/m^3) growing in time
        self.truth = [_blob(self.grid, a) for a in self.amps]
        self.volumes = np.full(self.grid.n, 25.0**3, dtype=float)

        self.registry = ForwardOperatorRegistry()
        self.registry.register(gravity_forward_operator(self.grid.coordinates, self.volumes))
        self.registry.register(borehole_forward_operator())

        gx, gy = np.meshgrid(np.linspace(0, 100, 4), np.linspace(0, 100, 4))
        self.grav_loc = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)])
        Gjac = self.registry.get("gravity").jacobian(self.grid, self.grav_loc)

        # observe times 0, 1, 3 (skip time 2 to exercise an UN-observed time); borehole + gravity each
        obs_by_time = []
        for t in range(self.times.size):
            if t == 2:
                obs_by_time.append([])
                continue
            idx = self.rng.choice(self.grid.n, size=int(0.6 * self.grid.n), replace=False)
            bore = Observation(
                kind="borehole",
                location=self.grid.coordinates[idx],
                value=self.truth[t][idx] + self.rng.normal(0, 5.0, size=idx.shape),
                noise_cov=np.full(idx.shape, 25.0),
                time=self.times[t],
            )
            grav = Observation(
                kind="gravity",
                location=self.grav_loc,
                value=Gjac @ self.truth[t] + self.rng.normal(0, 2.0e-4, size=Gjac.shape[0]),
                noise_cov=np.full(Gjac.shape[0], (2.0e-4) ** 2),
                time=self.times[t],
            )
            obs_by_time.append([bore, grav])
        self.obs_by_time = obs_by_time
        self.prior = FieldGaussianPrior(
            mean=0.0, smoothness_precision=2.0e-3, marginal_precision=1.0e-5, length_scale=25.0, neighbors=6
        )

    def _assimilate(self):
        return assimilate_4d(self.grid, self.times, self.obs_by_time, self.registry, self.prior, process_var=2000.0)

    def test_posterior_tracks_the_evolving_field_at_observed_times(self):
        post = self._assimilate()
        self.assertIsInstance(post, PosteriorField4D)
        for t in (0, 1, 3):
            slice_t = post.at_time(self.times[t])
            self.assertIsInstance(slice_t, PosteriorField3D)
            corr = np.corrcoef(slice_t.mean, self.truth[t])[0, 1]
            self.assertGreater(corr, 0.85)

    def test_unobserved_time_still_gets_a_posterior_bracketed_by_its_neighbours(self):
        post = self._assimilate()
        mid = post.at_time(2.0).mean
        before = post.at_time(1.0).mean
        after = post.at_time(3.0).mean
        # the smoother interpolates the un-observed time between its observed neighbours
        lo = np.minimum(before, after) - 1e-6
        hi = np.maximum(before, after) + 1e-6
        frac_bracketed = float(np.mean((mid >= lo) & (mid <= hi)))
        self.assertGreater(frac_bracketed, 0.8)
        # and it is less certain than the observed times around it
        self.assertGreater(post.at_time(2.0).marginal_std.mean(), post.at_time(1.0).marginal_std.mean())

    def test_slices_expose_full_posterior_artifacts_at_each_time(self):
        post = self._assimilate()
        for t in self.times:
            s = post.at_time(t)
            self.assertTrue(np.all(s.marginal_variance > 0))
            samples = s.sample(16, self.rng)
            self.assertEqual(samples.shape, (16, self.grid.n))
            lo, hi = s.credible_interval(0.1)
            self.assertTrue(np.all(hi >= lo))

    def test_4d_posterior_exposes_time_axis_artifacts(self):
        post = self._assimilate()
        lo, hi = post.credible_interval(0.1)
        samples = post.sample(8, self.rng)

        self.assertEqual(post.mean_array.shape, (self.times.size, self.grid.n))
        self.assertEqual(post.marginal_std.shape, (self.times.size, self.grid.n))
        self.assertEqual(lo.shape, (self.times.size, self.grid.n))
        self.assertEqual(hi.shape, (self.times.size, self.grid.n))
        self.assertEqual(samples.shape, (8, self.times.size, self.grid.n))
        self.assertTrue(np.all(hi >= lo))

    def test_4d_posterior_interpolates_between_assimilated_times(self):
        post = self._assimilate()
        interp = post.at_time(1.5, interpolate=True)

        expected = 0.5 * (post.at_time(1.0).mean + post.at_time(2.0).mean)
        self.assertIsInstance(interp, PosteriorField3D)
        np.testing.assert_allclose(interp.mean, expected)

    def test_posterior_predictive_observation_at_multiple_times(self):
        post = self._assimilate()
        for t in (0, 3):
            held = Observation(
                kind="gravity",
                location=self.grav_loc,
                value=np.zeros(self.grav_loc.shape[0]),  # value unused by predict_observation
                noise_cov=np.full(self.grav_loc.shape[0], (2.0e-4) ** 2),
                time=self.times[t],
            )
            pred = post.predict_observation(self.registry, held)
            truth_grav = self.registry.get("gravity").jacobian(self.grid, self.grav_loc) @ self.truth[t]
            corr = np.corrcoef(pred, truth_grav)[0, 1]
            self.assertGreater(corr, 0.9)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            assimilate_4d(self.grid, self.times, self.obs_by_time, self.registry, self.prior, process_var=0.0)
        with self.assertRaises(ValueError):
            assimilate_4d(self.grid, self.times, self.obs_by_time[:2], self.registry, self.prior, process_var=1.0)
        bounded = Field3D(
            coordinates=self.grid.coordinates, spacing=25.0, units="", property_name="p", bounds=(0.0, 1.0)
        )
        with self.assertRaises(ValueError):
            assimilate_4d(bounded, self.times, self.obs_by_time, self.registry, self.prior, process_var=1.0)
        with self.assertRaises(ValueError):
            PosteriorField4D(self.grid, np.array([0.0, 0.0]), [np.zeros(self.grid.n)], [np.eye(self.grid.n)])


class EnsembleAssimilation4DTest(unittest.TestCase):
    def setUp(self):
        self.grid = Field3D(
            coordinates=np.array([[0.0, 0.0, -10.0]]),
            spacing=1.0,
            units="state",
            property_name="nonlinear_state",
        )
        self.times = np.array([0.0, 1.0, 2.0])
        self.truth = np.array([1.0, 1.35, 1.7])
        self.registry = ForwardOperatorRegistry()

        def predict(grid, field_values, obs_locations):
            return np.full(obs_locations.shape[0], float(field_values[0] ** 2))

        self.registry.register(ForwardOperator("square_sensor", predict=predict, differentiable=False))
        self.observations = [
            [
                Observation(
                    "square_sensor",
                    np.array([[0.0, 0.0, -10.0]]),
                    np.array([value**2]),
                    np.array([0.03**2]),
                    time=time,
                )
            ]
            for time, value in zip(self.times, self.truth, strict=True)
        ]
        self.prior = FieldGaussianPrior(mean=0.8, smoothness_precision=0.0, marginal_precision=4.0, length_scale=1.0)

    def test_ensemble_assimilation_tracks_nonlinear_observations_without_a_jacobian(self):
        post = assimilate_4d_ensemble(
            self.grid,
            self.times,
            self.observations,
            self.registry,
            self.prior,
            process_var=0.08,
            ensemble_size=256,
            rng=np.random.default_rng(123),
        )
        self.assertIsInstance(post, PosteriorField4D)
        final_mean = post.at_time(2.0).mean[0]
        self.assertLess(abs(final_mean - self.truth[-1]), abs(0.8 - self.truth[-1]))
        self.assertLess(abs(final_mean - self.truth[-1]), 0.25)
        self.assertGreater(post.at_time(2.0).marginal_std[0], 0.0)

    def test_ensemble_posterior_predictive_supports_nonlinear_operator_at_mean(self):
        post = assimilate_4d_ensemble(
            self.grid,
            self.times,
            self.observations,
            self.registry,
            self.prior,
            process_var=0.08,
            ensemble_size=256,
            rng=np.random.default_rng(321),
        )
        held = Observation(
            "square_sensor",
            np.array([[0.0, 0.0, -10.0]]),
            np.array([0.0]),
            np.array([0.03**2]),
            time=2.0,
        )
        predicted = post.predict_observation(self.registry, held)
        self.assertLess(abs(predicted[0] - self.truth[-1] ** 2), 0.7)

    def test_ensemble_assimilation_validates_inputs(self):
        with self.assertRaises(ValueError):
            assimilate_4d_ensemble(
                self.grid,
                self.times,
                self.observations,
                self.registry,
                self.prior,
                process_var=0.08,
                ensemble_size=1,
            )
        with self.assertRaises(ValueError):
            assimilate_4d_ensemble(
                self.grid,
                self.times,
                self.observations[:1],
                self.registry,
                self.prior,
                process_var=0.08,
            )


if __name__ == "__main__":
    unittest.main()
