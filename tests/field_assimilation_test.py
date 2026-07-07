"""4D assimilation and smoothing of an evolving latent field (workstream G7 acceptance).

Satisfies G's second acceptance criterion: create a 4D evolving Earth-state object, assimilate
observations over time (including a time with NO observations), and extract posterior slices and
posterior-predictive observations at multiple times.
"""

import unittest

import numpy as np

from mixle_pde.field_assimilation import PosteriorField4D, assimilate_4d
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.latent import Field3D, PosteriorField3D
from mixle_pde.observations import (
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


if __name__ == "__main__":
    unittest.main()
