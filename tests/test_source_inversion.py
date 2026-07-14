"""Acceptance test for G2 -- contaminant source inversion (work-plan workstream G, card G2).

A synthetic leak of known location/rate/onset is forward-simulated through a
:class:`~mixle_pde.groundwater.GroundwaterTransportOperator` to six monitoring wells, corrupted with
Gaussian reading noise, and handed to :func:`~mixle_pde.groundwater.invert_source`. The Definition of
Done is two-fold: (1) the true (location, rate) vector lies inside the returned posterior's 90% marginal
credible interval (checked, as the repo's own ``OdeParameterInferenceTestCase.test_coverage_over_seeds``
does, over several independent noise draws rather than a single seed -- a 90% interval is *expected* to
miss about one draw in ten, so a single-seed check would be flaky by construction); and (2) the split-
conformal band A3's recipe builds from the held-out wells actually covers *fresh* noise realizations of
those same held-out readings at least 88% of the time -- not just the one draw used to calibrate it,
which would be circular.
"""

import unittest

import numpy as np

from mixle_pde.groundwater import GroundwaterTransportOperator, invert_source, simulate_wells
from mixle_pde.observations import Observation

TRUE_LOCATION = np.array([2.0, 6.0])
TRUE_RATE = 5.0
TRUE_ONSET = 0.5
NOISE_STD = 0.03

SHAPE = (12, 12)
VELOCITY = [0.4, 0.05]
DISPERSIVITY = [0.3, 0.3]

TIMES = np.linspace(0.0, 12.0, 25)
WELL_XY = [(5, 6, 0), (8, 6, 0), (10, 6, 0), (5, 3, 0), (8, 9, 0), (10, 3, 0)]

# invert_source withholds the LAST held_out_fraction (default 1/3) of wells, in first-seen order.
_N_HELD = max(1, round(len(WELL_XY) / 3))
HELD_XY = WELL_XY[len(WELL_XY) - _N_HELD :]


def _operator() -> GroundwaterTransportOperator:
    return GroundwaterTransportOperator(
        velocity_field=VELOCITY,
        dispersivity=DISPERSIVITY,
        shape=SHAPE,
        spacing=1.0,
        bc="neumann",
        scheme="implicit",
    )


def _make_observations(operator: GroundwaterTransportOperator, noise_seed: int) -> list[Observation]:
    """One :class:`Observation` per (well, sample-time) reading, noisy around the clean forward solve."""
    clean = simulate_wells(TRUE_LOCATION, TRUE_RATE, TRUE_ONSET, operator, WELL_XY, TIMES)
    rng = np.random.default_rng(noise_seed)
    obs = []
    for w, xy in enumerate(WELL_XY):
        for ti, t in enumerate(TIMES):
            value = float(clean[ti, w] + rng.normal(0.0, NOISE_STD))
            obs.append(
                Observation(
                    kind="concentration",
                    location=np.array([[float(xy[0]), float(xy[1]), float(xy[2])]]),
                    value=np.array([value]),
                    noise_cov=np.array([NOISE_STD**2]),
                    time=float(t),
                )
            )
    return obs


def _fit(operator: GroundwaterTransportOperator, noise_seed: int):
    observations = _make_observations(operator, noise_seed)
    return invert_source(observations, operator, seed=1)


class SourceInversionTestCase(unittest.TestCase):
    """A known synthetic leak is localized and its calibrated uncertainty is honest."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.operator = _operator()
        cls.posterior = _fit(cls.operator, noise_seed=7)

    def test_true_location_and_rate_covered_over_repeated_noise_draws(self):
        """A 90% CI is expected to miss the truth about 1 draw in 10; require most of several to cover it."""
        truth = np.concatenate([TRUE_LOCATION, [TRUE_RATE]])
        seeds = [7, 11, 23, 42, 99]
        covered = 0
        for seed in seeds:
            posterior = self.posterior if seed == 7 else _fit(self.operator, noise_seed=seed)
            lo, hi = posterior.credible_interval(0.9)
            self.assertTrue(np.all(lo <= hi))
            if np.all((truth >= lo) & (truth <= hi)):
                covered += 1
        self.assertGreaterEqual(covered, 4, msg=f"only {covered}/{len(seeds)} draws covered the truth at 90%")

    def test_posterior_mean_is_reasonably_close_to_truth(self):
        # A sanity bound distinct from the CI check above: the point estimate itself should be in the
        # right neighbourhood, not merely "covered" by a very wide interval.
        truth = np.concatenate([TRUE_LOCATION, [TRUE_RATE]])
        self.assertLess(np.linalg.norm(self.posterior.mean[:2] - truth[:2]), 2.0)
        self.assertLess(abs(self.posterior.mean[2] - truth[2]), 2.0)

    def test_split_conformal_band_covers_fresh_held_out_noise_at_88pct(self):
        """The A3-style recalibrated band must cover *new* draws of the held-out readings' noise --
        not just the single realization used to derive the conformal quantile (that would be circular)."""
        hop = self.posterior.held_out_prediction
        quantile = self.posterior.recalibration.conformal_quantile
        lower = hop.predictive_mean - quantile * hop.predictive_std
        upper = hop.predictive_mean + quantile * hop.predictive_std

        # The true noiseless concentration at the held-out (well, time) points, since the test knows the
        # source exactly; transpose to (well-major, time-minor) to match invert_source's own flattening.
        true_held_flat = simulate_wells(TRUE_LOCATION, TRUE_RATE, TRUE_ONSET, self.operator, HELD_XY, TIMES).T.reshape(
            -1
        )

        rng = np.random.default_rng(4242)
        n_trials = 400
        covered = 0
        total = 0
        for _ in range(n_trials):
            fresh = true_held_flat + rng.normal(0.0, NOISE_STD, size=true_held_flat.shape)
            covered += int(np.sum((fresh >= lower) & (fresh <= upper)))
            total += fresh.size

        coverage = covered / total
        self.assertGreaterEqual(coverage, 0.88, msg=f"empirical held-out coverage {coverage:.3f} < 0.88")


if __name__ == "__main__":
    unittest.main()
