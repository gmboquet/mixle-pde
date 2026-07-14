"""Decision-quantity UQ coverage (work-plan A5, IC-8 surface).

For each nonlinear decision quantity, build a synthetic truth field and a posterior noisily centered
around it, then check that the returned Monte Carlo distribution's 90% credible interval covers the
truth-evaluated value at close to the nominal rate across many independent synthetic realizations. This
is the calibration property that actually matters for a driller-facing UQ number: not that any single
call "looks reasonable", but that repeated use of the interval covers truth about as often as advertised.
"""

import unittest

import numpy as np

from mixle_pde import decision_quantities as dq
from mixle_pde.latent import Field3D, PosteriorField3D

N_TRIALS = 24
MC_DRAWS = 3000
MIN_COVERAGE = 0.85


def _grid(n_side=4):
    xs = np.linspace(0.0, 30.0, n_side)
    ys = np.linspace(0.0, 30.0, n_side)
    pts = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    return Field3D(coordinates=pts, spacing=10.0, units="frac", property_name="saturation", bounds=None)


def _posterior_around_truth(grid, truth, sigma, rng):
    """A Gaussian posterior whose mean is a noisy (never-exact) view of `truth`, diagonal covariance."""
    noise = rng.normal(scale=sigma, size=truth.shape)
    mean = truth + noise
    var = np.full(truth.shape, sigma**2)
    return PosteriorField3D(grid=grid, mean=mean, diag_var=var)


class ProbExceedCoverageTest(unittest.TestCase):
    def test_ninety_percent_interval_covers_truth_at_advertised_rate(self):
        grid = _grid()
        region = np.zeros(grid.n, dtype=bool)
        region[: grid.n // 2] = True
        threshold = 0.5
        rng = np.random.default_rng(1)
        hits = 0
        for _ in range(N_TRIALS):
            truth = rng.uniform(0.2, 0.8, size=grid.n)
            posterior = _posterior_around_truth(grid, truth, sigma=0.08, rng=rng)
            truth_value = float(np.mean(truth[region] > threshold))
            result = dq.prob_exceed(posterior, region, threshold=threshold, n=MC_DRAWS, rng=rng)
            lo, hi = result.credible_interval(0.9)
            if lo <= truth_value <= hi:
                hits += 1
        self.assertGreaterEqual(hits / N_TRIALS, MIN_COVERAGE)


class TonnageAboveCutoffCoverageTest(unittest.TestCase):
    def test_ninety_percent_interval_covers_truth_at_advertised_rate(self):
        grid = _grid()
        region = np.zeros(grid.n, dtype=bool)
        region[: grid.n // 2] = True
        cell_volumes = np.full(grid.n, 100.0)
        cutoff = 1.5
        rng = np.random.default_rng(2)
        hits = 0
        for _ in range(N_TRIALS):
            truth = rng.uniform(0.5, 3.0, size=grid.n)
            posterior = _posterior_around_truth(grid, truth, sigma=0.2, rng=rng)
            above = truth[region] > cutoff
            truth_value = float(np.sum(cell_volumes[region][above] * truth[region][above]))
            result = dq.tonnage_above_cutoff(posterior, region, cell_volumes, cutoff=cutoff, n=MC_DRAWS, rng=rng)
            lo, hi = result.credible_interval(0.9)
            if lo <= truth_value <= hi:
                hits += 1
        self.assertGreaterEqual(hits / N_TRIALS, MIN_COVERAGE)


class NetPayCoverageTest(unittest.TestCase):
    def test_ninety_percent_interval_covers_truth_at_advertised_rate(self):
        grid = _grid()
        column_index = np.arange(grid.n // 2)
        thickness = np.full(column_index.shape, 2.5)
        sat_cut = 0.5
        rng = np.random.default_rng(3)
        hits = 0
        for _ in range(N_TRIALS):
            truth = rng.uniform(0.2, 0.8, size=grid.n)
            posterior = _posterior_around_truth(grid, truth, sigma=0.08, rng=rng)
            above = truth[column_index] >= sat_cut
            truth_value = float(np.sum(thickness[above]))
            result = dq.net_pay(posterior, column_index, sat_cut=sat_cut, thickness=thickness, n=MC_DRAWS, rng=rng)
            lo, hi = result.credible_interval(0.9)
            if lo <= truth_value <= hi:
                hits += 1
        self.assertGreaterEqual(hits / N_TRIALS, MIN_COVERAGE)


class DrillTargetProbCoverageTest(unittest.TestCase):
    def test_ninety_percent_interval_covers_truth_at_advertised_rate(self):
        grid = _grid()
        region = np.zeros(grid.n, dtype=bool)
        region[: grid.n // 2] = True

        def criteria(region_values):
            return region_values > 0.5

        rng = np.random.default_rng(4)
        hits = 0
        for _ in range(N_TRIALS):
            truth = rng.uniform(0.2, 0.8, size=grid.n)
            posterior = _posterior_around_truth(grid, truth, sigma=0.08, rng=rng)
            truth_value = float(np.mean(criteria(truth[region])))
            result = dq.drill_target_prob(posterior, region, criteria=criteria, n=MC_DRAWS, rng=rng)
            lo, hi = result.credible_interval(0.9)
            if lo <= truth_value <= hi:
                hits += 1
        self.assertGreaterEqual(hits / N_TRIALS, MIN_COVERAGE)


class PriorDominatedFlagTest(unittest.TestCase):
    """No IC-8 number ships unflagged (work-plan A2 hook, algorithm step 6)."""

    def test_flag_defaults_false_without_prior_posterior_variance(self):
        grid = _grid()
        region = np.ones(grid.n, dtype=bool)
        rng = np.random.default_rng(5)
        truth = rng.uniform(0.2, 0.8, size=grid.n)
        posterior = _posterior_around_truth(grid, truth, sigma=0.08, rng=rng)
        result = dq.prob_exceed(posterior, region, threshold=0.5, n=512, rng=rng)
        self.assertFalse(result.prior_dominated)

    def test_flag_true_when_reduction_below_threshold(self):
        grid = _grid()
        region = np.ones(grid.n, dtype=bool)
        rng = np.random.default_rng(6)
        truth = rng.uniform(0.2, 0.8, size=grid.n)
        posterior = _posterior_around_truth(grid, truth, sigma=0.08, rng=rng)
        prior_var = np.full(grid.n, 1.0)
        posterior_var = np.full(grid.n, 0.95)  # reduction ~ 0.05 < 0.1 threshold
        result = dq.prob_exceed(
            posterior, region, threshold=0.5, n=512, rng=rng, prior_var=prior_var, posterior_var=posterior_var
        )
        self.assertTrue(result.prior_dominated)

    def test_flag_false_when_reduction_above_threshold(self):
        grid = _grid()
        region = np.ones(grid.n, dtype=bool)
        rng = np.random.default_rng(7)
        truth = rng.uniform(0.2, 0.8, size=grid.n)
        posterior = _posterior_around_truth(grid, truth, sigma=0.08, rng=rng)
        prior_var = np.full(grid.n, 1.0)
        posterior_var = np.full(grid.n, 0.1)  # reduction ~ 0.9 > 0.1 threshold
        result = dq.prob_exceed(
            posterior, region, threshold=0.5, n=512, rng=rng, prior_var=prior_var, posterior_var=posterior_var
        )
        self.assertFalse(result.prior_dominated)


class RegionMassStaysExactTest(unittest.TestCase):
    """Non-goal: region_mass is a linear functional and stays exact via posterior_query."""

    def test_region_mass_matches_posterior_query_moments(self):
        from mixle_pde.posterior_query import region_mass as exact_region_mass

        grid = _grid()
        region = np.zeros(grid.n, dtype=bool)
        region[: grid.n // 2] = True
        cell_volumes = np.full(grid.n, 10.0)
        rng = np.random.default_rng(8)
        mean = rng.normal(size=grid.n) * 3.0
        var = np.full(grid.n, 0.5)
        posterior = PosteriorField3D(grid=grid, mean=mean, diag_var=var)

        expected = exact_region_mass(posterior, region, cell_volumes)
        got = dq.region_mass(posterior, region, cell_volumes)

        self.assertAlmostEqual(got.mean, expected.mean)
        self.assertAlmostEqual(got.std, expected.std)
        self.assertFalse(got.prior_dominated)


if __name__ == "__main__":
    unittest.main()
