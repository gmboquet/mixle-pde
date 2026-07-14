"""G3 DoD: the Pasquill-Gifford plume matches the textbook centerline formula, and NNLS source
apportionment recovers known per-source rates from a synthetic 3-source, multi-receptor readout.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.dispersion import (
    DispersionOperator,
    _sigma_y,
    _sigma_z,
    apportion_sources,
    gaussian_plume,
)
from mixle_pde.observations import Observation


# --- (a) gaussian_plume vs. the textbook ground-level centerline formula -------------------
def test_gaussian_plume_matches_textbook_ground_level_centerline():
    # Ground-level source (H=0), ground-level centerline receptor (y=z=0): the two ground-
    # reflection image terms coincide and the general formula collapses to the standard
    # textbook Gaussian-plume centerline concentration C = Q / (pi * u * sigma_y * sigma_z)
    # (e.g. Wark, Warner & Davis, *Air Pollution: Its Origin and Control*; Turner's Workbook).
    Q = 100.0  # g/s
    u = 4.0  # m/s
    stability = "D"
    x = np.array([100.0, 500.0, 2000.0, 5000.0])

    got = gaussian_plume(Q, u, stability, x, 0.0, 0.0, H=0.0)
    expected = Q / (np.pi * u * _sigma_y(x, stability) * _sigma_z(x, stability))

    assert np.allclose(got, expected, rtol=5e-2)


@pytest.mark.parametrize("stability", ["A", "B", "C", "D", "E", "F"])
def test_gaussian_plume_centerline_holds_across_all_stability_classes(stability):
    Q, u = 50.0, 3.0
    x = np.array([200.0, 1000.0])
    got = gaussian_plume(Q, u, stability, x, 0.0, 0.0, H=0.0)
    expected = Q / (np.pi * u * _sigma_y(x, stability) * _sigma_z(x, stability))
    assert np.allclose(got, expected, rtol=5e-2)


def test_gaussian_plume_is_zero_upwind_and_positive_downwind():
    Q, u, stability = 10.0, 2.0, "D"
    assert gaussian_plume(Q, u, stability, -50.0, 0.0, 0.0) == 0.0
    assert gaussian_plume(Q, u, stability, 100.0, 0.0, 0.0) > 0.0


def test_gaussian_plume_elevated_release_ground_level_is_lower_than_ground_release():
    # An elevated stack's plume has not yet fully touched down at a modest downwind distance,
    # so its ground-level centerline concentration should be lower than a ground-level release
    # of the same rate at the same distance.
    Q, u, stability, x = 100.0, 4.0, "D", 300.0
    ground = gaussian_plume(Q, u, stability, x, 0.0, 0.0, H=0.0)
    elevated = gaussian_plume(Q, u, stability, x, 0.0, 0.0, H=50.0)
    assert elevated < ground


# --- DispersionOperator: registered, and reduces to a wind-advected adv-diff operator -------
def test_dispersion_operator_registers_and_matches_advection_diffusion():
    from mixle_pde.dynamics import AdvectionDiffusionOperator, make_operator

    op = make_operator("dispersion", wind=2.0, turbulent_diffusivity=0.5, n=25, length=10.0)
    assert isinstance(op, DispersionOperator)
    assert isinstance(op, AdvectionDiffusionOperator)

    reference = AdvectionDiffusionOperator(diffusivity=0.5, velocity=2.0, n=25, length=10.0)
    assert np.allclose(op.operator_matrix(), reference.operator_matrix())


def test_dispersion_operator_wind_vector_uses_its_magnitude():
    op = DispersionOperator(wind=(3.0, 4.0), turbulent_diffusivity=0.2, n=15, length=5.0)
    assert op.velocity == pytest.approx(5.0)  # |[3, 4]| = 5


# --- (b) 3-source synthetic apportionment recovers rates within 10% ------------------------
def _synthetic_receptor_observation(sources, true_rates, wind, *, seed=0):
    speed, stability, direction = wind["speed"], wind["stability"], wind.get("direction", 0.0)
    xs = np.linspace(100.0, 700.0, 5)
    ys = np.linspace(-150.0, 150.0, 5)
    points = np.array([[x, y, 0.0] for x in xs for y in ys])

    theta = np.radians(direction)
    downwind = np.array([np.cos(theta), np.sin(theta)])
    crosswind = np.array([-np.sin(theta), np.cos(theta)])

    truth = np.zeros(points.shape[0])
    for (sx, sy, sh), rate in zip(sources, true_rates, strict=True):
        dx = points[:, 0] - sx
        dy = points[:, 1] - sy
        x_prime = dx * downwind[0] + dy * downwind[1]
        y_prime = dx * crosswind[0] + dy * crosswind[1]
        truth += rate * gaussian_plume(1.0, speed, stability, x_prime, y_prime, points[:, 2], H=sh)

    rng = np.random.default_rng(seed)
    noise_sd = 0.01 * truth.mean()
    values = truth + rng.normal(scale=noise_sd, size=truth.shape[0])
    noise_cov = np.full(truth.shape[0], noise_sd**2)
    obs = Observation(kind="dust_concentration", location=points, value=values, noise_cov=noise_cov)
    return obs


def test_apportion_sources_recovers_3_source_rates_within_10_percent():
    wind = {"speed": 3.0, "stability": "D", "direction": 0.0}
    sources = [(0.0, -100.0, 0.0), (0.0, 0.0, 0.0), (0.0, 100.0, 0.0)]
    true_rates = np.array([50.0, 120.0, 80.0])

    obs = _synthetic_receptor_observation(sources, true_rates, wind, seed=0)
    candidate_sources = [{"location": (sx, sy), "height": sh} for sx, sy, sh in sources]

    posterior = apportion_sources([obs], candidate_sources, wind)

    assert posterior.mean.shape == (3,)
    rel_err = np.abs(posterior.mean - true_rates) / true_rates
    assert np.all(rel_err < 0.10)


def test_apportion_sources_credible_interval_covers_truth_and_is_nonnegative():
    wind = {"speed": 3.0, "stability": "D", "direction": 0.0}
    sources = [(0.0, -100.0, 0.0), (0.0, 0.0, 0.0), (0.0, 100.0, 0.0)]
    true_rates = np.array([50.0, 120.0, 80.0])

    obs = _synthetic_receptor_observation(sources, true_rates, wind, seed=1)
    candidate_sources = [{"location": (sx, sy), "height": sh} for sx, sy, sh in sources]

    posterior = apportion_sources([obs], candidate_sources, wind)
    lo, hi = posterior.credible_interval(0.9)

    assert np.all(lo >= 0.0)
    assert np.all(lo <= true_rates + 1e-6)
    assert np.all(hi >= true_rates - 1e-6)

    rng = np.random.default_rng(2)
    draws = posterior.samples(500, rng)
    assert draws.shape == (500, 3)
    assert np.all(draws >= 0.0)

    dq = posterior.derived_quantity(lambda r: r.sum(axis=-1), 500, np.random.default_rng(3))
    assert dq.prior_dominated is False
    dq_lo, dq_hi = dq.credible_interval(0.9)
    assert dq_lo <= dq_hi
