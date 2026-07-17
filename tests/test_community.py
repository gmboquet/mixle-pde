"""Community exposure pathway coverage (work-plan K2, IC-1, IC-4).

``receptor_truth.csv`` is a static Pasquill-Gifford class-D ground-level Gaussian-plume reference
(Q=8.0 g/s, u=3.0 m/s, H=0) at a single downstream community receptor -- the "transport reference" the
Definition-of-Done pins against. Neither G3's `dispersion.gaussian_plume` nor G1's
`groundwater.GroundwaterTransportOperator` had landed in this branch when this test was written, so the
same analytic formula is used here, once, purely to build a synthetic `ConcentrationField` grid and an
independent truth value -- exactly the role `dispersion.py`/`groundwater.py` will play once they exist;
this test does not add a transport operator to the production module (K2's own non-goal).
"""

from __future__ import annotations

import csv
import os

import numpy as np
from mixle.reason.posterior_protocol import Posterior

from mixle_pde.exposure_pathways import ConcentrationField, couple_pathways, receptor_exposure

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "receptor_truth.csv")

# Pasquill-Gifford class-D (neutral, rural) Briggs sigma coefficients -- textbook reference dispersion,
# standing in for G3's `gaussian_plume` (not yet landed).
_Q = 8.0  # g/s emission rate
_U = 3.0  # m/s wind speed along +x
_H = 0.0  # ground-level release


def _sigma_y(x: np.ndarray) -> np.ndarray:
    return 0.08 * x * (1.0 + 0.0001 * x) ** -0.5


def _sigma_z(x: np.ndarray) -> np.ndarray:
    return 0.06 * x * (1.0 + 0.0015 * x) ** -0.5


def _gaussian_plume_reference(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    sy = _sigma_y(np.asarray(x, dtype=float))
    sz = _sigma_z(np.asarray(x, dtype=float))
    horiz = np.exp(-(np.asarray(y, dtype=float) ** 2) / (2.0 * sy**2))
    vert = np.exp(-((z - _H) ** 2) / (2.0 * sz**2)) + np.exp(-((z + _H) ** 2) / (2.0 * sz**2))
    return _Q / (2.0 * np.pi * _U * sy * sz) * horiz * vert


def _read_truth():
    with open(FIXTURE, newline="") as f:
        row = next(csv.DictReader(f))
    receptor = np.array([[float(row["x"]), float(row["y"]), float(row["z"])]])
    return receptor, float(row["concentration_g_m3"])


def _air_field():
    """A ground-level (z=0) plume grid dense enough that linear interpolation to an off-grid receptor
    is well within the 5% Definition-of-Done tolerance."""
    xs = np.arange(50.0, 1000.0 + 1.0, 25.0)
    ys = np.arange(-300.0, 300.0 + 1.0, 10.0)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack([xg.ravel(), yg.ravel(), np.zeros(xg.size)])
    values = _gaussian_plume_reference(coords[:, 0], coords[:, 1], coords[:, 2])
    return ConcentrationField(coordinates=coords, values=values, units="g/m3")


class _GaussianFieldPosterior:
    """A minimal IC-1 `Posterior` over a field's node values: an independent Gaussian per node centered
    on `mean` with a small relative std -- enough to exercise `receptor_exposure`'s UQ-propagation path
    without depending on a real G3/G1 inversion."""

    def __init__(self, mean: np.ndarray, rel_std: float):
        self.mean_ = np.asarray(mean, dtype=float)
        self.std_ = np.abs(self.mean_) * rel_std

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.mean_[None, :] + self.std_[None, :] * rng.standard_normal((n, self.mean_.size))

    @property
    def mean(self) -> np.ndarray:
        return self.mean_

    @property
    def cov(self) -> np.ndarray:
        return np.diag(self.std_**2)

    def credible_interval(self, level: float):
        from scipy.special import ndtri

        z = ndtri(0.5 + level / 2.0)
        return self.mean_ - z * self.std_, self.mean_ + z * self.std_

    def derived_quantity(self, fn, n, rng):
        draws = fn(self.samples(n, rng))

        class _DQ:
            def __init__(self, samples):
                self.samples = samples
                self.prior_dominated = False

            def credible_interval(self, level):
                alpha = 1.0 - level
                lo, hi = np.quantile(self.samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
                return lo, hi

        return _DQ(draws)


def _water_field(coords: np.ndarray):
    """A simple ground-level solute field decaying with downstream distance -- a stand-in for a G1
    `GroundwaterTransportOperator` state, distinct in shape/decay from the air field on purpose."""
    values = 0.02 * np.exp(-coords[:, 0] / 400.0) * np.exp(-(coords[:, 1] ** 2) / (2.0 * 150.0**2))
    return ConcentrationField(coordinates=coords, values=values, units="mg/L")


def test_downstream_receptor_matches_transport():
    receptor, truth = _read_truth()
    field = _air_field()

    result = receptor_exposure(field, receptor, pathway="air")

    assert result.concentration.shape == (1,)
    rel_err = abs(result.concentration[0] - truth) / truth
    assert rel_err < 0.05, f"relative error {rel_err:.4f} exceeds 5% tolerance"
    assert result.ci is None
    assert result.provenance["source_content_hash"] == field.content_hash
    assert result.provenance["pathway"] == "air"


def test_receptor_exposure_propagates_posterior_to_credible_interval():
    field = _air_field()
    posterior = _GaussianFieldPosterior(field.values, rel_std=0.1)
    field.posterior = posterior
    assert isinstance(posterior, Posterior)

    receptor = np.array([[620.0, 15.0, 0.0]])
    rng = np.random.default_rng(0)
    result = receptor_exposure(field, receptor, n_samples=2000, rng=rng, credible_level=0.9)

    assert result.ci is not None
    lo, hi = result.ci
    assert np.all(lo <= result.concentration)
    assert np.all(result.concentration <= hi)
    assert result.provenance["n_samples"] == 2000
    assert "prior_dominated" in result.provenance


def test_couple_pathways_combines_air_and_water_by_intake_weights():
    air = _air_field()
    water = _water_field(air.coordinates)
    receptors = np.array([[620.0, 15.0, 0.0], [800.0, -50.0, 0.0]])
    weights = {"air": 20.0, "water": 2.0}

    result = couple_pathways(air, water, receptors, weights=weights)

    expected_air = receptor_exposure(air, receptors).concentration
    expected_water = receptor_exposure(water, receptors).concentration
    expected = weights["air"] * expected_air + weights["water"] * expected_water
    np.testing.assert_allclose(result.concentration, expected, rtol=1e-8)
    assert result.provenance["air_content_hash"] == air.content_hash
    assert result.provenance["water_content_hash"] == water.content_hash
    assert result.ci is None


def test_couple_pathways_propagates_combined_uncertainty():
    air = _air_field()
    water = _water_field(air.coordinates)
    air.posterior = _GaussianFieldPosterior(air.values, rel_std=0.15)
    water.posterior = _GaussianFieldPosterior(water.values, rel_std=0.15)

    receptors = np.array([[620.0, 15.0, 0.0]])
    rng = np.random.default_rng(1)
    result = couple_pathways(air, water, receptors, weights={"air": 20.0, "water": 2.0}, n_samples=1500, rng=rng)

    assert result.ci is not None
    lo, hi = result.ci
    assert np.all(lo <= result.concentration)
    assert np.all(result.concentration <= hi)
