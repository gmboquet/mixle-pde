"""DoD test for workstream B4 -- potential-field grid ingest + reductions.

`mag_grid.tif` is a synthetic 32x32 aeromagnetic (TMI) grid: a single induced dipole beneath the
grid centre, observed at the survey's true field (inclination=60, declination=0). At that
inclination the raw anomaly peak is skewed off the true source location; reducing to the pole at
the same inclination/declination should pull the peak back toward the grid centre.
"""

import os

import numpy as np

from mixle_pde.io.potfield import load_grid
from mixle_pde.reductions import reduce_to_pole, upward_continue

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mag_grid.tif")


def _peak_index(values: np.ndarray) -> tuple[int, int]:
    return np.unravel_index(np.argmax(np.abs(values)), values.shape)


def _dist_to_centre(index: tuple[int, int], shape: tuple[int, int]) -> float:
    centre = (shape[0] // 2, shape[1] // 2)
    return float(np.hypot(index[0] - centre[0], index[1] - centre[1]))


def test_load_grid_shape_and_framing():
    grid = load_grid(FIXTURE)
    assert grid.values.shape == (32, 32)
    assert grid.x.shape == (32,)
    assert grid.y.shape == (32,)
    assert grid.crs is not None and "32611" in grid.crs
    assert len(grid.transform) == 6
    assert np.all(np.isfinite(grid.values))


def test_reduce_to_pole_recentres_the_anomaly_peak():
    grid = load_grid(FIXTURE)
    raw_peak = _peak_index(grid.values)

    rtp = reduce_to_pole(grid, inclination=60.0, declination=0.0)

    assert rtp.shape == grid.values.shape
    assert np.all(np.isfinite(rtp))

    rtp_peak = _peak_index(rtp)
    raw_dist = _dist_to_centre(raw_peak, grid.values.shape)
    rtp_dist = _dist_to_centre(rtp_peak, rtp.shape)
    assert rtp_dist < raw_dist


def test_upward_continue_preserves_shape_and_attenuates():
    grid = load_grid(FIXTURE)
    continued = upward_continue(grid, 200.0)

    assert continued.shape == grid.values.shape
    assert np.all(np.isfinite(continued))
    # continuing upward smooths/attenuates the anomaly: its peak-to-peak range shrinks.
    assert np.ptp(continued) < np.ptp(grid.values)
