"""Validation of the basin thermal-history forwards against textbook physics and the EASY%Ro calibration."""
import numpy as np
import pytest

from mixle_pde.basin import easy_ro, easy_ro_profile, geotherm, EASYRO_WEIGHTS


def test_geotherm_linear_without_heat_production():
    # constant conductivity, no production -> exactly linear geotherm at gradient q/k
    dz = np.full(30, 100.0)
    depth, temp = geotherm(dz, conductivity=2.5, surface_temp=10.0, surface_heat_flow=0.060)
    grad = (temp[-1] - 10.0) / (depth[-1] / 1000.0)  # deg C / km
    assert depth[-1] == pytest.approx(3000.0)
    assert grad == pytest.approx(0.060 / 2.5 * 1000.0, rel=1e-6)  # 24 C/km
    # exactly linear in depth
    assert np.allclose(np.diff(temp), np.diff(temp)[0])


def test_geotherm_heat_production_curves_concave():
    dz = np.full(30, 100.0)
    _, t0 = geotherm(dz, conductivity=2.5, surface_heat_flow=0.060, heat_production=0.0)
    _, th = geotherm(dz, conductivity=2.5, surface_heat_flow=0.060, heat_production=2.5e-6)
    # production draws down deep heat flow -> deep gradient shallower -> concave (gradient decreases with depth)
    g = np.diff(th)
    assert np.all(np.diff(g) < 0)
    assert th[-1] < t0[-1]  # less heat reaches depth than the no-production column predicts at base flux


def test_easy_ro_endpoints():
    # absolute floor exp(-1.6) = 0.20 % is reached only at negligible exposure
    floor = easy_ro(np.array([0.0, 1e-3]), np.array([5.0, 5.0]))
    assert floor == pytest.approx(np.exp(-1.6), abs=1e-3)  # 0.202 %
    # a cold horizon held 100 Myr is still immature, but the low-E reactions creep it slightly above the floor
    cold = easy_ro(np.array([0.0, 100.0]), np.array([5.0, 5.0]))
    assert np.exp(-1.6) <= cold < 0.30
    # fully cooked limit -> exp(-1.6 + 3.7 * sum(weights)) = 4.69 %
    hot = easy_ro(np.linspace(0, 500, 5000), np.full(5000, 300.0))
    assert hot == pytest.approx(np.exp(-1.6 + 3.7 * EASYRO_WEIGHTS.sum()), rel=2e-3)  # 4.69 %


def test_easy_ro_oil_window_calibration():
    # constant heating; oil-window onset (Ro ~ 0.6 %) should fall near 100-115 C, the published EASY%Ro behaviour
    def ro_at(tmax, rate):
        dur = (tmax - 10.0) / rate
        t = np.linspace(0.0, dur, max(int(dur), 50))
        return easy_ro(t, 10.0 + rate * t)
    for rate, expect in [(1.0, 99.0), (3.0, 105.0), (10.0, 114.0)]:
        tmaxes = np.arange(60.0, 200.0, 1.0)
        ros = np.array([ro_at(tm, rate) for tm in tmaxes])
        onset = tmaxes[np.argmin(np.abs(ros - 0.6))]
        assert onset == pytest.approx(expect, abs=4.0)
    # maturity is monotone in temperature
    r = [ro_at(tm, 3.0) for tm in (80, 120, 160, 200)]
    assert all(np.diff(r) > 0)


def test_easy_ro_profile_matches_scalar():
    t = np.linspace(0, 80, 400)
    temps = np.vstack([10 + 1.5 * t, 10 + 2.5 * t])
    prof = easy_ro_profile(t, temps)
    assert prof[0] == pytest.approx(easy_ro(t, temps[0]))
    assert prof[1] > prof[0]  # hotter horizon is more mature
