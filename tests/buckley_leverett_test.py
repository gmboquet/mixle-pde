"""Tests for mixle_pde.buckley_leverett (MP-H6): the two-phase immiscible-displacement solver.

Two independent lines of evidence, checked against closed-form theory rather than against each other:

1. :func:`welge_shock_front` (pure algebra on the fractional-flow curve, no PDE solve) is checked
   against its own defining property -- the Rankine-Hugoniot jump condition -- and against genuine
   tangency (the secant line must not cross the curve anywhere on the range), both to machine precision.
2. :func:`solve_buckley_leverett_upwind` (the finite-difference march) is checked against
   :func:`buckley_leverett_analytic` (the closed-form self-similar solution built from the same Welge
   shock via the method of characteristics): an L1-norm profile comparison, an exact discrete mass-
   balance identity, a shock-location match, and a resolution sweep confirming the L1 error genuinely
   shrinks (not a coincidence at one grid size) at close to the expected first-order rate.

Physical scenario: water (1 cP) displacing oil (5 cP) through a homogeneous column, Corey quadratic
relative permeabilities, connate water / residual oil saturation both 0.2 -- an unremarkable textbook
waterflood, chosen only so the classic S-shaped fractional-flow curve (and hence a genuine shock) exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.buckley_leverett import (
    CoreyFractionalFlow,
    buckley_leverett_analytic,
    solve_buckley_leverett_upwind,
    welge_shock_front,
)

_MODEL = CoreyFractionalFlow(
    water_viscosity=1.0,
    oil_viscosity=5.0,
    connate_water_saturation=0.2,
    residual_oil_saturation=0.2,
)
_VELOCITY = 1.0
_POROSITY = 0.2
_LENGTH = 10.0


def _pre_breakthrough_time(model: CoreyFractionalFlow, length: float, velocity: float, porosity: float) -> float:
    """A time at which the analytic shock sits at 60% of the domain (comfortably pre-breakthrough)."""
    shock = welge_shock_front(model)
    u = velocity / porosity
    return 0.6 * length / (u * shock.speed)


# --------------------------------------------------------------------------- fractional-flow model


def test_fractional_flow_is_monotone_and_bounded():
    s = np.linspace(0.2, 0.8, 2000)
    f = _MODEL.fractional_flow(s)
    assert f.min() >= 0.0 - 1e-12
    assert f.max() <= 1.0 + 1e-12
    assert np.all(np.diff(f) >= -1e-12)  # monotonically non-decreasing
    assert _MODEL.fractional_flow(0.2) == pytest.approx(0.0, abs=1e-12)
    assert _MODEL.fractional_flow(0.8) == pytest.approx(1.0, abs=1e-9)


def test_fractional_flow_derivative_matches_finite_difference():
    s = np.linspace(0.21, 0.79, 500)
    eps = 1e-6
    numeric = (_MODEL.fractional_flow(s + eps) - _MODEL.fractional_flow(s - eps)) / (2 * eps)
    analytic = _MODEL.fractional_flow_derivative(s)
    assert np.max(np.abs(numeric - analytic)) < 1e-6


def test_fractional_flow_derivative_vanishes_at_endpoints():
    # Both Corey exponents are 2 (> 1), so f' -> 0 at both S_wc and 1 - S_or (a genuine S-shaped curve).
    assert _MODEL.fractional_flow_derivative(0.2) == pytest.approx(0.0, abs=1e-9)
    assert _MODEL.fractional_flow_derivative(0.8) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"water_viscosity": 0.0, "oil_viscosity": 1.0},
        {"water_viscosity": 1.0, "oil_viscosity": -1.0},
        {"water_viscosity": 1.0, "oil_viscosity": 1.0, "water_exponent": 0.5},
        {"water_viscosity": 1.0, "oil_viscosity": 1.0, "connate_water_saturation": 0.6, "residual_oil_saturation": 0.5},
        {"water_viscosity": 1.0, "oil_viscosity": 1.0, "connate_water_saturation": 1.2},
    ],
)
def test_corey_fractional_flow_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        CoreyFractionalFlow(**kwargs)


# --------------------------------------------------------------------------- Welge tangent construction


def test_welge_shock_satisfies_rankine_hugoniot_to_machine_precision():
    shock = welge_shock_front(_MODEL)
    s_init = _MODEL.connate_water_saturation
    secant_speed = (_MODEL.fractional_flow(shock.saturation) - _MODEL.fractional_flow(s_init)) / (
        shock.saturation - s_init
    )
    assert shock.speed == pytest.approx(float(secant_speed), abs=1e-8)
    assert shock.speed == pytest.approx(float(_MODEL.fractional_flow_derivative(shock.saturation)), abs=1e-10)


def test_welge_shock_is_a_genuine_tangent_not_an_arbitrary_root():
    """The line from (S_wc, 0) through the shock point must not cross ABOVE the f(S) curve anywhere
    on [S_wc, 1 - S_or] -- i.e. it touches, it does not merely intersect. A false root of the tangent
    equation (e.g. a spurious secant crossing) would fail this."""
    shock = welge_shock_front(_MODEL)
    s_init = _MODEL.connate_water_saturation
    s = np.linspace(s_init, 1.0 - _MODEL.residual_oil_saturation, 5000)
    line = shock.speed * (s - s_init)
    curve = _MODEL.fractional_flow(s)
    assert np.min(line - curve) > -1e-6  # the line stays at/above the curve everywhere


def test_welge_shock_saturation_lies_strictly_between_endpoints():
    shock = welge_shock_front(_MODEL)
    assert _MODEL.connate_water_saturation < shock.saturation < 1.0 - _MODEL.residual_oil_saturation


def test_welge_shock_front_rejects_bad_initial_saturation():
    with pytest.raises(ValueError):
        welge_shock_front(_MODEL, initial_saturation=1.0)  # >= 1 - S_or


# --------------------------------------------------------------------------- numerical vs. analytic


def test_upwind_solver_matches_analytic_profile_pre_breakthrough():
    t_end = _pre_breakthrough_time(_MODEL, _LENGTH, _VELOCITY, _POROSITY)
    result = solve_buckley_leverett_upwind(
        _MODEL, length=_LENGTH, n_cells=800, velocity=_VELOCITY, porosity=_POROSITY, time=t_end
    )
    analytic = buckley_leverett_analytic(_MODEL, result.positions, t_end, velocity=_VELOCITY, porosity=_POROSITY)

    # Profile-wide agreement (L1 -- the appropriate norm for a first-order shock-capturing scheme; a
    # pointwise/L-infinity comparison would be dominated by the few cells straddling the discontinuity,
    # where a first-order scheme is *expected* to disagree with a true jump by construction).
    l1_error = float(np.mean(np.abs(result.saturation - analytic.saturation)))
    assert l1_error < 0.005

    # The numerical solution must stay within the physical bounds (maximum principle for a monotone
    # upwind scheme under the CFL condition): no overshoot/undershoot past the injected or initial value.
    assert result.saturation.min() >= _MODEL.connate_water_saturation - 1e-9
    assert result.saturation.max() <= (1.0 - _MODEL.residual_oil_saturation) + 1e-9

    # Shock location: estimate it from the numerical profile's steepest gradient and compare to the
    # analytic Rankine-Hugoniot front position; the estimate is resolution-limited (~a cell width).
    dx = _LENGTH / 800
    steepest = int(np.argmax(np.abs(np.diff(result.saturation))))
    x_shock_numeric = 0.5 * (result.positions[steepest] + result.positions[steepest + 1])
    assert abs(x_shock_numeric - analytic.shock_position) < 3.0 * dx

    # Exact discrete mass balance: for a conservative finite-volume scheme with no outflow yet
    # (pre-breakthrough), the extra water stored in the column must equal the cumulative influx exactly,
    # not just approximately -- this checks the scheme itself, independent of the analytic reference.
    stored = float(np.sum(result.saturation - _MODEL.connate_water_saturation)) * dx * _POROSITY
    influx = _VELOCITY * result.time * _MODEL.fractional_flow(1.0 - _MODEL.residual_oil_saturation)
    assert stored == pytest.approx(influx, rel=1e-10)


def test_l1_error_shrinks_with_resolution():
    """Refute a coincidental one-resolution match: the L1 error against the analytic profile must
    genuinely decrease as the grid is refined, at close to a first-order rate (halving resolution
    should roughly halve the error for a first-order upwind scheme; a factor > 1.3 per doubling is a
    safe, non-flaky lower bound well under the ~1.7-1.8 actually observed)."""
    t_end = _pre_breakthrough_time(_MODEL, _LENGTH, _VELOCITY, _POROSITY)
    errors = []
    for n_cells in (200, 400, 800, 1600):
        result = solve_buckley_leverett_upwind(
            _MODEL, length=_LENGTH, n_cells=n_cells, velocity=_VELOCITY, porosity=_POROSITY, time=t_end
        )
        analytic = buckley_leverett_analytic(_MODEL, result.positions, t_end, velocity=_VELOCITY, porosity=_POROSITY)
        errors.append(float(np.mean(np.abs(result.saturation - analytic.saturation))))

    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    for earlier, later in zip(errors, errors[1:]):
        assert earlier / later > 1.3


def test_analytic_profile_boundary_and_rarefaction_values():
    shock = welge_shock_front(_MODEL)
    u = _VELOCITY / _POROSITY
    t_end = _pre_breakthrough_time(_MODEL, _LENGTH, _VELOCITY, _POROSITY)
    x_shock = u * shock.speed * t_end
    positions = np.array([0.0, 0.5 * x_shock, _LENGTH])  # injector, mid-rarefaction, undisturbed far field
    analytic = buckley_leverett_analytic(_MODEL, positions, t_end, velocity=_VELOCITY, porosity=_POROSITY)

    assert analytic.saturation[0] == pytest.approx(1.0 - _MODEL.residual_oil_saturation, abs=1e-6)
    assert analytic.saturation[-1] == pytest.approx(_MODEL.connate_water_saturation, abs=1e-12)
    # mid-rarefaction sits strictly between the shock and injected saturations
    assert shock.saturation < analytic.saturation[1] < 1.0 - _MODEL.residual_oil_saturation


# --------------------------------------------------------------------------- validation / error paths


def test_solver_rejects_injection_at_or_below_initial_saturation():
    with pytest.raises(ValueError):
        solve_buckley_leverett_upwind(
            _MODEL,
            length=_LENGTH,
            n_cells=100,
            velocity=_VELOCITY,
            porosity=_POROSITY,
            time=0.1,
            injection_saturation=0.2,  # == connate_water_saturation
        )


def test_solver_rejects_bad_cfl():
    with pytest.raises(ValueError):
        solve_buckley_leverett_upwind(
            _MODEL, length=_LENGTH, n_cells=100, velocity=_VELOCITY, porosity=_POROSITY, time=0.1, cfl=1.5
        )


def test_solver_rejects_too_few_cells():
    with pytest.raises(ValueError):
        solve_buckley_leverett_upwind(
            _MODEL, length=_LENGTH, n_cells=1, velocity=_VELOCITY, porosity=_POROSITY, time=0.1
        )


def test_analytic_rejects_injection_saturation_below_shock():
    shock = welge_shock_front(_MODEL)
    with pytest.raises(ValueError):
        buckley_leverett_analytic(
            _MODEL,
            np.linspace(0.0, _LENGTH, 10),
            0.1,
            velocity=_VELOCITY,
            porosity=_POROSITY,
            injection_saturation=shock.saturation - 0.05,
        )
