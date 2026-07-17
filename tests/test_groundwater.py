"""G1 DoD: darcy_velocity + GroundwaterTransportOperator reproduce the Ogata-Banks plume.

The scenario is a genuine two-piece "pumping + tracer" synthetic, matching the work-plan
wording, not a hand-set uniform velocity:

1. A coarse 1-D Darcy flow grid with an injection well and a pumping well (equal and
   opposite ``source`` terms) drives :func:`darcy_velocity`. Between the two wells the
   resulting specific discharge is (by 1-D flux conservation) exactly uniform -- this is
   verified, not assumed -- and its value is read off numerically rather than hard-coded.
2. That uniform velocity feeds a much finer :class:`GroundwaterTransportOperator` column
   (flow and transport need not share a grid -- the pumping test only needs to be coarse
   enough to be cheap; the ADE needs fine enough spacing to keep first-order-upwind
   numerical dispersion a small fraction of the physical dispersion). A continuously held
   ``c=c0`` boundary at the injection end is driven *exactly* (no time-step boundary-reset
   error) by solving the transport operator's own linear ODE ``du/dt = G u`` as an affine
   system in the interior state with the pinned boundary as a constant forcing term, via one
   matrix-exponential per requested time -- isolating the spatial discretization (the actual
   deliverable) from any time-integration artifact.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from mixle_pde.dynamics import available_dynamics_operators, make_operator
from mixle_pde.groundwater import (
    GroundwaterTransportOperator,
    darcy_velocity,
    ogata_banks_plume,
)


def _uniform_pumping_velocity() -> float:
    """Solve a 1-D injection/extraction pumping test and read off the (exactly uniform,
    by 1-D flux conservation) interior specific discharge between the two wells."""
    n_flow = 201
    i_inj, i_ext = 20, 180
    source = np.zeros(n_flow)
    source[i_inj] = 10.0
    source[i_ext] = -10.0
    q = darcy_velocity(1.0, source, (n_flow,), spacing=1.0)
    interior = q[0, 60:141]
    assert interior.std() < 1e-8 * max(abs(interior.mean()), 1.0), "pumping test should yield a uniform interior flux"
    return float(interior.mean())


def _continuous_injection_profile(operator: GroundwaterTransportOperator, c0: float, t: float) -> np.ndarray:
    """Exact concentration profile at time ``t`` for a column held at ``c0`` at ``x=0`` from
    ``t=0`` (state[0] fixed forever after), zero elsewhere initially -- solved as an affine
    ODE (the pinned boundary is a constant forcing on the interior states) via the augmented-
    matrix trick, so there is no boundary-reset time-step error at all, only the spatial
    discretization's own truncation error."""
    g = operator.operator_matrix()
    a = g[1:, 1:]
    b = g[1:, 0] * c0
    n_i = a.shape[0]
    m = np.zeros((n_i + 1, n_i + 1))
    m[:n_i, :n_i] = a
    m[:n_i, n_i] = b
    w0 = np.zeros(n_i + 1)
    w0[-1] = 1.0
    w = expm(m * t) @ w0
    return np.concatenate([[c0], w[:n_i]])


class _OgataBanksColumn:
    """A fine 1-D transport column driven by a pumping-derived uniform velocity."""

    def __init__(self) -> None:
        self.velocity = _uniform_pumping_velocity()
        self.spacing = 0.05
        self.length = 40.0
        self.n = int(round(self.length / self.spacing)) + 1
        self.dispersivity = 0.5
        self.dispersion = self.dispersivity * abs(self.velocity)
        velocity_field = np.full((1, self.n), self.velocity)
        self.operator = GroundwaterTransportOperator(
            velocity_field, self.dispersivity, (self.n,), spacing=self.spacing, scheme="exact"
        )
        self.x = np.arange(self.n) * self.spacing
        self.c0 = 1.0


@pytest.fixture(scope="module")
def column() -> _OgataBanksColumn:
    return _OgataBanksColumn()


def test_pumping_tracer_plume_matches_ogata_banks(column: _OgataBanksColumn) -> None:
    for t in (4.0, 8.0, 12.0):
        sim = _continuous_injection_profile(column.operator, column.c0, t)
        analytic = ogata_banks_plume(column.x, t, column.velocity, column.dispersion, c0=column.c0)
        np.testing.assert_allclose(sim, analytic, rtol=5e-2, atol=3e-3)


def test_plume_front_advances_with_time(column: _OgataBanksColumn) -> None:
    # sanity check independent of the analytic formula: the half-height point of the plume
    # should move downstream roughly at the pumping-derived velocity.
    def half_height_position(t: float) -> float:
        c = _continuous_injection_profile(column.operator, column.c0, t)
        below = np.where(c < 0.5 * column.c0)[0]
        return float(column.x[below[0]]) if below.size else float(column.x[-1])

    x4, x12 = half_height_position(4.0), half_height_position(12.0)
    assert x12 > x4
    assert abs((x12 - x4) / (12.0 - 4.0) - column.velocity) < 0.35 * column.velocity


def test_darcy_velocity_pumping_test_is_uniform_between_wells() -> None:
    v = _uniform_pumping_velocity()
    assert v > 0.0


def test_darcy_velocity_shape_and_neumann_edges() -> None:
    n = 41
    source = np.zeros(n)
    source[n // 2] = 1.0
    q = darcy_velocity(1.0, source, (n,), spacing=1.0, bc="neumann")
    assert q.shape == (1, n)
    assert q[0, 0] == 0.0 and q[0, -1] == 0.0  # neumann: zero outward-normal flux at the edges


def test_darcy_velocity_non_neumann_edges_are_not_forced_zero() -> None:
    n = 41
    source = np.zeros(n)
    source[5] = 1.0
    source[35] = -1.0
    q = darcy_velocity(1.0, source, (n,), spacing=1.0, bc="dirichlet")
    assert q[0, 0] != 0.0 or q[0, -1] != 0.0


def test_groundwater_registered_and_make_operator_builds_it() -> None:
    assert "groundwater" in available_dynamics_operators()
    n = 11
    op = make_operator(
        "groundwater",
        velocity_field=np.full((1, n), 1.0),
        dispersivity=0.2,
        shape=(n,),
        spacing=1.0,
    )
    assert isinstance(op, GroundwaterTransportOperator)
    g = op.operator_matrix()
    assert g.shape == (n, n)
    a = op.transition_matrix(dt=0.05)
    assert a.shape == (n, n)


def test_decay_removes_mass_and_retardation_slows_the_front() -> None:
    n = 61
    shape = (n,)
    velocity_field = np.full((1, n), 1.0)
    baseline = GroundwaterTransportOperator(velocity_field, 0.3, shape, spacing=1.0, scheme="exact")
    decaying = GroundwaterTransportOperator(velocity_field, 0.3, shape, spacing=1.0, decay=0.2, scheme="exact")
    retarded = GroundwaterTransportOperator(velocity_field, 0.3, shape, spacing=1.0, retardation=4.0, scheme="exact")

    c0 = 1.0
    t = 8.0
    base_profile = _continuous_injection_profile(baseline, c0, t)
    decay_profile = _continuous_injection_profile(decaying, c0, t)
    retarded_profile = _continuous_injection_profile(retarded, c0, t)

    assert decay_profile.sum() < base_profile.sum()  # first-order decay removes mass
    # retardation divides the whole transient term, so the retarded front lags the baseline one.
    assert retarded_profile.sum() < base_profile.sum()


def test_velocity_field_shape_validation() -> None:
    n = 21
    with pytest.raises(ValueError):
        GroundwaterTransportOperator(np.zeros((2, n)), 0.1, (n,), spacing=1.0)


def test_multidimensional_shape_builds_a_consistent_operator() -> None:
    shape = (7, 5)
    n = 7 * 5
    velocity_field = np.stack([np.full(shape, 0.5), np.full(shape, -0.2)])
    op = GroundwaterTransportOperator(velocity_field, 0.1, shape, spacing=(1.0, 1.0), decay=0.05)
    g = op.operator_matrix()
    assert g.shape == (n, n)
    assert np.all(np.isfinite(g))


def test_ogata_banks_plume_is_bounded_and_causal() -> None:
    x = np.linspace(0.0, 50.0, 51)
    c_before = ogata_banks_plume(x, 0.0, 1.0, 1.0, c0=1.0)
    assert np.all(c_before == 0.0)
    c_after = ogata_banks_plume(x, 5.0, 1.0, 1.0, c0=1.0)
    assert np.all(c_after >= -1e-12) and np.all(c_after <= 1.0 + 1e-9)
    assert ogata_banks_plume(0.0, 5.0, 1.0, 1.0, c0=1.0) == pytest.approx(1.0, abs=1e-6)
