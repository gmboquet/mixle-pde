"""Tests for mixle_pde.scaling (MP-F6): nondimensionalization round trip + solver-hint advisories.

Grounds the module against the real ``"transport-fd-advdiff"`` registration in
:mod:`mixle_pde.pde_backend_registry` (which wraps
:class:`mixle_pde.dynamics.AdvectionDiffusionOperator`) rather than a hypothetical parameter set: that
registration's own :class:`~mixle_pde.pde_backend_registry.PDEPort` tuple declares ``diffusivity`` in
``"m^2/s"`` and ``velocity`` in ``"m/s"``, and its invoker (``_invoke_transport_fd``) defaults to
``diffusivity=0.01``, ``velocity=0.5``, ``length=1.0``. This file reads those units directly off the live
registration (so a future change to the registered ports fails this test loudly rather than silently
invalidating the grounding) and reuses the same default numbers below.
"""

from __future__ import annotations

import math

import pytest

from mixle_pde.pde_backend_registry import get_kernel_registration
from mixle_pde.scaling import (
    CharacteristicScales,
    PDEParameter,
    PDEParameterSet,
    dimensionless_groups,
    nondimensionalize,
    peclet_number,
    recommend_solver_hints,
    redimensionalize,
)

# mixle_pde/pde_backend_registry.py::_invoke_transport_fd's own defaults, reused here so this test's
# parameter set matches what the registered "transport-fd-advdiff" backend actually runs by default.
_DIFFUSIVITY = 0.01
_VELOCITY = 0.5
_LENGTH = 1.0


def _transport_registration_ports():
    registration = get_kernel_registration("transport-fd-advdiff")
    return {port.id: port for port in registration.ports}


def test_transport_fd_advdiff_ports_match_this_test_s_assumed_units():
    ports = _transport_registration_ports()
    assert ports["diffusivity"].units == "m^2/s"
    assert ports["velocity"].units == "m/s"


def _transport_parameter_set() -> PDEParameterSet:
    return PDEParameterSet(
        parameters=(
            PDEParameter(name="domain_length", value=_LENGTH, units="m", dimension="length"),
            PDEParameter(name="velocity", value=_VELOCITY, units="m/s", dimension="velocity"),
            PDEParameter(name="diffusivity", value=_DIFFUSIVITY, units="m^2/s", dimension="diffusivity"),
            PDEParameter(name="dt", value=0.01, units="s", dimension="time"),
        )
    )


def _transport_scales() -> CharacteristicScales:
    return CharacteristicScales(
        length=_LENGTH,
        time=_LENGTH / _VELOCITY,
        velocity=_VELOCITY,
        diffusivity=_DIFFUSIVITY,
    )


# --- CharacteristicScales -------------------------------------------------------------------------


def test_characteristic_scales_requires_positive_length():
    with pytest.raises(ValueError, match="length"):
        CharacteristicScales(length=0.0)


def test_characteristic_scales_rejects_non_positive_optional_scale_when_declared():
    with pytest.raises(ValueError, match="velocity"):
        CharacteristicScales(length=1.0, velocity=-2.0)


def test_characteristic_scales_undeclared_optional_scales_are_none_not_a_fabricated_default():
    scales = CharacteristicScales(length=1.0)
    assert scales.velocity is None
    assert scales.diffusivity is None
    assert scales.viscosity is None


# --- PDEParameter / PDEParameterSet ---------------------------------------------------------------


def test_pde_parameter_rejects_unknown_dimension():
    with pytest.raises(ValueError, match="unknown dimension"):
        PDEParameter(name="x", value=1.0, units="m", dimension="not_a_real_scale")


def test_pde_parameter_rejects_non_finite_value():
    with pytest.raises(ValueError, match="finite"):
        PDEParameter(name="x", value=float("nan"), units="m", dimension="length")


def test_pde_parameter_set_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        PDEParameterSet(
            parameters=(
                PDEParameter(name="x", value=1.0, units="m", dimension="length"),
                PDEParameter(name="x", value=2.0, units="m", dimension="length"),
            )
        )


# --- nondimensionalize / redimensionalize round trip, grounded on the real registered kernel -------


def test_round_trip_recovers_original_values_within_floating_point_precision():
    params = _transport_parameter_set()
    scales = _transport_scales()

    dimensionless = nondimensionalize(params, scales)
    restored = redimensionalize(dimensionless, scales)

    original = params.as_mapping()
    recovered = restored.as_mapping()
    assert original.keys() == recovered.keys()
    for name, value in original.items():
        assert math.isclose(value, recovered[name], rel_tol=1e-12)


def test_nondimensionalize_never_mutates_the_input_parameter_set():
    params = _transport_parameter_set()
    before = params.as_mapping()

    nondimensionalize(params, _transport_scales())

    assert params.as_mapping() == before


def test_nondimensional_values_match_hand_computed_scale_ratios():
    params = _transport_parameter_set()
    scales = _transport_scales()

    dimensionless = nondimensionalize(params, scales).as_mapping()

    # velocity/length/diffusivity were each nondimensionalized against a declared scale equal to their
    # own physical value (the natural choice here, since this kernel has exactly one coefficient per
    # role), so each collapses to exactly 1.0; dt (0.01 s) against the derived advective time scale
    # (length / velocity = 2.0 s) does not, and should equal 0.01 / 2.0 = 0.005.
    assert dimensionless["domain_length"] == pytest.approx(1.0)
    assert dimensionless["velocity"] == pytest.approx(1.0)
    assert dimensionless["diffusivity"] == pytest.approx(1.0)
    assert dimensionless["dt"] == pytest.approx(0.005)


def test_nondimensionalize_raises_rather_than_silently_defaulting_an_undeclared_scale():
    params = _transport_parameter_set()
    bare_scales = CharacteristicScales(length=_LENGTH)  # velocity/diffusivity/time left undeclared

    with pytest.raises(ValueError, match="CharacteristicScales.velocity"):
        nondimensionalize(params, bare_scales)


# --- dimensionless groups + solver-hint advisories, grounded on the same kernel defaults -----------


def test_peclet_number_matches_hand_computation_from_the_kernel_s_own_defaults():
    pe = peclet_number(velocity=_VELOCITY, length=_LENGTH, diffusivity=_DIFFUSIVITY)
    assert pe == pytest.approx(50.0)


def test_dimensionless_groups_reports_only_peclet_when_only_diffusivity_is_declared():
    scales = _transport_scales()
    groups = dimensionless_groups(scales)
    names = {g.name for g in groups}
    assert names == {"peclet"}
    (peclet_group,) = groups
    assert peclet_group.value == pytest.approx(50.0)


def test_dimensionless_groups_omits_reynolds_when_viscosity_was_never_declared():
    groups = dimensionless_groups(_transport_scales())
    assert "reynolds" not in {g.name for g in groups}


def test_recommend_solver_hints_flags_the_transport_kernel_s_defaults_as_advection_dominated():
    report = recommend_solver_hints(_transport_scales())

    codes = {a.code for a in report.advisories}
    assert "advection_dominated" in codes

    advisory = next(a for a in report.advisories if a.code == "advection_dominated")
    assert advisory.severity == "advisory"
    assert advisory.group_name == "peclet"
    assert advisory.group_value == pytest.approx(50.0)


def test_recommend_solver_hints_flags_diffusion_dominated_for_a_low_peclet_declaration():
    diffusion_heavy = CharacteristicScales(length=1.0, velocity=1e-4, diffusivity=1.0)
    report = recommend_solver_hints(diffusion_heavy)
    codes = {a.code for a in report.advisories}
    assert "diffusion_dominated" in codes
    assert "advection_dominated" not in codes


def test_recommend_solver_hints_returns_no_advisory_for_a_moderate_peclet_declaration():
    balanced = CharacteristicScales(length=1.0, velocity=1.0, diffusivity=1.0)  # Pe == 1.0
    report = recommend_solver_hints(balanced)
    assert report.advisories == ()
    assert report.groups_by_name()["peclet"] == pytest.approx(1.0)


def test_solver_advisory_rejects_an_invalid_severity():
    with pytest.raises(ValueError, match="severity"):
        from mixle_pde.scaling import SolverAdvisory

        SolverAdvisory(code="x", severity="critical", group_name="peclet", group_value=1.0, message="m")
