import numpy as np
import pytest

from mixle_pde.coupling_templates import (
    ConvergencePolicy,
    CouplingTemplateError,
    InterfaceVerdict,
    ParticipantInterfaceState,
    PortRole,
    check_scenario_against_template,
    coupling_template_catalog,
    evaluate_interface_balance,
    get_coupling_template,
)


def test_catalog_contains_the_four_baseline_templates():
    catalog = coupling_template_catalog()
    assert set(catalog) == {
        "one-way",
        "monolithic",
        "partitioned-dirichlet-neumann",
        "partitioned-quasi-newton",
    }
    for name, template in catalog.items():
        assert template.name == name
        assert template.participant_count == 2
        assert len(template.ports) >= 1
        assert {port.role for port in template.ports} == {PortRole.DIRICHLET, PortRole.NEUMANN}


def test_get_coupling_template_matches_catalog_entry():
    template = get_coupling_template("partitioned-dirichlet-neumann")
    assert template.convergence_policy == ConvergencePolicy.FIXED_POINT
    assert template.requires_relaxation is True


def test_monolithic_template_requires_no_iteration():
    template = get_coupling_template("monolithic")
    assert template.convergence_policy == ConvergencePolicy.EXACT
    assert template.requires_relaxation is False


def test_get_coupling_template_unknown_name_raises_with_known_names_listed():
    with pytest.raises(CouplingTemplateError, match="one-way"):
        get_coupling_template("no-such-template")


def test_coupling_template_rejects_a_single_participant():
    from mixle_pde.coupling_templates import CouplingTemplate, PortRequirement

    with pytest.raises(ValueError, match="at least two"):
        CouplingTemplate(
            name="solo",
            description="not actually a coupling",
            participant_count=1,
            ports=(PortRequirement(field_name="x", role=PortRole.DIRICHLET),),
            convergence_policy=ConvergencePolicy.NONE,
        )


def test_check_scenario_against_template_satisfied():
    check = check_scenario_against_template("one-way", ["dirichlet", "neumann"])
    assert check.satisfied is True
    assert check.template_name == "one-way"


def test_check_scenario_against_template_missing_role_reports_detail():
    check = check_scenario_against_template("monolithic", ["dirichlet", "dirichlet"])
    assert check.satisfied is False
    assert "requires roles" in check.detail


def test_participant_interface_state_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        ParticipantInterfaceState(participant_id="a", value=np.array([1.0, 2.0]), flux=np.array([1.0]))


def test_participant_interface_state_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        ParticipantInterfaceState(participant_id="a", value=np.array([1.0, np.nan]), flux=np.array([1.0, 1.0]))


def test_participant_interface_state_rejects_empty_array():
    with pytest.raises(ValueError, match="at least one"):
        ParticipantInterfaceState(participant_id="a", value=np.array([]), flux=np.array([]))


def _two_layer_conduction_states(*, temperature_offset: float = 0.0, flux_offset: float = 0.0):
    """Interface samples along a shared two-material steady-conduction boundary (5 points).

    A well-posed steady-conduction interface requires the temperature to be continuous
    (``state_left.value == state_right.value``) and, by Fourier's law with each side's own
    outward-normal convention, the heat flux leaving the left material to equal the heat flux
    entering the right material (``state_left.flux == -state_right.flux``). ``temperature_offset``/
    ``flux_offset`` inject a controlled violation of one or the other to exercise the FAIL path.
    """

    interface_temperature = np.array([300.0, 301.0, 302.5, 303.0, 305.0])
    outward_flux_left = np.array([10.0, 12.0, 9.0, 11.0, 8.0])
    state_left = ParticipantInterfaceState(
        participant_id="left-bar",
        value=interface_temperature,
        flux=outward_flux_left,
        unit_value="K",
        unit_flux="W/m^2",
    )
    state_right = ParticipantInterfaceState(
        participant_id="right-bar",
        value=interface_temperature + temperature_offset,
        flux=-outward_flux_left + flux_offset,
        unit_value="K",
        unit_flux="W/m^2",
    )
    return state_left, state_right


def test_evaluate_interface_balance_passes_for_a_conserved_continuous_interface():
    state_left, state_right = _two_layer_conduction_states()
    receipt = evaluate_interface_balance(state_left, state_right)
    assert receipt.verdict == InterfaceVerdict.PASS
    assert receipt.jump_residual == pytest.approx(0.0, abs=1e-12)
    assert receipt.flux_mismatch_residual == pytest.approx(0.0, abs=1e-12)


def test_evaluate_interface_balance_fails_on_a_temperature_jump():
    state_left, state_right = _two_layer_conduction_states(temperature_offset=0.5)
    receipt = evaluate_interface_balance(state_left, state_right)
    assert receipt.verdict == InterfaceVerdict.FAIL
    assert receipt.jump_residual == pytest.approx(0.5, abs=1e-12)
    assert receipt.flux_mismatch_residual == pytest.approx(0.0, abs=1e-12)


def test_evaluate_interface_balance_fails_on_a_flux_mismatch():
    state_left, state_right = _two_layer_conduction_states(flux_offset=2.0)
    receipt = evaluate_interface_balance(state_left, state_right)
    assert receipt.verdict == InterfaceVerdict.FAIL
    assert receipt.flux_mismatch_residual == pytest.approx(2.0, abs=1e-12)


def test_evaluate_interface_balance_unknown_on_point_count_mismatch():
    state_left, _ = _two_layer_conduction_states()
    state_right = ParticipantInterfaceState(
        participant_id="right-bar", value=np.array([300.0, 301.0]), flux=np.array([-10.0, -12.0])
    )
    receipt = evaluate_interface_balance(state_left, state_right)
    assert receipt.verdict == InterfaceVerdict.UNKNOWN
    assert np.isnan(receipt.jump_residual)


def test_evaluate_interface_balance_unknown_on_incompatible_units():
    state_left, state_right = _two_layer_conduction_states()
    relabeled_right = ParticipantInterfaceState(
        participant_id="right-bar", value=state_right.value, flux=state_right.flux, unit_value="degC"
    )
    receipt = evaluate_interface_balance(state_left, relabeled_right)
    assert receipt.verdict == InterfaceVerdict.UNKNOWN
    assert "unit" in receipt.detail


def test_evaluate_interface_balance_rejects_identical_participant_ids():
    state_left, _ = _two_layer_conduction_states()
    duplicate = ParticipantInterfaceState(participant_id="left-bar", value=state_left.value, flux=state_left.flux)
    with pytest.raises(ValueError, match="distinctly identified"):
        evaluate_interface_balance(state_left, duplicate)


def test_evaluate_interface_balance_rejects_negative_tolerance():
    state_left, state_right = _two_layer_conduction_states()
    with pytest.raises(ValueError, match="tolerance"):
        evaluate_interface_balance(state_left, state_right, jump_tolerance=-1.0)
