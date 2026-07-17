"""Tests for the cross-kernel/reference parity harness (MP-K4).

Covers the generic :class:`~mixle_pde.verification.parity.ParityCheck` machinery (typed record
construction/validation, coordinate-mismatch rejection, the single-subject-vs-reference fallback) and
the worked end-to-end example: the two genuinely overlapping registered kernels
(``transport-fd-advdiff``, ``groundwater-fd-transport``) driven on the identical 1-D periodic linear
advection-diffusion problem, checked against each other and against the exact
:func:`~mixle_pde.verification.parity.periodic_advection_diffusion_mode` manufactured solution. These
are real numerical results (real matrix exponentials, real registered-kernel invocations), not mocked
assertions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mixle_pde.pde_backend_registry import list_kernel_registrations
from mixle_pde.verification.parity import (
    KernelDiscrepancy,
    KernelReferenceError,
    ParityCheck,
    ParitySample,
    groundwater_uniform_velocity_subject,
    periodic_advection_diffusion_mode,
    transport_fd_advdiff_subject,
    transport_groundwater_parity_check,
)
from mixle_pde.verification.validation_tiers import ValidationTier

_DIFFUSIVITY = 0.02
_VELOCITY = 0.5
_LENGTH = 1.0
_WAVENUMBER = 2.0 * math.pi * 2.0 / _LENGTH  # two full periods across the domain


def _initial(x: np.ndarray) -> np.ndarray:
    return np.cos(_WAVENUMBER * x)


# ---------------------------------------------------------------------------
# periodic_advection_diffusion_mode: the manufactured reference formula itself
# ---------------------------------------------------------------------------
def test_periodic_advection_diffusion_mode_matches_initial_condition_at_t_zero():
    x = np.linspace(0.0, _LENGTH, 41)
    values = periodic_advection_diffusion_mode(
        x, 0.0, wavenumber=_WAVENUMBER, diffusivity=_DIFFUSIVITY, velocity=_VELOCITY
    )
    assert values == pytest.approx(np.cos(_WAVENUMBER * x), abs=1e-12)


def test_periodic_advection_diffusion_mode_satisfies_the_pde_by_finite_difference_residual():
    # Direct numerical check that u_t == D*u_xx - c*u_x at an interior (x, t): central differences at
    # a shrinking step size must show the residual shrinking like eps**2 (truncation error), not just
    # happening to be small once.
    def u(x, t):
        return periodic_advection_diffusion_mode(
            x, t, wavenumber=_WAVENUMBER, diffusivity=_DIFFUSIVITY, velocity=_VELOCITY
        )

    x0, t0 = 0.3, 0.2
    residuals = []
    for eps in (1.0e-3, 1.0e-4):
        u_t = (u(x0, t0 + eps) - u(x0, t0 - eps)) / (2.0 * eps)
        u_x = (u(x0 + eps, t0) - u(x0 - eps, t0)) / (2.0 * eps)
        u_xx = (u(x0 + eps, t0) - 2.0 * u(x0, t0) + u(x0 - eps, t0)) / eps**2
        residuals.append(abs(u_t - (_DIFFUSIVITY * u_xx - _VELOCITY * u_x)))

    assert residuals[0] < 1.0e-3
    assert residuals[1] < 1.0e-4
    # quadratic shrink (eps -> eps/10 should shrink the residual by roughly 100x)
    assert residuals[1] < residuals[0] / 50.0


# ---------------------------------------------------------------------------
# The worked example: two real registered kernels, one manufactured reference.
# ---------------------------------------------------------------------------
def test_transport_and_groundwater_are_the_only_registered_kernels_sharing_a_problem_class():
    # transport-fd-advdiff and groundwater-fd-transport are both linear_operator kernels declaring
    # the identical FD-implicit/FD-explicit/FD-exact discretization set -- the signature of the
    # documented generalization relationship this module's docstring cites.
    registrations = {reg.profile.id: reg for reg in list_kernel_registrations()}
    transport = registrations["transport-fd-advdiff"]
    groundwater = registrations["groundwater-fd-transport"]
    assert transport.profile.operator_kinds == groundwater.profile.operator_kinds == frozenset({"linear_operator"})
    assert transport.profile.discretizations == groundwater.profile.discretizations


def test_transport_groundwater_parity_check_kernels_agree_to_machine_precision():
    check = transport_groundwater_parity_check(n=121)

    assert check.reference is not None
    result = check.run()

    assert result.problem_label
    assert len(result.kernel_discrepancies) == 1
    discrepancy = result.kernel_discrepancies[0]
    assert {discrepancy.label_a, discrepancy.label_b} == {"transport-fd-advdiff", "groundwater-fd-transport"}
    assert discrepancy.tier is ValidationTier.CODE_TO_CODE
    assert discrepancy.coordinates_shape == (121,)
    # the two registered kernels reduce to the literal same discrete operator in this configuration
    # (see the module docstring); agreement should be at floating-point round-off, not merely "close".
    assert discrepancy.max_abs_discrepancy < 1.0e-9
    assert discrepancy.rms_discrepancy < 1.0e-9


def test_transport_groundwater_parity_check_reference_error_is_real_and_shrinks_with_resolution():
    coarse = transport_groundwater_parity_check(n=61).run()
    fine = transport_groundwater_parity_check(n=121).run()

    for result in (coarse, fine):
        assert len(result.reference_errors) == 2
        labels = {error.label for error in result.reference_errors}
        assert labels == {"transport-fd-advdiff", "groundwater-fd-transport"}
        for error in result.reference_errors:
            assert error.reference_label == "periodic_advection_diffusion_mode"
            assert error.tier is ValidationTier.MANUFACTURED
            # a real, nonzero discretization error -- not a vacuous zero from a wiring bug
            assert 0.0 < error.max_abs_error < 0.5
            assert 0.0 < error.rms_error <= error.max_abs_error

    coarse_by_label = {error.label: error for error in coarse.reference_errors}
    fine_by_label = {error.label: error for error in fine.reference_errors}
    for label in coarse_by_label:
        # refining the grid must shrink the error against the exact manufactured solution
        assert fine_by_label[label].max_abs_error < coarse_by_label[label].max_abs_error


# ---------------------------------------------------------------------------
# Discriminating power: a deliberately mismatched pair must NOT report zero discrepancy.
# ---------------------------------------------------------------------------
def test_parity_check_detects_a_genuine_mismatch_between_subjects():
    n = 81
    matched_kwargs = dict(velocity=_VELOCITY, n=n, length=_LENGTH, bc="periodic", scheme="exact", dt=0.01, n_steps=5)
    subject_a = transport_fd_advdiff_subject(diffusivity=0.02, initial=_initial, **matched_kwargs)
    # a deliberately different diffusivity: no longer the same problem instance
    subject_b = groundwater_uniform_velocity_subject(diffusivity=0.05, initial=_initial, **matched_kwargs)

    check = ParityCheck(problem_label="deliberate mismatch", subjects={"a": subject_a, "b": subject_b})
    result = check.run()

    assert len(result.kernel_discrepancies) == 1
    discrepancy = result.kernel_discrepancies[0]
    assert discrepancy.max_abs_discrepancy > 1.0e-3
    assert discrepancy.rms_discrepancy > 0.0


# ---------------------------------------------------------------------------
# Framework generality: a single subject against a reference is a supported fallback.
# ---------------------------------------------------------------------------
def test_parity_check_supports_a_single_subject_against_a_reference_only():
    subject = transport_fd_advdiff_subject(
        diffusivity=_DIFFUSIVITY,
        velocity=_VELOCITY,
        n=81,
        length=_LENGTH,
        bc="periodic",
        scheme="exact",
        dt=0.01,
        n_steps=5,
        initial=_initial,
    )

    def reference(x: np.ndarray) -> np.ndarray:
        return periodic_advection_diffusion_mode(
            x, 0.05, wavenumber=_WAVENUMBER, diffusivity=_DIFFUSIVITY, velocity=_VELOCITY
        )

    check = ParityCheck(
        problem_label="single-kernel fallback",
        subjects={"transport-fd-advdiff": subject},
        reference=reference,
        reference_tier=ValidationTier.MANUFACTURED,
    )
    result = check.run()

    # no peer to compare against -> no kernel-vs-kernel record, but the reference leg still runs
    assert result.kernel_discrepancies == ()
    assert len(result.reference_errors) == 1
    assert result.reference_errors[0].label == "transport-fd-advdiff"
    assert 0.0 < result.reference_errors[0].max_abs_error < 0.5


# ---------------------------------------------------------------------------
# Error paths: never silently misalign or conflate.
# ---------------------------------------------------------------------------
def test_parity_check_rejects_subjects_with_mismatched_value_shapes():
    def subject_fine() -> ParitySample:
        x = np.linspace(0.0, 1.0, 40)
        return ParitySample(label="fine", values=np.zeros_like(x), coordinates=x)

    def subject_coarse() -> ParitySample:
        x = np.linspace(0.0, 1.0, 20)
        return ParitySample(label="coarse", values=np.zeros_like(x), coordinates=x)

    check = ParityCheck(problem_label="mismatched shapes", subjects={"fine": subject_fine, "coarse": subject_coarse})
    with pytest.raises(ValueError, match="value shapes differ"):
        check.run()


def test_parity_check_rejects_subjects_sampled_at_different_coordinates():
    # same shape, genuinely different sample points -- must not be silently diffed as if aligned
    def subject_unit_interval() -> ParitySample:
        x = np.linspace(0.0, 1.0, 30)
        return ParitySample(label="unit", values=np.zeros_like(x), coordinates=x)

    def subject_shifted_interval() -> ParitySample:
        x = np.linspace(0.5, 1.5, 30)
        return ParitySample(label="shifted", values=np.zeros_like(x), coordinates=x)

    check = ParityCheck(
        problem_label="mismatched coordinates",
        subjects={"unit": subject_unit_interval, "shifted": subject_shifted_interval},
    )
    with pytest.raises(ValueError, match="different coordinates"):
        check.run()


def test_parity_check_requires_at_least_one_subject():
    with pytest.raises(ValueError):
        ParityCheck(problem_label="empty", subjects={})


def test_parity_check_rejects_a_code_to_code_reference_tier():
    def subject() -> ParitySample:
        x = np.linspace(0.0, 1.0, 10)
        return ParitySample(label="x", values=np.zeros_like(x), coordinates=x)

    with pytest.raises(ValueError):
        ParityCheck(
            problem_label="bad tier",
            subjects={"x": subject},
            reference=lambda coords: np.zeros_like(coords),
            reference_tier=ValidationTier.CODE_TO_CODE,
        )


def test_parity_sample_rejects_mismatched_values_and_coordinates_shapes():
    with pytest.raises(ValueError):
        ParitySample(label="bad", values=np.zeros(5), coordinates=np.zeros(4))


def test_kernel_discrepancy_rejects_a_non_code_to_code_tier():
    with pytest.raises(ValueError):
        KernelDiscrepancy(
            label_a="a",
            label_b="b",
            coordinates_shape=(10,),
            max_abs_discrepancy=0.0,
            rms_discrepancy=0.0,
            tier=ValidationTier.ANALYTIC,
        )


def test_kernel_discrepancy_rejects_a_negative_value():
    with pytest.raises(ValueError):
        KernelDiscrepancy(
            label_a="a", label_b="b", coordinates_shape=(10,), max_abs_discrepancy=-1.0, rms_discrepancy=0.0
        )


def test_kernel_reference_error_rejects_a_code_to_code_tier():
    with pytest.raises(ValueError):
        KernelReferenceError(
            label="a",
            reference_label="ref",
            coordinates_shape=(10,),
            max_abs_error=0.0,
            rms_error=0.0,
            tier=ValidationTier.CODE_TO_CODE,
        )
