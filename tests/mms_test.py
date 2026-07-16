"""Tests for the standalone method-of-manufactured-solutions convergence utility (MP-K3).

The worked end-to-end case manufactures a solution for the registered ``transport-fd-advdiff``
kernel (:class:`mixle_pde.dynamics.AdvectionDiffusionOperator`) -- a nonzero-source, time-dependent
sinusoid that does *not* solve the homogeneous advection-diffusion equation on its own -- runs it at
four grid resolutions, and asserts the receipt's measured spatial order of convergence lands within
tolerance of the discretization's documented theoretical order (1, from the first-order upwind
advection term). This is a real numerical result, not a mocked assertion: the error values, the
measured order, and the pass/fail verdict all come from actually stepping the unmodified legacy
solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.verification.mms import (
    ConvergenceReceipt,
    MMSResolutionResult,
    estimate_convergence_order,
    evaluate_convergence,
    run_transport_fd_advdiff_mms,
    transport_fd_advdiff_convergence_receipt,
)

# ---------------------------------------------------------------------------
# Manufactured solution: u(x, t) = sin(k x) cos(omega t) on a periodic [0, L) grid.
#
# This does NOT solve the homogeneous PDE du/dt = D u_xx - c u_x on its own (omega is chosen
# independently of the physical dispersion relation), so the source term below is genuinely
# nonzero -- derived by hand from du/dt - D*u_xx + c*u_x, plain calculus, no symbolic-differentiation
# tooling involved.
# ---------------------------------------------------------------------------
_LENGTH = 1.0
_WAVENUMBER = 2.0 * np.pi * 2.0 / _LENGTH  # two full periods across the domain
_OMEGA = 3.0
_DIFFUSIVITY = 0.02
_VELOCITY = 0.5


def _exact_solution(x: np.ndarray, t: float) -> np.ndarray:
    return np.sin(_WAVENUMBER * x) * np.cos(_OMEGA * t)


def _source_term(x: np.ndarray, t: float) -> np.ndarray:
    # du/dt - D*u_xx + c*u_x, evaluated in closed form for u = sin(kx)cos(wt):
    #   u_t  = -w sin(kx) sin(wt)
    #   u_xx = -k^2 sin(kx) cos(wt)
    #   u_x  =  k cos(kx) cos(wt)
    return (
        -_OMEGA * np.sin(_WAVENUMBER * x) * np.sin(_OMEGA * t)
        + _DIFFUSIVITY * _WAVENUMBER**2 * np.sin(_WAVENUMBER * x) * np.cos(_OMEGA * t)
        + _VELOCITY * _WAVENUMBER * np.cos(_WAVENUMBER * x) * np.cos(_OMEGA * t)
    )


_RESOLUTIONS = (161, 321, 641, 1281)
_DT = 2.0e-5
_N_STEPS = 250


def _run_receipt(*, expected_order: float = 1.0, order_tolerance: float = 0.3) -> ConvergenceReceipt:
    return transport_fd_advdiff_convergence_receipt(
        exact_solution=_exact_solution,
        source_term=_source_term,
        resolutions=_RESOLUTIONS,
        diffusivity=_DIFFUSIVITY,
        velocity=_VELOCITY,
        length=_LENGTH,
        dt=_DT,
        n_steps=_N_STEPS,
        norm="max",
        expected_order=expected_order,
        order_tolerance=order_tolerance,
    )


# ---------------------------------------------------------------------------
# estimate_convergence_order
# ---------------------------------------------------------------------------
def test_estimate_convergence_order_recovers_known_synthetic_slope():
    # error = C * h^2 exactly, by construction -- the estimator must recover p=2.
    h = np.array([0.1, 0.05, 0.025, 0.0125])
    errors = 3.0 * h**2
    order = estimate_convergence_order(h, errors)
    assert order == pytest.approx(2.0, abs=1e-8)


def test_estimate_convergence_order_requires_at_least_two_points():
    with pytest.raises(ValueError):
        estimate_convergence_order([0.1], [1.0])


def test_estimate_convergence_order_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        estimate_convergence_order([0.1, 0.05], [1.0, 0.5, 0.25])


def test_estimate_convergence_order_rejects_zero_error():
    with pytest.raises(ValueError):
        estimate_convergence_order([0.1, 0.05], [1.0, 0.0])


# ---------------------------------------------------------------------------
# evaluate_convergence
# ---------------------------------------------------------------------------
def test_evaluate_convergence_builds_a_passing_receipt_for_matching_order():
    results = (
        MMSResolutionResult(resolution=20, grid_spacing=0.1, error=1.0e-2, norm="max"),
        MMSResolutionResult(resolution=40, grid_spacing=0.05, error=5.0e-3, norm="max"),
        MMSResolutionResult(resolution=80, grid_spacing=0.025, error=2.5e-3, norm="max"),
    )
    receipt = evaluate_convergence(
        results,
        solver_id="synthetic-first-order",
        discretization="synthetic",
        expected_order=1.0,
        order_tolerance=0.1,
    )
    assert receipt.passed is True
    assert receipt.measured_order == pytest.approx(1.0, abs=1e-6)
    assert receipt.resolutions == (20, 40, 80)
    assert "PASS" in receipt.detail


def test_evaluate_convergence_rejects_mixed_norms():
    results = (
        MMSResolutionResult(resolution=20, grid_spacing=0.1, error=1.0e-2, norm="max"),
        MMSResolutionResult(resolution=40, grid_spacing=0.05, error=5.0e-3, norm="l2"),
    )
    with pytest.raises(ValueError):
        evaluate_convergence(results, solver_id="x", discretization="x", expected_order=1.0)


def test_evaluate_convergence_requires_at_least_two_results():
    results = (MMSResolutionResult(resolution=20, grid_spacing=0.1, error=1.0e-2, norm="max"),)
    with pytest.raises(ValueError):
        evaluate_convergence(results, solver_id="x", discretization="x", expected_order=1.0)


# ---------------------------------------------------------------------------
# run_transport_fd_advdiff_mms -- direct low-level run (no receipt)
# ---------------------------------------------------------------------------
def test_run_transport_fd_advdiff_mms_returns_one_result_per_resolution_and_errors_shrink():
    results = run_transport_fd_advdiff_mms(
        exact_solution=_exact_solution,
        source_term=_source_term,
        resolutions=_RESOLUTIONS,
        diffusivity=_DIFFUSIVITY,
        velocity=_VELOCITY,
        length=_LENGTH,
        dt=_DT,
        n_steps=_N_STEPS,
        norm="max",
    )
    assert len(results) == len(_RESOLUTIONS)
    for result, resolution in zip(results, _RESOLUTIONS, strict=True):
        assert result.resolution == resolution
        assert result.norm == "max"
        assert np.isfinite(result.error)
        assert result.error > 0.0
    # refinement must monotonically shrink the error for this manufactured solution
    errors = [result.error for result in results]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_run_transport_fd_advdiff_mms_rejects_bad_norm():
    with pytest.raises(ValueError):
        run_transport_fd_advdiff_mms(
            exact_solution=_exact_solution,
            source_term=_source_term,
            resolutions=_RESOLUTIONS,
            diffusivity=_DIFFUSIVITY,
            velocity=_VELOCITY,
            dt=_DT,
            n_steps=_N_STEPS,
            norm="rms",
        )


def test_run_transport_fd_advdiff_mms_rejects_single_resolution():
    with pytest.raises(ValueError):
        run_transport_fd_advdiff_mms(
            exact_solution=_exact_solution,
            source_term=_source_term,
            resolutions=(161,),
            diffusivity=_DIFFUSIVITY,
            velocity=_VELOCITY,
            dt=_DT,
            n_steps=_N_STEPS,
        )


# ---------------------------------------------------------------------------
# Worked end-to-end example: transport-fd-advdiff genuinely converges at its expected order.
# ---------------------------------------------------------------------------
def test_transport_fd_advdiff_mms_convergence_receipt_passes_at_expected_first_order():
    receipt = _run_receipt(expected_order=1.0, order_tolerance=0.3)

    assert receipt.solver_id == "transport-fd-advdiff"
    assert receipt.resolutions == _RESOLUTIONS
    assert len(receipt.errors) == len(_RESOLUTIONS)
    assert all(error > 0.0 for error in receipt.errors)
    assert receipt.expected_order == pytest.approx(1.0)
    # a genuinely measured order close to the documented first-order upwind truncation rate
    assert receipt.measured_order == pytest.approx(1.0, abs=0.3)
    assert receipt.passed is True
    assert "PASS" in receipt.detail


def test_transport_fd_advdiff_mms_receipt_solver_id_matches_a_registered_backend():
    from mixle_pde.pde_backend_registry import list_kernel_registrations

    receipt = _run_receipt()
    registered_ids = {registration.profile.id for registration in list_kernel_registrations()}
    assert receipt.solver_id in registered_ids


# ---------------------------------------------------------------------------
# Negative test: a deliberately wrong expected order must fail the verdict, not rubber-stamp it.
# ---------------------------------------------------------------------------
def test_transport_fd_advdiff_mms_receipt_fails_for_a_deliberately_wrong_expected_order():
    # The kernel is genuinely first-order (upwind advection dominates); claiming second order and
    # using a tight tolerance must be rejected by the same evaluator that just passed the correct claim.
    receipt = _run_receipt(expected_order=2.0, order_tolerance=0.3)

    assert receipt.passed is False
    assert receipt.expected_order == pytest.approx(2.0)
    # the measured order itself is unaffected by the (wrong) claim -- only the verdict differs
    assert receipt.measured_order == pytest.approx(1.0, abs=0.3)
    assert "FAIL" in receipt.detail
