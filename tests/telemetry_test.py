"""Tests for the MP-L5 solve-observability primitives (mixle_pde.telemetry).

Covers ``SolveTelemetry``/``ConvergenceStatus`` construction and validation, ``digest_solve_inputs``
determinism (including its lenient, never-raises fallback for array-likes and other odd values),
``record_solve_telemetry``'s capture-handle mechanics (including the negative path: an exception before
``complete()`` leaves no fabricated record), and the ``deprecation_notice`` decorator/registry
mechanics. The suite's one live, non-hypothetical worked example drives the real, already-merged
``fem-p1-simplex`` solve path (``mixle_pde.mesh.box_simplex_mesh`` +
``mixle_pde.fem.solve_simplex_poisson``, the same calls ``mixle_pde.pde_backend_registry``'s
``_invoke_fem_p1`` makes) through ``record_solve_telemetry`` and checks the resulting record against
real wall-clock time and a real input digest, not mocked values.
"""

from __future__ import annotations

import math
import time

import pytest

from mixle_pde.fem import solve_simplex_poisson
from mixle_pde.mesh import box_simplex_mesh
from mixle_pde.telemetry import (
    ConvergenceStatus,
    DeprecationNotice,
    SolveTelemetry,
    clear_deprecations,
    deprecation_notice,
    digest_solve_inputs,
    get_deprecation,
    list_deprecations,
    record_solve_telemetry,
)

# ---------------------------------------------------------------------------
# SolveTelemetry / ConvergenceStatus: construction and validation.
# ---------------------------------------------------------------------------


def test_solve_telemetry_accepts_a_well_formed_record():
    telemetry = SolveTelemetry(
        kernel_id="fem-p1-simplex",
        wall_clock_seconds=0.125,
        convergence_status=ConvergenceStatus.CONVERGED,
        input_digest="a" * 64,
        iteration_count=12,
    )
    assert telemetry.iteration_count == 12
    assert telemetry.convergence_status is ConvergenceStatus.CONVERGED


def test_solve_telemetry_iteration_count_defaults_to_none_for_non_iterative_solves():
    telemetry = SolveTelemetry(
        kernel_id="fem-p1-simplex",
        wall_clock_seconds=0.01,
        convergence_status=ConvergenceStatus.NOT_APPLICABLE,
        input_digest="b" * 64,
    )
    assert telemetry.iteration_count is None


def test_convergence_status_includes_the_required_non_boolean_outcomes():
    # Standing convention: unknown/timeout/resource_limit are valid typed outcomes, never a fabricated
    # Boolean.
    names = {member.value for member in ConvergenceStatus}
    assert {"unknown", "timeout", "resource_limit"} <= names


def test_solve_telemetry_rejects_blank_kernel_id():
    with pytest.raises(ValueError):
        SolveTelemetry(
            kernel_id="   ",
            wall_clock_seconds=0.01,
            convergence_status=ConvergenceStatus.UNKNOWN,
            input_digest="c" * 64,
        )


def test_solve_telemetry_rejects_non_convergence_status_member():
    with pytest.raises(TypeError):
        SolveTelemetry(
            kernel_id="fem-p1-simplex",
            wall_clock_seconds=0.01,
            convergence_status="converged",  # a raw string, not ConvergenceStatus.CONVERGED
            input_digest="d" * 64,
        )


@pytest.mark.parametrize("bad_digest", ["", "not-hex-and-wrong-length", "A" * 64, "a" * 63])
def test_solve_telemetry_rejects_malformed_input_digest(bad_digest):
    with pytest.raises(ValueError):
        SolveTelemetry(
            kernel_id="fem-p1-simplex",
            wall_clock_seconds=0.01,
            convergence_status=ConvergenceStatus.UNKNOWN,
            input_digest=bad_digest,
        )


@pytest.mark.parametrize("bad_seconds", [-0.001, math.inf, math.nan])
def test_solve_telemetry_rejects_negative_or_non_finite_wall_clock(bad_seconds):
    with pytest.raises(ValueError):
        SolveTelemetry(
            kernel_id="fem-p1-simplex",
            wall_clock_seconds=bad_seconds,
            convergence_status=ConvergenceStatus.UNKNOWN,
            input_digest="e" * 64,
        )


def test_solve_telemetry_rejects_negative_iteration_count():
    with pytest.raises(ValueError):
        SolveTelemetry(
            kernel_id="fem-p1-simplex",
            wall_clock_seconds=0.01,
            convergence_status=ConvergenceStatus.DIVERGED,
            input_digest="f" * 64,
            iteration_count=-1,
        )


def test_solve_telemetry_rejects_bool_iteration_count():
    # bool is an int subclass in Python; guard against `iteration_count=True` sneaking through.
    with pytest.raises(TypeError):
        SolveTelemetry(
            kernel_id="fem-p1-simplex",
            wall_clock_seconds=0.01,
            convergence_status=ConvergenceStatus.CONVERGED,
            input_digest="1" * 64,
            iteration_count=True,
        )


# ---------------------------------------------------------------------------
# digest_solve_inputs: determinism, key-order independence, and the lenient fallback.
# ---------------------------------------------------------------------------


def test_digest_solve_inputs_is_deterministic_and_key_order_independent():
    first = digest_solve_inputs({"grid_shape": (6, 6), "diffusion": 1.0, "source": 1.0})
    second = digest_solve_inputs({"source": 1.0, "diffusion": 1.0, "grid_shape": (6, 6)})
    assert first == second
    assert len(first) == 64
    assert all(char in "0123456789abcdef" for char in first)


def test_digest_solve_inputs_distinguishes_different_inputs():
    first = digest_solve_inputs({"diffusion": 1.0})
    second = digest_solve_inputs({"diffusion": 2.0})
    assert first != second


def test_digest_solve_inputs_handles_array_like_values_via_duck_typed_tolist():
    class _FakeTensor:
        def tolist(self):
            return [1.0, 2.0, 3.0]

    digest = digest_solve_inputs({"initial": _FakeTensor()})
    assert digest == digest_solve_inputs({"initial": [1.0, 2.0, 3.0]})


def test_digest_solve_inputs_never_raises_on_an_unrepresentable_value():
    class _Opaque:
        def __repr__(self):
            return "<opaque>"

    # Must not raise -- falls back to repr() rather than aborting telemetry capture.
    digest = digest_solve_inputs({"weird": _Opaque()})
    assert len(digest) == 64


def test_digest_solve_inputs_rejects_non_mapping():
    with pytest.raises(TypeError):
        digest_solve_inputs([("diffusion", 1.0)])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# record_solve_telemetry: capture-handle mechanics.
# ---------------------------------------------------------------------------


def test_record_solve_telemetry_captures_a_positive_wall_clock_duration():
    with record_solve_telemetry(kernel_id="wave-fd-leapfrog", inputs={"n_steps": 3}) as capture:
        time.sleep(0.01)
        capture.complete(convergence_status=ConvergenceStatus.CONVERGED, iteration_count=3)

    telemetry = capture.telemetry
    assert telemetry is not None
    assert telemetry.kernel_id == "wave-fd-leapfrog"
    assert telemetry.wall_clock_seconds > 0.0
    assert telemetry.iteration_count == 3
    assert telemetry.input_digest == digest_solve_inputs({"n_steps": 3})


def test_record_solve_telemetry_leaves_no_record_if_complete_is_never_called():
    with record_solve_telemetry(kernel_id="fem-p1-simplex", inputs={}) as capture:
        pass
    assert capture.telemetry is None


def test_record_solve_telemetry_propagates_exceptions_without_fabricating_a_record():
    with pytest.raises(ValueError, match="deliberate failure"):
        with record_solve_telemetry(kernel_id="fem-p1-simplex", inputs={}) as capture:
            raise ValueError("deliberate failure")
    assert capture.telemetry is None


def test_capture_complete_raises_if_called_twice():
    with record_solve_telemetry(kernel_id="fem-p1-simplex", inputs={}) as capture:
        capture.complete(convergence_status=ConvergenceStatus.NOT_APPLICABLE)
        with pytest.raises(RuntimeError):
            capture.complete(convergence_status=ConvergenceStatus.NOT_APPLICABLE)


# ---------------------------------------------------------------------------
# Live, non-hypothetical worked example: the real fem-p1-simplex solve path.
# ---------------------------------------------------------------------------


def test_fem_p1_simplex_solve_produces_real_telemetry():
    """Drives the same real calls `pde_backend_registry._invoke_fem_p1` makes, unmodified.

    Not a mock: a real P1 simplex mesh is built and a real sparse linear solve runs inside the capture
    block, so the measured wall-clock time and input digest both come from a genuine, already-merged
    solver path.
    """
    params = {"grid_shape": (6, 6), "lengths": (1.0, 1.0), "diffusion": 1.0, "source": 1.0}
    with record_solve_telemetry(kernel_id="fem-p1-simplex", inputs=params) as capture:
        mesh = box_simplex_mesh(params["grid_shape"], lengths=params["lengths"])
        solution = solve_simplex_poisson(mesh, params["source"], diffusion=params["diffusion"])
        capture.complete(convergence_status=ConvergenceStatus.NOT_APPLICABLE)

    telemetry = capture.telemetry
    assert telemetry is not None
    assert solution.shape == (mesh.n_nodes,)
    assert telemetry.wall_clock_seconds >= 0.0
    assert telemetry.convergence_status is ConvergenceStatus.NOT_APPLICABLE
    assert telemetry.input_digest == digest_solve_inputs(params)


# ---------------------------------------------------------------------------
# deprecation_notice: decorator + registry mechanics.
#
# Each test clears the module-level registry on entry and exit and never inspects state left by
# another test item, since the registry is in-process and this suite runs under pytest-xdist
# load-balanced distribution (tests/conftest.py).
# ---------------------------------------------------------------------------


def test_deprecation_notice_registers_a_typed_notice_at_decoration_time():
    clear_deprecations()
    try:

        @deprecation_notice(replacement="tests.telemetry_test.new_thing", reason="superseded by new_thing")
        def old_thing():
            return "ok"

        notice = get_deprecation(old_thing.__deprecation_notice__.qualified_name)
        assert isinstance(notice, DeprecationNotice)
        assert notice.replacement == "tests.telemetry_test.new_thing"
        assert notice.qualified_name.endswith("old_thing")
        assert notice in list_deprecations()
    finally:
        clear_deprecations()


def test_deprecation_notice_registers_before_any_call():
    # Declarative: the notice exists once the module defining the decorated callable is imported, not
    # only after the callable is first invoked (unlike SolveTelemetry capture, which is call-scoped).
    clear_deprecations()
    try:

        @deprecation_notice(replacement="tests.telemetry_test.new_thing", reason="not called yet")
        def never_called():
            raise AssertionError("must not be invoked by this test")

        assert any(n.qualified_name.endswith("never_called") for n in list_deprecations())
    finally:
        clear_deprecations()


def test_deprecation_notice_warns_on_every_call_and_preserves_behavior():
    clear_deprecations()
    try:

        @deprecation_notice(replacement="tests.telemetry_test.new_add", reason="renamed")
        def old_add(a, b):
            return a + b

        with pytest.warns(DeprecationWarning, match="new_add"):
            result = old_add(2, 3)
        assert result == 5

        with pytest.warns(DeprecationWarning):
            old_add(1, 1)
    finally:
        clear_deprecations()


def test_deprecation_notice_preserves_exceptions_from_the_wrapped_callable():
    clear_deprecations()
    try:

        @deprecation_notice(replacement="tests.telemetry_test.new_thing", reason="demo")
        def old_raiser():
            raise ValueError("deliberate failure")

        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="deliberate failure"):
                old_raiser()
    finally:
        clear_deprecations()


def test_get_deprecation_raises_a_helpful_error_for_an_unregistered_name():
    clear_deprecations()
    with pytest.raises(KeyError):
        get_deprecation("mixle_pde.nonexistent.thing")


def test_list_deprecations_is_sorted_by_qualified_name():
    clear_deprecations()
    try:

        @deprecation_notice(replacement="r", reason="z-named for sort check")
        def zzz_last():
            pass

        @deprecation_notice(replacement="r", reason="a-named for sort check")
        def aaa_first():
            pass

        names = [n.qualified_name for n in list_deprecations()]
        assert names == sorted(names)
    finally:
        clear_deprecations()


@pytest.mark.parametrize("field_name", ["qualified_name", "replacement", "reason"])
def test_deprecation_notice_record_rejects_blank_required_strings(field_name):
    kwargs = {"qualified_name": "mod.func", "replacement": "mod.new_func", "reason": "because"}
    kwargs[field_name] = "   "
    with pytest.raises(ValueError):
        DeprecationNotice(**kwargs)


def test_deprecation_notice_record_since_and_remove_after_are_optional():
    notice = DeprecationNotice(qualified_name="mod.func", replacement="mod.new_func", reason="because")
    assert notice.since is None
    assert notice.remove_after is None
