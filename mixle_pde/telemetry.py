"""Minimal solve-observability primitives (MP-L5 remainder).

The M2 reconciliation ledger (``docs/reconciliation/mp-task-ledger.md``, MP-L5) records that
mixle-pde has real packaging/extras infrastructure (PR #69, the batteries-included-extras contract)
but "no logs/metrics/traces observability or deprecation-policy telemetry found" anywhere in the
package. This module fills that gap with exactly two typed primitives:

1. :class:`SolveTelemetry` -- a frozen record any registered kernel invocation can *optionally* build
   and attach to its own result: wall-clock duration, iteration count (when the method is iterative),
   convergence status, kernel id, and a deterministic digest of the inputs that were run. Nothing in
   this module writes to a log, a file, a metrics endpoint, or any other side-effecting sink -- it is
   pure data capture. A caller decides what, if anything, to do with the record (attach it to a result,
   print it, forward it to a real telemetry backend). That full logs/metrics/traces stack is
   deliberately *not* built here: it is a much larger, differently-owned lift (arguably
   ``mixle-mlops``'s concern, per the portfolio scope split between domain packages and the operational
   MLOps package), not a scientific-computing library's job.
2. :func:`deprecation_notice` -- a decorator that records, once, that a call path is deprecated with a
   typed :class:`DeprecationNotice` (a string-identified replacement pointer, never a live callable --
   the same "no callables in canonical/serializable records" rule
   :mod:`mixle_pde.pde_backend_registry` documents for its own registry) and emits a
   ``DeprecationWarning`` on every call, same as Python's standard deprecation idiom.

Honesty note on the demonstration search
-----------------------------------------
This PR does not apply :func:`deprecation_notice` to any production call site. One real deprecation
already exists in this repo -- :meth:`mixle_pde.latent.PosteriorField3D.credible_interval` and
:meth:`~mixle_pde.latent.PosteriorFieldSamples3D.credible_interval` accept a deprecated ``alpha=``
keyword alias for the current ``level=`` parameter, with an ad hoc ``warnings.warn(...,
DeprecationWarning)`` inline -- but it is a *parameter*-level alias on a method that is itself not
deprecated, not a whole superseded callable with a distinct newer replacement. Decorating the method
itself with :func:`deprecation_notice` would misrepresent it as deprecated, which it is not.
``docs/migrations/legacy-to-canonical-adapters.md`` is explicit that nothing in this repo has actually
been removed, renamed, or reinterpreted yet ("remain compatible until an explicit deprecation gate
passes"), and ``ownership.py``'s own module classification confirms ``mesh.py``'s ``SimplexMesh`` --
the closest candidate for "superseded by a newer replacement" -- is explicitly *not yet* superseded
(disposition ``adapt``, final owner ``PRJ-SIM``; the replacement lives in a different package that has
not landed the integration). No genuinely obvious, same-repo, whole-callable deprecated-with-replacement
path exists yet to demonstrate on without fabricating one.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "ConvergenceStatus",
    "SolveTelemetry",
    "digest_solve_inputs",
    "SolveTelemetryCapture",
    "record_solve_telemetry",
    "DeprecationNotice",
    "deprecation_notice",
    "list_deprecations",
    "get_deprecation",
    "clear_deprecations",
]


class ConvergenceStatus(Enum):
    """The outcome of one solve/iteration loop.

    Includes the three typed non-Boolean outcomes the project's standing conventions require
    ("`unknown`/`timeout`/`resource_limit` are valid typed outcomes, never a fabricated Boolean") so a
    caller that genuinely cannot determine convergence, hit a wall-clock budget, or hit a resource cap
    is never forced to round that down to a fabricated ``True``/``False``.
    """

    CONVERGED = "converged"
    DIVERGED = "diverged"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SolveTelemetry:
    """One kernel invocation's execution facts -- pure data, no sink attached.

    ``kernel_id`` should match a registered backend id (e.g. one of
    :func:`mixle_pde.pde_backend_registry.list_kernel_registrations`'s ``profile.id`` values) when the
    invocation went through that registry, but this record does not require or validate that binding --
    it is equally usable for a solve that never touches the registry at all.
    """

    kernel_id: str
    wall_clock_seconds: float
    convergence_status: ConvergenceStatus
    input_digest: str
    iteration_count: int | None = None

    def __post_init__(self) -> None:
        if not self.kernel_id.strip():
            raise ValueError("kernel_id must be a non-empty string")
        if not isinstance(self.convergence_status, ConvergenceStatus):
            raise TypeError("convergence_status must be a ConvergenceStatus member")
        if not _DIGEST_PATTERN.match(self.input_digest):
            raise ValueError("input_digest must be a 64-character lowercase hex sha256 digest")
        if not math.isfinite(self.wall_clock_seconds) or self.wall_clock_seconds < 0:
            raise ValueError("wall_clock_seconds must be a finite, non-negative number")
        if self.iteration_count is not None:
            if isinstance(self.iteration_count, bool) or not isinstance(self.iteration_count, int):
                raise TypeError("iteration_count must be an int or None")
            if self.iteration_count < 0:
                raise ValueError("iteration_count must be non-negative")


def _normalize_for_digest(value: Any) -> Any:
    """Best-effort canonicalization for :func:`digest_solve_inputs` -- never raises.

    Deliberately more lenient than :mod:`mixle_pde.canonical_adapter`'s ``_normalize`` (which raises on
    anything it cannot represent exactly, because it backs a canonical identity record). A telemetry
    digest only needs to be deterministic and collision-resistant enough to correlate repeated
    invocations with the same inputs; an unrepresentable value falls back to its ``repr()`` rather than
    aborting telemetry capture for an otherwise-successful solve. Array-likes (``numpy.ndarray``,
    ``torch.Tensor``, ...) are handled via a duck-typed ``.tolist()`` check rather than importing either
    library, so this module has no third-party import of its own.
    """
    if isinstance(value, Mapping):
        try:
            keys = sorted(value)
        except TypeError:
            return repr(value)
        return {str(key): _normalize_for_digest(value[key]) for key in keys}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_digest(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _normalize_for_digest(tolist())
        except Exception:
            return repr(value)
    return repr(value)


def digest_solve_inputs(inputs: Mapping[str, Any]) -> str:
    """Deterministic sha256 fingerprint of a kernel invocation's keyword inputs.

    Same canonical-JSON-then-sha256 idiom already used in this package (``ownership.py``'s
    ``migration_inventory_digest``, ``canonical_adapter.py``'s ``_digest``): sort mapping keys, encode
    with fixed separators, hash. Two calls with equal (or key-reordered) ``inputs`` always produce the
    same digest; this is a correlation fingerprint for telemetry, not a cryptographic commitment.
    """
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping of keyword parameters")
    encoded = json.dumps(_normalize_for_digest(dict(inputs)), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SolveTelemetryCapture:
    """Mutable in-progress capture handle yielded by :func:`record_solve_telemetry`.

    Construct via :func:`record_solve_telemetry`, not directly. Call :meth:`complete` once, before the
    ``with`` block exits, to stop the clock and finalize an immutable :class:`SolveTelemetry` onto
    :attr:`telemetry`. If the block exits (normally or via exception) before :meth:`complete` runs,
    :attr:`telemetry` stays ``None`` -- an invocation that never reached a determinate outcome emits no
    fabricated record, the same "only record a genuine outcome" discipline
    :func:`mixle_pde.verification.validation_tiers.validation_evidence` already applies to evidence
    registration.
    """

    def __init__(self, *, kernel_id: str, input_digest: str) -> None:
        self._kernel_id = kernel_id
        self._input_digest = input_digest
        self._start = time.perf_counter()
        self.telemetry: SolveTelemetry | None = None

    def complete(
        self,
        *,
        convergence_status: ConvergenceStatus,
        iteration_count: int | None = None,
    ) -> SolveTelemetry:
        """Stop the clock and finalize the record. Safe to call at most once."""
        if self.telemetry is not None:
            raise RuntimeError("telemetry already completed for this capture")
        elapsed = time.perf_counter() - self._start
        self.telemetry = SolveTelemetry(
            kernel_id=self._kernel_id,
            wall_clock_seconds=elapsed,
            convergence_status=convergence_status,
            input_digest=self._input_digest,
            iteration_count=iteration_count,
        )
        return self.telemetry


@contextmanager
def record_solve_telemetry(*, kernel_id: str, inputs: Mapping[str, Any]) -> Iterator[SolveTelemetryCapture]:
    """Time a kernel invocation and hand back a capture handle to finalize it.

    ``inputs`` is digested once, up front, via :func:`digest_solve_inputs`. Usage from inside a kernel
    invoker (entirely opt-in -- no registered kernel is required to call this)::

        with record_solve_telemetry(kernel_id="fem-p1-simplex", inputs=params) as capture:
            solution = solve_simplex_poisson(mesh, source, diffusion=diffusion)
            capture.complete(convergence_status=ConvergenceStatus.NOT_APPLICABLE)
        telemetry = capture.telemetry  # SolveTelemetry, or None if complete() was never reached

    An exception raised inside the block propagates unchanged; this context manager never swallows it
    and never fabricates a convergence status on the caller's behalf.
    """
    capture = SolveTelemetryCapture(kernel_id=kernel_id, input_digest=digest_solve_inputs(inputs))
    yield capture


@dataclass(frozen=True)
class DeprecationNotice:
    """A typed record that one call path is deprecated, with a typed replacement pointer.

    ``replacement`` is a string identifier (e.g. a dotted qualified name), never a live callable --
    the same "no live callables in canonical/serializable artifacts" rule
    :mod:`mixle_pde.pde_backend_registry` documents for ``PDEKernelRegistration.invoke_key``, applied
    here so a :class:`DeprecationNotice` stays safe to collect, sort, and print without importing
    whatever it points at.
    """

    qualified_name: str
    replacement: str
    reason: str
    since: str | None = None
    remove_after: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.qualified_name, "qualified_name"),
            (self.replacement, "replacement"),
            (self.reason, "reason"),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be a non-empty string")


# Registered at decoration time (declarative: "this call path is deprecated"), independent of whether
# the decorated callable has ever actually been called. Mirrors the "register, don't branch" table
# idiom `mixle_pde.pde_backend_registry` uses for `_INVOKERS` -- keyed by qualified name, later
# registration for the same key overwrites (a decorator is expected to run at most once per name under
# normal import, so overwrite-by-key is simplicity, not a hazard).
_DEPRECATIONS: dict[str, DeprecationNotice] = {}


def deprecation_notice(
    *,
    replacement: str,
    reason: str,
    since: str | None = None,
    remove_after: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a deprecated callable: register a typed notice and warn on every call.

    The wrapped callable's behavior is unchanged (return value and raised exceptions pass through
    exactly as before) -- this only adds two side effects: a one-time registration of a
    :class:`DeprecationNotice` (readable via :func:`list_deprecations`/:func:`get_deprecation`, and
    also attached directly to the wrapped function as ``.__deprecation_notice__``), and a
    ``DeprecationWarning`` emitted on every call, naming ``replacement``.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        qualified_name = f"{func.__module__}.{func.__qualname__}"
        notice = DeprecationNotice(
            qualified_name=qualified_name,
            replacement=replacement,
            reason=reason,
            since=since,
            remove_after=remove_after,
        )
        _DEPRECATIONS[qualified_name] = notice

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{qualified_name} is deprecated: {reason} Use {replacement} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        wrapper.__deprecation_notice__ = notice
        return wrapper

    return decorator


def list_deprecations() -> tuple[DeprecationNotice, ...]:
    """Return every registered deprecation notice, sorted by qualified name."""
    return tuple(_DEPRECATIONS[key] for key in sorted(_DEPRECATIONS))


def get_deprecation(qualified_name: str) -> DeprecationNotice:
    """Look up a registered deprecation notice by qualified name."""
    try:
        return _DEPRECATIONS[qualified_name]
    except KeyError as exc:
        raise KeyError(f"no deprecation notice registered for {qualified_name!r}") from exc


def clear_deprecations() -> None:
    """Empty the in-process deprecation registry (tests only; production code never needs this)."""
    _DEPRECATIONS.clear()
