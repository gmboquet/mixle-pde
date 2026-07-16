"""In-process job governance and resource-declaration bookkeeping (MP-L4 slice).

``docs/reconciliation/mp-task-ledger.md`` records MP-L4 ("Sandboxed job orchestration and resource
governance") as ``not-started``: "``mixle_pde/simulation_service.py`` has scenario/result plumbing but
no queue/quota/heartbeat/retry governance." The ledger also flagged an in-flight PR (#71) claiming
"durable create, inspect, monitor, cancel, resume, run, and retrieve operations with typed failures" as
not-yet-counted evidence. That PR has since merged as ``mixle_pde/reference_lifecycle.py`` (#86), but it
is a lifecycle bound to one domain object (``ReferenceData``/``multiphysics_reference``), persisted as
one JSON file per study, with no liveness heartbeat, no stale-job detector, and no bounded retry count/
limit -- its ``resume`` can be called an unbounded number of times and nothing there ever notices a
study that silently stopped reporting progress. This module fills the remaining, still-generic gap: a
typed job record and an in-process registry that any future executor -- sandboxed or not, and regardless
of which mixle_pde forward it runs -- can sit behind.

Explicitly NOT in scope, matching the boundary MP-I9's checkpoint/restart module already drew around
real distributed execution for this same repo: this module never starts a thread, process, or MPI rank,
and never measures or enforces CPU/memory/GPU/wall-clock usage against the operating system.
:class:`ResourceBudget` is a typed *declaration* a submitted job carries; a real executor is what would
read and enforce it. :class:`JobRegistry` is bookkeeping only -- ``submit``/``heartbeat``/``cancel``/
``inspect``, plus the ``complete``/``fail`` counterparts a real executor calls around its own run loop,
and a stale-job detector a scheduler polls -- never the executor itself. Nothing here spawns a
background thread, timer, or process to call that detector automatically; a caller is responsible for
invoking :meth:`JobRegistry.check_stale` on whatever cadence its own execution loop runs. This is a
governance/bookkeeping layer a real executor would sit behind, not the executor itself.

State machine: ``queued -> running`` (a job's first :meth:`~JobRegistry.heartbeat` call) ``->
{completed, failed, cancelled}``. :meth:`~JobRegistry.fail` and the stale-job detector share one bounded
-retry pipeline: a failure is granted a retry -- the job returns to ``queued`` and ``retry_count`` (the
number of retries already used) is incremented -- while ``retry_count`` is less than ``retry_limit``;
``retry_limit`` defaults to 0, so the out-of-the-box behavior is exactly "fail on the first bad attempt"
-- retries are opt-in, never silently assumed. Once ``retry_count`` reaches ``retry_limit``, the next
failure is terminal: the job moves to ``failed`` with ``last_failure_reason`` set to a typed
:class:`FailureReason`, never left hanging in ``running`` behind a dead heartbeat.

Every registry operation returns a deep-copied snapshot :class:`JobRecord`, never the live object the
registry mutates internally, so a caller cannot corrupt registry state by editing a record it was handed.
``JobRegistry`` is a plain dict-backed object with no internal locking -- it is intended for single-
threaded use (e.g. one scheduler loop owning one registry instance), consistent with this module's
explicit non-goal of real concurrent/distributed execution.
"""

from __future__ import annotations

import enum
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobGovernanceError(RuntimeError):
    """Base for typed job-governance failures; carries a stable, machine-checkable ``code``."""

    def __init__(self, code: str, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.diagnostics = dict(diagnostics or {})


class JobNotFoundError(JobGovernanceError):
    """Raised by any operation given a ``job_id`` the registry never issued or has never seen."""

    def __init__(self, job_id: str) -> None:
        super().__init__("E_JOB_NOT_FOUND", f"no job registered under id {job_id!r}", diagnostics={"job_id": job_id})


class InvalidJobTransitionError(JobGovernanceError):
    """Raised when an operation is attempted from a status that does not support it.

    Mirrors the repo-wide convention that unsupported operations fail explicitly rather than silently
    degrading or coercing into a fabricated result.
    """

    def __init__(self, job_id: str, *, from_status: JobStatus, operation: str) -> None:
        super().__init__(
            "E_INVALID_TRANSITION",
            f"cannot {operation} job {job_id!r} while it is {from_status.value!r}",
            diagnostics={"job_id": job_id, "status": from_status.value, "operation": operation},
        )


class JobStatus(enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


class FailureReason(enum.Enum):
    """Typed proximate cause of one failed attempt (not "why retries stopped" -- that is always
    exactly ``retry_count >= retry_limit``, already visible on the record without a dedicated reason)."""

    STALE_HEARTBEAT = "stale_heartbeat"
    EXECUTOR_ERROR = "executor_error"
    RESOURCE_BUDGET_EXCEEDED = "resource_budget_exceeded"


@dataclass(frozen=True)
class ResourceBudget:
    """A caller-declared resource ceiling a real executor is expected to read and enforce.

    This module never measures or enforces CPU/memory/GPU/wall-clock usage itself -- see the module
    docstring. ``ResourceBudget`` exists so every submitted job carries a typed, non-optional
    declaration of what it is allowed to consume, rather than that intent living only in a comment or
    a backend's private config.
    """

    cpu_cores: float = 1.0
    memory_mb: float = 512.0
    wall_clock_seconds: float = 300.0
    gpu_count: int = 0

    def __post_init__(self) -> None:
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        if self.memory_mb <= 0:
            raise ValueError("memory_mb must be positive")
        if self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive")
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be non-negative")


@dataclass
class JobRecord:
    """The typed bookkeeping record for one submitted job.

    ``JobRegistry`` is the sole mutator of the live instance; callers only ever see deep-copied
    snapshots returned from its operations (see the module docstring).
    """

    id: str
    status: JobStatus
    budget: ResourceBudget
    retry_limit: int = 0
    retry_count: int = 0
    submitted_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    finished_at: datetime | None = None
    last_failure_reason: FailureReason | None = None
    last_failure_detail: str = ""
    cancel_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _snapshot(record: JobRecord) -> JobRecord:
    return deepcopy(record)


class JobRegistry:
    """A dict-backed, in-process registry of :class:`JobRecord` bookkeeping entries.

    Not an executor (see the module docstring): it never runs a job's actual work, only tracks what a
    real executor would report about it. Not thread-safe: intended for one owner (e.g. a single
    scheduler loop) per instance.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    def _get(self, job_id: str) -> JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFoundError(job_id) from None

    def submit(
        self,
        *,
        budget: ResourceBudget | None = None,
        retry_limit: int = 0,
        metadata: dict[str, Any] | None = None,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> JobRecord:
        """Register a new job as ``queued`` and return its snapshot.

        ``budget`` defaults to a conservative :class:`ResourceBudget` when omitted -- never left
        undeclared. ``retry_limit`` defaults to 0 (no retries; see the module docstring). ``job_id``
        lets a caller supply its own identifier (e.g. one derived from an upstream request); omit it to
        get a fresh random one. Re-submitting an already-registered ``job_id`` is a caller error, not an
        idempotent no-op -- this registry does not attempt request-identity deduplication.
        """
        if retry_limit < 0:
            raise ValueError("retry_limit must be non-negative")
        resolved_id = job_id or f"job-{uuid.uuid4().hex[:20]}"
        if resolved_id in self._jobs:
            raise ValueError(f"job id {resolved_id!r} is already registered")
        record = JobRecord(
            id=resolved_id,
            status=JobStatus.QUEUED,
            budget=budget or ResourceBudget(),
            retry_limit=retry_limit,
            submitted_at=now or _utcnow(),
            metadata=dict(metadata or {}),
        )
        self._jobs[resolved_id] = record
        return _snapshot(record)

    def heartbeat(self, job_id: str, *, now: datetime | None = None) -> JobRecord:
        """Record liveness for ``job_id``. The first heartbeat on a ``queued`` job promotes it to
        ``running`` and sets ``started_at``; every heartbeat (including that first one) updates
        ``heartbeat_at``, which :meth:`check_stale` reads. Raises :class:`InvalidJobTransitionError` for
        a job already in a terminal status -- a heartbeat past the end of a job's life is a caller bug,
        not something to absorb silently.
        """
        record = self._get(job_id)
        if record.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise InvalidJobTransitionError(job_id, from_status=record.status, operation="heartbeat")
        moment = now or _utcnow()
        if record.status is JobStatus.QUEUED:
            record.status = JobStatus.RUNNING
            record.started_at = moment
        record.heartbeat_at = moment
        return _snapshot(record)

    def cancel(self, job_id: str, *, reason: str = "", now: datetime | None = None) -> JobRecord:
        """Terminally cancel ``job_id`` from ``queued`` or ``running``. Cancelling an already-terminal
        job raises :class:`InvalidJobTransitionError` rather than silently no-op'ing, so a caller cannot
        mistake "already cancelled" for "just cancelled" or paper over cancelling a completed job."""
        record = self._get(job_id)
        if record.status in _TERMINAL_STATUSES:
            raise InvalidJobTransitionError(job_id, from_status=record.status, operation="cancel")
        record.status = JobStatus.CANCELLED
        record.cancel_reason = reason
        record.finished_at = now or _utcnow()
        return _snapshot(record)

    def complete(self, job_id: str, *, now: datetime | None = None) -> JobRecord:
        """Terminally complete ``job_id``. Only valid from ``running`` -- a job a real executor never
        heartbeated cannot be reported complete, since nothing ever recorded it as having started."""
        record = self._get(job_id)
        if record.status is not JobStatus.RUNNING:
            raise InvalidJobTransitionError(job_id, from_status=record.status, operation="complete")
        record.status = JobStatus.COMPLETED
        record.finished_at = now or _utcnow()
        return _snapshot(record)

    def fail(self, job_id: str, *, reason: FailureReason, detail: str = "", now: datetime | None = None) -> JobRecord:
        """Report that the current attempt at ``job_id`` failed for typed ``reason``; applies the
        bounded-retry policy described in the module docstring. Valid from ``queued`` or ``running``."""
        record = self._get(job_id)
        if record.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise InvalidJobTransitionError(job_id, from_status=record.status, operation="fail")
        self._apply_failure(record, reason=reason, detail=detail, now=now or _utcnow())
        return _snapshot(record)

    @staticmethod
    def _apply_failure(record: JobRecord, *, reason: FailureReason, detail: str, now: datetime) -> None:
        """The shared bounded-retry pipeline used by both :meth:`fail` and :meth:`check_stale`.

        Whether a retry is granted is decided from ``retry_count`` (retries already used) *before* this
        attempt's failure is counted, then ``retry_count`` is incremented for the attempt just granted
        (or just exhausted). ``retry_limit=2`` therefore permits exactly two retries -- three total
        attempts -- with the third failure landing terminal.
        """
        retry_granted = record.retry_count < record.retry_limit
        record.retry_count += 1
        record.last_failure_reason = reason
        record.last_failure_detail = detail
        if retry_granted:
            record.status = JobStatus.QUEUED
            record.started_at = None
            record.heartbeat_at = None
        else:
            record.status = JobStatus.FAILED
            record.finished_at = now

    def inspect(self, job_id: str) -> JobRecord:
        """Return a snapshot of ``job_id``'s current record."""
        return _snapshot(self._get(job_id))

    def list_jobs(self, *, status: JobStatus | None = None) -> list[JobRecord]:
        """Snapshots of every registered job, optionally filtered to one ``status``."""
        return [_snapshot(r) for r in self._jobs.values() if status is None or r.status is status]

    def check_stale(self, *, timeout_seconds: float, now: datetime | None = None) -> list[JobRecord]:
        """Scan every ``running`` job; any whose most recent heartbeat (or, if it has none yet despite
        being ``running``, its ``started_at``) is older than ``timeout_seconds`` is routed through the
        same bounded-retry pipeline :meth:`fail` uses, with :attr:`FailureReason.STALE_HEARTBEAT`. A
        ``queued`` job (never started) or a job already in a terminal status is never considered stale
        by this check -- only a ``running`` job can have a dead heartbeat. Returns a snapshot of every
        job this call transitioned (to ``queued`` for a retry, or terminally to ``failed``); a job whose
        heartbeat is still within ``timeout_seconds`` is left untouched and excluded from the result.

        This method does not schedule itself -- see the module docstring. A caller polls it.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        moment = now or _utcnow()
        transitioned: list[JobRecord] = []
        for record in self._jobs.values():
            if record.status is not JobStatus.RUNNING:
                continue
            last_seen = record.heartbeat_at or record.started_at
            if last_seen is None:
                continue
            age_seconds = (moment - last_seen).total_seconds()
            if age_seconds > timeout_seconds:
                self._apply_failure(
                    record,
                    reason=FailureReason.STALE_HEARTBEAT,
                    detail=f"no heartbeat for {age_seconds:.1f}s (timeout {timeout_seconds:.1f}s)",
                    now=moment,
                )
                transitioned.append(_snapshot(record))
        return transitioned


__all__ = [
    "FailureReason",
    "InvalidJobTransitionError",
    "JobGovernanceError",
    "JobNotFoundError",
    "JobRecord",
    "JobRegistry",
    "JobStatus",
    "ResourceBudget",
]
