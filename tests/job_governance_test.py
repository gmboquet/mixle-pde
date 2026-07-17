"""Tests for mixle_pde.job_governance: the MP-L4 in-process job registry.

Every test drives the registry with an explicit, synthetic ``now=`` timestamp rather than sleeping on
the real clock, so staleness/heartbeat-age behavior is exercised deterministically and fast. Coverage
follows the module's own stated acceptance surface: the queued/running/completed/failed/cancelled state
machine, snapshot isolation (a caller mutating a returned record must never affect registry state), the
bounded-retry pipeline shared by ``fail`` and ``check_stale`` (retry while ``retry_count < retry_limit``,
else terminal ``failed``), the stale-job detector, and typed-error behavior for unknown ids and invalid
transitions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mixle_pde.job_governance import (
    FailureReason,
    InvalidJobTransitionError,
    JobNotFoundError,
    JobRegistry,
    JobStatus,
    ResourceBudget,
)

T0 = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)


def test_submit_defaults_to_queued_with_no_retries():
    registry = JobRegistry()
    record = registry.submit(now=T0)
    assert record.status is JobStatus.QUEUED
    assert record.retry_limit == 0
    assert record.retry_count == 0
    assert record.started_at is None
    assert record.heartbeat_at is None
    assert record.budget == ResourceBudget()
    assert record.submitted_at == T0
    assert record.id.startswith("job-")


def test_submit_rejects_duplicate_job_id():
    registry = JobRegistry()
    registry.submit(job_id="job-fixed")
    with pytest.raises(ValueError):
        registry.submit(job_id="job-fixed")


def test_submit_rejects_negative_retry_limit():
    registry = JobRegistry()
    with pytest.raises(ValueError):
        registry.submit(retry_limit=-1)


def test_heartbeat_promotes_queued_to_running():
    registry = JobRegistry()
    record = registry.submit(now=T0)
    running = registry.heartbeat(record.id, now=T0)
    assert running.status is JobStatus.RUNNING
    assert running.started_at == T0
    assert running.heartbeat_at == T0


def test_heartbeat_again_updates_heartbeat_but_not_started_at():
    registry = JobRegistry()
    record = registry.submit(now=T0)
    registry.heartbeat(record.id, now=T0)
    t1 = T0 + timedelta(seconds=5)
    second = registry.heartbeat(record.id, now=t1)
    assert second.started_at == T0
    assert second.heartbeat_at == t1


def test_heartbeat_after_terminal_status_raises():
    registry = JobRegistry()
    record = registry.submit()
    registry.cancel(record.id)
    with pytest.raises(InvalidJobTransitionError):
        registry.heartbeat(record.id)


def test_cancel_valid_from_queued_and_running():
    registry = JobRegistry()
    queued = registry.submit()
    cancelled = registry.cancel(queued.id, reason="superseded", now=T0)
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.cancel_reason == "superseded"
    assert cancelled.finished_at == T0

    running = registry.submit()
    registry.heartbeat(running.id)
    cancelled_running = registry.cancel(running.id)
    assert cancelled_running.status is JobStatus.CANCELLED


def test_cancel_already_terminal_raises():
    registry = JobRegistry()
    record = registry.submit()
    registry.cancel(record.id)
    with pytest.raises(InvalidJobTransitionError):
        registry.cancel(record.id)


def test_complete_requires_running():
    registry = JobRegistry()
    record = registry.submit()
    with pytest.raises(InvalidJobTransitionError):
        registry.complete(record.id)
    registry.heartbeat(record.id)
    done = registry.complete(record.id, now=T0)
    assert done.status is JobStatus.COMPLETED
    assert done.finished_at == T0


def test_inspect_returns_isolated_snapshot():
    registry = JobRegistry()
    record = registry.submit(metadata={"op": "flow"})
    snapshot = registry.inspect(record.id)
    snapshot.metadata["op"] = "mutated"
    snapshot.retry_count = 999
    snapshot.status = JobStatus.FAILED

    fresh = registry.inspect(record.id)
    assert fresh.metadata == {"op": "flow"}
    assert fresh.retry_count == 0
    assert fresh.status is JobStatus.QUEUED


def test_fail_retries_until_limit_then_terminal():
    registry = JobRegistry()
    record = registry.submit(retry_limit=2)
    registry.heartbeat(record.id)

    first = registry.fail(record.id, reason=FailureReason.EXECUTOR_ERROR, detail="boom-1", now=T0)
    assert first.status is JobStatus.QUEUED
    assert first.retry_count == 1
    assert first.last_failure_reason is FailureReason.EXECUTOR_ERROR
    assert first.last_failure_detail == "boom-1"
    assert first.heartbeat_at is None
    assert first.started_at is None

    registry.heartbeat(record.id)
    second = registry.fail(record.id, reason=FailureReason.EXECUTOR_ERROR, detail="boom-2")
    assert second.status is JobStatus.QUEUED
    assert second.retry_count == 2

    registry.heartbeat(record.id)
    third = registry.fail(record.id, reason=FailureReason.EXECUTOR_ERROR, detail="boom-3", now=T0)
    assert third.status is JobStatus.FAILED
    assert third.retry_count == 3
    assert third.finished_at == T0


def test_fail_with_default_retry_limit_is_immediately_terminal():
    registry = JobRegistry()
    record = registry.submit()
    registry.heartbeat(record.id)
    failed = registry.fail(record.id, reason=FailureReason.RESOURCE_BUDGET_EXCEEDED)
    assert failed.status is JobStatus.FAILED
    assert failed.retry_count == 1
    assert failed.last_failure_reason is FailureReason.RESOURCE_BUDGET_EXCEEDED


def test_fail_invalid_from_terminal_status_raises():
    registry = JobRegistry()
    record = registry.submit()
    registry.heartbeat(record.id)
    registry.complete(record.id)
    with pytest.raises(InvalidJobTransitionError):
        registry.fail(record.id, reason=FailureReason.EXECUTOR_ERROR)


def test_check_stale_defaults_to_terminal_failure():
    registry = JobRegistry()
    record = registry.submit()
    registry.heartbeat(record.id, now=T0)
    later = T0 + timedelta(seconds=100)

    transitioned = registry.check_stale(timeout_seconds=30, now=later)

    assert [r.id for r in transitioned] == [record.id]
    assert transitioned[0].status is JobStatus.FAILED
    assert transitioned[0].last_failure_reason is FailureReason.STALE_HEARTBEAT
    assert registry.inspect(record.id).status is JobStatus.FAILED


def test_check_stale_requeues_then_exhausts_retry_budget():
    registry = JobRegistry()
    record = registry.submit(retry_limit=1)
    registry.heartbeat(record.id, now=T0)
    later = T0 + timedelta(seconds=100)

    first_pass = registry.check_stale(timeout_seconds=30, now=later)
    assert first_pass[0].status is JobStatus.QUEUED
    assert first_pass[0].retry_count == 1

    registry.heartbeat(record.id, now=later)
    even_later = later + timedelta(seconds=100)
    second_pass = registry.check_stale(timeout_seconds=30, now=even_later)
    assert second_pass[0].status is JobStatus.FAILED
    assert second_pass[0].retry_count == 2


def test_check_stale_ignores_fresh_heartbeats():
    registry = JobRegistry()
    record = registry.submit()
    registry.heartbeat(record.id, now=T0)
    soon = T0 + timedelta(seconds=5)

    transitioned = registry.check_stale(timeout_seconds=30, now=soon)

    assert transitioned == []
    assert registry.inspect(record.id).status is JobStatus.RUNNING


def test_check_stale_ignores_queued_and_terminal_jobs():
    registry = JobRegistry()
    queued = registry.submit()

    completed = registry.submit()
    registry.heartbeat(completed.id)
    registry.complete(completed.id)

    transitioned = registry.check_stale(timeout_seconds=0.001, now=T0 + timedelta(days=1))

    assert transitioned == []
    assert registry.inspect(queued.id).status is JobStatus.QUEUED
    assert registry.inspect(completed.id).status is JobStatus.COMPLETED


def test_check_stale_rejects_non_positive_timeout():
    registry = JobRegistry()
    with pytest.raises(ValueError):
        registry.check_stale(timeout_seconds=0)


def test_unknown_job_id_raises_not_found_for_every_operation():
    registry = JobRegistry()
    with pytest.raises(JobNotFoundError):
        registry.inspect("job-does-not-exist")
    with pytest.raises(JobNotFoundError):
        registry.heartbeat("job-does-not-exist")
    with pytest.raises(JobNotFoundError):
        registry.cancel("job-does-not-exist")
    with pytest.raises(JobNotFoundError):
        registry.complete("job-does-not-exist")
    with pytest.raises(JobNotFoundError):
        registry.fail("job-does-not-exist", reason=FailureReason.EXECUTOR_ERROR)


def test_list_jobs_filters_by_status():
    registry = JobRegistry()
    queued = registry.submit()
    running = registry.submit()
    registry.heartbeat(running.id)

    assert [j.id for j in registry.list_jobs(status=JobStatus.QUEUED)] == [queued.id]
    assert [j.id for j in registry.list_jobs(status=JobStatus.RUNNING)] == [running.id]
    assert len(registry.list_jobs()) == 2


def test_resource_budget_rejects_non_positive_fields():
    with pytest.raises(ValueError):
        ResourceBudget(cpu_cores=0)
    with pytest.raises(ValueError):
        ResourceBudget(memory_mb=-1)
    with pytest.raises(ValueError):
        ResourceBudget(wall_clock_seconds=0)
    with pytest.raises(ValueError):
        ResourceBudget(gpu_count=-1)
