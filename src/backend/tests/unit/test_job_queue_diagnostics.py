"""Unit tests for the in-memory JobQueueService single-job diagnostic snapshot.

Covers ``JobQueueService.job_diagnostic_snapshot``:

* status derivation: not_started / running / completed (incl. failed tasks) / cancelled
* local pending-event counting without consuming the queue
* explicit not-found for unknown IDs and for jobs after cleanup
* no leakage of event payloads, user identity or exception bodies
* no mutation of the existing metrics_snapshot
* Redis backend refusal (local buffer counts must never pose as global length)
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid

import pytest
from langflow.services.job_queue.service import (
    JobQueueNotFoundError,
    JobQueueService,
    RedisJobQueueService,
)

# Distinctive marker strings used to assert diagnostics never surface
# payloads or error bodies. Names avoid words (e.g. "password") that trip
# security lint heuristics.
_LEAKABLE_PAYLOAD = b"private-event-content-must-not-appear"
_LEAKABLE_ERROR = "private-error-body-must-not-appear"
_SNAPSHOT_KEYS = {"backend", "job_id", "status", "pending_events"}


def _make_service() -> JobQueueService:
    """Return an unstarted in-memory service (no periodic-cleanup task)."""
    service = JobQueueService()
    service._closed = False
    return service


async def _blocking_task(ready: asyncio.Event) -> None:
    """A job body that stays running until the test releases *ready*."""
    await ready.wait()


def _enqueue(queue: asyncio.Queue, count: int) -> None:
    for _ in range(count):
        queue.put_nowait(("event-id", _LEAKABLE_PAYLOAD, time.time()))


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------


async def test_snapshot_not_started_with_empty_queue():
    service = _make_service()
    queue, _ = service.create_queue("job-not-started")

    snapshot = service.job_diagnostic_snapshot("job-not-started")

    assert snapshot["backend"] == "memory"
    assert snapshot["job_id"] == "job-not-started"
    assert snapshot["status"] == JobQueueService.JOB_STATUS_NOT_STARTED
    assert snapshot["pending_events"] == 0
    assert set(snapshot.keys()) == _SNAPSHOT_KEYS
    assert queue.qsize() == 0


async def test_snapshot_running_with_pending_events_does_not_consume_queue():
    service = _make_service()
    queue, _ = service.create_queue("job-running")
    ready = asyncio.Event()
    service.start_job("job-running", _blocking_task(ready))
    _enqueue(queue, 3)

    first = service.job_diagnostic_snapshot("job-running")
    second = service.job_diagnostic_snapshot("job-running")

    assert first["status"] == JobQueueService.JOB_STATUS_RUNNING
    assert first["pending_events"] == 3
    # Read-only: repeated calls must not drain or otherwise alter the queue.
    assert second == first
    assert queue.qsize() == 3

    # The items are still physically retrievable, in order, after the reads.
    drained = [queue.get_nowait() for _ in range(3)]
    assert queue.qsize() == 0
    assert all(item[1] == _LEAKABLE_PAYLOAD for item in drained)

    ready.set()
    _, _, task, _ = service.get_queue_data("job-running")
    await task


async def test_snapshot_completed_after_task_finishes():
    service = _make_service()
    service.create_queue("job-done")

    async def finish_cleanly() -> None:
        await asyncio.sleep(0)

    service.start_job("job-done", finish_cleanly())
    _, _, task, _ = service.get_queue_data("job-done")
    await task

    snapshot = service.job_diagnostic_snapshot("job-done")

    assert snapshot["status"] == JobQueueService.JOB_STATUS_COMPLETED
    assert snapshot["pending_events"] == 0


async def test_snapshot_failed_task_is_completed_without_error_body():
    service = _make_service()
    queue, _ = service.create_queue("job-failed")

    async def fail() -> None:
        await asyncio.sleep(0)
        raise RuntimeError(_LEAKABLE_ERROR)

    service.start_job("job-failed", fail())
    _, _, task, _ = service.get_queue_data("job-failed")
    with contextlib.suppress(RuntimeError):
        await task

    snapshot = service.job_diagnostic_snapshot("job-failed")

    # A crash is a terminal state, not a fifth status, and the error text and
    # the error event payload must not surface through the diagnostic.
    assert snapshot["status"] == JobQueueService.JOB_STATUS_COMPLETED
    assert _LEAKABLE_ERROR not in str(snapshot)
    # The guarded task emits an on_error event + sentinel on crash; their
    # contents count as items but must not be exposed either.
    assert queue.qsize() == snapshot["pending_events"]
    assert _LEAKABLE_ERROR not in repr(snapshot)


async def test_snapshot_cancelled_while_registry_entry_survives_grace_period():
    service = _make_service()
    service.create_queue("job-cancelled")
    ready = asyncio.Event()
    service.start_job("job-cancelled", _blocking_task(ready))
    await asyncio.sleep(0)  # let the guarded task start awaiting the job body
    _, _, task, _ = service.get_queue_data("job-cancelled")

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # cleanup_job re-raises the user-cancel and leaves the entry in place for
    # the grace-period sweep; diagnostics must still resolve the job meanwhile.
    snapshot = service.job_diagnostic_snapshot("job-cancelled")
    assert snapshot["status"] == JobQueueService.JOB_STATUS_CANCELLED


# ---------------------------------------------------------------------------
# Not found: unknown ID and post-cleanup
# ---------------------------------------------------------------------------


async def test_snapshot_unknown_job_raises_not_found():
    service = _make_service()

    with pytest.raises(JobQueueNotFoundError) as exc_info:
        service.job_diagnostic_snapshot("does-not-exist")

    assert exc_info.value.job_id == "does-not-exist"
    # The not-found error must not carry event content or exception bodies.
    assert _LEAKABLE_PAYLOAD.decode() not in str(exc_info.value)


async def test_snapshot_after_cleanup_raises_not_found():
    service = _make_service()
    service.create_queue("job-cleaned-never-started")
    # An orphaned (never started) queue is removed without a cancel re-raise.
    await service.cleanup_job("job-cleaned-never-started")

    with pytest.raises(JobQueueNotFoundError):
        service.job_diagnostic_snapshot("job-cleaned-never-started")


async def test_snapshot_after_cancelled_task_cleanup_raises_not_found():
    service = _make_service()
    queue, _ = service.create_queue("job-cancelled-then-swept")
    _enqueue(queue, 2)
    ready = asyncio.Event()
    service.start_job("job-cancelled-then-swept", _blocking_task(ready))
    await asyncio.sleep(0)  # let the guarded task start awaiting the job body
    _, _, task, _ = service.get_queue_data("job-cancelled-then-swept")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Mirrors the periodic grace-period sweep: the second cleanup sees a done
    # task, drains the buffered items and pops the registry entry.
    await service.cleanup_job("job-cancelled-then-swept")

    assert queue.qsize() == 0
    with pytest.raises(JobQueueNotFoundError):
        service.job_diagnostic_snapshot("job-cancelled-then-swept")


# ---------------------------------------------------------------------------
# Privacy / non-mutation guarantees
# ---------------------------------------------------------------------------


async def test_snapshot_does_not_leak_events_or_user_identity():
    service = _make_service()
    queue, _ = service.create_queue("job-private")
    owner_id = uuid.uuid4()
    await service.register_job_owner("job-private", owner_id)
    _enqueue(queue, 2)

    snapshot = service.job_diagnostic_snapshot("job-private")

    assert set(snapshot.keys()) == _SNAPSHOT_KEYS
    rendered = str(snapshot)
    assert _LEAKABLE_PAYLOAD.decode() not in rendered
    assert str(owner_id) not in rendered


async def test_snapshot_does_not_change_metrics_snapshot():
    service = _make_service()
    service.create_queue("job-metrics-a")
    service.create_queue("job-metrics-b")

    before = service.metrics_snapshot()
    for job_id in ("job-metrics-a", "job-metrics-b", "unknown"):
        with contextlib.suppress(JobQueueNotFoundError):
            service.job_diagnostic_snapshot(job_id)
    after = service.metrics_snapshot()

    assert before == after
    assert before == {"backend": "memory", "active_jobs": 2}


async def test_snapshot_does_not_cancel_running_task():
    service = _make_service()
    service.create_queue("job-still-running")
    ready = asyncio.Event()
    service.start_job("job-still-running", _blocking_task(ready))
    _, _, task_before, _ = service.get_queue_data("job-still-running")

    service.job_diagnostic_snapshot("job-still-running")
    await asyncio.sleep(0)

    assert not task_before.done()
    assert service.job_diagnostic_snapshot("job-still-running")["status"] == "running"
    ready.set()
    await task_before


# ---------------------------------------------------------------------------
# Redis backend is explicitly unsupported
# ---------------------------------------------------------------------------


def test_redis_backend_refuses_diagnostic_snapshot():
    # Construction alone performs no I/O and spawns no tasks.
    service = RedisJobQueueService(cancel_channel_enabled=False)

    with pytest.raises(NotImplementedError, match="in-memory backend"):
        service.job_diagnostic_snapshot("any-job")
