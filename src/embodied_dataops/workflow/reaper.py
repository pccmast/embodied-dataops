"""Lease reaper: PostgreSQL heals every lost worker (SPEC 6.3.6-6.3.7).

Runs every ``reaper_interval_seconds`` (10s) and does four independent jobs:

1. **expire leases** - ``RUNNING`` tasks whose ``lease_until`` passed are moved to
   ``RETRY_WAIT`` (backoff, attempt budget left) or ``FAILED``; metrics
   ``dataops_lease_expirations_total`` and ``dataops_retries_total{reason="lease_timeout"}``
   are incremented.  A crashed worker can therefore never leave a permanent
   ``RUNNING`` task.
2. **promote retry waits** - due ``RETRY_WAIT`` tasks become ``PENDING`` and a new
   outbox wakeup row is written in the same transaction.
3. **re-publish pending** - ``PENDING`` tasks that are due but have no unpublished
   outbox row are re-notified, compensating a lost/never-sent Redis wakeup.
4. **reclaim stale pending entries** - ``XAUTOCLAIM`` entries idle > 2 min are
   acknowledged (they are only hints; #3 re-notifies from PostgreSQL).  Redis is
   never treated as task state.
5. **release stale model reservations** - a ``model_inferences`` row still ``RUNNING``
   long after its worker must have died is settled as ``FAILED``/``STALE_RESERVATION``
   so the budget it reserved is not held forever by a killed process.
"""

from __future__ import annotations

import logging
import signal
import threading
import uuid
from datetime import datetime, timedelta

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from embodied_dataops.config import get_settings
from embodied_dataops.db.models import ImportJob, OutboxEvent, PipelineRun, Task
from embodied_dataops.db.session import session_scope
from embodied_dataops.domain.enums import ImportJobState, RunState, TaskState
from embodied_dataops.logging_setup import configure_logging, log_event
from embodied_dataops.observability.metrics import LEASE_EXPIRATIONS, RETRIES, TASKS_TOTAL
from embodied_dataops.operations.revisions import bump as bump_revision
from embodied_dataops.workflow import queue
from embodied_dataops.workflow.completion import (
    _handle_terminal_failure,
    backoff_delay_seconds,
    cancel_running_task,
    dependencies_satisfied,
)
from embodied_dataops.workflow.outbox import ALL_STREAMS, enqueue_task_wakeup, utcnow

logger = logging.getLogger(__name__)

_RUN_TERMINAL_STATES = (RunState.SUCCEEDED.value, RunState.FAILED.value, RunState.CANCELLED.value)
_JOB_TERMINAL_STATES = (ImportJobState.SUCCEEDED.value, ImportJobState.FAILED.value)


def reap_once(
    session: Session,
    *,
    now: datetime | None = None,
    redis_client=None,
    streams: tuple[str, ...] | None = None,
    commit: bool = True,
    republish_limit: int = 200,
    stale_inference_seconds: int | None = None,
) -> dict[str, int]:
    """One reap cycle; returns counters for logging/testing."""
    now = now or utcnow()
    expired = _expire_leases(session, now)
    promoted = _promote_retry_wait(session, now)
    republished = _republish_pending(session, now, limit=republish_limit)
    reclaimed = _reclaim_stale_entries(redis_client, streams or ALL_STREAMS)
    released = _release_stale_inferences(
        session,
        now,
        older_than_seconds=(
            stale_inference_seconds
            if stale_inference_seconds is not None
            else int(get_settings().vlm_stale_inference_seconds)
        ),
    )
    if commit:
        session.commit()
    stats = {
        "expired": expired,
        "promoted": promoted,
        "republished": republished,
        "reclaimed": reclaimed,
        "released_inferences": released,
    }
    if any(stats.values()):
        logger.info(
            "reaper.cycle expired=%d promoted=%d republished=%d reclaimed=%d released=%d",
            expired,
            promoted,
            republished,
            reclaimed,
            released,
            extra={"event": "reaper.cycle", "extra_fields": {"reaper": stats}},
        )
    return stats


# ---------------------------------------------------------------------------
# 1. expired leases
# ---------------------------------------------------------------------------
def _expire_leases(session: Session, now: datetime) -> int:
    stmt = (
        select(Task)
        .where(
            Task.state == TaskState.RUNNING.value,
            Task.lease_until.is_not(None),
            Task.lease_until < now,
        )
        .order_by(Task.lease_until)
        .with_for_update(skip_locked=True)
    )
    count = 0
    for task in session.scalars(stmt):
        LEASE_EXPIRATIONS.inc()
        count += 1
        if task.cancel_requested:
            # The user asked to cancel; the worker is gone, so stop right here.
            cancel_running_task(session, task.id, task.lease_token or "", now=now, commit=False)
            continue
        if _parent_cancelled(session, task):
            task.state = TaskState.CANCELLED.value
            task.error_code = task.error_code or "CANCEL_REQUESTED"
            task.lease_token = None
            task.lease_until = None
            task.updated_at = now
            bump_revision(session, task)
            TASKS_TOTAL.labels(step=task.step, state=TaskState.CANCELLED.value).inc()
            continue
        if task.attempt >= task.max_attempts:
            task.state = TaskState.FAILED.value
            task.error_code = task.error_code or "LEASE_EXPIRED"
            task.error_detail = task.error_detail or "lease expired and retry budget exhausted"
            task.lease_token = None
            task.lease_until = None
            task.updated_at = now
            bump_revision(session, task)
            _handle_terminal_failure(session, task, now)
            TASKS_TOTAL.labels(step=task.step, state=TaskState.FAILED.value).inc()
            log_event(
                logger,
                "reaper.lease_expired_failed",
                task_id=task.id,
                run_id=task.run_id,
                error_code=task.error_code,
            )
            continue
        delay = backoff_delay_seconds(task.id, task.attempt)
        task.state = TaskState.RETRY_WAIT.value
        task.next_run_at = now + timedelta(seconds=delay)
        task.error_code = task.error_code or "LEASE_EXPIRED"
        task.error_detail = task.error_detail or "worker lease expired"
        task.lease_token = None
        task.lease_until = None
        task.updated_at = now
        bump_revision(session, task)
        RETRIES.labels(reason="lease_timeout").inc()
        TASKS_TOTAL.labels(step=task.step, state=TaskState.RETRY_WAIT.value).inc()
        log_event(
            logger,
            "reaper.lease_expired",
            task_id=task.id,
            run_id=task.run_id,
            error_code="LEASE_EXPIRED",
            delay_seconds=round(delay, 3),
        )
    return count


# ---------------------------------------------------------------------------
# 1b. stale model reservations
# ---------------------------------------------------------------------------
def _stale_inference_statement(now: datetime, *, older_than_seconds: int):
    """The exact set the reaper is allowed to settle (kept separate so it is testable)."""
    from embodied_dataops.db.models import ModelInference  # noqa: PLC0415

    cutoff = now - timedelta(seconds=max(1, int(older_than_seconds)))
    return (
        select(ModelInference)
        .where(
            ModelInference.status == "RUNNING",
            ModelInference.created_at < cutoff,
        )
        .with_for_update(skip_locked=True)
    )


def _release_stale_inferences(session: Session, now: datetime, *, older_than_seconds: int) -> int:
    """Settle ``model_inferences`` rows whose worker died between reserve and settle.

    The reservation row is written *before* the remote call and is what makes budget
    admission atomic.  If the process is killed inside that window, nobody settles it:
    the row stays ``RUNNING`` and its estimate is counted as reserved forever, so the
    run's remaining budget shrinks for no reason.  ``VLM_PREANNOTATE_CLIP`` is bounded
    by a 900 s wall clock, so anything older than ``older_than_seconds`` cannot be a
    live call.

    The cost is set to the sum of the attempt rows that actually recorded one: a call
    that reached the provider is still a spent call, and a call that never left the
    process releases its reservation completely.
    """
    from decimal import Decimal  # noqa: PLC0415 - only needed on the reaping path

    from sqlalchemy import func  # noqa: PLC0415

    from embodied_dataops.db.models import ModelCallAttempt  # noqa: PLC0415

    count = 0
    for inference in session.scalars(_stale_inference_statement(now, older_than_seconds=older_than_seconds)):
        billed = session.scalar(
            select(func.coalesce(func.sum(ModelCallAttempt.estimated_cost_usd), 0)).where(
                ModelCallAttempt.inference_id == inference.id
            )
        )
        inference.status = "FAILED"
        inference.error_code = "STALE_RESERVATION"
        inference.finished_at = now
        inference.estimated_cost_usd = (
            Decimal(str(billed)) if billed not in (None, 0) else None
        )
        count += 1
        log_event(
            logger,
            "reaper.stale_inference_released",
            run_id=inference.run_id,
            error_code="STALE_RESERVATION",
            inference_id=inference.id,
        )
    return count


# ---------------------------------------------------------------------------
# 2. due retries -> PENDING
# ---------------------------------------------------------------------------
def _promote_retry_wait(session: Session, now: datetime) -> int:
    stmt = (
        select(Task)
        .where(
            Task.state == TaskState.RETRY_WAIT.value,
            Task.next_run_at <= now,
            Task.cancel_requested.is_(False),
        )
        .order_by(Task.next_run_at, Task.created_at)
        .with_for_update(skip_locked=True)
    )
    count = 0
    for task in session.scalars(stmt):
        if _parent_cancelled(session, task):
            task.state = TaskState.CANCELLED.value
            task.lease_token = None
            task.lease_until = None
            task.updated_at = now
            bump_revision(session, task)
            TASKS_TOTAL.labels(step=task.step, state=TaskState.CANCELLED.value).inc()
            continue
        if not dependencies_satisfied(session, task.id):
            task.state = TaskState.CANCELLED.value
            task.error_code = task.error_code or "DEPENDENCY_UNSATISFIED"
            task.updated_at = now
            bump_revision(session, task)
            TASKS_TOTAL.labels(step=task.step, state=TaskState.CANCELLED.value).inc()
            continue
        task.state = TaskState.PENDING.value
        task.updated_at = now
        bump_revision(session, task)
        enqueue_task_wakeup(session, task.id, task.pool)
        TASKS_TOTAL.labels(step=task.step, state=TaskState.PENDING.value).inc()
        count += 1
        log_event(logger, "reaper.retry_promoted", task_id=task.id, run_id=task.run_id)
    return count


# ---------------------------------------------------------------------------
# 3. re-publish due PENDING tasks (compensate lost Redis wakeups)
# ---------------------------------------------------------------------------
def _republish_pending(session: Session, now: datetime, *, limit: int = 200) -> int:
    already_queued = exists(
        select(1)
        .select_from(OutboxEvent)
        .where(
            OutboxEvent.published_at.is_(None),
            OutboxEvent.payload_json["task_id"].astext == Task.id,
        )
    )
    stmt = (
        select(Task)
        .where(
            Task.state == TaskState.PENDING.value,
            Task.next_run_at <= now,
            Task.cancel_requested.is_(False),
            ~already_queued,
        )
        .order_by(Task.next_run_at, Task.created_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    count = 0
    for task in session.scalars(stmt):
        if _parent_cancelled(session, task):
            task.state = TaskState.CANCELLED.value
            task.lease_token = None
            task.lease_until = None
            task.updated_at = now
            bump_revision(session, task)
            TASKS_TOTAL.labels(step=task.step, state=TaskState.CANCELLED.value).inc()
            continue
        if not dependencies_satisfied(session, task.id):
            continue
        enqueue_task_wakeup(session, task.id, task.pool)
        count += 1
        log_event(logger, "reaper.wakeup_republished", task_id=task.id, run_id=task.run_id)
    return count


def _parent_cancelled(session: Session, task: Task) -> bool:
    if task.run_id:
        run = session.get(PipelineRun, task.run_id, populate_existing=True)
        if run is not None and (run.cancel_requested or run.state in _RUN_TERMINAL_STATES):
            return True
    if task.import_job_id:
        job = session.get(ImportJob, task.import_job_id, populate_existing=True)
        if job is not None and job.state in _JOB_TERMINAL_STATES:
            return True
    return False


# ---------------------------------------------------------------------------
# 4. stale pending entries in Redis (hints only)
# ---------------------------------------------------------------------------
def _reclaim_stale_entries(redis_client, streams: tuple[str, ...], min_idle_ms: int = 120_000) -> int:
    if redis_client is None:
        return 0
    consumer = f"reaper-{uuid.uuid4().hex[:8]}"
    reclaimed = 0
    for stream in streams:
        try:
            entry_ids = queue.reclaim_stale(
                redis_client, stream, consumer=consumer, min_idle_ms=min_idle_ms
            )
        except Exception:  # pragma: no cover - Redis unavailable
            logger.warning("reaper.xautoclaim_failed stream=%s", stream, exc_info=True)
            continue
        for entry_id in entry_ids:
            try:
                queue.ack(redis_client, stream, entry_id)
                reclaimed += 1
            except Exception:  # pragma: no cover - Redis unavailable
                logger.warning("reaper.ack_failed stream=%s entry=%s", stream, entry_id)
    return reclaimed


# ---------------------------------------------------------------------------
# CLI loop: python -m embodied_dataops.workflow.reaper
# ---------------------------------------------------------------------------
def run_forever(*, stop_event: threading.Event | None = None, redis_client=None) -> None:
    settings = get_settings()
    stop = stop_event or threading.Event()
    interval = settings.reaper_interval_seconds
    while not stop.is_set():
        try:
            with session_scope() as session:
                reap_once(session, now=utcnow(), redis_client=redis_client, commit=False)
        except Exception:  # pragma: no cover - keep the reaper alive
            logger.exception("reaper.cycle_failed")
        stop.wait(interval)


def main() -> int:
    settings = get_settings()
    configure_logging("reaper", settings.log_level)
    redis_client = None
    try:
        candidate = queue.create_client()
        if queue.ping(candidate):
            redis_client = candidate
            for stream in ALL_STREAMS:
                queue.ensure_group(redis_client, stream)
    except Exception:  # pragma: no cover - Redis optional for the DB-only reaper
        logger.warning("reaper.redis_unavailable", exc_info=True)
        redis_client = None
    stop = threading.Event()

    def _stop(signum, _frame):  # pragma: no cover - signal handling
        logger.info("reaper.signal", extra={"event": "reaper.signal", "extra_fields": {"signum": signum}})
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    logger.info(
        "reaper.started",
        extra={
            "event": "reaper.started",
            "extra_fields": {"interval_seconds": settings.reaper_interval_seconds},
        },
    )
    run_forever(stop_event=stop, redis_client=redis_client)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
