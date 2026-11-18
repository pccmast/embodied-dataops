"""Lease acquisition: PostgreSQL row locks are the only claim authority (SPEC 6.3).

A Redis Stream message is a wakeup hint, never a claim.  Every execution begins
with one of the functions below, which:

* takes ``SELECT ... FOR UPDATE SKIP LOCKED`` on an eligible task,
* re-checks state, ``next_run_at``, dependencies and ``cancel_requested`` under
  that lock,
* increments ``attempt`` exactly once per *new database lease* (a duplicated
  Redis notification cannot increase it),
* writes ``RUNNING`` + a fresh ``lease_token`` (UUIDv4) + ``lease_until``,
* flips the parent PipelineRun / ImportJob to RUNNING on the first claim.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session, aliased

from embodied_dataops.config import get_settings
from embodied_dataops.db.models import ImportJob, PipelineRun, Task, TaskDependency
from embodied_dataops.domain.enums import ImportJobState, RunState, TaskState
from embodied_dataops.ids import new_token
from embodied_dataops.logging_setup import log_event
from embodied_dataops.observability.metrics import QUEUE_WAIT, TASKS_TOTAL
from embodied_dataops.workflow.completion import _handle_terminal_failure
from embodied_dataops.workflow.outbox import utcnow

logger = logging.getLogger(__name__)

_CLAIMABLE_STATES = (TaskState.PENDING.value, TaskState.RETRY_WAIT.value)
_RUN_TERMINAL_STATES = (RunState.SUCCEEDED.value, RunState.FAILED.value, RunState.CANCELLED.value)
_JOB_TERMINAL_STATES = (ImportJobState.SUCCEEDED.value, ImportJobState.FAILED.value)


def eligible_clause(now: datetime):
    """SQL clause for tasks claimable at ``now`` (state, delay, cancel, deps)."""
    dependency_task = aliased(Task)
    unsatisfied_dependency = exists(
        select(1)
        .select_from(TaskDependency)
        .join(dependency_task, dependency_task.id == TaskDependency.depends_on_task_id)
        .where(
            TaskDependency.task_id == Task.id,
            dependency_task.state != TaskState.SUCCEEDED.value,
        )
    )
    return and_(
        Task.state.in_(_CLAIMABLE_STATES),
        Task.next_run_at <= now,
        Task.cancel_requested.is_(False),
        ~unsatisfied_dependency,
    )


def claim_next_task(
    session: Session,
    pool: str,
    worker_id: str,
    *,
    now: datetime | None = None,
    commit: bool = True,
) -> Task | None:
    """Claim the oldest eligible task for ``pool`` or return ``None``.

    ``SKIP LOCKED`` lets two workers compete: the loser skips the locked row and
    either takes another task or returns ``None`` without blocking.
    """
    now = now or utcnow()
    stmt = (
        select(Task)
        .where(eligible_clause(now), Task.pool == pool)
        .order_by(Task.next_run_at, Task.created_at, Task.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    task = session.scalars(stmt).first()
    if task is None:
        if commit:
            session.rollback()
        return None
    claimed = _claim_locked(session, task, worker_id, now)
    if commit:
        session.commit()
    return claimed


def claim_task_by_id(
    session: Session,
    task_id: str,
    worker_id: str,
    *,
    pool: str | None = None,
    now: datetime | None = None,
    commit: bool = True,
) -> Task | None:
    """Claim a specific task (used when a worker consumes a task-id wakeup)."""
    now = now or utcnow()
    task = session.scalars(select(Task).where(Task.id == task_id).with_for_update()).first()
    if task is None:
        if commit:
            session.rollback()
        return None
    if not _is_eligible(session, task, now, pool=pool):
        if commit:
            session.rollback()
        return None
    claimed = _claim_locked(session, task, worker_id, now)
    if commit:
        session.commit()
    return claimed


def _is_eligible(session: Session, task: Task, now: datetime, *, pool: str | None = None) -> bool:
    if task.state not in _CLAIMABLE_STATES:
        return False
    if task.next_run_at is None or task.next_run_at > now:
        return False
    if task.cancel_requested:
        return False
    if pool is not None and task.pool != pool:
        return False
    dependency_ids = list(
        session.scalars(select(TaskDependency.depends_on_task_id).where(TaskDependency.task_id == task.id))
    )
    if dependency_ids:
        states = list(session.scalars(select(Task.state).where(Task.id.in_(dependency_ids))))
        if len(states) != len(dependency_ids) or any(
            state != TaskState.SUCCEEDED.value for state in states
        ):
            return False
    return True


def _claim_locked(session: Session, task: Task, worker_id: str, now: datetime) -> Task | None:
    """Apply the claim mutation to an already row-locked, eligibility-checked task."""
    settings = get_settings()

    if task.attempt + 1 > task.max_attempts:
        # Safety net: a task that reached the attempt limit before a new lease is
        # granted must never run again (SPEC 6.2).
        task.state = TaskState.FAILED.value
        task.error_code = task.error_code or "MAX_ATTEMPTS_EXCEEDED"
        task.error_detail = task.error_detail or "attempt limit reached before claim"
        task.lease_token = None
        task.lease_until = None
        task.updated_at = now
        _handle_terminal_failure(session, task, now)
        TASKS_TOTAL.labels(step=task.step, state=TaskState.FAILED.value).inc()
        log_event(
            logger,
            "task.claim.attempt_limit",
            task_id=task.id,
            run_id=task.run_id,
            error_code="MAX_ATTEMPTS_EXCEEDED",
        )
        return None

    cancel_reason = _parent_cancel_reason(session, task)
    if cancel_reason is not None:
        task.state = TaskState.CANCELLED.value
        task.error_code = task.error_code or cancel_reason
        task.lease_token = None
        task.lease_until = None
        task.updated_at = now
        TASKS_TOTAL.labels(step=task.step, state=TaskState.CANCELLED.value).inc()
        log_event(logger, "task.claim.parent_cancelled", task_id=task.id, error_code=cancel_reason)
        return None

    task.attempt += 1
    task.state = TaskState.RUNNING.value
    task.lease_token = new_token()
    task.lease_until = now + timedelta(seconds=settings.lease_seconds)
    task.worker_id = worker_id
    task.updated_at = now

    if task.run_id:
        run = session.get(PipelineRun, task.run_id, populate_existing=True)
        if run is not None:
            if run.state == RunState.QUEUED.value:
                run.state = RunState.RUNNING.value
            if run.started_at is None:
                run.started_at = now
    if task.import_job_id:
        job = session.get(ImportJob, task.import_job_id, populate_existing=True)
        if job is not None:
            if job.state == ImportJobState.QUEUED.value:
                job.state = ImportJobState.RUNNING.value
            if job.started_at is None:
                job.started_at = now

    if task.created_at is not None:
        try:
            QUEUE_WAIT.labels(step=task.step).observe(max(0.0, (now - task.created_at).total_seconds()))
        except TypeError:  # pragma: no cover - naive timestamp
            pass
    TASKS_TOTAL.labels(step=task.step, state=TaskState.RUNNING.value).inc()
    log_event(
        logger,
        "task.claimed",
        task_id=task.id,
        run_id=task.run_id,
        worker_id=worker_id,
        attempt=task.attempt,
    )
    return task


def _parent_cancel_reason(session: Session, task: Task) -> str | None:
    """A pending task must not run when its run is cancelled or its job already failed."""
    if task.run_id:
        run = session.get(PipelineRun, task.run_id, populate_existing=True)
        if run is not None and (run.cancel_requested or run.state in _RUN_TERMINAL_STATES):
            return "CANCEL_REQUESTED" if run.cancel_requested else "RUN_ALREADY_TERMINAL"
    if task.import_job_id:
        job = session.get(ImportJob, task.import_job_id, populate_existing=True)
        if job is not None and job.state in _JOB_TERMINAL_STATES:
            return "IMPORT_JOB_TERMINAL"
    return None


__all__ = ["eligible_clause", "claim_next_task", "claim_task_by_id"]
