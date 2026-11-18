"""Task completion, heartbeat, failure classification and retry backoff (SPEC 6.2-6.4).

The engine runs a handler inside the task's database transaction.  The *only*
success path for a worker is a fenced compare-and-set::

    UPDATE tasks
       SET state='SUCCEEDED', output_asset_id=..., output_json=...
     WHERE id = :id AND state = 'RUNNING' AND lease_token = :token

An old worker whose lease was reclaimed can therefore never overwrite the new
worker's output.  When the CAS fails the whole transaction is rolled back, which
also discards every row (assets, quality reports, ...) the handler created in
this attempt.  Asset registration itself belongs to handlers
(``workflow.handlers.base.register_bytes/register_file/register_json``); the
engine only links ``HandlerResult.output_asset_id``.

Downstream wakeups are written to the outbox in the *same* transaction as the
successful CAS, so PostgreSQL remains the single truth source.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from embodied_dataops.config import get_settings
from embodied_dataops.db.models import (
    Asset,
    Episode,
    ImportJob,
    PipelineRun,
    SourceImport,
    Task,
    TaskDependency,
)
from embodied_dataops.domain.enums import (
    EpisodeStatus,
    ImportJobState,
    RunState,
    TaskState,
)
from embodied_dataops.errors import ErrorCode
from embodied_dataops.logging_setup import log_event
from embodied_dataops.observability.metrics import RETRIES, TASK_DURATION, TASKS_TOTAL
from embodied_dataops.operations.revisions import bump as bump_revision
from embodied_dataops.workflow.outbox import enqueue_task_wakeup, utcnow

logger = logging.getLogger(__name__)

#: Fraction of deterministic jitter applied to every retry delay (SPEC 6.2).
BACKOFF_JITTER_FRACTION = 0.10

_ACTIVE_TASK_STATES = (TaskState.PENDING.value, TaskState.RUNNING.value, TaskState.RETRY_WAIT.value)
_RUN_TERMINAL_STATES = (RunState.SUCCEEDED.value, RunState.FAILED.value, RunState.CANCELLED.value)


# ---------------------------------------------------------------------------
# Deterministic retry backoff (SPEC 6.2, AT-14)
# ---------------------------------------------------------------------------
def jitter_unit(task_id: str, attempt: int) -> float:
    """Deterministic value in ``[0, 1)`` derived only from task id and attempt."""
    digest = hashlib.sha256(f"retry-backoff:{task_id}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def backoff_delay_seconds(task_id: str, attempt: int, *, settings=None) -> float:
    """Delay before retry number ``attempt`` (1 -> 5s, 2 -> 20s) with ±10% jitter.

    The jitter is deterministic (``jitter_unit``), so tests can assert exact
    values without sleeping and support can reproduce any schedule.
    """
    schedules = (settings or get_settings()).retry_backoff_seconds
    if not schedules:
        return 0.0
    index = min(max(int(attempt), 1), len(schedules)) - 1
    base = float(schedules[index])
    return base * (1.0 + BACKOFF_JITTER_FRACTION * (2.0 * jitter_unit(task_id, attempt) - 1.0))


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------
def heartbeat(session: Session, task_id: str, token: str, *, now: datetime | None = None, commit: bool = True) -> bool:
    """Renew a RUNNING lease, but only for the current fencing token.

    Returns ``False`` when the token is stale, the task is no longer RUNNING or
    the lease has already expired (an expired lease must be reclaimed by the
    reaper instead of silently revived).
    """
    now = now or utcnow()
    settings = get_settings()
    result = session.execute(
        update(Task)
        .where(
            Task.id == task_id,
            Task.state == TaskState.RUNNING.value,
            Task.lease_token == token,
            Task.lease_until.is_not(None),
            Task.lease_until >= now,
        )
        .values(lease_until=now + timedelta(seconds=settings.lease_seconds), updated_at=now)
    )
    if result.rowcount != 1:
        session.rollback()
        return False
    if commit:
        session.commit()
    return True


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------
def complete_task(
    session: Session,
    task_id: str,
    token: str,
    result: Any = None,
    *,
    now: datetime | None = None,
    commit: bool = True,
) -> bool:
    """Commit one successful task execution with the fencing CAS (SPEC 6.3.5).

    ``result`` is a :class:`~embodied_dataops.workflow.handlers.base.HandlerResult`
    (or ``None`` for side-effect-only handlers).  The handler already ran in this
    transaction; a stale token rolls the entire transaction back and returns
    ``False``.
    """
    now = now or utcnow()
    data = _json_safe(dict(getattr(result, "data", None) or {}))
    output_asset_id = getattr(result, "output_asset_id", None)
    if output_asset_id is not None:
        asset = session.get(Asset, str(output_asset_id))
        if asset is None:
            raise ValueError(f"HandlerResult.output_asset_id {output_asset_id!r} is not a registered asset")

    cas = session.execute(
        update(Task)
        .where(
            Task.id == task_id,
            Task.state == TaskState.RUNNING.value,
            Task.lease_token == token,
        )
        .values(
            state=TaskState.SUCCEEDED.value,
            lease_token=None,
            lease_until=None,
            output_asset_id=output_asset_id,
            output_json=data,
            updated_at=now,
        )
    )
    if cas.rowcount != 1:
        session.rollback()
        log_event(logger, "task.complete.rejected_stale_token", task_id=task_id)
        return False

    task = session.get(Task, task_id, populate_existing=True)
    if task is None:  # pragma: no cover - row deleted mid-flight
        session.rollback()
        return False

    verdict = getattr(result, "verdict", None)
    if verdict:
        task.output_json = {**data, "verdict": str(verdict)}
    episode_status = getattr(result, "episode_status", None)
    if episode_status:
        _apply_episode_status(session, task, str(episode_status), now)
    # DO-A02 C2: entering a terminal task state is a business change -> new revision.
    bump_revision(session, task)

    TASKS_TOTAL.labels(step=task.step, state=TaskState.SUCCEEDED.value).inc()
    TASK_DURATION.labels(step=task.step).observe(_task_duration_seconds(task, now))
    log_event(logger, "task.succeeded", task_id=task.id, run_id=task.run_id, worker_id=task.worker_id)

    # A FAIL verdict means the step produced a valid *negative diagnosis* (QC
    # report, undecodable source, ...): commit it, start no downstream work and
    # finalise the run/import job as FAILED.
    if verdict == "FAIL":
        _fail_after_verdict(session, task, now)
    else:
        _wake_downstream(session, task)
        _maybe_finish_run(session, task.run_id, now)
        _maybe_finish_import_job(session, task.import_job_id, now)
    if commit:
        session.commit()
    return True


def _json_safe(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure ``output_json`` is JSONB-serialisable without losing structure."""
    try:
        return json.loads(json.dumps(data, ensure_ascii=False, default=str))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {"repr": str(data)}


def _task_duration_seconds(task: Task, now: datetime) -> float:
    started = task.created_at
    if started is None:  # pragma: no cover - server default always set
        return 0.0
    try:
        return max(0.0, (now - started).total_seconds())
    except TypeError:  # pragma: no cover - naive timestamp from a foreign driver
        return 0.0


def _apply_episode_status(session: Session, task: Task, status_value: str, now: datetime) -> None:
    from embodied_dataops.domain.state_machines import EPISODE_STATUS_TRANSITIONS, can_transition

    if status_value not in {status.value for status in EpisodeStatus}:
        logger.warning("task.episode_status.invalid task=%s status=%s", task.id, status_value)
        return
    episode = _episode_for_task(session, task)
    if episode is None:
        return
    if episode.status == status_value:
        return
    if not can_transition(EPISODE_STATUS_TRANSITIONS, EpisodeStatus(episode.status), EpisodeStatus(status_value)):
        logger.warning(
            "task.episode_status.illegal episode=%s %s -> %s", episode.id, episode.status, status_value
        )
        return
    episode.status = status_value
    bump_revision(session, episode)
    episode.updated_at = now


def _episode_for_task(session: Session, task: Task) -> Episode | None:
    if not task.run_id:
        return None
    run = session.get(PipelineRun, task.run_id, populate_existing=True)
    if run is None:
        return None
    return session.get(Episode, run.episode_id, populate_existing=True)


# ---------------------------------------------------------------------------
# Downstream wakeups and run/job finalisation
# ---------------------------------------------------------------------------
def _wake_downstream(session: Session, task: Task) -> None:
    """Enqueue outbox wakeups for children whose dependencies are now satisfied."""
    if task.run_id:
        run = session.get(PipelineRun, task.run_id, populate_existing=True)
        if run is not None and run.cancel_requested:
            return
    child_ids = list(
        session.scalars(select(TaskDependency.task_id).where(TaskDependency.depends_on_task_id == task.id))
    )
    for child_id in child_ids:
        child = session.get(Task, child_id, populate_existing=True)
        if child is None or child.state != TaskState.PENDING.value or child.cancel_requested:
            continue
        if not dependencies_satisfied(session, child.id):
            continue
        enqueue_task_wakeup(session, child.id, child.pool)
        log_event(logger, "task.downstream_enqueued", task_id=child.id, run_id=child.run_id)


def dependencies_satisfied(session: Session, task_id: str, *, exclude: Iterable[str] = ()) -> bool:
    """All declared dependencies of ``task_id`` are SUCCEEDED."""
    deps = list(
        session.scalars(select(TaskDependency.depends_on_task_id).where(TaskDependency.task_id == task_id))
    )
    excluded = set(exclude)
    deps = [dep for dep in deps if dep not in excluded]
    if not deps:
        return True
    succeeded = set(
        session.scalars(select(Task.id).where(Task.id.in_(deps), Task.state == TaskState.SUCCEEDED.value))
    )
    return len(succeeded) == len(deps)


def _maybe_finish_run(session: Session, run_id: str | None, now: datetime) -> None:
    if not run_id:
        return
    run = session.get(PipelineRun, run_id, populate_existing=True)
    if run is None or run.state in _RUN_TERMINAL_STATES:
        return
    states = list(session.scalars(select(Task.state).where(Task.run_id == run_id)))
    if not states or any(state in _ACTIVE_TASK_STATES for state in states):
        return
    if TaskState.FAILED.value in states:
        target = RunState.FAILED.value
    elif all(state == TaskState.SUCCEEDED.value for state in states):
        target = RunState.CANCELLED.value if run.cancel_requested else RunState.SUCCEEDED.value
    else:
        target = RunState.CANCELLED.value if run.cancel_requested else RunState.FAILED.value
    run.state = target
    run.finished_at = now
    if target == RunState.FAILED.value and not run.error_code:
        run.error_code = ErrorCode.INTERNAL_ERROR.value
    log_event(logger, "run.finished", run_id=run.id)


def _maybe_finish_import_job(session: Session, import_job_id: str | None, now: datetime) -> None:
    if not import_job_id:
        return
    job = session.get(ImportJob, import_job_id, populate_existing=True)
    if job is None or job.state in (ImportJobState.SUCCEEDED.value, ImportJobState.FAILED.value):
        return
    states = list(session.scalars(select(Task.state).where(Task.import_job_id == import_job_id)))
    if not states or any(state in _ACTIVE_TASK_STATES for state in states):
        return
    succeeded = all(state == TaskState.SUCCEEDED.value for state in states)
    job.state = ImportJobState.SUCCEEDED.value if succeeded else ImportJobState.FAILED.value
    job.finished_at = now
    if not succeeded and not job.error_code:
        job.error_code = ErrorCode.INTERNAL_ERROR.value
    source_import = session.get(SourceImport, job.import_id, populate_existing=True)
    if source_import is not None:
        if succeeded:
            source_import.upload_state = "READY"
            source_import.completed_at = source_import.completed_at or now
        else:
            source_import.upload_state = "FAILED"
            source_import.error_code = source_import.error_code or job.error_code
            source_import.error_detail = source_import.error_detail or job.error_detail
    log_event(logger, "import_job.finished")


def _fail_after_verdict(session: Session, task: Task, now: datetime) -> None:
    """A handler reported ``verdict="FAIL"``: keep its diagnostics, stop downstream work.

    * Episode pipeline step (QC FAIL, undecodable normalize, ...): the run becomes
      FAILED, the Episode ``QUALITY_FAILED`` and every PENDING/RETRY_WAIT
      downstream task is CANCELLED.  The producing task itself stays SUCCEEDED and
      its report/evidence rows are committed (SPEC 6.1).
    * Import-level step (source cannot be imported): the ImportJob is FAILED, the
      remaining import tasks are cancelled; per-Episode PipelineRuns are untouched
      because none may have been created yet.
    """
    if task.run_id:
        run = session.get(PipelineRun, task.run_id, populate_existing=True)
        if run is not None and run.state not in _RUN_TERMINAL_STATES:
            _cancel_active_tasks(session, run_id=run.id, now=now, reason="step_failed", exclude_task_id=task.id)
            run.state = RunState.FAILED.value
            run.error_code = run.error_code or stage_error_code(task)
            run.finished_at = now
            episode = session.get(Episode, run.episode_id, populate_existing=True)
            if episode is not None and episode.status != EpisodeStatus.QUALITY_FAILED.value:
                if episode.status in (
                    EpisodeStatus.IMPORTED.value,
                    EpisodeStatus.PROCESSING.value,
                    EpisodeStatus.READY_FOR_REVIEW.value,
                ):
                    episode.status = EpisodeStatus.QUALITY_FAILED.value
                    episode.updated_at = now
                    bump_revision(session, episode)
            log_event(
                logger,
                "run.failed_verdict",
                run_id=run.id,
                task_id=task.id,
                error_code=run.error_code,
            )
    if task.import_job_id:
        job = session.get(ImportJob, task.import_job_id, populate_existing=True)
        if job is not None and job.state not in (
            ImportJobState.SUCCEEDED.value,
            ImportJobState.FAILED.value,
        ):
            _cancel_active_tasks(
                session,
                import_job_id=job.id,
                now=now,
                reason="step_failed",
                exclude_task_id=task.id,
            )
            job.state = ImportJobState.FAILED.value
            job.error_code = job.error_code or stage_error_code(task)
            job.error_detail = job.error_detail or task.error_detail
            job.finished_at = now
            source_import = session.get(SourceImport, job.import_id, populate_existing=True)
            if source_import is not None:
                source_import.upload_state = "FAILED"
                source_import.error_code = source_import.error_code or job.error_code
            log_event(logger, "import_job.failed_verdict", task_id=task.id, error_code=job.error_code)


def stage_error_code(task: Task) -> str:
    """Error code to record for a negative verdict produced by this step."""
    data = task.output_json if isinstance(task.output_json, dict) else {}
    if data.get("error_code"):
        return str(data["error_code"])
    return "QUALITY_FAILED" if task.step == "QUALITY_CHECK" else "STEP_FAILED"


def _cancel_active_tasks(
    session: Session,
    *,
    run_id: str | None = None,
    import_job_id: str | None = None,
    now: datetime,
    reason: str,
    exclude_task_id: str | None = None,
) -> int:
    conditions = [Task.state.in_((TaskState.PENDING.value, TaskState.RETRY_WAIT.value))]
    if run_id:
        conditions.append(Task.run_id == run_id)
    if import_job_id:
        conditions.append(Task.import_job_id == import_job_id)
    stmt = select(Task).where(*conditions).with_for_update(skip_locked=True)
    cancelled = 0
    for pending in session.scalars(stmt):
        if exclude_task_id and pending.id == exclude_task_id:
            continue
        pending.state = TaskState.CANCELLED.value
        pending.lease_token = None
        pending.lease_until = None
        pending.updated_at = now
        pending.error_code = pending.error_code or reason
        bump_revision(session, pending)
        cancelled += 1
        TASKS_TOTAL.labels(step=pending.step, state=TaskState.CANCELLED.value).inc()
    return cancelled


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------
def fail_task(
    session: Session,
    task_id: str,
    token: str,
    error: BaseException,
    *,
    retryable: bool,
    reason: str | None = None,
    now: datetime | None = None,
    commit: bool = True,
) -> str | None:
    """Record a failed execution: RETRY_WAIT with backoff or terminal FAILED.

    Returns the new task state, or ``None`` when the fencing token is stale (the
    old worker is not allowed to touch the task any more).
    """
    now = now or utcnow()
    task = session.scalars(select(Task).where(Task.id == task_id).with_for_update()).first()
    if task is None or task.state != TaskState.RUNNING.value or task.lease_token != token:
        session.rollback()
        log_event(logger, "task.fail.rejected_stale_token", task_id=task_id)
        return None

    code = getattr(error, "code", None)
    code = code.value if hasattr(code, "value") else (str(code) if code else ErrorCode.INTERNAL_ERROR.value)
    message = getattr(error, "message", None) or str(error) or error.__class__.__name__
    task.error_code = code
    task.error_detail = message[:4000]
    task.lease_token = None
    task.lease_until = None
    task.updated_at = now

    if retryable and task.attempt < task.max_attempts:
        delay = backoff_delay_seconds(task.id, task.attempt)
        task.state = TaskState.RETRY_WAIT.value
        task.next_run_at = now + timedelta(seconds=delay)
        bump_revision(session, task)
        RETRIES.labels(reason=reason or code).inc()
        TASKS_TOTAL.labels(step=task.step, state=TaskState.RETRY_WAIT.value).inc()
        log_event(
            logger,
            "task.retry_wait",
            task_id=task.id,
            run_id=task.run_id,
            error_code=code,
            delay_seconds=round(delay, 3),
        )
        if commit:
            session.commit()
        return TaskState.RETRY_WAIT.value

    task.state = TaskState.FAILED.value
    bump_revision(session, task)
    _handle_terminal_failure(session, task, now)
    TASKS_TOTAL.labels(step=task.step, state=TaskState.FAILED.value).inc()
    log_event(logger, "task.failed", task_id=task.id, run_id=task.run_id, error_code=code)
    if commit:
        session.commit()
    return TaskState.FAILED.value


def _handle_terminal_failure(session: Session, task: Task, now: datetime) -> None:
    """Propagate a terminal task failure to its run/import job and downstream tasks."""
    if task.run_id:
        run = session.get(PipelineRun, task.run_id, populate_existing=True)
        if run is not None and run.state not in _RUN_TERMINAL_STATES:
            cancelled = _cancel_active_tasks(
                session, run_id=run.id, now=now, reason="upstream_failed", exclude_task_id=task.id
            )
            run.state = RunState.CANCELLED.value if run.cancel_requested else RunState.FAILED.value
            run.finished_at = now
            run.error_code = run.error_code or task.error_code
            episode = session.get(Episode, run.episode_id, populate_existing=True)
            if episode is not None and episode.status in (
                EpisodeStatus.IMPORTED.value,
                EpisodeStatus.PROCESSING.value,
            ):
                episode.status = EpisodeStatus.QUALITY_FAILED.value
                episode.updated_at = now
                bump_revision(session, episode)
            log_event(
                logger,
                "run.failed",
                run_id=run.id,
                task_id=task.id,
                error_code=task.error_code,
                cancelled_tasks=cancelled,
            )
    if task.import_job_id:
        job = session.get(ImportJob, task.import_job_id, populate_existing=True)
        if job is not None and job.state not in (
            ImportJobState.SUCCEEDED.value,
            ImportJobState.FAILED.value,
        ):
            _cancel_active_tasks(
                session,
                import_job_id=job.id,
                now=now,
                reason="upstream_failed",
                exclude_task_id=task.id,
            )
            job.state = ImportJobState.FAILED.value
            job.error_code = job.error_code or task.error_code
            job.error_detail = job.error_detail or task.error_detail
            job.finished_at = now
            source_import = session.get(SourceImport, job.import_id, populate_existing=True)
            if source_import is not None and source_import.upload_state != "FAILED":
                source_import.upload_state = "FAILED"
                source_import.error_code = source_import.error_code or task.error_code
                source_import.error_detail = source_import.error_detail or task.error_detail
            log_event(logger, "import_job.failed", task_id=task.id, error_code=task.error_code)


# ---------------------------------------------------------------------------
# Cooperative cancellation
# ---------------------------------------------------------------------------
def cancel_running_task(
    session: Session,
    task_id: str,
    token: str,
    *,
    now: datetime | None = None,
    commit: bool = True,
) -> bool:
    """CAS RUNNING -> CANCELLED (cooperative stop) and stop the run's downstream work."""
    now = now or utcnow()
    cas = session.execute(
        update(Task)
        .where(
            Task.id == task_id,
            Task.state == TaskState.RUNNING.value,
            Task.lease_token == token,
        )
        .values(
            state=TaskState.CANCELLED.value,
            lease_token=None,
            lease_until=None,
            error_code=ErrorCode.CANCEL_REQUESTED.value,
            updated_at=now,
        )
    )
    if cas.rowcount != 1:
        session.rollback()
        return False
    task = session.get(Task, task_id, populate_existing=True)
    if task is None:  # pragma: no cover
        session.rollback()
        return False
    bump_revision(session, task)
    TASKS_TOTAL.labels(step=task.step, state=TaskState.CANCELLED.value).inc()
    if task.run_id:
        run = session.get(PipelineRun, task.run_id, populate_existing=True)
        if run is not None and run.state not in _RUN_TERMINAL_STATES:
            _cancel_active_tasks(session, run_id=run.id, now=now, reason="cancelled", exclude_task_id=task.id)
            run.state = RunState.CANCELLED.value
            run.finished_at = now
            log_event(logger, "run.cancelled", run_id=run.id, task_id=task.id)
    if task.import_job_id:
        job = session.get(ImportJob, task.import_job_id, populate_existing=True)
        if job is not None and job.state not in (
            ImportJobState.SUCCEEDED.value,
            ImportJobState.FAILED.value,
        ):
            _cancel_active_tasks(
                session, import_job_id=job.id, now=now, reason="cancelled", exclude_task_id=task.id
            )
            job.state = ImportJobState.FAILED.value
            job.error_code = ErrorCode.CANCEL_REQUESTED.value
            job.finished_at = now
    log_event(logger, "task.cancelled", task_id=task.id, run_id=task.run_id)
    if commit:
        session.commit()
    return True


__all__ = [
    "BACKOFF_JITTER_FRACTION",
    "backoff_delay_seconds",
    "jitter_unit",
    "heartbeat",
    "complete_task",
    "fail_task",
    "cancel_running_task",
    "dependencies_satisfied",
]
