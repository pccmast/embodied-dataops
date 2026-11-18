"""Operator-facing workflow commands used by the HTTP API (SPEC 9.3, 6.2).

These helpers exist so routes never manipulate task rows directly: cancellation
is idempotent and enforces the documented state machine, and a failed task is
retried by creating a **new** PipelineRun from the failed step (the old run is
kept intact for audit).
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import Episode, ImportJob, PipelineRun, Task
from embodied_dataops.domain.enums import ImportJobState, RunState, Step, TaskState
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.observability.metrics import TASKS_TOTAL
from embodied_dataops.operations.revisions import bump as bump_revision
from embodied_dataops.workflow.dag import create_pipeline_run
from embodied_dataops.workflow.outbox import utcnow

logger = logging.getLogger(__name__)

_RUN_TERMINAL = (RunState.SUCCEEDED.value, RunState.FAILED.value, RunState.CANCELLED.value)
_TASK_TERMINAL = (TaskState.SUCCEEDED.value, TaskState.FAILED.value, TaskState.CANCELLED.value)
_CANCELLABLE = (TaskState.PENDING.value, TaskState.RETRY_WAIT.value)


def request_run_cancel(
    session: Session,
    run_id: str,
    *,
    now: datetime | None = None,
    commit: bool = True,
) -> dict:
    """Idempotently cancel a PipelineRun (SPEC 6.2 state machine).

    * PENDING / RETRY_WAIT tasks are CANCELLED immediately,
    * RUNNING tasks only get ``cancel_requested=true`` (the worker exits
      cooperatively and the CAS moves it to CANCELLED),
    * the run becomes CANCELLED once no RUNNING task is left.
    """
    now = now or utcnow()
    run = session.get(PipelineRun, run_id, populate_existing=True)
    if run is None:
        raise ApiError(ErrorCode.NOT_FOUND, "PipelineRun not found", details={"run_id": run_id})
    result = {"run_id": run.id, "state": run.state, "cancelled_tasks": 0, "running_tasks": 0}
    if run.state in _RUN_TERMINAL:
        result["already_terminal"] = True
        return result

    run.cancel_requested = True
    tasks = list(
        session.scalars(select(Task).where(Task.run_id == run.id).with_for_update(skip_locked=True))
    )
    for task in tasks:
        if task.state in _CANCELLABLE:
            task.state = TaskState.CANCELLED.value
            task.cancel_requested = True
            task.error_code = task.error_code or ErrorCode.CANCEL_REQUESTED.value
            task.lease_token = None
            task.lease_until = None
            task.updated_at = now
            bump_revision(session, task)
            result["cancelled_tasks"] += 1
            TASKS_TOTAL.labels(step=task.step, state=TaskState.CANCELLED.value).inc()
        elif task.state == TaskState.RUNNING.value:
            task.cancel_requested = True
            task.updated_at = now
            result["running_tasks"] += 1
    if result["running_tasks"] == 0:
        run.state = RunState.CANCELLED.value
        run.finished_at = now
    result["state"] = run.state
    logger.info(
        "run.cancel_requested run_id=%s cancelled=%d running=%d",
        run.id,
        result["cancelled_tasks"],
        result["running_tasks"],
    )
    if commit:
        session.commit()
    return result


def request_task_cancel(
    session: Session,
    task_id: str,
    *,
    now: datetime | None = None,
    commit: bool = True,
) -> dict:
    """Cancel one task; a PipelineRun cannot continue without one of its steps."""
    now = now or utcnow()
    task = session.scalars(select(Task).where(Task.id == task_id).with_for_update()).first()
    if task is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Task not found", details={"task_id": task_id})
    if task.state in _TASK_TERMINAL:
        return {"task_id": task.id, "state": task.state, "already_terminal": True}
    if task.run_id:
        session.rollback()  # release the row lock before cancelling the whole run
        return request_run_cancel(session, task.run_id, now=now, commit=commit)

    task.state = TaskState.CANCELLED.value
    task.cancel_requested = True
    task.error_code = task.error_code or ErrorCode.CANCEL_REQUESTED.value
    task.updated_at = now
    bump_revision(session, task)
    TASKS_TOTAL.labels(step=task.step, state=TaskState.CANCELLED.value).inc()
    if task.import_job_id:
        job = session.get(ImportJob, task.import_job_id, populate_existing=True)
        if job is not None and job.state not in (
            ImportJobState.SUCCEEDED.value,
            ImportJobState.FAILED.value,
        ):
            for sibling in session.scalars(
                select(Task)
                .where(
                    Task.import_job_id == job.id,
                    Task.id != task.id,
                    Task.state.in_(_CANCELLABLE),
                )
                .with_for_update(skip_locked=True)
            ):
                sibling.state = TaskState.CANCELLED.value
                sibling.cancel_requested = True
                sibling.error_code = sibling.error_code or ErrorCode.CANCEL_REQUESTED.value
                sibling.updated_at = now
                bump_revision(session, sibling)
            job.state = ImportJobState.FAILED.value
            job.error_code = ErrorCode.CANCEL_REQUESTED.value
            job.finished_at = now
    if commit:
        session.commit()
    return {"task_id": task.id, "state": task.state, "already_terminal": False}


def retry_failed_task(
    session: Session,
    task_id: str,
    *,
    requested_by: str | None = None,
    pipeline_version: str | None = None,
    commit: bool = True,
) -> PipelineRun:
    """POST /tasks/{id}/retry: new run from the failed step + its downstream closure."""
    from embodied_dataops import PIPELINE_VERSION

    task = session.get(Task, task_id, populate_existing=True)
    if task is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Task not found", details={"task_id": task_id})
    if task.state != TaskState.FAILED.value:
        raise ApiError(
            ErrorCode.TASK_NOT_RETRIABLE,
            "Only FAILED tasks can be retried",
            details={"task_id": task.id, "state": task.state},
        )
    if not task.run_id:
        raise ApiError(
            ErrorCode.TASK_NOT_RETRIABLE,
            "Import-level tasks are retried by creating a new import, not a new pipeline run",
            details={"task_id": task.id},
        )
    run = session.get(PipelineRun, task.run_id, populate_existing=True)
    if run is None:  # pragma: no cover - FK guarantees the run exists
        raise ApiError(ErrorCode.NOT_FOUND, "PipelineRun not found", details={"run_id": task.run_id})
    episode = session.get(Episode, run.episode_id, populate_existing=True)
    if episode is None:  # pragma: no cover
        raise ApiError(ErrorCode.NOT_FOUND, "Episode not found", details={"episode_id": run.episode_id})

    new_run = create_pipeline_run(
        session,
        episode,
        requested_by=requested_by,
        pipeline_version=pipeline_version or PIPELINE_VERSION,
        from_step=Step(task.step),
    )
    logger.info(
        "task.retry_new_run task_id=%s old_run_id=%s new_run_id=%s",
        task.id,
        run.id,
        new_run.id,
    )
    if commit:
        session.commit()
    return new_run


def _task_retryable(task: Task) -> bool:
    """DO-A02: expose the retry eligibility flag without importing at module load."""
    from embodied_dataops.operations.effects import is_task_retryable

    return is_task_retryable(task)


def run_task_summary(session: Session, run_id: str) -> dict:
    """Compact run/task projection used by GET /pipeline-runs/{id}."""
    run = session.get(PipelineRun, run_id, populate_existing=True)
    if run is None:
        raise ApiError(ErrorCode.NOT_FOUND, "PipelineRun not found", details={"run_id": run_id})
    tasks = list(
        session.scalars(select(Task).where(Task.run_id == run_id).order_by(Task.created_at, Task.id))
    )
    return {
        "run_id": run.id,
        "episode_id": run.episode_id,
        "pipeline_version": run.pipeline_version,
        "state": run.state,
        "cancel_requested": run.cancel_requested,
        "error_code": run.error_code,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "tasks": [
            {
                "task_id": task.id,
                "step": task.step,
                "pool": task.pool,
                "state": task.state,
                "attempt": task.attempt,
                "max_attempts": task.max_attempts,
                "business_revision": int(task.business_revision or 1),
                "retryable": _task_retryable(task),
                "next_run_at": task.next_run_at.isoformat() if task.next_run_at else None,
                "worker_id": task.worker_id,
                "error_code": task.error_code,
                "lease_until": task.lease_until.isoformat() if task.lease_until else None,
                "output_asset_id": task.output_asset_id,
            }
            for task in tasks
        ],
    }


__all__ = [
    "request_run_cancel",
    "request_task_cancel",
    "retry_failed_task",
    "run_task_summary",
]
