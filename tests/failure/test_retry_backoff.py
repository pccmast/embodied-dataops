"""AT-14: exactly three claims then FAILED, with 5s/20s deterministic backoff.

No test sleeps: the expected delay is computed with the same deterministic jitter
function and compared against ``next_run_at``; time is advanced by moving
``next_run_at`` into the past.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select, update

from embodied_dataops.db.models import ImportJob, SourceImport, Task
from embodied_dataops.domain.enums import Step, TaskState
from embodied_dataops.errors import ApiError, ErrorCode, RetryableError, TerminalTaskError
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.completion import backoff_delay_seconds
from embodied_dataops.workflow.dag import create_import_job_tasks
from embodied_dataops.workflow.handlers import HANDLERS
from embodied_dataops.workflow.outbox import utcnow
from embodied_dataops.workflow.runner import execute_task_by_id

pytestmark = [pytest.mark.failure, pytest.mark.integration]


def _setup_import_task(session) -> tuple[ImportJob, Task]:
    source_import = SourceImport(
        id=new_id(),
        source_type="LEROBOT_V3",
        upload_state="READY",
        data_origin="synthetic",
        license="MIT",
        filename="retry.zip",
        size_bytes=0,
    )
    session.add(source_import)
    session.flush()
    job = create_import_job_tasks(session, source_import)
    session.commit()
    validate = session.scalars(
        select(Task).where(
            Task.import_job_id == job.id, Task.step == Step.VALIDATE_SOURCE.value
        )
    ).one()
    return job, validate


def _advance_clock(session_factory, task_id: str, seconds: float = 1.0) -> None:
    """Simulate the passage of time without sleeping."""
    with session_factory() as session:
        session.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(next_run_at=utcnow() - timedelta(seconds=seconds))
        )
        session.commit()


def _install(step, handler) -> object:
    return HANDLERS.install(step, handler)


def test_three_claims_then_failed_with_5s_and_20s_backoff(session_factory, storage):
    with session_factory() as session:
        job, validate = _setup_import_task(session)
        task_id = validate.id

    invocations = {"count": 0}

    def always_retryable(_ctx):
        invocations["count"] += 1
        raise RetryableError(
            ErrorCode.OBJECT_STORE_UNAVAILABLE, "object store is temporarily unavailable"
        )

    previous = _install(Step.VALIDATE_SOURCE, always_retryable)
    try:
        for attempt, base in ((1, 5.0), (2, 20.0)):
            before = utcnow()
            outcome = execute_task_by_id(
                task_id, f"worker-{attempt}", session_factory=session_factory, storage=storage
            )
            assert outcome == "retried"
            with session_factory() as session:
                task = session.get(Task, task_id)
                assert task.state == TaskState.RETRY_WAIT.value
                assert task.attempt == attempt
                assert task.lease_token is None and task.lease_until is None
                assert task.worker_id == f"worker-{attempt}"
                assert task.error_code == ErrorCode.OBJECT_STORE_UNAVAILABLE.value
                expected_delay = backoff_delay_seconds(task_id, attempt)
                assert base * 0.9 <= expected_delay <= base * 1.1
                observed_delay = (task.next_run_at - before).total_seconds()
                assert abs(observed_delay - expected_delay) < 2.0, (
                    f"attempt {attempt}: expected {expected_delay:.3f}s, got {observed_delay:.3f}s"
                )
            _advance_clock(session_factory, task_id)

        # Third claim: retry budget is exhausted, the task becomes FAILED.
        outcome = execute_task_by_id(
            task_id, "worker-3", session_factory=session_factory, storage=storage
        )
        assert outcome == "failed"
        with session_factory() as session:
            task = session.get(Task, task_id)
            assert task.state == TaskState.FAILED.value
            assert task.attempt == 3
            assert task.error_code == ErrorCode.OBJECT_STORE_UNAVAILABLE.value
            assert task.lease_token is None
            run_job = session.get(ImportJob, job.id)
            assert run_job.state == "FAILED"
            assert run_job.error_code == ErrorCode.OBJECT_STORE_UNAVAILABLE.value
            sibling = session.scalars(
                select(Task).where(Task.import_job_id == job.id, Task.id != task_id)
            ).one()
            assert sibling.state == TaskState.CANCELLED.value
    finally:
        HANDLERS.restore(Step.VALIDATE_SOURCE, previous)

    assert invocations["count"] == 3, "the handler runs exactly as often as a lease is granted"
    # A fourth duplicate hint cannot claim anything.
    assert (
        execute_task_by_id(task_id, "worker-4", session_factory=session_factory, storage=storage)
        == "skipped"
    )
    with session_factory() as session:
        assert session.get(Task, task_id).attempt == 3


@pytest.mark.parametrize(
    ("error_factory", "expected_code"),
    [
        (
            lambda: TerminalTaskError(ErrorCode.FILE_DECODE_ERROR, "source video is corrupt"),
            "FILE_DECODE_ERROR",
        ),
        (lambda: ApiError(ErrorCode.INVALID_INPUT, "invalid schema"), "INVALID_INPUT"),
    ],
)
def test_terminal_errors_fail_on_first_attempt(session_factory, storage, error_factory, expected_code):
    with session_factory() as session:
        job, validate = _setup_import_task(session)
        task_id = validate.id

    invocations = {"count": 0}

    def terminal(_ctx):
        invocations["count"] += 1
        raise error_factory()

    previous = _install(Step.VALIDATE_SOURCE, terminal)
    try:
        outcome = execute_task_by_id(
            task_id, "worker-a", session_factory=session_factory, storage=storage
        )
    finally:
        HANDLERS.restore(Step.VALIDATE_SOURCE, previous)

    assert outcome == "failed"
    assert invocations["count"] == 1
    with session_factory() as session:
        task = session.get(Task, task_id)
        assert task.state == TaskState.FAILED.value
        assert task.attempt == 1
        assert task.error_code == expected_code
        assert session.get(ImportJob, job.id).state == "FAILED"


def test_ambiguous_handler_crash_is_retried_but_bounded(session_factory, storage):
    """An unexpected handler exception is retried like a worker crash, never forever."""
    with session_factory() as session:
        _, validate = _setup_import_task(session)
        task_id = validate.id

    def boom(_ctx):
        raise RuntimeError("unexpected worker bug")

    previous = _install(Step.VALIDATE_SOURCE, boom)
    try:
        outcomes = []
        for attempt in range(3):
            outcomes.append(
                execute_task_by_id(
                    task_id, f"worker-{attempt}", session_factory=session_factory, storage=storage
                )
            )
            if attempt < 2:
                _advance_clock(session_factory, task_id)
    finally:
        HANDLERS.restore(Step.VALIDATE_SOURCE, previous)

    assert outcomes == ["retried", "retried", "failed"]
    with session_factory() as session:
        task = session.get(Task, task_id)
        assert task.state == TaskState.FAILED.value
        assert task.attempt == 3
