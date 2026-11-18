"""Task protocol integration tests (SPEC 6.1-6.4: AT-09, dependency gating, cancel).

PostgreSQL is the only truth source: every test drives real claims/leases through
committed transactions (no in-memory shortcuts, no Redis required).
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select

from embodied_dataops.db.models import (
    Asset,
    Episode,
    ImportJob,
    PipelineRun,
    QualityReport,
    SourceImport,
    Task,
)
from embodied_dataops.domain.enums import RunState, Step, TaskState
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.claim import claim_next_task, claim_task_by_id
from embodied_dataops.workflow.commands import request_run_cancel, run_task_summary
from embodied_dataops.workflow.completion import complete_task, heartbeat
from embodied_dataops.workflow.dag import create_import_job_tasks, create_pipeline_run
from embodied_dataops.workflow.handlers import HANDLERS
from embodied_dataops.workflow.handlers.base import HandlerResult, register_json
from embodied_dataops.workflow.outbox import utcnow
from embodied_dataops.workflow.runner import execute_task_by_id, run_until_idle

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _source_import(session, *, index: int = 0) -> SourceImport:
    record = SourceImport(
        id=new_id(),
        source_type="LEROBOT_V3",
        upload_state="READY",
        data_origin="synthetic",
        license="MIT",
        filename=f"synthetic-{index}.zip",
        size_bytes=0,
    )
    session.add(record)
    session.flush()
    return record


def _episode(session, *, index: int = 0) -> Episode:
    source_import = _source_import(session, index=index)
    episode = Episode(
        id=new_id(),
        source_import_id=source_import.id,
        source_episode_index=index,
        episode_type="robot_episode",
        task_label="pick-and-place",
        duration_ns=10_000_000_000,
        status="IMPORTED",
    )
    session.add(episode)
    session.flush()
    return episode


def _run_with_tasks(session, *, index: int = 0) -> tuple[Episode, PipelineRun, dict[str, Task]]:
    episode = _episode(session, index=index)
    run = create_pipeline_run(session, episode)
    session.commit()
    tasks = {task.step: task for task in session.scalars(select(Task).where(Task.run_id == run.id))}
    return episode, run, tasks


def _task(session, task_id: str) -> Task:
    return session.get(Task, task_id)


def _complete(session_factory, task_id: str, token: str, result: HandlerResult | None = None) -> bool:
    with session_factory() as session:
        return complete_task(session, task_id, token, result or HandlerResult(data={}))


# ---------------------------------------------------------------------------
# AT-09: one lease per task, never two
# ---------------------------------------------------------------------------
@pytest.mark.concurrency
def test_claim_is_exclusive_between_two_workers(session_factory):
    with session_factory() as session:
        _, run, tasks = _run_with_tasks(session)
        root_id = tasks[Step.NORMALIZE_EPISODE.value].id

    barrier = threading.Barrier(2)

    def attempt(worker_id: str):
        with session_factory() as session:
            barrier.wait(timeout=10)
            return claim_task_by_id(session, root_id, worker_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt, "worker-a"), executor.submit(attempt, "worker-b")]
        results = [future.result(timeout=30) for future in futures]

    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1, "exactly one worker may hold the lease"
    winner = claimed[0]
    uuid.UUID(winner.lease_token)  # opaque UUIDv4 fencing token

    with session_factory() as session:
        stored = _task(session, root_id)
        assert stored.state == TaskState.RUNNING.value
        assert stored.attempt == 1, "a lost race must not increment attempt"
        assert stored.lease_token == winner.lease_token
        assert stored.worker_id == winner.worker_id
        remaining = (stored.lease_until - utcnow()).total_seconds()
        assert 50 <= remaining <= 65, "lease_seconds defaults to 60"
        run_row = session.get(PipelineRun, run.id)
        assert run_row.state == RunState.RUNNING.value
        assert run_row.started_at is not None


@pytest.mark.concurrency
def test_duplicate_wakeup_cannot_claim_twice(session_factory):
    with session_factory() as session:
        _, _, tasks = _run_with_tasks(session)
        root_id = tasks[Step.NORMALIZE_EPISODE.value].id

    with session_factory() as session:
        assert claim_task_by_id(session, root_id, "worker-a") is not None
    with session_factory() as session:
        assert claim_task_by_id(session, root_id, "worker-b") is None
    with session_factory() as session:
        stored = _task(session, root_id)
        assert stored.attempt == 1
        assert stored.state == TaskState.RUNNING.value


# ---------------------------------------------------------------------------
# dependency gating / DAG ordering
# ---------------------------------------------------------------------------
def test_dependencies_gate_claims_and_run_completes_in_dag_order(session_factory):
    with session_factory() as session:
        _, run, tasks = _run_with_tasks(session)

    with session_factory() as session:
        first = claim_next_task(session, "cpu", "worker-a")
    assert first.step == Step.NORMALIZE_EPISODE.value
    assert first.attempt == 1

    # Nothing else can run while NORMALIZE_EPISODE holds its lease.
    with session_factory() as session:
        assert claim_next_task(session, "cpu", "worker-b") is None
    with session_factory() as session:
        assert claim_next_task(session, "vision", "worker-v") is None

    assert _complete(session_factory, first.id, first.lease_token) is True

    with session_factory() as session:
        quality = claim_next_task(session, "cpu", "worker-a")
    assert quality.step == Step.QUALITY_CHECK.value
    with session_factory() as session:
        assert claim_next_task(session, "cpu", "worker-b") is None
        assert claim_next_task(session, "vision", "worker-v") is None
    assert _complete(
        session_factory, quality.id, quality.lease_token, HandlerResult(data={}, verdict="PASS")
    ) is True

    with session_factory() as session:
        preview = claim_next_task(session, "cpu", "worker-a")
    with session_factory() as session:
        detect = claim_next_task(session, "vision", "worker-v")
    assert preview.step == Step.BUILD_PREVIEW.value
    assert detect.step == Step.DETECT_OBJECTS.value
    # READY_FOR_REVIEW needs both branches.
    with session_factory() as session:
        assert claim_next_task(session, "cpu", "worker-b") is None
    assert _complete(session_factory, detect.id, detect.lease_token) is True
    assert _complete(session_factory, preview.id, preview.lease_token) is True

    with session_factory() as session:
        ready = claim_next_task(session, "cpu", "worker-a")
    assert ready.step == Step.READY_FOR_REVIEW.value
    assert _complete(session_factory, ready.id, ready.lease_token) is True

    with session_factory() as session:
        run_row = session.get(PipelineRun, run.id)
        assert run_row.state == RunState.SUCCEEDED.value
        assert run_row.finished_at is not None
        states = {task.step: task.state for task in session.scalars(select(Task).where(Task.run_id == run.id))}
        assert set(states.values()) == {TaskState.SUCCEEDED.value}


def test_claim_marks_parent_import_job_running(session_factory):
    with session_factory() as session:
        source_import = _source_import(session)
        job = create_import_job_tasks(session, source_import)
        session.commit()
        validate_id = session.scalars(
            select(Task.id).where(Task.import_job_id == job.id, Task.step == Step.VALIDATE_SOURCE.value)
        ).one()

    with session_factory() as session:
        claimed = claim_task_by_id(session, validate_id, "worker-a")
    assert claimed is not None
    with session_factory() as session:
        job_row = session.get(ImportJob, job.id)
        assert job_row.state == "RUNNING"
        assert job_row.started_at is not None


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------
def test_cancel_before_claim_is_immediate_and_idempotent(session_factory):
    with session_factory() as session:
        _, run, tasks = _run_with_tasks(session)

    with session_factory() as session:
        result = request_run_cancel(session, run.id)
    assert result["state"] == RunState.CANCELLED.value
    assert result["cancelled_tasks"] == len(tasks)
    assert result["running_tasks"] == 0

    with session_factory() as session:
        again = request_run_cancel(session, run.id)
    assert again.get("already_terminal") is True
    assert again["state"] == RunState.CANCELLED.value

    # No pool can start work for a cancelled run, even with a stale wakeup.
    for pool in ("cpu", "vision", "ros2"):
        with session_factory() as session:
            assert claim_next_task(session, pool, "worker-a") is None


def test_cancel_stops_running_task_cooperatively(session_factory):
    with session_factory() as session:
        _, run, tasks = _run_with_tasks(session)
        root_id = tasks[Step.NORMALIZE_EPISODE.value].id

    with session_factory() as session:
        running = claim_task_by_id(session, root_id, "worker-a")
    assert running is not None

    with session_factory() as session:
        result = request_run_cancel(session, run.id)
    assert result["running_tasks"] == 1
    assert result["cancelled_tasks"] == 4
    with session_factory() as session:
        run_row = session.get(PipelineRun, run.id)
        assert run_row.cancel_requested is True
        assert run_row.state == RunState.RUNNING.value, "run waits for cooperative stop of the running task"
        stored = _task(session, root_id)
        assert stored.cancel_requested is True
        assert stored.state == TaskState.RUNNING.value

    # The worker observes cancel_requested and performs the CAS to CANCELLED.
    with session_factory() as session:
        from embodied_dataops.workflow.completion import cancel_running_task

        assert cancel_running_task(session, root_id, running.lease_token) is True
    with session_factory() as session:
        run_row = session.get(PipelineRun, run.id)
        assert run_row.state == RunState.CANCELLED.value
        assert _task(session, root_id).state == TaskState.CANCELLED.value
        assert claim_next_task(session, "cpu", "worker-b") is None


# ---------------------------------------------------------------------------
# heartbeat
# ---------------------------------------------------------------------------
@pytest.mark.concurrency
def test_heartbeat_renews_only_the_current_lease(session_factory):
    with session_factory() as session:
        _, _, tasks = _run_with_tasks(session)
        root_id = tasks[Step.NORMALIZE_EPISODE.value].id

    with session_factory() as session:
        claimed = claim_task_by_id(session, root_id, "worker-a")
    original_lease = claimed.lease_until

    with session_factory() as session:
        assert heartbeat(session, root_id, claimed.lease_token) is True
    with session_factory() as session:
        renewed = _task(session, root_id).lease_until
    assert renewed >= original_lease
    assert (renewed - utcnow()).total_seconds() > 55

    with session_factory() as session:
        assert heartbeat(session, root_id, "00000000-0000-4000-8000-000000000000") is False


# ---------------------------------------------------------------------------
# QC verdict FAIL stops the downstream closure (SPEC 6.1)
# ---------------------------------------------------------------------------
def test_qc_fail_verdict_stops_downstream_and_fails_run(session_factory):
    with session_factory() as session:
        episode, run, tasks = _run_with_tasks(session)

    with session_factory() as session:
        normalize = claim_task_by_id(session, tasks[Step.NORMALIZE_EPISODE.value].id, "worker-a")
    assert _complete(session_factory, normalize.id, normalize.lease_token) is True

    with session_factory() as session:
        quality = claim_task_by_id(session, tasks[Step.QUALITY_CHECK.value].id, "worker-a")
    assert _complete(
        session_factory, quality.id, quality.lease_token, HandlerResult(data={"findings": ["SYNC"]}, verdict="FAIL")
    ) is True

    with session_factory() as session:
        quality_row = _task(session, quality.id)
        assert quality_row.state == TaskState.SUCCEEDED.value, "a FAIL verdict is a valid diagnostic result"
        assert quality_row.output_json["verdict"] == "FAIL"
        run_row = session.get(PipelineRun, run.id)
        assert run_row.state == RunState.FAILED.value
        assert run_row.error_code == "QUALITY_FAILED"
        episode_row = session.get(Episode, episode.id)
        assert episode_row.status == "QUALITY_FAILED"
        downstream = {
            task.step: task.state
            for task in session.scalars(select(Task).where(Task.run_id == run.id))
            if task.step in {
                Step.DETECT_OBJECTS.value,
                Step.BUILD_PREVIEW.value,
                Step.READY_FOR_REVIEW.value,
            }
        }
        assert set(downstream.values()) == {TaskState.CANCELLED.value}
    for pool in ("cpu", "vision"):
        with session_factory() as session:
            assert claim_next_task(session, pool, "worker-b") is None


# ---------------------------------------------------------------------------
# READY_FOR_REVIEW handler (real handler, not a fake)
# ---------------------------------------------------------------------------
def _prepare_ready(session_factory, *, verdict: str):
    with session_factory() as session:
        episode = _episode(session)
        run = create_pipeline_run(session, episode)
        tasks = {task.step: task for task in session.scalars(select(Task).where(Task.run_id == run.id))}
        for step in (
            Step.NORMALIZE_EPISODE,
            Step.QUALITY_CHECK,
            Step.DETECT_OBJECTS,
            Step.BUILD_PREVIEW,
        ):
            tasks[step.value].state = TaskState.SUCCEEDED.value
        session.add(
            QualityReport(
                id=new_id(),
                episode_id=episode.id,
                run_id=run.id,
                policy_version="qc-policy/1.0",
                status=verdict,
                summary_json={"verdict": verdict},
            )
        )
        session.commit()
        return episode.id, run.id, tasks[Step.READY_FOR_REVIEW.value].id


def test_ready_for_review_marks_episode_for_pass_verdict(session_factory, storage):
    episode_id, run_id, ready_id = _prepare_ready(session_factory, verdict="PASS")
    outcome = execute_task_by_id(
        ready_id, "worker-a", session_factory=session_factory, storage=storage
    )
    assert outcome == "succeeded"
    with session_factory() as session:
        assert session.get(Episode, episode_id).status == "READY_FOR_REVIEW"
        assert session.get(PipelineRun, run_id).state == RunState.SUCCEEDED.value
        assert _task(session, ready_id).state == TaskState.SUCCEEDED.value


def test_ready_for_review_rejects_fail_verdict(session_factory, storage):
    episode_id, run_id, ready_id = _prepare_ready(session_factory, verdict="FAIL")
    outcome = execute_task_by_id(
        ready_id, "worker-a", session_factory=session_factory, storage=storage
    )
    assert outcome == "failed"
    with session_factory() as session:
        assert _task(session, ready_id).state == TaskState.FAILED.value
        assert _task(session, ready_id).error_code == "QC_STATUS_NOT_ELIGIBLE"
        assert session.get(PipelineRun, run_id).state == RunState.FAILED.value
        assert session.get(Episode, episode_id).status == "QUALITY_FAILED"


# ---------------------------------------------------------------------------
# synchronous driver (run_until_idle) without Redis
# ---------------------------------------------------------------------------
def test_run_until_idle_executes_whole_dag_without_redis(session_factory, storage):
    with session_factory() as session:
        _, run, _ = _run_with_tasks(session)

    steps = (
        Step.NORMALIZE_EPISODE,
        Step.QUALITY_CHECK,
        Step.DETECT_OBJECTS,
        Step.BUILD_PREVIEW,
        Step.READY_FOR_REVIEW,
    )
    previous = {}
    try:
        for step in steps:
            def _handler(_ctx, _step=step):
                return HandlerResult(
                    data={"step": _step.value},
                    verdict="PASS" if _step is Step.QUALITY_CHECK else None,
                )

            previous[step] = HANDLERS.install(step, _handler)
        counts = run_until_idle(
            session_factory,
            pools=("cpu", "vision"),
            storage=storage,
            max_iterations=20,
        )
    finally:
        for step, old in previous.items():
            HANDLERS.restore(step, old)

    assert counts["failed"] == 0
    assert counts["retried"] == 0
    assert counts["succeeded"] == len(steps)
    with session_factory() as session:
        assert session.get(PipelineRun, run.id).state == RunState.SUCCEEDED.value
        summary = run_task_summary(session, run.id)
        assert summary["state"] == RunState.SUCCEEDED.value
        assert all(task["state"] == TaskState.SUCCEEDED.value for task in summary["tasks"])


# ---------------------------------------------------------------------------
# AT-09: duplicate wakeups produce exactly one effective output asset
# ---------------------------------------------------------------------------
def _setup_import_task(session) -> tuple[ImportJob, Task]:
    source_import = _source_import(session)
    job = create_import_job_tasks(session, source_import)
    session.commit()
    validate = session.scalars(
        select(Task).where(
            Task.import_job_id == job.id, Task.step == Step.VALIDATE_SOURCE.value
        )
    ).one()
    return job, validate


@pytest.mark.concurrency
def test_duplicate_notifications_produce_one_output_asset(session_factory, storage):
    with session_factory() as session:
        _, validate = _setup_import_task(session)
        task_id = validate.id

    def handler(ctx):
        asset = register_json(
            ctx,
            key=f"derived/{task_id}/validate.json",
            payload={"ok": True},
            origin="derived",
            logical_name="validate.json",
        )
        return HandlerResult(data={"ok": True}, output_asset_id=asset.id)

    previous = HANDLERS.install(Step.VALIDATE_SOURCE, handler)
    try:
        first = execute_task_by_id(
            task_id, "worker-a", session_factory=session_factory, storage=storage
        )
        # A duplicate Redis hint for an already-succeeded task is a no-op.
        second = execute_task_by_id(
            task_id, "worker-b", session_factory=session_factory, storage=storage
        )
    finally:
        HANDLERS.restore(Step.VALIDATE_SOURCE, previous)

    assert first == "succeeded"
    assert second == "skipped"
    with session_factory() as session:
        task = session.get(Task, task_id)
        assets = list(
            session.scalars(
                select(Asset).where(Asset.object_key == f"derived/{task_id}/validate.json")
            )
        )
        assert len(assets) == 1, "exactly one immutable output asset is referenced"
        assert task.output_asset_id == assets[0].id
        assert task.attempt == 1, "duplicate hints never increment the attempt counter"
        assert task.state == TaskState.SUCCEEDED.value
        assert assets[0].origin == "derived"

