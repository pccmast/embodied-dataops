"""AT-10: ``kill -9`` on a RUNNING worker is recovered by the lease reaper.

The crash is simulated at the only place it matters - the database row is left
``RUNNING`` with an expired lease and no heartbeat.  The reaper must reclaim it,
the task must be retried and the system must never keep a permanent RUNNING task.
"""

from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import select, update

from embodied_dataops.db.models import Episode, OutboxEvent, PipelineRun, SourceImport, Task
from embodied_dataops.domain.enums import RunState, Step, TaskState
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.claim import claim_task_by_id
from embodied_dataops.workflow.dag import create_pipeline_run
from embodied_dataops.workflow.dispatcher import run_forever
from embodied_dataops.workflow.handlers import HANDLERS
from embodied_dataops.workflow.handlers.base import HandlerResult
from embodied_dataops.workflow.outbox import utcnow
from embodied_dataops.workflow.reaper import reap_once
from embodied_dataops.workflow.runner import execute_task_by_id

pytestmark = [pytest.mark.failure, pytest.mark.integration]


def _setup_pipeline(session) -> tuple[Episode, PipelineRun, dict[str, Task]]:
    source_import = SourceImport(
        id=new_id(),
        source_type="LEROBOT_V3",
        upload_state="READY",
        data_origin="synthetic",
        license="MIT",
        filename="crash.zip",
        size_bytes=0,
    )
    session.add(source_import)
    session.flush()
    episode = Episode(
        id=new_id(),
        source_import_id=source_import.id,
        source_episode_index=0,
        episode_type="robot_episode",
        task_label="crash-recovery",
        duration_ns=10_000_000_000,
        status="IMPORTED",
    )
    session.add(episode)
    session.flush()
    run = create_pipeline_run(session, episode)
    session.commit()
    tasks = {task.step: task for task in session.scalars(select(Task).where(Task.run_id == run.id))}
    return episode, run, tasks


def _expire_lease(session_factory, task_id: str) -> None:
    with session_factory() as session:
        session.execute(
            update(Task).where(Task.id == task_id).values(lease_until=utcnow() - timedelta(seconds=1))
        )
        session.commit()


def _metric(name: str) -> float:
    return float(REGISTRY.get_sample_value(name) or 0.0)


def _running_tasks(session) -> list[Task]:
    return list(session.scalars(select(Task).where(Task.state == TaskState.RUNNING.value)))


def test_expired_lease_is_recovered_and_retried(session_factory, storage):
    with session_factory() as session:
        _, run, tasks = _setup_pipeline(session)
        root_id = tasks[Step.NORMALIZE_EPISODE.value].id

    # Worker A claims the task and is then killed -9 (no heartbeat, no CAS).
    with session_factory() as session:
        crashed = claim_task_by_id(session, root_id, "worker-crashed")
    assert crashed is not None
    _expire_lease(session_factory, root_id)
    with session_factory() as session:
        running = _running_tasks(session)
        assert len(running) == 1, "crash leaves one RUNNING row with an expired lease"

    expirations_before = _metric("dataops_lease_expirations_total")
    with session_factory() as session:
        stats = reap_once(session, now=utcnow())
    assert stats["expired"] == 1
    assert _metric("dataops_lease_expirations_total") - expirations_before == 1

    with session_factory() as session:
        task = session.get(Task, root_id)
        assert task.state == TaskState.RETRY_WAIT.value, "expired lease -> retry, not permanent RUNNING"
        assert task.attempt == 1
        assert task.lease_token is None
        assert task.lease_until is None
        assert task.error_code == "LEASE_EXPIRED"
        assert _running_tasks(session) == []
        assert session.get(PipelineRun, run.id).state == RunState.RUNNING.value

    # The retry delay is due -> reaper promotes it and re-notifies via the outbox.
    with session_factory() as session:
        session.execute(
            update(Task)
            .where(Task.id == root_id)
            .values(next_run_at=utcnow() - timedelta(seconds=1))
        )
        session.commit()
    with session_factory() as session:
        stats = reap_once(session, now=utcnow())
    assert stats["promoted"] == 1
    with session_factory() as session:
        assert session.get(Task, root_id).state == TaskState.PENDING.value
        wakeups = list(
            session.scalars(
                select(OutboxEvent).where(
                    OutboxEvent.published_at.is_(None),
                    OutboxEvent.payload_json["task_id"].astext == root_id,
                )
            )
        )
        assert len(wakeups) >= 1, "promoted retry must be re-notified from PostgreSQL"
        assert all(event.topic == "stream:tasks:cpu" for event in wakeups)

    # A healthy worker picks it up again; the second claim increments attempt to 2.
    def normalize(_ctx):
        return HandlerResult(data={"normalized": True})

    previous = HANDLERS.install(Step.NORMALIZE_EPISODE, normalize)
    try:
        outcome = execute_task_by_id(
            root_id, "worker-recovered", session_factory=session_factory, storage=storage
        )
    finally:
        HANDLERS.restore(Step.NORMALIZE_EPISODE, previous)
    assert outcome == "succeeded"

    with session_factory() as session:
        task = session.get(Task, root_id)
        assert task.state == TaskState.SUCCEEDED.value
        assert task.attempt == 2
        assert task.worker_id == "worker-recovered"
        assert _running_tasks(session) == [], "no permanent RUNNING task after recovery"


def test_lease_expiry_on_last_attempt_fails_task_and_run(session_factory):
    with session_factory() as session:
        episode, run, tasks = _setup_pipeline(session)
        root_id = tasks[Step.NORMALIZE_EPISODE.value].id

    with session_factory() as session:
        crashed = claim_task_by_id(session, root_id, "worker-crashed")
    assert crashed is not None
    # Simulate the last allowed attempt dying (attempts 1 and 2 consumed earlier).
    with session_factory() as session:
        session.execute(
            update(Task)
            .where(Task.id == root_id)
            .values(attempt=3, lease_until=utcnow() - timedelta(seconds=1))
        )
        session.commit()

    with session_factory() as session:
        stats = reap_once(session, now=utcnow())
    assert stats["expired"] == 1
    with session_factory() as session:
        task = session.get(Task, root_id)
        assert task.state == TaskState.FAILED.value
        assert task.error_code == "LEASE_EXPIRED"
        assert _running_tasks(session) == []
        run_row = session.get(PipelineRun, run.id)
        assert run_row.state == RunState.FAILED.value
        assert run_row.finished_at is not None
        assert session.get(Episode, episode.id).status == "QUALITY_FAILED"
        downstream = {
            other.step: other.state
            for other in session.scalars(select(Task).where(Task.run_id == run.id))
            if other.id != root_id
        }
        assert set(downstream.values()) == {TaskState.CANCELLED.value}


def test_reaper_is_idempotent_when_no_lease_expired(session_factory):
    with session_factory() as session:
        _, run, tasks = _setup_pipeline(session)
        root_id = tasks[Step.NORMALIZE_EPISODE.value].id
    with session_factory() as session:
        assert claim_task_by_id(session, root_id, "worker-healthy") is not None
    with session_factory() as session:
        stats = reap_once(session, now=utcnow())
    assert stats["expired"] == 0
    with session_factory() as session:
        task = session.get(Task, root_id)
        assert task.state == TaskState.RUNNING.value, "a live lease must never be reclaimed"
        assert task.attempt == 1


def test_dispatcher_process_also_recovers_expired_leases(session_factory, redis_client):
    """SPEC 3.1: the dispatcher owns lost-worker recovery; no separate cron needed."""
    with session_factory() as session:
        _, run, tasks = _setup_pipeline(session)
        root_id = tasks[Step.NORMALIZE_EPISODE.value].id
    with session_factory() as session:
        assert claim_task_by_id(session, root_id, "worker-crashed") is not None
    _expire_lease(session_factory, root_id)

    stop = threading.Event()
    thread = threading.Thread(
        target=run_forever,
        kwargs={"stop_event": stop, "redis_client": redis_client},
        name="dispatcher-under-test",
        daemon=True,
    )
    thread.start()
    try:
        deadline = time.time() + 15
        recovered = False
        while time.time() < deadline:
            with session_factory() as session:
                if session.get(Task, root_id).state == TaskState.RETRY_WAIT.value:
                    recovered = True
                    break
            time.sleep(0.2)
    finally:
        stop.set()
        thread.join(timeout=10)
    assert recovered, "the dispatcher loop must run lease recovery"
    with session_factory() as session:
        assert session.get(Task, root_id).state == TaskState.RETRY_WAIT.value

