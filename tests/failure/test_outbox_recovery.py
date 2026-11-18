"""AT-12 / AT-13: transactional outbox survives Redis failures.

* AT-12: PostgreSQL commits but ``XADD`` fails -> the row stays unpublished, the
  dispatcher retries, the task still runs and is not lost.
* AT-13: ``XADD`` succeeds but marking ``published_at`` fails -> the wakeup is
  duplicated on the next cycle, yet the database lease still allows exactly one
  execution.

All assertions use a recording fake Redis; no real Redis is required.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
import redis
from prometheus_client import REGISTRY
from sqlalchemy import select, update

from embodied_dataops.db.models import ImportJob, OutboxEvent, SourceImport, Task
from embodied_dataops.domain.enums import Step, TaskState
from embodied_dataops.ids import new_id
from embodied_dataops.workflow import dispatcher as dispatcher_module
from embodied_dataops.workflow.claim import claim_task_by_id
from embodied_dataops.workflow.completion import complete_task
from embodied_dataops.workflow.dag import create_import_job_tasks
from embodied_dataops.workflow.dispatcher import dispatch_once
from embodied_dataops.workflow.handlers import HANDLERS
from embodied_dataops.workflow.handlers.base import HandlerResult
from embodied_dataops.workflow.outbox import utcnow
from embodied_dataops.workflow.reaper import reap_once
from embodied_dataops.workflow.runner import execute_task_by_id

pytestmark = [pytest.mark.failure, pytest.mark.integration]


class FakeRedis:
    """Records XADD calls; can simulate a total Redis outage."""

    def __init__(self) -> None:
        self.fail = False
        self.entries: list[tuple[str, dict]] = []

    def xadd(self, stream, fields, **kwargs):  # noqa: ANN001 - redis-client shape
        if self.fail:
            raise redis.RedisError("simulated Redis outage")
        self.entries.append((stream, dict(fields)))
        return f"{len(self.entries)}-0"


class BrokenRedis:
    """Redis that is reachable for nothing (used to prove DB-only recovery)."""

    def xadd(self, *_args, **_kwargs):
        raise redis.RedisError("redis down")

    def xautoclaim(self, *_args, **_kwargs):
        raise redis.RedisError("redis down")

    def xack(self, *_args, **_kwargs):
        raise redis.RedisError("redis down")


def _metric(name: str, labels: dict | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels or {})
    return float(value or 0.0)


def _setup_import_task(session) -> tuple[ImportJob, Task]:
    source_import = SourceImport(
        id=new_id(),
        source_type="LEROBOT_V3",
        upload_state="READY",
        data_origin="synthetic",
        license="MIT",
        filename="outbox.zip",
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


def _install_fake_handlers(steps) -> dict:
    previous = {}

    def _handler(_ctx):
        return HandlerResult(data={"fake": True})

    for step in steps:
        previous[step] = HANDLERS.install(step, _handler)
    return previous


def _restore_handlers(previous: dict) -> None:
    for step, old in previous.items():
        HANDLERS.restore(step, old)


def _unpublished(session) -> list[OutboxEvent]:
    return list(session.scalars(select(OutboxEvent).where(OutboxEvent.published_at.is_(None))))


# ---------------------------------------------------------------------------
# AT-12
# ---------------------------------------------------------------------------
def test_outbox_republish_after_redis_xadd_failure_keeps_task(session_factory, storage):
    with session_factory() as session:
        job, validate = _setup_import_task(session)
        task_id = validate.id

    fake = FakeRedis()
    fake.fail = True
    failures_before = _metric("dataops_outbox_publish_failures_total")
    with session_factory() as session:
        stats = dispatch_once(session, fake)
    assert stats["failed"] == 1
    assert stats["published"] == 0
    assert _metric("dataops_outbox_publish_failures_total") - failures_before == 1
    with session_factory() as session:
        pending = _unpublished(session)
        assert len(pending) == 1, "an XADD failure must keep the outbox row unpublished"
        assert pending[0].payload_json["task_id"] == task_id
        stored = session.get(Task, task_id)
        assert stored.state == TaskState.PENDING.value, "Redis failure must not touch task state"

    # Redis recovers: the next cycle publishes the stored event.
    fake.fail = False
    with session_factory() as session:
        stats = dispatch_once(session, fake)
    assert stats["published"] == 1
    assert len(fake.entries) == 1
    stream, fields = fake.entries[0]
    assert stream == "stream:tasks:cpu"
    assert json.loads(fields["payload"])["task_id"] == task_id
    with session_factory() as session:
        assert _unpublished(session) == []

    # The task still continues to completion (not lost).
    previous = _install_fake_handlers((Step.VALIDATE_SOURCE, Step.IMPORT_EPISODES))
    try:
        outcome = execute_task_by_id(
            task_id, "worker-a", session_factory=session_factory, storage=storage
        )
        assert outcome == "succeeded"
        with session_factory() as session:
            # Finishing VALIDATE_SOURCE creates the downstream IMPORT_EPISODES wakeup.
            pending = _unpublished(session)
            assert len(pending) == 1
            assert pending[0].payload_json["task_id"] != task_id
            downstream_id = pending[0].payload_json["task_id"]
            stats = dispatch_once(session, fake)
        assert stats["published"] == 1
        assert execute_task_by_id(
            downstream_id, "worker-a", session_factory=session_factory, storage=storage
        ) == "succeeded"
    finally:
        _restore_handlers(previous)

    with session_factory() as session:
        job_row = session.get(ImportJob, job.id)
        assert job_row.state == "SUCCEEDED"
        states = {task.step: task.state for task in session.scalars(
            select(Task).where(Task.import_job_id == job.id)
        )}
        assert set(states.values()) == {TaskState.SUCCEEDED.value}


# ---------------------------------------------------------------------------
# AT-13
# ---------------------------------------------------------------------------
def test_published_mark_failure_causes_only_duplicate_hints(session_factory, monkeypatch):
    with session_factory() as session:
        job, validate = _setup_import_task(session)
        task_id = validate.id

    fake = FakeRedis()
    real_mark = dispatcher_module.mark_published
    state = {"raised": False}

    def flaky_mark(event):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("simulated failure writing published_at")
        return real_mark(event)

    monkeypatch.setattr(dispatcher_module, "mark_published", flaky_mark)

    with session_factory() as session:
        first = dispatch_once(session, fake)
    assert first["published"] == 0
    assert first["failed"] == 1
    assert len(fake.entries) == 1, "XADD happened before the mark failure"
    with session_factory() as session:
        assert len(_unpublished(session)) == 1

    # Retry publishes a duplicate hint (safe by design).
    with session_factory() as session:
        second = dispatch_once(session, fake)
    assert second["published"] == 1
    assert len(fake.entries) == 2
    assert {json.loads(fields["payload"])["task_id"] for _, fields in fake.entries} == {task_id}

    # The duplicate hint produces exactly one claim; a second claim is refused.
    with session_factory() as session:
        claimed = claim_task_by_id(session, task_id, "worker-a")
    assert claimed is not None
    assert claimed.attempt == 1
    with session_factory() as session:
        assert claim_task_by_id(session, task_id, "worker-b") is None
    with session_factory() as session:
        assert complete_task(session, task_id, claimed.lease_token, HandlerResult(data={"ok": True}))

    with session_factory() as session:
        records = list(
            session.scalars(
                select(Task).where(
                    Task.import_job_id == job.id, Task.step == Step.VALIDATE_SOURCE.value
                )
            )
        )
        assert len(records) == 1, "a task row is never duplicated by a redelivered hint"
        assert records[0].state == TaskState.SUCCEEDED.value
        assert records[0].attempt == 1
        assert records[0].output_json == {"ok": True}


# ---------------------------------------------------------------------------
# reaper: lost PENDING wakeup is re-notified from PostgreSQL
# ---------------------------------------------------------------------------
def test_reaper_republishes_lost_pending_wakeup(session_factory):
    with session_factory() as session:
        _, validate = _setup_import_task(session)
        task_id = validate.id

    # Simulate "XADD succeeded but the consumer never saw the message": the outbox
    # row is marked published while the task is still PENDING.
    with session_factory() as session:
        for event in session.scalars(select(OutboxEvent)):
            event.published_at = utcnow()
        session.commit()

    with session_factory() as session:
        stats = reap_once(session, now=utcnow())
    assert stats["republished"] == 1
    with session_factory() as session:
        pending = _unpublished(session)
        assert len(pending) == 1
        assert pending[0].payload_json["task_id"] == task_id
        assert session.get(Task, task_id).state == TaskState.PENDING.value

    # While the unpublished wakeup exists, the reaper does not spam duplicates.
    with session_factory() as session:
        assert reap_once(session, now=utcnow())["republished"] == 0


# ---------------------------------------------------------------------------
# reaper: PostgreSQL recovery does not depend on Redis being reachable
# ---------------------------------------------------------------------------
def test_reaper_recovers_lease_with_broken_redis(session_factory):
    with session_factory() as session:
        _, validate = _setup_import_task(session)
        task_id = validate.id
    with session_factory() as session:
        claimed = claim_task_by_id(session, task_id, "worker-crashed")
    assert claimed is not None
    with session_factory() as session:
        session.execute(
            update(Task).where(Task.id == task_id).values(lease_until=utcnow() - timedelta(seconds=1))
        )
        session.commit()

    with session_factory() as session:
        stats = reap_once(session, now=utcnow(), redis_client=BrokenRedis())
    assert stats["expired"] == 1
    assert stats["reclaimed"] == 0  # Redis cleanup failed, DB recovery still happened
    with session_factory() as session:
        assert session.get(Task, task_id).state == TaskState.RETRY_WAIT.value
