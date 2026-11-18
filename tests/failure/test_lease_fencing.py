"""AT-11: fencing tokens reject a delayed old-lease worker.

A worker whose lease expired and whose task was reclaimed by the reaper must not
be able to commit anything.  The CAS ``WHERE state='RUNNING' AND
lease_token=:token`` is the only success path, so the stale worker's whole
transaction is rolled back and the new worker's output is preserved.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select, update

from embodied_dataops.db.models import Asset, ImportJob, SourceImport, Task
from embodied_dataops.domain.enums import Step, TaskState
from embodied_dataops.errors import ErrorCode, RetryableError
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.claim import claim_task_by_id
from embodied_dataops.workflow.completion import complete_task, fail_task, heartbeat
from embodied_dataops.workflow.dag import create_import_job_tasks
from embodied_dataops.workflow.handlers.base import HandlerResult
from embodied_dataops.workflow.outbox import utcnow
from embodied_dataops.workflow.reaper import reap_once

pytestmark = [pytest.mark.failure, pytest.mark.integration]


def _setup_import_task(session) -> tuple[ImportJob, Task]:
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
    job = create_import_job_tasks(session, source_import)
    session.commit()
    validate = session.scalars(
        select(Task).where(
            Task.import_job_id == job.id, Task.step == Step.VALIDATE_SOURCE.value
        )
    ).one()
    return job, validate


def _force_expired_lease(session_factory, task_id: str) -> None:
    with session_factory() as session:
        session.execute(
            update(Task)
            .where(Task.id == task_id)
            .values(lease_until=utcnow() - timedelta(seconds=1), next_run_at=utcnow() - timedelta(seconds=1))
        )
        session.commit()


def _make_due(session_factory, task_id: str) -> None:
    with session_factory() as session:
        session.execute(
            update(Task).where(Task.id == task_id).values(next_run_at=utcnow() - timedelta(seconds=1))
        )
        session.commit()


@pytest.mark.concurrency
def test_old_lease_token_cannot_overwrite_the_new_lease(session_factory):
    with session_factory() as session:
        _, validate = _setup_import_task(session)
        task_id = validate.id

    # --- first worker takes the lease, then "dies" (no more heartbeats) --------
    with session_factory() as session:
        stale = claim_task_by_id(session, task_id, "worker-old")
    assert stale is not None
    stale_token = stale.lease_token
    assert stale.attempt == 1

    _force_expired_lease(session_factory, task_id)
    with session_factory() as session:
        stats = reap_once(session, now=utcnow())
    assert stats["expired"] == 1
    with session_factory() as session:
        task = session.get(Task, task_id)
        assert task.state == TaskState.RETRY_WAIT.value
        assert task.attempt == 1, "lease expiry itself does not consume an attempt"

    # --- retry becomes due; the new worker takes a fresh lease ----------------
    _make_due(session_factory, task_id)
    with session_factory() as session:
        stats = reap_once(session, now=utcnow())
    assert stats["promoted"] >= 1
    with session_factory() as session:
        fresh = claim_task_by_id(session, task_id, "worker-new")
    assert fresh is not None
    fresh_token = fresh.lease_token
    assert fresh_token != stale_token
    assert fresh.attempt == 2

    # --- the stale worker wakes up and tries everything ----------------------
    with session_factory() as session:
        assert heartbeat(session, task_id, stale_token) is False, "stale token cannot renew"
    with session_factory() as session:
        assert (
            complete_task(session, task_id, stale_token, HandlerResult(data={"writer": "stale"}))
            is False
        ), "stale completion must be rejected"
    with session_factory() as session:
        assert (
            fail_task(
                session,
                task_id,
                stale_token,
                RetryableError(ErrorCode.OBJECT_STORE_UNAVAILABLE, "old worker giving up"),
                retryable=True,
            )
            is None
        ), "stale worker cannot even change the failure state"

    with session_factory() as session:
        task = session.get(Task, task_id)
        assert task.state == TaskState.RUNNING.value
        assert task.lease_token == fresh_token
        assert task.worker_id == "worker-new"
        assert task.output_json == {}, "rejected completion must not leave output behind"
        assert task.attempt == 2

    # --- the new worker commits normally ------------------------------------
    with session_factory() as session:
        assert (
            complete_task(session, task_id, fresh_token, HandlerResult(data={"writer": "fresh"}))
            is True
        )
    with session_factory() as session:
        task = session.get(Task, task_id)
        assert task.state == TaskState.SUCCEEDED.value
        assert task.output_json == {"writer": "fresh"}
        assert task.attempt == 2
        assert task.lease_token is None

    # --- a very late stale completion still cannot overwrite -----------------
    with session_factory() as session:
        assert (
            complete_task(session, task_id, stale_token, HandlerResult(data={"writer": "stale-again"}))
            is False
        )
    with session_factory() as session:
        task = session.get(Task, task_id)
        assert task.output_json == {"writer": "fresh"}
        assert task.state == TaskState.SUCCEEDED.value


@pytest.mark.concurrency
def test_expired_lease_heartbeat_cannot_revive_the_lease(session_factory):
    with session_factory() as session:
        _, validate = _setup_import_task(session)
        task_id = validate.id
    with session_factory() as session:
        claimed = claim_task_by_id(session, task_id, "worker-old")
    # The worker stalls past its lease but the reaper has not run yet.
    with session_factory() as session:
        session.execute(
            update(Task).where(Task.id == task_id).values(lease_until=utcnow() - timedelta(seconds=5))
        )
        session.commit()
    with session_factory() as session:
        assert heartbeat(session, task_id, claimed.lease_token) is False
    with session_factory() as session:
        still_running = session.get(Task, task_id)
        assert still_running.state == TaskState.RUNNING.value
    # The reaper, not the worker, is the recovery authority.
    with session_factory() as session:
        reap_once(session, now=utcnow())
    with session_factory() as session:
        assert session.get(Task, task_id).state == TaskState.RETRY_WAIT.value


@pytest.mark.concurrency
def test_stale_completion_rolls_back_handler_created_rows(session_factory):
    """Assets created by the stale worker's handler disappear with its transaction."""
    with session_factory() as session:
        _, validate = _setup_import_task(session)
        task_id = validate.id

    with session_factory() as session:
        stale = claim_task_by_id(session, task_id, "worker-old")
    _force_expired_lease(session_factory, task_id)
    with session_factory() as session:
        reap_once(session, now=utcnow())
    _make_due(session_factory, task_id)
    with session_factory() as session:
        reap_once(session, now=utcnow())
    with session_factory() as session:
        fresh = claim_task_by_id(session, task_id, "worker-new")
    assert fresh is not None

    stale_asset_id = new_id()
    with session_factory() as session:
        # This mirrors a handler that uploaded an object and registered its Asset
        # inside the task transaction before attempting the CAS.
        session.add(
            Asset(
                id=stale_asset_id,
                sha256="0" * 64,
                byte_size=0,
                media_type="application/json",
                object_key="artifacts/stale/output.json",
                origin="derived",
                logical_name="stale-output.json",
            )
        )
        session.flush()
        committed = complete_task(
            session,
            task_id,
            stale.lease_token,
            HandlerResult(data={"writer": "stale"}, output_asset_id=stale_asset_id),
        )
    assert committed is False
    with session_factory() as session:
        assert session.get(Asset, stale_asset_id) is None, "stale transaction must be rolled back"
        task = session.get(Task, task_id)
        assert task.state == TaskState.RUNNING.value
        assert task.lease_token == fresh.lease_token
        assert task.output_asset_id is None
        assert task.output_json == {}

