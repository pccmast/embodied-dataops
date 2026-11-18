"""DO2-20: one effect per business revision across Agent/Key races.

The database unique fence ``(tenant_id, action_kind, target_id,
source_business_revision)`` is the authority: a second proposal for the same
revision (different Idempotency-Key) gets ``409 OPERATION_ALREADY_APPLIED`` with
the original ``operation_id`` and creates no second Run/Analysis.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

_SUPPORT = Path(__file__).resolve().parents[1]
if str(_SUPPORT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_SUPPORT))

from do_a02_support import new_episode, new_failed_task  # noqa: E402

from embodied_dataops.db.models import (  # noqa: E402
    ActionEffectFence,
    GapAnalysis,
    OperationReceipt,
    PipelineRun,
    Task,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

RUNS = "/api/v1/episodes/{episode_id}/pipeline-runs"
RETRY = "/api/v1/tasks/{task_id}/retry"


def _headers(operator_headers, if_match: str) -> dict[str, str]:
    return {
        **operator_headers,
        "If-Match": if_match,
        "Idempotency-Key": str(uuid.uuid4()),
    }


def _run_count(session_factory) -> int:
    with session_factory() as session:
        return int(session.scalar(select(func.count()).select_from(PipelineRun)) or 0)


def _assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code, body
    assert body["request_id"]
    return body


def test_second_agent_key_same_task_revision_returns_operation_id(
    client, operator_headers, session_factory
):
    with session_factory() as session:
        _, run, task = new_failed_task(session)
        task_id, expected_run_id = task.id, run.id
        session.commit()
    runs_before = _run_count(session_factory)

    path = RETRY.format(task_id=task_id)
    body = {"expected_run_id": expected_run_id}
    agent_a = client.post(path, json=body, headers=_headers(operator_headers, '"task-rev:1"'))
    assert agent_a.status_code == 202, agent_a.text
    operation_id = agent_a.json()["data"]["operation_id"]
    first_run_id = agent_a.json()["data"]["new_run_id"]

    agent_b = client.post(path, json=body, headers=_headers(operator_headers, '"task-rev:1"'))
    error = _assert_error(agent_b, 409, "OPERATION_ALREADY_APPLIED")
    assert error["error"]["details"]["operation_id"] == operation_id
    assert error["error"]["details"]["source_business_revision"] == 1

    assert _run_count(session_factory) == runs_before + 1, "exactly one new Run may exist"
    with session_factory() as session:
        task = session.get(Task, task_id)
        assert int(task.business_revision) == 2
        assert session.get(PipelineRun, first_run_id) is not None
        assert (
            int(session.scalar(select(func.count()).select_from(OperationReceipt)) or 0) == 1
        )


def test_second_agent_key_same_episode_revision_returns_operation_id(
    client, operator_headers, session_factory
):
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    runs_before = _run_count(session_factory)
    path = RUNS.format(episode_id=episode_id)
    body = {"pipeline_version": "2.0.0"}

    first = client.post(path, json=body, headers=_headers(operator_headers, '"episode-rev:1"'))
    assert first.status_code == 202, first.text
    operation_id = first.json()["data"]["operation_id"]

    second = client.post(path, json=body, headers=_headers(operator_headers, '"episode-rev:1"'))
    error = _assert_error(second, 409, "OPERATION_ALREADY_APPLIED")
    assert error["error"]["details"]["operation_id"] == operation_id
    assert _run_count(session_factory) == runs_before + 1


def test_new_revision_allows_a_new_effect(client, operator_headers, session_factory):
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    path = RUNS.format(episode_id=episode_id)
    body = {"pipeline_version": "2.0.0"}

    first = client.post(path, json=body, headers=_headers(operator_headers, '"episode-rev:1"'))
    assert first.status_code == 202, first.text
    second = client.post(path, json=body, headers=_headers(operator_headers, '"episode-rev:2"'))
    assert second.status_code == 202, second.text
    assert second.json()["data"]["operation_id"] != first.json()["data"]["operation_id"]
    assert _run_count(session_factory) == 2

    with session_factory() as session:
        fences = list(session.scalars(select(ActionEffectFence)))
        revisions = sorted(int(fence.source_business_revision) for fence in fences)
        assert revisions == [1, 2]


def test_database_unique_fence_blocks_duplicate_row(session_factory):
    with session_factory() as session:
        receipt = OperationReceipt(
            id=str(uuid.uuid4()),
            tenant_id="local",
            action_kind="REPROCESS_EPISODE",
            target_id=str(uuid.uuid4()),
            source_business_revision=1,
            idempotency_key=str(uuid.uuid4()),
            request_sha256="a" * 64,
            effect_state="COMMITTED",
            downstream_kind="pipeline_run",
            downstream_id=str(uuid.uuid4()),
        )
        session.add(receipt)
        session.flush()
        target = str(uuid.uuid4())
        session.add(
            ActionEffectFence(
                id=str(uuid.uuid4()),
                tenant_id="local",
                action_kind="REPROCESS_EPISODE",
                target_id=target,
                source_business_revision=1,
                operation_id=receipt.id,
            )
        )
        session.flush()
        session.commit()

    with session_factory() as session:
        receipt = session.scalars(select(OperationReceipt)).one()
        session.add(
            ActionEffectFence(
                id=str(uuid.uuid4()),
                tenant_id="local",
                action_kind="REPROCESS_EPISODE",
                target_id=session.scalars(select(ActionEffectFence.target_id)).one(),
                source_business_revision=1,
                operation_id=receipt.id,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()


def test_retry_requires_matching_expected_run_and_retryable_state(
    client, operator_headers, session_factory
):
    with session_factory() as session:
        _, run, task = new_failed_task(session, error_code="FILE_DECODE_ERROR")
        task_id, expected_run_id = task.id, run.id
        session.commit()

    def post(if_match: str, body: dict):
        return client.post(
            RETRY.format(task_id=task_id),
            json=body,
            headers={
                **operator_headers,
                "If-Match": if_match,
                "Idempotency-Key": str(uuid.uuid4()),
            },
        )

    # expected_run_id mismatch
    mismatch = post('"task-rev:1"', {"expected_run_id": str(uuid.uuid4())})
    _assert_error(mismatch, 409, "RETRY_NOT_SAFE")
    # retryable=False for non-retryable errors
    not_retryable = post('"task-rev:1"', {"expected_run_id": expected_run_id})
    _assert_error(not_retryable, 409, "RETRY_NOT_SAFE")
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(PipelineRun)) == 1
        assert session.scalar(select(func.count()).select_from(ActionEffectFence)) == 0


def test_retry_rejects_a_task_that_is_not_failed(client, operator_headers, session_factory):
    from do_a02_support import new_run_with_tasks

    with session_factory() as session:
        episode = new_episode(session)
        run, tasks = new_run_with_tasks(session, episode, commit=True)
        task = tasks["NORMALIZE_EPISODE"]
        task_id = task.id
        session.commit()
    response = client.post(
        RETRY.format(task_id=task_id),
        json={"expected_run_id": run.id},
        headers={
            **operator_headers,
            "If-Match": '"task-rev:1"',
            "Idempotency-Key": str(uuid.uuid4()),
        },
    )
    _assert_error(response, 409, "RETRY_NOT_SAFE")
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(GapAnalysis)) == 0
