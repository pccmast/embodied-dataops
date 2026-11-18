"""DO2-18 / DO2-19: If-Match is mandatory and stale revisions have no effect.

Every controlled POST must carry ``If-Match: "<kind>-rev:<n>"`` +
``Idempotency-Key``; a missing precondition is 428, a stale one is 412 with the
safe ``current_business_revision`` detail, and neither may create a Run, a
receipt, a fence or an outbox wakeup.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select

_SUPPORT = Path(__file__).resolve().parents[1]
if str(_SUPPORT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_SUPPORT))

from do_a02_support import (  # noqa: E402
    new_episode,
    new_evaluation,
    new_failed_task,
)

from embodied_dataops.db.models import (  # noqa: E402
    ActionEffectFence,
    Episode,
    EvaluationReport,
    GapAnalysis,
    OperationReceipt,
    OutboxEvent,
    PipelineRun,
    Task,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

RUNS = "/api/v1/episodes/{episode_id}/pipeline-runs"
RETRY = "/api/v1/tasks/{task_id}/retry"
GAPS = "/api/v1/evaluations/{evaluation_id}/analyze-gaps"


def _control_headers(operator_headers, if_match: str, key: str | None = None) -> dict[str, str]:
    return {
        **operator_headers,
        "If-Match": if_match,
        "Idempotency-Key": key or str(uuid.uuid4()),
    }


def _side_effects(session_factory) -> dict[str, int]:
    with session_factory() as session:
        return {
            "runs": int(session.scalar(select(func.count()).select_from(PipelineRun)) or 0),
            "receipts": int(session.scalar(select(func.count()).select_from(OperationReceipt)) or 0),
            "fences": int(session.scalar(select(func.count()).select_from(ActionEffectFence)) or 0),
            "outbox": int(session.scalar(select(func.count()).select_from(OutboxEvent)) or 0),
            "analyses": int(session.scalar(select(func.count()).select_from(GapAnalysis)) or 0),
        }


def _assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error", "request_id"}, body
    assert body["error"]["code"] == code, body
    assert body["error"]["message"], body
    assert isinstance(body["error"]["details"], dict), body
    assert body["request_id"], body
    return body


def _targets(session_factory):
    with session_factory() as session:
        episode = new_episode(session, index=0)
        _, run, task = new_failed_task(session)
        evaluation = new_evaluation(session)
        session.commit()
        return {
            "episode_id": episode.id,
            "episode_revision": int(episode.business_revision or 1),
            "task_id": task.id,
            "task_revision": int(task.business_revision or 1),
            "expected_run_id": run.id,
            "evaluation_id": evaluation.id,
            "evaluation_revision": int(evaluation.business_revision or 1),
        }


# ---------------------------------------------------------------------------
# DO2-18: missing If-Match -> 428 with zero upstream side effects
# ---------------------------------------------------------------------------
def test_missing_if_match_returns_428_for_all_three_writes(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    before = _side_effects(session_factory)

    bodies = {
        RUNS.format(episode_id=targets["episode_id"]): {"pipeline_version": "2.0.0"},
        RETRY.format(task_id=targets["task_id"]): {"expected_run_id": targets["expected_run_id"]},
        GAPS.format(evaluation_id=targets["evaluation_id"]): {
            "success_rate_threshold": 0.8,
            "min_trials": 20,
        },
    }
    for path, body in bodies.items():
        response = client.post(
            path,
            json=body,
            headers={**operator_headers, "Idempotency-Key": str(uuid.uuid4())},
        )
        body_json = _assert_error(response, 428, "PRECONDITION_REQUIRED")
        assert "expected_etag" in body_json["error"]["details"] or "required_header" in body_json["error"]["details"]

    assert _side_effects(session_factory) == before, "428 must not create any business effect"


# ---------------------------------------------------------------------------
# DO2-19: stale If-Match -> 412 with current_business_revision, no effect
# ---------------------------------------------------------------------------
def test_stale_episode_etag_returns_412_without_effect(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    with session_factory() as session:
        episode = session.get(Episode, targets["episode_id"])
        episode.business_revision = 2  # another business change already happened
        session.commit()
    stale = '"episode-rev:1"'
    before = _side_effects(session_factory)
    response = client.post(
        RUNS.format(episode_id=targets["episode_id"]),
        json={"pipeline_version": "2.0.0"},
        headers=_control_headers(operator_headers, stale),
    )
    body = _assert_error(response, 412, "PRECONDITION_FAILED")
    assert body["error"]["details"]["current_business_revision"] == 2
    assert _side_effects(session_factory) == before


def test_stale_task_etag_returns_412_without_effect(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    with session_factory() as session:
        task = session.get(Task, targets["task_id"])
        task.business_revision = 2
        session.commit()
    stale = '"task-rev:1"'
    before = _side_effects(session_factory)
    response = client.post(
        RETRY.format(task_id=targets["task_id"]),
        json={"expected_run_id": targets["expected_run_id"]},
        headers=_control_headers(operator_headers, stale),
    )
    body = _assert_error(response, 412, "PRECONDITION_FAILED")
    assert body["error"]["details"]["current_business_revision"] == 2
    assert _side_effects(session_factory) == before


def test_stale_evaluation_etag_returns_412_without_effect(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    with session_factory() as session:
        report = session.get(EvaluationReport, targets["evaluation_id"])
        report.business_revision = 2
        session.commit()
    stale = '"evaluation-rev:1"'
    before = _side_effects(session_factory)
    response = client.post(
        GAPS.format(evaluation_id=targets["evaluation_id"]),
        json={"success_rate_threshold": 0.8, "min_trials": 20},
        headers=_control_headers(operator_headers, stale),
    )
    body = _assert_error(response, 412, "PRECONDITION_FAILED")
    assert body["error"]["details"]["current_business_revision"] == 2
    assert _side_effects(session_factory) == before


# ---------------------------------------------------------------------------
# Happy path: 202 with the fixed C3 shape, effect + receipt + fence + revision
# ---------------------------------------------------------------------------
def test_conditional_reprocess_creates_exactly_one_run(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    headers = _control_headers(operator_headers, '"episode-rev:1"')
    response = client.post(
        RUNS.format(episode_id=targets["episode_id"]),
        json={"pipeline_version": "2.0.0"},
        headers=headers,
    )
    assert response.status_code == 202, response.text
    data = response.json()["data"]
    assert set(data) == {"operation_id", "new_run_id", "state", "effect_state"}
    assert data["state"] == "QUEUED"
    assert data["effect_state"] == "COMMITTED"

    with session_factory() as session:
        episode = session.get(Episode, targets["episode_id"])
        assert int(episode.business_revision) == 2
        run = session.get(PipelineRun, data["new_run_id"])
        assert run is not None and run.pipeline_version == "2.0.0"
        receipt = session.get(OperationReceipt, data["operation_id"])
        assert receipt.action_kind == "REPROCESS_EPISODE"
        assert receipt.effect_state == "COMMITTED"
        assert receipt.source_business_revision == 1
        assert receipt.downstream_kind == "pipeline_run"
        assert receipt.downstream_id == data["new_run_id"]
        fence = session.scalars(
            select(ActionEffectFence).where(ActionEffectFence.operation_id == data["operation_id"])
        ).one()
        assert fence.source_business_revision == 1

    refreshed = client.get(f"/api/v1/episodes/{targets['episode_id']}", headers=operator_headers)
    assert refreshed.status_code == 200
    assert refreshed.headers["etag"] == '"episode-rev:2"'
    assert refreshed.json()["data"]["episode"]["business_revision"] == 2


def test_conditional_retry_creates_new_run_from_failed_step(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    response = client.post(
        RETRY.format(task_id=targets["task_id"]),
        json={"expected_run_id": targets["expected_run_id"]},
        headers=_control_headers(operator_headers, '"task-rev:1"'),
    )
    assert response.status_code == 202, response.text
    data = response.json()["data"]
    assert set(data) == {"operation_id", "new_run_id", "state", "effect_state"}
    with session_factory() as session:
        task = session.get(Task, targets["task_id"])
        assert int(task.business_revision) == 2
        old_run = session.get(PipelineRun, targets["expected_run_id"])
        assert old_run is not None, "the failed run must be kept for audit"
        new_run = session.get(PipelineRun, data["new_run_id"])
        assert new_run is not None and new_run.id != old_run.id
        assert new_run.pipeline_version == "2.0.0"
        receipt = session.get(OperationReceipt, data["operation_id"])
        assert receipt.action_kind == "RETRY_FAILED_TASK"
        assert receipt.downstream_id == new_run.id


def test_conditional_gap_analysis_is_queued_and_receipted(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    response = client.post(
        GAPS.format(evaluation_id=targets["evaluation_id"]),
        json={"success_rate_threshold": 0.8, "min_trials": 20},
        headers=_control_headers(operator_headers, '"evaluation-rev:1"'),
    )
    assert response.status_code == 202, response.text
    data = response.json()["data"]
    assert set(data) == {"operation_id", "analysis_id", "state", "effect_state"}
    assert data["state"] == "QUEUED"
    assert data["effect_state"] == "COMMITTED"
    with session_factory() as session:
        report = session.get(EvaluationReport, targets["evaluation_id"])
        assert int(report.business_revision) == 2
        analysis = session.get(GapAnalysis, data["analysis_id"])
        assert analysis is not None and analysis.state == "QUEUED"
        receipt = session.get(OperationReceipt, data["operation_id"])
        assert receipt.downstream_kind == "gap_analysis"
        assert receipt.downstream_id == analysis.id
        wakeup = session.scalars(
            select(OutboxEvent).where(
                OutboxEvent.payload_json["gap_analysis_id"].astext == analysis.id
            )
        ).one()
        assert wakeup.topic == "stream:ops:feedback"


# ---------------------------------------------------------------------------
# Header/idempotency validation and RBAC
# ---------------------------------------------------------------------------
def test_idempotency_key_is_required_and_must_be_uuid(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    path = RUNS.format(episode_id=targets["episode_id"])
    missing = client.post(
        path,
        json={"pipeline_version": "2.0.0"},
        headers={**operator_headers, "If-Match": '"episode-rev:1"'},
    )
    _assert_error(missing, 400, "IDEMPOTENCY_KEY_REQUIRED")
    invalid = client.post(
        path,
        json={"pipeline_version": "2.0.0"},
        headers={**operator_headers, "If-Match": '"episode-rev:1"', "Idempotency-Key": "not-a-uuid"},
    )
    _assert_error(invalid, 400, "INVALID_INPUT")


def test_same_key_with_different_body_is_rejected(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    path = RUNS.format(episode_id=targets["episode_id"])
    key = str(uuid.uuid4())
    first = client.post(
        path,
        json={"pipeline_version": "2.0.0"},
        headers=_control_headers(operator_headers, '"episode-rev:1"', key),
    )
    assert first.status_code == 202, first.text
    second = client.post(
        path,
        json={"pipeline_version": "1.0.0"},
        headers=_control_headers(operator_headers, '"episode-rev:1"', key),
    )
    _assert_error(second, 409, "IDEMPOTENCY_KEY_REUSED")


def test_weak_and_malformed_etags_are_rejected(client, operator_headers, session_factory):
    targets = _targets(session_factory)
    path = RUNS.format(episode_id=targets["episode_id"])
    for bad in ('W/"episode-rev:1"', "*", '"wrong-rev:1"', "episode-rev:1"):
        response = client.post(
            path,
            json={"pipeline_version": "2.0.0"},
            headers=_control_headers(operator_headers, bad),
        )
        _assert_error(response, 412, "PRECONDITION_FAILED")


def test_viewer_cannot_execute_controlled_writes(client, viewer_headers, session_factory):
    targets = _targets(session_factory)
    response = client.post(
        RUNS.format(episode_id=targets["episode_id"]),
        json={"pipeline_version": "2.0.0"},
        headers=_control_headers(viewer_headers, '"episode-rev:1"'),
    )
    _assert_error(response, 403, "FORBIDDEN")


def test_unknown_target_returns_404_with_valid_preconditions(client, operator_headers):
    response = client.post(
        RUNS.format(episode_id=str(uuid.uuid4())),
        json={"pipeline_version": "2.0.0"},
        headers=_control_headers(operator_headers, '"episode-rev:1"'),
    )
    _assert_error(response, 404, "NOT_FOUND")
