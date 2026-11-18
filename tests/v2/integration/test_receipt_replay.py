"""DO2-21: a lost HTTP 202 can be replayed safely with the same Idempotency-Key.

The receipt lookup happens **before** the If-Match comparison, so replaying the
original request returns the original ``operation_id``/``new_run_id`` even after
the business revision moved on - and never creates a second Run.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select

_SUPPORT = Path(__file__).resolve().parents[1]
if str(_SUPPORT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_SUPPORT))

from do_a02_support import new_episode, new_evaluation  # noqa: E402

from embodied_dataops.db.models import (  # noqa: E402
    Episode,
    GapAnalysis,
    OperationReceipt,
    PipelineRun,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

RUNS = "/api/v1/episodes/{episode_id}/pipeline-runs"
GAPS = "/api/v1/evaluations/{evaluation_id}/analyze-gaps"


def _headers(operator_headers, if_match: str, key: str) -> dict[str, str]:
    return {**operator_headers, "If-Match": if_match, "Idempotency-Key": key}


def test_replay_after_commit_returns_original_operation_and_run(client, operator_headers, session_factory):
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    path = RUNS.format(episode_id=episode_id)
    body = {"pipeline_version": "2.0.0"}
    key = str(uuid.uuid4())
    # the client lost this successful response
    first = client.post(path, json=body, headers=_headers(operator_headers, '"episode-rev:1"', key))
    assert first.status_code == 202, first.text
    original = first.json()["data"]

    # Same Key + same If-Match + same Body, even though the episode is now rev 2
    replay = client.post(path, json=body, headers=_headers(operator_headers, '"episode-rev:1"', key))
    assert replay.status_code == 202, replay.text
    assert replay.json()["data"] == original

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(PipelineRun)) == 1
        assert session.scalar(select(func.count()).select_from(OperationReceipt)) == 1
        episode = session.get(Episode, episode_id)
        assert int(episode.business_revision) == 2


def test_replay_rejects_a_different_if_match_value(client, operator_headers, session_factory):
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    path = RUNS.format(episode_id=episode_id)
    body = {"pipeline_version": "2.0.0"}
    key = str(uuid.uuid4())
    first = client.post(path, json=body, headers=_headers(operator_headers, '"episode-rev:1"', key))
    assert first.status_code == 202, first.text
    # A different strong ETag is a different request body per C3 -> key reuse.
    second = client.post(path, json=body, headers=_headers(operator_headers, '"episode-rev:2"', key))
    assert second.status_code == 409, second.text
    assert second.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(PipelineRun)) == 1


def test_receipt_lookup_returns_real_downstream_state(client, operator_headers, session_factory):
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    path = RUNS.format(episode_id=episode_id)
    key = str(uuid.uuid4())
    response = client.post(
        path,
        json={"pipeline_version": "2.0.0"},
        headers=_headers(operator_headers, '"episode-rev:1"', key),
    )
    assert response.status_code == 202, response.text
    data = response.json()["data"]

    operator_read = client.get(f"/api/v1/operations/{data['operation_id']}", headers=operator_headers)
    assert operator_read.status_code == 200, operator_read.text
    receipt = operator_read.json()["data"]
    assert receipt["effect_state"] == "COMMITTED"
    assert receipt["downstream_kind"] == "pipeline_run"
    assert receipt["downstream_id"] == data["new_run_id"]
    assert receipt["downstream_state"] in {"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}
    assert receipt["downstream_updated_at"] is not None

    by_key = client.get(f"/api/v1/operations/by-key/{key}", headers=operator_headers)
    assert by_key.status_code == 200
    assert by_key.json()["data"]["operation_id"] == data["operation_id"]

    # The downstream state must be read from the real PipelineRun row.
    with session_factory() as session:
        run = session.get(PipelineRun, data["new_run_id"])
        run.state = "RUNNING"
        session.commit()
    live = client.get(f"/api/v1/operations/{data['operation_id']}", headers=operator_headers)
    assert live.json()["data"]["downstream_state"] == "RUNNING"


def test_same_key_waits_for_inflight_commit_then_503(client, operator_headers, session_factory):
    """A same-key retry that arrives before the first commit waits max 3 s -> 503."""
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    path = RUNS.format(episode_id=episode_id)
    key = str(uuid.uuid4())
    headers = _headers(operator_headers, '"episode-rev:1"', key)

    # Simulate an in-flight first transaction: hold the episode row lock.
    blocker = session_factory()
    blocker.execute(select(Episode.id).where(Episode.id == episode_id).with_for_update())
    try:
        started = time.monotonic()
        response = client.post(path, json={"pipeline_version": "2.0.0"}, headers=headers)
        elapsed = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["error"]["code"] == "COMMIT_IN_PROGRESS"
    assert body["error"]["details"]["retry_after_ms"] >= 1000
    assert elapsed >= 2.5, f"the server must actually wait for the lock (waited {elapsed:.2f}s)"

    # The client retries with the same key; now the effect is created exactly once.
    retry = client.post(path, json={"pipeline_version": "2.0.0"}, headers=headers)
    assert retry.status_code == 202, retry.text
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(PipelineRun)) == 1
        assert session.scalar(select(func.count()).select_from(OperationReceipt)) == 1


def test_gap_analysis_replay_and_processor_both_states(client, operator_headers, session_factory):
    from embodied_dataops.operations.gap_analysis import process_queued_gap_analyses
    from embodied_dataops.workflow.outbox import utcnow

    with session_factory() as session:
        evaluation = new_evaluation(session)
        session.commit()
        evaluation_id = evaluation.id
    path = GAPS.format(evaluation_id=evaluation_id)
    body = {"success_rate_threshold": 0.8, "min_trials": 20}
    key = str(uuid.uuid4())
    first = client.post(path, json=body, headers=_headers(operator_headers, '"evaluation-rev:1"', key))
    assert first.status_code == 202, first.text
    data = first.json()["data"]
    assert data["state"] == "QUEUED"

    # Truthful immediate state: the analysis really is QUEUED before the processor.
    with session_factory() as session:
        assert session.get(GapAnalysis, data["analysis_id"]).state == "QUEUED"
        immediate = client.get(f"/api/v1/operations/{data['operation_id']}", headers=operator_headers)
        assert immediate.json()["data"]["downstream_state"] == "QUEUED"

    # Replay must not create a second analysis.
    replay = client.post(path, json=body, headers=_headers(operator_headers, '"evaluation-rev:1"', key))
    assert replay.status_code == 202, replay.text
    assert replay.json()["data"] == data
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(GapAnalysis)) == 1

    # The bounded processor completes it once; running it again is idempotent.
    for _ in range(2):
        with session_factory() as session:
            process_queued_gap_analyses(session, now=utcnow())
            session.commit()
    with session_factory() as session:
        analysis = session.get(GapAnalysis, data["analysis_id"])
        assert analysis.state == "SUCCEEDED"
        assert int((analysis.summary_json or {}).get("order_count", 0)) >= 1
    final = client.get(f"/api/v1/operations/{data['operation_id']}", headers=operator_headers)
    assert final.json()["data"]["downstream_state"] == "SUCCEEDED"
