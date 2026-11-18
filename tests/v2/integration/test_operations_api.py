"""DO2-22: operation receipts are operator-only and expose the REAL downstream state.

Also covers DO2-24's ``GET /capabilities`` start-up probe contract (viewer).
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

from do_a02_support import new_episode, new_evaluation  # noqa: E402

from embodied_dataops.db.models import CollectionOrder, GapAnalysis, PipelineRun  # noqa: E402
from embodied_dataops.operations.contract import CONTRACT_VERSION  # noqa: E402
from embodied_dataops.operations.gap_analysis import process_queued_gap_analyses  # noqa: E402
from embodied_dataops.workflow.outbox import utcnow  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

RUNS = "/api/v1/episodes/{episode_id}/pipeline-runs"
GAPS = "/api/v1/evaluations/{evaluation_id}/analyze-gaps"

RECEIPT_FIELDS = {
    "operation_id",
    "action_kind",
    "target_id",
    "source_business_revision",
    "effect_state",
    "downstream_kind",
    "downstream_id",
    "downstream_state",
    "downstream_updated_at",
    "error_code",
}

SENSITIVE_MARKERS = ("url", "signature", "secret", "stack", "traceback", "object_key", "presign", "password")


def _headers(operator_headers, if_match: str) -> dict[str, str]:
    return {**operator_headers, "If-Match": if_match, "Idempotency-Key": str(uuid.uuid4())}


def _create_episode(client, operator_headers, session_factory) -> tuple[str, dict]:
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    response = client.post(
        RUNS.format(episode_id=episode_id),
        json={"pipeline_version": "2.0.0"},
        headers=_headers(operator_headers, '"episode-rev:1"'),
    )
    assert response.status_code == 202, response.text
    return episode_id, response.json()["data"]


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------
def test_receipts_are_operator_only(client, viewer_headers, reviewer_headers, operator_headers, session_factory):
    _, data = _create_episode(client, operator_headers, session_factory)
    for headers in (viewer_headers, reviewer_headers):
        for path in (
            f"/api/v1/operations/{data['operation_id']}",
            "/api/v1/operations/by-key/does-not-exist",
        ):
            response = client.get(path, headers=headers)
            assert response.status_code == 403, response.text
            assert response.json()["error"]["code"] == "FORBIDDEN"
    operator = client.get(f"/api/v1/operations/{data['operation_id']}", headers=operator_headers)
    assert operator.status_code == 200, operator.text


def test_receipt_requires_authentication(client, session_factory):
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    response = client.get(f"/api/v1/operations/{episode_id}")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


# ---------------------------------------------------------------------------
# Receipt content / safety
# ---------------------------------------------------------------------------
def test_receipt_has_exact_fields_and_no_sensitive_values(client, operator_headers, session_factory):
    _, data = _create_episode(client, operator_headers, session_factory)
    response = client.get(f"/api/v1/operations/{data['operation_id']}", headers=operator_headers)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"data", "request_id"}
    receipt = payload["data"]
    assert set(receipt) == RECEIPT_FIELDS
    assert receipt["operation_id"] == data["operation_id"]
    assert receipt["action_kind"] == "REPROCESS_EPISODE"
    assert receipt["effect_state"] == "COMMITTED"
    assert receipt["downstream_kind"] == "pipeline_run"
    assert receipt["downstream_id"] == data["new_run_id"]
    assert isinstance(receipt["source_business_revision"], int)
    assert receipt["error_code"] is None
    serialized = str(receipt).lower()
    for marker in SENSITIVE_MARKERS:
        assert marker not in serialized, f"receipt leaks {marker}: {receipt}"


def test_unknown_receipt_and_key_return_404(client, operator_headers):
    assert client.get(f"/api/v1/operations/{uuid.uuid4()}", headers=operator_headers).status_code == 404
    assert (
        client.get(f"/api/v1/operations/by-key/{uuid.uuid4()}", headers=operator_headers).status_code
        == 404
    )


def test_receipt_by_key_matches_by_id(client, operator_headers, session_factory):
    with session_factory() as session:
        episode = new_episode(session)
        session.commit()
        episode_id = episode.id
    key = str(uuid.uuid4())
    response = client.post(
        RUNS.format(episode_id=episode_id),
        json={"pipeline_version": "2.0.0"},
        headers={**operator_headers, "If-Match": '"episode-rev:1"', "Idempotency-Key": key},
    )
    assert response.status_code == 202, response.text
    operation_id = response.json()["data"]["operation_id"]
    by_id = client.get(f"/api/v1/operations/{operation_id}", headers=operator_headers).json()["data"]
    by_key = client.get(f"/api/v1/operations/by-key/{key}", headers=operator_headers).json()["data"]
    assert by_id == by_key


# ---------------------------------------------------------------------------
# Real downstream state (PipelineRun + GapAnalysis)
# ---------------------------------------------------------------------------
def test_receipt_reads_live_pipeline_run_state(client, operator_headers, session_factory):
    _, data = _create_episode(client, operator_headers, session_factory)
    with session_factory() as session:
        run = session.get(PipelineRun, data["new_run_id"])
        run.state = "SUCCEEDED"
        run.finished_at = utcnow()
        session.commit()
    receipt = client.get(f"/api/v1/operations/{data['operation_id']}", headers=operator_headers).json()["data"]
    assert receipt["downstream_state"] == "SUCCEEDED"
    assert receipt["downstream_updated_at"].endswith("Z")


def test_receipt_reads_live_gap_analysis_state_queued_then_succeeded(
    client, operator_headers, session_factory
):
    with session_factory() as session:
        evaluation = new_evaluation(session)
        session.commit()
        evaluation_id = evaluation.id
    response = client.post(
        GAPS.format(evaluation_id=evaluation_id),
        json={"success_rate_threshold": 0.8, "min_trials": 20},
        headers=_headers(operator_headers, '"evaluation-rev:1"'),
    )
    assert response.status_code == 202, response.text
    data = response.json()["data"]

    queued = client.get(f"/api/v1/operations/{data['operation_id']}", headers=operator_headers).json()["data"]
    assert queued["downstream_kind"] == "gap_analysis"
    assert queued["downstream_state"] == "QUEUED"
    assert queued["downstream_id"] == data["analysis_id"]

    with session_factory() as session:
        process_queued_gap_analyses(session, now=utcnow())
        session.commit()
        assert session.get(GapAnalysis, data["analysis_id"]).state == "SUCCEEDED"
        orders = int(
            session.scalar(
                select(func.count())
                .select_from(CollectionOrder)
                .where(CollectionOrder.evaluation_report_id == evaluation_id)
            )
            or 0
        )
        # transparent_cup_low_light 30/12 < 0.8 -> exactly one order (requested 36)
        assert orders == 1
        transparent = session.scalars(
            select(CollectionOrder).where(CollectionOrder.scenario_key == "transparent_cup_low_light")
        ).one()
        assert transparent.requested_count == 36
        assert transparent.state == "PROPOSED"

    succeeded = client.get(
        f"/api/v1/operations/{data['operation_id']}", headers=operator_headers
    ).json()["data"]
    assert succeeded["downstream_state"] == "SUCCEEDED"


# ---------------------------------------------------------------------------
# GET /capabilities (DO-A02 C1)
# ---------------------------------------------------------------------------
def test_capabilities_contract_payload(client, viewer_headers):
    response = client.get("/api/v1/capabilities", headers=viewer_headers)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {"data", "request_id"}
    data = payload["data"]
    assert data == {
        "contract_version": CONTRACT_VERSION,
        "canonical_versions": ["1.0", "2.0"],
        "robot_profiles": ["arm6_joint_position_v1", "arm7_joint_position_v1"],
        "qc_policy_versions": ["qc-policy/1.0", "qc-policy/2.0"],
        "conditional_writes": True,
        "operation_receipts": True,
        "diagnostic_queries": True,
    }
    assert CONTRACT_VERSION == "do-agent-contract/2.0.0"


def test_capabilities_requires_authentication(client):
    assert client.get("/api/v1/capabilities").status_code == 401
