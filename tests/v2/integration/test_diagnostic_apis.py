"""DO2-23: the three bounded read-only diagnostic APIs (DO-A02 C5).

Strict fields, strict limits, deterministic evidence and no leakage of stack
traces, host secrets or raw object bytes.
"""

from __future__ import annotations

import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

_SUPPORT = Path(__file__).resolve().parents[1]
if str(_SUPPORT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_SUPPORT))

from do_a02_support import (  # noqa: E402
    add_quality_report,
    epoch_now,
    new_episode,
    new_run_with_tasks,
    write_stream_asset,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

TIMELINE = "/api/v1/episodes/{episode_id}/timeline"
FAILURE = "/api/v1/pipeline-runs/{run_id}/failure-analysis"
AGGREGATE = "/api/v1/quality/aggregate"

TIMELINE_STREAM_FIELDS = {
    "name",
    "kind",
    "clock_domain",
    "sample_count",
    "observed_gap_count",
    "max_gap_ns",
    "p95_alignment_error_ns",
    "points",
    "truncated",
}
POINT_FIELDS = {"t_ns", "source_t_ns", "valid"}
FAILED_TASK_FIELDS = {
    "task_id",
    "step",
    "error_code",
    "attempt",
    "worker_pool",
    "first_failed_at",
    "last_failed_at",
}
SENSITIVE_MARKERS = ("stack", "traceback", "secret", "password", "object_key", "presign", "http://", "https://")


def _error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code, body
    return body


# ---------------------------------------------------------------------------
# 1. Timeline
# ---------------------------------------------------------------------------
def _timeline_episode(session, storage, *, duration_ns=20_000_000_000):
    episode = new_episode(session, duration_ns=duration_ns)
    write_stream_asset(
        session,
        storage,
        episode,
        name="front",
        kind="camera",
        times_ns=[i * 100_000_000 for i in range(10)],
        stream_index=0,
    )
    write_stream_asset(
        session,
        storage,
        episode,
        name="state",
        kind="state",
        clock_domain="ros_header_stamp",
        times_ns=[i * 10_000_000 for i in range(100)],
        stream_index=1,
    )
    write_stream_asset(
        session,
        storage,
        episode,
        name="action",
        kind="action",
        times_ns=[i * 20_000_000 for i in range(50)],
        stream_index=2,
    )
    session.commit()
    return episode


def test_timeline_returns_bounded_deterministic_evidence(client, viewer_headers, session_factory, storage):
    with session_factory() as session:
        episode = _timeline_episode(session, storage)
        episode_id = episode.id

    response = client.get(
        TIMELINE.format(episode_id=episode_id),
        params={"start_ns": 0, "end_ns": 1_000_000_000, "max_points": 20},
        headers=viewer_headers,
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert set(data) == {"episode_id", "profile_id", "window", "streams", "evidence_digest"}
    assert data["episode_id"] == episode_id
    assert data["window"] == {"start_ns": 0, "end_ns": 1_000_000_000}
    assert len(data["streams"]) == 3
    assert len(data["evidence_digest"]) == 64

    by_name = {stream["name"]: stream for stream in data["streams"]}
    for stream in data["streams"]:
        assert set(stream) == TIMELINE_STREAM_FIELDS
        assert len(stream["points"]) <= 20
        for point in stream["points"]:
            assert set(point) == POINT_FIELDS

    front = by_name["front"]
    assert front["sample_count"] == 10
    assert front["observed_gap_count"] == 0
    assert front["max_gap_ns"] == 100_000_000
    assert front["p95_alignment_error_ns"] == 1_000_000
    assert front["truncated"] is False
    assert [point["t_ns"] for point in front["points"]][0] == 0
    assert [point["t_ns"] for point in front["points"]][-1] == 900_000_000

    state = by_name["state"]
    assert state["truncated"] is True, "100 samples must be down-sampled to max_points"
    assert len(state["points"]) == 20
    assert state["points"][0]["t_ns"] == 0
    assert state["points"][-1]["t_ns"] == 990_000_000, "the last point in the window must be kept"
    assert state["clock_domain"] == "ros_header_stamp"

    again = client.get(
        TIMELINE.format(episode_id=episode_id),
        params={"start_ns": 0, "end_ns": 1_000_000_000, "max_points": 20},
        headers=viewer_headers,
    )
    assert again.json()["data"]["evidence_digest"] == data["evidence_digest"]


def test_timeline_rejects_out_of_bound_windows(client, viewer_headers, session_factory, storage):
    with session_factory() as session:
        episode = _timeline_episode(session, storage)
        episode_id = episode.id
    path = TIMELINE.format(episode_id=episode_id)
    _error(
        client.get(path, params={"start_ns": 0, "end_ns": 15_000_000_000}, headers=viewer_headers),
        422,
        "INVALID_INPUT",
    )
    _error(
        client.get(path, params={"start_ns": 0, "end_ns": 1, "max_points": 0}, headers=viewer_headers),
        422,
        "VALIDATION_FAILED",
    )
    _error(
        client.get(path, params={"start_ns": 5, "end_ns": 5}, headers=viewer_headers),
        422,
        "INVALID_INPUT",
    )
    _error(
        client.get(
            path, params={"start_ns": 0, "end_ns": 25_000_000_000}, headers=viewer_headers
        ),
        422,
        "INVALID_INPUT",
    )
    _error(client.get(path, headers=viewer_headers), 404, "NOT_FOUND")


def test_timeline_rejects_more_than_eight_streams(client, viewer_headers, session_factory, storage):
    with session_factory() as session:
        episode = new_episode(session)
        for index in range(9):
            write_stream_asset(
                session,
                storage,
                episode,
                name=f"stream_{index}",
                kind="camera",
                stream_index=index,
            )
        session.commit()
        episode_id = episode.id
    response = client.get(
        TIMELINE.format(episode_id=episode_id),
        params={"start_ns": 0, "end_ns": 1_000_000_000},
        headers=viewer_headers,
    )
    body = _error(response, 413, "TOO_MANY_MODALITIES")
    assert body["error"]["details"]["modality_count"] == 9
    assert body["error"]["details"]["max_streams"] == 8


# ---------------------------------------------------------------------------
# 2. Failure analysis
# ---------------------------------------------------------------------------
def test_failure_analysis_returns_structured_bounded_evidence(client, viewer_headers, session_factory):
    with session_factory() as session:
        episode = new_episode(session)
        run, tasks = new_run_with_tasks(session, episode, commit=False)
        normalize = tasks["NORMALIZE_EPISODE"]
        quality = tasks["QUALITY_CHECK"]
        normalize.state = "FAILED"
        normalize.error_code = "FILE_DECODE_ERROR"
        normalize.attempt = 1
        normalize.worker_id = "worker-cpu-a"
        quality.state = "CANCELLED"
        session.commit()
        run_id = run.id

    response = client.get(FAILURE.format(run_id=run_id), headers=viewer_headers)
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert set(data) == {
        "run_id",
        "failed_tasks",
        "dependencies",
        "worker_incidents",
        "summary_rule_codes",
        "truncated",
    }
    assert data["run_id"] == run_id
    assert len(data["failed_tasks"]) == 1
    failed = data["failed_tasks"][0]
    assert set(failed) == FAILED_TASK_FIELDS
    assert failed["error_code"] == "FILE_DECODE_ERROR"
    assert failed["worker_pool"] == "cpu"
    assert data["summary_rule_codes"] == ["FILE_DECODE_ERROR"]
    assert data["truncated"] is False
    assert len(data["worker_incidents"]) == 1
    assert set(data["worker_incidents"][0]) == {"incident_code", "at", "worker_pool"}
    assert data["dependencies"], "the DAG edges must be reported"
    for dependency in data["dependencies"]:
        assert set(dependency) == {"task_id", "depends_on_task_id", "state"}

    serialized = str(data).lower()
    for marker in SENSITIVE_MARKERS:
        assert marker not in serialized, f"failure analysis leaks {marker}"
    assert "error_detail" not in serialized


def test_failure_analysis_unknown_run_is_404(client, viewer_headers):
    assert client.get(FAILURE.format(run_id=uuid.uuid4()), headers=viewer_headers).status_code == 404


# ---------------------------------------------------------------------------
# 3. Quality aggregate
# ---------------------------------------------------------------------------
def test_quality_aggregate_is_a_population_query_not_a_page(client, viewer_headers, session_factory):
    now = epoch_now()
    with session_factory() as session:
        episode_pass = new_episode(session, index=0, source_type="LEROBOT_V3")
        add_quality_report(
            session, episode_pass, status="PASS", rule_codes=("TIMESTAMP_GAP",), created_at=now
        )
        episode_fail = new_episode(session, index=1, source_type="LEROBOT_V3")
        add_quality_report(
            session, episode_fail, status="FAIL", rule_codes=("TIMESTAMP_GAP",), created_at=now
        )
        episode_ego = new_episode(
            session, index=2, source_type="EGO_MP4", data_origin="public_dataset"
        )
        add_quality_report(
            session, episode_ego, status="FAIL", rule_codes=("LOW_QUALITY",), created_at=now
        )
        # Outside the window: must not be counted.
        old_episode = new_episode(session, index=3, source_type="LEROBOT_V3")
        add_quality_report(
            session,
            old_episode,
            status="FAIL",
            rule_codes=("TIMESTAMP_GAP",),
            created_at=now - timedelta(days=10),
        )
        # Only the newest completed report per episode counts.
        episode_multi = new_episode(session, index=4, source_type="LEROBOT_V3")
        add_quality_report(
            session, episode_multi, status="FAIL", rule_codes=("STALE",), created_at=now - timedelta(hours=1)
        )
        add_quality_report(session, episode_multi, status="PASS", created_at=now)
        session.commit()

    window = {
        "from": (now - timedelta(days=2)).isoformat(),
        "to": (now + timedelta(seconds=30)).isoformat(),
    }

    grouped = client.get(
        AGGREGATE,
        params={**window, "group_by": "source_type", "limit": 20},
        headers=viewer_headers,
    )
    assert grouped.status_code == 200, grouped.text
    data = grouped.json()["data"]
    assert set(data) >= {
        "matched_episode_count",
        "group_by",
        "groups",
        "truncated",
        "computed_at",
        "filter_window",
    }
    assert data["matched_episode_count"] == 4
    assert data["group_by"] == "source_type"
    groups = {group["key"]: group for group in data["groups"]}
    assert groups["LEROBOT_V3"] == {
        "key": "LEROBOT_V3",
        "episode_count": 3,
        "fail_episode_count": 1,
        "warn_episode_count": 0,
    }
    assert groups["EGO_MP4"]["episode_count"] == 1
    assert groups["EGO_MP4"]["fail_episode_count"] == 1
    assert data["truncated"] is False
    assert data["statistical_basis"]

    by_rule = client.get(
        AGGREGATE,
        params={**window, "group_by": "rule_code", "limit": 20},
        headers=viewer_headers,
    )
    rule_groups = {group["key"]: group for group in by_rule.json()["data"]["groups"]}
    assert rule_groups["TIMESTAMP_GAP"]["episode_count"] == 2
    assert rule_groups["TIMESTAMP_GAP"]["fail_episode_count"] == 1
    assert rule_groups["LOW_QUALITY"]["episode_count"] == 1
    assert "STALE" not in rule_groups, "only the latest report per episode may count"
    # rule_code groups may legitimately sum to more than matched_episode_count.

    by_profile = client.get(
        AGGREGATE,
        params={**window, "group_by": "profile_id", "limit": 20},
        headers=viewer_headers,
    )
    profile_groups = {group["key"]: group for group in by_profile.json()["data"]["groups"]}
    assert profile_groups["arm6_joint_position_v1"]["episode_count"] == 4

    by_collection = client.get(
        AGGREGATE,
        params={**window, "group_by": "collection_id", "limit": 20},
        headers=viewer_headers,
    )
    collection_groups = by_collection.json()["data"]["groups"]
    assert len(collection_groups) == 1
    assert collection_groups[0]["key"] == "unassigned"
    assert collection_groups[0]["episode_count"] == 4

    # Sorting is count DESC then key ASC and truncation is declared.
    limited = client.get(
        AGGREGATE,
        params={**window, "group_by": "source_type", "limit": 1},
        headers=viewer_headers,
    )
    limited_data = limited.json()["data"]
    assert len(limited_data["groups"]) == 1
    assert limited_data["groups"][0]["episode_count"] == 3
    assert limited_data["truncated"] is True


def test_quality_aggregate_rejects_invalid_bounds(client, viewer_headers):
    now = epoch_now()
    params = {"from": (now - timedelta(days=2)).isoformat(), "to": now.isoformat()}
    _error(
        client.get(AGGREGATE, params={**params, "group_by": "not_a_group"}, headers=viewer_headers),
        422,
        "INVALID_INPUT",
    )
    _error(
        client.get(
            AGGREGATE,
            params={
                "from": (now - timedelta(days=40)).isoformat(),
                "to": now.isoformat(),
                "group_by": "source_type",
            },
            headers=viewer_headers,
        ),
        422,
        "INVALID_INPUT",
    )
    _error(
        client.get(
            AGGREGATE,
            params={
                "from": now.isoformat(),
                "to": (now - timedelta(days=1)).isoformat(),
                "group_by": "source_type",
            },
            headers=viewer_headers,
        ),
        422,
        "INVALID_INPUT",
    )
    _error(
        client.get(AGGREGATE, params={**params, "group_by": "source_type", "limit": 51}, headers=viewer_headers),
        422,
        "VALIDATION_FAILED",
    )


def test_diagnostics_require_authentication(client, session_factory):
    with session_factory() as session:
        episode = new_episode(session)
        run, _ = new_run_with_tasks(session, episode)
        session.commit()
        episode_id, run_id = episode.id, run.id
    assert client.get(TIMELINE.format(episode_id=episode_id)).status_code == 401
    assert client.get(FAILURE.format(run_id=run_id)).status_code == 401
    assert (
        client.get(
            AGGREGATE,
            params={
                "from": (epoch_now() - timedelta(days=1)).isoformat(),
                "to": epoch_now().isoformat(),
                "group_by": "source_type",
            },
        ).status_code
        == 401
    )
