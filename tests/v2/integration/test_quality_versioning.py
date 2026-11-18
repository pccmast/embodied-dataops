"""DO2-13 acceptance: qc-policy/1.0 + qc-policy/2.0 reports coexist and never overwrite.

The test is a real end-to-end run over PostgreSQL + the local object store + the
FastAPI app (``TestClient``):

1. one Episode is quality checked with ``qc-policy/1.0`` while its canonical
   version is ``1`` (WARN report with ``FRAME_GAP_EXCESS``);
2. the Episode advances to ``canonical-episode/2.0`` ``canonical_version=2`` and
   is quality checked again with ``qc-policy/2.0`` (FAIL report with
   ``JOINT_VELOCITY_SPIKE`` and full evidence provenance);
3. both ``quality_reports`` rows and their immutable report artefacts coexist;
   the v1 row, its report object and a published ``dataset_version_episodes``
   reference to it are byte/field unchanged;
4. ``GET /api/v1/episodes/{id}/quality`` returns the latest completed report for
   the current canonical version by default, ``?policy_version=qc-policy/1.0``
   returns the historical v1 report with its own ``canonical_version`` and
   ``policy_version``, and an invalid policy version is rejected with 422;
5. v2 findings persist ``source_stream``/``source_clock_domain``/``method``/
   ``limits``/``uncertainty_note``; v1 findings remain null/empty-tolerant.

Run with the private DB::

    TEST_DATABASE_URL=postgresql+psycopg://dataops:dataops@localhost:5432/dataops_test_v2qc13
    TEST_REDIS_URL=redis://localhost:6379/12
    STORAGE_BACKEND=local DETECTOR=fake
    pytest tests/v2/integration/test_quality_versioning.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SUPPORT_DIR = Path(__file__).resolve().parents[1]
if str(_SUPPORT_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_SUPPORT_DIR))

from do_a02_support import add_quality_report, new_dataset_version, new_episode  # noqa: E402
from fixtures.v2.semantic_qc_cases import write_canonical_v2_objects  # noqa: E402
from sqlalchemy import select  # noqa: E402

from embodied_dataops.canonical.contract import (  # noqa: E402
    SampleRow,
    canonical_json_bytes,
    canonical_keys,
    write_samples_parquet,
)
from embodied_dataops.config import get_settings  # noqa: E402
from embodied_dataops.db.models import (  # noqa: E402
    Asset,
    DatasetVersionEpisode,
    Episode,
    PipelineRun,
    QualityFinding,
    QualityReport,
    Task,
)
from embodied_dataops.domain.profiles import load_profiles, upsert_builtin_profiles  # noqa: E402
from embodied_dataops.ids import new_id, new_token  # noqa: E402
from embodied_dataops.quality.v2_policy import (  # noqa: E402
    QUALITY_REPORT_V2_SCHEMA_VERSION,
)
from embodied_dataops.storage.base import LocalStorage  # noqa: E402
from embodied_dataops.workflow.handlers.base import HandlerContext  # noqa: E402
from embodied_dataops.workflow.handlers.quality_check import QualityCheckHandler  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

QUALITY_API = "/api/v1/episodes/{episode_id}/quality"
INTERVAL_NS = 100_000_000
JOINTS = [f"joint_{index}" for index in range(1, 7)]
PROFILE = load_profiles()[0]
PROFILE_SHA = PROFILE.config_sha256()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _add_run(db, episode: Episode, *, pipeline_version: str) -> PipelineRun:
    run = PipelineRun(
        id=new_id(),
        episode_id=episode.id,
        pipeline_version=pipeline_version,
        state="RUNNING",
    )
    db.add(run)
    db.flush()
    return run


def _add_task(
    db,
    run: PipelineRun,
    *,
    step: str,
    output_json: dict | None = None,
    state: str = "RUNNING",
) -> Task:
    task = Task(
        id=new_id(),
        run_id=run.id,
        step=step,
        pool="cpu",
        state=state,
        attempt=1,
        max_attempts=3,
        worker_id="test-worker",
        lease_token=new_token(),
        output_json=output_json or {},
    )
    db.add(task)
    db.flush()
    return task


def _context(db, task: Task, storage: LocalStorage, workspace: Path) -> HandlerContext:
    return HandlerContext(
        session=db,
        task=task,
        settings=get_settings(),
        storage=storage,
        workspace=workspace,
        cancel_check=lambda: False,
    )


def _v1_samples() -> list[SampleRow]:
    rows: list[SampleRow] = []
    for index in range(100):
        t_ns = index * INTERVAL_NS
        rows.append(
            SampleRow(
                sample_index=index,
                t_ns=t_ns,
                frame_index=index,
                state=[0.05 * joint + 0.001 * index for joint in range(6)],
                action=[0.06 * joint + 0.001 * index for joint in range(6)],
                state_source_t_ns=t_ns,
                action_source_t_ns=t_ns,
            )
        )
    return rows


def _write_v1_canonical(storage: LocalStorage, episode: Episode, workspace: Path) -> tuple[dict, dict]:
    """Write canonical-episode/1.0 metadata + samples and the normalizer output."""
    keys = canonical_keys(episode.id, 1)
    metadata = {
        "schema_version": "canonical-episode/1.0",
        "episode_id": episode.id,
        "episode_type": episode.episode_type,
        "data_origin": "synthetic",
        "source_type": "LEROBOT_V3",
        "source_asset_sha256": "a" * 64,
        "source_episode_index": episode.source_episode_index,
        "duration_ns": episode.duration_ns,
        "target_fps": 10,
        "camera_name": "front",
        "clock_domain": "device_monotonic",
        "joint_names": JOINTS,
        "state_dim": 6,
        "action_dim": 6,
        "samples_parquet": "samples.parquet",
        "video_path": "frames/front.mp4",
        "pipeline_version": "1.0.0",
        "normalization_valid": True,
        "sample_count": 100,
        "robot_profile": "arm6_joint_position_v1",
        "video_available": False,
    }
    storage.put_bytes(keys["metadata"], canonical_json_bytes(metadata), content_type="application/json")
    samples_path = workspace / "v1-samples.parquet"
    write_samples_parquet(samples_path, _v1_samples(), "robot_episode")
    storage.put_file(keys["samples"], samples_path, content_type="application/vnd.apache.parquet")

    # A 300 ms source camera gap is a genuine qc-policy/1.0 WARN finding: the v1
    # report must own its finding rows before the v2 report exists.
    camera_times = [index * INTERVAL_NS for index in range(100)]
    for index in range(51, 100):
        camera_times[index] += 300_000_000
    normalize_output = {
        "normalization_valid": True,
        "episode_type": episode.episode_type,
        "duration_ns": episode.duration_ns,
        "sample_count": 100,
        "canonical": {
            "metadata_key": keys["metadata"],
            "samples_key": keys["samples"],
            "video_key": keys["video"],
        },
        "alignment": {
            "grid": [index * INTERVAL_NS for index in range(100)],
            "grid_points": 100,
            "missing_sample_count": 0,
            "missing_frame_indexes": [],
            "missing_state_indexes": [],
            "stale_action_indexes": [],
            "origin_ns": 0,
        },
        "stream_times": {"camera": camera_times},
        "state_dim": 6,
        "action_dim": 6,
        "joint_names": JOINTS,
        "semantics_verified": True,
        "diagnostics": [],
        "clock_domains": {
            "camera": "device_monotonic",
            "state": "device_monotonic",
            "action": "device_monotonic",
        },
    }
    return keys, normalize_output


def _run_v1_quality_check(
    db, storage: LocalStorage, episode: Episode, workspace: Path
) -> tuple[PipelineRun, Task, object]:
    run = _add_run(db, episode, pipeline_version="1.0.0")
    _keys, normalize_output = _write_v1_canonical(storage, episode, workspace)
    _add_task(db, run, step="NORMALIZE_EPISODE", output_json=normalize_output, state="SUCCEEDED")
    task = _add_task(db, run, step="QUALITY_CHECK")
    result = QualityCheckHandler().handle(_context(db, task, storage, workspace))
    db.commit()
    db.expire_all()
    return run, task, result


def _run_v2_quality_check(
    db, storage: LocalStorage, episode: Episode, workspace: Path
) -> tuple[PipelineRun, Task, object]:
    write_canonical_v2_objects(
        storage,
        workspace,
        episode_id=episode.id,
        case="velocity_spike",
        canonical_version=episode.canonical_version,
    )
    upsert_builtin_profiles(db)
    run = _add_run(db, episode, pipeline_version="2.0.0")
    task = _add_task(db, run, step="QUALITY_CHECK")
    result = QualityCheckHandler().handle(_context(db, task, storage, workspace))
    db.commit()
    db.expire_all()
    return run, task, result


def _report_for_run(db, run_id: str) -> QualityReport:
    report = db.scalars(
        select(QualityReport).where(QualityReport.run_id == run_id)
    ).one()
    return report


def _report_snapshot(report: QualityReport) -> dict:
    return {
        "id": report.id,
        "episode_id": report.episode_id,
        "run_id": report.run_id,
        "policy_version": report.policy_version,
        "status": report.status,
        "schema_version": report.schema_version,
        "summary_json": dict(report.summary_json),
        "report_asset_id": report.report_asset_id,
        "profile_id": report.profile_id,
        "profile_revision": report.profile_revision,
        "config_sha256": report.config_sha256,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


def _api(client, episode_id: str, headers: dict, **params):
    return client.get(QUALITY_API.format(episode_id=episode_id), params=params, headers=headers)


# ---------------------------------------------------------------------------
# DO2-13 main acceptance: 1.0 then 2.0 on one Episode, both coexist
# ---------------------------------------------------------------------------


def test_qc_policy_v1_and_v2_reports_coexist_and_default_api_selects_current(
    client, viewer_headers, db, storage: LocalStorage, tmp_path: Path
) -> None:
    episode = new_episode(
        db,
        canonical_schema_version="1.0",
        canonical_version=1,
        profile_id="arm6_joint_position_v1",
        profile_revision=1,
        data_origin="synthetic",
    )
    db.commit()

    # -- 1. qc-policy/1.0 report while canonical-episode/1.0 is current --------
    run1, _task1, result1 = _run_v1_quality_check(db, storage, episode, tmp_path)
    assert result1.verdict == "WARN"
    report1 = _report_for_run(db, run1.id)
    assert report1.policy_version == "qc-policy/1.0"
    assert report1.schema_version == "quality-report/1.0"
    assert report1.status == "WARN"

    findings1 = list(
        db.scalars(select(QualityFinding).where(QualityFinding.report_id == report1.id))
    )
    assert [finding.rule_code for finding in findings1] == ["FRAME_GAP_EXCESS"]
    v1_finding = findings1[0]
    # v1 rows are null/empty tolerant for the V2 evidence-provenance columns
    assert v1_finding.source_stream is None
    assert v1_finding.source_clock_domain is None
    assert v1_finding.method is None
    assert v1_finding.limits_json == {}
    assert v1_finding.uncertainty_note is None

    assert report1.report_asset_id is not None
    asset1 = db.get(Asset, report1.report_asset_id)
    assert asset1 is not None
    v1_report_bytes = storage.get_bytes(asset1.object_key)
    v1_report_sha = asset1.sha256
    snapshot_before = _report_snapshot(report1)

    # an "approved/published" reference to the v1 report must never be rewritten
    _dataset, version = new_dataset_version(db)
    reference = DatasetVersionEpisode(
        version_id=version.id,
        episode_id=episode.id,
        canonical_digest="e" * 64,
        quality_report_id=report1.id,
        annotation_set_id=None,
        position=0,
    )
    db.add(reference)
    db.commit()
    db.expire_all()
    reference_before = {
        "version_id": reference.version_id,
        "episode_id": reference.episode_id,
        "canonical_digest": reference.canonical_digest,
        "quality_report_id": reference.quality_report_id,
    }

    # -- 2. Episode advances to canonical-episode/2.0 version 2 ----------------
    episode = db.get(Episode, episode.id)
    assert episode is not None
    episode.canonical_schema_version = "2.0"
    episode.canonical_version = 2
    db.commit()

    run2, _task2, result2 = _run_v2_quality_check(db, storage, episode, tmp_path)
    assert result2.verdict == "FAIL"
    report2 = _report_for_run(db, run2.id)
    assert report2.policy_version == "qc-policy/2.0"
    assert report2.schema_version == QUALITY_REPORT_V2_SCHEMA_VERSION
    assert report2.status == "FAIL"
    assert report2.profile_id == "arm6_joint_position_v1"
    assert report2.profile_revision == 1
    assert report2.config_sha256 == PROFILE_SHA
    assert report2.summary_json["canonical_version"] == 2
    assert report2.summary_json["policy_config_sha256"]
    assert report2.run_id != report1.run_id

    # -- 3. the v1 row, its artefact and the published reference are unchanged -
    report1_after = db.get(QualityReport, report1.id)
    assert report1_after is not None
    assert _report_snapshot(report1_after) == snapshot_before
    asset1_after = db.get(Asset, report1.report_asset_id)
    assert asset1_after is not None
    assert asset1_after.sha256 == v1_report_sha
    assert storage.get_bytes(asset1_after.object_key) == v1_report_bytes
    reference_after = db.get(
        DatasetVersionEpisode,
        {"version_id": reference.version_id, "episode_id": reference.episode_id},
    )
    assert reference_after is not None
    assert {
        "version_id": reference_after.version_id,
        "episode_id": reference_after.episode_id,
        "canonical_digest": reference_after.canonical_digest,
        "quality_report_id": reference_after.quality_report_id,
    } == reference_before
    assert db.get(Episode, episode.id).quality_status == "FAIL"

    # -- 4. v2 findings persist full evidence provenance ----------------------
    findings2 = list(
        db.scalars(select(QualityFinding).where(QualityFinding.report_id == report2.id))
    )
    assert [finding.rule_code for finding in findings2] == ["JOINT_VELOCITY_SPIKE"]
    v2_finding = findings2[0]
    assert v2_finding.source_stream == "state"
    assert v2_finding.source_clock_domain == "mcap_log_time"
    assert v2_finding.method is not None and "adjacent_valid_state" in v2_finding.method
    assert isinstance(v2_finding.limits_json, dict) and v2_finding.limits_json
    assert v2_finding.uncertainty_note is None or isinstance(v2_finding.uncertainty_note, str)

    # -- 5. API version selection (real FastAPI app over the same DB) ---------
    default = _api(client, episode.id, viewer_headers)
    assert default.status_code == 200, default.text
    data = default.json()["data"]
    assert data["policy_version"] == "qc-policy/2.0"
    assert data["schema_version"] == QUALITY_REPORT_V2_SCHEMA_VERSION
    assert data["canonical_version"] == 2
    assert data["verdict"] == "FAIL" and data["status"] == "FAIL"
    assert data["report_id"] == report2.id
    # legacy V1 response fields are retained
    assert {"report_id", "run_id", "policy_version", "verdict", "status", "summary", "metrics", "findings", "report_url", "created_at"} <= set(data)
    finding_payload = data["findings"][0]
    assert finding_payload["rule_code"] == "JOINT_VELOCITY_SPIKE"
    assert finding_payload["source_stream"] == "state"
    assert finding_payload["source_clock_domain"] == "mcap_log_time"
    assert finding_payload["method"]
    assert isinstance(finding_payload["limits"], dict) and finding_payload["limits"]
    assert "uncertainty_note" in finding_payload
    assert data["metrics"]["policy_version"] == "qc-policy/2.0"
    assert data["metrics"]["profile"]["profile_id"] == "arm6_joint_position_v1"

    v1_response = _api(client, episode.id, viewer_headers, policy_version="qc-policy/1.0")
    assert v1_response.status_code == 200, v1_response.text
    v1_data = v1_response.json()["data"]
    assert v1_data["policy_version"] == "qc-policy/1.0"
    assert v1_data["canonical_version"] == 1  # historical binding, not the current 2
    assert v1_data["schema_version"] == "quality-report/1.0"
    assert v1_data["verdict"] == "WARN"
    assert v1_data["report_id"] == report1.id
    assert v1_data["run_id"] == run1.id
    assert len(v1_data["findings"]) == 1
    v1_payload = v1_data["findings"][0]
    assert v1_payload["rule_code"] == "FRAME_GAP_EXCESS"
    assert v1_payload["source_stream"] is None
    assert v1_payload["source_clock_domain"] is None
    assert v1_payload["method"] is None
    assert v1_payload["limits"] == {}
    assert v1_payload["uncertainty_note"] is None
    assert v1_payload["observed"] and v1_payload["threshold"] and v1_payload["evidence"]

    v2_explicit = _api(client, episode.id, viewer_headers, policy_version="qc-policy/2.0")
    assert v2_explicit.status_code == 200, v2_explicit.text
    assert v2_explicit.json()["data"]["report_id"] == report2.id

    invalid = _api(client, episode.id, viewer_headers, policy_version="qc-policy/9.9")
    assert invalid.status_code == 422, invalid.text
    assert invalid.json()["error"]["code"] == "VALIDATION_FAILED"
    assert "qc-policy/1.0" in invalid.json()["error"]["details"]["allowed"]

    # the v1 report row is still untouched after the API reads
    assert _report_snapshot(db.get(QualityReport, report1.id)) == snapshot_before
    assert storage.get_bytes(asset1_after.object_key) == v1_report_bytes


# ---------------------------------------------------------------------------
# DO2-13 default selection: current canonical v2 without a v2 report is 404
# ---------------------------------------------------------------------------


def test_default_quality_report_is_404_when_current_canonical_has_no_report(
    client, viewer_headers, db
) -> None:
    episode = new_episode(
        db,
        canonical_schema_version="2.0",
        canonical_version=1,
        profile_id="arm6_joint_position_v1",
        profile_revision=1,
        data_origin="synthetic",
    )
    v1_report = add_quality_report(
        db, episode, status="PASS", policy_version="qc-policy/1.0", rule_codes=()
    )
    db.commit()
    episode_id = episode.id
    v1_report_id = v1_report.id

    default = _api(client, episode_id, viewer_headers)
    assert default.status_code == 404, default.text
    details = default.json()["error"]["details"]
    assert details["canonical_version"] == 1
    assert details["canonical_schema_version"] == "2.0"
    assert "canonical" in details["reason"]

    historical = _api(client, episode_id, viewer_headers, policy_version="qc-policy/1.0")
    assert historical.status_code == 200, historical.text
    data = historical.json()["data"]
    assert data["report_id"] == v1_report_id
    assert data["policy_version"] == "qc-policy/1.0"
    assert data["canonical_version"] == 1

    missing = _api(client, new_id(), viewer_headers)
    assert missing.status_code == 404, missing.text
    assert missing.json()["error"]["code"] == "NOT_FOUND"
