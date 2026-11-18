"""Integration tests for the QUALITY_CHECK handler and report persistence.

Covers the quality halves of AT-05 / AT-06 (SPEC 6.1, 7.1, 4.3):

* AT-06 bad timestamps: an ERROR finding is stored with evidence, the verdict is
  ``FAIL`` and the data is not silently repaired;
* AT-06 bad dimensions: a structurally incompatible (5D action) Episode still
  produces a complete report even though ``NORMALIZE_EPISODE`` emitted
  ``normalization_valid=false`` and no publishable ``samples.parquet``;
* AT-05 ``F06_mcap_gap``: ``SYNC_MISSING_STATE`` is persisted and the Episode is
  ``QUALITY_FAILED`` with no fabricated samples;
* ``PASS`` / ``WARN`` reports keep the Episode in ``PROCESSING``;
* re-execution of the same run is idempotent (one report, one asset, no
  duplicated findings);
* ``quality.json`` lives at ``quality/{episode_id}/{run_id}/quality.json`` and is
  bound to ``qc-policy/1.0``.

PostgreSQL is the only task truth source (CONTRIBUTING.md); these tests use the real
``db``/``storage`` fixtures from ``tests/conftest.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fixtures.make_mcap_ros2 import generate_fixtures
from sqlalchemy import func, select

from embodied_dataops.canonical.contract import SampleRow, canonical_keys, write_samples_parquet
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import (
    Asset,
    Episode,
    PipelineRun,
    QualityFinding,
    QualityReport,
    SourceImport,
    Task,
)
from embodied_dataops.domain.enums import DataOrigin, SourceType
from embodied_dataops.ids import new_id, new_token
from embodied_dataops.imports.base import ImportContext
from embodied_dataops.imports.mcap_profile_v1 import McapProfileV1Parser
from embodied_dataops.quality.report import QUALITY_REPORT_SCHEMA_VERSION, quality_report_key
from embodied_dataops.storage.base import LocalStorage
from embodied_dataops.workflow.handlers.base import HandlerContext
from embodied_dataops.workflow.handlers.quality_check import QualityCheckHandler

INTERVAL_NS = 100_000_000
JOINTS = [f"joint_{index}" for index in range(1, 7)]
ORIGIN_NS = 1_700_000_000_000_000_000


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gap_mcap_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("quality-mcap-gap")
    generate_fixtures(directory, update_manifest=False, fixture_ids=["F06_mcap_gap"])
    return directory


def _create_episode(
    db,
    *,
    source_type: str = "MCAP_ROS2_PROFILE_V1",
    episode_type: str = "robot_episode",
    duration_ns: int = 10_000_000_000,
) -> tuple[SourceImport, Episode, PipelineRun]:
    source_import = SourceImport(
        id=new_id(),
        source_type=source_type,
        upload_state="READY",
        data_origin="synthetic",
        license="CC0-1.0",
        filename="fixture.mcap",
        size_bytes=1024,
        sha256="0" * 64,
    )
    db.add(source_import)
    db.flush()
    episode = Episode(
        id=new_id(),
        source_import_id=source_import.id,
        source_episode_index=0,
        episode_type=episode_type,
        task_label="pick the cup",
        duration_ns=duration_ns,
        status="PROCESSING",
        canonical_version=1,
        data_origin="synthetic",
    )
    db.add(episode)
    db.flush()
    run = PipelineRun(
        id=new_id(),
        episode_id=episode.id,
        pipeline_version="1.0.0",
        state="RUNNING",
    )
    db.add(run)
    db.flush()
    return source_import, episode, run


def _normalize_task(db, run: PipelineRun, output: dict) -> Task:
    task = Task(
        id=new_id(),
        run_id=run.id,
        step="NORMALIZE_EPISODE",
        pool="cpu",
        state="SUCCEEDED",
        attempt=1,
        max_attempts=3,
        output_json=output,
    )
    db.add(task)
    db.flush()
    return task


def _quality_task(db, run: PipelineRun) -> Task:
    task = Task(
        id=new_id(),
        run_id=run.id,
        step="QUALITY_CHECK",
        pool="cpu",
        state="RUNNING",
        attempt=1,
        max_attempts=3,
        worker_id="test-worker",
        lease_token=new_token(),
    )
    db.add(task)
    db.flush()
    return task


def _handler_context(db, task: Task, storage: LocalStorage, workspace: Path) -> HandlerContext:
    return HandlerContext(
        session=db,
        task=task,
        settings=get_settings(),
        storage=storage,
        workspace=workspace,
        cancel_check=lambda: False,
    )


def _robot_rows(count: int = 100) -> list[SampleRow]:
    rows: list[SampleRow] = []
    for index in range(count):
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


def _canonical_metadata(episode: Episode, **overrides) -> dict:
    metadata = {
        "schema_version": "canonical-episode/1.0",
        "episode_id": episode.id,
        "episode_type": episode.episode_type,
        "data_origin": "synthetic",
        "source_type": "MCAP_ROS2_PROFILE_V1",
        "source_asset_sha256": "0" * 64,
        "source_episode_index": 0,
        "duration_ns": episode.duration_ns,
        "target_fps": 10,
        "camera_name": "front",
        "clock_domain": "mcap_log_time_episode_relative",
        "joint_names": list(JOINTS),
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
    metadata.update(overrides)
    return metadata


def _write_canonical(
    storage: LocalStorage,
    episode: Episode,
    workspace: Path,
    *,
    metadata: dict | None = None,
    rows: list[SampleRow] | None = None,
) -> dict[str, str]:
    keys = canonical_keys(episode.id, episode.canonical_version)
    storage.put_bytes(
        keys["metadata"],
        json.dumps(metadata if metadata is not None else _canonical_metadata(episode)).encode("utf-8"),
        content_type="application/json",
    )
    if rows:
        parquet_path = workspace / "samples.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        write_samples_parquet(parquet_path, rows, episode.episode_type)
        storage.put_file(keys["samples"], parquet_path, content_type="application/vnd.apache.parquet")
    return keys


def _normalize_output(
    keys: dict[str, str],
    episode: Episode,
    *,
    normalization_valid: bool = True,
    state_dim: int = 6,
    action_dim: int = 6,
    joint_names: list[str] | None = None,
    alignment: dict | None = None,
    diagnostics: list[dict] | None = None,
    stream_times: dict[str, list[int]] | None = None,
) -> dict:
    base_alignment = {
        "grid_points": 100,
        "missing_sample_count": 0,
        "max_abs_sync_error_ms": 0.0,
        "p95_abs_sync_error_ms": 0.0,
        "missing_frame_indexes": [],
        "missing_state_indexes": [],
        "stale_action_indexes": [],
        "origin_ns": ORIGIN_NS,
        "grid": [index * INTERVAL_NS for index in range(100)],
    }
    if alignment:
        base_alignment.update(alignment)
    output = {
        "normalization_valid": normalization_valid,
        "episode_type": episode.episode_type,
        "sample_count": 100,
        "duration_ns": episode.duration_ns,
        "canonical": {
            "metadata_key": keys["metadata"],
            "samples_key": keys["samples"],
            "video_key": keys["video"],
        },
        "alignment": base_alignment,
        "diagnostics": diagnostics or [],
        "clock_domains": {"camera": "mcap_log_time", "state": "mcap_log_time", "action": "mcap_log_time"},
        "state_dim": state_dim,
        "action_dim": action_dim,
        "joint_names": joint_names if joint_names is not None else list(JOINTS),
        "semantics_verified": True,
    }
    if stream_times is not None:
        output["stream_times"] = stream_times
    return output


def _camera_times_10hz(count: int = 100) -> list[int]:
    return [ORIGIN_NS + index * INTERVAL_NS for index in range(count)]


def _run_handler(db, task: Task, storage: LocalStorage, workspace: Path):
    return QualityCheckHandler().handle(_handler_context(db, task, storage, workspace))


def _report_for(db, episode: Episode, run: PipelineRun) -> QualityReport:
    return db.execute(
        select(QualityReport).where(
            QualityReport.episode_id == episode.id, QualityReport.run_id == run.id
        )
    ).scalar_one()


# ---------------------------------------------------------------------------
# AT-06: bad timestamps
# ---------------------------------------------------------------------------


def test_at_06_bad_timestamps_persist_error_finding_and_fail_verdict(db, storage, tmp_path: Path) -> None:
    _, episode, run = _create_episode(db)
    keys = _write_canonical(storage, episode, tmp_path, rows=_robot_rows())
    # duplicate source timestamp at index 50 (t[50] == t[49]) -> non-monotonic
    state_times = [ORIGIN_NS + index * INTERVAL_NS for index in range(100)]
    state_times[50] = state_times[49]
    output = _normalize_output(
        keys,
        episode,
        alignment={"grid": [index * INTERVAL_NS for index in range(100)]},
        stream_times={"camera": _camera_times_10hz(), "state": state_times},
    )
    _normalize_task(db, run, output)
    task = _quality_task(db, run)

    result = _run_handler(db, task, storage, tmp_path)

    assert result.verdict == "FAIL"
    # the Episode status transition is applied by the report persistence, so the
    # handler does not request a duplicate (illegal) transition from the engine
    assert result.episode_status is None
    assert result.output_asset_id
    assert result.data["quality_report_key"] == quality_report_key(episode.id, run.id)

    report = _report_for(db, episode, run)
    assert report.status == "FAIL"
    assert report.policy_version == "qc-policy/1.0"
    assert report.report_asset_id == result.output_asset_id
    assert report.summary_json["finding_counts"]["ERROR"] == 1

    findings = list(db.execute(select(QualityFinding).where(QualityFinding.report_id == report.id)).scalars())
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_code == "TIMESTAMP_NON_MONOTONIC"
    assert finding.severity == "ERROR"
    assert finding.observed_json["streams"]["state"]["violation_count"] == 1
    assert finding.evidence_json["affected_time_ranges"]
    assert finding.limit_json["strictly_increasing"] is True

    assert episode.quality_status == "FAIL"
    assert episode.status == "QUALITY_FAILED"

    # the report object itself is stored and readable
    report_key = quality_report_key(episode.id, run.id)
    assert storage.exists(report_key)
    payload = json.loads(storage.get_bytes(report_key).decode("utf-8"))
    assert payload["schema_version"] == QUALITY_REPORT_SCHEMA_VERSION
    assert payload["status"] == "FAIL"
    assert payload["policy_version"] == "qc-policy/1.0"
    assert payload["findings"][0]["rule_code"] == "TIMESTAMP_NON_MONOTONIC"
    assert payload["metrics"]["valid_sample_count"] == 100
    assert payload["metrics"]["missing_sample_count"] == 0

    # nothing was silently repaired: the canonical samples object is unchanged
    samples_table = pq.read_table(tmp_path / "samples.parquet")
    assert samples_table.num_rows == 100
    assert payload["source"]["normalization_valid"] is True


def test_at_06_bad_dimensions_still_produce_a_complete_report(db, storage, tmp_path: Path) -> None:
    _, episode, run = _create_episode(db)
    # structural incompatibility: normalisation produced metadata + diagnostics only
    metadata = _canonical_metadata(
        episode,
        normalization_valid=False,
        action_dim=5,
        sample_count=0,
        video_available=False,
    )
    keys = _write_canonical(storage, episode, tmp_path, metadata=metadata, rows=None)
    output = _normalize_output(
        keys,
        episode,
        normalization_valid=False,
        action_dim=5,
        alignment={"grid_points": 100, "grid": [], "origin_ns": ORIGIN_NS},
        diagnostics=[
            {
                "code": "DIMENSION_MISMATCH",
                "severity": "ERROR",
                "message": "action has dimension 5, arm6_joint_position_v1 requires 6",
                "details": {"action_dim": 5},
            }
        ],
    )
    _normalize_task(db, run, output)
    task = _quality_task(db, run)

    result = _run_handler(db, task, storage, tmp_path)

    assert result.verdict == "FAIL"
    report = _report_for(db, episode, run)
    assert report.status == "FAIL"
    findings = list(db.execute(select(QualityFinding).where(QualityFinding.report_id == report.id)).scalars())
    assert [finding.rule_code for finding in findings] == ["DIMENSION_MISMATCH"]
    assert findings[0].observed_json["observed_action_dim"] == 5
    assert findings[0].evidence_json["no_truncation_or_padding"] is True
    assert storage.exists(quality_report_key(episode.id, run.id))
    assert episode.quality_status == "FAIL"
    assert episode.status == "QUALITY_FAILED"


# ---------------------------------------------------------------------------
# AT-05: F06 gap through the parser diagnostics
# ---------------------------------------------------------------------------


def test_at_05_gap_report_persists_sync_missing_state(db, storage, tmp_path: Path, gap_mcap_dir: Path) -> None:
    context = ImportContext(
        import_id="import-gap",
        source_type=SourceType.MCAP_ROS2_PROFILE_V1,
        data_origin=DataOrigin.SYNTHETIC,
        source_path=gap_mcap_dir / "F06_mcap_gap.mcap",
        workdir=tmp_path,
        mapping={},
        license="CC0-1.0",
        source_sha256="0" * 64,
    )
    parsed = McapProfileV1Parser(backend="fallback").parse(context)
    episode_payload = parsed.episodes[0]
    payload = episode_payload.payload

    _, episode, run = _create_episode(db)
    metadata = _canonical_metadata(
        episode,
        normalization_valid=False,
        sample_count=89,
        state_dim=payload["state_dim"],
        action_dim=payload["action_dim"],
        joint_names=payload["joint_names"],
        video_available=False,
    )
    keys = _write_canonical(storage, episode, tmp_path, metadata=metadata, rows=None)
    output = _normalize_output(
        keys,
        episode,
        normalization_valid=False,
        state_dim=payload["state_dim"],
        action_dim=payload["action_dim"],
        joint_names=payload["joint_names"],
        alignment={**payload["alignment"], "grid": payload["grid"]},
        diagnostics=payload["diagnostics"],
        stream_times=payload["stream_times"],
    )
    _normalize_task(db, run, output)
    task = _quality_task(db, run)

    result = _run_handler(db, task, storage, tmp_path)

    assert result.verdict == "FAIL"
    assert episode.quality_status == "FAIL"
    assert episode.status == "QUALITY_FAILED"
    report = _report_for(db, episode, run)
    assert report.status == "FAIL"
    findings = list(db.execute(select(QualityFinding).where(QualityFinding.report_id == report.id)).scalars())
    by_rule = {finding.rule_code: finding for finding in findings}
    assert "SYNC_MISSING_STATE" in by_rule
    sync_finding = by_rule["SYNC_MISSING_STATE"]
    assert sync_finding.severity == "ERROR"
    assert sync_finding.observed_json["missing_grid_point_count"] == 11
    assert sync_finding.evidence_json["no_zero_fill"] is True
    assert report.summary_json["metrics"]["valid_sample_count"] == 89
    assert report.summary_json["metrics"]["missing_sample_count"] == 11
    assert report.summary_json["normalization_valid"] is False
    # no canonical samples object was written for a structurally invalid episode
    assert not storage.exists(keys["samples"])


# ---------------------------------------------------------------------------
# PASS / WARN persistence
# ---------------------------------------------------------------------------


def test_pass_report_keeps_episode_processing(db, storage, tmp_path: Path) -> None:
    _, episode, run = _create_episode(db)
    keys = _write_canonical(storage, episode, tmp_path, rows=_robot_rows())
    stream_times = {
        "camera": [ORIGIN_NS + index * 33_333_333 for index in range(300)],
        "state": [ORIGIN_NS + index * 10_000_000 for index in range(1000)],
        "action": [ORIGIN_NS + index * 20_000_000 for index in range(500)],
    }
    _normalize_task(db, run, _normalize_output(keys, episode, stream_times=stream_times))
    task = _quality_task(db, run)

    result = _run_handler(db, task, storage, tmp_path)

    assert result.verdict == "PASS"
    assert result.episode_status is None
    report = _report_for(db, episode, run)
    assert report.status == "PASS"
    assert db.execute(
        select(func.count()).select_from(QualityFinding).where(QualityFinding.report_id == report.id)
    ).scalar_one() == 0
    assert episode.quality_status == "PASS"
    assert episode.status == "PROCESSING"
    assert report.summary_json["metrics"]["valid_sample_count"] == 100
    assert report.summary_json["metrics"]["missing_sample_count"] == 0
    payload = json.loads(storage.get_bytes(quality_report_key(episode.id, run.id)).decode("utf-8"))
    assert payload["status"] == "PASS"
    assert payload["findings"] == []
    assert "valid_sample_count" in payload["metrics"]
    assert "max_abs_sync_error_ms" in payload["metrics"]
    assert "p95_abs_sync_error_ms" in payload["metrics"]


def test_warn_report_keeps_episode_processing(db, storage, tmp_path: Path) -> None:
    _, episode, run = _create_episode(db)
    keys = _write_canonical(storage, episode, tmp_path, rows=_robot_rows())
    # a 250 ms source camera gap is > 200 ms -> FRAME_GAP_EXCESS WARN (no missing
    # grid image); every later timestamp is shifted so the stream stays monotonic
    camera_times = [
        ORIGIN_NS + index * INTERVAL_NS + (150_000_000 if index >= 50 else 0) for index in range(100)
    ]
    _normalize_task(
        db,
        run,
        _normalize_output(keys, episode, stream_times={"camera": camera_times}),
    )
    task = _quality_task(db, run)

    result = _run_handler(db, task, storage, tmp_path)

    assert result.verdict == "WARN"
    assert result.episode_status is None
    report = _report_for(db, episode, run)
    assert report.status == "WARN"
    assert episode.quality_status == "WARN"
    assert episode.status == "PROCESSING"
    findings = list(db.execute(select(QualityFinding).where(QualityFinding.report_id == report.id)).scalars())
    assert [finding.rule_code for finding in findings] == ["FRAME_GAP_EXCESS"]
    assert findings[0].severity == "WARN"


# ---------------------------------------------------------------------------
# idempotency / policy binding
# ---------------------------------------------------------------------------


def test_repeated_execution_of_the_same_run_is_idempotent(db, storage, tmp_path: Path) -> None:
    _, episode, run = _create_episode(db)
    keys = _write_canonical(storage, episode, tmp_path, rows=_robot_rows())
    _normalize_task(db, run, _normalize_output(keys, episode, stream_times={"camera": _camera_times_10hz()}))
    task = _quality_task(db, run)

    first = _run_handler(db, task, storage, tmp_path)
    second = _run_handler(db, task, storage, tmp_path)

    assert first.data["quality_report_id"] == second.data["quality_report_id"]
    assert db.execute(
        select(func.count())
        .select_from(QualityReport)
        .where(QualityReport.episode_id == episode.id, QualityReport.run_id == run.id)
    ).scalar_one() == 1
    assert db.execute(
        select(func.count())
        .select_from(Asset)
        .where(Asset.object_key == quality_report_key(episode.id, run.id))
    ).scalar_one() == 1
    report = _report_for(db, episode, run)
    finding_count = db.execute(
        select(func.count()).select_from(QualityFinding).where(QualityFinding.report_id == report.id)
    ).scalar_one()
    assert finding_count == len(second.data["rule_codes"])


def test_report_binds_policy_version_and_metrics_contract(db, storage, tmp_path: Path) -> None:
    _, episode, run = _create_episode(db)
    keys = _write_canonical(storage, episode, tmp_path, rows=_robot_rows())
    _normalize_task(db, run, _normalize_output(keys, episode))
    task = _quality_task(db, run)

    _run_handler(db, task, storage, tmp_path)

    report = _report_for(db, episode, run)
    assert report.policy_version == "qc-policy/1.0"
    summary = report.summary_json
    assert summary["schema_version"] == QUALITY_REPORT_SCHEMA_VERSION
    assert summary["report_key"] == quality_report_key(episode.id, run.id)
    assert len(summary["report_sha256"]) == 64
    assert summary["metrics"]["policy_version"] == "qc-policy/1.0"
    for key in (
        "valid_sample_count",
        "missing_sample_count",
        "max_abs_sync_error_ms",
        "p95_abs_sync_error_ms",
    ):
        assert key in summary["metrics"]
    asset = db.get(Asset, report.report_asset_id)
    assert asset is not None
    assert asset.object_key == quality_report_key(episode.id, run.id)
    assert asset.origin == "report"
    assert asset.media_type == "application/json"
