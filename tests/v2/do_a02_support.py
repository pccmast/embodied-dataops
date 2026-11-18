"""Shared fixtures/builders for the DO-A02 (D2-M4) test suites.

Kept as a plain importable module (not a pytest plugin) so the integration and
contract suites can share realistic DB/object-store fixtures without touching the
Lead-owned ``tests/conftest.py``.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import (
    Asset,
    Dataset,
    DatasetVersion,
    Episode,
    EvaluationReport,
    Modality,
    PipelineRun,
    QualityFinding,
    QualityReport,
    SourceImport,
    Task,
)
from embodied_dataops.feedback import service as feedback_service
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.dag import create_pipeline_run
from embodied_dataops.workflow.outbox import utcnow

TEN_SECONDS_NS = 10_000_000_000


def new_source_import(
    session: Session,
    *,
    source_type: str = "LEROBOT_V3",
    data_origin: str = "synthetic",
    sha256: str | None = None,
) -> SourceImport:
    record = SourceImport(
        id=new_id(),
        source_type=source_type,
        upload_state="READY",
        data_origin=data_origin,
        license="MIT",
        filename="v2-fixture.zip",
        size_bytes=1024,
        sha256=sha256,
    )
    session.add(record)
    session.flush()
    return record


def new_episode(
    session: Session,
    *,
    index: int = 0,
    source_type: str = "LEROBOT_V3",
    data_origin: str = "synthetic",
    duration_ns: int = TEN_SECONDS_NS,
    status: str = "IMPORTED",
    profile_id: str | None = "arm6_joint_position_v1",
    profile_revision: int | None = 1,
    canonical_schema_version: str = "2.0",
    canonical_version: int = 1,
) -> Episode:
    source = new_source_import(session, source_type=source_type, data_origin=data_origin)
    episode = Episode(
        id=new_id(),
        source_import_id=source.id,
        source_episode_index=index,
        episode_type="robot_episode",
        task_label="pick-and-place",
        duration_ns=duration_ns,
        status=status,
        canonical_version=canonical_version,
        canonical_schema_version=canonical_schema_version,
        canonical_sha256="c" * 64,
        profile_id=profile_id,
        profile_revision=profile_revision,
        data_origin=data_origin,
    )
    session.add(episode)
    session.flush()
    return episode


def new_run_with_tasks(session: Session, episode: Episode, *, commit: bool = True) -> tuple[PipelineRun, dict[str, Task]]:
    run = create_pipeline_run(session, episode)
    if commit:
        session.commit()
    tasks = {task.step: task for task in session.scalars(select(Task).where(Task.run_id == run.id))}
    return run, tasks


def new_failed_task(
    session: Session,
    *,
    step: str = "NORMALIZE_EPISODE",
    error_code: str = "OBJECT_STORE_UNAVAILABLE",
    attempt: int = 3,
    status: str = "FAILED",
) -> tuple[Episode, PipelineRun, Task]:
    """A pipeline run whose root task is (by default) FAILED but retryable."""
    episode = new_episode(session)
    run, tasks = new_run_with_tasks(session, episode, commit=False)
    task = tasks[step]
    task.state = status
    task.attempt = attempt
    task.error_code = error_code
    task.updated_at = utcnow()
    session.commit()
    return episode, run, task


def new_dataset_version(session: Session) -> tuple[Dataset, DatasetVersion]:
    dataset = Dataset(id=new_id(), slug=f"ds-{new_id()[:8]}", description="do-a02 fixture")
    session.add(dataset)
    session.flush()
    version = DatasetVersion(
        id=new_id(),
        dataset_id=dataset.id,
        version=1,
        state="READY",
        description="fixture",
    )
    session.add(version)
    session.flush()
    return dataset, version


def new_evaluation(
    session: Session,
    *,
    results: list[dict] | None = None,
    is_simulated: bool = True,
    model_id: str = "sim-policy-v1",
    dataset_version: DatasetVersion | None = None,
) -> EvaluationReport:
    if dataset_version is None:
        _, dataset_version = new_dataset_version(session)
    report = feedback_service.create_evaluation(
        session,
        dataset_version_id=dataset_version.id,
        payload={
            "schema_version": "evaluation/1.0",
            "model_id": model_id,
            "is_simulated": is_simulated,
            "results": results
            or [
                {"scenario_key": "opaque_cup_normal_light", "trials": 30, "successes": 27},
                {"scenario_key": "transparent_cup_low_light", "trials": 30, "successes": 12},
            ],
        },
        actor=None,
        request_id="fixture",
    )
    session.commit()
    return report


def add_quality_report(
    session: Session,
    episode: Episode,
    *,
    status: str,
    policy_version: str = "qc-policy/1.0",
    rule_codes: tuple[str, ...] = (),
    created_at: datetime | None = None,
    profile_id: str | None = None,
    profile_revision: int | None = None,
) -> QualityReport:
    run = create_pipeline_run(session, episode)
    report = QualityReport(
        id=new_id(),
        episode_id=episode.id,
        run_id=run.id,
        policy_version=policy_version,
        status=status,
        summary_json={"verdict": status},
        profile_id=profile_id if profile_id is not None else episode.profile_id,
        profile_revision=profile_revision if profile_revision is not None else episode.profile_revision,
        created_at=created_at or utcnow(),
    )
    session.add(report)
    session.flush()
    for rule_code in rule_codes:
        session.add(
            QualityFinding(
                id=new_id(),
                report_id=report.id,
                rule_code=rule_code,
                severity="ERROR" if status == "FAIL" else "WARN",
                observed_json={"value": 1},
                limit_json={"max": 0},
                evidence_json={},
            )
        )
    session.flush()
    return report


def write_stream_asset(
    session: Session,
    storage,
    episode: Episode,
    *,
    name: str,
    kind: str = "camera",
    clock_domain: str = "device_monotonic",
    times_ns: list[int] | None = None,
    source_times_ns: list[int] | None = None,
    valid_flags: list[bool] | None = None,
    stream_index: int = 0,
    interval_ns: int = 100_000_000,
) -> tuple[Asset, Modality]:
    """Write a native timestamp parquet and link it to a Modality row."""
    if times_ns is None:
        times_ns = [i * interval_ns for i in range(100)]
    if source_times_ns is None:
        source_times_ns = [value + 1_000_000 for value in times_ns]
    if valid_flags is None:
        valid_flags = [True] * len(times_ns)
    table = pa.table(
        {
            "t_ns": pa.array(times_ns, pa.int64()),
            "source_t_ns": pa.array(source_times_ns, pa.int64()),
            "valid": pa.array(valid_flags, pa.bool_()),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    payload = buffer.getvalue()
    key = f"canonical/{episode.id}/v2/{episode.canonical_version}/native/{name}.parquet"
    stored = storage.put_bytes(key, payload, content_type="application/vnd.apache.parquet")
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or "",
        byte_size=len(payload),
        media_type="application/vnd.apache.parquet",
        object_key=key,
        origin="canonical",
        logical_name=f"{name}.parquet",
        meta_json={"kind": "native_index"},
    )
    session.add(asset)
    session.flush()
    modality = Modality(
        id=new_id(),
        episode_id=episode.id,
        name=name,
        kind=kind,
        dtype="float32",
        shape_json=[6],
        clock_domain=clock_domain,
        sample_count=len(times_ns),
        min_t_ns=min(times_ns) if times_ns else None,
        max_t_ns=max(times_ns) if times_ns else None,
        source_key="fixture",
        stream_index=stream_index,
        native_rate_hz=10.0,
        native_index_asset_id=asset.id,
        timestamp_provenance="source_timestamps",
        clock_verified=True,
    )
    session.add(modality)
    session.flush()
    return asset, modality


def write_aligned_samples(session: Session, storage, episode: Episode, *, rows: int = 100) -> Asset:
    """Write a canonical 2.0 aligned preview-ish parquet for the timeline endpoint."""
    table = pa.table(
        {
            "t_ns": pa.array([i * 100_000_000 for i in range(rows)], pa.int64()),
            "source_t_ns": pa.array([i * 100_000_000 + 2_000_000 for i in range(rows)], pa.int64()),
            "valid": pa.array([True] * rows, pa.bool_()),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    payload = buffer.getvalue()
    key = f"canonical/{episode.id}/v2/{episode.canonical_version}/aligned/10hz/samples.parquet"
    stored = storage.put_bytes(key, payload, content_type="application/vnd.apache.parquet")
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or "",
        byte_size=len(payload),
        media_type="application/vnd.apache.parquet",
        object_key=key,
        origin="canonical",
        logical_name="samples.parquet",
        meta_json={"kind": "aligned"},
    )
    session.add(asset)
    session.flush()
    return asset


def etag(kind: str, revision: int) -> str:
    return f'"{kind}:{int(revision)}"'


def episode_etag(episode: Episode) -> str:
    return etag("episode-rev", int(episode.business_revision or 1))


def task_etag(task: Task) -> str:
    return etag("task-rev", int(task.business_revision or 1))


def evaluation_etag(report: EvaluationReport) -> str:
    return etag("evaluation-rev", int(report.business_revision or 1))


def write_canonical_metadata(
    session: Session,
    storage,
    episode: Episode,
    *,
    schema_version: str = "canonical-episode/2.0",
    extra: dict | None = None,
) -> Asset:
    import json

    version = int(episode.canonical_version or 1)
    key = f"canonical/{episode.id}/v2/{version}/metadata.json"
    payload = {
        "schema_version": schema_version,
        "episode_id": episode.id,
        "episode_type": episode.episode_type,
        "data_origin": episode.data_origin,
        "source_type": "LEROBOT_V3",
        "source_asset_sha256": "d" * 64,
        "source_episode_index": episode.source_episode_index,
        "duration_ns": episode.duration_ns,
        "target_fps": 10,
        "camera_name": "front",
        "clock_domain": "device_monotonic",
        "profile_id": episode.profile_id,
        "profile_revision": episode.profile_revision,
        "pipeline_version": "2.0.0",
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload.update(extra or {})
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    stored = storage.put_bytes(key, body, content_type="application/json")
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or "",
        byte_size=len(body),
        media_type="application/json",
        object_key=key,
        origin="canonical",
        logical_name="metadata.json",
        meta_json={"kind": "canonical_metadata"},
    )
    session.add(asset)
    session.flush()
    return asset


def epoch_now() -> datetime:
    return datetime.now(UTC)


def days_ago(days: int) -> datetime:
    return epoch_now() - timedelta(days=days)
