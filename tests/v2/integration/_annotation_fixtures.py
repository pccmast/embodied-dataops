"""Shared fixtures for the V2 annotation revision tests (DO2-14..17).

Not collected by pytest (leading underscore).  Every helper writes *real* rows
and *real* objects through the local object store; nothing is faked at the API
or service boundary.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.api.auth import Principal, Role, generate_api_key
from embodied_dataops.canonical.contract import annotation_key, canonical_json_bytes
from embodied_dataops.db.models import (
    AnnotationSet,
    ApiKey,
    Asset,
    Episode,
    PipelineRun,
    QualityReport,
    SourceImport,
)
from embodied_dataops.ids import new_id
from embodied_dataops.processing.detection import FAKE_WEIGHTS_SHA256
from embodied_dataops.workflow.outbox import utcnow

#: One detection in frame_index 0 (a cup).
FRAME_0_DETECTIONS: list[dict[str, Any]] = [
    {"class_id": 41, "class_name": "cup", "score": 0.9, "xyxy": [10.0, 10.0, 120.0, 90.0]}
]
#: Two detections in frame_index 10, already in detector ordering (class_id, -score).
FRAME_10_DETECTIONS: list[dict[str, Any]] = [
    {"class_id": 0, "class_name": "person", "score": 0.5, "xyxy": [200.0, 50.0, 320.0, 400.0]},
    {"class_id": 41, "class_name": "cup", "score": 0.75, "xyxy": [60.0, 200.0, 180.0, 300.0]},
]

#: Deterministic detection ids the baseline conversion assigns to the model output.
BASE_FRAME_0_IDS = ["model-0-000"]
BASE_FRAME_10_IDS = ["model-10-000", "model-10-001"]


def auth_headers(raw_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {raw_key}"}


def make_principal(
    db, role: str = "reviewer", *, name: str | None = None
) -> tuple[Principal, str]:
    """Create one enabled API key row and return (Principal, plaintext key)."""
    role_enum = Role(role)
    raw, record = generate_api_key(role_enum, name=name or f"test-{role}")
    db.add(record)
    db.flush()
    return Principal(key_id=record.id, role=role_enum, name=record.name), raw


def key_id_for(db, raw_key: str) -> str:
    from embodied_dataops.api.auth import hash_api_key

    return db.scalars(select(ApiKey).where(ApiKey.key_hash == hash_api_key(raw_key))).one().id


def seed_episode(
    db,
    *,
    quality_status: str = "PASS",
    canonical_schema_version: str = "2.0",
    episode_type: str = "robot_episode",
    sample_count: int = 100,
    task_label: str = "pick up the transparent cup",
    license_value: str = "CC-BY-4.0",
    source_type: str = "LEROBOT_V3",
) -> tuple[Episode, PipelineRun, QualityReport]:
    """Episode + succeeded run + quality report (no canonical objects unless added)."""
    source = SourceImport(
        id=new_id(),
        source_type=source_type,
        upload_state="READY",
        data_origin="synthetic",
        license=license_value,
        filename="fixture",
        size_bytes=0,
        sha256="a" * 64,
        mapping_json={},
        created_at=utcnow(),
    )
    db.add(source)
    db.flush()
    episode = Episode(
        id=new_id(),
        source_import_id=source.id,
        source_episode_index=0,
        episode_type=episode_type,
        task_label=task_label,
        duration_ns=sample_count * SAMPLE_INTERVAL_NS,
        status="READY_FOR_REVIEW",
        canonical_version=1,
        canonical_sha256="b" * 64,
        quality_status=quality_status,
        sample_count=sample_count if episode_type == "robot_episode" else 0,
        data_origin="synthetic",
        canonical_schema_version=canonical_schema_version,
        created_at=utcnow(),
    )
    db.add(episode)
    db.flush()
    run = PipelineRun(
        id=new_id(),
        episode_id=episode.id,
        pipeline_version="2.0.0",
        state="SUCCEEDED",
        created_at=utcnow(),
    )
    db.add(run)
    db.flush()
    report = QualityReport(
        id=new_id(),
        episode_id=episode.id,
        run_id=run.id,
        policy_version="qc-policy/2.0",
        status=quality_status,
        summary_json={"verdict": quality_status, "findings": []},
        created_at=utcnow(),
    )
    db.add(report)
    db.flush()
    return episode, run, report


def base_annotation_frames() -> list[dict[str, Any]]:
    """Two sampled frames of an ``annotation/1.0`` artifact (frame_index 0 and 10)."""
    return [
        {"frame_index": 0, "t_ns": 0, "detections": [dict(item) for item in FRAME_0_DETECTIONS]},
        {
            "frame_index": 10,
            "t_ns": 10 * SAMPLE_INTERVAL_NS,
            "detections": [dict(item) for item in FRAME_10_DETECTIONS],
        },
    ]


def seed_annotation_set(
    db,
    storage,
    episode: Episode,
    run: PipelineRun,
    *,
    frames: list[dict[str, Any]] | None = None,
    status: str = "PENDING_REVIEW",
    tool: str = "fake",
) -> tuple[AnnotationSet, bytes]:
    """Store a real ``annotation/1.0`` artifact and its AnnotationSet row."""
    document = {
        "schema_version": "annotation/1.0",
        "episode_id": episode.id,
        "model": {
            "name": tool,
            "weights_sha256": FAKE_WEIGHTS_SHA256,
            "conf": 0.35,
            "iou": 0.5,
        },
        "tool": tool,
        "is_synthetic": tool == "fake",
        "params": {"imgsz": 640, "conf": 0.35, "iou": 0.5, "max_det": 100, "device": "cpu"},
        "frames": frames if frames is not None else base_annotation_frames(),
        "stats": {},
    }
    payload = canonical_json_bytes(document)
    # The real annotation key embeds the set id; create the set id first.
    annotation_set_id = new_id()
    key = annotation_key(episode.id, annotation_set_id)
    stored = storage.put_bytes(key, payload, "application/json")
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256,
        byte_size=len(payload),
        media_type="application/json",
        object_key=key,
        origin="derived",
        logical_name="annotation_candidates",
        meta_json={"tool": tool},
        created_at=utcnow(),
    )
    db.add(asset)
    db.flush()
    annotation_set = AnnotationSet(
        id=annotation_set_id,
        episode_id=episode.id,
        run_id=run.id,
        model_id=tool,
        model_sha256=FAKE_WEIGHTS_SHA256,
        status=status,
        artifact_asset_id=asset.id,
        summary_json={"frames": len(document["frames"])},
        created_at=utcnow(),
    )
    db.add(annotation_set)
    db.flush()
    return annotation_set, payload


def revision_frame(
    frame_index: int,
    *,
    t_ns: int | None = None,
    detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "t_ns": int(frame_index) * SAMPLE_INTERVAL_NS if t_ns is None else int(t_ns),
        "detections": [dict(item) for item in (detections or [])],
    }


def detection(
    detection_id: str,
    *,
    class_id: int = 41,
    class_name: str = "cup",
    score: float | None = 0.9,
    xyxy: list[float] | None = None,
    origin: str = "HUMAN",
) -> dict[str, Any]:
    return {
        "detection_id": detection_id,
        "class_id": class_id,
        "class_name": class_name,
        "score": score,
        "xyxy": list(xyxy or [10.0, 10.0, 120.0, 90.0]),
        "origin": origin,
    }


def storage_bytes(storage, episode_id: str, annotation_set_id: str) -> bytes:
    return storage.get_bytes(annotation_key(episode_id, annotation_set_id))


def load_json(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("utf-8"))
