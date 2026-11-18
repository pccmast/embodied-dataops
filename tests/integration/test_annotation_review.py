"""Annotation review API and M4 handler integration tests (SPEC 7.2, 8.1, 9.4).

Covers:

* ``GET /api/v1/episodes/{id}/annotations`` -- latest set, preview URLs, envelope;
* ``POST /api/v1/annotation-sets/{id}/reviews`` -- reviewer RBAC (operator 403),
  PENDING_REVIEW -> APPROVED/REJECTED one-shot state machine, duplicate 409,
  WARN note >= 10 chars rule, immutable raw annotation JSON, audit trail;
* ``DETECT_OBJECTS`` + ``BUILD_PREVIEW`` handlers with the clearly-labelled
  FakeDetector (never reported as real inference).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import func, select

from embodied_dataops import SAMPLE_INTERVAL_NS, TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.annotations.service import MIN_WARN_RELEASE_NOTE_CHARS
from embodied_dataops.canonical.contract import (
    CanonicalMetadata,
    annotation_key,
    canonical_json_bytes,
    canonical_keys,
)
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import (
    AnnotationReview,
    AnnotationSet,
    ApiKey,
    Asset,
    AuditEvent,
    Episode,
    Modality,
    PipelineRun,
    SourceImport,
    Task,
)
from embodied_dataops.domain.enums import Step
from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.ids import new_id
from embodied_dataops.processing import detection
from embodied_dataops.processing.detection import FAKE_WEIGHTS_SHA256
from embodied_dataops.storage.base import sha256_bytes, sha256_file
from embodied_dataops.workflow.handlers.base import HandlerContext
from embodied_dataops.workflow.handlers.build_preview import BuildPreviewHandler
from embodied_dataops.workflow.handlers.detect_objects import DetectObjectsHandler
from embodied_dataops.workflow.outbox import utcnow

pytestmark = pytest.mark.integration

SPEC_DETECTIONS = [
    {"class_id": 41, "class_name": "cup", "score": 0.823456, "xyxy": [21.2, 12.5, 200.1, 300.0]}
]


# ---------------------------------------------------------------------------
# seeding helpers
# ---------------------------------------------------------------------------


def seed_episode(
    db,
    *,
    quality_status: str = "PASS",
    episode_type: str = "robot_episode",
    sample_count: int = 100,
    source_type: str = "LEROBOT_V3",
    data_origin: str = "synthetic",
) -> tuple[Episode, PipelineRun]:
    source = SourceImport(
        id=new_id(),
        source_type=source_type,
        upload_state="READY",
        data_origin=data_origin,
        license="synthetic-fixture",
        filename="fixture",
        size_bytes=0,
        created_at=utcnow(),
    )
    db.add(source)
    db.flush()
    episode = Episode(
        id=new_id(),
        source_import_id=source.id,
        source_episode_index=0,
        episode_type=episode_type,
        task_label="pick up the cup",
        duration_ns=sample_count * SAMPLE_INTERVAL_NS,
        status="READY_FOR_REVIEW",
        canonical_version=1,
        quality_status=quality_status,
        sample_count=sample_count,
        data_origin=data_origin,
        created_at=utcnow(),
    )
    db.add(episode)
    db.flush()
    run = PipelineRun(
        id=new_id(),
        episode_id=episode.id,
        pipeline_version="1.0.0",
        state="SUCCEEDED",
        created_at=utcnow(),
    )
    db.add(run)
    db.flush()
    return episode, run


def seed_media(
    storage,
    episode: Episode,
    *,
    video_bytes: bytes = b"fake-mp4-bytes-for-presign-test",
    thumbnail_bytes: bytes = b"\xff\xd8\xff\xe0fake-jpeg",
) -> None:
    keys = canonical_keys(episode.id, episode.canonical_version)
    storage.put_bytes(keys["video"], video_bytes, "video/mp4")
    storage.put_bytes(keys["thumbnail"], thumbnail_bytes, "image/jpeg")


def seed_annotation_set(
    db,
    storage,
    episode: Episode,
    run: PipelineRun,
    *,
    detections: list[dict] | None = None,
    status: str = "PENDING_REVIEW",
    created_at=None,
    tool: str = "fake",
    model_name: str = "fake",
    weights_sha256: str = FAKE_WEIGHTS_SHA256,
) -> tuple[AnnotationSet, bytes]:
    detections = SPEC_DETECTIONS if detections is None else detections
    annotation_set_id = new_id()
    document = {
        "schema_version": "annotation/1.0",
        "episode_id": episode.id,
        "model": {
            "name": model_name,
            "weights_sha256": weights_sha256,
            "conf": 0.35,
            "iou": 0.5,
        },
        "tool": tool,
        "is_synthetic": tool == "fake",
        "frames": [{"frame_index": 0, "t_ns": 0, "detections": detections}],
        "stats": {"frames": 1, "detections": len(detections), "empty_frames": 0, "low_confidence_excluded": 0},
    }
    payload = canonical_json_bytes(document)
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
        model_id=model_name,
        model_sha256=weights_sha256,
        status=status,
        artifact_asset_id=asset.id,
        summary_json={"frames": 1, "detections": len(detections)},
        created_at=created_at or utcnow(),
    )
    db.add(annotation_set)
    db.flush()
    return annotation_set, payload


def write_test_video(path: Path, *, frame_count: int = 20) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (TARGET_WIDTH, TARGET_HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder is unavailable in this environment")
    try:
        for index in range(frame_count):
            frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
            cv2.rectangle(frame, (60, 70), (260, 320), (0, 180, 240), -1)
            cv2.putText(
                frame, str(index), (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def revision_of(db, raw_key: str) -> str:
    from embodied_dataops.api.auth import hash_api_key

    return db.scalars(select(ApiKey).where(ApiKey.key_hash == hash_api_key(raw_key))).one().id


# ---------------------------------------------------------------------------
# GET /episodes/{id}/annotations
# ---------------------------------------------------------------------------


def test_get_annotations_returns_latest_set_with_preview_urls(
    client, db, storage, api_keys, viewer_headers
) -> None:
    episode, run = seed_episode(db)
    db.commit()
    seed_media(storage, episode)
    older, _ = seed_annotation_set(
        db, storage, episode, run, created_at=utcnow() - timedelta(minutes=5), detections=[]
    )
    newer, newer_payload = seed_annotation_set(db, storage, episode, run, created_at=utcnow())
    db.commit()

    response = client.get(
        f"/api/v1/episodes/{episode.id}/annotations", headers=viewer_headers
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"data", "request_id"}
    data = body["data"]
    assert data["annotation_set"]["annotation_set_id"] == newer.id
    assert data["annotation_set"]["status"] == "PENDING_REVIEW"
    assert data["annotation_set"]["tool"] == "fake"
    assert data["annotation_set"]["is_synthetic"] is True
    assert data["annotation_set"]["model"]["name"] == "fake"
    assert data["annotation_set"]["artifact_sha256"]
    assert len(data["frames"]) == 1
    frame = data["frames"][0]
    assert frame["frame_index"] == 0
    assert frame["t_ns"] == 0
    assert frame["detections"] == SPEC_DETECTIONS
    assert frame["preview_url"].startswith("http://testserver/api/v1/_storage/get/")
    assert "sig=" in frame["preview_url"] and "exp=" in frame["preview_url"]

    # Preview URLs are real presigned GETs on the object store.
    preview = client.get(frame["preview_url"])
    assert preview.status_code == 200
    assert preview.content == b"fake-mp4-bytes-for-presign-test"
    thumbnail = client.get(data["annotation_set"]["thumbnail_url"])
    assert thumbnail.status_code == 200
    assert thumbnail.content == b"\xff\xd8\xff\xe0fake-jpeg"
    # The older set was stored but is not the latest one served.
    assert data["annotation_set"]["annotation_set_id"] != older.id
    assert json.loads(newer_payload)["episode_id"] == episode.id


def test_get_annotations_requires_authentication(client, db, storage) -> None:
    episode, run = seed_episode(db)
    seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = client.get(f"/api/v1/episodes/{episode.id}/annotations")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == ErrorCode.UNAUTHENTICATED.value


def test_get_annotations_unknown_episode_is_404(client, viewer_headers) -> None:
    response = client.get(
        f"/api/v1/episodes/{new_id()}/annotations", headers=viewer_headers
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.NOT_FOUND.value


def test_get_annotations_without_annotation_set_is_404(
    client, db, viewer_headers
) -> None:
    episode, _run = seed_episode(db)
    db.commit()
    response = client.get(
        f"/api/v1/episodes/{episode.id}/annotations", headers=viewer_headers
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.NOT_FOUND.value


def test_invalid_stored_annotation_artifact_is_rejected(
    client, db, storage, viewer_headers, reviewer_headers
) -> None:
    """Model output is untrusted (SPEC 12.1): corrupt bounds are never served/approved."""
    episode, run = seed_episode(db)
    annotation_set, payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    tampered = json.loads(payload)
    tampered["frames"][0]["detections"][0]["xyxy"] = [10.0, 10.0, float(TARGET_WIDTH + 5), 100.0]
    storage.put_bytes(
        annotation_key(episode.id, annotation_set.id), canonical_json_bytes(tampered), "application/json"
    )

    listing = client.get(
        f"/api/v1/episodes/{episode.id}/annotations", headers=viewer_headers
    )
    assert listing.status_code == 500
    assert listing.json()["error"]["code"] == ErrorCode.INTERNAL_ERROR.value

    approval = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED", "note": "should be refused"},
    )
    assert approval.status_code == 500
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "PENDING_REVIEW"


# ---------------------------------------------------------------------------
# POST /annotation-sets/{id}/reviews
# ---------------------------------------------------------------------------


def test_operator_cannot_review(client, db, storage, api_keys, operator_headers) -> None:
    episode, run = seed_episode(db)
    annotation_set, payload = seed_annotation_set(db, storage, episode, run)
    db.commit()

    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=operator_headers,
        json={"decision": "APPROVED", "note": "looks good to me"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == ErrorCode.FORBIDDEN.value
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "PENDING_REVIEW"
    assert db.scalars(select(AnnotationReview)).all() == []
    # The raw annotation JSON is untouched by a forbidden request.
    assert storage.get_bytes(annotation_key(episode.id, annotation_set.id)) == payload


def test_viewer_cannot_review(client, db, storage, viewer_headers) -> None:
    episode, run = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=viewer_headers,
        json={"decision": "APPROVED"},
    )
    assert response.status_code == 403


def test_reviewer_approves_and_duplicate_review_is_409(
    client, db, storage, api_keys, reviewer_headers
) -> None:
    episode, run = seed_episode(db, quality_status="PASS")
    annotation_set, payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    artifact_key = annotation_key(episode.id, annotation_set.id)

    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED", "note": "single reviewer approval"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["data"]["new_status"] == "APPROVED"
    assert body["data"]["review_id"]
    assert body["request_id"]

    db.expire_all()
    stored = db.get(AnnotationSet, annotation_set.id)
    assert stored.status == "APPROVED"
    reviews = db.scalars(select(AnnotationReview)).all()
    assert len(reviews) == 1
    assert reviews[0].decision == "APPROVED"
    assert reviews[0].note == "single reviewer approval"
    assert reviews[0].reviewer_key_id == revision_of(db, api_keys["reviewer"])
    audit = db.scalars(
        select(AuditEvent).where(AuditEvent.action == "annotation_set.reviewed")
    ).all()
    assert len(audit) == 1
    assert audit[0].entity_id == annotation_set.id
    assert audit[0].before_json == {"status": "PENDING_REVIEW"}
    assert audit[0].after_json["status"] == "APPROVED"
    # Raw annotation JSON is immutable.
    assert storage.get_bytes(artifact_key) == payload

    duplicate = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "REJECTED", "note": "changed my mind"},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == ErrorCode.ANNOTATION_ALREADY_REVIEWED.value
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "APPROVED"
    assert len(db.scalars(select(AnnotationReview)).all()) == 1


def test_reviewer_rejects_annotation_set(
    client, db, storage, reviewer_headers
) -> None:
    episode, run = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "REJECTED", "note": "boxes are wrong"},
    )
    assert response.status_code == 201
    assert response.json()["data"]["new_status"] == "REJECTED"
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "REJECTED"


def test_warn_episode_requires_explanatory_note(
    client, db, storage, reviewer_headers
) -> None:
    episode, run = seed_episode(db, quality_status="WARN")
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()

    too_short = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED", "note": "ok"},
    )
    assert too_short.status_code == 422
    error = too_short.json()["error"]
    assert error["code"] == ErrorCode.VALIDATION_FAILED.value
    assert error["details"]["note_min_chars"] == MIN_WARN_RELEASE_NOTE_CHARS
    assert error["details"]["quality_status"] == "WARN"
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "PENDING_REVIEW"

    explained = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED", "note": "光照不足，但数据质量可接受，允许放行"},
    )
    assert explained.status_code == 201, explained.text
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "APPROVED"


def test_warn_episode_can_be_rejected_with_short_note(
    client, db, storage, reviewer_headers
) -> None:
    episode, run = seed_episode(db, quality_status="WARN")
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "REJECTED", "note": "no"},
    )
    assert response.status_code == 201
    assert response.json()["data"]["new_status"] == "REJECTED"


def test_pass_episode_approval_note_is_optional(
    client, db, storage, reviewer_headers
) -> None:
    episode, run = seed_episode(db, quality_status="PASS")
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED"},
    )
    assert response.status_code == 201


def test_invalid_decision_is_422(client, db, storage, reviewer_headers) -> None:
    episode, run = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "MAYBE"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == ErrorCode.VALIDATION_FAILED.value


def test_unknown_annotation_set_is_404(client, reviewer_headers) -> None:
    response = client.post(
        f"/api/v1/annotation-sets/{new_id()}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.NOT_FOUND.value


# ---------------------------------------------------------------------------
# DETECT_OBJECTS + BUILD_PREVIEW handlers (FakeDetector, clearly labelled)
# ---------------------------------------------------------------------------


def test_handler_registry_resolves_m4_steps() -> None:
    """The workflow engine registry must resolve both M4 step handlers."""
    from embodied_dataops.workflow.handlers import get_handler

    detect = get_handler(Step.DETECT_OBJECTS)
    preview = get_handler(Step.BUILD_PREVIEW)
    assert callable(detect) and callable(preview)
    assert detect.__self__.__class__ is DetectObjectsHandler
    assert preview.__self__.__class__ is BuildPreviewHandler


def _handler_context(db, storage, workspace: Path, task: Task) -> HandlerContext:
    return HandlerContext(
        session=db,
        task=task,
        settings=get_settings(),
        storage=storage,
        workspace=workspace,
        cancel_check=lambda: False,
    )


def _running_task(db, run_id: str, step: Step, *, pool: str = "cpu") -> Task:
    task = Task(
        id=new_id(),
        run_id=run_id,
        step=step.value,
        pool=pool,
        state="RUNNING",
        attempt=1,
        max_attempts=3,
        next_run_at=utcnow(),
        created_at=utcnow(),
    )
    db.add(task)
    db.flush()
    return task


def _new_run(db, episode: Episode) -> PipelineRun:
    run = PipelineRun(
        id=new_id(),
        episode_id=episode.id,
        pipeline_version="1.0.0",
        state="RUNNING",
        created_at=utcnow(),
    )
    db.add(run)
    db.flush()
    return run


def _seed_canonical_episode(
    db,
    storage,
    tmp_path: Path,
    *,
    frame_count: int = 20,
    with_normalized_assets: bool = False,
):
    episode, run = seed_episode(
        db, episode_type="ego_video_episode", sample_count=frame_count, source_type="EGO_MP4"
    )
    video_path = write_test_video(tmp_path / "source.mp4", frame_count=frame_count)
    keys = canonical_keys(episode.id, 1)
    storage.put_file(keys["video"], video_path, "video/mp4")
    metadata = CanonicalMetadata(
        episode_id=episode.id,
        episode_type="ego_video_episode",
        data_origin="synthetic",
        source_type="EGO_MP4",
        source_asset_sha256="c" * 64,
        source_episode_index=0,
        duration_ns=frame_count * SAMPLE_INTERVAL_NS,
        clock_domain="ego_video_relative",
        joint_names=[],
        state_dim=0,
        action_dim=0,
        samples_parquet="samples.parquet",
        video_path="frames/front.mp4",
        normalization_valid=True,
        sample_count=frame_count,
        video_available=True,
    )
    storage.put_bytes(keys["metadata"], metadata.canonical_json(), "application/json")
    if with_normalized_assets:
        # Mimic NORMALIZE_EPISODE: it registers video + a 320px thumbnail Asset
        # and creates the camera Modality row (object_key still empty), so
        # BUILD_PREVIEW must reuse them instead of registering duplicates.
        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        assert ok and frame is not None
        thumb_w, thumb_h = 320, 240
        resized = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        assert ok
        thumbnail_bytes = buffer.tobytes()
        storage.put_bytes(keys["thumbnail"], thumbnail_bytes, "image/jpeg")
        db.add(
            Asset(
                id=new_id(),
                sha256=sha256_file(video_path),
                byte_size=video_path.stat().st_size,
                media_type="video/mp4",
                object_key=keys["video"],
                origin="canonical",
                logical_name="canonical:frames/front.mp4",
                meta_json={"fps": 10, "width": TARGET_WIDTH, "height": TARGET_HEIGHT},
                created_at=utcnow(),
            )
        )
        db.add(
            Asset(
                id=new_id(),
                sha256=sha256_bytes(thumbnail_bytes),
                byte_size=len(thumbnail_bytes),
                media_type="image/jpeg",
                object_key=keys["thumbnail"],
                origin="derived",
                logical_name="canonical:frames/thumbnail.jpg",
                meta_json={},
                created_at=utcnow(),
            )
        )
        db.add(
            Modality(
                id=new_id(),
                episode_id=episode.id,
                name="front",
                kind="camera",
                dtype="uint8",
                shape_json=[TARGET_HEIGHT, TARGET_WIDTH, 3],
                clock_domain="ego_video_pts_relative",
                sample_count=frame_count,
                min_t_ns=0,
                max_t_ns=(frame_count - 1) * SAMPLE_INTERVAL_NS,
                source_key="front",
                object_key=None,
            )
        )
    normalize_task = Task(
        id=new_id(),
        run_id=run.id,
        step=Step.NORMALIZE_EPISODE.value,
        pool="cpu",
        state="SUCCEEDED",
        attempt=1,
        max_attempts=3,
        next_run_at=utcnow(),
        output_json={
            "canonical_version": 1,
            "metadata_key": keys["metadata"],
            "video_key": keys["video"],
            "samples_key": keys["samples"],
        },
        created_at=utcnow(),
    )
    db.add(normalize_task)
    db.flush()
    return episode, run


def test_detect_objects_handler_creates_reviewable_fake_annotation(
    client, db, storage, reviewer_headers, viewer_headers, tmp_path
) -> None:
    episode, run = _seed_canonical_episode(db, storage, tmp_path, frame_count=20)
    task = Task(
        id=new_id(),
        run_id=run.id,
        step=Step.DETECT_OBJECTS.value,
        pool="vision",
        state="RUNNING",
        attempt=1,
        max_attempts=3,
        next_run_at=utcnow(),
        created_at=utcnow(),
    )
    db.add(task)
    db.commit()

    result = DetectObjectsHandler().handle(
        _handler_context(db, storage, tmp_path / "workspace-detect", task)
    )
    db.commit()

    # 20 canonical frames at 10 Hz -> sampled frame_index 0 and 10 (1 fps).
    assert result.data["sampled_frames"] == 2
    assert result.data["expected_sampled_frames"] == 2
    assert result.data["tool"] == "fake"
    assert result.data["is_synthetic"] is True
    assert result.output_asset_id
    assert result.data["artifact_key"] == annotation_key(
        episode.id, result.data["annotation_set_id"]
    )

    annotation_set = db.get(AnnotationSet, result.data["annotation_set_id"])
    assert annotation_set is not None
    assert annotation_set.status == "PENDING_REVIEW"
    assert annotation_set.model_id == "fake"
    assert annotation_set.model_sha256 == FAKE_WEIGHTS_SHA256
    assert annotation_set.artifact_asset_id == result.output_asset_id
    assert annotation_set.summary_json["tool"] == "fake"
    assert annotation_set.summary_json["is_synthetic"] is True

    document = json.loads(
        storage.get_bytes(result.data["artifact_key"]).decode("utf-8")
    )
    assert document["tool"] == "fake"
    assert document["is_synthetic"] is True
    assert detection.validate_annotation_document(document) == []
    assert [frame["frame_index"] for frame in document["frames"]] == [0, 10]

    # The reviewer sees the FakeDetector result explicitly labelled as fake.
    listing = client.get(
        f"/api/v1/episodes/{episode.id}/annotations", headers=viewer_headers
    )
    assert listing.status_code == 200
    payload = listing.json()["data"]
    assert payload["annotation_set"]["tool"] == "fake"
    assert payload["annotation_set"]["is_synthetic"] is True
    assert [frame["frame_index"] for frame in payload["frames"]] == [0, 10]

    review = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED", "note": "fake detection accepted for contract test"},
    )
    assert review.status_code == 201
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "APPROVED"


def test_build_preview_handler_writes_640x480_thumbnail_and_references_video(
    db, storage, tmp_path
) -> None:
    episode, run = _seed_canonical_episode(db, storage, tmp_path, frame_count=20)
    task = Task(
        id=new_id(),
        run_id=run.id,
        step=Step.BUILD_PREVIEW.value,
        pool="cpu",
        state="RUNNING",
        attempt=1,
        max_attempts=3,
        next_run_at=utcnow(),
        created_at=utcnow(),
    )
    db.add(task)
    db.commit()

    result = BuildPreviewHandler().handle(
        _handler_context(db, storage, tmp_path / "workspace-preview", task)
    )
    db.commit()

    assert result.data["width"] == TARGET_WIDTH
    assert result.data["height"] == TARGET_HEIGHT
    assert result.data["thumbnail_frame_index"] == 0
    assert result.data["thumbnail_reused"] is False
    assert result.data["video_asset_created"] is True
    assert result.data["video_asset_reused"] is False
    keys = canonical_keys(episode.id, 1)
    assert result.data["thumbnail_key"] == keys["thumbnail"]
    thumbnail_asset = db.get(Asset, result.data["thumbnail_asset_id"])
    assert thumbnail_asset is not None
    assert thumbnail_asset.media_type == "image/jpeg"
    assert thumbnail_asset.object_key == keys["thumbnail"]
    decoded = cv2.imdecode(
        np.frombuffer(storage.get_bytes(keys["thumbnail"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None and decoded.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3)

    # Preview video reference: the camera Modality points at the canonical video.
    modality = db.scalars(
        select(Modality).where(Modality.episode_id == episode.id, Modality.kind == "camera")
    ).one()
    assert modality.object_key == keys["video"]
    assert result.data["video_key"] == keys["video"]
    assert result.data["video_asset_id"] is not None

    # BUILD_PREVIEW must not mutate the raw canonical video object.
    original = storage.get_bytes(keys["video"])
    assert len(original) > 0
    assert result.data["decode_backend"] in {"cv2", "av"}


def test_build_preview_reuses_normalized_assets_without_duplicates(db, storage, tmp_path) -> None:
    """Regression: NORMALIZE_EPISODE already registered video+thumbnail Assets.

    A duplicate ``assets.object_key`` insert rolled the whole task transaction
    back, so READY_FOR_REVIEW was never claimable and the Episode stayed
    PROCESSING on real F04 ego pipelines.
    """
    episode, run = _seed_canonical_episode(
        db, storage, tmp_path, frame_count=20, with_normalized_assets=True
    )
    keys = canonical_keys(episode.id, 1)
    normalized_thumbnail = storage.get_bytes(keys["thumbnail"])
    assets_before = db.scalar(select(func.count()).select_from(Asset))
    task = _running_task(db, run.id, Step.BUILD_PREVIEW)
    db.commit()

    result = BuildPreviewHandler().handle(
        _handler_context(db, storage, tmp_path / "workspace-reuse", task)
    )
    db.commit()

    assert result.data["thumbnail_reused"] is True
    assert result.data["video_asset_reused"] is True
    assert result.data["thumbnail_width"] == 320
    assert result.data["thumbnail_height"] == 240
    assert result.data["thumbnail_sha256"] == sha256_bytes(normalized_thumbnail)
    assert storage.get_bytes(keys["thumbnail"]) == normalized_thumbnail
    assert db.scalar(select(func.count()).select_from(Asset)) == assets_before
    assert len(db.scalars(select(Asset).where(Asset.object_key == keys["thumbnail"])).all()) == 1
    assert len(db.scalars(select(Asset).where(Asset.object_key == keys["video"])).all()) == 1
    cameras = [
        modality
        for modality in db.scalars(
            select(Modality).where(Modality.episode_id == episode.id)
        ).all()
        if modality.kind == "camera"
    ]
    assert len(cameras) == 1
    assert cameras[0].object_key == keys["video"]

    # A retry run (fresh PipelineRun with only BUILD_PREVIEW, as
    # POST /tasks/{id}/retry creates) must stay idempotent and resolve the
    # canonical layout without a NORMALIZE_EPISODE task in that run.
    retry_run = _new_run(db, episode)
    retry_task = _running_task(db, retry_run.id, Step.BUILD_PREVIEW)
    db.commit()
    again = BuildPreviewHandler().handle(
        _handler_context(db, storage, tmp_path / "workspace-reuse-retry", retry_task)
    )
    db.commit()
    assert again.data["thumbnail_reused"] is True
    assert again.output_asset_id == result.output_asset_id
    assert db.scalar(select(func.count()).select_from(Asset)) == assets_before
    assert (
        len(
            [
                modality
                for modality in db.scalars(
                    select(Modality).where(Modality.episode_id == episode.id)
                ).all()
                if modality.kind == "camera"
            ]
        )
        == 1
    )


def test_build_preview_generates_thumbnail_when_normalizer_skipped_it(db, storage, tmp_path) -> None:
    episode, run = _seed_canonical_episode(
        db, storage, tmp_path, frame_count=20, with_normalized_assets=True
    )
    keys = canonical_keys(episode.id, 1)
    thumbnail_asset = db.scalars(
        select(Asset).where(Asset.object_key == keys["thumbnail"])
    ).one()
    db.delete(thumbnail_asset)
    db.commit()
    storage.delete(keys["thumbnail"])
    task = _running_task(db, run.id, Step.BUILD_PREVIEW)
    db.commit()

    result = BuildPreviewHandler().handle(
        _handler_context(db, storage, tmp_path / "workspace-thumb", task)
    )
    db.commit()

    assert result.data["thumbnail_reused"] is False
    assert result.data["video_asset_reused"] is True
    assert result.data["thumbnail_width"] == TARGET_WIDTH
    assert result.data["thumbnail_height"] == TARGET_HEIGHT
    decoded = cv2.imdecode(
        np.frombuffer(storage.get_bytes(keys["thumbnail"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None and decoded.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3)
    assert len(db.scalars(select(Asset).where(Asset.object_key == keys["thumbnail"])).all()) == 1


def test_build_preview_rejects_asset_referencing_missing_object(db, storage, tmp_path) -> None:
    episode, run = _seed_canonical_episode(
        db, storage, tmp_path, frame_count=20, with_normalized_assets=True
    )
    keys = canonical_keys(episode.id, 1)
    storage.delete(keys["thumbnail"])
    task = _running_task(db, run.id, Step.BUILD_PREVIEW)
    db.commit()

    with pytest.raises(TerminalTaskError) as excinfo:
        BuildPreviewHandler().handle(
            _handler_context(db, storage, tmp_path / "workspace-missing", task)
        )
    assert excinfo.value.code == ErrorCode.OBJECT_NOT_FOUND
    db.rollback()  # the engine rolls the transaction back on a terminal task error
