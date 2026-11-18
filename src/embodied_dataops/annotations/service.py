"""Annotation review business logic (SPEC 7.2, 9.4).

Invariants enforced here:

* Only the latest ``AnnotationSet`` of an Episode is reviewed; it must be
  ``PENDING_REVIEW``.  Approving/rejecting is a one-shot transition -- a second
  review attempt (or a concurrent reviewer who lost the race) gets
  ``409 ANNOTATION_ALREADY_REVIEWED`` instead of overwriting the decision.
* The raw ``annotation/1.0`` JSON object is immutable: it is only ever read.
  A rejected set is not edited; re-running detection creates a new set.
* ``WARN`` Episodes require an explanatory release note of at least 10
  characters before they can be approved for a publishable dataset version
  (SPEC 7.2) -- a 422 ``VALIDATION_FAILED`` names the requirement.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_keys
from embodied_dataops.config import Settings, get_settings
from embodied_dataops.db.models import (
    AnnotationReview,
    AnnotationRevision,
    AnnotationSet,
    Asset,
    Episode,
    Modality,
)
from embodied_dataops.domain.enums import AnnotationStatus, ReviewDecision
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.processing.detection import validate_annotation_document
from embodied_dataops.storage.base import StorageBackend
from embodied_dataops.workflow.outbox import utcnow

#: SPEC 7.2: WARN release note must explain the exception in >= 10 characters.
MIN_WARN_RELEASE_NOTE_CHARS = 10


def latest_annotation_set(session: Session, episode_id: str) -> AnnotationSet | None:
    """Newest AnnotationSet for an Episode (UUIDv7 ids keep tie ordering stable)."""
    statement = (
        select(AnnotationSet)
        .where(AnnotationSet.episode_id == episode_id)
        .order_by(AnnotationSet.created_at.desc(), AnnotationSet.id.desc())
        .limit(1)
    )
    return session.scalars(statement).first()


def get_annotation_set(session: Session, annotation_set_id: str) -> AnnotationSet:
    annotation_set = session.get(AnnotationSet, annotation_set_id)
    if annotation_set is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            f"AnnotationSet {annotation_set_id} not found",
            details={"annotation_set_id": annotation_set_id},
        )
    return annotation_set


def _load_artifact(session: Session, storage: StorageBackend, annotation_set: AnnotationSet) -> tuple[dict[str, Any], Asset]:
    if not annotation_set.artifact_asset_id:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "AnnotationSet has no stored annotation artifact",
            details={"annotation_set_id": annotation_set.id},
        )
    asset = session.get(Asset, annotation_set.artifact_asset_id)
    if asset is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "Annotation artifact Asset row is missing",
            details={"annotation_set_id": annotation_set.id},
        )
    try:
        document = json.loads(storage.get_bytes(asset.object_key).decode("utf-8"))
    except ApiError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Stored annotation artifact is not valid JSON",
            details={"annotation_set_id": annotation_set.id},
        ) from exc
    if not isinstance(document, dict):
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Stored annotation artifact must be a JSON object",
            details={"annotation_set_id": annotation_set.id},
        )
    # Model output is untrusted input (SPEC 12.1): never serve or approve an
    # artifact that violates annotation/1.0 or the 640x480 coordinate bounds.
    problems = validate_annotation_document(document)
    if problems:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Stored annotation artifact violates annotation/1.0",
            details={"annotation_set_id": annotation_set.id, "errors": problems[:20]},
        )
    return document, asset


def _preview_video_key(session: Session, episode: Episode) -> str:
    """Camera modality object key, falling back to the deterministic canonical path."""
    modality = session.scalars(
        select(Modality)
        .where(Modality.episode_id == episode.id, Modality.kind == "camera")
        .order_by(Modality.name)
    ).first()
    if modality is not None and modality.object_key:
        return modality.object_key
    return canonical_keys(episode.id, int(episode.canonical_version or 1))["video"]


def build_annotation_view(
    session: Session,
    *,
    storage: StorageBackend,
    episode: Episode,
    annotation_set: AnnotationSet,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Response payload for ``GET /episodes/{id}/annotations`` (SPEC 9.4).

    ``preview_url`` on each frame is the short-lived presigned URL of the
    canonical preview video; consumers seek to ``frame_index`` (1 fps sample).
    ``thumbnail_url`` is the frame-0 still produced by BUILD_PREVIEW when it
    exists.  No presigned URL is ever logged.
    """
    settings = settings or get_settings()
    document, artifact = _load_artifact(session, storage, annotation_set)
    ttl = settings.download_url_ttl_seconds
    preview_url = storage.presign_get(_preview_video_key(session, episode), expires_seconds=ttl)
    thumbnail_key = canonical_keys(episode.id, int(episode.canonical_version or 1))["thumbnail"]
    thumbnail_url = (
        storage.presign_get(thumbnail_key, expires_seconds=ttl)
        if storage.exists(thumbnail_key)
        else None
    )
    expires_at = utcnow() + timedelta(seconds=ttl)

    frames: list[dict[str, Any]] = []
    for frame in document.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        detections: list[dict[str, Any]] = []
        for detection in frame.get("detections") or []:
            if not isinstance(detection, dict):
                continue
            detections.append(
                {
                    "class_id": int(detection["class_id"]),
                    "class_name": str(detection["class_name"]),
                    "score": float(detection["score"]),
                    "xyxy": [float(value) for value in detection["xyxy"]],
                }
            )
        frames.append(
            {
                "frame_index": int(frame["frame_index"]),
                "t_ns": int(frame["t_ns"]),
                "preview_url": preview_url,
                "detections": detections,
            }
        )

    annotation_set_data = {
        "annotation_set_id": annotation_set.id,
        "episode_id": annotation_set.episode_id,
        "run_id": annotation_set.run_id,
        "model_id": annotation_set.model_id,
        "model_sha256": annotation_set.model_sha256,
        "status": annotation_set.status,
        "created_at": annotation_set.created_at,
        "schema_version": document.get("schema_version"),
        "tool": document.get("tool"),
        "is_synthetic": bool(document.get("is_synthetic")),
        "model": document.get("model") or {},
        "stats": document.get("stats") or {},
        "summary": annotation_set.summary_json or {},
        "artifact_url": storage.presign_get(artifact.object_key, expires_seconds=ttl),
        "artifact_sha256": artifact.sha256,
        "artifact_size": int(artifact.byte_size),
        "preview_video_url": preview_url,
        "thumbnail_url": thumbnail_url,
        "preview_expires_at": expires_at,
        # V2 D2-M3: explicit revision pointers so clients know whether the set is
        # reviewed through the annotation-revision API instead of the v1 reviews.
        "current_revision_no": int(annotation_set.current_revision_no or 1),
        "approved_revision_id": annotation_set.approved_revision_id,
        "revision_count": int(
            session.scalar(
                select(func.count())
                .select_from(AnnotationRevision)
                .where(AnnotationRevision.annotation_set_id == annotation_set.id)
            )
            or 0
        ),
    }
    return {"annotation_set": annotation_set_data, "frames": frames}


def review_annotation_set(
    session: Session,
    *,
    annotation_set_id: str,
    decision: ReviewDecision | str,
    note: str,
    reviewer: Any,
    storage: StorageBackend | None = None,
) -> tuple[AnnotationReview, str, str]:
    """Apply a reviewer decision, returning ``(review, previous_status, new_status)``.

    ``reviewer`` is a ``Principal`` (has ``key_id``); the route enforces the
    ``review:write`` permission before calling this function.  When ``storage``
    is supplied, approval re-validates the immutable annotation artifact
    (schema + 640x480 bounds) because model output is untrusted input.
    """
    decision_value = decision.value if isinstance(decision, ReviewDecision) else str(decision)
    if decision_value not in (ReviewDecision.APPROVED.value, ReviewDecision.REJECTED.value):
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "decision must be APPROVED or REJECTED",
            details={"decision": decision_value},
        )
    # Row lock serialises concurrent reviewers; the unique constraint on
    # annotation_reviews.annotation_set_id is the second line of defence.
    annotation_set = session.scalars(
        select(AnnotationSet).where(AnnotationSet.id == annotation_set_id).with_for_update()
    ).first()
    if annotation_set is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            f"AnnotationSet {annotation_set_id} not found",
            details={"annotation_set_id": annotation_set_id},
        )
    if annotation_set.status != AnnotationStatus.PENDING_REVIEW.value:
        raise ApiError(
            ErrorCode.ANNOTATION_ALREADY_REVIEWED,
            "AnnotationSet has already been reviewed; detection must be re-run for a new review",
            details={"annotation_set_id": annotation_set.id, "status": annotation_set.status},
        )

    cleaned_note = (note or "").strip()
    episode = session.get(Episode, annotation_set.episode_id)
    if (
        decision_value == ReviewDecision.APPROVED.value
        and episode is not None
        and episode.quality_status == "WARN"
        and len(cleaned_note) < MIN_WARN_RELEASE_NOTE_CHARS
    ):
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "A WARN episode requires a release note of at least "
            f"{MIN_WARN_RELEASE_NOTE_CHARS} characters before it can be approved",
            status_code=422,
            details={
                "quality_status": "WARN",
                "note_min_chars": MIN_WARN_RELEASE_NOTE_CHARS,
                "note_chars": len(cleaned_note),
            },
        )

    previous_status = annotation_set.status
    if decision_value == ReviewDecision.APPROVED.value and storage is not None:
        # Refuse to approve corrupt/invalid model output, even if the stored
        # artifact was tampered with after DETECT_OBJECTS validated it.
        _load_artifact(session, storage, annotation_set)
    annotation_set.status = decision_value
    review = AnnotationReview(
        id=new_id(),
        annotation_set_id=annotation_set.id,
        reviewer_key_id=reviewer.key_id,
        decision=decision_value,
        note=cleaned_note,
        created_at=utcnow(),
    )
    session.add(review)
    session.flush()
    return review, previous_status, decision_value


__all__ = [
    "MIN_WARN_RELEASE_NOTE_CHARS",
    "build_annotation_view",
    "get_annotation_set",
    "latest_annotation_set",
    "review_annotation_set",
]
