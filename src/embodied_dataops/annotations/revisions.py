"""V2 annotation revisions, manual edits and two-person approval (SPEC §4).

This module implements the ``annotation-revision/2.0`` content contract and the
immutable revision chain behind ``annotation_sets`` (DO2-14..17).  Design rules
enforced here:

* ``AnnotationSet`` stays the logical annotation set.  Its ``artifact_asset_id``
  (the original ``annotation/1.0`` model output) is **never** rewritten; every
  manual edit creates a new ``AnnotationRevision`` row plus a new immutable
  JSON Asset.  Revision 1 is the baseline content derived from the original
  model artifact; it is materialised in ``IN_REVIEW`` state by
  :func:`ensure_baseline_revision` (called by the v2 detection creation path and
  lazily by the first edit) so a v2 set is distinguishable from a legacy v1 set
  even before the first human edit.
* ``annotation_sets.current_revision_no`` is a CAS pointer.  ``POST`` edits hold
  ``SELECT ... FOR UPDATE`` on the set row, compare the strong ETag
  ``"annotation-rev:<n>"`` and only then append revision ``n+1``.  A stale ETag
  is ``412 PRECONDITION_FAILED`` and never overwrites somebody else's edit.
* A request replaces the **complete detections of the listed frames**.  All
  other frames are copied byte-for-byte (as canonical JSON) from the base
  revision; no frame is silently dropped.
* Approving a revision requires a reviewer who is not the revision creator and
  atomically switches ``annotation_sets.approved_revision_id``.  Existing
  approvals are never withdrawn implicitly: creating a newer revision or
  rejecting it leaves the previous approved pointer intact, which is what keeps
  an already published dataset snapshot stable (DO2-17).
* Every revision state transition bumps the owning Episode's
  ``business_revision`` in the same transaction via
  :func:`embodied_dataops.operations.revisions.bump` so DO-A02 conditional
  writes on the Episode cannot miss the annotation change.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops import ANNOTATION_SCHEMA_VERSION, TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.canonical.contract import canonical_json_bytes, canonical_json_sha256
from embodied_dataops.db.models import (
    AnnotationRevision,
    AnnotationRevisionEdit,
    AnnotationSet,
    Asset,
    Episode,
)
from embodied_dataops.domain.enums import AnnotationStatus
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.processing.detection import COCO_CLASS_NAMES, validate_annotation_document
from embodied_dataops.storage.base import StorageBackend
from embodied_dataops.workflow.outbox import utcnow

REVISION_SCHEMA_VERSION = "annotation-revision/2.0"
FRAME_WIDTH = TARGET_WIDTH
FRAME_HEIGHT = TARGET_HEIGHT

#: SPEC §4.2 hard content limits.
MAX_FRAMES = 600
MAX_DETECTIONS_PER_FRAME = 100

#: ``POST /annotation-sets/{id}/revisions`` note length (SPEC §4.2).
NOTE_MIN_CHARS = 10
NOTE_MAX_CHARS = 500
#: A v2 approval must explain the decision (DO2-16 "强制备注").
APPROVAL_NOTE_MIN_CHARS = 10

MANUAL_CLASS_NAME = "other"
COCO_CLASS_COUNT = len(COCO_CLASS_NAMES)
DETECTION_ORIGINS = ("MODEL", "HUMAN")
DETECTION_KEYS = frozenset({"detection_id", "class_id", "class_name", "score", "xyxy", "origin"})
FRAME_KEYS = frozenset({"frame_index", "t_ns", "detections"})
DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "annotation_set_id",
        "revision_no",
        "camera_name",
        "frame_width",
        "frame_height",
        "frames",
    }
)

ETAG_KIND = "annotation-rev"
#: Default camera name for the legacy single-camera ``annotation/1.0`` baseline.
DEFAULT_CAMERA_NAME = "front"

STATE_DRAFT = "DRAFT"
STATE_IN_REVIEW = "IN_REVIEW"
STATE_APPROVED = "APPROVED"
STATE_REJECTED = "REJECTED"

REVISION_KEY_TEMPLATE = (
    "annotations/{episode_id}/{annotation_set_id}/revisions/rev-{revision_no:06d}.json"
)


# ---------------------------------------------------------------------------
# strong ETag helpers
# ---------------------------------------------------------------------------
def revision_etag(revision_no: int) -> str:
    """Strong ETag value (including quotes) for one revision pointer."""
    return f'"{ETAG_KIND}:{int(revision_no)}"'


def parse_revision_if_match(raw: str | None) -> int:
    """Parse an ``If-Match`` header into a revision number (syntax only).

    Missing header -> ``428 PRECONDITION_REQUIRED`` (DO-A02 C2).  Weak, ``*``,
    malformed or wrong-kind validators -> ``412 PRECONDITION_FAILED``.  The
    comparison against the locked ``current_revision_no`` happens later inside
    :func:`create_annotation_revision`, after the Idempotency-Key replay check.
    """
    if raw is None or not str(raw).strip():
        raise ApiError(
            ErrorCode.PRECONDITION_REQUIRED,
            "If-Match header with a strong annotation revision ETag is required",
            details={"expected_etag": f'"{ETAG_KIND}:<current_revision_no>"'},
        )
    value = str(raw).strip()
    if value.lower().startswith("w/"):
        raise ApiError(
            ErrorCode.PRECONDITION_FAILED,
            "Weak ETags are not accepted; use the strong ETag returned by GET",
            details={"reason": "weak_etag_not_allowed"},
        )
    candidates = [item.strip() for item in value.split(",") if item.strip()]
    annotations: list[int] = []
    for candidate in candidates:
        if candidate == "*":
            raise ApiError(
                ErrorCode.PRECONDITION_FAILED,
                "If-Match: * is not accepted; send the explicit annotation revision",
                details={"reason": "explicit_revision_required"},
            )
        unquoted = candidate
        if len(unquoted) >= 2 and unquoted[0] == '"' and unquoted[-1] == '"':
            unquoted = unquoted[1:-1]
        prefix = f"{ETAG_KIND}:"
        if not unquoted.startswith(prefix):
            continue
        suffix = unquoted[len(prefix) :]
        if not suffix.isdigit() or int(suffix) < 1:
            continue
        annotations.append(int(suffix))
    if not annotations:
        raise ApiError(
            ErrorCode.PRECONDITION_FAILED,
            f"If-Match must carry a strong ETag of the form '\"{ETAG_KIND}:<n>\"'",
            details={"reason": "malformed_or_wrong_kind", "provided": value[:120]},
        )
    return annotations[0]


def revision_object_key(episode_id: str, annotation_set_id: str, revision_no: int) -> str:
    return REVISION_KEY_TEMPLATE.format(
        episode_id=episode_id,
        annotation_set_id=annotation_set_id,
        revision_no=int(revision_no),
    )


# ---------------------------------------------------------------------------
# annotation-revision/2.0 validation
# ---------------------------------------------------------------------------
def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_xyxy(xyxy: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4 or not all(
        _is_finite_number(value) for value in xyxy
    ):
        errors.append(f"{prefix}.xyxy must be four finite numbers")
        return
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    if not (0.0 <= x1 < x2 <= float(FRAME_WIDTH)):
        errors.append(
            f"{prefix}.xyxy x-range must satisfy 0 <= x1 < x2 <= {FRAME_WIDTH}"
        )
    if not (0.0 <= y1 < y2 <= float(FRAME_HEIGHT)):
        errors.append(
            f"{prefix}.xyxy y-range must satisfy 0 <= y1 < y2 <= {FRAME_HEIGHT}"
        )


def _validate_detection(
    detection: Any,
    prefix: str,
    *,
    seen_ids: set[str],
    errors: list[str],
) -> None:
    if not isinstance(detection, Mapping):
        errors.append(f"{prefix} must be an object")
        return
    unknown = set(detection) - DETECTION_KEYS
    if unknown:
        errors.append(f"{prefix} has unexpected keys: {sorted(unknown)}")

    detection_id = detection.get("detection_id")
    if not isinstance(detection_id, str) or not detection_id.strip():
        errors.append(f"{prefix}.detection_id must be a non-empty string")
    elif len(detection_id) > 100:
        errors.append(f"{prefix}.detection_id must be at most 100 characters")
    elif detection_id in seen_ids:
        errors.append(f"{prefix}.detection_id {detection_id!r} is duplicated in the frame")
    else:
        seen_ids.add(detection_id)

    class_id = detection.get("class_id")
    class_name = detection.get("class_name")
    if not _is_int(class_id) or not 0 <= int(class_id) < COCO_CLASS_COUNT:
        errors.append(f"{prefix}.class_id must be an integer in [0, {COCO_CLASS_COUNT - 1}]")
    if not isinstance(class_name, str) or not class_name.strip():
        errors.append(f"{prefix}.class_name must be a non-empty string")
    elif class_name != MANUAL_CLASS_NAME:
        if _is_int(class_id) and 0 <= int(class_id) < COCO_CLASS_COUNT:
            expected = COCO_CLASS_NAMES[int(class_id)]
            if class_name != expected:
                errors.append(
                    f"{prefix}.class_name {class_name!r} does not match the detector "
                    f"taxonomy for class_id {int(class_id)} ({expected!r}); use 'other' "
                    "for a manual class"
                )

    score = detection.get("score")
    if score is not None:
        if not _is_finite_number(score) or not 0.0 <= float(score) <= 1.0:
            errors.append(f"{prefix}.score must be null or a number in [0, 1]")

    _validate_xyxy(detection.get("xyxy"), prefix, errors)

    origin = detection.get("origin")
    if origin not in DETECTION_ORIGINS:
        errors.append(f"{prefix}.origin must be one of {list(DETECTION_ORIGINS)}")


def validate_revision_document(document: Any) -> list[str]:
    """Validate a complete ``annotation-revision/2.0`` document.

    Returns a list of violation strings (empty means valid).  The same
    validation runs on freshly built revisions, on every stored revision read
    back from the object store, and on the ``annotation/1.0`` baseline after
    conversion, so a tampered object can never be served or approved.
    """
    errors: list[str] = []
    if not isinstance(document, Mapping):
        return ["annotation revision document must be a JSON object"]
    unknown = set(document) - DOCUMENT_KEYS
    if unknown:
        errors.append(f"unexpected top-level keys: {sorted(unknown)}")

    if document.get("schema_version") != REVISION_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {REVISION_SCHEMA_VERSION!r}, "
            f"got {document.get('schema_version')!r}"
        )
    annotation_set_id = document.get("annotation_set_id")
    if not isinstance(annotation_set_id, str) or not annotation_set_id.strip():
        errors.append("annotation_set_id must be a non-empty string")
    revision_no = document.get("revision_no")
    if not _is_int(revision_no) or int(revision_no) < 1:
        errors.append("revision_no must be a positive integer")
    camera_name = document.get("camera_name")
    if not isinstance(camera_name, str) or not camera_name.strip():
        errors.append("camera_name must be a non-empty string")
    elif len(camera_name) > 120:
        errors.append("camera_name must be at most 120 characters")
    if document.get("frame_width") != FRAME_WIDTH:
        errors.append(f"frame_width must be {FRAME_WIDTH}")
    if document.get("frame_height") != FRAME_HEIGHT:
        errors.append(f"frame_height must be {FRAME_HEIGHT}")

    frames = document.get("frames")
    if not isinstance(frames, list):
        return errors + ["frames must be a list"]
    if len(frames) > MAX_FRAMES:
        errors.append(f"frames must contain at most {MAX_FRAMES} video frames, got {len(frames)}")
    previous_index = -1
    for position, frame in enumerate(frames):
        prefix = f"frames[{position}]"
        if not isinstance(frame, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        unknown_frame = set(frame) - FRAME_KEYS
        if unknown_frame:
            errors.append(f"{prefix} has unexpected keys: {sorted(unknown_frame)}")
        frame_index = frame.get("frame_index")
        if not _is_int(frame_index) or int(frame_index) < 0:
            errors.append(f"{prefix}.frame_index must be a non-negative integer")
            continue
        if int(frame_index) <= previous_index:
            errors.append(f"{prefix}.frame_index is not strictly increasing")
        previous_index = int(frame_index)
        t_ns = frame.get("t_ns")
        if not _is_int(t_ns) or int(t_ns) < 0:
            errors.append(f"{prefix}.t_ns must be a non-negative integer")
        detections = frame.get("detections")
        if not isinstance(detections, list):
            errors.append(f"{prefix}.detections must be a list")
            continue
        if len(detections) > MAX_DETECTIONS_PER_FRAME:
            errors.append(
                f"{prefix}.detections must contain at most {MAX_DETECTIONS_PER_FRAME} "
                f"detections, got {len(detections)}"
            )
        seen_ids: set[str] = set()
        for offset, detection in enumerate(detections):
            _validate_detection(
                detection,
                f"{prefix}.detections[{offset}]",
                seen_ids=seen_ids,
                errors=errors,
            )
    return errors


def validate_replacement_frames(frames: Sequence[Any]) -> list[str]:
    """Validate the request ``frames`` list (not necessarily sorted, keyed by index)."""
    errors: list[str] = []
    previous: set[int] = set()
    if len(frames) > MAX_FRAMES:
        return [f"at most {MAX_FRAMES} replacement frames are accepted, got {len(frames)}"]
    for position, raw_frame in enumerate(frames):
        frame = _as_mapping(raw_frame)
        prefix = f"frames[{position}]"
        if frame is None:
            errors.append(f"{prefix} must be an object")
            continue
        unknown_frame = set(frame) - FRAME_KEYS
        if unknown_frame:
            errors.append(f"{prefix} has unexpected keys: {sorted(unknown_frame)}")
        frame_index = frame.get("frame_index")
        if not _is_int(frame_index) or int(frame_index) < 0:
            errors.append(f"{prefix}.frame_index must be a non-negative integer")
            continue
        if int(frame_index) in previous:
            errors.append(f"{prefix}.frame_index {int(frame_index)} is duplicated in the request")
        previous.add(int(frame_index))
        t_ns = frame.get("t_ns")
        if not _is_int(t_ns) or int(t_ns) < 0:
            errors.append(f"{prefix}.t_ns must be a non-negative integer")
        detections = frame.get("detections")
        if detections is None:
            detections = []
        if not isinstance(detections, (list, tuple)):
            errors.append(f"{prefix}.detections must be a list")
            continue
        if len(detections) > MAX_DETECTIONS_PER_FRAME:
            errors.append(
                f"{prefix}.detections must contain at most {MAX_DETECTIONS_PER_FRAME} "
                f"detections, got {len(detections)}"
            )
        seen_ids: set[str] = set()
        for offset, detection in enumerate(detections):
            _validate_detection(
                detection,
                f"{prefix}.detections[{offset}]",
                seen_ids=seen_ids,
                errors=errors,
            )
    return errors


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _normalise_detection(detection: Any) -> dict[str, Any]:
    frame = _as_mapping(detection)
    if frame is None:  # pragma: no cover - validated before this point
        raise ApiError(ErrorCode.VALIDATION_FAILED, "detection must be an object")
    score = frame.get("score")
    return {
        "detection_id": str(frame["detection_id"]),
        "class_id": int(frame["class_id"]),
        "class_name": str(frame["class_name"]),
        "score": None if score is None else float(score),
        "xyxy": [float(value) for value in frame["xyxy"]],
        "origin": str(frame["origin"]),
    }


def normalise_replacement_frames(frames: Sequence[Any]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for raw_frame in frames:
        frame = _as_mapping(raw_frame)
        if frame is None:  # pragma: no cover - validated before this point
            raise ApiError(ErrorCode.VALIDATION_FAILED, "frame must be an object")
        normalised.append(
            {
                "frame_index": int(frame["frame_index"]),
                "t_ns": int(frame["t_ns"]),
                "detections": [
                    _normalise_detection(detection) for detection in frame.get("detections") or []
                ],
            }
        )
    return normalised


def build_revision_document(
    *,
    annotation_set_id: str,
    revision_no: int,
    camera_name: str,
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Canonical ``annotation-revision/2.0`` object (frames must be sorted)."""
    return {
        "schema_version": REVISION_SCHEMA_VERSION,
        "annotation_set_id": str(annotation_set_id),
        "revision_no": int(revision_no),
        "camera_name": str(camera_name or DEFAULT_CAMERA_NAME),
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "frames": [dict(frame) for frame in frames],
    }


def merge_replacement_frames(
    base_document: Mapping[str, Any],
    replacement_frames: Sequence[Mapping[str, Any]],
    *,
    revision_no: int,
) -> dict[str, Any]:
    """Replace the listed frames wholesale; copy every other frame from the base."""
    frame_map: dict[int, dict[str, Any]] = {
        int(frame["frame_index"]): dict(frame)
        for frame in (base_document.get("frames") or [])
        if isinstance(frame, Mapping)
    }
    for frame in replacement_frames:
        frame_map[int(frame["frame_index"])] = dict(frame)
    ordered = [frame_map[index] for index in sorted(frame_map)]
    return build_revision_document(
        annotation_set_id=str(base_document.get("annotation_set_id") or ""),
        revision_no=revision_no,
        camera_name=str(base_document.get("camera_name") or DEFAULT_CAMERA_NAME),
        frames=ordered,
    )


# ---------------------------------------------------------------------------
# change summary (before/after digests + per-frame diff)
# ---------------------------------------------------------------------------
def _diff_detections(
    before_frame: Mapping[str, Any] | None, after_frame: Mapping[str, Any]
) -> dict[str, Any]:
    before = {
        str(detection["detection_id"]): detection
        for detection in (before_frame or {}).get("detections") or []
    }
    after = {
        str(detection["detection_id"]): detection
        for detection in after_frame.get("detections") or []
    }
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    class_changed: list[str] = []
    geometry_changed: list[str] = []
    score_changed: list[str] = []
    origin_changed: list[str] = []
    for detection_id in sorted(set(before) & set(after)):
        old = before[detection_id]
        new = after[detection_id]
        if old.get("class_id") != new.get("class_id") or old.get("class_name") != new.get("class_name"):
            class_changed.append(detection_id)
        if [float(value) for value in old.get("xyxy") or []] != [
            float(value) for value in new.get("xyxy") or []
        ]:
            geometry_changed.append(detection_id)
        if old.get("score") != new.get("score"):
            score_changed.append(detection_id)
        if old.get("origin") != new.get("origin"):
            origin_changed.append(detection_id)
    return {
        "before_count": len(before),
        "after_count": len(after),
        "added": added,
        "removed": removed,
        "class_changed": class_changed,
        "geometry_changed": geometry_changed,
        "score_changed": score_changed,
        "origin_changed": origin_changed,
    }


def build_change_summary(
    base_document: Mapping[str, Any],
    merged_document: Mapping[str, Any],
    replacement_frames: Sequence[Mapping[str, Any]],
    *,
    note: str,
    base_revision_no: int,
) -> dict[str, Any]:
    """Deterministic before/after summary persisted in ``annotation_revision_edits``."""
    base_map = {
        int(frame["frame_index"]): frame for frame in base_document.get("frames") or []
    }
    merged_map = {
        int(frame["frame_index"]): frame for frame in merged_document.get("frames") or []
    }
    requested = sorted({int(frame["frame_index"]) for frame in replacement_frames})
    per_frame: list[dict[str, Any]] = []
    added_frames: list[int] = []
    replaced_frames: list[int] = []
    unchanged_frames: list[int] = []
    totals = {
        "added": 0,
        "removed": 0,
        "class_changed": 0,
        "geometry_changed": 0,
        "score_changed": 0,
        "origin_changed": 0,
    }
    for frame_index in requested:
        before = base_map.get(frame_index)
        after = merged_map[frame_index]
        if before is None:
            added_frames.append(frame_index)
        elif before == after:
            unchanged_frames.append(frame_index)
        else:
            replaced_frames.append(frame_index)
        diff = _diff_detections(before, after)
        for key in totals:
            totals[key] += len(diff[key])
        per_frame.append({"frame_index": frame_index, **diff})

    effective = sorted(added_frames + replaced_frames)
    return {
        "base_revision_no": int(base_revision_no),
        "revision_no": int(merged_document.get("revision_no") or 0),
        "note": note,
        "frames": {
            "requested": len(requested),
            "added": added_frames,
            "replaced": replaced_frames,
            "unchanged": unchanged_frames,
            "effective": effective,
        },
        "detections": {
            "before": sum(len(frame.get("detections") or []) for frame in base_map.values()),
            "after": sum(len(frame.get("detections") or []) for frame in merged_map.values()),
            **totals,
        },
        "per_frame": per_frame,
    }


# ---------------------------------------------------------------------------
# read helpers
# ---------------------------------------------------------------------------
def get_annotation_set(session: Session, annotation_set_id: str) -> AnnotationSet:
    annotation_set = session.get(AnnotationSet, annotation_set_id)
    if annotation_set is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            f"AnnotationSet {annotation_set_id} not found",
            details={"annotation_set_id": annotation_set_id},
        )
    return annotation_set


def annotation_set_has_revisions(session: Session, annotation_set_id: str) -> bool:
    """A set is a v2 set iff at least one annotation revision row exists."""
    statement = select(AnnotationRevision.id).where(
        AnnotationRevision.annotation_set_id == annotation_set_id
    )
    return session.scalars(statement.limit(1)).first() is not None


def resolve_approved_revision_id(session: Session, annotation_set_id: str) -> str | None:
    """Approved revision pointer used to bind a dataset snapshot (DO2-17)."""
    annotation_set = session.get(AnnotationSet, annotation_set_id)
    if annotation_set is None:
        return None
    return annotation_set.approved_revision_id


def list_annotation_revisions(
    session: Session, annotation_set_id: str
) -> tuple[AnnotationSet, list[AnnotationRevision]]:
    annotation_set = get_annotation_set(session, annotation_set_id)
    rows = list(
        session.scalars(
            select(AnnotationRevision)
            .where(AnnotationRevision.annotation_set_id == annotation_set_id)
            .order_by(AnnotationRevision.revision_no)
        )
    )
    return annotation_set, rows


def _load_asset_bytes(session: Session, storage: StorageBackend, asset_id: str | None) -> bytes:
    if not asset_id:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Annotation revision has no stored artifact",
        )
    asset = session.get(Asset, asset_id)
    if asset is None:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Annotation revision artifact Asset row is missing",
            details={"asset_id": asset_id},
        )
    try:
        return storage.get_bytes(asset.object_key)
    except ApiError as exc:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Annotation revision artifact object is not readable",
            details={"asset_id": asset_id},
        ) from exc


def _parse_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            f"{label} is not valid JSON",
        ) from exc
    if not isinstance(document, dict):
        raise ApiError(ErrorCode.INTERNAL_ERROR, f"{label} must be a JSON object")
    return document


def load_revision_document(
    session: Session,
    storage: StorageBackend,
    revision: AnnotationRevision,
) -> dict[str, Any]:
    """Read + verify one immutable revision object (schema and content digest)."""
    raw = _load_asset_bytes(session, storage, revision.artifact_asset_id)
    document = _parse_json_object(raw, label="Annotation revision artifact")
    problems = validate_revision_document(document)
    if problems:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Stored annotation revision violates annotation-revision/2.0",
            details={"revision_id": revision.id, "errors": problems[:20]},
        )
    if int(document.get("revision_no") or 0) != int(revision.revision_no):
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Stored annotation revision number does not match its row",
            details={"revision_id": revision.id, "revision_no": revision.revision_no},
        )
    digest = canonical_json_sha256(document)
    if digest != revision.content_sha256:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Stored annotation revision content SHA-256 does not match the recorded digest",
            details={"revision_id": revision.id, "expected": revision.content_sha256, "observed": digest},
        )
    return document


def _annotation_set_base_document(
    session: Session, storage: StorageBackend, annotation_set: AnnotationSet
) -> dict[str, Any]:
    """Derive ``annotation-revision/2.0`` revision 1 from the original artifact.

    The original ``annotation/1.0`` artifact is never modified; this is a pure
    deterministic conversion so a v1 set can enter the revision workflow.  A
    ``annotation-revision/2.0`` object is accepted as-is (used by the v2
    detection creation path).
    """
    raw = _load_asset_bytes(session, storage, annotation_set.artifact_asset_id)
    source = _parse_json_object(raw, label="AnnotationSet artifact")
    schema = source.get("schema_version")
    if schema == REVISION_SCHEMA_VERSION:
        problems = validate_revision_document(source)
        if problems:
            raise ApiError(
                ErrorCode.INTERNAL_ERROR,
                "Stored annotation revision baseline violates annotation-revision/2.0",
                details={"annotation_set_id": annotation_set.id, "errors": problems[:20]},
            )
        return build_revision_document(
            annotation_set_id=annotation_set.id,
            revision_no=1,
            camera_name=str(source.get("camera_name") or DEFAULT_CAMERA_NAME),
            frames=list(source.get("frames") or []),
        )
    if schema != ANNOTATION_SCHEMA_VERSION:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "AnnotationSet artifact has an unsupported schema_version; cannot seed revision 1",
            details={"annotation_set_id": annotation_set.id, "schema_version": schema},
        )
    problems = validate_annotation_document(source)
    if problems:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Stored annotation artifact violates annotation/1.0 and cannot seed revision 1",
            details={"annotation_set_id": annotation_set.id, "errors": problems[:20]},
        )
    frames: list[dict[str, Any]] = []
    for frame in source.get("frames") or []:
        frame_index = int(frame["frame_index"])
        detections = [
            {
                "detection_id": f"model-{frame_index}-{ordinal:03d}",
                "class_id": int(detection["class_id"]),
                "class_name": str(detection["class_name"]),
                "score": float(detection["score"]),
                "xyxy": [float(value) for value in detection["xyxy"]],
                "origin": "MODEL",
            }
            for ordinal, detection in enumerate(frame.get("detections") or [])
        ]
        frames.append({"frame_index": frame_index, "t_ns": int(frame["t_ns"]), "detections": detections})
    document = build_revision_document(
        annotation_set_id=annotation_set.id,
        revision_no=1,
        camera_name=DEFAULT_CAMERA_NAME,
        frames=frames,
    )
    problems = validate_revision_document(document)
    if problems:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "Converted annotation/1.0 artifact violates annotation-revision/2.0",
            details={"annotation_set_id": annotation_set.id, "errors": problems[:20]},
        )
    return document


def get_revision(
    session: Session, annotation_set_id: str, revision_no: int
) -> tuple[AnnotationSet, AnnotationRevision | None]:
    annotation_set = get_annotation_set(session, annotation_set_id)
    revision = session.scalars(
        select(AnnotationRevision).where(
            AnnotationRevision.annotation_set_id == annotation_set_id,
            AnnotationRevision.revision_no == int(revision_no),
        )
    ).first()
    return annotation_set, revision


def get_revision_document(
    session: Session, storage: StorageBackend, annotation_set_id: str, revision_no: int
) -> tuple[AnnotationSet, AnnotationRevision | None, dict[str, Any], str]:
    """Return the revision row (None for the implicit baseline) and its content."""
    annotation_set, revision = get_revision(session, annotation_set_id, revision_no)
    if revision is None:
        has_any = annotation_set_has_revisions(session, annotation_set_id)
        if int(revision_no) == 1 and not has_any:
            document = _annotation_set_base_document(session, storage, annotation_set)
            return annotation_set, None, document, canonical_json_sha256(document)
        raise ApiError(
            ErrorCode.NOT_FOUND,
            f"Annotation revision {revision_no} not found for AnnotationSet {annotation_set_id}",
            details={"annotation_set_id": annotation_set_id, "revision_no": int(revision_no)},
        )
    document = load_revision_document(session, storage, revision)
    return annotation_set, revision, document, revision.content_sha256


# ---------------------------------------------------------------------------
# persistence helpers
# ---------------------------------------------------------------------------
def _store_revision_asset(
    session: Session,
    storage: StorageBackend,
    *,
    annotation_set: AnnotationSet,
    revision_no: int,
    payload: bytes,
) -> Asset:
    key = revision_object_key(annotation_set.episode_id, annotation_set.id, revision_no)
    stored = storage.put_bytes(key, payload, "application/json")
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        media_type="application/json",
        object_key=key,
        origin="derived",
        logical_name=f"annotation_revision:{annotation_set.episode_id}:{revision_no}",
        meta_json={
            "kind": "annotation_revision",
            "schema_version": REVISION_SCHEMA_VERSION,
            "annotation_set_id": annotation_set.id,
            "episode_id": annotation_set.episode_id,
            "revision_no": int(revision_no),
        },
        created_at=utcnow(),
    )
    session.add(asset)
    session.flush()
    return asset


def _bump_episode_business_revision(session: Session, annotation_set: AnnotationSet) -> None:
    """Single call site: reflect the annotation state change on the Episode (DO-A02 C2)."""
    episode = session.execute(
        select(Episode)
        .where(Episode.id == annotation_set.episode_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if episode is None:  # pragma: no cover - FK protects this
        return
    try:
        from embodied_dataops.operations.revisions import bump
    except ImportError:  # pragma: no cover - operations package is part of V2
        episode.business_revision = int(episode.business_revision or 1) + 1
        session.flush()
    else:
        bump(session, episode)


def ensure_baseline_revision(
    session: Session,
    *,
    storage: StorageBackend,
    annotation_set: AnnotationSet,
) -> AnnotationRevision:
    """Materialise revision 1 (the model output baseline) exactly once."""
    existing = session.scalars(
        select(AnnotationRevision).where(
            AnnotationRevision.annotation_set_id == annotation_set.id,
            AnnotationRevision.revision_no == 1,
        )
    ).first()
    if existing is not None:
        return existing
    document = _annotation_set_base_document(session, storage, annotation_set)
    payload = canonical_json_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    asset = _store_revision_asset(
        session,
        storage,
        annotation_set=annotation_set,
        revision_no=1,
        payload=payload,
    )
    revision = AnnotationRevision(
        id=new_id(),
        annotation_set_id=annotation_set.id,
        revision_no=1,
        base_revision_id=None,
        artifact_asset_id=asset.id,
        content_sha256=digest,
        state=STATE_IN_REVIEW,
        created_by=None,
        created_at=utcnow(),
    )
    session.add(revision)
    session.flush()
    return revision


@dataclass
class CreatedRevision:
    revision: AnnotationRevision
    edit: AnnotationRevisionEdit
    change_summary: dict[str, Any]


def create_annotation_revision(
    session: Session,
    *,
    storage: StorageBackend,
    annotation_set_id: str,
    expected_revision_no: int,
    base_revision_no: int,
    frames: Sequence[Any],
    note: str,
    actor: Any,
) -> CreatedRevision:
    """CAS-append revision ``current+1`` replacing the listed frames (SPEC §4.2)."""
    # Serialise concurrent editors on the AnnotationSet row.  populate_existing
    # guarantees the second waiter re-reads the committed current_revision_no.
    annotation_set = session.execute(
        select(AnnotationSet)
        .where(AnnotationSet.id == annotation_set_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if annotation_set is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            f"AnnotationSet {annotation_set_id} not found",
            details={"annotation_set_id": annotation_set_id},
        )

    current = int(annotation_set.current_revision_no or 1)
    if int(expected_revision_no) != current:
        raise ApiError(
            ErrorCode.PRECONDITION_FAILED,
            "The annotation set has changed since the supplied If-Match revision",
            details={
                "annotation_set_id": annotation_set.id,
                "current_revision_no": current,
                "expected_etag": revision_etag(current),
                "provided_revision_no": int(expected_revision_no),
            },
        )
    if int(base_revision_no) != current:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "base_revision_no is not the current annotation revision",
            details={
                "annotation_set_id": annotation_set.id,
                "current_revision_no": current,
                "base_revision_no": int(base_revision_no),
            },
        )

    cleaned_note = (note or "").strip()
    if not (NOTE_MIN_CHARS <= len(cleaned_note) <= NOTE_MAX_CHARS):
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            f"note must be {NOTE_MIN_CHARS}..{NOTE_MAX_CHARS} characters after trimming",
            details={"note_min_chars": NOTE_MIN_CHARS, "note_max_chars": NOTE_MAX_CHARS, "note_chars": len(cleaned_note)},
        )

    frame_errors = validate_replacement_frames(frames)
    if frame_errors:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "Replacement frames violate annotation-revision/2.0",
            details={"errors": frame_errors[:20]},
        )
    replacement_frames = normalise_replacement_frames(frames)

    base_row = session.scalars(
        select(AnnotationRevision).where(
            AnnotationRevision.annotation_set_id == annotation_set.id,
            AnnotationRevision.revision_no == current,
        )
    ).first()
    if base_row is None:
        if current != 1:
            raise ApiError(
                ErrorCode.INTERNAL_ERROR,
                "AnnotationSet.current_revision_no has no stored revision row",
                details={"annotation_set_id": annotation_set.id, "current_revision_no": current},
            )
        # A v1 set entering the revision workflow: seed the immutable baseline first.
        base_row = ensure_baseline_revision(session, storage=storage, annotation_set=annotation_set)
    base_document = load_revision_document(session, storage, base_row)

    new_revision_no = current + 1
    merged = merge_replacement_frames(
        base_document, replacement_frames, revision_no=new_revision_no
    )
    merged["annotation_set_id"] = annotation_set.id
    problems = validate_revision_document(merged)
    if problems:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "Merged annotation revision violates annotation-revision/2.0",
            details={"errors": problems[:20]},
        )
    if merged.get("frames") == base_document.get("frames"):
        raise ApiError(
            ErrorCode.NO_CHANGE,
            "No effective frame change; nothing to save as a new annotation revision",
            details={"annotation_set_id": annotation_set.id, "current_revision_no": current},
        )

    existing_target = session.scalars(
        select(AnnotationRevision.id).where(
            AnnotationRevision.annotation_set_id == annotation_set.id,
            AnnotationRevision.revision_no == new_revision_no,
        )
    ).first()
    if existing_target is not None:  # pragma: no cover - CAS makes this unreachable
        raise ApiError(
            ErrorCode.CONFLICT,
            "Annotation revision number is already taken",
            details={"annotation_set_id": annotation_set.id, "revision_no": new_revision_no},
        )

    payload = canonical_json_bytes(merged)
    digest = hashlib.sha256(payload).hexdigest()
    asset = _store_revision_asset(
        session,
        storage,
        annotation_set=annotation_set,
        revision_no=new_revision_no,
        payload=payload,
    )
    revision = AnnotationRevision(
        id=new_id(),
        annotation_set_id=annotation_set.id,
        revision_no=new_revision_no,
        base_revision_id=base_row.id,
        artifact_asset_id=asset.id,
        content_sha256=digest,
        state=STATE_IN_REVIEW,
        created_by=actor.key_id,
        created_at=utcnow(),
    )
    session.add(revision)
    session.flush()
    change_summary = build_change_summary(
        base_document,
        merged,
        replacement_frames,
        note=cleaned_note,
        base_revision_no=current,
    )
    edit = AnnotationRevisionEdit(
        id=new_id(),
        revision_id=revision.id,
        actor_key_id=actor.key_id,
        change_summary_json=change_summary,
        before_sha256=base_row.content_sha256,
        after_sha256=digest,
        created_at=utcnow(),
    )
    session.add(edit)
    annotation_set.current_revision_no = new_revision_no
    _bump_episode_business_revision(session, annotation_set)
    session.flush()
    return CreatedRevision(revision=revision, edit=edit, change_summary=change_summary)


@dataclass
class DecidedRevision:
    revision: AnnotationRevision
    annotation_set: AnnotationSet
    previous_state: str
    note: str


def decide_annotation_revision(
    session: Session,
    *,
    revision_id: str,
    decision: Any,
    note: str,
    reviewer: Any,
) -> DecidedRevision:
    """Approve/reject one ``IN_REVIEW`` revision (two-person approval for APPROVED)."""
    decision_value = decision.value if hasattr(decision, "value") else str(decision)
    if decision_value not in (STATE_APPROVED, STATE_REJECTED):
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "decision must be APPROVED or REJECTED",
            details={"decision": decision_value},
        )
    cleaned_note = (note or "").strip()
    if decision_value == STATE_APPROVED and len(cleaned_note) < APPROVAL_NOTE_MIN_CHARS:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            f"Approving an annotation revision requires a note of at least {APPROVAL_NOTE_MIN_CHARS} characters",
            status_code=422,
            details={"note_min_chars": APPROVAL_NOTE_MIN_CHARS, "note_chars": len(cleaned_note)},
        )

    revision = session.execute(
        select(AnnotationRevision)
        .where(AnnotationRevision.id == revision_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if revision is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            f"AnnotationRevision {revision_id} not found",
            details={"revision_id": revision_id},
        )
    annotation_set = session.execute(
        select(AnnotationSet)
        .where(AnnotationSet.id == revision.annotation_set_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if annotation_set is None:  # pragma: no cover - FK protects this
        raise ApiError(ErrorCode.NOT_FOUND, "AnnotationSet for the revision is missing")

    previous_state = revision.state
    if previous_state != STATE_IN_REVIEW:
        raise ApiError(
            ErrorCode.ANNOTATION_ALREADY_REVIEWED,
            "Annotation revision has already been decided; revisions are immutable",
            details={
                "revision_id": revision.id,
                "revision_no": revision.revision_no,
                "state": previous_state,
            },
        )

    if decision_value == STATE_APPROVED:
        if revision.created_by is not None and revision.created_by == reviewer.key_id:
            raise ApiError(
                ErrorCode.FORBIDDEN,
                "Approval requires a reviewer different from the revision creator",
                details={
                    "reason": "SELF_APPROVAL_FORBIDDEN",
                    "revision_id": revision.id,
                    "created_by": revision.created_by,
                },
            )
        revision.state = STATE_APPROVED
        revision.approved_by = reviewer.key_id
        revision.approved_at = utcnow()
        # The only publication pointer; older approved content stays referenced.
        annotation_set.approved_revision_id = revision.id
        annotation_set.status = AnnotationStatus.APPROVED.value
    else:
        revision.state = STATE_REJECTED
    _bump_episode_business_revision(session, annotation_set)
    session.flush()
    return DecidedRevision(
        revision=revision,
        annotation_set=annotation_set,
        previous_state=previous_state,
        note=cleaned_note,
    )


# ---------------------------------------------------------------------------
# view helpers (JSON-safe; also persisted as the Idempotency-Key response)
# ---------------------------------------------------------------------------
def revision_view(session: Session, revision: AnnotationRevision) -> dict[str, Any]:
    edit = session.scalars(
        select(AnnotationRevisionEdit).where(AnnotationRevisionEdit.revision_id == revision.id)
    ).first()
    asset = session.get(Asset, revision.artifact_asset_id) if revision.artifact_asset_id else None
    summary = dict(edit.change_summary_json) if edit is not None and edit.change_summary_json else None
    return {
        "revision_id": revision.id,
        "annotation_set_id": revision.annotation_set_id,
        "revision_no": int(revision.revision_no),
        "state": revision.state,
        "base_revision_id": revision.base_revision_id,
        "content_sha256": revision.content_sha256,
        "artifact_sha256": asset.sha256 if asset is not None else None,
        "created_by": revision.created_by,
        "created_at": revision.created_at.isoformat() if revision.created_at else None,
        "approved_by": revision.approved_by,
        "approved_at": revision.approved_at.isoformat() if revision.approved_at else None,
        "change_summary": summary,
        "note": (summary or {}).get("note") if summary else None,
    }


def baseline_view(
    annotation_set: AnnotationSet, document: Mapping[str, Any], digest: str
) -> dict[str, Any]:
    """Read-only view of the implicit revision 1 before it is materialised."""
    return {
        "revision_id": None,
        "annotation_set_id": annotation_set.id,
        "revision_no": 1,
        "state": STATE_IN_REVIEW,
        "base_revision_id": None,
        "content_sha256": digest,
        "artifact_sha256": None,
        "created_by": None,
        "created_at": None,
        "approved_by": None,
        "approved_at": None,
        "change_summary": None,
        "note": None,
    }


__all__ = [
    "APPROVAL_NOTE_MIN_CHARS",
    "COCO_CLASS_COUNT",
    "CreatedRevision",
    "DEFAULT_CAMERA_NAME",
    "DecidedRevision",
    "DETECTION_ORIGINS",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "MAX_DETECTIONS_PER_FRAME",
    "MAX_FRAMES",
    "NOTE_MAX_CHARS",
    "NOTE_MIN_CHARS",
    "REVISION_SCHEMA_VERSION",
    "STATE_APPROVED",
    "STATE_IN_REVIEW",
    "STATE_REJECTED",
    "annotation_set_has_revisions",
    "baseline_view",
    "build_change_summary",
    "build_revision_document",
    "create_annotation_revision",
    "decide_annotation_revision",
    "ensure_baseline_revision",
    "get_annotation_set",
    "get_revision",
    "get_revision_document",
    "list_annotation_revisions",
    "load_revision_document",
    "merge_replacement_frames",
    "normalise_replacement_frames",
    "parse_revision_if_match",
    "resolve_approved_revision_id",
    "revision_etag",
    "revision_object_key",
    "revision_view",
    "validate_replacement_frames",
    "validate_revision_document",
]
