"""``ACTION_INTERVAL`` annotation revisions (SPEC_V2.1 §14, V21-14).

Ego video produces **action intervals**, not object boxes, so it needs its own revision
document.  The invariants are the same as the V2 detection revisions, because they are
about people rather than about payloads:

* the model's original answer is never rewritten -- a human edit creates a *new*
  immutable revision;
* the writer must hold a strong ``If-Match`` on the current revision number (CAS);
* the editor and the approver must be different people.

The document is deliberately flat: a list of intervals with their evidence frames.  Any
richer structure (segments, phases) can be projected from it later; the reverse is not
true.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import (
    AnnotationRevision,
    AnnotationRevisionEdit,
    AnnotationSet,
    Asset,
    Episode,
)
from embodied_dataops.domain.enums import AssetOrigin
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id

ACTION_DOCUMENT_SCHEMA = "annotation-action-interval/2.1"
MAX_ACTIONS = 200


@dataclass(frozen=True)
class CreatedActionRevision:
    annotation_set: AnnotationSet
    revision: AnnotationRevision
    edit: AnnotationRevisionEdit
    asset: Asset
    action_count: int


def action_document(
    *,
    episode_id: str,
    annotation_set_id: str,
    actions: Sequence[dict[str, Any]],
    revision_no: int,
    note: str,
) -> dict[str, Any]:
    """The immutable content of one action-interval revision."""
    ordered = sorted(
        (dict(item) for item in actions),
        key=lambda item: (int(item["start_ns"]), int(item["end_ns"]), str(item["verb"])),
    )
    return {
        "schema_version": ACTION_DOCUMENT_SCHEMA,
        "episode_id": episode_id,
        "annotation_set_id": annotation_set_id,
        "annotation_kind": "ACTION_INTERVAL",
        "revision_no": int(revision_no),
        "note": str(note or ""),
        "actions": ordered,
        "stats": {"actions": len(ordered)},
    }


def validate_actions(actions: Sequence[dict[str, Any]]) -> list[str]:
    """Structural checks that a JSON patch can still get wrong."""
    errors: list[str] = []
    if len(actions) > MAX_ACTIONS:
        errors.append(f"{len(actions)} actions exceeds {MAX_ACTIONS}")
    for index, action in enumerate(actions):
        start = int(action.get("start_ns", -1))
        end = int(action.get("end_ns", -1))
        if start < 0:
            errors.append(f"actions[{index}].start_ns must be >= 0")
        if end <= start:
            errors.append(f"actions[{index}].end_ns must be greater than start_ns")
        frames = action.get("evidence_frame_indices") or []
        if not frames:
            errors.append(f"actions[{index}] needs at least one evidence frame")
        elif any(int(frame) < 0 for frame in frames):
            errors.append(f"actions[{index}] has a negative evidence frame")
    # Overlaps are allowed (two hands can act at once), but a duplicated interval is a
    # sign the client resent a patch instead of replacing the list.
    seen: set[tuple[int, int, str, str]] = set()
    for action in actions:
        key = (
            int(action.get("start_ns", 0)),
            int(action.get("end_ns", 0)),
            str(action.get("verb", "")),
            str(action.get("noun", "")),
        )
        if key in seen:
            errors.append(f"duplicate interval {key[2]}/{key[3]} at {key[0]}-{key[1]}")
        seen.add(key)
    return errors


def create_action_revision(
    session,
    *,
    storage,
    annotation_set_id: str,
    expected_revision_no: int,
    actions: Sequence[dict[str, Any]],
    note: str,
    actor: Any,
) -> CreatedActionRevision:
    """CAS-append the next ACTION_INTERVAL revision for one annotation set."""
    errors = validate_actions(actions)
    if errors:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "action interval document is invalid",
            details={"errors": errors[:20]},
        )

    annotation_set = session.execute(
        select(AnnotationSet)
        .where(AnnotationSet.id == annotation_set_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if annotation_set is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "annotation set not found",
            details={"annotation_set_id": annotation_set_id},
        )
    current = int(annotation_set.current_revision_no or 1)
    if int(expected_revision_no) != current:
        raise ApiError(
            ErrorCode.PRECONDITION_FAILED,
            "annotation set moved since the client read it",
            details={"expected_revision_no": int(expected_revision_no), "current_revision_no": current},
        )
    episode = session.get(Episode, annotation_set.episode_id)
    if episode is None:  # pragma: no cover - FK guarantees existence
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "episode not found for this annotation set",
            details={"annotation_set_id": annotation_set_id},
        )

    next_no = current + 1
    latest = session.scalars(
        select(AnnotationRevision)
        .where(AnnotationRevision.annotation_set_id == annotation_set_id)
        .order_by(AnnotationRevision.revision_no.desc())
    ).first()
    document = action_document(
        episode_id=episode.id,
        annotation_set_id=annotation_set_id,
        actions=actions,
        revision_no=next_no,
        note=note,
    )
    data = canonical_json_bytes(document)
    digest = hashlib.sha256(data).hexdigest()
    key = f"annotations/{episode.id}/{annotation_set_id}/action_revision_{next_no:04d}.json"
    stored = storage.put_bytes(key, data, content_type="application/json")
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or digest,
        byte_size=len(data),
        media_type="application/json",
        object_key=key,
        origin=AssetOrigin.DERIVED.value,
        logical_name=f"action_revision_{next_no:04d}",
        meta_json={
            "kind": "annotation_action_interval",
            "schema_version": ACTION_DOCUMENT_SCHEMA,
            "revision_no": next_no,
            "action_count": len(document["actions"]),
        },
    )
    session.add(asset)
    session.flush()

    before_sha = latest.content_sha256 if latest is not None else annotation_set.model_sha256
    revision = AnnotationRevision(
        id=new_id(),
        annotation_set_id=annotation_set_id,
        revision_no=next_no,
        base_revision_id=latest.id if latest is not None else None,
        artifact_asset_id=asset.id,
        content_sha256=digest,
        state="IN_REVIEW",
        created_by=getattr(actor, "key_id", None),
    )
    session.add(revision)
    session.flush()

    edit = AnnotationRevisionEdit(
        id=new_id(),
        revision_id=revision.id,
        actor_key_id=getattr(actor, "key_id", None) or getattr(actor, "id", ""),
        change_summary_json={
            "kind": "ACTION_INTERVAL",
            "note": str(note or ""),
            "actions_before": 0,
            "actions_after": len(document["actions"]),
            "base_revision_no": current,
        },
        before_sha256=before_sha or "",
        after_sha256=digest,
    )
    session.add(edit)

    annotation_set.current_revision_no = next_no
    annotation_set.status = "PENDING_REVIEW"
    session.flush()

    return CreatedActionRevision(
        annotation_set=annotation_set,
        revision=revision,
        edit=edit,
        asset=asset,
        action_count=len(document["actions"]),
    )


def latest_action_revision(session, annotation_set_id: str) -> AnnotationRevision | None:
    return session.scalars(
        select(AnnotationRevision)
        .where(AnnotationRevision.annotation_set_id == annotation_set_id)
        .order_by(AnnotationRevision.revision_no.desc())
    ).first()


__all__ = [
    "ACTION_DOCUMENT_SCHEMA",
    "CreatedActionRevision",
    "action_document",
    "create_action_revision",
    "latest_action_revision",
    "validate_actions",
]
