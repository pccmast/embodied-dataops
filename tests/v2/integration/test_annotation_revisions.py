"""V2 D2-M3: annotation revision creation, content contract and v1 compatibility.

Covers SPEC §4.2 / DO2-14:

* ``POST /annotation-sets/{id}/revisions`` creates immutable revision N+1 from a
  full baseline, replaces only the listed frames, copies every other frame,
  records the before/after SHA + change summary and CAS-updates
  ``current_revision_no``;
* ``annotation-revision/2.0`` content limits (600 frames, 100 detections/frame,
  640x480 strict xyxy, detector taxonomy/``other``, MODEL|HUMAN origin, note
  10..500) are enforced server-side;
* stale/weak/missing ``If-Match`` and the ``Idempotency-Key`` replay contract;
* ``GET`` list/content with the strong ``annotation-rev:<n>`` ETag;
* legacy v1 sets keep the old reviews endpoint, v2 sets return
  ``409 USE_REVISION_REVIEW_API``.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from _annotation_fixtures import (
    BASE_FRAME_10_IDS,
    FRAME_10_DETECTIONS,
    auth_headers,
    detection,
    key_id_for,
    revision_frame,
    seed_annotation_set,
    seed_episode,
)
from sqlalchemy import select

from embodied_dataops.annotations.revisions import (
    MAX_FRAMES,
    NOTE_MAX_CHARS,
    NOTE_MIN_CHARS,
    REVISION_SCHEMA_VERSION,
    STATE_IN_REVIEW,
    annotation_set_has_revisions,
    revision_etag,
    validate_revision_document,
)
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import (
    AnnotationReview,
    AnnotationRevision,
    AnnotationRevisionEdit,
    AnnotationSet,
    Asset,
    AuditEvent,
)
from embodied_dataops.errors import ErrorCode

pytestmark = pytest.mark.integration


def _idem() -> str:
    return str(uuid.uuid4())


def _revision_payload(
    frames: list[dict[str, Any]],
    *,
    base_revision_no: int = 1,
    note: str = "manual correction of the cup boxes",
) -> dict[str, Any]:
    return {"base_revision_no": base_revision_no, "frames": frames, "note": note}


def _post_revision(
    client,
    headers: dict[str, str],
    annotation_set_id: str,
    payload: dict[str, Any],
    *,
    if_match: str | None = '"annotation-rev:1"',
    idempotency_key: str | None = None,
):
    request_headers = dict(headers)
    if if_match is not None:
        request_headers["If-Match"] = if_match
    key = _idem() if idempotency_key is None else idempotency_key
    if key is not None:
        request_headers["Idempotency-Key"] = key
    return client.post(
        f"/api/v1/annotation-sets/{annotation_set_id}/revisions",
        headers=request_headers,
        json=payload,
    )


# ---------------------------------------------------------------------------
# DO2-14: first edit creates revision 2 and an immutable baseline revision 1
# ---------------------------------------------------------------------------
def test_first_edit_creates_immutable_revisions_with_change_summary(
    client, db, storage, reviewer_headers
) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, base_payload = seed_annotation_set(db, storage, episode, run)
    db.commit()

    replacement = revision_frame(
        0,
        detections=[
            detection(
                "human-0-000",
                class_id=0,
                class_name="person",
                score=None,
                xyxy=[20.0, 30.0, 140.0, 260.0],
            )
        ],
    )
    response = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload([replacement]),
    )
    assert response.status_code == 201, response.text
    data = response.json()["data"]
    assert data["annotation_set_id"] == annotation_set.id
    assert data["current_revision_no"] == 2
    assert data["etag"] == '"annotation-rev:2"'
    assert response.headers["etag"] == '"annotation-rev:2"'
    assert data["revision"]["revision_no"] == 2
    assert data["revision"]["state"] == STATE_IN_REVIEW
    assert data["revision"]["content_sha256"]
    assert data["revision"]["note"] == "manual correction of the cup boxes"
    summary = data["revision"]["change_summary"]
    assert summary["frames"]["effective"] == [0]
    assert summary["detections"]["added"] == 1
    assert summary["detections"]["removed"] == 1

    db.expire_all()
    stored_set = db.get(AnnotationSet, annotation_set.id)
    assert stored_set.current_revision_no == 2
    assert stored_set.status == "PENDING_REVIEW"  # an edit does not approve anything
    assert stored_set.approved_revision_id is None

    revisions = list(
        db.scalars(
            select(AnnotationRevision)
            .where(AnnotationRevision.annotation_set_id == annotation_set.id)
            .order_by(AnnotationRevision.revision_no)
        )
    )
    assert [row.revision_no for row in revisions] == [1, 2]
    baseline, edit_revision = revisions
    assert baseline.state == STATE_IN_REVIEW
    assert baseline.created_by is None  # model output baseline, no human creator
    assert baseline.base_revision_id is None
    assert edit_revision.created_by == key_id_for(db, reviewer_headers["Authorization"].split(" ", 1)[1])
    assert edit_revision.base_revision_id == baseline.id

    edit_rows = list(
        db.scalars(
            select(AnnotationRevisionEdit).where(
                AnnotationRevisionEdit.revision_id == edit_revision.id
            )
        )
    )
    assert len(edit_rows) == 1
    assert edit_rows[0].before_sha256 == baseline.content_sha256
    assert edit_rows[0].after_sha256 == edit_revision.content_sha256
    assert edit_rows[0].change_summary_json["note"] == "manual correction of the cup boxes"

    # The revision asset is canonical JSON whose SHA-256 matches the row.
    asset = db.get(Asset, edit_revision.artifact_asset_id)
    assert asset is not None
    raw = storage.get_bytes(asset.object_key)
    document = json.loads(raw.decode("utf-8"))
    assert canonical_json_bytes(document) == raw
    assert validate_revision_document(document) == []
    assert document["schema_version"] == REVISION_SCHEMA_VERSION
    assert document["revision_no"] == 2
    assert document["camera_name"] == "front"
    assert document["frame_width"] == 640
    assert document["frame_height"] == 480
    # frame 0 is fully replaced, frame 10 is copied from the base revision
    assert [frame["frame_index"] for frame in document["frames"]] == [0, 10]
    assert [item["detection_id"] for item in document["frames"][0]["detections"]] == ["human-0-000"]
    assert [item["detection_id"] for item in document["frames"][1]["detections"]] == BASE_FRAME_10_IDS
    assert document["frames"][1]["detections"] == [
        {**item, "detection_id": detection_id, "origin": "MODEL"}
        for item, detection_id in zip(FRAME_10_DETECTIONS, BASE_FRAME_10_IDS, strict=True)
    ]

    # The original annotation/1.0 artifact is immutable.
    assert storage.get_bytes(f"annotations/{episode.id}/{annotation_set.id}/annotation_candidates.json") == base_payload

    audit = list(
        db.scalars(select(AuditEvent).where(AuditEvent.action == "annotation_revision.created"))
    )
    assert len(audit) == 1
    assert audit[0].entity_id == edit_revision.id
    assert audit[0].after_json["revision_no"] == 2


def test_second_edit_replaces_another_frame_and_keeps_previous_edit(
    client, db, storage, reviewer_headers
) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()

    first = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload(
            [
                revision_frame(
                    0,
                    detections=[detection("human-0-000", score=0.8, origin="HUMAN")],
                )
            ]
        ),
    )
    assert first.status_code == 201, first.text

    second = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload(
            [revision_frame(10, detections=[])],
            base_revision_no=2,
            note="clear the two false positives in frame 10",
        ),
        if_match='"annotation-rev:2"',
    )
    assert second.status_code == 201, second.text
    body = second.json()["data"]
    assert body["current_revision_no"] == 3
    assert body["revision"]["change_summary"]["frames"]["effective"] == [10]
    assert body["revision"]["change_summary"]["detections"]["removed"] == 2

    content = client.get(
        f"/api/v1/annotation-sets/{annotation_set.id}/revisions/3", headers=reviewer_headers
    )
    assert content.status_code == 200, content.text
    assert content.headers["etag"] == '"annotation-rev:3"'
    document = content.json()["data"]["annotation"]
    frames = {frame["frame_index"]: frame for frame in document["frames"]}
    assert [item["detection_id"] for item in frames[0]["detections"]] == ["human-0-000"]
    assert frames[10]["detections"] == []


# ---------------------------------------------------------------------------
# GET list/content and the implicit baseline revision 1
# ---------------------------------------------------------------------------
def test_get_revisions_exposes_implicit_baseline_and_strong_etag(
    client, db, storage, viewer_headers
) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    assert not annotation_set_has_revisions(db, annotation_set.id)

    listing = client.get(
        f"/api/v1/annotation-sets/{annotation_set.id}/revisions", headers=viewer_headers
    )
    assert listing.status_code == 200, listing.text
    data = listing.json()["data"]
    assert data["annotation_set_id"] == annotation_set.id
    assert data["current_revision_no"] == 1
    assert data["approved_revision_id"] is None
    assert data["status"] == "PENDING_REVIEW"
    assert data["revisions"] == []

    baseline = client.get(
        f"/api/v1/annotation-sets/{annotation_set.id}/revisions/1", headers=viewer_headers
    )
    assert baseline.status_code == 200, baseline.text
    assert baseline.headers["etag"] == revision_etag(1)
    body = baseline.json()["data"]
    assert body["revision_id"] is None
    assert body["state"] == STATE_IN_REVIEW
    document = body["annotation"]
    assert document["schema_version"] == REVISION_SCHEMA_VERSION
    assert document["revision_no"] == 1
    assert [frame["frame_index"] for frame in document["frames"]] == [0, 10]
    assert [item["detection_id"] for item in document["frames"][0]["detections"]] == ["model-0-000"]
    assert document["frames"][0]["detections"][0]["origin"] == "MODEL"

    assert (
        client.get(
            f"/api/v1/annotation-sets/{annotation_set.id}/revisions", headers=auth_headers("")
        ).status_code
        == 401
    )
    assert (
        client.get(
            f"/api/v1/annotation-sets/{uuid.uuid4()}/revisions", headers=viewer_headers
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/v1/annotation-sets/{annotation_set.id}/revisions/9", headers=viewer_headers
        ).status_code
        == 404
    )


def test_viewer_cannot_create_or_decide_revisions(
    client, db, storage, viewer_headers, operator_headers
) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    payload = _revision_payload(
        [revision_frame(0, detections=[detection("human-0-000", origin="HUMAN")])]
    )
    for headers in (viewer_headers, operator_headers):
        response = _post_revision(client, headers, annotation_set.id, payload)
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == ErrorCode.FORBIDDEN.value
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).current_revision_no == 1
    assert db.scalars(select(AnnotationRevision)).all() == []


# ---------------------------------------------------------------------------
# validation (SPEC §4.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "frame,expected_fragment",
    [
        (
            revision_frame(
                0,
                detections=[detection("human-0-000", xyxy=[10.0, 10.0, 10.0, 90.0])],
            ),
            "x1 < x2",
        ),
        (
            revision_frame(
                0,
                detections=[detection("human-0-000", xyxy=[10.0, 10.0, 641.0, 90.0])],
            ),
            "x2 <=",
        ),
        (
            revision_frame(
                0,
                detections=[detection("human-0-000", xyxy=[10.0, 10.0, 120.0, 481.0])],
            ),
            "y2 <=",
        ),
        (
            revision_frame(
                0,
                detections=[detection("human-0-000", origin="ROBOT")],
            ),
            "decision",
        ),
        (
            revision_frame(
                0,
                detections=[
                    detection(
                        f"human-0-{index:03d}",
                        class_id=41,
                        class_name="banana",
                    )
                    for index in range(1)
                ],
            ),
            "taxonomy",
        ),
        (
            revision_frame(
                0,
                detections=[
                    detection("human-0-000"),
                    detection("human-0-000"),
                ],
            ),
            "duplicated",
        ),
        (
            revision_frame(
                0,
                detections=[
                    detection(f"human-0-{index:03d}") for index in range(101)
                ],
            ),
            "detections",
        ),
    ],
)
def test_invalid_revision_frames_are_rejected_without_side_effects(
    client, db, storage, reviewer_headers, frame: dict[str, Any], expected_fragment: str
) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload([frame]),
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == ErrorCode.VALIDATION_FAILED.value
    db.expire_all()
    stored = db.get(AnnotationSet, annotation_set.id)
    assert stored.current_revision_no == 1
    assert stored.status == "PENDING_REVIEW"
    assert db.scalars(select(AnnotationRevision)).all() == []


@pytest.mark.parametrize("note", ["too short", "x" * (NOTE_MAX_CHARS + 1)])
def test_note_length_is_enforced(client, db, storage, reviewer_headers, note: str) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload(
            [revision_frame(0, detections=[detection("human-0-000")])],
            note=note,
        ),
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == ErrorCode.VALIDATION_FAILED.value
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).current_revision_no == 1


def test_note_boundaries_are_accepted(client, db, storage, reviewer_headers) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    for index, note in enumerate(("x" * NOTE_MIN_CHARS, "y" * NOTE_MAX_CHARS)):
        response = _post_revision(
            client,
            reviewer_headers,
            annotation_set.id,
            _revision_payload(
                [
                    revision_frame(
                        0,
                        detections=[detection(f"human-0-{index:03d}", origin="HUMAN")],
                    )
                ],
                base_revision_no=index + 1,
                note=note,
            ),
            if_match=f'"annotation-rev:{index + 1}"',
        )
        assert response.status_code == 201, response.text


def test_more_than_600_frames_after_merge_is_422(client, db, storage, reviewer_headers) -> None:
    episode, run, _report = seed_episode(db, sample_count=7000)
    frames = [
        {"frame_index": index, "t_ns": index * 100_000_000, "detections": []}
        for index in range(MAX_FRAMES)
    ]
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run, frames=frames)
    db.commit()

    response = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload(
            [revision_frame(MAX_FRAMES, detections=[detection("human-0700-000")])]
        ),
    )
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == ErrorCode.VALIDATION_FAILED.value
    assert any(str(MAX_FRAMES) in item for item in error["details"]["errors"])
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).current_revision_no == 1
    assert db.scalars(select(AnnotationRevision)).all() == []


# ---------------------------------------------------------------------------
# conditional write semantics (If-Match + Idempotency-Key)
# ---------------------------------------------------------------------------
def _create_first_revision(client, db, storage, reviewer_headers):
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload(
            [revision_frame(0, detections=[detection("human-0-000", origin="HUMAN")])]
        ),
    )
    assert response.status_code == 201, response.text
    return episode, annotation_set, response.json()["data"]


def test_stale_if_match_is_412_and_never_overwrites(
    client, db, storage, reviewer_headers
) -> None:
    _episode, annotation_set, _data = _create_first_revision(
        client, db, storage, reviewer_headers
    )
    response = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload(
            [revision_frame(0, detections=[detection("human-0-999", origin="HUMAN")])],
            base_revision_no=1,
            note="a stale edit that must not win",
        ),
        if_match='"annotation-rev:1"',
    )
    assert response.status_code == 412, response.text
    error = response.json()["error"]
    assert error["code"] == ErrorCode.PRECONDITION_FAILED.value
    assert error["details"]["current_revision_no"] == 2
    db.expire_all()
    stored = db.get(AnnotationSet, annotation_set.id)
    assert stored.current_revision_no == 2
    rows = db.scalars(select(AnnotationRevision)).all()
    assert [row.revision_no for row in rows] == [1, 2]


def test_weak_etag_is_412_and_missing_etag_is_428(
    client, db, storage, reviewer_headers
) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    payload = _revision_payload(
        [revision_frame(0, detections=[detection("human-0-000", origin="HUMAN")])]
    )
    weak = _post_revision(
        client, reviewer_headers, annotation_set.id, payload, if_match='W/"annotation-rev:1"'
    )
    assert weak.status_code == 412, weak.text
    assert weak.json()["error"]["code"] == ErrorCode.PRECONDITION_FAILED.value
    assert weak.json()["error"]["details"]["reason"] == "weak_etag_not_allowed"

    missing = _post_revision(
        client, reviewer_headers, annotation_set.id, payload, if_match=None
    )
    assert missing.status_code == 428, missing.text
    assert missing.json()["error"]["code"] == ErrorCode.PRECONDITION_REQUIRED.value
    db.expire_all()
    assert db.scalars(select(AnnotationRevision)).all() == []


def test_missing_idempotency_key_is_rejected(client, db, storage, reviewer_headers) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    missing_headers = dict(reviewer_headers)
    missing_headers["If-Match"] = '"annotation-rev:1"'
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/revisions",
        headers=missing_headers,
        json=_revision_payload(
            [revision_frame(0, detections=[detection("human-0-000", origin="HUMAN")])]
        ),
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == ErrorCode.IDEMPOTENCY_KEY_REQUIRED.value


def test_idempotent_replay_returns_the_same_revision(client, db, storage, reviewer_headers) -> None:
    _episode, annotation_set, first = _create_first_revision(
        client, db, storage, reviewer_headers
    )
    key = _idem()
    payload = _revision_payload(
        [revision_frame(0, detections=[detection("human-0-777", origin="HUMAN")])],
        base_revision_no=2,
        note="idempotent second edit",
    )
    created = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        payload,
        if_match='"annotation-rev:2"',
        idempotency_key=key,
    )
    assert created.status_code == 201, created.text
    first_data = created.json()["data"]
    assert first_data["current_revision_no"] == 3

    replay = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        payload,
        if_match='"annotation-rev:2"',
        idempotency_key=key,
    )
    assert replay.status_code == 201, replay.text
    replay_data = replay.json()["data"]
    assert replay_data["revision"]["revision_id"] == first_data["revision"]["revision_id"]
    assert replay_data["current_revision_no"] == 3
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).current_revision_no == 3
    assert len(db.scalars(select(AnnotationRevision)).all()) == 3

    reused = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        {**payload, "note": "different body under the same key"},
        if_match='"annotation-rev:3"',
        idempotency_key=key,
    )
    assert reused.status_code == 409, reused.text
    assert reused.json()["error"]["code"] == ErrorCode.IDEMPOTENCY_KEY_REUSED.value


def test_no_effective_change_is_422_no_change(client, db, storage, reviewer_headers) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    unchanged = revision_frame(
        0,
        detections=[detection("model-0-000", class_id=41, class_name="cup", score=0.9, xyxy=[10.0, 10.0, 120.0, 90.0], origin="MODEL")],
    )
    response = _post_revision(
        client,
        reviewer_headers,
        annotation_set.id,
        _revision_payload([unchanged], note="same frame identical to the base"),
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == ErrorCode.NO_CHANGE.value
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).current_revision_no == 1
    assert db.scalars(select(AnnotationRevision)).all() == []


# ---------------------------------------------------------------------------
# v1 review endpoint compatibility (SPEC §4.2)
# ---------------------------------------------------------------------------
def test_legacy_v1_set_still_uses_the_reviews_endpoint(
    client, db, storage, reviewer_headers
) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED", "note": "legacy single-reviewer approval"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["data"]["new_status"] == "APPROVED"
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "APPROVED"


def test_v2_set_refuses_the_legacy_reviews_endpoint(client, db, storage, reviewer_headers) -> None:
    _episode, annotation_set, _data = _create_first_revision(
        client, db, storage, reviewer_headers
    )
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/reviews",
        headers=reviewer_headers,
        json={"decision": "APPROVED", "note": "should be refused"},
    )
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == ErrorCode.USE_REVISION_REVIEW_API.value
    assert error["details"]["annotation_set_id"] == annotation_set.id
    assert error["details"]["current_revision_no"] == 2
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).status == "PENDING_REVIEW"
    assert db.scalars(select(AnnotationReview)).all() == []


def test_v1_episode_annotations_payload_exposes_revision_metadata(
    client, db, storage, viewer_headers, reviewer_headers
) -> None:
    """The legacy GET keeps its shape and adds the v2 revision pointers."""
    _episode, annotation_set, data = _create_first_revision(
        client, db, storage, reviewer_headers
    )
    response = client.get(
        f"/api/v1/episodes/{annotation_set.episode_id}/annotations", headers=viewer_headers
    )
    assert response.status_code == 200, response.text
    payload = response.json()["data"]["annotation_set"]
    assert payload["annotation_set_id"] == annotation_set.id
    assert payload["current_revision_no"] == 2
    assert payload["revision_count"] == 2
    assert payload["approved_revision_id"] is None
    assert data["revision"]["revision_no"] == 2
