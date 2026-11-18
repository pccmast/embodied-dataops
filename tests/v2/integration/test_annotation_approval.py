"""DO2-16 acceptance: two-person annotation revision approval (SPEC §4.2).

Flow exercised through the public API:

1. reviewer A creates an immutable ``annotation-revision/2.0`` revision 2;
2. reviewer A tries to approve their own revision -> rejected with
   ``403 FORBIDDEN`` + ``SELF_APPROVAL_FORBIDDEN`` and the approval pointer stays
   ``NULL`` (the annotation set cannot become publishable through self-approval);
3. reviewer B must supply an explanatory note (``422`` when too short) and then
   approves successfully: revision state ``APPROVED``, approved_by/approved_at
   recorded, ``annotation_sets.approved_revision_id`` switches to that revision
   and the set becomes ``APPROVED`` for dataset eligibility;
4. the approved revision is immutable: a second decision returns
   ``409 ANNOTATION_ALREADY_REVIEWED`` and the stored artifact bytes are
   byte-identical; a newer revision does not silently withdraw the approval.
"""

from __future__ import annotations

import uuid

import pytest
from _annotation_fixtures import (
    auth_headers,
    detection,
    key_id_for,
    make_principal,
    revision_frame,
    seed_annotation_set,
    seed_episode,
)
from sqlalchemy import select

from embodied_dataops.annotations.revisions import (
    APPROVAL_NOTE_MIN_CHARS,
    REVISION_SCHEMA_VERSION,
    annotation_set_has_revisions,
)
from embodied_dataops.db.models import (
    AnnotationRevision,
    AnnotationSet,
    Asset,
    AuditEvent,
)
from embodied_dataops.errors import ErrorCode

pytestmark = pytest.mark.integration


def _create_revision_as_a(client, headers, annotation_set_id: str) -> dict:
    response = client.post(
        f"/api/v1/annotation-sets/{annotation_set_id}/revisions",
        headers={
            **headers,
            "If-Match": '"annotation-rev:1"',
            "Idempotency-Key": str(uuid.uuid4()),
        },
        json={
            "base_revision_no": 1,
            "frames": [
                revision_frame(
                    0,
                    detections=[
                        detection(
                            "human-0-000",
                            class_id=0,
                            class_name="person",
                            score=0.88,
                            xyxy=[15.0, 25.0, 180.0, 320.0],
                        )
                    ],
                )
            ],
            "note": "reviewer A corrected the frame 0 person box",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]


def _decide(client, headers, revision_id: str, decision: str, note: str):
    return client.post(
        f"/api/v1/annotation-revisions/{revision_id}/decision",
        headers=headers,
        json={"decision": decision, "note": note},
    )


def test_creator_cannot_self_approve_and_second_reviewer_can(client, db, storage, reviewer_headers) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    # reviewer B is a second, distinct reviewer identity (single-machine
    # bootstrap creates two reviewer keys, see cli/keys.py).
    reviewer_b, reviewer_b_raw = make_principal(db, "reviewer", name="acceptance-reviewer-b")
    db.commit()
    reviewer_b_headers = auth_headers(reviewer_b_raw)
    creator_key_id = key_id_for(db, reviewer_headers["Authorization"].split(" ", 1)[1])

    created = _create_revision_as_a(client, reviewer_headers, annotation_set.id)
    revision_id = created["revision"]["revision_id"]
    revision_no = created["revision"]["revision_no"]
    assert revision_no == 2
    assert created["revision"]["state"] == "IN_REVIEW"
    artifact_sha_before = created["revision"]["artifact_sha256"]

    # 1) self-approval is refused and changes nothing.
    self_approval = _decide(
        client,
        reviewer_headers,
        revision_id,
        "APPROVED",
        "I reviewed my own correction and it is fine",
    )
    assert self_approval.status_code == 403, self_approval.text
    error = self_approval.json()["error"]
    assert error["code"] == ErrorCode.FORBIDDEN.value
    assert error["details"]["reason"] == "SELF_APPROVAL_FORBIDDEN"
    assert error["details"]["created_by"] == creator_key_id
    db.expire_all()
    set_row = db.get(AnnotationSet, annotation_set.id)
    assert set_row.approved_revision_id is None
    assert set_row.status == "PENDING_REVIEW"
    revision_row = db.get(AnnotationRevision, revision_id)
    assert revision_row.state == "IN_REVIEW"
    assert revision_row.approved_by is None and revision_row.approved_at is None

    # 2) the second reviewer must justify the approval.
    too_short = _decide(client, reviewer_b_headers, revision_id, "APPROVED", "ok")
    assert too_short.status_code == 422, too_short.text
    assert too_short.json()["error"]["code"] == ErrorCode.VALIDATION_FAILED.value
    assert too_short.json()["error"]["details"]["note_min_chars"] == APPROVAL_NOTE_MIN_CHARS
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).approved_revision_id is None

    # 3) reviewer B approves: the explicit pointer switches to this revision.
    approval_note = "reviewer B verified the corrected box against the preview frame"
    approved = _decide(client, reviewer_b_headers, revision_id, "APPROVED", approval_note)
    assert approved.status_code == 200, approved.text
    data = approved.json()["data"]
    assert data["revision"]["state"] == "APPROVED"
    assert data["revision"]["revision_id"] == revision_id
    # the revision keeps the edit note; the decision note lives in the audit row
    assert data["revision"]["note"] == "reviewer A corrected the frame 0 person box"
    assert data["approved_revision_id"] == revision_id
    assert data["status"] == "APPROVED"
    assert data["current_revision_no"] == revision_no

    db.expire_all()
    set_row = db.get(AnnotationSet, annotation_set.id)
    assert set_row.approved_revision_id == revision_id
    assert set_row.status == "APPROVED"
    assert set_row.current_revision_no == revision_no
    revision_row = db.get(AnnotationRevision, revision_id)
    assert revision_row.state == "APPROVED"
    assert revision_row.approved_by == reviewer_b.key_id
    assert revision_row.approved_at is not None
    assert revision_row.created_by is not None
    assert revision_row.approved_by != revision_row.created_by
    assert revision_row.created_by != reviewer_b.key_id

    # audit trail names the acting reviewer and the new pointer.
    audit = list(db.scalars(select(AuditEvent).where(AuditEvent.action == "annotation_revision.approved")))
    assert len(audit) == 1
    assert audit[0].entity_id == revision_id
    assert audit[0].actor_id == reviewer_b.key_id
    assert audit[0].before_json == {"state": "IN_REVIEW"}
    assert audit[0].after_json["state"] == "APPROVED"
    assert audit[0].after_json["annotation_set_id"] == annotation_set.id
    assert audit[0].after_json["approved_revision_id"] == revision_id
    assert audit[0].after_json["note"] == approval_note

    # 4) approved content is immutable: a second decision is a 409 and the
    # stored artifact bytes never change.
    repeat = _decide(client, reviewer_b_headers, revision_id, "REJECTED", "changed my mind")
    assert repeat.status_code == 409, repeat.text
    assert repeat.json()["error"]["code"] == ErrorCode.ANNOTATION_ALREADY_REVIEWED.value
    db.expire_all()
    assert db.get(AnnotationRevision, revision_id).state == "APPROVED"
    assert db.get(AnnotationSet, annotation_set.id).approved_revision_id == revision_id
    asset = db.get(Asset, db.get(AnnotationRevision, revision_id).artifact_asset_id)
    assert asset is not None
    assert storage.get_bytes(asset.object_key)
    assert asset.sha256 == artifact_sha_before
    content = client.get(
        f"/api/v1/annotation-sets/{annotation_set.id}/revisions/{revision_no}",
        headers=reviewer_headers,
    )
    assert content.status_code == 200, content.text
    assert content.json()["data"]["content_sha256"] == revision_row.content_sha256
    assert content.json()["data"]["annotation"]["schema_version"] == REVISION_SCHEMA_VERSION
    assert content.json()["data"]["state"] == "APPROVED"

    # A newer revision created after approval does not withdraw the pointer; only
    # an explicit decision may switch it (SPEC §4.2).
    newer = client.post(
        f"/api/v1/annotation-sets/{annotation_set.id}/revisions",
        headers={
            **reviewer_headers,
            "If-Match": f'"annotation-rev:{revision_no}"',
            "Idempotency-Key": str(uuid.uuid4()),
        },
        json={
            "base_revision_no": revision_no,
            "frames": [
                revision_frame(
                    10,
                    detections=[
                        detection(
                            "human-10-000",
                            class_id=41,
                            class_name="cup",
                            score=0.7,
                            xyxy=[50.0, 120.0, 160.0, 240.0],
                        )
                    ],
                )
            ],
            "note": "another correction after approval",
        },
    )
    assert newer.status_code == 201, newer.text
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).approved_revision_id == revision_id
    assert db.get(AnnotationSet, annotation_set.id).status == "APPROVED"
    assert db.get(AnnotationRevision, revision_id).content_sha256 == revision_row.content_sha256
    assert db.scalars(select(AnnotationRevision)).all()  # baseline + edit + newer
    assert annotation_set_has_revisions(db, annotation_set.id)


def test_viewer_and_operator_cannot_approve_revisions(
    client, db, storage, reviewer_headers, viewer_headers, operator_headers
) -> None:
    episode, run, _report = seed_episode(db)
    annotation_set, _payload = seed_annotation_set(db, storage, episode, run)
    db.commit()
    created = _create_revision_as_a(client, reviewer_headers, annotation_set.id)
    revision_id = created["revision"]["revision_id"]
    for headers in (viewer_headers, operator_headers):
        response = _decide(client, headers, revision_id, "APPROVED", "no permission to approve")
        assert response.status_code == 403, response.text
    db.expire_all()
    assert db.get(AnnotationSet, annotation_set.id).approved_revision_id is None
    assert db.get(AnnotationRevision, revision_id).state == "IN_REVIEW"
