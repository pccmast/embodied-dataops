"""V2 annotation revision API (SPEC §4.2, DO2-14..17).

* ``GET  /api/v1/annotation-sets/{id}/revisions``            -- viewer
* ``GET  /api/v1/annotation-sets/{id}/revisions/{no}``       -- viewer, strong
  ``ETag: "annotation-rev:<no>"`` header
* ``POST /api/v1/annotation-sets/{id}/revisions``            -- reviewer, requires
  a strong ``If-Match: "annotation-rev:<current>"`` and an ``Idempotency-Key``
* ``POST /api/v1/annotation-revisions/{id}/decision``        -- reviewer, two-person
  approval (``approved_by != created_by``)

The legacy ``POST /annotation-sets/{id}/reviews`` endpoint is intentionally
refused for v2 sets (``409 USE_REVISION_REVIEW_API``) so two review state
machines can never contradict each other; legacy v1 sets keep the V1 behaviour.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from embodied_dataops.annotations.revisions import (
    baseline_view,
    create_annotation_revision,
    decide_annotation_revision,
    get_revision_document,
    list_annotation_revisions,
    parse_revision_if_match,
    revision_etag,
    revision_view,
)
from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of, run_idempotent
from embodied_dataops.api.errors import envelope
from embodied_dataops.api.schemas.annotation import (
    AnnotationRevisionCreatedOut,
    AnnotationRevisionCreatedResponse,
    AnnotationRevisionCreateRequest,
    AnnotationRevisionDecisionOut,
    AnnotationRevisionDecisionRequest,
    AnnotationRevisionDecisionResponse,
    AnnotationRevisionListOut,
    AnnotationRevisionListResponse,
    AnnotationRevisionResponse,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.observability.audit import record_audit
from embodied_dataops.storage.base import get_storage

router = APIRouter(tags=["annotation-revisions"])

#: SPEC §4.2/DO-A02: a missing Idempotency-Key is answered by the shared helper.
IDEMPOTENCY_HEADER = "Idempotency-Key"
IF_MATCH_HEADER = "If-Match"


def _require_principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if principal is None:  # pragma: no cover - authentication middleware guarantees this
        raise ApiError(ErrorCode.UNAUTHENTICATED, "Authentication is required")
    return principal


@router.get(
    "/api/v1/annotation-sets/{annotation_set_id}/revisions",
    response_model=AnnotationRevisionListResponse,
)
def list_revisions(
    annotation_set_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """List every immutable revision of the set plus the current/approved pointers."""
    principal = _require_principal(request)
    principal.require("read")
    annotation_set, rows = list_annotation_revisions(session, annotation_set_id)
    data = AnnotationRevisionListOut(
        annotation_set_id=annotation_set.id,
        status=annotation_set.status,
        current_revision_no=int(annotation_set.current_revision_no or 1),
        approved_revision_id=annotation_set.approved_revision_id,
        revisions=[revision_view(session, row) for row in rows],
    ).model_dump(mode="json")
    return envelope(data, request_id_of(request))


@router.get(
    "/api/v1/annotation-sets/{annotation_set_id}/revisions/{revision_no}",
    response_model=AnnotationRevisionResponse,
)
def get_revision(
    annotation_set_id: str,
    revision_no: int,
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
) -> dict:
    """Return one revision content and its strong ETag."""
    principal = _require_principal(request)
    principal.require("read")
    annotation_set, revision, document, digest = get_revision_document(
        session, get_storage(), annotation_set_id, revision_no
    )
    etag = revision_etag(revision_no)
    payload = (
        revision_view(session, revision) if revision is not None else baseline_view(annotation_set, document, digest)
    )
    data = {
        "annotation_set_id": annotation_set.id,
        "revision_no": int(revision_no),
        "revision_id": payload["revision_id"],
        "state": payload["state"],
        "content_sha256": digest,
        "etag": etag,
        "annotation": document,
    }
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "no-store"
    return envelope(data, request_id_of(request))


@router.post(
    "/api/v1/annotation-sets/{annotation_set_id}/revisions",
    response_model=AnnotationRevisionCreatedResponse,
    status_code=201,
)
def create_revision(
    annotation_set_id: str,
    payload: AnnotationRevisionCreateRequest,
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
) -> dict:
    """Append revision ``current+1`` (complete replacement of the listed frames)."""
    principal = _require_principal(request)
    principal.require("review:write")
    # Syntax validation happens before the idempotency lookup; the actual CAS
    # comparison against the locked current_revision_no happens in the service
    # so a committed request can still be replayed after the revision moved.
    expected_revision_no = parse_revision_if_match(request.headers.get(IF_MATCH_HEADER))
    idempotency_key = request.headers.get(IDEMPOTENCY_HEADER)
    if not (idempotency_key or "").strip():
        raise ApiError(
            ErrorCode.IDEMPOTENCY_KEY_REQUIRED,
            "Idempotency-Key header is required for annotation revision POST",
        )

    def _apply() -> tuple[int, dict]:
        created = create_annotation_revision(
            session,
            storage=get_storage(),
            annotation_set_id=annotation_set_id,
            expected_revision_no=expected_revision_no,
            base_revision_no=payload.base_revision_no,
            frames=payload.frames,
            note=payload.note,
            actor=principal,
        )
        record_audit(
            session,
            actor=principal,
            action="annotation_revision.created",
            entity_type="annotation_revision",
            entity_id=created.revision.id,
            after={
                "annotation_set_id": annotation_set_id,
                "revision_no": created.revision.revision_no,
                "base_revision_no": payload.base_revision_no,
                "content_sha256": created.revision.content_sha256,
                "changed_frames": created.change_summary["frames"]["effective"],
                "note": created.edit.change_summary_json.get("note"),
            },
            request_id=request_id_of(request),
        )
        data = AnnotationRevisionCreatedOut(
            annotation_set_id=annotation_set_id,
            current_revision_no=int(created.revision.revision_no),
            etag=revision_etag(created.revision.revision_no),
            revision=revision_view(session, created.revision),
        ).model_dump(mode="json")
        return 201, data

    status, data, _replayed = run_idempotent(
        session,
        principal=principal,
        route=f"POST /api/v1/annotation-sets/{annotation_set_id}/revisions",
        idempotency_key=idempotency_key,
        request_payload={
            "if_match": request.headers.get(IF_MATCH_HEADER),
            "body": payload.model_dump(mode="json"),
        },
        handler=_apply,
    )
    response.headers["ETag"] = revision_etag(int(data["current_revision_no"]))
    response.status_code = status
    return envelope(data, request_id_of(request))


@router.post(
    "/api/v1/annotation-revisions/{revision_id}/decision",
    response_model=AnnotationRevisionDecisionResponse,
)
def decide_revision(
    revision_id: str,
    payload: AnnotationRevisionDecisionRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Approve/reject a revision; approval switches the explicit publication pointer."""
    principal = _require_principal(request)
    principal.require("review:write")
    decided = decide_annotation_revision(
        session,
        revision_id=revision_id,
        decision=payload.decision,
        note=payload.note,
        reviewer=principal,
    )
    record_audit(
        session,
        actor=principal,
        action=(
            "annotation_revision.approved"
            if decided.revision.state == "APPROVED"
            else "annotation_revision.rejected"
        ),
        entity_type="annotation_revision",
        entity_id=decided.revision.id,
        before={"state": decided.previous_state},
        after={
            "state": decided.revision.state,
            "annotation_set_id": decided.annotation_set.id,
            "revision_no": decided.revision.revision_no,
            "note": decided.note,
            "approved_revision_id": decided.annotation_set.approved_revision_id,
        },
        request_id=request_id_of(request),
    )
    data = AnnotationRevisionDecisionOut(
        annotation_set_id=decided.annotation_set.id,
        status=decided.annotation_set.status,
        current_revision_no=int(decided.annotation_set.current_revision_no or 1),
        approved_revision_id=decided.annotation_set.approved_revision_id,
        revision=revision_view(session, decided.revision),
    ).model_dump(mode="json")
    return envelope(data, request_id_of(request))


__all__ = ["router"]
