"""Annotation inspection and review endpoints (SPEC 9.4, 7.2, 8.1).

* ``GET  /api/v1/episodes/{episode_id}/annotations``      -- permission ``read``
* ``POST /api/v1/annotation-sets/{annotation_set_id}/reviews`` -- permission
  ``review:write`` (only the ``reviewer`` and ``admin`` roles hold it).

The reviewer flow is not idempotent by design: SPEC 6.4 only requires
``Idempotency-Key`` for uploads and pipeline creation, and a repeated review is
a state-machine conflict answered with ``409 ANNOTATION_ALREADY_REVIEWED``.

V2 compatibility (SPEC §4.2): a set that owns ``annotation-revision/2.0``
revision rows is reviewed through ``POST /annotation-revisions/{id}/decision``.
The legacy endpoint refuses those sets with ``409 USE_REVISION_REVIEW_API`` so
the two review state machines cannot diverge; legacy v1 sets are unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from embodied_dataops.annotations.revisions import (
    annotation_set_has_revisions,
    get_annotation_set,
)
from embodied_dataops.annotations.service import (
    build_annotation_view,
    latest_annotation_set,
    review_annotation_set,
)
from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of
from embodied_dataops.api.errors import envelope
from embodied_dataops.api.schemas.annotation import (
    AnnotationReviewRequest,
    AnnotationReviewResponse,
    EpisodeAnnotationsResponse,
)
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import Episode
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.observability.audit import record_audit
from embodied_dataops.storage.base import get_storage

router = APIRouter(tags=["annotations"])


def _require_principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if principal is None:  # pragma: no cover - authentication middleware guarantees this
        raise ApiError(ErrorCode.UNAUTHENTICATED, "Authentication is required")
    return principal


@router.get(
    "/api/v1/episodes/{episode_id}/annotations",
    response_model=EpisodeAnnotationsResponse,
)
def get_episode_annotations(
    episode_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Latest annotation candidates for an Episode plus 10-minute preview URLs."""
    principal = _require_principal(request)
    principal.require("read")
    episode = session.get(Episode, episode_id)
    if episode is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            f"Episode {episode_id} not found",
            details={"episode_id": episode_id},
        )
    annotation_set = latest_annotation_set(session, episode.id)
    if annotation_set is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "No annotation set exists for this episode",
            details={"episode_id": episode.id},
        )
    data = build_annotation_view(
        session,
        storage=get_storage(),
        episode=episode,
        annotation_set=annotation_set,
        settings=get_settings(),
    )
    return envelope(data, request_id_of(request))


@router.post(
    "/api/v1/annotation-sets/{annotation_set_id}/reviews",
    response_model=AnnotationReviewResponse,
    status_code=201,
)
def create_annotation_review(
    annotation_set_id: str,
    payload: AnnotationReviewRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Approve or reject the latest annotation candidates (SPEC 7.2)."""
    principal = _require_principal(request)
    principal.require("review:write")
    if annotation_set_has_revisions(session, annotation_set_id):
        annotation_set = get_annotation_set(session, annotation_set_id)
        raise ApiError(
            ErrorCode.USE_REVISION_REVIEW_API,
            "This annotation set is reviewed through annotation revisions; "
            "use POST /api/v1/annotation-revisions/{id}/decision instead",
            details={
                "annotation_set_id": annotation_set.id,
                "current_revision_no": int(annotation_set.current_revision_no or 1),
                "review_endpoint": "/api/v1/annotation-revisions/{revision_id}/decision",
            },
        )
    review, previous_status, new_status = review_annotation_set(
        session,
        annotation_set_id=annotation_set_id,
        decision=payload.decision,
        note=payload.note,
        reviewer=principal,
        storage=get_storage(),
    )
    record_audit(
        session,
        actor=principal,
        action="annotation_set.reviewed",
        entity_type="annotation_set",
        entity_id=annotation_set_id,
        before={"status": previous_status},
        after={"status": new_status, "decision": payload.decision, "note": review.note},
        request_id=request_id_of(request),
    )
    return envelope(
        {"review_id": review.id, "new_status": new_status},
        request_id_of(request),
    )
