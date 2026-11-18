"""V2.1 AI pre-annotation API (SPEC_V2.1 §5.5).

Seven endpoints, all under ``/api/v1`` and all returning the V1 ``{data, request_id}``
envelope:

* ``POST /episodes/{id}/ai-preannotation-runs``   operator, ``Idempotency-Key``
* ``GET  /episodes/{id}/ai-preannotations``       viewer
* ``GET  /ai-inferences/{id}``                    viewer
* ``POST /ai-preannotations/{id}/revisions``      reviewer, ``If-Match`` + ``Idempotency-Key``
* ``POST /ai-preannotations/{id}/decision``       reviewer, two-person approval
* ``GET  /episodes/{id}/ai-quality-summary``      viewer
* ``GET  /model-usage?run_id=``                   operator

Two rules are enforced at the edge rather than deep in a worker, because a worker
failure is a 500 while these are actionable 4xx:

* ``mode=live`` requires ``VLM_PROVIDER=dashscope`` **and** ``egress_allowed`` on the
  source import -- otherwise ``503``/``403`` before a single byte leaves;
* ``mode=replay`` requires ``VLM_PROVIDER=replay``.  The platform never silently turns a
  requested live run into a replay run.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from embodied_dataops.annotations.action_intervals import (
    create_action_revision,
    latest_action_revision,
)
from embodied_dataops.annotations.revisions import (
    decide_annotation_revision,
    parse_revision_if_match,
    revision_etag,
)
from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of, run_idempotent
from embodied_dataops.api.errors import envelope
from embodied_dataops.api.schemas.ai_preannotation import (
    AiCandidateOut,
    AiClipOut,
    AiDecisionOut,
    AiDecisionRequest,
    AiInferenceOut,
    AiPreannotationRunOut,
    AiPreannotationRunRequest,
    AiPreannotationsOut,
    AiQualitySummaryOut,
    AiRevisionOut,
    AiRevisionRequest,
    ModelUsageOut,
)
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import (
    AIClipInput,
    AnnotationRevision,
    AnnotationSet,
    Asset,
    Episode,
    ModelCallAttempt,
    ModelInference,
    PipelineRun,
    SourceImport,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.observability.audit import record_audit
from embodied_dataops.storage.base import get_storage
from embodied_dataops.workflow.dag import create_pipeline_run

router = APIRouter(prefix="/api/v1", tags=["ai-preannotation"])

IDEMPOTENCY_HEADER = "Idempotency-Key"
IF_MATCH_HEADER = "If-Match"
AI_PIPELINE_VERSION = "2.1.0"
AI_ANNOTATION_KIND = "ACTION_INTERVAL"


def _principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if principal is None:  # pragma: no cover - authentication middleware guarantees this
        raise ApiError(ErrorCode.UNAUTHENTICATED, "Authentication is required")
    return principal


@router.post("/episodes/{episode_id}/ai-preannotation-runs", status_code=202)
def create_ai_run(
    episode_id: str,
    payload: AiPreannotationRunRequest,
    request: Request,
    session: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias=IDEMPOTENCY_HEADER),
) -> JSONResponse:
    """Start a V2.1 run whose DAG ends in a reviewed candidate annotation set."""
    principal = _principal(request)
    principal.require("pipeline:write")

    def _apply() -> tuple[int, dict]:
        episode = session.get(Episode, episode_id)
        if episode is None:
            raise ApiError(
                ErrorCode.NOT_FOUND, "episode not found", details={"episode_id": episode_id}
            )
        _check_mode(payload.mode, get_settings())
        if payload.mode == "live":
            _require_egress(session, episode)

        run = create_pipeline_run(
            session,
            episode,
            requested_by=principal.key_id,
            pipeline_version=AI_PIPELINE_VERSION,
            config={
                "ai_preannotation": True,
                "mode": payload.mode,
                "keyframe_policy": payload.keyframe_policy,
                "model_pipeline_version": payload.model_pipeline_version,
            },
        )
        record_audit(
            session,
            actor=principal,
            action="ai_preannotation.run_created",
            entity_type="pipeline_run",
            entity_id=run.id,
            after={
                "episode_id": episode.id,
                "mode": payload.mode,
                "pipeline_version": run.pipeline_version,
                "keyframe_policy": payload.keyframe_policy,
            },
            request_id=request_id_of(request),
        )
        data = AiPreannotationRunOut(
            run_id=run.id,
            state=run.state,
            mode=payload.mode,
            pipeline_version=run.pipeline_version,
        ).model_dump(mode="json")
        return 202, data

    status, data, _replayed = run_idempotent(
        session,
        principal=principal,
        route=f"POST /api/v1/episodes/{episode_id}/ai-preannotation-runs",
        idempotency_key=idempotency_key,
        request_payload=payload.model_dump(mode="json"),
        handler=_apply,
    )
    return JSONResponse(status_code=status, content=envelope(data, request_id_of(request)))


@router.get("/episodes/{episode_id}/ai-preannotations")
def list_ai_preannotations(
    episode_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Everything a reviewer needs: clips, provenance, candidates and schema errors."""
    principal = _principal(request)
    principal.require("read")
    annotation_set = _latest_ai_set(session, episode_id)
    if annotation_set is None:
        raise ApiError(
            ErrorCode.AI_PREANNOTATION_NOT_FOUND,
            "no AI pre-annotation exists for this episode",
            details={"episode_id": episode_id},
        )
    document = _read_artifact(session, annotation_set)
    clips = [
        AiClipOut(
            clip_id=str(item["clip_id"]),
            start_ns=int(item["start_ns"]),
            end_ns=int(item["end_ns"]),
            frame_count=int(item.get("frame_count") or 0),
            inference_id=item.get("inference_id"),
            mode=item.get("inference_mode"),
            model_name=item.get("model_name"),
            prompt_version=item.get("prompt_version"),
            status="SUCCEEDED",
            insufficient_evidence=bool(item.get("insufficient_evidence")),
            action_count=int(item.get("action_count") or 0),
        )
        for item in (document.get("provenance") or {}).get("clips") or []
    ]
    candidates = [
        AiCandidateOut(
            candidate_id=str(item["candidate_id"]),
            clip_id=str(item["clip_id"]),
            verb=str(item["verb"]),
            noun=str(item["noun"]),
            start_ns=int(item["start_ns"]),
            end_ns=int(item["end_ns"]),
            evidence_frame_indices=[int(value) for value in item.get("evidence_frame_indices") or []],
            uncertainty=str(item.get("uncertainty") or "medium"),
            review_state=str(item.get("review_state") or "PENDING"),
        )
        for item in document.get("actions") or []
    ]
    quality = (annotation_set.summary_json or {}).get("quality") or {}
    artifact_sha = None
    if annotation_set.artifact_asset_id:
        asset = session.get(Asset, annotation_set.artifact_asset_id)
        artifact_sha = asset.sha256 if asset is not None else None
    data = AiPreannotationsOut(
        episode_id=episode_id,
        annotation_set_id=annotation_set.id,
        annotation_kind=str((annotation_set.summary_json or {}).get("annotation_kind") or AI_ANNOTATION_KIND),
        status=annotation_set.status,
        artifact_sha256=artifact_sha,
        clips=clips,
        candidates=candidates,
        schema_errors=[],
        quality=quality,
    ).model_dump(mode="json")
    return envelope(data, request_id_of(request))


@router.get("/ai-inferences/{inference_id}")
def get_ai_inference(
    inference_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Status, latency, token usage and a bounded summary; never the raw prompt."""
    principal = _principal(request)
    principal.require("read")
    row = session.get(ModelInference, inference_id)
    if row is None:
        raise ApiError(
            ErrorCode.NOT_FOUND, "inference not found", details={"inference_id": inference_id}
        )
    clip = session.get(AIClipInput, row.clip_input_id)
    attempts = session.scalar(
        select(func.count())
        .select_from(ModelCallAttempt)
        .where(ModelCallAttempt.inference_id == row.id)
    )
    prompt_tokens = session.scalar(
        select(func.coalesce(func.sum(ModelCallAttempt.usage_prompt_tokens), 0)).where(
            ModelCallAttempt.inference_id == row.id
        )
    )
    completion_tokens = session.scalar(
        select(func.coalesce(func.sum(ModelCallAttempt.usage_completion_tokens), 0)).where(
            ModelCallAttempt.inference_id == row.id
        )
    )
    data = AiInferenceOut(
        inference_id=row.id,
        run_id=row.run_id,
        clip_id=clip.clip_id if clip is not None else "",
        provider=row.provider,
        model_name=row.model_name,
        mode=row.mode,
        prompt_version=row.prompt_version,
        prompt_sha256=row.prompt_sha256,
        status=row.status,
        error_code=row.error_code,
        latency_ms=row.latency_ms,
        estimated_cost_usd=(
            float(row.estimated_cost_usd) if row.estimated_cost_usd is not None else None
        ),
        attempts=int(attempts or 0),
        output_sha256=row.output_sha256,
        summary={
            "clip_start_ns": int(clip.start_ns) if clip is not None else None,
            "clip_end_ns": int(clip.end_ns) if clip is not None else None,
            "frame_count": int(clip.frame_count) if clip is not None else None,
            "usage_prompt_tokens": int(prompt_tokens or 0),
            "usage_completion_tokens": int(completion_tokens or 0),
        },
    ).model_dump(mode="json")
    return envelope(data, request_id_of(request))


@router.post("/ai-preannotations/{annotation_set_id}/revisions", status_code=201)
def create_ai_revision(
    annotation_set_id: str,
    payload: AiRevisionRequest,
    request: Request,
    session: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias=IDEMPOTENCY_HEADER),
    if_match: str | None = Header(default=None, alias=IF_MATCH_HEADER),
) -> JSONResponse:
    """Append an ACTION_INTERVAL revision; the model answer itself is never rewritten."""
    principal = _principal(request)
    principal.require("review:write")
    expected_revision_no = parse_revision_if_match(if_match)

    def _apply() -> tuple[int, dict]:
        created = create_action_revision(
            session,
            storage=get_storage(),
            annotation_set_id=annotation_set_id,
            expected_revision_no=expected_revision_no,
            actions=[action.model_dump(mode="json") for action in payload.actions],
            note=payload.note,
            actor=principal,
        )
        record_audit(
            session,
            actor=principal,
            action="ai_preannotation.revision_created",
            entity_type="annotation_revision",
            entity_id=created.revision.id,
            after={
                "annotation_set_id": annotation_set_id,
                "revision_no": created.revision.revision_no,
                "actions": created.action_count,
                "content_sha256": created.revision.content_sha256,
                "note": payload.note,
            },
            request_id=request_id_of(request),
        )
        data = AiRevisionOut(
            annotation_set_id=annotation_set_id,
            revision_id=created.revision.id,
            revision_no=int(created.revision.revision_no),
            state=created.revision.state,
            content_sha256=created.revision.content_sha256,
            etag=revision_etag(int(created.revision.revision_no)),
            action_count=created.action_count,
        ).model_dump(mode="json")
        return 201, data

    status, data, _replayed = run_idempotent(
        session,
        principal=principal,
        route=f"POST /api/v1/ai-preannotations/{annotation_set_id}/revisions",
        idempotency_key=idempotency_key,
        request_payload={"if_match": if_match, "body": payload.model_dump(mode="json")},
        handler=_apply,
    )
    response = JSONResponse(status_code=status, content=envelope(data, request_id_of(request)))
    response.headers["ETag"] = revision_etag(int(data["revision_no"]))
    return response


@router.post("/ai-preannotations/{annotation_set_id}/decision")
def decide_ai_preannotation(
    annotation_set_id: str,
    payload: AiDecisionRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Approve or reject the latest revision; the approver must not be its editor."""
    principal = _principal(request)
    principal.require("review:write")
    revision = latest_action_revision(session, annotation_set_id)
    if revision is None:
        raise ApiError(
            ErrorCode.AI_REVIEW_REQUIRED,
            "this annotation set has no revision to decide on",
            details={"annotation_set_id": annotation_set_id},
        )
    decided = decide_annotation_revision(
        session,
        revision_id=revision.id,
        decision=payload.decision,
        note=payload.note,
        reviewer=principal,
    )
    record_audit(
        session,
        actor=principal,
        action=(
            "ai_preannotation.approved"
            if decided.revision.state == "APPROVED"
            else "ai_preannotation.rejected"
        ),
        entity_type="annotation_revision",
        entity_id=decided.revision.id,
        before={"state": decided.previous_state},
        after={
            "state": decided.revision.state,
            "annotation_set_id": annotation_set_id,
            "note": payload.note,
        },
        request_id=request_id_of(request),
    )
    data = AiDecisionOut(
        annotation_set_id=annotation_set_id,
        revision_id=decided.revision.id,
        state=decided.revision.state,
        status=decided.annotation_set.status,
        approved_revision_id=decided.annotation_set.approved_revision_id,
    ).model_dump(mode="json")
    return envelope(data, request_id_of(request))


@router.get("/episodes/{episode_id}/ai-quality-summary")
def ai_quality_summary(
    episode_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Schema validity, approval state and the warnings a human should read first."""
    principal = _principal(request)
    principal.require("read")
    annotation_set = _latest_ai_set(session, episode_id)
    if annotation_set is None:
        raise ApiError(
            ErrorCode.AI_PREANNOTATION_NOT_FOUND,
            "no AI pre-annotation exists for this episode",
            details={"episode_id": episode_id},
        )
    summary = (annotation_set.summary_json or {}).get("quality") or {}
    revisions = list(
        session.scalars(
            select(AnnotationRevision).where(
                AnnotationRevision.annotation_set_id == annotation_set.id
            )
        )
    )
    approved = annotation_set.approved_revision_id is not None
    approval_rate = 1.0 if approved else 0.0
    inferences = list(
        session.scalars(
            select(ModelInference).where(
                ModelInference.run_id.in_(
                    select(PipelineRun.id).where(PipelineRun.episode_id == episode_id)
                )
            )
        )
    )
    data = AiQualitySummaryOut(
        episode_id=episode_id,
        annotation_set_id=annotation_set.id,
        run_id=inferences[0].run_id if inferences else None,
        mode=summary.get("mode"),
        live_evidence=bool(summary.get("live_evidence")),
        schema_valid_rate=summary.get("schema_valid_rate"),
        call_success_rate=summary.get("call_success_rate"),
        approval_rate=approval_rate,
        action_count=int(summary.get("action_count") or 0),
        warnings=list(summary.get("warnings") or []),
        model_versions=sorted({row.model_name for row in inferences}),
        prompt_versions=sorted({row.prompt_version for row in inferences}),
        source_refs=[
            {
                "inference_id": row.id,
                "clip_input_id": row.clip_input_id,
                "prompt_sha256": row.prompt_sha256,
                "output_sha256": row.output_sha256,
                "mode": row.mode,
            }
            for row in inferences
        ],
    ).model_dump(mode="json")
    data["revisions"] = len(revisions)
    return envelope(data, request_id_of(request))


@router.get("/model-usage")
def model_usage(
    request: Request,
    run_id: str = Query(..., min_length=1),
    session: Session = Depends(get_session),
) -> dict:
    """Calls, retries, tokens and an *estimated* cost -- never a precise bill."""
    principal = _principal(request)
    principal.require("analysis:write")
    inferences = list(
        session.scalars(select(ModelInference).where(ModelInference.run_id == run_id))
    )
    if not inferences:
        raise ApiError(
            ErrorCode.NOT_FOUND, "no inference for this run", details={"run_id": run_id}
        )
    ids = [row.id for row in inferences]
    calls = session.scalar(
        select(func.count()).select_from(ModelCallAttempt).where(ModelCallAttempt.inference_id.in_(ids))
    ) or 0
    ok = session.scalar(
        select(func.count())
        .select_from(ModelCallAttempt)
        .where(ModelCallAttempt.inference_id.in_(ids), ModelCallAttempt.response_status == 200)
    ) or 0
    prompt_tokens = session.scalar(
        select(func.coalesce(func.sum(ModelCallAttempt.usage_prompt_tokens), 0)).where(
            ModelCallAttempt.inference_id.in_(ids)
        )
    )
    completion_tokens = session.scalar(
        select(func.coalesce(func.sum(ModelCallAttempt.usage_completion_tokens), 0)).where(
            ModelCallAttempt.inference_id.in_(ids)
        )
    )
    cost = session.scalar(
        select(func.coalesce(func.sum(ModelCallAttempt.estimated_cost_usd), 0)).where(
            ModelCallAttempt.inference_id.in_(ids)
        )
    )
    budget = float(get_settings().vlm_budget_usd_per_run)
    cost_value = float(cost) if cost is not None else None
    data = ModelUsageOut(
        run_id=run_id,
        calls=int(calls),
        calls_succeeded=int(ok),
        retries=max(0, int(calls) - len(ids)),
        prompt_tokens=int(prompt_tokens or 0),
        completion_tokens=int(completion_tokens or 0),
        estimated_cost_usd=cost_value,
        cost_known=cost_value is not None and cost_value > 0,
        budget_usd=budget,
        over_budget=bool(cost_value is not None and cost_value > budget),
        models=sorted({row.model_name for row in inferences}),
    ).model_dump(mode="json")
    return envelope(data, request_id_of(request))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _check_mode(mode: str, settings) -> None:
    configured = str(getattr(settings, "vlm_provider", "disabled") or "disabled").lower()
    if mode == "live" and configured != "dashscope":
        raise ApiError(
            ErrorCode.VLM_PROVIDER_UNAVAILABLE,
            "mode=live requires VLM_PROVIDER=dashscope; the platform never downgrades to replay",
            details={"requested": mode, "configured": configured},
        )
    if mode == "replay" and configured != "replay":
        raise ApiError(
            ErrorCode.VLM_PROVIDER_UNAVAILABLE,
            "mode=replay requires VLM_PROVIDER=replay",
            details={"requested": mode, "configured": configured},
        )


def _require_egress(session: Session, episode: Episode) -> None:
    source = session.get(SourceImport, episode.source_import_id)
    if source is None or not bool(source.egress_allowed):
        raise ApiError(
            ErrorCode.SOURCE_EGRESS_NOT_ALLOWED,
            "source is not cleared for outbound model calls",
            details={
                "episode_id": episode.id,
                "egress_allowed": bool(getattr(source, "egress_allowed", False)),
            },
        )


def _latest_ai_set(session: Session, episode_id: str) -> AnnotationSet | None:
    rows = list(
        session.scalars(
            select(AnnotationSet)
            .where(AnnotationSet.episode_id == episode_id)
            .order_by(AnnotationSet.created_at.desc())
        )
    )
    for row in rows:
        if str((row.summary_json or {}).get("annotation_kind") or "") == AI_ANNOTATION_KIND:
            return row
    return None


def _read_artifact(session: Session, annotation_set: AnnotationSet) -> dict:
    if not annotation_set.artifact_asset_id:
        return {}
    asset = session.get(Asset, annotation_set.artifact_asset_id)
    if asset is None:
        return {}
    try:
        return json.loads(get_storage().get_bytes(asset.object_key))
    except Exception:  # pragma: no cover - a broken artifact must not 500 the read path
        return {}


__all__ = ["router"]
