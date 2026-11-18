"""Evaluation and feedback-loop routes (SPEC 9.4, 10.2, DO-A01 / DO-A02 C2/C3)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import Field
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of
from embodied_dataops.api.schemas.common import StrictModel
from embodied_dataops.api.schemas.operations import AnalyzeGapsRequestV2
from embodied_dataops.feedback import service as feedback_service
from embodied_dataops.operations.contract import ETAG_EVALUATION
from embodied_dataops.operations.effects import analyze_evaluation_gaps
from embodied_dataops.operations.etag import etag_header
from embodied_dataops.operations.evaluations import evaluation_results
from embodied_dataops.operations.revisions import current_revision

router = APIRouter(prefix="/api/v1", tags=["feedback"])


def _principal(request: Request) -> Principal:
    return request.state.principal


class CreateEvaluationRequest(StrictModel):
    schema_version: str = Field(default="evaluation/1.0", pattern=r"^evaluation/1\.0$")
    model_id: str = Field(default="unknown", max_length=200)
    results: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)
    is_simulated: bool = True


@router.post("/dataset-versions/{version_id}/evaluations", status_code=201)
async def create_evaluation(
    request: Request,
    version_id: str,
    payload: CreateEvaluationRequest,
    db: Session = Depends(get_session),
) -> JSONResponse:
    principal = _principal(request)
    principal.require("analysis:write")
    report = feedback_service.create_evaluation(
        db,
        dataset_version_id=version_id,
        payload=payload.model_dump(mode="json"),
        actor=principal,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        status_code=201,
        content={
            "data": {
                "evaluation_id": report.id,
                "dataset_version_id": report.dataset_version_id,
                "schema_version": report.schema_version,
                "result_count": len((report.summary_json or {}).get("results") or []),
                "is_simulated": bool((report.summary_json or {}).get("is_simulated")),
                "created_at": report.created_at.isoformat() if report.created_at else None,
            },
            "request_id": request_id_of(request),
        },
    )


@router.post("/evaluations/{evaluation_id}/analyze-gaps", status_code=202)
async def analyze_gaps(
    request: Request,
    evaluation_id: str,
    payload: AnalyzeGapsRequestV2,
    db: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> JSONResponse:
    """DO-A02 C3: conditional gap analysis; 202 carries the operation receipt id."""
    principal = _principal(request)
    principal.require("analysis:write")
    data = analyze_evaluation_gaps(
        db,
        principal=principal,
        evaluation_id=evaluation_id,
        success_rate_threshold=payload.success_rate_threshold,
        min_trials=payload.min_trials,
        idempotency_key=idempotency_key,
        if_match=if_match,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        status_code=202, content={"data": data, "request_id": request_id_of(request)}
    )


@router.get("/collection-orders")
async def list_collection_orders(
    request: Request,
    db: Session = Depends(get_session),
    evaluation_id: str | None = Query(default=None),
    scenario_key: str | None = Query(default=None),
    state: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> JSONResponse:
    _principal(request).require("read")
    result = feedback_service.list_collection_orders(
        db,
        limit=limit,
        cursor=cursor,
        filters={
            "evaluation_report_id": evaluation_id,
            "scenario_key": scenario_key,
            "state": state,
        },
    )
    return JSONResponse(
        content={
            "data": result["data"],
            "next_cursor": result.get("next_cursor"),
            "request_id": request_id_of(request),
        }
    )


@router.get("/evaluations/{evaluation_id}")
async def get_evaluation(
    request: Request, evaluation_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    """DO-A01 strict read model + DO-A02 C2 business revision / results hash."""
    _principal(request).require("read")
    report = feedback_service.get_evaluation(db, evaluation_id)
    results, results_hash, validated = evaluation_results(db, report)
    revision = current_revision(report)
    data = {
        "evaluation_id": report.id,
        "dataset_version_id": report.dataset_version_id,
        "schema_version": report.schema_version,
        "model_id": validated.model_id,
        "is_simulated": bool(validated.is_simulated),
        "results": results,
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "business_revision": revision,
        "results_sha256": results_hash,
    }
    return JSONResponse(
        content={"data": data, "request_id": request_id_of(request)},
        headers=etag_header(ETAG_EVALUATION, revision),
    )


__all__ = ["router"]
