"""Scenario-tag, coverage and collection-order lifecycle routes (V2 D2-6).

DO-A02 C6 / SPEC_ADDENDUM_V2 §7.  V1 endpoints stay in ``routes/feedback.py``;
this module only adds the V2 business surface:

* ``GET|POST /episodes/{id}/scenario-tags``  (evidence-backed scenario labels)
* ``GET  /quality/coverage``                 (PUBLISHED version, Episode count)
* ``GET  /collection-orders/{id}``           (lifecycle detail + linked episodes)
* ``POST /collection-orders/{id}/approve``   (different reviewer, audited)
* ``POST /collection-orders/{id}/link-episodes``
* ``POST /collection-orders/{id}/cancel``
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import Field
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of
from embodied_dataops.api.schemas.common import StrictModel
from embodied_dataops.feedback import coverage as coverage_service
from embodied_dataops.feedback import service as feedback_service

router = APIRouter(prefix="/api/v1", tags=["collection"])


def _principal(request: Request) -> Principal:
    return request.state.principal


class AddScenarioTagRequest(StrictModel):
    scenario_key: str = Field(min_length=3, max_length=64)
    basis: str = Field(default="HUMAN_VERIFIED", pattern="^(SOURCE_DECLARED|HUMAN_VERIFIED|UNKNOWN)$")
    note: str = Field(default="", max_length=2000)
    source_evidence_asset_id: str | None = Field(default=None, max_length=36)


class ApproveCollectionOrderRequest(StrictModel):
    note: str = Field(min_length=1, max_length=2000)


class LinkCollectionOrderEpisodesRequest(StrictModel):
    episode_ids: list[str] = Field(min_length=1, max_length=feedback_service.MAX_LINK_EPISODES)


class CancelCollectionOrderRequest(StrictModel):
    reason: str = Field(min_length=1, max_length=2000)


@router.get("/episodes/{episode_id}/scenario-tags")
async def list_scenario_tags(
    request: Request, episode_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    _principal(request).require("read")
    tags = coverage_service.list_scenario_tags(db, episode_id)
    return JSONResponse(
        content={
            "data": {
                "episode_id": episode_id,
                "tags": tags,
                "credible_bases": list(coverage_service.CREDIBLE_BASE_PRIORITY),
                "unknown_never_counts": True,
            },
            "request_id": request_id_of(request),
        }
    )


@router.post("/episodes/{episode_id}/scenario-tags", status_code=201)
async def add_scenario_tag(
    request: Request,
    episode_id: str,
    payload: AddScenarioTagRequest,
    db: Session = Depends(get_session),
) -> JSONResponse:
    principal = _principal(request)
    # SOURCE_DECLARED comes from the import/operator side; HUMAN_VERIFIED is reviewer-only.
    principal.require("review:write" if payload.basis == "HUMAN_VERIFIED" else "import:write")
    tag = coverage_service.add_scenario_tag(
        db,
        episode_id=episode_id,
        scenario_key=payload.scenario_key,
        basis=payload.basis,
        actor=principal,
        note=payload.note,
        source_evidence_asset_id=payload.source_evidence_asset_id,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        status_code=201,
        content={
            "data": {
                "tag": coverage_service._serialize_tag(tag),  # noqa: SLF001 - stable serializer
                "counts_as_coverage": tag.basis in coverage_service.CREDIBLE_BASE_PRIORITY,
            },
            "request_id": request_id_of(request),
        },
    )


@router.get("/quality/coverage")
async def get_quality_coverage(
    request: Request,
    dataset_version_id: str = Query(min_length=1, max_length=36),
    scenario_key: str = Query(min_length=3, max_length=64),
    db: Session = Depends(get_session),
) -> JSONResponse:
    _principal(request).require("read")
    payload = coverage_service.get_coverage(
        db, dataset_version_id=dataset_version_id, scenario_key=scenario_key
    )
    return JSONResponse(
        content={"data": payload, "request_id": request_id_of(request)}
    )


@router.get("/collection-orders/{order_id}")
async def get_collection_order(
    request: Request, order_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    _principal(request).require("read")
    return JSONResponse(
        content={
            "data": feedback_service.get_collection_order(db, order_id),
            "request_id": request_id_of(request),
        }
    )


@router.post("/collection-orders/{order_id}/approve")
async def approve_collection_order(
    request: Request,
    order_id: str,
    payload: ApproveCollectionOrderRequest,
    db: Session = Depends(get_session),
) -> JSONResponse:
    principal = _principal(request)
    principal.require("review:write")
    order = feedback_service.approve_collection_order(
        db,
        order_id=order_id,
        actor=principal,
        note=payload.note,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        content={
            "data": {
                "order_id": order.id,
                "state": order.state,
                "approved_by": order.approved_by,
                "approved_at": order.approved_at.isoformat() if order.approved_at else None,
                "is_simulated_source": bool(order.is_simulated_source),
            },
            "request_id": request_id_of(request),
        }
    )


@router.post("/collection-orders/{order_id}/link-episodes")
async def link_collection_order_episodes(
    request: Request,
    order_id: str,
    payload: LinkCollectionOrderEpisodesRequest,
    db: Session = Depends(get_session),
) -> JSONResponse:
    principal = _principal(request)
    principal.require("analysis:write")
    order = feedback_service.link_collection_order_episodes(
        db,
        order_id=order_id,
        episode_ids=payload.episode_ids,
        actor=principal,
        request_id=request_id_of(request),
    )
    detail = feedback_service.get_collection_order(db, order.id)
    return JSONResponse(
        content={
            "data": {
                "order_id": order.id,
                "state": order.state,
                "linked_count": detail["linked_count"],
                "requested_count": detail["requested_count"],
                "fulfilled": detail["fulfilled"],
                "is_simulated_source": bool(order.is_simulated_source),
            },
            "request_id": request_id_of(request),
        }
    )


@router.post("/collection-orders/{order_id}/cancel")
async def cancel_collection_order(
    request: Request,
    order_id: str,
    payload: CancelCollectionOrderRequest,
    db: Session = Depends(get_session),
) -> JSONResponse:
    principal = _principal(request)
    principal.require("analysis:write")
    order = feedback_service.cancel_collection_order(
        db,
        order_id=order_id,
        actor=principal,
        reason=payload.reason,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        content={
            "data": {"order_id": order.id, "state": order.state},
            "request_id": request_id_of(request),
        }
    )
