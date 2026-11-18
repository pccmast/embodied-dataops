"""Bounded read-only diagnostics (DO-A02 C5)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of
from embodied_dataops.db.models import Episode
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.operations.diagnostics import (
    MAX_AGGREGATE_LIMIT,
    MAX_TIMELINE_POINTS,
    episode_timeline,
    failure_analysis,
    parse_rfc3339,
    quality_aggregate,
)

router = APIRouter(prefix="/api/v1", tags=["diagnostics"])


def _principal(request: Request) -> Principal:
    return request.state.principal


@router.get("/episodes/{episode_id}/timeline")
async def get_timeline(
    request: Request,
    episode_id: str,
    db: Session = Depends(get_session),
    start_ns: int | None = Query(default=None, ge=0),
    end_ns: int | None = Query(default=None, ge=0),
    max_points: int = Query(default=100, ge=1, le=MAX_TIMELINE_POINTS),
) -> JSONResponse:
    _principal(request).require("read")
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Episode not found", details={"episode_id": episode_id})
    if start_ns is None or end_ns is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "timeline window not found; start_ns and end_ns are required",
            details={"episode_id": episode_id, "required": ["start_ns", "end_ns"]},
        )
    data = episode_timeline(
        db, episode, start_ns=int(start_ns), end_ns=int(end_ns), max_points=int(max_points)
    )
    return JSONResponse(content={"data": data, "request_id": request_id_of(request)})


@router.get("/pipeline-runs/{run_id}/failure-analysis")
async def get_failure_analysis(
    request: Request, run_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    _principal(request).require("read")
    data = failure_analysis(db, run_id)
    return JSONResponse(content={"data": data, "request_id": request_id_of(request)})


@router.get("/quality/aggregate")
async def get_quality_aggregate(
    request: Request,
    db: Session = Depends(get_session),
    from_time: str = Query(alias="from"),
    to_time: str = Query(alias="to"),
    group_by: str = Query(default="source_type"),
    limit: int = Query(default=20, ge=1, le=MAX_AGGREGATE_LIMIT),
) -> JSONResponse:
    _principal(request).require("read")
    data = quality_aggregate(
        db,
        from_time=parse_rfc3339(from_time, "from"),
        to_time=parse_rfc3339(to_time, "to"),
        group_by=group_by,
        limit=int(limit),
    )
    return JSONResponse(content={"data": data, "request_id": request_id_of(request)})
