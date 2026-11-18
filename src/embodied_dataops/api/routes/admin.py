"""Admin routes: audit trail and diagnostics (SPEC 9.5)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, tuple_
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import decode_cursor, encode_cursor, get_session, request_id_of
from embodied_dataops.db.models import AuditEvent, Episode, SourceImport, Task
from embodied_dataops.domain.enums import TaskState

router = APIRouter(prefix="/api/v1", tags=["admin"])


def _principal(request: Request) -> Principal:
    return request.state.principal


@router.get("/audit-events")
async def list_audit_events(
    request: Request,
    db: Session = Depends(get_session),
    entity_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    action: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> JSONResponse:
    _principal(request).require("admin")
    stmt = select(AuditEvent).order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
    if entity_type:
        stmt = stmt.where(AuditEvent.entity_type == entity_type)
    if entity_id:
        stmt = stmt.where(AuditEvent.entity_id == entity_id)
    if action:
        stmt = stmt.where(AuditEvent.action == action)
    if cursor:
        timestamp, entity = decode_cursor(cursor)
        stmt = stmt.where(
            tuple_(AuditEvent.created_at, AuditEvent.id)
            < tuple_(datetime.fromisoformat(timestamp), entity)
        )
    rows = list(db.scalars(stmt.limit(limit + 1)))
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = encode_cursor(rows[-1].created_at, rows[-1].id)
    return JSONResponse(
        content={
            "data": [
                {
                    "id": row.id,
                    "actor_id": row.actor_id,
                    "action": row.action,
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "before": row.before_json,
                    "after": row.after_json,
                    "request_id": row.request_id,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ],
            "next_cursor": next_cursor,
            "request_id": request_id_of(request),
        }
    )


@router.get("/admin/stats")
async def stats(request: Request, db: Session = Depends(get_session)) -> JSONResponse:
    _principal(request).require("admin")
    from sqlalchemy import func

    return JSONResponse(
        content={
            "data": {
                "imports": db.scalar(select(func.count()).select_from(SourceImport)),
                "episodes": db.scalar(select(func.count()).select_from(Episode)),
                "tasks": db.scalar(select(func.count()).select_from(Task)),
                "running_tasks": db.scalar(
                    select(func.count()).select_from(Task).where(Task.state == TaskState.RUNNING.value)
                ),
            },
            "request_id": request_id_of(request),
        }
    )
