"""Pipeline run and task control routes (SPEC 9.3 + DO-A02 C2/C3).

The three V1 unconditional writes are deliberately **replaced** by DO-A02
conditional writes: a missing ``If-Match`` is ``428`` and a stale one is ``412``.
Business mutations are delegated to ``operations.effects`` so the API never edits
task/run rows directly and every accepted write produces exactly one receipt and
one effect.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of
from embodied_dataops.api.schemas.operations import CreatePipelineRunRequestV2, RetryTaskRequest
from embodied_dataops.db.models import Task
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.operations.contract import ETAG_TASK
from embodied_dataops.operations.effects import is_task_retryable, reprocess_episode, retry_failed_task
from embodied_dataops.operations.etag import etag_header
from embodied_dataops.operations.revisions import current_revision
from embodied_dataops.workflow.commands import request_run_cancel, run_task_summary

router = APIRouter(prefix="/api/v1", tags=["pipeline"])


def _principal(request: Request) -> Principal:
    return request.state.principal


@router.post("/episodes/{episode_id}/pipeline-runs", status_code=202)
async def create_run(
    request: Request,
    episode_id: str,
    payload: CreatePipelineRunRequestV2,
    db: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> JSONResponse:
    """DO-A02 C3: conditional re-process; requires If-Match episode-rev + Idempotency-Key."""
    principal = _principal(request)
    principal.require("pipeline:write")
    data = reprocess_episode(
        db,
        principal=principal,
        episode_id=episode_id,
        pipeline_version=payload.pipeline_version,
        idempotency_key=idempotency_key,
        if_match=if_match,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        status_code=202, content={"data": data, "request_id": request_id_of(request)}
    )


@router.get("/pipeline-runs/{run_id}")
async def get_run(request: Request, run_id: str, db: Session = Depends(get_session)) -> JSONResponse:
    _principal(request).require("read")
    summary = run_task_summary(db, run_id)
    run = {key: value for key, value in summary.items() if key != "tasks"}
    return JSONResponse(
        content={
            "data": {"run": run, "tasks": summary["tasks"]},
            "request_id": request_id_of(request),
        }
    )


@router.post("/pipeline-runs/{run_id}/cancel", status_code=202)
async def cancel_run(request: Request, run_id: str, db: Session = Depends(get_session)) -> JSONResponse:
    principal = _principal(request)
    principal.require("pipeline:write")
    result = request_run_cancel(db, run_id, commit=False)
    from embodied_dataops.observability.audit import record_audit

    record_audit(
        db,
        actor=principal,
        action="pipeline.run.cancel",
        entity_type="pipeline_run",
        entity_id=run_id,
        after=result,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        status_code=202,
        content={"data": result, "request_id": request_id_of(request)},
    )


@router.post("/tasks/{task_id}/retry", status_code=202)
async def retry_task(
    request: Request,
    task_id: str,
    payload: RetryTaskRequest,
    db: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> JSONResponse:
    """DO-A02 C3: retry a FAILED task; requires If-Match task-rev + expected_run_id."""
    principal = _principal(request)
    principal.require("pipeline:write")
    data = retry_failed_task(
        db,
        principal=principal,
        task_id=task_id,
        expected_run_id=payload.expected_run_id,
        idempotency_key=idempotency_key,
        if_match=if_match,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        status_code=202, content={"data": data, "request_id": request_id_of(request)}
    )


@router.get("/tasks/{task_id}")
async def get_task(
    request: Request,
    task_id: str,
    db: Session = Depends(get_session),
    include_output: bool = Query(default=False),
) -> JSONResponse:
    """DO-A02 C2: single task read with a strong ``task-rev`` ETag (viewer)."""
    _principal(request).require("read")
    task = db.get(Task, task_id)
    if task is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Task not found", details={"task_id": task_id})
    payload = {
        "task_id": task.id,
        "run_id": task.run_id,
        "import_job_id": task.import_job_id,
        "step": task.step,
        "pool": task.pool,
        "state": task.state,
        "attempt": int(task.attempt),
        "max_attempts": int(task.max_attempts),
        "business_revision": current_revision(task),
        "retryable": is_task_retryable(task),
        "cancel_requested": bool(task.cancel_requested),
        "error_code": task.error_code,
        "worker_id": task.worker_id,
        "next_run_at": task.next_run_at.isoformat() if task.next_run_at else None,
        "lease_until": task.lease_until.isoformat() if task.lease_until else None,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        "output_asset_id": task.output_asset_id,
    }
    if include_output:
        payload["output_json"] = task.output_json
    return JSONResponse(
        content={"data": payload, "request_id": request_id_of(request)},
        headers=etag_header(ETAG_TASK, payload["business_revision"]),
    )
