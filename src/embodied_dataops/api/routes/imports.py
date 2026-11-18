"""Import upload routes (SPEC 9.2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of, run_idempotent
from embodied_dataops.api.schemas.imports import (
    InitiateImportRequest,
)
from embodied_dataops.imports.service import complete_import, get_import_detail, initiate_import

router = APIRouter(prefix="/api/v1/imports", tags=["imports"])


def _principal(request: Request) -> Principal:
    return request.state.principal


@router.post("/initiate", status_code=201)
async def initiate(
    request: Request,
    payload: InitiateImportRequest,
    db: Session = Depends(get_session),
) -> JSONResponse:
    principal = _principal(request)
    principal.require("import:write")
    result = initiate_import(
        db, principal=principal, payload=payload.model_dump(mode="json"), request_id=request_id_of(request)
    )
    return JSONResponse(
        status_code=201,
        content={"data": result, "request_id": request_id_of(request)},
    )


@router.post("/{import_id}/complete", status_code=202)
async def complete(
    request: Request,
    import_id: str,
    db: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    principal = _principal(request)
    principal.require("import:write")
    rid = request_id_of(request)

    def handler():
        return 202, complete_import(db, principal=principal, import_id=import_id, request_id=rid)

    status, body, _replayed = run_idempotent(
        db,
        principal=principal,
        route="/api/v1/imports/{import_id}/complete",
        idempotency_key=idempotency_key,
        request_payload={"import_id": import_id},
        handler=handler,
    )
    return JSONResponse(status_code=status, content={"data": body, "request_id": rid})


@router.get("/{import_id}")
async def detail(
    request: Request,
    import_id: str,
    db: Session = Depends(get_session),
) -> JSONResponse:
    _principal(request).require("read")
    return JSONResponse(
        content={"data": get_import_detail(db, import_id), "request_id": request_id_of(request)}
    )
