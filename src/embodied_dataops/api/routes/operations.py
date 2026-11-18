"""DO-A02 capabilities and operation-receipt routes (C1/C4)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal, Role
from embodied_dataops.api.deps import get_session, request_id_of
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.operations.contract import capabilities_payload
from embodied_dataops.operations.receipts import get_receipt, get_receipt_by_key, serialize_receipt

router = APIRouter(prefix="/api/v1", tags=["operations"])


def _principal(request: Request) -> Principal:
    return request.state.principal


def _require_operator(principal: Principal) -> None:
    """Receipts are operator-only (admin is a superset); an Agent LLM worker is viewer."""
    if principal.role not in (Role.OPERATOR, Role.ADMIN):
        raise ApiError(
            ErrorCode.FORBIDDEN,
            "Operation receipts require the operator role",
            details={"required_role": "operator", "role": principal.role.value},
        )


@router.get("/capabilities")
async def capabilities(request: Request) -> JSONResponse:
    """C1: contract start-up probe (viewer)."""
    _principal(request).require("read")
    return JSONResponse(
        content={"data": capabilities_payload(), "request_id": request_id_of(request)}
    )


@router.get("/operations/{operation_id}")
async def get_operation(
    request: Request, operation_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    """C4: receipt + REAL downstream state, operator-only."""
    principal = _principal(request)
    _require_operator(principal)
    receipt = get_receipt(db, operation_id)
    return JSONResponse(
        content={"data": serialize_receipt(db, receipt), "request_id": request_id_of(request)}
    )


@router.get("/operations/by-key/{idempotency_key}")
async def get_operation_by_key(
    request: Request, idempotency_key: str, db: Session = Depends(get_session)
) -> JSONResponse:
    """C4: look up a receipt by Idempotency-Key, operator-only."""
    principal = _principal(request)
    _require_operator(principal)
    receipt = get_receipt_by_key(db, idempotency_key)
    return JSONResponse(
        content={"data": serialize_receipt(db, receipt), "request_id": request_id_of(request)}
    )
