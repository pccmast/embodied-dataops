"""Uniform error handling and response envelopes (SPEC 9.1)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from embodied_dataops.errors import ApiError, ErrorCode

logger = logging.getLogger(__name__)


def envelope(data: Any, request_id: str) -> dict[str, Any]:
    return {"data": data, "request_id": request_id}


def list_envelope(data: list[Any], next_cursor: str | None, request_id: str) -> dict[str, Any]:
    return {"data": data, "next_cursor": next_cursor, "request_id": request_id}


def error_response(
    status_code: int, code: ErrorCode | str, message: str, request_id: str, details: dict | None = None
) -> JSONResponse:
    code_value = code.value if isinstance(code, ErrorCode) else code
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code_value, "message": message, "details": details or {}},
            "request_id": request_id,
        },
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        return error_response(exc.status_code, exc.code, exc.message, request_id, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        simplified = [
            {"loc": list(err.get("loc", [])), "msg": err.get("msg", ""), "type": err.get("type", "")}
            for err in exc.errors()[:20]
        ]
        return error_response(
            422, ErrorCode.VALIDATION_FAILED, "Request validation failed", request_id, {"errors": simplified}
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        code = {
            401: ErrorCode.UNAUTHENTICATED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            405: ErrorCode.METHOD_NOT_ALLOWED,
        }.get(exc.status_code, ErrorCode.INVALID_INPUT)
        return error_response(exc.status_code, code, str(exc.detail), request_id)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("unhandled_api_error", extra={"event": "unhandled_api_error"})
        return error_response(500, ErrorCode.INTERNAL_ERROR, "Internal server error", request_id)
