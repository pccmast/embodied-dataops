"""Health endpoints (SPEC 2, 9.3).  Anonymous, internal-only."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from embodied_dataops.config import get_settings
from embodied_dataops.db.session import get_session_factory
from embodied_dataops.storage.base import get_storage
from embodied_dataops.workflow.queue import create_client, ping

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health/live")
async def live() -> dict:
    return {"status": "ok", "service": get_settings().service_name}


@router.get("/health/ready")
async def ready() -> JSONResponse:
    components: dict[str, str] = {}
    healthy = True

    try:
        session = get_session_factory()()
        try:
            session.execute(text("SELECT 1"))
            components["postgres"] = "ok"
        finally:
            session.close()
    except Exception as exc:  # pragma: no cover - environment dependent
        components["postgres"] = f"error: {type(exc).__name__}"
        healthy = False

    try:
        components["redis"] = "ok" if ping(create_client()) else "error"
        healthy = healthy and components["redis"] == "ok"
    except Exception as exc:
        components["redis"] = f"error: {type(exc).__name__}"
        healthy = False

    try:
        storage_ok = get_storage().healthcheck()
        components["object_store"] = "ok" if storage_ok else "error"
        healthy = healthy and storage_ok
    except Exception as exc:
        components["object_store"] = f"error: {type(exc).__name__}"
        healthy = False

    body = {"status": "ready" if healthy else "not_ready", "components": components}
    return JSONResponse(body, status_code=200 if healthy else 503)
