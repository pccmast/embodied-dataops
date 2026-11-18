"""Metrics endpoint (SPEC 9.3, 12.2)."""

from __future__ import annotations

import ipaddress

from fastapi import APIRouter, Request
from fastapi.responses import Response

from embodied_dataops.api.auth import authenticate
from embodied_dataops.db.session import get_session_factory
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.observability.metrics import render

router = APIRouter(tags=["observability"])


def _is_internal_client(request: Request) -> bool:
    host = request.client.host if request.client else "0.0.0.0"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """Prometheus text format.

    Prometheus scrapes without an API key, so anonymous access is limited to
    internal (loopback/private) clients; external callers must present an
    admin key.
    """
    if not _is_internal_client(request):
        header = request.headers.get("authorization") or ""
        if not header.lower().startswith("bearer "):
            raise ApiError(ErrorCode.UNAUTHENTICATED, "Authentication required for /metrics")
        session = get_session_factory()()
        try:
            authenticate(session, header.split(" ", 1)[1].strip()).require("metrics")
        finally:
            session.close()
    body, content_type = render()
    return Response(content=body, media_type=content_type)
