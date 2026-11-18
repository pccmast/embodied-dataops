"""FastAPI application factory (SPEC 9).

The app is deliberately thin: request validation and transaction commit only.
Heavy work is always an asynchronous task; endpoints return 202 with a task or
run identifier.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from embodied_dataops import __version__
from embodied_dataops.api.errors import install_error_handlers
from embodied_dataops.api.routes import ROUTERS
from embodied_dataops.config import get_settings
from embodied_dataops.logging_setup import configure_logging, log_context

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.service_name, settings.log_level)
    app = FastAPI(
        title="Embodied DataOps API",
        version=__version__,
        description=(
            "Multimodal embodied-data production platform: imports, canonical episodes, "
            "quality reports, review, dataset versions and feedback orders."
        ),
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        from embodied_dataops.api.auth import get_principal
        from embodied_dataops.api.errors import error_response
        from embodied_dataops.db.session import get_session_factory
        from embodied_dataops.errors import ApiError

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        path = request.url.path
        anonymous = (
            path in {"/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc"}
            or path.startswith("/api/v1/_storage/")
            or path == "/metrics"
        )
        if not anonymous:
            session = get_session_factory()()
            try:
                get_principal(request, session)
            except ApiError as exc:
                return error_response(
                    exc.status_code, exc.code, exc.message, request_id, exc.details
                )
            finally:
                session.close()
        with log_context(request_id=request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    install_error_handlers(app)
    for router in ROUTERS:
        app.include_router(router)

    # DO-A02 C1: expose the contract version on the runtime OpenAPI document as well
    # as in GET /capabilities.  Imported lazily so a partially implemented tree keeps
    # importing (operations.contract lands with D2-M4).
    def _openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        try:
            from embodied_dataops.operations.contract import CONTRACT_VERSION
        except ImportError:  # pragma: no cover - until D2-M4 lands
            CONTRACT_VERSION = "do-agent-contract/2.0.0"
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema["x-contract-version"] = CONTRACT_VERSION
        app.openapi_schema = schema
        return schema

    app.openapi = _openapi

    @app.get("/", include_in_schema=False)
    async def _root() -> JSONResponse:
        return JSONResponse({"service": "embodied-dataops", "version": __version__})

    return app


app = create_app()
