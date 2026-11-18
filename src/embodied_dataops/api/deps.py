"""Request dependencies: database session, request id, idempotency, audit."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Request
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import IdempotencyRequest
from embodied_dataops.db.session import get_session_factory
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.logging_setup import log_context


def get_session(request: Request) -> Iterator[Session]:
    """One transaction per request; commit on success, rollback on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db(request: Request) -> Session:  # pragma: no cover - FastAPI alias
    return request.state.db


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def log_ctx(request: Request) -> Any:
    return log_context(request_id=request_id_of(request))


def encode_cursor(created_at: Any, entity_id: str) -> str:
    import base64
    import hmac

    payload = json.dumps({"t": created_at.isoformat(), "i": entity_id}, separators=(",", ":"))
    signature = hmac.new(
        get_settings().cursor_secret.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()[:16]
    return base64.urlsafe_b64encode(f"{payload}|{signature}".encode()).decode()


def decode_cursor(cursor: str) -> tuple[str, str]:
    import base64
    import hmac

    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        payload, signature = raw.rsplit("|", 1)
        expected = hmac.new(
            get_settings().cursor_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()[:16]
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        data = json.loads(payload)
        return data["t"], data["i"]
    except Exception as exc:
        raise ApiError(
            ErrorCode.INVALID_INPUT, "Invalid pagination cursor", details={"cursor": cursor[:64]}
        ) from exc


def _request_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def run_idempotent(
    session: Session,
    *,
    principal: Principal,
    route: str,
    idempotency_key: str | None,
    request_payload: Any,
    handler,
) -> tuple[int, dict[str, Any], bool]:
    """Execute ``handler() -> (status, payload)`` exactly once per key+body.

    Same key + same body  -> replay the stored response.
    Same key + other body -> 409 IDEMPOTENCY_KEY_REUSED.
    A PostgreSQL advisory lock serialises concurrent retries of the same key.
    """
    if not idempotency_key:
        raise ApiError(
            ErrorCode.IDEMPOTENCY_KEY_REQUIRED,
            "Idempotency-Key header is required for this endpoint",
        )
    if len(idempotency_key) > 200:
        raise ApiError(ErrorCode.INVALID_INPUT, "Idempotency-Key is too long")
    request_sha = _request_hash(request_payload)
    lock_key = hashlib.sha256(f"{principal.key_id}:{route}:{idempotency_key}".encode()).hexdigest()
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"), {"lock_key": lock_key[:60]}
    )
    existing = session.scalars(
        select(IdempotencyRequest).where(
            IdempotencyRequest.actor_key_id == principal.key_id,
            IdempotencyRequest.route == route,
            IdempotencyRequest.idempotency_key == idempotency_key,
        )
    ).first()
    if existing is not None:
        if existing.request_sha256 != request_sha:
            raise ApiError(
                ErrorCode.IDEMPOTENCY_KEY_REUSED,
                "Idempotency-Key was reused with a different request body",
            )
        return existing.response_status, existing.response_json, True
    status, payload = handler()
    record = IdempotencyRequest(
        id=new_id(),
        actor_key_id=principal.key_id,
        route=route,
        idempotency_key=idempotency_key,
        request_sha256=request_sha,
        response_status=status,
        response_json=payload,
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    session.add(record)
    session.flush()
    return status, payload, False


def new_request_id() -> str:
    return str(uuid.uuid4())
