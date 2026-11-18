"""Structured JSON logging with the field set required by SPEC 12.2."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

_log_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("dataops_log_context")

_FIELD_ORDER = (
    "ts",
    "level",
    "service",
    "event",
    "message",
    "request_id",
    "run_id",
    "task_id",
    "episode_id",
    "worker_id",
    "error_code",
    "duration_ms",
    "http_status",
    "http_method",
    "http_path",
)


@contextlib.contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind contextual fields (request_id, task_id, ...) for nested log calls."""
    current = dict(_log_context.get({}))
    current.update({k: v for k, v in fields.items() if v is not None})
    token = _log_context.set(current)
    try:
        yield
    finally:
        _log_context.reset(token)


def current_log_context() -> dict[str, Any]:
    return dict(_log_context.get({}))


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = dict(_log_context.get({}))
        payload["ts"] = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        payload["level"] = record.levelname
        payload["service"] = getattr(record, "service", None) or self.service
        payload["event"] = getattr(record, "event", None) or record.getMessage()
        payload["message"] = record.getMessage()
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update({k: v for k, v in extra.items() if v is not None})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        ordered: dict[str, Any] = {}
        for key in _FIELD_ORDER:
            if key in payload and payload[key] is not None:
                ordered[key] = payload[key]
        for key, value in payload.items():
            if key not in ordered and value is not None:
                ordered[key] = value
        return json.dumps(ordered, ensure_ascii=False, default=str)


def configure_logging(service: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "botocore", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event; unknown keyword fields are dropped by the formatter."""
    known = {
        "request_id",
        "run_id",
        "task_id",
        "episode_id",
        "worker_id",
        "error_code",
        "duration_ms",
        "http_status",
        "http_method",
        "http_path",
        "service",
        # SPEC_V2.1 §5.4 requires the AI log line to name the clip and the inference
        # row; without these two the fields were silently dropped by the formatter.
        "clip_id",
        "inference_id",
    }
    extra = {"event": event, "extra_fields": {k: v for k, v in fields.items() if k in known}}
    logger.log(level, event, extra=extra)
