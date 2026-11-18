"""Common API schema helpers."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    data: T
    request_id: str


class ListEnvelope(BaseModel, Generic[T]):
    data: list[T]
    next_cursor: str | None = None
    request_id: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorEnvelope(BaseModel):
    error: ErrorBody
    request_id: str


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
