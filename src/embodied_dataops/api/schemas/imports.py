"""Request/response schemas: imports and episodes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from embodied_dataops.api.schemas.common import StrictModel
from embodied_dataops.domain.enums import DataOrigin, SourceType


class InitiateImportRequest(StrictModel):
    source_type: SourceType
    filename: str = Field(min_length=1, max_length=500)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(min_length=64, max_length=64, pattern="^[0-9a-f]{64}$")
    data_origin: DataOrigin
    license: str = Field(default="", max_length=200)
    source_uri: str | None = Field(default=None, max_length=1024)
    source_revision: str | None = Field(default=None, max_length=200)
    mapping: dict[str, Any] | None = None
    # V2 D2-6 / DO-A02 C6: optional, immutable provenance to an approved
    # collection order. Absent = exact V1 behaviour.
    collection_order_id: str | None = Field(default=None, max_length=36)


class InitiateImportResponse(StrictModel):
    import_id: str
    put_url: str
    expires_at: str
    staging_key: str


class CompleteImportResponse(StrictModel):
    import_job_id: str
    state: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]
    import_id: str


class CreatePipelineRunRequest(StrictModel):
    pipeline_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")


class EpisodeSample(BaseModel):
    sample_index: int
    t_ns: int
    frame_index: int
    state: list[float] | None = None
    action: list[float] | None = None
    state_source_t_ns: int | None = None
    action_source_t_ns: int | None = None


class EpisodeListQuery(StrictModel):
    source_type: SourceType | None = None
    quality_status: str | None = Field(default=None, pattern="^(PASS|WARN|FAIL)$")
    task_label: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = None


class ReviewRequest(StrictModel):
    decision: Literal["APPROVED", "REJECTED"]
    note: str = Field(default="", max_length=2000)

    @field_validator("note")
    @classmethod
    def _note_len(cls, value: str) -> str:
        if len(value) > 2000:
            raise ValueError("note too long")
        return value
