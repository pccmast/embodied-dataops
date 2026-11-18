"""Request schemas for the DO-A02 controlled writes (C3)."""

from __future__ import annotations

import uuid

from pydantic import Field, field_validator

from embodied_dataops.api.schemas.common import StrictModel

_UUID_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"


class CreatePipelineRunRequestV2(StrictModel):
    """Body of ``POST /episodes/{id}/pipeline-runs`` (DO-A02 C3)."""

    pipeline_version: str = Field(default="2.0.0", pattern=r"^\d+\.\d+\.\d+$")


class RetryTaskRequest(StrictModel):
    """Body of ``POST /tasks/{id}/retry`` (DO-A02 C3)."""

    expected_run_id: str = Field(pattern=_UUID_RE)

    @field_validator("expected_run_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        uuid.UUID(value)
        return value


class AnalyzeGapsRequestV2(StrictModel):
    """Body of ``POST /evaluations/{id}/analyze-gaps`` (DO-A02 C3)."""

    success_rate_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    min_trials: int = Field(default=20, ge=1, le=1_000_000)


class OperationAccepted(StrictModel):
    """Shape of the 202 payload (kept explicit for the OpenAPI contract)."""

    operation_id: str
    state: str
    effect_state: str
    new_run_id: str | None = None
    analysis_id: str | None = None


class OperationReceiptView(StrictModel):
    operation_id: str
    action_kind: str
    target_id: str
    source_business_revision: int
    effect_state: str
    downstream_kind: str | None = None
    downstream_id: str | None = None
    downstream_state: str | None = None
    downstream_updated_at: str | None = None
    error_code: str | None = None


class CapabilitiesView(StrictModel):
    contract_version: str
    canonical_versions: list[str]
    robot_profiles: list[str]
    qc_policy_versions: list[str]
    conditional_writes: bool
    operation_receipts: bool
    diagnostic_queries: bool
