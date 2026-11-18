"""Pydantic envelopes for the V2.1 AI pre-annotation API (SPEC_V2.1 §5.5)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AiPreannotationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_pipeline_version: Literal["2.1.0"] = "2.1.0"
    mode: Literal["live", "replay"] = "replay"
    keyframe_policy: Literal["fps1_clips8s"] = "fps1_clips8s"


class AiPreannotationRunOut(BaseModel):
    run_id: str
    state: str
    mode: str
    pipeline_version: str
    annotation_set_id: str | None = None


class AiClipOut(BaseModel):
    clip_id: str
    start_ns: int
    end_ns: int
    frame_count: int
    inference_id: str | None = None
    mode: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    status: str | None = None
    insufficient_evidence: bool = False
    action_count: int = 0


class AiCandidateOut(BaseModel):
    candidate_id: str
    clip_id: str
    verb: str
    noun: str
    start_ns: int
    end_ns: int
    evidence_frame_indices: list[int]
    uncertainty: str
    confidence_source: str = "MODEL_QUALITATIVE"
    review_state: str = "PENDING"


class AiPreannotationsOut(BaseModel):
    episode_id: str
    annotation_set_id: str
    annotation_kind: str
    status: str
    artifact_sha256: str | None = None
    clips: list[AiClipOut]
    candidates: list[AiCandidateOut]
    schema_errors: list[dict[str, Any]] = Field(default_factory=list)
    quality: dict[str, Any] = Field(default_factory=dict)


class AiInferenceOut(BaseModel):
    inference_id: str
    run_id: str
    clip_id: str
    provider: str
    model_name: str
    mode: str
    prompt_version: str
    prompt_sha256: str
    status: str
    error_code: str | None = None
    latency_ms: int | None = None
    estimated_cost_usd: float | None = None
    attempts: int = 0
    output_sha256: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class AiActionInterval(BaseModel):
    """One reviewed action interval (what a human submits, not what a model guesses)."""

    model_config = ConfigDict(extra="forbid")

    verb: Literal["pick_up", "put_down", "move", "open", "close", "pour", "cut", "other", "uncertain"]
    noun: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)
    evidence_frame_indices: list[int] = Field(min_length=1)


class AiRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_revision_no: int | None = None
    actions: list[AiActionInterval] = Field(min_length=0, max_length=200)
    note: str = ""


class AiRevisionOut(BaseModel):
    annotation_set_id: str
    revision_id: str
    revision_no: int
    state: str
    content_sha256: str
    etag: str
    action_count: int


class AiDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["APPROVED", "REJECTED"]
    note: str = ""


class AiDecisionOut(BaseModel):
    annotation_set_id: str
    revision_id: str
    state: str
    status: str
    approved_revision_id: str | None = None


class AiQualitySummaryOut(BaseModel):
    episode_id: str
    annotation_set_id: str | None = None
    run_id: str | None = None
    mode: str | None = None
    live_evidence: bool = False
    schema_valid_rate: float | None = None
    call_success_rate: float | None = None
    approval_rate: float | None = None
    action_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    model_versions: list[str] = Field(default_factory=list)
    prompt_versions: list[str] = Field(default_factory=list)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)


class ModelUsageOut(BaseModel):
    run_id: str
    calls: int
    calls_succeeded: int
    retries: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float | None = None
    cost_known: bool
    budget_usd: float
    over_budget: bool
    models: list[str] = Field(default_factory=list)


class _Envelope(BaseModel):
    data: dict[str, Any]
    request_id: str


class AiPreannotationRunResponse(_Envelope):
    data: AiPreannotationRunOut  # type: ignore[assignment]


class AiPreannotationsResponse(_Envelope):
    data: AiPreannotationsOut  # type: ignore[assignment]


class AiInferenceResponse(_Envelope):
    data: AiInferenceOut  # type: ignore[assignment]


class AiRevisionResponse(_Envelope):
    data: AiRevisionOut  # type: ignore[assignment]


class AiDecisionResponse(_Envelope):
    data: AiDecisionOut  # type: ignore[assignment]


class AiQualitySummaryResponse(_Envelope):
    data: AiQualitySummaryOut  # type: ignore[assignment]


class ModelUsageResponse(_Envelope):
    data: ModelUsageOut  # type: ignore[assignment]
