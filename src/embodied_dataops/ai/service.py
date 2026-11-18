"""AI pre-annotation orchestration shared by the DAG handlers (SPEC_V2.1 §5).

This module owns the parts that must not drift between handlers:

* **prompt identity** -- the prompt text and its SHA-256, so every inference row can be
  traced to the exact wording that produced it;
* **provider selection** -- ``disabled`` / ``replay`` / ``dashscope``, never a silent
  fallback from a missing key to a fake result;
* **budget admission against the database** -- spend already committed plus inferences
  currently RUNNING, so two workers cannot both see enough balance;
* **the retry loop** -- classification, attempt rows, backoff, and the rule that a
  retried remote call creates another *attempt*, never another *inference*.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from embodied_dataops.ai.providers import (
    ERROR_BUDGET_EXCEEDED,
    ERROR_MODEL_OUTPUT_SCHEMA_INVALID,
    ERROR_PROVIDER_UNAVAILABLE,
    MODE_LIVE,
    MODE_REPLAY,
    DashScopeOpenAICompatibleProvider,
    InferenceRequest,
    InferenceResult,
    KeyframeImage,
    ProviderError,
    ReplayProvider,
    VisionCaptionProvider,
    backoff_seconds,
    estimate_cost_usd,
    should_retry,
)
from embodied_dataops.ai.schema import ERROR_NOT_JSON, ValidationOutcome, validate_candidate_document
from embodied_dataops.config import Settings
from embodied_dataops.db.models import ModelCallAttempt, ModelInference, PipelineRun
from embodied_dataops.errors import ApiError, ErrorCode, TerminalTaskError

#: ``model_call_attempts.cost_state`` values (SPEC_V2.1 §5.4).
COST_KNOWN = "KNOWN"
COST_UNKNOWN_UPSTREAM_CHARGE = "UNKNOWN_UPSTREAM_CHARGE"
COST_NOT_BILLED = "NOT_BILLED"

#: ``model_call_attempts.attempt_no`` is constrained to 1..3 (SPEC_V2.1 §13).
MAX_RECORDED_ATTEMPTS = 3

#: Used only when the price snapshot is absent, so budget admission still has a number.
#: It is *not* reported as a cost: the report says "unknown" instead.
FALLBACK_ESTIMATE_USD = 0.01


@dataclass(frozen=True)
class PromptBundle:
    version: str
    sha256: str
    text: str


@dataclass
class BudgetState:
    limit_usd: float
    committed_usd: float
    running_reserved_usd: float
    estimate_usd: float

    @property
    def available_usd(self) -> float:
        return max(0.0, self.limit_usd - self.committed_usd - self.running_reserved_usd)

    def admitted(self) -> bool:
        return self.estimate_usd <= self.available_usd + 1e-9

    def admit(self) -> None:
        if not self.admitted():
            raise ProviderError(
                ERROR_BUDGET_EXCEEDED,
                f"run budget {self.limit_usd:.2f} USD exhausted: committed "
                f"{self.committed_usd:.4f}, reserved by running inferences "
                f"{self.running_reserved_usd:.4f}, next call needs {self.estimate_usd:.4f}",
                retryable=False,
            )


def load_prompt(settings: Settings) -> PromptBundle:
    """Read the fixed prompt file; a missing prompt is a configuration error, not a default."""
    path = Path(settings.vlm_prompt_path)
    if not path.is_file():
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            "VLM prompt file is missing",
            details={"path": str(path), "expected_version": settings.vlm_prompt_version},
        )
    text = path.read_text(encoding="utf-8")
    return PromptBundle(
        version=settings.vlm_prompt_version,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        text=text,
    )


def build_provider(settings: Settings, prompt: PromptBundle) -> VisionCaptionProvider:
    """Return the configured provider, or raise ``VLM_PROVIDER_UNAVAILABLE``."""
    provider = (settings.vlm_provider or "disabled").strip().lower()
    if provider == "replay":
        return ReplayProvider.from_directory(settings.vlm_replay_dir)
    if provider == "dashscope":
        api_key = settings.vlm_api_key.strip()
        if not api_key:
            raise TerminalTaskError(
                ErrorCode.VLM_PROVIDER_UNAVAILABLE,
                "VLM_PROVIDER=dashscope requires VLM_API_KEY (DASHSCOPE_API_KEY) in the environment",
                details={"provider": provider},
            )
        return DashScopeOpenAICompatibleProvider(
            api_key=api_key,
            base_url=settings.vlm_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=settings.vlm_model or "qwen2.5-vl-7b-instruct",
            prompt=prompt.text,
            timeout_s=float(settings.vlm_timeout_s),
            max_output_tokens=int(settings.vlm_max_output_tokens),
        )
    raise TerminalTaskError(
        ErrorCode.VLM_PROVIDER_UNAVAILABLE,
        "VLM provider is disabled; an AI pre-annotation run cannot start",
        details={"vlm_provider": settings.vlm_provider, "allowed": ["replay", "dashscope"]},
    )


def provider_mode(provider: VisionCaptionProvider) -> str:
    return MODE_LIVE if getattr(provider, "mode", MODE_REPLAY) == MODE_LIVE else MODE_REPLAY


def estimate_call_cost_usd(settings: Settings) -> float:
    """Conservative per-call estimate used for admission (never reported as spend)."""
    price_prompt = settings.vlm_price_per_million_prompt_usd
    price_completion = settings.vlm_price_per_million_completion_usd
    if price_prompt is None or price_completion is None:
        return FALLBACK_ESTIMATE_USD
    # 8 images x ~1000 tokens + prompt is the practical worst case for one clip.
    return (
        estimate_cost_usd(
            8000,
            int(settings.vlm_max_output_tokens),
            price_per_million_prompt=price_prompt,
            price_per_million_completion=price_completion,
        )
        or FALLBACK_ESTIMATE_USD
    )


def budget_state(session, run_id: str, settings: Settings) -> BudgetState:
    """Spend from the database: committed attempts plus inferences still running."""
    committed = session.scalar(
        select(func.coalesce(func.sum(ModelCallAttempt.estimated_cost_usd), 0)).where(
            ModelCallAttempt.inference_id.in_(
                select(ModelInference.id).where(ModelInference.run_id == run_id)
            )
        )
    )
    reserved = session.scalar(
        select(func.coalesce(func.sum(ModelInference.estimated_cost_usd), 0)).where(
            ModelInference.run_id == run_id,
            ModelInference.status == "RUNNING",
        )
    )
    return BudgetState(
        limit_usd=float(settings.vlm_budget_usd_per_run),
        committed_usd=float(committed or 0.0),
        running_reserved_usd=float(reserved or 0.0),
        estimate_usd=estimate_call_cost_usd(settings),
    )


def _lock_run(session, run_id: str) -> None:
    """Serialise budget admission for one run on its ``pipeline_runs`` row.

    Without this lock two workers both read "enough balance" and both call the
    provider; the run-level row lock plus the reservation written below makes the
    admission check atomic (SPEC_V2.1 §13 "调用前在 DB 以 run_id 锁定预算").
    """
    session.execute(select(PipelineRun.id).where(PipelineRun.id == run_id).with_for_update())


def reserve_budget(
    session,
    *,
    inference: ModelInference,
    settings: Settings,
) -> tuple[bool, BudgetState]:
    """Admit and reserve under the run lock (SPEC_V2.1 §13).

    Order matters.  The ``pipeline_runs`` row is locked first, so a competing worker
    blocks until this transaction ends; the balance is then read *without* this call's
    own reservation (nothing is written yet), and only an admitted call writes the
    ``RUNNING`` + estimated-cost row.  Committing that row is what makes the spend
    visible to the next worker -- checking before writing would let two workers both
    read "enough balance" (the defect the audit flagged).

    Returns ``(admitted, state)``.  When refused no reservation was written, the
    inference is marked ``FAILED``/``VLM_BUDGET_EXCEEDED`` with no cost, and the caller
    must not make a remote call.
    """
    _lock_run(session, inference.run_id)
    state = budget_state(session, inference.run_id, settings)
    if not state.admitted():
        inference.status = "FAILED"
        inference.error_code = ERROR_BUDGET_EXCEEDED
        inference.estimated_cost_usd = None
        inference.finished_at = _ts(time.time())
        session.flush()
        return False, state
    inference.status = "RUNNING"
    inference.estimated_cost_usd = Decimal(str(state.estimate_usd))
    session.flush()
    return True, state


def settle_inference(
    session,
    *,
    inference: ModelInference,
    status: str,
    error_code: str | None = None,
    cost_usd: float | None = None,
    latency_ms: int | None = None,
) -> None:
    """Write a terminal status/cost for the call itself (never a product claim)."""
    inference.status = status
    inference.error_code = error_code
    if cost_usd is not None:
        inference.estimated_cost_usd = Decimal(str(cost_usd))
    if latency_ms is not None:
        inference.latency_ms = int(latency_ms)
    inference.finished_at = _ts(time.time())
    session.flush()


def _attempt_cost_state(exc: ProviderError) -> str:
    """Classify what the attempt may have cost.

    A transport failure can happen *after* the provider accepted the request, so the
    charge is unknown rather than zero.  401/403/422 were rejected before any
    inference ran and are not billed.
    """
    if exc.status in {401, 403, 404, 422}:
        return COST_NOT_BILLED
    if exc.code in {ERROR_MODEL_OUTPUT_SCHEMA_INVALID}:
        return COST_UNKNOWN_UPSTREAM_CHARGE
    if exc.retryable or exc.status is None:
        return COST_UNKNOWN_UPSTREAM_CHARGE
    return COST_NOT_BILLED


def record_attempt(
    session,
    *,
    inference_id: str,
    attempt_no: int,
    started_at,
    finished_at=None,
    provider_request_id: str | None = None,
    response_sha256: str | None = None,
    usage_prompt_tokens: int | None = None,
    usage_completion_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
    response_status: int | None = None,
    error_code: str | None = None,
    cost_state: str | None = None,
) -> ModelCallAttempt:
    from embodied_dataops.ids import new_id  # noqa: PLC0415 -- avoids a cycle at import time

    row = ModelCallAttempt(
        id=new_id(),
        inference_id=inference_id,
        attempt_no=int(attempt_no),
        provider_request_id=provider_request_id,
        response_sha256=response_sha256,
        usage_prompt_tokens=usage_prompt_tokens,
        usage_completion_tokens=usage_completion_tokens,
        estimated_cost_usd=estimated_cost_usd,
        response_status=response_status,
        error_code=error_code,
        cost_state=cost_state or (COST_KNOWN if response_status == 200 else COST_NOT_BILLED),
        started_at=started_at,
        finished_at=finished_at,
    )
    session.add(row)
    session.flush()
    return row


@dataclass
class ClipInferenceOutcome:
    """What one clip's inference produced."""

    status: str
    attempts: int
    validation: ValidationOutcome | None
    result: InferenceResult | None
    error_code: str | None = None
    detail: str = ""
    cost_usd: float | None = None
    #: ``UNKNOWN_UPSTREAM_CHARGE`` when a failed attempt may still have been billed.
    cost_state: str = COST_KNOWN

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"


def perform_inference(
    session,
    *,
    inference: ModelInference,
    provider: VisionCaptionProvider,
    request: InferenceRequest,
    settings: Settings,
    clip_start_ns: int,
    clip_end_ns: int,
    sent_positions: list[int],
    sleep: Any = time.sleep,
    admitted: bool = False,
    persist: Any = None,
    attempt_offset: int = 0,
) -> ClipInferenceOutcome:
    """Call the provider with the fixed retry policy and validate the answer.

    The caller owns the ``ModelInference`` row's lifecycle; this function only appends
    attempt rows and returns the verdict.  ``inference.id`` must already be flushed.

    ``admitted=True`` means the caller already reserved budget under the run lock
    (:func:`reserve_budget`) and the reservation is in the database.  ``persist`` is
    invoked after every attempt row, so the audit writer can commit each real HTTP
    attempt before the task transaction is allowed to disappear.
    """
    if not admitted:
        try:
            budget = budget_state(session, inference.run_id, settings)
            budget.admit()
        except ProviderError as exc:
            return ClipInferenceOutcome(
                status="FAILED",
                attempts=0,
                validation=None,
                result=None,
                error_code=exc.code,
                detail=exc.detail,
                cost_state=COST_NOT_BILLED,
            )

    max_attempts = max(1, int(settings.vlm_max_attempts))
    last_error: ProviderError | None = None
    result: InferenceResult | None = None
    validation: ValidationOutcome | None = None
    attempts_used = 0

    def _flush() -> None:
        if persist is not None:
            persist()

    for local_attempt in range(1, max_attempts + 1):
        attempt_no = int(attempt_offset) + local_attempt
        if attempt_no > MAX_RECORDED_ATTEMPTS:
            # Attempt rows are durable now, so a task-level retry cannot silently start
            # a fourth billed call under the same inference identity.
            return ClipInferenceOutcome(
                status="FAILED",
                attempts=attempts_used,
                validation=validation,
                result=result,
                error_code=(last_error.code if last_error else ERROR_PROVIDER_UNAVAILABLE),
                detail=(
                    f"this inference already recorded {attempt_offset} attempts; the "
                    f"{MAX_RECORDED_ATTEMPTS}-attempt ceiling forbids another remote call"
                ),
                cost_state=(_attempt_cost_state(last_error) if last_error else COST_NOT_BILLED),
            )
        attempts_used = attempt_no
        started = time.time()
        try:
            result = _call(provider, request)
        except ProviderError as exc:
            last_error = exc
            record_attempt(
                session,
                inference_id=inference.id,
                attempt_no=attempt_no,
                started_at=_ts(started),
                finished_at=_ts(time.time()),
                provider_request_id=exc.provider_request_id,
                response_status=exc.status,
                error_code=exc.code,
                estimated_cost_usd=None,
                cost_state=_attempt_cost_state(exc),
            )
            _flush()
            if should_retry(exc, attempt_no, max_attempts=max_attempts):
                sleep(backoff_seconds(attempt_no, request.clip_id))
                continue
            break

        cost = estimate_cost_usd(
            result.usage_prompt_tokens,
            result.usage_completion_tokens,
            price_per_million_prompt=settings.vlm_price_per_million_prompt_usd,
            price_per_million_completion=settings.vlm_price_per_million_completion_usd,
        )
        record_attempt(
            session,
            inference_id=inference.id,
            attempt_no=attempt_no,
            started_at=_ts(started),
            finished_at=_ts(time.time()),
            provider_request_id=result.provider_request_id,
            response_sha256=hashlib.sha256(result.raw_text.encode("utf-8")).hexdigest(),
            usage_prompt_tokens=result.usage_prompt_tokens,
            usage_completion_tokens=result.usage_completion_tokens,
            estimated_cost_usd=cost,
            response_status=200,
            cost_state=COST_KNOWN if cost is not None else COST_UNKNOWN_UPSTREAM_CHARGE,
        )
        _flush()
        validation = validate_candidate_document(
            result.raw_text,
            expected_clip_start_ns=clip_start_ns,
            expected_clip_end_ns=clip_end_ns,
            sent_frame_indices=sent_positions,
        )
        if validation.is_valid:
            return ClipInferenceOutcome(
                status="SUCCEEDED",
                attempts=attempts_used,
                validation=validation,
                result=result,
                cost_usd=cost,
                cost_state=COST_KNOWN if cost is not None else COST_UNKNOWN_UPSTREAM_CHARGE,
            )

        # A non-JSON body may be retried exactly once with the same evidence (SPEC §5.4);
        # a structurally valid but semantically wrong answer is *not* retried.
        if ERROR_NOT_JSON in validation.error_codes and attempt_no < max_attempts:
            sleep(backoff_seconds(attempt_no, request.clip_id))
            continue
        return ClipInferenceOutcome(
            status="FAILED",
            attempts=attempts_used,
            validation=validation,
            result=result,
            error_code=ERROR_MODEL_OUTPUT_SCHEMA_INVALID,
            detail="; ".join(f"{error.code}: {error.detail}" for error in validation.errors[:5]),
            cost_usd=cost,
            cost_state=COST_UNKNOWN_UPSTREAM_CHARGE,
        )

    return ClipInferenceOutcome(
        status="FAILED",
        attempts=attempts_used,
        validation=validation,
        result=result,
        error_code=(last_error.code if last_error else ERROR_PROVIDER_UNAVAILABLE),
        detail=(last_error.detail if last_error else "provider retries exhausted"),
        cost_state=(_attempt_cost_state(last_error) if last_error else COST_NOT_BILLED),
    )


def _call(provider: VisionCaptionProvider, request: InferenceRequest) -> InferenceResult:
    import asyncio  # noqa: PLC0415 -- the handler is synchronous, the provider is async

    try:
        return asyncio.run(provider.infer_clip(request))
    except ProviderError:
        # Must come first: ProviderError subclasses RuntimeError, and the broad
        # handler below would relabel every provider failure (429, timeout, ...) as
        # "cannot run the async provider here", losing both the code and retryability.
        raise
    except RuntimeError as exc:  # already inside an event loop (unexpected in a worker)
        raise ProviderError(
            ERROR_PROVIDER_UNAVAILABLE,
            f"cannot run the async provider here: {exc}",
            retryable=False,
        ) from exc


def _ts(value: float):
    from datetime import UTC, datetime  # noqa: PLC0415

    return datetime.fromtimestamp(value, tz=UTC)


def build_request(
    *,
    clip_id: str,
    clip_start_ns: int,
    clip_end_ns: int,
    frames,
    prompt: PromptBundle,
    request_id: str,
) -> InferenceRequest:
    """Compose the request, including the per-frame timestamps the model must cite."""
    lines = [
        f"clip_start_ns={int(clip_start_ns)}",
        f"clip_end_ns={int(clip_end_ns)}",
        f"keyframe_count={len(frames)}",
        "frames:",
    ]
    for frame in frames:
        lines.append(
            f"  position={frame.position} grid_index={frame.grid_index} t_ns={frame.t_ns}"
        )
    return InferenceRequest(
        clip_id=clip_id,
        clip_start_ns=int(clip_start_ns),
        clip_end_ns=int(clip_end_ns),
        keyframes=tuple(
            KeyframeImage.from_bytes(frame.grid_index, frame.jpeg) for frame in frames
        ),
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
        request_id=request_id,
        context_text="\n".join(lines),
    )


def canonical_payload(payload: Any) -> bytes:
    """Deterministic bytes for an artifact we hash and store."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")


def as_error(exc: Exception) -> ApiError:
    if isinstance(exc, ApiError):
        return exc
    return ApiError(ErrorCode.INTERNAL_ERROR, str(exc)[:300])


__all__ = [
    "COST_KNOWN",
    "COST_NOT_BILLED",
    "COST_UNKNOWN_UPSTREAM_CHARGE",
    "MAX_RECORDED_ATTEMPTS",
    "BudgetState",
    "ClipInferenceOutcome",
    "PromptBundle",
    "build_provider",
    "build_request",
    "budget_state",
    "canonical_payload",
    "estimate_call_cost_usd",
    "load_prompt",
    "perform_inference",
    "provider_mode",
    "record_attempt",
    "reserve_budget",
    "settle_inference",
]
