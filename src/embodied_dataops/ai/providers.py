"""Vision caption providers and call governance (SPEC_V2.1 §5.2, §5.4, §13, V21-09/12/13).

Two implementations, and their difference is load-bearing:

``ReplayProvider``
    No network, no key.  Returns a recorded document from a fixture directory.  Every
    result carries ``mode="REPLAY"``; a replay run can never satisfy ``AI_LIVE_COMPLETE``.

``DashScopeOpenAICompatibleProvider``
    The real path: OpenAI-compatible ``/chat/completions`` with multi-image input as
    base64 JPEG data URLs.  It never sends an internal MinIO URL, joint values or the
    raw video.

Governance lives here too because it is the same unit of reasoning:

* **Retry classification.** 429/502/503/504/timeout are retryable; 401/403, an
  unsupported model and an invalid image are not.  Retrying an auth failure just
  burns the budget and hides the real problem.
* **Budget.**  A reservation is taken *before* the call and settled after.  Two
  concurrent clips cannot both see enough balance and then both overspend.
* **Cost.**  A remote call is billed even when the response is lost, so the honest
  statement is "at most one committed product, possibly more than one billed
  attempt" -- recorded as ``UNKNOWN_UPSTREAM_CHARGE`` instead of a fake exact total.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from embodied_dataops.ai.schema import SCHEMA_VERSION, insufficient_evidence_document

MODE_LIVE = "LIVE"
MODE_REPLAY = "REPLAY"

PROMPT_VERSION = "ego-action-prompt/2.1"
MAX_IMAGES_PER_CLIP = 8
MAX_ENCODED_BYTES_PER_IMAGE = 524_288
DEFAULT_TIMEOUT_S = 60
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (5, 20)

#: Retryable HTTP statuses (SPEC_V2.1 §5.4): the request may succeed unchanged.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
#: Non-retryable statuses: repeating them cannot help.
FATAL_STATUS = frozenset({400, 401, 403, 404, 422})

ERROR_PROVIDER_UNAVAILABLE = "VLM_PROVIDER_UNAVAILABLE"
ERROR_BUDGET_EXCEEDED = "VLM_BUDGET_EXCEEDED"
ERROR_EGRESS_NOT_ALLOWED = "SOURCE_EGRESS_NOT_ALLOWED"
ERROR_IMAGE_TOO_LARGE = "VLM_IMAGE_TOO_LARGE"
ERROR_MODEL_OUTPUT_SCHEMA_INVALID = "MODEL_OUTPUT_SCHEMA_INVALID"


class ProviderError(RuntimeError):
    """A provider call that failed, with an explicit retryability verdict."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        retryable: bool,
        status: int | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.retryable = retryable
        self.status = status
        self.provider_request_id = provider_request_id


@dataclass(frozen=True)
class KeyframeImage:
    """One already-encoded keyframe handed to the provider."""

    grid_index: int
    jpeg_bytes: bytes
    sha256: str

    @classmethod
    def from_bytes(cls, grid_index: int, data: bytes) -> KeyframeImage:
        return cls(grid_index=int(grid_index), jpeg_bytes=data, sha256=hashlib.sha256(data).hexdigest())

    @property
    def encoded_size(self) -> int:
        return len(self.jpeg_bytes)


@dataclass(frozen=True)
class InferenceRequest:
    clip_id: str
    clip_start_ns: int
    clip_end_ns: int
    keyframes: tuple[KeyframeImage, ...]
    prompt_version: str = PROMPT_VERSION
    prompt_sha256: str = ""
    request_id: str = ""
    #: Per-call context (clip bounds and the timestamp of every attached frame).  It is
    #: not part of the versioned prompt, but it *is* part of the request hash.
    context_text: str = ""

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(frame.grid_index for frame in self.keyframes)

    def request_sha256(self) -> str:
        """Identity of the exact bytes we would send (never includes the API key)."""
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "clip_id": self.clip_id,
                    "clip_start_ns": self.clip_start_ns,
                    "clip_end_ns": self.clip_end_ns,
                    "prompt_version": self.prompt_version,
                    "prompt_sha256": self.prompt_sha256,
                    "context": self.context_text,
                    "model": getattr(self, "model_name", ""),
                },
                sort_keys=True,
            ).encode()
        )
        for frame in self.keyframes:
            digest.update(frame.sha256.encode())
        return digest.hexdigest()


@dataclass
class InferenceResult:
    """One logical model answer, already normalised for storage."""

    raw_text: str
    mode: str
    provider: str
    model_name: str
    latency_ms: int
    provider_request_id: str | None = None
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    attempt_no: int = 1
    status: str = "SUCCEEDED"
    error_code: str | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def is_live(self) -> bool:
        return self.mode == MODE_LIVE


class VisionCaptionProvider(Protocol):
    name: str
    mode: str
    model_name: str

    async def infer_clip(self, request: InferenceRequest) -> InferenceResult: ...


def _check_request(request: InferenceRequest) -> None:
    frames = request.keyframes
    if not frames:
        raise ProviderError(ERROR_PROVIDER_UNAVAILABLE, "clip has no keyframes", retryable=False)
    if len(frames) > MAX_IMAGES_PER_CLIP:
        raise ProviderError(
            ERROR_IMAGE_TOO_LARGE,
            f"{len(frames)} keyframes exceeds the {MAX_IMAGES_PER_CLIP}-image limit",
            retryable=False,
        )
    for frame in frames:
        if frame.encoded_size > MAX_ENCODED_BYTES_PER_IMAGE:
            raise ProviderError(
                ERROR_IMAGE_TOO_LARGE,
                f"frame {frame.grid_index} encodes to {frame.encoded_size} bytes "
                f"(limit {MAX_ENCODED_BYTES_PER_IMAGE})",
                retryable=False,
            )
        if not frame.jpeg_bytes.startswith(b"\xff\xd8"):
            raise ProviderError(
                ERROR_PROVIDER_UNAVAILABLE,
                f"frame {frame.grid_index} is not JPEG-encoded",
                retryable=False,
            )


@dataclass
class ReplayProvider:
    """Deterministic offline provider (SPEC_V2.1 §5.2 ``ReplayProvider``).

    ``responses`` maps ``clip_id`` to a raw JSON string.  Missing clips answer
    ``insufficient_evidence=true`` rather than inventing an action, which keeps the
    offline pipeline honest and idempotent.
    """

    responses: dict[str, str] = field(default_factory=dict)
    name: str = "replay"
    mode: str = MODE_REPLAY
    model_name: str = "replay-qwen2.5-vl-7b-instruct"
    fail_with: ProviderError | None = None

    @classmethod
    def from_directory(cls, directory: Path | str, **kwargs: Any) -> ReplayProvider:
        root = Path(directory)
        responses: dict[str, str] = {}
        if root.is_dir():
            for path in sorted(root.glob("*.json")):
                responses[path.stem] = path.read_text(encoding="utf-8")
        return cls(responses=responses, **kwargs)

    async def infer_clip(self, request: InferenceRequest) -> InferenceResult:
        _check_request(request)
        if self.fail_with is not None:
            raise self.fail_with
        started = time.monotonic()
        raw = self.responses.get(request.clip_id)
        if raw is None:
            raw = json.dumps(
                insufficient_evidence_document(request.clip_start_ns, request.clip_end_ns)
            )
        return InferenceResult(
            raw_text=raw,
            mode=MODE_REPLAY,
            provider=self.name,
            model_name=self.model_name,
            latency_ms=int((time.monotonic() - started) * 1000),
            provider_request_id=None,
            status="SUCCEEDED",
        )


def _data_url(frame: KeyframeImage) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(frame.jpeg_bytes).decode("ascii")


@dataclass
class DashScopeOpenAICompatibleProvider:
    """Real Qwen2.5-VL calls through the OpenAI-compatible endpoint.

    ``transport`` is injected so the governance tests can exercise every failure path
    (401 / 429 / timeout / truncated body) without spending money or needing a key.
    """

    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen2.5-vl-7b-instruct"
    prompt: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_output_tokens: int = 512
    temperature: float = 0.0
    name: str = "dashscope"
    mode: str = MODE_LIVE
    transport: Any = None

    def _payload(self, request: InferenceRequest) -> dict[str, Any]:
        text = self.prompt
        if request.context_text:
            text = f"{text}\n\n{request.context_text}"
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for frame in request.keyframes:
            content.append({"type": "image_url", "image_url": {"url": _data_url(frame)}})
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

    async def infer_clip(self, request: InferenceRequest) -> InferenceResult:
        _check_request(request)
        if not self.api_key:
            raise ProviderError(
                ERROR_PROVIDER_UNAVAILABLE, "no DASHSCOPE_API_KEY configured", retryable=False
            )
        import httpx  # noqa: PLC0415 -- optional at import time, required for live mode

        started = time.monotonic()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        try:
            if self.transport is not None:
                response = await self.transport.post(
                    url, json=self._payload(request), headers=headers, timeout=self.timeout_s
                )
            else:  # pragma: no cover - exercised only with a real key
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(url, json=self._payload(request), headers=headers)
        except Exception as exc:  # transport-level failure: timeout, DNS, reset
            raise ProviderError(
                ERROR_PROVIDER_UNAVAILABLE,
                f"transport failure: {type(exc).__name__}",
                retryable=True,
            ) from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        status = int(getattr(response, "status_code", 0) or 0)
        request_id = None
        try:
            request_id = response.headers.get("x-request-id")
        except Exception:  # pragma: no cover - defensive for fake transports
            request_id = None

        if status in FATAL_STATUS:
            raise ProviderError(
                ERROR_PROVIDER_UNAVAILABLE,
                f"provider rejected the request with HTTP {status}",
                retryable=False,
                status=status,
                provider_request_id=request_id,
            )
        if status in RETRYABLE_STATUS:
            raise ProviderError(
                ERROR_PROVIDER_UNAVAILABLE,
                f"provider is temporarily unavailable (HTTP {status})",
                retryable=True,
                status=status,
                provider_request_id=request_id,
            )
        if status != 200:
            raise ProviderError(
                ERROR_PROVIDER_UNAVAILABLE,
                f"unexpected HTTP {status}",
                retryable=False,
                status=status,
                provider_request_id=request_id,
            )

        try:
            body = response.json()
        except Exception as exc:
            raise ProviderError(
                ERROR_MODEL_OUTPUT_SCHEMA_INVALID,
                "response body is not JSON",
                retryable=True,
                status=status,
                provider_request_id=request_id,
            ) from exc

        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(
                ERROR_MODEL_OUTPUT_SCHEMA_INVALID,
                "response has no choices",
                retryable=True,
                status=status,
                provider_request_id=request_id,
            )
        raw_text = (choices[0].get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        return InferenceResult(
            raw_text=raw_text,
            mode=MODE_LIVE,
            provider=self.name,
            model_name=self.model_name,
            latency_ms=latency_ms,
            provider_request_id=request_id,
            usage_prompt_tokens=usage.get("prompt_tokens"),
            usage_completion_tokens=usage.get("completion_tokens"),
            status="SUCCEEDED",
            raw_response=body,
        )


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Raised before a call when the run cannot afford another one."""


@dataclass
class RunBudget:
    """In-memory view of one run's budget.

    The database owns the truth (``run_id`` row lock + committed reservations); this
    class is the arithmetic and the ordering, so the SQL can stay boring.
    """

    limit_usd: float
    estimated_cost_per_call_usd: float
    reserved_usd: float = 0.0
    spent_usd: float = 0.0
    calls: int = 0
    unknown_upstream_charge: int = 0

    def reserve(self) -> float:
        """Reserve one call's estimated cost, or raise if it would overspend."""
        projected = self.reserved_usd + self.estimated_cost_per_call_usd
        if self.spent_usd + projected > self.limit_usd + 1e-9:
            raise BudgetExceeded(
                f"budget {self.limit_usd:.2f} USD would be exceeded by another "
                f"{self.estimated_cost_per_call_usd:.4f} USD call "
                f"(spent {self.spent_usd:.4f}, reserved {self.reserved_usd:.4f})"
            )
        self.reserved_usd = projected
        return self.estimated_cost_per_call_usd

    def settle(self, actual_usd: float | None, *, charge_unknown: bool = False) -> None:
        """Release the reservation and record the actual (or unknown) charge."""
        self.reserved_usd = max(0.0, self.reserved_usd - self.estimated_cost_per_call_usd)
        self.calls += 1
        if actual_usd is None:
            if charge_unknown:
                self.unknown_upstream_charge += 1
                # Conservatively charge the estimate so the budget cannot drift up.
                self.spent_usd += self.estimated_cost_per_call_usd
            return
        self.spent_usd += max(0.0, float(actual_usd))

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit_usd": round(self.limit_usd, 6),
            "spent_usd": round(self.spent_usd, 6),
            "reserved_usd": round(self.reserved_usd, 6),
            "calls": self.calls,
            "unknown_upstream_charge": self.unknown_upstream_charge,
        }


def estimate_cost_usd(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    price_per_million_prompt: float | None,
    price_per_million_completion: float | None,
) -> float | None:
    """Cost from a *recorded* price snapshot; ``None`` when the price is unknown.

    Returning ``None`` is deliberate: a fabricated zero would make the usage report
    look complete when it is not.
    """
    if prompt_tokens is None or completion_tokens is None:
        return None
    if price_per_million_prompt is None or price_per_million_completion is None:
        return None
    return (
        prompt_tokens / 1_000_000.0 * price_per_million_prompt
        + completion_tokens / 1_000_000.0 * price_per_million_completion
    )


def backoff_seconds(attempt_no: int, clip_id: str) -> float:
    """Fixed schedule with a deterministic +/-10% jitter derived from ``clip_id``."""
    base = BACKOFF_SECONDS[min(max(attempt_no, 1), len(BACKOFF_SECONDS)) - 1]
    raw = hashlib.sha256(clip_id.encode("utf-8")).digest()[0]
    jitter = (raw / 255.0) * 0.2 - 0.1  # -10% .. +10%
    return float(base) * (1.0 + jitter)


def should_retry(error: ProviderError, attempt_no: int, *, max_attempts: int = MAX_ATTEMPTS) -> bool:
    return bool(error.retryable) and attempt_no < max_attempts


__all__ = [
    "MAX_ATTEMPTS",
    "MAX_IMAGES_PER_CLIP",
    "MODE_LIVE",
    "MODE_REPLAY",
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "BudgetExceeded",
    "DashScopeOpenAICompatibleProvider",
    "InferenceRequest",
    "InferenceResult",
    "KeyframeImage",
    "ProviderError",
    "ReplayProvider",
    "RunBudget",
    "VisionCaptionProvider",
    "backoff_seconds",
    "estimate_cost_usd",
    "should_retry",
]
