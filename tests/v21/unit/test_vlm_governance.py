"""Provider and call-governance tests (SPEC_V2.1 §5.4/§13, V21-09/12/13).

Every remote path is exercised through an injected transport, so the failure taxonomy
(401 vs 429 vs timeout vs truncated body) is pinned without spending money or needing
a key.  ``ReplayProvider`` is tested for the opposite property: it must be provably
offline and provably *not* live evidence.
"""

from __future__ import annotations

import json

import pytest

from embodied_dataops.ai.providers import (
    BACKOFF_SECONDS,
    MAX_IMAGES_PER_CLIP,
    MODE_LIVE,
    MODE_REPLAY,
    BudgetExceeded,
    DashScopeOpenAICompatibleProvider,
    InferenceRequest,
    KeyframeImage,
    ProviderError,
    ReplayProvider,
    RunBudget,
    backoff_seconds,
    estimate_cost_usd,
    should_retry,
)

pytestmark = pytest.mark.no_db

#: Minimal byte sequences that pass the JPEG magic check.
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _frames(count: int, *, size: int = len(JPEG)) -> tuple[KeyframeImage, ...]:
    return tuple(
        KeyframeImage.from_bytes(index * 10, JPEG + b"\x00" * max(0, size - len(JPEG)))
        for index in range(count)
    )


def _request(frames: int = 4) -> InferenceRequest:
    return InferenceRequest(
        clip_id="clip_abc",
        clip_start_ns=0,
        clip_end_ns=8_000_000_000,
        keyframes=_frames(frames),
        prompt_sha256="d" * 64,
        request_id="req_1",
    )


class _Response:
    def __init__(self, status: int, body: object = None, headers: dict | None = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Transport:
    def __init__(self, response: _Response | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _ok_body(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 140},
    }


def _doc() -> dict:
    return {
        "schema_version": "ego-action-candidates/2.1",
        "clip_start_ns": 0,
        "clip_end_ns": 8_000_000_000,
        "actions": [
            {
                "verb": "pick_up",
                "noun": "cup",
                "start_ns": 1_000_000_000,
                "end_ns": 3_000_000_000,
                "evidence_frame_indices": [1, 2],
                "uncertainty": "medium",
            }
        ],
        "insufficient_evidence": False,
    }


# ---------------------------------------------------------------------------
# replay: offline and honest
# ---------------------------------------------------------------------------


async def test_replay_is_marked_replay_and_needs_no_network():
    provider = ReplayProvider()
    result = await provider.infer_clip(_request())
    assert provider.mode == MODE_REPLAY
    assert result.mode == MODE_REPLAY
    assert result.is_live is False
    assert result.provider_request_id is None


async def test_replay_without_a_recording_answers_insufficient_evidence():
    provider = ReplayProvider()
    result = await provider.infer_clip(_request())
    body = json.loads(result.raw_text)
    assert body["insufficient_evidence"] is True
    assert body["actions"] == []


async def test_replay_returns_the_recorded_document_for_a_known_clip():
    provider = ReplayProvider(responses={"clip_abc": json.dumps(_doc())})
    result = await provider.infer_clip(_request())
    assert json.loads(result.raw_text)["actions"][0]["noun"] == "cup"


async def test_replay_loads_a_fixture_directory(tmp_path):
    (tmp_path / "clip_abc.json").write_text(json.dumps(_doc()), encoding="utf-8")
    provider = ReplayProvider.from_directory(tmp_path)
    result = await provider.infer_clip(_request())
    assert json.loads(result.raw_text)["actions"][0]["verb"] == "pick_up"


async def test_replay_can_simulate_a_provider_error():
    error = ProviderError("VLM_PROVIDER_UNAVAILABLE", "down", retryable=True)
    provider = ReplayProvider(fail_with=error)
    with pytest.raises(ProviderError) as excinfo:
        await provider.infer_clip(_request())
    assert excinfo.value.retryable is True


# ---------------------------------------------------------------------------
# request validation, independent of provider
# ---------------------------------------------------------------------------


async def test_more_than_eight_images_is_rejected_before_any_call():
    provider = ReplayProvider()
    with pytest.raises(ProviderError) as excinfo:
        await provider.infer_clip(_request(MAX_IMAGES_PER_CLIP + 1))
    assert excinfo.value.code == "VLM_IMAGE_TOO_LARGE"
    assert excinfo.value.retryable is False


async def test_oversized_image_is_rejected():
    provider = ReplayProvider()
    request = InferenceRequest(
        clip_id="clip_big",
        clip_start_ns=0,
        clip_end_ns=1_000_000_000,
        keyframes=(KeyframeImage.from_bytes(0, JPEG + b"\x00" * 600_000),),
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.infer_clip(request)
    assert excinfo.value.code == "VLM_IMAGE_TOO_LARGE"


async def test_non_jpeg_payload_is_rejected():
    provider = ReplayProvider()
    request = InferenceRequest(
        clip_id="clip_png",
        clip_start_ns=0,
        clip_end_ns=1_000_000_000,
        keyframes=(KeyframeImage.from_bytes(0, b"\x89PNG\r\n\x1a\n" + b"\x00" * 16),),
    )
    with pytest.raises(ProviderError):
        await provider.infer_clip(request)


def test_request_hash_changes_with_the_images():
    first = _request(2).request_sha256()
    second = _request(3).request_sha256()
    assert first != second
    assert len(first) == 64


# ---------------------------------------------------------------------------
# live provider: failure taxonomy
# ---------------------------------------------------------------------------


async def test_missing_key_is_a_fatal_provider_error():
    provider = DashScopeOpenAICompatibleProvider(api_key="", transport=_Transport(_Response(200, {})))
    with pytest.raises(ProviderError) as excinfo:
        await provider.infer_clip(_request())
    assert excinfo.value.retryable is False
    assert excinfo.value.code == "VLM_PROVIDER_UNAVAILABLE"


@pytest.mark.parametrize("status", [401, 403, 404, 422])
async def test_auth_and_client_errors_are_not_retryable(status):
    transport = _Transport(_Response(status, {"error": "nope"}))
    provider = DashScopeOpenAICompatibleProvider(api_key="k", transport=transport)
    with pytest.raises(ProviderError) as excinfo:
        await provider.infer_clip(_request())
    assert excinfo.value.retryable is False
    assert excinfo.value.status == status


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_rate_limit_and_server_errors_are_retryable(status):
    transport = _Transport(_Response(status, {}))
    provider = DashScopeOpenAICompatibleProvider(api_key="k", transport=transport)
    with pytest.raises(ProviderError) as excinfo:
        await provider.infer_clip(_request())
    assert excinfo.value.retryable is True


async def test_transport_timeout_is_retryable():
    transport = _Transport(TimeoutError("read timeout"))
    provider = DashScopeOpenAICompatibleProvider(api_key="k", transport=transport)
    with pytest.raises(ProviderError) as excinfo:
        await provider.infer_clip(_request())
    assert excinfo.value.retryable is True
    assert "transport failure" in excinfo.value.detail


async def test_truncated_json_body_is_retryable_and_flagged_as_schema_invalid():
    transport = _Transport(_Response(200, ValueError("bad json")))
    provider = DashScopeOpenAICompatibleProvider(api_key="k", transport=transport)
    with pytest.raises(ProviderError) as excinfo:
        await provider.infer_clip(_request())
    assert excinfo.value.code == "MODEL_OUTPUT_SCHEMA_INVALID"
    assert excinfo.value.retryable is True


async def test_successful_call_records_usage_and_request_id():
    transport = _Transport(_Response(200, _ok_body(json.dumps(_doc())), {"x-request-id": "rid_9"}))
    provider = DashScopeOpenAICompatibleProvider(api_key="k", transport=transport)
    result = await provider.infer_clip(_request())
    assert result.mode == MODE_LIVE
    assert result.is_live is True
    assert result.usage_prompt_tokens == 1200
    assert result.usage_completion_tokens == 140
    assert result.provider_request_id == "rid_9"


async def test_payload_uses_data_urls_and_never_a_storage_url():
    transport = _Transport(_Response(200, _ok_body(json.dumps(_doc()))))
    provider = DashScopeOpenAICompatibleProvider(
        api_key="k", transport=transport, prompt="return json"
    )
    await provider.infer_clip(_request(2))
    payload = transport.calls[0]["json"]
    images = [part for part in payload["messages"][0]["content"] if part["type"] == "image_url"]
    assert len(images) == 2
    for part in images:
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "minio" not in json.dumps(payload).lower()
    assert payload["temperature"] == 0.0
    assert payload["response_format"] == {"type": "json_object"}


async def test_api_key_never_leaks_into_the_payload():
    transport = _Transport(_Response(200, _ok_body(json.dumps(_doc()))))
    provider = DashScopeOpenAICompatibleProvider(api_key="secret-key-value", transport=transport)
    await provider.infer_clip(_request(1))
    assert "secret-key-value" not in json.dumps(transport.calls[0]["json"])


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_budget_reserves_then_settles():
    budget = RunBudget(limit_usd=2.0, estimated_cost_per_call_usd=0.05)
    reserved = budget.reserve()
    assert reserved == pytest.approx(0.05)
    assert budget.reserved_usd == pytest.approx(0.05)
    budget.settle(0.03)
    assert budget.spent_usd == pytest.approx(0.03)
    assert budget.reserved_usd == pytest.approx(0.0)
    assert budget.calls == 1


def test_budget_refuses_a_call_that_would_overspend():
    budget = RunBudget(limit_usd=0.10, estimated_cost_per_call_usd=0.06)
    budget.reserve()
    budget.settle(0.06)
    with pytest.raises(BudgetExceeded):
        budget.reserve()


def test_two_reservations_cannot_both_see_the_same_balance():
    """The whole point of reserving before the call."""
    budget = RunBudget(limit_usd=0.10, estimated_cost_per_call_usd=0.06)
    budget.reserve()
    with pytest.raises(BudgetExceeded):
        budget.reserve()


def test_lost_response_is_recorded_as_unknown_upstream_charge():
    budget = RunBudget(limit_usd=1.0, estimated_cost_per_call_usd=0.05)
    budget.reserve()
    budget.settle(None, charge_unknown=True)
    assert budget.unknown_upstream_charge == 1
    # conservatively charged at the estimate so the budget cannot drift upwards
    assert budget.spent_usd == pytest.approx(0.05)


def test_unknown_charge_does_not_invent_spend_when_not_charged():
    budget = RunBudget(limit_usd=1.0, estimated_cost_per_call_usd=0.05)
    budget.reserve()
    budget.settle(None, charge_unknown=False)
    assert budget.spent_usd == 0.0
    assert budget.calls == 1


def test_cost_is_none_when_the_price_snapshot_is_missing():
    assert estimate_cost_usd(1000, 100, price_per_million_prompt=None, price_per_million_completion=None) is None
    assert estimate_cost_usd(None, None, price_per_million_prompt=1.0, price_per_million_completion=2.0) is None
    assert estimate_cost_usd(
        1_000_000, 500_000, price_per_million_prompt=1.0, price_per_million_completion=2.0
    ) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# retry policy
# ---------------------------------------------------------------------------


def test_should_retry_respects_attempt_ceiling():
    retryable = ProviderError("X", "y", retryable=True)
    fatal = ProviderError("X", "y", retryable=False)
    assert should_retry(retryable, 1, max_attempts=3) is True
    assert should_retry(retryable, 3, max_attempts=3) is False
    assert should_retry(fatal, 1, max_attempts=3) is False


def test_backoff_is_deterministic_and_within_ten_percent():
    first = backoff_seconds(1, "clip_abc")
    second = backoff_seconds(1, "clip_abc")
    assert first == second
    base = BACKOFF_SECONDS[0]
    assert base * 0.9 <= first <= base * 1.1


def test_backoff_uses_the_second_schedule_for_the_second_retry():
    value = backoff_seconds(2, "clip_abc")
    base = BACKOFF_SECONDS[1]
    assert base * 0.9 <= value <= base * 1.1
