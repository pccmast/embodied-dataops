"""AI workflow wiring tests: mode gating, budget admission, DAG selection.

Everything here is pure or in-process -- no PostgreSQL, no network, no model.
"""

from __future__ import annotations

import pytest

from embodied_dataops.ai.providers import (
    ERROR_BUDGET_EXCEEDED,
    MODE_REPLAY,
    InferenceRequest,
    KeyframeImage,
    ProviderError,
    ReplayProvider,
    RunBudget,
)
from embodied_dataops.ai.service import (
    BudgetState,
    build_provider,
    build_request,
    estimate_call_cost_usd,
    load_prompt,
    provider_mode,
)
from embodied_dataops.api.routes.ai_preannotation import _check_mode
from embodied_dataops.config import Settings
from embodied_dataops.domain.enums import PIPELINE_DAG, PIPELINE_DAG_V21, Step
from embodied_dataops.errors import ApiError, ErrorCode, TerminalTaskError
from embodied_dataops.workflow.dag import FAN_OUT_STEPS, _topological_steps

pytestmark = pytest.mark.no_db

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def _settings(**overrides) -> Settings:
    base = {"vlm_provider": "replay", "vlm_api_key": "", "vlm_budget_usd_per_run": 2.0}
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# mode gating
# ---------------------------------------------------------------------------


def test_live_mode_requires_the_dashscope_provider():
    with pytest.raises(ApiError) as excinfo:
        _check_mode("live", _settings(vlm_provider="replay"))
    assert excinfo.value.code is ErrorCode.VLM_PROVIDER_UNAVAILABLE


def test_live_mode_is_allowed_only_when_the_live_provider_is_configured():
    _check_mode("live", _settings(vlm_provider="dashscope"))
    _check_mode("replay", _settings(vlm_provider="replay"))


def test_replay_mode_is_refused_when_the_platform_is_disabled():
    with pytest.raises(ApiError):
        _check_mode("replay", _settings(vlm_provider="disabled"))


def test_disabled_provider_cannot_build_a_provider():
    prompt = load_prompt(_settings())
    with pytest.raises(TerminalTaskError) as excinfo:
        build_provider(_settings(vlm_provider="disabled"), prompt)
    assert excinfo.value.code is ErrorCode.VLM_PROVIDER_UNAVAILABLE


def test_live_provider_without_a_key_is_refused_not_downgraded():
    prompt = load_prompt(_settings())
    with pytest.raises(TerminalTaskError) as excinfo:
        build_provider(_settings(vlm_provider="dashscope", vlm_api_key=""), prompt)
    assert excinfo.value.code is ErrorCode.VLM_PROVIDER_UNAVAILABLE
    assert "VLM_API_KEY" in excinfo.value.message


def test_replay_provider_reports_replay_mode():
    prompt = load_prompt(_settings())
    provider = build_provider(_settings(vlm_provider="replay"), prompt)
    assert isinstance(provider, ReplayProvider)
    assert provider_mode(provider) == MODE_REPLAY


# ---------------------------------------------------------------------------
# prompt identity
# ---------------------------------------------------------------------------


def test_prompt_file_is_versioned_and_stable():
    first = load_prompt(_settings())
    second = load_prompt(_settings())
    assert first.version == "ego-action-prompt/2.1"
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert "JSON" in first.text


def test_missing_prompt_file_is_a_configuration_error(tmp_path):
    with pytest.raises(TerminalTaskError):
        load_prompt(_settings(vlm_prompt_path=tmp_path / "nope.md"))


# ---------------------------------------------------------------------------
# request composition
# ---------------------------------------------------------------------------


def test_request_carries_clip_bounds_and_frame_timestamps():
    prompt = load_prompt(_settings())
    frames = [
        type("F", (), {"position": 0, "grid_index": 0, "t_ns": 0, "jpeg": JPEG})(),
        type("F", (), {"position": 1, "grid_index": 10, "t_ns": 1_000_000_000, "jpeg": JPEG})(),
    ]
    request = build_request(
        clip_id="c1",
        clip_start_ns=0,
        clip_end_ns=8_000_000_000,
        frames=frames,
        prompt=prompt,
        request_id="t1",
    )
    assert "clip_start_ns=0" in request.context_text
    assert "position=1 grid_index=10 t_ns=1000000000" in request.context_text
    assert len(request.keyframes) == 2


def test_request_hash_covers_the_context():
    prompt = load_prompt(_settings())
    frames = [type("F", (), {"position": 0, "grid_index": 0, "t_ns": 0, "jpeg": JPEG})()]
    first = build_request(
        clip_id="c1", clip_start_ns=0, clip_end_ns=1, frames=frames, prompt=prompt, request_id="a"
    )
    second = build_request(
        clip_id="c1", clip_start_ns=0, clip_end_ns=2, frames=frames, prompt=prompt, request_id="a"
    )
    assert first.request_sha256() != second.request_sha256()


def test_prompt_sha_reaches_the_request():
    prompt = load_prompt(_settings())
    frames = [type("F", (), {"position": 0, "grid_index": 0, "t_ns": 0, "jpeg": JPEG})()]
    request = build_request(
        clip_id="c1", clip_start_ns=0, clip_end_ns=1, frames=frames, prompt=prompt, request_id="a"
    )
    assert request.prompt_sha256 == prompt.sha256
    assert request.prompt_version == prompt.version


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_budget_state_admits_while_there_is_room():
    state = BudgetState(limit_usd=2.0, committed_usd=0.5, running_reserved_usd=0.25, estimate_usd=0.1)
    state.admit()  # must not raise
    assert state.available_usd == pytest.approx(1.25)


def test_budget_state_refuses_when_the_estimate_exceeds_what_is_left():
    state = BudgetState(limit_usd=1.0, committed_usd=0.95, running_reserved_usd=0.0, estimate_usd=0.1)
    with pytest.raises(ProviderError) as excinfo:
        state.admit()
    assert excinfo.value.code == ERROR_BUDGET_EXCEEDED
    assert excinfo.value.retryable is False


def test_running_inferences_count_as_reserved():
    state = BudgetState(limit_usd=1.0, committed_usd=0.0, running_reserved_usd=0.95, estimate_usd=0.1)
    with pytest.raises(ProviderError):
        state.admit()


def test_estimate_falls_back_to_a_conservative_number_without_a_price_snapshot():
    assert estimate_call_cost_usd(_settings()) == pytest.approx(0.01)


def test_estimate_uses_the_price_snapshot_when_present():
    value = estimate_call_cost_usd(
        _settings(
            vlm_price_per_million_prompt_usd=1.0,
            vlm_price_per_million_completion_usd=2.0,
        )
    )
    # 8000 prompt tokens at 1 USD/M + 512 completion tokens at 2 USD/M.
    assert value == pytest.approx(8000 / 1_000_000 * 1.0 + 512 / 1_000_000 * 2.0)
    assert value != pytest.approx(0.01)  # the fallback must not win when a price exists


def test_run_budget_settles_and_releases():
    budget = RunBudget(limit_usd=2.0, estimated_cost_per_call_usd=0.05)
    budget.reserve()
    budget.settle(0.02)
    assert budget.spent_usd == pytest.approx(0.02)
    assert budget.reserved_usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# DAG selection
# ---------------------------------------------------------------------------


def test_v21_run_materialises_the_ai_branch_but_not_the_fan_out():
    order = _topological_steps(PIPELINE_DAG_V21)
    assert Step.EXTRACT_KEYFRAMES in order
    assert Step.MERGE_CANDIDATE_ANNOTATIONS in order
    assert Step.VALIDATE_AI_OUTPUT in order
    # The per-clip tasks are created at run time, so the builder must not pre-create one.
    assert Step.VLM_PREANNOTATE_CLIP not in order
    assert Step.VLM_PREANNOTATE_CLIP in FAN_OUT_STEPS


def test_v2_run_is_untouched():
    order = _topological_steps(PIPELINE_DAG)
    assert Step.EXTRACT_KEYFRAMES not in order
    assert order[-1] is Step.READY_FOR_REVIEW


def test_replay_request_rejects_more_than_eight_images():
    provider = ReplayProvider()
    request = InferenceRequest(
        clip_id="c1",
        clip_start_ns=0,
        clip_end_ns=1,
        keyframes=tuple(KeyframeImage.from_bytes(i, JPEG) for i in range(9)),
    )
    with pytest.raises(ProviderError):
        import asyncio

        asyncio.run(provider.infer_clip(request))
