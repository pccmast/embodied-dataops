"""官方 ACT processor 的冻结与一致性校验（EDO-TRAIN-PREP-1.1 §8.2，TP-12/TP-13）。

这里最重要的是**两条负例**：
常数维必须得到有限且可逆的结果；任何绕过 ``build_act_processors`` 的 stats 覆盖
必须被 ``verify_processor_stats`` 抓住。
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from embodied_dataops.errors import ApiError, ErrorCode  # noqa: E402
from embodied_dataops.training.contract import (  # noqa: E402
    VerifiedSource,
    parse_recipe,
    resolve_recipe,
)
from embodied_dataops.training.processors import (  # noqa: E402
    PROCESSOR_EPS,
    build_act_processors,
    verify_processor_stats,
)
from embodied_dataops.training.split import SplitResult  # noqa: E402
from embodied_dataops.training.stats import fit_train_stats  # noqa: E402
from embodied_dataops.training.windows import EpisodeFrames, build_window_index  # noqa: E402

pytestmark = pytest.mark.no_db


class _Ramp:
    """状态/动作恒等于 frame index（单维），便于手算 mean/std。"""

    def state(self, episode_id: str, frame_index: int) -> list[float]:
        return [float(frame_index)]

    def action(self, episode_id: str, frame_index: int) -> list[float]:
        return [float(frame_index)]


class _Constant(_Ramp):
    def action(self, episode_id: str, frame_index: int) -> list[float]:
        return [0.5]


def _resolved(*, visual_mode: str = "IDENTITY", epsilon: float = 1e-6):
    recipe = parse_recipe(
        {
            "target": {},
            "features": {},
            "temporal": {"chunk_size": 4, "n_action_steps": 4, "stride": 2},
            "split": {
                "strategy": "explicit_group_assignment_v1",
                "group_mapping": "g.json",
                "assignments": "a.json",
            },
            "normalization": {
                "stats_source": "fit_train",
                "modes": {"VISUAL": visual_mode, "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                "epsilon": epsilon,
            },
        }
    )
    source = VerifiedSource(
        dataset_version_id="dv-1",
        manifest_sha256="a" * 64,
        profile_id="arm6_joint_position_v1",
        profile_revision=2,
        camera_keys=("front",),
        aligned_fps=10,
        state_dim=1,
        action_dim=1,
        state_names=("joint_0",),
        action_names=("joint_0",),
        state_unit="rad",
        action_unit="rad",
        action_semantics="joint_target_position",
    )
    return resolve_recipe(recipe, source)


def _stats(source: object = _Ramp(), *, epsilon: float = 1e-6):
    split = SplitResult(
        strategy="explicit_group_assignment_v1",
        seed=None,
        ratios=None,
        by_group={"g_ep-1": "train"},
        by_episode={"ep-1": "train"},
        counts={"train": 1},
    )
    index = build_window_index(
        [EpisodeFrames("ep-1", 0, 8, {}, "d" * 64)],
        split,
        chunk_size=4,
        stride=2,
        require_nonempty=["train"],
    )
    return fit_train_stats(index.windows, source, state_dim=1, action_dim=1, epsilon=epsilon)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def test_processors_are_built_with_eps_zero():
    processors = build_act_processors(_resolved(), _stats())
    assert processors.pre.eps == PROCESSOR_EPS == 0.0
    assert processors.post.eps == PROCESSOR_EPS


def test_a_recipe_that_normalises_images_without_image_stats_is_refused():
    with pytest.raises(ApiError) as excinfo:
        build_act_processors(_resolved(visual_mode="MEAN_STD"), _stats())
    assert excinfo.value.code is ErrorCode.PROCESSOR_STATS_MISMATCH
    assert "image statistics" in excinfo.value.message


def test_a_stat_dimension_mismatch_is_refused():
    class _TwoDim(_Ramp):
        def state(self, episode_id: str, frame_index: int) -> list[float]:
            return [float(frame_index), float(frame_index)]

        def action(self, episode_id: str, frame_index: int) -> list[float]:
            return [float(frame_index), float(frame_index)]

    split = SplitResult(
        strategy="explicit_group_assignment_v1",
        seed=None,
        ratios=None,
        by_group={"g": "train"},
        by_episode={"ep-1": "train"},
        counts={"train": 1},
    )
    index = build_window_index(
        [EpisodeFrames("ep-1", 0, 8, {}, "d" * 64)],
        split,
        chunk_size=4,
        stride=2,
        require_nonempty=["train"],
    )
    two_dim_stats = fit_train_stats(
        index.windows, _TwoDim(), state_dim=2, action_dim=2, epsilon=1e-6
    )
    with pytest.raises(ApiError) as excinfo:
        build_act_processors(_resolved(), two_dim_stats)
    assert excinfo.value.code is ErrorCode.PROCESSOR_STATS_MISMATCH
    assert excinfo.value.details["stats_dim"] == 2


# ---------------------------------------------------------------------------
# normalisation semantics (TP-13)
# ---------------------------------------------------------------------------
def test_mean_maps_to_zero_and_mean_plus_std_maps_to_one():
    stats = _stats()
    processors = build_act_processors(_resolved(), stats)
    state_mean = stats.state.channels[0].mean
    state_std = stats.state.channels[0].effective_std
    action_mean = stats.action.channels[0].mean
    action_std = stats.action.channels[0].effective_std
    assert state_mean == pytest.approx(2.0)
    assert state_std == pytest.approx(math.sqrt(8 / 3))
    # 动作覆盖 0..7，因此它有自己的均值（3.5）——状态和动作不是同一组统计量。
    assert action_mean == pytest.approx(3.5)

    normalised = processors.normalise(
        {
            "observation.state": torch.tensor([[state_mean]], dtype=torch.float32),
            "action": torch.tensor([[action_mean + action_std]], dtype=torch.float32),
        }
    )
    assert normalised["observation.state"].item() == pytest.approx(0.0, abs=1e-6)
    assert normalised["action"].item() == pytest.approx(1.0, abs=1e-5)


def test_roundtrip_is_invertible_within_the_declared_tolerance():
    processors = build_act_processors(_resolved(), _stats())
    original = torch.tensor([[0.0], [1.7], [5.25]], dtype=torch.float32)
    roundtrip = processors.unnormalise(
        processors.normalise({"observation.state": original, "action": original.clone()})
    )["action"]
    assert torch.allclose(roundtrip, original, atol=1e-5, rtol=1e-4)


def test_a_constant_channel_stays_finite_and_invertible():
    """raw_std=0 时若沿用官方默认 eps=1e-8，这里会放大成 1e8 量级的噪声。"""
    stats = _stats(_Constant())
    channel = stats.action.channels[0]
    assert channel.raw_std == 0.0
    assert channel.effective_std == pytest.approx(1e-6)

    processors = build_act_processors(_resolved(), stats)
    original = torch.tensor([[0.5]], dtype=torch.float32)
    normalised = processors.normalise({"observation.state": original, "action": original.clone()})
    assert normalised["action"].item() == pytest.approx(0.0, abs=1e-6)
    roundtrip = processors.unnormalise(normalised)["action"]
    assert roundtrip.item() == pytest.approx(0.5, abs=1e-5)


def test_a_sequence_is_broadcast_over_the_time_axis():
    stats = _stats()
    processors = build_act_processors(_resolved(), stats)
    action_mean = stats.action.channels[0].mean
    action_std = stats.action.channels[0].effective_std
    sequence = torch.tensor(
        [[action_mean - action_std], [action_mean], [action_mean + action_std], [action_mean]],
        dtype=torch.float32,
    )
    normalised = processors.normalise(
        {"observation.state": sequence[:1], "action": sequence.clone()}
    )["action"]
    assert normalised.shape == sequence.shape
    # 同一组统计量对整条序列逐帧生效：均值帧为 0，均值±std 为 ±1。
    assert normalised[1].item() == pytest.approx(0.0, abs=1e-6)
    assert normalised[0].item() == pytest.approx(-1.0, abs=1e-5)
    assert normalised[2].item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# tamper detection (TP-12)
# ---------------------------------------------------------------------------
def test_verify_passes_on_a_freshly_built_processor():
    resolved = _resolved()
    stats = _stats()
    processors = build_act_processors(resolved, stats)
    verify_processor_stats(processors, stats, resolved=resolved)


def test_an_overridden_statistic_is_detected():
    resolved = _resolved()
    stats = _stats()
    processors = build_act_processors(resolved, stats)

    # 模拟训练入口把 dataset.meta.stats（全量统计）塞进 processor 的情况。
    processors.pre._tensor_stats[resolved.features.action_key]["std"] = torch.tensor(
        [99.0], dtype=torch.float32
    )
    with pytest.raises(ApiError) as excinfo:
        verify_processor_stats(processors, stats, resolved=resolved)
    assert excinfo.value.code is ErrorCode.PROCESSOR_STATS_MISMATCH
    assert excinfo.value.details["field"] == "std"


def test_a_non_zero_eps_is_detected():
    resolved = _resolved()
    stats = _stats()
    processors = build_act_processors(resolved, stats)

    processors.pre.eps = 1e-8  # 恢复官方默认值
    with pytest.raises(ApiError) as excinfo:
        verify_processor_stats(processors, stats, resolved=resolved)
    assert excinfo.value.code is ErrorCode.PROCESSOR_STATS_MISMATCH
    assert excinfo.value.details["observed_eps"] == pytest.approx(1e-8)


def test_processor_config_digest_is_stable_and_stat_sensitive():
    resolved = _resolved()
    first = build_act_processors(resolved, _stats())
    second = build_act_processors(resolved, _stats())
    assert first.config_sha256 == second.config_sha256

    changed = build_act_processors(resolved, _stats(epsilon=1e-3))
    assert changed.config_sha256 != first.config_sha256
