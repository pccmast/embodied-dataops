"""train-only 统计（EDO-TRAIN-PREP-1.1 §8.1）。

规格给了一个手算基线，这里把它变成可执行的断言：
frames 0..7、chunk=4、stride=2 → anchors 0/2/4 →
**状态 count=3、动作 count=8（不是 3×4=12）**。
"""

from __future__ import annotations

import math

import pytest

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.split import SplitResult
from embodied_dataops.training.stats import fit_train_stats
from embodied_dataops.training.windows import EpisodeFrames, build_window_index

pytestmark = pytest.mark.no_db


class _IndexSource:
    """以「frame index 就是数值」构造手算基线。"""

    def __init__(self, frames: int = 8, *, dim: int = 1) -> None:
        self.frames = frames
        self.dim = dim

    def state(self, episode_id: str, frame_index: int) -> list[float]:
        return [float(frame_index)] * self.dim

    def action(self, episode_id: str, frame_index: int) -> list[float]:
        return [float(frame_index)] * self.dim


def _train_windows(frames: int = 8, *, chunk: int = 4, stride: int = 2):
    split = SplitResult(
        strategy="explicit_group_assignment_v1",
        seed=None,
        ratios=None,
        by_group={"g_ep-1": "train"},
        by_episode={"ep-1": "train"},
        counts={"train": 1},
    )
    index = build_window_index(
        [EpisodeFrames("ep-1", 0, frames, {}, "d" * 64)],
        split,
        chunk_size=chunk,
        stride=stride,
        require_nonempty=["train"],
    )
    return index.windows


# ---------------------------------------------------------------------------
# the hand-calculated baseline from §8.1
# ---------------------------------------------------------------------------
def test_overlapping_windows_do_not_double_count_frames():
    windows = _train_windows()
    stats = fit_train_stats(
        windows, _IndexSource(), state_dim=1, action_dim=1, epsilon=1e-6
    )

    # anchors 0/2/4 -> 状态唯一帧 3 个
    assert stats.unique_state_frames == 3
    assert stats.state.channels[0].count == 3
    assert stats.state.channels[0].mean == pytest.approx(2.0)
    assert stats.state.channels[0].raw_std == pytest.approx(math.sqrt(8 / 3))

    # 动作覆盖 0..7，去重后 8 个，不是 3×4=12
    assert stats.unique_action_frames == 8
    assert stats.action.channels[0].count == 8
    assert stats.action.channels[0].mean == pytest.approx(3.5)
    assert stats.action.channels[0].raw_std == pytest.approx(math.sqrt(5.25))


def test_document_declares_train_only_scope():
    stats = fit_train_stats(_train_windows(), _IndexSource(), state_dim=1, action_dim=1, epsilon=1e-6)
    document = stats.as_document()
    assert document["stats_scope"] == "train_only"
    assert document["features"]["state"]["count"] == [3]
    assert document["features"]["action"]["count"] == [8]
    assert len(document["fitted_frame_digest"]) == 64


def test_fitted_frame_digest_is_stable_and_content_dependent():
    windows = _train_windows()
    first = fit_train_stats(windows, _IndexSource(), state_dim=1, action_dim=1, epsilon=1e-6)
    second = fit_train_stats(windows, _IndexSource(), state_dim=1, action_dim=1, epsilon=1e-6)
    assert first.fitted_frame_digest == second.fitted_frame_digest

    other = fit_train_stats(
        _train_windows(frames=10), _IndexSource(10), state_dim=1, action_dim=1, epsilon=1e-6
    )
    assert other.fitted_frame_digest != first.fitted_frame_digest


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------
def test_val_or_test_windows_are_refused_not_ignored():
    split = SplitResult(
        strategy="explicit_group_assignment_v1",
        seed=None,
        ratios=None,
        by_group={"g_ep-1": "train", "g_ep-2": "val"},
        by_episode={"ep-1": "train", "ep-2": "val"},
        counts={"train": 1, "val": 1},
    )
    index = build_window_index(
        [EpisodeFrames("ep-1", 0, 8, {}, "a" * 64), EpisodeFrames("ep-2", 1, 8, {}, "b" * 64)],
        split,
        chunk_size=4,
        stride=4,
        require_nonempty=["train", "val"],
    )
    with pytest.raises(ApiError) as excinfo:
        fit_train_stats(index.windows, _IndexSource(), state_dim=1, action_dim=1, epsilon=1e-6)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.details["unexpected_splits"] == ["val"]


def test_frames_outside_the_train_windows_cannot_influence_the_fit():
    """只有被 train 窗口覆盖的帧参与拟合：把 episode 拉长到 40 帧也不改变结果。"""
    windows = _train_windows()
    baseline = fit_train_stats(windows, _IndexSource(), state_dim=1, action_dim=1, epsilon=1e-6)

    class _LongEpisode(_IndexSource):
        def action(self, episode_id: str, frame_index: int) -> list[float]:
            return [float(frame_index) * 100.0]  # 未覆盖的帧取值很夸张

    same_windows = fit_train_stats(
        windows, _LongEpisode(), state_dim=1, action_dim=1, epsilon=1e-6
    )
    # 同一条窗口上取值变了会变；但"窗口之外"的帧根本不会被读到，
    # 所以这里断言的是唯一帧集合，而不是数值本身。
    assert same_windows.unique_action_frames == baseline.unique_action_frames == 8
    assert same_windows.fitted_frame_digest == baseline.fitted_frame_digest


# ---------------------------------------------------------------------------
# refusals and numerical edges
# ---------------------------------------------------------------------------
def test_empty_train_windows_are_refused():
    with pytest.raises(ApiError) as excinfo:
        fit_train_stats([], _IndexSource(), state_dim=1, action_dim=1, epsilon=1e-6)
    assert excinfo.value.code is ErrorCode.EMPTY_TRAIN_WINDOWS


def test_a_non_finite_value_is_refused():
    class _NaN(_IndexSource):
        def state(self, episode_id: str, frame_index: int) -> list[float]:
            return [float("nan")]

    with pytest.raises(ApiError) as excinfo:
        fit_train_stats(_train_windows(), _NaN(), state_dim=1, action_dim=1, epsilon=1e-6)
    assert excinfo.value.code is ErrorCode.NONFINITE_INPUT


def test_a_dimension_mismatch_is_refused():
    class _Short(_IndexSource):
        def action(self, episode_id: str, frame_index: int) -> list[float]:
            return [0.0]

    with pytest.raises(ApiError) as excinfo:
        fit_train_stats(_train_windows(), _Short(), state_dim=1, action_dim=6, epsilon=1e-6)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.details["expected_dim"] == 6


def test_a_constant_channel_reports_constant_and_a_finite_effective_std():
    class _Constant(_IndexSource):
        def action(self, episode_id: str, frame_index: int) -> list[float]:
            return [0.5]

    stats = fit_train_stats(
        _train_windows(), _Constant(), state_dim=1, action_dim=1, epsilon=1e-6
    )
    channel = stats.action.channels[0]
    assert channel.raw_std == 0.0
    assert channel.constant is True
    # effective_std 必须非零，否则官方 normalizer 会除以 eps 把常数维放大成噪声。
    assert channel.effective_std == pytest.approx(1e-6)


def test_a_zero_dimension_is_refused():
    with pytest.raises(ApiError) as excinfo:
        fit_train_stats(_train_windows(), _IndexSource(), state_dim=0, action_dim=1, epsilon=1e-6)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_a_non_positive_epsilon_is_refused():
    with pytest.raises(ApiError) as excinfo:
        fit_train_stats(_train_windows(), _IndexSource(), state_dim=1, action_dim=1, epsilon=0)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_multi_dimensional_features_are_fitted_per_channel():
    class _Ramp(_IndexSource):
        def state(self, episode_id: str, frame_index: int) -> list[float]:
            return [float(frame_index), float(frame_index) * 2]

        def action(self, episode_id: str, frame_index: int) -> list[float]:
            return [float(frame_index), float(frame_index) * 2]

    stats = fit_train_stats(
        _train_windows(), _Ramp(), state_dim=2, action_dim=2, epsilon=1e-6
    )
    first, second = stats.state.channels
    assert first.mean == pytest.approx(2.0)
    assert second.mean == pytest.approx(4.0)
    # 第二维是两倍，标准差也应两倍：线性变换下的分布性质。
    assert second.raw_std == pytest.approx(first.raw_std * 2)
