"""train-only 流式统计（EDO-TRAIN-PREP-1.1 §8.1）。

三条规则决定了这个模块的形状：

1. **只用训练集**：调用方必须传 train 窗口；非 train 窗口一出现就失败，而不是被忽略
   ——静默忽略等于把泄漏藏进实现细节。
2. **按唯一原始帧去重**：重叠窗口和 stride 会让同一帧被多个窗口覆盖，但它在统计里
   只能算一次。状态按输入帧去重，动作按监督帧去重，主键是 ``(episode, frame_index)``。
3. **float64 流式累加**：Welford 更新均值和方差，总体标准差 ``ddof=0``；
   同时保存 ``raw_std`` 与 ``effective_std = max(raw_std, epsilon)``，
   后者才是喂给官方 processor 的值（默认 ``std + eps`` 会把常数维放大成噪声）。

本模块不读文件、不 import torch：它接收 :class:`FrameSource` 提供的数值，便于在没有
数据、没有 GPU 的环境里用手算样例把语义钉死。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.contract import semantic_sha256
from embodied_dataops.training.windows import Window

__all__ = [
    "ChannelStats",
    "FeatureStats",
    "FrameSource",
    "TrainStats",
    "fit_train_stats",
    "stats_from_document",
]

STATS_SCOPE = "train_only"
STATS_ALGORITHM_VERSION = "welford-f64-ddof0-v1"


class FrameSource(Protocol):
    """按 ``(episode_id, frame_index)`` 取原始数值的只读接口。"""

    def state(self, episode_id: str, frame_index: int) -> Sequence[float]: ...

    def action(self, episode_id: str, frame_index: int) -> Sequence[float]: ...


@dataclass
class _Welford:
    """单维流式累加器：count / mean / M2。"""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            raise ApiError(
                ErrorCode.NONFINITE_INPUT,
                "a fitted frame carries a non-finite value",
                details={"value": repr(value)},
            )
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class ChannelStats:
    count: int
    mean: float
    raw_std: float
    effective_std: float
    min: float
    max: float
    constant: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "raw_std": self.raw_std,
            "effective_std": self.effective_std,
            "min": self.min,
            "max": self.max,
            "constant": self.constant,
        }


@dataclass(frozen=True)
class FeatureStats:
    dim: int
    channels: tuple[ChannelStats, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "count": [channel.count for channel in self.channels],
            "mean": [channel.mean for channel in self.channels],
            "raw_std": [channel.raw_std for channel in self.channels],
            "effective_std": [channel.effective_std for channel in self.channels],
            "min": [channel.min for channel in self.channels],
            "max": [channel.max for channel in self.channels],
            "constant_mask": [channel.constant for channel in self.channels],
        }


@dataclass(frozen=True)
class TrainStats:
    state: FeatureStats
    action: FeatureStats
    epsilon: float
    unique_state_frames: int
    unique_action_frames: int
    fitted_frame_digest: str
    algorithm: str = STATS_ALGORITHM_VERSION
    stats_scope: str = STATS_SCOPE
    extra: dict[str, Any] = field(default_factory=dict)

    def as_document(self) -> dict[str, Any]:
        return {
            "schema_version": "training-stats/1.0",
            "stats_scope": self.stats_scope,
            "algorithm": self.algorithm,
            "epsilon": self.epsilon,
            "unique_state_frames": self.unique_state_frames,
            "unique_action_frames": self.unique_action_frames,
            "fitted_frame_digest": self.fitted_frame_digest,
            "features": {"state": self.state.as_dict(), "action": self.action.as_dict()},
        }


def stats_from_document(payload: Mapping[str, Any]) -> TrainStats:
    """从 ``train_stats.json`` 精确重建 :class:`TrainStats`。

    重载必须是**字节级身份**的还原而不是"读个近似值"：字段缺失、维度不符、
    ``stats_scope`` 不是 train_only 都直接失败，避免验证阶段悄悄用一个
    与冻结产物不同的统计量。
    """
    if payload.get("schema_version") != "training-stats/1.0":
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "train_stats.json has an unsupported schema_version",
            details={"schema_version": payload.get("schema_version")},
        )
    if payload.get("stats_scope") != STATS_SCOPE:
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "train_stats.json is not scoped to the training split",
            details={"stats_scope": payload.get("stats_scope"), "expected": STATS_SCOPE},
        )
    features = payload.get("features") or {}
    state = _feature_stats_from_document(features.get("state"), label="state")
    action = _feature_stats_from_document(features.get("action"), label="action")
    try:
        return TrainStats(
            state=state,
            action=action,
            epsilon=float(payload["epsilon"]),
            unique_state_frames=int(payload["unique_state_frames"]),
            unique_action_frames=int(payload["unique_action_frames"]),
            fitted_frame_digest=str(payload["fitted_frame_digest"]),
            algorithm=str(payload.get("algorithm") or STATS_ALGORITHM_VERSION),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "train_stats.json is missing required identity fields",
            details={"error": type(exc).__name__},
        ) from exc


def _feature_stats_from_document(payload: Any, *, label: str) -> FeatureStats:
    if not isinstance(payload, Mapping):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "train_stats.json is missing a feature block",
            details={"feature": label},
        )
    fields = ("count", "mean", "raw_std", "effective_std", "min", "max", "constant_mask")
    missing = [name for name in fields if name not in payload]
    if missing:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "train_stats.json feature block is incomplete",
            details={"feature": label, "missing": missing},
        )
    dim = int(payload.get("dim") or 0)
    lengths = {name: len(payload[name]) for name in fields}
    if dim <= 0 or set(lengths.values()) != {dim}:
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "train_stats.json feature block has inconsistent dimensions",
            details={"feature": label, "dim": dim, "lengths": lengths},
        )
    channels = tuple(
        ChannelStats(
            count=int(payload["count"][index]),
            mean=float(payload["mean"][index]),
            raw_std=float(payload["raw_std"][index]),
            effective_std=float(payload["effective_std"][index]),
            min=float(payload["min"][index]),
            max=float(payload["max"][index]),
            constant=bool(payload["constant_mask"][index]),
        )
        for index in range(dim)
    )
    return FeatureStats(dim=dim, channels=channels)


def _accumulators(dim: int) -> list[_Welford]:
    if dim <= 0:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "a feature needs a positive dimension to fit statistics",
            details={"dim": dim},
        )
    return [_Welford() for _ in range(dim)]


def _update(accumulators: list[_Welford], values: Sequence[float], *, label: str, frame: tuple[str, int]) -> None:
    if len(values) != len(accumulators):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "a fitted frame does not match the feature dimension",
            details={
                "feature": label,
                "expected_dim": len(accumulators),
                "observed_dim": len(values),
                "frame": {"episode_id": frame[0], "frame_index": frame[1]},
            },
        )
    for accumulator, value in zip(accumulators, values, strict=True):
        accumulator.update(float(value))


def _finalize(accumulators: list[_Welford], epsilon: float, *, label: str) -> FeatureStats:
    channels: list[ChannelStats] = []
    for index, accumulator in enumerate(accumulators):
        if accumulator.count == 0:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "a required feature dimension has no valid training sample",
                details={"feature": label, "dimension": index},
            )
        raw_std = accumulator.std
        effective_std = max(raw_std, epsilon)
        channels.append(
            ChannelStats(
                count=accumulator.count,
                mean=accumulator.mean,
                raw_std=raw_std,
                effective_std=effective_std,
                min=accumulator.minimum,
                max=accumulator.maximum,
                constant=raw_std == 0.0,
            )
        )
    return FeatureStats(dim=len(channels), channels=tuple(channels))


def fit_train_stats(
    windows: Sequence[Window],
    source: FrameSource,
    *,
    state_dim: int,
    action_dim: int,
    epsilon: float,
) -> TrainStats:
    """只在 train 窗口覆盖的唯一原始帧上拟合统计量。

    ``state`` 用每个窗口的 anchor（输入观察帧），``action`` 用窗口覆盖的全部监督帧；
    两者分别去重，因此 stride 重叠不会让同一帧被计多次。
    """
    if epsilon <= 0:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "normalization epsilon must be positive",
            details={"epsilon": epsilon},
        )
    train_windows = [window for window in windows if window.split == "train"]
    foreign = sorted({window.split for window in windows} - {"train"})
    if foreign:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "statistics may only be fitted on train windows",
            details={"unexpected_splits": foreign},
        )
    if not train_windows:
        raise ApiError(
            ErrorCode.EMPTY_TRAIN_WINDOWS,
            "no train window was provided to fit statistics",
            details={"window_count": 0},
        )

    state_frames = sorted(
        {(window.source_episode_id, window.anchor_frame_index) for window in train_windows}
    )
    action_frames = sorted(
        {
            (window.source_episode_id, frame_index)
            for window in train_windows
            for frame_index in window.action_frame_indices
        }
    )

    state_accumulators = _accumulators(state_dim)
    action_accumulators = _accumulators(action_dim)

    for episode_id, frame_index in state_frames:
        _update(
            state_accumulators,
            source.state(episode_id, frame_index),
            label="state",
            frame=(episode_id, frame_index),
        )
    for episode_id, frame_index in action_frames:
        _update(
            action_accumulators,
            source.action(episode_id, frame_index),
            label="action",
            frame=(episode_id, frame_index),
        )

    fitted_digest = semantic_sha256(
        {
            "algorithm": STATS_ALGORITHM_VERSION,
            "state_frames": [[episode_id, frame] for episode_id, frame in state_frames],
            "action_frames": [[episode_id, frame] for episode_id, frame in action_frames],
        }
    )

    return TrainStats(
        state=_finalize(state_accumulators, epsilon, label="state"),
        action=_finalize(action_accumulators, epsilon, label="action"),
        epsilon=epsilon,
        unique_state_frames=len(state_frames),
        unique_action_frames=len(action_frames),
        fitted_frame_digest=fitted_digest,
    )
