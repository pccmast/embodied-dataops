"""官方 ACT pre/post processor 的冻结构造与一致性校验（EDO-TRAIN-PREP-1.1 §8.2）。

这一层只做三件事，且顺序不能变：

1. **用 ``effective_std`` 构造官方 processor，并把它们的内部 ``eps`` 设为 0**。
   官方 MEAN_STD 是 ``(x - mean) / (std + eps)``；若沿用默认 ``eps=1e-8``，
   一个常数通道（``raw_std=0``）会被放大成 ``x / 1e-8`` 的噪声，而且不会报错。
   把 ``std`` 换成 ``max(raw_std, epsilon)``、再把 `eps` 置 0，是唯一能同时
   保住「常数维输出为 0」和「往返可逆」的写法。
2. **反归一化只回动作**：``x * std + mean`` 把动作还原到合同里的物理空间，
   状态与图像不需要还原。
3. **核对 processor 里真正生效的 stats 与冻结值逐字节相同**。LeRobot 0.4.4 的训练
   入口会把 ``dataset.meta.stats`` 覆盖到已加载的 processor 上，所以"我构造时传对了"
   不等于"运行时还是对的"——必须在入口处断言，否则就是静默双重归一化。

本模块不 import torch 也能完成构造（由 lerobot 内部加载），但校验函数会用到张量比较。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.contract import ResolvedRecipe, semantic_sha256
from embodied_dataops.training.stats import FeatureStats, TrainStats

__all__ = [
    "ActProcessors",
    "build_act_processors",
    "processor_config_document",
    "processor_stats_fingerprint",
    "verify_processor_stats",
]

PROCESSOR_EPS = 0.0
PROCESSOR_SCHEMA_VERSION = "training-processors/1.0"


@dataclass(frozen=True)
class ActProcessors:
    """一对官方 processor 加上它们的冻结配置摘要。"""

    pre: Any
    post: Any
    stats_payload: dict[str, dict[str, list[float]]]
    config_sha256: str
    action_key: str

    #: 官方 processor 吃的是 ``EnvTransition``（observation 嵌套 + action 平铺），
    #: 而 bundle 的样本层用扁平的 ``{policy_key: tensor}``。这里做一次显式换算，
    #: 让调用方不必知道 LeRobot 的内部结构，又不会绕过官方实现。
    def _to_transition(self, flat: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        from lerobot.processor.core import TransitionKey  # noqa: PLC0415

        observation = {key: value for key, value in flat.items() if key.startswith("observation.")}
        passthrough = {
            key: value
            for key, value in flat.items()
            if not key.startswith("observation.") and key != self.action_key
        }
        transition: dict[str, Any] = {}
        if observation:
            transition[TransitionKey.OBSERVATION] = observation
        if self.action_key in flat:
            transition[TransitionKey.ACTION] = flat[self.action_key]
        return transition, passthrough

    def _from_transition(self, transition: dict[str, Any], passthrough: dict[str, Any]) -> dict[str, Any]:
        from lerobot.processor.core import TransitionKey  # noqa: PLC0415

        flat = dict(passthrough)
        flat.update(transition.get(TransitionKey.OBSERVATION) or {})
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            flat[self.action_key] = action
        return flat

    def normalise(self, flat: dict[str, Any]) -> dict[str, Any]:
        transition, passthrough = self._to_transition(flat)
        return self._from_transition(self.pre(transition), passthrough)

    def unnormalise(self, flat: dict[str, Any]) -> dict[str, Any]:
        transition, passthrough = self._to_transition(flat)
        return self._from_transition(self.post(transition), passthrough)


def _payload_from(stats: FeatureStats) -> dict[str, list[float]]:
    """把冻结统计转成官方 processor 认的 per-feature 结构。

    只保留 ``mean/std/min/max/count``：官方 normalize 只用 mean/std，
    其余字段进 payload 是为了让「这份 processor 用的是哪次拟合」可被审计。
    """
    channels = stats.channels
    return {
        "mean": [channel.mean for channel in channels],
        "std": [channel.effective_std for channel in channels],
        "min": [channel.min for channel in channels],
        "max": [channel.max for channel in channels],
        "count": [float(channel.count) for channel in channels],
    }


def _check_finite(payload: dict[str, dict[str, list[float]]]) -> None:
    for feature, fields in payload.items():
        for name, values in fields.items():
            for index, value in enumerate(values):
                if not math.isfinite(float(value)):
                    raise ApiError(
                        ErrorCode.NONFINITE_INPUT,
                        "a processor statistic is not finite",
                        details={"feature": feature, "field": name, "index": index},
                    )


def build_act_processors(
    resolved: ResolvedRecipe,
    stats: TrainStats,
    *,
    device: str = "cpu",
) -> ActProcessors:
    """按冻结统计构造官方 pre/post processor（内部 eps=0）。"""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )

    features = resolved.features
    if stats.state.dim != features.state_dim:
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "fitted state statistics do not match the resolved feature dimension",
            details={"stats_dim": stats.state.dim, "feature_dim": features.state_dim},
        )
    if stats.action.dim != features.action_dim:
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "fitted action statistics do not match the resolved feature dimension",
            details={"stats_dim": stats.action.dim, "feature_dim": features.action_dim},
        )

    payload: dict[str, dict[str, list[float]]] = {
        features.state_key: _payload_from(stats.state),
        features.action_key: _payload_from(stats.action),
    }
    _check_finite(payload)

    input_features: dict[str, PolicyFeature] = {}
    output_features: dict[str, PolicyFeature] = {}
    for key in (features.state_key,):
        input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=(features.state_dim,))
    for camera in features.cameras:
        image_key = f"observation.images.{camera}"
        # 首期 image_transform=identity，因此没有图像统计；只声明特征形状。
        input_features[image_key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    output_features[features.action_key] = PolicyFeature(
        type=FeatureType.ACTION, shape=(features.action_dim,)
    )

    norm_map = {
        FeatureType.STATE: NormalizationMode(resolved.normalization.modes.get("STATE", "MEAN_STD")),
        FeatureType.ACTION: NormalizationMode(resolved.normalization.modes.get("ACTION", "MEAN_STD")),
        FeatureType.VISUAL: NormalizationMode(resolved.normalization.modes.get("VISUAL", "MEAN_STD")),
    }

    # 配方要求对图像做 MEAN_STD，却没有图像统计 —— 不能静默降级成 IDENTITY，
    # 那会让 bundle 宣称一个它没有执行过的变换。首期图像统计尚未实现，明确失败。
    if norm_map[FeatureType.VISUAL] is not NormalizationMode.IDENTITY and not any(
        key.startswith("observation.images.") for key in payload
    ):
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "the recipe normalises images but this bundle carries no image statistics",
            details={
                "visual_mode": resolved.normalization.modes.get("VISUAL"),
                "cameras": list(features.cameras),
                "remedy": "fit visual statistics over train pixels, or set VISUAL: IDENTITY explicitly",
            },
        )

    pre = NormalizerProcessorStep(
        features={**input_features, **output_features},
        norm_map=norm_map,
        stats=payload,
        device=device,
        eps=PROCESSOR_EPS,
    )
    post = UnnormalizerProcessorStep(
        features=output_features,
        norm_map=norm_map,
        stats=payload,
        device=device,
        eps=PROCESSOR_EPS,
    )

    document = processor_config_document(resolved, stats)
    return ActProcessors(
        pre=pre,
        post=post,
        stats_payload=payload,
        config_sha256=semantic_sha256(document),
        action_key=features.action_key,
    )


def processor_config_document(resolved: ResolvedRecipe, stats: TrainStats) -> dict[str, Any]:
    """进 ``preparation_id`` 的语义内容：策略、模式、eps、每维统计与来源。"""
    features = resolved.features
    return {
        "schema_version": PROCESSOR_SCHEMA_VERSION,
        "policy_type": resolved.target.policy_type,
        "normalization": {
            "stats_source": resolved.normalization.stats_source,
            "modes": dict(sorted(resolved.normalization.modes.items())),
            "epsilon": resolved.normalization.epsilon,
            "processor_eps": PROCESSOR_EPS,
        },
        "features": {
            "state_key": features.state_key,
            "action_key": features.action_key,
            "cameras": list(features.cameras),
            "state_dim": features.state_dim,
            "action_dim": features.action_dim,
            "image_transform": features.image_transform,
        },
        "stats": stats.as_document(),
    }


def processor_stats_fingerprint(processor: Any) -> str:
    """读取 processor 内部**真正生效**的统计量并取摘要。"""
    tensor_stats = getattr(processor, "_tensor_stats", None)
    if not tensor_stats:
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "the processor carries no effective statistics",
        )
    payload: dict[str, Any] = {}
    for feature, fields in sorted(tensor_stats.items()):
        payload[feature] = {
            name: [float(value) for value in _flatten(values)]
            for name, values in sorted(fields.items())
        }
    return semantic_sha256(payload)


def _flatten(values: Any) -> Sequence[float]:
    if hasattr(values, "reshape"):
        return values.detach().reshape(-1).tolist()  # type: ignore[no-any-return]
    if isinstance(values, (list, tuple)):
        return [float(item) for item in values]
    return [float(values)]


def _as_processor_float(value: float) -> float:
    """把冻结的 float64 值折成 processor 实际存储的精度。

    官方 processor 把 stats 存成 ``torch.float32``，所以 `sqrt(8/3)` 的 float64
    末位必然与张量里的值不同。冻结值的**身份**由摘要保证（精确相等），这里比较的是
    「processor 现在用的这组数」是否就是那组数——在它自己的精度上比较才是唯一有意义的。
    """
    import torch  # noqa: PLC0415 - 只在需要比较时导入

    return float(torch.tensor(float(value), dtype=torch.float32).item())


def verify_processor_stats(
    processors: ActProcessors, stats: TrainStats, *, resolved: ResolvedRecipe
) -> None:
    """断言 processor 里生效的 std/mean 与冻结的 effective 统计完全相同。

    这是 TP-12 的执行体：伪造覆盖值（例如训练入口塞进全量 ``dataset.meta.stats``）
    必须在这里失败，而不是等到训练出问题才发现。
    """
    expected = processors.stats_payload
    for processor, keys in ((processors.pre, expected.keys()), (processors.post, (resolved.features.action_key,))):
        tensor_stats = getattr(processor, "_tensor_stats", {})
        eps = float(getattr(processor, "eps", -1.0))
        if eps != PROCESSOR_EPS:
            raise ApiError(
                ErrorCode.PROCESSOR_STATS_MISMATCH,
                "the processor does not use eps=0; a constant channel would be divided by eps",
                details={"observed_eps": eps, "expected_eps": PROCESSOR_EPS},
            )
        for key in keys:
            observed = tensor_stats.get(key)
            if observed is None:
                raise ApiError(
                    ErrorCode.PROCESSOR_STATS_MISMATCH,
                    "the processor has no statistics for a required feature",
                    details={"feature": key},
                )
            for field in ("mean", "std"):
                observed_values = [float(value) for value in _flatten(observed[field])]
                expected_values = [float(value) for value in expected[key][field]]
                if observed_values != [_as_processor_float(value) for value in expected_values]:
                    raise ApiError(
                        ErrorCode.PROCESSOR_STATS_MISMATCH,
                        "the processor statistics were overridden after construction",
                        details={
                            "feature": key,
                            "field": field,
                            "observed": observed_values[:8],
                            "expected": expected_values[:8],
                        },
                    )
    if stats.action.dim != resolved.features.action_dim:  # pragma: no cover - guarded earlier
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "action statistics do not match the resolved action dimension",
        )
