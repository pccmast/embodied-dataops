"""训练准备层：TrainingRecipe 合同、来源绑定与规范语义摘要。

对应 `docs/SPEC_TRAINING_PREP_ACT_V1.md`（EDO-TRAIN-PREP-1.1）：

* §5.2 配方必须是严格 Pydantic 模型，拒绝未知字段，序列化为规范 JSON 后取 SHA-256；
* §11.1 区分「文件 SHA」与「语义 SHA」，语义摘要使用 ``edo-canonical-json-v1``：
  UTF-8、无 BOM、键按 Unicode 字符序、``ensure_ascii=False``、紧凑分隔符、
  ``allow_nan=False``；有限 float64 参与摘要前转 ``float.hex()``，负零规范为正零，
  非有限值直接拒绝。

本模块**不做**来源校验的 I/O（读 PostgreSQL、算文件摘要）：它接收一个已核验的
:class:`VerifiedSource` 快照，只负责「配方 × 来源」→ ``ResolvedRecipe`` 的兼容性判定。
来源怎么被证明是 PUBLISHED 属于 CLI 层职责，两者分开才能在没有数据库的环境里测。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from embodied_dataops.errors import ApiError, ErrorCode

RECIPE_SCHEMA_VERSION = "training-recipe/1.0"
CANONICAL_JSON_VERSION = "edo-canonical-json-v1"

#: 配方里的「来源绑定指令」哨兵。它们不能原样传给 LeRobot，必须由 resolver 解析成
#: 具体值；没有对应来源证据就失败，绝不用默认值猜。
SENTINEL_CAMERAS = "verified_source_camera_keys"
SENTINEL_FPS = "verified_source_aligned_fps"
SENTINEL_ACTION = "verified_source_action_semantics"

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

SPLIT_STRATEGY_GROUPED_HASH = "grouped_hash_v1"
SPLIT_STRATEGY_EXPLICIT = "explicit_group_assignment_v1"


class _Strict(BaseModel):
    """未知字段一律拒绝：拼错的键不能静默变成默认值。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetConfig(_Strict):
    policy_type: Literal["act"] = "act"
    lerobot_version: Literal["0.4.4"] = "0.4.4"
    initialization: Literal["random", "checkpoint"] = "random"
    checkpoint: str | None = None
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None

    @model_validator(mode="after")
    def _checkpoint_consistency(self) -> TargetConfig:
        if self.initialization == "checkpoint" and not self.checkpoint:
            raise ValueError("initialization=checkpoint requires a checkpoint reference")
        if self.initialization == "random" and self.checkpoint:
            raise ValueError("initialization=random must not carry a checkpoint reference")
        return self


class FeaturesConfig(_Strict):
    #: 首期唯一允许的绑定方式：特征从已核验来源派生。
    binding: Literal["verified_source_profile"] = "verified_source_profile"
    state_key: str = "observation.state"
    action_key: str = "action"
    #: 哨兵或显式相机列表。显式列表必须是来源相机键的子集，不允许凭空增删。
    cameras: list[str] | Literal["verified_source_camera_keys"] = SENTINEL_CAMERAS
    #: 首期只支持 identity；任何缩放都必须显式配置并单独验收。
    image_transform: Literal["identity"] = "identity"
    fps: int | Literal["verified_source_aligned_fps"] = SENTINEL_FPS
    #: 哨兵或显式的动作语义声明；显式值必须与来源一致。
    action_representation: str | None = SENTINEL_ACTION


class TemporalConfig(_Strict):
    n_obs_steps: Literal[1] = 1
    chunk_size: int = Field(gt=0)
    n_action_steps: int = Field(gt=0)
    stride: int = Field(gt=0)
    tail_policy: Literal["drop"] = "drop"

    @model_validator(mode="after")
    def _check_actions(self) -> TemporalConfig:
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size")
        return self


class GroupedHashSplit(_Strict):
    strategy: Literal["grouped_hash_v1"] = SPLIT_STRATEGY_GROUPED_HASH
    seed: int
    ratios: dict[str, float]
    group_mapping: str
    assignments: None = None
    require_nonempty: list[str] = Field(default_factory=lambda: list(SPLIT_NAMES))

    @model_validator(mode="after")
    def _check_ratios(self) -> GroupedHashSplit:
        unknown = set(self.ratios) - set(SPLIT_NAMES)
        if unknown:
            raise ValueError(f"unknown split keys in ratios: {sorted(unknown)}")
        total = math.fsum(self.ratios.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total!r}")
        if any(value < 0 for value in self.ratios.values()):
            raise ValueError("split ratios must be non-negative")
        unknown_required = set(self.require_nonempty) - set(SPLIT_NAMES)
        if unknown_required:
            raise ValueError(f"unknown split keys in require_nonempty: {sorted(unknown_required)}")
        return self


class ExplicitAssignmentSplit(_Strict):
    #: 仅用于 synthetic-fixture 的正常路径：三组显式分配，不靠哈希偶然命中。
    strategy: Literal["explicit_group_assignment_v1"] = SPLIT_STRATEGY_EXPLICIT
    seed: None = None
    ratios: None = None
    group_mapping: str
    assignments: str
    require_nonempty: list[str] = Field(default_factory=lambda: list(SPLIT_NAMES))

    @model_validator(mode="after")
    def _check_required(self) -> ExplicitAssignmentSplit:
        unknown_required = set(self.require_nonempty) - set(SPLIT_NAMES)
        if unknown_required:
            raise ValueError(f"unknown split keys in require_nonempty: {sorted(unknown_required)}")
        return self


SplitConfig = Annotated[
    GroupedHashSplit | ExplicitAssignmentSplit, Field(discriminator="strategy")
]


class NormalizationConfig(_Strict):
    stats_source: Literal["fit_train", "checkpoint"] = "fit_train"
    modes: dict[str, str] = Field(
        default_factory=lambda: {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    )
    epsilon: float = Field(default=1e-6, gt=0)

    @model_validator(mode="after")
    def _check_modes(self) -> NormalizationConfig:
        allowed_keys = {"VISUAL", "STATE", "ACTION"}
        unknown = set(self.modes) - allowed_keys
        if unknown:
            raise ValueError(f"unknown normalization feature types: {sorted(unknown)}")
        allowed_modes = {"MEAN_STD", "MIN_MAX", "IDENTITY", "QUANTILES", "QUANTILE10"}
        bad = {key: value for key, value in self.modes.items() if value not in allowed_modes}
        if bad:
            raise ValueError(f"unsupported normalization modes: {bad}")
        return self


class QualityConfig(_Strict):
    #: 首期只允许整窗拒绝：不用 padding 或零填充掩盖坏样本。
    invalid_policy: Literal["reject_window"] = "reject_window"


class ValidationConfig(_Strict):
    device: str = "cpu"
    batch_size: int = Field(default=1, gt=0)
    batches_per_split: int = Field(default=1, gt=0)
    optimizer_steps: int = Field(default=1, gt=0)
    seed: int = 20261031


class TrainingRecipe(_Strict):
    schema_version: Literal["training-recipe/1.0"] = RECIPE_SCHEMA_VERSION
    target: TargetConfig
    features: FeaturesConfig
    temporal: TemporalConfig
    split: SplitConfig
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


# ---------------------------------------------------------------------------
# 规范序列化与语义摘要（§11.1）
# ---------------------------------------------------------------------------
def _canonical_value(value: Any) -> Any:
    """把任意值转成规范序列化的中间表示。

    有限 float 转 ``float.hex()``：``0.8`` → ``'0x1.999999999999ap-1'``。这样跨版本、
    跨平台的 JSON 往返不会因为最短往返表示或精度截断改变摘要；负零按正零处理，
    非有限值直接失败（NaN/Inf 没有确定性文本表示，进摘要就意味着不可复现）。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ApiError(
                ErrorCode.NONFINITE_INPUT,
                "non-finite float cannot participate in a deterministic digest",
                details={"value": repr(value)},
            )
        if value == 0.0:
            value = 0.0
        return float(value).hex()
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "canonical JSON requires string keys",
                details={"keys": [type(key).__name__ for key in value]},
            )
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise ApiError(
        ErrorCode.INVALID_INPUT,
        f"value of type {type(value).__name__} has no canonical encoding",
        details={"canonical_json_version": CANONICAL_JSON_VERSION},
    )


def canonical_json_bytes(value: Any) -> bytes:
    """规范 JSON 字节流：紧分隔符、键排序、UTF-8 无 BOM、尾部单个 LF。"""
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (payload + "\n").encode("utf-8")


def semantic_sha256(value: Any) -> str:
    """语义摘要：只随内容变化，不随键顺序、整数/浮点文本表示变化。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# ---------------------------------------------------------------------------
# 来源快照与解析
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerifiedSource:
    """一个已核验来源的最小快照（由 CLI 层证明为 PUBLISHED 或 FIXTURE_ONLY）。"""

    dataset_version_id: str
    manifest_sha256: str
    profile_id: str
    profile_revision: int
    camera_keys: tuple[str, ...]
    aligned_fps: int
    state_key: str = "observation.state"
    action_key: str = "action"
    state_dim: int = 0
    action_dim: int = 0
    state_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()
    state_unit: str = ""
    action_unit: str = ""
    action_semantics: str = ""
    #: 相机短名 → ``(C, H, W)``。有值时会写进 resolved recipe；缺值时由
    #: artifact 层用官方 ``meta.shapes`` 补齐，绝不在样本层猜尺寸。
    image_shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    episode_ids: tuple[str, ...] = ()
    data_origin: str = ""
    source_verification: str = "FIXTURE_ONLY"
    extra: dict[str, Any] = field(default_factory=dict)


class ResolvedFeatures(_Strict):
    """所有哨兵都已解析成具体值：这份对象可以直接喂给 adapter。"""

    state_key: str
    action_key: str
    cameras: list[str]
    image_transform: Literal["identity"] = "identity"
    fps: int
    action_representation: str
    state_dim: int
    action_dim: int
    state_names: list[str]
    action_names: list[str]
    state_unit: str
    action_unit: str
    #: 相机短名 → ``(C, H, W)``；resolved 之后必须是具体尺寸，不能留哨兵。
    image_shapes: dict[str, list[int]] = Field(default_factory=dict)


class ResolvedSourceBinding(_Strict):
    dataset_version_id: str
    manifest_sha256: str
    profile_id: str
    profile_revision: int
    data_origin: str
    source_verification: Literal["FIXTURE_ONLY", "PUBLISHED_VERIFIED"]


class ResolvedRecipe(_Strict):
    schema_version: Literal["training-recipe/1.0"] = RECIPE_SCHEMA_VERSION
    target: TargetConfig
    features: ResolvedFeatures
    temporal: TemporalConfig
    split: SplitConfig
    normalization: NormalizationConfig
    quality: QualityConfig
    validation: ValidationConfig
    source: ResolvedSourceBinding
    #: 原始配方（含哨兵）的语义摘要，用于证明「这份 resolved 来自哪份 recipe」。
    recipe_sha256: str


def parse_recipe(payload: Any) -> TrainingRecipe:
    """严格解析；Pydantic 的报错统一转成 ``INVALID_INPUT``。"""
    if not isinstance(payload, dict):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "a training recipe must be a JSON object",
            details={"type": type(payload).__name__},
        )
    try:
        return TrainingRecipe.model_validate(payload)
    except ValidationError as exc:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "training recipe does not satisfy training-recipe/1.0",
            details={"errors": exc.errors(include_url=False)[:10]},
        ) from exc


def recipe_sha256(recipe: TrainingRecipe) -> str:
    """原始配方的语义摘要（含哨兵，不含解析结果）。"""
    return semantic_sha256(recipe.model_dump(mode="json"))


def _incompatible(message: str, **details: Any) -> ApiError:
    return ApiError(ErrorCode.FEATURE_INCOMPATIBLE, message, details=details)


def resolve_recipe(recipe: TrainingRecipe, source: VerifiedSource) -> ResolvedRecipe:
    """把哨兵解析成来源的具体值，并做兼容性判定（§5.2）。

    解析只允许「来源有什么就用什么」或「配方显式写死且与来源一致」。任何一处对不上
    都失败，绝不用默认值补齐——否则 bundle 会宣称一个它没有验证过的特征合同。
    """
    features = recipe.features

    if features.state_key != source.state_key or features.action_key != source.action_key:
        raise _incompatible(
            "recipe state/action keys do not match the verified source features",
            recipe={"state_key": features.state_key, "action_key": features.action_key},
            source={"state_key": source.state_key, "action_key": source.action_key},
        )

    if source.state_dim <= 0 or source.action_dim <= 0:
        raise _incompatible(
            "the verified source has no robot state/action; an Ego video cannot be an ACT demonstration",
            source={"state_dim": source.state_dim, "action_dim": source.action_dim},
        )
    if len(source.state_names) != source.state_dim or len(source.action_names) != source.action_dim:
        raise _incompatible(
            "verified source channel names do not match its dimensions",
            source={
                "state_dim": source.state_dim,
                "state_names": len(source.state_names),
                "action_dim": source.action_dim,
                "action_names": len(source.action_names),
            },
        )

    if features.cameras == SENTINEL_CAMERAS:
        cameras = list(source.camera_keys)
    else:
        cameras = list(features.cameras)
        missing = [name for name in cameras if name not in source.camera_keys]
        if missing:
            raise _incompatible(
                "recipe requests cameras that the verified source does not have",
                missing=missing,
                source_cameras=list(source.camera_keys),
            )
    if not cameras:
        raise _incompatible("at least one camera is required")

    if features.fps == SENTINEL_FPS:
        fps = int(source.aligned_fps)
    else:
        fps = int(features.fps)
        if fps != int(source.aligned_fps):
            raise _incompatible(
                "recipe fps contradicts the verified source aligned fps",
                recipe=fps,
                source=int(source.aligned_fps),
            )
    if fps <= 0:
        raise _incompatible("aligned fps must be positive", fps=fps)

    if features.action_representation == SENTINEL_ACTION:
        action_representation = source.action_semantics
    else:
        action_representation = str(features.action_representation or "")
        if action_representation != source.action_semantics:
            raise _incompatible(
                "recipe action representation contradicts the verified source semantics",
                recipe=action_representation,
                source=source.action_semantics,
            )
    if not action_representation:
        raise _incompatible("the verified source does not document its action semantics")

    if not source.manifest_sha256 or len(source.manifest_sha256) != 64:
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "the verified source does not carry a frozen manifest digest",
            details={"dataset_version_id": source.dataset_version_id},
        )

    # 图像尺寸只允许来自来源证据：没有就留空，由 artifact 层用官方 meta 补齐后
    # 再进 preparation_id，绝不在样本层猜一个默认分辨率。
    image_shapes: dict[str, list[int]] = {}
    for camera in cameras:
        shape = source.image_shapes.get(camera)
        if shape is None:
            continue
        channels, height, width = (int(size) for size in shape)
        if channels != 3 or height <= 0 or width <= 0:
            raise _incompatible(
                "the verified source declares an unusable image shape",
                camera=camera,
                shape=[channels, height, width],
            )
        image_shapes[camera] = [channels, height, width]

    resolved_features = ResolvedFeatures(
        state_key=source.state_key,
        action_key=source.action_key,
        cameras=cameras,
        image_transform=features.image_transform,
        fps=fps,
        action_representation=action_representation,
        state_dim=int(source.state_dim),
        action_dim=int(source.action_dim),
        state_names=list(source.state_names),
        action_names=list(source.action_names),
        state_unit=source.state_unit,
        action_unit=source.action_unit,
        image_shapes=image_shapes,
    )

    return ResolvedRecipe(
        target=recipe.target,
        features=resolved_features,
        temporal=recipe.temporal,
        split=recipe.split,
        normalization=recipe.normalization,
        quality=recipe.quality,
        validation=recipe.validation,
        source=ResolvedSourceBinding(
            dataset_version_id=source.dataset_version_id,
            manifest_sha256=source.manifest_sha256,
            profile_id=source.profile_id,
            profile_revision=int(source.profile_revision),
            data_origin=source.data_origin,
            source_verification=source.source_verification,
        ),
        recipe_sha256=recipe_sha256(recipe),
    )


def resolved_recipe_sha256(resolved: ResolvedRecipe) -> str:
    return semantic_sha256(resolved.model_dump(mode="json"))


__all__ = [
    "CANONICAL_JSON_VERSION",
    "RECIPE_SCHEMA_VERSION",
    "SENTINEL_ACTION",
    "SENTINEL_CAMERAS",
    "SENTINEL_FPS",
    "SPLIT_NAMES",
    "SPLIT_STRATEGY_EXPLICIT",
    "SPLIT_STRATEGY_GROUPED_HASH",
    "ExplicitAssignmentSplit",
    "FeaturesConfig",
    "GroupedHashSplit",
    "NormalizationConfig",
    "QualityConfig",
    "ResolvedFeatures",
    "ResolvedRecipe",
    "ResolvedSourceBinding",
    "SplitConfig",
    "TargetConfig",
    "TemporalConfig",
    "TrainingRecipe",
    "ValidationConfig",
    "VerifiedSource",
    "canonical_json_bytes",
    "parse_recipe",
    "recipe_sha256",
    "resolve_recipe",
    "resolved_recipe_sha256",
    "semantic_sha256",
]
