"""受限 ACT 训练样本适配器（EDO-TRAIN-PREP-1.1 §9）。

这一层把「bundle 里冻结的窗口索引」接到「官方 LeRobot Reader」上，只解决三件事：

1. **范围受限**：``__len__`` 是**本 split 的冻结窗口数**，不是源数据集的全量帧数。
   官方 Reader 的原始长度不得成为 sampler 的边界——否则训练会悄悄跨摊。
2. **身份可证**：每次取样本都回读官方返回的 ``index / episode_index / frame_index``，
   与窗口索引逐项比对；对不上就失败而不是"取到哪算哪"。
3. **动作是未来序列**：监督标签来自窗口覆盖的 chunk 帧（官方 ``delta_timestamps``
   机制），观察只来自 anchor 帧；首期 ``tail_policy=drop``，因此 ``action_is_pad``
   必须全为 False——padding 不能用来掩盖缺动作。

模块不构造 bundle、不做统计、不跑模型：它的输入全部来自已经冻结的产物。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.contract import ResolvedRecipe
from embodied_dataops.training.windows import Window

__all__ = [
    "IMAGE_KEY_PREFIX",
    "PreparedACTDataset",
    "ReaderIndex",
    "build_reader_index",
    "official_image_key",
    "open_prepared_dataset",
]

#: 官方 LeRobot 把相机放在这个名字空间下；recipe 里的相机名是短名。
IMAGE_KEY_PREFIX = "observation.images."

#: 图像被显式做过的一次变换：解码为 RGB、HWC→CHW、uint8→float[0,1]。
IMAGE_TRANSFORM_STEPS: tuple[str, ...] = ("decode_rgb", "hwc_to_chw", "uint8_to_float01")


@dataclass(frozen=True)
class ReaderIndex:
    """官方 Reader 的两张索引表，均由官方 ``hf_dataset`` 的列读出。"""

    absolute_to_relative: dict[int, int]
    #: ``(official episode_index, frame_index) -> 源绝对 index``
    position_to_absolute: dict[tuple[int, int], int]

    def absolute_for(self, episode_index: int, frame_index: int) -> int:
        try:
            return self.position_to_absolute[(int(episode_index), int(frame_index))]
        except KeyError as exc:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "a frozen window points at a frame the official reader does not expose",
                details={
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_index),
                },
            ) from exc


def official_image_key(camera: str) -> str:
    return camera if camera.startswith(IMAGE_KEY_PREFIX) else f"{IMAGE_KEY_PREFIX}{camera}"


def build_reader_index(reader: Any) -> ReaderIndex:
    """从官方 ``hf_dataset`` 读出绝对索引与 (episode, frame) 的映射。

    优先使用官方返回的 ``index`` 列（源绝对索引），而不是自己按 parquet 文件名
    推算——SPEC §7 明确禁止猜文件名或假设每个 episode 独立成文件。
    """
    try:
        columns = reader.hf_dataset.select_columns(["index", "episode_index", "frame_index"])
    except (AttributeError, KeyError) as exc:  # pragma: no cover - 依赖官方布局
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "the official reader does not expose index/episode_index/frame_index columns",
            details={"error": type(exc).__name__},
        ) from exc

    absolute_to_relative: dict[int, int] = {}
    position_to_absolute: dict[tuple[int, int], int] = {}
    for relative, (absolute, episode, frame) in enumerate(
        zip(columns["index"], columns["episode_index"], columns["frame_index"], strict=True)
    ):
        abs_index = int(absolute)
        key = (int(episode), int(frame))
        if key in position_to_absolute:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the official reader exposes the same (episode, frame) twice",
                details={"episode_index": key[0], "frame_index": key[1]},
            )
        position_to_absolute[key] = abs_index
        absolute_to_relative[abs_index] = relative
    return ReaderIndex(
        absolute_to_relative=absolute_to_relative,
        position_to_absolute=position_to_absolute,
    )


class PreparedACTDataset:
    """一个 split 的冻结窗口视图，包装官方 :class:`LeRobotDataset`。

    只实现 ``__len__`` / ``__getitem__``，不继承 ``torch.utils.data.Dataset``：
    首期 DataLoader 只需序列协议，继承会把 torch 变成模块级硬依赖。
    """

    def __init__(
        self,
        *,
        split: str,
        windows: Sequence[Window],
        reader: Any,
        index: ReaderIndex,
        resolved: ResolvedRecipe,
        tolerance_s: float = 1e-4,
        verify_frame_identity: bool = True,
    ) -> None:
        self.split = str(split)
        self.windows: tuple[Window, ...] = tuple(windows)
        self.resolved = resolved
        self.tolerance_s = float(tolerance_s)
        self.verify_frame_identity = bool(verify_frame_identity)
        self._reader = reader
        self._index = index
        self._chunk_size = int(resolved.temporal.chunk_size)

        foreign = sorted({window.split for window in self.windows} - {self.split})
        if foreign:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "a PreparedACTDataset may only serve one split",
                details={"split": self.split, "foreign_splits": foreign},
            )
        for position, window in enumerate(self.windows):
            self._check_window(window, position)

    # -- 只读属性 ---------------------------------------------------------
    @property
    def action_key(self) -> str:
        return self.resolved.features.action_key

    @property
    def state_key(self) -> str:
        return self.resolved.features.state_key

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(official_image_key(camera) for camera in self.resolved.features.cameras)

    def window_at(self, position: int) -> Window:
        if not 0 <= position < len(self.windows):
            raise IndexError(f"window position out of range: {position}")
        return self.windows[position]

    # -- 协议 -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, position: int) -> dict[str, Any]:
        window = self.window_at(position)
        absolute = self._index.absolute_for(
            window.official_episode_index, window.anchor_frame_index
        )
        try:
            relative = self._index.absolute_to_relative[absolute]
        except KeyError as exc:  # pragma: no cover - 两张表由同一次扫描建立
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the official reader cannot address the window's absolute frame",
                details={"absolute_index": absolute, "split": self.split},
            ) from exc

        try:
            item = self._reader[relative]
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 - 解码失败的两种原因必须分开报
            message = str(exc)
            if "torchcodec" in message or "FFmpeg" in message or "libav" in message:
                raise ApiError(
                    ErrorCode.DEPENDENCY_UNAVAILABLE,
                    "video decoding is unavailable: torchcodec cannot load its FFmpeg shared libraries",
                    details={
                        "error": type(exc).__name__,
                        "remedy": (
                            "install FFmpeg (libavutil/libavcodec/libavformat) or run inside "
                            "the worker-vision image; do not treat this as a data defect"
                        ),
                        "split": self.split,
                    },
                ) from exc
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "the official reader failed to decode a frozen window",
                details={
                    "error": type(exc).__name__,
                    "source_episode_id": window.source_episode_id,
                    "anchor_frame_index": int(window.anchor_frame_index),
                    "split": self.split,
                },
            ) from exc
        if self.verify_frame_identity:
            self._check_identity(item, window, absolute)
        return self._to_batch(item, window)

    # -- 内部 -------------------------------------------------------------
    def _check_window(self, window: Window, position: int) -> None:
        anchor = int(window.anchor_frame_index)
        expected = tuple(range(anchor, anchor + self._chunk_size))
        observed = tuple(int(index) for index in window.action_frame_indices)
        if observed != expected:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "a frozen window does not cover exactly one action chunk from its anchor",
                details={
                    "position": position,
                    "anchor_frame_index": anchor,
                    "chunk_size": self._chunk_size,
                    "action_frame_indices": list(observed),
                },
            )

    def _check_identity(self, item: dict[str, Any], window: Window, absolute: int) -> None:
        """TP-11：官方返回的帧身份必须与窗口索引逐项一致。"""
        observed = {
            "index": int(item["index"]),
            "episode_index": int(item["episode_index"]),
            "frame_index": int(item["frame_index"]),
        }
        expected = {
            "index": absolute,
            "episode_index": int(window.official_episode_index),
            "frame_index": int(window.anchor_frame_index),
        }
        if observed != expected:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the official reader returned a frame that does not match the frozen window",
                details={
                    "expected": expected,
                    "observed": observed,
                    "source_episode_id": window.source_episode_id,
                    "split": self.split,
                },
            )

    def _to_batch(self, item: dict[str, Any], window: Window) -> dict[str, Any]:
        import torch  # noqa: PLC0415 - 只在真正取样本时导入

        features = self.resolved.features
        if self.state_key not in item:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the official reader did not return the frozen state key",
                details={"state_key": self.state_key, "available": sorted(item)},
            )

        state = torch.as_tensor(item[self.state_key], dtype=torch.float32).reshape(-1)
        if tuple(state.shape) != (features.state_dim,):
            raise ApiError(
                ErrorCode.FEATURE_INCOMPATIBLE,
                "the reader returned a state vector that contradicts the resolved feature contract",
                details={
                    "expected_dim": features.state_dim,
                    "observed_shape": list(state.shape),
                    "state_key": self.state_key,
                },
            )

        images: dict[str, Any] = {}
        for camera in features.cameras:
            key = official_image_key(camera)
            if key not in item:
                raise ApiError(
                    ErrorCode.INVALID_INPUT,
                    "the official reader did not return a frozen camera",
                    details={"camera_key": key, "available": sorted(item)},
                )
            images[key] = self._as_image(item[key], camera)

        if self.action_key not in item:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the official reader did not return the frozen action key",
                details={"action_key": self.action_key},
            )
        action = torch.as_tensor(item[self.action_key], dtype=torch.float32)
        if action.ndim != 2 or tuple(action.shape) != (self._chunk_size, features.action_dim):
            raise ApiError(
                ErrorCode.FEATURE_INCOMPATIBLE,
                "the action chunk does not match chunk_size x action_dim",
                details={
                    "expected_shape": [self._chunk_size, features.action_dim],
                    "observed_shape": list(action.shape),
                    "action_key": self.action_key,
                },
            )

        pad_key = f"{self.action_key}_is_pad"
        is_pad = item.get(pad_key)
        if is_pad is None:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the official reader did not report action padding",
                details={"expected_key": pad_key},
            )
        is_pad = torch.as_tensor(is_pad, dtype=torch.bool).reshape(-1)
        if tuple(is_pad.shape) != (self._chunk_size,):
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the reported action padding has the wrong length",
                details={"expected": self._chunk_size, "observed": list(is_pad.shape)},
            )
        # tail_policy=drop：任何 padding 都意味着这个窗口根本不该存在。
        if bool(is_pad.any()):
            raise ApiError(
                ErrorCode.EMPTY_REQUIRED_WINDOWS,
                "a frozen window relies on padding; tail_policy=drop forbids padded labels",
                details={
                    "source_episode_id": window.source_episode_id,
                    "anchor_frame_index": int(window.anchor_frame_index),
                    "padded_positions": [index for index, flag in enumerate(is_pad.tolist()) if flag],
                    "split": self.split,
                },
            )

        batch: dict[str, Any] = {self.state_key: state, **images}
        batch[self.action_key] = action
        batch[pad_key] = is_pad
        batch["frame_index"] = int(item["frame_index"])
        batch["episode_index"] = int(item["episode_index"])
        batch["index"] = int(item["index"])
        return batch

    def _as_image(self, value: Any, camera: str) -> Any:
        """显式执行 decode→CHW→float[0,1]，并把结果与合同里的图像 shape 对齐。"""
        import torch  # noqa: PLC0415

        image = torch.as_tensor(value)
        if image.ndim != 3:
            raise ApiError(
                ErrorCode.FEATURE_INCOMPATIBLE,
                "a decoded camera frame is not a 3D tensor",
                details={"camera": camera, "observed_shape": list(image.shape)},
            )
        if image.shape[0] != 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.shape[0] != 3:
            raise ApiError(
                ErrorCode.FEATURE_INCOMPATIBLE,
                "a decoded camera frame is not RGB",
                details={"camera": camera, "observed_shape": list(image.shape)},
            )
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0
        else:
            image = image.to(torch.float32)

        expected = self.resolved.features.image_shapes.get(camera)
        if expected and list(image.shape) != [int(size) for size in expected]:
            raise ApiError(
                ErrorCode.FEATURE_INCOMPATIBLE,
                "a decoded camera frame does not match the resolved image shape",
                details={
                    "camera": camera,
                    "expected_shape": [int(size) for size in expected],
                    "observed_shape": list(image.shape),
                    "transforms": list(IMAGE_TRANSFORM_STEPS),
                },
            )
        return image


def open_prepared_dataset(
    *,
    split: str,
    windows: Sequence[Window],
    resolved: ResolvedRecipe,
    source_root: str | Path,
    repo_id: str,
    tolerance_s: float = 1e-4,
    video_backend: str | None = None,
    download_videos: bool = False,
    verify_frame_identity: bool = True,
) -> PreparedACTDataset:
    """按冻结窗口打开官方 Reader，并返回受限视图。

    ``episodes`` 只包含窗口涉及的官方 episode：训练 sampler 因此**在物理上**无法
    访问其他 split 的帧（TP-11），而不是靠调用方自觉。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    windows = tuple(windows)
    if not windows:
        raise ApiError(
            ErrorCode.EMPTY_REQUIRED_WINDOWS,
            "a prepared dataset needs at least one frozen window",
            details={"split": split},
        )

    features = resolved.features
    chunk_size = int(resolved.temporal.chunk_size)
    fps = int(features.fps)
    if fps <= 0:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "the resolved recipe has no positive aligned fps",
            details={"fps": fps},
        )

    episodes = sorted({int(window.official_episode_index) for window in windows})
    delta_timestamps = {features.action_key: [index / float(fps) for index in range(chunk_size)]}

    root = Path(source_root)
    if not root.is_dir():
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "the source root passed to the reader does not exist",
            details={"source_root": str(root)},
        )

    try:
        reader = LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            tolerance_s=float(tolerance_s),
            download_videos=bool(download_videos),
            video_backend=video_backend,
        )
    except Exception as exc:  # noqa: BLE001 - 官方构造失败要分类，不能原样抛 traceback
        message = str(exc)
        if "torchcodec" in message or "FFmpeg" in message or "libav" in message:
            raise ApiError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "video decoding is unavailable: the official reader needs torchcodec plus FFmpeg shared libraries",
                details={
                    "error": type(exc).__name__,
                    "remedy": "install FFmpeg (libavutil/libavcodec/libavformat) or run inside the worker-vision image",
                },
            ) from exc
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "the official reader could not be constructed for this source root",
            details={"error": type(exc).__name__, "message": message[:400]},
        ) from exc
    index = build_reader_index(reader)
    return PreparedACTDataset(
        split=split,
        windows=windows,
        reader=reader,
        index=index,
        resolved=resolved,
        tolerance_s=float(tolerance_s),
        verify_frame_identity=bool(verify_frame_identity),
    )
