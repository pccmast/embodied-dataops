"""有效窗口索引与拒绝统计（EDO-TRAIN-PREP-1.1 §7）。

这一层决定「哪些帧可以被训练消费」，并且**只做过滤，不修数据**：

* 一个窗口内任意一帧缺失、非有限、解码失败、时间断裂、QC 不合格 → 整窗拒绝，
  拒绝原因逐条计数；不用 padding、不用零填充、不跨 episode 借帧；
* 尾部不足一个 chunk 的部分按 ``tail_policy=drop`` 丢弃并单独计数，
  **不复制成有效标签**（复制出来的标签会让 loss 看起来更稳，是自欺）；
* 检查覆盖全部候选窗口，不抽样——抽样通过不等于数据完整。

``source_row_digest`` 在本层是「窗口声称覆盖了哪些行」的可复算摘要；
真正的行内容核验由 M2 的受限 Reader 完成（TP-11：Reader 返回的帧身份必须与
本索引一致）。两层分开，才能在没有数据的情况下先把索引逻辑测干净。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.split import SplitResult

__all__ = [
    "REJECTION_REASONS",
    "EpisodeFrames",
    "RejectedWindow",
    "Window",
    "WindowIndex",
    "build_window_index",
]

#: 拒绝原因的封闭词表。新增原因必须同时改这里，避免"随便写个字符串"变成统计黑洞。
REJECTION_REASONS: tuple[str, ...] = (
    "MISSING_STATE",
    "MISSING_ACTION",
    "NONFINITE_STATE",
    "NONFINITE_ACTION",
    "DECODE_ERROR",
    "TIME_GAP",
    "QC_INVALID",
    "PROFILE_INVALID",
)


@dataclass(frozen=True)
class EpisodeFrames:
    """一个 episode 的帧可用性声明（由 M2 的 Reader 层产出）。"""

    episode_id: str
    official_episode_index: int
    frame_count: int
    #: ``frame_index -> 拒绝原因``。未出现的帧表示可用。
    invalid_frames: dict[int, str] = field(default_factory=dict)
    #: 该 episode 的行身份摘要，参与窗口的 ``source_row_digest``。
    row_digest: str = ""


@dataclass(frozen=True)
class Window:
    split: str
    source_episode_id: str
    official_episode_index: int
    anchor_frame_index: int
    action_frame_indices: tuple[int, ...]
    source_row_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_episode_id": self.source_episode_id,
            "official_episode_index": int(self.official_episode_index),
            "anchor_frame_index": int(self.anchor_frame_index),
            "action_frame_indices": [int(index) for index in self.action_frame_indices],
            "source_row_digest": self.source_row_digest,
        }


@dataclass(frozen=True)
class RejectedWindow:
    split: str
    source_episode_id: str
    anchor_frame_index: int
    reason: str
    #: 触发拒绝的具体帧号，便于回查而不是只看到一句"窗口无效"。
    frame_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_episode_id": self.source_episode_id,
            "anchor_frame_index": int(self.anchor_frame_index),
            "reason": self.reason,
            "frame_index": None if self.frame_index is None else int(self.frame_index),
        }


@dataclass(frozen=True)
class WindowIndex:
    chunk_size: int
    stride: int
    windows: tuple[Window, ...]
    rejected: tuple[RejectedWindow, ...]
    tail_dropped: int
    counts: dict[str, int]
    rejected_by_reason: dict[str, int]

    def windows_for(self, split: str) -> tuple[Window, ...]:
        return tuple(window for window in self.windows if window.split == split)

    def as_documents(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [window.as_dict() for window in self.windows],
            [rejected.as_dict() for rejected in self.rejected],
        )

    def summary(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "window_counts": dict(sorted(self.counts.items())),
            "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
            "tail_dropped": self.tail_dropped,
        }


def _window_digest(row_digest: str, anchor: int, chunk_size: int, stride: int) -> str:
    """窗口声称覆盖的行身份：episode 行摘要 + 精确帧区间 + 切分参数。"""
    payload = f"edo-window-v1|{row_digest}|{anchor}|{chunk_size}|{stride}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_window_index(
    episodes: Sequence[EpisodeFrames],
    split: SplitResult,
    *,
    chunk_size: int,
    stride: int,
    require_nonempty: Iterable[str] = (),
) -> WindowIndex:
    """枚举全部候选窗口，逐个判定有效/拒绝，并按 split 校验非空。

    参数不做默认猜测：``chunk_size`` / ``stride`` 必须来自已经冻结的 resolved recipe。
    """
    if chunk_size <= 0 or stride <= 0:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "chunk_size and stride must be positive",
            details={"chunk_size": chunk_size, "stride": stride},
        )

    windows: list[Window] = []
    rejected: list[RejectedWindow] = []
    rejected_by_reason: dict[str, int] = {}
    counts: dict[str, int] = {}
    tail_dropped = 0
    seen: set[str] = set()

    for episode in episodes:
        if episode.episode_id in seen:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "duplicate episode in the frame listing",
                details={"episode_id": episode.episode_id},
            )
        seen.add(episode.episode_id)

        owner = split.by_episode.get(episode.episode_id)
        if owner is None:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "an episode has no split assignment",
                details={"episode_id": episode.episode_id},
            )
        if episode.frame_count < 0:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "frame_count must be non-negative",
                details={"episode_id": episode.episode_id, "frame_count": episode.frame_count},
            )

        for frame_index, reason in episode.invalid_frames.items():
            if reason not in REJECTION_REASONS:
                raise ApiError(
                    ErrorCode.INVALID_INPUT,
                    "unknown frame rejection reason",
                    details={
                        "episode_id": episode.episode_id,
                        "frame_index": frame_index,
                        "reason": reason,
                        "allowed": list(REJECTION_REASONS),
                    },
                )

        # 候选 anchor：只要窗口起点在 episode 内就算候选，尾部不足 chunk 的单独计数。
        for anchor in range(0, max(episode.frame_count, 0), stride):
            end = anchor + chunk_size
            if end > episode.frame_count:
                tail_dropped += 1
                continue
            blocking = next(
                (
                    (frame_index, episode.invalid_frames[frame_index])
                    for frame_index in range(anchor, end)
                    if frame_index in episode.invalid_frames
                ),
                None,
            )
            if blocking is not None:
                frame_index, reason = blocking
                rejected.append(
                    RejectedWindow(
                        split=owner,
                        source_episode_id=episode.episode_id,
                        anchor_frame_index=anchor,
                        reason=reason,
                        frame_index=frame_index,
                    )
                )
                rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
                continue
            windows.append(
                Window(
                    split=owner,
                    source_episode_id=episode.episode_id,
                    official_episode_index=int(episode.official_episode_index),
                    anchor_frame_index=anchor,
                    action_frame_indices=tuple(range(anchor, end)),
                    source_row_digest=_window_digest(
                        episode.row_digest, anchor, chunk_size, stride
                    ),
                )
            )
            counts[owner] = counts.get(owner, 0) + 1

    # 稳定顺序：split、episode、anchor。下游按这个顺序取 batch 才能复现。
    windows.sort(key=lambda item: (item.split, item.source_episode_id, item.anchor_frame_index))
    rejected.sort(key=lambda item: (item.split, item.source_episode_id, item.anchor_frame_index))

    required = list(require_nonempty)
    empty = [name for name in required if counts.get(name, 0) == 0]
    if empty:
        is_train_only = empty == ["train"] and len(required) == 1
        raise ApiError(
            ErrorCode.EMPTY_TRAIN_WINDOWS if is_train_only else ErrorCode.EMPTY_REQUIRED_WINDOWS,
            "a required split has no valid window after filtering",
            details={
                "empty_splits": empty,
                "window_counts": dict(sorted(counts.items())),
                "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
                "tail_dropped": tail_dropped,
            },
        )

    return WindowIndex(
        chunk_size=chunk_size,
        stride=stride,
        windows=tuple(windows),
        rejected=tuple(rejected),
        tail_dropped=tail_dropped,
        counts=dict(sorted(counts.items())),
        rejected_by_reason=dict(sorted(rejected_by_reason.items())),
    )
