"""有效窗口索引与拒绝统计（EDO-TRAIN-PREP-1.1 §7）。

重点不是"能生成窗口"，而是**坏样本必须变成计数，而不是变成 padding**。
"""

from __future__ import annotations

import pytest

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.split import SplitResult
from embodied_dataops.training.windows import (
    REJECTION_REASONS,
    EpisodeFrames,
    build_window_index,
)

pytestmark = pytest.mark.no_db


def _split(**splits_by_episode: str) -> SplitResult:
    counts: dict[str, int] = {}
    for split in splits_by_episode.values():
        counts[split] = counts.get(split, 0) + 1
    return SplitResult(
        strategy="explicit_group_assignment_v1",
        seed=None,
        ratios=None,
        by_group={f"g_{episode}": split for episode, split in splits_by_episode.items()},
        by_episode=dict(splits_by_episode),
        counts=counts,
    )


def _episode(frames: int = 10, **overrides) -> EpisodeFrames:
    values = {
        "episode_id": "ep-1",
        "official_episode_index": 0,
        "frame_count": frames,
        "invalid_frames": {},
        "row_digest": "d" * 64,
    }
    values.update(overrides)
    return EpisodeFrames(**values)


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------
def test_windows_cover_every_full_chunk_and_drop_the_tail():
    index = build_window_index(
        [_episode(10)], _split(**{"ep-1": "train"}), chunk_size=4, stride=2
    )
    assert [window.anchor_frame_index for window in index.windows] == [0, 2, 4, 6]
    # anchor 8 需要 8..11，只有 10 帧：丢弃并计数，不复制成有效标签。
    assert index.tail_dropped == 1
    assert index.counts == {"train": 4}


def test_a_window_labels_exactly_one_chunk_of_actions():
    index = build_window_index(
        [_episode(8)], _split(**{"ep-1": "train"}), chunk_size=4, stride=4
    )
    window = index.windows[0]
    assert window.action_frame_indices == (0, 1, 2, 3)
    assert window.source_episode_id == "ep-1"
    assert window.official_episode_index == 0
    assert window.split == "train"


def test_an_episode_shorter_than_one_chunk_yields_no_window():
    index = build_window_index(
        [_episode(3)], _split(**{"ep-1": "train"}), chunk_size=4, stride=1, require_nonempty=()
    )
    assert index.windows == ()
    assert index.tail_dropped == 3


def test_windows_are_sorted_and_digests_are_stable():
    episodes = [
        _episode(6, episode_id="ep-b", official_episode_index=1, row_digest="b" * 64),
        _episode(6, episode_id="ep-a", official_episode_index=0, row_digest="a" * 64),
    ]
    split = _split(**{"ep-a": "train", "ep-b": "train"})
    first = build_window_index(episodes, split, chunk_size=4, stride=2)
    second = build_window_index(list(reversed(episodes)), split, chunk_size=4, stride=2)

    assert [w.source_episode_id for w in first.windows] == [
        "ep-a",
        "ep-a",
        "ep-b",
        "ep-b",
    ]
    assert [w.source_row_digest for w in first.windows] == [
        w.source_row_digest for w in second.windows
    ]
    # 不同 anchor 的窗口即使行摘要相同，也必须得到不同的窗口身份。
    assert first.windows[0].source_row_digest != first.windows[1].source_row_digest


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_one_bad_frame_rejects_every_window_that_covers_it():
    index = build_window_index(
        [_episode(10, invalid_frames={3: "NONFINITE_STATE"})],
        _split(**{"ep-1": "train"}),
        chunk_size=4,
        stride=2,
        require_nonempty=["train"],
    )
    assert [window.anchor_frame_index for window in index.windows] == [4, 6]
    assert [(item.anchor_frame_index, item.reason, item.frame_index) for item in index.rejected] == [
        (0, "NONFINITE_STATE", 3),
        (2, "NONFINITE_STATE", 3),
    ]
    assert index.rejected_by_reason == {"NONFINITE_STATE": 2}


def test_rejection_counts_are_aggregated_per_reason():
    frames = [
        _episode(8, episode_id="ep-a", invalid_frames={0: "DECODE_ERROR"}, row_digest="a" * 64),
        _episode(8, episode_id="ep-b", invalid_frames={1: "TIME_GAP"}, row_digest="b" * 64),
    ]
    index = build_window_index(
        frames,
        _split(**{"ep-a": "train", "ep-b": "train"}),
        chunk_size=4,
        stride=4,
        require_nonempty=["train"],
    )
    assert index.rejected_by_reason == {"DECODE_ERROR": 1, "TIME_GAP": 1}
    assert index.counts == {"train": 2}


def test_an_unknown_rejection_reason_is_refused():
    with pytest.raises(ApiError) as excinfo:
        build_window_index(
            [_episode(8, invalid_frames={0: "BAD_LUCK"})],
            _split(**{"ep-1": "train"}),
            chunk_size=4,
            stride=1,
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.details["allowed"] == list(REJECTION_REASONS)


def test_an_episode_without_a_split_is_refused():
    with pytest.raises(ApiError) as excinfo:
        build_window_index([_episode(8)], _split(), chunk_size=4, stride=1)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_a_duplicate_episode_in_the_listing_is_refused():
    with pytest.raises(ApiError) as excinfo:
        build_window_index(
            [_episode(8), _episode(8)],
            _split(**{"ep-1": "train"}),
            chunk_size=4,
            stride=1,
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_a_train_split_with_no_valid_window_fails_as_empty_train():
    with pytest.raises(ApiError) as excinfo:
        build_window_index(
            [_episode(4, invalid_frames={0: "MISSING_ACTION"})],
            _split(**{"ep-1": "train"}),
            chunk_size=4,
            stride=1,
            require_nonempty=["train"],
        )
    assert excinfo.value.code is ErrorCode.EMPTY_TRAIN_WINDOWS
    assert excinfo.value.details["rejected_by_reason"] == {"MISSING_ACTION": 1}


def test_a_required_split_without_any_episode_fails_as_empty_required():
    with pytest.raises(ApiError) as excinfo:
        build_window_index(
            [_episode(8)],
            _split(**{"ep-1": "train"}),
            chunk_size=4,
            stride=4,
            require_nonempty=["train", "val", "test"],
        )
    assert excinfo.value.code is ErrorCode.EMPTY_REQUIRED_WINDOWS
    assert excinfo.value.details["empty_splits"] == ["val", "test"]


def test_non_positive_chunk_or_stride_is_refused():
    for chunk, stride in ((0, 1), (4, 0), (-1, 1)):
        with pytest.raises(ApiError) as excinfo:
            build_window_index([_episode(8)], _split(**{"ep-1": "train"}), chunk_size=chunk, stride=stride)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def test_summary_reports_every_filter_effect():
    index = build_window_index(
        [_episode(10, invalid_frames={3: "NONFINITE_STATE"})],
        _split(**{"ep-1": "train"}),
        chunk_size=4,
        stride=2,
        require_nonempty=["train"],
    )
    summary = index.summary()
    assert summary["window_counts"] == {"train": 2}
    assert summary["rejected_by_reason"] == {"NONFINITE_STATE": 2}
    assert summary["tail_dropped"] == 1
    assert summary["chunk_size"] == 4 and summary["stride"] == 2


def test_documents_are_plain_json_ready():
    index = build_window_index(
        [_episode(6)], _split(**{"ep-1": "train"}), chunk_size=4, stride=2
    )
    windows, rejected = index.as_documents()
    assert windows[0]["action_frame_indices"] == [0, 1, 2, 3]
    assert isinstance(windows[0]["anchor_frame_index"], int)
    assert rejected == []
