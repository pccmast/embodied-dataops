"""Clip/keyframe planning tests (SPEC_V2.1 §5.1, V21-07).

Pure functions: no ffmpeg, no database, no object store.  Every assertion is about
the half-open window arithmetic, because a one-frame shift here silently desynchronises
the whole AI annotation layer.
"""

from __future__ import annotations

import pytest

from embodied_dataops.errors import ApiError
from embodied_dataops.processing.clips import (
    CLIP_SECONDS,
    MAX_FRAMES_PER_CLIP,
    NS_PER_SECOND,
    PREPROCESS_CONFIG_SHA256,
    clip_id_for,
    keyframe_seconds,
    plan_clips,
    target_size,
)

pytestmark = pytest.mark.no_db

SHA = "a" * 64


def _plan(duration_s: float, fps: int = 10):
    return plan_clips(
        episode_id="ep_test",
        source_asset_sha256=SHA,
        duration_ns=int(round(duration_s * NS_PER_SECOND)),
        aligned_fps=fps,
    )


# ---------------------------------------------------------------------------
# keyframe selection
# ---------------------------------------------------------------------------


def test_keyframes_are_integer_seconds_excluding_an_exact_end():
    assert keyframe_seconds(10 * NS_PER_SECOND) == list(range(10))
    # exactly 8 s: the point at t=8 s is the end of a half-open window, so it is not a frame
    assert keyframe_seconds(8 * NS_PER_SECOND) == list(range(8))
    assert keyframe_seconds(8 * NS_PER_SECOND + 1) == list(range(9))


def test_sub_second_episode_keeps_only_the_first_keyframe():
    assert keyframe_seconds(NS_PER_SECOND // 2) == [0]
    assert keyframe_seconds(0) == []


# ---------------------------------------------------------------------------
# clip windows
# ---------------------------------------------------------------------------


def test_ten_second_episode_splits_into_eight_plus_two():
    plan = _plan(10.0)
    assert plan.clip_count == 2
    first, second = plan.clips
    assert (first.start_ns, first.end_ns) == (0, 8 * NS_PER_SECOND)
    assert first.frame_count == 8
    assert (second.start_ns, second.end_ns) == (8 * NS_PER_SECOND, 10 * NS_PER_SECOND)
    assert second.frame_count == 2


def test_exact_eight_second_episode_yields_one_full_clip():
    plan = _plan(8.0)
    assert plan.clip_count == 1
    assert plan.clips[0].frame_count == 8


def test_trailing_clip_shorter_than_one_second_is_kept():
    plan = _plan(8.5)
    assert plan.clip_count == 2
    assert plan.clips[1].duration_ns == NS_PER_SECOND // 2
    assert plan.clips[1].frame_count == 1


def test_every_clip_respects_the_eight_frame_cap():
    plan = _plan(95.0)
    assert plan.clip_count == 12  # ceil(95/8)
    assert all(clip.frame_count <= MAX_FRAMES_PER_CLIP for clip in plan.clips)
    assert sum(clip.frame_count for clip in plan.clips) == 95


def test_clips_are_contiguous_and_non_overlapping():
    plan = _plan(37.0)
    for previous, current in zip(plan.clips, plan.clips[1:], strict=False):
        assert previous.end_ns == current.start_ns
    assert plan.clips[-1].end_ns == plan.duration_ns


def test_grid_indices_follow_the_aligned_fps():
    plan = _plan(10.0, fps=10)
    assert plan.clips[0].frame_indices == (0, 10, 20, 30, 40, 50, 60, 70)
    assert plan.clips[1].frame_indices == (80, 90)


def test_twenty_fps_grid_doubles_the_indices_without_moving_the_time():
    plan = _plan(4.0, fps=20)
    assert plan.clips[0].frame_indices == (0, 20, 40, 60)
    assert [frame.t_ns for frame in plan.clips[0].keyframes] == [
        0,
        NS_PER_SECOND,
        2 * NS_PER_SECOND,
        3 * NS_PER_SECOND,
    ]


# ---------------------------------------------------------------------------
# identifier stability
# ---------------------------------------------------------------------------


def test_clip_id_is_deterministic():
    first = _plan(10.0)
    second = _plan(10.0)
    assert [c.clip_id for c in first.clips] == [c.clip_id for c in second.clips]
    assert all(len(clip.clip_id) == 64 for clip in first.clips)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episode_id": "ep_other"},
        {"source_asset_sha256": "b" * 64},
        {"start_ns": 1},
        {"end_ns": 9 * NS_PER_SECOND},
        {"preprocessing_sha256": "c" * 64},
    ],
)
def test_clip_id_changes_with_every_input(kwargs):
    base = {
        "episode_id": "ep_test",
        "source_asset_sha256": SHA,
        "start_ns": 0,
        "end_ns": 8 * NS_PER_SECOND,
        "preprocessing_sha256": PREPROCESS_CONFIG_SHA256,
    }
    assert clip_id_for(**base) != clip_id_for(**{**base, **kwargs})


# ---------------------------------------------------------------------------
# rejection paths
# ---------------------------------------------------------------------------


def test_over_long_episode_is_rejected_not_truncated():
    with pytest.raises(ApiError) as excinfo:
        _plan(601.0)
    assert "limit" in str(excinfo.value)


def test_non_positive_duration_is_rejected():
    with pytest.raises(ApiError):
        plan_clips(episode_id="ep", source_asset_sha256=SHA, duration_ns=0)


def test_short_hash_is_rejected():
    with pytest.raises(ApiError):
        plan_clips(episode_id="ep", source_asset_sha256="deadbeef", duration_ns=NS_PER_SECOND)


def test_clip_seconds_constant_matches_the_spec():
    assert CLIP_SECONDS == 8


# ---------------------------------------------------------------------------
# preprocessing geometry
# ---------------------------------------------------------------------------


def test_target_size_caps_the_longest_side_and_keeps_the_ratio():
    assert target_size(1920, 1080) == (768, 432)
    assert target_size(1080, 1920) == (432, 768)
    assert target_size(640, 480) == (640, 480)
    assert target_size(4000, 100) == (768, 19)


def test_target_size_rejects_degenerate_images():
    with pytest.raises(ApiError):
        target_size(0, 480)
