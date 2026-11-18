"""Deterministic keyframe and clip planning (SPEC_V2.1 §5.1, acceptance V21-07).

The algorithm is fully specified and deliberately boring, because every downstream
hash (``clip_id``, prompt hash, inference request hash) depends on it:

* keyframes are the **integer seconds** ``0, 1, ..., floor(duration_s)`` of the
  episode timeline, and the last point is dropped when it lands exactly on the end
  (``[start, end)`` semantics -- a frame at ``t == duration`` does not exist);
* clips are consecutive half-open windows of ``min(8 s, remaining)`` with no overlap;
  a trailing window shorter than one second is still kept;
* a clip therefore carries at most 8 keyframes (8 s x 1 fps), which is also the
  provider's per-call image limit;
* ``clip_id = sha256(episode_id | source_asset_sha256 | start_ns | end_ns | preprocess_sha)``.

Nothing here decodes video or touches storage: planning is a pure function of the
duration, so it can be tested without ffmpeg and replayed exactly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.errors import ApiError, ErrorCode

#: Fixed preprocessing profile (CONFIG_DECISIONS_V21 §vlm).  Changing any of these
#: values changes every ``clip_id``, so they live in one place.
KEYFRAME_FPS = 1
CLIP_SECONDS = 8
MAX_FRAMES_PER_CLIP = 8
MAX_SIDE_PX = 768
JPEG_QUALITY = 80
MAX_EPISODE_SECONDS = 600
MAX_CLIPS_PER_EPISODE = MAX_EPISODE_SECONDS // CLIP_SECONDS  # 75

NS_PER_SECOND = 1_000_000_000

PREPROCESS_CONFIG: dict[str, Any] = {
    "keyframe_fps": KEYFRAME_FPS,
    "clip_s": CLIP_SECONDS,
    "max_frames_per_clip": MAX_FRAMES_PER_CLIP,
    "max_side_px": MAX_SIDE_PX,
    "jpeg_quality": JPEG_QUALITY,
}

PREPROCESS_CONFIG_SHA256 = hashlib.sha256(canonical_json_bytes(PREPROCESS_CONFIG)).hexdigest()


@dataclass(frozen=True)
class Keyframe:
    """One integer-second sample inside a clip."""

    second: int
    t_ns: int
    #: Index on the 10 Hz aligned grid (``second * aligned_fps``); the decoder turns it
    #: into a real source frame, and must fail loudly rather than shift by one frame.
    grid_index: int


@dataclass(frozen=True)
class Clip:
    """One ``[start_ns, end_ns)`` window and its keyframes."""

    clip_id: str
    start_ns: int
    end_ns: int
    keyframes: tuple[Keyframe, ...]

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def frame_count(self) -> int:
        return len(self.keyframes)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(frame.grid_index for frame in self.keyframes)


@dataclass(frozen=True)
class ClipPlan:
    """All clips for one episode, plus the evidence needed to reproduce them."""

    episode_id: str
    source_asset_sha256: str
    duration_ns: int
    aligned_fps: int
    preprocessing_sha256: str
    clips: tuple[Clip, ...]

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_asset_sha256": self.source_asset_sha256,
            "duration_ns": self.duration_ns,
            "aligned_fps": self.aligned_fps,
            "preprocessing_sha256": self.preprocessing_sha256,
            "preprocess": dict(PREPROCESS_CONFIG),
            "clip_count": self.clip_count,
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "start_ns": clip.start_ns,
                    "end_ns": clip.end_ns,
                    "frame_count": clip.frame_count,
                    "frame_indices": list(clip.frame_indices),
                }
                for clip in self.clips
            ],
        }


def keyframe_seconds(duration_ns: int) -> list[int]:
    """``0..floor(duration_s)`` with the end point excluded (half-open semantics)."""
    if duration_ns <= 0:
        return []
    whole = duration_ns // NS_PER_SECOND
    remainder = duration_ns % NS_PER_SECOND
    last = int(whole) if remainder else int(whole) - 1
    if last < 0:
        return []
    return list(range(last + 1))


def clip_id_for(
    episode_id: str,
    source_asset_sha256: str,
    start_ns: int,
    end_ns: int,
    preprocessing_sha256: str = PREPROCESS_CONFIG_SHA256,
) -> str:
    """``sha256(episode_id|source_asset_sha|start_ns|end_ns|preprocess_config_sha)``."""
    payload = "|".join(
        [
            str(episode_id),
            str(source_asset_sha256),
            str(int(start_ns)),
            str(int(end_ns)),
            str(preprocessing_sha256),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_clips(
    *,
    episode_id: str,
    source_asset_sha256: str,
    duration_ns: int,
    aligned_fps: int = 10,
    preprocessing_sha256: str = PREPROCESS_CONFIG_SHA256,
    max_episode_seconds: int = MAX_EPISODE_SECONDS,
) -> ClipPlan:
    """Split one episode into deterministic 8 s clips (SPEC_V2.1 §5.1)."""
    if not episode_id:
        raise ApiError(ErrorCode.INVALID_INPUT, "episode_id is required for clip planning")
    if len(source_asset_sha256) != 64:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "source_asset_sha256 must be a 64-character hex digest",
            details={"length": len(source_asset_sha256)},
        )
    if duration_ns <= 0:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "cannot plan clips for an episode with non-positive duration",
            details={"duration_ns": int(duration_ns)},
        )
    if int(duration_ns) > max_episode_seconds * NS_PER_SECOND:
        # Truncating would silently drop evidence, so an over-long episode is rejected.
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "episode exceeds the V2.1 clip-planning limit",
            details={
                "duration_ns": int(duration_ns),
                "max_episode_seconds": int(max_episode_seconds),
            },
        )
    if aligned_fps < 1:
        raise ApiError(ErrorCode.INVALID_INPUT, "aligned_fps must be >= 1")

    seconds = keyframe_seconds(int(duration_ns))
    clip_ns = CLIP_SECONDS * NS_PER_SECOND
    clips: list[Clip] = []
    start_ns = 0
    while start_ns < int(duration_ns):
        end_ns = min(start_ns + clip_ns, int(duration_ns))
        frames: list[Keyframe] = []
        for second in seconds:
            t_ns = second * NS_PER_SECOND
            if start_ns <= t_ns < end_ns:
                frames.append(
                    Keyframe(
                        second=second,
                        t_ns=t_ns,
                        grid_index=int(round(t_ns * aligned_fps / NS_PER_SECOND)),
                    )
                )
        if len(frames) > MAX_FRAMES_PER_CLIP:  # pragma: no cover - guarded by construction
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "clip produced more keyframes than the provider accepts",
                details={"clip_start_ns": start_ns, "frames": len(frames)},
            )
        if frames:
            clips.append(
                Clip(
                    clip_id=clip_id_for(
                        episode_id, source_asset_sha256, start_ns, end_ns, preprocessing_sha256
                    ),
                    start_ns=start_ns,
                    end_ns=end_ns,
                    keyframes=tuple(frames),
                )
            )
        start_ns = end_ns

    return ClipPlan(
        episode_id=episode_id,
        source_asset_sha256=source_asset_sha256,
        duration_ns=int(duration_ns),
        aligned_fps=int(aligned_fps),
        preprocessing_sha256=preprocessing_sha256,
        clips=tuple(clips),
    )


def target_size(width: int, height: int, *, max_side_px: int = MAX_SIDE_PX) -> tuple[int, int]:
    """Longest side capped at ``max_side_px`` while preserving the aspect ratio."""
    if width <= 0 or height <= 0:
        raise ApiError(ErrorCode.INVALID_INPUT, "image dimensions must be positive")
    longest = max(width, height)
    if longest <= max_side_px:
        return width, height
    scale = max_side_px / float(longest)
    # ``max(1, ...)`` keeps a thin strip from collapsing to zero pixels.
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))
