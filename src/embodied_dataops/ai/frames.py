"""Keyframe decoding and JPEG encoding for one clip (SPEC_V2.1 §5.1).

The clip planner decides *which* frames (grid indices on the aligned timeline); this
module turns those indices into the exact bytes sent to the model.  Two properties
matter and are both enforced here:

* **no shifting.**  A frame that cannot be decoded at its planned index is an error, not
  an excuse to take the next one -- a silent shift would break the evidence chain;
* **no upscaling and no re-crop.**  Only the long side is capped (aspect preserved), so
  the model never sees a geometrically distorted view of the scene.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.processing.clips import Clip, target_size


@dataclass(frozen=True)
class EncodedKeyframe:
    """One JPEG keyframe ready to be attached to a provider request."""

    position: int
    grid_index: int
    t_ns: int
    jpeg: bytes
    sha256: str
    width: int
    height: int
    source_width: int
    source_height: int

    @property
    def byte_size(self) -> int:
        return len(self.jpeg)


def encode_clip_keyframes(
    video_path: str | Path,
    clip: Clip,
    *,
    max_side_px: int,
    jpeg_quality: int,
) -> tuple[EncodedKeyframe, ...]:
    """Decode ``clip``'s keyframes from ``video_path`` and encode them as JPEG.

    Decoding is sequential from the first keyframe of the clip: seeking 8 times on a
    long-GOP file is both slower and, on some backends, off by a frame.
    """
    import cv2  # noqa: PLC0415 -- optional dependency, imported only when running

    path = Path(video_path)
    if not path.is_file():
        raise TerminalTaskError(
            ErrorCode.OBJECT_NOT_FOUND,
            "clip source video is missing",
            details={"path": path.name},
        )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "clip source video cannot be opened",
            details={"path": path.name},
        )

    targets = {frame.grid_index: frame for frame in clip.keyframes}
    encoded: list[EncodedKeyframe] = []
    try:
        first_index = min(targets)
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(first_index))
        cursor = first_index
        last_index = max(targets)
        while cursor <= last_index:
            ok, frame = capture.read()
            if not ok:
                break
            target = targets.get(cursor)
            if target is not None:
                encoded.append(
                    _encode(
                        frame,
                        position=len(encoded),
                        grid_index=cursor,
                        t_ns=target.t_ns,
                        max_side_px=max_side_px,
                        jpeg_quality=jpeg_quality,
                        cv2=cv2,
                        path=path,
                    )
                )
            cursor += 1
    finally:
        capture.release()

    if len(encoded) != len(targets):
        missing = sorted(set(targets) - {frame.grid_index for frame in encoded})
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "clip keyframes could not all be decoded at their planned indices",
            details={
                "expected": len(targets),
                "decoded": len(encoded),
                "missing_grid_indexes": missing[:16],
                "path": path.name,
            },
        )
    return tuple(encoded)


def _encode(
    frame,
    *,
    position: int,
    grid_index: int,
    t_ns: int,
    max_side_px: int,
    jpeg_quality: int,
    cv2,
    path: Path,
) -> EncodedKeyframe:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    source_height, source_width = int(rgb.shape[0]), int(rgb.shape[1])
    width, height = target_size(source_width, source_height, max_side_px=max_side_px)
    if (width, height) != (source_width, source_height):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "JPEG encoding failed for a keyframe",
            details={"grid_index": grid_index, "path": path.name},
        )
    data = buffer.tobytes()
    return EncodedKeyframe(
        position=position,
        grid_index=grid_index,
        t_ns=t_ns,
        jpeg=data,
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
        source_width=source_width,
        source_height=source_height,
    )


def frame_manifest(
    *,
    clip_id: str,
    episode_id: str,
    plan_episode_sha: str,
    frames: tuple[EncodedKeyframe, ...],
    max_side_px: int,
    jpeg_quality: int,
) -> dict:
    """The object stored next to the frames: position -> grid index, plus hashes.

    ``evidence_frame_indices`` in a model answer are positions in this list, so this
    manifest is what makes an evidence citation auditable months later.
    """
    return {
        "schema_version": "ai-clip-frames/2.1",
        "clip_id": clip_id,
        "episode_id": episode_id,
        "source_asset_sha256": plan_episode_sha,
        "preprocessing": {"max_side_px": int(max_side_px), "jpeg_quality": int(jpeg_quality)},
        "frame_count": len(frames),
        "frames": [
            {
                "position": frame.position,
                "grid_index": frame.grid_index,
                "t_ns": frame.t_ns,
                "sha256": frame.sha256,
                "byte_size": frame.byte_size,
                "width": frame.width,
                "height": frame.height,
                "source_width": frame.source_width,
                "source_height": frame.source_height,
            }
            for frame in frames
        ],
    }
