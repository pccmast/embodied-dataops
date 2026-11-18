"""Preview video generation for canonical-episode/2.0 (V2 SPEC 2.3).

Previews are **regenerable display artifacts**: ``{camera}_10hz_640x480.mp4`` is
built by picking, for each aligned 10 Hz grid point, the nearest native frame of
that camera and letterboxing it onto a 640x480 canvas.  It is explicitly *not*
training truth and never replaces the native frame bytes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops.canonical.timebase import generate_grid, nearest_index
from embodied_dataops.canonical.v2.contract import ALIGNED_INTERVAL_NS
from embodied_dataops.canonical.v2.model import NativeCameraStream
from embodied_dataops.canonical.video import apply_letterbox, compute_letterbox

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
PREVIEW_FPS = 10


@dataclass
class PreviewEncoding:
    path: Path
    frame_count: int
    frame_indexes: list[int]
    transform_variants: list[dict[str, Any]]


def preview_grid(
    *,
    times_ns: list[int],
    origin_ns: int,
    interval_ns: int = ALIGNED_INTERVAL_NS,
) -> list[int]:
    """The 10 Hz grid the preview must cover (same coverage as the aligned view)."""
    if not times_ns:
        return []
    return generate_grid(
        origin_ns=origin_ns,
        coverage_start_ns=int(times_ns[0]),
        coverage_end_ns=int(times_ns[-1]),
        interval_ns=interval_ns,
    )


def preview_frame_indexes(
    *,
    times_ns: list[int],
    origin_ns: int,
    interval_ns: int = ALIGNED_INTERVAL_NS,
) -> list[int]:
    """Nearest native frame index for every preview grid point (never fails)."""
    if not times_ns:
        return []
    grid = preview_grid(times_ns=times_ns, origin_ns=origin_ns, interval_ns=interval_ns)
    return [nearest_index(times_ns, origin_ns + t_rel) for t_rel in grid]


def encode_camera_preview(
    camera: NativeCameraStream,
    *,
    destination: Path,
    origin_ns: int,
    fps: int = PREVIEW_FPS,
    interval_ns: int = ALIGNED_INTERVAL_NS,
    width: int = PREVIEW_WIDTH,
    height: int = PREVIEW_HEIGHT,
) -> PreviewEncoding:
    """Encode one camera preview; requires ``camera.frame_getter``."""
    if camera.frame_getter is None:
        raise ValueError(f"camera {camera.name!r} has no frame getter; cannot generate a preview")
    indexes = preview_frame_indexes(times_ns=camera.times_ns, origin_ns=origin_ns, interval_ns=interval_ns)
    if not indexes:
        raise ValueError(f"camera {camera.name!r} has no frames; cannot generate a preview")
    variants: list[dict[str, Any]] = []

    def frames() -> Iterator[np.ndarray]:
        for source_index in indexes:
            frame = np.asarray(camera.frame_getter(source_index))
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"camera {camera.name!r} frame {source_index} is not an RGB image"
                )
            transform = compute_letterbox(int(frame.shape[1]), int(frame.shape[0]), out_w=width, out_h=height)
            if not variants:
                variants.append(transform.as_record())
            yield apply_letterbox(frame, transform)

    from embodied_dataops.processing.ffmpeg import encode_canvas_frames_h264

    destination = Path(destination)
    written = encode_canvas_frames_h264(
        frames(), destination, fps=fps, width=width, height=height
    )
    if written != len(indexes):
        raise ValueError(
            f"preview encoder wrote {written} frames but {len(indexes)} were selected"
        )
    return PreviewEncoding(
        path=destination,
        frame_count=written,
        frame_indexes=indexes,
        transform_variants=variants,
    )


__all__ = [
    "PREVIEW_FPS",
    "PREVIEW_HEIGHT",
    "PREVIEW_WIDTH",
    "PreviewEncoding",
    "encode_camera_preview",
    "preview_frame_indexes",
    "preview_grid",
]
