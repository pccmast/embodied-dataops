"""Letterbox transform math for canonical videos (SPEC 5.2).

Every canonical frame is produced on a fixed ``640x480`` black canvas: the
source frame is scaled uniformly (never stretched) and centred, so the inverse
mapping from a canonical pixel back to the source pixel is exact and can be
recorded per frame.

The per-frame record saved in ``metadata.json`` is::

    {"scale": 0.5, "pad_left": 0, "pad_top": 60,
     "src_w": 1280, "src_h": 720,          # required by SPEC 5.2
     "new_w": 640, "new_h": 360,           # additive traceability fields
     "out_w": 640, "out_h": 480}

``scale`` is the single uniform factor (``min(640/src_w, 480/src_h)``); the
mapping of a canonical pixel ``(x, y)`` to the source pixel is
``((x - pad_left) / scale, (y - pad_top) / scale)`` -- which is what detection
annotation overlays need in order to map boxes back to the original video.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied_dataops import TARGET_HEIGHT, TARGET_WIDTH

#: Canvas the canonical contract fixes for the first version (SPEC 5.2).
CANVAS_WIDTH = TARGET_WIDTH
CANVAS_HEIGHT = TARGET_HEIGHT


@dataclass(frozen=True)
class LetterboxTransform:
    """Uniform scale + centring offsets of one source frame on the canvas."""

    src_w: int
    src_h: int
    scale: float
    new_w: int
    new_h: int
    pad_left: int
    pad_top: int
    out_w: int = CANVAS_WIDTH
    out_h: int = CANVAS_HEIGHT

    @property
    def pad_right(self) -> int:
        return self.out_w - self.new_w - self.pad_left

    @property
    def pad_bottom(self) -> int:
        return self.out_h - self.new_h - self.pad_top

    def as_record(self) -> dict[str, Any]:
        """The JSON record stored in ``metadata.json.frame_transforms``."""
        return {
            "scale": self.scale,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "src_w": self.src_w,
            "src_h": self.src_h,
            "new_w": self.new_w,
            "new_h": self.new_h,
            "out_w": self.out_w,
            "out_h": self.out_h,
        }

    def to_source_point(self, x: float, y: float) -> tuple[float, float]:
        """Map a canonical-canvas point back to source pixel coordinates."""
        if self.scale <= 0:
            raise ValueError("letterbox scale must be positive")
        return (x - self.pad_left) / self.scale, (y - self.pad_top) / self.scale

    def to_canvas_point(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.pad_left, y * self.scale + self.pad_top

    def to_source_box(self, box: Sequence[float]) -> list[float]:
        """Map an ``xyxy`` canonical box back to source coordinates."""
        if len(box) != 4:
            raise ValueError("xyxy box must have four values")
        x0, y0 = self.to_source_point(float(box[0]), float(box[1]))
        x1, y1 = self.to_source_point(float(box[2]), float(box[3]))
        return [x0, y0, x1, y1]


def compute_letterbox(
    src_w: int,
    src_h: int,
    *,
    out_w: int = CANVAS_WIDTH,
    out_h: int = CANVAS_HEIGHT,
) -> LetterboxTransform:
    """Compute the deterministic uniform scale + centring for one frame size.

    A ``640x480`` source maps 1:1; wider sources get top/bottom bars; tall
    sources get left/right bars.  The scaled content never exceeds the canvas.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source frame size {src_w}x{src_h}")
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"invalid canvas size {out_w}x{out_h}")
    scale = min(out_w / float(src_w), out_h / float(src_h))
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    # ``round`` on a floating scale can overshoot by one pixel; clamp it so the
    # padded copy can never write outside the canvas.
    new_w = max(1, min(out_w, new_w))
    new_h = max(1, min(out_h, new_h))
    pad_left = (out_w - new_w) // 2
    pad_top = (out_h - new_h) // 2
    return LetterboxTransform(
        src_w=int(src_w),
        src_h=int(src_h),
        scale=float(scale),
        new_w=new_w,
        new_h=new_h,
        pad_left=pad_left,
        pad_top=pad_top,
        out_w=int(out_w),
        out_h=int(out_h),
    )


def resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB ``uint8`` frame (OpenCV first, Pillow as fallback)."""
    if frame.dtype != np.uint8:
        frame = np.clip(np.asarray(frame), 0, 255).astype(np.uint8)
    if frame.ndim == 2:  # grayscale -> RGB, never emit a single-channel canvas
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    try:
        import cv2

        interpolation = cv2.INTER_AREA if (width < frame.shape[1] or height < frame.shape[0]) else cv2.INTER_LINEAR
        return cv2.resize(frame, (width, height), interpolation=interpolation)
    except Exception:  # pragma: no cover - Pillow fallback path
        from PIL import Image

        image = Image.fromarray(frame, mode="RGB")
        return np.asarray(image.resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def apply_letterbox(frame: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    """Place one RGB frame on the black canvas exactly as the transform records."""
    if frame.shape[0] != transform.src_h or frame.shape[1] != transform.src_w:
        raise ValueError(
            f"frame size {frame.shape[1]}x{frame.shape[0]} does not match "
            f"letterbox source {transform.src_w}x{transform.src_h}"
        )
    resized = resize_rgb(frame, transform.new_w, transform.new_h)
    canvas = np.zeros((transform.out_h, transform.out_w, 3), dtype=np.uint8)
    canvas[
        transform.pad_top : transform.pad_top + transform.new_h,
        transform.pad_left : transform.pad_left + transform.new_w,
        :,
    ] = resized
    return canvas


def letterbox_frame(
    frame: np.ndarray,
    *,
    out_w: int = CANVAS_WIDTH,
    out_h: int = CANVAS_HEIGHT,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Convenience: compute the transform from the frame itself and apply it."""
    transform = compute_letterbox(frame.shape[1], frame.shape[0], out_w=out_w, out_h=out_h)
    return apply_letterbox(frame, transform), transform


def letterbox_supported_aspect(src_w: int, src_h: int, *, tolerance: float = 0.02) -> bool:
    """True when the aspect ratio is within 2% of the 4:3 canvas (no bars)."""
    if src_h <= 0:
        return False
    return abs((src_w / src_h) - (CANVAS_WIDTH / CANVAS_HEIGHT)) <= tolerance * (CANVAS_WIDTH / CANVAS_HEIGHT)


def summarize_transforms(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compact summary of per-frame transforms for metadata / diagnostics."""
    if not records:
        return {"frame_count": 0, "uniform": True, "variants": []}
    variants: list[dict[str, Any]] = []
    for record in records:
        key = (
            record.get("src_w"),
            record.get("src_h"),
            record.get("scale"),
            record.get("pad_left"),
            record.get("pad_top"),
        )
        if key not in [
            (v["src_w"], v["src_h"], v["scale"], v["pad_left"], v["pad_top"]) for v in variants
        ]:
            variants.append(record)
            if len(variants) >= 10:
                break
    return {
        "frame_count": len(records),
        "uniform": len(variants) == 1,
        "variants": variants,
        "frames_with_bars": sum(1 for r in records if r.get("pad_left") or r.get("pad_top")),
    }


def fit_dimensions(src_w: int, src_h: int, out_w: int = CANVAS_WIDTH, out_h: int = CANVAS_HEIGHT) -> tuple[int, int]:
    """Scaled content size (no padding) for a source frame."""
    transform = compute_letterbox(src_w, src_h, out_w=out_w, out_h=out_h)
    return transform.new_w, transform.new_h


def letterbox_variance_metrics(frames: Sequence[np.ndarray]) -> dict[str, float]:
    """Gray Laplacian variance per frame, used by the BLUR_FRAME_RATE rule.

    The quality rule itself lives in ``embodied_dataops.quality``; this helper
    only measures the canonical frames so both modules share one definition.
    """
    values: list[float] = []
    for frame in frames:
        gray = (
            np.asarray(frame, dtype=np.float32).mean(axis=2)
            if frame.ndim == 3
            else np.asarray(frame, dtype=np.float32)
        )
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        values.append(float(lap.var()))
    if not values:
        return {"frames": 0, "blurry_frames": 0, "blurry_ratio": 0.0, "min_variance": 0.0, "mean_variance": 0.0}
    blurry = sum(1 for value in values if value < 40.0)
    return {
        "frames": len(values),
        "blurry_frames": blurry,
        "blurry_ratio": blurry / len(values),
        "min_variance": min(values),
        "mean_variance": sum(values) / len(values),
    }


__all__ = [
    "CANVAS_HEIGHT",
    "CANVAS_WIDTH",
    "LetterboxTransform",
    "apply_letterbox",
    "compute_letterbox",
    "fit_dimensions",
    "letterbox_frame",
    "letterbox_supported_aspect",
    "letterbox_variance_metrics",
    "resize_rgb",
    "summarize_transforms",
]
