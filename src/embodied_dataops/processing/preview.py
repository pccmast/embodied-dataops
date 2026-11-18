"""Frame extraction and thumbnail helpers for canonical 10 Hz video (SPEC 5.2, 8.1).

Canonical episodes are constant-frame-rate 10 fps H.264 MP4 at 640x480; the
canonical sample ``frame_index`` is the decoded frame number, so ``t_ns =
frame_index * 100 ms``.  Decoding uses OpenCV (bundled FFmpeg) as the primary
path and PyAV as a fallback when OpenCV cannot probe/decode the container
(the ROS/MCAP and Ego importers normalise through FFmpeg, so both backends see
the same H.264 stream).

Everything here is read-only with respect to the input video.  A video that
cannot be opened or decoded raises ``TerminalTaskError(FILE_DECODE_ERROR)`` --
never an empty preview or a zero-filled frame (SPEC 5.2/5.5, AGENTS rule 4).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from embodied_dataops import SAMPLE_INTERVAL_NS, TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.errors import ErrorCode, TerminalTaskError

#: 10 Hz canonical video -> 1 fps sampling.
SAMPLE_STRIDE = TARGET_FPS
THUMBNAIL_JPEG_QUALITY = 85


def frame_t_ns(frame_index: int) -> int:
    """Episode-relative timestamp of one 10 Hz canonical frame."""
    return int(frame_index) * SAMPLE_INTERVAL_NS


@dataclass(frozen=True)
class VideoProbe:
    """Basic probing facts recovered from the container."""

    width: int
    height: int
    fps: float
    frame_count: int
    backend: str


@dataclass(frozen=True)
class DecodedFrame:
    """One decoded RGB frame in the canonical 640x480 space."""

    frame_index: int
    t_ns: int
    rgb: np.ndarray


@dataclass(frozen=True)
class LetterboxResult:
    """Letterboxed 640x480 canvas plus the transform used to build it."""

    image: np.ndarray
    scale: float
    pad_left: int
    pad_top: int
    src_w: int
    src_h: int


@dataclass(frozen=True)
class Thumbnail:
    """Encoded JPEG thumbnail plus provenance used for the derived Asset row."""

    data: bytes
    width: int
    height: int
    scale: float
    pad_left: int
    pad_top: int
    src_w: int
    src_h: int
    frame_index: int
    t_ns: int
    jpeg_quality: int = THUMBNAIL_JPEG_QUALITY

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def _probe_cv2(video_path: Path) -> VideoProbe | None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        ok, _ = capture.read()
        if not ok or width <= 0 or height <= 0:
            return None
    finally:
        capture.release()
    if frame_count <= 0:
        frame_count = _count_cv2_frames(video_path)
        if frame_count <= 0:
            return None
    return VideoProbe(width=width, height=height, fps=fps, frame_count=frame_count, backend="cv2")


def _count_cv2_frames(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    count = 0
    try:
        if not capture.isOpened():
            return 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    return count


def _probe_av(video_path: Path) -> VideoProbe | None:
    try:
        import av
    except Exception:  # pragma: no cover - optional dependency
        return None
    try:  # pragma: no cover - only reached when OpenCV cannot decode
        container = av.open(str(video_path))
    except Exception:
        return None
    try:  # pragma: no cover
        streams = container.streams.video
        if not streams:
            return None
        stream = streams[0]
        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        frame_count = int(stream.frames or 0)
        if frame_count <= 0:
            frame_count = sum(1 for _ in container.decode(stream))
        if width <= 0 or height <= 0 or frame_count <= 0:
            return None
        return VideoProbe(
            width=width, height=height, fps=fps, frame_count=frame_count, backend="av"
        )
    except Exception:
        return None
    finally:  # pragma: no cover
        container.close()


def probe_video(video_path: str | Path) -> VideoProbe:
    """Probe a canonical video; raises FILE_DECODE_ERROR when undecodable."""
    path = Path(video_path)
    probe = _probe_cv2(path) or _probe_av(path)
    if probe is None:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical video cannot be decoded by OpenCV or PyAV",
            details={"file": path.name, "backends_tried": ["cv2", "av"]},
        )
    return probe


# ---------------------------------------------------------------------------
# frame decoding
# ---------------------------------------------------------------------------


def ensure_640x480(
    frame: np.ndarray, *, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT
) -> np.ndarray:
    """Reject frames that do not match the canonical 640x480 RGB contract."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Decoded frame is not a 3-channel RGB image",
            details={"shape": [int(value) for value in array.shape]},
        )
    frame_height, frame_width = int(array.shape[0]), int(array.shape[1])
    if (frame_width, frame_height) != (width, height):
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical video frame is not 640x480",
            details={
                "width": frame_width,
                "height": frame_height,
                "expected_width": width,
                "expected_height": height,
            },
        )
    return array


def _iter_cv2_frames(
    video_path: Path, *, stride: int, require_640x480: bool, start: int = 0
) -> Iterator[DecodedFrame]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return
        index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index >= start and (index - start) % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if require_640x480:
                    ensure_640x480(rgb)
                yield DecodedFrame(frame_index=index, t_ns=frame_t_ns(index), rgb=rgb)
            index += 1
    finally:
        capture.release()


def _iter_av_frames(
    video_path: Path, *, stride: int, require_640x480: bool, start: int = 0
) -> Iterator[DecodedFrame]:
    try:
        import av
    except Exception:  # pragma: no cover - optional dependency
        return
    try:  # pragma: no cover - only reached when OpenCV cannot decode
        container = av.open(str(video_path))
    except Exception:
        return
    try:  # pragma: no cover
        streams = container.streams.video
        if not streams:
            return
        stream = streams[0]
        index = 0
        for frame in container.decode(stream):
            if index >= start and (index - start) % stride == 0:
                rgb = frame.to_ndarray(format="rgb24")
                if require_640x480:
                    ensure_640x480(rgb)
                yield DecodedFrame(frame_index=index, t_ns=frame_t_ns(index), rgb=rgb)
            index += 1
    finally:  # pragma: no cover
        container.close()


def iter_frames(
    video_path: str | Path,
    *,
    stride: int = SAMPLE_STRIDE,
    require_640x480: bool = True,
    start: int = 0,
) -> Iterator[DecodedFrame]:
    """Yield every ``stride``-th decoded frame (default: 1 fps of a 10 Hz video).

    Raises ``TerminalTaskError(FILE_DECODE_ERROR)`` when the video yields no
    decodable frame at all.  A truncated stream simply ends early; full-stream
    corruption is a quality-policy concern (``FILE_DECODE_ERROR``), not a reason
    to fabricate frames here.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    path = Path(video_path)
    yielded = 0
    for frame in _iter_cv2_frames(path, stride=stride, require_640x480=require_640x480, start=start):
        yielded += 1
        yield frame
    if yielded:
        return
    for frame in _iter_av_frames(path, stride=stride, require_640x480=require_640x480, start=start):
        yielded += 1
        yield frame
    if not yielded:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical video contains no decodable frames",
            details={"file": path.name, "stride": stride, "backends_tried": ["cv2", "av"]},
        )


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------


def letterbox_to_canvas(
    frame: np.ndarray, *, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT
) -> LetterboxResult:
    """Scale a frame proportionally and centre it on a black 640x480 canvas."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("letterbox_to_canvas expects an HxWx3 RGB image")
    src_h, src_w = int(array.shape[0]), int(array.shape[1])
    if src_w <= 0 or src_h <= 0:
        raise ValueError("letterbox_to_canvas expects a non-empty image")
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(array, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    pad_left = (width - resized_w) // 2
    pad_top = (height - resized_h) // 2
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    return LetterboxResult(
        image=canvas,
        scale=float(scale),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        src_w=src_w,
        src_h=src_h,
    )


def read_first_frame(video_path: str | Path) -> DecodedFrame:
    """Return the first decodable frame (raises FILE_DECODE_ERROR when none)."""
    path = Path(video_path)
    for frame in _iter_cv2_frames(path, stride=1, require_640x480=False):
        return frame
    for frame in _iter_av_frames(path, stride=1, require_640x480=False):
        return frame
    raise TerminalTaskError(
        ErrorCode.FILE_DECODE_ERROR,
        "Canonical video contains no decodable frames",
        details={"file": path.name},
    )


def _encode_jpeg(image_rgb: np.ndarray, jpeg_quality: int) -> bytes:
    bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:  # pragma: no cover - OpenCV JPEG encoder is always present
        raise TerminalTaskError(ErrorCode.INTERNAL_ERROR, "Thumbnail JPEG encoding failed")
    return buffer.tobytes()


def make_thumbnail(
    video_path: str | Path,
    *,
    width: int = TARGET_WIDTH,
    height: int = TARGET_HEIGHT,
    jpeg_quality: int = THUMBNAIL_JPEG_QUALITY,
) -> Thumbnail:
    """Encode the first frame as a 640x480 JPEG (letterboxed when necessary)."""
    first = read_first_frame(video_path)
    boxed = letterbox_to_canvas(first.rgb, width=width, height=height)
    data = _encode_jpeg(boxed.image, jpeg_quality)
    return Thumbnail(
        data=data,
        width=width,
        height=height,
        scale=boxed.scale,
        pad_left=boxed.pad_left,
        pad_top=boxed.pad_top,
        src_w=boxed.src_w,
        src_h=boxed.src_h,
        frame_index=first.frame_index,
        t_ns=first.t_ns,
        jpeg_quality=jpeg_quality,
    )


def decode_jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Decode a stored JPEG and return ``(width, height)``.

    Used by BUILD_PREVIEW to verify a pre-existing canonical thumbnail before it
    is exposed through presigned preview URLs.  Raises
    ``TerminalTaskError(FILE_DECODE_ERROR)`` for undecodable bytes.
    """
    if not data:
        raise TerminalTaskError(ErrorCode.FILE_DECODE_ERROR, "Stored thumbnail is empty")
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR, "Stored thumbnail is not a decodable JPEG"
        )
    return int(array.shape[1]), int(array.shape[0])


def write_thumbnail(
    video_path: str | Path,
    output_path: str | Path,
    *,
    width: int = TARGET_WIDTH,
    height: int = TARGET_HEIGHT,
    jpeg_quality: int = THUMBNAIL_JPEG_QUALITY,
) -> Thumbnail:
    """Write ``thumbnail.jpg`` for one video and return its provenance."""
    thumbnail = make_thumbnail(
        video_path, width=width, height=height, jpeg_quality=jpeg_quality
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(thumbnail.data)
    return thumbnail
