"""Unit tests for preview frame extraction and thumbnails (SPEC 5.2, 8.1).

The tests generate tiny synthetic MP4s with OpenCV; they never touch fixture
data or the object store.  If the local OpenCV build cannot encode MP4 the
video-dependent tests skip with an explicit reason (the DETECT_OBJECTS /
BUILD_PREVIEW behaviour is additionally covered by integration tests).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from embodied_dataops import TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.processing import preview

pytestmark = pytest.mark.unit


def write_test_video(
    path: Path, *, frame_count: int = 25, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT, fps: float = 10.0
) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder is unavailable in this environment")
    try:
        for index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, : max(1, width // 2)] = (30, 120, 200)
            cv2.rectangle(frame, (width // 8, height // 8), (width // 2, height // 2), (0, 200, 255), -1)
            cv2.putText(
                frame, str(index), (width // 2 + 10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture()
def video_640(tmp_path: Path) -> Path:
    return write_test_video(tmp_path / "canonical_front.mp4")


# ---------------------------------------------------------------------------
# probing / decoding
# ---------------------------------------------------------------------------


def test_probe_video_reports_canonical_facts(video_640: Path) -> None:
    probe = preview.probe_video(video_640)
    assert (probe.width, probe.height) == (TARGET_WIDTH, TARGET_HEIGHT)
    assert probe.fps == pytest.approx(10.0)
    assert probe.frame_count == 25
    assert probe.backend in {"cv2", "av"}


def test_frames_are_sampled_at_one_fps(video_640: Path) -> None:
    frames = list(preview.iter_frames(video_640, stride=preview.SAMPLE_STRIDE, require_640x480=True))
    assert [frame.frame_index for frame in frames] == [0, 10, 20]
    assert [frame.t_ns for frame in frames] == [0, 1_000_000_000, 2_000_000_000]
    for frame in frames:
        assert frame.rgb.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3)
        assert frame.rgb.dtype == np.uint8


def test_first_frame_is_frame_zero(video_640: Path) -> None:
    first = preview.read_first_frame(video_640)
    assert first.frame_index == 0
    assert first.t_ns == 0


def test_non_640x480_canonical_frame_is_rejected(tmp_path: Path) -> None:
    video = write_test_video(tmp_path / "small.mp4", frame_count=5, width=320, height=240)
    with pytest.raises(TerminalTaskError) as excinfo:
        list(preview.iter_frames(video, stride=10, require_640x480=True))
    assert excinfo.value.code == ErrorCode.FILE_DECODE_ERROR
    assert excinfo.value.details["width"] == 320


def test_corrupt_video_is_file_decode_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is definitely not an mp4 video")
    with pytest.raises(TerminalTaskError) as excinfo:
        preview.probe_video(corrupt)
    assert excinfo.value.code == ErrorCode.FILE_DECODE_ERROR
    with pytest.raises(TerminalTaskError) as excinfo:
        list(preview.iter_frames(corrupt))
    assert excinfo.value.code == ErrorCode.FILE_DECODE_ERROR
    with pytest.raises(TerminalTaskError) as excinfo:
        preview.make_thumbnail(corrupt)
    assert excinfo.value.code == ErrorCode.FILE_DECODE_ERROR


def test_empty_file_is_file_decode_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(TerminalTaskError) as excinfo:
        preview.read_first_frame(empty)
    assert excinfo.value.code == ErrorCode.FILE_DECODE_ERROR


# ---------------------------------------------------------------------------
# thumbnails / letterboxing
# ---------------------------------------------------------------------------


def test_thumbnail_is_a_640x480_jpeg(video_640: Path) -> None:
    thumbnail = preview.make_thumbnail(video_640)
    assert thumbnail.width == TARGET_WIDTH
    assert thumbnail.height == TARGET_HEIGHT
    assert thumbnail.frame_index == 0
    assert thumbnail.t_ns == 0
    assert thumbnail.data[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert thumbnail.sha256 == hashlib.sha256(thumbnail.data).hexdigest()
    decoded = cv2.imdecode(np.frombuffer(thumbnail.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3)


def test_write_thumbnail_writes_file(tmp_path: Path, video_640: Path) -> None:
    output = tmp_path / "frames" / "thumbnail.jpg"
    thumbnail = preview.write_thumbnail(video_640, output)
    assert output.is_file()
    assert output.stat().st_size == len(thumbnail.data)


def test_letterbox_centres_non_canonical_input() -> None:
    frame = np.full((200, 320, 3), 200, dtype=np.uint8)
    result = preview.letterbox_to_canvas(frame)
    assert result.image.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3)
    assert result.src_w == 320
    assert result.src_h == 200
    assert result.scale == pytest.approx(min(TARGET_WIDTH / 320, TARGET_HEIGHT / 200))
    assert result.pad_top > 0  # wide input -> vertical letterbox bands
    assert result.pad_left == 0
    assert (result.image[0, :] == 0).all()  # black padding
    centre = result.image[result.pad_top + 10, TARGET_WIDTH // 2]
    assert tuple(int(value) for value in centre) == (200, 200, 200)


def test_ensure_640x480_rejects_bad_shape_and_channels() -> None:
    with pytest.raises(TerminalTaskError) as excinfo:
        preview.ensure_640x480(np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8))
    assert excinfo.value.code == ErrorCode.FILE_DECODE_ERROR
    with pytest.raises(TerminalTaskError):
        preview.ensure_640x480(np.zeros((100, 100, 3), dtype=np.uint8))
    assert (
        preview.ensure_640x480(np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)).shape
        == (TARGET_HEIGHT, TARGET_WIDTH, 3)
    )


def test_decode_jpeg_dimensions_validates_stored_thumbnail(video_640: Path) -> None:
    thumbnail = preview.make_thumbnail(video_640)
    assert preview.decode_jpeg_dimensions(thumbnail.data) == (TARGET_WIDTH, TARGET_HEIGHT)
    with pytest.raises(TerminalTaskError) as excinfo:
        preview.decode_jpeg_dimensions(b"definitely-not-a-jpeg")
    assert excinfo.value.code == ErrorCode.FILE_DECODE_ERROR
    with pytest.raises(TerminalTaskError):
        preview.decode_jpeg_dimensions(b"")


def test_frame_timestamps_use_the_ten_hz_grid() -> None:
    assert preview.frame_t_ns(0) == 0
    assert preview.frame_t_ns(10) == 1_000_000_000
    assert preview.frame_t_ns(25) == 2_500_000_000
