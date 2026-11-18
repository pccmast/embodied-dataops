"""Ego MP4 importer tests (SPEC 5.5, AT-03, AT-07 video half, F04/F07/F10).

The media backend comes from the shared wrapper: ffmpeg (``FFMPEG_BIN`` /
PATH / ``imageio-ffmpeg``) for decoding/encoding and ffprobe, PyAV or a CFR
fallback for presentation timestamps.  Missing *all* backends is a hard failure
rather than a silent skip, because the local acceptance environment is required
to provide the pinned media extras.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from fixtures.generate_synthetic import generate_all, laplacian_variance  # noqa: E402

from embodied_dataops.domain.enums import DataOrigin, EpisodeType, SourceType  # noqa: E402
from embodied_dataops.errors import ApiError, ErrorCode  # noqa: E402
from embodied_dataops.imports.base import ImportContext  # noqa: E402
from embodied_dataops.imports.ego_mp4 import (  # noqa: E402
    MAX_EGO_DURATION_NS,
    EgoMp4Parser,
    _diagnostics_for,
)
from embodied_dataops.processing import ffmpeg as ffmpeg_module  # noqa: E402
from embodied_dataops.processing.ffmpeg import (  # noqa: E402
    FRAME_TOLERANCE_NS,
    FrameTimeline,
    VideoProbe,
    ffmpeg_available,
    frame_timeline,
    full_decode_check,
    iter_frames,
    media_backend_summary,
    probe,
    pyav_available,
    select_grid_frames,
    transcode_to_h264_10fps_640x480_letterbox,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ego_fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if not (ffmpeg_available() or pyav_available()):
        raise RuntimeError(
            "no media backend is available: install the pinned extras (imageio-ffmpeg/PyAV) "
            "or set FFMPEG_BIN/FFPROBE_BIN (see embodied_dataops.processing.ffmpeg)"
        )
    out = tmp_path_factory.mktemp("ego-fixtures")
    generate_all(
        out,
        manifest_path=None,
        only=["F04_ego_video", "F07_corrupt_mp4", "F10_blurry_video"],
        force=True,
    )
    return {
        "F04": out / "F04_ego_video.mp4",
        "F07": out / "F07_corrupt_mp4.mp4",
        "F10": out / "F10_blurry_video.mp4",
    }


def make_context(path: Path, workdir: Path, *, mapping: dict | None = None) -> ImportContext:
    workdir.mkdir(parents=True, exist_ok=True)
    return ImportContext(
        import_id="test-import",
        source_type=SourceType.EGO_MP4,
        data_origin=DataOrigin.SYNTHETIC,
        source_path=path,
        workdir=workdir,
        mapping=mapping or {},
        license="synthetic fixture",
    )


def test_f04_probe_and_full_decode(ego_fixtures, tmp_path: Path) -> None:
    source = ego_fixtures["F04"]
    result = probe(source)
    assert result.width == 640
    assert result.height == 480
    assert abs(result.duration_seconds - 10.0) < 0.1
    assert result.has_audio is False
    assert result.fps is not None and abs(result.fps - 30.0) < 0.01
    check = full_decode_check(source, expected_frames=result.nb_frames)
    assert check.ok, check.stderr_tail
    assert check.frame_count == 300


def test_f04_validate_returns_10hz_grid_evidence(ego_fixtures, tmp_path: Path) -> None:
    parser = EgoMp4Parser()
    context = make_context(ego_fixtures["F04"], tmp_path / "work")
    validation = parser.validate(context)
    assert validation["format"] == "ego-mp4"
    assert validation["frame_count"] == 300
    assert validation["grid_sample_count"] == 100
    assert validation["missing_grid_sample_count"] == 0
    assert validation["duration_ns"] == 10_000_000_000
    assert validation["has_audio"] is False
    assert validation["decode"]["ok"] is True


def test_f04_parse_is_one_ego_episode_without_robot_streams(ego_fixtures, tmp_path: Path) -> None:
    parser = EgoMp4Parser()
    context = make_context(ego_fixtures["F04"], tmp_path / "work")
    parsed = parser.parse(context)
    assert len(parsed.episodes) == 1
    episode = parsed.episodes[0]
    assert episode.episode_type == EpisodeType.EGO_VIDEO_EPISODE
    assert episode.source_episode_index == 0
    assert episode.duration_ns == 10_000_000_000
    assert episode.normalization_valid is True
    assert episode.diagnostics == []
    # A video episode must never carry synthesized robot streams.
    assert "state" not in episode.payload
    assert "action" not in episode.payload
    assert parsed.metadata["sample_count"] == 100


def test_f04_read_episode_gives_100_samples_0_to_9_9s(ego_fixtures, tmp_path: Path) -> None:
    parser = EgoMp4Parser()
    context = make_context(ego_fixtures["F04"], tmp_path / "work")
    data = parser.read_episode(context)
    assert data.sample_count == 100
    assert data.duration_ns == 10_000_000_000
    assert data.normalization_valid is True
    assert data.decoded_frame_count == 300
    assert data.selection.missing_count == 0
    assert data.selection.grid_t_ns[0] == 0
    assert data.selection.grid_t_ns[-1] == 9_900_000_000
    assert data.selection.max_abs_error_ns <= FRAME_TOLERANCE_NS
    frames = list(data.iter_frames())
    assert len(frames) == 100
    assert frames[0].shape == (480, 640, 3)
    assert frames[0].dtype.name == "uint8"
    # A 30 fps source keeps only every third frame; consecutive canonical frames
    # must not be identical in this moving synthetic video.
    assert not (frames[0] == frames[1]).all()


def test_f07_corrupt_video_is_file_decode_error(ego_fixtures, tmp_path: Path) -> None:
    parser = EgoMp4Parser()
    context = make_context(ego_fixtures["F07"], tmp_path / "work")
    with pytest.raises(ApiError) as excinfo:
        parser.validate(context)
    error = excinfo.value
    assert error.code is ErrorCode.FILE_DECODE_ERROR
    details_text = json.dumps(error.details)
    assert "decoded_frame_count" in details_text
    # internal absolute paths are never leaked through diagnostic details
    assert str(context.workdir) not in details_text
    assert str(context.source_path) not in details_text
    assert "C:\\" not in details_text


def test_f10_blurry_video_is_decodable_but_low_variance(ego_fixtures, tmp_path: Path) -> None:
    parser = EgoMp4Parser()
    context = make_context(ego_fixtures["F10"], tmp_path / "work")
    data = parser.read_episode(context)
    assert data.normalization_valid is True
    assert data.sample_count == 100
    frames = list(iter_frames(ego_fixtures["F10"], fps=2.0))
    assert len(frames) >= 5
    variances = [laplacian_variance(frame) for frame in frames]
    blurry = sum(1 for value in variances if value < 40.0)
    assert blurry / len(variances) > 0.3


def test_duration_limit_is_enforced(ego_fixtures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parser = EgoMp4Parser()
    context = make_context(ego_fixtures["F04"], tmp_path / "work")
    long_probe = VideoProbe(
        duration_ns=MAX_EGO_DURATION_NS + 1,
        width=640,
        height=480,
        codec_name="h264",
        pix_fmt="yuv420p",
        has_audio=False,
        format_name="mov,mp4",
        avg_frame_rate=30.0,
        r_frame_rate=30.0,
        nb_frames=300,
    )
    monkeypatch.setattr("embodied_dataops.imports.ego_mp4.probe", lambda path: long_probe)
    with pytest.raises(ApiError) as excinfo:
        parser.validate(context)
    assert excinfo.value.code is ErrorCode.SOURCE_DURATION_LIMIT


def test_select_grid_frames_handles_vfr_and_missing_points() -> None:
    # VFR PTS: 0, 33, 66, 140, 250, 350, 450, 560 ms.
    timeline = FrameTimeline(pts_ns=(0, 33_000_000, 66_000_000, 140_000_000, 250_000_000, 350_000_000, 450_000_000, 560_000_000))
    selection = select_grid_frames(timeline, tolerance_ns=FRAME_TOLERANCE_NS)
    assert selection.sample_count >= 5
    for error in selection.abs_errors_ns:
        assert error <= FRAME_TOLERANCE_NS
    # every selected grid point maps to the closest source index
    assert selection.source_indexes[0] == 0
    assert selection.grid_t_ns == sorted(selection.grid_t_ns)


def test_nonmonotonic_source_timestamps_are_structural_diagnostics() -> None:
    probe_result = VideoProbe(
        duration_ns=1_000_000_000,
        width=640,
        height=480,
        codec_name="h264",
        pix_fmt="yuv420p",
        has_audio=False,
        format_name="mov,mp4",
    )
    timeline = FrameTimeline(pts_ns=(0, 100_000_000, 50_000_000, 200_000_000))
    diagnostics = _diagnostics_for(timeline, select_grid_frames(timeline), probe_result)
    codes = {item["code"] for item in diagnostics}
    assert "TIMESTAMP_NON_MONOTONIC" in codes
    monotonic = next(item for item in diagnostics if item["code"] == "TIMESTAMP_NON_MONOTONIC")
    assert monotonic["structural"] is True
    assert monotonic["severity"] == "ERROR"


def test_probe_and_timeline_work_without_ffprobe_via_pyav(
    ego_fixtures, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: resolving ffprobe used to recurse into resolving ffmpeg while
    # holding a non-reentrant lock (permanent deadlock).
    summary = media_backend_summary()
    assert summary["ffmpeg"], summary
    if not pyav_available():
        pytest.fail("PyAV is required as the local ffprobe substitute (pinned extra)")
    monkeypatch.setattr(ffmpeg_module, "ffprobe_available", lambda: False)
    result = probe(ego_fixtures["F04"])
    assert result.probed_with == "pyav"
    timeline = frame_timeline(ego_fixtures["F04"], probe_result=result)
    assert timeline.backend == "pyav"
    assert timeline.derived_from_cfr is False
    assert timeline.frame_count == 300


def test_letterbox_transcode_is_byte_deterministic(ego_fixtures, tmp_path: Path) -> None:
    source = ego_fixtures["F04"]
    first = tmp_path / "canonical-a.mp4"
    second = tmp_path / "canonical-b.mp4"
    first_result = transcode_to_h264_10fps_640x480_letterbox(source, first)
    second_result = transcode_to_h264_10fps_640x480_letterbox(source, second)
    assert first_result.frame_count == second_result.frame_count == 100
    assert first_result.frame_transforms == second_result.frame_transforms
    # deterministic encode keeps canonical_sha256 stable across re-runs (immutability)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(
        second.read_bytes()
    ).hexdigest()


def test_letterbox_math_and_inverse_mapping() -> None:
    from embodied_dataops.canonical.video import (
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        apply_letterbox,
        compute_letterbox,
        letterbox_supported_aspect,
    )

    square = compute_letterbox(480, 480)
    assert (square.new_w, square.new_h) == (480, 480)
    assert (square.pad_left, square.pad_top) == (80, 0)

    wide = compute_letterbox(1280, 720)
    assert wide.scale == 0.5
    assert (wide.new_w, wide.new_h) == (640, 360)
    assert (wide.pad_left, wide.pad_top) == (0, 60)

    portrait = compute_letterbox(480, 640)
    assert (portrait.new_w, portrait.new_h) == (360, 480)
    assert portrait.pad_left == 140 and portrait.pad_top == 0

    # a canonical pixel maps back to the exact source pixel; letterbox bars stay black
    assert wide.to_source_box([0.0, 60.0, 640.0, 420.0]) == [0.0, 0.0, 1280.0, 720.0]
    canvas = apply_letterbox(np.full((720, 1280, 3), 200, dtype=np.uint8), wide)
    assert canvas.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert int(canvas[0, 0].max()) == 0  # top bar
    assert int(canvas[60, 0].max()) == 200  # first content row
    assert letterbox_supported_aspect(640, 480) is True
    assert letterbox_supported_aspect(1280, 720) is False


def test_sparse_video_timeline_records_missing_frames_and_gaps() -> None:
    probe_result = VideoProbe(
        duration_ns=500_000_000,
        width=640,
        height=480,
        codec_name="h264",
        pix_fmt="yuv420p",
        has_audio=False,
        format_name="mov,mp4",
    )
    timeline = FrameTimeline(pts_ns=(0, 500_000_000))
    selection = select_grid_frames(timeline)
    assert selection.missing_count >= 1
    diagnostics = _diagnostics_for(timeline, selection, probe_result)
    codes = {item["code"] for item in diagnostics}
    assert "SYNC_MISSING_FRAME" in codes
    assert "FRAME_GAP_EXCESS" in codes
    missing = next(item for item in diagnostics if item["code"] == "SYNC_MISSING_FRAME")
    assert missing["structural"] is True
    gap = next(item for item in diagnostics if item["code"] == "FRAME_GAP_EXCESS")
    assert gap["severity"] == "WARN"
    assert gap["structural"] is False
