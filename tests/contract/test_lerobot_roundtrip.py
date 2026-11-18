"""Contract test: official LeRobot v3 Writer → official Reader round-trip (AT-17).

The export is produced by ``lerobot.datasets.lerobot_dataset.LeRobotDataset``
written by the pinned ``lerobot==0.4.4`` and re-opened by a **new Python
process**.  The reader's episode/frame counts, 6D state/action shape, task labels
and 10 random rows must match the canonical Parquet values within 1e-5.

If the official package is genuinely absent the test skips with an explicit
reason (never with ``xfail``); the service report records the blocked acceptance
item in that case.
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.canonical.contract import (
    ROBOT_JOINT_NAMES,
    ROBOT_PROFILE,
    CanonicalEpisode,
    CanonicalMetadata,
    SampleRow,
    rows_to_robot_table,
)
from embodied_dataops.datasets import lerobot_reader, lerobot_writer

pytestmark = pytest.mark.contract

FRAME_COUNT = 20
VIDEO_FPS = 10


@pytest.fixture(scope="module")
def official_lerobot() -> None:
    if not lerobot_writer.lerobot_importable():
        pytest.skip(
            "official lerobot==0.4.4 is not importable in this environment; "
            "install the `lerobot` dependency extra to exercise the AT-17 round-trip"
        )


def _write_canonical_video(path, frame_count: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(VIDEO_FPS), (640, 480)
    )
    assert writer.isOpened(), "OpenCV could not open a VideoWriter for the fixture video"
    rng = np.random.default_rng(20261029)
    for index in range(frame_count):
        frame = np.full((480, 640, 3), fill_value=(index * 7) % 255, dtype=np.uint8)
        frame[:, :, 0] = (rng.integers(0, 64, size=(480, 640), dtype=np.uint8))
        writer.write(frame)
    writer.release()


def _canonical_episode(directory, episode_id: str, frame_count: int = FRAME_COUNT) -> CanonicalEpisode:
    directory.mkdir(parents=True, exist_ok=True)
    video_path = directory / "front.mp4"
    _write_canonical_video(video_path, frame_count)
    rows = [
        SampleRow(
            sample_index=index,
            t_ns=index * SAMPLE_INTERVAL_NS,
            frame_index=index,
            state=[0.1 * index + joint for joint in range(6)],
            action=[0.2 * index - joint for joint in range(6)],
            state_source_t_ns=index * SAMPLE_INTERVAL_NS,
            action_source_t_ns=index * SAMPLE_INTERVAL_NS,
        )
        for index in range(frame_count)
    ]
    table = rows_to_robot_table(rows)
    samples_path = directory / "samples.parquet"
    pq.write_table(table, samples_path)
    import hashlib

    video_sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
    metadata = CanonicalMetadata(
        episode_id=episode_id,
        episode_type="robot_episode",
        data_origin="synthetic",
        source_type="LEROBOT_V3",
        source_asset_sha256="a" * 64,
        source_episode_index=0,
        duration_ns=(frame_count - 1) * SAMPLE_INTERVAL_NS,
        clock_domain="lerobot_episode_relative",
        joint_names=ROBOT_JOINT_NAMES,
        state_dim=6,
        action_dim=6,
        task_label="pick up the transparent cup",
        robot_profile=ROBOT_PROFILE,
        source_license="CC-BY-4.0",
        sample_count=frame_count,
        video_sha256=video_sha,
    )
    return CanonicalEpisode(metadata=metadata, samples=table, video_path=video_path)


def _expected(episodes: list[CanonicalEpisode]) -> list[lerobot_reader.ExpectedEpisode]:
    expected = []
    for episode in episodes:
        rows = episode.samples.to_pylist()
        expected.append(
            lerobot_reader.ExpectedEpisode(
                episode_id=episode.metadata.episode_id,
                task_label=episode.metadata.task_label,
                samples=[
                    lerobot_reader.ExpectedSample(
                        frame_index=int(row["frame_index"]),
                        t_ns=int(row["t_ns"]),
                        state=[float(value) for value in row["state"]],
                        action=[float(value) for value in row["action"]],
                    )
                    for row in rows
                ],
            )
        )
    return expected


def test_official_writer_export_then_independent_reader_roundtrip(official_lerobot, tmp_path) -> None:
    episodes = [
        _canonical_episode(tmp_path / "canonical-a", "00000000-0000-7000-8000-0000000000a1"),
        _canonical_episode(tmp_path / "canonical-b", "00000000-0000-7000-8000-0000000000b2"),
    ]
    repo_id = "embodied-dataops/roundtrip-contract"
    export_root = tmp_path / "export"
    result = lerobot_writer.export_canonical_episodes(export_root, episodes, repo_id=repo_id)

    assert result.episode_count == 2
    assert result.total_frames == 2 * FRAME_COUNT
    assert (export_root / "meta" / "info.json").is_file()
    assert (export_root / "meta" / "episodes").is_dir()

    archive = lerobot_writer.tar_directory(export_root, tmp_path / "lerobot-v3.tar.gz")
    assert archive.is_file() and archive.stat().st_size > 0

    verification = lerobot_reader.verify_export(
        archive,
        repo_id=repo_id,
        expected=_expected(episodes),
        timeout=300,
        workdir=tmp_path / "verify",
    )
    assert verification.episode_count == 2
    assert verification.frame_count == 2 * FRAME_COUNT
    assert verification.state_dim == 6
    assert verification.action_dim == 6
    assert verification.samples_checked == lerobot_reader.MAX_SAMPLES
    assert verification.max_abs_error <= lerobot_reader.MAX_ABS_ERROR
    assert verification.max_timestamp_error <= lerobot_reader.MAX_TIMESTAMP_ERROR
    assert verification.package_version == "0.4.4"  # official version pinned by SPEC
    assert "pick up the transparent cup" in verification.tasks


def test_tar_package_contains_official_lerobot_metadata(official_lerobot, tmp_path) -> None:
    import json
    import tarfile

    episode = _canonical_episode(tmp_path / "canonical", "00000000-0000-7000-8000-0000000000c3")
    export_root = tmp_path / "export"
    lerobot_writer.export_canonical_episodes(
        export_root, [episode], repo_id="embodied-dataops/package-check"
    )
    archive = lerobot_writer.tar_directory(export_root, tmp_path / "package.tar.gz")
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        info_member = next(name for name in names if name.endswith("meta/info.json"))
        info = json.loads(tar.extractfile(info_member).read())
    assert "meta/episodes" in "\n".join(names)
    assert info.get("fps") == VIDEO_FPS
    assert "features" in info and "observation.state" in info["features"]
