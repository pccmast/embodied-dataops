"""DO2-08: official LeRobot 0.4.4 v2 export + separate-process Reader round-trip.

The canonical-episode/2.0 objects are produced in-memory (builder + preview
encoding), exported with the official ``LeRobotDataset`` Writer and re-opened by
the official Reader in a **fresh Python process**.  The round-trip asserts
episode/frame counts, fps, camera count+keys, state/action dimensions, task labels
and >=10 deterministic random samples with abs error <= 1e-5.

Cases: 6D front-only, 7D front-only, 6D front+wrist (1-2 cameras), and a refusal
when the canonical metadata is not publishable.
"""

from __future__ import annotations

import pytest
from _canonical_helpers import prealigned_source

from embodied_dataops.canonical.v2.build import build_canonical_v2
from embodied_dataops.canonical.v2.model import PreviewArtifact
from embodied_dataops.canonical.v2.parquet import read_parquet_bytes
from embodied_dataops.canonical.v2.preview import encode_camera_preview
from embodied_dataops.datasets.export_v2 import (
    CanonicalV2ExportEpisode,
    _expected_episodes,
    export_canonical_v2_episodes,
    verify_export_v2,
)
from embodied_dataops.datasets.lerobot_writer import tar_directory
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.storage.base import sha256_file

pytestmark = pytest.mark.contract

ROWS = 12


def _build_episode(tmp_path, *, profile_id: str, cameras: tuple[str, ...]):
    source = prealigned_source(profile_id=profile_id, cameras=cameras, rows=ROWS)
    previews: list[PreviewArtifact] = []
    preview_paths = {}
    for camera in source.cameras:
        destination = tmp_path / f"{camera.name}_preview.mp4"
        encoding = encode_camera_preview(camera, destination=destination, origin_ns=0)
        preview_paths[camera.name] = destination
        previews.append(
            PreviewArtifact(
                camera_name=camera.name,
                asset_key=f"canonical/{source.episode_id}/v2/1/preview/{camera.name}.mp4",
                asset_sha256=sha256_file(destination),
                fps=10,
                width=640,
                height=480,
                frame_count=encoding.frame_count,
            )
        )
    bundle = build_canonical_v2(source, canonical_version=1, previews=previews)
    assert bundle.publishable is True, bundle.blocking_reasons
    aligned = read_parquet_bytes(bundle.artifacts["aligned_samples"])
    return CanonicalV2ExportEpisode(
        episode_id=source.episode_id,
        metadata=bundle.metadata,
        aligned=aligned,
        preview_paths=preview_paths,
        native_indexes={"aligned": bundle.metadata.aligned_view.asset_sha256},
    )


def _export_and_verify(tmp_path, episode: CanonicalV2ExportEpisode):
    root = tmp_path / "lerobot-v3"
    result = export_canonical_v2_episodes(root, [episode], repo_id="dataops/roundtrip-v2")
    archive = tar_directory(root, tmp_path / "lerobot-v3.tar.gz")
    roundtrip = verify_export_v2(
        archive,
        repo_id=result.repo_id,
        expected=_expected_episodes([episode]),
        camera_keys=result.camera_keys,
        state_dim=result.state_dim,
        action_dim=result.action_dim,
        aligned_fps=result.aligned_fps,
        workdir=tmp_path / "verify",
    )
    return result, roundtrip


@pytest.mark.parametrize(
    ("profile_id", "cameras", "dim"),
    [
        ("arm6_joint_position_v1", ("front",), 6),
        ("arm7_joint_position_v1", ("front",), 7),
        ("arm6_joint_position_v1", ("front", "wrist"), 6),
    ],
)
def test_official_writer_export_and_reader_roundtrip(tmp_path, profile_id, cameras, dim):
    episode = _build_episode(tmp_path, profile_id=profile_id, cameras=cameras)
    result, roundtrip = _export_and_verify(tmp_path, episode)
    assert result.episode_count == 1
    assert result.total_frames == ROWS
    assert result.camera_keys == sorted(cameras)
    assert result.state_dim == dim
    assert roundtrip.episode_count == 1
    assert roundtrip.frame_count == ROWS
    assert roundtrip.state_dim == dim
    assert roundtrip.action_dim == dim
    assert roundtrip.camera_keys == sorted(cameras)
    assert roundtrip.tasks == [episode.metadata.task_label]
    assert roundtrip.samples_checked >= min(10, ROWS)
    assert roundtrip.max_abs_error <= 1e-5
    # camera features actually exist in the exported dataset
    assert set(result.features) >= {f"observation.images.{camera}" for camera in cameras}


def test_export_refuses_non_publishable_metadata(tmp_path):
    episode = _build_episode(tmp_path, profile_id="arm6_joint_position_v1", cameras=("front",))
    episode.metadata.publishable = False
    with pytest.raises(ApiError) as excinfo:
        export_canonical_v2_episodes(tmp_path / "refused", [episode], repo_id="dataops/refused")
    assert excinfo.value.code is ErrorCode.INCOMPATIBLE_EPISODE
