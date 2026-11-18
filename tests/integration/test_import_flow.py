"""M1 integration tests: handlers + PostgreSQL + object store (AT-02/03/06/07).

These tests call the three import handlers directly against the real database
session and a filesystem object store (same code the workers execute):

    VALIDATE_SOURCE -> IMPORT_EPISODES -> (PipelineRun) NORMALIZE_EPISODE

They assert the acceptance-relevant behaviour: no filename-based LeRobot episode
guessing, 10 Hz canonical contract sizes, ego videos never gaining robot streams,
corruption producing explicit error codes without path leakage, and structural
failures producing diagnostics-only metadata instead of silent repair.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import func, select

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from fixtures.generate_synthetic import (  # noqa: E402
    _corrupt_action_dim,
    _corrupt_timestamps,
    _tar_directory,
    generate_all,
    laplacian_variance,
)

from embodied_dataops.canonical.contract import canonical_keys  # noqa: E402
from embodied_dataops.config import get_settings  # noqa: E402
from embodied_dataops.db.models import (  # noqa: E402
    Asset,
    Episode,
    ImportJob,
    Modality,
    PipelineRun,
    SourceImport,
    Task,
)
from embodied_dataops.domain.enums import (  # noqa: E402
    EpisodeType,
    SourceType,
    Step,
)
from embodied_dataops.ids import new_id  # noqa: E402
from embodied_dataops.imports.base import EpisodeSource  # noqa: E402
from embodied_dataops.imports.lerobot_v3 import lerobot_api_available  # noqa: E402
from embodied_dataops.processing.ffmpeg import (  # noqa: E402
    ffmpeg_available,
    full_decode_check,
    iter_frames,
    probe,
    pyav_available,
)
from embodied_dataops.storage.base import sha256_file  # noqa: E402
from embodied_dataops.workflow.dag import create_import_job_tasks  # noqa: E402
from embodied_dataops.workflow.handlers.base import HandlerContext, HandlerResult  # noqa: E402
from embodied_dataops.workflow.handlers.import_episodes import ImportEpisodesHandler  # noqa: E402
from embodied_dataops.workflow.handlers.normalize_episode import (  # noqa: E402
    NormalizeEpisodeHandler,
    _draft_from_mcap,
)
from embodied_dataops.workflow.handlers.validate_source import ValidateSourceHandler  # noqa: E402

pytestmark = pytest.mark.integration

LEROBOX_AVAILABLE = lerobot_api_available()
LEROBOT_SKIP = pytest.mark.skipif(
    not LEROBOX_AVAILABLE,
    reason=(
        "official lerobot==0.4.4 API not installed; AT-02 / F01-F03 flow needs "
        "`uv sync --extra lerobot` (the fixture generator reports them BLOCKED)"
    ),
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if not (ffmpeg_available() or pyav_available()):
        raise RuntimeError(
            "no media backend available (install imageio-ffmpeg/PyAV or set FFMPEG_BIN/FFPROBE_BIN)"
        )
    out = tmp_path_factory.mktemp("flow-fixtures")
    generate_all(
        out,
        manifest_path=None,
        only=["F04_ego_video", "F07_corrupt_mp4", "F08_bad_tar", "F10_blurry_video"],
        force=True,
    )
    return {
        "F04": out / "F04_ego_video.mp4",
        "F07": out / "F07_corrupt_mp4.mp4",
        "F08": out / "F08_bad_tar.tar.gz",
        "F10": out / "F10_blurry_video.mp4",
    }


@pytest.fixture(scope="module")
def lerobot_fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not LEROBOX_AVAILABLE:
        pytest.skip("official lerobot==0.4.4 API not installed")
    out = tmp_path_factory.mktemp("flow-lerobot-fixtures")
    results = generate_all(out, manifest_path=None, only=["F01_good_lerobot"], force=True)
    result = results[0]
    if result.status != "generated":
        pytest.fail(f"official LeRobot Writer failed: {result.blocked_reason}")
    return result.path


def make_context(session, storage, task: Task, workspace: Path) -> HandlerContext:
    workspace.mkdir(parents=True, exist_ok=True)
    return HandlerContext(
        session=session,
        task=task,
        settings=get_settings(),
        storage=storage,
        workspace=workspace,
        cancel_check=lambda: False,
    )


MEDIA_TYPES = {
    SourceType.LEROBOT_V3: ("application/gzip", "dataset.tar.gz"),
    SourceType.EGO_MP4: ("video/mp4", "video.mp4"),
    SourceType.MCAP_ROS2_PROFILE_V1: ("application/octet-stream", "episode.mcap"),
}


def create_source_import(
    db,
    storage,
    path: Path,
    *,
    source_type: SourceType,
    mapping: dict | None = None,
    filename: str | None = None,
) -> tuple[SourceImport, ImportJob]:
    media_type, default_name = MEDIA_TYPES[source_type]
    sha = sha256_file(path)
    object_key = f"raw/sha256/{sha}"
    if not storage.exists(object_key):
        storage.put_file(object_key, path, content_type=media_type)
    asset = Asset(
        id=new_id(),
        sha256=sha,
        byte_size=path.stat().st_size,
        media_type=media_type,
        object_key=object_key,
        origin="raw",
        logical_name=filename or default_name,
    )
    db.add(asset)
    db.flush()
    source_import = SourceImport(
        id=new_id(),
        source_type=source_type.value,
        upload_state="UPLOADED",
        asset_id=asset.id,
        data_origin="synthetic",
        license="synthetic fixture - not real robot capture",
        filename=filename or default_name,
        size_bytes=path.stat().st_size,
        sha256=sha,
        mapping_json=mapping or {},
    )
    db.add(source_import)
    db.flush()
    job = create_import_job_tasks(db, source_import)
    db.flush()
    return source_import, job


def import_task(db, job: ImportJob, step: Step) -> Task:
    task = db.scalar(
        select(Task).where(Task.import_job_id == job.id, Task.step == step.value)
    )
    assert task is not None, f"missing import task {step.value}"
    return task


def normalize_task(db, run: PipelineRun) -> Task:
    task = db.scalar(
        select(Task).where(Task.run_id == run.id, Task.step == Step.NORMALIZE_EPISODE.value)
    )
    assert task is not None, "NORMALIZE_EPISODE task was not created by the DAG"
    return task


@dataclass
class Flow:
    source_import: SourceImport
    job: ImportJob
    validation: HandlerResult
    importer: HandlerResult | None
    episodes: list[Episode]
    runs: list[PipelineRun]
    normalizations: dict[int, HandlerResult]


def run_flow(
    db,
    storage,
    tmp_path: Path,
    path: Path,
    *,
    source_type: SourceType,
    mapping: dict | None = None,
    normalize: bool = True,
    corrupt: Any = None,
) -> Flow:
    job_dir = tmp_path / "import"
    job_dir.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        # LeRobot v3 uploads arrive as a root-directory ``.tar.gz`` (SPEC 5.1),
        # so the flow always exercises the archive + safe-extract path.
        dataset_dir = job_dir / "dataset_root"
        shutil.copytree(path, dataset_dir)
        if corrupt is not None:
            corrupt(dataset_dir)
        archive = job_dir / "dataset.tar.gz"
        _tar_directory(dataset_dir, archive, root_name=dataset_dir.name)
        path = archive
    elif corrupt is not None:
        target = job_dir / "corrupted"
        target.write_bytes(path.read_bytes())
        corrupt(target)
        path = target
    source_import, job = create_source_import(
        db, storage, path, source_type=source_type, mapping=mapping
    )
    workspace = tmp_path / "workspace-validate"
    validation = ValidateSourceHandler().handle(
        make_context(db, storage, import_task(db, job, Step.VALIDATE_SOURCE), workspace)
    )
    importer: HandlerResult | None = None
    episodes: list[Episode] = []
    runs: list[PipelineRun] = []
    normalizations: dict[int, HandlerResult] = {}
    if validation.verdict != "FAIL":
        importer = ImportEpisodesHandler().handle(
            make_context(db, storage, import_task(db, job, Step.IMPORT_EPISODES), tmp_path / "workspace-import")
        )
        episodes = list(
            db.scalars(
                select(Episode)
                .where(Episode.source_import_id == source_import.id)
                .order_by(Episode.source_episode_index)
            )
        )
        runs = list(
            db.scalars(select(PipelineRun).order_by(PipelineRun.created_at))
        )
        if normalize:
            for episode in episodes:
                run = db.scalar(select(PipelineRun).where(PipelineRun.episode_id == episode.id))
                assert run is not None
                normalizations[episode.source_episode_index] = NormalizeEpisodeHandler().handle(
                    make_context(
                        db,
                        storage,
                        normalize_task(db, run),
                        tmp_path / f"workspace-normalize-{episode.source_episode_index}",
                    )
                )
    return Flow(
        source_import=source_import,
        job=job,
        validation=validation,
        importer=importer,
        episodes=episodes,
        runs=runs,
        normalizations=normalizations,
    )


def download_bytes(storage, key: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(storage.get_bytes(key))
    return destination


def read_metadata(storage, episode: Episode) -> dict:
    payload = json.loads(storage.get_bytes(canonical_keys(episode.id, 1)["metadata"]))
    return payload


def read_samples(storage, episode: Episode, tmp_path: Path):
    import pyarrow.parquet as pq

    path = download_bytes(storage, canonical_keys(episode.id, 1)["samples"], tmp_path / "samples.parquet")
    return pq.read_table(path)


# ---------------------------------------------------------------------------
# F04 / AT-03: ego video import and 10 Hz canonical normalization
# ---------------------------------------------------------------------------


def test_f04_ego_flow_normalizes_to_100_samples_without_robot_streams(
    db, storage, tmp_path: Path, synthetic_fixtures
) -> None:
    flow = run_flow(
        db,
        storage,
        tmp_path,
        synthetic_fixtures["F04"],
        source_type=SourceType.EGO_MP4,
    )
    assert flow.validation.verdict is None
    validation_data = flow.validation.data
    assert validation_data["valid"] is True
    assert validation_data["validation"]["frame_count"] == 300
    assert validation_data["validation"]["grid_sample_count"] == 100
    assert validation_data["source"]["data_origin"] == "synthetic"

    assert flow.importer is not None and flow.importer.verdict is None
    assert flow.importer.data["episode_count"] == 1
    assert len(flow.episodes) == 1
    episode = flow.episodes[0]
    assert episode.episode_type == EpisodeType.EGO_VIDEO_EPISODE.value
    assert episode.data_origin == "synthetic"
    assert len(flow.runs) == 1
    assert flow.runs[0].episode_id == episode.id
    assert flow.job.state == "SUCCEEDED"
    assert flow.source_import.upload_state == "READY"

    result = flow.normalizations[0]
    assert result.verdict is None
    data = result.data
    assert data["normalization_valid"] is True
    assert data["episode_type"] == "ego_video_episode"
    assert data["sample_count"] == 100
    assert data["duration_ns"] == 10_000_000_000
    assert data["state_dim"] == 0
    assert data["action_dim"] == 0
    assert data["joint_names"] == []
    assert data["diagnostics"] == []
    assert data["canonical"]["samples_key"] and data["canonical"]["video_key"]

    keys = canonical_keys(episode.id, 1)
    for key in (keys["metadata"], keys["samples"], keys["video"], keys["thumbnail"]):
        assert storage.exists(key), key

    metadata = read_metadata(storage, episode)
    assert metadata["schema_version"] == "canonical-episode/1.0"
    assert metadata["episode_type"] == "ego_video_episode"
    assert metadata["data_origin"] == "synthetic"
    assert metadata["source_type"] == "EGO_MP4"
    assert metadata["target_fps"] == 10
    assert metadata["clock_domain"] == "ego_video_pts_relative"
    assert metadata["state_dim"] == 0 and metadata["action_dim"] == 0
    assert metadata["joint_names"] == []
    assert metadata["sample_count"] == 100
    assert metadata["duration_ns"] == 10_000_000_000
    assert metadata["normalization_valid"] is True
    assert metadata["video_available"] is True
    assert len(metadata["frame_transforms"]) == 100
    assert metadata["frame_transforms"][0]["src_w"] == 640
    assert metadata["frame_transforms"][0]["src_h"] == 480
    assert metadata["frame_transforms"][0]["scale"] == 1.0

    # samples.parquet: exactly the ego schema (no synthesized state/action)
    table = read_samples(storage, episode, tmp_path)
    assert table.num_rows == 100
    assert table.column_names == ["sample_index", "t_ns", "frame_index"]
    assert table.column("t_ns").to_pylist() == [index * 100_000_000 for index in range(100)]
    assert table.column("frame_index").to_pylist() == list(range(100))

    # canonical video: 10 fps 640x480 H.264, 100 decodable frames, no audio
    video_path = download_bytes(storage, keys["video"], tmp_path / "front.mp4")
    video_probe = probe(video_path)
    assert (video_probe.width, video_probe.height) == (640, 480)
    assert abs((video_probe.fps or 0) - 10.0) < 0.01
    assert video_probe.has_audio is False
    decode = full_decode_check(video_path, expected_frames=100)
    assert decode.ok, decode.stderr_tail
    assert decode.frame_count == 100

    # episode row bookkeeping
    db.refresh(episode)
    assert episode.sample_count == 100
    assert episode.duration_ns == 10_000_000_000
    assert episode.canonical_sha256 and len(episode.canonical_sha256) == 64
    assert episode.canonical_version == 1

    modalities = list(db.scalars(select(Modality).where(Modality.episode_id == episode.id)))
    assert {modality.name for modality in modalities} == {"front"}
    assert modalities[0].kind == "camera"
    assert modalities[0].sample_count == 100
    assert modalities[0].clock_domain == "ego_video_pts_relative"


def test_f04_normalization_is_idempotent_and_immutable(
    db, storage, tmp_path: Path, synthetic_fixtures
) -> None:
    flow = run_flow(
        db, storage, tmp_path, synthetic_fixtures["F04"], source_type=SourceType.EGO_MP4
    )
    episode = flow.episodes[0]
    run = db.scalar(select(PipelineRun).where(PipelineRun.episode_id == episode.id))
    task = normalize_task(db, run)
    second = NormalizeEpisodeHandler().handle(
        make_context(db, storage, task, tmp_path / "workspace-normalize-again")
    )
    first = flow.normalizations[0]
    assert second.data["canonical_sha256"] == first.data["canonical_sha256"]
    assert second.data["canonical"]["metadata_sha256"] == first.data["canonical"]["metadata_sha256"]
    metadata_assets = list(
        db.scalars(select(Asset).where(Asset.object_key == canonical_keys(episode.id, 1)["metadata"]))
    )
    assert len(metadata_assets) == 1
    # re-running the import step must not duplicate episodes or pipeline runs
    ImportEpisodesHandler().handle(
        make_context(
            db, storage, import_task(db, flow.job, Step.IMPORT_EPISODES), tmp_path / "workspace-reimport"
        )
    )
    assert db.scalar(select(func.count()).select_from(Episode)) == 1
    assert db.scalar(select(func.count()).select_from(PipelineRun)) == 1


def test_f10_blurry_video_still_normalizes_and_stays_blurry(
    db, storage, tmp_path: Path, synthetic_fixtures
) -> None:
    flow = run_flow(
        db, storage, tmp_path, synthetic_fixtures["F10"], source_type=SourceType.EGO_MP4
    )
    result = flow.normalizations[0]
    assert result.data["normalization_valid"] is True
    episode = flow.episodes[0]
    video_path = download_bytes(
        storage, canonical_keys(episode.id, 1)["video"], tmp_path / "blurry.mp4"
    )
    frames = list(iter_frames(video_path, fps=1.0))
    variances = [laplacian_variance(frame) for frame in frames]
    blurry_ratio = sum(1 for value in variances if value < 40.0) / len(variances)
    assert blurry_ratio > 0.3, variances


# ---------------------------------------------------------------------------
# F07 / F08: AT-07 error codes, evidence and no path escape
# ---------------------------------------------------------------------------


def test_f07_corrupt_video_produces_file_decode_error_without_path_leak(
    db, storage, tmp_path: Path, synthetic_fixtures
) -> None:
    flow = run_flow(
        db,
        storage,
        tmp_path,
        synthetic_fixtures["F07"],
        source_type=SourceType.EGO_MP4,
        normalize=False,
    )
    assert flow.validation.verdict == "FAIL"
    data = flow.validation.data
    assert data["error_code"] == "FILE_DECODE_ERROR"
    assert data["diagnostics"][0]["code"] == "FILE_DECODE_ERROR"
    detail_text = json.dumps(data)
    assert str(tmp_path) not in detail_text
    assert "workspace-validate" not in detail_text

    db.refresh(flow.job)
    db.refresh(flow.source_import)
    assert flow.job.state == "FAILED"
    assert flow.job.error_code == "FILE_DECODE_ERROR"
    assert flow.source_import.upload_state == "FAILED"
    assert flow.source_import.error_code == "FILE_DECODE_ERROR"
    assert flow.episodes == []
    assert db.scalar(select(func.count()).select_from(Episode)) == 0


def test_f08_malicious_tar_is_rejected_without_path_escape(
    db, storage, tmp_path: Path, synthetic_fixtures
) -> None:
    flow = run_flow(
        db,
        storage,
        tmp_path,
        synthetic_fixtures["F08"],
        source_type=SourceType.LEROBOT_V3,
        mapping={"robot_profile": "arm6_joint_position_v1"},
        normalize=False,
    )
    assert flow.validation.verdict == "FAIL"
    data = flow.validation.data
    assert data["error_code"] == "UNSAFE_ARCHIVE"
    assert data["diagnostics"][0]["details"]  # structured evidence retained
    assert flow.job.state == "FAILED"
    assert flow.episodes == []
    # nothing was written outside the extraction directory
    assert list(tmp_path.rglob("escape.txt")) == []
    assert not (tmp_path.parent / "escape.txt").exists()


# ---------------------------------------------------------------------------
# F01 / AT-02: official LeRobot v3 two-episode import
# ---------------------------------------------------------------------------


@LEROBOT_SKIP
def test_f01_lerobot_flow_imports_exactly_two_episodes_with_100_samples(
    db, storage, tmp_path: Path, lerobot_fixture_dir: Path
) -> None:
    flow = run_flow(
        db,
        storage,
        tmp_path,
        lerobot_fixture_dir,
        source_type=SourceType.LEROBOT_V3,
        mapping={"robot_profile": "arm6_joint_position_v1"},
    )
    assert flow.validation.verdict is None
    assert flow.validation.data["validation"]["episode_count_declared"] == 2
    assert flow.importer is not None and flow.importer.verdict is None
    assert flow.importer.data["episode_count"] == 2
    assert len(flow.episodes) == 2
    assert [episode.source_episode_index for episode in flow.episodes] == [0, 1]
    assert len(flow.runs) == 2
    assert flow.job.state == "SUCCEEDED"

    for episode in flow.episodes:
        result = flow.normalizations[episode.source_episode_index]
        data = result.data
        assert data["normalization_valid"] is True
        assert data["sample_count"] == 100
        assert data["state_dim"] == 6
        assert data["action_dim"] == 6
        assert data["joint_names"] == [f"joint_{index}" for index in range(1, 7)]
        assert data["diagnostics"] == []
        assert data["alignment"]["valid_sample_count"] == 100
        assert data["alignment"]["missing_sample_count"] == 0

        metadata = read_metadata(storage, episode)
        assert metadata["episode_type"] == "robot_episode"
        assert metadata["source_type"] == "LEROBOT_V3"
        assert metadata["robot_profile"] == "arm6_joint_position_v1"
        assert metadata["state_dim"] == 6 and metadata["action_dim"] == 6
        assert metadata["clock_domain"] == "lerobot_episode_relative"
        assert metadata["target_fps"] == 10
        assert metadata["provenance"]["semantics_verified"] is True

        table = read_samples(storage, episode, tmp_path)
        assert table.num_rows == 100
        assert table.column("t_ns").to_pylist() == [index * 100_000_000 for index in range(100)]
        state_type = table.schema.field("state").type
        action_type = table.schema.field("action").type
        assert state_type.list_size == 6 and action_type.list_size == 6

        video_path = download_bytes(
            storage, canonical_keys(episode.id, 1)["video"], tmp_path / f"front-{episode.source_episode_index}.mp4"
        )
        video_probe = probe(video_path)
        assert (video_probe.width, video_probe.height) == (640, 480)
        decode = full_decode_check(video_path, expected_frames=100)
        assert decode.ok, decode.stderr_tail
        assert decode.frame_count == 100


@LEROBOT_SKIP
def test_f01_samples_match_the_official_reader_values(
    db, storage, tmp_path: Path, lerobot_fixture_dir: Path
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    flow = run_flow(
        db,
        storage,
        tmp_path,
        lerobot_fixture_dir,
        source_type=SourceType.LEROBOT_V3,
        mapping={"robot_profile": "arm6_joint_position_v1"},
    )
    episode = flow.episodes[0]
    table = read_samples(storage, episode, tmp_path)
    repo_id = json.loads(
        (lerobot_fixture_dir / "meta" / "dataops_source.json").read_text(encoding="utf-8")
    )["repo_id"]
    dataset = LeRobotDataset(repo_id=repo_id, root=str(lerobot_fixture_dir), download_videos=False)
    for row_index in (0, 57, 99):
        source_row = dataset[row_index]
        expected_state = [float(value) for value in np.asarray(source_row["observation.state"]).reshape(-1)]
        expected_action = [float(value) for value in np.asarray(source_row["action"]).reshape(-1)]
        assert np.allclose(table.column("state")[row_index].as_py(), expected_state, atol=1e-6)
        assert np.allclose(table.column("action")[row_index].as_py(), expected_action, atol=1e-6)


# ---------------------------------------------------------------------------
# F02 / F03 / AT-06: structural problems keep diagnostics, never repair silently
# ---------------------------------------------------------------------------


@LEROBOT_SKIP
def test_f02_bad_timestamps_write_diagnostics_only_metadata(
    db, storage, tmp_path: Path, lerobot_fixture_dir: Path
) -> None:
    flow = run_flow(
        db,
        storage,
        tmp_path,
        lerobot_fixture_dir,
        source_type=SourceType.LEROBOT_V3,
        mapping={"robot_profile": "arm6_joint_position_v1"},
        corrupt=_corrupt_timestamps,
    )
    assert flow.validation.verdict is None
    assert flow.importer is not None and flow.importer.data["episode_count"] == 2
    episode = flow.episodes[0]
    result = flow.normalizations[0]
    data = result.data
    assert data["normalization_valid"] is False
    assert data["samples_key"] is None
    codes = {item["code"] for item in data["diagnostics"]}
    assert "TIMESTAMP_NON_MONOTONIC" in codes
    error = next(item for item in data["diagnostics"] if item["code"] == "TIMESTAMP_NON_MONOTONIC")
    assert error["severity"] == "ERROR" and error["structural"] is True
    assert error["details"]["violation_count"] >= 1

    keys = canonical_keys(episode.id, 1)
    assert storage.exists(keys["metadata"])
    assert not storage.exists(keys["samples"])
    assert not storage.exists(keys["video"])
    metadata = read_metadata(storage, episode)
    assert metadata["normalization_valid"] is False
    assert metadata["samples_parquet"] == ""
    assert metadata["video_available"] is False
    assert any(item["code"] == "TIMESTAMP_NON_MONOTONIC" for item in metadata["diagnostics"])
    db.refresh(episode)
    assert episode.sample_count == 0
    assert episode.canonical_sha256 is not None


@LEROBOT_SKIP
def test_f03_five_dimensional_action_is_diagnosed_not_truncated(
    db, storage, tmp_path: Path, lerobot_fixture_dir: Path
) -> None:
    flow = run_flow(
        db,
        storage,
        tmp_path,
        lerobot_fixture_dir,
        source_type=SourceType.LEROBOT_V3,
        mapping={"robot_profile": "arm6_joint_position_v1"},
        corrupt=_corrupt_action_dim,
    )
    assert flow.importer is not None and flow.importer.verdict is None
    episode = flow.episodes[0]
    result = flow.normalizations[0]
    data = result.data
    assert data["normalization_valid"] is False
    assert data["action_dim"] == 5
    assert data["state_dim"] == 6
    codes = {item["code"] for item in data["diagnostics"]}
    assert "DIMENSION_MISMATCH" in codes
    mismatch = next(item for item in data["diagnostics"] if item["code"] == "DIMENSION_MISMATCH")
    assert mismatch["severity"] == "ERROR"
    assert data["samples_key"] is None

    keys = canonical_keys(episode.id, 1)
    assert storage.exists(keys["metadata"])
    assert not storage.exists(keys["samples"])
    metadata = read_metadata(storage, episode)
    assert metadata["normalization_valid"] is False
    assert metadata["action_dim"] == 5
    assert metadata["state_dim"] == 6
    assert metadata["joint_names"] == []
    assert metadata["samples_parquet"] == ""


# ---------------------------------------------------------------------------
# MCAP payload -> canonical dispatch (cross-scope contract with task-3)
# ---------------------------------------------------------------------------


def _fake_mcap_payload(*, valid: bool = True) -> dict:
    frames = []
    for index in range(3):
        image = np.full((4, 6, 3), index * 60, dtype=np.uint8)
        frames.append(
            {
                "source_index": index,
                "width": 6,
                "height": 4,
                "encoding": "rgb8",
                "step": 18,
                "data": image.tobytes(),
                "log_time_ns": index * 100_000_000,
                "header_stamp_ns": index * 100_000_000,
            }
        )
    samples = [
        {
            "sample_index": index,
            "t_ns": index * 100_000_000,
            "frame_index": index,
            "state": [0.1 * index + dim for dim in range(6)],
            "action": [0.2 * index + dim for dim in range(6)],
            "state_source_t_ns": index * 100_000_000,
            "action_source_t_ns": index * 100_000_000,
            "camera_source_index": index,
        }
        for index in range(3)
    ]
    return {
        "origin_ns": 0,
        "duration_ns": 300_000_000,
        "streams": {
            "camera": {
                "topic": "/camera/front/image_raw",
                "clock_domain": "mcap_log_time",
                "frames": frames,
            },
            "state": {
                "topic": "/joint_states",
                "clock_domain": "mcap_log_time",
                "joint_names": [f"joint_{index}" for index in range(1, 7)],
                "values": [[0.0] * 6],
            },
            "action": {
                "topic": "/policy/action",
                "clock_domain": "mcap_log_time",
                "values": [[0.0] * 6],
            },
        },
        "grid": [index * 100_000_000 for index in range(3)],
        "samples": samples,
        "alignment": {
            "grid_points": 3,
            "valid_sample_count": 3,
            "missing_sample_count": 0,
            "max_abs_sync_error_ms": 0.0,
            "p95_abs_sync_error_ms": 0.0,
        },
        "diagnostics": [],
        "normalization_valid": valid,
    }


@pytest.mark.unit
def test_mcap_payload_becomes_canonical_rows_and_frames() -> None:
    source = EpisodeSource(
        source_episode_index=0,
        episode_type=EpisodeType.ROBOT_EPISODE,
        task_label="pick cup",
        payload=_fake_mcap_payload(),
    )
    draft = _draft_from_mcap(source)
    assert draft.normalization_valid is True
    assert draft.episode_type == "robot_episode"
    assert draft.state_dim == 6 and draft.action_dim == 6
    assert draft.joint_names == [f"joint_{index}" for index in range(1, 7)]
    assert draft.duration_ns == 300_000_000
    assert draft.rows is not None and len(draft.rows) == 3
    assert [row.t_ns for row in draft.rows] == [0, 100_000_000, 200_000_000]
    assert list(draft.rows[1].state) == pytest.approx([0.1, 1.1, 2.1, 3.1, 4.1, 5.1])
    assert list(draft.rows[2].action) == pytest.approx([0.4, 1.4, 2.4, 3.4, 4.4, 5.4])
    assert draft.clock_domains["camera"] == "mcap_log_time"
    frames = list(draft.frames or [])
    assert len(frames) == 3
    assert frames[0].shape == (4, 6, 3)


@pytest.mark.unit
def test_mcap_bgr_frames_are_converted_to_rgb_order() -> None:
    payload = _fake_mcap_payload()
    frame = payload["streams"]["camera"]["frames"][0]
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # blue in BGR
    frame["encoding"] = "bgr8"
    frame["data"] = bgr.tobytes()
    source = EpisodeSource(
        source_episode_index=0,
        episode_type=EpisodeType.ROBOT_EPISODE,
        payload=payload,
    )
    draft = _draft_from_mcap(source)
    first = next(iter(draft.frames or []))
    assert int(first[0, 0, 0]) == 0  # red channel is 0
    assert int(first[0, 0, 2]) == 255  # blue channel carries the BGR red=255


@pytest.mark.unit
def test_mcap_invalid_payload_writes_no_publishable_rows() -> None:
    source = EpisodeSource(
        source_episode_index=0,
        episode_type=EpisodeType.ROBOT_EPISODE,
        payload=_fake_mcap_payload(valid=False),
        diagnostics=[
            {
                "code": "SYNC_MISSING_STATE",
                "severity": "ERROR",
                "message": "joint state gap",
                "details": {"missing": 2},
                "structural": True,
            }
        ],
        normalization_valid=False,
    )
    draft = _draft_from_mcap(source)
    assert draft.normalization_valid is False
    assert draft.rows is None
    assert draft.frames is None
    assert any(item["code"] == "SYNC_MISSING_STATE" for item in draft.diagnostics)
    assert draft.alignment["missing_sample_count"] == 0  # echoed from the payload
