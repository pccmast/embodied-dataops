"""LeRobot v3 adapter tests (SPEC 5.3, AT-02, AT-06 across F01/F02/F03).

The pure mapping/metadata/archive rules always run.  The official-API tests are
activated as soon as the pinned ``lerobot==0.4.4`` extra is importable; they are
skipped with an explicit reason otherwise (never ``xfail``), and the fixture
generator reports F01 ``blocked`` in that environment.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from fixtures.generate_synthetic import (  # noqa: E402
    _corrupt_action_dim,
    _corrupt_timestamps,
    _tar_directory,
    generate_all,
)

from embodied_dataops.domain.enums import DataOrigin, EpisodeType, SourceType  # noqa: E402
from embodied_dataops.errors import ApiError, ErrorCode  # noqa: E402
from embodied_dataops.imports.base import ImportContext  # noqa: E402
from embodied_dataops.imports.lerobot_v3 import (  # noqa: E402
    LerobotV3Parser,
    _import_lerobot_dataset_class,
    _select_rows,
    declared_codebase_major,
    inspect_lerobot_archive,
    lerobot_api_available,
    normalize_episode_rows,
    parse_info_payload,
    resolve_dataset_root,
    resolve_mapping,
    semantics_evidence_from,
    snap_lerobot_timestamps,
    to_float_list,
    to_rgb_uint8,
)
from embodied_dataops.processing.files import safe_extract_tar  # noqa: E402

pytestmark = pytest.mark.unit

LEROBOX_AVAILABLE = lerobot_api_available()
LEROBOT_SKIP = pytest.mark.skipif(
    not LEROBOX_AVAILABLE,
    reason=(
        "official lerobot==0.4.4 API not installed; run `uv sync --extra lerobot` "
        "(fixtures F01/F02/F03 report BLOCKED without it)"
    ),
)

ROBOT_PROFILE = "arm6_joint_position_v1"


# ---------------------------------------------------------------------------
# pure metadata / mapping rules
# ---------------------------------------------------------------------------


def make_info(**overrides):
    payload = {
        "codebase_version": "v3.0",
        "fps": 10,
        "robot_type": "aloha",
        "total_episodes": 2,
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [3, 480, 640]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        },
    }
    payload.update(overrides)
    return payload


def test_declared_codebase_major() -> None:
    assert declared_codebase_major("v3.0") == 3
    assert declared_codebase_major("3.1") == 3
    assert declared_codebase_major("v2.1") == 2
    assert declared_codebase_major(3) == 3
    assert declared_codebase_major(None) is None
    assert declared_codebase_major(True) is None


def test_parse_info_accepts_v3_declarations() -> None:
    for version in ("v3.0", "3.0", "v3"):
        info = parse_info_payload(make_info(codebase_version=version))
        assert info.codebase_version == version
        assert info.fps == 10.0
        assert info.camera_keys == ["observation.images.front"]
        assert info.feature_dim("observation.state") == 6
        assert info.feature_dim("action") == 6
        assert info.as_summary()["total_episodes"] == 2


def test_parse_info_rejects_non_v3_and_malformed() -> None:
    with pytest.raises(ApiError) as excinfo:
        parse_info_payload(make_info(codebase_version="v2.1"))
    assert excinfo.value.code is ErrorCode.LEROBOT_METADATA_INVALID
    assert excinfo.value.details["declared"] == "v2.1"

    with pytest.raises(ApiError):
        parse_info_payload({})
    with pytest.raises(ApiError):
        parse_info_payload(make_info(features={}))
    with pytest.raises(ApiError):
        parse_info_payload(make_info(fps=0))
    with pytest.raises(ApiError):
        parse_info_payload(make_info(fps="fast"))


def test_mapping_auto_selects_the_only_camera() -> None:
    info = parse_info_payload(make_info())
    resolved = resolve_mapping(info, {"robot_profile": ROBOT_PROFILE})
    assert resolved.camera_key == "observation.images.front"
    assert resolved.auto_selected_camera is True
    assert resolved.state_key == "observation.state"
    assert resolved.action_key == "action"
    assert resolved.state_dim == 6 and resolved.action_dim == 6
    assert resolved.semantics_verified is False


def test_mapping_requires_explicit_camera_when_multiple() -> None:
    payload = make_info()
    payload["features"]["observation.images.wrist"] = {"dtype": "video", "shape": [3, 480, 640]}
    info = parse_info_payload(payload)
    with pytest.raises(ApiError) as excinfo:
        resolve_mapping(info, {"robot_profile": ROBOT_PROFILE})
    assert excinfo.value.code is ErrorCode.MAPPING_REQUIRED
    assert excinfo.value.details["available_cameras"] == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    resolved = resolve_mapping(
        info, {"camera_key": "observation.images.wrist", "robot_profile": ROBOT_PROFILE}
    )
    assert resolved.camera_key == "observation.images.wrist"
    assert resolved.auto_selected_camera is False


def test_mapping_rejects_unknown_feature_keys() -> None:
    info = parse_info_payload(make_info())
    for mapping in (
        {"camera_key": "observation.images.nope", "robot_profile": ROBOT_PROFILE},
        {"state_key": "observation.nope", "robot_profile": ROBOT_PROFILE},
        {"action_key": "nope", "robot_profile": ROBOT_PROFILE},
    ):
        with pytest.raises(ApiError) as excinfo:
            resolve_mapping(info, mapping)
        assert excinfo.value.code is ErrorCode.INVALID_MAPPING


def test_mapping_requires_the_supported_robot_profile() -> None:
    info = parse_info_payload(make_info())
    with pytest.raises(ApiError) as excinfo:
        resolve_mapping(info, {})
    assert excinfo.value.code is ErrorCode.MAPPING_REQUIRED
    with pytest.raises(ApiError) as excinfo:
        resolve_mapping(info, {"robot_profile": "arm7_torque_v2"})
    assert excinfo.value.code is ErrorCode.INVALID_MAPPING


def test_semantics_evidence_comes_from_provenance_not_guessing() -> None:
    valid = {
        "robot_profile": ROBOT_PROFILE,
        "state_units": "rad",
        "action_semantics": "target_joint_position",
        "evidence": "public dataset card documents joint positions in radians",
    }
    assert semantics_evidence_from(valid) is not None
    assert semantics_evidence_from({**valid, "state_units": "degrees"}) is None
    assert semantics_evidence_from({**valid, "action_semantics": "velocity"}) is None
    assert semantics_evidence_from({**valid, "evidence": ""}) is None
    assert semantics_evidence_from({"robot_profile": "other", **valid}) is not None

    info = parse_info_payload(make_info())
    verified = resolve_mapping(info, {"robot_profile": ROBOT_PROFILE}, semantics_sidecar=valid)
    assert verified.semantics_verified is True
    assert verified.semantics_evidence

    fallback = resolve_mapping(
        info,
        {
            "robot_profile": ROBOT_PROFILE,
            "semantics_evidence": "Hub dataset card: action = target joint position (rad)",
        },
    )
    assert fallback.semantics_verified is True


def test_normalize_episode_rows_uses_official_metadata_columns() -> None:
    rows = [
        {
            "episode_index": 1,
            "length": 100,
            "tasks": ["place cup"],
            "dataset_from_index": 100,
            "dataset_to_index": 200,
        },
        {
            "episode_index": 0,
            "length": 100,
            "tasks": ["pick cup"],
            "dataset_from_index": 0,
            "dataset_to_index": 100,
        },
    ]
    episodes = normalize_episode_rows(rows)
    assert [episode.episode_index for episode in episodes] == [0, 1]
    assert episodes[0].task_label == "pick cup"
    assert episodes[0].from_index == 0 and episodes[0].to_index == 100
    with pytest.raises(ApiError) as excinfo:
        normalize_episode_rows([{"episode_index": 0, "tasks": []}])
    assert excinfo.value.code is ErrorCode.EPISODE_ENUMERATION_FAILED


def test_alignment_uses_nearest_complete_row_without_future_action() -> None:
    # exact 10 Hz rows: 100 frames -> 100 grid points, t = k * 100 ms
    timestamps = [index * 100_000_000 for index in range(100)]
    selection = _select_rows(timestamps)
    assert selection["alignment"]["valid_sample_count"] == 100
    assert selection["alignment"]["missing_sample_count"] == 0
    assert selection["source_indexes"] == list(range(100))
    assert selection["alignment"]["max_abs_sync_error_ms"] == 0.0

    # when a past row and a future row are both inside the 50 ms tolerance, the
    # past row is preferred so action_source_t_ns never lies in the future
    past_preferred = _select_rows([0, 60_000_000, 150_000_000])
    assert past_preferred["source_indexes"][:2] == [0, 1]
    assert past_preferred["alignment"]["selected_after_target_count"] == 0
    future_used = _select_rows([0, 120_000_000])
    assert future_used["source_indexes"][:2] == [0, 1]
    assert future_used["alignment"]["selected_after_target_count"] >= 1

    # a grid point with no row within 50 ms is "missing", not zero-filled
    sparse = _select_rows([0, 300_000_000])
    assert sparse["alignment"]["missing_sample_count"] >= 1


def test_snap_lerobot_timestamps_recovers_the_exact_10hz_grid() -> None:
    # the official Writer stores frame_index/fps as float32; 9.9 becomes
    # 9.899999618..., which must still produce the 100th 10 Hz sample.
    float32_times = [float(np.float32(index / 10.0)) for index in range(100)]
    snapped = snap_lerobot_timestamps(float32_times, 10.0)
    assert snapped == [index * 100_000_000 for index in range(100)]
    # genuinely off-grid timestamps are preserved (no silent re-timing)
    assert snap_lerobot_timestamps([0.0, 0.15], 10.0) == [0, 150_000_000]
    # a duplicated corrupt timestamp stays duplicated so the quality rule fires
    corrupt = list(float32_times)
    corrupt[50] = corrupt[49]
    snapped_corrupt = snap_lerobot_timestamps(corrupt, 10.0)
    assert snapped_corrupt[50] == snapped_corrupt[49]


def test_tensor_and_image_conversion_helpers() -> None:
    assert to_float_list(np.array([1.5, -2.0], dtype=np.float32)) == [1.5, -2.0]
    assert to_float_list([[1.0, 2.0]]) == [1.0, 2.0]
    assert to_float_list(None) == []

    chw_uint8 = np.zeros((3, 4, 6), dtype=np.uint8)
    chw_uint8[0] = 255
    rgb = to_rgb_uint8(chw_uint8)
    assert rgb.shape == (4, 6, 3)
    assert rgb[0, 0, 0] == 255
    assert rgb[0, 0, 1] == 0

    float_chw = np.ones((3, 2, 2), dtype=np.float32)
    converted = to_rgb_uint8(float_chw)
    assert converted.dtype == np.uint8
    assert int(converted[0, 0, 0]) in {254, 255}

    grayscale = np.full((2, 2), 128, dtype=np.uint8)
    assert to_rgb_uint8(grayscale).shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# archive safety + dataset-root rules (F08, adversarial inputs)
# ---------------------------------------------------------------------------


def test_archive_preflight_rejects_path_traversal(tmp_path: Path) -> None:
    generate_all(tmp_path, manifest_path=None, only=["F08_bad_tar"], force=True)
    with pytest.raises(ApiError) as excinfo:
        inspect_lerobot_archive(tmp_path / "F08_bad_tar.tar.gz")
    assert excinfo.value.code is ErrorCode.UNSAFE_ARCHIVE
    # nothing was extracted by the preflight
    assert not (tmp_path / "escape.txt").exists()
    with pytest.raises(ApiError) as excinfo:
        safe_extract_tar(tmp_path / "F08_bad_tar.tar.gz", tmp_path / "extract")
    assert excinfo.value.code is ErrorCode.UNSAFE_ARCHIVE
    assert not (tmp_path / "escape.txt").exists()


def test_archive_preflight_rejects_non_gzip(tmp_path: Path) -> None:
    fake = tmp_path / "not-an-archive.tar.gz"
    fake.write_bytes(b"plain text, not gzip")
    with pytest.raises(ApiError) as excinfo:
        inspect_lerobot_archive(fake)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_ARCHIVE_FORMAT


def _write_tar_gz(
    path: Path, members: list[tuple[tarfile.TarInfo, bytes | None, str | None]]
) -> None:
    import gzip
    import io

    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for info, payload, linkname in members:
                    if linkname is not None:
                        info.type = tarfile.SYMTYPE
                        info.linkname = linkname
                        archive.addfile(info)
                        continue
                    assert payload is not None
                    info.size = len(payload)
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(payload))


def test_archive_preflight_rejects_symlinks(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.tar.gz"
    _write_tar_gz(
        archive_path,
        [
            (
                tarfile.TarInfo("dataset/meta/info.json"),
                b'{"codebase_version": "v3.0", "fps": 10, "features": {"action": {}}}',
                None,
            ),
            (tarfile.TarInfo("dataset/link"), None, "/etc/passwd"),
        ],
    )
    with pytest.raises(ApiError) as excinfo:
        inspect_lerobot_archive(archive_path)
    assert excinfo.value.code is ErrorCode.UNSAFE_ARCHIVE
    assert excinfo.value.details["rule"] == "link"


def test_archive_preflight_enforces_entry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodied_dataops.config import get_settings

    monkeypatch.setattr(get_settings(), "max_tar_entries", 1)
    archive_path = tmp_path / "many.tar.gz"
    _write_tar_gz(
        archive_path,
        [
            (tarfile.TarInfo("dataset/meta/info.json"), b"{}", None),
            (tarfile.TarInfo("dataset/data/chunk-000/file-000.parquet"), b"x", None),
        ],
    )
    with pytest.raises(ApiError) as excinfo:
        inspect_lerobot_archive(archive_path)
    assert excinfo.value.code is ErrorCode.ARCHIVE_TOO_LARGE


def test_resolve_dataset_root_handles_wrapper_directory(tmp_path: Path) -> None:
    root = tmp_path / "direct"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    assert resolve_dataset_root(root) == root

    wrapped = tmp_path / "wrapped"
    inner = wrapped / "F01_good_lerobot"
    (inner / "meta").mkdir(parents=True)
    (inner / "meta" / "info.json").write_text("{}", encoding="utf-8")
    assert resolve_dataset_root(wrapped) == inner

    ambiguous = tmp_path / "ambiguous"
    for name in ("a", "b"):
        (ambiguous / name / "meta").mkdir(parents=True)
        (ambiguous / name / "meta" / "info.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ApiError) as excinfo:
        resolve_dataset_root(ambiguous)
    assert excinfo.value.code is ErrorCode.LEROBOT_METADATA_INVALID


def test_official_api_unavailable_error_is_actionable() -> None:
    if LEROBOX_AVAILABLE:
        assert lerobot_api_available() is True
        assert _import_lerobot_dataset_class() is not None
        return
    with pytest.raises(ApiError) as excinfo:
        _import_lerobot_dataset_class()
    assert excinfo.value.code is ErrorCode.LEROBOT_API_UNAVAILABLE
    assert "lerobot==0.4.4" in excinfo.value.message
    assert lerobot_api_available() is False


# ---------------------------------------------------------------------------
# official-API tests against the generated fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def f01_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("lerobot-fixtures")
    results = generate_all(
        out, manifest_path=None, only=["F01_good_lerobot"], force=True
    )
    result = results[0]
    if result.status != "generated":
        if not LEROBOX_AVAILABLE:
            pytest.skip(
                "F01_good_lerobot is BLOCKED: official lerobot==0.4.4 Writer is not installed"
            )
        pytest.fail(f"official LeRobot Writer failed while the API is importable: {result.blocked_reason}")
    return result.path


def make_lerobot_context(
    source: Path, workdir: Path, *, mapping: dict | None = None
) -> ImportContext:
    workdir.mkdir(parents=True, exist_ok=True)
    return ImportContext(
        import_id="test-lerobot-import",
        source_type=SourceType.LEROBOT_V3,
        data_origin=DataOrigin.SYNTHETIC,
        source_path=source,
        workdir=workdir,
        mapping=mapping if mapping is not None else {"robot_profile": ROBOT_PROFILE},
        license="synthetic fixture",
    )


@LEROBOT_SKIP
def test_f01_validate_reports_v3_metadata(f01_dataset: Path, tmp_path: Path) -> None:
    parser = LerobotV3Parser()
    context = make_lerobot_context(f01_dataset, tmp_path / "work")
    validation = parser.validate(context)
    assert validation["format"] == "lerobot-v3"
    assert validation["info"]["codebase_version"].startswith("v3")
    assert validation["info"]["fps"] == 10.0
    assert validation["episode_count_declared"] == 2
    assert validation["resolved_mapping"]["camera_key"] == "observation.images.front"
    assert validation["resolved_mapping"]["semantics_verified"] is True


@LEROBOT_SKIP
def test_f01_archive_validate_uses_only_metadata(f01_dataset: Path, tmp_path: Path) -> None:
    archive = tmp_path / "F01_good_lerobot.tar.gz"
    _tar_directory(f01_dataset, archive, root_name="F01_good_lerobot")
    parser = LerobotV3Parser()
    context = make_lerobot_context(archive, tmp_path / "work")
    validation = parser.validate(context)
    assert validation["archive"]["kind"] == "tar.gz"
    assert validation["episode_count_declared"] == 2
    assert validation["api_available"] is True


@LEROBOT_SKIP
def test_f01_parse_enumerates_exactly_two_episodes_at_10hz(f01_dataset: Path, tmp_path: Path) -> None:
    parser = LerobotV3Parser()
    context = make_lerobot_context(f01_dataset, tmp_path / "work")
    parsed = parser.parse(context)
    assert len(parsed.episodes) == 2
    assert [episode.source_episode_index for episode in parsed.episodes] == [0, 1]
    for episode in parsed.episodes:
        assert episode.episode_type == EpisodeType.ROBOT_EPISODE
        assert episode.payload["length"] == 100
        assert episode.payload["source_fps"] == 10
        assert episode.duration_ns == 10_000_000_000
        assert episode.normalization_valid is True
        dimensions = {
            item["code"] for item in episode.diagnostics
        }
        assert "DIMENSION_MISMATCH" not in dimensions
    assert parsed.metadata["enumerated_by"] == "official_lerobot_api"


@LEROBOT_SKIP
def test_f01_read_aligned_episode_matches_official_rows(f01_dataset: Path, tmp_path: Path) -> None:
    parser = LerobotV3Parser()
    context = make_lerobot_context(f01_dataset, tmp_path / "work")
    aligned = parser.read_aligned_episode(context, 0)
    assert aligned.normalization_valid is True
    assert aligned.rows is not None
    assert len(aligned.rows) == 100
    assert aligned.state_dim == 6 == aligned.action_dim
    assert aligned.joint_names and len(aligned.joint_names) == 6
    assert [row.t_ns for row in aligned.rows] == [index * 100_000_000 for index in range(100)]
    assert aligned.duration_ns == 10_000_000_000
    assert aligned.selected_source_indexes == list(range(100))
    assert aligned.diagnostics == []
    assert aligned.semantics_verified is True

    # cross-check two rows against the official API in the same process
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = json.loads(
        (f01_dataset / "meta" / "dataops_source.json").read_text(encoding="utf-8")
    )["repo_id"]
    dataset = LeRobotDataset(repo_id=repo_id, root=str(f01_dataset), download_videos=False)
    for grid_index in (0, 57, 99):
        source_index = aligned.selected_source_indexes[grid_index]
        row = dataset[source_index]
        expected_state = to_float_list(row["observation.state"])
        expected_action = to_float_list(row["action"])
        assert np.allclose(aligned.rows[grid_index].state, expected_state, atol=1e-6)
        assert np.allclose(aligned.rows[grid_index].action, expected_action, atol=1e-6)


@LEROBOT_SKIP
def test_f02_nonmonotonic_timestamps_produce_error_evidence(
    f01_dataset: Path, tmp_path: Path
) -> None:
    corrupt = tmp_path / "F02_bad_timestamps"
    import shutil

    shutil.copytree(f01_dataset, corrupt)
    _corrupt_timestamps(corrupt)
    parser = LerobotV3Parser()
    context = make_lerobot_context(corrupt, tmp_path / "work")
    aligned = parser.read_aligned_episode(context, 0)
    codes = {item["code"] for item in aligned.diagnostics}
    assert "TIMESTAMP_NON_MONOTONIC" in codes
    assert aligned.rows is None
    assert aligned.normalization_valid is False
    violation = next(
        item for item in aligned.diagnostics if item["code"] == "TIMESTAMP_NON_MONOTONIC"
    )
    assert violation["severity"] == "ERROR"
    assert violation["structural"] is True
    assert violation["details"]["violation_count"] >= 1


@LEROBOT_SKIP
def test_f03_five_dimensional_action_is_reported_not_truncated(
    f01_dataset: Path, tmp_path: Path
) -> None:
    corrupt = tmp_path / "F03_bad_dims"
    import shutil

    shutil.copytree(f01_dataset, corrupt)
    _corrupt_action_dim(corrupt)
    parser = LerobotV3Parser()
    context = make_lerobot_context(corrupt, tmp_path / "work")
    parsed = parser.parse(context)
    assert len(parsed.episodes) == 2
    assert any(item["code"] == "DIMENSION_MISMATCH" for item in parsed.episodes[0].diagnostics)
    aligned = parser.read_aligned_episode(context, 0)
    assert aligned.action_dim == 5
    assert aligned.state_dim == 6
    assert aligned.rows is None
    assert aligned.normalization_valid is False
    mismatch = next(
        item for item in aligned.diagnostics if item["code"] == "DIMENSION_MISMATCH"
    )
    assert mismatch["details"]["required"] == 6
    assert aligned.joint_names == []
