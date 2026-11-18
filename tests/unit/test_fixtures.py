"""Synthetic fixture generator tests (SPEC 13.1, AT-01 fixture evidence).

These tests never need a network or the official LeRobot API: the non-LeRobot
fixtures must generate offline.  The F01 test records whether the official
Writer is available (``generated``) or honestly reports ``blocked`` -- it never
fakes a LeRobot dataset with a private layout.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT))

from fixtures import DATA_ORIGIN, SEED  # noqa: E402
from fixtures.generate_synthetic import (  # noqa: E402
    OWNED_FIXTURE_IDS,
    _blurry_frame,
    _ego_frame,
    _pattern_base,
    _sha256_file,
    generate_all,
    laplacian_variance,
    read_manifest,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("synthetic-fixtures")
    manifest = out / "manifest.json"
    results = generate_all(
        out,
        manifest_path=manifest,
        only=["F04_ego_video", "F07_corrupt_mp4", "F08_bad_tar", "F10_blurry_video"],
        force=True,
    )
    return out, manifest, {result.fixture_id: result for result in results}


def test_seed_is_the_fixed_spec_seed() -> None:
    assert SEED == 20261029
    assert DATA_ORIGIN == "synthetic"


def test_unknown_seed_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_all(tmp_path, manifest_path=None, only=["F08_bad_tar"], seed=20260101)


def test_generated_fixtures_and_manifest(generated) -> None:
    out, manifest_path, results = generated
    assert set(results) == {"F04_ego_video", "F07_corrupt_mp4", "F08_bad_tar", "F10_blurry_video"}
    for fixture_id, result in results.items():
        assert result.status == "generated", (fixture_id, result.blocked_reason)
        assert result.path.is_file(), fixture_id
        assert result.sha256 and len(result.sha256) == 64
        assert result.sha256 == _sha256_file(result.path)
    manifest = read_manifest(manifest_path)
    assert manifest["schema_version"] == "fixtures-manifest/1.0"
    assert manifest["seed"] == SEED
    assert manifest["data_origin"] == "synthetic"
    entries = {entry["fixture_id"]: entry for entry in manifest["fixtures"]}
    for fixture_id in results:
        entry = entries[fixture_id]
        assert entry["data_origin"] == "synthetic"
        assert entry["seed"] == SEED
        assert entry["sha256"] == results[fixture_id].sha256
        assert entry["expected"], fixture_id
        assert entry["generated_by"] == "fixtures/generate_synthetic.py"
        assert entry["status"] == "generated"
    # every fixture records its expected outcome and the manifest never claims
    # real robot capture
    assert "ego_video_episode" in entries["F04_ego_video"]["expected"]["outcome"]
    assert entries["F07_corrupt_mp4"]["expected"]["error_code"] == "FILE_DECODE_ERROR"
    assert entries["F08_bad_tar"]["expected"]["error_code"] == "UNSAFE_ARCHIVE"
    assert entries["F10_blurry_video"]["expected"]["quality_rule"] == "BLUR_FRAME_RATE"


def test_ego_and_blur_fixtures_are_visually_distinct(generated) -> None:
    _out, _manifest, results = generated
    sharp = results["F04_ego_video"].extra["sharpness_estimate"]
    blurry = results["F10_blurry_video"].extra["blur_estimate"]
    assert sharp["sampled_frames"] >= 5
    assert sharp["blurry_ratio"] == 0.0
    assert sharp["mean_laplacian_variance"] > 40
    assert blurry["blurry_ratio"] >= 0.9
    assert blurry["mean_laplacian_variance"] < 40
    # frame helpers stay deterministic for the fixed seed
    base = _pattern_base(640, 480, SEED)
    first = _ego_frame(base, 3)
    assert first.shape == (480, 640, 3)
    assert first.dtype.name == "uint8"
    assert laplacian_variance(_blurry_frame(0)) < 40


def test_corrupt_mp4_is_a_truncated_copy(generated) -> None:
    _out, _manifest, results = generated
    source = results["F04_ego_video"].path.read_bytes()
    corrupt = results["F07_corrupt_mp4"].path.read_bytes()
    assert len(corrupt) < len(source)
    assert source.startswith(corrupt)
    assert results["F07_corrupt_mp4"].sha256 != results["F04_ego_video"].sha256


def test_bad_tar_contains_path_traversal_member(generated) -> None:
    _out, _manifest, results = generated
    with tarfile.open(results["F08_bad_tar"].path, mode="r:gz") as archive:
        names = archive.getnames()
    assert "../escape.txt" in names
    assert any(name.endswith("meta/info.json") for name in names)


def test_manifest_merge_preserves_foreign_entries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    foreign = {
        "schema_version": "fixtures-manifest/1.0",
        "seed": SEED,
        "data_origin": "synthetic",
        "fixtures": [
            {
                "fixture_id": "F05_mcap_good",
                "path": "fixtures/golden/mcap/F05_mcap_good.mcap",
                "sha256": "a" * 64,
                "data_origin": "synthetic",
                "seed": SEED,
                "expected": {"outcome": "MCAP import"},
                "generated_by": "fixtures/make_mcap_ros2.py",
                "status": "generated",
            }
        ],
    }
    manifest_path.write_text(json.dumps(foreign), encoding="utf-8")
    generate_all(
        tmp_path,
        manifest_path=manifest_path,
        only=["F08_bad_tar"],
        force=True,
    )
    manifest = read_manifest(manifest_path)
    fixture_ids = [entry["fixture_id"] for entry in manifest["fixtures"]]
    assert "F05_mcap_good" in fixture_ids
    assert "F08_bad_tar" in fixture_ids
    foreign_entry = next(entry for entry in manifest["fixtures"] if entry["fixture_id"] == "F05_mcap_good")
    assert foreign_entry["sha256"] == "a" * 64


def test_owned_fixture_ids_cover_spec_f01_to_f11() -> None:
    assert set(OWNED_FIXTURE_IDS) == {
        "F01_good_lerobot",
        "F02_bad_timestamps",
        "F03_bad_dims",
        "F04_ego_video",
        "F07_corrupt_mp4",
        "F08_bad_tar",
        "F10_blurry_video",
        "F11_three_group",
    }


def test_f01_is_generated_by_the_official_writer_or_reported_blocked(tmp_path: Path) -> None:
    results = generate_all(
        tmp_path,
        manifest_path=tmp_path / "manifest.json",
        only=["F01_good_lerobot"],
        force=True,
    )
    [result] = results
    assert result.status in {"generated", "blocked"}
    if result.status == "blocked":
        # Honest environment report: the pinned official extra is missing.  The
        # importer code path is still implemented and tested in
        # tests/unit/test_import_lerobot.py once the API exists.
        assert result.blocked_reason
        assert "lerobot" in result.blocked_reason.lower()
        return
    root = result.path
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["fps"] == 10
    assert info["features"]["observation.state"]["shape"] == [6]
    assert info["features"]["action"]["shape"] == [6]
    assert info.get("total_episodes", 2) == 2
    assert info["codebase_version"].startswith("v3")
    sidecar = json.loads((root / "meta" / "dataops_semantics.json").read_text(encoding="utf-8"))
    assert sidecar["robot_profile"] == "arm6_joint_position_v1"
    assert sidecar["data_origin"] == "synthetic"
    manifest = read_manifest(tmp_path / "manifest.json")
    entry = next(item for item in manifest["fixtures"] if item["fixture_id"] == "F01_good_lerobot")
    assert entry["data_origin"] == "synthetic"
    assert entry["status"] == "generated"


def test_f11_three_group_builds_three_independent_splits_or_reports_blocked(tmp_path: Path) -> None:
    """§6.1 的三组正常路径：每组一个 episode，显式分到 train/val/test。"""
    results = generate_all(
        tmp_path,
        manifest_path=tmp_path / "manifest.json",
        only=["F11_three_group"],
        force=True,
    )
    [result] = results
    assert result.status in {"generated", "blocked"}
    if result.status == "blocked":
        assert result.blocked_reason
        assert "lerobot" in result.blocked_reason.lower()
        return
    root = result.path
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 360
    assert info["fps"] == 10
    groups = json.loads((root / "source_groups.json").read_text(encoding="utf-8"))["groups"]
    assert [group["group_id"] for group in groups] == ["g_0", "g_1", "g_2"]
    assert [group["episode_ids"] for group in groups] == [["episode_0"], ["episode_1"], ["episode_2"]]
    assignments = json.loads((root / "group_assignments.json").read_text(encoding="utf-8"))
    assert assignments["assignments"] == {"g_0": "train", "g_1": "val", "g_2": "test"}


def test_second_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = generate_all(tmp_path / "first", manifest_path=None, only=["F10_blurry_video"], force=True)
    second = generate_all(tmp_path / "second", manifest_path=None, only=["F10_blurry_video"], force=True)
    assert first[0].sha256 == second[0].sha256
