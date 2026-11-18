"""TrainingBundle 准备与原子发布（§11）。

这里的断言全部围绕"身份"和"不覆盖"两条：同一输入必须得到同一语义身份，
任何字节改动都必须被逐文件验出来，不同内容绝不能盖掉已发布产物。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

# The bundle preparation path goes through the official LeRobot API. CI installs the
# `dev` and `ui` extras only (torch/lerobot are heavy), so this module is skipped there
# instead of failing — consistent with the `ros2`/`vision` marker policy in CONTRIBUTING.md.
pytest.importorskip("lerobot", reason="training bundle preparation requires lerobot==0.4.4")

from embodied_dataops.errors import ApiError, ErrorCode  # noqa: E402
from embodied_dataops.training.artifact import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    _manifest_action_semantics,
    digest_paths,
    load_bundle,
    load_source_package,
    prepare_bundle,
    snapshot_source_document,
    verify_source_root,
)
from embodied_dataops.training.contract import (  # noqa: E402
    canonical_json_bytes,
    parse_recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "fixtures" / "F01_good_lerobot"


pytestmark = [
    pytest.mark.unit,
    pytest.mark.no_db,
    pytest.mark.skipif(not FIXTURE_ROOT.is_dir(), reason="F01 fixture is not generated"),
]


def _recipe_document(*, chunk_size: int = 4, stride: int = 2) -> dict:
    return {
        "target": {},
        "features": {},
        "temporal": {
            "chunk_size": chunk_size,
            "n_action_steps": chunk_size,
            "stride": stride,
        },
        "split": {
            "strategy": "explicit_group_assignment_v1",
            "group_mapping": "source_groups.json",
            "assignments": "group_assignments.json",
            "require_nonempty": ["train"],
        },
        "normalization": {"modes": {"VISUAL": "IDENTITY"}, "epsilon": 1e-6},
        "validation": {"device": "cpu", "batch_size": 1, "optimizer_steps": 1},
    }


def _inputs(tmp_path: Path, *, chunk_size: int = 4) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = snapshot_source_document(
        dataset_root=FIXTURE_ROOT,
        dataset_version_id="fixture:f01",
        license_id="synthetic-fixture",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    groups = {
        "schema_version": "training-source-groups/1.0",
        "groups": [
            {"group_id": "g_train", "episode_ids": ["episode_0"], "evidence": {}},
            {"group_id": "g_val", "episode_ids": ["episode_1"], "evidence": {}},
        ],
    }
    (tmp_path / "source_groups.json").write_text(json.dumps(groups), encoding="utf-8")
    assignments = {
        "schema_version": "training-group-assignments/1.0",
        "assignments": {"g_train": "train", "g_val": "val"},
    }
    (tmp_path / "group_assignments.json").write_text(json.dumps(assignments), encoding="utf-8")

    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(_recipe_document(chunk_size=chunk_size)), encoding="utf-8")
    return {
        "manifest": manifest_path,
        "recipe": recipe_path,
        "base": tmp_path,
    }


def _source(inputs: dict[str, Path], *, digest_files: bool = False):
    return load_source_package(
        source_root=FIXTURE_ROOT,
        manifest_path=inputs["manifest"],
        source_mode="synthetic-fixture",
        digest_files=digest_files,
    )


def _prepare(inputs: dict[str, Path], output: Path, **kwargs):
    recipe = parse_recipe(json.loads(inputs["recipe"].read_text(encoding="utf-8")))
    return prepare_bundle(
        recipe=recipe,
        source=_source(inputs, digest_files=kwargs.pop("digest_files", False)),
        output_dir=output,
        report_dir=kwargs.pop("report_dir", output.parent / f"{output.name}-reports"),
        check_video=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 正例
# ---------------------------------------------------------------------------
def test_a_prepared_bundle_contains_the_contract_files(tmp_path):
    inputs = _inputs(tmp_path)
    prepared = _prepare(inputs, tmp_path / "bundle")

    expected = {
        "manifest.json",
        "artifact.json",
        "resolved_recipe.json",
        "splits.json",
        "source_groups.json",
        "group_assignments.json",
        "windows.parquet",
        "rejected_windows.parquet",
        "train_stats.json",
        "provenance.json",
        "reports/preparation.json",
        "processors/pre.json",
        "processors/post.json",
    }
    present = {path.relative_to(prepared.path).as_posix() for path in prepared.path.rglob("*") if path.is_file()}
    assert expected <= present

    manifest = json.loads((prepared.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert manifest["preparation_id"] == prepared.preparation_id
    assert manifest["artifact_id"] == prepared.artifact_id
    assert manifest["verification"]["level"] == "PREPARED"
    assert manifest["verification"]["source_verification"] == "FIXTURE_ONLY"


def test_statistics_are_fitted_on_train_frames_only(tmp_path):
    inputs = _inputs(tmp_path)
    prepared = _prepare(inputs, tmp_path / "bundle")

    stats = json.loads((prepared.path / "train_stats.json").read_text(encoding="utf-8"))
    assert stats["stats_scope"] == "train_only"
    # F01: 每个 episode 100 帧；train 只有 episode_0，动作覆盖 0..99，观察锚点 49 个。
    assert stats["unique_state_frames"] == 49
    assert stats["unique_action_frames"] == 100


def test_windows_are_split_scoped_and_deduplicated(tmp_path):
    inputs = _inputs(tmp_path)
    prepared = _prepare(inputs, tmp_path / "bundle")
    loaded = load_bundle(prepared.path)

    train = loaded.windows_for("train")
    val = loaded.windows_for("val")
    assert len(train) == 49
    assert len(val) == 49
    assert {window.split for window in train} == {"train"}
    assert all(window.official_episode_index == 0 for window in train)
    assert all(window.official_episode_index == 1 for window in val)


def test_the_two_identities_are_reproducible_across_runs(tmp_path):
    first = _prepare(_inputs(tmp_path / "a"), tmp_path / "a" / "bundle")
    second = _prepare(_inputs(tmp_path / "b"), tmp_path / "b" / "bundle")
    assert first.preparation_id == second.preparation_id
    assert first.artifact_id == second.artifact_id


def test_the_same_content_is_reused_without_rewriting(tmp_path):
    inputs = _inputs(tmp_path)
    first = _prepare(inputs, tmp_path / "bundle")
    second = _prepare(inputs, tmp_path / "bundle")
    assert second.reused is True
    assert second.artifact_id == first.artifact_id


# ---------------------------------------------------------------------------
# 负例
# ---------------------------------------------------------------------------
def test_different_content_never_overwrites_a_published_bundle(tmp_path):
    _prepare(_inputs(tmp_path / "a", chunk_size=4), tmp_path / "bundle")
    with pytest.raises(ApiError) as excinfo:
        _prepare(_inputs(tmp_path / "b", chunk_size=8), tmp_path / "bundle")
    assert excinfo.value.code is ErrorCode.CONFLICT


def test_a_forged_artifact_id_is_detected(tmp_path):
    inputs = _inputs(tmp_path)
    prepared = _prepare(inputs, tmp_path / "bundle")
    manifest_path = prepared.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_id"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ApiError) as excinfo:
        load_bundle(prepared.path)
    assert excinfo.value.code is ErrorCode.SOURCE_HASH_MISMATCH


def test_a_modified_bundle_file_is_detected(tmp_path):
    inputs = _inputs(tmp_path)
    prepared = _prepare(inputs, tmp_path / "bundle")
    target = prepared.path / "windows.parquet"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ApiError) as excinfo:
        load_bundle(prepared.path)
    assert excinfo.value.code is ErrorCode.SOURCE_HASH_MISMATCH


def test_a_changed_source_root_is_refused(tmp_path):
    inputs = _inputs(tmp_path)
    prepared = _prepare(inputs, tmp_path / "bundle", digest_files=True)
    loaded = load_bundle(prepared.path)

    copy = tmp_path / "source-copy"
    shutil.copytree(FIXTURE_ROOT, copy)
    verify_source_root(copy, loaded.source_layout())  # 原样复制仍然一致

    victim = next(path for path in copy.rglob("*") if path.is_file() and path.name != "info.json")
    victim.write_bytes(victim.read_bytes() + b"tamper")
    with pytest.raises(ApiError) as excinfo:
        verify_source_root(copy, loaded.source_layout())
    assert excinfo.value.code is ErrorCode.SOURCE_HASH_MISMATCH


def test_synthetic_mode_refuses_a_non_synthetic_origin(tmp_path):
    manifest = snapshot_source_document(
        dataset_root=FIXTURE_ROOT,
        dataset_version_id="fixture:f01",
        data_origin="real_robot_capture",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ApiError) as excinfo:
        load_source_package(
            source_root=FIXTURE_ROOT,
            manifest_path=manifest_path,
            source_mode="synthetic-fixture",
            digest_files=False,
        )
    assert excinfo.value.code is ErrorCode.SOURCE_UNVERIFIED


def test_published_mode_without_a_version_id_is_unverified(tmp_path):
    with pytest.raises(ApiError) as excinfo:
        load_source_package(
            source_root=FIXTURE_ROOT,
            manifest_path=None,
            source_mode="published",
            digest_files=False,
        )
    assert excinfo.value.code is ErrorCode.SOURCE_UNVERIFIED


def test_digest_paths_refuses_a_missing_directory(tmp_path):
    with pytest.raises(ApiError) as excinfo:
        digest_paths(tmp_path / "nope")
    assert excinfo.value.code is ErrorCode.SOURCE_UNVERIFIED


def test_published_semantics_come_from_the_frozen_manifest_episodes():
    """导出包不带 dataops_semantics.json 时，权威语义在冻结 manifest 里。"""
    manifest = {
        "episodes": [
            {"action_semantics": "joint_target_position", "action_names": ["a", "b"]},
            {"action_semantics": "joint_target_position", "action_names": ["a", "b"]},
        ]
    }
    assert _manifest_action_semantics(manifest) == {
        "action_semantics": "joint_target_position",
        "action_names": ["a", "b"],
    }


def test_published_semantics_refuse_to_pick_one_when_episodes_disagree():
    manifest = {
        "episodes": [
            {"action_semantics": "joint_target_position"},
            {"action_semantics": "delta_ee_pose"},
        ]
    }
    assert _manifest_action_semantics(manifest) == {}


@pytest.mark.parametrize(
    "manifest",
    [
        {"episodes": [{"action_semantics": None}, {}]},
        {"episodes": []},
        {},
    ],
)
def test_published_semantics_stay_empty_without_a_declaration(manifest):
    assert _manifest_action_semantics(manifest) == {}
