"""``EGO_ANNOTATION_BUNDLE`` tests (SPEC_V2.1 §14, V21-15).

The central property under test is separation: an ego video must never be able to leave
the platform looking like a robot control trajectory, and a bundle must never carry video
bytes the licence does not allow.
"""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from embodied_dataops.datasets.ego_bundle import (
    BUNDLE_FILES,
    BundleError,
    BundleInput,
    assemble_bundle,
    build_bundle_payloads,
)

pytestmark = pytest.mark.no_db

SHA = "a" * 64


def _episode(**overrides):
    payload = {
        "episode_id": "ep_1",
        "episode_type": "ego_video_episode",
        "duration_ns": 6_000_000_000,
        "source_asset_sha256": SHA,
        "data_origin": "ego_capture",
        "license_id": "CC-BY-NC-4.0",
    }
    payload.update(overrides)
    return payload


def _action(**overrides):
    payload = {
        "episode_id": "ep_1",
        "verb": "pick_up",
        "noun": "cup",
        "start_ns": 1_000_000_000,
        "end_ns": 2_500_000_000,
        "evidence_frame_indices": [1, 2],
        "origin": "MODEL",
        "review_state": "APPROVED",
        "annotation_revision_id": "rev_2",
    }
    payload.update(overrides)
    return payload


def _bundle(**overrides) -> BundleInput:
    payload = {
        "dataset_slug": "ego-kitchen",
        "version": 1,
        "episodes": [_episode()],
        "actions": [_action()],
        "provenance": {"sources": [{"episode_id": "ep_1", "egress_allowed": False}]},
        "video_redistributable": False,
    }
    payload.update(overrides)
    return BundleInput(**payload)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def test_bundle_has_exactly_the_four_contracted_files():
    result = build_bundle_payloads(_bundle())
    assert set(result.files) == set(BUNDLE_FILES)
    assert result.manifest["files"] == list(BUNDLE_FILES)


def test_manifest_declares_the_export_kind_and_counts():
    result = build_bundle_payloads(_bundle())
    manifest = result.manifest
    assert manifest["export_kind"] == "EGO_ANNOTATION_BUNDLE"
    assert manifest["counts"] == {"episodes": 1, "actions": 1}
    assert manifest["robot_actions_present"] is False


def test_jsonl_files_are_one_row_per_line():
    result = build_bundle_payloads(_bundle(actions=[_action(), _action(start_ns=3_000_000_000, end_ns=4_000_000_000)]))
    lines = [line for line in result.files["annotation.jsonl"].decode().splitlines() if line]
    assert len(lines) == 2
    assert json.loads(lines[0])["verb"] == "pick_up"


def test_provenance_is_carried_verbatim():
    result = build_bundle_payloads(_bundle())
    provenance = json.loads(result.files["provenance.json"])
    assert provenance["export_kind"] == "EGO_ANNOTATION_BUNDLE"
    assert provenance["sources"][0]["egress_allowed"] is False


# ---------------------------------------------------------------------------
# separation
# ---------------------------------------------------------------------------


def test_robot_episode_is_rejected():
    with pytest.raises(BundleError) as excinfo:
        build_bundle_payloads(_bundle(episodes=[_episode(episode_type="robot_episode")]))
    assert "robot_episode" in str(excinfo.value)


def test_duplicate_episode_is_rejected():
    with pytest.raises(BundleError):
        build_bundle_payloads(_bundle(episodes=[_episode(), _episode()]))


def test_action_outside_the_version_is_rejected():
    with pytest.raises(BundleError) as excinfo:
        build_bundle_payloads(_bundle(actions=[_action(episode_id="ep_other")]))
    assert "outside this version" in str(excinfo.value)


def test_non_positive_action_interval_is_rejected():
    with pytest.raises(BundleError):
        build_bundle_payloads(_bundle(actions=[_action(start_ns=5, end_ns=5)]))


def test_empty_version_is_rejected():
    with pytest.raises(BundleError):
        build_bundle_payloads(_bundle(episodes=[], actions=[]))


# ---------------------------------------------------------------------------
# video policy
# ---------------------------------------------------------------------------


def test_video_is_not_included_by_default_and_is_replaced_by_a_hash_pointer():
    result = build_bundle_payloads(_bundle())
    clip = json.loads(result.files["clips.jsonl"].decode().splitlines()[0])
    assert clip["video_included"] is False
    assert clip["controlled_access_pointer"] == f"sha256:{SHA}"
    assert result.manifest["video_included"] is False


def test_archive_contains_no_video_when_redistribution_is_not_allowed():
    result = assemble_bundle(_bundle(), video_bytes=b"\x00\x00\x00\x18ftypmp42")
    names = _tar_names(result.archive)
    assert "video.mp4" not in names
    assert set(names) == set(BUNDLE_FILES)


def test_archive_includes_video_only_when_the_licence_allows_it():
    result = assemble_bundle(
        _bundle(video_redistributable=True), video_bytes=b"\x00\x00\x00\x18ftypmp42"
    )
    assert "video.mp4" in _tar_names(result.archive)
    clip = json.loads(result.files["clips.jsonl"].decode().splitlines()[0])
    assert clip["video_included"] is True
    assert clip["controlled_access_pointer"] is None


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_archive_is_byte_identical_for_the_same_input():
    first = assemble_bundle(_bundle())
    second = assemble_bundle(_bundle())
    assert first.archive == second.archive
    assert first.archive_sha256 == second.archive_sha256


def test_archive_changes_when_an_action_changes():
    first = assemble_bundle(_bundle())
    second = assemble_bundle(_bundle(actions=[_action(noun="mug")]))
    assert first.archive_sha256 != second.archive_sha256


def _tar_names(archive: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        return handle.getnames()
