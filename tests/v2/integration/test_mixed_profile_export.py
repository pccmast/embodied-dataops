"""DO2-09: mixed profile / camera / fps / action-semantics export is rejected.

One LeRobot export version must use a single profile revision, camera-key
combination, aligned fps and action semantics; anything mixed returns
``409 MIXED_PROFILE_EXPORT_UNSUPPORTED``.  The dataset-manifest/2.0 episode entry
must also carry the profile/native-index/aligned-view evidence.
"""

from __future__ import annotations

import pytest

from embodied_dataops.datasets.export_v2 import episode_manifest_entry, validate_v2_membership
from embodied_dataops.datasets.manifest_v2 import (
    build_manifest_v2,
    manifest_v2_bytes,
    manifest_v2_sha256,
)
from embodied_dataops.errors import ApiError, ErrorCode

pytestmark = pytest.mark.integration


def _episode(suffix: str, **overrides) -> dict:
    entry = {
        "episode_id": f"episode-{suffix}",
        "schema_version": "canonical-episode/2.0",
        "profile_id": "arm6_joint_position_v1",
        "profile_revision": 1,
        "profile_config_sha256": "1" * 64,
        "camera_keys": ["front", "wrist"],
        "aligned_fps": 10,
        "action_semantics": "joint_target_position",
    }
    entry.update(overrides)
    return entry


def test_consistent_membership_is_accepted():
    contract = validate_v2_membership([_episode("a"), _episode("b")])
    assert contract["profile_id"] == "arm6_joint_position_v1"
    assert contract["camera_keys"] == ["front", "wrist"]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"profile_id": "arm7_joint_position_v1", "profile_revision": 1}, "profile_id"),
        ({"profile_config_sha256": "2" * 64}, "profile_config_sha256"),
        ({"camera_keys": ["front"]}, "camera_keys"),
        ({"aligned_fps": 15}, "aligned_fps"),
        ({"action_semantics": "joint_velocity"}, "action_semantics"),
    ],
)
def test_mixed_membership_is_rejected_with_409(overrides: dict, field: str):
    with pytest.raises(ApiError) as excinfo:
        validate_v2_membership([_episode("a"), _episode("b", **overrides)])
    assert excinfo.value.code is ErrorCode.MIXED_PROFILE_EXPORT_UNSUPPORTED
    assert excinfo.value.status_code == 409
    assert field in str(excinfo.value.details)


def test_v1_episode_in_v2_export_is_rejected():
    with pytest.raises(ApiError) as excinfo:
        validate_v2_membership(
            [_episode("a"), {"episode_id": "v1", "schema_version": "canonical-episode/1.0"}]
        )
    assert excinfo.value.code is ErrorCode.MIXED_PROFILE_EXPORT_UNSUPPORTED


def test_manifest_v2_is_canonical_and_deterministic():
    episodes = [
        episode_manifest_entry(
            episode_id=f"episode-{suffix}",
            metadata={
                "schema_version": "canonical-episode/2.0",
                "publishable": True,
                "profile_id": "arm6_joint_position_v1",
                "profile_revision": 1,
                "profile_config_sha256": "1" * 64,
                "camera_names": ["front"],
                "state_dim": 6,
                "action_dim": 6,
                "action_semantics": "joint_target_position",
                "aligned_view": {"fps": 10, "asset_sha256": "3" * 64},
            },
            canonical_sha256="4" * 64,
            source_sha256="5" * 64,
            quality_report_id=f"quality-{suffix}",
            annotation_set_id=f"annotation-{suffix}",
            annotation_revision_id=f"revision-{suffix}",
            native_indexes={"aligned": "3" * 64, "camera_front": "6" * 64},
        )
        for suffix in ("b", "a")
    ]
    manifest = build_manifest_v2(
        dataset_slug="demo-v2",
        version=1,
        episodes=episodes,
        profile={
            "profile_id": "arm6_joint_position_v1",
            "profile_revision": 1,
            "profile_config_sha256": "1" * 64,
            "authority": "SYNTHETIC_FIXTURE",
        },
        camera_transforms=[{"camera": "front", "scale": 1.0, "pad_left": 0, "pad_top": 0}],
        created_at="2026-10-29T00:00:00Z",
    )
    assert manifest["schema_version"] == "dataset-manifest/2.0"
    assert [entry["episode_id"] for entry in manifest["episodes"]] == ["episode-a", "episode-b"]
    body = manifest_v2_bytes(manifest)
    assert body == manifest_v2_bytes(manifest)
    assert manifest_v2_sha256(manifest) == manifest_v2_sha256(manifest)
    profile = manifest["profile"]
    assert profile["profile_id"] == "arm6_joint_position_v1"


def test_unpublishable_episode_is_refused():
    with pytest.raises(ApiError) as excinfo:
        episode_manifest_entry(
            episode_id="episode-x",
            metadata={
                "schema_version": "canonical-episode/2.0",
                "publishable": False,
                "profile_id": "arm6_joint_position_v1",
                "profile_revision": 1,
                "profile_config_sha256": "1" * 64,
                "camera_names": ["front"],
                "aligned_view": {"fps": 10, "asset_sha256": "3" * 64},
            },
            canonical_sha256="4" * 64,
            source_sha256="5" * 64,
            quality_report_id="quality-x",
            annotation_set_id="annotation-x",
            annotation_revision_id=None,
            native_indexes={"aligned": "3" * 64},
        )
    assert excinfo.value.code is ErrorCode.INCOMPATIBLE_EPISODE
