"""Unit tests for ``dataset-manifest/1.0`` (SPEC 10.1, AT-17/18 contract)."""

from __future__ import annotations

import hashlib
import json

import pytest

from embodied_dataops import MANIFEST_SCHEMA_VERSION
from embodied_dataops.datasets.manifest import (
    build_manifest,
    episode_entry,
    manifest_bytes,
    manifest_sha256,
    parse_manifest,
    utc_timestamp,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

EPISODE_1 = {
    "episode_id": "00000000-0000-7000-8000-000000000001",
    "canonical_sha256": SHA_A,
    "source_sha256": SHA_B,
    "quality_report_id": "11111111-1111-7111-8111-111111111111",
    "annotation_set_id": "22222222-2222-7222-8222-222222222222",
}
EPISODE_2 = {
    "episode_id": "00000000-0000-7000-8000-000000000002",
    "canonical_sha256": SHA_C,
    "source_sha256": SHA_B,
    "quality_report_id": "33333333-3333-7333-8333-333333333333",
    "annotation_set_id": "44444444-4444-7444-8444-444444444444",
}


def _manifest(episodes=(EPISODE_1, EPISODE_2)) -> dict:
    return build_manifest(
        dataset_slug="robot-grasp",
        version=1,
        episodes=list(episodes),
        created_at="2026-10-29T00:00:00Z",
    )


def test_manifest_required_schema_fields() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["dataset_slug"] == "robot-grasp"
    assert manifest["version"] == 1
    assert manifest["export_format"] == "lerobot-v3"
    assert manifest["target_fps"] == 10
    assert manifest["pipeline_version"] == "1.0.0"
    assert manifest["qc_policy_version"] == "qc-policy/1.0"
    assert manifest["created_at"] == "2026-10-29T00:00:00Z"
    for entry in manifest["episodes"]:
        assert set(entry) == {
            "episode_id",
            "canonical_sha256",
            "source_sha256",
            "quality_report_id",
            "annotation_set_id",
        }


def test_manifest_orders_episodes_by_episode_id_regardless_of_input_order() -> None:
    forward = _manifest((EPISODE_1, EPISODE_2))
    backward = _manifest((EPISODE_2, EPISODE_1))
    assert [entry["episode_id"] for entry in forward["episodes"]] == [
        EPISODE_1["episode_id"],
        EPISODE_2["episode_id"],
    ]
    assert manifest_bytes(forward) == manifest_bytes(backward)
    assert manifest_sha256(forward) == manifest_sha256(backward)


def test_manifest_canonical_json_is_sorted_compact_utf8() -> None:
    manifest = _manifest()
    body = manifest_bytes(manifest)
    text = body.decode("utf-8")
    assert " " not in text
    assert "\n" not in text
    assert json.loads(text) == manifest
    # Sorted keys: the first key of the document is "created_at".
    assert text.startswith('{"created_at":')


def test_manifest_sha256_matches_hashlib() -> None:
    manifest = _manifest()
    expected = hashlib.sha256(manifest_bytes(manifest)).hexdigest()
    assert manifest_sha256(manifest) == expected


def test_manifest_is_deterministic_for_identical_membership() -> None:
    assert manifest_bytes(_manifest()) == manifest_bytes(_manifest())
    assert manifest_sha256(_manifest()) == manifest_sha256(_manifest())


def test_manifest_duplicate_episode_ids_rejected() -> None:
    with pytest.raises(ValueError):
        _manifest((EPISODE_1, EPISODE_1))


def test_manifest_empty_membership_rejected() -> None:
    with pytest.raises(ValueError):
        build_manifest(dataset_slug="robot-grasp", version=1, episodes=[])


def test_manifest_invalid_sha256_rejected() -> None:
    broken = {**EPISODE_1, "canonical_sha256": "not-a-sha"}
    with pytest.raises(ValueError):
        episode_entry(broken)


def test_manifest_missing_episode_field_rejected() -> None:
    broken = dict(EPISODE_1)
    broken.pop("quality_report_id")
    with pytest.raises(ValueError):
        episode_entry(broken)


def test_manifest_version_must_be_positive() -> None:
    with pytest.raises(ValueError):
        build_manifest(dataset_slug="robot-grasp", version=0, episodes=[EPISODE_1])


def test_utc_timestamp_is_rfc3339_z() -> None:
    value = utc_timestamp()
    assert value.endswith("Z")
    assert len(value) == 20


def test_parse_manifest_roundtrip_and_non_canonical_rejection() -> None:
    manifest = _manifest()
    body = manifest_bytes(manifest)
    assert parse_manifest(body) == manifest
    pretty = json.dumps(manifest, indent=2, sort_keys=True).encode()
    with pytest.raises(ValueError):
        parse_manifest(pretty)
    wrong_schema = {**manifest, "schema_version": "dataset-manifest/2.0"}
    with pytest.raises(ValueError):
        parse_manifest(manifest_bytes(wrong_schema))


def test_manifest_sha256_changes_when_membership_changes() -> None:
    one = _manifest((EPISODE_1,))
    two = _manifest((EPISODE_1, EPISODE_2))
    assert manifest_sha256(one) != manifest_sha256(two)
