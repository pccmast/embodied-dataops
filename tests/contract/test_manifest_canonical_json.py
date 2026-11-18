"""Contract test: canonical ``dataset-manifest/1.0`` JSON (SPEC 10.1).

The manifest bytes are the publish-time identity of a dataset version, so this
test pins the serialisation contract independently of the service/database code.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from embodied_dataops.datasets.manifest import (
    EPISODE_MANIFEST_FIELDS,
    build_manifest,
    manifest_bytes,
    manifest_sha256,
    parse_manifest,
)

EPISODE_ORDER = [
    "11111111-1111-7111-8111-111111111111",
    "22222222-2222-7222-8222-222222222222",
]


def _episodes() -> list[dict]:
    return [
        {
            "episode_id": episode_id,
            "canonical_sha256": hashlib.sha256(episode_id.encode()).hexdigest(),
            "source_sha256": hashlib.sha256(f"source:{episode_id}".encode()).hexdigest(),
            "quality_report_id": f"qr-{index}",
            "annotation_set_id": f"as-{index}",
        }
        for index, episode_id in enumerate(reversed(EPISODE_ORDER))
    ]


def _manifest() -> dict:
    return build_manifest(
        dataset_slug="robot-grasp",
        version=3,
        episodes=_episodes(),
        created_at="2026-10-29T00:00:00Z",
    )


def test_manifest_exposes_exactly_the_spec_episode_fields() -> None:
    manifest = _manifest()
    assert EPISODE_MANIFEST_FIELDS == (
        "episode_id",
        "canonical_sha256",
        "source_sha256",
        "quality_report_id",
        "annotation_set_id",
    )
    for entry in manifest["episodes"]:
        assert tuple(entry) == EPISODE_MANIFEST_FIELDS


def test_manifest_matches_spec_section_10_1_example_shape() -> None:
    manifest = _manifest()
    expected_keys = {
        "schema_version",
        "dataset_slug",
        "version",
        "export_format",
        "target_fps",
        "episodes",
        "pipeline_version",
        "qc_policy_version",
        "created_at",
    }
    assert set(manifest) == expected_keys
    assert manifest["schema_version"] == "dataset-manifest/1.0"
    assert manifest["export_format"] == "lerobot-v3"
    assert manifest["target_fps"] == 10


def test_manifest_episode_order_is_episode_id_ascending() -> None:
    manifest = _manifest()
    ids = [entry["episode_id"] for entry in manifest["episodes"]]
    assert ids == sorted(EPISODE_ORDER)


def test_manifest_bytes_are_stable_canonical_json() -> None:
    first = manifest_bytes(_manifest())
    second = manifest_bytes(_manifest())
    assert first == second
    reparsed = json.loads(first.decode("utf-8"))
    assert manifest_bytes(reparsed) == first
    assert manifest_sha256(reparsed) == hashlib.sha256(first).hexdigest()
    with pytest.raises(ValueError):
        parse_manifest(json.dumps(reparsed, indent=1).encode())
