"""Research-snapshot export path (FORMAT_VERIFIED profiles).

A ``FORMAT_VERIFIED`` profile (units/semantics not evidenced) must still be freezable as
``RESEARCH_SNAPSHOT``, while broken bytes keep failing.  The manifest derives
``training_ready`` from the frozen ``publish_level`` so a later export cannot relabel a
research snapshot as training-ready.
"""

from __future__ import annotations

import pytest

from embodied_dataops.canonical.v2.contract import CANONICAL_V2_SCHEMA_VERSION
from embodied_dataops.datasets.export_v2 import episode_manifest_entry, research_snapshot_allowed
from embodied_dataops.datasets.manifest_v2 import build_manifest_v2
from embodied_dataops.datasets.publish_policy import (
    EXPORT_KIND_RESEARCH,
    EXPORT_KIND_ROBOT,
    PUBLISH_LEVEL_FORMAT,
    PUBLISH_LEVEL_SEMANTIC,
)
from embodied_dataops.errors import ApiError, ErrorCode

pytestmark = pytest.mark.no_db


def _metadata(*, publishable: bool, codes: tuple[str, ...]) -> dict:
    return {
        "schema_version": CANONICAL_V2_SCHEMA_VERSION,
        "publishable": publishable,
        "diagnostics": [{"code": code, "message": code} for code in codes],
        "profile_id": "utokyo_xarm_pick_and_place_v1",
        "profile_revision": 1,
        "profile_config_sha256": "b" * 64,
        "camera_names": ["front", "wrist"],
        "state_dim": 8,
        "action_dim": 7,
        "joint_names": [f"state_{index}" for index in range(8)],
        "action_names": [f"action_{index}" for index in range(7)],
        "action_semantics": "unknown_7d_action",
        "aligned_view": {"fps": 10, "asset_sha256": "d" * 64},
    }


def _entry(metadata: dict, *, allow_research_snapshot: bool) -> dict:
    return episode_manifest_entry(
        episode_id="ep-1",
        metadata=metadata,
        canonical_sha256="a" * 64,
        source_sha256="c" * 64,
        quality_report_id="qr-1",
        annotation_set_id="as-1",
        annotation_revision_id=None,
        native_indexes={"state": "e" * 64, "action": "f" * 64, "aligned": "d" * 64},
        allow_research_snapshot=allow_research_snapshot,
    )


def test_research_snapshot_allowed_for_semantic_gaps_only():
    semantic_gap = type(
        "M",
        (),
        {
            "publishable": False,
            "diagnostics": [{"code": "UNVERIFIED_SEMANTICS"}, {"code": "CAMERA_UNMAPPED_IN_SOURCE"}],
        },
    )()
    assert research_snapshot_allowed(semantic_gap) is True


def test_research_snapshot_rejected_when_a_format_diagnostic_is_present():
    mixed = type(
        "M",
        (),
        {
            "publishable": False,
            "diagnostics": [{"code": "UNVERIFIED_SEMANTICS"}, {"code": "NON_MONOTONIC_TIMESTAMPS"}],
        },
    )()
    assert research_snapshot_allowed(mixed) is False
    # No diagnostics at all is not evidence of "semantics only" either.
    empty = type("M", (), {"publishable": False, "diagnostics": []})()
    assert research_snapshot_allowed(empty) is False


def test_episode_entry_is_frozen_for_a_research_snapshot():
    entry = _entry(
        _metadata(publishable=False, codes=("UNVERIFIED_SEMANTICS",)),
        allow_research_snapshot=True,
    )
    assert entry["episode_id"] == "ep-1"
    assert entry["state_dim"] == 8 and entry["action_dim"] == 7


def test_episode_entry_still_rejects_the_same_episode_for_a_training_version():
    with pytest.raises(ApiError) as excinfo:
        _entry(
            _metadata(publishable=False, codes=("UNVERIFIED_SEMANTICS",)),
            allow_research_snapshot=False,
        )
    assert excinfo.value.code == ErrorCode.INCOMPATIBLE_EPISODE


def test_episode_entry_rejects_format_diagnostics_even_for_a_research_snapshot():
    with pytest.raises(ApiError):
        _entry(
            _metadata(publishable=False, codes=("UNVERIFIED_SEMANTICS", "NULL_STATE_SAMPLES")),
            allow_research_snapshot=True,
        )


def _manifest(publish_level: str) -> dict:
    entry = _entry(_metadata(publishable=True, codes=()), allow_research_snapshot=False)
    return build_manifest_v2(
        dataset_slug="utokyo",
        version=1,
        episodes=[entry],
        profile={
            "profile_id": entry["profile_id"],
            "profile_revision": entry["profile_revision"],
            "profile_config_sha256": entry["profile_config_sha256"],
            "authority": "SOURCE_DOCUMENTED_PARTIAL",
            "publish_level": publish_level,
        },
        created_at="2026-11-02T00:00:00Z",
    )


def test_manifest_freezes_research_snapshot_for_format_level():
    manifest = _manifest(PUBLISH_LEVEL_FORMAT)
    assert manifest["profile"]["publish_level"] == PUBLISH_LEVEL_FORMAT
    assert manifest["profile"]["training_ready"] is False
    assert manifest["profile"]["export_kind"] == EXPORT_KIND_RESEARCH


def test_manifest_allows_training_ready_claim_only_for_semantic_level():
    manifest = _manifest(PUBLISH_LEVEL_SEMANTIC)
    assert manifest["profile"]["training_ready"] is True
    assert manifest["profile"]["export_kind"] == EXPORT_KIND_ROBOT
