"""Unequal-dimension export labels (C3, SPEC_V2.1 OV-08).

The bug: ``build_v2_features`` labelled ``action`` with ``episode.joint_names``.  For
an 8-state / 7-action profile the exported feature had ``shape=(7,)`` and 8 labels --
the official Writer either rejects it or, worse, a consumer reads the labels as the
channel identity of a dimension that does not exist.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from embodied_dataops.canonical.v2.contract import (
    AlignedViewDescriptor,
    CanonicalMetadataV2,
    ModalityDescriptor,
)
from embodied_dataops.datasets.export_v2 import (
    CanonicalV2ExportEpisode,
    _action_feature_names,
    build_v2_features,
    episode_manifest_entry,
)
from embodied_dataops.errors import ApiError, ErrorCode

pytestmark = pytest.mark.no_db

STATE_NAMES = [f"state_{index}" for index in range(8)]
ACTION_NAMES = [f"action_{index}" for index in range(7)]


def _modality(name: str, kind: str) -> ModalityDescriptor:
    return ModalityDescriptor(
        name=name,
        kind=kind,
        dtype="float64",
        source_clock_domain="lerobot_episode_relative",
        timestamp_provenance="source_rows",
        native_sample_count=3,
        native_asset_sha256="c" * 64,
        alignment_method="source_row_aligned",
        alignment_tolerance_ns=0,
    )


def _metadata(*, state_dim: int = 8, action_dim: int = 7, action_names: list[str] | None = None):
    return CanonicalMetadataV2(
        episode_id="ep-1",
        source_asset_sha256="a" * 64,
        profile_id="libero_franka_sim_v1",
        profile_revision=1,
        profile_config_sha256="b" * 64,
        camera_names=["front"],
        state_dim=state_dim,
        action_dim=action_dim,
        joint_names=STATE_NAMES[:state_dim],
        action_names=ACTION_NAMES[:action_dim] if action_names is None else action_names,
        action_semantics="end_effector_delta",
        aligned_view=AlignedViewDescriptor(
            fps=10,
            interval_ns=100_000_000,
            sample_count=3,
            valid_sample_count=3,
            invalid_sample_count=0,
            state_dim=state_dim,
            action_dim=action_dim,
            camera_names=["front"],
            alignment_method="source_row_aligned",
        ),
        modalities=[_modality("front", "camera"), _modality("state", "state"), _modality("action", "action")],
        publishable=True,
    )


def _episode(metadata: CanonicalMetadataV2) -> CanonicalV2ExportEpisode:
    return CanonicalV2ExportEpisode(
        episode_id=metadata.episode_id,
        metadata=metadata,
        aligned=pa.table({}),
        preview_paths={"front": None},
    )


def test_action_feature_is_labelled_with_action_names_not_joint_names():
    features = build_v2_features(_episode(_metadata()))
    assert features["observation.state"]["shape"] == (8,)
    assert features["action"]["shape"] == (7,)
    assert features["observation.state"]["names"] == STATE_NAMES
    assert features["action"]["names"] == ACTION_NAMES


def test_labels_always_match_their_dimension():
    features = build_v2_features(_episode(_metadata()))
    for feature in ("observation.state", "action"):
        assert len(features[feature]["names"]) == features[feature]["shape"][0]


def test_missing_action_names_on_an_unequal_profile_is_refused():
    """No silent fallback: a 7-wide action vector never borrows the 8 state labels."""
    metadata = _metadata(action_names=[])
    assert metadata.action_names == []
    with pytest.raises(ApiError) as excinfo:
        build_v2_features(_episode(metadata))
    assert excinfo.value.code is ErrorCode.INCOMPATIBLE_EPISODE
    assert "action_dim" in str(excinfo.value.details)


def test_same_width_profile_keeps_the_historical_behaviour():
    metadata = _metadata(state_dim=6, action_dim=6, action_names=[])
    assert _action_feature_names(_episode(metadata)) == STATE_NAMES[:6]


def test_contract_rejects_action_names_that_do_not_match_action_dim():
    with pytest.raises(ValueError) as excinfo:
        _metadata(action_names=["only_one"])
    assert "action_names" in str(excinfo.value)


def test_manifest_freezes_both_label_lists():
    entry = episode_manifest_entry(
        episode_id="ep-1",
        metadata=_metadata().model_dump(mode="json"),
        canonical_sha256="d" * 64,
        source_sha256="e" * 64,
        quality_report_id="qr-1",
        annotation_set_id="as-1",
        annotation_revision_id=None,
        native_indexes={"aligned": "f" * 64},
    )
    assert entry["joint_names"] == STATE_NAMES
    assert entry["action_names"] == ACTION_NAMES


def test_an_8_based_state_name_list_cannot_label_a_7_wide_action():
    episode = _episode(_metadata(action_names=[]))
    with pytest.raises(ApiError):
        _action_feature_names(episode)
