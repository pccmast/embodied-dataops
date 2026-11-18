"""DO2-05: explicit robot-profile import mapping for canonical-episode/2.0.

These tests exercise the profile-bound contract without external services: the
LeRobot/MCAP adapters themselves are covered by the fixture smoke tests; here we
verify the profile registry selection rules and that a prealigned (already
row-paired) source never claims a fabricated native frequency.
"""

from __future__ import annotations

import pytest
from _canonical_helpers import prealigned_source

from embodied_dataops.canonical.v2 import build_canonical_v2
from embodied_dataops.domain.profiles import load_profiles, resolve_profile
from embodied_dataops.errors import ApiError, ErrorCode

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("profile_id", "dim"),
    [("arm6_joint_position_v1", 6), ("arm7_joint_position_v1", 7)],
)
def test_prealigned_import_binds_explicit_profile(profile_id: str, dim: int):
    source = prealigned_source(profile_id=profile_id)
    bundle = build_canonical_v2(source, canonical_version=1)
    metadata = bundle.metadata
    assert metadata.schema_version == "canonical-episode/2.0"
    assert metadata.profile_id == profile_id
    assert metadata.profile_revision == 1
    assert metadata.profile_config_sha256 == source.profile_config_sha256
    assert metadata.state_dim == dim
    assert metadata.action_dim == dim
    assert metadata.aligned_view.prealigned is True
    assert metadata.aligned_view.source_row_rate_hz == 10.0
    assert metadata.aligned_view.valid_sample_count == metadata.aligned_view.sample_count
    assert bundle.publishable is True
    # already-paired rows: no fabricated native rate is claimed for robot streams
    for modality in metadata.modalities:
        if modality.name in {"state", "action"}:
            assert modality.native_rate_hz is None
            assert modality.alignment_method == "source_row_pairing"


def test_missing_profile_selection_is_rejected():
    with pytest.raises(ApiError) as excinfo:
        resolve_profile({})
    assert excinfo.value.code in {ErrorCode.MAPPING_REQUIRED, ErrorCode.INVALID_MAPPING}


def test_diagnostic_only_mode_is_not_publishable():
    resolved = resolve_profile({"diagnostic_only": True})
    assert resolved.mode == "DIAGNOSTIC_ONLY"
    assert resolved.profile_id is None
    assert resolved.semantic_authority is False


def test_profile_registry_is_versioned_and_explicit():
    profiles = load_profiles()
    assert {(item.profile_id, item.profile_revision) for item in profiles} >= {
        ("arm6_joint_position_v1", 1),
        ("arm7_joint_position_v1", 1),
    }
    # profile selection is never inferred from shape: a 7D source under the 6D
    # profile must be reported as a dimension problem, not silently accepted.
    source = prealigned_source(profile_id="arm6_joint_position_v1")
    source.state = type(source.state)(
        name="state",
        source_key=source.state.source_key,
        clock_domain=source.state.clock_domain,
        timestamp_provenance=source.state.timestamp_provenance,
        times_ns=source.state.times_ns,
        values=[row + [0.0] for row in source.state.values],
        unit=source.state.unit,
    )
    bundle = build_canonical_v2(source, canonical_version=1)
    assert bundle.publishable is False
    assert "structural_diagnostics_present" in bundle.blocking_reasons
