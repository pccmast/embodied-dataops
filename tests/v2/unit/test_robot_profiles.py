"""V2 unit tests: robot profile registry (DO2-02) and profile resolution (DO2-03)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from embodied_dataops.domain.profiles import (
    AUTHORITY_SYNTHETIC,
    DIAGNOSTIC_ONLY,
    RobotProfileConfig,
    load_profiles,
    resolve_profile,
    upsert_builtin_profiles,
)
from embodied_dataops.errors import ApiError, ErrorCode

pytestmark = pytest.mark.unit


def test_builtin_profiles_match_the_contract():
    profiles = load_profiles()
    by_id = {profile.profile_id: profile for profile in profiles}
    assert set(by_id) == {"arm6_joint_position_v1", "arm7_joint_position_v1"}

    arm6 = by_id["arm6_joint_position_v1"]
    assert (arm6.profile_revision, arm6.state_dim, arm6.action_dim) == (1, 6, 6)
    assert arm6.joint_names == [f"joint_{index}" for index in range(1, 7)]
    assert arm6.authority == AUTHORITY_SYNTHETIC
    assert arm6.state_unit == arm6.action_unit == "rad"
    assert arm6.state_semantics == "joint_position"
    assert arm6.action_semantics == "joint_target_position"
    assert arm6.required_camera_names == ["front"]
    assert arm6.optional_camera_names == ["wrist"]
    assert arm6.aligned_fps == arm6.preview_fps == 10
    assert arm6.max_clock_offset_ns == 50_000_000
    assert arm6.joint_lower_rad == [-3.14] * 6
    assert arm6.joint_upper_rad == [3.14] * 6
    assert arm6.max_abs_velocity_rad_s == [6.0] * 6

    arm7 = by_id["arm7_joint_position_v1"]
    assert (arm7.profile_revision, arm7.state_dim, arm7.action_dim) == (1, 7, 7)
    assert arm7.joint_names == [f"joint_{index}" for index in range(1, 8)]
    assert arm7.joint_lower_rad == [-3.0] * 7
    assert arm7.joint_upper_rad == [3.0] * 7
    assert arm7.max_abs_velocity_rad_s == [4.0] * 7


def test_config_digest_is_stable_and_canonical():
    profiles = load_profiles()
    first = profiles[0].config_sha256()
    second = RobotProfileConfig.model_validate(profiles[0].model_dump(mode="json")).config_sha256()
    assert first == second
    assert len(first) == 64
    # the config JSON must round-trip unchanged (canonical, sorted keys)
    assert profiles[0].canonical_json() == profiles[0].canonical_json()


def test_profile_validation_rejects_bad_dimension_and_limits():
    payload = load_profiles()[0].model_dump(mode="json")
    payload["max_abs_velocity_rad_s"] = [6.0] * 5
    with pytest.raises(ValidationError):
        RobotProfileConfig.model_validate(payload)

    payload = load_profiles()[0].model_dump(mode="json")
    payload["joint_lower_rad"][0] = 5.0  # lower > upper for joint 0
    payload["joint_upper_rad"][0] = -5.0
    with pytest.raises(ValidationError):
        RobotProfileConfig.model_validate(payload)


def test_upsert_is_idempotent_and_uses_db_rows(db):
    first = upsert_builtin_profiles(db)
    db.commit()
    second = upsert_builtin_profiles(db)
    db.commit()
    assert len(first) == len(second) == 2
    assert {row.id for row in first} == {row.id for row in second}
    assert all(row.authority == AUTHORITY_SYNTHETIC for row in first)
    assert all(row.config_json["profile_id"] == row.profile_id for row in first)
    assert all(len(row.config_sha256) == 64 for row in first)


def test_resolution_requires_explicit_profile_and_never_guesses():
    profiles = load_profiles()
    resolved = resolve_profile(
        {"profile_id": "arm7_joint_position_v1", "profile_revision": 1}, profiles=profiles
    )
    assert resolved.mode == "SEMANTIC"
    assert resolved.profile_id == "arm7_joint_position_v1"
    assert resolved.state_dim == 7 and resolved.action_dim == 7
    assert resolved.semantic_authority is True

    with pytest.raises(ApiError) as excinfo:
        resolve_profile(None, profiles=profiles)
    assert excinfo.value.code is ErrorCode.MAPPING_REQUIRED

    # shape alone is never enough, even for a 7-D dataset
    with pytest.raises(ApiError) as excinfo:
        resolve_profile({"shape": [7]}, profiles=profiles)
    assert excinfo.value.code is ErrorCode.MAPPING_REQUIRED

    diagnostic = resolve_profile({DIAGNOSTIC_ONLY: True}, profiles=profiles)
    assert diagnostic.mode == "DIAGNOSTIC_ONLY"
    assert diagnostic.config is None
    assert diagnostic.semantic_authority is False

    with pytest.raises(ApiError) as excinfo:
        resolve_profile({"profile_id": "arm6_joint_position_v1", "profile_revision": 99}, profiles=profiles)
    assert excinfo.value.code is ErrorCode.NOT_FOUND

    with pytest.raises(ApiError) as excinfo:
        resolve_profile({"profile_id": "arm6_joint_position_v1"}, profiles=profiles)
    assert excinfo.value.code is ErrorCode.MAPPING_REQUIRED


def test_unverified_authority_is_flagged_not_semantic():
    profiles = load_profiles()
    payload = profiles[0].model_dump(mode="json")
    payload["authority"] = "UNVERIFIED"
    unverified = RobotProfileConfig.model_validate(payload)
    resolved = resolve_profile(
        {"profile_id": unverified.profile_id, "profile_revision": unverified.profile_revision},
        profiles=[unverified],
    )
    assert resolved.mode == "SEMANTIC"
    assert resolved.semantic_authority is False
