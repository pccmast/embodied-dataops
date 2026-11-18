"""``robot-profile/2.1`` contract tests (SPEC_V2.1 §4.2, OV-03/OV-04).

These are pure in-process checks: no PostgreSQL, no object store.  They pin the
behaviour that lets one platform describe a 6/6 synthetic arm, a 16/16 real
dual-arm sample and an 8/7 simulation dataset without ever coercing one into
another.
"""

from __future__ import annotations

import json

import pytest

from embodied_dataops.domain.profiles import (
    DEFAULT_PROFILE_FILE_V21,
    RobotProfileConfig,
    load_all_profiles,
    load_profiles,
    resolve_profile,
)
from embodied_dataops.domain.profiles_v21 import (
    AUTHORITY_DOCUMENTED_PARTIAL,
    PUBLISH_LEVEL_FORMAT,
    PUBLISH_LEVEL_SEMANTIC,
    RobotProfileV21,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.quality.v2_rules import _physics_limits_available

pytestmark = pytest.mark.no_db


def _profile(**overrides):
    payload = {
        "schema_version": "robot-profile/2.1",
        "profile_id": "test_profile",
        "profile_revision": 1,
        "authority": "SOURCE_DOCUMENTED_PARTIAL",
        "state": {
            "dimension": 3,
            "features": [
                {"name": "a", "unit": "unknown", "role": "unknown"},
                {"name": "b", "unit": "unknown", "role": "unknown"},
                {"name": "c", "unit": "0-1", "role": "gripper"},
            ],
            "semantics": "unknown_3d_state",
            "unit": "unknown",
        },
        "action": {
            "dimension": 2,
            "features": [
                {"name": "x", "unit": "unknown", "role": "unknown"},
                {"name": "y", "unit": "unknown", "role": "unknown"},
            ],
            "semantics": "unknown_2d_action",
            "unit": "unknown",
        },
        "cameras": ["front"],
        "aligned_fps": 10,
        "preview_fps": 10,
        "max_clock_offset_ns": 50_000_000,
        "semantic_qc_enabled": False,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_merges_both_schema_generations_without_collisions():
    v2_only = load_profiles()
    merged = load_all_profiles()
    assert len(v2_only) == 2
    # 2.1 file: arm6 rev2, arm7 rev2, agibot, libero, utokyo rev1/rev2, aloha rev1/rev2/rev3
    assert len(merged) == 11
    keys = [(p.profile_id, p.profile_revision) for p in merged]
    assert len(set(keys)) == len(keys)


def test_v2_default_loader_is_unchanged():
    """V2 callers must keep seeing exactly the two synthetic profiles."""
    profiles = load_profiles()
    assert {p.profile_id for p in profiles} == {"arm6_joint_position_v1", "arm7_joint_position_v1"}
    assert all(isinstance(p, RobotProfileConfig) for p in profiles)


def test_synthetic_arm_revision_two_restates_the_same_limits():
    resolved = resolve_profile({"profile_id": "arm6_joint_position_v1", "profile_revision": 2})
    config = resolved.config
    assert isinstance(config, RobotProfileV21)
    assert (resolved.state_dim, resolved.action_dim) == (6, 6)
    assert config.limits_available is True
    assert config.joint_lower_rad == [-3.14] * 6
    assert config.max_abs_velocity_rad_s == [6.0] * 6
    assert config.publish_level == PUBLISH_LEVEL_SEMANTIC


# ---------------------------------------------------------------------------
# unequal dimensions (OV-04, V21-04)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_id", ["libero_franka_sim_v1", "utokyo_xarm_pick_and_place_v1"])
def test_unequal_state_action_dimensions_are_preserved(profile_id):
    resolved = resolve_profile({"profile_id": profile_id, "profile_revision": 1})
    assert resolved.state_dim == 8
    assert resolved.action_dim == 7
    config = resolved.config
    assert config.unequal_dimensions is True
    # the action width must never be inferred from the state width
    assert len(config.action_names) == 7
    assert len(config.joint_names) == 8


def test_real_dataset_profile_uses_the_names_declared_by_the_source():
    """The utokyo snapshot declares ``motor_0..motor_7`` / ``motor_0..motor_6``."""
    config = resolve_profile(
        {"profile_id": "utokyo_xarm_pick_and_place_v1", "profile_revision": 1}
    ).config
    assert config.joint_names == [f"motor_{index}" for index in range(8)]
    assert config.action_names == [f"motor_{index}" for index in range(7)]
    assert config.expected_source_cameras == 3
    assert "da56c23e" in config.source_dataset


def test_aloha_profile_is_fourteen_by_fourteen_with_one_source_camera():
    """ALOHA sim: state[14] / action[14], one source video mapped to the front camera."""
    config = resolve_profile(
        {"profile_id": "aloha_sim_transfer_cube_human_v1", "profile_revision": 1}
    ).config
    assert (config.state_dim, config.action_dim) == (14, 14)
    assert config.cameras == ["front"]
    assert config.expected_source_cameras == 1
    assert set(config.state.roles) == {"unknown"}
    assert config.state_unit == "unknown"
    # No published limits: physics QC must be NOT_APPLICABLE, never a silent PASS.
    assert config.limits_available is False
    assert config.semantic_qc_enabled is False
    assert config.publish_level == PUBLISH_LEVEL_FORMAT


def test_aloha_revision_2_uses_the_published_channel_names():
    """Rev 2 carries the card's channel names; rev 1 stays frozen with generic ones."""
    rev1 = resolve_profile(
        {"profile_id": "aloha_sim_transfer_cube_human_v1", "profile_revision": 1}
    ).config
    rev2 = resolve_profile(
        {"profile_id": "aloha_sim_transfer_cube_human_v1", "profile_revision": 2}
    ).config
    assert rev1.state.features[0].name == "state_0"
    assert rev2.state.features[0].name == "left_waist"
    assert rev2.action.features[6].name == "left_gripper"
    # A published name is not evidence for units/limits: the level must not move.
    assert rev2.authority == rev1.authority
    assert rev2.publish_level == rev1.publish_level
    assert rev2.config_sha256() != rev1.config_sha256()


def test_agibot_profile_keeps_sixteen_unknown_channels():
    config = resolve_profile({"profile_id": "agibot_g2_public_v1", "profile_revision": 1}).config
    assert (config.state_dim, config.action_dim) == (16, 16)
    assert set(config.state.roles) == {"unknown"}
    assert set(config.state.names) == {f"state_{index}" for index in range(16)}
    assert config.expected_source_cameras == 3


# ---------------------------------------------------------------------------
# publication level (V21-05)
# ---------------------------------------------------------------------------


def test_partial_authority_can_only_reach_format_verified():
    config = resolve_profile({"profile_id": "libero_franka_sim_v1", "profile_revision": 1}).config
    assert config.authority == AUTHORITY_DOCUMENTED_PARTIAL
    assert config.semantic_authority is False
    assert config.publish_level == PUBLISH_LEVEL_FORMAT
    assert config.training_ready_allowed is False


def test_profiles_without_limits_disable_physics_rules():
    libero = resolve_profile({"profile_id": "libero_franka_sim_v1", "profile_revision": 1}).config
    arm6 = resolve_profile({"profile_id": "arm6_joint_position_v1", "profile_revision": 2}).config
    assert _physics_limits_available(libero) is False
    assert _physics_limits_available(arm6) is True


# ---------------------------------------------------------------------------
# validation guards
# ---------------------------------------------------------------------------


def test_gripper_channel_may_not_claim_radians():
    payload = _profile()
    payload["state"]["features"][2] = {"name": "grip", "unit": "rad", "role": "gripper"}
    with pytest.raises(Exception) as excinfo:
        RobotProfileV21.model_validate(payload)
    assert "gripper" in str(excinfo.value)


def test_features_length_must_match_declared_dimension():
    payload = _profile()
    payload["state"]["dimension"] = 4
    with pytest.raises(Exception) as excinfo:
        RobotProfileV21.model_validate(payload)
    assert "dimension" in str(excinfo.value)


def test_duplicate_feature_names_are_rejected():
    payload = _profile()
    payload["state"]["features"][1]["name"] = "a"
    with pytest.raises(Exception) as excinfo:
        RobotProfileV21.model_validate(payload)
    assert "unique" in str(excinfo.value)


def test_semantic_qc_requires_limits():
    payload = _profile(semantic_qc_enabled=True)
    with pytest.raises(Exception) as excinfo:
        RobotProfileV21.model_validate(payload)
    assert "limits" in str(excinfo.value)


def test_limits_length_must_match_state_dimension():
    payload = _profile(
        limits={"lower": [-1.0] * 2, "upper": [1.0] * 2, "max_abs_velocity": [1.0] * 2, "continuous": [False] * 2}
    )
    payload["semantic_qc_enabled"] = True
    with pytest.raises(Exception) as excinfo:
        RobotProfileV21.model_validate(payload)
    assert "limits" in str(excinfo.value)


def test_unknown_schema_version_is_rejected_not_coerced(tmp_path):
    payload = json.loads(DEFAULT_PROFILE_FILE_V21.read_text(encoding="utf-8"))
    payload["profiles"][0]["schema_version"] = "robot-profile/9.9"
    broken = tmp_path / "profiles.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ApiError) as excinfo:
        load_profiles(broken)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_profile_sha_is_stable_and_matches_the_file():
    first = load_all_profiles()
    second = load_all_profiles()
    assert [p.config_sha256() for p in first] == [p.config_sha256() for p in second]
    assert all(len(p.config_sha256()) == 64 for p in first)
