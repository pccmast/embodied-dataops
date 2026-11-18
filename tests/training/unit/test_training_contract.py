"""配方合同与语义摘要（EDO-TRAIN-PREP-1.1 §5.2、§11.1）。"""

from __future__ import annotations

import pytest

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.contract import (
    SENTINEL_ACTION,
    SENTINEL_CAMERAS,
    SENTINEL_FPS,
    VerifiedSource,
    canonical_json_bytes,
    parse_recipe,
    recipe_sha256,
    resolve_recipe,
    resolved_recipe_sha256,
    semantic_sha256,
)

pytestmark = pytest.mark.no_db


def _recipe(**overrides):
    payload = {
        "target": {},
        "features": {},
        "temporal": {"chunk_size": 16, "n_action_steps": 16, "stride": 1},
        "split": {
            "strategy": "grouped_hash_v1",
            "seed": 20261031,
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "group_mapping": "source_groups.json",
        },
    }
    payload.update(overrides)
    return payload


def _source(**overrides) -> VerifiedSource:
    values = {
        "dataset_version_id": "dv-1",
        "manifest_sha256": "a" * 64,
        "profile_id": "arm6_joint_position_v1",
        "profile_revision": 2,
        "camera_keys": ("front", "wrist"),
        "aligned_fps": 10,
        "state_dim": 6,
        "action_dim": 6,
        "state_names": tuple(f"joint_{i}" for i in range(6)),
        "action_names": tuple(f"joint_{i}" for i in range(6)),
        "state_unit": "rad",
        "action_unit": "rad",
        "action_semantics": "joint_target_position",
        "episode_ids": ("ep-1", "ep-2"),
        "data_origin": "synthetic",
        "source_verification": "FIXTURE_ONLY",
    }
    values.update(overrides)
    return VerifiedSource(**values)


# ---------------------------------------------------------------------------
# strictness
# ---------------------------------------------------------------------------
def test_recipe_parses_with_documented_defaults():
    recipe = parse_recipe(_recipe())
    assert recipe.schema_version == "training-recipe/1.0"
    assert recipe.target.policy_type == "act"
    assert recipe.target.vision_backbone == "resnet18"
    assert recipe.temporal.n_obs_steps == 1
    assert recipe.temporal.tail_policy == "drop"
    assert recipe.normalization.modes == {
        "VISUAL": "MEAN_STD",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }
    assert recipe.quality.invalid_policy == "reject_window"


def test_unknown_field_is_rejected():
    with pytest.raises(ApiError) as excinfo:
        parse_recipe(_recipe(extra_key=True))
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert "training recipe" in excinfo.value.message


def test_n_action_steps_must_not_exceed_chunk_size():
    with pytest.raises(ApiError) as excinfo:
        parse_recipe(
            _recipe(temporal={"chunk_size": 8, "n_action_steps": 16, "stride": 1})
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_ratios_must_sum_to_one():
    split = {
        "strategy": "grouped_hash_v1",
        "seed": 1,
        "ratios": {"train": 0.8, "val": 0.1},
        "group_mapping": "g.json",
    }
    with pytest.raises(ApiError) as excinfo:
        parse_recipe(_recipe(split=split))
    assert "sum to 1.0" in str(excinfo.value.details)


def test_grouped_hash_strategy_must_not_carry_assignments():
    split = {
        "strategy": "grouped_hash_v1",
        "seed": 1,
        "ratios": {"train": 1.0},
        "group_mapping": "g.json",
        "assignments": "group_assignments.json",
    }
    with pytest.raises(ApiError):
        parse_recipe(_recipe(split=split))


def test_explicit_strategy_must_not_carry_seed_or_ratios():
    split = {
        "strategy": "explicit_group_assignment_v1",
        "seed": 1,
        "group_mapping": "g.json",
        "assignments": "group_assignments.json",
    }
    with pytest.raises(ApiError):
        parse_recipe(_recipe(split=split))


def test_explicit_strategy_parses_without_seed():
    split = {
        "strategy": "explicit_group_assignment_v1",
        "group_mapping": "g.json",
        "assignments": "group_assignments.json",
    }
    recipe = parse_recipe(_recipe(split=split))
    assert recipe.split.strategy == "explicit_group_assignment_v1"
    assert recipe.split.seed is None


# ---------------------------------------------------------------------------
# canonical serialisation (§11.1)
# ---------------------------------------------------------------------------
def test_semantic_digest_ignores_key_order_and_float_spelling():
    left = {"b": 0.8, "a": [1, 2.5], "c": "x"}
    right = {"c": "x", "a": [1, 2.50], "b": 0.80}
    assert semantic_sha256(left) == semantic_sha256(right)


def test_semantic_digest_changes_when_a_value_changes():
    assert semantic_sha256({"train": 0.8}) != semantic_sha256({"train": 0.81})


def test_negative_zero_is_canonicalised_to_positive_zero():
    assert canonical_json_bytes({"x": -0.0}) == canonical_json_bytes({"x": 0.0})


def test_floats_use_hex_representation_in_the_digest_input():
    assert b"0x1.999999999999ap-1" in canonical_json_bytes({"x": 0.8})


def test_non_finite_values_are_refused():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ApiError) as excinfo:
            semantic_sha256({"x": bad})
        assert excinfo.value.code is ErrorCode.NONFINITE_INPUT


def test_canonical_bytes_end_with_a_single_newline():
    payload = canonical_json_bytes({"a": 1})
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")


# ---------------------------------------------------------------------------
# source binding (§5.2)
# ---------------------------------------------------------------------------
def test_sentinels_are_resolved_from_the_verified_source():
    recipe = parse_recipe(_recipe())
    assert recipe.features.cameras == SENTINEL_CAMERAS
    assert recipe.features.fps == SENTINEL_FPS
    assert recipe.features.action_representation == SENTINEL_ACTION

    resolved = resolve_recipe(recipe, _source())
    assert resolved.features.cameras == ["front", "wrist"]
    assert resolved.features.fps == 10
    assert resolved.features.action_representation == "joint_target_position"
    assert resolved.features.state_dim == 6
    assert resolved.source.source_verification == "FIXTURE_ONLY"
    assert resolved.recipe_sha256 == recipe_sha256(recipe)


def test_explicit_camera_must_exist_in_the_source():
    recipe = parse_recipe(_recipe(features={"cameras": ["front", "side"]}))
    with pytest.raises(ApiError) as excinfo:
        resolve_recipe(recipe, _source())
    assert excinfo.value.code is ErrorCode.FEATURE_INCOMPATIBLE
    assert excinfo.value.details["missing"] == ["side"]


def test_explicit_fps_must_match_the_source():
    recipe = parse_recipe(_recipe(features={"fps": 30}))
    with pytest.raises(ApiError) as excinfo:
        resolve_recipe(recipe, _source())
    assert excinfo.value.code is ErrorCode.FEATURE_INCOMPATIBLE


def test_an_ego_source_cannot_become_an_act_demonstration():
    recipe = parse_recipe(_recipe())
    ego = _source(state_dim=0, action_dim=0, state_names=(), action_names=())
    with pytest.raises(ApiError) as excinfo:
        resolve_recipe(recipe, ego)
    assert excinfo.value.code is ErrorCode.FEATURE_INCOMPATIBLE
    assert "Ego" in excinfo.value.message


def test_undocumented_action_semantics_are_refused():
    recipe = parse_recipe(_recipe())
    with pytest.raises(ApiError) as excinfo:
        resolve_recipe(recipe, _source(action_semantics=""))
    assert excinfo.value.code is ErrorCode.FEATURE_INCOMPATIBLE


def test_source_without_a_frozen_manifest_digest_is_unverified():
    recipe = parse_recipe(_recipe())
    with pytest.raises(ApiError) as excinfo:
        resolve_recipe(recipe, _source(manifest_sha256=""))
    assert excinfo.value.code is ErrorCode.SOURCE_UNVERIFIED


def test_resolved_digest_tracks_the_source_binding():
    recipe = parse_recipe(_recipe())
    first = resolve_recipe(recipe, _source())
    second = resolve_recipe(recipe, _source(manifest_sha256="b" * 64))
    assert resolved_recipe_sha256(first) != resolved_recipe_sha256(second)
