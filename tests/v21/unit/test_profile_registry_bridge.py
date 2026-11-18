"""Registry bridge: a ``robot-profile/2.1`` row must survive normalisation (C1).

The audit's minimal reproduction was ``_profiles_for_resolution`` validating every
``robot_profiles`` row through the **2.0** model, which produced 19 field errors for
the first 2.1 profile in ``contracts/robot_profiles_v21.json`` and also poisoned plain
V2 normalisation as soon as one 2.1 row was registered.
"""

from __future__ import annotations

import json

import pytest

from embodied_dataops.domain.profiles import (
    AUTHORITY_SYNTHETIC,
    PROFILE_SCHEMA_VERSION,
    RobotProfileConfig,
    combined_registry,
    load_all_profiles,
    parse_profile,
    resolve_profile,
)
from embodied_dataops.domain.profiles_v21 import PROFILE_SCHEMA_VERSION_21, RobotProfileV21
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.workflow.handlers.normalize_episode_v2 import _profiles_for_resolution

pytestmark = pytest.mark.no_db


class _Row:
    """The only attribute ``_profiles_for_resolution`` reads from a registry row."""

    def __init__(self, config: dict) -> None:
        self.config_json = config


class _ScalarsSession:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def scalars(self, _statement) -> list[_Row]:
        return list(self._rows)


def _v21_entry() -> dict:
    payload = json.loads(
        (__import__("pathlib").Path("contracts") / "robot_profiles_v21.json").read_text(
            encoding="utf-8"
        )
    )
    return dict(payload["profiles"][0])


def test_a_2_1_registry_row_is_parsed_by_its_own_generation():
    entry = _v21_entry()
    profile = parse_profile(entry)
    assert isinstance(profile, RobotProfileV21)
    assert profile.schema_version == PROFILE_SCHEMA_VERSION_21


def test_profiles_for_resolution_accepts_a_2_1_row_and_resolves_it():
    entry = _v21_entry()
    profiles = _profiles_for_resolution(_ScalarsSession([_Row(entry)]))

    resolved = resolve_profile(
        {"profile_id": entry["profile_id"], "profile_revision": entry["profile_revision"]},
        profiles=profiles,
    )
    assert resolved.profile_id == entry["profile_id"]
    assert resolved.state_dim == len(entry["state"]["features"])
    assert resolved.action_dim == len(entry["action"]["features"])


def test_a_registered_2_0_row_still_resolves():
    legacy = next(p for p in load_all_profiles() if p.schema_version == PROFILE_SCHEMA_VERSION)
    profiles = _profiles_for_resolution(_ScalarsSession([_Row(legacy.model_dump(mode="json"))]))
    resolved = resolve_profile(
        {"profile_id": legacy.profile_id, "profile_revision": legacy.profile_revision},
        profiles=profiles,
    )
    assert resolved.mode == "SEMANTIC"
    assert isinstance(resolved.config, RobotProfileConfig)


def test_shipped_2_1_profiles_are_visible_even_when_the_registry_is_v2_only():
    """A stale registry must not hide a profile the code ships."""
    legacy = [p for p in load_all_profiles() if p.schema_version == PROFILE_SCHEMA_VERSION]
    rows = [_Row(p.model_dump(mode="json")) for p in legacy]
    profiles = _profiles_for_resolution(_ScalarsSession(rows))
    keys = {(p.profile_id, p.profile_revision) for p in profiles}
    assert ("libero_franka_sim_v1", 1) in keys


def test_a_conflicting_registered_revision_is_a_configuration_error():
    entry = _v21_entry()
    registered = parse_profile(entry)
    broken = entry.copy()
    broken["notes"] = "a different contract under the same revision number"
    with pytest.raises(ApiError) as excinfo:
        combined_registry([parse_profile(broken)], builtin=[registered])
    assert excinfo.value.code is ErrorCode.CONFLICT


def test_registered_rows_win_over_the_file_when_they_agree():
    profiles = load_all_profiles()
    merged = combined_registry(list(profiles), builtin=profiles)
    assert len(merged) == len({(p.profile_id, p.profile_revision) for p in profiles})


def test_unknown_generation_is_rejected_not_coerced():
    with pytest.raises(ApiError) as excinfo:
        parse_profile({"schema_version": "robot-profile/9.9", "profile_id": "x"})
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_synthetic_authority_is_still_semantic():
    profile = next(p for p in load_all_profiles() if p.authority == AUTHORITY_SYNTHETIC)
    assert profile.semantic_authority is True
