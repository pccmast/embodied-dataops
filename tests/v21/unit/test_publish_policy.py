"""Layered publication admission (C6, SPEC_V2.1 §4.2 / §14).

``publish_level`` existed on the profile model but nothing consumed it: a
``SOURCE_DOCUMENTED_PARTIAL`` profile could be exported and published as a training
dataset.  These tests pin the derivation and the admission rule.
"""

from __future__ import annotations

import pytest

from embodied_dataops.datasets.publish_policy import (
    EXPORT_KIND_RESEARCH,
    EXPORT_KIND_ROBOT,
    assert_publishable,
    export_kind_for_publish_level,
    publish_level_for_profile,
    training_ready_allowed,
)
from embodied_dataops.domain.profiles_v21 import PUBLISH_LEVEL_FORMAT, PUBLISH_LEVEL_SEMANTIC
from embodied_dataops.errors import ApiError, ErrorCode

pytestmark = pytest.mark.no_db


class _NoRows:
    """A session stub: no registered rows, so the shipped files decide."""

    def scalars(self, _statement):
        return _Empty()


class _Empty:
    def first(self):
        return None


def test_a_semantic_profile_is_exported_as_a_robot_training_version():
    level = publish_level_for_profile(
        _NoRows(), profile_id="arm6_joint_position_v1", profile_revision=2
    )
    assert level == PUBLISH_LEVEL_SEMANTIC
    assert export_kind_for_publish_level(level) == EXPORT_KIND_ROBOT
    assert training_ready_allowed(level) is True
    assert_publishable(publish_level=level, export_kind=EXPORT_KIND_ROBOT)


def test_a_partial_profile_can_only_become_a_research_snapshot():
    level = publish_level_for_profile(
        _NoRows(), profile_id="libero_franka_sim_v1", profile_revision=1
    )
    assert level == PUBLISH_LEVEL_FORMAT
    export_kind = export_kind_for_publish_level(level)
    assert export_kind == EXPORT_KIND_RESEARCH
    assert training_ready_allowed(level) is False
    assert_publishable(publish_level=level, export_kind=export_kind)


def test_publishing_a_partial_profile_as_training_data_is_refused():
    with pytest.raises(ApiError) as excinfo:
        assert_publishable(
            publish_level=PUBLISH_LEVEL_FORMAT, export_kind=EXPORT_KIND_ROBOT
        )
    assert excinfo.value.code is ErrorCode.PUBLISH_LEVEL_CONFLICT
    assert excinfo.value.status_code == 409


def test_an_unknown_profile_is_treated_as_format_verified():
    """Unknown provenance is not evidence of semantic correctness."""
    level = publish_level_for_profile(
        _NoRows(), profile_id="does_not_exist", profile_revision=7
    )
    assert level == PUBLISH_LEVEL_FORMAT
    assert export_kind_for_publish_level(level) == EXPORT_KIND_RESEARCH


def test_a_missing_binding_is_treated_as_format_verified():
    assert (
        publish_level_for_profile(_NoRows(), profile_id=None, profile_revision=None)
        == PUBLISH_LEVEL_FORMAT
    )


def test_the_registered_row_wins_when_it_is_readable():
    from embodied_dataops.domain.profiles import load_all_profiles

    profile = next(p for p in load_all_profiles() if p.profile_id == "arm7_joint_position_v1")

    class _Row:
        config_json = profile.model_dump(mode="json")

    class _Session:
        def scalars(self, _statement):
            class _Result:
                def first(self_inner):
                    return _Row()

            return _Result()

    level = publish_level_for_profile(
        _Session(), profile_id=profile.profile_id, profile_revision=profile.profile_revision
    )
    assert level == profile.publish_level
