"""Integration tests: evidence-backed scenario coverage (DO2-27)."""

from __future__ import annotations

import pytest
from feedback_fixtures import FeedbackFactory

from embodied_dataops.db.models import ScenarioTag
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.feedback import coverage as coverage_service
from embodied_dataops.feedback.coverage import CREDIBLE_BASE_PRIORITY, get_coverage
from embodied_dataops.ids import new_id

pytestmark = pytest.mark.integration


@pytest.fixture()
def feedback(db, storage):
    return FeedbackFactory(db, storage)

SCENARIO = "transparent_cup_low_light"


def _publish(feedback, members):
    return feedback.version(members, state="PUBLISHED")


def test_coverage_counts_distinct_episodes_with_credible_tags(feedback) -> None:
    human = feedback.episode(scenario_key=SCENARIO, basis="HUMAN_VERIFIED")
    declared = feedback.episode(scenario_key=SCENARIO, basis="SOURCE_DECLARED")
    unknown = feedback.episode(scenario_key=SCENARIO, basis="UNKNOWN")
    not_in_version = feedback.episode(scenario_key=SCENARIO, basis="HUMAN_VERIFIED")
    version = _publish(feedback, [human, declared, unknown])

    result = get_coverage(
        feedback.db,
        dataset_version_id=version["version_id"],
        scenario_key=SCENARIO,
    )
    assert result["episode_count"] == 2  # UNKNOWN excluded, not_in_version excluded
    assert result["tagging_basis"] == "HUMAN_VERIFIED"
    assert result["breakdown"] == {"HUMAN_VERIFIED": 1, "SOURCE_DECLARED": 1, "UNKNOWN": 0}
    assert result["denominator"] == "episode_count"
    assert result["snapshot_frozen"] is True
    assert result["source_digest"]
    assert result["computed_at"]
    assert not_in_version  # silence unused warning for clarity


def test_coverage_is_episode_not_frame_denominated(feedback) -> None:
    first = feedback.episode(scenario_key=SCENARIO, basis="SOURCE_DECLARED")
    second = feedback.episode(scenario_key=SCENARIO, basis="SOURCE_DECLARED")
    version = _publish(feedback, [first, second])
    result = get_coverage(
        feedback.db, dataset_version_id=version["version_id"], scenario_key=SCENARIO
    )
    assert result["episode_count"] == 2
    assert result["denominator"] == "episode_count"


def test_unknown_only_gives_zero_coverage_and_unknown_basis(feedback) -> None:
    unknown = feedback.episode(scenario_key=SCENARIO, basis="UNKNOWN")
    version = _publish(feedback, [unknown])
    result = get_coverage(
        feedback.db, dataset_version_id=version["version_id"], scenario_key=SCENARIO
    )
    assert result["episode_count"] == 0
    assert result["tagging_basis"] == "UNKNOWN"
    assert result["breakdown"]["HUMAN_VERIFIED"] == 0
    assert result["breakdown"]["SOURCE_DECLARED"] == 0


def test_same_episode_with_two_credible_bases_counts_once_with_strongest(feedback) -> None:
    member = feedback.episode(scenario_key=SCENARIO, basis="SOURCE_DECLARED")
    feedback.db.add(
        ScenarioTag(
            id=new_id(),
            episode_id=member["episode_id"],
            scenario_key=SCENARIO,
            basis="HUMAN_VERIFIED",
        )
    )
    version = _publish(feedback, [member])
    result = get_coverage(
        feedback.db, dataset_version_id=version["version_id"], scenario_key=SCENARIO
    )
    assert result["episode_count"] == 1
    assert result["tagging_basis"] == "HUMAN_VERIFIED"
    assert result["breakdown"] == {"HUMAN_VERIFIED": 1, "SOURCE_DECLARED": 0, "UNKNOWN": 0}


def test_publish_snapshot_is_frozen_against_later_tag_changes(feedback) -> None:
    declared = feedback.episode(scenario_key=SCENARIO, basis="SOURCE_DECLARED")
    version = _publish(feedback, [declared])
    first = get_coverage(
        feedback.db, dataset_version_id=version["version_id"], scenario_key=SCENARIO
    )
    assert first["episode_count"] == 1
    # After publish, a reviewer verifies the tag: the frozen snapshot must not drift.
    coverage_service.add_scenario_tag(
        feedback.db,
        episode_id=declared["episode_id"],
        scenario_key=SCENARIO,
        basis="HUMAN_VERIFIED",
        actor=feedback.principal("reviewer"),
    )
    feedback.db.flush()
    second = get_coverage(
        feedback.db, dataset_version_id=version["version_id"], scenario_key=SCENARIO
    )
    assert second["episode_count"] == first["episode_count"] == 1
    assert second["tagging_basis"] == first["tagging_basis"] == "SOURCE_DECLARED"
    assert second["source_digest"] == first["source_digest"]
    assert second["computed_at"] == first["computed_at"]


def test_snapshot_generation_is_idempotent(feedback) -> None:
    member = feedback.episode(scenario_key=SCENARIO, basis="HUMAN_VERIFIED")
    version = _publish(feedback, [member])
    version_row = version["version"]
    first = coverage_service.ensure_coverage_snapshot(feedback.db, version_row)
    second = coverage_service.ensure_coverage_snapshot(feedback.db, version_row)
    assert [row.id for row in first] == [row.id for row in second]
    assert len(first) == 1


def test_coverage_requires_published_version(feedback) -> None:
    member = feedback.episode(scenario_key=SCENARIO, basis="HUMAN_VERIFIED")
    version = feedback.version([member], state="READY")
    with pytest.raises(ApiError) as excinfo:
        get_coverage(
            feedback.db, dataset_version_id=version["version_id"], scenario_key=SCENARIO
        )
    assert excinfo.value.code is ErrorCode.CONFLICT
    assert excinfo.value.status_code == 409


def test_coverage_missing_version_is_404(feedback) -> None:
    with pytest.raises(ApiError) as excinfo:
        get_coverage(
            feedback.db,
            dataset_version_id="00000000-0000-7000-8000-000000000000",
            scenario_key=SCENARIO,
        )
    assert excinfo.value.code is ErrorCode.NOT_FOUND


def test_coverage_invalid_scenario_key_is_422(feedback) -> None:
    member = feedback.episode(scenario_key=SCENARIO, basis="HUMAN_VERIFIED")
    version = _publish(feedback, [member])
    with pytest.raises(ApiError) as excinfo:
        get_coverage(
            feedback.db, dataset_version_id=version["version_id"], scenario_key="Bad-Key"
        )
    assert excinfo.value.code is ErrorCode.SCENARIO_KEY_INVALID
    assert excinfo.value.status_code == 422


def test_credible_basis_priority_is_human_then_source(feedback) -> None:
    assert CREDIBLE_BASE_PRIORITY == ("HUMAN_VERIFIED", "SOURCE_DECLARED")
    member = feedback.episode(scenario_key=SCENARIO, basis="SOURCE_DECLARED")
    assert (
        coverage_service.credible_scenario_basis(
            feedback.db, member["episode_id"], SCENARIO
        )
        == "SOURCE_DECLARED"
    )
    unknown = feedback.episode(scenario_key=SCENARIO, basis="UNKNOWN")
    assert (
        coverage_service.credible_scenario_basis(
            feedback.db, unknown["episode_id"], SCENARIO
        )
        is None
    )


def test_scenario_tags_are_idempotent_per_episode_key_basis(feedback) -> None:
    member = feedback.episode(scenario_key=None)
    actor = feedback.principal("reviewer")
    first = coverage_service.add_scenario_tag(
        feedback.db,
        episode_id=member["episode_id"],
        scenario_key=SCENARIO,
        basis="HUMAN_VERIFIED",
        actor=actor,
    )
    second = coverage_service.add_scenario_tag(
        feedback.db,
        episode_id=member["episode_id"],
        scenario_key=SCENARIO,
        basis="HUMAN_VERIFIED",
        actor=actor,
    )
    assert first.id == second.id
    tags = coverage_service.list_scenario_tags(feedback.db, member["episode_id"])
    assert len(tags) == 1
    assert tags[0]["basis"] == "HUMAN_VERIFIED"
    assert tags[0]["verified_by"] == actor.key_id
