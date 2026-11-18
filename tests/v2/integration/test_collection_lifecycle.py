"""Integration tests: collection-order lifecycle and fulfillment (DO2-28)."""

from __future__ import annotations

import pytest
from feedback_fixtures import FeedbackFactory
from sqlalchemy import select

from embodied_dataops.db.models import AuditEvent, CollectionOrderEpisode, Episode
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.feedback.service import (
    approve_collection_order,
    cancel_collection_order,
    get_collection_order,
    link_collection_order_episodes,
)

pytestmark = pytest.mark.integration

SCENARIO = "transparent_cup_low_light"


@pytest.fixture()
def feedback(db, storage):
    return FeedbackFactory(db, storage)


def _order_setup(feedback, *, requested_count: int = 20):
    """Evaluation on a published version; returns (version, report, order)."""
    placeholder = feedback.episode(scenario_key="placeholder_scene", basis="HUMAN_VERIFIED")
    version = feedback.version([placeholder], state="PUBLISHED")
    # 20 trials / 11 successes -> success rate 55% < 80%, requested_count=max(20,18)=20
    trials = 20
    successes = trials - (requested_count // 2 if requested_count > 20 else 9)
    report = feedback.evaluation(
        version["version_id"],
        [{"scenario_key": SCENARIO, "trials": trials, "successes": successes}],
        is_simulated=True,
    )
    order = feedback.order(report.id, SCENARIO)
    return version, report, order


def _eligible_episode(feedback, order, **kwargs):
    return feedback.episode(
        scenario_key=SCENARIO,
        basis="HUMAN_VERIFIED",
        collection_order_id=order.id,
        **kwargs,
    )


def test_creator_cannot_approve_own_order(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    operator = feedback.principal("operator")  # same principal that created the order
    assert order.created_by == operator.key_id
    with pytest.raises(ApiError) as excinfo:
        approve_collection_order(
            feedback.db, order_id=order.id, actor=operator, note="looks reasonable"
        )
    assert excinfo.value.code is ErrorCode.FORBIDDEN
    assert excinfo.value.status_code == 403
    feedback.db.refresh(order)
    assert order.state == "PROPOSED"


def test_different_reviewer_approves_and_audit_is_recorded(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    reviewer = feedback.principal("reviewer")
    approved = approve_collection_order(
        feedback.db,
        order_id=order.id,
        actor=reviewer,
        note="Simulated gap confirmed for the transparent cup scenario.",
        request_id="req-approve",
    )
    assert approved.state == "APPROVED"
    assert approved.approved_by == reviewer.key_id
    assert approved.approved_at is not None
    assert "transparent cup" in approved.approval_note
    assert approved.is_simulated_source is True

    audit = feedback.db.scalars(
        select(AuditEvent).where(AuditEvent.action == "collection_order.approve")
    ).first()
    assert audit is not None
    assert audit.entity_id == order.id
    assert audit.after_json["state"] == "APPROVED"

    with pytest.raises(ApiError) as excinfo:
        approve_collection_order(
            feedback.db,
            order_id=order.id,
            actor=feedback.principal("admin", "second-reviewer"),
            note="approve again",
        )
    assert excinfo.value.code is ErrorCode.CONFLICT


def test_cannot_link_before_approval(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    member = _eligible_episode(feedback, order)
    with pytest.raises(ApiError) as excinfo:
        link_collection_order_episodes(
            feedback.db,
            order_id=order.id,
            episode_ids=[member["episode_id"]],
            actor=feedback.principal("operator"),
        )
    assert excinfo.value.code is ErrorCode.CONFLICT
    assert "APPROVED or IN_PROGRESS" in excinfo.value.message


def test_link_requires_credible_scenario_tag(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="approve"
    )
    untagged = feedback.episode(collection_order_id=order.id)
    with pytest.raises(ApiError) as excinfo:
        link_collection_order_episodes(
            feedback.db,
            order_id=order.id,
            episode_ids=[untagged["episode_id"]],
            actor=feedback.principal("operator"),
        )
    assert excinfo.value.code is ErrorCode.CONFLICT
    assert "scenario tag" in excinfo.value.message


def test_link_requires_matching_import_provenance(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="approve"
    )
    tagged_but_not_linked_import = feedback.episode(scenario_key=SCENARIO, basis="HUMAN_VERIFIED")
    with pytest.raises(ApiError) as excinfo:
        link_collection_order_episodes(
            feedback.db,
            order_id=order.id,
            episode_ids=[tagged_but_not_linked_import["episode_id"]],
            actor=feedback.principal("operator"),
        )
    assert excinfo.value.code is ErrorCode.CONFLICT
    assert "traceable" in excinfo.value.message


def test_link_requires_pass_warn_quality(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="approve"
    )
    failed = _eligible_episode(feedback, order, quality_status="FAIL")
    with pytest.raises(ApiError) as excinfo:
        link_collection_order_episodes(
            feedback.db,
            order_id=order.id,
            episode_ids=[failed["episode_id"]],
            actor=feedback.principal("operator"),
        )
    assert excinfo.value.code is ErrorCode.CONFLICT
    assert "quality" in excinfo.value.message.lower()


def test_link_requires_approved_annotation(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="approve"
    )
    unapproved = _eligible_episode(feedback, order, approved_annotation=False)
    with pytest.raises(ApiError) as excinfo:
        link_collection_order_episodes(
            feedback.db,
            order_id=order.id,
            episode_ids=[unapproved["episode_id"]],
            actor=feedback.principal("operator"),
        )
    assert excinfo.value.code is ErrorCode.CONFLICT
    assert "annotation" in excinfo.value.message.lower()


def test_partial_link_moves_to_in_progress_and_bumps_episode_revision(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="approve"
    )
    member = _eligible_episode(feedback, order)
    episode = feedback.db.get(Episode, member["episode_id"])
    assert episode is not None
    revision_before = episode.business_revision

    linked = link_collection_order_episodes(
        feedback.db,
        order_id=order.id,
        episode_ids=[member["episode_id"]],
        actor=feedback.principal("operator"),
        request_id="req-link",
    )
    assert linked.state == "IN_PROGRESS"
    rows = list(
        feedback.db.scalars(
            select(CollectionOrderEpisode).where(CollectionOrderEpisode.order_id == order.id)
        )
    )
    assert len(rows) == 1
    feedback.db.refresh(episode)
    assert episode.business_revision == revision_before + 1

    # Replaying the same link is idempotent (no duplicate row, same state).
    again = link_collection_order_episodes(
        feedback.db,
        order_id=order.id,
        episode_ids=[member["episode_id"]],
        actor=feedback.principal("operator"),
    )
    assert again.state == "IN_PROGRESS"
    assert len(
        list(
            feedback.db.scalars(
                select(CollectionOrderEpisode).where(CollectionOrderEpisode.order_id == order.id)
            )
        )
    ) == 1


def test_episode_cannot_be_linked_to_two_orders(feedback) -> None:
    version, report, first_order = _order_setup(feedback)
    approve_collection_order(
        feedback.db,
        order_id=first_order.id,
        actor=feedback.principal("reviewer"),
        note="approve first order",
    )
    member = _eligible_episode(feedback, first_order)
    link_collection_order_episodes(
        feedback.db,
        order_id=first_order.id,
        episode_ids=[member["episode_id"]],
        actor=feedback.principal("operator"),
    )
    # Second evaluation -> second order; point the import provenance at it to reach
    # the cross-order uniqueness check.
    second_report = feedback.evaluation(
        version["version_id"],
        [{"scenario_key": SCENARIO, "trials": 20, "successes": 9}],
        is_simulated=True,
    )
    second_order = feedback.order(second_report.id, SCENARIO)
    approve_collection_order(
        feedback.db,
        order_id=second_order.id,
        actor=feedback.principal("reviewer", "reviewer-two"),
        note="approve second order",
    )
    member["source"].collection_order_id = second_order.id
    feedback.db.flush()
    with pytest.raises(ApiError) as excinfo:
        link_collection_order_episodes(
            feedback.db,
            order_id=second_order.id,
            episode_ids=[member["episode_id"]],
            actor=feedback.principal("operator"),
        )
    assert excinfo.value.code is ErrorCode.CONFLICT
    assert "another collection order" in excinfo.value.message


def test_order_fulfilled_only_after_count_and_publication(feedback) -> None:
    _version, _report, order = _order_setup(feedback)  # requested_count == 20
    assert order.requested_count == 20
    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="approve"
    )
    members = [_eligible_episode(feedback, order) for _ in range(20)]

    # Not yet published: linking all requested episodes stays IN_PROGRESS.
    link_collection_order_episodes(
        feedback.db,
        order_id=order.id,
        episode_ids=[member["episode_id"] for member in members],
        actor=feedback.principal("operator"),
    )
    feedback.db.refresh(order)
    assert order.state == "IN_PROGRESS"

    # Once every linked Episode is in a PUBLISHED version, fulfillment completes.
    feedback.version(members, state="PUBLISHED")
    link_collection_order_episodes(
        feedback.db,
        order_id=order.id,
        episode_ids=[members[0]["episode_id"]],
        actor=feedback.principal("operator"),
    )
    feedback.db.refresh(order)
    assert order.state == "FULFILLED"

    detail = get_collection_order(feedback.db, order.id)
    assert detail["fulfilled"] is True
    assert detail["linked_count"] == 20
    assert detail["requested_count"] == 20
    assert detail["is_simulated_source"] is True
    assert detail["evaluation"]["is_simulated"] is True
    assert all(item["in_published_version"] for item in detail["linked_episodes"])


def test_cancel_lifecycle_and_terminal_state(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="approve"
    )
    with pytest.raises(ApiError) as missing_reason:
        cancel_collection_order(
            feedback.db, order_id=order.id, actor=feedback.principal("operator"), reason=" "
        )
    assert missing_reason.value.code is ErrorCode.VALIDATION_FAILED
    cancelled = cancel_collection_order(
        feedback.db,
        order_id=order.id,
        actor=feedback.principal("operator"),
        reason="Scenario no longer part of the next data version",
        request_id="req-cancel",
    )
    assert cancelled.state == "CANCELLED"
    with pytest.raises(ApiError) as excinfo:
        cancel_collection_order(
            feedback.db,
            order_id=order.id,
            actor=feedback.principal("operator"),
            reason="cancel twice",
        )
    assert excinfo.value.code is ErrorCode.CONFLICT


def test_link_requires_existing_order_and_valid_ids(feedback) -> None:
    _version, _report, order = _order_setup(feedback)
    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="approve"
    )
    with pytest.raises(ApiError) as missing:
        link_collection_order_episodes(
            feedback.db,
            order_id="00000000-0000-7000-8000-000000000000",
            episode_ids=["00000000-0000-7000-8000-000000000001"],
            actor=feedback.principal("operator"),
        )
    assert missing.value.code is ErrorCode.NOT_FOUND
    with pytest.raises(ApiError) as empty:
        link_collection_order_episodes(
            feedback.db,
            order_id=order.id,
            episode_ids=[],
            actor=feedback.principal("operator"),
        )
    assert empty.value.code is ErrorCode.VALIDATION_FAILED
