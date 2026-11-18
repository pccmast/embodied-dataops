"""Integration tests: simulated-evaluation boundary (DO2-29).

``is_simulated`` must survive analysis → order → approval → linking and must be
visible in the API payloads and the Streamlit feedback page.  Nothing here may
claim real robot-training improvement.
"""

from __future__ import annotations

import pytest
from feedback_fixtures import FeedbackFactory

from embodied_dataops.feedback.service import (
    approve_collection_order,
    get_collection_order,
    link_collection_order_episodes,
    list_collection_orders,
)

pytestmark = pytest.mark.integration

SCENARIO = "transparent_cup_low_light"


@pytest.fixture()
def feedback(db, storage):
    return FeedbackFactory(db, storage)


def _setup(feedback, *, is_simulated: bool):
    placeholder = feedback.episode(scenario_key="placeholder_scene", basis="HUMAN_VERIFIED")
    version = feedback.version([placeholder], state="PUBLISHED")
    report = feedback.evaluation(
        version["version_id"],
        [
            {"scenario_key": SCENARIO, "trials": 30, "successes": 12},
            {"scenario_key": "opaque_cup_normal_light", "trials": 30, "successes": 27},
        ],
        is_simulated=is_simulated,
    )
    order = feedback.order(report.id, SCENARIO)
    return version, report, order


def test_simulated_flag_is_preserved_across_the_whole_lifecycle(feedback) -> None:
    version, report, order = _setup(feedback, is_simulated=True)
    assert order.is_simulated_source is True
    assert order.state == "PROPOSED"
    assert report.summary_json["is_simulated"] is True

    listed = list_collection_orders(feedback.db, limit=20, cursor=None, filters={})
    assert listed["data"][0]["is_simulated_source"] is True
    assert listed["data"][0]["state"] == "PROPOSED"
    assert listed["data"][0]["lifecycle"][0] == "PROPOSED"
    assert listed["data"][0]["terminal"] is False

    approve_collection_order(
        feedback.db, order_id=order.id, actor=feedback.principal("reviewer"), note="simulated gap"
    )
    assert order.is_simulated_source is True

    member = feedback.episode(
        scenario_key=SCENARIO,
        basis="HUMAN_VERIFIED",
        collection_order_id=order.id,
    )
    link_collection_order_episodes(
        feedback.db,
        order_id=order.id,
        episode_ids=[member["episode_id"]],
        actor=feedback.principal("operator"),
    )
    feedback.db.refresh(order)
    assert order.state == "IN_PROGRESS"
    assert order.is_simulated_source is True

    detail = get_collection_order(feedback.db, order.id)
    assert detail["is_simulated_source"] is True
    assert detail["evaluation"]["is_simulated"] is True
    assert detail["evaluation"]["evaluation_id"] == report.id
    assert version["version_id"] == report.dataset_version_id


def test_real_evaluation_is_not_marked_simulated(feedback) -> None:
    _version, _report, order = _setup(feedback, is_simulated=False)
    assert order.is_simulated_source is False
    listed = list_collection_orders(feedback.db, limit=20, cursor=None, filters={})
    assert listed["data"][0]["is_simulated_source"] is False
    detail = get_collection_order(feedback.db, order.id)
    assert detail["evaluation"]["is_simulated"] is False


def test_gap_analysis_remains_deterministic_and_proposed_for_simulated_reports(feedback) -> None:
    _version, report, first_order = _setup(feedback, is_simulated=True)
    # Re-run the analysis: same order rows, no duplicates, still PROPOSED.
    from embodied_dataops.feedback.service import analyze_gaps, orders_for_evaluation

    analysis = analyze_gaps(
        feedback.db,
        evaluation_id=report.id,
        success_rate_threshold=0.8,
        min_trials=20,
        actor=feedback.principal("operator"),
    )
    orders = orders_for_evaluation(feedback.db, report.id)
    assert len(orders) == 1
    assert orders[0].id == first_order.id
    assert orders[0].state == "PROPOSED"
    assert orders[0].requested_count == 36  # 2*(30-12)
    assert analysis.state == "SUCCEEDED"
    assert analysis.results_asset_id is not None


