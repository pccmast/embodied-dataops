"""Unit tests for ``evaluation/1.0`` validation and the deterministic gap rule.

Covers AT-19: F09 (transparent cup 12/30) must produce exactly one transparent
order with ``requested_count=36``; 27/30 produces nothing; 19 trials is below the
minimum and produces nothing; invalid scenario keys are rejected.
"""

from __future__ import annotations

import pytest

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.feedback.evaluation import validate_evaluation_payload
from embodied_dataops.feedback.gaps import (
    DEFAULT_MIN_TRIALS,
    DEFAULT_SUCCESS_RATE_THRESHOLD,
    compute_gap_orders,
    gap_summary,
    requested_count_for,
)

F09_RESULTS = [
    {"scenario_key": "transparent_cup_low_light", "trials": 30, "successes": 12},
    {"scenario_key": "opaque_cup_normal_light", "trials": 30, "successes": 27},
]


def test_f09_transparent_cup_gets_one_order_with_36_requests() -> None:
    orders = compute_gap_orders(F09_RESULTS)
    assert [order["scenario_key"] for order in orders] == ["transparent_cup_low_light"]
    assert orders[0]["requested_count"] == 36
    assert requested_count_for(30, 12) == 36
    assert "40.0%" in orders[0]["reason"]
    assert "80.0%" in orders[0]["reason"]


def test_f09_opaque_cup_27_of_30_gets_no_order() -> None:
    orders = compute_gap_orders([F09_RESULTS[1]])
    assert orders == []


def test_below_min_trials_gives_no_gap() -> None:
    orders = compute_gap_orders([{"scenario_key": "few_trials", "trials": 19, "successes": 0}])
    assert orders == []
    assert DEFAULT_MIN_TRIALS == 20


def test_exactly_min_trials_is_considered() -> None:
    orders = compute_gap_orders([{"scenario_key": "boundary_case", "trials": 20, "successes": 0}])
    assert len(orders) == 1
    assert orders[0]["requested_count"] == 40  # max(20, 2*(20-0))


def test_success_rate_equal_to_threshold_is_not_a_gap() -> None:
    assert DEFAULT_SUCCESS_RATE_THRESHOLD == 0.80
    assert compute_gap_orders([{"scenario_key": "at_threshold", "trials": 30, "successes": 24}]) == []


def test_success_rate_just_below_threshold_is_a_gap() -> None:
    orders = compute_gap_orders([{"scenario_key": "just_below", "trials": 30, "successes": 23}])
    assert len(orders) == 1
    assert orders[0]["requested_count"] == 20  # max(20, 14)


def test_orders_are_sorted_by_scenario_key() -> None:
    results = [
        {"scenario_key": "zeta_scenario", "trials": 30, "successes": 0},
        {"scenario_key": "alpha_scenario", "trials": 30, "successes": 3},
        {"scenario_key": "middle_scenario", "trials": 30, "successes": 10},
    ]
    orders = compute_gap_orders(results)
    assert [order["scenario_key"] for order in orders] == [
        "alpha_scenario",
        "middle_scenario",
        "zeta_scenario",
    ]


def test_custom_threshold_and_min_trials_are_applied() -> None:
    results = [{"scenario_key": "custom_rule", "trials": 10, "successes": 2}]
    assert compute_gap_orders(results) == []
    strict = compute_gap_orders(results, success_rate_threshold=0.9, min_trials=10)
    assert len(strict) == 1
    assert strict[0]["requested_count"] == 20


def test_requested_count_never_below_twenty() -> None:
    assert requested_count_for(30, 30) == 20
    assert requested_count_for(30, 29) == 20
    assert requested_count_for(30, 10) == 40
    assert requested_count_for(30, 0) == 60


def test_gap_summary_counts() -> None:
    orders = compute_gap_orders(F09_RESULTS)
    summary = gap_summary(orders, success_rate_threshold=0.8, min_trials=20)
    assert summary["order_count"] == 1
    assert summary["total_requested_count"] == 36
    assert summary["orders"][0]["scenario_key"] == "transparent_cup_low_light"


def _payload(results: list[dict], **overrides) -> dict:
    payload = {
        "schema_version": "evaluation/1.0",
        "model_id": "demo-policy-v1",
        "results": results,
        "is_simulated": True,
    }
    payload.update(overrides)
    return payload


def test_valid_f09_payload_is_accepted() -> None:
    validated = validate_evaluation_payload(_payload(F09_RESULTS))
    assert validated.is_simulated is True
    assert validated.model_id == "demo-policy-v1"
    assert [result.scenario_key for result in validated.results] == [
        "transparent_cup_low_light",
        "opaque_cup_normal_light",
    ]


@pytest.mark.parametrize(
    "scenario_key",
    ["Bad-Key", "ab", "UPPER", "has space", "x" * 65, "", "emoji_🙂"],
)
def test_invalid_scenario_key_is_rejected(scenario_key: str) -> None:
    with pytest.raises(ApiError) as excinfo:
        validate_evaluation_payload(
            _payload([{"scenario_key": scenario_key, "trials": 30, "successes": 12}])
        )
    assert excinfo.value.code is ErrorCode.SCENARIO_KEY_INVALID
    assert excinfo.value.status_code == 422


def test_scenario_key_minimum_length_three_is_accepted() -> None:
    validated = validate_evaluation_payload(
        _payload([{"scenario_key": "abc", "trials": 20, "successes": 0}])
    )
    assert validated.results[0].scenario_key == "abc"


def test_wrong_schema_version_is_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        validate_evaluation_payload(_payload(F09_RESULTS, schema_version="evaluation/2.0"))
    assert excinfo.value.code is ErrorCode.EVALUATION_SCHEMA_INVALID


def test_missing_is_simulated_is_rejected() -> None:
    payload = _payload(F09_RESULTS)
    payload.pop("is_simulated")
    with pytest.raises(ApiError) as excinfo:
        validate_evaluation_payload(payload)
    assert excinfo.value.code is ErrorCode.EVALUATION_SCHEMA_INVALID


def test_successes_greater_than_trials_is_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        validate_evaluation_payload(
            _payload([{"scenario_key": "bad_counts", "trials": 10, "successes": 11}])
        )
    assert excinfo.value.code is ErrorCode.EVALUATION_SCHEMA_INVALID


def test_boolean_trials_are_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        validate_evaluation_payload(
            _payload([{"scenario_key": "bool_trials", "trials": True, "successes": 0}])
        )
    assert excinfo.value.code is ErrorCode.EVALUATION_SCHEMA_INVALID


def test_duplicate_scenario_keys_are_rejected() -> None:
    results = [
        {"scenario_key": "dupe_key", "trials": 30, "successes": 1},
        {"scenario_key": "dupe_key", "trials": 30, "successes": 2},
    ]
    with pytest.raises(ApiError) as excinfo:
        validate_evaluation_payload(_payload(results))
    assert excinfo.value.code is ErrorCode.EVALUATION_SCHEMA_INVALID


def test_empty_results_are_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        validate_evaluation_payload(_payload([]))
    assert excinfo.value.code is ErrorCode.EVALUATION_SCHEMA_INVALID
