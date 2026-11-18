"""Deterministic gap analysis (SPEC 10.2, F09).

Rule: ``trials >= min_trials`` **and** ``successes / trials < threshold`` creates
one collection order with ``requested_count = max(20, 2 * (trials - successes))``.

There is deliberately no model, no LLM and no random sampling here; the analysis
is a pure function of the report so repeated calls produce identical orders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_SUCCESS_RATE_THRESHOLD = 0.80
DEFAULT_MIN_TRIALS = 20
MIN_REQUESTED_COUNT = 20

#: Collected scenario keys are emitted in lexicographic order (SPEC 10.2).
ORDER_SCENARIO_SORT_KEY = "scenario_key"


def _coerce_threshold(value: Any) -> float:
    threshold = float(value)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("success_rate_threshold must be in (0, 1]")
    return threshold


def _coerce_min_trials(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("min_trials must be an integer")
    min_trials = int(value)
    if min_trials < 1:
        raise ValueError("min_trials must be >= 1")
    return min_trials


def success_rate(trials: int, successes: int) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    return successes / trials


def requested_count_for(trials: int, successes: int) -> int:
    """``max(20, 2 * (trials - successes))`` — F09: 30 trials / 12 successes -> 36."""
    return max(MIN_REQUESTED_COUNT, 2 * (int(trials) - int(successes)))


def reason_for(
    scenario_key: str,
    trials: int,
    successes: int,
    *,
    threshold: float,
    min_trials: int,
) -> str:
    observed = successes / trials
    return (
        f"scenario '{scenario_key}' observed success rate {successes}/{trials} = "
        f"{observed * 100:.1f}% is below the {threshold * 100:.1f}% threshold "
        f"(minimum {min_trials} trials); request {requested_count_for(trials, successes)} "
        f"additional successful demonstrations"
    )


def compute_gap_orders(
    results: Iterable[Mapping[str, Any]],
    *,
    success_rate_threshold: float = DEFAULT_SUCCESS_RATE_THRESHOLD,
    min_trials: int = DEFAULT_MIN_TRIALS,
) -> list[dict[str, Any]]:
    """Return the collection-order specifications for one evaluation report.

    ``results`` items are the validated ``{scenario_key, trials, successes}``
    mappings.  Output is sorted by ``scenario_key`` so the persisted order set is
    deterministic.
    """
    threshold = _coerce_threshold(success_rate_threshold)
    minimum = _coerce_min_trials(min_trials)
    orders: list[dict[str, Any]] = []
    for result in results:
        scenario_key = str(result["scenario_key"])
        trials = int(result["trials"])
        successes = int(result["successes"])
        if trials < 1 or successes < 0 or successes > trials:
            raise ValueError(f"invalid trial counters for scenario {scenario_key!r}")
        if trials < minimum:
            # Too few trials: not statistically meaningful, no order (SPEC 10.2).
            continue
        observed = successes / trials
        if observed >= threshold:
            continue
        orders.append(
            {
                "scenario_key": scenario_key,
                "reason": reason_for(
                    scenario_key,
                    trials,
                    successes,
                    threshold=threshold,
                    min_trials=minimum,
                ),
                "requested_count": requested_count_for(trials, successes),
                "observed_success_rate": observed,
                "trials": trials,
                "successes": successes,
            }
        )
    orders.sort(key=lambda order: order[ORDER_SCENARIO_SORT_KEY])
    return orders


def gap_summary(
    orders: Iterable[Mapping[str, Any]],
    *,
    success_rate_threshold: float,
    min_trials: int,
) -> dict[str, Any]:
    """Machine-readable summary stored on the ``gap_analyses`` row."""
    materialised = [
        {
            "scenario_key": order["scenario_key"],
            "requested_count": int(order["requested_count"]),
            "observed_success_rate": float(order["observed_success_rate"]),
            "trials": int(order["trials"]),
            "successes": int(order["successes"]),
        }
        for order in orders
    ]
    return {
        "success_rate_threshold": float(success_rate_threshold),
        "min_trials": int(min_trials),
        "order_count": len(materialised),
        "total_requested_count": sum(order["requested_count"] for order in materialised),
        "orders": materialised,
    }
