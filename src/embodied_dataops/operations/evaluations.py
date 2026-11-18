"""Evaluation read model for DO-A01 / DO-A02 C2/C5.

The evaluation report is an untrusted import; this module validates the stored
summary again before exposing it to an Agent, checks that the immutable report
asset is present and has the recorded digest, and produces the stable
``results_sha256`` used by the DO-A01 GET response.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import Asset, EvaluationReport
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.feedback.evaluation import (
    EvaluationPayload,
    ScenarioResult,
    validate_evaluation_payload,
)

logger = logging.getLogger(__name__)

MAX_RESULTS = 500
MAX_TRIALS = 100_000


def results_sha256(results: list[dict[str, Any]]) -> str:
    """Stable digest of the ordered business statistics (not the raw file bytes)."""
    return hashlib.sha256(canonical_json_bytes(results)).hexdigest()


def evaluation_results(session: Session, report: EvaluationReport) -> tuple[list[dict[str, Any]], str, EvaluationPayload]:
    """Return ``(sorted_results, results_sha256, validated_payload)`` or 409."""
    summary = report.summary_json or {}
    payload = {
        "schema_version": report.schema_version,
        "model_id": summary.get("model_id", ""),
        "is_simulated": summary.get("is_simulated"),
        "results": summary.get("results"),
    }
    try:
        validated = validate_evaluation_payload(payload)
    except ApiError as exc:
        raise ApiError(
            ErrorCode.EVALUATION_NOT_READY,
            "Stored evaluation report is not readable under evaluation/1.0",
            details={"evaluation_id": report.id, "reason": exc.code.value if hasattr(exc, "code") else "invalid"},
        ) from exc
    if len(validated.results) > MAX_RESULTS:
        raise ApiError(
            ErrorCode.EVALUATION_NOT_READY,
            "Evaluation report has more results than the DO-A01 contract allows",
            details={"evaluation_id": report.id, "result_count": len(validated.results), "max": MAX_RESULTS},
        )
    for result in validated.results:
        if result.trials > MAX_TRIALS:
            raise ApiError(
                ErrorCode.EVALUATION_NOT_READY,
                "Evaluation report has a trial count beyond the DO-A01 contract bound",
                details={"evaluation_id": report.id, "trials": result.trials, "max": MAX_TRIALS},
            )
    _verify_report_asset(session, report)
    sorted_results = [
        {"scenario_key": result.scenario_key, "trials": result.trials, "successes": result.successes}
        for result in sorted(validated.results, key=lambda item: item.scenario_key)
    ]
    return sorted_results, results_sha256(sorted_results), validated


def _verify_report_asset(session: Session, report: EvaluationReport) -> None:
    summary = report.summary_json or {}
    if report.report_asset_id is None:
        raise ApiError(
            ErrorCode.EVALUATION_NOT_READY,
            "Evaluation report asset is missing",
            details={"evaluation_id": report.id},
        )
    asset = session.get(Asset, report.report_asset_id)
    if asset is None:
        raise ApiError(
            ErrorCode.EVALUATION_NOT_READY,
            "Evaluation report asset row is missing",
            details={"evaluation_id": report.id},
        )
    expected = summary.get("report_sha256")
    if expected and asset.sha256 != expected:
        raise ApiError(
            ErrorCode.EVALUATION_NOT_READY,
            "Evaluation report asset digest does not match the stored digest",
            details={"evaluation_id": report.id},
        )


def stored_results(report: EvaluationReport) -> list[dict[str, Any]]:
    """Best-effort results extraction for the gap-analysis processor."""
    summary = report.summary_json or {}
    raw = summary.get("results") or []
    results: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            scenario_key = str(item["scenario_key"])
            trials = int(item["trials"])
            successes = int(item["successes"])
        except (KeyError, TypeError, ValueError):
            continue
        results.append(
            {"scenario_key": scenario_key, "trials": trials, "successes": successes}
        )
    return results


__all__ = [
    "MAX_RESULTS",
    "MAX_TRIALS",
    "ScenarioResult",
    "evaluation_results",
    "results_sha256",
    "stored_results",
]
