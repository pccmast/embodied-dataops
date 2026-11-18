"""Bounded processor that completes QUEUED gap analyses (DO-A02 C3/C4).

The controlled ``POST /evaluations/{id}/analyze-gaps`` commits a truthful
``GapAnalysis(state="QUEUED")`` + outbox wakeup in the same transaction as its
receipt/fence.  The dispatcher process then runs :func:`process_queued_gap_analyses`
on every cycle: it row-locks ``QUEUED`` rows (``FOR UPDATE SKIP LOCKED``), runs the
deterministic V1 gap rule (``feedback.gaps``) and CASes the analysis to
``SUCCEEDED`` while inserting the collection orders exactly once.

The V1 synchronous ``feedback.service.analyze_gaps`` entry point is untouched, so
direct callers keep their historical behaviour.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import CollectionOrder, EvaluationReport, GapAnalysis
from embodied_dataops.feedback.gaps import compute_gap_orders, gap_summary
from embodied_dataops.ids import new_id
from embodied_dataops.operations.evaluations import stored_results
from embodied_dataops.workflow.outbox import utcnow

logger = logging.getLogger(__name__)

MAX_ANALYSES_PER_CYCLE = 20


def process_queued_gap_analyses(
    session: Session, *, limit: int = MAX_ANALYSES_PER_CYCLE, now: datetime | None = None
) -> int:
    """Advance up to ``limit`` QUEUED analyses; returns the number processed."""
    now = now or utcnow()
    statement = (
        select(GapAnalysis)
        .where(GapAnalysis.state == "QUEUED")
        .order_by(GapAnalysis.created_at, GapAnalysis.id)
        .limit(max(1, int(limit)))
        .with_for_update(skip_locked=True)
    )
    processed = 0
    for analysis in session.scalars(statement):
        report = session.get(EvaluationReport, analysis.evaluation_report_id, populate_existing=True)
        if report is None:
            _fail(analysis, now, "EVALUATION_NOT_FOUND")
            processed += 1
            continue
        results = stored_results(report)
        if not results:
            _fail(analysis, now, "EVALUATION_NOT_READY")
            processed += 1
            continue
        try:
            orders = compute_gap_orders(
                results,
                success_rate_threshold=analysis.threshold,
                min_trials=analysis.min_trials,
            )
            summary = gap_summary(
                orders,
                success_rate_threshold=analysis.threshold,
                min_trials=analysis.min_trials,
            )
            analysis.summary_json = summary
            analysis.state = "SUCCEEDED"
            analysis.finished_at = now
            _ensure_orders(session, report_id=report.id, orders=orders)
        except Exception:  # noqa: BLE001 - a bad report must not wedge the queue
            logger.exception("gap_analysis.failed analysis_id=%s", analysis.id)
            _fail(analysis, now, "GAP_ANALYSIS_FAILED")
        processed += 1
        logger.info(
            "gap_analysis.processed analysis_id=%s state=%s orders=%d",
            analysis.id,
            analysis.state,
            int((analysis.summary_json or {}).get("order_count", 0)),
        )
    return processed


def _fail(analysis: GapAnalysis, now: datetime, error_code: str) -> None:
    analysis.state = "FAILED"
    analysis.summary_json = {"error_code": error_code}
    analysis.finished_at = now


def _ensure_orders(session: Session, *, report_id: str, orders: list[dict[str, Any]]) -> None:
    """Insert collection orders idempotently (unique per report+scenario_key)."""
    existing_keys = set(
        session.scalars(
            select(CollectionOrder.scenario_key).where(
                CollectionOrder.evaluation_report_id == report_id
            )
        )
    )
    for order in orders:
        if order["scenario_key"] in existing_keys:
            continue
        session.add(
            CollectionOrder(
                id=new_id(),
                evaluation_report_id=report_id,
                scenario_key=order["scenario_key"],
                reason=order["reason"],
                requested_count=int(order["requested_count"]),
                state="PROPOSED",
                created_at=utcnow(),
            )
        )
        existing_keys.add(order["scenario_key"])
    session.flush()


__all__ = ["MAX_ANALYSES_PER_CYCLE", "process_queued_gap_analyses"]
