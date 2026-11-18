"""Feedback service layer (SPEC 9.4, 10.2).

Evaluation import and gap analysis are synchronous, deterministic operations:
the HTTP route only validates and commits, and every call is idempotent.  The
functions here are also used directly by tests and by the Streamlit UI through
the API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import decode_cursor, encode_cursor
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import (
    AnnotationSet,
    Asset,
    CollectionOrder,
    CollectionOrderEpisode,
    DatasetVersion,
    DatasetVersionEpisode,
    Episode,
    EvaluationReport,
    GapAnalysis,
    SourceImport,
)
from embodied_dataops.domain.enums import AnalysisState
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.feedback.coverage import (
    CREDIBLE_BASE_PRIORITY,
    bump_episode_revisions,
    credible_scenario_basis,
)
from embodied_dataops.feedback.evaluation import validate_evaluation_payload
from embodied_dataops.feedback.gaps import (
    DEFAULT_MIN_TRIALS,
    DEFAULT_SUCCESS_RATE_THRESHOLD,
    compute_gap_orders,
    gap_summary,
)
from embodied_dataops.ids import new_id
from embodied_dataops.observability.audit import record_audit
from embodied_dataops.storage.base import get_storage
from embodied_dataops.workflow.outbox import utcnow

EVALUATION_ASSET_KEY = "evaluations/{evaluation_id}/evaluation.json"
GAP_ANALYSIS_ASSET_KEY = "evaluations/{evaluation_id}/gap_analysis.json"

#: V2 lifecycle (SPEC §7.2 / DO-A02 C6).
COLLECTION_ORDER_STATES: tuple[str, ...] = (
    "PROPOSED",
    "APPROVED",
    "IN_PROGRESS",
    "FULFILLED",
    "CANCELLED",
)
COLLECTION_ORDER_TERMINAL_STATES = frozenset({"FULFILLED", "CANCELLED"})
MAX_LINK_EPISODES = 100


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _serialize_order(order: CollectionOrder, *, linked_count: int | None = None) -> dict[str, Any]:
    payload = {
        "id": order.id,
        "evaluation_report_id": order.evaluation_report_id,
        "scenario_key": order.scenario_key,
        "reason": order.reason,
        "requested_count": order.requested_count,
        "state": order.state,
        "lifecycle": list(COLLECTION_ORDER_STATES),
        "terminal": order.state in COLLECTION_ORDER_TERMINAL_STATES,
        "created_by": order.created_by,
        "approved_by": order.approved_by,
        "approved_at": order.approved_at.isoformat() if order.approved_at else None,
        "approval_note": order.approval_note,
        "is_simulated_source": bool(order.is_simulated_source),
        "created_at": order.created_at.isoformat() if order.created_at else None,
        "updated_at": order.updated_at.isoformat() if order.updated_at else None,
    }
    if linked_count is not None:
        payload["linked_count"] = int(linked_count)
    return payload


def _linked_counts(session: Session, order_ids: Sequence[str]) -> dict[str, int]:
    if not order_ids:
        return {}
    statement = (
        select(CollectionOrderEpisode.order_id, func.count(CollectionOrderEpisode.episode_id))
        .where(CollectionOrderEpisode.order_id.in_(order_ids))
        .group_by(CollectionOrderEpisode.order_id)
    )
    return {row[0]: int(row[1]) for row in session.execute(statement)}


def _max_basis(bases: Sequence[str]) -> str | None:
    for basis in CREDIBLE_BASE_PRIORITY:
        if basis in bases:
            return basis
    return None


def get_evaluation(session: Session, evaluation_id: str) -> EvaluationReport:
    report = session.get(EvaluationReport, evaluation_id)
    if report is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Evaluation report not found: {evaluation_id}")
    return report


# ---------------------------------------------------------------------------
# evaluation import
# ---------------------------------------------------------------------------
def create_evaluation(
    session: Session,
    *,
    dataset_version_id: str,
    payload: Mapping[str, Any],
    actor: Principal | None = None,
    request_id: str | None = None,
) -> EvaluationReport:
    """Validate ``evaluation/1.0`` and persist the immutable report asset."""
    version = session.get(DatasetVersion, dataset_version_id)
    if version is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Dataset version not found: {dataset_version_id}")

    validated = validate_evaluation_payload(dict(payload))
    body = validated.canonical_bytes()
    evaluation_id = new_id()
    object_key = EVALUATION_ASSET_KEY.format(evaluation_id=evaluation_id)
    stored = get_storage().put_bytes(object_key, body, content_type="application/json")

    from embodied_dataops.db.models import Asset

    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or "",
        byte_size=len(body),
        media_type="application/json",
        object_key=object_key,
        origin="report",
        logical_name="evaluation.json",
        meta_json={"kind": "evaluation_report", "schema_version": validated.schema_version},
    )
    session.add(asset)
    session.flush()

    report = EvaluationReport(
        id=evaluation_id,
        dataset_version_id=dataset_version_id,
        source="imported",
        schema_version=validated.schema_version,
        report_asset_id=asset.id,
        summary_json={
            "model_id": validated.model_id,
            "is_simulated": validated.is_simulated,
            "results": [result.as_dict() for result in validated.results],
            "report_sha256": stored.sha256,
        },
        created_at=utcnow(),
    )
    session.add(report)
    session.flush()

    record_audit(
        session,
        actor=actor,
        action="evaluation.create",
        entity_type="evaluation_report",
        entity_id=report.id,
        after={"dataset_version_id": dataset_version_id, "sha256": stored.sha256},
        request_id=request_id,
    )
    return report


# ---------------------------------------------------------------------------
# gap analysis
# ---------------------------------------------------------------------------
def analyze_gaps(
    session: Session,
    *,
    evaluation_id: str,
    success_rate_threshold: float | None = None,
    min_trials: int | None = None,
    actor: Principal | None = None,
    request_id: str | None = None,
) -> GapAnalysis:
    """Create (or return) the deterministic gap analysis for a report.

    Re-running the analysis for the same report returns the existing row and the
    same collection orders; it never creates duplicates (AT-19).
    """
    report = get_evaluation(session, evaluation_id)
    threshold = float(
        DEFAULT_SUCCESS_RATE_THRESHOLD if success_rate_threshold is None else success_rate_threshold
    )
    minimum = int(DEFAULT_MIN_TRIALS if min_trials is None else min_trials)
    if not 0.0 < threshold <= 1.0:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "success_rate_threshold must be in (0, 1]",
            details={"success_rate_threshold": threshold},
        )
    if minimum < 1:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "min_trials must be >= 1",
            details={"min_trials": minimum},
        )

    existing = session.scalars(
        select(GapAnalysis).where(GapAnalysis.evaluation_report_id == report.id)
    ).first()
    if existing is not None:
        return existing

    results = (report.summary_json or {}).get("results") or []
    orders = compute_gap_orders(
        results, success_rate_threshold=threshold, min_trials=minimum
    )
    summary = gap_summary(orders, success_rate_threshold=threshold, min_trials=minimum)

    analysis = GapAnalysis(
        id=new_id(),
        evaluation_report_id=report.id,
        state=AnalysisState.SUCCEEDED.value,
        threshold=threshold,
        min_trials=minimum,
        summary_json=summary,
        created_at=utcnow(),
        finished_at=utcnow(),
    )
    try:
        with session.begin_nested():
            session.add(analysis)
            session.flush()
    except IntegrityError:
        # Concurrent identical request won the unique constraint; return it.
        existing = session.scalars(
            select(GapAnalysis).where(GapAnalysis.evaluation_report_id == report.id)
        ).one()
        return existing

    analysis.results_asset_id = _store_gap_analysis_asset(
        session, evaluation_id=report.id, analysis_id=analysis.id, summary=summary
    )
    _ensure_orders(session, report_id=report.id, orders=orders, actor=actor)
    session.flush()
    record_audit(
        session,
        actor=actor,
        action="gap_analysis.run",
        entity_type="gap_analysis",
        entity_id=analysis.id,
        after={
            "evaluation_report_id": report.id,
            "order_count": summary["order_count"],
            "requested_total": summary["total_requested_count"],
        },
        request_id=request_id,
    )
    return analysis


def _store_gap_analysis_asset(
    session: Session,
    *,
    evaluation_id: str,
    analysis_id: str,
    summary: Mapping[str, Any],
) -> str:
    """Persist the deterministic gap-analysis evidence asset (V2 results_asset_id)."""
    body = canonical_json_bytes(dict(summary))
    object_key = GAP_ANALYSIS_ASSET_KEY.format(evaluation_id=evaluation_id)
    stored = get_storage().put_bytes(object_key, body, content_type="application/json")
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or "",
        byte_size=len(body),
        media_type="application/json",
        object_key=object_key,
        origin="report",
        logical_name="gap_analysis.json",
        meta_json={
            "kind": "gap_analysis",
            "evaluation_report_id": evaluation_id,
            "gap_analysis_id": analysis_id,
        },
    )
    session.add(asset)
    session.flush()
    return asset.id


def _source_import_links_order(session: Session, episode: Episode, order_id: str) -> bool:
    source = session.get(SourceImport, episode.source_import_id)
    return source is not None and source.collection_order_id == order_id


def _has_approved_annotation(session: Session, episode_id: str) -> bool:
    statement = select(AnnotationSet.id).where(
        AnnotationSet.episode_id == episode_id,
        or_(
            AnnotationSet.status == "APPROVED",
            AnnotationSet.approved_revision_id.is_not(None),
        ),
    )
    return session.scalars(statement).first() is not None


def _ensure_orders(
    session: Session,
    *,
    report_id: str,
    orders: list[dict[str, Any]],
    actor: Principal | None = None,
) -> None:
    """Insert the PROPOSED collection orders for a report, tolerating a concurrent run."""
    report = session.get(EvaluationReport, report_id)
    is_simulated = bool((report.summary_json or {}).get("is_simulated")) if report else True
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
                # SPEC §7.2: an analysis only *proposes*; it never confirms a root cause.
                state="PROPOSED",
                created_by=actor.key_id if actor is not None else None,
                is_simulated_source=is_simulated,
                created_at=utcnow(),
            )
        )
    session.flush()


def orders_for_evaluation(session: Session, evaluation_id: str) -> list[CollectionOrder]:
    statement = (
        select(CollectionOrder)
        .where(CollectionOrder.evaluation_report_id == evaluation_id)
        .order_by(CollectionOrder.scenario_key)
    )
    return list(session.scalars(statement))


# ---------------------------------------------------------------------------
# read API
# ---------------------------------------------------------------------------
def list_collection_orders(
    session: Session,
    *,
    limit: int = 20,
    cursor: str | None = None,
    filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Cursor-paginated collection-order listing (SPEC 9.1/9.4)."""
    if limit < 1:
        raise ApiError(ErrorCode.INVALID_INPUT, "limit must be >= 1")
    limit = min(int(limit), 100)
    filters = dict(filters or {})

    statement = select(CollectionOrder)
    evaluation_report_id = filters.get("evaluation_report_id")
    if evaluation_report_id:
        statement = statement.where(CollectionOrder.evaluation_report_id == evaluation_report_id)
    scenario_key = filters.get("scenario_key")
    if scenario_key:
        statement = statement.where(CollectionOrder.scenario_key == scenario_key)
    state = filters.get("state")
    if state:
        statement = statement.where(CollectionOrder.state == state)

    if cursor:
        created_at_text, entity_id = decode_cursor(cursor)
        try:
            cursor_time = datetime.fromisoformat(created_at_text)
        except ValueError as exc:  # pragma: no cover - encode_cursor never emits this
            raise ApiError(ErrorCode.INVALID_INPUT, "Invalid pagination cursor") from exc
        statement = statement.where(
            (CollectionOrder.created_at < cursor_time)
            | ((CollectionOrder.created_at == cursor_time) & (CollectionOrder.id < entity_id))
        )

    statement = statement.order_by(CollectionOrder.created_at.desc(), CollectionOrder.id.desc())
    rows = list(session.scalars(statement.limit(limit + 1)))
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = (
        encode_cursor(rows[-1].created_at, rows[-1].id) if has_more and rows else None
    )
    counts = _linked_counts(session, [row.id for row in rows])
    return {
        "data": [_serialize_order(row, linked_count=counts.get(row.id, 0)) for row in rows],
        "next_cursor": next_cursor,
    }


# ---------------------------------------------------------------------------
# collection-order lifecycle (V2 D2-6 / DO-A02 C6)
# ---------------------------------------------------------------------------
def get_collection_order(session: Session, order_id: str) -> dict[str, Any]:
    order = session.get(CollectionOrder, order_id)
    if order is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Collection order not found: {order_id}")
    report = session.get(EvaluationReport, order.evaluation_report_id)
    linked_rows = list(
        session.scalars(
            select(CollectionOrderEpisode)
            .where(CollectionOrderEpisode.order_id == order.id)
            .order_by(CollectionOrderEpisode.linked_at, CollectionOrderEpisode.episode_id)
        )
    )
    linked_episodes: list[dict[str, Any]] = []
    published_episode_ids = _published_episode_ids(session, [row.episode_id for row in linked_rows])
    for row in linked_rows:
        episode = session.get(Episode, row.episode_id)
        basis = (
            credible_scenario_basis(session, row.episode_id, order.scenario_key)
            if episode is not None
            else None
        )
        linked_episodes.append(
            {
                "episode_id": row.episode_id,
                "linked_by": row.linked_by,
                "linked_at": row.linked_at.isoformat() if row.linked_at else None,
                "quality_status": episode.quality_status if episode is not None else None,
                "scenario_basis": basis,
                "in_published_version": row.episode_id in published_episode_ids,
            }
        )
    evaluation = {
        "evaluation_id": report.id if report is not None else None,
        "dataset_version_id": report.dataset_version_id if report is not None else None,
        "is_simulated": bool((report.summary_json or {}).get("is_simulated"))
        if report is not None
        else bool(order.is_simulated_source),
        "model_id": (report.summary_json or {}).get("model_id") if report is not None else None,
        "schema_version": report.schema_version if report is not None else None,
    }
    return {
        "order": _serialize_order(order, linked_count=len(linked_rows)),
        "evaluation": evaluation,
        "is_simulated_source": bool(order.is_simulated_source),
        "linked_episodes": linked_episodes,
        "linked_count": len(linked_episodes),
        "requested_count": int(order.requested_count),
        "fulfilled": order.state == "FULFILLED",
    }


def approve_collection_order(
    session: Session,
    *,
    order_id: str,
    actor: Principal,
    note: str,
    request_id: str | None = None,
) -> CollectionOrder:
    """PROPOSED -> APPROVED by a reviewer who did not create the order."""
    order = session.get(CollectionOrder, order_id, with_for_update=True)
    if order is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Collection order not found: {order_id}")
    if order.state != "PROPOSED":
        raise ApiError(
            ErrorCode.CONFLICT,
            "Only a PROPOSED collection order can be approved",
            details={"order_id": order.id, "state": order.state},
        )
    if order.created_by is not None and order.created_by == actor.key_id:
        raise ApiError(
            ErrorCode.FORBIDDEN,
            "A reviewer cannot approve a collection order they created",
            details={"order_id": order.id},
        )
    cleaned = (note or "").strip()
    if not cleaned:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "An approval note is required",
            details={"order_id": order.id},
        )
    if len(cleaned) > 2000:
        raise ApiError(ErrorCode.VALIDATION_FAILED, "Approval note is too long")
    order.state = "APPROVED"
    order.approved_by = actor.key_id
    order.approved_at = utcnow()
    order.approval_note = cleaned
    order.updated_at = utcnow()
    session.flush()
    record_audit(
        session,
        actor=actor,
        action="collection_order.approve",
        entity_type="collection_order",
        entity_id=order.id,
        before={"state": "PROPOSED"},
        after={
            "state": "APPROVED",
            "note": cleaned,
            "is_simulated_source": bool(order.is_simulated_source),
        },
        request_id=request_id,
    )
    return order


def _require_linkable_order(order: CollectionOrder) -> None:
    if order.state not in {"APPROVED", "IN_PROGRESS"}:
        raise ApiError(
            ErrorCode.CONFLICT,
            "Episodes can only be linked to an APPROVED or IN_PROGRESS collection order",
            details={"order_id": order.id, "state": order.state},
        )


def _published_episode_ids(session: Session, episode_ids: Sequence[str]) -> set[str]:
    if not episode_ids:
        return set()
    statement = (
        select(DatasetVersionEpisode.episode_id)
        .join(DatasetVersion, DatasetVersion.id == DatasetVersionEpisode.version_id)
        .where(
            DatasetVersion.state == "PUBLISHED",
            DatasetVersionEpisode.episode_id.in_(episode_ids),
        )
    )
    return set(session.scalars(statement))


def _linked_episode_ids(session: Session, order_id: str) -> list[str]:
    return list(
        session.scalars(
            select(CollectionOrderEpisode.episode_id).where(
                CollectionOrderEpisode.order_id == order_id
            )
        )
    )


def _refresh_fulfillment(session: Session, order: CollectionOrder) -> str:
    linked = _linked_episode_ids(session, order.id)
    if not linked:
        return order.state
    published = _published_episode_ids(session, linked)
    if len(linked) >= int(order.requested_count) and set(linked).issubset(published):
        return "FULFILLED"
    return "IN_PROGRESS"


def _validate_link_candidate(session: Session, episode: Episode, order: CollectionOrder) -> str:
    if not _source_import_links_order(session, episode, order.id):
        raise ApiError(
            ErrorCode.CONFLICT,
            "Episode source import is not traceable to this collection order",
            details={"episode_id": episode.id, "order_id": order.id},
        )
    basis = credible_scenario_basis(session, episode.id, order.scenario_key)
    if basis is None:
        raise ApiError(
            ErrorCode.CONFLICT,
            "Episode has no credible scenario tag for this order's scenario_key",
            details={
                "episode_id": episode.id,
                "scenario_key": order.scenario_key,
                "accepted_bases": list(CREDIBLE_BASE_PRIORITY),
            },
        )
    if episode.quality_status not in {"PASS", "WARN"}:
        raise ApiError(
            ErrorCode.CONFLICT,
            "Episode quality verdict is not PASS/WARN",
            details={"episode_id": episode.id, "quality_status": episode.quality_status},
        )
    if not _has_approved_annotation(session, episode.id):
        raise ApiError(
            ErrorCode.CONFLICT,
            "Episode has no approved annotation revision/set",
            details={"episode_id": episode.id},
        )
    return basis


def link_collection_order_episodes(
    session: Session,
    *,
    order_id: str,
    episode_ids: Sequence[str],
    actor: Principal,
    request_id: str | None = None,
) -> CollectionOrder:
    """Link fulfillment Episodes to an approved order; recompute fulfillment."""
    order = session.get(CollectionOrder, order_id, with_for_update=True)
    if order is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Collection order not found: {order_id}")
    _require_linkable_order(order)
    ids = [str(value) for value in episode_ids]
    if not ids:
        raise ApiError(ErrorCode.VALIDATION_FAILED, "episode_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ApiError(ErrorCode.VALIDATION_FAILED, "episode_ids contains duplicates")
    if len(ids) > MAX_LINK_EPISODES:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            f"At most {MAX_LINK_EPISODES} episodes can be linked per request",
            details={"limit": MAX_LINK_EPISODES, "count": len(ids)},
        )

    linked_basis: dict[str, str] = {}
    for episode_id in ids:
        episode = session.get(Episode, episode_id)
        if episode is None:
            raise ApiError(ErrorCode.NOT_FOUND, f"Episode not found: {episode_id}")
        existing = session.scalars(
            select(CollectionOrderEpisode).where(
                CollectionOrderEpisode.episode_id == episode_id
            )
        ).first()
        if existing is not None:
            if existing.order_id == order.id:
                continue  # idempotent re-link
            raise ApiError(
                ErrorCode.CONFLICT,
                "Episode is already linked to another collection order",
                details={
                    "episode_id": episode_id,
                    "other_order_id": existing.order_id,
                    "order_id": order.id,
                },
            )
        linked_basis[episode_id] = _validate_link_candidate(session, episode, order)
        session.add(
            CollectionOrderEpisode(
                order_id=order.id,
                episode_id=episode_id,
                linked_by=actor.key_id,
                linked_at=utcnow(),
            )
        )
    session.flush()

    previous_state = order.state
    order.state = _refresh_fulfillment(session, order)
    order.updated_at = utcnow()
    session.flush()
    # DO-A02 C2: linked/fulfilled Episodes changed business state; bump their
    # business_revision in the same transaction (single call site for this order
    # lifecycle path).
    affected = _linked_episode_ids(session, order.id)
    revisions = bump_episode_revisions(session, affected)
    record_audit(
        session,
        actor=actor,
        action="collection_order.link_episodes",
        entity_type="collection_order",
        entity_id=order.id,
        before={"state": previous_state},
        after={
            "state": order.state,
            "episode_ids": sorted(linked_basis) or ids,
            "linked_count": len(affected),
            "requested_count": int(order.requested_count),
            "episode_business_revisions": revisions,
            "is_simulated_source": bool(order.is_simulated_source),
        },
        request_id=request_id,
    )
    return order


def cancel_collection_order(
    session: Session,
    *,
    order_id: str,
    actor: Principal,
    reason: str,
    request_id: str | None = None,
) -> CollectionOrder:
    """PROPOSED|APPROVED|IN_PROGRESS -> CANCELLED (terminal, audited)."""
    order = session.get(CollectionOrder, order_id, with_for_update=True)
    if order is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Collection order not found: {order_id}")
    if order.state in COLLECTION_ORDER_TERMINAL_STATES:
        raise ApiError(
            ErrorCode.CONFLICT,
            "Terminal collection orders cannot be cancelled",
            details={"order_id": order.id, "state": order.state},
        )
    cleaned = (reason or "").strip()
    if not cleaned:
        raise ApiError(ErrorCode.VALIDATION_FAILED, "A cancellation reason is required")
    previous_state = order.state
    order.state = "CANCELLED"
    order.updated_at = utcnow()
    session.flush()
    affected = _linked_episode_ids(session, order.id)
    revisions = bump_episode_revisions(session, affected)
    record_audit(
        session,
        actor=actor,
        action="collection_order.cancel",
        entity_type="collection_order",
        entity_id=order.id,
        before={"state": previous_state},
        after={
            "state": "CANCELLED",
            "reason": cleaned,
            "episode_business_revisions": revisions,
        },
        request_id=request_id,
    )
    return order
