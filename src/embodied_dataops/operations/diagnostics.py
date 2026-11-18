"""Bounded read-only diagnostics required by DO-A02 C5.

All three queries are evidence-only: they read normalized data (parquet indices,
task rows, quality reports) and never expose object bytes, presigned URLs, host
secrets or stack traces.  Every response declares ``truncated``/limits explicitly.
"""

from __future__ import annotations

import hashlib
import io
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow.parquet as pq
from sqlalchemy import Text, case, func, select
from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import (
    Asset,
    Episode,
    Modality,
    PipelineRun,
    QualityFinding,
    QualityReport,
    SourceImport,
    Task,
    TaskDependency,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.operations.receipts import iso_utc
from embodied_dataops.storage.base import get_storage

logger = logging.getLogger(__name__)

#: C5 hard bounds.
MAX_TIMELINE_WINDOW_NS = 10_000_000_000
MAX_TIMELINE_STREAMS = 8
MAX_TIMELINE_POINTS = 200
MAX_FAILED_TASKS = 30
MAX_FAILURE_EVENTS = 50
MAX_AGGREGATE_WINDOW_DAYS = 31
MAX_AGGREGATE_LIMIT = 50

AGGREGATE_GROUP_BY = ("source_type", "profile_id", "rule_code", "collection_id")


# ---------------------------------------------------------------------------
# 1. GET /episodes/{id}/timeline
# ---------------------------------------------------------------------------
def episode_timeline(
    session: Session,
    episode: Episode,
    *,
    start_ns: int,
    end_ns: int,
    max_points: int,
) -> dict[str, Any]:
    if not (0 <= start_ns < end_ns):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "start_ns must be >= 0 and end_ns must be greater than start_ns",
            status_code=422,
            details={"start_ns": start_ns, "end_ns": end_ns},
        )
    if end_ns > int(episode.duration_ns):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "end_ns must not exceed the episode duration",
            status_code=422,
            details={"end_ns": end_ns, "duration_ns": episode.duration_ns},
        )
    if end_ns - start_ns > MAX_TIMELINE_WINDOW_NS:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "the requested timeline window is longer than 10 seconds",
            status_code=422,
            details={"start_ns": start_ns, "end_ns": end_ns, "max_window_ns": MAX_TIMELINE_WINDOW_NS},
        )
    if not 1 <= max_points <= MAX_TIMELINE_POINTS:  # pragma: no cover - route validates
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "max_points must be between 1 and 200",
            status_code=422,
        )

    modalities = list(
        session.scalars(
            select(Modality)
            .where(Modality.episode_id == episode.id)
            .order_by(Modality.stream_index, Modality.name)
        )
    )
    if len(modalities) > MAX_TIMELINE_STREAMS:
        raise ApiError(
            ErrorCode.TOO_MANY_MODALITIES,
            "The episode has more streams than the diagnostic timeline contract allows",
            details={"modality_count": len(modalities), "max_streams": MAX_TIMELINE_STREAMS},
        )

    storage = get_storage()
    streams: list[dict[str, Any]] = []
    for modality in modalities:
        asset = _modality_asset(session, modality)
        evidence = _stream_evidence(asset, storage, start_ns, end_ns, max_points)
        streams.append(
            {
                "name": modality.name,
                "kind": modality.kind,
                "clock_domain": modality.clock_domain,
                "sample_count": int(modality.sample_count or 0),
                **evidence,
            }
        )
    digest_source = {
        "episode_id": episode.id,
        "window": {"start_ns": start_ns, "end_ns": end_ns},
        "streams": [
            {"name": stream["name"], "points": stream["points"], "truncated": stream["truncated"]}
            for stream in streams
        ],
    }
    evidence_digest = hashlib.sha256(canonical_json_bytes(digest_source)).hexdigest()
    return {
        "episode_id": episode.id,
        "profile_id": episode.profile_id or "unknown",
        "window": {"start_ns": start_ns, "end_ns": end_ns},
        "streams": streams,
        "evidence_digest": evidence_digest,
    }


def _modality_asset(session: Session, modality: Modality) -> Asset | None:
    for asset_id in (modality.native_index_asset_id, modality.aligned_index_asset_id):
        if asset_id:
            asset = session.get(Asset, asset_id)
            if asset is not None:
                return asset
    if modality.object_key:
        return session.scalars(select(Asset).where(Asset.object_key == modality.object_key)).first()
    return None


_EMPTY_EVIDENCE: dict[str, Any] = {
    "observed_gap_count": 0,
    "max_gap_ns": 0,
    "p95_alignment_error_ns": 0,
    "points": [],
    "truncated": False,
}


def _stream_evidence(
    asset: Asset | None,
    storage: Any,
    start_ns: int,
    end_ns: int,
    max_points: int,
) -> dict[str, Any]:
    if asset is None:
        return dict(_EMPTY_EVIDENCE)
    try:
        table = pq.read_table(io.BytesIO(storage.get_bytes(asset.object_key)))
    except Exception:  # noqa: BLE001 - missing/corrupt index is explicit empty evidence
        logger.warning("timeline.index_unreadable asset_id=%s", asset.id, exc_info=True)
        return dict(_EMPTY_EVIDENCE)
    if "t_ns" not in table.column_names:
        return dict(_EMPTY_EVIDENCE)
    t_values = [int(value) for value in table.column("t_ns").to_pylist() if value is not None]
    source_values = _source_times(table, len(t_values))
    valid_values = _valid_flags(table, len(t_values))

    rows = [
        index
        for index, time_ns in enumerate(t_values)
        if start_ns <= time_ns <= end_ns
    ]
    if not rows:
        return dict(_EMPTY_EVIDENCE)
    window_times = [t_values[index] for index in rows]
    deltas = [window_times[i] - window_times[i - 1] for i in range(1, len(window_times))]
    max_gap = max(deltas) if deltas else 0
    if deltas:
        ordered = sorted(deltas)
        median_delta = ordered[len(ordered) // 2]
        observed_gaps = sum(1 for delta in deltas if median_delta > 0 and delta > 1.5 * median_delta)
    else:
        observed_gaps = 0
    if source_values is None:
        alignment_errors = [0] * len(rows)
    else:
        alignment_errors = [abs(window_times[i] - source_values[rows[i]]) for i in range(len(rows))]
    p95 = _p95(alignment_errors)

    selected = _evenly_spaced(len(rows), max_points)
    points = []
    for position in selected:
        index = rows[position]
        points.append(
            {
                "t_ns": t_values[index],
                "source_t_ns": int(source_values[index]) if source_values is not None else t_values[index],
                "valid": bool(valid_values[index]) if valid_values is not None else True,
            }
        )
    return {
        "observed_gap_count": int(observed_gaps),
        "max_gap_ns": int(max_gap),
        "p95_alignment_error_ns": int(p95),
        "points": points,
        "truncated": len(rows) > len(points),
    }


def _source_times(table: Any, count: int) -> list[int] | None:
    for name in ("source_t_ns", "state_source_t_ns", "action_source_t_ns"):
        if name in table.column_names:
            values = table.column(name).to_pylist()
            if len(values) == count:
                return [int(value) for value in values]
    return None


def _valid_flags(table: Any, count: int) -> list[bool] | None:
    if "valid" not in table.column_names:
        return None
    values = table.column("valid").to_pylist()
    if len(values) != count:
        return None
    return [bool(value) for value in values]


def _evenly_spaced(count: int, limit: int) -> list[int]:
    """Deterministic equally spaced indices, always retaining the first and last."""
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    step = (count - 1) / (limit - 1)
    indexes = sorted({round(i * step) for i in range(limit)})
    return indexes


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return int(ordered[rank])


# ---------------------------------------------------------------------------
# 2. GET /pipeline-runs/{id}/failure-analysis
# ---------------------------------------------------------------------------
def failure_analysis(session: Session, run_id: str) -> dict[str, Any]:
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise ApiError(ErrorCode.NOT_FOUND, "PipelineRun not found", details={"run_id": run_id})
    tasks = list(session.scalars(select(Task).where(Task.run_id == run_id).order_by(Task.created_at, Task.id)))
    failed = [task for task in tasks if task.state == "FAILED"]
    if len(failed) > MAX_FAILED_TASKS:
        raise ApiError(
            ErrorCode.DIAGNOSTIC_RESULT_TOO_LARGE,
            "Too many failed tasks for the bounded failure-analysis response",
            details={"failed_task_count": len(failed), "max": MAX_FAILED_TASKS},
        )
    failed_tasks = [
        {
            "task_id": task.id,
            "step": task.step,
            "error_code": task.error_code or "UNKNOWN",
            "attempt": int(task.attempt),
            "worker_pool": task.pool,
            "first_failed_at": iso_utc(task.created_at),
            "last_failed_at": iso_utc(task.updated_at or task.created_at),
        }
        for task in failed
    ]
    dependencies = [
        {"task_id": dependency.task_id, "depends_on_task_id": dependency.depends_on_task_id, "state": _task_state(session, dependency.depends_on_task_id)}
        for dependency in session.scalars(
            select(TaskDependency).where(TaskDependency.task_id.in_([task.id for task in tasks]))
        )
    ] if tasks else []
    if len(dependencies) > MAX_FAILURE_EVENTS:
        raise ApiError(
            ErrorCode.DIAGNOSTIC_RESULT_TOO_LARGE,
            "Too many dependency edges for the bounded failure-analysis response",
            details={"dependency_count": len(dependencies), "max": MAX_FAILURE_EVENTS},
        )
    incidents = [
        {
            "incident_code": task.error_code or "UNKNOWN",
            "at": iso_utc(task.updated_at or task.created_at),
            "worker_pool": task.pool,
        }
        for task in failed
    ]
    if len(incidents) > MAX_FAILURE_EVENTS:
        raise ApiError(
            ErrorCode.DIAGNOSTIC_RESULT_TOO_LARGE,
            "Too many worker incidents for the bounded failure-analysis response",
            details={"incident_count": len(incidents), "max": MAX_FAILURE_EVENTS},
        )
    rule_codes = sorted({task.error_code for task in failed if task.error_code})
    return {
        "run_id": run.id,
        "failed_tasks": failed_tasks,
        "dependencies": dependencies,
        "worker_incidents": incidents,
        "summary_rule_codes": rule_codes,
        "truncated": False,
    }


def _task_state(session: Session, task_id: str) -> str:
    state = session.scalar(select(Task.state).where(Task.id == task_id))
    return state or "UNKNOWN"


# ---------------------------------------------------------------------------
# 3. GET /quality/aggregate
# ---------------------------------------------------------------------------
def parse_rfc3339(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            f"{field} must be an RFC3339 timestamp",
            status_code=422,
            details={"field": field},
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def quality_aggregate(
    session: Session,
    *,
    from_time: datetime,
    to_time: datetime,
    group_by: str,
    limit: int,
) -> dict[str, Any]:
    if group_by not in AGGREGATE_GROUP_BY:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            f"group_by must be one of {list(AGGREGATE_GROUP_BY)}",
            status_code=422,
            details={"group_by": group_by},
        )
    if from_time >= to_time:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "from must be earlier than to",
            status_code=422,
            details={"from": iso_utc(from_time), "to": iso_utc(to_time)},
        )
    if to_time - from_time > timedelta(days=MAX_AGGREGATE_WINDOW_DAYS):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            f"the aggregate window must be at most {MAX_AGGREGATE_WINDOW_DAYS} days",
            status_code=422,
            details={"from": iso_utc(from_time), "to": iso_utc(to_time), "max_days": MAX_AGGREGATE_WINDOW_DAYS},
        )
    if not 1 <= limit <= MAX_AGGREGATE_LIMIT:  # pragma: no cover - route validates
        raise ApiError(ErrorCode.INVALID_INPUT, "limit must be between 1 and 50", status_code=422)

    # "latest completed quality report per episode as of computed_at"
    latest = (
        select(
            QualityReport.episode_id.label("episode_id"),
            QualityReport.id.label("report_id"),
            QualityReport.status.label("status"),
        )
        .where(QualityReport.created_at >= from_time, QualityReport.created_at <= to_time)
        .distinct(QualityReport.episode_id)
        .order_by(QualityReport.episode_id, QualityReport.created_at.desc(), QualityReport.id.desc())
        .subquery()
    )

    matched_episode_count = int(
        session.scalar(select(func.count()).select_from(latest)) or 0
    )

    key_column = _aggregate_key(group_by)
    if group_by == "rule_code":
        episode_count = func.count(func.distinct(latest.c.episode_id))
        stmt = (
            select(
                QualityFinding.rule_code.label("key"),
                episode_count.label("episode_count"),
                func.count(
                    func.distinct(case((latest.c.status == "FAIL", latest.c.episode_id)))
                ).label("fail_episode_count"),
                func.count(
                    func.distinct(case((latest.c.status == "WARN", latest.c.episode_id)))
                ).label("warn_episode_count"),
            )
            .select_from(latest)
            .join(QualityFinding, QualityFinding.report_id == latest.c.report_id)
            .group_by(QualityFinding.rule_code)
        )
        stmt = stmt.order_by(episode_count.desc(), QualityFinding.rule_code.asc())
    else:
        episode_count = func.count()
        stmt = (
            select(
                key_column.label("key"),
                episode_count.label("episode_count"),
                func.count(case((latest.c.status == "FAIL", 1))).label("fail_episode_count"),
                func.count(case((latest.c.status == "WARN", 1))).label("warn_episode_count"),
            )
            .select_from(latest)
            .join(Episode, Episode.id == latest.c.episode_id)
            .join(SourceImport, SourceImport.id == Episode.source_import_id)
            .group_by(key_column)
        )
        stmt = stmt.order_by(episode_count.desc(), key_column.asc())
    stmt = stmt.limit(limit + 1)
    rows = list(session.execute(stmt))
    truncated = len(rows) > limit
    groups = [
        {
            "key": str(row.key),
            "episode_count": int(row.episode_count),
            "fail_episode_count": int(row.fail_episode_count),
            "warn_episode_count": int(row.warn_episode_count),
        }
        for row in rows[:limit]
    ]
    return {
        "matched_episode_count": matched_episode_count,
        "group_by": group_by,
        "groups": groups,
        "truncated": truncated,
        "computed_at": iso_utc(datetime.now(UTC)),
        "filter_window": {"from": iso_utc(from_time), "to": iso_utc(to_time)},
        "statistical_basis": (
            "latest completed quality report per episode within the window, "
            "aggregated over the whole matching population before top-k truncation"
        ),
    }


def _aggregate_key(group_by: str):
    if group_by == "source_type":
        return SourceImport.source_type
    if group_by == "profile_id":
        return func.coalesce(Episode.profile_id, "unknown")
    if group_by == "collection_id":
        return func.coalesce(func.cast(SourceImport.collection_order_id, Text), "unassigned")
    return QualityFinding.rule_code


__all__ = [
    "AGGREGATE_GROUP_BY",
    "MAX_AGGREGATE_LIMIT",
    "MAX_AGGREGATE_WINDOW_DAYS",
    "MAX_FAILED_TASKS",
    "MAX_FAILURE_EVENTS",
    "MAX_TIMELINE_POINTS",
    "MAX_TIMELINE_STREAMS",
    "MAX_TIMELINE_WINDOW_NS",
    "episode_timeline",
    "failure_analysis",
    "parse_rfc3339",
    "quality_aggregate",
]
