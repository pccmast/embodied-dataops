"""Build, merge and persist one ``quality-report/2.0`` (V2 SPEC §3.2, DO2-13).

Persistence contract
--------------------
* the immutable report artefact is written to
  ``quality/{episode_id}/{run_id}/quality-report-2.0.json`` and registered as an
  ``assets`` row with ``origin=report`` (the V1 ``quality.json`` layout is never
  reused or overwritten);
* exactly one ``quality_reports`` row exists per ``(episode_id, run_id)``; a V1
  row is **never** recomputed or overwritten in place -- attempting to persist a
  2.0 report over a 1.0 report raises ``CONFLICT``;
* the report row binds ``profile_id``/``profile_revision``/``config_sha256`` (the
  *profile* config SHA), ``schema_version=quality-report/2.0`` and the policy
  config SHA inside ``summary_json``;
* findings keep the V1 columns (``observed``/``limit``/``evidence``) and add
  ``source_stream``/``source_clock_domain``/``method``/``limits``/
  ``uncertainty_note``;
* duplicate findings of the same rule with overlapping/touching time ranges are
  merged (max observed preserved).  A single Episode may store at most 10,000
  finding rows; exceeding that raises ``QUALITY_REPORT_TOO_LARGE`` (413) and the
  caller records the complete raw diagnostics as an immutable Asset -- the
  publish decision is failed, never silently truncated;
* ``episodes.business_revision`` is bumped exactly once per committed quality
  state change (DO-A02 C2, single :func:`bump` call site).

The handler runs inside the engine transaction: this module never commits,
rolls back or touches ``tasks.state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import Asset, Episode, QualityFinding, QualityReport
from embodied_dataops.domain.enums import AssetOrigin, EpisodeStatus, QualityStatus
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.observability.metrics import EPISODE_QC
from embodied_dataops.operations.revisions import bump
from embodied_dataops.quality.rules import TimeRange
from embodied_dataops.quality.v2_policy import (
    POLICY_CONFIG_SHA256,
    QC_POLICY_V2_VERSION,
    QUALITY_REPORT_V2_SCHEMA_VERSION,
    THRESHOLDS_V2,
)
from embodied_dataops.quality.v2_rules import SemanticEvaluation, SemanticFinding, SemanticQcInput
from embodied_dataops.storage.base import sha256_bytes
from embodied_dataops.workflow.handlers.base import HandlerContext, register_json
from embodied_dataops.workflow.outbox import utcnow

#: Finding-row limit for one Episode (SPEC §3.2).
MAX_FINDING_ROWS = THRESHOLDS_V2.max_findings_per_report
#: Maximum time ranges kept on one merged finding row.
MAX_MERGED_RANGES = THRESHOLDS_V2.max_merged_ranges_per_finding
#: Maximum evidence items merged into one list.
MAX_MERGED_EVIDENCE_ITEMS = 200

#: Rule code used for the "report too large" diagnostics finding.
TOO_LARGE_RULE_CODE = "QUALITY_REPORT_TOO_LARGE"

__all__ = [
    "MAX_FINDING_ROWS",
    "MergeResult",
    "PreparedReport",
    "QualityReportTooLarge",
    "build_raw_diagnostics_payload",
    "merge_semantic_findings",
    "persist_quality_report_too_large",
    "persist_quality_report_v2",
    "prepare_quality_report_v2",
    "quality_diagnostics_v2_key",
    "quality_report_v2_key",
]


def quality_report_v2_key(episode_id: str, run_id: str) -> str:
    """Deterministic immutable object key for one run's quality-report/2.0."""
    return f"quality/{episode_id}/{run_id}/quality-report-2.0.json"


def quality_diagnostics_v2_key(episode_id: str, run_id: str) -> str:
    """Object key of the raw diagnostics recorded when the row cap is exceeded."""
    return f"quality/{episode_id}/{run_id}/diagnostics-too-large.json"


def _utc_isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class MergeStats:
    """Deterministic counters of one finding merge pass."""

    raw_finding_count: int
    merged_row_count: int
    merged_group_count: int
    limit: int
    max_ranges_per_row: int

    @property
    def overflowed(self) -> bool:
        return self.merged_row_count > self.limit

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_finding_count": self.raw_finding_count,
            "merged_finding_row_count": self.merged_row_count,
            "merged_group_count": self.merged_group_count,
            "max_finding_rows": self.limit,
            "max_ranges_per_row": self.max_ranges_per_row,
        }


class QualityReportTooLarge(ApiError):
    """Raised when one Episode would exceed the 10,000 finding-row limit."""

    def __init__(self, stats: MergeStats) -> None:
        super().__init__(
            ErrorCode.QUALITY_REPORT_TOO_LARGE,
            "quality report exceeds the finding-row limit; raw diagnostics are recorded "
            "and the Episode cannot pass the publish gate",
            details=stats.as_dict(),
        )
        self.stats = stats


@dataclass
class MergeResult:
    findings: list[SemanticFinding]
    stats: MergeStats


@dataclass
class PreparedReport:
    """A merged, serialisable ``quality-report/2.0`` payload."""

    payload: dict[str, Any]
    findings: list[SemanticFinding]
    stats: MergeStats


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _merge_values(first: Any, second: Any) -> Any:
    """Merge two observed/evidence values, preserving the maximum numeric value."""
    if isinstance(first, dict) and isinstance(second, dict):
        merged = dict(first)
        for key, value in second.items():
            merged[key] = _merge_values(merged[key], value) if key in merged else value
        return merged
    if _is_number(first) and _is_number(second):
        return max(first, second)
    if isinstance(first, list) and isinstance(second, list):
        combined = list(first)
        for item in second:
            if item not in combined:
                combined.append(item)
        return combined[:MAX_MERGED_EVIDENCE_ITEMS]
    if first is None:
        return second
    return first


def _merge_ranges(ranges: list[TimeRange]) -> tuple[TimeRange, ...]:
    ordered = sorted(ranges, key=lambda item: (item.start_ns, item.end_ns))
    merged: list[TimeRange] = []
    for item in ordered:
        if merged and item.start_ns <= merged[-1].end_ns:
            previous = merged[-1]
            merged[-1] = TimeRange(previous.start_ns, max(previous.end_ns, item.end_ns))
        else:
            merged.append(TimeRange(item.start_ns, item.end_ns))
    return tuple(merged)


def _has_overlap(first: list[TimeRange], second: list[TimeRange]) -> bool:
    if not first or not second:
        return not first and not second
    first_start = min(item.start_ns for item in first)
    first_end = max(item.end_ns for item in first)
    second_start = min(item.start_ns for item in second)
    second_end = max(item.end_ns for item in second)
    return second_start <= first_end and second_end >= first_start


def _merge_finding_rows(rows: list[SemanticFinding], finding: SemanticFinding) -> None:
    """Merge ``finding`` into the last row of its group (max observed preserved)."""
    target = rows[-1]
    merged_observed = _merge_values(target.observed, finding.observed)
    merged_evidence = _merge_values(target.evidence, finding.evidence)
    merged_limits = _merge_values(target.limits, finding.limits)
    merged_count = int(merged_evidence.get("merged_from_finding_count", 1)) + int(
        finding.evidence.get("merged_from_finding_count", 1)
    )
    merged_evidence["merged_from_finding_count"] = merged_count
    rows[-1] = SemanticFinding(
        rule_code=target.rule_code,
        severity=target.severity,
        observed=merged_observed,
        threshold=target.threshold,
        affected_time_ranges=_merge_ranges(
            list(target.affected_time_ranges) + list(finding.affected_time_ranges)
        ),
        evidence=merged_evidence,
        source_stream=target.source_stream,
        source_clock_domain=target.source_clock_domain,
        method=target.method,
        limits=merged_limits,
        uncertainty_note=target.uncertainty_note or finding.uncertainty_note,
    )


def merge_semantic_findings(
    findings: list[SemanticFinding],
    *,
    limit: int = MAX_FINDING_ROWS,
    max_ranges: int = MAX_MERGED_RANGES,
) -> MergeResult:
    """Deduplicate/merge findings of the same rule and report the row count.

    Only findings whose time ranges overlap or touch are merged; disjoint
    occurrences of the same rule stay separate rows.  Merging never drops the
    maximum ``observed`` value: numeric leaves take the maximum, lists keep every
    distinct item (bounded) and ``merged_from_finding_count`` records the merge.
    """
    groups: dict[tuple[str, str, str, str], list[SemanticFinding]] = {}
    for finding in findings:
        key = (
            finding.rule_code,
            finding.severity.value,
            finding.source_stream or "",
            finding.source_clock_domain or "",
        )
        groups.setdefault(key, []).append(finding)

    merged_rows: list[SemanticFinding] = []
    for key in sorted(groups):
        group = groups[key]
        ordered = sorted(
            group,
            key=lambda item: (
                min((rng.start_ns for rng in item.affected_time_ranges), default=-1),
                max((rng.end_ns for rng in item.affected_time_ranges), default=-1),
                item.rule_code,
            ),
        )
        rows: list[SemanticFinding] = []
        for finding in ordered:
            if rows and _has_overlap(
                list(rows[-1].affected_time_ranges), list(finding.affected_time_ranges)
            ):
                _merge_finding_rows(rows, finding)
                continue
            rows.append(finding)
        merged_rows.extend(rows)

    stats = MergeStats(
        raw_finding_count=len(findings),
        merged_row_count=len(merged_rows),
        merged_group_count=len(groups),
        limit=limit,
        max_ranges_per_row=max_ranges,
    )
    return MergeResult(findings=merged_rows, stats=stats)


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------


def _profile_payload(input: SemanticQcInput) -> dict[str, Any]:
    binding = input.profile
    config = binding.config
    return {
        "profile_id": binding.profile_id,
        "profile_revision": binding.profile_revision,
        "authority": binding.authority,
        "config_sha256": binding.config_sha256,
        "mode": binding.mode,
        "binding_verified": binding.binding_verified,
        "state_semantics": config.state_semantics if config else None,
        "action_semantics": config.action_semantics if config else None,
        "state_dim": config.state_dim if config else None,
        "action_dim": config.action_dim if config else None,
        "missing_fields": list(binding.missing_fields),
    }


def prepare_quality_report_v2(
    *,
    episode_id: str,
    run_id: str,
    evaluation: SemanticEvaluation,
    input: SemanticQcInput,
    canonical_version: int,
    generated_at: datetime | None = None,
) -> PreparedReport:
    """Merge findings and assemble the ``quality-report/2.0`` payload.

    Raises :class:`QualityReportTooLarge` when the merged row count exceeds the
    Episodes' limit; the caller persists the raw diagnostics Asset and a FAIL
    report instead of truncating the publish decision.
    """
    merge = merge_semantic_findings(evaluation.findings)
    if merge.stats.overflowed:
        raise QualityReportTooLarge(merge.stats)
    payload: dict[str, Any] = {
        "schema_version": QUALITY_REPORT_V2_SCHEMA_VERSION,
        "episode_id": episode_id,
        "run_id": run_id,
        "canonical_version": int(canonical_version),
        "policy_version": input.policy_version or QC_POLICY_V2_VERSION,
        "policy_config_sha256": input.policy_config_sha256 or POLICY_CONFIG_SHA256,
        "profile": _profile_payload(input),
        "status": evaluation.verdict.value,
        "findings": [finding.as_dict() for finding in merge.findings],
        "metrics": {
            **evaluation.metrics,
            "report": merge.stats.as_dict(),
        },
        "checks": {
            "run": list(evaluation.checks_run),
            "unavailable": list(evaluation.checks_unavailable),
            "not_applicable": list(evaluation.checks_not_applicable),
        },
        "source": {
            "episode_type": input.episode_type,
            "source_type": input.source_type,
            "data_origin": input.data_origin,
            "normalization_valid": bool(input.normalization_valid),
            "robot_profile": input.profile.profile_id or "",
            "canonical_schema_version": input.schema_version,
            "canonical_missing_objects": list(input.extra.get("canonical_missing_objects", [])),
        },
    }
    if generated_at is not None:
        payload["generated_at"] = _utc_isoformat(generated_at)
    return PreparedReport(payload=payload, findings=merge.findings, stats=merge.stats)


def build_raw_diagnostics_payload(
    *,
    episode_id: str,
    run_id: str,
    evaluation: SemanticEvaluation,
    input: SemanticQcInput,
    canonical_version: int,
    error: QualityReportTooLarge,
) -> dict[str, Any]:
    """Complete, untruncated diagnostics recorded when the row cap is exceeded."""
    return {
        "schema_version": "quality-diagnostics/1.0",
        "episode_id": episode_id,
        "run_id": run_id,
        "canonical_version": int(canonical_version),
        "policy_version": input.policy_version,
        "policy_config_sha256": input.policy_config_sha256,
        "profile": _profile_payload(input),
        "error": {
            "code": ErrorCode.QUALITY_REPORT_TOO_LARGE.value,
            "message": str(error.message),
            "details": dict(error.details),
        },
        "status": evaluation.verdict.value,
        "checks": {
            "run": list(evaluation.checks_run),
            "unavailable": list(evaluation.checks_unavailable),
            "not_applicable": list(evaluation.checks_not_applicable),
        },
        "metrics": evaluation.metrics,
        "raw_findings": [finding.as_dict() for finding in evaluation.findings],
    }


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _get_or_create_asset(
    ctx: HandlerContext,
    *,
    key: str,
    data: bytes,
    payload: dict[str, Any],
    logical_name: str,
) -> Asset:
    """Register the immutable report asset, tolerating a re-execution of the same run."""
    session: Session = ctx.session
    existing = session.execute(select(Asset).where(Asset.object_key == key)).scalar_one_or_none()
    digest = sha256_bytes(data)
    if existing is not None:
        if existing.sha256 != digest:
            stored = ctx.storage.put_bytes(key, data, content_type="application/json")
            existing.sha256 = stored.sha256 or digest
            existing.byte_size = len(data)
            existing.media_type = "application/json"
            existing.meta_json = {
                "policy_version": payload.get("policy_version"),
                "status": payload.get("status"),
                "schema_version": payload.get("schema_version"),
            }
            session.flush()
        return existing
    return register_json(
        ctx,
        key=key,
        payload=payload,
        media_type="application/json",
        origin=AssetOrigin.REPORT,
        logical_name=logical_name,
        meta={
            "policy_version": payload.get("policy_version"),
            "status": payload.get("status"),
            "schema_version": payload.get("schema_version"),
            "canonical_version": payload.get("canonical_version"),
        },
    )


def _existing_report(session: Session, episode_id: str, run_id: str) -> QualityReport | None:
    return session.execute(
        select(QualityReport).where(
            QualityReport.episode_id == episode_id, QualityReport.run_id == run_id
        )
    ).scalar_one_or_none()


def _refuse_v1_overwrite(existing: QualityReport | None) -> None:
    if existing is not None and (existing.schema_version or "quality-report/1.0") != (
        QUALITY_REPORT_V2_SCHEMA_VERSION
    ):
        raise ApiError(
            ErrorCode.CONFLICT,
            "refusing to recompute or overwrite a qc-policy/1.0 quality report in place",
            details={
                "existing_schema_version": existing.schema_version,
                "existing_policy_version": existing.policy_version,
                "required": "a new PipelineRun for pipeline_version 2.0.0",
            },
        )


def _apply_episode_quality_state(
    session: Session,
    *,
    episode: Episode,
    verdict: str,
    previous_quality_status: str | None,
) -> None:
    """Apply the quality state transition and bump business_revision once."""
    episode.quality_status = verdict
    if verdict == QualityStatus.FAIL.value:
        episode.status = EpisodeStatus.QUALITY_FAILED.value
    elif episode.status in {EpisodeStatus.IMPORTED.value, EpisodeStatus.PROCESSING.value}:
        episode.status = EpisodeStatus.PROCESSING.value
    # DO-A02 C2: a quality pointer/status change is a business precondition change.
    if previous_quality_status != verdict:
        bump(session, episode)
    else:
        episode.updated_at = utcnow()
        session.flush()


def _summary_common(
    *,
    payload: dict[str, Any],
    evaluation: SemanticEvaluation,
    input: SemanticQcInput,
    canonical_version: int,
    stats: MergeStats,
    report_key: str | None,
    report_sha256: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": QUALITY_REPORT_V2_SCHEMA_VERSION,
        "policy_version": input.policy_version,
        "policy_config_sha256": input.policy_config_sha256,
        "canonical_version": int(canonical_version),
        "profile_id": input.profile.profile_id,
        "profile_revision": input.profile.profile_revision,
        "config_sha256": input.profile.config_sha256,
        "metrics": evaluation.metrics,
        "finding_count": stats.merged_row_count,
        "raw_finding_count": stats.raw_finding_count,
        "finding_counts": evaluation.counts_by_severity(),
        "rule_codes": [finding.rule_code for finding in evaluation.findings],
        "report_key": report_key,
        "report_sha256": report_sha256,
        "checks": payload.get("checks", {}),
        "normalization_valid": bool(input.normalization_valid),
        "canonical_missing_objects": list(input.extra.get("canonical_missing_objects", [])),
    }
    if extra:
        summary.update(extra)
    return summary


def persist_quality_report_v2(
    ctx: HandlerContext,
    *,
    episode: Episode,
    run_id: str,
    prepared: PreparedReport,
    evaluation: SemanticEvaluation,
    input: SemanticQcInput,
) -> tuple[QualityReport, Asset]:
    """Write ``quality-report/2.0``, upsert the report row and the findings rows."""
    session: Session = ctx.session
    payload = prepared.payload
    key = quality_report_v2_key(episode.id, run_id)
    data = canonical_json_bytes(payload)
    asset = _get_or_create_asset(
        ctx,
        key=key,
        data=data,
        payload=payload,
        logical_name=f"quality-report-v2:{payload.get('status')}",
    )

    existing = _existing_report(session, episode.id, run_id)
    _refuse_v1_overwrite(existing)
    canonical_version = int(payload.get("canonical_version") or episode.canonical_version)
    previous_quality_status = episode.quality_status
    verdict = str(payload.get("status"))
    summary = _summary_common(
        payload=payload,
        evaluation=evaluation,
        input=input,
        canonical_version=canonical_version,
        stats=prepared.stats,
        report_key=key,
        report_sha256=asset.sha256,
    )
    profile = payload.get("profile", {})
    if existing is None:
        report = QualityReport(
            id=new_id(),
            episode_id=episode.id,
            run_id=run_id,
            policy_version=str(payload.get("policy_version") or QC_POLICY_V2_VERSION),
            status=verdict,
            summary_json=summary,
            report_asset_id=asset.id,
            profile_id=profile.get("profile_id"),
            profile_revision=profile.get("profile_revision"),
            config_sha256=profile.get("config_sha256"),
            schema_version=QUALITY_REPORT_V2_SCHEMA_VERSION,
            created_at=utcnow(),
        )
        session.add(report)
        session.flush()
    else:
        report = existing
        report.policy_version = str(payload.get("policy_version") or QC_POLICY_V2_VERSION)
        report.status = verdict
        report.summary_json = summary
        report.report_asset_id = asset.id
        report.profile_id = profile.get("profile_id")
        report.profile_revision = profile.get("profile_revision")
        report.config_sha256 = profile.get("config_sha256")
        report.schema_version = QUALITY_REPORT_V2_SCHEMA_VERSION
        session.execute(delete(QualityFinding).where(QualityFinding.report_id == report.id))
        session.flush()

    for finding in prepared.findings:
        evidence = dict(finding.evidence)
        evidence["affected_time_ranges"] = [item.as_dict() for item in finding.affected_time_ranges]
        session.add(
            QualityFinding(
                id=new_id(),
                report_id=report.id,
                rule_code=finding.rule_code,
                severity=finding.severity.value,
                observed_json=dict(finding.observed),
                limit_json=dict(finding.threshold),
                evidence_json=evidence,
                source_stream=finding.source_stream,
                source_clock_domain=finding.source_clock_domain,
                method=finding.method or None,
                limits_json=dict(finding.limits),
                uncertainty_note=finding.uncertainty_note,
            )
        )
    session.flush()

    _apply_episode_quality_state(
        session, episode=episode, verdict=verdict, previous_quality_status=previous_quality_status
    )
    EPISODE_QC.labels(verdict=verdict, source_type=str(payload.get("source", {}).get("source_type") or "unknown")).inc()
    return report, asset


def persist_quality_report_too_large(
    ctx: HandlerContext,
    *,
    episode: Episode,
    run_id: str,
    evaluation: SemanticEvaluation,
    input: SemanticQcInput,
    canonical_version: int,
    error: QualityReportTooLarge,
    generated_at: datetime | None = None,
) -> tuple[QualityReport, Asset]:
    """Record full raw diagnostics + a FAIL report when the row cap is exceeded.

    The publish decision is failed explicitly (never truncated silently); the
    diagnostics Asset keeps every raw finding so an operator can inspect why the
    limit was hit.
    """
    session: Session = ctx.session
    payload = build_raw_diagnostics_payload(
        episode_id=episode.id,
        run_id=run_id,
        evaluation=evaluation,
        input=input,
        canonical_version=canonical_version,
        error=error,
    )
    if generated_at is not None:
        payload["generated_at"] = _utc_isoformat(generated_at)
    key = quality_diagnostics_v2_key(episode.id, run_id)
    data = canonical_json_bytes(payload)
    asset = _get_or_create_asset(
        ctx,
        key=key,
        data=data,
        payload=payload,
        logical_name="quality-diagnostics-too-large",
    )
    existing = _existing_report(session, episode.id, run_id)
    _refuse_v1_overwrite(existing)
    previous_quality_status = episode.quality_status
    stats = error.stats
    summary = {
        "schema_version": QUALITY_REPORT_V2_SCHEMA_VERSION,
        "policy_version": input.policy_version,
        "policy_config_sha256": input.policy_config_sha256,
        "canonical_version": int(canonical_version),
        "profile_id": input.profile.profile_id,
        "profile_revision": input.profile.profile_revision,
        "config_sha256": input.profile.config_sha256,
        "metrics": evaluation.metrics,
        "finding_count": 1,
        "raw_finding_count": stats.raw_finding_count,
        "finding_counts": {"ERROR": 1, "WARN": 0, "INFO": 0},
        "rule_codes": [TOO_LARGE_RULE_CODE],
        "error_code": ErrorCode.QUALITY_REPORT_TOO_LARGE.value,
        "error_details": stats.as_dict(),
        "diagnostics_asset_id": asset.id,
        "diagnostics_key": key,
        "report_key": None,
        "report_sha256": asset.sha256,
        "checks": {
            "run": list(evaluation.checks_run),
            "unavailable": list(evaluation.checks_unavailable),
            "not_applicable": list(evaluation.checks_not_applicable),
        },
        "normalization_valid": bool(input.normalization_valid),
        "canonical_missing_objects": list(input.extra.get("canonical_missing_objects", [])),
    }
    profile = _profile_payload(input)
    if existing is None:
        report = QualityReport(
            id=new_id(),
            episode_id=episode.id,
            run_id=run_id,
            policy_version=input.policy_version,
            status=QualityStatus.FAIL.value,
            summary_json=summary,
            report_asset_id=asset.id,
            profile_id=profile.get("profile_id"),
            profile_revision=profile.get("profile_revision"),
            config_sha256=profile.get("config_sha256"),
            schema_version=QUALITY_REPORT_V2_SCHEMA_VERSION,
            created_at=utcnow(),
        )
        session.add(report)
        session.flush()
    else:
        report = existing
        report.policy_version = input.policy_version
        report.status = QualityStatus.FAIL.value
        report.summary_json = summary
        report.report_asset_id = asset.id
        report.profile_id = profile.get("profile_id")
        report.profile_revision = profile.get("profile_revision")
        report.config_sha256 = profile.get("config_sha256")
        report.schema_version = QUALITY_REPORT_V2_SCHEMA_VERSION
        session.execute(delete(QualityFinding).where(QualityFinding.report_id == report.id))
        session.flush()
    session.add(
        QualityFinding(
            id=new_id(),
            report_id=report.id,
            rule_code=TOO_LARGE_RULE_CODE,
            severity="ERROR",
            observed_json=stats.as_dict(),
            limit_json={"max_finding_rows": stats.limit},
            evidence_json={
                "error_code": ErrorCode.QUALITY_REPORT_TOO_LARGE.value,
                "diagnostics_asset_id": asset.id,
                "diagnostics_key": key,
                "raw_finding_count": stats.raw_finding_count,
                "note": "publish decision failed explicitly; raw diagnostics retained, never truncated",
            },
            source_stream="quality_report",
            method="finding_row_cap",
            limits_json={"max_finding_rows": stats.limit},
            uncertainty_note=None,
        )
    )
    session.flush()
    _apply_episode_quality_state(
        session,
        episode=episode,
        verdict=QualityStatus.FAIL.value,
        previous_quality_status=previous_quality_status,
    )
    EPISODE_QC.labels(verdict=QualityStatus.FAIL.value, source_type=input.source_type or "unknown").inc()
    return report, asset
