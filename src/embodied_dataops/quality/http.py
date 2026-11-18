"""``GET /api/v1/episodes/{id}/quality`` version selection and response contract.

V1 returned the newest report for an Episode and ignored policy versions.  The V2
contract (SPEC §3.2, DO2-13) keeps every V1 field and adds:

* query parameter ``policy_version=qc-policy/1.0|qc-policy/2.0`` (omitted means
  *the latest completed report for the current canonical version*);
* response fields ``canonical_version``, ``policy_version`` and
  ``schema_version`` next to the legacy ``verdict``/``findings``/``metrics``/
  ``report_url`` fields;
* new finding fields ``source_stream``/``source_clock_domain``/``method``/
  ``limits``/``uncertainty_note`` (empty for historical V1 reports).

The route module only needs :func:`build_quality_response`; all selection logic
lives here so it is testable without FastAPI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import Asset, Episode, QualityFinding, QualityReport
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.quality.v2_policy import QC_POLICY_VERSIONS, QUALITY_REPORT_V2_SCHEMA_VERSION
from embodied_dataops.storage.base import get_storage

#: V1 report schema version (historical rows have this exact value).
QUALITY_REPORT_V1_SCHEMA_VERSION = "quality-report/1.0"

__all__ = [
    "QUALITY_REPORT_V1_SCHEMA_VERSION",
    "build_quality_response",
    "canonical_schema_version_of",
    "finding_payload",
    "findings_for_report",
    "quality_report_payload",
    "report_asset_url",
    "report_canonical_version",
    "report_schema_version",
    "select_quality_report",
    "validate_policy_version",
]


def validate_policy_version(value: str | None) -> str | None:
    """Validate the ``policy_version`` query parameter (invalid -> 422)."""
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    if candidate not in QC_POLICY_VERSIONS:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "policy_version must be one of the supported quality policy versions",
            details={
                "policy_version": candidate,
                "allowed": list(QC_POLICY_VERSIONS),
            },
        )
    return candidate


def canonical_schema_version_of(episode: Episode) -> str:
    """Current canonical schema version of the Episode (V1 default ``1.0``)."""
    value = getattr(episode, "canonical_schema_version", None)
    return str(value) if value else "1.0"


def report_schema_version(report: QualityReport) -> str:
    return report.schema_version or QUALITY_REPORT_V1_SCHEMA_VERSION


def report_canonical_version(report: QualityReport, episode: Episode) -> int:
    """Canonical version a report is bound to.

    ``quality-report/2.0`` stores the binding explicitly; historical 1.0 reports
    were only ever produced for canonical-episode/1.0, so they report ``1`` rather
    than a later ``episode.canonical_version`` that did not exist when the report
    was written.
    """
    summary = report.summary_json if isinstance(report.summary_json, dict) else {}
    raw = summary.get("canonical_version")
    try:
        if raw is not None:
            return int(raw)
    except (TypeError, ValueError):
        pass
    if report_schema_version(report) == QUALITY_REPORT_V1_SCHEMA_VERSION:
        return 1
    return int(episode.canonical_version or 1)


def _not_found(
    *,
    episode: Episode,
    policy_version: str | None,
    run_id: str | None,
) -> ApiError:
    details: dict[str, Any] = {
        "canonical_version": int(episode.canonical_version or 1),
        "canonical_schema_version": canonical_schema_version_of(episode),
        "reason": "no completed quality report matches the current canonical version",
    }
    if policy_version is not None:
        details["policy_version"] = policy_version
        details["reason"] = "no completed quality report exists for the requested policy version"
    if run_id is not None:
        details["run_id"] = run_id
        details["reason"] = "no completed quality report exists for the requested pipeline run"
    return ApiError(ErrorCode.NOT_FOUND, "Quality report not found for this episode", details=details)


def select_quality_report(
    session: Session,
    *,
    episode: Episode,
    policy_version: str | None = None,
    run_id: str | None = None,
) -> QualityReport:
    """Select the report for the quality API (V1 compatibility included).

    * ``run_id`` (V1 behaviour) narrows to that run;
    * an explicit ``policy_version`` selects the newest report of that policy;
    * the default selects the newest completed report **for the current canonical
      version** -- a historical V1 report never satisfies a V2 Episode default;
    * a V1 Episode (``canonical_schema_version=1.0``) keeps returning its V1
      report so existing callers are unaffected.
    """
    requested = validate_policy_version(policy_version)
    stmt = select(QualityReport).where(QualityReport.episode_id == episode.id)
    if run_id:
        stmt = stmt.where(QualityReport.run_id == run_id)
    stmt = stmt.order_by(QualityReport.created_at.desc(), QualityReport.id.desc())
    reports = list(session.scalars(stmt))
    if not reports:
        raise _not_found(episode=episode, policy_version=requested, run_id=run_id)

    if requested is not None:
        for report in reports:
            if report.policy_version == requested:
                return report
        raise _not_found(episode=episode, policy_version=requested, run_id=run_id)

    current_schema = canonical_schema_version_of(episode)
    target_schema = (
        QUALITY_REPORT_V2_SCHEMA_VERSION
        if current_schema.startswith("2")
        else QUALITY_REPORT_V1_SCHEMA_VERSION
    )
    target_canonical = int(episode.canonical_version or 1)
    for report in reports:
        if report_schema_version(report) != target_schema:
            continue
        if target_schema == QUALITY_REPORT_V2_SCHEMA_VERSION:
            if report_canonical_version(report, episode) != target_canonical:
                continue
        return report
    raise _not_found(episode=episode, policy_version=None, run_id=run_id)


def findings_for_report(session: Session, report: QualityReport) -> list[QualityFinding]:
    return list(
        session.scalars(
            select(QualityFinding)
            .where(QualityFinding.report_id == report.id)
            .order_by(QualityFinding.id)
        )
    )


def report_asset_url(session: Session, report: QualityReport) -> str | None:
    if not report.report_asset_id:
        return None
    asset = session.get(Asset, report.report_asset_id)
    if asset is None:
        return None
    try:
        return get_storage().presign_get(asset.object_key)
    except ApiError:  # pragma: no cover - storage outage must not hide the report
        return None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def finding_payload(finding: QualityFinding) -> dict[str, Any]:
    """One finding with the V1 fields and the additive V2 provenance fields."""
    return {
        "rule_code": finding.rule_code,
        "severity": finding.severity,
        "observed": finding.observed_json,
        "threshold": finding.limit_json,
        "evidence": finding.evidence_json,
        "source_stream": finding.source_stream,
        "source_clock_domain": finding.source_clock_domain,
        "method": finding.method,
        "limits": finding.limits_json if isinstance(finding.limits_json, dict) else {},
        "uncertainty_note": finding.uncertainty_note,
    }


def quality_report_payload(
    report: QualityReport,
    findings: list[QualityFinding],
    *,
    episode: Episode,
    report_url: str | None = None,
) -> dict[str, Any]:
    """Response body for the quality endpoint (legacy fields retained)."""
    summary = report.summary_json if isinstance(report.summary_json, dict) else {}
    schema_version = report_schema_version(report)
    return {
        "report_id": report.id,
        "run_id": report.run_id,
        "policy_version": report.policy_version,
        "canonical_version": report_canonical_version(report, episode),
        "schema_version": schema_version,
        "verdict": report.status,
        "status": report.status,
        "summary": summary,
        "metrics": summary.get("metrics", {}),
        "findings": [finding_payload(finding) for finding in findings],
        "report_url": report_url,
        "created_at": _iso(report.created_at),
    }


def build_quality_response(
    session: Session,
    *,
    episode: Episode,
    policy_version: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Select the report and build the complete response body."""
    report = select_quality_report(
        session, episode=episode, policy_version=policy_version, run_id=run_id
    )
    findings = findings_for_report(session, report)
    return quality_report_payload(
        report,
        findings,
        episode=episode,
        report_url=report_asset_url(session, report),
    )
