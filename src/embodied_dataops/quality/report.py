"""Build, persist and announce one ``qc-policy/1.0`` quality report (SPEC 4.3, 7.1).

Persistence contract
--------------------
* the immutable ``quality.json`` artefact is written to
  ``quality/{episode_id}/{run_id}/quality.json`` (Lead-owned layout for
  handlers) and registered as an ``assets`` row with ``origin=report``;
* exactly one ``quality_reports`` row exists per ``(episode_id, run_id)``
  (unique constraint); findings are one row per rule code in
  ``quality_findings`` with ``observed`` / ``limit`` / ``evidence`` JSON;
* ``episodes.quality_status`` becomes the verdict and a ``FAIL`` verdict also
  moves the Episode to ``QUALITY_FAILED`` (SPEC 6.1: the report is still stored
  completely);
* Prometheus counters ``dataops_episode_qc_total`` and
  ``dataops_sync_error_ms`` are incremented (SPEC 12.2).

The report JSON itself is canonical (sorted keys, no extra whitespace) so a
re-execution of the same run with the same inputs and evidence produces
byte-identical output; that keeps the resource key stable and makes the
operation idempotent inside one task execution.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import Asset, Episode, QualityFinding, QualityReport
from embodied_dataops.domain.enums import AssetOrigin, EpisodeStatus, QualityStatus
from embodied_dataops.ids import new_id
from embodied_dataops.observability.metrics import EPISODE_QC, SYNC_ERROR
from embodied_dataops.quality.policy import QC_POLICY_VERSION
from embodied_dataops.quality.rules import QualityEvaluation, QualityInput
from embodied_dataops.storage.base import sha256_bytes
from embodied_dataops.workflow.handlers.base import HandlerContext, register_json
from embodied_dataops.workflow.outbox import utcnow

#: Schema version of the ``quality.json`` artefact.
QUALITY_REPORT_SCHEMA_VERSION = "quality-report/1.0"


def quality_report_key(episode_id: str, run_id: str) -> str:
    """Deterministic immutable object key for one run's quality report."""
    return f"quality/{episode_id}/{run_id}/quality.json"


def _utc_isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_quality_report(
    *,
    episode_id: str,
    run_id: str,
    evaluation: QualityEvaluation,
    input: QualityInput,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the ``quality.json`` payload (SPEC 7.1)."""
    payload: dict[str, Any] = {
        "schema_version": QUALITY_REPORT_SCHEMA_VERSION,
        "episode_id": episode_id,
        "run_id": run_id,
        "policy_version": input.policy_version or QC_POLICY_VERSION,
        "status": evaluation.verdict.value,
        "findings": evaluation.findings_as_dicts(),
        "metrics": evaluation.metrics,
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
            "robot_profile": input.robot_profile or input.metadata.get("robot_profile", ""),
        },
    }
    if generated_at is not None:
        payload["generated_at"] = _utc_isoformat(generated_at)
    return payload


def report_json_bytes(payload: dict[str, Any]) -> bytes:
    return canonical_json_bytes(payload)


def _get_or_create_report_asset(
    ctx: HandlerContext,
    *,
    key: str,
    data: bytes,
    payload: dict[str, Any],
) -> tuple[Asset, bool]:
    """Register the report asset, tolerating a re-execution of the same run.

    The engine wraps a handler execution in a transaction whose CAS can roll
    back, so a failed attempt leaves no Asset row behind.  A re-execution of the
    *same* run (the only case that can revisit this key) produces byte-identical
    canonical JSON, therefore the existing asset can be re-used; if a test or an
    operator mutates the payload inside the same run, the row is refreshed in
    place for this deterministic per-run key instead of violating the unique
    object-key constraint.
    """
    existing = ctx.session.execute(select(Asset).where(Asset.object_key == key)).scalar_one_or_none()
    digest = sha256_bytes(data)
    if existing is not None:
        if existing.sha256 != digest:
            stored = ctx.storage.put_bytes(key, data, content_type="application/json")
            existing.sha256 = stored.sha256 or digest
            existing.byte_size = len(data)
            existing.media_type = "application/json"
            existing.meta_json = {"policy_version": payload.get("policy_version"), "status": payload.get("status")}
            ctx.session.flush()
        return existing, False
    asset = register_json(
        ctx,
        key=key,
        payload=payload,
        media_type="application/json",
        origin=AssetOrigin.REPORT,
        logical_name=f"quality-report:{payload.get('status')}",
        meta={
            "policy_version": payload.get("policy_version"),
            "status": payload.get("status"),
            "rule_codes": [finding["rule_code"] for finding in payload.get("findings", [])],
        },
    )
    return asset, True


def persist_quality_report(
    ctx: HandlerContext,
    *,
    episode: Episode,
    run_id: str,
    evaluation: QualityEvaluation,
    report_payload: dict[str, Any],
) -> tuple[QualityReport, Asset]:
    """Write ``quality.json``, upsert ``quality_reports``/``quality_findings``, update the Episode.

    Runs inside the engine transaction: this function never commits or rolls back.
    """
    session: Session = ctx.session
    key = quality_report_key(episode.id, run_id)
    data = report_json_bytes(report_payload)
    asset, _created = _get_or_create_report_asset(ctx, key=key, data=data, payload=report_payload)

    report = session.execute(
        select(QualityReport).where(
            QualityReport.episode_id == episode.id, QualityReport.run_id == run_id
        )
    ).scalar_one_or_none()
    verdict = evaluation.verdict.value
    summary = {
        "policy_version": report_payload.get("policy_version"),
        "metrics": evaluation.metrics,
        "finding_count": len(evaluation.findings),
        "finding_counts": evaluation.counts_by_severity(),
        "rule_codes": [finding.rule_code.value for finding in evaluation.findings],
        "report_key": key,
        "report_sha256": asset.sha256,
        "checks": report_payload.get("checks", {}),
        "normalization_valid": bool(evaluation.metrics.get("normalization_valid", True)),
        "schema_version": QUALITY_REPORT_SCHEMA_VERSION,
    }
    if report is None:
        report = QualityReport(
            id=new_id(),
            episode_id=episode.id,
            run_id=run_id,
            policy_version=report_payload.get("policy_version") or QC_POLICY_VERSION,
            status=verdict,
            summary_json=summary,
            report_asset_id=asset.id,
            created_at=utcnow(),
        )
        session.add(report)
        session.flush()
    else:
        report.policy_version = report_payload.get("policy_version") or QC_POLICY_VERSION
        report.status = verdict
        report.summary_json = summary
        report.report_asset_id = asset.id
        session.execute(delete(QualityFinding).where(QualityFinding.report_id == report.id))
        session.flush()

    for finding in evaluation.findings:
        evidence = dict(finding.evidence)
        evidence["affected_time_ranges"] = [item.as_dict() for item in finding.affected_time_ranges]
        session.add(
            QualityFinding(
                id=new_id(),
                report_id=report.id,
                rule_code=finding.rule_code.value,
                severity=finding.severity.value,
                observed_json=dict(finding.observed),
                limit_json=dict(finding.threshold),
                evidence_json=evidence,
            )
        )
    session.flush()

    # Episode quality state (SPEC 6.1).  A PASS/WARN Episode keeps processing;
    # the READY_FOR_REVIEW step owns the final review transition.
    episode.quality_status = verdict
    if evaluation.verdict == QualityStatus.FAIL:
        episode.status = EpisodeStatus.QUALITY_FAILED.value
    elif episode.status in {EpisodeStatus.IMPORTED.value, EpisodeStatus.PROCESSING.value}:
        episode.status = EpisodeStatus.PROCESSING.value
    episode.updated_at = utcnow()
    session.flush()

    source_type = evaluation.metrics.get("source_type") or ""
    EPISODE_QC.labels(verdict=verdict, source_type=source_type or "unknown").inc()
    max_sync_error = evaluation.metrics.get("max_abs_sync_error_ms")
    if isinstance(max_sync_error, (int, float)):
        SYNC_ERROR.labels(source_type=source_type or "unknown").observe(float(max_sync_error))

    return report, asset


__all__ = [
    "QUALITY_REPORT_SCHEMA_VERSION",
    "build_quality_report",
    "persist_quality_report",
    "quality_report_key",
    "report_json_bytes",
]
