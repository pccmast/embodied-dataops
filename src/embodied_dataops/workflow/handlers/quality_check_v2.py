"""``QUALITY_CHECK`` handler for ``pipeline_version=2.0.0`` / canonical-episode/2.0.

The V1 handler (:mod:`embodied_dataops.workflow.handlers.quality_check`) stays the
owner of ``qc-policy/1.0``; this module implements the V2 path and is delegated to
from the V1 handler when the Episode is canonical 2.0 (or the run is pipeline 2.x).

What it reads
-------------
* canonical-episode/2.0 objects from the object store (SPEC §2.3): ``metadata.json``,
  native state/action indices, native camera timestamps, the aligned 10 Hz view and
  the optional PointCloud2 summary;
* the Robot Profile Registry row bound by the Episode (``profile_id`` +
  ``profile_revision``) and its ``config_sha256``; a missing profile is an explicit
  ``MISSING_ROBOT_PROFILE`` finding, never a guessed shape;
* the ``NORMALIZE_EPISODE`` task output of the same run for diagnostics.

It never reads raw source bytes, never re-zero-fills missing samples and never
recomputes/overwrites a V1 report.  The handler runs inside the engine
transaction and returns ``HandlerResult(verdict=...)`` so a FAIL stops downstream
steps (SPEC 6.1).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from embodied_dataops.db.models import Episode, PipelineRun, RobotProfile, SourceImport, Task
from embodied_dataops.domain.enums import Step
from embodied_dataops.domain.profiles import RobotProfile as RobotProfileType
from embodied_dataops.domain.profiles import parse_profile
from embodied_dataops.errors import CancelledError, ErrorCode, TerminalTaskError
from embodied_dataops.quality.canonical_v2 import build_semantic_qc_input, load_canonical_v2
from embodied_dataops.quality.v2_report import (
    QualityReportTooLarge,
    persist_quality_report_too_large,
    persist_quality_report_v2,
    prepare_quality_report_v2,
)
from embodied_dataops.quality.v2_rules import (
    ProfileBinding,
    SemanticEvaluation,
    SemanticQcInput,
    evaluate_semantic_quality,
)
from embodied_dataops.workflow.handlers.base import HandlerContext, HandlerResult, TaskHandler

__all__ = [
    "QualityCheckV2Handler",
    "is_v2_quality_pipeline",
    "load_profile_binding",
    "run_quality_check_v2",
]


def is_v2_quality_pipeline(run: PipelineRun | None, episode: Episode) -> bool:
    """True when the Episode/run must be evaluated with ``qc-policy/2.0``."""
    schema_version = str(getattr(episode, "canonical_schema_version", "") or "")
    if schema_version.startswith("2"):
        return True
    pipeline_version = str(getattr(run, "pipeline_version", "") or "")
    return pipeline_version.startswith("2")


def load_profile_binding(
    session, episode: Episode, metadata: dict[str, Any]
) -> ProfileBinding:
    """Resolve the Episode's explicit profile binding (never inferred from shape)."""
    profile_id = episode.profile_id or metadata.get("profile_id")
    profile_id = str(profile_id) if profile_id else None
    revision_raw = (
        episode.profile_revision if episode.profile_revision is not None else metadata.get("profile_revision")
    )
    try:
        profile_revision = int(revision_raw) if revision_raw is not None else None
    except (TypeError, ValueError):
        profile_revision = None

    missing_fields: list[str] = []
    if not profile_id:
        missing_fields.append("profile_id")
    if profile_revision is None:
        missing_fields.append("profile_revision")

    detail: dict[str, Any] = {
        "registry_profile_id": episode.profile_id,
        "registry_profile_revision": episode.profile_revision,
        "metadata_profile_id": metadata.get("profile_id"),
        "metadata_profile_revision": metadata.get("profile_revision"),
        "metadata_profile_config_sha256": metadata.get("profile_config_sha256"),
    }
    row: RobotProfile | None = None
    if profile_id and profile_revision is not None:
        row = session.execute(
            select(RobotProfile).where(
                RobotProfile.profile_id == profile_id,
                RobotProfile.profile_revision == profile_revision,
            )
        ).scalars().first()
        if row is None:
            missing_fields.append("robot_profiles row (profile_id, profile_revision)")

    config: RobotProfileType | None = None
    authority = ""
    config_sha256 = row.config_sha256 if row is not None else None
    if row is not None:
        authority = str(row.authority or "")
        try:
            # Parse by the row's own generation: a 2.1 profile is not a 2.0 profile
            # with missing fields, and forcing it through the 2.0 model silently
            # turned a valid binding into ``robot_profile_config_json`` missing.
            config = parse_profile(dict(row.config_json))
            authority = authority or config.authority
            config_sha256 = config_sha256 or config.config_sha256()
        except Exception:  # noqa: BLE001 - a corrupt registry row is an explicit gap
            missing_fields.append("robot_profile_config_json")
            detail["registry_config_parse_error"] = "robot profile config is not a valid revision"

    metadata_id = metadata.get("profile_id")
    metadata_revision = metadata.get("profile_revision")
    metadata_sha = metadata.get("profile_config_sha256")
    binding_verified = (
        config is not None
        and row is not None
        and (not metadata_id or str(metadata_id) == profile_id)
        and (metadata_revision is None or int(metadata_revision) == profile_revision)
        and (not metadata_sha or str(metadata_sha) == config_sha256)
    )
    mismatches: list[str] = []
    if config is not None:
        if metadata_id and profile_id and str(metadata_id) != profile_id:
            mismatches.append("profile_id")
        if metadata_revision is not None and profile_revision is not None and int(metadata_revision) != profile_revision:
            mismatches.append("profile_revision")
        if metadata_sha and config_sha256 and str(metadata_sha) != config_sha256:
            mismatches.append("profile_config_sha256")
    detail["binding_mismatches"] = mismatches
    mode = "SEMANTIC" if config is not None else (
        "DIAGNOSTIC_ONLY" if metadata.get("diagnostic_only") is True else "MISSING"
    )
    return ProfileBinding(
        profile_id=profile_id,
        profile_revision=profile_revision,
        authority=authority,
        config=config,
        config_sha256=config_sha256,
        mode=mode,
        binding_verified=binding_verified,
        missing_fields=missing_fields,
        detail=detail,
    )


def _normalize_task_output(session, run_id: str) -> dict[str, Any]:
    task = session.execute(
        select(Task).where(Task.run_id == run_id, Task.step == Step.NORMALIZE_EPISODE.value)
    ).scalar_one_or_none()
    if task is None or not task.output_json:
        return {}
    return dict(task.output_json)


def _normalize_diagnostics(normalize_output: dict[str, Any]) -> list[Any]:
    diagnostics = normalize_output.get("diagnostics")
    return list(diagnostics) if isinstance(diagnostics, list) else []


def _handler_data(
    *,
    report,
    asset,
    evaluation: SemanticEvaluation,
    quality_input: SemanticQcInput,
    episode: Episode,
    verdict: str,
    overflow: bool,
) -> dict[str, Any]:
    summary = report.summary_json if isinstance(report.summary_json, dict) else {}
    data: dict[str, Any] = {
        "quality_report_id": report.id,
        "quality_report_key": getattr(asset, "object_key", None),
        "quality_report_asset_id": asset.id,
        "schema_version": report.schema_version,
        "policy_version": report.policy_version,
        "policy_config_sha256": quality_input.policy_config_sha256,
        "canonical_version": quality_input.canonical_version,
        "profile_id": report.profile_id,
        "profile_revision": report.profile_revision,
        "profile_config_sha256": report.config_sha256,
        "verdict": verdict,
        "finding_count": summary.get("finding_count", 0),
        "raw_finding_count": summary.get("raw_finding_count", 0),
        "rule_codes": summary.get("rule_codes", []),
        "finding_counts": summary.get("finding_counts", evaluation.counts_by_severity()),
        "checks": summary.get("checks", {}),
        "metrics": summary.get("metrics", evaluation.metrics),
        "normalization_valid": quality_input.normalization_valid,
        "episode_quality_status": episode.quality_status,
        "episode_status": episode.status,
    }
    if overflow:
        data["error_code"] = ErrorCode.QUALITY_REPORT_TOO_LARGE.value
        data["diagnostics_asset_id"] = summary.get("diagnostics_asset_id")
        data["diagnostics_key"] = summary.get("diagnostics_key")
    return data


def run_quality_check_v2(
    ctx: HandlerContext,
    *,
    run: PipelineRun,
    episode: Episode,
    source_import: SourceImport | None = None,
) -> HandlerResult:
    """Evaluate ``qc-policy/2.0`` for one Episode and persist the report."""
    if not ctx.task.run_id:
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            "QUALITY_CHECK requires an Episode-level PipelineRun (tasks.run_id is NULL)",
        )
    if ctx.is_cancelled():
        raise CancelledError("QUALITY_CHECK cancelled before execution")

    snapshot = load_canonical_v2(
        ctx.storage,
        ctx.workspace,
        episode_id=episode.id,
        canonical_version=episode.canonical_version,
    )
    if not snapshot.metadata_present:
        raise TerminalTaskError(
            ErrorCode.OBJECT_NOT_FOUND,
            "canonical-episode/2.0 metadata.json is missing; qc-policy/2.0 cannot evaluate "
            "this Episode",
            details={
                "metadata_key": snapshot.keys["metadata"],
                "missing_objects": snapshot.missing_objects,
                "canonical_version": episode.canonical_version,
            },
        )

    metadata = snapshot.metadata
    normalize_output = _normalize_task_output(ctx.session, run.id)
    binding = load_profile_binding(ctx.session, episode, metadata)
    quality_input = build_semantic_qc_input(
        snapshot,
        episode_id=episode.id,
        episode_type=episode.episode_type,
        data_origin=str(metadata.get("data_origin") or episode.data_origin or ""),
        source_type=(source_import.source_type if source_import is not None else ""),
        profile=binding,
        duration_ns=int(episode.duration_ns or 0),
        normalize_diagnostics=_normalize_diagnostics(normalize_output),
    )
    evaluation = evaluate_semantic_quality(quality_input)

    overflow = False
    try:
        prepared = prepare_quality_report_v2(
            episode_id=episode.id,
            run_id=run.id,
            evaluation=evaluation,
            input=quality_input,
            canonical_version=episode.canonical_version,
        )
    except QualityReportTooLarge as exc:
        overflow = True
        report, asset = persist_quality_report_too_large(
            ctx,
            episode=episode,
            run_id=run.id,
            evaluation=evaluation,
            input=quality_input,
            canonical_version=episode.canonical_version,
            error=exc,
        )
        verdict = "FAIL"
    else:
        report, asset = persist_quality_report_v2(
            ctx,
            episode=episode,
            run_id=run.id,
            prepared=prepared,
            evaluation=evaluation,
            input=quality_input,
        )
        verdict = evaluation.verdict.value

    return HandlerResult(
        data=_handler_data(
            report=report,
            asset=asset,
            evaluation=evaluation,
            quality_input=quality_input,
            episode=episode,
            verdict=verdict,
            overflow=overflow,
        ),
        verdict=verdict,
        episode_status=None,
        output_asset_id=asset.id,
    )


class QualityCheckV2Handler(TaskHandler):
    """Explicit V2 handler (the V1 handler delegates here for canonical 2.0)."""

    step = Step.QUALITY_CHECK

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        task = ctx.task
        if not task.run_id:
            raise TerminalTaskError(
                ErrorCode.INVALID_INPUT,
                "QUALITY_CHECK requires an Episode-level PipelineRun (tasks.run_id is NULL)",
            )
        run = ctx.session.get(PipelineRun, task.run_id)
        if run is None:
            raise TerminalTaskError(
                ErrorCode.NOT_FOUND, "PipelineRun for this QUALITY_CHECK task was not found"
            )
        episode = ctx.session.get(Episode, run.episode_id)
        if episode is None:
            raise TerminalTaskError(ErrorCode.NOT_FOUND, "Episode for this PipelineRun was not found")
        source_import = (
            ctx.session.get(SourceImport, episode.source_import_id)
            if episode.source_import_id
            else None
        )
        return run_quality_check_v2(
            ctx, run=run, episode=episode, source_import=source_import
        )
