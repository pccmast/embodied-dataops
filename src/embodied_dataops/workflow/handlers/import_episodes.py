"""IMPORT_EPISODES handler (SPEC 5.3, 6.1).

Import-level step: the adapter enumerates the authoritative Episode list (for
LeRobot v3 this is the official ``meta/episodes`` metadata -- never filenames),
then the handler creates one ``episodes`` row per source episode (idempotent on
the ``UNIQUE(source_import_id, source_episode_index)`` constraint) and one
``PipelineRun`` per Episode via :func:`embodied_dataops.workflow.dag.create_pipeline_run`.

If enumeration fails, the ImportJob/SourceImport are marked ``FAILED`` with
structured diagnostics, no Episode rows are created, and ``verdict="FAIL"`` tells
the engine that the import-level chain is done (SPEC 6.1: an unenumerable source
may legitimately leave no Episodes and no QualityReport).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import Episode, PipelineRun
from embodied_dataops.domain.enums import (
    EpisodeStatus,
    EpisodeType,
    ImportJobState,
    Step,
    UploadState,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.imports.base import EpisodeSource, get_parser
from embodied_dataops.imports.common import (
    import_context_payload,
    load_import_job,
    load_source_import,
    materialize_import_source,
    record_import_failure,
    utcnow,
)
from embodied_dataops.workflow.dag import create_pipeline_run
from embodied_dataops.workflow.handlers.base import HandlerContext, HandlerResult, TaskHandler


class ImportEpisodesHandler(TaskHandler):
    step = Step.IMPORT_EPISODES

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        session: Session = ctx.session
        job = load_import_job(session, ctx.task.import_job_id)
        source_import = load_source_import(session, job.import_id)
        job.state = ImportJobState.RUNNING.value
        job.started_at = job.started_at or utcnow()
        session.flush()

        try:
            bundle = materialize_import_source(
                session, ctx.storage, job, source_import, ctx.workspace
            )
            parser = get_parser(source_import.source_type)
            parsed = parser.parse(bundle.context)
        except ApiError as exc:
            data = record_import_failure(
                session,
                job,
                source_import,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
            data["step"] = self.step.value
            data["source_type"] = source_import.source_type
            return HandlerResult(data=data, verdict="FAIL")

        if not parsed.episodes:
            data = record_import_failure(
                session,
                job,
                source_import,
                code=ErrorCode.EPISODE_ENUMERATION_FAILED,
                message="the source adapter enumerated no Episodes",
            )
            data["step"] = self.step.value
            data["source_type"] = source_import.source_type
            return HandlerResult(data=data, verdict="FAIL")

        created: list[dict[str, Any]] = []
        for episode_source in parsed.episodes:
            episode = self._upsert_episode(session, source_import, episode_source)
            run = self._ensure_pipeline_run(session, episode)
            created.append(
                {
                    "episode_id": episode.id,
                    "source_episode_index": episode.source_episode_index,
                    "episode_type": episode.episode_type,
                    "task_label": episode.task_label,
                    "duration_ns": episode.duration_ns,
                    "normalization_valid": episode_source.normalization_valid,
                    "diagnostics": episode_source.diagnostics,
                    "run_id": run.id if run is not None else None,
                }
            )

        now = utcnow()
        source_import.upload_state = UploadState.READY.value
        source_import.completed_at = now
        source_import.error_code = None
        source_import.error_detail = None
        job.state = ImportJobState.SUCCEEDED.value
        job.finished_at = now
        session.flush()

        data: dict[str, Any] = {
            "step": self.step.value,
            "valid": True,
            "source_type": source_import.source_type,
            "source": import_context_payload(bundle),
            "episode_count": len(created),
            "episodes": created,
            "warnings": parsed.warnings,
            "source_metadata": parsed.metadata,
            "imported_at": now.isoformat(),
        }
        return HandlerResult(data=data)

    # -- helpers ---------------------------------------------------------------

    def _upsert_episode(
        self, session: Session, source_import: Any, source: EpisodeSource
    ) -> Episode:
        episode = session.scalar(
            select(Episode).where(
                Episode.source_import_id == source_import.id,
                Episode.source_episode_index == source.source_episode_index,
            )
        )
        episode_type = (
            source.episode_type.value
            if isinstance(source.episode_type, EpisodeType)
            else str(source.episode_type)
        )
        if episode is None:
            episode = Episode(
                id=new_id(),
                source_import_id=source_import.id,
                source_episode_index=source.source_episode_index,
                episode_type=episode_type,
                task_label=source.task_label,
                duration_ns=source.duration_ns,
                status=EpisodeStatus.IMPORTED.value,
                canonical_version=1,
                data_origin=source_import.data_origin,
            )
            session.add(episode)
            session.flush()
        else:
            episode.episode_type = episode_type
            episode.task_label = source.task_label or episode.task_label
            episode.duration_ns = source.duration_ns or episode.duration_ns
            episode.data_origin = source_import.data_origin
            episode.updated_at = utcnow()
            session.flush()
        return episode

    def _ensure_pipeline_run(self, session: Session, episode: Episode) -> PipelineRun | None:
        """Create one run per newly imported episode; never duplicate on re-run."""
        existing = session.scalar(
            select(PipelineRun).where(PipelineRun.episode_id == episode.id).limit(1)
        )
        if existing is not None:
            return existing
        return create_pipeline_run(session, episode)

    # -- module-level object ---------------------------------------------------

HANDLER = ImportEpisodesHandler()

__all__ = ["HANDLER", "ImportEpisodesHandler"]
