"""READY_FOR_REVIEW step (SPEC 6.1, 6.2).

The task is an *operation*, not an approval: it flips the Episode to
``READY_FOR_REVIEW`` once the quality verdict allows it and writes no heavy
artifact.  A FAIL verdict must never reach this step (the engine cancels the
downstream closure), but the guard is enforced here as well so a hand-crafted
task cannot bypass quality policy.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from embodied_dataops.db.models import Episode, PipelineRun, QualityReport
from embodied_dataops.domain.enums import EpisodeStatus, QualityStatus, Step
from embodied_dataops.errors import CancelledError, ErrorCode, TerminalTaskError
from embodied_dataops.operations.revisions import bump as bump_revision
from embodied_dataops.workflow.handlers.base import HandlerContext, HandlerResult, TaskHandler
from embodied_dataops.workflow.outbox import utcnow

logger = logging.getLogger(__name__)

_ELIGIBLE_VERDICTS = frozenset({QualityStatus.PASS.value, QualityStatus.WARN.value})


class ReadyForReviewHandler(TaskHandler):
    step = Step.READY_FOR_REVIEW

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        if ctx.is_cancelled():
            raise CancelledError("task cancelled before READY_FOR_REVIEW")
        task = ctx.task
        if not task.run_id:
            raise TerminalTaskError(
                ErrorCode.INVALID_INPUT,
                "READY_FOR_REVIEW requires an Episode-level PipelineRun",
                details={"task_id": task.id},
            )
        run = ctx.session.get(PipelineRun, task.run_id)
        if run is None:  # pragma: no cover - FK guarantees the run exists
            raise TerminalTaskError(ErrorCode.NOT_FOUND, "PipelineRun for task not found")
        if run.cancel_requested:
            raise CancelledError("pipeline run cancelled")
        episode = ctx.session.get(Episode, run.episode_id)
        if episode is None:  # pragma: no cover - FK guarantees the episode exists
            raise TerminalTaskError(ErrorCode.NOT_FOUND, "Episode for pipeline run not found")

        report = ctx.session.scalars(
            select(QualityReport).where(
                QualityReport.episode_id == episode.id,
                QualityReport.run_id == run.id,
            )
        ).first()
        verdict = report.status if report is not None else None
        if verdict not in _ELIGIBLE_VERDICTS:
            raise TerminalTaskError(
                ErrorCode.QC_STATUS_NOT_ELIGIBLE,
                "Episode quality verdict does not allow review",
                details={"episode_id": episode.id, "verdict": verdict},
            )

        if episode.status == EpisodeStatus.READY_FOR_REVIEW.value:
            # Idempotent re-execution (e.g. retried run over the same episode).
            return HandlerResult(
                data={
                    "verdict": verdict,
                    "episode_id": episode.id,
                    "run_id": run.id,
                    "already_ready": True,
                },
                episode_status=EpisodeStatus.READY_FOR_REVIEW.value,
            )

        episode.status = EpisodeStatus.READY_FOR_REVIEW.value
        episode.updated_at = utcnow()
        bump_revision(ctx.session, episode)
        logger.info(
            "episode.ready_for_review episode_id=%s run_id=%s verdict=%s",
            episode.id,
            run.id,
            verdict,
        )
        return HandlerResult(
            data={
                "verdict": verdict,
                "episode_id": episode.id,
                "run_id": run.id,
                "quality_report_id": report.id if report is not None else None,
            },
            episode_status=EpisodeStatus.READY_FOR_REVIEW.value,
        )


HANDLER = ReadyForReviewHandler()
handle = HANDLER.handle

__all__ = ["ReadyForReviewHandler", "HANDLER", "handle"]
