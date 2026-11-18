"""The three DO-A02 C3 controlled business effects.

Each function runs inside :func:`embodied_dataops.operations.service.execute_controlled_write`
after the resource row is locked, the ETag/fence are validated and only then
creates the downstream object.  Raising an ``ApiError`` anywhere rolls the whole
transaction back, so a failed request can never leave a Run/Analysis behind.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.db.models import (
    Episode,
    EvaluationReport,
    GapAnalysis,
    OutboxEvent,
    PipelineRun,
    Task,
)
from embodied_dataops.domain.enums import RunState, Step, TaskState
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.operations import revisions
from embodied_dataops.operations.contract import (
    ACTION_ANALYZE_EVALUATION_GAPS,
    ACTION_REPROCESS_EPISODE,
    ACTION_RETRY_FAILED_TASK,
    ETAG_EPISODE,
    ETAG_EVALUATION,
    ETAG_TASK,
    TENANT_ID,
    V2_PIPELINE_VERSION,
)
from embodied_dataops.operations.evaluations import evaluation_results
from embodied_dataops.operations.service import execute_controlled_write
from embodied_dataops.workflow.dag import create_pipeline_run
from embodied_dataops.workflow.outbox import utcnow

logger = logging.getLogger(__name__)

#: Terminal task errors that represent invalid/corrupt input: retrying cannot fix
#: them without a new import, so the retry API rejects them (DO-A02 C3).
NON_RETRYABLE_ERROR_CODES: frozenset[str] = frozenset(
    {
        ErrorCode.FILE_DECODE_ERROR.value,
        ErrorCode.LEROBOT_METADATA_INVALID.value,
        ErrorCode.EPISODE_ENUMERATION_FAILED.value,
        ErrorCode.UNSAFE_ARCHIVE.value,
        ErrorCode.UNSUPPORTED_ARCHIVE_FORMAT.value,
        ErrorCode.ARCHIVE_TOO_LARGE.value,
        ErrorCode.SOURCE_HASH_MISMATCH.value,
        ErrorCode.SOURCE_SIZE_LIMIT.value,
        ErrorCode.SOURCE_DURATION_LIMIT.value,
        ErrorCode.SOURCE_NOT_FOUND.value,
        ErrorCode.UPLOAD_STATE_INVALID.value,
        ErrorCode.CONTENT_TYPE_MISMATCH.value,
        ErrorCode.UNSUPPORTED_SOURCE_TYPE.value,
        ErrorCode.UNSUPPORTED_TOPIC_PROFILE.value,
        ErrorCode.INVALID_MAPPING.value,
        ErrorCode.MAPPING_REQUIRED.value,
        ErrorCode.INVALID_INPUT.value,
        ErrorCode.VALIDATION_FAILED.value,
        ErrorCode.QC_STATUS_NOT_ELIGIBLE.value,
        ErrorCode.INCOMPATIBLE_EPISODE.value,
        ErrorCode.CANCEL_REQUESTED.value,
    }
)

SUPPORTED_PIPELINE_VERSIONS = frozenset({"1.0.0", V2_PIPELINE_VERSION})

#: Outbox topic for asynchronous feedback work (gap analysis completion).
FEEDBACK_TOPIC = "stream:ops:feedback"


def is_task_retryable(task: Task) -> bool:
    """A retry is safe only for a FAILED task whose error is not data-fatal."""
    if task.state != TaskState.FAILED.value:
        return False
    if task.error_code and task.error_code in NON_RETRYABLE_ERROR_CODES:
        return False
    return True


def _retry_not_safe(task: Task, reason: str, **details: Any) -> ApiError:
    return ApiError(
        ErrorCode.RETRY_NOT_SAFE,
        reason,
        details={
            "task_id": task.id,
            "state": task.state,
            "error_code": task.error_code,
            **details,
        },
    )


# ---------------------------------------------------------------------------
# POST /episodes/{id}/pipeline-runs
# ---------------------------------------------------------------------------
def reprocess_episode(
    session: Session,
    *,
    principal: Principal,
    episode_id: str,
    pipeline_version: str,
    idempotency_key: str | None,
    if_match: str | None,
    request_id: str | None = None,
) -> dict[str, Any]:
    body = {"pipeline_version": pipeline_version}
    if pipeline_version not in SUPPORTED_PIPELINE_VERSIONS:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            f"pipeline_version must be one of {sorted(SUPPORTED_PIPELINE_VERSIONS)}",
            details={"pipeline_version": pipeline_version},
            status_code=422,
        )

    def lock(session_: Session) -> Episode | None:
        return session_.scalars(
            select(Episode).where(Episode.id == episode_id).with_for_update()
        ).first()

    def effect(
        session_: Session, episode: Episode, payload: dict[str, Any], operation_id: str
    ) -> tuple[str, str]:
        run = create_pipeline_run(
            session_,
            episode,
            requested_by=principal.key_id,
            pipeline_version=payload["pipeline_version"],
        )
        if run.state not in (RunState.QUEUED.value, RunState.RUNNING.value):
            raise ApiError(
                ErrorCode.INVALID_STATE,
                "New pipeline run was not queued",
                details={"run_id": run.id, "state": run.state},
            )
        return "pipeline_run", run.id

    return execute_controlled_write(
        session,
        principal=principal,
        action_kind=ACTION_REPROCESS_EPISODE,
        etag_kind=ETAG_EPISODE,
        echo_field="new_run_id",
        target_id=episode_id,
        body=body,
        idempotency_key=idempotency_key,
        if_match=if_match,
        lock_resource=lock,
        apply_effect=effect,
        request_id=request_id,
        audit_entity_type="episode",
    )


# ---------------------------------------------------------------------------
# POST /tasks/{id}/retry
# ---------------------------------------------------------------------------
def retry_failed_task(
    session: Session,
    *,
    principal: Principal,
    task_id: str,
    expected_run_id: str,
    idempotency_key: str | None,
    if_match: str | None,
    request_id: str | None = None,
) -> dict[str, Any]:
    body = {"expected_run_id": expected_run_id}

    def lock(session_: Session) -> Task | None:
        return session_.scalars(
            select(Task).where(Task.id == task_id).with_for_update()
        ).first()

    def effect(
        session_: Session, task: Task, payload: dict[str, Any], operation_id: str
    ) -> tuple[str, str]:
        expected = str(payload["expected_run_id"])
        if task.state != TaskState.FAILED.value:
            raise _retry_not_safe(task, "Only FAILED tasks can be retried")
        if not is_task_retryable(task):
            raise _retry_not_safe(
                task, "The task failed with a non-retryable error; fix the source and import again"
            )
        if task.run_id != expected:
            raise _retry_not_safe(
                task,
                "expected_run_id does not match the task's pipeline run",
                expected_run_id=expected,
                actual_run_id=task.run_id,
            )
        run = session_.get(PipelineRun, expected)
        if run is None:
            raise _retry_not_safe(task, "Task has no pipeline run to retry")
        episode = session_.get(Episode, run.episode_id)
        if episode is None:  # pragma: no cover - FK guarantees the episode exists
            raise _retry_not_safe(task, "Episode for the failed task no longer exists")
        new_run = create_pipeline_run(
            session_,
            episode,
            requested_by=principal.key_id,
            pipeline_version=V2_PIPELINE_VERSION,
            from_step=Step(task.step),
        )
        # The episode status changed (business precondition) -> bump its revision too.
        revisions.bump(session_, episode)
        return "pipeline_run", new_run.id

    return execute_controlled_write(
        session,
        principal=principal,
        action_kind=ACTION_RETRY_FAILED_TASK,
        etag_kind=ETAG_TASK,
        echo_field="new_run_id",
        target_id=task_id,
        body=body,
        idempotency_key=idempotency_key,
        if_match=if_match,
        lock_resource=lock,
        apply_effect=effect,
        request_id=request_id,
        audit_entity_type="task",
    )


# ---------------------------------------------------------------------------
# POST /evaluations/{id}/analyze-gaps
# ---------------------------------------------------------------------------
def analyze_evaluation_gaps(
    session: Session,
    *,
    principal: Principal,
    evaluation_id: str,
    success_rate_threshold: float,
    min_trials: int,
    idempotency_key: str | None,
    if_match: str | None,
    request_id: str | None = None,
) -> dict[str, Any]:
    body = {
        "success_rate_threshold": float(success_rate_threshold),
        "min_trials": int(min_trials),
    }

    def lock(session_: Session) -> EvaluationReport | None:
        return session_.scalars(
            select(EvaluationReport)
            .where(EvaluationReport.id == evaluation_id)
            .with_for_update()
        ).first()

    def effect(
        session_: Session,
        report: EvaluationReport,
        payload: dict[str, Any],
        operation_id: str,
    ) -> tuple[str, str]:
        # The report must be readable (asset digest + evaluation/1.0) before a
        # GapAnalysis may be queued; otherwise the Agent gets a precise 409.
        evaluation_results(session_, report)
        existing = session_.scalars(
            select(GapAnalysis).where(GapAnalysis.evaluation_report_id == report.id)
        ).first()
        if existing is not None:
            raise ApiError(
                ErrorCode.INVALID_STATE,
                "A gap analysis already exists for this evaluation report",
                details={
                    "evaluation_id": report.id,
                    "analysis_id": existing.id,
                    "analysis_state": existing.state,
                },
            )
        analysis = GapAnalysis(
            id=new_id(),
            evaluation_report_id=report.id,
            state="QUEUED",
            threshold=float(payload["success_rate_threshold"]),
            min_trials=int(payload["min_trials"]),
            summary_json={},
            created_at=utcnow(),
        )
        session_.add(analysis)
        session_.flush()
        session_.add(
            OutboxEvent(
                id=new_id(),
                topic=FEEDBACK_TOPIC,
                payload_json={
                    "kind": "gap_analysis",
                    "gap_analysis_id": analysis.id,
                    "evaluation_id": report.id,
                },
                created_at=utcnow(),
            )
        )
        return "gap_analysis", analysis.id

    return execute_controlled_write(
        session,
        principal=principal,
        action_kind=ACTION_ANALYZE_EVALUATION_GAPS,
        etag_kind=ETAG_EVALUATION,
        echo_field="analysis_id",
        target_id=evaluation_id,
        body=body,
        idempotency_key=idempotency_key,
        if_match=if_match,
        lock_resource=lock,
        apply_effect=effect,
        request_id=request_id,
        audit_entity_type="evaluation_report",
    )


__all__ = [
    "FEEDBACK_TOPIC",
    "NON_RETRYABLE_ERROR_CODES",
    "SUPPORTED_PIPELINE_VERSIONS",
    "TENANT_ID",
    "analyze_evaluation_gaps",
    "is_task_retryable",
    "reprocess_episode",
    "retry_failed_task",
]
