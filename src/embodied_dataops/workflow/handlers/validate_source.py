"""VALIDATE_SOURCE handler (SPEC 5.1, 6.1).

Import-level step: it verifies the immutable upload object, runs the cheap
structural validation of the matching adapter (archive safety + ``meta/info.json``
for LeRobot, probe + full decode for Ego MP4) and records everything as task
evidence.

Failure policy (see ``imports.common.record_import_failure``): an *expected*
source problem -- corrupt media, unsafe archive, invalid metadata, missing
mapping -- marks the ImportJob/SourceImport ``FAILED`` and returns
``verdict="FAIL"`` so the engine cancels the downstream IMPORT_EPISODES step
without creating Episodes.  Unexpected infrastructure errors are raised and are
retried by the engine.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from embodied_dataops.domain.enums import ImportJobState, Step, UploadState
from embodied_dataops.errors import ApiError
from embodied_dataops.imports.base import get_parser
from embodied_dataops.imports.common import (
    import_context_payload,
    load_import_job,
    load_source_import,
    materialize_import_source,
    record_import_failure,
    utcnow,
)
from embodied_dataops.processing.ffmpeg import media_backend_summary
from embodied_dataops.workflow.handlers.base import HandlerContext, HandlerResult, TaskHandler


class ValidateSourceHandler(TaskHandler):
    step = Step.VALIDATE_SOURCE

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        session: Session = ctx.session
        job = load_import_job(session, ctx.task.import_job_id)
        source_import = load_source_import(session, job.import_id)
        job.state = ImportJobState.RUNNING.value
        job.started_at = job.started_at or utcnow()
        source_import.upload_state = UploadState.VALIDATING.value
        session.flush()

        try:
            bundle = materialize_import_source(
                session, ctx.storage, job, source_import, ctx.workspace
            )
            parser = get_parser(source_import.source_type)
            validation = parser.validate(bundle.context)
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

        data: dict[str, Any] = {
            "step": self.step.value,
            "valid": True,
            "source_type": source_import.source_type,
            "source": import_context_payload(bundle),
            "validation": validation,
            "media_backend": media_backend_summary(),
            "validated_at": utcnow().isoformat(),
        }
        return HandlerResult(data=data)


#: Module-level handler object understood by the workflow handler registry.
HANDLER = ValidateSourceHandler()

__all__ = ["HANDLER", "ValidateSourceHandler"]
