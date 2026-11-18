"""VALIDATE_AI_OUTPUT handler (SPEC_V2.1 §5.3, V21-08/12).

Produces the numbers the report has to quote, and nothing it cannot compute:

``schema_valid_rate``   successful validated inferences / attempted inferences
``call_success_rate``   HTTP-successful attempts / all attempts
``action_count``        validated candidate actions, plus how many clips abstained
``disagreement``        YOLO object labels vs VLM nouns, reported as ``WARN`` evidence

Two honesty rules are enforced here:

* a **REPLAY** run is reported with ``mode=REPLAY`` and never sets
  ``live_evidence=true``, so a recorded answer cannot be counted as a model result;
* a disagreement is *evidence for a human*, not an automatic verdict -- the handler does
  not decide which model is right (SPEC §5.3).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from embodied_dataops.ai.service import canonical_payload
from embodied_dataops.canonical.contract import annotation_key
from embodied_dataops.db.models import (
    AnnotationSet,
    CandidateActionAnnotation,
    Episode,
    ModelCallAttempt,
    ModelInference,
    PipelineRun,
    Task,
)
from embodied_dataops.domain.enums import AssetOrigin, Step
from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.workflow.handlers.base import (
    HandlerContext,
    HandlerResult,
    TaskHandler,
    read_json,
    register_bytes,
)

QUALITY_SCHEMA = "ai-quality-summary/2.1"

#: Nouns the model may emit that carry no claim about the world.
NON_COMMITTAL_NOUNS = frozenset({"unknown", "other", "object", "item"})


class ValidateAiOutputHandler(TaskHandler):
    step = Step.VALIDATE_AI_OUTPUT

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        run, episode = _resolve_run(ctx)
        extract = ctx.session.scalars(
            select(Task).where(
                Task.run_id == run.id, Task.step == Step.EXTRACT_KEYFRAMES.value
            )
        ).first()
        annotation_set_id = str((extract.output_json or {}).get("annotation_set_id") or "") if extract else ""
        if not annotation_set_id:
            raise TerminalTaskError(
                ErrorCode.INVALID_STATE,
                "EXTRACT_KEYFRAMES output has no annotation_set_id",
                details={"run_id": run.id},
            )
        annotation_set = ctx.session.get(AnnotationSet, annotation_set_id)
        if annotation_set is None:
            raise TerminalTaskError(
                ErrorCode.NOT_FOUND,
                "AnnotationSet is missing",
                details={"annotation_set_id": annotation_set_id},
            )

        inferences = list(
            ctx.session.scalars(select(ModelInference).where(ModelInference.run_id == run.id))
        )
        if not inferences:
            raise TerminalTaskError(
                ErrorCode.INVALID_STATE,
                "no model inference exists for this run",
                details={"run_id": run.id},
            )
        inference_ids = [row.id for row in inferences]
        attempts_total = ctx.session.scalar(
            select(func.count())
            .select_from(ModelCallAttempt)
            .where(ModelCallAttempt.inference_id.in_(inference_ids))
        ) or 0
        attempts_ok = ctx.session.scalar(
            select(func.count())
            .select_from(ModelCallAttempt)
            .where(
                ModelCallAttempt.inference_id.in_(inference_ids),
                ModelCallAttempt.response_status == 200,
            )
        ) or 0
        action_count = ctx.session.scalar(
            select(func.count())
            .select_from(CandidateActionAnnotation)
            .where(CandidateActionAnnotation.model_inference_id.in_(inference_ids))
        ) or 0
        cost_total = ctx.session.scalar(
            select(func.coalesce(func.sum(ModelCallAttempt.estimated_cost_usd), 0)).where(
                ModelCallAttempt.inference_id.in_(inference_ids)
            )
        )
        prompt_tokens = ctx.session.scalar(
            select(func.coalesce(func.sum(ModelCallAttempt.usage_prompt_tokens), 0)).where(
                ModelCallAttempt.inference_id.in_(inference_ids)
            )
        )
        completion_tokens = ctx.session.scalar(
            select(func.coalesce(func.sum(ModelCallAttempt.usage_completion_tokens), 0)).where(
                ModelCallAttempt.inference_id.in_(inference_ids)
            )
        )

        succeeded = sum(1 for row in inferences if row.status == "SUCCEEDED")
        modes = sorted({row.mode for row in inferences})
        schema_valid_rate = succeeded / len(inferences) if inferences else 0.0
        call_success_rate = (attempts_ok / attempts_total) if attempts_total else 0.0

        disagreements = _disagreements(ctx, run, episode, annotation_set)
        summary = {
            "schema_version": QUALITY_SCHEMA,
            "episode_id": episode.id,
            "annotation_set_id": annotation_set.id,
            "run_id": run.id,
            "mode": modes[0] if len(modes) == 1 else "MIXED",
            # A replay run can never be live evidence, even when every schema check passes.
            "live_evidence": modes == ["LIVE"],
            "model_names": sorted({row.model_name for row in inferences}),
            "prompt_versions": sorted({row.prompt_version for row in inferences}),
            "clips": len(inferences),
            "inferences_succeeded": succeeded,
            "inferences_failed": len(inferences) - succeeded,
            "schema_valid_rate": round(schema_valid_rate, 6),
            "calls": int(attempts_total),
            "calls_succeeded": int(attempts_ok),
            "call_success_rate": round(call_success_rate, 6),
            "action_count": int(action_count),
            "clips_without_actions": sum(
                1
                for row in inferences
                if not ctx.session.scalar(
                    select(func.count())
                    .select_from(CandidateActionAnnotation)
                    .where(CandidateActionAnnotation.model_inference_id == row.id)
                )
            ),
            "usage": {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "estimated_cost_usd": (float(cost_total) if cost_total is not None else None),
                "cost_known": cost_total is not None and float(cost_total) > 0,
            },
            "disagreements": disagreements,
            "review_state": annotation_set.status,
            "warnings": [item["code"] for item in disagreements],
        }

        key = f"{annotation_set_prefix(episode.id, annotation_set.id)}/ai_quality_summary.json"
        artifact = register_bytes(
            ctx,
            key=key,
            data=canonical_payload(summary),
            media_type="application/json",
            origin=AssetOrigin.REPORT,
            logical_name="ai_quality_summary",
            meta={
                "kind": "ai_quality_summary",
                "schema_version": QUALITY_SCHEMA,
                "mode": summary["mode"],
                "live_evidence": summary["live_evidence"],
            },
        )
        annotation_set.summary_json = {
            **(annotation_set.summary_json or {}),
            "quality": {
                "schema_valid_rate": summary["schema_valid_rate"],
                "call_success_rate": summary["call_success_rate"],
                "action_count": summary["action_count"],
                "mode": summary["mode"],
                "live_evidence": summary["live_evidence"],
                "warnings": summary["warnings"],
            },
        }

        return HandlerResult(
            data={
                "annotation_set_id": annotation_set.id,
                "summary_key": key,
                "summary_asset_id": artifact.id,
                "summary_sha256": artifact.sha256,
                **{
                    key_name: summary[key_name]
                    for key_name in (
                        "mode",
                        "live_evidence",
                        "schema_valid_rate",
                        "call_success_rate",
                        "clips",
                        "action_count",
                        "warnings",
                    )
                },
            },
            output_asset_id=artifact.id,
        )


def annotation_set_prefix(episode_id: str, annotation_set_id: str) -> str:
    # Reuse the annotation namespace so everything about one set lives together.
    return annotation_key(episode_id, annotation_set_id).rsplit("/", 1)[0]


def _disagreements(
    ctx: HandlerContext, run: PipelineRun, episode: Episode, annotation_set: AnnotationSet
) -> list[dict[str, Any]]:
    """YOLO labels vs VLM nouns.  A WARN for a human, never an automatic verdict."""
    detect = ctx.session.scalars(
        select(Task).where(Task.run_id == run.id, Task.step == Step.DETECT_OBJECTS.value)
    ).first()
    detection_set_id = str((detect.output_json or {}).get("annotation_set_id") or "") if detect else ""
    if not detection_set_id:
        return []
    try:
        document = read_json(ctx, annotation_key(episode.id, detection_set_id))
    except Exception:  # pragma: no cover - a missing artifact must not fail validation
        return []
    detected = {
        str(item.get("class_name"))
        for frame in document.get("frames") or []
        for item in frame.get("detections") or []
        if item.get("class_name")
    }
    if not detected:
        return []

    rows = list(
        ctx.session.scalars(
            select(CandidateActionAnnotation).where(
                CandidateActionAnnotation.annotation_set_id == annotation_set.id
            )
        )
    )
    findings: list[dict[str, Any]] = []
    for row in rows:
        noun = str(row.noun).lower()
        if noun in NON_COMMITTAL_NOUNS or noun in detected:
            continue
        # Singular/plural and compound names ("coffee_mug" vs "cup") are the common case;
        # this stays a soft signal on purpose instead of a fuzzy matcher that hides errors.
        if any(noun in label or label in noun for label in detected):
            continue
        findings.append(
            {
                "code": "MODEL_DISAGREEMENT",
                "severity": "WARN",
                "candidate_id": row.id,
                "vlm_noun": row.noun,
                "yolo_classes": sorted(detected),
                "note": "YOLO COCO labels cannot represent every hand-object interaction; "
                "a human must decide, the platform does not",
            }
        )
    return findings[:20]


def _resolve_run(ctx: HandlerContext) -> tuple[PipelineRun, Episode]:
    task = ctx.task
    if not task.run_id:
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            "VALIDATE_AI_OUTPUT requires an episode-level PipelineRun",
            details={"task_id": task.id},
        )
    run = ctx.session.get(PipelineRun, task.run_id)
    episode = ctx.session.get(Episode, run.episode_id) if run is not None else None
    if run is None or episode is None:  # pragma: no cover - FK guarantees existence
        raise TerminalTaskError(
            ErrorCode.NOT_FOUND, "PipelineRun or Episode is missing", details={"run_id": task.run_id}
        )
    return run, episode


HANDLER = ValidateAiOutputHandler()

__all__ = ["HANDLER", "QUALITY_SCHEMA", "ValidateAiOutputHandler", "annotation_set_prefix"]
