"""MERGE_CANDIDATE_ANNOTATIONS handler (SPEC_V2.1 §5.1, §14).

Runs once per Episode, after **every** per-clip VLM task succeeded (the DAG guarantees
this through one dependency edge per clip).  It freezes the candidate actions into one
``annotation/2.1`` document, which is what a human then reviews.

The merge is deliberately dumb: it copies validated rows, orders them by
``(clip_start_ns, start_ns, verb, noun)`` and records the provenance of every clip.  It
never edits an interval and never fills a gap -- a clip that answered
``insufficient_evidence`` contributes no actions, and that is recorded rather than
"repaired".
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from embodied_dataops.ai.service import canonical_payload
from embodied_dataops.canonical.contract import annotation_key
from embodied_dataops.db.models import (
    AIClipInput,
    AnnotationSet,
    CandidateActionAnnotation,
    Episode,
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
    register_bytes,
)

ANNOTATION_SCHEMA_V21 = "annotation/2.1"
ANNOTATION_KIND_ACTION_INTERVAL = "ACTION_INTERVAL"


class MergeCandidateAnnotationsHandler(TaskHandler):
    step = Step.MERGE_CANDIDATE_ANNOTATIONS

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        run, episode = _resolve_run(ctx)
        extract = ctx.session.scalars(
            select(Task).where(
                Task.run_id == run.id, Task.step == Step.EXTRACT_KEYFRAMES.value
            )
        ).first()
        if extract is None:
            raise TerminalTaskError(
                ErrorCode.INVALID_STATE,
                "EXTRACT_KEYFRAMES task is missing from this run",
                details={"run_id": run.id},
            )
        output = dict(extract.output_json or {})
        annotation_set_id = str(output.get("annotation_set_id") or "")
        planned = list(output.get("clips") or [])
        if not annotation_set_id or not planned:
            raise TerminalTaskError(
                ErrorCode.INVALID_STATE,
                "EXTRACT_KEYFRAMES output is incomplete (annotation_set_id / clips)",
                details={"run_id": run.id},
            )
        annotation_set = ctx.session.get(AnnotationSet, annotation_set_id)
        if annotation_set is None:
            raise TerminalTaskError(
                ErrorCode.NOT_FOUND,
                "AnnotationSet created by EXTRACT_KEYFRAMES is missing",
                details={"annotation_set_id": annotation_set_id},
            )

        clips: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        insufficient = 0
        modes: dict[str, int] = {}

        for planned_clip in sorted(planned, key=lambda item: int(item["start_ns"])):
            clip_id = str(planned_clip["clip_id"])
            clip = ctx.session.scalars(
                select(AIClipInput).where(AIClipInput.clip_id == clip_id)
            ).first()
            if clip is None:
                raise TerminalTaskError(
                    ErrorCode.NOT_FOUND,
                    "ai_clip_inputs row is missing during merge",
                    details={"clip_id": clip_id},
                )
            inference = ctx.session.scalars(
                select(ModelInference).where(
                    ModelInference.run_id == run.id,
                    ModelInference.clip_input_id == clip.id,
                )
            ).first()
            if inference is None or inference.status != "SUCCEEDED":
                raise TerminalTaskError(
                    ErrorCode.INVALID_STATE,
                    "merge runs only when every clip has a successful inference",
                    details={
                        "clip_id": clip_id,
                        "inference_status": getattr(inference, "status", None),
                    },
                )
            modes[inference.mode] = modes.get(inference.mode, 0) + 1
            rows = list(
                ctx.session.scalars(
                    select(CandidateActionAnnotation)
                    .where(CandidateActionAnnotation.model_inference_id == inference.id)
                    .order_by(
                        CandidateActionAnnotation.start_ns,
                        CandidateActionAnnotation.verb,
                        CandidateActionAnnotation.noun,
                    )
                )
            )
            if not rows:
                insufficient += 1
            for row in rows:
                actions.append(
                    {
                        "candidate_id": row.id,
                        "clip_id": clip_id,
                        "verb": row.verb,
                        "noun": row.noun,
                        "start_ns": int(row.start_ns),
                        "end_ns": int(row.end_ns),
                        "evidence_frame_indices": list(row.evidence_frame_indexes or []),
                        "uncertainty": row.uncertainty,
                        # The model's own word is not a calibrated probability.
                        "confidence_source": "MODEL_QUALITATIVE",
                        "origin": "MODEL",
                        "review_state": row.review_state,
                    }
                )
            clips.append(
                {
                    "clip_id": clip_id,
                    "start_ns": int(clip.start_ns),
                    "end_ns": int(clip.end_ns),
                    "frame_count": int(clip.frame_count),
                    "inference_id": inference.id,
                    "inference_mode": inference.mode,
                    "model_name": inference.model_name,
                    "prompt_version": inference.prompt_version,
                    "insufficient_evidence": not rows,
                    "action_count": len(rows),
                }
            )

        document = {
            "schema_version": ANNOTATION_SCHEMA_V21,
            "episode_id": episode.id,
            "annotation_kind": ANNOTATION_KIND_ACTION_INTERVAL,
            "annotation_set_id": annotation_set_id,
            "pipeline_version": "2.1.0",
            "provenance": {
                "clips": clips,
                "modes": modes,
                "prompt_version": _single(clips, "prompt_version"),
                "model_names": sorted({str(item.get("model_name")) for item in clips if item.get("model_name")}),
            },
            "actions": actions,
            "stats": {
                "clips": len(clips),
                "actions": len(actions),
                "insufficient_evidence_clips": insufficient,
                "live_clips": modes.get("LIVE", 0),
                "replay_clips": modes.get("REPLAY", 0),
            },
        }

        artifact = register_bytes(
            ctx,
            key=annotation_key(episode.id, annotation_set.id),
            data=canonical_payload(document),
            media_type="application/json",
            origin=AssetOrigin.DERIVED,
            logical_name="action_candidates",
            meta={
                "kind": "annotation_action_interval",
                "schema_version": ANNOTATION_SCHEMA_V21,
                "annotation_set_id": annotation_set.id,
                "clips": len(clips),
                "actions": len(actions),
                "modes": modes,
            },
        )
        annotation_set.artifact_asset_id = artifact.id
        annotation_set.summary_json = {
            **(annotation_set.summary_json or {}),
            "annotation_kind": ANNOTATION_KIND_ACTION_INTERVAL,
            "clips": len(clips),
            "actions": len(actions),
            "insufficient_evidence_clips": insufficient,
            "modes": modes,
        }

        return HandlerResult(
            data={
                "annotation_set_id": annotation_set.id,
                "annotation_kind": ANNOTATION_KIND_ACTION_INTERVAL,
                "artifact_key": annotation_key(episode.id, annotation_set.id),
                "artifact_asset_id": artifact.id,
                "artifact_sha256": artifact.sha256,
                "clips": len(clips),
                "actions": len(actions),
                "insufficient_evidence_clips": insufficient,
                "modes": modes,
            },
            output_asset_id=artifact.id,
        )


def _single(clips: list[dict[str, Any]], key: str) -> str | None:
    values = {str(item.get(key)) for item in clips if item.get(key)}
    return next(iter(values)) if len(values) == 1 else None


def _resolve_run(ctx: HandlerContext) -> tuple[PipelineRun, Episode]:
    task = ctx.task
    if not task.run_id:
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            "MERGE_CANDIDATE_ANNOTATIONS requires an episode-level PipelineRun",
            details={"task_id": task.id},
        )
    run = ctx.session.get(PipelineRun, task.run_id)
    episode = ctx.session.get(Episode, run.episode_id) if run is not None else None
    if run is None or episode is None:  # pragma: no cover - FK guarantees existence
        raise TerminalTaskError(
            ErrorCode.NOT_FOUND, "PipelineRun or Episode is missing", details={"run_id": task.run_id}
        )
    return run, episode


HANDLER = MergeCandidateAnnotationsHandler()

__all__ = ["ANNOTATION_KIND_ACTION_INTERVAL", "ANNOTATION_SCHEMA_V21", "HANDLER"]
