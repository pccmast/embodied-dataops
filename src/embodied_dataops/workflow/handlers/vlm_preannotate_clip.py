"""VLM_PREANNOTATE_CLIP handler (SPEC_V2.1 §5.2–§5.4, V21-08/09/11/12/13).

One task per clip (``tasks.unit_key == clip_id``).  The handler is the only place that

* refuses to send bytes that the source has not cleared for egress;
* creates exactly one ``model_inferences`` row per (run, clip, provider, model, prompt)
  and lets :func:`perform_inference` append ``model_call_attempts`` rows;
* converts a *validated* answer into ``candidate_action_annotations`` rows.

**Two transactions, on purpose.**  The inference identity, its budget reservation and
every attempt row are written through :mod:`embodied_dataops.ai.audit` -- an
independent session that commits immediately.  A failed clip raises a terminal task
error and the engine rolls the task transaction back; if the audit rows lived in that
transaction, the record of a real, possibly billed remote call would disappear with
it (SPEC_V2.1 §5.4).  The *product* (``output_asset_id`` + candidate rows) stays in
the task transaction, so "one valid committed product per inference" is still atomic.

A failed clip is a **terminal** task failure with an explicit error code.  It is never
"success with no candidates": that would let an empty model answer look like a studied
conclusion.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import func, select

from embodied_dataops.ai.audit import audit_scope
from embodied_dataops.ai.frames import EncodedKeyframe
from embodied_dataops.ai.schema import insufficient_evidence_document
from embodied_dataops.ai.service import (
    build_provider,
    build_request,
    canonical_payload,
    load_prompt,
    perform_inference,
    provider_mode,
    reserve_budget,
    settle_inference,
)
from embodied_dataops.db.models import (
    AIClipInput,
    CandidateActionAnnotation,
    Episode,
    ModelCallAttempt,
    ModelInference,
    PipelineRun,
    SourceImport,
    Task,
)
from embodied_dataops.domain.enums import AssetOrigin, Step
from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.handlers.base import (
    HandlerContext,
    HandlerResult,
    TaskHandler,
    register_bytes,
)
from embodied_dataops.workflow.handlers.extract_keyframes import (
    frame_key,
    manifest_key,
    output_key,
)


class VlmPreannotateClipHandler(TaskHandler):
    """Call the configured VLM for one clip and persist the candidate actions."""

    step = Step.VLM_PREANNOTATE_CLIP

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        from embodied_dataops.workflow.handlers.base import read_json

        task = ctx.task
        clip_id = task.unit_key or ""
        if not clip_id:
            raise TerminalTaskError(
                ErrorCode.INVALID_STATE,
                "VLM_PREANNOTATE_CLIP requires tasks.unit_key = clip_id",
                details={"task_id": task.id},
            )
        run, episode = _resolve_run(ctx)
        clip = ctx.session.scalars(
            select(AIClipInput).where(AIClipInput.clip_id == clip_id)
        ).first()
        if clip is None:
            raise TerminalTaskError(
                ErrorCode.NOT_FOUND,
                "ai_clip_inputs row is missing for this clip",
                details={"clip_id": clip_id},
            )
        if clip.episode_id != episode.id:
            raise TerminalTaskError(
                ErrorCode.INVALID_INPUT,
                "clip belongs to a different Episode",
                details={"clip_episode_id": clip.episode_id, "episode_id": episode.id},
            )

        annotation_set_id = _annotation_set_id(ctx, run.id)
        prompt = load_prompt(ctx.settings)
        provider = build_provider(ctx.settings, prompt)
        mode = provider_mode(provider)

        if mode == "LIVE":
            _require_egress_allowed(ctx, episode)

        manifest = read_json(ctx, manifest_key(episode.id, clip_id))
        frames = _load_frames(ctx, episode.id, clip_id, manifest)
        request = build_request(
            clip_id=clip_id,
            clip_start_ns=clip.start_ns,
            clip_end_ns=clip.end_ns,
            frames=frames,
            prompt=prompt,
            request_id=task.id,
        )

        with audit_scope(ctx.session) as audit:
            inference, prior_attempts = _get_or_create_inference(
                audit,
                run=run,
                task_id=task.id,
                clip=clip,
                provider_name=str(getattr(provider, "name", "unknown")),
                model_name=str(getattr(provider, "model_name", "unknown")),
                mode=mode,
                prompt_sha256=prompt.sha256,
                prompt_version=prompt.version,
                request_sha256=request.request_sha256(),
            )
            audit.commit()

            # Admission reads the balance under the run-row lock, *before* this call
            # writes its own reservation; committing that reservation is what makes the
            # spend visible to the next worker.
            admitted, budget = reserve_budget(audit, inference=inference, settings=ctx.settings)
            audit.commit()
            if not admitted:
                raise TerminalTaskError(
                    ErrorCode.VLM_BUDGET_EXCEEDED,
                    "run budget is exhausted; no remote call was made",
                    details={
                        "clip_id": clip_id,
                        "run_id": run.id,
                        "limit_usd": budget.limit_usd,
                        "committed_usd": budget.committed_usd,
                        "reserved_usd": budget.running_reserved_usd,
                        "estimate_usd": budget.estimate_usd,
                        "inference_id": inference.id,
                    },
                )

            outcome = perform_inference(
                audit,
                inference=inference,
                provider=provider,
                request=request,
                settings=ctx.settings,
                clip_start_ns=clip.start_ns,
                clip_end_ns=clip.end_ns,
                sent_positions=[frame.position for frame in frames],
                admitted=True,
                persist=audit.commit,
                attempt_offset=prior_attempts,
            )

            if not outcome.succeeded or outcome.result is None or outcome.validation is None:
                settle_inference(
                    audit,
                    inference=inference,
                    status=(
                        "BLOCKED"
                        if outcome.error_code == "SOURCE_EGRESS_NOT_ALLOWED"
                        else "FAILED"
                    ),
                    error_code=outcome.error_code,
                    cost_usd=outcome.cost_usd,
                )
                audit.commit()
                raise TerminalTaskError(
                    _error_code(outcome.error_code),
                    "clip inference did not produce a valid answer",
                    details={
                        "clip_id": clip_id,
                        "attempts": outcome.attempts,
                        "attempt_offset": prior_attempts,
                        "error_code": outcome.error_code,
                        "cost_state": outcome.cost_state,
                        "detail": outcome.detail[:400],
                        "inference_id": inference.id,
                    },
                )

            inference_id = inference.id
            model_name = inference.model_name
            latency_ms = int(outcome.result.latency_ms)
            cost_usd = outcome.cost_usd
            attempts_used = outcome.attempts

        document = (
            outcome.validation.document.model_dump(mode="json")
            if outcome.validation.document is not None
            else insufficient_evidence_document(clip.start_ns, clip.end_ns)
        )
        artifact = register_bytes(
            ctx,
            key=output_key(episode.id, clip_id),
            data=canonical_payload(document),
            media_type="application/json",
            origin=AssetOrigin.DERIVED,
            logical_name=f"{clip_id}-vlm-output",
            meta={
                "kind": "ai_clip_output",
                "clip_id": clip_id,
                "schema_version": document["schema_version"],
                "model_name": model_name,
                "mode": mode,
                "prompt_version": prompt.version,
                "action_count": len(document.get("actions") or []),
            },
        )
        # The product is committed with the task transaction; the audit row was left
        # RUNNING (reserved) by the audit session so a lost commit never claims success.
        inference = ctx.session.get(ModelInference, inference_id, populate_existing=True)
        if inference is None:  # pragma: no cover - the audit row is committed by now
            raise TerminalTaskError(
                ErrorCode.INTERNAL_ERROR,
                "the committed model inference row disappeared",
                details={"inference_id": inference_id},
            )
        inference.output_asset_id = artifact.id
        inference.output_sha256 = artifact.sha256
        inference.status = "SUCCEEDED"
        inference.error_code = None
        inference.latency_ms = latency_ms
        inference.finished_at = _now()
        if cost_usd is not None:
            inference.estimated_cost_usd = Decimal(str(cost_usd))
        ctx.session.flush()

        candidate_ids = _persist_candidates(
            ctx,
            annotation_set_id=annotation_set_id,
            clip=clip,
            inference=inference,
            document=document,
        )

        return HandlerResult(
            data={
                "clip_id": clip_id,
                "inference_id": inference.id,
                "annotation_set_id": annotation_set_id,
                "mode": mode,
                "model_name": inference.model_name,
                "prompt_version": prompt.version,
                "prompt_sha256": prompt.sha256,
                "attempts": attempts_used,
                "action_count": len(document.get("actions") or []),
                "insufficient_evidence": bool(document.get("insufficient_evidence")),
                "candidate_ids": candidate_ids,
                "output_sha256": artifact.sha256,
                "latency_ms": inference.latency_ms,
                "estimated_cost_usd": (
                    float(inference.estimated_cost_usd)
                    if inference.estimated_cost_usd is not None
                    else None
                ),
            },
            output_asset_id=artifact.id,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now():
    from datetime import UTC, datetime  # noqa: PLC0415

    return datetime.now(UTC)


def _error_code(code: str | None):
    mapping = {
        "VLM_PROVIDER_UNAVAILABLE": ErrorCode.VLM_PROVIDER_UNAVAILABLE,
        "VLM_BUDGET_EXCEEDED": ErrorCode.VLM_BUDGET_EXCEEDED,
        "MODEL_OUTPUT_SCHEMA_INVALID": ErrorCode.MODEL_OUTPUT_SCHEMA_INVALID,
        "SOURCE_EGRESS_NOT_ALLOWED": ErrorCode.SOURCE_EGRESS_NOT_ALLOWED,
    }
    return mapping.get(code or "", ErrorCode.DEPENDENCY_UNAVAILABLE)


def _resolve_run(ctx: HandlerContext) -> tuple[PipelineRun, Episode]:
    task = ctx.task
    if not task.run_id:
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            "VLM_PREANNOTATE_CLIP requires an episode-level PipelineRun",
            details={"task_id": task.id},
        )
    run = ctx.session.get(PipelineRun, task.run_id)
    if run is None:
        raise TerminalTaskError(ErrorCode.NOT_FOUND, "PipelineRun is missing", details={"run_id": task.run_id})
    episode = ctx.session.get(Episode, run.episode_id)
    if episode is None:
        raise TerminalTaskError(
            ErrorCode.NOT_FOUND, "Episode is missing", details={"episode_id": run.episode_id}
        )
    return run, episode


def _annotation_set_id(ctx: HandlerContext, run_id: str) -> str:
    task = ctx.session.scalars(
        select(Task).where(
            Task.run_id == run_id, Task.step == Step.EXTRACT_KEYFRAMES.value
        )
    ).first()
    value = (task.output_json or {}).get("annotation_set_id") if task else None
    if not isinstance(value, str) or not value:
        raise TerminalTaskError(
            ErrorCode.INVALID_STATE,
            "EXTRACT_KEYFRAMES did not record an annotation_set_id for this run",
            details={"run_id": run_id},
        )
    return value


def _require_egress_allowed(ctx: HandlerContext, episode: Episode) -> None:
    """A live call needs an explicit, recorded authorisation on the source import."""
    source = ctx.session.get(SourceImport, episode.source_import_id)
    if source is None:
        raise TerminalTaskError(
            ErrorCode.NOT_FOUND,
            "SourceImport is missing for this Episode",
            details={"episode_id": episode.id},
        )
    if not bool(source.egress_allowed):
        raise TerminalTaskError(
            ErrorCode.SOURCE_EGRESS_NOT_ALLOWED,
            "the source is not cleared for outbound model calls (egress_allowed=false)",
            details={
                "episode_id": episode.id,
                "import_id": source.id,
                "data_origin": source.data_origin,
            },
        )


def _load_frames(
    ctx: HandlerContext, episode_id: str, clip_id: str, manifest: dict[str, Any]
) -> tuple[EncodedKeyframe, ...]:
    frames: list[EncodedKeyframe] = []
    for entry in manifest.get("frames") or []:
        position = int(entry["position"])
        data = ctx.storage.get_bytes(frame_key(episode_id, clip_id, position))
        frames.append(
            EncodedKeyframe(
                position=position,
                grid_index=int(entry["grid_index"]),
                t_ns=int(entry["t_ns"]),
                jpeg=data,
                sha256=str(entry["sha256"]),
                width=int(entry.get("width") or 0),
                height=int(entry.get("height") or 0),
                source_width=int(entry.get("source_width") or 0),
                source_height=int(entry.get("source_height") or 0),
            )
        )
    if not frames:
        raise TerminalTaskError(
            ErrorCode.INVALID_STATE,
            "frame manifest contains no frames",
            details={"clip_id": clip_id},
        )
    return tuple(frames)


def _find_inference(
    session,
    *,
    run: PipelineRun,
    clip: AIClipInput,
    provider_name: str,
    model_name: str,
    prompt_sha256: str,
) -> ModelInference | None:
    return session.scalars(
        select(ModelInference).where(
            ModelInference.run_id == run.id,
            ModelInference.clip_input_id == clip.id,
            ModelInference.provider == provider_name,
            ModelInference.model_name == model_name,
            ModelInference.prompt_sha256 == prompt_sha256,
        )
    ).first()


def _get_or_create_inference(
    session,
    *,
    run: PipelineRun,
    task_id: str,
    clip: AIClipInput,
    provider_name: str,
    model_name: str,
    mode: str,
    prompt_sha256: str,
    prompt_version: str,
    request_sha256: str,
) -> tuple[ModelInference, int]:
    """At most one inference per (run, clip, provider, model, prompt) -- enforced here too.

    Returns the row and how many attempts it already recorded, so a retried task cannot
    restart the attempt counter and slip past the 3-attempt ceiling.

    The SELECT-then-INSERT below is not atomic on its own.  It only loses a race when a
    *stale* worker is still alive after its lease expired (clock skew or a suspended
    process): both workers miss the row and both insert.  The unique key keeps the data
    correct but raises ``IntegrityError``; losing worker B would then crash a task that
    has nothing wrong with it.  Re-reading after a rollback turns that into the normal
    "reuse the committed row" path.  This is safe because the caller owns an independent
    audit transaction that holds nothing else at this point.
    """
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415

    existing = _find_inference(
        session,
        run=run,
        clip=clip,
        provider_name=provider_name,
        model_name=model_name,
        prompt_sha256=prompt_sha256,
    )
    if existing is None:
        row = ModelInference(
            id=new_id(),
            run_id=run.id,
            task_id=task_id,
            clip_input_id=clip.id,
            provider=provider_name,
            model_name=model_name,
            mode=mode,
            prompt_version=prompt_version,
            prompt_sha256=prompt_sha256,
            request_sha256=request_sha256,
            status="QUEUED",
        )
        session.add(row)
        try:
            session.flush()
            existing = row
        except IntegrityError:
            session.rollback()
            existing = _find_inference(
                session,
                run=run,
                clip=clip,
                provider_name=provider_name,
                model_name=model_name,
                prompt_sha256=prompt_sha256,
            )
            if existing is None:  # pragma: no cover - a different constraint failed
                raise
    # A retried task reuses the row and therefore cannot create a second product.
    existing.task_id = task_id
    existing.request_sha256 = request_sha256
    session.flush()
    prior_attempts = int(
        session.scalar(
            select(func.count())
            .select_from(ModelCallAttempt)
            .where(ModelCallAttempt.inference_id == existing.id)
        )
        or 0
    )
    return existing, prior_attempts


def _persist_candidates(
    ctx: HandlerContext,
    *,
    annotation_set_id: str,
    clip: AIClipInput,
    inference: ModelInference,
    document: dict[str, Any],
) -> list[str]:
    """Replace this inference's candidate rows with the validated answer."""
    existing = list(
        ctx.session.scalars(
            select(CandidateActionAnnotation).where(
                CandidateActionAnnotation.model_inference_id == inference.id
            )
        )
    )
    for row in existing:
        ctx.session.delete(row)
    ctx.session.flush()

    created: list[str] = []
    for index, action in enumerate(document.get("actions") or []):
        row = CandidateActionAnnotation(
            id=new_id(),
            annotation_set_id=annotation_set_id,
            model_inference_id=inference.id,
            clip_input_id=clip.id,
            action_index=index,
            verb=str(action["verb"]),
            noun=str(action["noun"]),
            start_ns=int(action["start_ns"]),
            end_ns=int(action["end_ns"]),
            evidence_frame_indexes=list(action["evidence_frame_indices"]),
            uncertainty=str(action["uncertainty"]),
            review_state="PENDING",
        )
        ctx.session.add(row)
        created.append(row.id)
    ctx.session.flush()
    return created


HANDLER = VlmPreannotateClipHandler()

__all__ = ["HANDLER", "VlmPreannotateClipHandler"]
