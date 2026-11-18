"""EXTRACT_KEYFRAMES handler (SPEC_V2.1 §5.1, V21-07).

One task per Episode run.  It:

1. resolves the Episode's **video time base** -- canonical-episode/2.0 (robot
   episodes) or canonical-episode/1.0 (``ego_video_episode``, which has no
   state/action by contract).  The clip timeline must come from recorded metadata,
   never from a filename or a default duration;
2. plans the fixed 8 s / 1 fps clips on the episode timeline;
3. decodes and encodes the keyframes, storing each JPEG plus a frame manifest;
4. creates one ``ai_clip_inputs`` row per clip;
5. creates the ``AnnotationSet`` that will hold the candidate actions;
6. **fans out one ``VLM_PREANNOTATE_CLIP`` task per clip** and wires every one of them as
   a dependency of the single ``MERGE_CANDIDATE_ANNOTATIONS`` task.

An Ego video never gains a state/action vector here: this stage only reads a video
and its recorded timestamps.  That is why the same handler serves both episode
types without inventing robot channels (SPEC_V2.1 §0.1, §4.3).

Step 6 is the reason ``tasks`` gained ``unit_key``: without it the per-run uniqueness of
``(run_id, step)`` would collapse the whole fan-out into one task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from embodied_dataops.ai.frames import encode_clip_keyframes, frame_manifest
from embodied_dataops.canonical.contract import canonical_keys
from embodied_dataops.canonical.v2.contract import (
    CANONICAL_V2_SCHEMA_VERSION,
    canonical_v2_keys,
)
from embodied_dataops.db.models import (
    AIClipInput,
    AnnotationSet,
    Episode,
    PipelineRun,
    Task,
    TaskDependency,
)
from embodied_dataops.domain.enums import (
    AnnotationStatus,
    AssetOrigin,
    EpisodeType,
    Step,
    TaskState,
)
from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.ids import new_id
from embodied_dataops.processing.clips import plan_clips
from embodied_dataops.workflow.dag import input_digest
from embodied_dataops.workflow.handlers.base import (
    HandlerContext,
    HandlerResult,
    TaskHandler,
    register_bytes,
    register_json,
)
from embodied_dataops.workflow.outbox import enqueue_task_wakeup

#: Preview key written by the canonical v2 normalizer (SPEC_V2.1 §4.3).
PREVIEW_NAME = "front_10hz_640x480.mp4"


def ai_prefix(episode_id: str, clip_id: str) -> str:
    return f"ai/{episode_id}/{clip_id}"


def frame_key(episode_id: str, clip_id: str, position: int) -> str:
    return f"{ai_prefix(episode_id, clip_id)}/frame_{position:02d}.jpg"


def manifest_key(episode_id: str, clip_id: str) -> str:
    return f"{ai_prefix(episode_id, clip_id)}/frame_manifest.json"


def output_key(episode_id: str, clip_id: str) -> str:
    return f"{ai_prefix(episode_id, clip_id)}/output.json"


class ExtractKeyframesHandler(TaskHandler):
    """Plan clips and fan out the per-clip VLM tasks."""

    step = Step.EXTRACT_KEYFRAMES

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        run, episode = _resolve_run(ctx)
        if episode.episode_type != EpisodeType.ROBOT_EPISODE.value and (
            episode.episode_type != EpisodeType.EGO_VIDEO_EPISODE.value
        ):  # pragma: no cover - enum guard
            raise TerminalTaskError(
                ErrorCode.INVALID_INPUT,
                "unknown episode_type for AI pre-annotation",
                details={"episode_type": episode.episode_type},
            )

        source = _resolve_video_source(ctx, episode)
        metadata = source.metadata
        duration_ns = source.duration_ns
        if duration_ns <= 0:
            raise TerminalTaskError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "canonical metadata reports a non-positive duration",
                details={"duration_ns": duration_ns, "metadata_key": source.metadata_key},
            )
        aligned_fps = source.fps

        plan = plan_clips(
            episode_id=episode.id,
            source_asset_sha256=str(metadata.get("source_asset_sha256") or ""),
            duration_ns=duration_ns,
            aligned_fps=aligned_fps,
        )

        annotation_set = AnnotationSet(
            id=new_id(),
            episode_id=episode.id,
            run_id=run.id,
            model_id=str(metadata.get("pipeline_version") or "2.1.0"),
            model_sha256=str(metadata.get("source_asset_sha256") or ""),
            status=AnnotationStatus.PENDING_REVIEW.value,
            summary_json={"annotation_kind": "ACTION_INTERVAL", "clips_planned": plan.clip_count},
        )
        ctx.session.add(annotation_set)
        ctx.session.flush()

        video_path = ctx.workspace / "canonical_preview.mp4"
        ctx.storage.download_to(source.video_key, video_path)

        merge_task = _merge_task(ctx, run.id)
        if merge_task is None:
            raise TerminalTaskError(
                ErrorCode.INVALID_STATE,
                "the run has no MERGE_CANDIDATE_ANNOTATIONS task; it was not started with the "
                "V2.1 DAG",
                details={"run_id": run.id, "pipeline_version": run.pipeline_version},
            )
        created: list[dict[str, Any]] = []
        for clip in plan.clips:
            frames = encode_clip_keyframes(
                video_path,
                clip,
                max_side_px=ctx.settings.vlm_max_side_px,
                jpeg_quality=ctx.settings.vlm_jpeg_quality,
            )
            for frame in frames:
                register_bytes(
                    ctx,
                    key=frame_key(episode.id, clip.clip_id, frame.position),
                    data=frame.jpeg,
                    media_type="image/jpeg",
                    origin=AssetOrigin.DERIVED,
                    logical_name=f"{clip.clip_id}-frame-{frame.position:02d}",
                    meta={
                        "clip_id": clip.clip_id,
                        "position": frame.position,
                        "grid_index": frame.grid_index,
                        "t_ns": frame.t_ns,
                        "width": frame.width,
                        "height": frame.height,
                    },
                )
            manifest = frame_manifest(
                clip_id=clip.clip_id,
                episode_id=episode.id,
                plan_episode_sha=str(metadata.get("source_asset_sha256") or ""),
                frames=frames,
                max_side_px=ctx.settings.vlm_max_side_px,
                jpeg_quality=ctx.settings.vlm_jpeg_quality,
            )
            manifest_asset = register_json(
                ctx,
                key=manifest_key(episode.id, clip.clip_id),
                payload=manifest,
                origin=AssetOrigin.DERIVED,
                logical_name=f"{clip.clip_id}-frame-manifest",
                meta={"kind": "ai_clip_frames", "clip_id": clip.clip_id, "frame_count": len(frames)},
            )

            clip_row = AIClipInput(
                id=new_id(),
                episode_id=episode.id,
                clip_id=clip.clip_id,
                start_ns=clip.start_ns,
                end_ns=clip.end_ns,
                source_asset_sha256=str(metadata.get("source_asset_sha256") or ""),
                preprocessing_sha256=plan.preprocessing_sha256,
                frame_manifest_asset_id=manifest_asset.id,
                frame_count=clip.frame_count,
            )
            ctx.session.add(clip_row)
            ctx.session.flush()

            clip_task = _clip_task(ctx, run, clip.clip_id, episode.id)
            # MERGE waits for *every* clip task.  Without this edge the merge could run
            # while a clip is still in flight and publish a partial annotation set.
            ctx.session.add(
                TaskDependency(task_id=merge_task.id, depends_on_task_id=clip_task.id)
            )
            enqueue_task_wakeup(ctx.session, clip_task.id, clip_task.pool)
            created.append(
                {
                    "clip_id": clip.clip_id,
                    "start_ns": clip.start_ns,
                    "end_ns": clip.end_ns,
                    "frame_count": clip.frame_count,
                    "task_id": clip_task.id,
                }
            )

        return HandlerResult(
            data={
                "annotation_set_id": annotation_set.id,
                "canonical_version": source.canonical_version,
                "canonical_schema_version": source.schema_version,
                "episode_type": episode.episode_type,
                "metadata_key": source.metadata_key,
                "video_key": source.video_key,
                "duration_ns": duration_ns,
                "aligned_fps": aligned_fps,
                "clip_count": plan.clip_count,
                "preprocessing_sha256": plan.preprocessing_sha256,
                "clips": created,
            }
        )


@dataclass(frozen=True)
class KeyframeVideoSource:
    """The recorded video time base one AI run plans its clips on."""

    metadata: dict[str, Any]
    metadata_key: str
    video_key: str
    duration_ns: int
    fps: int
    schema_version: str
    canonical_version: int


def _resolve_video_source(ctx: HandlerContext, episode: Episode) -> KeyframeVideoSource:
    """Pick the canonical video for this Episode type, without inventing channels.

    ``robot_episode`` reads the canonical-episode/2.0 preview (the aligned 10 Hz grid).
    ``ego_video_episode`` never produces state/action (SPEC_V2.1 §4.3), so it reads
    the canonical-episode/1.0 video it actually has.  Both branches only need a video
    and its recorded timestamps; neither fabricates a robot trajectory.
    """
    canonical_version = int(episode.canonical_version or 1)
    is_ego = str(episode.episode_type) == EpisodeType.EGO_VIDEO_EPISODE.value
    keys_v2 = canonical_v2_keys(episode.id, canonical_version)
    keys_v1 = canonical_keys(episode.id, canonical_version)
    schema_version = str(getattr(episode, "canonical_schema_version", "1.0") or "1.0")

    candidates: list[tuple[str, str, str]] = []
    if is_ego:
        # Ego is v1 by contract; a v2 snapshot would still be honoured if present.
        if schema_version.startswith("2"):
            candidates.append(
                (keys_v2["metadata"], f"{keys_v2['prefix']}/preview/{PREVIEW_NAME}", CANONICAL_V2_SCHEMA_VERSION)
            )
        candidates.append((keys_v1["metadata"], keys_v1["video"], "canonical-episode/1.0"))
    else:
        if schema_version.startswith("2"):
            candidates.append(
                (keys_v2["metadata"], f"{keys_v2['prefix']}/preview/{PREVIEW_NAME}", CANONICAL_V2_SCHEMA_VERSION)
            )
        candidates.append((keys_v1["metadata"], keys_v1["video"], "canonical-episode/1.0"))

    for metadata_key, video_key, expected_schema in candidates:
        if not ctx.storage.exists(metadata_key):
            continue
        metadata = _parse_metadata(ctx.storage.get_bytes(metadata_key), metadata_key)
        actual_schema = str(metadata.get("schema_version") or "")
        if actual_schema != expected_schema:
            continue
        if not ctx.storage.exists(video_key):
            raise TerminalTaskError(
                ErrorCode.OBJECT_NOT_FOUND,
                "canonical video is required to extract keyframes",
                details={"video_key": video_key, "episode_id": episode.id},
            )
        metadata_type = str(metadata.get("episode_type") or "")
        if metadata_type and metadata_type != str(episode.episode_type):
            raise TerminalTaskError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "canonical metadata episode_type disagrees with the Episode row",
                details={"metadata": metadata_type, "episode": str(episode.episode_type)},
            )
        duration_ns = int(metadata.get("duration_ns") or 0)
        aligned = metadata.get("aligned_view") or {}
        fps = int(aligned.get("fps") or metadata.get("target_fps") or 10)
        return KeyframeVideoSource(
            metadata=metadata,
            metadata_key=metadata_key,
            video_key=video_key,
            duration_ns=duration_ns,
            fps=max(1, fps),
            schema_version=actual_schema,
            canonical_version=canonical_version,
        )
    raise TerminalTaskError(
        ErrorCode.INCOMPATIBLE_EPISODE,
        "AI pre-annotation requires a normalized canonical episode with a video",
        details={
            "episode_id": episode.id,
            "episode_type": str(episode.episode_type),
            "canonical_version": canonical_version,
            "expected": [item[2] for item in candidates],
        },
    )


def _parse_metadata(raw: bytes, key: str) -> dict[str, Any]:
    import json  # noqa: PLC0415

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "canonical metadata is not valid JSON",
            details={"metadata_key": key},
        ) from exc
    if not isinstance(payload, dict):
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "canonical metadata must be a JSON object",
            details={"metadata_key": key},
        )
    return payload


def _resolve_run(ctx: HandlerContext) -> tuple[PipelineRun, Episode]:
    task = ctx.task
    if not task.run_id:
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            f"{task.step} requires an episode-level PipelineRun",
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


def _merge_task(ctx: HandlerContext, run_id: str) -> Task | None:
    return ctx.session.scalars(
        select(Task).where(
            Task.run_id == run_id, Task.step == Step.MERGE_CANDIDATE_ANNOTATIONS.value
        )
    ).first()


def _clip_task(ctx: HandlerContext, run: PipelineRun, clip_id: str, episode_id: str) -> Task:
    """Create (or reuse) the per-clip VLM task, keyed by ``unit_key=clip_id``."""
    existing = ctx.session.scalars(
        select(Task).where(
            Task.run_id == run.id,
            Task.step == Step.VLM_PREANNOTATE_CLIP.value,
            Task.unit_key == clip_id,
        )
    ).first()
    if existing is not None:
        return existing
    task = Task(
        id=new_id(),
        run_id=run.id,
        step=Step.VLM_PREANNOTATE_CLIP.value,
        unit_key=clip_id,
        pool="vision",
        state=TaskState.PENDING.value,
        attempt=0,
        max_attempts=max(1, int(ctx.settings.vlm_max_attempts)),
        input_digest=input_digest(
            Step.VLM_PREANNOTATE_CLIP, [clip_id], {"episode_id": episode_id, "prompt": ctx.settings.vlm_prompt_version}
        ),
        output_json={},
    )
    ctx.session.add(task)
    ctx.session.flush()
    return task


HANDLER = ExtractKeyframesHandler()

__all__ = [
    "HANDLER",
    "ExtractKeyframesHandler",
    "KeyframeVideoSource",
    "ai_prefix",
    "frame_key",
    "manifest_key",
    "output_key",
]
