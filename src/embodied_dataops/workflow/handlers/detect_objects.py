"""DETECT_OBJECTS handler (SPEC 6.1, 8.1).

Loads the canonical episode produced by ``NORMALIZE_EPISODE`` in the same
PipelineRun, samples the 10 Hz video at 1 fps (``frame_index = 0, 10, 20, ...``),
runs the detector selected by ``settings.detector`` and writes the immutable
``annotation/1.0`` artifact to
``annotations/{episode_id}/{annotation_set_id}/annotation_candidates.json``.

The handler also creates the ``AnnotationSet`` row (``PENDING_REVIEW``) and owns
its ``artifact_asset_id``.  It never mutates ``tasks.state`` and never commits:
the workflow engine owns lease fencing, retries and downstream wakeups.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from embodied_dataops import TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.canonical.contract import (
    CanonicalMetadata,
    annotation_key,
    canonical_keys,
    canonical_prefix,
)
from embodied_dataops.db.models import AnnotationSet, Episode, PipelineRun, Task
from embodied_dataops.domain.enums import AnnotationStatus, AssetOrigin, Step
from embodied_dataops.errors import ApiError, CancelledError, ErrorCode, TerminalTaskError
from embodied_dataops.ids import new_id
from embodied_dataops.observability.metrics import ANNOTATION_FRAMES
from embodied_dataops.processing import detection, preview
from embodied_dataops.workflow.handlers.base import (
    HandlerContext,
    HandlerResult,
    TaskHandler,
    register_json,
)


@dataclass
class CanonicalInputs:
    """Resolved canonical episode inputs for one Episode pipeline task."""

    episode: Episode
    run: PipelineRun
    #: V1 hands over :class:`CanonicalMetadata`; 2.0 hands over :class:`MetadataView`.
    #: Both expose the handful of fields the preview/detection handlers actually read.
    metadata: Any
    canonical_version: int
    metadata_key: str
    video_key: str
    samples_key: str
    thumbnail_key: str
    schema_version: str


@dataclass
class MetadataView:
    """The canonical-episode/2.0 fields BUILD_PREVIEW / DETECT_OBJECTS read.

    V2 has no thumbnail object and no ``frames/front.mp4``; the preview video is the
    only decodable artifact (``preview/{camera}_10hz_640x480.mp4``) and the aligned
    grid carries the sample count.  Mapping that into the V1 field names keeps the two
    handlers version-agnostic instead of duplicating their bodies.
    """

    schema_version: str
    episode_id: str
    sample_count: int
    camera_name: str
    clock_domain: str
    width: int
    height: int


def _first_key(output: dict[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = output.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_canonical_inputs(session: Any, ctx: HandlerContext) -> CanonicalInputs:
    """Resolve the canonical metadata/video/samples keys for the current run.

    ``NORMALIZE_EPISODE`` output_json key names are read tolerantly; the
    deterministic canonical layout (``canonical-episode/1.0``) is the fallback,
    so this stays compatible if the normalizer only records a subset of keys.
    """
    task = ctx.task
    if not task.run_id:
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            f"{task.step} requires an episode-level PipelineRun",
            details={"task_id": task.id, "step": task.step},
        )
    run = session.get(PipelineRun, task.run_id)
    if run is None:
        raise TerminalTaskError(
            ErrorCode.NOT_FOUND,
            "PipelineRun referenced by the task is missing",
            details={"run_id": task.run_id},
        )
    episode = session.get(Episode, run.episode_id)
    if episode is None:
        raise TerminalTaskError(
            ErrorCode.NOT_FOUND,
            "Episode referenced by the PipelineRun is missing",
            details={"episode_id": run.episode_id},
        )
    normalize_task = session.scalars(
        select(Task).where(
            Task.run_id == run.id,
            Task.step == Step.NORMALIZE_EPISODE.value,
        )
    ).first()
    output: dict[str, Any] = dict(normalize_task.output_json or {}) if normalize_task else {}
    version_value = output.get("canonical_version") or episode.canonical_version or 1
    try:
        canonical_version = int(version_value)
    except (TypeError, ValueError):
        canonical_version = int(episode.canonical_version or 1)
    if str(getattr(episode, "canonical_schema_version", "") or "").startswith("2"):
        return _resolve_v2_inputs(
            ctx,
            episode=episode,
            run=run,
            output=output,
            canonical_version=canonical_version,
        )
    keys = canonical_keys(episode.id, canonical_version)
    prefix = canonical_prefix(episode.id, canonical_version)

    metadata_key = (
        _first_key(output, ("metadata_key", "metadata_object_key", "metadata_path", "metadata"))
        or keys["metadata"]
    )
    try:
        raw_metadata = ctx.storage.get_bytes(metadata_key)
    except ApiError as exc:
        raise TerminalTaskError(
            ErrorCode.OBJECT_NOT_FOUND,
            "Canonical metadata object is missing for this run",
            details={"metadata_key": metadata_key, "task_id": task.id},
        ) from exc
    try:
        metadata = CanonicalMetadata.model_validate_json(raw_metadata)
    except Exception as exc:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical metadata does not match canonical-episode/1.0",
            details={"metadata_key": metadata_key, "error_type": type(exc).__name__},
        ) from exc
    if metadata.episode_id != episode.id:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical metadata belongs to a different Episode",
            details={"metadata_episode_id": metadata.episode_id, "episode_id": episode.id},
        )
    if not metadata.normalization_valid:
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            "Canonical episode is not normalization-valid; detection cannot run",
            details={"episode_id": episode.id, "task_id": task.id},
        )
    if not metadata.video_available:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical episode has no decodable video; detection cannot run",
            details={"episode_id": episode.id},
        )
    video_key = (
        _first_key(output, ("video_key", "video_object_key", "video")) or keys["video"]
    )
    if metadata.video_path:
        video_key = f"{prefix}/{str(metadata.video_path).lstrip('/')}"
    samples_key = (
        _first_key(output, ("samples_key", "samples_object_key", "samples_parquet"))
        or keys["samples"]
    )
    if metadata.samples_parquet:
        samples_key = f"{prefix}/{str(metadata.samples_parquet).lstrip('/')}"
    if not ctx.storage.exists(video_key):
        raise TerminalTaskError(
            ErrorCode.OBJECT_NOT_FOUND,
            "Canonical video object is missing for this run",
            details={"video_key": video_key, "episode_id": episode.id},
        )
    return CanonicalInputs(
        episode=episode,
        run=run,
        metadata=metadata,
        canonical_version=canonical_version,
        metadata_key=metadata_key,
        video_key=video_key,
        samples_key=samples_key,
        thumbnail_key=keys["thumbnail"],
        schema_version=str(metadata.schema_version),
    )


def _resolve_v2_inputs(
    ctx: HandlerContext,
    *,
    episode: Episode,
    run: PipelineRun,
    output: dict[str, Any],
    canonical_version: int,
) -> CanonicalInputs:
    """Resolve ``canonical-episode/2.0`` inputs (preview video + aligned grid).

    The v1 layout (``canonical/{id}/v{n}/metadata.json`` + ``frames/front.mp4``) does
    not exist for a 2.0 Episode, so resolving it there failed with
    ``Canonical metadata object is missing for this run`` and left
    ``GET /episodes/{id}/preview`` at 409.
    """
    from embodied_dataops.canonical.v2.contract import (
        CanonicalMetadataV2,
        canonical_v2_keys,
    )

    keys = canonical_v2_keys(episode.id, canonical_version)
    metadata_key = (
        _first_key(output, ("metadata_key", "metadata_object_key", "metadata_path", "metadata"))
        or keys["metadata"]
    )
    try:
        raw_metadata = ctx.storage.get_bytes(metadata_key)
    except ApiError as exc:
        raise TerminalTaskError(
            ErrorCode.OBJECT_NOT_FOUND,
            "Canonical metadata object is missing for this run",
            details={"metadata_key": metadata_key, "schema_version": "canonical-episode/2.0"},
        ) from exc
    try:
        metadata = CanonicalMetadataV2.model_validate_json(raw_metadata)
    except Exception as exc:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical metadata does not match canonical-episode/2.0",
            details={"metadata_key": metadata_key, "error_type": type(exc).__name__},
        ) from exc
    if metadata.episode_id != episode.id:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical metadata belongs to a different Episode",
            details={"metadata_episode_id": metadata.episode_id, "episode_id": episode.id},
        )

    camera_names = [modality.name for modality in metadata.modalities if modality.kind == "camera"]
    if not camera_names:
        raise TerminalTaskError(
            ErrorCode.INVALID_INPUT,
            "canonical-episode/2.0 declares no camera modality; preview cannot be built",
            details={"episode_id": episode.id, "metadata_key": metadata_key},
        )
    camera_name = "front" if "front" in camera_names else camera_names[0]
    video_key = (
        _first_key(output, ("video_key", "video_object_key", "video"))
        or keys["preview"].format(camera=camera_name)
    )
    if not ctx.storage.exists(video_key):
        raise TerminalTaskError(
            ErrorCode.OBJECT_NOT_FOUND,
            "Canonical preview video object is missing for this run",
            details={"video_key": video_key, "episode_id": episode.id},
        )
    aligned = metadata.aligned_view
    view = MetadataView(
        schema_version=str(metadata.schema_version),
        episode_id=metadata.episode_id,
        sample_count=int(getattr(aligned, "sample_count", 0) or 0),
        camera_name=camera_name,
        clock_domain=str(metadata.clock_domains.get("camera") or ""),
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
    )
    return CanonicalInputs(
        episode=episode,
        run=run,
        metadata=view,
        canonical_version=canonical_version,
        metadata_key=metadata_key,
        video_key=video_key,
        samples_key=keys["aligned_samples"],
        thumbnail_key=keys["thumbnail"],
        schema_version=view.schema_version,
    )


class DetectObjectsHandler(TaskHandler):
    """Run AI pre-annotation for one canonical Episode (SPEC 6.1, 8.1)."""

    step = Step.DETECT_OBJECTS

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        if ctx.is_cancelled():
            raise CancelledError()
        inputs = resolve_canonical_inputs(ctx.session, ctx)

        video_path = ctx.workspace / "canonical_front.mp4"
        ctx.storage.download_to(inputs.video_key, video_path)

        detector = detection.build_detector(ctx.settings)
        model_info = detector.prepare()

        def samples():
            """Cancellation-aware 1 fps RGB sample stream (10 Hz -> frame_index 0,10,20...)."""
            for frame in preview.iter_frames(
                video_path,
                stride=detection.SAMPLE_STRIDE,
                require_640x480=True,
            ):
                if ctx.is_cancelled():
                    raise CancelledError()
                yield frame.frame_index, frame.rgb

        sampled_frames = detector.detect(samples())
        if ctx.is_cancelled():
            raise CancelledError()

        annotation_set = AnnotationSet(
            id=new_id(),
            episode_id=inputs.episode.id,
            run_id=inputs.run.id,
            model_id=detector.name,
            model_sha256=model_info.weights_sha256,
            status=AnnotationStatus.PENDING_REVIEW.value,
            summary_json={},
        )
        ctx.session.add(annotation_set)
        ctx.session.flush()

        document = detection.build_annotation_document(
            episode_id=inputs.episode.id,
            model=model_info,
            frames=sampled_frames,
            tool=detector.tool,
            is_synthetic=detector.is_synthetic,
            extra={"canonical_version": inputs.canonical_version},
        )
        errors = detection.validate_annotation_document(document)
        if errors:
            raise TerminalTaskError(
                ErrorCode.INTERNAL_ERROR,
                "Generated annotation document violates annotation/1.0",
                details={"errors": errors[:20]},
            )

        artifact = register_json(
            ctx,
            key=annotation_key(inputs.episode.id, annotation_set.id),
            payload=document,
            media_type="application/json",
            origin=AssetOrigin.DERIVED,
            logical_name="annotation_candidates",
            meta={
                "schema_version": document["schema_version"],
                "tool": detector.tool,
                "model_id": detector.name,
                "is_synthetic": detector.is_synthetic,
                "frame_count": document["stats"]["frames"],
                "detection_count": document["stats"]["detections"],
            },
        )
        annotation_set.artifact_asset_id = artifact.id
        annotation_set.summary_json = {
            **document["stats"],
            "model": model_info.to_json(),
            "tool": detector.tool,
            "is_synthetic": detector.is_synthetic,
            "sample_fps": 1,
            "source_fps": TARGET_FPS,
        }
        ANNOTATION_FRAMES.labels(model=detector.name).inc(len(document["frames"]))

        expected_sample_count = int(inputs.metadata.sample_count or 0)
        expected_sampled_frames = (
            len(detection.sampled_frame_indices(expected_sample_count))
            if expected_sample_count
            else None
        )
        return HandlerResult(
            data={
                "annotation_set_id": annotation_set.id,
                "artifact_key": annotation_key(inputs.episode.id, annotation_set.id),
                "artifact_asset_id": artifact.id,
                "artifact_sha256": artifact.sha256,
                "canonical_version": inputs.canonical_version,
                "video_key": inputs.video_key,
                "metadata_key": inputs.metadata_key,
                "sampled_frames": len(document["frames"]),
                "expected_sampled_frames": expected_sampled_frames,
                "detections": document["stats"]["detections"],
                "empty_frames": document["stats"]["empty_frames"],
                "low_confidence_excluded": document["stats"]["low_confidence_excluded"],
                "model": model_info.to_json(),
                "tool": detector.tool,
                "is_synthetic": detector.is_synthetic,
                "sample_fps": 1,
            },
            output_asset_id=artifact.id,
        )


#: Registry integration point used by the Lead-owned handler registry.
HANDLER = DetectObjectsHandler()
