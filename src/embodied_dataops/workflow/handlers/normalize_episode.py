"""NORMALIZE_EPISODE handler (SPEC 5.2, 5.3, 5.4, 5.5, 6.1).

Turns one imported Episode into the internal ``canonical-episode/1.0`` contract:

``canonical/{episode_id}/v1/``
  ``metadata.json``            required fields + diagnostics/alignment extras
  ``frames/front.mp4``         10 fps H.264 yuv420p, 640x480 letterbox, no audio
  ``samples.parquet``          10 Hz rows, robot or ego schema
  ``frames/thumbnail.jpg``     preview thumbnail (derived asset)

Contract rules enforced here:

* objects are written to their **final deterministic keys** and registered as
  immutable ``assets`` rows (no staging prefix, no overwrite: an identical
  re-run reuses the Asset, a different payload is a ``CONFLICT``);
* a *structural* incompatibility (non-monotonic timestamps, 5D action, NaN,
  missing 10 Hz grid point, over-duration) still produces minimal
  ``metadata.json`` with ``normalization_valid=false`` and structured
  diagnostics, but **no publishable samples.parquet / video**: nothing is
  truncated, zero-filled or silently repaired (SPEC 6.1);
* an Ego video stays ``ego_video_episode`` with ``state_dim=0`` /
  ``action_dim=0`` / ``joint_names=[]``: robot action is never synthesised;
* the handler never commits/rolls back and never touches ``tasks.state`` --
  the workflow engine owns lease fencing and commits these rows atomically;
* ``output_json`` carries everything QUALITY_CHECK needs (diagnostics,
  alignment/sync metrics, canonical keys, dimensions) so downstream steps read
  upstream ``tasks.output_json`` instead of re-parsing the source.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops import PIPELINE_VERSION, SAMPLE_INTERVAL_NS, TARGET_FPS
from embodied_dataops.canonical.contract import (
    ACTION_DIM,
    ROBOT_JOINT_NAMES,
    ROBOT_PROFILE,
    STATE_DIM,
    CanonicalMetadata,
    SampleRow,
    canonical_digest,
    canonical_json_bytes,
    canonical_keys,
    write_samples_parquet,
)
from embodied_dataops.canonical.video import apply_letterbox, compute_letterbox, summarize_transforms
from embodied_dataops.db.models import Asset, Episode, Modality, PipelineRun
from embodied_dataops.domain.enums import (
    AssetOrigin,
    ModalityKind,
    SourceType,
    Step,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.imports.base import EpisodeSource, get_parser
from embodied_dataops.imports.common import (
    build_import_context,
    load_asset,
    load_source_import,
    materialize_source,
    merge_diagnostics,
    utcnow,
)
from embodied_dataops.storage.base import sha256_file
from embodied_dataops.workflow.handlers.base import (
    HandlerContext,
    HandlerResult,
    TaskHandler,
    register_bytes,
    register_file,
)

# ---------------------------------------------------------------------------
# intermediate draft shared by the three source types
# ---------------------------------------------------------------------------


@dataclass
class CanonicalDraft:
    """Everything the publisher needs, produced by one source-type adapter."""

    episode_type: str
    rows: list[SampleRow] | None = None
    frames: Iterator[np.ndarray] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    alignment: dict[str, Any] = field(default_factory=dict)
    clock_domains: dict[str, str] = field(default_factory=dict)
    joint_names: list[str] = field(default_factory=list)
    state_dim: int = 0
    action_dim: int = 0
    duration_ns: int = 0
    decoded_frame_count: int = 0
    metadata_extra: dict[str, Any] = field(default_factory=dict)

    @property
    def normalization_valid(self) -> bool:
        return self.rows is not None

    @property
    def sample_count(self) -> int:
        return len(self.rows) if self.rows is not None else 0

    @property
    def primary_clock_domain(self) -> str:
        if self.clock_domains:
            return next(iter(self.clock_domains.values()))
        return "unknown"


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------


class NormalizeEpisodeHandler(TaskHandler):
    step = Step.NORMALIZE_EPISODE

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        session: Session = ctx.session
        run = session.get(PipelineRun, ctx.task.run_id) if ctx.task.run_id else None
        if run is None:
            raise ApiError(ErrorCode.INVALID_INPUT, "NORMALIZE_EPISODE is not linked to a pipeline run")
        episode = session.get(Episode, run.episode_id)
        if episode is None:
            raise ApiError(ErrorCode.NOT_FOUND, "episode of this pipeline run does not exist")
        source_import = load_source_import(session, episode.source_import_id)
        # V2 dispatch (ADR-0007): explicit profile selection / canonical 2.0 uses the
        # v2 normalizer; the V1 path below is byte-identical for V1 imports.
        try:
            from embodied_dataops.workflow.handlers.normalize_episode_v2 import (
                HANDLER as NORMALIZE_V2,
            )
            from embodied_dataops.workflow.handlers.normalize_episode_v2 import applies_to
        except ImportError:
            NORMALIZE_V2 = None
        if NORMALIZE_V2 is not None and applies_to(source_import, episode):
            return NORMALIZE_V2.handle(ctx)
        asset = load_asset(session, source_import)
        source_path = materialize_source(ctx.storage, source_import, asset, ctx.workspace)
        context = build_import_context(source_import, source_path, ctx.workspace)
        if not context.source_sha256:
            context.source_sha256 = asset.sha256
        parser = get_parser(source_import.source_type)

        if source_import.source_type == SourceType.LEROBOT_V3.value:
            aligned = parser.read_aligned_episode(context, episode.source_episode_index)
            draft = CanonicalDraft(
                episode_type=aligned_episode_type(aligned),
                rows=aligned.rows,
                frames=parser.iter_aligned_frames(context, aligned) if aligned.rows is not None else None,
                diagnostics=aligned.diagnostics,
                alignment=aligned.alignment,
                clock_domains={
                    "camera": "lerobot_episode_relative",
                    "state": "lerobot_episode_relative",
                    "action": "lerobot_episode_relative",
                },
                joint_names=(
                    aligned.joint_names
                    if aligned.state_dim == STATE_DIM and aligned.action_dim == ACTION_DIM
                    else []
                ),
                state_dim=aligned.state_dim,
                action_dim=aligned.action_dim,
                duration_ns=aligned.duration_ns,
                decoded_frame_count=aligned.source_row_count,
                metadata_extra={
                    "robot_profile": aligned.robot_profile,
                    "state_key": aligned.state_key,
                    "action_key": aligned.action_key,
                    "camera_key": aligned.camera_key,
                    "source_fps": aligned.source_fps,
                    "source_episode_index": aligned.episode_index,
                    "source_row_count": aligned.source_row_count,
                    "source_duration_ns": aligned.source_duration_ns,
                    "semantics_verified": aligned.semantics_verified,
                    "semantics_evidence": aligned.semantics_evidence,
                },
            )
        elif source_import.source_type == SourceType.EGO_MP4.value:
            data = parser.read_episode(context, episode.source_episode_index)
            selection = data.selection
            rows = (
                [
                    SampleRow(
                        sample_index=index,
                        t_ns=index * SAMPLE_INTERVAL_NS,
                        frame_index=index,
                    )
                    for index in range(selection.sample_count)
                ]
                if data.normalization_valid
                else None
            )
            draft = CanonicalDraft(
                episode_type="ego_video_episode",
                rows=rows,
                frames=data.iter_frames() if rows is not None else None,
                diagnostics=data.diagnostics,
                alignment=data.alignment(),
                clock_domains={"camera": "ego_video_pts_relative"},
                joint_names=[],
                state_dim=0,
                action_dim=0,
                duration_ns=data.duration_ns,
                decoded_frame_count=data.decoded_frame_count,
                metadata_extra={
                    "camera_key": "front",
                    "source_fps": data.probe_result.fps,
                    "source_frame_count": data.timeline.frame_count,
                    "has_audio": data.probe_result.has_audio,
                    "timeline_backend": data.timeline.backend,
                    "timestamps_derived_from_cfr": data.timeline.derived_from_cfr,
                },
            )
        elif source_import.source_type == SourceType.MCAP_ROS2_PROFILE_V1.value:
            parsed = parser.parse(context)
            episode_source = _find_episode_source(parsed.episodes, episode.source_episode_index)
            draft = _draft_from_mcap(episode_source)
        else:  # pragma: no cover - enum is closed by the DB constraint
            raise ApiError(
                ErrorCode.UNSUPPORTED_SOURCE_TYPE,
                f"unsupported source_type {source_import.source_type}",
            )

        # The parser reported diagnostics but a structurally valid draft only
        # exists when there are no structural problems.
        draft.diagnostics = merge_diagnostics(draft.diagnostics)
        if any(item.get("structural") for item in draft.diagnostics):
            draft.rows = None
            draft.frames = None

        return self._publish(ctx, episode=episode, source_import=source_import, asset=asset, draft=draft)

    # -- canonical publication -------------------------------------------------

    def _publish(
        self,
        ctx: HandlerContext,
        *,
        episode: Episode,
        source_import: Any,
        asset: Asset,
        draft: CanonicalDraft,
    ) -> HandlerResult:
        session = ctx.session
        version = max(1, int(episode.canonical_version or 1))
        keys = canonical_keys(episode.id, version)
        workdir = Path(ctx.workspace) / "canonical"
        workdir.mkdir(parents=True, exist_ok=True)
        valid = draft.normalization_valid and draft.rows is not None

        video_asset: Asset | None = None
        samples_asset: Asset | None = None
        thumbnail_asset: Asset | None = None
        transforms: list[dict[str, Any]] = []
        if valid and draft.frames is not None:
            video_path = workdir / "front.mp4"
            transforms, written = _encode_letterbox_video(draft.frames, video_path)
            if written != len(draft.rows or []):
                raise ApiError(
                    ErrorCode.INTERNAL_ERROR,
                    "canonical video frame count does not match canonical samples",
                    details={"written": written, "samples": len(draft.rows or [])},
                )
            video_asset = _register_file_idempotent(
                ctx,
                key=keys["video"],
                path=video_path,
                media_type="video/mp4",
                origin=AssetOrigin.CANONICAL,
                logical_name="canonical:frames/front.mp4",
                meta={"fps": TARGET_FPS, "width": 640, "height": 480},
            )
            try:
                thumb_path = workdir / "thumbnail.jpg"
                from embodied_dataops.processing.ffmpeg import thumbnail

                thumbnail(video_path, thumb_path)
                thumbnail_asset = _register_file_idempotent(
                    ctx,
                    key=keys["thumbnail"],
                    path=thumb_path,
                    media_type="image/jpeg",
                    origin=AssetOrigin.DERIVED,
                    logical_name="canonical:frames/thumbnail.jpg",
                )
            except ApiError:  # preview thumbnail is non-essential evidence
                thumbnail_asset = None

        if valid and draft.rows:
            samples_path = workdir / "samples.parquet"
            write_samples_parquet(samples_path, draft.rows, draft.episode_type)
            samples_asset = _register_file_idempotent(
                ctx,
                key=keys["samples"],
                path=samples_path,
                media_type="application/vnd.apache.parquet",
                origin=AssetOrigin.CANONICAL,
                logical_name="canonical:samples.parquet",
                meta={
                    "sample_count": len(draft.rows),
                    "schema_version": "canonical-episode/1.0",
                },
            )

        metadata = self._build_metadata(
            episode=episode,
            source_import=source_import,
            asset=asset,
            draft=draft,
            valid=valid,
            samples_asset=samples_asset,
            video_asset=video_asset,
            transforms=transforms,
        )
        metadata_payload = metadata.model_dump(mode="json")
        metadata_payload.update(
            {
                "diagnostics": draft.diagnostics,
                "normalization_diagnostics": draft.diagnostics,
                "sync_diagnostics": draft.alignment,
                "alignment": draft.alignment,
                "clock_domains": draft.clock_domains,
                "decoded_frame_count": draft.decoded_frame_count,
                "source": import_context_payload_from(episode, source_import, asset),
                "provenance": draft.metadata_extra,
            }
        )
        metadata_bytes = canonical_json_bytes(metadata_payload)
        metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
        metadata_asset = _register_bytes_idempotent(
            ctx,
            key=keys["metadata"],
            data=metadata_bytes,
            media_type="application/json",
            origin=AssetOrigin.CANONICAL,
            logical_name="canonical:metadata.json",
            meta={"normalization_valid": valid, "sha256": metadata_sha256},
        )
        canonical_sha256 = canonical_digest(
            metadata,
            samples_asset.sha256 if samples_asset is not None else "",
            video_asset.sha256 if video_asset is not None else None,
        )

        episode.canonical_sha256 = canonical_sha256
        episode.duration_ns = int(draft.duration_ns)
        episode.sample_count = len(draft.rows) if draft.rows else 0
        episode.canonical_version = version
        episode.updated_at = utcnow()
        self._sync_modalities(
            session,
            episode=episode,
            draft=draft,
            valid=valid,
            video_available=video_asset is not None,
        )
        session.flush()

        canonical_block = {
            "prefix": keys["prefix"],
            "metadata_key": keys["metadata"],
            "metadata_sha256": metadata_asset.sha256,
            "samples_key": keys["samples"] if samples_asset is not None else None,
            "samples_sha256": samples_asset.sha256 if samples_asset is not None else None,
            "video_key": keys["video"] if video_asset is not None else None,
            "video_sha256": video_asset.sha256 if video_asset is not None else None,
            "thumbnail_key": keys["thumbnail"] if thumbnail_asset is not None else None,
            "thumbnail_sha256": thumbnail_asset.sha256 if thumbnail_asset is not None else None,
            "canonical_sha256": canonical_sha256,
            "canonical_version": version,
        }
        data: dict[str, Any] = {
            "step": self.step.value,
            "episode_id": episode.id,
            "episode_type": draft.episode_type,
            "source_type": source_import.source_type,
            "normalization_valid": valid,
            "sample_count": len(draft.rows) if draft.rows else 0,
            "duration_ns": int(draft.duration_ns),
            "state_dim": int(draft.state_dim),
            "action_dim": int(draft.action_dim),
            "joint_names": list(draft.joint_names),
            "clock_domains": draft.clock_domains,
            "diagnostics": draft.diagnostics,
            "normalization_diagnostics": draft.diagnostics,
            "alignment": draft.alignment,
            "sync_metrics": {
                "valid_sample_count": int(draft.alignment.get("valid_sample_count", 0) or 0),
                "missing_sample_count": int(draft.alignment.get("missing_sample_count", 0) or 0),
                "max_abs_sync_error_ms": draft.alignment.get("max_abs_sync_error_ms", 0.0),
                "p95_abs_sync_error_ms": draft.alignment.get("p95_abs_sync_error_ms", 0.0),
            },
            "frame_transforms_summary": summarize_transforms(transforms),
            "decoded_frame_count": int(draft.decoded_frame_count),
            "canonical": canonical_block,
            "metadata_key": canonical_block["metadata_key"],
            "metadata_sha256": canonical_block["metadata_sha256"],
            "samples_key": canonical_block["samples_key"],
            "video_key": canonical_block["video_key"],
            "thumbnail_key": canonical_block["thumbnail_key"],
            "canonical_sha256": canonical_sha256,
            "provenance": draft.metadata_extra,
            "normalized_at": utcnow().isoformat(),
        }
        return HandlerResult(data=data, output_asset_id=metadata_asset.id)

    # -- metadata --------------------------------------------------------------

    def _build_metadata(
        self,
        *,
        episode: Episode,
        source_import: Any,
        asset: Asset,
        draft: CanonicalDraft,
        valid: bool,
        samples_asset: Asset | None,
        video_asset: Asset | None,
        transforms: list[dict[str, Any]],
    ) -> CanonicalMetadata:
        created_at = episode.created_at
        if created_at is not None:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            created_at_text = created_at.astimezone(UTC).isoformat()
        else:
            created_at_text = ""
        source_sha = asset.sha256 or source_import.sha256 or ""
        if len(source_sha) != 64:
            source_sha = hashlib.sha256(source_sha.encode()).hexdigest()
        return CanonicalMetadata(
            episode_id=episode.id,
            episode_type=draft.episode_type,
            data_origin=source_import.data_origin,
            source_type=source_import.source_type,
            source_asset_sha256=source_sha,
            source_episode_index=episode.source_episode_index,
            duration_ns=int(draft.duration_ns),
            target_fps=TARGET_FPS,
            camera_name="front",
            clock_domain=draft.primary_clock_domain,
            joint_names=list(draft.joint_names),
            state_dim=int(draft.state_dim),
            action_dim=int(draft.action_dim),
            samples_parquet="samples.parquet" if samples_asset is not None else "",
            video_path="frames/front.mp4" if video_asset is not None else "",
            pipeline_version=PIPELINE_VERSION,
            task_label=episode.task_label or "",
            robot_profile=str(draft.metadata_extra.get("robot_profile") or ""),
            source_filename=source_import.filename or "",
            source_license=source_import.license or "",
            source_uri=source_import.source_uri,
            source_revision=source_import.source_revision,
            mapping=dict(source_import.mapping_json or {}),
            normalization_valid=valid,
            sample_count=len(draft.rows) if draft.rows else 0,
            video_available=video_asset is not None,
            frame_transforms=transforms,
            created_at=created_at_text,
        )

    # -- modalities ------------------------------------------------------------

    def _sync_modalities(
        self,
        session: Session,
        *,
        episode: Episode,
        draft: CanonicalDraft,
        valid: bool,
        video_available: bool,
    ) -> None:
        sample_count = len(draft.rows) if draft.rows else 0
        max_t = (sample_count - 1) * SAMPLE_INTERVAL_NS if sample_count else None
        camera_clock = draft.clock_domains.get("camera", draft.primary_clock_domain)
        if video_available:
            _upsert_modality(
                session,
                episode_id=episode.id,
                name="front",
                kind=ModalityKind.CAMERA.value,
                dtype="uint8",
                shape=[480, 640, 3],
                clock_domain=camera_clock,
                sample_count=sample_count,
                min_t_ns=0 if sample_count else None,
                max_t_ns=max_t,
                source_key=str(draft.metadata_extra.get("camera_key") or "front"),
            )
        if draft.episode_type == "robot_episode":
            for name, kind, dim in (
                ("state", ModalityKind.STATE.value, draft.state_dim),
                ("action", ModalityKind.ACTION.value, draft.action_dim),
            ):
                _upsert_modality(
                    session,
                    episode_id=episode.id,
                    name=name,
                    kind=kind,
                    dtype="float32",
                    shape=[int(dim)],
                    clock_domain=draft.clock_domains.get(name, draft.primary_clock_domain),
                    sample_count=sample_count,
                    min_t_ns=0 if sample_count else None,
                    max_t_ns=max_t,
                    source_key=str(
                        draft.metadata_extra.get(f"{name}_key")
                        or ("observation.state" if name == "state" else "action")
                    ),
                )
        # Structural failure with no canonical video/samples still leaves the
        # pre-existing modality rows untouched; they describe the last good
        # normalization for the episode and the QC report carries the verdict.


# ---------------------------------------------------------------------------
# draft builders
# ---------------------------------------------------------------------------


def aligned_episode_type(aligned: Any) -> str:
    return "robot_episode"


def _find_episode_source(episodes: list[EpisodeSource], episode_index: int) -> EpisodeSource:
    for item in episodes:
        if item.source_episode_index == episode_index:
            return item
    raise ApiError(
        ErrorCode.EPISODE_ENUMERATION_FAILED,
        "the MCAP adapter did not enumerate the requested episode",
        details={"episode_index": episode_index},
    )


def _draft_from_mcap(source: EpisodeSource) -> CanonicalDraft:
    """MCAP payload contract supplied by the ROS 2 adapter (see task-3)."""
    payload: dict[str, Any] = dict(source.payload or {})
    streams = payload.get("streams") or {}
    state_stream = streams.get("state") or {}
    action_stream = streams.get("action") or {}
    camera_stream = streams.get("camera") or {}
    state_values = state_stream.get("values") or []
    action_values = action_stream.get("values") or []
    state_dim = len(state_values[0]) if state_values else 0
    action_dim = len(action_values[0]) if action_values else 0
    joint_names = list(state_stream.get("joint_names") or [])
    samples = payload.get("samples") or []
    normalization_valid = bool(payload.get("normalization_valid", source.normalization_valid))
    diagnostics = merge_diagnostics(
        source.diagnostics,
        payload.get("diagnostics") or [],
    )
    if any(item.get("structural") for item in diagnostics):
        normalization_valid = False
    rows: list[SampleRow] | None = None
    frames: Iterator[np.ndarray] | None = None
    if normalization_valid and samples:
        rows = [
            SampleRow(
                sample_index=int(item["sample_index"]),
                t_ns=int(item["t_ns"]),
                frame_index=int(item["frame_index"]),
                state=[float(value) for value in item.get("state") or []],
                action=[float(value) for value in item.get("action") or []],
                state_source_t_ns=int(item.get("state_source_t_ns", 0)),
                action_source_t_ns=int(item.get("action_source_t_ns", 0)),
            )
            for item in samples
        ]
        frames = _iter_mcap_frames(samples, camera_stream)
    alignment = dict(payload.get("alignment") or {})
    alignment.setdefault("alignment_unit", "mcap_log_time_grid")
    alignment.setdefault("grid_points", len(samples))
    alignment.setdefault("valid_sample_count", len(samples))
    alignment.setdefault("missing_sample_count", 0)
    clock_domains = {
        "camera": str(camera_stream.get("clock_domain") or "mcap_log_time"),
        "state": str(state_stream.get("clock_domain") or "mcap_log_time"),
        "action": str(action_stream.get("clock_domain") or "mcap_log_time"),
    }
    duration_ns = int(payload.get("duration_ns") or (len(rows) * SAMPLE_INTERVAL_NS if rows else 0))
    if joint_names and joint_names != ROBOT_JOINT_NAMES:
        # Ordering/dimension problems are quality findings; keep the observed
        # names in provenance and let QUALITY_CHECK report DIMENSION_MISMATCH.
        pass
    return CanonicalDraft(
        episode_type="robot_episode",
        rows=rows if normalization_valid else None,
        frames=frames if normalization_valid else None,
        diagnostics=diagnostics,
        alignment=alignment,
        clock_domains=clock_domains,
        joint_names=joint_names if state_dim == STATE_DIM else [],
        state_dim=state_dim,
        action_dim=action_dim,
        duration_ns=duration_ns,
        decoded_frame_count=len(samples),
        metadata_extra={
            "robot_profile": ROBOT_PROFILE,
            "state_key": "/joint_states",
            "action_key": "/policy/action",
            "camera_key": "/camera/front/image_raw",
            "joint_names_observed": joint_names,
            "topics": {
                "camera": camera_stream.get("topic"),
                "state": state_stream.get("topic"),
                "action": action_stream.get("topic"),
            },
        },
    )


def _mcap_rgb(frame_meta: dict[str, Any]) -> np.ndarray:
    width = int(frame_meta.get("width") or 0)
    height = int(frame_meta.get("height") or 0)
    step = int(frame_meta.get("step") or width * 3)
    encoding = str(frame_meta.get("encoding") or "rgb8")
    data = frame_meta.get("data")
    if isinstance(data, str):
        data = data.encode("latin-1")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ApiError(ErrorCode.INTERNAL_ERROR, "MCAP camera frame payload is not bytes")
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    if width <= 0 or height <= 0 or raw.size < height * step:
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "MCAP camera frame does not match its declared geometry",
            details={"width": width, "height": height, "step": step, "bytes": int(raw.size)},
        )
    rows = raw[: height * step].reshape(height, step)
    if encoding not in {"rgb8", "bgr8"}:
        raise ApiError(
            ErrorCode.UNSUPPORTED_TOPIC_PROFILE,
            "MCAP camera encoding must be rgb8 or bgr8 for the v1 profile",
            details={"encoding": encoding},
        )
    pixels = rows[:, : width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        pixels = pixels[:, :, ::-1]
    return np.ascontiguousarray(pixels)


def _iter_mcap_frames(
    samples: list[dict[str, Any]], camera_stream: dict[str, Any]
) -> Iterator[np.ndarray]:
    frames = camera_stream.get("frames") or []
    for sample in samples:
        index = int(sample.get("camera_source_index", sample.get("frame_index", 0)))
        if index < 0 or index >= len(frames):
            raise ApiError(
                ErrorCode.INTERNAL_ERROR,
                "MCAP sample references a camera frame outside the payload",
                details={"camera_source_index": index, "frames": len(frames)},
            )
        yield _mcap_rgb(frames[index])


# ---------------------------------------------------------------------------
# helpers: encoding, idempotent asset registration, modalities
# ---------------------------------------------------------------------------


def _encode_letterbox_video(
    frames: Iterator[np.ndarray], destination: Path
) -> tuple[list[dict[str, Any]], int]:
    from embodied_dataops.processing.ffmpeg import encode_canvas_frames_h264

    records: list[dict[str, Any]] = []

    def canvas_frames() -> Iterator[np.ndarray]:
        for frame in frames:
            array = np.asarray(frame)
            if array.ndim != 3:
                raise ApiError(
                    ErrorCode.INTERNAL_ERROR,
                    "source frame for canonical video is not an RGB image",
                    details={"shape": list(array.shape)},
                )
            transform = compute_letterbox(int(array.shape[1]), int(array.shape[0]))
            records.append(transform.as_record())
            yield apply_letterbox(array, transform)

    written = encode_canvas_frames_h264(canvas_frames(), destination, fps=TARGET_FPS)
    return records, written


def _existing_asset(session: Session, key: str) -> Asset | None:
    return session.scalar(select(Asset).where(Asset.object_key == key))


def _conflict(key: str) -> ApiError:
    return ApiError(
        ErrorCode.CONFLICT,
        "canonical object key already holds different content; canonical versions are immutable",
        details={"key": key},
    )


def _register_bytes_idempotent(
    ctx: HandlerContext,
    *,
    key: str,
    data: bytes,
    media_type: str,
    origin: AssetOrigin,
    logical_name: str,
    meta: dict[str, Any] | None = None,
) -> Asset:
    digest = hashlib.sha256(data).hexdigest()
    existing = _existing_asset(ctx.session, key)
    if existing is not None:
        if existing.sha256 and existing.sha256 != digest:
            raise _conflict(key)
        return existing
    return register_bytes(
        ctx,
        key=key,
        data=data,
        media_type=media_type,
        origin=origin,
        logical_name=logical_name,
        meta=meta,
    )


def _register_file_idempotent(
    ctx: HandlerContext,
    *,
    key: str,
    path: Path,
    media_type: str,
    origin: AssetOrigin,
    logical_name: str,
    meta: dict[str, Any] | None = None,
) -> Asset:
    digest = sha256_file(path)
    existing = _existing_asset(ctx.session, key)
    if existing is not None:
        if existing.sha256 and existing.sha256 != digest:
            raise _conflict(key)
        return existing
    return register_file(
        ctx,
        key=key,
        path=path,
        media_type=media_type,
        origin=origin,
        logical_name=logical_name,
        meta=meta,
    )


def _upsert_modality(
    session: Session,
    *,
    episode_id: str,
    name: str,
    kind: str,
    dtype: str,
    shape: list[int],
    clock_domain: str,
    sample_count: int,
    min_t_ns: int | None,
    max_t_ns: int | None,
    source_key: str,
) -> Modality:
    modality = session.scalar(
        select(Modality).where(Modality.episode_id == episode_id, Modality.name == name)
    )
    if modality is None:
        modality = Modality(id=new_id(), episode_id=episode_id, name=name)
        session.add(modality)
    modality.kind = kind
    modality.dtype = dtype
    modality.shape_json = list(shape)
    modality.clock_domain = clock_domain
    modality.sample_count = int(sample_count)
    modality.min_t_ns = min_t_ns
    modality.max_t_ns = max_t_ns
    modality.source_key = source_key
    return modality


def import_context_payload_from(episode: Episode, source_import: Any, asset: Asset) -> dict[str, Any]:
    return {
        "import_id": source_import.id,
        "source_type": source_import.source_type,
        "data_origin": source_import.data_origin,
        "asset_sha256": asset.sha256,
        "asset_bytes": asset.byte_size,
        "filename": source_import.filename,
        "source_uri": source_import.source_uri,
        "source_revision": source_import.source_revision,
        "license": source_import.license,
        "mapping": dict(source_import.mapping_json or {}),
        "source_episode_index": episode.source_episode_index,
    }


HANDLER = NormalizeEpisodeHandler()

__all__ = ["CanonicalDraft", "HANDLER", "NormalizeEpisodeHandler"]
