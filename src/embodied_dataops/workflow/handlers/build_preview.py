"""BUILD_PREVIEW handler (SPEC 6.1, 9.2 ``GET /episodes/{id}/preview``).

The canonical preview artifacts are usually **already produced by
NORMALIZE_EPISODE** (``frames/front.mp4`` and ``frames/thumbnail.jpg``).  This
handler is therefore idempotent with respect to those immutable object keys:

* it looks Assets up by ``object_key`` and **reuses** the existing row instead of
  registering a duplicate Asset (``uq_assets_object_key`` would reject it and the
  whole transaction would roll back -- observed as BUILD_PREVIEW failing on real
  Ego pipelines, leaving READY_FOR_REVIEW unclaimable and the Episode PROCESSING);
* it only generates a 640x480 JPEG thumbnail when the canonical thumbnail key has
  no Asset yet (e.g. the normalizer skipped its optional thumbnail extraction);
* it reuses the Episode's camera Modality row and only fills in the preview video
  object key, so the ``episodes``/``modalities`` links stay consistent;
* a row that references a missing object, or a non-decodable stored JPEG, is a
  terminal ``FILE_DECODE_ERROR``/``OBJECT_NOT_FOUND`` rather than a silent
  overwrite of referenced immutable content.

The handler never commits, never changes ``tasks.state`` and never re-encodes
the canonical video.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops import SAMPLE_INTERVAL_NS, TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.db.models import Asset, Modality
from embodied_dataops.domain.enums import AssetOrigin, Step
from embodied_dataops.errors import CancelledError, ErrorCode, TerminalTaskError
from embodied_dataops.ids import new_id
from embodied_dataops.processing import preview
from embodied_dataops.storage.base import sha256_file
from embodied_dataops.workflow.handlers.base import (
    HandlerContext,
    HandlerResult,
    TaskHandler,
    register_bytes,
)
from embodied_dataops.workflow.handlers.detect_objects import resolve_canonical_inputs
from embodied_dataops.workflow.outbox import utcnow


class BuildPreviewHandler(TaskHandler):
    """Create/confirm the Episode thumbnail and preview references (SPEC 6.1)."""

    step = Step.BUILD_PREVIEW

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        if ctx.is_cancelled():
            raise CancelledError()
        inputs = resolve_canonical_inputs(ctx.session, ctx)
        session = ctx.session
        # The thumbnail lives under the same canonical version prefix as the video; the
        # resolver owns that choice because 1.0 and 2.0 use different layouts.
        video_key = inputs.video_key
        thumbnail_key = inputs.thumbnail_key
        video_asset = _asset_for_key(session, video_key)
        thumbnail_asset = _asset_for_key(session, thumbnail_key)

        # Assets are immutable: a row that references a missing object is a
        # storage inconsistency, never a reason to silently replace the object.
        if video_asset is not None and not ctx.storage.exists(video_key):
            raise TerminalTaskError(
                ErrorCode.OBJECT_NOT_FOUND,
                "Canonical video Asset exists but its object is missing",
                details={"video_key": video_key, "asset_id": video_asset.id},
            )
        if thumbnail_asset is not None and not ctx.storage.exists(thumbnail_key):
            raise TerminalTaskError(
                ErrorCode.OBJECT_NOT_FOUND,
                "Canonical thumbnail Asset exists but its object is missing",
                details={"thumbnail_key": thumbnail_key, "asset_id": thumbnail_asset.id},
            )

        video_path = ctx.workspace / "canonical_front.mp4"
        probe: preview.VideoProbe | None = None
        video_asset_created = False
        if video_asset is None:
            # NORMALIZE_EPISODE is expected to have written the object; register
            # the missing Asset row from the immutable object instead of
            # uploading a second copy (and never with a duplicate key).
            if not ctx.storage.exists(video_key):
                raise TerminalTaskError(
                    ErrorCode.OBJECT_NOT_FOUND,
                    "Canonical video object is missing; BUILD_PREVIEW cannot reference it",
                    details={"video_key": video_key, "episode_id": inputs.episode.id},
                )
            ctx.storage.download_to(video_key, video_path)
            video_asset = Asset(
                id=new_id(),
                sha256=sha256_file(video_path),
                byte_size=video_path.stat().st_size,
                media_type="video/mp4",
                object_key=video_key,
                origin=AssetOrigin.CANONICAL.value,
                logical_name="canonical:frames/front.mp4",
                meta_json={"width": TARGET_WIDTH, "height": TARGET_HEIGHT, "fps": TARGET_FPS},
                created_at=utcnow(),
            )
            session.add(video_asset)
            session.flush()
            video_asset_created = True

        thumbnail_created = False
        if thumbnail_asset is not None:
            # Reuse the normalizer's thumbnail (immutable derived Asset).
            thumbnail_width, thumbnail_height = preview.decode_jpeg_dimensions(
                ctx.storage.get_bytes(thumbnail_key)
            )
        else:
            if not video_path.exists():
                ctx.storage.download_to(video_key, video_path)
            probe = preview.probe_video(video_path)
            thumbnail = preview.make_thumbnail(video_path)
            # No Asset row references this key, so a stale object from a rolled
            # back attempt (orphan) may be overwritten safely.
            thumbnail_asset = register_bytes(
                ctx,
                key=thumbnail_key,
                data=thumbnail.data,
                media_type="image/jpeg",
                origin=AssetOrigin.DERIVED,
                logical_name="canonical:frames/thumbnail.jpg",
                meta={
                    "width": thumbnail.width,
                    "height": thumbnail.height,
                    "frame_index": thumbnail.frame_index,
                    "t_ns": thumbnail.t_ns,
                    "source_video_key": video_key,
                    "src_w": thumbnail.src_w,
                    "src_h": thumbnail.src_h,
                    "scale": round(thumbnail.scale, 8),
                    "pad_left": thumbnail.pad_left,
                    "pad_top": thumbnail.pad_top,
                    "jpeg_quality": thumbnail.jpeg_quality,
                    "source": "build_preview",
                },
            )
            thumbnail_created = True
            thumbnail_width, thumbnail_height = thumbnail.width, thumbnail.height

        # Video facts (no extra decode when the normalizer already registered them).
        video_meta = dict(video_asset.meta_json or {})
        if probe is not None:
            width, height, fps, frame_count, backend = (
                probe.width,
                probe.height,
                probe.fps,
                probe.frame_count,
                probe.backend,
            )
        else:
            width = int(video_meta.get("width") or inputs.metadata.width or TARGET_WIDTH)
            height = int(video_meta.get("height") or inputs.metadata.height or TARGET_HEIGHT)
            fps = float(video_meta.get("fps") or TARGET_FPS)
            frame_count = int(inputs.metadata.sample_count or 0)
            backend = "reused_asset"

        # Reuse the camera Modality row created by NORMALIZE_EPISODE; never insert
        # a second row for the same (episode_id, name) unique key.
        camera_name = inputs.metadata.camera_name or "front"
        sample_count = int(inputs.metadata.sample_count or 0)
        max_t_ns = max(0, (sample_count - 1) * SAMPLE_INTERVAL_NS) if sample_count else None
        modality = _find_camera_modality(session, inputs.episode.id, camera_name)
        if modality is None:
            modality = Modality(
                id=new_id(),
                episode_id=inputs.episode.id,
                name=camera_name,
                kind="camera",
                dtype="video/mp4",
                shape_json=[height, width, 3],
                clock_domain=inputs.metadata.clock_domain,
                sample_count=sample_count,
                min_t_ns=0 if sample_count else None,
                max_t_ns=max_t_ns,
                source_key=camera_name,
                object_key=video_key,
            )
            session.add(modality)
        else:
            modality.object_key = video_key
            if not modality.sample_count and sample_count:
                modality.sample_count = sample_count
                modality.min_t_ns = 0
                modality.max_t_ns = max_t_ns
        session.flush()

        return HandlerResult(
            data={
                "thumbnail_key": thumbnail_key,
                "thumbnail_asset_id": thumbnail_asset.id,
                "thumbnail_sha256": thumbnail_asset.sha256,
                "thumbnail_reused": not thumbnail_created,
                "thumbnail_width": thumbnail_width,
                "thumbnail_height": thumbnail_height,
                "thumbnail_frame_index": 0 if thumbnail_created else None,
                "video_key": video_key,
                "video_asset_id": video_asset.id,
                "video_sha256": video_asset.sha256,
                "video_asset_created": video_asset_created,
                "video_asset_reused": not video_asset_created,
                "modality_id": modality.id,
                "width": width,
                "height": height,
                "fps": round(float(fps), 6),
                "frame_count": frame_count,
                "decode_backend": backend,
                "canonical_version": inputs.canonical_version,
            },
            output_asset_id=thumbnail_asset.id,
        )


def _asset_for_key(session: Session, key: str) -> Asset | None:
    return session.scalars(select(Asset).where(Asset.object_key == key)).first()


def _find_camera_modality(session: Session, episode_id: str, camera_name: str) -> Modality | None:
    """Prefer the exact camera modality name; fall back to any camera stream."""
    modality = session.scalars(
        select(Modality).where(
            Modality.episode_id == episode_id,
            Modality.name == camera_name,
        )
    ).first()
    if modality is not None:
        return modality
    return session.scalars(
        select(Modality)
        .where(Modality.episode_id == episode_id, Modality.kind == "camera")
        .order_by(Modality.name)
    ).first()


#: Registry integration point used by the Lead-owned handler registry.
HANDLER = BuildPreviewHandler()
