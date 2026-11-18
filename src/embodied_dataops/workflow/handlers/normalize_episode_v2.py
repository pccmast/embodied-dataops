"""NORMALIZE_EPISODE v2 handler: write ``canonical-episode/2.0`` (V2 SPEC 2.3/2.4).

Dispatched from :class:`embodied_dataops.workflow.handlers.normalize_episode.NormalizeEpisodeHandler`
when :func:`applies_to` is true (explicit profile selection in the import mapping,
``diagnostic_only`` or an MCAP_ROS2_PROFILE_V2 source).  The V1 path stays
byte-identical for every V1 import.

What the handler writes (final deterministic keys, immutable assets):

* ``canonical/{episode_id}/v2/{version}/metadata.json`` (canonical JSON, SHA-256
  recorded on the episode);
* native camera timestamps + native state/action indices (source order/times);
* the derived 10 Hz aligned view (invalid rows carry null arrays and reasons);
* ``mapping_report.json`` (explicit mapping decisions + tolerances + diagnostics);
* regenerable front/wrist previews (``preview/*_10hz_640x480.mp4``);
* ``canonical_views`` rows for NATIVE / ALIGNED_TRAINING / PREVIEW;
* ``modalities`` rows with stream_index / native_rate_hz / native+aligned index
  assets / timestamp_provenance / clock_verified;
* the episode profile binding, ``canonical_schema_version=2.0`` and a new
  ``canonical_version`` snapshot pointer.  ``business_revision`` is untouched
  (only the condition-write layer owns it).

The handler never commits or rolls back and never touches ``tasks.state``; the
workflow engine owns the transaction/lease fencing.
"""

from __future__ import annotations

import hashlib
from datetime import UTC
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_keys
from embodied_dataops.canonical.v2.build import build_canonical_v2
from embodied_dataops.canonical.v2.contract import (
    ALIGNED_FPS,
    CANONICAL_V2_SCHEMA_VERSION,
    CanonicalMetadataV2,
    canonical_sha256_v2,
    canonical_v2_keys,
)
from embodied_dataops.canonical.v2.model import CanonicalV2Source, PreviewArtifact
from embodied_dataops.canonical.v2.preview import encode_camera_preview
from embodied_dataops.canonical.v2.views import build_all_view_configs, register_canonical_views
from embodied_dataops.db.models import Asset, Episode, Modality, PipelineRun, RobotProfile
from embodied_dataops.domain.enums import AssetOrigin, SourceType, Step
from embodied_dataops.domain.profiles import (
    RobotProfile as RobotProfileType,
)
from embodied_dataops.domain.profiles import (
    combined_registry,
    parse_profile,
    resolve_profile,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.imports.common import (
    build_import_context,
    load_asset,
    load_source_import,
    materialize_source,
    utcnow,
)
from embodied_dataops.storage.base import StorageBackend, sha256_file
from embodied_dataops.workflow.handlers.base import (
    HandlerContext,
    HandlerResult,
    TaskHandler,
    register_bytes,
    register_file,
)

PIPELINE_VERSION = "2.0.0"


def applies_to(source_import: Any, episode: Any) -> bool:
    """True when the import explicitly opted into canonical-episode/2.0.

    V1 imports (no ``profile_id``/``profile_revision``, not diagnostic-only,
    not MCAP_ROS2_PROFILE_V2) keep the V1 path untouched.
    """
    mapping = dict(getattr(source_import, "mapping_json", None) or {})
    if str(getattr(episode, "canonical_schema_version", "1.0") or "1.0") == "2.0":
        return True
    if mapping.get("diagnostic_only") is True:
        return True
    if mapping.get("profile_id") and mapping.get("profile_revision") is not None:
        return True
    return str(getattr(source_import, "source_type", "")) == SourceType.MCAP_ROS2_PROFILE_V2.value


def _profiles_for_resolution(session: Session) -> list[RobotProfileType]:
    """Registered revisions overlaid on the shipped ``2.0`` + ``2.1`` registry.

    Each row is parsed by its **own** ``schema_version``: forcing every row through
    the 2.0 model is what made a ``robot-profile/2.1`` import fail with 19 field
    errors, and it also poisoned V2 normalisation once a 2.1 row was in the table.
    """
    registered = [parse_profile(dict(row.config_json)) for row in session.scalars(select(RobotProfile))]
    return combined_registry(registered)


class NormalizeEpisodeV2Handler(TaskHandler):
    step = Step.NORMALIZE_EPISODE

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        session: Session = ctx.session
        run = session.get(PipelineRun, ctx.task.run_id) if ctx.task.run_id else None
        if run is None:
            raise ApiError(
                ErrorCode.INVALID_INPUT, "NORMALIZE_EPISODE is not linked to a pipeline run"
            )
        episode = session.get(Episode, run.episode_id)
        if episode is None:
            raise ApiError(ErrorCode.NOT_FOUND, "episode of this pipeline run does not exist")
        source_import = load_source_import(session, episode.source_import_id)
        asset = load_asset(session, source_import)
        source_path = materialize_source(ctx.storage, source_import, asset, ctx.workspace)
        context = build_import_context(source_import, source_path, ctx.workspace)
        if not context.source_sha256:
            context.source_sha256 = asset.sha256

        mapping = dict(source_import.mapping_json or {})
        resolved = resolve_profile(mapping, profiles=_profiles_for_resolution(session))
        source = self._read_source(
            source_import.source_type, context, episode, ctx, mapping
        )
        source.episode_id = episode.id
        if not source.source_asset_sha256:
            source.source_asset_sha256 = asset.sha256
        # ``import_context_payload`` 在 imports.common 里只接受一个 bundle；
        # 这里需要的是 normalize_episode 的三参数版本（episode/source/asset）。
        # 延迟导入避免与 normalize_episode 的顶层导入形成环。
        from embodied_dataops.workflow.handlers.normalize_episode import (
            import_context_payload_from,
        )

        source.source["import_context"] = import_context_payload_from(episode, source_import, asset)

        canonical_version = self._next_canonical_version(ctx.storage, episode)
        keys = canonical_v2_keys(episode.id, canonical_version)
        workdir = Path(ctx.workspace) / f"canonical_v2_{canonical_version}"
        workdir.mkdir(parents=True, exist_ok=True)

        previews, preview_paths = self._build_previews(ctx, source, workdir, keys)
        bundle = build_canonical_v2(
            source,
            canonical_version=canonical_version,
            previews=previews,
            pipeline_version=PIPELINE_VERSION,
            created_at=_created_at_text(episode),
        )
        published = self._publish(
            ctx,
            episode=episode,
            source_import=source_import,
            source=source,
            source_path=context.source_path,
            resolved=resolved,
            bundle=bundle,
            keys=keys,
            canonical_version=canonical_version,
            preview_paths=preview_paths,
        )
        data = {
            "step": self.step.value,
            "schema_version": CANONICAL_V2_SCHEMA_VERSION,
            "adapter": "normalize_episode_v2",
            "episode_id": episode.id,
            "episode_type": source.episode_type,
            "source_type": source_import.source_type,
            "profile_id": source.profile_id,
            "profile_revision": source.profile_revision,
            "profile_config_sha256": source.profile_config_sha256,
            "profile_authority": source.profile_authority,
            "semantic_authority": source.semantic_authority,
            "diagnostic_only": source.diagnostic_only,
            "canonical_schema_version": CANONICAL_V2_SCHEMA_VERSION,
            "canonical_version": canonical_version,
            "publishable": bundle.publishable,
            "publish_blockers": list(bundle.blocking_reasons),
            "camera_names": list(source.camera_names),
            "joint_names": list(source.joint_names),
            "state_dim": int(bundle.metadata.state_dim),
            "action_dim": int(bundle.metadata.action_dim),
            "aligned_fps": ALIGNED_FPS,
            "sample_count": int(bundle.metadata.aligned_view.sample_count),
            "valid_sample_count": int(bundle.metadata.aligned_view.valid_sample_count),
            "duration_ns": int(bundle.metadata.duration_ns),
            "float32_precision": bundle.precision,
            "diagnostics": list(bundle.metadata.diagnostics),
            "alignment": bundle.alignment_diagnostics,
            "mapping_report": bundle.mapping_report,
            "canonical": published,
        }
        return HandlerResult(
            data=data,
            verdict=None,
            output_asset_id=published["metadata_asset_id"],
        )

    # -- adapter selection -------------------------------------------------
    def _read_source(
        self,
        source_type: str,
        context: Any,
        episode: Episode,
        ctx: HandlerContext,
        mapping: dict[str, Any],
    ) -> CanonicalV2Source:
        if source_type == SourceType.LEROBOT_V3.value:
            from embodied_dataops.imports.lerobot_v3_v2 import LerobotV3V2Parser

            return LerobotV3V2Parser().read_v2_episode(context, episode.source_episode_index)
        if source_type == SourceType.MCAP_ROS2_PROFILE_V2.value:
            from embodied_dataops.imports.mcap_profile_v2 import McapProfileV2Parser

            return McapProfileV2Parser().read_v2_episode(context, 0)
        if source_type == SourceType.EGO_MP4.value:
            # Deliberate: canonical-episode/2.0 exists to carry the native/aligned
            # state+action pair, and an Ego video has neither (SPEC_V2.1 §4.3).  It
            # keeps canonical-episode/1.0, and EXTRACT_KEYFRAMES reads that video
            # directly instead of upgrading the contract to hold invented channels.
            raise ApiError(
                ErrorCode.UNSUPPORTED_SOURCE_TYPE,
                "an Ego video has no robot state/action, so it stays canonical-episode/1.0; "
                "the AI keyframe stage reads its canonical v1 video",
                details={"source_type": source_type, "canonical_schema_version": "1.0"},
            )
        raise ApiError(
            ErrorCode.UNSUPPORTED_SOURCE_TYPE,
            f"canonical-episode/2.0 does not support source_type {source_type}",
            details={"supported": [SourceType.LEROBOT_V3.value, SourceType.MCAP_ROS2_PROFILE_V2.value]},
        )

    # -- previews ----------------------------------------------------------
    def _build_previews(
        self,
        ctx: HandlerContext,
        source: CanonicalV2Source,
        workdir: Path,
        keys: dict[str, str],
    ) -> tuple[list[PreviewArtifact], dict[str, Path]]:
        previews: list[PreviewArtifact] = []
        paths: dict[str, Path] = {}
        origin = 0 if source.alignment_mode == "prealigned" else int(source.origin_ns)
        for camera in source.cameras:
            if camera.frame_getter is None:
                continue
            destination = workdir / f"{camera.name}_preview.mp4"
            encoding = encode_camera_preview(camera, destination=destination, origin_ns=origin)
            if encoding.frame_count == 0:
                continue
            paths[camera.name] = destination
            previews.append(
                PreviewArtifact(
                    camera_name=camera.name,
                    asset_key=keys["preview"].format(camera=camera.name),
                    asset_sha256=sha256_file(destination),
                    fps=10,
                    width=640,
                    height=480,
                    frame_count=encoding.frame_count,
                )
            )
        return previews, paths

    # -- publication -------------------------------------------------------
    def _publish(
        self,
        ctx: HandlerContext,
        *,
        episode: Episode,
        source_import: Any,
        source: CanonicalV2Source,
        source_path: Path,
        resolved: Any,
        bundle: Any,
        keys: dict[str, str],
        canonical_version: int,
        preview_paths: dict[str, Path],
    ) -> dict[str, Any]:
        session = ctx.session
        artifacts = bundle.artifacts
        media_types = {
            "metadata": "application/json",
            "mapping_report": "application/json",
        }
        registered: dict[str, Asset] = {}
        native_hashes: dict[str, str] = {}
        for artifact_name, data in artifacts.items():
            if artifact_name == "native_camera:front":
                key = keys["native_camera"].format(camera="front")
            elif artifact_name.startswith("native_camera:"):
                key = keys["native_camera"].format(camera=artifact_name.split(":", 1)[1])
            else:
                key = {
                    "metadata": keys["metadata"],
                    "mapping_report": keys["mapping_report"],
                    "native_state": keys["native_state"],
                    "native_action": keys["native_action"],
                    "aligned_samples": keys["aligned_samples"],
                }[artifact_name]
            media_type = media_types.get(artifact_name, "application/vnd.apache.parquet")
            asset = _register_bytes_idempotent(
                ctx,
                key=key,
                data=data,
                media_type=media_type,
                origin=AssetOrigin.CANONICAL,
                logical_name=f"canonical-v2:{artifact_name}",
                meta={
                    "schema_version": CANONICAL_V2_SCHEMA_VERSION,
                    "canonical_version": canonical_version,
                    "artifact": artifact_name,
                },
            )
            registered[artifact_name] = asset
            if artifact_name.startswith("native_camera:") or artifact_name in {
                "native_state",
                "native_action",
            }:
                native_hashes[artifact_name] = asset.sha256
        # previews are derived display artifacts
        preview_assets: dict[str, Asset] = {}
        for preview in bundle.metadata.preview_views:
            path = preview_paths.get(preview.camera_name)
            if path is None:
                continue
            key = keys["preview"].format(camera=preview.camera_name)
            asset = _register_file_idempotent(
                ctx,
                key=key,
                path=path,
                media_type="video/mp4",
                origin=AssetOrigin.DERIVED,
                logical_name=f"canonical-v2:preview:{preview.camera_name}",
                meta={
                    "fps": preview.fps,
                    "width": preview.width,
                    "height": preview.height,
                    "lossy": True,
                    "not_training_truth": True,
                },
            )
            preview_assets[preview.camera_name] = asset
        metadata_asset = registered["metadata"]
        aligned_asset = registered["aligned_samples"]
        canonical_digest = canonical_sha256_v2(
            metadata_asset.sha256, aligned_asset.sha256, native_hashes
        )
        episode.canonical_sha256 = canonical_digest
        episode.canonical_version = int(canonical_version)
        episode.canonical_schema_version = "2.0"
        episode.profile_id = source.profile_id
        episode.profile_revision = source.profile_revision
        episode.duration_ns = int(bundle.metadata.duration_ns)
        episode.sample_count = int(bundle.metadata.aligned_view.sample_count)
        episode.updated_at = utcnow()
        self._sync_modalities(
            session,
            episode=episode,
            source=source,
            metadata=bundle.metadata,
            registered=registered,
        )
        view_configs = build_all_view_configs(source)
        views = register_canonical_views(
            session,
            episode_id=episode.id,
            canonical_version=canonical_version,
            metadata_asset_id=metadata_asset.id,
            configs=view_configs,
        )
        session.flush()
        return {
            "prefix": keys["prefix"],
            "schema_version": CANONICAL_V2_SCHEMA_VERSION,
            "canonical_version": canonical_version,
            "canonical_sha256": canonical_digest,
            "metadata_asset_id": metadata_asset.id,
            "metadata_key": keys["metadata"],
            "metadata_sha256": metadata_asset.sha256,
            "mapping_report_key": keys["mapping_report"],
            "mapping_report_sha256": registered["mapping_report"].sha256,
            "aligned_samples_key": keys["aligned_samples"],
            "aligned_samples_sha256": aligned_asset.sha256,
            "native_assets": {
                name: {"key": keys[name], "sha256": registered[name].sha256}
                for name in ("native_state", "native_action")
                if name in registered
            },
            "camera_assets": {
                artifact.split(":", 1)[1]: {
                    "key": keys["native_camera"].format(camera=artifact.split(":", 1)[1]),
                    "sha256": asset.sha256,
                }
                for artifact, asset in registered.items()
                if artifact.startswith("native_camera:")
            },
            "preview_assets": {
                name: {"key": keys["preview"].format(camera=name), "sha256": asset.sha256}
                for name, asset in preview_assets.items()
            },
            "views": [
                {"view_kind": view.view_kind, "config_sha256": view.config_sha256, "id": view.id}
                for view in views
            ],
            "publishable": bundle.publishable,
            "publish_blockers": list(bundle.blocking_reasons),
            "source_path_present": bool(source_path),
            "profile_mode": resolved.mode,
        }

    def _sync_modalities(
        self,
        session: Session,
        *,
        episode: Episode,
        source: CanonicalV2Source,
        metadata: CanonicalMetadataV2,
        registered: dict[str, Asset],
    ) -> None:
        aligned_asset = registered["aligned_samples"]
        stream_index_by_name: dict[str, int] = {}
        for index, name in enumerate(source.camera_names):
            stream_index_by_name[name] = index
        for modality in metadata.modalities:
            is_camera = modality.kind == "camera"
            stream_index = stream_index_by_name.get(modality.name, 0) if is_camera else 0
            if is_camera:
                native_asset = registered.get(f"native_camera:{modality.name}")
            else:
                native_asset = registered.get(f"native_{modality.name}")
            if native_asset is None:
                continue
            modality_row = session.scalar(
                select(Modality).where(
                    Modality.episode_id == episode.id,
                    Modality.name == modality.name,
                    Modality.stream_index == stream_index,
                )
            )
            if modality_row is None:
                modality_row = Modality(
                    id=new_id(),
                    episode_id=episode.id,
                    name=modality.name,
                    stream_index=stream_index,
                )
                session.add(modality_row)
            modality_row.kind = modality.kind
            modality_row.dtype = modality.dtype
            modality_row.shape_json = list(modality.shape)
            modality_row.clock_domain = modality.source_clock_domain
            modality_row.sample_count = int(modality.native_sample_count)
            modality_row.min_t_ns = 0 if modality.native_sample_count else None
            modality_row.max_t_ns = (
                int(metadata.duration_ns) if modality.native_sample_count else None
            )
            modality_row.source_key = modality.source_key
            modality_row.object_key = modality.native_asset_key
            modality_row.native_rate_hz = modality.native_rate_hz
            modality_row.native_index_asset_id = native_asset.id
            modality_row.aligned_index_asset_id = aligned_asset.id
            modality_row.timestamp_provenance = modality.timestamp_provenance
            modality_row.clock_verified = bool(modality.clock_verified)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _next_canonical_version(storage: StorageBackend, episode: Episode) -> int:
        current = max(1, int(episode.canonical_version or 1))
        v1_keys = canonical_keys(episode.id, current)
        try:
            v1_exists = storage.exists(v1_keys["metadata"])
        except Exception:  # pragma: no cover - storage-specific
            v1_exists = False
        if v1_exists:
            # SPEC 2.2: the first successful v2 output creates a NEW canonical
            # version snapshot; the old v1 assets stay untouched.
            return current + 1
        return current


def _created_at_text(episode: Episode) -> str:
    created_at = episode.created_at
    if created_at is None:
        return ""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at.astimezone(UTC).isoformat()


# ---------------------------------------------------------------------------
# idempotent asset registration (final deterministic keys; immutable content)
# ---------------------------------------------------------------------------


def _existing_asset(session: Session, key: str) -> Asset | None:
    return session.scalar(select(Asset).where(Asset.object_key == key))


def _conflict(key: str) -> ApiError:
    return ApiError(
        ErrorCode.CONFLICT,
        "canonical v2 object key already holds different content; canonical versions are immutable",
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


HANDLER = NormalizeEpisodeV2Handler()

__all__ = ["HANDLER", "NormalizeEpisodeV2Handler", "applies_to"]
