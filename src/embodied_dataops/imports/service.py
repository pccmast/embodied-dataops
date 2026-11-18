"""Source upload flow (SPEC 5.1, 9.2, 12.1).

Flow: `initiate` creates a SourceImport with a short-lived presigned PUT URL ->
client uploads bytes directly to the object store (never through the API) ->
`complete` re-downloads/re-hashes the object server-side, promotes it to the
immutable ``raw/sha256/<sha256>`` key and enqueues the import job DAG.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import Asset, CollectionOrder, ImportJob, SourceImport
from embodied_dataops.domain.enums import AssetOrigin, DataOrigin, SourceType, UploadState
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import is_uuid, new_id
from embodied_dataops.observability.audit import record_audit
from embodied_dataops.storage.base import get_storage
from embodied_dataops.workflow.dag import create_import_job_tasks

SOURCE_MEDIA_TYPES: dict[SourceType, str] = {
    SourceType.LEROBOT_V3: "application/gzip",
    SourceType.EGO_MP4: "video/mp4",
    SourceType.MCAP_ROS2_PROFILE_V1: "application/octet-stream",
    SourceType.MCAP_ROS2_PROFILE_V2: "application/octet-stream",
}

SOURCE_EXTENSIONS: dict[SourceType, tuple[str, ...]] = {
    SourceType.LEROBOT_V3: (".tar.gz", ".tgz"),
    SourceType.EGO_MP4: (".mp4",),
    SourceType.MCAP_ROS2_PROFILE_V1: (".mcap",),
    SourceType.MCAP_ROS2_PROFILE_V2: (".mcap",),
}

ALLOWED_MAPPING_KEYS: dict[SourceType, set[str]] = {
    SourceType.LEROBOT_V3: {
        "camera_key",
        "camera_keys",
        "state_key",
        "action_key",
        "robot_profile",
        "profile_id",
        "profile_revision",
        "diagnostic_only",
    },
    SourceType.EGO_MP4: {"camera_key"},
    SourceType.MCAP_ROS2_PROFILE_V1: {"camera_key", "robot_profile"},
    SourceType.MCAP_ROS2_PROFILE_V2: {
        "camera_key",
        "wrist_camera_key",
        "profile_id",
        "profile_revision",
        "diagnostic_only",
    },
}

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def safe_filename(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = _SAFE_NAME_RE.sub("_", name).strip("._") or "upload.bin"
    return name[:180]


def _parse_source_type(value: str) -> SourceType:
    try:
        return SourceType(value)
    except ValueError as exc:
        raise ApiError(
            ErrorCode.UNSUPPORTED_SOURCE_TYPE,
            f"Unsupported source_type: {value}",
            details={"supported": [item.value for item in SourceType]},
        ) from exc


def _validate_mapping(source_type: SourceType, mapping: dict[str, Any] | None) -> dict[str, Any]:
    mapping = dict(mapping or {})
    allowed = ALLOWED_MAPPING_KEYS[source_type]
    unknown = set(mapping) - allowed
    if unknown:
        raise ApiError(
            ErrorCode.INVALID_MAPPING,
            "mapping contains unsupported fields",
            details={"unsupported": sorted(unknown), "allowed": sorted(allowed)},
        )
    if source_type is SourceType.LEROBOT_V3:
        mapping.setdefault("state_key", "observation.state")
        mapping.setdefault("action_key", "action")
        for key in ("camera_key", "state_key", "action_key"):
            if key in mapping and (not isinstance(mapping[key], str) or not mapping[key].strip()):
                raise ApiError(ErrorCode.INVALID_MAPPING, f"mapping.{key} must be a non-empty string")
        profile = mapping.get("robot_profile")
        if profile is not None and profile != "arm6_joint_position_v1":
            raise ApiError(
                ErrorCode.INVALID_MAPPING,
                "Only robot_profile arm6_joint_position_v1 is supported in v1",
                details={"robot_profile": profile},
            )
    return mapping


def _validate_collection_order_reference(
    session: Session, value: Any
) -> str | None:
    """Validate an optional V2 collection-order provenance reference (C6).

    Only an ``APPROVED`` or ``IN_PROGRESS`` order may be referenced; the value is
    persisted on ``source_imports.collection_order_id`` and never rewritten.
    """
    if value in (None, ""):
        return None
    candidate = str(value)
    if not is_uuid(candidate):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "collection_order_id must be a UUID",
            details={"collection_order_id": candidate[:64]},
        )
    order = session.get(CollectionOrder, candidate)
    if order is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "Collection order not found",
            details={"collection_order_id": candidate},
        )
    if order.state not in {"APPROVED", "IN_PROGRESS"}:
        raise ApiError(
            ErrorCode.CONFLICT,
            "Only APPROVED or IN_PROGRESS collection orders can be referenced by an import",
            details={"collection_order_id": candidate, "state": order.state},
        )
    return candidate


def initiate_import(
    session: Session, *, principal: Principal, payload: dict[str, Any], request_id: str
) -> dict[str, Any]:
    settings = get_settings()
    storage = get_storage()
    source_type = _parse_source_type(str(payload.get("source_type", "")))
    filename = str(payload.get("filename", "") or "")
    size_bytes = int(payload.get("size_bytes", 0) or 0)
    sha256 = str(payload.get("sha256", "") or "").lower()
    data_origin_raw = str(payload.get("data_origin", "") or "")
    license_text = str(payload.get("license", "") or "")

    if not filename:
        raise ApiError(ErrorCode.INVALID_INPUT, "filename is required")
    lowered = filename.lower()
    if not lowered.endswith(SOURCE_EXTENSIONS[source_type]):
        raise ApiError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            f"filename must end with one of {SOURCE_EXTENSIONS[source_type]} for {source_type.value}",
        )
    if size_bytes <= 0:
        raise ApiError(ErrorCode.INVALID_INPUT, "size_bytes must be positive")
    if size_bytes > settings.max_upload_bytes:
        raise ApiError(
            ErrorCode.SOURCE_SIZE_LIMIT,
            "Source exceeds the 2 GiB upload limit",
            details={"size_bytes": size_bytes, "limit": settings.max_upload_bytes},
        )
    if not _SHA_RE.match(sha256):
        raise ApiError(ErrorCode.INVALID_INPUT, "sha256 must be 64 lowercase hex characters")
    try:
        data_origin = DataOrigin(data_origin_raw)
    except ValueError as exc:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            f"data_origin must be one of {[item.value for item in DataOrigin]}",
        ) from exc
    if data_origin is DataOrigin.PUBLIC_DATASET:
        if not license_text.strip():
            raise ApiError(ErrorCode.INVALID_INPUT, "license is required for public_dataset imports")
        if not payload.get("source_uri") or not payload.get("source_revision"):
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "source_uri and source_revision are required for public_dataset imports",
            )
    mapping = _validate_mapping(source_type, payload.get("mapping"))
    collection_order_id = _validate_collection_order_reference(
        session, payload.get("collection_order_id")
    )

    import_id = new_id()
    staging_key = f"uploads/{import_id}/{safe_filename(filename)}"
    record = SourceImport(
        id=import_id,
        source_type=source_type.value,
        upload_state=UploadState.INITIATED.value,
        data_origin=data_origin.value,
        license=license_text,
        source_uri=payload.get("source_uri"),
        source_revision=payload.get("source_revision"),
        filename=filename,
        size_bytes=size_bytes,
        sha256=sha256,
        mapping_json=mapping,
        collection_order_id=collection_order_id,
        staging_key=staging_key,
        created_by=principal.key_id,
        created_at=_utcnow(),
    )
    session.add(record)
    session.flush()
    storage.ensure_bucket()
    put_url = storage.presign_put(staging_key, settings.upload_url_ttl_seconds)
    expires_at = _utcnow() + timedelta(seconds=settings.upload_url_ttl_seconds)
    record_audit(
        session,
        actor=principal,
        action="import.initiate",
        entity_type="source_import",
        entity_id=import_id,
        after={
            "source_type": source_type.value,
            "size_bytes": size_bytes,
            "collection_order_id": collection_order_id,
        },
        request_id=request_id,
    )
    session.flush()
    return {
        "import_id": import_id,
        "put_url": put_url,
        "expires_at": _iso(expires_at),
        "staging_key": staging_key,
    }


def _existing_job(session: Session, import_id: str) -> ImportJob | None:
    return session.scalars(
        select(ImportJob).where(ImportJob.import_id == import_id).order_by(ImportJob.created_at.desc())
    ).first()


def complete_import(
    session: Session, *, principal: Principal, import_id: str, request_id: str
) -> dict[str, Any]:
    storage = get_storage()
    record = session.get(SourceImport, import_id)
    if record is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Source import not found")

    existing = _existing_job(session, import_id)
    if existing is not None:
        return {"import_job_id": existing.id, "state": existing.state, "import_id": import_id}

    if record.upload_state not in (UploadState.INITIATED.value, UploadState.UPLOADED.value):
        raise ApiError(
            ErrorCode.UPLOAD_STATE_INVALID,
            "Import is not in a state where it can be completed",
            details={"upload_state": record.upload_state},
        )
    if not record.staging_key:
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, "Upload session has no staging object")
    if not storage.exists(record.staging_key):
        raise ApiError(ErrorCode.SOURCE_UPLOAD_INCOMPLETE, "Uploaded object is missing from the object store")

    stat = storage.stat(record.staging_key)
    if stat.size != record.size_bytes:
        raise ApiError(
            ErrorCode.SOURCE_UPLOAD_INCOMPLETE,
            "Uploaded object size does not match the declared size",
            details={"expected": record.size_bytes, "actual": stat.size},
        )
    actual_sha = storage.sha256_of(record.staging_key)
    if record.sha256 and actual_sha != record.sha256:
        raise ApiError(
            ErrorCode.SOURCE_HASH_MISMATCH,
            "Uploaded object sha256 does not match the declared sha256",
            details={"expected": record.sha256, "actual": actual_sha},
        )

    raw_key = f"raw/sha256/{actual_sha}"
    if not storage.exists(raw_key):
        storage.copy(record.staging_key, raw_key)
    # SPEC 4.2: identical content may be referenced by one immutable Asset while
    # different Source Imports each reference it.  Re-importing the same bytes
    # must therefore reuse the existing raw Asset instead of inserting a second
    # row with the same object_key (unique constraint).
    asset = session.scalars(select(Asset).where(Asset.object_key == raw_key)).first()
    if asset is None:
        asset = Asset(
            id=new_id(),
            sha256=actual_sha,
            byte_size=stat.size,
            media_type=SOURCE_MEDIA_TYPES[_parse_source_type(record.source_type)],
            object_key=raw_key,
            origin=AssetOrigin.RAW.value,
            logical_name="source",
            meta_json={
                "filename": record.filename,
                "source_type": record.source_type,
                "data_origin": record.data_origin,
                "license": record.license,
            },
        )
        session.add(asset)
        session.flush()
    elif asset.sha256 != actual_sha or asset.byte_size != stat.size:
        raise ApiError(
            ErrorCode.CONFLICT,
            "Existing raw asset row does not match the stored object bytes",
            details={"object_key": raw_key, "asset_sha256": asset.sha256},
        )
    record.asset_id = asset.id
    record.sha256 = actual_sha
    record.size_bytes = stat.size
    record.upload_state = UploadState.UPLOADED.value
    session.flush()
    storage.delete(record.staging_key)

    job = create_import_job_tasks(session, record)
    record_audit(
        session,
        actor=principal,
        action="import.complete",
        entity_type="source_import",
        entity_id=import_id,
        after={"sha256": actual_sha, "import_job_id": job.id},
        request_id=request_id,
    )
    session.flush()
    return {"import_job_id": job.id, "state": job.state, "import_id": import_id}


def get_import_detail(session: Session, import_id: str) -> dict[str, Any]:
    record = session.get(SourceImport, import_id)
    if record is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Source import not found")
    job = _existing_job(session, import_id)
    from embodied_dataops.db.models import Episode

    episodes = list(
        session.scalars(
            select(Episode)
            .where(Episode.source_import_id == import_id)
            .order_by(Episode.source_episode_index)
        )
    )
    return {
        "import": {
            "id": record.id,
            "source_type": record.source_type,
            "upload_state": record.upload_state,
            "data_origin": record.data_origin,
            "license": record.license,
            "source_uri": record.source_uri,
            "source_revision": record.source_revision,
            "filename": record.filename,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "mapping": record.mapping_json,
            "collection_order_id": record.collection_order_id,
            "created_at": _iso(record.created_at),
            "completed_at": _iso(record.completed_at),
            "error_code": record.error_code,
            "error_detail": record.error_detail,
        },
        "job": (
            {
                "id": job.id,
                "state": job.state,
                "error_code": job.error_code,
                "created_at": _iso(job.created_at),
                "started_at": _iso(job.started_at),
                "finished_at": _iso(job.finished_at),
            }
            if job
            else None
        ),
        "episodes_summary": {
            "count": len(episodes),
            "episodes": [
                {
                    "id": episode.id,
                    "source_episode_index": episode.source_episode_index,
                    "episode_type": episode.episode_type,
                    "status": episode.status,
                    "quality_status": episode.quality_status,
                }
                for episode in episodes
            ],
        },
    }
