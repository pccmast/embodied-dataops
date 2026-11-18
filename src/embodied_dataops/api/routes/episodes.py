"""Episode read routes (SPEC 9.2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, tuple_
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import decode_cursor, encode_cursor, get_session, request_id_of
from embodied_dataops.canonical.contract import canonical_keys
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import (
    Episode,
    Modality,
    PipelineRun,
    SourceImport,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.operations.canonical import episode_canonical
from embodied_dataops.operations.contract import ETAG_EPISODE
from embodied_dataops.operations.etag import etag_header
from embodied_dataops.operations.revisions import current_revision
from embodied_dataops.quality.http import build_quality_response
from embodied_dataops.storage.base import get_storage

router = APIRouter(prefix="/api/v1/episodes", tags=["episodes"])

#: DO-A01 C2: the Agent only accepts these three provenance values.
_AGENT_DATA_ORIGINS = {
    "synthetic": "synthetic",
    "public_dataset": "public_dataset",
}


def _agent_data_origin(source: SourceImport | None) -> str:
    """Map the DataOps source provenance to the DO-A01 ``synthetic|public_dataset|unknown``."""
    if source is None:
        return "unknown"
    return _AGENT_DATA_ORIGINS.get(source.data_origin, "unknown")


def _principal(request: Request) -> Principal:
    return request.state.principal


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _episode_payload(episode: Episode, source: SourceImport | None = None) -> dict[str, Any]:
    return {
        "id": episode.id,
        "source_import_id": episode.source_import_id,
        "source_type": source.source_type if source else None,
        "source_episode_index": episode.source_episode_index,
        "episode_type": episode.episode_type,
        "task_label": episode.task_label,
        "duration_ns": episode.duration_ns,
        "status": episode.status,
        "quality_status": episode.quality_status,
        "canonical_version": episode.canonical_version,
        "canonical_schema_version": episode.canonical_schema_version,
        "canonical_sha256": episode.canonical_sha256,
        "sample_count": episode.sample_count,
        "data_origin": _agent_data_origin(source),
        "profile_id": episode.profile_id,
        "profile_revision": episode.profile_revision,
        "business_revision": current_revision(episode),
        "created_at": _iso(episode.created_at),
    }


def _get_episode(db: Session, episode_id: str) -> Episode:
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Episode not found")
    return episode


@router.get("")
async def list_episodes(
    request: Request,
    db: Session = Depends(get_session),
    source_type: str | None = Query(default=None),
    quality_status: str | None = Query(default=None),
    task_label: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> JSONResponse:
    _principal(request).require("read")
    stmt = select(Episode).order_by(Episode.created_at.desc(), Episode.id.desc())
    if quality_status:
        stmt = stmt.where(Episode.quality_status == quality_status)
    if task_label:
        stmt = stmt.where(Episode.task_label.ilike(f"%{task_label}%"))
    if source_type:
        stmt = stmt.join(SourceImport, SourceImport.id == Episode.source_import_id).where(
            SourceImport.source_type == source_type
        )
    if cursor:
        timestamp, entity_id = decode_cursor(cursor)
        stmt = stmt.where(
            tuple_(Episode.created_at, Episode.id) < tuple_(datetime.fromisoformat(timestamp), entity_id)
        )
    rows = list(db.scalars(stmt.limit(limit + 1)))
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        last = rows[-1]
        next_cursor = encode_cursor(last.created_at, last.id)
    import_ids = {row.source_import_id for row in rows}
    sources = {
        record.id: record
        for record in db.scalars(select(SourceImport).where(SourceImport.id.in_(import_ids)))
    }
    return JSONResponse(
        content={
            "data": [_episode_payload(row, sources.get(row.source_import_id)) for row in rows],
            "next_cursor": next_cursor,
            "request_id": request_id_of(request),
        }
    )


@router.get("/{episode_id}")
async def get_episode(
    request: Request, episode_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    _principal(request).require("read")
    episode = _get_episode(db, episode_id)
    source = db.get(SourceImport, episode.source_import_id)
    modalities = list(db.scalars(select(Modality).where(Modality.episode_id == episode.id)))
    runs = list(
        db.scalars(
            select(PipelineRun)
            .where(PipelineRun.episode_id == episode.id)
            .order_by(PipelineRun.created_at.desc())
            .limit(5)
        )
    )
    revision = current_revision(episode)
    return JSONResponse(
        content={
            "data": {
                "episode": _episode_payload(episode, source),
                "modalities": [
                    {
                        "name": item.name,
                        "kind": item.kind,
                        "dtype": item.dtype,
                        "shape": item.shape_json,
                        "clock_domain": item.clock_domain,
                        "sample_count": item.sample_count,
                        "min_t_ns": item.min_t_ns,
                        "max_t_ns": item.max_t_ns,
                        "source_key": item.source_key,
                        "stream_index": item.stream_index,
                        "native_rate_hz": item.native_rate_hz,
                        "clock_verified": item.clock_verified,
                        "timestamp_provenance": item.timestamp_provenance,
                        "has_native_index": bool(item.native_index_asset_id),
                        "has_aligned_index": bool(item.aligned_index_asset_id),
                    }
                    for item in modalities
                ],
                "pipeline_runs": [{"id": run.id, "state": run.state} for run in runs],
            },
            "request_id": request_id_of(request),
        },
        headers=etag_header(ETAG_EPISODE, revision),
    )


@router.get("/{episode_id}/canonical")
async def get_canonical(
    request: Request, episode_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    """DO-A02 C7: v2 canonical metadata + safe asset descriptions (no raw bytes/URLs)."""
    _principal(request).require("read")
    episode = _get_episode(db, episode_id)
    return JSONResponse(
        content={
            "data": episode_canonical(db, episode),
            "request_id": request_id_of(request),
        }
    )


@router.get("/{episode_id}/samples")
async def get_samples(
    request: Request,
    episode_id: str,
    db: Session = Depends(get_session),
    start_ns: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=100),
) -> JSONResponse:
    _principal(request).require("read")
    episode = _get_episode(db, episode_id)
    keys = canonical_keys(episode.id, episode.canonical_version)
    storage = get_storage()
    if not storage.exists(keys["samples"]):
        raise ApiError(ErrorCode.NOT_FOUND, "Canonical samples are not available for this episode")
    import tempfile

    import pyarrow.parquet as pq

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
        temp_path = handle.name
    try:
        storage.download_to(keys["samples"], __import__("pathlib").Path(temp_path))
        table = pq.read_table(temp_path)
        rows = table.to_pylist()
    finally:
        __import__("pathlib").Path(temp_path).unlink(missing_ok=True)
    filtered = [row for row in rows if row["t_ns"] >= start_ns]
    sliced = filtered[:limit]
    next_cursor = str(sliced[-1]["t_ns"] + 1) if len(filtered) > limit else None
    return JSONResponse(
        content={
            "data": {"samples": sliced, "next_cursor": next_cursor},
            "next_cursor": next_cursor,
            "request_id": request_id_of(request),
        }
    )


def _preview_object_keys(db: Session, episode: Episode) -> tuple[str, str]:
    """Canonical preview video + thumbnail keys for this Episode's schema generation.

    1.0 keeps the object at ``canonical/{id}/v{n}/frames/front.mp4``; 2.0 stores one
    preview per camera at ``canonical/{id}/v2/{n}/preview/{camera}_10hz_640x480.mp4``
    and has no thumbnail until BUILD_PREVIEW writes it.  Resolving the 1.0 key for a
    2.0 Episode made this endpoint answer 409 forever ("Preview is not built yet") even
    though BUILD_PREVIEW had succeeded.
    """
    if str(getattr(episode, "canonical_schema_version", "") or "").startswith("2"):
        from embodied_dataops.canonical.v2.contract import canonical_v2_keys

        keys = canonical_v2_keys(episode.id, episode.canonical_version)
        camera_names = list(
            db.scalars(
                select(Modality.name)
                .where(Modality.episode_id == episode.id, Modality.kind == "camera")
                .order_by(Modality.name)
            )
        )
        camera = "front" if "front" in camera_names else (camera_names[0] if camera_names else "front")
        return keys["preview"].format(camera=camera), keys["thumbnail"]
    keys = canonical_keys(episode.id, episode.canonical_version)
    return keys["video"], keys["thumbnail"]


@router.get("/{episode_id}/preview")
async def get_preview(
    request: Request, episode_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    _principal(request).require("read")
    episode = _get_episode(db, episode_id)
    video_key, thumbnail_key = _preview_object_keys(db, episode)
    storage = get_storage()
    settings = get_settings()
    if not storage.exists(video_key):
        raise ApiError(
            ErrorCode.CONFLICT,
            "Preview is not built yet (BUILD_PREVIEW not completed for this episode)",
            details={"episode_status": episode.status},
        )
    return JSONResponse(
        content={
            "data": {
                "video_url": storage.presign_get(video_key, settings.download_url_ttl_seconds),
                "thumbnail_url": (
                    storage.presign_get(thumbnail_key, settings.download_url_ttl_seconds)
                    if storage.exists(thumbnail_key)
                    else None
                ),
                "expires_at": None,
            },
            "request_id": request_id_of(request),
        }
    )


@router.get("/{episode_id}/quality")
async def get_quality(
    request: Request,
    episode_id: str,
    db: Session = Depends(get_session),
    run_id: str | None = Query(default=None),
    policy_version: str | None = Query(default=None),
) -> JSONResponse:
    _principal(request).require("read")
    episode = _get_episode(db, episode_id)
    data = build_quality_response(
        db, episode=episode, policy_version=policy_version, run_id=run_id
    )
    return JSONResponse(content={"data": data, "request_id": request_id_of(request)})
