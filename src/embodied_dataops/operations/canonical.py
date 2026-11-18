"""GET /episodes/{id}/canonical — v2 metadata + safe asset descriptions (C7).

The Agent may read *normalized metadata* and safe asset descriptions but never the
raw object bytes, presigned URLs or internal object keys.  V1 episodes remain
readable through their ``canonical-episode/1.0`` metadata.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_keys
from embodied_dataops.db.models import Asset, Episode, Modality
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.storage.base import StorageBackend, get_storage

logger = logging.getLogger(__name__)


def _metadata_key(episode: Episode) -> tuple[str, str]:
    version = int(episode.canonical_version or 1)
    schema_version = episode.canonical_schema_version or "1.0"
    if str(schema_version).startswith("2"):
        return f"canonical/{episode.id}/v2/{version}/metadata.json", schema_version
    return canonical_keys(episode.id, version)["metadata"], schema_version


def _load_metadata(storage: StorageBackend, key: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(storage.get_bytes(key).decode("utf-8"))
    except Exception:  # noqa: BLE001 - missing object is an explicit 404
        return None
    return payload if isinstance(payload, dict) else None


def _safe_asset(asset: Asset, *, role: str) -> dict[str, Any]:
    return {
        "asset_id": asset.id,
        "role": role,
        "logical_name": asset.logical_name,
        "media_type": asset.media_type,
        "byte_size": int(asset.byte_size),
        "sha256": asset.sha256,
        "origin": asset.origin,
        "created_at": asset.created_at.isoformat() if asset.created_at else None,
    }


def episode_canonical(
    session: Session, episode: Episode, *, storage: StorageBackend | None = None
) -> dict[str, Any]:
    backend = storage or get_storage()
    key, schema_version = _metadata_key(episode)
    metadata = _load_metadata(backend, key)
    if metadata is None or metadata.get("episode_id") != episode.id:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "Canonical metadata is not available for this episode",
            details={
                "episode_id": episode.id,
                "canonical_version": episode.canonical_version,
                "canonical_schema_version": schema_version,
            },
        )
    def _major(value: object) -> str:
        return str(value or "").split("/")[-1].split(".")[0]

    if metadata.get("schema_version") and _major(metadata["schema_version"]) != _major(schema_version):
        raise ApiError(
            ErrorCode.INVALID_STATE,
            "Canonical metadata schema version does not match the episode row",
            details={"episode_id": episode.id, "expected": schema_version, "stored": metadata.get("schema_version")},
        )

    assets: dict[str, dict[str, Any]] = {}
    modality_rows = list(session.scalars(select(Modality).where(Modality.episode_id == episode.id)))
    for modality in modality_rows:
        if modality.native_index_asset_id:
            asset = session.get(Asset, modality.native_index_asset_id)
            if asset is not None:
                assets[asset.id] = _safe_asset(asset, role=f"modality:{modality.name}:native")
        if modality.aligned_index_asset_id:
            asset = session.get(Asset, modality.aligned_index_asset_id)
            if asset is not None:
                assets[asset.id] = _safe_asset(asset, role=f"modality:{modality.name}:aligned")

    if _major(schema_version) != "2":
        keys = canonical_keys(episode.id, int(episode.canonical_version or 1))
        roles = {
            keys["metadata"]: "canonical:metadata",
            keys["samples"]: "canonical:samples",
            keys["video"]: "canonical:video",
            keys["thumbnail"]: "canonical:thumbnail",
        }
        for asset in session.scalars(select(Asset).where(Asset.object_key.in_(list(roles)))):
            assets[asset.id] = _safe_asset(asset, role=roles[asset.object_key])

    return {
        "episode_id": episode.id,
        "canonical_version": int(episode.canonical_version or 1),
        "canonical_schema_version": schema_version,
        "canonical_sha256": episode.canonical_sha256,
        "profile_id": episode.profile_id,
        "profile_revision": episode.profile_revision,
        "metadata": metadata,
        "assets": sorted(assets.values(), key=lambda item: (item["role"], item["logical_name"])),
    }


__all__ = ["episode_canonical"]
