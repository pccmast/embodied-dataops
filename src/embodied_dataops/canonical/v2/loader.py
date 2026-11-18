"""Tolerant reader for stored ``canonical-episode/2.0`` objects.

Used by the quality stage, the v2 exporter and tests.  Absent *optional* streams
(wrist camera, pointcloud summary) are reported as ``None`` instead of being
fabricated; a missing required object raises an explicit API error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from embodied_dataops.canonical.v2.contract import CanonicalMetadataV2, canonical_v2_keys
from embodied_dataops.canonical.v2.parquet import read_parquet_bytes
from embodied_dataops.errors import ApiError, ErrorCode


@dataclass
class LoadedCanonicalV2:
    """Everything stored for one canonical v2 version that could be read."""

    metadata: CanonicalMetadataV2
    keys: dict[str, str]
    native_cameras: dict[str, pa.Table] = field(default_factory=dict)
    native_state: pa.Table | None = None
    native_action: pa.Table | None = None
    aligned_samples: pa.Table | None = None
    mapping_report: dict[str, Any] | None = None
    pointcloud_summary: pa.Table | None = None
    missing: list[str] = field(default_factory=list)


def load_canonical_v2(
    storage: Any,
    *,
    episode_id: str,
    canonical_version: int,
) -> LoadedCanonicalV2:
    """Read the stored canonical v2 objects through the object-store abstraction."""
    keys = canonical_v2_keys(episode_id, canonical_version)
    if not storage.exists(keys["metadata"]):
        raise ApiError(
            ErrorCode.OBJECT_NOT_FOUND,
            "canonical-episode/2.0 metadata.json is missing",
            details={"episode_id": episode_id, "canonical_version": int(canonical_version)},
        )
    try:
        metadata = CanonicalMetadataV2.model_validate_json(storage.get_bytes(keys["metadata"]))
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "canonical-episode/2.0 metadata.json is not a valid v2 document",
            details={"episode_id": episode_id, "reason": f"{type(exc).__name__}"},
        ) from exc
    loaded = LoadedCanonicalV2(metadata=metadata, keys=keys)
    camera_names = [modality.name for modality in metadata.modalities if modality.kind == "camera"]
    for name in camera_names:
        key = keys["native_camera"].format(camera=name)
        if storage.exists(key):
            loaded.native_cameras[name] = read_parquet_bytes(storage.get_bytes(key))
        else:
            loaded.missing.append(f"native/cameras/{name}/timestamps.parquet")
    for name in ("state", "action"):
        key = keys["native_" + name]
        if storage.exists(key):
            table = read_parquet_bytes(storage.get_bytes(key))
            setattr(loaded, "native_" + name, table)
        else:
            loaded.missing.append(f"native/{name}.parquet")
    if storage.exists(keys["aligned_samples"]):
        loaded.aligned_samples = read_parquet_bytes(storage.get_bytes(keys["aligned_samples"]))
    else:
        loaded.missing.append("aligned/10hz/samples.parquet")
    if storage.exists(keys["mapping_report"]):
        import json

        loaded.mapping_report = json.loads(storage.get_bytes(keys["mapping_report"]).decode("utf-8"))
    else:
        loaded.missing.append("mapping_report.json")
    if storage.exists(keys["pointcloud_summary"]):
        loaded.pointcloud_summary = read_parquet_bytes(storage.get_bytes(keys["pointcloud_summary"]))
    return loaded


__all__ = ["LoadedCanonicalV2", "load_canonical_v2"]
