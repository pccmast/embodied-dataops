"""``dataset-manifest/2.0`` builder (V2 SPEC 2.4).

The V2 manifest is an immutable, canonical-JSON description of one DatasetVersion
exported in ``lerobot-v3`` format.  Compared with ``dataset-manifest/1.0`` it
explicitly freezes the profile binding, the native index asset SHA-256 values,
the aligned view, the QC policy, the annotation revision and the camera
transforms, so a later rebuild can prove exactly which canonical objects were
exported.

Episodes are always sorted by ``episode_id`` so the same membership produces the
same bytes regardless of caller order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.datasets.manifest import utc_timestamp

MANIFEST_V2_SCHEMA_VERSION = "dataset-manifest/2.0"
PIPELINE_VERSION_V2 = "2.0.0"
EXPORT_FORMAT = "lerobot-v3"
QC_POLICY_VERSION_V2 = "qc-policy/2.0"

EPISODE_V2_FIELDS: tuple[str, ...] = (
    "episode_id",
    "canonical_sha256",
    "source_sha256",
    "quality_report_id",
    "annotation_set_id",
    "annotation_revision_id",
    "profile_id",
    "profile_revision",
    "profile_config_sha256",
    "camera_keys",
    "aligned_fps",
    "state_dim",
    "action_dim",
    "action_semantics",
    "native_indexes",
    "aligned_view_sha256",
)


def episode_entry_v2(episode: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in EPISODE_V2_FIELDS if name not in episode]
    if missing:
        raise ValueError(f"dataset-manifest/2.0 episode is missing fields: {missing}")
    native_indexes = episode["native_indexes"]
    if not isinstance(native_indexes, Mapping) or not native_indexes:
        raise ValueError("dataset-manifest/2.0 episode requires native_indexes {name: sha256}")
    for name, digest in native_indexes.items():
        if len(str(digest)) != 64:
            raise ValueError(f"native index {name!r} must be a 64-char sha256")
    return {
        "episode_id": str(episode["episode_id"]),
        "canonical_sha256": str(episode["canonical_sha256"]),
        "source_sha256": str(episode["source_sha256"]),
        "quality_report_id": str(episode["quality_report_id"]),
        "annotation_set_id": str(episode["annotation_set_id"]),
        "annotation_revision_id": (
            None if episode["annotation_revision_id"] is None else str(episode["annotation_revision_id"])
        ),
        "profile_id": str(episode["profile_id"]),
        "profile_revision": int(episode["profile_revision"]),
        "profile_config_sha256": str(episode["profile_config_sha256"]),
        "camera_keys": sorted(str(key) for key in episode["camera_keys"]),
        "aligned_fps": int(episode["aligned_fps"]),
        "state_dim": int(episode["state_dim"]),
        "action_dim": int(episode["action_dim"]),
        # Additive: a 2.0-era entry has no action_names and keeps the old bytes.
        "action_names": [str(name) for name in (episode.get("action_names") or [])],
        "action_semantics": str(episode["action_semantics"]),
        "native_indexes": {str(name): str(value) for name, value in sorted(native_indexes.items())},
        "aligned_view_sha256": str(episode["aligned_view_sha256"]),
    }


def build_manifest_v2(
    *,
    dataset_slug: str,
    version: int,
    episodes: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    camera_transforms: Sequence[Mapping[str, Any]] = (),
    qc_policy_version: str = QC_POLICY_VERSION_V2,
    pipeline_version: str = PIPELINE_VERSION_V2,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not dataset_slug or not str(dataset_slug).strip():
        raise ValueError("dataset_slug must not be empty")
    if int(version) < 1:
        raise ValueError("dataset version must be >= 1")
    if not episodes:
        raise ValueError("a dataset version must contain at least one episode")
    entries = [episode_entry_v2(episode) for episode in episodes]
    entries.sort(key=lambda item: item["episode_id"])
    if len({entry["episode_id"] for entry in entries}) != len(entries):
        raise ValueError("dataset version membership contains duplicate episode_id values")
    # A single Profile revision and camera/action/fps combination per version.
    for field_name in ("profile_id", "profile_revision", "profile_config_sha256", "aligned_fps"):
        values = {entry[field_name] for entry in entries}
        if len(values) != 1:
            raise ValueError(f"mixed {field_name} in one dataset-manifest/2.0")
    annotation_revision_ids = {entry["annotation_revision_id"] for entry in entries}
    from embodied_dataops.datasets.publish_policy import (  # noqa: PLC0415
        EXPORT_KIND_RESEARCH,
        EXPORT_KIND_ROBOT,
        training_ready_allowed,
    )

    publish_level = str(profile.get("publish_level") or "")
    # Derived here, never accepted from the caller: a FORMAT_VERIFIED revision stays a
    # research snapshot forever, and re-running the export cannot relabel it.
    training_ready = training_ready_allowed(publish_level)
    return {
        "schema_version": MANIFEST_V2_SCHEMA_VERSION,
        "dataset_slug": str(dataset_slug),
        "version": int(version),
        "export_format": EXPORT_FORMAT,
        "episodes": entries,
        "profile": {
            "profile_id": str(profile["profile_id"]),
            "profile_revision": int(profile["profile_revision"]),
            "profile_config_sha256": str(profile["profile_config_sha256"]),
            "authority": str(profile.get("authority") or ""),
            # Frozen so a later publish cannot upgrade the claim: a FORMAT_VERIFIED
            # revision stays a research snapshot forever (SPEC_V2.1 §4.2).
            "publish_level": publish_level,
            "training_ready": training_ready,
            "export_kind": EXPORT_KIND_ROBOT if training_ready else EXPORT_KIND_RESEARCH,
        },
        "camera_transforms": [dict(item) for item in camera_transforms],
        "annotation_revision_id": (
            next(iter(annotation_revision_ids)) if len(annotation_revision_ids) == 1 else None
        ),
        "qc_policy_version": str(qc_policy_version),
        "pipeline_version": str(pipeline_version),
        "created_at": created_at or utc_timestamp(),
    }


def manifest_v2_bytes(manifest: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(manifest))


def manifest_v2_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(manifest_v2_bytes(manifest)).hexdigest()


def parse_manifest_v2(data: bytes | str) -> dict[str, Any]:
    """Parse stored dataset-manifest/2.0 bytes and reject non-canonical encodings."""
    import json

    raw = data.encode("utf-8") if isinstance(data, str) else data
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a JSON object")
    if payload.get("schema_version") != MANIFEST_V2_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version: {payload.get('schema_version')!r}"
        )
    if manifest_v2_bytes(payload) != raw:
        raise ValueError("dataset manifest bytes are not canonical JSON")
    return payload


__all__ = [
    "EPISODE_V2_FIELDS",
    "MANIFEST_V2_SCHEMA_VERSION",
    "build_manifest_v2",
    "episode_entry_v2",
    "manifest_v2_bytes",
    "manifest_v2_sha256",
    "parse_manifest_v2",
]
