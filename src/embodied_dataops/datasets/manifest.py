"""``dataset-manifest/1.0`` builder (SPEC 10.1).

The manifest is the frozen, machine-readable definition of one DatasetVersion.
Its canonical JSON (UTF-8, sorted keys, no extra whitespace) is hashed with
SHA-256 and that hash is stored on the version row; publishing re-verifies it.

Ordering is part of the contract: episodes are always sorted by ``episode_id``
(ascending) so the same membership always produces the same bytes regardless of
the order the caller supplied.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from embodied_dataops import MANIFEST_SCHEMA_VERSION, PIPELINE_VERSION, QC_POLICY_VERSION, TARGET_FPS
from embodied_dataops.canonical.contract import canonical_json_bytes

EXPORT_FORMAT = "lerobot-v3"

#: Every manifest episode entry carries exactly these keys (SPEC 10.1).
EPISODE_MANIFEST_FIELDS: tuple[str, ...] = (
    "episode_id",
    "canonical_sha256",
    "source_sha256",
    "quality_report_id",
    "annotation_set_id",
)

_SHA256_CHARS = set("0123456789abcdef")


def utc_timestamp(value: datetime | None = None) -> str:
    """RFC 3339 UTC with a literal ``Z`` suffix, second precision."""
    moment = value or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    moment = moment.astimezone(UTC).replace(microsecond=0)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_sha256(value: Any, field_name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or set(text) - _SHA256_CHARS:
        raise ValueError(f"dataset manifest field {field_name!r} must be a 64-char hex sha256")
    return text


def episode_entry(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise one episode mapping into the manifest entry shape."""
    missing = [name for name in EPISODE_MANIFEST_FIELDS if name not in episode]
    if missing:
        raise ValueError(f"dataset manifest episode is missing fields: {missing}")
    return {
        "episode_id": str(episode["episode_id"]),
        "canonical_sha256": _require_sha256(episode["canonical_sha256"], "canonical_sha256"),
        "source_sha256": _require_sha256(episode["source_sha256"], "source_sha256"),
        "quality_report_id": str(episode["quality_report_id"]),
        "annotation_set_id": str(episode["annotation_set_id"]),
    }


def ordered_episode_ids(episode_ids: Iterable[str]) -> list[str]:
    """Stable membership order: ascending ``episode_id`` (SPEC 10.1)."""
    ids = [str(value) for value in episode_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset version membership contains duplicate episode_id values")
    return sorted(ids)


def build_manifest(
    *,
    dataset_slug: str,
    version: int,
    episodes: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any] | None = None,
    pipeline_version: str = PIPELINE_VERSION,
    qc_policy_version: str = QC_POLICY_VERSION,
    created_at: str | datetime | None = None,
    target_fps: int = TARGET_FPS,
    export_format: str = EXPORT_FORMAT,
) -> dict[str, Any]:
    """Build the canonical ``dataset-manifest/1.0`` payload.

    ``episodes`` may be given in any order; the manifest is sorted by
    ``episode_id``.  The returned dict is what gets serialised with
    :func:`manifest_bytes` and persisted as an immutable asset.

    ``profile`` freezes the publish level (and the profile revisions behind it)
    inside the hashed manifest, so publishing can refuse a version whose claimed
    export kind outruns what its profiles evidence.
    """
    if not dataset_slug or not str(dataset_slug).strip():
        raise ValueError("dataset_slug must not be empty")
    if int(version) < 1:
        raise ValueError("dataset version must be >= 1")
    if not episodes:
        raise ValueError("a dataset version must contain at least one episode")
    entries = [episode_entry(episode) for episode in episodes]
    entries.sort(key=lambda item: item["episode_id"])
    if len({entry["episode_id"] for entry in entries}) != len(entries):
        raise ValueError("dataset version membership contains duplicate episode_id values")
    timestamp = created_at if isinstance(created_at, str) else utc_timestamp(created_at)
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_slug": str(dataset_slug),
        "version": int(version),
        "export_format": str(export_format),
        "target_fps": int(target_fps),
        "episodes": entries,
        "pipeline_version": str(pipeline_version),
        "qc_policy_version": str(qc_policy_version),
        "created_at": timestamp,
    }
    if profile is not None:
        payload["profile"] = dict(profile)
    return payload


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Canonical JSON bytes: UTF-8, sorted keys, no extra whitespace."""
    return canonical_json_bytes(dict(manifest))


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(manifest_bytes(manifest)).hexdigest()


def manifest_sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_manifest(data: bytes | str) -> dict[str, Any]:
    """Parse manifest bytes and reject non-canonical encodings.

    A stored manifest is expected to be exactly the canonical bytes; accepting a
    looser encoding would make the published hash ambiguous.
    """
    import json

    raw = data.encode("utf-8") if isinstance(data, str) else data
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema_version: {payload.get('schema_version')!r}")
    if manifest_bytes(payload) != raw:
        raise ValueError("dataset manifest bytes are not canonical JSON")
    return payload


def episode_ids_of(manifest: Mapping[str, Any]) -> list[str]:
    return [str(entry["episode_id"]) for entry in manifest.get("episodes", [])]
