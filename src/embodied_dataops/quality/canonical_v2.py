"""Tolerant reader for ``canonical-episode/2.0`` objects (V2 SPEC §2.3).

The QUALITY_CHECK v2 handler must never invent data: every canonical object is
read defensively so a missing optional stream (wrist camera, point-cloud summary)
only makes the corresponding rule *unavailable*, while a missing mandatory
object (``metadata.json``, aligned view) is reported as an explicit gap.

Layout (fixed by SPEC §2.3 and confirmed with D2-M1)::

    canonical/{episode_id}/v2/{canonical_version}/
      metadata.json
      native/state.parquet
      native/action.parquet
      native/cameras/<name>/timestamps.parquet
      native/pointcloud/summary.parquet        # optional
      aligned/10hz/samples.parquet

Column names are exactly the SPEC ones.  ``values`` lists are kept as
``float32`` projections from the canonical file (never re-zero-filled); a row
with ``valid=False`` is preserved and is never treated as a sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.errors import ApiError
from embodied_dataops.quality.rules import VideoInfo
from embodied_dataops.quality.v2_rules import (
    CANONICAL_V2_SCHEMA_VERSION,
    AlignedView,
    ClockDomainInfo,
    NativeStream,
    PointCloudSummary,
    ProfileBinding,
    SemanticQcInput,
)
from embodied_dataops.storage.base import StorageBackend

#: Camera names always looked for when metadata does not enumerate modalities.
DEFAULT_CAMERA_NAMES: tuple[str, ...] = ("front",)


# ---------------------------------------------------------------------------
# object layout
# ---------------------------------------------------------------------------


def canonical_v2_prefix(episode_id: str, canonical_version: int) -> str:
    return f"canonical/{episode_id}/v2/{int(canonical_version)}"


def canonical_v2_camera_key(episode_id: str, canonical_version: int, camera_name: str) -> str:
    return f"{canonical_v2_prefix(episode_id, canonical_version)}/native/cameras/{camera_name}/timestamps.parquet"


def canonical_v2_keys(episode_id: str, canonical_version: int) -> dict[str, str]:
    """Deterministic object keys of one canonical-episode/2.0 version."""
    prefix = canonical_v2_prefix(episode_id, canonical_version)
    return {
        "prefix": prefix,
        "metadata": f"{prefix}/metadata.json",
        "native_state": f"{prefix}/native/state.parquet",
        "native_action": f"{prefix}/native/action.parquet",
        "aligned": f"{prefix}/aligned/10hz/samples.parquet",
        "pointcloud_summary": f"{prefix}/native/pointcloud/summary.parquet",
        "mapping_report": f"{prefix}/mapping_report.json",
    }


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


@dataclass
class CanonicalV2Snapshot:
    """Objects read for one canonical version (missing objects are explicit)."""

    episode_id: str
    canonical_version: int
    keys: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    native_state: pa.Table | None = None
    native_action: pa.Table | None = None
    native_cameras: dict[str, pa.Table] = field(default_factory=dict)
    aligned: pa.Table | None = None
    pointcloud_summary: pa.Table | None = None
    missing_objects: list[str] = field(default_factory=list)
    read_errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def metadata_present(self) -> bool:
        return bool(self.metadata)


def _record_missing(snapshot: CanonicalV2Snapshot, name: str) -> None:
    if name not in snapshot.missing_objects:
        snapshot.missing_objects.append(name)


def _read_json_object(storage: StorageBackend, key: str) -> dict[str, Any] | None:
    try:
        if not storage.exists(key):
            return None
        payload = json.loads(storage.get_bytes(key).decode("utf-8"))
    except (ApiError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_parquet_object(
    storage: StorageBackend, workspace: Path, key: str, filename: str
) -> pa.Table | None:
    try:
        if not storage.exists(key):
            return None
    except ApiError:
        return None
    target = Path(workspace) / f"canonical-v2-{filename}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        storage.download_to(key, target)
        return pq.read_table(target)
    except (ApiError, OSError, pa.ArrowException):
        return None


def _camera_names(metadata: dict[str, Any]) -> list[str]:
    names: list[str] = []
    raw = metadata.get("camera_names")
    if isinstance(raw, list):
        names.extend(str(name) for name in raw if str(name))
    modalities = metadata.get("modalities")
    if isinstance(modalities, list):
        for item in modalities:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").lower()
            name = str(item.get("name") or "")
            if name and kind in {"camera", "video", "image"} and name not in names:
                names.append(name)
    if not names:
        names.extend(DEFAULT_CAMERA_NAMES)
    return names


def load_canonical_v2(
    storage: StorageBackend,
    workspace: Path,
    *,
    episode_id: str,
    canonical_version: int,
) -> CanonicalV2Snapshot:
    """Read every canonical v2 object, tolerating absent optional streams."""
    keys = canonical_v2_keys(episode_id, canonical_version)
    snapshot = CanonicalV2Snapshot(
        episode_id=episode_id, canonical_version=int(canonical_version), keys=keys
    )
    metadata = _read_json_object(storage, keys["metadata"])
    if metadata is None:
        _record_missing(snapshot, "metadata")
    else:
        snapshot.metadata = metadata

    for name, key in (("native_state", keys["native_state"]), ("native_action", keys["native_action"])):
        table = _read_parquet_object(storage, workspace, key, name)
        if table is None:
            _record_missing(snapshot, name)
        setattr(snapshot, name, table)
    aligned = _read_parquet_object(storage, workspace, keys["aligned"], "aligned")
    if aligned is None:
        _record_missing(snapshot, "aligned")
    snapshot.aligned = aligned
    pointcloud = _read_parquet_object(storage, workspace, keys["pointcloud_summary"], "pointcloud-summary")
    if pointcloud is None:
        _record_missing(snapshot, "pointcloud_summary")
    snapshot.pointcloud_summary = pointcloud

    for camera_name in _camera_names(snapshot.metadata):
        key = canonical_v2_camera_key(episode_id, canonical_version, camera_name)
        table = _read_parquet_object(storage, workspace, key, f"camera-{camera_name}")
        if table is None:
            _record_missing(snapshot, f"camera:{camera_name}")
            continue
        snapshot.native_cameras[camera_name] = table
    return snapshot


# ---------------------------------------------------------------------------
# column helpers
# ---------------------------------------------------------------------------


def _has_column(table: pa.Table, name: str) -> bool:
    return table is not None and name in table.column_names


def _int_list(table: pa.Table, name: str) -> list[int]:
    if not _has_column(table, name):
        return []
    values = table.column(name).to_pylist()
    return [int(value) if value is not None else 0 for value in values]


def _bool_list(table: pa.Table, name: str) -> list[bool]:
    if not _has_column(table, name):
        return []
    return [bool(value) if value is not None else False for value in table.column(name).to_pylist()]


def _rows(table: pa.Table, name: str) -> list[list[float] | None]:
    if not _has_column(table, name):
        return []
    rows: list[list[float] | None] = []
    for row in table.column(name).to_pylist():
        if row is None:
            rows.append(None)
        else:
            rows.append([float(value) for value in row])
    return rows


def _reason_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _pick(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _as_str(value: Any) -> str:
    return str(value) if value is not None else ""


# ---------------------------------------------------------------------------
# table -> rule inputs
# ---------------------------------------------------------------------------


def modality_index(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    modalities = metadata.get("modalities")
    if isinstance(modalities, list):
        for item in modalities:
            if isinstance(item, dict) and item.get("name"):
                index[str(item["name"])] = item
    return index


def clock_domain_infos(metadata: dict[str, Any]) -> dict[str, ClockDomainInfo]:
    """Per-stream clock-domain facts declared in ``metadata.json``."""
    infos: dict[str, ClockDomainInfo] = {}
    for name, item in modality_index(metadata).items():
        domain = _as_str(item.get("source_clock_domain") or item.get("clock_domain"))
        infos[name] = ClockDomainInfo(
            stream=name,
            clock_domain=domain,
            timestamp_provenance=_as_str(item.get("timestamp_provenance")),
            clock_verified=item.get("clock_verified") is True,
        )
    raw = metadata.get("clock_domains")
    if isinstance(raw, dict):
        for name, value in raw.items():
            name = str(name)
            if name in infos and infos[name].clock_domain:
                continue
            if isinstance(value, dict):
                domain = _as_str(value.get("clock_domain") or value.get("domain"))
                provenance = _as_str(value.get("timestamp_provenance") or value.get("provenance"))
                verified = value.get("clock_verified") is True
            else:
                domain = _as_str(value)
                provenance = ""
                verified = False
            if domain:
                previous = infos.get(name)
                infos[name] = ClockDomainInfo(
                    stream=name,
                    clock_domain=domain,
                    timestamp_provenance=provenance or (previous.timestamp_provenance if previous else ""),
                    clock_verified=verified or (previous.clock_verified if previous else False),
                )
    return infos


def native_stream_from_table(
    name: str,
    kind: str,
    table: pa.Table | None,
    *,
    clock_domain: str = "",
    timestamp_provenance: str = "",
    clock_verified: bool = False,
    value_column: str = "values",
) -> NativeStream | None:
    if table is None:
        return None
    stream = NativeStream(
        name=name,
        kind=kind,
        clock_domain=clock_domain,
        timestamp_provenance=timestamp_provenance,
        clock_verified=clock_verified,
        source_t_ns=_int_list(table, "source_t_ns"),
        episode_t_ns=_int_list(table, "episode_t_ns"),
        valid=_bool_list(table, "valid"),
    )
    if kind in {"state", "action"}:
        rows = _rows(table, value_column)
        if not rows and value_column != name:
            rows = _rows(table, name)
        stream.values = rows
    return stream


def aligned_view_from_table(
    table: pa.Table | None, *, clock_verified: bool = False
) -> AlignedView | None:
    if table is None:
        return None
    view = AlignedView(
        t_ns=_int_list(table, "t_ns"),
        state=_rows(table, "state"),
        action=_rows(table, "action"),
        state_source_t_ns=_int_list(table, "state_source_t_ns"),
        action_source_t_ns=_int_list(table, "action_source_t_ns"),
        valid=_bool_list(table, "valid"),
        clock_verified=clock_verified,
    )
    if _has_column(table, "invalid_reason"):
        view.invalid_reason = [_reason_list(value) for value in table.column("invalid_reason").to_pylist()]
    return view


def _summary_from_dict(payload: dict[str, Any]) -> PointCloudSummary:
    errors: list[str] = []
    raw_errors = payload.get("errors") or payload.get("error_codes") or payload.get("diagnostics")
    if isinstance(raw_errors, (list, tuple)):
        errors = [str(item) for item in raw_errors if str(item)]
    elif isinstance(raw_errors, str) and raw_errors:
        errors = [raw_errors]
    fields = payload.get("fields")
    return PointCloudSummary(
        topic=_as_str(payload.get("topic") or payload.get("stream") or "points"),
        timestamp_ns=_as_int(payload.get("source_t_ns") or payload.get("timestamp_ns")),
        status=_as_str(payload.get("status")),
        point_count=_as_int(payload.get("point_count")),
        sampled_point_count=_as_int(payload.get("sampled_point_count")),
        nonfinite_sample_count=_as_int(payload.get("nonfinite_sample_count")),
        fields=list(fields) if isinstance(fields, list) else [],
        data_length=_as_int(payload.get("data_length")),
        row_step=_as_int(payload.get("row_step")),
        point_step=_as_int(payload.get("point_step")),
        endianness=_as_str(payload.get("endianness")),
        errors=errors,
    )


def pointcloud_summaries(
    metadata: dict[str, Any], table: pa.Table | None
) -> list[PointCloudSummary]:
    """Bounded PointCloud2 summaries/diagnostics (absent section => empty list)."""
    summaries: list[PointCloudSummary] = []
    if table is not None:
        for row in table.to_pylist():
            if isinstance(row, dict):
                summaries.append(_summary_from_dict(row))
    raw = metadata.get("pointcloud")
    if isinstance(raw, dict):
        summaries.append(_summary_from_dict(raw))
    elif isinstance(raw, list):
        summaries.extend(_summary_from_dict(item) for item in raw if isinstance(item, dict))
    diagnostics = metadata.get("diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            code = _as_str(diagnostic.get("code"))
            if not code.startswith("POINTCLOUD_"):
                continue
            details = diagnostic.get("details")
            payload = dict(details) if isinstance(details, dict) else {}
            payload.setdefault("status", "CORRUPT" if code == "POINTCLOUD_CORRUPT" else code)
            payload.setdefault("errors", [code])
            summaries.append(_summary_from_dict(payload))
    # de-duplicate identical topic+status diagnostics while preserving order
    unique: list[PointCloudSummary] = []
    seen: set[tuple[str, str, int | None]] = set()
    for summary in summaries:
        key = (summary.topic, summary.status, summary.timestamp_ns)
        if key in seen:
            continue
        seen.add(key)
        unique.append(summary)
    return unique


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _decode_video_info(metadata: dict[str, Any], *, expected: bool) -> VideoInfo:
    diagnostics = metadata.get("diagnostics")
    codes: list[str] = []
    if isinstance(diagnostics, list):
        codes = [
            _as_str(item.get("code"))
            for item in diagnostics
            if isinstance(item, dict) and item.get("code")
        ]
    frame_count = metadata.get("decoded_frame_count")
    decoded = frame_count if isinstance(frame_count, int) else None
    decode_ok: bool | None = None
    error = ""
    if "FILE_DECODE_ERROR" in codes:
        decode_ok = False
        error = "NORMALIZE_EPISODE reported FILE_DECODE_ERROR"
    elif decoded is not None:
        decode_ok = decoded > 0
        if not decode_ok:
            error = "NORMALIZE_EPISODE decoded zero frames"
    return VideoInfo(
        expected=expected,
        decode_ok=decode_ok,
        frame_count=decoded,
        decoded_frames=decoded,
        error=error,
        source="canonical-v2-metadata",
    )


def _terminal_marker(metadata: dict[str, Any]) -> tuple[bool | None, bool | None, dict[str, Any]]:
    expected = metadata.get("expected_terminal_marker")
    expected_bool = expected if isinstance(expected, bool) else None
    found = metadata.get("terminal_marker_found")
    found_bool = found if isinstance(found, bool) else None
    evidence: dict[str, Any] = {
        "source_field": "canonical metadata expected_terminal_marker/terminal_marker_found",
        "expected_terminal_marker": expected_bool,
        "terminal_marker_found": found_bool,
    }
    terminal = metadata.get("terminal_marker")
    if isinstance(terminal, dict):
        evidence["terminal_marker"] = terminal
        present = terminal.get("present")
        if found_bool is None and isinstance(present, bool):
            found_bool = present
    return expected_bool, found_bool, evidence


def _calibration_evidence(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("clock_calibration_evidence", "clock_calibrations", "calibration_records"):
        raw = metadata.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            return [raw]
    return []


def _common_timebase_verified(metadata: dict[str, Any], infos: dict[str, ClockDomainInfo]) -> bool:
    if metadata.get("clock_verified") is True:
        return True
    if not infos:
        return False
    return all(info.clock_verified for info in infos.values())


def build_semantic_qc_input(
    snapshot: CanonicalV2Snapshot,
    *,
    episode_id: str,
    episode_type: str,
    data_origin: str,
    source_type: str = "",
    profile: ProfileBinding | None = None,
    duration_ns: int = 0,
    normalize_diagnostics: list[Any] | None = None,
) -> SemanticQcInput:
    """Turn a canonical v2 snapshot into the pure rule input."""
    metadata = snapshot.metadata
    infos = clock_domain_infos(metadata)
    modalities = modality_index(metadata)
    camera_names = list(snapshot.native_cameras) or _camera_names(metadata)
    native_cameras: dict[str, NativeStream] = {}
    for camera_name in camera_names:
        info = infos.get(camera_name)
        stream = native_stream_from_table(
            camera_name,
            "camera",
            snapshot.native_cameras.get(camera_name),
            clock_domain=info.clock_domain if info else "",
            timestamp_provenance=info.timestamp_provenance if info else "",
            clock_verified=info.clock_verified if info else False,
        )
        if stream is not None:
            native_cameras[camera_name] = stream

    state_info = infos.get("state")
    action_info = infos.get("action")
    native_state = native_stream_from_table(
        "state",
        "state",
        snapshot.native_state,
        clock_domain=state_info.clock_domain if state_info else "",
        timestamp_provenance=state_info.timestamp_provenance if state_info else "",
        clock_verified=state_info.clock_verified if state_info else False,
    )
    native_action = native_stream_from_table(
        "action",
        "action",
        snapshot.native_action,
        clock_domain=action_info.clock_domain if action_info else "",
        timestamp_provenance=action_info.timestamp_provenance if action_info else "",
        clock_verified=action_info.clock_verified if action_info else False,
    )

    clock_verified = _common_timebase_verified(metadata, infos)
    aligned = aligned_view_from_table(snapshot.aligned, clock_verified=clock_verified)

    expected_marker, marker_found, marker_evidence = _terminal_marker(metadata)
    required = metadata.get("required_modalities")
    required_modalities = [str(name) for name in required] if isinstance(required, list) else None
    present: list[str] = sorted(modalities)
    if snapshot.native_state is not None and "state" not in present:
        present.append("state")
    if snapshot.native_action is not None and "action" not in present:
        present.append("action")
    for camera_name in native_cameras:
        if camera_name not in present:
            present.append(camera_name)
    if snapshot.pointcloud_summary is not None and "pointcloud" not in present:
        present.append("pointcloud")

    joint_names_raw = metadata.get("joint_names")
    joint_names = (
        [str(name) for name in joint_names_raw]
        if isinstance(joint_names_raw, list) and joint_names_raw
        else None
    )
    duration = metadata.get("duration_ns")
    duration_ns = int(duration) if isinstance(duration, int) else int(duration_ns)
    normalization_valid = metadata.get("normalization_valid")
    if not isinstance(normalization_valid, bool):
        normalization_valid = True

    extra: dict[str, Any] = {
        "canonical_missing_objects": list(snapshot.missing_objects),
        "canonical_read_errors": [dict(item) for item in snapshot.read_errors],
        "float32_precision": metadata.get("float32_precision"),
    }
    if normalize_diagnostics:
        extra["normalize_diagnostics"] = [
            item.get("code") if isinstance(item, dict) else str(item) for item in normalize_diagnostics
        ]

    return SemanticQcInput(
        episode_id=episode_id,
        episode_type=episode_type,
        source_type=source_type or _as_str(metadata.get("source_type")),
        data_origin=data_origin or _as_str(metadata.get("data_origin")),
        canonical_version=snapshot.canonical_version,
        schema_version=_as_str(metadata.get("schema_version")) or CANONICAL_V2_SCHEMA_VERSION,
        duration_ns=duration_ns,
        normalization_valid=normalization_valid,
        metadata=metadata,
        profile=profile or ProfileBinding(),
        native_state=native_state,
        native_action=native_action,
        native_cameras=native_cameras,
        aligned=aligned,
        clock_domains={
            name: info
            for name, info in infos.items()
            if name in {"state", "action"} or name in camera_names or name in present
        },
        calibration_evidence=_calibration_evidence(metadata),
        expected_terminal_marker=expected_marker,
        terminal_marker_present=marker_found,
        terminal_marker_evidence=marker_evidence,
        required_modalities=required_modalities,
        present_modalities=present,
        pointcloud=pointcloud_summaries(metadata, snapshot.pointcloud_summary),
        joint_names=joint_names,
        video=_decode_video_info(
            metadata, expected=episode_type in {"robot_episode", "ego_video_episode"}
        ),
        extra=extra,
    )


#: Re-exported for tests/consumers (SPEC §2.3 interval).
SAMPLE_INTERVAL = SAMPLE_INTERVAL_NS


__all__ = [
    "CanonicalV2Snapshot",
    "SAMPLE_INTERVAL",
    "aligned_view_from_table",
    "build_semantic_qc_input",
    "canonical_v2_camera_key",
    "canonical_v2_keys",
    "canonical_v2_prefix",
    "clock_domain_infos",
    "load_canonical_v2",
    "modality_index",
    "native_stream_from_table",
    "pointcloud_summaries",
]
