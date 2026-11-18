"""Build serialized ``canonical-episode/2.0`` objects from a parsed v2 source.

This module is pure: it turns one :class:`CanonicalV2Source` into bytes (parquet,
``metadata.json``, ``mapping_report.json``) and the metadata model that describes
them.  The NORMALIZE_EPISODE v2 handler uploads those exact bytes and records
their SHA-256; nothing here touches the database or the object store.

Rules enforced here (SPEC 2.3 / ADR-0007):

* native indices keep the source sample order and timestamps; the float64 ->
  float32 projection error is measured and recorded, and an error above ``1e-5``
  marks the version **not publishable for training** (the source asset keeps the
  original data);
* the aligned 10 Hz view never zero-fills: invalid rows carry ``null`` arrays and
  an explicit ``invalid_reason``;
* only a fully valid, semantically verified episode becomes ``publishable=true``;
* ``metadata.json`` carries every required SPEC 2.3 field, plus the evidence the
  quality stage reads (clock domains, provenance, precision, diagnostics).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import pyarrow as pa

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.canonical.v2.align import AlignmentOutcome, align_source, camera_names_of
from embodied_dataops.canonical.v2.contract import (
    ALIGNED_FPS,
    ALIGNED_INTERVAL_NS,
    CANONICAL_V2_SCHEMA_VERSION,
    FLOAT32_MAX_ABS_ERROR,
    PIPELINE_VERSION_V2,
    VIEW_ALIGNED_TRAINING,
    AlignedViewDescriptor,
    CanonicalMetadataV2,
    ModalityDescriptor,
    PreviewViewDescriptor,
    canonical_v2_keys,
)
from embodied_dataops.canonical.v2.model import (
    CanonicalV2Bundle,
    CanonicalV2Source,
    NativeCameraStream,
    NativeRobotStream,
    PointCloudSummary,
    PreviewArtifact,
)
from embodied_dataops.canonical.v2.parquet import (
    NATIVE_ROBOT_SCHEMA,
    aligned_summary,
    build_aligned_table,
    build_native_camera_table,
    build_native_robot_table,
    parquet_bytes,
    validate_aligned_table,
)

MAPPING_REPORT_SCHEMA_VERSION = "canonical-mapping-report/2.0"

_STRUCTURAL_CODES = frozenset(
    {
        "TIMESTAMP_NON_MONOTONIC",
        "DIMENSION_MISMATCH",
        "NONFINITE_VALUE",
        "SYNC_MISSING_FRAME",
        "SYNC_MISSING_STATE",
        "SYNC_STALE_ACTION",
        "DURATION_EXCEEDED",
        "FILE_DECODE_ERROR",
        "EPISODE_ENUMERATION_FAILED",
        "MAPPING_REQUIRED",
        "INVALID_MAPPING",
        "UNVERIFIED_SEMANTICS",
        "ALIGNED_VIEW_INVALID",
    }
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _merge_diagnostics(*groups: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for item in group or []:
            key = (str(item.get("code")), str(item.get("message")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _is_structural(diagnostics: Sequence[dict[str, Any]]) -> bool:
    return any(
        bool(item.get("structural")) or str(item.get("code")) in _STRUCTURAL_CODES
        for item in diagnostics
    )


def _expected_dims(source: CanonicalV2Source) -> tuple[int | None, int | None]:
    if source.profile is not None:
        return int(source.profile.state_dim), int(source.profile.action_dim)
    state_dim = len(source.state.values[0]) if source.state is not None and source.state.values else None
    action_dim = (
        len(source.action.values[0]) if source.action is not None and source.action.values else None
    )
    return state_dim, action_dim


def _check_camera_contract(source: CanonicalV2Source) -> None:
    names = camera_names_of(source)
    if "front" not in names:
        raise ValueError("canonical-episode/2.0 requires a front camera stream")
    if len(names) != len(set(names)):
        raise ValueError("duplicate camera stream names")
    if source.profile is not None:
        missing = [name for name in source.profile.required_camera_names if name not in names]
        if missing:
            raise ValueError(f"profile requires camera stream(s) {missing} but the source has {names}")
        allowed = set(source.profile.required_camera_names) | set(source.profile.optional_camera_names)
        unexpected = [name for name in names if name not in allowed]
        if unexpected:
            raise ValueError(f"camera stream(s) {unexpected} are not declared by the robot profile")


def _stream_rate_hz(times_ns: Sequence[int]) -> float | None:
    if len(times_ns) < 2:
        return None
    span_ns = int(times_ns[-1]) - int(times_ns[0])
    if span_ns <= 0:
        return None
    return round((len(times_ns) - 1) * 1_000_000_000 / span_ns, 6)


def _camera_modality(
    *,
    camera: NativeCameraStream,
    table_sha: str,
    native_asset_key: str,
    origin_ns: int,
    prealigned: bool,
    image_tolerance_ns: int,
) -> ModalityDescriptor:
    return ModalityDescriptor(
        name=camera.name,
        kind="camera",
        dtype="uint8",
        shape=[int(camera.height or 0), int(camera.width or 0), 3],
        unit="rgb8",
        source_clock_domain=camera.clock_domain,
        timestamp_provenance=camera.timestamp_provenance,
        source_t_ns_origin=int(origin_ns),
        native_sample_count=len(camera.times_ns),
        native_asset_sha256=table_sha,
        alignment_method="source_row_pairing" if prealigned else "nearest_frame_within_tolerance",
        alignment_tolerance_ns=int(image_tolerance_ns),
        clock_verified=False,
        native_rate_hz=None if prealigned else _stream_rate_hz(camera.times_ns),
        source_key=camera.source_key,
        native_asset_key=native_asset_key,
    )


def _robot_modality(
    *,
    stream: NativeRobotStream,
    table_sha: str,
    native_asset_key: str,
    dim: int,
    origin_ns: int,
    prealigned: bool,
    state_tolerance_ns: int,
    action_tolerance_ns: int,
) -> ModalityDescriptor:
    is_state = stream.name == "state"
    return ModalityDescriptor(
        name=stream.name,
        kind="state" if is_state else "action",
        dtype="float32",
        shape=[int(dim)],
        unit=stream.unit,
        source_clock_domain=stream.clock_domain,
        timestamp_provenance=stream.timestamp_provenance,
        source_t_ns_origin=int(origin_ns),
        native_sample_count=len(stream.times_ns),
        native_asset_sha256=table_sha,
        alignment_method=(
            "source_row_pairing"
            if prealigned
            else ("linear_interpolation" if is_state else "zero_order_hold_past")
        ),
        alignment_tolerance_ns=int(state_tolerance_ns if is_state else action_tolerance_ns),
        clock_verified=False,
        native_rate_hz=None if prealigned else _stream_rate_hz(stream.times_ns),
        source_key=stream.source_key,
        native_asset_key=native_asset_key,
    )


def _empty_robot_table() -> pa.Table:
    return pa.Table.from_pylist([], schema=NATIVE_ROBOT_SCHEMA)


def build_canonical_v2(
    source: CanonicalV2Source,
    *,
    canonical_version: int,
    previews: Sequence[PreviewArtifact] = (),
    pointcloud: PointCloudSummary | None = None,
    pipeline_version: str = PIPELINE_VERSION_V2,
    created_at: str = "",
    aligned_fps: int = ALIGNED_FPS,
    interval_ns: int = ALIGNED_INTERVAL_NS,
    image_tolerance_ns: int = 50_000_000,
    state_tolerance_ns: int = 50_000_000,
    action_tolerance_ns: int = 100_000_000,
) -> CanonicalV2Bundle:
    """Serialize one canonical v2 version; bad *data* becomes diagnostics, not an exception."""
    _check_camera_contract(source)
    keys = canonical_v2_keys(source.episode_id, canonical_version)
    prealigned = source.alignment_mode == "prealigned"
    origin_ns = 0 if prealigned else int(source.origin_ns)
    state_dim, action_dim = _expected_dims(source)

    # -- native camera indices ---------------------------------------------
    artifacts: dict[str, bytes] = {}
    native_hashes: dict[str, str] = {}
    modalities: list[ModalityDescriptor] = []
    native_counts: dict[str, int] = {}
    for camera in source.cameras:
        table = build_native_camera_table(
            times_ns=camera.times_ns,
            origin_ns=origin_ns,
            source_asset_id=camera.source_asset_id or source.source_asset_sha256,
        )
        data = parquet_bytes(table)
        artifact_key = f"native_camera:{camera.name}"
        artifacts[artifact_key] = data
        native_hashes[artifact_key] = _sha(data)
        native_counts[f"camera:{camera.name}"] = table.num_rows
        modalities.append(
            _camera_modality(
                camera=camera,
                table_sha=native_hashes[artifact_key],
                native_asset_key=keys["native_camera"].format(camera=camera.name),
                origin_ns=origin_ns,
                prealigned=prealigned,
                image_tolerance_ns=image_tolerance_ns,
            )
        )

    # -- native state/action indices ---------------------------------------
    precision: dict[str, Any] = {}
    native_valid_counts: dict[str, int] = {}
    for name in ("state", "action"):
        stream: NativeRobotStream | None = getattr(source, name)
        expected_dim = state_dim if name == "state" else action_dim
        if stream is None or not stream.times_ns:
            table = _empty_robot_table()
            projection = None
            valid_count = 0
        else:
            table, projection, valid_count = build_native_robot_table(
                times_ns=stream.times_ns,
                values=stream.values,
                origin_ns=origin_ns,
                expected_dim=expected_dim,
            )
        data = parquet_bytes(table)
        artifact_key = f"native_{name}"
        artifacts[artifact_key] = data
        native_hashes[artifact_key] = _sha(data)
        native_counts[name] = table.num_rows
        native_valid_counts[name] = valid_count
        if projection is not None:
            precision[name] = projection.as_dict()
        else:
            precision[name] = {
                "max_abs_error": 0.0,
                "max_abs_error_threshold": FLOAT32_MAX_ABS_ERROR,
                "blocked": False,
                "nonfinite_count": 0,
                "note": "no source samples",
            }
        modality_stream = stream
        if modality_stream is None:
            modality_stream = NativeRobotStream(
                name=name,
                source_key="/joint_states" if name == "state" else "/policy/action",
                clock_domain=source.clock_domains.get(name, "unknown"),
                timestamp_provenance="absent_required_stream",
                times_ns=[],
                values=[],
                unit=source.state_unit if name == "state" else source.action_unit,
            )
        modalities.append(
            _robot_modality(
                stream=modality_stream,
                table_sha=native_hashes[artifact_key],
                native_asset_key=keys["native_" + name],
                dim=int(expected_dim or 0),
                origin_ns=origin_ns,
                prealigned=prealigned,
                state_tolerance_ns=state_tolerance_ns,
                action_tolerance_ns=action_tolerance_ns,
            )
        )

    # -- aligned 10 Hz view -------------------------------------------------
    outcome = align_source(
        source,
        expected_state_dim=state_dim,
        expected_action_dim=action_dim,
        interval_ns=interval_ns,
        image_tolerance_ns=image_tolerance_ns,
        state_tolerance_ns=state_tolerance_ns,
        action_tolerance_ns=action_tolerance_ns,
    )
    aligned_table = build_aligned_table(outcome.rows)
    alignment_errors = (
        validate_aligned_table(aligned_table) if aligned_table.num_rows else ["aligned view is empty"]
    )
    aligned_bytes = parquet_bytes(aligned_table)
    artifacts["aligned_samples"] = aligned_bytes
    aligned_sha = _sha(aligned_bytes)
    summary = (
        aligned_summary(aligned_table)
        if aligned_table.num_rows
        else {
            "sample_count": 0,
            "valid_sample_count": 0,
            "invalid_sample_count": 0,
            "invalid_reason_counts": {},
        }
    )

    diagnostics = _merge_diagnostics(source.diagnostics, outcome.diagnostics)
    if alignment_errors:
        diagnostics.append(
            {
                "code": "ALIGNED_VIEW_INVALID",
                "severity": "ERROR",
                "message": "aligned/10hz/samples.parquet violates canonical-episode/2.0",
                "details": {"errors": alignment_errors[:20]},
                "structural": True,
            }
        )

    # -- mapping report -----------------------------------------------------
    mapping_report = _mapping_report(
        source=source,
        outcome=outcome,
        state_dim=state_dim,
        action_dim=action_dim,
        native_counts=native_counts,
        native_valid_counts=native_valid_counts,
        precision=precision,
        interval_ns=interval_ns,
        aligned_fps=aligned_fps,
        image_tolerance_ns=image_tolerance_ns,
        state_tolerance_ns=state_tolerance_ns,
        action_tolerance_ns=action_tolerance_ns,
        diagnostics=diagnostics,
    )
    mapping_bytes = canonical_json_bytes(mapping_report)
    artifacts["mapping_report"] = mapping_bytes
    mapping_sha = _sha(mapping_bytes)

    blocking_reasons = _publish_blockers(
        source=source,
        outcome=outcome,
        diagnostics=diagnostics,
        precision=precision,
        aligned_rows=aligned_table.num_rows,
        alignment_errors=alignment_errors,
    )
    publishable = not blocking_reasons

    # -- metadata -----------------------------------------------------------
    aligned_descriptor = AlignedViewDescriptor(
        view_kind=VIEW_ALIGNED_TRAINING,
        fps=int(aligned_fps),
        interval_ns=int(interval_ns),
        sample_count=summary["sample_count"],
        valid_sample_count=summary["valid_sample_count"],
        invalid_sample_count=summary["invalid_sample_count"],
        asset_key=keys["aligned_samples"],
        asset_sha256=aligned_sha,
        state_dim=int(state_dim or 0),
        action_dim=int(action_dim or 0),
        camera_names=list(source.camera_names),
        alignment_method=(
            "source_row_pairing" if prealigned else "nearest_frame+state_interpolation+action_zoh"
        ),
        prealigned=prealigned,
        source_row_rate_hz=source.source_row_rate_hz,
        coverage_start_ns=outcome.coverage_start_ns,
        coverage_end_ns=outcome.coverage_end_ns,
        image_tolerance_ns=int(image_tolerance_ns),
        state_tolerance_ns=int(state_tolerance_ns),
        action_tolerance_ns=int(action_tolerance_ns),
    )
    clock_domains: dict[str, str] = {}
    for camera in source.cameras:
        clock_domains[f"camera_{camera.name}"] = camera.clock_domain
    if source.state is not None:
        clock_domains["state"] = source.state.clock_domain
    if source.action is not None:
        clock_domains["action"] = source.action.clock_domain
    preview_views = [
        PreviewViewDescriptor(
            camera_name=preview.camera_name,
            fps=int(preview.fps),
            width=int(preview.width),
            height=int(preview.height),
            asset_key=preview.asset_key,
            asset_sha256=preview.asset_sha256,
            selection=preview.selection,
        )
        for preview in previews
    ]
    metadata = CanonicalMetadataV2(
        schema_version=CANONICAL_V2_SCHEMA_VERSION,
        episode_id=source.episode_id,
        source_asset_sha256=source.source_asset_sha256,
        profile_id=source.profile_id,
        profile_revision=source.profile_revision,
        profile_config_sha256=source.profile_config_sha256,
        episode_type=source.episode_type,
        clock_domains=clock_domains,
        modalities=modalities,
        aligned_view=aligned_descriptor,
        preview_views=preview_views,
        pipeline_version=pipeline_version,
        data_origin=source.data_origin,
        task_label=source.task_label,
        source_type=source.source_type,
        source_episode_index=int(source.source_episode_index),
        duration_ns=_duration_ns(outcome, aligned_fps),
        joint_names=list(source.joint_names),
        action_names=list(source.action_names),
        state_dim=int(state_dim or 0),
        action_dim=int(action_dim or 0),
        state_semantics=source.state_semantics,
        action_semantics=source.action_semantics,
        state_unit=source.state_unit,
        action_unit=source.action_unit,
        camera_names=list(source.camera_names),
        required_modalities=list(source.required_modalities) or ["front", "state", "action"],
        diagnostic_only=bool(source.diagnostic_only),
        semantic_authority=bool(source.semantic_authority),
        publishable=publishable,
        expected_terminal_marker=bool(source.expected_terminal_marker),
        terminal_marker_found=source.terminal_marker_found,
        float32_precision=precision,
        diagnostics=diagnostics,
        mapping_report_key=keys["mapping_report"],
        mapping_report_sha256=mapping_sha,
        pointcloud=(
            {
                "asset_key": pointcloud.asset_key,
                "asset_sha256": pointcloud.asset_sha256,
                "point_count": pointcloud.point_count,
                "sampled_point_count": pointcloud.sampled_point_count,
                "nonfinite_sample_count": pointcloud.nonfinite_sample_count,
                "endianness": pointcloud.endianness,
                "source_asset_sha256": pointcloud.source_asset_sha256,
                "bbox": pointcloud.bbox,
            }
            if pointcloud is not None
            else None
        ),
        source=dict(source.source),
        created_at=created_at,
    )
    artifacts["metadata"] = metadata.canonical_json()
    return CanonicalV2Bundle(
        metadata=metadata,
        metadata_bytes=artifacts["metadata"],
        artifacts=artifacts,
        mapping_report=mapping_report,
        alignment_diagnostics=outcome.as_dict(),
        precision=precision,
        publishable=publishable,
        blocking_reasons=blocking_reasons,
    )


def _duration_ns(outcome: AlignmentOutcome, aligned_fps: int) -> int:
    if not outcome.grid_rel_ns:
        return 0
    interval = int(1_000_000_000 / aligned_fps) if aligned_fps else ALIGNED_INTERVAL_NS
    return int(outcome.grid_rel_ns[-1]) + interval


def _publish_blockers(
    *,
    source: CanonicalV2Source,
    outcome: AlignmentOutcome,
    diagnostics: Sequence[dict[str, Any]],
    precision: dict[str, Any],
    aligned_rows: int,
    alignment_errors: Sequence[str],
) -> list[str]:
    reasons: list[str] = []
    if not source.semantic_authority:
        reasons.append("semantic_authority_unverified:UNVERIFIED_SEMANTICS")
    if source.diagnostic_only:
        reasons.append("diagnostic_only")
    if aligned_rows == 0 or alignment_errors:
        reasons.append("aligned_view_invalid")
    elif outcome.valid_sample_count != outcome.sample_count:
        reasons.append("aligned_view_has_invalid_samples")
    if _is_structural(diagnostics):
        reasons.append("structural_diagnostics_present")
    for stream, detail in precision.items():
        if bool(detail.get("blocked")):
            reasons.append(f"float32_projection_error_exceeds_{FLOAT32_MAX_ABS_ERROR}:{stream}")
    if source.expected_terminal_marker and source.terminal_marker_found is False:
        reasons.append("EPISODE_INCOMPLETE:missing_terminal_marker")
    return reasons


def _mapping_report(
    *,
    source: CanonicalV2Source,
    outcome: AlignmentOutcome,
    state_dim: int | None,
    action_dim: int | None,
    native_counts: dict[str, int],
    native_valid_counts: dict[str, int],
    precision: dict[str, Any],
    interval_ns: int,
    aligned_fps: int,
    image_tolerance_ns: int,
    state_tolerance_ns: int,
    action_tolerance_ns: int,
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    prealigned = source.alignment_mode == "prealigned"
    return {
        "schema_version": MAPPING_REPORT_SCHEMA_VERSION,
        "episode_id": source.episode_id,
        "source_type": source.source_type,
        "source_asset_sha256": source.source_asset_sha256,
        "source_episode_index": int(source.source_episode_index),
        "profile": (
            None
            if source.profile is None
            else {
                "profile_id": source.profile_id,
                "profile_revision": source.profile_revision,
                "profile_config_sha256": source.profile_config_sha256,
                "authority": source.profile_authority,
            }
        ),
        "diagnostic_only": bool(source.diagnostic_only),
        "semantic_authority": bool(source.semantic_authority),
        "cameras": [
            {
                "canonical_name": camera.name,
                "source_key": camera.source_key,
                "clock_domain": camera.clock_domain,
                "timestamp_provenance": camera.timestamp_provenance,
                "native_frame_count": len(camera.times_ns),
                "native_rate_hz": None if prealigned else _stream_rate_hz(camera.times_ns),
                "source_asset_id": camera.source_asset_id,
                "width": int(camera.width),
                "height": int(camera.height),
                "encoding": camera.encoding,
            }
            for camera in source.cameras
        ],
        "camera_count": len(source.cameras),
        "state": (
            None
            if source.state is None
            else {
                "source_key": source.state.source_key,
                "dim": int(state_dim or 0),
                "unit": source.state_unit,
                "semantics": source.state_semantics,
                "clock_domain": source.state.clock_domain,
                "timestamp_provenance": source.state.timestamp_provenance,
                "native_sample_count": len(source.state.times_ns),
                "native_rate_hz": None if prealigned else _stream_rate_hz(source.state.times_ns),
            }
        ),
        "action": (
            None
            if source.action is None
            else {
                "source_key": source.action.source_key,
                "dim": int(action_dim or 0),
                "unit": source.action_unit,
                "semantics": source.action_semantics,
                "clock_domain": source.action.clock_domain,
                "timestamp_provenance": source.action.timestamp_provenance,
                "native_sample_count": len(source.action.times_ns),
                "native_rate_hz": None if prealigned else _stream_rate_hz(source.action.times_ns),
            }
        ),
        "alignment": {
            "mode": outcome.mode,
            "prealigned": outcome.prealigned,
            "aligned_fps": int(aligned_fps),
            "interval_ns": int(interval_ns),
            "image_tolerance_ns": int(image_tolerance_ns),
            "state_tolerance_ns": int(state_tolerance_ns),
            "action_tolerance_ns": int(action_tolerance_ns),
            "source_row_rate_hz": source.source_row_rate_hz,
            "native_counts": dict(native_counts),
            "native_valid_counts": dict(native_valid_counts),
            "sample_count": outcome.sample_count,
            "valid_sample_count": outcome.valid_sample_count,
            "invalid_sample_count": outcome.invalid_sample_count,
        },
        "precision": precision,
        "diagnostics": list(diagnostics),
        "source": dict(source.source),
        "notes": [
            "Native indices keep the source sample order and payloads; the aligned view is derived.",
            "Preview videos are regenerable display artifacts, not training truth.",
            "MCAP log_time is receive time, not camera exposure time; clock_verified is false without "
            "external calibration evidence.",
        ],
    }


__all__ = ["MAPPING_REPORT_SCHEMA_VERSION", "build_canonical_v2"]
