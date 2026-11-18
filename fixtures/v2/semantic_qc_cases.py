"""Deterministic ``qc-policy/2.0`` semantic fixtures (D2-M2, DO2-10..13).

Every fixture is synthetic (``data_origin=synthetic``) and reproducible from the
shared seed ``20261029``; there is no randomness and no network access.  Cases
are built in memory for pure rule tests and can also be written as real
``canonical-episode/2.0`` objects (metadata + native/aligned parquet) for the
handler integration tests.

Boundary fixtures (``lower-1e-5``, ``vmax*1.10``, ``150 ms median``,
``nonfinite ratio 0.10``) are constructed directly in the unit tests so the exact
edge is visible there; this module provides one positive/negative fixture per
mandated case class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.domain.profiles import RobotProfileConfig, load_profiles
from embodied_dataops.quality.canonical_v2 import canonical_v2_keys
from embodied_dataops.quality.rules import VideoInfo
from embodied_dataops.quality.v2_policy import QC_POLICY_V2_VERSION
from embodied_dataops.quality.v2_rules import (
    AlignedView,
    ClockDomainInfo,
    NativeStream,
    PointCloudSummary,
    ProfileBinding,
    SemanticQcInput,
)

SEED = 20261029
ORIGIN_NS = 1_700_000_000_000_000_000
POINT_COUNT = 40
CLOCK_LOG = "mcap_log_time"
CLOCK_STAMP = "sensor_stamp"

#: Profile binding of each case (arm6 default, arm7 for the 7D fixture).
PROFILE_BY_CASE: dict[str, str] = {
    "good_6d": "arm6_joint_position_v1",
    "good_7d": "arm7_joint_position_v1",
    "out_of_range": "arm6_joint_position_v1",
    "velocity_spike": "arm6_joint_position_v1",
    "clock_unverified": "arm6_joint_position_v1",
    "action_state_lag": "arm6_joint_position_v1",
    "terminal_missing": "arm6_joint_position_v1",
    "pointcloud_corrupt": "arm6_joint_position_v1",
    "pointcloud_nonfinite_high": "arm6_joint_position_v1",
    "missing_profile": "arm6_joint_position_v1",
    "unverified_authority": "arm6_joint_position_v1",
    "synthetic_profile_public_data": "arm6_joint_position_v1",
    "dimension_mismatch": "arm6_joint_position_v1",
    "aligned_invalid": "arm6_joint_position_v1",
    "bad_timestamps": "arm6_joint_position_v1",
    "nonfinite": "arm6_joint_position_v1",
}

#: Exact rule codes a case must produce (no more, no fewer).
EXPECTED_RULE_CODES: dict[str, tuple[str, ...]] = {
    "good_6d": (),
    "good_7d": (),
    "out_of_range": ("JOINT_POSITION_OUT_OF_RANGE",),
    "velocity_spike": ("JOINT_VELOCITY_SPIKE",),
    "clock_unverified": ("CLOCK_OFFSET_UNVERIFIED",),
    "action_state_lag": ("ACTION_STATE_LAG",),
    "terminal_missing": ("EPISODE_INCOMPLETE",),
    "pointcloud_corrupt": ("POINTCLOUD_CORRUPT",),
    "pointcloud_nonfinite_high": ("POINTCLOUD_NONFINITE_HIGH",),
    "missing_profile": ("MISSING_ROBOT_PROFILE", "UNVERIFIED_SEMANTICS"),
    "unverified_authority": ("UNVERIFIED_SEMANTICS",),
    "synthetic_profile_public_data": ("UNVERIFIED_SEMANTICS",),
    "dimension_mismatch": ("DIMENSION_MISMATCH",),
    "aligned_invalid": ("SYNC_MISSING_STATE",),
    "bad_timestamps": ("TIMESTAMP_NON_MONOTONIC",),
    "nonfinite": ("NONFINITE_VALUE",),
}

#: Rule codes whose base severity is ERROR (their presence means FAIL).
ERROR_RULE_CODES: frozenset[str] = frozenset(
    {
        "MISSING_ROBOT_PROFILE",
        "JOINT_POSITION_OUT_OF_RANGE",
        "JOINT_VELOCITY_SPIKE",
        "EPISODE_INCOMPLETE",
        "POINTCLOUD_CORRUPT",
        "UNVERIFIED_SEMANTICS",
        "DIMENSION_MISMATCH",
        "SYNC_MISSING_STATE",
        "SYNC_MISSING_FRAME",
        "SYNC_STALE_ACTION",
        "TIMESTAMP_NON_MONOTONIC",
        "NONFINITE_VALUE",
        "FILE_DECODE_ERROR",
        "DURATION_EXCEEDED",
    }
)

#: Expected verdict of each case under ``qc-policy/2.0``.
EXPECTED_VERDICT: dict[str, str] = {}
for _case, _codes in EXPECTED_RULE_CODES.items():
    if not _codes:
        EXPECTED_VERDICT[_case] = "PASS"
    elif any(code in ERROR_RULE_CODES for code in _codes):
        EXPECTED_VERDICT[_case] = "FAIL"
    else:
        EXPECTED_VERDICT[_case] = "WARN"

CASE_NAMES: tuple[str, ...] = tuple(EXPECTED_RULE_CODES)

_PROFILE_CACHE: list[RobotProfileConfig] | None = None


def profiles() -> list[RobotProfileConfig]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = load_profiles()
    return _PROFILE_CACHE


def profile_config(profile_id: str) -> RobotProfileConfig:
    for config in profiles():
        if config.profile_id == profile_id:
            return config
    raise KeyError(f"unknown fixture profile {profile_id!r}")


def profile_binding(
    profile_id: str,
    *,
    authority: str | None = None,
    binding_verified: bool = True,
) -> ProfileBinding:
    config = profile_config(profile_id)
    if authority is not None:
        config = config.model_copy(update={"authority": authority})
    return ProfileBinding(
        profile_id=config.profile_id,
        profile_revision=config.profile_revision,
        authority=config.authority,
        config=config,
        config_sha256=config.config_sha256(),
        mode="SEMANTIC",
        binding_verified=binding_verified,
        missing_fields=[],
    )


def _base_values(dim: int, count: int, *, start: float = 0.05, step: float = 0.001) -> list[list[float]]:
    return [[round(start * (joint + 1) + step * index, 6) for joint in range(dim)] for index in range(count)]


def _episode_times(count: int) -> list[int]:
    return [index * SAMPLE_INTERVAL_NS for index in range(count)]


def _source_times(count: int) -> list[int]:
    return [ORIGIN_NS + index * SAMPLE_INTERVAL_NS for index in range(count)]


def _native_robot_stream(name: str, rows: list[list[float]], *, clock_domain: str) -> NativeStream:
    count = len(rows)
    return NativeStream(
        name=name,
        kind=name,
        clock_domain=clock_domain,
        timestamp_provenance="mcap_log_time",
        clock_verified=clock_domain == CLOCK_LOG,
        source_t_ns=_source_times(count),
        episode_t_ns=_episode_times(count),
        valid=[True] * count,
        values=rows,
    )


def _native_camera(count: int, *, clock_domain: str, source_times: list[int] | None = None) -> NativeStream:
    times = source_times if source_times is not None else _source_times(count)
    return NativeStream(
        name="front",
        kind="camera",
        clock_domain=clock_domain,
        timestamp_provenance="header_stamp" if clock_domain == CLOCK_STAMP else "mcap_log_time",
        clock_verified=clock_domain == CLOCK_LOG,
        source_t_ns=times,
        episode_t_ns=_episode_times(len(times)),
        valid=[True] * len(times),
    )


def _aligned(
    state: list[list[float] | None],
    action: list[list[float] | None],
    *,
    clock_verified: bool,
    action_offset_ns: int = 0,
    invalid_reason: list[list[str]] | None = None,
) -> AlignedView:
    count = len(state)
    times = _episode_times(count)
    reasons = invalid_reason if invalid_reason is not None else [[] for _ in range(count)]
    valid = [state[index] is not None and action[index] is not None for index in range(count)]
    return AlignedView(
        t_ns=times,
        state=state,
        action=action,
        state_source_t_ns=times,
        action_source_t_ns=[value + action_offset_ns for value in times],
        valid=valid,
        invalid_reason=reasons,
        clock_verified=clock_verified,
    )


def _clock_infos(
    *,
    state_domain: str = CLOCK_LOG,
    action_domain: str = CLOCK_LOG,
    camera_domain: str = CLOCK_LOG,
    verified: bool = True,
) -> dict[str, ClockDomainInfo]:
    return {
        "state": ClockDomainInfo("state", state_domain, "mcap_log_time", verified and state_domain == CLOCK_LOG),
        "action": ClockDomainInfo("action", action_domain, "mcap_log_time", verified and action_domain == CLOCK_LOG),
        "front": ClockDomainInfo(
            "front",
            camera_domain,
            "header_stamp" if camera_domain == CLOCK_STAMP else "mcap_log_time",
            verified and camera_domain == CLOCK_LOG,
        ),
    }


def build_case_input(case: str, *, data_origin: str = "synthetic") -> SemanticQcInput:
    """Build the pure :class:`SemanticQcInput` of one fixture case."""
    if case not in PROFILE_BY_CASE:
        raise KeyError(f"unknown semantic QC fixture case {case!r}")
    profile_id = PROFILE_BY_CASE[case]
    config = profile_config(profile_id)
    dim = config.state_dim
    binding = profile_binding(
        profile_id,
        authority="UNVERIFIED" if case == "unverified_authority" else None,
    )
    if case == "missing_profile":
        binding = ProfileBinding(
            profile_id=None,
            profile_revision=None,
            authority="",
            config=None,
            config_sha256=None,
            mode="MISSING",
            binding_verified=False,
            missing_fields=["profile_id", "profile_revision", "robot_profiles row"],
        )
    if case == "synthetic_profile_public_data":
        data_origin = "public_dataset"

    state_rows: list[list[float] | None] = [list(row) for row in _base_values(dim, POINT_COUNT)]
    action_rows: list[list[float] | None] = [list(row) for row in _base_values(dim, POINT_COUNT, step=0.001)]
    state_dim = dim

    if case == "out_of_range":
        # every sample of joint_3 is beyond upper+1e-5 but constant, so only the
        # range rule fires (no velocity spike is manufactured by the fixture)
        for row in state_rows:
            if row is not None:
                row[2] = 3.2  # arm6 upper bound is +3.14
    elif case == "velocity_spike":
        state_rows[10][0] = 0.60
        state_rows[11][0] = 0.60 + 0.67  # dt = 0.1 s -> 6.7 rad/s > 6.0*1.10
    elif case == "dimension_mismatch":
        state_rows = [[value for value in row[:5]] for row in state_rows if row is not None]
        action_rows = [[value for value in row[:5]] for row in action_rows if row is not None]
        state_dim = 5
    elif case == "nonfinite":
        state_rows[5] = [float("nan")] * dim

    invalid_reason: list[list[str]] | None = None
    if case == "aligned_invalid":
        for index in (10, 11, 12):
            state_rows[index] = None
        invalid_reason = [
            ["missing_state"] if index in {10, 11, 12} else [] for index in range(POINT_COUNT)
        ]

    action_offset_ns = -200_000_000 if case == "action_state_lag" else 0
    clock_unverified = case == "clock_unverified"
    state_domain = CLOCK_LOG
    camera_domain = CLOCK_STAMP if clock_unverified else CLOCK_LOG
    clocks = _clock_infos(
        state_domain=state_domain,
        action_domain=state_domain,
        camera_domain=camera_domain,
        verified=not clock_unverified,
    )
    aligned = _aligned(
        state_rows,
        action_rows,
        clock_verified=not clock_unverified,
        action_offset_ns=action_offset_ns,
        invalid_reason=invalid_reason,
    )
    camera_source_times = _source_times(POINT_COUNT)
    if case == "bad_timestamps":
        camera_source_times[20] = camera_source_times[19]

    pointcloud: list[PointCloudSummary] = []
    if case == "pointcloud_corrupt":
        pointcloud = [
            PointCloudSummary(
                topic="/points",
                timestamp_ns=ORIGIN_NS,
                status="CORRUPT",
                point_count=1_000,
                sampled_point_count=1_000,
                nonfinite_sample_count=0,
                fields=[{"name": "x", "offset": 0, "datatype": 7}],
                data_length=12_000,
                row_step=16_000,
                point_step=16,
                endianness="little",
                errors=["row_step < point_step*width (truncated data)"],
            )
        ]
    elif case == "pointcloud_nonfinite_high":
        pointcloud = [
            PointCloudSummary(
                topic="/points",
                timestamp_ns=ORIGIN_NS,
                status="OK",
                point_count=1_000,
                sampled_point_count=1_000,
                nonfinite_sample_count=150,
                endianness="little",
            )
        ]

    expected_marker = True if case == "terminal_missing" else False
    marker_present = False if case == "terminal_missing" else None

    return SemanticQcInput(
        episode_id=f"fixture-{case}",
        episode_type="robot_episode",
        source_type="MCAP_ROS2_PROFILE_V2",
        data_origin=data_origin,
        canonical_version=1,
        policy_version=QC_POLICY_V2_VERSION,
        duration_ns=POINT_COUNT * SAMPLE_INTERVAL_NS,
        normalization_valid=True,
        profile=binding,
        native_state=_native_robot_stream("state", [row for row in state_rows if row is not None], clock_domain=state_domain)
        if state_rows
        else None,
        native_action=_native_robot_stream(
            "action", [row for row in action_rows if row is not None], clock_domain=state_domain
        ),
        native_cameras={"front": _native_camera(POINT_COUNT, clock_domain=camera_domain, source_times=camera_source_times)},
        aligned=aligned,
        clock_domains=clocks,
        calibration_evidence=[],
        expected_terminal_marker=expected_marker,
        terminal_marker_present=marker_present,
        terminal_marker_evidence={
            "source_field": "fixture expected_terminal_marker/terminal_marker_found",
            "expected_terminal_marker": expected_marker,
            "terminal_marker_found": marker_present,
        },
        required_modalities=["front", "state", "action"],
        present_modalities=["front", "state", "action"],
        pointcloud=pointcloud,
        joint_names=list(config.joint_names) if case != "dimension_mismatch" else list(config.joint_names),
        video=VideoInfo(expected=True, decode_ok=True, frame_count=POINT_COUNT, decoded_frames=POINT_COUNT),
        metadata={
            "schema_version": "canonical-episode/2.0",
            "episode_id": f"fixture-{case}",
            "profile_id": binding.profile_id,
            "profile_revision": binding.profile_revision,
            "profile_config_sha256": binding.config_sha256,
            "state_dim": state_dim,
            "action_dim": state_dim,
        },
        extra={"fixture_case": case},
    )


# ---------------------------------------------------------------------------
# canonical-episode/2.0 object writer (for handler integration tests)
# ---------------------------------------------------------------------------


def case_metadata(
    case: str,
    *,
    episode_id: str,
    canonical_version: int = 1,
    pipeline_version: str = "2.0.0",
) -> dict[str, Any]:
    """Deterministic ``metadata.json`` for one fixture case (SPEC §2.3 fields)."""
    quality_input = build_case_input(case)
    binding = quality_input.profile
    config = binding.config
    cameras = list(quality_input.native_cameras)
    dimensions = config.state_dim if config is not None else 6
    modalities: list[dict[str, Any]] = []
    for name, info in quality_input.clock_domains.items():
        kind = "camera" if name in cameras else name
        modalities.append(
            {
                "name": name,
                "kind": kind,
                "dtype": "float32" if kind in {"state", "action"} else "uint8",
                "shape": [dimensions] if kind in {"state", "action"} else [480, 640, 3],
                "unit": "rad" if kind in {"state", "action"} else "pixel",
                "source_clock_domain": info.clock_domain,
                "timestamp_provenance": info.timestamp_provenance,
                "source_t_ns_origin": ORIGIN_NS,
                "native_sample_count": POINT_COUNT,
                "native_asset_sha256": "0" * 64,
                "alignment_method": "nearest_frame" if kind == "camera" else "linear_interpolation",
                "alignment_tolerance_ns": 50_000_000,
                "clock_verified": info.clock_verified,
            }
        )
    pointcloud = None
    if quality_input.pointcloud:
        summary = quality_input.pointcloud[0]
        pointcloud = {
            "topic": summary.topic,
            "status": summary.status,
            "point_count": summary.point_count,
            "sampled_point_count": summary.sampled_point_count,
            "nonfinite_sample_count": summary.nonfinite_sample_count,
            "errors": summary.errors,
        }
    return {
        "schema_version": "canonical-episode/2.0",
        "episode_id": episode_id,
        "episode_type": quality_input.episode_type,
        "data_origin": quality_input.data_origin,
        "source_type": quality_input.source_type,
        "source_asset_sha256": "0" * 64,
        "source_episode_index": 0,
        "duration_ns": quality_input.duration_ns,
        "profile_id": binding.profile_id,
        "profile_revision": binding.profile_revision,
        "profile_config_sha256": binding.config_sha256,
        "pipeline_version": pipeline_version,
        "clock_domains": {
            name: {
                "clock_domain": info.clock_domain,
                "timestamp_provenance": info.timestamp_provenance,
                "clock_verified": info.clock_verified,
            }
            for name, info in quality_input.clock_domains.items()
        },
        "modalities": modalities,
        "aligned_view": {
            "fps": 10,
            "path": "aligned/10hz/samples.parquet",
            "grid_point_count": POINT_COUNT,
            "valid_point_count": (
                quality_input.aligned.valid_point_count() if quality_input.aligned else 0
            ),
        },
        "preview_views": [],
        "camera_names": cameras,
        "joint_names": list(config.joint_names) if config is not None else [],
        "state_dim": config.state_dim if config is not None else None,
        "action_dim": config.action_dim if config is not None else None,
        "required_modalities": ["front", "state", "action"],
        "expected_terminal_marker": quality_input.expected_terminal_marker is True,
        "terminal_marker_found": quality_input.terminal_marker_present,
        "decoded_frame_count": POINT_COUNT,
        "normalization_valid": True,
        "diagnostics": [],
        "pointcloud": pointcloud,
    }


def _robot_table(stream: NativeStream) -> pa.Table:
    count = len(stream.values)
    return pa.table(
        {
            "source_index": pa.array(list(range(count)), type=pa.int64()),
            "source_t_ns": pa.array(stream.source_t_ns[:count], type=pa.int64()),
            "episode_t_ns": pa.array(stream.episode_t_ns[:count], type=pa.int64()),
            "values": pa.array(stream.values[:count], type=pa.list_(pa.float32())),
            "valid": pa.array(stream.valid[:count] if stream.valid else [True] * count, type=pa.bool_()),
        }
    )


def _camera_table(stream: NativeStream) -> pa.Table:
    count = len(stream.source_t_ns)
    return pa.table(
        {
            "source_index": pa.array(list(range(count)), type=pa.int64()),
            "source_t_ns": pa.array(stream.source_t_ns, type=pa.int64()),
            "episode_t_ns": pa.array(stream.episode_t_ns[:count], type=pa.int64()),
            "valid": pa.array(stream.valid[:count] if stream.valid else [True] * count, type=pa.bool_()),
            "source_asset_id": pa.array(["0" * 64] * count, type=pa.string()),
        }
    )


def _aligned_table(view: AlignedView) -> pa.Table:
    count = view.point_count()
    frame_indexes = [
        {"front": index if view.is_valid(index) else None, "wrist": None} for index in range(count)
    ]
    return pa.table(
        {
            "t_ns": pa.array(view.t_ns, type=pa.int64()),
            "frame_indexes": pa.array(
                frame_indexes, type=pa.struct([pa.field("front", pa.int64()), pa.field("wrist", pa.int64())])
            ),
            "state": pa.array(view.state, type=pa.list_(pa.float32())),
            "action": pa.array(view.action, type=pa.list_(pa.float32())),
            "state_source_t_ns": pa.array(view.state_source_t_ns, type=pa.int64()),
            "action_source_t_ns": pa.array(view.action_source_t_ns, type=pa.int64()),
            "valid": pa.array(view.valid, type=pa.bool_()),
            "invalid_reason": pa.array(view.invalid_reason, type=pa.list_(pa.string())),
        }
    )


def _pointcloud_table(summaries: list[PointCloudSummary]) -> pa.Table:
    return pa.table(
        {
            "topic": pa.array([summary.topic for summary in summaries], type=pa.string()),
            "source_t_ns": pa.array([summary.timestamp_ns for summary in summaries], type=pa.int64()),
            "status": pa.array([summary.status for summary in summaries], type=pa.string()),
            "point_count": pa.array([summary.point_count for summary in summaries], type=pa.int64()),
            "sampled_point_count": pa.array(
                [summary.sampled_point_count for summary in summaries], type=pa.int64()
            ),
            "nonfinite_sample_count": pa.array(
                [summary.nonfinite_sample_count for summary in summaries], type=pa.int64()
            ),
            "endianness": pa.array([summary.endianness for summary in summaries], type=pa.string()),
        }
    )


def write_canonical_v2_objects(
    storage,
    workspace: Path,
    *,
    episode_id: str,
    case: str,
    canonical_version: int = 1,
    pipeline_version: str = "2.0.0",
) -> dict[str, str]:
    """Write one fixture case as real canonical-episode/2.0 objects."""
    quality_input = build_case_input(case)
    keys = canonical_v2_keys(episode_id, canonical_version)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    import json

    storage.put_bytes(
        keys["metadata"],
        json.dumps(
            case_metadata(
                case,
                episode_id=episode_id,
                canonical_version=canonical_version,
                pipeline_version=pipeline_version,
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        content_type="application/json",
    )

    def _put(key: str, table: pa.Table, filename: str) -> None:
        path = workspace / filename
        pq.write_table(table, path, compression="snappy")
        storage.put_file(key, path, content_type="application/vnd.apache.parquet")

    if quality_input.native_state is not None:
        _put(keys["native_state"], _robot_table(quality_input.native_state), "state.parquet")
    if quality_input.native_action is not None:
        _put(keys["native_action"], _robot_table(quality_input.native_action), "action.parquet")
    for camera_name, stream in quality_input.native_cameras.items():
        camera_key = f"{keys['prefix']}/native/cameras/{camera_name}/timestamps.parquet"
        _put(camera_key, _camera_table(stream), f"camera-{camera_name}.parquet")
    if quality_input.aligned is not None:
        _put(keys["aligned"], _aligned_table(quality_input.aligned), "aligned.parquet")
    if quality_input.pointcloud:
        _put(keys["pointcloud_summary"], _pointcloud_table(quality_input.pointcloud), "pointcloud.parquet")
    return keys


__all__ = [
    "CASE_NAMES",
    "CLOCK_LOG",
    "CLOCK_STAMP",
    "EXPECTED_RULE_CODES",
    "EXPECTED_VERDICT",
    "ORIGIN_NS",
    "POINT_COUNT",
    "PROFILE_BY_CASE",
    "SEED",
    "build_case_input",
    "case_metadata",
    "profile_binding",
    "profile_config",
    "profiles",
    "write_canonical_v2_objects",
]
