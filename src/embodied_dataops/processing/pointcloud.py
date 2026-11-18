"""Bounded ROS 2 ``sensor_msgs/msg/PointCloud2`` parsing (V2 D2-7, SPEC §8).

Scope: only the ``MCAP_ROS2_PROFILE_V2`` ``/points`` topic, only ``x/y/z``
``FLOAT32`` fields (little- or big-endian).  The parser is deliberately bounded:

* a message with more than ``MAX_POINTS`` points is reported as
  ``POINTCLOUD_TOO_LARGE`` **without touching the point payload**;
* statistics (bounding box, non-finite ratio) use a deterministic sample of at
  most ``MAX_SAMPLE_POINTS`` points (``ceil(N / 10000)`` stride), so memory use is
  independent of the cloud size;
* corrupt layouts (truncated bytes, overlap/out-of-bounds fields, missing xyz)
  return a summary carrying a ``POINTCLOUD_CORRUPT`` QC finding instead of
  raising, so a worker records evidence and keeps running;
* genuinely unsupported layouts (non-FLOAT32 xyz, multi-channel fields, unknown
  datatypes in a required position) raise
  ``ApiError(UNSUPPORTED_POINTCLOUD_LAYOUT)`` (HTTP 422) — the platform never
  attempts unbounded interpretation.

Only the summary (counts, bounding box, endianness) is persisted/returned; raw
point coordinates are never exposed to the API or an Agent (SPEC §8.1).
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.orm import Session

from embodied_dataops.db.models import Asset
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.storage.base import StorageBackend, get_storage

POINTCLOUD_TOPIC = "/points"
MAX_POINTS = 1_000_000
MAX_SAMPLE_POINTS = 10_000
NONFINITE_RATIO_THRESHOLD = 0.10
POINTCLOUD_SUMMARY_SCHEMA_VERSION = "pointcloud-summary/2.0"

#: ROS ``sensor_msgs/PointField`` datatype -> (name, byte size).
POINT_FIELD_DATATYPES: dict[int, tuple[str, int]] = {
    1: ("INT8", 1),
    2: ("UINT8", 1),
    3: ("INT16", 2),
    4: ("UINT16", 2),
    5: ("INT32", 4),
    6: ("UINT32", 4),
    7: ("FLOAT32", 4),
    8: ("FLOAT64", 8),
}
FLOAT32_DATATYPE = 7
XYZ_NAMES = ("x", "y", "z")

POINTCLOUD_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("source_t_ns", pa.int64(), nullable=False),
        pa.field("point_count", pa.int64(), nullable=False),
        pa.field("sampled_point_count", pa.int64(), nullable=False),
        pa.field("nonfinite_sample_count", pa.int64(), nullable=False),
        pa.field("min_x", pa.float64(), nullable=True),
        pa.field("max_x", pa.float64(), nullable=True),
        pa.field("min_y", pa.float64(), nullable=True),
        pa.field("max_y", pa.float64(), nullable=True),
        pa.field("min_z", pa.float64(), nullable=True),
        pa.field("max_z", pa.float64(), nullable=True),
        pa.field("endianness", pa.string(), nullable=False),
        pa.field("source_asset_sha256", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class PointCloudField:
    """One ``sensor_msgs/PointField`` entry."""

    name: str
    offset: int
    datatype: int
    count: int = 1

    @property
    def byte_size(self) -> int:
        _, size = POINT_FIELD_DATATYPES[self.datatype]
        return size * max(1, int(self.count))


@dataclass
class PointCloud2Message:
    """The bounded subset of ``sensor_msgs/msg/PointCloud2`` we accept."""

    data: bytes
    width: int
    height: int
    point_step: int
    row_step: int
    fields: Sequence[PointCloudField]
    is_bigendian: bool = False
    source_t_ns: int = 0
    source_asset_sha256: str = ""
    topic: str = POINTCLOUD_TOPIC


@dataclass
class PointCloudSummary:
    """Deterministic, bounded summary of one point cloud message."""

    source_t_ns: int
    point_count: int
    sampled_point_count: int
    nonfinite_sample_count: int
    min_x: float | None
    max_x: float | None
    min_y: float | None
    max_y: float | None
    min_z: float | None
    max_z: float | None
    endianness: str
    source_asset_sha256: str
    valid: bool = True
    error_code: str | None = None
    topic: str = POINTCLOUD_TOPIC
    findings: list[dict[str, Any]] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        return {
            "source_t_ns": int(self.source_t_ns),
            "point_count": int(self.point_count),
            "sampled_point_count": int(self.sampled_point_count),
            "nonfinite_sample_count": int(self.nonfinite_sample_count),
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "endianness": self.endianness,
            "source_asset_sha256": self.source_asset_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.row()
        payload.update(
            {
                "valid": self.valid,
                "error_code": self.error_code,
                "topic": self.topic,
                "findings": list(self.findings),
            }
        )
        return payload


# ---------------------------------------------------------------------------
# findings helpers
# ---------------------------------------------------------------------------
def _finding(
    rule_code: str,
    severity: str,
    *,
    observed: dict[str, Any],
    threshold: dict[str, Any],
    source_t_ns: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rule_code": rule_code,
        "severity": severity,
        "observed": observed,
        "threshold": threshold,
        "affected_time_ranges": [{"start_ns": int(source_t_ns), "end_ns": int(source_t_ns)}],
        "source_stream": POINTCLOUD_TOPIC,
        "source_clock_domain": "log_time",
        "method": "bounded deterministic sampling",
        "limits": {"max_points": MAX_POINTS, "max_sample_points": MAX_SAMPLE_POINTS},
        "uncertainty_note": (
            "Point-cloud statistics are computed from a deterministic sample "
            f"of at most {MAX_SAMPLE_POINTS} points; point_count is the original message count."
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def _unsupported(reason: str, *, observed: dict[str, Any]) -> ApiError:
    return ApiError(
        ErrorCode.UNSUPPORTED_POINTCLOUD_LAYOUT,
        "PointCloud2 layout is outside the supported x/y/z FLOAT32 profile",
        details={"reason": reason, "topic": POINTCLOUD_TOPIC, **observed},
    )


def _base_kwargs(message: PointCloud2Message) -> dict[str, Any]:
    return {
        "source_t_ns": int(message.source_t_ns),
        "endianness": "big" if message.is_bigendian else "little",
        "source_asset_sha256": message.source_asset_sha256,
        "topic": message.topic,
    }


def _corrupt_summary(
    message: PointCloud2Message,
    *,
    reason: str,
    point_count: int,
    observed: dict[str, Any] | None = None,
) -> PointCloudSummary:
    base = _base_kwargs(message)
    observed_payload = {
        "reason": reason,
        "width": int(message.width),
        "height": int(message.height),
        "point_step": int(message.point_step),
        "row_step": int(message.row_step),
        "data_bytes": len(message.data),
        **(observed or {}),
    }
    return PointCloudSummary(
        point_count=int(point_count),
        sampled_point_count=0,
        nonfinite_sample_count=0,
        min_x=None,
        max_x=None,
        min_y=None,
        max_y=None,
        min_z=None,
        max_z=None,
        valid=False,
        error_code=ErrorCode.POINTCLOUD_CORRUPT.value,
        findings=[
            _finding(
                ErrorCode.POINTCLOUD_CORRUPT.value,
                "ERROR",
                observed=observed_payload,
                threshold={"valid_layout": True},
                source_t_ns=int(message.source_t_ns),
                extra={"evidence": {"topic": message.topic, "message": reason}},
            )
        ],
        **base,
    )


def _too_large_summary(message: PointCloud2Message, *, point_count: int) -> PointCloudSummary:
    base = _base_kwargs(message)
    return PointCloudSummary(
        point_count=point_count,
        sampled_point_count=0,
        nonfinite_sample_count=0,
        min_x=None,
        max_x=None,
        min_y=None,
        max_y=None,
        min_z=None,
        max_z=None,
        valid=False,
        error_code=ErrorCode.POINTCLOUD_TOO_LARGE.value,
        findings=[
            _finding(
                ErrorCode.POINTCLOUD_TOO_LARGE.value,
                "ERROR",
                observed={
                    "point_count": point_count,
                    "width": int(message.width),
                    "height": int(message.height),
                },
                threshold={"max_points": MAX_POINTS},
                source_t_ns=int(message.source_t_ns),
                extra={"evidence": {"topic": message.topic, "payload_not_loaded": True}},
            )
        ],
        **base,
    )


# ---------------------------------------------------------------------------
# layout validation
# ---------------------------------------------------------------------------
def _validate_fields(
    fields: Sequence[PointCloudField],
    *,
    point_step: int,
) -> tuple[PointCloudField, PointCloudField, PointCloudField]:
    """Return the x/y/z fields or raise/flag via a typed reason string."""
    names = [str(field.name) for field in fields]
    if len(set(names)) != len(names):
        raise _LayoutProblem("CORRUPT", "duplicate_field_name")
    by_name = {str(field.name): field for field in fields}
    missing = [name for name in XYZ_NAMES if name not in by_name]
    if missing:
        raise _LayoutProblem("CORRUPT", f"missing_xyz_fields:{','.join(missing)}")

    xyz: dict[str, PointCloudField] = {}
    for name in XYZ_NAMES:
        current = by_name[name]
        if current.datatype not in POINT_FIELD_DATATYPES:
            raise _LayoutProblem("CORRUPT", f"unknown_datatype:{current.datatype}")
        if current.datatype != FLOAT32_DATATYPE or int(current.count) != 1:
            # x/y/z must be scalar FLOAT32; anything else is beyond the profile.
            raise _unsupported(
                "xyz_not_scalar_float32",
                observed={"field": name, "datatype": current.datatype, "count": current.count},
            )
        xyz[name] = current

    for item in fields:
        if item.datatype not in POINT_FIELD_DATATYPES:
            # SPEC §3.1 lists unsupported data types as POINTCLOUD_CORRUPT.
            raise _LayoutProblem("CORRUPT", f"unknown_datatype:{item.datatype}:{item.name}")
        if item.offset < 0:
            raise _LayoutProblem("CORRUPT", f"negative_field_offset:{item.name}")
        if item.offset + item.byte_size > point_step:
            raise _LayoutProblem(
                "CORRUPT",
                f"field_out_of_bounds:{item.name}:offset={item.offset}:size={item.byte_size}:"
                f"point_step={point_step}",
            )
        if item.name not in XYZ_NAMES and int(item.count) != 1:
            raise _unsupported(
                "multi_channel_field",
                observed={"field": item.name, "datatype": item.datatype, "count": item.count},
            )

    spans = sorted(
        (item.offset, item.offset + item.byte_size, item.name) for item in fields
    )
    for previous, current in zip(spans, spans[1:], strict=False):
        if current[0] < previous[1]:
            raise _LayoutProblem(
                "CORRUPT",
                f"overlapping_fields:{previous[2]}:{current[2]}",
            )
    return xyz["x"], xyz["y"], xyz["z"]


class _LayoutProblem(Exception):
    def __init__(self, kind: str, reason: str) -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def parse_pointcloud2(message: PointCloud2Message) -> PointCloudSummary:
    """Parse one PointCloud2 message into a bounded summary.

    Raises ``ApiError(UNSUPPORTED_POINTCLOUD_LAYOUT)`` for layouts outside the
    x/y/z FLOAT32 profile.  Corrupt layouts are returned with ``valid=False`` and
    a ``POINTCLOUD_CORRUPT`` finding so callers record evidence without crashing.
    """
    width = int(message.width)
    height = int(message.height)
    point_step = int(message.point_step)
    row_step = int(message.row_step)

    if width <= 0 or height <= 0 or point_step <= 0 or row_step <= 0:
        return _corrupt_summary(
            message,
            reason="non_positive_dimensions",
            point_count=max(0, width) * max(0, height),
        )
    point_count = width * height
    if point_count > MAX_POINTS:
        # Bounded: return immediately without reading the payload at all.
        return _too_large_summary(message, point_count=point_count)
    if row_step < point_step * width:
        return _corrupt_summary(
            message,
            reason="row_step_too_small",
            point_count=point_count,
            observed={"required_row_step": point_step * width},
        )
    try:
        x_field, y_field, z_field = _validate_fields(message.fields, point_step=point_step)
    except _LayoutProblem as problem:
        if problem.kind == "UNSUPPORTED":
            raise _unsupported(problem.reason, observed={}) from problem
        return _corrupt_summary(
            message,
            reason=problem.reason,
            point_count=point_count,
            observed={"fields": [field.name for field in message.fields]},
        )

    required_bytes = row_step * height
    if len(message.data) < required_bytes:
        return _corrupt_summary(
            message,
            reason="truncated_data",
            point_count=point_count,
            observed={"required_bytes": required_bytes, "actual_bytes": len(message.data)},
        )

    stride = max(1, -(-point_count // MAX_SAMPLE_POINTS))
    unpack = struct.Struct((">" if message.is_bigendian else "<") + "f").unpack_from
    data = message.data
    sampled = 0
    nonfinite = 0
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf
    has_finite = False
    for index in range(0, point_count, stride):
        base_offset = (index // width) * row_step + (index % width) * point_step
        x = unpack(data, base_offset + x_field.offset)[0]
        y = unpack(data, base_offset + y_field.offset)[0]
        z = unpack(data, base_offset + z_field.offset)[0]
        sampled += 1
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            nonfinite += 1
            continue
        has_finite = True
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)

    findings: list[dict[str, Any]] = []
    if sampled > 0:
        ratio = nonfinite / sampled
        if ratio > NONFINITE_RATIO_THRESHOLD:
            findings.append(
                _finding(
                    ErrorCode.POINTCLOUD_NONFINITE_HIGH.value,
                    "WARN",
                    observed={
                        "point_count": point_count,
                        "sampled_point_count": sampled,
                        "nonfinite_sample_count": nonfinite,
                        "nonfinite_ratio": round(ratio, 6),
                    },
                    threshold={"nonfinite_ratio_gt": NONFINITE_RATIO_THRESHOLD},
                    source_t_ns=int(message.source_t_ns),
                )
            )
    base = _base_kwargs(message)
    return PointCloudSummary(
        point_count=point_count,
        sampled_point_count=sampled,
        nonfinite_sample_count=nonfinite,
        min_x=min_x if has_finite else None,
        max_x=max_x if has_finite else None,
        min_y=min_y if has_finite else None,
        max_y=max_y if has_finite else None,
        min_z=min_z if has_finite else None,
        max_z=max_z if has_finite else None,
        valid=True,
        findings=findings,
        **base,
    )


# ---------------------------------------------------------------------------
# summary parquet
# ---------------------------------------------------------------------------
def summary_table(summaries: Iterable[PointCloudSummary]) -> pa.Table:
    rows = [summary.row() for summary in summaries]
    return pa.Table.from_pylist(rows, schema=POINTCLOUD_SUMMARY_SCHEMA)


def write_summary_parquet(path: Path, summaries: Iterable[PointCloudSummary]) -> dict[str, Any]:
    table = summary_table(summaries)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return {
        "path": str(path),
        "rows": table.num_rows,
        "schema": [f"{field.name}:{field.type}" for field in table.schema],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def summary_asset_key(episode_id: str, canonical_version: int) -> str:
    """ADR-0007 canonical-episode/2.0 native point-cloud summary location."""
    return (
        f"canonical/{episode_id}/v2/{max(1, int(canonical_version))}"
        "/native/pointcloud/summary.parquet"
    )


def store_pointcloud_summary_asset(
    session: Session,
    *,
    episode_id: str,
    canonical_version: int,
    summaries: Sequence[PointCloudSummary],
    storage: StorageBackend | None = None,
    logical_episode_id: str | None = None,
) -> Asset:
    """Write ``summary.parquet`` into the object store and register its Asset."""
    if not summaries:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "Cannot store an empty point-cloud summary",
            details={"episode_id": episode_id},
        )
    backend = storage or get_storage()
    key = summary_asset_key(episode_id, canonical_version)
    with _temporary_parquet(summaries) as path:
        stored = backend.put_file(
            key, path, content_type="application/vnd.apache.parquet"
        )
        asset = Asset(
            id=new_id(),
            sha256=stored.sha256 or "",
            byte_size=path.stat().st_size,
            media_type="application/vnd.apache.parquet",
            object_key=key,
            origin="derived",
            logical_name="native/pointcloud/summary.parquet",
            meta_json={
                "kind": "pointcloud_summary",
                "schema_version": POINTCLOUD_SUMMARY_SCHEMA_VERSION,
                "episode_id": episode_id,
                "canonical_version": int(canonical_version),
                "rows": len(summaries),
                "max_points": MAX_POINTS,
                "max_sample_points": MAX_SAMPLE_POINTS,
            },
        )
    session.add(asset)
    session.flush()
    return asset


class _temporary_parquet:
    def __init__(self, summaries: Sequence[PointCloudSummary]) -> None:
        self.summaries = summaries
        self._directory: Any = None

    def __enter__(self) -> Path:
        import tempfile

        self._directory = tempfile.TemporaryDirectory(prefix="dataops-pointcloud-")
        path = Path(self._directory.name) / "summary.parquet"
        write_summary_parquet(path, self.summaries)
        return path

    def __exit__(self, *exc_info: object) -> None:
        if self._directory is not None:
            self._directory.cleanup()


def summarize_pointcloud_messages(
    messages: Iterable[PointCloud2Message],
) -> list[PointCloudSummary]:
    return [parse_pointcloud2(message) for message in messages]
