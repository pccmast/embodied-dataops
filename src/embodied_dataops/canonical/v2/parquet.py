"""Parquet writers/readers for ``canonical-episode/2.0`` (V2 SPEC 2.3).

Three index families are written, each as an immutable Parquet object:

* ``native/cameras/<name>/timestamps.parquet`` -- one row per decoded source frame
  (columns ``source_index, source_t_ns, episode_t_ns, valid, source_asset_id``);
* ``native/state.parquet`` / ``native/action.parquet`` -- one row per source
  sample (columns ``source_index, source_t_ns, episode_t_ns, values, valid``);
* ``aligned/10hz/samples.parquet`` -- the derived training grid (columns
  ``t_ns, frame_indexes, state, action, state_source_t_ns, action_source_t_ns,
  valid, invalid_reason``).

``values``/``state``/``action`` are ``list<float32>``.  The canonical files store
the float32 projection of the original float64 source values; the maximum
absolute projection error is computed here (:func:`float32_projection`) and
recorded in ``metadata.json``.  A missing aligned sample is represented by
``valid=false`` with ``null`` arrays -- never by a zero-filled vector.

All writers serialize to bytes (``parquet_bytes``) so the handler can upload the
exact bytes whose SHA-256 it records in ``metadata.json``.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embodied_dataops.canonical.v2.contract import (
    ALIGNED_COLUMNS,
    CAMERA_NAMES,
    FLOAT32_MAX_ABS_ERROR,
    NATIVE_CAMERA_COLUMNS,
    NATIVE_ROBOT_COLUMNS,
)

#: Struct column with one optional index per supported camera.  A ``null`` entry
#: means "no frame within the alignment tolerance"; the row is then ``valid=false``.
FRAME_INDEXES_TYPE = pa.struct(
    [
        pa.field("front", pa.int64(), nullable=True),
        pa.field("wrist", pa.int64(), nullable=True),
    ]
)

NATIVE_CAMERA_SCHEMA = pa.schema(
    [
        pa.field("source_index", pa.int64(), nullable=False),
        pa.field("source_t_ns", pa.int64(), nullable=False),
        pa.field("episode_t_ns", pa.int64(), nullable=False),
        pa.field("valid", pa.bool_(), nullable=False),
        pa.field("source_asset_id", pa.string(), nullable=False),
    ]
)

NATIVE_ROBOT_SCHEMA = pa.schema(
    [
        pa.field("source_index", pa.int64(), nullable=False),
        pa.field("source_t_ns", pa.int64(), nullable=False),
        pa.field("episode_t_ns", pa.int64(), nullable=False),
        pa.field("values", pa.list_(pa.float32()), nullable=False),
        pa.field("valid", pa.bool_(), nullable=False),
    ]
)

ALIGNED_SAMPLES_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64(), nullable=False),
        pa.field("frame_indexes", FRAME_INDEXES_TYPE, nullable=False),
        pa.field("state", pa.list_(pa.float32()), nullable=True),
        pa.field("action", pa.list_(pa.float32()), nullable=True),
        pa.field("state_source_t_ns", pa.int64(), nullable=True),
        pa.field("action_source_t_ns", pa.int64(), nullable=True),
        pa.field("valid", pa.bool_(), nullable=False),
        pa.field("invalid_reason", pa.list_(pa.string()), nullable=False),
    ]
)

_PRECISION = 1_000_000


# ---------------------------------------------------------------------------
# float64 -> float32 projection (precision evidence, SPEC 2.3 / ADR-0007 §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Float32Projection:
    """Result of one float64 -> float32 projection."""

    values: list[list[float]]
    max_abs_error: float
    nonfinite_count: int

    @property
    def blocked(self) -> bool:
        return self.max_abs_error > FLOAT32_MAX_ABS_ERROR

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_abs_error": round(float(self.max_abs_error), 12),
            "max_abs_error_threshold": FLOAT32_MAX_ABS_ERROR,
            "blocked": self.blocked,
            "nonfinite_count": int(self.nonfinite_count),
        }


def float32_projection(rows: Sequence[Sequence[float]]) -> Float32Projection:
    """Cast float64 source rows to float32 and measure the worst absolute error.

    Non-finite inputs are preserved as float32 NaN/Inf (they are evidence for the
    quality rules) and counted; they never silently become zero.
    """
    if not rows:
        return Float32Projection(values=[], max_abs_error=0.0, nonfinite_count=0)
    source = np.asarray([list(row) for row in rows], dtype=np.float64)
    projected = source.astype(np.float32)
    finite = np.isfinite(source) & np.isfinite(projected)
    if finite.any():
        error = np.abs(source[finite].astype(np.float64) - projected[finite].astype(np.float64))
        max_error = float(error.max()) if error.size else 0.0
    else:
        max_error = 0.0
    nonfinite = int(np.count_nonzero(~np.isfinite(source)))
    values = projected.tolist()
    return Float32Projection(values=values, max_abs_error=max_error, nonfinite_count=nonfinite)


# ---------------------------------------------------------------------------
# generic parquet bytes / tables
# ---------------------------------------------------------------------------


def parquet_bytes(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    return buffer.getvalue()


def table_bytes_sha256(table: pa.Table) -> str:
    import hashlib

    return hashlib.sha256(parquet_bytes(table)).hexdigest()


def read_parquet_bytes(data: bytes) -> pa.Table:
    return pq.read_table(io.BytesIO(data))


def read_parquet_path(path: Any) -> pa.Table:
    return pq.read_table(path)


# ---------------------------------------------------------------------------
# native indices
# ---------------------------------------------------------------------------


def build_native_camera_table(
    *,
    times_ns: Sequence[int],
    origin_ns: int,
    source_asset_id: str,
    valid: Sequence[bool] | None = None,
) -> pa.Table:
    """Native camera timestamps, one row per source frame (SPEC 2.3 columns)."""
    times = [int(value) for value in times_ns]
    rows = []
    for index, source_t_ns in enumerate(times):
        is_valid = True if valid is None else bool(valid[index])
        rows.append(
            {
                "source_index": index,
                "source_t_ns": source_t_ns,
                "episode_t_ns": source_t_ns - int(origin_ns),
                "valid": is_valid,
                "source_asset_id": str(source_asset_id),
            }
        )
    table = pa.Table.from_pylist(rows, schema=NATIVE_CAMERA_SCHEMA)
    return _check_columns(table, NATIVE_CAMERA_COLUMNS)


def build_native_robot_table(
    *,
    times_ns: Sequence[int],
    values: Sequence[Sequence[float]],
    origin_ns: int,
    expected_dim: int | None,
) -> tuple[pa.Table, Float32Projection, int]:
    """Native state/action index plus float32 projection evidence and valid count."""
    if len(times_ns) != len(values):
        raise ValueError("native robot times and values must have the same length")
    projection = float32_projection(values)
    rows = []
    invalid = 0
    for index, source_t_ns in enumerate(times_ns):
        row_values = projection.values[index]
        is_valid = True
        if expected_dim is not None and len(row_values) != int(expected_dim):
            is_valid = False
        if any(not np.isfinite(value) for value in np.asarray(row_values, dtype=np.float64)):
            is_valid = False
        if not is_valid:
            invalid += 1
        rows.append(
            {
                "source_index": index,
                "source_t_ns": int(source_t_ns),
                "episode_t_ns": int(source_t_ns) - int(origin_ns),
                "values": list(row_values),
                "valid": is_valid,
            }
        )
    table = pa.Table.from_pylist(rows, schema=NATIVE_ROBOT_SCHEMA)
    return _check_columns(table, NATIVE_ROBOT_COLUMNS), projection, len(rows) - invalid


# ---------------------------------------------------------------------------
# aligned view
# ---------------------------------------------------------------------------


def aligned_row(
    *,
    t_ns: int,
    frame_indexes: dict[str, int | None],
    state: Sequence[float] | None,
    action: Sequence[float] | None,
    state_source_t_ns: int | None,
    action_source_t_ns: int | None,
    valid: bool,
    invalid_reason: Sequence[str],
) -> dict[str, Any]:
    """One aligned sample in the *internal* row representation (before Arrow)."""
    return {
        "t_ns": int(t_ns),
        "frame_indexes": {name: frame_indexes.get(name) for name in CAMERA_NAMES},
        "state": None if state is None else [float(value) for value in state],
        "action": None if action is None else [float(value) for value in action],
        "state_source_t_ns": state_source_t_ns,
        "action_source_t_ns": action_source_t_ns,
        "valid": bool(valid),
        "invalid_reason": [str(item) for item in invalid_reason],
    }


def build_aligned_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    """Arrow table for the 10 Hz aligned view, enforcing the no-zero-fill rule."""
    for row in rows:
        if row["valid"]:
            if row["state"] is None or row["action"] is None:
                raise ValueError("a valid aligned row must carry state and action (never null)")
            if row["invalid_reason"]:
                raise ValueError("a valid aligned row must not carry invalid_reason entries")
        else:
            if row["state"] is not None or row["action"] is not None:
                raise ValueError("an invalid aligned row must have null state/action (never zero-filled)")
            if not row["invalid_reason"]:
                raise ValueError("an invalid aligned row must explain itself via invalid_reason")
    table = pa.Table.from_pylist(list(rows), schema=ALIGNED_SAMPLES_SCHEMA)
    return _check_columns(table, ALIGNED_COLUMNS)


def read_frame_indexes(table: pa.Table) -> list[dict[str, int | None]]:
    column = table.column("frame_indexes").to_pylist()
    return [dict(value or {}) for value in column]


# ---------------------------------------------------------------------------
# validation (used by tests, quality and export)
# ---------------------------------------------------------------------------


def validate_native_camera_table(table: pa.Table, *, source_index_start: int = 0) -> list[str]:
    errors: list[str] = []
    missing = [name for name in NATIVE_CAMERA_COLUMNS if name not in table.column_names]
    if missing:
        return [f"missing columns {missing}"]
    if table.num_rows == 0:
        return ["native camera index is empty"]
    source_index = table.column("source_index").to_pylist()
    if source_index != list(range(source_index_start, source_index_start + len(source_index))):
        errors.append("source_index must be contiguous from the declared start")
    source_t = table.column("source_t_ns").to_pylist()
    for index in range(1, len(source_t)):
        if source_t[index] < source_t[index - 1]:
            errors.append(f"source_t_ns decreases at row {index}")
            break
    return errors


def validate_native_robot_table(table: pa.Table, *, expected_dim: int | None) -> list[str]:
    errors: list[str] = []
    missing = [name for name in NATIVE_ROBOT_COLUMNS if name not in table.column_names]
    if missing:
        return [f"missing columns {missing}"]
    if table.num_rows == 0:
        return ["native robot index is empty"]
    values = table.column("values").to_pylist()
    if expected_dim is not None:
        for index, row in enumerate(values):
            if row is not None and len(row) != expected_dim:
                errors.append(f"values[{index}] has {len(row)} dims, profile requires {expected_dim}")
                break
    source_index = table.column("source_index").to_pylist()
    if source_index != list(range(len(source_index))):
        errors.append("source_index must be contiguous from 0")
    return errors


def validate_aligned_table(table: pa.Table) -> list[str]:
    """Return contract violations of one aligned 10 Hz table."""
    errors: list[str] = []
    missing = [name for name in ALIGNED_COLUMNS if name not in table.column_names]
    if missing:
        return [f"missing columns {missing}"]
    if table.num_rows == 0:
        return ["aligned samples table is empty"]
    t_ns = table.column("t_ns").to_pylist()
    for index, value in enumerate(t_ns):
        if index and value <= t_ns[index - 1]:
            errors.append(f"t_ns not strictly increasing at row {index}")
            break
        if value < 0:
            errors.append(f"t_ns[{index}] is negative")
            break
        if value != index * 100_000_000:
            errors.append(f"t_ns[{index}]={value} is not on the 10 Hz grid")
            break
    valid = table.column("valid").to_pylist()
    state = table.column("state").to_pylist()
    action = table.column("action").to_pylist()
    reasons = table.column("invalid_reason").to_pylist()
    frame_indexes = read_frame_indexes(table)
    state_source = table.column("state_source_t_ns").to_pylist()
    action_source = table.column("action_source_t_ns").to_pylist()
    for index in range(table.num_rows):
        row_reasons = list(reasons[index] or [])
        if valid[index]:
            if not row_reasons:
                pass
            else:
                errors.append(f"valid row {index} must not have invalid_reason")
            if state[index] is None or action[index] is None:
                errors.append(f"valid row {index} must carry state and action")
            if frame_indexes[index].get("front") is None:
                errors.append(f"valid row {index} has no front frame index")
            if state_source[index] is None or action_source[index] is None:
                errors.append(f"valid row {index} has no source timestamps")
        else:
            if state[index] is not None or action[index] is not None:
                errors.append(f"invalid row {index} must have null state/action (no zero-fill)")
            if not row_reasons:
                errors.append(f"invalid row {index} must record invalid_reason")
    return errors


def aligned_summary(table: pa.Table) -> dict[str, Any]:
    valid = table.column("valid").to_pylist()
    reasons: dict[str, int] = {}
    for row in table.column("invalid_reason").to_pylist():
        for reason in row or []:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "sample_count": table.num_rows,
        "valid_sample_count": int(sum(1 for value in valid if value)),
        "invalid_sample_count": int(sum(1 for value in valid if not value)),
        "invalid_reason_counts": dict(sorted(reasons.items())),
    }


def _check_columns(table: pa.Table, expected: Sequence[str]) -> pa.Table:
    if list(table.column_names) != list(expected):
        raise ValueError(f"parquet columns {table.column_names} do not match contract {list(expected)}")
    return table


__all__ = [
    "ALIGNED_SAMPLES_SCHEMA",
    "FRAME_INDEXES_TYPE",
    "Float32Projection",
    "NATIVE_CAMERA_SCHEMA",
    "NATIVE_ROBOT_SCHEMA",
    "aligned_row",
    "aligned_summary",
    "build_aligned_table",
    "build_native_camera_table",
    "build_native_robot_table",
    "float32_projection",
    "parquet_bytes",
    "read_frame_indexes",
    "read_parquet_bytes",
    "read_parquet_path",
    "table_bytes_sha256",
    "validate_aligned_table",
    "validate_native_camera_table",
    "validate_native_robot_table",
]
