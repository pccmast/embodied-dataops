"""Internal canonical Episode Contract v1 (SPEC 5.2).

The canonical contract is the only interchange format inside the platform:
imports produce it, quality checks read it, exports consume it.  It is *not*
the LeRobot on-disk layout; exports always go through the official Writer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from embodied_dataops import (
    CANONICAL_SCHEMA_VERSION,
    PIPELINE_VERSION,
    SAMPLE_INTERVAL_NS,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from embodied_dataops.errors import ApiError, ErrorCode

ROBOT_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("sample_index", pa.int32(), nullable=False),
        pa.field("t_ns", pa.int64(), nullable=False),
        pa.field("frame_index", pa.int32(), nullable=False),
        pa.field("state", pa.list_(pa.float32(), 6), nullable=False),
        pa.field("action", pa.list_(pa.float32(), 6), nullable=False),
        pa.field("state_source_t_ns", pa.int64(), nullable=False),
        pa.field("action_source_t_ns", pa.int64(), nullable=False),
    ]
)

EGO_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("sample_index", pa.int32(), nullable=False),
        pa.field("t_ns", pa.int64(), nullable=False),
        pa.field("frame_index", pa.int32(), nullable=False),
    ]
)

ROBOT_PROFILE = "arm6_joint_position_v1"
ROBOT_JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
STATE_DIM = 6
ACTION_DIM = 6


class CanonicalMetadata(BaseModel):
    """`metadata.json` for canonical-episode/1.0 (required fields from SPEC 5.2)."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = CANONICAL_SCHEMA_VERSION
    episode_id: str
    episode_type: str
    data_origin: str
    source_type: str
    source_asset_sha256: str = Field(min_length=64, max_length=64)
    source_episode_index: int
    duration_ns: int = Field(ge=0)
    target_fps: int = TARGET_FPS
    camera_name: str = "front"
    clock_domain: str
    joint_names: list[str] = Field(default_factory=list)
    state_dim: int = 0
    action_dim: int = 0
    samples_parquet: str = "samples.parquet"
    video_path: str = "frames/front.mp4"
    pipeline_version: str = PIPELINE_VERSION

    # additive, non-destructive contract extensions
    task_label: str = ""
    robot_profile: str = ""
    source_filename: str = ""
    source_license: str = ""
    source_uri: str | None = None
    source_revision: str | None = None
    mapping: dict[str, Any] = Field(default_factory=dict)
    normalization_valid: bool = True
    sample_count: int = 0
    video_available: bool = True
    width: int = TARGET_WIDTH
    height: int = TARGET_HEIGHT
    frame_transforms: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = ""

    @field_validator("episode_type")
    @classmethod
    def _episode_type_valid(cls, value: str) -> str:
        if value not in {"robot_episode", "ego_video_episode"}:
            raise ValueError("episode_type must be robot_episode or ego_video_episode")
        return value

    @field_validator("joint_names")
    @classmethod
    def _joint_names_valid(cls, value: list[str]) -> list[str]:
        if value and value != ROBOT_JOINT_NAMES:
            raise ValueError(f"robot profile arm6_joint_position_v1 requires joints {ROBOT_JOINT_NAMES}")
        return value

    def canonical_json(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Canonical JSON: UTF-8, sorted keys, no extra whitespace (SPEC 10.1)."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def canonical_prefix(episode_id: str, version: int = 1) -> str:
    return f"canonical/{episode_id}/v{version}"


def canonical_keys(episode_id: str, version: int = 1) -> dict[str, str]:
    prefix = canonical_prefix(episode_id, version)
    return {
        "metadata": f"{prefix}/metadata.json",
        "samples": f"{prefix}/samples.parquet",
        "video": f"{prefix}/frames/front.mp4",
        "thumbnail": f"{prefix}/frames/thumbnail.jpg",
        "quality": f"{prefix}/quality.json",
        "prefix": prefix,
    }


def annotation_key(episode_id: str, annotation_set_id: str) -> str:
    return f"annotations/{episode_id}/{annotation_set_id}/annotation_candidates.json"


@dataclass
class SampleRow:
    sample_index: int
    t_ns: int
    frame_index: int
    state: Sequence[float] | None = None
    action: Sequence[float] | None = None
    state_source_t_ns: int | None = None
    action_source_t_ns: int | None = None


def rows_to_robot_table(rows: Sequence[SampleRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_index": row.sample_index,
                "t_ns": row.t_ns,
                "frame_index": row.frame_index,
                "state": list(row.state or []),
                "action": list(row.action or []),
                "state_source_t_ns": int(row.state_source_t_ns or 0),
                "action_source_t_ns": int(row.action_source_t_ns or 0),
            }
            for row in rows
        ],
        schema=ROBOT_SAMPLE_SCHEMA,
    )


def rows_to_ego_table(rows: Sequence[SampleRow]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_index": row.sample_index,
                "t_ns": row.t_ns,
                "frame_index": row.frame_index,
            }
            for row in rows
        ],
        schema=EGO_SAMPLE_SCHEMA,
    )


def write_samples_parquet(
    path: Path, rows: Sequence[SampleRow], episode_type: str
) -> dict[str, Any]:
    """Write the canonical samples parquet and return a small summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = rows_to_robot_table(rows) if episode_type == "robot_episode" else rows_to_ego_table(rows)
    pq.write_table(table, path, compression="snappy")
    return {
        "path": str(path),
        "sample_count": table.num_rows,
        "schema": [f"{f.name}:{f.type}" for f in table.schema],
    }


def read_samples_parquet(path: Path) -> pa.Table:
    return pq.read_table(path)


def validate_samples_table(table: pa.Table, episode_type: str) -> list[str]:
    """Return a list of contract violations (empty means valid)."""
    errors: list[str] = []
    required = ["sample_index", "t_ns", "frame_index"]
    if episode_type == "robot_episode":
        required += ["state", "action", "state_source_t_ns", "action_source_t_ns"]
    names = set(table.column_names)
    for name in required:
        if name not in names:
            errors.append(f"missing column {name}")
    if errors:
        return errors
    if table.num_rows == 0:
        return ["empty samples table"]
    sample_index = table.column("sample_index").to_pylist()
    if sample_index != list(range(len(sample_index))):
        errors.append("sample_index must be contiguous from 0")
    t_ns = table.column("t_ns").to_pylist()
    for i in range(1, len(t_ns)):
        if t_ns[i] <= t_ns[i - 1]:
            errors.append(f"t_ns not strictly increasing at row {i}")
            break
    for i, value in enumerate(t_ns):
        if value != i * SAMPLE_INTERVAL_NS:
            errors.append(f"t_ns[{i}]={value} != {i * SAMPLE_INTERVAL_NS} (10Hz grid)")
            break
    frame_index = table.column("frame_index").to_pylist()
    if frame_index != list(range(len(frame_index))):
        errors.append("frame_index must match the 10Hz sample index")
    if episode_type == "robot_episode":
        for column, dim in (("state", STATE_DIM), ("action", ACTION_DIM)):
            values = table.column(column).to_pylist()
            for i, row in enumerate(values):
                if row is None or len(row) != dim:
                    errors.append(f"{column}[{i}] has dimension {0 if row is None else len(row)} != {dim}")
                    break
        action_t = table.column("action_source_t_ns").to_pylist()
        for i, value in enumerate(action_t):
            if value > t_ns[i]:
                errors.append(f"action_source_t_ns[{i}] is in the future")
                break
    return errors


def canonical_digest(metadata: CanonicalMetadata, samples_sha256: str, video_sha256: str | None) -> str:
    """Deterministic identity of one canonical episode version."""
    payload = {
        "metadata_sha256": metadata.sha256(),
        "samples_sha256": samples_sha256,
        "video_sha256": video_sha256 or "",
        "schema_version": metadata.schema_version,
        "episode_id": metadata.episode_id,
    }
    return canonical_json_sha256(payload)


@dataclass
class CanonicalEpisode:
    """In-memory canonical episode used by tests and the exporter."""

    metadata: CanonicalMetadata
    samples: pa.Table
    video_path: Path | None = None
    quality: dict[str, Any] = field(default_factory=dict)


def load_canonical_episode(
    *,
    metadata: CanonicalMetadata,
    samples_path: Path,
    video_path: Path | None = None,
    quality: dict[str, Any] | None = None,
) -> CanonicalEpisode:
    table = read_samples_parquet(samples_path)
    errors = validate_samples_table(table, metadata.episode_type)
    if errors:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "Canonical samples violate canonical-episode/1.0",
            status_code=422,
            details={"errors": errors[:20]},
        )
    return CanonicalEpisode(metadata=metadata, samples=table, video_path=video_path, quality=quality or {})
