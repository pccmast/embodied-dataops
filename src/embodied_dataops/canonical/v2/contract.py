"""``canonical-episode/2.0`` file contract (V2 SPEC 2.3, ADR-0007).

Object layout written by :mod:`embodied_dataops.workflow.handlers.normalize_episode_v2`::

    canonical/{episode_id}/v2/{canonical_version}/
      metadata.json
      native/cameras/front/timestamps.parquet
      native/cameras/wrist/timestamps.parquet      (absent => metadata says absent)
      native/state.parquet
      native/action.parquet
      native/pointcloud/summary.parquet            (optional, ROS 2 PointCloud2 only)
      aligned/10hz/samples.parquet
      preview/front_10hz_640x480.mp4
      preview/wrist_10hz_640x480.mp4               (only when the source has a wrist stream)
      mapping_report.json

``metadata.json`` is canonical JSON (UTF-8, sorted keys, no extra whitespace) and
its SHA-256 is persisted as the canonical identity of the version.  The native
index files keep the *source* sample times; the aligned view is a derived 10 Hz
training grid that never zero-fills a missing sample.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from embodied_dataops.canonical.contract import canonical_json_bytes

#: Version strings fixed by SPEC_ADDENDUM_V2 §0.4.
CANONICAL_V2_SCHEMA_VERSION = "canonical-episode/2.0"
PIPELINE_VERSION_V2 = "2.0.0"

#: ``canonical_views.view_kind`` values (SPEC 2.2).
VIEW_NATIVE = "NATIVE"
VIEW_ALIGNED_TRAINING = "ALIGNED_TRAINING"
VIEW_PREVIEW = "PREVIEW"
VIEW_KINDS: tuple[str, ...] = (VIEW_NATIVE, VIEW_ALIGNED_TRAINING, VIEW_PREVIEW)

#: Default training grid is 10 Hz; it does not throw away the native samples.
ALIGNED_FPS = 10
ALIGNED_INTERVAL_NS = 100_000_000

#: float64 -> float32 projection error above which the training export is blocked
#: (SPEC 2.3).  The original source asset keeps the float64 data.
FLOAT32_MAX_ABS_ERROR = 1e-5

#: Camera names supported by the v2 profile / canonical layout.  ``front`` is the
#: profile-required camera, ``wrist`` is optional.
CAMERA_NAMES: tuple[str, ...] = ("front", "wrist")

STATE_MODALITY = "state"
ACTION_MODALITY = "action"
ROBOT_MODALITIES: tuple[str, ...] = (STATE_MODALITY, ACTION_MODALITY)

#: Exact parquet column names of SPEC 2.3 (shared with the quality/export code).
NATIVE_CAMERA_COLUMNS: tuple[str, ...] = (
    "source_index",
    "source_t_ns",
    "episode_t_ns",
    "valid",
    "source_asset_id",
)
NATIVE_ROBOT_COLUMNS: tuple[str, ...] = (
    "source_index",
    "source_t_ns",
    "episode_t_ns",
    "values",
    "valid",
)
ALIGNED_COLUMNS: tuple[str, ...] = (
    "t_ns",
    "frame_indexes",
    "state",
    "action",
    "state_source_t_ns",
    "action_source_t_ns",
    "valid",
    "invalid_reason",
)

_SHA256_CHARS = set("0123456789abcdef")


class ModalityDescriptor(BaseModel):
    """One ``modalities[]`` entry of ``metadata.json`` (SPEC 2.3, exact fields)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(pattern="^(camera|state|action|depth|pointcloud)$")
    dtype: str = Field(min_length=1)
    shape: list[int] = Field(default_factory=list)
    unit: str = ""
    source_clock_domain: str = Field(min_length=1)
    timestamp_provenance: str = Field(min_length=1)
    source_t_ns_origin: int = 0
    native_sample_count: int = Field(ge=0)
    native_asset_sha256: str = Field(min_length=64, max_length=64)
    alignment_method: str = Field(min_length=1)
    alignment_tolerance_ns: int = Field(ge=0)

    # additive, always present evidence fields
    clock_verified: bool = False
    native_rate_hz: float | None = None
    source_key: str = ""
    native_asset_key: str = ""

    @model_validator(mode="after")
    def _sha256_hex(self) -> ModalityDescriptor:
        if set(self.native_asset_sha256) - _SHA256_CHARS:
            raise ValueError("native_asset_sha256 must be lowercase hex")
        return self


class AlignedViewDescriptor(BaseModel):
    """``aligned_view`` block of ``metadata.json`` (the derived training grid)."""

    model_config = ConfigDict(extra="forbid")

    view_kind: str = VIEW_ALIGNED_TRAINING
    fps: int = Field(ge=1, le=240)
    interval_ns: int = Field(ge=1)
    sample_count: int = Field(ge=0)
    valid_sample_count: int = Field(ge=0)
    invalid_sample_count: int = Field(ge=0)
    asset_key: str = ""
    asset_sha256: str = ""
    state_dim: int = Field(ge=0)
    action_dim: int = Field(ge=0)
    camera_names: list[str] = Field(default_factory=list)
    alignment_method: str = Field(min_length=1)
    prealigned: bool = False
    source_row_rate_hz: float | None = None
    coverage_start_ns: int | None = None
    coverage_end_ns: int | None = None
    image_tolerance_ns: int = 50_000_000
    state_tolerance_ns: int = 50_000_000
    action_tolerance_ns: int = 100_000_000

    @model_validator(mode="after")
    def _counts(self) -> AlignedViewDescriptor:
        if self.valid_sample_count + self.invalid_sample_count != self.sample_count:
            raise ValueError("aligned_view valid+invalid must equal sample_count")
        return self


class PreviewViewDescriptor(BaseModel):
    """One regenerable preview video (never training truth, SPEC 2.3)."""

    model_config = ConfigDict(extra="forbid")

    camera_name: str = Field(min_length=1)
    fps: int = Field(ge=1, le=240)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    asset_key: str = ""
    asset_sha256: str = ""
    selection: str = "nearest_native_frame_to_aligned_grid"
    lossy: bool = True
    note: str = "regenerable H.264 preview; not training-conversation truth and not the native frame bytes"


class CanonicalMetadataV2(BaseModel):
    """``metadata.json`` for ``canonical-episode/2.0`` (SPEC 2.3 required fields)."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = CANONICAL_V2_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    source_asset_sha256: str = Field(min_length=64, max_length=64)
    profile_id: str | None = None
    profile_revision: int | None = None
    profile_config_sha256: str | None = None
    episode_type: str = "robot_episode"
    clock_domains: dict[str, str] = Field(default_factory=dict)
    modalities: list[ModalityDescriptor] = Field(default_factory=list)
    aligned_view: AlignedViewDescriptor | None = None
    preview_views: list[PreviewViewDescriptor] = Field(default_factory=list)
    pipeline_version: str = PIPELINE_VERSION_V2
    data_origin: str = ""

    # additive evidence fields (present, never guessed)
    task_label: str = ""
    source_type: str = ""
    source_episode_index: int = 0
    duration_ns: int = 0
    #: State channel names (the V2 vocabulary calls any state channel a "joint").
    joint_names: list[str] = Field(default_factory=list)
    #: Action channel names, **independent** of ``joint_names``.  A 2.1 profile may
    #: declare 8 state and 7 action channels; deriving the action labels from the
    #: state list used to produce an action vector of width 7 with 8 labels.
    action_names: list[str] = Field(default_factory=list)
    state_dim: int = 0
    action_dim: int = 0
    state_semantics: str = ""
    action_semantics: str = ""
    state_unit: str = ""
    action_unit: str = ""
    camera_names: list[str] = Field(default_factory=list)
    required_modalities: list[str] = Field(default_factory=list)
    diagnostic_only: bool = False
    semantic_authority: bool = False
    publishable: bool = False
    expected_terminal_marker: bool = False
    terminal_marker_found: bool | None = None
    float32_precision: dict[str, Any] = Field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    mapping_report_key: str = ""
    mapping_report_sha256: str = ""
    pointcloud: dict[str, Any] | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""

    @model_validator(mode="after")
    def _profile_binding(self) -> CanonicalMetadataV2:
        if set(self.source_asset_sha256) - _SHA256_CHARS:
            raise ValueError("source_asset_sha256 must be lowercase hex")
        if self.episode_type != "robot_episode":
            raise ValueError("canonical-episode/2.0 is produced for robot_episode only")
        if not self.diagnostic_only:
            missing = [
                name
                for name, value in (
                    ("profile_id", self.profile_id),
                    ("profile_revision", self.profile_revision),
                    ("profile_config_sha256", self.profile_config_sha256),
                )
                if value in (None, "")
            ]
            if missing:
                raise ValueError(
                    f"canonical-episode/2.0 semantic mode requires profile binding: missing {missing}"
                )
        names = [modality.name for modality in self.modalities]
        if len(set(names)) != len(names):
            raise ValueError("modalities[] names must be unique")
        if "front" not in self.camera_names:
            raise ValueError("canonical-episode/2.0 requires the profile-required front camera")
        for required in ROBOT_MODALITIES:
            if required not in names:
                raise ValueError(f"modalities[] must include {required}")
        if self.aligned_view is None:
            raise ValueError("metadata.json requires an aligned_view block")
        if self.joint_names and self.state_dim and len(self.joint_names) != self.state_dim:
            raise ValueError(
                f"joint_names has {len(self.joint_names)} entries but state_dim is {self.state_dim}"
            )
        if self.action_names and self.action_dim and len(self.action_names) != self.action_dim:
            raise ValueError(
                f"action_names has {len(self.action_names)} entries but action_dim is "
                f"{self.action_dim}"
            )
        return self

    def canonical_json(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


def canonical_v2_prefix(episode_id: str, canonical_version: int) -> str:
    """``canonical/{episode_id}/v2/{canonical_version}`` (SPEC 2.3 literal layout)."""
    version = int(canonical_version)
    if version < 1:
        raise ValueError("canonical_version must be >= 1")
    return f"canonical/{episode_id}/v2/{version}"


def canonical_v2_keys(episode_id: str, canonical_version: int) -> dict[str, str]:
    prefix = canonical_v2_prefix(episode_id, canonical_version)
    return {
        "prefix": prefix,
        "metadata": f"{prefix}/metadata.json",
        "mapping_report": f"{prefix}/mapping_report.json",
        "native_state": f"{prefix}/native/state.parquet",
        "native_action": f"{prefix}/native/action.parquet",
        "native_camera": prefix + "/native/cameras/{camera}/timestamps.parquet",
        "aligned_samples": f"{prefix}/aligned/10hz/samples.parquet",
        "pointcloud_summary": f"{prefix}/native/pointcloud/summary.parquet",
        "preview": prefix + "/preview/{camera}_10hz_640x480.mp4",
        #: Written by BUILD_PREVIEW (not by the normalizer): the canonical v2 layout has
        #: preview videos but no thumbnail object, and ``GET /episodes/{id}/preview``
        #: needs one.  Kept inside the same version prefix so a re-normalized version
        #: never serves a thumbnail from an older one.
        "thumbnail": prefix + "/preview/thumbnail.jpg",
    }


def canonical_sha256_v2(metadata_sha256: str, aligned_sha256: str, native_hashes: dict[str, str]) -> str:
    """Deterministic identity of one canonical v2 version.

    Every input is a SHA-256 the caller observed on the *stored* bytes; nothing is
    re-derived from in-memory values, so the digest can be re-verified at export
    and publish time.
    """
    payload = {
        "schema_version": CANONICAL_V2_SCHEMA_VERSION,
        "metadata_sha256": metadata_sha256,
        "aligned_sha256": aligned_sha256,
        "native_assets": {str(key): str(value) for key, value in sorted(native_hashes.items())},
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


__all__ = [
    "ACTION_MODALITY",
    "ALIGNED_COLUMNS",
    "ALIGNED_FPS",
    "ALIGNED_INTERVAL_NS",
    "AlignedViewDescriptor",
    "CAMERA_NAMES",
    "CANONICAL_V2_SCHEMA_VERSION",
    "CanonicalMetadataV2",
    "FLOAT32_MAX_ABS_ERROR",
    "ModalityDescriptor",
    "NATIVE_CAMERA_COLUMNS",
    "NATIVE_ROBOT_COLUMNS",
    "PIPELINE_VERSION_V2",
    "PreviewViewDescriptor",
    "ROBOT_MODALITIES",
    "STATE_MODALITY",
    "VIEW_ALIGNED_TRAINING",
    "VIEW_KINDS",
    "VIEW_NATIVE",
    "VIEW_PREVIEW",
    "canonical_sha256_v2",
    "canonical_v2_keys",
    "canonical_v2_prefix",
]
