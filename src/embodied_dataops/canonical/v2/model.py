"""In-memory model shared by the v2 importers, builder and normalize handler.

The adapters (:mod:`embodied_dataops.imports.lerobot_v3_v2`,
:mod:`embodied_dataops.imports.mcap_profile_v2`) produce a
:class:`CanonicalV2Source`; :func:`embodied_dataops.canonical.v2.build.build_canonical_v2`
turns it into serialized canonical v2 objects.  Nothing here writes to the object
store or the database, so the whole contract is unit-testable in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NativeCameraStream:
    """One camera stream with its *native* timestamps and frame accessor."""

    name: str
    source_key: str
    clock_domain: str
    timestamp_provenance: str
    times_ns: list[int]
    width: int = 0
    height: int = 0
    encoding: str = "rgb8"
    #: ``getter(source_index) -> RGB uint8 array``; optional so pure contract
    #: tests can build canonical objects without decoding any video.
    frame_getter: Callable[[int], np.ndarray] | None = None
    #: Same as ``frame_getter`` but only used for previews/export.
    source_asset_id: str = ""


@dataclass
class NativeRobotStream:
    """One state/action stream in source order (original float64 values)."""

    name: str
    source_key: str
    clock_domain: str
    timestamp_provenance: str
    times_ns: list[int]
    values: list[list[float]]
    unit: str = "rad"
    source_asset_id: str = ""


@dataclass
class CanonicalV2Source:
    """Everything one adapter parsed for one Episode (no side effects)."""

    episode_id: str
    source_asset_sha256: str
    source_type: str
    source_episode_index: int
    data_origin: str
    episode_type: str = "robot_episode"
    task_label: str = ""
    #: Resolved robot profile (``None`` only in diagnostic-only mode).
    profile: Any | None = None
    profile_id: str | None = None
    profile_revision: int | None = None
    profile_config_sha256: str | None = None
    profile_authority: str = ""
    joint_names: list[str] = field(default_factory=list)
    #: Action channel names.  Kept separate from ``joint_names`` because a 2.1
    #: profile may declare a different action width (LIBERO 8/7).
    action_names: list[str] = field(default_factory=list)
    state_semantics: str = ""
    action_semantics: str = ""
    state_unit: str = "rad"
    action_unit: str = "rad"
    diagnostic_only: bool = False
    semantic_authority: bool = False

    cameras: list[NativeCameraStream] = field(default_factory=list)
    state: NativeRobotStream | None = None
    action: NativeRobotStream | None = None

    #: "native" -> nearest-frame / linear-interp / ZOH alignment of independent
    #:          source streams;
    #: "prealigned" -> LeRobot rows are already paired; row pairing is preserved
    #:          and no independent native rate is claimed.
    alignment_mode: str = "native"
    prealigned_row_times_ns: list[int] = field(default_factory=list)
    source_row_rate_hz: float | None = None
    source_fps_claimed: bool = False

    origin_ns: int = 0
    clock_domains: dict[str, str] = field(default_factory=dict)
    native_rates_hz: dict[str, float | None] = field(default_factory=dict)
    required_modalities: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    expected_terminal_marker: bool = False
    terminal_marker_found: bool | None = None
    pointcloud: dict[str, Any] | None = None
    mapping_report: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def camera(self, name: str) -> NativeCameraStream | None:
        for stream in self.cameras:
            if stream.name == name:
                return stream
        return None

    @property
    def camera_names(self) -> list[str]:
        return [stream.name for stream in self.cameras]


@dataclass
class CanonicalV2Bundle:
    """Serialized canonical v2 objects plus the metadata that describes them."""

    metadata: Any
    metadata_bytes: bytes
    artifacts: dict[str, bytes]
    mapping_report: dict[str, Any]
    alignment_diagnostics: dict[str, Any]
    precision: dict[str, Any]
    publishable: bool
    blocking_reasons: list[str] = field(default_factory=list)

    @property
    def metadata_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(self.metadata_bytes).hexdigest()

    @property
    def aligned_sha256(self) -> str:
        import hashlib

        data = self.artifacts.get("aligned_samples")
        return hashlib.sha256(data or b"").hexdigest()


@dataclass
class PreviewArtifact:
    """One encoded preview video handed to :func:`build_canonical_v2`."""

    camera_name: str
    asset_key: str
    asset_sha256: str
    fps: int
    width: int
    height: int
    frame_count: int
    selection: str = "nearest_native_frame_to_aligned_grid"


@dataclass
class PointCloudSummary:
    """Optional ``native/pointcloud/summary.parquet`` metadata (ROS 2 only)."""

    asset_key: str
    asset_sha256: str
    point_count: int
    sampled_point_count: int
    nonfinite_sample_count: int
    endianness: str
    source_asset_sha256: str
    bbox: dict[str, float | None] = field(default_factory=dict)


def rows_from_values(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


__all__ = [
    "CanonicalV2Bundle",
    "CanonicalV2Source",
    "NativeCameraStream",
    "NativeRobotStream",
    "PointCloudSummary",
    "PreviewArtifact",
    "rows_from_values",
]
