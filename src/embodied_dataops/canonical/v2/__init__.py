"""``canonical-episode/2.0`` package (V2 SPEC 2.3, ADR-0007).

Public surface:

* contract/models: :class:`CanonicalMetadataV2`, :func:`canonical_v2_keys`;
* pure builder: :func:`build_canonical_v2` (parsed source -> serialized objects);
* alignment: :func:`align_source` / :func:`align_native` / :func:`align_prealigned`;
* parquet readers/writers and the no-zero-fill validation;
* views/preview helpers for the NORMALIZE_EPISODE v2 handler;
* :func:`load_canonical_v2` for the quality stage and the exporter.
"""

from __future__ import annotations

from embodied_dataops.canonical.v2.align import (
    AlignmentOutcome,
    align_native,
    align_prealigned,
    align_source,
)
from embodied_dataops.canonical.v2.build import build_canonical_v2
from embodied_dataops.canonical.v2.contract import (
    ALIGNED_FPS,
    ALIGNED_INTERVAL_NS,
    CANONICAL_V2_SCHEMA_VERSION,
    FLOAT32_MAX_ABS_ERROR,
    PIPELINE_VERSION_V2,
    VIEW_ALIGNED_TRAINING,
    VIEW_NATIVE,
    VIEW_PREVIEW,
    AlignedViewDescriptor,
    CanonicalMetadataV2,
    ModalityDescriptor,
    PreviewViewDescriptor,
    canonical_sha256_v2,
    canonical_v2_keys,
    canonical_v2_prefix,
)
from embodied_dataops.canonical.v2.loader import LoadedCanonicalV2, load_canonical_v2
from embodied_dataops.canonical.v2.model import (
    CanonicalV2Bundle,
    CanonicalV2Source,
    NativeCameraStream,
    NativeRobotStream,
    PointCloudSummary,
    PreviewArtifact,
)
from embodied_dataops.canonical.v2.parquet import (
    ALIGNED_SAMPLES_SCHEMA,
    NATIVE_CAMERA_SCHEMA,
    NATIVE_ROBOT_SCHEMA,
    aligned_summary,
    build_aligned_table,
    build_native_camera_table,
    build_native_robot_table,
    float32_projection,
    parquet_bytes,
    read_frame_indexes,
    read_parquet_bytes,
    validate_aligned_table,
    validate_native_camera_table,
    validate_native_robot_table,
)
from embodied_dataops.canonical.v2.preview import (
    PREVIEW_FPS,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    PreviewEncoding,
    encode_camera_preview,
    preview_frame_indexes,
    preview_grid,
)
from embodied_dataops.canonical.v2.views import (
    build_all_view_configs,
    config_sha256,
    register_canonical_views,
    view_config,
)

__all__ = [
    "ALIGNED_FPS",
    "ALIGNED_INTERVAL_NS",
    "ALIGNED_SAMPLES_SCHEMA",
    "AlignmentOutcome",
    "AlignedViewDescriptor",
    "CANONICAL_V2_SCHEMA_VERSION",
    "CanonicalMetadataV2",
    "CanonicalV2Bundle",
    "CanonicalV2Source",
    "FLOAT32_MAX_ABS_ERROR",
    "LoadedCanonicalV2",
    "ModalityDescriptor",
    "NATIVE_CAMERA_SCHEMA",
    "NATIVE_ROBOT_SCHEMA",
    "NativeCameraStream",
    "NativeRobotStream",
    "PIPELINE_VERSION_V2",
    "PREVIEW_FPS",
    "PREVIEW_HEIGHT",
    "PREVIEW_WIDTH",
    "PointCloudSummary",
    "PreviewArtifact",
    "PreviewEncoding",
    "PreviewViewDescriptor",
    "VIEW_ALIGNED_TRAINING",
    "VIEW_NATIVE",
    "VIEW_PREVIEW",
    "align_native",
    "align_prealigned",
    "align_source",
    "aligned_summary",
    "build_aligned_table",
    "build_all_view_configs",
    "build_canonical_v2",
    "build_native_camera_table",
    "build_native_robot_table",
    "canonical_sha256_v2",
    "canonical_v2_keys",
    "canonical_v2_prefix",
    "config_sha256",
    "encode_camera_preview",
    "float32_projection",
    "load_canonical_v2",
    "parquet_bytes",
    "preview_frame_indexes",
    "preview_grid",
    "read_frame_indexes",
    "read_parquet_bytes",
    "register_canonical_views",
    "validate_aligned_table",
    "validate_native_camera_table",
    "validate_native_robot_table",
    "view_config",
]
