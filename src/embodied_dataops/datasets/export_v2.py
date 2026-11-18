"""D2-M1 canonical-episode/2.0 dataset export (official LeRobot 0.4.4 Writer).

This module implements DO2-08 on top of the D2-M1 compat gate:

* ``export_canonical_v2_episodes`` writes 6D/7D canonical v2 episodes (front
  required, wrist optional, aligned 10 Hz) through the **official**
  ``LeRobotDataset.create/add_frame/save_episode/finalize`` API -- one dataset
  root, one ``repo_id`` per version, features
  ``observation.images.<camera>`` / ``observation.state`` / ``action``;
* ``verify_export_v2`` re-opens the packaged export in a **separate Python
  process** with the official ``LeRobotDataset`` reader and compares episode and
  frame counts, camera count/keys, state/action dimensions, task labels and >=10
  deterministic random samples (abs error <= 1e-5);
* ``create_v2_version`` is the service-level V2 branch: eligibility, the
  mixed-profile/camera/fps/action-semantics gate (409
  ``MIXED_PROFILE_EXPORT_UNSUPPORTED``), dataset-manifest/2.0 freezing, official
  Writer export, independent Reader round-trip and the BUILDING -> READY
  transition.  The V1 service path stays byte-identical.

Image source for the aligned training frames: the canonical preview video of the
same camera, decoded sequentially (streaming, never fully in RAM).  The preview is
generated from the native camera frames at exactly the aligned 10 Hz grid, so the
exported image count always equals the aligned row count; preview compression is
a documented lossy re-encoding of those aligned frames (the native frame bytes
remain in the source asset).  This is recorded in the export asset meta as
``image_source=canonical_preview_10hz``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops.canonical.v2.contract import (
    CANONICAL_V2_SCHEMA_VERSION,
    CanonicalMetadataV2,
    canonical_v2_keys,
)
from embodied_dataops.canonical.v2.loader import load_canonical_v2
from embodied_dataops.canonical.v2.parquet import read_parquet_bytes, validate_aligned_table
from embodied_dataops.datasets.publish_policy import EXPORT_KIND_RESEARCH
from embodied_dataops.datasets.manifest import utc_timestamp
from embodied_dataops.datasets.manifest_v2 import (
    MANIFEST_V2_SCHEMA_VERSION,
    build_manifest_v2,
    manifest_v2_bytes,
    manifest_v2_sha256,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.storage.base import sha256_file

CANONICAL_PREVIEW_WIDTH = 640
CANONICAL_PREVIEW_HEIGHT = 480
ROUNDTRIP_MAX_ABS_ERROR = 1e-5
ROUNDTRIP_MIN_SAMPLES = 10


# ---------------------------------------------------------------------------
# compat gate (kept from the first D2-M1 pass)
# ---------------------------------------------------------------------------


def _mixed(details: dict[str, Any]) -> ApiError:
    return ApiError(
        ErrorCode.MIXED_PROFILE_EXPORT_UNSUPPORTED,
        "A LeRobot export version must use one profile revision, camera-key "
        "combination, aligned fps and action semantics",
        details=details,
    )


def validate_v2_membership(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate one candidate v2 export membership and return its shared contract."""
    if not episodes:
        raise ApiError(ErrorCode.EMPTY_VERSION, "A dataset version needs at least one episode")
    violations: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    for episode in episodes:
        schema = str(episode.get("schema_version") or "")
        if schema != CANONICAL_V2_SCHEMA_VERSION:
            violations.append(
                {
                    "episode_id": str(episode.get("episode_id") or ""),
                    "reason": "NOT_CANONICAL_V2",
                    "schema_version": schema,
                }
            )
            continue
        signature = {
            "profile_id": str(episode.get("profile_id") or ""),
            "profile_revision": int(episode.get("profile_revision") or 0),
            "profile_config_sha256": str(episode.get("profile_config_sha256") or ""),
            "camera_keys": sorted(str(key) for key in (episode.get("camera_keys") or [])),
            "aligned_fps": int(episode.get("aligned_fps") or 0),
            "action_semantics": str(episode.get("action_semantics") or ""),
            "state_dim": int(episode.get("state_dim") or 0),
            "action_dim": int(episode.get("action_dim") or 0),
            # Raw label lists: two episodes with the same widths but different
            # channel identities cannot share one LeRobot feature block.
            "joint_names": [str(name) for name in (episode.get("joint_names") or [])],
            "action_names": [str(name) for name in (episode.get("action_names") or [])],
        }
        if reference is None:
            reference = signature
            continue
        for field_name, value in signature.items():
            if value != reference[field_name]:
                violations.append(
                    {
                        "episode_id": str(episode.get("episode_id") or ""),
                        "reason": "MIXED_MEMBERSHIP",
                        "field": field_name,
                        "expected": reference[field_name],
                        "observed": value,
                    }
                )
    if violations:
        raise _mixed({"violations": violations[:20], "violation_count": len(violations)})
    assert reference is not None
    return reference


def episode_manifest_entry(
    *,
    episode_id: str,
    metadata: Mapping[str, Any],
    canonical_sha256: str,
    source_sha256: str,
    quality_report_id: str,
    annotation_set_id: str,
    annotation_revision_id: str | None,
    native_indexes: Mapping[str, str],
    allow_research_snapshot: bool = False,
) -> dict[str, Any]:
    """Build the dataset-manifest/2.0 episode entry from canonical v2 metadata."""
    if metadata.get("schema_version") != CANONICAL_V2_SCHEMA_VERSION:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "dataset-manifest/2.0 requires canonical-episode/2.0 metadata",
            details={"episode_id": episode_id, "schema_version": metadata.get("schema_version")},
        )
    if not bool(metadata.get("publishable")):
        codes = {str(item.get("code")) for item in (metadata.get("diagnostics") or [])}
        semantic_only = bool(codes) and codes <= RESEARCH_SNAPSHOT_DIAGNOSTICS
        if not (allow_research_snapshot and semantic_only):
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "Episode is not publishable for training (invalid aligned view / precision / semantics)",
                details={
                    "episode_id": episode_id,
                    "diagnostics": [
                        item.get("code") for item in (metadata.get("diagnostics") or [])[:20]
                    ],
                    "research_snapshot_possible": bool(allow_research_snapshot) and semantic_only,
                },
            )
    aligned = metadata.get("aligned_view") or {}
    return {
        "episode_id": episode_id,
        "canonical_sha256": canonical_sha256,
        "source_sha256": source_sha256,
        "quality_report_id": quality_report_id,
        "annotation_set_id": annotation_set_id,
        "annotation_revision_id": annotation_revision_id,
        "profile_id": str(metadata["profile_id"]),
        "profile_revision": int(metadata["profile_revision"]),
        "profile_config_sha256": str(metadata["profile_config_sha256"]),
        "camera_keys": sorted(str(name) for name in (metadata.get("camera_names") or [])),
        "aligned_fps": int(aligned.get("fps") or 0),
        "state_dim": int(metadata.get("state_dim") or 0),
        "action_dim": int(metadata.get("action_dim") or 0),
        "joint_names": [str(name) for name in (metadata.get("joint_names") or [])],
        # Frozen separately from joint_names: an 8/7 profile has 8 state and 7 action
        # labels, and the manifest must record which is which.
        "action_names": [str(name) for name in (metadata.get("action_names") or [])],
        "action_semantics": str(metadata.get("action_semantics") or ""),
        "native_indexes": {str(name): str(value) for name, value in native_indexes.items()},
        "aligned_view_sha256": str(aligned.get("asset_sha256") or ""),
    }


# ---------------------------------------------------------------------------
# materialised export input
# ---------------------------------------------------------------------------


@dataclass
class CanonicalV2ExportEpisode:
    """Everything the official Writer needs for one canonical v2 episode."""

    episode_id: str
    metadata: CanonicalMetadataV2
    aligned: Any  # pyarrow.Table
    preview_paths: dict[str, Path] = field(default_factory=dict)
    native_indexes: dict[str, str] = field(default_factory=dict)
    canonical_sha256: str = ""
    source_sha256: str = ""
    quality_report_id: str = ""
    annotation_set_id: str = ""
    annotation_revision_id: str | None = None

    @property
    def camera_keys(self) -> list[str]:
        return sorted(self.preview_paths)

    @property
    def task_label(self) -> str:
        return (self.metadata.task_label or "").strip()

    @property
    def aligned_fps(self) -> int:
        assert self.metadata.aligned_view is not None
        return int(self.metadata.aligned_view.fps)

    @property
    def state_dim(self) -> int:
        return int(self.metadata.state_dim)

    @property
    def action_dim(self) -> int:
        return int(self.metadata.action_dim)

    @property
    def joint_names(self) -> list[str]:
        return list(self.metadata.joint_names)

    @property
    def action_names(self) -> list[str]:
        """Action labels, independent of ``joint_names`` (2.1 profiles may differ)."""
        return list(self.metadata.action_names)

    def metadata_dict(self) -> dict[str, Any]:
        return self.metadata.model_dump(mode="json")


@dataclass
class V2ExpectedSample:
    frame_index: int
    t_ns: int
    state: list[float]
    action: list[float]


@dataclass
class V2ExpectedEpisode:
    episode_id: str
    task_label: str
    samples: Sequence[V2ExpectedSample] = field(default_factory=list)


@dataclass
class V2RoundTripResult:
    episode_count: int
    frame_count: int
    state_dim: int
    action_dim: int
    camera_keys: list[str]
    aligned_fps: int
    tasks: list[str]
    samples_checked: int
    max_abs_error: float
    reader_process: str
    package_version: str | None
    max_timestamp_error: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "lerobot-roundtrip/2.0",
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "camera_keys": list(self.camera_keys),
            "aligned_fps": self.aligned_fps,
            "tasks": self.tasks,
            "samples_checked": self.samples_checked,
            "max_abs_error": self.max_abs_error,
            "max_timestamp_error": self.max_timestamp_error,
            "reader_process": self.reader_process,
            "package_version": self.package_version,
        }


@dataclass
class V2ExportResult:
    root: Path
    repo_id: str
    episode_count: int
    total_frames: int
    features: dict[str, Any]
    package_version: str
    camera_keys: list[str]
    state_dim: int
    action_dim: int
    aligned_fps: int


# ---------------------------------------------------------------------------
# loading one canonical v2 member from storage
# ---------------------------------------------------------------------------


def _require_sha256(metadata: Mapping[str, Any], key: str, episode_id: str) -> str:
    value = str(metadata.get(key) or "")
    if len(value) != 64:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "canonical-episode/2.0 metadata is missing a required SHA-256",
            details={"episode_id": episode_id, "field": key},
        )
    return value


def materialize_export_episode(
    storage: Any,
    *,
    episode_id: str,
    canonical_version: int,
    workdir: Path,
    allow_research_snapshot: bool = False,
) -> CanonicalV2ExportEpisode:
    """Load one stored canonical v2 version and verify its recorded SHA-256 values."""
    loaded = load_canonical_v2(storage, episode_id=episode_id, canonical_version=canonical_version)
    metadata = loaded.metadata
    keys = canonical_v2_keys(episode_id, canonical_version)
    aligned = loaded.aligned_samples
    if aligned is None:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "canonical-episode/2.0 aligned view is missing; cannot export",
            details={"episode_id": episode_id, "missing": loaded.missing},
        )
    errors = validate_aligned_table(aligned)
    if errors:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "canonical-episode/2.0 aligned view violates the contract",
            details={"episode_id": episode_id, "errors": errors[:10]},
        )
    valid_column = aligned.column("valid").to_pylist()
    if not all(valid_column):
        invalid = sum(1 for value in valid_column if not value)
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "aligned view contains invalid samples (null state/action); a training export "
            "requires a fully valid episode",
            details={"episode_id": episode_id, "invalid_sample_count": invalid},
        )
    if not bool(metadata.publishable):
        if not (allow_research_snapshot and research_snapshot_allowed(metadata)):
            raise _not_publishable(
                episode_id=episode_id,
                metadata=metadata,
                allow_research_snapshot=allow_research_snapshot,
            )
    # Verify the recorded native/aligned digests against the stored bytes.
    native_indexes: dict[str, str] = {}
    modality_digests = {modality.name: modality.native_asset_sha256 for modality in metadata.modalities}
    camera_dir = workdir / "canonical" / episode_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    preview_paths: dict[str, Path] = {}
    for modality in metadata.modalities:
        if modality.kind == "camera":
            native_key = keys["native_camera"].format(camera=modality.name)
            camera_native = camera_dir / f"{modality.name}_timestamps.parquet"
            storage.download_to(native_key, camera_native)
            if sha256_file(camera_native) != modality.native_asset_sha256:
                raise ApiError(
                    ErrorCode.INCOMPATIBLE_EPISODE,
                    "stored native camera index does not match the recorded SHA-256",
                    details={"episode_id": episode_id, "camera": modality.name},
                )
            native_indexes[f"camera:{modality.name}"] = modality.native_asset_sha256
            preview = next(
                (view for view in metadata.preview_views if view.camera_name == modality.name), None
            )
            if preview is None:
                raise ApiError(
                    ErrorCode.INCOMPATIBLE_EPISODE,
                    "canonical-episode/2.0 has no preview video for an exported camera",
                    details={"episode_id": episode_id, "camera": modality.name},
                )
            preview_path = workdir / "previews" / episode_id / f"{modality.name}.mp4"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            storage.download_to(keys["preview"].format(camera=modality.name), preview_path)
            if sha256_file(preview_path) != preview.asset_sha256:
                raise ApiError(
                    ErrorCode.INCOMPATIBLE_EPISODE,
                    "stored preview video does not match the recorded SHA-256",
                    details={"episode_id": episode_id, "camera": modality.name},
                )
            preview_paths[modality.name] = preview_path
        else:
            native_key = keys["native_" + modality.name]
            target = camera_dir / f"{modality.name}.parquet"
            storage.download_to(native_key, target)
            if sha256_file(target) != modality.native_asset_sha256:
                raise ApiError(
                    ErrorCode.INCOMPATIBLE_EPISODE,
                    "stored native robot index does not match the recorded SHA-256",
                    details={"episode_id": episode_id, "stream": modality.name},
                )
            native_indexes[modality.name] = modality.native_asset_sha256
    del modality_digests
    aligned_sha = _require_sha256(
        {"aligned": (metadata.aligned_view.asset_sha256 if metadata.aligned_view else "")},
        "aligned",
        episode_id,
    )
    native_indexes["aligned"] = aligned_sha
    if not preview_paths:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "canonical-episode/2.0 has no exportable camera preview",
            details={"episode_id": episode_id},
        )
    return CanonicalV2ExportEpisode(
        episode_id=episode_id,
        metadata=metadata,
        aligned=aligned,
        preview_paths=preview_paths,
        native_indexes=native_indexes,
    )


# ---------------------------------------------------------------------------
# official Writer export
# ---------------------------------------------------------------------------


def _camera_feature_key(camera: str) -> str:
    return f"observation.images.{camera}"


def _action_feature_names(episode: CanonicalV2ExportEpisode) -> list[str]:
    """Label every action channel exactly once, or refuse the export.

    The V2 code used ``joint_names`` for both features.  With an 8/7 profile that
    produced ``shape=(7,)`` and 8 labels: the official Writer then rejects the
    dataset, or (worse) a consumer reads the labels as channel identities.  A
    same-width profile keeps the historical behaviour; an unequal-width profile
    must declare its own action names.
    """
    action_names = episode.action_names
    if len(action_names) == episode.action_dim:
        return action_names
    if episode.action_dim == episode.state_dim and len(episode.joint_names) == episode.action_dim:
        return list(episode.joint_names)
    raise ApiError(
        ErrorCode.INCOMPATIBLE_EPISODE,
        "canonical-episode/2.0 action channels have no labels matching action_dim",
        details={
            "episode_id": episode.episode_id,
            "action_dim": episode.action_dim,
            "state_dim": episode.state_dim,
            "action_names": len(action_names),
            "joint_names": len(episode.joint_names),
        },
    )


def build_v2_features(episode: CanonicalV2ExportEpisode) -> dict[str, Any]:
    if not episode.joint_names:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "canonical-episode/2.0 has no joint names; the LeRobot features need labels",
            details={"episode_id": episode.episode_id},
        )
    if len(episode.joint_names) != episode.state_dim:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "canonical-episode/2.0 joint names do not match state_dim",
            details={
                "episode_id": episode.episode_id,
                "state_dim": episode.state_dim,
                "joint_names": len(episode.joint_names),
            },
        )
    features: dict[str, Any] = {
        _camera_feature_key(camera): {
            "dtype": "video",
            "shape": (3, CANONICAL_PREVIEW_HEIGHT, CANONICAL_PREVIEW_WIDTH),
            "names": ["channels", "height", "width"],
        }
        for camera in episode.camera_keys
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (episode.state_dim,),
        "names": list(episode.joint_names),
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (episode.action_dim,),
        "names": _action_feature_names(episode),
    }
    return features


class _SequentialPreviewReader:
    """Streaming RGB decoder for one canonical preview video (no full buffering)."""

    def __init__(self, path: Path, *, width: int, height: int) -> None:
        import cv2  # noqa: PLC0415

        self._cv2 = cv2
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():  # pragma: no cover - corrupt preview
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "cannot open the canonical preview video for export",
                details={"path": self.path.name},
            )
        self.consumed = 0

    def next_frame(self) -> np.ndarray:
        ok, frame = self._capture.read()
        if not ok:
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "canonical preview video ended before the aligned view did",
                details={"path": self.path.name, "consumed": self.consumed},
            )
        self.consumed += 1
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "canonical preview frame does not match 640x480",
                details={
                    "path": self.path.name,
                    "shape": [int(rgb.shape[0]), int(rgb.shape[1])],
                },
            )
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def assert_exhausted(self) -> None:
        ok, _frame = self._capture.read()
        if ok:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "canonical preview video has more frames than the aligned view",
                details={"path": self.path.name},
            )

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> _SequentialPreviewReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def export_canonical_v2_episodes(
    root: Path,
    episodes: Sequence[CanonicalV2ExportEpisode],
    *,
    repo_id: str,
    fps: int | None = None,
    allow_research_snapshot: bool = False,
) -> V2ExportResult:
    """Write canonical v2 episodes through the official LeRobot 0.4.4 Writer."""
    from embodied_dataops.datasets.lerobot_writer import (
        _accepted_call_kwargs,
        _add_frame,
        _import_lerobot_dataset,
        _save_episode,
        lerobot_version,
    )

    if not episodes:
        raise ApiError(ErrorCode.EMPTY_VERSION, "Cannot export a dataset version without episodes")
    for candidate in episodes:
        if candidate.metadata.schema_version != CANONICAL_V2_SCHEMA_VERSION:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "LeRobot v3 v2 export requires canonical-episode/2.0 metadata",
                details={
                    "episode_id": candidate.episode_id,
                    "schema_version": candidate.metadata.schema_version,
                },
            )
        if not bool(candidate.metadata.publishable):
            if not (allow_research_snapshot and research_snapshot_allowed(candidate.metadata)):
                raise _not_publishable(
                    episode_id=candidate.episode_id,
                    metadata=candidate.metadata,
                    allow_research_snapshot=allow_research_snapshot,
                )
        aligned_errors = validate_aligned_table(candidate.aligned)
        if aligned_errors:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "canonical-episode/2.0 aligned view violates the contract",
                details={"episode_id": candidate.episode_id, "errors": aligned_errors[:10]},
            )
        if not all(candidate.aligned.column("valid").to_pylist()):
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "aligned view contains invalid samples; a training export requires a fully valid episode",
                details={"episode_id": candidate.episode_id},
            )
    first = episodes[0]
    contract = validate_v2_membership(
        [
            {
                "episode_id": episode.episode_id,
                "schema_version": episode.metadata.schema_version,
                "profile_id": episode.metadata.profile_id,
                "profile_revision": episode.metadata.profile_revision,
                "profile_config_sha256": episode.metadata.profile_config_sha256,
                "camera_keys": episode.camera_keys,
                "aligned_fps": episode.aligned_fps,
                "action_semantics": episode.metadata.action_semantics,
                "state_dim": episode.state_dim,
                "action_dim": episode.action_dim,
                "joint_names": list(episode.joint_names),
                "action_names": list(episode.action_names),
            }
            for episode in episodes
        ]
    )
    del contract
    aligned_fps = int(fps or first.aligned_fps)
    if aligned_fps < 1:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "canonical-episode/2.0 aligned fps is invalid",
            details={"aligned_fps": aligned_fps},
        )
    root = Path(root)
    if root.exists():
        if any(root.iterdir()):
            raise ApiError(ErrorCode.INVALID_INPUT, "LeRobot export directory must be empty")
        root.rmdir()
    root.parent.mkdir(parents=True, exist_ok=True)
    LeRobotDataset = _import_lerobot_dataset()
    features = build_v2_features(first)
    robot_type = str(first.metadata.profile_id or "")
    candidate = {
        "repo_id": repo_id,
        "fps": aligned_fps,
        "root": root,
        "robot_type": robot_type,
        "features": features,
        "use_videos": True,
        "tolerance_s": 1e-4,
        "image_writer_processes": 0,
        "image_writer_threads": 0,
        "batch_encoding_size": 1,
    }
    kwargs = _accepted_call_kwargs(LeRobotDataset.create, candidate)
    kwargs.setdefault("repo_id", repo_id)
    kwargs.setdefault("fps", aligned_fps)
    kwargs.setdefault("root", root)
    kwargs.setdefault("features", features)
    dataset = LeRobotDataset.create(**kwargs)

    total_frames = 0
    camera_keys = list(first.camera_keys)
    for episode in episodes:
        if not episode.task_label:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "canonical-episode/2.0 has no task label; a LeRobot export requires one task",
                details={"episode_id": episode.episode_id},
            )
        if episode.camera_keys != camera_keys:
            raise _mixed(
                {
                    "episode_id": episode.episode_id,
                    "reason": "CAMERA_KEYS_DIFFER",
                    "expected": camera_keys,
                    "observed": episode.camera_keys,
                }
            )
        rows = episode.aligned.to_pylist()
        if not rows:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "canonical-episode/2.0 aligned view is empty",
                details={"episode_id": episode.episode_id},
            )
        readers = {
            camera: _SequentialPreviewReader(
                episode.preview_paths[camera],
                width=CANONICAL_PREVIEW_WIDTH,
                height=CANONICAL_PREVIEW_HEIGHT,
            )
            for camera in camera_keys
        }
        try:
            for row in rows:
                frame_indexes = row.get("frame_indexes") or {}
                frame: dict[str, Any] = {}
                for camera in camera_keys:
                    index = frame_indexes.get(camera)
                    if index is None:
                        raise ApiError(
                            ErrorCode.INCOMPATIBLE_EPISODE,
                            "aligned row has no frame index for an exported camera",
                            details={"episode_id": episode.episode_id, "camera": camera},
                        )
                    frame[_camera_feature_key(camera)] = readers[camera].next_frame()
                state = np.asarray(row["state"], dtype=np.float32)
                action = np.asarray(row["action"], dtype=np.float32)
                if state.shape != (episode.state_dim,) or action.shape != (episode.action_dim,):
                    raise ApiError(
                        ErrorCode.INCOMPATIBLE_EPISODE,
                        "aligned row dimensions do not match the canonical metadata",
                        details={
                            "episode_id": episode.episode_id,
                            "state_shape": list(state.shape),
                            "action_shape": list(action.shape),
                        },
                    )
                frame["observation.state"] = state
                frame["action"] = action
                _add_frame(dataset, frame, episode.task_label, int(row["t_ns"]) / 1_000_000_000)
                total_frames += 1
            for reader in readers.values():
                reader.assert_exhausted()
        finally:
            for reader in readers.values():
                reader.close()
        _save_episode(dataset, episode.task_label)

    finalize = getattr(dataset, "finalize", None)
    if not callable(finalize):
        raise ApiError(
            ErrorCode.LEROBOT_API_UNAVAILABLE,
            "lerobot 0.4.4 must expose LeRobotDataset.finalize()",
        )
    finalize()
    return V2ExportResult(
        root=root,
        repo_id=repo_id,
        episode_count=len(episodes),
        total_frames=total_frames,
        features=features,
        package_version=lerobot_version() or "0.4.4",
        camera_keys=camera_keys,
        state_dim=first.state_dim,
        action_dim=first.action_dim,
        aligned_fps=aligned_fps,
    )


# ---------------------------------------------------------------------------
# separate-process official Reader round-trip
# ---------------------------------------------------------------------------


def _expected_episodes(episodes: Sequence[CanonicalV2ExportEpisode]) -> list[V2ExpectedEpisode]:
    expected: list[V2ExpectedEpisode] = []
    for episode in episodes:
        expected.append(
            V2ExpectedEpisode(
                episode_id=episode.episode_id,
                task_label=episode.task_label,
                samples=[
                    V2ExpectedSample(
                        frame_index=int(row["t_ns"]) // max(1, 1_000_000_000 // episode.aligned_fps),
                        t_ns=int(row["t_ns"]),
                        state=[float(value) for value in row["state"]],
                        action=[float(value) for value in row["action"]],
                    )
                    for row in episode.aligned.to_pylist()
                ],
            )
        )
    return expected


def _fail(message: str, **details: Any) -> ApiError:
    return ApiError(ErrorCode.EXPORT_ROUND_TRIP_FAILED, message, details=details)


def verify_export_v2(
    package_path: Path,
    *,
    repo_id: str,
    expected: Sequence[V2ExpectedEpisode],
    camera_keys: Sequence[str],
    state_dim: int,
    action_dim: int,
    aligned_fps: int,
    state_names: Sequence[str] = (),
    action_names: Sequence[str] = (),
    timeout: int = 300,
    workdir: Path | None = None,
) -> V2RoundTripResult:
    """Re-open the export with the official Reader in a fresh process and compare."""
    from embodied_dataops.datasets import lerobot_reader

    total_rows = sum(len(episode.samples) for episode in expected)
    if total_rows == 0:
        raise _fail("Cannot verify an export with no samples")
    root = lerobot_reader.resolve_export_root(Path(package_path), workdir)
    payload = lerobot_reader._run_inspector(
        root,
        repo_id=repo_id,
        requests=lerobot_reader.sample_requests(total_rows),
        timeout=timeout,
    )
    episode_count = int(payload.get("num_episodes") or 0)
    frame_count = int(payload.get("num_frames") or 0)
    if episode_count != len(expected):
        raise _fail("Round-trip episode count mismatch", expected=len(expected), observed=episode_count)
    if frame_count != total_rows:
        raise _fail("Round-trip total frame count mismatch", expected=total_rows, observed=frame_count)
    if int(payload.get("fps") or 0) != int(aligned_fps):
        raise _fail("Round-trip fps mismatch", expected=aligned_fps, observed=payload.get("fps"))
    features = payload.get("features") or {}
    observed_cameras = sorted(
        name for name in features if str(name).startswith("observation.images.")
    )
    expected_cameras = sorted(_camera_feature_key(name) for name in camera_keys)
    if observed_cameras != expected_cameras:
        raise _fail(
            "Round-trip camera feature keys mismatch",
            expected=expected_cameras,
            observed=observed_cameras,
        )
    state_shape = list((features.get("observation.state") or {}).get("shape") or [])
    action_shape = list((features.get("action") or {}).get("shape") or [])
    observed_state_dim = int(state_shape[0]) if state_shape else 0
    observed_action_dim = int(action_shape[0]) if action_shape else 0
    if observed_state_dim != int(state_dim) or observed_action_dim != int(action_dim):
        raise _fail(
            "Round-trip state/action dimensions mismatch",
            expected={"state_dim": state_dim, "action_dim": action_dim},
            observed={"state_dim": observed_state_dim, "action_dim": observed_action_dim},
        )
    for feature_name, expected_labels in (
        ("observation.state", state_names),
        ("action", action_names),
    ):
        if not expected_labels:
            continue
        observed_labels = [
            str(item) for item in ((features.get(feature_name) or {}).get("names") or [])
        ]
        # The official Reader may omit labels entirely; when it reports them they
        # must match channel-for-channel, otherwise the export mislabels a dimension.
        if observed_labels and list(expected_labels) != observed_labels:
            raise _fail(
                f"Round-trip {feature_name} feature names mismatch",
                expected=list(expected_labels),
                observed=observed_labels,
            )
    observed_lengths = {
        int(key): int(value) for key, value in (payload.get("episode_lengths") or {}).items()
    }
    expected_lengths = {index: len(episode.samples) for index, episode in enumerate(expected)}
    if observed_lengths and observed_lengths != expected_lengths:
        raise _fail(
            "Round-trip per-episode frame counts mismatch",
            expected=expected_lengths,
            observed=observed_lengths,
        )
    observed_episode_tasks = {
        int(key): str(value) for key, value in (payload.get("episode_tasks") or {}).items()
    }
    expected_tasks = [episode.task_label for episode in expected]
    for index, task in enumerate(expected_tasks):
        if index in observed_episode_tasks and observed_episode_tasks[index] != task:
            raise _fail(
                "Round-trip task label mismatch",
                episode_index=index,
                expected=task,
                observed=observed_episode_tasks[index],
            )
    offsets: list[int] = []
    running = 0
    for episode in expected:
        offsets.append(running)
        running += len(episode.samples)
    max_error = 0.0
    max_timestamp_error = 0.0
    checked = 0
    for sample in payload.get("samples") or []:
        flat_index = int(sample.get("index") or 0)
        episode_index = int(sample.get("episode_index") or 0)
        if not 0 <= episode_index < len(expected):
            raise _fail("Round-trip returned an unknown episode_index", episode_index=episode_index)
        zero_based = flat_index - offsets[episode_index]
        episode = expected[episode_index]
        if not 0 <= zero_based < len(episode.samples):
            raise _fail(
                "Round-trip sample row maps outside the canonical episode",
                flat_index=flat_index,
                episode_index=episode_index,
            )
        canonical = episode.samples[zero_based]
        observed_frame_index = sample.get("frame_index")
        if observed_frame_index is not None and int(observed_frame_index) != int(
            canonical.frame_index
        ):
            raise _fail(
                "Round-trip frame_index mismatch",
                episode_index=episode_index,
                expected=canonical.frame_index,
                observed=observed_frame_index,
            )
        for field_name in ("state", "action"):
            observed_values = sample.get(field_name) or []
            expected_values = getattr(canonical, field_name)
            if len(observed_values) != len(expected_values):
                raise _fail(
                    f"Round-trip {field_name} length mismatch",
                    expected=len(expected_values),
                    observed=len(observed_values),
                )
            for expected_value, observed_value in zip(expected_values, observed_values, strict=False):
                max_error = max(max_error, abs(float(expected_value) - float(observed_value)))
        observed_timestamp = sample.get("timestamp")
        if observed_timestamp is not None:
            expected_timestamp = canonical.t_ns / 1_000_000_000
            max_timestamp_error = max(
                max_timestamp_error, abs(float(observed_timestamp) - expected_timestamp)
            )
        checked += 1
    if checked == 0:
        raise _fail("Official LeRobot reader returned no samples to compare")
    if checked < min(ROUNDTRIP_MIN_SAMPLES, total_rows):
        raise _fail("Official LeRobot reader returned too few samples", checked=checked)
    if max_error > ROUNDTRIP_MAX_ABS_ERROR:
        raise _fail(
            "Round-trip state/action absolute error exceeds tolerance",
            max_abs_error=max_error,
            threshold=ROUNDTRIP_MAX_ABS_ERROR,
        )
    return V2RoundTripResult(
        episode_count=episode_count,
        frame_count=frame_count,
        state_dim=observed_state_dim,
        action_dim=observed_action_dim,
        camera_keys=list(camera_keys),
        aligned_fps=int(aligned_fps),
        tasks=expected_tasks,
        samples_checked=checked,
        max_abs_error=max_error,
        max_timestamp_error=max_timestamp_error,
        reader_process="separate_process:official_lerobot_reader",
        package_version=payload.get("package_version"),
    )


# ---------------------------------------------------------------------------
# service-level V2 branch
# ---------------------------------------------------------------------------


def is_canonical_v2_episode(episode: Any) -> bool:
    return str(getattr(episode, "canonical_schema_version", "1.0") or "1.0") == "2.0"


def episodes_are_v2(session: Any, episode_ids: Sequence[str]) -> bool:
    from embodied_dataops.db.models import Episode  # noqa: PLC0415

    if not episode_ids:
        return False
    rows = [session.get(Episode, episode_id) for episode_id in episode_ids]
    return all(row is not None and is_canonical_v2_episode(row) for row in rows)


def _v2_metadata_for(session: Any, episode: Any) -> tuple[CanonicalMetadataV2, dict[str, Any]]:
    storage = _storage()
    loaded = load_canonical_v2(
        storage,
        episode_id=episode.id,
        canonical_version=int(episode.canonical_version or 1),
    )
    return loaded.metadata, loaded.metadata.model_dump(mode="json")


def _storage() -> Any:
    from embodied_dataops.storage.base import get_storage  # noqa: PLC0415

    return get_storage()


#: Diagnostics that mean "the physics is not evidenced", not "the bytes are broken".
#: A research snapshot may carry these -- it must then stay labelled
#: ``publish_level=FORMAT_VERIFIED`` / ``training_ready=false``.  Anything outside this
#: allow-list (dimension mismatch, null state/action, bad aligned table, missing objects)
#: still blocks the export, so the research path cannot smuggle broken data out.
RESEARCH_SNAPSHOT_DIAGNOSTICS = frozenset({"UNVERIFIED_SEMANTICS", "CAMERA_UNMAPPED_IN_SOURCE"})


def research_snapshot_allowed(metadata: Any) -> bool:
    """True when a non-publishable episode is blocked only by unevidenced semantics."""
    if bool(getattr(metadata, "publishable", False)):
        return True
    codes = {str(item.get("code")) for item in (getattr(metadata, "diagnostics", None) or [])}
    return bool(codes) and codes <= RESEARCH_SNAPSHOT_DIAGNOSTICS


def _not_publishable(
    *, episode_id: str, metadata: Any, allow_research_snapshot: bool
) -> ApiError:
    diagnostics = [item.get("code") for item in metadata.diagnostics[:20]]
    return ApiError(
        ErrorCode.INCOMPATIBLE_EPISODE,
        "canonical-episode/2.0 episode is not publishable for training",
        details={
            "episode_id": episode_id,
            "diagnostics": diagnostics,
            "allowed_for_training": False,
            "research_snapshot_possible": bool(allow_research_snapshot)
            and research_snapshot_allowed(metadata),
        },
    )


def validate_v2_episode_eligible(
    session: Any, episode: Any, *, allow_research_snapshot: bool = False
) -> Any:
    """V2 replacement for ``service.validate_episode_eligible`` (same 409 codes)."""
    from embodied_dataops.datasets import service as service_module  # noqa: PLC0415
    from embodied_dataops.db.models import SourceImport  # noqa: PLC0415
    from embodied_dataops.domain.enums import EpisodeType, QualityStatus  # noqa: PLC0415

    if episode is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Episode not found")
    if episode.episode_type != EpisodeType.ROBOT_EPISODE.value:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Only robot_episode data can be exported as LeRobot robot trajectories",
            details={"episode_id": episode.id, "episode_type": episode.episode_type},
        )
    if episode.quality_status not in {QualityStatus.PASS.value, QualityStatus.WARN.value}:
        raise ApiError(
            ErrorCode.QC_STATUS_NOT_ELIGIBLE,
            "Episode quality verdict is not PASS or WARN",
            details={"episode_id": episode.id, "quality_status": episode.quality_status},
        )
    quality = service_module._latest_quality_report(session, episode.id)
    if quality is None:
        raise ApiError(
            ErrorCode.QC_STATUS_NOT_ELIGIBLE,
            "Episode has no stored QualityReport",
            details={"episode_id": episode.id},
        )
    annotation = service_module._approved_annotation_set(session, episode.id)
    if annotation is None:
        raise ApiError(
            ErrorCode.ANNOTATION_NOT_APPROVED,
            "Episode has no APPROVED AnnotationSet",
            details={"episode_id": episode.id},
        )
    if episode.quality_status == QualityStatus.WARN.value:
        from sqlalchemy import select  # noqa: PLC0415

        from embodied_dataops.db.models import AnnotationReview  # noqa: PLC0415

        review = session.scalars(
            select(AnnotationReview).where(AnnotationReview.annotation_set_id == annotation.id)
        ).first()
        note = (review.note or "").strip() if review is not None else ""
        if len(note) < service_module.WARN_RELEASE_NOTE_MIN_CHARS:
            raise ApiError(
                ErrorCode.ANNOTATION_NOT_APPROVED,
                f"WARN release requires an approval note of at least "
                f"{service_module.WARN_RELEASE_NOTE_MIN_CHARS} characters",
                details={"episode_id": episode.id, "annotation_set_id": annotation.id},
            )
    source = session.get(SourceImport, episode.source_import_id)
    license_value = (source.license or "").strip().lower() if source is not None else ""
    if license_value in service_module.UNKNOWN_LICENSES:
        raise ApiError(
            ErrorCode.LICENSE_UNKNOWN,
            "Episode source licence is unknown; unknown-licence data cannot be published",
            details={"episode_id": episode.id, "license": source.license if source else None},
        )
    metadata, metadata_dict = _v2_metadata_for(session, episode)
    if metadata.schema_version != CANONICAL_V2_SCHEMA_VERSION:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode canonical metadata is not canonical-episode/2.0",
            details={"episode_id": episode.id, "schema_version": metadata.schema_version},
        )
    if not bool(metadata.publishable):
        if not (allow_research_snapshot and research_snapshot_allowed(metadata)):
            raise _not_publishable(
                episode_id=episode.id,
                metadata=metadata,
                allow_research_snapshot=allow_research_snapshot,
            )
    if not (metadata.task_label or "").strip():
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode has no task label; a LeRobot export requires one task per episode",
            details={"episode_id": episode.id},
        )
    canonical_digest = str(episode.canonical_sha256 or "")
    if len(canonical_digest) != 64:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode canonical_sha256 is missing; normalization must complete first",
            details={"episode_id": episode.id},
        )
    storage = _storage()
    keys = canonical_v2_keys(episode.id, int(episode.canonical_version or 1))
    for required in (
        keys["metadata"],
        keys["aligned_samples"],
        keys["mapping_report"],
    ):
        if not storage.exists(required):
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "Episode canonical v2 object is missing; run NORMALIZE_EPISODE first",
                details={"episode_id": episode.id, "object": required},
            )
    if not metadata.preview_views:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode has no canonical preview video to export aligned frames from",
            details={"episode_id": episode.id},
        )
    source_sha = (
        source.sha256
        if source is not None and source.sha256 and len(source.sha256) == 64
        else metadata.source_asset_sha256
    )
    del metadata_dict
    return service_module.EpisodeEligibility(
        episode=episode,
        quality_report_id=quality.id,
        annotation_set_id=annotation.id,
        canonical_digest=canonical_digest,
        source_sha256=str(source_sha),
        task_label=(metadata.task_label or "").strip(),
    )


def freeze_manifest_v2(session: Any, version: Any, dataset: Any, rows: Sequence[Any]) -> Any:
    """Build + persist a dataset-manifest/2.0 for the version membership."""
    from embodied_dataops.datasets import service as service_module  # noqa: PLC0415

    entries: list[dict[str, Any]] = []
    camera_transforms: list[dict[str, Any]] = []
    for row in rows:
        episode = session.get(service_module.Episode, row.episode_id)
        if episode is None:
            raise ApiError(ErrorCode.NOT_FOUND, "episode of this dataset version no longer exists")
        metadata, metadata_dict = _v2_metadata_for(session, episode)
        native_indexes = {
            f"camera:{modality.name}": modality.native_asset_sha256
            for modality in metadata.modalities
            if modality.kind == "camera"
        }
        native_indexes.update(
            {
                modality.name: modality.native_asset_sha256
                for modality in metadata.modalities
                if modality.kind in {"state", "action"}
            }
        )
        if metadata.aligned_view is not None:
            native_indexes["aligned"] = metadata.aligned_view.asset_sha256
        entries.append(
            episode_manifest_entry(
                episode_id=episode.id,
                metadata=metadata_dict,
                canonical_sha256=str(row.canonical_digest),
                source_sha256=str(metadata_dict.get("source_asset_sha256") or ""),
                quality_report_id=str(row.quality_report_id),
                annotation_set_id=(
                    "" if row.annotation_set_id is None else str(row.annotation_set_id)
                ),
                annotation_revision_id=(
                    None if row.annotation_revision_id is None else str(row.annotation_revision_id)
                ),
                native_indexes=native_indexes,
                allow_research_snapshot=str(getattr(version, "export_kind", ""))
                == EXPORT_KIND_RESEARCH,
            )
        )
        for preview in metadata.preview_views:
            camera_transforms.append(
                {
                    "camera": preview.camera_name,
                    "fps": int(preview.fps),
                    "width": int(preview.width),
                    "height": int(preview.height),
                    "selection": preview.selection,
                    "lossy_preview": True,
                }
            )
    # Membership compatibility is guaranteed by create_v2_version / the export
    # gate before this point; the manifest profile block comes from the frozen
    # per-episode entries (which do not carry a schema_version field themselves).
    first_entry = entries[0]
    from embodied_dataops.datasets.publish_policy import (  # noqa: PLC0415
        MANIFEST_PUBLISH_LEVEL_KEY,
        publish_level_for_profile,
    )

    publish_level = publish_level_for_profile(
        session,
        profile_id=first_entry["profile_id"],
        profile_revision=first_entry["profile_revision"],
    )
    manifest = build_manifest_v2(
        dataset_slug=dataset.slug,
        version=int(version.version),
        episodes=entries,
        profile={
            "profile_id": first_entry["profile_id"],
            "profile_revision": first_entry["profile_revision"],
            "profile_config_sha256": first_entry["profile_config_sha256"],
            "authority": "",
            MANIFEST_PUBLISH_LEVEL_KEY: publish_level,
        },
        camera_transforms=_dedupe_transforms(camera_transforms),
        created_at=utc_timestamp(),
    )
    body = manifest_v2_bytes(manifest)
    digest = manifest_v2_sha256(manifest)
    key = service_module.MANIFEST_KEY_TEMPLATE.format(
        dataset_id=version.dataset_id, version=version.version
    )
    asset = service_module._store_bytes(
        session,
        key=key,
        data=body,
        media_type="application/json",
        origin="export",
        logical_name=f"{dataset.slug}-v{version.version}-manifest.json",
        meta={
            "kind": "dataset_manifest",
            "schema_version": MANIFEST_V2_SCHEMA_VERSION,
            "dataset_id": version.dataset_id,
            "version": version.version,
            "episode_count": len(entries),
            "profile_id": first_entry["profile_id"],
            "profile_revision": first_entry["profile_revision"],
        },
    )
    if asset.sha256 != digest:  # pragma: no cover - defensive
        raise ApiError(
            ErrorCode.INTERNAL_ERROR, "Manifest asset digest does not match canonical manifest"
        )
    version.manifest_asset_id = asset.id
    version.manifest_sha256 = digest
    session.flush()
    return manifest, digest, asset


def _dedupe_transforms(transforms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in transforms:
        key = tuple(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(item))
    return unique


def materialize_export_episodes(
    session: Any,
    snapshots: Sequence[Any],
    *,
    workdir: Path,
    allow_research_snapshot: bool = False,
) -> list[CanonicalV2ExportEpisode]:
    storage = _storage()
    from embodied_dataops.db.models import Episode  # noqa: PLC0415

    episodes: list[CanonicalV2ExportEpisode] = []
    for snapshot in snapshots:
        episode = session.get(Episode, snapshot.episode_id)
        if episode is None:
            raise ApiError(
                ErrorCode.NOT_FOUND, f"episode of this dataset version is missing: {snapshot.episode_id}"
            )
        materialized = materialize_export_episode(
            storage,
            episode_id=episode.id,
            canonical_version=int(episode.canonical_version or 1),
            workdir=workdir,
            allow_research_snapshot=allow_research_snapshot,
        )
        materialized.canonical_sha256 = str(snapshot.canonical_digest)
        materialized.quality_report_id = str(snapshot.quality_report_id)
        materialized.annotation_set_id = (
            "" if snapshot.annotation_set_id is None else str(snapshot.annotation_set_id)
        )
        materialized.annotation_revision_id = (
            None if snapshot.annotation_revision_id is None else str(snapshot.annotation_revision_id)
        )
        materialized.source_sha256 = str(
            materialized.metadata.model_dump(mode="json").get("source_asset_sha256") or ""
        )
        episodes.append(materialized)
    return episodes


def build_export_v2(session: Any, version: Any, *, snapshots: Sequence[Any]) -> Any:
    """Materialise + export one canonical v2 dataset version (service branch)."""
    from embodied_dataops.datasets import service as service_module  # noqa: PLC0415
    from embodied_dataops.datasets.lerobot_writer import cleanup, tar_directory  # noqa: PLC0415

    dataset = service_module.get_dataset(session, version.dataset_id)
    workdir = Path(tempfile.mkdtemp(prefix=f"dataops-export-v2-{version.id[:8]}-"))
    export_root = workdir / "lerobot-v3"
    try:
        ordered = sorted(snapshots, key=lambda row: row.episode_id)
        # The version's export_kind is derived from the profile (never passed in), so a
        # FORMAT_VERIFIED profile lands here as RESEARCH_SNAPSHOT and may carry
        # semantic-only diagnostics; everything else keeps failing closed.
        allow_research_snapshot = str(getattr(version, "export_kind", "")) == EXPORT_KIND_RESEARCH
        episodes = materialize_export_episodes(
            session,
            ordered,
            workdir=workdir,
            allow_research_snapshot=allow_research_snapshot,
        )
        repo_id = f"embodied-dataops/{dataset.slug}-v{version.version}"
        result = export_canonical_v2_episodes(
            export_root,
            episodes,
            repo_id=repo_id,
            allow_research_snapshot=allow_research_snapshot,
        )
        archive = tar_directory(export_root, workdir / "lerobot-v3.tar.gz")
        key = service_module.EXPORT_KEY_TEMPLATE.format(
            dataset_id=version.dataset_id, version=version.version
        )
        asset = service_module._store_file(
            session,
            key=key,
            path=archive,
            media_type="application/gzip",
            origin="export",
            logical_name=f"{dataset.slug}-v{version.version}-lerobot-v3.tar.gz",
            meta={
                "kind": "lerobot_v3_export",
                "schema_version": "lerobot-v3-export/2.0",
                "canonical_schema_version": CANONICAL_V2_SCHEMA_VERSION,
                "dataset_id": version.dataset_id,
                "version": version.version,
                "repo_id": repo_id,
                "episode_count": result.episode_count,
                "total_frames": result.total_frames,
                "camera_keys": list(result.camera_keys),
                "state_dim": result.state_dim,
                "action_dim": result.action_dim,
                "aligned_fps": result.aligned_fps,
                "writer_package_version": result.package_version,
                "image_source": "canonical_preview_10hz",
            },
        )
        version.export_asset_id = asset.id
        session.flush()
        return service_module.ExportBuild(
            asset_id=asset.id,
            archive_path=archive,
            repo_id=repo_id,
            expected=_expected_episodes(episodes),
            episode_count=result.episode_count,
            total_frames=result.total_frames,
            workdir=workdir,
            package_version=result.package_version,
            schema_version="lerobot-v3-export/2.0",
            camera_keys=list(result.camera_keys),
            state_dim=result.state_dim,
            action_dim=result.action_dim,
            aligned_fps=result.aligned_fps,
            state_names=list(episodes[0].joint_names),
            action_names=_action_feature_names(episodes[0]),
        )
    except Exception:
        cleanup(workdir)
        raise


def verify_build_v2(build: Any, *, workdir: Path) -> V2RoundTripResult:
    return verify_export_v2(
        build.archive_path,
        repo_id=build.repo_id,
        expected=build.expected,
        camera_keys=build.camera_keys,
        state_dim=build.state_dim,
        action_dim=build.action_dim,
        aligned_fps=build.aligned_fps,
        state_names=list(getattr(build, "state_names", []) or []),
        action_names=list(getattr(build, "action_names", []) or []),
        workdir=workdir,
    )


def create_v2_version(
    session: Any,
    *,
    dataset_id: str,
    episode_ids: Sequence[str],
    description: str = "",
    actor: Any = None,
    request_id: str | None = None,
) -> Any:
    """V2 branch of ``service.create_version`` (same state machine and audit)."""
    from embodied_dataops.datasets import service as service_module  # noqa: PLC0415
    from embodied_dataops.db.models import Dataset, Episode, SourceImport  # noqa: PLC0415
    from embodied_dataops.domain.enums import DatasetVersionState  # noqa: PLC0415
    from embodied_dataops.ids import new_id  # noqa: PLC0415
    from embodied_dataops.observability.audit import record_audit  # noqa: PLC0415
    from embodied_dataops.workflow.outbox import utcnow  # noqa: PLC0415

    dataset = session.get(Dataset, dataset_id, with_for_update=True)
    if dataset is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Dataset not found: {dataset_id}")
    ordered_ids = service_module._normalise_episode_ids(episode_ids)
    episodes = [session.get(Episode, episode_id) for episode_id in ordered_ids]
    metadata_list: list[dict[str, Any]] = []
    for episode in episodes:
        if episode is None:
            raise ApiError(ErrorCode.NOT_FOUND, "episode does not exist")
        metadata, metadata_dict = _v2_metadata_for(session, episode)
        metadata_list.append(
            {
                "episode_id": episode.id,
                "schema_version": metadata.schema_version,
                "profile_id": metadata.profile_id,
                "profile_revision": metadata.profile_revision,
                "profile_config_sha256": metadata.profile_config_sha256,
                "camera_keys": list(metadata.camera_names),
                "aligned_fps": int(metadata.aligned_view.fps) if metadata.aligned_view else 0,
                "action_semantics": metadata.action_semantics,
                "state_dim": metadata.state_dim,
                "action_dim": metadata.action_dim,
                "joint_names": list(metadata.joint_names),
                "action_names": list(metadata.action_names),
            }
        )
    # 409 MIXED_PROFILE_EXPORT_UNSUPPORTED before any version row is created.
    contract = validate_v2_membership(metadata_list)
    from embodied_dataops.datasets.publish_policy import (  # noqa: PLC0415
        export_kind_for_publish_level,
        publish_level_for_profile,
        training_ready_allowed,
    )

    # The level is decided by the profile before eligibility runs: a FORMAT_VERIFIED
    # revision may still be frozen, but only as a research snapshot.
    publish_level = publish_level_for_profile(
        session,
        profile_id=contract["profile_id"],
        profile_revision=contract["profile_revision"],
    )
    allow_research_snapshot = not training_ready_allowed(publish_level)
    eligibility = [
        validate_v2_episode_eligible(
            session, episode, allow_research_snapshot=allow_research_snapshot
        )
        for episode in episodes
    ]
    export_kind = export_kind_for_publish_level(publish_level)
    version = service_module.DatasetVersion(
        id=new_id(),
        dataset_id=dataset.id,
        version=service_module._next_version_number(session, dataset.id),
        state=DatasetVersionState.BUILDING.value,
        description=(description or "").strip(),
        created_by=actor.key_id if actor is not None else None,
        created_at=utcnow(),
        export_kind=export_kind,
    )
    session.add(version)
    session.flush()
    service_module._snapshot_episodes(session, version, eligibility)
    record_audit(
        session,
        actor=actor,
        action="dataset_version.create",
        entity_type="dataset_version",
        entity_id=version.id,
        after={
            "dataset_id": dataset.id,
            "version": version.version,
            "episode_ids": ordered_ids,
            "schema_version": MANIFEST_V2_SCHEMA_VERSION,
        },
        request_id=request_id,
    )
    try:
        frozen = service_module._freeze_and_verify(session, version)
    except ApiError as exc:
        version.state = DatasetVersionState.FAILED.value
        version.error_code = exc.code.value
        version.error_detail = exc.message[:2000]
        session.flush()
        record_audit(
            session,
            actor=actor,
            action="dataset_version.failed",
            entity_type="dataset_version",
            entity_id=version.id,
            after={"error_code": exc.code.value},
            request_id=request_id,
        )
        return version
    service_module._set_state(version, DatasetVersionState.READY)
    session.flush()
    record_audit(
        session,
        actor=actor,
        action="dataset_version.ready",
        entity_type="dataset_version",
        entity_id=version.id,
        after={
            "manifest_sha256": frozen.manifest_sha256,
            "episode_count": frozen.episode_count,
            "total_frames": frozen.total_frames,
            "roundtrip": frozen.roundtrip.as_dict(),
            "schema_version": MANIFEST_V2_SCHEMA_VERSION,
        },
        request_id=request_id,
    )
    del SourceImport
    return version


# Convenience re-export for tests/other modules that need the raw table schema.
def read_aligned_parquet(path: Path):
    return read_parquet_bytes(Path(path).read_bytes())


__all__ = [
    "CANONICAL_PREVIEW_HEIGHT",
    "CANONICAL_PREVIEW_WIDTH",
    "CanonicalV2ExportEpisode",
    "ROUNDTRIP_MAX_ABS_ERROR",
    "ROUNDTRIP_MIN_SAMPLES",
    "V2ExpectedEpisode",
    "V2ExpectedSample",
    "V2ExportResult",
    "V2RoundTripResult",
    "build_export_v2",
    "build_v2_features",
    "create_v2_version",
    "episode_manifest_entry",
    "episodes_are_v2",
    "export_canonical_v2_episodes",
    "freeze_manifest_v2",
    "is_canonical_v2_episode",
    "materialize_export_episode",
    "materialize_export_episodes",
    "validate_v2_episode_eligible",
    "validate_v2_membership",
    "verify_build_v2",
    "verify_export_v2",
]
