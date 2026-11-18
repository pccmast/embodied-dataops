"""Official-API LeRobot v3 import adapter (SPEC 5.3).

Hard rules implemented here:

* Episode boundaries come **only** from the official ``lerobot`` API metadata
  (``meta/episodes``); filenames are never inspected to infer boundaries.  When
  the official API is not importable the adapter fails with
  ``LEROBOT_API_UNAVAILABLE`` instead of silently degrading to a private parser.
* ``meta/info.json`` must exist, parse, and explicitly declare codebase v3.
* The upload ``mapping`` fixes ``camera_key`` / ``state_key`` / ``action_key`` /
  ``robot_profile``.  One camera may be auto-selected; more than one camera
  without an explicit ``camera_key`` is ``MAPPING_REQUIRED`` (never guessed).
* ``state_key``/``action_key`` default to ``observation.state`` / ``action``;
  ``robot_profile`` is mandatory and must be ``arm6_joint_position_v1``.
  Without a provenance statement for units/semantics the episode imports as
  ``UNVERIFIED_SEMANTICS`` and can only preview, never publish.
* LeRobot rows are already aligned: the 10 Hz grid copies the nearest **complete
  row** (absolute error <= 50 ms) so state/action/frame stay one temporal unit.
  Duplicate/non-monotonic timestamps are rejected with evidence; dims other than
  6D are recorded as ``DIMENSION_MISMATCH`` and never truncated or padded.

Memory behaviour: episode metadata and the ``timestamp`` / state / action
columns are read column-wise from the official Arrow table; RGB frames are
streamed in one sequential pass only for the grid positions that survive
alignment, so a long episode never materializes thousands of frames at once.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops import SAMPLE_INTERVAL_NS, TARGET_FPS
from embodied_dataops.canonical.contract import (
    ACTION_DIM,
    ROBOT_JOINT_NAMES,
    ROBOT_PROFILE,
    STATE_DIM,
    SampleRow,
)
from embodied_dataops.config import get_settings
from embodied_dataops.domain.enums import EpisodeType, Severity, SourceType
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.imports.base import EpisodeSource, ImportContext, ParsedSource, SourceParser
from embodied_dataops.imports.common import (
    diagnostic,
    has_structural,
    merge_diagnostics,
)

STATE_KEY_DEFAULT = "observation.state"
ACTION_KEY_DEFAULT = "action"
CAMERA_PREFIX = "observation.images."
SEMANTICS_SIDECAR = "meta/dataops_semantics.json"
SOURCE_SIDECAR = "meta/dataops_source.json"

#: Nearest-neighbour tolerance for the 10 Hz grid (SPEC 5.3.4).
ROW_TOLERANCE_NS = 50_000_000
MAX_EPISODE_DURATION_NS = 600 * 1_000_000_000
_MAX_INFO_BYTES = 4 * 1024 * 1024
_MAX_SIDECAR_BYTES = 256 * 1024


# ---------------------------------------------------------------------------
# official API import
# ---------------------------------------------------------------------------


def _import_lerobot_dataset_class() -> type:
    """Import the official ``LeRobotDataset`` or raise a stable error code."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
    except Exception as exc:  # ImportError or a broken optional extra
        raise ApiError(
            ErrorCode.LEROBOT_API_UNAVAILABLE,
            "The official lerobot API is not importable: install the pinned "
            "lerobot==0.4.4 extra (`uv sync --extra lerobot`). The importer refuses "
            "to guess LeRobot v3 layout from filenames.",
            details={"exception": type(exc).__name__},
        ) from exc
    return LeRobotDataset


def lerobot_api_available() -> bool:
    try:
        _import_lerobot_dataset_class()
    except ApiError:
        return False
    return True


# ---------------------------------------------------------------------------
# meta/info.json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LerobotInfo:
    """Validated subset of ``meta/info.json`` (v3)."""

    codebase_version: str
    fps: float
    features: dict[str, Any]
    robot_type: str = ""
    total_episodes: int | None = None
    total_frames: int | None = None
    total_tasks: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def camera_keys(self) -> list[str]:
        return sorted(key for key in self.features if key.startswith(CAMERA_PREFIX))

    def feature(self, key: str) -> dict[str, Any] | None:
        value = self.features.get(key)
        return value if isinstance(value, dict) else None

    def feature_dim(self, key: str) -> int | None:
        feature = self.feature(key)
        if not feature:
            return None
        shape = feature.get("shape")
        if isinstance(shape, (list, tuple)) and shape:
            try:
                return int(shape[0])
            except (TypeError, ValueError):
                return None
        return None

    def as_summary(self) -> dict[str, Any]:
        return {
            "codebase_version": self.codebase_version,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": self.total_tasks,
            "cameras": self.camera_keys,
            "features": sorted(self.features),
        }


def declared_codebase_major(value: Any) -> int | None:
    """Extract the major version from ``v3.0`` / ``3.0`` / ``3`` declarations."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    match = re.match(r"^\s*v?(\d+)", value.strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_info_payload(payload: Any) -> LerobotInfo:
    """Validate a parsed ``meta/info.json`` payload (SPEC 5.3.1)."""
    if not isinstance(payload, dict) or not payload:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            "meta/info.json is missing or is not a JSON object",
        )
    declared = None
    for key in ("codebase_version", "dataset_version", "version"):
        if payload.get(key) is not None:
            declared = payload.get(key)
            break
    major = declared_codebase_major(declared)
    if major != 3:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            "meta/info.json does not declare LeRobot codebase v3",
            details={"declared": declared, "supported": "v3"},
        )
    features = payload.get("features")
    if not isinstance(features, dict) or not features:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            "meta/info.json does not declare any features",
        )
    raw_fps = payload.get("fps")
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            "meta/info.json does not declare a numeric fps",
            details={"fps": raw_fps},
        ) from None
    if not math.isfinite(fps) or fps <= 0:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            "meta/info.json declares a non-positive fps",
            details={"fps": raw_fps},
        )

    def _int_or_none(key: str) -> int | None:
        value = payload.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return LerobotInfo(
        codebase_version=str(declared),
        fps=fps,
        features=features,
        robot_type=str(payload.get("robot_type") or ""),
        total_episodes=_int_or_none("total_episodes"),
        total_frames=_int_or_none("total_frames"),
        total_tasks=_int_or_none("total_tasks"),
        raw=payload,
    )


# ---------------------------------------------------------------------------
# mapping rules (camera/state/action keys + robot profile semantics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedMapping:
    camera_key: str
    state_key: str
    action_key: str
    robot_profile: str
    semantics_verified: bool
    semantics_evidence: str
    cameras_available: list[str]
    state_dim: int | None
    action_dim: int | None
    auto_selected_camera: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_key": self.camera_key,
            "state_key": self.state_key,
            "action_key": self.action_key,
            "robot_profile": self.robot_profile,
            "semantics_verified": self.semantics_verified,
            "semantics_evidence": self.semantics_evidence,
            "cameras_available": self.cameras_available,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "auto_selected_camera": self.auto_selected_camera,
        }


def semantics_evidence_from(payload: Any) -> str | None:
    """Return a provenance statement if it proves the arm6 profile semantics."""
    if not isinstance(payload, dict):
        return None
    profile = str(payload.get("robot_profile") or "").strip()
    if profile and profile != ROBOT_PROFILE:
        return None
    state_units = str(payload.get("state_units") or payload.get("units") or "").strip().lower()
    if state_units not in {"rad", "radian", "radians"}:
        return None
    action_semantics = " ".join(
        str(payload.get(key) or "")
        for key in ("action_semantics", "action_units", "action_meaning")
    ).lower()
    if not any(
        token in action_semantics
        for token in ("target_joint_position", "joint_position_target", "target_position", "target joint")
    ):
        return None
    evidence = str(payload.get("evidence") or payload.get("description") or "").strip()
    if not evidence:
        return None
    return (
        f"{evidence} (profile={profile or ROBOT_PROFILE}, state_units={state_units}, "
        f"action_semantics={action_semantics.strip()})"
    )


def resolve_mapping(
    info: LerobotInfo,
    mapping: dict[str, Any] | None,
    *,
    semantics_sidecar: Any = None,
) -> ResolvedMapping:
    """Apply SPEC 5.3.2/5.3.3 mapping rules; never guess a camera."""
    mapping = dict(mapping or {})
    cameras = info.camera_keys
    requested_camera = mapping.get("camera_key")
    auto_selected = False
    if requested_camera:
        camera_key = str(requested_camera)
        if camera_key not in info.features:
            raise ApiError(
                ErrorCode.INVALID_MAPPING,
                "mapping.camera_key is not a feature of this dataset",
                details={"camera_key": camera_key, "available_cameras": cameras},
            )
        if camera_key not in cameras:
            raise ApiError(
                ErrorCode.INVALID_MAPPING,
                "mapping.camera_key must be an observation.images.* feature",
                details={"camera_key": camera_key, "available_cameras": cameras},
            )
    elif len(cameras) == 1:
        camera_key = cameras[0]
        auto_selected = True
    elif not cameras:
        raise ApiError(
            ErrorCode.MAPPING_REQUIRED,
            "the dataset declares no observation.images.* camera feature",
            details={"available_cameras": []},
        )
    else:
        raise ApiError(
            ErrorCode.MAPPING_REQUIRED,
            "the dataset has more than one camera; mapping.camera_key is required",
            details={"available_cameras": cameras},
        )

    state_key = str(mapping.get("state_key") or STATE_KEY_DEFAULT)
    if state_key not in info.features:
        raise ApiError(
            ErrorCode.INVALID_MAPPING,
            "mapping.state_key is not a feature of this dataset",
            details={"state_key": state_key, "features": sorted(info.features)},
        )
    action_key = str(mapping.get("action_key") or ACTION_KEY_DEFAULT)
    if action_key not in info.features:
        raise ApiError(
            ErrorCode.INVALID_MAPPING,
            "mapping.action_key is not a feature of this dataset",
            details={"action_key": action_key, "features": sorted(info.features)},
        )

    robot_profile = str(mapping.get("robot_profile") or "").strip()
    if not robot_profile:
        raise ApiError(
            ErrorCode.MAPPING_REQUIRED,
            f"mapping.robot_profile is required for LeRobot v3 imports ({ROBOT_PROFILE})",
            details={"supported_profiles": [ROBOT_PROFILE]},
        )
    if robot_profile != ROBOT_PROFILE:
        raise ApiError(
            ErrorCode.INVALID_MAPPING,
            "only robot_profile arm6_joint_position_v1 is supported in v1",
            details={"robot_profile": robot_profile, "supported_profiles": [ROBOT_PROFILE]},
        )

    evidence = semantics_evidence_from(semantics_sidecar)
    if evidence is None and mapping.get("semantics_evidence"):
        raw_evidence = str(mapping.get("semantics_evidence")).strip()
        if raw_evidence:
            evidence = f"upload mapping evidence: {raw_evidence}"
    return ResolvedMapping(
        camera_key=camera_key,
        state_key=state_key,
        action_key=action_key,
        robot_profile=robot_profile,
        semantics_verified=bool(evidence),
        semantics_evidence=evidence or "",
        cameras_available=cameras,
        state_dim=info.feature_dim(state_key),
        action_dim=info.feature_dim(action_key),
        auto_selected_camera=auto_selected,
    )


# ---------------------------------------------------------------------------
# archive safety preflight (SPEC 5.1/12.1, F08)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveInspection:
    entries: int
    total_bytes: int
    info_payload: Any
    semantics_payload: Any
    source_payload: Any
    dataset_prefix: str
    member_names: list[str] = field(default_factory=list, repr=False)


def _member_parts(name: str) -> list[str] | None:
    normalised = name.replace("\\", "/")
    if normalised.startswith("/"):
        return None
    if len(normalised) > 1 and normalised[1] == ":":
        return None
    parts = [part for part in normalised.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return None
    return parts


def _is_gzip(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo, limit: int) -> bytes:
    if member.size > limit:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            "a LeRobot metadata file inside the archive is too large",
            details={"limit_bytes": limit},
        )
    handle = archive.extractfile(member)
    if handle is None:  # pragma: no cover - unreadable member
        raise ApiError(ErrorCode.UNSUPPORTED_ARCHIVE_FORMAT, "archive member is not readable")
    with handle:
        return handle.read(limit + 1)[:limit]


def _loads_member(raw: bytes, what: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            f"{what} is not valid JSON",
            details={"exception": type(exc).__name__},
        ) from exc


def inspect_lerobot_archive(path: str | Path) -> ArchiveInspection:
    """Reject unsafe/oversized archives and read ``meta/info.json`` without unpacking.

    Same rules as :func:`embodied_dataops.processing.files.safe_extract_tar`
    (absolute paths, ``..``, links, device nodes, entry/size limits) applied
    before a single byte is written anywhere.
    """
    archive_path = Path(path)
    settings = get_settings()
    if not _is_gzip(archive_path):
        raise ApiError(
            ErrorCode.UNSUPPORTED_ARCHIVE_FORMAT,
            "LeRobot v3 input must be a .tar.gz archive (gzip magic bytes not found)",
        )
    info_member: tuple[str, tarfile.TarInfo] | None = None
    semantics_member: tuple[str, tarfile.TarInfo] | None = None
    source_member: tuple[str, tarfile.TarInfo] | None = None
    entries = 0
    total_bytes = 0
    dataset_prefix: str | None = None
    names: list[str] = []
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                entries += 1
                if entries > settings.max_tar_entries:
                    raise ApiError(
                        ErrorCode.ARCHIVE_TOO_LARGE,
                        "archive contains too many entries",
                        details={"limit": settings.max_tar_entries},
                    )
                parts = _member_parts(member.name)
                if parts is None:
                    raise ApiError(
                        ErrorCode.UNSAFE_ARCHIVE,
                        "unsafe path in archive",
                        details={
                            "rule": "absolute_path_or_traversal",
                            "member": str(member.name)[:200],
                        },
                    )
                if not parts:
                    continue
                names.append("/".join(parts))
                if member.issym() or member.islnk():
                    raise ApiError(
                        ErrorCode.UNSAFE_ARCHIVE,
                        "links are not allowed in archives",
                        details={"rule": "link", "member": "/".join(parts)[:200]},
                    )
                if member.isdev() or member.isfifo():
                    raise ApiError(
                        ErrorCode.UNSAFE_ARCHIVE,
                        "device nodes are not allowed in archives",
                        details={"rule": "device_node", "member": "/".join(parts)[:200]},
                    )
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ApiError(
                        ErrorCode.UNSAFE_ARCHIVE,
                        "unsupported archive member type",
                        details={"rule": "member_type", "member": "/".join(parts)[:200]},
                    )
                total_bytes += member.size
                if total_bytes > settings.max_unpacked_bytes:
                    raise ApiError(
                        ErrorCode.ARCHIVE_TOO_LARGE,
                        "expanded archive exceeds the size limit",
                        details={"limit": settings.max_unpacked_bytes},
                    )
                normalised = "/".join(parts)
                for candidate, suffix in (
                    ("info", "meta/info.json"),
                    ("semantics", SEMANTICS_SIDECAR),
                    ("source", SOURCE_SIDECAR),
                ):
                    if normalised == suffix or normalised.endswith("/" + suffix):
                        prefix = normalised[: -len(suffix)].rstrip("/")
                        if prefix.count("/") > 2:
                            continue
                        if candidate == "info":
                            if info_member is None:
                                info_member = (prefix, member)
                                dataset_prefix = prefix
                        elif suffix == SEMANTICS_SIDECAR:
                            if semantics_member is None and prefix == dataset_prefix:
                                semantics_member = (prefix, member)
                        else:
                            if source_member is None and prefix == dataset_prefix:
                                source_member = (prefix, member)
    except ApiError:
        raise
    except tarfile.TarError as exc:
        raise ApiError(
            ErrorCode.UNSUPPORTED_ARCHIVE_FORMAT,
            "archive is not a valid .tar.gz",
            details={"exception": type(exc).__name__},
        ) from exc
    if info_member is None:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            "dataset archive does not contain meta/info.json",
        )
    with tarfile.open(archive_path, mode="r:gz") as archive:
        info_payload = _loads_member(_read_member(archive, info_member[1], _MAX_INFO_BYTES), "meta/info.json")
        semantics_payload = (
            _loads_member(
                _read_member(archive, semantics_member[1], _MAX_SIDECAR_BYTES),
                SEMANTICS_SIDECAR,
            )
            if semantics_member is not None
            else None
        )
        source_payload = (
            _loads_member(_read_member(archive, source_member[1], _MAX_SIDECAR_BYTES), SOURCE_SIDECAR)
            if source_member is not None
            else None
        )
    return ArchiveInspection(
        entries=entries,
        total_bytes=total_bytes,
        info_payload=info_payload,
        semantics_payload=semantics_payload,
        source_payload=source_payload,
        dataset_prefix=info_member[0] if info_member else "",
        member_names=names[:5000],
    )


def resolve_dataset_root(directory: str | Path) -> Path:
    """Find the directory that owns ``meta/info.json`` (max one nesting level)."""
    root = Path(directory)
    if (root / "meta" / "info.json").is_file():
        return root
    candidates: list[Path] = []
    try:
        children = [child for child in root.iterdir() if child.is_dir()]
    except OSError as exc:
        raise ApiError(ErrorCode.LEROBOT_METADATA_INVALID, "dataset root is not readable") from exc
    for child in children:
        if child.name in {"__MACOSX", ".git"}:
            continue
        if (child / "meta" / "info.json").is_file():
            candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            "dataset root does not contain meta/info.json",
        )
    raise ApiError(
        ErrorCode.LEROBOT_METADATA_INVALID,
        "dataset archive contains more than one candidate root (meta/info.json)",
        details={"candidates": [candidate.name for candidate in candidates[:10]]},
    )


def _read_json_file(path: Path, limit: int, what: str) -> Any:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > limit:
        raise ApiError(
            ErrorCode.LEROBOT_METADATA_INVALID,
            f"{what} is too large",
            details={"limit_bytes": limit},
        )
    return _loads_member(path.read_bytes(), what)


# ---------------------------------------------------------------------------
# per-episode metadata / aligned read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LerobotEpisodeMeta:
    episode_index: int
    length: int
    tasks: list[str]
    from_index: int | None = None
    to_index: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def task_label(self) -> str:
        return "; ".join(self.tasks) if self.tasks else ""


@dataclass(frozen=True)
class LerobotEpisodeArrays:
    """Column-wise source arrays of one episode (no images decoded)."""

    timestamps_ns: list[int]
    states: list[list[float]]
    actions: list[list[float]]
    from_index: int
    to_index: int
    state_dim: int
    action_dim: int
    task_label: str
    backend: str


@dataclass
class AlignedLerobotEpisode:
    """One episode after the 10 Hz nearest-complete-row alignment."""

    episode_index: int
    rows: list[SampleRow] | None
    alignment: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    normalization_valid: bool
    state_dim: int
    action_dim: int
    joint_names: list[str]
    task_label: str
    duration_ns: int
    source_fps: float
    semantics_verified: bool
    semantics_evidence: str
    source_row_count: int
    source_duration_ns: int
    camera_key: str
    state_key: str
    action_key: str
    robot_profile: str
    #: source-frame indexes (episode-relative) selected for every output row
    selected_source_indexes: list[int] = field(default_factory=list)
    #: opaque reference used by :meth:`LerobotV3Parser.iter_aligned_frames`
    source_reference: dict[str, Any] = field(default_factory=dict, repr=False)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def to_float_list(value: Any) -> list[float]:
    """Flatten a tensor/array/list row value into ``list[float]``."""
    if value is None:
        return []
    array = _to_numpy(value)
    try:
        return [float(item) for item in array.reshape(-1).tolist()]
    except (TypeError, ValueError):
        return []


def to_rgb_uint8(value: Any) -> np.ndarray:
    """Convert an official-dataset image value to ``HxWx3 uint8`` RGB."""
    array = _to_numpy(value)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "dataset image feature is not an RGB frame",
            details={"shape": list(array.shape)},
        )
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    elif array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            finite = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            if float(finite.max(initial=0.0)) <= 1.0 + 1e-6:
                finite = finite * 255.0
            array = np.clip(np.rint(finite), 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_records(value: Any) -> list[dict[str, Any]] | None:
    """Normalize a pandas DataFrame / HF Dataset / pyarrow table / list of rows."""
    if value is None:
        return None
    try:
        records = value.to_dict("records")
    except (TypeError, ValueError, AttributeError):
        records = None
    if isinstance(records, list):
        return [dict(row) for row in records]
    try:
        rows = value.to_pylist()
    except (AttributeError, TypeError):
        rows = None
    if isinstance(rows, list):
        return [dict(row) for row in rows]
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return [dict(row) for row in value]
    try:
        rows = [dict(row) for row in value]
    except (TypeError, ValueError):
        return None
    return rows or None


def _row_int(row: dict[str, Any], candidates: Sequence[str]) -> int | None:
    for key in candidates:
        if key in row and row[key] is not None:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                continue
    return None


def _row_tasks(row: dict[str, Any]) -> list[str]:
    for key in ("tasks", "task", "task_label"):
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, str):
            return [value]
        try:
            return [str(item) for item in value]
        except TypeError:
            return [str(value)]
    return []


def normalize_episode_rows(records: Sequence[dict[str, Any]]) -> list[LerobotEpisodeMeta]:
    """Turn official episode-metadata records into stable episode descriptors."""
    episodes: list[LerobotEpisodeMeta] = []
    for row in records:
        index = _row_int(row, ("episode_index", "episode_id"))
        length = _row_int(row, ("length", "num_frames", "frames"))
        if index is None or length is None:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "official episode metadata is missing episode_index/length",
                details={"columns": sorted(str(key) for key in row)},
            )
        episodes.append(
            LerobotEpisodeMeta(
                episode_index=index,
                length=length,
                tasks=_row_tasks(row),
                from_index=_row_int(
                    row,
                    (
                        "dataset_from_index",
                        "from_index",
                        "data/dataset_from_index",
                        "meta/dataset_from_index",
                    ),
                ),
                to_index=_row_int(
                    row,
                    (
                        "dataset_to_index",
                        "to_index",
                        "data/dataset_to_index",
                        "meta/dataset_to_index",
                    ),
                ),
                raw=dict(row),
            )
        )
    episodes.sort(key=lambda item: item.episode_index)
    return episodes


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------


class LerobotV3Parser(SourceParser):
    """LeRobot v3 adapter driven exclusively by the official ``lerobot`` API."""

    source_type = SourceType.LEROBOT_V3

    # -- source materialisation ------------------------------------------------

    def _materialize_root(self, ctx: ImportContext) -> Path:
        source = Path(ctx.source_path)
        if source.is_dir():
            return resolve_dataset_root(source)
        target = Path(ctx.workdir) / "lerobot_source"
        if (target / "meta" / "info.json").is_file():
            return target
        inspect_lerobot_archive(source)  # safety/limits/format before writing anything
        from embodied_dataops.processing.files import safe_extract_tar

        target.mkdir(parents=True, exist_ok=True)
        safe_extract_tar(source, target)
        return resolve_dataset_root(target)

    def _read_info(self, root: Path) -> LerobotInfo:
        payload = _read_json_file(root / "meta" / "info.json", _MAX_INFO_BYTES, "meta/info.json")
        if payload is None:
            raise ApiError(ErrorCode.LEROBOT_METADATA_INVALID, "meta/info.json is missing")
        return parse_info_payload(payload)

    def _read_sidecars(self, root: Path) -> tuple[Any, Any]:
        semantics = None
        source = None
        try:
            if (root / SEMANTICS_SIDECAR).is_file():
                semantics = _read_json_file(root / SEMANTICS_SIDECAR, _MAX_SIDECAR_BYTES, SEMANTICS_SIDECAR)
            if (root / SOURCE_SIDECAR).is_file():
                source = _read_json_file(root / SOURCE_SIDECAR, _MAX_SIDECAR_BYTES, SOURCE_SIDECAR)
        except ApiError:
            raise
        return semantics, source

    # -- SourceParser interface ------------------------------------------------

    def validate(self, ctx: ImportContext) -> dict[str, Any]:
        source = Path(ctx.source_path)
        if source.is_dir():
            root = resolve_dataset_root(source)
            info = self._read_info(root)
            semantics, _source = self._read_sidecars(root)
            archive_meta: dict[str, Any] = {"kind": "directory"}
        else:
            inspection = inspect_lerobot_archive(source)
            info = parse_info_payload(inspection.info_payload)
            semantics = inspection.semantics_payload
            archive_meta = {
                "kind": "tar.gz",
                "entries": inspection.entries,
                "unpacked_bytes": inspection.total_bytes,
                "dataset_prefix": inspection.dataset_prefix,
            }
        resolved = resolve_mapping(info, ctx.mapping, semantics_sidecar=semantics)
        return {
            "format": "lerobot-v3",
            "archive": archive_meta,
            "info": info.as_summary(),
            "resolved_mapping": resolved.as_dict(),
            "episode_count_declared": info.total_episodes,
            "total_frames_declared": info.total_frames,
            "required_api": "lerobot==0.4.4",
            "api_available": lerobot_api_available(),
        }

    def parse(self, ctx: ImportContext) -> ParsedSource:
        root = self._materialize_root(ctx)
        info = self._read_info(root)
        semantics, source_sidecar = self._read_sidecars(root)
        resolved = resolve_mapping(info, ctx.mapping, semantics_sidecar=semantics)
        dataset = self._open_dataset(ctx, root, source_sidecar)
        episodes = self._enumerate_episodes(dataset, info)
        if not episodes:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "the official LeRobot API enumerated no Episodes",
            )
        warnings: list[str] = []
        if info.total_episodes is not None and info.total_episodes != len(episodes):
            warnings.append(
                f"meta/info.json declares {info.total_episodes} episodes but the official API "
                f"enumerated {len(episodes)}"
            )
        sources: list[EpisodeSource] = []
        for meta in episodes:
            duration_ns = int(round(meta.length / info.fps * 1_000_000_000))
            diagnostics: list[dict[str, Any]] = []
            if resolved.state_dim is not None and resolved.state_dim != STATE_DIM:
                diagnostics.append(
                    diagnostic(
                        "DIMENSION_MISMATCH",
                        Severity.ERROR,
                        f"observation.state is {resolved.state_dim}D; the v1 profile requires {STATE_DIM}D",
                        details={"state_dim": resolved.state_dim, "required": STATE_DIM},
                    )
                )
            if resolved.action_dim is not None and resolved.action_dim != ACTION_DIM:
                diagnostics.append(
                    diagnostic(
                        "DIMENSION_MISMATCH",
                        Severity.ERROR,
                        f"action is {resolved.action_dim}D; the v1 profile requires {ACTION_DIM}D",
                        details={"action_dim": resolved.action_dim, "required": ACTION_DIM},
                    )
                )
            if not resolved.semantics_verified:
                diagnostics.append(_unverified_semantics_diagnostic(resolved))
            sources.append(
                EpisodeSource(
                    source_episode_index=meta.episode_index,
                    episode_type=EpisodeType.ROBOT_EPISODE,
                    task_label=meta.task_label,
                    duration_ns=duration_ns,
                    payload={
                        "camera_key": resolved.camera_key,
                        "state_key": resolved.state_key,
                        "action_key": resolved.action_key,
                        "robot_profile": resolved.robot_profile,
                        "source_fps": info.fps,
                        "length": meta.length,
                        "dataset_from_index": meta.from_index,
                        "dataset_to_index": meta.to_index,
                        "semantics_verified": resolved.semantics_verified,
                        "state_dim": resolved.state_dim,
                        "action_dim": resolved.action_dim,
                    },
                    diagnostics=diagnostics,
                    normalization_valid=not has_structural(diagnostics),
                )
            )
        metadata = {
            "codebase_version": info.codebase_version,
            "fps": info.fps,
            "robot_type": info.robot_type,
            "episode_count": len(sources),
            "resolved_mapping": resolved.as_dict(),
            "enumerated_by": "official_lerobot_api",
            "source_sidecar": _safe_sidecar_summary(source_sidecar),
        }
        return ParsedSource(episodes=sources, metadata=metadata, warnings=warnings)

    # -- official dataset access ----------------------------------------------

    def _resolve_repo_id(self, ctx: ImportContext, root: Path) -> str:
        if ctx.source_uri and "://" not in ctx.source_uri and "/" in ctx.source_uri:
            return ctx.source_uri.strip().strip("/")
        sidecar = _read_json_file(root / SOURCE_SIDECAR, _MAX_SIDECAR_BYTES, SOURCE_SIDECAR)
        if isinstance(sidecar, dict) and sidecar.get("repo_id"):
            return str(sidecar["repo_id"])
        fingerprint = hashlib.sha256(str(ctx.source_sha256 or "local").encode()).hexdigest()[:12]
        return f"dataops/import-{fingerprint}"

    def _open_dataset(self, ctx: ImportContext, root: Path, source_sidecar: Any = None) -> Any:
        dataset_class = _import_lerobot_dataset_class()
        repo_id = self._resolve_repo_id(ctx, root)
        kwargs: dict[str, Any] = {"repo_id": repo_id, "root": str(root)}
        try:
            signature = inspect.signature(dataset_class.__init__)
            parameters = signature.parameters
            if "download_videos" in parameters:
                kwargs["download_videos"] = False
        except (TypeError, ValueError):  # pragma: no cover - builtin/extension class
            parameters = {}
        try:
            return dataset_class(**kwargs)
        except TypeError:
            try:
                return dataset_class(repo_id, str(root))
            except Exception as exc:  # pragma: no cover - API drift
                raise ApiError(
                    ErrorCode.EPISODE_ENUMERATION_FAILED,
                    "the official LeRobot API could not open this dataset root",
                    details={"exception": type(exc).__name__},
                ) from exc
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "the official LeRobot API could not open this dataset root",
                details={"exception": type(exc).__name__},
            ) from exc

    def _enumerate_episodes(self, dataset: Any, info: LerobotInfo) -> list[LerobotEpisodeMeta]:
        meta = getattr(dataset, "meta", None)
        episodes_object = getattr(meta, "episodes", None) if meta is not None else None
        records = _as_records(episodes_object)
        if not records:
            # Explicit degraded path: some 0.4.x builds expose the legacy index
            # only; we accept it but label the source so evidence is honest.
            records = self._legacy_episode_records(dataset, info)
        if not records:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "official LeRobot API exposes no episode metadata; refusing to guess "
                "episode boundaries from file names",
            )
        episodes = normalize_episode_rows(records)
        # Fill from/to ranges from the legacy index when the v3 metadata omits them.
        legacy = getattr(dataset, "episode_data_index", None)
        if legacy is not None:
            try:
                froms = getattr(legacy, "from_index", None)
                tos = getattr(legacy, "to_index", None)
                if froms is not None and tos is not None:
                    from_values = _to_numpy(froms).reshape(-1).tolist()
                    to_values = _to_numpy(tos).reshape(-1).tolist()
                    episodes = [
                        LerobotEpisodeMeta(
                            episode_index=item.episode_index,
                            length=item.length,
                            tasks=item.tasks,
                            from_index=item.from_index
                            if item.from_index is not None
                            else int(from_values[item.episode_index]),
                            to_index=item.to_index
                            if item.to_index is not None
                            else int(to_values[item.episode_index]),
                            raw=item.raw,
                        )
                        for item in episodes
                    ]
            except (IndexError, TypeError, ValueError):  # pragma: no cover - API drift
                pass
        return episodes

    def _legacy_episode_records(self, dataset: Any, info: LerobotInfo) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        index_container = getattr(dataset, "episode_data_index", None)
        if index_container is not None:
            try:
                froms = _to_numpy(index_container.from_index).reshape(-1).tolist()
                tos = _to_numpy(index_container.to_index).reshape(-1).tolist()
            except (AttributeError, TypeError, ValueError):
                return []
            for index, (start, end) in enumerate(zip(froms, tos, strict=False)):
                records.append(
                    {
                        "episode_index": index,
                        "length": int(end) - int(start),
                        "tasks": [],
                        "dataset_from_index": int(start),
                        "dataset_to_index": int(end),
                    }
                )
            return records
        return []

    # -- episode reading -------------------------------------------------------

    def _episode_meta(self, ctx: ImportContext, dataset: Any, info: LerobotInfo, episode_index: int) -> LerobotEpisodeMeta:
        episodes = self._enumerate_episodes(dataset, info)
        for meta in episodes:
            if meta.episode_index == episode_index:
                return meta
        raise ApiError(
            ErrorCode.EPISODE_ENUMERATION_FAILED,
            "requested episode index is not present in the official metadata",
            details={"episode_index": episode_index, "available": [m.episode_index for m in episodes[:50]]},
        )

    def _episode_arrays(
        self,
        dataset: Any,
        meta: LerobotEpisodeMeta,
        info: LerobotInfo,
        state_key: str,
        action_key: str,
    ) -> LerobotEpisodeArrays:
        from_index = meta.from_index
        to_index = meta.to_index
        if from_index is None and to_index is None:
            episodes = self._enumerate_episodes(dataset, info)
            from_index = sum(item.length for item in episodes if item.episode_index < meta.episode_index)
            to_index = from_index + meta.length
        if from_index is None or to_index is None or to_index <= from_index:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "episode metadata has no usable dataset row range",
                details={"episode_index": meta.episode_index},
            )
        timestamps: list[int] = []
        states: list[list[float]] = []
        actions: list[list[float]] = []
        backend = "row"
        hf = getattr(dataset, "hf_dataset", None)
        if hf is None and hasattr(dataset, "_ensure_hf_dataset_loaded"):
            # The official reader loads the Arrow table lazily; forcing that load
            # lets us read timestamp/state/action column-wise without decoding any
            # video frame (important for long episodes and for corrupt-timestamp
            # diagnostics, which must not trip video seeking).
            try:
                dataset._ensure_hf_dataset_loaded()
            except Exception:  # pragma: no cover - API drift
                pass
            hf = getattr(dataset, "hf_dataset", None)
        columns = _columns_from_hf_dataset(hf, from_index, to_index, state_key, action_key)
        if columns is not None:
            raw_times, raw_states, raw_actions = columns
            try:
                times_seconds = [float(item) for item in raw_times]
            except (TypeError, ValueError):
                times_seconds = []
            if len(times_seconds) == to_index - from_index and len(raw_states) == len(times_seconds):
                snapped = snap_lerobot_timestamps(times_seconds, info.fps)
                origin = snapped[0] if snapped else 0
                timestamps = [value - origin for value in snapped]
                states = [to_float_list(row) for row in raw_states]
                actions = [to_float_list(row) for row in raw_actions]
                backend = "arrow-columns"
        if not timestamps:
            for global_index in range(from_index, to_index):
                row = dataset[global_index]
                raw = float(_scalar(row.get("timestamp", 0.0)))
                timestamps.append(int(round(raw * 1_000_000_000)))
                states.append(to_float_list(row.get(state_key)))
                actions.append(to_float_list(row.get(action_key)))
            if timestamps:
                timestamps = snap_lerobot_timestamps(
                    [value / 1_000_000_000 for value in timestamps], info.fps
                )
                origin = timestamps[0]
                timestamps = [value - origin for value in timestamps]
        if not timestamps:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "episode contains no frames according to the official API",
                details={"episode_index": meta.episode_index},
            )
        state_dim = len(states[0]) if states else 0
        action_dim = len(actions[0]) if actions else 0
        return LerobotEpisodeArrays(
            timestamps_ns=timestamps,
            states=states,
            actions=actions,
            from_index=from_index,
            to_index=to_index,
            state_dim=state_dim,
            action_dim=action_dim,
            task_label=meta.task_label,
            backend=backend,
        )

    def read_aligned_episode(self, ctx: ImportContext, episode_index: int) -> AlignedLerobotEpisode:
        """Read one episode and align it to the 10 Hz grid (streaming images)."""
        root = self._materialize_root(ctx)
        info = self._read_info(root)
        semantics, source_sidecar = self._read_sidecars(root)
        resolved = resolve_mapping(info, ctx.mapping, semantics_sidecar=semantics)
        dataset = self._open_dataset(ctx, root, source_sidecar)
        meta = self._episode_meta(ctx, dataset, info, episode_index)
        arrays = self._episode_arrays(dataset, meta, info, resolved.state_key, resolved.action_key)

        diagnostics: list[dict[str, Any]] = []
        if not resolved.semantics_verified:
            diagnostics.append(_unverified_semantics_diagnostic(resolved))
        if arrays.state_dim != STATE_DIM:
            diagnostics.append(
                diagnostic(
                    "DIMENSION_MISMATCH",
                    Severity.ERROR,
                    f"observation.state is {arrays.state_dim}D; the v1 profile requires {STATE_DIM}D",
                    details={"state_dim": arrays.state_dim, "required": STATE_DIM},
                )
            )
        if arrays.action_dim != ACTION_DIM:
            diagnostics.append(
                diagnostic(
                    "DIMENSION_MISMATCH",
                    Severity.ERROR,
                    f"action is {arrays.action_dim}D; the v1 profile requires {ACTION_DIM}D",
                    details={"action_dim": arrays.action_dim, "required": ACTION_DIM},
                )
            )
        monotonic = [
            index
            for index in range(1, len(arrays.timestamps_ns))
            if arrays.timestamps_ns[index] <= arrays.timestamps_ns[index - 1]
        ]
        if monotonic:
            diagnostics.append(
                diagnostic(
                    "TIMESTAMP_NON_MONOTONIC",
                    Severity.ERROR,
                    "LeRobot episode timestamps are not strictly increasing",
                    details={
                        "violation_indexes": monotonic[:20],
                        "violation_count": len(monotonic),
                    },
                )
            )
        nonfinite = _nonfinite_count(arrays.states) + _nonfinite_count(arrays.actions)
        if nonfinite:
            diagnostics.append(
                diagnostic(
                    "NONFINITE_VALUE",
                    Severity.ERROR,
                    "state/action contains NaN or infinite values",
                    details={"count": nonfinite},
                )
            )
        source_duration_ns = arrays.timestamps_ns[-1] if arrays.timestamps_ns else 0
        if source_duration_ns > MAX_EPISODE_DURATION_NS:
            diagnostics.append(
                diagnostic(
                    "DURATION_EXCEEDED",
                    Severity.ERROR,
                    "episode exceeds the 600 s limit",
                    details={"duration_ns": source_duration_ns, "limit_ns": MAX_EPISODE_DURATION_NS},
                )
            )

        alignment_payload: dict[str, Any] = {
            "grid_points": 0,
            "valid_sample_count": 0,
            "missing_sample_count": 0,
            "max_abs_sync_error_ms": 0.0,
            "p95_abs_sync_error_ms": 0.0,
            "alignment_unit": "nearest_complete_row",
            "tolerance_ns": ROW_TOLERANCE_NS,
            "source_row_count": len(arrays.timestamps_ns),
            "arrays_backend": arrays.backend,
        }
        selected: list[int] = []
        rows: list[SampleRow] | None = None
        blocking = has_structural(diagnostics)
        if not blocking:
            selection = _select_rows(arrays.timestamps_ns)
            selected = selection["source_indexes"]
            alignment_payload.update(selection["alignment"])
            alignment_payload["grid_points"] = selection["alignment"]["valid_sample_count"] + selection[
                "alignment"
            ]["missing_sample_count"]
            missing = selection["missing_grid_t_ns"]
            if missing:
                diagnostics.append(
                    diagnostic(
                        "SYNC_MISSING_FRAME",
                        Severity.ERROR,
                        "one or more 10 Hz grid points have no LeRobot row within 50 ms",
                        details={"missing_grid_t_ns": missing[:200], "missing_count": len(missing)},
                    )
                )
            else:
                rows = [
                    SampleRow(
                        sample_index=index,
                        t_ns=index * SAMPLE_INTERVAL_NS,
                        frame_index=index,
                        state=arrays.states[source_index],
                        action=arrays.actions[source_index],
                        state_source_t_ns=arrays.timestamps_ns[source_index],
                        action_source_t_ns=arrays.timestamps_ns[source_index],
                    )
                    for index, source_index in enumerate(selected)
                ]
        blocking = has_structural(diagnostics)
        valid = not blocking and rows is not None
        duration_ns = (
            len(rows) * SAMPLE_INTERVAL_NS if valid and rows is not None else _estimate_duration(alignment_payload, arrays)
        )
        # The v1 profile's joint names are only claimable when *both* vectors are
        # the 6D arm6 profile; a 5D observation must not inherit joint_1..joint_6.
        joint_names = (
            list(ROBOT_JOINT_NAMES)
            if arrays.state_dim == STATE_DIM and arrays.action_dim == ACTION_DIM
            else []
        )
        return AlignedLerobotEpisode(
            episode_index=episode_index,
            rows=rows,
            alignment=alignment_payload,
            diagnostics=merge_diagnostics(diagnostics),
            normalization_valid=valid,
            state_dim=arrays.state_dim,
            action_dim=arrays.action_dim,
            joint_names=joint_names,
            task_label=arrays.task_label,
            duration_ns=duration_ns,
            source_fps=info.fps,
            semantics_verified=resolved.semantics_verified,
            semantics_evidence=resolved.semantics_evidence,
            source_row_count=len(arrays.timestamps_ns),
            source_duration_ns=source_duration_ns,
            camera_key=resolved.camera_key,
            state_key=resolved.state_key,
            action_key=resolved.action_key,
            robot_profile=resolved.robot_profile,
            selected_source_indexes=selected,
            source_reference={
                "root": str(root),
                "repo_id": self._resolve_repo_id(ctx, root),
                "from_index": arrays.from_index,
                "to_index": arrays.to_index,
                "camera_key": resolved.camera_key,
                "state_sidecar": bool(source_sidecar),
            },
        )

    def iter_aligned_frames(
        self, ctx: ImportContext, aligned: AlignedLerobotEpisode
    ) -> Iterator[np.ndarray]:
        """Yield source-resolution RGB frames in output-grid order (streaming)."""
        if not aligned.selected_source_indexes:
            return
        reference = aligned.source_reference
        dataset_class = _import_lerobot_dataset_class()
        kwargs: dict[str, Any] = {"repo_id": reference["repo_id"], "root": reference["root"]}
        try:
            if "download_videos" in inspect.signature(dataset_class.__init__).parameters:
                kwargs["download_videos"] = False
        except (TypeError, ValueError):  # pragma: no cover
            pass
        try:
            dataset = dataset_class(**kwargs)
        except TypeError:  # pragma: no cover - API drift
            try:
                dataset = dataset_class(reference["repo_id"], reference["root"])
            except Exception as exc:
                raise ApiError(
                    ErrorCode.EPISODE_ENUMERATION_FAILED,
                    "the official LeRobot API could not reopen this dataset root",
                    details={"exception": type(exc).__name__},
                ) from exc
        occurrences: dict[int, int] = {}
        for offset in aligned.selected_source_indexes:
            occurrences[offset] = occurrences.get(offset, 0) + 1
        from_index = reference["from_index"]
        to_index = reference["to_index"]
        needed = sorted(occurrences)
        position = 0
        for offset in range(to_index - from_index):
            if position >= len(needed):
                break
            if offset != needed[position]:
                continue
            position += 1
            row = dataset[from_index + offset]
            camera_value = row.get(reference["camera_key"])
            if camera_value is None:
                raise ApiError(
                    ErrorCode.FILE_DECODE_ERROR,
                    "LeRobot row is missing the mapped camera feature",
                    details={"camera_key": reference["camera_key"]},
                )
            frame = to_rgb_uint8(camera_value)
            for _ in range(occurrences[offset]):
                yield frame


# ---------------------------------------------------------------------------
# alignment helpers
# ---------------------------------------------------------------------------


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return float(value)


#: Maximum float32 round-trip error accepted when snapping LeRobot timestamps
#: back onto the exact ``frame_index / fps`` rational time (SPEC 5.3).
LEROBOT_TIME_SNAP_EPSILON_NS = 1_000_000


def snap_lerobot_timestamps(seconds: Sequence[float], fps: float) -> list[int]:
    """Convert LeRobot ``timestamp`` seconds to exact episode-relative nanoseconds.

    LeRobot stores ``timestamp = frame_index / fps`` as float32, so the 100th
    frame of a 10 fps episode reads ``9.8999996`` s; a naive floor would drop
    the last 10 Hz grid point (99 instead of 100 samples).  Reconstructing the
    rational time from the declared ``fps`` is metadata-based, not guessed:
    values that are not on the source grid (or duplicated/inverted corrupt
    timestamps, which map to the same frame position) are preserved so the
    TIMESTAMP_NON_MONOTONIC rule still fires.
    """
    snapped: list[int] = []
    for value in seconds:
        raw_ns = int(round(float(value) * 1_000_000_000))
        if fps <= 0:
            snapped.append(raw_ns)
            continue
        position = round(float(value) * fps)
        exact_ns = int(round(position * 1_000_000_000 / fps))
        if abs(exact_ns - raw_ns) <= LEROBOT_TIME_SNAP_EPSILON_NS:
            snapped.append(exact_ns)
        else:
            snapped.append(raw_ns)
    return snapped


def _unverified_semantics_diagnostic(resolved: ResolvedMapping) -> dict[str, Any]:
    return diagnostic(
        "UNVERIFIED_SEMANTICS",
        Severity.ERROR,
        "no provenance statement proves arm6_joint_position_v1 units/semantics; "
        "the episode may be previewed but must not be published",
        details={
            "robot_profile": resolved.robot_profile,
            "required_evidence": (
                "meta/dataops_semantics.json (robot_profile, state_units=rad, "
                "action_semantics=target_joint_position, evidence) or mapping.semantics_evidence"
            ),
        },
        structural=False,
    )


def _nonfinite_count(rows: Sequence[Sequence[float]]) -> int:
    if not rows:
        return 0
    total = 0
    for row in rows:
        array = np.asarray(row, dtype=np.float64)
        if array.size:
            total += int(np.count_nonzero(~np.isfinite(array)))
    return total


def _columns_from_hf_dataset(
    hf_dataset: Any, from_index: int, to_index: int, state_key: str, action_key: str
) -> tuple[list[Any], list[Any], list[Any]] | None:
    """Read timestamp/state/action columns without decoding any video frame."""
    if hf_dataset is None:
        return None
    subset = None
    try:
        select = getattr(hf_dataset, "select", None)
        if callable(select):
            subset = select(range(from_index, to_index))
    except (TypeError, ValueError):
        subset = None
    source = subset if subset is not None else hf_dataset
    try:
        raw_times = _column_values(source, "timestamp")
        raw_states = _column_values(source, state_key)
        raw_actions = _column_values(source, action_key)
    except Exception:  # arbitrary API drift: caller falls back to rows
        return None
    if raw_times is None or raw_states is None or raw_actions is None:
        return None
    if subset is None:
        return (
            raw_times[from_index:to_index],
            raw_states[from_index:to_index],
            raw_actions[from_index:to_index],
        )
    return raw_times, raw_states, raw_actions


def _column_values(source: Any, column: str) -> list[Any] | None:
    try:
        if hasattr(source, "select_columns"):
            return source.select_columns([column])[column]
        if hasattr(source, "column_names") and column in source.column_names:
            return source[column]
        return None
    except (KeyError, ValueError, TypeError, IndexError):
        return None


def _select_rows(timestamps_ns: Sequence[int]) -> dict[str, Any]:
    """Nearest complete row for each 10 Hz grid point, never selecting the future.

    SPEC 5.3 treats the LeRobot row as one aligned temporal unit, so state,
    action and frame are always copied together.  When the nearest row is after
    the target time we prefer an at-or-before row that is still within tolerance,
    which keeps ``action_source_t_ns <= t_ns`` on the canonical contract; if only
    a future row qualifies we keep it (rows are atomic) and flag it explicitly.
    """
    import bisect

    if not timestamps_ns:
        return {
            "source_indexes": [],
            "alignment": {
                "valid_sample_count": 0,
                "missing_sample_count": 0,
                "max_abs_sync_error_ms": 0.0,
                "p95_abs_sync_error_ms": 0.0,
            },
            "missing_grid_t_ns": [],
        }
    times = list(timestamps_ns)
    last_k = int(times[-1] // SAMPLE_INTERVAL_NS)
    selected: list[int] = []
    errors: list[int] = []
    missing: list[int] = []
    future_rows = 0
    for k in range(last_k + 1):
        target = k * SAMPLE_INTERVAL_NS
        position = bisect.bisect_left(times, target)
        candidates: list[int] = []
        if position < len(times):
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        if not candidates:  # pragma: no cover - position can always be 0
            missing.append(target)
            continue
        candidates.sort(key=lambda idx: (abs(times[idx] - target), times[idx]))
        chosen: int | None = None
        for candidate in candidates:
            if times[candidate] <= target and abs(times[candidate] - target) <= ROW_TOLERANCE_NS:
                chosen = candidate
                break
        if chosen is None:
            for candidate in candidates:
                if abs(times[candidate] - target) <= ROW_TOLERANCE_NS:
                    chosen = candidate
                    future_rows += 1
                    break
        if chosen is None:
            missing.append(target)
            continue
        selected.append(chosen)
        errors.append(abs(times[chosen] - target))
    max_error = max(errors) if errors else 0
    ordered = sorted(errors)
    p95 = float(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]) if ordered else 0.0
    return {
        "source_indexes": selected,
        "alignment": {
            "valid_sample_count": len(selected),
            "missing_sample_count": len(missing),
            "max_abs_sync_error_ms": max_error / 1e6,
            "p95_abs_sync_error_ms": p95 / 1e6,
            "selected_after_target_count": future_rows,
            "grid_points": len(selected) + len(missing),
        },
        "missing_grid_t_ns": missing,
    }


def _estimate_duration(alignment: dict[str, Any], arrays: LerobotEpisodeArrays) -> int:
    count = int(alignment.get("grid_points") or 0)
    if count:
        return count * SAMPLE_INTERVAL_NS
    return arrays.timestamps_ns[-1] if arrays.timestamps_ns else 0


def _safe_sidecar_summary(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return {
        "keys": sorted(str(key) for key in payload),
        "repo_id": payload.get("repo_id"),
        "hub_revision": payload.get("revision") or payload.get("hub_revision"),
        "notes": payload.get("notes"),
    }


__all__ = [
    "ACTION_DIM",
    "AlignedLerobotEpisode",
    "ArchiveInspection",
    "CAMERA_PREFIX",
    "LerobotEpisodeArrays",
    "LerobotEpisodeMeta",
    "LerobotInfo",
    "LerobotV3Parser",
    "ROW_TOLERANCE_NS",
    "ResolvedMapping",
    "SEMANTICS_SIDECAR",
    "SOURCE_SIDECAR",
    "STATE_DIM",
    "TARGET_FPS",
    "declared_codebase_major",
    "inspect_lerobot_archive",
    "lerobot_api_available",
    "normalize_episode_rows",
    "parse_info_payload",
    "resolve_dataset_root",
    "resolve_mapping",
    "semantics_evidence_from",
    "snap_lerobot_timestamps",
    "to_float_list",
    "to_rgb_uint8",
]
