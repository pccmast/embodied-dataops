"""TrainingBundle 的准备与原子发布（EDO-TRAIN-PREP-1.1 §11）。

职责边界：

* **准备**：来源快照 → 分组 → 划分 → 有效窗口 → train-only 统计 → 冻结 processor；
* **发布**：在目标同文件系统的临时兄弟目录里写完、逐文件算摘要、算出两种身份
  （``preparation_id`` 语义身份 / ``artifact_id`` 字节身份），再原子 ``rename``；
* **不覆盖**：目标已存在且内容不同就报错，相同内容可复用但验证记录独立。

两条不能含糊的规则：

1. ``manifest.json`` 不把自身摘要递归纳入；``artifact_id`` 由 manifest_core 加
   排序后的文件清单算出，因此 ``manifest.json`` / ``artifact.json`` 本身不进清单。
2. 内部报告只引用已经算好的 ``preparation_id``，不嵌入 ``artifact_id``——
   否则文件 SHA 与身份会互相依赖，永远算不出稳定值。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.contract import (
    ResolvedRecipe,
    TrainingRecipe,
    VerifiedSource,
    canonical_json_bytes,
    resolve_recipe,
    resolved_recipe_sha256,
    semantic_sha256,
)
from embodied_dataops.training.processors import (
    build_act_processors,
    processor_config_document,
)
from embodied_dataops.training.split import (
    EpisodeRecord,
    SourceGroup,
    build_source_groups,
    build_split,
)
from embodied_dataops.training.stats import (
    FrameSource,
    TrainStats,
    fit_train_stats,
    stats_from_document,
)
from embodied_dataops.training.windows import (
    EpisodeFrames,
    Window,
    WindowIndex,
    build_window_index,
)

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "IMPLEMENTATION_SOURCE",
    "LoadedBundle",
    "PreparedBundle",
    "SourcePackage",
    "digest_paths",
    "load_bundle",
    "load_processor_step",
    "load_source_package",
    "prepare_bundle",
    "save_processor_step",
    "snapshot_source_document",
    "verify_source_root",
]

BUNDLE_SCHEMA_VERSION = "training-bundle/1.0"
SOURCE_SNAPSHOT_SCHEMA = "training-source-snapshot/1.0"
DATASET_MANIFEST_SCHEMA = "dataset-manifest/2.0"
SOURCE_GROUPS_SCHEMA = "training-source-groups/1.0"
ASSIGNMENTS_SCHEMA = "training-group-assignments/1.0"
PROVENANCE_SCHEMA = "training-provenance/1.0"
PREPARATION_REPORT_SCHEMA = "training-preparation-report/1.0"
WINDOW_INDEX_SCHEMA = "training-windows/1.0"
ROW_DIGEST_ALGORITHM = "edo-episode-row-digest-v1"
IMPLEMENTATION_SOURCE = "edo-train-prep/1.1"

WINDOW_COLUMNS: tuple[str, ...] = (
    "split",
    "source_episode_id",
    "official_episode_index",
    "anchor_frame_index",
    "action_frame_indices",
    "source_row_digest",
)
REJECTED_COLUMNS: tuple[str, ...] = (
    "split",
    "source_episode_id",
    "anchor_frame_index",
    "reason",
    "frame_index",
)

_IMAGE_PREFIX = "observation.images."
#: 计算 ``artifact_id`` 时排除的身份文件：它们互相引用，纳入清单会形成环。
_IDENTITY_FILES = frozenset({"manifest.json", "artifact.json"})


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FileDigest:
    path: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.size}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_paths(root: Path, *, exclude: Sequence[str] = ()) -> tuple[FileDigest, ...]:
    """对一个目录做逐文件摘要；路径一律用 POSIX 相对形式，便于跨平台复算。"""
    root = Path(root)
    if not root.is_dir():
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "cannot digest a directory that does not exist",
            details={"path": str(root)},
        )
    skip = set(exclude)
    entries: list[FileDigest] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in skip:
            continue
        entries.append(FileDigest(path=relative, sha256=_sha256_file(path), size=path.stat().st_size))
    return tuple(entries)


def _write_json(path: Path, payload: Any) -> None:
    """写人类可读的确定性 JSON。

    产物文件的字节编码与语义身份是两件事：``preparation_id`` 走
    ``canonical_json_bytes``（float→hex），文件本身用缩进排序 JSON，
    这样审阅者能直接读，重编码只改变 ``artifact_id`` 而不改变语义身份。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "a required JSON input cannot be read",
            details={"path": str(path), "error": type(exc).__name__},
        ) from exc


def _safe_relative(base: Path, relative: str) -> Path:
    """解析伴生文件路径并拒绝目录穿越（SPEC §9 明确要求）。"""
    candidate = (base / relative).resolve()
    base_resolved = base.resolve()
    if base_resolved != candidate and base_resolved not in candidate.parents:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "a referenced input escapes the source directory",
            details={"base": str(base_resolved), "reference": relative},
        )
    return candidate


def implementation_info() -> dict[str, Any]:
    """实现版本：Git revision / dirty 以及实际锁定的依赖版本。"""
    revision = "unknown"
    dirty: bool | None = None
    try:
        revision = subprocess.run(  # noqa: S603 - 固定参数，无 shell
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip() or "unknown"
        status = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 无 git 的环境
        dirty = None

    dependencies: dict[str, str] = {}
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    for name in ("lerobot", "torch", "torchvision", "torchcodec", "numpy", "pyarrow", "pydantic"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - 可选依赖
            continue
    return {
        "source": IMPLEMENTATION_SOURCE,
        "revision": revision,
        "dirty": dirty,
        "dependencies": dict(sorted(dependencies.items())),
    }


# ---------------------------------------------------------------------------
# 来源
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceEpisode:
    episode_id: str
    official_episode_index: int
    source_sha256: str
    session_key: str | None = None
    canonical_sha256: str | None = None

    def as_record(self) -> EpisodeRecord:
        return EpisodeRecord(
            episode_id=self.episode_id,
            source_sha256=self.source_sha256,
            session_key=self.session_key,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "official_episode_index": int(self.official_episode_index),
            "source_sha256": self.source_sha256,
            "session_key": self.session_key,
            "canonical_sha256": self.canonical_sha256,
        }


@dataclass(frozen=True)
class SourcePackage:
    """一个已核验来源：冻结 manifest、导出根目录、派生特征与文件清单。"""

    dataset_version_id: str
    dataset_slug: str
    manifest: dict[str, Any]
    manifest_sha256: str
    source_root: Path
    base_dir: Path
    data_origin: str
    source_verification: str
    license_id: str
    profile_id: str
    profile_revision: int
    aligned_fps: int
    camera_keys: tuple[str, ...]
    state_dim: int
    action_dim: int
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_unit: str
    action_unit: str
    action_semantics: str
    image_shapes: dict[str, tuple[int, int, int]]
    episodes: tuple[SourceEpisode, ...]
    files: tuple[FileDigest, ...] = ()
    quality_evidence: dict[str, Any] = field(default_factory=dict)

    def verified_source(self) -> VerifiedSource:
        return VerifiedSource(
            dataset_version_id=self.dataset_version_id,
            manifest_sha256=self.manifest_sha256,
            profile_id=self.profile_id,
            profile_revision=self.profile_revision,
            camera_keys=self.camera_keys,
            aligned_fps=self.aligned_fps,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            state_names=self.state_names,
            action_names=self.action_names,
            state_unit=self.state_unit,
            action_unit=self.action_unit,
            action_semantics=self.action_semantics,
            image_shapes=dict(self.image_shapes),
            episode_ids=tuple(episode.episode_id for episode in self.episodes),
            data_origin=self.data_origin,
            source_verification=self.source_verification,
        )

    def digest_document(self) -> dict[str, Any]:
        """进入 preparation_id 的来源身份：只含内容身份，不含路径与时间。"""
        return {
            "dataset_version_id": self.dataset_version_id,
            "dataset_slug": self.dataset_slug,
            "manifest_sha256": self.manifest_sha256,
            "data_origin": self.data_origin,
            "source_verification": self.source_verification,
            "license": self.license_id,
            "profile": {"profile_id": self.profile_id, "profile_revision": self.profile_revision},
            "aligned_fps": self.aligned_fps,
            "cameras": list(self.camera_keys),
            "features": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "state_names": list(self.state_names),
                "action_names": list(self.action_names),
                "state_unit": self.state_unit,
                "action_unit": self.action_unit,
                "action_semantics": self.action_semantics,
                "image_shapes": {name: list(shape) for name, shape in sorted(self.image_shapes.items())},
            },
            "episodes": [episode.as_dict() for episode in self.episodes],
            "source_files": [entry.as_dict() for entry in self.files],
        }


def _dataset_meta(root: Path) -> dict[str, Any]:
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.is_file():
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "the source root is not an exported LeRobot dataset (meta/info.json is missing)",
            details={"source_root": str(root)},
        )
    info = _read_json(info_path)
    semantics: dict[str, Any] = {}
    semantics_path = meta_dir / "dataops_semantics.json"
    if semantics_path.is_file():
        semantics = _read_json(semantics_path)
    origin: dict[str, Any] = {}
    origin_path = meta_dir / "dataops_source.json"
    if origin_path.is_file():
        origin = _read_json(origin_path)
    if not isinstance(info, dict):
        raise ApiError(ErrorCode.LEROBOT_METADATA_INVALID, "meta/info.json is not a JSON object")
    return {"info": info, "semantics": semantics or {}, "origin": origin or {}}


def _camera_short_name(key: str) -> str:
    return key[len(_IMAGE_PREFIX) :] if key.startswith(_IMAGE_PREFIX) else key


def _vector_dim(features: Mapping[str, Any], key: str) -> int | None:
    """一维特征（state/action）的维度；形状不是一维就不认。"""
    entry = features.get(key) or {}
    shape = entry.get("shape")
    if not shape:
        return None
    values = [int(size) for size in shape]
    if len(values) != 1 or values[0] <= 0:
        return None
    return values[0]


def _image_shape(features: Mapping[str, Any], key: str) -> tuple[int, int, int] | None:
    """图像特征的 ``(C, H, W)``；只接受官方声明的三维形状。"""
    entry = features.get(key) or {}
    shape = entry.get("shape")
    if not shape:
        return None
    values = [int(size) for size in shape]
    if len(values) == 3 and all(size > 0 for size in values):
        return (values[0], values[1], values[2])
    return None


def snapshot_source_document(
    *,
    dataset_root: str | Path,
    dataset_version_id: str,
    license_id: str = "unknown",
    data_origin: str | None = None,
) -> dict[str, Any]:
    """从导出的 LeRobot 数据集派生一份 ``training-source-snapshot/1.0``。

    只用于 synthetic/无 PG 的场景：它记录的是「这份导出目录长什么样」，
    不能替代 ``--source-mode published`` 的数据库发布核验。
    """
    root = Path(dataset_root)
    meta = _dataset_meta(root)
    info = meta["info"]
    semantics = meta["semantics"]
    origin = meta["origin"]
    features = info.get("features") or {}
    cameras = sorted(
        _camera_short_name(key) for key in features if key.startswith(_IMAGE_PREFIX)
    )
    episode_count = int(info.get("total_episodes") or 0)

    episodes: list[dict[str, Any]] = []
    for index in range(episode_count):
        episodes.append(
            {
                "episode_id": f"episode_{index}",
                "official_episode_index": index,
                # 没有会话证据时，用 episode 身份本身作为摘要：它只能保证
                # 「不同 episode 不会被误连」，不能声称同源分组已完成。
                "source_sha256": hashlib.sha256(
                    f"{dataset_version_id}|{index}".encode()
                ).hexdigest(),
                "session_key": None,
                "canonical_sha256": None,
            }
        )

    return {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "dataset_version_id": dataset_version_id,
        "dataset_slug": str(origin.get("repo_id") or info.get("repo_id") or root.name),
        "data_origin": str(data_origin or origin.get("data_origin") or semantics.get("data_origin") or "unknown"),
        "license": license_id,
        "profile": {
            "profile_id": str(semantics.get("robot_profile") or ""),
            "profile_revision": int(semantics.get("profile_revision") or 0),
        },
        "aligned_fps": int(info.get("fps") or 0),
        "cameras": cameras,
        "semantics": {
            "state_names": list(semantics.get("joint_order") or []),
            "action_names": list(semantics.get("joint_order") or []),
            "state_unit": str(semantics.get("state_units") or ""),
            "action_unit": str(semantics.get("action_units") or ""),
            "action_semantics": str(semantics.get("action_semantics") or ""),
        },
        "features": {
            key: {"shape": list(value.get("shape") or [])}
            for key, value in sorted(features.items())
            if key in (semantics.get("state_key") or "observation.state", "action")
        },
        "episodes": episodes,
    }


def _verify_published(dataset_version_id: str | None, manifest_sha256: str) -> dict[str, Any]:
    """只读核验 DatasetVersion 确为 PUBLISHED 且摘要一致。

    数据库不可用是 **BLOCKED**（退出码 3），不是"文件里写着 PUBLISHED 就算"。
    """
    if not dataset_version_id:
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "published mode needs the platform dataset version id; file content alone is not an identity",
            details={"hint": "pass --dataset-version-id or use --source-mode synthetic-fixture"},
        )
    try:
        from sqlalchemy import select  # noqa: PLC0415

        from embodied_dataops.db.models import DatasetVersion  # noqa: PLC0415
        from embodied_dataops.db.session import get_session_factory  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise ApiError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "the database layer is not importable in this environment",
            details={"error": type(exc).__name__},
        ) from exc

    try:
        session = get_session_factory()()
        try:
            row = session.execute(
                select(DatasetVersion).where(DatasetVersion.id == dataset_version_id)
            ).scalar_one_or_none()
            snapshot = {
                "id": getattr(row, "id", None),
                "dataset_id": getattr(row, "dataset_id", None),
                "version": getattr(row, "version", None),
                "state": getattr(row, "state", None),
                "manifest_asset_id": getattr(row, "manifest_asset_id", None),
                "export_asset_id": getattr(row, "export_asset_id", None),
                "manifest_sha256": getattr(row, "manifest_sha256", None),
                "export_kind": getattr(row, "export_kind", None),
                "published_at": str(getattr(row, "published_at", None)),
            }
            verified_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        finally:
            session.close()
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001 - 任何连接/查询失败都是 BLOCKED
        raise ApiError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "cannot reach PostgreSQL to verify the published dataset version",
            details={"error": type(exc).__name__, "dataset_version_id": dataset_version_id},
        ) from exc

    if row is None:
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "the dataset version does not exist",
            details={"dataset_version_id": dataset_version_id},
        )
    if row.state != "PUBLISHED":
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "the dataset version is not PUBLISHED",
            details={"dataset_version_id": dataset_version_id, "state": row.state},
        )
    if row.manifest_sha256 != manifest_sha256:
        raise ApiError(
            ErrorCode.SOURCE_HASH_MISMATCH,
            "the manifest digest does not match the published version",
            details={
                "dataset_version_id": dataset_version_id,
                "expected": row.manifest_sha256,
                "observed": manifest_sha256,
            },
        )
    snapshot["verified_at"] = verified_at
    return snapshot


def _manifest_action_semantics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """从冻结 manifest 的 episode 记录里取动作语义。

    published 模式的权威语义在平台发布时冻结的 manifest 里，而导出包不保证携带
    ``meta/dataops_semantics.json``。只有在所有 episode 都声明同一个非空语义时才
    采用它；一旦出现分歧或全部为空，就保持空白让下游照常拒绝——不替数据挑一个。
    """
    episodes = [ep for ep in (manifest.get("episodes") or []) if isinstance(ep, Mapping)]
    declared = {str(ep.get("action_semantics") or "").strip() for ep in episodes}
    declared.discard("")
    if len(declared) != 1 or not episodes:
        return {}
    derived: dict[str, Any] = {"action_semantics": declared.pop()}
    action_names = [str(name) for name in (episodes[0].get("action_names") or [])]
    if action_names:
        derived["action_names"] = action_names
    return derived


def load_source_package(
    *,
    source_root: str | Path,
    manifest_path: str | Path | None = None,
    source_mode: str = "synthetic-fixture",
    dataset_version_id: str | None = None,
    license_id: str | None = None,
    digest_files: bool = True,
) -> SourcePackage:
    """核验并载入来源。``published`` 必须带 manifest 并完成数据库核验。"""
    root = Path(source_root)
    if not root.is_dir():
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "the source root does not exist",
            details={"source_root": str(root)},
        )
    meta = _dataset_meta(root)
    info = meta["info"]
    semantics = meta["semantics"]
    origin = meta["origin"]

    if manifest_path is None:
        if source_mode == "published":
            raise ApiError(
                ErrorCode.SOURCE_UNVERIFIED,
                "published mode requires --source-manifest; a dataset directory is not a publication claim",
            )
        manifest = snapshot_source_document(
            dataset_root=root,
            dataset_version_id=dataset_version_id or f"fixture:{root.name}",
            license_id=license_id or "unknown",
        )
        manifest_bytes = canonical_json_bytes(manifest)
        base_dir = root
    else:
        path = Path(manifest_path)
        if not path.is_file():
            raise ApiError(
                ErrorCode.SOURCE_UNVERIFIED,
                "the source manifest does not exist",
                details={"manifest": str(path)},
            )
        manifest_bytes = path.read_bytes()
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ApiError(
                ErrorCode.SOURCE_UNVERIFIED,
                "the source manifest is not valid JSON",
                details={"manifest": str(path), "error": type(exc).__name__},
            ) from exc
        if not isinstance(manifest, dict):
            raise ApiError(ErrorCode.SOURCE_UNVERIFIED, "the source manifest is not a JSON object")
        schema = manifest.get("schema_version")
        if schema not in (DATASET_MANIFEST_SCHEMA, SOURCE_SNAPSHOT_SCHEMA):
            raise ApiError(
                ErrorCode.SOURCE_UNVERIFIED,
                "unsupported source manifest schema_version",
                details={"schema_version": schema, "supported": [DATASET_MANIFEST_SCHEMA, SOURCE_SNAPSHOT_SCHEMA]},
            )
        base_dir = path.parent

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    if source_mode == "published" and not semantics.get("action_semantics"):
        # 导出包没有 sidecar 时回退到冻结 manifest；sidecar 里已有的值优先。
        derived = _manifest_action_semantics(manifest)
        if derived:
            semantics = {**derived, **semantics}

    features = info.get("features") or {}
    camera_keys = tuple(
        sorted(_camera_short_name(key) for key in features if key.startswith(_IMAGE_PREFIX))
    )
    if not camera_keys:
        raise ApiError(
            ErrorCode.FEATURE_INCOMPATIBLE,
            "the source has no camera feature; an ACT bundle needs at least one",
            details={"features": sorted(features)},
        )

    state_key = str(semantics.get("state_key") or "observation.state")
    action_key = str(semantics.get("action_key") or "action")
    state_dim = _vector_dim(features, state_key)
    action_dim = _vector_dim(features, action_key)
    if state_dim is None or action_dim is None:
        raise ApiError(
            ErrorCode.FEATURE_INCOMPATIBLE,
            "the source does not declare a state/action feature shape",
            details={"state_key": state_key, "action_key": action_key},
        )

    joint_names = tuple(str(name) for name in (semantics.get("joint_order") or []))
    state_names = joint_names if len(joint_names) == state_dim else tuple(
        f"joint_{index}" for index in range(state_dim)
    )
    action_names = state_names if len(state_names) == action_dim else tuple(
        f"action_{index}" for index in range(action_dim)
    )

    image_shapes: dict[str, tuple[int, int, int]] = {}
    for camera in camera_keys:
        shape = _image_shape(features, f"{_IMAGE_PREFIX}{camera}")
        if shape is None:
            raise ApiError(
                ErrorCode.FEATURE_INCOMPATIBLE,
                "a camera feature has no usable shape",
                details={"camera": camera},
            )
        image_shapes[camera] = shape

    profile = manifest.get("profile") or {}
    profile_id = str(profile.get("profile_id") or semantics.get("robot_profile") or "")
    profile_revision = int(profile.get("profile_revision") or semantics.get("profile_revision") or 0)
    if not profile_id:
        raise ApiError(
            ErrorCode.FEATURE_INCOMPATIBLE,
            "the source does not identify a robot profile revision",
            details={"source_root": str(root)},
        )

    aligned_fps = int(manifest.get("aligned_fps") or info.get("fps") or 0)
    if aligned_fps <= 0:
        raise ApiError(
            ErrorCode.FEATURE_INCOMPATIBLE,
            "the source has no positive aligned fps",
            details={"aligned_fps": aligned_fps},
        )

    data_origin = str(
        manifest.get("data_origin")
        or origin.get("data_origin")
        or semantics.get("data_origin")
        or "unknown"
    )
    license_id_resolved = str(manifest.get("license") or license_id or "unknown")
    dataset_slug = str(manifest.get("dataset_slug") or root.name)
    version_id = str(manifest.get("dataset_version_id") or dataset_version_id or f"fixture:{root.name}")

    episodes = _source_episodes(manifest, info, root)
    _cross_check_manifest(manifest, info, state_dim, action_dim, aligned_fps, camera_keys)

    if source_mode == "published":
        _verify_published(dataset_version_id or manifest.get("dataset_version_id"), manifest_sha256)
        source_verification = "PUBLISHED_VERIFIED"
    elif source_mode == "synthetic-fixture":
        if data_origin != "synthetic":
            raise ApiError(
                ErrorCode.SOURCE_UNVERIFIED,
                "synthetic-fixture mode only accepts data_origin=synthetic",
                details={"data_origin": data_origin},
            )
        source_verification = "FIXTURE_ONLY"
    else:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "unsupported source mode",
            details={"source_mode": source_mode, "supported": ["published", "synthetic-fixture"]},
        )

    files = digest_paths(root) if digest_files else ()

    return SourcePackage(
        dataset_version_id=version_id,
        dataset_slug=dataset_slug,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        source_root=root,
        base_dir=base_dir,
        data_origin=data_origin,
        source_verification=source_verification,
        license_id=license_id_resolved,
        profile_id=profile_id,
        profile_revision=profile_revision,
        aligned_fps=aligned_fps,
        camera_keys=camera_keys,
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=state_names,
        action_names=action_names,
        state_unit=str(semantics.get("state_units") or ""),
        action_unit=str(semantics.get("action_units") or ""),
        action_semantics=str(semantics.get("action_semantics") or ""),
        image_shapes=image_shapes,
        episodes=episodes,
        files=files,
        quality_evidence={
            "qc_policy_version": manifest.get("qc_policy_version"),
            "semantics_evidence": semantics.get("evidence"),
            "source_notes": origin.get("notes"),
        },
    )


def _source_episodes(
    manifest: Mapping[str, Any], info: Mapping[str, Any], root: Path
) -> tuple[SourceEpisode, ...]:
    total = int(info.get("total_episodes") or 0)
    entries = manifest.get("episodes")
    if not entries:
        if total <= 0:
            raise ApiError(ErrorCode.SOURCE_UNVERIFIED, "the source declares no episode")
        return tuple(
            SourceEpisode(
                episode_id=f"episode_{index}",
                official_episode_index=index,
                source_sha256=hashlib.sha256(f"{root.name}|{index}".encode()).hexdigest(),
            )
            for index in range(total)
        )
    if len(entries) != total:
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "the manifest episode count does not match the exported dataset",
            details={"manifest_episodes": len(entries), "dataset_episodes": total},
        )
    episodes: list[SourceEpisode] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries):
        episode_id = str(entry.get("episode_id") or "")
        if not episode_id:
            raise ApiError(
                ErrorCode.SOURCE_UNVERIFIED,
                "a manifest episode entry has no episode_id",
                details={"position": position},
            )
        if episode_id in seen:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the manifest lists an episode twice",
                details={"episode_id": episode_id},
            )
        seen.add(episode_id)
        source_sha = str(entry.get("source_sha256") or entry.get("canonical_sha256") or "")
        if len(source_sha) != 64:
            raise ApiError(
                ErrorCode.SOURCE_UNVERIFIED,
                "a manifest episode entry carries no source digest",
                details={"episode_id": episode_id},
            )
        episodes.append(
            SourceEpisode(
                episode_id=episode_id,
                official_episode_index=position,
                source_sha256=source_sha,
                session_key=entry.get("session_key"),
                canonical_sha256=entry.get("canonical_sha256"),
            )
        )
    return tuple(episodes)


def _cross_check_manifest(
    manifest: Mapping[str, Any],
    info: Mapping[str, Any],
    state_dim: int,
    action_dim: int,
    aligned_fps: int,
    camera_keys: Sequence[str],
) -> None:
    """manifest 与导出目录必须互相印证；不一致就是来源身份问题。"""
    entries = manifest.get("episodes") or []
    for entry in entries:
        if "aligned_fps" in entry and int(entry["aligned_fps"]) != aligned_fps:
            raise ApiError(
                ErrorCode.SOURCE_HASH_MISMATCH,
                "the manifest episode fps contradicts the exported dataset",
                details={"episode_id": entry.get("episode_id"), "fps": entry["aligned_fps"]},
            )
        for field_name, expected in (("state_dim", state_dim), ("action_dim", action_dim)):
            if field_name in entry and int(entry[field_name]) != expected:
                raise ApiError(
                    ErrorCode.SOURCE_HASH_MISMATCH,
                    f"the manifest episode {field_name} contradicts the exported dataset",
                    details={"episode_id": entry.get("episode_id"), field_name: entry[field_name]},
                )
        declared = entry.get("camera_keys")
        if declared:
            short = sorted(_camera_short_name(str(key)) for key in declared)
            if short != sorted(camera_keys):
                raise ApiError(
                    ErrorCode.SOURCE_HASH_MISMATCH,
                    "the manifest camera keys contradict the exported dataset",
                    details={"declared": short, "exported": list(camera_keys)},
                )
    if int(info.get("total_episodes") or 0) <= 0:
        raise ApiError(ErrorCode.EMPTY_VERSION, "the exported dataset has no episode")


# ---------------------------------------------------------------------------
# 数据扫描
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Scan:
    frames: tuple[EpisodeFrames, ...]
    frame_source: FrameSource
    reader: Any
    relative_by_position: dict[tuple[int, int], int]
    episodes_by_index: dict[int, str]


class _MatrixFrameSource:
    """按 ``(episode_id, frame_index)`` 从已读入的数值矩阵取原始值。

    只读 state/action 两列，**不触碰视频**：统计量拟合不需要像素，
    把它绑到解码上会让没有 FFmpeg 的环境连统计都算不出来。
    """

    def __init__(
        self,
        *,
        state_matrix: Any,
        action_matrix: Any,
        row_of: Mapping[tuple[int, int], int],
        episode_ids: Mapping[int, str],
    ) -> None:
        self._state = state_matrix
        self._action = action_matrix
        self._row_of = row_of
        self._episode_ids = episode_ids

    def _row(self, episode_id: str, frame_index: int) -> int:
        for official, name in self._episode_ids.items():
            if name == episode_id:
                key = (official, int(frame_index))
                if key in self._row_of:
                    return self._row_of[key]
                break
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "statistics were requested for a frame outside the scanned source",
            details={"episode_id": episode_id, "frame_index": int(frame_index)},
        )

    def state(self, episode_id: str, frame_index: int) -> list[float]:
        row = self._row(episode_id, frame_index)
        return [float(value) for value in self._state[row]]

    def action(self, episode_id: str, frame_index: int) -> list[float]:
        row = self._row(episode_id, frame_index)
        return [float(value) for value in self._action[row]]


def _scan_dataset(
    package: SourcePackage,
    *,
    state_key: str,
    action_key: str,
    tolerance_s: float,
) -> _Scan:
    import numpy as np  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    reader = LeRobotDataset(
        package.dataset_slug,
        root=package.source_root,
        download_videos=False,
    )
    hf = reader.hf_dataset

    episode_column = np.asarray(hf["episode_index"], dtype=np.int64)
    frame_column = np.asarray(hf["frame_index"], dtype=np.int64)
    timestamp_column = np.asarray(hf["timestamp"], dtype=np.float64)
    state_matrix = np.stack([np.asarray(value, dtype=np.float64) for value in hf[state_key]])
    action_matrix = np.stack([np.asarray(value, dtype=np.float64) for value in hf[action_key]])

    if state_matrix.shape[0] != episode_column.shape[0]:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "the exported columns have inconsistent lengths",
            details={"rows": int(episode_column.shape[0]), "states": int(state_matrix.shape[0])},
        )

    episode_ids = {episode.official_episode_index: episode.episode_id for episode in package.episodes}
    relative_by_position: dict[tuple[int, int], int] = {}
    rows_by_episode: dict[int, list[int]] = {}
    for row in range(episode_column.shape[0]):
        official = int(episode_column[row])
        frame = int(frame_column[row])
        relative_by_position[(official, frame)] = row
        rows_by_episode.setdefault(official, []).append(row)

    state_finite = np.isfinite(state_matrix).all(axis=1)
    action_finite = np.isfinite(action_matrix).all(axis=1)
    expected_ts = frame_column / float(package.aligned_fps)
    time_ok = np.abs(timestamp_column - expected_ts) <= max(float(tolerance_s), 1e-9)

    frames: list[EpisodeFrames] = []
    for episode in package.episodes:
        rows = rows_by_episode.get(episode.official_episode_index)
        if not rows:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "the manifest lists an episode that the exported dataset does not contain",
                details={
                    "episode_id": episode.episode_id,
                    "official_episode_index": episode.official_episode_index,
                },
            )
        invalid: dict[int, str] = {}
        digest = hashlib.sha256()
        digest.update(f"{ROW_DIGEST_ALGORITHM}|{episode.episode_id}\n".encode())
        row_frames = sorted(rows, key=lambda row: int(frame_column[row]))
        for row in row_frames:
            frame = int(frame_column[row])
            if not state_finite[row]:
                invalid[frame] = "NONFINITE_STATE"
            elif not action_finite[row]:
                invalid[frame] = "NONFINITE_ACTION"
            elif not time_ok[row]:
                invalid[frame] = "TIME_GAP"
            digest.update(f"{frame}|".encode())
            digest.update(state_matrix[row].astype("<f8").tobytes())
            digest.update(action_matrix[row].astype("<f8").tobytes())
            digest.update(b"\n")
        frames.append(
            EpisodeFrames(
                episode_id=episode.episode_id,
                official_episode_index=episode.official_episode_index,
                frame_count=len(row_frames),
                invalid_frames=invalid,
                row_digest=digest.hexdigest(),
            )
        )

    return _Scan(
        frames=tuple(frames),
        frame_source=_MatrixFrameSource(
            state_matrix=state_matrix,
            action_matrix=action_matrix,
            row_of=relative_by_position,
            episode_ids=episode_ids,
        ),
        reader=reader,
        relative_by_position=relative_by_position,
        episodes_by_index=episode_ids,
    )


def _probe_video(
    scan: _Scan,
    windows: Sequence[Window],
    *,
    failures: dict[int, str],
) -> dict[tuple[int, int], str]:
    """对每个窗口的 anchor 做一次真实解码，返回 ``(episode, frame) -> 原因``。

    覆盖所有候选窗口的 anchor（不是只试第一个 batch）。解码整体不可用时由调用方
    转成 BLOCKED，而不是把所有窗口悄悄判成不可用。
    """
    try:
        import torchcodec  # noqa: F401, PLC0415 - 只探测可用性
    except Exception as exc:  # noqa: BLE001 - 任何加载失败都是依赖不可用
        raise ApiError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "video decoding is unavailable in this environment (torchcodec/FFmpeg)",
            details={"error": type(exc).__name__, "remedy": "install FFmpeg shared libraries or pass --no-video-probe"},
        ) from exc

    decoded: set[tuple[int, int]] = set()
    rejected: dict[tuple[int, int], str] = {}
    for window in windows:
        key = (int(window.official_episode_index), int(window.anchor_frame_index))
        if key in decoded:
            continue
        decoded.add(key)
        relative = scan.relative_by_position.get(key)
        if relative is None:  # pragma: no cover - 扫描阶段已校验
            rejected[key] = "DECODE_ERROR"
            continue
        try:
            scan.reader[relative]
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 - 解码失败需要计数，不吞掉整批
            message = str(exc)
            if "torchcodec" in message or "FFmpeg" in message or "libav" in message:
                # 环境本身没有解码能力：这不是数据缺陷，不能让所有窗口变成拒绝样本。
                raise ApiError(
                    ErrorCode.DEPENDENCY_UNAVAILABLE,
                    "video decoding is unavailable: torchcodec cannot load its FFmpeg shared libraries",
                    details={
                        "error": type(exc).__name__,
                        "remedy": "install FFmpeg shared libraries or pass --no-video-probe",
                    },
                ) from exc
            failures[type(exc).__name__] = failures.get(type(exc).__name__, 0) + 1
            rejected[key] = "DECODE_ERROR"
    return rejected


# ---------------------------------------------------------------------------
# 发布
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreparedBundle:
    path: Path
    preparation_id: str
    artifact_id: str
    manifest: dict[str, Any]
    reused: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "bundle": str(self.path),
            "preparation_id": self.preparation_id,
            "artifact_id": self.artifact_id,
            "reused": self.reused,
            "verification_level": (self.manifest.get("verification") or {}).get("level"),
        }


def _window_semantic_digest(index: WindowIndex) -> str:
    """窗口索引的语义摘要：规范 JSONL 流（§11.1），与 parquet 编码无关。"""
    digest = hashlib.sha256()
    for window in index.windows:
        digest.update(canonical_json_bytes(window.as_dict()))
    return digest.hexdigest()


def _rejected_semantic_digest(index: WindowIndex) -> str:
    digest = hashlib.sha256()
    for rejected in index.rejected:
        digest.update(canonical_json_bytes(rejected.as_dict()))
    return digest.hexdigest()


def _write_windows_parquet(path: Path, index: WindowIndex) -> None:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pa.table(
        {
            "split": pa.array([w.split for w in index.windows], type=pa.string()),
            "source_episode_id": pa.array([w.source_episode_id for w in index.windows], type=pa.string()),
            "official_episode_index": pa.array(
                [w.official_episode_index for w in index.windows], type=pa.int64()
            ),
            "anchor_frame_index": pa.array([w.anchor_frame_index for w in index.windows], type=pa.int64()),
            "action_frame_indices": pa.array(
                [list(w.action_frame_indices) for w in index.windows], type=pa.list_(pa.int64())
            ),
            "source_row_digest": pa.array([w.source_row_digest for w in index.windows], type=pa.string()),
        },
        schema=pa.schema(
            [
                ("split", pa.string()),
                ("source_episode_id", pa.string()),
                ("official_episode_index", pa.int64()),
                ("anchor_frame_index", pa.int64()),
                ("action_frame_indices", pa.list_(pa.int64())),
                ("source_row_digest", pa.string()),
            ]
        ),
    )
    pq.write_table(table, path)


def _write_rejected_parquet(path: Path, index: WindowIndex) -> None:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pa.table(
        {
            "split": pa.array([r.split for r in index.rejected], type=pa.string()),
            "source_episode_id": pa.array([r.source_episode_id for r in index.rejected], type=pa.string()),
            "anchor_frame_index": pa.array([r.anchor_frame_index for r in index.rejected], type=pa.int64()),
            "reason": pa.array([r.reason for r in index.rejected], type=pa.string()),
            "frame_index": pa.array(
                [r.frame_index for r in index.rejected], type=pa.int64()
            ),
        },
        schema=pa.schema(
            [
                ("split", pa.string()),
                ("source_episode_id", pa.string()),
                ("anchor_frame_index", pa.int64()),
                ("reason", pa.string()),
                ("frame_index", pa.int64()),
            ]
        ),
    )
    pq.write_table(table, path)


def save_processor_step(step: Any, directory: Path, config_filename: str) -> None:
    """用官方 ``DataProcessorPipeline`` 把单个 step 落盘（配置 + safetensors 状态）。

    单个 ``ProcessorStep`` 没有 ``save_pretrained``；官方可重载的最小单位是 pipeline，
    所以这里包一层，保证重载走的是官方加载路径而不是自定义反序列化。
    """
    from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: PLC0415

    name = f"edo_act_{config_filename.rsplit('.', 1)[0]}"
    pipeline = DataProcessorPipeline(steps=[step], name=name)
    pipeline.save_pretrained(Path(directory), config_filename=config_filename)


def load_processor_step(directory: Path, config_filename: str) -> Any:
    """重载官方 step；用 identity 转换，因为调用方直接传 ``EnvTransition``。"""
    from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: PLC0415

    try:
        pipeline = DataProcessorPipeline.from_pretrained(
            Path(directory),
            config_filename=config_filename,
            to_transition=lambda data: data,
            to_output=lambda transition: transition,
        )
    except Exception as exc:  # noqa: BLE001 - 官方加载失败即产物不可用
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "a frozen processor cannot be reloaded from the bundle",
            details={
                "directory": str(directory),
                "config_filename": config_filename,
                "error": type(exc).__name__,
            },
        ) from exc

    steps = list(getattr(pipeline, "steps", ()) or ())
    if len(steps) != 1:
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "a frozen processor file does not contain exactly one step",
            details={"config_filename": config_filename, "steps": len(steps)},
        )
    return steps[0]


def _load_groups(package: SourcePackage, reference: str) -> tuple[SourceGroup, ...]:
    """读取冻结分组；不存在时按来源证据推导（并记下依据）。"""
    path = _safe_relative(package.base_dir, reference)
    if not path.is_file():
        path = _safe_relative(package.source_root, reference)
    if not path.is_file():
        return build_source_groups([episode.as_record() for episode in package.episodes])
    payload = _read_json(path)
    if payload.get("schema_version") != SOURCE_GROUPS_SCHEMA:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "source_groups.json has an unsupported schema_version",
            details={"schema_version": payload.get("schema_version"), "path": str(path)},
        )
    groups: list[SourceGroup] = []
    seen: set[str] = set()
    for entry in payload.get("groups") or []:
        episode_ids = tuple(sorted(str(value) for value in entry.get("episode_ids") or []))
        group_id = str(entry.get("group_id") or "")
        if not group_id or not episode_ids:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "source_groups.json has an empty group",
                details={"path": str(path), "group_id": group_id},
            )
        overlap = seen & set(episode_ids)
        if overlap:
            raise ApiError(
                ErrorCode.GROUP_LEAKAGE,
                "an episode belongs to more than one source group",
                details={"episodes": sorted(overlap)},
            )
        seen |= set(episode_ids)
        groups.append(
            SourceGroup(
                group_id=group_id,
                episode_ids=episode_ids,
                evidence=dict(entry.get("evidence") or {}),
            )
        )
    known = {episode.episode_id for episode in package.episodes}
    if seen != known:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "source_groups.json does not cover exactly the source episodes",
            details={
                "missing": sorted(known - seen),
                "unknown": sorted(seen - known),
                "path": str(path),
            },
        )
    return tuple(sorted(groups, key=lambda group: group.group_id))


def _load_assignments(package: SourcePackage, reference: str) -> dict[str, str]:
    path = _safe_relative(package.base_dir, reference)
    if not path.is_file():
        path = _safe_relative(package.source_root, reference)
    if not path.is_file():
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "explicit_group_assignment_v1 requires group_assignments.json",
            details={"reference": reference},
        )
    payload = _read_json(path)
    schema = payload.get("schema_version")
    if schema not in (ASSIGNMENTS_SCHEMA, None):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "group_assignments.json has an unsupported schema_version",
            details={"schema_version": schema, "path": str(path)},
        )
    assignments = payload.get("assignments") if "assignments" in payload else payload
    if not isinstance(assignments, Mapping):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "group_assignments.json is not a group→split mapping",
            details={"path": str(path)},
        )
    return {str(key): str(value) for key, value in assignments.items()}


def _prepare_report(
    *,
    status: str,
    level: str,
    preparation_id: str | None,
    error: ApiError | None,
    source: SourcePackage | None,
    window_index: WindowIndex | None = None,
    stats: TrainStats | None = None,
    extra: Mapping[str, Any] | None = None,
    timestamp: bool = True,
) -> dict[str, Any]:
    """构造 preparation 报告。

    ``timestamp=False`` 用于**bundle 内部**的副本：写进 bundle 的文件必须逐字节
    可复现，否则同一份输入每次构建都会得到不同的 ``artifact_id``，既无法复用
    也无法判断"内容是否相同"。运行时间只留在外部报告目录。
    """
    report: dict[str, Any] = {
        "schema_version": PREPARATION_REPORT_SCHEMA,
        "status": status,
        "verification_level": level,
        "preparation_id": preparation_id,
    }
    if timestamp:
        report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if source is not None:
        report["source"] = {
            "dataset_version_id": source.dataset_version_id,
            "manifest_sha256": source.manifest_sha256,
            "data_origin": source.data_origin,
            "source_verification": source.source_verification,
            "profile_id": source.profile_id,
            "profile_revision": source.profile_revision,
        }
    if window_index is not None:
        report["windows"] = window_index.summary()
    if stats is not None:
        report["stats"] = {
            "unique_state_frames": stats.unique_state_frames,
            "unique_action_frames": stats.unique_action_frames,
            "fitted_frame_digest": stats.fitted_frame_digest,
        }
    if error is not None:
        report["error"] = {"code": error.code.value, "message": error.message, "details": error.details}
    if extra:
        report.update(dict(extra))
    return report


def prepare_bundle(
    *,
    recipe: TrainingRecipe,
    source: SourcePackage,
    output_dir: str | Path,
    report_dir: str | Path | None = None,
    requested_level: str = "PREPARED",
    check_video: bool = True,
    tolerance_s: float = 1e-4,
    implementation: Mapping[str, Any] | None = None,
) -> PreparedBundle:
    """准备并原子发布一个 TrainingBundle。"""
    if requested_level not in ("PREPARED", "CPU_VERIFIED"):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "prepare only supports PREPARED or CPU_VERIFIED",
            details={"requested_level": requested_level},
        )

    output = Path(output_dir)
    report_root = Path(report_dir) if report_dir else output.parent / f"{output.name}-reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "preparation.json"

    resolved = resolve_recipe(recipe, source.verified_source())
    missing_shapes = [
        camera for camera in resolved.features.cameras if camera not in resolved.features.image_shapes
    ]
    if missing_shapes:
        raise ApiError(
            ErrorCode.FEATURE_INCOMPATIBLE,
            "the resolved recipe has no image shape for a required camera",
            details={"cameras": missing_shapes},
        )

    implementation = dict(implementation or implementation_info())
    chunk_size = int(resolved.temporal.chunk_size)
    stride = int(resolved.temporal.stride)

    try:
        groups = _load_groups(source, recipe.split.group_mapping)
        assignments = (
            _load_assignments(source, recipe.split.assignments)
            if getattr(recipe.split, "assignments", None)
            else None
        )
        split = build_split(groups, recipe.split, assignments=assignments)

        scan = _scan_dataset(
            source,
            state_key=resolved.features.state_key,
            action_key=resolved.features.action_key,
            tolerance_s=tolerance_s,
        )
        window_index = build_window_index(
            scan.frames,
            split,
            chunk_size=chunk_size,
            stride=stride,
            require_nonempty=resolved.split.require_nonempty,
        )

        video_probe = "disabled"
        if check_video:
            failures: dict[str, int] = {}
            rejected = _probe_video(scan, window_index.windows, failures=failures)
            if rejected:
                frames = tuple(
                    EpisodeFrames(
                        episode_id=frame.episode_id,
                        official_episode_index=frame.official_episode_index,
                        frame_count=frame.frame_count,
                        invalid_frames={**frame.invalid_frames, **{
                            index: reason
                            for (episode_index, index), reason in rejected.items()
                            if episode_index == frame.official_episode_index
                        }},
                        row_digest=frame.row_digest,
                    )
                    for frame in scan.frames
                )
                window_index = build_window_index(
                    frames,
                    split,
                    chunk_size=chunk_size,
                    stride=stride,
                    require_nonempty=resolved.split.require_nonempty,
                )
            video_probe = "anchor"

        train_windows = window_index.windows_for("train")
        stats = fit_train_stats(
            train_windows,
            scan.frame_source,
            state_dim=resolved.features.state_dim,
            action_dim=resolved.features.action_dim,
            epsilon=resolved.normalization.epsilon,
        )
        processors = build_act_processors(resolved, stats)
        processor_document = processor_config_document(resolved, stats)

        groups_document = {
            "schema_version": SOURCE_GROUPS_SCHEMA,
            "groups": [group.as_dict() for group in groups],
        }
        assignments_document = (
            {"schema_version": ASSIGNMENTS_SCHEMA, "assignments": dict(sorted(assignments.items()))}
            if assignments is not None
            else None
        )

        preparation_id = semantic_sha256(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "source": source.digest_document(),
                "resolved_recipe_sha256": resolved_recipe_sha256(resolved),
                "source_groups_sha256": semantic_sha256(groups_document),
                "group_assignments_sha256": (
                    semantic_sha256(assignments_document) if assignments_document else None
                ),
                "split_sha256": semantic_sha256(split.as_document()),
                "windows_sha256": _window_semantic_digest(window_index),
                "rejected_windows_sha256": _rejected_semantic_digest(window_index),
                "stats_sha256": semantic_sha256(stats.as_document()),
                "processor_sha256": processors.config_sha256,
                "implementation": {
                    "source": implementation.get("source"),
                    "revision": implementation.get("revision"),
                    "dependencies": implementation.get("dependencies"),
                },
            }
        )

        tmp = output.parent / f".{output.name}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
        if tmp.exists():  # pragma: no cover - 同一进程同一毫秒的极端情况
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        _write_json(tmp / "resolved_recipe.json", resolved.model_dump(mode="json"))
        _write_json(tmp / "splits.json", split.as_document())
        _write_json(tmp / "source_groups.json", groups_document)
        if assignments_document is not None:
            _write_json(tmp / "group_assignments.json", assignments_document)
        _write_json(
            tmp / "train_stats.json",
            {
                **stats.as_document(),
                "processor": {
                    "sha256": processors.config_sha256,
                    "document": processor_document,
                },
            },
        )
        _write_windows_parquet(tmp / "windows.parquet", window_index)
        _write_rejected_parquet(tmp / "rejected_windows.parquet", window_index)

        processors_dir = tmp / "processors"
        processors_dir.mkdir()
        save_processor_step(processors.pre, processors_dir, "pre.json")
        save_processor_step(processors.post, processors_dir, "post.json")

        _write_json(
            tmp / "provenance.json",
            {
                "schema_version": PROVENANCE_SCHEMA,
                "preparation_id": preparation_id,
                "source": {
                    "dataset_version_id": source.dataset_version_id,
                    "dataset_slug": source.dataset_slug,
                    "manifest_sha256": source.manifest_sha256,
                    "data_origin": source.data_origin,
                    "source_verification": source.source_verification,
                    "license": source.license_id,
                    "quality_evidence": source.quality_evidence,
                },
                "source_root_layout": {
                    "files": [entry.as_dict() for entry in source.files],
                    "note": "relative paths only; the absolute source root is a runtime argument, not an identity",
                },
            },
        )

        _write_json(
            tmp / "reports" / "preparation.json",
            _prepare_report(
                status="PASS" if requested_level == "PREPARED" else "PENDING",
                level="PREPARED",
                preparation_id=preparation_id,
                error=None,
                source=source,
                window_index=window_index,
                stats=stats,
                extra={
                    "video_probe": video_probe,
                    "split": {
                        "strategy": split.strategy,
                        "counts": split.counts,
                        "actual_ratios": split.actual_ratios,
                    },
                    "rejected_by_reason": window_index.rejected_by_reason,
                    "tail_dropped": window_index.tail_dropped,
                    "requested_level": requested_level,
                },
                timestamp=False,
            ),
        )

        files = list(digest_paths(tmp, exclude=_IDENTITY_FILES))
        manifest_core = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "preparation_id": preparation_id,
            "source": source.digest_document(),
            "recipe": {
                "resolved_recipe_sha256": resolved_recipe_sha256(resolved),
                "target": resolved.model_dump(mode="json")["target"],
                "temporal": resolved.model_dump(mode="json")["temporal"],
                "normalization": resolved.model_dump(mode="json")["normalization"],
            },
            "digests": {
                "split": semantic_sha256(split.as_document()),
                "source_groups": semantic_sha256(groups_document),
                "group_assignments": (
                    semantic_sha256(assignments_document) if assignments_document else None
                ),
                "windows": _window_semantic_digest(window_index),
                "rejected_windows": _rejected_semantic_digest(window_index),
                "stats": semantic_sha256(stats.as_document()),
                "processor": processors.config_sha256,
            },
            "implementation": implementation,
            "verification": {
                "level": "PREPARED" if requested_level == "PREPARED" else "CPU_VERIFIED",
                "requested_level": requested_level,
                "video_probe": video_probe,
                "source_verification": source.source_verification,
            },
            "counts": {
                "episodes": len(source.episodes),
                "groups": len(groups),
                "windows": {name: window_index.counts.get(name, 0) for name in ("train", "val", "test")},
            },
        }
        artifact_id = semantic_sha256(
            {"manifest_core": manifest_core, "files": [entry.as_dict() for entry in files]}
        )
        manifest = {
            **manifest_core,
            "artifact_id": artifact_id,
            "files": [entry.as_dict() for entry in files],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(tmp / "manifest.json", manifest)
        _write_json(
            tmp / "artifact.json",
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "preparation_id": preparation_id,
                "artifact_id": artifact_id,
                "files_digest": semantic_sha256([entry.as_dict() for entry in files]),
                "note": "manifest.json and artifact.json are identity files and are excluded from the file manifest",
            },
        )

        # 请求 CPU_VERIFIED 时，验证报告必须在**发布前**写进 bundle：
        # 已发布的 bundle 不允许再追加或改写任何文件（§11.1）。
        if requested_level == "CPU_VERIFIED":
            from embodied_dataops.training.validate import (  # noqa: PLC0415 - 避免循环导入
                run_act_validation,
            )

            run_act_validation(
                bundle=tmp,
                source_root=source.source_root,
                device=resolved.validation.device,
                steps=int(resolved.validation.optimizer_steps),
                report_dir=tmp / "reports",
                report_name="cpu-validation.json",
                batch_size=int(resolved.validation.batch_size),
                seed=int(resolved.validation.seed),
                record_timestamp=False,
                redact_paths=True,
            )
            manifest_core["verification"]["level"] = "CPU_VERIFIED"
            _write_json(
                tmp / "reports" / "preparation.json",
                _prepare_report(
                    status="PASS",
                    level="CPU_VERIFIED",
                    preparation_id=preparation_id,
                    error=None,
                    source=source,
                    window_index=window_index,
                    stats=stats,
                    extra={
                        "video_probe": video_probe,
                        "split": {
                            "strategy": split.strategy,
                            "counts": split.counts,
                            "actual_ratios": split.actual_ratios,
                        },
                        "rejected_by_reason": window_index.rejected_by_reason,
                        "tail_dropped": window_index.tail_dropped,
                        "requested_level": requested_level,
                        "cpu_validation_report": "reports/cpu-validation.json",
                    },
                    timestamp=False,
                ),
            )
            files = list(digest_paths(tmp, exclude=_IDENTITY_FILES))
            artifact_id = semantic_sha256(
                {"manifest_core": manifest_core, "files": [entry.as_dict() for entry in files]}
            )
            manifest = {
                **manifest_core,
                "artifact_id": artifact_id,
                "files": [entry.as_dict() for entry in files],
                "created_at": manifest["created_at"],
            }
            _write_json(tmp / "manifest.json", manifest)
            _write_json(
                tmp / "artifact.json",
                {
                    "schema_version": BUNDLE_SCHEMA_VERSION,
                    "preparation_id": preparation_id,
                    "artifact_id": artifact_id,
                    "files_digest": semantic_sha256([entry.as_dict() for entry in files]),
                    "note": (
                        "manifest.json and artifact.json are identity files and are excluded "
                        "from the file manifest"
                    ),
                },
            )

        if output.exists():
            existing = load_bundle(output, verify_files=True)
            if existing.artifact_id != artifact_id:
                shutil.rmtree(tmp, ignore_errors=True)
                raise ApiError(
                    ErrorCode.CONFLICT,
                    "the target bundle already exists with different content",
                    details={
                        "target": str(output),
                        "existing_artifact_id": existing.artifact_id,
                        "requested_artifact_id": artifact_id,
                    },
                )
            shutil.rmtree(tmp, ignore_errors=True)
            prepared = PreparedBundle(
                path=output,
                preparation_id=preparation_id,
                artifact_id=artifact_id,
                manifest=existing.manifest,
                reused=True,
            )
        else:
            os.replace(tmp, output)
            prepared = PreparedBundle(
                path=output,
                preparation_id=preparation_id,
                artifact_id=artifact_id,
                manifest=manifest,
            )

        # 外部报告目录也要有一份可读的 preparation 记录；它只引用 preparation_id，
        # 不嵌入 artifact_id，避免文件摘要与身份互相依赖。
        _write_json(
            report_path,
            _prepare_report(
                status="PASS",
                level=str(manifest_core["verification"]["level"]),
                preparation_id=preparation_id,
                error=None,
                source=source,
                window_index=window_index,
                stats=stats,
                extra={
                    "video_probe": video_probe,
                    "split": {
                        "strategy": split.strategy,
                        "counts": split.counts,
                        "actual_ratios": split.actual_ratios,
                    },
                    "rejected_by_reason": window_index.rejected_by_reason,
                    "tail_dropped": window_index.tail_dropped,
                    "requested_level": requested_level,
                    "bundle": str(prepared.path),
                    "reused": prepared.reused,
                },
            ),
        )
        return prepared
    except ApiError as exc:
        _write_json(
            report_path,
            _prepare_report(
                status="FAIL",
                level=requested_level,
                preparation_id=None,
                error=exc,
                source=source,
            ),
        )
        raise


# ---------------------------------------------------------------------------
# 重载
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedBundle:
    """已验证的 bundle；任何文件摘要不符都在这里失败（TP-16）。"""

    path: Path
    manifest: dict[str, Any]
    preparation_id: str
    artifact_id: str
    resolved: ResolvedRecipe
    splits: dict[str, Any]
    stats: TrainStats
    windows: tuple[Window, ...]
    files: tuple[FileDigest, ...]

    def windows_for(self, split: str) -> tuple[Window, ...]:
        return tuple(window for window in self.windows if window.split == split)

    def source_layout(self) -> tuple[FileDigest, ...]:
        provenance = _read_json(self.path / "provenance.json")
        entries = (provenance.get("source_root_layout") or {}).get("files") or []
        return tuple(
            FileDigest(path=str(entry["path"]), sha256=str(entry["sha256"]), size=int(entry["bytes"]))
            for entry in entries
        )


def _windows_from_parquet(path: Path) -> tuple[Window, ...]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(path)
    if tuple(table.column_names) != WINDOW_COLUMNS:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "windows.parquet does not have the frozen column order",
            details={"observed": list(table.column_names), "expected": list(WINDOW_COLUMNS)},
        )
    rows = table.to_pylist()
    windows: list[Window] = []
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (str(row["split"]), str(row["source_episode_id"]), int(row["anchor_frame_index"]))
        if key in seen:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "windows.parquet contains a duplicate window key",
                details={"split": key[0], "source_episode_id": key[1], "anchor": key[2]},
            )
        seen.add(key)
        windows.append(
            Window(
                split=str(row["split"]),
                source_episode_id=str(row["source_episode_id"]),
                official_episode_index=int(row["official_episode_index"]),
                anchor_frame_index=int(row["anchor_frame_index"]),
                action_frame_indices=tuple(int(value) for value in row["action_frame_indices"]),
                source_row_digest=str(row["source_row_digest"]),
            )
        )
    ordered = sorted(
        windows, key=lambda item: (item.split, item.source_episode_id, item.anchor_frame_index)
    )
    if ordered != windows:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "windows.parquet is not sorted by (split, source_episode_id, anchor_frame_index)",
        )
    return tuple(windows)


def verify_source_root(source_root: str | Path, files: Sequence[FileDigest]) -> None:
    """逐文件校验 ``--source-root``；缺文件、摘要不符、目录穿越都拒绝。"""
    root = Path(source_root)
    if not root.is_dir():
        raise ApiError(
            ErrorCode.SOURCE_UNVERIFIED,
            "the source root does not exist",
            details={"source_root": str(root)},
        )
    for entry in files:
        path = _safe_relative(root, entry.path)
        if not path.is_file():
            raise ApiError(
                ErrorCode.SOURCE_HASH_MISMATCH,
                "a file recorded in the bundle is missing from the source root",
                details={"path": entry.path},
            )
        observed = _sha256_file(path)
        if observed != entry.sha256:
            raise ApiError(
                ErrorCode.SOURCE_HASH_MISMATCH,
                "the source root content no longer matches the bundle",
                details={"path": entry.path, "expected": entry.sha256, "observed": observed},
            )


def load_bundle(path: str | Path, *, verify_files: bool = True) -> LoadedBundle:
    """读取 bundle 并重算两种身份；不修改任何已有文件。"""
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "the path is not a published training bundle (manifest.json is missing)",
            details={"bundle": str(root)},
        )
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "the bundle has an unsupported schema_version",
            details={"schema_version": manifest.get("schema_version")},
        )

    declared = {entry["path"]: entry for entry in manifest.get("files") or []}
    files = digest_paths(root, exclude=_IDENTITY_FILES)
    if verify_files:
        observed = {entry.path: entry for entry in files}
        if set(observed) != set(declared):
            raise ApiError(
                ErrorCode.SOURCE_HASH_MISMATCH,
                "the bundle file listing does not match the published manifest",
                details={
                    "missing": sorted(set(declared) - set(observed)),
                    "unexpected": sorted(set(observed) - set(declared)),
                },
            )
        for name, entry in observed.items():
            if entry.sha256 != declared[name]["sha256"]:
                raise ApiError(
                    ErrorCode.SOURCE_HASH_MISMATCH,
                    "a bundle file was modified after publication",
                    details={"path": name, "expected": declared[name]["sha256"], "observed": entry.sha256},
                )

    core = {key: value for key, value in manifest.items() if key not in ("artifact_id", "files", "created_at")}
    recomputed = semantic_sha256(
        {"manifest_core": core, "files": [entry.as_dict() for entry in files]}
    )
    if recomputed != manifest.get("artifact_id"):
        raise ApiError(
            ErrorCode.SOURCE_HASH_MISMATCH,
            "the bundle artifact_id cannot be reproduced from its content",
            details={"declared": manifest.get("artifact_id"), "recomputed": recomputed},
        )

    resolved_payload = _read_json(root / "resolved_recipe.json")
    try:
        resolved = ResolvedRecipe.model_validate(resolved_payload)
    except Exception as exc:  # noqa: BLE001 - pydantic 的具体异常类型不稳定
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "resolved_recipe.json does not satisfy the resolved recipe contract",
            details={"error": type(exc).__name__},
        ) from exc

    stats_document = _read_json(root / "train_stats.json")
    stats = stats_from_document(stats_document)
    splits = _read_json(root / "splits.json")
    windows = _windows_from_parquet(root / "windows.parquet")

    if semantic_sha256(stats_document.get("processor", {}).get("document")) != (
        stats_document.get("processor", {}).get("sha256")
    ):
        raise ApiError(
            ErrorCode.PROCESSOR_STATS_MISMATCH,
            "the stored processor document does not match its digest",
        )

    return LoadedBundle(
        path=root,
        manifest=manifest,
        preparation_id=str(manifest.get("preparation_id")),
        artifact_id=str(manifest.get("artifact_id")),
        resolved=resolved,
        splits=splits,
        stats=stats,
        windows=windows,
        files=files,
    )
