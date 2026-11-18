"""Deterministic synthetic fixture generator (SPEC 13.1, F01-F04/F07/F08/F10).

Run::

    python -m fixtures.generate_synthetic --out fixtures/generate

Rules:

* fixed seed :data:`fixtures.SEED` (20261029), fully offline, no network;
* every generated file is ``data_origin=synthetic`` and recorded with SHA-256,
  content type, expected outcome and generator identity in
  ``fixtures/manifest.json``;
* F01 is produced with the **official** ``lerobot==0.4.4`` Writer when the
  pinned extra is importable; otherwise the fixture is reported ``blocked`` with
  the concrete reason (the code path is still implemented and exercised the
  moment the extra exists);
* F02/F03 are honest corruptions of the official F01 layout (one timestamp
  inversion, a 5D action schema) so they exercise the quality rules on
  real LeRobot v3 files instead of a private format;
* F04/F10 are 10 s / 30 fps / 640x480 synthetic H.264 videos; F07 is a truncated
  copy of F04 that no decoder can fully read; F08 is a malicious ``.tar.gz``
  containing ``../escape.txt``;
* large media is written under ``fixtures/generate/`` which is git-ignored; only
  the manifest (metadata) is versioned.

The manifest is shared with the MCAP fixtures produced by
``fixtures/make_mcap_ros2.py`` (task-3); writes are read-modify-write guarded by
the ``fixtures/.manifest.lock`` file and entries owned by other generators are
preserved byte-for-byte.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import io
import json
import os
import shutil
import sys
import tarfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from fixtures import DATA_ORIGIN, SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "fixtures" / "generate"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "fixtures" / "manifest.json"
MANIFEST_SCHEMA_VERSION = "fixtures-manifest/1.0"
MANIFEST_LOCK_NAME = ".manifest.lock"

#: Fixture ids owned by this generator (other generators keep their own ids).
OWNED_FIXTURE_IDS: tuple[str, ...] = (
    "F01_good_lerobot",
    "F02_bad_timestamps",
    "F03_bad_dims",
    "F04_ego_video",
    "F07_corrupt_mp4",
    "F08_bad_tar",
    "F10_blurry_video",
    "F11_three_group",
)

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
EGO_FPS = 30
EGO_SECONDS = 10
EGO_FRAMES = EGO_FPS * EGO_SECONDS
ROBOT_FPS = 10
ROBOT_FRAMES = 100
ROBOT_EPISODES = 2
ROBOT_STATE_DIM = 6
ROBOT_ACTION_DIM = 6
#: §6.1 三组正常路径：3 个独立组，每组 ≥100 帧（chunk=16 时仍需足够窗口）。
THREE_GROUP_EPISODES = 3
THREE_GROUP_FRAMES = 120

SEMANTICS_SIDECAR = "meta/dataops_semantics.json"
SOURCE_SIDECAR = "meta/dataops_source.json"


# ---------------------------------------------------------------------------
# fixture result + manifest plumbing
# ---------------------------------------------------------------------------


@dataclass
class FixtureResult:
    fixture_id: str
    path: Path
    kind: str  # "file" | "directory"
    content_type: str
    sha256: str | None
    size_bytes: int | None
    status: str  # "generated" | "blocked" | "failed"
    expected: dict[str, Any]
    notes: str = ""
    blocked_reason: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def manifest_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "fixture_id": self.fixture_id,
            "path": _display_path(self.path),
            "kind": self.kind,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "data_origin": DATA_ORIGIN,
            "seed": SEED,
            "expected": self.expected,
            "generated_by": "fixtures/generate_synthetic.py",
            "status": self.status,
            "notes": self.notes,
        }
        if self.blocked_reason:
            entry["blocked_reason"] = self.blocked_reason
        if self.artifacts:
            entry["artifacts"] = self.artifacts
        if self.extra:
            entry["extra"] = self.extra
        return entry


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: str | Path) -> str:
    """Deterministic digest of a directory tree: sha256 of sorted (path, sha)."""
    root = Path(path)
    records: list[list[Any]] = []
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        records.append(
            [
                child.relative_to(root).as_posix(),
                _sha256_file(child),
                child.stat().st_size,
            ]
        )
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def directory_size(path: str | Path) -> int:
    root = Path(path)
    return sum(child.stat().st_size for child in root.rglob("*") if child.is_file())


class ManifestLock:
    """Cross-process exclusive lock file (retry + stale detection)."""

    def __init__(self, path: Path, *, timeout: float = 30.0, stale_after: float = 60.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.stale_after = stale_after
        self._fd: int | None = None

    def __enter__(self) -> ManifestLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except OSError:
                    age = 0.0
                if age > self.stale_after:
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"timed out waiting for manifest lock {self.path}"
                    ) from None
                time.sleep(0.05)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except OSError:
            pass


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    base: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": SEED,
        "data_origin": DATA_ORIGIN,
        "fixtures": [],
    }
    if not manifest_path.is_file():
        return base
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")
    except ValueError as exc:
        raise RuntimeError(f"fixtures manifest is not valid JSON: {manifest_path}") from exc
    if isinstance(payload, list):  # tolerate the older top-level list shape
        base["fixtures"] = [dict(item) for item in payload if isinstance(item, dict)]
        return base
    if not isinstance(payload, dict):
        raise RuntimeError(f"fixtures manifest has an unexpected shape: {manifest_path}")
    base.update({key: value for key, value in payload.items() if key != "fixtures"})
    fixtures = payload.get("fixtures")
    base["fixtures"] = [dict(item) for item in fixtures] if isinstance(fixtures, list) else []
    return base


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def update_manifest(path: str | Path, results: Sequence[FixtureResult]) -> dict[str, Any]:
    """Merge this generator's entries into the shared manifest atomically."""
    manifest_path = Path(path)
    with ManifestLock(manifest_path.parent / MANIFEST_LOCK_NAME):
        payload = read_manifest(manifest_path)
        owned = {result.fixture_id for result in results}
        merged: list[dict[str, Any]] = []
        for entry in payload.get("fixtures", []):
            fixture_id = str(entry.get("fixture_id") or "")
            if fixture_id in owned or (fixture_id in OWNED_FIXTURE_IDS and not owned):
                # Regenerate only entries we own; foreign entries (F05/F06/F09...)
                # are preserved untouched.
                if fixture_id in owned:
                    continue
            merged.append(entry)
        merged.extend(result.manifest_entry() for result in results)
        merged.sort(key=lambda item: str(item.get("fixture_id") or ""))
        payload["fixtures"] = merged
        payload["schema_version"] = MANIFEST_SCHEMA_VERSION
        payload["seed"] = SEED
        payload["data_origin"] = DATA_ORIGIN
        payload["generated_at"] = datetime.now(UTC).isoformat()
        payload["generator"] = "fixtures/generate_synthetic.py"
        write_manifest(manifest_path, payload)
        return payload


# ---------------------------------------------------------------------------
# deterministic synthetic frames
# ---------------------------------------------------------------------------


def _pattern_base(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, width, height])
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :, 0] = np.linspace(10, 120, width, dtype=np.uint8)[None, :]
    base[:, :, 1] = np.linspace(30, 90, height, dtype=np.uint8)[:, None]
    # deterministic sharp checkerboard patches make the Laplacian variance high
    for _ in range(6):
        x = int(rng.integers(0, max(1, width - 80)))
        y = int(rng.integers(0, max(1, height - 80)))
        size = int(rng.integers(30, 80))
        base[y : y + size, x : x + size, :] = 235
        base[y + size // 3 : y + 2 * size // 3, x + size // 4 : x + 3 * size // 4, :] = 20
    return base


def _ego_frame(base: np.ndarray, index: int, *, seed: int = SEED) -> np.ndarray:
    frame = base.copy()
    height, width = frame.shape[:2]
    x = int((index * 9) % max(1, width - 100))
    y = int((index * 5) % max(1, height - 100))
    frame[y : y + 90, x : x + 90, :] = (245, 30, 20)
    frame[y + 20 : y + 45, x + 20 : x + 45, :] = (10, 240, 200)
    return frame


def _blurry_frame(index: int, *, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT) -> np.ndarray:
    """Smooth low-contrast gradient (Laplacian variance far below 40)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    phase = index / max(1, EGO_FRAMES)
    frame[:, :, 0] = (60 + 20 * np.sin(np.linspace(0, 2 * np.pi + phase, width)))[None, :]
    frame[:, :, 1] = (60 + 20 * np.sin(np.linspace(phase, 2 * np.pi + phase, height)))[:, None]
    frame[:, :, 2] = 70
    try:
        import cv2

        return cv2.GaussianBlur(frame, (61, 61), 25)
    except Exception:  # pragma: no cover - cv2 is a hard dependency
        return frame


def laplacian_variance(frame: np.ndarray) -> float:
    """640x480 gray Laplacian variance used by ``BLUR_FRAME_RATE`` (SPEC 7.1)."""
    array = np.asarray(frame, dtype=np.float64)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.shape[0] < 3 or array.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * array[1:-1, 1:-1]
        + array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
    )
    return float(lap.var())


def _estimate_blur_ratio(frame_iter: Iterable[np.ndarray], sample: int = 10) -> dict[str, Any]:
    values: list[float] = []
    for index, frame in enumerate(frame_iter):
        if index >= sample:
            break
        values.append(laplacian_variance(frame))
    if not values:
        return {"sampled_frames": 0, "mean_laplacian_variance": 0.0, "blurry_ratio": 1.0}
    blurry = sum(1 for value in values if value < 40.0)
    return {
        "sampled_frames": len(values),
        "mean_laplacian_variance": sum(values) / len(values),
        "blurry_ratio": blurry / len(values),
    }


# ---------------------------------------------------------------------------
# video writing
# ---------------------------------------------------------------------------


def _write_video(
    frames: Iterable[np.ndarray],
    destination: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Encode RGB frames as H.264 (ffmpeg) with an OpenCV mp4v fallback."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    from embodied_dataops.processing import ffmpeg as fx

    if fx.ffmpeg_available():
        try:
            count = fx.encode_canvas_frames_h264(
                frames, destination, fps=fps, width=width, height=height
            )
            return {"encoder": "ffmpeg-libx264", "fps": fps, "frames": count}
        except Exception:  # fall through to OpenCV so fixtures stay offline-capable
            pass
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destination), fourcc, float(fps), (width, height))
    if not writer.isOpened():  # pragma: no cover - environment specific
        raise RuntimeError("no usable video encoder (ffmpeg libx264 and cv2.VideoWriter both failed)")
    count = 0
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        writer.release()
    return {"encoder": "opencv-mp4v", "fps": fps, "frames": count}


# ---------------------------------------------------------------------------
# F01/F02/F03: official LeRobot v3 datasets
# ---------------------------------------------------------------------------


def _feature_specs() -> dict[str, dict[str, Any]]:
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (3, VIDEO_HEIGHT, VIDEO_WIDTH),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (ROBOT_STATE_DIM,),
            "names": [f"joint_{index}" for index in range(1, ROBOT_STATE_DIM + 1)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ROBOT_ACTION_DIM,),
            "names": [f"joint_{index}" for index in range(1, ROBOT_ACTION_DIM + 1)],
        },
    }


def _robot_state(episode: int, index: int) -> np.ndarray:
    phase = index / ROBOT_FRAMES * 2 * np.pi + episode
    return np.array(
        [
            0.30 * np.sin(phase + dim * 0.35) + 0.02 * episode
            for dim in range(ROBOT_STATE_DIM)
        ],
        dtype=np.float32,
    )


def _robot_action(episode: int, index: int) -> np.ndarray:
    return (_robot_state(episode, index) + np.float32(0.01)).astype(np.float32)


def _robot_image(episode: int, index: int) -> np.ndarray:
    base = _pattern_base(VIDEO_WIDTH, VIDEO_HEIGHT, SEED + 100 + episode)
    frame = base.copy()
    x = int((index * 5 + episode * 40) % (VIDEO_WIDTH - 120))
    y = int((index * 3) % (VIDEO_HEIGHT - 120))
    frame[y : y + 110, x : x + 110, :] = (240, 40, 40 + episode * 60)
    frame[y + 30 : y + 60, x + 30 : x + 60, :] = (30, 230, 30)
    return frame


def _writer_image_value(rgb: np.ndarray, mode: str) -> Any:
    import torch

    chw = np.ascontiguousarray(np.transpose(np.asarray(rgb, dtype=np.uint8), (2, 0, 1)))
    if mode == "hwc_uint8":
        return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if mode == "float":
        return torch.from_numpy(chw.astype(np.float32) / 255.0)
    return torch.from_numpy(chw)


def _create_dataset(
    dataset_class: type, root: Path, *, repo_id: str = "dataops/F01_good_lerobot"
) -> Any:
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "fps": ROBOT_FPS,
        "features": _feature_specs(),
        "root": str(root),
        "robot_type": "arm6_joint_position_v1",
        "use_videos": True,
    }
    try:
        parameters = inspect.signature(dataset_class.create).parameters
    except (TypeError, ValueError):  # pragma: no cover
        parameters = {}
    if "video_backend" in parameters:
        try:
            import av  # noqa: F401

            kwargs["video_backend"] = "pyav"
        except Exception:
            pass
    if "image_writer_threads" in parameters:
        kwargs["image_writer_threads"] = 0
    if "image_writer_processes" in parameters:
        kwargs["image_writer_processes"] = 0
    if "vcodec" in parameters:
        vcodec = _available_pyav_encoder()
        if vcodec is not None:
            kwargs["vcodec"] = vcodec
    try:
        return dataset_class.create(**kwargs)
    except TypeError:
        minimal = {
            key: value
            for key, value in kwargs.items()
            if key in {"repo_id", "fps", "features", "root", "use_videos"}
        }
        return dataset_class.create(**minimal)


def _available_pyav_encoder() -> str | None:
    """Pick a codec name accepted by both PyAV and lerobot's vcodec whitelist."""
    try:
        import av
    except Exception:  # pragma: no cover - PyAV is part of the pinned extras
        return None
    for candidate in ("libsvtav1", "h264", "hevc"):
        try:
            av.codec.Codec(candidate, "w")
            return candidate
        except Exception:
            continue
    return None


def _add_frame(dataset: Any, frame: dict[str, Any], task: str) -> None:
    """``add_frame`` requires the task string inside every frame (v3 API)."""
    payload = dict(frame)
    payload["task"] = task
    dataset.add_frame(payload)


def _save_episode(dataset: Any, task: str) -> None:
    try:
        parameters = inspect.signature(dataset.save_episode).parameters
    except (TypeError, ValueError):  # pragma: no cover
        parameters = {}
    if "episode_data" in parameters and "parallel_encoding" in parameters:
        # Sequential encoding keeps fixture generation deterministic and avoids
        # Windows multiprocessing spawn issues in the CI sandbox.
        dataset.save_episode(None, parallel_encoding=False)
        return
    try:
        dataset.save_episode(task=task)
        return
    except TypeError:
        pass
    try:
        dataset.save_episode(episode_task=task)
        return
    except TypeError:
        pass
    dataset.save_episode()


def _finalize_dataset(dataset: Any) -> None:
    for attribute in ("finalize", "consolidate"):
        callable_attribute = getattr(dataset, attribute, None)
        if callable(callable_attribute):
            callable_attribute()
            return


def _write_provenance_sidecars(
    root: Path,
    *,
    repo_id: str = "dataops/F01_good_lerobot",
    evidence: str | None = None,
) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    semantics = {
        "robot_profile": "arm6_joint_position_v1",
        "state_units": "rad",
        "action_units": "rad",
        "action_semantics": "target_joint_position",
        "joint_order": [f"joint_{index}" for index in range(1, 7)],
        "data_origin": DATA_ORIGIN,
        "evidence": evidence
        or (
            "synthetic F01 fixture generated by fixtures/generate_synthetic.py with the "
            "official lerobot==0.4.4 Writer; state/action are 6D joint positions in radians "
            "and action is the target joint position"
        ),
        "generator": "fixtures/generate_synthetic.py",
        "seed": SEED,
    }
    source = {
        "repo_id": repo_id,
        "data_origin": DATA_ORIGIN,
        "generator": "fixtures/generate_synthetic.py",
        "seed": SEED,
        "notes": "synthetic fixture; not real robot capture",
    }
    (meta_dir / "dataops_semantics.json").write_text(
        json.dumps(semantics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (meta_dir / "dataops_source.json").write_text(
        json.dumps(source, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _try_create_official_f01(root: Path) -> dict[str, Any]:
    from embodied_dataops.imports.lerobot_v3 import lerobot_api_available

    if not lerobot_api_available():
        raise RuntimeError(
            "official lerobot==0.4.4 API is not importable (uv sync --extra lerobot); "
            "F01 must be produced with the official Writer"
        )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    errors: list[str] = []
    for mode in ("uint8", "float", "hwc_uint8"):
        shutil.rmtree(root, ignore_errors=True)
        try:
            dataset = _create_dataset(LeRobotDataset, root)
            for episode in range(ROBOT_EPISODES):
                task = f"synthetic pick-and-place episode {episode}"
                for index in range(ROBOT_FRAMES):
                    frame = {
                        "observation.images.front": _writer_image_value(
                            _robot_image(episode, index), mode
                        ),
                        "observation.state": __import__("torch").from_numpy(
                            _robot_state(episode, index)
                        ),
                        "action": __import__("torch").from_numpy(_robot_action(episode, index)),
                    }
                    _add_frame(dataset, frame, task)
                _save_episode(dataset, task)
            _finalize_dataset(dataset)
            _write_provenance_sidecars(root)
            return {
                "writer": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
                "writer_mode": mode,
                "episodes": ROBOT_EPISODES,
                "frames_per_episode": ROBOT_FRAMES,
                "fps": ROBOT_FPS,
                "state_dim": ROBOT_STATE_DIM,
                "action_dim": ROBOT_ACTION_DIM,
                "determinism_note": (
                    "the official Writer encodes videos with PyAV/SVT-AV1; logical content and "
                    "metadata are stable, raw video bytes may differ across encoder builds"
                ),
            }
        except Exception as exc:  # try the next image-value convention
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")
    shutil.rmtree(root, ignore_errors=True)
    raise RuntimeError("official LeRobot Writer failed for every frame convention: " + " | ".join(errors[:3]))


def _three_group_state(episode: int, index: int) -> np.ndarray:
    """三个组各自一条不同的轨迹，避免退化成同一份数据的复制品。"""
    phase = index / THREE_GROUP_FRAMES * 2 * np.pi + episode * 1.1
    return np.array(
        [0.35 * np.sin(phase + dim * 0.4) + 0.08 * episode for dim in range(ROBOT_STATE_DIM)],
        dtype=np.float32,
    )


def _write_group_sidecars(root: Path) -> None:
    """§6.1 的三组来源与显式分配：每组一个 episode，分别进 train/val/test。"""
    groups = [
        {
            "group_id": f"g_{index}",
            "episode_ids": [f"episode_{index}"],
            "evidence": {"session_keys": [], "note": f"synthetic three-group episode {index}"},
        }
        for index in range(THREE_GROUP_EPISODES)
    ]
    assignments = {f"g_{index}": name for index, name in enumerate(("train", "val", "test"))}
    (root / "source_groups.json").write_text(
        json.dumps(
            {"schema_version": "training-source-groups/1.0", "groups": groups},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "group_assignments.json").write_text(
        json.dumps(
            {"schema_version": "training-group-assignments/1.0", "assignments": assignments},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _try_create_official_three_group(root: Path) -> dict[str, Any]:
    from embodied_dataops.imports.lerobot_v3 import lerobot_api_available

    if not lerobot_api_available():
        raise RuntimeError(
            "official lerobot==0.4.4 API is not importable (uv sync --extra lerobot); "
            "F11 must be produced with the official Writer"
        )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "dataops/F11_three_group"
    errors: list[str] = []
    for mode in ("uint8", "float", "hwc_uint8"):
        shutil.rmtree(root, ignore_errors=True)
        try:
            dataset = _create_dataset(LeRobotDataset, root, repo_id=repo_id)
            for episode in range(THREE_GROUP_EPISODES):
                task = f"synthetic three-group episode {episode}"
                for index in range(THREE_GROUP_FRAMES):
                    frame = {
                        "observation.images.front": _writer_image_value(
                            _robot_image(episode, index), mode
                        ),
                        "observation.state": __import__("torch").from_numpy(
                            _three_group_state(episode, index)
                        ),
                        "action": __import__("torch").from_numpy(
                            _three_group_state(episode, index) + np.float32(0.01)
                        ),
                    }
                    _add_frame(dataset, frame, task)
                _save_episode(dataset, task)
            _finalize_dataset(dataset)
            _write_provenance_sidecars(
                root,
                repo_id=repo_id,
                evidence=(
                    "synthetic F11 three-group fixture generated with the official "
                    "lerobot==0.4.4 Writer: 3 independent episodes of "
                    f"{THREE_GROUP_FRAMES} frames at {ROBOT_FPS} Hz, one source group each, "
                    "assigned to train/val/test for the §6.1 normal path"
                ),
            )
            _write_group_sidecars(root)
            return {
                "writer": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
                "writer_mode": mode,
                "episodes": THREE_GROUP_EPISODES,
                "frames_per_episode": THREE_GROUP_FRAMES,
                "fps": ROBOT_FPS,
                "state_dim": ROBOT_STATE_DIM,
                "action_dim": ROBOT_ACTION_DIM,
                "groups": {"g_0": "train", "g_1": "val", "g_2": "test"},
                "determinism_note": (
                    "the official Writer encodes videos with PyAV/SVT-AV1; logical content and "
                    "metadata are stable, raw video bytes may differ across encoder builds"
                ),
            }
        except Exception as exc:  # try the next image-value convention
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")
    shutil.rmtree(root, ignore_errors=True)
    raise RuntimeError(
        "official LeRobot Writer failed for F11 in every frame convention: " + " | ".join(errors[:3])
    )


def _tar_directory(source: Path, destination: Path, *, root_name: str) -> dict[str, Any]:
    """Deterministic tar.gz (gzip mtime 0, normalized member metadata)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(child for child in source.rglob("*") if child.is_file())
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for path in files:
                    info = tarfile.TarInfo(f"{root_name}/{path.relative_to(source).as_posix()}")
                    info.size = path.stat().st_size
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return {
        "path": _display_path(destination),
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def _dataset_parquet_files(root: Path) -> list[Path]:
    data_dir = root / "data"
    if data_dir.is_dir():
        files = sorted(data_dir.rglob("*.parquet"))
        if files:
            return files
    return sorted(root.rglob("*.parquet"))


def _write_table(table: Any, path: Path) -> None:
    import pyarrow.parquet as pq

    pq.write_table(table, path)


def _episode_row_locations(
    root: Path, episode_index: int
) -> list[tuple[int, Path, int, Any]]:
    """Locate every physical row of one episode: (ordering_key, shard, row, timestamp)."""
    import pyarrow.parquet as pq

    located: list[tuple[int, Path, int, Any]] = []
    for path in _dataset_parquet_files(root):
        table = pq.read_table(path, columns=["episode_index", "timestamp"] + (["index"] if "index" in pq.read_schema(path).names else []))
        columns = table.column_names
        episodes = table.column("episode_index").to_pylist()
        timestamps = table.column("timestamp").to_pylist()
        indexes = table.column("index").to_pylist() if "index" in columns else None
        for row_position, episode in enumerate(episodes):
            if int(episode) != episode_index:
                continue
            ordering_key = int(indexes[row_position]) if indexes is not None else len(located)
            located.append((ordering_key, path, row_position, float(timestamps[row_position])))
    located.sort(key=lambda item: item[0])
    return located


def _corrupt_timestamps(root: Path, *, episode_index: int = 0, position: int = 50) -> str:
    """Make ``timestamp[position] <= timestamp[position-1]`` inside one episode."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    located = _episode_row_locations(root, episode_index)
    if len(located) <= position:
        raise RuntimeError(
            f"cannot corrupt timestamps: episode {episode_index} has only {len(located)} rows"
        )
    previous_timestamp = located[position - 1][3]
    target_key, target_path, target_row, _target_timestamp = located[position]
    table = pq.read_table(target_path)
    columns = table.column_names
    timestamps = table.column("timestamp").to_pylist()
    timestamps[target_row] = previous_timestamp
    field_index = columns.index("timestamp")
    field = table.schema.field(field_index)
    table = table.set_column(field_index, field, pa.array(timestamps, type=field.type))
    _write_table(table, target_path)
    return (
        f"set episode {episode_index} frame {position} timestamp equal to frame "
        f"{position - 1} (<= inversion, no repair applied)"
    )


def _trim_numeric_deep(value: Any, new_dim: int) -> Any:
    if isinstance(value, dict):
        return {key: _trim_numeric_deep(item, new_dim) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (int, float)):
            return list(value[:new_dim])
        return [_trim_numeric_deep(item, new_dim) for item in value]
    return value


def _is_action_stat_key(key: str) -> bool:
    return "action" in [token for token in str(key).replace("/", ".").split(".") if token]


def _trim_action_stats(value: Any, new_dim: int) -> Any:
    """Trim only action statistics (never the 6D state statistics)."""
    if isinstance(value, dict):
        return {
            key: (
                _trim_numeric_deep(item, new_dim)
                if _is_action_stat_key(str(key))
                else _trim_action_stats(item, new_dim)
            )
            for key, item in value.items()
        }
    return value


def _trimmed_action_field(field: Any, new_dim: int, pa: Any) -> Any:
    """Return the replacement Arrow field for an action column."""
    if pa.types.is_fixed_size_list(field.type):
        new_type = pa.list_(field.type.value_type, new_dim)
    elif pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
        new_type = field.type
    else:
        new_type = None
    if new_type is None:
        return None
    return pa.field(field.name, new_type, nullable=field.nullable, metadata=field.metadata)


def _corrupt_action_dim(root: Path, *, new_dim: int = 5) -> str:
    """Rewrite ``action`` to ``new_dim`` dims in info.json, parquet and stats."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    action_feature = features.get("action")
    if not isinstance(action_feature, dict):
        raise RuntimeError("F01 dataset has no action feature to corrupt")
    action_feature["shape"] = [new_dim]
    names = action_feature.get("names")
    if isinstance(names, list) and len(names) >= new_dim:
        action_feature["names"] = names[:new_dim]
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    for path in _dataset_parquet_files(root):
        table = pq.read_table(path)
        columns = table.column_names
        if "action" not in columns:
            continue
        values = table.column("action").to_pylist()
        trimmed = [list(row)[:new_dim] if row is not None else None for row in values]
        field_index = columns.index("action")
        field = table.schema.field(field_index)
        new_field = _trimmed_action_field(field, new_dim, pa)
        if new_field is None:
            continue
        table = table.set_column(field_index, new_field, pa.array(trimmed, type=new_field.type))
        _write_table(table, path)

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if isinstance(stats, dict):
            stats = _trim_action_stats(stats, new_dim)
            stats_path.write_text(
                json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )

    episodes_meta = root / "meta" / "episodes"
    for path in sorted(episodes_meta.rglob("*.parquet")) if episodes_meta.is_dir() else []:
        table = pq.read_table(path)
        columns = table.column_names
        changed = False
        for column_name in columns:
            if not _is_action_stat_key(column_name):
                continue
            field_index = columns.index(column_name)
            field = table.schema.field(field_index)
            values = table.column(column_name).to_pylist()
            trimmed = [
                list(value)[:new_dim] if isinstance(value, list) else value for value in values
            ]
            new_field = _trimmed_action_field(field, new_dim, pa)
            if new_field is None:
                continue
            table = table.set_column(field_index, new_field, pa.array(trimmed, type=new_field.type))
            changed = True
        if changed:
            _write_table(table, path)
    return f"rewrote action to {new_dim}D in meta/info.json, data parquet and stats"


# ---------------------------------------------------------------------------
# individual fixture builders
# ---------------------------------------------------------------------------


def _fixture_f01(out_dir: Path, *, force: bool) -> FixtureResult:
    fixture_id = "F01_good_lerobot"
    root = out_dir / fixture_id
    archive = out_dir / f"{fixture_id}.tar.gz"
    expected = {
        "outcome": "official LeRobot v3 import: exactly 2 episodes, 2x100 samples at 10 Hz, 6D state/action, QC PASS",
        "episode_count": ROBOT_EPISODES,
        "samples_per_episode": ROBOT_FRAMES,
        "state_dim": ROBOT_STATE_DIM,
        "action_dim": ROBOT_ACTION_DIM,
        "fps": ROBOT_FPS,
    }
    try:
        if force or not (root / "meta" / "info.json").is_file():
            extras = _try_create_official_f01(root)
        else:
            extras = {"writer": "reused existing official dataset", "writer_mode": "n/a"}
        artifacts = [_tar_directory(root, archive, root_name=fixture_id)]
        return FixtureResult(
            fixture_id=fixture_id,
            path=root,
            kind="directory",
            content_type="application/x-directory",
            sha256=directory_digest(root),
            size_bytes=directory_size(root),
            status="generated",
            expected=expected,
            notes="created with the official lerobot==0.4.4 Writer",
            artifacts=artifacts,
            extra=extras,
        )
    except Exception as exc:
        return FixtureResult(
            fixture_id=fixture_id,
            path=root,
            kind="directory",
            content_type="application/x-directory",
            sha256=None,
            size_bytes=None,
            status="blocked",
            expected=expected,
            notes="official Writer unavailable; importer code path implemented but not exercised",
            blocked_reason=str(exc),
        )


def _fixture_f11(out_dir: Path, *, force: bool) -> FixtureResult:
    fixture_id = "F11_three_group"
    root = out_dir / fixture_id
    archive = out_dir / f"{fixture_id}.tar.gz"
    expected = {
        "outcome": (
            "official LeRobot v3 three-group normal path: 3 independent episodes of "
            f"{THREE_GROUP_FRAMES} frames at {ROBOT_FPS} Hz, one source group each, "
            "assigned to train/val/test"
        ),
        "episode_count": THREE_GROUP_EPISODES,
        "samples_per_episode": THREE_GROUP_FRAMES,
        "state_dim": ROBOT_STATE_DIM,
        "action_dim": ROBOT_ACTION_DIM,
        "fps": ROBOT_FPS,
        "groups": {"g_0": "train", "g_1": "val", "g_2": "test"},
    }
    try:
        if force or not (root / "meta" / "info.json").is_file():
            extras = _try_create_official_three_group(root)
        else:
            extras = {"writer": "reused existing official dataset", "writer_mode": "n/a"}
        artifacts = [_tar_directory(root, archive, root_name=fixture_id)]
        return FixtureResult(
            fixture_id=fixture_id,
            path=root,
            kind="directory",
            content_type="application/x-directory",
            sha256=directory_digest(root),
            size_bytes=directory_size(root),
            status="generated",
            expected=expected,
            notes="created with the official lerobot==0.4.4 Writer; three independent groups",
            artifacts=artifacts,
            extra=extras,
        )
    except Exception as exc:
        return FixtureResult(
            fixture_id=fixture_id,
            path=root,
            kind="directory",
            content_type="application/x-directory",
            sha256=None,
            size_bytes=None,
            status="blocked",
            expected=expected,
            notes="official Writer unavailable; three-group path implemented but not exercised",
            blocked_reason=str(exc),
        )


def _copy_and_corrupt(
    out_dir: Path,
    source_root: Path | None,
    *,
    fixture_id: str,
    expected: dict[str, Any],
    mutator: Any,
    description: str,
    force: bool,
) -> FixtureResult:
    root = out_dir / fixture_id
    archive = out_dir / f"{fixture_id}.tar.gz"
    if source_root is None or not (source_root / "meta" / "info.json").is_file():
        return FixtureResult(
            fixture_id=fixture_id,
            path=root,
            kind="directory",
            content_type="application/x-directory",
            sha256=None,
            size_bytes=None,
            status="blocked",
            expected=expected,
            blocked_reason="requires F01_good_lerobot dataset directory (official Writer fixture)",
        )
    try:
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(source_root, root)
        strategy = mutator(root)
        artifacts = [_tar_directory(root, archive, root_name=fixture_id)]
        return FixtureResult(
            fixture_id=fixture_id,
            path=root,
            kind="directory",
            content_type="application/x-directory",
            sha256=directory_digest(root),
            size_bytes=directory_size(root),
            status="generated",
            expected=expected,
            notes=f"derived from F01: {description}",
            artifacts=artifacts,
            extra={"corruption_strategy": strategy},
        )
    except Exception as exc:
        return FixtureResult(
            fixture_id=fixture_id,
            path=root,
            kind="directory",
            content_type="application/x-directory",
            sha256=None,
            size_bytes=None,
            status="failed",
            expected=expected,
            blocked_reason=f"{type(exc).__name__}: {exc}",
        )


def _fixture_f04(out_dir: Path, *, force: bool) -> FixtureResult:
    fixture_id = "F04_ego_video"
    path = out_dir / f"{fixture_id}.mp4"
    base = _pattern_base(VIDEO_WIDTH, VIDEO_HEIGHT, SEED)
    frames = (_ego_frame(base, index) for index in range(EGO_FRAMES))
    blur = _estimate_blur_ratio(_ego_frame(base, index) for index in range(10))
    info = _write_video(frames, path, fps=EGO_FPS, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
    return FixtureResult(
        fixture_id=fixture_id,
        path=path,
        kind="file",
        content_type="video/mp4",
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        status="generated",
        expected={
            "outcome": "EGO_MP4 import: ego_video_episode, 100 samples at 10 Hz (0-9.9 s), no state/action",
            "episode_type": "ego_video_episode",
            "sample_count": 100,
            "duration_ns": 10_000_000_000,
            "has_state": False,
            "has_action": False,
        },
        notes="10 s / 30 fps / 640x480 synthetic sharp video",
        extra={
            "encoder": info["encoder"],
            "fps": info["fps"],
            "frames": info["frames"],
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "sharpness_estimate": blur,
        },
    )


def _fixture_f07(out_dir: Path, source_path: Path | None) -> FixtureResult:
    fixture_id = "F07_corrupt_mp4"
    path = out_dir / f"{fixture_id}.mp4"
    if source_path is None or not source_path.is_file():
        return FixtureResult(
            fixture_id=fixture_id,
            path=path,
            kind="file",
            content_type="video/mp4",
            sha256=None,
            size_bytes=None,
            status="blocked",
            expected={"outcome": "FILE_DECODE_ERROR / 422, no leaked internal path"},
            blocked_reason="requires F04_ego_video.mp4",
        )
    data = source_path.read_bytes()
    cut = max(1024, int(len(data) * 0.55))
    path.write_bytes(data[:cut])
    return FixtureResult(
        fixture_id=fixture_id,
        path=path,
        kind="file",
        content_type="video/mp4",
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        status="generated",
        expected={
            "outcome": "FILE_DECODE_ERROR ERROR finding, import rejected; internal paths not leaked",
            "error_code": "FILE_DECODE_ERROR",
        },
        notes=f"truncated copy of F04 at {cut}/{len(data)} bytes",
        extra={"source_size_bytes": len(data), "truncated_bytes": cut},
    )


def _fixture_f08(out_dir: Path) -> FixtureResult:
    fixture_id = "F08_bad_tar"
    path = out_dir / f"{fixture_id}.tar.gz"
    plausible = {
        "codebase_version": "v3.0",
        "fps": 10,
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [3, 480, 640]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        },
        "total_episodes": 1,
    }
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                payload = json.dumps(plausible).encode()
                info = tarfile.TarInfo("F08_bad_tar/meta/info.json")
                info.size = len(payload)
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
                escape = b"this must never be written outside the extraction directory\n"
                evil = tarfile.TarInfo("../escape.txt")
                evil.size = len(escape)
                evil.mtime = 0
                archive.addfile(evil, io.BytesIO(escape))
    return FixtureResult(
        fixture_id=fixture_id,
        path=path,
        kind="file",
        content_type="application/gzip",
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        status="generated",
        expected={
            "outcome": "422 UNSAFE_ARCHIVE, no file written outside the extraction directory",
            "error_code": "UNSAFE_ARCHIVE",
            "escape_member": "../escape.txt",
        },
        notes="malicious .tar.gz with a path-traversal member",
    )


def _fixture_f10(out_dir: Path) -> FixtureResult:
    fixture_id = "F10_blurry_video"
    path = out_dir / f"{fixture_id}.mp4"
    frames = (_blurry_frame(index) for index in range(EGO_FRAMES))
    blur = _estimate_blur_ratio(_blurry_frame(index) for index in range(10))
    info = _write_video(frames, path, fps=EGO_FPS, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
    return FixtureResult(
        fixture_id=fixture_id,
        path=path,
        kind="file",
        content_type="video/mp4",
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        status="generated",
        expected={
            "outcome": "BLUR_FRAME_RATE WARN (>30% of frames with 640x480 gray Laplacian variance < 40)",
            "quality_rule": "BLUR_FRAME_RATE",
            "severity": "WARN",
        },
        notes="10 s / 30 fps / 640x480 synthetically blurred video",
        extra={"encoder": info["encoder"], "fps": info["fps"], "frames": info["frames"], "blur_estimate": blur},
    )


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def generate_all(
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    manifest_path: str | Path | None = DEFAULT_MANIFEST_PATH,
    only: Sequence[str] | None = None,
    force: bool = False,
    seed: int = SEED,
) -> list[FixtureResult]:
    """Generate the requested fixtures and merge them into the shared manifest."""
    if seed != SEED:
        # SPEC 13.1 fixes one seed; refuse silent drift that would invalidate
        # committed SHAs in the manifest.
        raise ValueError(f"synthetic fixtures must use the fixed seed {SEED} (got {seed})")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    requested = set(only) if only else set(OWNED_FIXTURE_IDS)
    unknown = requested - set(OWNED_FIXTURE_IDS)
    if unknown:
        raise ValueError(f"unknown fixture ids requested: {sorted(unknown)}")

    results: list[FixtureResult] = []

    f01_root = out / "F01_good_lerobot"
    need_f01 = bool(
        {"F01_good_lerobot", "F02_bad_timestamps", "F03_bad_dims"} & requested
    )
    if need_f01:
        result = _fixture_f01(out, force=force)
        if "F01_good_lerobot" in requested:
            results.append(result)
        if result.status != "generated":
            f01_root = None
    elif (f01_root / "meta" / "info.json").is_file():
        pass
    else:
        f01_root = None

    if "F11_three_group" in requested:
        results.append(_fixture_f11(out, force=force))

    if "F02_bad_timestamps" in requested:
        results.append(
            _copy_and_corrupt(
                out,
                f01_root,
                fixture_id="F02_bad_timestamps",
                expected={
                    "outcome": "TIMESTAMP_NON_MONOTONIC ERROR, no silent repair, not publishable",
                    "quality_rule": "TIMESTAMP_NON_MONOTONIC",
                },
                mutator=_corrupt_timestamps,
                description="t_ns[50] <= t_ns[49] in episode 0",
                force=force,
            )
        )
    if "F03_bad_dims" in requested:
        results.append(
            _copy_and_corrupt(
                out,
                f01_root,
                fixture_id="F03_bad_dims",
                expected={
                    "outcome": "DIMENSION_MISMATCH ERROR for 5D action, no truncation/zero-fill",
                    "quality_rule": "DIMENSION_MISMATCH",
                },
                mutator=_corrupt_action_dim,
                description="action schema rewritten to 5D",
                force=force,
            )
        )

    need_f04 = bool({"F04_ego_video", "F07_corrupt_mp4"} & requested)
    f04_path = out / "F04_ego_video.mp4"
    if "F04_ego_video" in requested:
        f04_result = _fixture_f04(out, force=force)
        results.append(f04_result)
        f04_source = f04_result.path if f04_result.status == "generated" else None
    elif need_f04:
        if force or not f04_path.is_file():
            dependency = _fixture_f04(out, force=force)
            f04_source = dependency.path if dependency.status == "generated" else None
        else:
            f04_source = f04_path
    else:
        f04_source = None
    if "F07_corrupt_mp4" in requested:
        results.append(_fixture_f07(out, f04_source if f04_source and f04_source.is_file() else None))
    if "F08_bad_tar" in requested:
        results.append(_fixture_f08(out))
    if "F10_blurry_video" in requested:
        results.append(_fixture_f10(out))

    if manifest_path is not None and results:
        update_manifest(manifest_path, results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fixtures.generate_synthetic",
        description="Generate the deterministic synthetic fixtures (SPEC 13.1).",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output directory (git-ignored)")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="shared fixtures manifest (merged under a lock)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=list(OWNED_FIXTURE_IDS),
        help="subset of fixture ids to generate",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"fixed seed (must be {SEED})")
    parser.add_argument("--force", action="store_true", help="regenerate even when files exist")
    parser.add_argument("--no-manifest", action="store_true", help="do not write the shared manifest")
    args = parser.parse_args(argv)

    try:
        results = generate_all(
            args.out,
            manifest_path=None if args.no_manifest else args.manifest,
            only=args.only,
            force=args.force,
            seed=args.seed,
        )
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(f"fixture generation failed: {exc}", file=sys.stderr)
        return 2
    for result in results:
        digest = (result.sha256 or "-")[:16]
        print(
            f"{result.fixture_id:22s} {result.status:9s} {result.content_type:24s} "
            f"{digest:16s} {_display_path(result.path)}"
        )
        if result.blocked_reason:
            print(f"  blocked: {result.blocked_reason}", file=sys.stderr)
    blocked = [result.fixture_id for result in results if result.status == "blocked"]
    if blocked:
        print(f"BLOCKED fixtures: {', '.join(blocked)}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
