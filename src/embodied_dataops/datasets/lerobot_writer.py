"""Official LeRobot v3 Dataset Writer export (SPEC 10.1, M5).

This module is the *only* place that writes a LeRobot on-disk dataset.  It calls
the official ``lerobot`` API (pinned to 0.4.4) instead of hand-writing a lookalike
layout:

* ``LeRobotDataset.create(...)``
* ``LeRobotDataset.add_frame(...)`` once per 10 Hz sample
* ``LeRobotDataset.save_episode()``
* ``LeRobotDataset.finalize()``

Frames are decoded from the canonical H.264 MP4 (``frames/front.mp4``) with
OpenCV and written as 640x480 RGB; ``observation.state`` / ``action`` are the
6D ``float32`` canonical columns and ``task`` is the episode task label.

If the official package is not importable the writer raises
``ApiError(LEROBOT_API_UNAVAILABLE)``: it never fabricates a "LeRobot-like"
directory as a fallback, because that would defeat the round-trip acceptance
test (AT-17).
"""

from __future__ import annotations

import inspect
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops import TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.canonical.contract import CanonicalEpisode
from embodied_dataops.errors import ApiError, ErrorCode

PACKAGE_VERSION = "0.4.4"
VIDEO_FEATURE_KEY = "observation.images.front"
STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"
DEFAULT_ROBOT_TYPE = "arm6_joint_position_v1"


@dataclass
class ExportResult:
    root: Path
    repo_id: str
    episode_count: int
    total_frames: int
    features: dict[str, Any]
    package_version: str


def lerobot_importable() -> bool:
    try:
        _import_lerobot_dataset()
    except ApiError:
        return False
    return True


def lerobot_version() -> str | None:
    try:
        import importlib.metadata as metadata

        return metadata.version("lerobot")
    except Exception:  # pragma: no cover - environment dependent
        return None


def _import_lerobot_dataset() -> Any:
    """Import the official ``LeRobotDataset`` across supported module paths."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        return LeRobotDataset
    except Exception as first_error:  # pragma: no cover - version specific
        try:
            from lerobot.common.datasets.lerobot_dataset import (  # type: ignore[no-redef]  # noqa: PLC0415
                LeRobotDataset,
            )

            return LeRobotDataset
        except Exception as second_error:
            raise ApiError(
                ErrorCode.LEROBOT_API_UNAVAILABLE,
                "The official lerobot package is not importable; install the pinned "
                f"'lerobot=={PACKAGE_VERSION}' dependency to export LeRobot v3 datasets.",
                details={
                    "expected_version": PACKAGE_VERSION,
                    "import_error": (
                        f"{type(first_error).__name__}: {first_error}; "
                        f"{type(second_error).__name__}: {second_error}"
                    ),
                },
            ) from second_error


def _accepted_call_kwargs(target: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    """Filter ``candidate`` to kwargs the installed official signature accepts.

    This is a compatibility shim, not a second implementation: every accepted
    argument is forwarded to the official constructor method unchanged.
    """
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return dict(candidate)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(candidate)
    return {key: value for key, value in candidate.items() if key in parameters}


def build_features(episode: CanonicalEpisode) -> dict[str, Any]:
    state_dim = int(episode.metadata.state_dim)
    action_dim = int(episode.metadata.action_dim)
    joint_names = list(episode.metadata.joint_names) or [f"joint_{i}" for i in range(1, 7)]
    return {
        VIDEO_FEATURE_KEY: {
            "dtype": "video",
            "shape": (3, TARGET_HEIGHT, TARGET_WIDTH),
            "names": ["channels", "height", "width"],
        },
        STATE_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": joint_names,
        },
        ACTION_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": joint_names,
        },
    }


class _SequentialFrameReader:
    """Memory-bounded sequential decoder for the canonical MP4.

    Canonical samples are ordered and their ``frame_index`` values are
    non-decreasing (validated by the canonical contract), so we advance one
    ``cv2.VideoCapture`` instead of buffering every frame.
    """

    def __init__(self, path: Path) -> None:
        import cv2  # noqa: PLC0415

        self._cv2 = cv2
        self.path = path
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():  # pragma: no cover - corrupt fixture path
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "Cannot open the canonical video for export",
                details={"path": path.name},
            )
        self._next_index = 0
        self._last_frame: np.ndarray | None = None

    def frame_at(self, frame_index: int) -> np.ndarray:
        target = int(frame_index)
        if target < self._next_index:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "Canonical frame_index values must be non-decreasing for export",
                details={"requested": target, "already_consumed": self._next_index},
            )
        while self._next_index <= target:
            ok, frame = self._capture.read()
            if not ok:
                raise ApiError(
                    ErrorCode.FILE_DECODE_ERROR,
                    "Canonical video ended before the last canonical sample",
                    details={"requested_frame_index": target},
                )
            self._last_frame = frame
            self._next_index += 1
        assert self._last_frame is not None
        return self._last_frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> _SequentialFrameReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _to_rgb_640x480(frame: np.ndarray) -> np.ndarray:
    import cv2  # noqa: PLC0415

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if rgb.shape[1] != TARGET_WIDTH or rgb.shape[0] != TARGET_HEIGHT:
        # Canonical episodes are already letterboxed; resize only as a safety net
        # and never change the documented aspect handling here.
        rgb = cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _add_frame(dataset: Any, frame: dict[str, Any], task: str, timestamp: float) -> None:
    """Call the official ``add_frame`` for both 0.3-era and 0.4-era signatures.

    In lerobot 0.4.4 ``add_frame(frame: dict)`` expects only the declared feature
    keys plus ``task``; ``timestamp`` is *not* accepted there (``validate_frame``
    rejects unknown keys) and is derived by the official writer as
    ``frame_index / fps``.  Because our canonical samples sit exactly on the
    10 Hz grid that derived timestamp equals ``t_ns / 1e9``, so no information is
    lost.  Older signatures that take ``task``/``timestamp`` positional/keyword
    arguments are still supported.
    """
    try:
        parameters = inspect.signature(dataset.add_frame).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        parameters = {}
    if params_style_supported(parameters):
        kwargs: dict[str, Any] = {}
        payload = dict(frame)
        if "task" in parameters:
            kwargs["task"] = task
        else:
            payload["task"] = task
        if "timestamp" in parameters:
            kwargs["timestamp"] = float(timestamp)
        dataset.add_frame(payload, **kwargs)
        return
    # v0.4.4 frame-dict API: task travels inside the frame mapping; the official
    # writer computes timestamp = frame_index / fps.
    dataset.add_frame({**frame, "task": task})


def params_style_supported(parameters: dict[str, Any]) -> bool:
    """True when the installed ``add_frame`` takes task/timestamp as arguments."""
    return "task" in parameters or "timestamp" in parameters


def _save_episode(dataset: Any, task: str) -> None:
    try:
        parameters = inspect.signature(dataset.save_episode).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        parameters = {}
    if "task" in parameters:
        dataset.save_episode(task=task)
    else:
        dataset.save_episode()


def _create_dataset(LeRobotDataset: Any, *, repo_id: str, root: Path, features: dict[str, Any]) -> Any:
    candidate = {
        "repo_id": repo_id,
        "fps": TARGET_FPS,
        "root": root,
        "robot_type": DEFAULT_ROBOT_TYPE,
        "features": features,
        "use_videos": True,
        "tolerance_s": 1e-4,
        "image_writer_processes": 0,
        "image_writer_threads": 0,
        "batch_encoding_size": 1,
    }
    kwargs = _accepted_call_kwargs(LeRobotDataset.create, candidate)
    kwargs.setdefault("repo_id", repo_id)
    kwargs.setdefault("fps", TARGET_FPS)
    kwargs.setdefault("root", root)
    kwargs.setdefault("features", features)
    return LeRobotDataset.create(**kwargs)


def export_canonical_episodes(
    root: Path,
    episodes: Sequence[CanonicalEpisode],
    *,
    repo_id: str,
) -> ExportResult:
    """Write canonical episodes through the official Writer into ``root``.

    ``root`` must be a fresh directory (the caller owns cleanup).  Episodes are
    written in the given order, which the dataset service keeps sorted by
    ``episode_id`` so ``episode_index`` is deterministic.
    """
    if not episodes:
        raise ApiError(ErrorCode.EMPTY_VERSION, "Cannot export a dataset version without episodes")
    root = Path(root)
    # The official ``LeRobotDataset.create``/``LeRobotDatasetMetadata.create``
    # performs ``root.mkdir(parents=True, exist_ok=False)``; an existing directory
    # (even an empty one) is a hard error in lerobot 0.4.4. Only a completely
    # absent root is accepted; remove a leftover empty one owned by the caller.
    if root.exists():
        if any(root.iterdir()):
            raise ApiError(ErrorCode.INVALID_INPUT, "LeRobot export directory must be empty")
        root.rmdir()
    root.parent.mkdir(parents=True, exist_ok=True)

    LeRobotDataset = _import_lerobot_dataset()
    features = build_features(episodes[0])
    dataset = _create_dataset(LeRobotDataset, repo_id=repo_id, root=root, features=features)

    total_frames = 0
    for _position, episode in enumerate(episodes):
        metadata = episode.metadata
        if metadata.episode_type != "robot_episode":
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "Only robot_episode canonical episodes can be exported to LeRobot v3",
                details={"episode_id": metadata.episode_id, "episode_type": metadata.episode_type},
            )
        if int(metadata.state_dim) != 6 or int(metadata.action_dim) != 6:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "LeRobot export requires 6D state and action (arm6_joint_position_v1)",
                details={"state_dim": metadata.state_dim, "action_dim": metadata.action_dim},
            )
        task = (metadata.task_label or "").strip()
        if not task:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "LeRobot export requires a non-empty task label for every episode",
                details={"episode_id": metadata.episode_id},
            )
        if episode.video_path is None or not Path(episode.video_path).is_file():
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "Canonical episode video is missing",
                details={"episode_id": metadata.episode_id},
            )
        table = episode.samples
        rows = table.to_pylist()
        if not rows:
            raise ApiError(
                ErrorCode.INCOMPATIBLE_EPISODE,
                "Canonical episode has no samples",
                details={"episode_id": metadata.episode_id},
            )
        with _SequentialFrameReader(Path(episode.video_path)) as reader:
            for row in rows:
                image = _to_rgb_640x480(reader.frame_at(int(row["frame_index"])))
                state = np.asarray(row["state"], dtype=np.float32)
                action = np.asarray(row["action"], dtype=np.float32)
                if state.shape != (6,) or action.shape != (6,):
                    raise ApiError(
                        ErrorCode.INCOMPATIBLE_EPISODE,
                        "Canonical sample state/action is not 6D",
                        details={"episode_id": metadata.episode_id, "sample": row["sample_index"]},
                    )
                frame = {
                    VIDEO_FEATURE_KEY: image,
                    STATE_FEATURE_KEY: state,
                    ACTION_FEATURE_KEY: action,
                }
                _add_frame(dataset, frame, task, int(row["t_ns"]) / 1_000_000_000)
                total_frames += 1
        _save_episode(dataset, task)

    finalize = getattr(dataset, "finalize", None)
    if not callable(finalize):
        raise ApiError(
            ErrorCode.LEROBOT_API_UNAVAILABLE,
            f"lerobot {PACKAGE_VERSION} must expose LeRobotDataset.finalize()",
        )
    finalize()

    return ExportResult(
        root=root,
        repo_id=repo_id,
        episode_count=len(episodes),
        total_frames=total_frames,
        features=features,
        package_version=lerobot_version() or PACKAGE_VERSION,
    )


def tar_directory(directory: Path, archive_path: Path, *, arcname: str = "lerobot-v3") -> Path:
    """Deterministically pack an exported dataset root into ``.tar.gz``."""
    import tarfile

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(
        archive_path, mode="w:gz", format=tarfile.PAX_FORMAT, compresslevel=6
    ) as archive:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            archive.add(path, arcname=str(Path(arcname) / relative), recursive=False)
    return archive_path


def cleanup(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
