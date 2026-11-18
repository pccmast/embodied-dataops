"""Independent-reader verification of an exported LeRobot v3 package (AT-17).

The verification runs in a **fresh Python process** (``sys.executable -c ...``)
that imports the official ``LeRobotDataset`` reader and re-opens the exported
root.  The parent process then compares the reader's observations against the
canonical Parquet values:

* episode count and total frame count match the canonical membership;
* ``observation.state`` / ``action`` are 6D;
* task labels match per episode;
* 10 deterministic random rows agree with the canonical values within 1e-5.

Nothing here executes application/fixture code in the child; only the trusted
inspection script and the official reader are imported.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_dataops.errors import ApiError, ErrorCode

VERIFY_SEED = 20261029
MAX_SAMPLES = 10
MAX_ABS_ERROR = 1e-5
#: Derived ``frame_index / fps`` timestamps are float32 in the official dataset.
MAX_TIMESTAMP_ERROR = 1e-4
DEFAULT_TIMEOUT_SECONDS = 300

INSPECT_SCRIPT = r"""
import json
import sys

root, repo_id, state_key, action_key, request_json = sys.argv[1:6]
requests = json.loads(request_json)
result = {}
try:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except Exception as exc:
    print(json.dumps({"import_error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(3)


def to_plain(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def scalar(value):
    value = to_plain(value)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


dataset = LeRobotDataset(repo_id=repo_id, root=root)
result["num_episodes"] = int(getattr(dataset, "num_episodes", 0) or 0)
result["num_frames"] = int(getattr(dataset, "num_frames", len(dataset)) or 0)
result["fps"] = int(getattr(dataset, "fps", 0) or 0)
features = dict(getattr(dataset, "features", {}) or {})
meta = getattr(dataset, "meta", None)
if not features and meta is not None:
    features = dict(getattr(meta, "features", {}) or {})
result["features"] = {
    name: {
        "dtype": str(spec.get("dtype")),
        "shape": list(spec.get("shape") or []),
        # Feature labels are part of the contract: an 8-state/7-action export must
        # come back with 8 state names and 7 action names, not one shared list.
        "names": [str(item) for item in (spec.get("names") or [])],
    }
    for name, spec in features.items()
    if isinstance(spec, dict)
}
# Official 0.4.4: ``meta.tasks`` is a pandas DataFrame whose index is the task
# string and which has a ``task_index`` column.  Older layouts expose a dict or
# a bare list; all three are normalised here.
tasks = getattr(meta, "tasks", None) if meta is not None else None
task_labels = []
task_index_to_label = {}
try:
    if tasks is None:
        pass
    elif hasattr(tasks, "columns") and hasattr(tasks, "index"):
        task_labels = [str(label) for label in list(tasks.index)]
        for label, row in tasks.iterrows():
            try:
                task_index_to_label[int(row["task_index"])] = str(label)
            except Exception:
                pass
    elif isinstance(tasks, dict):
        for label, value in tasks.items():
            task_labels.append(str(label))
            try:
                task_index_to_label[int(value)] = str(label)
            except Exception:
                pass
    else:
        task_labels = [str(task) for task in tasks]
except Exception:
    task_labels = []
result["tasks"] = task_labels

hf = getattr(dataset, "hf_dataset", None)
columns_available = list(getattr(hf, "column_names", []) or [])
result["hf_columns"] = columns_available

selected_columns = [
    name
    for name in [
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "task",
        state_key,
        action_key,
    ]
    if name in columns_available
]
view = None
if hf is not None and selected_columns:
    try:
        view = hf.select_columns(selected_columns)
    except Exception:
        view = None

episode_lengths = {}
episode_tasks = {}
rows = []
if view is not None:
    length = len(view)
    result["hf_rows"] = int(length)
    episode_column = to_plain(view["episode_index"]) if "episode_index" in selected_columns else []
    for value in episode_column:
        key = int(value)
        episode_lengths[key] = episode_lengths.get(key, 0) + 1
    if "task_index" in selected_columns:
        task_index_column = to_plain(view["task_index"])
        for index, value in enumerate(task_index_column):
            episode_key = int(episode_column[index]) if episode_column else 0
            if episode_key in episode_tasks:
                continue
            label = task_index_to_label.get(int(value))
            if label is not None:
                episode_tasks[episode_key] = label
    elif "task" in selected_columns:
        task_column = to_plain(view["task"])
        for index, value in enumerate(task_column):
            episode_key = int(episode_column[index]) if episode_column else 0
            episode_tasks.setdefault(episode_key, str(value))
    for request in requests:
        row_index = int(request["index"])
        raw = view[row_index]
        sample = {
            "index": row_index,
            "episode_index": scalar(raw.get("episode_index")),
            "frame_index": scalar(raw.get("frame_index")),
            "timestamp": float(to_plain(raw.get("timestamp")) or 0.0),
            "task": raw.get("task"),
            "state": [float(v) for v in to_plain(raw.get(state_key)) or []],
            "action": [float(v) for v in to_plain(raw.get(action_key)) or []],
        }
        if "task_index" in selected_columns:
            task_index = scalar(raw.get("task_index"))
            sample["task_index"] = task_index
            if task_index is not None:
                sample["task"] = task_index_to_label.get(int(task_index))
        rows.append(sample)
    result["sample_source"] = "hf_dataset"
else:
    result["hf_rows"] = None
    for request in requests:
        row_index = int(request["index"])
        raw = dataset[row_index]
        rows.append(
            {
                "index": row_index,
                "episode_index": scalar(raw.get("episode_index")),
                "frame_index": scalar(raw.get("frame_index")),
                "timestamp": float(to_plain(raw.get("timestamp")) or 0.0),
                "task": raw.get("task"),
                "state": [float(v) for v in to_plain(raw.get(state_key)) or []],
                "action": [float(v) for v in to_plain(raw.get(action_key)) or []],
            }
        )
    result["sample_source"] = "dataset_getitem"

# Episode metadata: official 0.4.4 exposes ``meta.episodes`` as an HF Dataset
# with ``length`` and ``tasks`` columns; older layouts expose a list of dicts.
episodes_meta = getattr(meta, "episodes", None) if meta is not None else None
if episodes_meta is not None:
    try:
        if hasattr(episodes_meta, "column_names"):
            for position in range(len(episodes_meta)):
                row = episodes_meta[position]
                episode_key = int(row.get("episode_index", position))
                length = row.get("length")
                if length is not None:
                    episode_lengths[episode_key] = int(length)
                task_text = row.get("tasks", row.get("task"))
                if task_text is not None:
                    if isinstance(task_text, (list, tuple)):
                        episode_tasks[episode_key] = str(task_text[0]) if task_text else ""
                    else:
                        episode_tasks[episode_key] = str(task_text)
        else:
            rows_meta = None
            if hasattr(episodes_meta, "to_dict"):
                try:
                    rows_meta = episodes_meta.to_dict("records")
                except Exception:
                    rows_meta = None
            if rows_meta is None and isinstance(episodes_meta, list):
                rows_meta = episodes_meta
            for position, row in enumerate(rows_meta or []):
                if not isinstance(row, dict):
                    continue
                episode_key = int(row.get("episode_index", position))
                length = row.get("length")
                if length is not None:
                    episode_lengths[episode_key] = int(length)
                task_text = row.get("tasks", row.get("task"))
                if task_text is not None:
                    if isinstance(task_text, (list, tuple)):
                        episode_tasks[episode_key] = str(task_text[0]) if task_text else ""
                    else:
                        episode_tasks[episode_key] = str(task_text)
    except Exception:
        pass

result["episode_lengths"] = {str(key): value for key, value in sorted(episode_lengths.items())}
result["episode_tasks"] = {str(key): value for key, value in sorted(episode_tasks.items())}
result["samples"] = rows
print(json.dumps(result))
"""


def _installed_lerobot_version() -> str | None:
    """Read the pinned official package version without importing torch."""
    try:
        import importlib.metadata as metadata

        return metadata.version("lerobot")
    except Exception:  # pragma: no cover - environment dependent
        return None


@dataclass
class ExpectedSample:
    frame_index: int
    t_ns: int
    state: list[float]
    action: list[float]


@dataclass
class ExpectedEpisode:
    episode_id: str
    task_label: str
    samples: Sequence[ExpectedSample] = field(default_factory=list)


@dataclass
class RoundTripResult:
    episode_count: int
    frame_count: int
    state_dim: int
    action_dim: int
    tasks: list[str]
    samples_checked: int
    max_abs_error: float
    reader_process: str
    package_version: str | None
    max_timestamp_error: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "tasks": self.tasks,
            "samples_checked": self.samples_checked,
            "max_abs_error": self.max_abs_error,
            "max_timestamp_error": self.max_timestamp_error,
            "reader_process": self.reader_process,
            "package_version": self.package_version,
        }


def _fail(message: str, **details: Any) -> ApiError:
    return ApiError(
        ErrorCode.EXPORT_ROUND_TRIP_FAILED,
        message,
        details=details,
    )


def _safe_extract(archive_path: Path, destination: Path) -> Path:
    """Extract a .tar.gz with traversal/link protection and return the root."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.replace("\\", "/")
            if member.name.startswith("/") or name.startswith("/") or ".." in Path(name).parts:
                raise ApiError(
                    ErrorCode.UNSAFE_ARCHIVE,
                    "Export package contains an unsafe path",
                    details={"member": member.name[:200]},
                )
            if member.issym() or member.islnk() or member.isdev():
                raise ApiError(
                    ErrorCode.UNSAFE_ARCHIVE,
                    "Export package contains a link or device member",
                    details={"member": member.name[:200]},
                )
        extract_kwargs: dict[str, Any] = {"path": destination}
        try:
            archive.extractall(filter="data", **extract_kwargs)  # type: ignore[call-overload]
        except TypeError:  # pragma: no cover - Python < 3.12
            archive.extractall(**extract_kwargs)
    candidates = sorted(
        {path.parent for path in destination.rglob("info.json") if (path.parent / "info.json").is_file()},
        key=lambda item: len(item.parts),
    )
    for parent in candidates:
        if parent.name == "meta":
            return parent.parent
    if candidates:
        return candidates[0]
    raise ApiError(
        ErrorCode.EXPORT_ROUND_TRIP_FAILED,
        "Export package has no meta/info.json LeRobot metadata",
    )


def resolve_export_root(package_path: Path, workdir: Path | None = None) -> Path:
    """Accept either a dataset root or a ``.tar.gz`` package."""
    package_path = Path(package_path)
    if package_path.is_dir():
        return package_path
    if not package_path.is_file():
        raise ApiError(ErrorCode.OBJECT_NOT_FOUND, "Exported package is missing")
    target = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="lerobot-verify-"))
    return _safe_extract(package_path, target)


def sample_requests(total_rows: int) -> list[dict[str, int]]:
    count = min(MAX_SAMPLES, total_rows)
    if count <= 0:
        return []
    generator = random.Random(VERIFY_SEED)
    indices = sorted(generator.sample(range(total_rows), count))
    return [{"index": index} for index in indices]


def _run_inspector(
    root: Path,
    *,
    repo_id: str,
    requests: list[dict[str, int]],
    timeout: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-c",
        INSPECT_SCRIPT,
        str(root),
        repo_id,
        "observation.state",
        "action",
        json.dumps(requests),
    ]
    import os

    environment = dict(os.environ)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _fail(
            "Official LeRobot reader verification timed out",
            timeout_seconds=timeout,
            reader_process=sys.executable,
        ) from exc
    stdout = (completed.stdout or "").strip()
    if completed.returncode != 0:
        # Exit code 3 is the inspector's explicit "official API import failed".
        if completed.returncode == 3:
            try:
                import_error = json.loads(stdout.splitlines()[-1]).get("import_error", "")
            except (IndexError, json.JSONDecodeError):
                import_error = stdout[-2000:]
            raise ApiError(
                ErrorCode.LEROBOT_API_UNAVAILABLE,
                "The official lerobot package is not importable in a fresh Python process.",
                details={"import_error": str(import_error)[:2000]},
            )
        raise _fail(
            "Official LeRobot reader could not open the exported dataset",
            returncode=completed.returncode,
            stderr=(completed.stderr or "")[-2000:],
            stdout=stdout[-2000:],
        )
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise _fail(
            "Official LeRobot reader returned unparsable output",
            stdout=stdout[-2000:],
            stderr=(completed.stderr or "")[-2000:],
        ) from exc
    if "import_error" in payload:
        raise ApiError(
            ErrorCode.LEROBOT_API_UNAVAILABLE,
            "The official lerobot package is not importable in a fresh Python process.",
            details={"import_error": payload["import_error"][:2000]},
        )
    if not payload.get("ok", True):  # pragma: no cover - defensive
        raise _fail("Official LeRobot reader verification failed", reader_output=payload)
    return payload


def verify_export(
    package_path: Path,
    *,
    repo_id: str,
    expected: Sequence[ExpectedEpisode],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    workdir: Path | None = None,
) -> RoundTripResult:
    """Re-open the exported package with the official Reader and compare values."""
    total_rows = sum(len(episode.samples) for episode in expected)
    if total_rows == 0:
        raise _fail("Cannot verify an export with no samples")
    root = resolve_export_root(Path(package_path), workdir)
    payload = _run_inspector(
        root,
        repo_id=repo_id,
        requests=sample_requests(total_rows),
        timeout=timeout,
    )

    episode_count = int(payload.get("num_episodes") or 0)
    frame_count = int(payload.get("num_frames") or 0)
    if episode_count != len(expected):
        raise _fail(
            "Round-trip episode count mismatch",
            expected=len(expected),
            observed=episode_count,
        )
    if frame_count != total_rows:
        raise _fail(
            "Round-trip total frame count mismatch",
            expected=total_rows,
            observed=frame_count,
        )

    features = payload.get("features") or {}
    state_shape = list((features.get("observation.state") or {}).get("shape") or [])
    action_shape = list((features.get("action") or {}).get("shape") or [])
    state_dim = int(state_shape[0]) if state_shape else 0
    action_dim = int(action_shape[0]) if action_shape else 0
    if state_dim != 6 or action_dim != 6:
        raise _fail(
            "Round-trip state/action dimensions are not 6D",
            state_shape=state_shape,
            action_shape=action_shape,
        )

    observed_lengths = {int(key): int(value) for key, value in (payload.get("episode_lengths") or {}).items()}
    if observed_lengths:
        expected_lengths = {index: len(episode.samples) for index, episode in enumerate(expected)}
        if observed_lengths != expected_lengths:
            raise _fail(
                "Round-trip per-episode frame counts mismatch",
                expected=expected_lengths,
                observed=observed_lengths,
            )

    expected_tasks = [episode.task_label for episode in expected]
    observed_tasks = [str(task) for task in (payload.get("tasks") or [])]
    observed_episode_tasks = {
        int(key): str(value) for key, value in (payload.get("episode_tasks") or {}).items()
    }
    if observed_episode_tasks:
        for index, task in enumerate(expected_tasks):
            if int(index) in observed_episode_tasks and observed_episode_tasks[index] != task:
                raise _fail(
                    "Round-trip task label mismatch",
                    episode_index=index,
                    expected=task,
                    observed=observed_episode_tasks[index],
                )
    elif observed_tasks:
        # ``meta/tasks.jsonl`` stores the unique task set; per-episode mapping is
        # verified above when the reader exposes it.
        if sorted(set(expected_tasks)) != sorted(set(observed_tasks)):
            raise _fail(
                "Round-trip task labels mismatch",
                expected=sorted(set(expected_tasks)),
                observed=sorted(set(observed_tasks)),
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
        if observed_frame_index is not None and int(observed_frame_index) != canonical.frame_index:
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
    if max_error > MAX_ABS_ERROR:
        raise _fail(
            "Round-trip state/action absolute error exceeds tolerance",
            max_abs_error=max_error,
            tolerance=MAX_ABS_ERROR,
        )
    if max_timestamp_error > MAX_TIMESTAMP_ERROR:
        raise _fail(
            "Round-trip timestamp absolute error exceeds tolerance",
            max_timestamp_error=max_timestamp_error,
            tolerance=MAX_TIMESTAMP_ERROR,
        )

    return RoundTripResult(
        episode_count=episode_count,
        frame_count=frame_count,
        state_dim=state_dim,
        action_dim=action_dim,
        tasks=sorted(observed_tasks),
        samples_checked=checked,
        max_abs_error=max_error,
        max_timestamp_error=max_timestamp_error,
        reader_process=f"{sys.executable} -c <official-reader>",
        package_version=_installed_lerobot_version(),
    )
