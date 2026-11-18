"""Generate the synthetic ``F05_mcap_good`` / ``F06_mcap_gap`` ROS 2 MCAP fixtures.

These fixtures are **synthetic** (``data_origin=synthetic``, fixed seed
``20261029``); they are never real robot captures and must be described as such
everywhere they are used (see CONTRIBUTING.md, non-negotiable 1).

The files are real ROS 2 CDR MCAP containers written with the ``mcap`` pip
writer and the hand-written CDR encoders in
``services/ros2_adapter/cdr.py``:

* ``F05_mcap_good`` -- 160x120 rgb8 ``sensor_msgs/Image`` at 30 Hz,
  ``sensor_msgs/JointState`` for ``joint_1``..``joint_6`` at 100 Hz and
  ``std_msgs/Float64MultiArray`` (6 float64 target joint positions) at 50 Hz,
  10 s of camera coverage; the joint-state and action streams deliberately start
  *before* the first camera frame so the profile's required history exists
  (SPEC 5.4: state interpolation / past-only action hold at ``t=0``).
* ``F06_mcap_gap`` -- identical streams with the joint-state messages between
  2 s and 3 s removed, so the 10 Hz grid points in that window have no valid
  state bracket.  Expected outcome: ``SYNC_MISSING_STATE`` ERROR, QC ``FAIL``,
  no fabricated samples.

The generator is deterministic: re-running it produces byte-identical files
(asserted by ``tests/unit/test_mcap_fixture.py``).

Usage::

    python fixtures/make_mcap_ros2.py [--output-dir fixtures/golden/mcap] [--no-manifest]

It appends its own entries to ``fixtures/manifest.json`` using the same
exclusive-create lock (``fixtures/.manifest.lock``) as
``fixtures/generate_synthetic.py``, tolerating concurrent edits and preserving
every fixture entry it does not own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:  # make ``services.*`` importable when run as a script
    sys.path.insert(0, str(REPOSITORY_ROOT))

from services.ros2_adapter.cdr import (  # noqa: E402
    FLOAT64_MULTI_ARRAY_TYPE,
    IMAGE_TYPE,
    JOINT_STATE_TYPE,
    ROS2_MSG_DEFINITIONS,
    encode_float64_multi_array,
    encode_image,
    encode_joint_state,
)
from services.ros2_adapter.reader import (  # noqa: E402
    TOPIC_ACTION,
    TOPIC_IMAGE,
    TOPIC_JOINT_STATES,
)

#: Fixed random seed for every synthetic fixture (SPEC 13.1).
SEED = 20261029
#: Fixed absolute ``log_time`` base so fixture bytes are reproducible (2023-11-14T22:13:20Z).
ORIGIN_NS = 1_700_000_000_000_000_000
#: Images: 160x120 rgb8 at 30 Hz for 10 s (300 frames).
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
IMAGE_FPS = 30
IMAGE_COUNT = 300
#: JointState at 100 Hz, starting 200 ms before the first camera frame.
JOINT_PERIOD_NS = 10_000_000
JOINT_HISTORY_NS = 200_000_000
JOINT_COUNT = 1020  # -0.2 s .. 9.99 s
#: Float64MultiArray at 50 Hz, starting 300 ms before the first camera frame.
ACTION_PERIOD_NS = 20_000_000
ACTION_HISTORY_NS = 300_000_000
ACTION_COUNT = 515  # -0.3 s .. 9.98 s
#: Header stamps are intentionally offset from log_time to exercise diagnostics.
IMAGE_HEADER_OFFSET_NS = -5_000_000
JOINT_HEADER_OFFSET_NS = -2_000_000
FRAME_ID = "camera_front"
LIBRARY = "embodied-dataops-fixtures/1.0"

MANIFEST_LOCK_NAME = ".manifest.lock"
MANIFEST_LOCK_TIMEOUT_S = 30.0
MANIFEST_LOCK_STALE_S = 60.0

FIXTURE_SPECS: dict[str, dict[str, Any]] = {
    "F05_mcap_good": {
        "drop_state_range_s": None,
        "notes": (
            "Synthetic ROS 2 CDR MCAP, fixed topic profile; state/action history starts "
            "200/300 ms before the first camera frame"
        ),
        "expected": {
            "outcome": "quality_pass",
            "verdict": "PASS",
            "episode_type": "robot_episode",
            "source_type": "MCAP_ROS2_PROFILE_V1",
            "normalization_valid": True,
            "grid_hz": 10,
            "duration_s": 10.0,
            "topics": [TOPIC_IMAGE, TOPIC_JOINT_STATES, TOPIC_ACTION],
        },
    },
    "F06_mcap_gap": {
        "drop_state_range_s": (2.0, 3.0),
        "notes": (
            "Same profile as F05 but joint-state messages in [2 s, 3 s] are removed; no "
            "zero-fill and no future action is allowed, so QC must FAIL"
        ),
        "expected": {
            "outcome": "quality_fail",
            "verdict": "FAIL",
            "error_rules": ["SYNC_MISSING_STATE"],
            "normalization_valid": False,
            "fabricated_samples_allowed": False,
            "topics": [TOPIC_IMAGE, TOPIC_JOINT_STATES, TOPIC_ACTION],
        },
    },
}


# ---------------------------------------------------------------------------
# deterministic signal generation
# ---------------------------------------------------------------------------


def _joint_positions(log_time_ns: int) -> list[float]:
    t = (log_time_ns - ORIGIN_NS) / 1_000_000_000.0
    return [round(0.4 * math.sin(0.7 * t + 0.3 * index), 9) for index in range(6)]


def _action_values(log_time_ns: int) -> list[float]:
    # Semantics: target joint positions, slightly ahead of the measured state.
    t = (log_time_ns - ORIGIN_NS) / 1_000_000_000.0 + 0.02
    return [round(0.4 * math.sin(0.7 * t + 0.3 * index), 9) for index in range(6)]


def _image_bytes(frame_index: int, rng: np.random.RandomState) -> bytes:
    """Deterministic, high-contrast 160x120 rgb8 frame that stays sharp after normalisation.

    The canonical pipeline upscales this 4:3 frame to the 640x480 letterbox canvas
    and H.264-encodes it.  Fine per-pixel noise does not survive that upscale, so
    the synthetic scene uses 4x4-pixel high-contrast blocks (random black/white
    with a moving gradient): the block edges are real image structure at source
    resolution and keep a 640x480 grayscale Laplacian variance well above the
    ``BLUR_FRAME_RATE`` heuristic on the canonical video, which is required for
    ``F05_mcap_good`` to be QC PASS (SPEC 13.1).
    """
    block = rng.randint(0, 2, size=(IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4, 3), dtype=np.int16)
    pattern = np.repeat(np.repeat(block, 4, axis=0), 4, axis=1) * 215 + 20
    x = np.arange(IMAGE_WIDTH, dtype=np.int16)[None, :]
    y = np.arange(IMAGE_HEIGHT, dtype=np.int16)[:, None]
    gradient = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.int16)
    gradient[..., 0] = (x + frame_index * 7) % 32
    gradient[..., 1] = (y + frame_index * 11) % 32
    gradient[..., 2] = ((x + y) // 2 + frame_index * 13) % 32
    return np.clip(pattern + gradient, 0, 255).astype(np.uint8).tobytes()


def build_message_stream(*, drop_state_range_s: tuple[float, float] | None = None) -> list[tuple[int, str, bytes]]:
    """Build every profile message as ``(log_time_ns, channel_key, cdr_bytes)``."""
    rng = np.random.RandomState(SEED)
    messages: list[tuple[int, str, bytes]] = []

    step = IMAGE_WIDTH * 3
    for frame_index in range(IMAGE_COUNT):
        log_time = ORIGIN_NS + int(round(frame_index * 1_000_000_000 / IMAGE_FPS))
        messages.append(
            (
                log_time,
                "camera",
                encode_image(
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    encoding="rgb8",
                    data=_image_bytes(frame_index, rng),
                    step=step,
                    header_stamp_ns=log_time + IMAGE_HEADER_OFFSET_NS,
                    frame_id=FRAME_ID,
                ),
            )
        )

    drop_low = drop_high = None
    if drop_state_range_s is not None:
        drop_low = ORIGIN_NS + int(round(drop_state_range_s[0] * 1_000_000_000))
        drop_high = ORIGIN_NS + int(round(drop_state_range_s[1] * 1_000_000_000))

    for index in range(JOINT_COUNT):
        log_time = ORIGIN_NS - JOINT_HISTORY_NS + index * JOINT_PERIOD_NS
        if drop_low is not None and drop_high is not None and drop_low <= log_time <= drop_high:
            continue
        messages.append(
            (
                log_time,
                "state",
                encode_joint_state(
                    names=[f"joint_{position}" for position in range(1, 7)],
                    position=_joint_positions(log_time),
                    header_stamp_ns=log_time + JOINT_HEADER_OFFSET_NS,
                    frame_id="base_link",
                ),
            )
        )

    for index in range(ACTION_COUNT):
        log_time = ORIGIN_NS - ACTION_HISTORY_NS + index * ACTION_PERIOD_NS
        messages.append(
            (log_time, "action", encode_float64_multi_array(data=_action_values(log_time)))
        )

    # MCAP/rosbag2 consumers expect non-decreasing log_time in file order.
    messages.sort(key=lambda item: (item[0], item[1]))
    return messages


def write_mcap(path: Path, messages: Sequence[tuple[int, str, bytes]]) -> None:
    """Write one deterministic ROS 2 profile MCAP container."""
    from mcap.writer import Writer

    path.parent.mkdir(parents=True, exist_ok=True)
    channels = (
        ("camera", TOPIC_IMAGE, IMAGE_TYPE),
        ("state", TOPIC_JOINT_STATES, JOINT_STATE_TYPE),
        ("action", TOPIC_ACTION, FLOAT64_MULTI_ARRAY_TYPE),
    )
    with path.open("wb") as stream:
        writer = Writer(stream)
        try:
            writer.start(profile="ros2", library=LIBRARY)
        except TypeError:  # pragma: no cover - older mcap writer signature
            writer.start()
        channel_ids: dict[str, int] = {}
        for key, topic, message_type in channels:
            schema_id = writer.register_schema(
                name=message_type,
                encoding="ros2msg",
                data=ROS2_MSG_DEFINITIONS[message_type].encode("utf-8"),
            )
            channel_ids[key] = writer.register_channel(
                topic=topic,
                message_encoding="cdr",
                schema_id=schema_id,
            )
        for log_time, key, data in messages:
            writer.add_message(
                channel_id=channel_ids[key],
                log_time=log_time,
                publish_time=log_time,
                data=data,
            )
        writer.finish()


# ---------------------------------------------------------------------------
# manifest helpers (shared lock with fixtures/generate_synthetic.py)
# ---------------------------------------------------------------------------


def _manifest_path() -> Path:
    return REPOSITORY_ROOT / "fixtures" / "manifest.json"


def _acquire_manifest_lock(
    lock_path: Path | None = None,
    timeout_s: float = MANIFEST_LOCK_TIMEOUT_S,
) -> Path:
    lock_path = lock_path or (REPOSITORY_ROOT / "fixtures" / MANIFEST_LOCK_NAME)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > MANIFEST_LOCK_STALE_S:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for the fixtures manifest lock at {lock_path}"
                ) from None
            time.sleep(0.05)
            continue
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        finally:
            os.close(descriptor)
        return lock_path


def _release_manifest_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best effort cleanup
        pass


def merge_manifest_entries(
    entries: Sequence[dict[str, Any]],
    *,
    manifest_path: Path | None = None,
    lock: bool = True,
) -> dict[str, Any]:
    """Read-modify-write ``fixtures/manifest.json`` preserving every other entry.

    Tolerates a top-level list legacy format, a missing file and concurrent edits
    (exclusive-create lock file shared with ``fixtures/generate_synthetic.py``).
    Only entries whose ``fixture_id`` we own are replaced; all others are kept
    byte-for-byte in their original order.
    """
    manifest_path = manifest_path or _manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    incoming = {entry["fixture_id"]: entry for entry in entries}
    lock_path = _acquire_manifest_lock(manifest_path.parent / MANIFEST_LOCK_NAME) if lock else None
    try:
        if manifest_path.is_file():
            try:
                document = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                document = {}
        else:
            document = {}
        if isinstance(document, list):
            document = {"fixtures": document}
        elif not isinstance(document, dict):
            document = {}
        document.setdefault("schema_version", "fixtures-manifest/1.0")
        document.setdefault("seed", SEED)
        document.setdefault("data_origin", "synthetic")
        fixtures = document.get("fixtures")
        if not isinstance(fixtures, list):
            fixtures = []
        merged: list[dict[str, Any]] = []
        replaced: set[str] = set()
        for existing in fixtures:
            fixture_id = existing.get("fixture_id") if isinstance(existing, dict) else None
            if fixture_id in incoming:
                merged.append(incoming[fixture_id])
                replaced.add(fixture_id)
            else:
                merged.append(existing)
        for fixture_id in sorted(incoming):
            if fixture_id not in replaced:
                merged.append(incoming[fixture_id])
        document["fixtures"] = merged
        if not document.get("generated_at"):
            document["generated_at"] = "2026-10-29T00:00:00Z"
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        return document
    finally:
        if lock_path is not None:
            _release_manifest_lock(lock_path)


# ---------------------------------------------------------------------------
# public generation API
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def generate_fixtures(
    output_dir: Path | str | None = None,
    *,
    update_manifest: bool = True,
    fixture_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate the requested F05/F06 fixtures and return their manifest entries.

    ``fixture_ids`` lets tests generate only the fixture they need; the manifest
    is then updated with just those entries (never touching other fixtures).
    """
    output_dir = Path(output_dir) if output_dir is not None else REPOSITORY_ROOT / "fixtures" / "golden" / "mcap"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(fixture_ids) if fixture_ids is not None else list(FIXTURE_SPECS)
    unknown = [fixture_id for fixture_id in selected if fixture_id not in FIXTURE_SPECS]
    if unknown:
        raise ValueError(f"Unknown MCAP fixture ids: {unknown}")
    entries: list[dict[str, Any]] = []
    for fixture_id in selected:
        spec = FIXTURE_SPECS[fixture_id]
        path = output_dir / f"{fixture_id}.mcap"
        messages = build_message_stream(drop_state_range_s=spec["drop_state_range_s"])
        write_mcap(path, messages)
        entries.append(
            {
                "fixture_id": fixture_id,
                "path": _repo_relative(path),
                "kind": "file",
                "content_type": "application/vnd.mcap",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "data_origin": "synthetic",
                "seed": SEED,
                "expected": dict(spec["expected"]),
                "generated_by": "fixtures/make_mcap_ros2.py",
                "status": "generated",
                "notes": spec["notes"],
            }
        )
    if update_manifest:
        merge_manifest_entries(entries)
        redundant = REPOSITORY_ROOT / "fixtures" / "manifest_mcap.json"
        redundant.write_text(
            json.dumps(
                {
                    "schema_version": "fixtures-manifest/1.0",
                    "seed": SEED,
                    "data_origin": "synthetic",
                    "generated_by": "fixtures/make_mcap_ros2.py",
                    "fixtures": entries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(REPOSITORY_ROOT / "fixtures" / "golden" / "mcap"),
        help="directory for the generated .mcap fixtures",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not touch fixtures/manifest.json (used by isolated tests)",
    )
    arguments = parser.parse_args(argv)
    entries = generate_fixtures(arguments.output_dir, update_manifest=not arguments.no_manifest)
    print(json.dumps(entries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
