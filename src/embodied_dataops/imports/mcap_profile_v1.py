"""``MCAP_ROS2_PROFILE_V1`` importer (SPEC 5.4).

Scope, deliberately narrow
--------------------------
Exactly one Episode per MCAP file, ROS 2 **CDR** encoding, and exactly three
topics:

===============================  =========================================  ==========================
topic                            type                                       handling
===============================  =========================================  ==========================
``/camera/front/image_raw``      ``sensor_msgs/msg/Image``                  ``rgb8`` / ``bgr8`` only, complete
                                                                            single frames
``/joint_states``                ``sensor_msgs/msg/JointState``             exactly ``joint_1``..``joint_6``,
                                                                            positions sorted by joint name
``/policy/action``               ``std_msgs/msg/Float64MultiArray``         exactly six ``float64``; semantics
                                                                            = **target joint position**, not
                                                                            torque and not velocity
===============================  =========================================  ==========================

Anything else -- another topic, another schema type, a ROS 1 bag, ROS 1/no CDR
message encoding, or schema information the reader cannot establish -- is
rejected with :class:`ErrorCode.UNSUPPORTED_TOPIC_PROFILE` and a readable
message; no stack trace ever reaches the API/UI.

Timebase (SPEC 5.4, the part tests pin down)
-------------------------------------------
All three streams are timed by the MCAP ``log_time`` in nanoseconds.  The
origin ``t=0`` is the ``log_time`` of the **first valid camera frame**; every
other timestamp is stored relative to it.  ``header.stamp`` is never used for
alignment: it is preserved (with ``header.stamp - log_time`` deltas) purely as a
diagnostic, because ``log_time`` is the recording system's receive time, not the
camera exposure time or a hard real-time synchronisation source.  This limit is
documented, not papered over.

The 10 Hz grid covers ``[first camera frame, min(last camera, last state, last
action)]``.  Per grid point:

* image: nearest frame within ``<= 50 ms``;
* state: linear interpolation whose **both** bracketing samples are within
  ``<= 50 ms`` of the target (a state is a continuous scalar signal);
* action: zero-order hold strictly towards the **past** within ``<= 100 ms`` --
  a future action command is never used to explain an earlier sample.

Missing points are recorded as ``SYNC_MISSING_FRAME`` / ``SYNC_MISSING_STATE`` /
``SYNC_STALE_ACTION`` diagnostics; values are never zero-filled and the episode
is not silently truncated.  If any grid point is incomplete the Episode is
normalised with ``normalization_valid=false`` (metadata + diagnostics only, no
publishable ``samples.parquet``) so ``QUALITY_CHECK`` can still emit the explicit
finding (SPEC 6.1).  An Episode longer than 600 s raises
``SOURCE_DURATION_LIMIT``.

Payload contract consumed by ``NORMALIZE_EPISODE``
--------------------------------------------------
``EpisodeSource.payload`` hands the normalizer the raw per-stream arrays
(absolute log times, decoded values, header diagnostics) *and* the already
aligned 10 Hz rows, so the canonical writer never has to re-implement
alignment::

    payload = {
      "origin_ns": int, "duration_ns": int,
      "streams": {"camera": {...frames...}, "state": {...}, "action": {...}},
      "stream_times": {"camera": [...], "state": [...], "action": [...]},
      "grid": [relative_t_ns, ...],
      "samples": [{"sample_index", "t_ns", "frame_index", "state", "action",
                   "state_source_t_ns", "action_source_t_ns",
                   "camera_source_index"}],
      "alignment": {...sync diagnostics / metrics...},
      "diagnostics": [...],
      "clock_domains": {...}, "header_to_log_delta_ns": {...},
      "normalization_valid": bool,
    }

The dual reader backend (``rosbag2_py`` in the ROS 2 Jazzy worker vs the local
``mcap`` pip + hand-written CDR fallback) is documented in
``services/ros2_adapter/reader.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.canonical.alignment import compute_p95
from embodied_dataops.canonical.contract import ROBOT_JOINT_NAMES, ROBOT_PROFILE, SampleRow
from embodied_dataops.canonical.timebase import (
    ACTION_ZOH_TOLERANCE_NS,
    IMAGE_TOLERANCE_NS,
    MAX_EPISODE_DURATION_NS,
    STATE_TOLERANCE_NS,
    generate_grid,
    interpolate_state,
    nearest_match,
    zoh_action,
)
from embodied_dataops.domain.enums import EpisodeType, SourceType
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.imports.base import EpisodeSource, ImportContext, ParsedSource, SourceParser

#: MCAP magic number of a valid container (SPEC 5.4 accepts only MCAP).
MCAP_MAGIC = b"\x89MCAP0\r\n"

#: Diagnostics emitted by the parser; these codes are also quality rule codes
#: (SPEC 7.1) so downstream reporting is uniform.
DIAG_SYNC_MISSING_FRAME = "SYNC_MISSING_FRAME"
DIAG_SYNC_MISSING_STATE = "SYNC_MISSING_STATE"
DIAG_SYNC_STALE_ACTION = "SYNC_STALE_ACTION"
DIAG_DIMENSION_MISMATCH = "DIMENSION_MISMATCH"


def _ros2_reader():
    """Import the adapter package, tolerating a non-installed ``services`` tree."""
    try:
        from services.ros2_adapter import reader  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        repository_root = Path(__file__).resolve().parents[3]
        if str(repository_root) not in sys.path:
            sys.path.insert(0, str(repository_root))
        from services.ros2_adapter import reader  # type: ignore[import-not-found]
    return reader


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """Aligned 10 Hz rows plus exact per-grid-point diagnostics."""

    origin_ns: int
    grid_rel_ns: list[int] = field(default_factory=list)
    rows: list[SampleRow] = field(default_factory=list)
    camera_source_indexes: list[int] = field(default_factory=list)
    missing_frame_indexes: list[int] = field(default_factory=list)
    missing_state_indexes: list[int] = field(default_factory=list)
    stale_action_indexes: list[int] = field(default_factory=list)
    frame_errors_ns: list[int] = field(default_factory=list)
    state_errors_ns: list[int] = field(default_factory=list)
    action_ages_ns: list[int] = field(default_factory=list)
    max_abs_sync_error_ms: float = 0.0
    p95_abs_sync_error_ms: float = 0.0
    coverage_start_ns: int | None = None
    coverage_end_ns: int | None = None

    @property
    def missing_grid_indexes(self) -> list[int]:
        return sorted(
            set(self.missing_frame_indexes) | set(self.missing_state_indexes) | set(self.stale_action_indexes)
        )

    @property
    def missing_sample_count(self) -> int:
        return len(self.missing_grid_indexes)

    @property
    def valid_sample_count(self) -> int:
        return len(self.grid_rel_ns) - self.missing_sample_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "grid_points": len(self.grid_rel_ns),
            "missing_sample_count": self.missing_sample_count,
            "valid_sample_count": self.valid_sample_count,
            "max_abs_sync_error_ms": self.max_abs_sync_error_ms,
            "p95_abs_sync_error_ms": self.p95_abs_sync_error_ms,
            "missing_frame_indexes": self.missing_frame_indexes[:500],
            "missing_state_indexes": self.missing_state_indexes[:500],
            "stale_action_indexes": self.stale_action_indexes[:500],
            "coverage_start_ns": self.coverage_start_ns,
            "coverage_end_ns": self.coverage_end_ns,
            "origin_ns": self.origin_ns,
        }

    def diagnostics(self) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        if self.missing_frame_indexes:
            diagnostics.append(
                {
                    "code": DIAG_SYNC_MISSING_FRAME,
                    "severity": "ERROR",
                    "message": (
                        f"{len(self.missing_frame_indexes)} of {len(self.grid_rel_ns)} 10 Hz grid points "
                        f"have no camera frame within {IMAGE_TOLERANCE_NS // 1_000_000} ms"
                    ),
                    "details": {"grid_indexes": self.missing_frame_indexes[:500]},
                }
            )
        if self.missing_state_indexes:
            diagnostics.append(
                {
                    "code": DIAG_SYNC_MISSING_STATE,
                    "severity": "ERROR",
                    "message": (
                        f"{len(self.missing_state_indexes)} of {len(self.grid_rel_ns)} 10 Hz grid points "
                        "have no valid bracketing joint-state interpolation "
                        f"(both endpoints must be within {STATE_TOLERANCE_NS // 1_000_000} ms)"
                    ),
                    "details": {"grid_indexes": self.missing_state_indexes[:500]},
                }
            )
        if self.stale_action_indexes:
            diagnostics.append(
                {
                    "code": DIAG_SYNC_STALE_ACTION,
                    "severity": "ERROR",
                    "message": (
                        f"{len(self.stale_action_indexes)} of {len(self.grid_rel_ns)} 10 Hz grid points "
                        "have no action message at or before the sample time within "
                        f"{ACTION_ZOH_TOLERANCE_NS // 1_000_000} ms (future actions are never used)"
                    ),
                    "details": {"grid_indexes": self.stale_action_indexes[:500]},
                }
            )
        return diagnostics


def align_mcap_streams(
    *,
    origin_ns: int,
    frame_times_ns: Sequence[int],
    state_times_ns: Sequence[int],
    state_values: Sequence[Sequence[float]],
    action_times_ns: Sequence[int],
    action_values: Sequence[Sequence[float]],
    coverage_end_ns: int,
    image_tolerance_ns: int = IMAGE_TOLERANCE_NS,
    state_tolerance_ns: int = STATE_TOLERANCE_NS,
    action_tolerance_ns: int = ACTION_ZOH_TOLERANCE_NS,
) -> AlignmentResult:
    """Build the 10 Hz grid and rows with exact diagnostics.

    Uses the shared ``canonical.timebase`` primitives, so the alignment semantics
    are identical to ``canonical.alignment.align_robot_streams``; the difference
    here is that diagnostics keep the *grid index* (not the running row count)
    and that the start of the grid is always the first camera frame, so a missing
    state/action history is reported instead of silently shrinking the episode.
    """
    result = AlignmentResult(origin_ns=origin_ns)
    if not frame_times_ns:
        return result
    if len(state_times_ns) != len(state_values):
        raise ApiError(ErrorCode.INVALID_INPUT, "JointState times/values length mismatch")
    if len(action_times_ns) != len(action_values):
        raise ApiError(ErrorCode.INVALID_INPUT, "Action times/values length mismatch")

    coverage_start = int(frame_times_ns[0])
    coverage_end = min(int(coverage_end_ns), int(frame_times_ns[-1]))
    if state_times_ns:
        coverage_end = min(coverage_end, int(state_times_ns[-1]))
    if action_times_ns:
        coverage_end = min(coverage_end, int(action_times_ns[-1]))
    result.coverage_start_ns = coverage_start
    result.coverage_end_ns = coverage_end

    grid = generate_grid(
        origin_ns=origin_ns, coverage_start_ns=coverage_start, coverage_end_ns=coverage_end
    )
    result.grid_rel_ns = grid
    all_errors_ms: list[float] = []
    for grid_index, t_rel in enumerate(grid):
        target = origin_ns + t_rel
        frame = nearest_match(frame_times_ns, target, image_tolerance_ns)
        if frame is None:
            result.missing_frame_indexes.append(grid_index)
            continue
        state = interpolate_state(state_times_ns, state_values, target, state_tolerance_ns)
        if state is None:
            result.missing_state_indexes.append(grid_index)
            continue
        action = zoh_action(action_times_ns, action_values, target, action_tolerance_ns)
        if action is None:
            result.stale_action_indexes.append(grid_index)
            continue
        row_index = len(result.rows)
        result.rows.append(
            SampleRow(
                sample_index=row_index,
                t_ns=t_rel,
                frame_index=row_index,
                state=state.value,
                action=tuple(float(v) for v in action_values[action.index]),
                state_source_t_ns=state.source_t_ns - origin_ns,
                action_source_t_ns=action.source_t_ns - origin_ns,
            )
        )
        result.camera_source_indexes.append(frame.index)
        result.frame_errors_ns.append(int(frame.abs_error_ns))
        result.state_errors_ns.append(int(state.max_endpoint_error_ns))
        result.action_ages_ns.append(int(action.age_ns))
        all_errors_ms.append(
            max(frame.abs_error_ns, state.max_endpoint_error_ns, action.age_ns) / 1e6
        )
    result.max_abs_sync_error_ms = round(max(all_errors_ms), 6) if all_errors_ms else 0.0
    result.p95_abs_sync_error_ms = round(compute_p95(all_errors_ms), 6)
    return result


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class McapProfileV1Parser(SourceParser):
    """``MCAP_ROS2_PROFILE_V1`` adapter (single Episode, fixed three-topic profile)."""

    source_type = SourceType.MCAP_ROS2_PROFILE_V1

    def __init__(self, *, backend: str = "auto") -> None:
        #: "auto" prefers rosbag2_py (production) and falls back to mcap pip.
        self.backend = backend
        self._last_probe: dict[str, Any] = {}

    # -- helpers -----------------------------------------------------------
    def _single_mcap_path(self, ctx: ImportContext) -> Path:
        source = Path(ctx.source_path)
        if source.is_dir():
            candidates = sorted(source.glob("*.mcap"))
            if len(candidates) != 1:
                raise ApiError(
                    ErrorCode.UNSUPPORTED_TOPIC_PROFILE,
                    "MCAP_ROS2_PROFILE_V1 requires exactly one single-Episode .mcap file",
                    details={"mcap_files": [path.name for path in candidates]},
                )
            source = candidates[0]
        if not source.is_file():
            raise ApiError(ErrorCode.SOURCE_NOT_FOUND, f"MCAP source not found: {source.name}")
        try:
            with source.open("rb") as handle:
                head = handle.read(8)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            raise ApiError(ErrorCode.SOURCE_NOT_FOUND, "MCAP source cannot be read") from exc
        if head != MCAP_MAGIC:
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "File is not an MCAP container (bad magic bytes); MCAP_ROS2_PROFILE_V1 "
                "accepts ROS 2 CDR .mcap files only",
                details={"filename": source.name},
            )
        return source

    def _reader_backend(self) -> str:
        reader = _ros2_reader()
        if self.backend == "auto":
            return "auto"
        return reader.resolve_backend(self.backend)

    # -- SourceParser ------------------------------------------------------
    def validate(self, ctx: ImportContext) -> dict[str, Any]:
        """Cheap structural validation: container, profile topics, CDR encoding.

        The authoritative 600 s duration check happens in :meth:`parse` on camera
        frame timestamps (a bag may carry a little joint/action history before
        the first camera frame, so the container statistics span is not the
        Episode duration).
        """
        reader = _ros2_reader()
        path = self._single_mcap_path(ctx)
        probe = reader.probe_mcap_profile(path, backend=self._reader_backend())
        self._last_probe = {
            "backend": probe.backend,
            "profile": probe.profile,
            "topics": dict(probe.topics),
            "message_counts": dict(probe.message_counts or {}),
            "duration_ns": probe.duration_ns,
        }
        return dict(self._last_probe)

    def parse(self, ctx: ImportContext) -> ParsedSource:
        reader = _ros2_reader()
        path = self._single_mcap_path(ctx)
        backend = self._reader_backend()
        data = reader.read_mcap_profile(path, backend=backend)

        if not data.camera:
            raise ApiError(
                ErrorCode.UNSUPPORTED_TOPIC_PROFILE,
                f"MCAP contains no decodable frames on {reader.TOPIC_IMAGE}",
            )

        origin_ns = int(data.camera[0].log_time_ns)
        frame_times = [int(frame.log_time_ns) for frame in data.camera]
        span_ns = frame_times[-1] - origin_ns
        if span_ns > MAX_EPISODE_DURATION_NS:
            raise ApiError(
                ErrorCode.SOURCE_DURATION_LIMIT,
                "MCAP episode exceeds the 600 s limit of MCAP_ROS2_PROFILE_V1",
                details={"duration_ns": span_ns, "limit_ns": MAX_EPISODE_DURATION_NS},
            )

        state_times = [int(sample.log_time_ns) for sample in data.joint_states]
        action_times = [int(sample.log_time_ns) for sample in data.actions]

        diagnostics: list[dict[str, Any]] = []
        joint_names, state_values, joint_diag = self._normalise_joint_states(data)
        diagnostics.extend(joint_diag)
        action_values, action_diag = self._normalise_actions(data)
        diagnostics.extend(action_diag)

        alignment = align_mcap_streams(
            origin_ns=origin_ns,
            frame_times_ns=frame_times,
            state_times_ns=state_times,
            state_values=state_values,
            action_times_ns=action_times,
            action_values=action_values,
            coverage_end_ns=frame_times[-1],
        )
        diagnostics.extend(alignment.diagnostics())
        normalization_valid = bool(alignment.grid_rel_ns) and not diagnostics

        payload = self._build_payload(
            data=data,
            origin_ns=origin_ns,
            alignment=alignment,
            state_values=state_values,
            action_values=action_values,
            joint_names=joint_names,
            diagnostics=diagnostics,
            normalization_valid=normalization_valid,
            backend=backend,
        )
        duration_ns = (
            alignment.grid_rel_ns[-1] + SAMPLE_INTERVAL_NS if alignment.grid_rel_ns else max(span_ns, 0)
        )
        episode = EpisodeSource(
            source_episode_index=0,
            episode_type=EpisodeType.ROBOT_EPISODE,
            task_label=str(ctx.mapping.get("task_label") or ""),
            duration_ns=int(duration_ns),
            payload=payload,
            diagnostics=diagnostics,
            normalization_valid=normalization_valid,
        )
        return ParsedSource(
            episodes=[episode],
            metadata={
                "parser": "mcap_profile_v1",
                "backend": backend,
                "profile": data.profile,
                "episode_count": 1,
                "clock_domain": "mcap_log_time_episode_relative",
                "message_counts": dict(data.message_counts),
            },
            warnings=[
                "MCAP log_time is the recording system receive time; it is not camera "
                "exposure time or a hard real-time synchronisation source",
            ],
        )

    # -- stream normalisation ---------------------------------------------
    @staticmethod
    def _normalise_joint_states(data) -> tuple[list[str], list[tuple[float, ...]], list[dict[str, Any]]]:
        """Validate joint names and reorder positions into sorted joint-name order."""
        diagnostics: list[dict[str, Any]] = []
        required = list(ROBOT_JOINT_NAMES)
        all_names_valid = True
        observed_names: set[str] = set()
        values: list[tuple[float, ...]] = []
        for sample in data.joint_states:
            names = list(sample.message.names)
            positions = list(sample.message.position)
            observed_names.update(names)
            if len(set(names)) != len(names) or set(names) != set(required):
                all_names_valid = False
                values.append(tuple(float(value) for value in positions))
                continue
            order = sorted(range(len(names)), key=lambda index: names[index])
            values.append(tuple(float(positions[index]) for index in order))
        if all_names_valid:
            joint_names = required
        else:
            joint_names = sorted(observed_names)
            diagnostics.append(
                {
                    "code": DIAG_DIMENSION_MISMATCH,
                    "severity": "ERROR",
                    "message": (
                        "/joint_states must provide exactly "
                        f"{required} without duplicates; observed names: {sorted(observed_names)}"
                    ),
                    "details": {"observed_joint_names": sorted(observed_names)},
                }
            )
        return joint_names, values, diagnostics

    @staticmethod
    def _normalise_actions(data) -> tuple[list[tuple[float, ...]], list[dict[str, Any]]]:
        diagnostics: list[dict[str, Any]] = []
        values: list[tuple[float, ...]] = []
        bad_lengths: set[int] = set()
        for sample in data.actions:
            raw = tuple(float(value) for value in sample.message.data)
            values.append(raw)
            if len(raw) != len(ROBOT_JOINT_NAMES):
                bad_lengths.add(len(raw))
        if bad_lengths:
            diagnostics.append(
                {
                    "code": DIAG_DIMENSION_MISMATCH,
                    "severity": "ERROR",
                    "message": (
                        f"/policy/action must contain exactly 6 float64 target joint positions; "
                        f"observed lengths: {sorted(bad_lengths)}"
                    ),
                    "details": {
                        "observed_action_dims": sorted(bad_lengths),
                        "semantics": "target joint position (not torque, not velocity)",
                    },
                }
            )
        return values, diagnostics

    # -- payload -----------------------------------------------------------
    @staticmethod
    def _header_deltas(samples, *, has_header: bool = True) -> list[int]:
        deltas: list[int] = []
        for sample in samples:
            header_stamp = getattr(sample.message, "header_stamp_ns", None)
            if has_header and header_stamp is not None:
                deltas.append(int(header_stamp) - int(sample.log_time_ns))
        return deltas

    @staticmethod
    def _delta_summary(deltas: Sequence[int]) -> dict[str, Any]:
        if not deltas:
            return {"count": 0}
        array = np.asarray(list(deltas), dtype=np.int64)
        return {
            "count": int(array.size),
            "min_ns": int(array.min()),
            "max_ns": int(array.max()),
            "mean_ns": round(float(array.mean()), 3),
            "clock_note": "header.stamp - mcap log_time; diagnostic only, never used for alignment",
        }

    def _build_payload(
        self,
        *,
        data,
        origin_ns: int,
        alignment: AlignmentResult,
        state_values: Sequence[Sequence[float]],
        action_values: Sequence[Sequence[float]],
        joint_names: list[str],
        diagnostics: list[dict[str, Any]],
        normalization_valid: bool,
        backend: str,
    ) -> dict[str, Any]:
        camera_header_deltas = self._header_deltas(data.camera)
        state_header_deltas = self._header_deltas(data.joint_states)

        camera_stream = {
            "topic": _ros2_reader().TOPIC_IMAGE,
            "clock_domain": "mcap_log_time",
            "times_ns": [int(frame.log_time_ns) for frame in data.camera],
            "header_stamp_ns": [int(frame.message.header_stamp_ns or 0) for frame in data.camera],
            "header_minus_log_ns": camera_header_deltas,
            "frames": [
                {
                    "source_index": index,
                    "log_time_ns": int(frame.log_time_ns),
                    "header_stamp_ns": frame.message.header_stamp_ns,
                    "frame_id": frame.message.frame_id,
                    "width": frame.message.width,
                    "height": frame.message.height,
                    "encoding": frame.message.encoding,
                    "step": frame.message.step,
                    "data": frame.message.data,
                }
                for index, frame in enumerate(data.camera)
            ],
        }
        state_stream = {
            "topic": _ros2_reader().TOPIC_JOINT_STATES,
            "clock_domain": "mcap_log_time",
            "times_ns": [int(sample.log_time_ns) for sample in data.joint_states],
            "header_stamp_ns": [int(sample.message.header_stamp_ns or 0) for sample in data.joint_states],
            "header_minus_log_ns": state_header_deltas,
            "joint_names": list(joint_names),
            "values": [list(values) for values in state_values],
        }
        action_stream = {
            "topic": _ros2_reader().TOPIC_ACTION,
            "clock_domain": "mcap_log_time",
            "times_ns": [int(sample.log_time_ns) for sample in data.actions],
            "values": [list(values) for values in action_values],
            "semantics": "target_joint_position",
        }
        samples = [
            {
                "sample_index": row.sample_index,
                "t_ns": row.t_ns,
                "frame_index": row.frame_index,
                "state": list(row.state or []),
                "action": list(row.action or []),
                "state_source_t_ns": int(row.state_source_t_ns or 0),
                "action_source_t_ns": int(row.action_source_t_ns or 0),
                "camera_source_index": alignment.camera_source_indexes[index],
            }
            for index, row in enumerate(alignment.rows)
        ]
        stream_times = {
            "camera": camera_stream["times_ns"],
            "state": state_stream["times_ns"],
            "action": action_stream["times_ns"],
        }
        # The NORMALIZE_EPISODE handler copies ``payload["alignment"]`` into its
        # task output_json / metadata extras verbatim, and QUALITY_CHECK reads the
        # grid and source timestamps from there: embed them in the alignment block
        # so the frame-gap and non-monotonic timestamp rules have real evidence in
        # the pipeline (they are also exposed at the top level for clarity).
        alignment_payload = alignment.as_dict()
        alignment_payload["grid"] = list(alignment.grid_rel_ns)
        alignment_payload["stream_times"] = stream_times
        return {
            "origin_ns": origin_ns,
            "duration_ns": (
                alignment.grid_rel_ns[-1] + SAMPLE_INTERVAL_NS if alignment.grid_rel_ns else 0
            ),
            "streams": {"camera": camera_stream, "state": state_stream, "action": action_stream},
            "stream_times": stream_times,
            "grid": list(alignment.grid_rel_ns),
            "samples": samples,
            "alignment": alignment_payload,
            "diagnostics": diagnostics,
            "clock_domains": {
                "camera": "mcap_log_time",
                "state": "mcap_log_time",
                "action": "mcap_log_time",
            },
            "header_to_log_delta_ns": {
                "camera": self._delta_summary(camera_header_deltas),
                "state": self._delta_summary(state_header_deltas),
                "action": {"count": 0, "message_shapes": "no header on std_msgs/Float64MultiArray"},
            },
            "robot_profile": ROBOT_PROFILE,
            "joint_names": list(joint_names),
            "state_dim": len(state_values[0]) if state_values else 0,
            "action_dim": len(action_values[0]) if action_values else 0,
            "normalization_valid": normalization_valid,
            "reader_backend": backend,
        }


__all__ = [
    "AlignmentResult",
    "DIAG_DIMENSION_MISMATCH",
    "DIAG_SYNC_MISSING_FRAME",
    "DIAG_SYNC_MISSING_STATE",
    "DIAG_SYNC_STALE_ACTION",
    "MCAP_MAGIC",
    "McapProfileV1Parser",
    "align_mcap_streams",
]
