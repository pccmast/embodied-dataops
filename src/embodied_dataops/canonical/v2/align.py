"""Alignment of native v2 streams onto the 10 Hz training grid (V2 SPEC 2.3/2.4).

Two modes are implemented and never mixed:

``native``
    Independent source streams.  Images use the nearest frame of each camera
    independently (``<= 50 ms`` by default), robot state is linearly interpolated
    between a bracketing pair (both endpoints within tolerance), and actions are
    zero-order-held strictly towards the past (``<= 100 ms``; a future action is
    never used).  A grid point without a required sample is ``valid=false`` with
    ``null`` arrays -- it is never zero-filled and it is never dropped from the
    view, so the missing data stays auditable.

``prealigned``
    LeRobot v3 rows already pair state, action and camera frames.  Row pairing is
    preserved (the nearest source row is copied as one unit); no interpolation,
    no re-timing and **no claim** that independent 100/50 Hz native signals were
    produced.

The aligned view is derived data: the native indices keep the original sample
times and payloads.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from embodied_dataops.canonical.alignment import compute_p95
from embodied_dataops.canonical.timebase import (
    ACTION_ZOH_TOLERANCE_NS,
    IMAGE_TOLERANCE_NS,
    STATE_TOLERANCE_NS,
    generate_grid,
    interpolate_state,
    nearest_index,
    nearest_match,
    zoh_action,
)
from embodied_dataops.canonical.v2.contract import ALIGNED_INTERVAL_NS, CAMERA_NAMES
from embodied_dataops.canonical.v2.model import CanonicalV2Source
from embodied_dataops.canonical.v2.parquet import aligned_row

#: Diagnostics use the same codes as the quality rules so evidence is uniform.
DIAG_TIMESTAMP_NON_MONOTONIC = "TIMESTAMP_NON_MONOTONIC"
DIAG_SYNC_MISSING_FRAME = "SYNC_MISSING_FRAME"
DIAG_SYNC_MISSING_STATE = "SYNC_MISSING_STATE"
DIAG_SYNC_STALE_ACTION = "SYNC_STALE_ACTION"
DIAG_DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
DIAG_NONFINITE_VALUE = "NONFINITE_VALUE"

#: Extra aligned-row reason for a prealigned source row that does not exist.
REASON_MISSING_ROW = "missing_source_row"


@dataclass
class AlignmentOutcome:
    """Aligned rows plus the evidence needed for metadata and quality rules."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    grid_rel_ns: list[int] = field(default_factory=list)
    origin_ns: int = 0
    coverage_start_ns: int | None = None
    coverage_end_ns: int | None = None
    mode: str = "native"
    prealigned: bool = False
    camera_frame_indexes: dict[str, list[int | None]] = field(default_factory=dict)
    max_abs_sync_error_ns: int = 0
    p95_abs_sync_error_ns: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    missing_reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.rows)

    @property
    def valid_sample_count(self) -> int:
        return sum(1 for row in self.rows if row["valid"])

    @property
    def invalid_sample_count(self) -> int:
        return self.sample_count - self.valid_sample_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "prealigned": self.prealigned,
            "origin_ns": self.origin_ns,
            "coverage_start_ns": self.coverage_start_ns,
            "coverage_end_ns": self.coverage_end_ns,
            "grid_points": len(self.grid_rel_ns),
            "valid_sample_count": self.valid_sample_count,
            "invalid_sample_count": self.invalid_sample_count,
            "max_abs_sync_error_ms": round(self.max_abs_sync_error_ns / 1e6, 6),
            "p95_abs_sync_error_ms": round(self.p95_abs_sync_error_ns / 1e6, 6),
            "missing_reason_counts": dict(sorted(self.missing_reason_counts.items())),
        }


def _bisect_nearest(times_ns: Sequence[int], target_ns: int) -> int | None:
    if not times_ns:
        return None
    return nearest_index(times_ns, target_ns)


def _dimension_reasons(
    *,
    prefix: str,
    values: Sequence[float] | None,
    expected_dim: int | None,
) -> list[str]:
    reasons: list[str] = []
    if values is None:
        return reasons
    if expected_dim is not None and len(values) != int(expected_dim):
        reasons.append(f"dimension_mismatch:{prefix}")
    if len(values) and not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        reasons.append(f"nonfinite:{prefix}")
    return reasons


def align_native(
    source: CanonicalV2Source,
    *,
    interval_ns: int = ALIGNED_INTERVAL_NS,
    image_tolerance_ns: int = IMAGE_TOLERANCE_NS,
    state_tolerance_ns: int = STATE_TOLERANCE_NS,
    action_tolerance_ns: int = ACTION_ZOH_TOLERANCE_NS,
    expected_state_dim: int | None = None,
    expected_action_dim: int | None = None,
) -> AlignmentOutcome:
    """Nearest-frame / interpolation / ZOH alignment of independent streams."""
    front = source.camera("front")
    if front is None or not front.times_ns:
        return AlignmentOutcome(mode="native")
    cameras = [stream for stream in source.cameras if stream.times_ns]
    origin = int(source.origin_ns or front.times_ns[0])
    coverage_start = int(front.times_ns[0])
    coverage_end = int(front.times_ns[-1])
    grid = generate_grid(
        origin_ns=origin,
        coverage_start_ns=coverage_start,
        coverage_end_ns=coverage_end,
        interval_ns=interval_ns,
    )
    outcome = AlignmentOutcome(
        grid_rel_ns=grid,
        origin_ns=origin,
        coverage_start_ns=coverage_start,
        coverage_end_ns=coverage_end,
        mode="native",
        camera_frame_indexes={stream.name: [] for stream in cameras},
    )
    all_errors_ns: list[float] = []
    missing: dict[str, list[int]] = {}
    for grid_index, t_rel in enumerate(grid):
        target = origin + t_rel
        reasons: list[str] = []
        frame_indexes: dict[str, int | None] = {}
        for camera in cameras:
            match = nearest_match(camera.times_ns, target, image_tolerance_ns)
            index = match.index if match is not None else None
            frame_indexes[camera.name] = index
            outcome.camera_frame_indexes[camera.name].append(index)
            if index is None:
                reasons.append(f"missing_frame:{camera.name}")
                missing.setdefault(DIAG_SYNC_MISSING_FRAME, []).append(grid_index)
        state_values: list[float] | None = None
        action_values: list[float] | None = None
        state_source_t_ns: int | None = None
        action_source_t_ns: int | None = None
        error_ns = 0
        if source.state is not None and source.state.times_ns:
            state = interpolate_state(
                source.state.times_ns, source.state.values, target, state_tolerance_ns
            )
            if state is None:
                reasons.append("missing_state")
                missing.setdefault(DIAG_SYNC_MISSING_STATE, []).append(grid_index)
            else:
                state_values = [float(value) for value in state.value]
                state_source_t_ns = int(state.source_t_ns) - origin
                error_ns = max(error_ns, int(state.max_endpoint_error_ns))
        else:
            reasons.append("missing_state")
            missing.setdefault(DIAG_SYNC_MISSING_STATE, []).append(grid_index)
        if source.action is not None and source.action.times_ns:
            action = zoh_action(
                source.action.times_ns, source.action.values, target, action_tolerance_ns
            )
            if action is None:
                reasons.append("stale_action")
                missing.setdefault(DIAG_SYNC_STALE_ACTION, []).append(grid_index)
            else:
                action_values = [float(value) for value in source.action.values[action.index]]
                action_source_t_ns = int(action.source_t_ns) - origin
                error_ns = max(error_ns, int(action.age_ns))
        else:
            reasons.append("stale_action")
            missing.setdefault(DIAG_SYNC_STALE_ACTION, []).append(grid_index)
        reasons.extend(
            _dimension_reasons(prefix="state", values=state_values, expected_dim=expected_state_dim)
        )
        reasons.extend(
            _dimension_reasons(prefix="action", values=action_values, expected_dim=expected_action_dim)
        )
        valid = not reasons
        if valid:
            row_errors = [error_ns]
            frame_error = 0
            for camera in cameras:
                index = frame_indexes.get(camera.name)
                if index is not None:
                    frame_error = max(
                        frame_error, abs(int(camera.times_ns[index]) - target)
                    )
            row_errors.append(frame_error)
            all_errors_ns.append(float(max(row_errors)))
        for reason in reasons:
            outcome.missing_reason_counts[reason] = outcome.missing_reason_counts.get(reason, 0) + 1
        if not valid:
            state_values = None
            action_values = None
            state_source_t_ns = None
            action_source_t_ns = None
        outcome.rows.append(
            aligned_row(
                t_ns=t_rel,
                frame_indexes=frame_indexes,
                state=state_values,
                action=action_values,
                state_source_t_ns=state_source_t_ns,
                action_source_t_ns=action_source_t_ns,
                valid=valid,
                invalid_reason=reasons,
            )
        )
    outcome.max_abs_sync_error_ns = int(max(all_errors_ns)) if all_errors_ns else 0
    outcome.p95_abs_sync_error_ns = int(compute_p95(all_errors_ns))
    outcome.diagnostics = _build_diagnostics(
        source=source, outcome=outcome, missing=missing, tolerance_ns=image_tolerance_ns
    )
    return outcome


def align_prealigned(
    source: CanonicalV2Source,
    *,
    interval_ns: int = ALIGNED_INTERVAL_NS,
    row_tolerance_ns: int = STATE_TOLERANCE_NS,
    expected_state_dim: int | None = None,
    expected_action_dim: int | None = None,
) -> AlignmentOutcome:
    """Preserve LeRobot row pairing on the 10 Hz grid (no re-alignment claim)."""
    row_times = [int(value) for value in source.prealigned_row_times_ns]
    if not row_times:
        return AlignmentOutcome(mode="prealigned", prealigned=True)
    if source.state is None or source.action is None:
        return AlignmentOutcome(mode="prealigned", prealigned=True)
    state_values = source.state.values
    action_values = source.action.values
    origin = 0
    last = row_times[-1]
    grid = generate_grid(
        origin_ns=origin, coverage_start_ns=0, coverage_end_ns=last, interval_ns=interval_ns
    )
    cameras = [stream for stream in source.cameras]
    outcome = AlignmentOutcome(
        grid_rel_ns=grid,
        origin_ns=origin,
        coverage_start_ns=0,
        coverage_end_ns=last,
        mode="prealigned",
        prealigned=True,
        camera_frame_indexes={stream.name: [] for stream in cameras},
    )
    errors: list[float] = []
    missing: dict[str, list[int]] = {}
    for grid_index, t_rel in enumerate(grid):
        reasons: list[str] = []
        position = bisect.bisect_left(row_times, t_rel)
        candidates = [index for index in (position, position - 1) if 0 <= index < len(row_times)]
        chosen: int | None = None
        best_distance: int | None = None
        for index in sorted(candidates, key=lambda item: (abs(row_times[item] - t_rel), row_times[item])):
            distance = abs(row_times[index] - t_rel)
            if distance <= row_tolerance_ns:
                chosen = index
                best_distance = distance
                break
        frame_indexes: dict[str, int | None] = {}
        for camera in cameras:
            frame_indexes[camera.name] = chosen
            outcome.camera_frame_indexes[camera.name].append(chosen)
        if chosen is None:
            reasons.append(REASON_MISSING_ROW)
            if "front" in [camera.name for camera in cameras]:
                reasons.append("missing_frame:front")
            reasons.extend(["missing_state", "stale_action"])
            missing.setdefault(DIAG_SYNC_MISSING_FRAME, []).append(grid_index)
            missing.setdefault(DIAG_SYNC_MISSING_STATE, []).append(grid_index)
            missing.setdefault(DIAG_SYNC_STALE_ACTION, []).append(grid_index)
        else:
            state_row = [float(value) for value in state_values[chosen]]
            action_row = [float(value) for value in action_values[chosen]]
            reasons.extend(
                _dimension_reasons(prefix="state", values=state_row, expected_dim=expected_state_dim)
            )
            reasons.extend(
                _dimension_reasons(prefix="action", values=action_row, expected_dim=expected_action_dim)
            )
        valid = not reasons
        for reason in reasons:
            outcome.missing_reason_counts[reason] = outcome.missing_reason_counts.get(reason, 0) + 1
        if valid:
            errors.append(float(best_distance or 0))
            outcome.rows.append(
                aligned_row(
                    t_ns=t_rel,
                    frame_indexes=frame_indexes,
                    state=[float(value) for value in state_values[chosen]],
                    action=[float(value) for value in action_values[chosen]],
                    state_source_t_ns=row_times[chosen],
                    action_source_t_ns=row_times[chosen],
                    valid=True,
                    invalid_reason=[],
                )
            )
        else:
            outcome.rows.append(
                aligned_row(
                    t_ns=t_rel,
                    frame_indexes=frame_indexes,
                    state=None,
                    action=None,
                    state_source_t_ns=None,
                    action_source_t_ns=None,
                    valid=False,
                    invalid_reason=reasons,
                )
            )
    outcome.max_abs_sync_error_ns = int(max(errors)) if errors else 0
    outcome.p95_abs_sync_error_ns = int(compute_p95(errors))
    outcome.diagnostics = _build_diagnostics(
        source=source, outcome=outcome, missing=missing, tolerance_ns=row_tolerance_ns
    )
    return outcome


def align_source(
    source: CanonicalV2Source,
    *,
    expected_state_dim: int | None = None,
    expected_action_dim: int | None = None,
    interval_ns: int = ALIGNED_INTERVAL_NS,
    image_tolerance_ns: int = IMAGE_TOLERANCE_NS,
    state_tolerance_ns: int = STATE_TOLERANCE_NS,
    action_tolerance_ns: int = ACTION_ZOH_TOLERANCE_NS,
) -> AlignmentOutcome:
    """Dispatch to the alignment mode declared by the adapter."""
    if source.alignment_mode == "prealigned":
        return align_prealigned(
            source,
            interval_ns=interval_ns,
            row_tolerance_ns=max(image_tolerance_ns, state_tolerance_ns),
            expected_state_dim=expected_state_dim,
            expected_action_dim=expected_action_dim,
        )
    if source.alignment_mode == "native":
        return align_native(
            source,
            interval_ns=interval_ns,
            image_tolerance_ns=image_tolerance_ns,
            state_tolerance_ns=state_tolerance_ns,
            action_tolerance_ns=action_tolerance_ns,
            expected_state_dim=expected_state_dim,
            expected_action_dim=expected_action_dim,
        )
    raise ValueError(f"unsupported alignment_mode {source.alignment_mode!r}")


def _build_diagnostics(
    *,
    source: CanonicalV2Source,
    outcome: AlignmentOutcome,
    missing: dict[str, list[int]],
    tolerance_ns: int,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for name, times in _stream_times(source):
        violations = [
            index for index in range(1, len(times)) if times[index] <= times[index - 1]
        ]
        if violations:
            diagnostics.append(
                {
                    "code": DIAG_TIMESTAMP_NON_MONOTONIC,
                    "severity": "ERROR",
                    "message": f"{name} source timestamps are not strictly increasing",
                    "details": {
                        "stream": name,
                        "violation_indexes": violations[:20],
                        "violation_count": len(violations),
                    },
                    "structural": True,
                }
            )
    if missing.get(DIAG_SYNC_MISSING_FRAME):
        diagnostics.append(
            {
                "code": DIAG_SYNC_MISSING_FRAME,
                "severity": "ERROR",
                "message": (
                    f"{len(missing[DIAG_SYNC_MISSING_FRAME])} aligned grid points have no camera "
                    f"frame within {tolerance_ns // 1_000_000} ms"
                ),
                "details": {"grid_indexes": missing[DIAG_SYNC_MISSING_FRAME][:200]},
                "structural": True,
            }
        )
    if missing.get(DIAG_SYNC_MISSING_STATE):
        diagnostics.append(
            {
                "code": DIAG_SYNC_MISSING_STATE,
                "severity": "ERROR",
                "message": (
                    f"{len(missing[DIAG_SYNC_MISSING_STATE])} aligned grid points have no valid "
                    "robot-state sample"
                ),
                "details": {"grid_indexes": missing[DIAG_SYNC_MISSING_STATE][:200]},
                "structural": True,
            }
        )
    if missing.get(DIAG_SYNC_STALE_ACTION):
        diagnostics.append(
            {
                "code": DIAG_SYNC_STALE_ACTION,
                "severity": "ERROR",
                "message": (
                    f"{len(missing[DIAG_SYNC_STALE_ACTION])} aligned grid points have no action at or "
                    "before the sample time within tolerance (future actions are never used)"
                ),
                "details": {"grid_indexes": missing[DIAG_SYNC_STALE_ACTION][:200]},
                "structural": True,
            }
        )
    dimension_reasons = {
        reason: count
        for reason, count in outcome.missing_reason_counts.items()
        if reason.startswith("dimension_mismatch:")
    }
    if dimension_reasons:
        diagnostics.append(
            {
                "code": DIAG_DIMENSION_MISMATCH,
                "severity": "ERROR",
                "message": "source sample dimensions do not match the resolved robot profile",
                "details": {"reasons": dimension_reasons},
                "structural": True,
            }
        )
    nonfinite_reasons = {
        reason: count
        for reason, count in outcome.missing_reason_counts.items()
        if reason.startswith("nonfinite:")
    }
    if nonfinite_reasons:
        diagnostics.append(
            {
                "code": DIAG_NONFINITE_VALUE,
                "severity": "ERROR",
                "message": "source state/action contains NaN or infinite values",
                "details": {"reasons": nonfinite_reasons},
                "structural": True,
            }
        )
    return diagnostics


def _stream_times(source: CanonicalV2Source) -> list[tuple[str, list[int]]]:
    streams: list[tuple[str, list[int]]] = [
        (f"camera:{camera.name}", camera.times_ns) for camera in source.cameras
    ]
    if source.state is not None:
        streams.append(("state", source.state.times_ns))
    if source.action is not None:
        streams.append(("action", source.action.times_ns))
    return streams


def camera_names_of(source: CanonicalV2Source) -> list[str]:
    names = [stream.name for stream in source.cameras]
    for name in names:
        if name not in CAMERA_NAMES:
            raise ValueError(f"unsupported camera stream name {name!r}; expected one of {CAMERA_NAMES}")
    return names


__all__ = [
    "AlignmentOutcome",
    "DIAG_DIMENSION_MISMATCH",
    "DIAG_NONFINITE_VALUE",
    "DIAG_SYNC_MISSING_FRAME",
    "DIAG_SYNC_MISSING_STATE",
    "DIAG_SYNC_STALE_ACTION",
    "DIAG_TIMESTAMP_NON_MONOTONIC",
    "REASON_MISSING_ROW",
    "align_native",
    "align_prealigned",
    "align_source",
    "camera_names_of",
]
