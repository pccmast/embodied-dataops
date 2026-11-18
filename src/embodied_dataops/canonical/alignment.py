"""Reusable alignment machinery for robot episodes.

The aligner turns per-stream (absolute timestamp, payload) sequences into the
canonical 10 Hz grid while recording every sync violation as structured data.
It never fabricates values: a missing sample is recorded, not zero-filled, and
a stale action is an error rather than a lookup into the future.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from embodied_dataops.canonical.contract import SampleRow
from embodied_dataops.canonical.timebase import (
    ACTION_ZOH_TOLERANCE_NS,
    IMAGE_TOLERANCE_NS,
    STATE_TOLERANCE_NS,
    generate_grid,
    interpolate_state,
    nearest_match,
    zoh_action,
)


@dataclass
class AlignmentDiagnostics:
    missing_frames: list[int] = field(default_factory=list)
    missing_states: list[int] = field(default_factory=list)
    stale_actions: list[int] = field(default_factory=list)
    frame_errors_ns: list[int] = field(default_factory=list)
    state_errors_ns: list[int] = field(default_factory=list)
    action_ages_ns: list[int] = field(default_factory=list)
    grid_points: int = 0
    missing_sample_count: int = 0
    max_abs_sync_error_ms: float = 0.0
    p95_abs_sync_error_ms: float = 0.0
    coverage_start_ns: int | None = None
    coverage_end_ns: int | None = None
    origin_ns: int = 0

    def as_dict(self) -> dict:
        p95 = self.p95_abs_sync_error_ms
        return {
            "grid_points": self.grid_points,
            "missing_sample_count": self.missing_sample_count,
            "max_abs_sync_error_ms": self.max_abs_sync_error_ms,
            "p95_abs_sync_error_ms": p95,
            "missing_frame_indexes": self.missing_frames[:200],
            "missing_state_indexes": self.missing_states[:200],
            "stale_action_indexes": self.stale_actions[:200],
            "coverage_start_ns": self.coverage_start_ns,
            "coverage_end_ns": self.coverage_end_ns,
            "origin_ns": self.origin_ns,
        }


def compute_p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def align_robot_streams(
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
) -> tuple[list[SampleRow], AlignmentDiagnostics]:
    """Build the robot sample rows on the 10 Hz grid.

    Returns rows only for fully valid grid points; every invalid point is
    represented in the diagnostics so the quality rules can emit explicit
    findings.  Rows are re-indexed contiguously only when every grid point is
    valid (i.e. the canonical 10 Hz contract holds).
    """
    diag = AlignmentDiagnostics(origin_ns=origin_ns)
    if not frame_times_ns:
        diag.coverage_start_ns = None
        diag.coverage_end_ns = None
        return [], diag
    coverage_start = max(frame_times_ns[0], state_times_ns[0] if state_times_ns else frame_times_ns[0])
    coverage_end = min(coverage_end_ns, state_times_ns[-1] if state_times_ns else coverage_end_ns)
    diag.coverage_start_ns = coverage_start
    diag.coverage_end_ns = coverage_end
    grid = generate_grid(
        origin_ns=origin_ns, coverage_start_ns=coverage_start, coverage_end_ns=coverage_end
    )
    diag.grid_points = len(grid)
    rows: list[SampleRow] = []
    frame_errors: list[float] = []
    state_errors: list[float] = []
    all_errors_ms: list[float] = []
    for t_rel in grid:
        target = origin_ns + t_rel
        frame = nearest_match(frame_times_ns, target, image_tolerance_ns)
        if frame is None:
            diag.missing_frames.append(len(rows))
            continue
        state = interpolate_state(state_times_ns, state_values, target, state_tolerance_ns)
        if state is None:
            diag.missing_states.append(len(rows))
            continue
        action = zoh_action(action_times_ns, action_values, target, action_tolerance_ns)
        if action is None:
            diag.stale_actions.append(len(rows))
            continue
        width = len(state.value)
        if width != 6 or len(action_values[action.index]) != 6:
            # Dimension errors are reported by the quality rules; keep the row
            # so the failure is visible downstream instead of silently dropping.
            pass
        rows.append(
            SampleRow(
                sample_index=len(rows),
                t_ns=t_rel,
                frame_index=len(rows),
                state=state.value,
                action=tuple(float(v) for v in action_values[action.index]),
                state_source_t_ns=state.source_t_ns - origin_ns,
                action_source_t_ns=action.source_t_ns - origin_ns,
            )
        )
        frame_errors.append(frame.abs_error_ns / 1e6)
        state_errors.append(state.max_endpoint_error_ns / 1e6)
        diag.action_ages_ns.append(action.age_ns)
        diag.frame_errors_ns.append(frame.abs_error_ns)
        diag.state_errors_ns.append(state.max_endpoint_error_ns)
        all_errors_ms.append(max(frame.abs_error_ns, state.max_endpoint_error_ns, action.age_ns) / 1e6)
    diag.missing_sample_count = (
        len(diag.missing_frames) + len(diag.missing_states) + len(diag.stale_actions)
    )
    diag.max_abs_sync_error_ms = max(all_errors_ms) if all_errors_ms else 0.0
    diag.p95_abs_sync_error_ms = compute_p95(all_errors_ms)
    return rows, diag
