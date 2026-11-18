"""Timebase and alignment primitives (SPEC 5.4 / 5.5).

Rules encoded here:

* every stream declares its clock domain; absolute source times are only ever
  compared inside one declared domain and then converted to episode-relative
  ``t_ns`` by subtracting one episode origin;
* the 10 Hz grid is generated over the *common* valid coverage interval;
* image: nearest frame within the tolerance (ties pick the earlier frame);
* robot state: linear interpolation between a bracketing pair, both endpoints
  within tolerance (a state is a continuous scalar signal);
* robot action: zero-order hold strictly towards the past -- an action message
  from the future is never selected.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from embodied_dataops import SAMPLE_INTERVAL_NS

IMAGE_TOLERANCE_NS = 50_000_000
STATE_TOLERANCE_NS = 50_000_000
ACTION_ZOH_TOLERANCE_NS = 100_000_000
MAX_EPISODE_DURATION_NS = 600 * 1_000_000_000


@dataclass(frozen=True)
class NearestMatch:
    index: int
    source_t_ns: int
    abs_error_ns: int


@dataclass(frozen=True)
class InterpolationMatch:
    low_index: int
    high_index: int
    source_t_ns: int
    max_endpoint_error_ns: int
    value: tuple[float, ...]


@dataclass(frozen=True)
class ZohMatch:
    index: int
    source_t_ns: int
    age_ns: int


def generate_grid(
    *,
    origin_ns: int,
    coverage_start_ns: int,
    coverage_end_ns: int,
    interval_ns: int = SAMPLE_INTERVAL_NS,
    max_duration_ns: int = MAX_EPISODE_DURATION_NS,
) -> list[int]:
    """Episode-relative 10 Hz grid inside the common coverage interval.

    ``origin_ns`` is the absolute time of the first camera frame (t=0).  The
    grid points are ``k * interval_ns`` with ``k`` chosen so that the absolute
    time stays inside ``[coverage_start_ns, coverage_end_ns]``.
    """
    if coverage_end_ns < coverage_start_ns:
        return []
    relative_start = coverage_start_ns - origin_ns
    relative_end = coverage_end_ns - origin_ns
    first_k = max(0, math.ceil(relative_start / interval_ns))
    last_k = math.floor(relative_end / interval_ns)
    last_k = min(last_k, max_duration_ns // interval_ns - 1)
    if last_k < first_k:
        return []
    return [k * interval_ns for k in range(first_k, last_k + 1)]


def nearest_index(times_ns: Sequence[int], target_ns: int) -> int:
    index = bisect.bisect_left(times_ns, target_ns)
    if index <= 0:
        return 0
    if index >= len(times_ns):
        return len(times_ns) - 1
    before = times_ns[index - 1]
    after = times_ns[index]
    if abs(before - target_ns) <= abs(after - target_ns):
        return index - 1
    return index


def nearest_match(
    times_ns: Sequence[int], target_ns: int, tolerance_ns: int
) -> NearestMatch | None:
    if not times_ns:
        return None
    index = nearest_index(times_ns, target_ns)
    error = abs(times_ns[index] - target_ns)
    if error > tolerance_ns:
        return None
    return NearestMatch(index=index, source_t_ns=times_ns[index], abs_error_ns=error)


def interpolate_state(
    times_ns: Sequence[int],
    values: Sequence[Sequence[float]],
    target_ns: int,
    tolerance_ns: int = STATE_TOLERANCE_NS,
) -> InterpolationMatch | None:
    """Linear interpolation with both bracket endpoints within tolerance."""
    if len(times_ns) != len(values) or not times_ns:
        return None
    index = bisect.bisect_left(times_ns, target_ns)
    if index < len(times_ns) and times_ns[index] == target_ns:
        vec = values[index]
        return InterpolationMatch(
            low_index=index,
            high_index=index,
            source_t_ns=target_ns,
            max_endpoint_error_ns=0,
            value=tuple(float(v) for v in vec),
        )
    if index == 0 or index >= len(times_ns):
        return None
    low, high = index - 1, index
    low_error = target_ns - times_ns[low]
    high_error = times_ns[high] - target_ns
    if low_error > tolerance_ns or high_error > tolerance_ns:
        return None
    span = times_ns[high] - times_ns[low]
    if span <= 0:
        return None
    ratio = low_error / span
    low_vec = values[low]
    high_vec = values[high]
    if len(low_vec) != len(high_vec):
        return None
    interpolated = tuple(float(a + (b - a) * ratio) for a, b in zip(low_vec, high_vec, strict=False))
    return InterpolationMatch(
        low_index=low,
        high_index=high,
        source_t_ns=target_ns,
        max_endpoint_error_ns=max(low_error, high_error),
        value=interpolated,
    )


def zoh_action(
    times_ns: Sequence[int],
    values: Sequence[Sequence[float]],
    target_ns: int,
    tolerance_ns: int = ACTION_ZOH_TOLERANCE_NS,
) -> ZohMatch | None:
    """Select the most recent action at or before ``target_ns`` (never the future)."""
    if len(times_ns) != len(values) or not times_ns:
        return None
    index = bisect.bisect_right(times_ns, target_ns) - 1
    if index < 0:
        return None
    age = target_ns - times_ns[index]
    if age > tolerance_ns:
        return None
    return ZohMatch(index=index, source_t_ns=times_ns[index], age_ns=age)


def rows_to_array(values: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray([list(v) for v in values], dtype=np.float32)


def nonfinite_count(rows: Sequence[Sequence[float]] | None) -> int:
    if rows is None:
        return 0
    array = np.asarray([list(row) for row in rows], dtype=np.float64)
    if array.size == 0:
        return 0
    return int(np.count_nonzero(~np.isfinite(array)))


def timestamp_monotonic_violations(times_ns: Sequence[int], limit: int = 20) -> list[int]:
    """Return indexes where the source timestamps are not strictly increasing."""
    violations: list[int] = []
    for i in range(1, len(times_ns)):
        if times_ns[i] <= times_ns[i - 1]:
            violations.append(i)
            if len(violations) >= limit:
                break
    return violations


def frame_gap_windows(
    times_ns: Sequence[int], max_gap_ns: int = 200_000_000, limit: int = 100
) -> list[dict[str, int]]:
    windows: list[dict[str, int]] = []
    for i in range(1, len(times_ns)):
        gap = times_ns[i] - times_ns[i - 1]
        if gap > max_gap_ns:
            windows.append(
                {
                    "prev_source_t_ns": times_ns[i - 1],
                    "next_source_t_ns": times_ns[i],
                    "gap_ns": gap,
                }
            )
            if len(windows) >= limit:
                break
    return windows


def episode_duration_ns_from_grid(sample_count: int, interval_ns: int = SAMPLE_INTERVAL_NS) -> int:
    """Duration is ``last_sample_time + interval`` for a 10 Hz grid."""
    if sample_count <= 0:
        return 0
    return (sample_count - 1) * interval_ns + interval_ns
