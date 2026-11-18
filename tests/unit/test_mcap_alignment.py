"""Alignment semantics for the MCAP profile (SPEC 5.4, AT-04).

Pinned behaviours:

* image nearest match: ``<= 50 ms`` accepted, one nanosecond more rejected, ties
  choose the earlier frame;
* state interpolation needs **both** bracketing samples within ``<= 50 ms``;
* action zero-order hold only looks strictly to the past (``<= 100 ms``) -- a
  future action can never explain an earlier sample;
* the parser's grid/rows are identical to the shared
  ``canonical.alignment.align_robot_streams`` behaviour while keeping exact
  grid-point diagnostics.
"""

from __future__ import annotations

import pytest

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.canonical.alignment import align_robot_streams
from embodied_dataops.canonical.timebase import (
    ACTION_ZOH_TOLERANCE_NS,
    IMAGE_TOLERANCE_NS,
    STATE_TOLERANCE_NS,
    generate_grid,
    interpolate_state,
    nearest_match,
    zoh_action,
)
from embodied_dataops.imports.mcap_profile_v1 import align_mcap_streams

ORIGIN_NS = 1_700_000_000_000_000_000


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def test_image_nearest_match_50ms_boundary() -> None:
    times = [ORIGIN_NS, ORIGIN_NS + 200_000_000]
    match = nearest_match(times, ORIGIN_NS + 50_000_000, IMAGE_TOLERANCE_NS)
    assert match is not None
    assert match.index == 0
    assert match.abs_error_ns == 50_000_000
    assert nearest_match(times, ORIGIN_NS + 50_000_001, IMAGE_TOLERANCE_NS) is None


def test_image_nearest_match_tie_prefers_earlier_frame() -> None:
    times = [ORIGIN_NS, ORIGIN_NS + 100_000_000]
    match = nearest_match(times, ORIGIN_NS + 50_000_000, IMAGE_TOLERANCE_NS)
    assert match is not None
    assert match.index == 0


def test_state_interpolation_requires_both_endpoints_within_50ms() -> None:
    times = [ORIGIN_NS, ORIGIN_NS + 100_000_000]
    values = [(0.0, 1.0), (10.0, 11.0)]
    exact = interpolate_state(times, values, ORIGIN_NS + 50_000_000, STATE_TOLERANCE_NS)
    assert exact is not None
    assert exact.value == pytest.approx((5.0, 6.0))
    assert exact.max_endpoint_error_ns == 50_000_000
    assert interpolate_state(times, values, ORIGIN_NS + 50_000_001, STATE_TOLERANCE_NS) is None


def test_state_interpolation_needs_upper_bracket() -> None:
    times = [ORIGIN_NS, ORIGIN_NS + 120_000_000]
    values = [(0.0,), (1.0,)]
    target = ORIGIN_NS + 60_000_000
    # lower endpoint is 60 ms away (> 50 ms) and the upper endpoint 60 ms away
    assert interpolate_state(times, values, target, STATE_TOLERANCE_NS) is None
    assert interpolate_state(times, values, ORIGIN_NS, STATE_TOLERANCE_NS) is not None
    # no sample at/after the target at all
    assert interpolate_state(times, values, ORIGIN_NS + 200_000_000, STATE_TOLERANCE_NS) is None


def test_state_interpolation_uses_exact_sample_when_present() -> None:
    times = [ORIGIN_NS, ORIGIN_NS + 100_000_000]
    values = [(1.0,), (2.0,)]
    match = interpolate_state(times, values, ORIGIN_NS + 100_000_000, STATE_TOLERANCE_NS)
    assert match is not None
    assert match.value == (2.0,)
    assert match.max_endpoint_error_ns == 0


def test_action_zoh_never_uses_the_future() -> None:
    future_only = [ORIGIN_NS + 100_000_000]
    assert zoh_action(future_only, [(1.0,)], ORIGIN_NS, ACTION_ZOH_TOLERANCE_NS) is None

    times = [ORIGIN_NS, ORIGIN_NS + 100_000_000]
    values = [(0.0,), (10.0,)]
    match = zoh_action(times, values, ORIGIN_NS + 50_000_000, ACTION_ZOH_TOLERANCE_NS)
    assert match is not None
    assert match.index == 0  # strictly past, not the 100 ms future command
    assert match.age_ns == 50_000_000


def test_action_zoh_100ms_boundary() -> None:
    times = [ORIGIN_NS]
    values = [(3.0,)]
    assert zoh_action(times, values, ORIGIN_NS + 100_000_000, ACTION_ZOH_TOLERANCE_NS) is not None
    assert zoh_action(times, values, ORIGIN_NS + 100_000_001, ACTION_ZOH_TOLERANCE_NS) is None
    assert zoh_action(times, values, ORIGIN_NS - 1, ACTION_ZOH_TOLERANCE_NS) is None


def test_grid_starts_at_first_camera_frame_and_ends_at_coverage_end() -> None:
    grid = generate_grid(
        origin_ns=ORIGIN_NS,
        coverage_start_ns=ORIGIN_NS,
        coverage_end_ns=ORIGIN_NS + 1_000_000_000,
    )
    assert grid == [index * SAMPLE_INTERVAL_NS for index in range(11)]


# ---------------------------------------------------------------------------
# parser alignment
# ---------------------------------------------------------------------------


def _aligned_episode():
    frame_times = [ORIGIN_NS + index * 33_333_333 for index in range(301)]
    state_times = [ORIGIN_NS - 200_000_000 + index * 10_000_000 for index in range(1020)]
    action_times = [ORIGIN_NS - 300_000_000 + index * 20_000_000 for index in range(515)]
    state_values = [[0.001 * index] * 6 for index in range(len(state_times))]
    action_values = [[0.002 * index] * 6 for index in range(len(action_times))]
    return align_mcap_streams(
        origin_ns=ORIGIN_NS,
        frame_times_ns=frame_times,
        state_times_ns=state_times,
        state_values=state_values,
        action_times_ns=action_times,
        action_values=action_values,
        coverage_end_ns=frame_times[-1],
    )


def test_parser_alignment_builds_10hz_grid_without_missing_points() -> None:
    result = _aligned_episode()
    assert len(result.grid_rel_ns) == 100
    assert result.grid_rel_ns[0] == 0
    assert result.grid_rel_ns[-1] == 9_900_000_000
    assert result.missing_grid_indexes == []
    assert result.valid_sample_count == 100
    assert result.missing_sample_count == 0
    assert result.coverage_start_ns == ORIGIN_NS


def test_parser_alignment_action_is_never_in_the_future() -> None:
    result = _aligned_episode()
    for row in result.rows:
        assert row.action_source_t_ns <= row.t_ns
        assert row.t_ns - row.action_source_t_ns <= ACTION_ZOH_TOLERANCE_NS


def test_parser_alignment_matches_shared_canonical_alignment() -> None:
    frame_times = [ORIGIN_NS + index * 33_333_333 for index in range(301)]
    state_times = [ORIGIN_NS - 200_000_000 + index * 10_000_000 for index in range(1020)]
    action_times = [ORIGIN_NS - 300_000_000 + index * 20_000_000 for index in range(515)]
    state_values = [[0.001 * index] * 6 for index in range(len(state_times))]
    action_values = [[0.002 * index] * 6 for index in range(len(action_times))]
    parser_rows = align_mcap_streams(
        origin_ns=ORIGIN_NS,
        frame_times_ns=frame_times,
        state_times_ns=state_times,
        state_values=state_values,
        action_times_ns=action_times,
        action_values=action_values,
        coverage_end_ns=frame_times[-1],
    ).rows
    shared_rows, _diag = align_robot_streams(
        origin_ns=ORIGIN_NS,
        frame_times_ns=frame_times,
        state_times_ns=state_times,
        state_values=state_values,
        action_times_ns=action_times,
        action_values=action_values,
        coverage_end_ns=frame_times[-1],
    )
    assert len(parser_rows) == len(shared_rows) == 100
    for parser_row, shared_row in zip(parser_rows, shared_rows, strict=True):
        assert parser_row.t_ns == shared_row.t_ns
        assert parser_row.frame_index == shared_row.frame_index
        assert parser_row.state == shared_row.state
        assert parser_row.action == shared_row.action
        assert parser_row.action_source_t_ns <= parser_row.t_ns
        assert shared_row.action_source_t_ns <= shared_row.t_ns


def test_parser_alignment_reports_missing_state_instead_of_shrinking_the_grid() -> None:
    frame_times = [ORIGIN_NS + index * 100_000_000 for index in range(101)]
    state_times = [ORIGIN_NS + 10_000_000 + index * 100_000_000 for index in range(50)]
    action_times = [ORIGIN_NS - 10_000_000 + index * 50_000_000 for index in range(60)]
    result = align_mcap_streams(
        origin_ns=ORIGIN_NS,
        frame_times_ns=frame_times,
        state_times_ns=state_times,
        state_values=[[0.0] * 6] * len(state_times),
        action_times_ns=action_times,
        action_values=[[0.0] * 6] * len(action_times),
        coverage_end_ns=frame_times[-1],
    )
    # grid still starts at t=0; the first point has no valid state bracket at all
    assert result.grid_rel_ns[0] == 0
    assert 0 in result.missing_state_indexes
    assert len(result.rows) == result.valid_sample_count
    codes = [diagnostic["code"] for diagnostic in result.diagnostics()]
    assert "SYNC_MISSING_STATE" in codes


def test_parser_alignment_records_stale_action_instead_of_using_future_value() -> None:
    frame_times = [ORIGIN_NS + index * 100_000_000 for index in range(11)]
    state_times = [ORIGIN_NS - 10_000_000 + index * 20_000_000 for index in range(6)]
    action_times = [ORIGIN_NS + 200_000_000]  # only a future action exists
    result = align_mcap_streams(
        origin_ns=ORIGIN_NS,
        frame_times_ns=frame_times,
        state_times_ns=state_times,
        state_values=[[0.0] * 6] * len(state_times),
        action_times_ns=action_times,
        action_values=[[9.0] * 6],
        coverage_end_ns=frame_times[-1],
    )
    assert result.grid_rel_ns[0] == 0
    assert result.stale_action_indexes
    assert result.rows == []
    assert "SYNC_STALE_ACTION" in [diagnostic["code"] for diagnostic in result.diagnostics()]
