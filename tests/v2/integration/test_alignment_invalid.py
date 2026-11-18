"""DO2-07: misaligned / missing native samples must never be zero-filled.

The invalid fixture drops front frames, removes a joint-state window and starts
the action stream late.  The aligned view must keep the grid, mark the affected
rows ``valid=false`` with explicit ``invalid_reason`` entries and leave the
state/action arrays **null** (a zero vector would be a fabricated observation).
"""

from __future__ import annotations

import pytest
from _canonical_helpers import native_source

from embodied_dataops.canonical.v2 import build_canonical_v2, validate_aligned_table
from embodied_dataops.canonical.v2.parquet import read_parquet_bytes

pytestmark = pytest.mark.integration


def _invalid_bundle():
    source = native_source(
        drop_front_window=(900_000_000, 1_200_000_000),
        drop_state_window=(400_000_000, 600_000_000),
        action_start_ns=200_000_000,
    )
    return build_canonical_v2(source, canonical_version=1)


def test_invalid_rows_are_explicit_and_not_dropped():
    bundle = _invalid_bundle()
    table = read_parquet_bytes(bundle.artifacts["aligned_samples"])
    assert validate_aligned_table(table) == []  # nulls are legal, zero-fill is not
    valid = table.column("valid").to_pylist()
    reasons = table.column("invalid_reason").to_pylist()
    assert any(not value for value in valid)
    assert any(reasons)
    flat = {reason for row in reasons for reason in row}
    assert "missing_frame:front" in flat
    assert "missing_state" in flat


def test_invalid_rows_have_null_arrays():
    bundle = _invalid_bundle()
    table = read_parquet_bytes(bundle.artifacts["aligned_samples"])
    state = table.column("state").to_pylist()
    action = table.column("action").to_pylist()
    valid = table.column("valid").to_pylist()
    for index, is_valid in enumerate(valid):
        if is_valid:
            continue
        assert state[index] is None
        assert action[index] is None
    for index, is_valid in enumerate(valid):
        if is_valid:
            assert state[index] is not None and action[index] is not None


def test_episode_with_invalid_rows_is_not_publishable():
    bundle = _invalid_bundle()
    assert bundle.publishable is False
    assert "aligned_view_has_invalid_samples" in bundle.blocking_reasons
    assert bundle.metadata.aligned_view.valid_sample_count < bundle.metadata.aligned_view.sample_count
    # the grid is not silently truncated to the valid prefix
    assert bundle.metadata.aligned_view.sample_count == 20


def test_structural_diagnostics_reference_the_alignment_problem():
    bundle = _invalid_bundle()
    codes = {item["code"] for item in bundle.metadata.diagnostics}
    assert {"SYNC_MISSING_FRAME", "SYNC_MISSING_STATE"} <= codes
    assert "structural_diagnostics_present" in bundle.blocking_reasons
