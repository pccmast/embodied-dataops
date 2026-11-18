"""DO2-06: native multi-camera / multi-rate indices and the aligned 10 Hz view.

A native source (front 30 Hz, wrist 15 Hz, state 100 Hz, action 50 Hz) must keep
the native sample times in the native indices and derive the 10 Hz training grid
with per-camera nearest-frame selection -- without fabricating native rates.
"""

from __future__ import annotations

import pytest
from _canonical_helpers import native_source

from embodied_dataops.canonical.v2 import build_canonical_v2, validate_aligned_table
from embodied_dataops.canonical.v2.parquet import read_parquet_bytes

pytestmark = pytest.mark.integration


def _bundle():
    return build_canonical_v2(native_source(), canonical_version=1)


def test_native_rates_are_preserved_per_stream():
    bundle = _bundle()
    rates = {modality.name: modality.native_rate_hz for modality in bundle.metadata.modalities}
    assert rates["front"] == pytest.approx(30.0, rel=0.01)
    assert rates["wrist"] == pytest.approx(15.0, rel=0.01)
    assert rates["state"] == pytest.approx(100.0, rel=0.01)
    assert rates["action"] == pytest.approx(50.0, rel=0.01)
    counts = {modality.name: modality.native_sample_count for modality in bundle.metadata.modalities}
    assert counts["front"] == 61
    assert counts["wrist"] == 31
    assert counts["state"] == 201
    assert counts["action"] == 101


def test_native_camera_index_keeps_source_times():
    bundle = _bundle()
    table = read_parquet_bytes(bundle.artifacts["native_camera:front"])
    assert table.column_names == ["source_index", "source_t_ns", "episode_t_ns", "valid", "source_asset_id"]
    source_times = table.column("source_t_ns").to_pylist()
    assert source_times[1] - source_times[0] == pytest.approx(33_333_333, abs=1)
    assert table.column("episode_t_ns").to_pylist()[0] == 0


def test_aligned_view_is_10hz_with_both_cameras():
    bundle = _bundle()
    table = read_parquet_bytes(bundle.artifacts["aligned_samples"])
    assert validate_aligned_table(table) == []
    assert table.column("t_ns").to_pylist() == [index * 100_000_000 for index in range(20)]
    frame_indexes = table.column("frame_indexes").to_pylist()
    assert all(row["front"] is not None for row in frame_indexes)
    assert all(row["wrist"] is not None for row in frame_indexes)
    assert bundle.metadata.aligned_view.valid_sample_count == 20
    assert bundle.metadata.aligned_view.camera_names == ["front", "wrist"]


def test_float32_projection_error_is_recorded():
    bundle = _bundle()
    for stream in ("state", "action"):
        detail = bundle.precision[stream]
        assert detail["max_abs_error"] <= detail["max_abs_error_threshold"]
        assert detail["blocked"] is False
    assert bundle.publishable is True


def test_single_camera_source_has_no_wrist_modality():
    bundle = build_canonical_v2(native_source(with_wrist=False), canonical_version=1)
    names = [modality.name for modality in bundle.metadata.modalities]
    assert names == ["front", "state", "action"]
    assert bundle.metadata.camera_names == ["front"]
