"""Unit tests: bounded PointCloud2 parsing, summaries and endianness (DO2-25)."""

from __future__ import annotations

import math
import struct

import pyarrow.parquet as pq
import pytest

from embodied_dataops.processing.pointcloud import (
    FLOAT32_DATATYPE,
    MAX_POINTS,
    MAX_SAMPLE_POINTS,
    POINTCLOUD_SUMMARY_SCHEMA,
    PointCloud2Message,
    PointCloudField,
    parse_pointcloud2,
    summary_table,
    write_summary_parquet,
)

pytestmark = pytest.mark.unit


def make_message(
    points: list[tuple[float, float, float]],
    *,
    width: int | None = None,
    height: int | None = None,
    fields: list[PointCloudField] | None = None,
    point_step: int | None = None,
    row_step: int | None = None,
    big_endian: bool = False,
    extra_padding: int = 0,
    data_override: bytes | None = None,
    source_t_ns: int = 1_000,
) -> PointCloud2Message:
    fields = fields or [
        PointCloudField("x", 0, FLOAT32_DATATYPE),
        PointCloudField("y", 4, FLOAT32_DATATYPE),
        PointCloudField("z", 8, FLOAT32_DATATYPE),
    ]
    point_step = point_step or 12
    width = width or len(points)
    height = height or 1
    assert width * height == len(points)
    row_step = row_step or point_step * width + extra_padding
    prefix = ">" if big_endian else "<"
    by_name = {field.name: field for field in fields}
    x_offset = by_name["x"].offset
    y_offset = by_name["y"].offset
    z_offset = by_name["z"].offset
    data = bytearray()
    for row in range(height):
        body = bytearray(point_step * width)
        for column in range(width):
            x, y, z = points[row * width + column]
            base = column * point_step
            struct.pack_into(prefix + "f", body, base + x_offset, x)
            struct.pack_into(prefix + "f", body, base + y_offset, y)
            struct.pack_into(prefix + "f", body, base + z_offset, z)
        data.extend(body)
        data.extend(b"\x00" * (row_step - point_step * width))
    return PointCloud2Message(
        data=data_override if data_override is not None else bytes(data),
        width=width,
        height=height,
        point_step=point_step,
        row_step=row_step,
        fields=fields,
        is_bigendian=big_endian,
        source_t_ns=source_t_ns,
        source_asset_sha256="e" * 64,
    )


def test_little_endian_organized_cloud_summary() -> None:
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (-4.0, 5.5, -6.0),
        (7.0, -8.0, 9.0),
        (1.5, 1.5, 1.5),
        (2.0, 2.0, 2.0),
        (-1.0, -1.0, -1.0),
        (0.5, -0.5, 0.25),
    ]
    message = make_message(points, width=4, height=2, source_t_ns=123)
    summary = parse_pointcloud2(message)
    assert summary.valid is True
    assert summary.error_code is None
    assert summary.point_count == 8
    assert summary.sampled_point_count == 8
    assert summary.endianness == "little"
    assert summary.source_t_ns == 123
    assert summary.min_x == -4.0
    assert summary.max_x == 7.0
    assert summary.min_y == -8.0
    assert summary.max_y == 5.5
    assert summary.min_z == -6.0
    assert summary.max_z == 9.0
    assert summary.nonfinite_sample_count == 0
    assert summary.findings == []
    assert summary.row()["source_asset_sha256"] == "e" * 64


def test_big_endian_cloud_has_same_bounds_and_no_false_nonfinite() -> None:
    points = [(1.0, -2.0, 3.0), (-4.0, 5.0, -6.0), (7.0, 8.0, 9.0)]
    little = parse_pointcloud2(make_message(points, big_endian=False))
    big = parse_pointcloud2(make_message(points, big_endian=True))
    assert big.endianness == "big"
    assert (big.min_x, big.max_x, big.min_y, big.max_y, big.min_z, big.max_z) == (
        little.min_x,
        little.max_x,
        little.min_y,
        little.max_y,
        little.min_z,
        little.max_z,
    )
    assert big.nonfinite_sample_count == 0
    assert big.valid is True


def test_nonzero_field_offset_and_point_step_padding() -> None:
    # Common ROS layout: 4-byte unspecified header, xyz at 4/8/12, intensity at 16,
    # point_step 20 (4 bytes of trailing padding).
    fields = [
        PointCloudField("x", 4, FLOAT32_DATATYPE),
        PointCloudField("y", 8, FLOAT32_DATATYPE),
        PointCloudField("z", 12, FLOAT32_DATATYPE),
        PointCloudField("intensity", 16, FLOAT32_DATATYPE),
    ]
    message = make_message(
        [(0.5, 1.5, 2.5), (-0.5, -1.5, -2.5)],
        fields=fields,
        point_step=20,
        extra_padding=4,
    )
    summary = parse_pointcloud2(message)
    assert summary.valid is True
    assert summary.point_count == 2
    assert summary.min_x == -0.5
    assert summary.max_z == 2.5


def test_deterministic_sampling_caps_at_ten_thousand_points() -> None:
    point_count = 25_000
    data = struct.pack("<3f", 1.0, 2.0, 3.0) * point_count
    message = PointCloud2Message(
        data=data,
        width=point_count,
        height=1,
        point_step=12,
        row_step=12 * point_count,
        fields=[
            PointCloudField("x", 0, FLOAT32_DATATYPE),
            PointCloudField("y", 4, FLOAT32_DATATYPE),
            PointCloudField("z", 8, FLOAT32_DATATYPE),
        ],
    )
    first = parse_pointcloud2(message)
    second = parse_pointcloud2(message)
    stride = math.ceil(point_count / MAX_SAMPLE_POINTS)
    assert first.point_count == point_count
    assert first.sampled_point_count == math.ceil(point_count / stride)
    assert first.sampled_point_count <= MAX_SAMPLE_POINTS
    assert first.sampled_point_count == second.sampled_point_count
    assert (first.min_x, first.max_x) == (second.min_x, second.max_x) == (1.0, 1.0)


def test_one_million_points_is_accepted_within_bound() -> None:
    point_count = MAX_POINTS
    data = struct.pack("<3f", 0.0, 0.0, 0.0) * point_count
    message = PointCloud2Message(
        data=data,
        width=point_count,
        height=1,
        point_step=12,
        row_step=12 * point_count,
        fields=[
            PointCloudField("x", 0, FLOAT32_DATATYPE),
            PointCloudField("y", 4, FLOAT32_DATATYPE),
            PointCloudField("z", 8, FLOAT32_DATATYPE),
        ],
    )
    summary = parse_pointcloud2(message)
    assert summary.valid is True
    assert summary.point_count == point_count
    assert summary.sampled_point_count == MAX_SAMPLE_POINTS
    assert summary.error_code is None


def test_nonfinite_ratio_above_threshold_is_warned_not_silently_dropped() -> None:
    points = [
        (float("nan"), 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
        (7.0, 0.0, 0.0),
    ]
    summary = parse_pointcloud2(make_message(points))
    assert summary.valid is True
    assert summary.nonfinite_sample_count == 1
    assert summary.min_x == 1.0  # NaN never enters the bounding box
    assert len(summary.findings) == 1
    finding = summary.findings[0]
    assert finding["rule_code"] == "POINTCLOUD_NONFINITE_HIGH"
    assert finding["severity"] == "WARN"
    assert finding["observed"]["nonfinite_ratio"] == pytest.approx(1 / 8)
    assert finding["threshold"]["nonfinite_ratio_gt"] == 0.10


def test_nonfinite_ratio_equal_to_threshold_does_not_warn() -> None:
    points = [(float("nan"), 0.0, 0.0)] + [(float(i), 0.0, 0.0) for i in range(9)]
    summary = parse_pointcloud2(make_message(points))
    assert summary.nonfinite_sample_count == 1
    assert summary.findings == []


def test_summary_parquet_schema_and_values_roundtrip(tmp_path) -> None:
    summary = parse_pointcloud2(make_message([(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)]))
    path = tmp_path / "summary.parquet"
    info = write_summary_parquet(path, [summary])
    assert info["rows"] == 1
    table = pq.read_table(path)
    assert table.schema.equals(POINTCLOUD_SUMMARY_SCHEMA)
    row = table.to_pylist()[0]
    assert row["point_count"] == 2
    assert row["sampled_point_count"] == 2
    assert row["min_x"] == -1.0
    assert row["max_z"] == 3.0
    assert row["endianness"] == "little"
    assert row["source_asset_sha256"] == "e" * 64
    assert set(row) == {
        "source_t_ns",
        "point_count",
        "sampled_point_count",
        "nonfinite_sample_count",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "min_z",
        "max_z",
        "endianness",
        "source_asset_sha256",
    }


def test_summary_table_keeps_full_point_count_separate_from_sample() -> None:
    point_count = 20
    data = struct.pack("<3f", 0.0, 0.0, 0.0) * point_count
    message = PointCloud2Message(
        data=data,
        width=point_count,
        height=1,
        point_step=12,
        row_step=12 * point_count,
        fields=[
            PointCloudField("x", 0, FLOAT32_DATATYPE),
            PointCloudField("y", 4, FLOAT32_DATATYPE),
            PointCloudField("z", 8, FLOAT32_DATATYPE),
        ],
    )
    summary = parse_pointcloud2(message)
    table = summary_table([summary])
    row = table.to_pylist()[0]
    assert row["point_count"] == point_count
    assert row["sampled_point_count"] == point_count  # below the cap: all points sampled
