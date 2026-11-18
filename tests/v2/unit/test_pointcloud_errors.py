"""Unit tests: corrupt/unsupported/oversize PointCloud2 inputs (DO2-26)."""

from __future__ import annotations

import struct

import pytest

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.processing.pointcloud import (
    FLOAT32_DATATYPE,
    MAX_POINTS,
    PointCloud2Message,
    PointCloudField,
    parse_pointcloud2,
)

pytestmark = pytest.mark.unit

XYZ_FIELDS = [
    PointCloudField("x", 0, FLOAT32_DATATYPE),
    PointCloudField("y", 4, FLOAT32_DATATYPE),
    PointCloudField("z", 8, FLOAT32_DATATYPE),
]


def message(
    *,
    width: int,
    height: int = 1,
    point_step: int = 12,
    row_step: int | None = None,
    fields=None,
    data: bytes | None = None,
    big_endian: bool = False,
) -> PointCloud2Message:
    if data is None:
        data = b"".join(
            struct.pack((">" if big_endian else "<") + "3f", 0.0, 0.0, 0.0)
            for _ in range(width * height)
        )
    return PointCloud2Message(
        data=data,
        width=width,
        height=height,
        point_step=point_step,
        row_step=row_step if row_step is not None else point_step * width,
        fields=fields if fields is not None else list(XYZ_FIELDS),
        is_bigendian=big_endian,
        source_t_ns=42,
        source_asset_sha256="f" * 64,
    )


def assert_corrupt(summary, *, reason_part: str | None = None) -> None:
    assert summary.valid is False
    assert summary.error_code == ErrorCode.POINTCLOUD_CORRUPT.value
    assert summary.sampled_point_count == 0
    assert summary.findings
    finding = summary.findings[0]
    assert finding["rule_code"] == "POINTCLOUD_CORRUPT"
    assert finding["severity"] == "ERROR"
    assert finding["source_stream"] == "/points"
    if reason_part is not None:
        assert reason_part in finding["observed"]["reason"]


def test_truncated_data_is_corrupt_without_crashing() -> None:
    summary = parse_pointcloud2(message(width=4, data=b"\x00" * 10))
    assert_corrupt(summary, reason_part="truncated_data")
    assert summary.findings[0]["observed"]["required_bytes"] == 48


def test_missing_xyz_field_is_corrupt() -> None:
    summary = parse_pointcloud2(
        message(width=2, fields=[PointCloudField("x", 0, FLOAT32_DATATYPE)])
    )
    assert_corrupt(summary, reason_part="missing_xyz_fields")


def test_duplicate_field_names_are_corrupt() -> None:
    summary = parse_pointcloud2(
        message(
            width=2,
            fields=[
                PointCloudField("x", 0, FLOAT32_DATATYPE),
                PointCloudField("x", 4, FLOAT32_DATATYPE),
                PointCloudField("y", 8, FLOAT32_DATATYPE),
                PointCloudField("z", 12, FLOAT32_DATATYPE),
            ],
            point_step=16,
        )
    )
    assert_corrupt(summary, reason_part="duplicate_field_name")


def test_overlapping_fields_are_corrupt() -> None:
    summary = parse_pointcloud2(
        message(
            width=2,
            fields=[
                PointCloudField("x", 0, FLOAT32_DATATYPE),
                PointCloudField("y", 2, FLOAT32_DATATYPE),  # overlaps x[0..4)
                PointCloudField("z", 8, FLOAT32_DATATYPE),
            ],
        )
    )
    assert_corrupt(summary, reason_part="overlapping_fields")


def test_field_out_of_point_step_is_corrupt() -> None:
    summary = parse_pointcloud2(
        message(width=2, fields=list(XYZ_FIELDS), point_step=10)
    )
    assert_corrupt(summary, reason_part="field_out_of_bounds")


def test_row_step_smaller_than_point_step_times_width_is_corrupt() -> None:
    summary = parse_pointcloud2(message(width=4, row_step=12))
    assert_corrupt(summary, reason_part="row_step_too_small")


def test_unknown_field_datatype_is_corrupt() -> None:
    fields = [
        PointCloudField("x", 0, FLOAT32_DATATYPE),
        PointCloudField("y", 4, FLOAT32_DATATYPE),
        PointCloudField("z", 8, FLOAT32_DATATYPE),
        PointCloudField("weird", 12, 99),
    ]
    summary = parse_pointcloud2(message(width=2, fields=fields, point_step=16))
    assert_corrupt(summary, reason_part="unknown_datatype")


def test_non_float32_xyz_is_unsupported_layout_422() -> None:
    fields = [
        PointCloudField("x", 0, FLOAT32_DATATYPE),
        PointCloudField("y", 4, FLOAT32_DATATYPE),
        PointCloudField("z", 8, 8),  # FLOAT64
    ]
    with pytest.raises(ApiError) as excinfo:
        parse_pointcloud2(message(width=2, fields=fields, point_step=16))
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_POINTCLOUD_LAYOUT
    assert excinfo.value.status_code == 422
    assert excinfo.value.details["reason"] == "xyz_not_scalar_float32"


def test_multi_channel_extra_field_is_unsupported_layout_422() -> None:
    fields = [
        PointCloudField("x", 0, FLOAT32_DATATYPE),
        PointCloudField("y", 4, FLOAT32_DATATYPE),
        PointCloudField("z", 8, FLOAT32_DATATYPE),
        PointCloudField("rgb", 12, FLOAT32_DATATYPE, count=3),
    ]
    with pytest.raises(ApiError) as excinfo:
        parse_pointcloud2(message(width=2, fields=fields, point_step=24))
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_POINTCLOUD_LAYOUT
    assert excinfo.value.status_code == 422
    assert excinfo.value.details["reason"] == "multi_channel_field"


def test_more_than_one_million_points_is_too_large_without_loading_payload() -> None:
    # 1,000,001 points but an EMPTY payload: the parser must classify by metadata
    # only and return immediately instead of trying to read/allocate the cloud.
    summary = parse_pointcloud2(message(width=MAX_POINTS + 1, data=b""))
    assert summary.valid is False
    assert summary.error_code == ErrorCode.POINTCLOUD_TOO_LARGE.value
    assert summary.point_count == MAX_POINTS + 1
    assert summary.sampled_point_count == 0
    assert summary.findings[0]["rule_code"] == "POINTCLOUD_TOO_LARGE"
    assert summary.findings[0]["severity"] == "ERROR"
    assert summary.findings[0]["observed"]["point_count"] == MAX_POINTS + 1
    assert summary.findings[0]["threshold"]["max_points"] == MAX_POINTS


def test_huge_organized_cloud_is_classified_by_dimensions_only() -> None:
    # width*height over the bound with a tiny payload: no OOM, no full load.
    summary = parse_pointcloud2(message(width=2_000_000, height=1, data=b""))
    assert summary.error_code == ErrorCode.POINTCLOUD_TOO_LARGE.value
    assert summary.point_count == 2_000_000


def test_zero_dimensions_are_corrupt() -> None:
    summary = parse_pointcloud2(message(width=0, data=b""))
    assert_corrupt(summary)


def test_corrupt_summary_still_carries_metadata_for_evidence() -> None:
    summary = parse_pointcloud2(message(width=4, data=b"\x00" * 10))
    payload = summary.as_dict()
    assert payload["valid"] is False
    assert payload["error_code"] == "POINTCLOUD_CORRUPT"
    assert payload["topic"] == "/points"
    assert payload["source_t_ns"] == 42
    assert payload["findings"][0]["affected_time_ranges"] == [{"start_ns": 42, "end_ns": 42}]
