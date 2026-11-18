"""Integration test: PointCloud summary.parquet is stored as an immutable Asset."""

from __future__ import annotations

import struct

import pyarrow.parquet as pq
import pytest

from embodied_dataops.db.models import Asset
from embodied_dataops.ids import new_id
from embodied_dataops.processing.pointcloud import (
    FLOAT32_DATATYPE,
    POINTCLOUD_SUMMARY_SCHEMA,
    PointCloud2Message,
    PointCloudField,
    parse_pointcloud2,
    store_pointcloud_summary_asset,
)

pytestmark = pytest.mark.integration


def _message() -> PointCloud2Message:
    fields = [
        PointCloudField("x", 0, FLOAT32_DATATYPE),
        PointCloudField("y", 4, FLOAT32_DATATYPE),
        PointCloudField("z", 8, FLOAT32_DATATYPE),
    ]
    data = b"".join(
        struct.pack("<3f", float(index), -float(index), 0.5 * index) for index in range(4)
    )
    return PointCloud2Message(
        data=data,
        width=4,
        height=1,
        point_step=12,
        row_step=48,
        fields=fields,
        source_t_ns=1_000_000_000,
        source_asset_sha256="a" * 64,
    )


def test_pointcloud_summary_parquet_asset_is_stored_and_bounded(db, storage, tmp_path) -> None:
    episode_id = new_id()
    summaries = [parse_pointcloud2(_message())]
    asset = store_pointcloud_summary_asset(
        db,
        episode_id=episode_id,
        canonical_version=2,
        summaries=summaries,
        storage=storage,
    )
    db.flush()
    assert asset.object_key == (
        f"canonical/{episode_id}/v2/2/native/pointcloud/summary.parquet"
    )
    assert asset.origin == "derived"
    assert asset.byte_size > 0
    assert asset.sha256
    assert asset.meta_json["schema_version"] == "pointcloud-summary/2.0"
    assert asset.meta_json["rows"] == 1
    assert storage.exists(asset.object_key)

    local = tmp_path / "summary.parquet"
    storage.download_to(asset.object_key, local)
    table = pq.read_table(local)
    assert table.schema.equals(POINTCLOUD_SUMMARY_SCHEMA)
    row = table.to_pylist()[0]
    assert row["point_count"] == 4
    assert row["min_x"] == 0.0
    assert row["max_x"] == 3.0
    assert row["source_asset_sha256"] == "a" * 64
    # Only aggregate evidence is persisted; raw point coordinates are never stored here.
    assert "x" not in row or row["x"] is None  # column names are min_x/max_x only


def test_pointcloud_summary_asset_rejects_empty_summary(db, storage) -> None:
    from embodied_dataops.errors import ApiError, ErrorCode

    with pytest.raises(ApiError) as excinfo:
        store_pointcloud_summary_asset(
            db,
            episode_id=new_id(),
            canonical_version=2,
            summaries=[],
            storage=storage,
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_stored_asset_row_has_no_raw_coordinate_payload(db, storage) -> None:
    episode_id = new_id()
    asset = store_pointcloud_summary_asset(
        db,
        episode_id=episode_id,
        canonical_version=2,
        summaries=[parse_pointcloud2(_message())],
        storage=storage,
    )
    db.flush()
    row = db.get(Asset, asset.id)
    assert row is not None
    assert set(row.meta_json) == {
        "kind",
        "schema_version",
        "episode_id",
        "canonical_version",
        "rows",
        "max_points",
        "max_sample_points",
    }
