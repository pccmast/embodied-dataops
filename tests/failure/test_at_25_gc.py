"""AT-25: orphan GC never deletes referenced assets and respects the 24 h window."""

from __future__ import annotations

import os
import time

import pytest

from embodied_dataops.cli.gc import collect_garbage
from embodied_dataops.db.models import Asset
from embodied_dataops.ids import new_id

pytestmark = [pytest.mark.failure, pytest.mark.integration]

OLD = 48 * 3600


def _age_file(storage, key: str, seconds: int) -> None:
    path = storage._path(key)
    past = time.time() - seconds
    os.utime(path, (past, past))


def test_gc_protects_referenced_assets_and_old_orphans(db, storage):
    referenced_key = "artifacts/staging/task-ref/output.json"
    old_orphan_key = "artifacts/staging/task-old/orphan.bin"
    young_orphan_key = "artifacts/staging/task-young/orphan.bin"

    storage.put_bytes(referenced_key, b"referenced")
    storage.put_bytes(old_orphan_key, b"old-orphan")
    storage.put_bytes(young_orphan_key, b"young-orphan")
    _age_file(storage, referenced_key, OLD)
    _age_file(storage, old_orphan_key, OLD)

    db.add(
        Asset(
            id=new_id(),
            sha256="a" * 64,
            byte_size=10,
            media_type="application/json",
            object_key=referenced_key,
            origin="report",
            logical_name="referenced",
        )
    )
    db.commit()

    dry = collect_garbage(db, storage, dry_run=True)
    assert dry["deleted_count"] == 0
    assert [row["key"] for row in dry["candidates"]] == [old_orphan_key]
    assert any(row["key"] == referenced_key for row in dry["protected"])
    assert any(row["key"] == young_orphan_key for row in dry["protected"])

    real = collect_garbage(db, storage, dry_run=False)
    assert real["deleted"] == [old_orphan_key]
    assert storage.exists(referenced_key), "a referenced asset must never be deleted"
    assert storage.exists(young_orphan_key), "orphans younger than 24h must survive"
    assert not storage.exists(old_orphan_key)


def test_gc_reports_why_objects_are_protected(db, storage):
    key = "artifacts/staging/task-x/file.bin"
    storage.put_bytes(key, b"payload")
    result = collect_garbage(db, storage, dry_run=True, orphan_hours=24)
    protected = {row["key"]: row for row in result["protected"]}
    assert key in protected
    assert protected[key]["reason"] == "younger_than_orphan_window"
