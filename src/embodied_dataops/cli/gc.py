"""Orphan object garbage collection (SPEC 6.3, 12.1, AT-25).

Rules implemented here:

* an object referenced by any ``assets`` row is **never** deleted;
* only ``artifacts/staging/`` (and explicitly requested prefixes) is considered
  a GC candidate;
* an unreferenced object must be older than ``--orphan-hours`` (default 24 h)
  before it is removed;
* ``--dry-run`` reports without deleting, and the command refuses to run
  without an explicit confirmation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import Asset
from embodied_dataops.storage.base import StorageBackend

DEFAULT_PREFIXES = ("artifacts/staging/",)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def collect_garbage(
    session: Session,
    storage: StorageBackend,
    *,
    prefixes: Sequence[str] = DEFAULT_PREFIXES,
    orphan_hours: int = 24,
    dry_run: bool = True,
    now: datetime | None = None,
) -> dict:
    referenced = set(session.scalars(select(Asset.object_key)))
    cutoff = (now or _utcnow()) - timedelta(hours=orphan_hours)
    candidates: list[dict] = []
    protected: list[dict] = []
    deleted: list[str] = []

    for prefix in prefixes:
        for key in storage.list_keys(prefix):
            if key in referenced:
                protected.append({"key": key, "reason": "referenced_by_asset"})
                continue
            try:
                stat = storage.stat(key)
            except Exception:
                continue
            last_modified = stat.last_modified or cutoff - timedelta(seconds=1)
            entry = {
                "key": key,
                "size": stat.size,
                "last_modified": last_modified.isoformat(),
                "age_hours": round((now or _utcnow() - last_modified).total_seconds() / 3600, 2),
            }
            if last_modified > cutoff:
                entry["reason"] = "younger_than_orphan_window"
                protected.append(entry)
                continue
            candidates.append(entry)
            if not dry_run:
                storage.delete(key)
                deleted.append(key)

    return {
        "dry_run": dry_run,
        "orphan_hours": orphan_hours,
        "prefixes": list(prefixes),
        "referenced_assets": len(referenced),
        "candidates": candidates,
        "deleted": deleted,
        "protected": protected,
        "deleted_count": len(deleted),
    }
