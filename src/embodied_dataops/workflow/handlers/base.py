"""Task handler contract shared by every pipeline step module.

A handler runs inside the engine's database transaction.  It may:

* upload/locate objects in the object store,
* create rows (assets, annotation sets, quality reports, ...) through ``ctx.session``,
* return a :class:`HandlerResult`.

It must NOT commit, roll back, change ``tasks.state``, or enqueue wakeups: the
engine owns the lease fencing CAS, retries and downstream tasks.  If the CAS
later fails (stale lease) the whole transaction is rolled back, so rows created
here can never become stale valid results.

Object layout convention (SPEC 5.2, 6.3):

* handlers write **final, immutable, deterministic** keys (e.g.
  ``canonical/{episode_id}/v1/samples.parquet``);
* helper :func:`register_bytes` / :func:`register_file` upload the object and
  create the ``assets`` row in one call;
* unrereferenced objects left behind by a crashed execution are reclaimed by
  the 24 h orphan GC (``embodied_dataops.cli.gc``).
"""

from __future__ import annotations

import abc
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.config import Settings
from embodied_dataops.db.models import Asset, Task
from embodied_dataops.domain.enums import AssetOrigin, Step
from embodied_dataops.ids import new_id
from embodied_dataops.storage.base import StorageBackend


@dataclass
class HandlerContext:
    session: Session
    task: Task
    settings: Settings
    storage: StorageBackend
    workspace: Path
    cancel_check: Callable[[], bool]

    def is_cancelled(self) -> bool:
        return self.cancel_check()

    def staging_key(self, name: str) -> str:
        token = self.task.lease_token or "no-lease"
        return f"artifacts/staging/{self.task.id}/{token}/{name}"


@dataclass
class HandlerResult:
    """Outcome of one successful task.

    ``output_asset_id`` is persisted as ``tasks.output_asset_id`` (the primary
    artifact of the step).  ``data`` becomes ``tasks.output_json`` and is what
    downstream steps read.  ``verdict`` lets the engine stop downstream steps
    (QC FAIL) even though the task itself succeeded.
    """

    data: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    episode_status: str | None = None
    output_asset_id: str | None = None


class TaskHandler(abc.ABC):
    step: Step

    @abc.abstractmethod
    def handle(self, ctx: HandlerContext) -> HandlerResult: ...


# ---------------------------------------------------------------------------
# asset registration helpers (used by handlers, Lead-owned)
# ---------------------------------------------------------------------------


def register_bytes(
    ctx: HandlerContext,
    *,
    key: str,
    data: bytes,
    media_type: str,
    origin: AssetOrigin | str,
    logical_name: str = "",
    meta: dict[str, Any] | None = None,
) -> Asset:
    """Upload bytes to ``key`` and create the immutable Asset row."""
    stored = ctx.storage.put_bytes(key, data, content_type=media_type)
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        media_type=media_type,
        object_key=key,
        origin=origin.value if isinstance(origin, AssetOrigin) else str(origin),
        logical_name=logical_name,
        meta_json=meta or {},
    )
    ctx.session.add(asset)
    ctx.session.flush()
    return asset


def register_file(
    ctx: HandlerContext,
    *,
    key: str,
    path: Path,
    media_type: str,
    origin: AssetOrigin | str,
    logical_name: str = "",
    meta: dict[str, Any] | None = None,
) -> Asset:
    stored = ctx.storage.put_file(key, path, content_type=media_type)
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or "",
        byte_size=path.stat().st_size,
        media_type=media_type,
        object_key=key,
        origin=origin.value if isinstance(origin, AssetOrigin) else str(origin),
        logical_name=logical_name,
        meta_json=meta or {},
    )
    ctx.session.add(asset)
    ctx.session.flush()
    return asset


def register_json(
    ctx: HandlerContext,
    *,
    key: str,
    payload: Any,
    media_type: str = "application/json",
    origin: AssetOrigin | str = AssetOrigin.REPORT,
    logical_name: str = "",
    meta: dict[str, Any] | None = None,
    canonical: bool = True,
) -> Asset:
    data = canonical_json_bytes(payload) if canonical else json.dumps(payload).encode()
    return register_bytes(
        ctx,
        key=key,
        data=data,
        media_type=media_type,
        origin=origin,
        logical_name=logical_name,
        meta=meta,
    )


def read_json(ctx: HandlerContext, key: str) -> Any:
    return json.loads(ctx.storage.get_bytes(key).decode("utf-8"))
