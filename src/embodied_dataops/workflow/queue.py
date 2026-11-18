"""Redis Streams transport (SPEC 6.3).

Redis carries wakeup hints only.  Every execution starts by taking a database
lease, so duplicated, lost or replayed stream entries can never produce two
valid executions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import redis

from embodied_dataops.config import get_settings

GROUP_NAME = "dataops-workers"


def create_client(url: str | None = None) -> redis.Redis:
    settings = get_settings()
    client = redis.Redis.from_url(url or settings.redis_url, decode_responses=True)
    return client


def ensure_group(client: redis.Redis, stream: str, group: str = GROUP_NAME) -> None:
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as exc:  # BUSYGROUP -> already exists
        if "BUSYGROUP" not in str(exc):
            raise


def publish_task(client: redis.Redis, stream: str, payload: dict[str, Any]) -> str:
    return client.xadd(stream, {"payload": json.dumps(payload, sort_keys=True)}, maxlen=100_000)


def read_tasks(
    client: redis.Redis,
    stream: str,
    consumer: str,
    *,
    count: int = 1,
    block_ms: int = 1000,
    group: str = GROUP_NAME,
) -> list[tuple[str, dict[str, Any]]]:
    ensure_group(client, stream, group)
    response = client.xreadgroup(
        group, consumer, {stream: ">"}, count=count, block=block_ms
    )
    messages: list[tuple[str, dict[str, Any]]] = []
    for _stream, entries in response or []:
        for entry_id, fields in entries:
            raw = fields.get("payload")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:  # pragma: no cover - corrupted entry
                continue
            messages.append((entry_id, payload))
    return messages


def ack(client: redis.Redis, stream: str, entry_id: str, group: str = GROUP_NAME) -> None:
    client.xack(stream, group, entry_id)


def stream_depth(client: redis.Redis, stream: str) -> int:
    try:
        return int(client.xlen(stream))
    except Exception:  # pragma: no cover - redis unavailable
        return -1


def reclaim_stale(
    client: redis.Redis, stream: str, consumer: str, min_idle_ms: int = 120_000, count: int = 32
) -> list[str]:
    """XAUTOCLAIM old pending entries; the database lease remains authoritative."""
    try:
        response = client.xautoclaim(
            stream, GROUP_NAME, consumer, min_idle_time=min_idle_ms, start_id="0-0", count=count
        )
    except redis.ResponseError:
        return []
    claimed = response[1] if isinstance(response, (list, tuple)) and len(response) > 1 else []
    return [entry[0] for entry in claimed or []]


def trim(client: redis.Redis, stream: str, maxlen: int = 100_000) -> None:
    try:
        client.xtrim(stream, maxlen=maxlen, approximate=True)
    except Exception:  # pragma: no cover
        pass


def ping(client: redis.Redis) -> bool:
    try:
        return bool(client.ping())
    except Exception:
        return False


def wait_for(client: redis.Redis, streams: Iterable[str], timeout_seconds: float = 30.0) -> None:
    """Block until every stream has a consumer group (worker startup helper)."""
    import time

    deadline = time.time() + timeout_seconds
    for stream in streams:
        while time.time() < deadline:
            try:
                ensure_group(client, stream)
                break
            except Exception:  # pragma: no cover - startup race
                time.sleep(0.2)
