"""UUIDv7 identifiers (SPEC 4.2): time-ordered, generated outside the database."""

from __future__ import annotations

import os
import threading
import time
import uuid

_lock = threading.Lock()
_last_ms = 0
_sequence = 0


def uuid7() -> uuid.UUID:
    """Return a monotonic UUIDv7 for the current millisecond.

    Layout (RFC 9562): 48-bit unix_ts_ms | 4-bit version | 12-bit rand_a |
    2-bit variant | 62-bit rand_b.  Within one millisecond the 12-bit rand_a
    counter is incremented to preserve ordering even for tight loops.
    """
    global _last_ms, _sequence
    with _lock:
        now_ms = int(time.time() * 1000)
        if now_ms > _last_ms:
            _last_ms = now_ms
            _sequence = 0
        else:
            _sequence += 1
            if _sequence >= 0xFFF:
                # Exhausted the per-millisecond space: borrow the next ms.
                _last_ms += 1
                now_ms = _last_ms
                _sequence = 0
        unix_ts_ms = _last_ms
        rand_a = _sequence
    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
    value = (unix_ts_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0b10 << 62
    value |= rand_b
    return uuid.UUID(int=value)


def new_id() -> str:
    return str(uuid7())


def new_token() -> str:
    """Opaque random token for leases / idempotency keys."""
    return str(uuid.uuid4())


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def uuid7_timestamp_ms(value: str | uuid.UUID) -> int:
    """Extract the millisecond timestamp encoded in a UUIDv7."""
    parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    return parsed.int >> 80
