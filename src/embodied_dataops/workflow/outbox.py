"""Transactional outbox (SPEC 6.3).

Task creation and the outbox row are written in one PostgreSQL transaction.
The dispatcher publishes outbox rows to Redis Streams; a publish failure only
delays a wakeup because PostgreSQL remains the task truth source and the reaper
re-publishes missed wakeups.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import OutboxEvent
from embodied_dataops.domain.enums import STEP_POOL, Step
from embodied_dataops.ids import new_id

TOPIC_BY_POOL: dict[str, str] = {
    "cpu": "stream:tasks:cpu",
    "vision": "stream:tasks:vision",
    "ros2": "stream:tasks:ros2",
}
ALL_STREAMS = tuple(TOPIC_BY_POOL.values())


def topic_for_step(step: Step) -> str:
    return TOPIC_BY_POOL[STEP_POOL[step]]


def utcnow() -> datetime:
    return datetime.now(UTC)


def enqueue_task_wakeup(session: Session, task_id: str, pool: str) -> OutboxEvent:
    """Add an outbox row inside the caller's transaction (no commit here)."""
    event = OutboxEvent(
        id=new_id(),
        topic=TOPIC_BY_POOL[pool],
        payload_json={"task_id": task_id, "pool": pool},
        created_at=utcnow(),
    )
    session.add(event)
    return event


def take_pending(session: Session, limit: int = 100) -> list[OutboxEvent]:
    """Select unpublished outbox rows ordered by creation time (row-locked)."""
    stmt = (
        select(OutboxEvent)
        .where(OutboxEvent.published_at.is_(None))
        .order_by(OutboxEvent.created_at, OutboxEvent.id)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    return list(session.scalars(stmt))


def mark_published(event: OutboxEvent, *, when: datetime | None = None) -> None:
    event.published_at = when or utcnow()
    event.delivery_attempt += 1


def pending_count(session: Session) -> int:
    stmt = select(OutboxEvent.id).where(OutboxEvent.published_at.is_(None))
    return len(list(session.scalars(stmt)))


def serialize(event: OutboxEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "topic": event.topic,
        "payload": event.payload_json,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "delivery_attempt": event.delivery_attempt,
    }
