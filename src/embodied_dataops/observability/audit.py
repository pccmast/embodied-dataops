"""Audit trail (SPEC 4.3, 9.5).  Every privileged mutation is recorded."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.db.models import AuditEvent
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.outbox import utcnow


def record_audit(
    session: Session,
    *,
    actor: Principal | None,
    action: str,
    entity_type: str,
    entity_id: str,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> AuditEvent:
    event = AuditEvent(
        id=new_id(),
        actor_id=actor.key_id if actor else None,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id),
        before_json=before,
        after_json=after,
        request_id=request_id,
        created_at=utcnow(),
    )
    session.add(event)
    return event
