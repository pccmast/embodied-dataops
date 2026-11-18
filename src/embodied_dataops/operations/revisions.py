"""``business_revision`` helpers (DO-A02 C2).

``business_revision`` is the *business* state version used for conditional writes;
it is deliberately not ``canonical_version`` and not a source/model version.  Any
business precondition change (task state transition, episode status / quality
pointer change, evaluation result version) must bump it inside the same
transaction.  Repositories in other V2 modules can call :func:`bump` so no
ordinary state update bypasses the revision.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def current_revision(resource: Any) -> int:
    return int(getattr(resource, "business_revision", 1) or 1)


def bump(session: Session, resource: Any, *, amount: int = 1) -> int:
    """Increase ``resource.business_revision`` by ``amount`` (same transaction)."""
    revision = current_revision(resource) + int(amount)
    resource.business_revision = revision
    if hasattr(resource, "updated_at"):
        from embodied_dataops.workflow.outbox import utcnow

        resource.updated_at = utcnow()
    session.flush()
    return revision


def bump_if_business_change(session: Session, resource: Any, *, before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Bump only when one of the watched business fields changed."""
    changed = any(before.get(key) != after.get(key) for key in before)
    if changed:
        bump(session, resource)
    return changed
