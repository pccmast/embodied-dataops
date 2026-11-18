"""The DO-A02 conditional-write transaction (C2/C3).

One request = one PostgreSQL transaction with a fixed order (ADR-0007 §7):

1. validate ``If-Match`` presence and ``Idempotency-Key`` format;
2. look up ``operation_receipts`` by ``(tenant, idempotency_key)`` and compare the
   canonical request hash - **idempotency before If-Match**, so a committed
   request can always be replayed even after the business revision moved;
3. ``SET LOCAL lock_timeout`` and lock the business resource row;
4. re-check the receipt (a concurrent same-key request may have committed while
   we waited for the row lock) - replay it, or 503 if the lock wait expired;
5. validate the strong ETag, then the business fence: the same
   ``(tenant, action_kind, target, source_revision)`` may only produce one effect,
   a re-entry returns ``409 OPERATION_ALREADY_APPLIED`` with the original
   ``operation_id``;
6. insert the fence, create the downstream effect (Run/Analysis) + outbox row +
   ``operation_receipt`` and bump ``business_revision`` by 1;
7. commit, then the route returns 202 ``effect_state=COMMITTED``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import ActionEffectFence, OperationReceipt
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.observability.audit import record_audit
from embodied_dataops.operations import revisions
from embodied_dataops.operations.contract import COMMIT_WAIT_MS, TENANT_ID
from embodied_dataops.operations.etag import parse_precondition, parse_requested_revision

logger = logging.getLogger(__name__)

ResourceLock = Callable[[Session], Any | None]
EffectFn = Callable[[Session, Any, dict[str, Any], str], tuple[str, str]]


def request_fingerprint(
    *,
    action_kind: str,
    target_id: str,
    body: dict[str, Any],
    if_match: str,
) -> str:
    """``sha256`` over tenant+action+target+canonical body+raw strong ETag (C3)."""
    payload = {
        "tenant_id": TENANT_ID,
        "action_kind": action_kind,
        "target_id": target_id,
        "body": body,
        "if_match": if_match,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def find_receipt(session: Session, idempotency_key: str) -> OperationReceipt | None:
    statement = select(OperationReceipt).where(
        OperationReceipt.tenant_id == TENANT_ID,
        OperationReceipt.idempotency_key == idempotency_key,
    )
    return session.scalars(statement).first()


def find_fence(
    session: Session, *, action_kind: str, target_id: str, source_revision: int
) -> ActionEffectFence | None:
    statement = select(ActionEffectFence).where(
        ActionEffectFence.tenant_id == TENANT_ID,
        ActionEffectFence.action_kind == action_kind,
        ActionEffectFence.target_id == target_id,
        ActionEffectFence.source_business_revision == int(source_revision),
    )
    return session.scalars(statement).first()


def execute_controlled_write(
    session: Session,
    *,
    principal: Principal,
    action_kind: str,
    etag_kind: str,
    echo_field: str,
    target_id: str,
    body: dict[str, Any],
    idempotency_key: str | None,
    if_match: str | None,
    lock_resource: ResourceLock,
    apply_effect: EffectFn,
    request_id: str | None = None,
    audit_entity_type: str,
) -> dict[str, Any]:
    """Run one conditional write; return the 202 ``data`` payload."""
    if if_match is None or not str(if_match).strip():
        raise ApiError(
            ErrorCode.PRECONDITION_REQUIRED,
            "If-Match with the resource's strong ETag is required for this operation",
            details={"required_header": "If-Match", "resource_kind": etag_kind},
        )
    if not idempotency_key:
        raise ApiError(
            ErrorCode.IDEMPOTENCY_KEY_REQUIRED,
            "Idempotency-Key header is required for this operation",
        )
    try:
        uuid.UUID(str(idempotency_key))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "Idempotency-Key must be a UUID",
            details={"idempotency_key": str(idempotency_key)[:64]},
        ) from exc
    idempotency_key = str(idempotency_key)
    if_match = str(if_match).strip()

    fingerprint = request_fingerprint(
        action_kind=action_kind, target_id=target_id, body=body, if_match=if_match
    )
    receipt = find_receipt(session, idempotency_key)
    if receipt is not None:
        return _replay(receipt, fingerprint=fingerprint, echo_field=echo_field)

    # In-flight same-key retries may wait for the row lock, but never forever.
    session.execute(text(f"SET LOCAL lock_timeout = '{int(COMMIT_WAIT_MS)}ms'"))
    try:
        resource = lock_resource(session)
    except DBAPIError as exc:
        if _is_lock_timeout(exc):
            session.rollback()
            raise ApiError(
                ErrorCode.COMMIT_IN_PROGRESS,
                "The same business operation is still committing; retry with the same Idempotency-Key",
                details={"retry_after_ms": COMMIT_WAIT_MS},
            ) from exc
        raise
    if resource is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "Target resource not found",
            details={"target_id": target_id, "action_kind": action_kind},
        )

    # A concurrent same-key request may have committed while we waited for the lock.
    receipt = find_receipt(session, idempotency_key)
    if receipt is not None:
        return _replay(receipt, fingerprint=fingerprint, echo_field=echo_field)

    current_revision = revisions.current_revision(resource)
    # Fence check first: the same requested source revision already applied by a
    # different Idempotency-Key must return 409 with the original operation id,
    # even though that revision is now stale relative to the resource (C3/C4).
    requested_revision = parse_requested_revision(if_match, expected_kind=etag_kind)
    existing = find_fence(
        session,
        action_kind=action_kind,
        target_id=target_id,
        source_revision=requested_revision,
    )
    if existing is not None:
        raise _already_applied(
            existing.operation_id, action_kind, target_id, requested_revision, current_revision
        )
    precondition = parse_precondition(
        if_match, expected_kind=etag_kind, current_revision=current_revision
    )

    operation_id = new_id()
    # The fence has an FK to operation_receipts, so the receipt row must exist
    # before the fence is inserted (and therefore before apply_effect).
    receipt = OperationReceipt(
        id=operation_id,
        tenant_id=TENANT_ID,
        action_kind=action_kind,
        target_id=target_id,
        source_business_revision=precondition.revision,
        idempotency_key=idempotency_key,
        request_sha256=fingerprint,
        effect_state="IN_PROGRESS",
    )
    session.add(receipt)
    session.flush()
    _insert_fence(
        session,
        operation_id=operation_id,
        action_kind=action_kind,
        target_id=target_id,
        source_revision=precondition.revision,
        current_revision=current_revision,
    )

    downstream_kind, downstream_id = apply_effect(session, resource, body, operation_id)
    receipt.effect_state = "COMMITTED"
    receipt.downstream_kind = downstream_kind
    receipt.downstream_id = downstream_id
    session.flush()
    resource.business_revision = precondition.revision + 1
    if hasattr(resource, "updated_at"):
        from embodied_dataops.workflow.outbox import utcnow

        resource.updated_at = utcnow()
    session.flush()
    record_audit(
        session,
        actor=principal,
        action=f"operation.{action_kind.lower()}",
        entity_type=audit_entity_type,
        entity_id=target_id,
        after={
            "operation_id": operation_id,
            "source_business_revision": precondition.revision,
            "business_revision": precondition.revision + 1,
            "downstream_kind": downstream_kind,
            "downstream_id": downstream_id,
        },
        request_id=request_id,
    )
    session.commit()
    logger.info(
        "operation.committed action=%s target=%s operation_id=%s downstream=%s:%s",
        action_kind,
        target_id,
        operation_id,
        downstream_kind,
        downstream_id,
    )
    return _success_payload(operation_id, echo_field, downstream_id, "COMMITTED")


def _insert_fence(
    session: Session,
    *,
    operation_id: str,
    action_kind: str,
    target_id: str,
    source_revision: int,
    current_revision: int,
) -> None:
    try:
        with session.begin_nested():
            session.add(
                ActionEffectFence(
                    id=new_id(),
                    tenant_id=TENANT_ID,
                    action_kind=action_kind,
                    target_id=target_id,
                    source_business_revision=source_revision,
                    operation_id=operation_id,
                )
            )
            session.flush()
    except IntegrityError as exc:
        # Two proposals raced on the unique fence key: only one may take effect.
        existing = find_fence(
            session,
            action_kind=action_kind,
            target_id=target_id,
            source_revision=source_revision,
        )
        if existing is not None:
            raise _already_applied(
                existing.operation_id, action_kind, target_id, source_revision, current_revision
            ) from exc
        raise


def _already_applied(
    operation_id: str,
    action_kind: str,
    target_id: str,
    source_revision: int,
    current_revision: int,
) -> ApiError:
    return ApiError(
        ErrorCode.OPERATION_ALREADY_APPLIED,
        "An equivalent operation was already applied for this business revision",
        details={
            "operation_id": operation_id,
            "action_kind": action_kind,
            "target_id": target_id,
            "source_business_revision": source_revision,
            "current_business_revision": current_revision,
        },
    )


def _replay(receipt: OperationReceipt, *, fingerprint: str, echo_field: str) -> dict[str, Any]:
    if receipt.request_sha256 != fingerprint:
        raise ApiError(
            ErrorCode.IDEMPOTENCY_KEY_REUSED,
            "Idempotency-Key was reused with a different request body or If-Match",
            details={"operation_id": receipt.id, "action_kind": receipt.action_kind},
        )
    logger.info("operation.replayed operation_id=%s", receipt.id)
    return _success_payload(receipt.id, echo_field, receipt.downstream_id, receipt.effect_state)


def _success_payload(
    operation_id: str, echo_field: str, downstream_id: str | None, effect_state: str
) -> dict[str, Any]:
    return {
        "operation_id": operation_id,
        echo_field: downstream_id,
        "state": "QUEUED",
        "effect_state": effect_state or "COMMITTED",
    }


def _is_lock_timeout(exc: DBAPIError) -> bool:
    original = getattr(exc, "orig", None)
    sqlstate = getattr(original, "sqlstate", None)
    if sqlstate == "55P03":  # lock_not_available
        return True
    return "lock timeout" in str(original).lower() or "canceling statement due to lock timeout" in str(exc).lower()
