"""Operation receipt read model (DO-A02 C4).

``GET /operations/{id}`` and ``GET /operations/by-key/{key}`` expose the receipt
plus the **real** current state of the downstream business object (PipelineRun or
GapAnalysis).  They never expose signatures, presigned URLs, object keys, secret
values or stack traces.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.db.models import GapAnalysis, OperationReceipt, PipelineRun
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.operations.contract import TENANT_ID

#: Contract state vocabulary per downstream kind (DO-A02 C4).
PIPELINE_RUN_STATES = frozenset({"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"})
GAP_ANALYSIS_STATES = frozenset({"QUEUED", "RUNNING", "SUCCEEDED", "FAILED"})


def iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_receipt(session: Session, operation_id: str) -> OperationReceipt:
    receipt = session.get(OperationReceipt, operation_id)
    if receipt is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Operation receipt not found", details={"operation_id": operation_id})
    return receipt


def get_receipt_by_key(session: Session, idempotency_key: str) -> OperationReceipt:
    receipt = session.scalars(
        select(OperationReceipt).where(
            OperationReceipt.tenant_id == TENANT_ID,
            OperationReceipt.idempotency_key == idempotency_key,
        )
    ).first()
    if receipt is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "No operation receipt exists for this Idempotency-Key",
            details={"idempotency_key": idempotency_key},
        )
    return receipt


def _downstream_snapshot(
    session: Session, receipt: OperationReceipt
) -> tuple[str | None, str | None]:
    if receipt.downstream_id is None:
        return None, None
    if receipt.downstream_kind == "pipeline_run":
        run = session.get(PipelineRun, receipt.downstream_id)
        if run is None:
            return "UNKNOWN", None
        state = run.state if run.state in PIPELINE_RUN_STATES else "UNKNOWN"
        updated = run.finished_at or run.started_at or run.created_at
        return state, iso_utc(updated)
    if receipt.downstream_kind == "gap_analysis":
        analysis = session.get(GapAnalysis, receipt.downstream_id)
        if analysis is None:
            return "UNKNOWN", None
        state = analysis.state if analysis.state in GAP_ANALYSIS_STATES else "UNKNOWN"
        updated = analysis.finished_at or analysis.created_at
        return state, iso_utc(updated)
    return "UNKNOWN", None


def serialize_receipt(session: Session, receipt: OperationReceipt) -> dict:
    state, updated_at = _downstream_snapshot(session, receipt)
    return {
        "operation_id": receipt.id,
        "action_kind": receipt.action_kind,
        "target_id": receipt.target_id,
        "source_business_revision": receipt.source_business_revision,
        "effect_state": receipt.effect_state,
        "downstream_kind": receipt.downstream_kind,
        "downstream_id": receipt.downstream_id,
        "downstream_state": state,
        "downstream_updated_at": updated_at,
        "error_code": receipt.error_code,
    }


__all__ = [
    "GAP_ANALYSIS_STATES",
    "PIPELINE_RUN_STATES",
    "get_receipt",
    "get_receipt_by_key",
    "iso_utc",
    "serialize_receipt",
]
