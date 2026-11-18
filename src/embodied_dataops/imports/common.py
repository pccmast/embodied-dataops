"""Shared plumbing for the M1 import handlers (SPEC 5.1, 6.1).

``embodied_dataops.imports.base`` is the Lead-owned adapter interface; this
module keeps the *handler-facing* helpers next to the adapters so the three
pipeline steps (VALIDATE_SOURCE / IMPORT_EPISODES / NORMALIZE_EPISODE) share one
implementation of: loading the import job, verifying + downloading the immutable
raw object, building an :class:`~embodied_dataops.imports.base.ImportContext`,
and recording structured, path-free import failures.

Diagnostics are plain dicts (JSON-ready) so they can be persisted in
``ImportJob.error_detail`` / canonical ``metadata.json`` and consumed by the
quality stage without importing this module::

    {"code": "TIMESTAMP_NON_MONOTONIC", "severity": "ERROR",
     "message": "...", "details": {...}, "structural": true}
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from embodied_dataops.db.models import Asset, ImportJob, SourceImport
from embodied_dataops.domain.enums import DataOrigin, Severity, SourceType
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.imports.base import ImportContext
from embodied_dataops.storage.base import StorageBackend, sha256_file

#: Codes that make a canonical episode *structurally* invalid: SPEC 6.1 requires
#: minimal metadata + diagnostics and forbids publishing a samples.parquet.
STRUCTURAL_DIAGNOSTIC_CODES: frozenset[str] = frozenset(
    {
        "TIMESTAMP_NON_MONOTONIC",
        "DIMENSION_MISMATCH",
        "NONFINITE_VALUE",
        "SYNC_MISSING_FRAME",
        "SYNC_MISSING_STATE",
        "SYNC_STALE_ACTION",
        "FRAME_GAP_EXCESS",
        "DURATION_EXCEEDED",
        "SOURCE_DURATION_LIMIT",
        "FILE_DECODE_ERROR",
        "LEROBOT_METADATA_INVALID",
        "EPISODE_ENUMERATION_FAILED",
        "MAPPING_REQUIRED",
        "INVALID_MAPPING",
    }
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def diagnostic(
    code: str,
    severity: Severity | str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    structural: bool | None = None,
) -> dict[str, Any]:
    """Build one JSON-ready diagnostic record (never contains local paths)."""
    severity_value = severity.value if isinstance(severity, Severity) else str(severity)
    if structural is None:
        structural = code in STRUCTURAL_DIAGNOSTIC_CODES
    return {
        "code": code,
        "severity": severity_value,
        "message": message,
        "details": details or {},
        "structural": bool(structural),
    }


def merge_diagnostics(*groups: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for item in group or []:
            key = (str(item.get("code")), str(item.get("message")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def has_structural(diagnostics: Sequence[dict[str, Any]]) -> bool:
    return any(bool(item.get("structural")) for item in diagnostics)


def error_diagnostic(exc: ApiError) -> dict[str, Any]:
    return diagnostic(exc.code.value, Severity.ERROR, exc.message, details=dict(exc.details))


# ---------------------------------------------------------------------------
# loading the import + immutable raw object
# ---------------------------------------------------------------------------


@dataclass
class ImportSourceBundle:
    """Everything a handler may read while validating/parsing one import."""

    job: ImportJob
    source_import: SourceImport
    asset: Asset
    path: Path
    context: ImportContext
    workdir: Path = field(default_factory=lambda: Path("."))


def load_import_job(session: Session, job_id: str | None) -> ImportJob:
    if not job_id:
        raise ApiError(ErrorCode.INVALID_INPUT, "task is not linked to an import job")
    job = session.get(ImportJob, job_id)
    if job is None:
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, "import job does not exist")
    return job


def load_source_import(session: Session, import_id: str) -> SourceImport:
    source_import = session.get(SourceImport, import_id)
    if source_import is None:
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, "source import does not exist")
    return source_import


def load_asset(session: Session, source_import: SourceImport) -> Asset:
    if not source_import.asset_id:
        raise ApiError(
            ErrorCode.SOURCE_UPLOAD_INCOMPLETE,
            "source import has no completed immutable object",
        )
    asset = session.get(Asset, source_import.asset_id)
    if asset is None:
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, "source asset record is missing")
    return asset


def _download_name(source_import: SourceImport, asset: Asset) -> str:
    filename = Path(source_import.filename or "").name
    if filename:
        return filename
    return Path(asset.object_key).name or "source.bin"


def materialize_source(
    storage: StorageBackend,
    source_import: SourceImport,
    asset: Asset,
    workspace: Path,
) -> Path:
    """Download the immutable raw object and re-verify its SHA-256 (SPEC 12.1)."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / _download_name(source_import, asset)
    stored = storage.download_to(asset.object_key, target)
    actual_sha = stored.sha256 or sha256_file(target)
    expected_sha = asset.sha256 or source_import.sha256 or ""
    if expected_sha and actual_sha != expected_sha:
        raise ApiError(
            ErrorCode.SOURCE_HASH_MISMATCH,
            "downloaded source object does not match the recorded SHA-256",
            details={"expected_sha256": expected_sha, "actual_sha256": actual_sha},
        )
    if asset.byte_size and target.stat().st_size != asset.byte_size:
        raise ApiError(
            ErrorCode.SOURCE_UPLOAD_INCOMPLETE,
            "downloaded source object size does not match the recorded size",
            details={"expected_bytes": asset.byte_size, "actual_bytes": target.stat().st_size},
        )
    return target


def load_import_source(
    session: Session,
    storage: StorageBackend,
    task: Any,
    workspace: Path,
) -> ImportSourceBundle:
    """Resolve task -> import job -> source import -> verified local raw object."""
    job = load_import_job(session, getattr(task, "import_job_id", None))
    source_import = load_source_import(session, job.import_id)
    return materialize_import_source(session, storage, job, source_import, workspace)


def build_import_context(source_import: SourceImport, path: Path, workdir: Path) -> ImportContext:
    """Build the adapter context for one verified local raw object."""
    return ImportContext(
        import_id=source_import.id,
        source_type=SourceType(source_import.source_type),
        data_origin=DataOrigin(source_import.data_origin),
        source_path=Path(path),
        workdir=Path(workdir),
        mapping=dict(source_import.mapping_json or {}),
        license=source_import.license or "",
        source_uri=source_import.source_uri,
        source_revision=source_import.source_revision,
        source_sha256=source_import.sha256 or "",
    )


def materialize_import_source(
    session: Session,
    storage: StorageBackend,
    job: ImportJob,
    source_import: SourceImport,
    workspace: Path,
) -> ImportSourceBundle:
    asset = load_asset(session, source_import)
    path = materialize_source(storage, source_import, asset, workspace)
    context = build_import_context(source_import, path, workspace)
    if not context.source_sha256:
        context.source_sha256 = asset.sha256
    return ImportSourceBundle(
        job=job,
        source_import=source_import,
        asset=asset,
        path=path,
        context=context,
        workdir=Path(workspace),
    )


# ---------------------------------------------------------------------------
# failure recording (returns the HandlerResult.data payload, never raises)
# ---------------------------------------------------------------------------


def record_import_failure(
    session: Session,
    job: ImportJob,
    source_import: SourceImport,
    *,
    code: ErrorCode | str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a failed import job and return the handler ``data`` payload.

    The caller still returns ``verdict="FAIL"``; the workflow engine commits the
    in-handler rows and cancels downstream import steps (SPEC 6.1: an
    undecodable/unenumerable source yields failure evidence and no Episodes).
    """
    code_value = code.value if isinstance(code, ErrorCode) else str(code)
    now = utcnow()
    session.flush()
    job.state = "FAILED"
    job.error_code = code_value
    job.error_detail = message[:2000]
    job.finished_at = now
    source_import.upload_state = "FAILED"
    source_import.error_code = code_value
    source_import.error_detail = message[:2000]
    source_import.completed_at = now
    session.flush()
    return {
        "valid": False,
        "error_code": code_value,
        "error_detail": message,
        "diagnostics": [diagnostic(code_value, Severity.ERROR, message, details=details)],
    }


def import_context_payload(bundle: ImportSourceBundle) -> dict[str, Any]:
    """Path-free identity of the imported object, safe to persist as evidence."""
    return {
        "import_id": bundle.source_import.id,
        "source_type": bundle.source_import.source_type,
        "data_origin": bundle.source_import.data_origin,
        "asset_sha256": bundle.asset.sha256,
        "asset_bytes": bundle.asset.byte_size,
        "filename": bundle.source_import.filename,
        "source_uri": bundle.source_import.source_uri,
        "source_revision": bundle.source_import.source_revision,
        "license": bundle.source_import.license,
        "mapping": dict(bundle.source_import.mapping_json or {}),
    }


__all__ = [
    "STRUCTURAL_DIAGNOSTIC_CODES",
    "ImportSourceBundle",
    "build_import_context",
    "diagnostic",
    "error_diagnostic",
    "has_structural",
    "import_context_payload",
    "load_asset",
    "load_import_job",
    "load_import_source",
    "load_source_import",
    "materialize_import_source",
    "materialize_source",
    "merge_diagnostics",
    "record_import_failure",
    "utcnow",
]
