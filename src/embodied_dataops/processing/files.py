"""Filesystem helpers: safe archive extraction and scratch workspaces (SPEC 5.1, 12.1)."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from embodied_dataops.config import get_settings
from embodied_dataops.errors import ApiError, ErrorCode


@contextmanager
def scratch_workspace(task_id: str, lease_token: str | None = None) -> Iterator[Path]:
    """Isolated scratch directory for one task execution."""
    settings = get_settings()
    base = Path(settings.scratch_dir).resolve()
    suffix = f"{task_id}-{lease_token}" if lease_token else task_id
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f"{suffix[:60]}-", dir=base))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _is_within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar(
    archive_path: Path,
    destination: Path,
    *,
    max_entries: int | None = None,
    max_total_bytes: int | None = None,
) -> dict[str, int]:
    """Bounded, traversal-safe tar extraction (`.tar.gz` only).

    Rejects absolute paths, `..`, links, device/fifo nodes, and archives whose
    expanded size or entry count exceeds the limits.  Nothing is written
    outside ``destination``; the archive itself is never executed.
    """
    settings = get_settings()
    max_entries = max_entries or settings.max_tar_entries
    max_total_bytes = max_total_bytes or settings.max_unpacked_bytes
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    entries = 0
    total_bytes = 0
    extracted = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > max_entries:
                raise ApiError(
                    ErrorCode.ARCHIVE_TOO_LARGE,
                    "Archive contains too many entries",
                    details={"entries": len(members), "limit": max_entries},
                )
            for member in members:
                entries += 1
                name = member.name.replace("\\", "/")
                if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
                    raise ApiError(ErrorCode.UNSAFE_ARCHIVE, "Absolute path in archive")
                parts = [part for part in name.split("/") if part not in ("", ".")]
                if any(part == ".." for part in parts):
                    raise ApiError(ErrorCode.UNSAFE_ARCHIVE, "Path traversal in archive")
                if member.issym() or member.islnk():
                    raise ApiError(ErrorCode.UNSAFE_ARCHIVE, "Links are not allowed in archives")
                if member.isdev() or member.isfifo():
                    raise ApiError(ErrorCode.UNSAFE_ARCHIVE, "Device nodes are not allowed")
                if member.isdir():
                    (destination / Path(*parts)).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ApiError(ErrorCode.UNSAFE_ARCHIVE, "Unsupported archive member type")
                total_bytes += member.size
                if total_bytes > max_total_bytes:
                    raise ApiError(
                        ErrorCode.ARCHIVE_TOO_LARGE,
                        "Expanded archive exceeds the size limit",
                        details={"limit": max_total_bytes},
                    )
                target = (destination / Path(*parts)).resolve()
                if not _is_within(destination, target):
                    raise ApiError(ErrorCode.UNSAFE_ARCHIVE, "Extraction escapes the target directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
                extracted += 1
    except tarfile.TarError as exc:
        raise ApiError(ErrorCode.UNSUPPORTED_ARCHIVE_FORMAT, "Archive is not a valid .tar.gz") from exc
    return {"entries": entries, "files": extracted, "total_bytes": total_bytes}


def detect_archive_kind(path: Path) -> str:
    """Sniff the real archive kind from magic bytes (never trust the extension)."""
    with path.open("rb") as handle:
        head = handle.read(8)
    if head.startswith(b"\x1f\x8b"):
        return "tar.gz"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    return "unknown"


def stream_download_size_guard(path: Path, limit: int | None = None) -> int:
    settings = get_settings()
    limit = limit or settings.max_upload_bytes
    size = path.stat().st_size
    if size > limit:
        raise ApiError(
            ErrorCode.SOURCE_SIZE_LIMIT,
            "Source object exceeds the upload limit",
            details={"size_bytes": size, "limit": limit},
        )
    return size


def wipe_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def zip_is_safe(path: Path) -> bool:  # pragma: no cover - kept for future zip imports
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if os.path.isabs(name) or ".." in Path(name).parts:
                    return False
    except zipfile.BadZipFile:
        return False
    return True
