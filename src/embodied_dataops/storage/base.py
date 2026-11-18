"""Object storage abstraction.

Business code only ever sees ``StorageBackend``; the MinIO/S3 implementation is
the production path, while a filesystem implementation is used by unit tests
and offline demos.  Keys are immutable: raw content lives under
``raw/sha256/<sha256>`` and is never overwritten.
"""

from __future__ import annotations

import abc
import hashlib
import hmac
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlencode

from embodied_dataops.config import Settings, get_settings
from embodied_dataops.errors import ApiError, ErrorCode


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    sha256: str | None = None
    content_type: str | None = None
    last_modified: datetime | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StorageBackend(abc.ABC):
    """Minimal object-store interface used by the platform."""

    @abc.abstractmethod
    def ensure_bucket(self) -> None: ...

    @abc.abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> StoredObject: ...

    @abc.abstractmethod
    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> StoredObject: ...

    @abc.abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abc.abstractmethod
    def download_to(self, key: str, path: Path) -> StoredObject: ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool: ...

    @abc.abstractmethod
    def stat(self, key: str) -> StoredObject: ...

    @abc.abstractmethod
    def delete(self, key: str) -> None: ...

    @abc.abstractmethod
    def presign_get(self, key: str, expires_seconds: int | None = None) -> str: ...

    @abc.abstractmethod
    def presign_put(self, key: str, expires_seconds: int | None = None) -> str: ...

    @abc.abstractmethod
    def copy(self, source_key: str, destination_key: str) -> None: ...

    @abc.abstractmethod
    def sha256_of(self, key: str) -> str: ...

    @abc.abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]: ...

    def healthcheck(self) -> bool:
        try:
            self.ensure_bucket()
            return True
        except Exception:  # pragma: no cover - environment dependent
            return False


class S3Storage(StorageBackend):
    """MinIO / S3 compatible backend.

    Two clients are maintained: an internal one for byte operations (container
    DNS name) and a public-endpoint one used only for presigning, so browser
    URLs never point at the compose-internal hostname (SPEC 3.2).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        import boto3
        from botocore.client import Config

        self.settings = settings or get_settings()
        self.bucket = self.settings.s3_bucket
        common = dict(
            aws_access_key_id=self.settings.s3_access_key,
            aws_secret_access_key=self.settings.s3_secret_key,
            region_name=self.settings.s3_region,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
        )
        self._internal = boto3.client("s3", endpoint_url=self.settings.s3_endpoint_internal, **common)
        self._public = boto3.client("s3", endpoint_url=self.settings.s3_endpoint_public, **common)

    def ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self._internal.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self._internal.create_bucket(Bucket=self.bucket)
            except ClientError as exc:  # pragma: no cover - permission issues
                raise ApiError(ErrorCode.OBJECT_STORE_UNAVAILABLE, "Cannot create object bucket") from exc
        except Exception as exc:  # network/endpoint issues
            raise ApiError(ErrorCode.OBJECT_STORE_UNAVAILABLE, "Object store unavailable") from exc

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> StoredObject:
        self._internal.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return StoredObject(key=key, size=len(data), sha256=sha256_bytes(data), content_type=content_type)

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> StoredObject:
        self._internal.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type})
        return StoredObject(key=key, size=path.stat().st_size, sha256=sha256_file(path), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._internal.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                raise ApiError(ErrorCode.OBJECT_NOT_FOUND, f"Object not found: {key}") from exc
            raise

    def download_to(self, key: str, path: Path) -> StoredObject:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._internal.download_file(self.bucket, key, str(path))
        return StoredObject(key=key, size=path.stat().st_size, sha256=sha256_file(path))

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._internal.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def stat(self, key: str) -> StoredObject:
        from botocore.exceptions import ClientError

        try:
            response = self._internal.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            raise ApiError(ErrorCode.OBJECT_NOT_FOUND, f"Object not found: {key}") from exc
        return StoredObject(
            key=key,
            size=int(response.get("ContentLength", 0)),
            content_type=response.get("ContentType"),
            last_modified=response.get("LastModified"),
        )

    def delete(self, key: str) -> None:
        self._internal.delete_object(Bucket=self.bucket, Key=key)

    def presign_get(self, key: str, expires_seconds: int | None = None) -> str:
        ttl = expires_seconds or self.settings.download_url_ttl_seconds
        return self._public.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=ttl
        )

    def presign_put(self, key: str, expires_seconds: int | None = None) -> str:
        ttl = expires_seconds or self.settings.upload_url_ttl_seconds
        return self._public.generate_presigned_url(
            "put_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=ttl
        )

    def copy(self, source_key: str, destination_key: str) -> None:
        self._internal.copy_object(
            Bucket=self.bucket,
            Key=destination_key,
            CopySource={"Bucket": self.bucket, "Key": source_key},
        )

    def sha256_of(self, key: str) -> str:
        response = self._internal.get_object(Bucket=self.bucket, Key=key)
        digest = hashlib.sha256()
        for chunk in response["Body"].iter_chunks(chunk_size=1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()

    def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        paginator = self._internal.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                keys.append(item["Key"])
        return keys


class LocalStorage(StorageBackend):
    """Filesystem backend with HMAC-signed URLs served by the API.

    This backend exists so unit tests and the offline demo need no object
    store.  It is NOT a substitute for MinIO in the acceptance environment.
    """

    SIGNING_CONTEXT = "dataops-local-storage-v1"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = Path(self.settings.local_storage_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise ApiError(ErrorCode.INVALID_INPUT, "Object key escapes the storage root")
        return candidate

    def ensure_bucket(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> StoredObject:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return StoredObject(key=key, size=len(data), sha256=sha256_bytes(data), content_type=content_type)

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> StoredObject:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        return StoredObject(key=key, size=target.stat().st_size, sha256=sha256_file(target), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ApiError(ErrorCode.OBJECT_NOT_FOUND, f"Object not found: {key}")
        return path.read_bytes()

    def download_to(self, key: str, path: Path) -> StoredObject:
        data = self.get_bytes(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return StoredObject(key=key, size=len(data), sha256=sha256_bytes(data))

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def stat(self, key: str) -> StoredObject:
        path = self._path(key)
        if not path.is_file():
            raise ApiError(ErrorCode.OBJECT_NOT_FOUND, f"Object not found: {key}")
        return StoredObject(
            key=key,
            size=path.stat().st_size,
            sha256=sha256_file(path),
            last_modified=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        )

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink()

    def _sign(self, key: str, expires_at: int, method: str) -> str:
        message = f"{method}:{key}:{expires_at}".encode()
        return hmac.new(self.SIGNING_CONTEXT.encode(), message, hashlib.sha256).hexdigest()

    def presign_get(self, key: str, expires_seconds: int | None = None) -> str:
        ttl = expires_seconds or self.settings.download_url_ttl_seconds
        expires_at = int(time.time()) + ttl
        query = urlencode({"exp": expires_at, "sig": self._sign(key, expires_at, "GET")})
        return f"{self.settings.api_public_base_url.rstrip('/')}/api/v1/_storage/get/{quote(key)}?{query}"

    def presign_put(self, key: str, expires_seconds: int | None = None) -> str:
        ttl = expires_seconds or self.settings.upload_url_ttl_seconds
        expires_at = int(time.time()) + ttl
        query = urlencode({"exp": expires_at, "sig": self._sign(key, expires_at, "PUT")})
        return f"{self.settings.api_public_base_url.rstrip('/')}/api/v1/_storage/put/{quote(key)}?{query}"

    def verify(self, key: str, expires_at: int, signature: str, method: str) -> None:
        if expires_at < int(time.time()):
            raise ApiError(ErrorCode.NOT_FOUND, "Signed URL expired", status_code=403)
        if not hmac.compare_digest(self._sign(key, expires_at, method), signature):
            raise ApiError(ErrorCode.NOT_FOUND, "Invalid signature", status_code=403)

    def copy(self, source_key: str, destination_key: str) -> None:
        source = self._path(source_key)
        if not source.is_file():
            raise ApiError(ErrorCode.OBJECT_NOT_FOUND, f"Object not found: {source_key}")
        target = self._path(destination_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    def sha256_of(self, key: str) -> str:
        path = self._path(key)
        if not path.is_file():
            raise ApiError(ErrorCode.OBJECT_NOT_FOUND, f"Object not found: {key}")
        return sha256_file(path)

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if base.is_file():
            return [str(base.relative_to(self.root)).replace("\\", "/")]
        if not base.exists():
            return []
        keys: list[str] = []
        for path in base.rglob("*"):
            if path.is_file():
                keys.append(str(path.relative_to(self.root)).replace("\\", "/"))
        return sorted(keys)


_backend: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _backend
    if _backend is None:
        settings = get_settings()
        _backend = LocalStorage(settings) if settings.storage_backend == "local" else S3Storage(settings)
    return _backend


def set_storage_for_tests(backend: StorageBackend | None) -> None:
    global _backend
    _backend = backend


def reset_storage() -> None:
    global _backend
    _backend = None
