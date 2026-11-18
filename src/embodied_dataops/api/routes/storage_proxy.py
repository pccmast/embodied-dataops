"""Local-storage signed URL endpoints.

Only mounted meaningfully when STORAGE_BACKEND=local (tests / offline demo);
with MinIO the API returns 404 because presigned URLs come from MinIO.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from embodied_dataops.config import get_settings
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.storage.base import LocalStorage, get_storage

router = APIRouter(prefix="/api/v1/_storage", tags=["storage"])


def _local_storage() -> LocalStorage:
    storage = get_storage()
    if not isinstance(storage, LocalStorage):
        raise ApiError(ErrorCode.NOT_FOUND, "Signed URL proxy is not enabled for this storage backend")
    return storage


@router.put("/put/{key:path}")
async def signed_put(key: str, request: Request, exp: int, sig: str) -> Response:
    storage = _local_storage()
    storage.verify(key, exp, sig, "PUT")
    body = await request.body()
    settings = get_settings()
    if len(body) > settings.max_upload_bytes:
        raise ApiError(
            ErrorCode.PAYLOAD_TOO_LARGE,
            "Uploaded body exceeds the configured limit",
            details={"limit": settings.max_upload_bytes},
        )
    storage.put_bytes(key, body, request.headers.get("content-type", "application/octet-stream"))
    return Response(status_code=200)


@router.get("/get/{key:path}")
async def signed_get(key: str, exp: int, sig: str) -> Response:
    storage = _local_storage()
    storage.verify(key, exp, sig, "GET")
    data = storage.get_bytes(key)
    return Response(content=data, media_type="application/octet-stream")
