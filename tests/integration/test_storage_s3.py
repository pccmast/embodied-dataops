"""S3-compatible storage backend integration (moto server).

The compose stack uses MinIO; this test exercises the same boto3 S3 code path
(presigned GET/PUT, server-side copy, streaming SHA-256, listing, metadata)
against an in-process S3-compatible server.  It is storage-backend evidence, not
a substitute for the MinIO compose smoke test (recorded in test-report.md).
"""

from __future__ import annotations

import hashlib

import pytest

from embodied_dataops.config import Settings
from embodied_dataops.storage.base import S3Storage

pytestmark = [pytest.mark.integration]

moto = pytest.importorskip("moto", reason="moto is an optional test dependency")


@pytest.fixture(scope="module")
def moto_endpoint():
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture()
def s3_storage(moto_endpoint: str) -> S3Storage:
    settings = Settings(
        storage_backend="s3",
        s3_bucket="dataops-test",
        s3_endpoint_internal=moto_endpoint,
        s3_endpoint_public=moto_endpoint,
        s3_access_key="testing",
        s3_secret_key="testing",
        s3_region="us-east-1",
    )
    storage = S3Storage(settings)
    storage.ensure_bucket()
    return storage


def test_put_get_presign_copy_and_list(s3_storage: S3Storage):
    data = b"hello-embodied-dataops"
    stored = s3_storage.put_bytes("raw/sha256/abc", data, content_type="text/plain")
    assert stored.size == len(data)
    assert stored.sha256 == hashlib.sha256(data).hexdigest()

    assert s3_storage.get_bytes("raw/sha256/abc") == data
    assert s3_storage.exists("raw/sha256/abc")
    stat = s3_storage.stat("raw/sha256/abc")
    assert stat.size == len(data)
    assert stat.last_modified is not None
    assert s3_storage.sha256_of("raw/sha256/abc") == hashlib.sha256(data).hexdigest()

    presigned_get = s3_storage.presign_get("raw/sha256/abc", 60)
    presigned_put = s3_storage.presign_put("uploads/x/file.bin", 60)
    assert "X-Amz-Signature" in presigned_get and presigned_get.startswith("http")
    assert "X-Amz-Signature" in presigned_put

    s3_storage.copy("raw/sha256/abc", "canonical/ep/v1/metadata.json")
    assert s3_storage.get_bytes("canonical/ep/v1/metadata.json") == data
    keys = s3_storage.list_keys("canonical/")
    assert "canonical/ep/v1/metadata.json" in keys

    s3_storage.delete("canonical/ep/v1/metadata.json")
    assert not s3_storage.exists("canonical/ep/v1/metadata.json")


def test_presigned_put_accepts_upload(s3_storage: S3Storage):
    import urllib.request

    url = s3_storage.presign_put("uploads/presigned.bin", 60)
    body = b"presigned-body"
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={"Content-Length": str(len(body)), "Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - moto loopback
        assert response.status == 200
    assert s3_storage.get_bytes("uploads/presigned.bin") == body
