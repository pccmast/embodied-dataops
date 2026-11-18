"""API contract: upload initiation, server-side hash verification, idempotency.

Covers the API half of AT-08 and the security envelope (401/403/415/413).
"""

from __future__ import annotations

import hashlib
import uuid
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.integration


def _payload(data: bytes, **overrides):
    base = {
        "source_type": "EGO_MP4",
        "filename": "demo.mp4",
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "data_origin": "synthetic",
        "license": "",
    }
    base.update(overrides)
    return base


def _upload(client, put_url: str, data: bytes) -> None:
    parsed = urlparse(put_url)
    response = client.put(parsed.path + ("?" + parsed.query if parsed.query else ""), content=data)
    assert response.status_code == 200, response.text


def test_import_upload_flow_and_idempotent_complete(client, operator_headers):
    data = b"fake-mp4-bytes-for-contract-test"
    initiated = client.post(
        "/api/v1/imports/initiate", json=_payload(data), headers=operator_headers
    )
    assert initiated.status_code == 201, initiated.text
    body = initiated.json()["data"]
    assert body["import_id"]
    assert body["put_url"].startswith("http://testserver/api/v1/_storage/put/")
    _upload(client, body["put_url"], data)

    key = str(uuid.uuid4())
    complete_headers = {**operator_headers, "Idempotency-Key": key}
    first = client.post(f"/api/v1/imports/{body['import_id']}/complete", headers=complete_headers)
    assert first.status_code == 202, first.text
    job_id = first.json()["data"]["import_job_id"]
    assert first.json()["data"]["state"] == "QUEUED"

    for _ in range(2):
        repeat = client.post(f"/api/v1/imports/{body['import_id']}/complete", headers=complete_headers)
        assert repeat.status_code == 202
        assert repeat.json()["data"]["import_job_id"] == job_id

    detail = client.get(f"/api/v1/imports/{body['import_id']}", headers=operator_headers)
    assert detail.status_code == 200
    detail_body = detail.json()["data"]
    assert detail_body["import"]["upload_state"] == "UPLOADED"
    assert detail_body["job"]["id"] == job_id
    assert detail_body["episodes_summary"]["count"] == 0


def test_hash_mismatch_is_rejected(client, operator_headers):
    data = b"some-real-bytes"
    payload = _payload(data)
    payload["sha256"] = "0" * 64  # declared hash does not match the uploaded bytes
    initiated = client.post("/api/v1/imports/initiate", json=payload, headers=operator_headers)
    assert initiated.status_code == 201
    _upload(client, initiated.json()["data"]["put_url"], data)
    response = client.post(
        f"/api/v1/imports/{initiated.json()['data']['import_id']}/complete",
        headers={**operator_headers, "Idempotency-Key": str(uuid.uuid4())},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SOURCE_HASH_MISMATCH"


def test_same_key_different_body_conflicts(client, operator_headers):
    payload = _payload(b"aaaa")
    initiated = client.post("/api/v1/imports/initiate", json=payload, headers=operator_headers)
    _upload(client, initiated.json()["data"]["put_url"], b"aaaa")
    import_id = initiated.json()["data"]["import_id"]
    key = str(uuid.uuid4())
    assert (
        client.post(
            f"/api/v1/imports/{import_id}/complete",
            headers={**operator_headers, "Idempotency-Key": key},
        ).status_code
        == 202
    )
    # Same idempotency key, different request body -> 409.
    conflict = client.post(
        "/api/v1/imports/initiate",
        json=_payload(b"bbbb"),
        headers=operator_headers,
    )
    _upload(client, conflict.json()["data"]["put_url"], b"bbbb")
    reused = client.post(
        f"/api/v1/imports/{conflict.json()['data']['import_id']}/complete",
        headers={**operator_headers, "Idempotency-Key": key},
    )
    assert reused.status_code == 409
    assert reused.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"


def test_rbac_and_validation(client, viewer_headers, operator_headers):
    data = b"x"
    # viewer cannot initiate imports
    forbidden = client.post("/api/v1/imports/initiate", json=_payload(data), headers=viewer_headers)
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "FORBIDDEN"

    # unauthenticated
    unauthorized = client.get("/api/v1/episodes")
    assert unauthorized.status_code == 401

    # wrong file extension for the declared source type
    bad_ext = client.post(
        "/api/v1/imports/initiate",
        json=_payload(data, filename="demo.mov"),
        headers=operator_headers,
    )
    assert bad_ext.status_code == 415

    # oversized declared upload
    too_big = client.post(
        "/api/v1/imports/initiate",
        json=_payload(data, size_bytes=3 * 1024**3),
        headers=operator_headers,
    )
    assert too_big.status_code == 413

    # unknown mapping field
    bad_mapping = client.post(
        "/api/v1/imports/initiate",
        json=_payload(data, mapping={"nope": "x"}),
        headers=operator_headers,
    )
    assert bad_mapping.status_code == 422


def test_health_endpoints(client):
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code in (200, 503)
    body = ready.json()
    assert "components" in body
