"""Unit tests for the Streamlit UI HTTP client error handling (SPEC 11).

The UI must never show fabricated data: every failure path becomes an explicit
``ApiError`` with an HTTP status/code and a recovery hint.  These tests use
``httpx.MockTransport`` so no server is required.
"""

from __future__ import annotations

import httpx
import pytest

from embodied_ui.api_client import ApiClient, ApiError


def _client(handler, *, api_key: str = "dataops_test") -> ApiClient:
    return ApiClient(
        "http://api.test",
        api_key,
        transport=httpx.MockTransport(handler),
    )


def test_success_envelope_is_returned_unchanged() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization", "")
        seen["request_id"] = request.headers.get("x-request-id", "")
        return httpx.Response(200, json={"data": {"id": "ep-1"}, "request_id": "r-1"})

    client = _client(handler)
    payload = client.get("/episodes/ep-1")
    assert payload["data"] == {"id": "ep-1"}
    assert seen["path"] == "/api/v1/episodes/ep-1"
    assert seen["auth"] == "Bearer dataops_test"
    assert seen["request_id"]


def test_entity_404_uses_spec_error_envelope_and_is_not_endpoint_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {"code": "NOT_FOUND", "message": "Episode not found", "details": {}},
                "request_id": "r-1",
            },
        )

    with pytest.raises(ApiError) as excinfo:
        _client(handler).get("/episodes/missing")
    error = excinfo.value
    assert error.code == "NOT_FOUND"
    assert error.status_code == 404
    assert error.endpoint_missing is False
    assert "does not exist" in error.recovery_hint


def test_missing_route_is_reported_as_endpoint_not_available() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    with pytest.raises(ApiError) as excinfo:
        _client(handler).get("/collection-orders")
    error = excinfo.value
    assert error.endpoint_missing is True
    assert error.code == "ENDPOINT_NOT_AVAILABLE"
    assert "not available yet" in error.recovery_hint


def test_401_hint_mentions_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {"code": "UNAUTHENTICATED", "message": "Invalid API key", "details": {}},
                "request_id": "r-1",
            },
        )

    with pytest.raises(ApiError) as excinfo:
        _client(handler).get("/episodes")
    assert excinfo.value.code == "UNAUTHENTICATED"
    assert "API_KEY" in excinfo.value.recovery_hint


def test_403_hint_mentions_permission() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"code": "FORBIDDEN", "message": "Role lacks permission", "details": {}}},
        )

    with pytest.raises(ApiError) as excinfo:
        _client(handler).post("/dataset-versions/v1/publish", json={})
    assert excinfo.value.status_code == 403
    assert "permission" in excinfo.value.recovery_hint.lower()


def test_network_failure_becomes_api_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ApiError) as excinfo:
        _client(handler).get("/episodes")
    error = excinfo.value
    assert error.code == "API_UNREACHABLE"
    assert error.status_code == 0
    assert "api" in error.recovery_hint.lower()


def test_non_json_error_body_still_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="upstream bad gateway")

    with pytest.raises(ApiError) as excinfo:
        _client(handler).get("/episodes")
    assert excinfo.value.status_code == 502
    assert "server error" in excinfo.value.recovery_hint.lower()


def test_items_helper_extracts_list_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": 1}, {"id": 2}], "next_cursor": None})

    assert _client(handler).items("/episodes") == [{"id": 1}, {"id": 2}]


def test_items_helper_returns_empty_list_for_non_list_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"unexpected": True}})

    assert _client(handler).items("/episodes") == []


def test_from_env_selects_first_configured_key_and_base_url() -> None:
    client = ApiClient.from_env(
        {
            "API_KEY": "dataops_env_key",
            "UI_API_BASE_URL": "http://api.local:9000/",
        }
    )
    assert client.api_key == "dataops_env_key"
    assert client.base_url == "http://api.local:9000"
    assert client.is_configured is True
    client.close()


def test_from_env_falls_back_to_api_public_base_url() -> None:
    client = ApiClient.from_env({"API_KEY": "k", "API_PUBLIC_BASE_URL": "http://public.test"})
    assert client.base_url == "http://public.test"
    client.close()


def test_from_env_without_key_is_not_configured() -> None:
    client = ApiClient.from_env({"UI_API_BASE_URL": "http://api.test"})
    assert client.is_configured is False
    client.close()


def test_put_bytes_uses_presigned_url_without_api_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_put(url: str, content: bytes, headers, timeout):
        calls.append({"url": url, "content": content, "headers": headers})
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "put", fake_put)
    client = _client(lambda request: httpx.Response(500))
    client.put_bytes("http://objects.test/upload?sig=x", b"payload", content_type="application/gzip")
    assert calls[0]["content"] == b"payload"
    assert calls[0]["headers"]["Content-Type"] == "application/gzip"
    assert "Authorization" not in calls[0]["headers"]
    client.close()


def test_put_bytes_failure_raises_upload_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "put", lambda *args, **kwargs: httpx.Response(403))
    client = _client(lambda request: httpx.Response(500))
    with pytest.raises(ApiError) as excinfo:
        client.put_bytes("http://objects.test/upload?sig=x", b"payload")
    assert excinfo.value.code == "UPLOAD_FAILED"
    assert excinfo.value.status_code == 403
    client.close()


# ---------------------------------------------------------------------------
# Page render smoke tests (AT-23): each page renders a real error/empty state
# with a recovery hint instead of a traceback, and never invents data.
# ---------------------------------------------------------------------------
streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

PAGE_MODULES = {
    "import_episode": "Import & Episode",
    "pipeline_board": "Pipeline Board",
    "quality_review": "Quality & Annotation Review",
    "dataset_feedback": "Dataset & Feedback",
}


def _run_page_script(script_body: str) -> AppTest:
    at = AppTest.from_string(script_body, default_timeout=120)
    at.run()
    return at


@pytest.mark.parametrize("module_name,title", list(PAGE_MODULES.items()))
def test_page_without_api_key_renders_recovery_hint(module_name: str, title: str) -> None:
    at = _run_page_script(
        "from embodied_ui.api_client import ApiClient\n"
        f"from embodied_ui.pages import {module_name}\n"
        'client = ApiClient("http://api.test", "")\n'
        f"{module_name}.render(client)\n"
    )
    assert not at.exception, [str(exc.value) for exc in at.exception]
    assert [item.value for item in at.title] == [title]
    assert at.error, "a page without API_KEY must show an explicit error state"
    assert any("API_KEY" in info.value for info in at.info)


def test_page_reports_endpoint_not_available_for_missing_route() -> None:
    at = _run_page_script(
        "import httpx\n"
        "from embodied_ui.api_client import ApiClient\n"
        "from embodied_ui.pages import quality_review\n"
        'transport = httpx.MockTransport(lambda request: httpx.Response(404, json={"detail": "Not Found"}))\n'
        'client = ApiClient("http://api.test", "dataops_test", transport=transport)\n'
        "quality_review.render(client)\n"
    )
    assert not at.exception, [str(exc.value) for exc in at.exception]
    rendered_info = " ".join(item.value for item in at.info)
    assert "not available yet" in rendered_info
    rendered_caption = " ".join(item.value for item in at.caption)
    assert "not available yet" in rendered_caption


def test_page_reports_authentication_failure_with_hint() -> None:
    at = _run_page_script(
        "import httpx\n"
        "from embodied_ui.api_client import ApiClient\n"
        "from embodied_ui.pages import pipeline_board\n"
        "def handler(request):\n"
        "    return httpx.Response(401, json={'error': {'code': 'UNAUTHENTICATED', 'message': 'bad key', 'details': {}}})\n"
        'client = ApiClient("http://api.test", "dataops_bad", transport=httpx.MockTransport(handler))\n'
        "pipeline_board.render(client)\n"
    )
    assert not at.exception, [str(exc.value) for exc in at.exception]
    rendered_errors = " ".join(item.value for item in at.error)
    assert "UNAUTHENTICATED" in rendered_errors
    rendered_caption = " ".join(item.value for item in at.caption)
    assert "API_KEY" in rendered_caption
