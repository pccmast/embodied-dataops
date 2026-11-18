"""HTTP client used by every Streamlit page.

The UI is a pure API consumer (SPEC 11): it holds an API key, calls the REST API
and never opens a database connection.  Errors are normalised into
:class:`ApiError` so pages can render an explicit error state *and* a recovery
hint instead of a traceback.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_TIMEOUT_SECONDS = 30.0
ENV_API_KEY_NAMES = ("API_KEY", "UI_API_KEY", "DATAOPS_API_KEY")
ENV_BASE_URL_NAMES = ("UI_API_BASE_URL", "API_PUBLIC_BASE_URL", "API_BASE_URL")
API_PREFIX = "/api/v1"


@dataclass
class ApiError(Exception):
    """Uniform API/transport failure rendered by the UI."""

    code: str
    message: str
    status_code: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    endpoint_missing: bool = False

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def recovery_hint(self) -> str:
        """A concrete next step for the operator (SPEC 11 error recovery)."""
        if self.endpoint_missing:
            return (
                "This API endpoint is not available yet in the running backend. "
                "Start the current API build (or check UI_API_BASE_URL) and reload the page."
            )
        if self.code == "API_UNREACHABLE":
            return (
                "The API could not be reached. Check that the `api` service is running and that "
                "UI_API_BASE_URL points at it, then use the Retry button."
            )
        if self.status_code == 401:
            return (
                "Authentication failed. Set API_KEY to a valid `dataops_...` key (printed once by "
                "`make bootstrap`) and restart Streamlit."
            )
        if self.status_code == 403:
            return "This role lacks the required permission; ask an admin for the right API key."
        if self.status_code == 404:
            return "The entity does not exist (it may have been deleted or the id is wrong). Refresh the list."
        if self.status_code == 409:
            return "The object is in a conflicting state; reload it and check the version state."
        if self.status_code == 412 or self.code == "PRECONDITION_FAILED":
            return (
                "The resource changed since this page loaded it (stale If-Match). "
                f"Reload it and retry; current revision: {self.details.get('current_business_revision', 'unknown')}."
            )
        if self.status_code == 428 or self.code == "PRECONDITION_REQUIRED":
            return (
                "This is a conditional write and needs a strong If-Match value. "
                "Reload the resource so the page can read its current ETag, then retry."
            )
        if self.status_code == 503:
            return (
                "A dependency is temporarily unavailable or the same idempotency key is still "
                "committing; retry shortly with the same Idempotency-Key."
            )
        if self.status_code >= 500:
            return "The API reported a server error; retry shortly and check the API logs."
        return "Check the input values and retry."

    def as_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "endpoint_missing": self.endpoint_missing,
        }


class ApiClient:
    """Small synchronous client around the versioned REST API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    # -- construction --------------------------------------------------
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ApiClient:
        env = dict(os.environ if environ is None else environ)
        api_key = ""
        for name in ENV_API_KEY_NAMES:
            value = (env.get(name) or "").strip()
            if value:
                api_key = value
                break
        base_url = ""
        for name in ENV_BASE_URL_NAMES:
            value = (env.get(name) or "").strip()
            if value:
                base_url = value
                break
        return cls(base_url or "http://localhost:8000", api_key)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- core request ---------------------------------------------------
    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Request-ID": str(uuid.uuid4()),
        }
        if extra:
            headers.update(extra)
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
        with_headers: bool = False,
    ) -> Any:
        """Perform one request and return the decoded JSON body.

        With ``with_headers=True`` the return value is ``(body, headers)`` so
        pages can read the strong ``ETag`` required by DO-A02 conditional writes.
        The SPEC success envelope ``{"data": ..., "request_id": ...}`` is returned
        unchanged (pages read ``payload["data"]``).  A missing route (FastAPI
        default 404 without the SPEC error envelope) is flagged separately from a
        missing entity so the pages can say "not available yet" honestly.
        """
        url = path if path.startswith("http") else f"{API_PREFIX}{path}"
        try:
            response = self._client.request(
                method,
                url,
                params=dict(params or {}) or None,
                json=json,
                headers=self._headers(headers),
            )
        except httpx.HTTPError as exc:
            raise ApiError(
                code="API_UNREACHABLE",
                message=f"{method} {url} failed: {type(exc).__name__}: {exc}",
                status_code=0,
            ) from exc

        body: dict[str, Any]
        try:
            decoded = response.json()
            body = decoded if isinstance(decoded, dict) else {"data": decoded}
        except ValueError:
            body = {}

        if response.status_code >= 400:
            endpoint_missing = response.status_code == 404 and "error" not in body
            error = body.get("error") if isinstance(body.get("error"), dict) else {}
            code = str(error.get("code") or ("NOT_FOUND" if response.status_code == 404 else "HTTP_ERROR"))
            if endpoint_missing:
                code = "ENDPOINT_NOT_AVAILABLE"
            message = str(
                error.get("message")
                or body.get("detail")
                or f"{method} {url} returned HTTP {response.status_code}"
            )
            details = error.get("details") if isinstance(error.get("details"), dict) else {}
            raise ApiError(
                code=code,
                message=message,
                status_code=response.status_code,
                details=dict(details),
                endpoint_missing=endpoint_missing,
            )
        if with_headers:
            return body, {str(key): str(value) for key, value in response.headers.items()}
        return body

    def get_with_headers(
        self, path: str, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """GET returning ``(body, response_headers)`` for ETag-based writes."""
        return self.request("GET", path, with_headers=True, **kwargs)

    def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request("DELETE", path, **kwargs)

    # -- convenience ----------------------------------------------------
    def data(self, path: str, **kwargs: Any) -> Any:
        return self.get(path, **kwargs).get("data")

    def items(self, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        payload = self.get(path, **kwargs)
        value = payload.get("data")
        return list(value) if isinstance(value, list) else []

    def put_bytes(self, url: str, content: bytes, *, content_type: str = "application/octet-stream") -> None:
        """PUT bytes to a presigned upload URL (no API auth header is added)."""
        try:
            response = httpx.put(
                url,
                content=content,
                headers={"Content-Type": content_type},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ApiError(
                code="UPLOAD_UNREACHABLE",
                message=f"Upload URL is not reachable: {type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code >= 400:
            raise ApiError(
                code="UPLOAD_FAILED",
                message=f"Presigned upload failed with HTTP {response.status_code}",
                status_code=response.status_code,
            )

    def health(self) -> dict[str, Any]:
        try:
            response = self._client.get("/health/ready")
        except httpx.HTTPError as exc:
            raise ApiError(
                code="API_UNREACHABLE",
                message=f"Health check failed: {type(exc).__name__}: {exc}",
            ) from exc
        return {"status_code": response.status_code}

    def fetch_bytes(self, url: str) -> bytes:
        """Download an object via its (presigned) URL for preview rendering."""
        try:
            response = httpx.get(url, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ApiError(
                code="PREVIEW_UNREACHABLE",
                message=f"Preview URL is not reachable: {type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code >= 400:
            raise ApiError(
                code="PREVIEW_FAILED",
                message=f"Preview download failed with HTTP {response.status_code}",
                status_code=response.status_code,
            )
        return response.content
