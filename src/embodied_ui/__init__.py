"""Streamlit UI package (SPEC 11): HTTP API client and four pages.

The UI never talks to PostgreSQL/Redis/MinIO directly.  Every value shown on a
page comes from an HTTP response of the DataOps API.
"""

from __future__ import annotations

__all__ = ["api_client", "app"]
