"""Embodied DataOps — Streamlit admin UI (SPEC 11).

Run with::

    uv run streamlit run src/embodied_ui/app.py

The four pages are: Import & Episode, Pipeline Board, Quality & Annotation
Review, Dataset & Feedback.  This process only speaks HTTP to the DataOps API
(``API_KEY`` + ``UI_API_BASE_URL`` / ``API_PUBLIC_BASE_URL``); it never connects
to PostgreSQL, Redis or MinIO directly.
"""

from __future__ import annotations

import streamlit as st

from embodied_ui.api_client import ApiClient, ApiError
from embodied_ui.pages import dataset_feedback, import_episode, pipeline_board, quality_review

PAGES = (
    ("Import & Episode", import_episode.render),
    ("Pipeline Board", pipeline_board.render),
    ("Quality & Annotation Review", quality_review.render),
    ("Dataset & Feedback", dataset_feedback.render),
)


def _sidebar(client: ApiClient) -> None:
    with st.sidebar:
        st.header("DataOps UI")
        st.caption(f"API base URL: `{client.base_url}`")
        st.caption("API key configured" if client.is_configured else "API key NOT configured")
        if not client.is_configured:
            st.error("Set API_KEY to a `dataops_...` key and restart Streamlit.")
        if st.button("Check API health"):
            try:
                result = client.health()
            except ApiError as exc:
                st.error(exc.message)
                st.caption(f"How to recover: {exc.recovery_hint}")
            else:
                if result.get("status_code") == 200:
                    st.success("API /health/ready → 200")
                else:
                    st.warning(f"API /health/ready → {result.get('status_code')}")
        st.divider()
        st.caption(
            "All data shown here is returned by the versioned HTTP API. "
            "The backend enforces RBAC; hiding a button is not authorization."
        )


def _render(client: ApiClient) -> None:
    renderers = dict(PAGES)
    if hasattr(st, "navigation") and hasattr(st, "Page"):
        navigation = st.navigation(
            [
                st.Page(renderer, title=title, url_path=title.lower().replace(" & ", "-").replace(" ", "-"))
                for title, renderer in PAGES
            ]
        )
        navigation.run()
        return
    # Fallback for Streamlit versions without st.navigation.
    selection = st.sidebar.radio("Page", [title for title, _ in PAGES])
    renderers[selection](client)


def main() -> None:
    st.set_page_config(page_title="Embodied DataOps", page_icon="🤖", layout="wide")
    client = ApiClient.from_env()
    try:
        _sidebar(client)
        _render(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
