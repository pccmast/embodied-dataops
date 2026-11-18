"""Page 1 — Import & Episode (SPEC 11.1).

Upload flow start/finish, import status, episode filtering, preview and the
canonical state/action sample chart with timestamp-anomaly findings.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import streamlit as st

from embodied_ui.api_client import ApiClient, ApiError
from embodied_ui.pages._shared import (
    dataframe_rows,
    render_empty,
    render_error,
    sample_chart,
    state_get,
    state_set,
)

SOURCE_TYPES = ("LEROBOT_V3", "EGO_MP4", "MCAP_ROS2_PROFILE_V1")
DATA_ORIGINS = ("synthetic", "public_dataset", "robot_capture", "ego_capture")


def _render_upload(client: ApiClient) -> None:
    st.subheader("Start an upload")
    st.caption(
        "The API never proxies the bytes: it returns a short-lived presigned PUT URL, "
        "then `complete` re-verifies size and SHA-256 server-side."
    )
    with st.form("upload_form"):
        source_type = st.selectbox("Source type", SOURCE_TYPES)
        data_origin = st.selectbox("Data origin", DATA_ORIGINS, index=0)
        license_value = st.text_input("License (required for later publishing)", value="synthetic-fixture")
        source_uri = st.text_input("Source URI (optional)")
        source_revision = st.text_input("Source revision (optional)")
        uploaded = st.file_uploader("Source archive / video", type=["tgz", "gz", "tar", "mp4", "mcap"])
        mapping: dict[str, Any] = {}
        if source_type == "LEROBOT_V3":
            st.markdown("**LeRobot mapping** (required for LEROBOT_V3)")
            camera_key = st.text_input("camera_key", value="observation.images.front")
            state_key = st.text_input("state_key", value="observation.state")
            action_key = st.text_input("action_key", value="action")
            robot_profile = st.text_input("robot_profile", value="arm6_joint_position_v1")
            mapping = {
                "camera_key": camera_key,
                "state_key": state_key,
                "action_key": action_key,
                "robot_profile": robot_profile,
            }
        submitted = st.form_submit_button("Initiate upload", use_container_width=True)

    if not submitted:
        if state_get("last_import"):
            st.success(f"Last import: {state_get('last_import')}")
        return
    if uploaded is None:
        render_empty("Choose a file first — no upload session was created.")
        return

    content = uploaded.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    body: dict[str, Any] = {
        "source_type": source_type,
        "filename": uploaded.name,
        "size_bytes": len(content),
        "sha256": digest,
        "data_origin": data_origin,
        "license": license_value,
    }
    if source_uri:
        body["source_uri"] = source_uri
    if source_revision:
        body["source_revision"] = source_revision
    if mapping:
        body["mapping"] = mapping

    try:
        with st.spinner("Creating upload session…"):
            initiated = client.post("/imports/initiate", json=body)["data"]
            put_url = initiated["put_url"]
            import_id = initiated["import_id"]
            client.put_bytes(put_url, content, content_type="application/octet-stream")
            completed = client.post(
                f"/imports/{import_id}/complete",
                json={},
                headers={"Idempotency-Key": str(uuid.uuid4())},
            )["data"]
        state_set("last_import", import_id)
        st.success(f"Upload accepted: import {import_id}, job state {completed.get('state')}")
        st.json({"initiate": initiated, "complete": completed})
    except ApiError as exc:
        render_error(exc, context="Upload")


def _render_import_status(client: ApiClient) -> None:
    st.subheader("Import status")
    import_id = st.text_input("Import id", value=state_get("last_import", ""))
    if not st.button("Load import", key="load_import"):
        return
    if not import_id:
        render_empty("Enter an import id (returned by the upload flow).")
        return
    try:
        payload = client.get(f"/imports/{import_id}")["data"]
    except ApiError as exc:
        render_error(exc, context="Import status")
        return
    st.json(payload)


def _render_episode_browser(client: ApiClient) -> None:
    st.subheader("Episodes")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        source_type = st.selectbox("Filter source_type", ("",) + SOURCE_TYPES, key="ep_source")
    with col2:
        quality_status = st.selectbox("Filter quality_status", ("", "PASS", "WARN", "FAIL"), key="ep_qc")
    with col3:
        task_label = st.text_input("Filter task_label", key="ep_task")
    with col4:
        limit = st.number_input("limit", min_value=1, max_value=100, value=20, key="ep_limit")

    params: dict[str, Any] = {"limit": int(limit)}
    if source_type:
        params["source_type"] = source_type
    if quality_status:
        params["quality_status"] = quality_status
    if task_label:
        params["task_label"] = task_label
    try:
        payload = client.get("/episodes", params=params)
    except ApiError as exc:
        render_error(exc, context="Episode list")
        return
    episodes = payload.get("data") or []
    dataframe_rows(
        episodes,
        empty_message="No episodes match the filters. Process an import in the Pipeline Board first.",
    )
    if not episodes:
        return
    options = {f"{item.get('id')} · {item.get('task_label') or 'no task'} · {item.get('status')}": item.get("id") for item in episodes}
    selected = st.selectbox("Inspect episode", list(options), key="ep_selected")
    episode_id = options[selected]
    _render_episode_detail(client, episode_id)


def _render_episode_detail(client: ApiClient, episode_id: str) -> None:
    try:
        detail = client.get(f"/episodes/{episode_id}")["data"]
    except ApiError as exc:
        render_error(exc, context="Episode detail")
        return
    st.markdown(f"**Episode `{episode_id}`**")
    st.json(detail)

    preview_tab, samples_tab, quality_tab = st.tabs(["Preview", "State / action", "Quality findings"])
    with preview_tab:
        try:
            preview = client.get(f"/episodes/{episode_id}/preview")["data"]
        except ApiError as exc:
            render_error(exc, context="Preview")
        else:
            video_url = preview.get("video_url")
            thumbnail_url = preview.get("thumbnail_url")
            if thumbnail_url:
                try:
                    st.image(client.fetch_bytes(thumbnail_url), caption="thumbnail (presigned URL)")
                except ApiError as exc:
                    render_error(exc, context="Thumbnail")
            if video_url:
                st.video(video_url)
                st.caption(f"Presigned URL expires at {preview.get('expires_at')}")
            else:
                render_empty("The API returned no preview URL for this episode.")

    with samples_tab:
        try:
            samples_payload = client.get(f"/episodes/{episode_id}/samples", params={"limit": 100})["data"]
        except ApiError as exc:
            render_error(exc, context="Samples")
        else:
            if isinstance(samples_payload, list):
                samples = samples_payload
            elif isinstance(samples_payload, dict):
                samples = samples_payload.get("samples") or []
            else:
                samples = []
            sample_chart(samples)

    with quality_tab:
        try:
            quality = client.get(f"/episodes/{episode_id}/quality")["data"]
        except ApiError as exc:
            render_error(exc, context="Quality")
        else:
            st.metric("verdict", str(quality.get("verdict")))
            findings = quality.get("findings") or []
            anomaly_findings = [
                finding
                for finding in findings
                if finding.get("rule_code") in {"TIMESTAMP_NON_MONOTONIC", "FRAME_GAP_EXCESS"}
            ]
            if anomaly_findings:
                st.warning(f"{len(anomaly_findings)} timestamp/video anomaly finding(s) — see evidence below.")
            dataframe_rows(findings, empty_message="No quality findings were returned for this episode.")
            if quality.get("metrics"):
                st.json({"metrics": quality["metrics"]})
            if quality.get("report_url"):
                st.link_button("Open quality report", quality["report_url"])


def render(client: ApiClient) -> None:
    st.title("Import & Episode")
    st.caption("All values on this page are live API responses; nothing is fabricated client-side.")
    if not client.is_configured:
        st.error("API_KEY is not set. The UI cannot authenticate.")
        st.info("Set the API_KEY environment variable to a `dataops_...` key and restart Streamlit.")
        return
    tabs = st.tabs(["Upload", "Import status", "Episodes"])
    with tabs[0]:
        _render_upload(client)
    with tabs[1]:
        _render_import_status(client)
    with tabs[2]:
        _render_episode_browser(client)
