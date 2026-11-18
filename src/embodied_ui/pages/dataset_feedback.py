"""Page 4 — Dataset & Feedback (SPEC 11.4, V2 SPEC §7).

Create dataset versions, publish READY versions, download the frozen manifest /
LeRobot package, import ``evaluation/1.0``, run the conditional gap analysis and
track the V2 collection-order lifecycle (PROPOSED → APPROVED → IN_PROGRESS →
FULFILLED/CANCELLED), scenario coverage and simulated-evaluation labels.

The simulated-evaluation warning is always rendered when the API reports
``is_simulated`` / ``is_simulated_source``; it is never hidden client-side.
"""

from __future__ import annotations

import json
import uuid

import streamlit as st

from embodied_ui.api_client import ApiClient, ApiError
from embodied_ui.pages._shared import (
    dataframe_rows,
    render_empty,
    render_error,
    simulated_banner,
    state_get,
    state_set,
)


def _idempotency_key() -> str:
    return str(uuid.uuid4())


def _etag(headers: dict[str, str], data: dict | None, kind: str) -> str | None:
    for candidate in ("etag", "ETag"):
        if headers.get(candidate):
            return headers[candidate]
    if isinstance(data, dict) and data.get("business_revision") is not None:
        return f'"{kind}:{int(data["business_revision"])}"'
    return None


def _create_dataset(client: ApiClient) -> None:
    st.subheader("Dataset")
    with st.form("create_dataset_form"):
        slug = st.text_input("slug", value="robot-grasp")
        description = st.text_input("description", value="Robot grasping demonstrations")
        submitted = st.form_submit_button("Create dataset")
    if submitted:
        try:
            payload = client.post("/datasets", json={"slug": slug, "description": description})["data"]
        except ApiError as exc:
            render_error(exc, context="Create dataset")
        else:
            dataset_id = payload.get("dataset_id") or payload.get("id")
            if dataset_id:
                state_set("df_dataset_id", dataset_id)
            st.success(f"Dataset created: {dataset_id}")
            st.json(payload)
    dataset_id = st.text_input("dataset_id", value=state_get("df_dataset_id", ""), key="df_dataset_input")
    if dataset_id:
        state_set("df_dataset_id", dataset_id)


def _create_version(client: ApiClient, dataset_id: str) -> None:
    st.subheader("Create a version")
    st.caption(
        "The API validates every episode (profile, QC PASS|WARN, approved annotation, known "
        "licence), freezes the manifest and runs the official LeRobot export + reader round-trip."
    )
    try:
        episodes = client.get("/episodes", params={"limit": 100, "quality_status": "PASS"}).get("data") or []
    except ApiError as exc:
        render_error(exc, context="Episode list")
        episodes = []
    if not episodes:
        st.info("No PASS episodes are available; WARN episodes may also be eligible after review.")
    labels = {
        f"{item.get('id')} · {item.get('quality_status')} · {item.get('task_label') or 'no task'}": item.get("id")
        for item in episodes
    }
    chosen = st.multiselect("episode_ids (membership)", list(labels), key="df_episode_ids")
    description = st.text_input("version description", key="df_version_description")
    if st.button("Create version", key="df_create_version"):
        if not dataset_id:
            render_empty("Create or enter a dataset_id first.")
            return
        if not chosen:
            render_empty("Select at least one episode for the version.")
            return
        selected_ids = sorted(labels[label] for label in chosen)
        try:
            payload = client.post(
                f"/datasets/{dataset_id}/versions",
                json={"episode_ids": selected_ids, "description": description},
            )["data"]
        except ApiError as exc:
            render_error(exc, context="Create version")
            return
        version_id = payload.get("version_id") or payload.get("id")
        if version_id:
            state_set("df_version_id", version_id)
            state_set("df_version_number", payload.get("version"))
        st.success(f"Version created: {payload}")
        st.json(payload)


def _render_version(client: ApiClient, dataset_id: str) -> None:
    st.subheader("Version status")
    version_number = st.number_input(
        "version", min_value=1, value=int(state_get("df_version_number", 1) or 1), key="df_version_number_input"
    )
    if not st.button("Load version", key="df_load_version"):
        return
    if not dataset_id:
        render_empty("Enter a dataset_id first.")
        return
    try:
        payload = client.get(f"/datasets/{dataset_id}/versions/{int(version_number)}")["data"]
    except ApiError as exc:
        render_error(exc, context="Version")
        return
    version_id = payload.get("id")
    state_set("df_version_id", version_id)
    state = str(payload.get("state"))
    st.metric("state", state)
    if state == "PUBLISHED":
        st.success(
            "PUBLISHED — immutable. Membership can no longer change (HTTP 409); scenario "
            "coverage is read from the frozen publish-time snapshot."
        )
    elif state == "FAILED":
        st.error(f"FAILED: {payload.get('error_code')} — {payload.get('error_detail')}")
    elif state == "READY":
        st.info("READY — verified round-trip. A reviewer can publish it.")
    st.json(
        {
            key: payload.get(key)
            for key in ("version", "manifest_sha256", "created_at", "published_at", "episodes")
        }
    )
    col1, col2 = st.columns(2)
    if payload.get("manifest_url"):
        col1.link_button("Download manifest.json", payload["manifest_url"])
    if payload.get("export_url"):
        col2.link_button("Download LeRobot v3 package", payload["export_url"])
    if payload.get("export_roundtrip"):
        st.caption("Official-reader round-trip evidence")
        st.json(payload["export_roundtrip"])
    manifest = payload.get("manifest")
    if isinstance(manifest, dict):
        with st.expander(f"manifest ({manifest.get('schema_version', 'unknown schema')})"):
            st.json(manifest)

    if state == "READY" and version_id:
        st.caption("Publishing requires the reviewer role and re-verifies the stored object hashes.")
        if st.button("Publish version", key="df_publish"):
            try:
                published = client.post(f"/dataset-versions/{version_id}/publish")["data"]
            except ApiError as exc:
                render_error(exc, context="Publish")
            else:
                st.success(f"Published: {published}")


def _render_coverage_lookup(client: ApiClient, version_id: str) -> None:
    st.markdown("**Scenario coverage (PUBLISHED versions only, Episodes not frames)**")
    scenario_key = st.text_input("scenario_key for coverage", key="df_coverage_scenario")
    if st.button("Check coverage", key="df_coverage_button"):
        if not version_id:
            render_empty("Load a dataset version first (Coverage needs its dataset_version_id).")
            return
        if not scenario_key:
            render_empty("Enter a scenario_key such as `transparent_cup_low_light`.")
            return
        try:
            coverage = client.get(
                "/quality/coverage",
                params={"dataset_version_id": version_id, "scenario_key": scenario_key},
            )["data"]
        except ApiError as exc:
            render_error(exc, context="Coverage")
            return
        st.metric("covered episodes", int(coverage.get("episode_count") or 0))
        st.caption(
            f"tagging_basis={coverage.get('tagging_basis')} · denominator={coverage.get('denominator')} "
            f"· computed_at={coverage.get('computed_at')} (UNKNOWN tags never count)"
        )
        st.json(coverage)


def _render_feedback(client: ApiClient) -> None:
    st.subheader("Feedback: simulated evaluation → collection orders")
    version_id = st.text_input(
        "dataset_version_id", value=state_get("df_version_id", ""), key="df_eval_version_id"
    )
    uploaded = st.file_uploader("evaluation/1.0 JSON", type=["json"], key="df_eval_file")
    if uploaded is not None:
        try:
            payload = json.loads(uploaded.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            st.error(f"Uploaded file is not valid JSON: {exc}")
            payload = None
        if isinstance(payload, dict):
            simulated_banner(payload)
            if st.button("Import evaluation", key="df_import_eval"):
                if not version_id:
                    render_empty("Enter the dataset_version_id the evaluation belongs to.")
                else:
                    try:
                        created = client.post(
                            f"/dataset-versions/{version_id}/evaluations", json=payload
                        )["data"]
                    except ApiError as exc:
                        render_error(exc, context="Evaluation import")
                    else:
                        state_set("df_evaluation_id", created.get("evaluation_id") or created.get("id"))
                        st.success(f"Evaluation imported: {created}")

    evaluation_id = st.text_input(
        "evaluation_id", value=state_get("df_evaluation_id", ""), key="df_eval_id_input"
    )
    threshold = st.number_input(
        "success_rate_threshold", min_value=0.01, max_value=1.0, value=0.8, step=0.01, key="df_threshold"
    )
    min_trials = st.number_input("min_trials", min_value=1, value=20, key="df_min_trials")
    if st.button("Analyze gaps", key="df_analyze"):
        _analyze_gaps_conditionally(client, evaluation_id, float(threshold), int(min_trials))

    _render_coverage_lookup(client, version_id)


def _analyze_gaps_conditionally(
    client: ApiClient, evaluation_id: str, threshold: float, min_trials: int
) -> None:
    """Conditional write (DO-A02 C3): read evaluation-rev, send If-Match + key."""
    if not evaluation_id:
        render_empty("Import an evaluation first.")
        return
    etag: str | None = None
    try:
        eval_body, eval_headers = client.get_with_headers(f"/evaluations/{evaluation_id}")
        eval_data = eval_body.get("data") if isinstance(eval_body.get("data"), dict) else {}
        etag = _etag(eval_headers, eval_data, "evaluation-rev")
    except ApiError as exc:
        if not exc.endpoint_missing:
            render_error(exc, context="Load evaluation revision")
            return
        st.caption("Evaluation revision endpoint not available yet; falling back to the V1 write shape.")

    request_headers = {"Idempotency-Key": _idempotency_key()}
    if etag:
        request_headers["If-Match"] = etag
    try:
        analysis = client.post(
            f"/evaluations/{evaluation_id}/analyze-gaps",
            json={"success_rate_threshold": threshold, "min_trials": min_trials},
            headers=request_headers,
        )["data"]
    except ApiError as exc:
        render_error(exc, context="Gap analysis")
        return
    st.success("Gap analysis completed.")
    st.caption(
        f"operation_id={analysis.get('operation_id', 'n/a')} · "
        f"effect_state={analysis.get('effect_state', 'n/a')}"
    )
    st.json(analysis)


def _render_collection_orders(client: ApiClient, version_id: str) -> None:
    st.subheader("Collection orders (V2 lifecycle)")
    try:
        orders = client.get("/collection-orders", params={"limit": 50}).get("data") or []
    except ApiError as exc:
        render_error(exc, context="Collection orders")
        return
    if not orders:
        render_empty("No collection orders yet — import a simulated evaluation and run the analysis.")
        return
    dataframe_rows(
        [
            {
                "order_id": order.get("id"),
                "scenario_key": order.get("scenario_key"),
                "state": order.get("state"),
                "requested": order.get("requested_count"),
                "linked": order.get("linked_count"),
                "is_simulated_source": order.get("is_simulated_source"),
                "approval_note": order.get("approval_note"),
            }
            for order in orders
        ],
        empty_message="No collection orders yet.",
    )
    if any(order.get("is_simulated_source") for order in orders):
        st.warning(
            "SIMULATED EVALUATION SOURCE — these collection orders were derived from a simulated "
            "evaluation report; they are not evidence of real robot training improvement."
        )

    labels = {f"{order.get('scenario_key')} · {order.get('state')} ({order.get('id')})": order for order in orders}
    selected_label = st.selectbox("Collection order", list(labels), key="df_order_selected")
    order = labels[selected_label]
    order_id = order.get("id")
    if not order_id:
        return
    try:
        detail = client.get(f"/collection-orders/{order_id}")["data"]
    except ApiError as exc:
        render_error(exc, context="Collection order")
        return
    evaluation = detail.get("evaluation") or {}
    if evaluation.get("is_simulated"):
        st.warning(
            f"SIMULATED EVALUATION `{evaluation.get('evaluation_id')}` "
            f"(model {evaluation.get('model_id')}) — not a real training result."
        )
    st.metric("order state", str((detail.get("order") or {}).get("state")))
    st.caption(
        f"linked {detail.get('linked_count')}/{detail.get('requested_count')} · "
        f"fulfilled={detail.get('fulfilled')} · simulated_source={detail.get('is_simulated_source')}"
    )
    dataframe_rows(
        detail.get("linked_episodes") or [],
        empty_message="No episodes linked to this order yet.",
    )

    if (detail.get("order") or {}).get("state") == "PROPOSED":
        st.markdown("**Approve order (reviewer, must differ from creator)**")
        note = st.text_area("approval note", key="df_order_approval_note")
        if st.button("Approve order", key="df_order_approve"):
            try:
                approved = client.post(
                    f"/collection-orders/{order_id}/approve", json={"note": note}
                )["data"]
            except ApiError as exc:
                render_error(exc, context="Approve order")
            else:
                st.success(f"Order approved: state={approved.get('state')}")
                st.json(approved)

    if (detail.get("order") or {}).get("state") in {"APPROVED", "IN_PROGRESS"}:
        st.markdown("**Link fulfillment episodes (operator)**")
        st.caption(
            "Each episode needs: a credible scenario tag for this scenario_key, an import that "
            "references this order, QC PASS/WARN and an approved annotation."
        )
        raw_ids = st.text_area(
            "episode_ids (one per line)", key="df_link_episode_ids", height=100
        )
        if st.button("Link episodes", key="df_order_link"):
            episode_ids = [line.strip() for line in raw_ids.splitlines() if line.strip()]
            if not episode_ids:
                render_empty("Enter at least one episode_id to link.")
            else:
                try:
                    linked = client.post(
                        f"/collection-orders/{order_id}/link-episodes",
                        json={"episode_ids": episode_ids},
                    )["data"]
                except ApiError as exc:
                    render_error(exc, context="Link episodes")
                else:
                    st.success(f"Linked: state={linked.get('state')} · fulfilled={linked.get('fulfilled')}")
                    st.json(linked)

    if (detail.get("order") or {}).get("state") not in {"FULFILLED", "CANCELLED"}:
        with st.expander("Cancel order"):
            reason = st.text_input("cancellation reason", key="df_order_cancel_reason")
            if st.button("Cancel order", key="df_order_cancel"):
                try:
                    cancelled = client.post(
                        f"/collection-orders/{order_id}/cancel", json={"reason": reason}
                    )["data"]
                except ApiError as exc:
                    render_error(exc, context="Cancel order")
                else:
                    st.success(f"Order cancelled: state={cancelled.get('state')}")
    order_version = version_id or state_get("df_version_id", "")
    if order_version:
        st.caption(
            "Coverage for this order's scenario is read from the frozen PUBLISHED-version snapshot above."
        )


def render(client: ApiClient) -> None:
    st.title("Dataset & Feedback")
    st.caption("Versions are frozen and immutable once published; feedback orders are deterministic.")
    if not client.is_configured:
        st.error("API_KEY is not set; dataset operations need an operator/reviewer key.")
        st.info("Set API_KEY and restart Streamlit.")
        return
    _create_dataset(client)
    dataset_id = state_get("df_dataset_id", "")
    _create_version(client, dataset_id)
    _render_version(client, dataset_id)
    _render_feedback(client)
    _render_collection_orders(client, state_get("df_version_id", ""))
