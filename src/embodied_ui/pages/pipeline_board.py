"""Page 2 — Pipeline Board (SPEC 11.2, DO-A02 C3).

Shows run/task states, attempts, worker and timings; supports creating a run,
retrying a failed step and cooperative cancellation.

The three controlled writes are **conditional** in V2 (DO-A02): the page reads
the resource (``GET /episodes/{id}`` / ``GET /tasks/{id}``), forwards the strong
``ETag`` as ``If-Match`` and sends a per-click ``Idempotency-Key``.  Responses for
``GET`` that predate the ETag contract are handled defensively: if no ETag is
available the write is still attempted (V1 APIs ignore the header and V2 replies
428, which is rendered with an explicit recovery hint instead of a traceback).
"""

from __future__ import annotations

import uuid

import streamlit as st

from embodied_ui.api_client import ApiClient, ApiError
from embodied_ui.pages._shared import dataframe_rows, render_empty, render_error, state_get, state_set


def _idempotency_key() -> str:
    return str(uuid.uuid4())


def _etag(headers: dict[str, str], data: dict | None, kind: str) -> str | None:
    """Strong ETag from the response header, else derive from business_revision."""
    for candidate in ("etag", "ETag"):
        value = headers.get(candidate)
        if value:
            return value
    revision = None
    if isinstance(data, dict):
        revision = data.get("business_revision")
    if revision is not None:
        return f'"{kind}:{int(revision)}"'
    return None


def _extract_run_id(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("run_id", "new_run_id", "id"):
        if payload.get(key):
            return str(payload[key])
    nested = payload.get("run")
    if isinstance(nested, dict) and nested.get("id"):
        return str(nested["id"])
    downstream = payload.get("downstream")
    if isinstance(downstream, dict) and downstream.get("run_id"):
        return str(downstream["run_id"])
    return None


def _show_operation(payload: dict) -> None:
    """Render the DO-A02 receipt fields when the API returns them."""
    if not isinstance(payload, dict):
        return
    operation_id = payload.get("operation_id")
    effect_state = payload.get("effect_state")
    if operation_id or effect_state:
        st.caption(
            f"operation_id={operation_id or 'n/a'} · effect_state={effect_state or 'n/a'} "
            "(effect committed before HTTP 202)"
        )


def _create_run(client: ApiClient) -> None:
    st.subheader("Start a pipeline run")
    episode_id = st.text_input("Episode id", key="pb_episode_id", value=state_get("pb_last_episode", ""))
    pipeline_version = st.text_input("pipeline_version", value="2.0.0", key="pb_pipeline_version")
    if not st.button("Create run", key="pb_create_run"):
        return
    if not episode_id:
        render_empty("Enter the episode id (from the Import & Episode page).")
        return

    etag: str | None = None
    try:
        body, headers = client.get_with_headers(f"/episodes/{episode_id}")
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        etag = _etag(headers, data, "episode-rev")
    except ApiError as exc:
        if exc.endpoint_missing:
            st.caption("Episode revision endpoint not available yet; falling back to the V1 write shape.")
        else:
            render_error(exc, context="Load episode revision")
            return

    request_headers = {"Idempotency-Key": _idempotency_key()}
    if etag:
        request_headers["If-Match"] = etag
    try:
        payload = client.post(
            f"/episodes/{episode_id}/pipeline-runs",
            json={"pipeline_version": pipeline_version},
            headers=request_headers,
        )["data"]
    except ApiError as exc:
        render_error(exc, context="Create run")
        return
    run_id = _extract_run_id(payload)
    if run_id:
        known = list(state_get("pb_runs", []) or [])
        if run_id not in known:
            known.insert(0, run_id)
            state_set("pb_runs", known[:20])
        state_set("pb_last_episode", episode_id)
        st.success(f"Run {run_id} created with state {payload.get('state')}")
    else:
        st.warning("The API accepted the request but returned no run id.")
    _show_operation(payload)
    st.json(payload)


def _render_run(client: ApiClient, run_id: str) -> None:
    try:
        payload = client.get(f"/pipeline-runs/{run_id}")["data"]
    except ApiError as exc:
        render_error(exc, context=f"Run {run_id}")
        return
    run = payload.get("run") or {}
    tasks = payload.get("tasks") or []
    current_run_id = str(run.get("id") or run.get("run_id") or run_id)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Run state", str(run.get("state")))
    col2.metric("Tasks", len(tasks))
    col3.metric("Attempts", sum(int(task.get("attempt") or 0) for task in tasks))
    workers = {str(task.get("worker_id")) for task in tasks if task.get("worker_id")}
    col4.metric("Workers", len(workers))
    st.caption(f"started_at={run.get('started_at')} finished_at={run.get('finished_at')}")
    if run.get("business_revision") is not None:
        st.caption(f"run business_revision={run.get('business_revision')}")
    if run.get("error_code"):
        st.error(f"Run error: {run['error_code']} — {run.get('error_detail') or ''}")

    if tasks:
        rows = [
            {
                "step": task.get("step"),
                "state": task.get("state"),
                "attempt": task.get("attempt"),
                "max_attempts": task.get("max_attempts"),
                "worker_id": task.get("worker_id"),
                "started_at": task.get("started_at"),
                "finished_at": task.get("finished_at"),
                "error_code": task.get("error_code"),
                "error_detail": (task.get("error_detail") or "")[:200],
            }
            for task in tasks
        ]
        dataframe_rows(rows, empty_message="This run has no tasks.")
        failed = [task for task in tasks if task.get("state") == "FAILED"]
        if failed:
            st.markdown("**Retry failed work**")
            selected = st.selectbox(
                "Failed task",
                [f"{task.get('step')} ({task.get('id')})" for task in failed],
                key=f"pb_failed_{run_id}",
            )
            if st.button("Create retry run from failed step", key=f"pb_retry_{run_id}"):
                _retry_failed_task(client, selected, current_run_id)

    run_state = str(run.get("state"))
    if run_state not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        with st.expander("Cancel this run"):
            st.caption("Cancellation is cooperative: the worker stops after its current unit of work.")
            if st.button("Cancel run", key=f"pb_cancel_{run_id}"):
                try:
                    cancelled = client.post(f"/pipeline-runs/{run_id}/cancel")["data"]
                except ApiError as exc:
                    render_error(exc, context="Cancel")
                else:
                    st.success(f"Cancel accepted; run state now {cancelled.get('state')}")
                    _show_operation(cancelled)


def _retry_failed_task(client: ApiClient, selected: str, expected_run_id: str) -> None:
    task_id = selected.rsplit("(", 1)[-1].rstrip(")")
    etag: str | None = None
    try:
        task_body, task_headers = client.get_with_headers(f"/tasks/{task_id}")
        task_data = task_body.get("data") if isinstance(task_body.get("data"), dict) else {}
        etag = _etag(task_headers, task_data, "task-rev")
        task_record = task_data.get("task") if isinstance(task_data.get("task"), dict) else task_data
        if isinstance(task_record, dict):
            expected_run_id = str(task_record.get("run_id") or expected_run_id)
    except ApiError as exc:
        if not exc.endpoint_missing:
            render_error(exc, context="Load task revision")
            return
        st.caption("Task revision endpoint not available yet; falling back to the V1 retry shape.")

    request_headers = {"Idempotency-Key": _idempotency_key()}
    if etag:
        request_headers["If-Match"] = etag
    try:
        retried = client.post(
            f"/tasks/{task_id}/retry",
            json={"expected_run_id": expected_run_id},
            headers=request_headers,
        )["data"]
    except ApiError as exc:
        render_error(exc, context="Retry")
        return
    new_run = _extract_run_id(retried)
    if new_run:
        known = list(state_get("pb_runs", []) or [])
        if new_run not in known:
            known.insert(0, new_run)
            state_set("pb_runs", known)
    st.success(f"New retry run: {new_run}")
    _show_operation(retried)


def render(client: ApiClient) -> None:
    st.title("Pipeline Board")
    if not client.is_configured:
        st.error("API_KEY is not set; the board cannot query the API.")
        st.info("Set API_KEY and restart Streamlit.")
        return
    st.caption(
        "V2 controlled writes are conditional: this page sends If-Match (strong ETag) and an "
        "Idempotency-Key; 412/428/503 responses are shown with recovery hints."
    )
    _create_run(client)

    st.subheader("Tracked runs")
    st.caption("Run ids are kept in this browser session; paste a run id to inspect any run.")
    manual = st.text_input("Run id", key="pb_manual_run")
    try:
        payload = client.get("/episodes", params={"limit": 20})
    except ApiError as exc:
        render_error(exc, context="Episode list")
        payload = {}
    episodes = payload.get("data") or []
    if episodes:
        st.caption(f"{len(episodes)} episode(s) available for new runs.")
    else:
        st.caption("No episodes yet — run an import first.")

    known = list(state_get("pb_runs", []) or [])
    if manual and manual not in known:
        known.insert(0, manual)
    if not known:
        render_empty("No run tracked yet. Create a run above or paste a run id.")
        return
    choice = st.selectbox("Select run", known, key="pb_selected_run")
    if choice:
        _render_run(client, choice)
