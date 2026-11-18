"""Page 3 — Quality & Annotation Review (SPEC 11.3, V2 D2-M3).

Shows quality findings, detection-box overlays from the API preview and lets a
reviewer approve/reject annotations.  V2 annotation sets are edited through the
immutable ``annotation-revision/2.0`` chain: this page lists revisions, offers a
numeric box-edit form that saves a new revision with a strong ``If-Match`` ETag
and an ``Idempotency-Key``, and renders the explicit conflict message when
another reviewer saved first (``412 PRECONDITION_FAILED``).  Legacy v1 sets keep
the original one-shot review form; the backend enforces roles, the WARN
release-note rule and the two-person approval rule.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import streamlit as st

from embodied_ui.api_client import ApiClient, ApiError
from embodied_ui.pages._shared import (
    dataframe_rows,
    draw_detections,
    render_empty,
    render_error,
    state_get,
    state_set,
)

WARN_NOTE_MIN_CHARS = 10
REVISION_NOTE_MIN_CHARS = 10
REVISION_NOTE_MAX_CHARS = 500
APPROVAL_NOTE_MIN_CHARS = 10
EDITOR_COLUMNS = ("detection_id", "class_id", "class_name", "score", "x1", "y1", "x2", "y2", "origin")


def _select_episode(client: ApiClient) -> str | None:
    quality_status = st.selectbox(
        "Filter quality_status", ("", "PASS", "WARN", "FAIL"), key="qr_qc_filter"
    )
    params: dict[str, Any] = {"limit": 30}
    if quality_status:
        params["quality_status"] = quality_status
    try:
        payload = client.get("/episodes", params=params)
    except ApiError as exc:
        render_error(exc, context="Episode list")
        return None
    episodes = payload.get("data") or []
    if not episodes:
        render_empty("No episodes returned by the API for this filter.")
        return None
    labels = {
        f"{item.get('id')} · {item.get('quality_status') or 'QC?'} · {item.get('task_label') or 'no task'}": item.get(
            "id"
        )
        for item in episodes
    }
    selected = st.selectbox("Episode", list(labels), key="qr_episode")
    return labels.get(selected)


def _render_quality(client: ApiClient, episode_id: str) -> None:
    st.subheader("Quality report")
    try:
        quality = client.get(f"/episodes/{episode_id}/quality")["data"]
    except ApiError as exc:
        render_error(exc, context="Quality report")
        return
    verdict = str(quality.get("verdict"))
    if verdict == "FAIL":
        st.error("Verdict FAIL — this episode cannot be published in a dataset version.")
    elif verdict == "WARN":
        st.warning("Verdict WARN — publishing requires an explicit reviewer note of at least 10 characters.")
    else:
        st.success(f"Verdict {verdict}")
    findings = quality.get("findings") or []
    dataframe_rows(
        [
            {
                "rule_code": finding.get("rule_code"),
                "severity": finding.get("severity"),
                "observed": finding.get("observed"),
                "threshold": finding.get("limit") or finding.get("threshold"),
                "affected_time_ranges": finding.get("affected_time_ranges"),
                "evidence_asset_ids": finding.get("evidence_asset_ids"),
            }
            for finding in findings
        ],
        empty_message="The API returned no findings for this episode.",
    )
    if quality.get("metrics"):
        st.json({"metrics": quality["metrics"]})
    if quality.get("report_url"):
        st.link_button("Open stored quality report", quality["report_url"])


# ---------------------------------------------------------------------------
# V2 annotation revisions (DO2-14..17)
# ---------------------------------------------------------------------------
def _editor_rows(frame: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detection in frame.get("detections") or []:
        xyxy = detection.get("xyxy") or [0.0, 0.0, 0.0, 0.0]
        rows.append(
            {
                "detection_id": detection.get("detection_id") or "",
                "class_id": int(detection.get("class_id") or 0),
                "class_name": detection.get("class_name") or "other",
                "score": detection.get("score"),
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "origin": detection.get("origin") or "HUMAN",
            }
        )
    return rows


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _edited_records(edited: Any) -> list[dict[str, Any]]:
    if hasattr(edited, "to_dict"):  # pandas DataFrame returned by st.data_editor
        return list(edited.to_dict("records"))
    return [dict(row) for row in edited]


def _rows_to_detections(rows: list[dict[str, Any]], frame_index: int) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        detection_id = str(row.get("detection_id") or "").strip()
        if not detection_id:
            # A newly added blank row: use a deterministic HUMAN identifier.
            detection_id = f"human-{frame_index}-{ordinal:03d}"
        if _is_missing(row.get("x1")) or _is_missing(row.get("y1")):
            continue
        score_value = row.get("score")
        score = None if _is_missing(score_value) else float(score_value)
        class_id = int(row.get("class_id") or 0)
        detections.append(
            {
                "detection_id": detection_id,
                "class_id": class_id,
                "class_name": str(row.get("class_name") or "other"),
                "score": score,
                "xyxy": [
                    float(row.get("x1") or 0.0),
                    float(row.get("y1") or 0.0),
                    float(row.get("x2") or 0.0),
                    float(row.get("y2") or 0.0),
                ],
                "origin": str(row.get("origin") or "HUMAN"),
            }
        )
    return detections


def _revision_rows(revisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "revision_no": revision.get("revision_no"),
            "state": revision.get("state"),
            "created_by": revision.get("created_by"),
            "content_sha256": revision.get("content_sha256"),
            "frames_changed": ((revision.get("change_summary") or {}).get("frames") or {}).get("effective"),
            "note": revision.get("note"),
        }
        for revision in revisions
    ]


def _save_revision(
    client: ApiClient,
    annotation_set_id: str,
    current_revision_no: int,
    frame: dict[str, Any],
    edited_rows: list[dict[str, Any]],
    note: str,
) -> None:
    frame_index = int(frame.get("frame_index") or 0)
    state_key = f"rev_idem_{annotation_set_id}_{current_revision_no}"
    idempotency_key = state_get(state_key) or str(uuid.uuid4())
    state_set(state_key, idempotency_key)
    body = {
        "base_revision_no": current_revision_no,
        "frames": [
            {
                "frame_index": frame_index,
                "t_ns": int(frame.get("t_ns") or 0),
                "detections": _rows_to_detections(edited_rows, frame_index),
            }
        ],
        "note": note,
    }
    try:
        result = client.post(
            f"/annotation-sets/{annotation_set_id}/revisions",
            json=body,
            headers={
                "If-Match": f'"annotation-rev:{current_revision_no}"',
                "Idempotency-Key": idempotency_key,
            },
        )["data"]
    except ApiError as exc:
        if exc.code == "PRECONDITION_FAILED":
            st.error(
                "Conflict: another reviewer saved a newer annotation revision before this save. "
                "Nothing was overwritten — reload the revision list and re-apply your boxes."
            )
            st.caption(f"How to recover: {exc.recovery_hint}")
            if exc.details:
                st.json(exc.details)
        elif exc.code == "NO_CHANGE":
            st.info("No effective box change was detected; the API created no new revision.")
        else:
            render_error(exc, context="Save annotation revision")
        return
    state_set(state_key, None)
    st.success(
        f"Saved immutable annotation revision {result.get('current_revision_no')} "
        f"({result.get('etag')})."
    )


def _decide_revision(
    client: ApiClient,
    revision_id: str,
    decision: str,
    note: str,
) -> None:
    try:
        result = client.post(
            f"/annotation-revisions/{revision_id}/decision",
            json={"decision": decision, "note": note},
        )["data"]
    except ApiError as exc:
        render_error(exc, context="Annotation revision decision")
        return
    st.success(
        f"Revision {result['revision']['revision_no']} is now {result['revision']['state']}; "
        f"approved_revision_id={result.get('approved_revision_id')}."
    )


def _render_revisions(client: ApiClient, annotation_set: dict[str, Any]) -> None:
    annotation_set_id = annotation_set.get("annotation_set_id")
    if not annotation_set_id:
        return
    st.subheader("Annotation revisions (annotation-revision/2.0)")
    try:
        listing = client.get(f"/annotation-sets/{annotation_set_id}/revisions")["data"]
    except ApiError as exc:
        render_error(exc, context="Annotation revisions")
        return
    current_revision_no = int(listing.get("current_revision_no") or 1)
    revisions = listing.get("revisions") or []
    st.json(
        {
            "current_revision_no": current_revision_no,
            "approved_revision_id": listing.get("approved_revision_id"),
            "status": listing.get("status"),
        }
    )
    dataframe_rows(_revision_rows(revisions), empty_message="No revision rows exist yet for this set.")
    if not revisions:
        st.info(
            "No immutable revision exists yet. Saving the first box edit creates revision 1 "
            "(the model-output baseline) and revision 2 (your edit) atomically."
        )
        return

    # -- numeric box editor -------------------------------------------------
    try:
        current = client.get(
            f"/annotation-sets/{annotation_set_id}/revisions/{current_revision_no}"
        )["data"]
    except ApiError as exc:
        render_error(exc, context="Current revision content")
        return
    frames = (current.get("annotation") or {}).get("frames") or []
    if not frames:
        render_empty("The current revision has no annotated frames to edit.")
        return
    st.caption(
        f"Editing revision {current_revision_no} with If-Match {current.get('etag')}. "
        "The API replaces all detections of the selected frame and copies every other frame."
    )
    frame_labels = {
        f"frame_index={frame.get('frame_index')} · t_ns={frame.get('t_ns')}": frame for frame in frames
    }
    selected = st.selectbox("Frame to replace", list(frame_labels), key="qr_rev_frame")
    frame = frame_labels[selected]
    edited = st.data_editor(
        _editor_rows(frame),
        num_rows="dynamic",
        key=f"qr_rev_editor_{annotation_set_id}_{current_revision_no}_{frame.get('frame_index')}",
        column_config={column: st.column_config.Column(column) for column in EDITOR_COLUMNS},
    )
    note = st.text_input(
        "Change note (10..500 characters)",
        key=f"qr_rev_note_{annotation_set_id}_{current_revision_no}",
    )
    if st.button("Save annotation revision", key="qr_rev_save"):
        stripped = note.strip()
        if not (REVISION_NOTE_MIN_CHARS <= len(stripped) <= REVISION_NOTE_MAX_CHARS):
            st.warning(
                f"The API requires a change note of {REVISION_NOTE_MIN_CHARS}..{REVISION_NOTE_MAX_CHARS} "
                "characters; nothing was submitted."
            )
        else:
            _save_revision(
                client,
                annotation_set_id,
                current_revision_no,
                frame,
                _edited_records(edited),
                stripped,
            )

    # -- two-person decision ------------------------------------------------
    pending = [revision for revision in revisions if revision.get("state") == "IN_REVIEW"]
    if pending:
        st.markdown("**Decide an in-review revision**")
        pending_labels = {
            f"revision {revision.get('revision_no')} · created_by={revision.get('created_by')}": revision
            for revision in pending
        }
        selected_revision = pending_labels[
            st.selectbox("Revision", list(pending_labels), key="qr_rev_decision_target")
        ]
        decision = st.radio(
            "Decision", ("APPROVED", "REJECTED"), key="qr_rev_decision", horizontal=True
        )
        decision_note = st.text_area(
            "Decision note",
            key="qr_rev_decision_note",
            help=(
                "Approval requires a reviewer different from the revision creator and an "
                f"explanatory note of at least {APPROVAL_NOTE_MIN_CHARS} characters."
            ),
        )
        if st.button("Submit revision decision", key="qr_rev_decision_submit"):
            if decision == "APPROVED" and len(decision_note.strip()) < APPROVAL_NOTE_MIN_CHARS:
                st.warning(
                    f"Approval requires a note of at least {APPROVAL_NOTE_MIN_CHARS} characters; "
                    "nothing was submitted."
                )
            else:
                _decide_revision(
                    client,
                    str(selected_revision.get("revision_id")),
                    decision,
                    decision_note.strip(),
                )
    else:
        st.info("No revision is IN_REVIEW; every stored revision has already been decided.")


# ---------------------------------------------------------------------------
# legacy v1 review form
# ---------------------------------------------------------------------------
def _render_legacy_review(client: ApiClient, annotation_set: dict[str, Any]) -> None:
    status = str(annotation_set.get("status") or "")
    if status == "PENDING_REVIEW":
        st.markdown("**Review this AnnotationSet**")
        st.caption("Reviewer role required. Rejected sets stay immutable; re-run detection for a new set.")
        decision = st.radio("Decision", ("APPROVED", "REJECTED"), key="qr_decision", horizontal=True)
        note = st.text_area(
            "Review note",
            key="qr_note",
            help="Required for WARN releases: explain why the WARN is acceptable (>= 10 characters).",
        )
        if st.button("Submit review", key="qr_submit_review"):
            if len(note.strip()) < WARN_NOTE_MIN_CHARS:
                st.warning(
                    f"The backend requires at least {WARN_NOTE_MIN_CHARS} characters for a WARN release; "
                    "this note is shorter and will be rejected in that case."
                )
            try:
                result = client.post(
                    f"/annotation-sets/{annotation_set.get('id')}/reviews",
                    json={"decision": decision, "note": note},
                )["data"]
            except ApiError as exc:
                render_error(exc, context="Review")
            else:
                st.success(f"Review recorded; new status {result.get('new_status')}")
    else:
        st.info(f"AnnotationSet status is {status}; no further review action is possible on this immutable set.")


def _render_annotations(client: ApiClient, episode_id: str) -> None:
    st.subheader("Candidate annotations & review")
    try:
        payload = client.get(f"/episodes/{episode_id}/annotations")["data"]
    except ApiError as exc:
        render_error(exc, context="Annotations")
        return
    annotation_set = payload.get("annotation_set") or {}
    frames = payload.get("frames") or []
    if annotation_set:
        st.json(
            {
                key: annotation_set.get(key)
                for key in ("annotation_set_id", "status", "model_id", "model_sha256", "summary")
            }
        )
    else:
        render_empty("No AnnotationSet exists for this episode yet (run DETECT_OBJECTS first).")
        return

    preview_url = payload.get("preview_url") or annotation_set.get("thumbnail_url")
    if preview_url and frames:
        first_frame = frames[0]
        try:
            image = client.fetch_bytes(preview_url)
        except ApiError as exc:
            render_error(exc, context="Preview")
        else:
            overlay = draw_detections(image, [first_frame])
            if overlay is not None:
                st.image(overlay, caption=f"frame_index={first_frame.get('frame_index')} overlay")
    elif not frames:
        st.caption("The API returned no annotated frames (EMPTY_ANNOTATION is a valid INFO finding).")

    if frames:
        dataframe_rows(
            [
                {
                    "frame_index": frame.get("frame_index"),
                    "t_ns": frame.get("t_ns"),
                    "detections": len(frame.get("detections") or []),
                }
                for frame in frames[:50]
            ],
            empty_message="No frames returned.",
        )

    is_v2_set = int(annotation_set.get("revision_count") or 0) > 0 or bool(
        annotation_set.get("approved_revision_id")
    )
    if is_v2_set:
        _render_revisions(client, annotation_set)
    else:
        _render_legacy_review(client, annotation_set)
        st.caption(
            "Saving a box edit through the annotation revision API converts this set to the "
            "v2 review workflow; the legacy review endpoint then returns 409 USE_REVISION_REVIEW_API."
        )


def render(client: ApiClient) -> None:
    st.title("Quality & Annotation Review")
    st.caption("Findings, evidence and review decisions come from the API; the backend enforces roles.")
    if not client.is_configured:
        st.error("API_KEY is not set; review actions are unavailable.")
        st.info("Set API_KEY to a reviewer-role key and restart Streamlit.")
        return
    episode_id = _select_episode(client)
    if not episode_id:
        return
    _render_quality(client, episode_id)
    _render_annotations(client, episode_id)
