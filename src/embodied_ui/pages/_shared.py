"""Shared Streamlit rendering helpers for the four pages.

Only presentation logic lives here; no business data is fabricated.  Every
helper either renders a real API value or an explicit empty/error state.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from typing import Any

import streamlit as st

from embodied_ui.api_client import ApiError

STATE_PREFIX = "dataops_ui_"


def state_get(key: str, default: Any = None) -> Any:
    return st.session_state.get(f"{STATE_PREFIX}{key}", default)


def state_set(key: str, value: Any) -> None:
    st.session_state[f"{STATE_PREFIX}{key}"] = value


def render_error(exc: ApiError, *, context: str = "") -> None:
    """Error state + recovery hint (SPEC 11 requires at least one recovery hint)."""
    prefix = f"{context}: " if context else ""
    if exc.endpoint_missing:
        st.info(
            f"{prefix}API endpoint not available yet "
            f"({exc.status_code} {exc.code}). It will appear once the backend route is deployed."
        )
    else:
        st.error(f"{prefix}{exc.code} (HTTP {exc.status_code}): {exc.message}")
        if exc.details:
            with st.expander("Error details"):
                st.json(exc.details)
    st.caption(f"How to recover: {exc.recovery_hint}")


def render_empty(message: str) -> None:
    st.info(message)


def action_buttons(label: str, key: str) -> bool:
    return st.button(label, key=f"{STATE_PREFIX}{key}")


def dataframe_rows(rows: Iterable[dict[str, Any]], *, empty_message: str) -> None:
    materialised = list(rows)
    if not materialised:
        render_empty(empty_message)
        return
    st.dataframe(materialised, use_container_width=True)


def sample_chart(samples: Sequence[dict[str, Any]]) -> None:
    """Plot the 6D state/action series of an episode sample page."""
    if not samples:
        render_empty("No samples returned for this episode.")
        return
    chart: dict[str, list[float]] = {f"state_{i}": [] for i in range(1, 7)}
    for i in range(1, 7):
        chart[f"action_{i}"] = []
    t_values: list[float] = []
    for row in samples:
        t_values.append(float(row.get("t_ns", 0)) / 1_000_000_000.0)
        state = row.get("state") or []
        action = row.get("action") or []
        for index in range(6):
            chart[f"state_{index + 1}"].append(float(state[index]) if index < len(state) else float("nan"))
            chart[f"action_{index + 1}"].append(
                float(action[index]) if index < len(action) else float("nan")
            )
    st.caption("t_ns / 1e9 (seconds). State and action are the canonical 6D columns from the API.")
    st.line_chart({"t_s": t_values, **chart}, x="t_s")


def draw_detections(image_bytes: bytes, frames: Sequence[dict[str, Any]]) -> bytes | None:
    """Overlay YOLO boxes on a preview image using the API's frame coordinates."""
    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover - pillow is a core dependency
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    draw = ImageDraw.Draw(image)
    width, height = image.size
    scale_x = width / 640.0
    scale_y = height / 480.0
    for frame in frames:
        for detection in frame.get("detections") or []:
            xyxy = detection.get("xyxy") or []
            if len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = (
                float(xyxy[0]) * scale_x,
                float(xyxy[1]) * scale_y,
                float(xyxy[2]) * scale_x,
                float(xyxy[3]) * scale_y,
            )
            draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=2)
            label = f"{detection.get('class_name', '?')} {float(detection.get('score', 0)):.2f}"
            draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=(255, 64, 64))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def simulated_banner(payload: dict[str, Any]) -> None:
    if payload.get("is_simulated") is True:
        st.warning(
            "SIMULATED EVALUATION — these numbers come from an offline/simulated evaluation "
            "report, not from real robot training or production measurements."
        )
