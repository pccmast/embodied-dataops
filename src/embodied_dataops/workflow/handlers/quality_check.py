"""``QUALITY_CHECK`` handler: load the canonical Episode, run qc-policy/1.0, persist the report.

What it reads
-------------
* the Episode and its ``PipelineRun`` from PostgreSQL;
* the ``NORMALIZE_EPISODE`` task ``output_json`` of the same run: canonical object
  keys (``canonical.metadata_key`` / ``samples_key`` / ``video_key``), the
  alignment/sync diagnostics, ``normalization_valid``, observed dimensions and
  joint names (the normalizer contract documented in
  ``imports/mcap_profile_v1.py``);
* the canonical ``metadata.json`` and ``samples.parquet`` objects, plus the
  canonical ``frames/front.mp4`` when it exists;
* nothing else -- no network, no ROS runtime, never the raw source bytes.

Robustness required by SPEC 6.1
-------------------------------
A structural incompatibility (for example a 5D action or a missing sync
window) makes ``NORMALIZE_EPISODE`` write diagnostics and
``normalization_valid=false`` instead of a publishable ``samples.parquet``.
``QUALITY_CHECK`` must still produce a complete report: it falls back to the
normalizer diagnostics/metadata, emits ``DIMENSION_MISMATCH`` /
``SYNC_MISSING_*`` and stores the report with a ``FAIL`` verdict.  Rules whose
evidence is genuinely absent (for example blurred-frame statistics without a
decoded video) are reported as *unavailable* rather than silently passing.

The handler runs inside the engine transaction: it never commits, rolls back or
touches ``tasks.state``; it returns the verdict so the engine stops downstream
steps on ``FAIL`` (SPEC 6.1).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.canonical.contract import (
    ROBOT_PROFILE,
    canonical_keys,
    read_samples_parquet,
)
from embodied_dataops.db.models import Episode, PipelineRun, SourceImport, Task
from embodied_dataops.domain.enums import EpisodeType, Step
from embodied_dataops.errors import ApiError, CancelledError, ErrorCode, TerminalTaskError
from embodied_dataops.quality.policy import QC_POLICY_VERSION
from embodied_dataops.quality.report import build_quality_report, persist_quality_report
from embodied_dataops.quality.rules import (
    QualityEvaluation,
    QualityInput,
    VideoInfo,
    evaluate_quality,
)
from embodied_dataops.workflow.handlers.base import HandlerContext, HandlerResult, TaskHandler

#: Keys under which a normalizer may publish canonical object keys (tolerant of
#: both the flat and the nested contract documented with the teammate modules).
_METADATA_KEY_CANDIDATES = ("metadata_key", "metadata_object_key", "metadata")
_SAMPLES_KEY_CANDIDATES = ("samples_key", "samples_object_key", "samples")
_VIDEO_KEY_CANDIDATES = ("video_key", "video_object_key", "video")
_THUMBNAIL_KEY_CANDIDATES = ("thumbnail_key", "thumbnail_object_key", "thumbnail")

#: Keys carrying the alignment/sync diagnostics.
_ALIGNMENT_KEY_CANDIDATES = ("alignment", "sync_diagnostics", "alignment_diagnostics")
#: Keys carrying the intended 10 Hz grid (episode-relative ns).
_GRID_KEY_CANDIDATES = ("grid", "grid_t_ns", "grid_rel_ns")
#: Keys carrying absolute per-stream source timestamps.
_STREAM_TIMES_KEY_CANDIDATES = ("stream_times", "source_stream_times")

#: Bound on decoded canonical video frames analysed for BLUR_FRAME_RATE.
_MAX_VIDEO_FRAMES = 20_000


# ---------------------------------------------------------------------------
# small resolution helpers
# ---------------------------------------------------------------------------


def _normalize_task_output(session: Session, run_id: str) -> dict[str, Any]:
    task = session.execute(
        select(Task).where(Task.run_id == run_id, Task.step == Step.NORMALIZE_EPISODE.value)
    ).scalar_one_or_none()
    if task is None or not task.output_json:
        return {}
    return dict(task.output_json)


def _first_value(sources: Sequence[dict[str, Any]], keys: Iterable[str]) -> Any:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _resolve_canonical_keys(episode: Episode, normalize_output: dict[str, Any]) -> dict[str, str]:
    fallback = canonical_keys(episode.id, episode.canonical_version)
    canonical = normalize_output.get("canonical") if isinstance(normalize_output.get("canonical"), dict) else {}
    sources = [canonical, normalize_output]
    return {
        "metadata": _first_value(sources, _METADATA_KEY_CANDIDATES) or fallback["metadata"],
        "samples": _first_value(sources, _SAMPLES_KEY_CANDIDATES) or fallback["samples"],
        "video": _first_value(sources, _VIDEO_KEY_CANDIDATES) or fallback["video"],
        "thumbnail": _first_value(sources, _THUMBNAIL_KEY_CANDIDATES) or fallback["thumbnail"],
    }


def _read_metadata(ctx: HandlerContext, key: str) -> dict[str, Any]:
    if not key:
        return {}
    try:
        if not ctx.storage.exists(key):
            return {}
        return json.loads(ctx.storage.get_bytes(key).decode("utf-8"))
    except (ApiError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _load_samples_table(ctx: HandlerContext, key: str) -> pa.Table | None:
    if not key:
        return None
    try:
        if not ctx.storage.exists(key):
            return None
    except ApiError:
        return None
    target = Path(ctx.workspace) / "samples.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        ctx.storage.download_to(key, target)
        return read_samples_parquet(target)
    except ApiError as exc:
        if exc.code == ErrorCode.OBJECT_NOT_FOUND:
            return None
        raise
    except (OSError, pa.ArrowException) as exc:
        raise TerminalTaskError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical samples.parquet cannot be read; the Episode cannot be quality checked",
            details={"samples_key": key},
        ) from exc


def _column_lists(table: pa.Table | None, column: str) -> list[list[float]]:
    if table is None or column not in table.column_names:
        return []
    rows = table.column(column).to_pylist()
    return [ [float(value) for value in row] if row is not None else [] for row in rows ]


def _column_ints(table: pa.Table | None, column: str) -> list[int]:
    if table is None or column not in table.column_names:
        return []
    return [int(value) if value is not None else 0 for value in table.column(column).to_pylist()]


def _coerce_stream_times(value: Any) -> dict[str, list[int]]:
    times: dict[str, list[int]] = {}
    if not isinstance(value, dict):
        return times
    for name, raw in value.items():
        if not isinstance(raw, (list, tuple)):
            continue
        try:
            times[str(name)] = [int(item) for item in raw]
        except (TypeError, ValueError):
            continue
    return times


def _coerce_grid(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _coerce_int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _resolve_semantics_verified(
    normalize_output: dict[str, Any], metadata: dict[str, Any]
) -> bool | None:
    """Resolve the declared/verified semantics flag for LeRobot robot episodes.

    An explicit boolean from the normalizer or metadata always wins (the
    NORMALIZE_EPISODE handler stores it under ``provenance``).  Otherwise a
    synthetic fixture whose manifest declares ``arm6_joint_position_v1`` counts as
    verified provenance (the fixture manifest is the source evidence); public or
    real captures need an explicit ``semantics_verified=true``.
    """
    provenance = metadata.get("provenance")
    sources: list[dict[str, Any]] = [normalize_output, metadata]
    if isinstance(provenance, dict):
        sources.insert(1, provenance)
    for source in sources:
        value = source.get("semantics_verified")
        if isinstance(value, bool):
            return value
    robot_profile = (
        metadata.get("robot_profile")
        or (provenance.get("robot_profile") if isinstance(provenance, dict) else "")
        or normalize_output.get("robot_profile")
        or ""
    )
    if robot_profile != ROBOT_PROFILE:
        return None
    data_origin = metadata.get("data_origin") or normalize_output.get("data_origin") or ""
    if data_origin == "synthetic":
        return True
    return None


# ---------------------------------------------------------------------------
# video probe
# ---------------------------------------------------------------------------


def _probe_canonical_video(ctx: HandlerContext, *, video_key: str) -> VideoInfo:
    """Decode the canonical 10 fps video and collect frame / blur statistics.

    Uses OpenCV (already a project dependency) rather than ffmpeg subprocesses:
    the canonical video was produced and fully decoded by the normalizer, this is
    a bounded re-check that mainly feeds ``BLUR_FRAME_RATE``.  All failures are
    reported as evidence, never raised; the caller decides whether they escalate
    to ``FILE_DECODE_ERROR`` or only make the blur heuristic unavailable.
    """
    try:
        exists = bool(video_key) and ctx.storage.exists(video_key)
    except ApiError:
        exists = False
    if not exists:
        return VideoInfo(expected=True, decode_ok=None, error="canonical video object is absent")
    try:
        import cv2  # noqa: PLC0415 - optional heavy import, only for video episodes
    except ImportError:  # pragma: no cover - opencv is a project dependency
        return VideoInfo(expected=True, decode_ok=None, error="OpenCV is not available in this worker")
    target = Path(ctx.workspace) / "front.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        ctx.storage.download_to(video_key, target)
    except ApiError as exc:
        return VideoInfo(expected=True, decode_ok=None, error=str(exc.message))
    capture = cv2.VideoCapture(str(target))
    if not capture.isOpened():
        capture.release()
        return VideoInfo(expected=True, decode_ok=None, error="OpenCV could not open the canonical video")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    variances: list[float] = []
    frame_times: list[int] = []
    decoded = 0
    truncated = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded >= _MAX_VIDEO_FRAMES:
                truncated = True
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variances.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            frame_times.append(decoded * SAMPLE_INTERVAL_NS)
            decoded += 1
    finally:
        capture.release()
    return VideoInfo(
        expected=True,
        decode_ok=decoded > 0,
        frame_count=reported_frames or decoded,
        decoded_frames=decoded,
        fps=fps or None,
        frame_t_ns=frame_times if not truncated else None,
        laplacian_variances=variances if not truncated else None,
        error="canonical video exceeds the bounded decode analysis limit" if truncated else "",
        source="opencv",
    )


def _resolve_video_info(
    ctx: HandlerContext,
    *,
    episode: Episode,
    canonical: dict[str, str],
    metadata: dict[str, Any],
    normalize_output: dict[str, Any],
    normalization_valid: bool,
) -> VideoInfo:
    """Combine normalizer decode evidence with the canonical-video blur probe.

    The source decode verdict belongs to ``NORMALIZE_EPISODE`` (SPEC 6.1): its
    ``decoded_frame_count`` and ``FILE_DECODE_ERROR`` diagnostics are the
    authoritative evidence.  The canonical video probe only adds blur statistics
    and can never turn a healthy normalisation into a decode error merely because
    a worker lacks a codec.  If normalisation is structurally invalid, no
    canonical video is expected and the blur rule is simply unavailable.
    """
    if episode.episode_type not in {
        EpisodeType.ROBOT_EPISODE.value,
        EpisodeType.EGO_VIDEO_EPISODE.value,
    }:  # pragma: no cover - only two episode types exist
        return VideoInfo(expected=False)

    diagnostics = normalize_output.get("diagnostics") or metadata.get("diagnostics") or []
    diagnostic_codes = {
        str(item.get("code")) for item in diagnostics if isinstance(item, dict) and item.get("code")
    }
    decoded_frame_count = normalize_output.get("decoded_frame_count")
    if not isinstance(decoded_frame_count, int):
        decoded_frame_count = metadata.get("decoded_frame_count")
    normalizer_frame_count = decoded_frame_count if isinstance(decoded_frame_count, int) else None

    decode_ok: bool | None = None
    error = ""
    if "FILE_DECODE_ERROR" in diagnostic_codes:
        decode_ok = False
        error = "NORMALIZE_EPISODE reported FILE_DECODE_ERROR"
    elif normalizer_frame_count is not None:
        decode_ok = normalizer_frame_count > 0
        if not decode_ok:
            error = "NORMALIZE_EPISODE decoded zero frames"

    video_key = canonical.get("video") or ""
    if not normalization_valid:
        # no publishable canonical video is expected for a structurally invalid
        # episode; keep the normalizer's decode verdict and skip blur analysis
        return VideoInfo(
            expected=True,
            decode_ok=decode_ok,
            decoded_frames=normalizer_frame_count,
            frame_count=normalizer_frame_count,
            error=error or "normalisation invalid; canonical video not published",
        )
    if not video_key:
        return VideoInfo(
            expected=True,
            decode_ok=False if decode_ok is None else decode_ok,
            decoded_frames=normalizer_frame_count,
            frame_count=normalizer_frame_count or 0,
            error="normalisation was valid but metadata declares no canonical video",
        )
    probe = _probe_canonical_video(ctx, video_key=video_key)
    if decode_ok is None:
        decode_ok = probe.decode_ok
    return VideoInfo(
        expected=True,
        decode_ok=decode_ok,
        frame_count=probe.frame_count or normalizer_frame_count,
        decoded_frames=(
            probe.decoded_frames if probe.decoded_frames is not None else normalizer_frame_count
        ),
        fps=probe.fps,
        frame_t_ns=probe.frame_t_ns,
        laplacian_variances=probe.laplacian_variances,
        error=error or probe.error,
        source=probe.source or "normalizer",
    )


# ---------------------------------------------------------------------------
# QualityInput assembly
# ---------------------------------------------------------------------------


def build_quality_input(
    ctx: HandlerContext,
    *,
    episode: Episode,
    normalize_output: dict[str, Any],
    source_type: str | None = None,
) -> QualityInput:
    """Assemble the pure :class:`QualityInput` from DB + object-store evidence."""
    canonical = _resolve_canonical_keys(episode, normalize_output)
    metadata = _read_metadata(ctx, canonical["metadata"])
    table = _load_samples_table(ctx, canonical["samples"])

    alignment = _first_value([normalize_output, metadata], _ALIGNMENT_KEY_CANDIDATES)
    alignment = alignment if isinstance(alignment, dict) else {}
    diagnostics = normalize_output.get("diagnostics") or metadata.get("diagnostics") or []
    normalization_valid = normalize_output.get("normalization_valid")
    if not isinstance(normalization_valid, bool):
        normalization_valid = metadata.get("normalization_valid", True)
    normalization_valid = bool(normalization_valid)

    stream_times = _coerce_stream_times(
        _first_value([normalize_output, alignment, metadata], _STREAM_TIMES_KEY_CANDIDATES)
    )
    grid_t_ns = _coerce_grid(_first_value([normalize_output, alignment, metadata], _GRID_KEY_CANDIDATES))
    origin_ns = int(
        _first_value([normalize_output, alignment, metadata], ("origin_ns",)) or 0
    )

    frame_errors = _coerce_int_list(alignment.get("frame_errors_ns"))
    state_errors = _coerce_int_list(alignment.get("state_errors_ns"))
    action_ages = _coerce_int_list(alignment.get("action_ages_ns"))
    max_sync = _coerce_float(alignment.get("max_abs_sync_error_ms"))
    p95_sync = _coerce_float(alignment.get("p95_abs_sync_error_ms"))
    if table is not None and max_sync is None:
        max_sync = _coerce_float(normalize_output.get("max_abs_sync_error_ms"))
        p95_sync = _coerce_float(normalize_output.get("p95_abs_sync_error_ms"))

    t_ns = _column_ints(table, "t_ns")
    state = _column_lists(table, "state")
    action = _column_lists(table, "action")
    state_source = _column_ints(table, "state_source_t_ns")
    action_source = _column_ints(table, "action_source_t_ns")

    state_dim = _first_value([normalize_output, metadata], ("state_dim",))
    action_dim = _first_value([normalize_output, metadata], ("action_dim",))
    if state_dim is None and state:
        state_dim = len(state[0])
    if action_dim is None and action:
        action_dim = len(action[0])
    joint_names = _first_value([normalize_output, metadata], ("joint_names",))
    if joint_names is not None and not isinstance(joint_names, (list, tuple)):
        joint_names = None
    joint_names = list(joint_names) if joint_names is not None else None

    robot_profile = str(
        _first_value([metadata, normalize_output], ("robot_profile",)) or ""
    )
    if not robot_profile and episode.episode_type == EpisodeType.ROBOT_EPISODE.value:
        robot_profile = str(metadata.get("robot_profile") or "")

    video = _resolve_video_info(
        ctx,
        episode=episode,
        canonical=canonical,
        metadata=metadata,
        normalize_output=normalize_output,
        normalization_valid=normalization_valid,
    )

    data_origin = str(metadata.get("data_origin") or episode.data_origin or "")
    duration_ns = metadata.get("duration_ns")
    if not isinstance(duration_ns, int):
        duration_ns = episode.duration_ns or 0

    return QualityInput(
        episode_id=episode.id,
        episode_type=episode.episode_type,
        source_type=source_type or str(metadata.get("source_type") or ""),
        data_origin=data_origin,
        duration_ns=int(duration_ns),
        policy_version=QC_POLICY_VERSION,
        normalization_valid=normalization_valid,
        metadata=metadata,
        origin_ns=origin_ns,
        t_ns=t_ns,
        state=state,
        action=action,
        state_source_t_ns=state_source,
        action_source_t_ns=action_source,
        joint_names=joint_names,
        state_dim=int(state_dim) if isinstance(state_dim, int) else None,
        action_dim=int(action_dim) if isinstance(action_dim, int) else None,
        stream_times=stream_times,
        grid_t_ns=grid_t_ns,
        missing_frame_indexes=_coerce_int_list(alignment.get("missing_frame_indexes")),
        missing_state_indexes=_coerce_int_list(alignment.get("missing_state_indexes")),
        stale_action_indexes=_coerce_int_list(alignment.get("stale_action_indexes")),
        frame_errors_ns=frame_errors,
        state_errors_ns=state_errors,
        action_ages_ns=action_ages,
        max_abs_sync_error_ms=max_sync,
        p95_abs_sync_error_ms=p95_sync,
        video=video,
        annotations=None,
        semantics_verified=_resolve_semantics_verified(normalize_output, metadata),
        robot_profile=robot_profile,
        structural_diagnostics=[
            item
            for item in diagnostics
            if isinstance(item, dict) and bool(item.get("structural", True))
        ],
        extra={
            "normalize_diagnostics": [
                item.get("code") for item in diagnostics if isinstance(item, dict) and item.get("code")
            ],
            "episode_status": episode.status,
        },
    )


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------


class QualityCheckHandler(TaskHandler):
    """Run ``qc-policy/1.0`` and persist an immutable quality report."""

    step = Step.QUALITY_CHECK

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        task = ctx.task
        if not task.run_id:
            raise TerminalTaskError(
                ErrorCode.INVALID_INPUT,
                "QUALITY_CHECK requires an Episode-level PipelineRun (tasks.run_id is NULL)",
            )
        run = ctx.session.get(PipelineRun, task.run_id)
        if run is None:
            raise TerminalTaskError(ErrorCode.NOT_FOUND, "PipelineRun for this QUALITY_CHECK task was not found")
        episode = ctx.session.get(Episode, run.episode_id)
        if episode is None:
            raise TerminalTaskError(ErrorCode.NOT_FOUND, "Episode for this PipelineRun was not found")
        if ctx.is_cancelled():
            raise CancelledError("QUALITY_CHECK cancelled before execution")

        # V2 (SPEC_ADDENDUM_V2 §3): canonical-episode/2.0 or pipeline_version 2.x
        # Episodes are evaluated with qc-policy/2.0 bound to their robot profile.
        # The import is lazy so the V1 path never depends on the V2 modules.
        from embodied_dataops.workflow.handlers.quality_check_v2 import (
            is_v2_quality_pipeline,
            run_quality_check_v2,
        )

        if is_v2_quality_pipeline(run, episode):
            source_import_v2 = (
                ctx.session.get(SourceImport, episode.source_import_id)
                if episode.source_import_id
                else None
            )
            return run_quality_check_v2(
                ctx, run=run, episode=episode, source_import=source_import_v2
            )

        source_import = (
            ctx.session.get(SourceImport, episode.source_import_id)
            if episode.source_import_id
            else None
        )
        normalize_output = _normalize_task_output(ctx.session, run.id)
        quality_input = build_quality_input(
            ctx,
            episode=episode,
            normalize_output=normalize_output,
            source_type=source_import.source_type if source_import is not None else None,
        )
        evaluation: QualityEvaluation = evaluate_quality(quality_input)
        report_payload = build_quality_report(
            episode_id=episode.id,
            run_id=run.id,
            evaluation=evaluation,
            input=quality_input,
        )
        report, asset = persist_quality_report(
            ctx,
            episode=episode,
            run_id=run.id,
            evaluation=evaluation,
            report_payload=report_payload,
        )
        verdict = evaluation.verdict.value
        data = {
            "quality_report_id": report.id,
            "quality_report_key": asset.object_key,
            "quality_report_asset_id": asset.id,
            "policy_version": quality_input.policy_version,
            "verdict": verdict,
            "finding_count": len(evaluation.findings),
            "rule_codes": [finding.rule_code.value for finding in evaluation.findings],
            "finding_counts": evaluation.counts_by_severity(),
            "checks": {
                "run": evaluation.checks_run,
                "unavailable": evaluation.checks_unavailable,
                "not_applicable": evaluation.checks_not_applicable,
            },
            "metrics": evaluation.metrics,
            "normalization_valid": quality_input.normalization_valid,
            "episode_quality_status": episode.quality_status,
            "episode_status": episode.status,
        }
        # ``persist_quality_report`` already moved the Episode to QUALITY_FAILED
        # through the report persistence path, so no duplicate transition is
        # requested from the engine (which would try QUALITY_FAILED -> QUALITY_FAILED
        # and log an illegal transition).  The FAIL verdict is what makes the
        # engine fail the run and stop downstream steps (SPEC 6.1).
        return HandlerResult(
            data=data,
            verdict=verdict,
            episode_status=None,
            output_asset_id=asset.id,
        )


__all__ = [
    "QualityCheckHandler",
    "build_quality_input",
]
