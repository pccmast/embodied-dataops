"""Pure quality rule functions for ``qc-policy/1.0`` (SPEC 7.1).

Every rule is a plain function of a :class:`QualityInput`; nothing here touches
the database, the object store or the clock.  A rule returns zero or more
:class:`Finding` objects carrying ``rule_code``, ``severity``, ``observed``,
``threshold``, ``affected_time_ranges`` and machine-readable ``evidence`` (SPEC
7.1: findings must be clickable evidence, not a boolean).

Missing evidence is never invented.  When a rule's input is not available (for
example no decoded video for an Episode normalised with
``normalization_valid=false``) the rule is reported as *unavailable* rather than
silently passing.

Threshold boundaries (SPEC 4.2) are tested exactly: image/state tolerances use
``<= 50 ms``, action hold ``<= 100 ms``, while frame gaps (> 200 ms), blurry
frame ratio (> 30 %), duration (> 600 s) and low confidence (< 0.35) use their
declared strict comparisons.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.canonical.alignment import compute_p95
from embodied_dataops.domain.enums import EpisodeType, QualityStatus, Severity, SourceType
from embodied_dataops.quality.policy import (
    QC_POLICY_VERSION,
    THRESHOLDS,
    QcThresholds,
    RuleCode,
    decide_verdict,
    rule_definition,
)

#: Cap on the number of affected ranges embedded in one finding (evidence is
#: bounded; counts remain exact in ``observed``).
MAX_AFFECTED_RANGES = 200
MAX_EVIDENCE_INDEXES = 500


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeRange:
    """One affected episode-relative time window."""

    start_ns: int
    end_ns: int

    def as_dict(self) -> dict[str, int]:
        return {"start_ns": int(self.start_ns), "end_ns": int(self.end_ns)}


@dataclass(frozen=True)
class Finding:
    """One machine-readable rule violation (SPEC 7.1)."""

    rule_code: RuleCode
    severity: Severity
    observed: dict[str, Any]
    threshold: dict[str, Any]
    affected_time_ranges: tuple[TimeRange, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_code": self.rule_code.value,
            "severity": self.severity.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "affected_time_ranges": [item.as_dict() for item in self.affected_time_ranges],
            "evidence": self.evidence,
        }


@dataclass
class VideoInfo:
    """Decoded-video facts used by FILE_DECODE_ERROR and BLUR_FRAME_RATE."""

    expected: bool = False
    decode_ok: bool | None = None
    frame_count: int | None = None
    decoded_frames: int | None = None
    fps: float | None = None
    frame_t_ns: list[int] | None = None
    laplacian_variances: list[float] | None = None
    error: str = ""
    source: str = ""

    def blurry_indexes(self, threshold: float) -> list[int]:
        if not self.laplacian_variances:
            return []
        return [index for index, value in enumerate(self.laplacian_variances) if value < threshold]


@dataclass
class AnnotationInfo:
    """Detection facts for EMPTY_ANNOTATION and LOW_CONF_DETECTION."""

    frame_count: int = 0
    total_detections: int | None = None
    confidences: list[float] | None = None
    frame_indexes: list[int] | None = None
    detection_frame_indexes: list[int] | None = None
    model_id: str = ""


@dataclass
class QualityInput:
    """Everything the rules may look at, assembled by the QUALITY_CHECK handler.

    ``t_ns``/``state``/``action`` are the canonical 10 Hz rows that were actually
    written (only fully valid points).  ``grid_t_ns`` is the *intended* grid;
    the difference is what ``missing_sample_count`` measures.  ``stream_times``
    holds absolute source timestamps per stream (``camera``/``state``/``action``)
    for the timestamp and frame-gap rules.
    """

    episode_id: str
    episode_type: str = EpisodeType.ROBOT_EPISODE.value
    source_type: str = ""
    data_origin: str = ""
    duration_ns: int = 0
    policy_version: str = QC_POLICY_VERSION
    normalization_valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    #: absolute origin used to convert source timestamps to episode-relative ones
    origin_ns: int = 0

    # canonical samples (only fully valid rows, never zero-filled)
    t_ns: list[int] = field(default_factory=list)
    state: list[list[float]] = field(default_factory=list)
    action: list[list[float]] = field(default_factory=list)
    state_source_t_ns: list[int] = field(default_factory=list)
    action_source_t_ns: list[int] = field(default_factory=list)
    joint_names: list[str] | None = None
    state_dim: int | None = None
    action_dim: int | None = None

    # source stream diagnostics
    stream_times: dict[str, list[int]] = field(default_factory=dict)
    grid_t_ns: list[int] = field(default_factory=list)
    missing_frame_indexes: list[int] = field(default_factory=list)
    missing_state_indexes: list[int] = field(default_factory=list)
    stale_action_indexes: list[int] = field(default_factory=list)
    frame_errors_ns: list[int] = field(default_factory=list)
    state_errors_ns: list[int] = field(default_factory=list)
    action_ages_ns: list[int] = field(default_factory=list)
    max_abs_sync_error_ms: float | None = None
    p95_abs_sync_error_ms: float | None = None

    # optional evidence sources
    video: VideoInfo = field(default_factory=VideoInfo)
    annotations: AnnotationInfo | None = None
    semantics_verified: bool | None = None
    robot_profile: str = ""
    #: structural diagnostics emitted by NORMALIZE_EPISODE.  When one exists the
    #: episode has no canonical samples; the codes are carried into the report so a
    #: 0-sample episode cannot silently pass (see :func:`carried_findings`).
    structural_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived helpers ---------------------------------------------------
    @property
    def is_robot(self) -> bool:
        return self.episode_type == EpisodeType.ROBOT_EPISODE.value

    def resolved_state_dim(self) -> int | None:
        if self.state_dim:
            return int(self.state_dim)
        if self.state:
            return len(self.state[0])
        return None

    def resolved_action_dim(self) -> int | None:
        if self.action_dim:
            return int(self.action_dim)
        if self.action:
            return len(self.action[0])
        return None

    def missing_grid_indexes(self) -> list[int]:
        return sorted(
            set(self.missing_frame_indexes)
            | set(self.missing_state_indexes)
            | set(self.stale_action_indexes)
        )

    def grid_length(self) -> int:
        if self.grid_t_ns:
            return len(self.grid_t_ns)
        return len(self.t_ns)


@dataclass
class QualityEvaluation:
    """Deterministic outcome of running every applicable rule once."""

    verdict: QualityStatus
    findings: list[Finding]
    metrics: dict[str, Any]
    checks_run: list[str]
    checks_unavailable: list[str]
    checks_not_applicable: list[str]

    def findings_as_dicts(self) -> list[dict[str, Any]]:
        return [finding.as_dict() for finding in self.findings]

    def counts_by_severity(self) -> dict[str, int]:
        counts = {Severity.INFO.value: 0, Severity.WARN.value: 0, Severity.ERROR.value: 0}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return counts


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _affected_grid_ranges(inp: QualityInput, indexes: Iterable[int]) -> tuple[TimeRange, ...]:
    times = inp.grid_t_ns if inp.grid_t_ns else inp.t_ns
    ranges: list[TimeRange] = []
    for index in list(indexes)[:MAX_AFFECTED_RANGES]:
        if 0 <= index < len(times):
            ranges.append(TimeRange(int(times[index]), int(times[index]) + SAMPLE_INTERVAL_NS))
    return tuple(ranges)


def _affected_index_ranges(
    inp: QualityInput, indexes: Iterable[int], times: Sequence[int] | None
) -> tuple[TimeRange, ...]:
    ranges: list[TimeRange] = []
    for index in list(indexes)[:MAX_AFFECTED_RANGES]:
        if times is not None and 0 <= index < len(times):
            start = int(times[index])
            ranges.append(TimeRange(start, start + SAMPLE_INTERVAL_NS))
    return tuple(ranges)


def _relative(inp: QualityInput, source_time_ns: int) -> int:
    return int(source_time_ns) - int(inp.origin_ns) if inp.origin_ns else int(source_time_ns)


def _bound(indexes: Iterable[int]) -> list[int]:
    return list(indexes)[:MAX_EVIDENCE_INDEXES]


def _finding_from_rule(
    rule_code: RuleCode,
    *,
    severity: Severity | None,
    observed: dict[str, Any],
    affected_time_ranges: tuple[TimeRange, ...] = (),
    evidence: dict[str, Any] | None = None,
) -> Finding:
    definition = rule_definition(rule_code)
    return Finding(
        rule_code=rule_code,
        severity=severity or definition.severity,
        observed=observed,
        threshold=dict(definition.threshold),
        affected_time_ranges=affected_time_ranges,
        evidence=evidence or {},
    )


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def check_file_decode_error(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """FILE_DECODE_ERROR: full decode failed, or zero decodable frames."""
    video = inp.video
    if not video.expected:
        return []
    failed = video.decode_ok is False
    empty = video.decoded_frames is not None and video.decoded_frames < thresholds.min_video_frames
    zero_probe = video.decoded_frames == 0 or (video.frame_count is not None and video.frame_count == 0)
    if not (failed or empty or zero_probe):
        return []
    observed = {
        "decode_ok": video.decode_ok,
        "frame_count": video.frame_count,
        "decoded_frames": video.decoded_frames,
        "source": video.source,
    }
    evidence = {
        "rule": "video must be fully decodable (ffmpeg full decode) with at least one frame",
        "decode_error": video.error,
        "minimum_decodable_frames": thresholds.min_video_frames,
    }
    return [_finding_from_rule(RuleCode.FILE_DECODE_ERROR, severity=None, observed=observed, evidence=evidence)]


def check_duration_exceeded(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """DURATION_EXCEEDED: Episode duration strictly greater than 600 s."""
    if int(inp.duration_ns) <= thresholds.max_duration_ns:
        return []
    observed = {
        "duration_ns": int(inp.duration_ns),
        # no rounding: a 1 ns boundary violation must stay visible in the evidence
        "duration_s": int(inp.duration_ns) / 1_000_000_000,
        "limit_ns": thresholds.max_duration_ns,
        "limit_s": thresholds.max_duration_ns / 1_000_000_000,
    }
    return [
        _finding_from_rule(
            RuleCode.DURATION_EXCEEDED,
            severity=None,
            observed=observed,
            evidence={"comparison": "duration_ns > max_duration_ns (strict)"},
        )
    ]


def check_timestamp_non_monotonic(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """TIMESTAMP_NON_MONOTONIC: source timestamps must be strictly increasing (no duplicates)."""
    if not inp.is_robot and not inp.stream_times:
        return []
    streams: dict[str, Any] = {}
    ranges: list[TimeRange] = []
    for name in sorted(inp.stream_times):
        times = list(inp.stream_times[name])
        violations: list[int] = []
        for index in range(1, len(times)):
            if times[index] <= times[index - 1]:
                violations.append(index)
        if violations:
            streams[name] = {
                "violation_count": len(violations),
                "first_indexes": _bound(violations),
                "first_pairs": [
                    {"index": index, "prev_ns": int(times[index - 1]), "current_ns": int(times[index])}
                    for index in violations[:3]
                ],
            }
            relative_times = [_relative(inp, value) for value in times]
            ranges.extend(_affected_index_ranges(inp, violations, relative_times))
    if not streams:
        return []
    observed = {"streams": streams, "violation_count": sum(item["violation_count"] for item in streams.values())}
    evidence = {
        "comparison": "times[i] > times[i-1] (strict; duplicates are violations)",
        "clock_domains": dict(inp.metadata.get("clock_domains") or {}),
        "note": "canonical t_ns is always a contiguous 10 Hz grid; this rule checks source stream timestamps",
    }
    return [
        _finding_from_rule(
            RuleCode.TIMESTAMP_NON_MONOTONIC,
            severity=None,
            observed=observed,
            affected_time_ranges=tuple(ranges),
            evidence=evidence,
        )
    ]


def _frame_gap_windows(
    camera_times: Sequence[int], max_gap_ns: int
) -> list[dict[str, int]]:
    windows: list[dict[str, int]] = []
    for index in range(1, len(camera_times)):
        gap = int(camera_times[index]) - int(camera_times[index - 1])
        if gap > max_gap_ns:
            windows.append(
                {
                    "prev_source_t_ns": int(camera_times[index - 1]),
                    "next_source_t_ns": int(camera_times[index]),
                    "gap_ns": gap,
                }
            )
    return windows


def check_frame_gap_excess(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """FRAME_GAP_EXCESS: gap > 200 ms WARN; a missing 10 Hz grid image escalates to ERROR."""
    camera_times = inp.stream_times.get("camera") or inp.stream_times.get("video") or []
    windows = _frame_gap_windows(camera_times, thresholds.frame_gap_max_ns) if camera_times else []
    missing_frames = list(inp.missing_frame_indexes)
    if not windows and not missing_frames:
        return []
    severity = Severity.ERROR if missing_frames else Severity.WARN
    ranges: list[TimeRange] = []
    for window in windows[:MAX_AFFECTED_RANGES]:
        start = _relative(inp, window["prev_source_t_ns"])
        end = _relative(inp, window["next_source_t_ns"])
        ranges.append(TimeRange(start, end))
    if missing_frames:
        ranges.extend(_affected_grid_ranges(inp, missing_frames))
    observed = {
        "gap_windows": [
            {
                "prev_source_t_ns": window["prev_source_t_ns"],
                "next_source_t_ns": window["next_source_t_ns"],
                "gap_ns": window["gap_ns"],
                "gap_ms": round(window["gap_ns"] / 1e6, 6),
            }
            for window in windows[:100]
        ],
        "gap_window_count": len(windows),
        "max_gap_ns": max((window["gap_ns"] for window in windows), default=0),
        "missing_grid_frame_count": len(missing_frames),
        "missing_grid_frame_indexes": _bound(missing_frames),
        "camera_sample_count": len(camera_times),
    }
    evidence = {
        "comparison": "camera_gap_ns > 200 ms (strict, WARN); missing 10 Hz grid image -> ERROR",
        "escalated_to_error": bool(missing_frames),
    }
    return [
        _finding_from_rule(
            RuleCode.FRAME_GAP_EXCESS,
            severity=severity,
            observed=observed,
            affected_time_ranges=tuple(ranges),
            evidence=evidence,
        )
    ]


def _sync_finding(
    inp: QualityInput,
    rule_code: RuleCode,
    indexes: Sequence[int],
    *,
    extra_observed: dict[str, Any] | None = None,
    extra_evidence: dict[str, Any] | None = None,
) -> list[Finding]:
    if not indexes:
        return []
    observed = {
        "missing_grid_point_count": len(indexes),
        "grid_point_count": inp.grid_length(),
        "missing_grid_indexes": _bound(indexes),
    }
    if extra_observed:
        observed.update(extra_observed)
    evidence = {
        "missing_index_count": len(indexes),
        "grid_point_count": inp.grid_length(),
        "no_zero_fill": True,
        "canonical_valid_sample_count": len(inp.t_ns),
    }
    if extra_evidence:
        evidence.update(extra_evidence)
    return [
        _finding_from_rule(
            rule_code,
            severity=None,
            observed=observed,
            affected_time_ranges=_affected_grid_ranges(inp, indexes),
            evidence=evidence,
        )
    ]


def check_sync_missing_frame(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """SYNC_MISSING_FRAME: a 10 Hz grid point has no camera frame within <= 50 ms."""
    if not inp.is_robot:
        return []
    return _sync_finding(
        inp,
        RuleCode.SYNC_MISSING_FRAME,
        inp.missing_frame_indexes,
        extra_observed={"image_tolerance_ms": thresholds.image_tolerance_ns / 1e6},
        extra_evidence={"image_tolerance_ms": thresholds.image_tolerance_ns / 1e6},
    )


def check_sync_missing_state(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """SYNC_MISSING_STATE: no bracketing state pair with both endpoints within <= 50 ms."""
    if not inp.is_robot:
        return []
    return _sync_finding(
        inp,
        RuleCode.SYNC_MISSING_STATE,
        inp.missing_state_indexes,
        extra_observed={"state_tolerance_ms": thresholds.state_tolerance_ns / 1e6},
        extra_evidence={
            "state_tolerance_ms": thresholds.state_tolerance_ns / 1e6,
            "interpolation": "linear between a bracketing pair; both endpoints required",
        },
    )


def check_sync_stale_action(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """SYNC_STALE_ACTION: no action at or before the sample time within <= 100 ms."""
    if not inp.is_robot:
        return []
    return _sync_finding(
        inp,
        RuleCode.SYNC_STALE_ACTION,
        inp.stale_action_indexes,
        extra_observed={"action_zoh_tolerance_ms": thresholds.action_zoh_tolerance_ns / 1e6},
        extra_evidence={
            "action_zoh_tolerance_ms": thresholds.action_zoh_tolerance_ns / 1e6,
            "zero_order_hold": "strictly past only; future actions are never selected",
        },
    )


def check_dimension_mismatch(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """DIMENSION_MISMATCH: state/action not 6D or the six joint names are not exact."""
    if not inp.is_robot:
        return []
    state_dim = inp.resolved_state_dim()
    action_dim = inp.resolved_action_dim()
    joint_names = inp.joint_names
    expected_joint_names = list(thresholds.joint_names)
    problems: dict[str, Any] = {}
    if state_dim is not None and state_dim != thresholds.state_dim:
        problems["state_dim"] = state_dim
    if action_dim is not None and action_dim != thresholds.action_dim:
        problems["action_dim"] = action_dim
    names_mismatch = joint_names is not None and list(joint_names) != expected_joint_names
    if names_mismatch:
        problems["joint_names"] = list(joint_names or [])
    if not problems:
        return []
    observed = {
        "observed_state_dim": state_dim,
        "observed_action_dim": action_dim,
        "observed_joint_names": list(joint_names) if joint_names is not None else None,
        "problems": problems,
    }
    evidence = {
        "expected_state_dim": thresholds.state_dim,
        "expected_action_dim": thresholds.action_dim,
        "expected_joint_names": expected_joint_names,
        "robot_profile": "arm6_joint_position_v1",
        "no_truncation_or_padding": True,
    }
    if "state_dim" in problems or "action_dim" in problems:
        evidence["note"] = (
            "structural incompatibility is recorded as a diagnostic; data is never truncated or padded"
        )
    return [
        _finding_from_rule(
            RuleCode.DIMENSION_MISMATCH, severity=None, observed=observed, evidence=evidence
        )
    ]


def check_nonfinite_value(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """NONFINITE_VALUE: NaN or +-Inf in state/action."""
    if not inp.is_robot:
        return []
    if not inp.state and not inp.action:
        return []

    def _scan(rows: Sequence[Sequence[float]]) -> tuple[int, list[int], list[str]]:
        count = 0
        indexes: list[int] = []
        kinds: set[str] = set()
        for index, row in enumerate(rows):
            bad = False
            for value in row:
                number = float(value)
                if math.isnan(number):
                    kinds.add("NaN")
                    bad = True
                elif math.isinf(number):
                    kinds.add("+Inf" if number > 0 else "-Inf")
                    bad = True
            if bad:
                count += 1
                indexes.append(index)
        return count, indexes, sorted(kinds)

    state_count, state_indexes, state_kinds = _scan(inp.state)
    action_count, action_indexes, action_kinds = _scan(inp.action)
    if state_count == 0 and action_count == 0:
        return []
    observed = {
        "state": {
            "nonfinite_row_count": state_count,
            "row_indexes": _bound(state_indexes),
            "kinds": state_kinds,
        },
        "action": {
            "nonfinite_row_count": action_count,
            "row_indexes": _bound(action_indexes),
            "kinds": action_kinds,
        },
        "nonfinite_row_count": state_count + action_count,
    }
    ranges = _affected_index_ranges(inp, state_indexes + action_indexes, inp.t_ns)
    evidence = {
        "forbidden_values": ["NaN", "+Inf", "-Inf"],
        "checked_columns": {"state": "state", "action": "action"},
        "sample_time_source": "canonical t_ns",
    }
    return [
        _finding_from_rule(
            RuleCode.NONFINITE_VALUE,
            severity=None,
            observed=observed,
            affected_time_ranges=ranges,
            evidence=evidence,
        )
    ]


def check_unverified_semantics(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """UNVERIFIED_SEMANTICS: LeRobot robot episode without proof of arm6 joint semantics."""
    if not inp.is_robot or inp.source_type != SourceType.LEROBOT_V3.value:
        return []
    if inp.semantics_verified is True:
        return []
    robot_profile = inp.robot_profile or inp.metadata.get("robot_profile") or ""
    observed = {
        "semantics_verified": inp.semantics_verified,
        "robot_profile": robot_profile,
        "source_type": inp.source_type,
    }
    evidence = {
        "required_profile": "arm6_joint_position_v1",
        "required_semantics": (
            "state = joint positions (rad) for joint_1..joint_6; "
            "action = target joint positions (rad), not torque/velocity"
        ),
        "requires_source_evidence": True,
        "publishable": False,
    }
    return [
        _finding_from_rule(
            RuleCode.UNVERIFIED_SEMANTICS, severity=None, observed=observed, evidence=evidence
        )
    ]


def check_empty_annotation(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """EMPTY_ANNOTATION: zero detections (INFO; never changes the verdict)."""
    annotations = inp.annotations
    if annotations is None or annotations.total_detections is None:
        return []
    if annotations.total_detections > 0:
        return []
    observed = {
        "total_detections": annotations.total_detections,
        "frame_count": annotations.frame_count,
        "model_id": annotations.model_id,
    }
    evidence = {
        "informational_only": True,
        "verdict_impact": "none (INFO)",
        "note": "an empty pre-annotation result is still a valid review artefact",
    }
    return [
        _finding_from_rule(
            RuleCode.EMPTY_ANNOTATION, severity=None, observed=observed, evidence=evidence
        )
    ]


def check_low_conf_detection(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """LOW_CONF_DETECTION: confidence < 0.35 (INFO; excluded from official candidates)."""
    annotations = inp.annotations
    if annotations is None or not annotations.confidences:
        return []
    low_indexes = [
        index
        for index, confidence in enumerate(annotations.confidences)
        if float(confidence) < thresholds.low_conf_threshold
    ]
    if not low_indexes:
        return []
    low_values = [float(annotations.confidences[index]) for index in low_indexes]
    frame_indexes = None
    if annotations.detection_frame_indexes is not None:
        frame_indexes = [
            int(annotations.detection_frame_indexes[index])
            for index in low_indexes
            if index < len(annotations.detection_frame_indexes)
        ]
    observed = {
        "detection_count": len(annotations.confidences),
        "low_conf_count": len(low_indexes),
        "min_confidence": round(min(low_values), 6),
        "confidence_threshold": thresholds.low_conf_threshold,
        "excluded_from_candidates": True,
    }
    evidence = {
        "informational_only": True,
        "verdict_impact": "none (INFO)",
        "detection_indexes": _bound(low_indexes),
        "frame_indexes": frame_indexes[:MAX_EVIDENCE_INDEXES] if frame_indexes else [],
    }
    return [
        _finding_from_rule(
            RuleCode.LOW_CONF_DETECTION, severity=None, observed=observed, evidence=evidence
        )
    ]


def check_blur_frame_rate(inp: QualityInput, thresholds: QcThresholds = THRESHOLDS) -> list[Finding]:
    """BLUR_FRAME_RATE: > 30 % of frames with Laplacian variance < 40 (WARN)."""
    variances = inp.video.laplacian_variances
    if not variances:
        return []
    total = len(variances)
    blurry = inp.video.blurry_indexes(thresholds.blur_laplacian_variance_threshold)
    ratio = len(blurry) / total
    if not ratio > thresholds.blur_frame_ratio_threshold:
        return []
    ranges: tuple[TimeRange, ...] = ()
    if inp.video.frame_t_ns is not None:
        times = inp.video.frame_t_ns
        selected: list[TimeRange] = []
        for index in blurry[:MAX_AFFECTED_RANGES]:
            if 0 <= index < len(times):
                selected.append(TimeRange(int(times[index]), int(times[index]) + SAMPLE_INTERVAL_NS))
        ranges = tuple(selected)
    observed = {
        "frame_count": total,
        "blurry_frame_count": len(blurry),
        "blurry_frame_ratio": round(ratio, 6),
        "laplacian_variance_threshold": thresholds.blur_laplacian_variance_threshold,
        "blurry_frame_ratio_threshold": thresholds.blur_frame_ratio_threshold,
        "min_laplacian_variance": round(min(variances), 6),
        "mean_laplacian_variance": round(sum(variances) / total, 6),
    }
    evidence = {
        "heuristic": True,
        "not_a_quality_label": "Laplacian variance is a heuristic; it must not be presented as a real quality label",
        "frame_size": "640x480 grayscale",
        "comparison": "blurry_frame_ratio > 0.30 (strict)",
        "blurry_frame_indexes": _bound(blurry),
    }
    return [
        _finding_from_rule(
            RuleCode.BLUR_FRAME_RATE,
            severity=None,
            observed=observed,
            affected_time_ranges=ranges,
            evidence=evidence,
        )
    ]


#: Rules in deterministic evaluation/report order.
RULE_FUNCTIONS = (
    (RuleCode.FILE_DECODE_ERROR, check_file_decode_error),
    (RuleCode.DURATION_EXCEEDED, check_duration_exceeded),
    (RuleCode.TIMESTAMP_NON_MONOTONIC, check_timestamp_non_monotonic),
    (RuleCode.FRAME_GAP_EXCESS, check_frame_gap_excess),
    (RuleCode.SYNC_MISSING_FRAME, check_sync_missing_frame),
    (RuleCode.SYNC_MISSING_STATE, check_sync_missing_state),
    (RuleCode.SYNC_STALE_ACTION, check_sync_stale_action),
    (RuleCode.DIMENSION_MISMATCH, check_dimension_mismatch),
    (RuleCode.NONFINITE_VALUE, check_nonfinite_value),
    (RuleCode.UNVERIFIED_SEMANTICS, check_unverified_semantics),
    (RuleCode.EMPTY_ANNOTATION, check_empty_annotation),
    (RuleCode.LOW_CONF_DETECTION, check_low_conf_detection),
    (RuleCode.BLUR_FRAME_RATE, check_blur_frame_rate),
)


# ---------------------------------------------------------------------------
# metrics + evaluation
# ---------------------------------------------------------------------------


def _sync_metrics(inp: QualityInput) -> dict[str, Any]:
    grid_len = inp.grid_length()
    missing_indexes = inp.missing_grid_indexes()
    if grid_len:
        missing = len(missing_indexes)
        valid = grid_len - missing
    else:
        missing = 0
        valid = len(inp.t_ns)
    per_point_ms: list[float] = []
    if inp.max_abs_sync_error_ms is None or inp.p95_abs_sync_error_ms is None:
        count = min(len(inp.frame_errors_ns), len(inp.state_errors_ns))
        for index in range(count):
            age = inp.action_ages_ns[index] if index < len(inp.action_ages_ns) else 0
            per_point_ms.append(
                max(inp.frame_errors_ns[index], inp.state_errors_ns[index], age) / 1e6
            )
    max_error = (
        round(float(inp.max_abs_sync_error_ms), 6)
        if inp.max_abs_sync_error_ms is not None
        else (round(max(per_point_ms), 6) if per_point_ms else 0.0)
    )
    p95_error = (
        round(float(inp.p95_abs_sync_error_ms), 6)
        if inp.p95_abs_sync_error_ms is not None
        else round(compute_p95(per_point_ms), 6)
    )
    return {
        "valid_sample_count": int(valid),
        "missing_sample_count": int(missing),
        "max_abs_sync_error_ms": max_error,
        "p95_abs_sync_error_ms": p95_error,
        "grid_point_count": int(grid_len),
        "missing_grid_index_count": len(missing_indexes),
        "missing_frame_count": len(inp.missing_frame_indexes),
        "missing_state_count": len(inp.missing_state_indexes),
        "stale_action_count": len(inp.stale_action_indexes),
        "frame_error_sample_count": len(inp.frame_errors_ns),
    }


def _video_metrics(inp: QualityInput, thresholds: QcThresholds) -> dict[str, Any]:
    video = inp.video
    metrics: dict[str, Any] = {
        "expected": bool(video.expected),
        "checked": video.decode_ok is not None,
        "decode_ok": video.decode_ok,
        "frame_count": video.frame_count,
        "decoded_frames": video.decoded_frames,
        "blur_checked": bool(video.laplacian_variances),
    }
    if video.laplacian_variances:
        total = len(video.laplacian_variances)
        blurry = len(video.blurry_indexes(thresholds.blur_laplacian_variance_threshold))
        metrics.update(
            {
                "blurry_frame_count": blurry,
                "blurry_frame_ratio": round(blurry / total, 6),
                "laplacian_variance_threshold": thresholds.blur_laplacian_variance_threshold,
            }
        )
    if video.error:
        metrics["decode_error"] = video.error
    return metrics


def _annotation_metrics(inp: QualityInput) -> dict[str, Any] | None:
    annotations = inp.annotations
    if annotations is None:
        return None
    return {
        "frame_count": annotations.frame_count,
        "total_detections": annotations.total_detections,
        "confidence_sample_count": len(annotations.confidences or []),
        "model_id": annotations.model_id,
    }


def _classify_checks(inp: QualityInput, findings: list[Finding]) -> tuple[list[str], list[str], list[str]]:
    """Report which rules ran with usable evidence, which lacked evidence, which do not apply."""
    fired = {finding.rule_code.value for finding in findings}
    run: list[str] = []
    unavailable: list[str] = []
    not_applicable: list[str] = []

    def filled(code: RuleCode, available: bool, applicable: bool = True) -> None:
        if not applicable:
            not_applicable.append(code.value)
        elif available:
            run.append(code.value)
        else:
            unavailable.append(code.value)

    video = inp.video
    has_camera_times = bool(inp.stream_times.get("camera") or inp.stream_times.get("video"))
    has_grid_evidence = bool(inp.grid_t_ns or inp.missing_frame_indexes or inp.missing_state_indexes or inp.stale_action_indexes)
    has_samples = bool(inp.t_ns)
    has_dims = inp.resolved_state_dim() is not None or inp.resolved_action_dim() is not None or inp.joint_names is not None

    filled(RuleCode.FILE_DECODE_ERROR, video.expected and video.decode_ok is not None, video.expected)
    filled(RuleCode.DURATION_EXCEEDED, True)
    filled(
        RuleCode.TIMESTAMP_NON_MONOTONIC,
        any(len(times) >= 2 for times in inp.stream_times.values()),
        bool(inp.stream_times) or inp.is_robot,
    )
    filled(
        RuleCode.FRAME_GAP_EXCESS,
        has_camera_times or bool(inp.missing_frame_indexes),
        True,
    )
    sync_applicable = inp.is_robot
    filled(RuleCode.SYNC_MISSING_FRAME, has_grid_evidence or bool(inp.missing_frame_indexes), sync_applicable)
    filled(RuleCode.SYNC_MISSING_STATE, has_grid_evidence or bool(inp.missing_state_indexes), sync_applicable)
    filled(RuleCode.SYNC_STALE_ACTION, has_grid_evidence or bool(inp.stale_action_indexes), sync_applicable)
    filled(RuleCode.DIMENSION_MISMATCH, has_dims, sync_applicable)
    filled(RuleCode.NONFINITE_VALUE, has_samples, sync_applicable)
    filled(
        RuleCode.UNVERIFIED_SEMANTICS,
        True,
        inp.is_robot and inp.source_type == SourceType.LEROBOT_V3.value,
    )
    filled(RuleCode.EMPTY_ANNOTATION, inp.annotations is not None and inp.annotations.total_detections is not None,
           inp.annotations is not None)
    filled(RuleCode.LOW_CONF_DETECTION, inp.annotations is not None and bool(inp.annotations.confidences),
           inp.annotations is not None)
    filled(RuleCode.BLUR_FRAME_RATE, bool(video.laplacian_variances), video.expected)
    # rules that fired always count as run even when the generic check misclassified
    for code in sorted(fired):
        if code not in run and code not in not_applicable:
            if code in unavailable:
                unavailable.remove(code)
            run.append(code)
    return run, unavailable, not_applicable


def build_metrics(
    inp: QualityInput,
    findings: Sequence[Finding] = (),
    *,
    thresholds: QcThresholds = THRESHOLDS,
) -> dict[str, Any]:
    """Machine-readable metrics recorded in ``quality.json`` and ``quality_reports.summary_json``."""
    counts = {Severity.INFO.value: 0, Severity.WARN.value: 0, Severity.ERROR.value: 0}
    for finding in findings:
        counts[finding.severity.value] += 1
    sync = _sync_metrics(inp)
    metrics: dict[str, Any] = {
        "policy_version": inp.policy_version,
        "episode_type": inp.episode_type,
        "source_type": inp.source_type,
        "normalization_valid": bool(inp.normalization_valid),
        "duration_ns": int(inp.duration_ns),
        "duration_s": round(int(inp.duration_ns) / 1_000_000_000, 6),
        **sync,
        "finding_count": len(findings),
        "error_count": counts[Severity.ERROR.value],
        "warn_count": counts[Severity.WARN.value],
        "info_count": counts[Severity.INFO.value],
        "video": _video_metrics(inp, thresholds),
        "sync": sync,
    }
    annotations = _annotation_metrics(inp)
    if annotations is not None:
        metrics["annotations"] = annotations
    if inp.extra:
        metrics["extra"] = dict(inp.extra)
    return metrics


def carried_findings(inp: QualityInput) -> list[Finding]:
    """Turn structural normalisation diagnostics into ``qc-policy/1.0`` findings.

    Normalisation refuses to emit samples when the source is structurally broken
    (non-monotonic timestamps, wrong dimensions, non-finite values, ...).  Every
    rule then reports ``unavailable``, and the aggregation used to turn that silence
    into ``PASS`` -- a vacuous pass on an episode with zero usable samples.

    The importer has already decided these are violations, so they are carried into
    the report instead of being contradicted by the verdict.  Only diagnostic codes
    that name a real rule are carried, and only structural ones (the same set that
    makes ``normalization_valid`` false).
    """
    known = {code.value for code in RuleCode}
    found: list[Finding] = []
    for item in inp.structural_diagnostics:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        severity = str(item.get("severity") or "").upper()
        if code not in known or severity not in {Severity.ERROR.value, Severity.WARN.value}:
            continue
        definition = rule_definition(RuleCode(code))
        found.append(
            Finding(
                rule_code=RuleCode(code),
                severity=Severity(severity),
                observed={
                    "carried_from": "normalization",
                    "message": str(item.get("message") or ""),
                    "details": dict(item.get("details") or {}),
                },
                threshold=dict(definition.threshold),
                evidence={
                    "origin": "canonical episode diagnostics",
                    "structural": True,
                    "note": (
                        "normalization produced no usable samples, so the rules had no "
                        "evidence; the importer's structural diagnostic is the verdict"
                    ),
                },
            )
        )
    return found


def evaluate_quality(inp: QualityInput, *, thresholds: QcThresholds = THRESHOLDS) -> QualityEvaluation:
    """Run every ``qc-policy/1.0`` rule once and aggregate the verdict."""
    findings: list[Finding] = []
    for _code, rule in RULE_FUNCTIONS:
        findings.extend(rule(inp, thresholds))
    fired = {finding.rule_code for finding in findings}
    for carried in carried_findings(inp):
        if carried.rule_code in fired:
            continue
        findings.append(carried)
        fired.add(carried.rule_code)
    verdict = decide_verdict(findings)
    if inp.episode_type == EpisodeType.ROBOT_EPISODE.value and not inp.normalization_valid:
        # Never certify an episode that produced no canonical samples: a missing
        # verdict is not a passing one (fail-closed).
        verdict = QualityStatus.FAIL
    run, unavailable, not_applicable = _classify_checks(inp, findings)
    return QualityEvaluation(
        verdict=verdict,
        findings=findings,
        metrics=build_metrics(inp, findings, thresholds=thresholds),
        checks_run=run,
        checks_unavailable=unavailable,
        checks_not_applicable=not_applicable,
    )


__all__ = [
    "AnnotationInfo",
    "Finding",
    "QualityEvaluation",
    "QualityInput",
    "MAX_AFFECTED_RANGES",
    "MAX_EVIDENCE_INDEXES",
    "RULE_FUNCTIONS",
    "TimeRange",
    "VideoInfo",
    "build_metrics",
    "carried_findings",
    "check_blur_frame_rate",
    "check_dimension_mismatch",
    "check_duration_exceeded",
    "check_empty_annotation",
    "check_file_decode_error",
    "check_frame_gap_excess",
    "check_low_conf_detection",
    "check_nonfinite_value",
    "check_sync_missing_frame",
    "check_sync_missing_state",
    "check_sync_stale_action",
    "check_timestamp_non_monotonic",
    "check_unverified_semantics",
    "evaluate_quality",
]
