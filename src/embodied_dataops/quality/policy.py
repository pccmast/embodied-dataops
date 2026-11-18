"""Declarative quality policy ``qc-policy/1.0`` (SPEC 7.1).

The policy is data, not business logic: rule codes, base severities, thresholds
and the verdict aggregation are declared once here and consumed by
:mod:`embodied_dataops.quality.rules` and
:mod:`embodied_dataops.quality.report`.

Verdict aggregation (SPEC 7.1)::

    any ERROR            -> FAIL
    else any WARN        -> WARN
    else                 -> PASS
    INFO never changes the verdict

Boundary conventions from SPEC 4.2: tolerances (50 ms / 100 ms image, state and
action synchronisation) use ``<=``; quality thresholds expressed as "greater
than" (> 200 ms frame gap, > 30 % blurry frames, > 600 s duration) use a strict
comparison.  Changing any threshold requires a **new policy version**; already
approved reports are immutable and keep pointing at the version that produced
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied_dataops.domain.enums import QualityStatus, Severity, StrEnum

#: Version string bound to every QualityReport (SPEC 7.1, 10.1).
QC_POLICY_VERSION = "qc-policy/1.0"


class RuleCode(StrEnum):
    """Every rule code of ``qc-policy/1.0`` (SPEC 7.1)."""

    FILE_DECODE_ERROR = "FILE_DECODE_ERROR"
    DURATION_EXCEEDED = "DURATION_EXCEEDED"
    TIMESTAMP_NON_MONOTONIC = "TIMESTAMP_NON_MONOTONIC"
    FRAME_GAP_EXCESS = "FRAME_GAP_EXCESS"
    SYNC_MISSING_FRAME = "SYNC_MISSING_FRAME"
    SYNC_MISSING_STATE = "SYNC_MISSING_STATE"
    SYNC_STALE_ACTION = "SYNC_STALE_ACTION"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    NONFINITE_VALUE = "NONFINITE_VALUE"
    UNVERIFIED_SEMANTICS = "UNVERIFIED_SEMANTICS"
    EMPTY_ANNOTATION = "EMPTY_ANNOTATION"
    LOW_CONF_DETECTION = "LOW_CONF_DETECTION"
    BLUR_FRAME_RATE = "BLUR_FRAME_RATE"


@dataclass(frozen=True)
class QcThresholds:
    """Exact numeric thresholds of ``qc-policy/1.0`` (integers in ns where possible)."""

    #: Episode duration limit: > 600 s is an ERROR (SPEC 7.1).
    max_duration_ns: int = 600 * 1_000_000_000
    #: Source camera adjacent gap: strictly greater than 200 ms is a WARN.
    frame_gap_max_ns: int = 200_000_000
    #: Image nearest-neighbour tolerance: <= 50 ms.
    image_tolerance_ns: int = 50_000_000
    #: State interpolation endpoint tolerance (both ends): <= 50 ms.
    state_tolerance_ns: int = 50_000_000
    #: Action zero-order-hold age limit (strictly past): <= 100 ms.
    action_zoh_tolerance_ns: int = 100_000_000
    #: Fixed first-version robot profile dimensions (SPEC 5.2).
    state_dim: int = 6
    action_dim: int = 6
    joint_names: tuple[str, ...] = (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
    )
    #: Detection confidence below this value is a LOW_CONF_DETECTION INFO.
    low_conf_threshold: float = 0.35
    #: 640x480 grayscale Laplacian variance below this value counts as blurry.
    blur_laplacian_variance_threshold: float = 40.0
    #: Blurry frames strictly greater than this ratio are a BLUR_FRAME_RATE WARN.
    blur_frame_ratio_threshold: float = 0.30
    #: At least this many decoded frames are required for FILE_DECODE_ERROR.
    min_video_frames: int = 1


#: The single frozen threshold instance of qc-policy/1.0.
THRESHOLDS = QcThresholds()


@dataclass(frozen=True)
class RuleDefinition:
    """Declarative definition of one rule.

    ``severity`` is the base severity.  ``FRAME_GAP_EXCESS`` escalates from WARN
    to ERROR when the 10 Hz grid is missing an image (SPEC 7.1), which the rule
    implementation decides and records; every other rule uses its base severity.
    """

    code: RuleCode
    severity: Severity
    applies_to: str
    description: str
    threshold: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_code": self.code.value,
            "severity": self.severity.value,
            "applies_to": self.applies_to,
            "description": self.description,
            "threshold": self.threshold,
        }


def _rule(
    code: RuleCode,
    severity: Severity,
    applies_to: str,
    description: str,
    threshold: dict[str, Any],
) -> RuleDefinition:
    return RuleDefinition(code, severity, applies_to, description, threshold)


#: Ordered rule registry: this order is also the deterministic finding order.
RULES: dict[RuleCode, RuleDefinition] = {
    RuleCode.FILE_DECODE_ERROR: _rule(
        RuleCode.FILE_DECODE_ERROR,
        Severity.ERROR,
        "all video",
        "ffmpeg full decode failed (non-zero exit) or zero decodable frames",
        {"min_decodable_frames": THRESHOLDS.min_video_frames},
    ),
    RuleCode.DURATION_EXCEEDED: _rule(
        RuleCode.DURATION_EXCEEDED,
        Severity.ERROR,
        "all",
        "Episode duration exceeds 600 s",
        {"max_duration_ns": THRESHOLDS.max_duration_ns, "max_duration_s": 600},
    ),
    RuleCode.TIMESTAMP_NON_MONOTONIC: _rule(
        RuleCode.TIMESTAMP_NON_MONOTONIC,
        Severity.ERROR,
        "robot/video",
        "source timestamps are not strictly increasing (duplicates are violations)",
        {"strictly_increasing": True, "duplicates_allowed": False},
    ),
    RuleCode.FRAME_GAP_EXCESS: _rule(
        RuleCode.FRAME_GAP_EXCESS,
        Severity.WARN,
        "all",
        "adjacent source camera timestamps differ by more than 200 ms (WARN); "
        "a missing 10 Hz grid image escalates to ERROR",
        {
            "max_gap_ns": THRESHOLDS.frame_gap_max_ns,
            "max_gap_ms": 200,
            "missing_grid_image_severity": "ERROR",
        },
    ),
    RuleCode.SYNC_MISSING_FRAME: _rule(
        RuleCode.SYNC_MISSING_FRAME,
        Severity.ERROR,
        "robot",
        "a 10 Hz grid point has no camera frame within <= 50 ms",
        {"image_tolerance_ns": THRESHOLDS.image_tolerance_ns, "image_tolerance_ms": 50},
    ),
    RuleCode.SYNC_MISSING_STATE: _rule(
        RuleCode.SYNC_MISSING_STATE,
        Severity.ERROR,
        "robot",
        "a 10 Hz grid point has no valid bracketing state sample with both endpoints within <= 50 ms",
        {"state_tolerance_ns": THRESHOLDS.state_tolerance_ns, "state_tolerance_ms": 50},
    ),
    RuleCode.SYNC_STALE_ACTION: _rule(
        RuleCode.SYNC_STALE_ACTION,
        Severity.ERROR,
        "robot",
        "a 10 Hz grid point has no action at or before it within <= 100 ms (future actions are never used)",
        {"action_zoh_tolerance_ns": THRESHOLDS.action_zoh_tolerance_ns, "action_zoh_tolerance_ms": 100},
    ),
    RuleCode.DIMENSION_MISMATCH: _rule(
        RuleCode.DIMENSION_MISMATCH,
        Severity.ERROR,
        "robot",
        "state/action are not 6-dimensional or joint names are missing/duplicated/out of profile",
        {
            "state_dim": THRESHOLDS.state_dim,
            "action_dim": THRESHOLDS.action_dim,
            "joint_names": list(THRESHOLDS.joint_names),
        },
    ),
    RuleCode.NONFINITE_VALUE: _rule(
        RuleCode.NONFINITE_VALUE,
        Severity.ERROR,
        "robot",
        "state or action contains NaN or +-Inf",
        {"forbidden_values": ["NaN", "+Inf", "-Inf"]},
    ),
    RuleCode.UNVERIFIED_SEMANTICS: _rule(
        RuleCode.UNVERIFIED_SEMANTICS,
        Severity.ERROR,
        "LeRobot robot episodes",
        "source evidence does not prove that 6D state/action are joint positions / "
        "target joint positions in arm6_joint_position_v1",
        {"robot_profile": "arm6_joint_position_v1", "requires_source_evidence": True},
    ),
    RuleCode.EMPTY_ANNOTATION: _rule(
        RuleCode.EMPTY_ANNOTATION,
        Severity.INFO,
        "episodes with pre-annotation",
        "model output contains zero detections (informational only)",
        {"max_detections": 0},
    ),
    RuleCode.LOW_CONF_DETECTION: _rule(
        RuleCode.LOW_CONF_DETECTION,
        Severity.INFO,
        "episodes with pre-annotation",
        "a detection has confidence below 0.35; it is excluded from the official candidate set",
        {"min_confidence": THRESHOLDS.low_conf_threshold},
    ),
    RuleCode.BLUR_FRAME_RATE: _rule(
        RuleCode.BLUR_FRAME_RATE,
        Severity.WARN,
        "all video",
        "more than 30% of 640x480 grayscale frames have Laplacian variance below 40 "
        "(heuristic, never a real quality label)",
        {
            "laplacian_variance": THRESHOLDS.blur_laplacian_variance_threshold,
            "blurry_frame_ratio": THRESHOLDS.blur_frame_ratio_threshold,
            "frame_size": [640, 480],
        },
    ),
}


def rule_definition(code: RuleCode | str) -> RuleDefinition:
    return RULES[RuleCode(code)]


def decide_verdict(findings) -> QualityStatus:
    """Aggregate findings into the qc-policy/1.0 verdict (SPEC 7.1)."""
    severities = {finding.severity for finding in findings}
    if Severity.ERROR in severities:
        return QualityStatus.FAIL
    if Severity.WARN in severities:
        return QualityStatus.WARN
    return QualityStatus.PASS


def policy_document() -> dict[str, Any]:
    """Machine-readable policy document (for docs, API and policy tests)."""
    return {
        "policy_version": QC_POLICY_VERSION,
        "verdict_rules": {
            "any_error": QualityStatus.FAIL.value,
            "else_any_warn": QualityStatus.WARN.value,
            "else": QualityStatus.PASS.value,
            "info_never_changes_verdict": True,
        },
        "thresholds": {
            "max_duration_ns": THRESHOLDS.max_duration_ns,
            "frame_gap_max_ns": THRESHOLDS.frame_gap_max_ns,
            "image_tolerance_ns": THRESHOLDS.image_tolerance_ns,
            "state_tolerance_ns": THRESHOLDS.state_tolerance_ns,
            "action_zoh_tolerance_ns": THRESHOLDS.action_zoh_tolerance_ns,
            "state_dim": THRESHOLDS.state_dim,
            "action_dim": THRESHOLDS.action_dim,
            "joint_names": list(THRESHOLDS.joint_names),
            "low_conf_threshold": THRESHOLDS.low_conf_threshold,
            "blur_laplacian_variance_threshold": THRESHOLDS.blur_laplacian_variance_threshold,
            "blur_frame_ratio_threshold": THRESHOLDS.blur_frame_ratio_threshold,
        },
        "rules": [definition.as_dict() for definition in RULES.values()],
    }


__all__ = [
    "QC_POLICY_VERSION",
    "QcThresholds",
    "THRESHOLDS",
    "RULES",
    "RuleCode",
    "RuleDefinition",
    "decide_verdict",
    "policy_document",
    "rule_definition",
]
