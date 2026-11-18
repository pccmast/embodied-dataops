"""Pure robot-semantic quality rules for ``qc-policy/2.0`` (V2 SPEC §3.1).

Every rule is a plain function of a :class:`SemanticQcInput`; nothing here
touches the database, the object store or the clock.  A rule returns zero or
more :class:`SemanticFinding` objects that carry the V1 evidence fields
(``observed``/``threshold``/``affected_time_ranges``/``evidence``) plus the V2
provenance fields (``source_stream``/``source_clock_domain``/``method``/
``limits``/``uncertainty_note``).

Rules are exact, not heuristic:

* joint position: ``state[j] < lower[j] - 1e-5`` or ``state[j] > upper[j] + 1e-5``
  (the boundary itself is *inside* the allowed range);
* joint velocity: only adjacent **valid** state samples in one clock domain,
  ``abs(dq)/dt_s > vmax * 1.10`` with ``dt_s > 0`` (equal is allowed);
* action/state lag: median absolute offset strictly greater than 150 ms over at
  least 20 aligned points, only with a verified common timebase -- and it is
  reported as a *lag indicator*, never as causal control latency;
* point cloud non-finite: strictly greater than 10 % of points.

The carried ``qc-policy/1.0`` file/frame/monotonic/sync/non-finite rules are
evaluated on the **native** and **aligned** views with unchanged thresholds;
dimension checks are re-implemented against the profile because 1.0 hard-coded a
six-joint robot.  ``continuous_joints`` from the profile are honoured: range
checks are skipped for a continuous joint and velocity differences are wrapped
to ``[-pi, pi]``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.domain.enums import QualityStatus, Severity
from embodied_dataops.domain.profiles import (
    AUTHORITY_DOCUMENTED,
    AUTHORITY_SYNTHETIC,
    AUTHORITY_UNVERIFIED,
    SEMANTIC_AUTHORITIES,
    RobotProfile,
)
from embodied_dataops.quality import rules as v1_rules
from embodied_dataops.quality.policy import RuleCode, decide_verdict
from embodied_dataops.quality.rules import AnnotationInfo, Finding, QualityInput, TimeRange, VideoInfo
from embodied_dataops.quality.v2_policy import (
    POLICY_CONFIG_SHA256,
    QC_POLICY_V2_VERSION,
    THRESHOLDS_V2,
    QcV2Thresholds,
    SemanticRuleCode,
    rule_definition_v2,
)

#: Native / aligned canonical views (SPEC §2.3).
VIEW_NATIVE = "native"
VIEW_ALIGNED = "aligned"

#: Canonical schema version evaluated by this module.
CANONICAL_V2_SCHEMA_VERSION = "canonical-episode/2.0"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class NativeStream:
    """One native stream of ``canonical-episode/2.0`` (never resampled)."""

    name: str
    kind: str = ""
    clock_domain: str = ""
    timestamp_provenance: str = ""
    clock_verified: bool = False
    source_t_ns: list[int] = field(default_factory=list)
    episode_t_ns: list[int] = field(default_factory=list)
    valid: list[bool] = field(default_factory=list)
    #: ``values`` for state/action streams; camera streams only carry timestamps.
    values: list[list[float] | None] = field(default_factory=list)

    def is_valid(self, index: int) -> bool:
        if 0 <= index < len(self.valid):
            return bool(self.valid[index])
        return True

    def episode_time(self, index: int) -> int:
        if 0 <= index < len(self.episode_t_ns):
            return int(self.episode_t_ns[index])
        if 0 <= index < len(self.source_t_ns):
            return int(self.source_t_ns[index])
        return index * SAMPLE_INTERVAL_NS

    def source_time(self, index: int) -> int | None:
        if 0 <= index < len(self.source_t_ns):
            return int(self.source_t_ns[index])
        return None

    def value(self, index: int) -> list[float] | None:
        if 0 <= index < len(self.values):
            return self.values[index]
        return None

    def sample_count(self) -> int:
        return max(len(self.source_t_ns), len(self.episode_t_ns), len(self.values))

    def valid_sample_count(self) -> int:
        return sum(1 for index in range(self.sample_count()) if self.is_valid(index))


@dataclass
class AlignedView:
    """The ``aligned/10hz/samples.parquet`` view (training grid + validity)."""

    t_ns: list[int] = field(default_factory=list)
    state: list[list[float] | None] = field(default_factory=list)
    action: list[list[float] | None] = field(default_factory=list)
    state_source_t_ns: list[int] = field(default_factory=list)
    action_source_t_ns: list[int] = field(default_factory=list)
    valid: list[bool] = field(default_factory=list)
    invalid_reason: list[list[str]] = field(default_factory=list)
    clock_verified: bool = False

    def is_valid(self, index: int) -> bool:
        if 0 <= index < len(self.valid):
            return bool(self.valid[index])
        return self.state_at(index) is not None

    def state_at(self, index: int) -> list[float] | None:
        if 0 <= index < len(self.state):
            return self.state[index]
        return None

    def action_at(self, index: int) -> list[float] | None:
        if 0 <= index < len(self.action):
            return self.action[index]
        return None

    def state_time(self, index: int) -> int | None:
        if 0 <= index < len(self.state_source_t_ns) and self.state_source_t_ns[index]:
            return int(self.state_source_t_ns[index])
        if 0 <= index < len(self.t_ns):
            return int(self.t_ns[index])
        return None

    def action_time(self, index: int) -> int | None:
        if 0 <= index < len(self.action_source_t_ns):
            return int(self.action_source_t_ns[index])
        return None

    def reasons_at(self, index: int) -> list[str]:
        if 0 <= index < len(self.invalid_reason) and self.invalid_reason[index]:
            return [str(reason) for reason in self.invalid_reason[index]]
        return []

    def point_count(self) -> int:
        return len(self.t_ns)

    def valid_point_count(self) -> int:
        return sum(1 for index in range(self.point_count()) if self.is_valid(index))

    def indexes_with_reason(self, needle: str) -> list[int]:
        return [
            index
            for index in range(self.point_count())
            if any(needle in reason.lower() for reason in self.reasons_at(index))
        ]


@dataclass
class ProfileBinding:
    """How the Episode was bound to a versioned robot profile (SPEC §2.1)."""

    profile_id: str | None = None
    profile_revision: int | None = None
    authority: str = ""
    config: RobotProfile | None = None
    config_sha256: str | None = None
    mode: str = "MISSING"  # SEMANTIC | DIAGNOSTIC_ONLY | MISSING
    binding_verified: bool = False
    missing_fields: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def present(self) -> bool:
        return bool(self.profile_id) and self.config is not None

    @property
    def semantic_authority(self) -> bool:
        return self.config is not None and self.authority in SEMANTIC_AUTHORITIES


@dataclass
class ClockDomainInfo:
    """Clock domain of one stream that participates in the aligned view."""

    stream: str
    clock_domain: str
    timestamp_provenance: str = ""
    clock_verified: bool = False


@dataclass
class PointCloudSummary:
    """Bounded PointCloud2 summary/diagnostics produced by D2-M1/D2-M5.

    ``presence=True`` means a summary/diagnostic for this topic exists.  When the
    whole pointcloud section is absent the quality rules simply do not run
    (they never crash and never invent a pass).
    """

    topic: str = ""
    timestamp_ns: int | None = None
    status: str = ""
    point_count: int | None = None
    sampled_point_count: int | None = None
    nonfinite_sample_count: int | None = None
    fields: list[dict[str, Any]] = field(default_factory=list)
    data_length: int | None = None
    row_step: int | None = None
    point_step: int | None = None
    endianness: str = ""
    errors: list[str] = field(default_factory=list)
    presence: bool = True

    def corrupt_reasons(self) -> list[str]:
        reasons = [str(error) for error in self.errors if str(error)]
        status = (self.status or "").upper()
        if status in {"CORRUPT", "INVALID", "ERROR", "UNSUPPORTED", "UNSUPPORTED_LAYOUT"}:
            reasons.append(f"status={status or 'UNKNOWN'}")
        return reasons

    def nonfinite_ratio(self) -> float | None:
        if self.nonfinite_sample_count is None:
            return None
        if self.sampled_point_count:
            denominator = int(self.sampled_point_count)
        elif self.point_count:
            denominator = int(self.point_count)
        else:
            return None
        if denominator <= 0:
            return None
        return float(self.nonfinite_sample_count) / float(denominator)


@dataclass
class SemanticFinding:
    """One ``qc-policy/2.0`` finding with explicit evidence provenance."""

    rule_code: str
    severity: Severity
    observed: dict[str, Any]
    threshold: dict[str, Any]
    affected_time_ranges: tuple[TimeRange, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    source_stream: str | None = None
    source_clock_domain: str | None = None
    method: str = ""
    limits: dict[str, Any] = field(default_factory=dict)
    uncertainty_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_code": self.rule_code,
            "severity": self.severity.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "affected_time_ranges": [item.as_dict() for item in self.affected_time_ranges],
            "evidence": self.evidence,
            "source_stream": self.source_stream,
            "source_clock_domain": self.source_clock_domain,
            "method": self.method,
            "limits": self.limits,
            "uncertainty_note": self.uncertainty_note,
        }


@dataclass
class SemanticEvaluation:
    """Deterministic outcome of running every applicable ``qc-policy/2.0`` rule."""

    verdict: QualityStatus
    findings: list[SemanticFinding]
    metrics: dict[str, Any]
    checks_run: list[str]
    checks_unavailable: list[str]
    checks_not_applicable: list[str]

    def counts_by_severity(self) -> dict[str, int]:
        counts = {Severity.INFO.value: 0, Severity.WARN.value: 0, Severity.ERROR.value: 0}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return counts


@dataclass
class SemanticQcInput:
    """Everything the V2 rules may look at, assembled by the QUALITY_CHECK v2 handler."""

    episode_id: str
    episode_type: str = "robot_episode"
    source_type: str = ""
    data_origin: str = ""
    canonical_version: int = 1
    schema_version: str = CANONICAL_V2_SCHEMA_VERSION
    policy_version: str = QC_POLICY_V2_VERSION
    policy_config_sha256: str = POLICY_CONFIG_SHA256
    duration_ns: int = 0
    normalization_valid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    profile: ProfileBinding = field(default_factory=ProfileBinding)
    native_state: NativeStream | None = None
    native_action: NativeStream | None = None
    native_cameras: dict[str, NativeStream] = field(default_factory=dict)
    aligned: AlignedView | None = None
    clock_domains: dict[str, ClockDomainInfo] = field(default_factory=dict)
    calibration_evidence: list[dict[str, Any]] = field(default_factory=list)
    expected_terminal_marker: bool | None = None
    terminal_marker_present: bool | None = None
    terminal_marker_evidence: dict[str, Any] = field(default_factory=dict)
    required_modalities: list[str] | None = None
    present_modalities: list[str] = field(default_factory=list)
    pointcloud: list[PointCloudSummary] = field(default_factory=list)
    joint_names: list[str] | None = None
    video: VideoInfo = field(default_factory=VideoInfo)
    annotations: AnnotationInfo | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_robot(self) -> bool:
        return self.episode_type == "robot_episode"

    def resolved_state_dim(self) -> int | None:
        if self.profile.config is not None:
            return self.profile.config.state_dim
        for stream in (self.native_state, self.native_action):
            if stream is not None:
                for index in range(stream.sample_count()):
                    values = stream.value(index)
                    if values:
                        return len(values)
        if self.aligned is not None:
            for index in range(self.aligned.point_count()):
                values = self.aligned.state_at(index)
                if values:
                    return len(values)
        return None

    def resolved_action_dim(self) -> int | None:
        if self.profile.config is not None:
            return self.profile.config.action_dim
        return self.resolved_state_dim()

    def resolved_joint_names(self) -> list[str] | None:
        if self.joint_names is not None:
            return list(self.joint_names)
        if self.profile.config is not None:
            return list(self.profile.config.joint_names)
        return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bounded(items: Sequence[Any], limit: int = THRESHOLDS_V2.max_evidence_items) -> list[Any]:
    return list(items[:limit])


def _range_at(t_ns: int | None, *, duration_ns: int = SAMPLE_INTERVAL_NS) -> TimeRange | None:
    if t_ns is None:
        return None
    return TimeRange(int(t_ns), int(t_ns) + duration_ns)


def _ranges_from_times(times: Iterable[int | None]) -> tuple[TimeRange, ...]:
    ranges: list[TimeRange] = []
    for value in times:
        if value is not None:
            ranges.append(TimeRange(int(value), int(value) + SAMPLE_INTERVAL_NS))
    return tuple(ranges)


def _wrap_to_pi(value: float) -> float:
    return math.remainder(value, 2.0 * math.pi)


def _physics_limits_available(config: Any) -> bool:
    """Whether documented physical limits exist for this profile.

    A ``robot-profile/2.1`` revision of authority ``SOURCE_DOCUMENTED_PARTIAL`` has no
    published limits, so the range/velocity rules are **NOT_APPLICABLE** — never PASS.
    """
    return bool(getattr(config, "limits_available", True)) and bool(
        getattr(config, "semantic_qc_enabled", True)
    )


def _profile_limits(config: RobotProfile) -> tuple[list[float], list[float], list[float], list[bool]]:
    return (
        list(config.joint_lower_rad),
        list(config.joint_upper_rad),
        list(config.max_abs_velocity_rad_s),
        list(config.continuous_joints),
    )


def _state_clock_domain(inp: SemanticQcInput) -> str | None:
    """Return the single clock domain of the native state stream, if verifiable."""
    stream = inp.native_state
    if stream is not None and stream.clock_domain:
        return stream.clock_domain
    declared = sorted({info.clock_domain for info in inp.clock_domains.values() if info.clock_domain})
    if len(declared) == 1:
        return declared[0]
    return None


def _native_state_rows(inp: SemanticQcInput) -> list[tuple[int, int, list[float]]]:
    """Valid native state rows as ``(index, episode_t_ns, values)``."""
    stream = inp.native_state
    rows: list[tuple[int, int, list[float]]] = []
    if stream is None:
        return rows
    for index in range(stream.sample_count()):
        if not stream.is_valid(index):
            continue
        values = stream.value(index)
        if not values:
            continue
        rows.append((index, stream.episode_time(index), values))
    return rows


def _aligned_state_rows(inp: SemanticQcInput) -> list[tuple[int, int, list[float]]]:
    aligned = inp.aligned
    rows: list[tuple[int, int, list[float]]] = []
    if aligned is None:
        return rows
    for index in range(aligned.point_count()):
        if not aligned.is_valid(index):
            continue
        values = aligned.state_at(index)
        if not values:
            continue
        rows.append((index, int(aligned.t_ns[index]), values))
    return rows


def _make_finding(
    inp: SemanticQcInput,
    code: str,
    *,
    observed: dict[str, Any],
    affected_time_ranges: tuple[TimeRange, ...] = (),
    evidence: dict[str, Any] | None = None,
    severity: Severity | None = None,
    source_stream: str | None = None,
    source_clock_domain: str | None = None,
    method: str | None = None,
    limits: dict[str, Any] | None = None,
    uncertainty_note: str | None = None,
) -> SemanticFinding:
    definition = rule_definition_v2(code)
    merged_evidence = dict(evidence or {})
    merged_evidence.setdefault("policy_version", inp.policy_version)
    merged_evidence.setdefault("capability", "emits finding as machine-readable evidence")
    if source_clock_domain is None and code not in {
        SemanticRuleCode.POINTCLOUD_CORRUPT.value,
        SemanticRuleCode.POINTCLOUD_NONFINITE_HIGH.value,
        SemanticRuleCode.EPISODE_INCOMPLETE.value,
        SemanticRuleCode.MISSING_ROBOT_PROFILE.value,
        RuleCode.UNVERIFIED_SEMANTICS.value,
    }:
        source_clock_domain = _state_clock_domain(inp)
    return SemanticFinding(
        rule_code=definition.code,
        severity=severity or definition.severity,
        observed=observed,
        threshold=dict(definition.threshold),
        affected_time_ranges=affected_time_ranges,
        evidence=merged_evidence,
        source_stream=source_stream,
        source_clock_domain=source_clock_domain or _state_clock_domain(inp),
        method=method or definition.method,
        limits=dict(limits or definition.threshold),
        uncertainty_note=uncertainty_note,
    )


# ---------------------------------------------------------------------------
# Carried qc-policy/1.0 rules on the native and aligned views
# ---------------------------------------------------------------------------


def _native_origin(inp: SemanticQcInput) -> int:
    for stream in (inp.native_state, inp.native_action, *inp.native_cameras.values()):
        if stream is None:
            continue
        for index in range(min(len(stream.source_t_ns), len(stream.episode_t_ns))):
            return int(stream.source_t_ns[index]) - int(stream.episode_t_ns[index])
    origin = inp.metadata.get("source_t_ns_origin")
    return int(origin) if isinstance(origin, int) else 0


def _stream_times(inp: SemanticQcInput) -> dict[str, list[int]]:
    stream_times: dict[str, list[int]] = {}
    for name, stream in inp.native_cameras.items():
        if stream.source_t_ns:
            stream_times[f"camera:{name}"] = list(stream.source_t_ns)
            # qc-policy/1.0 carried rules look up the legacy "camera" key; expose the
            # first native camera under it as well (multi-camera detail stays under
            # camera:<name> for qc-policy/2.0 evidence).
            stream_times.setdefault("camera", list(stream.source_t_ns))
    if inp.native_state is not None and inp.native_state.source_t_ns:
        stream_times["state"] = list(inp.native_state.source_t_ns)
    if inp.native_action is not None and inp.native_action.source_t_ns:
        stream_times["action"] = list(inp.native_action.source_t_ns)
    return stream_times


def _v1_base_input(inp: SemanticQcInput) -> QualityInput:
    return QualityInput(
        episode_id=inp.episode_id,
        episode_type=inp.episode_type,
        source_type=inp.source_type,
        data_origin=inp.data_origin,
        duration_ns=inp.duration_ns,
        policy_version=inp.policy_version,
        normalization_valid=inp.normalization_valid,
        metadata=inp.metadata,
        origin_ns=_native_origin(inp),
        video=inp.video,
        annotations=inp.annotations,
        joint_names=inp.resolved_joint_names(),
        state_dim=inp.resolved_state_dim(),
        action_dim=inp.resolved_action_dim(),
        extra=dict(inp.extra),
    )


def _dense_rows(
    times: Sequence[int], rows: Sequence[Sequence[float] | None]
) -> tuple[list[int], list[list[float]]]:
    dense_times: list[int] = []
    dense_rows: list[list[float]] = []
    for index, row in enumerate(rows):
        if row is None:
            continue
        dense_times.append(int(times[index]) if index < len(times) else index * SAMPLE_INTERVAL_NS)
        dense_rows.append([float(value) for value in row])
    return dense_times, dense_rows


def _upgrade_v1_finding(
    finding: Finding,
    *,
    view: str,
    method: str,
    source_stream: str,
    source_clock_domain: str | None = None,
) -> SemanticFinding:
    observed = dict(finding.observed)
    observed.setdefault("canonical_view", view)
    evidence = dict(finding.evidence)
    evidence.setdefault("carried_from", "qc-policy/1.0")
    evidence.setdefault("canonical_view", view)
    return SemanticFinding(
        rule_code=finding.rule_code.value,
        severity=finding.severity,
        observed=observed,
        threshold=dict(finding.threshold),
        affected_time_ranges=tuple(finding.affected_time_ranges),
        evidence=evidence,
        source_stream=source_stream,
        source_clock_domain=source_clock_domain,
        method=method,
        limits=dict(finding.threshold),
        uncertainty_note=None,
    )


def _canonical_episode_times(stream: NativeStream | None) -> list[int]:
    """Episode-relative times for a native stream (source times only as fallback)."""
    if stream is None:
        return []
    if stream.episode_t_ns:
        return list(stream.episode_t_ns)
    if stream.source_t_ns:
        return list(stream.source_t_ns)
    return []


def evaluate_carried_rules(inp: SemanticQcInput) -> list[SemanticFinding]:
    """Evaluate the carried ``qc-policy/1.0`` rules on native and aligned views."""
    findings: list[SemanticFinding] = []
    base = _v1_base_input(inp)
    stream_times = _stream_times(inp)
    native_state_times = _canonical_episode_times(inp.native_state)
    native_action_times = _canonical_episode_times(inp.native_action)

    # --- native view -------------------------------------------------------
    native = replace(base, stream_times=stream_times)
    for rule in (
        v1_rules.check_file_decode_error,
        v1_rules.check_duration_exceeded,
        v1_rules.check_timestamp_non_monotonic,
        v1_rules.check_frame_gap_excess,
        v1_rules.check_blur_frame_rate,
    ):
        for finding in rule(native, THRESHOLDS_V2.v1):
            findings.append(
                _upgrade_v1_finding(
                    finding,
                    view=VIEW_NATIVE,
                    method=f"qc-policy/1.0-carried:{finding.rule_code.value}@native",
                    source_stream="camera",
                    source_clock_domain=(next(iter(inp.native_cameras.values())).clock_domain or None)
                    if inp.native_cameras
                    else None,
                )
            )

    if inp.native_state is not None and inp.native_state.values:
        state_times, state_rows = _dense_rows(native_state_times, inp.native_state.values)
        for finding in v1_rules.check_nonfinite_value(
            replace(base, t_ns=state_times, state=state_rows, action=[], stream_times=stream_times),
            THRESHOLDS_V2.v1,
        ):
            findings.append(
                _upgrade_v1_finding(
                    finding,
                    view=VIEW_NATIVE,
                    method="qc-policy/1.0-carried:NONFINITE_VALUE@native:state",
                    source_stream="state",
                    source_clock_domain=inp.native_state.clock_domain or None,
                )
            )
    if inp.native_action is not None and inp.native_action.values:
        action_times, action_rows = _dense_rows(native_action_times, inp.native_action.values)
        for finding in v1_rules.check_nonfinite_value(
            replace(base, t_ns=action_times, state=[], action=action_rows, stream_times=stream_times),
            THRESHOLDS_V2.v1,
        ):
            findings.append(
                _upgrade_v1_finding(
                    finding,
                    view=VIEW_NATIVE,
                    method="qc-policy/1.0-carried:NONFINITE_VALUE@native:action",
                    source_stream="action",
                    source_clock_domain=inp.native_action.clock_domain or None,
                )
            )

    # --- aligned view ------------------------------------------------------
    aligned = inp.aligned
    if aligned is not None and aligned.point_count():
        grid_t_ns = list(aligned.t_ns)
        state_times, state_rows = _dense_rows(grid_t_ns, aligned.state)
        action_times, action_rows = _dense_rows(grid_t_ns, aligned.action)
        aligned_input = replace(
            base,
            t_ns=grid_t_ns,
            state=state_rows,
            action=action_rows,
            stream_times={"aligned": grid_t_ns},
            grid_t_ns=grid_t_ns,
            missing_frame_indexes=aligned.indexes_with_reason("frame"),
            missing_state_indexes=aligned.indexes_with_reason("state"),
            stale_action_indexes=aligned.indexes_with_reason("action"),
        )
        for rule in (
            v1_rules.check_sync_missing_frame,
            v1_rules.check_sync_missing_state,
            v1_rules.check_sync_stale_action,
            v1_rules.check_timestamp_non_monotonic,
        ):
            for finding in rule(aligned_input, THRESHOLDS_V2.v1):
                findings.append(
                    _upgrade_v1_finding(
                        finding,
                        view=VIEW_ALIGNED,
                        method=f"qc-policy/1.0-carried:{finding.rule_code.value}@aligned",
                        source_stream="aligned_view",
                        source_clock_domain=_state_clock_domain(inp),
                    )
                )
        for name, rows in (("state", state_rows), ("action", action_rows)):
            if not rows:
                continue
            times = state_times if name == "state" else action_times
            for finding in v1_rules.check_nonfinite_value(
                replace(
                    base,
                    t_ns=times,
                    state=rows if name == "state" else [],
                    action=rows if name == "action" else [],
                    stream_times={"aligned": grid_t_ns},
                ),
                THRESHOLDS_V2.v1,
            ):
                findings.append(
                    _upgrade_v1_finding(
                        finding,
                        view=VIEW_ALIGNED,
                        method=f"qc-policy/1.0-carried:NONFINITE_VALUE@aligned:{name}",
                        source_stream=f"aligned_view:{name}",
                        source_clock_domain=_state_clock_domain(inp),
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# New qc-policy/2.0 rules
# ---------------------------------------------------------------------------


def check_missing_robot_profile(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """MISSING_ROBOT_PROFILE: no explicit/trustworthy profile mapping."""
    if not inp.is_robot or inp.profile.present:
        return []
    missing = list(inp.profile.missing_fields) or ["profile_id", "profile_revision", "robot_profile_registry_row"]
    observed = {
        "profile_id": inp.profile.profile_id,
        "profile_revision": inp.profile.profile_revision,
        "mode": inp.profile.mode,
        "missing_mapping_fields": missing,
        "canonical_version": inp.canonical_version,
    }
    evidence = {
        "requires_explicit_profile": True,
        "profile_inferred_from_shape": False,
        "profile_inferred_from_filename": False,
        "diagnostic_only": inp.profile.mode == "DIAGNOSTIC_ONLY",
        "note": (
            "a profile must be selected explicitly by profile_id + profile_revision in the "
            "import mapping; shape=[6|7] never selects a profile"
        ),
    }
    return [
        _make_finding(
            inp,
            SemanticRuleCode.MISSING_ROBOT_PROFILE.value,
            observed=observed,
            evidence=evidence,
            source_stream="episode",
            source_clock_domain=None,
        )
    ]


def check_joint_position_out_of_range(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """JOINT_POSITION_OUT_OF_RANGE with exact ``+-1e-5`` boundaries."""
    config = inp.profile.config
    if not inp.is_robot or config is None:
        return []
    if not _physics_limits_available(config):
        return []
    lower, upper, _velocity, continuous = _profile_limits(config)
    tolerance = thresholds.joint_position_tolerance_rad

    native_rows = _native_state_rows(inp)
    aligned_rows = _aligned_state_rows(inp)
    if native_rows:
        view = VIEW_NATIVE
        rows = native_rows
        stream_name = "state"
        clock_domain = inp.native_state.clock_domain if inp.native_state else None
    elif aligned_rows:
        view = VIEW_ALIGNED
        rows = aligned_rows
        stream_name = "aligned_view:state"
        clock_domain = _state_clock_domain(inp)
    else:
        return []

    violations: dict[int, list[dict[str, Any]]] = {}
    for _index, t_ns, values in rows:
        for joint_index in range(min(len(values), config.state_dim)):
            if continuous[joint_index]:
                continue
            value = float(values[joint_index])
            low_bound = lower[joint_index] - tolerance
            high_bound = upper[joint_index] + tolerance
            if value < low_bound or value > high_bound:
                violations.setdefault(joint_index, []).append(
                    {
                        "t_ns": int(t_ns),
                        "value_rad": value,
                        "bound": "below" if value < low_bound else "above",
                        "limit_rad": lower[joint_index] if value < low_bound else upper[joint_index],
                        "lower_bound_rad": low_bound,
                        "upper_bound_rad": high_bound,
                    }
                )
    findings: list[SemanticFinding] = []
    for joint_index in sorted(violations):
        items = violations[joint_index]
        values = [float(item["value_rad"]) for item in items]
        observed = {
            "joint": config.joint_names[joint_index],
            "joint_index": joint_index,
            "canonical_view": view,
            "lower_rad": lower[joint_index],
            "upper_rad": upper[joint_index],
            "tolerance_rad": tolerance,
            "violation_count": len(items),
            "min_value_rad": min(values),
            "max_value_rad": max(values),
            "worst_value_rad": max(
                values,
                key=lambda value: max(lower[joint_index] - value, value - upper[joint_index], 0.0),
            ),
            "max_violation_rad": max(
                max(lower[joint_index] - value, value - upper[joint_index], 0.0) for value in values
            ),
            "first_t_ns": items[0]["t_ns"],
            "last_t_ns": items[-1]["t_ns"],
        }
        evidence = {
            "violations": _bounded(items),
            "comparison": "value < lower-1e-5 or value > upper+1e-5",
            "continuous_joint_skipped": False,
        }
        findings.append(
            _make_finding(
                inp,
                SemanticRuleCode.JOINT_POSITION_OUT_OF_RANGE.value,
                observed=observed,
                affected_time_ranges=_ranges_from_times(item["t_ns"] for item in items),
                evidence=evidence,
                source_stream=stream_name,
                source_clock_domain=clock_domain,
                method=f"profile_joint_limit_check@{view}",
                limits={
                    "joint": config.joint_names[joint_index],
                    "lower_rad": lower[joint_index],
                    "upper_rad": upper[joint_index],
                    "tolerance_rad": tolerance,
                },
            )
        )
    return findings


def check_joint_velocity_spike(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """JOINT_VELOCITY_SPIKE on adjacent valid same-clock state samples."""
    config = inp.profile.config
    if not inp.is_robot or config is None or inp.native_state is None:
        return []
    if not _physics_limits_available(config):
        return []
    clock_domain = _state_clock_domain(inp)
    if not clock_domain:
        return []
    _lower, _upper, max_velocity, continuous = _profile_limits(config)
    factor = thresholds.velocity_spike_factor
    rows = _native_state_rows(inp)
    if len(rows) < 2:
        return []

    spikes: dict[int, list[dict[str, Any]]] = {}
    previous = rows[0]
    for current in rows[1:]:
        prev_index, prev_t, prev_values = previous
        cur_index, cur_t, cur_values = current
        dt_ns = int(cur_t) - int(prev_t)
        previous = current
        if dt_ns <= 0:
            continue
        dt_s = dt_ns / 1_000_000_000.0
        for joint_index in range(min(len(prev_values), len(cur_values), config.state_dim)):
            delta = float(cur_values[joint_index]) - float(prev_values[joint_index])
            if continuous[joint_index]:
                delta = _wrap_to_pi(delta)
            implied = abs(delta) / dt_s
            limit = max_velocity[joint_index] * factor
            if implied > limit:
                spikes.setdefault(joint_index, []).append(
                    {
                        "previous_index": prev_index,
                        "current_index": cur_index,
                        "previous_t_ns": int(prev_t),
                        "current_t_ns": int(cur_t),
                        "dt_ns": dt_ns,
                        "dt_s": dt_s,
                        "previous_value_rad": float(prev_values[joint_index]),
                        "current_value_rad": float(cur_values[joint_index]),
                        "delta_rad": delta,
                        "implied_velocity_rad_s": implied,
                        "limit_rad_s": limit,
                    }
                )
    findings: list[SemanticFinding] = []
    for joint_index in sorted(spikes):
        items = spikes[joint_index]
        implied_values = [float(item["implied_velocity_rad_s"]) for item in items]
        observed = {
            "joint": config.joint_names[joint_index],
            "joint_index": joint_index,
            "violation_count": len(items),
            "max_implied_velocity_rad_s": max(implied_values),
            "max_abs_velocity_rad_s": max_velocity[joint_index],
            "velocity_factor": factor,
            "limit_rad_s": max_velocity[joint_index] * factor,
            "state_sample_count": len(rows),
            "continuous_joint": bool(continuous[joint_index]),
        }
        evidence = {
            "windows": _bounded(items),
            "comparison": "abs(dq)/dt_s > vmax*1.10 with dt_s > 0",
            "same_clock_domain": clock_domain,
            "invalid_samples_skipped": True,
        }
        findings.append(
            _make_finding(
                inp,
                SemanticRuleCode.JOINT_VELOCITY_SPIKE.value,
                observed=observed,
                affected_time_ranges=_ranges_from_times(item["current_t_ns"] for item in items),
                evidence=evidence,
                source_stream="state",
                source_clock_domain=clock_domain,
                method="adjacent_valid_state_finite_difference@native",
                limits={
                    "joint": config.joint_names[joint_index],
                    "max_abs_velocity_rad_s": max_velocity[joint_index],
                    "velocity_factor": factor,
                    "limit_rad_s": max_velocity[joint_index] * factor,
                },
            )
        )
    return findings


def _calibration_covers(
    pair: tuple[str, str], evidence: Sequence[dict[str, Any]], *, max_offset_ns: int | None
) -> bool:
    pair_set = set(pair)
    for record in evidence:
        if not isinstance(record, dict):
            continue
        domains_raw = record.get("clock_domains") or record.get("domains") or record.get("pair")
        if not isinstance(domains_raw, (list, tuple)):
            continue
        domains = {str(item) for item in domains_raw}
        if not pair_set.issubset(domains):
            continue
        verified = record.get("verified", record.get("clock_verified", record.get("calibrated")))
        if verified is not True:
            continue
        offset = record.get("max_offset_ns", record.get("offset_ns"))
        if isinstance(offset, (int, float)) and max_offset_ns is not None and int(offset) > max_offset_ns:
            continue
        return True
    return False


def check_clock_offset_unverified(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """CLOCK_OFFSET_UNVERIFIED when alignment crosses clock domains uncalibrated."""
    if not inp.is_robot:
        return []
    domains: dict[str, str] = {
        name: info.clock_domain
        for name, info in inp.clock_domains.items()
        if info.clock_domain
    }
    if inp.aligned is None and not inp.native_cameras and inp.native_state is None:
        return []
    distinct = sorted(set(domains.values()))
    if len(distinct) <= 1:
        return []

    pairs = [(a, b) for index, a in enumerate(distinct) for b in distinct[index + 1 :]]
    max_offset_ns = inp.profile.config.max_clock_offset_ns if inp.profile.config else None
    verified_streams = all(
        info.clock_verified for name, info in inp.clock_domains.items() if name in domains
    )
    missing_pairs = [
        pair
        for pair in pairs
        if not _calibration_covers(pair, inp.calibration_evidence, max_offset_ns=max_offset_ns)
    ]
    if not missing_pairs:
        return []
    if verified_streams:
        # every involved stream declares verified calibration evidence
        return []
    observed = {
        "clock_domains": domains,
        "independent_clock_domain_count": len(distinct),
        "cross_domain_pairs": [list(pair) for pair in pairs],
        "missing_calibration_pairs": [list(pair) for pair in missing_pairs],
        "calibration_record_ids": [
            str(record.get("id") or record.get("record_id"))
            for record in inp.calibration_evidence
            if isinstance(record, dict) and (record.get("id") or record.get("record_id"))
        ],
        "max_clock_offset_ns": max_offset_ns,
    }
    evidence = {
        "requires_calibration_evidence": True,
        "missing_calibration_record_ids": [f"clock_calibration:{a}->{b}" for a, b in missing_pairs],
        "single_clock_fast_path": False,
        "note": (
            "cross-domain alignment without calibration evidence may only be described as "
            "approximate log-time synchronization; clock offset is unverified"
        ),
    }
    return [
        _make_finding(
            inp,
            SemanticRuleCode.CLOCK_OFFSET_UNVERIFIED.value,
            observed=observed,
            evidence=evidence,
            source_stream="aligned_view",
            source_clock_domain=",".join(distinct),
            uncertainty_note=(
                "Clock offset across independent domains is not proven; synchronization is "
                "approximate and based on log time only."
            ),
        )
    ]


def check_action_state_lag(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """ACTION_STATE_LAG: median action/state sample offset > 150 ms (lag indicator)."""
    config = inp.profile.config
    aligned = inp.aligned
    if not inp.is_robot or config is None or aligned is None:
        return []
    if config.action_semantics != "joint_target_position":
        return []
    if not aligned.clock_verified:
        return []
    offsets: list[int] = []
    for index in range(aligned.point_count()):
        if not aligned.is_valid(index) or aligned.action_at(index) is None:
            continue
        action_time = aligned.action_time(index)
        state_time = aligned.state_time(index)
        if action_time is None or state_time is None:
            continue
        offsets.append(abs(int(action_time) - int(state_time)))
    if len(offsets) < thresholds.action_state_lag_min_points:
        return []
    median_offset = statistics.median(offsets)
    if not median_offset > thresholds.action_state_lag_median_ns:
        return []
    observed = {
        "aligned_point_count": len(offsets),
        "median_abs_offset_ns": int(median_offset) if float(median_offset).is_integer() else median_offset,
        "median_abs_offset_ms": round(float(median_offset) / 1e6, 6),
        "max_abs_offset_ns": max(offsets),
        "min_abs_offset_ns": min(offsets),
        "action_semantics": config.action_semantics,
        "clock_verified": True,
    }
    evidence = {
        "limitation": (
            "time-lag indicator only: this is NOT evidence of control causal latency and must "
            "not be presented as one"
        ),
        "median_method": "median of abs(action_source_t_ns - compared state sample time)",
        "min_aligned_points": thresholds.action_state_lag_min_points,
        "sample_offsets_ns": _bounded(offsets, 1000),
        "no_causal_claim": True,
    }
    return [
        _make_finding(
            inp,
            SemanticRuleCode.ACTION_STATE_LAG.value,
            observed=observed,
            evidence=evidence,
            severity=Severity.WARN,
            source_stream="action",
            source_clock_domain=_state_clock_domain(inp),
            method="aligned_action_state_offset_median@aligned",
            limits={
                "median_abs_offset_ns": thresholds.action_state_lag_median_ns,
                "min_aligned_points": thresholds.action_state_lag_min_points,
                "causal_claim": False,
            },
            uncertainty_note=(
                "Reported as a lag indicator only; no causal control-delay claim is made."
            ),
        )
    ]


def check_episode_incomplete(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """EPISODE_INCOMPLETE from source declarations only."""
    terminal_missing = inp.expected_terminal_marker is True and inp.terminal_marker_present is not True
    required = list(inp.required_modalities or [])
    present = {str(name) for name in inp.present_modalities}
    missing_modalities = [name for name in required if name not in present]
    if not terminal_missing and not missing_modalities:
        return []
    observed = {
        "expected_terminal_marker": inp.expected_terminal_marker,
        "terminal_marker_present": inp.terminal_marker_present,
        "terminal_marker_missing": terminal_missing,
        "required_modalities": required,
        "present_modalities": sorted(present),
        "missing_required_modalities": missing_modalities,
        "manifest_required_modality_missing": bool(missing_modalities),
    }
    evidence = {
        "terminal_marker_evidence": dict(inp.terminal_marker_evidence),
        "source_declares_expected_terminal_marker": bool(inp.expected_terminal_marker),
        "success_inferred_from_action_magnitude": False,
        "note": (
            "missing terminal marker is only reported when the source declares one; task "
            "success is never guessed from action magnitude"
        ),
    }
    return [
        _make_finding(
            inp,
            SemanticRuleCode.EPISODE_INCOMPLETE.value,
            observed=observed,
            evidence=evidence,
            source_stream="episode",
            source_clock_domain=None,
            method="source_terminal_and_modality_check",
        )
    ]


def check_pointcloud_corrupt(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """POINTCLOUD_CORRUPT from structural PointCloud2 diagnostics (tolerant)."""
    summaries = [summary for summary in inp.pointcloud if summary.presence]
    corrupt: list[dict[str, Any]] = []
    for summary in summaries:
        reasons = summary.corrupt_reasons()
        if not reasons:
            continue
        corrupt.append(
            {
                "topic": summary.topic,
                "timestamp_ns": summary.timestamp_ns,
                "status": summary.status,
                "reasons": reasons,
                "fields": summary.fields,
                "data_length": summary.data_length,
                "row_step": summary.row_step,
                "point_step": summary.point_step,
            }
        )
    if not corrupt:
        return []
    observed = {
        "corrupt_topic_count": len(corrupt),
        "topics": [item["topic"] for item in corrupt],
        "details": _bounded(corrupt),
    }
    evidence = {
        "bounded_parse": True,
        "raw_coordinates_retained": False,
        "structural_validation": (
            "len(data) >= row_step*height, field offset+4 <= point_step, "
            "row_step >= point_step*width, supported FLOAT32 datatype"
        ),
    }
    return [
        _make_finding(
            inp,
            SemanticRuleCode.POINTCLOUD_CORRUPT.value,
            observed=observed,
            evidence=evidence,
            source_stream="pointcloud",
            source_clock_domain=None,
        )
    ]


def check_pointcloud_nonfinite_high(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """POINTCLOUD_NONFINITE_HIGH: non-finite XYZ ratio strictly greater than 0.10."""
    high: list[dict[str, Any]] = []
    for summary in inp.pointcloud:
        ratio = summary.nonfinite_ratio()
        if ratio is None:
            continue
        if ratio > thresholds.pointcloud_nonfinite_ratio:
            high.append(
                {
                    "topic": summary.topic,
                    "timestamp_ns": summary.timestamp_ns,
                    "point_count": summary.point_count,
                    "sampled_point_count": summary.sampled_point_count,
                    "nonfinite_sample_count": summary.nonfinite_sample_count,
                    "nonfinite_ratio": round(ratio, 6),
                }
            )
    if not high:
        return []
    max_ratio = max(item["nonfinite_ratio"] for item in high)
    observed = {
        "topics": [item["topic"] for item in high],
        "topic_count": len(high),
        "max_nonfinite_ratio": max_ratio,
        "nonfinite_ratio_threshold": thresholds.pointcloud_nonfinite_ratio,
        "details": _bounded(high),
    }
    evidence = {
        "comparison": "nonfinite_sample_count / sampled_point_count > 0.10 (strict)",
        "sampling": "deterministic sampling of at most 10,000 points (D2-M5 ADR-0007 §9)",
        "raw_coordinates_retained": False,
    }
    return [
        _make_finding(
            inp,
            SemanticRuleCode.POINTCLOUD_NONFINITE_HIGH.value,
            observed=observed,
            evidence=evidence,
            source_stream="pointcloud",
            source_clock_domain=None,
        )
    ]


def check_dimension_mismatch_v2(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """DIMENSION_MISMATCH against the bound profile (6D or 7D), never hard-coded."""
    config = inp.profile.config
    if not inp.is_robot or config is None:
        return []
    expected_names = list(config.joint_names)
    findings: list[SemanticFinding] = []

    def _view_problem(view: str, state_dim: int | None, action_dim: int | None, names: list[str] | None) -> None:
        problems: dict[str, Any] = {}
        if state_dim is not None and state_dim != config.state_dim:
            problems["state_dim"] = state_dim
        if action_dim is not None and action_dim != config.action_dim:
            problems["action_dim"] = action_dim
        if names is not None and list(names) != expected_names:
            problems["joint_names"] = list(names)
        if not problems:
            return
        observed = {
            "canonical_view": view,
            "profile_id": config.profile_id,
            "profile_revision": config.profile_revision,
            "expected_state_dim": config.state_dim,
            "expected_action_dim": config.action_dim,
            "observed_state_dim": state_dim,
            "observed_action_dim": action_dim,
            "observed_joint_names": list(names) if names is not None else None,
            "problems": problems,
        }
        evidence = {
            "expected_joint_names": expected_names,
            "profile_id": config.profile_id,
            "profile_revision": config.profile_revision,
            "no_truncation_or_padding": True,
            "shape_not_used_to_select_profile": True,
        }
        findings.append(
            _make_finding(
                inp,
                v1_rules.RuleCode.DIMENSION_MISMATCH.value,
                observed=observed,
                evidence=evidence,
                source_stream=view,
                source_clock_domain=_state_clock_domain(inp),
                method=f"profile_dimension_check@{view}",
                limits={
                    "state_dim": config.state_dim,
                    "action_dim": config.action_dim,
                    "joint_names": expected_names,
                },
            )
        )

    native_state_dim = _first_width(inp.native_state)
    native_action_dim = _first_width(inp.native_action)
    if native_state_dim is not None or native_action_dim is not None:
        _view_problem(VIEW_NATIVE, native_state_dim, native_action_dim, inp.resolved_joint_names())
    aligned = inp.aligned
    if aligned is not None:
        aligned_state_dim = _first_aligned_width(aligned.state)
        aligned_action_dim = _first_aligned_width(aligned.action)
        if aligned_state_dim is not None or aligned_action_dim is not None:
            _view_problem(VIEW_ALIGNED, aligned_state_dim, aligned_action_dim, inp.resolved_joint_names())
    return findings


def check_unverified_semantics_v2(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> list[SemanticFinding]:
    """UNVERIFIED_SEMANTICS: profile authority/binding does not prove robot semantics."""
    if not inp.is_robot:
        return []
    binding = inp.profile
    reasons: list[str] = []
    if binding.config is None:
        reasons.append("robot_profile_missing")
    elif binding.authority == AUTHORITY_UNVERIFIED:
        reasons.append("profile_authority_unverified")
    elif not binding.binding_verified:
        reasons.append("profile_binding_unverified")
    elif binding.authority == AUTHORITY_SYNTHETIC and inp.data_origin != "synthetic":
        reasons.append("synthetic_fixture_profile_not_valid_for_non_synthetic_data")
    if not reasons:
        return []
    observed = {
        "profile_id": binding.profile_id,
        "profile_revision": binding.profile_revision,
        "authority": binding.authority,
        "binding_verified": binding.binding_verified,
        "data_origin": inp.data_origin,
        "reasons": reasons,
        "state_semantics": binding.config.state_semantics if binding.config else None,
        "action_semantics": binding.config.action_semantics if binding.config else None,
    }
    evidence = {
        "publishable": False,
        "required_authorities": [AUTHORITY_DOCUMENTED, AUTHORITY_SYNTHETIC],
        "required_source_evidence": True,
        "note": (
            "unverified robot semantics can never be published: synthetic limit values do not "
            "establish a physical specification and a synthetic-fixture profile does not "
            "verify public/real capture semantics"
        ),
        "detail": dict(binding.detail),
    }
    return [
        _make_finding(
            inp,
            v1_rules.RuleCode.UNVERIFIED_SEMANTICS.value,
            observed=observed,
            evidence=evidence,
            source_stream="episode",
            source_clock_domain=None,
            method="profile_authority_check",
        )
    ]


def _first_width(stream: NativeStream | None) -> int | None:
    if stream is None:
        return None
    for index in range(stream.sample_count()):
        values = stream.value(index)
        if values:
            return len(values)
    return None


def _first_aligned_width(rows: Sequence[Sequence[float] | None]) -> int | None:
    for row in rows:
        if row:
            return len(row)
    return None


#: Rule functions in deterministic evaluation order (carried rules are run separately).
SEMANTIC_RULE_FUNCTIONS = (
    check_missing_robot_profile,
    check_joint_position_out_of_range,
    check_joint_velocity_spike,
    check_clock_offset_unverified,
    check_action_state_lag,
    check_episode_incomplete,
    check_pointcloud_corrupt,
    check_pointcloud_nonfinite_high,
    check_dimension_mismatch_v2,
    check_unverified_semantics_v2,
)


# ---------------------------------------------------------------------------
# metrics + evaluation
# ---------------------------------------------------------------------------


def build_semantic_metrics(
    inp: SemanticQcInput, findings: Sequence[SemanticFinding]
) -> dict[str, Any]:
    """Machine-readable metrics recorded in ``quality.json`` and the report summary."""
    counts = {Severity.INFO.value: 0, Severity.WARN.value: 0, Severity.ERROR.value: 0}
    for finding in findings:
        counts[finding.severity.value] += 1
    aligned = inp.aligned
    state_stream = inp.native_state
    action_stream = inp.native_action
    config = inp.profile.config
    return {
        "policy_version": inp.policy_version,
        "policy_config_sha256": inp.policy_config_sha256,
        "schema_version": inp.schema_version,
        "canonical_version": inp.canonical_version,
        "episode_type": inp.episode_type,
        "source_type": inp.source_type,
        "data_origin": inp.data_origin,
        "normalization_valid": bool(inp.normalization_valid),
        "duration_ns": int(inp.duration_ns),
        "profile": {
            "profile_id": inp.profile.profile_id,
            "profile_revision": inp.profile.profile_revision,
            "authority": inp.profile.authority,
            "config_sha256": inp.profile.config_sha256,
            "mode": inp.profile.mode,
            "binding_verified": inp.profile.binding_verified,
            "state_dim": config.state_dim if config else None,
            "action_dim": config.action_dim if config else None,
        },
        "native": {
            "state_sample_count": state_stream.sample_count() if state_stream else 0,
            "state_valid_sample_count": state_stream.valid_sample_count() if state_stream else 0,
            "action_sample_count": action_stream.sample_count() if action_stream else 0,
            "action_valid_sample_count": action_stream.valid_sample_count() if action_stream else 0,
            "camera_streams": {
                name: {
                    "timestamp_count": len(stream.source_t_ns),
                    "clock_domain": stream.clock_domain,
                    "timestamp_provenance": stream.timestamp_provenance,
                }
                for name, stream in inp.native_cameras.items()
            },
            "clock_domain": _state_clock_domain(inp),
            "clock_verified": all(
                info.clock_verified for info in inp.clock_domains.values()
            )
            if inp.clock_domains
            else None,
        },
        "aligned": {
            "grid_point_count": aligned.point_count() if aligned else 0,
            "valid_point_count": aligned.valid_point_count() if aligned else 0,
            "invalid_point_count": (
                aligned.point_count() - aligned.valid_point_count() if aligned else 0
            ),
            "missing_frame_count": len(aligned.indexes_with_reason("frame")) if aligned else 0,
            "missing_state_count": len(aligned.indexes_with_reason("state")) if aligned else 0,
            "stale_action_count": len(aligned.indexes_with_reason("action")) if aligned else 0,
            "clock_verified": bool(aligned.clock_verified) if aligned else None,
        },
        "clock_domains": {
            name: {
                "clock_domain": info.clock_domain,
                "timestamp_provenance": info.timestamp_provenance,
                "clock_verified": info.clock_verified,
            }
            for name, info in inp.clock_domains.items()
        },
        "pointcloud": {
            "summary_count": len(inp.pointcloud),
            "corrupt_count": sum(1 for summary in inp.pointcloud if summary.corrupt_reasons()),
            "nonfinite_high_count": sum(
                1
                for summary in inp.pointcloud
                if (summary.nonfinite_ratio() or 0.0) > THRESHOLDS_V2.pointcloud_nonfinite_ratio
            ),
        },
        "finding_count": len(findings),
        "error_count": counts[Severity.ERROR.value],
        "warn_count": counts[Severity.WARN.value],
        "info_count": counts[Severity.INFO.value],
    }


def _availability(inp: SemanticQcInput) -> dict[str, str]:
    """Map every qc-policy/2.0 rule code to ``run``/``unavailable``/``not_applicable``."""
    availability: dict[str, str] = {}
    config = inp.profile.config
    aligned = inp.aligned
    has_camera = bool(inp.native_cameras)
    has_native_samples = bool(
        (inp.native_state and inp.native_state.sample_count())
        or (inp.native_action and inp.native_action.sample_count())
    )
    has_aligned = aligned is not None and aligned.point_count() > 0
    has_clock_info = bool(inp.clock_domains)

    def put(code: str, value: str) -> None:
        availability[code] = value

    put(v1_rules.RuleCode.FILE_DECODE_ERROR.value, "run" if inp.video.expected else "not_applicable")
    put(v1_rules.RuleCode.DURATION_EXCEEDED.value, "run")
    put(
        v1_rules.RuleCode.TIMESTAMP_NON_MONOTONIC.value,
        "run" if (has_camera or has_native_samples) else "unavailable",
    )
    put(
        v1_rules.RuleCode.FRAME_GAP_EXCESS.value,
        "run" if has_camera else "unavailable",
    )
    for code in (
        v1_rules.RuleCode.SYNC_MISSING_FRAME.value,
        v1_rules.RuleCode.SYNC_MISSING_STATE.value,
        v1_rules.RuleCode.SYNC_STALE_ACTION.value,
    ):
        put(code, "run" if has_aligned else "unavailable")
    put(
        v1_rules.RuleCode.DIMENSION_MISMATCH.value,
        "run" if (config is not None and (has_native_samples or has_aligned)) else "unavailable",
    )
    put(
        v1_rules.RuleCode.NONFINITE_VALUE.value,
        "run" if (has_native_samples or has_aligned) else "unavailable",
    )
    put(v1_rules.RuleCode.UNVERIFIED_SEMANTICS.value, "run")
    put(
        v1_rules.RuleCode.BLUR_FRAME_RATE.value,
        "run" if inp.video.expected else "not_applicable",
    )
    put(v1_rules.RuleCode.EMPTY_ANNOTATION.value, "not_applicable")
    put(v1_rules.RuleCode.LOW_CONF_DETECTION.value, "not_applicable")

    put(SemanticRuleCode.MISSING_ROBOT_PROFILE.value, "run")
    put(
        SemanticRuleCode.JOINT_POSITION_OUT_OF_RANGE.value,
        "run" if (config is not None and (has_native_samples or has_aligned)) else "unavailable",
    )
    put(
        SemanticRuleCode.JOINT_VELOCITY_SPIKE.value,
        "run" if (config is not None and inp.native_state is not None) else "unavailable",
    )
    put(
        SemanticRuleCode.CLOCK_OFFSET_UNVERIFIED.value,
        "run" if (has_aligned and has_clock_info) else "unavailable",
    )
    lag_preconditions = (
        config is not None
        and has_aligned
        and config.action_semantics == "joint_target_position"
    )
    put(
        SemanticRuleCode.ACTION_STATE_LAG.value,
        "run" if (lag_preconditions and aligned is not None and aligned.clock_verified) else "not_applicable",
    )
    declaration = inp.expected_terminal_marker is True or bool(inp.required_modalities)
    put(
        SemanticRuleCode.EPISODE_INCOMPLETE.value,
        "run" if declaration else "not_applicable",
    )
    put(
        SemanticRuleCode.POINTCLOUD_CORRUPT.value,
        "run" if inp.pointcloud else "not_applicable",
    )
    put(
        SemanticRuleCode.POINTCLOUD_NONFINITE_HIGH.value,
        "run" if inp.pointcloud else "not_applicable",
    )
    return availability


def evaluate_semantic_quality(
    inp: SemanticQcInput, thresholds: QcV2Thresholds = THRESHOLDS_V2
) -> SemanticEvaluation:
    """Run every ``qc-policy/2.0`` rule once and aggregate the verdict."""
    findings: list[SemanticFinding] = []
    findings.extend(evaluate_carried_rules(inp))
    for rule in SEMANTIC_RULE_FUNCTIONS:
        findings.extend(rule(inp, thresholds))
    verdict = decide_verdict(findings)
    if inp.normalization_valid is False:
        # Same fail-closed rule as qc-policy/1.0: an episode whose normalisation
        # produced no usable samples can never be certified, and "all rules
        # unavailable" must not aggregate to PASS.
        verdict = QualityStatus.FAIL
    availability = _availability(inp)
    fired = {finding.rule_code for finding in findings}
    run: list[str] = []
    unavailable: list[str] = []
    not_applicable: list[str] = []
    for code, state in availability.items():
        if code in fired or state == "run":
            run.append(code)
        elif state == "unavailable":
            unavailable.append(code)
        else:
            not_applicable.append(code)
    return SemanticEvaluation(
        verdict=verdict,
        findings=findings,
        metrics=build_semantic_metrics(inp, findings),
        checks_run=sorted(run),
        checks_unavailable=sorted(unavailable),
        checks_not_applicable=sorted(not_applicable),
    )


__all__ = [
    "AlignedView",
    "CANONICAL_V2_SCHEMA_VERSION",
    "ClockDomainInfo",
    "NativeStream",
    "PointCloudSummary",
    "ProfileBinding",
    "SEMANTIC_RULE_FUNCTIONS",
    "SemanticEvaluation",
    "SemanticFinding",
    "SemanticQcInput",
    "VIEW_ALIGNED",
    "VIEW_NATIVE",
    "build_semantic_metrics",
    "check_action_state_lag",
    "check_clock_offset_unverified",
    "check_dimension_mismatch_v2",
    "check_episode_incomplete",
    "check_joint_position_out_of_range",
    "check_joint_velocity_spike",
    "check_missing_robot_profile",
    "check_pointcloud_corrupt",
    "check_pointcloud_nonfinite_high",
    "check_unverified_semantics_v2",
    "evaluate_carried_rules",
    "evaluate_semantic_quality",
]
