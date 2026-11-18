"""Declarative quality policy ``qc-policy/2.0`` (V2 SPEC §3, DO2-13).

The V2 policy is data, not business logic.  It keeps every ``qc-policy/1.0``
rule (file/frame/monotonic/dimension/sync/non-finite) with **unchanged**
thresholds and adds the eight robot-semantic rules of SPEC_ADDENDUM_V2 §3.1:

============== ==================== =================================================
rule_code                          exact algorithm (never approximated)
============== ==================== =================================================
MISSING_ROBOT_PROFILE  ERROR       no explicit/trustworthy profile mapping
JOINT_POSITION_OUT_OF_RANGE ERROR  valid state[j] < lower[j]-1e-5 or > upper[j]+1e-5
JOINT_VELOCITY_SPIKE   ERROR       adjacent valid same-clock state:
                                   |dq| / dt > vmax * 1.10  (dt > 0)
CLOCK_OFFSET_UNVERIFIED WARN       alignment crosses independent clock domains without
                                   calibration evidence (single clock fast path: no)
ACTION_STATE_LAG       WARN        verified common timebase + target-position actions;
                                   >= 20 aligned points; median |action_source_t_ns -
                                   state sample time| > 150 ms (lag indicator only)
EPISODE_INCOMPLETE     ERROR       source declares expected_terminal_marker=true and it
                                   is missing, or a manifest-required modality is absent
POINTCLOUD_CORRUPT     ERROR       PointCloud2 structural diagnostics from D2-M1/M5
POINTCLOUD_NONFINITE_HIGH WARN     non-finite XYZ points / points > 0.10 (strict)
============== ==================== =================================================

Threshold changes require a **new policy version**; the canonical JSON SHA-256 of
:func:`policy_config` is stored with every ``quality-report/2.0`` so a report can
always be traced back to the exact policy that produced it.  ``qc-policy/1.0``
reports are never recomputed in place (ADR-0007).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied_dataops.domain.enums import QualityStatus, Severity, StrEnum
from embodied_dataops.quality.policy import (
    QC_POLICY_VERSION,
    QcThresholds,
    RuleCode,
    decide_verdict,
)
from embodied_dataops.quality.policy import (
    RULES as V1_RULES,
)
from embodied_dataops.quality.policy import (
    THRESHOLDS as V1_THRESHOLDS,
)

#: Version string bound to every ``quality-report/2.0`` row and payload.
QC_POLICY_V2_VERSION = "qc-policy/2.0"
#: Schema version of the ``quality.json`` artefact produced by this policy.
QUALITY_REPORT_V2_SCHEMA_VERSION = "quality-report/2.0"

#: Values accepted by ``GET /episodes/{id}/quality?policy_version=``.
QC_POLICY_VERSIONS: tuple[str, ...] = (QC_POLICY_VERSION, QC_POLICY_V2_VERSION)


class SemanticRuleCode(StrEnum):
    """The eight robot-semantic rule codes added by ``qc-policy/2.0``."""

    MISSING_ROBOT_PROFILE = "MISSING_ROBOT_PROFILE"
    JOINT_POSITION_OUT_OF_RANGE = "JOINT_POSITION_OUT_OF_RANGE"
    JOINT_VELOCITY_SPIKE = "JOINT_VELOCITY_SPIKE"
    CLOCK_OFFSET_UNVERIFIED = "CLOCK_OFFSET_UNVERIFIED"
    ACTION_STATE_LAG = "ACTION_STATE_LAG"
    EPISODE_INCOMPLETE = "EPISODE_INCOMPLETE"
    POINTCLOUD_CORRUPT = "POINTCLOUD_CORRUPT"
    POINTCLOUD_NONFINITE_HIGH = "POINTCLOUD_NONFINITE_HIGH"


#: ``qc-policy/1.0`` rule codes carried into 2.0 with unchanged thresholds.
CARRIED_V1_RULE_CODES: tuple[str, ...] = tuple(code.value for code in RuleCode)


@dataclass(frozen=True)
class QcV2Thresholds:
    """Frozen numeric thresholds of ``qc-policy/2.0``.

    ``v1`` keeps the carried rule thresholds exactly as ``qc-policy/1.0``
    declared them; V2 rules use the explicit fields below.
    """

    #: Absolute tolerance around a profile joint limit (SPEC §3.1).
    joint_position_tolerance_rad: float = 1e-5
    #: Velocity spike factor: strictly greater than ``vmax * 1.10`` violates.
    velocity_spike_factor: float = 1.10
    #: ACTION_STATE_LAG median absolute offset threshold in ns (strictly >).
    action_state_lag_median_ns: int = 150_000_000
    #: ACTION_STATE_LAG requires at least this many aligned points.
    action_state_lag_min_points: int = 20
    #: POINTCLOUD_NONFINITE_HIGH ratio threshold (strictly > 0.10).
    pointcloud_nonfinite_ratio: float = 0.10
    #: Maximum finding rows persisted for one Episode (DO2-13).
    max_findings_per_report: int = 10_000
    #: Maximum merged time ranges kept on one finding row (bounded evidence).
    max_merged_ranges_per_finding: int = 10_000
    #: Bound on individual violation evidence items kept per finding.
    max_evidence_items: int = 200
    #: Carried ``qc-policy/1.0`` thresholds (never re-declared here).
    v1: QcThresholds = V1_THRESHOLDS


#: The single frozen threshold instance of qc-policy/2.0.
THRESHOLDS_V2 = QcV2Thresholds()


@dataclass(frozen=True)
class RuleDefinitionV2:
    """Declarative definition of one ``qc-policy/2.0`` rule."""

    code: str
    severity: Severity
    applies_to: str
    description: str
    threshold: dict[str, Any]
    view: str
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_code": self.code,
            "severity": self.severity.value,
            "applies_to": self.applies_to,
            "description": self.description,
            "threshold": self.threshold,
            "view": self.view,
            "method": self.method,
        }


def _semantic_rule(
    code: SemanticRuleCode,
    severity: Severity,
    applies_to: str,
    description: str,
    threshold: dict[str, Any],
    *,
    view: str,
    method: str,
) -> RuleDefinitionV2:
    return RuleDefinitionV2(
        code=code.value,
        severity=severity,
        applies_to=applies_to,
        description=description,
        threshold=threshold,
        view=view,
        method=method,
    )


#: The eight new rules in deterministic evaluation/report order.
SEMANTIC_RULES: dict[str, RuleDefinitionV2] = {
    SemanticRuleCode.MISSING_ROBOT_PROFILE.value: _semantic_rule(
        SemanticRuleCode.MISSING_ROBOT_PROFILE,
        Severity.ERROR,
        "robot episodes",
        "no explicit robot profile mapping and no trustworthy profile binding; semantic QC "
        "cannot be inferred from tensor shape, file name or model text",
        {"requires_explicit_profile": True, "inference_from_shape_forbidden": True},
        view="episode",
        method="profile_binding_check",
    ),
    SemanticRuleCode.JOINT_POSITION_OUT_OF_RANGE.value: _semantic_rule(
        SemanticRuleCode.JOINT_POSITION_OUT_OF_RANGE,
        Severity.ERROR,
        "valid native state (fallback: valid aligned state)",
        "a valid state dimension is strictly below lower_i-1e-5 or strictly above upper_i+1e-5",
        {
            "tolerance_rad": THRESHOLDS_V2.joint_position_tolerance_rad,
            "comparison": "value < lower-1e-5 or value > upper+1e-5 (strict)",
        },
        view="native+aligned",
        method="profile_joint_limit_check",
    ),
    SemanticRuleCode.JOINT_VELOCITY_SPIKE.value: _semantic_rule(
        SemanticRuleCode.JOINT_VELOCITY_SPIKE,
        Severity.ERROR,
        "adjacent valid native state samples in one clock domain",
        "abs(dq_j)/dt_s > max_abs_velocity_rad_s[j] * 1.10 with dt_s > 0; the implied "
        "velocity and the two samples are recorded as evidence",
        {
            "velocity_factor": THRESHOLDS_V2.velocity_spike_factor,
            "requires_positive_dt": True,
            "comparison": "abs(dq)/dt_s > vmax*1.10 (strict)",
        },
        view="native",
        method="adjacent_valid_state_finite_difference",
    ),
    SemanticRuleCode.CLOCK_OFFSET_UNVERIFIED.value: _semantic_rule(
        SemanticRuleCode.CLOCK_OFFSET_UNVERIFIED,
        Severity.WARN,
        "alignment across independent clock domains",
        "alignment needs cross-domain alignment but no calibration evidence exists; a "
        "single reliable source clock never triggers this rule",
        {"requires_calibration_evidence": True, "max_clock_offset_ns_from_profile": True},
        view="aligned",
        method="clock_domain_calibration_check",
    ),
    SemanticRuleCode.ACTION_STATE_LAG.value: _semantic_rule(
        SemanticRuleCode.ACTION_STATE_LAG,
        Severity.WARN,
        "verified common timebase + target-position action semantics",
        "median abs(action_source_t_ns - compared state sample time) > 150 ms over at "
        "least 20 aligned points; this is a lag indicator, never a causal control delay",
        {
            "median_abs_offset_ns": THRESHOLDS_V2.action_state_lag_median_ns,
            "median_abs_offset_ms": THRESHOLDS_V2.action_state_lag_median_ns / 1e6,
            "min_aligned_points": THRESHOLDS_V2.action_state_lag_min_points,
            "causal_claim": False,
        },
        view="aligned",
        method="aligned_action_state_offset_median",
    ),
    SemanticRuleCode.EPISODE_INCOMPLETE.value: _semantic_rule(
        SemanticRuleCode.EPISODE_INCOMPLETE,
        Severity.ERROR,
        "source declarations",
        "the source declares expected_terminal_marker=true and no real terminal marker "
        "exists, or a manifest-required modality is missing; task success is never "
        "inferred from action magnitude",
        {"requires_source_declaration": True, "terminal_marker_inferred_from_actions": False},
        view="episode",
        method="source_terminal_and_modality_check",
    ),
    SemanticRuleCode.POINTCLOUD_CORRUPT.value: _semantic_rule(
        SemanticRuleCode.POINTCLOUD_CORRUPT,
        Severity.ERROR,
        "PointCloud2 sources",
        "PointCloud2 has an illegal data length, overlapping/out-of-range fields or an "
        "unsupported datatype/layout",
        {"bounded_parse": True, "raw_coordinates_retained": False},
        view="pointcloud",
        method="pointcloud2_structural_validation",
    ),
    SemanticRuleCode.POINTCLOUD_NONFINITE_HIGH.value: _semantic_rule(
        SemanticRuleCode.POINTCLOUD_NONFINITE_HIGH,
        Severity.WARN,
        "PointCloud2 sources",
        "non-finite XYZ points / total points > 0.10 (strictly greater)",
        {
            "nonfinite_ratio_threshold": THRESHOLDS_V2.pointcloud_nonfinite_ratio,
            "comparison": "ratio > 0.10 (strict)",
        },
        view="pointcloud",
        method="pointcloud2_nonfinite_ratio_sampled",
    ),
}


def rule_definition_v2(code: str | StrEnum) -> RuleDefinitionV2:
    """Resolve a ``qc-policy/2.0`` rule definition (new or carried)."""
    value = code.value if isinstance(code, StrEnum) else str(code)
    definition = SEMANTIC_RULES.get(value)
    if definition is not None:
        return definition
    carried = V1_RULES.get(RuleCode(value))
    if carried is not None:
        return RuleDefinitionV2(
            code=carried.code.value,
            severity=carried.severity,
            applies_to=carried.applies_to,
            description=carried.description,
            threshold=dict(carried.threshold),
            view="native+aligned",
            method=f"qc-policy/1.0-carried:{carried.code.value}",
        )
    raise KeyError(f"rule code {value!r} is not part of qc-policy/2.0")


def policy_config() -> dict[str, Any]:
    """Canonical, SHA-stable policy configuration for ``qc-policy/2.0``.

    Only version, thresholds and rule definitions participate: documentation
    strings are excluded so a comment edit cannot silently change the SHA.  The
    carried 1.0 rules are included because changing any of them changes what a
    ``quality-report/2.0`` means.
    """
    return {
        "policy_version": QC_POLICY_V2_VERSION,
        "thresholds": {
            "joint_position_tolerance_rad": THRESHOLDS_V2.joint_position_tolerance_rad,
            "velocity_spike_factor": THRESHOLDS_V2.velocity_spike_factor,
            "action_state_lag_median_ns": THRESHOLDS_V2.action_state_lag_median_ns,
            "action_state_lag_min_points": THRESHOLDS_V2.action_state_lag_min_points,
            "pointcloud_nonfinite_ratio": THRESHOLDS_V2.pointcloud_nonfinite_ratio,
            "max_findings_per_report": THRESHOLDS_V2.max_findings_per_report,
            "carried_v1": {
                "max_duration_ns": THRESHOLDS_V2.v1.max_duration_ns,
                "frame_gap_max_ns": THRESHOLDS_V2.v1.frame_gap_max_ns,
                "image_tolerance_ns": THRESHOLDS_V2.v1.image_tolerance_ns,
                "state_tolerance_ns": THRESHOLDS_V2.v1.state_tolerance_ns,
                "action_zoh_tolerance_ns": THRESHOLDS_V2.v1.action_zoh_tolerance_ns,
                "low_conf_threshold": THRESHOLDS_V2.v1.low_conf_threshold,
                "blur_laplacian_variance_threshold": (
                    THRESHOLDS_V2.v1.blur_laplacian_variance_threshold
                ),
                "blur_frame_ratio_threshold": THRESHOLDS_V2.v1.blur_frame_ratio_threshold,
            },
        },
        "rules": [
            {
                "rule_code": code,
                "severity": V1_RULES[RuleCode(code)].severity.value,
                "threshold": dict(V1_RULES[RuleCode(code)].threshold),
                "policy_origin": "qc-policy/1.0",
            }
            for code in CARRIED_V1_RULE_CODES
        ]
        + [
            {
                "rule_code": definition.code,
                "severity": definition.severity.value,
                "threshold": dict(definition.threshold),
                "policy_origin": QC_POLICY_V2_VERSION,
            }
            for definition in SEMANTIC_RULES.values()
        ],
    }


def policy_config_sha256() -> str:
    """SHA-256 of the canonical ``qc-policy/2.0`` configuration."""
    from embodied_dataops.canonical.contract import canonical_json_sha256

    return canonical_json_sha256(policy_config())


#: Frozen SHA-256 of :func:`policy_config`; bound to every ``quality-report/2.0``.
POLICY_CONFIG_SHA256 = policy_config_sha256()


def verdict_for(findings) -> QualityStatus:
    """Aggregate findings with the unchanged ``qc-policy/1.0`` verdict rules."""
    return decide_verdict(findings)


def policy_document_v2() -> dict[str, Any]:
    """Machine-readable ``qc-policy/2.0`` document (docs / API / policy tests)."""
    return {
        "policy_version": QC_POLICY_V2_VERSION,
        "schema_version": QUALITY_REPORT_V2_SCHEMA_VERSION,
        "config_sha256": POLICY_CONFIG_SHA256,
        "verdict_rules": {
            "any_error": QualityStatus.FAIL.value,
            "else_any_warn": QualityStatus.WARN.value,
            "else": QualityStatus.PASS.value,
            "info_never_changes_verdict": True,
        },
        "thresholds": policy_config()["thresholds"],
        "carried_v1_rules": [dict(V1_RULES[RuleCode(code)].as_dict()) for code in CARRIED_V1_RULE_CODES],
        "rules": [definition.as_dict() for definition in SEMANTIC_RULES.values()],
    }


__all__ = [
    "CARRIED_V1_RULE_CODES",
    "POLICY_CONFIG_SHA256",
    "QC_POLICY_V2_VERSION",
    "QC_POLICY_VERSIONS",
    "QUALITY_REPORT_V2_SCHEMA_VERSION",
    "QcV2Thresholds",
    "RuleDefinitionV2",
    "SEMANTIC_RULES",
    "SemanticRuleCode",
    "THRESHOLDS_V2",
    "policy_config",
    "policy_config_sha256",
    "policy_document_v2",
    "rule_definition_v2",
    "verdict_for",
]
