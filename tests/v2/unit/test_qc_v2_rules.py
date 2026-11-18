"""``qc-policy/2.0`` rule tests (V2 SPEC §3.1, DO2-10..13).

Every rule has a positive, a negative and an exact-boundary case:

* ``JOINT_POSITION_OUT_OF_RANGE``: ``lower-1e-5`` / ``upper+1e-5`` are inside the
  allowed range, one ulp beyond is a finding;
* ``JOINT_VELOCITY_SPIKE``: ``abs(dq)/dt == vmax*1.10`` is allowed, above is ERROR,
  ``dt <= 0`` never fires and only adjacent *valid* same-clock samples count;
* ``ACTION_STATE_LAG``: 150 ms median exactly is allowed, > 150 ms fires, 19
  points never fire, an unverified timebase never fires, no causal claim;
* ``POINTCLOUD_NONFINITE_HIGH``: ratio 0.10 exactly is allowed, > 0.10 is WARN;
* ``MISSING_ROBOT_PROFILE`` / ``UNVERIFIED_SEMANTICS``: profile authority is
  mandatory and never inferred from shape;
* carried V1 rules are evaluated on the native *and* aligned canonical views.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fixtures.v2 import semantic_qc_cases as fixtures

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.domain.enums import Severity
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.quality.rules import TimeRange, VideoInfo
from embodied_dataops.quality.v2_policy import (
    POLICY_CONFIG_SHA256,
    QC_POLICY_V2_VERSION,
    QUALITY_REPORT_V2_SCHEMA_VERSION,
    SEMANTIC_RULES,
    SemanticRuleCode,
    policy_config,
    policy_config_sha256,
    policy_document_v2,
)
from embodied_dataops.quality.v2_report import (
    MAX_FINDING_ROWS,
    QualityReportTooLarge,
    merge_semantic_findings,
    prepare_quality_report_v2,
)
from embodied_dataops.quality.v2_rules import (
    AlignedView,
    ClockDomainInfo,
    NativeStream,
    PointCloudSummary,
    SemanticFinding,
    SemanticQcInput,
    check_action_state_lag,
    check_clock_offset_unverified,
    check_dimension_mismatch_v2,
    check_episode_incomplete,
    check_joint_position_out_of_range,
    check_joint_velocity_spike,
    check_missing_robot_profile,
    check_pointcloud_corrupt,
    check_pointcloud_nonfinite_high,
    check_unverified_semantics_v2,
    evaluate_semantic_quality,
)

pytestmark = pytest.mark.unit

INTERVAL = SAMPLE_INTERVAL_NS
ORIGIN = fixtures.ORIGIN_NS


# ---------------------------------------------------------------------------
# input builders
# ---------------------------------------------------------------------------


def _times(count: int) -> tuple[list[int], list[int]]:
    return (
        [ORIGIN + index * INTERVAL for index in range(count)],
        [index * INTERVAL for index in range(count)],
    )


def _state_stream(
    values: list[list[float]],
    *,
    valid: list[bool] | None = None,
    clock_domain: str = fixtures.CLOCK_LOG,
) -> NativeStream:
    source, episode = _times(len(values))
    return NativeStream(
        name="state",
        kind="state",
        clock_domain=clock_domain,
        timestamp_provenance="mcap_log_time",
        clock_verified=True,
        source_t_ns=source,
        episode_t_ns=episode,
        valid=valid if valid is not None else [True] * len(values),
        values=values,
    )


def _base_input(
    state_values: list[list[float]],
    *,
    profile_id: str = "arm6_joint_position_v1",
    valid: list[bool] | None = None,
    clock_domain: str = fixtures.CLOCK_LOG,
    duration_ns: int | None = None,
) -> SemanticQcInput:
    config = fixtures.profile_config(profile_id)
    binding = fixtures.profile_binding(profile_id)
    count = len(state_values)
    return SemanticQcInput(
        episode_id="unit-episode",
        episode_type="robot_episode",
        source_type="MCAP_ROS2_PROFILE_V2",
        data_origin="synthetic",
        canonical_version=1,
        policy_version=QC_POLICY_V2_VERSION,
        duration_ns=duration_ns if duration_ns is not None else count * INTERVAL,
        profile=binding,
        native_state=_state_stream(state_values, valid=valid, clock_domain=clock_domain),
        clock_domains={
            "state": ClockDomainInfo("state", clock_domain, "mcap_log_time", True),
        },
        joint_names=list(config.joint_names),
        video=VideoInfo(expected=True, decode_ok=True, frame_count=count, decoded_frames=count),
        metadata={"schema_version": "canonical-episode/2.0"},
    )


def _aligned_input(
    *,
    points: int = 40,
    action_offset_ns: int = 0,
    clock_verified: bool = True,
    action_semantics: str | None = None,
    invalid_reason: dict[int, list[str]] | None = None,
    missing_state: set[int] | None = None,
    profile_id: str = "arm6_joint_position_v1",
    state_values: list[list[float]] | None = None,
) -> SemanticQcInput:
    config = fixtures.profile_config(profile_id)
    dim = config.state_dim
    base = state_values or [
        [0.05 * (joint + 1) + 0.001 * index for joint in range(dim)] for index in range(points)
    ]
    times_relative = [index * INTERVAL for index in range(points)]
    missing = missing_state or set()
    state: list[list[float] | None] = [
        None if index in missing else base[min(index, len(base) - 1)] for index in range(points)
    ]
    action: list[list[float] | None] = [
        None if index in missing else base[min(index, len(base) - 1)] for index in range(points)
    ]
    reasons = [
        (invalid_reason or {}).get(index, ["missing_state"] if index in missing else [])
        for index in range(points)
    ]
    binding = fixtures.profile_binding(profile_id)
    if action_semantics is not None:
        updated = config.model_copy(update={"action_semantics": action_semantics})
        binding = replace(
            binding, config=updated, config_sha256=updated.config_sha256()
        )
    aligned = AlignedView(
        t_ns=times_relative,
        state=state,
        action=action,
        state_source_t_ns=times_relative,
        action_source_t_ns=[value + action_offset_ns for value in times_relative],
        valid=[state[index] is not None for index in range(points)],
        invalid_reason=reasons,
        clock_verified=clock_verified,
    )
    return SemanticQcInput(
        episode_id="unit-aligned",
        episode_type="robot_episode",
        source_type="MCAP_ROS2_PROFILE_V2",
        data_origin="synthetic",
        canonical_version=1,
        policy_version=QC_POLICY_V2_VERSION,
        duration_ns=points * INTERVAL,
        profile=binding,
        native_state=_state_stream(base),
        aligned=aligned,
        clock_domains={
            "state": ClockDomainInfo("state", fixtures.CLOCK_LOG, "mcap_log_time", clock_verified),
            "front": ClockDomainInfo("front", fixtures.CLOCK_LOG, "mcap_log_time", clock_verified),
        },
        joint_names=list(config.joint_names),
        video=VideoInfo(expected=True, decode_ok=True, frame_count=points, decoded_frames=points),
        metadata={"schema_version": "canonical-episode/2.0"},
    )


def _codes(inp: SemanticQcInput) -> list[str]:
    return [finding.rule_code for finding in evaluate_semantic_quality(inp).findings]


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def test_policy_declares_all_eight_semantic_rules_with_required_severity() -> None:
    expected = {
        "MISSING_ROBOT_PROFILE": Severity.ERROR,
        "JOINT_POSITION_OUT_OF_RANGE": Severity.ERROR,
        "JOINT_VELOCITY_SPIKE": Severity.ERROR,
        "CLOCK_OFFSET_UNVERIFIED": Severity.WARN,
        "ACTION_STATE_LAG": Severity.WARN,
        "EPISODE_INCOMPLETE": Severity.ERROR,
        "POINTCLOUD_CORRUPT": Severity.ERROR,
        "POINTCLOUD_NONFINITE_HIGH": Severity.WARN,
    }
    assert {code.value for code in SemanticRuleCode} == set(expected)
    assert {code: definition.severity for code, definition in SEMANTIC_RULES.items()} == expected


def test_policy_config_sha_is_frozen_and_covers_thresholds() -> None:
    assert len(POLICY_CONFIG_SHA256) == 64
    assert policy_config_sha256() == POLICY_CONFIG_SHA256
    config = policy_config()
    assert config["policy_version"] == QC_POLICY_V2_VERSION
    thresholds = config["thresholds"]
    assert thresholds["joint_position_tolerance_rad"] == 1e-5
    assert thresholds["velocity_spike_factor"] == 1.10
    assert thresholds["action_state_lag_median_ns"] == 150_000_000
    assert thresholds["action_state_lag_min_points"] == 20
    assert thresholds["pointcloud_nonfinite_ratio"] == 0.10
    assert thresholds["max_findings_per_report"] == 10_000
    document = policy_document_v2()
    assert document["schema_version"] == QUALITY_REPORT_V2_SCHEMA_VERSION
    assert document["config_sha256"] == POLICY_CONFIG_SHA256
    assert len(document["carried_v1_rules"]) == 13
    assert "ACTION_STATE_LAG" in [rule["rule_code"] for rule in document["rules"]]


# ---------------------------------------------------------------------------
# MISSING_ROBOT_PROFILE
# ---------------------------------------------------------------------------


def test_missing_robot_profile_positive_lists_missing_mapping_fields() -> None:
    inp = fixtures.build_case_input("missing_profile")
    findings = check_missing_robot_profile(inp)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_code == "MISSING_ROBOT_PROFILE"
    assert finding.severity == Severity.ERROR
    assert "profile_id" in finding.observed["missing_mapping_fields"]
    assert finding.evidence["profile_inferred_from_shape"] is False


def test_missing_robot_profile_negative_when_profile_is_bound() -> None:
    assert check_missing_robot_profile(_base_input([[0.1] * 6, [0.1] * 6])) == []


def test_missing_robot_profile_diagnostic_only_mode_still_errors() -> None:
    inp = _base_input([[0.1] * 6])
    inp.profile = replace(inp.profile, config=None, mode="DIAGNOSTIC_ONLY")
    findings = check_missing_robot_profile(inp)
    assert len(findings) == 1
    assert findings[0].evidence["diagnostic_only"] is True


# ---------------------------------------------------------------------------
# JOINT_POSITION_OUT_OF_RANGE (exact +-1e-5 boundary)
# ---------------------------------------------------------------------------


def _position_row(**joint_values: float) -> list[float]:
    row = [0.0] * 6
    for key, value in joint_values.items():
        index = int(key.replace("joint", "")) - 1
        row[index] = value
    return row


def test_joint_position_boundary_lower_minus_tolerance_is_allowed() -> None:
    lower = -3.14
    row = _position_row(joint1=lower - 1e-5)
    assert check_joint_position_out_of_range(_base_input([row])) == []


def test_joint_position_below_lower_minus_tolerance_is_error() -> None:
    lower = -3.14
    row = _position_row(joint1=lower - 1e-5 - 1e-9)
    findings = check_joint_position_out_of_range(_base_input([row]))
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_code == "JOINT_POSITION_OUT_OF_RANGE"
    assert finding.severity == Severity.ERROR
    assert finding.observed["joint"] == "joint_1"
    assert finding.evidence["violations"][0]["bound"] == "below"
    assert finding.affected_time_ranges == (TimeRange(0, INTERVAL),)


def test_joint_position_boundary_upper_plus_tolerance_is_allowed() -> None:
    upper = 3.14
    row = _position_row(joint6=upper + 1e-5)
    assert check_joint_position_out_of_range(_base_input([row])) == []


def test_joint_position_above_upper_plus_tolerance_is_error_per_joint() -> None:
    upper = 3.14
    row = _position_row(joint6=upper + 1e-5 + 1e-9, joint2=upper + 0.5)
    findings = check_joint_position_out_of_range(_base_input([row]))
    assert {finding.observed["joint"] for finding in findings} == {"joint_6", "joint_2"}
    assert all(finding.severity == Severity.ERROR for finding in findings)
    assert all(finding.source_stream == "state" for finding in findings)


def test_joint_position_ignores_invalid_state_rows() -> None:
    row = _position_row(joint1=3.5)
    inp = _base_input([row], valid=[False])
    assert check_joint_position_out_of_range(inp) == []


def test_joint_position_skips_continuous_joints_but_keeps_others() -> None:
    config = fixtures.profile_config("arm6_joint_position_v1").model_copy(
        update={"continuous_joints": [True, False, False, False, False, False]}
    )
    binding = replace(
        fixtures.profile_binding("arm6_joint_position_v1"),
        config=config,
        config_sha256=config.config_sha256(),
    )
    row = _position_row(joint1=3.6, joint2=3.6)
    inp = _base_input([row])
    inp.profile = binding
    findings = check_joint_position_out_of_range(inp)
    assert [finding.observed["joint"] for finding in findings] == ["joint_2"]


def test_joint_position_falls_back_to_aligned_view() -> None:
    inp = _aligned_input(state_values=[[3.5] + [0.0] * 5] * 5)
    inp.native_state = None
    findings = check_joint_position_out_of_range(inp)
    assert findings and findings[0].observed["canonical_view"] == "aligned"


# ---------------------------------------------------------------------------
# JOINT_VELOCITY_SPIKE (vmax * 1.10, dt > 0, adjacent valid same clock)
# ---------------------------------------------------------------------------


def test_joint_velocity_exactly_vmax_times_factor_is_allowed() -> None:
    vmax = fixtures.profile_config("arm6_joint_position_v1").max_abs_velocity_rad_s[0]
    limit = vmax * 1.10
    dt_s = 1.0
    delta = limit * dt_s
    rows = [[0.0] * 6, [delta] + [0.0] * 5]
    source, episode = _times(2)
    stream = NativeStream(
        "state", "state", fixtures.CLOCK_LOG, "mcap_log_time", True, source, [0, 1_000_000_000], [True, True], rows
    )
    inp = _base_input(rows)
    inp.native_state = stream
    assert check_joint_velocity_spike(inp) == []


def test_joint_velocity_above_vmax_times_factor_is_error_with_evidence() -> None:
    vmax = fixtures.profile_config("arm6_joint_position_v1").max_abs_velocity_rad_s[0]
    limit = vmax * 1.10
    delta = limit * 1.000001
    rows = [[0.0] * 6, [delta] + [0.0] * 5]
    source, _episode = _times(2)
    stream = NativeStream(
        "state", "state", fixtures.CLOCK_LOG, "mcap_log_time", True, source, [0, 1_000_000_000], [True, True], rows
    )
    inp = _base_input(rows)
    inp.native_state = stream
    findings = check_joint_velocity_spike(inp)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_code == "JOINT_VELOCITY_SPIKE"
    assert finding.severity == Severity.ERROR
    window = finding.evidence["windows"][0]
    assert window["dt_s"] > 0
    assert window["implied_velocity_rad_s"] > window["limit_rad_s"]
    assert finding.observed["max_implied_velocity_rad_s"] > limit


def test_joint_velocity_non_positive_dt_never_fires() -> None:
    rows = [[0.0] * 6, [5.0] + [0.0] * 5]
    source, _episode = _times(2)
    stream = NativeStream(
        "state", "state", fixtures.CLOCK_LOG, "mcap_log_time", True, source, [5, 5], [True, True], rows
    )
    inp = _base_input(rows)
    inp.native_state = stream
    assert check_joint_velocity_spike(inp) == []


def test_joint_velocity_skips_invalid_samples() -> None:
    rows = [[0.0] * 6, [9.0] + [0.0] * 5, [0.0] * 6]
    source, episode = _times(3)
    stream = NativeStream(
        "state",
        "state",
        fixtures.CLOCK_LOG,
        "mcap_log_time",
        True,
        source,
        episode,
        [True, False, True],
        rows,
    )
    inp = _base_input(rows)
    inp.native_state = stream
    findings = check_joint_velocity_spike(inp)
    # the invalid middle sample is skipped: the valid pair (0, 2) implies 0 rad/s
    assert findings == []


def test_joint_velocity_requires_one_clock_domain() -> None:
    rows = [[0.0] * 6, [5.0] + [0.0] * 5]
    inp = _base_input(rows, clock_domain="")
    inp.clock_domains = {}
    assert check_joint_velocity_spike(inp) == []


def test_joint_velocity_continuous_joint_uses_wrapped_difference() -> None:
    config = fixtures.profile_config("arm6_joint_position_v1").model_copy(
        update={"continuous_joints": [True, False, False, False, False, False]}
    )
    binding = replace(
        fixtures.profile_binding("arm6_joint_position_v1"),
        config=config,
        config_sha256=config.config_sha256(),
    )
    rows = [[3.13] + [0.0] * 5, [-3.13] + [0.0] * 5]
    source, _episode = _times(2)
    stream = NativeStream(
        "state", "state", fixtures.CLOCK_LOG, "mcap_log_time", True, source, [0, 100_000_000], [True, True], rows
    )
    inp = _base_input(rows)
    inp.profile = binding
    inp.native_state = stream
    assert check_joint_velocity_spike(inp) == []
    # the same raw delta without continuous handling would be a spike
    inp.profile = fixtures.profile_binding("arm6_joint_position_v1")
    assert check_joint_velocity_spike(inp)


def test_joint_velocity_fixture_positive() -> None:
    findings = check_joint_velocity_spike(fixtures.build_case_input("velocity_spike"))
    assert len(findings) == 1
    assert findings[0].observed["violation_count"] >= 1
    assert findings[0].evidence["same_clock_domain"] == fixtures.CLOCK_LOG


# ---------------------------------------------------------------------------
# CLOCK_OFFSET_UNVERIFIED
# ---------------------------------------------------------------------------


def test_clock_offset_unverified_warns_for_two_domains_without_calibration() -> None:
    inp = fixtures.build_case_input("clock_unverified")
    findings = check_clock_offset_unverified(inp)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_code == "CLOCK_OFFSET_UNVERIFIED"
    assert finding.severity == Severity.WARN
    assert finding.observed["independent_clock_domain_count"] == 2
    assert finding.observed["missing_calibration_pairs"]
    assert finding.evidence["missing_calibration_record_ids"]


def test_clock_offset_unverified_not_triggered_for_single_domain() -> None:
    assert check_clock_offset_unverified(fixtures.build_case_input("good_6d")) == []


def test_clock_offset_unverified_accepts_calibrated_pair() -> None:
    inp = fixtures.build_case_input("clock_unverified")
    inp.calibration_evidence = [
        {
            "id": "cal-1",
            "clock_domains": [fixtures.CLOCK_LOG, fixtures.CLOCK_STAMP],
            "verified": True,
            "max_offset_ns": 20_000_000,
        }
    ]
    assert check_clock_offset_unverified(inp) == []


def test_clock_offset_unverified_accepts_verified_streams() -> None:
    inp = fixtures.build_case_input("clock_unverified")
    inp.clock_domains = {
        name: replace(info, clock_verified=True) for name, info in inp.clock_domains.items()
    }
    assert check_clock_offset_unverified(inp) == []


def test_clock_offset_unverified_rejects_calibration_above_profile_limit() -> None:
    inp = fixtures.build_case_input("clock_unverified")
    inp.calibration_evidence = [
        {
            "id": "cal-over-limit",
            "clock_domains": [fixtures.CLOCK_LOG, fixtures.CLOCK_STAMP],
            "verified": True,
            "max_offset_ns": 80_000_000,  # profile max_clock_offset_ns is 50 ms
        }
    ]
    assert len(check_clock_offset_unverified(inp)) == 1


# ---------------------------------------------------------------------------
# ACTION_STATE_LAG (median > 150 ms, >= 20 points, verified timebase only)
# ---------------------------------------------------------------------------


def test_action_state_lag_median_exactly_150ms_is_allowed() -> None:
    inp = _aligned_input(points=20, action_offset_ns=-150_000_000)
    assert check_action_state_lag(inp) == []


def test_action_state_lag_above_150ms_is_warn_without_causal_claim() -> None:
    inp = _aligned_input(points=20, action_offset_ns=-150_000_001)
    findings = check_action_state_lag(inp)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_code == "ACTION_STATE_LAG"
    assert finding.severity == Severity.WARN
    assert finding.observed["median_abs_offset_ns"] == 150_000_001
    assert finding.observed["aligned_point_count"] == 20
    assert finding.evidence["no_causal_claim"] is True
    assert "not" in finding.evidence["limitation"].lower()
    assert finding.uncertainty_note


def test_action_state_lag_needs_at_least_20_points() -> None:
    inp = _aligned_input(points=19, action_offset_ns=-200_000_000)
    assert check_action_state_lag(inp) == []


def test_action_state_lag_not_triggered_for_unverified_timebase() -> None:
    inp = _aligned_input(points=20, action_offset_ns=-200_000_000, clock_verified=False)
    assert check_action_state_lag(inp) == []


def test_action_state_lag_not_triggered_for_non_target_position_actions() -> None:
    inp = _aligned_input(
        points=20, action_offset_ns=-200_000_000, action_semantics="joint_velocity"
    )
    assert check_action_state_lag(inp) == []


def test_action_state_lag_fixture_positive() -> None:
    findings = check_action_state_lag(fixtures.build_case_input("action_state_lag"))
    assert len(findings) == 1
    assert findings[0].observed["median_abs_offset_ms"] == 200.0


# ---------------------------------------------------------------------------
# EPISODE_INCOMPLETE (source declarations only)
# ---------------------------------------------------------------------------


def test_episode_incomplete_terminal_marker_missing_is_error() -> None:
    inp = fixtures.build_case_input("terminal_missing")
    findings = check_episode_incomplete(inp)
    assert len(findings) == 1
    assert findings[0].rule_code == "EPISODE_INCOMPLETE"
    assert findings[0].severity == Severity.ERROR
    assert findings[0].observed["terminal_marker_missing"] is True
    assert findings[0].evidence["success_inferred_from_action_magnitude"] is False


def test_episode_incomplete_not_triggered_when_marker_present() -> None:
    inp = fixtures.build_case_input("terminal_missing")
    inp.terminal_marker_present = True
    assert check_episode_incomplete(inp) == []


def test_episode_incomplete_not_triggered_without_source_declaration() -> None:
    inp = fixtures.build_case_input("good_6d")
    inp.expected_terminal_marker = False
    inp.terminal_marker_present = False
    assert check_episode_incomplete(inp) == []


def test_episode_incomplete_required_modality_missing_is_error() -> None:
    inp = fixtures.build_case_input("good_6d")
    inp.required_modalities = ["front", "wrist"]
    inp.present_modalities = ["front", "state", "action"]
    findings = check_episode_incomplete(inp)
    assert len(findings) == 1
    assert findings[0].observed["missing_required_modalities"] == ["wrist"]


# ---------------------------------------------------------------------------
# POINTCLOUD_CORRUPT
# ---------------------------------------------------------------------------


def test_pointcloud_corrupt_is_error_with_structural_details() -> None:
    inp = fixtures.build_case_input("pointcloud_corrupt")
    findings = check_pointcloud_corrupt(inp)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_code == "POINTCLOUD_CORRUPT"
    assert finding.severity == Severity.ERROR
    assert finding.observed["corrupt_topic_count"] == 1
    assert finding.observed["details"][0]["topic"] == "/points"
    assert finding.evidence["raw_coordinates_retained"] is False


def test_pointcloud_corrupt_absent_section_does_not_crash_or_fire() -> None:
    inp = fixtures.build_case_input("good_6d")
    assert check_pointcloud_corrupt(inp) == []


def test_pointcloud_corrupt_from_error_list_without_status() -> None:
    inp = fixtures.build_case_input("good_6d")
    inp.pointcloud = [PointCloudSummary(topic="/points", errors=["field offset overlaps"])]
    findings = check_pointcloud_corrupt(inp)
    assert len(findings) == 1


# ---------------------------------------------------------------------------
# POINTCLOUD_NONFINITE_HIGH (ratio > 0.10 strict)
# ---------------------------------------------------------------------------


def test_pointcloud_nonfinite_ratio_exactly_10_percent_is_allowed() -> None:
    inp = fixtures.build_case_input("good_6d")
    inp.pointcloud = [
        PointCloudSummary(
            topic="/points",
            point_count=100,
            sampled_point_count=100,
            nonfinite_sample_count=10,
            status="OK",
        )
    ]
    assert check_pointcloud_nonfinite_high(inp) == []


def test_pointcloud_nonfinite_above_10_percent_is_warn() -> None:
    inp = fixtures.build_case_input("pointcloud_nonfinite_high")
    findings = check_pointcloud_nonfinite_high(inp)
    assert len(findings) == 1
    assert findings[0].rule_code == "POINTCLOUD_NONFINITE_HIGH"
    assert findings[0].severity == Severity.WARN
    assert findings[0].observed["max_nonfinite_ratio"] == 0.15


def test_pointcloud_nonfinite_uses_total_points_when_sampling_absent() -> None:
    inp = fixtures.build_case_input("good_6d")
    inp.pointcloud = [
        PointCloudSummary(topic="/points", point_count=100, nonfinite_sample_count=11, status="OK")
    ]
    findings = check_pointcloud_nonfinite_high(inp)
    assert len(findings) == 1
    assert findings[0].observed["max_nonfinite_ratio"] == 0.11


def test_pointcloud_nonfinite_missing_counts_are_skipped() -> None:
    inp = fixtures.build_case_input("good_6d")
    inp.pointcloud = [PointCloudSummary(topic="/points", status="OK")]
    assert check_pointcloud_nonfinite_high(inp) == []


# ---------------------------------------------------------------------------
# DIMENSION_MISMATCH / UNVERIFIED_SEMANTICS
# ---------------------------------------------------------------------------


def test_dimension_mismatch_uses_profile_dimensions_not_hardcoded_six() -> None:
    inp = fixtures.build_case_input("dimension_mismatch")
    findings = check_dimension_mismatch_v2(inp)
    assert findings
    assert findings[0].rule_code == "DIMENSION_MISMATCH"
    assert findings[0].observed["expected_state_dim"] == 6
    assert findings[0].observed["problems"]["state_dim"] == 5


def test_dimension_mismatch_passes_for_seven_dof_profile() -> None:
    inp = fixtures.build_case_input("good_7d")
    assert check_dimension_mismatch_v2(inp) == []


def test_dimension_mismatch_flags_wrong_joint_names() -> None:
    inp = fixtures.build_case_input("good_6d")
    inp.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "elbow"]
    findings = check_dimension_mismatch_v2(inp)
    assert findings and "joint_names" in findings[0].observed["problems"]


def test_unverified_semantics_missing_profile_is_error() -> None:
    findings = check_unverified_semantics_v2(fixtures.build_case_input("missing_profile"))
    assert len(findings) == 1
    assert findings[0].severity == Severity.ERROR
    assert "robot_profile_missing" in findings[0].observed["reasons"]
    assert findings[0].evidence["publishable"] is False


def test_unverified_semantics_unverified_authority_is_error() -> None:
    findings = check_unverified_semantics_v2(fixtures.build_case_input("unverified_authority"))
    assert len(findings) == 1
    assert findings[0].observed["reasons"] == ["profile_authority_unverified"]


def test_unverified_semantics_synthetic_profile_for_public_data_is_error() -> None:
    findings = check_unverified_semantics_v2(
        fixtures.build_case_input("synthetic_profile_public_data")
    )
    assert len(findings) == 1
    assert findings[0].observed["reasons"] == [
        "synthetic_fixture_profile_not_valid_for_non_synthetic_data"
    ]


def test_unverified_semantics_synthetic_fixture_data_passes() -> None:
    assert check_unverified_semantics_v2(fixtures.build_case_input("good_6d")) == []


# ---------------------------------------------------------------------------
# carried qc-policy/1.0 rules on native and aligned views
# ---------------------------------------------------------------------------


def test_carried_timestamp_rule_runs_on_native_camera_view() -> None:
    inp = fixtures.build_case_input("bad_timestamps")
    codes = [finding.rule_code for finding in evaluate_semantic_quality(inp).findings]
    assert "TIMESTAMP_NON_MONOTONIC" in codes


def test_carried_sync_rule_runs_on_aligned_view() -> None:
    evaluation = evaluate_semantic_quality(fixtures.build_case_input("aligned_invalid"))
    codes = [finding.rule_code for finding in evaluation.findings]
    assert "SYNC_MISSING_STATE" in codes
    finding = next(item for item in evaluation.findings if item.rule_code == "SYNC_MISSING_STATE")
    assert finding.observed["missing_grid_point_count"] == 3
    assert finding.source_stream == "aligned_view"


def test_carried_nonfinite_rule_scans_native_state() -> None:
    evaluation = evaluate_semantic_quality(fixtures.build_case_input("nonfinite"))
    codes = [finding.rule_code for finding in evaluation.findings]
    assert "NONFINITE_VALUE" in codes


def test_carried_frame_gap_rule_fires_on_native_camera_gap() -> None:
    inp = fixtures.build_case_input("good_6d")
    camera = inp.native_cameras["front"]
    # Insert a 300 ms gap while keeping the stream monotonic (shift the tail).
    for index in range(10, len(camera.source_t_ns)):
        camera.source_t_ns[index] += 500_000_000
    codes = [finding.rule_code for finding in evaluate_semantic_quality(inp).findings]
    assert "FRAME_GAP_EXCESS" in codes


def test_carried_duration_rule_fires_above_600_seconds() -> None:
    inp = fixtures.build_case_input("good_6d")
    inp.duration_ns = 600 * 1_000_000_000 + 1
    codes = [finding.rule_code for finding in evaluate_semantic_quality(inp).findings]
    assert "DURATION_EXCEEDED" in codes


# ---------------------------------------------------------------------------
# evaluation / aggregation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", fixtures.CASE_NAMES)
def test_fixture_case_produces_exactly_the_expected_rule_codes(case: str) -> None:
    evaluation = evaluate_semantic_quality(fixtures.build_case_input(case))
    assert sorted({finding.rule_code for finding in evaluation.findings}) == sorted(
        fixtures.EXPECTED_RULE_CODES[case]
    )
    assert evaluation.verdict.value == fixtures.EXPECTED_VERDICT[case]


def test_good_fixtures_report_every_semantic_rule_as_run() -> None:
    evaluation = evaluate_semantic_quality(fixtures.build_case_input("good_6d"))
    assert evaluation.findings == []
    assert set(SEMANTIC_RULES).issubset(
        set(evaluation.checks_run) | set(evaluation.checks_not_applicable)
    )


def test_evaluation_metrics_bind_profile_policy_and_views() -> None:
    evaluation = evaluate_semantic_quality(fixtures.build_case_input("good_6d"))
    metrics = evaluation.metrics
    assert metrics["policy_version"] == QC_POLICY_V2_VERSION
    assert metrics["policy_config_sha256"] == POLICY_CONFIG_SHA256
    assert metrics["canonical_version"] == 1
    assert metrics["profile"]["profile_id"] == "arm6_joint_position_v1"
    assert metrics["profile"]["authority"] == "SYNTHETIC_FIXTURE"
    assert metrics["native"]["state_valid_sample_count"] == fixtures.POINT_COUNT
    assert metrics["aligned"]["grid_point_count"] == fixtures.POINT_COUNT


# ---------------------------------------------------------------------------
# finding merge / 10,000-row cap (quality-report/2.0)
# ---------------------------------------------------------------------------


def _finding(index: int, *, rule: str = "JOINT_VELOCITY_SPIKE", start: int | None = None) -> SemanticFinding:
    start_ns = start if start is not None else index * 2 * INTERVAL
    return SemanticFinding(
        rule_code=rule,
        severity=Severity.ERROR,
        observed={"max_implied_velocity_rad_s": float(index), "violation_count": 1},
        threshold={"comparison": "abs(dq)/dt > vmax*1.10"},
        affected_time_ranges=(TimeRange(start_ns, start_ns + INTERVAL),),
        evidence={"window_index": index},
        source_stream="state",
        method="adjacent_valid_state_finite_difference",
    )


def test_merge_overlapping_duplicates_keeps_max_observed_and_ranges() -> None:
    findings = [
        SemanticFinding(
            rule_code="JOINT_VELOCITY_SPIKE",
            severity=Severity.ERROR,
            observed={"max_implied_velocity_rad_s": 7.0, "violation_count": 1},
            threshold={"comparison": "x"},
            affected_time_ranges=(TimeRange(0, 100),),
            evidence={"windows": ["a"]},
            source_stream="state",
        ),
        SemanticFinding(
            rule_code="JOINT_VELOCITY_SPIKE",
            severity=Severity.ERROR,
            observed={"max_implied_velocity_rad_s": 9.5, "violation_count": 2},
            threshold={"comparison": "x"},
            affected_time_ranges=(TimeRange(50, 200),),
            evidence={"windows": ["b"]},
            source_stream="state",
        ),
    ]
    result = merge_semantic_findings(findings)
    assert result.stats.merged_row_count == 1
    assert result.findings[0].observed["max_implied_velocity_rad_s"] == 9.5
    assert result.findings[0].affected_time_ranges == (TimeRange(0, 200),)
    assert result.findings[0].evidence["merged_from_finding_count"] == 2


def test_merge_keeps_disjoint_occurrences_as_separate_rows() -> None:
    findings = [
        SemanticFinding(
            rule_code="JOINT_VELOCITY_SPIKE",
            severity=Severity.ERROR,
            observed={"max_implied_velocity_rad_s": 7.0},
            threshold={},
            affected_time_ranges=(TimeRange(0, 100),),
            source_stream="state",
        ),
        SemanticFinding(
            rule_code="JOINT_VELOCITY_SPIKE",
            severity=Severity.ERROR,
            observed={"max_implied_velocity_rad_s": 8.0},
            threshold={},
            affected_time_ranges=(TimeRange(1000, 1100),),
            source_stream="state",
        ),
    ]
    result = merge_semantic_findings(findings)
    assert result.stats.merged_row_count == 2


def test_more_than_10000_rows_raises_quality_report_too_large_413() -> None:
    findings = [_finding(index) for index in range(MAX_FINDING_ROWS + 1)]
    evaluation = evaluate_semantic_quality(fixtures.build_case_input("good_6d"))
    evaluation.findings = findings
    with pytest.raises(QualityReportTooLarge) as excinfo:
        prepare_quality_report_v2(
            episode_id="unit-episode",
            run_id="unit-run",
            evaluation=evaluation,
            input=fixtures.build_case_input("good_6d"),
            canonical_version=1,
        )
        # the raised ApiError must carry the 413 status and the exact error code
    error = excinfo.value
    assert isinstance(error, ApiError)
    assert error.code == ErrorCode.QUALITY_REPORT_TOO_LARGE
    assert error.status_code == 413
    assert error.details["max_finding_rows"] == MAX_FINDING_ROWS
    assert error.details["merged_finding_row_count"] == MAX_FINDING_ROWS + 1


def test_prepared_payload_carries_v2_schema_profile_and_finding_fields() -> None:
    inp = fixtures.build_case_input("velocity_spike")
    evaluation = evaluate_semantic_quality(inp)
    prepared = prepare_quality_report_v2(
        episode_id=inp.episode_id,
        run_id="unit-run",
        evaluation=evaluation,
        input=inp,
        canonical_version=1,
    )
    payload = prepared.payload
    assert payload["schema_version"] == "quality-report/2.0"
    assert payload["policy_version"] == QC_POLICY_V2_VERSION
    assert payload["canonical_version"] == 1
    assert payload["profile"]["profile_id"] == "arm6_joint_position_v1"
    assert payload["profile"]["config_sha256"]
    finding = payload["findings"][0]
    for key in (
        "source_stream",
        "source_clock_domain",
        "method",
        "limits",
        "uncertainty_note",
    ):
        assert key in finding
    assert payload["metrics"]["report"]["raw_finding_count"] >= 1
