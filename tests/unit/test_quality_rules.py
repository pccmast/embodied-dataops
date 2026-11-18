"""Unit tests for ``qc-policy/1.0`` rules (SPEC 7.1, AT-06 quality parts).

One test per rule, plus exact boundary values:

* image/state tolerance ``<= 50 ms`` passes, ``50 ms + 1 ns`` fails;
* frame gap ``200 ms`` passes, ``200.001 ms`` is a WARN;
* blurry ratio ``30 %`` passes, ``30.001 %`` is a WARN;
* duration ``600 s`` passes, ``600 s + 1 ns`` fails;
* low confidence ``0.35`` is not low, ``0.349999`` is;
* duplicate source timestamps are a TIMESTAMP_NON_MONOTONIC ERROR;
* a 5D action is a DIMENSION_MISMATCH ERROR;
* NaN / Inf are NONFINITE_VALUE errors;
* INFO findings never change the verdict.

Every rule is exercised through its pure function: no database, no object store.
"""

from __future__ import annotations

import json
import math

import pytest

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.domain.enums import QualityStatus, Severity
from embodied_dataops.quality.policy import (
    QC_POLICY_VERSION,
    RULES,
    THRESHOLDS,
    RuleCode,
    decide_verdict,
    policy_document,
)
from embodied_dataops.quality.report import build_quality_report
from embodied_dataops.quality.rules import (
    AnnotationInfo,
    Finding,
    QualityInput,
    TimeRange,
    VideoInfo,
    check_blur_frame_rate,
    check_dimension_mismatch,
    check_duration_exceeded,
    check_empty_annotation,
    check_file_decode_error,
    check_frame_gap_excess,
    check_low_conf_detection,
    check_nonfinite_value,
    check_sync_missing_frame,
    check_sync_missing_state,
    check_sync_stale_action,
    check_timestamp_non_monotonic,
    check_unverified_semantics,
    evaluate_quality,
)

JOINTS = [f"joint_{index}" for index in range(1, 7)]
ORIGIN_NS = 1_700_000_000_000_000_000


def make_robot_input(**overrides) -> QualityInput:
    """A clean 100-sample (10 s) robot Episode that must not produce findings."""
    count = 100
    t_ns = [index * SAMPLE_INTERVAL_NS for index in range(count)]
    state = [[0.01 * joint for joint in range(6)] for _ in range(count)]
    action = [[0.01 * joint for joint in range(6)] for _ in range(count)]
    values: dict = {
        "episode_id": "ep-robot",
        "episode_type": "robot_episode",
        "source_type": "MCAP_ROS2_PROFILE_V1",
        "data_origin": "synthetic",
        "duration_ns": 10_000_000_000,
        "t_ns": t_ns,
        "state": state,
        "action": action,
        "state_source_t_ns": list(t_ns),
        "action_source_t_ns": list(t_ns),
        "joint_names": list(JOINTS),
        "state_dim": 6,
        "action_dim": 6,
        "grid_t_ns": list(t_ns),
        "video": VideoInfo(
            expected=True,
            decode_ok=True,
            frame_count=count,
            decoded_frames=count,
            frame_t_ns=list(t_ns),
            laplacian_variances=[120.0] * count,
        ),
    }
    values.update(overrides)
    return QualityInput(**values)


def codes(findings) -> list[str]:
    return [finding.rule_code.value for finding in findings]


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def test_policy_declares_every_spec_rule_with_expected_base_severity() -> None:
    document = policy_document()
    assert document["policy_version"] == QC_POLICY_VERSION == "qc-policy/1.0"
    declared = {rule["rule_code"]: rule["severity"] for rule in document["rules"]}
    assert set(declared) == {code.value for code in RuleCode}
    assert len(declared) == 13
    assert declared == {
        "FILE_DECODE_ERROR": "ERROR",
        "DURATION_EXCEEDED": "ERROR",
        "TIMESTAMP_NON_MONOTONIC": "ERROR",
        "FRAME_GAP_EXCESS": "WARN",
        "SYNC_MISSING_FRAME": "ERROR",
        "SYNC_MISSING_STATE": "ERROR",
        "SYNC_STALE_ACTION": "ERROR",
        "DIMENSION_MISMATCH": "ERROR",
        "NONFINITE_VALUE": "ERROR",
        "UNVERIFIED_SEMANTICS": "ERROR",
        "EMPTY_ANNOTATION": "INFO",
        "LOW_CONF_DETECTION": "INFO",
        "BLUR_FRAME_RATE": "WARN",
    }
    # thresholds are part of the frozen policy document
    thresholds = document["thresholds"]
    assert thresholds["max_duration_ns"] == 600 * 1_000_000_000
    assert thresholds["frame_gap_max_ns"] == 200_000_000
    assert thresholds["image_tolerance_ns"] == 50_000_000
    assert thresholds["state_tolerance_ns"] == 50_000_000
    assert thresholds["action_zoh_tolerance_ns"] == 100_000_000
    assert thresholds["blur_frame_ratio_threshold"] == 0.30
    assert thresholds["blur_laplacian_variance_threshold"] == 40.0
    assert thresholds["low_conf_threshold"] == 0.35


def test_verdict_aggregation_error_warn_info() -> None:
    def finding(severity: Severity) -> Finding:
        return Finding(
            rule_code=RuleCode.DURATION_EXCEEDED,
            severity=severity,
            observed={},
            threshold={},
        )

    assert decide_verdict([]) == QualityStatus.PASS
    assert decide_verdict([finding(Severity.INFO)]) == QualityStatus.PASS
    assert decide_verdict([finding(Severity.INFO), finding(Severity.WARN)]) == QualityStatus.WARN
    assert decide_verdict([finding(Severity.WARN), finding(Severity.ERROR)]) == QualityStatus.FAIL
    assert decide_verdict([finding(Severity.ERROR)]) == QualityStatus.FAIL


def test_clean_robot_episode_passes_without_findings() -> None:
    evaluation = evaluate_quality(make_robot_input())
    assert evaluation.verdict == QualityStatus.PASS
    assert evaluation.findings == []
    assert evaluation.metrics["policy_version"] == QC_POLICY_VERSION
    assert evaluation.metrics["valid_sample_count"] == 100
    assert evaluation.metrics["missing_sample_count"] == 0
    assert evaluation.metrics["max_abs_sync_error_ms"] == 0.0
    assert evaluation.metrics["p95_abs_sync_error_ms"] == 0.0


# ---------------------------------------------------------------------------
# FILE_DECODE_ERROR
# ---------------------------------------------------------------------------


def test_file_decode_error_when_full_decode_fails() -> None:
    clean = make_robot_input()
    assert check_file_decode_error(clean) == []

    inp = make_robot_input(
        video=VideoInfo(expected=True, decode_ok=False, decoded_frames=0, frame_count=0, error="truncated")
    )
    findings = check_file_decode_error(inp)
    assert codes(findings) == ["FILE_DECODE_ERROR"]
    assert findings[0].severity == Severity.ERROR
    assert findings[0].observed["decode_ok"] is False
    assert findings[0].evidence["decode_error"] == "truncated"


def test_file_decode_error_when_zero_frames_decoded() -> None:
    inp = make_robot_input(video=VideoInfo(expected=True, decode_ok=True, decoded_frames=0, frame_count=120))
    findings = check_file_decode_error(inp)
    assert codes(findings) == ["FILE_DECODE_ERROR"]


def test_file_decode_rule_not_applicable_without_video_expectation() -> None:
    inp = make_robot_input(video=VideoInfo(expected=False))
    assert check_file_decode_error(inp) == []


# ---------------------------------------------------------------------------
# DURATION_EXCEEDED
# ---------------------------------------------------------------------------


def test_duration_exactly_600_seconds_passes() -> None:
    inp = make_robot_input(duration_ns=THRESHOLDS.max_duration_ns)
    assert check_duration_exceeded(inp) == []


def test_duration_one_nanosecond_over_600_seconds_fails() -> None:
    inp = make_robot_input(duration_ns=THRESHOLDS.max_duration_ns + 1)
    findings = check_duration_exceeded(inp)
    assert codes(findings) == ["DURATION_EXCEEDED"]
    assert findings[0].severity == Severity.ERROR
    assert findings[0].observed["duration_s"] == 600.000000001
    assert findings[0].threshold["max_duration_ns"] == 600 * 1_000_000_000


# ---------------------------------------------------------------------------
# TIMESTAMP_NON_MONOTONIC
# ---------------------------------------------------------------------------


def test_timestamp_duplicate_is_error() -> None:
    stream = [0, 100_000_000, 100_000_000, 200_000_000]
    inp = make_robot_input(stream_times={"state": stream}, origin_ns=0)
    findings = check_timestamp_non_monotonic(inp)
    assert codes(findings) == ["TIMESTAMP_NON_MONOTONIC"]
    assert findings[0].severity == Severity.ERROR
    assert findings[0].observed["streams"]["state"]["violation_count"] == 1
    assert findings[0].affected_time_ranges[0] == TimeRange(100_000_000, 200_000_000)


def test_timestamp_strictly_increasing_passes() -> None:
    inp = make_robot_input(stream_times={"state": [0, 10_000_000, 20_000_000]}, origin_ns=0)
    assert check_timestamp_non_monotonic(inp) == []


def test_timestamp_rule_covers_every_source_stream() -> None:
    inp = make_robot_input(
        stream_times={"camera": [0, 33_333_333, 33_333_333], "state": [0, 10_000_000, 20_000_000]},
        origin_ns=0,
    )
    findings = check_timestamp_non_monotonic(inp)
    assert findings[0].observed["streams"].keys() == {"camera"}
    assert findings[0].observed["violation_count"] == 1


# ---------------------------------------------------------------------------
# FRAME_GAP_EXCESS
# ---------------------------------------------------------------------------


def test_frame_gap_exactly_200ms_passes() -> None:
    inp = make_robot_input(stream_times={"camera": [0, 200_000_000, 400_000_000]}, origin_ns=0)
    assert check_frame_gap_excess(inp) == []


def test_frame_gap_200_001ms_is_warn() -> None:
    camera = [0, 200_000_001]
    inp = make_robot_input(stream_times={"camera": camera}, origin_ns=0)
    findings = check_frame_gap_excess(inp)
    assert codes(findings) == ["FRAME_GAP_EXCESS"]
    assert findings[0].severity == Severity.WARN
    assert findings[0].observed["max_gap_ns"] == 200_000_001
    assert findings[0].observed["gap_windows"][0]["gap_ms"] == 200.000001


def test_frame_gap_with_missing_grid_image_escalates_to_error() -> None:
    inp = make_robot_input(missing_frame_indexes=[3], stream_times={"camera": [0, 300_000_000]}, origin_ns=0)
    findings = check_frame_gap_excess(inp)
    assert findings[0].severity == Severity.ERROR
    assert findings[0].observed["missing_grid_frame_count"] == 1
    assert findings[0].evidence["escalated_to_error"] is True


# ---------------------------------------------------------------------------
# SYNC_MISSING_* (tolerance boundaries are pinned in test_mcap_alignment)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "field", "expected_code", "expected_range"),
    [
        (check_sync_missing_frame, "missing_frame_indexes", "SYNC_MISSING_FRAME", TimeRange(500_000_000, 600_000_000)),
        (check_sync_missing_state, "missing_state_indexes", "SYNC_MISSING_STATE", TimeRange(500_000_000, 600_000_000)),
        (check_sync_stale_action, "stale_action_indexes", "SYNC_STALE_ACTION", TimeRange(500_000_000, 600_000_000)),
    ],
)
def test_sync_rules_emit_error_with_affected_range(rule, field, expected_code, expected_range) -> None:
    inp = make_robot_input(**{field: [5]})
    findings = rule(inp)
    assert len(findings) == 1
    assert findings[0].severity == Severity.ERROR
    assert findings[0].rule_code.value == expected_code
    assert findings[0].affected_time_ranges[0] == expected_range
    assert findings[0].observed["grid_point_count"] == 100
    assert findings[0].evidence["no_zero_fill"] is True


def test_sync_rules_pass_when_no_grid_point_is_missing() -> None:
    inp = make_robot_input()
    assert check_sync_missing_frame(inp) == []
    assert check_sync_missing_state(inp) == []
    assert check_sync_stale_action(inp) == []


# ---------------------------------------------------------------------------
# DIMENSION_MISMATCH
# ---------------------------------------------------------------------------


def test_dimension_mismatch_for_5d_action() -> None:
    inp = make_robot_input(
        state=[[0.0] * 6 for _ in range(100)],
        action=[[0.0] * 5 for _ in range(100)],
        action_dim=5,
    )
    findings = check_dimension_mismatch(inp)
    assert codes(findings) == ["DIMENSION_MISMATCH"]
    assert findings[0].severity == Severity.ERROR
    assert findings[0].observed["observed_action_dim"] == 5
    assert findings[0].observed["problems"] == {"action_dim": 5}
    assert findings[0].evidence["no_truncation_or_padding"] is True


def test_dimension_mismatch_for_duplicate_or_missing_joint_names() -> None:
    duplicate = make_robot_input(joint_names=["joint_1", "joint_1", "joint_3", "joint_4", "joint_5", "joint_6"])
    findings = check_dimension_mismatch(duplicate)
    assert codes(findings) == ["DIMENSION_MISMATCH"]
    assert "joint_names" in findings[0].observed["problems"]

    missing = make_robot_input(joint_names=["joint_1", "joint_2", "joint_3"])
    assert codes(check_dimension_mismatch(missing)) == ["DIMENSION_MISMATCH"]


def test_dimension_mismatch_unknown_evidence_does_not_invent_a_finding() -> None:
    inp = make_robot_input(joint_names=None, state=[], action=[], state_dim=None, action_dim=None)
    assert check_dimension_mismatch(inp) == []


def test_dimension_mismatch_not_applicable_to_ego_episodes() -> None:
    inp = make_robot_input(episode_type="ego_video_episode", joint_names=[], state=[], action=[], state_dim=0, action_dim=0)
    assert check_dimension_mismatch(inp) == []


# ---------------------------------------------------------------------------
# NONFINITE_VALUE
# ---------------------------------------------------------------------------


def test_nonfinite_nan_in_state_is_error() -> None:
    state = [[0.0] * 6 for _ in range(100)]
    state[3][2] = math.nan
    inp = make_robot_input(state=state)
    findings = check_nonfinite_value(inp)
    assert codes(findings) == ["NONFINITE_VALUE"]
    assert findings[0].severity == Severity.ERROR
    assert findings[0].observed["state"]["row_indexes"] == [3]
    assert findings[0].observed["state"]["kinds"] == ["NaN"]
    assert findings[0].affected_time_ranges[0] == TimeRange(300_000_000, 400_000_000)


def test_nonfinite_inf_in_action_is_error() -> None:
    action = [[0.0] * 6 for _ in range(100)]
    action[7][0] = math.inf
    action[8][1] = -math.inf
    inp = make_robot_input(action=action)
    findings = check_nonfinite_value(inp)
    assert codes(findings) == ["NONFINITE_VALUE"]
    assert findings[0].observed["action"]["row_indexes"] == [7, 8]
    assert findings[0].observed["action"]["kinds"] == ["+Inf", "-Inf"]


def test_nonfinite_clean_data_passes() -> None:
    assert check_nonfinite_value(make_robot_input()) == []


# ---------------------------------------------------------------------------
# UNVERIFIED_SEMANTICS
# ---------------------------------------------------------------------------


def test_unverified_semantics_fails_lerobot_without_evidence() -> None:
    inp = make_robot_input(source_type="LEROBOT_V3", semantics_verified=None, robot_profile="arm6_joint_position_v1")
    findings = check_unverified_semantics(inp)
    assert codes(findings) == ["UNVERIFIED_SEMANTICS"]
    assert findings[0].severity == Severity.ERROR
    assert findings[0].observed["semantics_verified"] is None
    assert findings[0].evidence["publishable"] is False


def test_unverified_semantics_explicit_false_fails() -> None:
    inp = make_robot_input(source_type="LEROBOT_V3", semantics_verified=False)
    assert codes(check_unverified_semantics(inp)) == ["UNVERIFIED_SEMANTICS"]


def test_unverified_semantics_verified_passes() -> None:
    inp = make_robot_input(source_type="LEROBOT_V3", semantics_verified=True)
    assert check_unverified_semantics(inp) == []


def test_unverified_semantics_not_applicable_to_mcap() -> None:
    inp = make_robot_input(source_type="MCAP_ROS2_PROFILE_V1", semantics_verified=None)
    assert check_unverified_semantics(inp) == []


# ---------------------------------------------------------------------------
# EMPTY_ANNOTATION / LOW_CONF_DETECTION (INFO)
# ---------------------------------------------------------------------------


def test_empty_annotation_is_info_only() -> None:
    inp = make_robot_input(annotations=AnnotationInfo(frame_count=100, total_detections=0, model_id="fake"))
    findings = check_empty_annotation(inp)
    assert codes(findings) == ["EMPTY_ANNOTATION"]
    assert findings[0].severity == Severity.INFO
    assert evaluate_quality(inp).verdict == QualityStatus.PASS


def test_low_confidence_boundary() -> None:
    exactly_threshold = make_robot_input(
        annotations=AnnotationInfo(frame_count=10, total_detections=2, confidences=[0.35, 0.5])
    )
    assert check_low_conf_detection(exactly_threshold) == []

    below_threshold = make_robot_input(
        annotations=AnnotationInfo(
            frame_count=10,
            total_detections=2,
            confidences=[0.349999, 0.5],
            detection_frame_indexes=[3, 4],
        )
    )
    findings = check_low_conf_detection(below_threshold)
    assert codes(findings) == ["LOW_CONF_DETECTION"]
    assert findings[0].severity == Severity.INFO
    assert findings[0].observed["low_conf_count"] == 1
    assert findings[0].observed["min_confidence"] == 0.349999
    assert findings[0].observed["excluded_from_candidates"] is True
    assert evaluate_quality(below_threshold).verdict == QualityStatus.PASS


# ---------------------------------------------------------------------------
# BLUR_FRAME_RATE
# ---------------------------------------------------------------------------


def _variances(blurry: int, total: int) -> list[float]:
    return [39.999] * blurry + [40.0] * (total - blurry)


def test_blur_ratio_exactly_30_percent_passes() -> None:
    inp = make_robot_input(
        video=VideoInfo(
            expected=True,
            decode_ok=True,
            decoded_frames=100,
            frame_count=100,
            laplacian_variances=_variances(30, 100),
            frame_t_ns=[index * SAMPLE_INTERVAL_NS for index in range(100)],
        )
    )
    assert check_blur_frame_rate(inp) == []


def test_blur_ratio_31_percent_is_warn() -> None:
    inp = make_robot_input(
        video=VideoInfo(
            expected=True,
            decode_ok=True,
            decoded_frames=100,
            frame_count=100,
            laplacian_variances=_variances(31, 100),
            frame_t_ns=[index * SAMPLE_INTERVAL_NS for index in range(100)],
        )
    )
    findings = check_blur_frame_rate(inp)
    assert codes(findings) == ["BLUR_FRAME_RATE"]
    assert findings[0].severity == Severity.WARN
    assert findings[0].observed["blurry_frame_ratio"] == 0.31
    assert findings[0].affected_time_ranges[0] == TimeRange(0, SAMPLE_INTERVAL_NS)
    assert evaluate_quality(inp).verdict == QualityStatus.WARN


def test_blur_ratio_30_001_percent_is_warn() -> None:
    inp = make_robot_input(
        video=VideoInfo(
            expected=True,
            decode_ok=True,
            decoded_frames=1000,
            frame_count=1000,
            laplacian_variances=_variances(301, 1000),
        )
    )
    findings = check_blur_frame_rate(inp)
    assert findings[0].observed["blurry_frame_ratio"] == 0.301

    exactly = make_robot_input(
        video=VideoInfo(
            expected=True,
            decode_ok=True,
            decoded_frames=1000,
            frame_count=1000,
            laplacian_variances=_variances(300, 1000),
        )
    )
    assert check_blur_frame_rate(exactly) == []


# ---------------------------------------------------------------------------
# vacuous pass: an episode with no usable samples must never PASS
# ---------------------------------------------------------------------------


def _sampleless_robot_input(**overrides) -> QualityInput:
    """A robot episode whose normalisation produced zero samples."""
    values: dict = {
        "episode_id": "ep-empty",
        "episode_type": "robot_episode",
        "source_type": "LEROBOT_V3",
        "semantics_verified": True,
        "normalization_valid": False,
        "t_ns": [],
        "state": [],
        "action": [],
        "state_dim": 6,
        "action_dim": 6,
    }
    values.update(overrides)
    return QualityInput(**values)


def test_structural_diagnostic_is_carried_as_a_finding() -> None:
    """A 0-sample episode inherits the importer's verdict instead of silently passing."""
    inp = _sampleless_robot_input(
        structural_diagnostics=[
            {
                "code": RuleCode.TIMESTAMP_NON_MONOTONIC.value,
                "severity": Severity.ERROR.value,
                "message": "LeRobot episode timestamps are not strictly increasing",
                "details": {"violation_count": 1, "violation_indexes": [50]},
                "structural": True,
            }
        ]
    )
    evaluation = evaluate_quality(inp)
    assert evaluation.verdict == QualityStatus.FAIL
    assert codes(evaluation.findings) == ["TIMESTAMP_NON_MONOTONIC"]
    finding = evaluation.findings[0]
    assert finding.observed["carried_from"] == "normalization"
    assert finding.observed["details"]["violation_indexes"] == [50]
    assert finding.evidence["structural"] is True
    # the code must be reported as *run*, not as unavailable
    assert RuleCode.TIMESTAMP_NON_MONOTONIC.value in evaluation.checks_run
    assert RuleCode.TIMESTAMP_NON_MONOTONIC.value not in evaluation.checks_unavailable


def test_carried_finding_does_not_duplicate_a_fired_rule() -> None:
    """A rule that already fired is not repeated by the carried diagnostic."""
    inp = make_robot_input(
        action=[[0.0] * 5 for _ in range(100)],
        action_dim=5,
        joint_names=[],
        structural_diagnostics=[
            {"code": RuleCode.DIMENSION_MISMATCH.value, "severity": Severity.ERROR.value}
        ],
    )
    assert codes(evaluate_quality(inp).findings) == ["DIMENSION_MISMATCH"]


def test_unknown_or_non_rule_diagnostic_codes_are_ignored() -> None:
    inp = _sampleless_robot_input(
        structural_diagnostics=[
            {"code": "NOT_A_RULE_CODE", "severity": "ERROR"},
            {"code": RuleCode.DIMENSION_MISMATCH.value, "severity": "INFO"},
        ]
    )
    assert codes(evaluate_quality(inp).findings) == []


def test_episode_without_samples_never_passes() -> None:
    """Fail-closed: no samples is not a passing verdict, even with no diagnostic."""
    evaluation = evaluate_quality(_sampleless_robot_input())
    assert evaluation.verdict == QualityStatus.FAIL
    assert evaluation.metrics["normalization_valid"] is False


# ---------------------------------------------------------------------------
# metrics / report
# ---------------------------------------------------------------------------


def test_sync_metrics_count_missing_samples_and_errors() -> None:
    count = 100
    inp = make_robot_input(
        missing_state_indexes=list(range(10, 20)),
        frame_errors_ns=[10_000_000] * count,
        state_errors_ns=[25_000_000] * count,
        action_ages_ns=[100_000_000] * count,
    )
    evaluation = evaluate_quality(inp)
    metrics = evaluation.metrics
    assert metrics["valid_sample_count"] == 90
    assert metrics["missing_sample_count"] == 10
    assert metrics["max_abs_sync_error_ms"] == 100.0
    assert metrics["p95_abs_sync_error_ms"] == 100.0
    assert evaluation.verdict == QualityStatus.FAIL
    assert codes(evaluation.findings) == ["SYNC_MISSING_STATE"]


def test_sync_metrics_are_computed_from_per_point_errors_when_absent() -> None:
    inp = make_robot_input(
        frame_errors_ns=[49_000_000, 50_000_000],
        state_errors_ns=[0, 25_000_000],
        action_ages_ns=[0, 100_000_000],
    )
    metrics = evaluate_quality(inp).metrics
    assert metrics["max_abs_sync_error_ms"] == 100.0
    assert metrics["p95_abs_sync_error_ms"] == 100.0


def test_sync_error_50ms_boundary_values() -> None:
    """``<= 50 ms`` passes; the same construct 1 ns over fails (SPEC 4.2)."""
    from embodied_dataops.canonical.timebase import (
        ACTION_ZOH_TOLERANCE_NS,
        IMAGE_TOLERANCE_NS,
        STATE_TOLERANCE_NS,
        interpolate_state,
        nearest_match,
        zoh_action,
    )

    assert IMAGE_TOLERANCE_NS == STATE_TOLERANCE_NS == 50_000_000
    assert ACTION_ZOH_TOLERANCE_NS == 100_000_000
    times = [0, 200_000_000]
    assert nearest_match(times, 50_000_000, IMAGE_TOLERANCE_NS) is not None
    assert nearest_match(times, 50_000_001, IMAGE_TOLERANCE_NS) is None
    state_times = [0, 100_000_000]
    values = [(0.0,), (1.0,)]
    assert interpolate_state(state_times, values, 50_000_000, STATE_TOLERANCE_NS) is not None
    assert interpolate_state(state_times, values, 50_000_001, STATE_TOLERANCE_NS) is None
    assert zoh_action([0], [(0.0,)], 100_000_000, ACTION_ZOH_TOLERANCE_NS) is not None
    assert zoh_action([0], [(0.0,)], 100_000_001, ACTION_ZOH_TOLERANCE_NS) is None


def test_findings_are_json_serializable_with_required_fields() -> None:
    inp = make_robot_input(
        missing_state_indexes=[5],
        stream_times={"state": [0, 0, 1]},
        annotations=AnnotationInfo(frame_count=10, total_detections=0, confidences=[0.1]),
    )
    evaluation = evaluate_quality(inp)
    payload = [finding.as_dict() for finding in evaluation.findings]
    json.dumps(payload)  # must not raise
    for finding in payload:
        assert set(finding) == {
            "rule_code",
            "severity",
            "observed",
            "threshold",
            "affected_time_ranges",
            "evidence",
        }
        for window in finding["affected_time_ranges"]:
            assert set(window) == {"start_ns", "end_ns"}
    assert evaluation.verdict == QualityStatus.FAIL


def test_build_quality_report_contains_verdict_findings_metrics() -> None:
    inp = make_robot_input(missing_state_indexes=[5])
    evaluation = evaluate_quality(inp)
    report = build_quality_report(
        episode_id=inp.episode_id,
        run_id="run-1",
        evaluation=evaluation,
        input=inp,
    )
    assert report["schema_version"] == "quality-report/1.0"
    assert report["status"] == "FAIL"
    assert report["policy_version"] == "qc-policy/1.0"
    assert report["episode_id"] == "ep-robot"
    assert report["run_id"] == "run-1"
    assert [finding["rule_code"] for finding in report["findings"]] == ["SYNC_MISSING_STATE"]
    assert report["metrics"]["valid_sample_count"] == 99
    assert report["metrics"]["missing_sample_count"] == 1
    assert report["checks"]["run"]
    assert report["source"]["episode_type"] == "robot_episode"
    json.dumps(report)  # canonical JSON payload must be serializable


def test_rule_registry_thresholds_are_frozen_policy_constants() -> None:
    assert RULES[RuleCode.SYNC_MISSING_FRAME].threshold["image_tolerance_ns"] == 50_000_000
    assert RULES[RuleCode.SYNC_MISSING_STATE].threshold["state_tolerance_ns"] == 50_000_000
    assert RULES[RuleCode.SYNC_STALE_ACTION].threshold["action_zoh_tolerance_ns"] == 100_000_000
    assert RULES[RuleCode.BLUR_FRAME_RATE].threshold["laplacian_variance"] == 40.0
    assert RULES[RuleCode.DIMENSION_MISMATCH].threshold["joint_names"] == JOINTS
