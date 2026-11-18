"""Quality policy, rules and reporting (SPEC 7.1 ``qc-policy/1.0`` + V2 ``qc-policy/2.0``).

V1 (frozen, read compatible)
----------------------------
* :mod:`embodied_dataops.quality.policy` -- declarative rule codes, severities,
  exact thresholds and verdict aggregation; thresholds are frozen in a policy
  version and reports are immutable;
* :mod:`embodied_dataops.quality.rules` -- pure rule functions over a
  :class:`~embodied_dataops.quality.rules.QualityInput`, unit-testable without a
  database;
* :mod:`embodied_dataops.quality.report` -- ``quality.json`` assembly and
  persistence to the object store + ``quality_reports`` / ``quality_findings``.

V2 (``qc-policy/2.0``, SPEC_ADDENDUM_V2 §3)
------------------------------------------
* :mod:`embodied_dataops.quality.v2_policy` -- declarative V2 policy: the eight
  robot-semantic rules with exact thresholds plus a frozen configuration SHA-256;
* :mod:`embodied_dataops.quality.v2_rules` -- pure robot-semantic rules over a
  :class:`~embodied_dataops.quality.v2_rules.SemanticQcInput`, including the
  carried V1 file/frame/dimension/sync/non-finite rules evaluated on the native
  and aligned canonical views;
* :mod:`embodied_dataops.quality.v2_report` -- ``quality-report/2.0`` assembly,
  duplicate/time-range merging and the 10,000-row ``QUALITY_REPORT_TOO_LARGE``
  cap; V1 reports are never recomputed in place;
* :mod:`embodied_dataops.quality.canonical_v2` -- tolerant canonical-episode/2.0
  reader used by the ``QUALITY_CHECK`` v2 handler;
* :mod:`embodied_dataops.quality.http` -- ``GET /episodes/{id}/quality``
  ``policy_version`` selection and response contract.
"""

from __future__ import annotations

from .policy import (
    QC_POLICY_VERSION,
    RULES,
    THRESHOLDS,
    QcThresholds,
    RuleCode,
    RuleDefinition,
    decide_verdict,
    policy_document,
    rule_definition,
)
from .report import (
    QUALITY_REPORT_SCHEMA_VERSION,
    build_quality_report,
    persist_quality_report,
    quality_report_key,
    report_json_bytes,
)
from .rules import (
    AnnotationInfo,
    Finding,
    QualityEvaluation,
    QualityInput,
    TimeRange,
    VideoInfo,
    build_metrics,
    evaluate_quality,
)
from .v2_policy import (
    CARRIED_V1_RULE_CODES,
    POLICY_CONFIG_SHA256,
    QC_POLICY_V2_VERSION,
    QC_POLICY_VERSIONS,
    QUALITY_REPORT_V2_SCHEMA_VERSION,
    SEMANTIC_RULES,
    THRESHOLDS_V2,
    QcV2Thresholds,
    SemanticRuleCode,
    policy_config,
    policy_config_sha256,
    policy_document_v2,
    rule_definition_v2,
)
from .v2_report import (
    MAX_FINDING_ROWS,
    MergeStats,
    PreparedReport,
    QualityReportTooLarge,
    build_raw_diagnostics_payload,
    merge_semantic_findings,
    persist_quality_report_too_large,
    persist_quality_report_v2,
    prepare_quality_report_v2,
    quality_diagnostics_v2_key,
    quality_report_v2_key,
)
from .v2_rules import (
    AlignedView,
    ClockDomainInfo,
    NativeStream,
    PointCloudSummary,
    ProfileBinding,
    SemanticEvaluation,
    SemanticFinding,
    SemanticQcInput,
    build_semantic_metrics,
    evaluate_carried_rules,
    evaluate_semantic_quality,
)

__all__ = [
    "AlignedView",
    "AnnotationInfo",
    "CARRIED_V1_RULE_CODES",
    "ClockDomainInfo",
    "Finding",
    "MAX_FINDING_ROWS",
    "MergeStats",
    "NativeStream",
    "POLICY_CONFIG_SHA256",
    "PointCloudSummary",
    "PreparedReport",
    "ProfileBinding",
    "QC_POLICY_V2_VERSION",
    "QC_POLICY_VERSION",
    "QC_POLICY_VERSIONS",
    "QUALITY_REPORT_SCHEMA_VERSION",
    "QUALITY_REPORT_V2_SCHEMA_VERSION",
    "QcThresholds",
    "QcV2Thresholds",
    "QualityEvaluation",
    "QualityInput",
    "QualityReportTooLarge",
    "RULES",
    "RuleCode",
    "RuleDefinition",
    "SEMANTIC_RULES",
    "SemanticEvaluation",
    "SemanticFinding",
    "SemanticQcInput",
    "SemanticRuleCode",
    "THRESHOLDS",
    "THRESHOLDS_V2",
    "TimeRange",
    "VideoInfo",
    "build_metrics",
    "build_quality_report",
    "build_raw_diagnostics_payload",
    "build_semantic_metrics",
    "decide_verdict",
    "evaluate_carried_rules",
    "evaluate_quality",
    "evaluate_semantic_quality",
    "merge_semantic_findings",
    "persist_quality_report",
    "persist_quality_report_too_large",
    "persist_quality_report_v2",
    "policy_config",
    "policy_config_sha256",
    "policy_document",
    "policy_document_v2",
    "prepare_quality_report_v2",
    "quality_diagnostics_v2_key",
    "quality_report_key",
    "quality_report_v2_key",
    "report_json_bytes",
    "rule_definition",
    "rule_definition_v2",
]
