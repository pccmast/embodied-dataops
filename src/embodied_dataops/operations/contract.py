"""DO-A02 C1 contract constants and the ``GET /capabilities`` payload.

The values are fixed by the cross-project contract; they are defined once here so
the API route, the OpenAPI ``x-contract-version`` extension and the contract tests
cannot drift apart.
"""

from __future__ import annotations

from typing import Any

#: Cross-project contract version (DO-A02 C1).
CONTRACT_VERSION = "do-agent-contract/2.0.0"

#: Canonical episode schema versions this DataOps build can serve.
CANONICAL_VERSIONS: list[str] = ["1.0", "2.0"]

#: Robot profiles accepted by the V2 import path (DO-A02 C1, SPEC 2.1).
ROBOT_PROFILES: list[str] = ["arm6_joint_position_v1", "arm7_joint_position_v1"]

#: Quality policy versions this DataOps build can serve.
QC_POLICY_VERSIONS: list[str] = ["qc-policy/1.0", "qc-policy/2.0"]

#: Single local tenant (V1 SPEC 4.2); kept explicit for future multi-tenancy.
TENANT_ID = "local"

#: Control-write actions (fixed strings from DO-A02 C3; also DB CHECK values).
ACTION_REPROCESS_EPISODE = "REPROCESS_EPISODE"
ACTION_RETRY_FAILED_TASK = "RETRY_FAILED_TASK"
ACTION_ANALYZE_EVALUATION_GAPS = "ANALYZE_EVALUATION_GAPS"
CONTROLLED_ACTIONS = (
    ACTION_REPROCESS_EPISODE,
    ACTION_RETRY_FAILED_TASK,
    ACTION_ANALYZE_EVALUATION_GAPS,
)

#: ETag kinds per controlled target (strong ETags, no ``W/``).
ETAG_EPISODE = "episode-rev"
ETAG_TASK = "task-rev"
ETAG_EVALUATION = "evaluation-rev"
ETAG_KINDS = (ETAG_EPISODE, ETAG_TASK, ETAG_EVALUATION)

#: Pipeline version written by V2 controlled operations (SPEC 0.4).
V2_PIPELINE_VERSION = "2.0.0"

#: In-flight same-key retry waits at most this long before returning 503.
COMMIT_WAIT_MS = 3000


def capabilities_payload() -> dict[str, Any]:
    """Exact ``GET /capabilities`` body required by DO-A02 C1."""
    return {
        "contract_version": CONTRACT_VERSION,
        "canonical_versions": list(CANONICAL_VERSIONS),
        "robot_profiles": list(ROBOT_PROFILES),
        "qc_policy_versions": list(QC_POLICY_VERSIONS),
        "conditional_writes": True,
        "operation_receipts": True,
        "diagnostic_queries": True,
    }
