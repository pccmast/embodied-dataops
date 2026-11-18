"""Acceptance runner: executes the automated AT suite and writes evidence.

Evidence written under ``artifacts/``:

* ``artifacts/test-results/junit.xml``       - machine-readable pytest results
* ``artifacts/test-results/at-matrix.json``  - AT-01..AT-25 status derived from
                                               matching test node ids
* ``artifacts/test-results/summary.md``      - human-readable summary

A test proves an AT only if it runs and passes; skipped/failed tests yield
PARTIAL/FAILED, and ATs with no executed test stay NOT_RUN.  Nothing is ever
inferred from the existence of source files.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path("artifacts/test-results")
JUNIT_PATH = RESULTS_DIR / "junit.xml"
MATRIX_PATH = RESULTS_DIR / "at-matrix.json"
SUMMARY_PATH = RESULTS_DIR / "summary.md"

#: AT id -> (description, regex matched against "<classname>::<name>")
AT_MATRIX: dict[str, tuple[str, str]] = {
    "AT-01": ("make up && make seed; ten fixtures; /health/ready", r"at_01|at01|make_up|fixtures_available"),
    "AT-02": ("LeRobot v3 import F01: 2 episodes, 2x100 samples, 6D", r"at_02|at02|import_lerobot|lerobot_import"),
    "AT-03": ("Ego MP4 import F04: ego type, 100 samples, no state/action", r"at_03|at03|import_ego|ego_import"),
    "AT-04": ("MCAP F05: three topics, 10Hz, action ZOH never future", r"at_04|at04|mcap_good|mcap_import"),
    "AT-05": ("MCAP F06: SYNC_MISSING_STATE, QC FAIL, not publishable", r"at_05|at05|mcap_gap|sync_missing"),
    "AT-06": ("F02/F03: ERROR findings with evidence, no silent repair", r"at_06|at06|bad_timestamps|bad_dims|quality_rules"),
    "AT-07": ("corrupt MP4 / malicious TAR: explicit codes, no escape", r"at_07|at07|corrupt|bad_tar|unsafe"),
    "AT-08": ("idempotent complete 3x returns one Import Job", r"at_08|at08|idempotent|idempotency"),
    "AT-09": ("two workers claim one task: one lease, one output", r"at_09|at09|claim_exclusiv|lease_exclusiv|task_protocol"),
    "AT-10": ("worker kill -9: lease reclaimed, no permanent RUNNING", r"at_10|at10|worker_crash|lease_reclaim"),
    "AT-11": ("stale lease commit rejected by CAS", r"at_11|at11|fencing|stale_token|lease_fencing"),
    "AT-12": ("outbox survives Redis XADD failure", r"at_12|at12|outbox_recovery|outbox_publish"),
    "AT-13": ("duplicate XADD does not duplicate valid work", r"at_13|at13|duplicate|replay"),
    "AT-14": ("retryable error: exactly 3 claims, 5s/20s backoff", r"at_14|at14|retry_backoff|backoff"),
    "AT-15": ("QC FAIL episode cannot be published (409)", r"at_15|at15|qc_fail|eligible"),
    "AT-16": ("unreviewed annotation cannot be published (409)", r"at_16|at16|annotation_not_approved|not_approved"),
    "AT-17": ("official Writer export re-read by official Reader", r"at_17|at17|lerobot_roundtrip|round_trip"),
    "AT-18": ("PUBLISHED version is immutable, manifest hash stable", r"at_18|at18|immutab"),
    "AT-19": ("F09 analysis idempotent, transparent 36, one order", r"at_19|at19|gap|feedback"),
    "AT-20": ("RBAC 403 for viewer/operator, audit complete", r"at_20|at20|rbac|forbidden|audit"),
    "AT-21": ("real YOLO11n inference, weights hash + real JSON", r"at_21|at21|yolo"),
    "AT-22": ("100-episode synthetic load, no stuck RUNNING, metrics", r"at_22|at22|benchmark|load_100|scale"),
    "AT-23": ("Streamlit four pages driven by the API", r"at_23|at23|streamlit|ui_pages"),
    "AT-24": ("real public LeRobot v3 dataset import provenance", r"at_24|at24|public_dataset|public_example"),
    "AT-25": ("GC never deletes referenced assets; 24h orphan window", r"at_25|at25|garbage|gc_orphan"),
}


def _parse_junit(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    tree = ET.parse(path)
    cases = []
    for case in tree.iter("testcase"):
        status = "passed"
        detail = ""
        for child in case:
            if child.tag in ("failure", "error"):
                status = "failed"
                detail = (child.get("message") or "")[:300]
            elif child.tag == "skipped":
                status = "skipped"
                detail = (child.get("message") or "")[:300]
        cases.append(
            {
                "id": f"{case.get('classname', '')}::{case.get('name', '')}",
                "test": case.get("name", ""),
                "classname": case.get("classname", ""),
                "time": float(case.get("time") or 0.0),
                "status": status,
                "detail": detail,
            }
        )
    return cases


def evaluate_matrix(cases: list[dict]) -> dict[str, dict]:
    matrix: dict[str, dict] = {}
    for at_id, (description, pattern) in AT_MATRIX.items():
        regex = re.compile(pattern, re.IGNORECASE)
        matched = [
            case
            for case in cases
            if regex.search(f"{case['classname']}::{case['test']}")
        ]
        passed = [case for case in matched if case["status"] == "passed"]
        failed = [case for case in matched if case["status"] == "failed"]
        skipped = [case for case in matched if case["status"] == "skipped"]
        if passed and not failed:
            status = "DONE"
        elif failed:
            status = "FAILED"
        elif skipped:
            status = "PARTIAL"
        elif matched:
            status = "PARTIAL"
        else:
            status = "NOT_RUN"
        matrix[at_id] = {
            "description": description,
            "status": status,
            "passed": len(passed),
            "failed": len(failed),
            "skipped": len(skipped),
            "tests": [case["id"] for case in matched][:10],
        }
    return matrix


def run_acceptance(*, include_performance: bool = False, extra_args: list[str] | None = None) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    marker = "not performance" if not include_performance else None
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={JUNIT_PATH}"]
    if marker:
        command += ["-m", marker]
    if extra_args:
        command += extra_args
    env = dict(os.environ)
    print("[acceptance] " + " ".join(command))
    completed = subprocess.run(command, env=env)
    cases = _parse_junit(JUNIT_PATH)
    matrix = evaluate_matrix(cases)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": command,
        "pytest_exit_code": completed.returncode,
        "total_tests": len(cases),
        "passed": sum(1 for case in cases if case["status"] == "passed"),
        "failed": sum(1 for case in cases if case["status"] == "failed"),
        "skipped": sum(1 for case in cases if case["status"] == "skipped"),
        "matrix": matrix,
    }
    MATRIX_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Acceptance run summary",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- command: `{' '.join(command)}`",
        f"- tests: {payload['total_tests']} total / {payload['passed']} passed / "
        f"{payload['failed']} failed / {payload['skipped']} skipped",
        "",
        "| AT | status | description | tests |",
        "|---|---|---|---|",
    ]
    for at_id, row in matrix.items():
        lines.append(
            f"| {at_id} | {row['status']} | {row['description']} | {row['passed']}/{len(row['tests'])} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[acceptance] junit -> {JUNIT_PATH}")
    print(f"[acceptance] matrix -> {MATRIX_PATH}")
    print(f"[acceptance] summary -> {SUMMARY_PATH}")
    return completed.returncode
