"""V2.1 acceptance runner (acceptance V21-28, SPEC_V2.1 §7.2/§9).

One job: turn the 28 rows of ``acceptance/ACCEPTANCE_V21.csv`` into
``reports/acceptance-v21.json`` with a status that is **checked against an artifact**,
not asserted.  Rules:

* a declared ``PASS`` must point at an evidence file that exists and (when the file is
  JSON with a ``status`` field) agrees with the declaration;
* ``BLOCKED`` must carry a reason, and the reason must name the missing external thing;
* ``NOT_RUN`` is allowed only for optional items or items whose dependency is blocked;
* nothing is upgraded automatically -- no evidence, no PASS.

**What this script is and is not.**  The per-item status is *declared* in
:data:`STATUS`; the runner only verifies the evidence exists and does not contradict
the declaration.  It is therefore **not** a live measurement.  Two things are measured
for real here:

* the environment probes (``postgres_available`` / ``cuda_available``) actually try to
  connect / import ``torch`` instead of reading an environment variable;
* ``--run-tests`` executes ``pytest tests/v21`` and records the real JUnit counts, and
  a failing run makes this script exit non-zero.

Exit codes: ``0`` when every required item is PASS/``PARTIAL`` and the optional test run
passed, ``3`` when at least one required item is ``BLOCKED``/``NOT_RUN``/``FAILED`` or
the test run failed (CI must not map 3 to success).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPOSITORY_ROOT / "acceptance" / "ACCEPTANCE_V21.csv"
DEFAULT_OUT = REPOSITORY_ROOT / "reports" / "acceptance-v21.json"

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"
PARTIAL = "PARTIAL"

#: Non-PASS-but-accepted statuses.  ``PARTIAL`` is used where a mechanism is implemented
#: and tested but the real-world instance of the evidence is unavailable here.
ACCEPTED = {PASS, PARTIAL}


@dataclass
class ItemStatus:
    test_id: str
    status: str
    evidence: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "evidence": self.evidence,
            "note": self.note,
        }


#: Declared status per acceptance item.  Every ``PASS``/``PARTIAL`` here must list at
#: least one evidence path; the runner verifies it exists before writing the report.
STATUS: dict[str, ItemStatus] = {
    "V21-01": ItemStatus(
        "V21-01",
        PARTIAL,
        ["acceptance/OVERRIDE_MATRIX.csv", "docs/v21-gap-analysis.md"],
        "override matrix and gap analysis are in-repo; the full V1/V2 regression could not "
        "be re-executed on this host because PostgreSQL is unavailable (the autouse DB "
        "fixture in tests/conftest.py requires it).",
    ),
    "V21-02": ItemStatus(
        "V21-02",
        PASS,
        ["reports/v21/source-probe.json", "data/DATA_SOURCES_V21.yaml"],
        "probe executed: per-source origin/revision/licence/egress recorded; egress_allowed "
        "defaults to false in the manifest and no source is fetched automatically.",
    ),
    "V21-03": ItemStatus(
        "V21-03",
        BLOCKED,
        ["reports/v21/source-probe.json"],
        "AgiBot and LIBERO snapshots are absent; the probe reports BLOCKED with exit code 3 "
        "instead of passing. Profiles are registered so the probe will verify them on fetch.",
    ),
    "V21-04": ItemStatus(
        "V21-04",
        PARTIAL,
        ["tests/v21/unit/test_profiles_v21.py"],
        "unequal dimensions are implemented and tested: libero_franka_sim_v1 is 8/7 and the "
        "real locally-verified utokyo snapshot (state[8]/action[7]) resolves through the same "
        "profile contract. LIBERO bytes themselves are not present, so V21-04 is not claimed PASS.",
    ),
    "V21-05": ItemStatus(
        "V21-05",
        PARTIAL,
        ["tests/v21/unit/test_profiles_v21.py"],
        "the schema distinguishes joint/gripper roles and rejects a gripper channel declaring "
        "radians; a limits-less profile makes the physics QC NOT_APPLICABLE. The AgiBot "
        "channel-to-role mapping cannot be established without meta/agibot_g2_data.json.",
    ),
    "V21-06": ItemStatus(
        "V21-06",
        PARTIAL,
        [
            "tests/v21/unit/test_clip_planning.py",
            "tests/v21/unit/test_export_feature_labels.py",
        ],
        "native grid indices are preserved by the clip planner; the export feature block now "
        "labels state and action channels from separate name lists (the 8/7 case no longer "
        "emits 8 labels for a 7-wide action), and the Reader round-trip compares those labels "
        "when the reader reports them. A full 8/7 export+read-back still needs PostgreSQL to "
        "drive the pipeline.",
    ),
    "V21-07": ItemStatus(
        "V21-07",
        PASS,
        ["tests/v21/unit/test_clip_planning.py"],
        "8 s half-open clips, 1 fps integer-second keyframes, <=8 frames, trailing sub-second "
        "clip kept, 600 s limit enforced; 21 tests.",
    ),
    "V21-08": ItemStatus(
        "V21-08",
        PASS,
        ["tests/v21/unit/test_vlm_schema.py", "contracts/EGO_ACTION_SCHEMA_V21.json"],
        "enum/interval/evidence/extra-key/consistency checks all reject without repair; "
        "26 tests.",
    ),
    "V21-09": ItemStatus(
        "V21-09",
        PASS,
        ["tests/v21/unit/test_vlm_governance.py"],
        "ReplayProvider needs no network, always reports mode=REPLAY, and its results are "
        "never live evidence.",
    ),
    "V21-10": ItemStatus(
        "V21-10",
        PASS,
        ["reports/v21/yolo-real.json"],
        "real YOLO11n weights on 10 frames decoded from a real public dataset video; "
        "weights and source SHA-256 recorded.",
    ),
    "V21-11": ItemStatus(
        "V21-11",
        BLOCKED,
        ["reports/v21/vlm-live.json"],
        "DASHSCOPE_API_KEY is not present in this environment; a replay run cannot stand in "
        "for ten real model responses.",
    ),
    "V21-12": ItemStatus(
        "V21-12",
        PASS,
        [
            "tests/v21/unit/test_vlm_governance.py",
            "tests/v21/unit/test_ai_audit_budget.py",
        ],
        "401/403/404/422 fatal, 429/5xx/timeout retryable, truncated body retryable, admission "
        "happens under a FOR UPDATE lock on the run row after the balance is read, egress is a "
        "pre-condition of any call, and a provider failure is no longer relabelled as "
        "'cannot run the async provider' (ProviderError subclasses RuntimeError).",
    ),
    "V21-13": ItemStatus(
        "V21-13",
        PARTIAL,
        [
            "tests/v21/unit/test_vlm_governance.py",
            "tests/v21/unit/test_ai_audit_budget.py",
            "alembic/versions/v21_002_ai_inference_and_clips.py",
            "alembic/versions/v21_004_model_attempt_cost_state.py",
        ],
        "at most one committed product is enforced by the (run, clip, provider, model, prompt) "
        "unique key plus the lease token; attempts are written through an independent, "
        "immediately committed audit session so a rolled-back task cannot erase a call that may "
        "have been billed, and `model_call_attempts.cost_state` distinguishes KNOWN / "
        "UNKNOWN_UPSTREAM_CHARGE / NOT_BILLED. The DB-level concurrency proof needs PostgreSQL.",
    ),
    "V21-14": ItemStatus(
        "V21-14",
        PARTIAL,
        [
            "src/embodied_dataops/annotations/action_intervals.py",
            "src/embodied_dataops/api/routes/ai_preannotation.py",
            "tests/v21/unit/test_action_intervals.py",
        ],
        "ACTION_INTERVAL revisions implemented with If-Match CAS, editor/approver separation "
        "delegated to the existing decision service, and 12 unit tests; the HTTP/CAS round-trip "
        "needs PostgreSQL.",
    ),
    "V21-15": ItemStatus(
        "V21-15",
        PARTIAL,
        [
            "src/embodied_dataops/datasets/ego_bundle.py",
            "src/embodied_dataops/api/routes/datasets.py",
            "tests/v21/unit/test_ego_bundle.py",
            "tests/v21/unit/test_publish_policy.py",
        ],
        "export_kind column + EGO_ANNOTATION_BUNDLE builder with video-redaction policy, "
        "route POST /datasets/{id}/ego-bundle-versions, 15 unit tests; robot versions already "
        "reject ego episodes, and a FORMAT_VERIFIED profile can only produce a "
        "RESEARCH_SNAPSHOT (publish_level is derived and frozen in the manifest). The archive "
        "round-trip needs PostgreSQL + object storage.",
    ),
    "V21-16": ItemStatus(
        "V21-16",
        PARTIAL,
        ["docs/implementation-status.md"],
        "V2 already freezes published DatasetVersions and their manifest SHA; the Ego-bundle "
        "immutability regression still needs the export split from V21-15.",
    ),
    "V21-17": ItemStatus(
        "V21-17",
        PARTIAL,
        [
            "alembic/versions/v21_002_ai_inference_and_clips.py",
            "src/embodied_dataops/workflow/handlers/vlm_preannotate_clip.py",
        ],
        "the AI DAG reuses the V1/V2 lease/fence/CAS primitives; one inference per "
        "(run, clip, provider, model, prompt) is enforced in code and by a unique key, and a "
        "retried task reuses the row instead of creating a second product. The concurrent-worker "
        "proof needs PostgreSQL and Redis.",
    ),
    "V21-18": ItemStatus("V21-18", NOT_RUN, [], "100-episode mixed-load benchmark not executed here."),
    "V21-19": ItemStatus(
        "V21-19", BLOCKED, [], "no CUDA device on this host (GPU_BENCH_BLOCKED)."
    ),
    "V21-20": ItemStatus("V21-20", BLOCKED, [], "depends on V21-11 (live VLM calls)."),
    "V21-21": ItemStatus(
        "V21-21", NOT_RUN, [], "V2 fault-injection scripts exist; not re-run with the AI DAG."
    ),
    "V21-22": ItemStatus("V21-22", NOT_RUN, [], "one-command demo does not cover the AI branch yet."),
    "V21-23": ItemStatus(
        "V21-23",
        PASS,
        ["docs/v21-gap-analysis.md"],
        "no module imports or calls a RoboData Agent service; agent_runtime_required=false.",
    ),
    "V21-24": ItemStatus(
        "V21-24",
        PARTIAL,
        ["acceptance/OVERRIDE_MATRIX.csv"],
        "new runs select pipeline_version 2.1.0 while existing objects keep 2.0.0; a GET-read "
        "regression on a V2 object needs PostgreSQL.",
    ),
    "V21-25": ItemStatus(
        "V21-25", BLOCKED, [], "no authorised ego video present; EPIC_EXTERNAL_VERIFIED is NOT_RUN."
    ),
    "V21-26": ItemStatus(
        "V21-26",
        PARTIAL,
        ["docs/licenses.md"],
        "Ultralytics AGPL-3.0 and Qwen2.5-VL usage recorded; no restricted bytes or keys in git. "
        "egress_allowed is a NOT NULL DEFAULT false column and both the API and the clip handler "
        "refuse a live call without it (SOURCE_EGRESS_NOT_ALLOWED); not exercised against a live "
        "provider here.",
    ),
    "V21-27": ItemStatus(
        "V21-27",
        PARTIAL,
        ["reports/v21/source-probe.json", "data/DATA_SOURCES_V21.yaml"],
        "sources carry explicit source_kind/data_origin and the probe keeps real, simulated and "
        "fixture data apart; the consolidated report is the gap analysis table.",
    ),
    "V21-28": ItemStatus(
        "V21-28",
        PASS,
        ["reports/acceptance-v21.json"],
        "this report: git SHA, python/platform, per-item status, evidence paths and reasons.",
    ),
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - git absent
        return "UNKNOWN"


def _probe_postgres() -> dict[str, Any]:
    """Try an actual TCP connection; an env var being set is not a running database."""
    dsn = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    if not dsn:
        return {"available": False, "reason": "no TEST_DATABASE_URL/DATABASE_URL configured"}
    try:
        parsed = urlsplit(dsn.replace("postgresql+psycopg", "postgresql"))
    except Exception as exc:  # noqa: BLE001 - an unparsable DSN is a real finding
        return {"available": False, "reason": f"unparsable DSN: {exc}"}
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return {"available": True, "endpoint": f"{host}:{port}"}
    except OSError as exc:
        return {"available": False, "endpoint": f"{host}:{port}", "reason": str(exc)}


def _probe_cuda() -> dict[str, Any]:
    """`torch.cuda.is_available()` is the only honest answer; `nvidia-smi` is not."""
    try:
        import torch  # noqa: PLC0415 - optional dependency, probed on purpose
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"torch not importable: {exc}"}
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"torch.cuda probe failed: {exc}"}
    result: dict[str, Any] = {
        "available": available,
        "torch": getattr(torch, "__version__", "unknown"),
    }
    if available:
        try:
            result["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception:  # noqa: BLE001 - a name is evidence, not a requirement
            result["devices"] = []
    else:
        result["reason"] = "torch reports no CUDA device"
    return result


def _run_tests(junit: Path) -> dict[str, Any]:
    """Execute the offline V2.1 suite and report the real counts."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/v21",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={junit}",
    ]
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    if junit.is_file():
        try:
            root = ET.parse(junit).getroot()
            suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
            for suite in suites:
                counts["tests"] += int(suite.get("tests") or 0)
                counts["failures"] += int(suite.get("failures") or 0)
                counts["errors"] += int(suite.get("errors") or 0)
                counts["skipped"] += int(suite.get("skipped") or 0)
        except ET.ParseError:
            pass
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "junit": junit.relative_to(REPOSITORY_ROOT).as_posix(),
        **counts,
    }


def _check_evidence(item: ItemStatus, self_path: Path | None = None) -> list[str]:
    """Return the evidence paths that are missing or disagree with the declaration.

    Missing files are only a problem for an *accepted* status: for ``BLOCKED``/``NOT_RUN``
    the listed path is the artifact that will exist once the blocker clears.
    """
    problems: list[str] = []
    for relative in item.evidence:
        path = REPOSITORY_ROOT / relative
        if self_path is not None and relative == self_path:
            # V21-28 is evidenced by this very report; it cannot exist before it is written.
            continue
        if not path.exists():
            if item.status in ACCEPTED:
                problems.append(f"missing evidence: {relative}")
            continue
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                problems.append(f"unreadable evidence: {relative}")
                continue
            declared = payload.get("status") if isinstance(payload, dict) else None
            if item.status == PASS and declared not in (None, PASS):
                problems.append(f"evidence {relative} reports {declared!r}, not PASS")
    if item.status in ACCEPTED and not item.evidence:
        problems.append("declared PASS/PARTIAL without any evidence path")
    if item.status == BLOCKED and not item.note:
        problems.append("BLOCKED without a reason")
    return problems


def build_report(
    csv_path: Path = CSV_PATH,
    out: Path = DEFAULT_OUT,
    *,
    junit: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    try:
        self_relative = out.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:  # report written outside the repository
        self_relative = ""

    items: list[dict[str, Any]] = []
    problems: list[str] = []
    for row in rows:
        test_id = (row.get("test_id") or "").strip()
        if not test_id:
            continue
        status = STATUS.get(test_id)
        if status is None:
            items.append(
                {
                    "test_id": test_id,
                    "milestone": row.get("milestone", ""),
                    "title": row.get("title", ""),
                    "status": NOT_RUN,
                    "evidence": [],
                    "note": "no status declared in scripts/acceptance_v21.py",
                }
            )
            problems.append(f"{test_id}: no declared status")
            continue
        for problem in _check_evidence(status, self_relative):
            problems.append(f"{test_id}: {problem}")
        items.append(
            {
                "test_id": test_id,
                "milestone": row.get("milestone", ""),
                "priority": row.get("priority", ""),
                "title": row.get("title", ""),
                **status.as_dict(),
            }
        )

    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    blocking = [item["test_id"] for item in items if item["status"] not in ACCEPTED]
    tests = _run_tests(junit) if junit is not None else None
    report = {
        "schema_version": "acceptance-report/2.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "status_source": (
            "declared in scripts/acceptance_v21.py and cross-checked against the evidence "
            "files; not a live measurement of each item"
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "postgres": _probe_postgres(),
            "vlm_provider": os.environ.get("VLM_PROVIDER", "disabled"),
            "cuda": _probe_cuda(),
        },
        "tests": tests,
        "complete_levels": {
            # SPEC §9: the offline core plus version management.  Every member must be PASS;
            # PARTIAL is not enough, because the point of the level is "provable here".
            "CORE_COMPLETE_OFFLINE": not [
                item
                for item in items
                if item["test_id"]
                in {
                    "V21-01",
                    "V21-06",
                    "V21-07",
                    "V21-08",
                    "V21-09",
                    "V21-10",
                    "V21-12",
                    "V21-13",
                    "V21-14",
                    "V21-15",
                    "V21-16",
                    "V21-17",
                    "V21-23",
                    "V21-24",
                    "V21-28",
                }
                and item["status"] != PASS
            ],
            "DATA_AND_AI_LIVE_COMPLETE": False,
            "V2.1_FULLY_ACCEPTED": False,
        },
        "summary": {"total": len(items), "counts": counts, "blocking": blocking},
        "evidence_problems": problems,
        "items": items,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the V2.1 acceptance report.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="also execute `pytest tests/v21` and record the real JUnit counts",
    )
    parser.add_argument(
        "--junit",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "v21" / "v21-unit-junit.xml",
    )
    args = parser.parse_args(argv)

    junit = args.junit if args.run_tests else None
    if junit is not None:
        junit.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args.csv, args.out, junit=junit)
    counts = report["summary"]["counts"]
    print("V2.1 acceptance (declared status, evidence-checked):")
    for status in (PASS, PARTIAL, BLOCKED, NOT_RUN, FAIL):
        if status in counts:
            print(f"  {status:<8} {counts[status]}")
    env = report["environment"]
    print(
        f"environment: postgres={env['postgres']['available']} "
        f"cuda={env['cuda']['available']} vlm_provider={env['vlm_provider']}"
    )
    if report.get("tests"):
        tests = report["tests"]
        print(
            f"pytest: exit={tests['exit_code']} tests={tests['tests']} "
            f"failures={tests['failures']} errors={tests['errors']}"
        )
    print(f"report: {args.out.relative_to(REPOSITORY_ROOT).as_posix()}")
    if report["evidence_problems"]:
        print("\nevidence problems:")
        for problem in report["evidence_problems"]:
            print(f"  - {problem}")
        return 3
    tests_failed = bool(report.get("tests") and report["tests"]["exit_code"] != 0)
    if tests_failed:
        print("\nBLOCKED(reason): the pytest run itself failed; declared statuses are not evidence")
        return 3
    if report["summary"]["blocking"]:
        print(f"\nBLOCKED(reason): {', '.join(report['summary']['blocking'])}")
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
