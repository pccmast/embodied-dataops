"""Data-source probe for V2.1 (SPEC_V2.1 §4.1, acceptance V21-02/V21-03).

The probe answers exactly one question per source: **is the bytes actually here and
does it match the declared shape?**  It never downloads anything, never invents a
revision and never reports a source as usable when the files are absent.

Status vocabulary (SPEC_V2.1 §9):

``PASS``     the local copy exists and its ``meta/info.json`` matches the declared
             dimensions / camera count.
``BLOCKED``  the source is required but absent (or the local shape disagrees with the
             declaration).  The reason is recorded; this is never a silent skip.
``NOT_RUN``  optional source, absent.  Does not block anything.

Exit codes: ``0`` all required sources usable, ``3`` at least one required source is
BLOCKED (CLI convention from SPEC_V2.1 §7.2 -- CI must not map 3 to PASS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "DATA_SOURCES_V21.yaml"
DEFAULT_PUBLIC_ROOT = REPOSITORY_ROOT / "public-data"
DEFAULT_FIXTURE_ROOT = REPOSITORY_ROOT / "fixtures"

#: ``repo_id`` is a Hub id; the local snapshot directory is the trailing path part.
#: Mapped explicitly so the probe never guesses a filesystem name from a Hub id.
LOCAL_DIR_BY_REPO: dict[str, str] = {
    "zhengfeirao/2026-9-18": "2026-9-18",
    "nvidia/LIBERO_LeRobot_v3": "LIBERO_LeRobot_v3",
}

PASS = "PASS"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"


@dataclass
class ProbeResult:
    key: str
    source_kind: str
    status: str
    required_for_full: bool
    repo_id: str
    reason: str = ""
    local_path: str = ""
    observed: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source_kind": self.source_kind,
            "status": self.status,
            "required_for_full": self.required_for_full,
            "repo_id": self.repo_id,
            "reason": self.reason,
            "local_path": self.local_path,
            "observed": self.observed,
            "expected": self.expected,
        }


def _load_manifest(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 -- optional dependency kept out of import time

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tree_digest(root: Path, *, limit: int = 200) -> tuple[str, int, int]:
    """Deterministic digest of a directory: ``sha256(name:size:...)``.

    Not a substitute for a per-file manifest; it exists so the probe can report a
    stable identity for a local snapshot without re-reading gigabytes.
    """
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}:{stat.st_size}\n".encode())
        count += 1
        total += stat.st_size
        if count >= limit:
            break
    return digest.hexdigest(), count, total


def _read_info_json(root: Path) -> dict[str, Any] | None:
    for candidate in (root / "meta" / "info.json", root / "info.json"):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
    return None


def _camera_keys(info: dict[str, Any]) -> list[str]:
    return sorted(
        name for name in (info.get("features") or {}) if str(name).startswith("observation.images.")
    )


def _feature_dim(info: dict[str, Any], name: str) -> int | None:
    feature = (info.get("features") or {}).get(name)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def probe_lerobot_source(entry: dict[str, Any], public_root: Path) -> ProbeResult:
    """Check a LeRobot v3 source against ``expected_state_dim`` / ``expected_action_dim``."""
    repo_id = str(entry.get("repo_id") or "")
    key = str(entry.get("key") or "")
    expected = {
        "state_dim": entry.get("expected_state_dim"),
        "action_dim": entry.get("expected_action_dim"),
        "cameras": entry.get("expected_cameras"),
    }
    folder = LOCAL_DIR_BY_REPO.get(repo_id, key)
    root = public_root / folder
    subdir = entry.get("subdir")
    if subdir and (root / str(subdir)).is_dir():
        root = root / str(subdir)

    if not root.is_dir():
        return ProbeResult(
            key=key,
            source_kind=str(entry.get("source_kind") or ""),
            status=BLOCKED if entry.get("required_for_full") else NOT_RUN,
            required_for_full=bool(entry.get("required_for_full")),
            repo_id=repo_id,
            reason=(
                "local snapshot absent; run the documented download step first "
                f"(expected at {root.relative_to(REPOSITORY_ROOT).as_posix()})"
            ),
            expected=expected,
        )

    info = _read_info_json(root)
    if info is None:
        return ProbeResult(
            key=key,
            source_kind=str(entry.get("source_kind") or ""),
            status=BLOCKED,
            required_for_full=bool(entry.get("required_for_full")),
            repo_id=repo_id,
            reason="meta/info.json missing or unreadable; format cannot be verified",
            local_path=str(root),
            expected=expected,
        )

    observed = {
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "state_dim": _feature_dim(info, "observation.state"),
        "action_dim": _feature_dim(info, "action"),
        "cameras": _camera_keys(info),
    }
    digest, files, size = _tree_digest(root)
    observed["snapshot_digest"] = digest
    observed["file_count"] = files
    observed["byte_size"] = size

    mismatches: list[str] = []
    if expected["state_dim"] is not None and observed["state_dim"] != expected["state_dim"]:
        mismatches.append(f"state_dim {observed['state_dim']} != expected {expected['state_dim']}")
    if expected["action_dim"] is not None and observed["action_dim"] != expected["action_dim"]:
        mismatches.append(f"action_dim {observed['action_dim']} != expected {expected['action_dim']}")
    if expected["cameras"] is not None and len(observed["cameras"]) != expected["cameras"]:
        mismatches.append(
            f"camera count {len(observed['cameras'])} != expected {expected['cameras']}"
        )
    min_episodes = entry.get("min_episodes")
    if isinstance(min_episodes, int) and int(observed["total_episodes"] or 0) < min_episodes:
        mismatches.append(
            f"total_episodes {observed['total_episodes']} < min_episodes {min_episodes}"
        )

    if mismatches:
        return ProbeResult(
            key=key,
            source_kind=str(entry.get("source_kind") or ""),
            status=BLOCKED,
            required_for_full=bool(entry.get("required_for_full")),
            repo_id=repo_id,
            reason="local snapshot disagrees with the declared shape: " + "; ".join(mismatches),
            local_path=str(root),
            observed=observed,
            expected=expected,
        )

    return ProbeResult(
        key=key,
        source_kind=str(entry.get("source_kind") or ""),
        status=PASS,
        required_for_full=bool(entry.get("required_for_full")),
        repo_id=repo_id,
        reason="local snapshot present and shape matches the declaration",
        local_path=str(root),
        observed=observed,
        expected=expected,
    )


def probe_ego_source(entry: dict[str, Any], public_root: Path) -> ProbeResult:
    """Ego sources are legitimate only when video bytes exist, not just a CSV."""
    key = str(entry.get("key") or "")
    expected = {"min_clips": entry.get("min_clips"), "dataset_format": entry.get("dataset_format")}
    candidates = [
        public_root / key.lower(),
        public_root / "ego",
        public_root / "ego_authorized",
    ]
    found = next((path for path in candidates if path.is_dir()), None)
    videos = sorted(found.rglob("*.mp4")) if found is not None else []
    min_clips = entry.get("min_clips")
    enough = isinstance(min_clips, int) and len(videos) >= min_clips
    if found is not None and enough:
        return ProbeResult(
            key=key,
            source_kind=str(entry.get("source_kind") or ""),
            status=PASS,
            required_for_full=bool(entry.get("required_for_full")),
            repo_id=str(entry.get("repo_id") or ""),
            reason=f"{len(videos)} authorized ego clips present",
            local_path=str(found),
            observed={"clips": len(videos)},
            expected=expected,
        )
    reason = (
        "no authorized ego video directory found"
        if found is None
        else f"only {len(videos)} clips present, need >= {min_clips}"
    )
    return ProbeResult(
        key=key,
        source_kind=str(entry.get("source_kind") or ""),
        status=BLOCKED if entry.get("required_for_full") else NOT_RUN,
        required_for_full=bool(entry.get("required_for_full")),
        repo_id=str(entry.get("repo_id") or ""),
        reason=reason + " (an annotations CSV alone does not count as ego video ingest)",
        local_path=str(found) if found is not None else "",
        observed={"clips": len(videos)},
        expected=expected,
    )


def probe_mcap_source(entry: dict[str, Any], fixture_root: Path) -> ProbeResult:
    key = str(entry.get("key") or "")
    min_files = entry.get("min_files")
    mcap_root = fixture_root / "golden" / "mcap"
    files = sorted(mcap_root.glob("*.mcap")) if mcap_root.is_dir() else []
    enough = isinstance(min_files, int) and len(files) >= min_files
    return ProbeResult(
        key=key,
        source_kind=str(entry.get("source_kind") or ""),
        status=PASS if enough else (BLOCKED if entry.get("required_for_full") else NOT_RUN),
        required_for_full=bool(entry.get("required_for_full")),
        repo_id=str(entry.get("repo_id") or ""),
        reason=(
            f"{len(files)} MCAP fixtures present"
            if enough
            else f"only {len(files)} MCAP fixtures, need >= {min_files}"
        ),
        local_path=str(mcap_root),
        observed={"mcap_files": [path.name for path in files]},
        expected={"min_files": min_files},
    )


def run_probe(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    public_root: Path = DEFAULT_PUBLIC_ROOT,
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    results: list[ProbeResult] = []
    for entry in manifest.get("sources") or []:
        kind = str(entry.get("source_kind") or "")
        if kind in {"PUBLIC_REAL_ROBOT", "PUBLIC_SIMULATION"}:
            results.append(probe_lerobot_source(entry, public_root))
        elif kind in {"PUBLIC_HUMAN_EGO", "USER_AUTHORIZED"}:
            results.append(probe_ego_source(entry, public_root))
        elif kind == "SYNTHETIC_FIXTURE":
            results.append(probe_mcap_source(entry, fixture_root))
        else:  # pragma: no cover - manifest is a reviewed file
            results.append(
                ProbeResult(
                    key=str(entry.get("key") or ""),
                    source_kind=kind,
                    status=NOT_RUN,
                    required_for_full=bool(entry.get("required_for_full")),
                    repo_id=str(entry.get("repo_id") or ""),
                    reason=f"unknown source_kind {kind!r}",
                )
            )
    blocked = [r.key for r in results if r.required_for_full and r.status == BLOCKED]
    return {
        "schema_version": str(manifest.get("schema_version") or "data-sources/2.1"),
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sources": [r.as_dict() for r in results],
        "summary": {
            "total": len(results),
            "pass": sum(1 for r in results if r.status == PASS),
            "blocked": sum(1 for r in results if r.status == BLOCKED),
            "not_run": sum(1 for r in results if r.status == NOT_RUN),
            "required_blocked": blocked,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the V2.1 declared data sources (offline).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    report = run_probe(args.manifest, public_root=args.public_root, fixture_root=args.fixture_root)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")

    for item in report["sources"]:
        marker = {"PASS": "PASS   ", "BLOCKED": "BLOCKED", "NOT_RUN": "NOT_RUN"}[item["status"]]
        print(f"[{marker}] {item['key']}: {item['reason']}")
    summary = report["summary"]
    print(
        f"\n{summary['pass']} pass / {summary['blocked']} blocked / "
        f"{summary['not_run']} not run (of {summary['total']})"
    )
    if summary["required_blocked"]:
        print(f"BLOCKED(required): {', '.join(summary['required_blocked'])}")
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
