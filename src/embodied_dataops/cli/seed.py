"""Deterministic seed: generate the fixture set, import it through the real
service/import/task pipeline, and print an evidence summary.

This command intentionally uses the same production code paths as the HTTP API
(`imports.service.initiate_import/complete_import`) and the same task engine
(`workflow.runner.run_until_idle`) - only the HTTP transport is skipped so the
command also works from a host shell during development.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from embodied_dataops.api.auth import hash_api_key
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import ApiKey, Episode, ImportJob, QualityReport, SourceImport
from embodied_dataops.db.session import get_session_factory
from embodied_dataops.domain.enums import Role, SourceType
from embodied_dataops.ids import new_id
from embodied_dataops.imports.service import complete_import, initiate_import
from embodied_dataops.storage.base import StorageBackend, get_storage

FIXTURES_DIR = Path("fixtures")
#: Generated (git-ignored) fixtures live here; ``_fixture_files`` reads from it.
GENERATED_DIR = FIXTURES_DIR / "generate"
WORK_DIR = Path("var/seed")
REPORT_PATH = Path("artifacts/seed-report.json")

#: fixture id -> (source_type, mapping, expected OUTCOME keyword or None)
SEED_CASES: dict[str, dict[str, Any]] = {
    "F01_good_lerobot": {
        "source_type": SourceType.LEROBOT_V3,
        "kind": "directory",
        "mapping": {
            "camera_key": "observation.images.front",
            "state_key": "observation.state",
            "action_key": "action",
            "robot_profile": "arm6_joint_position_v1",
        },
        "expected": "imported and QC PASS",
    },
    "F02_bad_timestamps": {
        "source_type": SourceType.LEROBOT_V3,
        "kind": "directory",
        "mapping": {
            "camera_key": "observation.images.front",
            "robot_profile": "arm6_joint_position_v1",
        },
        "expected": "TIMESTAMP_NON_MONOTONIC quality FAIL",
    },
    "F03_bad_dims": {
        "source_type": SourceType.LEROBOT_V3,
        "kind": "directory",
        "mapping": {
            "camera_key": "observation.images.front",
            "robot_profile": "arm6_joint_position_v1",
        },
        "expected": "DIMENSION_MISMATCH quality FAIL",
    },
    "F04_ego_video": {
        "source_type": SourceType.EGO_MP4,
        "kind": "file",
        "filename": "F04_ego_video.mp4",
        "mapping": {"camera_key": "front"},
        "expected": "ego_video_episode, 100 10Hz samples, no state/action",
    },
    "F05_mcap_good": {
        "source_type": SourceType.MCAP_ROS2_PROFILE_V1,
        "kind": "file",
        "filename": "F05_mcap_good.mcap",
        "mapping": {"camera_key": "/camera/front/image_raw"},
        "expected": "MCAP profile v1, QC PASS",
    },
    "F06_mcap_gap": {
        "source_type": SourceType.MCAP_ROS2_PROFILE_V1,
        "kind": "file",
        "filename": "F06_mcap_gap.mcap",
        "mapping": {"camera_key": "/camera/front/image_raw"},
        "expected": "SYNC_MISSING_STATE quality FAIL",
    },
    "F07_corrupt_mp4": {
        "source_type": SourceType.EGO_MP4,
        "kind": "file",
        "filename": "F07_corrupt_mp4.mp4",
        "mapping": {},
        "expected": "explicit decode error, no silent repair",
    },
    "F08_bad_tar": {
        "source_type": SourceType.LEROBOT_V3,
        "kind": "file",
        "filename": "F08_bad_tar.tar.gz",
        "mapping": {},
        "expected": "rejected archive, no path escape",
    },
    "F10_blurry_video": {
        "source_type": SourceType.EGO_MP4,
        "kind": "file",
        "filename": "F10_blurry_video.mp4",
        "mapping": {},
        "expected": "BLUR_FRAME_RATE WARN",
    },
    "F09_fake_eval": {
        "source_type": None,  # generated as a JSON evaluation payload, not an import
        "kind": "evaluation",
        "expected": "transparent_cup_low_light order requested_count=36",
    },
}


def ensure_seed_principal(session: Session):
    """Return an admin Principal for service calls (creates a key row if needed)."""
    from embodied_dataops.api.auth import Principal

    row = session.scalars(
        select(ApiKey).where(ApiKey.role == Role.ADMIN.value, ApiKey.enabled.is_(True))
    ).first()
    if row is None:
        raw = "dataops_seed_" + new_id()
        row = ApiKey(
            id=new_id(),
            name="seed",
            key_hash=hash_api_key(raw),
            role=Role.ADMIN.value,
            enabled=True,
        )
        session.add(row)
        session.flush()
    return Principal(key_id=row.id, role=Role.ADMIN, name=row.name)


def ensure_fixtures(out_dir: Path | None = None, *, force: bool = False) -> list[dict[str, Any]]:
    """Generate F01-F10; returns the merged manifest entries."""
    from fixtures.fake_eval import generate_f09
    from fixtures.generate_synthetic import generate_all
    from fixtures.make_mcap_ros2 import generate_fixtures

    results = generate_all(out_dir or GENERATED_DIR, force=force)
    generate_fixtures(FIXTURES_DIR / "golden" / "mcap")
    generate_f09()
    manifest_path = FIXTURES_DIR / "manifest.json"
    if manifest_path.is_file():
        return list(json.loads(manifest_path.read_text(encoding="utf-8")).get("fixtures") or [])
    return [result.as_dict() if hasattr(result, "as_dict") else dict(result) for result in results]


def _tar_directory(source: Path, destination: Path) -> Path:
    """Deterministic tar.gz of one generated fixture directory (~hundreds of KB)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        raise FileNotFoundError(f"fixture directory missing: {source} (run `make seed-synthetic`)")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"fixture directory is empty: {source} (run `make seed-synthetic`)")

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        return info

    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(source.name / path.relative_to(source)), filter=_filter)
    return destination


def _fixture_files(case_id: str, spec: dict[str, Any], fixtures_dir: Path) -> tuple[Path, str]:
    """Return (path_to_upload, uploaded_filename)."""
    if spec["kind"] == "directory":
        source = fixtures_dir / "generate" / case_id
        tarball = WORK_DIR / f"{case_id}.tar.gz"
        return _tar_directory(source, tarball), tarball.name
    if spec["kind"] == "evaluation":
        raise RuntimeError("evaluation fixtures are not imports")
    filename = spec["filename"]
    for candidate in (
        fixtures_dir / "golden" / "mcap" / filename,
        fixtures_dir / "generate" / filename,
    ):
        if candidate.is_file():
            return candidate, filename
    raise FileNotFoundError(f"fixture file missing: {filename} (run `make seed-synthetic`)")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_one(
    session_factory: sessionmaker[Session],
    storage: StorageBackend,
    *,
    case_id: str,
    spec: dict[str, Any],
    path: Path,
    filename: str,
) -> str:
    session = session_factory()
    try:
        principal = ensure_seed_principal(session)
        session.flush()
        sha256 = _sha256_file(path)
        size = path.stat().st_size
        payload = {
            "source_type": spec["source_type"].value,
            "filename": filename,
            "size_bytes": size,
            "sha256": sha256,
            "data_origin": "synthetic",
            "license": "synthetic fixture generated by this repository (not real robot data)",
            "mapping": spec.get("mapping") or {},
        }
        initiated = initiate_import(session, principal=principal, payload=payload, request_id="seed")
        storage.put_file(initiated["staging_key"], path, content_type="application/octet-stream")
        complete_import(session, principal=principal, import_id=initiated["import_id"], request_id="seed")
        session.commit()
        return initiated["import_id"]
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _summarize_import(session: Session, import_id: str) -> dict[str, Any]:
    record = session.get(SourceImport, import_id)
    job = session.scalars(
        select(ImportJob).where(ImportJob.import_id == import_id).order_by(ImportJob.created_at.desc())
    ).first()
    episodes = list(
        session.scalars(
            select(Episode)
            .where(Episode.source_import_id == import_id)
            .order_by(Episode.source_episode_index)
        )
    )
    episode_rows = []
    for episode in episodes:
        report = session.scalars(
            select(QualityReport)
            .where(QualityReport.episode_id == episode.id)
            .order_by(QualityReport.created_at.desc())
        ).first()
        episode_rows.append(
            {
                "episode_id": episode.id,
                "source_episode_index": episode.source_episode_index,
                "episode_type": episode.episode_type,
                "status": episode.status,
                "quality_status": episode.quality_status,
                "sample_count": episode.sample_count,
                "duration_ns": episode.duration_ns,
                "quality_verdict": report.status if report else None,
                "finding_rules": [],
            }
        )
    return {
        "import_id": import_id,
        "upload_state": record.upload_state if record else None,
        "error_code": record.error_code if record else None,
        "job_state": job.state if job else None,
        "job_error_code": job.error_code if job else None,
        "episodes": episode_rows,
    }


def run_seed(
    *,
    only: Iterable[str] | None = None,
    run_tasks: bool = True,
    force_fixtures: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    from embodied_dataops.workflow.runner import run_until_idle

    settings = get_settings()
    settings.ensure_directories()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ensure_fixtures(force=force_fixtures)
    storage = get_storage()
    storage.ensure_bucket()
    factory = get_session_factory()

    selected = [case_id for case_id in SEED_CASES if only is None or case_id in set(only)]
    imports: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    for case_id in selected:
        spec = SEED_CASES[case_id]
        if spec["kind"] == "evaluation":
            continue
        try:
            path, filename = _fixture_files(case_id, spec, FIXTURES_DIR)
            imports[case_id] = _import_one(
                factory, storage, case_id=case_id, spec=spec, path=path, filename=filename
            )
            print(f"[seed] queued {case_id} -> import {imports[case_id]}")
        except Exception as exc:  # keep going: one bad fixture must not hide the rest
            failures.append({"fixture_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[seed] FAILED to queue {case_id}: {type(exc).__name__}: {exc}")

    task_counts: dict[str, int] = {}
    if run_tasks and imports:
        task_counts = run_until_idle(factory, max_iterations=5000)
        print(f"[seed] task execution: {task_counts}")

    session = factory()
    try:
        results = []
        for case_id, import_id in imports.items():
            summary = _summarize_import(session, import_id)
            summary["fixture_id"] = case_id
            summary["expected"] = SEED_CASES[case_id]["expected"]
            results.append(summary)
    finally:
        session.close()

    report = {
        "schema_version": "seed-report/1.0",
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "storage_backend": settings.storage_backend,
        "task_counts": task_counts,
        "imports": results,
        "failures": failures,
    }
    if write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(f"[seed] report written to {REPORT_PATH}")
    return report


def print_seed_summary(report: dict[str, Any]) -> None:
    print("-" * 78)
    for row in report.get("imports", []):
        episodes = ", ".join(
            f"ep{episode['source_episode_index']}:{episode['status']}/{episode['quality_status'] or '-'}"
            for episode in row["episodes"]
        )
        print(
            f"  {row['fixture_id']:<22} job={row['job_state'] or '-':<10} "
            f"import={row['upload_state']:<10} episodes=[{episodes}]"
        )
        if report.get("failures"):
            for failure in report["failures"]:
                print(f"  !! {failure['fixture_id']}: {failure['error']}")
    print("-" * 78)

