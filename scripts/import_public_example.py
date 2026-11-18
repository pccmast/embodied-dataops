"""Import one real public LeRobot v3 dataset through the production pipeline.

This is the explicit opt-in path required by SPEC 13.2 / AT-24 / DO2-31:
it downloads a pinned public dataset revision (hf-mirror by default), records
provenance (repo id, revision SHA, licence), packages it as the `.tar.gz`
supported by the import API and runs the real import + pipeline.

Nothing runs at build/test time; the command must be requested by the operator:

    python -m scripts.import_public_example \
        --repo lerobot/utokyo_xarm_pick_and_place \
        --target public-data --storage s3

Output evidence: artifacts/public-dataset/import-report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

REPORT = Path("artifacts/public-dataset/import-report.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_directory(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in source.rglob("*") if path.is_file())

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="lerobot/utokyo_xarm_pick_and_place")
    parser.add_argument("--revision", default=None, help="commit SHA (default: record current HEAD)")
    parser.add_argument("--target", default="public-data")
    parser.add_argument("--camera-key", default="observation.images.hand_image")
    parser.add_argument("--storage", default="s3", choices=["s3", "local"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-tasks", action="store_true")
    parser.add_argument(
        "--pipeline-version",
        default="1.0.0",
        help="pipeline DAG version; use 2.1.0 for profile-bound V2.1 imports",
    )
    parser.add_argument(
        "--profile-id",
        default=None,
        help="robot profile id for V2.1 imports (e.g. utokyo_xarm_pick_and_place_v1)",
    )
    parser.add_argument(
        "--profile-revision",
        type=int,
        default=None,
        help="robot profile revision for V2.1 imports",
    )
    parser.add_argument(
        "--robot-profile",
        default="arm6_joint_position_v1",
        help="V1 mapping.robot_profile; required by VALIDATE_SOURCE even for V2.1 imports",
    )
    parser.add_argument(
        "--camera-keys",
        default=None,
        help=(
            "V2.1 multi-camera mapping as role=feature pairs, e.g. "
            "'front=observation.images.image,wrist=observation.images.hand_image'"
        ),
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.storage:
        os.environ["STORAGE_BACKEND"] = args.storage

    local_dir = Path(args.target) / args.repo.split("/")[-1]
    revision = args.revision
    licence = "unknown"
    if not args.skip_download:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
        info = api.dataset_info(args.repo)
        revision = revision or info.sha
        licence = str((info.cardData or {}).get("license") or "unknown")
        started = time.time()
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
        )
        download_seconds = round(time.time() - started, 1)
    else:
        download_seconds = 0.0
        info_path = local_dir / "meta" / "info.json"
        if not info_path.is_file():
            raise SystemExit(f"local dataset not found: {local_dir}")
        if not revision or licence == "unknown":
            # Metadata-only lookup (no data download) so provenance is always recorded.
            from huggingface_hub import HfApi

            info = HfApi(endpoint=os.environ["HF_ENDPOINT"]).dataset_info(args.repo)
            revision = revision or info.sha
            licence = str((info.cardData or {}).get("license") or "unknown")

    meta = json.loads((local_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    archive = _tar_directory(local_dir, Path("var/public-sample") / f"{local_dir.name}.tar.gz")
    sha256 = _sha256_file(archive)
    size = archive.stat().st_size

    # ---- import through the production services --------------------------
    from embodied_dataops.cli.seed import ensure_seed_principal
    from embodied_dataops.db.session import get_session_factory
    from embodied_dataops.imports.service import complete_import, initiate_import
    from embodied_dataops.storage.base import get_storage
    from embodied_dataops.workflow.runner import run_until_idle

    factory = get_session_factory()
    storage = get_storage()
    storage.ensure_bucket()
    session = factory()
    try:
        principal = ensure_seed_principal(session)
        session.commit()
        payload = {
            "source_type": "LEROBOT_V3",
            "filename": archive.name,
            "size_bytes": size,
            "sha256": sha256,
            "data_origin": "public_dataset",
            "license": licence,
            "source_uri": f"https://huggingface.co/datasets/{args.repo}",
            "source_revision": revision,
            "pipeline_version": args.pipeline_version,
            "mapping": {
                "camera_key": args.camera_key,
                "state_key": "observation.state",
                "action_key": "action",
                "robot_profile": args.robot_profile,
                **(
                    {
                        "camera_keys": {
                            role.strip(): key.strip()
                            for role, key in (
                                pair.split("=", 1) for pair in args.camera_keys.split(",") if "=" in pair
                            )
                        }
                    }
                    if args.camera_keys
                    else {}
                ),
                **(
                    {
                        "profile_id": args.profile_id,
                        "profile_revision": args.profile_revision,
                    }
                    if args.profile_id and args.profile_revision is not None
                    else {}
                ),
            },
        }
        initiated = initiate_import(session, principal=principal, payload=payload, request_id="public-example")
        storage.put_file(initiated["staging_key"], archive, content_type="application/gzip")
        completed = complete_import(
            session, principal=principal, import_id=initiated["import_id"], request_id="public-example"
        )
        session.commit()
        import_id = initiated["import_id"]
        job_id = completed["import_job_id"]
    finally:
        session.close()

    task_counts = {} if args.skip_tasks else run_until_idle(factory, max_iterations=2000)

    # ---- evidence -------------------------------------------------------
    from sqlalchemy import select

    from embodied_dataops.db.models import Episode, ImportJob, QualityFinding, QualityReport

    session = factory()
    try:
        job = session.get(ImportJob, job_id)
        episodes = list(
            session.scalars(select(Episode).where(Episode.source_import_id == import_id))
        )
        episode_rows = []
        for episode in episodes:
            report = session.scalars(
                select(QualityReport)
                .where(QualityReport.episode_id == episode.id)
                .order_by(QualityReport.created_at.desc())
            ).first()
            findings = []
            if report is not None:
                findings = [
                    {"rule_code": row.rule_code, "severity": row.severity, "observed": row.observed_json}
                    for row in session.scalars(
                        select(QualityFinding).where(QualityFinding.report_id == report.id)
                    )
                ]
            episode_rows.append(
                {
                    "episode_id": episode.id,
                    "source_episode_index": episode.source_episode_index,
                    "episode_type": episode.episode_type,
                    "status": episode.status,
                    "quality_status": episode.quality_status,
                    "sample_count": episode.sample_count,
                    "duration_ns": episode.duration_ns,
                    "quality_findings": findings,
                }
            )
    finally:
        session.close()

    report = {
        "schema_version": "public-dataset-import-report/1.0",
        "dataset": {
            "repo_id": args.repo,
            "revision": revision,
            "license": licence,
            "source_uri": f"https://huggingface.co/datasets/{args.repo}",
            "codebase_version": meta.get("codebase_version"),
            "total_episodes": meta.get("total_episodes"),
            "total_frames": meta.get("total_frames"),
            "fps": meta.get("fps"),
            "features": {
                key: {"dtype": value.get("dtype"), "shape": value.get("shape")}
                for key, value in (meta.get("features") or {}).items()
            },
            "archive_sha256": sha256,
            "archive_bytes": size,
            "download_seconds": download_seconds,
            "reader": "official lerobot==0.4.4 LeRobotDataset",
        },
        "import": {
            "import_id": import_id,
            "job_id": job_id,
            "job_state": job.state if job else None,
            "job_error_code": job.error_code if job else None,
            "storage_backend": os.environ.get("STORAGE_BACKEND"),
            "task_counts": task_counts,
            "episodes": episode_rows,
        },
        "limitations": [
            "This public dataset is 8-D state / 7-D action with 3 cameras; it does not match the V1 6-D arm profile.",
            "It is therefore imported for read-only inspection/preview and must not be published as a 6-D robot training set; its action vector is never truncated.",
            "The default v1 importer records DIMENSION_MISMATCH diagnostics instead of silently adapting dimensions.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report["import"][k] for k in ("job_state", "job_error_code", "task_counts")}, default=str))
    print("episodes:", len(episode_rows), "quality:", [row["quality_status"] for row in episode_rows])
    print(f"report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
