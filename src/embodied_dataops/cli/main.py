"""`python -m embodied_dataops.cli.main <command>` - single local entrypoint.

Mirrors the Makefile targets.  Commands never fabricate results: every state
transition they print comes from PostgreSQL/object storage observations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from embodied_dataops import __version__

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"


def _python() -> str:
    return sys.executable


def _subprocess(command: list[str], *, env: dict | None = None, cwd: Path | None = None) -> int:
    print("[dataops] " + " ".join(command))
    return subprocess.run(command, env=env, cwd=str(cwd or REPO_ROOT)).returncode


def _migrate() -> int:
    return _subprocess([_python(), "-m", "alembic", "upgrade", "head"])


def _database_reachable() -> bool:
    """A TCP probe, so bootstrap can explain the real problem instead of a traceback."""
    import socket
    from urllib.parse import urlsplit

    from embodied_dataops.config import get_settings

    try:
        url = urlsplit(get_settings().sync_database_url.replace("postgresql+psycopg", "postgresql"))
    except Exception:  # noqa: BLE001 - an unparsable DSN is reported by Alembic itself
        return True
    host = url.hostname or "localhost"
    port = url.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


def cmd_bootstrap(args: argparse.Namespace) -> int:
    if not ENV_FILE.exists():
        shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
        print(f"[bootstrap] created {ENV_FILE} from .env.example")
    if not args.no_sync:
        code = _subprocess(["uv", "sync", "--extra", "dev", "--extra", "ui"])
        if code != 0:
            print("[bootstrap] uv sync failed", file=sys.stderr)
            return code
    if not _database_reachable():
        print(
            "[bootstrap] cannot reach the database. Migrations need the dependency "
            "services running first:\n"
            "    docker compose up -d db redis minio\n"
            "  then re-run `make bootstrap`.  (A fresh host cannot migrate against a "
            "database that was never started; this used to surface as a raw Alembic "
            "traceback.)",
            file=sys.stderr,
        )
        return 2
    code = _migrate()
    if code != 0:
        return code
    from embodied_dataops.cli.keys import ensure_bootstrap_keys
    from embodied_dataops.db.session import get_session_factory

    session = get_session_factory()()
    try:
        ensure_bootstrap_keys(session)
        session.commit()
    finally:
        session.close()
    print("[bootstrap] ready: `make up` (docker) or start the API/worker processes directly")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    return _migrate()


def cmd_fixtures(args: argparse.Namespace) -> int:
    from embodied_dataops.cli.seed import FIXTURES_DIR, ensure_fixtures

    out_dir = Path(args.out) if args.out else FIXTURES_DIR / "generate"
    if args.only:
        from fixtures.generate_synthetic import generate_all

        generate_all(out_dir, only=args.only, force=args.force)
    else:
        ensure_fixtures(out_dir, force=args.force)
    manifest = FIXTURES_DIR / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for fixture in payload.get("fixtures", []):
            print(f"  {fixture['fixture_id']:<22} {fixture.get('status', '?'):<10} {fixture.get('sha256') or '-'}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    from embodied_dataops.cli.seed import print_seed_summary, run_seed

    report = run_seed(
        only=args.only or None,
        run_tasks=not args.no_tasks,
        force_fixtures=args.force_fixtures,
    )
    print_seed_summary(report)
    return 0 if not report.get("failures") else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from embodied_dataops.cli.demo import run_demo

    return run_demo(force_fixtures=args.force_fixtures)


def cmd_acceptance(args: argparse.Namespace) -> int:
    from embodied_dataops.cli.acceptance import run_acceptance

    return run_acceptance(include_performance=args.include_performance)


def cmd_ui(args: argparse.Namespace) -> int:
    """Run the Streamlit UI on the host (no container image required).

    The container image is built on Linux/CI (ADR-0010); on Windows this host
    launcher is the supported path.  It uses the same source code and an API key
    from ``var/bootstrap/api_keys.json`` (or the API_KEY env var).
    """
    from embodied_dataops.cli.keys import load_dev_keys

    script = Path(args.script)
    if not script.is_file():
        print(f"UI entrypoint not found: {script}", file=sys.stderr)
        return 2
    env = dict(os.environ)
    if not any(env.get(name) for name in ("API_KEY", "UI_API_KEY", "DATAOPS_API_KEY")):
        keys = load_dev_keys()
        for role in ("viewer", "operator", "reviewer"):
            if keys.get(role):
                env["API_KEY"] = keys[role]
                break
        if not env.get("API_KEY"):
            print(
                "No API key available. Run `make bootstrap` first or set API_KEY.",
                file=sys.stderr,
            )
            return 2
    if not any(env.get(name) for name in ("UI_API_BASE_URL", "API_PUBLIC_BASE_URL", "API_BASE_URL")):
        from embodied_dataops.config import get_settings

        env["UI_API_BASE_URL"] = get_settings().api_public_base_url
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        "true",
    ]
    print("[ui] " + " ".join(command))
    print(f"[ui] API base URL: {env.get('UI_API_BASE_URL')}")
    return subprocess.run(command, env=env).returncode


def cmd_profiles(args: argparse.Namespace) -> int:
    """Register/print the built-in robot profiles (idempotent)."""
    from sqlalchemy import select

    from embodied_dataops.db.models import RobotProfile
    from embodied_dataops.db.session import get_session_factory
    from embodied_dataops.domain.profiles import upsert_builtin_profiles

    session = get_session_factory()()
    try:
        # ``--include-v21`` also registers the robot-profile/2.1 revisions.  It stays
        # opt-in so V2 seeding is byte-identical, but a 2.1 import *needs* the row:
        # qc-policy/2.0 reports MISSING_ROBOT_PROFILE when the registry has no
        # (profile_id, profile_revision) row, even though the profile file has one.
        upsert_builtin_profiles(session, include_v21=bool(getattr(args, "include_v21", False)))
        session.commit()
        rows = list(session.scalars(select(RobotProfile).order_by(RobotProfile.profile_id)))
    finally:
        session.close()
    print(f"profiles: {len(rows)} registered")
    for row in rows:
        config = row.config_json
        # 2.0 rows carry ``joint_names``; 2.1 rows describe ``state.dimension`` instead.
        state = config.get("state") if isinstance(config.get("state"), dict) else {}
        state_dim = len(config.get("joint_names") or []) or int(state.get("dimension") or 0)
        print(
            f"  {row.profile_id}@rev{row.profile_revision} authority={row.authority} "
            f"state_dim={state_dim} sha256={row.config_sha256[:16]}..."
        )
    return 0


def cmd_fixtures_v2(args: argparse.Namespace) -> int:
    """Generate the deterministic V2 fixture manifest (D2-M0 skeleton)."""
    from fixtures.v2.generate import generate_v2_fixtures

    manifest = generate_v2_fixtures(args.output_dir, profile_file=args.profile_file)
    print(f"fixture-v2: {len(manifest['profiles'])} profiles, {len(manifest['fixtures'])} fixtures")
    return 0


def cmd_openapi(args: argparse.Namespace) -> int:
    from embodied_dataops.api.app import app

    target = Path(args.output) if args.output else REPO_ROOT / "docs" / "api-openapi.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[openapi] wrote {target} ({len(app.routes)} routes)")
    return 0


def cmd_openapi_v2(args: argparse.Namespace) -> int:
    """Write the DO-A02 v2 OpenAPI document (with x-contract-version)."""
    from embodied_dataops.api.app import app

    target = Path(args.output) if args.output else REPO_ROOT / "docs" / "api-openapi-v2.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[openapi-v2] wrote {target} ({len(app.routes)} routes)")
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from embodied_dataops.cli.keys import create_key, list_keys
    from embodied_dataops.db.models import ApiKey
    from embodied_dataops.db.session import get_session_factory

    session = get_session_factory()()
    try:
        if args.create:
            create_key(session, args.create, args.name or "")
            return 0
        rows = list(session.scalars(select(ApiKey).order_by(ApiKey.created_at)))
        if not rows:
            print("No API keys yet. Run `make bootstrap` (prints keys once).")
            return 1
        for row in list_keys(session):
            print(
                f"  {row['role']:<9} {row['id']}  {row['name']:<20} "
                f"enabled={row['enabled']} hash={row['key_hash_prefix']}"
            )
    finally:
        session.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    matrix_file = REPO_ROOT / "artifacts" / "test-results" / "at-matrix.json"
    if not matrix_file.is_file():
        print("No acceptance matrix yet. Run `make acceptance`.")
        return 1
    payload = json.loads(matrix_file.read_text(encoding="utf-8"))
    print(f"Acceptance matrix generated at {payload['generated_at']}")
    print(f"tests: {payload['passed']} passed / {payload['failed']} failed / {payload['skipped']} skipped")
    for at_id, row in payload["matrix"].items():
        print(f"  {at_id:<6} {row['status']:<9} {row['description']}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    if (args.confirm or "").upper() != "YES":
        print(
            "Refusing to clean without explicit confirmation.\n"
            "Run `make clean CONFIRM=YES` (removes var/, fixtures/generate, caches; keeps DB volumes).",
            file=sys.stderr,
        )
        return 2
    for path in (REPO_ROOT / "var", REPO_ROOT / "fixtures" / "generate", REPO_ROOT / ".pytest_cache"):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"[clean] removed {path}")
    if args.volumes:
        _subprocess(["docker", "compose", "down", "-v"])
    print("[clean] done (database/object-store volumes kept unless --volumes was passed)")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    from embodied_dataops.cli.gc import collect_garbage
    from embodied_dataops.db.session import get_session_factory
    from embodied_dataops.storage.base import get_storage

    session = get_session_factory()()
    try:
        result = collect_garbage(
            session,
            get_storage(),
            orphan_hours=args.orphan_hours,
            dry_run=not args.delete,
        )
    finally:
        session.close()
    print(json.dumps({k: v for k, v in result.items() if k != "protected"}, indent=2, default=str))
    print(f"protected: {len(result['protected'])} referenced/young objects")
    return 0


def cmd_fetch_public_example(args: argparse.Namespace) -> int:
    from embodied_dataops.cli.public_data import fetch_public_example

    return fetch_public_example(args.repo, args.revision, args.target)


def cmd_benchmark(args: argparse.Namespace) -> int:
    from embodied_dataops.cli.benchmark import run_benchmark

    return run_benchmark(episodes=args.episodes, workers=args.workers)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _read_recipe(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"recipe must be a mapping, got {type(payload).__name__}")
    return payload


def cmd_training_snapshot_source(args: argparse.Namespace) -> int:
    """派生一份 ``training-source-snapshot/1.0``，供 synthetic-fixture 模式使用。"""
    from embodied_dataops.training.artifact import snapshot_source_document
    from embodied_dataops.training.contract import canonical_json_bytes

    document = snapshot_source_document(
        dataset_root=args.dataset_root,
        dataset_version_id=args.dataset_version_id,
        license_id=args.license,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(document))
    _emit({"status": "PASS", "snapshot": str(target), "episodes": len(document["episodes"])})
    return 0


def cmd_training_prepare(args: argparse.Namespace) -> int:
    """准备并原子发布 TrainingBundle；退出码按规格 §12 映射。"""
    from embodied_dataops.errors import ApiError
    from embodied_dataops.training import exit_code_for
    from embodied_dataops.training.artifact import load_source_package, prepare_bundle
    from embodied_dataops.training.contract import parse_recipe

    level = "CPU_VERIFIED" if args.verify == "cpu" else "PREPARED"
    try:
        recipe = parse_recipe(_read_recipe(Path(args.recipe)))
        source = load_source_package(
            source_root=args.source_root,
            manifest_path=args.source_manifest,
            source_mode=args.source_mode,
            dataset_version_id=args.dataset_version_id,
            license_id=args.license,
            digest_files=not args.no_source_digest,
        )
        prepared = prepare_bundle(
            recipe=recipe,
            source=source,
            output_dir=args.output,
            report_dir=args.report_dir,
            requested_level=level,
            check_video=not args.no_video_probe,
            tolerance_s=args.tolerance_s,
        )
    except ApiError as exc:
        _emit({"status": "FAIL", "requested_level": level, "error": exc.to_dict()})
        print(f"[training-prepare] {exc.code.value}: {exc.message}", file=sys.stderr)
        return exit_code_for(exc)
    except (OSError, ValueError) as exc:
        _emit({"status": "FAIL", "requested_level": level, "error": {"code": "INVALID_INPUT", "message": str(exc)}})
        print(f"[training-prepare] {exc}", file=sys.stderr)
        return 2

    summary = prepared.summary()
    summary["status"] = "PASS"
    summary["requested_level"] = level
    summary["source_verification"] = source.source_verification
    summary["report_dir"] = args.report_dir or str(Path(args.output).parent / f"{Path(args.output).name}-reports")
    _emit(summary)
    return 0


def cmd_training_validate(args: argparse.Namespace) -> int:
    """对已发布 bundle 做真实计算验证；只写外部 verification record。"""
    from embodied_dataops.errors import ApiError
    from embodied_dataops.training import exit_code_for
    from embodied_dataops.training.validate import run_act_validation

    try:
        report = run_act_validation(
            bundle=args.bundle,
            source_root=args.source_root,
            device=args.device,
            steps=args.steps,
            report_dir=args.report_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            report_name=args.report_name,
            command=getattr(args, "_command", None),
        )
    except ApiError as exc:
        _emit(
            {
                "status": "BLOCKED" if exit_code_for(exc) == 3 else "FAIL",
                "requested_level": "GPU_VERIFIED" if str(args.device).startswith("cuda") else "CPU_VERIFIED",
                "error": exc.to_dict(),
            }
        )
        print(f"[training-validate] {exc.code.value}: {exc.message}", file=sys.stderr)
        return exit_code_for(exc)
    except (OSError, ValueError) as exc:
        _emit({"status": "FAIL", "error": {"code": "INVALID_INPUT", "message": str(exc)}})
        print(f"[training-validate] {exc}", file=sys.stderr)
        return 2

    _emit(
        {
            "status": report["status"],
            "requested_level": report["requested_level"],
            "level_status": report.get("level_status"),
            "achieved_level": report["achieved_level"],
            "limitations": report.get("limitations", []),
            "bundle": report["bundle"],
            "report_dir": args.report_dir,
        }
    )
    return 0 if report["status"] == "PASS" else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataops", description="Embodied DataOps local control CLI")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("help", help="show this help")
    p.set_defaults(func=lambda args: (parser.print_help() or 0))

    p = sub.add_parser("bootstrap", help="create .env, install deps, migrate, print initial API keys")
    p.add_argument("--no-sync", action="store_true")
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("migrate", help="apply Alembic migrations")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("profiles", help="register/print the built-in robot profiles (V2)")
    p.add_argument(
        "--include-v21",
        action="store_true",
        help="also register the robot-profile/2.1 revisions (required before a 2.1 import can pass QC)",
    )
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("fixtures-v2", help="generate the deterministic V2 fixture manifest")
    p.add_argument("--output-dir", default="fixtures/v2")
    p.add_argument("--profile-file", default=None)
    p.set_defaults(func=cmd_fixtures_v2)

    p = sub.add_parser("fixtures", help="generate the ten synthetic fixtures")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fixtures)

    p = sub.add_parser("seed", help="import the fixtures through the real pipeline")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--no-tasks", action="store_true")
    p.add_argument("--force-fixtures", action="store_true")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("demo", help="scripted end-to-end demo with evidence output")
    p.add_argument("--force-fixtures", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("ui", help="run the Streamlit UI on the host (no container image)")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--address", default="127.0.0.1")
    p.add_argument("--script", default=str(REPO_ROOT / "src" / "embodied_ui" / "app.py"))
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("openapi-v2", help="write docs/api-openapi-v2.json (DO-A02)")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_openapi_v2)

    p = sub.add_parser("acceptance", help="run the AT suite and write junit/matrix evidence")
    p.add_argument("--include-performance", action="store_true")
    p.set_defaults(func=cmd_acceptance)

    p = sub.add_parser("openapi", help="regenerate docs/api-openapi.json")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_openapi)

    p = sub.add_parser("keys", help="list key metadata or create one (prints plaintext once)")
    p.add_argument("--create", default=None, choices=["viewer", "operator", "reviewer", "admin"])
    p.add_argument("--name", default="")
    p.set_defaults(func=cmd_keys)

    p = sub.add_parser("status", help="print the last acceptance matrix")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("clean", help="dev cleanup, requires --confirm YES")
    p.add_argument("--confirm", default="")
    p.add_argument("--volumes", action="store_true")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("gc", help="orphan-object GC (never deletes referenced assets)")
    p.add_argument("--orphan-hours", type=int, default=24)
    p.add_argument("--delete", action="store_true", help="actually delete (default is dry-run)")
    p.set_defaults(func=cmd_gc)

    p = sub.add_parser("fetch-public-example", help="explicitly download the public LeRobot v3 dataset")
    p.add_argument("--repo", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--target", default=None)
    p.set_defaults(func=cmd_fetch_public_example)

    p = sub.add_parser("benchmark", help="100-episode synthetic load benchmark (AT-22)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser(
        "training-snapshot-source",
        help="derive a training-source-snapshot/1.0 for --source-mode synthetic-fixture",
    )
    p.add_argument("--dataset-root", required=True, help="exported LeRobot v3 dataset root")
    p.add_argument("--dataset-version-id", required=True)
    p.add_argument("--license", default="unknown")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_training_snapshot_source)

    p = sub.add_parser(
        "training-prepare",
        help="prepare and atomically publish an ACT TrainingBundle (PREPARED/CPU_VERIFIED)",
    )
    p.add_argument("--recipe", required=True, help="training-recipe/1.0 YAML or JSON")
    p.add_argument("--source-root", required=True, help="exported LeRobot dataset root (read-only)")
    p.add_argument("--source-manifest", default=None, help="frozen source manifest (required for published)")
    p.add_argument(
        "--source-mode",
        default="published",
        choices=["published", "synthetic-fixture"],
        help="published performs a read-only PostgreSQL verification of the DatasetVersion",
    )
    p.add_argument("--dataset-version-id", default=None, help="platform DatasetVersion id (published mode)")
    p.add_argument("--license", default=None)
    p.add_argument("--output", required=True, help="target bundle directory (must not conflict)")
    p.add_argument("--report-dir", default=None)
    p.add_argument("--verify", default="none", choices=["none", "cpu"])
    p.add_argument(
        "--no-video-probe",
        action="store_true",
        help="skip the anchor decode probe (recorded as video_probe=disabled; not CPU_VERIFIED evidence)",
    )
    p.add_argument("--no-source-digest", action="store_true")
    p.add_argument("--tolerance-s", type=float, default=1e-4)
    p.set_defaults(func=cmd_training_prepare)

    p = sub.add_parser(
        "training-validate",
        help="run real ACT forward/backward/reload validation against a published bundle",
    )
    p.add_argument("--bundle", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--device", default="cpu", help="cpu or cuda:N (CUDA failure is BLOCKED, never CPU fallback)")
    p.add_argument("--steps", type=int, default=1, help="optimizer steps (positive integer)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--report-name", default=None)
    p.set_defaults(func=cmd_training_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    # 验证报告要记录可复现的调用命令（§13 的机器记录项之一），这里统一回填。
    raw = list(sys.argv[1:] if argv is None else argv)
    args._command = " ".join(["embodied-dataops", *raw])
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
