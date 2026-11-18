"""100-Episode synthetic load benchmark (SPEC 12.3, AT-22).

The benchmark deliberately uses the FakeDetector: real YOLO CPU inference is
measured separately (AT-21) because its runtime is dominated by the model, not
the platform.  The report records the machine, configuration, timings, final
task states and the duplicate-output check; it never claims production SLA.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread

from sqlalchemy import func, select

from embodied_dataops.config import get_settings
from embodied_dataops.db.models import Asset, Episode, Task
from embodied_dataops.db.session import get_session_factory
from embodied_dataops.imports.service import complete_import, initiate_import
from embodied_dataops.storage.base import get_storage

REPORT_PATH = Path("artifacts/benchmarks/report.json")
WORK_DIR = Path("var/benchmark")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _machine() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
    }


def _import_episodes(session_factory, storage, video: Path, count: int, principal) -> list[str]:
    import_ids: list[str] = []
    data = video.read_bytes()
    sha256 = _sha256_file(video)
    for index in range(count):
        session = session_factory()
        try:
            payload = {
                "source_type": "EGO_MP4",
                "filename": f"bench_episode_{index:03d}.mp4",
                "size_bytes": len(data),
                "sha256": sha256,
                "data_origin": "synthetic",
                "license": "synthetic benchmark fixture",
                "mapping": {},
            }
            initiated = initiate_import(session, principal=principal, payload=payload, request_id="benchmark")
            storage.put_bytes(initiated["staging_key"], data, content_type="video/mp4")
            complete_import(session, principal=principal, import_id=initiated["import_id"], request_id="benchmark")
            session.commit()
            import_ids.append(initiated["import_id"])
        finally:
            session.close()
    return import_ids


def _drain(session_factory, pool: str, worker_id: str, storage, stop_after_seconds: float) -> None:
    from embodied_dataops.workflow.runner import run_until_idle

    deadline = time.time() + stop_after_seconds
    while time.time() < deadline:
        counts = run_until_idle(session_factory, pools=(pool,), max_iterations=50, worker_id=worker_id, storage=storage)
        if counts["iterations"] == 0:
            time.sleep(0.2)
            # one more check for tasks produced by the other worker threads
            continue
    return


def run_benchmark(*, episodes: int = 100, workers: int = 2) -> int:
    from embodied_dataops.cli.seed import ensure_fixtures, ensure_seed_principal

    settings = get_settings()
    settings.ensure_directories()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ensure_fixtures()
    video = Path("fixtures/generate/F04_ego_video.mp4")
    if not video.is_file():
        print("[benchmark] F04 fixture missing; run `make seed-synthetic` first")
        return 2
    storage = get_storage()
    storage.ensure_bucket()
    factory = get_session_factory()
    session = factory()
    try:
        principal = ensure_seed_principal(session)
        session.commit()
    finally:
        session.close()

    started = time.time()
    import_ids = _import_episodes(factory, storage, video, episodes, principal)
    ingest_seconds = time.time() - started
    print(f"[benchmark] ingested {len(import_ids)} episodes in {ingest_seconds:.1f}s")

    started = time.time()
    cpu_threads = [
        Thread(target=_drain, args=(factory, "cpu", f"bench-cpu-{i}", storage, 900), daemon=True)
        for i in range(workers)
    ]
    vision_thread = Thread(
        target=_drain, args=(factory, "vision", "bench-vision-0", storage, 900), daemon=True
    )
    for thread in cpu_threads:
        thread.start()
    vision_thread.start()
    for thread in cpu_threads + [vision_thread]:
        thread.join(timeout=900)
    processing_seconds = time.time() - started

    session = factory()
    try:
        task_states = dict(
            session.execute(select(Task.state, func.count()).group_by(Task.state)).all()
        )
        running = task_states.get("RUNNING", 0)
        outputs = session.execute(
            select(Task.output_asset_id, func.count())
            .where(Task.output_asset_id.is_not(None))
            .group_by(Task.output_asset_id)
            .having(func.count() > 1)
        ).all()
        episodes_ready = session.scalar(
            select(func.count()).select_from(Episode).where(Episode.status == "READY_FOR_REVIEW")
        )
        episode_total = session.scalar(select(func.count()).select_from(Episode))
        assets = session.scalar(select(func.count()).select_from(Asset))
    finally:
        session.close()

    report = {
        "schema_version": "benchmark-report/1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "machine": _machine(),
        "configuration": {
            "episodes": episodes,
            "cpu_workers": workers,
            "vision_workers": 1,
            "detector": settings.detector,
            "storage_backend": settings.storage_backend,
            "database": "postgresql",
            "queue": "redis-streams (wakeup hints only)",
        },
        "timings": {
            "ingest_seconds": round(ingest_seconds, 3),
            "processing_seconds": round(processing_seconds, 3),
            "episodes_per_minute": round(episodes / max(processing_seconds, 1e-9) * 60, 2),
        },
        "results": {
            "episodes_total": episode_total,
            "episodes_ready_for_review": episodes_ready,
            "task_states": task_states,
            "permanent_running_tasks": running,
            "duplicate_output_assets": len(outputs),
            "assets": assets,
        },
        "limitations": [
            "Synthetic 10-second Ego MP4 episodes with FakeDetector; not a production SLA.",
            "Single-host PostgreSQL/Redis and one process per pool in this run.",
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2))
    print(f"[benchmark] report -> {REPORT_PATH}")
    ok = running == 0 and len(outputs) == 0 and episodes_ready == episode_total
    return 0 if ok else 1
