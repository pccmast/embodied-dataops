#!/usr/bin/env python
"""D2-M6 / DO2-32: real measured 100-Episode bench for the V2 platform.

What it does
------------
* ingests N deterministic synthetic Ego MP4 Episodes through the *real* import
  service (no HTTP proxy; same DB transaction + outbox + task engine),
* drains the CPU pool with ``concurrency`` workers (and 1 vision worker) using the
  production ``workflow.runner.run_until_idle`` driver,
* records ingest/processing wall time, Episodes/min, task completion latency
  P50/P95 from DB timestamps, final task/episode states, permanent RUNNING tasks,
  duplicate output assets, process/system CPU+memory and the exact environment,
* repeats each CPU concurrency N times and reports mean/stddev,
* writes ``artifacts/benchmarks/bench-v2.json`` (schema ``benchmark-report/2.0``).

Honesty rules
-------------
* only measured values are written; the 4-worker/1-worker ratio is computed from
  measured means and is never asserted to be an improvement,
* synthetic Episodes + FakeDetector are labelled as such; this is not production
  robot throughput,
* workers are in-process threads driving the same production task engine (one
  host process); this limitation is recorded explicitly.
* exit code is non-zero when any run leaves a permanent RUNNING task, a
  non-terminal episode/task, or a duplicate referenced output asset.

Usage
-----
    python scripts/bench_v2.py --episodes 100 --repetitions 3 --concurrencies 1,4
    python scripts/bench_v2.py --episodes 5 --repetitions 1 --quick   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

REPORT_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "bench-v2.json"
DEFAULT_VIDEO = REPO_ROOT / "fixtures" / "generate" / "F04_ego_video.mp4"
FALLBACK_VIDEO = REPO_ROOT / "fixtures" / "F04_ego_video.mp4"

TERMINAL_TASK_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}
TERMINAL_EPISODE_STATES = {"READY_FOR_REVIEW", "QUALITY_FAILED"}

FULL_RUN_COMMAND = (
    "python scripts/bench_v2.py --episodes 100 --repetitions 3 --concurrencies 1,4"
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 6)


def _mean_std(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), 6),
        "stddev": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _mask_credentials(url: str) -> str:
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", url)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:  # noqa: BLE001 - git is optional
        return None
    sha = (result.stdout or "").strip()
    return sha if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def _machine() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "fqdn": platform.node(),
        "git_sha": _git_sha(),
        "git_sha_note": None
        if _git_sha()
        else "workspace is not a git checkout or git is unavailable",
    }
    try:
        import psutil  # noqa: PLC0415 - optional dependency

        memory = psutil.virtual_memory()
        info["memory_total_mb"] = round(memory.total / (1024**2), 1)
        try:
            info["cpu_freq_mhz"] = round(psutil.cpu_freq().max or 0.0, 1)
        except Exception:  # noqa: BLE001 - Windows sometimes lacks cpu_freq
            info["cpu_freq_mhz"] = None
    except Exception:  # noqa: BLE001 - psutil optional
        info["memory_total_mb"] = None
        info["cpu_freq_mhz"] = None
    return info


def _environment() -> dict:
    from embodied_dataops.config import get_settings

    settings = get_settings()
    return {
        "database": _mask_credentials(settings.database_url),
        "redis": _mask_credentials(settings.redis_url),
        "storage_backend": settings.storage_backend,
        "storage_root": str(settings.local_storage_root)
        if settings.storage_backend == "local"
        else settings.s3_endpoint_internal,
        "detector": settings.detector,
        "pipeline_version": settings.pipeline_version,
        "lease_seconds": settings.lease_seconds,
        "heartbeat_seconds": settings.heartbeat_seconds,
        "reaper_interval_seconds": settings.reaper_interval_seconds,
        "max_attempts": settings.max_attempts,
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "env_subset": {
            key: value
            for key, value in os.environ.items()
            if key in {"ENVIRONMENT", "TEST_DATABASE_URL", "TEST_REDIS_URL", "STORAGE_BACKEND", "DETECTOR"}
        },
    }


class _ResourceMonitor(threading.Thread):
    """Samples process-tree RSS, system CPU and system memory during a run."""

    def __init__(self, interval: float = 0.3) -> None:
        super().__init__(daemon=True, name="bench-resource-monitor")
        self.interval = interval
        self._stop_event = threading.Event()
        self.available = False
        self.reason: str | None = None
        self.cpu_samples: list[float] = []
        self.memory_used_samples: list[int] = []
        self.peak_process_rss: int = 0
        self.peak_tree_rss: int = 0
        try:
            import psutil  # noqa: PLC0415

            self._psutil = psutil
            self._process = psutil.Process()
            self.available = True
        except Exception as exc:  # noqa: BLE001 - resource metrics are optional
            self._psutil = None
            self._process = None
            self.reason = f"{exc.__class__.__name__}: {exc}"

    def run(self) -> None:  # pragma: no cover - timing thread
        if not self.available:
            return
        psutil = self._psutil
        try:
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            while not self._stop_event.wait(self.interval):
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
                try:
                    rss = self._process.memory_info().rss
                    self.peak_process_rss = max(self.peak_process_rss, rss)
                    tree = rss + sum(
                        child.memory_info().rss
                        for child in self._process.children(recursive=True)
                        if child.is_running()
                    )
                    self.peak_tree_rss = max(self.peak_tree_rss, tree)
                except Exception:  # noqa: BLE001 - short-lived children race
                    pass
                self.memory_used_samples.append(psutil.virtual_memory().used)
        except Exception:  # noqa: BLE001 - monitoring must not break the benchmark
            self.available = False

    def stop(self) -> dict:
        self._stop_event.set()
        self.join(timeout=5)
        if not self.available:
            return {"available": False, "reason": self.reason, "cpu": "N/A", "memory": "N/A"}
        return {
            "available": True,
            "cpu_percent_avg": round(statistics.fmean(self.cpu_samples), 2) if self.cpu_samples else None,
            "cpu_percent_max": round(max(self.cpu_samples), 2) if self.cpu_samples else None,
            "process_peak_rss_mb": round(self.peak_process_rss / (1024**2), 1),
            "process_tree_peak_rss_mb": round(self.peak_tree_rss / (1024**2), 1),
            "system_memory_peak_used_mb": round(max(self.memory_used_samples) / (1024**2), 1)
            if self.memory_used_samples
            else None,
            "system_memory_total_mb": round(self._psutil.virtual_memory().total / (1024**2), 1),
            "capability_note": "no GPU utilisation: CPU-only host, Vision device N/A",
        }


# ---------------------------------------------------------------------------
# ingest + drain
# ---------------------------------------------------------------------------
def _video_path(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"video fixture not found: {candidate}")
        return candidate
    if DEFAULT_VIDEO.is_file():
        return DEFAULT_VIDEO
    if FALLBACK_VIDEO.is_file():
        return FALLBACK_VIDEO
    raise FileNotFoundError(
        f"no synthetic video fixture found ({DEFAULT_VIDEO} / {FALLBACK_VIDEO}); "
        "run `make seed-synthetic` first"
    )


def _ingest_episodes(factory, storage, video: Path, count: int, run_label: str, principal) -> list[str]:
    """Import ``count`` Episodes through the real import service; returns import ids."""
    from embodied_dataops.imports.service import complete_import, initiate_import

    data = video.read_bytes()
    import hashlib

    sha256 = hashlib.sha256(data).hexdigest()
    import_ids: list[str] = []
    for index in range(count):
        session = factory()
        try:
            payload = {
                "source_type": "EGO_MP4",
                "filename": f"bench_{run_label}_{index:03d}.mp4",
                "size_bytes": len(data),
                "sha256": sha256,
                "data_origin": "synthetic",
                "license": "synthetic 10s Ego MP4 fixture generated by this repository",
                "mapping": {"camera_key": "front"},
                "source_uri": "fixtures/generate/F04_ego_video.mp4",
                "source_revision": "bench-v2",
            }
            initiated = initiate_import(
                session, principal=principal, payload=payload, request_id=f"bench-{run_label}"
            )
            storage.put_bytes(initiated["staging_key"], data, content_type="video/mp4")
            complete_import(
                session,
                principal=principal,
                import_id=initiated["import_id"],
                request_id=f"bench-{run_label}",
            )
            session.commit()
            import_ids.append(initiated["import_id"])
        finally:
            session.close()
    return import_ids


def _scope_episode_ids(session, import_ids: list[str]) -> list[str]:
    from sqlalchemy import select

    from embodied_dataops.db.models import Episode

    if not import_ids:
        return []
    return list(session.scalars(select(Episode.id).where(Episode.source_import_id.in_(import_ids))))


def _scope_task_rows(session, import_ids: list[str]) -> list[tuple]:
    """Return (task_id, state, step, created_at, updated_at, output_asset_id, run_id) in scope."""
    from sqlalchemy import select

    from embodied_dataops.db.models import ImportJob, PipelineRun, Task

    if not import_ids:
        return []
    job_ids = list(
        session.scalars(select(ImportJob.id).where(ImportJob.import_id.in_(import_ids)))
    )
    episode_ids = _scope_episode_ids(session, import_ids)
    run_ids = (
        list(session.scalars(select(PipelineRun.id).where(PipelineRun.episode_id.in_(episode_ids))))
        if episode_ids
        else []
    )
    conditions = []
    if job_ids:
        conditions.append(Task.import_job_id.in_(job_ids))
    if run_ids:
        conditions.append(Task.run_id.in_(run_ids))
    if not conditions:
        return []
    from sqlalchemy import or_

    statement = select(
        Task.id,
        Task.state,
        Task.step,
        Task.created_at,
        Task.updated_at,
        Task.output_asset_id,
        Task.run_id,
    ).where(or_(*conditions))
    return list(session.execute(statement))


def _scope_pending_counts(session, import_ids: list[str]) -> tuple[int, int]:
    rows = _scope_task_rows(session, import_ids)
    pending = sum(1 for row in rows if row[1] not in TERMINAL_TASK_STATES)
    running = sum(1 for row in rows if row[1] == "RUNNING")
    return pending, running


def _drain_worker(factory, pool: str, worker_id: str, storage, stop: threading.Event, deadline: float, poll: float = 0.2) -> None:
    from embodied_dataops.workflow.runner import run_until_idle

    while not stop.is_set() and time.time() < deadline:
        counts = run_until_idle(
            factory, pools=(pool,), max_iterations=25, worker_id=worker_id, storage=storage
        )
        if counts["iterations"] == 0:
            time.sleep(poll)


def _drain(
    factory, storage, import_ids: list[str], *, cpu_concurrency: int, vision_concurrency: int, timeout_seconds: float, run_label: str
) -> tuple[float, dict]:
    """Drain the task DAG; returns (processing_seconds, resource snapshot)."""
    stop = threading.Event()
    deadline = time.time() + timeout_seconds
    monitor = _ResourceMonitor()
    monitor.start()
    threads = [
        threading.Thread(
            target=_drain_worker,
            args=(factory, "cpu", f"bench-{run_label}-cpu-{index}", storage, stop, deadline),
            name=f"bench-{run_label}-cpu-{index}",
            daemon=True,
        )
        for index in range(cpu_concurrency)
    ] + [
        threading.Thread(
            target=_drain_worker,
            args=(factory, "vision", f"bench-{run_label}-vision-{index}", storage, stop, deadline),
            name=f"bench-{run_label}-vision-{index}",
            daemon=True,
        )
        for index in range(vision_concurrency)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    try:
        while time.time() < deadline:
            session = factory()
            try:
                pending, _running = _scope_pending_counts(session, import_ids)
            finally:
                session.close()
            if pending == 0:
                # Everything terminal: stop the drainers after a short settle.
                time.sleep(0.5)
                session = factory()
                try:
                    pending, _running = _scope_pending_counts(session, import_ids)
                finally:
                    session.close()
                if pending == 0:
                    break
            time.sleep(0.25)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=10)
        elapsed = time.perf_counter() - started
        resources = monitor.stop()
    return elapsed, resources


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _run_metrics(factory, import_ids: list[str], *, ingest_seconds: float, processing_seconds: float) -> dict:
    from sqlalchemy import select

    from embodied_dataops.db.models import Asset, Episode, PipelineRun

    session = factory()
    try:
        rows = _scope_task_rows(session, import_ids)
        state_counts = Counter(row[1] for row in rows)
        terminal = [row for row in rows if row[1] in TERMINAL_TASK_STATES]
        latencies = [
            (row[4] - row[3]).total_seconds()
            for row in terminal
            if row[3] is not None and row[4] is not None and row[4] >= row[3]
        ]
        permanent_running = [
            {"task_id": row[0], "step": row[2]}
            for row in rows
            if row[1] == "RUNNING"
        ]
        non_terminal = [
            {"task_id": row[0], "step": row[2], "state": row[1]}
            for row in rows
            if row[1] not in TERMINAL_TASK_STATES
        ]
        output_ids = [row[5] for row in terminal if row[5]]
        duplicates = {
            asset_id: count
            for asset_id, count in Counter(output_ids).items()
            if count > 1
        }
        duplicate_output_tasks = [
            {"task_id": row[0], "step": row[2], "output_asset_id": row[5]}
            for row in terminal
            if row[5] in duplicates
        ]
        duplicate_asset_keys = (
            list(
                session.execute(
                    select(Asset.object_key, Asset.sha256).where(
                        Asset.id.in_(list(duplicates))
                    )
                ).all()
            )
            if duplicates
            else []
        )

        episode_ids = _scope_episode_ids(session, import_ids)
        episodes = list(session.scalars(select(Episode).where(Episode.id.in_(episode_ids)))) if episode_ids else []
        episode_status_counts = Counter(episode.status for episode in episodes)
        non_terminal_episodes = [
            {"episode_id": episode.id, "status": episode.status}
            for episode in episodes
            if episode.status not in TERMINAL_EPISODE_STATES
        ]
        run_latencies: list[float] = []
        if episode_ids:
            runs = list(
                session.scalars(
                    select(PipelineRun).where(PipelineRun.episode_id.in_(episode_ids))
                )
            )
            run_latencies = [
                (run.finished_at - run.started_at).total_seconds()
                for run in runs
                if run.started_at is not None and run.finished_at is not None and run.finished_at >= run.started_at
            ]
        else:
            runs = []
        assets_referenced = {row[5] for row in terminal if row[5]}
        return {
            "episodes_ingested": len(import_ids),
            "episodes_seen": len(episodes),
            "episodes_terminal": sum(
                1 for episode in episodes if episode.status in TERMINAL_EPISODE_STATES
            ),
            "episode_status_counts": dict(episode_status_counts),
            "episodes_non_terminal": non_terminal_episodes,
            "pipeline_runs": len(runs),
            "pipeline_runs_succeeded": sum(1 for run in runs if run.state == "SUCCEEDED"),
            "pipeline_runs_failed": sum(1 for run in runs if run.state == "FAILED"),
            "tasks_total": len(rows),
            "task_state_counts": dict(state_counts),
            "tasks_terminal": len(terminal),
            "tasks_non_terminal": non_terminal,
            "permanent_running_tasks": len(permanent_running),
            "permanent_running_detail": permanent_running,
            "task_latency_seconds_p50": _percentile(latencies, 0.50),
            "task_latency_seconds_p95": _percentile(latencies, 0.95),
            "task_latency_seconds_max": round(max(latencies), 6) if latencies else None,
            "pipeline_run_wall_seconds_p50": _percentile(run_latencies, 0.50),
            "pipeline_run_wall_seconds_p95": _percentile(run_latencies, 0.95),
            "duplicate_output_asset_ids": duplicates,
            "duplicate_output_tasks": duplicate_output_tasks,
            "duplicate_asset_keys": [
                {"object_key": key, "sha256": sha} for key, sha in duplicate_asset_keys
            ],
            "referenced_output_assets": len(assets_referenced),
            "ingest_seconds": round(ingest_seconds, 3),
            "processing_seconds": round(processing_seconds, 3),
            "episodes_per_minute": round(len(import_ids) / max(processing_seconds, 1e-9) * 60, 3),
            "ingest_episodes_per_minute": round(len(import_ids) / max(ingest_seconds, 1e-9) * 60, 3),
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def _run_once(
    factory,
    storage,
    video: Path,
    *,
    episodes: int,
    cpu_concurrency: int,
    vision_concurrency: int,
    run_index: int,
    run_label: str,
    timeout_seconds: float,
    principal,
) -> dict:
    started_ingest = time.perf_counter()
    import_ids = _ingest_episodes(factory, storage, video, episodes, run_label, principal)
    ingest_seconds = time.perf_counter() - started_ingest
    print(
        f"[bench_v2] run {run_label}: ingested {len(import_ids)} episodes in {ingest_seconds:.1f}s",
        flush=True,
    )
    processing_seconds, resources = _drain(
        factory,
        storage,
        import_ids,
        cpu_concurrency=cpu_concurrency,
        vision_concurrency=vision_concurrency,
        timeout_seconds=timeout_seconds,
        run_label=run_label,
    )
    metrics = _run_metrics(
        factory, import_ids, ingest_seconds=ingest_seconds, processing_seconds=processing_seconds
    )
    ok = (
        metrics["permanent_running_tasks"] == 0
        and not metrics["tasks_non_terminal"]
        and not metrics["episodes_non_terminal"]
        and not metrics["duplicate_output_asset_ids"]
        and metrics["episodes_seen"] == len(import_ids)
        and metrics["episodes_terminal"] == len(import_ids)
    )
    reasons = []
    if metrics["permanent_running_tasks"]:
        reasons.append(f"{metrics['permanent_running_tasks']} permanent RUNNING task(s)")
    if metrics["tasks_non_terminal"]:
        reasons.append(f"{len(metrics['tasks_non_terminal'])} non-terminal task(s)")
    if metrics["episodes_non_terminal"]:
        reasons.append(f"{len(metrics['episodes_non_terminal'])} non-terminal episode(s)")
    if metrics["duplicate_output_asset_ids"]:
        reasons.append(f"{len(metrics['duplicate_output_asset_ids'])} duplicate output asset(s)")
    if metrics["episodes_seen"] != len(import_ids):
        reasons.append("episode count mismatch after import")
    if metrics["episodes_terminal"] != len(import_ids):
        reasons.append("not every episode reached a terminal status")
    print(
        f"[bench_v2] run {run_label}: {metrics['episodes_per_minute']:.2f} episodes/min, "
        f"tasks={metrics['tasks_total']}, ok={ok}",
        flush=True,
    )
    return {
        "run_index": run_index,
        "run_label": run_label,
        "cpu_concurrency": cpu_concurrency,
        "vision_concurrency": vision_concurrency,
        "episodes_requested": episodes,
        "measured": metrics,
        "resources": resources,
        "ok": ok,
        "failure_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _aggregate(runs: list[dict], concurrency: int) -> dict | None:
    selected = [run for run in runs if run["cpu_concurrency"] == concurrency]
    if not selected:
        return None
    measured = [run["measured"] for run in selected]
    resources = [run["resources"] for run in selected if run["resources"].get("available")]
    return {
        "cpu_concurrency": concurrency,
        "repetitions": len(selected),
        "runs_ok": sum(1 for run in selected if run["ok"]),
        "episodes_per_minute": _mean_std([row["episodes_per_minute"] for row in measured]),
        "processing_seconds": _mean_std([row["processing_seconds"] for row in measured]),
        "ingest_seconds": _mean_std([row["ingest_seconds"] for row in measured]),
        "task_latency_seconds_p50": _mean_std(
            [row["task_latency_seconds_p50"] for row in measured if row["task_latency_seconds_p50"] is not None]
        ),
        "task_latency_seconds_p95": _mean_std(
            [row["task_latency_seconds_p95"] for row in measured if row["task_latency_seconds_p95"] is not None]
        ),
        "permanent_running_tasks": _mean_std([float(row["permanent_running_tasks"]) for row in measured]),
        "duplicate_output_assets": _mean_std([float(len(row["duplicate_output_asset_ids"])) for row in measured]),
        "process_peak_rss_mb": _mean_std(
            [float(resource["process_tree_peak_rss_mb"]) for resource in resources]
        ),
        "cpu_percent_avg": _mean_std([float(resource["cpu_percent_avg"]) for resource in resources if resource.get("cpu_percent_avg") is not None]),
    }


def _ratio(aggregates: dict[int, dict]) -> dict | None:
    low = aggregates.get(1)
    high = aggregates.get(4)
    if not low or not high:
        return None
    low_epm = (low.get("episodes_per_minute") or {}).get("mean")
    high_epm = (high.get("episodes_per_minute") or {}).get("mean")
    if not low_epm or not high_epm:
        return None
    return {
        "metric": "episodes_per_minute",
        "baseline_concurrency": 1,
        "compared_concurrency": 4,
        "baseline_mean": low_epm,
        "compared_mean": high_epm,
        "ratio_4_over_1": round(high_epm / low_epm, 4),
        "basis": "ratio of measured means on the same host, DB, object store and fixture",
        "claim": "no improvement claim: the ratio is reported as measured, and I/O/SQLite-style contention may dominate",
    }


def build_report(
    runs: list[dict],
    *,
    episodes: int,
    repetitions: int,
    concurrencies: list[int],
    vision_concurrency: int,
    quick: bool,
    timeout_seconds: float,
) -> dict:
    materialised: dict[int, dict] = {}
    for concurrency in concurrencies:
        aggregate = _aggregate(runs, concurrency)
        if aggregate is not None:
            materialised[concurrency] = aggregate
    all_ok = all(run["ok"] for run in runs) and len(runs) == len(concurrencies) * repetitions
    failure_reasons = [
        f"{run['run_label']}: {reason}" for run in runs for reason in run["failure_reasons"]
    ]
    return {
        "schema_version": "benchmark-report/2.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "bench_v2 (D2-M6 / DO2-32)",
        "command": " ".join([sys.executable, *sys.argv]),
        "full_run_command": FULL_RUN_COMMAND,
        "quick_mode": quick,
        "environment": {**_environment(), "machine": _machine()},
        "configuration": {
            "episodes_per_run": episodes,
            "repetitions": repetitions,
            "cpu_concurrencies": concurrencies,
            "vision_concurrency": vision_concurrency,
            "total_runs": len(runs),
            "data_source": "deterministic synthetic 10s Ego MP4 fixture (F04_ego_video.mp4), no downloads",
            "episode_type": "ego_video_episode",
            "detector": "fake (FakeDetector); not real YOLO inference",
            "workers": "in-process drain threads using workflow.runner.run_until_idle; one host process",
            "task_timeout_seconds": timeout_seconds,
            "queue": "PostgreSQL state machine + outbox; run_until_idle does not consume Redis",
        },
        "runs": runs,
        "aggregates": {str(key): value for key, value in materialised.items()},
        "ratio": _ratio(materialised),
        "verdict": {
            "ok": all_ok,
            "runs_ok": sum(1 for run in runs if run["ok"]),
            "runs_total": len(runs),
            "failure_reasons": failure_reasons,
        },
        "limitations": [
            "Synthetic 10-second Ego MP4 Episodes and the FakeDetector: this is platform throughput evidence, not real robot production throughput and not real YOLO inference.",
            "Workers are in-process threads in one host process sharing the same Postgres/Redis/local-filesystem object store; this does not measure multi-container or multi-host scaling.",
            "run_until_idle drains from PostgreSQL directly, so the reported throughput excludes Redis Streams dispatch/consumer-group overhead (the V1 benchmarks cover the queue path separately).",
            "GPU utilisation is N/A: CPU-only host.",
            "Task completion latency is measured as (tasks.updated_at - tasks.created_at) for terminal tasks; it includes queue wait and retries and is not a pure handler duration.",
            "No improvement claim is made for concurrency=4; the ratio is computed only from measured means and may be limited by I/O, ffmpeg subprocesses and the shared host.",
            "Full 100-Episode x 3 repetitions x {1,4} run is the acceptance command; the quick mode numbers are smoke evidence only.",
        ],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _preflight(factory) -> list[str]:
    """Check that the migrated schema is present; returns missing table names."""
    from sqlalchemy import inspect

    from embodied_dataops.db.session import get_engine

    try:
        names = set(inspect(get_engine()).get_table_names())
    except Exception:  # noqa: BLE001 - connection error is reported by the caller
        return ["<database unreachable>"]
    required = {"episodes", "tasks", "pipeline_runs", "import_jobs", "assets", "outbox_events"}
    return sorted(required - names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D2-M6 measured benchmark for the V2 platform (DO2-32)"
    )
    parser.add_argument("--episodes", type=int, default=100, help="episodes per run (default 100)")
    parser.add_argument("--repetitions", type=int, default=3, help="repetitions per concurrency (default 3)")
    parser.add_argument(
        "--concurrencies", default="1,4", help="comma separated CPU worker counts (default 1,4)"
    )
    parser.add_argument("--vision-concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-run processing deadline seconds")
    parser.add_argument("--video", default=None, help="synthetic MP4 fixture path")
    parser.add_argument("--report", default=str(REPORT_PATH), help="report JSON path")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke mode: 5 episodes, 1 repetition, concurrency 1,4 (does not overwrite the full-run claim)",
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.episodes = 5
        args.repetitions = 1
    if args.episodes < 1 or args.repetitions < 1:
        parser.error("--episodes and --repetitions must be >= 1")
    concurrencies = sorted({int(item) for item in str(args.concurrencies).split(",") if item.strip()})
    if not concurrencies or any(value < 1 for value in concurrencies):
        parser.error("--concurrencies must be a comma separated list of positive integers")

    try:
        from embodied_dataops.cli.seed import ensure_seed_principal
        from embodied_dataops.config import get_settings
        from embodied_dataops.db.session import get_session_factory
        from embodied_dataops.storage.base import get_storage
    except Exception as exc:  # noqa: BLE001 - dependency/database preflight
        print(f"[bench_v2] FATAL: cannot import the application: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    settings = get_settings()
    settings.ensure_directories()
    factory = get_session_factory()
    missing = _preflight(factory)
    if missing:
        print(
            "[bench_v2] FATAL: database schema is not ready (missing: "
            f"{', '.join(missing)}). Run `DATABASE_URL=... alembic upgrade head` first.",
            file=sys.stderr,
        )
        return 2

    try:
        video = _video_path(args.video)
    except FileNotFoundError as exc:
        print(f"[bench_v2] FATAL: {exc}", file=sys.stderr)
        return 2

    storage = get_storage()
    try:
        storage.ensure_bucket()
    except Exception as exc:  # noqa: BLE001 - object store is a hard dependency
        print(f"[bench_v2] FATAL: object store unavailable: {exc}", file=sys.stderr)
        return 2

    session = factory()
    try:
        principal = ensure_seed_principal(session)
        session.commit()
    finally:
        session.close()

    print(
        f"[bench_v2] episodes={args.episodes} repetitions={args.repetitions} "
        f"concurrencies={concurrencies} detector={settings.detector} "
        f"storage={settings.storage_backend}",
        flush=True,
    )
    runs: list[dict] = []
    for concurrency in concurrencies:
        for repetition in range(1, args.repetitions + 1):
            label = f"cpu{concurrency}-rep{repetition}"
            try:
                run = _run_once(
                    factory,
                    storage,
                    video,
                    episodes=args.episodes,
                    cpu_concurrency=concurrency,
                    vision_concurrency=args.vision_concurrency,
                    run_index=len(runs) + 1,
                    run_label=label,
                    timeout_seconds=args.timeout,
                    principal=principal,
                )
            except Exception as exc:  # noqa: BLE001 - one bad run must be reported, not hidden
                print(f"[bench_v2] run {label} crashed: {exc}", file=sys.stderr)
                traceback.print_exc()
                run = {
                    "run_index": len(runs) + 1,
                    "run_label": label,
                    "cpu_concurrency": concurrency,
                    "vision_concurrency": args.vision_concurrency,
                    "episodes_requested": args.episodes,
                    "measured": {},
                    "resources": {"available": False, "reason": f"run crashed: {exc}"},
                    "ok": False,
                    "failure_reasons": [f"run crashed: {exc.__class__.__name__}: {exc}"],
                }
            runs.append(run)

    report = build_report(
        runs,
        episodes=args.episodes,
        repetitions=args.repetitions,
        concurrencies=concurrencies,
        vision_concurrency=args.vision_concurrency,
        quick=args.quick,
        timeout_seconds=args.timeout,
    )
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["verdict"], indent=2))
    print(f"[bench_v2] report -> {report_path}")
    return 0 if report["verdict"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
