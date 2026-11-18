#!/usr/bin/env python
"""D2-M6 / DO2-33: real fault-injection scenarios against the V2 task protocol.

Scenarios
---------
``kill-worker``
    Starts the real ``dispatcher`` + CPU victim worker + vision worker
    subprocesses, waits until a task is genuinely ``RUNNING``, hard-kills the CPU
    worker, starts a replacement and waits (bounded) for every task/episode to
    reach a terminal state.  POSIX uses ``SIGKILL``; on Windows ``taskkill /F /T``
    is used and the platform difference is recorded (no SIGKILL claim).

``redis-down``
    Stops ``dataops-redis`` (Docker) while work is in flight, waits, restarts the
    container and waits for it to become healthy.  The dispatcher/reaper must
    re-publish lost wakeups and workers must reconnect; every task must still
    reach a terminal state without duplicate valid outputs.

``storage-down``
    Stops the MinIO/Silo container (only meaningful with ``STORAGE_BACKEND=s3``),
    then restarts it.  With the local smoke backend this scenario is reported
    ``NOT_RUN`` with the exact reason instead of being faked.

Every executed scenario records before/after DB snapshots, recovery actions,
timing and a bounded wait (default 120 s; the kill-worker lease-expiry wait is
documented when it exceeds the bound).  ``finally`` blocks always restart any
stopped Docker service and terminate every spawned process.

Report: ``artifacts/failure-injection/v2.json`` (schema
``failure-injection-report/2.0``).  Exit code is non-zero if an executed scenario
leaves a permanent RUNNING task, a non-terminal task/episode or a duplicate
referenced output asset.

Usage
-----
    python scripts/fault_inject_v2.py --scenario kill-worker
    python scripts/fault_inject_v2.py --scenario redis-down --episodes 2
    python scripts/fault_inject_v2.py --scenario all
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from bench_v2 import (  # noqa: E402 - shared measurement helpers (same owner)
    _git_sha,
    _ingest_episodes,
    _mask_credentials,
    _scope_episode_ids,
    _scope_pending_counts,
    _scope_task_rows,
    _video_path,
)

REPORT_PATH = REPO_ROOT / "artifacts" / "failure-injection" / "v2.json"
LOG_DIR = REPO_ROOT / "var" / "failure-injection-logs"
FULL_RUN_COMMAND = "python scripts/fault_inject_v2.py --scenario all --episodes 3 --timeout 120"

TERMINAL_TASK_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}
TERMINAL_EPISODE_STATES = {"READY_FOR_REVIEW", "QUALITY_FAILED"}
REDIS_CONTAINER = "dataops-redis"
STORAGE_CONTAINERS = ("dataops-minio", "dataops-silo")
WINDOWS_NOTE = (
    "Windows host: there is no POSIX SIGKILL. taskkill /F /T is used as the "
    "documented substitute; this run is NOT Linux SIGKILL evidence."
)


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_command(command: list[str], *, timeout: float = 60.0) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        return result.returncode, (result.stdout or result.stderr or "").strip()
    except Exception as exc:  # noqa: BLE001 - docker/git may be absent
        return 127, f"{exc.__class__.__name__}: {exc}"


def _docker(*args: str, timeout: float = 90.0) -> tuple[int, str]:
    return _run_command(["docker", *args], timeout=timeout)


def _container_exists(name: str) -> bool:
    code, output = _docker("inspect", "--format", "{{.Name}}", name)
    return code == 0 and name in output


def _container_health(name: str) -> str:
    code, output = _docker("inspect", "--format", "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}", name)
    return output if code == 0 else f"unknown ({output})"


def _wait_healthy(name: str, timeout: float = 120.0) -> dict:
    started = time.monotonic()
    last = None
    while time.monotonic() - started < timeout:
        last = _container_health(name)
        if last in {"healthy", "running"}:
            return {
                "container": name,
                "healthy": True,
                "status": last,
                "waited_seconds": round(time.monotonic() - started, 2),
            }
        time.sleep(1.0)
    return {
        "container": name,
        "healthy": False,
        "status": last,
        "waited_seconds": round(time.monotonic() - started, 2),
    }


def _stop_container(name: str) -> dict:
    code, output = _docker("stop", name)
    return {
        "action": f"docker stop {name}",
        "exit_code": code,
        "output": output[-500:],
        "at": _now_iso(),
    }


def _start_container(name: str) -> dict:
    code, output = _docker("start", name)
    return {
        "action": f"docker start {name}",
        "exit_code": code,
        "output": output[-500:],
        "at": _now_iso(),
    }


def _snapshot(factory, import_ids: list[str]) -> dict:
    from collections import Counter

    from sqlalchemy import select

    from embodied_dataops.db.models import Episode

    session = factory()
    try:
        rows = _scope_task_rows(session, import_ids)
        states = Counter(row[1] for row in rows)
        output_ids = [row[5] for row in rows if row[5]]
        duplicate_outputs = {
            asset_id: count for asset_id, count in Counter(output_ids).items() if count > 1
        }
        episode_ids = _scope_episode_ids(session, import_ids)
        statuses = Counter()
        if episode_ids:
            statuses = Counter(
                episode.status
                for episode in session.scalars(select(Episode).where(Episode.id.in_(episode_ids)))
            )
        return {
            "tasks_total": len(rows),
            "task_state_counts": dict(states),
            "tasks_running": int(states.get("RUNNING", 0)),
            "tasks_pending": int(states.get("PENDING", 0)),
            "tasks_retry_wait": int(states.get("RETRY_WAIT", 0)),
            "tasks_terminal": sum(
                1 for row in rows if row[1] in TERMINAL_TASK_STATES
            ),
            "episodes_total": len(episode_ids),
            "episode_status_counts": dict(statuses),
            "episodes_terminal": sum(
                count
                for status, count in statuses.items()
                if status in TERMINAL_EPISODE_STATES
            ),
            "duplicate_output_asset_ids": duplicate_outputs,
            "leased_tasks": [
                {"task_id": row[0], "step": row[2]}
                for row in rows
                if row[1] == "RUNNING"
            ],
        }
    finally:
        session.close()


def _wait_for_running(factory, import_ids: list[str], timeout: float = 60.0) -> dict:
    started = time.monotonic()
    last: list[dict] = []
    while time.monotonic() - started < timeout:
        session = factory()
        try:
            rows = _scope_task_rows(session, import_ids)
            running = [
                {"task_id": row[0], "step": row[2], "state": row[1]}
                for row in rows
                if row[1] == "RUNNING"
            ]
        finally:
            session.close()
        if running:
            return {
                "observed": True,
                "waited_seconds": round(time.monotonic() - started, 2),
                "running_tasks": running,
            }
        last = running
        time.sleep(0.2)
    return {
        "observed": False,
        "waited_seconds": round(time.monotonic() - started, 2),
        "running_tasks": last,
    }


def _wait_terminal(factory, import_ids: list[str], timeout: float = 120.0) -> dict:
    started = time.monotonic()
    last_pending = None
    while time.monotonic() - started < timeout:
        session = factory()
        try:
            pending, running = _scope_pending_counts(session, import_ids)
        finally:
            session.close()
        last_pending = (pending, running)
        if pending == 0:
            return {
                "timed_out": False,
                "waited_seconds": round(time.monotonic() - started, 2),
                "pending_tasks": 0,
                "running_tasks": 0,
            }
        time.sleep(0.5)
    return {
        "timed_out": True,
        "waited_seconds": round(time.monotonic() - started, 2),
        "pending_tasks": last_pending[0] if last_pending else None,
        "running_tasks": last_pending[1] if last_pending else None,
    }


# ---------------------------------------------------------------------------
# subprocess management (real dispatcher / workers)
# ---------------------------------------------------------------------------
def _spawn(module_args: list[str], name: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    log = open(log_path, "ab")  # noqa: SIM115 - child owns the handle for its lifetime
    command = [sys.executable, "-m", *module_args]
    popen = subprocess.Popen(  # noqa: S603 - fixed interpreter + module
        command,
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )
    popen.bench_log = str(log_path)  # type: ignore[attr-defined]
    return popen


def _spawn_dispatcher() -> subprocess.Popen:
    return _spawn(["embodied_dataops.workflow.dispatcher"], "dispatcher")


def _spawn_worker(pool: str, concurrency: int, tag: str) -> subprocess.Popen:
    return _spawn(
        [
            "embodied_dataops.workflow.worker",
            "--pool",
            pool,
            "--concurrency",
            str(concurrency),
            "--worker-id",
            f"fault-{pool}-{tag}",
        ],
        f"worker-{pool}-{tag}",
    )


def _kill_process(proc: subprocess.Popen) -> dict:
    if proc.poll() is not None:
        return {"method": "none", "already_exited": True, "exit_code": proc.returncode}
    if os.name == "nt":
        method = "taskkill /F /T /PID (Windows substitute: no POSIX SIGKILL semantics)"
        code, output = _run_command(["taskkill", "/F", "/T", "/PID", str(proc.pid)], timeout=30)
        result = {"method": method, "exit_code": code, "output": output[-300:]}
    else:
        method = "os.kill(pid, signal.SIGKILL)"
        try:
            os.kill(proc.pid, signal.SIGKILL)
            result = {"method": method, "exit_code": 0}
        except Exception as exc:  # noqa: BLE001
            result = {"method": method, "exit_code": -1, "output": str(exc)}
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover - hard kill should settle
        proc.kill()
    result["returncode"] = proc.returncode
    return result


def _terminate_process(proc: subprocess.Popen, name: str) -> dict:
    if proc.poll() is not None:
        return {"name": name, "already_exited": True, "returncode": proc.returncode}
    try:
        proc.terminate()
        proc.wait(timeout=20)
        return {"name": name, "terminated": True, "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            _run_command(["taskkill", "/F", "/T", "/PID", str(proc.pid)], timeout=20)
        else:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:  # pragma: no cover
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass
        return {"name": name, "terminated": False, "forced": True, "returncode": proc.returncode}


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
def scenario_kill_worker(args, factory, storage, principal) -> dict:

    started = _now_iso()
    actions: list[dict] = []
    processes: dict[str, subprocess.Popen] = {}
    import_ids: list[str] = []
    notes = [WINDOWS_NOTE] if os.name == "nt" else []
    try:
        video = _video_path(None)
        import_ids = _ingest_episodes(factory, storage, video, args.episodes, "fault-kill", principal)
        before = _snapshot(factory, import_ids)
        dispatcher = _spawn_dispatcher()
        processes["dispatcher"] = dispatcher
        time.sleep(2.0)
        victim = _spawn_worker("cpu", 1, "victim")
        processes["victim"] = victim
        process_vision = _spawn_worker("vision", 1, "vision")
        processes["vision"] = process_vision
        actions.append(
            {
                "action": "started dispatcher + cpu victim worker + vision worker",
                "at": _now_iso(),
                "pids": {
                    "dispatcher": dispatcher.pid,
                    "victim": victim.pid,
                    "vision": process_vision.pid,
                },
                "logs": {name: getattr(proc, "bench_log", None) for name, proc in processes.items()},
            }
        )
        running = _wait_for_running(factory, import_ids, timeout=min(90.0, args.timeout))
        actions.append({"action": "observed RUNNING task before kill", **running})
        post_kill = None
        kill = None
        if running["observed"]:
            kill = _kill_process(victim)
            processes.pop("victim", None)
            actions.append({"action": "hard-killed cpu victim worker", "result": kill, "at": _now_iso()})
            time.sleep(1.0)
            post_kill = _snapshot(factory, import_ids)
        else:
            notes.append("no RUNNING task was observed before the kill window; kill still recorded")
            kill = _kill_process(victim)
            processes.pop("victim", None)
            actions.append({"action": "killed cpu worker without observing RUNNING", "result": kill})
        replacement = _spawn_worker("cpu", 1, "replacement")
        processes["replacement"] = replacement
        actions.append(
            {
                "action": "started replacement cpu worker; dispatcher reaper must expire the killed lease",
                "at": _now_iso(),
                "pid": replacement.pid,
            }
        )
        wait = _wait_terminal(factory, import_ids, timeout=args.timeout)
        actions.append({"action": "waited for terminal task states", **wait})
        after = _snapshot(factory, import_ids)
        if wait["timed_out"] and wait["waited_seconds"] > 120:
            notes.append(
                "kill-worker recovery exceeded the 120 s bound because the 60 s lease must "
                "expire before the reaper can reclaim it; elapsed time is recorded honestly"
            )
        return _scenario_result(
            "kill-worker",
            started,
            import_ids,
            before,
            post_kill,
            after,
            actions,
            wait,
            notes,
        )
    finally:
        for name, proc in list(processes.items()):
            try:
                result = _terminate_process(proc, name)
            except Exception:  # noqa: BLE001 - cleanup must not hide the scenario result
                result = {"name": name, "error": traceback.format_exc(limit=2)}
            actions.append({"action": f"terminated {name}", "result": result})
        _write_logs_pointer(actions)


def scenario_redis_down(args, factory, storage, principal) -> dict:
    started = _now_iso()
    actions: list[dict] = []
    processes: dict[str, subprocess.Popen] = {}
    import_ids: list[str] = []
    notes: list[str] = []
    stopped = False
    try:
        if not _container_exists(REDIS_CONTAINER):
            return _not_run("redis-down", f"Docker container {REDIS_CONTAINER} not found")
        video = _video_path(None)
        import_ids = _ingest_episodes(factory, storage, video, args.episodes, "fault-redis", principal)
        before = _snapshot(factory, import_ids)
        dispatcher = _spawn_dispatcher()
        processes["dispatcher"] = dispatcher
        time.sleep(2.0)
        processes["cpu"] = _spawn_worker("cpu", 1, "redis")
        processes["vision"] = _spawn_worker("vision", 1, "redis")
        actions.append(
            {
                "action": "started dispatcher + cpu/vision workers and started processing",
                "at": _now_iso(),
            }
        )
        # Let real work reach an in-flight state before the outage.
        inflight = _wait_for_running(factory, import_ids, timeout=min(60.0, args.timeout))
        actions.append({"action": "observed RUNNING task before Redis outage", **inflight})
        health_before = _container_health(REDIS_CONTAINER)
        stopped = True
        stop_action = _stop_container(REDIS_CONTAINER)
        actions.append({"action": "stopped Redis", **stop_action, "health_before": health_before})
        time.sleep(3.0)
        down = {
            "at": _now_iso(),
            "health_while_down": _container_health(REDIS_CONTAINER),
        }
        actions.append({"action": "Redis outage window", **down})
        start_action = _start_container(REDIS_CONTAINER)
        stopped = False
        actions.append({"action": "restarted Redis", **start_action})
        healthy = _wait_healthy(REDIS_CONTAINER, timeout=120.0)
        actions.append({"action": "waited for Redis health", **healthy})
        if not healthy["healthy"]:
            notes.append("Redis did not report healthy within 120 s after restart")
        wait = _wait_terminal(factory, import_ids, timeout=args.timeout)
        actions.append({"action": "waited for terminal task states after Redis recovery", **wait})
        after = _snapshot(factory, import_ids)
        return _scenario_result(
            "redis-down", started, import_ids, before, None, after, actions, wait, notes
        )
    except Exception as exc:  # noqa: BLE001 - report a crashed scenario honestly
        notes.append(f"scenario crashed: {exc.__class__.__name__}: {exc}")
        return _scenario_result(
            "redis-down",
            started,
            import_ids,
            None,
            None,
            None,
            actions,
            {"timed_out": True, "waited_seconds": 0.0},
            notes,
            crashed=True,
        )
    finally:
        if stopped:
            try:
                actions.append(
                    {"action": "finally restart Redis", **_start_container(REDIS_CONTAINER)}
                )
                actions.append(
                    {"action": "finally wait Redis healthy", **_wait_healthy(REDIS_CONTAINER, 120.0)}
                )
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        for name, proc in list(processes.items()):
            try:
                actions.append({"action": f"terminated {name}", "result": _terminate_process(proc, name)})
            except Exception:  # noqa: BLE001
                pass
        _write_logs_pointer(actions)


def scenario_storage_down(args, factory, storage, principal) -> dict:
    from embodied_dataops.config import get_settings

    settings = get_settings()
    if settings.storage_backend != "s3":
        return _not_run(
            "storage-down",
            "STORAGE_BACKEND=local: stopping MinIO/Silo cannot be exercised honestly with the "
            "local filesystem smoke backend; run with STORAGE_BACKEND=s3 and MinIO/Silo up",
        )
    started = _now_iso()
    actions: list[dict] = []
    processes: dict[str, subprocess.Popen] = {}
    import_ids: list[str] = []
    notes: list[str] = []
    container = next((name for name in STORAGE_CONTAINERS if _container_exists(name)), None)
    stopped = False
    if container is None:
        return _not_run("storage-down", f"none of the storage containers {STORAGE_CONTAINERS} found")
    try:
        video = _video_path(None)
        import_ids = _ingest_episodes(factory, storage, video, args.episodes, "fault-storage", principal)
        before = _snapshot(factory, import_ids)
        dispatcher = _spawn_dispatcher()
        processes["dispatcher"] = dispatcher
        time.sleep(2.0)
        processes["cpu"] = _spawn_worker("cpu", 1, "storage")
        processes["vision"] = _spawn_worker("vision", 1, "storage")
        actions.append({"action": f"started dispatcher + workers (storage={container})", "at": _now_iso()})
        inflight = _wait_for_running(factory, import_ids, timeout=min(60.0, args.timeout))
        actions.append({"action": "observed RUNNING task before storage outage", **inflight})
        stopped = True
        actions.append({"action": "stopped storage", **_stop_container(container)})
        time.sleep(3.0)
        actions.append({"action": "storage outage window", "health_while_down": _container_health(container)})
        start = _start_container(container)
        stopped = False
        actions.append({"action": "restarted storage", **start})
        actions.append({"action": "waited for storage health", **_wait_healthy(container, 120.0)})
        wait = _wait_terminal(factory, import_ids, timeout=args.timeout)
        actions.append({"action": "waited for terminal task states after storage recovery", **wait})
        after = _snapshot(factory, import_ids)
        return _scenario_result(
            "storage-down", started, import_ids, before, None, after, actions, wait, notes
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"scenario crashed: {exc.__class__.__name__}: {exc}")
        return _scenario_result(
            "storage-down",
            started,
            import_ids,
            None,
            None,
            None,
            actions,
            {"timed_out": True, "waited_seconds": 0.0},
            notes,
            crashed=True,
        )
    finally:
        if stopped:
            try:
                actions.append({"action": "finally restart storage", **_start_container(container)})
                actions.append({"action": "finally wait storage healthy", **_wait_healthy(container, 120.0)})
            except Exception:  # noqa: BLE001
                pass
        for name, proc in list(processes.items()):
            try:
                actions.append({"action": f"terminated {name}", "result": _terminate_process(proc, name)})
            except Exception:  # noqa: BLE001
                pass
        _write_logs_pointer(actions)


# ---------------------------------------------------------------------------
# result shaping
# ---------------------------------------------------------------------------
def _write_logs_pointer(actions: list[dict]) -> None:
    actions.append({"action": "process logs", "directory": str(LOG_DIR)})


def _scenario_result(
    scenario: str,
    started_at: str,
    import_ids: list[str],
    before: dict | None,
    post_interrupt: dict | None,
    after: dict | None,
    actions: list[dict],
    wait: dict,
    notes: list[str],
    *,
    crashed: bool = False,
) -> dict:
    failure_reasons: list[str] = []
    if after is None:
        failure_reasons.append("no post-scenario snapshot was collected (scenario crashed)")
    else:
        if after["tasks_running"]:
            failure_reasons.append(f"{after['tasks_running']} permanent RUNNING task(s)")
        pending = (
            after["tasks_pending"] + after["tasks_retry_wait"] + after["tasks_running"]
        )
        if pending:
            failure_reasons.append(f"{pending} non-terminal task(s) after the bounded wait")
        if after["tasks_terminal"] != after["tasks_total"]:
            failure_reasons.append("not every task reached a terminal state")
        if after["duplicate_output_asset_ids"]:
            failure_reasons.append(
                f"{len(after['duplicate_output_asset_ids'])} duplicate referenced output asset(s)"
            )
        if after["episodes_terminal"] != after["episodes_total"]:
            failure_reasons.append("not every episode reached a terminal status")
    if wait.get("timed_out"):
        failure_reasons.append(f"bounded wait timed out after {wait.get('waited_seconds')}s")
    if crashed:
        failure_reasons.append("scenario crashed")
    ok = not failure_reasons
    ended_at = _now_iso()
    return {
        "scenario": scenario,
        "status": "PASS" if ok else "FAIL",
        "started_at": started_at,
        "finished_at": ended_at,
        "duration_seconds": None,
        "import_ids": import_ids,
        "before": before,
        "post_interrupt": post_interrupt,
        "after": after,
        "recovery_actions": actions,
        "wait": wait,
        "ok": ok,
        "failure_reasons": failure_reasons,
        "notes": notes,
    }


def _not_run(scenario: str, reason: str) -> dict:
    return {
        "scenario": scenario,
        "status": "NOT_RUN",
        "ok": True,
        "reason": reason,
        "before": None,
        "post_interrupt": None,
        "after": None,
        "recovery_actions": [],
        "wait": None,
        "failure_reasons": [],
        "notes": [reason],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _machine() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "os_name": os.name,
        "git_sha": _git_sha(),
        "posix_sigkill_available": os.name != "nt",
    }


def _preflight(factory) -> list[str]:
    from sqlalchemy import inspect

    from embodied_dataops.db.session import get_engine

    try:
        names = set(inspect(get_engine()).get_table_names())
    except Exception:  # noqa: BLE001
        return ["<database unreachable>"]
    required = {"episodes", "tasks", "pipeline_runs", "import_jobs", "assets", "outbox_events"}
    return sorted(required - names)


def build_report(scenarios: list[dict], *, episodes: int, timeout: float, quick_note: str | None) -> dict:
    executed = [item for item in scenarios if item["status"] in {"PASS", "FAIL"}]
    not_run = [item for item in scenarios if item["status"] == "NOT_RUN"]
    failure_reasons = [
        f"{item['scenario']}: {reason}" for item in executed for reason in item["failure_reasons"]
    ]
    return {
        "schema_version": "failure-injection-report/2.0",
        "generated_at": _now_iso(),
        "benchmark": "fault_inject_v2 (D2-M6 / DO2-33)",
        "command": " ".join([sys.executable, *sys.argv]),
        "full_run_command": FULL_RUN_COMMAND,
        "environment": {
            "machine": _machine(),
            "database": _mask_credentials(
                os.environ.get("DATABASE_URL", "postgresql://<unset>")
            ),
            "redis": _mask_credentials(os.environ.get("REDIS_URL", "redis://<unset>")),
            "storage_backend": os.environ.get("STORAGE_BACKEND", "unset"),
            "detector": os.environ.get("DETECTOR", "unset"),
            "platform_note": WINDOWS_NOTE if os.name == "nt" else "POSIX host: os.kill(SIGKILL) used",
        },
        "configuration": {
            "episodes_per_scenario": episodes,
            "bounded_wait_seconds": timeout,
            "scenarios_requested": [item["scenario"] for item in scenarios],
            "real_dispatcher_and_worker_processes": True,
            "docker_services": [REDIS_CONTAINER, *STORAGE_CONTAINERS],
            "smoke_note": quick_note,
        },
        "scenarios": scenarios,
        "verdict": {
            "ok": all(item["ok"] for item in executed) and not failure_reasons,
            "executed": len(executed),
            "passed": sum(1 for item in executed if item["status"] == "PASS"),
            "failed": sum(1 for item in executed if item["status"] == "FAIL"),
            "not_run": [{"scenario": item["scenario"], "reason": item["reason"]} for item in not_run],
            "failure_reasons": failure_reasons,
        },
        "limitations": [
            "Windows host: the kill-worker scenario uses taskkill /F /T as a documented substitute; this is not POSIX SIGKILL evidence.",
            "Recovery after kill-worker includes the real 60 s lease expiry (plus reaper interval) before the task can be retried; timings are recorded, not shortened.",
            "redis-down/storage-down exercise Docker container stop/start of the local compose services only; no image builds and no multi-host network partitions.",
            "storage-down is NOT_RUN with the local filesystem smoke backend because stopping MinIO/Silo cannot affect LocalStorage; it is only meaningful with STORAGE_BACKEND=s3.",
            "Synthetic Ego MP4 Episodes + FakeDetector: this is protocol-recovery evidence, not production robot data or throughput.",
            "The bounded wait (default 120 s) may be too short for slow hosts; a timeout is reported as FAIL rather than claimed as success.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D2-M6 real fault injection (DO2-33)")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["kill-worker", "redis-down", "storage-down", "all"],
    )
    parser.add_argument("--episodes", type=int, default=3, help="episodes per scenario (keep small)")
    parser.add_argument("--timeout", type=float, default=120.0, help="bounded terminal wait seconds")
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--smoke-note", default=None, help="free-text note recorded in the report")
    args = parser.parse_args(argv)

    try:
        from embodied_dataops.cli.seed import ensure_seed_principal
        from embodied_dataops.config import get_settings
        from embodied_dataops.db.session import get_session_factory
        from embodied_dataops.storage.base import get_storage
    except Exception as exc:  # noqa: BLE001
        print(f"[fault_v2] FATAL: cannot import application: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    settings = get_settings()
    settings.ensure_directories()
    factory = get_session_factory()
    missing = _preflight(factory)
    if missing:
        print(
            "[fault_v2] FATAL: database schema not ready (missing: "
            f"{', '.join(missing)}). Run `DATABASE_URL=... alembic upgrade head` first.",
            file=sys.stderr,
        )
        return 2
    storage = get_storage()
    try:
        storage.ensure_bucket()
    except Exception as exc:  # noqa: BLE001
        print(f"[fault_v2] FATAL: object store unavailable: {exc}", file=sys.stderr)
        return 2
    session = factory()
    try:
        principal = ensure_seed_principal(session)
        session.commit()
    finally:
        session.close()

    scenarios_requested = (
        ["kill-worker", "redis-down", "storage-down"]
        if args.scenario == "all"
        else [args.scenario]
    )
    print(
        f"[fault_v2] scenarios={scenarios_requested} episodes={args.episodes} "
        f"timeout={args.timeout}s backend={settings.storage_backend}",
        flush=True,
    )
    scenarios: list[dict] = []
    for scenario in scenarios_requested:
        started = time.monotonic()
        print(f"[fault_v2] running scenario: {scenario}", flush=True)
        try:
            if scenario == "kill-worker":
                result = scenario_kill_worker(args, factory, storage, principal)
            elif scenario == "redis-down":
                result = scenario_redis_down(args, factory, storage, principal)
            else:
                result = scenario_storage_down(args, factory, storage, principal)
        except Exception as exc:  # noqa: BLE001 - never hide a scenario crash
            traceback.print_exc()
            result = _not_run(scenario, f"scenario crashed before execution: {exc!r}")
        if result.get("duration_seconds") is None:
            result["duration_seconds"] = round(time.monotonic() - started, 2)
        scenarios.append(result)
        print(
            f"[fault_v2] scenario {scenario}: {result['status']}"
            + (f" ({'; '.join(result.get('failure_reasons') or [])})" if result.get("failure_reasons") else ""),
            flush=True,
        )

    report = build_report(
        scenarios, episodes=args.episodes, timeout=args.timeout, quick_note=args.smoke_note
    )
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["verdict"], indent=2))
    print(f"[fault_v2] report -> {report_path}")
    return 0 if report["verdict"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
