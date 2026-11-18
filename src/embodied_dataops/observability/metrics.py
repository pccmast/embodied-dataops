"""Prometheus metrics (SPEC 12.2)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

TASKS_TOTAL = Counter(
    "dataops_tasks_total", "Task state transitions", labelnames=["step", "state"]
)
TASK_DURATION = Histogram(
    "dataops_task_duration_seconds",
    "Task execution duration",
    labelnames=["step"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
QUEUE_WAIT = Histogram(
    "dataops_queue_wait_seconds",
    "Time from task creation to lease acquisition",
    labelnames=["step"],
    buckets=(0.05, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
ACTIVE_WORKERS = Gauge("dataops_active_workers", "Worker processes currently serving a pool", ["pool"])
LEASE_EXPIRATIONS = Counter(
    "dataops_lease_expirations_total", "Expired leases reclaimed by the reaper"
)
RETRIES = Counter("dataops_retries_total", "Task retries", labelnames=["reason"])
EPISODE_QC = Counter(
    "dataops_episode_qc_total", "Quality verdicts", labelnames=["verdict", "source_type"]
)
SYNC_ERROR = Histogram(
    "dataops_sync_error_ms",
    "Maximum absolute sync error per episode",
    labelnames=["source_type"],
    buckets=(1, 5, 10, 25, 50, 100, 200, 500, 1000, 5000),
)
IMPORT_BYTES = Counter(
    "dataops_import_bytes_total", "Bytes imported", labelnames=["source_type"]
)
ANNOTATION_FRAMES = Counter(
    "dataops_annotation_frames_total", "Frames annotated", labelnames=["model"]
)
OUTBOX_PUBLISH_FAILURES = Counter(
    "dataops_outbox_publish_failures_total", "Outbox publish attempts that failed"
)


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
