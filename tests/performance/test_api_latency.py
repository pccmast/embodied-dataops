"""Performance budgets from SPEC 12.3 (local, same-host, honest reporting).

* ``GET /api/v1/episodes`` with 10,000 metadata rows: P95 < 300 ms
  (≥500 requests, 10 concurrent, no image generation).
* pipeline-run creation: P95 < 400 ms (≥200 requests, excluding processing).

The measured values, machine information and command are written to
``artifacts/benchmarks/api-latency.json``.  A missed budget fails the test and is
reported as a missed acceptance item — the thresholds are never relaxed.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import insert

from embodied_dataops.db.models import Episode, SourceImport
from embodied_dataops.ids import new_id

pytestmark = pytest.mark.performance

REPORT = Path("artifacts/benchmarks/api-latency.json")
EPISODE_ROWS = 10_000


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _machine() -> dict:
    import os

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "database": "PostgreSQL 16 on localhost",
    }


@pytest.fixture()
def seeded_episodes(engine, session_factory):
    """Insert 10,000 episode rows with one source import (metadata only).

    Function-scoped on purpose: ``tests/conftest.py`` truncates every business
    table before each test, so a module-scoped seed is wiped before the first
    measurement and the benchmark silently measured an empty table.
    """
    with session_factory() as session:
        import_id = new_id()
        session.add(
            SourceImport(
                id=import_id,
                source_type="EGO_MP4",
                upload_state="READY",
                data_origin="synthetic",
                license="synthetic benchmark metadata",
                filename="benchmark.mp4",
                size_bytes=1,
            )
        )
        session.flush()
        rows = [
            {
                "id": str(uuid.uuid4()),
                "source_import_id": import_id,
                "source_episode_index": index,
                "episode_type": "ego_video_episode",
                "task_label": f"benchmark episode {index}",
                "duration_ns": 10_000_000_000,
                "status": "READY_FOR_REVIEW",
                "canonical_version": 1,
                "quality_status": "PASS",
                "sample_count": 100,
                "data_origin": "synthetic",
            }
            for index in range(EPISODE_ROWS)
        ]
        session.execute(insert(Episode), rows)
        session.commit()
    yield


@pytest.mark.timeout(900)
def test_episodes_list_p95(client, operator_headers, seeded_episodes):
    total_requests = 500

    def one(_: int) -> float:
        started = time.perf_counter()
        response = client.get("/api/v1/episodes?limit=20", headers=operator_headers)
        elapsed = (time.perf_counter() - started) * 1000
        assert response.status_code == 200, response.text
        assert len(response.json()["data"]) == 20
        return elapsed

    with ThreadPoolExecutor(max_workers=10) as pool:
        timings = list(pool.map(one, range(total_requests)))

    p95 = _percentile(timings, 0.95)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "machine": _machine(),
        "measurement": "GET /api/v1/episodes?limit=20",
        "dataset_rows": EPISODE_ROWS,
        "requests": total_requests,
        "concurrency": 10,
        "p50_ms": round(statistics.median(timings), 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(max(timings), 2),
        "budget_ms": 300,
        "passed": p95 < 300,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert p95 < 300, f"GET /episodes P95 {p95:.1f} ms exceeds the 300 ms budget: {report}"


@pytest.mark.timeout(900)
def test_create_pipeline_run_p95(client, operator_headers, seeded_episodes):
    """DO-A02 C3: every sample must GET the strong ETag and POST If-Match + Idempotency-Key."""
    episodes = client.get("/api/v1/episodes?limit=100", headers=operator_headers).json()["data"]
    assert episodes
    ids = [episode["id"] for episode in episodes]
    total_requests = 200
    timings: list[float] = []
    for index in range(total_requests):
        episode_id = ids[index % len(ids)]
        detail = client.get(f"/api/v1/episodes/{episode_id}", headers=operator_headers)
        assert detail.status_code == 200, detail.text
        etag = detail.headers["etag"]
        started = time.perf_counter()
        response = client.post(
            f"/api/v1/episodes/{episode_id}/pipeline-runs",
            json={"pipeline_version": "2.0.0"},
            headers={
                **operator_headers,
                "If-Match": etag,
                "Idempotency-Key": str(uuid.uuid4()),
            },
        )
        elapsed = (time.perf_counter() - started) * 1000
        assert response.status_code == 202, response.text
        timings.append(elapsed)

    p95 = _percentile(timings, 0.95)
    payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else {}
    payload["pipeline_create"] = {
        "measurement": "POST /api/v1/episodes/{id}/pipeline-runs (DO-A02 If-Match + Idempotency-Key)",
        "requests": total_requests,
        "p50_ms": round(statistics.median(timings), 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(max(timings), 2),
        "budget_ms": 400,
        "passed": p95 < 400,
    }
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert p95 < 400, f"pipeline-run creation P95 {p95:.1f} ms exceeds the 400 ms budget"
