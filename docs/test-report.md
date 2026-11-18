# V1 测试与验收报告（最终）

Generated: 2026-10-30 (Asia/Shanghai)
Environment: Windows 11 host-mode (Python 3.12.13 local runtime; delivery images pin Python 3.11),
PostgreSQL 16.10 / Redis 7.4.5 / PGSTY Silo containers, local object store; project container
images are **not built on Windows** (ADR-0011).

## 1. Full suite

```bash
TEST_DATABASE_URL=...dataops_test_accept_v2 \
TEST_REDIS_URL=redis://localhost:6379/10 STORAGE_BACKEND=local DETECTOR=fake \
uv run python -m embodied_dataops.cli.main acceptance
```

Result: **604 total / 602 passed / 0 failed / 2 skipped / 2 deselected**, 14m34s (final run after DO2-08/13/16/17).
Skipped with explicit reasons (environment-dependent, not failures):

- `test_mcap_import.py`: `rosbag2_py` is only available inside the ROS 2 Jazzy worker image;
- `test_detection.py::...`: ultralytics installed; real loading covered by the vision marker test;
- `test_lerobot_v2_roundtrip.py`: D2-M1 official v2 export is not implemented (V2 gap DO2-08).

Independent run without acceptance wrapper: `pytest -m "not performance and not ros2 and not vision"`
→ **589 passed, 2 skipped, 5 deselected** (10m56s). Evidence: `artifacts/test-results/v1-full-junit.xml`.

## 2. AT matrix (`artifacts/test-results/at-matrix.json`)

| AT | status | note |
|---|---|---|
| AT-01 | DONE (host-mode) | `docker compose` core services are not built on Windows; evidence is host services + `/health/ready` (container `make up` is DEFERRED_LINUX per ADR-0011) |
| AT-02..AT-15 | DONE | import/normalize/MCAP/quality/task protocol/lease fencing/outbox/backoff |
| AT-16 | NOT_RUN | no automated binding in the acceptance map; annotation-review behavior is covered by integration/RBAC tests but the matrix stays honest |
| AT-17..AT-21 | DONE | official LeRobot Writer→Reader round-trip, immutability, gap analysis, RBAC/audit, real YOLO11n |
| AT-22 | NOT_RUN | 100-episode benchmark (`make benchmark`) not executed in this time box |
| AT-23 | NOT_RUN | Streamlit four pages: host UI verified manually (`http://127.0.0.1:8599` 200 + `_stcore/health` ok) but not bound to an automated AT test |
| AT-24 | NOT_RUN | public dataset provenance import deferred: V1 importer requires 6D profile; public dataset is 8D state/7D action |
| AT-25 | DONE | GC reference protection |

`make up` container evidence, GPU YOLO, real SIGKILL/network fault injection and UI screenshots are
`DEFERRED_LINUX` per ADR-0011 (`docs/linux-demo-plan.md`).

## 3. V2 suite

V2 full suite: **245 passed, 0 skipped** (`pytest tests/v2 -q` after DO2-08/13/16/17); details and
DO2-01..35 honest statuses are in [docs/test-report-v2.md](test-report-v2.md). Overall ceiling:
`CORE_COMPLETE_REAL_DATA_BLOCKED` (DO2-30..35 container/GPU/real-load/delivery not run).

## 4. Reproduce

```bash
uv run alembic upgrade head
uv run python -m embodied_dataops.cli.main acceptance
uv run python -m embodied_dataops.cli.main openapi-v2
```

Artifacts: `artifacts/test-results/{junit.xml,at-matrix.json,summary.md,v1-full-junit.xml}`,
`artifacts/api-smoke/`, `artifacts/dataset-export/`, `artifacts/public-dataset/` (when fetched).
