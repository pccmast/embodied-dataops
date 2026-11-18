# Operations runbook

## Commands

| Command | Purpose |
|---|---|
| `make bootstrap` | create `.env`, install deps, migrate, print initial API keys once |
| `make up` | start PostgreSQL, Redis, MinIO, migrate, API, dispatcher, workers |
| `make migrate` | apply Alembic migrations (idempotent) |
| `make seed` | generate the ten fixtures and import/process them through the real pipeline |
| `make test` | unit + integration + contract tests (excludes performance/ros2/vision by default) |
| `make e2e` | end-to-end AT tests |
| `make failure-test` | lease/outbox/crash failure-injection tests |
| `make benchmark` | 100-episode synthetic load, writes `artifacts/benchmarks/report.json` |
| `make acceptance` | runs the AT suite, writes junit + AT matrix |
| `make demo` | scripted demo, writes `artifacts/demo-report.json` |
| `make down` | stop the stack (volumes kept) |
| `make clean CONFIRM=YES` | delete local `var/`, generated fixtures and caches (not the DB volumes) |

## Health & readiness

* `GET /health/live` — process liveness, always cheap.
* `GET /health/ready` — checks PostgreSQL (`SELECT 1`), Redis (`PING`) and the
  object store (bucket head/create); returns 503 with a per-component map when a
  dependency is down. The API returns `503 DEPENDENCY_UNAVAILABLE` for external
  dependency failures after startup.

## Host-mode stack (Windows development, ADR-0011)

```powershell
scripts/dev_stack.ps1 -Force       # dependency containers + migrate + API + dispatcher + workers
scripts/stop_dev_stack.ps1         # stop host processes; containers keep running
```

Container images are **not built on Windows**; `make up` / `make ui-image` / GPU / ROS-container
verification are executed on the Linux + cloud-GPU environment per `docs/linux-demo-plan.md`.
Host-mode uses the same PostgreSQL/Redis/Silo containers and the same API/worker code, so
functional behaviour is identical; only the packaging evidence is deferred.

## UI

The UI calls the HTTP API only. On Windows development hosts the supported path is
the host runner (the Docker image export path hangs in Docker Desktop for Windows,
see ADR-0010):

```bash
make ui            # streamlit run src/embodied_ui/app.py (API key from var/bootstrap or API_KEY)
```

On Linux/CI the container image is built and used:

```bash
make ui-image && docker compose --profile ui up -d ui
```

Both paths use the same `src/embodied_ui` code and the same `UI_API_BASE_URL` /
`API_KEY` configuration. A UI that starts but cannot authenticate shows the
recovery hint from the API error envelope.

## Key rotation

The bootstrap keys are printed once; only digests are stored. Create a new key
with `dataops keys --create admin --name rotation-2026-10` and revoke the old row
by setting `api_keys.enabled=false` / `revoked_at` (SQL or a future admin route).
Never put keys into `.env` or logs.

## Task operations

* Inspect a run: `GET /api/v1/pipeline-runs/{run_id}` — per-step state, attempt,
  worker, lease and error code.
* Cancel a run: `POST /api/v1/pipeline-runs/{run_id}/cancel` — idempotent; pending
  tasks become `CANCELLED`, running steps stop cooperatively, downstream steps
  never start.
* Retry a failed step: `POST /api/v1/tasks/{task_id}/retry` — creates a **new**
  run starting at the failed step; the old run remains as evidence.
* Requeue after a worker crash: nothing to do — the reaper reclaims the expired
  lease within ~10–70 s and either retries or fails the task at attempt 3.
* Redis lost: tasks stay in PostgreSQL. Restart the dispatcher; it republishes
  the outbox, and the reaper republishes due tasks. No task is lost.

## Object store

* Raw source bytes live at `raw/sha256/<sha256>` and are immutable.
* Canonical/episode artifacts are deterministic per episode version.
* `dataops gc --orphan-hours 24 [--delete]` scans `artifacts/staging/`: an object
  referenced by any `assets` row is never deleted; unreferenced objects younger
  than the window are kept. Default is a dry run.
* Deleting a PUBLISHED dataset's export object is an incident: publishes
  re-verify SHA-256, and a missing/tampered object blocks further publication.

## Backups and migrations

* Migrations are forward-only in production (`alembic upgrade head`); take a
  PostgreSQL logical backup (`pg_dump`) before upgrading.
* `make migrate` is idempotent and is executed by the compose `migrate` service
  before API/workers start.

## Common failures

| Symptom | Action |
|---|---|
| `/health/ready` 503 with `postgres: error` | check `dataops-db`, connection string, `make migrate` |
| `/health/ready` 503 with `object_store: error` | check `dataops-minio` (PGSTY Silo) container, S3 credentials and `S3_ENDPOINT_INTERNAL` |
| `503 DEPENDENCY_UNAVAILABLE` during import | object store down; fix the object store (PGSTY Silo / S3), retry the task (retryable) |
| tasks stuck `PENDING` | dispatcher/worker down or Redis unreachable; the dispatcher republishes every second and the reaper every 10 s |
| `UNSUPPORTED_TOPIC_PROFILE` | MCAP is not the restricted ROS 2 CDR profile; re-record or extend the profile (new ADR) |
| `SOURCE_HASH_MISMATCH` | upload was corrupted or the declared sha256 was wrong; upload again |
| `UNSAFE_ARCHIVE` | tar contains absolute paths/links/`..`; rejected by design |
| export `FAILED` with `LEROBOT_API_UNAVAILABLE` | official `lerobot==0.4.4` not installed in the worker image; rebuild with `INSTALL_LEROBOT=1` |
