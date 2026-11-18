# Architecture

## 1. Services

```text
                 +-------------------+        presigned PUT/GET (15/10 min)
 browser/Streamlit|  api (FastAPI)   |<-------------------------------> MinIO
                 +---------+---------+
                           |  one transaction per request
                           v
                 +-------------------+        +---------------------+
                 |    PostgreSQL 16  |<------>|  dispatcher (1 s)   |
                 |  task truth source|        | outbox -> Redis     |
                 +---------+---------+        +----------+----------+
                           ^                             |
              lease/fencing|                             v
                 +---------+---------+        +---------------------+
                 |  worker-cpu  x2   |<-------| Redis Streams 7.4   |
                 |  worker-vision x1 |        | wakeup hints only   |
                 |  worker-ros2 x1   |        +---------------------+
                 +---------+---------+
                           | deterministic handlers
                           v
                 canonical parquet/mp4, quality reports, annotations, exports
```

* **api** validates requests, enforces RBAC/idempotency and commits business rows.
  It never performs work longer than a couple of seconds; heavy work becomes a
  task and returns `202`.
* **dispatcher** publishes outbox rows to `stream:tasks:{cpu|vision|ros2}` and
  compensates for lost wakeups by periodic re-publication from PostgreSQL.
* **workers** claim tasks with `SELECT … FOR UPDATE SKIP LOCKED`, execute one
  step, heartbeat the lease and commit results through a fencing-token CAS.
* **worker-ros2** is the only place with ROS 2 Jazzy / `rosbag2_py`; the API
  Python 3.11 environment never imports ROS libraries.

## 2. Task protocol

```text
PENDING --claim--> RUNNING --commit(CAS token)--> SUCCEEDED
                      |  \
                      |   +-- terminal error / 3rd attempt --> FAILED
                      +------ retryable / lease expiry --> RETRY_WAIT --due--> PENDING
PENDING / RETRY_WAIT --cancel--> CANCELLED
RUNNING --cancel_requested--> cooperative stop --CAS--> CANCELLED
```

* creating tasks + dependencies + the outbox row happens in **one** PostgreSQL
  transaction; Redis is never the source of truth;
* `lease_until = now + 60 s`, heartbeat every 15 s (token-checked), reaper scan
  every 10 s;
* retries: 5 s then 20 s with a deterministic ±10 % jitter derived from
  `hash(task_id)`, exactly 3 claims then `FAILED`;
* completion is `UPDATE … WHERE state='RUNNING' AND lease_token=:token`;
  a stale worker can never overwrite newer output;
* handler-created rows are rolled back with the same transaction if the CAS
  fails, so no stale "valid" asset can survive;
* outputs left behind by a crashed execution are unreferenced objects; the GC
  removes only unreferenced staging objects older than 24 h.

## 3. DAG

```text
ImportJob: VALIDATE_SOURCE -> IMPORT_EPISODES
                                   |
Episode PipelineRun:  NORMALIZE_EPISODE -> QUALITY_CHECK
                       -> DETECT_OBJECTS (stopped on QC FAIL)
                       -> BUILD_PREVIEW
                       -> READY_FOR_REVIEW
```

`READY_FOR_REVIEW` never approves annotations automatically: approval is an
explicit reviewer action, and only approved robot episodes can enter a
publishable dataset version.

## 4. Data flow / object boundaries

* raw source bytes are immutable at `raw/sha256/<sha256>`;
* canonical artifacts are deterministic per `episode_id/v1`;
* the database stores metadata, states, verdicts and asset pointers only — never
  video bytes, frame JPEGs or masks;
* `episodes` are separated into `robot_episode` and `ego_video_episode`;
  an ego video can never be exported as a robot action trajectory.

## 5. Failure model

| Failure | Behaviour |
|---|---|
| worker crash / lease expiry | reaper moves the task to `RETRY_WAIT` or `FAILED`; no permanent `RUNNING` |
| duplicated Redis delivery | second claim finds `RUNNING`/terminal state and acks without executing |
| Redis unavailable at publish | outbox rows stay unpublished; dispatcher retries; reaper also re-publishes due tasks |
| object store unavailable | handler raises a retryable error; attempt/backoff applies |
| cancel | pending/retry tasks become `CANCELLED`; running tasks observe `cancel_requested` and stop cooperatively; downstream steps are never started |
| QC FAIL | task succeeds, report persists, pipeline run becomes `FAILED`, DETECT_OBJECTS is not started |

## 6. Observability

Structured JSON logs carry `ts, level, service, event, request_id, run_id,
task_id, episode_id, worker_id, error_code`. Prometheus metrics:
`dataops_tasks_total`, `dataops_task_duration_seconds`, `dataops_queue_wait_seconds`,
`dataops_active_workers`, `dataops_lease_expirations_total`,
`dataops_retries_total`, `dataops_episode_qc_total`, `dataops_sync_error_ms`,
`dataops_import_bytes_total`, `dataops_annotation_frames_total`.
