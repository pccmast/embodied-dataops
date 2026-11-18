# ADR-0001: Task truth source, outbox and import-level tasks

Status: accepted (2026-10-29)

## Context

SPEC §6 requires a persistent task protocol with leases, retries, fencing and
recovery, while explicitly forbidding Redis as the task truth source and
forbidding a `pipeline_runs.episode_id` overload for source-import jobs.

## Decision

1. PostgreSQL is the single source of truth for task state. Redis Streams carry
   wakeup hints only: every execution starts with a database lease.
2. Task creation, dependency rows and the outbox event are written in one
   PostgreSQL transaction. The dispatcher publishes outbox rows and only marks
   them published after `XADD`; the 10 s reaper re-publishes due tasks so a lost
   Redis message cannot lose work.
3. Completion uses a fencing-token CAS (`state='RUNNING' AND lease_token=:t`),
   so a stale worker can never overwrite newer output.
4. `tasks.run_id` is nullable and an additive `tasks.import_job_id` column
   references `import_jobs`. A check constraint requires exactly one parent.
   Episode-level tasks keep the spec's `UNIQUE(run_id, step)`; import-level
   tasks get the equivalent partial unique index on `(import_job_id, step)`.
5. Retry backoff is 5 s then 20 s with a deterministic jitter derived from
   `sha256(task_id)` so tests can assert exact timings; attempt 3 is terminal.

## Consequences

* Correctness does not depend on Redis availability; only latency does.
* Two worker pools can race for one task without duplicating outputs.
* Import jobs are auditable independently from Episode pipelines.

## Alternatives rejected

* Celery/Temporal/Airflow: would hide the core reliability design the project
  is meant to demonstrate (SPEC §2).
* Redis as state store: violates SPEC §6.3 and loses state on eviction.
