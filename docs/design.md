# Design: what this platform guarantees, and what it refuses to do

This document is the condensed design record. It replaces the long specification that the
implementation was built against; the binding decisions are in `docs/ADR/`, and the
per-item delivery state (DONE / PARTIAL / BLOCKED / NOT_RUN) is in
`docs/implementation-status.md`.

## 1. The problem

Robot and ego-video data arrive as heterogeneous archives: LeRobot v3 `.tar.gz`, raw MCAP
recordings with several topics at different rates, plain MP4. Turning that into a training
dataset means answering four questions for every Episode, identically and repeatably:

1. **What is the canonical sample?** Which timestamps, which grid, which units?
2. **Is it usable?** Which quality violations exist, with what evidence, at which severity?
3. **Who approved it?** Which human, with which note, on which immutable artifact?
4. **Can this exact version be reproduced?** Same inputs, same policy, same bytes?

The interesting engineering is not "run 7 steps" — it is making those four answers survive
crashes, duplicate deliveries, stale workers and policy changes.

## 2. Canonical contract (`canonical-episode/1.0`)

* **One grid**: 10 Hz, `t_ns = k × 100 ms`, `frame_index = k`. V2 adds a variable-rate
  native layout (`canonical-episode/2.0`) that keeps per-stream native indices.
* **Fixed geometry**: 640×480, single `front` camera, 6-D `state` / `action` for V1.
* **Alignment is semantic, not "sampling"**: images use nearest-neighbour within 50 ms;
  state is linearly interpolated only when both ends are within 50 ms; action uses
  zero-order hold, backward only, within 100 ms. **A future action is never used.**
* **Source clock is preserved**: every modality row records `clock_domain` and
  min/max timestamps; V2 keeps each stream's native time index as its own immutable Asset.
  The canonical sample also carries `state_source_t_ns` / `action_source_t_ns`.
* **No fabrication**: a grid point without a legal image/state/action is **not written** to
  the publishable samples. It is recorded in `AlignmentDiagnostics` and becomes a
  `SYNC_MISSING_*` finding. Zero-fill does not exist in this codebase.
* **Origin**: `t = 0` is the `log_time` of the first camera frame; MCAP uses `log_time`
  uniformly and keeps `header.stamp` only as a diagnostic difference.

## 3. Task reliability: PostgreSQL is the truth source

The DAG for one Episode is fixed and linear:

```
NORMALIZE → QUALITY_CHECK → DETECT_OBJECTS → BUILD_PREVIEW → READY_FOR_REVIEW
```

Reliability comes from four primitives, all in the database:

| Primitive | Mechanism | Guarantee |
|---|---|---|
| Claim | `SELECT … FOR UPDATE SKIP LOCKED` + in-lock eligibility re-check | two workers cannot both take one task |
| Lease | `lease_token` (UUIDv4) + `lease_until = now + 60 s`, heartbeat every 15 s | a crashed worker cannot hold a task forever |
| Commit | CAS `UPDATE … WHERE id AND state='RUNNING' AND lease_token=:token` | a stale worker cannot overwrite a new result |
| Wakeup | transactional outbox → Redis Streams | a lost or duplicated notification only delays work |

`attempt` increments only on a **new database lease**, so duplicated Redis notifications
cannot consume the retry budget. The reaper runs every 10 s and expires dead leases to
`RETRY_WAIT` (deterministic backoff 5 s / 20 s ±10 %), promotes due retries back to
`PENDING` with a fresh outbox row, and re-notifies `PENDING` tasks whose wakeup was lost.
Recovery after `kill -9` is bounded by ≈ lease (60 s) + reap interval (10 s) + backoff.

Handler output is written under `artifacts/staging/{task_id}/{lease_token}/…`, hashed, then
registered as an immutable Asset only inside the committed transaction. A failed CAS rolls
back the whole attempt, so its Assets, reports and outbox rows never existed.

## 4. Quality: policy as data, rules as pure functions

* Every rule is a pure function `(QualityInput, thresholds) → list[Finding]`. No clock, no
  network, no database. Findings carry observed values, thresholds, bounded evidence and
  affected time ranges.
* Applicability is **evidence-driven**: a rule whose evidence is absent is classified
  `not_applicable` (e.g. robot rules on an ego video) or `unavailable` (evidence should
  exist but is missing). Silence never means PASS.
* Severity is declared per rule: `ERROR` ⇒ verdict `FAIL` and downstream tasks are
  cancelled; `WARN` ⇒ verdict `WARN` and the pipeline continues; `INFO` never changes the
  verdict. `FRAME_GAP_EXCESS` escalates from WARN to ERROR when grid images are missing.
* The report is immutable and pinned to a policy version. Changing a threshold or adding a
  rule requires a new policy version; historical reports keep pointing at the old one.
* V2 binds thresholds to a versioned robot profile (6D/7D, joint names, limits). When a
  profile has no documented limits, the physics rules are `not_applicable` — never PASS.

## 5. Human review is a separate gate, not a pipeline step

The pipeline ends at `READY_FOR_REVIEW`. Review is a synchronous API operation on the
**annotation set**, not on the quality report:

* `PENDING_REVIEW → APPROVED | REJECTED`, one-shot: a second decision is `409`.
* Rejection never edits the artifact; re-running detection creates a new annotation set.
* Approval re-validates the stored artifact (schema + coordinate bounds) because model
  output is untrusted input.
* A `WARN` Episode may only be approved with a release note of **≥ 10 characters**; the API
  enforces it (`422`). PASS Episodes do not need a note.
* Publication requires both gates: `quality_status ∈ {PASS, WARN}` **and** an approved
  annotation set. `FAIL` Episodes have no path to a published version.

## 6. Delivery: frozen versions, official writer

A dataset version is built synchronously, lists its Episodes and their assets in a
`dataset-manifest/1.0` document, and is immutable once frozen. Export uses the official
LeRobot v3 Writer; verification re-opens the export with the official Reader in a separate
process and compares schema, lengths and values. Synthetic Episodes are labelled
`data_origin=synthetic` and are blocked from `PUBLISHED` publication paths that require real
capture provenance.

## 7. Deliberate non-goals

* Not a robot controller, not real-time, not a VLA training stack.
* No claim of production scale, million-Episode throughput, or automatic point-cloud
  registration. PointCloud2 is metadata + quality only.
* No in-memory queue, no "best effort" completion, no zero-fill shortcut.
* No silent success. Blocked items are reported as BLOCKED with the reason.

## 8. Why the orchestration is hand-written

| Candidate | Why not (for this prototype) |
|---|---|
| Airflow | Schedules time-based DAGs that repeat; this needs one dynamically created run per Episode, plus staging/CAS semantics at commit. Whole-task retry has no fencing. |
| Celery | Queue-as-truth conflicts with the outbox model; no DAG, no dependency-correct cancellation. |
| Ray | A compute framework whose fault tolerance is recomputation. Recomputing a step here means re-writing objects and re-billing a provider call. |
| Temporal | The closest match and the natural migration target at larger scale; rejected for V1 because the cluster's operational surface dwarfs the project, and the domain contracts (input digest idempotency, staging promotion, QC gating, immutable reports) would still be custom code. |

The design choice is explicit: **be able to explain every failure path**, and prove the
three critical ones with fault injection (duplicate delivery, worker `kill -9`, stale-token
late commit).

## 9. Reading order for a reviewer

1. `src/embodied_dataops/workflow/claim.py` → `runner.py` → `completion.py` → `reaper.py`
2. `src/embodied_dataops/canonical/alignment.py` + `timebase.py`
3. `src/embodied_dataops/quality/rules.py` + `policy.py`
4. `src/embodied_dataops/workflow/dispatcher.py` + `outbox.py` + `queue.py`
5. `docs/ADR/0001-task-protocol.md`, `docs/ADR/0007-v2-canonical-profile-conditional-writes.md`
6. `tests/concurrency/`, `tests/failure/` — the tests that back the claims above
