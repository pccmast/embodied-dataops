# Changelog

The public repository starts from a curated snapshot rather than the full development
history. The milestones below are what that snapshot contains; per-item delivery state is in
[`docs/implementation-status.md`](docs/implementation-status.md).

## 1.0.0 — V1 platform (2026-10)

* Ingest: LeRobot v3 `.tar.gz` through the official API, Ego MP4 through ffmpeg, restricted
  ROS 2 MCAP profile through a separate Jazzy worker.
* Canonical contract `canonical-episode/1.0`: 10 Hz / 640×480 / 6-D state and action, with
  source timestamps retained and no zero-filling.
* Task engine: PostgreSQL state machine, transactional outbox, Redis Streams wakeups,
  60 s lease / 15 s heartbeat / fencing token / 3 attempts / reaper, cancellation cascade.
* Quality `qc-policy/1.0`: declarative rules, evidence-carrying findings, PASS/WARN/FAIL
  verdicts, machine-readable policy document.
* Annotation: YOLO11n pre-annotation with immutable annotation sets and one-shot review;
  WARN Episodes require a ≥ 10-character release note.
* Delivery: immutable dataset versions + `dataset-manifest/1.0`, official LeRobot v3 export
  and separate-process round-trip verification.
* Ops: Prometheus metrics, structured JSON logs, Alembic migrations, failure-injection
  scripts, synthetic fixture set with provenance.

## 1.1.0 — V2 incremental (2026-10-30)

* Robot profile registry (6D/7D) bound to Episodes; semantic QC uses profile limits and
  reports `not_applicable` when limits are undocumented.
* Native multi-stream time indices plus `NATIVE` / `ALIGNED_TRAINING` / `PREVIEW` canonical
  views keyed by a generation-config hash.
* Annotation revisions, conditional writes with operation receipts and effect fences,
  diagnostic read APIs, bounded PointCloud2 metadata and quality checks.
* Measured benchmark harness (`scripts/bench_v2.py`): throughput, CPU, memory, duplicate
  assets and permanent-RUNNING assertions.

## 1.2.0 — V2.1 AI pre-annotation (2026-10-30)

* Clip extraction and per-clip VLM inference tasks with strict `ego-action-candidates/2.1`
  schema validation (no repair, no auto-approval, evidence-frame checks).
* Provider governance: replay vs live isolation, retry classification, budget reservation,
  honest `UNKNOWN_UPSTREAM_CHARGE` accounting, egress gating.
* Live VLM acceptance items remain `BLOCKED` (no provider key in the measured environment);
  the replay path is explicitly not a substitute.

## Public snapshot edits (2026-11)

* Licence set to AGPL-3.0-only to match the Ultralytics dependency and the intent to keep
  the task protocol open as a network service.
* Removed internal specification packages, job-search notes and the raw experiment archive;
  added `docs/design.md` as the condensed design record.
* Added `NOTICE`, `SECURITY.md`, `CONTRIBUTING.md` and sanitised `docs/evidence/` extracts.
