# ADR-0005: Synchronous dataset build inside the create-version transaction

Status: accepted (2026-10-29), v1 scope decision

## Context

SPEC §9.4 returns `202 {version_id, state: BUILDING}` for
`POST /datasets/{id}/versions` and §10.1 defines `BUILDING → READY`. The
specification also requires the frozen manifest, the official LeRobot export and
the independent-reader round-trip to succeed before a version becomes `READY`.
The v1 task DAG (§6.1) has no dataset-build step and adding one would require a
second task-parent type beyond the import/pipeline split decided in ADR-0001.

## Decision

`create_version` performs validation, manifest freezing, official export and
round-trip verification synchronously inside the request transaction, then
persists the version as `READY` (or `FAILED` with `error_code`/`error_detail`).
The HTTP contract still returns `202` with the version id and its state; the
response is emitted after the build. The API route is the only caller, and the
operation is safe to retry (a retry after a failure creates the next version
number).

## Consequences

* Simple, verifiable v1 behaviour with an honest state machine: no version is
  ever marked `READY` without a successful round-trip.
* Long exports occupy one API request. This is acceptable for the demo scale
  (≤ a few thousand frames); a v2 must move export into a dedicated
  `EXPORT_DATASET` task step when datasets grow.
* The benchmark/report does not claim asynchronous dataset builds.
