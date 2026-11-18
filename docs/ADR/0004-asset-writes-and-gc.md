# ADR-0004: Asset writes, fencing and orphan GC

Status: accepted (2026-10-29)

## Context

SPEC §6.3 describes writing worker output to a staging prefix and committing it
with a fencing token; §6.3.6 requires GC of orphan staging objects after 24 h
and forbids deleting objects referenced by a valid Asset.

## Decision

1. A handler runs inside the engine transaction and writes objects to their
   final deterministic keys (e.g. `canonical/{episode}/v1/samples.parquet`),
   creating the corresponding `assets` rows itself via the shared registration
   helpers.
2. The engine commits handler rows only through the lease-CAS update
   (`RUNNING` + matching `lease_token`). If the CAS fails, the transaction is
   rolled back: the stale worker's assets and metadata disappear from the
   database by construction, which is the real anti-overwrite guarantee.
3. A crash before the CAS leaves only unreferenced objects. `dataops gc`
   (default dry-run) removes unreferenced `artifacts/staging/` objects older
   than 24 h; referenced assets are always protected. Raw source bytes
   (`raw/sha256/…`) are never GC candidates.
4. Known limitation (reported, not hidden): object bytes are content-addressed
   only for raw sources. A zombie worker that keeps running after losing its
   lease could still overwrite a canonical object key on the object store. Such
   bytes can never become a valid asset, and the next deterministic
   normalization/exports re-write them; the database is not corrupted. Removing
   this residual object-store race would require a content-addressed canonical
   layout (proposed for v2) or pre-CAS server-side copies with a head-object
   version check.

## Consequences

* No valid asset is ever deleted by GC (verified in AT-25 tests).
* Data correctness is guaranteed at the database level; the residual
  object-store race is documented instead of being claimed away.
