# Contributing

Thanks for looking. This repository is an engineering prototype with unusually strict
rules about data correctness — most of them exist because a looser rule produced a real bug
(see `docs/implementation-status.md`). Read the invariants below before changing behaviour.

## Project invariants (non-negotiable)

1. **Never present synthetic data as real robot capture.** Synthetic fixtures carry
   `data_origin=synthetic` and a provenance row in `fixtures/manifest.json`.
2. **PostgreSQL is the only task truth source.** Redis Streams carry wakeup hints only.
   Do not add an in-memory queue substitute, and never treat an unacked stream entry as a
   pending task.
3. **No filename guessing for LeRobot v3 episode boundaries.** Use the official
   `lerobot==0.4.4` API and `meta/info.json`.
4. **Never zero-fill, never interpolate into the future, never guess a robot action.**
   A missing sample is a quality finding; a stale action is an error. The publishable
   artifact simply does not contain the grid point.
5. **No TODOs, fake success responses, empty routes or pseudo-code** counted as done.
6. **No `xfail` for a MUST acceptance item.** Environment-dependent tests use the
   `ros2`, `vision` and `performance` markers and record an explicit result
   (DONE / PARTIAL / BLOCKED / NOT_RUN).
7. **No unmigrated schema.** Any table or column change needs an Alembic revision.
8. **No unverified performance claims.** P50/P95/throughput numbers need a stored report
   with the machine, the command and the fixture used.
9. **Immutable data.** Raw objects and published dataset versions are never mutated;
   a correction creates a new object or version.
10. **Security basics.** API keys are stored as SHA-256 only; archives are extracted with
    traversal/link/bomb protection; logs never contain keys or presigned URLs.

When a rule and a convenience conflict, the rule wins. A deviation needs an ADR under
`docs/ADR/`.

## Setup

```bash
make bootstrap        # .env, venv, dependencies
make up               # postgres, redis, object store, migrate, api, dispatcher, workers
make seed             # generate the synthetic fixtures and import them
```

Windows is a development host only: container images are built on Linux/CI
(ADR-0010/0011). Use `scripts/dev_stack.ps1` to run the host-process stack.

## Tests

```bash
uv run pytest -m unit
uv run pytest -m concurrency        # the release gate: two workers on real row locks
uv run pytest -m "integration and not ros2 and not vision"
uv run pytest tests/failure -q
make bench-v2                       # only when you intend to publish numbers
```

Markers describe *dependencies*, not scope. `-m integration` mixes import/export flows with
atomicity checks; `-m concurrency` is the ~12 cases that actually prove two workers cannot
both win the same lease, budget or unique key. A green integration run says nothing about
atomicity — run the concurrency gate before claiming a change is safe.

## Definition of done for one work package

1. The code implements the behaviour, not a stub.
2. Tests exist (unit + at least one negative case) and pass in this environment.
3. `docs/implementation-status.md` reflects DONE / PARTIAL / BLOCKED with evidence.
4. Any new dependency is pinned in `pyproject.toml` + `uv.lock` and listed in
   `docs/licenses.md`.

## Commit style

Short imperative subject lines; explain the *why* in the body when a rule or an invariant
is involved. Keep refactors separate from behaviour changes.

## Licence

By contributing you agree that your contribution is licensed under **AGPL-3.0-only**
(see `LICENSE`).
