# ADR-0006: API key bootstrap output

Status: accepted (2026-10-29), development convenience

## Context

SPEC §9.5 requires the initial keys to be shown **once** and stores only their
SHA-256 digests; `.env` must not contain a distributed key. The one-shot
`make seed`/`make demo` scripts and the local Streamlit dev server still need a
usable key without a human copying it between terminals.

## Decision

`make bootstrap` / `dataops bootstrap` prints the plaintext keys once and writes
the same values to the gitignored file `var/bootstrap/api_keys.json` (mode 0600
on POSIX), explicitly marked as a local-development convenience. The database
stores only digests. Docker-compose deployments receive keys through the same
bootstrap step; the file is never part of the delivered configuration and is
deleted by `make clean`.

## Consequences

* Automated local demo/seed flows work without embedding secrets in `.env`.
* The delivered repository contains no plaintext keys; rotation means creating a
  new key (`dataops keys --create admin`) and revoking the old row.
* This is a documented local-only exception, not a production secret store.
