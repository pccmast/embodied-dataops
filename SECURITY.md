# Security policy

## What this project is

Embodied DataOps is a **single-host engineering prototype** for robot/ego data production.
It is not a production robot-control system and it has not been hardened for exposure to the
public internet. Run it on localhost or behind an authenticating reverse proxy.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately to the
maintainer of the fork/repository you are using (for the upstream project: open a GitHub
security advisory on the canonical repository, or contact the address listed on the
maintainer's profile).

Include: affected component, version/commit, reproduction steps, and the impact you believe
it has. Expect an initial reply within a week. This is a personal open-source project, not a
commercial product with an SLA.

## Known security-relevant design decisions

These are documented behaviours, not vulnerabilities — but they matter if you deploy this:

* **API keys** are stored only as SHA-256 digests. `make bootstrap` prints the plaintext key
  exactly once; there is no recovery path by design.
* **Presigned URLs** are short-lived and built from `S3_ENDPOINT_PUBLIC`. If that variable
  still points at `localhost`, the links will not work from another machine — this is a
  configuration issue, not a leak.
* **Published ports** are bound to `PUBLISHED_HOST` (default `127.0.0.1`). Setting it to
  `0.0.0.0` exposes PostgreSQL, Redis and the object store directly. Do not do that on a
  shared host.
* **Outbound egress** to a VLM provider is gated by `source_imports.egress_allowed`
  (default `false`); the API and the clip handler both refuse a live call without it.
* **Default credentials** in `.env.example` and `docker-compose.yml` (`dataops`,
  `dataops-secret`) exist so a local demo works out of the box. Change them before any
  non-local use.
* **Archive extraction** rejects traversal, symlinks and size bombs; a rejected archive is a
  terminal import error rather than a partial extraction.

## Out of scope

* Denial of service by design (the platform accepts large uploads up to a documented limit).
* Host-level compromise, container escape and cloud-account misconfiguration.
* Vulnerabilities in dependencies (report them upstream; we still want to know if the
  platform's usage makes them exploitable).
