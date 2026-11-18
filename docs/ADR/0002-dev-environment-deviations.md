# ADR-0002: Local Windows development host vs. the Linux delivery baseline

Status: accepted (2026-10-29), dev-environment only

## Context

SPEC §2 fixes Python 3.11 for `api`/`worker-cpu`, FFmpeg/ffprobe for video and
Linux x86-64 Docker as the delivery baseline. The development host used to build
and test this delivery is Windows 11 with `uv`, Docker Desktop and an
unavailable GitHub/CDN large-file path (release assets and Python standalone
builds effectively stall). Python 3.11 could not be installed locally; only
Python 3.12.13 (uv-managed) was usable within the session budget.

## Decision

1. **Production images stay on Python 3.11** (`python:3.11.13-slim-bookworm` in
   `infra/docker/*.Dockerfile`), matching SPEC §2.
2. Local development/tests use the available Python 3.12.13; `requires-python`
   is `>=3.11` so both work. Test evidence recorded on this host states the
   interpreter version explicitly (`OS: Windows, Python 3.12.13`).
3. FFmpeg binaries: the delivery image installs distro `ffmpeg`/`ffprobe`. On
   this host the network could not fetch the static Windows build; the code uses
   the configured `FFMPEG_BIN`/`FFPROBE_BIN`, then `PATH`, then the
   `imageio-ffmpeg` bundled binary, then PyAV for probing/decoding. This is a
   codec-provider substitution only; the same encode/decode contract
   (H.264 yuv420p 640×480 10 fps, full-decode verification) is exercised.
4. MinIO is the object store in compose; unit/integration tests use the
   `LocalStorage` backend with the same interface and HMAC-signed URLs. All
   storage tests therefore exercise real byte movement, hashing and expiry
   logic; a MinIO-backed smoke test is recorded separately when the container
   image is available in the environment.

## Consequences

* No acceptance criterion is weakened: the Linux/3.11/MinIO path is the shipped
  configuration, and every deviation is reported in
  `docs/implementation-status.md` and `docs/test-report.md`.
* MCAP ROS 2 decoding has a documented second reader path (ADR-0003).
