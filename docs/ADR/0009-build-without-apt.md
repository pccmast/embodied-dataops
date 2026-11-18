# ADR-0009: Docker images without `apt-get` (build-network workaround)

Status: accepted (2026-10-29)
Related: ADR-0002 (dev-host deviations), ADR-0008 (object-store image).

## Context

The delivered compose baseline expects Linux x86-64 images built from
`python:3.11.13-slim-bookworm` with distro FFmpeg/`curl` installed via
`apt-get`. On the build host used for this delivery:

- `deb.debian.org` was unreachable; the Tsinghua apt mirror returned indexes but
  many `.deb` fetches failed with HTTP 502, and Docker builds failed at the apt
  step;
- `ghcr.io/astral-sh/uv` (used via `COPY --from`) could not be relied on;
- concurrently building and running the acceptance suite caused the Docker
  Desktop 8 GB VM to OOM-kill the PostgreSQL/Redis containers (`exit 137`).

## Decision

1. **api / ui images**: no `apt-get` at all. `uv` is installed from PyPI
   (`pip install uv==0.9.7`) using the configurable `PYPI_INDEX_URL`; the rest of
   the dependency install is the existing `uv sync --frozen`.
2. **cpu worker image**: no `apt-get` by default. FFmpeg/FFprobe capability comes
   from the pip `imageio-ffmpeg` wheel plus PyAV (the same fallback already used
   on the Windows dev host and documented in ADR-0002). Setting
   `INSTALL_SYSTEM_FFMPEG=1` restores the distro package for networks where apt
   works.
3. **ros worker image**: base `ros:jazzy-ros-base` already contains `rosbag2_py`
   and the **MCAP storage plugin** (verified: F05 read natively inside the
   image: 515 `/policy/action`, 1020 `/joint_states`, 300
   `/camera/front/image_raw` messages). No apt install of ROS packages is
   required; application dependencies are installed into the ROS distribution's
   Python 3.12 from `services/ros2_adapter/requirements-v2.txt` (exported from
   `uv.lock`), and the project code is imported via `PYTHONPATH`.
4. Build mirrors are explicit build arguments (`PYPI_INDEX_URL`,
   `UV_DEFAULT_INDEX`) so the same Dockerfile builds on the open internet
   (defaults are official indexes) and on a mirrored network without editing the
   delivered files.

## Consequences

- The image build no longer depends on Debian apt reachability; the only network
  requirement is a reachable PyPI index (and the base images themselves).
- The FFmpeg provider in the default worker image is the `imageio-ffmpeg` binary,
  not the distro package. `ffprobe` is still emulated by PyAV (ADR-0002); the
  processing wrapper exposes the same contract (probe, full decode, encode,
  frame extraction, thumbnails). Set `INSTALL_SYSTEM_FFMPEG=1` when distro
  FFmpeg is required by policy.
- The image has no `curl`; healthchecks use the Python interpreter already in the
  image (API) or the application itself (workers).
- `uv.lock` remains the single resolved dependency set; the ROS image consumes an
  exported requirements file generated from it, so version drift is visible.
- This is a build-environment workaround, not a change to the runtime data
  contract; the V1/V2 acceptance criteria are unaffected.
