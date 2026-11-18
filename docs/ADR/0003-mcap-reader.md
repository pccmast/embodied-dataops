# ADR-0003: MCAP reading in the ROS 2 worker and the local fallback reader

Status: accepted (2026-10-29)

## Context

SPEC §5.4 requires MCAP profile v1 to be read by `worker-ros2` using
`rosbag2_py` with the MCAP storage plugin, and forbids the main API Python
environment from importing ROS dynamic libraries. Development and unit tests run
on a host without ROS 2, so the profile parser must be testable there while the
production container still uses `rosbag2_py`.

## Decision

1. `worker-ros2` runs `ros:jazzy-ros-base` with `rosbag2_py`,
   `rosbag2_storage_mcap` and the three message packages, using the
   distribution's Python (no pip-installed project, see
   `infra/docker/ros_worker.Dockerfile`).
2. `services/ros2_adapter/reader.py` prefers the `rosbag2_py` reader whenever it
   is importable. Otherwise it falls back to the pure-Python `mcap` package plus
   a hand-written CDR decoder (`services/ros2_adapter/cdr.py`) for exactly the
   three allowed ROS 2 message types.
3. Both readers feed the same `ParsedSource`/`EpisodeSource` interface and the
   same alignment implementation, so the alignment/ZOH/quality rules are
   identical in both paths. The reader used for a given import/test is recorded
   in the adapter's metadata (`reader_backend`), and ROS-native reader tests are
   marked `ros2` and skipped only when `rosbag2_py` is genuinely unavailable.
4. Fixtures F05/F06 are generated as real ROS 2 CDR MCAP files with the `mcap`
   writer (schema encoding `ros2msg`, CDR payloads) so they can be read back by
   either reader.

## Consequences

* Production keeps the spec-required `rosbag2_py` path.
* Local tests exercise the identical profile validation, timebase and alignment
  logic without installing ROS on the host.
* The fallback does not support arbitrary ROS 2 types: anything outside the
  profile returns `UNSUPPORTED_TOPIC_PROFILE`, as required.
