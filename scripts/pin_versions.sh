#!/usr/bin/env bash
# Record the exact image digests currently used by docker-compose into
# infra/versions.lock (SPEC 2 requires pinned tags/digests).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p infra
for image in postgres:16.10-alpine redis:7.4.5-alpine \
             minio/minio:RELEASE.2025-09-07T16-13-09Z \
             minio/mc:RELEASE.2025-08-13T08-35-41Z \
             prom/prometheus:v3.5.0 grafana/grafana:12.2.0 ros:jazzy-ros-base; do
  digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$image" 2>/dev/null || true)
  printf '%s = "%s"\n' "${image%%:*}" "${digest:-<not-pulled>}"
done
