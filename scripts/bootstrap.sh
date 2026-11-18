#!/usr/bin/env bash
# Bootstrap a local development environment (Linux/macOS/Git-Bash).
# Prefer `make bootstrap`; this script is a thin wrapper kept for CI parity.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m embodied_dataops.cli.main bootstrap "$@"
