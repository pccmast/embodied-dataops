#!/usr/bin/env bash
# Scripted end-to-end demo: writes artifacts/demo-report.json.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m embodied_dataops.cli.main demo "$@"
