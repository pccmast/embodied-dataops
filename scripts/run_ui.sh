#!/usr/bin/env bash
# Run the Streamlit UI on the host (no container image required).
# The UI image itself is built on Linux/CI (ADR-0010).
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m embodied_dataops.cli.main ui "$@"
