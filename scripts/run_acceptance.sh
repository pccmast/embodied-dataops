#!/usr/bin/env bash
# Embodied DataOps - acceptance entry point (AT-01..AT-25, SPEC 14).
#
# Thin wrapper over pytest: the acceptance scenarios and their assertions live in
# tests/ (unit, integration, failure, contract, e2e). Machine-readable evidence is
# written to artifacts/test-results/junit.xml. Performance benchmarks are excluded
# here because they have their own target (`make benchmark`).
#
# Usage: scripts/run_acceptance.sh [pytest args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  PY="$ROOT/.venv/Scripts/python.exe"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python}"
fi

: "${TEST_DATABASE_URL:=postgresql+psycopg://dataops:dataops@localhost:5432/dataops_test}"
: "${TEST_REDIS_URL:=redis://localhost:6379/15}"
export TEST_DATABASE_URL TEST_REDIS_URL

mkdir -p artifacts/test-results
echo "== Embodied DataOps acceptance suite =="
echo "== DB: ${TEST_DATABASE_URL}"
echo "== Redis: ${TEST_REDIS_URL}"
exec "$PY" -m pytest tests \
  -m "not performance" \
  --junitxml=artifacts/test-results/junit.xml \
  -p no:cacheprovider \
  "$@"
