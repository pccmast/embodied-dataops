#!/usr/bin/env bash
# Embodied DataOps - failure-injection driver for the M2 task protocol (AT-09..AT-14).
#
# This is a thin, honest wrapper over pytest: every scenario lives in
# tests/failure and asserts real PostgreSQL/Redis behaviour. It writes JUnit
# evidence under artifacts/test-results/.
#
# Usage: scripts/inject_failure.sh [all|lease|worker-crash|outbox|retry] [pytest args...]
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

CASE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

case "$CASE" in
  all)          TARGETS=(tests/failure) ;;
  lease)        TARGETS=(tests/failure/test_lease_fencing.py) ;;
  worker-crash) TARGETS=(tests/failure/test_worker_crash.py) ;;
  outbox)       TARGETS=(tests/failure/test_outbox_recovery.py) ;;
  retry)        TARGETS=(tests/failure/test_retry_backoff.py) ;;
  *)
    echo "usage: $0 [all|lease|worker-crash|outbox|retry] [pytest args...]" >&2
    exit 2
    ;;
esac

mkdir -p artifacts/test-results
echo "== Embodied DataOps failure injection: ${CASE} =="
echo "== DB: ${TEST_DATABASE_URL}"
echo "== Redis: ${TEST_REDIS_URL}"
exec "$PY" -m pytest "${TARGETS[@]}" \
  -q -p no:cacheprovider \
  --junitxml="artifacts/test-results/failure-${CASE}.xml" \
  "$@"
