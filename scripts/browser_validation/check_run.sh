#!/usr/bin/env bash
# check_run.sh -- verify a bathos run BY ITS RECORD, never by exit code or console text
# (T2, common context "Run verification").
#
# Usage:
#   check_run.sh --stem S --hash H --sidecar PATH --allowed a,b,c [--run-id R]
#                [--sync-remote titanix]
#   check_run.sh --self-test-lock
#
# Exit codes (uniform across the `bth sql` path and the Parquet fallback):
#   0 = ok, 1 = an assertion failed (including row count > 1), 4 = lock timeout,
#   6 = no matching row (a missing row is a failure, never a pass).

set -euo pipefail

LOCK=/tmp/bv-local.lock
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

acquire_lock() {
  # Acquires $LOCK on fd 9 with a wait of "$1" seconds, or exits 4.
  local wait_s="$1"
  exec 9>"$LOCK"
  if ! flock -w "$wait_s" 9; then
    echo "check_run.sh: could not acquire $LOCK within ${wait_s}s" >&2
    exit 4
  fi
}

if [ "${1:-}" = "--self-test-lock" ]; then
  # Hold the lock in a background process, then confirm our OWN acquisition attempt
  # (wait 1s) is refused with exit 4 while it is held.
  flock "$LOCK" sleep 30 &
  holder_pid=$!
  # Give the background flock a moment to actually acquire before we race it.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [ -e "/proc/$holder_pid" ]; then
      break
    fi
    sleep 0.1
  done
  sleep 0.3

  set +e
  ( acquire_lock 1 )
  rc=$?
  set -e

  kill "$holder_pid" 2>/dev/null || true
  wait "$holder_pid" 2>/dev/null || true

  if [ "$rc" -eq 4 ]; then
    echo "self-test-lock OK: exit 4 while /tmp/bv-local.lock is held (waited 1s)"
    exit 0
  fi
  echo "self-test-lock FAILED: expected exit 4 while lock held, got $rc" >&2
  exit 1
fi

usage() {
  cat >&2 <<'EOF'
Usage: check_run.sh --stem S --hash H --sidecar PATH --allowed a,b,c
                     [--run-id R] [--sync-remote REMOTE]
       check_run.sh --self-test-lock
EOF
}

STEM=""
HASH=""
SIDECAR=""
ALLOWED=""
RUN_ID=""
SYNC_REMOTE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stem) STEM="$2"; shift 2 ;;
    --hash) HASH="$2"; shift 2 ;;
    --sidecar) SIDECAR="$2"; shift 2 ;;
    --allowed) ALLOWED="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --sync-remote) SYNC_REMOTE="$2"; shift 2 ;;
    *)
      echo "check_run.sh: unrecognized argument '$1'" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$STEM" ] || [ -z "$HASH" ] || [ -z "$SIDECAR" ] || [ -z "$ALLOWED" ]; then
  usage
  exit 2
fi

cd "$WT"

acquire_lock 10800

if [ -n "$SYNC_REMOTE" ]; then
  bth sync --pull --remote-name "$SYNC_REMOTE"
fi

# Plain form, NEVER --force-rebuild (Hard rule).
bth compact

expected_sha="$(git show "${HASH}:${SIDECAR}" | sha256sum | cut -d' ' -f1)"

fail_assert() {
  echo "check_run.sh: $1" >&2
  exit 1
}

evaluate_row() {
  # $1 = json response ({"rows": [...], "count": N}); $2 = "sql" or "parquet" (for the
  # printed source= line only).
  local resp="$1" source_label="$2"
  local count
  count="$(echo "$resp" | jq -r '.count')"

  if [ "$count" -eq 0 ]; then
    echo "check_run.sh: no row found for stem=${STEM} hash=${HASH} (source=${source_label})" >&2
    exit 6
  fi
  if [ "$count" -ne 1 ]; then
    fail_assert "expected exactly 1 row, got ${count} (duplicate rows, source=${source_label})"
  fi

  local row id outcome git_dirty sidecar_sha256 exit_code
  row="$(echo "$resp" | jq -c '.rows[0]')"
  id="$(echo "$row" | jq -r '.[0]')"
  outcome="$(echo "$row" | jq -r '.[1]')"
  git_dirty="$(echo "$row" | jq -r '.[2]')"
  sidecar_sha256="$(echo "$row" | jq -r '.[3]')"
  exit_code="$(echo "$row" | jq -r '.[4]')"

  [ "$git_dirty" = "false" ] || fail_assert "git_dirty=${git_dirty}, expected false (id=${id})"
  { [ -n "$sidecar_sha256" ] && [ "$sidecar_sha256" != "null" ]; } \
    || fail_assert "sidecar_sha256 is empty/null (id=${id})"
  [ "$sidecar_sha256" = "$expected_sha" ] \
    || fail_assert "sidecar_sha256=${sidecar_sha256} != sha256(git show ${HASH}:${SIDECAR})=${expected_sha} (id=${id})"
  [ "$outcome" != "error" ] || fail_assert "outcome=error (id=${id}, exit_code=${exit_code})"

  local ok=1
  IFS=',' read -ra allowed_arr <<<"$ALLOWED"
  for a in "${allowed_arr[@]}"; do
    if [ "$a" = "$outcome" ]; then
      ok=0
      break
    fi
  done
  [ "$ok" -eq 0 ] || fail_assert "outcome=${outcome} not in allowed set {${ALLOWED}} (id=${id})"

  echo "check_run.sh: OK id=${id} outcome=${outcome} git_dirty=${git_dirty} sidecar_sha256=${sidecar_sha256} exit_code=${exit_code} source=${source_label}"
}

sql="SELECT id, outcome, git_dirty, sidecar_sha256, exit_code FROM runs WHERE command LIKE '%${STEM}.py --out%' AND git_hash = '${HASH}'"
set +e
resp="$(bth sql --sql "$sql" 2>/tmp/check_run_sql_err.$$)"
sql_rc=$?
set -e

if [ "$sql_rc" -eq 0 ]; then
  rm -f "/tmp/check_run_sql_err.$$"
  evaluate_row "$resp" "sql"
  exit 0
fi

echo "check_run.sh: 'bth sql' failed (locked/stale warm index?); falling back to Parquet" >&2
cat "/tmp/check_run_sql_err.$$" >&2 2>/dev/null || true
rm -f "/tmp/check_run_sql_err.$$"

if [ -z "$RUN_ID" ]; then
  echo "check_run.sh: no --run-id given; cannot locate the cool-tier Parquet fragment" >&2
  exit 1
fi

PARQUET="$HOME/.bth/catalog/runs/aminx/run_${RUN_ID}.parquet"
if [ ! -f "$PARQUET" ]; then
  echo "check_run.sh: no row found -- ${PARQUET} does not exist (source=parquet)" >&2
  exit 6
fi

resp="$(uv run --frozen --with=duckdb python3 -c "
import duckdb, json, sys
con = duckdb.connect()
rows = con.execute('''
    SELECT id, outcome, git_dirty, sidecar_sha256, exit_code
    FROM read_parquet(?)
    WHERE command LIKE ? AND git_hash = ?
''', ['${PARQUET}', '%${STEM}.py --out%', '${HASH}']).fetchall()
print(json.dumps({'rows': [list(r) for r in rows], 'count': len(rows)}))
")"
evaluate_row "$resp" "parquet"
