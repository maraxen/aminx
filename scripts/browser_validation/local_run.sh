#!/usr/bin/env bash
# local_run.sh -- run a bathos-tracked browser-validation script from a clean, DETACHED
# local git worktree, never mutating $WT itself (T2, common context "Local bathos runs").
#
# Usage:
#   local_run.sh --stem STEM --campaign CAMPAIGN_ID --subdir SUBDIR [--npm-dir DIR] \
#     [--with=... | --extra=... | --prerelease=...]... -- [SCRIPT_ARGS...]
#   local_run.sh --self-test
#
# On success, prints "RUN_ID=<id> H=<sha>" as its LAST line (step 9); a caller does
# `eval "$(bash local_run.sh ... | tail -1)"` to pick up $RUN_ID/$H.
#
# Every local bathos interaction takes ONE lock (r1 B1): /tmp/bv-local.lock,
# `flock -w 10800` (3h). This script holds it for its whole body.

set -euo pipefail

LOCK=/tmp/bv-local.lock
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BV_REL="scripts/browser_validation"

# --------------------------------------------------------------------------------------
# Dirty-tree check (Common context step 1; Hard rule: never touch/inspect the worktree's
# pre-existing tool-log dirt -- .praxia/audits.jsonl, .praxia/recon.jsonl -- so BOTH the
# tracked-dirty check and the untracked check are scoped to code paths only, same paths
# titanix_launch.sh:112-119 scopes its OWN untracked check to).
# --------------------------------------------------------------------------------------
refuse_if_dirty() {
  local root="$1"
  if [ -n "$(cd "$root" && git status --porcelain --untracked-files=no -- scripts src pyproject.toml uv.lock)" ]; then
    echo "local_run.sh: tracked files under scripts/src/pyproject.toml/uv.lock are dirty in $root" >&2
    return 1
  fi
  if [ -n "$(cd "$root" && git ls-files --others --exclude-standard -- scripts src pyproject.toml uv.lock)" ]; then
    echo "local_run.sh: untracked files under scripts/src/pyproject.toml/uv.lock in $root" >&2
    return 1
  fi
  return 0
}

self_test() {
  local scratch rc
  scratch="$(mktemp -d)"
  git -C "$WT" worktree add --detach "$scratch" HEAD >/dev/null 2>&1
  echo "# local_run.sh self-test dirt $$" >>"$scratch/${BV_REL}/__init__.py"
  set +e
  refuse_if_dirty "$scratch"
  rc=$?
  set -e
  git -C "$WT" worktree remove --force "$scratch" >/dev/null 2>&1 || rm -rf "$scratch"
  if [ "$rc" -ne 0 ]; then
    echo "self-test OK: planted tracked-dirty file was correctly refused (exit != 0)"
    return 0
  fi
  echo "self-test FAILED: planted tracked-dirty file was NOT refused" >&2
  return 1
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

usage() {
  cat >&2 <<'EOF'
Usage: local_run.sh --stem STEM --campaign CAMPAIGN_ID --subdir SUBDIR [--npm-dir DIR]
                     [UV_WITH_TOKENS...] -- [SCRIPT_ARGS...]
       local_run.sh --self-test
EOF
}

STEM=""
CAMPAIGN=""
SUBDIR=""
NPM_DIR=""
WITH_TOKENS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stem) STEM="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --subdir) SUBDIR="$2"; shift 2 ;;
    --npm-dir) NPM_DIR="$2"; shift 2 ;;
    --) shift; break ;;
    --with=* | --extra=* | --prerelease=*) WITH_TOKENS+=("$1"); shift ;;
    *)
      echo "local_run.sh: unrecognized argument '$1'" >&2
      usage
      exit 2
      ;;
  esac
done
SCRIPT_ARGS=("$@")

if [ -z "$STEM" ] || [ -z "$CAMPAIGN" ] || [ -z "$SUBDIR" ]; then
  usage
  exit 2
fi

exec 9>"$LOCK"
if ! flock -w 10800 9; then
  echo "local_run.sh: could not acquire $LOCK within 10800s" >&2
  exit 4
fi

cd "$WT"
refuse_if_dirty "$WT" || exit 1

H="$(git rev-parse HEAD)"
H12="${H:0:12}"
RUN_WT="/tmp/bv-run-${STEM}-${H12}"
rm -rf "$RUN_WT"
git worktree add --detach "$RUN_WT" "$H" >/dev/null

cleanup() {
  git worktree remove --force "$RUN_WT" >/dev/null 2>&1 || rm -rf "$RUN_WT"
}
trap cleanup EXIT

OUT_REL="outputs/browser_validation/${SUBDIR}/${STEM}.json"
DEST="$WT/outputs/browser_validation/${SUBDIR}/runs/${STEM}-${H12}"
RUN_LOG="$(mktemp)"

set +e
(
  set -euo pipefail
  cd "$RUN_WT"

  uv sync --frozen --extra=dev --extra=benchmark
  if [ -n "$NPM_DIR" ]; then
    ( cd "$NPM_DIR" && npm ci )
  fi

  if [ -n "$(git status --porcelain)" ]; then
    echo "local_run.sh: detached worktree $RUN_WT is dirty after sync (git_dirty would be true)" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$OUT_REL")"

  env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    bth run --campaign-id "$CAMPAIGN" --output-paths "$OUT_REL" -- \
    uv run --frozen --no-sync --extra=dev --extra=benchmark "${WITH_TOKENS[@]}" python \
    "${BV_REL}/${STEM}.py" --out "$OUT_REL" "${SCRIPT_ARGS[@]}"
) 2>&1 | tee "$RUN_LOG"
RUN_RC="${PIPESTATUS[0]}"
set -e

# `bth run`'s own stdout is `json.dumps({"script_path", "run_id", "exit_code", "success"},
# indent=2)` (bathos mcp.py `run_cli_tool`); pull the run id off its own `"run_id": "..."`
# line in the captured combined log (last occurrence, in case the wrapped script's own
# logging also happens to contain that literal string).
RUN_ID="$(grep -o '"run_id": *"[^"]*"' "$RUN_LOG" | tail -1 | sed -E 's/.*"run_id": *"([^"]*)".*/\1/')"
rm -f "$RUN_LOG"

mkdir -p "$DEST"
if [ -d "$RUN_WT/outputs/browser_validation/${SUBDIR}" ]; then
  cp -a "$RUN_WT/outputs/browser_validation/${SUBDIR}/." "$DEST/" 2>/dev/null || true
fi

# bth provenance (common context "bth provenance", r1 M4): the INSTALLED bth tool's code
# identity, independent of any --with=bathos overlay a script imports.
BATHOS_TOOL_DIR="$(ls -d "$HOME"/.local/share/uv/tools/bathos/lib/python3*/site-packages/bathos 2>/dev/null | head -1)"
{
  echo "{"
  if [ -n "$BATHOS_TOOL_DIR" ]; then
    for f in sidecar.py mcp.py cli.py; do
      if [ -f "$BATHOS_TOOL_DIR/$f" ]; then
        sha="$(sha256sum "$BATHOS_TOOL_DIR/$f" | cut -d' ' -f1)"
        echo "  \"${f}_sha256\": \"${sha}\","
      fi
    done
    mcp_guard="false"
    grep -q 'endswith(".py")' "$BATHOS_TOOL_DIR/mcp.py" 2>/dev/null && mcp_guard="true"
    echo "  \"mcp_wrapped_command_guard_present\": ${mcp_guard},"
    evaluate_outcome_line="$(grep -n 'def evaluate_outcome' "$BATHOS_TOOL_DIR/sidecar.py" 2>/dev/null | head -1 | cut -d: -f1)"
    echo "  \"sidecar_evaluate_outcome_line\": \"${evaluate_outcome_line:-unknown}\","
    echo "  \"bathos_tool_dir\": \"${BATHOS_TOOL_DIR}\""
  else
    echo "  \"bathos_tool_dir\": null"
  fi
  echo "}"
} >"$DEST/bth_provenance.json"

echo "local_run.sh: stem=${STEM} campaign=${CAMPAIGN} subdir=${SUBDIR} exit=${RUN_RC} dest=${DEST}"
echo "RUN_ID=${RUN_ID} H=${H}"
exit "$RUN_RC"
