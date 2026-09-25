#!/usr/bin/env bash
# titanix_launch.sh -- push a layer-(a) bathos run to titanix and dispatch it inside a
# transient systemd --user unit (O6', T4 common context, ODQ-13/F-C7).
#
# ORCHESTRATOR-ONLY in normal operation (T4 fixer step 5 is explicitly out of scope for
# the fixer that wrote this file); the fixer's own gate only syntax-checks this script
# and runs its --self-test mode, which touches no network.
#
# Usage:
#   titanix_launch.sh <stem> <campaign-id> [SCRIPT_ARGS...]
#   titanix_launch.sh --self-test      # verify the O9 argv-safety check in isolation
#
# Deviation #1 (orchestrator-approved, 260924): .git/config is write-protected in this
# worktree, so `git remote add titanix-bv ...` is impossible. Push by URL instead --
# there is no named remote anywhere in this script.
TX_REPO_URL="titanix:/home/solab/bv/aminx.git"

set -euo pipefail

# O9: ssh joins the remote command's argv with spaces exactly as `bth submit` used to,
# so an argument containing whitespace or a shell metacharacter would be silently
# re-split/re-interpreted on the remote end. Refuse (exit 2) rather than risk that.
check_arg_safe() {
  local arg="$1"
  case "$arg" in
    *[' 	'\;\&\|\<\>\$\`\\\"\'\(\)\{\}\*\?\!\#\~]*)
      echo "titanix_launch.sh: argument '$arg' contains whitespace or a shell metacharacter (O9); refusing" >&2
      return 2
      ;;
  esac
  return 0
}

self_test() {
  local failures=0
  if check_arg_safe "a b"; then
    echo "self-test FAILED: 'a b' should have been rejected (exit 2)" >&2
    failures=$((failures + 1))
  fi
  if check_arg_safe "a;b"; then
    echo "self-test FAILED: 'a;b' should have been rejected (exit 2)" >&2
    failures=$((failures + 1))
  fi
  if [ "$failures" -eq 0 ]; then
    echo "self-test OK: 'a b' and 'a;b' both rejected with exit 2 by check_arg_safe"
    return 0
  fi
  return 1
}

usage() {
  cat >&2 <<'EOF'
Usage: titanix_launch.sh <stem> <campaign-id> [SCRIPT_ARGS...]
       titanix_launch.sh --self-test
EOF
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

if [ "$#" -lt 2 ]; then
  usage
  exit 2
fi

STEM="$1"
CAMPAIGN_ID="$2"
shift 2
SCRIPT_ARGS=("$@")

for arg in "$STEM" "$CAMPAIGN_ID" "${SCRIPT_ARGS[@]}"; do
  check_arg_safe "$arg" || exit 2
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# F-C3/deviation #2: the worktree carries many untracked harness files (.praxia/*,
# .claude/*, .mcp.json, .bth/refs/, *.ses) that must never be committed or deleted, so
# a bare `git status --porcelain` (as titanix_run.sh's OWN clean-tree check still uses,
# on its fresh checkout) would never pass here. `--untracked-files=no` handles tracked
# dirt; folding in a `git ls-files --others` scoped to the actual code paths closes the
# remaining gap -- an untracked file under scripts/src/pyproject.toml/uv.lock still
# blocks the push, because a titanix run must reproduce from git alone.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "titanix_launch.sh: tracked files are dirty; commit or revert before dispatching" >&2
  exit 1
fi
if [ -n "$(git ls-files --others --exclude-standard -- scripts src pyproject.toml uv.lock)" ]; then
  echo "titanix_launch.sh: untracked files exist under scripts/src/pyproject.toml/uv.lock; commit or remove before dispatching" >&2
  exit 1
fi

H="$(git rev-parse HEAD)"
SESSION="bv-${STEM}-${H:0:12}"

# F-C3: one heavy run at a time on titanix.
out=$(ssh titanix systemctl --user list-units 'bv-*' --state=active,activating --no-legend --plain)
if [ -n "$out" ]; then
  echo "titanix_launch.sh: a bv-* unit is already active/activating on titanix; refusing a second concurrent run:" >&2
  echo "$out" >&2
  exit 1
fi

git push "$TX_REPO_URL" "$H:refs/heads/bv/${STEM}-${H:0:12}"

RD="/home/solab/bv/aminx-browser-validation-${STEM}-${H:0:12}"

ssh titanix bash -s -- "$RD" "$H" <<'REMOTE'
set -euo pipefail
RD="$1"
H="$2"
avail=$(df -B1G --output=avail /home/solab | tail -1 | tr -d ' ')
if [ "$avail" -lt 30 ]; then
  echo "titanix_launch.sh (remote): only ${avail}G free under /home/solab (need >=30G); refusing" >&2
  exit 1
fi
git -C /home/solab/bv/aminx.git worktree add --detach "$RD" "$H"
cd "$RD"
/home/solab/.local/bin/uv sync --frozen --extra dev --extra benchmark
REMOTE

# A transient systemd --user unit, detached from this ssh session (F-C7: this is the
# ODQ-13 mechanism -- tmux is NOT installed on titanix). --collect lets a finished unit
# be garbage-collected; the orchestrator's poll (T4 common context step 5) treats that
# as terminal via `systemctl --user is-active`, not via unit persistence.
ssh titanix systemd-run --user "--unit=${SESSION}" --collect -p MemoryMax=64G -p MemorySwapMax=0 \
  "${RD}/scripts/browser_validation/titanix_run.sh" "$RD" "$H" "$CAMPAIGN_ID" "$STEM" "${SCRIPT_ARGS[@]}"

echo "$SESSION"
