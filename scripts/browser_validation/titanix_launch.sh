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

# Deviation D8: model weights (*.eqx.zst etc.) are Git LFS files. A push by URL carries only
# the 132-byte pointers and titanix has no git-lfs, so the checkout would hold pointers, not
# weights. Ship every LFS object referenced at H into a content-addressed store on titanix
# (same aa/bb/<oid> layout as .git/lfs/objects); the remote step below materializes them.
TX_LFS_STORE="/home/solab/bv/lfs-objects"
LFS_DIR="$(git rev-parse --git-common-dir)/lfs/objects"
lfs_list="$(mktemp)"
git lfs ls-files -l "$H" | awk '{print $1}' | while read -r oid; do
  if [ ! -f "${LFS_DIR}/${oid:0:2}/${oid:2:2}/${oid}" ]; then
    echo "titanix_launch.sh: LFS object ${oid} referenced at ${H} is missing locally (git lfs fetch)" >&2
    exit 1
  fi
  echo "${oid:0:2}/${oid:2:2}/${oid}"
done > "$lfs_list"
ssh titanix mkdir -p "$TX_LFS_STORE"
rsync -a --files-from="$lfs_list" "${LFS_DIR}/" "titanix:${TX_LFS_STORE}/"
rm -f "$lfs_list"

ssh titanix bash -s -- "$RD" "$H" "$TX_LFS_STORE" <<'REMOTE'
set -euo pipefail
RD="$1"
H="$2"
STORE="$3"
avail=$(df -B1G --output=avail /home/solab | tail -1 | tr -d ' ')
if [ "$avail" -lt 30 ]; then
  echo "titanix_launch.sh (remote): only ${avail}G free under /home/solab (need >=30G); refusing" >&2
  exit 1
fi
git -C /home/solab/bv/aminx.git worktree add --detach "$RD" "$H"
cd "$RD"
# D8: replace each LFS pointer with its object after checking sha256 == the committed oid
# (exactly what the git-lfs smudge filter does), then mark it skip-worktree so the clean-tree
# checks compare against the pointer blob git actually committed, not the smudged bytes.
n_lfs=0
while IFS= read -r -d '' path; do
  [ "$(head -c 40 "$path")" = "version https://git-lfs.github.com/spec/" ] || continue
  oid="$(sed -n 's/^oid sha256:\([0-9a-f]\{64\}\)$/\1/p' "$path")"
  obj="${STORE}/${oid:0:2}/${oid:2:2}/${oid}"
  if [ ! -f "$obj" ]; then
    echo "titanix_launch.sh (remote): LFS object ${oid} for ${path} not in ${STORE}" >&2
    exit 1
  fi
  if [ "$(sha256sum "$obj" | cut -d' ' -f1)" != "$oid" ]; then
    echo "titanix_launch.sh (remote): LFS object ${oid} for ${path} fails its sha256" >&2
    exit 1
  fi
  cp "$obj" "$path"
  git update-index --skip-worktree -- "$path"
  n_lfs=$((n_lfs + 1))
done < <(git ls-files -z -- '*.eqx' '*.eqx.zst' '*.npz' '*.array_record' '*.tar.gz')
echo "titanix_launch.sh (remote): materialized ${n_lfs} LFS object(s)"
if [ -n "$(git status --porcelain)" ]; then
  echo "titanix_launch.sh (remote): worktree dirty after LFS materialization" >&2
  exit 1
fi
/home/solab/.local/bin/uv sync --frozen --extra dev --extra benchmark
REMOTE

# A transient systemd --user unit, detached from this ssh session (F-C7: this is the
# ODQ-13 mechanism -- tmux is NOT installed on titanix). --collect lets a finished unit
# be garbage-collected; the orchestrator's poll (T4 common context step 5) treats that
# as terminal via `systemctl --user is-active`, not via unit persistence.
ssh titanix systemd-run --user "--unit=${SESSION}" --collect -p MemoryMax=64G -p MemorySwapMax=0 \
  "${RD}/scripts/browser_validation/titanix_run.sh" "$RD" "$H" "$CAMPAIGN_ID" "$STEM" "${SCRIPT_ARGS[@]}"

echo "$SESSION"
