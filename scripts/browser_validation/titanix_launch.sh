#!/usr/bin/env bash
# titanix_launch.sh -- push a layer-(a) bathos run to titanix and dispatch it inside a
# transient systemd --user unit (O6', T4 common context, ODQ-13/F-C7).
#
# ORCHESTRATOR-ONLY in normal operation (T4 fixer step 5 is explicitly out of scope for
# the fixer that wrote this file); the fixer's own gate only syntax-checks this script
# and runs its --self-test mode, which touches no network.
#
# Usage:
#   titanix_launch.sh [--gpu N] [--tag TAG] [--prepare-only] <stem> <campaign-id> [SCRIPT_ARGS...]
#   titanix_launch.sh --self-test      # O9 argv-safety plus --tag session names; no network
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

# --tag is appended so two shards of one stem at one commit get distinct units, logs,
# .exit files, result paths, and remote worktrees. With no tag the historical names
# are unchanged. titanix_run.sh honours BV_SESSION / BV_OUT_REL only when this script
# exports them (tagged launches).
session_name() {
  local stem="$1" hash="$2" tag="${3:-}"
  local session="bv-${stem}-${hash:0:12}"
  if [ -n "$tag" ]; then
    session="${session}-${tag}"
  fi
  printf '%s\n' "$session"
}

checkout_dir() {
  local stem="$1" hash="$2" tag="${3:-}"
  local rd="/home/solab/bv/aminx-browser-validation-${stem}-${hash:0:12}"
  if [ -n "$tag" ]; then
    rd="${rd}-${tag}"
  fi
  printf '%s\n' "$rd"
}

result_relpath() {
  local stem="$1" tag="${2:-}"
  if [ -n "$tag" ]; then
    printf 'outputs/browser_validation/layer_a/%s-%s.json\n' "$stem" "$tag"
  else
    printf 'outputs/browser_validation/layer_a/%s.json\n' "$stem"
  fi
}

tag_valid() {
  case "$1" in
    ""|*[!a-z0-9-]*) return 1 ;;
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
  untagged="$(session_name "layer_a_sampling_validate" "abcdef0123456789ffff")"
  shard0="$(session_name "layer_a_sampling_validate" "abcdef0123456789ffff" "shard-0")"
  shard1="$(session_name "layer_a_sampling_validate" "abcdef0123456789ffff" "shard-1")"
  if [ "$untagged" != "bv-layer_a_sampling_validate-abcdef012345" ]; then
    echo "self-test FAILED: untagged session is '$untagged'" >&2
    failures=$((failures + 1))
  fi
  if [ "$shard0" = "$untagged" ] || [ "$shard0" = "$shard1" ]; then
    echo "self-test FAILED: tagged sessions are not distinct from each other and from untagged" >&2
    failures=$((failures + 1))
  fi
  if [ "${shard0}.log" = "${shard1}.log" ] || [ "${shard0}.exit" = "${shard1}.exit" ]; then
    echo "self-test FAILED: tagged log or .exit names collide" >&2
    failures=$((failures + 1))
  fi
  plain_out="$(result_relpath "layer_a_sampling_validate")"
  tag_out="$(result_relpath "layer_a_sampling_validate" "shard-0")"
  other_out="$(result_relpath "layer_a_sampling_validate" "shard-1")"
  if [ "$plain_out" != "outputs/browser_validation/layer_a/layer_a_sampling_validate.json" ]; then
    echo "self-test FAILED: untagged result path is '$plain_out'" >&2
    failures=$((failures + 1))
  fi
  if [ "$tag_out" = "$plain_out" ] || [ "$tag_out" = "$other_out" ]; then
    echo "self-test FAILED: tagged result paths are not distinct" >&2
    failures=$((failures + 1))
  fi
  plain_rd="$(checkout_dir "layer_a_sampling_validate" "abcdef0123456789ffff")"
  tag_rd="$(checkout_dir "layer_a_sampling_validate" "abcdef0123456789ffff" "shard-0")"
  other_rd="$(checkout_dir "layer_a_sampling_validate" "abcdef0123456789ffff" "shard-1")"
  expected_rd="/home/solab/bv/aminx-browser-validation-layer_a_sampling_validate-abcdef012345"
  if [ "$plain_rd" != "$expected_rd" ]; then
    echo "self-test FAILED: untagged checkout is '$plain_rd'" >&2
    failures=$((failures + 1))
  fi
  if [ "$tag_rd" = "$plain_rd" ] || [ "$tag_rd" = "$other_rd" ]; then
    echo "self-test FAILED: tagged checkouts are not distinct" >&2
    failures=$((failures + 1))
  fi
  if ! tag_valid "shard-0"; then
    echo "self-test FAILED: 'shard-0' should be a valid tag" >&2
    failures=$((failures + 1))
  fi
  if tag_valid "Shard" || tag_valid "" || tag_valid "a_b"; then
    echo "self-test FAILED: invalid tags were accepted" >&2
    failures=$((failures + 1))
  fi
  if [ "$failures" -eq 0 ]; then
    echo "self-test OK: metacharacters rejected; --tag session/log/exit/result/checkout names distinct"
    return 0
  fi
  return 1
}

usage() {
  cat >&2 <<'EOF'
Usage: titanix_launch.sh [--gpu N] [--tag TAG] [--prepare-only] <stem> <campaign-id> [SCRIPT_ARGS...]
       titanix_launch.sh --self-test
EOF
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

# Deviation D10 (user-approved 260925): the sampling tier (T8) runs its aminx draws on ONE
# titanix GPU -- aminx AR draws cost ~2 s each on CPU (the kernel recomputes the decoder
# every step), which puts the pre-registered protocol far past the 16 h CPU budget.
# `--gpu N` exposes only GPU N to the run (titanix's other GPUs serve vLLM and must stay
# untouched) and syncs the lock's `cuda12` extra. `--prepare-only` stops after the checkout
# is materialized and synced (no unit is started) and prints its path -- for probes.
GPU=""
TAG=""
PREPARE_ONLY=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu)
      case "${2:-}" in
        [0-9]) GPU="$2" ;;
        *) echo "titanix_launch.sh: --gpu needs a single GPU index" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      if ! tag_valid "$TAG"; then
        echo "titanix_launch.sh: --tag must match [a-z0-9-]+" >&2
        exit 2
      fi
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    *) break ;;
  esac
done

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
SESSION="$(session_name "$STEM" "$H" "$TAG")"
# Per-tag worktrees, not one shared checkout under a lock. Two shards of the same
# commit would otherwise race on `git worktree add` of one path and on `uv sync`
# in one venv. Distinct directories do not share an index; git's own lock serializes
# the bare repo. The push ref stays the untagged commit ref (same object either way).
RD="$(checkout_dir "$STEM" "$H" "$TAG")"

# F-C3: one heavy run at a time on titanix when --tag is absent (historical behaviour).
# A tagged launch is one shard of a parallel pair: refuse only that same unit name.
if [ -z "$TAG" ]; then
  out=$(ssh titanix systemctl --user list-units 'bv-*' --state=active,activating --no-legend --plain)
  if [ -n "$out" ]; then
    echo "titanix_launch.sh: a bv-* unit is already active/activating on titanix; refusing a second concurrent run:" >&2
    echo "$out" >&2
    exit 1
  fi
else
  unit_state="$(ssh titanix systemctl --user is-active "$SESSION" || true)"
  if [ "$unit_state" = "active" ] || [ "$unit_state" = "activating" ]; then
    echo "titanix_launch.sh: unit ${SESSION} is already active/activating on titanix; refusing a second launch of the same tag" >&2
    exit 1
  fi
fi

git push "$TX_REPO_URL" "$H:refs/heads/bv/${STEM}-${H:0:12}"

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

ssh titanix bash -s -- "$RD" "$H" "$TX_LFS_STORE" "${GPU:-none}" <<'REMOTE'
set -euo pipefail
RD="$1"
H="$2"
STORE="$3"
GPU="$4"
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
  # model_params/* are tracked symlinks into src/aminx/model_params/; following one would
  # smudge the target but flag the symlink, leaving the real file reported as modified.
  [ -L "$path" ] && continue
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
extras=(--extra dev --extra benchmark)
[ "$GPU" != "none" ] && extras+=(--extra cuda12)
/home/solab/.local/bin/uv sync --frozen "${extras[@]}"
REMOTE

if [ "$PREPARE_ONLY" -eq 1 ]; then
  echo "$RD"
  exit 0
fi

# A transient systemd --user unit, detached from this ssh session (F-C7: this is the
# ODQ-13 mechanism -- tmux is NOT installed on titanix). --collect lets a finished unit
# be garbage-collected; the orchestrator's poll (T4 common context step 5) treats that
# as terminal via `systemctl --user is-active`, not via unit persistence.
gpu_env=()
[ -n "$GPU" ] && gpu_env=("--setenv=BV_GPU=${GPU}")
tag_env=()
if [ -n "$TAG" ]; then
  tag_env=("--setenv=BV_SESSION=${SESSION}" "--setenv=BV_OUT_REL=$(result_relpath "$STEM" "$TAG")")
fi
ssh titanix systemd-run --user "--unit=${SESSION}" --collect -p MemoryMax=64G -p MemorySwapMax=0 "${gpu_env[@]}" "${tag_env[@]}" \
  "${RD}/scripts/browser_validation/titanix_run.sh" "$RD" "$H" "$CAMPAIGN_ID" "$STEM" "${SCRIPT_ARGS[@]}"

echo "$SESSION"
