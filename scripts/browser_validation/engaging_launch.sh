#!/usr/bin/env bash
# engaging_launch.sh -- push a layer-(a) bathos run to Engaging (MIT Slurm cluster) and
# submit it as an sbatch job. Twin of titanix_launch.sh (T10f), same guarantees, adapted
# to a Slurm-managed multi-tenant login/compute split instead of a titanix systemd unit:
#   * no `git push` anywhere -- Engaging is never authorised as a push target (only
#     titanix is, per titanix_launch.sh's own deviation #1); a git BUNDLE is rsynced up
#     and fetched into a bare repo on the remote instead.
#   * the remote home directory is NOT hardcoded (titanix hardcodes /home/solab/bv
#     throughout; Engaging's remote home is queried, never assumed, and every remote-side
#     command expands "$HOME" on the remote shell, never locally).
#   * dispatch is `sbatch`, not a systemd --user transient unit (no tmux/systemd-run
#     equivalent needed -- Slurm's job record IS the durable state).
#   * no uv/python on the LOGIN node (user rule, PreToolUse-hook enforced): the sbatch
#     script runs `uv sync --frozen` on the compute node (measured 260929, quicktest job
#     24334493: compute nodes reach GitHub/PyPI), then `uv run --frozen --no-sync`.
#
# ORCHESTRATOR-ONLY in normal operation, exactly like titanix_launch.sh: the fixer that
# wrote this file only syntax-checks it and runs its --self-test mode, which touches no
# network.
#
# Usage:
#   engaging_launch.sh [--tag TAG] [--time HH:MM:SS] [--prepare-only] <stem> <campaign-id> [SCRIPT_ARGS...]
#   engaging_launch.sh --self-test      # O9 argv-safety, --tag validation, session naming,
#                                        # and the sbatch command that WOULD run; no network
#
# One-time Engaging staging this script assumes already exists (see the README section
# this header also duplicates for readers who land here first):
#   $HOME/bv/ref/ProteinMPNN, $HOME/bv/ref/LigandMPNN   -- checked out at the commits
#       pinned in reference_pins.json (ligandmpnn_commit, proteinmpnn_commit), weights
#       already materialized (not LFS pointers).
#   $HOME/bv/bth-84be544e/                               -- a venv with bathos installed
#       at commit 84be544ecb45734f46e43d22f351a54d6edd6ae5 (BATHOS_COMMIT_PIN below),
#       bin/bth on that venv's PATH.
#   $HOME/bv/logs/                                       -- sbatch --output target dir.
#   $HOME/bv/catalog-aminx/                              -- BTH_CATALOG_DIR parent.
#   $HOME/.local/bin/uv                                  -- project uv binary (used ONLY
#       inside the sbatch job; never on the login node).
#
# Remote host alias: `engaging`. Remote root: $HOME/bv (expanded remotely; the remote
# home is NEVER hardcoded here, even though it is known out-of-band to be /home/maarxaru).

set -euo pipefail

# --- O9 (identical to titanix_launch.sh): ssh/sbatch join argv with spaces the same way
# `bth submit` used to, so an argument containing whitespace or a shell metacharacter
# would be silently re-split/re-interpreted downstream. Refuse (exit 2) rather than risk
# it.
check_arg_safe() {
  local arg="$1"
  case "$arg" in
    *[' 	'\;\&\|\<\>\$\`\\\"\'\(\)\{\}\*\?\!\#\~]*)
      echo "engaging_launch.sh: argument '$arg' contains whitespace or a shell metacharacter (O9); refusing" >&2
      return 2
      ;;
  esac
  return 0
}

tag_valid() {
  case "$1" in
    ""|*[!a-z0-9-]*) return 1 ;;
  esac
  return 0
}

# Session name: bv-<stem>-<h12>[-<tag>], identical scheme to titanix_launch.sh. This is
# also the Slurm --job-name, the .log/.exit basename, and (via checkout_dir) the remote
# worktree directory name -- so two shards of one stem at one commit get distinct
# everything, and an untagged launch reproduces the historical, tag-free name.
session_name() {
  local stem="$1" hash="$2" tag="${3:-}"
  local session="bv-${stem}-${hash:0:12}"
  if [ -n "$tag" ]; then
    session="${session}-${tag}"
  fi
  printf '%s\n' "$session"
}

# Remote worktree directory, relative to the remote $HOME (the caller prefixes
# "$HOME/bv/" -- kept relative here so self-test needs no remote-home knowledge).
checkout_dir_rel() {
  local session="$1"
  printf 'bv/aminx-%s\n' "$session"
}

result_relpath() {
  local stem="$1" tag="${2:-}"
  if [ -n "$tag" ]; then
    printf 'outputs/browser_validation/layer_a/%s-%s.json\n' "$stem" "$tag"
  else
    printf 'outputs/browser_validation/layer_a/%s.json\n' "$stem"
  fi
}

# Builds the exact `ssh engaging sbatch ...` argv (as a printable, shell-quoted string)
# for a given set of resolved values. Used both by the real dispatch path and by
# --self-test (with dummy values, never executed there).
build_sbatch_cmd() {
  local session="$1" h="$2" campaign_id="$3" stem="$4" tag="$5" time_limit="$6"
  shift 6
  local out_rel export_val rd_rel
  out_rel="$(result_relpath "$stem" "$tag")"
  rd_rel="$(checkout_dir_rel "$session")"
  export_val="ALL,BV_SESSION=${session},BV_OUT_REL=${out_rel},BV_RD=\$HOME/${rd_rel},BV_H=${h},BV_CAMPAIGN_ID=${campaign_id},BV_STEM=${stem}"
  printf 'ssh engaging sbatch --job-name=%s --output=$HOME/bv/logs/%s.log --time=%s --export=%s $HOME/%s/scripts/browser_validation/engaging_run.sbatch' \
    "$session" "$session" "$time_limit" "$export_val" "$rd_rel"
  for a in "$@"; do
    printf ' %s' "$a"
  done
  printf '\n'
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
  plain_rd="$(checkout_dir_rel "$untagged")"
  tag_rd="$(checkout_dir_rel "$shard0")"
  other_rd="$(checkout_dir_rel "$shard1")"
  expected_rd="bv/aminx-bv-layer_a_sampling_validate-abcdef012345"
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
  echo "self-test: sbatch command that WOULD run for a dummy shard-0 launch (no network):"
  build_sbatch_cmd "$shard0" "abcdef0123456789ffff0123456789ffff0123" "260926_browser-export-loop" \
    "layer_a_sampling_validate" "shard-0" "05:30:00" --fixture-set B --n-shards 2 --shard-index 0
  if [ "$failures" -eq 0 ]; then
    echo "self-test OK: metacharacters rejected; --tag session/log/exit/result/checkout names distinct; sbatch preview printed"
    return 0
  fi
  return 1
}

usage() {
  cat >&2 <<'EOF'
Usage: engaging_launch.sh [--tag TAG] [--time HH:MM:SS] [--prepare-only] <stem> <campaign-id> [SCRIPT_ARGS...]
       engaging_launch.sh --self-test
EOF
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

TAG=""
TIME_LIMIT="05:30:00"
PREPARE_ONLY=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)
      TAG="${2:-}"
      if ! tag_valid "$TAG"; then
        echo "engaging_launch.sh: --tag must match [a-z0-9-]+" >&2
        exit 2
      fi
      shift 2
      ;;
    --time)
      TIME_LIMIT="${2:-}"
      case "$TIME_LIMIT" in
        [0-9][0-9]:[0-9][0-9]:[0-9][0-9]) ;;
        *)
          echo "engaging_launch.sh: --time must be HH:MM:SS" >&2
          exit 2
          ;;
      esac
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

# MIT partition walltime is 12h max (myxcel-myxcel-cluster-rules.md); request <= 11:30:00
# to be safe. The 05:30:00 default is a launch-script convenience, not a hard cap, so only
# refuse values that would exceed the safe ceiling.
_time_secs() { local h m s; IFS=: read -r h m s <<<"$1"; echo $((10#$h * 3600 + 10#$m * 60 + 10#$s)); }
if [ "$(_time_secs "$TIME_LIMIT")" -gt "$(_time_secs "11:30:00")" ]; then
  echo "engaging_launch.sh: --time ${TIME_LIMIT} exceeds the safe 11:30:00 MIT partition ceiling" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Clean-tree gate, identical in scope to titanix_launch.sh's: the worktree carries many
# untracked harness files (.praxia/*, .claude/*, .mcp.json, .bth/refs/, *.ses) that must
# never be committed or deleted, so both checks are scoped to code paths only.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "engaging_launch.sh: tracked files are dirty; commit or revert before dispatching" >&2
  exit 1
fi
if [ -n "$(git ls-files --others --exclude-standard -- scripts src pyproject.toml uv.lock)" ]; then
  echo "engaging_launch.sh: untracked files exist under scripts/src/pyproject.toml/uv.lock; commit or remove before dispatching" >&2
  exit 1
fi

H="$(git rev-parse HEAD)"
H12="${H:0:12}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SESSION="$(session_name "$STEM" "$H" "$TAG")"
RD_REL="$(checkout_dir_rel "$SESSION")"
OUT_REL="$(result_relpath "$STEM" "$TAG")"

# Refuse a duplicate concurrent launch of the exact same session (stem+commit+tag) --
# Slurm job names are not unique the way systemd unit names are, so check squeue by name.
existing_jobs="$(ssh engaging squeue -h -n "$SESSION" --format='%i %T' || true)"
if [ -n "$existing_jobs" ]; then
  echo "engaging_launch.sh: a Slurm job named ${SESSION} is already queued/running; refusing a second launch:" >&2
  echo "$existing_jobs" >&2
  exit 1
fi

# Query (never hardcode) the remote home, for human-readable output only -- every actual
# remote-executed command below expands "$HOME" on the remote shell itself.
REMOTE_HOME="$(ssh engaging 'printf %s "$HOME"')"

TMP_LOCAL="$(mktemp -d)"
cleanup_local() { rm -rf "$TMP_LOCAL"; }
trap cleanup_local EXIT

BUNDLE="${TMP_LOCAL}/aminx-${H12}.bundle"
if [ "$BRANCH" = "HEAD" ]; then
  git bundle create "$BUNDLE" HEAD
else
  git bundle create "$BUNDLE" HEAD "refs/heads/${BRANCH}"
fi

ssh engaging 'mkdir -p "$HOME/bv/bundles" "$HOME/bv/lfs-objects" "$HOME/bv/logs"'
rsync -a "$BUNDLE" "engaging:bv/bundles/"

# --- LFS materialization (D8-equivalent to titanix_launch.sh, adapted): the working-tree
# copies in THIS checkout are already smudged (real bytes, not pointers) since this is an
# ordinary checkout, not a fresh clone -- so their sha256 IS the LFS object id directly,
# with no need to reach into .git/lfs/objects. Same file globs as titanix_launch.sh's
# remote discovery loop; same "skip symlinks" caveat (model_params/* are tracked symlinks
# into src/aminx/model_params/, and following one would smudge the target under the wrong
# path while leaving the symlink itself falsely reported as unmodified).
LFS_MANIFEST="${TMP_LOCAL}/lfs-manifest-${H12}.tsv"
LFS_STAGE="${TMP_LOCAL}/lfs-stage"
mkdir -p "$LFS_STAGE"
n_lfs_local=0
while IFS= read -r -d '' path; do
  [ -L "$path" ] && continue
  sha="$(sha256sum "$path" | cut -d' ' -f1)"
  printf '%s\t%s\n' "$path" "$sha" >> "$LFS_MANIFEST"
  if [ ! -e "${LFS_STAGE}/${sha}" ]; then
    ln -s "$(readlink -f "$path")" "${LFS_STAGE}/${sha}"
  fi
  n_lfs_local=$((n_lfs_local + 1))
done < <(git ls-files -z -- '*.eqx' '*.eqx.zst' '*.npz' '*.array_record' '*.tar.gz')
echo "engaging_launch.sh: staged ${n_lfs_local} LFS-tracked file(s) for materialization"

if [ -d "$LFS_STAGE" ] && [ -n "$(ls -A "$LFS_STAGE" 2>/dev/null)" ]; then
  # -L follows the staged symlinks (real bytes, not the symlink itself); --ignore-existing
  # is the "skip if present" dedupe -- an object already in the remote content-addressed
  # store by name is assumed correct (it is named by its own sha256) and is not re-sent.
  rsync -aL --ignore-existing "${LFS_STAGE}/" "engaging:bv/lfs-objects/"
fi
if [ -f "$LFS_MANIFEST" ]; then
  rsync -a "$LFS_MANIFEST" "engaging:bv/bundles/"
fi
LFS_MANIFEST_NAME="$(basename "$LFS_MANIFEST")"

# --- Remote: bare repo + fetch-from-bundle + worktree + LFS copy-in + uv sync. NO git
# push anywhere -- the user only authorised pushes to titanix (titanix_launch.sh deviation
# #1); Engaging only ever receives a bundle and fetches from it locally on that end.
ssh engaging bash -s -- "$H" "$H12" "$SESSION" "$RD_REL" "$LFS_MANIFEST_NAME" <<'REMOTE'
set -euo pipefail
H="$1"
H12="$2"
SESSION="$3"
RD_REL="$4"
LFS_MANIFEST_NAME="$5"

BV="$HOME/bv"
mkdir -p "$BV/logs" "$BV/lfs-objects" "$BV/bundles"

if [ ! -d "$BV/aminx.git" ]; then
  git init --bare "$BV/aminx.git"
fi
git -C "$BV/aminx.git" fetch "$BV/bundles/aminx-${H12}.bundle" "HEAD:refs/bv/${H12}"

RD="$HOME/${RD_REL}"
if [ ! -d "$RD" ]; then
  git -C "$BV/aminx.git" worktree add --detach "$RD" "$H"
fi
cd "$RD"

live_head="$(git rev-parse HEAD)"
if [ "$live_head" != "$H" ]; then
  echo "engaging_launch.sh (remote): HEAD is ${live_head}, expected ${H}" >&2
  exit 1
fi

# D8-equivalent: replace each LFS pointer with its verified object, then mark it
# skip-worktree so the clean-tree check below (and engaging_run.sbatch's own, later)
# compares against the pointer blob git actually committed, not the smudged bytes --
# exactly what the git-lfs smudge filter does, and exactly titanix_launch.sh's own logic.
n_lfs=0
if [ -f "$BV/bundles/${LFS_MANIFEST_NAME}" ]; then
  while IFS=$'\t' read -r path sha; do
    [ -z "$path" ] && continue
    obj="$BV/lfs-objects/${sha}"
    if [ ! -f "$obj" ]; then
      echo "engaging_launch.sh (remote): LFS object ${sha} for ${path} not in ${BV}/lfs-objects" >&2
      exit 1
    fi
    cp "$obj" "$path"
    actual="$(sha256sum "$path" | cut -d' ' -f1)"
    if [ "$actual" != "$sha" ]; then
      echo "engaging_launch.sh (remote): LFS object ${sha} for ${path} failed sha256 verification (got ${actual})" >&2
      exit 1
    fi
    git update-index --skip-worktree -- "$path"
    n_lfs=$((n_lfs + 1))
  done < "$BV/bundles/${LFS_MANIFEST_NAME}"
fi
echo "engaging_launch.sh (remote): materialized ${n_lfs} LFS object(s)"

if [ -n "$(git status --porcelain)" ]; then
  echo "engaging_launch.sh (remote): worktree dirty after LFS materialization" >&2
  exit 1
fi

# No uv/python on the LOGIN node (user rule, enforced by a PreToolUse hook). The
# environment is built by `uv sync --frozen` INSIDE engaging_run.sbatch on the compute
# node; measured 260929 (quicktest job 24334493) that compute nodes reach GitHub/PyPI.
REMOTE

if [ "$PREPARE_ONLY" -eq 1 ]; then
  echo "${REMOTE_HOME}/${RD_REL}"
  exit 0
fi

EXPORT_VAL="ALL,BV_SESSION=${SESSION},BV_OUT_REL=${OUT_REL},BV_RD=\$HOME/${RD_REL},BV_H=${H},BV_CAMPAIGN_ID=${CAMPAIGN_ID},BV_STEM=${STEM}"
JOB_ID="$(ssh engaging sbatch --job-name="$SESSION" --output="\$HOME/bv/logs/${SESSION}.log" \
  --time="$TIME_LIMIT" --export="$EXPORT_VAL" \
  "\$HOME/${RD_REL}/scripts/browser_validation/engaging_run.sbatch" "${SCRIPT_ARGS[@]}" \
  | grep -o '[0-9]\+$')"

echo "$JOB_ID"
