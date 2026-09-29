#!/usr/bin/env bash
# titanix_run.sh -- runs ON titanix, inside the O6' transient systemd --user unit that
# titanix_launch.sh dispatches (systemd-run --user, MemoryMax=64G, F-C7: not tmux --
# tmux is NOT installed on titanix). Never invoked directly; titanix_launch.sh supplies
# a fresh git-worktree checkout ($RD) as its cwd argument.
#
# Usage: titanix_run.sh <RD> <H> <CAMPAIGN_ID> <STEM> [SCRIPT_ARGS...]

set -euo pipefail

RD="$1"
H="$2"
CAMPAIGN_ID="$3"
STEM="$4"
shift 4
SCRIPT_ARGS=("$@")

# Untagged launches leave BV_SESSION unset, so the unit/log/.exit name stays
# bv-$STEM-${H:0:12}. titanix_launch.sh --tag exports BV_SESSION (the unit name).
SESSION="bv-${STEM}-${H:0:12}"
if [ -n "${BV_SESSION:-}" ]; then
  SESSION="$BV_SESSION"
fi
LOG_DIR="/home/solab/bv/logs"
LOG_FILE="${LOG_DIR}/${SESSION}.log"
EXIT_FILE="${LOG_DIR}/${SESSION}.exit"
mkdir -p "$LOG_DIR"

# F-C4: install the exit-code trap BEFORE the first guard below. --collect may garbage-
# collect this unit once it exits, so `.exit` (not unit persistence, not journalctl) is
# the only durable signal the orchestrator's poll (T4 common context step 5) can read --
# a guard failure (exit 3) must be captured exactly like a run failure.
trap 'echo $? > "'"$EXIT_FILE"'"' EXIT

exec >>"$LOG_FILE" 2>&1
echo "=== titanix_run.sh starting $(date -u +%FT%TZ): RD=${RD} H=${H} CAMPAIGN_ID=${CAMPAIGN_ID} STEM=${STEM} ARGS=${SCRIPT_ARGS[*]:-} ==="

TX_UV="/home/solab/.local/bin/uv"
TX_BTH="/home/solab/.local/bin/bth"
BATHOS_COMMIT_PIN="84be544ecb45734f46e43d22f351a54d6edd6ae5"
BATHOS_REQ="bathos @ git+https://github.com/maraxen/bathos@${BATHOS_COMMIT_PIN}"

cd "$RD"

# titanix's own worktree is a FRESH checkout (unlike the local $WT, which carries the
# many untracked harness files titanix_launch.sh's clean-tree check works around) --
# a plain `git status --porcelain` is the correct, unmodified check here.
live_head="$(git rev-parse HEAD)"
if [ "$live_head" != "$H" ]; then
  echo "titanix_run.sh: HEAD is ${live_head}, expected ${H}" >&2
  exit 3
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "titanix_run.sh: worktree at ${RD} is dirty" >&2
  exit 3
fi

# The bathos commit that matters is the `bth` uv TOOL's own install (the one that
# executes `bth run` below), not the project venv -- aminx's pyproject does not depend
# on bathos, so a project-venv lookup would always fail. Read its direct_url.json.
tool_direct_url="$(ls -d /home/solab/.local/share/uv/tools/bathos/lib/python3*/site-packages/bathos-*.dist-info | head -1)/direct_url.json"
installed_bathos_commit="$(jq -r '.vcs_info.commit_id // ""' "$tool_direct_url")"
if [ "$installed_bathos_commit" != "$BATHOS_COMMIT_PIN" ]; then
  echo "titanix_run.sh: installed bathos commit is '${installed_bathos_commit}', expected '${BATHOS_COMMIT_PIN}'" >&2
  exit 3
fi

export BTH_CATALOG_DIR="/home/solab/bv/catalog-aminx/.bth/catalog"
export BTH_PROJECT_SLUG="aminx"
export REFERENCE_PATH="/home/solab/bv/ref/LigandMPNN"
export PROTEINMPNN_PATH="/home/solab/bv/ref/ProteinMPNN"
export JAX_PLATFORMS="cpu"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export BTH_BIN="$TX_BTH"

# T10g: unbuffered stdout/stderr so long-running scripts' INFO progress logs (e.g.
# layer_a_sampling_calibrate.py's run_full, previously silent for 6-7+ hours after its
# "budget floor" lines) actually reach $LOG_FILE promptly instead of sitting in a
# buffer until process exit.
export PYTHONUNBUFFERED=1

# Deviation D10: GPU mode (set by titanix_launch.sh --gpu N). Only GPU N is visible, and JAX
# must not preallocate: titanix's other GPUs serve vLLM. The reference (torch) stays on CPU.
UV_EXTRAS=(--extra dev --extra benchmark)
if [ -n "${BV_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$BV_GPU"
  export JAX_PLATFORMS="cuda"
  export XLA_PYTHON_CLIENT_PREALLOCATE="false"
  UV_EXTRAS+=(--extra cuda12)
fi

# So prereq_check() sees a warm catalog (R2-C1: a local `bth run` writes only cool-tier
# parquet; `bth sql`/prereq checks read only the warm bathos.db).
"$TX_BTH" compact

# Untagged: the historical stem json. Tagged: BV_OUT_REL from titanix_launch.sh
# (outputs/browser_validation/layer_a/${STEM}-${TAG}.json) so two shards do not
# share one result file inside their own worktrees either.
OUT_PATH="outputs/browser_validation/layer_a/${STEM}.json"
if [ -n "${BV_OUT_REL:-}" ]; then
  OUT_PATH="$BV_OUT_REL"
fi

# T10g: layer_a_sampling_calibrate's run_full gets a checkpoint dir OUTSIDE this
# per-run worktree ($RD is a fresh checkout each dispatch), keyed on STEM + the
# committed HEAD ($H) -- a relaunch at the SAME commit finds and resumes it. Every
# other stem is byte-identical to before (empty array, no flag added).
CHECKPOINT_ARGS=()
if [ "$STEM" = "layer_a_sampling_calibrate" ]; then
  CHECKPOINT_DIR="/home/solab/bv/ckpt/${STEM}-${H:0:12}"
  mkdir -p "$CHECKPOINT_DIR"
  CHECKPOINT_ARGS=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

# F-C1: keep these uv option tokens in EXACTLY this order, nothing between `python` and
# the script path -- bathos `_find_script_path` (runner.py:57-75 at 84be544e) reads
# tokens in pairs after `run`, and `--no-sync` only resolves because it sits in a
# skipped slot. `--prerelease allow` MUST stay two tokens (a single
# `--prerelease=allow` token would swallow the following one and break the pairing);
# without it the `--with` overlay fails to resolve (fastmcp-slim prerelease). Script
# args (including CHECKPOINT_ARGS) go after the script path -- that's fine, only the
# `python <script path>` pairing above the script path is order-sensitive.
taskset -c 0-15 "$TX_BTH" run --campaign-id "$CAMPAIGN_ID" --output-paths "$OUT_PATH" -- \
  "$TX_UV" run --frozen --no-sync "${UV_EXTRAS[@]}" --with "$BATHOS_REQ" --prerelease allow python \
  "scripts/browser_validation/${STEM}.py" "${SCRIPT_ARGS[@]}" --out "$OUT_PATH" "${CHECKPOINT_ARGS[@]}"

echo "=== titanix_run.sh finished $(date -u +%FT%TZ) ==="
