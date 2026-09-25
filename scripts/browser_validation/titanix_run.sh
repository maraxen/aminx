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

SESSION="bv-${STEM}-${H:0:12}"
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

# So prereq_check() sees a warm catalog (R2-C1: a local `bth run` writes only cool-tier
# parquet; `bth sql`/prereq checks read only the warm bathos.db).
"$TX_BTH" compact

OUT_PATH="outputs/browser_validation/layer_a/${STEM}.json"

# F-C1: keep these uv option tokens in EXACTLY this order, nothing between `python` and
# the script path -- bathos `_find_script_path` (runner.py:57-75 at 84be544e) reads
# tokens in pairs after `run`, and `--no-sync` only resolves because it sits in a
# skipped slot. `--prerelease allow` MUST stay two tokens (a single
# `--prerelease=allow` token would swallow the following one and break the pairing);
# without it the `--with` overlay fails to resolve (fastmcp-slim prerelease).
taskset -c 0-15 "$TX_BTH" run --campaign-id "$CAMPAIGN_ID" --output-paths "$OUT_PATH" -- \
  "$TX_UV" run --frozen --no-sync --extra dev --extra benchmark --with "$BATHOS_REQ" --prerelease allow python \
  "scripts/browser_validation/${STEM}.py" "${SCRIPT_ARGS[@]}" --out "$OUT_PATH"

echo "=== titanix_run.sh finished $(date -u +%FT%TZ) ==="
