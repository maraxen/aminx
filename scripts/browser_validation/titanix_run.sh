#!/usr/bin/env bash
# titanix_run.sh -- runs ON titanix, inside the O6' transient systemd --user unit that
# titanix_launch.sh dispatches (systemd-run --user, MemoryMax=64G, F-C7: not tmux --
# tmux is NOT installed on titanix). Never invoked directly; titanix_launch.sh supplies
# a fresh git-worktree checkout ($RD) as its cwd argument.
#
# Usage: titanix_run.sh <RD> <H> <CAMPAIGN_ID> <STEM> [SCRIPT_ARGS...]
#
# T11g: browser stems read three optional env vars, set by titanix_launch.sh's --cpu,
# --npm-dir, --uv-with flags (all no-ops when unset, so non-browser stems are unaffected):
#   BV_CPU        non-empty -> JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES="", no cuda12 extra.
#   BV_NPM_DIR    repo-relative dir (e.g. browser/layer_c) to `npm ci` before the run, with
#                 BV_NODE_BIN_DIR (default /home/solab/bv/node/node-v24.14.1-linux-x64/bin)
#                 prepended to PATH so `node`/`npx` resolve.
#   BV_UV_WITH    space-joined extra `--with SPEC` packages for the uv run below (e.g.
#                 "jax2onnx==0.16.1 onnxruntime==1.30.0 onnx==1.23.0" -- none of these are
#                 pyproject dependencies).
#
# Example (p07_knobs_gate, driven via titanix_launch.sh):
#   bash scripts/browser_validation/titanix_launch.sh --cpu --npm-dir browser/layer_c \
#     --uv-with jax2onnx==0.16.1 --uv-with onnxruntime==1.30.0 --uv-with onnx==1.23.0 \
#     --tag T p07_knobs_gate <campaign> --browser --budget-s 28800 --chunk-cells 8

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

# T11g: node/npm before anything else, so a network problem there fails fast rather than
# after the guards below. browser/**/node_modules/ is gitignored (.gitignore:75), so this
# never dirties the tree the clean-tree check below inspects, but run it first anyway to
# match local_run.sh's convention (uv sync / npm ci, then the dirty check).
if [ -n "${BV_NPM_DIR:-}" ]; then
  NODE_BIN_DIR="${BV_NODE_BIN_DIR:-/home/solab/bv/node/node-v24.14.1-linux-x64/bin}"
  export PATH="${NODE_BIN_DIR}:${PATH}"
  ( cd "${RD}/${BV_NPM_DIR}" && npm ci )
fi

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

# Deviation D10: GPU mode (set by titanix_launch.sh --gpu N). Only GPU N is visible, and JAX
# must not preallocate: titanix's other GPUs serve vLLM. The reference (torch) stays on CPU.
# T11g: BV_CPU (set by --cpu) takes precedence over BV_GPU even if both were somehow set
# (titanix_launch.sh already refuses that combination) -- no cuda12 extra under --cpu, since
# the knobs gate's JAX side runs on CPU and the extra is pointless weight for a CPU run.
UV_EXTRAS=(--extra dev --extra benchmark)
if [ -n "${BV_GPU:-}" ] && [ -z "${BV_CPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$BV_GPU"
  export JAX_PLATFORMS="cuda"
  export XLA_PYTHON_CLIENT_PREALLOCATE="false"
  UV_EXTRAS+=(--extra cuda12)
elif [ -n "${BV_CPU:-}" ]; then
  export JAX_PLATFORMS="cpu"
  export CUDA_VISIBLE_DEVICES=""
fi

# So prereq_check() sees a warm catalog (R2-C1: a local `bth run` writes only cool-tier
# parquet; `bth sql`/prereq checks read only the warm bathos.db).
"$TX_BTH" compact

OUT_PATH="outputs/browser_validation/layer_a/${STEM}.json"

# T11g: extra --with SPEC pairs from BV_UV_WITH (e.g. jax2onnx, onnxruntime, onnx for
# browser stems -- none are pyproject dependencies). Each stays a two-token `--with SPEC`
# pair, inserted before `--prerelease allow` so the F-C1 pairing below is unaffected;
# check_arg_safe on the launcher side already refused any spec containing whitespace.
UV_WITH_ARGS=()
if [ -n "${BV_UV_WITH:-}" ]; then
  read -ra _uv_with_specs <<<"$BV_UV_WITH"
  for spec in "${_uv_with_specs[@]}"; do
    UV_WITH_ARGS+=(--with "$spec")
  done
fi

# F-C1: keep these uv option tokens in EXACTLY this order, nothing between `python` and
# the script path -- bathos `_find_script_path` (runner.py:57-75 at 84be544e) reads
# tokens in pairs after `run`, and `--no-sync` only resolves because it sits in a
# skipped slot. `--prerelease allow` MUST stay two tokens (a single
# `--prerelease=allow` token would swallow the following one and break the pairing);
# without it the `--with` overlay fails to resolve (fastmcp-slim prerelease). Any
# `${UV_WITH_ARGS[@]}` (T11g) are themselves two-token `--with SPEC` pairs, so inserting
# them here preserves the pairing.
taskset -c 0-15 "$TX_BTH" run --campaign-id "$CAMPAIGN_ID" --output-paths "$OUT_PATH" -- \
  "$TX_UV" run --frozen --no-sync "${UV_EXTRAS[@]}" --with "$BATHOS_REQ" "${UV_WITH_ARGS[@]}" --prerelease allow python \
  "scripts/browser_validation/${STEM}.py" "${SCRIPT_ARGS[@]}" --out "$OUT_PATH"

echo "=== titanix_run.sh finished $(date -u +%FT%TZ) ==="
