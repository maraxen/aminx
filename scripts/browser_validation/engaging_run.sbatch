#!/usr/bin/env bash
#SBATCH --job-name=bv-engaging-run
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# engaging_run.sbatch -- runs ON an Engaging compute node (MIT Slurm cluster). Twin of
# titanix_run.sh (T10f), submitted by engaging_launch.sh via `sbatch` with
# BV_SESSION/BV_OUT_REL/BV_RD/BV_H/BV_CAMPAIGN_ID/BV_STEM exported via --export=ALL,...
# (the --job-name/--output/--time header directives above are placeholders; sbatch CLI
# flags from engaging_launch.sh override them at submission time).
#
# No `srun` inside -- plain bash. Slurm's own cgroup already confines this job to
# --cpus-per-task CPUs and the one GPU named by --gres, so (unlike titanix_run.sh, which
# shares a host with vLLM and must `taskset` itself) no manual CPU pinning is needed here.
#
# Usage (as submitted by engaging_launch.sh): engaging_run.sbatch [SCRIPT_ARGS...]

set -euo pipefail

: "${BV_SESSION:?BV_SESSION not set (expected from --export)}"
: "${BV_RD:?BV_RD not set (expected from --export)}"
: "${BV_H:?BV_H not set (expected from --export)}"
: "${BV_CAMPAIGN_ID:?BV_CAMPAIGN_ID not set (expected from --export)}"
: "${BV_STEM:?BV_STEM not set (expected from --export)}"

SCRIPT_ARGS=("$@")

LOG_DIR="$HOME/bv/logs"
EXIT_FILE="${LOG_DIR}/${BV_SESSION}.exit"
mkdir -p "$LOG_DIR"

# F-C4 (titanix_run.sh): install the exit-code trap BEFORE the first guard below. A guard
# failure (exit 3) must be captured exactly like a run failure -- .exit is the only
# durable completion signal a poller can read (Slurm's own job-state history is a second,
# independent source, but this script does not rely on that).
trap 'echo $? > "'"$EXIT_FILE"'"' EXIT

echo "=== engaging_run.sbatch starting $(date -u +%FT%TZ): RD=${BV_RD} H=${BV_H} CAMPAIGN_ID=${BV_CAMPAIGN_ID} STEM=${BV_STEM} ARGS=${SCRIPT_ARGS[*]:-} ==="
echo "node: $(hostname -s)"
nvidia-smi -L || echo "engaging_run.sbatch: nvidia-smi -L failed (no GPU visible?)"

cd "$BV_RD"

# Unlike titanix_launch.sh's worktree (which carries untracked harness files and needs a
# scoped clean-tree check), this IS a fresh git-worktree checkout with nothing else ever
# written into it -- a plain `git status --porcelain` is the correct, unmodified check,
# exactly as titanix_run.sh uses on ITS fresh checkout.
live_head="$(git rev-parse HEAD)"
if [ "$live_head" != "$BV_H" ]; then
  echo "engaging_run.sbatch: HEAD is ${live_head}, expected ${BV_H}" >&2
  exit 3
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "engaging_run.sbatch: worktree at ${BV_RD} is dirty" >&2
  exit 3
fi

BATHOS_COMMIT_PIN="84be544ecb45734f46e43d22f351a54d6edd6ae5"
BATHOS_REQ="bathos @ git+https://github.com/maraxen/bathos@${BATHOS_COMMIT_PIN}"
BTH_BIN="$HOME/bv/bth-84be544e/bin/bth"

if [ ! -x "$BTH_BIN" ]; then
  echo "engaging_run.sbatch: no bth binary at ${BTH_BIN} (one-time Engaging staging missing)" >&2
  exit 3
fi

# Adapted from titanix_run.sh's uv-TOOL direct_url.json check: titanix's `bth` is a `uv
# tool install`, so its dist-info lives under ~/.local/share/uv/tools/bathos/. Here
# $BTH_BIN is a plain venv at $HOME/bv/bth-84be544e (one-time Engaging staging), so its
# dist-info sits under that venv's own site-packages instead. Same check, same pin.
# Glob unquoted so python3* expands (a quoted glob made `ls` exit 2, which pipefail +
# set -e turned into a silent exit 2 -- L3 smoke job 24335840). No jq dependency.
venv_root="$(dirname "$(dirname "$BTH_BIN")")"
venv_dist_info=""
for candidate in "$venv_root"/lib/python3*/site-packages/bathos-*.dist-info; do
  if [ -f "$candidate/direct_url.json" ]; then
    venv_dist_info="$candidate"
    break
  fi
done
if [ -z "$venv_dist_info" ]; then
  echo "engaging_run.sbatch: cannot find bathos dist-info under ${venv_root}" >&2
  exit 3
fi
installed_bathos_commit="$(sed -n 's/.*"commit_id" *: *"\([0-9a-f]*\)".*/\1/p' "${venv_dist_info}/direct_url.json")"
if [ "$installed_bathos_commit" != "$BATHOS_COMMIT_PIN" ]; then
  echo "engaging_run.sbatch: installed bathos commit is '${installed_bathos_commit}', expected '${BATHOS_COMMIT_PIN}'" >&2
  exit 3
fi

export BTH_CATALOG_DIR="$HOME/bv/catalog-aminx/.bth/catalog"
export BTH_PROJECT_SLUG="aminx"
export REFERENCE_PATH="$HOME/bv/ref/LigandMPNN"
export PROTEINMPNN_PATH="$HOME/bv/ref/ProteinMPNN"
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
# CUDA_VISIBLE_DEVICES is intentionally left unset here -- Slurm's --gres=gpu:l40s:1
# already sets it for this job step; overriding it would fight the scheduler's own
# device-cgroup assignment.
CPUS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export BTH_BIN

# CLUSTER.md #2 / myxcel-myxcel-cluster-rules.md: JAX autotuning hangs on SM120 (Blackwell,
# node4007/node4008) even with jaxlib >= 0.4.38. Keyed on hostname, not partition, and a
# no-op on non-Blackwell hardware -- safe to set unconditionally.
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

UV_BIN="$HOME/.local/bin/uv"
UV_EXTRAS=(--extra dev --extra benchmark --extra cuda12)

# Build the project env HERE, on the compute node: login-node uv/python is prohibited.
# `--frozen` installs exactly uv.lock; `.venv` is gitignored so the tree stays clean.
"$UV_BIN" sync --frozen "${UV_EXTRAS[@]}"

# So prereq_check() sees a warm catalog (R2-C1, same reasoning as titanix_run.sh: a local
# `bth run` writes only cool-tier parquet; sql/prereq reads need `bth compact` first).
"$BTH_BIN" compact

OUT_PATH="outputs/browser_validation/layer_a/${BV_STEM}.json"
if [ -n "${BV_OUT_REL:-}" ]; then
  OUT_PATH="$BV_OUT_REL"
fi

# F-C1 (titanix_run.sh): keep these uv option tokens in EXACTLY this order, nothing
# between `python` and the script path -- bathos `_find_script_path` (runner.py:57-75 at
# 84be544e) reads tokens in pairs after `run`, and `--no-sync` only resolves because it
# sits in a skipped slot. `--prerelease allow` MUST stay two tokens (a single
# `--prerelease=allow` token would swallow the following one and break the pairing);
# without it the `--with` overlay fails to resolve (fastmcp-slim prerelease). No `taskset`
# here (unlike titanix_run.sh) -- Slurm already confines this job to $CPUS CPUs.
"$BTH_BIN" run --campaign-id "$BV_CAMPAIGN_ID" --output-paths "$OUT_PATH" -- \
  "$UV_BIN" run --frozen --no-sync "${UV_EXTRAS[@]}" --with "$BATHOS_REQ" --prerelease allow python \
  "scripts/browser_validation/${BV_STEM}.py" "${SCRIPT_ARGS[@]}" --out "$OUT_PATH"

echo "=== engaging_run.sbatch finished $(date -u +%FT%TZ) ==="
