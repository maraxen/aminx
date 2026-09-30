#!/bin/bash
# Debt #2158 reproduction: sample_states_fused at production sample counts, one cell per array task.
#
# Pre-registered: scripts/benchmarks/poe_fused_scaling.py.bth.toml (committed BEFORE any run).
# Why an ARRAY: each cell is its own process, its own job and its own walltime, so a hang or crash
# in one cell (the original failure was an 809 s compile followed by an XLA autotuning error) loses
# that cell only, never the sweep. Cells 0-2 are the shipped path at n=128/512/2048; cell 3 is the
# NEGATIVE CONTROL (the pre-#127 unbounded-vmap configuration) and is EXPECTED to fail.
#
# Resume: rerunning an array index skips a cell whose .done stamp matches the current inputs
# (args + POE_CODE_COMMIT + weights sha256) and the result file's hash, and recomputes otherwise.
#
# Usage (POE_PROJ = directory holding the code under test; POE_CODE_COMMIT = its git commit):
#
#   sbatch --partition=mit_normal_gpu --gres=gpu:h200:1 --array=0-3 \
#          --export=ALL,POE_PROJ=$HOME/projects/aminx-poe-2158,POE_CODE_COMMIT=<sha> \
#          scripts/slurm/poe_fused_scaling.slurm
#
#   L3 smoke (reduced budget, verifies GPU + env + resume, NOT evidence):
#   sbatch ... --array=0-0 --time=00:30:00 --export=ALL,...,POE_SMOKE=1 scripts/slurm/poe_fused_scaling.slurm

#SBATCH --job-name=poe-scaling
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=outputs/logs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/logs/slurm/%x_%A_%a.err

set -euo pipefail

# Absolute project path, NOT $SLURM_SUBMIT_DIR (see t2gate_gpu_sweep.slurm: `myxcel submit-job`
# leaves it as $HOME). The code under test lives in its own directory so this never touches the
# shared ~/projects/aminx checkout other jobs use.
PROJ="${POE_PROJ:?set POE_PROJ to the directory holding the code under test}"
cd "${PROJ}"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR=/orcd/scratch/orcd/011/maarxaru/.cache/uv
export UV_PROJECT_ENVIRONMENT=/orcd/scratch/orcd/011/maarxaru/venvs/aminx

# The shared venv installs aminx editable from ~/projects/aminx; put THIS tree first so the code
# under test is what imports. poe_fused_scaling.py records aminx.__file__ and the check below
# refuses to run against anything else.
export PYTHONPATH="${PROJ}/src${PYTHONPATH:+:${PYTHONPATH}}"

CELLS=(planned_n128 planned_n512 planned_n2048 control_forced_vmap_n128)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
CELL="${CELLS[$IDX]}"
OUT_DIR="outputs/results/poe_fused_scaling"
mkdir -p "${OUT_DIR}" outputs/logs/slurm
EXTRA=()
if [[ "${POE_SMOKE:-0}" == "1" ]]; then EXTRA+=(--smoke); OUT_DIR="${OUT_DIR}_smoke"; fi

_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
echo "[poe] cell=${CELL} idx=${IDX} node=${_NODE} proj=${PROJ} commit=${POE_CODE_COMMIT:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Fail loudly rather than silently running on CPU, and refuse a stale import.
uv run --no-sync python -c "
import sys, jax, aminx
d = jax.devices()
print('[poe] devices:', d)
assert d and d[0].platform == 'gpu', f'expected a GPU device, got {d}'
assert aminx.__file__.startswith('${PROJ}/'), f'aminx imported from {aminx.__file__}, not ${PROJ}'
print('[poe] aminx:', aminx.__file__)
"

# The script records the outcome in files, including a caught crash; a non-zero exit here would
# only mean the harness itself broke.
uv run --no-sync python scripts/benchmarks/poe_fused_scaling.py \
  --cell "${CELL}" --out-dir "${OUT_DIR}" --code-commit "${POE_CODE_COMMIT:-unknown}" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo "[poe] cell ${CELL} finished; record in ${OUT_DIR}/${CELL}.json"
