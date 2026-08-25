#!/bin/bash
# xtrax.profiling dogfood: stage-2 ProbeRecords from the real ConditionalDecode
# path (via make_axis_dispatch_via_xtrax) on an L40S under mit_preemptable.
#
# DEPENDS on prof_dogfood_envprep_quicktest (afterok) so the wheel is already
# installed in the shared scratch venv. Mirrors t2gate_gpu_sweep.slurm's
# environment handling (scratch uv paths, absolute cd, GPU assertion,
# hostname-keyed Blackwell XLA workaround -- no-op on L40S).

#SBATCH --job-name=prof-dogfood
#SBATCH --partition=mit_preemptable
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/logs/slurm/%x_%A.out
#SBATCH --error=outputs/logs/slurm/%x_%A.err

set -euo pipefail

PROJ="${HOME}/projects/aminx"
cd "${PROJ}"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR=/orcd/scratch/orcd/011/maarxaru/.cache/uv
export UV_PROJECT_ENVIRONMENT=/orcd/scratch/orcd/011/maarxaru/venvs/aminx

# Auto-detect can be defeated by an inherited JAX_PLATFORMS=cpu from the
# submission environment; force platform discovery inside the allocation
# (first L40S attempt silently landed on CpuDevice despite a live GPU).
export JAX_PLATFORMS=""
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

echo "[dogfood] node=${_NODE} partition=${SLURM_JOB_PARTITION}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "[dogfood] nvidia-smi not on PATH (preemptable image) -- relying on jax device check"
fi

uv run --no-sync python -c "
import jax, xtrax.profiling
d = jax.devices()
print('[dogfood] devices:', d)
assert d and d[0].platform == 'gpu', f'expected GPU, got {d}'
"

uv run --no-sync python scripts/profiling/dogfood_stage2_dispatch.py \
  --n-warmup 10 --n-timed 50 \
  --out-dir outputs/profiling/stage2

echo "[dogfood] done"
