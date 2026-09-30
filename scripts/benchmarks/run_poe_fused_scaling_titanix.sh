#!/usr/bin/env bash
# Run the #2158 PoE scaling cells on a SHARED titanix GPU, one process per cell.
#
# Pre-registered: scripts/benchmarks/poe_fused_scaling.py.bth.toml.  Usage (from the repo root on
# titanix, after `uv sync --frozen --extra cuda12 --extra dev` into this tree's .venv):
#
#   bash scripts/benchmarks/run_poe_fused_scaling_titanix.sh <out-dir> <code-commit> [--smoke]
#
# Sharing discipline. titanix GPUs are occupied (vLLM servers, other experiments), so this pins ONE
# GPU and CAPS JAX's pool with XLA_PYTHON_CLIENT_MEM_FRACTION. That is not a workaround, it is what
# makes the experiment honest: the BatchPlanner budgets against memory_stats()["bytes_limit"], the
# limit this process was granted, so it plans for the memory it really has. The launcher refuses to
# start when the GPU lacks the free memory for the cap, rather than OOM-ing a neighbour.
#
# Chunking / recovery: every cell is a separate process under its own `timeout`, so a hang or crash
# (the original failure included an 809 s compile) loses at most that cell. poe_fused_scaling.py
# writes hash-verified .started/.done stamps, so rerunning this script skips finished cells and
# recomputes anything stale. The launcher never stops early on a failed cell: a failure IS data.

set -uo pipefail

OUT_DIR="${1:?usage: $0 <out-dir> <code-commit> [--smoke]}"
CODE_COMMIT="${2:?usage: $0 <out-dir> <code-commit> [--smoke]}"
SMOKE="${3:-}"

GPU="${POE_GPU:-3}"
FRACTION="${POE_MEM_FRACTION:-0.45}"
CELL_TIMEOUT="${POE_CELL_TIMEOUT:-7200}"   # seconds, per cell

export PATH="${HOME}/.local/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${FRACTION}"
# Require the GPU backend outright: an inherited JAX_PLATFORMS=cpu (or a plugin that fails to load)
# would otherwise fall back to CPU and produce a plausible but meaningless result.
export JAX_PLATFORMS=cuda

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Refuse to start unless the GPU has room for the cap plus ~1 GiB for the CUDA context.
read -r TOTAL_MIB FREE_MIB < <(nvidia-smi -i "${GPU}" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr -d ',')
NEED_MIB=$(python3 -c "print(int(${TOTAL_MIB} * ${FRACTION}) + 1024)")
echo "[poe] gpu=${GPU} total=${TOTAL_MIB}MiB free=${FREE_MIB}MiB cap=${FRACTION} need>=${NEED_MIB}MiB"
if (( FREE_MIB < NEED_MIB )); then
  echo "[poe] REFUSING: GPU ${GPU} has ${FREE_MIB} MiB free, need ${NEED_MIB}. Lower POE_MEM_FRACTION or pick another GPU." >&2
  exit 3
fi

# Fail loudly rather than silently running on CPU; refuse a stale import.
uv run --no-sync python -c "
import jax, aminx
d = jax.devices()
assert d and d[0].platform == 'gpu', f'expected a GPU device, got {d}'
assert aminx.__file__.startswith('${REPO}/'), f'aminx imported from {aminx.__file__}'
print('[poe] device:', d[0], '| bytes_limit:', d[0].memory_stats().get('bytes_limit'))
"

CELLS=(planned_n128 planned_n512 planned_n2048 control_forced_vmap_n128)
for CELL in "${CELLS[@]}"; do
  echo "[poe] $(date -Is) starting ${CELL} (timeout ${CELL_TIMEOUT}s)"
  timeout --kill-after=60 "${CELL_TIMEOUT}" \
    uv run --no-sync python scripts/benchmarks/poe_fused_scaling.py \
      --cell "${CELL}" --out-dir "${OUT_DIR}" --code-commit "${CODE_COMMIT}" ${SMOKE:+--smoke}
  rc=$?
  echo "[poe] $(date -Is) ${CELL} exited rc=${rc}$([[ ${rc} -eq 124 ]] && echo ' (TIMEOUT: no record; counts as failure)')"
done

uv run --no-sync python scripts/benchmarks/poe_fused_scaling.py --aggregate "${OUT_DIR}" > "${OUT_DIR}/aggregate.json"
echo "[poe] aggregate written to ${OUT_DIR}/aggregate.json"
