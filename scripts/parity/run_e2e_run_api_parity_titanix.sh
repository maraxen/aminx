#!/usr/bin/env bash
# Run the e2e run-API parity cells on titanix (CPU by default, or one shared GPU), one process per cell.
#
# Pre-registered: scripts/parity/e2e_run_api_parity.bth.toml.  Usage, from the repo root on titanix:
#
#   bash scripts/parity/run_e2e_run_api_parity_titanix.sh <out-dir> <code-commit> [--smoke]
#
# E2E_PLATFORM=cpu (default): aminx's JAX side on CPU.  E2E_PLATFORM=gpu: aminx's JAX side on ONE titanix GPU, named
# explicitly with E2E_GPU (there is no default: the GPUs here are shared with vLLM and long production jobs), with the
# JAX pool capped by E2E_MEM_FRACTION (default 0.35 per job).  The upstream reference oracle always runs on CPU.
# The platform is part of each cell's inputs hash and is asserted at run time, so a cell cannot silently fall back.
# Chunking / recovery: each cell is its own process under its own `timeout`; e2e_run_api_parity.py writes
# hash-verified .started/.done stamps, so re-running skips finished cells and recomputes stale ones.  The
# launcher never stops early on a failed cell: a failure IS data.  The verdict is computed from the files.

set -uo pipefail

OUT_DIR="${1:?usage: $0 <out-dir> <code-commit> [--smoke]}"
CODE_COMMIT="${2:?usage: $0 <out-dir> <code-commit> [--smoke]}"
SMOKE="${3:-}"
CELL_TIMEOUT="${E2E_CELL_TIMEOUT:-7200}"   # seconds, per cell

export PATH="${HOME}/.local/bin:${PATH}"
PLATFORM="${E2E_PLATFORM:-cpu}"
export E2E_PLATFORM="${PLATFORM}"
JOBS="${E2E_JOBS:-1}"
if [[ "${PLATFORM}" == "gpu" ]]; then
  GPU="${E2E_GPU:?E2E_PLATFORM=gpu needs E2E_GPU=<index>; GPUs on this box are shared, so there is no default}"
  FRACTION="${E2E_MEM_FRACTION:-0.35}"
  # cuda,cpu -- NOT cuda alone: aminx's sampler uses jax.io_callback, which needs a CPU device in the platform list
  # (with JAX_PLATFORMS=cuda every sampling cell dies at its first call; the GPU stays the default device).
  export CUDA_VISIBLE_DEVICES="${GPU}" JAX_PLATFORMS=cuda,cpu
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${FRACTION}" XLA_PYTHON_CLIENT_PREALLOCATE=false
  # Refuse to start unless the GPU has room for every job's cap plus ~1 GiB of CUDA context each.
  read -r TOTAL_MIB FREE_MIB < <(nvidia-smi -i "${GPU}" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr -d ',')
  NEED_MIB=$(python3 -c "print(int(${TOTAL_MIB} * ${FRACTION} * ${JOBS}) + 1024 * ${JOBS})")
  echo "[e2e] gpu=${GPU} total=${TOTAL_MIB}MiB free=${FREE_MIB}MiB cap=${FRACTION} jobs=${JOBS} need>=${NEED_MIB}MiB"
  if (( FREE_MIB < NEED_MIB )); then
    echo "[e2e] REFUSING: GPU ${GPU} has ${FREE_MIB} MiB free, need ${NEED_MIB}. Lower E2E_MEM_FRACTION/E2E_JOBS or pick another GPU." >&2
    exit 3
  fi
else
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export REFERENCE_PATH="${REFERENCE_PATH:-/home/solab/bv/ref/LigandMPNN}"
# Persistent XLA compilation cache, shared by every cell process and surviving a preemption/re-run: the
# sampler's first compile is minutes under load, and 20 separate processes would otherwise each repeat it.
# Numerics are unaffected (the key covers the HLO, jaxlib version and device); the dir is recorded per cell.
# Location: E2E_JAX_CACHE (explicit) > this default; E2E_JAX_CACHE=none disables the cache.
if [[ "${E2E_JAX_CACHE:-}" == "none" ]]; then
  unset JAX_COMPILATION_CACHE_DIR
else
  export JAX_COMPILATION_CACHE_DIR="${E2E_JAX_CACHE:-${HOME}/.cache/e2e-run-api-parity-jax}"
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
fi
export PYTHONUNBUFFERED=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
# Run the venv interpreter directly: the tree under test is first on PYTHONPATH, and the overlay supplies
# prody (upstream's parser; not an aminx dependency) without touching the venv.
export PYTHONPATH="${REPO}/src${E2E_OVERLAY:+:${E2E_OVERLAY}}"
PY="${E2E_PYTHON:?set E2E_PYTHON to a python with jax, torch, scipy and equinox}"

"${PY}" -c "
import aminx, jax
assert aminx.__file__.startswith('${REPO}/'), f'aminx imported from {aminx.__file__}'
assert jax.devices()[0].platform == '${PLATFORM}', f'expected ${PLATFORM}, got {jax.devices()}'
print('[e2e] aminx:', aminx.__file__, '| device:', jax.devices()[0])
"

# E2E_CELLS restricts the run to the named cells (e.g. a documented re-run of the cells a fix affects).
CELLS="${E2E_CELLS:-$("${PY}" scripts/parity/e2e_run_api_parity.py --dry-run | "${PY}" -c "import json,sys; print(' '.join(json.load(sys.stdin)['cells']))")}"

# Cells are independent processes, so E2E_JOBS of them can run at once (default 1).  Each gets OMP threads
# split so the jobs do not oversubscribe the box.
export OUT_DIR CODE_COMMIT SMOKE CELL_TIMEOUT PY
PER_JOB=$(( ${OMP_NUM_THREADS:-8} / JOBS > 0 ? ${OMP_NUM_THREADS:-8} / JOBS : 1 ))
# torch/numpy BLAS honour OPENBLAS/MKL, not just OMP, so divide all three or the box is oversubscribed.
export OMP_NUM_THREADS="${PER_JOB}" OPENBLAS_NUM_THREADS="${PER_JOB}" MKL_NUM_THREADS="${PER_JOB}"
run_one() {
  local CELL="$1" rc
  echo "[e2e] $(date -Is) starting ${CELL} (timeout ${CELL_TIMEOUT}s)"
  timeout --kill-after=60 "${CELL_TIMEOUT}" \
    "${PY}" scripts/parity/e2e_run_api_parity.py \
      --cell "${CELL}" --out-dir "${OUT_DIR}" --code-commit "${CODE_COMMIT}" ${SMOKE:+--smoke} 2>&1 \
    | grep -v -E '(User|Deprecation|Future|Runtime)Warning'   # drop library warning classes only; keep the harness's own WARNING lines
  rc=${PIPESTATUS[0]}
  echo "[e2e] $(date -Is) ${CELL} exited rc=${rc}$([[ ${rc} -eq 124 ]] && echo ' (TIMEOUT: no record; counts as failure)')"
}
export -f run_one
printf '%s\n' ${CELLS} | xargs -P "${JOBS}" -I{} bash -c 'run_one {}'

"${PY}" scripts/parity/e2e_run_api_parity.py --aggregate "${OUT_DIR}" > "${OUT_DIR}/aggregate.json"
echo "[e2e] aggregate written to ${OUT_DIR}/aggregate.json"
