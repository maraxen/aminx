#!/usr/bin/env bash
# Run the e2e run-API parity cells on titanix CPU, one process per cell.
#
# Pre-registered: scripts/parity/e2e_run_api_parity.bth.toml.  Usage, from the repo root on titanix:
#
#   bash scripts/parity/run_e2e_run_api_parity_titanix.sh <out-dir> <code-commit> [--smoke]
#
# CPU on purpose: the upstream reference is driven in-process with torch, and GPU 3 is held by the #2158 run.
# Chunking / recovery: each cell is its own process under its own `timeout`; e2e_run_api_parity.py writes
# hash-verified .started/.done stamps, so re-running skips finished cells and recomputes stale ones.  The
# launcher never stops early on a failed cell: a failure IS data.  The verdict is computed from the files.

set -uo pipefail

OUT_DIR="${1:?usage: $0 <out-dir> <code-commit> [--smoke]}"
CODE_COMMIT="${2:?usage: $0 <out-dir> <code-commit> [--smoke]}"
SMOKE="${3:-}"
CELL_TIMEOUT="${E2E_CELL_TIMEOUT:-7200}"   # seconds, per cell

export PATH="${HOME}/.local/bin:${PATH}"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export REFERENCE_PATH="${REFERENCE_PATH:-/home/solab/bv/ref/LigandMPNN}"
# Persistent XLA compilation cache, shared by every cell process and surviving a preemption/re-run: the
# sampler's first compile is minutes under load, and 20 separate processes would otherwise each repeat it.
# Numerics are unaffected (the key covers the HLO, jaxlib version and device); the dir is recorded per cell.
export JAX_COMPILATION_CACHE_DIR="${E2E_JAX_CACHE:-${HOME}/.cache/e2e-run-api-parity-jax}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export PYTHONUNBUFFERED=1
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
# Run the venv interpreter directly: the tree under test is first on PYTHONPATH, and the overlay supplies
# prody (upstream's parser; not an aminx dependency) without touching the venv.
export PYTHONPATH="${REPO}/src${E2E_OVERLAY:+:${E2E_OVERLAY}}"
PY="${E2E_PYTHON:?set E2E_PYTHON to a python with jax, torch, scipy and equinox}"

"${PY}" -c "
import aminx, jax
assert aminx.__file__.startswith('${REPO}/'), f'aminx imported from {aminx.__file__}'
assert jax.devices()[0].platform == 'cpu', jax.devices()
print('[e2e] aminx:', aminx.__file__, '| device:', jax.devices()[0])
"

CELLS=$("${PY}" scripts/parity/e2e_run_api_parity.py --dry-run | "${PY}" -c "import json,sys; print(' '.join(json.load(sys.stdin)['cells']))")
for CELL in ${CELLS}; do
  echo "[e2e] $(date -Is) starting ${CELL} (timeout ${CELL_TIMEOUT}s)"
  timeout --kill-after=60 "${CELL_TIMEOUT}" \
    "${PY}" scripts/parity/e2e_run_api_parity.py \
      --cell "${CELL}" --out-dir "${OUT_DIR}" --code-commit "${CODE_COMMIT}" ${SMOKE:+--smoke} 2>&1 | grep -v -i warn
  rc=${PIPESTATUS[0]}
  echo "[e2e] $(date -Is) ${CELL} exited rc=${rc}$([[ ${rc} -eq 124 ]] && echo ' (TIMEOUT: no record; counts as failure)')"
done

"${PY}" scripts/parity/e2e_run_api_parity.py --aggregate "${OUT_DIR}" > "${OUT_DIR}/aggregate.json"
echo "[e2e] aggregate written to ${OUT_DIR}/aggregate.json"
