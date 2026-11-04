#!/usr/bin/env bash
# Run the FAMPNN parity vehicle on titanix against the sealed oracle dumps (CPU JAX, x64 on).
#   usage: launch_fampnn_parity.sh [extra fampnn_parity.py args]
# Needs the layout described in launch_fampnn_oracle.sh plus a synced checkout at $P4_ROOT/aminx.
# Under bathos: `bth run scripts/parity/fampnn_parity.py -- --oracle-dir ... ` with these env vars set.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
P4_ROOT="${P4_ROOT:-/home/solab/scratch/p4-fampnn}"
VENV_PY="${VENV_PY:-/home/solab/projects/aminx-verify-261102-256/.venv/bin/python}"
cd "$P4_ROOT/aminx"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$P4_ROOT/aminx/src"
export AMINX_FAMPNN_ORACLE_DIR="${AMINX_FAMPNN_ORACLE_DIR:-$P4_ROOT/oracle}"
export AMINX_FAMPNN_ROOT="${AMINX_FAMPNN_ROOT:-/home/solab/repos/fampnn}"
exec "$VENV_PY" -u -I "$P4_ROOT/aminx/scripts/parity/fampnn_parity.py" \
  --out "$P4_ROOT/runs/fampnn_parity.json" --controls-out "$P4_ROOT/runs/branch_controls.json" \
  --work-dir "$P4_ROOT/runs/work" "$@"
