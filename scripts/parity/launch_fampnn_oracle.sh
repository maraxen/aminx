#!/usr/bin/env bash
# Run the sealed FAMPNN upstream oracle on titanix (CPU, torch only, no aminx on PYTHONPATH).
#   usage: launch_fampnn_oracle.sh <dtype f32|f64> [extra dump args]
# Layout under $P4_ROOT (default /home/solab/scratch/p4-fampnn):
#   stubs/        scripts/parity/fampnn_oracle_env (timm stub; torchvision-free)
#   extra/        pure-python upstream deps installed with `uv pip install --target` (omegaconf,
#                 torchtyping, typeguard==2.13.3, gemmi, rdkit, torch_geometric, xxhash, ...)
#   oracle_scripts/ this repo's dump_fampnn_oracle.py + fampnn_fixtures.py
#   oracle/<dtype>/ outputs (npz + manifest.json)
set -euo pipefail
dtype="${1:?dtype f32|f64}"
shift || true
P4_ROOT="${P4_ROOT:-/home/solab/scratch/p4-fampnn}"
VENV_PY="${VENV_PY:-/home/solab/projects/aminx-verify-261102-256/.venv/bin/python}"
FAMPNN_ROOT="${FAMPNN_ROOT:-/home/solab/repos/fampnn}"
cd "$P4_ROOT"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$P4_ROOT/stubs:$P4_ROOT/extra"
exec timeout 1700 "$VENV_PY" "$P4_ROOT/oracle_scripts/dump_fampnn_oracle.py" \
  --fampnn-root "$FAMPNN_ROOT" --pdb "$P4_ROOT/aminx/tests/data/1ubq.pdb" \
  --out-dir "$P4_ROOT/oracle/$dtype" --dtype "$dtype" "$@" > "$P4_ROOT/oracle_$dtype.log" 2>&1
