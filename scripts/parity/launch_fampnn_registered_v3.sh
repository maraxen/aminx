#!/usr/bin/env bash
# FAMPNN parity v3 on titanix (CPU JAX), either a plain development run or the bth-recorded graded run.
#   usage: launch_fampnn_registered_v3.sh <dev|bth> <git ref>   (the ref is required; there is no default)
# Needs: the sealed oracle dumps ($ORACLE_DIR/{f32,f64}) and the float64-table diagnostic oracle ($F64TAB_DIR/f64).
# The checkout lives under $P4FIX_ROOT/repo and is reset to the ref, so a bth run records git_dirty=false.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
MODE="${1:?dev|bth}"
REF="${2:?git ref required: pass the branch or sha to grade as the second argument}"
P4FIX_ROOT="${P4FIX_ROOT:-/home/solab/scratch/p4fix2-fampnn}"
VENV_PY="${VENV_PY:-/home/solab/projects/aminx-verify-261102-256/.venv/bin/python}"
REPO="$P4FIX_ROOT/repo"
if [ ! -d "$REPO/.git" ]; then
  git clone -q https://github.com/maraxen/aminx.git "$REPO"
fi
cd "$REPO"
git checkout -q -- . 2>/dev/null || true
git fetch -q https://github.com/maraxen/aminx.git "$REF"
git checkout -q FETCH_HEAD
H=$(git rev-parse --short=8 HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSING: tree is dirty" >&2
  git status --porcelain >&2
  exit 1
fi
OUT="$HOME/.aminx/sidecars/fampnn_parity_v3/$MODE-$H"
mkdir -p "$OUT"
MUT="sigma_min_old_default,cosine_unmask_schedule,rank_decoding_order,selfcond_off,step_scale_one,conf_bins_0_8,ghost_atoms_to_origin,chain_embedding_off,categorical_reversed_cdf"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export AMINX_FAMPNN_ORACLE_DIR="${AMINX_FAMPNN_ORACLE_DIR:-/home/solab/scratch/p4-fampnn/oracle}"
export AMINX_FAMPNN_F64TAB_DIR="${AMINX_FAMPNN_F64TAB_DIR:-/home/solab/scratch/p4fix-fampnn/oracle_f64tab}"
export AMINX_FAMPNN_ROOT="${AMINX_FAMPNN_ROOT:-/home/solab/repos/fampnn}"
echo "HEAD=$H mode=$MODE out=$OUT"
ARGS=(--mutants "$MUT" --work-dir "/tmp/fampnn_parity_v3_$H" --out "$OUT/fampnn_parity_v3.json" --controls-out "$OUT/branch_controls.json")
if [ "$MODE" = "bth" ]; then
  exec bth run --project-slug aminx --output-paths "$OUT/branch_controls.json" \
    -- "$VENV_PY" scripts/parity/fampnn_parity_v3.py "${ARGS[@]}"
fi
exec "$VENV_PY" scripts/parity/fampnn_parity_v3.py "${ARGS[@]}"
