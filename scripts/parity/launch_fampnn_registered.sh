#!/usr/bin/env bash
# The registered FAMPNN parity run: `bth run` inside a clean git checkout of the pushed branch (titanix, CPU JAX).
#   usage: launch_fampnn_registered.sh <git ref, default wt/261104-p4-fampnn>
# Needs: the sealed oracle dumps ($P4_ROOT/oracle/{f32,f64}); a python with jax, equinox, torch (VENV_PY).
# The checkout lives under $P4_ROOT/repo and is reset to the ref, so the run records git_dirty=false.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REF="${1:-wt/261104-p4-fampnn}"
P4_ROOT="${P4_ROOT:-/home/solab/scratch/p4-fampnn}"
VENV_PY="${VENV_PY:-/home/solab/projects/aminx-verify-261102-256/.venv/bin/python}"
REPO="$P4_ROOT/repo"
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
OUT="$HOME/.aminx/sidecars/fampnn_parity/$H"
mkdir -p "$OUT"
MUT="sigma_min_old_default,cosine_unmask_schedule,rank_decoding_order,selfcond_off,step_scale_one,conf_bins_0_8,ghost_atoms_to_origin,chain_embedding_off,categorical_reversed_cdf"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export AMINX_FAMPNN_ORACLE_DIR="${AMINX_FAMPNN_ORACLE_DIR:-$P4_ROOT/oracle}"
export AMINX_FAMPNN_ROOT="${AMINX_FAMPNN_ROOT:-/home/solab/repos/fampnn}"
echo "HEAD=$H out=$OUT"
exec bth run --project-slug aminx \
  --output-paths "$OUT/branch_controls.json" \
  -- "$VENV_PY" scripts/parity/fampnn_parity.py \
     --mutants "$MUT" \
     --work-dir "/tmp/fampnn_parity_$H" \
     --out "$OUT/fampnn_parity.json" \
     --controls-out "$OUT/branch_controls.json"
