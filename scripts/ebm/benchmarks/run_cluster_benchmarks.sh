#!/usr/bin/env bash
# Cluster-side runner for the E11a/E11b/E11c/E11d ProteinEBM benchmark
# harnesses (backlog nodes #3307/#3308/#3309/E11d, EPIC #3294). Runs all four
# JAX-vs-PyTorch throughput/latency comparisons at the design spec's
# bucket-aligned lengths {64,128,256,512} against the real 85M-param
# ProteinEBM-x checkpoint.
#
# Expects (pushed separately, outside the aminx project's git tree, since
# neither is committed to the repo):
#   ~/proteinebm_bench_assets/ProteinEBM              (reference PyTorch repo)
#   ~/proteinebm_bench_assets/model_6_expert_frozen_1m_md.pt  (checkpoint)
#
# SM120/Blackwell workaround: the pi-so3-gpu/gpu-h200 myxcel presets already
# set XLA_FLAGS=--xla_gpu_shard_autotuning=false in their environment: this
# script also sets it explicitly (idempotent, safe) per ~/.claude/rules/CLUSTER.md
# so the job is correct even if invoked under a different preset by mistake.
#
# CUDA extras: `torch` lives under the `benchmark` group, EBM runtime deps live under `ebm`,
# and jax's CUDA plugin (bundling the matching libcudnn) lives under `cuda12` -- NEITHER is
# in the base dependency set, so a bare `uv run` (which implicitly syncs only
# the base set) silently drops them, even if a prior manual sync installed
# them. `import torch` then fails with `ImportError: libcudnn.so.9: cannot
# open shared object file` on GPU nodes (confirmed via a diagnostic job on
# node4007). Explicit `uv sync --extra cuda12 --extra ebm --group benchmark` fixes this.
#
# jax/jaxlib grad-path regression (decoy/ddg only): jaxlib 0.10.2's XLA:GPU
# compiler crashes on `-jax.grad(energy)` (the `score_grad_ms` metric) on
# every modern GPU generation tested (Blackwell/H100/A100/L40S) --
# 'scf.if' control-flow shape mismatch, confirmed via a version bisection
# (see .praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md §11) to be
# a genuine 0.9.2->0.10.2 regression, not present in 0.8.0-0.9.2. Pin these
# two scripts to the last known-good version via an ad-hoc `uv run --with`
# override -- NOT a pyproject.toml/lockfile change, so it affects only
# these two invocations, not the rest of the project's resolved jax
# version. biasing/langevin don't hit this bug (no jax.grad in their hot
# path) and are deliberately left on the project's normal jax version so
# their numbers stay comparable to the already-recorded §9 results.
JAX_GRAD_PATH_PIN=(--with "jax[cuda12]==0.9.2" --with "jaxlib==0.9.2")
set -euo pipefail

ASSETS_DIR="${PROTEINEBM_ASSETS_DIR:-$HOME/proteinebm_bench_assets}"
CHECKPOINT="${ASSETS_DIR}/model_6_expert_frozen_1m_md.pt"
REFERENCE_REPO="${ASSETS_DIR}/ProteinEBM"
LENGTHS="${BENCHMARK_LENGTHS:-64,128,256,512}"
N_REPEATS="${BENCHMARK_N_REPEATS:-10}"
OUT_DIR="${BENCHMARK_OUT_DIR:-outputs/ebm_benchmarks}"

# Batch-size sweep, capped PER LENGTH to avoid a guaranteed OOM crash: the epic already found a
# real memory ceiling (L=512, n_decoys=16 -> ~17GB, RESOURCE_EXHAUSTED above that; see
# .praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md §7 finding 2). The reference's own
# score_decoys.py halves its batch size past a length threshold for the identical reason (its own
# `if decoys['atom37'].shape[1] > 175: bsize = args.bsize // 2`). This is the same discipline: sweep
# the reference's real usage scale (256/400) only where it's plausible to fit, shrink at the
# lengths already confirmed to be memory-constrained, rather than let one crash abort the whole run.
BATCH_SIZES_SMALL_L="${BENCHMARK_BATCH_SIZES_SMALL_L:-4 16 64 256}"    # L in {64, 128}
BATCH_SIZES_MED_L="${BENCHMARK_BATCH_SIZES_MED_L:-4 16 64}"            # L = 256
BATCH_SIZES_LARGE_L="${BENCHMARK_BATCH_SIZES_LARGE_L:-4 16}"           # L = 512 (confirmed ceiling)
LANGEVIN_BATCH_SIZES_SMALL_L="${BENCHMARK_LANGEVIN_BATCH_SMALL_L:-4 16 64 400}"
LANGEVIN_BATCH_SIZES_MED_L="${BENCHMARK_LANGEVIN_BATCH_MED_L:-4 16 64}"
LANGEVIN_BATCH_SIZES_LARGE_L="${BENCHMARK_LANGEVIN_BATCH_LARGE_L:-4 16}"

# Comma-separated step names to skip (decoy,ddg,biasing,langevin,heterogeneous,annealing) --
# for resuming a job that already produced good output for some steps (e.g. after job
# 18059808 completed decoy/ddg/biasing but crashed inside langevin on a real CUDA
# ILLEGAL_ADDRESS at length=128/batch=400 on Blackwell, aborting everything after it
# because this script previously had no per-step failure isolation).
SKIP_STEPS="${BENCHMARK_SKIP_STEPS:-}"

mkdir -p "${OUT_DIR}"

echo "=== uv sync --extra cuda12 --extra ebm --group benchmark ==="
uv sync --extra cuda12 --extra ebm --group benchmark

# Mandatory Blackwell/SM120 workaround (see CLUSTER.md; safe no-op elsewhere).
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_shard_autotuning=false}"

echo "=== Node: ${_NODE} | XLA_FLAGS=${XLA_FLAGS} ==="
echo "=== Checkpoint: ${CHECKPOINT} | Reference repo: ${REFERENCE_REPO} ==="
echo "=== Lengths: ${LENGTHS} | n_repeats: ${N_REPEATS} ==="

# Step-name gate: `bash -c "false"` behavior would abort the whole job under
# set -e, so every step body below runs through this wrapper -- a single
# step's failure is logged and the job moves on to the remaining steps
# instead of aborting them (see 18059808 postmortem in the comment above
# SKIP_STEPS: langevin's real Blackwell CUDA crash previously killed
# heterogeneous/annealing before they got a single run).
should_run() {
  local step="$1"
  [[ ",${SKIP_STEPS}," != *",${step},"* ]]
}

run_step() {
  local step="$1"
  shift
  if ! should_run "${step}"; then
    echo "=== SKIP (${step}): requested via BENCHMARK_SKIP_STEPS ==="
    return 0
  fi
  echo "=== RUN (${step}) ==="
  set +e
  ( set -e; "$@" )
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "=== FAILED (${step}): exit ${rc} -- continuing with remaining steps ==="
  else
    echo "=== OK (${step}) ==="
  fi
}

run_per_length_group() {
  # Each of the 3 length groups is isolated (own set +e/-e cycle) -- previously
  # a crash in the FIRST call (e.g. a real GPU OOM from decoy/ddg now that
  # PyTorch actually runs on the GPU alongside JAX, see the device-placement
  # fix in .praxia/docs/audits/260716_proteinebm-parity-report.md §1) tripped
  # this whole function's implicit set -euo pipefail and skipped the other two
  # length groups entirely, even though they never crashed themselves. This
  # bit decoy_benchmark.py/ddg_benchmark.py in job 18284162: zero output files
  # were written for either script even though only the L64-128 invocation
  # actually OOM'd.
  local script="$1" out_prefix="$2" small_batches="$3" med_batches="$4" large_batches="$5"
  shift 5
  local group group_name rest group_lengths group_batches rc
  for group in "L64-128:64,128:${small_batches}" "L256:256:${med_batches}" "L512:512:${large_batches}"; do
    group_name="${group%%:*}"
    rest="${group#*:}"
    group_lengths="${rest%%:*}"
    group_batches="${rest#*:}"
    echo "=== RUN (${script} ${group_name}) ==="
    set +e
    ( set -e
      uv run --no-sync "${JAX_GRAD_PATH_PIN[@]}" python "scripts/ebm/benchmarks/${script}" \
        --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
        --lengths "${group_lengths}" --batch-sizes ${group_batches} --n-repeats "${N_REPEATS}" "$@" \
        --out "${OUT_DIR}/${out_prefix}_${group_name}.json"
    )
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      echo "=== FAILED (${script} ${group_name}): exit ${rc} -- continuing with remaining length groups ==="
    else
      echo "=== OK (${script} ${group_name}) ==="
    fi
  done
}

run_step decoy run_per_length_group decoy_benchmark.py decoy_benchmark_full \
  "${BATCH_SIZES_SMALL_L}" "${BATCH_SIZES_MED_L}" "${BATCH_SIZES_LARGE_L}"

run_step ddg run_per_length_group ddg_benchmark.py ddg_benchmark_full \
  "${BATCH_SIZES_SMALL_L}" "${BATCH_SIZES_MED_L}" "${BATCH_SIZES_LARGE_L}"

run_step biasing uv run --no-sync python scripts/ebm/benchmarks/biasing_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/biasing_benchmark_full.json"

# heterogeneous + annealing run BEFORE langevin: both are brand-new scripts with zero
# prior data, while langevin already has partial coverage from 18059808 and (as of the
# incremental-write fix above) now salvages every completed cell even if a later one
# crashes -- so if something crashes again, these two get priority for first data.
#
# BENCHMARK_HETEROGENEOUS_N default capped at 64, not 256: confirmed (jobs 18632128, 18634944,
# 18639374, 2026-07-23) that pytorch_pad_to_batch_max's whole-batch dense padding needs
# ~140-145GiB at n_structures=256/max_length=512, more than a single H200's 139.8GiB -- a real
# O(n_structures * max_length^2) capacity wall (the dominant cost is a pairwise conditioning
# tensor every attention layer consumes in full), NOT fixable by torch.compile (confirmed
# empirically: Inductor kernel fusion still needed the same ~140GiB). At n_structures=64 the same
# seed's sampled batch still includes a length-512 structure (max_length stays pinned at 512 --
# the distribution reliably draws near-boundary real lengths regardless of sample size), so memory
# scales ~linearly with n_structures alone: ~35GiB, a ~4x safety margin under the H200's capacity.
# Override via BENCHMARK_HETEROGENEOUS_N=256 only if deliberately re-confirming the OOM itself.
run_step heterogeneous uv run --no-sync python scripts/ebm/benchmarks/heterogeneous_batch_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --n-structures "${BENCHMARK_HETEROGENEOUS_N:-64}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/heterogeneous_batch_benchmark_full.json"

run_step annealing uv run --no-sync python scripts/ebm/benchmarks/langevin_annealing_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/langevin_annealing_benchmark_full.json"

run_step langevin run_per_length_group langevin_benchmark.py langevin_benchmark_full \
  "${LANGEVIN_BATCH_SIZES_SMALL_L}" "${LANGEVIN_BATCH_SIZES_MED_L}" "${LANGEVIN_BATCH_SIZES_LARGE_L}"

echo "=== All benchmark steps attempted; results in ${OUT_DIR} (check FAILED markers above) ==="
