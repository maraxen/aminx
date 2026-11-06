#!/bin/bash
# task_id 260929_potts-laser-xtrax-compose
#
# THE LEDGER WAVE. Run ONLY after the last commit under a scoped path
# (src/aminx/, scripts/parity/, scripts/recapture/, aminx-oracles/,
# pyproject.toml, uv.lock). Staleness is PER ROW since 261008: a row is stale
# only if a changed path is in its import closure (tests/knob_gate/_closure.py,
# declared edges in tests/knob_gate/closure_edges.toml). Run
# scripts/redsox/stale_rows.py to see which rows a wave must cover.
# tests/knob_gate/ is NOT scoped, so ledger ids may be written afterwards.
#
# Usage:  launch_wave.sh [--stale-only] 1 | 2 | 3 | 4 | 5 | 6 | 7 | positive
#   1-4  Potts / LASEr / ProtonPotts (GPU-assigned, see backend_env)
#   5    caliby_parity_v2 + antibmpnn_parity_v3   (P4, CPU only)
#   6    fampnn_parity_v3                          (P4, CPU only)
#   7    ca_only_parity + thermo_mpnn_parity + tier1_variants_parity (P4, CPU only)
# Groups 5-7 need titanix inputs that this script cannot verify; slug_extras()
# reads them from env vars and exits with REFUSING: if one is unset. P4 slugs
# only exist on the P4 branch, so set AMINX_WAVE_REF to it (default is main).
#
# --stale-only (or AMINX_WAVE_ONLY_STALE=1) launches only the slugs of the
# chosen group that stale_rows.py reports stale at HEAD, so a Potts-only change
# does not re-run the LASEr vehicles. Without it every slug of the group runs,
# which is the only safe choice when the ledger holds no usable run ids.
# `positive` is the ungraded control for laser_proofread_parity and is skipped
# under --stale-only unless that vehicle is stale.
#
# The --mutants string for each slug is CHECKED against that slug's rows in
# tests/knob_gate/branch_manifest.toml before anything launches, because step
# 1c compares the raw argv string to the manifest row set by exact equality
# (_coverage.py:309) and a mismatch throws the run out hours later for a
# reason that has nothing to do with what it measured.
#
# scalar_both_off is the POSITIVE control and appears in no graded string: it
# has no manifest row (every row must report "failed", and it must PASS), and
# _argv_mutants reads the raw string, so appending it voids an otherwise
# perfect run. `launch_wave.sh positive` runs it alone, with no
# --controls-out, so it cannot be mistaken for a ledger run.
set -euo pipefail

REPO=${AMINX_WAVE_REPO:-$HOME/projects/aminx-sprint-git}
cd "$REPO"

# .praxia/audits.jsonl is TRACKED and the PORT TIER appends to it, so a dirty
# tree makes every later run ineligible (git_dirty=true). Restore first.
#
# Measured 261002, and I got this wrong in both directions before measuring it
# properly, so the scope is written out: running `tests/knob_gate` and
# `tests/lint` leaves the file byte-identical (same sha256, same 206 lines),
# which is what misled me into deleting this comment as false. Running the PORT
# tier appends hundreds of records -- a gate run plus two `tests/port/` runs
# added 259 lines, each a `{"domain": "port", "rule_id": "parity_tier_1", ...}`
# finding emitted by the port harness itself.
#
# So: "anything running pytest" is too strong, "pytest never touches it" is
# false, and the rule that matters here is that ANY run of tests/port/ dirties
# a tracked file. That is exactly what this script runs next to.
git checkout -- . 2>/dev/null || true
# AMINX_WAVE_REF defaults to main. It used to be the long-merged sprint branch,
# which a wave launched months later would silently have measured instead.
git fetch -q https://github.com/maraxen/aminx.git "${AMINX_WAVE_REF:-main}"
git checkout -q FETCH_HEAD
H=$(git rev-parse --short=8 HEAD)
DIRTY=$(git status --porcelain | wc -l)
echo "HEAD=$H  porcelain=$DIRTY"
if [ "$DIRTY" -ne 0 ]; then
  echo "REFUSING: tree is dirty, every run would record git_dirty=true" >&2
  git status --porcelain >&2
  exit 1
fi

ONLY_STALE=${AMINX_WAVE_ONLY_STALE:-0}
if [ "${1:-}" = "--stale-only" ]; then ONLY_STALE=1; shift; fi
GROUP=${1:-}
[ -n "$GROUP" ] || { echo "usage: $0 [--stale-only] [1|2|3|4|5|6|7|positive]" >&2; exit 2; }

STALE=""
if [ "$ONLY_STALE" = "1" ]; then
  STALE=$(uv run --no-sync python3 scripts/redsox/stale_rows.py --slugs) \
    || { echo "REFUSING: stale_rows.py failed, so --stale-only cannot be trusted" >&2; exit 1; }
  echo "stale rows: ${STALE:-(none)}" | tr '\n' ' '; echo
fi

# slug -> the exact mutant string for its graded run.
declare -A MUT=(
  [laser_proofread_parity]="reduction_swap,ddof_0,scalar_off,vector_on"
  [laser_proofread_unconditional_parity]="unc_no_dropout,unc_scalar_off,unc_vector_on"
  [laser_decode_e2e]="reversed_order,chi_bin_plus_one,offset_zero,chi2_equals_chi1,lambda_on_logits"
  [potts_ar_refine_exact]="ar_mask_present_chain_m_pos,wrong_partition_sign,ntoc_refine_order"
  [potts_ar_decode]="ar_mask_present_chain_m_pos"
  [potts_energy_parity]="permute_etab_out_rows"
  [laser_score_parity]="permute_decoder_layer"
  [potts_ddg_megascale]="skip_transpose_merge_pair"
  [caliby_parity_v2]="rbf_range_shrunk,virtual_cb_sign_flipped,neighbors_farthest,neighbor_gather_off_by_one,gap_embedding_row_swapped,masked_atoms_not_ca_filled,potts_unsymmetrized,alphabet_unpermuted,energy_pair_term_doubled,ensemble_sum_not_mean,slot4_virtual_cb_written,potts_scale_single_factor,potts_j_right_transposed_read,k_neighbors_potts_truncated,packer_sigma_scaled,packer_selfcond_dropped"
  [antibmpnn_parity_v3]="reversed_ar_mask,offset_shift,entry_ar_mask_transposed,omit_penalty_dropped,lehninger_pka_shifted,barley_tables_swapped,parser_wrong_atom_order,zero_edge_embedding,swap_checkpoint_noise"
  [fampnn_parity_v3]="sigma_min_old_default,cosine_unmask_schedule,rank_decoding_order,selfcond_off,step_scale_one,conf_bins_0_8,ghost_atoms_to_origin,chain_embedding_off,categorical_reversed_cdf"
  [ca_only_parity]="ca_quaternion_sign_flip,ca_rbf_block_swap,ca_zero_edge_features,ca_permuted_output_weights"
  [thermo_mpnn_parity]="reversed_hidden,omit_seq_subtraction,zero_edge_features,permuted_output_weights"
  [tier1_variants_parity]="tier1_reversed_hidden,tier1_zero_edge_features,tier1_permuted_output_weights,tier1_shifted_decoding_order,tier1_linear_offsets_on_cyclic"
  [protonpotts_parity]="features.backbone_threshold_0_5,features.atom_threshold_just_below_half,features.backbone_without_o,features.no_arginine_fix,features.no_keep_last_by_number,features.atom_threshold_0_8,features.r_idx_last_plus_one,features.s_first_token_changed,features.chain_labels_inverted,features.x_m_first_atom_flipped,features.x_nudged_by_1e-4,encoder.pair_order_not_permuted,encoder.pair_order_inverse_permutation,encoder.token_order_not_permuted,encoder.etab_not_permuted,encoder.etab_first_axis_only,encoder.output_nudged_1e-6_f64,encoder.output_nudged_4x_band_f32,energy.double_merge,energy.no_merge,energy.merge_without_transpose,energy.upstream_token_order,energy.energy_halved,energy.energy_nudged_just_outside_f64,energy.energy_nudged_4x_band_f32,driver.token_names_in_upstream_order,driver.candidate_rows_reversed,driver.energy_nudged_4x_band,ph_block.uniforms_shifted_by_one,ph_block.combined_lambda_0p4,ph_block.block_size_2,ph_block.forbidden_tokens_not_applied,ph_block.repetitive_window_off,ph_block.wrong_contrast_token,ph_block.zscale_nudged_outside_f64,ph_block.energy_nudged_outside_f64,ph_greedy.uniforms_shifted_by_one,ph_greedy.block_size_2,ph_greedy.forbidden_tokens_not_applied,ph_greedy.repetitive_window_off,ph_greedy.cdf_in_aminx_token_order,ph_greedy.energy_nudged_outside_f64,ph_driver.combined_lambda_0p6,ph_driver.block_size_2,ph_driver.forbidden_tokens_reduced_to_unk,ph_driver.repetitive_window_off,ph_driver.wrong_contrast_token,decoder.bias_ignored,decoder.scalar_temperature,decoder.x_not_zeroed,decoder.fixed_treated_as_designed,decoder.token_order_upstream,decoder.cms_uses_conditional_mask,decoder.uniforms_shifted_one_step,decoder.cdf_in_aminx_order,decoder.nudged_4x_band,ddg.ddg_sign_flipped,ddg.token_names_in_upstream_order,ddg.mutant_rows_reversed,ddg.energies_without_reference_subtracted,ddg.ddg_nudged_4x_band,sample.shared_order_across_rows,sample.uniforms_rows_swapped,sample.all_residues_designed,sample.cdf_in_aminx_order,sample.uniforms_shifted_one_step,sample.single_row_repeated,sample.temperature_one,sample.nudged_4x_band,ph_methods.cdf_in_aminx_order,ph_methods.uniforms_shifted_one,ph_methods.randints_shifted_one,ph_methods.permutations_reversed,ph_methods.field_untempered,ph_methods.zscales_sample_std,ph_methods.temperature_doubled,ph_methods.combined_lambda_wrong,ph_methods.two_phase_frac_wrong,ph_methods.selective_flipped,ph_methods.backend_swapped,ph_methods.selective_source_swapped,ph_methods.valid_mask_all_true,ph_methods.pin_contrast_wrong,ph_methods.centre_among_designable,ph_methods.scan_unmasked,ph_methods.scan_selective_formula,ph_methods.gibbs_outgoing_only,ph_methods.cv_cap_wrong,ph_methods.nudged_4x_band,ph_knobs.position_knn,ph_knobs.position_energy,ph_knobs.swapped,ph_knobs.shifted,ph_objective_knobs.ignore_global,ph_objective_knobs.flip_global,ph_objective_knobs.ignore_adjrep,ph_objective_knobs.ignore_self,ph_objective_knobs.shifted,ph_objective_knobs.block_zscales"
)

# Fail before burning hours, not after. Compares every string above to the
# manifest, so an edit to either side that desynchronises them is caught here.
check_mutants () {
  python3 - "$@" <<'PY'
import sys, tomllib, collections, pathlib
pairs = sys.argv[1:]
manifest = pathlib.Path("tests/knob_gate/branch_manifest.toml")
rows = tomllib.loads(manifest.read_text())["branch"]
by = collections.defaultdict(set)
for row in rows:
    vehicle = row.get("vehicle", {})
    if vehicle.get("kind") == "sidecar":
        by[vehicle["slug"]].add(row["id"])
bad = False
for pair in pairs:
    slug, _, raw = pair.partition("=")
    listed = {p for p in raw.split(",") if p}
    expected = by.get(slug, set())
    if listed != expected:
        bad = True
        print(f"MISMATCH {slug}", file=sys.stderr)
        print(f"  --mutants {sorted(listed)}", file=sys.stderr)
        print(f"  manifest  {sorted(expected)}", file=sys.stderr)
        print(f"  argv-only {sorted(listed - expected)}", file=sys.stderr)
        print(f"  rows-only {sorted(expected - listed)}", file=sys.stderr)
    else:
        print(f"ok {slug}: {len(listed)} mutants match the manifest")
if bad:
    print("REFUSING: step 1c would discard these runs (_coverage.py:309)", file=sys.stderr)
    sys.exit(1)
PY
}

export POTTS_ORACLE_PYTHON=$HOME/projects/aminx-oracles/.venv/bin/python3
export LASER_ORACLE_PYTHON=$HOME/projects/aminx-oracles/.venv/bin/python3
export AMINX_POTTS_ROOT=$HOME/repos/PottsMPNN
export AMINX_PROTONPOTTS_ORACLES=$HOME/projects/aminx-oracles-protonpotts
export AMINX_PROTONPOTTS_STATE=$HOME/scratch/v6_state.npz

# BACKEND (R6, 261013). The aminx (JAX) side of every vehicle runs on a GPU by
# default; the upstream oracles are torch in the CPU-only oracle venv and are
# unaffected. Spike 261013: laser_proofread_parity's clean arm on GPU 3 took
# 1052 s against 54 min on CPU, same inputs and code, and its 450 cell values
# matched the CPU run to 3.2e-15 (both pass at ~5e-15 against upstream).
# AMINX_WAVE_BACKEND=cpu restores the old JAX_PLATFORMS=cpu behaviour.
# Each slug is pinned to one GPU by a hash of its name, so a rerun of a slug
# lands on the same device. Preallocation is off so vehicles can share a GPU.
WAVE_BACKEND=${AMINX_WAVE_BACKEND:-gpu}
read -ra WAVE_GPUS <<< "${AMINX_WAVE_GPUS:-2 3}"
# Groups 5-7 (P4 vehicles) are CPU only (see CPU_PINNED), so the GPU availability check does not apply to them.
NEEDS_GPU=1
case "$GROUP" in 5|6|7) NEEDS_GPU=0 ;; esac
if [ "$WAVE_BACKEND" = "gpu" ] && [ "$NEEDS_GPU" = "1" ]; then
  for g in "${WAVE_GPUS[@]}"; do
    used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo missing)
    # A GPU already holding >8 GB is someone else's (vLLM holds ~21.5 GB on 0/1).
    if [ "$used" = "missing" ] || [ "$used" -gt 8000 ]; then
      echo "REFUSING: GPU $g is unavailable or busy (${used} MiB); set AMINX_WAVE_GPUS or AMINX_WAVE_BACKEND=cpu" >&2
      exit 1
    fi
  done
elif [ "$WAVE_BACKEND" != "cpu" ]; then
  echo "REFUSING: AMINX_WAVE_BACKEND must be gpu or cpu, got $WAVE_BACKEND" >&2
  exit 2
fi
if [ "$WAVE_BACKEND" = "gpu" ] && [ "$NEEDS_GPU" = "1" ]; then echo "backend: gpu (GPUs ${WAVE_GPUS[*]})"; else echo "backend: cpu"; fi

# CPU-PINNED SLUGS (the six P4 slugs, below; no Potts/LASEr/ProtonPotts slug). protonpotts_parity was pinned to CPU at ab6897a0 after its R6 GPU run (e2a185de, at
# 31ded6de) failed f32 bands that had only been measured on CPU (f64 matched to 1e-15, 99 of 99 controls fired).
# The pin is lifted (261013): the f32 bands of ph_block and ph_methods were then re-measured on both backends
# (runs bcbd533d CPU, e6f508a6 GPU) and widened to the measured minimum; see those two sidecars' AMENDED paragraphs.
# P4 vehicles (261105): all six were developed and band-measured on titanix CPU (fampnn_parity_v3 and
# antibmpnn_parity_v3 set/force JAX_PLATFORMS=cpu; the caliby_parity_v2 sidecar estimates are titanix CPU). GPU would
# change the f32 numbers the bands were measured on (the protonpotts lesson above), so they are pinned to CPU.
CPU_PINNED=(caliby_parity_v2 antibmpnn_parity_v3 fampnn_parity_v3 ca_only_parity thermo_mpnn_parity tier1_variants_parity)

backend_env () {
  local v=$1
  if [ "$WAVE_BACKEND" = "cpu" ] || [[ " ${CPU_PINNED[*]} " == *" $v "* ]]; then
    echo "JAX_PLATFORMS=cpu"
    return
  fi
  local idx=$(( $(printf '%s' "$v" | cksum | cut -d' ' -f1) % ${#WAVE_GPUS[@]} ))
  # JAX_PLATFORMS=cuda, not merely unset: protonpotts_parity.py forwards
  # JAX_PLATFORMS to its wave subprocesses with a default of "cpu", and an
  # explicit cuda also makes a missing GPU an error instead of a silent CPU run.
  echo "JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=${WAVE_GPUS[$idx]} XLA_PYTHON_CLIENT_PREALLOCATE=false"
}

# Per-slug extra argv (XARGS) and env (XENV) beyond the shared `--mutants --work-dir --controls-out`. The Potts/LASEr/
# ProtonPotts slugs need none (their inputs are exported above). Every flag below was read from that vehicle's argparse
# (261105). A required titanix input comes from an env var and is a hard REFUSING: when unset; this script cannot verify
# the paths themselves, only that they were named (directories are checked to exist).
need_env () {
  local name=$1 hint=$2
  if [ -z "${!name:-}" ]; then
    echo "REFUSING: $name is unset ($hint)" >&2
    exit 1
  fi
}
need_dir () {
  local name=$1
  if [ ! -d "${!name}" ]; then
    echo "REFUSING: $name=${!name} is not a directory on this host" >&2
    exit 1
  fi
}
slug_extras () {
  local v=$1 out="$HOME/.aminx/sidecars/$1/$H"
  XARGS=()
  XENV=()
  case "$v" in
    caliby_parity_v2)
      need_env AMINX_CALIBY_ORACLE_DIR "sealed Caliby oracle dumps (<stem>.npz + <stem>.manifest.json), pinned by scripts/parity/caliby_oracle_dumps.sha256.json"
      need_dir AMINX_CALIBY_ORACLE_DIR
      XARGS=(--oracle-dir "$AMINX_CALIBY_ORACLE_DIR" --out "$out/caliby_parity_v2.json")
      # --weights-dir defaults to ~/repos/caliby_weights/caliby in the vehicle; only override when named.
      if [ -n "${AMINX_CALIBY_WEIGHTS_DIR:-}" ]; then XARGS+=(--weights-dir "$AMINX_CALIBY_WEIGHTS_DIR"); fi
      XENV=(OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4)
      ;;
    antibmpnn_parity_v3)
      need_env ANTIBMPNN_ORACLE_DIR "sealed v2 AntiBMPNN dump dir (sidecar oracle_location: titanix /home/solab/scratch/p4fix-antibmpnn/oracle)"
      need_dir ANTIBMPNN_ORACLE_DIR
      need_env AMINX_WEIGHTS_DIR "HF snapshot dir (~/.cache/huggingface/hub/models--maraxen--aminx/snapshots/<HF_REVISION>); a checkout holds LFS pointers"
      need_dir AMINX_WEIGHTS_DIR
      # --antibmpnn-root defaults from AMINX_ANTIBMPNN_ROOT, else ~/repos/AntiBMPNN, in the vehicle.
      # No --out flag: the vehicle reports through $BTH_RESULTS_PATH (set by bth) and --controls-out.
      XARGS=(--oracle-dir "$ANTIBMPNN_ORACLE_DIR" --weights-dir "$AMINX_WEIGHTS_DIR")
      if [ -n "${AMINX_ANTIBMPNN_ROOT:-}" ]; then XARGS+=(--antibmpnn-root "$AMINX_ANTIBMPNN_ROOT"); fi
      XENV=(OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4)
      ;;
    fampnn_parity_v3)
      # fampnn_parity_v3.py appends `--band-rule floor` itself and hands off to fampnn_parity.main.
      need_env AMINX_FAMPNN_ORACLE_DIR "sealed FAMPNN dumps with {f32,f64}/ subdirs (launch_fampnn_registered_v3.sh default: /home/solab/scratch/p4-fampnn/oracle)"
      need_dir AMINX_FAMPNN_ORACLE_DIR
      need_env AMINX_FAMPNN_F64TAB_DIR "float64-table diagnostic oracle (launch_fampnn_registered_v3.sh default: /home/solab/scratch/p4fix-fampnn/oracle_f64tab)"
      need_dir AMINX_FAMPNN_F64TAB_DIR
      XARGS=(--oracle-dir "$AMINX_FAMPNN_ORACLE_DIR" --f64tab-dir "$AMINX_FAMPNN_F64TAB_DIR" --out "$out/fampnn_parity_v3.json")
      # --fampnn-root defaults from AMINX_FAMPNN_ROOT, else ~/repos/fampnn, in the vehicle.
      if [ -n "${AMINX_FAMPNN_ROOT:-}" ]; then XARGS+=(--fampnn-root "$AMINX_FAMPNN_ROOT"); fi
      XENV=(OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1)
      ;;
    ca_only_parity|thermo_mpnn_parity|tier1_variants_parity)
      # Upstream .pt checkpoints are found under $AMINX_PARITY_REPOS (os.pathsep list) then ~/repos by
      # tests/parity/reference_utils.find_parity_repo_file; a missing one is a hard failure inside the vehicle,
      # and no --synthetic is passed (a synthetic run cannot grade pass). --out is redirected off the repo-relative default.
      XARGS=(--out "$out/$v.json")
      XENV=(OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4)
      # The aminx side resolves weights via AMINX_WEIGHTS_DIR when set (authoritative, fails closed); pass it through.
      if [ -n "${AMINX_WEIGHTS_DIR:-}" ]; then XENV+=("AMINX_WEIGHTS_DIR=$AMINX_WEIGHTS_DIR"); fi
      ;;
  esac
}

launch () {
  local v=$1 m=${MUT[$1]}
  mkdir -p "$HOME/.aminx/sidecars/$v/$H"
  local -a benv
  read -ra benv <<< "$(backend_env "$v")"
  slug_extras "$v"
  benv+=("${XENV[@]+"${XENV[@]}"}")
  echo "$v -> ${benv[*]}"
  # AMINX_CLOSURE_* switch on scripts/redsox/closure_hook, which records the repo
  # files this process really loads so verify_wave.py can check them against the
  # row's closure. It edits no vehicle (a vehicle is in its own closure).
  nohup setsid env "${benv[@]}" \
    AMINX_CLOSURE_OUT="$HOME/.aminx/sidecars/$v/$H/loaded_files.json" \
    AMINX_CLOSURE_SCRIPT="$v.py" AMINX_CLOSURE_REPO="$REPO" \
    PYTHONPATH="$REPO/scripts/redsox/closure_hook${PYTHONPATH:+:$PYTHONPATH}" \
    bth run --project-slug aminx \
    --output-paths "$HOME/.aminx/sidecars/$v/$H/branch_controls.json" \
    -- uv run --no-sync python3 "scripts/parity/$v.py" \
       --mutants "$m" \
       --work-dir "/tmp/${v}_$H" \
       --controls-out "$HOME/.aminx/sidecars/$v/$H/branch_controls.json" \
       ${XARGS[@]+"${XARGS[@]}"} \
    > "$HOME/${v}_$H.log" 2>&1 < /dev/null &
  echo "launched $v  (--mutants $m)"
  sleep 3
}

# Two groups, not five at once: five concurrent arms put load at 32-37 on 20
# cores and slowed every one of them. Run group 2 after group 1 has exited.
case "$GROUP" in
  1)
    SLUGS=(laser_proofread_parity laser_proofread_unconditional_parity laser_decode_e2e) ;;
  2)
    SLUGS=(potts_ar_refine_exact potts_ar_decode potts_energy_parity laser_score_parity) ;;
  3)
    # Alone: the largest vehicle in the sprint, 202804 rows over 371 per-PDB
    # resumable units per arm. It took ~25 min at e3bc540e, so the isolation is
    # about CPU contention with the other groups, not about duration.
    #
    # It is also the vehicle that proves the exports at :100-102 are load-
    # bearing rather than belt-and-braces. Launched by hand with an otherwise
    # identical `bth run`, it died in under a minute: its oracle arm imports
    # upstream run_utils, which imports seaborn at module scope, and the sprint
    # venv has none -- _oracle_python falls back to sys.executable when
    # POTTS_ORACLE_PYTHON is unset. Through this script it just works.
    SLUGS=(potts_ddg_megascale) ;;
  4)
    # Alone for the same reason as group 3: seven waves back to back, CPU and memory heavy (the sealed P4 dumps are
    # loaded per wave). ProtonPottsMPNN is the third family; its oracle dumps live under AMINX_PROTONPOTTS_ORACLES.
    SLUGS=(protonpotts_parity) ;;
  5)
    # P4, CPU only, 4 threads each (8 of 20 cores). caliby_parity_v2: ~12-17 min on titanix CPU (sidecar estimate);
    # antibmpnn_parity_v3: ~26 min class (v2 graded run is the evidence). Neither is a Potts-scale load, so they share.
    SLUGS=(caliby_parity_v2 antibmpnn_parity_v3) ;;
  6)
    # P4, CPU only. fampnn_parity_v3 ran ~26 min on titanix CPU (4 threads); alone because it also loads two sealed
    # oracle dumps (f32/f64) plus the f64-table diagnostic oracle and the upstream FAMPNN checkout.
    SLUGS=(fampnn_parity_v3) ;;
  7)
    # P4, CPU only. The three small suites (4-5 mutants each, a handful of checkpoints), 4 threads each.
    SLUGS=(ca_only_parity thermo_mpnn_parity tier1_variants_parity) ;;
  positive)
    if [ "$ONLY_STALE" = "1" ] && ! grep -qx laser_proofread_parity <<<"$STALE"; then
      echo "laser_proofread_parity is fresh: its positive control is not needed"
      exit 0
    fi
    # THIS ARM COULD NEVER HAVE RUN. It called the parity script directly,
    # while every graded arm goes through `bth run` (launch(), :95) -- and
    # `$BTH_RESULTS_PATH` is set by bth, not by the script. Launched as it was,
    # it died in under a second with
    #     laser_proofread_parity requires $BTH_RESULTS_PATH
    # and wrote no payload at all. Measured 261002 on b5272b07.
    #
    # So it goes through `bth run` too. It stays ungraded by what it OMITS:
    # no --controls-out, so it writes no branch_controls.json, and no
    # --output-paths into $HOME/.aminx/sidecars/<slug>/<sha>/, which is the
    # directory the ledger and step 1c read. Its work-dir is /tmp/positive_$H,
    # well clear of the graded /tmp/<slug>_$H.
    #
    # EXPECT THE RECORD'S OUTCOME TO READ `fail`, and do not "fix" it. The
    # sidecar's criteria are written for a graded run, where every mutant must
    # be killed; scalar_both_off is a no-op that must SURVIVE. An outcome of
    # `fail` here is therefore the positive control succeeding, and an outcome
    # of `pass` would mean a no-op mutation changed the result. Read the
    # payloads under the work-dir, not the outcome column.
    read -ra benv <<< "$(backend_env laser_proofread_parity)"
    nohup setsid env "${benv[@]}" bth run --project-slug aminx \
      -- uv run --no-sync python3 \
      scripts/parity/laser_proofread_parity.py \
      --mutants scalar_both_off --work-dir "/tmp/positive_$H" \
      > "$HOME/positive_$H.log" 2>&1 < /dev/null &
    echo "launched positive control, ungraded, no --controls-out"
    echo "it must report near-zero; a non-zero result is instrument floor"
    echo "expect the bathos outcome to read 'fail' -- see the comment above"
    exit 0 ;;
  *)
    echo "usage: $0 [--stale-only] [1|2|3|4|5|6|7|positive]" >&2; exit 2 ;;
esac

# check_mutants only inspects the slugs handed to it, so it could not see a slug
# missing from MUT or from every group. laser_score_parity was exactly that: it
# has a manifest row (permute_decoder_layer) and appeared in no group, so no
# invocation of this script would ever have run it and the manifest could never
# have been satisfied. Checked here against the manifest itself.
check_coverage () {
  python3 - "$@" <<'PY'
import pathlib
import sys
import tomllib

known = set(sys.argv[1:])
rows = tomllib.loads(
    pathlib.Path("tests/knob_gate/branch_manifest.toml").read_text()
)["branch"]
sidecar = [r for r in rows if r.get("vehicle", {}).get("kind") == "sidecar"]
slugs = {r["vehicle"]["slug"] for r in sidecar}
missing = sorted(slugs - known)
extra = sorted(known - slugs)
if missing:
    print(
        f"REFUSING: {len(missing)} manifest sidecar slug(s) are in no launch "
        f"group, so the manifest could never be satisfied: {missing}",
        file=sys.stderr,
    )
if extra:
    print(
        f"REFUSING: {len(extra)} launched slug(s) have no manifest rows: {extra}",
        file=sys.stderr,
    )
if missing or extra:
    sys.exit(1)
print(f"ok coverage: {len(slugs)} manifest slugs all reachable from a group")
PY
}

# Every slug any group can launch, so the check is about the SCRIPT rather than
# about whichever group happens to be running.
check_coverage laser_proofread_parity laser_proofread_unconditional_parity \
               laser_decode_e2e \
               potts_ar_refine_exact potts_ar_decode potts_energy_parity \
               laser_score_parity potts_ddg_megascale protonpotts_parity \
               caliby_parity_v2 antibmpnn_parity_v3 fampnn_parity_v3 \
               ca_only_parity thermo_mpnn_parity tier1_variants_parity

if [ "$ONLY_STALE" = "1" ]; then
  KEEP=()
  for s in "${SLUGS[@]}"; do
    if grep -qx "$s" <<<"$STALE"; then KEEP+=("$s"); fi
  done
  SLUGS=("${KEEP[@]}")
  if [ "${#SLUGS[@]}" -eq 0 ]; then
    echo "nothing stale in group $GROUP: no run to launch"
    exit 0
  fi
fi

ARGS=()
for s in "${SLUGS[@]}"; do ARGS+=("$s=${MUT[$s]}"); done
check_mutants "${ARGS[@]}"

# Resolve every slug's required inputs BEFORE launching any of them, so a missing env var on the second slug of a
# group cannot leave the first one running alone.
for s in "${SLUGS[@]}"; do slug_extras "$s"; done

for s in "${SLUGS[@]}"; do launch "$s"; done

sleep 30
ps -eo pid,etime,args | grep -E "[p]otts_|[l]aser_|[c]aliby_|[a]ntibmpnn_|[f]ampnn_|[c]a_only|[t]hermo_|[t]ier1_" | grep "uv run" | cut -c1-60
