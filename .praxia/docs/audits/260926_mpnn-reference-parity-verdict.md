---
title: aminx MPNN reference-parity verdict
description: 'Browser-validation Phase 1 literature-parity verdict: global FAIL (sampling prerequisite missing + 8 confirmed core defects); 3/17 paths advance'
status: final
task_id: 260923_aminx-browser-validation
date: '260926'
verdict: FAIL
base_sha: e4b86a0c0da96313449c2c72fe212eca994b6eea
---
# aminx MPNN reference-parity verdict (browser-validation Phase 1)

## Summary

**Global literature-parity grade: FAIL** (`parity_validate_mpnn`, run `e3d3ffa2`, HEAD `e4b86a0c`, clean tree,
campaign `aminx-bv-layer-a` / `32a520cf`). The FAIL is forced by the prerequisite mapping (O1),
not by a cap-lattice floor: `prereq_status = missing` because no `layer_a_sampling_validate`
exists. The sampling calibrate ended `budget_exceeded` (run `4f07f070`, 483 h projected against a
16 h budget even on one TITAN RTX, D10), so no sampling parameters were ever pre-registered.
`compute_grade` was therefore never called. The spec's expected grade was PARTIAL, which
depended on this sprint's distributional sampling test running.

**This is not a finding that sampling is broken.** Autoregressive sampling is already validated
against the reference by `test_autoregressive_sampling_parity` (`tests/parity/test_full_model_parity.py:480`,
in the 30/30 `parity_heavy` suite): with the reference's decoding order it requires at least 95%
token agreement and log-prob Pearson correlation of at least 0.95, for both weight sources.
Invariant 5 (below) additionally shows aminx's one-step sampling distribution equals
softmax((logits + bias)/T) to TV < 0.06 over 4000 draws. What did not run is the stricter,
pre-registered distributional comparison (`layer_a_sampling_validate`: 1500 draws per lane,
T = 0.1 and 1.0, tied and ligand lanes). It could not be calibrated within budget, mostly because
of aminx's per-draw cost (F-T6/F-T8).

The grade inputs, as measured, for when a sampling validate eventually exists:

| Input | Value | Cap-lattice reading |
|---|---|---|
| `prereq_status` / rung | `missing` / R4 | forces FAIL (O1) |
| exact validate (T7, H_v `4c3a9c3b`) | `partial_headroom` | R2-eligible, but FROZEN check fails (see F-T11b) |
| `clause_parity_pct` (pre-registered) | 19/33 = **0.576** | ≥ 0.5: not a FAIL floor, caps below PARITY |
| `invariant_pass` | **true**, 8/8, 0 skipped | ok |
| `adversarial_survived` | **false**: 8 confirmed core defects | a FAIL floor on its own |
| `ambiguity_load` | `load_bearing` (1: `topk_tie_break`) | caps below PARITY |

**Even with a validated sampling tier, the grade would be FAIL**, because
`adversarial_survived = false` is a FAIL floor. Reaching PARTIAL requires fixing the confirmed
core defects (below), not just more compute.

**Paths that advance to Phase 2 (advance table): P00, P03, P04, which is 3 of 17.**

## Clause scorecard (pre-registered, sha256 `fb3f17b7…`, committed before Phase 3)

40 clauses: 34 core, plus 6 designed non-core deviations.

| Core verdict | n | Clauses |
|---|---|---|
| MATCH | 19 | ar_visibility_mask, augment_eps, dropout_off_at_inference, knn_k_clamp, ligand_context_fusion†, message_passing_decoder, message_passing_encoder, omit_as_bias, packer_graph_and_context_size, packer_mixture_parameterisation, positional_encoding_protein, rbf_edge_features, score_nll_full_vocab, side_chain_context_toggle, temperature_default, temperature_on_logits_plus_bias, tied_group_step_placement, unconditional_scoring_backbone_only, undrawn_sequence_embedding |
| DEVIATION | 10 | checkpoint_topology_source, conditional_score_context_mask, knn_masked_pair_sentinel, ligand_atom_context, membrane_label_encoding, positional_encoding_ligand, random_order_fixed_first, sample_log_probs_output, tied_scoring_logit_fusion, weight_conversion_bias_handling |
| MISSING | 4 | ligand_cutoff_for_score, membrane_label_construction, packer_chi_sampling, score_only_multi_order_average |
| AMBIGUOUS | 1 | topk_tie_break (excluded from the ratio) |

The 6 designed non-core deviations are `decoding_order_default`, `tied_fixed_override`,
`tied_fusion_weights`, `tied_intra_group_visibility`, `tied_last_member_bias` and
`x_omit_at_sampling`.

† The attackers refuted `ligand_context_fusion`'s MATCH (defect `ligand_context_fusion_bias`,
below). The pre-registered clauses are not edited after the fact, so the graded value stays 0.576.
The adversarially corrected figure is **18/33 = 0.545**.

## Confirmed defects (Phase 4 adjudication, `litparity/adjudication.json`)

Rule: a defect is confirmed with at least 2 attacker votes OR a runnable test. Every defect below
has a runnable probe, and the orchestrator re-read the cited source lines for the two new ones.

| Defect | Paths | Sev | Evidence | New? |
|---|---|---|---|---|
| ligand_atom_context | P11 | core | `PrxteinLigandMPNN` wires atom_context_num = 16, not the checkpoint's 25; up to 3.96 nats on 5L33 | no |
| membrane_label_encoding | P13, P01 | core | `physics_projection.bias` is random-init (absmax 0.571); zeroing it moves logits up to 5.48 nats | no |
| ligand_context_fusion_bias | P11 | core | `ligand_mpnn.py:224` `v_c` Linear has a bias; reference `V_C` has `bias=False` (`model_utils.py:50`); the shipped bias equals random init | **yes** |
| positional_encoding_ligand | P11, P01 | core | aminx `use_bias=False` drops a real reference bias | no |
| weight_conversion_bias_handling | P01, P11, P13 | core | 14 reference `bias=False` sites against 12 aminx sites: 2 invented, 1 dropped | no |
| checkpoint_topology_source | P01, P11 | core | misparses a membrane checkpoint's noise suffix as atom_context_num | no |
| conditional_score_context_mask | P05, P06 | core | 2-hop self-identity leak under the 1−I default: up to 0.84 nats from a residue's own token | no |
| sample_log_probs_output | P07, P09 | core | fixed-position sample log-probs are full rows (mean abs 5.5 nats), where the reference gives 0 | no |
| ar_mask_order_rank_convention_mismatch | P07, P19 | major | `generate_ar_mask` reads rank[pos]; `sampling/sample.py:181` and `inference/decode/ste.py:383` pass order[step]; wrong mask for any custom order (24/64 cells, L = 8) | **yes** |

Rejected: `ar_visibility_mask` (struct lens). Full-recompute and incremental reveal were
bit-exact, so the MATCH stands.

## Invariants: red/green evidence

`tests/parity/test_mpnn_reference_invariants.py` (8 tests, marker `parity_heavy`) asserts each
property on BOTH the pinned PyTorch reference and aminx. In the graded run: **8 passed, 0 skipped**.
Each invariant was then shown to go **red** on a named injected `src/aminx` defect, with `src`
restored clean after each (`outputs/browser_validation/layer_a/invariant_redchecks/`):

| # | Invariant | Injected defect | Red assertion |
|---|---|---|---|
| 1 | AR causality | inverted AR attention mask (`decoder.py:144`) | future residue 16 leaked into 15 (0.466) |
| 2 | unconditional ⟂ sequence | sequence embedded into the unconditional decoder | log-probs depend on the sequence (2.77) |
| 3 | fixed-first | fixed-token override dropped | fixed position not held |
| 4 | omit/X never sampled | omit bias dropped from the draw | sampled an omitted AA or X |
| 5 | one-step = softmax((l+b)/T) | T multiplied, not divided | TV = 0.367 (bar 0.06) |
| 6 | tied positions | tie broadcast only to the representative | tie group [0, 276] got different tokens |
| 7 | membrane labels live | physics labels discarded | labels 0 vs 1 move log-probs 0.0 |
| 8 | k-NN = brute force | farthest-k instead of nearest-k | 68 no-tie residues differ |

## Per-path table (exact tier, T7 validate run `53ba3928`, fixture set B, 110 rows)

Worst row per path. The bar is the pre-registered tolerance, and ratio = value / bar.

| Path | Rows | Status | Worst row | Value | Bar | Ratio |
|---|---|---|---|---|---|---|
| P00 | 2 | validated | log_probs max_abs | 0 | 1e-4 | 0 |
| P01 | 1 | validated | weights max_abs (eqx vs pt_convert) | 0 | 0 | 0 |
| P02 | 20 | validated + over_bar‡ | backbone_coords max_abs | 0 | 1e-5 | 0 |
| P03 | 8 | validated | neighbor_set_k32 n_mismatch | 0 | 0 | 0 |
| P04 | 4 | validated | log_probs max_abs | 3.6e-5 | 1e-4 | 0.36 |
| P05 | 12 | validated + not_advanced | log_probs max_abs | 4.7e-5 | 1e-4 | 0.47 |
| P06 | 8 | validated + not_advanced | log_probs max_abs | 4.8e-5 | 1e-4 | 0.48 |
| P09 | 4 | not_advanced | log_probs max_abs | 6.4e-5 | 1e-4 | 0.64 |
| P11 | 3 | not_advanced | context_on log_probs max_abs | 0.222 | 1e-4 | 2221 |
| P12 | 16 | validated + not_advanced | conditional log_probs max_abs | 4.4e-5 | 1e-4 | 0.44 |
| P13 | 28 | not_advanced | per_residue log_probs max_abs | 5.2 | 1e-4 | 5.2e4 (argmax mismatches: 3) |
| P14 | 4 | not_implemented | packer_mixture | – | – | – |

P07, P08, P10, P19 and P26 have no exact-tier rows; their evidence would come from the sampling
tier, which is `budget_exceeded`.

‡ P02's `over_bar` rows are the informational `residue_alignment n_common` rows, which have
`bar = null`. The validate classifier marks a null-bar row `over_bar`. Every real P02 comparison
(mask, chain labels, sequence, backbone coordinates) is exact. This is a harness classification
quirk (F-T11c), not a divergence, but it does block P02 in the advance table as committed.

## Advance table (`outputs/browser_validation/layer_a/advance_table.json`, `advance_check.py`)

| P-ID | Advances | Blocking reasons |
|---|---|---|
| P00 | **yes** | – |
| P01 | no | defects: checkpoint_topology_source, membrane_label_encoding, positional_encoding_ligand, weight_conversion_bias_handling |
| P02 | no | exact rows not all validated (the null-bar quirk ‡) |
| P03 | **yes** | – |
| P04 | **yes** | – |
| P05 | no | not_advanced rows; defect conditional_score_context_mask |
| P06 | no | not_advanced rows; defect conditional_score_context_mask |
| P07 | no | no layer-(a) rows; distributional sampling test not run (calibrate budget_exceeded); defects ar_mask_order_rank_convention_mismatch, sample_log_probs_output |
| P08 | no | no layer-(a) rows; distributional sampling test not run (calibrate budget_exceeded) |
| P09 | no | not_advanced/over-bar log_probs; sampling budget_exceeded; defect sample_log_probs_output |
| P10 | no | no layer-(a) rows |
| P11 | no | over bar (0.222 nats); sampling budget_exceeded; 5 defects |
| P12 | no | not_advanced rows |
| P13 | no | 24 over-bar rows (≈ 5 nats); defects membrane_label_encoding, weight_conversion_bias_handling |
| P14 | no | not_implemented |
| P19 | no | no layer-(a) rows; defect ar_mask_order_rank_convention_mismatch |
| P26 | no | no layer-(a) rows |

## Run records

| Run | Record (cool tier) | Use |
|---|---|---|
| `e3d3ffa2` | completed, outcome **fail**, exit 0, git_dirty false, sidecar `95589c63…` (= committed file) | **graded run**: wrote the committed `layer_a/parity_validate.json` (last writer, 15:49:48) |
| `f79e26ce` | identical: completed, fail, exit 0, clean, same sidecar, same grade inputs | corroborating duplicate (both runs were launched against one clean worktree; an earlier lock-retry got through) |
| `6451771f` | completed, outcome '' (no sidecar resolved), git_dirty true | **void**, see F-T11a |

The warm index (`bth sql`) shows all three as `status=running, outcome=NULL`. A `bth compact` that
overlaps a run ingests its in-flight row, and compact never re-ingests an id it already has. The
only remedy is `bth compact --force-rebuild`, which was not run: it has wiped a table before and
needs a user decision. So `bth campaign attest-parity` refused (`outcome='None'`, expected pass or
partial). It would refuse the true `fail` as well, so `parity_run_id` stays unbound and the claim is
not attested. The layer-a campaign is left **open** for the future sampling validate.

## Findings from execution

- **F-T6/F-T8 (sampler cost, not correctness).** aminx's autoregressive decode ran the decoder
  over all L positions at every step and kept one position's logits
  (`inference/decode/autoregressive.py`), so decoder work per sequence was O(L²·k), against the
  reference's O(L·k) incremental update. The results were identical (full recompute and incremental
  reveal are bit-exact, struct attacker), but the cost was 25–40× the reference's per-draw cost. At the pre-registered n = 1500, the sampling calibrate projected
  435–483 h (71–105 h per lane) against a 16 h budget, even on a TITAN RTX. Reference CPU draw cost
  also varies up to 3× between runs. **The distributional sampling test cannot be
  calibrated at the pre-registered protocol with that sampler.** The fix is an incremental sampler,
  not a reduced protocol.
  **Fixed (260926, branch `feat/incremental-ar-sampler`, 8bcb2437):** each wave now decodes only its
  own positions against a per-layer cache, exact vs full recompute (identical tokens, logit gap
  9.5e-7). Pre-registered benchmark `bench_ar_incremental` (bathos run `b5f7ade5`, outcome
  **pass**, clean tree): 85.8× faster at L = 512 on CPU (42.7 s → 0.50 s per sequence); 4× length →
  4.8× time (full recompute: 13.2×, the negative control); `auto` picks the incremental branch
  (1.07× `force`). The sampling calibrate should be re-run before re-grading.
- **F-T8 (bathos exit override).** bathos 84be544e forces `outcome = error` on any non-zero exit
  before evaluating `[outcomes]`. Budget-fail and graded-FAIL scripts must write their result and
  exit 0 (fixes 799a1c4a, e4b86a0c).
- **F-T11a (bathos wrapped-command parsing).** The local `bth`'s `_find_script_path` takes the value
  of the first two-token uv option (`--extra dev`) as the script, so no sidecar is resolved and the
  record carries `outcome = ''`. Run `6451771f` is void for this reason, and any `bth compact` that
  overlaps a run (inside the script or from another shell) freezes its warm row at `running`. The graded run used `=`-form options, synced
  and compacted *before* `bth run`, and ran from a clean detached worktree (D11).
- **F-T11b (shared FROZEN list).** The T8 sampling work legitimately changed
  `layer_a_sampling.py`, `layer_a_sampling_validate.py` and `titanix_run.sh` after the exact
  validate's H_v, and the grader checks a single FROZEN list for both stems. The exact prerequisite
  therefore also reads "not ready". This is moot while sampling is missing, but a per-stem FROZEN
  list is needed before any regrade.
- **F-T11c.** A null-bar informational row is classified `over_bar` (P02).

## Deviations from the spec

- D1–D6: plumbing adjustments in T4–T6. Examples: D3, the bathos-tool check and the two-token
  `--prerelease allow`; D6, the campaign-on-titanix check greps file contents because campaign
  files are UUID-named. Each is in the T4–T6 commit messages.
- D7: bathos 84be544e never persists `runs.metadata`, so prerequisite and identity gates bind on
  native columns (git_hash, git_dirty, dependency_lock_sha256, sidecar_sha256, outcome) plus
  `git show H_v:` hashes.
- D8: Git LFS weights are materialized on titanix from a sha256-verified store (titanix has no
  git-lfs), skipping tracked symlinks.
- D9: the k-NN control queries the first unmasked residue (T7 ctrl_blind fix, 4c3a9c3b).
- D10 (user-approved 260925): the sampling tier runs on one titanix TITAN RTX (GPU 2) with a
  batched jitted sampler. It still exceeded 16 h, so it fell through to the honest budget-fail.
- D11 (260926): the graded run synced and compacted before `bth run` and ran the script with
  `--no-sync`, from a clean detached worktree at HEAD, because of F-T11a.

## Reproduction plan

1. Check out each validate's **H_v**, never HEAD: exact `4c3a9c3b`; sampling does not exist yet.
   Titanix runs go through `scripts/browser_validation/titanix_launch.sh`, which pushes by URL to
   `titanix:/home/solab/bv/aminx.git` and materializes LFS.
2. For the sampling tier: either choose a reduced protocol (a user decision, e.g. smaller n or
   fewer lanes, pre-registered in a new sidecar before any run), or, preferably, first make the
   sampler incremental (O(L·k) per sequence) and re-calibrate.
   Then run calibrate → commit params → validate.
3. Fix the 8 confirmed core defects and the major one. Most are one-line bias/topology fixes in
   `ligand_mpnn.py`, `encoder.py` and weight conversion, plus the conditional context mask. Then
   re-run the exact validate, update the FROZEN list per stem (F-T11b), and re-run
   `parity_validate_mpnn.py` under `bth run --campaign-id 32a520cf-…` from a clean tree.
4. Re-grade. PARTIAL becomes reachable once `adversarial_survived` holds and
   `clause_parity_pct ≥ 0.5`. PARITY additionally needs every core clause MATCH, zero ambiguity
   load and R0/R1.

## Post-verdict finding F-S1 (260927, sprint `260927_aminx-browser-export-phase2a`, T9)

**Facts, no re-grading** (the sampling tier is still absent from the clause scorecard above;
this note only documents a harness defect found and fixed while building its Phase 2a
budget-floor probe).

- **Defect.** `layer_a_sampling.aminx_sample_batch`'s vmapped batching (`_vmapped_sample`,
  introduced alongside the incremental sampler, `8bcb2437`/`9f800cfa`) builds ONE
  `InferenceBundle` from `waves[0]` and, per draw `i`, swaps only `bundle.wave` via
  `eqx.tree_at(lambda x: x.wave, bundle, wave)` before calling
  `sample_autoregressive.kernel`. `bundle.conditioning.ar_mask` is computed ONCE by
  `build_inference_bundle` from whichever wave built the bundle (`waves[0]`) and is never
  recomputed for `i >= 1`. The kernel reads `cond.ar_mask` (`autoregressive.py:485`, `:631`),
  not `bundle.wave` directly, for wave-schedule visibility -- so every draw `i >= 1` decoded
  under draw 0's autoregressive visibility, not its own. This is the harness defect the
  Phase 2a spec (D-F) named F-S1 and directed T9 to fix.
- **Test evidence (pre-fix RED, quoted from this task's own red-check).**
  `tests/parity/test_layer_a_sampling_batched.py::test_per_draw_ar_mask_structural`, run
  against the pre-fix construction (`eqx.tree_at(lambda x: x.wave, bundle, wave)` alone, no
  `ar_mask` recompute): 444/1024 elements of the rebuilt `ar_mask` mismatched the correct
  per-draw value (`generate_wave_ar_mask(waves[i], tie_group_map)`) at draw `i = 1` of 3 (first
  failing draw; the assertion loop stops at the first mismatch). Draw 0 was exact by
  construction (its own wave built the shared bundle), matching this module's own
  pre-existing docstring note of an empirically observed "0/85, 15/85, 10/85" per-draw
  mismatch pattern on 5L33 (V15) -- draw 0 exact, every later draw wrong at many positions.
- **Fix.** `layer_a_sampling._bundle_for_draw(bundle, wave)` (new) recomputes `ar_mask` from
  the draw's OWN `wave` via `generate_wave_ar_mask` (the same function
  `build_inference_bundle`'s own "wave" arm uses) and swaps both `.wave` and
  `.conditioning.ar_mask` together; `_vmapped_sample`'s inner `one(key, wave)` now calls this
  helper instead of swapping `.wave` alone. Post-fix, the same structural test is GREEN for
  all 3 draws, and the empirical test
  (`test_batched_vs_unbatched_empirical`, synthetic L=96, T=0.1, n=3 draws) shows <= 1
  mismatched position per draw between the batched harness and an unbatched per-draw kernel
  call (the residual is fused-XLA-vmap-vs-eager rounding, per this module's own documented
  caveat, not an ar_mask defect).
- **`incremental` plumbing (D-F).** `sample_autoregressive.kernel` gained an `incremental:
  Literal["auto", "off", "force"] = "auto"` parameter, forwarded to `AutoregressiveConfig`
  (previously hardcoded to its "auto" default with no caller override). Default behavior is
  provably unchanged (`test_kernel_incremental_default_bitwise`: an unspecified call and an
  explicit `incremental="auto"` call are bitwise identical). `layer_a_sampling.
  incremental_predicates_host` is a host NumPy mirror of the kernel's three-way device
  predicate (`consistent`, `identity_frame`, `fits_slab`, `autoregressive.py:591-756`),
  checked against a from-scratch jnp re-derivation
  (`test_host_predicates_match_device`, 6 sub-cases incl. a fixed-position lane, a planted
  inconsistent `ar_mask`, a non-identity `state_position_map`, a wave exceeding its own
  declared slab capacity, and a case isolating the `valid_nbr` term specifically) and against
  live kernel behavior (`test_force_equals_off_when_host_true`,
  `test_planted_inconsistent_mask_not_forced`).
- **Budget-floor probe (`layer_a_sampling_budget_floor.py`, campaign `aminx-bv-layer-a`,
  `32a520cf`).** Reuses `layer_a_sampling_calibrate._budget_at_floor` (the `n =
  N_REQUIRED_FLOOR` validate-cost projection) unmodified against the FIXED harness, plus a
  force-vs-off `fastpath_ratio` and a force-vs-force `aa_ratio` sanity control at the largest
  set-B fixture (4YOW, lane P07@1.0), and a diagnostic `n_draws_forced_off` per lane over the
  floor's own draw allocation. **Not yet run** (titanix GPU 2 dispatch and its `check_run.sh`
  verification are orchestrator-only steps per this sprint's division of labor and this
  worktree's own `titanix_launch.sh` header comment; `check_run.sh` itself does not exist yet
  in this branch, being a separate track's (T2) deliverable) -- the sidecar and its
  `.cases.json` are pre-registered and committed, and every declared outcome was independently
  verified against `bathos.sidecar.evaluate_outcome` (7 cases, one per outcome plus one extra
  `ctrl_blind` case for the sentinel-value path, all matching their declared `expect` label).
  This finding is therefore facts-plus-fix-plus-test-evidence only; the actual floor numbers
  (`fastpath_ratio`, `aa_ratio`, `budget_wall_hours` on real GPU hardware) are not yet measured
  and are NOT claimed here.
