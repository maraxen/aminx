---
title: "aminx MPNN browser validation — layered parity (PyTorch → JAX → exported artifact → browser) and in-browser benchmarking for every inference path"
date: '260923'
status: approved
task_id: 260923_aminx-browser-validation
converged_spec_sha256: 0307da720ff5cb9f85d081873432d0bd7d9891709a6e3ddc736a0f203f490c04
approved: '260924'
companion: 260923_aminx-browser-validation-carry-forward.md
---

# aminx MPNN browser validation (epic spec, draft 4)

## Overview

**One sentence.** Establish, with pre-registered bathos hypotheses and controls, which aminx
ProteinMPNN/LigandMPNN inference paths reproduce the original PyTorch implementations (layer a),
survive export to a portable artifact (layer b), and execute correctly and measurably fast inside a
browser (layer c), and hand the advisor an artifact bundle plus a per-path, per-layer report card that
claims nothing it has not measured.

**Why.** Marielle's advisor wants a server-free browser app for protein design built on aminx. Today
layer (a) exists but is weaker than advertised (Recon corrections); no real-weight aminx model has been
exported by any route (layer b); no aminx artifact has executed in a browser (layer c). The README's
"≥ 0.999 Pearson across all five decoding paths" claim is contradicted by the repo's own gates
(0.95 / 0.90 / 0.999) and must not reach the advisor.

**Principle: layers do not transfer.** A verdict at layer (a) says nothing about (b), and (b) nothing
about (c). Each layer has its own hypotheses, sidecars, controls and verdicts. Every integer-valued
output (neighbour indices, decoding order, tokens given identical noise, argmax) is compared EXACTLY at
every layer: every export defect measured so far was an integer divergence under bit-identical floats
(IREE tie order; `random.permutation` miscompile).

### Epic phases

| Phase | Goal | Depends on | Fixer tasks |
|---|---|---|---|
| 0 — De-risking spikes | Does a real aminx callable trace through the xtrax safety gate; does jax2onnx convert one real path; does ORT Web (wasm EP) run it headless | none | T1, T2, T3 |
| 1 — Layer (a) literature parity + bathos staging | PyTorch↔JAX parity for every in-scope path on realistic inputs (L ≥ k), per-path bars, controls, synthetic-truth metric checks, Mode-A literature parity, a per-path advance table; calibrate/validate runs on titanix (ODQ-13) | independent of Phase 0 | T4–T11 |
| 2 — Layer (b) export + divergence mapping | Export every v1 path (ODQ-10) with order and noise as host inputs; per-route gates (ODQ-1, AC-17/18); padding invariance | per-path advance table (T11 `advance_check.py`), NOT `requires_parity_stem`; T1; T2 | goals + ACs |
| 3 — Layer (c) browser execution + benchmark | Phase-2 artifacts in real browsers (ODQ-1, ODQ-8); JS host logic incl. noise/order generators; benchmark harness with an interleaved native arm | Phase 2 | goals + ACs |
| 4 — Advisor deliverable | Static page + artifacts + manifest + report card, every number traced to a bathos run | Phases 1–3 | goals + ACs |

xtrax-side work (emsdk IREE runtime, WebGPU backend, any jax2onnx plugin) is filed as xtrax backlog
items when a phase needs it (ODQ-12).

### Comparison taxonomy (every layer)

- **EXACT**: bitwise or integer equality: weight tensors, integer arrays, neighbour sets (no-tie
  residues, see bars), decoding orders, tokens given identical noise, argmax where the top-2 margin
  exceeds 2× the bar.
- **TOLERANCE**: every float comparison reports max-abs AND Pearson AND the aggregate scalar claims are
  denominated in (per-sequence NLL via the reference's `get_score`). Max-abs is the gate; Pearson is
  affine-blind (`scripts/measure_conditional_nll_parity.py`: `logits * 2.0` still gives r = 0.974).
- **STATISTICAL**: sampled outputs under different RNGs (torch Philox vs JAX threefry) are compared by
  pre-registered EQUIVALENCE tests (intersection-union across lanes, excess-JS margin). No token-level
  agreement is claimed across RNGs, and non-significance is never read as equivalence.

Exact-tier runs: CPU, float32, `jax_default_matmul_precision="highest"`, backbone noise 0, dropout off,
decoding order from the reference formula `argsort((mask*chain_mask + 1e-4) * |randn|)` (LigandMPNN
`model_utils.py:217-220`) computed once and fed to both sides. GPU runs are never pooled with CPU.

### Inference-path inventory and disposition

Source: `recon_inference_paths_v2.md` plus P21/P23 found while drafting. Enforced by a sweep (AC-5).

| ID | Path (aminx anchor, origin/main a5ced9c7) | Original analogue | Layers | Tier(s) / notes |
|---|---|---|---|---|
| P00 | LigandMPNN `protein_mpnn` path vs dauparas/ProteinMPNN `protein_mpnn_utils.py`, same v_48_020 weights | both originals | a | EXACT weights + TOL logits; makes 26ec57ac also a ProteinMPNN validation for protein-only paths (ODQ-6) |
| P01 | Weights — `io/weights.py` (`.eqx.zst`, HF pin `25fb7f6e…`, `LEGACY_ALIAS_MAP`, `get_topology_for_checkpoint`) | `.pt` | a, b, c | EXACT shipped `.eqx.zst` == `.pt` after layout map; sha256 carried to b/c |
| P02 | Structure ingestion — aminx PDB parse | `data_utils.parse_PDB` | a, c | EXACT integer arrays; coords ≤ 1e-5 Å. Layer c = JS parser. URI fetchers (`afdb`, `mdcath`) out of scope (network; no analogue) |
| P03 | Featurisation + k-NN — `model/features.py` (`top_k` `:46-92`, k clamp `:183`) | `ProteinFeatures` | a, b, c | EXACT neighbour sets on no-tie residues; TOL edge features; tie-lattice fixture recorded |
| P04 | Unconditional logits — `inference/score_unconditional.py:16` | `unconditional_probs_only` | a, b, c | TOL |
| P05 | Conditional logits — `inference/score_conditional.py:22,44,164`, `decoder.py:655` | `conditional_probs_only` / `single_aa_score` | a, b, c | TOL |
| P06 | Scoring / NLL — `scoring/score.py:87,114` | `score_only` | a, b, c | TOL pinned order; STATISTICAL multi-order mean |
| P07 | AR sampling — `inference.sample_autoregressive.kernel` with host `WaveScheduleBundle` (the ONE native sampler for layers a and b); legacy `aminx.sampling.sample` inventoried only | `sample` | a, b, c | a: teacher-forced TOL + STATISTICAL; b/c: EXACT tokens given host order + noise (ODQ-5, O2 contract) |
| P08 | omit_AA / per-residue omit via `bias` | `run.py:405-407` | a (b, c inherit P07) | bias configuration (ODQ-4); zero omitted-AA count |
| P09 | Tied positions — `tie_group_map`, `specs.py:314` | `symmetry_residues`/`symmetry_weights` | a, b, c | scoring lane TOL; sampling lane (P09-s) restricted per the bars table; tie-order flip control |
| P10 | Multi-state PoE — `sampling/multistate_poe.py:102,241,455` | matrix lanes only | a (matrix lanes) | other strategies `novel`; b/c deferred (ODQ-10) |
| P11 | LigandMPNN ligand context — `model/ligand_mpnn.py:25`, `ligand_features.py` | `ligand_mpnn` | a, b, c | as P04–P07, context on/off |
| P12 | SolubleMPNN weights | `soluble_mpnn` | a, b, c | weight variant of P04–P07 |
| P13 | Membrane MPNN (`physics_feature_dim=3`) | per-residue / global label membrane | a, b, c | TOL with VARIED labels + live-label invariant |
| P14 | Packer — `model/packer.py:651` | `sc_utils.Packer` | a; b, c second wave | TOL mixture params, SHIPPED weights, real structures; chi draw vs analytic mixture |
| P15 | Averaged encoding — `host/averaging.py` | none | out | aminx-only; internal parity_fast only |
| P16 | STE optimisation — `inference/optimize_ste.py:51` | none | out | gradient-based, no original |
| P17 | Jacobian | none | out | gradient analysis |
| P18 | Inspect | none | out | no numeric output |
| P19 | Decoding-order generation — `utils/decoding_order.py:28,110`; `schedule_selector.py:352` | reference argsort formula | a; c (JS generator) | a: reconciliation clause (`fixed_n_to_c` default). b: not exported (host input). c: fixed-first + pairwise-precedence uniformity |
| P20 | Chromatic wave schedules — `schedule_selector.py:222-349` | none | out | research schedules |
| P21 | MBR reranking — `sampling/mbr_consensus.py` | none | out | post-hoc utility; core is P05 |
| P22 | Potts | none | out | separate family (ADR 260605) |
| P23 | `model/diffusion_mpnn.py` | none | out | research subclass, no reference checkpoint |
| P24 | EBM | ProteinEBM | out | own parity campaign |
| P25 | CA-only models | ProteinMPNN `--ca_only` | not implemented | ODQ-2 DECIDED out of scope; listed as a known limit |
| P26 | Host runner / CLI / specs — `host/runner.py:76`, `run/_exports.py:60-99` | CLI `run.py` | a (existing parity_fast) | browser app replaces the host layer; its pre/post-processing is P27 and layer c |
| P27 | Bucketing / padding — `tiling/bucketing.py:32,43-67`, `tiling/pad.py` | none | b, c | score preservation iff L ≥ k_neighbors(checkpoint); refuse below k and above max bucket; reconcile ladders |
| P28 | Campaign orchestration — `host/campaign.py` | none | out | not an inference path |

### Pre-registered layer-(a) bars

This spec's own bars; the README's figures are not inherited. **Calibration may tighten a bar, never
loosen it.** Headroom rule: if a path's calibration measurement exceeds bar/2 it is marked
`not_advanced`. Validation STILL measures and reports it (`status = "not_advanced"`); any
`n_not_advanced > 0` makes the outcome `partial_headroom`, never `pass`, and the path does not advance
in T11's table.

| Quantity | Paths | Tier | Bar | Rationale |
|---|---|---|---|---|
| Weight tensors | P01, P00 | EXACT | max-abs 0.0, f32 | conversion is a copy |
| Parsed arrays | P02 | EXACT / TOL | integers equal; coords ≤ 1e-5 Å | PDB has 3 decimals |
| Neighbour sets, no-tie residues (L ≥ 64) | P03, P11, P14 | EXACT | 100 % set equality over residues whose k-th vs (k+1)-th CA distance gap > 1e-4 Å (computed per residue in `fixtures.py`); others excluded and counted as `n_near_tie_excluded` | 3-decimal coords make boundary near-ties real |
| Neighbour sets, tie lattice | P03 | EXACT (recorded) | disagreement is a DEVIATION clause, not a bar failure | `lax.sort` vs `torch.topk` tie-break unverified |
| Edge features | P03 | TOL | ≤ 2e-5 | `test_full_model_parity.py:300` |
| Log-probs | P04, P05, P06 (pinned), P09 scoring (k-NN-disjoint groups only), P11–P13 | TOL | max-abs ≤ 1e-4 nats AND Pearson ≥ 0.9999 | re-derived on SHIPPED weights (correction 5); P09 carries the same tie-visibility asymmetry as P09-s (R2-C3) |
| Per-sequence NLL | P05, P06, P11–P13 | TOL | \|Δ\| ≤ 1e-5 nats | `test_full_model_parity.py:36` |
| Argmax | P04, P05, P11–P13 | EXACT | 100 % where top-2 margin > 2e-4 | near-ties excluded explicitly |
| Packer mixture | P14 | TOL | mean atol 1e-4, conc. 1e-3, mix logits 1e-4, rtol 1e-4, Pearson ≥ 0.999 | shipped weights, real structures |
| Teacher-forced per-step log-probs | P07, P08, P09-s, P11-s | TOL | ≤ 1e-4 nats | per-step distribution is deterministic given the prefix |
| Sampled sequences | lanes P07@T0.1, P07@T1.0, P08, P09-s, P11-s | STATISTICAL | per lane, BOTH: recovery TOST δ = 0.01, α = 0.05; excess-JS bootstrap one-sided 95 % upper bound < m_ℓ. Equivalence iff every lane passes (IUT; no multiplicity adjustment) | see "Sampling statistics" |
| Multi-order mean score | P06 | STATISTICAL | TOST on mean NLL, δ = 0.005 nats | n from calibration SE |
| Omitted AAs and X | P08, all sampling lanes | EXACT | 0 omitted-AA and 0 X tokens (X hard-omitted by a `-1e8` bias column, both arms) | `-1e8` bias is a hard omit |

**Lane temperatures (pre-registered, R2-C11).** P07@T = 0.1, P07@T = 1.0, P08@T = 1.0, P09-s@T = 1.0,
P11-s@T = 1.0. Each lane's margin m_ℓ is defined at ITS temperature (T vs 1.05·T). The five values are
written to `preregistered_params.json` (`sampling.lane_temperatures`) and asserted by the validate
script.

**Sampling statistics.** Per lane ℓ each arm draws 2n sequences (n = `n_required`): A₁, A₂ (aminx),
R₁, R₂ (reference). D(X, Y) = mean over designable positions of JS(p̂_X, p̂_Y) (base e). Excess
E_ℓ = D(A₁, R₁) − ½[D(A₁, A₂) + D(R₁, R₂)]: same n on both terms, so the plug-in bias floor
(~(K−1)/n per position) cancels. The upper bound is a 1000-resample sequence bootstrap percentile.
Margin m_ℓ (committed in calibration) = mean E between aminx at T and aminx at 1.05·T on set A at n
(the smallest deviation of concern). `n_required = max(1500, ⌈21.6·σ̂²/δ²⌉)` (σ̂ = per-sequence
recovery SD); calibration then doubles n until 20 aminx-vs-aminx null replicates declare equivalence in
≥ 18/20. **Controls** (20 replicates each, disjoint seeds, per-replicate n = `n_required`): positive =
aminx arm with +β on alanine (β chosen so mean E ≥ 2·m_ℓ); *detected* = the IUT test fails to declare
equivalence; required ≥ 18/20. Negative = aminx vs aminx; a *false positive* = the IUT test fails to
declare equivalence; required ≤ 3/20.

**Pooling, bootstrap and compute budget (R2-C11).** D is computed on per-position token COUNTS pooled
over all of set B (one pooled estimate per lane, not per fixture), and the 1000-resample bootstrap is
**fixture-stratified**: sequences are resampled within each fixture, preserving the fixture composition
of the pooled estimate. `n_required` is the per-arm draw count pooled over set B, allocated across
fixtures proportionally to their designable-position counts (allocation recorded in the params).
Reference-side batching: the non-symmetric branch indexes `decoding_order[:, t_]` per row
(`model_utils.py:262-263`), so P07, P08 and P11-s run the reference BATCHED with one order per row;
only P09-s (symmetric branch, which uses `decoding_order[0]` for the whole batch, `:358`) runs at
batch 1. **Titanix budget (ODQ-13).** Every titanix run is pinned to 16 of titanix's 20 cores
(`taskset -c 0-15`, thread env vars = 16) and one heavy run executes at a time. Each calibrate records,
for its validate, `budget_wall_hours = 2n × lanes × per-draw cost / 16` (per-draw cost measured on
titanix at calibration, including the 20+20 control replicates; the round-3 estimate was ~4–12 h on 24
cores, i.e. ~5–14 h here) and `projected_peak_rss_gib`. The calibrate run fails if `budget_wall_hours >
16` or `projected_peak_rss_gib > 48` (titanix has 123 GiB; the hard cap is 64 GiB, see O6′); the fix is
to reduce set-B fixtures (recorded), never to raise a cap.

**P09-s lane restrictions (the reference's symmetric branch differs; `model_utils.py:357-460`).** The
reference uses `decoding_order[0]` for the whole batch and flattens tie groups at first occurrence
(`:357-368`); sums raw member logits times `symmetry_weights` and adds only the LAST member's bias
(`:443-446`); and applies fixed tokens per member in loop order, so a designable member after a fixed
one inherits its true token (`:452-460`). aminx fuses `sum(log_softmax(logit_transform(l_i, b_i)))`
(`TieGroupProductOfExperts`, `inference/logits.py:396-435`), i.e. every member's bias, and overrides the
whole group with `max(fixed tokens)` whenever any member is fixed (`autoregressive.py:400-406`). The
equivalence lane therefore uses: reference at batch 1 per draw; the host builds the group-flattened
order from that draw's randn and passes it to aminx as `from_tie_groups(groups, order)`; no mixed
fixed/designable groups; `symmetry_weights` all 1.0; zero bias on tied positions except the shared X-omit
column (−1e8 on every member is still a hard omit). Mixed groups, non-unit weights and non-zero tied bias
are T10 DEVIATION clauses outside the gate.

**k-NN-disjoint tie groups (R2-C3; applies to P09-s AND the P09 scoring row).** The reference decodes
tie-group members sequentially in flattened order, so a later member reads an earlier member's already-
updated decoder state, while aminx decodes the whole group in one parallel pass with members mutually
visible (`utils/autoregression.py:181-183,290-294`; reference `model_utils.py:370-382,412-460`). No
aminx `ar_mask` reproduces the reference ordering. **All** intra-group coupling on both sides flows
through the `E_idx` neighbour gather (reference `cat_neighbors_nodes(…, E_idx_t)` `:421-425`; aminx
`take_along_axis(ar_mask, neighbor_indices)` `decoder.py:144`, fixed-scale sum `:340,:507`, undrawn
sentinel embeds as zero `decode/autoregressive.py:238`) — there is no global channel and no two-hop leak
(an earlier-wave neighbour has the later group masked on both sides). The divergence is therefore
unreachable when no two members of a group are k-NN neighbours. So every fixture used by P09-s or by the
P09 scoring row satisfies: **for every same-group pair (i, j) in EVERY tie group of the fixture and for
each v1 `k` (48 and 32), `i ∉ E_idx[j]` and `j ∉ E_idx[i]`**, computed with the aminx featuriser at
backbone noise 0 on the fixture's CA coordinates. The harness checks this and exits 2 on any violation.
- **Non-vacuity floor (pre-registered).** Set B must contain **≥ 2 qualifying multi-member tie groups**
  (≥ 2 designable members each, k-NN-disjoint at both k) across ≥ 2 fixtures — in practice cross-chain
  homo-oligomer ties, whose members are far apart in space. The count is written to
  `preregistered_params.json` (`sampling.p09_qualifying_groups`) at calibration and asserted at
  validation; below the floor the P09-s lane is `incomplete` (residual), never a silent pass.
- Unrestricted intra-group visibility is a **mandatory T10 DEVIATION clause** (T10 step 3) and an AC-25
  known limit: tied sampling is validated only on k-NN-disjoint groups.
- **Tie-specific resolution (R3-C6).** (1) The P09-s lane computes D, E_ℓ, recovery TOST and m_ℓ over
  **tie-group member positions only** (never pooled with untied positions), so the global +β control
  acts on exactly those positions; their count on set B is pre-registered as
  `sampling.p09_tied_positions` and asserted at validation. (2) The P09-s teacher-forced comparison is
  on the **fused per-group distribution**: reference `log_softmax(Σ_i symmetry_weights_i · l_i)`
  (`model_utils.py:443-446`, unit weights, zero tied bias) vs aminx `log_softmax(_fuse_one_group(…))`
  (`inference/logits.py:396-435`; the per-member log-normalisers are a constant, so the two agree
  after renormalisation), max-abs ≤ 1e-4 nats. (3) A **sized fusion control**: the aminx fusion with
  the LAST member's logits scaled by (1 + ε) must move that fused max-abs into [2×, 10×] the 1e-4 bar;
  ε is chosen at calibration (`sampling.p09_fusion_ctrl_eps`) and the control must be detected at
  validation (`p09_fusion_ctrl_detected`). Shared-draw failures are caught by T11 invariant 6.

**Positive-control sizing.** A sized control (perturbed weight) must move its metric into [2×, 10×] the
bar; calibration picks the magnitude, records it in `preregistered_params.json`, commits it before any
validation run.

### Bathos staging

One confirmation campaign per layer (`aminx-bv-layer-{a,b,c}`); Phase 0 in exploration campaign
`aminx-bv-phase0`. The claim is registered before any validation run (Signal 12). Stages: `exploration`
→ `calibration` (set A) → `validation` (held-out set B); layer-(a) calibrate and validate runs execute
on titanix via `bth run` (O6′), never `bth submit`. **Prerequisite check (replaces
`requires_pass_stem`, which only `bth submit` enforces, `cli_cyclopts.py:746-757`).** Each calibrate
writes its own section of `preregistered_params.json` plus `params_section_sha256` (sha256 of the
section's canonical JSON, `sort_keys`, `(',', ':')` separators) and its `git_hash` into its result.
Each validate's `prereq_check()` recomputes that hash from the params it reads and runs `bth sql` against
the catalog of the machine it runs on: the count of rows with `command LIKE '%<stem>_calibrate.py%'`,
`outcome = 'pass'` and that `params_section_sha256` must be ≥ 1, and at least one such row's `git_hash`
must satisfy `git merge-base --is-ancestor <git_hash> HEAD`. Otherwise `prereq_ok = false` → outcome
`incomplete`. Results go to `$BTH_RESULTS_PATH` AND `--out` under `outputs/browser_validation/`,
except in differential phases (below). Anti-HARKing: every validation script refuses to start unless
`outputs/browser_validation/<layer>/preregistered_params.json` is tracked and clean
(`git ls-files --error-unmatch` and `git diff --quiet HEAD --`).

**Differential pre-flight rules (runner.py:194-270).** `bth run` re-executes the full argv twice with
ONLY `BTH_DIFFERENTIAL_KNOB`, `BTH_DIFFERENTIAL_VALUE`, `BTH_DIFFERENTIAL_PHASE` (plus
`BTH_RESULTS_PATH`, `BTH_OUTPUT_DIR`) set; `effect = |on − off|` of `metric`, and `differs = effect ≥
min_effect`. Therefore:
- scripts read the knob only from `BTH_DIFFERENTIAL_VALUE` and exit 2 unless `BTH_DIFFERENTIAL_KNOB`
  equals the sidecar knob;
- `metric` is a MAIN-arm quantity the knob moves;
- `min_effect` > 0, equal to the value committed in `preregistered_params.json`; `--dry-run` fails on
  0.0 or on a mismatch;
- when `BTH_DIFFERENTIAL_PHASE` is set, a script computes only the differential metric on a
  pre-registered reduced subset (size in `preregistered_params.json`, shown in calibration to exceed
  `min_effect`) and writes ONLY to `$BTH_RESULTS_PATH`, never `--out`;
- calibration sidecars carry NO `[differential]`: sensitivity is proven by the validate runs.

**Outcome labels.** bathos treats every non-residual label except `marginal`/`error`/`unknown` as
pass-direction (`sidecar.py:549-563`) and picks the first matching branch in declaration order
(`:679-695`). So `ctrl_blind`, `incomplete`, `underpowered`, `fail` are `is_residual = true` everywhere;
`partial_headroom` (and the parity `partial`) is deliberately non-residual. `lint_sidecars.py` (T7)
fails if any other non-pass label is non-residual.

**Layer-(a) exact validation sidecar** (`scripts/browser_validation/layer_a_exact_validate.bth.toml`):

```toml
[metadata]
title = "Layer (a) exact tier: aminx JAX vs LigandMPNN@26ec57ac on held-out fixture set B"
created = "<YYMMDD>"
schema_version = "0.3"

[experiment]
hypothesis = "On fixture set B (CPU f32, precision highest, noise 0, reference-formula order fed to both sides), every deterministic comparison in the layer-(a) bars table is within its bar with the SHIPPED .eqx.zst weights, every path advanced from calibration, and every in-run control exceeds its bar."
stage_name = "validation"
claim_discriminates = ["H_exact_within_bar", "H_null_misspec"]
claim_isolates = ["C_weight_source"]

[reproduction]
reproduces_paper = "dauparas/LigandMPNN@26ec57ac976ade5379920dbd43c7f97a91cf82de (reference code; bathos-literature-parity Mode A)"
# no requires_pass_stem: only `bth submit` enforces it; prereq_check() (Bathos staging) replaces it

[controls]
positive_outcome = ["pass"]
negative_outcome = ["incomplete", "ctrl_blind", "fail"]

[differential]
knob = "AMINX_BV_PERTURB"      # read via BTH_DIFFERENTIAL_VALUE; 1 = calibrated W_out-bias perturbation on the main arm
off = "0"
on = "1"
expect = "differs"
metric = "sentinel_ratio_to_bar"   # P05 log-prob max-abs / bar on the reduced subset
min_effect = 1.0                   # PINNED at 1.0 (R2-C6); calibration sets exact.min_effect = 1.0 and
                                   # asserts half the measured on-off effect on the reduced subset >= 1.0,
                                   # recording exact.measured_half_effect. No placeholder; --dry-run
                                   # requires sidecar min_effect == exact.min_effect == 1.0.

[outcomes.incomplete]
condition = "n_skipped > 0 OR NOT prereq_ok OR reference_commit <> '26ec57ac976ade5379920dbd43c7f97a91cf82de' OR NOT git_clean"
decision = "A prerequisite was missing, the reference drifted, or the run tree was dirty; the run proves nothing."
is_residual = true

[outcomes.ctrl_blind]
condition = "controls_detected < controls_total"
decision = "Instrument insensitive at the claimed resolution; no parity statement is permitted."
is_residual = true

[outcomes.pass]
condition = "n_over_bar = 0 AND n_not_advanced = 0"
decision = "All paths within bar; eligible for the advance table (T11)."
is_residual = false

[outcomes.partial_headroom]
condition = "n_over_bar = 0 AND n_not_advanced > 0"
decision = "Advanced paths within bar; not_advanced paths are listed and do not advance. Maps to rung R2 (T11)."
is_residual = false

[outcomes.fail]
condition = "n_over_bar > 0"
decision = "worst_path exceeds its bar with a sensitive instrument: finding + DEVIATION."
is_residual = true

[result_schema]
n_comparisons = "int"
n_over_bar = "int"
n_not_advanced = "int"
n_near_tie_excluded = "int"
n_skipped = "int"
prereq_ok = "bool"
controls_total = "int"
controls_detected = "int"
worst_ratio_to_bar = "float"
sentinel_ratio_to_bar = "float"
worst_path = "str"
reference_commit = "str"
weight_source = "str"
fixture_set = "str"
fixture_manifest_sha256 = "str"
git_hash = "str"
git_clean = "bool"
prereg_sha256 = "str"

[dependencies]
aminx_version = { type = "git_sha", required = true }
jax_version = { type = "string", required = true }
torch_version = { type = "string", required = true }
```

Per-row detail goes in the result JSON under `rows` (`{path, metric, value, bar, ratio, status ∈
{validated, not_advanced}, weight_source, fixture}`). `git_hash` and `git_clean` come from
`layer_a_common.provenance()` (T7) in the run's own checkout; `prereg_sha256` is the sha256 of the params
file actually read. The calibration twin (`layer_a_exact_calibrate.bth.toml`) has `stage_name =
"calibration"`, `novel = true` instead of `[reproduction]`, NO `[differential]` and NO
`claim_discriminates`. It writes the `exact` section of `preregistered_params.json` (tightened bars,
magnitudes, headroom status per path, `min_effect`, subset size, budget line, `git_hash`) and records
`params_section_sha256`.

**Calibration-twin outcomes (R2-C10).** `prereq_check()` accepts ONLY `outcome = 'pass'`, so a calibration that legitimately finds `not_advanced` paths must still be
labelled `pass`. Both calibration twins declare, in this order:
- `incomplete` (residual): `n_skipped > 0 OR NOT git_clean`;
- `ctrl_unsized` (residual): `controls_sized < controls_total`;
- `pass` (non-residual): `params_written AND controls_sized = controls_total`.

Headroom is a RESULT FIELD (`n_not_advanced`), never an outcome. Both calibrate `result_schema`s add
`controls_sized = "int"`, `controls_total = "int"`, `params_written = "bool"`, `n_not_advanced = "int"`,
`git_clean = "bool"`, `git_hash = "str"`, `params_section_sha256 = "str"`, `budget_wall_hours = "float"`,
`projected_peak_rss_gib = "float"`. `lint_sidecars.py` whitelists `ctrl_unsized` as residual.

**Layer-(a) sampling validation sidecar** (`layer_a_sampling_validate.bth.toml`), differences only:

```toml
[experiment]
hypothesis = "On fixture set B, for lanes P07@T0.1, P07@T1.0, P08, P09-s, P11-s: teacher-forced per-step log-probs match within 1e-4 nats, and every lane is declared equivalent (recovery TOST delta 0.01 AND excess-JS 95% upper bound < committed m_l; IUT, alpha 0.05 per lane) at n_required, with zero omitted-AA and zero X tokens in both arms; positive control detected >= 18/20, negative control false positives <= 3/20."
stage_name = "validation"
claim_discriminates = ["H_sampling_equiv", "H_null_misspec"]

[reproduction]
reproduces_paper = "dauparas/LigandMPNN@26ec57ac976ade5379920dbd43c7f97a91cf82de (reference code; Mode A)"
# no requires_pass_stem; prereq_check() against layer_a_sampling_calibrate replaces it

[differential]
knob = "AMINX_BV_SAMPLING_BIAS"   # 1 = calibrated +beta on alanine, MAIN aminx arm
off = "0"
on = "1"
expect = "differs"
metric = "main_js_vs_ref"         # pooled-composition JS, main aminx arm vs reference arm, reduced subset
min_effect = 0.0                  # PLACEHOLDER: set to sampling.min_effect (> 0) in the params commit, BEFORE any validate smoke (R3-C9); --dry-run fails on 0.0

[outcomes.incomplete]
condition = "n_skipped > 0 OR NOT prereq_ok OR NOT git_clean OR p09_tied_positions <> p09_tied_positions_prereg"
is_residual = true
[outcomes.ctrl_blind]
condition = "posctl_detected < 18 OR negctl_fp > 3 OR NOT p09_fusion_ctrl_detected"
is_residual = true
[outcomes.underpowered]
condition = "n_per_arm < 2 * n_required"
is_residual = true
[outcomes.pass]
condition = "tf_max_abs <= tf_bar AND n_lanes_equiv = n_lanes AND omitted_aa_count = 0 AND x_token_count_aminx = 0 AND x_token_count_reference = 0"
is_residual = false
[outcomes.fail]
condition = "tf_max_abs > tf_bar OR n_lanes_equiv < n_lanes OR omitted_aa_count > 0 OR x_token_count_aminx > 0 OR x_token_count_reference > 0"
is_residual = true

[result_schema]
tf_max_abs = "float"
tf_bar = "float"
n_lanes = "int"
n_lanes_equiv = "int"
n_per_arm = "int"
n_required = "int"
sigma_hat = "float"
excess_js_ub_max_ratio = "float"   # max over lanes of UB(E_l)/m_l
main_js_vs_ref = "float"
posctl_detected = "int"
negctl_fp = "int"
omitted_aa_count = "int"
x_token_count_aminx = "int"
x_token_count_reference = "int"
p09_tied_positions = "int"
p09_tied_positions_prereg = "int"
p09_fused_tf_max_abs = "float"      # included in tf_max_abs (R3-C6)
p09_fusion_ctrl_detected = "bool"
n_skipped = "int"
prereq_ok = "bool"
reference_commit = "str"
git_hash = "str"
git_clean = "bool"
prereg_sha256 = "str"
```

The `min_effect = 0.0` placeholder exists only because the value is unknown until calibration; the
orchestrator sets it in the same commit as the params file, before any validate `--dry-run`/`--smoke`,
and `lint_sidecars.py` (without `--allow-placeholder`, which only the fixer lint gate uses) fails any
validation sidecar with `min_effect <= 0`. Per-lane results (`{lane, tost_pass, excess_js, excess_js_ub, margin,
equiv}`) go under `lanes`.

**Literature-parity run** (`scripts/browser_validation/parity_validate_mpnn.py` + `.bth.toml`, config
`parity.bth.toml`). The result JSON contains `{"metadata": {"parity_run_type": "literature_parity"}}`,
which bathos extracts into the `parity_run_type` column (`runner.py:770-793`); `attest_parity` refuses
runs without it or with an outcome other than `pass`/`partial` (`claim.py:975-1005`).

```toml
# parity.bth.toml  (bathos parse_parity_toml)
[parity]
paper_pdf      = "docs/references/proteinmpnn_2022.pdf"        # acquired in T4
impl_paths     = ["src/aminx/model/features.py", "src/aminx/model/encoder.py", "src/aminx/model/decoder.py",
                  "src/aminx/model/mpnn.py", "src/aminx/model/ligand_mpnn.py", "src/aminx/model/ligand_features.py",
                  "src/aminx/model/packer.py", "src/aminx/inference/score_unconditional.py",
                  "src/aminx/inference/score_conditional.py", "src/aminx/inference/sample_autoregressive.py",
                  "src/aminx/inference/decode/autoregressive.py", "src/aminx/inference/logits.py",
                  "src/aminx/scoring/score.py", "src/aminx/sampling/sample.py", "src/aminx/utils/decoding_order.py",
                  "src/aminx/io/weights.py"]
reference_code = "$REFERENCE_PATH (dauparas/LigandMPNN@26ec57ac); secondary: .cache/reference/ProteinMPNN@<pin from T4>"
citation_note  = "Mode A. ProteinMPNN: Dauparas et al., Science 2022, doi:10.1126/science.add2187. LigandMPNN: paper + run.py/model_utils.py/sc_utils.py at 26ec57ac."
recon_lenses   = ["math", "algo", "protocol"]
attack_lenses  = ["stats", "hyper", "struct"]
hypotheses     = [
  "decoding-order semantics (fixed-first rule, AR visibility mask) match the reference",
  "temperature is applied to (logits + bias), and X is hard-omitted at sampling as in the reference",
  "k-NN neighbour selection matches for L >= k and at exact distance ties",
  "tied-group fusion, bias and fixed-token semantics match the reference symmetric branch",
  "membrane label channels are live and encoded as the reference encodes them",
  "ligand context featurisation (atom context num, cutoff, side-chain context) matches",
  "shipped .eqx.zst weights equal the reference .pt tensors",
]
equivalence_bound = 1e-4
N = 3
M = 3
```

```toml
# parity_validate_mpnn.bth.toml
[experiment]
hypothesis = "aminx's in-scope MPNN paths implement dauparas/LigandMPNN@26ec57ac (Mode A) at cap-lattice grade PARITY or PARTIAL, with both layer-(a) validate prerequisites pass (R1) or exact partial_headroom (R2), and orchestrator-re-derived invariant tests passing with zero skips."
stage_name = "validation"
claim_discriminates = ["H_lit_parity", "H_null_misspec"]
claim_isolates = ["C_reference_parity"]

[reproduction]
reproduces_paper = "dauparas/LigandMPNN@26ec57ac976ade5379920dbd43c7f97a91cf82de; doi:10.1126/science.add2187"
# no requires_pass_stem: it matches outcome='pass' only (prereg.py:302-336) and is enforced only by
# `bth submit`; the prerequisite mapping lives inside the script (O1).

[controls]
positive_outcome = ["pass", "partial"]
negative_outcome = ["fail"]

[outcomes.pass]
condition = "parity_grade = 'PARITY' AND prereq_status = 'ok' AND ambiguity_load = 'none' AND n_invariant_skipped = 0 AND n_invariant_tests >= 8"
is_residual = false
[outcomes.partial]
condition = "parity_grade = 'PARTIAL' AND prereq_status IN ('ok', 'headroom') AND n_invariant_skipped = 0 AND n_invariant_tests >= 8"
is_residual = false
[outcomes.fail]
condition = "parity_grade = 'FAIL' OR prereq_status NOT IN ('ok', 'headroom') OR n_invariant_skipped > 0 OR n_invariant_tests < 8"
is_residual = true

[result_schema]
parity_grade = "str"
prereq_status = "str"        # ok | headroom | failed | missing
exact_validate_outcome = "str"
sampling_validate_outcome = "str"
clause_parity_pct = "float"
invariant_pass = "bool"
n_invariant_tests = "int"
n_invariant_skipped = "int"
adversarial_survived = "bool"
reproduction_rung = "str"
ambiguity_load = "str"       # enum none | non_load_bearing | load_bearing (parity.py:31), rule below
n_ambiguous = "int"
```

**Grade inputs (R2-C8 / O8).** `compute_grade` takes `clause_parity_pct` and `adversarial_survived` as
caller-supplied scalars (`parity.py:86-97`; missing keys default to FAIL, `:137-141`), and imposes a FAIL
ceiling when `clause_parity_pct < 0.5` or `adversarial_survived` is false (`parity.py:56,86-87,95-97`).
This spec therefore defines both exactly:

- `clause_parity_pct = |{c : c.verdict = MATCH AND c.core}| / |{c : c.core}|` — **core-restricted in BOTH
  numerator and denominator** (a numerator over all clauses can exceed 1.0). `AMBIGUOUS` clauses are
  excluded from both terms and counted into `n_ambiguous`.
- **`core` is defined, not chosen (R3-C11).** `core = true` iff the clause's `paths` intersect the
  in-scope layer-(a) P-IDs, EXCEPT the six designed DEVIATIONs, which are `core = false` with a written
  reconciliation note: (1) native X-samplable default; (2) default decoding order `fixed_n_to_c`;
  (3) tied fixed-override (`max(fixed tokens)` vs per-member inheritance); (4) last-member bias;
  (5) tied fusion with non-unit weights; (6) intra-tie-group visibility. Every other clause —
  including any unexpected DEVIATION or MISSING — is `core = true`. `litparity_schema.py` enforces
  this mechanically (T10 step 5).
- **`ambiguity_load` is an enum computed by rule (R3-C5)** — `compute_grade` caps at PARTIAL only on the
  literal `'load_bearing'` (`parity.py:31,111-114`): `'load_bearing'` if any AMBIGUOUS clause is
  `core = true` or lists an in-scope layer-(a) P-ID; `'non_load_bearing'` if ≥ 1 AMBIGUOUS clause
  exists and none is load-bearing; else `'none'`. The parity sidecar's `pass` requires `'none'`;
  `litparity_schema.py` asserts the enum and its consistency with `clauses.json`.
- `adversarial_survived = not any(d.severity == 'core' for d in adjudication.defects_confirmed)` — a
  confirmed non-core defect does not flip it.
- **Anti-HARKing (O8):** the `core` flag and the reconciliation note for EVERY mandatory clause are
  committed in `litparity/clauses.json` BEFORE any parity run, and their sha256 is recorded in
  `preregistered_params.json` (`litparity.clauses_sha256`). `parity_validate_mpnn.py` recomputes that
  sha256 and REFUSES to grade (exit 7) if it differs, so core status cannot be assigned after seeing
  adjudication results.

The global grade is EXPECTED to be PARTIAL, but that expectation now rests on the **reproduction rung**
(R2 whenever exact-validate is `partial_headroom`) and on the **ambiguity ceiling**, NOT on
`clause_parity_pct < 1.0` — the designed DEVIATIONs are `core = false` and do not depress it, and the
verdict doc must show the numerator, denominator and margin above 0.5. The parity sidecar's `pass` is
reachable only at rung R1 with zero ambiguity load; Phase 2 advances PER PATH through T11's advance
table, not through the global grade.

**Claim** (`.bth/claims/aminx-bv-layer-a.claim.toml`; `bth claim scaffold`, then register before any
validation run). `discriminability` and `union_gate` live UNDER `[claim]` (`claim.py:162-164`; bathos
renders `[[claim.union_gate.clauses]]`); a top-level `[union_gate]` is silently ignored. **The
`parity_run_id` line's spelling is load-bearing (R2-C4):** `attest_parity` rebinds by matching the exact
literal `parity_run_id = ""` (`claim.py:40`) and raises when it matches nothing (`:1030-1035`), so the
line carries single spaces around `=` and no alignment padding; the T4 gate asserts it with `grep -qx`
and that it occurs exactly once.

```toml
[claim]
headline       = "Every in-scope deterministic aminx MPNN path matches LigandMPNN@26ec57ac within its pre-registered bar on held-out structures, and every in-scope sampling lane is statistically equivalent at its pre-registered margin."
kill_condition = "On fixture set B with all controls detected: any in-scope path exceeds its bar (exact-validate outcome fail), OR the sampling-validate outcome is fail, OR the literature-parity run has invariant_pass = false or a confirmed core-severity defect."
# R2-C8: keyed on invariant_pass / confirmed core defect, NOT on the global grade string -- a PARTIAL
# grade is the documented expectation (R19) and must not fire the kill condition. This wording is
# committed BEFORE `bth claim register` (T4 step 4); it is never edited after registration.
regime         = "CPU f32, matmul precision highest, noise 0 (exact tier); T in {0.1, 1.0}; fixture set B; checkpoints in reference_pins.json"
kill_condition_satisfiable_by_null = true

[[hypotheses]]
id = "H_exact_within_bar"
label = "deterministic aminx paths reproduce the reference computation within bar"
predicted_signature = "exact-validate outcome pass; controls detected; differential passed"
[[hypotheses]]
id = "H_sampling_equiv"
label = "aminx sampled distributions are equivalent to the reference per lane"
predicted_signature = "sampling-validate outcome pass; IUT equivalence in every lane; controls behave"
[[hypotheses]]
id = "H_lit_parity"
label = "the literature-parity protocol grades aminx PARITY or PARTIAL"
predicted_signature = "parity run outcome pass or partial with zero invariant skips"
[[hypotheses]]
id = "H_null_misspec"
label = "the comparison is misspecified or the instrument cannot see the claimed resolution"
predicted_signature = "controls undetected, skipped prerequisites, or agreement that survives a perturbation"

[[assumptions]]
id = "A_same_inputs"
label = "both sides consume byte-identical input arrays and decoding orders"
halt_if = "input hashes recorded per side differ"
status = "untested"

[[confounds]]
id = "C_reference_parity"
label = "the reference comparison is to the published code at a pinned commit, graded by the literature-parity protocol"
control = "literature-parity run"
isolating_run = ""
status = "uncontrolled"
[confounds.reference_parity]
reference_paper   = "dauparas/LigandMPNN@26ec57ac (Mode A)"
reference_metric  = "conditional_logprob_max_abs"
reference_value   = 0.0
equivalence_bound = 1e-4
parity_run_id = ""  # NEVER hand-edit: bound by bth campaign attest-parity (T11)

[[confounds]]
id = "C_metric_soundness"
label = "the comparison metrics recover planted differences"
control = "synthetic-truth gate"
isolating_run = ""
status = "uncontrolled"
[confounds.synthetic_recovery]
gate_name = "mpnn_metric_synthetic_truth"
guards = ["src/aminx/parity/compare.py", "tests/parity/test_compare_metrics.py"]

[[confounds]]
id = "C_weight_source"
label = "results are for the SHIPPED .eqx.zst weights, not weights converted at test time"
control = "weight_source = 'eqx' in every validation row"
isolating_run = ""
status = "uncontrolled"

[[claim.union_gate.clauses]]
id = "C_exact_paths"
description = "exact-validate run covers the deterministic paths"
hypothesis_ids = ["H_exact_within_bar", "H_null_misspec"]
[[claim.union_gate.clauses]]
id = "C_sampling_equiv"
description = "sampling-validate run covers the sampling lanes"
hypothesis_ids = ["H_sampling_equiv", "H_null_misspec"]
[[claim.union_gate.clauses]]
id = "C_lit_parity"
description = "literature-parity run covers the protocol grade"
hypothesis_ids = ["H_lit_parity", "H_null_misspec"]
[[claim.union_gate.clauses]]
id = "C_exact_sensitivity"
description = "the sized weight perturbation is detected at the bar (exact-validate differential passed)"
hypothesis_ids = ["H_exact_within_bar"]
positive_control = true
[[claim.union_gate.clauses]]
id = "C_sampling_sensitivity"
description = "the calibrated +beta bias moves main-arm JS by >= min_effect (sampling-validate differential passed)"
hypothesis_ids = ["H_sampling_equiv"]
positive_control = true
```

Union-gate coverage ignores run OUTCOME (`claim.py:869-898`): a failed run with the right
`claim_discriminates` covers its clause. The verdict is therefore gated by AC-8/AC-11/AC-13 (outcome
labels), not by `covered`. Positive-control clauses need a covering run with `differential_status =
'passed'` AND a `dependency_lock_sha256` matching the current `uv.lock` (`claim.py:813-819`); calibration
runs carry no `claim_discriminates` so they can never cover them.

**Phase 2/3 skeletons** (fixer steps written when the phase is specced). No `requires_parity_stem`: the
Phase-2 harness calls `scripts/browser_validation/advance_check.py <P-ID>` and refuses a path that does
not advance. `layer_b_export_parity_{calibrate,validate}.bth.toml`: `novel = true`; schema
`{n_int_mismatch:int, float_max_abs:float, float_bar:float, n_exempt_flips:int, exempt_flip_cap:int,
n_rng_primitives:int, safety_blockers:int, trace_ok:bool, target:str, route:str, bucket:int,
checkpoint:str, artifact_sha256:str}`. `layer_c_browser_parity_*.bth.toml` adds `{browser:str,
browser_version:str, runtime:str, runtime_version:str, ep:str, threads:int, cross_origin_isolated:bool}`.
`browser_benchmark.bth.toml` is a `[benchmark]` sidecar whose baseline is an INTERLEAVED arm: native JAX
CPU run in the same harness session (local subprocess or local server) alternating with the browser
routes. `regression_threshold` (pre-registered at Phase-3 calibration) references only that arm. If
the interleaved arm is infeasible at Phase-3 calibration, `baseline_ref` becomes provenance-only with no
threshold, and that decision is recorded. Schema `{route, browser, bucket, phase, p50_ms, p90_ms,
ci95_lo_ms, ci95_hi_ms, n_iter, n_warmup, repeat_idx, peak_mem_bytes, cross_origin_isolated,
native_arm_p50_ms}`.

### Sampling contract for export (ODQ-5; O2) — derived from the kernel, not assumed

`inference.sample_autoregressive.kernel(model, prng_key, bundle, …)` does `k_enc, k_dec =
jax.random.split(prng_key)` (`sample_autoregressive.py:106`) and passes `k_dec` unchanged as `key` to
`AutoregressiveDecode.__call__` (`decode/factory.py:119` returns the module itself). `step_fn` closes over
that same `key` for every wave, with no per-wave split or fold. Hence **wave_key_w = k_dec for every
wave w**. Per group slot it draws
`jax.random.categorical(fold_in(k_dec, gid), F / T)` with `gid = tie_group_map[0, group_positions[w, s,
0]]` (`autoregressive.py:287-288,394-396`). `categorical(k, x)` is `argmax(gumbel(k, x.shape, x.dtype) +
x)` (measured by the orchestrator on jax 0.11.1, 2026-09-24: 0/2000 mismatches over random 21-logit vectors
at scales 0.1/1/5 with `fold_in` keys; AC-16 remains the arbiter), and `gumbel` (default mode) is `-log(-log(u))`, `u ~ U[finfo(f32).tiny, 1)`. So the exact contract is:

- noise: `g[gid] = jax.random.gumbel(jax.random.fold_in(k_dec, gid), (21,), jnp.float32)`, host array
  of shape `(L, 21)` f32 indexed by gid (`gid = p` for untied `from_tie_groups(arange(L), order)`). Each
  group is drawn once, at its first occurrence. A fixed group still draws and discards, so per-group keys
  keep every other draw unchanged.
- tokens: `tok = argmax(F(logit_transform(l, b)) / T + g[gid])`, with exactly the kernel's op order.
  `logit_transform` is the stage set's `ar_logit_transform` (vmapped per position) or `logit_transform`,
  applied with `bias = cond_bias`. `F = _fuse_one_group(·, mask_group, stage_set.tie_group_fuse)`. Then
  `where(is_group_fixed, max fixed token, tok)`. Wrappers IMPORT these functions; they do not
  reimplement them.
- X-omit: a `-1e8` bias column at index 20, supplied identically to the native kernel (through
  `bundle.conditioning.bias`) and to the wrapper, never as `-inf` inside the wrapper only. The native
  no-X-omit default (X samplable) is compared statistically (X frequency, layer a DEVIATION clause),
  not bitwise.
- The categorical/gumbel identity is checked against the installed jax (version recorded) by AC-16's
  bitwise test, which is the arbiter.

**Per-path runtime-vs-baked inputs** (every other quantity is baked; ligand side-chain-context on/off and
checkpoint are separate artifacts):

| Path | Runtime inputs (all bucket-padded) |
|---|---|
| all | coords, mask, residue_index, chain_index |
| P05, P06 | + sequence, decoding order (→ ar_mask), chain/design mask |
| P07, P08 | + decoding order, wave-schedule arrays (host `from_tie_groups`), noise `g` (L×21 f32), temperature (f32 scalar), bias (L×21, incl. omit_AA and X-omit column), fixed_mask, fixed_tokens |
| P09 | P07 inputs + tie_group_map |
| P11 | P04–P07 inputs + ligand coords, types, mask |
| P13 | P04–P07 inputs + per-residue membrane labels / global label |

### Phase 2 plan (layer b) — goals

1. **Wrappers with RNG outside the graph.** Each v1 path gets a pure callable with the table's runtime
   inputs. Wrappers bypass the coordinate-noise branch (`features.py:200` always calls
   `apply_noise_to_coordinates`, which runs `split`/`normal` inside `lax.cond`) and the in-graph
   `categorical` (the xtrax threefry rule fires only for input-derived keys, `safety.py:474-478`). An AC
   walks the exported jaxpr and ONNX graph for RNG primitives (C12).
2. **Per-route gates (ODQ-1).** ORT route: jax2onnx conversion, ORT-CPU executed parity within bars,
   and an attached op-set / EP-assignment report. IREE route: `check_export_safety` clean on NATIVE and
   WASM32 with `trace_ok` asserted (the gate swallows trace failures, `safety.py:443-451`), plus native
   IREE executed parity. IREE results gate only the IREE route. wasm32 is compiled with sha256
   recorded, CODEGEN_ONLY.
3. **Teacher-forced token check.** Margin = top-2 gap of `F(logit_transform(l, b))/T + g`. A flip is
   exempt only where that gap < 2·bar/T. Validation runs the exported conditional wrapper on the JAX
   token prefix with the same order and noise and asserts per-step argmax equality except for exempt
   flips. Free-running divergence is reported only as a diagnostic. There is a committed per-path cap on
   exempt flips and a planted-flip control (one noise value moved to flip one token) that must be
   detected.
4. **Padding/bucketing (P27)**: refuse real L < `k_neighbors(checkpoint)` (`get_topology_for_checkpoint`:
   48 for v_48, 32 for ligandmpnn_v_32) and L > max bucket; reconcile aminx (64…512) with xtrax
   (64…2048) ladders; xtrax R0–R3 ladder (`rings.py`) attached.
5. Bars pre-registered at Phase-2 calibration, capped at the layer-(a) bar; integer outputs exact.

### Phase 3 plan (layer c) — goals

1. Browser runner per route (ODQ-1): run the Phase-2 artifact on set B inputs and compare against the
   SAME artifact's layer-(b) outputs.
2. JS host logic: PDB parser (P02), bucketing/padding (P27), decoding-order generator (P19) and the noise
   generator. For identical supplied noise, layer c vs b is exact. The generators' DISTRIBUTIONS are
   tested separately (AC-22), because identical-noise parity cannot see a wrong generator.
3. Benchmark harness (Playwright), with headless Chromium/Firefox/WebKit plus ≥ 1 headful run each. It
   asserts `crossOriginIsolated === true` (COOP `same-origin`, COEP `require-corp`) before trusting
   sub-ms timers or `measureUserAgentSpecificMemory`. It separates phases (fetch/load,
   session-create/compile, first inference, steady state ≥ 30 iterations after ≥ 5 warm-ups), sweeps
   buckets, and interleaves routes AND the native JAX arm in one session. The whole matrix repeats ≥ 3
   times, reporting p50/p90 with bootstrap 95 % CIs. Given ~5× run-to-run variance, there is no
   cross-session comparison. WebGPU `timestamp-query` numbers are reported separately. xtrax
   `profiling.claims` (`ClaimClass`, `assert_claim_supported`) gates claims.
4. No text claims browser compatibility for a route × browser pair without an evidence JSON (xtrax
   `260910_webgpu-export-route.md:94-97`).

### Phase 4 plan (advisor deliverable) — goals

A static page plus artifacts (ODQ-9): an artifact manifest (sha256, P-ID, bucket, target, route, source
checkpoint sha); a report card (path × layer → PARITY/PARTIAL/FAIL/not-validated, each cell linking a
bathos run ID); per-path bars; known limits; reproduction commands. Every number carries a
figure-manifest input pin with sha256.

### Recon corrections (verified on the a5ced9c7 snapshot and bathos source)

1. `utils/decoding_order.py:75` (tied branch) also calls `jax.random.permutation`; PR #157 fixed only
   the sort tie order. Both `:67` and `:75` carry the permutation hazard.
2. The kernel stack takes its order from a host `WaveScheduleBundle` (`bundle_builder.py:208-262`)
   defaulting to `fixed_n_to_c`. In-graph `random_decoding_order` is the default only on the legacy
   stack (`sampling/sample.py:22`, `scoring/score.py:17`, STE, `host/runner.py:377`). The
   `fixed_n_to_c` default vs the reference's random order is a reconciliation clause.
3. More host-side permutation sites: `schedule_selector.py:338,367`.
4. Existing layer-(a) coverage is weak:
   - fixtures have L = 20/12 (below k, so k-NN is never exercised);
   - AR/tied tests set `bias[..., 0] = 100.0` (`test_full_model_parity.py:104`), so token agreement is
     non-discriminating;
   - membrane uses constant class 0;
   - the packer test uses `pt_convert` weights only.
5. `scripts/measure_conditional_nll_parity.py:56` passes `"bundled"`, which maps to `pt_convert`. The
   "derived" 1e-4/1e-5 bars were measured on test-time-converted weights.
6. `tests/parity/reference_utils.py` turns missing prerequisites into `pytest.skip`, so an all-skipped
   heavy run exits 0.
7. `check_export_safety` ignores `decisions`/`axis_boundaries` (`safety.py:540`) and returns `[]` when
   tracing fails (`:443-451`).
8. aminx ruff config: `force-exclude = true` excluding `tests`/`scripts` (a bare gate prints "No Python
   files found", exit 0); `fix = true` rewrites source.
9. aminx `.bth.toml` `root = "/home/marielle/projects/aminx"` (the stale checkout). `bth submit`'s
   default push and `--then-pull` shell out to `myxcel push-project` / `myxcel pull-project`
   (`cluster.py:78,:151`), which do not exist. This spec never uses `bth submit` (O6′; R20).
10. The installed `xtrax[io,export]==0.4.0a10` may predate the `random-permutation` /
    `unbatched-threefry-key` rules; T1 checks.
11. Reference at 26ec57ac: omit_AA = `-1e8` folded into bias (`run.py:405-407`); sampling renormalises
    over the first 20 tokens (X hard-omit, `model_utils.py:320-322`); temperature divides
    `(logits + bias)` (`:318`). aminx samples over 21 (`autoregressive.py:396`). A `-1e8` bias on index
    20 reproduces the hard omit exactly (f32 `exp` underflows to 0).
12. The kernel's per-wave key is the SAME `k_dec` for every wave (see Sampling contract).
13. The reference symmetric branch differs from aminx's tie handling (see P09-s restrictions).
14. bathos CLI surface (generated from `mcp.py` signatures):
    - `bth run --campaign-id --output-paths`;
    - `claim register <path> --campaign-id`;
    - `campaign conclude --campaign-id --outcome-label [--negative-check]`;
    - `campaign attest-parity --campaign-id --parity-run-id`;
    - `gate stamp <name> --result`;
    - `gate status <name> --guards … --guards …`;
    - `sync [--pull] [--remote-name]` (default direction is push; the remote is auto-selected only
      when exactly one is configured, `mcp.py:1543-1576`, so this spec always names `titanix`);
    - `sql` prints `{"rows": [[…]], "count": n}` with rows as lists (`mcp.py:441-442`).

    `claim validate` and `gate status` return JSON and exit 0 even on `ok: false` / RED
    (`cli_render.py:35-49` exits 1 only on an `error` key), so gates assert with `jq -e`. `claim
    validate` checks the parity graded path only with `--catalog-dir` and a compacted run
    (`claim.py:331-368`).
15. Minor path fixes. The xtrax jax2onnx study lives at `.praxia/docs/research/260914_…`. The
    literature-parity templates live in the bathos repo (`agent_assets/skills/bathos-literature-parity/
    templates/`). Its `05_verdict.md` hand-fills `parity_run_id`, which contradicts `claim.py`; this spec
    uses `attest-parity`.

## Acceptance Criteria

**Phase 0**

- **AC-1** `outputs/browser_validation/phase0/export_safety_census.json` has one row per callable:
  - P04 kernel;
  - P05 kernel;
  - P06 `score_sequence` with pinned order;
  - P07 kernel with a host wave;
  - legacy `aminx.sampling.sample`;
  - P11 conditional kernel;
  - P14 packer.

  Each is built at L = 128 with shipped weights, and each row has `trace_ok` (independent
  `make_jaxpr`) and its blockers. Both controls are detected: bare `random_decoding_order` gives
  `random-permutation`, and bare `lax.top_k` gives `unlegalizable-op`. The file records the rule
  presence and `xtrax.__file__`.
- **AC-2** `outputs/browser_validation/phase0/jax2onnx_spike.json` records:
  - versions;
  - P05 conversion success or the verbatim error;
  - ORT-CPU vs JAX max-abs and EXACT neighbour indices;
  - the tie-lattice `top_k` indices-identical flag;
  - a detected perturbed-weight control;
  - the artifacts `p05_L128.onnx`, `p05_L128_perturbed.onnx`, `topk_lattice.onnx`, `p05_inputs.npz`,
    `p05_jax_outputs.npz`, `topk_lattice_io.npz`, each with its sha256;
  - optionally, the verbatim result of converting the P07 host-order wrapper (non-gating).

  It also states that ORT Python CPU EP is not browser evidence.
- **AC-3** `outputs/browser_validation/phase0/ort_web_smoke.json` records EITHER a headless-Chromium run
  (browser and ORT-web versions, EP = wasm, threads, `crossOriginIsolated`, P05 max-abs vs JAX, control
  detected, tie-lattice `top_k` indices vs JAX) OR `status = "BLOCKED"` with the reason. BLOCKED
  produces no compatibility statement.

**Phase 1**

- **AC-4** `scripts/browser_validation/reference_pins.json` records:
  - the LigandMPNN `26ec57ac…` (== live HEAD);
  - the ProteinMPNN clone SHA;
  - the `3870631` resolution;
  - the sha256 of every `.pt`/`.eqx.zst` used;
  - a `titanix` object (host, absolute `uv`/`bth` paths, `bathos_commit`, `catalog_dir`, reference-clone
    paths, `memory_cap`) and a `node_provenance` object from the provenance smoke (T4).

  Harness scripts exit non-zero when the live reference HEAD differs from the pin, on either machine.
- **AC-5** `tests/parity/test_browser_validation_inventory.py` passes. It AST-sweeps the public
  top-level names of `aminx.{inference,sampling,scoring,model,host,tiling,ebm,potts}` plus
  `aminx.run.__all__`, and requires each to map in `tests/parity/browser_validation_paths.json` to a
  P-ID or to `{"internal": "<non-empty reason>"}`. It fails when the internal count exceeds
  `tests/parity/browser_validation_internal_baseline.txt`, and it fails on stale mappings. Its control
  (an injected unmapped symbol) is reported.
- **AC-6** `tests/parity/test_compare_metrics.py` passes, covering:
  - planted max-abs recovered within 1e-12;
  - `pearson(2x, x) == 1` (so Pearson is not a gate);
  - `get_score`-equivalent NLL on a hand example;
  - JS(p, p) = 0 and a closed-form JS within 1e-9;
  - the excess-JS null (two samples from one distribution) centred at 0 within 3 SE over 200
    replicates;
  - excess-JS IUT declaring equivalence for identical sources in ≥ 90 % and rejecting a T×1.05 source
    in ≥ 90 % of 200 replicates at the formula n;
  - TOST rejecting 2δ and accepting 0;
  - neighbour-set equality detecting one swapped index;
  - the no-tie gap rule excluding a planted near-tie.

  The gate is stamped by `scripts/browser_validation/stamp_metric_gate.py` AFTER the T5 commit, and
  `bth gate status mpnn_metric_synthetic_truth --guards src/aminx/parity/compare.py --guards
  tests/parity/test_compare_metrics.py | jq -e '.state == "GREEN"'` holds at verdict time.
- **AC-7** `outputs/browser_validation/fixtures/manifest.json` lists every fixture with its sha256.
  Sets A and B are disjoint, and every protein has L ≥ 64. The corpus has ≥ 1 multichain, ≥ 2
  ligand-bearing and the tie-lattice fixture, and records per-fixture `near_tie_residues` (gap ≤
  1e-4 Å). P02 parser parity is recorded (T6, calibration stage).
- **AC-8** The `layer_a_exact_validate` run on set B has outcome `pass`, or `partial_headroom` with every
  `not_advanced` path listed as not advancing in the T11 table. It also meets all of:
  - `n_skipped = 0`;
  - `weight_source = "eqx"`;
  - a row for every exact/tolerance line of the bars table for P00–P06 (P02 rows included), P09
    scoring and P11–P14;
  - `pt_convert` rows reported as diagnostic;
  - one catalog row (after `bth sync --pull --remote-name titanix && bth compact`) whose native
    `git_hash` AND `metadata.git_hash` equal `H_v` from the run ledger, whose `metadata.prereg_sha256`
    equals `sha256(git show H_v:…/preregistered_params.json)`, and whose `dependency_lock_sha256`
    equals `sha256(git show H_v:uv.lock)` — asserted as ONE SQL equality query returning `[[1]]`
    (R3-C4; NULL never equals, so R2-C14 holds); never read from a local `--out` file.
- **AC-9** Every exact-tier comparison has an in-run control, with `controls_detected = controls_total`.
  Sized controls land in [2×, 10×] the bar. The structural controls are detected:
  - reversed order breaks P05/P06;
  - a tie-group flip breaks P09;
  - a mantissa bit-flip breaks P01;
  - a 1e-3 Å atom shift breaks P02;
  - a k-th-neighbour CA move breaks P03.

  The VALIDATE run has `differential_status = 'passed'`; calibrate runs have no differential.
- **AC-10** Teacher-forced per-step log-probs are ≤ 1e-4 nats for P07, P08, P09-s and P11-s on set B,
  recorded as `tf_max_abs`.
- **AC-11** The sampling-validate run has outcome `pass`, with `differential_status = 'passed'`, the
  same single-query identity/prereg/lock equality as AC-8 at its own `H_v`, and `min_effect > 0` equal
  to the committed value. The pre-registered lane temperatures, the P09-s non-vacuity floor
  (`sampling.p09_qualifying_groups >= 2`), `sampling.p09_tied_positions` and the detected fusion
  control (R3-C6) hold. `n_required`, σ̂, m_ℓ, β, ε and the subset size are in the params at `H_v`
  (committed before the titanix launch). `omitted_aa_count = 0` and
  `x_token_count_aminx = x_token_count_reference = 0` under the X-omit bias column; no waiver exists.
  aminx's native no-X-omit X frequency is reported and filed as a T10 DEVIATION clause.
- **AC-12** `tests/parity/test_mpnn_reference_invariants.py` has ≥ 8 orchestrator-written invariants.
  Each is shown RED against a named injected defect (diff and failure quoted in the verdict doc) and
  GREEN on clean code with zero skips (`assert_no_skips.py`).
- **AC-13** A `parity_validate_mpnn` run exists with:
  - `bth sync --pull --remote-name titanix` and `bth compact` executed first;
  - `prereq_status` derived by the single O1 mapping over the run ledger's `H_v` rows (T11);
  - outcome `pass` or `partial`;
  - `parity_run_type = 'literature_parity'`;
  - a grade from `bathos.parity.compute_grade`.

  The verdict doc `.praxia/docs/audits/<YYMMDD>_mpnn-reference-parity-verdict.md` contains the clause
  checklist, the ceilings, and the per-path advance table. The table is also written to
  `outputs/browser_validation/layer_a/advance_table.json`, and `advance_check.py` exits 0 exactly for
  advancing paths. The claim's `parity_run_id` was bound by `bth campaign attest-parity`. `bth claim
  validate <claim> --catalog-dir "${BTH_CATALOG_DIR:?}" | jq -e '.ok == true and ((.infos // []) |
  any(test("^reference_parity controlled")))'` holds after `bth compact`. `.ok == true` ALONE is
  insufficient — `claim validate` fails OPEN on an unresolvable catalog dir (`claim.py:416-420`) — and
  the asserted string is the literature-parity info (`claim.py:354-359`), not the legacy-path "baseline
  parity PASS" (O7). The claim file is committed immediately after `attest-parity` (R2-C13). `bth
  campaign conclude` reports the parity confound controlled or controlled-by-protocol.
- **AC-14** `README.md` no longer claims "≥ 0.999 Pearson across all five decoding paths"; it states
  per-path verdicts and bars and links the verdict doc. Every `parity_matrix.json` `code_paths` entry
  exists (`test_parity_matrix_paths_exist.py`). `autoregressive-sampling` notes its non-discriminating
  token agreement.
- **AC-15** `scripts/measure_conditional_nll_parity.py` takes a required `--weight-source
  {eqx,pt_convert}`; numbers for both are recorded in the calibration JSON.

**Phase 2**

- **AC-16** Every v1 path has an export wrapper with the runtime inputs of the per-path table. For 5
  fixed keys × T ∈ {0.1, 1.0} × ≥ 2 fixed_mask patterns, with the X-omit column in `bias`,
  `wrapper(order, g, …)` (with `g` built per the Sampling contract from `k_dec = split(key)[1]`) equals
  `inference.sample_autoregressive.kernel(model, key, bundle(from_tie_groups(arange(L), order), same
  bias), …)` bitwise in tokens and logits. The native no-X-omit default is compared statistically (X
  frequency), not bitwise.

  **Tied bundles (R2-C12).** At least one bundle in the matrix has a REAL multi-member tie map (≥ 2
  groups of ≥ 2 members, at least one group containing a fixed member), exercising the P09 wrapper with
  `gid ≠ p`. It asserts bitwise token AND logit equality to the kernel, that each group's noise is drawn
  once at its FIRST occurrence and indexed by `gid`, that `_fuse_one_group` fusion and the
  `max(fixed tokens)` group override reproduce exactly, and that **a fixed group still consumes its
  `gid` draw** (so every other group's draw is unchanged).
- **AC-17** For every wrapper, the exported jaxpr and ONNX graph contain zero `random_bits`,
  `threefry2x32`, `random_wrap`, `random_seed`, `random_fold_in` or `random_split` primitives/ops. The
  unmodified kernel is flagged as a positive control. IREE route only:
  `check_export_safety(…, NATIVE)` and `(…, WASM32)` return `[]` with `trace_ok = true` and the AC-1
  controls re-detected.
- **AC-18** Layer-(b) validate runs pass on set B per bucket and per route. ORT route: ORT-CPU executed
  parity within bars with the op-set/EP report attached. IREE route: native IREE executed parity, with
  wasm32 recorded as CODEGEN_ONLY with its sha256. For each route:
  - integer outputs are exact;
  - the teacher-forced token check passes with exempt flips (gap < 2·bar/T) within the committed cap;
  - the planted-flip control is detected;
  - R0–R3 results are attached;
  - T ∈ {0.1, 1.0} and ≥ 2 fixed_mask patterns are covered.
- **AC-19** For each v1 checkpoint and L ≥ `k_neighbors(checkpoint)` padded to its bucket, real-position
  log-probs are within bar of the unpadded run. L < k and L > max bucket raise typed errors, each tested
  per checkpoint (k = 48 and k = 32).
- **AC-20** The artifact manifest lists, for every artifact, the sha256, P-ID, bucket, target/route,
  checkpoint and source checkpoint sha256.

**Phase 3**

- **AC-21** Every claimed route × browser cell has an evidence JSON containing the UA, browser version,
  runtime and version, EP, threads and `crossOriginIsolated`. Outputs match the same artifact's
  layer-(b) outputs within Phase-3 bars, integers are exact, and controls are detected.
- **AC-22** JS host logic:
  - parser arrays equal aminx's;
  - bucketing/padding are identical on every fixture length;
  - the order generator obeys fixed-first on 100 % of 10,000 draws, passes a pairwise-precedence
    uniformity test among designable positions, and rejects a planted cyclic-shift generator;
  - JS noise matches the Gumbel CDF (KS) with `u ∈ [tiny, 1)` and a unit check that `u = 0` cannot
    occur;
  - in-browser sampled frequencies on a fixed logit vector at N = 10⁶ match analytic
    `softmax(F(logit_transform(l, b))/T)` (chi-square).

  Planted controls sized near the smallest deviation of concern (T × 1.05; a uniform-instead-of-Gumbel
  generator; the cyclic shift) each have a rejection rate recorded over 20 replicates, which must be ≥
  0.9. Non-rejection gates count only with that demonstrated power.
- **AC-23** Per route × browser × bucket × repeat, the benchmark reports separated phases, n_warmup ≥ 5,
  n_iter ≥ 30, p50/p90, bootstrap 95 % CI and peak memory where available. Routes AND the native JAX
  CPU arm are interleaved in one session, with ≥ 3 full-matrix repeats. The regression threshold
  references only the interleaved arm (or is absent, per the recorded fallback).
- **AC-24** A CI regex gate over deliverable text fails on any route/browser compatibility phrase that
  has no AC-21 evidence file, and a planted unsupported claim turns it red.

**Phase 4**

- **AC-25** The bundle contains the page, artifacts, manifest and a report card with a verdict for every
  in-scope path × layer cell, each with a bathos run ID. Known limits include at least:
  - CA-only models (ODQ-2);
  - P15–P24 and P28;
  - that layer-(a) sampling equivalence is statistical;
  - the P09-s restrictions, and explicitly that tied paths (P09 scoring and P09-s) are validated ONLY
    on k-NN-disjoint tie groups — unrestricted intra-group visibility is a known, unvalidated
    divergence from the reference's sequential in-group decoding (R2-C3), recorded as a T10 DEVIATION;
  - untested browsers/devices.
- **AC-26** Every reported number traces to a bathos output through a figure-manifest pin with sha256;
  `bth check --check-outputs` is clean.
- **AC-27** No advisor-facing text states a bar or verdict other than this spec's; grep finds no
  "0.999 … all … paths".

## Open Design Questions

| ID | Question | Alternatives | Status | Decision |
|---|---|---|---|---|
| ODQ-1 | Which browser runtime route is targeted first for layer (c)? | jax2onnx → ONNX Runtime Web (wasm EP first, WebGPU EP later); IREE wasm32 via an emsdk-built IREE runtime; both in parallel | DECIDED | User decision 2026-09-23: jax2onnx → ORT Web, wasm EP first, WebGPU EP later. It is the only route executable in a browser today. Phase 0 measures the multi-key sort tie order: T2 on ORT-CPU, T3 in-browser on the tie-lattice `top_k` graph. The int64 TopK EP-fallback risk applies to the WebGPU EP only and is deferred to the WebGPU phase. IREE wasm32 via emsdk is the fallback route; its results gate only that route (AC-17/18) |
| ODQ-2 | Are CA-only models in scope? | out of scope as a known limit; implement CA-only featuriser + conversion | DECIDED | User decision 2026-09-23: out of scope, listed as a known limit; may be filed as its own backlog item |
| ODQ-3 | Must `decoding_order.py:67`/`:75` be fixed for export? | fix now; host input + separate fix | DECIDED | Host input (ODQ-5); AC-17 re-checks the graphs; a separate aminx backlog item is drafted (T1 step 6), off the critical path |
| ODQ-4 | How is omit_AA supported? | `omit_aa` parameter; bias | DECIDED | Bias. The reference's `-1e8` omit is a bias (`run.py:405-407`), and the X hard-omit is expressed the same way (a `-1e8` column at index 20), supplied identically to native and exported samplers |
| ODQ-5 | Where does sampling randomness live in exported artifacts? | in-graph threefry; host-supplied order + noise | DECIDED | Host-supplied order and Gumbel noise under the Sampling contract (wave_key_w = k_dec; per-gid `fold_in`). It removes in-graph RNG only together with bypassing the coordinate-noise branch (AC-17). Layers b/c become EXACT for tokens given identical noise; AC-16 proves bitwise equality to the native kernel |
| ODQ-6 | Which reference commit is authoritative? | 26ec57ac only; + ProteinMPNN pin; 3870631 | DECIDED | 26ec57ac primary; ProteinMPNN secondary for protein-only paths via P00; 3870631 resolved in T4; README corrected in T11 |
| ODQ-7 | Which bars does the deliverable use? | README; parity_matrix; this spec | DECIDED | This spec's bars table (max-abs + Pearson + NLL; pre-registered IUT equivalence for sampling) |
| ODQ-8 | Browser/device matrix | Chromium+Firefox+WebKit headless + 1 headful each; + macOS/mobile; Chromium only | DECIDED | Chromium, Firefox, WebKit via Playwright, headless + ≥ 1 headful each on this box. Revisit when the advisor names devices |
| ODQ-9 | Deliverable form | static page; npm package; both | DECIDED | Static page + artifacts + report card; npm later over unchanged sha256 artifacts |
| ODQ-10 | v1 export scope | all; single-state core + packer second wave; + multi-state | DECIDED | Single-state core (P01–P09, P11–P13), packer second wave; P10/P15 deferred |
| ODQ-11 | Where do layer-(a) runs execute locally? | local calibration; local smoke only | DECIDED | Superseded in part by ODQ-13: locally only `--dry-run`, `--smoke` (< 60 s) and the local `bth run` smoke of each validate script, threads capped at 4 |
| ODQ-12 | Where does the work live? | aminx; aminx + xtrax items; xtrax | DECIDED | aminx; xtrax items filed when a phase needs them, cross-referenced by task_id |
| ODQ-13 | Execution venue for Phase-1 layer-(a) calibrate/validate runs | local (waiver of the narrow-runs rule); Engaging via `bth submit` + git-clone path; titanix via `bth run` over ssh | DECIDED | **User decision 2026-09-24: titanix** ("We should get bathos on titanix"; done by the orchestrator: bathos uv tool at commit `84be544e`). Path O6′: bare repo on titanix, per-run detached worktree at `H`, tracked wrapper under tmux, `bth sync --pull --remote-name titanix`. Engaging is not used by this spec; it remains an optional, unspecified future venue |

## Fixer Tasks

**Common context for every task.**
- **Worktree.** Work in an aminx worktree created from `origin/main` (a5ced9c7 or later); `$WT` is its
  root and all paths are relative to it. Do not read or edit the stale `/home/marielle/projects/aminx`
  except the read-only `$REFERENCE_PATH=/home/marielle/projects/aminx/reference_ligandmpnn_clone`.
- **Environment.** Use `uv run --extra dev` (the default `dev` group adds torch/scipy/biopython/dm-tree;
  the `dev` extra adds pytest/chex). Add `--extra benchmark` for prody and `--with "$BATHOS_REQ"` where
  bathos is imported, with `BATHOS_REQ="bathos @
  git+https://github.com/maraxen/bathos@84be544ecb45734f46e43d22f351a54d6edd6ae5"` on BOTH machines
  (never PyPI `bathos==0.13.0a4`: same version string, predates the `bth run` exec-as-given fix). Cap
  local threads with `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`.
- **Tests.** Never run a whole suite locally; use single files or `-k`. Heavy tests need
  `-m parity_heavy`.
- **Ruff.**
  - `src/`: `ruff check --no-fix <f>` and `ruff format --check <f>`.
  - `tests/`/`scripts/` (force-excluded): `ruff check --no-fix --config "exclude=[]" --select
    E,F,W,I,B <f> 2>&1 | tee <log>; grep -L "No Python files found" <log>` and `ruff format --check
    --config "exclude=[]" <f>`. GNU grep ≥ 3.5 `-L` exits 0 iff the log lacks the string; never write
    `! grep …` in a gate, because a `!`-negated command never trips `set -e`.
  - All ruff commands run under `uv run --extra dev`.
- **bathos CLI.** Run `bth <subcommand> --help` before a subcommand's first use and match its flags
  (Recon correction 14). Assert every `bth` gate on its OUTPUT with `jq -e` on exact values, never on
  its exit code and never by substring grep (R3-C4). `bth validate-sidecar <path>` must succeed for every new `.bth.toml`, and
  `scripts/browser_validation/lint_sidecars.py` (T7) must pass.
- **Catalog gates (R2-C1).** A local `bth run` writes only cool-tier parquet (`runner.py:847-870`) and
  `bth sql` reads only the warm `bathos.db` (`query.py:427-443`), so **every `bth sql` gate in this spec
  is preceded by `bth compact`** (for a titanix run: `bth sync --pull --remote-name titanix && bth
  compact`). Export `BTH_CATALOG_DIR` once per session and pass it wherever a bathos subcommand accepts
  `--catalog-dir`. Gate rules (R3-C4):
  - every gate block starts with `cd "$WT" && set -euo pipefail`; every interpolated value is `${VAR:?}`
    and length-checked (`test ${#H} -eq 40`, `test ${#PS} -eq 64`); hashes of committed files are
    `git cat-file -e "$H:<path>" && git show "$H:<path>" | sha256sum | cut -d' ' -f1`;
  - **latest-row form** (local runs, `H = git rev-parse HEAD` at run time): `bth sql "SELECT outcome
    FROM runs WHERE command LIKE '%<stem>.py --out%' AND git_hash = '${H:?}' ORDER BY timestamp DESC
    LIMIT 1" | jq -e '.count == 1 and (.rows[0][0] | IN("pass", …))'`;
  - **identity form** (titanix runs, `H_v` from the run ledger): one `SELECT count(*)` whose WHERE
    carries every equality (native `git_hash`, `json_extract_string(metadata,'$.git_hash')`,
    `$.prereg_sha256`, `dependency_lock_sha256`, outcome, `differential_status`) and `| jq -e '.rows ==
    [[1]]'`. NULL never equals, so a NULL lock hash fails (R2-C14, `checker.py:245-257`).

  Native `runs.git_hash` is `compact.py:339`; metadata is `compact.py:351` (DuckDB). No row is a
  **failure, never a pass** (O1: `prereq_status = 'missing'`, exit 6); no query omits the `git_hash`
  filter.
- **`git_clean` (R2-C13, R3-C2).** `layer_a_common.provenance()` calls `bathos.git.capture_git_state(cwd)`
  and exits 3 unless `provenance_source == 'git'` (the live-git channel; a myxcel env/sidecar channel
  would take precedence, `cisternal provenance/channels.py:170-185`); `git_hash` = its hash, `git_clean`
  = `not dirty`. Its `dirty` is `git status --porcelain` non-empty (`cisternal telemetry/git_state.py:
  224-225`), which counts untracked files, so T4 gitignores `outputs/browser_validation/` (files that
  must be tracked there — params, run ledger, fixture manifest and raw fixtures — are added with `git
  add -f` and stay tracked), and T3/T4/T5 gitignore `node_modules/`, `.cache/` and
  `.bth/synthetic_recovery_ledger.json` (`gate.py:25`). The claim file is committed immediately after
  `bth campaign attest-parity` rewrites it (T11 step 5).
- **Titanix execution path (orchestrator) — O6′ (ODQ-13).** All layer-(a) calibrate, validate and
  provenance-smoke runs execute on titanix (measured 2026-09-24: user `solab`, `HOME=/home/solab`, 20
  cores, 123 GiB RAM, a GPU, `/` 92 % used with ~148 GB free, GitHub reachable over https). The
  non-interactive ssh PATH lacks `/home/solab/.local/bin`, so every remote command uses absolute paths
  `TX_UV=/home/solab/.local/bin/uv` and `TX_BTH=/home/solab/.local/bin/bth` (bathos uv tool at commit
  `84be544e…`). `~/projects/*` on titanix are myxcel copies without `.git` and are never used. No `bth
  submit`, sbatch, myxcel push or `_bth_env.sh` is involved, which removes the unquoted `bth submit`
  argv join (O9) and its catalog-identity check (R3-C7) by construction.
  - **One-time setup (T4).** (a) `ssh titanix 'git init --bare /home/solab/bv/aminx.git && mkdir -p
    /home/solab/bv/logs'`; `git -C "$WT" remote add titanix-bv titanix:/home/solab/bv/aminx.git`.
    (b) Reference clones on titanix by `git clone` over https: LigandMPNN into
    `/home/solab/bv/ref/LigandMPNN` at `26ec57ac…` with `bash get_model_params.sh ./model_params` run
    there, and ProteinMPNN into `/home/solab/bv/ref/ProteinMPNN` at its T4 pin; every `.pt` sha256 must
    equal `reference_pins.json:weights_sha256`. (c) Catalog identity: `.bth.toml` gets
    `[remotes.titanix]` with `host = "titanix"`, `remote_root = "/home/solab/bv/catalog-aminx"`
    (hand-edited; `bth remote add` would also rewrite the tracked `scripts/slurm/_bth_env.sh`).
    `bth sync` targets `remote_catalog_path(remote_root)` = `/home/solab/bv/catalog-aminx/.bth/catalog`
    (`cluster_catalog.py:21-26`, called at `sync.py:78-80`) and rsyncs `runs/<slug>/` and `campaigns/`
    there (`sync.py:85-87,258-264`), so the wrapper exports exactly that path as `BTH_CATALOG_DIR`,
    plus `BTH_PROJECT_SLUG=aminx`.
  - **Per run of stem `s` with argv `A…`** — tracked scripts `scripts/browser_validation/titanix_launch.sh`
    (local) and `…/titanix_run.sh` (titanix), both written in T4:
    1. launch.sh, from `$WT`: `test -z "$(git status --porcelain)"`; `H=$(git rev-parse HEAD)`; refuse if
       any systemd user unit `bv-*` is active on titanix (`systemctl --user list-units 'bv-*' --state=active`
       non-empty; one heavy run at a time); `git push titanix-bv
       "$H:refs/heads/bv/$s-${H:0:12}"`.
    2. launch.sh, over `ssh titanix bash -s` with a heredoc, `RD=/home/solab/bv/aminx-<slug>-$s-${H:0:12}`:
       refuse if `df -B1G --output=avail /home/solab` reports < 30; `git -C /home/solab/bv/aminx.git
       worktree add --detach "$RD" "$H"`; `cd "$RD" && $TX_UV sync --frozen --extra dev --extra
       benchmark` (network available; `--frozen` keeps `uv.lock` byte-identical to `H`).
    3. launch.sh: `ssh titanix systemd-run --user --unit="bv-$s-${H:0:12}" --collect -p MemoryMax=64G
       -p MemorySwapMax=0 "$RD/scripts/browser_validation/titanix_run.sh" "$RD" "$H" "$CAMPAIGN_ID" "$s" A…`,
       a transient user unit detached from the ssh session. Measured on titanix 2026-09-24: `tmux` is NOT
       installed; user lingering is enabled (`loginctl show-user solab -p Linger` = `Linger=yes`); a transient
       `systemd-run --user --unit=… -p MemoryMax=1G --collect` unit ran to completion after the ssh session
       ended. The unit enforces the memory cap natively (no "when available" fallback) and its log is
       `journalctl --user -u bv-$s-${H:0:12}`. **O9:** ssh joins argv with spaces exactly as `bth submit`
       did, so launch.sh exits 2 if any argument contains whitespace or a shell metacharacter.
    4. run.sh: `cd "$RD"`; exit 3 unless `git rev-parse HEAD` = `H`, `git status --porcelain` is empty,
       and the bathos tool's `direct_url.json` `commit_id` = `84be544e…`; export `BTH_CATALOG_DIR`,
       `BTH_PROJECT_SLUG`, `REFERENCE_PATH=/home/solab/bv/ref/LigandMPNN`,
       `PROTEINMPNN_PATH=/home/solab/bv/ref/ProteinMPNN`, `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=`
       (the exact tier is CPU), `OMP/OPENBLAS/MKL_NUM_THREADS=16`, `BTH_BIN=$TX_BTH`; run `$TX_BTH
       compact` (so `prereq_check()` sees a warm catalog); then run `taskset -c 0-15 $TX_BTH run
       --campaign-id "$CAMPAIGN_ID" --output-paths outputs/browser_validation/layer_a/$s.json -- $TX_UV
       run --frozen --no-sync --extra dev --extra benchmark --with "$BATHOS_REQ" python
       scripts/browser_validation/$s.py A… --out outputs/browser_validation/layer_a/$s.json` (the memory cap
       comes from the enclosing transient unit, step 3). Output goes to
       `/home/solab/bv/logs/<session>.log`, the exit code to `…/<session>.exit`.
    5. The orchestrator polls `ssh titanix cat /home/solab/bv/logs/<session>.exit` in the foreground (≤
       600 s per call), then locally: `bth sync --pull --remote-name titanix && bth compact`; appends
       `{stem, H_v: H, session, exit, free_gb_before_sync, prereg_sha256, lock_sha256}` (the two hashes
       from `git show H:`) to the tracked `outputs/browser_validation/layer_a/run_ledger.json` and commits
       it; runs the gates; rsyncs `titanix:$RD/outputs/browser_validation/` into the gitignored
       `outputs/browser_validation/titanix/$s-${H:0:12}/`; then `ssh titanix git -C
       /home/solab/bv/aminx.git worktree remove --force "$RD"` (O10: frees the per-run venv).
  - **Campaigns** resolve against titanix's own catalog (`runner.py:317-341`), so `bth sync
    --remote-name titanix` (push) follows every campaign creation, checked by `ssh titanix ls
    /home/solab/bv/catalog-aminx/.bth/catalog/campaigns/ | grep -F <campaign-name>`.
  - **Provenance smoke** (T4 step 5, after the campaign exists, before any calibrate; R3-C8): stem
    `provenance_smoke`, space-free argv.
- **Reading results.** `bth sync --pull` moves catalog parquet and campaigns only; gates read the
  catalog, never a local `--out`. The rsynced `titanix/` copy serves only per-row detail in the verdict
  doc.
- **Commits.** Commit per task; never push `origin` (the only push is to `titanix-bv`). Run `git
  status --short` before editing.

### T1 — Phase 0: export-safety census of real aminx callables

1. Create `scripts/browser_validation/export_safety_census.py` (argparse + logging, `--out`, `--bucket`
   128, `--dry-run`). Log `xtrax.__file__`/`__version__`. If `xtrax.export.safety` lacks
   `_RANDOM_PERMUTATION_RULE` or `_UNBATCHED_THREEFRY_RULE`, exit 2 with an instruction to rerun with
   `PYTHONPATH=/home/marielle/projects/xtrax/src`.
2. Build the AC-1 callables with shipped weights (`aminx.io.weights.load_weights`) at L = `--bucket` on
   synthetic well-separated coords (random ×10 Å). The P07 kernel gets
   `WaveScheduleBundle.from_tie_groups(arange(L), host_order)`. For each callable:
   - run `jax.make_jaxpr` → `trace_ok` + exception text;
   - then `xtrax.export.check_export_safety([], {}, inputs, fn, target)` for NATIVE and WASM32;
   - record `blockers = null` unless `trace_ok`.
3. Controls: bare `random_decoding_order(key, L)` must yield `random-permutation`, and
   `lambda x: jax.lax.top_k(x, 8)` must yield `unlegalizable-op`. Write
   `controls_detected`/`controls_total`.
4. Write `--out` and `$BTH_RESULTS_PATH`. Sidecar `export_safety_census.bth.toml` (`exploration`,
   `novel = true`):
   - `ctrl_blind` (residual): `controls_detected < controls_total`;
   - `untraceable` (residual): `n_untraceable > 0`;
   - `pass`: `n_untraceable = 0`.

   Schema `{n_callables:int, n_untraceable:int, n_with_blockers:int, controls_total:int,
   controls_detected:int, xtrax_file:str, rules_present:bool}`.
5. `bth campaign create "aminx-bv-phase0" --mode exploration --question "Which real aminx callables are
   export-safe today?"`; run under `bth run --campaign-id`.
6. Draft (do not file) the `decoding_order.py:67/:75` backlog text into `followup_backlog_text`.

**Files**: `scripts/browser_validation/__init__.py` (create, empty),
`scripts/browser_validation/export_safety_census.py` (create), `…/export_safety_census.bth.toml`
(create).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
bth validate-sidecar scripts/browser_validation/export_safety_census.bth.toml
OMP_NUM_THREADS=4 uv run --extra dev python scripts/browser_validation/export_safety_census.py --dry-run --out outputs/browser_validation/phase0/census_dryrun.json
OMP_NUM_THREADS=4 bth run --campaign-id <phase0-id> --output-paths outputs/browser_validation/phase0/export_safety_census.json -- uv run --extra dev python scripts/browser_validation/export_safety_census.py --out outputs/browser_validation/phase0/export_safety_census.json
bth compact
H=$(git rev-parse HEAD); test ${#H} -eq 40
bth sql "SELECT outcome FROM runs WHERE command LIKE '%export_safety_census.py --out%' AND git_hash = '${H:?}' ORDER BY timestamp DESC LIMIT 1" | jq -e '.count == 1 and (.rows[0][0] | IN("pass", "untraceable"))'   # untraceable is a finding, report it
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B scripts/browser_validation/export_safety_census.py 2>&1 | tee outputs/browser_validation/phase0/ruff_t1.log; grep -L "No Python files found" outputs/browser_validation/phase0/ruff_t1.log
uv run --extra dev ruff format --check --config "exclude=[]" scripts/browser_validation/export_safety_census.py
```
**Scope estimate**: ~220 LOC.

### T2 — Phase 0: jax2onnx → ONNX Runtime (CPU EP) spike, incl. the multi-key sort tie test

1. Create `scripts/browser_validation/jax2onnx_spike.py` (argparse, `--out`, `--bucket 128`,
   `--dry-run`). Run it via `uv run --extra dev --with "jax2onnx==<pin>" --with "onnxruntime==<pin>"`,
   using the latest releases on the run date. Record the pins; do not add them to `pyproject.toml`.
2. Convert P05 conditional logits (shipped `proteinmpnn_v_48_020`, L = 128, `$REFERENCE_PATH/inputs/
   1BC8.pdb` parsed by aminx, noise 0). On failure, record the verbatim error and primitive and go to
   step 6.
3. Execute on ORT CPU EP: log-prob max-abs vs JAX and EXACT neighbour indices (exposed as a second
   output). Control: W_out bias[0] += 5e-4 must be detected (max-abs ≥ 2e-4).
4. Tie test: CA on a 1.0 Å cubic grid (`_lattice.py`); convert `aminx.model.features.top_k` alone;
   ORT-CPU vs JAX indices EXACT → `tie_indices_identical`.
5. Optional, non-gating: attempt converting the P07 host-order wrapper (order + noise as inputs) and
   record success or the verbatim error in `p07_conversion`.
6. Write to `outputs/browser_validation/phase0/`: `p05_L128.onnx`, `p05_L128_perturbed.onnx`,
   `topk_lattice.onnx`, `p05_inputs.npz`, `p05_jax_outputs.npz`, `topk_lattice_io.npz`, with every
   sha256 recorded in the JSON (`--out` + `$BTH_RESULTS_PATH`). Sidecar `jax2onnx_spike.bth.toml`
   (`exploration`, `novel = true`):
   - `not_converted` (residual): `NOT converted`;
   - `fail` (residual): `NOT (ort_max_abs <= 1e-4 AND nbr_exact AND control_detected)`;
   - `tie_unstable` (non-residual): `ort_max_abs <= 1e-4 AND nbr_exact AND control_detected AND NOT
     tie_indices_identical`;
   - `pass`: `ort_max_abs <= 1e-4 AND nbr_exact AND control_detected AND tie_indices_identical`;
   - note: "ORT Python CPU EP is not ORT Web; this is not browser evidence".

   Declaration ORDER is load-bearing: bathos picks the FIRST matching branch (`sidecar.py:679-695`) and
   every non-residual label is pass-direction, so `fail` must precede `tie_unstable` — otherwise a
   conversion with a large max-abs or an undetected control is labelled `tie_unstable` and T3 proceeds
   on a bad artifact (R2-C9). `lint_sidecars.py` keeps `tie_unstable`/`not_converted` in its whitelist
   unchanged.

**Files**: `scripts/browser_validation/jax2onnx_spike.py`, `…/jax2onnx_spike.bth.toml`,
`scripts/browser_validation/_lattice.py` (create; reused by T6).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
bth validate-sidecar scripts/browser_validation/jax2onnx_spike.bth.toml
OMP_NUM_THREADS=4 bth run --campaign-id <phase0-id> --output-paths outputs/browser_validation/phase0/jax2onnx_spike.json -- uv run --extra dev --with "jax2onnx==<pin>" --with "onnxruntime==<pin>" python scripts/browser_validation/jax2onnx_spike.py --out outputs/browser_validation/phase0/jax2onnx_spike.json
bth compact
H=$(git rev-parse HEAD); test ${#H} -eq 40
bth sql "SELECT outcome FROM runs WHERE command LIKE '%jax2onnx_spike.py --out%' AND git_hash = '${H:?}' ORDER BY timestamp DESC LIMIT 1" | jq -e '.count == 1 and (.rows[0][0] | IN("pass", "tie_unstable", "not_converted", "fail"))'   # error/unknown is a task failure
if jq -e '.converted' outputs/browser_validation/phase0/jax2onnx_spike.json >/dev/null; then for f in p05_L128.onnx p05_L128_perturbed.onnx topk_lattice.onnx p05_inputs.npz p05_jax_outputs.npz topk_lattice_io.npz; do test -s outputs/browser_validation/phase0/$f && jq -e --arg f "$f" '.artifacts[$f].sha256 | length == 64' outputs/browser_validation/phase0/jax2onnx_spike.json >/dev/null || exit 1; done; fi   # T3 precondition artifacts
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B scripts/browser_validation/jax2onnx_spike.py scripts/browser_validation/_lattice.py 2>&1 | tee outputs/browser_validation/phase0/ruff_t2.log; grep -L "No Python files found" outputs/browser_validation/phase0/ruff_t2.log
uv run --extra dev ruff format --check --config "exclude=[]" scripts/browser_validation/jax2onnx_spike.py scripts/browser_validation/_lattice.py
```
**Scope estimate**: ~300 LOC.

### T3 — Phase 0: ORT Web (wasm EP) headless smoke (after T2 outcome `pass` or `tie_unstable`)

1. Precondition: `ort_web_smoke.py` asserts every T2 artifact exists and matches its recorded sha256,
   exiting 3 otherwise. Check `node --version` and `npx playwright --version`. If either is unavailable
   and cannot be installed into `$WT/browser/` via npm, write `status = "BLOCKED"` with the reason and
   stop.
2. Create `browser/smoke/`:
   - `package.json` pinning `onnxruntime-web` and `@playwright/test` exactly;
   - `index.html` + `smoke.mjs`, which load the T2 `.onnx` and inputs (npz converted to JSON), run the
     wasm EP with `numThreads = 1` for P05, its perturbed twin and `topk_lattice.onnx`, and post the
     outputs, `navigator.userAgent`, `ort.env.versions` and `self.crossOriginIsolated`;
   - `serve.mjs` (COOP `same-origin`, COEP `require-corp`);
   - `run_smoke.mjs` (Playwright headless Chromium).
3. `scripts/browser_validation/ort_web_smoke.py` runs node, then compares against T2's JAX outputs:
   P05 max-abs, EXACT indices, the perturbed control, and tie-lattice indices vs JAX
   (`tie_indices_identical_browser`). It writes the AC-3 JSON.
4. Sidecar `ort_web_smoke.bth.toml` (`exploration`, `novel = true`):
   - `blocked`: `status = 'BLOCKED'`;
   - `pass`: `max_abs <= 1e-4 AND idx_exact AND control_detected`;
   - `fail` (residual): the rest.

   `tie_indices_identical_browser` is recorded, not gated. Schema adds `browser:str,
   browser_version:str, ort_web_version:str, cross_origin_isolated:bool, threads:int`.
5. Add `browser/**/node_modules/` to `.gitignore`.

**Files**: `browser/smoke/{package.json,index.html,smoke.mjs,serve.mjs,run_smoke.mjs}`,
`scripts/browser_validation/ort_web_smoke.py`, `…/ort_web_smoke.bth.toml` (create); `.gitignore`
(modify).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
bth validate-sidecar scripts/browser_validation/ort_web_smoke.bth.toml
bth run --campaign-id <phase0-id> --output-paths outputs/browser_validation/phase0/ort_web_smoke.json -- uv run --extra dev python scripts/browser_validation/ort_web_smoke.py --out outputs/browser_validation/phase0/ort_web_smoke.json
bth compact
H=$(git rev-parse HEAD); test ${#H} -eq 40
bth sql "SELECT outcome FROM runs WHERE command LIKE '%ort_web_smoke.py --out%' AND git_hash = '${H:?}' ORDER BY timestamp DESC LIMIT 1" | jq -e '.count == 1 and (.rows[0][0] | IN("pass", "blocked"))'
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B scripts/browser_validation/ort_web_smoke.py 2>&1 | tee outputs/browser_validation/phase0/ruff_t3.log; grep -L "No Python files found" outputs/browser_validation/phase0/ruff_t3.log
uv run --extra dev ruff format --check --config "exclude=[]" scripts/browser_validation/ort_web_smoke.py
```
**Scope estimate**: ~170 LOC Python + ~200 LOC JS.

### T4 — Phase 1: bathos staging, reference pins, literature-parity config

1. Pins:
   - `git -C $REFERENCE_PATH rev-parse HEAD` must equal `26ec57ac…`;
   - shallow-clone `https://github.com/dauparas/ProteinMPNN.git` into `$WT/.cache/reference/ProteinMPNN`
     (add `.cache/` to `.gitignore`; harnesses read `$PROTEINMPNN_PATH`, defaulting to this path) and
     record its HEAD;
   - resolve `3870631` in both clones (`git rev-parse --verify 3870631^{commit}`) as repository + SHA
     or `"not found"`;
   - sha256 every `.pt` in `$REFERENCE_PATH/model_params/` and `ProteinMPNN/vanilla_model_weights/`,
     and every `.eqx.zst` aminx resolves for `proteinmpnn_v_48_{002,020}`, `solublempnn_v_48_020`,
     both membrane checkpoints, `ligandmpnn_v_32_010_25` and `ligandmpnn_sc_v_32_002_16`;
   - write `scripts/browser_validation/reference_pins.json`, plus `pins.py` with
     `assert_reference_pinned(path, expected_sha)` (`SystemExit(3)` on mismatch).
2. Literature: `docs/references/proteinmpnn_2022.pdf` and `docs/references/ligandmpnn.pdf` (open-access
   preprints). Verify each DOI resolves before recording it in `citation_note`. Create
   `scripts/browser_validation/parity.bth.toml` as in the Overview.
3. Titanix plumbing (O6′), fixer-written: `.bth.toml` `[remotes.titanix]` (host `titanix`, `remote_root
   = "/home/solab/bv/catalog-aminx"`; `[project].root` unchanged); `.gitignore` gains
   `outputs/browser_validation/`; `scripts/browser_validation/titanix_launch.sh` and `titanix_run.sh`
   exactly as the common context specifies (`bash -n` clean; launch.sh's O9 argument check has a
   `--self-test` that feeds it `a b` and `a;b` and expects exit 2); and
   `scripts/browser_validation/provenance_smoke.py` (+ `provenance_smoke.bth.toml`, `stage_name =
   "calibration"`, `novel = true`). The stub takes only `--out`, and emits into `--out` and
   `$BTH_RESULTS_PATH`: `git_source`, `git_hash`, `git_dirty` (from
   `bathos.git.capture_git_state(Path.cwd())`), `bathos_commit` (the imported bathos's `direct_url.json`
   `commit_id`), `cisternal_version`, `torch_ok`/`jax_ok` (import + one tiny op), `cpu_only` (every
   `jax.devices()` is CPU), `locked_versions_match` (installed jax/jaxlib/numpy/torch/equinox versions
   equal the `uv.lock` entries, so the `--with` overlay shadowed nothing) and `systemd_scope_ok`
   (`systemd-run --user --scope -p MemoryMax=1G true` exits 0). Outcomes: `pass` iff `git_source = 'git'
   AND NOT git_dirty AND torch_ok AND jax_ok AND cpu_only AND locked_versions_match AND bathos_commit =
   '84be544ecb45734f46e43d22f351a54d6edd6ae5'`; `fail` (residual) otherwise.
4. Campaign and claim:
   - `bth campaign create "aminx-bv-layer-a" --mode confirmation --question "Do aminx MPNN paths
     reproduce LigandMPNN@26ec57ac?" --hypothesis "H_exact_within_bar"`;
   - `bth claim scaffold <id>`, then fill `.bth/claims/aminx-bv-layer-a.claim.toml` from the Overview
     (`parity_run_id = ""`);
   - `bth claim register <path> --campaign-id <id>`.

   `bth claim validate` is EXPECTED to report `ok: false` with exactly the baseline-admissibility error
   until T11 (`claim.py:324-329`).
5. **Orchestrator, after step 4 is committed (R3-C8):** the one-time titanix setup (O6′ (a)–(b)), with
   `.pt` sha256s checked against the pins; `bth sync --remote-name titanix` (pushes the campaign); then
   `titanix_launch.sh provenance_smoke <layer-a-id>` and the O6′ step-5 sync/ledger sequence. Record in
   `reference_pins.json` the `titanix` object (`memory_cap = "systemd-unit"`; the smoke itself runs inside
   the O6′ transient unit, so a smoke that completes proves the unit mechanism; `systemd_scope_ok` must be
   true, else the T4 gate fails — measured working 2026-09-24) and `node_provenance = {source, git_hash, smoke_head: H_smoke, dirty, bathos_commit,
   cisternal_version, locked_versions_match}` copied from the smoke's catalog row, and commit. If
   `locked_versions_match` is false, stop: drop `--with` from run.sh and make `provenance()` run the
   live channel's own three git calls (`cisternal telemetry/git_state.py:215-225`); record
   `titanix.provenance_impl`. No calibrate launches before this commit.

**Files**: `scripts/browser_validation/{reference_pins.json,pins.py,parity.bth.toml,titanix_launch.sh,
titanix_run.sh,provenance_smoke.py,provenance_smoke.bth.toml}`, `docs/references/*.pdf`,
`.bth/claims/aminx-bv-layer-a.claim.toml` (create); `.bth.toml`, `.gitignore` (modify);
`outputs/browser_validation/layer_a/run_ledger.json` (created by the orchestrator, `git add -f`).
**Gate** (fixer, steps 1–4):
```bash
cd "$WT" && set -euo pipefail
test "$(git -C $REFERENCE_PATH rev-parse HEAD)" = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
jq -e '(.ligandmpnn_commit|startswith("26ec57ac")) and (.proteinmpnn_commit|length>0) and has("r3870631") and (.weights_sha256|length>0)' scripts/browser_validation/reference_pins.json
grep -qx 'parity_run_id = ""  # NEVER hand-edit: bound by bth campaign attest-parity (T11)' .bth/claims/aminx-bv-layer-a.claim.toml
test "$(grep -c '^parity_run_id = ""' .bth/claims/aminx-bv-layer-a.claim.toml)" = 1   # R2-C4: attest-parity matches this exact literal
bth claim validate .bth/claims/aminx-bv-layer-a.claim.toml | jq -e '.ok == false and (.errors|length) == 1 and (.errors[0]|startswith("baseline admissibility not established for"))'
bth compact
bth sql "SELECT claim_path FROM campaigns WHERE name = 'aminx-bv-layer-a'" | jq -e '.count == 1 and (.rows[0][0] | endswith("aminx-bv-layer-a.claim.toml"))'
bash -n scripts/browser_validation/titanix_launch.sh && bash -n scripts/browser_validation/titanix_run.sh && bash scripts/browser_validation/titanix_launch.sh --self-test
test "$(git check-ignore -q outputs/browser_validation/x.json && echo ignored)" = ignored
bth validate-sidecar scripts/browser_validation/provenance_smoke.bth.toml
uv run --extra dev --with "$BATHOS_REQ" python scripts/browser_validation/provenance_smoke.py --out outputs/browser_validation/layer_a/provenance_smoke_local.json && jq -e '.git_source == "git" and .jax_ok and .torch_ok' outputs/browser_validation/layer_a/provenance_smoke_local.json
uv run --extra dev --with "$BATHOS_REQ" python -c "from bathos.parity import parse_parity_toml; parse_parity_toml('scripts/browser_validation/parity.bth.toml')"
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B scripts/browser_validation/pins.py scripts/browser_validation/provenance_smoke.py 2>&1 | tee outputs/browser_validation/ruff_t4.log; grep -L "No Python files found" outputs/browser_validation/ruff_t4.log
uv run --extra dev ruff format --check --config "exclude=[]" scripts/browser_validation/pins.py scripts/browser_validation/provenance_smoke.py
```
**Gate** (orchestrator, step 5):
```bash
cd "$WT" && set -euo pipefail
ssh titanix ls /home/solab/bv/catalog-aminx/.bth/catalog/campaigns/ | grep -F aminx-bv-layer-a   # campaign reached titanix before the smoke (R3-C8)
bth sync --pull --remote-name titanix && bth compact
HS=$(jq -r '.node_provenance.smoke_head' scripts/browser_validation/reference_pins.json); test ${#HS} -eq 40
jq -e '.node_provenance.source == "git" and .node_provenance.git_hash == .node_provenance.smoke_head and .node_provenance.dirty == false and .node_provenance.bathos_commit == "84be544ecb45734f46e43d22f351a54d6edd6ae5" and .node_provenance.locked_versions_match == true and (.titanix.memory_cap | IN("systemd-run", "projected"))' scripts/browser_validation/reference_pins.json
bth sql "SELECT count(*) FROM runs WHERE command LIKE '%provenance_smoke.py --out%' AND git_hash = '${HS:?}' AND json_extract_string(metadata,'\$.git_hash') = '${HS:?}' AND json_extract_string(metadata,'\$.bathos_commit') = '84be544ecb45734f46e43d22f351a54d6edd6ae5' AND outcome = 'pass' AND dependency_lock_sha256 IS NOT NULL" | jq -e '.rows == [[1]]'
test "$(ls "${BTH_CATALOG_DIR:?}"/runs/aminx/ | wc -l)" -gt 0   # pulled into runs/<slug>/ (sync.py:85-87)
```
**Scope estimate**: ~120 LOC + ~90 LOC shell + config.

### T5 — Phase 1: comparison-metric library with synthetic-truth tests

1. Create `src/aminx/parity/compare.py` (numpy/scipy, typed, no torch) with:
   - `max_abs`, `pearson`, `ratio_to_bar`;
   - `reference_nll(log_probs, seq, mask, get_score)` (caller's function, never retyped);
   - `argmax_agreement(a, b, margin)`;
   - `neighbor_set_equality(idx_a, idx_b, mask)`;
   - `no_tie_mask(ca, k, gap=1e-4)`, which excludes residues whose k-th vs (k+1)-th CA distance gap
     ≤ `gap`;
   - `js_divergence(p, q)` (base e) and `mean_positional_js(counts_a, counts_b)`;
   - `excess_js(a1, a2, r1, r2)`, returning E = D(a1, r1) − ½[D(a1, a2) + D(r1, r2)];
   - `excess_js_upper(a1, a2, r1, r2, n_boot=1000, alpha=0.05, rng)`;
   - `iut_equivalent(lane_results)`, which returns True iff every lane passes (no multiplicity
     adjustment);
   - `tost_mean_diff(x, y, delta, alpha)` → (pass, p_lower, p_upper);
   - `required_n(sigma, delta, alpha=0.05, power=0.9)` = ⌈2(z₁₋α + z₁₋β/2)²σ²/δ²⌉.

   No Holm function: multiplicity adjustment is not part of any gating path.
2. `tests/parity/test_compare_metrics.py` implements every AC-6 check against planted truth (fixed
   seeds), including `pearson(2*x, x) == 1.0`.
3. `scripts/browser_validation/assert_no_skips.py <junit_xml>` exits 1 if `skipped > 0` or `tests == 0`,
   printing both counts. It is tested against two hand-written junit strings.
4. `scripts/browser_validation/stamp_metric_gate.py`:
   - refuses unless both guarded files are committed and clean;
   - runs the test file with `--junitxml`;
   - runs `assert_no_skips.py`;
   - only then calls `bth gate stamp mpnn_metric_synthetic_truth --result pass`.

   Run it AFTER the T5 commit.
5. Add `.bth/synthetic_recovery_ledger.json` to `.gitignore` (R2-C13): `bth gate stamp` writes that
   ledger into the worktree (`gate.py:25`), and every later validate's `incomplete` branch reads
   `git_clean`. With the tracked-only definition an untracked ledger is already harmless; gitignoring it
   also keeps `git status --short` clean for the per-task checks.

**Files**: `src/aminx/parity/compare.py`, `tests/parity/test_compare_metrics.py`,
`scripts/browser_validation/{assert_no_skips,stamp_metric_gate}.py` (create); `.gitignore` (modify).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
OMP_NUM_THREADS=4 uv run --extra dev pytest tests/parity/test_compare_metrics.py -q -rs --junitxml=outputs/browser_validation/junit_t5.xml
uv run --extra dev python scripts/browser_validation/assert_no_skips.py outputs/browser_validation/junit_t5.xml
uv run --extra dev ruff check --no-fix src/aminx/parity/compare.py && uv run --extra dev ruff format --check src/aminx/parity/compare.py
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B tests/parity/test_compare_metrics.py scripts/browser_validation/assert_no_skips.py scripts/browser_validation/stamp_metric_gate.py 2>&1 | tee outputs/browser_validation/ruff_t5.log; grep -L "No Python files found" outputs/browser_validation/ruff_t5.log
uv run --extra dev ruff format --check --config "exclude=[]" tests/parity/test_compare_metrics.py scripts/browser_validation/assert_no_skips.py scripts/browser_validation/stamp_metric_gate.py
uv run --extra dev ty check src/aminx/parity/compare.py
git commit -m "..." && uv run --extra dev python scripts/browser_validation/stamp_metric_gate.py
bth gate status mpnn_metric_synthetic_truth --guards src/aminx/parity/compare.py --guards tests/parity/test_compare_metrics.py | jq -e '.state == "GREEN"'
```
**Scope estimate**: ~300 LOC src + ~320 LOC tests.

### T6 — Phase 1: fixture corpus, near-tie map, structure-ingestion parity (P02)

1. `scripts/browser_validation/fixtures.py` `build_corpus(out_dir)`:
   - candidates are `$REFERENCE_PATH/inputs/{1BC8,2GFB,4GYT}.pdb` plus the ProteinMPNN clone's
     `inputs/**/*.pdb`;
   - parse with aminx and record L, chain count, ligand heavy-atom count and sha256;
   - admit proteins with L ≥ 64;
   - if fewer than 2 ligand-bearing structures result, fetch ≥ 2 PDB entries with bound small molecules
     ONCE into `outputs/browser_validation/fixtures/raw/`, sha256-pinned;
   - record per fixture and per v1 `k` (48, 32) the `near_tie_residues` via `compare.no_tie_mask`.
2. Add the tie-lattice fixture (`_lattice.py`, L = 96) and one L ≥ 400 fixture if available. For every
   multichain fixture, propose candidate tie groups (cross-chain homo-oligomer equivalents), compute
   `E_idx` with the aminx featuriser at noise 0 for k = 48 and k = 32, and record per fixture
   `tie_groups_knn_disjoint` (the qualifying groups) and `tie_groups_rejected` with the offending pairs
   (R2-C3). The manifest records the per-set count of qualifying multi-member groups; set B must reach
   the pre-registered floor of 2 across ≥ 2 fixtures, otherwise `build_corpus` admits further
   multichain candidates or exits 2 with the shortfall.
3. Split deterministically (sort by sha256, alternate) so each set has ≥ 1 multichain and ≥ 1
   ligand-bearing fixture, swapping the next eligible entry if needed. Write `manifest.json` with
   per-fixture sha256, set membership, each qualifying group's `designable_members`, and each path
   relative to a root token (`$REFERENCE_PATH`, `$PROTEINMPNN_PATH`, or the worktree) so the same
   manifest resolves on titanix; loaders re-check every sha256. **Commit** `manifest.json` and
   `raw/**` with `git add -f` (titanix receives inputs only through git; these are also on T11's frozen
   list).
4. `scripts/browser_validation/parse_parity.py` exposes `compare_parse(fixture) -> rows` (reused by T7).
   It compares aminx arrays with the reference `data_utils.parse_PDB` (`--extra benchmark`): integers
   EXACT, coords ≤ 1e-5 Å. Control: a 1e-3 Å atom shift in a temp copy under `outputs/` must be
   detected. Sidecar `parse_parity.bth.toml` (`calibration`, `novel = true`):
   - `ctrl_blind` (residual);
   - `pass`;
   - `fail` (residual).

**Files**: `scripts/browser_validation/{fixtures,parse_parity}.py`, `…/parse_parity.bth.toml`,
`outputs/browser_validation/fixtures/{manifest.json,raw/**}` (create; committed with `git add -f`).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
uv run --extra dev python scripts/browser_validation/fixtures.py --out outputs/browser_validation/fixtures
jq -e '([.fixtures[]|select(.set=="A").sha256]) as $a | ([.fixtures[]|select(.set=="B").sha256]) as $b | ($a|length>0) and ($b|length>0) and (($a - $b)|length == ($a|length)) and all(.fixtures[]|select(.kind=="protein"); .L>=64) and any(.fixtures[]; .kind=="tie_lattice") and all(.fixtures[]; has("near_tie_residues"))' outputs/browser_validation/fixtures/manifest.json
jq -e '([.fixtures[]|select(.set=="B")|select(any(.tie_groups_knn_disjoint[]?; (.designable_members|length) >= 2))]|length) >= 2 and ([.fixtures[]|select(.set=="B")|.tie_groups_knn_disjoint[]?|select((.designable_members|length) >= 2)]|length) >= 2' outputs/browser_validation/fixtures/manifest.json   # R2-C3/R3-C6 floor: >= 2 qualifying groups across >= 2 distinct set-B fixtures
test -z "$(git status --porcelain -- outputs/browser_validation/fixtures/manifest.json)" && git ls-files --error-unmatch outputs/browser_validation/fixtures/manifest.json
bth validate-sidecar scripts/browser_validation/parse_parity.bth.toml
REFERENCE_PATH=$REFERENCE_PATH bth run --campaign-id <layer-a-id> --output-paths outputs/browser_validation/layer_a/parse_parity.json -- uv run --extra dev --extra benchmark python scripts/browser_validation/parse_parity.py --out outputs/browser_validation/layer_a/parse_parity.json
bth compact
H=$(git rev-parse HEAD); test ${#H} -eq 40
bth sql "SELECT outcome FROM runs WHERE command LIKE '%parse_parity.py --out%' AND git_hash = '${H:?}' ORDER BY timestamp DESC LIMIT 1" | jq -e '.count == 1 and .rows[0][0] == "pass"'
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B scripts/browser_validation/fixtures.py scripts/browser_validation/parse_parity.py 2>&1 | tee outputs/browser_validation/ruff_t6.log; grep -L "No Python files found" outputs/browser_validation/ruff_t6.log
uv run --extra dev ruff format --check --config "exclude=[]" scripts/browser_validation/fixtures.py scripts/browser_validation/parse_parity.py
```
**Scope estimate**: ~330 LOC.

### T7 — Phase 1: exact-tier harness (calibrate → validate) with in-run controls

1. `scripts/browser_validation/layer_a_common.py`:
   - model loaders reused by import from `tests/parity/test_{full_model,soluble_membrane,
     sidechain_context,packer}_parity.py`, with an explicit `weight_source` and no default;
   - every `reference_utils` call wrapped so `pytest.skip.Exception` increments `n_skipped` (exit 4
     after writing);
   - `pins.assert_reference_pinned` called;
   - the reference-formula order built from a fixed randn with `mask*chain_mask`;
   - per-side `input_sha256`;
   - `provenance()` → `git_hash`, `git_clean` via `bathos.git.capture_git_state(cwd)` (source must be
     `'git'`, else exit 3; see the common context), captured once at start before any write (a calibrate
     later rewrites the tracked params file), and `prereg_sha256`;
   - `section_sha(params, section)` (canonical-JSON sha256; also a CLI: `layer_a_common.py section-sha
     <file> <section>`) and `prereq_check(stem, section)` per Bathos staging, calling `$BTH_BIN` (default
     `bth`) `sql` and `git merge-base --is-ancestor`;
   - reference roots from `$REFERENCE_PATH` and `$PROTEINMPNN_PATH`;
   - `differential_mode()`, which reads `BTH_DIFFERENTIAL_{KNOB,VALUE,PHASE}` per the Differential
     rules;
   - `emit(result, out)`, which skips `--out` whenever a differential phase is set.
2. `layer_a_exact.py` (engine) has one function per bars row. It covers:
   - P00;
   - P01;
   - P02 (via `parse_parity.compare_parse`);
   - P03 (EXACT on no-tie residues, counting `n_near_tie_excluded`; tie lattice recorded, not gated);
   - P04 and P05;
   - P06 pinned;
   - P09 scoring, restricted to k-NN-disjoint tie groups (R2-C3): the row re-checks disjointness at
     k = 48 and k = 32 from the fixture manifest and exits 2 on any violation;
   - P11 (context on/off);
   - P12;
   - P13 (random per-residue labels in {0,1,2} and both global labels);
   - P14 (shipped + pt_convert).

   Rows carry `{path, metric, value, bar, ratio, status, weight_source, fixture}`, and `status` comes
   from the params' headroom verdict. The in-process controls are:
   - the sized W_out-bias perturbation;
   - reversed order (P05/P06);
   - a tie-group flip (P09);
   - a mantissa bit-flip (P01);
   - a 1e-3 Å atom shift (P02);
   - a k-th-neighbour CA move (P03);
   - a packer weight perturbation (P14).

   In a differential phase, only `sentinel_ratio_to_bar` (P05 on the reduced subset) is computed, with
   the perturbation applied when the value is `"1"`.
3. `layer_a_exact_calibrate.py` (set A):
   - searches each sized control over {1e-4, 2e-4, 5e-4, 1e-3, 2e-3} for an effect in [2×, 10×] the
     bar;
   - applies the headroom rule (`not_advanced` per path) and tightens bars;
   - measures `sentinel_ratio_to_bar` off vs on **on the pre-registered reduced subset** (size
     recorded), records `exact.measured_half_effect` = half that on−off effect, asserts it is ≥ 1.0
     (failing otherwise), and sets `exact.min_effect = 1.0` exactly — PINNED, not derived (R2-C6), so
     the sidecar's hard-coded `min_effect = 1.0` matches and `--dry-run` equality holds;
   - measures per-draw/per-row cost and peak RSS on the machine it runs on and writes
     `exact.budget_wall_hours` and `exact.projected_peak_rss_gib` (fails above 16 h / 48 GiB);
   - writes the `exact` section (incl. `git_hash`) of `--params-out` (default
     `outputs/browser_validation/layer_a/preregistered_params.json`), records `params_section_sha256`
     and the `pt_convert` vs `eqx` numbers (AC-15);
   - `--smoke` runs one set-A fixture, P05 only, in < 60 s and REQUIRES a non-default `--params-out`
     (a scratch file the validate script never reads).
4. `layer_a_exact_validate.py` (set B):
   - refuses unless the params are committed and clean; runs `prereq_check('layer_a_exact', 'exact')`;
   - `--dry-run` checks imports and the manifest, plus that the sidecar `min_effect` > 0 and equals
     `exact.min_effect`;
   - `--smoke` runs one fixture, P05 only, in < 60 s, under a local `bth run` with the differential
     pre-flight LIVE (R2-C15), AFTER the params commit (R3-C9).

   Create both sidecars from the Overview.
5. `scripts/browser_validation/lint_sidecars.py`: every `scripts/browser_validation/*.bth.toml` has
   `is_residual = true` on every label other than `pass`/`partial`/`partial_headroom`/`blocked`/
   `tie_unstable`/`not_converted`; no calibration sidecar has `[differential]` or
   `claim_discriminates`; validation `min_effect > 0`, except the documented sampling placeholder
   before T8's params commit (flag `--allow-placeholder`, used ONLY in fixer lint gates, never in a
   smoke path).
6. Order (R3-C9): the orchestrator runs calibrate on titanix → commits the params → only then runs
   validate `--dry-run`, `--smoke` and the local `bth run` smoke → then the titanix validate.

**Files**: `scripts/browser_validation/{layer_a_common,layer_a_exact,layer_a_exact_calibrate,
layer_a_exact_validate,lint_sidecars}.py`, `…/{layer_a_exact_calibrate,layer_a_exact_validate}.bth.toml`
(create); `outputs/browser_validation/layer_a/preregistered_params.json` (calibration output, committed).
**Gate** (fixer):
```bash
cd "$WT" && set -euo pipefail
bth validate-sidecar scripts/browser_validation/layer_a_exact_calibrate.bth.toml && bth validate-sidecar scripts/browser_validation/layer_a_exact_validate.bth.toml
uv run --extra dev python scripts/browser_validation/lint_sidecars.py --allow-placeholder
bth lint
OMP_NUM_THREADS=4 uv run --extra dev --extra benchmark --with "$BATHOS_REQ" python scripts/browser_validation/layer_a_exact_calibrate.py --smoke --fixture-set A --params-out outputs/browser_validation/layer_a/params_smoke.json --out outputs/browser_validation/layer_a/exact_calibrate_smoke.json
jq -e '.exact.min_effect == 1.0 and (.exact.subset_size > 0) and (.exact.budget_wall_hours > 0) and (.exact.git_hash | length == 40)' outputs/browser_validation/layer_a/params_smoke.json
jq -e '(.params_section_sha256 | length == 64) and has("controls_sized") and has("params_written")' outputs/browser_validation/layer_a/exact_calibrate_smoke.json
uv run --extra dev --extra benchmark --with "$BATHOS_REQ" python scripts/browser_validation/layer_a_exact_validate.py --help
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B scripts/browser_validation/layer_a_*.py scripts/browser_validation/lint_sidecars.py 2>&1 | tee outputs/browser_validation/ruff_t7.log; grep -L "No Python files found" outputs/browser_validation/ruff_t7.log
uv run --extra dev ruff format --check --config "exclude=[]" scripts/browser_validation/layer_a_*.py scripts/browser_validation/lint_sidecars.py
```
**Gate** (orchestrator, after T4's provenance record; each titanix launch is followed by the O6′
step-5 sequence before the next line runs):
```bash
cd "$WT" && set -euo pipefail
P=outputs/browser_validation/layer_a/preregistered_params.json; L=outputs/browser_validation/layer_a/run_ledger.json
# (1) calibrate on titanix (set A), bring the params back, prove they are the pass row's, commit
bash scripts/browser_validation/titanix_launch.sh layer_a_exact_calibrate <layer-a-id> --fixture-set A
HC=$(jq -r '[.[]|select(.stem=="layer_a_exact_calibrate")][-1].H_v' $L); test ${#HC} -eq 40
cp outputs/browser_validation/titanix/layer_a_exact_calibrate-${HC:0:12}/layer_a/preregistered_params.json $P
SEC=$(uv run --extra dev python scripts/browser_validation/layer_a_common.py section-sha $P exact); test ${#SEC} -eq 64
bth sql "SELECT count(*) FROM runs WHERE command LIKE '%layer_a_exact_calibrate.py --fixture-set A%' AND git_hash = '${HC:?}' AND outcome = 'pass' AND json_extract_string(metadata,'\$.params_section_sha256') = '${SEC:?}'" | jq -e '.rows == [[1]]'
jq -e --arg hc "$HC" '.exact.min_effect == 1.0 and .exact.measured_half_effect >= 1.0 and .exact.subset_size > 0 and .exact.git_hash == $hc and .exact.budget_wall_hours <= 16 and .exact.projected_peak_rss_gib <= 48' $P
git add -f $P && git commit -m "prereg(layer-a): exact params from titanix calibrate ${HC:0:12}"
# (2) validate script exercised locally at the params commit (R3-C9, R2-C15)
uv run --extra dev python scripts/browser_validation/lint_sidecars.py
OMP_NUM_THREADS=4 uv run --extra dev --extra benchmark --with "$BATHOS_REQ" python scripts/browser_validation/layer_a_exact_validate.py --dry-run --out outputs/browser_validation/layer_a/exact_dryrun.json
OMP_NUM_THREADS=4 bth run --campaign-id <layer-a-id> --output-paths outputs/browser_validation/layer_a/exact_smoke_run.json -- uv run --extra dev --extra benchmark --with "$BATHOS_REQ" python scripts/browser_validation/layer_a_exact_validate.py --smoke --fixture-set A --out outputs/browser_validation/layer_a/exact_smoke_run.json
bth compact; H=$(git rev-parse HEAD); test ${#H} -eq 40
bth sql "SELECT count(*) FROM runs WHERE command LIKE '%layer_a_exact_validate.py --smoke%' AND git_hash = '${H:?}' AND differential_status = 'passed' AND json_extract_string(metadata,'\$.prereq_ok') = 'true'" | jq -e '.rows == [[1]]'   # also proves one catalog row per bth run, differential phases included
jq -e 'has("n_comparisons") and has("n_over_bar") and has("sentinel_ratio_to_bar") and has("controls_total") and has("git_hash") and has("prereg_sha256") and has("prereq_ok")' outputs/browser_validation/layer_a/exact_smoke_run.json
# (3) validate on titanix (set B) at H_v
bash scripts/browser_validation/titanix_launch.sh layer_a_exact_validate <layer-a-id> --fixture-set B
HV=$(jq -r '[.[]|select(.stem=="layer_a_exact_validate")][-1].H_v' $L); test ${#HV} -eq 40
git cat-file -e "$HV:$P" && git cat-file -e "$HV:uv.lock"
PS=$(git show "$HV:$P" | sha256sum | cut -d' ' -f1); LS=$(git show "$HV:uv.lock" | sha256sum | cut -d' ' -f1); test ${#PS} -eq 64 && test ${#LS} -eq 64
bth sql "SELECT count(*) FROM runs WHERE command LIKE '%layer_a_exact_validate.py --fixture-set B%' AND git_hash = '${HV:?}' AND json_extract_string(metadata,'\$.git_hash') = '${HV:?}' AND json_extract_string(metadata,'\$.prereg_sha256') = '${PS:?}' AND dependency_lock_sha256 = '${LS:?}' AND json_extract_string(metadata,'\$.git_clean') = 'true' AND json_extract_string(metadata,'\$.prereq_ok') = 'true' AND outcome IN ('pass', 'partial_headroom') AND differential_status = 'passed'" | jq -e '.rows == [[1]]'
```
**Scope estimate**: ~750 LOC.

### T8 — Phase 1: sampling harness (teacher-forced exact tier + IUT statistical tier)

1. `scripts/browser_validation/layer_a_sampling.py` (engine). The lanes are:
   - P07 (T ∈ {0.1, 1.0}, fixed positions on 20 % of residues);
   - P08 (the `run.py` omit formula for `omit_AA="CW"` plus one per-residue omit);
   - P09-s (restricted per the bars table; fixtures and group definitions checked, exit 2 on a
     violation — including **k-NN-disjointness of every same-group pair in EVERY tie group at k = 48 and
     k = 32**, recomputed with the aminx featuriser at noise 0, plus the non-vacuity floor
     `sampling.p09_qualifying_groups >= 2` on set B, below which the lane is `incomplete`, R2-C3);
   - P11-s.

   Lane temperatures are read from `sampling.lane_temperatures` (P07@0.1, P07@1.0, P08@1.0, P09-s@1.0,
   P11-s@1.0) and asserted against the committed params (R2-C11).

   Every lane applies the X-omit bias column (−1e8 at index 20) on the aminx arm.
   - **Order per draw i.** Generate `randn_i` (seeded). The reference runs at **batch 1 for P09-s only**
     (its symmetric branch uses `decoding_order[0]` for the whole batch, `model_utils.py:358`); P07, P08
     and P11-s run BATCHED with one `randn_i` row per draw, which the non-symmetric branch honours
     (`:262-263`) — a blanket batch 1 costs ~5× for no gain (R2-C11). Each draw's `randn_i` goes into
     its feature dict. The host computes `order_i = argsort((mask*chain_mask + 1e-4)*|randn_i|)` (for
     P09-s, group-flattened at first occurrence) and asserts it equals the reference's returned
     `decoding_order`. aminx runs `inference.sample_autoregressive.kernel` with
     `WaveScheduleBundle.from_tie_groups(tie_map, order_i)` (tie_map = `arange(L)` except P09-s),
     `fixed_mask = 1 − mask*chain_mask` and `fixed_tokens = S_true`.
   - **(a) Teacher-forced.** Feed each reference sequence and its order into aminx conditional decoding
     with the matching AR mask, and compare per-step log-probs (`tf_max_abs`). For P09-s the compared
     quantity is the FUSED per-group distribution and the sized fusion control runs in-process
     (P09-s restrictions, R3-C6), giving `p09_fused_tf_max_abs` and `p09_fusion_ctrl_detected`.
   - **(b) Statistical.** Each arm draws 2n per lane (A₁/A₂, R₁/R₂). Per lane compute recovery TOST
     (δ = 0.01) and `excess_js_upper` vs the committed m_ℓ; for P09-s both run over tie-group member
     positions only. A lane is equivalent iff both pass; `n_lanes_equiv` counts such lanes and
     `iut_equivalent` decides the claim.
   - Count omitted-AA and X tokens per arm.
2. Controls (20 replicates each, disjoint seeds, per-replicate n = `n_required`, statistic as in
   "Sampling statistics"):
   - positive: +β on alanine in the aminx arm; detected = not declared equivalent; ≥ 18 required;
   - negative: aminx vs aminx; false positive = not declared equivalent; ≤ 3 allowed.

   In a differential phase, only `main_js_vs_ref` on the reduced subset is computed, with β applied
   to the main aminx arm when the value is `"1"`.
3. Packer draw lane (P14): 10,000 chi draws from fixed real mixture parameters vs the analytic von Mises
   mixture CDF (KS). A concentration ×1.2 control, run over 20 replicates, must be rejected in ≥ 18;
   the rejection rate is recorded.
4. `layer_a_sampling_calibrate.py` (set A, n = 300 pilot) computes and records, under `sampling` in the
   params:
   - σ̂ and `n_required = max(1500, required_n(σ̂, 0.01))`, doubled until the null-replicate criterion
     holds;
   - m_ℓ per lane (the T×1.05 rule);
   - β (mean E ≥ 2·m_ℓ);
   - the reduced-subset size and `min_effect` = half the measured `main_js_vs_ref` on−off effect
     (> 0 required);
   - `lane_temperatures`, `p09_qualifying_groups`, `p09_tied_positions` (set B), `p09_fusion_ctrl_eps`,
     the per-fixture draw allocation, `budget_wall_hours` and `projected_peak_rss_gib` (Titanix budget;
     fails above 16 h / 48 GiB), and `git_hash`; plus `params_section_sha256` in its result. It extends
     the committed params file, leaving `exact` byte-identical in canonical form; `--smoke` requires a
     scratch `--params-out` as in T7.

   The calibrate sidecar has no `[differential]`.
5. `layer_a_sampling_validate.py` (set B) applies the committed-params refusal rule and
   `prereq_check('layer_a_sampling', 'sampling')`. `--dry-run` checks the sidecar `min_effect` > 0 and
   that it equals the params value; `--smoke` runs one fixture, one lane, n = 50, in < 60 s. The
   orchestrator sets the sidecar `min_effect` to the params value in the SAME commit as the params,
   before any smoke (R3-C9).

**Files**: `scripts/browser_validation/{layer_a_sampling,layer_a_sampling_calibrate,
layer_a_sampling_validate}.py`, `…/{layer_a_sampling_calibrate,layer_a_sampling_validate}.bth.toml`
(create); `preregistered_params.json` (extended).
**Gate** (fixer):
```bash
cd "$WT" && set -euo pipefail
bth validate-sidecar scripts/browser_validation/layer_a_sampling_calibrate.bth.toml && bth validate-sidecar scripts/browser_validation/layer_a_sampling_validate.bth.toml
uv run --extra dev python scripts/browser_validation/lint_sidecars.py --allow-placeholder
OMP_NUM_THREADS=4 uv run --extra dev --extra benchmark --with "$BATHOS_REQ" python scripts/browser_validation/layer_a_sampling_calibrate.py --smoke --fixture-set A --params-out outputs/browser_validation/layer_a/params_smoke_sampling.json --out outputs/browser_validation/layer_a/sampling_calibrate_smoke.json
jq -e '.sampling.min_effect > 0 and (.sampling.margins|length) == 5 and (.sampling.lane_temperatures|length) == 5 and (.sampling | has("p09_tied_positions")) and .sampling.p09_fusion_ctrl_eps > 0 and .sampling.budget_wall_hours > 0 and .sampling.beta > 0' outputs/browser_validation/layer_a/params_smoke_sampling.json
uv run --extra dev --extra benchmark --with "$BATHOS_REQ" python scripts/browser_validation/layer_a_sampling_validate.py --help
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B scripts/browser_validation/layer_a_sampling*.py 2>&1 | tee outputs/browser_validation/ruff_t8.log; grep -L "No Python files found" outputs/browser_validation/ruff_t8.log
uv run --extra dev ruff format --check --config "exclude=[]" scripts/browser_validation/layer_a_sampling*.py
```
**Gate** (orchestrator): the T7 orchestrator gate with `exact` → `sampling` and `layer_a_exact_*` →
`layer_a_sampling_*`, with these differences:
- step (1) additionally asserts `jq -S .exact` is identical between `git show HEAD:$P` and the fetched
  file, and `.sampling.min_effect > 0 and .sampling.n_required >= 1500 and .sampling.p09_qualifying_groups
  >= 2 and .sampling.p09_tied_positions > 0 and .sampling.budget_wall_hours <= 16 and
  .sampling.projected_peak_rss_gib <= 48`; the sidecar's `min_effect` is set to `.sampling.min_effect`
  and committed in the SAME commit as the params;
- step (2) `jq` requires `tf_max_abs`, `n_lanes_equiv`, `main_js_vs_ref`, `posctl_detected`,
  `p09_fusion_ctrl_detected`, `git_hash`, `prereg_sha256`, `prereq_ok`;
- step (3)'s count query requires `outcome = 'pass'` (not `partial_headroom`).
**Scope estimate**: ~650 LOC.

### T9 — Phase 1: existing-suite hardening (inventory sweep, matrix paths, measurement script)

1. `tests/parity/browser_validation_paths.json` maps each symbol to a P-ID or to `{"internal":
   "<reason>"}`. `tests/parity/browser_validation_internal_baseline.txt` holds the committed internal
   count. `tests/parity/test_browser_validation_inventory.py`:
   - AST-walks the public top-level `def`/`class` names of
     `src/aminx/{inference,sampling,scoring,model,host,tiling,ebm,potts}/**/*.py` plus
     `aminx.run.__all__`;
   - fails on unmapped symbols, stale mappings, empty reasons, and an internal count above the
     baseline;
   - has a control test that injects a fake public function and asserts it is reported.
2. Fix `tests/parity/parity_matrix.json` `code_paths`:
   - `run/scoring.py` → `scoring/score.py`;
   - `run/sampling.py` → `sampling/sample.py` + `inference/sample_autoregressive.py`;
   - `run/averaging.py` → `host/averaging.py`;
   - `model/multistate_sampling.py` → `sampling/multistate_poe.py`.

   Add a note to `autoregressive-sampling` and `tied-positions-and-multi-state`: "token agreement is
   non-discriminating under bias[...,0]=100; the discriminating gate is the teacher-forced lane
   (layer_a_sampling)". Create `tests/parity/test_parity_matrix_paths_exist.py`.
3. `scripts/measure_conditional_nll_parity.py`: move the module-level work into `main()`; replace
   `"bundled"` with a required `--weight-source {eqx,pt_convert}` passed to `_jax_protein_for_source`;
   print the source in the header.
4. `tests/parity/test_parity_matrix_spec.py` still passes.

**Files**: `tests/parity/{browser_validation_paths.json,browser_validation_internal_baseline.txt,
test_browser_validation_inventory.py,test_parity_matrix_paths_exist.py}` (create);
`tests/parity/parity_matrix.json`, `scripts/measure_conditional_nll_parity.py` (modify).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
uv run --extra dev pytest tests/parity/test_browser_validation_inventory.py tests/parity/test_parity_matrix_paths_exist.py tests/parity/test_parity_matrix_spec.py -q -rs --junitxml=outputs/browser_validation/junit_t9.xml
uv run --extra dev python scripts/browser_validation/assert_no_skips.py outputs/browser_validation/junit_t9.xml
jq -e '[.[]|objects|select(has("internal"))|.internal|length>0]|all' tests/parity/browser_validation_paths.json
uv run --extra dev python scripts/measure_conditional_nll_parity.py --help | grep -qF -- "--weight-source"
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B tests/parity/test_browser_validation_inventory.py tests/parity/test_parity_matrix_paths_exist.py scripts/measure_conditional_nll_parity.py 2>&1 | tee outputs/browser_validation/ruff_t9.log; grep -L "No Python files found" outputs/browser_validation/ruff_t9.log
uv run --extra dev ruff format --check --config "exclude=[]" tests/parity/test_browser_validation_inventory.py tests/parity/test_parity_matrix_paths_exist.py
```
**Scope estimate**: ~260 LOC + JSON.

### T10 — Phase 1: literature-parity protocol, phases 1–4 (orchestrator-driven; fixer prepares packets)

1. The fixer builds packets under `outputs/browser_validation/litparity/`:
   - `reference_packet/`: `$REFERENCE_PATH/{run.py,model_utils.py,data_utils.py,sc_utils.py,score.py}`
     at 26ec57ac, ProteinMPNN `protein_mpnn_utils.py`, and the PDFs;
   - `impl_packet/`: the `impl_paths`;
   - `README.md`, stating which packet each phase sees (Phase-1 reconstructors: `reference_packet/`
     ONLY).
2. Orchestrator, Phase 1: N = 3 reconstructors (math/algo/protocol) using the bathos templates
   (`/home/marielle/projects/bathos/agent_assets/skills/bathos-literature-parity/templates/`), writing
   `litparity/recon_*.md`. Ambiguities are recorded, not guessed.
3. Orchestrator, Phase 2: reconciliation into `litparity/clauses.json`, each clause `{element, source,
   code_location, verdict ∈ MATCH|DEVIATION|MISSING|AMBIGUOUS, core, notes, paths: [P-IDs]}`. Mandatory
   clauses:
   - default decoding order (`fixed_n_to_c` vs random + fixed-first);
   - temperature on `logits + bias`;
   - X hard-omit at sampling (aminx native default samples X: DEVIATION);
   - omit as `-1e8` bias;
   - `augment_eps`;
   - k clamp `min(k, L)`;
   - top-k tie-break;
   - membrane label encoding;
   - ligand atom-context number and cutoff;
   - side-chain-context toggle;
   - the score_only multi-order average;
   - packer chi sampling;
   - **tied-group fixed override semantics** (aminx `max(fixed tokens)` group override vs the
     reference's per-member loop inheritance);
   - **last-member bias** (reference adds only the last member's bias; aminx PoE applies every
     member's);
   - tied-group fusion (weighted raw-logit sum vs `sum(log_softmax)`, non-unit weights);
   - **intra-tie-group visibility** (R2-C3): the reference decodes group members sequentially so a later
     member sees an earlier member's updated state, while aminx makes same-group members mutually
     visible in one parallel pass — DEVIATION, validated only on k-NN-disjoint groups.

   **Core flags are pre-registered (O8).** Every clause carries `core` (bool) and, when
   `verdict = DEVIATION`, a written reconciliation note, BOTH committed (`git add -f`) before any parity
   run. `core` follows the rule in "Grade inputs" (R3-C11): only the six enumerated designed
   DEVIATIONs are `core = false`; every other clause touching an in-scope layer-(a) P-ID is `core =
   true`. `clauses.json`'s sha256 is written to
   `preregistered_params.json` as `litparity.clauses_sha256` at this point; T11 recomputes it and
   refuses to grade if it changed, so `core` cannot be reassigned after adjudication.
4. Orchestrator, Phase 3: M = 3 attackers (stats/hyper/struct, honesty tax) → `litparity/refute_*.md`.
   Phase 4: adjudication → `litparity/adjudication.json` (`{defects_confirmed: [{id, paths, severity}],
   adversarial_survived, ambiguity_load, n_ambiguous}`), with `ambiguity_load` computed by the R3-C5
   rule. A defect is confirmed on ≥ 2 votes or a runnable test; test-confirmed defects become T11
   invariant candidates.
5. The fixer writes `scripts/browser_validation/litparity_schema.py`: required keys, the verdict
   vocabulary, every mandatory clause present (including the intra-tie-group visibility clause), `core`
   present and boolean on every clause, a non-empty reconciliation note on every DEVIATION, `severity`
   on every confirmed defect, and `paths` on every clause and defect. It also ENFORCES two rules
   mechanically: `core == (paths ∩ IN_SCOPE_LAYER_A ≠ ∅ and element ∉ DESIGNED_DEVIATIONS)` on every
   clause (the six-item list from "Grade inputs", hard-coded), and `ambiguity_load`/`n_ambiguous` in
   `adjudication.json` equal the values recomputed from `clauses.json`. Its tests cover one violating
   `core` flag and one inconsistent `ambiguity_load`.

**Files**: `outputs/browser_validation/litparity/**` (create; `git add -f`), `scripts/browser_validation/
litparity_schema.py`, `tests/parity/test_litparity_schema.py` (create).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
uv run --extra dev python scripts/browser_validation/litparity_schema.py outputs/browser_validation/litparity
uv run --extra dev pytest tests/parity/test_litparity_schema.py -q --junitxml=outputs/browser_validation/junit_t10.xml && uv run --extra dev python scripts/browser_validation/assert_no_skips.py outputs/browser_validation/junit_t10.xml
test "$(cat outputs/browser_validation/litparity/recon_*.md | grep -cE "src/aminx|aminx\.(model|inference|sampling|scoring)")" = 0
ls outputs/browser_validation/litparity/recon_*.md | wc -l | grep -qx 3
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B scripts/browser_validation/litparity_schema.py 2>&1 | tee outputs/browser_validation/ruff_t10.log; grep -L "No Python files found" outputs/browser_validation/ruff_t10.log
uv run --extra dev ruff format --check --config "exclude=[]" scripts/browser_validation/litparity_schema.py
```
**Scope estimate**: ~140 LOC + agent dispatch (orchestrator).

### T11 — Phase 1: invariants (orchestrator lock), graded verdict, advance table, attestation, README

1. **ORCHESTRATOR-OWNED (not delegable).** The orchestrator writes
   `tests/parity/test_mpnn_reference_invariants.py` (marker `parity_heavy`) with ≥ 8 invariants, each
   run against BOTH the reference and aminx:
   1. log-prob at i is unchanged by mutating a residue decoded after i, and changed by mutating one
      decoded before;
   2. unconditional logits are sequence-invariant;
   3. the order's fixed-first rule holds;
   4. an omitted AA has p < 1e-30, and X is never sampled under the X-omit column;
   5. the one-step distribution equals numpy `softmax((l + b)/T)`;
   6. tied positions receive identical tokens, and fused logits match the reference's unit-weight
      zero-bias sum up to a constant;
   7. membrane label channels are live (> 10× bar);
   8. on a non-tied fixture, the neighbour set equals numpy brute-force k-NN on no-tie residues.

   Each is shown RED against a named injected defect (scratch commit, failure quoted, reverted), then
   GREEN with zero skips.
2. `scripts/browser_validation/parity_validate_mpnn.py`:
   - run `bth sync --pull --remote-name titanix` then `bth compact` (exit 5 if either fails; R2-C1);
   - **per validate stem, via the run ledger (R3-C3)** — the graded HEAD legitimately differs from each
     validate's commit (T8/T10/T11 commit afterwards), so identity is `H_v`, never HEAD: take the last
     ledger entry for `layer_a_exact_validate` and for `layer_a_sampling_validate`; require
     `git merge-base --is-ancestor H_v HEAD` and `git diff --quiet H_v HEAD -- <FROZEN>`; require the
     T7 step-(3) identity-form query (native and metadata `git_hash = H_v`, `prereg_sha256 =
     sha256(git show H_v:<params>)`, `dependency_lock_sha256 = sha256(git show H_v:uv.lock)`,
     `git_clean`, `prereq_ok`, `differential_status = 'passed'`) to return `[[1]]`, and read that row's
     outcome. Any failed check or missing entry is `prereq_status = 'missing'` (fail closed, exit 6).
     `FROZEN` = `src/ uv.lock tests/parity/test_{full_model,soluble_membrane,sidechain_context,
     packer}_parity.py scripts/browser_validation/{layer_a_common,layer_a_exact,layer_a_exact_validate,
     layer_a_sampling,layer_a_sampling_validate,fixtures,parse_parity,pins,_lattice}.py
     scripts/browser_validation/{reference_pins.json,titanix_run.sh}
     outputs/browser_validation/fixtures/manifest.json outputs/browser_validation/fixtures/raw/`
     (the engine imports the test loaders; set-B structures and the manifest are inputs). The params
     file and sidecars are excluded (checked via `git show H_v:`; T10 legitimately extends the params);
   - recompute `sha256(litparity/clauses.json)` and exit 7 unless it equals
     `preregistered_params.json:litparity.clauses_sha256` (O8);
   - apply the SINGLE mapping (O1): both `pass` → `prereq_status = 'ok'`, rung R1; exact
     `partial_headroom` with sampling `pass` → `'headroom'`, R2; any other combination (including
     missing) → `'failed'`/`'missing'`, `parity_grade = 'FAIL'`, result written, then exit 6;
   - run the invariant file via pytest `--junitxml` → `invariant_pass`, counts;
   - run `litparity_schema.py` (exit 7 on failure), then read `clauses.json` → `clause_parity_pct =
     |MATCH ∩ core| / |core|` (AMBIGUOUS excluded from both terms), `n_ambiguous` and the rule-computed
     `ambiguity_load` enum (R3-C5), and `adjudication.json` → `adversarial_survived = not
     any(severity == 'core')`, per the grade-inputs rules (R2-C8/O8);
   - otherwise call `bathos.parity.compute_grade(ParityEvidence(...))`;
   - write the schema fields plus `ceilings` and `metadata.parity_run_type = "literature_parity"`.

   Create the sidecar from the Overview (no `requires_pass_stem`).
3. `scripts/browser_validation/advance_check.py`:
   - builds `outputs/browser_validation/layer_a/advance_table.json`, where each in-scope P-ID has
     `advances` = true iff its AC-8 rows have status `validated` and are within bar, its AC-10/AC-11
     lanes (if any) are within bar and equivalent, and no confirmed defect in `adjudication.json` names
     it;
   - `advance_check.py <P-ID>` exits 0 iff the path advances (the Phase-2 harness calls this);
   - its test fixture has one advancing and one blocked path.
4. Run the parity script under `bth run --campaign-id <layer-a-id>`. Write the verdict doc (`docs add`,
   category `audits`) containing: summary, clause scorecard, defects, invariant red/green evidence, the
   per-path table (value/bar/ratio/status), the advance table, the statement that the global grade is
   expected PARTIAL, and a reproduction plan.
5. `bth compact`; `bth campaign attest-parity --campaign-id <id> --parity-run-id <rid>`; **commit the
   rewritten claim file immediately** (`attest-parity` edits a tracked file; leaving it dirty makes
   every later run's `git_clean` false, R2-C13); then `bth campaign conclude --campaign-id <id>
   --outcome-label <pass|partial> [--negative-check …]`.
6. `README.md`: replace the `:12` claim and the `:58-65` table with the per-path verdict table and a
   verdict-doc link; reconcile the `3870631` pin with `reference_pins.json`.

**Files**: `tests/parity/test_mpnn_reference_invariants.py` (orchestrator),
`scripts/browser_validation/{parity_validate_mpnn,advance_check}.py`,
`…/parity_validate_mpnn.bth.toml`, `tests/parity/test_advance_check.py`, the verdict doc (create);
`README.md` (modify).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
OMP_NUM_THREADS=4 REFERENCE_PATH=$REFERENCE_PATH uv run --extra dev pytest tests/parity/test_mpnn_reference_invariants.py -m parity_heavy -q -rs --junitxml=outputs/browser_validation/junit_invariants.xml
uv run --extra dev python scripts/browser_validation/assert_no_skips.py outputs/browser_validation/junit_invariants.xml
uv run --extra dev pytest tests/parity/test_advance_check.py -q
bth validate-sidecar scripts/browser_validation/parity_validate_mpnn.bth.toml && uv run --extra dev python scripts/browser_validation/lint_sidecars.py
OMP_NUM_THREADS=4 REFERENCE_PATH=$REFERENCE_PATH bth run --campaign-id <layer-a-id> --output-paths outputs/browser_validation/layer_a/parity_validate.json -- uv run --extra dev --extra benchmark --with "$BATHOS_REQ" python scripts/browser_validation/parity_validate_mpnn.py --out outputs/browser_validation/layer_a/parity_validate.json
jq -e '(.prereq_status | IN("ok", "headroom")) and (.ambiguity_load | IN("none", "non_load_bearing", "load_bearing")) and (.n_ambiguous >= 0)' outputs/browser_validation/layer_a/parity_validate.json
bth compact; H=$(git rev-parse HEAD); test ${#H} -eq 40
bth sql "SELECT outcome, parity_run_type FROM runs WHERE command LIKE '%parity_validate_mpnn.py%' AND git_hash = '${H:?}' ORDER BY timestamp DESC LIMIT 1" | jq -e '.count == 1 and (.rows[0][0] | IN("pass", "partial")) and .rows[0][1] == "literature_parity"'
bth campaign attest-parity --campaign-id <layer-a-id> --parity-run-id <parity-run-id> && git commit -am "chore: bind parity_run_id"
bth claim validate .bth/claims/aminx-bv-layer-a.claim.toml --catalog-dir "${BTH_CATALOG_DIR:?}" | jq -e '.ok == true and ((.infos // []) | any(test("^reference_parity controlled")))'
bth gate status mpnn_metric_synthetic_truth --guards src/aminx/parity/compare.py --guards tests/parity/test_compare_metrics.py | jq -e '.state == "GREEN"'
jq -e '[.paths[]|.advances]|length > 0' outputs/browser_validation/layer_a/advance_table.json
test "$(grep -cE "0\.999.*(all|five)" README.md)" = 0
uv run --extra dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B tests/parity/test_mpnn_reference_invariants.py tests/parity/test_advance_check.py scripts/browser_validation/parity_validate_mpnn.py scripts/browser_validation/advance_check.py 2>&1 | tee outputs/browser_validation/ruff_t11.log; grep -L "No Python files found" outputs/browser_validation/ruff_t11.log
uv run --extra dev ruff format --check --config "exclude=[]" tests/parity/test_mpnn_reference_invariants.py tests/parity/test_advance_check.py scripts/browser_validation/parity_validate_mpnn.py scripts/browser_validation/advance_check.py
```
**Scope estimate**: ~350 LOC tests + ~300 LOC scripts + doc.

**Task dependencies.**
- Phase 0: T1 → T2 → T3 (independent of Phase 1).
- T4 → {T6, T7, T8}; T4 step 5 (titanix provenance smoke committed) → every titanix launch.
- T5 (committed + stamped) → {T7, T8}.
- T6 → {T7, T8}, and T7 → T8 (T8 uses `lint_sidecars.py` and `layer_a_common.py`).
- T2's `_lattice.py` → T6.
- T9 and T10 are independent.
- {T7-validate, T8-validate, T10} → T11.
- **T7 and T8 are NOT split (R2-C15).** Their calibrate gates already execute the full engine (a stub
  cannot produce the asserted params), and the residual risk — validate-only code — is closed by the
  local `bth run` of each VALIDATE script in `--smoke` mode with the differential pre-flight live. A
  purely scheduling split (T7a = `layer_a_common` + rows + `lint_sidecars`; T7b = calibrate + validate +
  sidecars) is permitted but not required.

## Risks

| Risk | Mitigation |
|---|---|
| R1 A heavy parity run silently skips and reads green | Skips → `n_skipped` + non-zero exit; `incomplete` (residual); `assert_no_skips.py` on every junit gate |
| R2 The metric cannot see the claimed resolution | Max-abs + NLL gate, Pearson secondary; sized controls in [2×, 10×]; synthetic-truth gate guards `compare.py` AND its test file, stamped only after commit by a script |
| R3 Bars loosened after seeing data | Calibration only tightens; validation refuses uncommitted params; results carry `prereg_sha256`; held-out set B; claim registered first |
| R4 Sampled comparison underpowered, or non-significance read as equivalence | Per-lane IUT equivalence (TOST + excess-JS upper bound vs committed m_ℓ); n doubled until null replicates declare equivalence ≥ 18/20; controls with operational definitions; `underpowered` residual label |
| R5 An inert or self-satisfying differential | Knob read only via `BTH_DIFFERENTIAL_VALUE`; main-arm metric; `min_effect > 0` committed and linted; no calibrate differentials |
| R6 Titanix runs code other than the validated commit | O6′: per-run detached worktree of the pushed `H` in a real git checkout; `titanix_run.sh` exits 3 unless HEAD = `H` and porcelain-clean; `provenance()` requires the live-git channel; single-query catalog equality on native + metadata `git_hash`, prereg and lock hashes from `git show H_v:`; T11 ancestry + frozen-path diff |
| R7 Ruff gates vacuous or rewriting | `--no-fix`; `--config "exclude=[]"` + "No Python files found" non-vacuity check |
| R8 Memory or CPU exhaustion (local box, or titanix shared with other work) | Locally only `--dry-run`/`--smoke`, 4 threads; titanix: one run at a time, `taskset -c 0-15`, `MemoryMax=64G` on the transient systemd user unit (measured working 2026-09-24), calibrate-projected wall ≤ 16 h and peak RSS ≤ 48 GiB, else fewer set-B fixtures |
| R9 Reference drift between machines | Both machines' clones pinned by commit and checked by `assert_reference_pinned`; titanix `.pt` sha256s checked against `reference_pins.json` |
| R10 Timing noise (~5×) misread as a route difference | Interleaved routes + native arm in one session; ≥ 3 repeats; bootstrap CIs; `ClaimClass` |
| R11 Browser-compatibility over-claim | AC-24 regex gate with a planted claim; BLOCKED → no claim; ORT CPU is not browser evidence |
| R12 ONNX sort re-exposes tie-order dependence | Tie lattice on ORT-CPU (T2) and in-browser (T3); integers exact at every layer |
| R13 Literature-parity agents contaminated | Packet segregation; blindness grep; orchestrator invariant lock |
| R14 Matrix/README edits break parity-fast tests | T9 runs `test_parity_matrix_spec.py`; revert per-task commits |
| R15 Advisor needs CA-only or multi-state | ODQ-2 DECIDED out of scope (known limit; separate backlog item possible); ODQ-10 revisit trigger |
| R16 Positive-control clauses silently go uncontrolled | The drift check **fails OPEN on a NULL recorded hash** (`checker.py:245-257`, reached from `claim.py:817`), so presence is asserted as well as equality: every validate gate requires `dependency_lock_sha256 = sha256(git show H_v:uv.lock)` (NULL never equals) on the `git_hash = H_v` row; `uv.lock` is on T11's frozen list, so a later lock change forces re-running the validates |
| R20 Upstream bathos bugs met while planning (not fixed here) | **File upstream** (bathos backlog, `task_id 260923_aminx-browser-validation`): (1) `bth submit` default push / `--then-pull` call nonexistent `myxcel push-project` / `pull-project` (`cluster.py:78,:151`); (2) titanix's `projects.toml` is polluted by pytest temp catalogs. This spec avoids both by using `bth run` over ssh with an explicit `BTH_CATALOG_DIR` and `BTH_PROJECT_SLUG` |
| R21 Titanix `/` is 92 % full (~148 GB free) | Launch refuses below 30 GB free; free space recorded in the ledger before each sync; per-run worktree + venv removed after the ledger commit (uv hard-links from its cache) |
| R22 The `--with "$BATHOS_REQ"` overlay shadows a locked dependency | Provenance smoke asserts `locked_versions_match`; on failure `--with` is dropped and `provenance()` runs the live channel's git calls itself (T4 step 5) |
| R17 Differential triples titanix compute and can clobber `--out` | Reduced pre-registered subset in differential phases, `$BTH_RESULTS_PATH`-only writes (O4) |
| R18 The bitwise export contract drifts from the kernel | Wrappers import the kernel's own `logit_transform`/`_fuse_one_group`; AC-16's 5-key bitwise test against `sample_autoregressive.kernel` is the arbiter |
| R19 Global grade PARTIAL misread as blocking | Documented expectation; Phase 2 is gated per path by `advance_check.py` |

## References

- Design brief: `/home/marielle/.praxia/design/260923_aminx-browser-validation/00_design_brief.md`;
  recon `…/recon_inference_paths_v2.md`, `…/recon_export_browser.md`; research `…/research_external.md`;
  round records `…/rounds/r{1,2}_{challenger,defender,oracle}.json`, `…/rounds/r3_{challenger,defender,
  claude}.json` (round-3 oracle: ESCALATE on venue)
- Titanix facts measured by the orchestrator 2026-09-24 (host, cores, RAM, disk, uv/bth paths, bathos
  uv tool at `84be544e`, native-git `bth run` provenance in a real clone: `git_hash = a5ced9c7`,
  `git_dirty = False`, lock hash set)
- cisternal (`/home/marielle/projects/cisternal/src/cisternal/`): `provenance/channels.py:155-216`
  (channel precedence), `telemetry/git_state.py:215-234` (live channel; `dirty` = porcelain non-empty)
- aminx origin/main a5ced9c7 (v0.2.0a1) snapshot `…/aminx_origin_main/`:
  - `src/aminx/inference/sample_autoregressive.py:57-119`;
    `src/aminx/inference/decode/autoregressive.py:57-75,238-245,254-262,287-292,353-413`;
    `src/aminx/inference/decode/factory.py:119`; `src/aminx/inference/logits.py:396-435,502-531`;
    `src/aminx/types/bundles.py:212-252`; `src/aminx/io/weights.py:65`
  - `src/aminx/utils/decoding_order.py:59,67,75,100-105`; `src/aminx/inference/bundle_builder.py:208-262`;
    `src/aminx/inference/schedule_selector.py:338,352-367`
  - `tests/parity/{parity_matrix.json, reference_utils.py:25-26,87-130, test_full_model_parity.py:35-36,76-118,300}`;
    `scripts/measure_conditional_nll_parity.py:56`; `pyproject.toml`; `.bth.toml`
- Reference: dauparas/LigandMPNN@26ec57ac976ade5379920dbd43c7f97a91cf82de
  (`/home/marielle/projects/aminx/reference_ligandmpnn_clone`: `run.py:405-407`;
  `model_utils.py:200-220,317-323,351-460`); dauparas/ProteinMPNN (pin recorded in T4)
- Papers: Dauparas et al., Science 2022, doi:10.1126/science.add2187; LigandMPNN (DOI verified in T4)
- xtrax (`ring-probes` worktree): `src/xtrax/export/{safety.py:250-331,432-543, targets.py,
  compile.py:117-277, parity.py, divergence.py, rings.py}`; `.praxia/docs/specs/260910_webgpu-export-route.md`
  (:94-97); `.praxia/docs/specs/260911_export-divergence-mapping.md`;
  `.praxia/docs/research/260914_browser-inference-routes-jaxjs-jax2onnx.md`
- bathos (`/home/marielle/projects/bathos/src/bathos/`):
  - `runner.py:194-270,317-341,770-793`;
  - `sync.py:54-110,258-283` and `cluster_catalog.py:21-52` (remote catalog = `{remote_root}/.bth/catalog`);
  - `git.py` (re-exports cisternal `capture_git_state`);
  - `claim.py:152-164,311-443,745-901,904-1060`;
  - `parity.py:22-122`;
  - `sidecar.py:43-80,549-584,679-695`;
  - `prereg.py:302-336`;
  - `cli_render.py:35-49`;
  - `cli_cyclopts.py:648-765`;
  - `mcp.py:420-442,1543-1583,1747-1792,1819-1843,1970-2044,3096-3154,3224-3311`;
  - `authoring/models.py:85-199`;
  - `config.py:86-107`;
  - skills `using-bathos`, `bathos-literature-parity`, `bathos-rigor-gates`, `bathos-campaigns`.
- Global rules: `~/.claude/rules/{BATHOS.md, local-compute-limits.md, ephemeral_scripts.md}`

## Revision log

- r1: addressed C1-C28, O1-O5
- r2: addressed R2-C1-R2-C15, O6-O8
- r3: user chose titanix (2026-09-24); Engaging path removed; addressed R3-C1..C11, O9, O10
