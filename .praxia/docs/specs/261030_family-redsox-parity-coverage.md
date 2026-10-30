---
title: "Redsox and bathos parity coverage for the remaining MPNN families, and the stock-MPNN parity fix"
description: "Scope for extending the redsox knob gate and bathos parity vehicles to ThermoMPNN, FAMPNN, Caliby and the Tier 1/2 variants, why the ledger wave is slow, and the stock ProteinMPNN/LigandMPNN fix order."
task_id: 261030_mpnn-parity-scope
status: draft
created: 261030
---

# Redsox and bathos parity coverage, remaining families, and the stock-MPNN fix

Measured against `origin/main` `ef8810c2` (2026-10-30). Assumptions are tagged **[verified]**
(read at file:line or measured) or **[unverified]** (inferred, to be probed before a vehicle is
built on it). Nothing here was run on titanix; no parity vehicle or gate was re-run.

## 1. Where coverage stands

| Family | Upstream (local) | Redsox knob rows | Bathos parity vehicle | Other parity tests |
|---|---|---|---|---|
| PottsMPNN | `~/repos/PottsMPNN` | 48 live, 111 excluded | 5 (`potts_*`) | in-gate |
| LASErMPNN | `~/repos/LASErMPNN` | 163 live, 70 excluded | 4 (`laser_*`) | in-gate |
| ProtonPottsMPNN | `~/repos/ProtonPottsMPNN` | 26 live, 22 excluded | 1 (13 waves, 99 controls) | in-gate |
| ThermoMPNN / -D | `~/repos/ThermoMPNN` `370f76e`, `ThermoMPNN-D` `df9a75a` | none | `scripts/parity/thermo_mpnn_parity` exists, **not in manifest or ledger** | 11 tests vs upstream equations |
| FAMPNN | `~/repos/fampnn` `aaf788b` | none | none | 10 tests that restate upstream math in torch |
| Caliby | `~/repos/caliby` `41d3156` | none | none | 19 tests, same pattern |
| AbMPNN, HyperMPNN, CyclicMPNN, CA-only | `~/repos/{AbMPNN,HyperMPNN,CyclicMPNN,ProteinMPNN}` | none | none | graded f64 1e-7 / f32 1e-4 test, **skips when checkpoints are absent** |
| AntiBMPNN | `~/repos/AntiBMPNN` `7e6329b` | none | none | unit tests against the README only |
| Stock ProteinMPNN / LigandMPNN / Soluble / Membrane | `~/repos/ProteinMPNN`, `LigandMPNN` | none (a RunSpec knob matrix script exists) | Layer A `parity_validate_mpnn` graded **FAIL** | 43 `parity_heavy` tests, green in CI |

**[verified]** the gate's reference surfaces (`tests/knob_gate/reference_surfaces.py`) name only
the Potts, LASEr and ProtonPotts upstreams, and `scripts/redsox/extract_upstream_knobs.py` is
hand-coded per upstream (`_PROTONPOTTS_ENGINE`, `_PROTONPOTTS_DATACLASSES`, a LASEr checkpoint
block). **[verified]** Caliby, FAMPNN and Thermo take options through `CalibyOptions`,
`FAMPNNOptions` and `ThermoMPNNOptions` (`src/aminx/run/options.py:149,183,221`), 10, 9 and 6
fields respectively.

## 2. Why the ledger wave is slow

Measured or read, not profiled. **I did not find a recorded wall-time for the 261013 R6
re-wave**, so no total is claimed.

1. **Every negative control is a full re-run of the arm.** The clean arm and each mutant arm are
   separate subprocesses (`laser_proofread_parity.py:30`). The nine vehicles carry 118 controls,
   99 of them on `protonpotts_parity` (13 waves in one row). **[verified]**
2. **The upstream side is CPU torch.** The oracle runs in a CPU-only venv
   (`launch_wave.sh`, sidecars: "torch in the CPU-only oracle venv"). GPUs only help the aminx
   side. **[verified]**
3. **The aminx side used to be CPU too.** One clean arm of `laser_proofread_parity` took 54 min on
   CPU and 1052 s on GPU (spike 261013, `launch_wave.sh` comment). The four LASEr rows were
   quoted at 7.75 h of compute before that move (`README.md:22`). **[verified]**, but the current
   post-GPU total is **[unverified]**.
4. **Largest single vehicle:** `potts_ddg_megascale`, 202,804 rows over 371 per-PDB units per arm,
   about 25 min at `e3bc540e` (`README.md`). **[verified]**
5. **Serialised groups.** Four launch groups run one after another because five concurrent arms
   put load at 32-37 on 20 cores (`README.md`). **[verified]**
6. **Then the gate:** the first green gate run took 50.8 min over 1048 ids (`README.md`, run
   `e4bc0904`). **[verified]**
7. **Invalidation is coarse where it matters.** `pyproject.toml`, `uv.lock` and any file in every
   row's import closure (`src/aminx/__init__.py`, `host/runner.py`, the shared model packages)
   invalidate all rows; `--stale-only` cannot help then. Main has all three changed since the
   ledger was written, so all nine rows are stale. **[verified]**

Levers, each **[unverified]** until measured under the production-shape profiling rule:

- Share one oracle output across the clean arm and its mutant arms, if mutants only perturb the
  aminx side. Check whether the oracle worker is invoked once per arm
  (`laser_proofread_parity.py:642`) before assuming a saving.
- Run the oracle arms of different vehicles concurrently on the idle cores, and the aminx arms
  across GPUs 2 and 3, instead of serial groups.
- For new vehicles: profile one unit at production shapes and put the wall-time estimate in the
  sidecar before any run, as `rules/vectorize.md` requires.

## 3. Coverage plan for the remaining families

Every new vehicle follows the existing contract: tracked script, committed `.bth.toml` with
`[outcomes]` **before** the run, `--mutants` in argv equal to its manifest rows, `branch_controls.json`
under `<slug>/<sha8>/`, upstream checkpoints resolved through
`tests/knob_gate/checkpoint_registry.json`, preemption-safe per-unit stamps.

**Stochastic parts use injected randomness, not seeds.** The ProtonPotts vehicles already do this
(controls such as `uniforms_shifted_by_one`, `cdf_in_aminx_order`), which keeps the comparison
deterministic. FAMPNN's masked-autoregressive steps and the EDM sidechain samplers in FAMPNN and
Caliby need the same treatment. **[unverified]** that the upstream samplers accept externally
supplied noise without patching; probe first (spike, throwaway).

### 3.1 ThermoMPNN / ThermoMPNN-D (smallest; do first)

- The vehicle exists with f64 1e-7 / f32 1e-4 bands and four in-script negative controls. Gaps:
  it carries a `[requires.waiver]`, its controls are not in `branch_manifest.toml` format, and
  there is no ledger row or alias map. **[verified]**
- Work: adopt the four controls as manifest rows, add the `--controls-out` contract, extract the
  ThermoMPNN knob surface (single/double mode, distance threshold, ensemble, dtype), write alias
  rows, and decide whether `dtype = float16` (the default) needs its own band. **[unverified]**
  that float16 has any graded comparison today.

### 3.2 Tier 1/2 variants (AbMPNN, HyperMPNN, CyclicMPNN, CA-only)

- A graded test exists (`tests/parity/test_mpnn_variants_tier1_tier2.py`) but every case
  `pytest.skip`s when the checkpoint is missing (`:153,475,550,661`), so on a runner without
  `~/repos` it proves nothing and reports green. **[verified]**
- These share the stock model code, so they inherit the stock defects in section 4. Seal their
  vehicles **after** the stock fix lands.
- Work: one vehicle per tier wrapping the existing assertions, with a hard failure (not a skip)
  on a missing checkpoint under `bth run`; CA-only chain-break and masked-residue cases kept.

### 3.3 FAMPNN and Caliby

- Existing tests rebuild "upstream-keyed" state dicts from the aminx model and restate the EDM
  denoiser in torch, so they check internal consistency and conversion round-trips, not upstream
  output. **[verified]** (`test_fampnn_parity.py:284,470,1000`; `test_caliby_parity.py:483,984`).
- Work per family: a sealed oracle dump from the upstream torch code (same pattern as
  `aminx-oracles`, content hashes pinned in the spec), then graded stages: features, encoder,
  decoder or Potts parameters, one sequence step, one sidechain sampler step with injected noise.
- Knob surface: FAMPNN `configs/{pack,score,seq_design}.yaml` (the extractor already reads
  example YAML for Potts); Caliby `caliby/api.py` and `caliby/configs/`. **[unverified]** which
  keys are read by code rather than only declared.

### 3.4 AntiBMPNN

- Only unit tests against the README (`tests/model/test_antibmpnn_and_cdr_bias.py`). Needs an
  oracle for the CDR region masks, the log-odds logit bias and the Barley descriptors against
  `~/repos/AntiBMPNN`.

### 3.5 Gate integration (shared by all)

1. Make `extract_upstream_knobs.py` table-driven so a new upstream is a config row, not new code;
   regenerate `reference_surfaces.py` with the new commit pins.
2. Alias rows: every live row names a `test_knob_semantics_*` test in `tests/knob_semantics/`;
   anything out of scope is excluded with `deferred:<debt id>`, and the debt must exist first.
3. Add each vehicle to `MUT` and to a new launch group in `launch_wave.sh`, and its slug to
   `closure_edges.toml`.
4. The ledger rewrite follows one wave at a frozen tree, as today.

## 4. Stock ProteinMPNN / LigandMPNN parity fix

The 260926 verdict (`.praxia/docs/audits/260926_mpnn-reference-parity-verdict.md`, grade FAIL at
`e4b86a0c`) is out of date in the code but not in the record: `outputs/browser_validation/layer_a/
parity_validate.json` still reads FAIL, and nothing has re-graded. Each defect below was re-read
at `ef8810c2`; a subagent's first pass was cross-checked line by line.

| # | Defect | Status at `ef8810c2` | Evidence |
|---|---|---|---|
| 1 | `ligand_atom_context` (16 vs 25) | Fixed in source | `io/weights.py:216-227` parses 25 for `ligandmpnn_v_32_*_25`; `:542` passes it to the model; `tests/model/test_ligand_topology_windows.py` |
| 2 | `membrane_label_encoding` (random-init `physics_projection.bias`) | Fixed in source; **shipped weights unchecked** | `model/encoder.py:469-473` `use_bias=False`; `tests/model/test_bias_slots_match_reference.py` |
| 3 | `ligand_context_fusion_bias` (`v_c`) | Fixed in source; shipped weights unchecked | `model/ligand_mpnn.py:234` |
| 4 | `positional_encoding_ligand` (dropped real bias) | Fixed | `model/ligand_features.py:120` `use_bias=True`, shape (16,) |
| 5 | `weight_conversion_bias_handling` | Fixed at the 3 known sites; **14-vs-12 count not redone** | converter refuses one-sided biases (`scripts/convert_weights.py`) |
| 6 | `checkpoint_topology_source` (membrane suffix) | Fixed | suffix parse runs only for `ligand`/`packer` (`weights.py:221`) |
| 7 | `conditional_score_context_mask` (2-hop self-identity leak, up to 0.84 nats) | **Undetermined, likely open** | `1 - I` is still the documented default (`autoregression.py:413`, `sampling/conditional_logits.py:241`); no test mentions the leak |
| 8 | `sample_log_probs_output` (fixed positions return full rows, reference gives 0) | **Undetermined** | no `sample_log_probs` symbol on main; where the output now lives was not located |
| 9 | `ar_mask` rank-vs-order | Call sites fixed, **hazard remains** | `ste.py`, `optimize_ste.py`, `score.py` use `ar_mask_from_decoding_order`; `generate_ar_mask` (`autoregression.py:142`) still reads rank when untied and order when tied, documented only. `export/wrappers.py:359` always passes a tie map, so it takes the tied (order) branch, but no test pins that |

Four MISSING core clauses (`ligand_cutoff_for_score`, `membrane_label_construction`,
`packer_chi_sampling`, `score_only_multi_order_average`) were **not** re-checked.

**Fix scope, in order:**

1. **Settle what is still open before writing any fix.** Re-run the adjudication probes for 7, 8
   and 9 (`outputs/browser_validation/litparity/`, plus
   `tests/parity/test_mpnn_reference_invariants.py`) at `ef8810c2`. The evidence for 7 is
   a stats attacker's single vote, and the reference's own `1 - I` behaviour was never
   compared at the mechanism level, so 7 may be a reference property rather than an aminx defect.
2. **Check the shipped weights for 2 and 3.** The source fix does not fix a published `.eqx` that
   already carries a random-init bias (`HF_REVISION` is `5ff5cf16`, `weights.py:50`). Load the
   pinned revision and compare each bias slot against zero. Cheap, and it decides whether an HF
   re-upload is owed (`project_bias-slot-fix-needs-hf-reupload`).
3. **Retire the `generate_ar_mask` rank/order hazard** (9): make it delegate to
   `ar_mask_from_decoding_order`, or rename it and fail on the untied path. Add a test pinning
   `export/wrappers.py:p07_bundle`.
4. **Implement or formally waive the four MISSING clauses**, each with a pre-registered probe.
5. **Re-grade once:** re-run `parity_validate_mpnn` under a new sidecar. The FAIL floor
   `adversarial_survived = false` clears only if no confirmed defect survives, and the sampling
   prerequisite (O1, `prereq_status = missing`) still forces FAIL until a
   `layer_a_sampling_validate` exists, which the 483 h projection (D10) blocked.

Items 3 and 4 edit shared model code, so they invalidate every redsox row: they belong before the
re-wave in section 5.

## 5. Order of work

1. **Stock fixes first.** They change files in every row's closure, so every redsox row goes stale
   again; do them before the re-wave, not after.
2. ThermoMPNN adoption (no stock dependency).
3. Re-seal Tier 1/2 vehicles after the stock fix.
4. FAMPNN, Caliby, AntiBMPNN oracles and vehicles.
5. One wave, one gate run, one ledger rewrite covering old and new rows.

## 6. Decisions needed

- Do new families enter the existing redsox gate as additional rows, or get a second gate so the
  Potts/LASEr/ProtonPotts ledger is not invalidated by their churn?
- Is the stock-path FAIL grade re-graded as a whole (re-run `parity_validate_mpnn`), or closed
  defect by defect?
