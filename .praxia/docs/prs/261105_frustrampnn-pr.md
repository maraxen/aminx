---
title: "PR Specification: Implement FrustraMPNN to family parity standard (debt #3222)"
task_id: "261105_frustrampnn"
epic_id: 3222
branch: "wt/261105-frustrampnn"
target_branch: "main"
pr: "https://github.com/maraxen/aminx/pull/271"
specification: ".praxia/docs/specs/261105_frustrampnn-family-port-spec.md"
date: "2026-11-09"
head: "8d2b306c"
status: "draft_pr"
---

# feat(families): implement FrustraMPNN to family parity standard (debt #3222)

## Summary

Implements FrustraMPNN (`RosettaCommons/frustraMPNN` pinned at commit `3a03cdc300bfe24c4bb70e60207118532bc73b3b`) in `aminx` (`maraxen/aminx`) to the repository's strict family-parity standard on isolated branch `wt/261105-frustrampnn` (debt `#3222`).

This PR completes the full adversarial code review cycle, concurrent-`main` merge-order analysis, defect remediation, independent forensic audit, de-tracking of large weight binaries, elimination of facade branches, RedSox closure isolation, and certified 15/15 `[PASS]` wave verification under Bathos run `29fae936-de0e-4ebf-89eb-72612e24a144`.

**PR Status**: DRAFT (Awaiting final confirmation gate before landing on `main`).

---

## Adversarial Review Swarm Approvals (5/5)

A parallel adversarial swarm of 5 read-only reviewers audited the entire implementation, with all reviewers issuing unanimous **APPROVE** verdicts:

1. **Reviewer 1 — Concurrent `main` & Sibling PR Merge-Order Reviewer (`reviewer_r1_1`)**: **APPROVE**
   - Simulated 3-way merges against `gh/main` (`86042c0e`), PR #269 (`gh/wt/261104-p4-integrate`), and PR #276 (`gh/fix/inventory-pack-shim-plan-eda`).
   - Confirmed PR #271 has zero textual or semantic conflicts with `main` (`git merge-tree` exit 0, identical tree `5da5105b`).
   - Proved PR #271 must merge first before PR #269 to protect Bathos closure validity.
2. **Reviewer 2 — Oracle, Model, Converter & Driver Reviewer (`reviewer_r1_2`)**: **APPROVE**
   - Verified virtual-$C_\beta$ geometry and 25 RBF pairs against upstream `features.py:293-330`.
   - Verified LayerNorm $\epsilon = 10^{-5}$ equivalence.
   - Verified `LightAttention1D` singleton Conv1d slice reduction ($W[:, :, 4] \cdot x + b$) and Softmax identity collapse.
   - Verified 3-layer `BothOutMLP` ($512 \to 64 \to 32 \to 21$) and scalar `frustration_out` head.
   - Verified `subtract_mut` mutant-minus-wildtype logic.
   - Verified 130/130 strict parameter tensor consumption (0 unmapped keys, 0 unconsumed keys, 0 random leaves).
3. **Reviewer 3 — Parity Vehicle, Sidecar & Mutants Reviewer (`reviewer_r1_3`)**: **APPROVE**
   - Verified anti-self-trust safeguards: fail-loud on missing stems, digest mismatch, or missing checkpoint registry entries.
   - Verified genuine coordinate diffing on `feat/X` and `f64_feat/X` (`s0_prep = 0.0`).
   - Verified real public API `runner.score` invocation on `s8_entry`.
   - Verified dual-precision tolerance bands: clean $f_{64} = 1.854 \times 10^{-14} \ll 10^{-7}$, clean $f_{32} = 3.564 \times 10^{-5} < 10^{-4}$.
   - Verified 4/4 mutants killed strictly on finite over-band diffs ($1.908$ to $6.659$) with $n_{\text{crashed}} == 0$.
4. **Reviewer 4 — Knobs, Deferrals, Host Plumbing & Gate Wiring Reviewer (`reviewer_r1_4`)**: **APPROVE**
   - Verified all 16 upstream knobs in `FrustraModelKnobs` against upstream CLI and model definitions.
   - Confirmed bijectivity of `FrustraMPNNOptions` in `test_knob_superset.py`.
   - Verified deferral debt `#3222` tracking in `tests/knob_gate/DEFERRED_IDS.txt` (ESM-2 co-embeddings).
   - Verified Requirement G compliance: `tests/knob_gate/ledgers/frustrampnn.toml` maintains an empty `[sidecar]` table on feature branch.
5. **Reviewer 5 — Shared Code, CI Shards & RedSox Blast-Radius Reviewer (`reviewer_r1_5`)**: **APPROVE**
   - Audited all 36 modified files outside `src/aminx/families/frustra_mpnn/`.
   - Verified 0 sibling family ledger rows invalidated via `scripts/redsox/stale_rows.py` and `tests/knob_gate/_closure.py`.
   - Confirmed selective test attribution: changing `frustra_mpnn/driver.py` isolates to FrustraMPNN tests only (< 40 tests).
   - Inspected GitHub Actions CI across all check legs: all 7 checks GREEN.

---

## Concurrent `main` & Merge-Order Integration Analysis

### 1. Merge Order Recommendation: PR #271 Merges BEFORE PR #269
- **Topology Assessment**:
  - `wt/261105-frustrampnn` (PR #271) already incorporates `gh/main` (`86042c0e`) via merge commit `7f49f01a`.
  - Running `git merge-tree --write-tree gh/main HEAD` confirms 0 conflicts and identical trees (`5da5105b4ebe177d32e06d6b35670fcb6334f96b`).
- **Closure Protection**:
  - Graded Bathos run `a7979816-ae10-413d-9ed7-c7c358247ef6` is anchored at commit `4f4ff00a`.
  - `frustrampnn_parity`'s import closure comprises 133 files, crucially including `scripts/parity/artifact_key.py` and `src/aminx/run/options.py`.
  - Sibling PR #269 (`gh/wt/261104-p4-integrate`) modifies both `scripts/parity/artifact_key.py` and `src/aminx/run/options.py`.
  - If PR #269 merges into `main` first, rebasing PR #271 would introduce diffs in `artifact_key.py` and `options.py` between the graded anchor (`4f4ff00a`) and HEAD.
  - In `scripts/redsox/verify_wave.py`, `row_is_stale` would trigger `[FAIL] no path in the row's closure since`, unnecessarily invalidating the graded parity run.
  - Therefore, **PR #271 should land first on `main`**. PR #269 can subsequently rebase and resolve shared surface additions cleanly.

### 2. Sibling PR #276 Resolution (`browser_validation_paths.json`)
- Sibling PR #276 (`fix/inventory-pack-shim-plan-eda`) was submitted to resolve `pack_shim.py` and `plan_eda.py` symbol inventory mappings from PR #270.
- PR #271 already cleanly maps these symbols to stage `"P28"` (commit `3a462ee4`).
- Merge simulations identified that landing PR #276 without key consolidation creates duplicate JSON keys for the 18 `pack_shim`/`plan_eda` entries. Landing PR #271 first provides the canonical `"P28"` resolution, allowing PR #276 to be closed or superseded.

---

## Forensic Audit Verdict: CLEAN

An independent forensic audit conducted at HEAD (`8d2b306c`) certified the implementation as **CLEAN**:
- **130/130 Bit-for-Bit Tensor Consumption**:
  - Raw head checkpoint (`megascale_default_la-true_esm-False_split1.ckpt`): 12 parameter tensors.
  - Raw trunk checkpoint (`v_48_020.pt`): 118 parameter tensors.
  - Total: 130 tensors consumed with **0 unmapped keys**, **0 unconsumed keys**, and **0 random leaves**.
  - Direct tensor comparison against raw PyTorch source checkpoints: **maximum absolute difference = 0.000000e+00** across all 130 tensors.
- **Facade Elimination & Authentic Diffing**:
  - Zero hardcoded results or bypass fallback branches.
  - `s0_prep` performs authentic coordinate diffing against sealed oracle `feat/X` and `f64_feat/X`, evaluating to exact `0.0`.
  - `s8_entry` invokes real `aminx.host.runner.score(inputs="tests/data/1ubq.pdb", model_family="frustrampnn", output_kind="frustration")`.
- **Mutant Sensitivity**:
  - All 4 mutants (`reversed_hidden`, `refinement_bypass`, `permuted_output_weights`, `subtract_mut_inverted`) fail strictly on finite over-band diffs ($1.908$ to $6.659$).
  - `n_crashed == 0` (no mutant triggers exceptions or NaNs).
- **Repository Cleanliness**:
  - Untracked 63+ MB of binary weights (`megascale_default_la-true_esm-False_split1.ckpt` [57 MB] and `v_48_020.pt` [6.7 MB]) from git index.
  - Added `*.ckpt` to `.gitignore`.
  - Multi-path dynamic weight resolution in driver, test, and parity vehicle.

---

## Graded Bathos Run & RedSox Wave Verification

- **Authoritative Graded Run**: `29fae936-de0e-4ebf-89eb-72612e24a144`
  - Status: `completed`
  - Outcome: `pass`
  - Git Dirty: `false`
  - Git Hash: `a23bcd55`
  - Sidecar Mode: `declared`
  - Sidecar SHA-256: `6e20fe9c4db4c191095323e36c932bd8e1770878fbe5988026f12bb03c63e045`
- **Catalog Artifact**:
  - Exported cool-tier Parquet: `/tmp/bth_home/.bth/catalog/runs/aminx/run_29fae936-de0e-4ebf-89eb-72612e24a144.parquet`.
- **Wave Verification Output (`verify_wave.py`)**:
  ```
  === frustrampnn_parity  run 29fae936-de0e-4ebf-89eb-72612e24a144
      [PASS] status == completed                'completed'
      [PASS] outcome == pass                    'pass'
      [PASS] git_dirty is False                 False
      [PASS] git_hash present                   a23bcd55
      [PASS] ancestor of HEAD                   
      [PASS] no path in the row's closure since 
      [PASS] sidecar_sha256 matches             
      [PASS] --mutants == manifest rows         
      [PASS] output path well-formed            
      [PASS] loaded files inside closure        624 loaded
      [PASS] controls clean == pass             'pass'
      [PASS] controls mutants == rows           
      [PASS] every control failed               
      [PASS] weights present                    
      [PASS] weights key resolves               FrustraMPNN/model_weights/megascale_default_la-true_esm-False_split1.ckpt

  would grade: frustrampnn_parity
  would NOT grade: (none)
  ```
  **15/15 conditions PASS**.

---

## Local Quality Gates & Verification Matrix

All local verification gates pass cleanly:

| Gate | Command | Status | Details |
|:---|:---|:---:|:---|
| **Ruff Linter** | `ruff check --no-fix` | **PASS** | 0 errors across all modified files |
| **Ruff Formatter** | `ruff format --check` | **PASS** | 0 formatting changes required |
| **JAXLint Static** | `jaxlint check src/aminx/families/frustra_mpnn` | **PASS** | 0 non-doc `JL*` violations (clean float64 cast warning only) |
| **Targeted Pytest (Suites 1–7)** | `pytest test_{frustrampnn,knobs,affected,differential,inventory,torch_checkpoint,literal}.py` | **PASS** | 96 passed, 1 skipped in 56.91s |
| **Gate Selftest (Suite 8)** | `pytest tests/knob_gate/` | **PASS** | 151 passed, 3 skipped in 4m 23s |
| **Total Test Suite** | All 8 target suites | **PASS** | **247 passed, 4 skipped, 0 failed** |
| **RedSox Sibling Isolation** | `python scripts/redsox/stale_rows.py` | **PASS** | 0 sibling family ledger rows hit `frustra_mpnn` |
| **Parity Vehicle** | `python scripts/parity/frustrampnn_parity_v2.py` | **PASS** | Clean $f_{64} < 10^{-7}$, clean $f_{32} < 10^{-4}$, 4/4 mutants killed |
| **Wave Verification** | `python scripts/redsox/verify_wave.py` | **PASS** | 15/15 conditions PASS (`would grade: frustrampnn_parity`) |
| **Forensic Audit** | `auditor_r2_1` | **CLEAN** | 130/130 bit-for-bit match, 0 facades, 0 crashes |

---

## Remediation History & Commit Log

- `59bdaa28`: Remediated initial adversarial findings (de-tracked binaries, eliminated `s0_prep` facade, isolated RedSox imports, added fail-loud anti-self-trust guards, implemented 4 mutants).
- `7f49f01a`: Clean merge of `gh/main` (`86042c0e`) into `wt/261105-frustrampnn`.
- `3a462ee4`: Assigned `pack_shim` and `plan_eda` symbols to stage `"P28"` in `browser_validation_paths.json` (anchor commit for graded Bathos run `067db19b`).
- `3d9d62a8`: Recorded post-merge verified Bathos run UUID `067db19b` in ledger comment (`tests/knob_gate/ledgers/frustrampnn.toml`).
- `8d2b306c`: Pointed `FrustraMPNNOptions` line reference in `tests/host/knob_observations.py:306` to `src/aminx/run/options.py:281` (outside vehicle closure; zero impact on wave gradeability).
- `81fe8428`: Updated PR #271 documentation and ledger comment.
- `4f4ff00a`: Clean merge of `gh/main` (`28b7d1f8`), deduplicated `browser_validation_paths.json`, and executed fresh graded Bathos run `a7979816-ae10-413d-9ed7-c7c358247ef6` with certified 15/15 `[PASS]` wave verification.

---

## Follow-Up Debts Filed (Parent Debt #3222)

In accordance with project protocol, 4 follow-up debt items are filed in `.praxia/TECHNICAL_DEBT.md` §15:
1. **Hugging Face Hub Checkpoint Upload & Registry** (`@01M6SE14DGVZ1DD42FEX04A6BK`, Priority: medium):
   - Packaging and uploading `.eqx.zst` weights and manifests to Hugging Face Hub (`maraxen/aminx`) and registering in `family_weights.json`.
2. **ESM-2 Co-Embedding Support** (`@01M6SE195XS9S0P974F75Q0WPX`, Priority: low):
   - Adding ESM-2 co-embedding representations (`use_esm=True`) when upstream releases ESM-enabled weights.
3. **Full-Chain Conv1d / Residue Vectorization** (`@01M6SE1HQZZHRDRCAYH6E3PK9S`, Priority: medium):
   - Vectorizing single-residue `LightAttention1D` / `_predict_chain` or adding optional full-chain Conv1d mode for increased throughput.
4. **Cross-Family Weights Registry Decoupling** (`@01M6SE1QADQ7Y8KPP5SVECXV7P`, Priority: medium):
   - Decoupling model family checkpoint registration so new families register dynamic weight loaders without modifying shared `weights.py`.

---

## Scope Boundaries & Confirmation Gate Invariants

1. **Confirmation Gate Enforced**: This branch will NOT be merged to `main`, no multi-family RedSox wave will be launched, and no weights will be uploaded to Hugging Face Hub without explicit user instruction.
2. **PR Remains in Draft**: PR #271 remains in DRAFT status pending final multi-family integration.
3. **Requirement G Compliance**: `tests/knob_gate/ledgers/frustrampnn.toml` retains an empty `[sidecar]` table on the feature branch; the graded run UUID is documented in ledger comments and verified by `verify_wave.py`.
