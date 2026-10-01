---
title: "S2: ProteinEBM extraction to its own project"
description: "Move the ProteinEBM JAX port (src/aminx/ebm, ~6.6k LOC) with its tests, scripts, evidence, bathos provenance and history into a new standalone project with no runtime edge to aminx, proving the move changed nothing via pre-registered accuracy parity. The MPNN package keeps the name aminx (no rename)."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
owner_repos: [aminx, ebmx]
related_specs: [S1, S3, S4, S5]
---

# S2: ProteinEBM extraction to its own project

Placeholder convention: the new project's name is an open question (section 8, Q1). This spec
uses the working name **`ebmx`** (distribution `ebmx`, import `ebmx`, repo `maraxen/ebmx`,
bathos slug `ebmx`) everywhere. A rename before S2-06 is a find-and-replace; after S2-06 it is a
second history-aware change, which is why S2-01 gates S2-06.

**Tree naming (no-rename revision).** The MPNN package keeps the name `aminx` (user decision 2026-10-01, revised D2): it is released on
PyPI as `aminx`, so nothing is renamed and no S3 rename item exists for S2 to order against. Everything this spec does in the MPNN
repository uses the tree as it is: `src/aminx/ebm`, `aminx.ebm`, `repo = "aminx"`. The "MPNN repo" means `maraxen/aminx`. The hub is a
separate project (working slug `aminx-hub`; its final repo name, PyPI policy and domain are the user's decision at S3-01) and has no part in
this extraction. Section 3 and the Assumption Ledger cite the lines read at `d1210e4a`; those are the paths S2 works on.

Provenance of this spec: user decision D3 and praxia tech debt #2369 (brief
`.praxia/docs/research/261001_ecosystem-hub-recon-brief.md`). Prior art: the original port design
`.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md` and parity report
`.praxia/docs/audits/260716_proteinebm-parity-report.md`. Nothing in this spec executes yet (D7).

## 2. Goal and non-goals

**Goal.** ProteinEBM (a forward energy/score model over CA coordinates) lives in its own
repository and distribution. aminx stops carrying it (aminx keeps its name and its PyPI project; nothing is renamed). The EBM project
imports nothing from aminx at runtime, in tests, or in scripts. The move is proven
behaviour-preserving by a pre-registered, paired, per-unit comparison of the decoy-ranking and
ddG-stability accuracy runs before and after, not by "tests still pass".

**Non-goals.**
- No behaviour change to the model, the checkpoint layout, the numerics or the public function
  signatures. The bias-free `Transition` redesign (the 12 zeroed SwiGLU biases) is explicitly
  deferred because it changes the pytree and invalidates every saved orbax model.
- No browser/hub executor for EBM. The browser-validation inventory already classifies EBM as
  out of scope (P24, "own parity campaign": `.praxia/docs/specs/260923_aminx-browser-validation.md:92`).
- No new science: no retuning of `pinned_t = 0.05`, no new benchmarks beyond re-running the two
  accuracy gates and the throughput benchmarks' import fix-ups.
- No rewrite of the MPNN repo's history. Its `main` is never touched by `git filter-repo`.
- No dependency edge from aminx onto the EBM project (the ecosystem partition forbids new
  heavy edges). S2 ships no `aminx.ebm` shim, no release of its own and nothing under the PyPI name `aminx`: the removal is announced by
  a CHANGELOG pointer (S2-18) and ships in the next release cut under S3-12's guard and train rule (4.10, A40, A43).
- No decision on the EBM-to-MPNN integration story. Combining EBM energies with MPNN logits is a
  consumer-side pipeline concern (hub/pipeline editor, S5/S6), not a library edge.

## 3. Current state

All anchors were read on 2026-10-01 in worktree `261001-xtrax-a11-pin` (aminx `origin/main`
`d1210e4a`, v0.2.0a3) unless a row says otherwise. xtrax was read through `git show
v0.4.0a11:<path>` because the local xtrax checkout is stale. Rows are tagged in the Assumption
Ledger below; the narrative here is the summary. Paths are current-tree paths (nothing is renamed).

### 3.1 What exists

| Area | Tracked content | Size |
|---|---|---|
| `src/aminx/ebm/` | 15 modules + `__init__` (checkpoint, conformational_biasing, contracts, ddg_stability, decoy_ranking, diffusion, dispatch, langevin, langevin_schedule, model, plan, readout, structure_prediction, training, trunk) | 6,608 LOC |
| `tests/ebm/` | 19 files, self-contained (own `conftest.py`) | 5,402 LOC |
| `scripts/ebm/` | 25 files: accuracy drivers, parity checks, 6 throughput benchmarks, each with a `.bth.toml` sidecar, `run_cluster_benchmarks.sh` | |
| `scripts/engaging/submit_accuracy_runs.sbatch` | the full-scale accuracy sbatch | |
| `outputs/ebm_benchmarks/` + `outputs/ebm_benchmarks_h200/` | 41 tracked result files | 1.4 MB |
| `.bth/claims/ebm-ddg-decoy-literature-parity.claim.toml` | campaign `aba3bfe6` | |
| `.praxia/docs/` | audits `260716_proteinebm-parity-report.md`, plans `260709_proteinebm-epic-backlog-dag.md`, specs `260709_proteinebm-aminx-decomposition.md`, research `260712_jax-xla-scf-if-gradient-regression-bug-report.md` | |
| `tests/parity/browser_validation_paths.json` | 79 entries `src/aminx/ebm/...` all mapped to `P24` | |

41 commits touch the code, test, script, output, claim and sbatch paths (`git log -- <paths>`, docs
excluded); the commits touching `src/aminx/ebm` are all by one author. The EBM code shipped in every
release tag from `v0.2.0a1` (2026-09-10) through `v0.2.0a3`; `v0.1.0a7` has none.

**Untracked evidence in the main checkout.** `ebm_benchmarks/` (33 entries) and
`ebm_benchmarks_h200/` (6) at the aminx root are not tracked and do not exist in this worktree.
23 + 6 of their files are byte-identical to the tracked copies under `outputs/`. Nine top-level
files (heterogeneous prod/smoke runs, pytorch_pad_to_batch_max variants,
`langevin_benchmark_batchplanner_verify_full.json`) and the `real_accuracy/` extras (every
`*.partial.jsonl`, the smoke runs, `decoy_ranking_diag2`, `decoy_ranking_diag_orbax`, and two
`*_full.json` that differ from the tracked ones) exist only there. Deleting the root dirs without
importing them would destroy the only copy of that evidence.

### 3.2 Coupling to the rest of aminx (measured by reading import statements)

Nothing outside the EBM trees imports `aminx.ebm`. The only non-import coupling is the AST
inventory test `tests/parity/test_browser_validation_inventory.py:27` (`"ebm"` is in
`_ROOT_PACKAGES`), which fails on stale mappings, so removing the package without editing the 79
JSON entries breaks aminx CI.

EBM imports from the rest of the MPNN package (`aminx.*`; this is the complete list of **aminx** imports, and it is longer than the brief's; the
direct proxide imports in tests are in the table too, see A2 and A32). The `parse_structure` import sites are exactly
five: `src/aminx/ebm/conformational_biasing.py:273`, `tests/ebm/test_alphabet_boundary_ebm.py:119`,
`tests/ebm/test_decoy_ranking.py:246`, `scripts/ebm/bucket_boundary_check.py:139` and
`scripts/ebm/real_decoy_ranking_benchmark.py:125` (A29). The string `aminx.ebm` occurs 343 times across the three trees:

| Dependency | Where | What EBM actually needs |
|---|---|---|
| `aminx.model.diffusion_mpnn.SwiGLU` | `src/aminx/ebm/trunk.py:113` (used `:285`, `:299`); `checkpoint.py:68,178,390,563` | a 27-line module; module import drags `aminx.model.mpnn.Aminx` (`diffusion_mpnn.py:11`) |
| `aminx.utils.safe_map.safe_map` | `plan.py:65` (used `:200`) | vmap-or-`lax.map` chunking |
| `aminx.utils.safe_scan.safe_scan` | `plan.py:66`, `langevin_schedule.py:201` (used `:459`) | `jax.lax.scan` plus an empty-pytree guard |
| `aminx.utils.aa_convert` constants | `ddg_stability.py:114`, `conformational_biasing.py:78` | two string literals; the module itself imports `proxide.chem.residues` and `aminx.types.arrays` at import (`aa_convert.py:7-11`) |
| `aminx.io.parsing.parse_structure` | `conformational_biasing.py:273` (lazy) | `.coordinates[:, 1, :]`, `.residue_index`, `.aatype` from a PDB/mmCIF file |
| tests: `aa_convert.af_to_mpnn`, `string_to_protein_sequence`, `parse_structure` | `tests/ebm/test_conformational_biasing.py:28`, `test_alphabet_boundary_ebm.py:48,119`, `test_decoy_ranking.py:246`, `test_ddg_stability.py:66` | permutation + string helpers |
| tests: direct `proxide.chem.residues.atom_order` | `tests/ebm/test_alphabet_boundary_ebm.py:120`, `tests/ebm/test_decoy_ranking.py:247` | an atom index; these tests need `proxide` even though they import no aminx name for it (they sit behind checkpoint skipifs today) |
| scripts: `aminx.parity.evidence` (8 symbols), `aa_convert.protein_sequence_to_string`, `parse_structure` | `scripts/ebm/collect_synthetic_parity_evidence.py:36`, `real_ddg_stability_benchmark.py:47`, `bucket_boundary_check.py:139`, `lpla_biasing_check.py:98` | metric helpers and record schemas (`parity/evidence.py`, 521 lines) |

EBM does **not** import `aminx.run`, `aminx.host`, `aminx.inference`, `aminx.training` or
`aminx.types` in code (those names appear only in docstrings). It is therefore independent of S1
(spec unification).

Third-party imports that aminx does not declare directly: `einops` (`trunk.py:111`; reaches aminx
only transitively), `scipy` (`decoy_ranking.py:43`; aminx lists it only in the `tests` extra),
`orbax.checkpoint` (`training.py:139`), `pandas` (accuracy scripts). `scripts/engaging/
submit_accuracy_runs.sbatch` says as much in a comment ("orbax/pandas are installed in it but are
not declared pyproject deps, so a sync would prune them"). The new project must declare all of them.

### 3.3 Weights, assets and paths

- The weights are upstream's: `model_6_expert_frozen_1m_md.pt` ("ProteinEBM-x") from
  `huggingface.co/jproney/ProteinEBM`, remapped by `checkpoint.load_pytorch_checkpoint` into a
  `ProteinEBMModel`, then saved as an orbax model. aminx's own HF repo (`maraxen/aminx`) hosts
  MPNN `.eqx.zst` weights only (`tests/conftest.py:44-55`); `HF_REVISION` is an MPNN concept and
  does not apply to EBM. The orbax model exists only on Engaging scratch
  (`submit_accuracy_runs.sbatch:35`) and in `/tmp` on dev boxes. Neither the `.pt` nor the orbax
  tree has a recorded sha256.
- Restore uses a template `ProteinEBMModel` and orbax `PyTreeRestore` by tree path
  (`scripts/ebm/real_decoy_ranking_benchmark.py:88-115`). Any change to field names or tree shape
  breaks restore. `SwiGLU.w_gate` and `w_val` have identical shapes, so a mis-assignment would be
  shape-valid and silent.
- Hardcoded locations violate the derived-data-is-configuration rule: `/tmp/proteinebm_weights/...`
  (checkpoint and orbax defaults in at least 6 scripts), `~/repos/ProteinEBM` (the PyTorch reference,
  imported as `protein_ebm` by the benchmarks), `~/proteinebm_bench_assets/ProteinEBM/eval_data` and
  Engaging scratch paths in the sbatch.
- Upstream licence: the local checkout of `jproney/ProteinEBM` has no `LICENSE` file and neither its
  `README.md` nor `pyproject.toml` mentions one. The trunk is a Boltz-1/AlphaFold3 diffusion module
  port. See Q2.

### 3.4 The accuracy baseline and what it can and cannot support

Tracked baseline: `outputs/ebm_benchmarks/real_accuracy/decoy_ranking_fixed_full.json` (133/133
natives, `mean_spearman_at_pinned_t_abs = 0.8281`, paper target 0.838) and
`ddg_stability_fixed_full.json` (64/64 assays, `mean_spearman_abs = 0.6988`, paper target 0.686),
both at `pinned_t = 0.05` (PR #130). Two limits matter for a move-parity gate:

1. The result JSONs record no jax/jaxlib/xtrax versions, device or code SHA. The baseline's
   environment is recoverable only from Slurm accounting (jobs 18847210/18847211 per the claim file).
2. The three full-scale runs the claim was meant to rest on were wrapped in `bth run` on
   2026-07-25/26 and their records show `outcome = 'unknown'` (two decoy runs, one ddG run) or a
   stale `running` row (two more) in the cool-tier parquet. Only the 2026-07-16 smoke-scale runs have
   an evaluated outcome (`marginal`, `pass`). So the claim's confirmatory re-run has never been
   closed with an evaluated outcome. The accuracy scripts append every unit to a `.partial.jsonl`
   but never read it back (`real_decoy_ranking_benchmark.py:212-226`,
   `real_ddg_stability_benchmark.py:191-205`): there is no resume, and no hashing.

Consequence: "re-run and see 0.83 / 0.68 again" cannot distinguish the move from dependency drift,
and a prior-run comparison alone is not a valid gate. The design in 4.8 therefore runs the
**pre-move code** (arm A0) and the **post-move code** (arm B1) in the same environment and compares
them unit by unit.

### 3.5 Assumption Ledger

Spike tooling (`scripts/loop/adversarial_metrics.py spike-run`) is not present in this workspace,
and this task forbids executing code. No row below was settled by a spike. Rows that need a runtime
probe stay UNVERIFIED with a `deferred:` that names the backlog item that performs the probe before
anything depends on it. Coherence r1 added A45 and A46; no-rename coherence r2 added A47 and A48 and narrowed A46. The no-rename revision re-read the rows it touches (A19, A31, A36, A37, A40-A44) and cites reads; read-only git-history
queries (`git log`, `git rev-list`) stand in for a file:line where a history fact is the evidence.

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | Nothing outside the EBM trees imports `aminx.ebm`; the only other coupling is the browser-validation inventory test and JSON | removal PR needs more edits; gates S2-18 | VERIFIED | read: src/aminx (exhaustive search, 0 hits outside src/aminx/ebm); read: tests/parity/test_browser_validation_inventory.py:27; read: tests/parity/browser_validation_paths.json:22 |
| A2 | The complete list of non-ebm **aminx** imports of EBM src/tests/scripts is the table in 3.2 (direct proxide imports are A32) | a missed coupling surfaces as ImportError in S2-08 | VERIFIED | read: src/aminx/ebm/trunk.py:113; read: src/aminx/ebm/plan.py:65-66; read: src/aminx/ebm/langevin_schedule.py:201; read: src/aminx/ebm/conformational_biasing.py:78,273; read: scripts/ebm/collect_synthetic_parity_evidence.py:36 |
| A3 | Importing `SwiGLU` from `aminx.model.diffusion_mpnn` loads the whole MPNN model module | the SwiGLU copy is still right, only less urgent | VERIFIED | read: src/aminx/model/diffusion_mpnn.py:11 |
| A4 | The alphabet constants EBM uses sit in a module that imports proxide at import time | alphabet seam becomes trivial | VERIFIED | read: src/aminx/utils/aa_convert.py:7-11; read: src/aminx/utils/aa_convert.py:16-17 |
| A5 | `SwiGLU` carries three biased `Linear`s and EBM zeroes the 12 biases with no reference counterpart; restore is by tree path against a template, so the copy must keep class fields `w_gate`, `w_val`, `w_out` | a renamed field breaks restore, possibly silently | VERIFIED | read: src/aminx/model/diffusion_mpnn.py:49-75; read: src/aminx/ebm/checkpoint.py:390; read: src/aminx/ebm/checkpoint.py:563; read: scripts/ebm/real_decoy_ranking_benchmark.py:88-115 |
| A6 | A verbatim `SwiGLU` copy restores the existing orbax model to identical leaves | S2-26 gate (tree-path listing) and S2-13 (bit-level) catch it; design falls back to importing nothing and re-converting | UNVERIFIED | deferred: needs the real checkpoint and a JAX process; performed by S2-26 (tree listing, no checkpoint) and S2-13 (real checkpoint, Engaging node class NC) |
| A7 | In xtrax 0.4.0a11 `SafeMap` was renamed `ChunkedMap`; the old name resolves only through a module `__getattr__` deprecation alias "for one release" | EBM would break on the next xtrax release without S2-11 | VERIFIED | read: git v0.4.0a11:src/xtrax/tiling/__init__.py:84; read: git v0.4.0a11:src/xtrax/_renamed.py:21; read: src/aminx/ebm/plan.py:62 |
| A8 | aminx `safe_map` and xtrax `chunked_map` have the same semantics (vmap if `batch_size` is None or `n <= batch_size`, else `lax.map(batch_size=...)`); they differ only in the empty-pytree error | `_tiling.py` keeps a local wrapper instead of a re-export | VERIFIED | read: src/aminx/utils/safe_map.py:21-52; read: git v0.4.0a11:src/xtrax/transforms/map.py:8-27 |
| A9 | Every xtrax symbol EBM imports exists at 0.4.0a11: `tiling.{AxisDecision,AxisSpec,BatchPlanner,CarrySpec,Scan,Vmap}`, `checkpoint.{get_checkpoint_manager,save_checkpoint,load_checkpoint}`, `engine.Engine`, `training.optim.adamw_with_schedule`, `training.types.ResumableState`, `data.module.DataModule` | S2-11 needs a shim | VERIFIED | read: git v0.4.0a11:src/xtrax/checkpoint/orbax.py:24,63,89; read: git v0.4.0a11:src/xtrax/engine/engine.py:71; read: git v0.4.0a11:src/xtrax/training/optim.py:46; read: git v0.4.0a11:src/xtrax/training/types.py:41; read: git v0.4.0a11:src/xtrax/data/module.py:17 |
| A10 | Those xtrax modules import cleanly from a base install with no extras (`xtrax` core depends on jax, jaxlib, equinox, optax, orbax-checkpoint, numpy; `grain` moved to the `data` extra) | `ebmx` must depend on `xtrax[data]` or another extra; heavier install | UNVERIFIED | deferred: file existence is read, import success needs a clean-venv import; performed by the S2-28 gate on titanix; read: git v0.4.0a11:pyproject.toml:7 |
| A11 | xtrax 0.4.0a11 requires Python >=3.13 and `jax>=0.10.2,<0.12`, so `ebmx` inherits both floors | ebmx could set a lower floor | VERIFIED | read: git v0.4.0a11:pyproject.toml:6-7 |
| A12 | EBM needs `einops`, `scipy`, `orbax`, `pandas` that aminx does not declare directly | none (the new project declares them anyway) | VERIFIED | read: src/aminx/ebm/trunk.py:111; read: src/aminx/ebm/decoy_ranking.py:43; read: src/aminx/ebm/training.py:139; read: scripts/engaging/submit_accuracy_runs.sbatch:27-28 |
| A13 | `aminx.io.parsing.parse_structure` is a thin wrapper over proxide with `OutputSpec(compute_rbf=True, ...)`; bare proxide with a default `OutputSpec` returns the same `coordinates[:,1,:]`, `residue_index`, `aatype` for EBM's inputs | the adapter must pass explicit OutputSpec flags; one golden test decides | UNVERIFIED | read: src/aminx/io/parsing/dispatch.py:23-38,42-79 (wrapper); deferred: equality needs a proxide process, performed by S2-09 against the goldens S2-21 produces from aminx |
| A14 | The accuracy scripts append per-unit results to `.partial.jsonl` but never read them back; no resume, no hashing | S2-04 shrinks | VERIFIED | read: scripts/ebm/real_decoy_ranking_benchmark.py:212-226; read: scripts/ebm/real_ddg_stability_benchmark.py:191-205 |
| A15 | The 2026-07-25/26 full-scale tracked runs have no evaluated outcome | the old claim is already closed and A0 is redundant | VERIFIED | read: ~/.bth/catalog/runs/aminx/run_9efd2b47-6472-4f4b-8180-ef578179bf76.parquet (outcome unknown); read: ~/.bth/catalog/runs/aminx/run_7e8e7f9a-a8f4-403e-aec6-15eb7a443048.parquet (unknown); read: ~/.bth/catalog/runs/aminx/run_3db03d31-9b24-4506-b752-21ccff033fd1.parquet (unknown); read: ~/.bth/catalog/runs/aminx/run_51db6a26-ef28-49ea-84f4-4c04709efaad.parquet (running, stale) |
| A16 | The baseline JSONs record no environment or hardware metadata | baseline could anchor the gate alone | VERIFIED | read: outputs/ebm_benchmarks/real_accuracy/ddg_stability_fixed_full.json:1 (top-level keys: n_assays_*, pinned_t, n_unfolded_ensemble, paper_target, mean_spearman_*, sign_convention_note, per_assay); read: outputs/ebm_benchmarks/real_accuracy/decoy_ranking_fixed_full.json:1 |
| A17 | Same code, same GPU, same env gives bit-identical per-unit Spearman values | tolerance must come from a measured floor (the design already measures it) | UNVERIFIED | deferred: needs GPU hours; the floor stage of S2-05 measures it before any tolerance is written |
| A18 | The root `ebm_benchmarks*` dirs are untracked and hold evidence not under `outputs/` | the import item S2-07 is unnecessary | VERIFIED | read: /home/marielle/projects/aminx/ebm_benchmarks (9 top-level files absent from outputs/ebm_benchmarks, plus real_accuracy extras; `cmp` shows 23 + 6 identical); read: git ls-files (no path under ebm_benchmarks/) |
| A19 | `aminx.ebm` is importable from tag v0.2.0a1 onward and the wheel includes it (`packages.find where=src`), so published alphas carry EBM and its removal needs a user-facing notice | none for S2: the CHANGELOG pointer in S2-18 is accurate either way and no S2 item depends on the answer | UNVERIFIED | read: git ls-tree v0.2.0a1 src/aminx/ebm (16 files); read: pyproject.toml:222-223; deferred: PyPI wheel contents need network; nothing in S2 depends on the result |
| A20 | `git filter-repo` is installed and the EBM path set has 41 commits | S2-06 needs an install step | VERIFIED | read: /home/marielle/.local/bin/git-filter-repo; read: git log -- <EBM paths> (41 commits) |
| A21 | bathos has no command that moves runs between projects; the project slug and id live in `.bth.toml`; `bth claim register` anchors a claim by path + SHA256 | the lineage design (4.9) could be simpler | VERIFIED | read: .bth.toml:2,4; read: bth --help (command list); read: bth claim --help (register: "path + SHA256 anchor") |
| A22 | Whether the claim anchor path is stored absolute or relative to the project, and whether `bth campaign` accepts runs from a second project | S2-15 may need a cross-project note instead of `campaign add` | UNVERIFIED | deferred: needs `bth` against a scratch catalog; performed first thing in S2-03, before S2-04 writes any sidecar |
| A23 | `alphex` is numpy-only and ships `known.AF_X_21` and `MPNN_X_21`; it is resolvable from PyPI as `alphex>=0.1.0a1` | dev-only conformance test uses a path or git source instead | UNVERIFIED | read: /home/marielle/projects/alphex/pyproject.toml:41-43; read: /home/marielle/projects/alphex/src/alphex/known.py:48,96; deferred: PyPI availability rests on a doc statement (alphex `260814_alphabet-contract.md:278-279`), not a probe |
| A24 | The aminx `requires_weights` marker gate checks MPNN weights, not the EBM checkpoint | the EBM conftest can reuse it | VERIFIED | read: tests/conftest.py:39-41; read: tests/ebm/test_alphabet_boundary_ebm.py:131-133 |
| A25 | A GPU box (titanix, or Engaging via sbatch) is available for the heavy gates | gates move to the other box | UNVERIFIED | deferred: availability is operational; taken from user memory notes, not re-probed; no design change |
| A26 | EBM is classified out of browser validation (P24) | EBM would need an ONNX route (S4) | VERIFIED | read: .praxia/docs/specs/260923_aminx-browser-validation.md:92 |
| A27 | At xtrax v0.4.0a11 `xtrax.tiling.strategy` defines `ChunkedMap` and keeps `SafeMap` only through a module `__getattr__` deprecation alias, so the deep import in `tests/ebm/test_plan.py:15` still resolves, with a `DeprecationWarning` (r1 C10) | test_plan.py would fail collection; mitigated anyway because the rename moved into S2-08 | VERIFIED | read: git v0.4.0a11:src/xtrax/tiling/strategy.py:66,156-163 (r1 C10); read: tests/ebm/test_plan.py:15,35,47,62 |
| A28 | The current `checkpoint.load_pytorch_checkpoint` plus converter, run on the upstream `.pt`, reproduces the existing Engaging-scratch orbax tree leaf for leaf (the scratch tree dates from an earlier conversion and `checkpoint.py` has changed since, e.g. the bias zeroing at `:390,:563`) (r1 C5) | S2-12 records the mismatching leaves, the legacy tree (hash recorded by S2-03) stays the A0/tier-1/B1 pairing reference, and a debt item is filed for the converter; S2-13 depends on the legacy hash, not on converter equality | UNVERIFIED | deferred: needs the `.pt` and a JAX process on the node that holds the scratch tree; performed by S2-12 |
| A29 | `parse_structure` is imported from aminx at exactly the five sites listed in 3.2 (r1 C1) | S2-27 must rewrite the missed site or its grep gate fails | VERIFIED | read: src/aminx/ebm/conformational_biasing.py:273; read: tests/ebm/test_alphabet_boundary_ebm.py:119; read: tests/ebm/test_decoy_ranking.py:246; read: scripts/ebm/bucket_boundary_check.py:139; read: scripts/ebm/real_decoy_ranking_benchmark.py:125 |
| A30 | `pi_so3` holds more than one GPU type (CLUSTER.md names Blackwell nodes 4007/4008 and an H200 node), so a bare `--partition=pi_so3` does not pin the device kind; the exact `--gres` type string for node class NC must be read from the cluster (r1 C4) | none of the gates change, only the pin string | UNVERIFIED | deferred: needs a `scontrol show node` call on the cluster, not run here; recorded by S2-03 into `goldens/node_class.toml` |
| A31 | The EBM path set has a single path era: `src/prxteinmpnn/ebm` has no commit on any ref, and no rename or delete in the history touches an ebm path (git default rename detection) (r1 C13) | the filter needs a second `--path-rename` and `ebm-paths.txt` an earlier era; S2-06's blob-history gate catches a miss | VERIFIED | read: git rev-list --all -- src/prxteinmpnn/ebm (0 commits); read: git log --all --diff-filter=R -M --name-status (565 renames repo-wide, none names ebm); read: git log --diff-filter=D --name-only over src/aminx/ebm, tests/ebm, scripts/ebm (empty); S2-06 still commits its own enumeration |
| A32 | Two EBM tests import `proxide.chem.residues.atom_order` directly, so the `tests` extra needs proxide (via the `structure` extra) or an `importorskip` guard (r1 C15) | clean-venv test run would error at import | VERIFIED | read: tests/ebm/test_alphabet_boundary_ebm.py:120; read: tests/ebm/test_decoy_ranking.py:247 |
| A36 | No S3 item (S3-09 sweep and CI wiring, S3-12 release guard, S3-21 public surfaces, S3-24 checker) edits `src/aminx/ebm`, the consumed files of the freeze manifest, `uv.lock` or the pyproject dependency tables | S2-32/33/34 fail `--check-freeze` and the A0 evidence is re-run from the first affected item | UNVERIFIED | read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:518-522,527-531 (S3-09 and S3-12 scopes: residue sweep, CI wiring, release.yml); deferred: S3 items are not implemented yet; enforced mechanically by `--check-freeze` in S2-32/33/34 and S2-06 rather than assumed; A44 shows S3-09 has nothing to sweep in the manifest files today |
| A37 | Draft PR #174 (xtrax 0.4.0a11 pin) has one owner, S2-02; S4-16 `depends_on` S2-02 and only resolves the orbax conflict on top of it; no S3 item depends on S2-02 any more (C7) | contradictory preconditions (merge-before vs rebase-on) would return | VERIFIED | read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:89 (S4-16 deps become [S4-03, S2-02]); read: .praxia/docs/specs/261001_xtrax-model-contract.md:1483 (S4-16 keeps S2-02; its retired S3 edges are S4's to drop) |
| A38 | Lock changes from other specs (S4-16 orbax/xtrax[export], S4-35 xtrax bump, S1-17 optional mistypotts dev dependency) can land on main between S2-05 and S2-34 because no edge orders them against S2 (CH2-04); the freeze manifest as written (EBM and consumed sources only) would not detect them | A0 runs would span different locks and the A0-vs-B1 comparison would rest on unequal environments (GPU-hour reruns) | VERIFIED | read: .praxia/docs/specs/261001_ebm-extraction.md:398-422; changed: manifest part (c) now hashes `uv.lock` and the pyproject dependency tables, stamps carry `orbax-checkpoint` and the lock hash, A0 items run from a detached worktree at F', and S2-22 pins ebmx's first lock to the A0-recorded versions |
| A39 | A detached worktree at `F'` contains everything an A0 run item needs (runner, sidecars, S2-04 outputs), and outputs written to an explicit out dir can be committed afterwards on the working branch without changing the frozen inputs | S2-34 uses a lock-hash-checked descendant of `F'` instead (already allowed) | UNVERIFIED | deferred: depends on S2-04/S2-21 landing order, checked by `--check-freeze` at the start of each run item |
| A40 | aminx has no `gone_modules.toml`, alias table or meta-path finder today, and S3's no-rename revision retired the `aminx.ebm` shim row, so removing `aminx.ebm` needs no shim edit | S2-18 would need a shim item | VERIFIED | read: src/aminx (directory listing: no gone_modules.toml; none at repo root either); read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:88 |
| A41 | The bathos project slug stays `aminx` (145 runs), so A0 and floor sidecars and the lineage document use it as it is | sidecar `--project-slug` values change | VERIFIED | read: .bth.toml:2; read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:266 |
| A42 | PyPI `aminx` stays the MPNN package and the hub is a separate project that never takes the name `aminx` as a PyPI project, import package or package.json name (S3-24 reserved-name mode) | the CHANGELOG pointer and install text in S2-18 would name a different distribution | VERIFIED | read: .praxia/docs/research/261001_ecosystem-hub-recon-brief.md:19-25; read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:40 |
| A43 | S2 needs no release item: the version is a static string in `pyproject.toml`, S3-12 guards whichever item cuts the next release, and with no S2 release item the EBM removal rides the next release cut after S2-18 | if the removal must ship in a dedicated alpha, S2 adds a release item ordered against S5-52 and S1-25 per S3 C4 | VERIFIED | read: pyproject.toml:7; read: .praxia/docs/specs/261001_aminx-identity-and-hub-naming.md:398 |
| A44 | No file in the EBM trees or the freeze manifest's consumed set contains `prxteinmpnn`, so the S3-09 residue sweep does not invalidate the freeze | S3-09 would need an order against S2-05..S2-34 or a re-freeze | VERIFIED | read: src/aminx/ebm, tests/ebm, scripts/ebm, src/aminx/utils/aa_convert.py, src/aminx/utils/safe_map.py, src/aminx/utils/safe_scan.py, src/aminx/io/parsing, src/aminx/parity/evidence.py, src/aminx/model/diffusion_mpnn.py, scripts/engaging/submit_accuracy_runs.sbatch (case-insensitive grep for prxtein: 0 files) |
| A45 | `tests/parity/test_browser_validation_inventory.py` and `browser_validation_paths.json` are shared by S1, S4, S5 and S2: the test fails on unmapped and stale public top-level symbols in `src/aminx/{inference,sampling,scoring,model,host,tiling,ebm,potts}` and on every `aminx.run.__all__` name, and the JSON maps the `aminx.run` names | S2-18's JSON edit would be independent of the others; if true it is a textual merge that must be re-counted | VERIFIED | read: tests/parity/test_browser_validation_inventory.py:18-29; read: tests/parity/test_browser_validation_inventory.py:101-137; read: tests/parity/browser_validation_paths.json:2-21; read: src/aminx/run/__init__.py:41 |
| A46 | No consumer repo's own main code imports `aminx.ebm`, and the only known consumer of the branch is the hautespout WIP worktree whose vendored aminx carries `hp_*` modules (S1 A39 names `hp_energy`, `hp_coarse_sampler`, `hp_windowing`; reported, not re-read here) | if a consumer imports `aminx.ebm`, S2-18 strands it (S2-18 now stops on a hit); if the WIP copy has `hp_*` files beyond the branch tip, S1-28's bump silently drops them | UNVERIFIED | read: .git/packed-refs:156; deferred: the consumer `rg` and the WIP vendored file list are performed by S2-35 and S2-18, which record the result and stop on a hit |
| A47 | The branch `hp-lattice-sanity-check` (tip ee82d66d29b8d959c2212d20bc05cadfba48a2ff, a local and an origin ref) has no merge base with main, and its tip tree holds exactly one hp module under `src/aminx/ebm/` (`hp_energy.py`), plus `tests/ebm/test_hp_energy.py` and `scripts/validate/hp_lattice_sanity.py` with its `.bth.toml`; `hp_coarse_sampler` and `hp_windowing` appear nowhere in the tip tree, so S1's three-module list comes from the WIP vendored copy, not from the branch; a `--single-branch main` clone carries none of it | if S2-35 finds more files on a descendant of the tip or in the WIP copy, the merge or port file list grows (S2-35 records the list it acts on) | VERIFIED | read: git ls-tree -r --name-only hp-lattice-sanity-check (r2 R2-04: src/aminx/ebm, tests/ebm, scripts/validate listings; 0 matches for hp_coarse_sampler or hp_windowing in the whole tip tree); read: git merge-base main hp-lattice-sanity-check (exit 1, no common ancestor); changed: S2-03 now only records the file list, S2-35 owns the decision and S2-36 the port |
| A48 | The lock and dependency tables are not in the moved path set (4.2), so `--check-freeze` equality of their hashes at `G` carries no information about whether the moved code is the code A0 ran, while asserting it there fails on lock churn from S4-16, S4-35 or S1-17 that never reached A0 (R2-01) | if the lock were part of the extraction path set, S2-06 would have to assert it | VERIFIED | read: .praxia/docs/specs/261001_ebm-extraction.md:259-261 (4.2 table: the moved sets are `src/aminx/ebm`, `goldens/ebm`, `tests/ebm`, `scripts/ebm`, the sbatch and `tools/ebm_inventory.py`); read: .praxia/docs/specs/261001_ebm-extraction.md:206 (A38); changed: 4.7 step 2 splits `--check-freeze` into asserted code parts and recorded-at-`G` environment pins, `--assert-env` only from the detached `F'` worktree |

## 4. Design

### 4.1 Target layout of the new repository

```text
ebmx/
  pyproject.toml            # hatchling (as xtrax); requires-python >=3.13
  .bth.toml                 # NEW slug "ebmx" + NEW project id (bth init); remotes engaging/titanix
  CLAUDE.md  README.md  NOTICE  LICENSE  CHANGELOG.md
  src/ebmx/
    __init__.py
    contracts.py diffusion.py trunk.py readout.py model.py checkpoint.py plan.py dispatch.py
    langevin.py langevin_schedule.py structure_prediction.py conformational_biasing.py
    ddg_stability.py decoy_ranking.py training.py          # moved verbatim, imports rewritten
    _layers.py      # SwiGLU copy (4.3)
    _tiling.py      # chunked_map + scan seam over xtrax (4.3)
    _alphabet.py    # AF/MPNN literals, perms, string<->aatype (4.3)
    _structure.py   # optional proxide adapter, lazy (4.3)
    _config.py      # location resolver (4.6)
    _weights.py     # weights.lock verification + converter entry point (4.5); vendors scripts/leaf_hash.py (hash-checked)
  tests/            # tests/ebm/* flattened; conftest owns requires_weights for EBM weights
  scripts/          # scripts/ebm/* flattened; benchmarks/ kept; _evidence.py vendored (4.3); leaf_hash.py (4.5)
    slurm/submit_accuracy_runs.sbatch
  outputs/ebm_benchmarks/ outputs/ebm_benchmarks_h200/     # verbatim paths (docs and sidecars cite them)
  outputs/ebm_benchmarks/untracked_import/                 # S2-07, with MANIFEST.sha256
  goldens/          # produced in the MPNN tree (aminx) by S2-03/S2-21/S2-05 (see 4.2): model_tree_paths.json, parse_structure goldens,
                    # tier-1 goldens, node_class.toml, MANIFEST.sha256 (plain git, <500 KB, no LFS)
  .bth/claims/      # copy of the old claim + the new extraction claim
  .praxia/docs/{audits,plans,specs,research,decisions,reference}/
  docs/HISTORY.md   # PR numbers in old messages refer to maraxen/aminx; commit-map location
  provenance/commit-map.txt   # filter-repo old->new SHA map
```

`.gitattributes` is not carried over (the MPNN repo's LFS filters for `*.npz`, `*.eqx`, `*.onnx` must not
apply here). Goldens are plain-git sized.

### 4.2 What moves, what is copied, what stays

| Item | Disposition | Notes |
|---|---|---|
| `src/aminx/ebm/` -> `src/ebmx/` | **move with history** (`--path-rename src/aminx/ebm/:src/ebmx/`) | flatten; contents rewritten only in post-filter commits; the code has lived at `src/aminx/ebm` for its whole history (A31) |
| `tests/ebm/` -> `tests/` | move with history | |
| `scripts/ebm/` -> `scripts/` | move with history | sidecars travel with their scripts |
| `goldens/ebm/` (new in aminx, S2-03/S2-21/S2-05) -> `goldens/` | move with history (`--path-rename goldens/ebm/:goldens/`) | every golden generated in the MPNN tree lives here so the filter keeps it (r1 C2); includes `goldens/ebm/freeze_manifest.json` (4.7 step 2) |
| `scripts/ebm/leaf_hash.py`, `scripts/ebm/goldens/` (generators, new in aminx) | move with history | generators travel with their outputs |
| `tools/ebm_inventory.py` (new in aminx, S2-03) | move with history, path kept (`tools/`) | S2-18 deletes the aminx copy; S2-20 uses the ebmx copy. `tools/verify_extraction.sh` is NOT in the MPNN tree: S2-06 creates it in the ebmx clone as a post-filter commit, taking `<aminx_repo> <G> <clone>` as arguments (r2 C8) |
| A0 per-unit outputs (new in aminx) -> `outputs/ebm_benchmarks/extraction_parity/a0/` | move with history (inside the `outputs/ebm_benchmarks*` rule) | small JSON units, tracked; if any file exceeds 1 MB it is manifested by sha256 instead and recorded in `ebm-frozen-baseline` (r2 C6) |
| `scripts/engaging/submit_accuracy_runs.sbatch` -> `scripts/slurm/` | move with history | hardcoded `/home/maarxaru/projects/aminx` and scratch paths replaced by config (4.6) |
| `outputs/ebm_benchmarks*/` | move with history, **same paths** | cited by sidecars, claim, report |
| root `ebm_benchmarks/`, `ebm_benchmarks_h200/` extras (main checkout path `/home/marielle/projects/aminx/ebm_benchmarks*`, recorded in the S2-03 manifest) | **import as new files** under `outputs/ebm_benchmarks/untracked_import/` with sha256 manifest | never history-rewritten into old commits; removed from the main checkout only in S2-20, a user decision, after the imported copy is on a remote and a tarball exists (r1 C7) |
| claim `ebm-ddg-decoy-literature-parity.claim.toml` | **copy to ebmx; aminx keeps its file byte-identical** | the bathos registration anchors path + SHA256 (A21); editing or deleting it in the MPNN repo can break `bth claim validate` |
| 4 EBM docs (audit, plan, spec, research) | move with history; aminx keeps a one-file pointer `.praxia/docs/reference/261001_ebm-moved.md` | historical docs are append-only |
| `SwiGLU` original in `src/aminx/model/diffusion_mpnn.py` | stays | still used by `DiffusionAminx` |
| `aminx.utils.{safe_map,safe_scan,aa_convert}`, `aminx.io.parsing`, `aminx.parity.evidence` | stay in aminx | EBM stops depending on them; copies/seams, not moves |
| browser-validation inventory: 79 JSON entries (paths `src/aminx/ebm/...` count re-read at S2-18; the file is shared with S1, S4 and S5, so a textual merge, 4.10), `"ebm"` in `_ROOT_PACKAGES` | edited in S2-18 | entries are all `P24` (not `internal`), so the internal-count baseline file (98) is unaffected |
| weights | **not hosted by the MPNN repo**; stay upstream | see 4.5 |
| bathos run records | stay in the shared catalog under project slug `aminx` (the slug stays `aminx`, A41; A0 and floor runs also use it, 4.8) | immutable; the lineage document (S2-15) maps them |

### 4.3 Coupling cuts (decisions)

**SwiGLU: copy verbatim into `ebmx/_layers.py`; do not share.** It is 27 lines, xtrax has no
SwiGLU (a11 grep), and a shared package for 27 lines is a new ecosystem edge. The copy keeps the
class fields `w_gate`, `w_val`, `w_out` and the three biased `eqx.nn.Linear`s so the pytree, the
`_zero_extra_swiglu_biases` selectors (`checkpoint.py:390,563`) and every saved orbax model keep
working. A frozen listing of `jax.tree_util.keystr` paths and shapes of the full
`ProteinEBMModel`, generated from the MPNN tree (aminx) at the freeze SHA, is committed as
`goldens/model_tree_paths.json` (generated by S2-21 in aminx, which owns every pre-move golden generator) and asserted in a test (S2-26) (catches a silent `w_gate`/`w_val` swap
without needing the checkpoint). Making `Transition` bias-free is a separate, parity-gated
follow-up (Q6), not part of the move.

**`safe_map` / `safe_scan`: a local `ebmx/_tiling.py` seam, not a dependency.**
`chunked_map` is imported from `xtrax.transforms` (a11), wrapped to keep the pre-move `safe_map`'s empty-pytree
`ValueError` (A8); `scan` is `jax.lax.scan` with the same empty guard. Precedent: demistify's single
`_tiling.py` seam over xtrax. `plan.py:dispatch_axis` keeps calling these directly (its docstring
at `plan.py:156-167` explains why `make_axis_dispatch` is not used).

**xtrax rename.** `plan.py:62` and `:199` import and test `SafeMap`, and `tests/ebm/test_plan.py:15` deep-imports it from
`xtrax.tiling.strategy`. On a11 that name is a deprecated alias on both modules (A7, A27) that returns `ChunkedMap` and warns;
it disappears next release. The rename to `ChunkedMap` is done in **S2-08** together with the mechanical rewrite (so no
interim item runs against the alias), and test imports go through the public `xtrax.tiling` (aminx's ban list in
`pyproject.toml` around :158-169 forbids `xtrax.tiling.strategy` and `.plan`, and the copied tests would violate it). S2-11 then
only adds the `-W error::DeprecationWarning` gate and the `<0.5` pin check.

**Alphabets: `ebmx/_alphabet.py` holds the two literals** (`AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"`,
`MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"`), numpy permutation arrays, `af_to_mpnn`,
`string_to_aatype` and `aatype_to_string`, with no proxide and no aminx. A dev-only conformance test
asserts both literals equal `alphex.known.AF_X_21` / `MPNN_X_21` (alphex decision D4 pattern: a
dev-only test dependency is not a coupling edge). Taking alphex as a runtime dependency is
deferred (Q5): it is numpy-only and light, but alphex's own contract asks the MPNN repo not to take a new
edge while the partition is in flight, and EBM is where the shipped alphabet-permutation bug
(decoy 0.83 -> 0.32) was found, so the conformance test is the right first step. The existing
`alphabet_boundary` test marker and the `test_alphabet_boundary_ebm.py` regression (which parses a
real structure and checks the AF-vs-MPNN convention at the real boundary) move unchanged.

**Structure parsing: `ebmx/_structure.py`, lazy, behind an extra.** The adapter and the rewrite of **all five** import sites
(A29: src, two tests, two scripts) land together in S2-27, so the grep gate and the import blocker hold from S2-28 on; S2-09 only
proves the adapter equals goldens that S2-21 made in aminx. `load_conformational_states`
calls `parse_structure` only to read CA coordinates, residue index and aatype. The new function
imports proxide inside the call and raises an actionable error naming `pip install ebmx[structure]`.
Array-level APIs (`align_conformational_states`, all scoring) never import proxide. The extra pins
`proxide>=0.1.0a17` (the MPNN package's current floor). Whether bare proxide matches the MPNN package's wrapper output is A13: S2-09
resolves it with a golden test, and if they differ the adapter passes the same `OutputSpec` flags.
EBM taking proxide as an optional extra is not a partition problem: the partition's concern is
heavyweight *inbound* coupling onto light packages (alphex's rationale), and `ebmx` is a new leaf.

**`aminx.parity.evidence` (scripts only): vendored** as `scripts/_evidence.py` with a provenance
header (source path, aminx SHA, sha256 of the vendored symbol sources). It is script-side, so the
package stays clean. A dev dependency on aminx was rejected: it would re-create the edge in the
project built to remove it.

**Test infrastructure.** `tests/ebm/conftest.py` already stands alone (A24 shows no use of root
fixtures). The `requires_weights` marker gets an EBM-specific skip hook that checks the EBM weights
resolver (4.6), not MPNN weights.

**Tests that need proxide directly** (`atom_order` at `test_alphabet_boundary_ebm.py:120` and `test_decoy_ranking.py:247`, A32):
the `tests` extra includes `ebmx[structure]`, and those tests also call `pytest.importorskip("proxide")` so a minimal install
skips them instead of erroring.

**Verdict on runtime coupling.** After these cuts (true only once S2-27 and S2-28 have landed) `ebmx` imports no `aminx` in
`src`, `tests` or `scripts`; proxide is optional; the PyTorch reference (`protein_ebm`) and `torch`
are dev/benchmark-only (extra `bench`). Enforced by S2-28 and S2-17.

### 4.4 `pyproject.toml` (new project)

Runtime dependencies: `xtrax>=0.4.0a11,<0.5` (core, **no extras**: avoids the jax2onnx -> orbax-export
-> orbax-checkpoint downgrade measured in the brief), `jax>=0.10.2,<0.12`, `equinox>=0.13.1`,
`optax>=0.2.3`, `orbax-checkpoint>=0.11.17`, `jaxtyping>=0.2.30`, `einops`, `scipy>=1.12`,
`numpy>=2.1`.
Extras: `structure = [proxide>=0.1.0a17]`, `bench = [torch>=2.12, pandas]` (PyTorch reference
comparisons, accuracy scripts), `cpu = [jax[cpu]]`, `cuda12 = [jax[cuda12]; linux]`,
`tests = [pytest, pytest-cov, chex, beartype, alphex>=0.1.0a1]`, `dev = [ruff, ty, jaxlint]`.
Markers carried over: `slow`, `requires_weights`, `alphabet_boundary`. `addopts = "-m 'not slow and not requires_weights'"`.
`ruff` config follows aminx's (line length 100, indent 2, `UP037` ignored: it mangles jaxtyping
shape strings).

### 4.5 Weights provenance

A tracked `weights.lock.toml` (a lock file, not an experiment config) records, for the single
supported checkpoint: HF repo `jproney/ProteinEBM`, **resolved revision commit**, filename
`model_6_expert_frozen_1m_md.pt`, sha256 of the `.pt`, and the leaf-hash of the converted orbax tree
(**normative definition**: sha256 over the UTF-8 lines
`"{jax.tree_util.keystr(path)}|{dtype}|{shape}|{sha256(np.ascontiguousarray(leaf).tobytes()).hexdigest()}\n"` sorted by the path
string; independent of orbax serialization bytes). There is exactly one implementation, `scripts/ebm/leaf_hash.py`, written in
aminx by S2-03 and moved by the filter; `ebmx._weights` vendors it with a header recording the source sha256, and a test asserts the
vendored copy's sha256 equals the recorded one (r1 C2). `ebmx._weights.verify` refuses a mismatch. A converter entry point
(promoted from `scripts/ebm/checkpoint_parity_check.py`) reproduces the orbax tree from the `.pt`;
the S2-12 gate **reports** leaf-hash equality with the hash recorded from the existing Engaging-scratch model in S2-03 (the
**legacy tree**) and, on a mismatch, lists the differing leaves (A28). The legacy tree, identified by its recorded leaf hash, is
the pairing reference for A0, tier 1 and B1; `weights.lock.toml` records both hashes and a boolean `converter_matches_legacy`.
A mismatch does not stall the B1 parity chain (S2-13 depends on the legacy hash) but it does gate the cutover (r2 C5): S2-31, a user-decision item (Q14), chooses between fixing the converter (then re-convert and re-run tier 1 on the converted tree) and proceeding with the legacy tree as the only supported checkpoint, and S2-17 and S2-18 depend on S2-31. On a match S2-31 is a one-line recorded acceptance. Whether to redistribute the converted weights (Q10) is a user decision because they are
derivative of an unlicensed-as-far-as-we-know upstream.

### 4.6 Location resolution (derived-data directories are configuration)

One resolver, `ebmx._config`, for the five locations the code currently hardcodes:
`weights_dir` (the `.pt`), `orbax_model_dir`, `assets_dir` (ProteinEBM `eval_data`: decoys,
tmscore, proteingym), `reference_repo` (the PyTorch `protein_ebm`, benchmarks only), and
`jax_cache_dir` (persistent XLA compilation cache, 15-20 min cold compile).
Precedence, exactly: explicit argument > environment variable (`EBMX_WEIGHTS_DIR`, `EBMX_ORBAX_DIR`,
`EBMX_ASSETS_DIR`, `EBMX_REFERENCE_REPO`, `EBMX_JAX_CACHE_DIR`; value `none` or empty disables) >
`[tool.ebmx]` table in the nearest `pyproject.toml` > `${XDG_CONFIG_HOME:-~/.config}/ebmx/config.toml`
> unset (raises with the list of layers tried). No default path in library code. `config_source(key)`
reports which layer decided. A malformed file fails loudly. Writes (the orbax model, the JAX cache)
are atomic via a per-writer temp dir and keyed by the real source identity (sha256 of the `.pt`), not
a staging name. Never write next to source data or into another project's directories. Check the
storage tier's quota/inode limits before choosing a cluster location (venvs and compile caches are
inode-heavy). The five `DEFAULT_*` constants in `scripts/` and the sbatch `ASSETS`/`ORBAX_MODEL`/
`JAX_COMPILATION_CACHE_DIR` lines become reads of this resolver (the sbatch exports the env vars from
the user's config file, never from a literal).

### 4.7 History-preserving extraction procedure

Rules: the MPNN repo's history is **never rewritten** (so no SHA changes in any worktree, branch, PR or
bathos `git_sha` record). Every step below runs on the tree as it is, so the EBM code lives at `src/aminx/ebm` in the commits that A0
runs and that the clone is cut from. All rewriting happens on a throwaway fresh clone. The demistify filter-repo
incident (a full-history rewrite saved only by a pre-rewrite bundle) is the reason for the order below.

0. **Home and remote (r1 C8).** The permanent clone location is chosen in S2-01 and recorded in the ADR; the remote is created in
   S2-17 (user-gated). Until then the ebmx items (S2-08 to S2-16, S2-22 to S2-28) are reviewable commit ranges on a local branch
   of the permanent clone, synced to titanix and Engaging by anchored rsync (never `myxcel push`, which resolves the main
   checkout, not a worktree). "PR-sized" means a self-contained commit range with its own gate; S2-17 pushes the whole branch.
1. **Freeze and back up (S2-03, which follows the hp-lattice disposition S2-35, so a merge-before-F lands before `F`).** Record freeze SHA `F`. `git bundle create` of all refs of the MPNN repo
   to a path outside the repo (demistify precedent: `~/projects/<repo>_pre_filterrepo_backup_<date>.bundle`),
   then `git bundle verify`. Tarball the untracked root evidence dirs. Write the sha256 inventory.
2. **Single frozen code SHA and file-level freeze manifest (r2 C7, coherence C3).** `F'` is the MPNN-repo SHA at which S2-05 (the first
   A0 run item) executes; S2-05 writes it into `ebm-frozen-baseline`. S2-03 depends only on S2-02 (the xtrax pin), so the freeze is taken on
   main as it stands: no other spec's item is a prerequisite of the A0 evidence. A directory-level empty-diff gate is replaced by a
   **freeze manifest**, because S1, S3 and S4 legitimately change other files in the same directories (and the lock) while the A0 evidence is
   being produced; S3's items touch CI, release and public-surface files, and the S3-09 residue sweep finds nothing in any manifest file (A36, A44):
   - S2-03 writes `goldens/ebm/freeze_manifest.json` (tool: `tools/ebm_inventory.py --write-freeze` / `--check-freeze`), sha256 per file of
     (a) every file under `src/aminx/ebm/`, and (b) exactly the consumed files: `src/aminx/utils/aa_convert.py`, `utils/safe_map.py`,
     `utils/safe_scan.py`, every file under `src/aminx/io/parsing/`, `src/aminx/parity/evidence.py`, and the **source text of the
     `SwiGLU` class** in `src/aminx/model/diffusion_mpnn.py` (hashed by `ast.get_source_segment`, because the file holds other code).
   - S2-05 appends part (c) at `F'`: the runner and driver scripts under `scripts/ebm/` that the A0 sidecars name, plus the
     sidecars themselves (these exist only after S2-04 and S2-21), plus the **environment pins (CH2-04)**: sha256 of `uv.lock` and
     sha256 of the canonical-JSON dump of the `pyproject.toml` dependency tables (`[project].dependencies`,
     `[project.optional-dependencies]`, `[dependency-groups]`, `[tool.uv]`). `--check-freeze` has two classes (R2-01). **Code parts** (a), (b) and the
     scripts and sidecars of (c) are **asserted**. The **environment pins** (the `uv.lock` and dependency-table hashes) are asserted only
     under `--check-freeze --assert-env`, which run items use when they execute from the detached `F'` worktree; in every other
     invocation, in particular S2-06 at `G`, they are only **recorded** (printed into the verification report, never failing). S2-05 also records in `ebm-frozen-baseline` the resolved `jax`, `jaxlib`,
     `xtrax` and `orbax-checkpoint` versions read from that `uv.lock`.
   - **Lock-churn protection (CH2-04).** S4-16 (base `orbax-checkpoint`, `xtrax[export]` out of base), S4-35 (xtrax bump to the S4-34
     release) and S1-17 (optional `mistypotts` dev dependency) change the lock and are not ordered against S2-05..S2-34, so S2 does not
     rely on ordering. S2-05, S2-32, S2-33 and S2-34 each execute from a **detached worktree at exactly `F'`** (`git worktree add
     --detach <dir> F'`; outputs go to an explicit out dir and are committed afterwards on the working branch), so later lock changes on
     main cannot reach A0. If a run item must use a descendant of `F'` instead (S2-34 when S2-21 is not an ancestor of `F'`), it
     asserts lock-hash equality through `--check-freeze --assert-env` and otherwise stops.
   - S2-32, S2-33 and S2-34 run `--check-freeze --assert-env` from the detached `F'` worktree and fail on any mismatch (code parts and
     environment pins), so all A0 evidence ran the same code in the same environment. S2-06 runs plain `--check-freeze` at `G` and stops
     on a **code-part** mismatch only, so the moved code is the code A0 ran. `G` is a commit on aminx main, where S4-16, S4-35 or S1-17 may
     have changed `uv.lock` after `F'` with no edge ordering them (A38), and the lock is not in the moved path set, so lock equality at `G`
     would say nothing about the moved code (A48); asserting it there would make S2-06 unexecutable for a change that never reached A0.
     A change to a **code-part** file by any other spec (for example an S1 item over `io` or `utils`) invalidates the freeze: the A0
     evidence is re-run from the first affected item. A change to `uv.lock` or the dependency tables on main **never** invalidates A0:
     A0 ran from the `F'` worktree under its own lock, whose hashes are in `ebm-frozen-baseline`. S1 needs no edge because files outside
     the manifest are irrelevant to the gate. The step-1 inventory is re-verified at `F'`, not `F`.
   **Fresh clone.** `git clone --no-local --single-branch --branch main <mpnn-repo> <scratch>/ebmx-extract` at `G` (only `main` is cloned, so no other branch, for example `hp-lattice-sanity-check`, enters the extraction; its disposition is decided by the user-decision item S2-35 and, if "port", executed by S2-36, A46, A47), the commit that includes the committed
   A0 outputs and goldens; S2-06 first runs `--check-freeze` at `G` and stops on a code-part mismatch (environment pins are recorded, not asserted).
3. **Filter.** `git filter-repo --paths-from-file ebm-paths.txt --path-rename src/aminx/ebm/:src/ebmx/
   --path-rename goldens/ebm/:goldens/ --path-rename tests/ebm/:tests/ --path-rename
   scripts/ebm/:scripts/ --path-rename scripts/engaging/submit_accuracy_runs.sbatch:scripts/slurm/submit_accuracy_runs.sbatch`
   (one `src` prefix suffices: the EBM code has never lived under another `src/` path, A31; path list = 4.2 **plus the historical paths**:
   S2-06 first runs `git log --name-status --diff-filter=R` over the EBM files and lists deleted paths with `git log --all
   --diff-filter=D --name-only`, and writes every earlier location it finds into `ebm-paths.txt`; both lists are empty today, A31). No content rewriting during the filter: contents stay authentic to their commit.
   Copy `.git/filter-repo/commit-map` to `provenance/commit-map.txt` in a later commit. Save
   `filter-repo --analyze` output next to it.
4. **Verify before anything else (the gate).** For every moved subtree and file, the git **tree/blob
   hash at `F'` in the MPNN repo equals the hash at the corresponding new commit in the clone** (renames do not
   change tree hashes, so this is an exact check, not a diff review). Both counts name their refs explicitly (`main` in the MPNN repo at `G`, and the clone's single `main`): `git rev-list --count --no-merges main -- <paths>` on each side, never `--all`, and the blob-history check walks `git log main`, so a non-main ref cannot change either number. Also: commit count equal (`git rev-list --count --no-merges main` over the full current-plus-historical path set in the MPNN repo equals the same
   count in the clone), plus a per-file blob-history check (every distinct blob hash a file had in the MPNN repo exists in the clone's
   history of its renamed path), which covers renames and merge simplification, authorship preserved, no blob larger than 5 MB, no `.praxia/subagent_outputs`
   or `.claude` paths present.
5. **Then, as ordinary commits on top** (never inside the filter): import rewrite, **path-reference rewrite** (r2 C2: the flatten
   breaks ~74 `scripts/ebm`, `tests/ebm`, `"ebm"` and `parents[...]` references in ~29 files, including sidecar `[commands]` and
   `script_path` and script docstrings; S2-08 rewrites them), seams, pyproject (S2-08),
   untracked-evidence import (S2-07), docs (S2-16).
6. Old commit messages cite PR numbers (#112, #118, #130, #132...) of `maraxen/aminx` (the repository keeps that name); `docs/HISTORY.md`
   says so. Sidecars that cite aminx SHAs stay true because the MPNN repo's history is intact; the commit-map
   resolves them to new SHAs where wanted.

### 4.8 Parity design (proving the move changed nothing)

Three tiers, cheapest first; a failing lower tier stops the line.

**Tier 0 (structure).** `goldens/model_tree_paths.json` equality (S2-08) and the fast CPU suite in a venv
with no aminx/proxide.

**Node class and hardware pin (r1 C4).** One node class **NC** is used for every arm and every number: Engaging partition `pi_so3`
with an explicit `--gres=gpu:<type>:1` (type string read from the cluster in S2-03 and written to `goldens/node_class.toml`; A30)
and **no `mit_preemptable` fallback**, so an array shard cannot land on another GPU type. The floor, tier-1 generation (A0), A0
accuracy, the tier-1 comparison (S2-13) and B1 (S2-14) all run as sbatch array tasks on NC; titanix is used only for CPU-only
suites that need no weights. Every result and stamp records `device_kind`, `jax`, `jaxlib`, `xtrax`, `orbax-checkpoint` (EBM restore uses orbax), the `uv.lock` sha256 and `XLA_FLAGS`, and
`compare_runs` and the tier-1 comparator assert device-kind and version equality between arms and fail loudly otherwise.
The orbax model (legacy tree) already lives on Engaging scratch; the `.pt` and eval assets are staged there by rsync and verified
against the S2-03 sha256 manifest, and each job checks the model leaf hash (vendored `leaf_hash.py`) before computing.

**Tier 1 (bit-level forward).** On the real checkpoint (legacy tree) on NC, a small fixed input set (LplA 1x2g/3a7r CA
coordinates from the existing `lpla_biasing_check`; one decoy; one ddG mutant set): energy, score (`-grad E`), a 5-step Langevin
trajectory with a fixed key, and a conformational-bias score. The **generator** (`scripts/ebm/goldens/gen_tier1.py`) and its
pre-registered sidecar are written in aminx by S2-21 and committed before S2-34 runs it; S2-34 produces the goldens **twice** on NC
at `F'` and commits them under `goldens/` with sha256; the comparison runs in `ebmx` (S2-13). The generator also emits a
**tier-1 swap golden** (r2 C1): the same tier-1 inputs with AF-ordered aatype fed as MPNN-ordered; S2-13 compares the ebmx
normal-input outputs against it and that comparison must fail (negative control).
Tolerances are **per quantity and relative-first**, not one absolute number. For each quantity q (energy, score, each Langevin
position array, bias score), `floor_q` is the observed max abs difference between the two generation replicates over the whole
tier-1 input set (small enough to measure in full, so there is no subset extrapolation) and `scale_q = max|golden_q|`. The
comparison passes iff `|b - g| <= atol_q + 1e-6 * |g|` elementwise, with `atol_q = max(1e-6 * scale_q, 10 * floor_q)`. The relative
term sits above float32 epsilon (1.2e-7), so it is never below machine precision for large-magnitude energies. **Stopping rule**,
applied before anything is written into a sidecar: if `floor_q / scale_q > 1e-3` for any q, the run is declared nondeterministic,
the line stops and the cause is investigated; the tolerance is never widened to fit. The sidecar with the literal `atol_q` values
is committed before the comparison runs.

**Tier 2 (the accuracy gates, paired).** Two arms, the same runner logic (r2 C6: a measured claim, not an assumed one): the runner
is written in aminx by S2-04 taking **every location (output, JAX cache, assets, orbax model, weights) by explicit argument or
env var, with no hardcoded path**; S2-10 later replaces the env/argument reads with the `ebmx._config` resolver. S2-24 adds a test
that diffs the A0 runner against the B1 runner and permits only the import prefix and resolver reads; any other delta fails and
must be explained in the B1 sidecar. The stamps' `code_manifest_sha256` legitimately differ between arms. A0 per-unit outputs live
at `outputs/ebm_benchmarks/extraction_parity/a0/units/` (tracked, inside the filter set, 4.2), recorded in `ebm-frozen-baseline`, so
`compare_runs` in the ebmx clone reads them in place. Same node class NC and container env, same `jax`/`jaxlib`/`xtrax`/`orbax-checkpoint` versions (the MPNN tree at the S2-02 merge SHA already has xtrax a11; S2-02 is the single owner of landing the pin and records the xtrax version that A0, tier 1 and B1 all use; the A0 runs are insulated from later lock churn by the detached-worktree-at-`F'` rule and the lock hashes in the freeze manifest, 4.7 step 2; ebmx's first lock pins to the A0-recorded versions, S2-22):
- **A0**: aminx at `F'`, pre-move code (the bathos slug is `aminx`, A41; A0 and floor sidecars state `--project-slug aminx` explicitly, so the wrapped invocation cannot pick up another project's slug). Its pass criterion includes the claim's own kill thresholds (decoy >= 0.75, ddG >= 0.60)
  and the guard bands below. **A0 outcome gate (r1 C6):** the A0 items (S2-05 floor, S2-32 decoy + swap, S2-33 ddG, S2-34 tier-1 goldens) only run and record; S2-25 (a user-decision item) reviews the
  recorded outcome. `pass` is accepted with a one-line record; `marginal` or `fail` stops the DAG there until the user picks one
  of: proceed with A0 as the pairing reference anyway, re-baseline in a new environment, or investigate dependency drift against
  the July baseline first. S2-06 depends on S2-25, so no extraction starts on an undecided A0.
- **B1**: `ebmx` after S2-08 to S2-13.
Comparison is per unit (133 natives, 64 assays): `max_u |rho_B1(u) - rho_A0(u)| <= TOL_UNIT` and
`|mean_B1 - mean_A0| <= TOL_AGG`. The computation is deterministic, so this is an equality check
within a measured nondeterminism floor, not a significance test with units as samples. If B1 fails,
the first diagnosis is the already-recorded July baseline vs A0 (environment drift) before blaming the move.

Floor and tolerance: the floor is measured in S2-05 by running **whole shards exactly as the full run partitions them** (same
`--shard i/k` boundaries, so batch padding, bucketing and shard composition match; r1 C12): the decoy shards containing the three
shortest and three longest natives and the ddG shard containing the longest assay, **twice**, under A0 on NC (a handful of
shard-runs, roughly an hour of GPU, against 7.7 h for a whole-population replicate; Q8 lets the user ask for the whole-population
second replicate instead). `TOL_UNIT = max(1e-4, 10 x observed floor)` and `TOL_AGG = TOL_UNIT / 4` (the 1e-4 minimum is a judgment, not derived: it sits about two orders below the +-0.02 guard bands and far
above float32-epsilon accumulation, so on a bit-deterministic GPU where the observed floor is 0 it is the operative value; r2 C9);
Spearman values are unitless
and comparable in magnitude, so one absolute tolerance is appropriate at this tier. **Stopping rule:** if the observed per-unit
floor exceeds `1e-2`, or a shard pair disagrees in unit membership or completion, the line stops for investigation instead of
widening `TOL_UNIT`. These literal numbers are committed in the B1 sidecars (own commit, S2-24) before any B1 job is submitted.
**Near-threshold misses (r2 C9).** On a bit-deterministic GPU the floor is 0, so `atol_q` and `TOL_UNIT` are the minimums, and a
benign op-order change at the `chunked_map` or `scan` seam can land just above them; Spearman is rank-based so near-tied ranks flip in
discrete steps. Therefore a miss is classified, not auto-failed: **hard fail** (stop, as above) if tier 1 exceeds `10 x atol_q` or tier 2
exceeds `10 x TOL_UNIT`; **near-threshold** (above tolerance but within those ceilings) raises a user-decision record in
`.praxia/docs/decisions/` choosing *accept with the recorded delta* or *investigate*; the gates of S2-13 and S2-14 are satisfied by
pass or by that record, never by a silent widening of the sidecar numbers (Q15).

Guard bands for A0 vs the July baseline (decoy 0.8281, ddG 0.6988) are `+-0.02` and `+-0.03` in the A0 sidecars; they are not
statistically derived and only flag gross environment drift, and the strict comparison is A0 vs B1.

**Controls (an instrument that can only pass is not a check).**
- Ownership (r2 C1): the `--inject-defect alphabet_swap` flag on the runner and its unit test are built in S2-04; the 10-native swap
  run on NC is executed by S2-32 under its own sidecar (pre-registered in S2-04) and S2-14 reuses it; the tier-1 swap input is
  produced by S2-21's generator and run by S2-34 (see Tier 1). The swap run is compared against A0 on the same 10 natives.
- Negative: a 10-native run with `--inject-defect alphabet_swap` (feeds AF-ordered aatype as MPNN-ordered, the exact defect that
  took decoy from 0.83 to 0.32) must **fail** the comparison. The margin is tied to the observed effect, not to a multiple of
  `TOL_UNIT`: the control is valid only if the swap run's mean Spearman drops by at least 0.25 against A0 on the same 10 natives
  (half the known 0.83 -> 0.32 drop; a smaller drop means the injection did not reproduce the defect, so the run is void, not
  passed) **and** `compare_runs` reports `max_unit_delta > TOL_UNIT`. If either fails the comparator is broken and the gate stops.
- Positive: the two floor replicates of A0 compared with each other must pass.
Both controls run inside the same `compare_runs` script and are recorded in the result.

**Pre-registration.** Every sidecar below is committed before the run it governs, in its own commit that precedes the job
submission, and each run item is separate from the item that commits its sidecars (r1 C16): floor + A0 decoy + A0 ddG + A0 swap-control sidecars
(S2-04), the tier-1 generation sidecar (S2-21), the tier-1 comparison sidecar (first commit inside S2-13, TOLs from the S2-05
floors), B1 decoy + B1 ddG sidecars (S2-24, TOLs filled from the measured floors; S2-14 only runs them).
**Campaign (r1 C11):** the floor, A0 and B1 sidecars name a **new** bathos campaign `ebm-extraction-parity`, created in the S2-04
pre-registration commit with the mechanism S2-03 finds for A22; they are not attached to the old campaign `aba3bfe6`. Whether
their evidence closes the old claim is the explicit user-decision item S2-29 (Q11); until then the old claim stays open.
Hypothesis in each: "the move is behaviour-preserving to within TOL"; `[outcomes]` use the numbers above,
including a `marginal` band and a residual `fail`. Pre-existing sidecars are only edited in
`[dependencies]` keys (`aminx_version` -> `ebmx_version`) and paths; a test asserts the `[outcomes]` tables of every
carried-over sidecar are byte-identical before and after the move.

**Invocation traps (all recorded earlier in this ecosystem).** Use `bth`, not `uv run bth`. Use the wrapped
form `bth run --project-slug ebmx -- uv run --no-sync python scripts/<script>.py ...` (not `--script-path`,
which uses bathos's own interpreter, and not `bash -c ...`, which makes the sidecar unresolvable and leaves the
outcome empty). The 2026-07-25/26 runs show the cost of getting this wrong (A15). Therefore S2-04's gate runs a
2-unit smoke through the real invocation and checks that `outcome` is **evaluated**, and every run is verified by
its catalog record (`bth compact`, then `bth sql "SELECT id, status, outcome, exit_code, command FROM runs WHERE id LIKE '<prefix>%'"`),
never by exit code. The warm index lags the cool tier and can keep a finished run as `running`; verify
against the cool-tier parquet before concluding anything is missing.

### 4.9 Resumable, chunked heavy runs (from the first write)

Applies to the floor, A0 (including the swap control) and B1 accuracy runs (decoy 2h33m and ddG 1h17m per arm on Engaging, from the
claim file's sacct figures; plus a 15-20 min cold XLA compile) and to tier-1 generation.

- **Unit** = one native (decoy) or one assay (ddG). Each completed unit is written atomically (temp + rename) to
  `<out>/units/<kind>/<unit_id>.json` with a completion stamp: `inputs_sha256` (canonical digest of the structure
  file bytes, the TM-score row or mutant CSV bytes, `pinned_t`, noise grid, `n_unfolded_ensemble`, seeds, caps),
  `model_leaf_hash` (4.5), `code_manifest_sha256` (hash of the package source files plus git SHA and a dirty flag),
  `env` (jax, jaxlib, xtrax, device kind, `XLA_FLAGS`), `result`, `completed_at`, `schema`.
- **Resume** reuses a unit only if `inputs_sha256`, `model_leaf_hash`, `code_manifest_sha256` and `env`
  versions all match the current run; otherwise it recomputes. The aggregate step records which units were
  reused, from which path, and their hashes, in the result JSON. The legacy `.partial.jsonl` stays as a log, never as the resume source.
- **Chunking**: `--shard i/k` partitions the sorted unit ids deterministically; each shard is its own process
  with its own timeout (an sbatch array task or a per-shard `timeout`), so a timeout or preemption loses at most one
  shard. Suggested shard sizes: decoy 10 natives (~12 min plus compile), ddG 8 assays; walltime per shard = 3x expected, never a
  single whole-run timeout. The JAX compile cache dir comes from an explicit argument or env var in A0 and from the resolver (4.6) in B1; preemptable partitions are safe by construction.
- **Retry rule (r1 C16):** a run item is done only when every unit has a valid stamp and the aggregate verifies. A preempted or
  timed-out shard is re-submitted with the identical command and resumes by skipping stamped units, with no new sidecar. An item
  whose gate is a bathos outcome is never closed on a half-run record.
- **Aggregation** is a separate cheap step that fails loudly on a missing, duplicated or stale-stamp unit.
- Local gates before cluster submission (cluster rules): L1 `--dry-run` (paths, imports, no network), L2 `--smoke`
  (<60 s on CPU), L3 a reduced-budget array task on the cluster. Never run the JAX suite locally; tests run on titanix
  or Engaging, local is for single-file or `-k` runs with capped threads.

### 4.10 MPNN-repo side: cutover (no shim, no S2 release)

S2-18 (user-gated, after parity passes and `ebmx` CI is green) does, in one PR on `aminx` main:
- delete `src/aminx/ebm/`, `tests/ebm/`, `scripts/ebm/` (including `scripts/ebm/goldens/` and the runner), `goldens/ebm/`,
  `tools/ebm_inventory.py`, the `ebm-extraction-parity` A0 sidecars and `outputs/ebm_benchmarks*` (including
  `extraction_parity/a0/`), the sbatch, and the 4 moved docs (leaving the one-file pointer); **keep** the claim file
  byte-identical. Gate (r2 C8): `git ls-files | rg -i ebm` returns only an explicit allow-list (the claim file and the pointer doc; no
  other tracked path is expected to contain "ebm");
- remove the `src/aminx/ebm/...` inventory entries (79 when read; **re-count at S2-18 time**) from `tests/parity/browser_validation_paths.json` and `"ebm"` from
  `_ROOT_PACKAGES` (same PR, otherwise CI fails on stale mappings). This file is **shared and not S2's alone** (coherence CH1-02):
  `test_browser_validation_inventory.py` fails on unmapped and on stale public top-level symbols in `src/aminx/{inference,sampling,scoring,model,host,tiling,ebm,potts}`
  and on every name in `aminx.run.__all__`, and the JSON maps `aminx.run::build_run_spec`, `RunSpec`, `RunSpecification` and the other `aminx.run` names.
  S1 (it removes `build_run_spec`, adds `as_run_spec` and the nested config classes), S4 (new public defs under scoring or run) and S5 (the
  `aminx serve` runtime) therefore also edit this JSON and, if the baseline file counts them, the internal-symbol baseline; none of those
  edits is ordered against S2-18, so S2-18's removal is a **textual merge** with them: rebase onto main first, re-count the `ebm` entries, delete exactly those, and run the
  inventory test (a merge that drops a foreign entry fails it as unmapped, one that keeps a stale `ebm` entry fails it as stale);
- **consumer and branch scan (coherence CH1-03).** S2-18 does not rely on the CHANGELOG pointer alone: before the PR it runs a one-time
  `rg 'aminx\.ebm'` over `mpnn_ext`, `asr`, `hautespout` (main and its WIP worktrees) and `tev_design`, and records the result. **A hit stops
  the item** (R2-04): S2-18 records an owner decision (migrate that consumer, choose Q4 option (b), or accept the break) before the PR is opened;
  it never migrates a consumer itself. The open branch `hp-lattice-sanity-check` (tip file list read in A47, disposition owned by S2-35, port by
  S2-36) adds `src/aminx/ebm/hp_energy.py` and is stranded by the deletion unless it was merged before `F`, abandoned or ported; S2-18
  states the recorded disposition in the CHANGELOG pointer;
- drop deps made unused by the removal only after `rg` confirms (the `benchmark` extra's `torch` and `prody` serve other code; do not assume);
- add a CHANGELOG entry: `aminx.ebm` moved to `ebmx`, with the install command recorded by S2-01 (`ebm-install-command`) and the
  statement that the module is removed, not forwarded.

**No shim, no release in S2.** aminx has no `gone_modules.toml`, alias table or meta-path finder today (A40), and the no-rename revision of
S3 retired the `aminx.ebm` forwarding row together with the shim. `aminx.ebm` is importable from `v0.2.0a1` and is in every published
alpha through `0.2.0a3` (A19), so after the removal `import aminx.ebm` raises a plain `ModuleNotFoundError` and the CHANGELOG pointer is the
whole user-facing notice (Q4 offers a five-line in-tree stub that raises an actionable `ImportError` instead; the CHANGELOG pointer also names the stranded `hp-lattice-sanity-check` branch and its disposition). S2 cuts no release (A43): the
removal ships in the next release cut by any item, which S3-12 guards (tag equals the built version, the PyPI version still unpublished,
dry run) and S3's train rule orders (S5-52 first, then S1-25). A release that precedes S2-18 simply still contains EBM; nothing breaks.
PyPI `aminx` stays the MPNN package throughout (A42): the hub is a separate project (working slug `aminx-hub`) and takes no part in the
cutover.

### 4.11 Ordering versus S1, S3, S4 and S5 (restated for the no-rename world)

Earlier revisions argued "rename first (option R) versus EBM first (option E)" and chose R: S2-03 waited for the tagged rename release
(S3-13), S2-18 for the codemod (S3-07), the EBM tree was cut from the renamed `src/` directory and a second `--path-rename` bridged two
path eras. With no rename the question is void (S1 Q10, S3 Q4 and S2 Q7 no longer exist). What replaces it:

- **S3.** No S3 item is a prerequisite of an S2 item, and no S3 item depends on an S2 item (S3-25 and S3-20, the old S3-to-S2 edges, are
  retired). The extraction runs on aminx main as it stands plus the #174 pin: `S2-03` `depends_on` `S2-02` and the hp-lattice decision S2-35 (no S3 edge), and `S2-18` has no S3
  edge. The cost the rename imposed (a codemod over ~6.6k src plus ~5.4k test LOC that later leave, a double-era filter, A0 waiting for a
  tagged release) is gone. S3's interactions are conventions, not edges: (a) any release carrying the removal goes through S3-12's guard
  and train rule (4.10); (b) S3-09, S3-12 and S3-21 edit CI, release, README, CHANGELOG and pyproject metadata, which the freeze manifest
  does not hash (A36, A44), and the CHANGELOG is shared with S2-18's pointer (a textual merge); (c) the stdlib stale-name checker (S3-24)
  could later be vendored into ebmx CI as an optional adoption, which this spec does not require because S2-28's `rg` and import-blocker
  gates already enforce the same property.
- **S1** (spec unification) is independent: EBM never touches `RunSpec`/`RunSpecification`. No edge either way. S1 items that edit a
  file in the freeze manifest (4.7 step 2) invalidate the freeze and are caught by `--check-freeze`, not by an edge; S1's own text adds
  S1-11 -> S2-06 if its rewrite touches the manifest paths. **Owed to S1 (not edited here, R2-04):** S1-28 bumps hautespout's vendored aminx
  to a main-line SHA, which lacks the `hp_*` modules that exist only on `hp-lattice-sanity-check`; S1-28 should gain `depends_on` S2-35 and its
  gate should state whether the bump target contains `hp_*` (merge-before-F) or the WIP loses them knowingly (abandon, or port to `ebmx` by S2-36).
  S2 cannot carry the edge itself because S2 depends on no S1 id.
- **S4/S5 and the hub.** No S2 item depends on them. The hub is a separate project (working slug `aminx-hub`; repo name, PyPI policy and
  domain are the user's S3-01 decision) and does not touch the extraction. The EBM descriptor facts in section 6 are inputs S4 and the hub
  catalog can use; adopting S4's manifest schema is a later, separate item the DAG integrator adds once S4's item ids exist.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| History rewrite loses or corrupts history (the demistify incident) | the MPNN repo is never rewritten; filter on a `--no-local` clone; verified full-ref bundle outside the repo first (S2-03); tree/blob-hash equality gate for every moved path plus a historical-path enumeration (S2-06); commit-map retained |
| Orbax restore silently mis-assigns shape-compatible leaves after the SwiGLU copy | keep field names (A5); frozen tree-path + shape listing test (S2-26, golden from S2-21); tier-1 bit-level goldens on the real checkpoint (S2-13) |
| Move-parity gate confounded by dependency drift (baseline JSONs carry no env, A16) | arm A0 is pre-move code in the same env as B1 (4.8); July baseline only a guard band |
| Nondeterminism makes exact comparison meaningless | measured floor drives TOL (A17), measured with the full run's shard decomposition, with a stopping rule instead of tolerance widening; positive control (replicate vs replicate) and effect-tied negative control (alphabet swap, mean drop >= 0.25) in the comparator |
| Cross-GPU-class comparison gives false failures or a vacuous tolerance | single node class NC pinned by `--gres` type with no preemptable fallback; device kind and library versions asserted equal between arms (4.8; A30) |
| Current converter does not reproduce the legacy orbax tree (A28) | legacy tree (hash recorded in S2-03) is the pairing reference; mismatch is recorded and filed as debt, not a stall |
| A0 is marginal or fails and the DAG proceeds anyway | S2-25 user-decision item sits between S2-05 and S2-06 |
| The EBM removal races another release for a PyPI version, or a second shim appears (coherence C2) | S2 has no release item and no shim (A40, A43); the removal rides the next release cut, guarded by S3-12 (tag equals the built version, PyPI version unpublished, dry run) and ordered by S3's train rule; S3-28 checks release ordering; S2-18 only removes code and adds a CHANGELOG pointer |
| Pre-move goldens have no owner | S2-21 owns the generators and goldens, outputs under `goldens/ebm/` inside the filter path set (4.2) |
| Tracked bathos run silently has no evaluated outcome (happened 260725-26) | S2-04 gate: 2-unit smoke through the real wrapped invocation with a record check; every heavy run verified by catalog record after `bth compact`, with the cool-tier parquet as fallback |
| Untracked evidence lost (9 files + all `*.partial.jsonl` exist only in the main checkout's root) | S2-03 sha256 inventory and tarball before anything; S2-07 import; S2-17 pushes it to a remote; deletion only in S2-20, a user decision that re-hashes against a fresh clone of the remote |
| Bathos claim anchor (path + SHA256) breaks when the file moves | aminx keeps the claim byte-identical; new claim in `ebmx`; lineage document links old run ids/campaign `aba3bfe6` (A21, A22) |
| A0 and floor runs land under the wrong bathos project | the slug stays `aminx` (`.bth.toml:2`, A41): A0 and floor sidecars state `--project-slug aminx` explicitly; `ebmx` takes its own slug and id via `bth init`; no bathos command moves runs between projects (A21), so the lineage document (S2-15) links old run ids and campaign `aba3bfe6` instead |
| xtrax drops the `SafeMap` alias next release | S2-08 renames to `ChunkedMap`; S2-11 gates on `-W error::DeprecationWarning`; pin `<0.5` |
| Hidden transitive deps vanish (einops reaches aminx only via proxide/jax-md) | all of `einops`, `scipy`, `orbax-checkpoint`, `pandas` declared; clean-venv install test with no extras (A10, A12) |
| External users `import aminx.ebm` (alpha releases since 2026-09-10) | CHANGELOG pointer with the install command from S2-01 (S2-18); alpha status keeps the audience small; Q4 offers a five-line in-tree stub that raises an actionable `ImportError`; no shim (S3 retired it) |
| MPNN-repo CI breaks on stale inventory mappings | the inventory edits are in the same PR as the deletion; gate runs that test |
| Upstream licence unknown; trunk derived from Boltz-1 | S2-01 ADR before publishing; keep the repo private until resolved; `NOTICE` with upstream attribution |
| New edges violating the ecosystem partition | no aminx edge; proxide optional; alphex dev-only; xtrax is core ecosystem; enforced by an import-blocker test |
| Project name `ebmx` collides with the checkpoint nickname "ProteinEBM-x" | always write the checkpoint by filename `model_6_expert_frozen_1m_md`; alternatives in Q1 |
| The hub is assumed to take the name `aminx` (it does not) | the hub is a separate project; S3-24's reserved-name mode forbids `aminx` as its PyPI, import or package.json name (A42); S2 names the hub only as the working slug `aminx-hub`, whose final form is the user's S3-01 decision |
| GPU spend and cold-compile time | ~2 x (2h33m + 1h17m) per full A0/B1 plus floors; chunked shards, persistent compile cache, preemption-safe; user OK requested (Q8) |
| Heavy suites on this machine | all gates run on titanix/Engaging; local runs are single-file/`-k` with thread caps; never a whole JAX suite locally |
| Blackwell (SM120) XLA autotune hang | carry the `XLA_FLAGS=--xla_gpu_shard_autotuning=false` hostname-keyed block from the sbatch into `scripts/slurm/` unchanged |
| Other specs' items change files near the frozen code while A0 evidence is being produced (S1 over `io`/`utils`, the S3-09 sweep, S4 lock changes; coherence C3) | S2-03 depends on S2-02 and S2-35 only; file-level freeze manifest: code parts asserted by S2-32/33/34 and S2-06, lock-hash pins asserted only by the run items from the detached worktree at `F'` and recorded (not asserted) at `G`, so lock churn on main cannot stop S2-06 or invalidate A0 (4.7 step 2, A38, A48); A44 shows the S3-09 sweep has nothing to change in the manifest files today; A36 stays deferred to `--check-freeze` |
| Converter does not reproduce the legacy orbax tree (A28) but cutover proceeds | S2-31 user decision (Q14) gates S2-17 and S2-18: fix converter and re-run tier 1, or declare the legacy tree the only supported checkpoint |
| Flatten breaks path references (sidecars, docstrings) | S2-08 rewrites them; gate `rg 'scripts/ebm|tests/ebm'` empty outside provenance plus a test that every sidecar `script_path` and command resolves |
| The branch `hp-lattice-sanity-check` and the hautespout WIP worktree lose `hp_*` files at cutover or at S1-28's vendored bump (R2-04) | S2-35 user decision (merge before `F`, abandon, port) with the tip file list; S2-36 ports conditionally; S2-18 stops on a consumer `aminx.ebm` hit; S1-28 owes an edge to S2-35 (4.11) |
| A0 evidence ran code other than what moved | single `F'` in `ebm-frozen-baseline`; `--check-freeze` against the freeze manifest (EBM tree plus exactly the consumed files, 4.7 step 2): code parts in S2-32/33/34 and S2-06, environment pins additionally in S2-32/33/34 via `--assert-env` |
| Rollback | S2-18 is a revert of one MPNN-repo PR (the CHANGELOG pointer reverts with it); until S2-18 the MPNN tree is untouched by S2, so S2-02..S2-17 roll back by deleting the new repo; the bundle from S2-03 restores everything |
| The xtrax 0.4.0a11 pin (draft PR #174) has two owners with contradictory preconditions (coherence C7) | S2-02 is the single owner of landing it (user-gated merge; S4-16 `depends_on` S2-02 per S4 and S3 section 2.1); S4-16 only resolves the orbax conflict on top; S2-02 records the xtrax version that A0, tier 1 and B1 all run on (A37) |

## 6. Interfaces

### Provides

| Contract | Provided by | Description | Consumers |
|---|---|---|---|
| `ebm-package` | S2-08 (import-isolated after S2-28, stable after S2-14) | importable `ebmx`: `model.ProteinEBMModel`, `checkpoint.load_pytorch_checkpoint`, `dispatch.score_*`, `langevin*`, `structure_prediction`, `conformational_biasing`, `ddg_stability`, `decoy_ranking`, `training`; public names unchanged from `aminx.ebm.*` | end users; the hub catalog entry (S5) |
| `ebm-frozen-baseline` | S2-05, S2-32, S2-33, S2-34, accepted by S2-25 | `F'` SHA, A0 output location, A0 run ids, per-unit stamped JSONs, floor measurement, tier-1 goldens, node class NC record, all sha256-manifested | S2-13, S2-14 (no other spec has an edge on it) |
| `mpnn-tree-without-ebm` | S2-18 | the MPNN tree (aminx) without EBM, inventory edited, claim file retained byte-identical, CHANGELOG pointer added | none (the S3-25 and S3-20 edges are retired); informational for S5's catalog |
| `ebm-install-command` | S2-01 | channel and exact install command for `ebmx` (private index, git URL or PyPI) | S2-18 (quotes it in the CHANGELOG pointer); S5 (catalog entry link) |
| `xtrax-a11-pin-landed` | S2-02 | merged SHA of #174 and the xtrax version A0/tier 1/B1 use | S4-16 (depends_on S2-02), S2-03 |
| `ebm-model-facts` | S2-16 | descriptor facts for catalog/manifest use: inputs `aatype (N,)` int, `coords (N,3)` nm scaled by `coordinate_scaling=0.1`, `mask (N,)`, `t` in [0,1]; outputs scalar energy and `-grad` score; executors: JAX only; browser: out (P24); checkpoint by filename and sha256 | S4 (manifest), S5 (hub catalog entry) |
| `config-resolver-pattern` | S2-10 | the explicit > env > `[tool.<name>]` > `~/.config/<name>/config.toml` resolver with `config_source()` | any repo, including S4/S5 adopters |

### Consumes

| Contract | Providing spec | Needed for | Blocking? |
|---|---|---|---|
| `S3:aminx-identity` (aminx keeps its name, repo, PyPI project, HF repo and bathos slug `aminx`) | S3 | A0 and floor sidecars state `--project-slug aminx`; the `maraxen/aminx` PR-number note in `HISTORY.md`; the CHANGELOG pointer | no: a contract about things that do not change |
| `S3:release-guard` (S3-12: build-job version guard, dry run, next-free-alpha train rule) | S3 | the next release cut that carries the EBM removal | no: S2 has no release item (A43); the rule binds whoever cuts the release |
| `model-manifest-schema` (ports, manifests) | S4 | later adoption: describe EBM in the shared manifest | no; adoption is a follow-up item added at DAG merge |
| `catalog-entry-format` | S5 | later listing of EBM in the hub catalog (link, facts, `executors: [jax]`) | no |
| none | S1 | EBM does not use `RunSpec` / `RunSpecification` (A2); an S1 edit to a freeze-manifest file is caught by `--check-freeze` (4.7 step 2) | no |
| external: `xtrax>=0.4.0a11` core (`tiling`, `transforms.chunked_map`, `checkpoint`, `engine`, `training`, `data.module`) | xtrax (A9) | all of `ebmx` | yes, S2-02 (single owner of landing the pin; S4-16 does not land it) |
| external: upstream ProteinEBM checkpoint, eval data, PyTorch reference | `jproney/ProteinEBM` | weights lock, parity gates, benchmarks | yes, resolved by config (4.6) |

## 7. Verification gates per item

Heavy runs always go to titanix or Engaging. **Weight-dependent gates (anything with `requires_weights`, A0, tier 1, B1, converter,
leaf hash) run on Engaging node class NC via sbatch**, because the orbax model lives there and device kind must match across
arms (4.8); titanix runs CPU-only suites that need no weights. Template for titanix (sandboxed shells need
`dangerouslyDisableSandbox` for `ssh`; rsync the worktree with anchored excludes; `uv` login shell):
`ssh titanix 'cd <remote_root> && uv run --extra tests pytest <path> -q -m "not slow"'`. No gate runs on the Engaging login node;
cluster work goes through `sbatch`/`myxcel`.

| Item | How done is proven |
|---|---|
| S2-01 | ADR in `.praxia/docs/decisions/` records name, licence, visibility, the permanent clone location and the **distribution channel with the exact install command** (private index, git URL or PyPI), with PyPI + GitHub availability probes (run by the executor; not done in this spec) |
| S2-02 | the owner has merged `maraxen/aminx#174` (executor takes no merge action); merged SHA recorded; `uv lock --check` clean on titanix at that SHA; the locked xtrax version read from `uv.lock` at that SHA is recorded as the version for the A0/tier-1/B1 environment (C7) |
| S2-03 | `git bundle verify` ok; `goldens/ebm/freeze_manifest.json` part (a)+(b) written and `tools/ebm_inventory.py --check-freeze` passes at the freeze SHA; the S2-35 disposition is on record and the non-main-ref inventory (tip SHA, `git ls-tree` file list) matches it; `tools/ebm_inventory.py --verify` re-hashes 100% of manifest entries (tracked files, the 39 root files with their main-checkout paths, `real_accuracy/` extras, eval assets, the `.pt` sha256, the orbax **legacy-tree leaf hash** from `scripts/ebm/leaf_hash.py`); the untracked-evidence tarball exists and its sha256 is in the manifest; the `bth` scratch-catalog probe settles A22 and the campaign mechanism is written down; node class NC and its `--gres` type are recorded in `goldens/node_class.toml` (A30); bathos run ids and campaign `aba3bfe6` listed |
| S2-21 | goldens under `goldens/ebm/` with `MANIFEST.sha256`: `model_tree_paths.json` and the `parse_structure` goldens (1x2g, 3a7r, 5 decoys) regenerate byte-identically twice on titanix CPU; the tier-1 generator and its sidecar are committed in their own commit before any GPU run; all outputs sit inside the 4.2 path set |
| S2-04 | titanix `pytest tests/ebm/test_accuracy_runner.py`: kill-and-resume skips completed units, a changed input byte or model hash forces recompute, shard partition is disjoint and complete, aggregate fails on a missing unit, every location is an explicit argument or env var (no hardcoded path), `--inject-defect alphabet_swap` swaps AF/MPNN order (unit test), `compare_runs` unit tests pass including the device-kind equality assertion and both controls on synthetic data, `--dry-run` and `--smoke` pass; a 2-unit smoke through `bth run --project-slug aminx -- uv run --no-sync python ...` (stated explicitly) ends with an **evaluated** outcome; floor, A0 decoy, A0 ddG and A0 swap-control sidecars (campaign `ebm-extraction-parity`) committed in their own commit |
| S2-05 | on NC from a detached worktree at `F'`: freeze manifest part (c) (runner scripts, sidecars, `uv.lock` sha256, dependency-table sha256) appended and `--check-freeze` passes; the stamps record `orbax-checkpoint` and the lock hash; floor shards run twice with the stopping rule evaluated; floor catalog record has an evaluated `outcome`; `F'` recorded in `ebm-frozen-baseline`; floors committed with sha256; a preempted shard was re-submitted by the retry rule, not a new sidecar |
| S2-30 | recorded user approval of the GPU budget (Q8) and node class NC in `.praxia/docs/decisions/` before any GPU job is submitted |
| S2-32 | on NC from a detached worktree at `F'` (`--check-freeze --assert-env` incl. lock hashes passes, 4.7 step 2): 133 decoy units stamped, aggregate verified, evaluated outcome; the 10-native `alphabet_swap` run completed under its own sidecar with mean Spearman drop >= 0.25 against A0 on the same 10 natives; outputs committed under `outputs/ebm_benchmarks/extraction_parity/a0/` |
| S2-33 | on NC from a detached worktree at `F'` (`--check-freeze --assert-env` incl. lock hashes passes): 64 ddG assays stamped, aggregate verified, evaluated outcome, guard band reported, outputs committed under the A0 path |
| S2-34 | on NC: tier-1 goldens (normal and swap inputs) generated twice from a detached worktree at `F'` (or a descendant passing `--check-freeze --assert-env`, incl. lock hashes), per-quantity floors and `atol_q` committed with sha256, stopping rule evaluated |
| S2-31 | decision record: `converter_matches_legacy` value read from `weights.lock.toml`; if false, the user's choice (fix converter and re-run tier 1 on the converted tree, or legacy tree as sole supported checkpoint) |
| S2-25 | recorded decision: A0 outcome `pass` accepted, or the user's choice (proceed, re-baseline, investigate) for `marginal`/`fail`, written to `.praxia/docs/decisions/` |
| S2-35 | decision record in `.praxia/docs/decisions/` names merge-before-F, abandon or port, with the tip SHA and the `git ls-tree -r --name-only` listing of the tip (A47: `src/aminx/ebm/hp_energy.py`, `tests/ebm/test_hp_energy.py`, `scripts/validate/hp_lattice_sanity.py` and `.bth.toml`) and, separately, any `hp_*` file the hautespout WIP vendored copy has beyond the tip (`hp_coarse_sampler`, `hp_windowing`) with its owner; for merge-before-F the files are on main before S2-03 starts and 4.2, the freeze manifest part (a) and `ebm-paths.txt` include them; the S1-28 bump target is stated |
| S2-36 | closes not-applicable (recorded in the item) unless S2-35 says port; if port: ebmx CPU unit test of the ported files passes on titanix, each ported file carries a header with the source tip SHA and sha256, no `aminx` import (S2-28 gate), ebmx CI green |
| S2-06 | `tools/verify_extraction.sh <mpnn-repo> <G> <clone>` (created in the ebmx clone): `--check-freeze` passes at `G` for the code parts (file-level manifest over the EBM tree and consumed files); the lock and dependency-table hashes are printed in the report and not asserted at `G` (R2-01, A48), tree/blob hash equality for every moved path, no-merges commit counts over current plus historical paths (refs named: `main` on both sides, not `--all`), per-file blob-history check, authorship, size and forbidden-path scans, commit-map present; `ebm-paths.txt` includes the rename and delete enumeration output (empty if A31 holds) and the filter used the single `src/aminx/ebm` path-rename; two fresh clones give identical new SHAs; permanent clone exists at the S2-01 location |
| S2-22 | `bth init` produced slug `ebmx` and a new project id (checked in `.bth.toml` and a catalog row); `uv lock --check` clean for the pyproject skeleton on titanix; the first `uv.lock` resolves `jax`, `jaxlib`, `xtrax` and `orbax-checkpoint` to exactly the versions recorded in `ebm-frozen-baseline` by S2-05 (a check script diffs them); local-branch plus anchored-rsync procedure documented in `CLAUDE.md` |
| S2-07 | `MANIFEST.sha256` verifies; originals in the main checkout untouched; tracked `outputs/` files unchanged |
| S2-08 | `rg 'aminx\.ebm' src tests scripts` is empty; `rg 'scripts/ebm|tests/ebm|goldens/ebm' .` is empty outside `docs/HISTORY.md` and provenance lines; a test asserts every sidecar `script_path` and `[commands]` entry resolves to an existing file; `ruff check` clean; the SafeMap to ChunkedMap rename is done and test imports use public `xtrax.tiling`; `pytest --collect-only` succeeds in a temporary scaffold env on titanix that still has aminx installed (explicitly not the final gate) |
| S2-26 | `_layers.py` (SwiGLU copy, fields `w_gate`, `w_val`, `w_out`) and `_tiling.py` in place; `goldens/model_tree_paths.json` equality test passes; `-k "swiglu or plan or tiling"` CPU tests green on titanix |
| S2-27 | `_alphabet.py`, `_structure.py` and vendored `scripts/_evidence.py` (provenance header) in place; all five `parse_structure` sites (A29) and the `aa_convert`/`parity.evidence` sites rewritten; tests needing proxide guarded by `importorskip` and the `structure` extra (A32); alphex conformance test passes (A23) or is skipped with a recorded reason |
| S2-28 | clean venv with no aminx/proxide and no xtrax extras: `pytest tests -m "not slow and not requires_weights"` green on titanix; `rg '\baminx\b' src` returns only allow-listed provenance/NOTICE lines; `import ebmx` under a meta-path blocker forbidding `aminx`, `proxide` passes; `[outcomes]`-identical test for carried-over sidecars; **ebmx-slug smoke**: a 2-unit `--smoke` run through `bth run --project-slug ebmx -- uv run --no-sync python ...` ends with an evaluated outcome (verified in the catalog record) |
| S2-09 | adapter output equals the S2-21 goldens (CA coords, residue index, aatype) for 1x2g, 3a7r and 5 decoys, exactly; missing proxide gives the actionable error |
| S2-10 | precedence tests for all five layers, malformed-file failure, `none` disables, `config_source`; grep for `/tmp/proteinebm`, `~/repos/ProteinEBM`, `proteinebm_bench_assets`, `/orcd/scratch`, `/home/maarxaru` in `src scripts` is empty |
| S2-11 | suite green under `-W error::DeprecationWarning`; `xtrax<0.5` pin present; grep for `SafeMap`/`safe_map` outside `_tiling.py` is empty |
| S2-12 | on NC: loader refuses a wrong-sha256 file (negative test); converter run from the `.pt` compared with the S2-03 legacy leaf hash and the result recorded (equal, or the list of differing leaves plus a filed debt item); `weights.lock.toml` records both hashes and `converter_matches_legacy`; a mismatch does not fail the item |
| S2-13 | on NC via sbatch: the vendored `leaf_hash.py` equals its recorded source sha256; model leaf hash equals the legacy hash in `weights.lock.toml`; the pre-registered tier-1 comparison (sidecar commit precedes the job) is within per-quantity `atol_q` with device-kind equality asserted (or a near-threshold decision record exists, 4.8); negative control (comparison against the S2-34 tier-1 swap golden) fails; positive control (replicate) passes; bathos outcome evaluated |
| S2-24 | B1 decoy and B1 ddG sidecars committed in their own commit with literal `TOL_UNIT`/`TOL_AGG` copied from the S2-05 floor record; the A0-vs-B1 runner-diff test (only import prefix and resolver reads differ) passes; `git log` shows this commit precedes any B1 job submission |
| S2-14 | on NC: bathos outcome `pass` for decoy (133/133) and ddG (64/64), all units stamped and the aggregate verified (or a near-threshold decision record, 4.8); `compare_runs` output: device kind equal to A0, `max_unit_delta <= TOL_UNIT`, `agg_delta <= TOL_AGG`, negative control valid (reusing the S2-32 swap run: mean drop >= 0.25 and flagged) and positive control passing; reused-unit report present |
| S2-15 | `bth claim validate` ok for the new claim and for the untouched old claim; old claim file sha256 equals the recorded one; lineage doc lists old run ids, campaign `aba3bfe6`, the new campaign and the `commit-map` location |
| S2-29 | recorded user decision on whether A0/B1 evidence closes campaign `aba3bfe6` (Q11) and, if yes, the `bth` command that does it executed and checked in the catalog; the old claim file sha256 is unchanged either way |
| S2-16 | link check passes; `docs` index check passes; `NOTICE` names upstream ProteinEBM and Boltz-1; `HISTORY.md` explains PR numbers |
| S2-17 | S2-31 decision recorded; remote created; the full local branch including `untracked_import/` is pushed; CI green on a PR: unit tests (CPU, fast markers, minimal install), lint, import-isolation job; documented titanix recipe produces the same result as CI; the S2-01 install command works from the pushed remote |
| S2-18 | on titanix: `pytest tests/parity/test_browser_validation_inventory.py` green after the `ebm` entries were re-counted and removed on top of any S1, S4 or S5 edits to the same JSON; one-time `rg 'aminx\.ebm'` over `mpnn_ext`, `asr`, `hautespout` (main and WIP) and `tev_design` recorded, and any hit halts the item until an owner decision is recorded; S2-36 is closed (ported or not applicable); the CHANGELOG pointer names the S2-35 `hp-lattice-sanity-check` disposition; `rg 'aminx\.ebm' src scripts tests` is empty (CHANGELOG and docs excluded); `git ls-files | rg -i ebm` matches only the allow-list in 4.10 (claim file, pointer doc); claim file sha256 unchanged; the CHANGELOG pointer names `ebmx` and the S2-01 install command; the `ebm_inventory` copy and A0 sidecars are gone; no shim, release-workflow, version or pyproject dependency edit in the diff |
| S2-20 | a fresh `git clone` from the S2-17 remote re-hashes 100% of `untracked_import/` against the manifest; the S2-03 tarball verifies; then (user-approved) the root `ebm_benchmarks*` directories are removed from the main checkout (`/home/marielle/projects/aminx`; the path is read from the S2-03 manifest, not assumed) and are absent afterwards; praxia debt #2369 status `completed` with link to this spec |

## 8. Open questions for the user

1. **Q1: project name.** Candidates: (a) `ebmx`, short and ecosystem-consistent (aminx, xtrax, alphex); mild clash with the
   upstream checkpoint nickname "ProteinEBM-x". (b) `proteinebm-jax` (import `proteinebm_jax`): unambiguous provenance and
   searchable, least ecosystem-like. (c) `proteinxebm`, spelling out domain and method (longer than the ecosystem's short `<noun>x` names). All three need a
   PyPI and GitHub availability check (not done here). **Recommendation: `ebmx`**, always naming the checkpoint by filename.
2. **Q2: licence and attribution.** The local `jproney/ProteinEBM` checkout has no `LICENSE` (the GitHub page was not
   checked), and the trunk is Boltz-1-derived. **Recommendation:** do not publish publicly until the executor reads upstream's
   licence on GitHub; if permissive, license new code MIT and add `NOTICE`; if absent, ask upstream for a licence statement first.
3. **Q3: remote, visibility and install channel.** **Recommendation:** private `maraxen/ebmx` until Q2 resolves, installed by git URL; the CHANGELOG pointer then quotes a git-URL install command (S2-01 records the channel and exact command; S2-18 uses it). Until S2-17 creates it, the ebmx items are commits on a local branch synced by rsync (4.7).
4. **Q4: telling `aminx.ebm` users after the cutover.** There is no shim any more (S3's no-rename revision retired it), so the options are:
   (a) a CHANGELOG pointer only, after which `import aminx.ebm` raises `ModuleNotFoundError` (S2-18's default); (b) additionally a
   five-line in-tree `src/aminx/ebm/__init__.py` that raises an `ImportError` naming `ebmx` and the S2-01 install command (one more
   path in the `rg -i ebm` allow-list, 4.10); (c) removal with no notice. **Recommendation: (a)**: `aminx.ebm` has been public since
   `v0.2.0a1` (2026-09-10) but the alpha audience is small and S3 already settled "no shim"; choose (b) if you know of outside users, including the hautespout WIP worktree that vendors an aminx with `src/aminx/ebm/hp_energy.py` (branch `hp-lattice-sanity-check`, A46); its disposition (merge before the freeze, abandon, or port into `ebmx` by S2-36 after S2-17) is a user decision at S2-35. A
   related choice, whether the removal gets a dedicated alpha, is answered no (A43): it rides the next release cut.
5. **Q5: alphex as a runtime dependency of `ebmx`.** **Recommendation:** dev-only conformance test now (alphex D4 pattern, no new
   runtime edge); take the runtime edge later, after the partition work completes.
6. **Q6: SwiGLU.** **Recommendation:** verbatim copy now; file a separate debt item for a bias-free `Transition` (needs a re-conversion
   and its own parity run, because it changes the orbax tree). Alternative (a shared layer in xtrax) is not recommended for 27 lines.
7. **Q7: retired (no rename).** The order relative to S3's rename (the question shared with S1 Q10 and S3 Q4) no longer exists: nothing is
   renamed, S2-03 and S2-18 carry no S3 edge, and the remaining ordering conventions are in 4.11. The number is kept so Q8-Q15 and the
   references to them stay stable. Nothing to answer.
8. **Q8: GPU budget and node class.** About 7.7 GPU-hours for the two full arms (2 x (2h33m + 1h17m)), plus the shard-level floor (about an
   hour), tier-1 generation and comparison (small) and a one-time 15-20 min compile; per-run figures come from the claim file's sacct
   notes. All of it runs on one node class NC: `pi_so3` with an explicit `--gres` GPU type and no `mit_preemptable` fallback (4.8; the type
   string is read from the cluster in S2-03). **Recommendation:** approve on those terms, chunked and preemption-safe. The approval is enforced in the DAG by the user-decision item S2-30, which every GPU item depends on (r2 C3). Optional upgrade (about +7.7 h):
   a whole-population second A0 replicate instead of the shard-level floor.
9. **Q9: tracked evidence in the new repo.** **Recommendation:** keep `outputs/ebm_benchmarks*` tracked verbatim (1.4 MB) and import the
   untracked extras under `untracked_import/` (all small JSON/JSONL).
10. **Q10: publishing the converted orbax weights.** Currently they exist only on Engaging scratch. **Recommendation:** do not redistribute until Q2
    is settled; ship the converter and `weights.lock.toml` so anyone can reproduce the tree from the upstream `.pt`.
11. **Q11: close the old claim with A0/B1.** The new runs go under a new campaign `ebm-extraction-parity` (4.8), and their sidecars carry the claim's kill
    thresholds and `claim_discriminates`, so a passing A0 supplies the missing evaluated confirmatory re-run for campaign `aba3bfe6`. Whether and
    how that evidence is attached to the old campaign depends on the A22 probe (S2-03). **Recommendation:** yes, but only through the explicit
    user-decision item S2-29; the old claim stays open and byte-identical until you approve.
12. **Q12: A0 outcome handling.** S2-25 stops the DAG on a `marginal` or `fail` A0. **Recommendation:** on `fail`, investigate dependency drift against the
    July baseline before proceeding; on `marginal`, proceed with A0 as the reference (the A0-vs-B1 comparison is the move gate, not A0 vs the paper).
13. **Q13: untracked evidence disposition.** S2-20 is user-gated. **Recommendation:** after the pushed copy re-hashes clean, move the root directories into
    the S2-03 tarball and delete the directories; keep the tarball outside every repo.
14. **Q14: converter mismatch (r2 C5).** If S2-12 finds `converter_matches_legacy = false`, does the cutover proceed with the legacy tree as the only
    supported checkpoint, or is the converter fixed first (and tier 1 re-run on the converted tree)? **Recommendation:** fix the converter first if the
    differing leaves are few and explainable; otherwise ship the legacy-tree path with the limitation documented. Decided in S2-31.
15. **Q15: near-threshold parity misses (r2 C9).** A tier-1 or tier-2 miss above tolerance but within 10x is a user decision (accept with the recorded delta, or
    investigate). **Recommendation:** accept only if the delta is traceable to the `chunked_map`/`scan` seam op order and the A0-vs-B1 mean difference stays within TOL_AGG.

## 9. Backlog items

Order of work is expressed only through `depends_on`. The ebmx items run as commit ranges on a local branch until S2-17 creates the
remote (4.7). Item ids are stable across revisions; round 1 added S2-21 to S2-29, round 2 added S2-30 to S2-34 and split S2-05; no-rename coherence r2 added S2-35 and S2-36; coherence round 1 retired S2-19 (the id is not reused). The no-rename revision added and deleted no item: it dropped every edge to an S3
id (S3-05, S3-07 and S3-13 had no replacement) and re-pointed titles, gates and `repo` to `aminx`; S2 now has no edge to any S3 id.

```toml
[[item]]
id = "S2-01"
title = "Decide EBM project name, licence, remote visibility, install channel and permanent clone location; record the ADR (resolves Q1-Q3)"
repo = "aminx"
size = "S"
depends_on = []
gate = "ADR in .praxia/docs/decisions/ records name, licence, visibility, clone location and the distribution channel with exact install command, with PyPI and GitHub availability probes and upstream licence evidence"
user_decision = true

[[item]]
id = "S2-02"
title = "Land the xtrax 0.4.0a11 pin (maraxen/aminx#174) as its single owner: confirm the owner merged it, record the merged SHA and the locked xtrax version used by A0, tier 1 and B1 (executor does not merge; S4-16 depends on this item)"
repo = "aminx"
size = "S"
depends_on = []
gate = "owner-merged SHA of #174 recorded, uv lock --check is clean on titanix at that SHA, and the locked xtrax version is recorded for the A0/B1 environment"
user_decision = true

[[item]]
id = "S2-03"
title = "Freeze: verified full-ref bundle, sha256 inventory (tracked, untracked root dirs with main-checkout paths, tarball, .pt, orbax legacy-tree leaf hash via one scripts/ebm/leaf_hash.py, eval assets, bathos run and claim ids), file-level freeze manifest goldens/ebm/freeze_manifest.json (EBM tree plus consumed files), node class NC and --gres type, bth campaign/claim probe (A22), and inventory of every non-main ref with commits under the EBM path set (at least hp-lattice-sanity-check): tip SHA and full git ls-tree file list, checked against the S2-35 disposition (A46, A47)"
repo = "aminx"
size = "M"
depends_on = ["S2-02", "S2-35"]
gate = "git bundle verify passes; tools/ebm_inventory.py --verify re-hashes 100% of manifest entries incl. .pt sha256 and orbax leaf hash; --check-freeze passes at the freeze SHA; untracked-evidence tarball sha256 recorded; A22 probe result and goldens/node_class.toml written; the non-main-ref inventory lists every ref with EBM-path commits with its tip SHA and git ls-tree file list and matches the S2-35 disposition, and a merge-before-F disposition is already on main before F is taken"
user_decision = false

[[item]]
id = "S2-21"
title = "Pre-move golden generators in aminx: model_tree_paths.json, parse_structure goldens (1x2g, 3a7r, 5 decoys), tier-1 generator with its pre-registered sidecar; all outputs under goldens/ebm/ with MANIFEST.sha256"
repo = "aminx"
size = "M"
depends_on = ["S2-03"]
gate = "tree-path and parse_structure goldens regenerate byte-identically twice on titanix CPU, outputs sit inside the filter path set, tier-1 generator and sidecar committed before any GPU run"
user_decision = false

[[item]]
id = "S2-04"
title = "Resumable, shardable, hash-stamped accuracy runner (all locations by explicit argument or env var) with --dry-run/--smoke, --inject-defect alphabet_swap, compare_runs (device-kind assertion, positive and effect-tied negative controls), and pre-registered floor, A0 decoy, A0 ddG and A0 swap-control sidecars under campaign ebm-extraction-parity"
repo = "aminx"
size = "M"
depends_on = ["S2-03", "S2-21"]
gate = "titanix pytest tests/ebm/test_accuracy_runner.py (kill-resume, stale-hash rejection, shard partition, aggregate failure, no hardcoded paths, inject-defect unit test, compare_runs controls) green; 2-unit smoke via the real bth run invocation records an evaluated outcome; four sidecars in their own commit"
user_decision = false

[[item]]
id = "S2-05"
title = "Run the shard-level determinism floor twice on NC at the frozen SHA, append the runner files to the freeze manifest and record F' in ebm-frozen-baseline"
repo = "aminx"
size = "S"
depends_on = ["S2-04", "S2-30"]
gate = "on NC from a detached worktree at F' floor shards stamped twice, stopping rule evaluated, floor catalog outcome evaluated, uv.lock and dependency-table hashes plus jax/jaxlib/xtrax/orbax versions recorded, F' and floors committed with sha256; preempted shards resumed by the retry rule"
user_decision = false

[[item]]
id = "S2-25"
title = "Review the A0 outcome across floor, decoy, ddG and tier-1: accept pass, or decide proceed, re-baseline or investigate on marginal or fail (Q12)"
repo = "aminx"
size = "S"
depends_on = ["S2-05", "S2-32", "S2-33", "S2-34"]
gate = "decision record in .praxia/docs/decisions/ states the A0 outcome and either acceptance of pass or the user's chosen branch"
user_decision = true

[[item]]
id = "S2-06"
title = "Extract history into the permanent ebmx clone with git filter-repo on a fresh --no-local --single-branch --branch main clone at G (freeze manifest verified): current plus historical paths (src/aminx/ebm renamed to src/ebmx), goldens/ tools/ and leaf_hash included, commit-map and analyze output kept; create tools/verify_extraction.sh in the clone"
repo = "ebmx"
size = "M"
depends_on = ["S2-01", "S2-05", "S2-25", "S2-21"]
gate = "tools/verify_extraction.sh: freeze manifest code parts verify at G (lock and dependency-table hashes recorded in the report, not asserted), tree/blob equality for every moved path, no-merges commit counts over the full path set with the refs named (main on both sides, never --all), per-file blob-history check, authorship, size and forbidden-path scans pass; two fresh clones give identical SHAs"
user_decision = false

[[item]]
id = "S2-22"
title = "ebmx bootstrap: bth init (slug ebmx, new project id), pyproject skeleton, first uv lock, local-branch plus rsync procedure in CLAUDE.md"
repo = "ebmx"
size = "S"
depends_on = ["S2-06", "S2-05"]
gate = "bth init record shows slug ebmx with a new project id, uv lock --check is clean on titanix, and the first lock pins jax, jaxlib, xtrax and orbax-checkpoint to the versions recorded by S2-05"
user_decision = false

[[item]]
id = "S2-07"
title = "Import the untracked root evidence (9 top-level files, real_accuracy extras, partial.jsonl, h200 files) under outputs/ebm_benchmarks/untracked_import with a sha256 manifest and a bathos lineage document"
repo = "ebmx"
size = "S"
depends_on = ["S2-06", "S2-03"]
gate = "MANIFEST.sha256 verifies against the originals and the tracked outputs files are unchanged"
user_decision = false

[[item]]
id = "S2-08"
title = "Mechanical rewrite aminx.ebm to ebmx across src, tests and scripts (343 occurrences), rewrite flattened path references (scripts/ebm, tests/ebm, goldens/ebm in sidecars, commands, docstrings), full pyproject deps, SafeMap to ChunkedMap rename with public xtrax.tiling imports"
repo = "ebmx"
size = "M"
depends_on = ["S2-22"]
gate = "rg 'aminx\\.ebm' src tests scripts is empty, rg 'scripts/ebm|tests/ebm|goldens/ebm' is empty outside HISTORY.md and provenance lines, every sidecar script_path and command resolves, ruff check is clean and pytest --collect-only succeeds in a temporary scaffold env that still has aminx installed"
user_decision = false

[[item]]
id = "S2-26"
title = "Seams part 1: _layers (SwiGLU verbatim copy), _tiling (chunked_map and scan with empty-pytree guard), model tree-path golden test"
repo = "ebmx"
size = "S"
depends_on = ["S2-08"]
gate = "goldens/model_tree_paths.json equality test and the swiglu, plan and tiling CPU tests pass on titanix"
user_decision = false

[[item]]
id = "S2-27"
title = "Seams part 2: _alphabet with alphex conformance test, _structure proxide adapter and rewrite of all five parse_structure sites, vendored scripts/_evidence.py, EBM requires_weights hook, importorskip and structure extra for proxide tests"
repo = "ebmx"
size = "M"
depends_on = ["S2-08"]
gate = "no aminx import remains in src, tests or scripts outside allow-listed provenance lines, and alphabet and evidence unit tests pass on titanix"
user_decision = false

[[item]]
id = "S2-28"
title = "Isolation gate: clean-venv suite, meta-path import blocker, sidecar [outcomes]-identical test, ebmx-slug 2-unit smoke through bth run"
repo = "ebmx"
size = "S"
depends_on = ["S2-26", "S2-27"]
gate = "clean venv without aminx/proxide/xtrax extras runs pytest tests -m 'not slow and not requires_weights' green on titanix, import blocker passes, and the ebmx-slug smoke records an evaluated outcome"
user_decision = false

[[item]]
id = "S2-09"
title = "Proxide adapter golden equality: ebmx[structure] output equals the S2-21 aminx goldens exactly; missing proxide gives the actionable install error"
repo = "ebmx"
size = "S"
depends_on = ["S2-27", "S2-21"]
gate = "adapter equals goldens exactly for 1x2g, 3a7r and 5 decoys and a missing proxide gives the actionable install error"
user_decision = false

[[item]]
id = "S2-10"
title = "Config resolver for weights, orbax model, eval assets, reference repo and JAX cache dirs (explicit > env > [tool.ebmx] > ~/.config/ebmx/config.toml); remove every hardcoded path"
repo = "ebmx"
size = "S"
depends_on = ["S2-28"]
gate = "precedence and malformed-file tests pass and grep for /tmp/proteinebm, ~/repos/ProteinEBM, proteinebm_bench_assets, /orcd/scratch and /home/maarxaru in src and scripts is empty"
user_decision = false

[[item]]
id = "S2-11"
title = "Deprecation gate for xtrax 0.4.0a11: full suite under -W error::DeprecationWarning, xtrax <0.5 pin check, no SafeMap or safe_map outside _tiling.py"
repo = "ebmx"
size = "S"
depends_on = ["S2-28", "S2-10"]
gate = "titanix suite green under -W error::DeprecationWarning and grep for SafeMap outside _tiling.py is empty"
user_decision = false

[[item]]
id = "S2-12"
title = "Weights provenance: weights.lock.toml (HF revision, .pt sha256, legacy-tree and converter leaf hashes), verifying loader, vendored hash-checked leaf_hash, converter entry point compared with the legacy tree (A28)"
repo = "ebmx"
size = "S"
depends_on = ["S2-10", "S2-03"]
gate = "on NC: loader refuses a wrong-sha256 file, converter result versus the legacy leaf hash is recorded (equal, or differing leaves listed and debt filed), lock records converter_matches_legacy"
user_decision = false

[[item]]
id = "S2-13"
title = "Tier-1 bit-level golden regression on NC against the legacy tree (energy, score, 5-step Langevin, conformational bias) with per-quantity tolerances, sidecar committed before the job, positive control and swap-golden negative control"
repo = "ebmx"
size = "M"
depends_on = ["S2-09", "S2-11", "S2-12", "S2-05", "S2-34", "S2-28"]
gate = "on NC via sbatch: leaf hash equals legacy hash, comparison within per-quantity atol_q (or near-threshold decision recorded) with device kind asserted equal, swap-golden comparison fails, replicate control passes, bathos outcome evaluated"
user_decision = false

[[item]]
id = "S2-24"
title = "Pre-register arm B1: commit the B1 decoy and ddG sidecars with literal TOL_UNIT and TOL_AGG from the S2-05 floor record, plus the A0-vs-B1 runner-diff test, in their own commit"
repo = "ebmx"
size = "S"
depends_on = ["S2-13", "S2-07", "S2-05"]
gate = "sidecars with literal tolerances committed, runner-diff test passes, and git log shows the commit precedes any B1 job submission"
user_decision = false

[[item]]
id = "S2-14"
title = "Run arm B1 on NC: full decoy (133) and ddG (64) accuracy on ebmx under the committed sidecars, paired per-unit comparison to A0, reusing the S2-32 swap run as negative control"
repo = "ebmx"
size = "L"
depends_on = ["S2-24", "S2-32", "S2-30"]
gate = "all units stamped and aggregates verified, bathos outcome pass for both B1 sidecars (or near-threshold decision recorded), compare_runs reports device kind equal, max unit delta within TOL_UNIT, aggregate delta within TOL_AGG, swap control valid (drop >= 0.25) and positive control passing"
user_decision = false

[[item]]
id = "S2-15"
title = "Claim migration: new extraction-parity claim in ebmx, copy of the old claim, old file left byte-identical in aminx, lineage note for campaign aba3bfe6 and the new campaign"
repo = "ebmx"
size = "S"
depends_on = ["S2-14"]
gate = "bth claim validate passes for the new and the untouched old claim and the old claim file sha256 equals the recorded value"
user_decision = false

[[item]]
id = "S2-29"
title = "Decide whether A0/B1 evidence closes the old claim campaign aba3bfe6 (Q11) and, if approved, attach it by the mechanism found in S2-03"
repo = "ebmx"
size = "S"
depends_on = ["S2-15"]
gate = "decision recorded; if approved the bth command ran and the catalog shows the claim state; old claim file sha256 unchanged"
user_decision = true

[[item]]
id = "S2-16"
title = "README, CLAUDE.md, NOTICE and attribution, CHANGELOG, using-<name> skill, docs relocation and HISTORY.md, ebm-model-facts descriptor"
repo = "ebmx"
size = "S"
depends_on = ["S2-08", "S2-01"]
gate = "link check and docs index check pass and NOTICE names upstream ProteinEBM and Boltz-1"
user_decision = false

[[item]]
id = "S2-17"
title = "Create the remote, push the full local branch including untracked_import, CI (CPU unit tests on minimal install, lint, import-isolation job), titanix recipe and Engaging sbatch under scripts/slurm"
repo = "ebmx"
size = "M"
depends_on = ["S2-10", "S2-11", "S2-16", "S2-01", "S2-07", "S2-31"]
gate = "remote holds the branch, CI is green on a pull request, the documented titanix recipe gives the same test result and the S2-01 install command works"
user_decision = true

[[item]]
id = "S2-18"
title = "aminx cutover: remove EBM code, tests, scripts, goldens/ebm, tools/ebm_inventory.py, A0 sidecars and outputs, and docs; edit the browser-validation inventory (the ebm entries, 79 when read and re-counted at S2-18 time, and _ROOT_PACKAGES) in the same PR as a textual merge with S1, S4 and S5 edits to the same JSON; keep the claim file byte-identical; add the CHANGELOG pointer to ebmx with the S2-01 install command; no shim and no version bump"
repo = "aminx"
size = "M"
depends_on = ["S2-14", "S2-15", "S2-17", "S2-31", "S2-35", "S2-36"]
gate = "titanix: test_browser_validation_inventory green, git ls-files | rg -i ebm matches only the allow-list in 4.10, rg for aminx.ebm is empty in src, scripts and tests, a one-time rg 'aminx\\.ebm' over mpnn_ext, asr, hautespout (main and WIP) and tev_design is recorded and any hit halts the item until an owner decision is recorded, S2-36 is closed (ported or not applicable), the hp-lattice-sanity-check disposition from S2-35 is stated in the CHANGELOG pointer, claim file sha256 unchanged, CHANGELOG pointer names ebmx and the S2-01 install command, no shim, release-workflow, version or pyproject dependency edit in the diff"
user_decision = true

[[item]]
id = "S2-20"
title = "Close-out (user-approved deletion): re-hash the untracked evidence against a fresh clone of the remote and the S2-03 tarball, then remove the root ebm_benchmarks dirs from the main checkout, mark praxia debt #2369 completed with a link to this spec"
repo = "aminx"
size = "S"
depends_on = ["S2-18", "S2-07", "S2-17"]
gate = "fresh-clone re-hash of untracked_import passes and tarball verifies immediately before deletion, root ebm_benchmarks directories are absent afterwards and debt #2369 status is completed"
user_decision = true

[[item]]
id = "S2-30"
title = "Approve GPU budget and node class NC for the A0 and B1 campaigns (Q8)"
repo = "aminx"
size = "S"
depends_on = ["S2-03"]
gate = "decision record in .praxia/docs/decisions/ approves the budget and the NC gres type before any GPU job is submitted"
user_decision = true

[[item]]
id = "S2-31"
title = "Decide cutover branch on converter result: read converter_matches_legacy; if false choose fix-converter-and-rerun-tier-1 or legacy-tree-only (Q14)"
repo = "ebmx"
size = "S"
depends_on = ["S2-12"]
gate = "decision record states converter_matches_legacy and, if false, the user's chosen branch"
user_decision = true

[[item]]
id = "S2-32"
title = "Run A0 full decoy (133 natives) plus the 10-native alphabet_swap negative-control run on NC under the committed sidecars; commit outputs under outputs/ebm_benchmarks/extraction_parity/a0/"
repo = "aminx"
size = "M"
depends_on = ["S2-04", "S2-05", "S2-30"]
gate = "133 units stamped and aggregate verified, evaluated outcomes, git diff F' to run SHA over EBM and consumed modules empty, --check-freeze --assert-env (code parts and lock/dependency-table hashes) passes from the detached F' worktree, swap run mean drop >= 0.25 against A0 on the same 10 natives"
user_decision = false

[[item]]
id = "S2-33"
title = "Run A0 full ddG (64 assays) on NC under the committed sidecar; commit outputs under the A0 path"
repo = "aminx"
size = "M"
depends_on = ["S2-04", "S2-05", "S2-30"]
gate = "64 assays stamped and aggregate verified, evaluated outcome, git diff F' to run SHA over EBM and consumed modules empty, --check-freeze --assert-env (code parts and lock/dependency-table hashes) passes from the detached F' worktree, guard band reported"
user_decision = false

[[item]]
id = "S2-34"
title = "Generate tier-1 goldens (normal and alphabet-swap inputs) twice on NC with per-quantity floors and atol_q, committed with sha256"
repo = "aminx"
size = "S"
depends_on = ["S2-21", "S2-05", "S2-30"]
gate = "generated from a detached F' worktree (or a descendant) with --check-freeze --assert-env (code parts and lock/dependency-table hashes) passing, two replicates per quantity, floors and atol_q committed with sha256, stopping rule evaluated, sidecar commit precedes the job"
user_decision = false

[[item]]
id = "S2-35"
title = "Decide the disposition of the aminx branch hp-lattice-sanity-check and its hautespout WIP consumer: merge to main before F (the four tip files, no merge base), abandon, or port into ebmx (S2-36); record the tip SHA, the git ls-tree file list, and any hp_* file the WIP vendored copy has beyond the tip"
repo = "aminx"
size = "S"
depends_on = ["S2-02"]
gate = "decision record in .praxia/docs/decisions/ names merge-before-F, abandon or port, with the tip SHA and git ls-tree listing (A47), the WIP-only hp_* files and their owner, and the S1-28 bump target; for merge-before-F the files are on main before S2-03 starts"
user_decision = true

[[item]]
id = "S2-36"
title = "Conditional: port the hp_* files (hp_energy.py, its test, the hp_lattice_sanity script and sidecar, plus any WIP-only hp_* module named by S2-35) into ebmx as new commits; closes not-applicable unless S2-35 recorded port"
repo = "ebmx"
size = "S"
depends_on = ["S2-17", "S2-35"]
gate = "closes not-applicable (recorded in the item) unless S2-35 says port; if port, ebmx CPU unit tests of the ported files pass on titanix, each file carries a header with the source tip SHA and sha256, no aminx import (S2-28 gate), ebmx CI green"
user_decision = false

```

## Revision log

### Round 1 (adversarial spec coherence, objections C1 to C16)

Applied (conceded or partial, all 16 touched):
- **C1** `parse_structure` coupling: the `_structure.py` adapter and the rewrite of all five import sites (A29) move out of S2-09 into the new S2-27, which gates S2-28's grep and import-blocker checks. S2-09 keeps only the golden-equality proof. S2-08 is now only the mechanical rewrite, and its gate no longer claims aminx-freedom.
- **C2** Golden ownership: new S2-21 owns every pre-move generator (tree paths, `parse_structure` goldens, tier-1 generator and sidecar) with outputs under `goldens/ebm/`, added to the 4.2 filter path set. The tier-1 goldens themselves are produced in S2-05. The leaf hash is defined normatively in 4.5 with one implementation (`scripts/ebm/leaf_hash.py`, S2-03), vendored with a source-sha256 check in `ebmx._weights`.
- **C3** Shim: 4.10 now specifies a meta-path finder aliasing the 15 submodules in `sys.modules`, and the S2-18 gate covers `from aminx.ebm.model import ...`, `import aminx.ebm.checkpoint`, `from aminx.ebm import training` and the attribute form.
- **C4** Hardware: node class NC (Engaging `pi_so3`, explicit `--gres` type, no preemptable fallback) pinned for floor, goldens, A0, tier-1 comparison and B1; device-kind and version equality asserted; weight-dependent gates moved from titanix to NC; model staging with sha256 verification stated. The exact gres type is UNVERIFIED (A30), recorded by S2-03.
- **C5** Converter vs legacy orbax: ledger row A28 (UNVERIFIED, deferred to S2-12), mismatch branch in 4.5, and S2-13 now depends on the legacy hash through the lock rather than on converter equality. S2-12 records the result and does not fail on a mismatch.
- **C6** A0 outcome: new user-decision item S2-25 between S2-05 and S2-06, with Q12. S2-05 only runs and records.
- **C7** Untracked evidence: S2-20 is now `user_decision = true`, depends on S2-17 (remote push), re-hashes against a fresh clone of the remote and the S2-03 tarball, and the main-checkout path is recorded in the manifest. Q13 added.
- **C8** Home and remote: 4.7 step 0 states the local-branch plus anchored-rsync convention until S2-17, S2-01 records the permanent clone location, new S2-22 owns `bth init` and the first `uv lock`, and the ebmx-slug smoke gate sits in S2-28 before S2-13.
- **C9** S2-08 split into S2-08 (mechanical rewrite and pyproject), S2-26 (SwiGLU and tiling seams, tree-path test), S2-27 (alphabet, structure, evidence seams) and S2-28 (isolation gate). S2-11 now depends on S2-10 so the two no longer edit the same files concurrently, and the rename moved into S2-08.
- **C10** The reviewer's premise was only partly right: a11's `xtrax.tiling.strategy` does keep a `SafeMap` alias (read at `v0.4.0a11:src/xtrax/tiling/strategy.py:156-163`, ledger A27), so test_plan.py collects. The rename and public-import fix still move into S2-08, because aminx's ban list forbids the deep paths and an alias-dependent interim item is avoidable.
- **C11** (partial) Probe ordering was already covered; added the new campaign `ebm-extraction-parity` for floor, A0 and B1, and moved claim closure to the explicit user-decision item S2-29.
- **C12** (partial) Tier 2 keeps its single absolute tolerance, since Spearman values are unitless. Tier 1 is now per quantity (`atol_q`, relative term 1e-6, floor over the whole tier-1 input set); the tier-2 floor uses whole shards under the full run's shard boundaries (extremes by length); stopping rules added for both; the negative-control margin is tied to the observed effect (mean drop >= 0.25 against the known 0.83 to 0.32).
- **C13** Historical rename paths enumerated into `ebm-paths.txt` (A31, UNVERIFIED), commit count defined as `rev-list --count --no-merges` plus a per-file blob-history check.
- **C14** S2-03 re-sized to M and its backlog gate now carries the full section-7 gate.
- **C15** Direct proxide imports in tests added to 3.2, A2 narrowed to aminx imports, new A32, `tests` extra includes `structure` and the two tests use `importorskip`.
- **C16** (partial) Pre-registration commits are separate from run items: S2-24 (B1 sidecars) split from S2-14, and the S2-04/S2-21 commits already precede S2-05; a retry/resume rule is in 4.9 and in the gates. The existing 4.9 resume design was already in place.

Declined: none outright. Declined parts: the reviewer's suggestion to lower C12 to per-quantity tolerances for tier 2 (unitless Spearman needs none), and the suggestion to move remote creation earlier (kept at S2-17 because it is user-gated; the local-branch convention covers the gap).

Spikes: none run. The task forbids executing code, so A28, A30 and A31 stay UNVERIFIED with deferred probes in named items, and A27 and A29 are settled by reads.

### Round 2 (adversarial spec coherence, objections C1 to C11)

Applied (conceded, or conceded in part):
- **C1** Negative-control ownership: `--inject-defect alphabet_swap` and its unit test are in S2-04; the 10-native swap run is in S2-32 under its own sidecar (pre-registered in S2-04); S2-14 reuses it; the tier-1 swap golden is produced by the S2-21 generator, run in S2-34 and used by S2-13 (4.8 Controls and Tier 1).
- **C2** Kept the flatten and added path-reference rewriting to S2-08 (4.7 step 5), with the `rg 'scripts/ebm|tests/ebm|goldens/ebm'` gate and a sidecar-resolution test. I did not drop the flatten because the rewrite is mechanical and gated, and dropping it would reopen every flattened path in sections 4.1 to 4.8.
- **C3** New user-decision item S2-30 (budget and NC, Q8); S2-05, S2-32, S2-33, S2-34 and S2-14 depend on it.
- **C4** S2-01 records the distribution channel and exact install command; the shim message and S2-19 follow it (advisory if private); S2-18 installs `ebmx` from the clone by path and tests the no-ebmx case under an import blocker.
- **C5** (partial) Mismatch was already handled but not gated; added Q14 and user-decision item S2-31, on which S2-17 and S2-18 depend.
- **C6** The runner takes every location by explicit argument or env var; A0 outputs live at `outputs/ebm_benchmarks/extraction_parity/a0/` (inside the filter set); the "modulo import prefix" claim is replaced by a runner-diff test in S2-24.
- **C7** `F'` is the SHA of the first A0 run (S2-05), recorded in `ebm-frozen-baseline`; S2-32/33/34 and S2-06 assert an empty code diff over the EBM and consumed modules; the clone is taken at `G` (A0 outputs committed) and S2-06 checks `F'..G`.
- **C8** S2-18 deletion list enumerated with a `git ls-files | rg -i ebm` allow-list gate; `tools/ebm_inventory.py` moves with history, `tools/verify_extraction.sh` is created in the ebmx clone by S2-06.
- **C9** (partial) Added the hard-fail versus near-threshold classification and Q15 (user decision, no silent widening) and stated that the 1e-4 minimum is a judgment. Declined: widening tolerances around seam differences; they remain real-difference signals.
- **C10** S2-02 restated as confirming the owner's merge, `user_decision = true`, no executor merge.
- **C11** S2-05 split into S2-05 (floor, records `F'`), S2-32 (A0 decoy + swap), S2-33 (A0 ddG), S2-34 (tier-1 goldens); S2-25 depends on all four and S2-13 on S2-05 and S2-34 only.

Declined: none outright beyond the notes above. Spikes: none (no code execution allowed); the 1 MB size of A0 unit JSONs and the exact path-reference count (74 versus 44 hits) are UNVERIFIED and are re-checked by the S2-08 and S2-32 gates.

### Coherence round 1 (cross-spec, objections C1, C2, C3, C6, C7)

Applied (S2 only; no other spec was edited):
- **C1** Rename-first is now the default (4.11, Q7): "R, accepted at the recommendation pending S3-01", one question shared with S1 Q10 and S3 Q4. The old arm uses the renamed tree (`src/molxmpnn/ebm`, `molxmpnn.ebm`, `repo = "molxmpnn"`) in 4.1 to 4.10, section 7 and items S2-01/02/03/04/05/18/20/21/25/30/32/33/34. Section 3 and the ledger deliberately keep the pre-rename paths, because those are the lines that were read at `d1210e4a`; a tree-naming paragraph and A33 map them. S2-18 gained `S3-07`; S2-03 gained `S3-13`. The language that S3 should depend on S2-05/S2-25/S2-19 is deleted; the S3-to-S2 edges are S3-25 and S3-20 (leaf list), plus S3-05 to S2-02 which S3 chose itself (A37). Option E is kept as an alternative regenerated through S3's re-open table and the round-2 text in git history.
- **C2** S2-19 is retired (id not reused). S2-18 no longer installs a meta-path finder, has no shim tests and no `aminx.ebm` allow-list entry; its gate is the inventory test, the `git ls-files | rg -i ebm` allow-list (claim file, pointer doc, S3-owned paths) and the unchanged claim sha256. Provides `aminx-ebm-shim` removed; added `ebm-install-command` (S2-01, consumed by S3-25) and a Consumes row for the S3 shim row. The "import aminx resolves to the hub" sentences are gone (A34). A19 is reworded and no longer feeds any S2 item. The bathos-slug risk row now follows S3 Q5 and A8 (no project-id column; legacy slug `aminx` for A0/floor sidecars; S3-20 runs after all S2 leaves).
- **C3** S2-03 `depends_on` S3-13. The directory-level empty-diff gate is replaced by a file-level freeze manifest (`goldens/ebm/freeze_manifest.json`, EBM tree plus exactly the consumed files, `SwiGLU` hashed by class source) checked by S2-32/33/34 and S2-06 (4.7 step 2). Deviation from the objection, stated: the manifest is written in two parts, (a)+(b) in S2-03 and (c) the runner and driver scripts appended by S2-05 at `F'`, because those scripts do not exist yet at S2-03 (S2-04 and S2-21 create them). The filter uses both `--path-rename src/molxmpnn/ebm/:src/ebmx/` and `--path-rename src/aminx/ebm/:src/ebmx/` (A31, A35).
- **C6** No separate change beyond C2. Observation for the integrator, not a decision: with S2-19 gone, S2 items with no S2 dependent are S2-20 and S2-29 only, so S3-20's list (S2-01, S2-19, S2-20, S2-25) names two non-leaves and a retired id and should be recomputed by S3-28; S2 has no stake in the exact leaf set.
- **C7** S2-02 stays the single owner of landing #174 (user-gated merge) and now also records the locked xtrax version used by A0, tier 1 and B1. S3-05 and S4-16 already depend on S2-02 per their own text (A37); no S2 change is needed beyond the wording and a risk row.

Declined: none. Spikes: none (no code execution allowed); A35 and A36 are UNVERIFIED with named deferred checks (S2-06, `--check-freeze`), and A33, A34, A37 are settled by reads of the S3 and S4 specs, which are agent-written and therefore leads about intent, not facts about the code.

### Coherence round 2 (cross-spec, objection CH2-04)

Applied (S2 only):
- **CH2-04** Lock churn from S4-16, S4-35 and S1-17 is unordered against S2-05..S2-34, so S2 no longer relies on ordering. Freeze-manifest part (c) now adds sha256 of `uv.lock` and of the pyproject dependency tables, checked by `--check-freeze`; the env stamp adds `orbax-checkpoint` (and the lock hash); S2-05, S2-32, S2-33 and S2-34 execute from a detached worktree at exactly `F'` (a descendant only with lock-hash equality asserted); S2-22 pins ebmx's first lock to the A0-recorded `jax`, `jaxlib`, `xtrax` and `orbax-checkpoint` versions and gained `depends_on` S2-05 so the versions exist when it runs. New ledger rows A38 (read, changed) and A39 (UNVERIFIED, deferred). No new edges into other specs, no change to the Provides/Consumes tables.

Declined: none. Spikes: none (no code execution allowed).

### No-rename revision (261001, user decision revising D2)

Trigger: the MPNN package is released on PyPI as `aminx` and keeps that name; there is no rename (no `molxmpnn`, no shim distribution, no
repository, Hugging Face or PyPI cutover, no consumer migration). The hub is a separate project (working slug `aminx-hub`; the user also
floated a name derived from the domain praxia.science, which works with any repo slug; the final name, PyPI policy and domain are the
user's decision at S3-01). D3 (EBM moves to its own project, debt #2369) is unchanged. The entries above are history and keep the names
they were written with; everything above the log was revised in place. Applied (S2 only; no other spec was edited):

- **Names.** `molxmpnn` means `aminx` again everywhere it meant the MPNN package or repository: tree `src/aminx/ebm`, `aminx.ebm`,
  `repo = "aminx"` in 13 items, front matter `owner_repos`, the claim and pointer-doc rows, and the gate commands. The tree-naming
  paragraph now says nothing is renamed, and the section-3 note about pre-rename paths is gone.
- **Edges to S3 ids (all dropped, none re-pointed).** S2-02's title lost "S3-05 and" (S4-16 remains its dependent); `S2-03` `depends_on`
  `["S2-02"]` only (the `S3-13` edge was an A0-freeze-on-the-released-renamed-tree ordering) and its title "Freeze after the rename" is
  now "Freeze"; `S2-18` lost `S3-07`. S2 now has no edge to any S3 id and no S3 id in any item field. S2 depends on none of S3-01, S3-09,
  S3-12, S3-18, S3-21, S3-24, S3-28, S3-29: they are conventions or later adoptions (4.11).
- **Design text removed because it existed only for the rename.** Rename-first option R and alternative E (Q7, 4.11, the tree-naming
  paragraph), the S3-13 "sole publisher of 0.2.0a4" and the S3-10/S3-25 `aminx.ebm` shim row (4.10, A34, the shim risk row, the
  `aminx.ebm` Consumes row and the `S3-25` consumers in Provides), the double-era filter (`--path-rename src/molxmpnn/ebm/` plus
  `src/aminx/ebm/`, A35), "GitHub URL redirects" (4.7 step 6), the legacy bathos slug and the union query `project_slug IN ('aminx',
  'molxmpnn')` (S3 Q5, S3-20; replaced by A41), `molxmpnn_version` in the sidecar `[dependencies]` rewrite, "rename codemod touches ~12k LOC",
  and the S3-20 checkout-rename caveat in S2-20.
- **Ordering decisions re-checked and restated (4.11).** "Rename first versus EBM first" is void. S2-03 now follows only S2-02. The
  freeze manifest, detached worktree at `F'` and lock-hash pins stay, because S1, S3 and S4 still change nearby files and the lock; the
  S3-07..S3-12 landing-before-freeze argument is replaced by A36 (deferred to `--check-freeze`) and A44 (S3-09's sweep has nothing to
  change in any manifest file). A release carrying the removal is ordered by convention behind S3-12's guard and train rule, with no S2
  release item (A43); S2-18 gains a CHANGELOG pointer that quotes the S2-01 install command instead of an S3 shim row, and Q4 is rewritten
  around "pointer only versus a five-line in-tree stub".
- **Hub.** Nothing in S2 assumes the hub inherits the name `aminx` any more: "the shim stays on PyPI `aminx`, the hub ships no Python
  package" is replaced by "PyPI `aminx` stays the MPNN package (A42); the hub is a separate project (`aminx-hub`, name pending S3-01)". A
  risk row records the reserved-name guard (S3-24). Hub mentions in Provides and 4.11 now say "the hub catalog entry (S5)".
- **Items.** None added and none deleted (no S2 item existed only for the rename; S2-19 was already retired). Retitled or re-gated: S2-02,
  S2-03, S2-06 (single `src` path-rename), S2-18 (no S3-07, CHANGELOG pointer, gate without `tests/compat/` or the shim), S2-28 (the
  import blocker names `aminx`, `proxide`). Ids stable.
- **Ledger.** Retired (ids never reused): A33 (S3-07 rename mapping), A34 (S3 as sole publisher and shim owner), A35 (one filter run across
  the rename commit). Reworded: A19 (now a CHANGELOG-pointer row, `If false` is none), A31 (now VERIFIED by read-only history queries:
  `src/prxteinmpnn/ebm` has 0 commits on any ref; 565 renames repo-wide, none naming ebm; no delete in the EBM trees), A36 (S3 items and
  the freeze manifest), A37 (S4-16 is the only other owner of a #174 dependent). Added: A40 (no shim infrastructure to edit), A41 (slug
  stays `aminx`), A42 (identity stays; hub separate), A43 (no S2 release item), A44 (no `prxteinmpnn` in any manifest file). A38's
  self-citation was moved to the current line range.
- **Gates.** S2-03 no longer requires `S3-13`; S2-06 enumerates renames expecting an empty result; S2-08, S2-18 and S2-28 use plain
  `aminx` patterns; the S2-20 gate drops the checkout-rename caveat.

Not changed: D1, D3-D7; the parity design (tiers 0-2, controls, floors), resumability and chunking (4.9), the config resolver, the weights
provenance, Q1-Q3, Q5, Q6 and Q8-Q15 apart from the wording noted above; the toml block keeps the keys id, title, repo, size, depends_on,
gate, user_decision.

Declined: none. Spikes: none (spec-only task, no code executed; the spike runner is absent from this workspace). New and reworded rows rest
on reads, including read-only git history queries.

### No-rename coherence r1

Applied (S2 only; no other spec was edited):
- **CH1-02** 4.10, the 4.2 table row, S2-18 title, gate and section-7 row now say `browser_validation_paths.json` is shared with S1 (removes `build_run_spec`, adds `as_run_spec` and nested config classes), S4 and S5 (new public defs under scoring or run, `aminx serve`), so S2-18's removal of the `ebm` entries is a textual merge; the 79 count is re-counted at S2-18 time and the gate runs the inventory test after the merge. New ledger row A45 (VERIFIED by reads of the test, the JSON and `aminx.run`). The internal-symbol baseline is named as a possible further shared file.
- **CH1-03** S2-03 now inventories every non-main ref with commits under the EBM path set (at least `hp-lattice-sanity-check`) with a disposition (merge before F, abandon, port into ebmx after S2-17). S2-06 clones with `--single-branch --branch main`, and the commit-count and blob-history gates (4.7 step 4, S2-06 gate, section 7) name the refs (`main` on both sides, never `--all`). S2-18 adds a one-time `rg 'aminx\.ebm'` over `mpnn_ext`, `asr`, `hautespout` (main and WIP) and `tev_design`, and the CHANGELOG pointer and Q4 mention the branch. New ledger row A46: the branch history is unrelated to main (no merge base) and its tip commit adds `src/aminx/ebm/hp_energy.py`, `tests/ebm/test_hp_energy.py` and `scripts/validate/hp_lattice_sanity.py` (read via `git show --stat`); the rest is UNVERIFIED with a deferred check in S2-03 and S2-18.

No items added or deleted; ids stable; the toml block keeps its keys and no new `depends_on` edge.

Declined: none. Spikes: none (spec-only task, no code executed).

### No-rename coherence r2

Applied (S2 only; the other specs are patched concurrently):
- **R2-01** `--check-freeze` now has two classes (4.7 step 2). Code parts (a), (b) and the scripts and sidecars of (c) are asserted at `G` by S2-06 and in the run items. The lock and dependency-table hashes are asserted only under `--check-freeze --assert-env` by S2-05, S2-32, S2-33 and S2-34 from the detached `F'` worktree, and are recorded, not asserted, at `G`. The invalidation sentence applies to code parts only; a lock change on main never invalidates A0. The split is stated in the 4.7 text, the S2-06, S2-32, S2-33, S2-34 section-7 rows, the toml gates of S2-06, S2-32, S2-33 and S2-34, and the risk rows. New ledger row A48 (VERIFIED by reads).
- **R2-04** New user-decision item S2-35 owns the disposition of `hp-lattice-sanity-check` (merge before `F`, abandon, port) and S2-03 now depends on it and only records the tip file list; new conditional item S2-36 ports the `hp_*` files into ebmx (closes not-applicable unless S2-35 says port; depends on S2-17 and S2-35) and S2-18 depends on it. The S2-18 consumer scan now halts the item on any `aminx.ebm` hit until an owner decision is recorded. A46 is narrowed to the consumer facts; new row A47 is VERIFIED by reading the branch tip tree: one hp module (`hp_energy.py`) plus its test and the `hp_lattice_sanity` script and sidecar, no merge base, and `hp_coarse_sampler` and `hp_windowing` are absent from the tip, so S1's three-module list reflects the WIP vendored copy. The edge S1-28 -> S2-35 is owed to S1 (4.11), not added here.

No item deleted; ids stable; the toml block keeps its keys, `depends_on` references only existing ids, and the new edges (S2-03 -> S2-35 -> S2-02, S2-36 -> S2-17/S2-35, S2-18 -> S2-35/S2-36) form no cycle. Spikes: none (spec-only task); A47 and A48 are settled by reads.
