---
title: PottsMPNN + ProtonPottsMPNN ONNX / browser export — scope
description: Scope, phase plan and acceleration levers for shipping Potts-family ONNX exports, reusing the ProteinMPNN split-export process
task_id: 261007_potts-onnx-export
status: decided (1,2); 3 open
---

# PottsMPNN + ProtonPottsMPNN ONNX / browser export — scope

Status: **draft scope, nothing has run.** Each claim is tagged *verified* (read at file:line in
the tree at `6592bdb9`, branch `wt/261008-protonpotts-p7`, off main `dab58900`) or
*unverified*. Recon by two Haiku agents was spot-checked by Opus; corrections are noted inline.

## 1. What "the same process" was (ProteinMPNN, 260923 → 260930)

Verified from specs `260923_aminx-browser-validation.md`, `260926_aminx-browser-export-phase2a.md`,
`260929_p07-split-export.md` and `git log -- src/aminx/export scripts/browser_validation browser`:

| phase | what it proves | gate |
|---|---|---|
| 0 | export-safety census, jax2onnx→ORT-CPU spike, ORT-Web wasm smoke | `export_safety_census.py`, `jax2onnx_spike.py`, `ort_web_smoke.py` |
| a | aminx JAX ↔ reference PyTorch (exact + sampling tiers) | `layer_a_exact_*`, `layer_a_sampling_*` |
| b | JAX ↔ ONNX under ORT-CPU (and IREE native, fallback only) | `layer_b_ort_*`, `layer_b_iree.py` |
| c | ONNX under ORT-Web in headless Chromium, cross-origin isolated | `layer_c_*`, `browser/layer_c/` |
| knobs | every runtime control (bias, fixed, temperature, order, ties) as a graph input | `p07_knobs_gate.py` |
| split | monolith → 4 loop-free graphs (E encoder, W wave schedule, D decoder step, F fuse+sample), AR loop in JS | `p07_split_*.py`, `browser/aminx-sampler/split_*.mjs` |

Every gate was a pre-registered `.bth.toml`, graded by record. ~7 calendar days. Where the time
went (verified, commit-cited in the recon report
`.praxia/subagent_outputs/toolu_01XrFcApw4ZCpvorjN1Znmg3.1.md`):

1. **A monolith was built first and then abandoned** for the split (nested `If`/`Scan` under
   `lax.cond`; 2.9–4.7× larger; WebGPU forces device copies on control flow).
2. **IREE** was built as a fallback and never shipped (stack limits at L≥512; no WebGPU target).
3. **"Converted" ≠ "runs"**: every layer-b artifact passed conversion and then failed
   `InferenceSession` (unclamped `Slice` from a clamping `dynamic_slice`).
4. ~20 `fix(browser)` commits on one day: x64 refusal, ORT-Web external data, absolute
   `wasmPaths`, Node bytes-not-paths, timer controls.
5. A one-off log-prob bound change (1e-4 → 2e-4, `5378da83`) — **by user decision after a graded
   fail, for a new run only** (debt #2056). Recon flagged it as a no-loosening violation; it is not.

## 2. What already exists for Potts (the big accelerator)

- **Layer a is already done.** The Potts parity programme graded aminx JAX against the upstream
  PottsMPNN oracles (all four Potts waves pass by record; spec 261007 §10). The browser chain
  only has to add JAX ↔ ORT-CPU ↔ ORT-Web. *Verified (task list / ledger); the oracles are sealed.*
- **The split machinery is model-agnostic in shape**: ORT session/driver
  (`split_driver.mjs:126`), JS decode loop (`split_loop.mjs:106`), Chromium harness
  (`browser/layer_c/`), sidecar/outcome pattern, `rng_audit.find_rng_primitives`,
  `zero_dropout`, sha256 manifests. *Verified.*
- **Hard-wired to ProteinMPNN** (must be parameterized, not forked): `MPNN_ALPHABET` /
  `OMIT_BIAS` (`runspec_core.mjs:21-22`), `N_TOKENS=21` (`split_loop.mjs:22`,
  `p07_split_export.py:55`), per-graph input orders (`split_driver.mjs:22-39`), buckets
  (`buckets.py:36`), checkpoint pin (`wrappers.py:71`), Pages allow-list
  (`tools/build_site.py:67-69`). *Verified.*
- **P7 (this branch) gives one alphabet object** (`families/potts_mpnn/alphabet.py`), so the
  JS side can read the alphabet from an export manifest instead of a literal — the same change
  serves PottsMPNN (21) and ProtonPotts (30).

## 3. Potts inference paths and their export shape

Verified unless marked:

| path | device-side structure | export shape |
|---|---|---|
| `score:energy`, `score:ddg` | one `filter_jit` graph: encoder → `potts_head` → `merge_pair` ×2 → `potts_energy` (`driver.py:110-159`) | **one loop-free graph `S`.** Mutant enumeration (DMS/CSV/FASTA) stays host-side in JS. Easiest path; ship first. |
| `sample` (AR decode) | `PottsARDecode`: own decoder cache over `DecoderLayer` modules, one `lax.scan` over tie groups (`decode.py:214-246, 366`). **Does not reuse** ProteinMPNN `_decode_one_step`. | split like P07: **E** (encoder + etab head, once/structure), **W** (order/rank, once/sample), **D** (decoder step with explicit cache in/out), **F** (fuse + sample). Cache I/O is new relative to P07. |
| refine `potts` (**the default**, `options.py:12`) | one sweep = `lax.scan` over positions on the etab (`refine.py:608/707`) | per-position graph **R** with the JS loop driving the sweep, or the sweep as one `Scan` graph — decide at the Phase-0 spike (the split lesson says avoid `Loop`/`Scan`). |
| refine `potts_converge` | `lax.while_loop`, ≤1000 sweeps (`refine.py:330`) | **no while in-graph**: JS repeats R-sweeps and checks convergence. |
| refine `nodes`, binding-energy modes | decoder-driven refine (`refine.py:964, 1052`), partition scans | **defer** to a later item. |

Note (verified): because the default `optimization_mode` is `"potts"`, plain `sample` **always
refines**; `driver.py:1-5`'s docstring understates this. Upstream parity holds, so this is
documentation debt, not a behaviour bug. A browser sampler that skips R would diverge from
`aminx sample` defaults.

### Known-risky ops (Phase 0 must spike each; all *unverified* under jax2onnx/ORT-Web)

- `scatter(..., mode="drop")` with out-of-range sentinels (`decode.py:186, 489, 492`;
  `refine.py:764, 825, 1160-1180`). ONNX `ScatterND` has no drop mode → expected to need a
  masked rewrite. Same failure class as the `dynamic_slice` clamp that broke layer b.
- scatter-max in `merge_pair` (`etab.py:139`) → `ScatterElements(reduction="max")` (opset ≥ 16).
- `jnp.argsort` default-stable (`decode.py:171, 174, 367`) → trips xtrax `sort-stability`;
  fix pattern known (unique key + explicit `stable=False`, `48e2b573`).
- unchunked `top_k` in `potts_edge_features` (`features.py:93`) vs row-chunked in the ProteinMPNN
  export (`wrappers.py:84-92`); int64 TopK blocks ORT-Web WebGPU (split spec §1.3).
- host-side `jax.random` for refine order (`sample_host.py:548-552, 581`) and float64 order key
  (`sample_host.py:69`) → noise/order become JS-generated inputs, as Gumbel noise was for P07.
- etab size: float32 `L×K×20×20`, K=48: ~9.8 MB at L128, ~19.7 MB at L256 (plus the 22×22
  padded energy copy). Fits wasm, but stays in a session output tensor, never round-tripped.

## 4. ProtonPottsMPNN

Depends on **#5781** (port the different upstream lineage) and P7 (this branch). With P7's
alphabet object, the export is the same graphs at `V=30`, identity pair layout (spec 261007
§35.2). Extra work: V=30 weights, an alphabet entry in the export manifest, its own layer-b/c
parity against the ProtonPotts oracles once #5781 seals them. *Unverified:* whether
ProtonPotts design needs an extra runtime input (e.g. a pH or protonation-state control) — that
is #5781's spec to answer; this scope reserves a manifest field for it.

## 5. Acceleration levers (what we do differently this time)

1. **Skip the monolith and IREE.** Go straight to the split; IREE is not a target.
2. **Skip layer a.** Reuse the sealed Potts parity record; start at layer b.
3. **Front-load Phase 0 on the Potts-specific ops** (§3 list) in one spike, with every artifact
   `.run()` in ORT-CPU *and* ORT-Web — conversion alone is never a pass.
4. **Generalize once, not per family**: one export manifest (`alphabet`, per-graph input orders,
   buckets, checkpoint pin) read by the JS driver/loop; ProteinMPNN migrates onto it, so the
   existing split stays the regression check for the harness change.
5. **Ship in value order**: score graph `S` (no loop) → sample E/W/D/F → refine R → ProtonPotts.
6. **Carry the known traps into the scaffolding** (jax2onnx import-time `jnp` patching → JAX arm
   first plus a `sys.modules` guard; absolute `wasmPaths`; bytes not paths under Node; record the
   ORT-reported thread count; planted-delay timer control; finite-JSON results; exit 0 on graded fail).
7. **Fan mechanical work out to Haiku** (input-order tables, manifest plumbing, sidecar scaffolds),
   Opus verifies; parity gates run on titanix.

## 6. Decisions for the user

**261007 — user DECIDED 1 and 2 as recommended.** 1: ship `score:energy`/`ddg` + `sample` with
the default one-sweep refine first; `potts_converge` next; `nodes`/binding deferred. 2: build the
generic harness now; add ProtonPotts once #5781 seals its oracles. 3 (distribution) stays open —
no recommendation was made; X7 waits on it.

1. **First shipping target** — recommended: `score:energy`/`ddg` + `sample` with the default
   one-sweep refine; `potts_converge` next; `nodes`/binding deferred.
2. **ProtonPotts sequencing** — recommended: build the generic harness now, add ProtonPotts when
   #5781 seals its oracles (no throwaway V=21-only code).
3. **Distribution** — same open question as ProteinMPNN: `.onnx` via LFS under `release/browser/`
   (current) vs release assets; Pages deploy is manual and its allow-list omits the split files
   (filed as a separate bug).

## 7. Backlog

Filed 261007 under task `261007_potts-onnx-export`:

| id | item | depends on |
|---|---|---|
| #5809 | EPIC | — |
| #5810 | X0 Phase-0 spike on the Potts ops (convert **and** run) | — |
| #5811 | X1 export manifest: de-hardwire the split harness | — |
| #5812 | X2 score graph S + layer b/c | #5810, #5811 |
| #5813 | X3 sample split E/W/D/F, AR loop in JS | #5812 |
| #5814 | X4 refine R + `potts_converge` from JS | #5813 |
| #5815 | X5 knobs gate + benchmark | #5814 |
| #5816 | X6 ProtonPotts (V=30) | #5815, #5781 |
| #5817 | bug: Pages allow-list omits the shipped ProteinMPNN split | — |
| #5818 | X7 release packaging + Pages | #5815, #5817 |

X0 and X1 are independent and can run in parallel. Debt #2056 (the ORT-Web log-prob gap behind
the 2e-4 bar) is P3 investigation; the user accepted the bar change 261007.


## 8. X0 result (run `1340fe84`, graded `blocked` by record, 261008)

Verified by record: `status=completed`, `outcome=blocked`, `git_hash=50a70781`, `git_dirty=false`,
`sidecar_sha256` = committed file. jax 0.10.2, xtrax 0.4.0a11, onnxruntime 1.30.0 (CPU) and
onnxruntime-web 1.30.0 (wasm, Node, ORT-reported 1 thread). Bucket 128, seeded random init.
Both controls fired (perturbed-weight score, reversed-uniform decode).

| probe | ORT-CPU | ORT-Web | max rel (float) | ONNX size | watched ops |
|---|---|---|---|---|---|
| sched (`schedule_groups`) | pass | pass | 0 (ints exact) | 22 KB | ScatterND 2, TopK 3 |
| decode untied (`PottsARDecode`) | pass | pass | 4.5e-7; tokens exact | 3.46 MB | Loop 2, If 3, ScatterND 7, TopK 4 |
| decode tied | pass | pass | 4.5e-7; tokens exact | 3.46 MB | same |
| score (`absolute_energies`) | **not runnable** | **not runnable** | — | 4.35 MB | ScatterND 4, TopK 2 |
| refine (`PottsRefine` `potts`) | **not loadable** | **not loadable** | — | 178 KB | Loop 2, ScatterND 2 |

What this settles (scope §3 risk list):

- **`mode="drop"` scatters are NOT a blocker**: decode and sched carry 7 and 2 ScatterND with
  out-of-range sentinels and match exactly on both backends.
- **In-scan `lax.cond` and `lax.scan` convert and run** (ONNX `Loop`/`If`), tokens exact on ORT-Web.
- **No int64 graph I/O** on any probe.

Blockers (each a concrete rewrite for its item):

1. **score (X2, #5812)** — ORT has no `Where` kernel for BOOL (`node_Where_1116`: BOOL condition
   `And(ge, lt)` over a BOOL `Gather` of input `pad_valid`). It is a bounds-masked bool gather;
   candidates `etab.py:101`, `etab.py:204` (`jnp.where(in_range, pad_valid[...], missing)`) and the
   fill-mode gather `potts_head.py:103`. Plain bool gathers are fine (`decode.py:121` passed).
   Fix: do that masking in int32 and compare (`> 0`), semantics unchanged.
2. **refine (X4, #5814)** — ONNX type inference rejects a `Where` whose branches are `float` and
   `int32` (`node_Where_154`, inside the nested `Loop`). JAX had already unified the types, so this
   is a jax2onnx lowering defect on a weak-typed operand; the aminx-side workaround is an explicit
   dtype on the offending `jnp.where` operand once located.

**Recommendation for X3 (#5813), not yet decided:** the whole AR decode is one 3.46 MB graph that
already runs token-exact on ORT-Web wasm, unlike the ProteinMPNN monolith (19–31 MB, two decoder
arms behind `lax.cond`). For the wasm target the E/W/D/F split may be unnecessary; it remains
the right shape for WebGPU (control flow forces device copies). Proposed: ship encoder+etab graph
plus the single decode graph for wasm first, and keep the split as the WebGPU follow-up. Needs
a benchmark at L128/L256 (X5) before it is final.

### 8.1 Re-run after the fixes (run `241749b2`, graded `pass` by record, 261008)

Commit `159f38ee` ("remove the two ONNX export blockers"): `in_range & pad_valid[safe]` instead of a
BOOL `where` (`etab.py` merge_pair, `_batch_energy`), and explicit-dtype zeros instead of a weak `0`
(`etab.py` positional_potts_energy, `refine.py` untied/tied energy accumulators). Same sidecar
(sha256 `df731e18…`), clean tree. **All five probes pass on ORT-CPU and ORT-Web**: score 6.4e-7 /
2.8e-6 rel, sched exact, decode and decode_tied 4.5e-7 with tokens exact, refine exact.

Neutrality of the fix against the sealed upstream oracles, run per wave as `run_gate.py` does
(`AMINX_PORT_WAVE=<wave>`): potts_head 4/4, merge_pair_d2 4/4, merge_pair_d4 4/4, potts_energy 4/4,
potts_order 3/3; `tests/families/potts_mpnn/` green. (A combined run WITHOUT `AMINX_PORT_WAVE`
fails 7 tier-1 tests identically at main `dab58900` — the oracle fixture falls back to
`port_selftest`; that is invocation, not a defect.)

So every Potts device path the decided first target needs (score, decode, one-sweep refine) is
exportable as-is at random init. X2 onward measure the REAL checkpoint at L128/L256.

## 9. X2 gate result (run `d6506199`, graded `pass` by record, 261008)

Verified by record: `status=completed`, `outcome=pass`, `git_hash=000724c6`, `git_dirty=false`,
`sidecar_sha256=c89ab50b…` = committed file. Real checkpoint `pottsmpnn_20.pt` (sha256 `77e797fd…`),
sha-pinned example PDBs featurized by the driver path (3dkm L_total 72, 4jox 118, 3gg7 243; 4jox fits
both buckets, so 5 cells). Both controls fired.

**20/20 graph checks pass on ORT-CPU and on ORT-Web (wasm, 1 thread).** Worst relative float error
2.7e-6 (bar 1e-4); decoded tokens, ranks, orders and neighbour indices exact; refine exact.

| graph | L128 | L256 | ops of note |
|---|---|---|---|
| encode (encoder + head + merged/padded etab) | 4.31 MB | 4.33 MB | ScatterND 4, TopK 2 |
| energy (score / ddG) | 19 KB | 19 KB | — |
| decode (AR, in-graph Loop/If) | 3.46 MB | 3.59 MB | Loop 2, If 3, ScatterND 7, TopK 4 |
| refine (one `potts` sweep) | 182 KB | 183 KB | Loop 2, ScatterND 2 |

About 7.8 MB per bucket, comparable to the ProteinMPNN split (6.96 MB). Manifest
`MANIFEST.json` sha256 `d35365c2…` (X1 format + `family`, `etab_alphabet`, checkpoint hash). The
artifacts live in the titanix export checkout (`outputs/potts_export_gate/work/`) and are NOT committed:
distribution (decision 3) is open.

What remains for the decided first target: the browser-side loop (JS: encode -> decode -> energy ->
refine, with JS-generated noise/uniforms/order) graded token-exact against JAX under Node and headless
Chromium (X3/X5), the runtime-controls gate, and the benchmark. Layer b/c of every graph is done.

## 10. X3: the shipping JS loop and input builder (261008)

**Loop gate** `potts_loop_gate.py` — run 1 `b0e1e1b8` graded `instrument_unverified` by record: all
15 samples agreed and the token control fired, but the cross-seed ENERGY control did not
(low-temperature draws a few residues apart can sit within 1e-4 relative energy). The energy
control was replaced by the X0/X2 planted difference (perturbed Potts head, same sequence) in
`28ea7bea`, before the re-run; bars and per-sample criteria unchanged. Run 2 **`2917e367` graded
`pass` by record** (clean, `28ea7bea`, sidecar `ea8c0ff5…`): the shipping
`browser/potts-sampler/potts_sampler.mjs` under Node wasm (1 thread), on the X2 artifacts verified
by sha256, reproduces the JAX sampler on 5 cells x 3 seeds = 15 samples — decoded tokens, decoding
order and refined tokens exact; sample energy <= 2.8e-6 relative; both controls fired.

**Input builder** `browser/potts-sampler/potts_inputs.mjs` (PDB text -> the graphs' input arrays)
matches the Python driver path (`parse_pdb_upstream -> _featurize_one -> prepare_sample`)
**bit-exactly** on 7 dumps: 3dkm L128/L256, 4jox L128/L256, 3gg7 L256, 2yc3 L512, 6w25 L512 (two
chains) — `node --test` 6/6 with `POTTS_INPUTS_FIXTURE_DIR`. Negative control: one corrupted
`residue_idx` entry in the 6w25 dump fails the test. (A deterministic equality test, not a
measured finding, so no sidecar.)

So the decided first target runs end to end in JS with no Python: PDB -> inputs -> encode ->
decode -> energy -> one-sweep refine, exact against aminx.

**Still open for shipping** (X5 #5815 and follow-ups):
- `temperature` is baked into the decode graph (0.1); the user's P07 rule is that controls are
  runtime inputs. Needs `PottsARDecode` to accept an array temperature (src change) and a re-export.
- The energy graph has a fixed 8-row batch; the loop fills unused rows. Fine for sampling; DMS
  scoring wants a row count chosen per bucket, or a loop over 8-row chunks in JS.
- Headless Chromium run (layer c in a real browser, COOP/COEP, threads) and the benchmark.
- `potts_converge` JS driver (X4 #5814); `nodes`/binding modes deferred.
- Distribution (decision 3).

## 11. Temperature as a runtime input (#5815 item 1, 261008)

`e512352d`: `PottsARDecode` accepts temperature as a float (unchanged path) or a scalar array
(`floor_temperature_array`, same 0 -> 1e-6 floor); the decode and refine graphs take
`temperature` as their last input (shape `[1]`, recorded in the manifest; refine is fed the raw
`optimization_temperature` and floors in-graph). Unit test `test_decode_temperature_input.py`
(array == float under jit at 0.0/0.1/1.0; a hotter temperature changes the draw) and the Potts
family suite pass on titanix (88 passed, 1 expected skip); `potts_order` port wave 3/3.

- X2 gate re-export **`3cb2974b` pass by record** (clean, `e512352d`): 20/20 on ORT-CPU and
  ORT-Web, same per-graph errors as `d6506199`. Manifest sha256 `7e6db052…`.
- Loop gate **`7f760bde` pass** (cool-tier record; the warm index row was stale `running`):
  15/15 token-exact, energy <= 2.8e-6 rel, both controls fired, on the `7e6db052…` artifacts.
  (Run `242518a8` errored before grading on a harness bug — `float()` of the `[1]`-shaped
  temperature — fixed in `f2f50a7a`.)

These `7e6db052…` artifacts supersede the `d35365c2…` set for everything downstream.

## 12. X5 runtime controls (261008)

`browser/potts-sampler/potts_controls.mjs` (`applyPottsControls`) maps the sample path's design
controls onto the graph inputs exactly as `prepare_sample` does: `fixed_positions` (zero
`chain_m_pos`, out-of-range ignored), `omit_aa` (`omit_letter_indices`), `bias` (global 21 /
per-position L / per-residue Lx21, `_split_bias` branch order). Deliberate deviation: non-finite
bias or fixed-position values throw in JS (Python would accept them). Fixture comparison against
12 Python dumps (omit, fixed incl. an out-of-range row, global/per-position/per-residue bias,
combined; 3gg7 L256 and 3dkm L128): **exact**, `node --test` 17/17.

Runtime-controls gate **`65082719` pass by record** (cool tier; clean, `06103c26`, sidecar
`a6d9a564…`): the FULL browser path under Node wasm — PDB text -> `buildPottsInputs` ->
`applyPottsControls` -> `sample` — equals aminx on 9 cases x 2 cells x 2 seeds = **36/36 samples
exact** (tokens, order, refined; energy <= 1e-4 rel), and every control acts: omitted letters absent,
fixed rows native, W bias raises W, per-residue P bias raises P in rows 0-29, temperature 1.0 changes
every draw, refine temperature 0.5 changes a refined sequence, refine-off leaves decode unchanged.
Tied positions excluded (not wired on the Python sample path; debt 2443).

Remaining for shipping: headless Chromium (real browser, COOP/COEP, threads) + benchmark; energy-graph
row count for DMS scoring; `potts_converge` JS driver; ProtonPotts (#5816, blocked on #5781);
distribution (decision 3).
