---
title: aminx / xtrax separation of concerns - stale knobs, duplicated contracts, misplaced logic
description: Deep audit (4 domain auditors, spot-verified) of where aminx still owns generic run, IO, execution, config and training contracts that xtrax should own, where xtrax holds aminx-specific logic, and the debts and sprint shape to fix it.
task_id: 261007_aminx-xtrax-separation-audit
status: final
---

# aminx / xtrax separation of concerns

**Ask (user, 2026-10-07):** "output_h5_path should no longer be a knob. that is stale from when we didn't have xtrax to manage the contract. we need to audit for this kind of thing. everything should be using xtrax. ... a deeper audit so we can pursue a bigger sprint to get aminx and xtrax properly respecting separation of concerns and avoid duplication."

**Baseline:** aminx origin/main `4572e70b` (#195 and #196 merged); xtrax = installed `0.4.0a11` wheel. The `~/projects/xtrax` checkout is behind the wheel (no `export/`, no `profiling/loop_scaling.py`) and was not used as the reference.

**Method:** four read-only Sonnet auditors, one per domain (run contract/IO, execution/planning, spec/config/CLI/seeding, training/export/agent), each required to cite file:line on both sides and to read code, not comments. I then re-checked the claims that set priority or touch numerics against the code (listed under *Verification*). Grep counts are approximate. This is a code-reading audit; no number in it is a measurement.

The earlier capability inventory (`261006_xtrax-usage-audit.md`, debt #2493) asked "which xtrax symbols does aminx use". This one asks "who owns which contract". It also corrects the earlier audit in two places (see T3 and E4 below).

## Ownership rule used

- **xtrax owns** everything generic: the run contract (run identity, seed, output sinks and their completion receipts, provenance capture, atomic writes, schema versioning, layered config resolution), execution (axis planning, dispatch, miscompile guards, bucketing primitives, memory estimates), training loop machinery, and export tooling.
- **aminx owns** protein/MPNN semantics: which arrays have a residue axis, the decode and scoring math, the sampling skip guards, domain provenance attributes, campaign *row* semantics (unit input hash, grid lineage), and model weights.
- A knob belongs in the aminx spec only if it changes protein-domain behaviour. Output routing, run identity, flush cadence and batch-size overrides do not.

## Headline problems

1. **Output routing is a stale aminx knob.** `output_h5_path` never writes HDF5. Its name and help text disagree ("HDF5" on some commands, "Zarr" on others). It is one of three overlapping location knobs (`output_dir` has no runner reader; `cache_path` is the JAX compile cache). Only `sample` and `jacobian` stream; `score` and `inspect` raise `NotImplementedError`. The in-memory path runs on a separate aminx sink. → **#2519** (P1), blocked by xtrax **#2523**.
2. **Runs have no identity.** aminx never fills xtrax `RunSpec.seed` or `run_id` (`build_run_spec` reads attributes no spec class has), so every sink mints a random id. There are three run-id formats and three incompatible `spec_sha256` computations. → **#2529** (P1).
3. **`random_seed=0` silently becomes 42** at every runner entry point. → **#2528** (P1).
4. **Spec JSON silently drops unknown keys.** There is no schema version, so removing any knob would silently change what saved specs mean. This is the prerequisite for every deletion below. → **#2530** (P1).
5. **Execution exists twice.** There are three live copies of axis dispatch, aminx-native strategy classes that only round-trip to xtrax, and `safe_map`/`safe_scan` duplicates. The #2391 size-1 miscompile guard is applied inconsistently inside aminx and is missing from xtrax entirely. → xtrax **#2520** (P1), **#2521**; aminx **#2533**.
6. **Training has real bugs** in a hand-rolled loop that should be Engine: grad accumulation cannot run, the two optimizer paths decay different parameters, and resume discards the restored key. → **#2535** (P1); later **#2536** after xtrax **#2525**.
7. **Misplaced in xtrax:** `BUCKET_LADDER` lives in `export/rings.py` but runtime code imports it, and `rings` also carries protein-specific input generators. → xtrax **#2522**, **#2526**.

## Findings by domain → debt

Prefixes: R run/IO, E execution, C spec/config, T training/export/agent. Classes: STALE (knob whose contract xtrax owns), DUP, DIV (divergent), MIS-A (generic logic in aminx), MIS-X (aminx logic in xtrax), GAP-X (xtrax lacks it), DEAD.

### Run contract and IO

| id | class | finding | debt |
|---|---|---|---|
| R1 / C1 | STALE | `output_h5_path`: about 40 src references, 8 CLI commands, not HDF5, score and inspect refuse it, campaign dirs named `<hash>.h5` and done markers keyed on that path | #2519 |
| R2 | STALE | `output_dir` has no runner reader; inferred from `cache_path` (the compile-cache dir) | #2519 |
| R3 | STALE | `IOConfig.sink_kind` mirrors `SinkSpec.format` | #2519 |
| R4 / C2 / C14 | DIV | `RunSpec.seed`/`run_id`/`axes` never populated; random id per sink; three id formats; rerun to the same store is refused | #2529 |
| R5 / T14 | GAP-X + DUP | xtrax git capture uses `Path.cwd()` (wrong for wheel installs); aminx `agent/provenance` is better; `telemetry.record` is a third copy | #2523, #2537 |
| R6 | GAP-X | provenance attrs hand-staged at 4 sites; no digest-algorithm version; one schema for every store level | #2523, #2537 |
| R7 | MIS-A | generic lease lock, marker and promote code in `campaign.py` | #2421 (updated) |
| R8 | DIV | four completion contracts for one Zarr store; `finalize()` never called; a store left by an exception looks complete | #2523, #2540 |
| R9 | GAP-X | no in-memory sink, so every verb has two output paths | #2523, #2540 |
| R10 / E14 | DEAD / MIS-A | unused aminx sink classes; raw `io_callback` instead of xtrax's pinned shim; session pattern copied 4 times | #2532, #2540, #2527 |
| R11 | GAP-X | `ZarrStagingSink` cannot append; aminx concatenates in memory | #2523 |
| R12 | STALE | jacobian `combine*` knobs; `combine_batch_size` repurposed as flush cadence; h5py kept only for dead `catjac` | #2531 |

### Execution and planning

| id | class | finding | debt |
|---|---|---|---|
| E1 | GAP-X + DIV | size-1 vmap guard (#2391) only in aminx `safe_map`; xtrax `chunked_map` and its planner (demotes to batch 1) lack it; every xtrax-iterator path in aminx is unguarded | xtrax #2520, #2533 |
| E2 | DUP | axis dispatch copied three times plus one dead copy; aminx Scan silently drops `CarrySpec.transition` | #2521, #2533 |
| E3 | STALE | aminx-native strategy classes exist only to round-trip to xtrax | #2533 |
| E4 | DUP | `safe_map`/`safe_scan` vs xtrax. Correction to the 261006 audit: the `batch_size=0` sentinel is not a real divergence; only the guard is | #2533 |
| E5 | STALE | `use_unified_driver=False` legacy paths and flag | #2533 |
| E6 | STALE | samples/temperature/noise/apc batch-size knobs have no readers | #2531 |
| E7 | GAP-X | planner ignores `AxisSpec.heterogeneous` in budget mode; no `BatchPlan.decision_for`; three disagreeing frozensets in aminx | #2521, #2533 |
| E8 / E9 | GAP-X + MIS-A | single-axis plan-and-dispatch copied about 7 times with divergent memory estimates (overlaps #2157); generic product estimator in aminx | #2521, #2533 |
| E10 / C8 | MIS-A + GAP-X | four layered config resolvers in aminx; xtrax has none | xtrax #2524, #2537 |
| E11 / T10 / T12 | MIS-A + MIS-X | generic half of `host/bucketing.py` belongs in `xtrax.tiling`; `BUCKET_LADDER` stranded in `export.rings`; two ladder wrappers | xtrax #2522, #2537 |
| E12 / E16 | DEAD | `tiling/bucketing.py`, `tiling/pad.py`, `bucket_config`, `inference/driver.py`, unused exception layer | #2532 |
| E13 | DUP + DEAD | `_validate_plan_topology` never called; StageSet fusion/sink slots duplicate `AxisBoundary`; global `eqx.Module.__hash__` monkeypatch at import | #2534, #2527 |
| E15 | GAP-X | AR decode hand-rolls while-loop-with-ys; stale `CarryShape` shape | #2527 |
| E17 | DIV | `multistate_poe` is a second sampler; `runner.sample` refuses multi-state only because the MVP was wired into campaigns | #2126 (updated) |

### Spec, config, CLI, seeding

| id | class | finding | debt |
|---|---|---|---|
| C3 | DIV (bug) | `random_seed or 42` turns 0 into 42 | #2528 |
| C4 / C5 | DEAD | about 12 fields with no reader; 6 `--average-encoding-mode` flags never passed into the spec; `n_devices` serialized only | #2531 |
| C6 / C7 | STALE + GAP | deprecated kwargs accepted silently; no `schema_version`; unknown keys dropped; xtrax.config unused | #2530 |
| C9 / T15 | DUP / DIV | local `_canonical_json_bytes` is byte-identical to xtrax's; three incompatible `spec_sha256` variants | #2529 |
| C10 | GAP-X (decision) | generic execution config (output root, devices, precision, shard lineage) lives only in aminx sub-configs | decision D4 |
| C11 | GAP-X (decision) | chunk- and resume-invariant key stream is aminx-only; the derivation must not change | decision D5 |
| C12 | DUP | training spec's optimizer, loop and checkpoint block overlaps xtrax `TrainConfig` | #2536 |
| C13 | covered | max_length / bucketing (S8, PR #197) | — |

### Training, EBM/Potts, export, agent

| id | class | finding | debt |
|---|---|---|---|
| T1 / T4 / T5 | GAP-X + DUP | same train-step body in `ebm/training.py` and `training/trainer.py`; byte-identical `init_state`; re-export wrappers; Engine lacks hooks aminx needs | xtrax #2525, #2536 |
| T2 / T3 / T5 | bugs | grad accumulation cannot run; warmup and no-warmup optimizers decay different params. Correction to the 261006 audit, which said biases decay everywhere. Resume discards the restored key | #2535 |
| T6 / T13 | DEAD | `train_diffusion.py`; `aminx/profiling/*` | #2532 |
| T7 / T8 / T9 | DUP + GAP-X | ONNX RNG audit, jaxpr walkers (3 copies plus 6 in tests), `convert_to_onnx` missing `model_name` and external-data embedding | xtrax #2526, #2538 |
| T11 | MIS-X | `rings.symmetric_geometry` / `sub_k_neighbours` are protein generators in xtrax | xtrax #2526 |
| T16 | GAP-X + DUP | no xtrax atomic write; 6 hand-rolled sites | #2523, #2537 |
| T17 / T19 | DUP / STALE | duplicated parity metrics; tooling pointed at the stale `~/projects/xtrax` checkout | #2538 |
| T18 / T20 | DUP + GAP | Potts: no planner, unbounded vmaps, uses state 0 only, weights bypass provenance | #2539 |

## Verification

These claims were re-checked against the code at `4572e70b`. All held.

- `random_seed or 42` at `run/spec.py:352` and `host/runner.py:857, 1192, 1349, 1681, 1798`.
- `build_run_spec`: `seed=getattr(spec, "seed", 0)`, `axes=getattr(spec, "axes", [])` (`run/spec.py:381-382`).
- `_validate_plan_topology` has no caller in `src`.
- No runner reads `output_dir`. `temperature_batch_size` and `noise_batch_size` have no readers outside spec, CLI and comments. Nothing imports `aminx.profiling`.
- `score` and `inspect` raise on `output_h5_path` (`runner.py:735, 1310`). `_infer_output_dir` falls back from `output_dir` to `output_h5_path.parent` to `cache_path`.
- Optimizer paths: `adamw_with_schedule` when `warmup_steps > 0`, raw `optax.adamw` otherwise (`training/trainer.py:95-110`).
- Accumulation concatenates inside the scan carry and does `tree_map(*grads_list)`. Read, not executed.
- Resume rebuilds `ResumableState` with `PRNGKey(spec.random_seed)` after restoring (`trainer.py:182-215`).
- xtrax `chunked_map` has no size-1 handling (`transforms/map.py`).
- `eqx.Module.__hash__` is reassigned at `sampling/conditional_logits.py:54`.

Still unverified: the T2 crash (not executed), whether anything jits over `run_spec` (R4 caveat), `convert_to_onnx` external-data behaviour (T9), and whether repos outside aminx read `output_h5_path` or `done.json`.

## Sprint shape

The order is forced by two facts. Nothing can be deleted until spec JSON fails loudly on unknown keys. Most aminx collapses wait on an xtrax release.

**Wave 0: aminx only, independent, small.** Each item is its own PR.
- #2528: seed 0. Pre-registered numerics change for seed 0 only.
- #2532: dead-code sweep.
- #2530: spec JSON schema version and unknown-key rejection.
- #2535: training correctness. Pre-registered, needs decision D6.
- #2534: wire or delete topology validation; remove the global hash patch.
- The cheap parts of #2533 (legacy `use_unified_driver` paths; consistent size-1 guard) and of #2538 (stale-path references, parity-metric dedupe).

**Wave 1: xtrax release (0.4.0a12 or later).** Order by what unblocks the most aminx work:
1. #2520 (size-1 guard, P1)
2. #2523 (run contract: memory sink, append, finalize receipt, run_id/seed, provenance injection, atomic write; P1)
3. #2521 (planner/dispatch)
4. #2522 (bucketing primitives)
5. #2524 (layered resolver)
6. #2525 (Trainer/Engine)
7. #2526 (export)
8. #2527 (stages)

**Wave 2: aminx onto the new xtrax.**
- #2529: run identity.
- #2531: dead fields and flags (needs #2530).
- #2519: retire `output_h5_path` and the two sibling knobs; needs #2530, #2529 and xtrax #2523; includes the dual-read migration for finished campaigns.
- #2540: one result contract per verb.
- #2533: execution collapse; retire the T2.GATE baselines with the copies.
- #2537: resolvers, bucketing, provenance.
- #2536: training onto Engine.
- #2538 (rest) and #2539 (Potts).

**Wave 3: numerics decisions.** #2126 (multi-state PoE folded into `_sample_batch`; sample keys change).

Gates that pin today's behaviour bit for bit and must be retired or re-baselined deliberately: the T2.GATE golden and dispatch-parity tests, the EBM score-matching parity tests, the p07/layer B/C ORT and IREE bars, campaign resume hashes and the grid seed-hash pin, `tests/host/knob_observations.py`, `host/spec_partition.py`, and the browser-validation inventory.

## Decisions for the user

- **D1** (#2528): fix seed 0 → 42? It changes outputs for seed 0 only.
- **D2** (#2521/#2533): should a user `CarrySpec.transition` execute? aminx silently drops it today.
- **D3** (#2526): `rings` protein generators: parameterise in xtrax, or move to aminx's export layer?
- **D4** (C10): upstream a generic execution-config section (output root, device count, precision, shard lineage) into xtrax `RunSpec`, or keep it in aminx?
- **D5** (C11): upstream a chunk-invariant key-stream utility to xtrax, keeping aminx's frozen derivation byte-identical?
- **D6** (#2535): which weight-decay policy is canonical: masked (no decay on 1-D params, the warmup path today) or unmasked? Either way, one of the two paths changes numerics.
- **D7** (#2126): accept that unifying multi-state sampling changes PoE sample keys?
- **D8** (#2531): delete the inert batch-size knobs, or wire them to a generic per-axis override?

## Related

- `261006_xtrax-usage-audit.md` (#2493): capability inventory. Corrected here on T3 and E4.
- `.praxia/docs/specs/261002_streaming-xtrax-migration-scope.md`: its S6 (rename `output_h5_path`) is superseded by #2519.
- `.praxia/docs/specs/261001_spec-system-unification.md` (draft S1): CLI flags generated from the field registry. Pairs with #2530 and #2531.
- Open debts this builds on: #2126, #2157, #2421, #2493, #2517.
