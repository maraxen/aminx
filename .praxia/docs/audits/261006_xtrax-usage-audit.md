---
title: What aminx does and does not use from xtrax (0.4.0a11)
description: Static inventory of all 619 public xtrax symbols against aminx (bathos da99c566, pass) plus per-capability verdicts. Includes duplicates to replace, gaps worth adopting, things that don't apply, and the native hook for runner length bucketing.
task_id: 261006_xtrax-native-audit
status: final
---

# What aminx does and does not use from xtrax

- **Debt:** #2493 (audit and replace aminx duplicates with xtrax-native). Related: #2371, #2391, #2421, #2156, #2157, #2358.
- **Pinned xtrax:** `0.4.0a11`, as in `pyproject.toml` and `uv.lock` on `origin/main`. The source read was the installed wheel, not the stale `~/projects/xtrax` checkout.
- **aminx commit scanned:** `origin/main` at the base of branch `wt/261006-xtrax-usage-audit`.

## 1. Inventory (facts)

**Method.** `scripts/audit/xtrax_usage_inventory.py`, with sidecar `scripts/audit/xtrax_usage_inventory.bth.toml`, was committed before the run.
- **Scan.** It uses `ast` only and imports neither package. Re-exports resolve to the defining module. Function-local imports and attribute chains through imported xtrax modules are counted.
- **Run.** bathos `da99c566-a2fe-4753-8e8a-90bd36629fda`, status `completed`, outcome **`pass`**.
- **Controls.** All held: `select_bucket` was detected in `src`, `BucketIterator` was reported unused, and a planted import of `BucketIterator` was detected. Zero parse errors.
- **Output.** `outputs/audit/xtrax_usage.json` (per-symbol file lists) and `outputs/audit/group_*.tsv`.
- **Limits.** Being static, it does not see dynamic `getattr` or `importlib` access.

| | count |
|---|---|
| xtrax modules | 174 |
| public symbols | 619 |
| used by aminx `src` | **45** |
| used only by tests or scripts | 16 |
| no static reference anywhere | **558** |

| subpackage | symbols | src | tests/scripts only | unused |
|---|---|---|---|---|
| tiling | 52 | 20 | 2 | 30 |
| run | 40 | 10 | 0 | 30 |
| stages | 24 | 6 | 0 | 18 |
| checkpoint | 3 | 3 | 0 | 0 |
| training | 14 | 3 | 0 | 11 |
| eda | 18 | 2 | 1 | 15 |
| engine | 6 | 1 | 0 | 5 |
| export | 83 | 0 | 5 | 78 |
| profiling | 38 | 0 | 7 | 31 |
| data | 2 | 0 | 1 | 1 |
| transforms | 3 | 0 | 0 | 3 |
| inference | 24 | 0 | 0 | 24 |
| composition | 29 | 0 | 0 | 29 |
| config | 6 | 0 | 0 | 6 |
| distributed / sparse / safety | 5 / 7 / 5 | 0 | 0 | all |
| telemetry | 62 | 0 | 0 | 62 |
| cli | 46 | 0 | 0 | 46 |
| loop | 132 | 0 | 0 | 132 |
| findings / tombstone / jaxlint_runner | 11 / 6 / 3 | 0 | 0 | all |

**Used in `src`:**
- **tiling.** `BatchPlanner`, `BatchPlan`, `AxisSpec`, `AxisDecision`, `MemoryBudget`, `BudgetInfeasibleError`, the strategies `Vmap`, `ChunkedMap` and `Scan`, the iterators `MapIterator`, `VmapIterator`, `ScanIterator` and `JaxScanIterator`, `make_axis_dispatch`, `DispatchRejected`, `CarrySpec`, `CarryShape`, `DedupSpec`, `lowered_memory_estimate` and `select_bucket`.
- **stages.** `Fuse`, `Tap`, `Sink`, `AxisBoundary`, `validate_plan_topology` and `PlanTopologyError`.
- **run.** `RunSpec`, `SinkSpec`, `derive_sink_spec`, `ZarrStagingSink`, `new_run_id`, `zarr_content_digest`, `canonical_json_bytes` and the `fsync_*` helpers.
- **checkpoint.** The three `orbax` functions.
- **training.** `adamw_with_schedule`, `make_optimizer` and `ResumableState`.
- **engine.** `Engine`, used by `aminx.ebm` only.
- **eda.** `explain_plan` and `extract_plan_stats`.

**Stale imports.**
- `scripts/ebm/benchmarks/langevin_benchmark.py` imports `xtrax.tiling.SafeMap`. It resolves only through the deprecation alias in `xtrax/tiling/__init__.py:83`, which is kept "for one release", so it breaks on the next xtrax bump.
- `src/aminx/host/plan.py:28` imports `DedupSpec` from `xtrax.tiling`, which does not export it. The import is `TYPE_CHECKING`-only, so it cannot fail at runtime, but a type checker sees an unexported name.

## 2. Is there an xtrax `InferencePlan`?

No.
- **xtrax.** There is no class by that name in xtrax 0.4.0a11. `xtrax.stages` is a generic, domain-free toolkit:
  - a `StageBundle` slot container;
  - `Fuse`, `Tap`, `Sink` and `AxisBoundary` boundaries;
  - `validate_plan_topology`;
  - `execute_map_axis` and `execute_scan_axis`, which fire taps and sinks per step;
  - an `EvaluateFn` registry.
- **aminx.** `InferencePlan` (`host/plan.py:559`) is a domain layer on top, not a reimplementation. It holds the model, encoder, `StageSet`, decoder and packer, plus the `filter_jit` encode, decode, sample and score paths. `make_inference_plan` (`:862`) resolves the decode mode. xtrax has none of this.
- **`StageSet`** (`types/stages.py:389`) is aminx's richer, domain-specific slot bag. It does not subclass xtrax `StageBundle`, and it doesn't need to.
- **Open finding: the boundaries are inert.** `StageSet.axis_boundaries` is validated (`host/plan.py:499-503`) but never populated: nothing in `src` passes `axis_boundaries=`, and `mbr_consensus.py:6` says so. The xtrax stage executor is therefore unused, and aminx has no equivalent. Either wire the boundaries through `execute_map_axis` / `execute_scan_axis`, or drop the field.
- **Dead code.** `host/stage_adapter.py` `StageBundleAdapter` is imported only by its own test and the browser-validation path inventory.

## 3. Verdicts by capability

Verdicts:
- **DUPLICATE:** replace.
- **DIVERGENT:** replace after reconciling semantics.
- **GAP:** worth adopting.
- **N/A:** does not apply.
- **KEEP:** a deliberate aminx layer.

### Execution core (`tiling`, `transforms`, `stages`, `inference`)

| xtrax | aminx counterpart | verdict | note |
|---|---|---|---|
| `transforms.chunked_map` | `utils/safe_map.py` (used at `host/kernel_dispatch.py:68,76`) | DIVERGENT | aminx carries the #2391 size-1 vmap guard and a `batch_size=0` means "vmap all" sentinel. Upstream the guard first (#2371), then re-export and delete. |
| `transforms.safe_scan` | `utils/safe_scan.py` | DIVERGENT, near-duplicate | Signatures differ: xtrax takes `(fn, init, xs, length, reverse, unroll)`, aminx `(f, xs, *, init)`. Error rules differ too. Three src call sites. |
| `tiling.axis_dispatch` | `host/kernel_dispatch._dispatch_axis` (9 sites), `ebm/plan.dispatch_axis` (5) | DIVERGENT | aminx routes Vmap through `safe_map` (the guard), passes a carry through Scan, and accepts both SafeMap and ChunkedMap names. Replace after the guard lands upstream. The ebm copy exists to keep `aminx.ebm` free of `aminx.host`. |
| iterators, `make_axis_dispatch` (aminx copies) | `tiling/iterator.py`, `tiling/dispatch.py` | DUPLICATE, gated | Dead in production. They are kept as the T2.GATE bit-for-bit baseline (`dispatch.py:1-20`). Retire the gate, then delete. |
| `select_bucket` / `bucketize` / `Bucket` | `tiling/bucketing.py` (64–512), `tiling/buckets.py` (100–1200, no src callers), `export/buckets.py` (128–1024, already wraps xtrax `select_bucket`) | DUPLICATE (selectors), GAP (`bucketize`) | aminx pads on device with `jnp.pad` (`buckets.py:47`, `tiling/pad.py:14`). xtrax warns that this is shape-specialised and recompiles. `bucketize` pads on the host with numpy. |
| `AxisSpec.bucket_boundaries` → `Bucket` plan | `N_RESIDUES` (`tiling/axes.py:27`) leaves it unset, and the runner never plans the residue axis | GAP | See section 4. |
| `host/plan.plan_bucketed` | — | dead / DIVERGENT | Test-only. Its own NOTE flags it overriding `n_structures` cardinality with a sequence length. |
| `device_memory_budget` | inline `memory_stats()["bytes_limit"]` at `host/plan.py:279` and `tiling/planner.py:109` | DIVERGENT | aminx silently falls back to 4 GiB, while xtrax raises when stats are missing. |
| `lowered_memory_estimate` | used (`runner.py:597,645`, `multistate_poe.py:245`) | in use | #2157 tracks migrating the remaining hand-typed estimates. |
| `dedup_synthesis` (`synthesize_dedup_spec`, `verify_*`, `merge_*`) | none; callers declare `dedup_specs` by hand | GAP, conditional | Could help repeated-backbone batches. Benchmark first. |
| `WhileCarry` / `WhileLoopIterator` | none | GAP, low | Nothing needs it today. |
| `inference.analyze_cse` / `memoize_jaxpr` | none | GAP, low | Encode is already hoisted by `InferencePlan`. |
| `inference.infer_bundle` / `synthesize_axes`, `sparse`, `distributed` | none | N/A | aminx declares its axes explicitly. There is no sharding or sparsity in `src`. Revisit `distributed` for multi-GPU work. |
| `stages.StageBundle`, `EvaluateFn`, protocols | `StageSet` | N/A / KEEP | |

### Run, training, I/O

| xtrax | aminx counterpart | verdict | note |
|---|---|---|---|
| inference work units / resume / locks | `host/campaign.py` (locks `:375-550`, done markers `:796`, `execute_manifest :1604`) | **absent in xtrax** | `cli/resume_verb.py` is training-only. Replacing the campaign path means building `xtrax.run.units` upstream first (#2421, `specs/261002_streaming-xtrax-migration-scope.md`). |
| `run.RuntimeBundle` / `InputResolver` | none built; `io/input_uri.py` and `io/proxide_fetch.py` are URI→file fetchers, a different layer | N/A until #1910 | ADR 260630: compose, don't subclass. |
| `run.make_sink` | `ZarrStagingSink(derive_sink_spec(...))` called directly | trivial | Adopt only if jsonl or h5 sinks land. |
| `run.freshness.evaluate_freshness` | none | GAP, low | Release and human gates only. |
| `run.seed_emission`, `repro_floor`, `baseline_budget_emission`, `component_binding` | none | N/A | bathos and agentic-loop bridges. aminx seeds via `jax.random.key` plus `fold_in` of sha256 words (`host/_sampling_grid_lineage.py:160-180`). |
| `training.Trainer` / `create_train_step` / `Engine` | `training/trainer.py:294,712` hand-rolled loop; `aminx.ebm` already uses `Engine` | DIVERGENT | aminx has a per-model loss, micro-batch scan, mixed precision and two checkpoint managers. The xtrax step must accept those first. Swap safety is unassessed. |
| `training.grad.accumulate_grads` | `trainer.py:487-526` | DIVERGENT | aminx also returns aux logits and metrics, so this needs `has_aux` upstream. |
| `training.optim.no_bias_wd_mask` | none; aminx decays biases and LayerNorm | GAP, small | This changes training numerics, so it needs a deliberate decision. |
| `training.state.init_state` | hand-built `ResumableState` (`trainer.py:175,210`) | DUPLICATE, trivial | |
| `data.DataModule` / pipeline | proxide `create_protein_dataset` | N/A | `create_distributed_pipeline` is a stub. |
| `config` (TOML) | none; aminx specs are JSON | N/A | |
| `checkpoint` beyond orbax | `io/weights.py` (`.eqx.zst` with provenance) | N/A / KEEP | |
| `run.ident.new_run_id` | used in `io/designs.py`; the agent layer on PR #195 mints its own timestamped id | DIVERGENT (PR #195 only) | |

### Tooling (`export`, `profiling`, `eda`, `telemetry`, `cli`, `composition`, `loop`)

| xtrax | aminx counterpart | verdict | note |
|---|---|---|---|
| `export.rings` + `divergence` | `scripts/browser_validation/layer_b_iree.py:116` hardcodes `RINGS = "not_available"`, reasoning that a10 had no rings module | GAP | The pinned a11 has it. Adopt it in layer B/C parity, and fix the stale a10 comment. |
| `export.onnx.convert_to_onnx` | nine scripts call jax2onnx `to_onnx` directly (`p07_split_*`, `p07_knobs_gate`, `layer_b_build`, `layer_b_ort_calibrate`, `layer_c_calibrate`, `jax2onnx_spike`) | GAP, already planned | `specs/261001_xtrax-model-contract.md:49` (AC6) routes the four-graph split through `convert_to_onnx`. It refuses x64, restores the jnp namespaces jax2onnx patches, refuses ONNX RNG ops (`onnx.py:330`) and pins opset 23. `export_pipeline` does not fit, because it needs a `BatchPlan`. |
| `export.onnx.find_onnx_rng_ops` | `scripts/browser_validation/onnx_audit.py:62` | DUPLICATE (scripts) | Confirmed by `specs/261001_xtrax-model-contract.md:190-191`. Once exports go through `convert_to_onnx`, the aminx walker is needed only for artifacts produced some other way. |
| `export.parity.verify_native_parity` | layer B/C bespoke compare with calibrated tolerances | DIVERGENT | Reconcile with the `compare_pytree` budgets first. |
| `export.safety` | `export/rng_audit.py` | KEEP both | They catch different failures: the xtrax rule is onnx-only and keyed on input-derived keys. |
| `export.compile` / `targets` | used by layer B scripts | in use (scripts) | |
| `export.hf_weights`, `spirv` | none | N/A | Weights are `.eqx.zst`. `load_hf_weights` calls `hf_hub_download` with no revision (`hf_weights.py:109`), so it cannot pin what aminx's `HF_REVISION` pins. |
| `profiling.loop_scaling` | none; benchmarks hand-roll `perf_counter` | GAP | It cites an aminx autoregressive O(L²) bug (#1983). A scan-extent check on the AR decode would be a cheap CI test. |
| `profiling` record/claims/trace | used by layer B/C scripts | in use (scripts) | aminx `profiling/` (vendored `hlo_tools`, `sampler_profile`) overlaps partially and is low value to migrate. |
| `eda` | `tiling/eda.py` re-export shim (0 callers) | in use / dead shim | |
| `telemetry` (ledger, IR capture) | stdlib logging; cisternal telemetry arrives on PR #195 | GAP, needs an ownership decision | Decide whether cisternal or xtrax owns the run ledger before adopting either. |
| `cli` REGISTRY | `cli.py` is Typer; xtrax verbs compose into tyro | N/A | Framework mismatch. The CLI port is deferred (S7-Q1). |
| `composition` (`HostPrepGraph`) | none | N/A | Confirmed by both hub specs (`261001_pipeline-editor.md:121-128`, A1/A2; `261001_xtrax-model-contract.md` A1–A3). Nodes hold a live `module:symbol` callable and edges have no ports. The hub defines its own graph IR v2 (S4-10), so aminx is not expected to adopt `xtrax.composition`; only the schema_version gate discipline carries over. |
| `loop`, `findings`, `tombstone`, `jaxlint_runner`, `safety` | none | N/A | autoresearch (#2181) and audit tooling. |

## 4. Native length bucketing for the core runner

**Today the runner does not bucket.**
- The proxide loader pads every batch to `spec.max_length`, 512 by default (`host/prep.py:117`).
- The runner loops over those batches (`runner.py:314` sample, `:843` score), and the planner never sees the residue axis.
- `_PaddingCheck` (`runner.py:84`) only warns (#2358).

**The xtrax-native chain:**
1. Give `N_RESIDUES` `bucket_boundaries=BUCKET_LADDER` from xtrax, and pass it to `_plan_with_joint_budget` / `make_sampling_planner` (`host/plan.py:74,216`). The planner then emits `Bucket(boundaries)`.
2. Per batch, on the host before the jit boundary, take the span and call `select_bucket`, then trim the loader-padded batch to the rung. Trimming is safe because padding past the span is masked. Alternatively, have the loader pad to the rung.
3. Keep `Bucket` out of `_dispatch_axis`, which has no `Bucket` branch and would fall through to the `safe_map` fallback. xtrax treats `Bucket` as host-tier and never executes it.
4. Retire the duplicates: aminx `select_bucket` and `group_by_bucket`, `get_length_bucket`, and the `jnp` pad paths. Rewrite or delete `plan_bucketed`.
5. `max_length` becomes a crop limit rather than the shape. The agent layer's `fitted_max_length` (PR #195) then becomes redundant.

**Consequences.**
- Seeded sampling changes, because draws happen at the padded shape (`runner.py:72`).
- Scores do not change; `tests/host/test_score_padding_invariance.py` checks that.

**Open decision (user):** whether bucketing is on by default, with an opt-out that reproduces old seeds.

## 5. Priority order (proposed)

1. **Zero-risk hygiene.**
   - Fix the two stale imports (`SafeMap`, `DedupSpec`).
   - Update the stale `RINGS = "not_available"` comment and assumption. Note that `261001_xtrax-model-contract.md` still cites `xtrax[io,export]==0.4.0a10`, while the pin is now a11.
   - Delete the dead code: `tiling/buckets.py`, `tiling/eda.py`, `StageBundleAdapter`, `DedupGather`, and `plan_bucketed` once its test is dropped.
2. **Runner bucketing** (section 4). This closes #2358 and the ladder sprawl, and is blocked on the default-on decision.
3. **Upstream the #2391 size-1 guard** into xtrax `chunked_map`. Then replace `safe_map`, `safe_scan` and `_dispatch_axis` (#2371). Retire T2.GATE and delete the aminx iterator/dispatch baseline.
4. **Adopt the gaps that carry value:**
   - `export.onnx.convert_to_onnx` for the nine direct jax2onnx callers (already spec'd as AC6 in `261001_xtrax-model-contract.md`);
   - `export.rings` / `divergence` in layer B/C parity;
   - `profiling.loop_scaling` as a CI complexity guard on AR decode;
   - `device_memory_budget`, replacing the silent 4 GiB fallback.
5. **Decide, then act:**
   - wire `axis_boundaries` through the xtrax stage executor, or drop the field;
   - telemetry ownership (cisternal vs xtrax);
   - `no_bias_wd_mask`, which changes numerics;
   - `Trainer`/`Engine` for the main trainer.
6. **Upstream-first (xtrax):** `xtrax.run.units` (#2421), `has_aux` on `accumulate_grads`, and ports on `HostPrepGraph` (S4-02).

## 6. Unconfirmed

- The xtrax `Trainer` and `repro_floor` bodies were not read.
- Op-set equivalence between `onnx_audit.find_onnx_rng_ops` and xtrax's version was not checked.
- Two claims come from the streaming scope spec and were not re-read: that `streaming.py` never calls `finalize()`, and that `ZarrStagingSink` refuses a foreign `run_id`.
- Whether `validate_plan_topology` rejects a `Bucket` decision outside export mode.
