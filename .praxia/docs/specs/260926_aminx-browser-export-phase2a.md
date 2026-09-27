---
title: "aminx browser validation Phase 2a — RNG-free P03/P04 export (ORT + IREE), browser parity, profiling, benchmark; sampling-budget re-estimate track"
description: "Sprint spec: RNG-free export wrappers for P03/P04, jax2onnx→ORT-CPU and IREE-native layer-(b) parity, ORT Web layer-(c) parity in headless Chromium, ORT/JAX profiling incl. a loop-body scaling report, a Playwright benchmark with an interleaved native arm, and an independent track that fixes the batched sampling harness and re-estimates the Phase-1 sampling budget on titanix GPU 2."
status: draft-r1
task_id: 260926_browser-export-loop
date: '260926'
parent: 260923_aminx-browser-validation.md
---

# Specification: aminx browser validation Phase 2a (P03/P04 export, layers b and c)

## Revision log (adversarial review r1)

The review verdict was REVISE. Every objection below is applied in the body. Everything the
defender rebutted is unchanged. New anchors were read at the current worktree state and are listed
as V17–V27.

| Obj. | Change |
|---|---|
| B1 | Real dependency graph (see "Fixer Tasks"). Local runs are serialised by `flock /tmp/bv-local.lock`, held by both `local_run.sh` and `check_run.sh`, so `bth compact` never overlaps a run. The manifest has one writer (`layer_b_common.manifest_append`, called under the lock), and a `T<n>: record` commit re-copies it after every run that adds rows (T2, T3a, T4). T3 is split into T3a (calibrate) and T3b (validate), with an explicit pre-registration commit between them. `$CB`/`$CC`/`$CP` are defined once. `BV_ARTIFACTS` is always the base directory; the `<H12>` subdir is recorded in the manifest (D-G) |
| B2 | `local_run.sh` step 7 is `env $CAPS bth run …` |
| B3 | Dropout path analysed (V17). "Native" = the `aminx.inference` kernel with `config.inference` asserted `True` **and** every `Dropout.p` zeroed/asserted 0. The analysis also found `config.inference` does not reach the encoder or the fallback decoder, so it does not disable dropout on its own either. Key-independence test added. Every "inference_mode suffices" claim removed. Footgun recorded as F-D1 plus proposed debt; not fixed |
| B4 | ORT Web thread count is fixed at wasm init: t1 and tN run in separate browser contexts, each asserting its wasm-init thread count. "Session" is defined. ORT 1.30 behaviour marked to-verify with a test (T5a) |
| B5 | Local runs sync `--extra=dev --extra=benchmark` (torch/prody, V18). The `bth run` command carries `--extra=benchmark` as one token |
| M1 | Boundary acceptance at L = 48 and L = 1024 is added next to the refusals at 47 and 1025 |
| M2 | ONNX walker recurses into `model_proto.functions` and resolves function-call nodes. Planted `RandomUniform` inside a function body. `rng_bit_generator`, `rng_uniform` added to the jaxpr set. `com.microsoft` is labelled separately |
| M3 | T2 gate allows `pass,not_converted`, with a per-path BLOCKED table |
| M4 | `$BATHOSW` carries `--prerelease=allow`. bth provenance comes from grepping and hashing the installed code. Every sidecar is dry-run through `evaluate_outcome(parse_sidecar(p), synthetic)` before commit (`$BV/sidecar_dryrun.py`) |
| M5 | New "Result-emission rule" in Common context. `emit()` sanitises NaN/Inf rather than rejecting them (V26), so gating relies on the `*_available` flags |
| M6 | Sized `w_e_proj.bias` edge control for layers b and c. The identical-array self-test is relabelled as a sanity check |
| M7 | T5 split into T5a (set-A calibrate of the c bars and c-sized δ) and T5b (validate) |
| M8 | Near-tie rule rewritten (adjacent pairs among the first k+1 sorted distances, id-aligned edge comparison, ε justified, and the same rule at layer c) |
| M9 | Headroom rows keyed by (path, quantity, route, bucket). A calibrate value above the bar → `not_advanced`, pre-registered and never gated |
| M10 | Real-data W rows on set B in T3b. "Native unpadded" = the kernel (D-H) |
| M11 | T9 host check mirrors `consistent`, `identity_frame` and `fits_slab` (V19). `wave(j)` = `decode_wave`. Fixed-position lanes are tested |
| M12 | T9 gate rewritten (no unset `$H`, `check_run.sh`, no `LIMIT`, sidecar sha, explicit node ids) |
| M13 | Thread-fairness rule D-I: the native arm is pinned by XLA flags (flag marked to-verify) and ratios across unequal budgets are suppressed. Profiling ORT-CPU runs at 1 thread for the cross-route table. Thread settings are recorded per arm |
| M14 | T7 metric renamed `body_operand_elements`. Fusions counted once at the call site. O(L)-reduce-to-scalar toy control added |
| M15 | AC-19 k = 32 (synthetic model; deviation stated), AC-23 peak memory, and AC-24 regex gate added |
| M16 | T8 END_TO_END records carry `metrics={"total_step_seconds": steady p50}` (V20) |
| minor | Parquet fallback is a gate path. Mutation (d) moved into `select_neighbors`. T3 refusal crop taken from set A. Lognormal σ fixed and every CI "within-session". T9 imports the caps and reports 2·c_ref. T10 checks `git diff --quiet`. AC-P1 gets a malformed stage-0 control plus a statement of where the counters are measured |
| found in r1 | The r0 `check_run.sh` query combined `LIMIT 1` with `count == 1`, so the check could never see duplicate rows (count ≤ 1 by construction). Fixed with M12 |

## Overview

Export the two paths that advanced from Phase 1 into the ORT and IREE layer-(b) routes: P03
(featurisation + k-NN) and P04 (unconditional logits). The exported graphs contain no RNG
primitives. Run the same ORT artifacts in ORT Web (layer c), then profile and benchmark them against
an interleaved native-JAX arm. A separate track re-estimates the Phase-1 sampling budget with the
incremental AR decoder, after fixing a batched-harness defect found while writing this spec.

**Scope.** In: P03, P04. Supporting: P01 (checkpoint sha256 carried into every manifest row, plus a
k-neighbours topology assertion) and P27 (bucketing, padding and refusal). The P07 AR kernel appears
only as an explicitly labelled **diagnostic** in T7. P05+ remain out (advance table:
`outputs/browser_validation/layer_a/advance_table.json`; only P00/P03/P04 advance, and P00 is
layer-(a)-only).

## Verified anchors and corrections to the brief

Every anchor below was read at the current worktree state. Where the brief disagreed, this section
is authoritative.

| # | Fact | Evidence |
|---|---|---|
| V1 | `apply_noise_to_coordinates` is called at `src/aminx/model/features.py:200-204`, in the non-precomputed branch (`:191-206`). Brief said "~:137-200" | read |
| V2 | The noise function splits the key **outside** the `lax.cond` (`src/aminx/utils/coordinates.py:41`); only `normal` is inside it (`:44`, cond `:51-56`). So `random_split` is emitted even at noise 0 | read |
| V3 | Passing `rbf_features` **and** `neighbor_indices` takes the precomputed branch `features.py:178-189`, which never reaches `:200`. The k-NN selection itself is `:208-229` (k clamp `:227`, `top_k` `:46-92`) | read |
| V4 | Other RNG sites on the P04 path: kernel `split` (`src/aminx/inference/score_unconditional.py:26`); `fold_in(k_enc, 0)` (`src/aminx/inference/encode.py:89`, `:193`); `PRNGKey(0)` default (`src/aminx/model/mpnn.py:147-148`). The encoder and decoder skip their `split` when `key is None` (`src/aminx/model/encoder.py:369-371`, `src/aminx/model/decoder.py:634-636`) | read |
| V5 | The xtrax gate is NOT an RNG-freedom check. The P04 kernel was "clean" in the Phase-0 census (`outputs/browser_validation/phase0/export_safety_census.json`) because the threefry rule fires only on input-derived keys (`.venv/.../xtrax/export/safety.py:474-478`), and trace failures return `[]` (`:443-451`) | read |
| V6 | Phase-0 pins: jax 0.10.2, jax2onnx 0.16.1, onnxruntime 1.30.0, onnx 1.23.0. The multi-key-sort `top_k` converted and was index-identical at L=128 | `outputs/browser_validation/phase0/jax2onnx_spike.json` |
| V7 | aminx has TWO ladders: `tiling/bucketing.py:32` `(64,128,256,512)` and `tiling/buckets.py:19` `LENGTH_BUCKETS=(100,200,400,800,1200)`. The spike padded with `buckets.pad_to_bucket` (`scripts/browser_validation/jax2onnx_spike.py:117-119`) | read |
| V8 | The "xtrax 64..2048" ladder is `BUCKET_LADDER=(64,128,256,512,1024,1536,2048)` in `rings.py:98` on UNMERGED xtrax worktrees (`/home/marielle/projects/xtrax/.claude/worktrees/ring-probes/src/xtrax/export/rings.py`). The installed `xtrax==0.4.0a10` (PyPI, `uv.lock:3977-3979`) has no default ladder and no `rings` module (only `xtrax.tiling.select_bucket`/`bucketize`). **R0–R3 are unavailable this sprint** | read |
| V9 | Set-B lengths: 6MRR 68, 1BC8 113, 3HTN 416, 4YOW 693. Set A: 5L33 106, tie_lattice_L96 96, 4GYT 354, 6EHB 955, 2GFB 3464. A 512 max bucket would refuse 4YOW | `outputs/browser_validation/fixtures/manifest.json` |
| V10 | `proteinmpnn_v_48_020` → `k_neighbors = 48` (`src/aminx/io/weights.py:65-84`). P01 does **not** advance (confirmed defects incl. `checkpoint_topology_source`), so wrappers assert `model.features.k_neighbors == get_topology_for_checkpoint(ckpt)["k_neighbors"]` instead of trusting either one | advance_table.json |
| V11 | IREE is installed in `.venv` (`iree/compiler/__init__.py`, `iree/runtime/__init__.py` present; locked at `uv.lock:1356,1372`). xtrax exports `check_export_safety`, `compile_for_target`, `run_native_vmfb`, `verify_native_parity`, `NATIVE`, `WASM32`. I checked file presence but did **not** run the import (no shell in this session); T4 step 1 runs it | read |
| V12 | Playwright browsers installed: only `chromium-{1208,1223,1228,1234}` + `chromium_headless_shell-*` in `~/.cache/ms-playwright`. **No Firefox or WebKit.** `browser/smoke/node_modules` has onnxruntime-web 1.30.0 (incl. `ort-wasm-simd-threaded.jsep.wasm`) and @playwright/test 1.63.0 | glob |
| V13 | `xtrax.profiling`: `ProbeRecord`, `ClaimClass{STRUCTURAL,DISPATCH_COUNT,TERM_RANKING,END_TO_END}`, `assert_claim_supported`, `paired_configs`; `trace.parse_hlo_op_times`/`parse_dispatch_counts`/`scope_map_from_hlo_text`. TERM_RANKING needs stage ≥ 2 = GPU (`claims.py:250-257`, `record.py:218-223`), so **CPU/wasm per-op data can never back a TERM_RANKING claim**. `git_sha` auto-capture resolves from `parents[3]` of the installed package (`record.py:36`), so scripts set `XTRAX_GIT_SHA` explicitly (`record.py:55-57`). No loop-body cost primitive exists (the gap is #1983) | read |
| V14 | `layer_a_sampling_calibrate.py` has **no** floor-only / `--prepare-only` mode. The floor projection runs automatically as the first step of `run_full` (`:641-687`); CLI `:979-1000`. `--prepare-only` belongs to `titanix_launch.sh` (`:53`, `:67-68`, `:80-82`) and only materialises the checkout | read |
| V15 | **Suspected harness defect (F-S1).** `aminx_sample_batch` builds ONE bundle from `waves[0]` (`scripts/browser_validation/layer_a_sampling.py:617-642`), whose `ar_mask` is `generate_wave_ar_mask(waves[0])` (`src/aminx/inference/bundle_builder.py:284`). The vmapped `one` replaces only `.wave` (`layer_a_sampling.py:565`), and the decoder reads `cond.ar_mask` (`src/aminx/inference/decode/autoregressive.py:485`, `:631`). So draws i ≥ 1 of every chunk decode with draw 0's visibility. That matches the docstring's "0/85, 15/85, 10/85" mismatch pattern (`layer_a_sampling.py:600-601`), where draw 0 was exact | read |
| V16 | The 85.8× speedup (`scripts/benchmarks/bench_ar_incremental.py:64`) was measured **unbatched**. Under the harness's `vmap` over per-draw waves, the `can_increment` predicate (`autoregressive.py:751-759`) is batched, so `lax.cond` lowers to `select` and **both** branches run. `incremental` is not plumbed through `sample_autoregressive.kernel` (`:113-118`; field at `mode.py:79-80`). So re-running the calibrate as-is would likely show no speedup | read |
| V17 | **Dropout path (r1, B3).** `Aminx.__call__` takes `inference: bool = True` (`src/aminx/model/mpnn.py:120`) but documents it as "accepted but ignored" (`:141`). It injects `PRNGKey(0)` when no key is given (`:147-148`) and calls `self.encoder(…, key=prng_key)` without `inference` (`:165-171`). The encoder's `inference` defaults to `False` (`encoder.py:332`) and is forced `True` only when `key is None` (`:369-370`). `Dropout.__call__` uses `self.inference` only when the caller passes `inference=None` (`dropout.py:45-46`), and it returns `x` unchanged only when `key is None` (`:52-54`). So `eqx.nn.inference_mode` has **no effect** on this path. The kernels pass `inference=config.inference` (`src/aminx/inference/encode.py:96`, `:200`), but only into `Aminx.__call__`, which drops it. The unconditional fallback decoder likewise calls `self.model.decoder(…, key=key)` without `inference` (`src/aminx/inference/decode/unconditional.py:100-112`, deliberately, per its comment). Net effect: on the kernel path with a key, dropout is active whenever any `Dropout.p > 0`. The `key=None` wrapper path is dropout-free by construction | read |
| V18 | `layer_a_exact.parse_canonical_fixture` loads the reference `data_utils.py` by path (`scripts/browser_validation/layer_a_exact.py:170-176`, `:344`). That module imports torch and prody, which exist only in the `benchmark` extra (`pyproject.toml:41-44`). The titanix scripts already sync `--extra dev --extra benchmark` (`titanix_launch.sh:195`, `titanix_run.sh:76`) | read |
| V19 | `can_increment = consistent & identity_frame & fits_slab` (`autoregressive.py:751-756`). `consistent` only covers `reads` = valid row ∧ valid neighbour ∧ `ar_mask>0.5` over the **k-NN slots** (`:630-636`, `:751`). `identity_frame` checks `state_position_map == arange(L)` (`:752-754`). `fits_slab` requires every wave size to be ≤ `slab` (`:755`), where `slab = min(max(max_groups_per_wave·group_positions.shape[2], max_positions_per_wave), L)` (`:609-612`). `decode_wave` = `pos_rank // max_groups_per_wave`, and unscheduled (fixed) positions get `n_waves` (`:616-621`) | read |
| V20 | `REQUIRED_METRICS`: DISPATCH_COUNT needs `{n_executions, n_compilations, n_jit_traces}`; END_TO_END needs `{total_step_seconds}` (`.venv/…/xtrax/profiling/claims.py:89-94`). `ProbeRecord.__post_init__` rejects stage ∉ {0..3}, `n_atoms ≤ 0` and boolean metrics (`record.py:213-233`) | read |
| V21 | `BUDGET_WALL_HOURS_CAP = 16.0` and `PROJECTED_PEAK_RSS_GIB_CAP = 48.0` are module constants in `scripts/browser_validation/layer_a_sampling_calibrate.py:122-123` | read |
| V22 | bathos (installed `bth` tool) has `parse_sidecar` (`bathos/sidecar.py:296`) and `evaluate_outcome(sidecar, result) -> str` (`:679`). The pinned `$BATHOSW` rev is assumed to match; T2 confirms it by grep. On titanix `--prerelease allow` must be **two** tokens (`titanix_run.sh:93-97`) | read |
| V23 | `model.features.w_e_proj` is an `eqx.nn.Linear` with bias, and it is the final edge projection (`features.py:119`, `:151`, `:274`). `model.w_out` is the logits Linear (`mpnn.py:104`) | read |
| V24 | No v1 original/soluble checkpoint with k = 32 exists: `LEGACY_ALIAS_MAP` holds only `v_48_*` (`src/aminx/io/weights.py:53-62`). k = 32 appears only for `v_32` ids (`:77-79`), e.g. `LIGAND_DEFAULT_CHECKPOINT = "ligandmpnn_v_32_020_25"` (`:48`), which is out of scope | read |
| V25 | `select_neighbors` source region: the mask → `jnp.inf` step (`features.py:209-215`), the structure-mapping mask with `.squeeze()` (`:217-225`), the `k = min(self.k_neighbors, L)` clamp (`:227`) and `top_k` (`:228`) | read |
| V26 | `layer_a_common.emit()` (`scripts/browser_validation/layer_a_common.py:519`) sanitises via `_finite_json` (`:498-513`, applied at `:527`): NaN → null and ±Inf → ±`sys.float_info.max`. It dumps with `allow_nan=False` and writes `$BTH_RESULTS_PATH` always, and `--out` unless a differential phase is active | read |
| V27 | Existing test files for the AR decoder: `tests/inference/decode/test_incremental_autoregressive.py` and `tests/inference/test_fixed_position_self_visibility.py` | glob |

## Design decisions

- **D-A (wrapper home).** `src/aminx/export/` is a new package. Wrappers compose existing public
  functions and do not reimplement numerics. The one source change is extracting `features.py:208-229`
  into a public `select_neighbors(...)` that `forward_edge_stages` calls at the same place
  (behaviour-preserving, pinned by a test).
- **D-B (RNG bypass).** Wrappers compute `compute_backbone_coordinates` → `compute_backbone_distance`
  → `select_neighbors` → `compute_radial_basis`. They then call
  `model.features.forward_edge_stages(None, …, rbf_features=rbf, neighbor_indices=idx)`, which takes
  the precomputed branch (V3), and `model.encoder(…, key=None)`. P04 additionally calls
  `make_decode_fn(model, mode=UnconditionalMode(), strategy=Vmap())(None, enc, bundle, config,
  stage_set)` with `bundle` from `build_inference_bundle(mode="score_unconditional", …)`. No key is
  created or consumed anywhere. With `key=None` the encoder forces `inference = True`
  (`encoder.py:369-370`), and every `Dropout` returns its input unchanged (`dropout.py:52-54`), so
  the wrapper is dropout-free by construction (V17). `eqx.nn.inference_mode` is **not** relied on
  for this property anywhere in this spec.
- **D-C (export ladder).** `EXPORT_BUCKETS = (128, 256, 512, 1024)` lives in
  `aminx.export.buckets`. The library `BucketingConfig`/`LENGTH_BUCKETS` stay unchanged (ODQ-B1). The
  function refuses `L_real < k_neighbors(ckpt)` (`LengthBelowNeighborsError`) and `L_real > 1024`
  (`LengthAboveMaxBucketError`), both subclasses of `ExportLengthError(TilingError)`. Padding
  convention (from `jax2onnx_spike.py:117-119`): coords 0, mask 0, `residue_index`/`chain_index` =
  last real value. Selection uses `xtrax.tiling.select_bucket`.
- **D-D (IREE).** IN for layer (b): native executed parity plus WASM32 compiled as `CODEGEN_ONLY`
  with its sha256. OUT for layer (c): there is no emsdk IREE runtime (parent ODQ-1 fallback).
- **D-E (browsers).** Headless Chromium only (V12). WebGPU gets a capability probe only
  (`navigator.gpu` / `requestAdapter()` result recorded, no numbers). No compatibility text for any
  other browser or EP.
- **D-F (sampling harness).** Vmap over per-draw `(key, wave, ar_mask)`. Plumb
  `incremental: Literal["auto","off","force"] = "auto"` through `sample_autoregressive.kernel`; the
  default preserves behaviour. The harness uses `"force"` for a draw only when a host-side mirror
  of all three device predicates holds for it (`consistent ∧ identity_frame ∧ fits_slab`, V19;
  exact definition in T9). A draw that fails the mirror runs with `"off"` and is counted in
  `n_draws_forced_off` (ODQ-B9).
- **D-G (artifact store).** `BV_ARTIFACTS=/home/marielle/bv-artifacts/260926_layer_b` is **always
  the base directory**, in every gate and every `--artifacts-dir` argument. `layer_b_build` writes
  its artifacts into `$BV_ARTIFACTS/<H12 of its own run>/` and records that subdir as
  `artifact_subdir` in the manifest. Every artifact path in the manifest is relative to
  `$BV_ARTIFACTS/<artifact_subdir>/`. Later producers (T3a perturbed rebuild, T4 vmfb) write into
  the same `artifact_subdir`, which they read from the tracked manifest. The sha256 manifest is
  tracked at `outputs/browser_validation/layer_b/artifact_manifest.json` (`git add -f`), and
  consumers verify against that tracked copy at their run's `git_hash`. Every consumer exits 3 on
  any sha mismatch or missing row.
  - **One writer.** Only `layer_b_common.manifest_append(rows)` writes
    `$BV_ARTIFACTS/<artifact_subdir>/artifact_manifest.json`. It is append-only (an existing row's
    sha is never rewritten; a conflicting duplicate key raises), writes atomically (tmp + `rename`),
    and is only ever reached inside a `local_run.sh` run, which holds `/tmp/bv-local.lock`. T2
    creates the file through the same function.
  - **Re-copy.** After every run that appended rows (T2, T3a, T4), a `T<n>: record` commit copies
    the base manifest over the tracked one (`cp` + `git add -f`) before any later task runs.
- **D-H (definition of "native").** "Native" (the W and P27 comparator, and "native unpadded")
  means `aminx.inference.score_unconditional.kernel` (P04), or `model.features(…, backbone_noise=0.0)`
  for P03, called on `native_model = zero_dropout(model)`. Two conditions are asserted before every
  native call: `config.inference is True`, and every `Dropout.p == 0` in `native_model`.
  `zero_dropout` (in `aminx.export.wrappers`) collects every `aminx.model.dropout.Dropout` via
  `jax.tree_util.tree_leaves(model, is_leaf=…)`, sets `p = 0.0` with `eqx.tree_at`, and returns
  `(model, {"n_dropout": n, "max_p_before": float})`. The wrapper uses the **unmodified** model:
  its constructor asserts that it never passes a key (checked structurally by the jaxpr RNG audit)
  and records `max_p_before` as well. Because config.inference does not disable dropout (V17), the
  zeroing is what makes native deterministic, and the key-independence test (T1 g) proves it. For
  real fixtures the native P04 arm stays `layer_a_exact._score_aminx_unconditional`, called on
  `native_model`.
- **D-I (thread fairness).** A timing ratio sentence between two arms is emitted only if both arms
  record the same thread budget. `OMP_NUM_THREADS` does **not** cap XLA:CPU, so the native JAX arm
  is pinned with `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.
  That exact flag spelling is **to be verified** against jax 0.10.2 in T8 step 0: a 2048² matmul
  must show process CPU-time / wall-time ≤ 1.2 with the flag, and > 1.5 without it. If the check
  fails, the native arm is recorded as `threads = "unpinned"` and no native ratio is printed. The
  native arm is only ever paired with `ort_wasm_t1`. `ort_wasm_tN` gets no native ratio. The
  profiling ORT-CPU pass used for cross-route per-op tables runs at `intra_op_num_threads = 1,
  inter_op_num_threads = 1`. Every arm records `{threads_requested, threads_observed, source}`.

## Pre-registered bars

Bars are capped at the layer-(a) bars (parent "Pre-registered layer-(a) bars") and are never
loosened. "Real rows" means the rows with `mask = 1`.

**Headroom rule (r1, M9).** Headroom rows are keyed by `(path, quantity, route, bucket)`. Each key
takes the max calibrate measurement over the set-A fixtures at that bucket. Buckets with no set-A
fixture (for example 256) inherit the nearest larger measured bucket, and the params file marks
them `inherited_from`. Three states:
- measurement ≤ bar/2: `advanced`;
- bar/2 < measurement ≤ bar: `low_headroom`. The row is gated in validate, but the outcome can be
  at best `partial_headroom`;
- measurement > bar: `not_advanced`. It is written to the params file before validate, the row is
  **never gated later** (validate reports it as `excluded_not_advanced`), and the validate outcome
  can be at best `partial_headroom`.

The calibrate outcome itself is not failed by `not_advanced` rows.

**Near-tie rule (r1, M8), used identically at layers b and c.** For real row i, sort that row's
CA–CA distances to real residues in ascending order (as computed in float64 on the host from the
same float32 inputs every backend receives): d(1) ≤ … ≤ d(k+1). Row i is a *near-tie row* if any
adjacent pair among those first k+1 values has gap `d(m+1) − d(m) ≤ ε`, with ε = 1e-4 Å. A
neighbour-index mismatch on a near-tie row is `tie_unstable`. A mismatch on any other row is
`fail`. Edge floats are compared **after aligning by neighbour id**: for each row, only the
neighbour ids present in both index sets are compared, and ids in the symmetric difference are
counted in `n_tie_swapped_slots`. On non-near-tie rows the symmetric difference must be empty.

*Why ε = 1e-4 Å.* Fixture coordinates carry 3 decimals (a 1e-3 Å grid). Every backend receives the
same float32 inputs, so two backends' distances can differ only by float32 evaluation order.
That error is a few ulp(d), ≈ 1e-5 Å at d ≤ 64 Å (ulp(64) = 7.6e-6). A gap ≤ 1e-4 Å is
therefore ≥ 10× below the input grid, and it is not a geometric distinction the data resolves.
ε is also ≥ 5× the expected cross-backend distance error. That second property is *measured*:
T3a records `max_abs_dist_err` = max |d_f32 − d_host64| across all set-A real pairs, where d_f32 is
JAX float32 `compute_backbone_distance`. This is a proxy, because the exported graphs do not
output distances. If ε < 5 ×
`max_abs_dist_err`, T3a outcome is `ctrl_unsized` (ε under-sized). ε is never widened after
validate starts.

| Comparison | Quantity | Tier | Bar |
|---|---|---|---|
| W: wrapper vs native (D-H), same JAX; synthetic (T1) and set B (T3b) | P03/P04 neighbour indices | EXACT | 0 mismatches |
| W | P03 edge features; P04 logits | TOL | max-abs ≤ 1e-6 (bitwise flag recorded) |
| P27: wrapper at bucket b vs native unpadded (D-H: the kernel on the unpadded structure) | P04 log-probs (real rows) / indices | TOL / EXACT | ≤ 1e-4 nats / 0 |
| b-ORT, b-IREE: artifact vs native wrapper | neighbour indices (all real rows) | EXACT | 0; if every mismatch is on a near-tie row (near-tie rule above) → `tie_unstable` |
| b-ORT, b-IREE | P03 edge features | TOL | max-abs ≤ 2e-5 |
| b-ORT, b-IREE | P04 log-probs (`log_softmax`, float64 host) | TOL | max-abs ≤ 1e-4 nats; Pearson reported, not gated |
| b-ORT, b-IREE | P04 argmax | EXACT | 0 mismatches where top-2 margin > 2e-4 |
| c: ORT Web vs ORT-CPU, same artifact | indices / argmax | EXACT | 0 |
| c | P04 log-probs / P03 edge features | TOL | ≤ 1e-5 / ≤ 2e-6 (bar/10, ODQ-B8); the transitive gap vs native is reported against the layer-(b) bars |

Sized controls (calibrate). Each control's δ is committed in the params file before its validate.
- **P04 log-prob, layer b (T3a):** `w_out.bias[0] += δ_b04`, with δ_b04 chosen so the P04 log-prob
  max-abs lands in [2×, 10×] of 1e-4. Phase 0 used 5e-4.
- **P03 edge features, layer b (T3a, r1 M6):** `features.w_e_proj.bias += δ_b03` (all channels; V23),
  with δ_b03 chosen so the P03 edge max-abs lands in [2×, 10×] of 2e-5.
- **Layer c (T5a, r1 M7):** δ_c04 on `w_out.bias[0]` sized to [2×, 10×] of 1e-5, and δ_c03 on
  `w_e_proj.bias` sized to [2×, 10×] of 2e-6. These are sized against the c bars, not taken from
  T3a's `ctrl_effect`.

Each perturbed artifact must be *detected* (max-abs > its bar) in validate. The identical-array
comparator self-test (identical inputs → 0, must pass) is kept, but it is a **sanity check, not a
control**, and no sidecar counts it in `controls_total`.

## Acceptance Criteria

- **AC-B1 (wrappers).** `tests/export/test_export_wrappers.py` passes with zero skips. It covers the
  W-equivalence and P27 bars on a synthetic structure at every export bucket. For k = 48 it covers
  the typed refusals at L = 47 and L = 1025 **and** acceptance at the boundaries L = 48 and
  L = 1024. For k = 32 (parent AC-19; synthetic random-init model, see deviation X1) it covers
  refusal at L = 31 and L = 1025, acceptance at L = 32 and L = 1024, and P27 padding at every
  bucket. It also covers: the `select_neighbors` refactor being bitwise-identical to the
  pre-refactor function on a tie-rich lattice, a random structure, a padded-mask case and a
  two-structure `structure_mapping` case; the topology assertion firing on a model whose
  `k_neighbors` is patched; and key-independence of native (D-H), meaning the native P04 kernel
  with `PRNGKey(0)` vs `PRNGKey(1)` at noise 0 is bitwise equal. Each test is shown RED under its
  named mutation.
- **AC-B2 (RNG-free, parent AC-17 for P03/P04).** `find_rng_primitives` returns `[]` for both
  wrappers' jaxprs at every bucket. The ONNX walker returns `[]` for every clean artifact. Five
  planted controls are all flagged: the unmodified `score_unconditional.kernel` jaxpr; a wrapper
  variant with a planted `normal`; RNG inside `lax.cond` inside `jit`; an ONNX copy with a
  `RandomUniform` injected into an `If` subgraph; and an ONNX copy with a `RandomUniform` inside a
  local function body (`model_proto.functions`) that the main graph calls. Recorded in the
  `layer_b_build` run.
- **AC-B3 (ORT layer b, parent AC-18/19/20).** `layer_b_ort_validate` has outcome `pass`, or
  `partial_headroom`/`tie_unstable` with the rows named. It covers every set-B (fixture, bucket ≥
  own) pair for P03/P04, the real-data W rows (wrapper vs native, D-H) and the refusal rows. All
  three planted controls are detected: perturbed P04 weight, perturbed P03 `w_e_proj.bias`, and
  the planted index swap. An op-set/EP report is attached (opset imports, op-type histogram,
  `get_providers()`, per-node provider from one profiled run). Every artifact has a manifest row
  (sha256, P-ID, bucket, route, checkpoint id and sha256, versions, I/O signature).
- **AC-B4 (IREE layer b).** `layer_b_iree` has outcome `pass`, or a pre-registered residual with its
  reason. For both wrappers, `check_export_safety(..., NATIVE)` and `(..., WASM32)` return `[]` with
  an independent `trace_ok`, and both census controls are re-detected. Native vmfb parity holds
  within the b bars, with planted controls detected. WASM32 vmfb sha256 recorded as `CODEGEN_ONLY`.
  `rings = "not_available"` (V8).
- **AC-C1 (browser layer c, parent AC-21).** `layer_c_calibrate` (set A) has outcome `pass`, with
  the c headroom and the c-sized δ committed before validate. `layer_c_parity` (set B) has outcome
  `pass`, `partial_headroom` or `tie_unstable`, with the rows named. The run writes an evidence
  JSON `outputs/browser_validation/layer_c/evidence/ort-wasm__chromium.json` with UA, browser
  version, ORT Web version, EP, the per-context wasm-init thread assertion, `crossOriginIsolated
  === true`, and the WebGPU probe. Integers are EXACT and floats are within the c bars. The
  c-sized in-browser planted controls are detected. A no-COOP/COEP negative control yields
  `crossOriginIsolated === false`.
- **AC-P1 (profiling).** `layer_b_profile` has outcome `pass` or `partial` (web per-op unavailable).
  It includes per-op-type ORT-CPU (and, if emitted, ORT Web) time shares per artifact, and JAX-native
  `cost_analysis` + HLO op histogram + measured per-HLO-op times, with a DISPATCH_COUNT claim
  asserted via `xtrax.profiling`. The DISPATCH_COUNT counters are measured over the 30-iteration
  steady-state loop, after warm-up, in the profile process (T6 step 2). The synthetic-profile
  controls pass. A malformed stage-0 record (missing `n_jit_traces`) is **refused** by
  `assert_claim_supported(…, DISPATCH_COUNT)`, and a boolean metric is refused at construction.
- **AC-P2 (loop-body structural report).** `layer_b_loop_cost` has outcome `pass` or
  `loop_body_scales`. The metric is `body_operand_elements` (T7). The toy instrument controls pass:
  linear matmul body α ∈ [0.9, 1.1], O(L)-reduce-to-scalar body α ∈ [0.9, 1.1], constant body
  α ∈ [-0.1, 0.1]. AR `off` α ≥ 0.8 (a negative control that must show scaling). The AR `force`
  α and the top-10 body instructions per L are reported, and draft xtrax #1983 text is written.
- **AC-BM (benchmark, parent AC-23).** `layer_c_bench` has outcome `pass` with separated phases,
  n_warmup ≥ 5, n_iter ≥ 30, 3 repeats, p50/p90 + stratified-bootstrap within-session 95 % CIs per
  cell, peak memory per cell where available (`peak_memory_available` flag otherwise), and routes
  plus a native arm interleaved in one session (definition in T8). Thread settings are recorded per
  arm, and no ratio is printed across unequal thread budgets (D-I). The planted busy-wait arm is
  detected and the A/A arm is not. Every printed timing or ratio sentence passes
  `assert_claim_supported` / `paired_configs`.
- **AC-CG (compatibility-claim gate, parent AC-24).** `$BV/compat_claim_gate.py` exits non-zero
  on any route/browser compatibility phrase in the sprint's deliverable text that has no AC-21
  evidence file. A planted unsupported claim turns it red, and the real deliverables pass.
- **AC-S1 (sampling track).** F-S1 is confirmed or refuted by a test. If confirmed, the fix is
  landed with RED→GREEN. `layer_a_sampling_budget_floor` on titanix GPU 2 has an evaluated outcome
  with the fast-path control (force/off ≤ 0.5 at the largest set-B fixture) and the A/A timer
  control. If `pass`, T10 launches the full calibrate and its record has an evaluated outcome.
- **AC-G (all runs).** Every finding traces to a `bth run` record with `git_dirty = false` and
  `sidecar_sha256` = sha256 of the sidecar at the run's `git_hash` (non-empty). Exactly one row
  matches. The outcome is one of the task's allowed set, and never `error`. Each sidecar was
  committed before its run, passes `bth validate-sidecar`, `lint_sidecars.py` and
  `sidecar_dryrun.py` (every declared outcome branch reachable from a synthetic result), and every
  result JSON obeys the result-emission rule.
- **Parent-AC deviations (declared, r1 M15).**
  - **X1:** parent AC-19 asks for "each v1 checkpoint … k = 48 and k = 32". No v1 original/soluble
    k = 32 checkpoint exists (V24), so k = 32 is covered on a random-init `Aminx` with
    `k_neighbors = 32` (topology from `get_topology_for_checkpoint("proteinmpnn_v_32_020")`), for
    refusal, boundary acceptance and P27 padding (T1). No k = 32 exported artifact is built.
  - **X2:** parent AC-23 "per route × browser": Chromium only (D-E).
  - Nothing else is deviated.

## Common context for every task

- **Worktree.** `WT=/home/marielle/projects/aminx/.claude/worktrees/browser-validation` (branch
  `autoloop/260926_browser-export-loop`). Run `git status --short` before editing; commit or stash
  (tagged, never bare) anything dirty. One commit per task for code+sidecar (this is the
  pre-registration, made BEFORE the run). An optional follow-up `T<n>: record` commit may hold only
  run-produced tracked files (manifest, params). Never push `origin`; never force-push; never merge
  to main.
- **Shorthands.**
  ```bash
  BV=scripts/browser_validation
  CAPS="OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4"
  ORTW="--with=jax2onnx==0.16.1 --with=onnxruntime==1.30.0 --with=onnx==1.23.0"
  BATHOSW="--with=bathos@git+https://github.com/maraxen/bathos@84be544ecb45734f46e43d22f351a54d6edd6ae5 --prerelease=allow"
  BV_ARTIFACTS=/home/marielle/bv-artifacts/260926_layer_b      # ALWAYS the base dir (D-G)
  CAMPAIGNS=outputs/browser_validation/layer_b/campaigns.json  # written by T2
  CB=$(jq -er '."aminx-bv-layer-b"' "$CAMPAIGNS")
  CC=$(jq -er '."aminx-bv-layer-c"' "$CAMPAIGNS")
  CP=$(jq -er '."aminx-bv-perf"'    "$CAMPAIGNS")
  CA=32a520cf                                                   # aminx-bv-layer-a (existing)
  ```
  Every uv option is ONE token locally (`--extra=dev`, `--extra=benchmark`, `--with=…`,
  `--prerelease=allow`), because the local `bth` argv parser splits otherwise. On titanix,
  `--prerelease allow` stays **two** tokens (V22); `titanix_run.sh` already does this, so it is not
  re-specified here. Always use `uv run --frozen`. Scripts that call
  `layer_a_common.provenance()` need `$BATHOSW` (`--prerelease=allow` is needed so the `--with`
  overlay resolves, `titanix_run.sh:93-95`). `CB`/`CC`/`CP` are set in every gate after T2 (jq `-e`
  makes a missing key fatal).
- **bth provenance (r1 M4).** `bth --version` is not evidence. `local_run.sh` records the installed
  `bth` tool's code identity into `$WT/outputs/browser_validation/<sub>/runs/<stem>-<H12>/bth_provenance.json`:
  the sha256 of `bathos/{sidecar,mcp,cli}.py` under
  `$(ls -d ~/.local/share/uv/tools/bathos/lib/python*/site-packages/bathos)`, plus the result of
  `grep -q 'endswith(".py")' …/mcp.py` and `grep -n 'def evaluate_outcome' …/sidecar.py`. Scripts
  also record the same hashes for the `$BATHOSW` overlay as seen by their own interpreter
  (`importlib.util.find_spec("bathos").origin`).
- **Sidecar dry-run (r1 M4).** `$BV/sidecar_dryrun.py --sidecar S --cases S.cases.json` (T2
  creates it) runs under `uv run --frozen --no-sync $BATHOSW`. It calls
  `bathos.sidecar.evaluate_outcome(bathos.sidecar.parse_sidecar(Path(S)), case["result"])` for each
  synthetic case and asserts the returned label equals `case["expect"]`. Every sidecar ships a
  tracked `<stem>.cases.json` with ≥ 1 case per declared outcome, and every case is a full
  schema-valid result (all `[result_schema]` keys present). This runs **before** the sidecar's
  commit and again in each gate. If the pinned rev's import path differs from V22, T2 adapts the
  import once and documents it in its commit.
- **Result-emission rule (r1 M5), for every script in this spec.**
  1. A script that writes a result JSON **exits 0**, including on BLOCKED, refusal, incomplete,
     `ctrl_blind` and every other residual path. The outcome label carries the verdict, not the
     exit code.
  2. The only non-zero exits are integrity refusals (artifact/weight sha mismatch or missing
     manifest row: exit 3) and argument errors (exit 2). These are *meant* to make bathos record
     `error`, and no gate's allowed set ever contains `error`.
  3. Every key declared in the sidecar's `[result_schema]` is present in every result, including
     early-exit results. Absent data is `null`, never a missing key.
  4. Every float is finite or `null`. Each nullable float `x` has a sibling `x_available: bool`.
     `emit()` already sanitises: NaN → `null`, ±Inf → ±`sys.float_info.max`
     (`layer_a_common.py:498-513`, called at `:527`), so a strict dump never fails. Scripts do not
     rely on that as a verdict, though. They set `x_available = false` themselves before emitting
     any non-finite or undefined quantity, and every sidecar pass-direction condition requires
     `x_available = true` for each gated `x`. A sanitised `null`/max therefore lands in
     `incomplete`/`fail`, never `pass`.
  5. Every result goes through `layer_a_common.emit()` (`layer_a_common.py:519`). No script calls
     `json.dump` for its result.
  6. Guarded statistics: Pearson returns `null` + `pearson_available = false` when either array has
     zero variance. α = log2(a/b) returns `null` when a ≤ 0 or b ≤ 0. Any ratio returns `null` when
     its denominator ≤ 0. These guards live in `$BV/bv_stats_guard.py` (T2) and are unit-tested
     there.
- **Local run lock (r1 B1).** Every local bathos interaction takes one lock: `exec
  9>/tmp/bv-local.lock; flock -w 10800 9 || exit 4`. Both `local_run.sh` (for its whole body) and
  `check_run.sh` (for its whole body, including `bth compact`/`bth sync`) take it. So local runs are
  serialised, and `bth compact` never runs concurrently with a run or another compact. Gates run
  tasks sequentially. Tasks on independent tracks may be *coded* in parallel, but their runs queue
  on the lock.
- **Tests.** Single files or `-k` only, prefixed with `$CAPS`. Never run a whole suite locally.
  Every new test is red-checked. Apply the named mutation, confirm the failure, quote it in the
  commit body, then revert.
- **Ruff.** `src/`: `uv run --frozen --extra=dev ruff check --no-fix <f>` + `ruff format --check
  <f>` + `uv run --frozen --extra=dev ty check <f>`. `tests/` and `scripts/` use the parent's
  force-exclude form: `ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B <f> 2>&1 | tee
  <log>; grep -L "No Python files found" <log>` and `ruff format --check --config "exclude=[]" <f>`.
- **Local bathos runs (T2 builds the helpers).** `$BV/local_run.sh` performs:
  0. take `/tmp/bv-local.lock` (above) and hold it until step 10;
  1. refuse on dirty tracked files or untracked files under `scripts src pyproject.toml uv.lock`
     (same check as `titanix_launch.sh:112-119`);
  2. `H=$(git rev-parse HEAD)`;
  3. `git worktree add --detach /tmp/bv-run-<stem>-<H12> $H`;
  4. `uv sync --frozen --extra=dev --extra=benchmark` there (its own `.venv`, so aminx is imported
     from the detached `src`; `benchmark` supplies torch/prody for `parse_canonical_fixture`, V18),
     plus `npm ci` in `--npm-dir` if given;
  5. assert `git status --porcelain` is fully empty (this is what bathos reports as `git_dirty`);
  6. write `bth_provenance.json` (above);
  7. `env $CAPS bth run --campaign-id C --output-paths outputs/browser_validation/<sub>/<stem>.json --
     uv run --frozen --no-sync --extra=dev --extra=benchmark <with-tokens> python $BV/<stem>.py --out
     outputs/browser_validation/<sub>/<stem>.json <args>`. `env` is required, because a bare
     `$CAPS bth …` expands the first assignment as a command name. `--extra=benchmark` is one
     token;
  8. copy `outputs/browser_validation/<sub>/` back to `$WT/outputs/browser_validation/<sub>/runs/<stem>-<H12>/`;
  9. print `RUN_ID=<id> H=<sha>` as its last line;
  10. remove the worktree, then release the lock.

  One detached worktree per run. Gate commands that run a script *outside* `local_run.sh` (dry-runs,
  pytest) and import `layer_a_exact` also pass `--extra=benchmark`.
- **Run verification.** `$BV/check_run.sh --stem S --hash H --sidecar $BV/S.bth.toml --allowed
  a,b,… [--run-id R] [--sync-remote titanix]`:
  0. take `/tmp/bv-local.lock`. If `--sync-remote` is given, run `bth sync --pull --remote-name
     <r>` under the lock;
  1. `bth compact` (plain form, never `--force-rebuild`);
  2. `bth sql "SELECT id, outcome, git_dirty, sidecar_sha256, exit_code FROM runs WHERE command LIKE
     '%S.py --out%' AND git_hash = '<H>'"`, with **no `LIMIT`**, so duplicate rows are visible;
  3. `jq -e` on: row count `== 1`, `git_dirty == false`, `sidecar_sha256 != ""` and `!= null`,
     `sidecar_sha256 == sha256(git show H:<sidecar>)`, `outcome != "error"`, and `outcome ∈
     allowed`.

  **Parquet fallback (a gate path, not inspection).** If `bth sql` fails on a locked or stale warm
  index, `check_run.sh` reads `~/.bth/catalog/runs/aminx/run_<R>.parquet` with `uv run --frozen
  --with=duckdb python -c …` (R from `--run-id`, else from the `RUN_ID=` line). It applies the
  **same** assertions (exactly one row whose `git_hash = H` and whose command matches, plus all the
  checks in step 3) and prints `source=parquet`. Exit codes are the same. Exit 0 = ok, 1 =
  assertion failed (including count > 1), 4 = lock timeout, 6 = no row (a missing row is a
  failure, never a pass). Before first use, run `bth sql "DESCRIBE runs"` to
  confirm the column names. If bathos hashes the sidecar differently than a plain sha256 of the
  bytes, confirm the formula from bathos source, encode it in `check_run.sh`, and document it in
  the T2 commit; never drop the check.
- **Campaigns (T2 creates them; ids go in `outputs/browser_validation/layer_b/campaigns.json`, `git
  add -f`).** `aminx-bv-layer-b` (confirmation; the layer-(b) claim is registered before the first
  validate run), `aminx-bv-layer-c` (confirmation), `aminx-bv-perf` (exploration, `novel = true`).
  The sampling track uses the existing `aminx-bv-layer-a` campaign (`32a520cf`, verdict doc).
  Check `bth campaign create --help` for the `--mode` values before first use.
- **Sidecars.** Every non-whitelisted outcome label is `is_residual = true` (`lint_sidecars.py:43-53`
  whitelist: pass, partial, partial_headroom, blocked, tie_unstable, not_converted, ctrl_unsized).
  Declaration order is load-bearing, because the first matching branch wins: order is
  `blocked`/`incomplete`/`ctrl_blind` first, then failure labels, then pass-direction labels.
  Every pass-direction condition also requires `<x>_available = true` for each gated float
  (result-emission rule 4). Every sidecar
  names a positive control (must be detected) and a negative control (must not be / must fail).
- **Fixtures.** Reuse `layer_a_exact.parse_canonical_fixture` (`layer_a_exact.py:292-365`) for every
  real structure; never re-parse. Native P04 arm = `layer_a_exact._score_aminx_unconditional`
  (`:740-752`) called on `zero_dropout(model)` (D-H). Checkpoint `proteinmpnn_v_48_020` via
  `load_model`; record its file sha256 and check it against `$BV/reference_pins.json`. Every script
  and fixture records `n_dropout` and `max_p_before` from `zero_dropout`. `eqx.nn.inference_mode`
  may still be applied, but nothing in this spec depends on it (V17). If the loaded checkpoint has
  `max_p_before > 0`, that is finding **F-D1b**: the Phase-1 P04 native arm ran with active dropout
  keyed on `PRNGKey(0)`. It goes into the T3a result JSON and the handoff and is **not** re-graded
  here.
- **Titanix (T9/T10 only).** `$BV/titanix_launch.sh --gpu 2 <stem> <campaign-id> <args>`. It pushes
  by URL to `titanix:/home/solab/bv/aminx.git` (`:132`) and exposes only GPU 2
  (`CUDA_VISIBLE_DEVICES=2`, `titanix_run.sh:74-79`). vLLM on GPUs 0/1/3 must not be touched. One
  `bv-*` unit at a time. Verify via `check_run.sh --sync-remote titanix …` (above), which performs
  `bth sync --pull` + `bth compact` + the identity-form query under the lock.

## Fixer Tasks

**Dependency graph (r1, B1).**

```
Main track:     T1 → T2 → T3a → T3b → { T4 , T5a → T5b } → T6 → T8
Side branch:    T2 → T7            (needs local_run/check_run, $CP; independent of T3..T6)
Sampling track: T9 → T10           (independent of T1..T8; T10 only if T9 outcome = pass)
```

- T2 needs T1 (the wrappers).
- T3a needs T2 (helpers, campaigns, manifest, `$BV_ARTIFACTS`, perturbed/swap artifacts).
- T3b needs T3a (params plus the pre-registration commit).
- T4 needs T2 (helpers, campaign, manifest, `BV_ARTIFACTS`) and T3a/T3b (`ctrl_effect`,
  `min_effect`, and the T3a manifest re-copy).
- T5a needs T3b (the validate pairs and the manifest incl. the T3a rows).
- T6 needs T5b, because it edits `browser/layer_c/parity.mjs`, which T5a creates and T5b freezes.
  T6 also waits for T4, so the tracked manifest is final (all ORT, c-perturbed and vmfb rows)
  before any perf run hashes it.
- T8 needs T6 (the profile harness reuses the page) and T5b.
- T4 and T5a/T5b may run in either order, but not concurrently (the lock). Whichever runs second
  reads the manifest re-copied by the other's `record` commit. T4's vmfb rows are not consumed by
  T5, so the order does not change T5's inputs.
- Pre-registration commits are explicit points in the graph. Between T3a→T3b and T5a→T5b there
  is a commit `T3b: pre-register` / `T5b: pre-register` that holds only the params, the validate
  sidecar's `min_effect`, the claim file and the re-copied manifest. That is the "one commit per
  task" rule applied per sub-task.

### Task 1: RNG-free P03/P04 wrappers, export ladder, jaxpr RNG walker

1. `src/aminx/model/features.py`: extract `:208-229` verbatim into a public `select_neighbors(distances:
   Array, mask: Array, k: int, structure_mapping: Array | None = None) -> Int[Array, "L k"]`, which
   includes the `jnp.inf` masking, the `.squeeze()` and the `min(k, L)` clamp. `forward_edge_stages`
   calls it at the same place. No other change to features.py.
2. `src/aminx/export/{__init__,buckets,wrappers,rng_audit}.py` (new):
   - `buckets`: `EXPORT_BUCKETS`, `ExportLengthError(TilingError)`, `LengthBelowNeighborsError`,
     `LengthAboveMaxBucketError`, `select_export_bucket(n_real, k_neighbors, buckets=EXPORT_BUCKETS)`,
     `pad_inputs(x4, mask, residue_index, chain_index, bucket) -> dict[str, np.ndarray]` (host NumPy;
     D-C convention; `mask` f32, indices int32).
   - `wrappers`: `make_p03_featurize(model) -> fn(coords, mask, residue_index, chain_index) ->
     (neighbor_indices int32 (L,k), edge_features f32 (L,k,H))` and `make_p04_unconditional(model,
     stage_set) -> fn(...) -> (logits f32 (L,21), neighbor_indices int32 (L,k))`, per D-B. Both
     assert the V10 topology equality at construction time and raise `ValueError` naming both values.
     Both record `zero_dropout`-style stats (`n_dropout`, `max_p_before`) on the returned fn's
     `.meta` without modifying the model (the path is dropout-free because key=None, V17). Also
     `zero_dropout(model) -> (model, stats)` per D-H.
   - `rng_audit`: `RNG_PRIMITIVES = {random_bits, random_seed, random_wrap, random_unwrap,
     random_split, random_fold_in, random_clone, threefry2x32, rng_bit_generator, rng_uniform}`;
     `find_rng_primitives(closed_jaxpr) -> list[str]` recurses into every eqn param holding a
     `Jaxpr`/`ClosedJaxpr` (or tuples of them: cond branches, pjit, while cond/body, scan,
     custom_jvp/vjp).
3. `tests/export/test_export_wrappers.py` (new), CPU only, synthetic structures (random × 10 Å, seed
   fixed):
   The shared fixture builds `native_model, stats = zero_dropout(model)` and asserts
   `stats["n_dropout"] > 0` (so the zeroing is not vacuous) and that every `Dropout.p == 0` in
   `native_model`. Every native call asserts `config.inference is True` (D-H). The wrapper is built
   on the unmodified `model`.
   - (a) W-equivalence for P03 and P04 vs native (D-H: `native_model.features(PRNGKey(0), …, 0.0)`;
     `score_unconditional.kernel(native_model, PRNGKey(0),
     build_inference_bundle(mode="score_unconditional", …), config, make_stage_set())`) at every
     bucket;
   - (b) P27 padded vs unpadded native, for k = 48 and for a random-init k = 32 model (X1);
   - (c) refusal at L = 47 and L = 1025 and acceptance at L = 48 and L = 1024 (k = 48); refusal at
     L = 31 and L = 1025 and acceptance at L = 32 and L = 1024 (k = 32);
   - (d) `select_neighbors` bitwise vs a frozen copy of the pre-refactor code kept in the test, on
     `_lattice.cubic_lattice_ca(96)`, a random structure, the same random structure with the last
     20 rows masked, a two-structure `structure_mapping` case, and a random L = 40 structure with
     k = 48 (so the clamp is live; `select_neighbors` is below the export refusal layer);
   - (e) `find_rng_primitives(make_jaxpr(wrapper)) == []` for both wrappers; the kernel jaxpr, a
     planted `normal`, RNG-in-cond-in-jit and a planted `jax.lax.rng_bit_generator` are all flagged;
   - (f) the topology assertion fires on a patched `k_neighbors`;
   - (g) key-independence of native: `kernel(native_model, PRNGKey(0), …)` and `kernel(native_model,
     PRNGKey(1), …)` at noise 0 are **bitwise** equal (`np.array_equal`), at bucket 128.

   Red-check mutations, one per test group:
   - pass `idx[:, ::-1]` into `forward_edge_stages` → (a) fails;
   - drop `mask` from `select_neighbors` masking → (b) fails;
   - `>` to `>=` on the max-bucket check → (c) fails at L = 1024 (boundary acceptance); `<` to `<=`
     on the min check → (c) fails at L = 48;
   - in `select_neighbors` itself: change the clamp `min(k, L)` → `min(k, L - 1)` (it bites on the
     L = 40 case: 39 vs 40 columns), and separately replace the `jnp.inf` mask fill with `0.0`
     (it bites on the masked case) → (d) fails for each;
   - insert `jax.random.normal(jax.random.PRNGKey(0), ()) * 0.0` into the P04 wrapper → (e) fails;
   - remove the topology assert → (f) fails;
   - use `model` instead of `zero_dropout(model)` for native → (g) fails. This also confirms V17
     empirically: if (g) stays green under this mutation, the checkpoint's `p` is already 0 and the
     commit body says so. In that case the mutation is repeated on a random-init model with
     `p = 0.1`, which must go red.

**Files**: `src/aminx/model/features.py` (modify); `src/aminx/export/__init__.py`, `buckets.py`,
`wrappers.py`, `rng_audit.py` (create); `tests/export/test_export_wrappers.py` (create).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
env $CAPS uv run --frozen --extra=dev pytest tests/export/test_export_wrappers.py -q -rs 2>&1 | tee /tmp/t1.log
grep -L "SKIPPED" /tmp/t1.log
env $CAPS uv run --frozen --extra=dev pytest tests/export/test_top_k_export.py -q   # existing top_k contract still holds
for f in src/aminx/model/features.py src/aminx/export/*.py; do uv run --frozen --extra=dev ruff check --no-fix "$f"; uv run --frozen --extra=dev ruff format --check "$f"; done
uv run --frozen --extra=dev ty check src/aminx/export src/aminx/model/features.py
uv run --frozen --extra=dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B tests/export/test_export_wrappers.py 2>&1 | tee /tmp/t1r.log; grep -L "No Python files found" /tmp/t1r.log
```
**Evidence that closes it**: the green test log plus the quoted red-check failures (one per mutation
above) in the commit body, and the loaded checkpoint's `max_p_before` (F-D1b input).
**Scope estimate**: ~290 LOC src + ~300 LOC tests.

### Task 2: run helpers, campaigns, ORT artifact build + RNG audit (jaxpr + ONNX)

1. `$BV/local_run.sh` and `$BV/check_run.sh` implement the Common-context contracts. Self-tests:
   `local_run.sh --self-test` refuses a planted tracked-dirty file (exit 1).
   `check_run.sh --stem no_such_stem --hash $(git rev-parse HEAD) …` exits 6, and an allowed set that
   excludes the real outcome exits 1 (red-check on the first real run below).
   `check_run.sh --self-test-lock` exits 4 while a background `flock /tmp/bv-local.lock sleep 30`
   holds the lock and it is run with `-w 1`.
   Also create `$BV/sidecar_dryrun.py` and `$BV/bv_stats_guard.py` (Common context), plus
   `tests/parity/test_bv_stats_guard.py`: constant-array Pearson → `null`, α with a zero operand →
   `null`, zero-denominator ratio → `null`, and a result with a NaN gated value plus its
   `_available = false` is classified `incomplete` by a sample sidecar's dry-run. Red-check: remove the
   zero-variance guard → the Pearson test fails.
   Before first use, run `bth sql "DESCRIBE runs"` to confirm the column names (`git_hash`,
   `sidecar_sha256`, `git_dirty`, `outcome`, `command`, `timestamp`), and confirm the
   sidecar-hash formula from bathos source (Common context "Run verification").
2. Create the three campaigns; record the ids in `layer_b/campaigns.json`.
3. `$BV/onnx_audit.py`: `find_onnx_rng_ops(model_proto) -> list[str]`. It walks the main graph,
   every subgraph attribute (If/Loop/Scan, at any depth), and every `FunctionProto` in
   `model_proto.functions`. A node whose `(domain, op_type)` matches a local function's
   `(domain, name)` is **resolved**: the walker descends into that function body (recursively,
   with a visited set against cycles) *before* any domain check is applied to the call node. It
   flags `{RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike, Multinomial,
   Bernoulli}` wherever found, as `rng:<path>/<op>`, where `<path>` names the graph/subgraph/function
   chain. An unresolved node in domain `com.microsoft` is labelled `ms-domain:<op>`, reported
   separately and treated as a flag. Any other unresolved node whose domain is not in `{"",
   "ai.onnx", "ai.onnx.ml"}` is labelled `unknown-domain:<d>/<op>`.
4. `$BV/layer_b_build.py` (`--out`, `--artifacts-dir`, `--buckets 128,256,512,1024`, `--dry-run`).
   For each path in {p03, p04} and each bucket:
   - build the wrapper;
   - convert with `jax2onnx.to_onnx(fn, specs, model_name=f"{path}_L{b}", output_path=…,
     return_mode="file")` (call form as `jax2onnx_spike.py:244-246`; record
     `inspect.signature(jax2onnx.to_onnx)`);
   - on failure, record the verbatim error and primitive (`jax2onnx_spike._extract_primitive`,
     imported, not copied).

   Control artifacts per bucket:
   - `p04_L{b}_perturbed.onnx` (δ = 5e-4 for build; T3a calibrates δ_b04);
   - `p03_L{b}_ebias.onnx` (`w_e_proj.bias += 1e-4` for build; T3a calibrates δ_b03);
   - `p03_L{b}_swap.onnx` (a wrapper variant that swaps neighbour slots k-2/k-1 of real row 0).

   Run the RNG audit on every clean wrapper jaxpr and clean ONNX, plus the five AC-B2 planted
   controls. The ONNX plants are copies of `p04_L128.onnx`, one with an injected `RandomUniform`
   inside an `If` subgraph and one with a `RandomUniform` inside a local `FunctionProto` (custom
   domain, e.g. `bv.test`) called from the main graph. The artifacts are written into
   `<--artifacts-dir>/<H12>/` (D-G). `artifact_manifest.json` is written there via
   `layer_b_common.manifest_append` with `artifact_subdir = <H12>` and the AC-B3 fields (opset from
   `model.opset_import`, op-type histogram, I/O names/shapes/dtypes, checkpoint sha256).
   `layer_b_common.py` is created here with `manifest_append`, `load_manifest_verified` and
   `artifact_path`. T3a extends it.
5. Sidecar `layer_b_build.bth.toml` (`exploration`, campaign `aminx-bv-layer-b`). Outcomes in order:
   - `ctrl_blind` (residual): any planted RNG control not flagged;
   - `rng_found` (residual): any clean jaxpr or ONNX has an RNG primitive/op;
   - `not_converted`: any clean artifact failed to convert;
   - `pass`.

   Schema `{n_artifacts:int, n_converted:int, n_rng_clean:int, controls_total:int,
   controls_detected:int, converted_by_path:dict, opsets:dict, versions:dict, artifacts:dict}`.
6. After a `pass` or `not_converted` record: copy the manifest to
   `outputs/browser_validation/layer_b/artifact_manifest.json` and `git add -f` it (the
   `T2: record` commit).

**Per-path BLOCKED table (r1 M3).** `converted_by_path[p]` is true iff every clean bucket artifact
for path p converted.

| Build result | Downstream |
|---|---|
| `pass` | all tasks proceed |
| `not_converted`, P03 only | T3a/T3b/T5a/T5b/T6/T8 run for P04 only. P03 rows are `blocked_not_converted`. The P03 edge control (δ_b03) and swap control are not built; the P04 perturbed control and a P04 swap variant (swap slots in the P04 wrapper's returned indices) carry the index-control role. T4 (IREE, no ONNX) still runs P03 |
| `not_converted`, P04 only | T3a/T3b/T5a/T5b/T6/T8 run for P03 only. The δ_b04 control is replaced by δ_b03 as `ctrl_effect`. T4 still runs P04 |
| `not_converted`, both | T3a/T3b/T5a/T5b/T6/T8 are BLOCKED (each writes a `blocked` result through its own local_run so the block is on record). T4 and T7 proceed |
| `ctrl_blind` / `rng_found` | the gate fails. Nothing downstream runs until fixed and re-run at a new commit |

Every downstream sidecar therefore also declares `blocked` (whitelisted) as its first-matching
outcome when `n_paths_available == 0`.

**Files**: `$BV/local_run.sh`, `$BV/check_run.sh`, `$BV/sidecar_dryrun.py`, `$BV/bv_stats_guard.py`,
`$BV/onnx_audit.py`, `$BV/layer_b_common.py`, `$BV/layer_b_build.py`, `$BV/layer_b_build.bth.toml`,
`$BV/layer_b_build.cases.json`, `tests/parity/test_bv_stats_guard.py`, `tests/parity/test_onnx_audit.py`
(create); `outputs/browser_validation/layer_b/{campaigns,artifact_manifest}.json` (force-added).
`test_onnx_audit.py` builds tiny ONNX models with `onnx.helper`: clean → `[]`; RandomUniform in an
`If` branch, in a nested `Loop` body, and in a local function → each flagged with its path; a
`com.microsoft` op → `ms-domain:`. Red-check: skip the `model_proto.functions` descent → the
function case fails.
**Gate**:
```bash
cd "$WT" && set -euo pipefail
bash $BV/local_run.sh --self-test
bash $BV/check_run.sh --self-test-lock
env $CAPS uv run --frozen --extra=dev $ORTW pytest tests/parity/test_onnx_audit.py tests/parity/test_bv_stats_guard.py -q -rs 2>&1 | tee /tmp/t2t.log; grep -L "SKIPPED" /tmp/t2t.log
bth validate-sidecar $BV/layer_b_build.bth.toml
uv run --frozen --extra=dev python $BV/lint_sidecars.py --allow-placeholder
uv run --frozen --no-sync $BATHOSW python $BV/sidecar_dryrun.py --sidecar $BV/layer_b_build.bth.toml --cases $BV/layer_b_build.cases.json
env $CAPS uv run --frozen --extra=dev --extra=benchmark $ORTW python $BV/layer_b_build.py --dry-run --out /tmp/build_dry.json --artifacts-dir /tmp/bvdry
git add $BV/local_run.sh $BV/check_run.sh $BV/sidecar_dryrun.py $BV/bv_stats_guard.py $BV/onnx_audit.py $BV/layer_b_common.py $BV/layer_b_build.py $BV/layer_b_build.bth.toml $BV/layer_b_build.cases.json tests/parity/test_onnx_audit.py tests/parity/test_bv_stats_guard.py && git commit -m "T2: layer-b build + RNG audit (pre-registered)"
# campaigns are created (step 2) and campaigns.json force-added before this point
CB=$(jq -er '."aminx-bv-layer-b"' "$CAMPAIGNS")
eval "$(bash $BV/local_run.sh --stem layer_b_build --campaign "$CB" --subdir layer_b $ORTW $BATHOSW -- --artifacts-dir "$BV_ARTIFACTS" | tail -1)"
bash $BV/check_run.sh --stem layer_b_build --hash "$H" --run-id "$RUN_ID" --sidecar $BV/layer_b_build.bth.toml --allowed pass,not_converted
if bash $BV/check_run.sh --stem no_such_stem --hash "$H" --sidecar $BV/layer_b_build.bth.toml --allowed pass,not_converted; then exit 1; fi   # red-check: must fail (6)
if bash $BV/check_run.sh --stem layer_b_build --hash "$H" --sidecar $BV/layer_b_build.bth.toml --allowed rng_found; then exit 1; fi   # red-check: must fail (1)
for f in $BV/onnx_audit.py $BV/layer_b_build.py $BV/layer_b_common.py $BV/sidecar_dryrun.py $BV/bv_stats_guard.py tests/parity/test_onnx_audit.py tests/parity/test_bv_stats_guard.py; do uv run --frozen --extra=dev ruff check --no-fix --config "exclude=[]" --select E,F,W,I,B "$f" 2>&1 | tee /tmp/t2r.log; grep -L "No Python files found" /tmp/t2r.log; uv run --frozen --extra=dev ruff format --check --config "exclude=[]" "$f"; done
```
**Evidence that closes it**: a record with outcome `pass` and `controls_detected = controls_total =
5`, plus the committed manifest (`T2: record`). Alternatively, a `not_converted` record with
`controls_detected = controls_total`, which is a legitimate finding: the task closes, downstream
follows the BLOCKED table, and the verbatim primitive goes to ODQ-B10.
**Scope estimate**: ~180 LOC shell + ~450 LOC Python + ~120 LOC tests.

### Task 3a: ORT-CPU layer-(b) calibrate (set A)

1. Extend `$BV/layer_b_common.py` (created in T2). It loads the tracked manifest and verifies every
   sha256 (exit 3). It builds padded inputs per (fixture, bucket) with `aminx.export.buckets`. It
   runs ORT-CPU (`providers = ["CPUExecutionProvider"]`, `intra_op_num_threads = 4`; recorded, D-I;
   timing is irrelevant here), the native wrapper, and native (D-H, on `zero_dropout(model)`). It
   computes the bar rows, applying the **near-tie rule** (id-aligned edge comparison,
   `n_tie_swapped_slots`) and the **headroom keys** `(path, quantity, route, bucket)`. It also runs
   one profiled session per artifact to build the EP-assignment histogram (`args.provider` of node
   events).
2. `$BV/layer_b_ort_calibrate.py` on set A:
   - 5L33 → 128; tie_lattice_L96 → 128 (P03 only); 4GYT → 512; 6EHB → 1024;
   - 2GFB (3464) → a `LengthAboveMaxBucketError` row, validated iff raised;
   - cropped **5L33[:40]** (set A; r1 minor) → a `LengthBelowNeighborsError` row;
   - size δ_b04 so the perturbed P04 log-prob max-abs is in [2×, 10×] of 1e-4, and δ_b03 so the
     `w_e_proj.bias`-perturbed P03 edge max-abs is in [2×, 10×] of 2e-5. Search: geometric
     bisection over δ ∈ [1e-7, 1e-1], ≤ 20 steps, on 5L33@128. Re-build both perturbed artifacts
     with the chosen δs into `$BV_ARTIFACTS/<artifact_subdir>/` and add their rows via
     `manifest_append`;
   - record `max_abs_dist_err` (near-tie ε check) and `max_p_before` (F-D1b);
   - apply the headroom rule; write `outputs/browser_validation/layer_b/preregistered_params.json`
     with `{delta_b04, delta_b03, ctrl_effect: {p04, p03}, epsilon_tie: 1e-4, max_abs_dist_err,
     headroom: {"<path>|<quantity>|<route>|<bucket>": {"state": "advanced"|"low_headroom"|"not_advanced",
     "calib_value": float, "inherited_from": int|null}}}`.
   - Sidecar `calibration`. Outcomes in order: `blocked` (no path available, T2 table),
     `incomplete` (residual), `ctrl_unsized` (a δ search failed to land in its window, or ε <
     5 × `max_abs_dist_err`), `pass`. `not_advanced` rows do not change the calibrate outcome.

**Files**: `$BV/layer_b_common.py` (modify), `$BV/layer_b_ort_calibrate.py`, `.bth.toml`,
`.cases.json` (create); `outputs/browser_validation/layer_b/{preregistered_params,artifact_manifest}.json`
(force-added in the pre-register commit).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
bth validate-sidecar $BV/layer_b_ort_calibrate.bth.toml
uv run --frozen --extra=dev python $BV/lint_sidecars.py --allow-placeholder
uv run --frozen --no-sync $BATHOSW python $BV/sidecar_dryrun.py --sidecar $BV/layer_b_ort_calibrate.bth.toml --cases $BV/layer_b_ort_calibrate.cases.json
git add $BV/layer_b_common.py $BV/layer_b_ort_calibrate.py $BV/layer_b_ort_calibrate.bth.toml $BV/layer_b_ort_calibrate.cases.json && git commit -m "T3a: layer-b ORT calibrate (pre-registered)"
eval "$(bash $BV/local_run.sh --stem layer_b_ort_calibrate --campaign "$CB" --subdir layer_b $ORTW $BATHOSW -- --artifacts-dir "$BV_ARTIFACTS" | tail -1)"
bash $BV/check_run.sh --stem layer_b_ort_calibrate --hash "$H" --run-id "$RUN_ID" --sidecar $BV/layer_b_ort_calibrate.bth.toml --allowed pass
```
Plus ruff (common-context form).
**Evidence that closes it**: the verified record, plus `preregistered_params.json` and the
re-copied manifest (with the δ-rebuilt rows) in the copied-back run directory.
**Scope estimate**: ~260 LOC.

### Task 3b: ORT-CPU layer-(b) validate (set B)

0. **Pre-registration commit `T3b: pre-register`** (before any validate code runs). It holds:
   `preregistered_params.json` copied from the T3a run dir; the base manifest re-copied to the
   tracked path (`git add -f`, per D-G); the validate sidecar with `[differential].min_effect =
   min(ctrl_effect.p04 / 1e-4, ctrl_effect.p03 / 2e-5)` (effect in bar units); and the layer-(b)
   claim file registered with `bth claim register <claim-file> --campaign-id "$CB"`, modelled on
   the layer-(a) claim file found via `git ls-files | grep -i claim`. The claim is registered
   before the first validate run.
1. `$BV/layer_b_ort_validate.py` on set B:
   - (fixture, bucket) pairs = every bucket ≥ the fixture's own (6MRR/1BC8: 128/256/512/1024;
     3HTN: 512/1024; 4YOW: 1024);
   - **W rows on real data (r1 M10):** wrapper vs native (D-H; P04 via
     `layer_a_exact._score_aminx_unconditional(zero_dropout(model)[0], batch)`, P03 via
     `native_model.features(PRNGKey(0), …, 0.0)`) at each fixture's own bucket, with the W bars;
   - P27 rows (wrapper at bucket vs native unpadded, D-H) and the two refusal rows (the same
     set-A inputs as T3a: 2GFB and 5L33[:40]);
   - rows whose headroom state is `not_advanced` are reported as `excluded_not_advanced` and not
     gated;
   - controls (counted in `controls_total = 3`): perturbed P04 weight detected (max-abs > 1e-4),
     perturbed P03 `w_e_proj.bias` detected (edge max-abs > 2e-5), planted swap detected (indices
     not exact on a non-near-tie row). Sanity check (not a control): comparator self-test
     (identical arrays → 0, must pass).
   - Sidecar `validation`, `[differential].min_effect` as in step 0. Outcomes in order: `blocked`,
     `incomplete` (skips > 0 or `NOT git_clean`), `ctrl_blind`, `fail`, `tie_unstable`,
     `partial_headroom` (any `low_headroom` or `not_advanced` key), `pass`.

**Files**: `$BV/layer_b_ort_validate.py`, `.bth.toml`, `.cases.json` (create); claim file (create);
params + manifest (force-added).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
# step 0 commit done; then:
bth validate-sidecar $BV/layer_b_ort_validate.bth.toml
uv run --frozen --extra=dev python $BV/lint_sidecars.py        # no placeholder allowed now
uv run --frozen --no-sync $BATHOSW python $BV/sidecar_dryrun.py --sidecar $BV/layer_b_ort_validate.bth.toml --cases $BV/layer_b_ort_validate.cases.json
git diff --quiet HEAD -- outputs/browser_validation/layer_b/artifact_manifest.json outputs/browser_validation/layer_b/preregistered_params.json
eval "$(bash $BV/local_run.sh --stem layer_b_ort_validate --campaign "$CB" --subdir layer_b $ORTW $BATHOSW -- --artifacts-dir "$BV_ARTIFACTS" | tail -1)"
bash $BV/check_run.sh --stem layer_b_ort_validate --hash "$H" --run-id "$RUN_ID" --sidecar $BV/layer_b_ort_validate.bth.toml --allowed pass,partial_headroom,tie_unstable
```
Plus ruff (common-context form).
**Evidence that closes it**: the verified record. The validate record's per-row detail (copied-back
JSON) lists every AC-B3 row (including the W rows and `excluded_not_advanced` rows) and the EP
report.
**Scope estimate**: ~260 LOC.

### Task 4: IREE route layer (b) — safety, native parity, WASM32 codegen

1. Step 1 check: `uv run --frozen --extra=dev python -c "import iree.compiler, iree.runtime; from
   xtrax.export import compile_for_target, run_native_vmfb, check_export_safety, NATIVE, WASM32"`.
   If this fails, the run writes `status = "BLOCKED"` with the error (outcome `blocked`) and T4 closes
   as a finding.
2. `$BV/layer_b_iree.py` (`--out`, `--artifacts-dir`, `--buckets`). For both wrappers per bucket:
   - `trace_ok` via an independent `jax.make_jaxpr`;
   - `check_export_safety([], {}, specs, fn, NATIVE)` and `(…, WASM32)` must both be `[]`;
   - the census controls must be re-detected (`random_decoding_order` → `random-permutation`;
     `lambda x: jax.lax.top_k(x, 8)` → `unlegalizable-op`);
   - `mlir = jax.export.export(jax.jit(fn))(*specs).mlir_module()`;
   - `compile_for_target(mlir, NATIVE, out_path=…)`;
   - run on set-B inputs via `run_native_vmfb` and compare against the native wrapper with the b
     bars, applying the near-tie rule and T3a's headroom keys with `route = "iree"`. IREE has no
     calibrate pass, so IREE keys are inherited from the `route = "ort"` key with the same
     `(path, quantity, bucket)` and marked `inherited_route`. That is stated as a limitation, not
     a bar change. The three planted variants (P04 δ_b04, P03 δ_b03 `w_e_proj.bias`, swap) are
     compiled from the same perturbed wrappers and must be detected;
   - `compile_for_target(mlir, WASM32, …)` → record the sha256 and `verification = CODEGEN_ONLY`;
   - add all vmfb rows via `manifest_append` (`route = "iree"`) into
     `$BV_ARTIFACTS/<artifact_subdir>/`.

   Record `rings = "not_available"` (V8) and compile wall time per bucket.
3. Sidecar `layer_b_iree.bth.toml` (`validation`, campaign b; `[differential].min_effect` = the
   T3b value from params). Outcomes in order: `blocked`, `incomplete`, `unsafe` (residual: any
   blocker or `trace_ok = false`), `ctrl_blind`, `fail`, `tie_unstable`, `partial_headroom`,
   `pass`. On the step-1 import failure the script still exits 0 with a full-schema result (the
   result-emission rule).
4. After the run: a `T4: record` commit re-copies the base manifest (vmfb rows) to the tracked
   path.

**Files**: `$BV/layer_b_iree.py`, `.bth.toml`, `.cases.json` (create); manifest (force-added update).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
bth validate-sidecar $BV/layer_b_iree.bth.toml && uv run --frozen --extra=dev python $BV/lint_sidecars.py
uv run --frozen --no-sync $BATHOSW python $BV/sidecar_dryrun.py --sidecar $BV/layer_b_iree.bth.toml --cases $BV/layer_b_iree.cases.json
eval "$(bash $BV/local_run.sh --stem layer_b_iree --campaign "$CB" --subdir layer_b $BATHOSW -- --artifacts-dir "$BV_ARTIFACTS" | tail -1)"
bash $BV/check_run.sh --stem layer_b_iree --hash "$H" --run-id "$RUN_ID" --sidecar $BV/layer_b_iree.bth.toml --allowed pass,tie_unstable,partial_headroom,blocked
```
Plus ruff (common-context form).
**Evidence that closes it**: the verified record. If `blocked`/`unsafe`, the reason is quoted in the
handoff and the IREE route is marked not-validated.
**Scope estimate**: ~300 LOC.

### Task 5a: Browser layer (c) harness + set-A calibrate (ORT Web wasm EP, headless Chromium)

**Definitions (r1 B4).**
- A **session** is one Chromium browser process launched by Playwright, plus a fixed, recorded set
  of browser contexts and pages opened at start-up (`session_id` = UUID4, recorded).
- A **context** is one `browser.newContext()` holding exactly one page. `ort.env.wasm.numThreads`
  is fixed when the wasm backend initialises, so **each thread setting gets its own context**, and
  `numThreads` is set before the first `InferenceSession.create` in that page. Parity uses one
  context with `numThreads = 1`.
- **wasm-init thread assertion.** After the first session-create, each page records
  `ort.env.wasm.numThreads` (value read back) and the thread count the backend actually uses. The
  mechanism is **to be verified for ORT Web 1.30.0** in this task: candidate signals are the
  number of pthread `Worker`s spawned (counted by wrapping `self.Worker` before ORT loads) and any
  ORT verbose log line naming the thread pool size. Whichever signal the test below validates
  becomes `threads_observed`. The assertion is `threads_observed == threads_requested`.
- **Verification test** (`browser/layer_c/tests/threads.spec.mjs`, Playwright). Two contexts in one
  session, requesting 1 and 4 threads. It asserts: `threads_observed` = 1 and 4 respectively; in
  the 1-thread page, setting `numThreads = 4` **after** the first session-create and then creating
  a second session still reports 1 (this documents the init-time fixing, and the test fails if 1.30
  behaves otherwise); with
  `--no-isolation`, a request for 4 reports 1 (no SAB). If the observed-thread signal cannot be
  validated, `threads_observed = null`, `threads_observed_available = false`, and T8 must not
  print any tN sentence.

1. `browser/layer_c/`:
   - `package.json` pinning `onnxruntime-web@1.30.0`, `@playwright/test@1.63.0` (same as
     `browser/smoke/package.json`), plus `package-lock.json` (commit it);
   - `serve.mjs`: COOP `same-origin` + COEP `require-corp`; `--no-isolation` flag; `POST
     /results/<name>` writes the raw body to `--out-dir` with a path-sanitised name;
   - `index.html` + `parity.mjs`: for each cell in `cells.json`, fetch the model and its raw input
     buffers; `ort.env.wasm.numThreads = <context's setting>` before the first create;
     `InferenceSession.create(buf, {executionProviders: ['wasm']})`; run; POST every output's raw
     bytes; finally POST `evidence.json` (UA, `ort.env.versions`, EP, the wasm-init thread
     assertion per context, `self.crossOriginIsolated`, `session_id`, and the WebGPU probe result);
   - `run_parity.mjs`: Playwright chromium headless; non-zero exit on any page error or on timeout
     (the Python driver turns that into a result row; it does not propagate the exit, per the
     result-emission rule);
   - `tests/threads.spec.mjs` (above).
2. `$BV/layer_c_common.py` (verify manifest shas, exit 3; write padded inputs + `cells.json`;
   compute ORT-CPU reference outputs in-process; run the page; compare with the c bars using the
   near-tie rule and headroom keys with `route = "ort_wasm"`). Then `$BV/layer_c_calibrate.py` on
   set A (5L33@128, tie_lattice_L96@128 P03 only, 4GYT@512, 6EHB@1024):
   - measure the c comparisons (ORT Web vs ORT-CPU, same artifact) and apply the headroom rule
     against the c bars;
   - size δ_c04 (`w_out.bias[0]`) to [2×, 10×] of 1e-5 and δ_c03 (`w_e_proj.bias`) to [2×, 10×]
     of 2e-6, using the T3a bisection procedure. The perturbation is applied to the *reference*
     side: a perturbed artifact run in the browser vs the clean artifact on ORT-CPU. Build the two
     c-perturbed artifacts via `manifest_append`;
   - write `outputs/browser_validation/layer_c/preregistered_params.json` (`{delta_c04, delta_c03,
     ctrl_effect_c, headroom_c}`).
   - Sidecar `layer_c_calibrate.bth.toml` (`calibration`, campaign c). Outcomes: `blocked`,
     `incomplete`, `not_isolated`, `ctrl_unsized`, `pass`.
3. Gitignore `browser/**/node_modules/` (already present per parent T3; verify).

**Files**: `browser/layer_c/{package.json,package-lock.json,serve.mjs,index.html,parity.mjs,run_parity.mjs,tests/threads.spec.mjs}`,
`$BV/layer_c_common.py`, `$BV/layer_c_calibrate.py`, `.bth.toml`, `.cases.json` (create).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
(cd browser/layer_c && npm ci && npx playwright test tests/threads.spec.mjs --reporter=line) 2>&1 | tee /tmp/t5a_threads.log
bth validate-sidecar $BV/layer_c_calibrate.bth.toml && uv run --frozen --extra=dev python $BV/lint_sidecars.py --allow-placeholder
uv run --frozen --no-sync $BATHOSW python $BV/sidecar_dryrun.py --sidecar $BV/layer_c_calibrate.bth.toml --cases $BV/layer_c_calibrate.cases.json
# commit "T5a: layer-c harness + calibrate (pre-registered)", then:
eval "$(bash $BV/local_run.sh --stem layer_c_calibrate --campaign "$CC" --subdir layer_c --npm-dir browser/layer_c $ORTW $BATHOSW -- --artifacts-dir "$BV_ARTIFACTS" | tail -1)"
bash $BV/check_run.sh --stem layer_c_calibrate --hash "$H" --run-id "$RUN_ID" --sidecar $BV/layer_c_calibrate.bth.toml --allowed pass
```
Plus ruff on the Python.
**Evidence that closes it**: the thread test log, the verified record, and the c params.
**Scope estimate**: ~260 LOC JS + ~250 LOC Python.

### Task 5b: Browser layer (c) validate (set B)

0. **Pre-registration commit `T5b: pre-register`**: the c params, the re-copied manifest (with the
   c-perturbed rows), and the validate sidecar's `min_effect = min(ctrl_effect_c.p04 / 1e-5,
   ctrl_effect_c.p03 / 2e-6)`.
1. `$BV/layer_c_parity.py`:
   - write padded set-B inputs and `cells.json` (the T3b validate pairs, plus the c-sized
     perturbed and the swap control artifacts at 128 and 1024);
   - run `node browser/layer_c/run_parity.mjs`, then once more with `--no-isolation` (negative
     control: `crossOriginIsolated` must be `false`);
   - compare with the c bars (near-tie rule; `not_advanced` keys excluded); report the transitive
     gap vs native against the b bars; write
     `outputs/browser_validation/layer_c/evidence/ort-wasm__chromium.json`.
2. Sidecar `layer_c_parity.bth.toml` (`validation`, campaign c; `min_effect` from step 0).
   Outcomes in order: `blocked`, `incomplete`, `not_isolated` (residual: the isolated run reported
   `false`), `ctrl_blind` (a c-sized control was missed, or the negative control reported `true`),
   `fail`, `tie_unstable`, `partial_headroom`, `pass`.

**Files**: `$BV/layer_c_parity.py`, `.bth.toml`, `.cases.json` (create); c params + manifest
(force-added).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
bth validate-sidecar $BV/layer_c_parity.bth.toml && uv run --frozen --extra=dev python $BV/lint_sidecars.py
uv run --frozen --no-sync $BATHOSW python $BV/sidecar_dryrun.py --sidecar $BV/layer_c_parity.bth.toml --cases $BV/layer_c_parity.cases.json
eval "$(bash $BV/local_run.sh --stem layer_c_parity --campaign "$CC" --subdir layer_c --npm-dir browser/layer_c $ORTW $BATHOSW -- --artifacts-dir "$BV_ARTIFACTS" | tail -1)"
bash $BV/check_run.sh --stem layer_c_parity --hash "$H" --run-id "$RUN_ID" --sidecar $BV/layer_c_parity.bth.toml --allowed pass,tie_unstable,partial_headroom
jq -e '.cross_origin_isolated == true and .execution_provider == "wasm" and (.browser_version|length) > 0 and ([.contexts[] | .threads_observed == .threads_requested or .threads_observed_available == false] | all)' outputs/browser_validation/layer_c/runs/layer_c_parity-${H:0:12}/evidence/ort-wasm__chromium.json
```
Plus ruff on the Python.
**Evidence that closes it**: the verified record plus the evidence JSON. No Firefox, WebKit or
WebGPU compatibility claim is made.
**Scope estimate**: ~180 LOC Python.

### Task 6: Profiling — ORT per-op (CPU + Web) and JAX-native cost analysis

1. `$BV/ort_profile_agg.py`: parses ORT profile JSON (node events `cat == "Node"`, `dur` in µs,
   `args.op_name`, `args.provider`). It drops the ≥ 5 warm-up runs by `model_run` boundaries, then
   aggregates per op type (mean µs/run, share, count) and computes coverage = Σ node dur / Σ
   model_run dur. Unit test `tests/parity/test_ort_profile_agg.py`: a synthetic profile with known
   durations is recovered within 1e-9, and a truncated or malformed JSON raises. Red-check: an
   off-by-one in warm-up dropping must fail the test.
2. `$BV/layer_b_profile.py`, per artifact (p03/p04 × 4 buckets):
   - **ORT-CPU**: `enable_profiling`, 5 warm-ups + 30 runs → aggregate. Two thread settings
     (D-I): `intra_op_num_threads = inter_op_num_threads = 1` feeds the cross-route
     per-op table against ORT Web t1; `intra_op = 4` is reported only in its own table, which
     makes no cross-route statement. Each table carries its thread record.
   - **ORT Web**: a `--profile` mode of the T5a page (`enableProfiling: true`, `endProfiling()`,
     console captured by Playwright), in a 1-thread context (T5a definitions); parse if per-node
     events are emitted, otherwise `web_per_op_available = false`.
   - **JAX native**: `jax.jit(wrapper).lower(*specs).compile()`; record `cost_analysis()` (flops,
     bytes accessed, transcendentals), the HLO op histogram from `compiled.as_text()`, and
     `jax.profiler.trace` over 30 iterations → `xtrax.profiling.trace.parse_hlo_op_times` and
     `parse_dispatch_counts`. The XLA thread flag and `threads_observed` are recorded per D-I.
   - **Where the DISPATCH_COUNT counters are measured.** All counts are taken in the
     `layer_b_profile.py` process, over the same 30-iteration steady-state loop that runs after the
     5 warm-ups, per (path, bucket):
     - `n_executions` = the host loop count (30), cross-checked against the executable-launch
       count from `parse_dispatch_counts` on that loop's trace (mismatch → `ctrl_blind`);
     - `n_compilations` and `n_jit_traces` = the number of `jax.monitoring` duration events
       fired during that loop, from `jax.monitoring.register_event_duration_secs_listener`,
       filtered to the backend-compile event and the jaxpr-trace event respectively. The exact
       event-name strings are **to be verified** against jax 0.10.2 (`jax/_src/dispatch.py`,
       `jax/_src/interpreters/pxla.py`). The verification is in-run: a deliberately fresh
       `jax.jit(lambda x: x + 1)` call must increment both counters by ≥ 1, and a repeated call
       must increment neither (else `ctrl_blind`). Expected steady-state values: 0 and 0.
   - Build `ProbeRecord(stage=1, platform="cpu", n_atoms=bucket, metrics={total_step_seconds,
     n_executions, n_compilations, n_jit_traces}, config={path, bucket, route})` with `XTRAX_GIT_SHA=$H`
     exported after the clean check, and call `assert_claim_supported(records,
     ClaimClass.DISPATCH_COUNT)`.
   - **Malformed-record controls (r1 minor):** (i) a `stage=0` record whose metrics omit
     `n_jit_traces` must make `assert_claim_supported([rec], ClaimClass.DISPATCH_COUNT)` raise
     (V20); (ii) constructing a record with `metrics={"n_executions": True}` must raise
     `ClaimValidityError` (`record.py:228-233`). If either is accepted, the outcome is `ctrl_blind`.
   - Report a flops scaling exponent across buckets per path (descriptive; α guarded per the
     result-emission rule).
3. Sidecar `layer_b_profile.bth.toml` (`exploration`, campaign perf). Outcomes in order:
   - `blocked`: no path available (T2 table);
   - `ctrl_blind`: synthetic aggregator check fails in-run, a counter self-check fails, or a
     malformed-record control is accepted;
   - `incomplete`: any cell missing;
   - `low_coverage` (residual): any ORT-CPU coverage < 0.8;
   - `partial`: web per-op unavailable;
   - `pass`.

**Files**: `$BV/ort_profile_agg.py`, `$BV/layer_b_profile.py` (+`.bth.toml`), `browser/layer_c/parity.mjs`
(modify: `--profile`), `tests/parity/test_ort_profile_agg.py` (create).
**Gate** (runs after T5b; `$CP` from Common-context shorthands): the test file (+ zero skips) and
its red-check; the sidecar validate/lint/`sidecar_dryrun.py`; `local_run.sh --stem layer_b_profile
--campaign "$CP" --subdir perf --npm-dir browser/layer_c $ORTW $BATHOSW -- --artifacts-dir
"$BV_ARTIFACTS"`; then `check_run.sh --stem layer_b_profile --hash "$H" --run-id "$RUN_ID" …
--allowed pass,partial`; ruff. The `parity.mjs` change must leave T5b's parity path unchanged:
re-run `npx playwright test tests/threads.spec.mjs` and diff the output bytes of one parity cell
before and after the change (`cmp`).
**Evidence that closes it**: the verified record, whose per-artifact op-share tables are in the
copied-back JSON.
**Scope estimate**: ~350 LOC.

### Task 7: Loop-body structural cost report (P07 diagnostic; xtrax #1983 tie-in)

1. `$BV/hlo_loop_cost.py`: from `compiled.as_text()`, find every `while` instruction's
   `body=`/`condition=` computations. The metric is **`body_operand_elements`** (renamed in r1
   M14; it is not an output-element count). It is Σ over the body's instructions of a
   per-instruction weight:
   - `fusion`: counted **once, at the call site**, as Σ operand elements + output elements. The
     walker does **not** descend into the fused computation (`calls=` of a fusion), so fusion
     internals are never double-counted;
   - unfused `reduce`, `reduce-window`, `dot`, `convolution`, `gather`, `sort`: Σ operand elements
     + output elements (a reduction to a scalar costs its operand, not 1);
   - `dynamic-update-slice` / `scatter`: update operand (+ scatter indices) elements, not the full
     output; `dynamic-slice`: output elements;
   - any other unfused op: output elements;
   - `parameter`, `constant`, `tuple`, `get-tuple-element` and `bitcast` are excluded; `copy` is
     counted (a carried-buffer copy is exactly what #1983 wants to find).

   Traversal: descend into `call` (`to_apply=`/`calls=` of a `call` op) and into `conditional`
   branches (max over branches). A nested `while` has its body counted once and is flagged
   `nested_while` (trip count unknown). The scalar reducer computations named by `to_apply=` on
   reduce/scatter/sort/all-reduce are **not** descended.

   Scaling α = log2(m(2L) / m(L)) for `m = body_operand_elements`, guarded per the result-emission
   rule. Also report the top-10 body instructions by weight. Reuse
   `xtrax.profiling.trace.scope_map_from_hlo_text` for named-scope attribution where labels exist.
2. `$BV/layer_b_loop_cost.py`, L ∈ {128, 256, 512}:
   - synthetic instrument checks, each a toy `lax.while_loop` with 8 iterations:
     - **linear**: carried `(L×16)` f32, body `x * 2.0 + 1.0` (one fusion: 16L in + 16L out, so
       α = 1 exactly), α ∈ [0.9, 1.1];
     - **reduce-to-scalar (r1 M14)**: carried `(L,)` f32 plus a scalar accumulator, body
       `acc + jnp.sum(x)`. The weight is (L + 1) + 1, so α → 1; required α ∈ [0.9, 1.1]. An
       output-element metric would give α ≈ 0 here, which is exactly the blindness this control
       catches;
     - **constant**: body `(48×16)@(16×16)` independent of L, α ∈ [-0.1, 0.1];
   - the AR kernel via `make_decode_fn(... AutoregressiveConfig(inference_only=True,
     incremental=mode))` as in `bench_ar_incremental.py:64`, with a host `from_tie_groups(arange(L),
     arange(L))` wave, for mode ∈ {off, force};
   - also time 10 steady decodes per mode for context.
3. Sidecar (`exploration`, perf; P07 is labelled diagnostic in the schema). Outcomes in order:
   - `ctrl_blind`: any of the three toy α outside its interval (or `null`), or AR-off α < 0.8;
   - `loop_body_scales` (residual, a finding): AR-force α > 0.35, with offending instructions listed;
   - `pass`.
4. Write draft backlog text for xtrax #1983 (upstreaming `hlo_loop_cost`) and aminx #1981 (if
   `loop_body_scales`) into the result JSON `followup_backlog_text`. Do not file them.

**Files**: `$BV/hlo_loop_cost.py`, `$BV/layer_b_loop_cost.py` (+`.bth.toml`),
`tests/parity/test_hlo_loop_cost.py` (create: the parser on a handwritten HLO snippet with known
counts, containing a fusion, a DUS, an unfused reduce-to-scalar and a conditional). Red-checks, each
of which must fail the test: counting the DUS full output; descending into the fusion body
(double count); counting the reduce by output elements only.
**Gate** (after T2; independent of T3..T6): the test + red-checks; sidecar
validate/lint/`sidecar_dryrun.py`; `local_run.sh --stem layer_b_loop_cost --campaign "$CP"
--subdir perf $BATHOSW`; `check_run.sh --stem layer_b_loop_cost --hash "$H" --run-id "$RUN_ID" …
--allowed pass,loop_body_scales`; ruff.
**Evidence that closes it**: the verified record with α per mode and the top-10 lists.
**Scope estimate**: ~300 LOC.

### Task 8: Benchmark harness (Playwright, interleaved native arm)

0. **XLA thread-pin verification (D-I).** `tests/parity/test_xla_thread_pin.py` runs a 2048²
   f32 matmul loop in a **subprocess** (XLA flags are read at backend init), with and without
   `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. It asserts
   CPU-time/wall ≤ 1.2 pinned and > 1.5 unpinned. The flag spelling is to-verify. If the test
   cannot be made green with any documented jax 0.10.2 flag, the test is marked `xfail(strict=True)`
   with the reason, and the bench records `native_threads = "unpinned"` and prints no native ratio.
1. `$BV/bench_stats.py`: p50/p90; bootstrap CIs stratified by repeat (1000 resamples, recorded seed)
   for p50, p90 and route/native p50 ratio. Every CI is **within-session** by construction and is
   labelled that way in every emitted sentence. `tests/parity/test_bench_stats.py` has three
   synthetic checks, all drawing from lognormal(μ = log(10 ms), **σ = 0.25**) with a fixed seed:
   - known quantiles: 95 % CI coverage of p50 ≥ 0.9 over 200 replicates at n = 90;
   - identical-distribution ratio CI contains 1 in ≥ 0.9 of 200 replicates;
   - a +5 % multiplicative shift (μ + log 1.05, same σ = 0.25) is detected (ratio CI excludes 1)
     in ≥ 0.9 of 200 replicates at n = 90 × 3.

   Red-check: swap the percentile interpolation or drop the stratification → coverage test fails.
2. `browser/layer_c/bench.mjs` + `run_bench.mjs`: one Chromium **session** (T5a definition) driven
   over a JSON-lines stdin/stdout protocol. It holds exactly two contexts, created at start-up:
   `ctx_t1` (numThreads = 1) and `ctx_tN` (numThreads = N = min(8, hardwareConcurrency)). Each
   context's wasm-init thread assertion is recorded; `tN` cells whose assertion fails are marked
   `threads_mismatch` and are excluded from sentences. Per cell it times four phases:
   - fetch/load (fetch ArrayBuffer);
   - session-create;
   - first inference;
   - steady state: ≥ 5 warm-ups then ≥ 30 timed with `performance.now()`, after asserting
     `crossOriginIsolated`.

   **Peak memory (parent AC-23).** After each cell: `performance.measureUserAgentSpecificMemory()`
   (requires cross-origin isolation; availability in headless Chromium is to-verify, so
   `peak_memory_available` records it), plus the wasm heap size (`WebAssembly.Memory.buffer.byteLength`
   via ORT's exposed module where available). For `native_jax_cpu`: process RSS high-water
   (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, recorded before and after each cell) and
   `jax.devices()[0].memory_stats()` (typically `None` on CPU → `_available = false`).
3. `$BV/layer_c_bench.py` (the driver, one Python process = one session).
   - Cells = paths {p03, p04} × buckets {128, 256, 512, 1024} × routes {`ort_wasm_t1` (ctx_t1),
     `ort_wasm_tN` (ctx_tN), `native_jax_cpu` (in-process, pinned per step 0; its phases are weight
     load, lower+compile, first call and steady state with `block_until_ready`)}.
   - Control arms at p04@256: `aa_dup` (a second copy of `ort_wasm_t1`) and `planted_5ms` (the same
     plus a 5 ms busy-wait inside the timed region).
   - Cell order is shuffled per repeat with a recorded seed; 3 full-matrix repeats. The script
     records `os.sched_getaffinity`, thread caps, `hardwareConcurrency`, and per arm
     `{threads_requested, threads_observed, source}` (D-I).
   - ProbeRecords per cell: stage 1, platform cpu, n_atoms = bucket, **`metrics =
     {"total_step_seconds": <steady-state p50 in seconds>}`** (required for END_TO_END,
     claims.py:93, V20), `config` incl. `session_id`, `route`, `threads`, and `XTRAX_GIT_SHA=$H`.
   - The report sentence generator prints a timing only after `assert_claim_supported(…, END_TO_END,
     target_n_atoms=bucket)`, and a ratio only for pairs returned by `paired_configs(…, axis="route",
     hold_fixed=("path","bucket","session_id","threads"))`. Because `threads` is held fixed, only
     `native_jax_cpu` ↔ `ort_wasm_t1` pairs can be emitted (when native is pinned), and a tN vs
     native ratio is structurally impossible. Unit tests check that a `-dirty` git_sha record makes
     it refuse, that a record missing `total_step_seconds` makes it refuse, and that an
     unequal-threads pair yields no sentence.
   - No cross-session comparison anywhere.
4. Sidecar `layer_c_bench.bth.toml` (`exploration`, perf). Outcomes in order:
   - `blocked`: no path available;
   - `incomplete`: any cell below warm-up/iteration/repeat minima, or erroring;
   - `not_isolated`;
   - `ctrl_blind`: `planted_5ms − base` p50 CI excludes 0 and its point estimate lies in [4, 6] ms
     must hold, else ctrl_blind; the `aa_dup/base` ratio CI must contain 1, else ctrl_blind;
   - `pass`.

   Budget: ≤ 90 min wall. At expiry the driver aborts and writes `incomplete` (exit 0).
5. **Compatibility-claim gate (parent AC-24, AC-CG).** `$BV/compat_claim_gate.py --paths <files…>`
   greps case-insensitively for `\b(firefox|webkit|safari|edge|webgpu|chrom(e|ium)|ios|android)\b`
   co-occurring in one sentence with `\b(support(s|ed)?|compatib\w*|works?|runs?|validated|passes)\b`.
   A hit is allowed only if the sentence names an evidence file
   `outputs/browser_validation/layer_c/evidence/<route>__<browser>.json` that exists and has
   `cross_origin_isolated == true`. Scope: the T8 report text, the handoff, and any `.md` added
   or changed on the branch since the T1 commit (`git diff --name-only <T1 sha>..HEAD -- '*.md'`).
   `tests/parity/test_compat_claim_gate.py`: the planted sentence "ORT Web runs on Firefox." →
   exit 1; "ORT Web wasm runs in headless Chromium (evidence:
   outputs/browser_validation/layer_c/evidence/ort-wasm__chromium.json)." with the file present →
   exit 0; the same sentence with the file absent → exit 1. Red-check: drop the evidence-existence
   check → the third case fails.

**Files**: `$BV/bench_stats.py`, `$BV/layer_c_bench.py` (+`.bth.toml`, `.cases.json`),
`$BV/compat_claim_gate.py`, `browser/layer_c/{bench.mjs,run_bench.mjs}`,
`tests/parity/{test_bench_stats,test_xla_thread_pin,test_compat_claim_gate}.py` (create).
**Gate** (after T6): the tests + red-checks; sidecar validate/lint/`sidecar_dryrun.py`;
`local_run.sh --stem layer_c_bench --campaign "$CP" --subdir perf --npm-dir browser/layer_c $ORTW
$BATHOSW -- --artifacts-dir "$BV_ARTIFACTS"`; `check_run.sh --stem layer_c_bench --hash "$H"
--run-id "$RUN_ID" … --allowed pass`; then
`uv run --frozen --extra=dev python $BV/compat_claim_gate.py --paths outputs/browser_validation/perf/runs/layer_c_bench-${H:0:12}/report.md $(git diff --name-only <T1 sha>..HEAD -- '*.md')`
exits 0; ruff.
**Evidence that closes it**: the verified record; its report contains only claim-gated,
within-session sentences; the compat gate is green on the real deliverables and red on the plant.
**Scope estimate**: ~300 LOC JS + ~520 LOC Python.

### Task 9 (sampling track): fix the batched harness (F-S1), plumb `incremental`, budget-floor probe

1. Test first: `tests/parity/test_layer_a_sampling_batched.py`.
   - (a) Structural (`test_per_draw_ar_mask_structural`): for 3 draws with distinct orders, the per-draw bundle passed to the kernel has
     `ar_mask == generate_wave_ar_mask(waves[i], tie_group_map)` broadcast to (S, L, L). This must
     be RED on current code for i ≥ 1.
   - (b) Empirical (`test_batched_vs_unbatched_empirical`), on a synthetic L = 96 structure (and 5L33 if `$PROTEINMPNN_PATH` is present;
     never skipped, it falls back to synthetic), T = 0.1 with the X-omit column: batched draw i vs
     the unbatched `kernel` with a bundle built from `waves[i]`. Pre-registered in the test:
     mismatches ≤ 1 position per draw after the fix. Record the pre-fix counts in the commit body.
2. Fix `layer_a_sampling._vmapped_sample`/`aminx_sample_batch` to vmap over `(key, wave, ar_mask)`
   with a host-built per-draw `ar_mask`. Add `incremental: Literal["auto","off","force"] = "auto"` to
   `sample_autoregressive.kernel`, passed to `AutoregressiveConfig`; the default is unchanged.
   Unit-test that the default path is bitwise unchanged on one draw
   (`test_kernel_incremental_default_bitwise`).

   **Host predicate mirror (r1 M11).** `incremental_predicates_host(...) -> {"consistent": bool,
   "identity_frame": bool, "fits_slab": bool}` in `layer_a_sampling.py` (host NumPy) mirrors
   `autoregressive.py` exactly:
   - `wave(j)` := `decode_wave[j]`, computed as at `:616-621`: `pos_rank =
     group_first_rank[tie_group_map[0]]`, and `decode_wave = pos_rank // max_groups_per_wave` where
     `pos_rank < no_occurrence_sentinel`, else `n_waves`. Fixed/unscheduled positions therefore get
     `n_waves`;
   - `consistent` := `¬∃(i, s): reads[i, s] ∧ wave(nbr[i, s]) > wave(i)`, where `reads =
     valid_row[i] ∧ valid_nbr[i, s] ∧ ar_mask[i, nbr[i, s]] > 0.5` over the **k-NN slots**
     `nbr = enc.neighbor_indices` (`:630-636`, `:751`), not all (i, j) pairs;
   - `identity_frame` := `all(state_position_map == arange(L))` (`:752-754`);
   - `fits_slab` := `all(diff(wave_start) ≤ slab)`, with `wave_start` and `slab` exactly as at
     `:609-612` and `:623-627` (`slab = min(max(max_groups_per_wave · group_positions.shape[2],
     max_positions_per_wave or 0), L)`).

   The neighbour indices come from the same deterministic encode the kernel runs (noise 0). They
   are computed once per fixture on the host path. A draw gets `"force"` iff all three predicates
   hold. Otherwise it gets `"off"` and is counted in `n_draws_forced_off[lane]`. Tests (in
   `test_layer_a_sampling_batched.py`):
   - `test_host_predicates_match_device`: for each case, the host verdict equals a jnp evaluation
     of the same three device expressions on the real `cond`/`enc` tensors (built with
     `build_inference_bundle` + `make_encode_fn`, as the kernel does). Cases: (i) a no-fixed lane;
     (ii) **a fixed-position lane** (10 fixed positions, synthetic L = 96, and the P07 lane's fixed
     set if the manifest defines one); (iii) a planted inconsistent `ar_mask`; (iv) a non-identity
     `state_position_map`; (v) a wave larger than `slab`;
   - `test_force_equals_off_when_host_true`: whenever the host verdict is True, one draw decoded
     with `force` matches `off` (tokens exact, logits ≤ 1e-5);
   - `test_planted_inconsistent_mask_not_forced`: case (iii) gets `"off"`.

   Red-check: drop the `valid_nbr` term from the host mirror → case (ii) or (iii) mismatches the
   device.
3. `$BV/layer_a_sampling_budget_floor.py` (`--out`). It imports
   `layer_a_sampling_calibrate._budget_at_floor`, `BUDGET_WALL_HOURS_CAP`,
   `PROJECTED_PEAK_RSS_GIB_CAP` (V21) and `las.*` (no copies, no hard-coded caps). It computes the
   floor projection with the fixed harness. Controls at the largest set-B fixture (4YOW, lane
   P07@1.0):
   - per-draw cost `force` vs `off`: `fastpath_ratio = c_force / c_off` (guarded ratio);
   - `force` vs `force` A/A: `aa_ratio`.

   Records `budget_wall_hours`, `projected_peak_rss_gib`, `budget_cap_hours` and `rss_cap_gib`
   (copied from the imported constants), per-lane hours, per-draw aminx costs, `n_draws_forced_off`
   per lane, and the device kind. The reference-arm cost is reported **separately** as `c_ref` and
   `ref_arm_cost = 2·c_ref` per lane, and it is not folded into the aminx per-draw costs. The
   budget total states whether it includes `ref_arm_cost` (field `budget_includes_ref`).
4. Sidecar `layer_a_sampling_budget_floor.bth.toml` (`calibration`, campaign `aminx-bv-layer-a`).
   Outcomes in order:
   - `incomplete`: `NOT git_clean`;
   - `ctrl_blind`: `aa_ratio` ∉ [0.8, 1.25] or `aa_ratio` null;
   - `fastpath_not_engaged`: `fastpath_ratio > 0.5` or null;
   - `floor_exceeded`: `budget_wall_hours > budget_cap_hours OR projected_peak_rss_gib > rss_cap_gib`
     (fields as recorded; the script asserts they equal the imported caps);
   - `pass`.
5. Run on titanix GPU 2 via `titanix_launch.sh --gpu 2 layer_a_sampling_budget_floor "$CA"`.
   Verify with `check_run.sh --sync-remote titanix` against the synced catalog; append to
   `layer_a/run_ledger.json` per the parent O6′ step 5.
6. Append a "Post-verdict finding F-S1" note (facts plus test evidence, no re-grading) to
   `.praxia/docs/audits/260926_mpnn-reference-parity-verdict.md`.

**Files**: `$BV/layer_a_sampling.py`, `src/aminx/inference/sample_autoregressive.py` (modify);
`tests/parity/test_layer_a_sampling_batched.py`, `$BV/layer_a_sampling_budget_floor.py` (+`.bth.toml`, `.cases.json`) (create);
the verdict doc (append).
**Gate** (r1 M12):
```bash
cd "$WT" && set -euo pipefail
T9_TESTS=(
  tests/parity/test_layer_a_sampling_batched.py::test_per_draw_ar_mask_structural
  tests/parity/test_layer_a_sampling_batched.py::test_batched_vs_unbatched_empirical
  tests/parity/test_layer_a_sampling_batched.py::test_kernel_incremental_default_bitwise
  tests/parity/test_layer_a_sampling_batched.py::test_host_predicates_match_device
  tests/parity/test_layer_a_sampling_batched.py::test_force_equals_off_when_host_true
  tests/parity/test_layer_a_sampling_batched.py::test_planted_inconsistent_mask_not_forced
)
env $CAPS uv run --frozen --extra=dev --extra=benchmark pytest "${T9_TESTS[@]}" -q -rs 2>&1 | tee /tmp/t9.log
grep -L "SKIPPED" /tmp/t9.log; grep -Eq "^[0-9]+ passed" /tmp/t9.log   # pytest exit 5 (0 collected) already fails under set -e
env $CAPS uv run --frozen --extra=dev pytest tests/inference/decode/test_incremental_autoregressive.py tests/inference/test_fixed_position_self_visibility.py -q 2>&1 | tee /tmp/t9b.log   # existing AR contracts, explicit files, no -k (V27)
bth validate-sidecar $BV/layer_a_sampling_budget_floor.bth.toml && uv run --frozen --extra=dev python $BV/lint_sidecars.py
uv run --frozen --no-sync $BATHOSW python $BV/sidecar_dryrun.py --sidecar $BV/layer_a_sampling_budget_floor.bth.toml --cases $BV/layer_a_sampling_budget_floor.cases.json
# commit "T9: F-S1 fix + budget-floor (pre-registered)", then capture the launched commit:
HT=$(git rev-parse HEAD)
bash $BV/titanix_launch.sh --gpu 2 layer_a_sampling_budget_floor "$CA"   # poll per O6'
bash $BV/check_run.sh --sync-remote titanix --stem layer_a_sampling_budget_floor --hash "$HT" --sidecar $BV/layer_a_sampling_budget_floor.bth.toml --allowed pass,floor_exceeded,fastpath_not_engaged
```
`check_run.sh` counts rows with no `LIMIT`, and asserts `sidecar_sha256` is non-empty and equals
the sha of `git show $HT:<sidecar>`, plus `git_dirty = false`. Plus ruff on the modified src and
the new files. The test function names above are normative; the fixer creates exactly these.
**Evidence that closes it**: the RED→GREEN test log, the verified titanix record, and the ledger entry.
`floor_exceeded`/`fastpath_not_engaged` are findings: the track stops and T10 does not run.
**Scope estimate**: ~280 LOC.

### Task 10 (sampling track, conditional on T9 `pass`): full sampling calibrate on titanix GPU 2

1. At the T9 HEAD (or later, with no change to sampling code), run `titanix_launch.sh --gpu 2
   layer_a_sampling_calibrate "$CA" --fixture-set A`. The calibrate re-checks the floor itself
   (`:641-687`). "No change to sampling code" is checked, not asserted (r1 minor):
   `git diff --quiet "$HT"..HEAD -- scripts/browser_validation/layer_a_sampling* src/aminx/inference`
   must exit 0 before launch (`$HT` = the T9 launched commit).
2. Poll per O6′. On completion: `check_run.sh --sync-remote titanix`, ledger append. If `pass`,
   bring back `preregistered_params.json`'s sampling section and commit it (`T10: record`). The
   sampling validate is the next sprint.

**Files**: `outputs/browser_validation/layer_a/{run_ledger,preregistered_params}.json` (force-added updates).
**Gate**:
```bash
cd "$WT" && set -euo pipefail
git diff --quiet "$HT"..HEAD -- scripts/browser_validation/layer_a_sampling* src/aminx/inference
HC=$(git rev-parse HEAD)
bash $BV/titanix_launch.sh --gpu 2 layer_a_sampling_calibrate "$CA" --fixture-set A   # poll per O6'
bash $BV/check_run.sh --sync-remote titanix --stem layer_a_sampling_calibrate --hash "$HC" --sidecar $BV/layer_a_sampling_calibrate.bth.toml --allowed pass,ctrl_unsized,budget_exceeded,incomplete
```
If the run is still in flight at sprint end, the handoff records the unit name and the expected
completion; that is not a pass.
**Scope estimate**: ~20 LOC (ledger/params only).

## Risks

| Risk | Mitigation |
|---|---|
| jax2onnx 0.16.1 lacks a plugin for a primitive on the RNG-free path (gathers, `take_along_axis`, `one_hot`, nested pjit from `@jax.jit compute_backbone_coordinates`) | T2 records it verbatim → `not_converted`; the IREE route (T4) proceeds independently; no custom plugin this sprint (ODQ-B10) |
| `select_neighbors` extraction changes numerics | Bitwise test vs a frozen pre-refactor copy (T1 d); rollback: revert `features.py` |
| The W-equivalence is not bitwise (vmap-of-1 vs direct, fusion differences) | The pre-registered 1e-6 bar; the bitwise flag is recorded, not gated |
| ORT-CPU vs JAX index flips on near-tie residues at 3-decimal coords | `tie_unstable` label with per-row near-tie accounting; never relabelled `pass` |
| Detached worktree differs from `$WT` (LFS weights, node_modules, venv) | `local_run.sh` syncs its own venv, runs `npm ci`, and every script checks weight sha against `reference_pins.json` |
| `bth` warm index locked or stale | `check_run.sh` uses the parquet fallback by run id, and serialises compacts |
| F-S1 fix invalidates earlier sampling cost numbers | Earlier numbers only produced `budget_exceeded` and wrote no params, so no pre-registered sampling parameter depended on them; recorded as a post-verdict note |
| `incremental="force"` diverges from `auto` numerics | Default unchanged (bitwise test); force is used only behind the host consistency assertion; ODQ-B9 |
| Titanix GPU contention with vLLM | `--gpu 2` only; one `bv-*` unit at a time; `nvidia-smi` is not required (`titanix_run.sh`) |
| Benchmark variance (~5× run-to-run) | Single-session interleaving, 3 repeats, stratified bootstrap, A/A control; no cross-session numbers |
| ORT Web does not emit per-op profiles | Outcome `partial`, not a failure; whole-run timings still come from T8 |
| Local memory at bucket 1024 | The dense L×L distance is 4 MB f32 and edge tensors ~25 MB; `$CAPS`; heavy AR work stays on titanix |
| Native comparator silently runs dropout (V17) | D-H: `zero_dropout` + `config.inference` assertion + key-independence test (T1 g) with a red-check on a p = 0.1 model; F-D1 recorded, not fixed |
| Concurrent local runs / compacts corrupt the warm index or race the manifest | `flock /tmp/bv-local.lock` in `local_run.sh` and `check_run.sh`; single append-only manifest writer; `record` commits re-copy it |
| ORT Web thread count silently differs from the request (init-time fixing) | One context per thread setting; per-context wasm-init assertion; T5a Playwright thread test; unverifiable → no tN sentences |
| Unequal thread budgets produce misleading ratios | D-I: XLA pin verified by test, `threads` held fixed in `paired_configs`, ratios suppressed otherwise |
| `uv run --frozen --extra=dev` in `$WT` lacks torch/prody | Gates and `local_run.sh` pass `--extra=benchmark` (V18) |
| Sidecar branch unreachable or mis-ordered | `sidecar_dryrun.py` with one synthetic case per outcome, before commit and in every gate |

## Open Design Questions (recommended defaults let the sprint proceed)

| ID | Question | Recommended default | Alternatives |
|---|---|---|---|
| ODQ-B1 | Export bucket ladder | `(128, 256, 512, 1024)`: covers set-B max L = 693; library ladders untouched | add 64 (needs a cropped fixture); xtrax `1536, 2048` (V8, unmerged; defer); adopt `bucketing.py` 64..512 (refuses 4YOW) |
| ODQ-B2 | ONNX opset | jax2onnx 0.16.1 default, recorded per artifact; pin only if ORT Web 1.30 rejects it | pin the highest opset ORT Web 1.30 supports up-front |
| ODQ-B3 | WebGPU EP | Out: capability probe only, no numbers or claims | separate WebGPU task (int64 TopK fallback risk, parent ODQ-1) |
| ODQ-B4 | Browser matrix | Headless Chromium only (Firefox/WebKit not installed, V12); a deviation from parent ODQ-8 with no claims for the others | `npx playwright install firefox webkit` (+ system deps) in a follow-up; headful Chromium if WSLg is available |
| ODQ-B5 | IREE route | In for layer b (native parity + WASM32 `CODEGEN_ONLY`); out for layer c | defer entirely |
| ODQ-B6 | Wrapper home | `src/aminx/export/` (library, importable by a future app) | `scripts/browser_validation/` |
| ODQ-B7 | k-NN selection reuse | Extract `select_neighbors` from `features.py:208-229` (behaviour-preserving) | wrapper-local copy (violates import-not-reimplement); add a `coordinate_noise=False` flag to `forward_edge_stages` and `Aminx.__call__` (wider change, touches `mpnn.py:147-148` key default) |
| ODQ-B8 | Layer-(c) float bars vs ORT-CPU | log-probs ≤ 1e-5, edge ≤ 2e-6 (bar/10) | reuse the layer-(b) bars |
| ODQ-B9 | Sampling harness decode path | `incremental="force"` behind the host consistency assertion (D-F; `auto` under vmap runs both branches, V16) | keep `auto` and accept no speedup; restructure to an unbatched loop |
| ODQ-B10 | Unsupported jax2onnx primitive | Record and stop for that path; file an xtrax/aminx item; IREE continues | write a jax2onnx plugin this sprint |
| ODQ-B11 | `ProbeRecord.n_atoms` semantics for aminx | `n_atoms := bucket L` (residues), stated in `config["n_atoms_semantics"]` | real residue count; 4·L backbone atoms |
| ODQ-B12 | Native-arm threading in the benchmark | (r1) Pin XLA:CPU to 1 thread (D-I, flag to-verify by T8 step 0) and pair it only with `ort_wasm_t1`; tN gets no native ratio; unverifiable pin → no native ratio | pin native to N threads to match tN (needs a verified N-thread XLA flag); disjoint core sets via `taskset` |

## Findings and proposed debt (recorded, not fixed this sprint)

- **F-D1 (r1 B3): `Aminx.__call__` default-dropout footgun.** `Aminx.__call__` takes `inference`
  and ignores it (`mpnn.py:120`, `:141`). It injects `PRNGKey(0)` when no key is given
  (`:147-148`) and calls the encoder without `inference` (`:165-171`). The encoder then defaults to
  `inference=False` (`encoder.py:332`), and `Dropout` honours the explicit flag over its own
  `.inference` (`dropout.py:45-46`). The unconditional fallback decoder likewise drops
  `config.inference` (`unconditional.py:100-112`). Any caller that expects `inference=True`,
  `config.inference=True` or `eqx.nn.inference_mode` to disable dropout gets dropout whenever
  `Dropout.p > 0`.
  **Proposed debt item** (draft text goes in the T1 commit body and the handoff; the orchestrator
  files it): "aminx: `Aminx.__call__` must forward `inference` to `self.encoder` (and the
  unconditional fallback decoder must forward `config.inference`) instead of ignoring it; add a
  key-independence test at `inference=True` with `p > 0`; audit callers relying on the current
  behaviour. This changes scoring numerics, so it is gated on the parity check that
  `unconditional.py:101-105` already references (spec 260709 §4.1)."
- **F-D1b (conditional):** raised only if the loaded `proteinmpnn_v_48_020` has `max_p_before > 0`
  (T1/T3a record it). It means the Phase-1 P04 native arm ran with dropout keyed on `PRNGKey(0)`.
  It is reported to the verdict owner with no re-grading here.
- **F-S1:** as V15 / T9.

## References

- Parent epic: `.praxia/docs/specs/260923_aminx-browser-validation.md` (Epic phases; Inference-path
  inventory; Sampling contract; Phase 2/3 plans; AC-17..24; ODQ-1/5/8/10; common context incl. O6′).
- Phase-1 verdict: `.praxia/docs/audits/260926_mpnn-reference-parity-verdict.md` (D10, D11);
  `outputs/browser_validation/layer_a/advance_table.json`; `scripts/browser_validation/advance_check.py`.
- Phase 0: `outputs/browser_validation/phase0/{export_safety_census,jax2onnx_spike,ort_web_smoke}.json`.
- Incremental AR: `src/aminx/inference/decode/autoregressive.py:313,596-602,743-759`;
  `scripts/benchmarks/bench_ar_incremental.py`; bathos run `b5f7ade5`.
- Debt: #1981 (loop-body complexity guards), #1982 (incremental decode fallbacks, legacy `sample.py`
  ar_mask convention), xtrax #1983 (profiling should flag loop bodies whose per-iteration cost scales
  with extent).
- Rules: `~/.claude/rules/BATHOS.md`, `~/.claude/rules/local-compute-limits.md`, repo `CLAUDE.md`.
