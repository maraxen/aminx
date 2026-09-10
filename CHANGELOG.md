# Changelog

## 0.2.0a1 (2026-09-10)

**Minor bump, not another `0.1.0a` alpha.** Two things in this release change what a run writes
to disk, and one of them introduces new hard failures on input that previously "worked". A reader
of an aminx store cannot assume a `0.2.0a1` store is shaped like a `0.1.0a28` one, so the version
says so. Still an alpha — the suffix is retained deliberately; nothing here claims the project has
left alpha.

### Added

- **The sink is now self-describing about logit semantics, and about what produced it**
  (backlog #4149, PR #154). `aminx.inference.decode.autoregressive` computes two logit arrays per
  wave — `stored_logits` (fused with a **zero** bias) and `sampling_logits` (fused with
  `cond.bias`, and the only one that reaches `jax.random.categorical`). Every sink writes the
  bias-free array as `"logits"`. That divergence is deliberate and correct, but nothing on disk
  recorded it, so a consumer had to already know — and the moment a nonzero bias is introduced,
  the stored array silently stops describing the distribution that was sampled from.

  New module `aminx.io.sink_provenance`, consumed by all three writers (`host/streaming.py`,
  `sampling/multistate_poe.py`, `io/designs.py`). New optional attrs — absent means unknown,
  since aminx has no reader for these stores yet:

  | attr | meaning |
  |---|---|
  | `logits_bias_semantics` | which array is stored, whether a bias was nonzero, whether it was persisted, and **which group** holds it |
  | `prng_seed` | the run seed |
  | `grid_job_seed_hash` | the other half of a grid-mode run's sampling key |
  | `aminx_version` | resolved via installed-distribution metadata, never `aminx.__version__` |
  | `sink_provenance_version` | hash-free marker letting a reader tell a store carries these fields |

  The raw pre-jit bias is persisted as a `bias` array when nonzero, so `sampling_logits` can be
  reconstructed from `stored_logits` without re-deriving it. **No `@jax.jit` signature changed** —
  the tag is derived host-side from the raw bias.

  `DesignZarrWriter` gains an opt-in `aminx_version=` override for bare-source and vendored
  imports, which have no `.dist-info` and therefore cannot resolve a wheel version.
  (`src/aminx/io/sink_provenance.py`, `src/aminx/host/streaming.py`,
  `src/aminx/sampling/multistate_poe.py`, `src/aminx/io/designs.py`)

### Changed

- **`xtrax[io]` `0.4.0a5` → `0.4.0a9`, crossing a breaking change** (PR #153). xtrax **a7** made
  `SinkSpec.run_id` a required constructor argument; aminx was pinned at `==0.4.0a5` and had never
  crossed it. Four construction sites were affected — CI caught two, and `host/streaming.py` and
  `host/runner.py` were equally broken but uncovered by tests. The three sites with a `RunSpec` in
  scope now go through the documented canonical seam `xtrax.run.derive_sink_spec`;
  `DesignZarrWriter`, which has no run context, takes an optional `run_id` defaulting to
  `new_run_id()` (optional rather than minted inline because `ZarrStagingSink` **raises** when a
  store on disk carries a different `run_id`, so a caller reopening a store must be able to supply
  the matching id).

  **This does not link provenance yet.** `derive_sink_spec`'s run-id precedence is explicit
  `run_id=`, then `run_spec.run_id`, then a fresh `new_run_id()` — and `build_run_spec` never
  populates `run_id`, so every call still falls through to a fresh unlinked id. Routing through the
  seam puts the decision in one place so populating `RunSpec.run_id` later fixes all three sites at
  once; what `run_id` should derive from is deliberately left open.
  (`pyproject.toml`, `uv.lock`, `src/aminx/host/streaming.py`, `src/aminx/host/runner.py`,
  `src/aminx/sampling/multistate_poe.py`, `src/aminx/io/designs.py`)

- **Two new hard failures, replacing silently-wrong output.** Both refuse to write a provenance
  claim that would be false rather than writing it:
  - `_sample_streaming` and `sample_multistate_poe_campaign_row` raise `NotImplementedError` when
    `return_logits=True` and `sampling_strategy != "temperature"`. Decode mode is a pure function
    of that field (`host/plan.py::resolve_decode_mode`: `"straight_through"` → `STEMode` →
    `decode/ste.py`), and `logits_bias_semantics` describes a split only the autoregressive path
    produces. `SamplingSpecification.__post_init__` already rejected this combination in **grid**
    mode, but both writers also serve non-grid runs, where it was legal.
  - `DesignZarrWriter.write()` raises `ValueError` on a `logits_bias_semantics` dict claiming
    `bias_persisted: True`. That writer stages no `bias` array and has no root group, so the claim
    can never be satisfied — and `logits_bias_semantics_outputs` returns it **by default**, so a
    caller doing the obvious thing would have written a lying attr.

### Bug Fixes

- **A metadata constant was load-bearing for randomness.** `_grid_job_seed_hash` hashed a payload
  keyed on `GRID_SCHEMA_VERSION`, and that digest folds into `_base_sampling_key`'s
  `jax.random.fold_in` chain — so bumping a *schema label* silently changed the sampled output of
  every grid-mode run (different tokens, and different logits at later AR positions, since decoded
  tokens feed forward). **This fired for real during development:** a bump to `grid_v2` made to
  satisfy one acceptance criterion violated another (`stored logit values bit-identical`) in the
  same commit, caught by audit rather than by any test. Fixed with `_SEED_HASH_SCHEMA_PIN`, a
  dedicated constant frozen forever and decoupled from the schema label. It is retained even though
  both values currently read `"grid_v1"` again — the point is the decoupling, not the coincidence.

  `GRID_SCHEMA_VERSION` also feeds `_grid_manifest_row_hash` → a campaign manifest row's **output
  path** → `_done_marker_path`, so bumping it makes `_read_done_marker` return `None` for every
  already-completed row on resume: a silent full recompute of an in-progress campaign, with the old
  store orphaned on disk. That coupling is **not** fixed here; the schema bump was withdrawn
  instead, and the hash-free `sink_provenance_version` carries the reader-facing intent.
  `GRID_SCHEMA_VERSION` should be treated as unbumpable without a migration plan.
  (`src/aminx/host/_sampling_grid_lineage.py`)

- **The persisted `bias` array was silently dropped in grid mode.** Both sampling writers built
  `root_arrays` via `.update(bias_arrays)` and then, only when `grid_lineage is not None`,
  *reassigned* `root_arrays = {...}` instead of updating it — discarding the staged array while
  `logits_bias_semantics["bias_persisted"]` still claimed `True`. A lying attr, worse than a
  missing one. (`src/aminx/host/streaming.py`, `src/aminx/sampling/multistate_poe.py`)

## Undocumented (0.1.0a7 – 0.1.0a28)

> **These entries were filed under a heading that read `Unreleased`, but they are not unreleased.**
> The section accumulated from 0.1.0a7 onward without ever being cut into per-version sections,
> while a7–a28 were in fact released. At least the Zarr storage migration and the `xtrax.tiling`
> decode-path dispatch described below shipped in `0.1.0a28` (they are present at tag `v0.1.0a28`).
> The heading is corrected here rather than renamed onto `0.2.0a1`, which would have mis-attributed
> roughly a year of already-released work to this release. No attempt is made to retroactively
> split these entries across a7–a28 — the per-version attribution was not recorded at the time and
> is not reliably recoverable. Only `v0.1.0a1`–`v0.1.0a7` and `v0.1.0a28` are tagged.

### Added

- **Decode-path dispatch now runs through `xtrax.tiling`** (EPIC #1541 P3): `make_decode_fn`
  (`ConditionalMode`/`UnconditionalMode`/`AutoregressiveMode`) resolves its state-axis strategy
  via `make_axis_dispatch_via_xtrax`, and the sampling planner (`make_sampling_planner`)
  delegates strategy selection to `xtrax.tiling.BatchPlanner`'s joint-budget mode
  (`xtrax==0.4.0a1`), replacing aminx's own hand-rolled greedy demotion loop.
  (`src/aminx/inference/decode/factory.py`, `src/aminx/host/plan.py`)

  *Closes the loop noted in 0.1.0a6: `PlannerTopology`'s planned `xtrax.ExecutionProfile`
  field awaited xtrax reaching multi-phase `BatchPlanner` parity (T2.5) — that parity is now
  gated and passed (see below).*

- **Sampling/jacobian streaming output migrated from HDF5/ArrayRecord/`.npz` onto Zarr**
  (`xtrax[io]==0.4.0a4`): `host/streaming.py`'s two write paths (the deprecated HDF5 path and
  the `use_arrayrecord=True` ArrayRecord path) are unified into one Zarr-backed path built on
  `xtrax.run.ZarrStagingSink`; `host/runner.py`'s jacobian runner writes each structure's
  jacobian into its own keyed Zarr group instead of one `np.savez_compressed` dump at the end
  (also removes a latent bug: the old dump required uniform jacobian shapes across all
  structures via `np.stack`, which fails for variable-length inputs — per-key Zarr groups have
  no such constraint). `io/designs.py`'s `DesignArrayRecordWriter` becomes `DesignZarrWriter`,
  same validation/dtype contract (sequence uint8, logits float16, scores/state_weights
  float32), now storing per-design metadata in the Zarr group's `.attrs` instead of a raw JSON
  byte suffix. `output_h5_path` (field name kept, semantics updated) now points at a Zarr
  store directory rather than a single file.

  Known simplifications from this migration, not full parity with the code it replaces:
  `DesignZarrWriter` drops the old writer's async thread-pool (writes are now synchronous);
  campaign-mode sampling accumulates a structure's sample-chunks in memory *within one batch*
  before staging (bounded by one batch's dispatch size, not the whole campaign) rather than
  the old HDF5 path's true per-chunk incremental resize-writes — `xtrax.run.ZarrStagingSink`
  would need an append-mode extension to restore that if a campaign's per-batch memory
  footprint proves too large in practice.

  `SamplingSpecification.use_arrayrecord` and its CLI flag are removed (the format is no
  longer a caller choice). `campaign.py`'s own HDF5-based lock/done-marker/content-verification
  machinery is explicitly OUT of scope for this migration — it's real distributed-systems
  infrastructure (retry/resume correctness across likely-SLURM-array campaign jobs), tracked
  separately (backlog #3182).
  (`src/aminx/host/streaming.py`, `src/aminx/host/runner.py`, `src/aminx/io/designs.py`,
  `src/aminx/inference/optimize_ste.py`, `src/aminx/run/specs.py`, `src/aminx/run/spec.py`,
  `src/aminx/cli.py`, `tests/io/test_designs.py`)

### Bug Fixes

- **Every autoregressive mask set its diagonal, so each position read its own slot — and
  that slot asserted ALANINE, not "undrawn". Numerical output of every AR sampling run
  changes.**
  ([`src/aminx/utils/autoregression.py`](src/aminx/utils/autoregression.py),
  [`src/aminx/inference/decode/autoregressive.py`](src/aminx/inference/decode/autoregressive.py))

  `generate_ar_mask`'s untied branch was `row_indices >= col_indices` — non-strict — and
  `generate_wave_ar_mask` returns `earlier_wave | (same_wave & same_group)`, trivially true
  when `i == j`. Both therefore set `ar_mask[i, i] = 1`. The decoder gathers this mask into
  `attention_mask` to gate the *sequence* edge features (`model/decoder.py:144-146`), and a
  residue is always among its own KNN neighbours (self-distance 0 is the global minimum), so
  the diagonal is a live self-loop, not an inert one.

  The rationale for keeping it was that a not-yet-drawn position's own slot holds a harmless
  placeholder. That premise was false twice over. First, `decode/autoregressive.py`
  initialized undrawn slots to token index **0**, and `MPNN_ALPHABET` is
  `"ACDEFGHIKLMNPQRSTVWYX"` — index 0 is **alanine**, while the unknown token `X` is index 20
  — and `model/decoder.py:124` embeds it as `one_hot_sequence @ w_s_weight`, a real nonzero
  row of a pretrained matrix. Each position was being told "I am alanine" about itself at the
  exact step it decided its own identity. Second, the placeholder only exists at the wave a
  position is drawn: `_decode_one_step` rebuilds from the encoder output every wave with no
  carried hidden state, so from the following wave onward the slot holds a real token.

  Both reference implementations disagree with what we were doing, verified against primary
  source: stock LigandMPNN builds `1 - torch.triu(torch.ones(L, L))` at
  `model_utils.py:227-231` and four further call sites (`torch.triu` defaults to
  `diagonal=0`, so this is strictly lower-triangular), and ColabDesign builds
  `jnp.tri(L, k=-1)` at `colabdesign/mpnn/utils.py:19-26`. Both are self-excluding under any
  permutation.

  Our own parity suite never caught it: `tests/parity/test_full_model_parity.py:90-95`
  hand-builds its `ar_mask` with a **strict** comparison and passes it explicitly, so the
  `ar_mask is None` branch never runs and `generate_ar_mask` — the function the production
  sampling path calls — was never invoked there at all.

  Both constructors are now self-excluding, and the undrawn sentinel is `-1`
  (`UNDRAWN_TOKEN`), whose one-hot is the zero vector, reproducing the reference's
  `h_S = torch.zeros_like(h_V)`. Self-exclusion alone already makes undrawn slots
  unreachable, so the sentinel is defence in depth — and it makes "not yet drawn"
  distinguishable from "alanine" in a stored or mid-decode sequence, which index 0 could not
  express.

  **Impact.** Measured on TEV protease with a fixed 25-residue active-site shell: mean JSD
  **0.0296** over designable positions between the old and new masks, with an
  alanine-specific probability shift of **−0.0049** against a mean of 0.0016 across the other
  twenty tokens. The score path is unaffected (it already used the diagonal-free
  `full_context_ar_mask`). There is no opt-out: the old behaviour was not a configuration.

  **Deliberately NOT changed**, both flagged for separate work: same-tie-group positions
  remain mutually visible (whether that matches the reference's *tied* decoding is unverified,
  and it is not the diagonal); and `generate_ar_mask` consumes a **rank** array
  (`rank[i]` = step of position `i`) while the parity fixture treats the same argument as an
  **order** array (`order[k]` = position at step `k`) — a live convention inconsistency that
  is invisible for uniform random permutations and is now pinned by a test.

- **`SamplingSpecification` had no `weight_profile`/`fixed_group` fields, so any manifest
  row carrying them in its nested `sampling_spec` hard-crashed real sampling**
  ([`src/aminx/run/specs.py`](src/aminx/run/specs.py))

  `run_manifest_row`'s `SamplingSpecification(**worker_payload)` construction
  (`host/campaign.py`) is deliberately strict about unknown keys — an audit safety
  mechanism, not something to weaken. A manifest-building caller (tev_design's necklace
  campaign) writes `weight_profile`/`fixed_group` provenance labels into a row's nested
  `sampling_spec` for anti-mislabel validation (checking the executed output against
  manifest intent) — a pattern already established for other CLI-flag-less fields
  (`batch_size`, `state_weights`, `multi_state_strategy`). But these two were never added
  as real fields, so any freshly-built manifest carrying them crashed real sampling with
  `TypeError: unexpected keyword argument 'weight_profile'`. Undetected until now because
  the L1 static validator that manifest builders typically gate on reads raw JSON and
  never constructs `SamplingSpecification` — only a real sampling call exercised this gap,
  and no one had re-run one against a freshly-built manifest since these labels were added.

  Fix: both are now real, first-class (if sampling-inert) `str | None = None` fields,
  matching the same pattern as `job_id`. See `tests/run/test_specs.py`'s
  `TestSamplingSpecificationValidation.test_weight_profile_and_fixed_group_are_real_fields`
  for the regression coverage.

- **`n_samples` axis planning was silently decoupled from the actual runtime sample count**
  ([`src/aminx/host/plan.py`](src/aminx/host/plan.py),
  [`src/aminx/host/kernel_dispatch.py`](src/aminx/host/kernel_dispatch.py))

  `make_sampling_planner`'s `N_SAMPLES` axis cardinality was computed from
  `SamplingSpecification.samples_batch_size` (a static default, 16), while the actual dispatched
  sample count was resolved independently via `resolve_target_samples` — two unrelated fields,
  no cross-validation. When the plan decided Vmap because the small default fit the memory budget,
  that decision was silently applied to the real (possibly far larger) array with no re-check,
  in both the default (`_dispatch_axis`, no fallback at all) and legacy (`safe_map`'s defeatable
  `batch_size==0` branch) dispatch paths.

  Fix: `make_sampling_planner` gained an optional `n_samples_override` parameter; `_sample_batch`
  now resolves the real per-call sample count first and passes it through, so the planner's
  decision is verified against the array size that's actually dispatched.

  Introduced in `5ca2abf` (2026-05-07); see
  `.praxia/docs/specs/260706_samples-axis-planner-cardinality-mismatch.md` for the full root-cause
  history and `tests/host/test_samples_cardinality_fix.py` for the regression coverage.

- **State axis in multistate PoE sampling hardcoded `Vmap`, bypassing `BatchPlanner` entirely**
  ([`src/aminx/inference/sample_autoregressive.py`](src/aminx/inference/sample_autoregressive.py),
  [`src/aminx/sampling/multistate_poe.py`](src/aminx/sampling/multistate_poe.py))

  `sample_autoregressive.kernel()` always passed `strategy=Vmap()` for the state axis to
  `make_decode_fn`, regardless of `bundle.geometry.n_states` or the device memory budget —
  invisible to any BatchPlanner accounting, even though `aminx.tiling.axes.N_STATES` already
  declares the canonical convention (`default_batch_size=1`, i.e. SafeMap-one-state-at-a-time
  whenever `num_states>1`). Harmless at `num_states=1` (the only case the single-structure
  campaign path hits), but for a genuinely fused multi-state bundle (`sample_states_fused`)
  this batches every decoder layer's per-state MLPs simultaneously. At production sample
  counts this produced a single fused GEMM XLA's autotuner could not find a valid kernel
  config for: `sample_count=128` crashed after an 809s compile ("Autotuning failed for HLO:
  `f32[128,12582912]{1,0}` fusion(...)"), `sample_count=512` failed differently ("9 out of 89
  instructions"). Found investigating tev_design's necklace PoE production-scaling
  bottleneck (praxia debt #942).

  Fix: `kernel()` gained an optional `state_strategy` parameter (`None` preserves the prior
  Vmap default for the single-state path); `sample_states_fused` now resolves the state axis
  through `BatchPlanner` against `N_STATES` (a plain cardinality-vs-`default_batch_size` rule,
  deliberately not a memory-estimator/budget call — an optimistic byte estimate is exactly
  what let this bug go unnoticed) and passes the resolved strategy through, translating
  xtrax-native decisions back to aminx's own `AxisStrategy` union the same way
  `_plan_axis_strategy` already does for the samples axis.

  See `tests/sampling/test_multistate_poe.py`'s
  `test_multistate_state_axis_resolved_via_batchplanner_not_hardcoded_vmap` and
  `test_single_state_still_passes_an_explicit_resolved_strategy` for the regression coverage.

  **Follow-up (same investigation):** fixing the state axis alone wasn't sufficient —
  `sample_count=128`/`512` still failed after this fix (differently: "Failed to get configs
  for: N out of M instructions", not the original 809s-compile blowup), because the samples
  axis's own memory estimate (`activation_bytes_per_element` in `sample_states_fused`) was
  *also* a hand-typed linear formula (`num_states * seq_len * a hidden-dim-sized magic
  constant`) that missed the AR mask's quadratic `(L, L)` term and the `(L, K, D)`
  edge-feature term entirely. Replaced with `xtrax.tiling.estimators.
  lowered_memory_estimate` — lowers+compiles a ONE-sample representative tile (state axis
  already resolved) and reads XLA's own buffer-assignment numbers, a real measurement
  instead of a formula that's now been shown to drift out of sync with reality twice in the
  same function. This primitive already existed in xtrax with zero call sites anywhere in
  aminx before this (praxia debt #945 tracks auditing every other hand-typed estimate
  in aminx for the same migration).

### Changed

- **`xtrax` pin bumped `0.4.0a1` → `0.4.0a2`** (`pyproject.toml`): picks up xtrax's
  `StageBundle` validator fix (PEP 563 annotation resolution, structural-callable `Protocol`
  acceptance, N-way union support) — no breaking changes to aminx's existing xtrax usage.
  Transitively bumps `jax`/`jaxlib` to `0.10.2` (xtrax's new floor). Unblocks 7 of `StageSet`'s
  10 fields for a future `StageBundle` adoption attempt; the remaining 3 container-shaped
  fields (`encoder_sink`, `decoder_sink`, `axis_boundaries`) still need backlog #3155's design
  work regardless of this bump.

### Gates

- **T2.GATE** (dispatch-layer parity, R1 DoD): bit-for-bit golden fixture, identical JIT-recompile
  count, and cluster GPU throughput within 0.2% (production shape L=208, TEV protease) all pass —
  see `tests/tiling/test_t2_gate_bitforbit_golden.py`,
  `scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py`.
- **T-PLANNER.GATE** (planner-layer parity): old (retired) and new joint-budget planner decisions
  match across representative demotion/budget scenarios, including the one deliberate behavior
  change this migration introduces (see Breaking Changes) — see
  `tests/host/test_t_planner_gate_parity.py`.

### Breaking Changes

- **Sampling/jacobian streaming output format is now Zarr, not HDF5/ArrayRecord/`.npz`** —
  `SamplingSpecification.use_arrayrecord` and its `--use-arrayrecord` CLI flag are removed.
  Existing `.h5`/`.arrayrecord`/`.npz` outputs from prior runs are not migrated; downstream
  readers need to move to Zarr's array/group API. See the Added entry above for the full
  scope. (`src/aminx/host/streaming.py`, `src/aminx/host/runner.py`, `src/aminx/io/designs.py`)

- **`host/campaign.py`'s manifest-row lock/done-marker/content-verification now targets Zarr
  stores, not HDF5 files** (`DONE_MARKER_SCHEMA_VERSION` bumped `campaign_done_marker_v1` →
  `v2`): completes the migration above for the campaign-orchestration path. The lock layer
  (lease-based, compare-and-swap stale-lock recovery), path-naming helpers, and atomic
  promotion (`Path.replace()`) needed no changes — verified empirically that directory-to-
  directory rename is atomic on POSIX, same as for files. What changed: `_h5_content_digest`
  → `_zarr_content_digest` (same hashing primitives, walks `zarr.Group`/`zarr.Array` instead
  of `h5py.Group`/`h5py.Dataset`); `_fsync_file` → `_fsync_tree` (a Zarr store is a directory
  of many chunk files, not one — durability requires recursively fsyncing all of them before
  trusting a content digest); the whole-file SHA256 (`artifact_sha256`) is dropped entirely —
  redundant with the semantic content digest and has no clean analog for a directory. Old
  `v1` done markers (from HDF5-era campaigns) correctly fail schema-mismatch validation
  instead of being silently misinterpreted; no migration path, matching this project's
  existing clean-break convention. Scoped in backlog #3182 (chose a direct semantic-digest
  port over a spot-check optimization — the existing HDF5 path already does a full content
  walk on every verification, so a full walk for Zarr is behavior-preserving, not a
  regression; sampling-based verification would be a genuine guarantee weakening not
  justified without a demonstrated performance problem).
  (`src/aminx/host/campaign.py`)

- **`make_sampling_planner` raises on an infeasible memory budget** instead of silently returning
  a plan that exceeds it. Previously, if no combination of Vmap/SafeMap demotions fit the budget,
  the planner returned a `BatchPlan` with `budget_exceeded=True` that callers could inspect (and
  which nothing in production actually did). It now raises `PlanBudgetInfeasibleError` (a
  `TilingError` subclass) instead. No production caller was found relying on the silent path, but
  this is a new exception type in `make_sampling_planner`'s call chain.
  (`src/aminx/host/plan.py`)

### Removed

- **`aminx.tiling.planner`'s local `BatchPlanner`/`AxisSpec`/`AxisDecision`/`BatchPlan`**,
  `aminx.tiling.carry`, `aminx.tiling.dedup`, and `aminx.tiling.carry_shape` — retired once their
  xtrax equivalents passed the parity gates above. `aminx.tiling.planner` now holds only
  `estimate_memory_theoretical`, the one piece of the old planner with no xtrax equivalent
  (invoked through xtrax's engine now, math unchanged). `aminx.tiling`'s `axes.py`, `bucketing.py`,
  `pad.py`, `strategy.py`, `dispatch.py`, and `errors.py` remain — see
  `.praxia/docs/decisions/260706_bucketing-pad-stay-local-epic-1541-p3-scope-closed.md` for why
  those specifically stay.

- **`RunSpec.tied`/`.batching`/`.averaging` sub-configs** (`TiedPositionsConfig`,
  `BatchingConfig`, `AveragingConfig` — 18 fields total): write-only scaffolding from the RS-1
  migration that was never finished. `build_run_spec()` populated these on every call but nothing
  downstream ever read them — all consumers (`host/kernel_dispatch.py`,
  `host/_sampling_grid_lineage.py`, etc.) read the equivalent flat `SamplingSpecification` field
  instead. Removing them doesn't change behavior; the flat fields they duplicated are untouched.
  Scoped in `.praxia/docs/specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md`
  (backlog #3158); `GridLineageConfig` and `LigandConfig` were NOT removed — each has one live
  field (`grid_mode`, `model_family`) plus existing partial-migration fallback logic worth
  finishing rather than discarding.
  (`src/aminx/run/spec.py`, `src/aminx/run/run_spec_portable_json.py`)

## 0.1.0a6 (2026-06-14)

### Added

- **`PlannerTopology` sub-config in `RunSpec`** (RS-2): New `eqx.Module` sub-config wrapping
  aminx kernel dispatch topology. `RunSpec.plan` carries a `PlannerTopology` with a single
  field `use_unified_driver: bool` (default `True`, consistent with RS-5 fix in 0.1.0a5).
  Module-level `topology_hash(plan)` produces a deterministic 16-char hex digest for
  cache-key derivation.
  ([`src/aminx/run/spec.py`](src/aminx/run/spec.py))

  *Note:* `PlannerTopology` will gain an `xtrax.ExecutionProfile` field once xtrax reaches
  multi-phase `BatchPlanner` parity (T2.5).

### Performance

- **`PoeModel.__call__`**: Replaced Python `for i in range(self.n_backbones)` loop with
  `eqx.filter_vmap`, matching the pattern already used in `infer_all_params`. Backbone
  inference is now fully vectorized via JAX rather than traced sequentially at Python level.
  ([`src/aminx/potts/poe.py`](src/aminx/potts/poe.py))

- **`PoeModel.joint_energy`**: Replaced Python `for h, j, w in params_list` loop with
  `jax.vmap(PottsModel.log_prob, in_axes=(None, 0, 0, 0))` + `jnp.sum` over stacked params.
  ([`src/aminx/potts/poe.py`](src/aminx/potts/poe.py))

- **`_parallel_tempering_exchange`**: Replaced Python `for parity / while i` loops with
  `jax.vmap` over non-overlapping replica-pair edges within each parity group. Even/odd
  parity passes remain sequential (odd uses seqs updated by even). Keys split once per
  parity group; results scattered back via `.at[...].set`.
  ([`src/aminx/potts/sampling.py`](src/aminx/potts/sampling.py))

### Gates

- **G1 training parity gate**: All three criteria pass — pytest suite (8/8), checkpoint
  round-trip smoke (`ResumableState` save → load, all leaves match to atol=1e-7), and
  50-step overfit smoke (loss 3.14 → 0.00 over 50 steps).

### Breaking Changes

- **Checkpoint format**: Adopted `xtrax.checkpoint.orbax` single-PyTree checkpointing via
  `PyTreeCheckpointHandler`, replacing the legacy `ocp.args.Composite` format. Checkpoints
  saved with the old format are **NOT compatible** with this version. Delete existing
  checkpoint directories before resuming training.
  ([`src/aminx/training/checkpoint.py`](src/aminx/training/checkpoint.py),
  [`src/aminx/training/trainer.py`](src/aminx/training/trainer.py))

  The `ResumableState` (params, optimizer state, step, RNG, and extras) is now written as
  a single flat PyTree, improving checkpoint composability and enabling future multi-device
  training. The metrics dict (if any) lives in `ResumableState.extras`, not as a top-level
  checkpoint key.

## 0.1.0a5 (2026-06-10)

### Breaking Changes

- **API rename**: `xyz_37` → `atom_37` and `xyz_37_m` → `atom_37_mask` across the
  side-chain atom-context API — the public `GeometryBundle` fields,
  `build_inference_bundle(...)` / kernel keyword arguments, and `model.features(...)`
  parameters. The new names describe the 37-atom representation and its validity mask
  clearly. No deprecated alias (pre-release clean break).

### Bug Fixes

- **`ProteinFeaturesLigand._make_angle_features`**: correct the residue-frame projection
  einsum ([`src/aminx/model/ligand_features.py`](src/aminx/model/ligand_features.py))

  The projection used `jnp.einsum("lqp, lym -> lyp", R_residue, diff)`. Because `q` and
  `m` each appear in only one operand and not in the output, einsum summed both
  independently — `(Σ_q R[l,q,p])·(Σ_m diff[l,y,m])`, an outer product of column-sums
  rather than the frame projection `e_p·diff`. Corrected to `"lqp, lyq -> lyp"`.

  This was the root cause of the side-chain-context cross-framework divergence (~0.85
  Pearson vs ~0.9998 baseline vs the LigandMPNN reference). It only surfaced with side
  chains ON: the no-side-chain baseline's dummy ligand is fully masked, so the corrupted
  node features never contributed.

### Tests

- Add side-chain-context logits parity test vs the PyTorch LigandMPNN reference, and
  migrate the tied-autoregressive / multistate side-chain tests to the current bundle
  API (tie groups via `build_inference_bundle(tie_group_map=...)`; side-chain context
  packaged onto `GeometryBundle` rather than loose kernel kwargs).
- Add `tests/model/test_unconditional_sidechain_bundle.py` covering the
  `build_inference_bundle` → `score_unconditional.kernel` side-chain path: shape
  normalization for 3-D and 4-D `atom_37` inputs, logits sensitivity to `atom_37` when
  `use_side_chains=True` (requires `fixed_mask=1` so fixed residues contribute context),
  logits invariance when `use_side_chains=False`, and `ValueError` guard for missing
  `atom_37` with a side-chain model.

## 0.1.0a4 (2026-06-09)

### Bug Fixes

- **`ProteinFeaturesLigand`**: fix `top_k` crash when ligand atom count < `atom_context_num`
  ([`src/aminx/model/ligand_features.py`](src/aminx/model/ligand_features.py))

  The 0.1.0a3 fix moved `top_k` (A → `atom_context_num=16`) outside the `use_side_chains`
  guard, but did not account for dummy ligand inputs (`with_ligand=False`) where A=1.
  `jax.lax.top_k` raises `ValueError: k argument to top_k must be no larger than size along
  axis` when k=16 > A=1.

  Fix: clamp k to `min(atom_context_num, A)`, matching the existing pattern for the protein
  graph at `k = min(self.k_neighbors, Ca.shape[0])` (line 303).

## 0.1.0a3 (2026-06-09)

### Bug Fixes

- **`ProteinFeaturesLigand`**: fix OOM on large ligand atom counts when `use_side_chains=False`
  ([`src/aminx/model/ligand_features.py`](src/aminx/model/ligand_features.py))

  The `top_k` atom selection (A → `atom_context_num=16`) was only applied inside the
  `use_side_chains` branch. Without sidechain mode the full `A=155` atoms flowed into
  `_y_edges_coords_to_embed`, whose output buffer is pre-allocated at
  `(L, A, A, node_features)`. With flat-multistate inputs (`L≈2048`, `A=155`) this
  allocates `≈20 GiB` before any other live buffers, causing
  `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.63 GiB` on all GPU
  tiers including H200.

  Fix: move the `top_k` selection outside the `use_side_chains` guard so it always runs.

- **`pyproject.toml`**: bump `proxide>=0.1.0a8`

## 0.1.0a2

- Initial public alpha — Sprint 2 inference API (`build_inference_bundle`,
  `score_unconditional`, `score_conditional`), flat-multistate support, ligand chunking.

## 0.1.0a1

- Initial release.
