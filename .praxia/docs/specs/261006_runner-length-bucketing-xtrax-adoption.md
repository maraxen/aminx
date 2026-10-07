---
title: "S8: default-on runner length bucketing on the xtrax ladder, plus xtrax-native hygiene and adoptions"
description: The core runner trims each batch from the loader's max_length to the smallest xtrax BUCKET_LADDER rung that covers it, then re-pads the outputs. On by default, with length_bucketing=False to opt out. Also retires the private bucket ladders and dead xtrax duplicates, and adopts device_memory_budget and loop_scaling. Closes #2358 and the bucketing and hygiene parts of #2493.
task_id: 261006_runner-bucketing
status: draft
---

# S8: runner length bucketing (default on) and xtrax-native adoption

- **Sources:**
  - audit `.praxia/docs/audits/261006_xtrax-usage-audit.md`;
  - aminx debt #2493 (audit) and #2358 (padding cost);
  - xtrax debt #2496–#2503 (skills, Track B).
- **User decision (261006):** "core runner should bucket by default".

## 1. Goal and non-goals

**Goal.** `aminx.host.runner.sample` and `score`, the `aminx run` CLI, and the streaming and campaign paths run each batch at the smallest xtrax `BUCKET_LADDER` rung covering its real residue span, instead of at `max_length` (default 512).
- Outputs are re-padded to the padded length, so every downstream shape is unchanged: sinks, zarr, campaign manifests, `io/designs.py`.
- `length_bucketing=False` restores today's behaviour bit for bit.

**Non-goals (v1).**
- `inspect` and `jacobian`. They return per-L visible tensors; v2 can add them.
- `_score_fused_multistate` and `sample_multistate_poe`, which have their own stacking.
- `pass_mode="inter"`.
- `max_length=None`.
- Any change to the JAX sampler API (`aminx.sample`).
- Replacing `safe_map` / `safe_scan` / `_dispatch_axis`. That waits on xtrax `chunked_map` gaining the size-1 guard (Track B, aminx #2371).
- `convert_to_onnx` adoption, which the hub DAG owns: S4-43, S4-14, S4-17.

## 2. Assumptions

| id | assumption | tag | evidence |
|---|---|---|---|
| A1 | Neighbour indices on the host path are computed in the model from coordinates, not taken from the loader's `neighbor_indices` field, so trimming coordinates re-derives k-NN over the rung | verified | `inference/encode.py:262`, `model/features.py:455`; no host use of `precomputed_neighbor_indices` |
| A2 | Every spec array (bias, fixed_mask, fixed_tokens, fixed_positions, tie_group_map, state_position_map, structure_mapping, ligand Y/Y_t/Y_m) is validated against the padded length before `per_structure` is built in `_sample_batch` | verified | `host/kernel_dispatch.py:198-241` |
| A3 | `make_inference_plan` is built once per run with no L baked in; `encode` and `decode` are `eqx.filter_jit`, so each rung is one trace | recon, partly read | `host/plan.py:559-772`, `runner.py:~296` |
| A4 | Scores are invariant to padding at backbone_noise 0; at noise > 0 the noise draw has the padded shape and scores change | first half verified (test exists); **second half refuted 261006**: the plain score path forwards no noise level (`runner.py` encode-once comment, #147), so 1ubq scores 4.4554 at noise 0, 0.1 and 1.0, bucketed or not (spike on titanix; debt #2509). Only the averaged-feature path uses noise, and it is not bucketed | `tests/host/test_score_padding_invariance.py`; `utils/coordinates.py:47` |
| A5 | Sampled sequences change with the padded length: `random_design_order` draws `uniform((L,))` | verified | `utils/decoding_order.py:143`, `runner.py:72` |
| A6 | Putting a `Bucket` decision for the residue axis into the joint planner would corrupt the memory estimate, so v1 picks the rung on the host and leaves the residue axis unplanned | verified | `tiling/planner.py:54-60` |
| A7 | `make_sampling_planner` is rebuilt per batch from `seq_len`, so passing the rung sizes memory correctly | recon | `kernel_dispatch.py:179-184` |
| A8 | `xtrax.tiling.select_bucket` raises above the largest boundary (2048) | verified | `xtrax/tiling/bucket.py:55` |
| A9 | Reading `BUCKET_LADDER` from `xtrax.export.rings` loads no onnx or jax2onnx | verified | test on PR #195, `tests/agent/test_max_length.py` |
| A10 | Max checkpoint k-NN is 48, at most the smallest rung (64) | verified (naming) | checkpoint ids `*_v_48_*` and `*_v_32_*` |
| A11 | No test or golden pins an exact seeded runner sample | recon | blast-radius recon, 261006 |
| A12 | Tie-group ids must be below the trimmed length | recon | `decoding_order.py:134` |
| A13 | Logits at masked positions today are not relied on downstream; the re-padded tail is 0 | **partly refuted 261006**: the opt-out path samples every padded position (golden tokens and logits are nonzero out to 511), so under bucketing masked positions inside the rung are sampled and only the tail past the rung is 0 | settled by G-OPTOUT (old path unchanged) and G-SHAPE |
| A14 | `xtrax.tiling.device_memory_budget` raises on devices without `bytes_limit` (CPU) | verified | `xtrax/tiling/estimators.py:52-56` |
| A15 | `xtrax.profiling.loop_scaling` can bound the AR decode scan body's growth from jaxpr | **unverified** | spiked in S8-10 before it is built on |

## 3. Design

### 3.1 Rung selection (one helper, host-side)

`aminx.host.bucketing` (new) provides `rung_for(span, padded_len, *, enabled) -> int | None`.
- It returns `None`, meaning no trim, when bucketing is disabled, `padded_len` is `None`, or `pass_mode == "inter"`.
- Otherwise it returns `min(select_bucket(span, BUCKET_LADDER), padded_len)`, or `None` when `span > BUCKET_LADDER[-1]`.
- `span` is the index of the last valid residue + 1, taken across the batch. This is the same computation as `_PaddingCheck`; factor it out.
- The ladder comes from `xtrax.export.rings.BUCKET_LADDER`, imported lazily. Bucket selection always goes through xtrax `select_bucket`, never a local copy.

### 3.2 sample: trim in `_sample_batch`

After `_prepare_fixed_controls` and `_prepare_ligand_context`, which validate at the padded length, and before `per_structure`:
1. Compute `rung = rung_for(...)`.
2. If it is not `None`, slice every residue-axis array to `[..., :rung]` on its residue axis. That covers coords, mask, residue_index, chain_index, mapping, atom_mask, aatype, fixed_mask, fixed_tokens, tie map, state_position_map (last axis), ligand Y/Y_t/Y_m and atom_37.
3. If any tie id is ≥ rung, skip the trim for that batch (A12) and log it at debug.
4. Recompute `order_num_groups` from the trimmed tie map.
5. Pass `rung` as `seq_len` to `make_sampling_planner` (A7).

After the transpose (`~:527`):
1. Compute `pseudo_perplexity` on the trimmed arrays.
2. Pad sequences, logits and masks back to the padded length with 0 before the `io_callback`.

### 3.3 score: trim after ligand prep (`runner.py:~893`)

- `rung = rung_for(max(span, longest sequence_to_score))`. The too-long check stays against the padded length.
- Slice the structure arrays and sequences to the rung. Re-pad `batch_logits` before the concatenate at `~:1036`.

### 3.4 Opt-out and records

- **Field.** `length_bucketing: bool = True` on `RunSpecification` (`run/specs.py`, next to `max_length`), as a flat field.
- **CLI.** `--length-bucketing/--no-length-bucketing` wherever `--max-length` is accepted.
- **Specs.** `spec_json` handles the field automatically. Old specs decode to True, so a replayed old spec now buckets. The docs and the CHANGELOG say so, and `--no-length-bucketing` reproduces old runs.
- **Knobs.** Add a `knob_observations.py` entry.
- **Campaigns.** Bump `SAMPLING_NUMERICS_EPOCH` 1 → 2 (`host/campaign.py:59`), so campaign units stamped before this change are not reused as equivalent.
- **Warning.** `_PaddingCheck` keeps warning only when the padding stays large: when bucketing is off, or when a skip condition kept the padded shape. Update its text and `_PADDING_COST_TEXT`.

### 3.5 Hygiene (no behaviour change)

- Fix the stale imports. `xtrax.tiling.SafeMap` → `ChunkedMap` in `scripts/ebm/benchmarks/langevin_benchmark.py`. `DedupSpec` → import it from `xtrax.tiling.dedup` in `host/plan.py:28`.
- Update the stale `RINGS = "not_available"` a10 assumption in `scripts/browser_validation/layer_b_iree.py:39,116`. The a11 pin has `xtrax.export.rings`.
- Delete dead code once every reference is cleaned up (tests, `tests/parity/browser_validation_paths.json`):
  - `tiling/buckets.py`;
  - `tiling/eda.py`;
  - `host/stage_adapter.py` with its test;
  - `host/plan.plan_bucketed` with its test;
  - the aminx `tiling/strategy.DedupGather`, if it has no references.
- `tiling/bucketing.select_bucket` delegates to `xtrax.tiling.select_bucket`; `BucketingConfig` stays as the opt-in `bundle_builder` config.

### 3.6 Adoptions

- **Memory budget (S8-09).** `resolve_memory_budget_bytes()` replaces the two silent 4 GiB fallbacks (`host/plan.py:279-281`, `tiling/planner.py:107-110`). It follows the derived-config rule:
  1. an explicit argument;
  2. `AMINX_MEMORY_BUDGET_BYTES`;
  3. `[tool.aminx] memory_budget_bytes` in the nearest `pyproject.toml`;
  4. `${XDG_CONFIG_HOME:-~/.config}/aminx/config.toml`, `[runtime] memory_budget_bytes`;
  5. `xtrax.tiling.device_memory_budget(fraction=headroom)`;
  6. the documented 4 GiB default, with a one-time INFO.
  `memory_budget_source()` reports which layer decided, and a malformed config fails loudly. GPU behaviour is unchanged, because layer 5 equals today's `bytes_limit * headroom`. CPU behaviour is unchanged except that the INFO names the source. CPU never reports `bytes_limit`, so the default-budget INFO line is the normal CPU path.
- **AR complexity guard (S8-10).** A test that uses `xtrax.profiling.loop_scaling` to assert that the AR decode scan body does not grow with L (the aminx #1983 class of bug). Spike the API first (A15). If it cannot express the check, record that and file it upstream; don't hand-roll.

## 4. Gates

| gate | check |
|---|---|
| **G-OPTOUT** | With `length_bucketing=False`, `sample` and `score` outputs equal a golden captured at the pre-change base commit (S8-01) bit for bit: 1ubq, seed 7, 2 samples, temperature 0.1, max_length 512. |
| **G-INVARIANCE** | At noise 0, `score` bucketed equals opt-out within rel 1e-6 on 1ubq, 5awl (10 residues, below k), a two-chain fixture, a gapped fixture (unresolved backbone) and a ligand fixture. Negative control: scoring a different sequence must differ (noise cannot serve, see A4). |
| **G-SHAPE** | Bucketed sample outputs have exactly the opt-out shapes (padded L), the re-padded tail past the rung is 0 (masked positions inside the rung are sampled, as in opt-out), and real positions are valid tokens. |
| **G-CONTROLS** | Under bucketing, fixed positions keep their `fixed_tokens`, tied groups share tokens, and bias at a real position still moves its logits. Negative control: the same bias placed past the span has no effect. |
| **G-COMPILE** | Inputs spanning 3 lengths in 2 rungs trace `decode` exactly 2 times (counted with `JAX_LOG_COMPILES` or a trace counter). |
| **G-SPEED** | bathos, pre-registered, titanix CPU, chunked per cell. Expect bucketed `sample` on 1ubq (76 → 128) ≥ 3× faster than opt-out. Control: 3pgk (415 → 512) within 0.8–1.25×. |
| **G-SUITE** | Full default `tests/` suite (repo addopts; includes `tests/host` and `tests/tiling`) on titanix; the parity e2e cells (`scripts/parity/e2e_run_api_parity.py`) pass. (Amended at run time: this row first named `tests/agent`, which exists only on PR #195, and opted out pad512; that cell also pins `length_bucketing=False` and costs ~7 min, so every cell runs and the registered e2e verdict applies unchanged.) |

## 5. Items

```toml
[[item]]
id = "S8-01"
title = "Capture opt-out golden at the pre-change base commit on titanix (sample+score, 1ubq, seed 7) via a tracked script; commit fixture + capture script"
size = "S"
depends_on = []
gate = "fixture committed with base sha recorded"

[[item]]
id = "S8-02"
title = "length_bucketing field (RunSpecification) + CLI flags + knob_observations entry + spec_json round-trip test; aminx.host.bucketing.rung_for helper on xtrax select_bucket + BUCKET_LADDER (lazy) with unit tests and an onnx-not-loaded test"
size = "S"
depends_on = []
gate = "unit tests; old JSON spec decodes with True"

[[item]]
id = "S8-03"
title = "sample: trim batch + spec arrays to the rung in kernel_dispatch._sample_batch after validation; tie-id guard; planner gets rung; pseudo-perplexity on trimmed; re-pad outputs to padded L before io_callback"
size = "M"
depends_on = ["S8-02"]
gate = "G-OPTOUT, G-SHAPE, G-CONTROLS, G-COMPILE"

[[item]]
id = "S8-04"
title = "score: rung = max(span, longest sequence); trim after ligand prep; re-pad batch_logits before concatenate"
size = "S"
depends_on = ["S8-02"]
gate = "G-INVARIANCE, G-OPTOUT"

[[item]]
id = "S8-05"
title = "SAMPLING_NUMERICS_EPOCH 1->2; _PaddingCheck only when padding stays large; update _PADDING_COST_TEXT, README, the 4 padding-warning e2e tests; opt-out in e2e_run_api_parity pad512 cell and repro_gpu_sampler_divergence; CHANGELOG entry (seeded outputs change; old specs now bucket; --no-length-bucketing reproduces)"
size = "S"
depends_on = ["S8-03", "S8-04"]
gate = "G-SUITE"

[[item]]
id = "S8-06"
title = "Pre-registered bathos timing: bucketed vs opt-out sample, 1ubq + 3pgk control, titanix CPU, per-cell persistence"
size = "S"
depends_on = ["S8-03"]
gate = "G-SPEED (sidecar committed before the run)"

[[item]]
id = "S8-07"
title = "Hygiene: stale SafeMap/DedupSpec imports; layer_b_iree RINGS a10 assumption; tiling/bucketing.select_bucket delegates to xtrax"
size = "S"
depends_on = []
gate = "G-SUITE"

[[item]]
id = "S8-08"
title = "Delete dead code: tiling/buckets.py, tiling/eda.py, host/stage_adapter.py (+test), plan_bucketed (+test), aminx DedupGather if unreferenced; update browser_validation_paths.json inventory"
size = "S"
depends_on = []
gate = "G-SUITE; rg shows no remaining references"

[[item]]
id = "S8-09"
title = "resolve_memory_budget_bytes() + memory_budget_source(): arg > env > pyproject > XDG config > xtrax device_memory_budget > documented 4 GiB with one-time warning; replace both silent fallbacks"
size = "S"
depends_on = []
gate = "layer-precedence tests; malformed config raises; GPU path equals bytes_limit*headroom"

[[item]]
id = "S8-10"
title = "Spike xtrax.profiling.loop_scaling on the AR decode; if it fits, add a CI test that the scan body does not grow with L; else record and file upstream"
size = "S"
depends_on = []
gate = "spike answer recorded; test has a red control (planted O(L) body)"

[[item]]
id = "S8-11"
title = "PR #195 follow-up (after S8 merges): drop agent fitted_max_length; tools rely on runner bucketing"
size = "S"
depends_on = ["S8-05"]
gate = "tests/agent green; parity on returned spec"
```

## 6. Risks

| risk | mitigation |
|---|---|
| A spec array missed in the trim silently misaligns | G-CONTROLS has a negative control: bias past the span must have no effect. List every residue-axis field in one place in the trim helper. |
| Mixed rungs across batches break concatenation | Outputs are re-padded to the padded length before they leave `_sample_batch` and score. G-SHAPE checks it. |
| Old campaign units reused across the numerics change | The epoch bump. |
| Users replaying old seeded specs get different sequences | The CHANGELOG, the docs, and `--no-length-bucketing`. |
| Score drift at noise > 0 | Does not arise: the plain score path ignores backbone_noise (A4, debt #2509). G-INVARIANCE's negative control scores a different sequence instead, which proves the instrument fires. |
