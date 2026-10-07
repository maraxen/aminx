---
title: "Track B brief: xtrax skill fixes and the chunked_map size-1 guard (for a session opened in the xtrax repo)"
description: A ready-to-run brief for an xtrax session. It fixes the skill gaps filed as xtrax debt #2496-#2503 and upstreams aminx's size-1 vmap guard into xtrax chunked_map, which unblocks aminx dropping safe_map, safe_scan and _dispatch_axis (aminx #2371).
task_id: 261006_xtrax-skills-coverage
status: final
---

# Track B brief (run this in a session opened in `~/projects/xtrax`)

**Why a separate session.** The aminx session that produced this is isolated to aminx worktrees and cannot run git in the xtrax repo. The local `~/projects/xtrax` checkout is behind `origin/main`, so start from a worktree off `origin/main`.

## Inputs

- **xtrax debt** #2496–#2503, filed 261006 with evidence and path roots.
- **aminx audit:** `.praxia/docs/audits/261006_xtrax-usage-audit.md` on aminx branch `wt/261006-xtrax-usage-audit`, which is pushed.
- **aminx inventory:** bathos run `da99c566`, `outputs/audit/xtrax_usage.json` on the same branch. Of 619 public xtrax a11 symbols, aminx `src` uses 45.

## Work, in order

1. **#2496 (P1). Fix the wrong skill examples and add a CI signature check.** Wrong examples:
   - `tiling.md`: `select_bucket` and `bucketize` signatures;
   - `training.md`: `SafetyTrainStep`, `Engine`, `make_optimizer` and `adamw_with_schedule`;
   - `sparse-distributed.md`: `init_dist`, `LogicalMesh` and `save_checkpoint`;
   - `inference.md`: the symbol count;
   - `export.md`: `compare_leaves`.
   Re-check each against main, since the evidence came from the a11 wheel. Then add a test that extracts the fenced python blocks under `agent_assets/skills/**`, resolves their imports and binds their calls with `inspect.signature`.
2. **Code: the `chunked_map` size-1 guard (aminx #2391/#2371).** Port aminx's guard (`src/aminx/utils/safe_map.py:22-79`) into `xtrax/transforms/map.py`. It never vmaps a chunk of 1: a size-1 axis, `batch_size == 1`, or a remainder of 1 is peeled off and run unbatched. The reason is a measured GPU miscompile, aminx #2391. Add a test with a red control. Once this releases, aminx can re-export it and delete `safe_map`.
3. **#2497.** An end-to-end bucketing recipe:
   - the chain `AxisSpec.bucket_boundaries` → `Bucket` → host `select_bucket`/`bucketize` → one compile per rung;
   - `BUCKET_LADDER` as the default ladder, re-exported from `xtrax.tiling`;
   - the warning that `jnp.pad` on device recompiles;
   - trimming a loader that pre-pads.
   aminx now implements this in its runner (aminx spec S8) and can serve as the worked example.
4. **#2498.** An "adopt xtrax / replace local copies" workflow, a table mapping symptoms to primitives, and deprecation-alias lifetimes (`_renamed.py`; SafeMap goes next release).
5. **#2499.** A standalone `convert_to_onnx` section, the rings/divergence ladder, `find_onnx_rng_ops`, `verify_native_parity`, and ONNX/browser triggers.
6. **#2500.** A "not provided" table (no inference work-unit/resume layer; `resume` is training-only), and `references/telemetry.md`: the ledger, `XTRAX_TELEMETRY_OPTOUT`, and `Engine.fit` failing closed.
7. **#2501.** Document `dedup_synthesis` (replacing the `np.unique` advice), `profiling.loop_scaling`, `accumulate_grads`, the `adamw_with_schedule` defaults, and inference CSE/memo.
8. **#2502.** Problem-phrased triggers, with a trigger eval.
9. **#2503.** Delivery: ship all five skills through the plugin export, version-stamp the reference sections, and have the preflight warn when the skill's `xtrax_version` differs from `xtrax.__version__`.

**Related open xtrax items:** #2101, #2103, #2105, #2107, #2325.
