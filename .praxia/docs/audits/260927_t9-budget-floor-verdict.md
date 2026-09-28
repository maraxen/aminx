---
title: T9 budget-floor probe verdict (floor_exceeded)
description: T9 sampling-track budget-floor probe result and root-cause diagnosis; gates T10 pending T9b + rerun.
task_id: 260926_browser-export-loop
status: closed
---

# T9: sampling-track budget-floor probe verdict

- **task_id**: `260926_browser-export-loop` (sprint `260927_aminx-browser-export-phase2a`, subtask T9)
- **run**: bathos run `eedd3993`, titanix, at commit `cb2b7264`, `git_dirty=false`
- **outcome**: `floor_exceeded` (an allowed, pre-registered outcome)

## Numbers (as measured on titanix)

- `aa_ratio` = 0.979 (force-vs-force cost-measurement stability control, in-range)
- `fastpath_ratio` = `c_force / c_off` = 0.0385
- `c_force` = 0.0883 s/draw
- `c_off` = 2.292 s/draw
- `budget_wall_hours` (projected, at `N_REQUIRED_FLOOR`) = 446.4 h vs the 16 h cap
- `projected_peak_rss_gib` = 4.38 / 48 GiB (RSS not the binding constraint)
- `n_draws_forced_off` = 0 on every lane (P07@0.1, P07@1.0, P08, P09-s, P11-s)
- `per_lane_hours`: P07@0.1 = 105.8, P07@1.0 = 106.1, P08 = 106.0, P09-s = 54.5, P11-s = 74.0
- At the 4YOW / P07@1.0 control fixture (L=693), the production per-draw cost was 2.372 s vs
  the reference's 0.291 s/draw

## Diagnosis

`aminx_sample_batch` (`scripts/browser_validation/layer_a_sampling.py`, then at line 718)
called `sample_autoregressive.kernel` with `incremental="auto"` (the default) under `jax.vmap`.
Under `vmap`, `AutoregressiveDecode.incremental` is a static, batch-homogeneous field, so
`lax.cond` executes BOTH branches for every draw (V16) -- `"auto"` therefore priced every draw
at (approximately) `c_off`, the full O(L^2 k) recompute cost, never the O(Lk) incremental
fastpath. The budget formula
(`layer_a_sampling_calibrate._budget_at_floor`: `hours += alloc * (162 * c_a + 2 * c_r) / 3600`,
with `c_a = layer_a_sampling.measure_aminx_draw_cost_s`) inherited this inflated per-draw cost
directly, since `measure_aminx_draw_cost_s` times `aminx_sample_batch` unmodified.

Separate, independent evidence (T7, bathos run `29a1a43b`, branch `feat/phase2a-loop-cost`,
commit `59fa37a5`): the HLO loop-body of the AR kernel under `incremental="force"` still scales
with `L` (alpha 0.679 vs the 0.35 bar; offenders: `bitcast_dynamic-slice_fusion`,
`select_dynamic-update-slice_fusion`, `bitcast_select_fusion` -- full-length buffer
selects/updates per wave). This is recorded here as context only; it is NOT fixed by T9b and is
out of scope for this task.

## Gate

**T10 does NOT run** until T9b (host-routed `incremental` mode for `aminx_sample_batch`, closing
the `"auto"`-under-`vmap` root cause above) lands AND the T9 budget-floor sidecar is rerun on
titanix against the fixed code. T9b's implementation, tests (exactness + routing + red-check),
and a CPU smoke before/after comparison are recorded in the T9b commit on this branch
(`feat/phase2a-sampling-track`); the rerun itself is the orchestrator's responsibility, not this
worktree's.

## Rerun after T9b (2026-09-27)

bathos run `27f9f25e` on titanix (GPU 2) at `f4176aaf` (T9b), git_dirty=false, sidecar sha256
`867c05d5…` == committed file (same pre-registration as `eedd3993`). Verified from the cool-tier
parquet record (`bth compact` is currently broken catalog-wide). **Outcome: `floor_exceeded`.**

| field | eedd3993 (cb2b7264) | 27f9f25e (f4176aaf) |
|---|---|---|
| budget_wall_hours (cap 16) | 446.4 | **119.4** |
| projected_peak_rss_gib (cap 48) | 4.38 | 5.67 |
| fastpath_ratio | 0.0385 | 0.0509 |
| aa_ratio | 0.979 | 1.019 |
| P07@0.1 / P07@1.0 / P08 h | 105.8 / 106.1 / 106.0 | 16.0 / 16.1 / 15.8 |
| P09-s / P11-s h | 54.5 / 74.0 | 9.2 / **62.3** |

Per-draw aminx cost is now at parity with the reference on P07/P08/P09-s (e.g. 4YOW L=693:
0.34 s vs 0.33 s). P11-s (side-chain lane) is the remaining outlier: 1.44 s vs 0.34 s at L=693.

**Reading:** the cap is infeasible for this protocol independent of aminx speed -- each
non-side-chain lane alone costs ≈ the whole 16 h cap at reference-parity per-draw cost; fixing
P11-s to parity would still leave ≈ 73 h. Per the pre-registration the sampling track stops and
T10 does not run. Unblocking needs a user decision (cap, lane split across GPUs, or a smaller
protocol) and a NEW pre-registration -- not a re-grade of this one.
