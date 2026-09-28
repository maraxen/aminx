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
