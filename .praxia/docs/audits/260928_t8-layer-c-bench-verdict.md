---
title: T8 layer-c browser benchmark verdict (ctrl_blind)
description: T8 tracked run edf8926e graded ctrl_blind; planted 5 ms control reads ~16 ms systematically; timing sentences not citable; v2 needs a dose-response instrument check
task_id: 260926_browser-export-loop
status: final
---

# T8: layer-c browser benchmark verdict

- **run**: bathos `edf8926e-2a44-40ca-88a8-fd87015fc3cb`, local, commit `41f7d892`, `git_dirty=false`,
  sidecar sha256 `fe348e8f…` == committed `layer_c_bench.bth.toml` (verified from the cool-tier parquet).
- **outcome**: `ctrl_blind` (a pre-registered failure state). This is the only tracked run of the stem.
- Artifacts: `outputs/browser_validation/perf/runs/layer_c_bench-41f7d892ee72/`.

## Numbers

| field | value |
|---|---|
| cells measured / errors / below minima | 78 / 0 / 0 |
| repeats, min warmup, min iterations | 3, 5, 30 |
| isolation_ok, harness_ok, budget_expired | true, true, false |
| aa_ratio (dup vs base, P04 L=256 t1) | 1.00005, CI contains 1 — **pass** |
| planted_5ms diff estimate | **15.87 ms**, CI excludes 0 — outside the 4–6 ms band — **fail** |
| native_threads | unpinned (jax 0.10.2 cannot pin; no native ratios cited) |

P04 L=256 one-thread p50s (n=90 each): base 767.9 ms (sd 14.0), dup 767.9 ms (sd 9.7), planted 783.7 ms (sd 26.8).

## Reading

The instrument resolves ~1 ms (timing CIs are ±1 ms wide) and the same-route control is exact, so
the planted miss is **systematic, not noise**: the untracked smoke run before it read 17.5 ms for the
same 5 ms plant. The `busyWait` in `bench.mjs` (inside the timed region, after `session.run`) is
therefore not a clean additive 5 ms in this runtime. Cause unknown.

Because the positive control failed, **none of the 24 timing sentences in `report.md` are citable.**
`compat_claim_gate.py` on the report exits 0 (it makes no compatibility claims).

## Before a v2

Replace the single 5 ms plant with a pre-registered dose-response check (plant 0/5/20/50 ms; pass on
slope ≈ 1 and intercept ≈ 0 within stated bounds), and diagnose the ~11 ms excess first. Do not
re-grade this run.

## Attempt history

1. Untracked smoke (driver at 93397a24) crashed: `KeyError: 'steady_ms'` — browser timings are nested
   under `phases`. Fixed in 41f7d892 (Cursor dispatch a07dedc6, reviewed + red-checked by the orchestrator).
2. Untracked smoke at 41f7d892 (buckets 128,256): 42/42 cells, planted 17.5 ms.
3. Tracked run edf8926e (this record).
