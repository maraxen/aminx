---
title: '261005 overnight: Potts r2 passed, and the four decisions that remain'
description: What landed overnight with its verified evidence, the four open decisions each priced, and the corrections made to earlier claims
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261005'
---
# 261005 overnight: Potts r2 passed, and the four decisions that remain

Branch `wt/260929-laser-confirm`, 21 commits, **every one outside the freeze** — the eight
ledger rows were never touched and are verified live (§4). This is the index; the detail lives
in `plans/261005_rewave-composition-scoped-batch.md` and
`reference/261005_laser-confirm-run-readiness.md`.

## 1. The result: the Potts confirmatory run PASSED

Run `096d0847-0930-4d0b-a86d-8757b3af79b6` @ `5c625b00`, verified by its cool-tier record:
`status=completed`, `outcome=pass`, `exit_code=0`, `git_dirty=False`,
`outcome_is_residual=False` (it matched the *declared* pass condition, not a fall-through),
`duration_s=41863.66` (11.63 h). Claimed artifact resolves.

```
refine@0.3  ->  PASS
  mean_delta 4.90e-05   upper_95 0.001114 < delta 0.01 (9x)
                        max_delta 0.001949 < 2 delta 0.02 (10x)
negative control CTRL_m (m=1.5)  ->  FAIL, as required
  lower_90 0.017839 > delta 0.01      controls_all_fail = true
per-structure delta: 3gg7 +0.00036  4jox +0.00195  6w25 -0.00178  swe1_ligand -0.00033
n_computed = 16   n_reused = 0
```

Three reasons this is a real pass and not a flattering one: the **negative control failed**, so
the instrument discriminates; **`n_reused = 0`**, so all sixteen units were computed fresh with
nothing carried from round 1; and two per-structure deltas are **negative** (aminx closer to
upstream U1 than U2 is), the signature of genuine indistinguishability rather than a near-miss.

**#2474 resolved.** Round 1 (`1cfc1e9d`) recorded `outcome=fail` **at exit 0** — one of three
graded failures found tonight that exit zero, and the standing argument for never reading an
exit code as evidence. Round 1 refined N-to-C while upstream refined each sample in that
sample's own AR order; round 2 changes only the aminx refine order.

## 2. The four decisions

### D1 — `laser_score` option 3 (the only thing blocking Z1)

Z1 has **exactly one unmet criterion**. Checked against spec line 1246: the gate sidecar digest
matches (`39babc8b…`), `git_dirty` is false, `status=completed`, and the ledger re-run set is
**empty** — all eight rows pass 1c today. Only `outcome=='pass'` fails, caused by step 1's 63
`laser_score` clean-arm failures and nothing else.

Band verified by its own record this session (run `3af3be02`, `completed/pass/exit 0`):
`proposed_tol` **4e-4** over a measured floor of **3.63e-5** — **10x headroom**, *not* the 100x
quoted earlier. Supporting evidence from the same artifact: `n_pairs = 63` (the band covers
exactly the failing population), `control_rejected = true` (instrument tested),
`loo_all_admitted = true` with `loo_worst_ratio = 0.121` (no single outlier props the floor up),
and `f32_violated_elementwise = true` / `f64_passes = true` (f32 precision, not a logic defect).

The vehicle lives in `scripts/analysis/`, which is **unscoped**, so this evidence survives the
wave and needs no re-measuring.

### D2 — redsox provisioning: the cheap half unblocks a gate run now

The source is already on titanix (`/home/solab/projects/redsox`, `redsox 0.1.0a1`, installable);
it is simply not installed. That splits the blocker:

- **(a) hand-install** → step 2 runs now, no repo change, **not scoped**, no row touched.
- **(b) declare it so it survives `uv sync`** → the durable fix #2319 wants, but
  `pyproject.toml` is scoped, so it rides the wave.

(a) lets the gate be re-run to confirm the only red really is `laser_score` **before** anyone
commits to a scoped wave.

### D3 — re-wave composition: 11.63 h, and the timing is already forced

Price is now **measured, not projected**: 11.63 h, serial, 4 targets × 4 arms, cost dominated by
*target* (the `swe1_ligand` tier runs 66–100 min/unit vs 6–14 for the cheapest). A re-run pays
the same mix.

**The wave is elective in composition but not in timing.** This branch carries 41 scoped
commits — including the `origin/main` merge (xtrax 0.4.0a11 + the ChunkedMap fix) and the
confirmatory drivers — none of which has reached the sprint lineage. The moment they do, all
eight rows die at once. So: wave *after* the merge, or pay for two re-measurement cycles.

Every Tier A/B candidate now carries a measured size, including **B3** (#2480(b)), diagnosed to
one line: `pad_bundle` (`src/aminx/tiling/pad.py:44-54`) pads six conditioning fields and omits
`state_position_map`.

### D4 — the LASEr distributional run: **~237 h, not the recorded ~60 h**

Not launched; not a gate slug, so it blocks nothing. But the cost was re-derived overnight and
**the figure in project memory is low by roughly 4x**, which changes the shape of this decision
more than anything else in this brief.

The ~60 h came from *upstream* throughput (~1 h per n=1000 unit) and appears to price every
unit at upstream cost. Measured overnight, the **aminx arms are ~17x the upstream arms**
(33.6 min vs ~2 min at n=50). The confirm is 2 cells × 5 structures × 4 arms = **40 units**,
half of them aminx:

```
upstream  20 units  ~0.67 h each  ->  ~13 h
aminx     20 units  ~11.2 h each  -> ~224 h     (linear scaling)
                                     ~237 h total, ~10 days serial
```

**Linearity is measured, not assumed.** An n=200 probe (same unit, same seed, 4x samples) ran
8046 s against the n=50 sampling time of 2018 s — ratio **3.86 for 4x**, exponent **0.975**,
i.e. linear to within ~2.5%. √-scaling was excluded en route (it predicted 67 min; the probe
passed that still running). Propagating the measured exponent gives **~10.5 h per aminx unit at
n=1000 → ~223 h total**, marginally under the linear estimate and the same order. Use ~223 h as
the planning number; **no sublinear rescue of ~60 h exists.**

Two further pre-flight results, both clean: the resume design is correct *and proven* (the same
`graded_resume` helper ran the 16-unit Potts job), and the grading contract audits **total and
disjoint** — a smoke cannot be mistaken for the confirmatory test, and `pass`/`fail` both
require `controls_all_fail = true`. The plumbing was also exercised end to end by a smoke: the
oracle subprocess, `--laser-root` resolution, job handoff and stamping all work on this branch
under xtrax 0.4.0a11.

**Trap if launched:** `script_sha256` is in the cache key, so editing the driver after launch
discards *every* completed unit. Freeze it first.

**The pilot, however, is already done — and nobody knew.** No LASEr distributional run had ever
completed: the one full pilot attempt (`03d2ef37`) burned **21.68 h** and exited 1, because its
cell list included `min_p0.05@0.3`, the cell the confirm now refuses for upstream NaNs. But its
24 completed units survive on disk and the pilot script is unchanged since that run, so a
resumed 2-cell run reused **all 24, computing zero**, and finished in seconds:
`all_cells_derived = true`, `any_instrument_invalid = false`, `shim_ok = true`, no missing cells,
no open questions. Its `h_hat`/`chosen_m` are **digit-for-digit identical** to the confirm's
pre-registered `CELLS` table, closing that provenance chain. So **spec row B5 is substantively
complete**; what is missing is one tracked `bth` invocation (minutes, since units are reused),
not ~134 h of compute.

**What this reframes:** a GPU oracle was previously a convenience. At ~10 days serial it is the
central question — and note it is the **aminx** arms a GPU must accelerate, not the upstream
oracle, which is the opposite of how that was framed before these measurements.

## 3. Also landed

`#2482` resolved — the import-isolation gates could not catch a `try/except`-guarded import of
an absent module (probed: a failed import leaves nothing in `sys.modules`); now a
`sys.meta_path` recorder plus a permanent negative control. Three tests taken off absolute home
paths (one was *silently skipping on the gate host*, hiding a stale `EXTRACTOR_SHA256`). The
`ar_sample` benchmark's `inference_only` relocated rather than deleted (deleting it would have
silently switched the benchmark to the slower-compiling path). `docs/MODEL_FAMILIES.md` plus
Z2's skill half staged as an applyable patch — including one line in the skill that is *wrong*
today, not merely missing (`--model-family` listed as two values; there are four).

`tests/knob_gate` verified green on this branch: **35 passed, 2 skipped**, both skips structural
(they need `AMINX_REDSOX_OUTCOMES_READ`, exported only by `run_gate.py`). It took three venvs to
find one that works — `aminx-b7i-git` has both xtrax 0.4.0a11 and pyarrow; `aminx-confirm-git`
dies on the xtrax floor, `aminx-ci` has no pyarrow.

## 4. Claims corrected this session

Recorded because the pattern matters more than any single number.

| claim | corrected to |
| :-- | :-- |
| B2 re-run "~5–6 h at 4-way concurrency" | no concurrency at all; **11.63 h measured** |
| then "~8.5–9 h", from a *bimodal, arm-driven* model | four **target** tiers; the four fast units are all four arms of target 1 |
| "aminx arms cost 2–3× upstream" | measured A/U1 = **0.43, 1.54, 0.84, 1.52** — sometimes *faster* |
| laser_score band "100x margin" | **10x** (`headroom = 10.0`) |
| #2480(b) is "a padded length"/`ar_sample`/sample-axis cardinality | a **bucket ceiling**; the failing cell is `score_conditional` |
| #2480(b) fix is in `bundle_builder.py` | `src/aminx/tiling/pad.py:44-54` |
| import-isolation tests "cannot fail" | they can; the gap is narrower and specific |
| rows anchored at `feade020` | rows sit at **three** earlier trees (`719c33e0`, `e9ef452f`, `124f6e9e`); re-checked per anchor |
| Potts 11.63 h "may be inflated by cold compiles" | withdrawn — target 1's aminx units ran 6 min *total* at n=1000, so compile is ≤7% |
| a persistent XLA cache is an "hours-saving lever" | measured **~12%** of a unit (4.4 min of 38); modest, not transformative |
| LASEr confirm "~60 h" (project memory) | **~237 h** — that figure priced all 40 units at upstream cost; aminx arms are ~17x |

Note the direction. Of the eleven, **every cost figure erred low and every safety margin erred
high** — the re-run looked cheaper, the band looked safer, the compile lever looked bigger, the
LASEr run looked shorter. Each came from extrapolating a partial observation in the direction
that made the work look easier. That is the bias to watch for in the next estimate, more than
any individual number here.

Every ledger-row claim is now verified against the gate's own predicate rather than asserted
from path reasoning, and all eight rows read `completed/pass/exit 0/git_dirty=False`.
