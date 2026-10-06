---
title: 'LASEr confirmatory run: readiness verified, and the one edit that would discard 60 hours'
description: The resume machinery for the ~60 h LASEr distributional run is correct and already proven by tonight's Potts run; its cache key includes the script hash, so the driver must be frozen before launch
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261005'
---
# LASEr confirmatory run: readiness verified, and the one edit that would discard 60 hours

Whether to spend ~60 h of CPU-only oracle time on the LASEr distributional confirmatory run is
an open decision and stays the user's. This note is only the pre-flight: *if* it is launched,
is it safe to launch? Checked tonight, while the Potts confirmatory run was in flight.

## Verdict: the resume design is correct, and it is not merely designed — it is running

`scripts/parity/laser_sample_dist_confirm.py` takes `--work-dir` and `--resume`
(`BooleanOptionalAction`, default **True**), and declares its unit granularity in its own
docstring: *"One resumable unit is one (cell, structure, run)."* It builds
`graded_resume.Unit`s carrying `checkpoint_sha256`, `input_sha256` and `script_sha256`.

`scripts/parity/graded_resume.py` is the shared helper, and its contract matches the standing
preemption-safety rule point for point:

| rule | implementation |
| :-- | :-- |
| persist each unit as it completes, never only at the end | per-unit body + completion stamp under the work dir |
| a crash must not leave a plausible corpse | *"The body is written first and the completion stamp second"* — a crash mid-unit leaves a body with no stamp, so the unit recomputes |
| reuse only when inputs and artifact hashes match | cache key is sha256 over unit id + arm id + `checkpoint_sha256` + `input_sha256` + `script_sha256` |
| record which units were reused | `n_reused` is carried in the result structures |

**The strongest evidence is not the code, it is tonight.** The Potts confirmatory run
(`096d0847`, `--resume`, 16 units) is using this same helper right now and has been stamping
units one at a time for ten hours. The machinery is exercised, not just reviewed.

## The outcome space is total and disjoint — audited, no finding

A 60 h run that lands on an *unmatched* outcome records an empty `outcome` at exit 0 and looks
indistinguishable from a successful one, so the sidecar's `[outcomes]` were checked against the
verdicts the driver can actually emit. `_worst` (`laser_sample_dist_confirm.py:765-774`) closes
over exactly `_VERDICT_RANK = {pass, inconclusive, fail}` and **raises `SystemExit` on an
unknown verdict** (`:766-768`) rather than letting it through. So the reachable states map
completely:

| state | declared outcome |
| :-- | :-- |
| `smoke = true` | `smoke` (residual) |
| `¬smoke ∧ ¬controls_all_fail` | `instrument_invalid` (residual) |
| `¬smoke ∧ controls_all_fail ∧ verdict = pass` | `pass` |
| `¬smoke ∧ controls_all_fail ∧ verdict = fail` | `fail` |
| `¬smoke ∧ controls_all_fail ∧ verdict = inconclusive` | `inconclusive` (residual) |

Two properties worth having explicitly. **A smoke cannot be mistaken for the confirmatory
test** — every other outcome requires `smoke = false`, and `[outcomes.smoke]` catches it as a
residual with the reasoning *"A smoke run is not the confirmatory test."* And **the negative
controls gate the verdict**: `pass` and `fail` both require `controls_all_fail = true`, so a run
whose controls did not fail cannot report either, it reports `instrument_invalid`. That is the
spec rule enforced in the grading contract rather than only in prose.

## Measured per-arm cost, and a cheap lever nobody has pulled

A truncated smoke (n=50, `min_p0@1.0`, `103m_1`) was run on titanix 261005 to check plumbing.
**The dual-environment seam works end to end** on this branch: the oracle subprocess spawns,
`--laser-root` resolves, the job handoff works, both upstream arms sampled and stamped. Per-arm
durations from the stamp mtimes:

| arm | n | duration |
| :-- | --: | --: |
| upstream U1 | 50 | ~2 min |
| upstream U2 | 50 | ~2 min |
| **aminx A** | 50 | **~34 min** |
| aminx CTRL_m | 50 | killed at ~12 min, unstamped |

So **aminx arms cost ~17x their upstream counterparts at the same n**, and four arms need
~72 min — the run was cut at three units by a 50 min `timeout` chosen from a wrong estimate
(mine). The two upstream units remain stamped and will be reused on resume.

**The lever: no persistent XLA compilation cache is configured anywhere.** Not in
`laser_sample_dist_confirm.py`, not in `laser_sample_dist_pilot.py`, and no `JAX_*`/`XLA_*`
variable is set in the titanix environment. Every unit runs in its **own
`--aminx-worker` subprocess**, so each pays a full cold compile, which this stack has measured
at 15–20 min (see the orbax/compile memory note). Against a 34 min aminx arm, that is
plausibly the majority of the cost — and it is paid once per unit instead of once per run.

**It IS confined to LASEr — the Potts run rules itself out.** An earlier revision of this
section said the Potts 11.63 h figure might be inflated the same way. That is **withdrawn**.
`_runs("refine@0.3")` returns `("U1", "U2", "A", "CTRL_m")`, so the last two units of each
target block are the aminx arms, and on target 1 those took **6 minutes each** — total,
including process start, weight load, compile and sampling at **n=1000**. A 15–20 min compile
cannot fit inside 6 minutes, so Potts compile is bounded below 6 min per unit: ≤7% of 11.63 h,
and that bound generously counts target 1's whole sampling cost as compile.

**Which sharpens the LASEr question rather than dissolving it.** Per-sample cost differs by
roughly two orders of magnitude between the two families:

| | n | unit wall time | per sample |
| :-- | --: | --: | --: |
| Potts aminx, cheapest target | 1000 | 6 min | **0.36 s** |
| LASEr aminx, smoke | 50 | 34 min | **41 s** |

Only two explanations fit a ~113x gap: LASEr sampling is intrinsically far heavier (rotamer
and chi sampling, a larger model, ligand features — all plausible), or LASEr's compile is
~30 of those 34 minutes.

**It is the first. The compile-dominance hypothesis is withdrawn.** An unregistered spike ran
the same aminx unit twice with `JAX_COMPILATION_CACHE_DIR` set and
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`, into fresh work dirs so `graded_resume` could
not short-circuit and sampling was identical across passes:

```
pass 1 (cold cache)   elapsed 2282 s (38.0 min)   cache entries   0 -> 310
pass 2 (warm cache)   elapsed 2018 s (33.6 min)   cache entries 310 -> 310
```

Compile is therefore **~264 s (4.4 min), about 12% of the unit**; sampling is the other ~88%.
`cache_entries` did not move during pass 2 — zero new compilations — so the cache was fully
effective and the contrast is real rather than a cache that never engaged.

**Consequences.** A persistent compilation cache is a genuine but *modest* lever (~12% per
unit), not the hours-saving one an earlier revision of this section suggested. LASEr's aminx
cost is real compute: ~40 s/sample at n=50 against Potts' 0.36 s/sample. So the ~60 h estimate
stands on sampling, and D4 is a straight "is this worth the CPU" decision with no cheap
optimisation hiding inside it.

**Status of that number: SPIKE, not evidence.** It was run inline without a sidecar, which is
legitimate for an answer but not for a citation. Before it is used to support any claim — in
particular any revised LASEr duration — it must be re-run as a tracked script with a
pre-registered sidecar (`scripts/analysis/` is unscoped, so that can be done without touching
a ledger row). It is recorded here because it *withdraws* a claim of mine, which needs only
enough evidence to stop asserting it.

**Not measured: how sampling scales to n=1000.** The smoke used n=50, and 20x the samples need
not cost 20x the time (batching). Do not multiply 33.6 min by 20 — that is precisely the
extrapolation error this document's sibling plan records three times.

*(An earlier revision of this section called the compile-versus-sampling split unmeasured and
proposed the experiment. It has since been run — see above. The split is ~12% compile / ~88%
sampling, and the proposal is therefore closed rather than pending.)*

## The trap: `script_sha256` is in the cache key

This is correct behaviour — a changed script invalidates its own results, which is what you
want from a cache — but over a 60 h horizon it is a sharp operational constraint:

> **Editing `laser_sample_dist_confirm.py` after launch discards every completed unit.**

Not just the unit in flight. Every unit, because the key changes for all of them. The same
applies to the checkpoint and to any input PDB.

So the pre-launch checklist is short and worth actually following:

1. **Freeze the driver.** Land every intended edit before launch, not during. A one-line fix
   twenty hours in costs the twenty hours.
2. **Pin the checkpoint and the fixtures**; both hashes are in the key.
3. Launch under `bth run` with `--output-paths` registered, so the run grades (a bare `--out`
   records `outcome: unknown` at exit 0).
4. Expect the record to read `status=running, outcome=''` from the moment it starts — the
   cool-tier parquet exists from launch, so its presence is not evidence of completion, and
   the bth wrapper's teardown is not instant. Verify only after the wrapper process exits.

## The smoke is safe to run, but needs an environment that does not exist yet

`--smoke` exists (`:157`) and is narrow — *"smoke runs min_p0@1.0 on 103m_1 only"* (`:204-206`)
— and per the table above it grades as the residual `smoke`, so it cannot pollute the
confirmatory record. It is the right way to check plumbing before committing 60 h.

It was **not** run here, because nothing on titanix currently satisfies all three
requirements at once: xtrax **0.4.0a11** (this branch's floor, after the main merge), the
LASEr upstream at `/home/solab/repos/LASErMPNN`, and a **clean git tree** so the `bth` run is
not recorded `git_dirty`. `aminx-confirm-git` fails the first (pre-0.4.0a11 — it dies at
import with `cannot import name 'ChunkedMap'`), `aminx-ci` has the right xtrax but no pyarrow,
and `aminx-b7i-git` has both but is someone else's checkout at another commit. Provisioning a
venv for this is a few minutes of work, but it is work in service of a run that has not been
approved, so it is left for whoever makes that call.

## The "~60 h" figure is likely low by ~4x, and the arithmetic is now checkable

**This is the single most decision-relevant number in this note.** The ~60 h estimate in
project memory derives from *upstream* throughput (~1 h per n=1000 unit). It appears to price
**every** unit at upstream cost — but tonight measured the aminx arms at **17x upstream**, so
pricing them alike understates the run badly.

Unit count, read from the driver rather than assumed: **2 selectable cells** (`min_p0@0.3` and
`min_p0@1.0`; `min_p0.05@0.3` and `min_p0@0.1` are in `_EXCLUDED_CELLS`) × **5 structures**
(`STRUCTURES`, `:62-77`) × **4 arms** (`U1, U2, A, CTRL_m`) = **40 units**, 20 upstream and
20 aminx, every cell at `n = 1000`.

| arm class | units | measured at n=50 | n=1000, linear | subtotal |
| :-- | --: | --: | --: | --: |
| upstream | 20 | ~2 min | ~0.67 h | ~13 h |
| **aminx** | 20 | **33.6 min** | **~11.2 h** | **~224 h** |
| | | | **total** | **~237 h** |

~237 h is **~10 days serial**, against a recorded estimate of ~60 h. Even restricting to a
single cell (as the smoke did) gives 20 units -> ~119 h.

**Why linear scaling is not merely assumed here.** It is corroborated independently *for the
upstream arm*: 2 min at n=50 scales to ~40 min at n=1000, and project memory records ~1 h per
upstream n=1000 unit. Those agree within the precision of "~1 h", which is what makes this
arithmetic worth acting on rather than dismissing.

**Linearity is now measured, and it holds.** The caveat that used to sit here — "aminx-arm
linearity is not measured, n=50 may be batch-inefficient" — is retired. A second spike ran the
same unit and seed at `n: 200` (a crafted job JSON, 4x the samples, fresh work dir):

```
n=200   elapsed 8046 s (134.1 min), rc=0
n=50    warm 2018 s   (sampling, compile already removed)
```

Subtracting the measured ~264 s compile gives sampling(200) ≈ 7782 s, so the ratio is
**3.86 for 4x the data** — an exponent of **log(3.86)/log(4) = 0.975**, i.e. **linear to within
~2.5%**. √-scaling was excluded along the way: it predicted 67 min and the probe passed that
still running.

Propagating the measured exponent to n=1000:

| | | |
| :-- | :-- | --: |
| aminx unit | 2018 s × 20^0.975 + compile | **~10.5 h** |
| 20 aminx units | | ~210 h |
| 20 upstream units | | ~13 h |
| | **total** | **~223 h ≈ 9.3 days serial** |

That is marginally below the 237 h linear-assumption figure above and of the same order. **The
decision-relevant claim is robust to the difference:** the confirm is multiple hundreds of
hours, not ~60. Treat ~223 h as the planning number.

**Recommendation before approving anything:** treat ~60 h as superseded, and either run that
intermediate-n test or plan against the ~237 h / ~119 h figures. A GPU oracle, previously
framed as a convenience, becomes the central question at this scale — and note the aminx arms,
not the upstream oracle, are what a GPU would have to accelerate.

## The pilot never completed — but 21.68 h of its work is still on disk and reusable

Searching the catalog for `laser_sample_dist` runs turns up something the ~36 h pilot estimate
in project memory does not mention: **no LASEr distributional run has ever completed.**

| run | kind | status | outcome | hours |
| :-- | :-- | :-- | :-- | --: |
| `03d2ef37` | pilot, full | **failed** | error (exit 1) | **21.68** |
| `6a7ec712` | confirm, smoke | completed | `smoke` | 1.21 |
| 4 others | confirm, smoke | failed | error | ≤0.08 |
| `abd2b8a1`, `e0cefaa7` | pilot | completed | `derived` | 0.01 |

So "~36 h" was an estimate, never a measurement — and the one serious attempt burned 21.68 h
and exited 1 without finishing. (Incidentally `6a7ec712` completing with `outcome=smoke`
**empirically confirms** the residual-outcome design audited above from the TOML alone.)

**Why it died is now obvious:** its command was
`--cells min_p0@0.3,min_p0@1.0,min_p0.05@0.3`, and `min_p0.05@0.3` is the cell the confirm
later *refuses* because upstream produces NaNs there (`cf2ea530`). It ran the valid cells, then
hit the broken one.

**And the work survives.** `/tmp/laser_sample_dist_pilot/units/` holds **24 stamped units**, and
`git log 96a148ba..HEAD -- scripts/parity/laser_sample_dist_pilot.py` is **empty** — the script
has not changed since that run, so `script_sha256` still matches and `graded_resume` can reuse
them.

**Inference, labelled as such:** each cell is 2 structures × 6 arms = 12 units, so 24 stamped is
*exactly two cells' worth*, and the run failed on the third. The two runnable cells are
therefore very likely complete. This is arithmetic plus the failure mode, not a direct read —
the unit files key on an opaque `cache_key`, so the cell names are not recoverable from them.

**The cheap test, and the prize.** Re-run the pilot with `--cells min_p0@0.3,min_p0@1.0` and
`--resume` under `bth`. If the inference holds it reports `n_reused = 24`, finishes in minutes,
and yields a **graded pilot result** — a spec deliverable (B5, "pilot → laser_sample_dist")
that has never been obtained. If it instead starts computing, the inference was wrong and it
should be killed rather than left to run for ~134 h.

That asymmetry is why this is worth doing deliberately rather than casually: minutes if right,
and it must be watched if wrong.

## What this note does NOT say

It does not argue for or against launching. The cost (~60 h CPU-only, because the titanix
oracle venv is CPU-only torch), whether a CUDA jaxlib / GPU oracle should be provisioned
first, and the fact that this is **not a gate slug** — so it blocks nothing in the redsox
gate — are all inputs to a decision that remains open. It also does not re-verify the
constants: those were filled at `cf2ea530`, which also refused `min_p0.05@0.3` because
upstream NaNs there.
