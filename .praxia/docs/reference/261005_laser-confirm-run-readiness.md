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

### Tested: the inference holds. The pilot's two runnable cells are COMPLETE.

Run untracked with a 30 min cap, `--cells min_p0@0.3,min_p0@1.0 --resume`:

```
n_reused = 24    n_computed = 0    smoke = False
cells_selected = ['min_p0@0.3', 'min_p0@1.0']
all_cells_derived = true   any_instrument_invalid = false   shim_ok = true
missing_cells = []         open_questions = []              n_boot = 2000
```

**Zero units computed** — the 21.68 h is fully recovered, and it finished in seconds. Per cell:

| cell | `delta` | `h_hat` | `q_hat` | `chosen_m` | `escalate` | status |
| :-- | --: | --: | --: | --: | :-- | :-- |
| `min_p0@0.3` | 0.01 | 0.001560551948051945 | 0.00311 | **1.25** | false | ok |
| `min_p0@1.0` | 0.01 | 0.0017875000000000252 | 0.00279 | **1.10** | false | ok |

**Those constants are digit-for-digit identical to the `CELLS` table in
`laser_sample_dist_confirm.py`.** So the confirm's pre-registered `h_hat` and `m` are traceably
derived from *this* pilot output — which `cf2ea530` ("fill the LASEr confirmatory constants")
recorded even though the pilot run itself never completed. The provenance chain closes.

**The m-selection is principled, not arbitrary.** For `min_p0@0.3` the m=1.1 control yields
`delta_neg = 0.0036`, *below* δ=0.01 and therefore undetectable, so the pilot escalated to
m=1.25 (0.0155 > δ). For `min_p0@1.0`, m=1.1 already gives 0.0156 > δ and was kept. The
instrument chose the smallest control it can actually see per cell — which is exactly what a
negative control is for, and it is why the confirm's controls fail as designed.

`shim_check` is clean on both cells (`mean_gap` 4.3e-4 and 2.7e-4 against `h_hat` ~1.6e-3 and
~1.8e-3, `instrument_invalid` false), and chi1 is derived over 252 positions per cell with
controls ordered correctly above the U2/U3-vs-U1 baselines.

### Spec row B5 is DONE: graded pilot record `9d621aee`, obtained in 32 seconds

The bookkeeping is now closed too. Verified by record, not by exit code:

```
run 9d621aee-7c96-43f9-9c1d-3d755adc8e54
status=completed   outcome=derived   outcome_is_residual=False   exit_code=0
git_dirty=False    git_hash=0e642d44 (aminx-b7i-git, clean)      duration_s=32.3
sidecar=scripts/parity/laser_sample_dist_pilot.bth.toml (9e05febf…)
output_paths=['outputs/laser_sample_dist_pilot/0e642d44/result.json']
```

The artifact resolves and matches the untracked probe exactly (`n_reused 24`, `n_computed 0`,
`all_cells_derived`, `shim_ok`, both cells' `h_hat`/`chosen_m`). `outcome=derived` with
`is_residual=False` means it matched a **declared** outcome — deriving the confirm's constants
is what the pilot is *for*, so that is its success state, not a fallthrough.

**32 seconds, against the 21.68 h that run `03d2ef37` burned before exiting 1.** The pilot's
only real failure was reaching the third cell, `min_p0.05@0.3`, since excluded for upstream
NaNs.

### Three traps that made this take four attempts — worth knowing before the re-wave

1. **`uv` is not on PATH in a non-login ssh shell, and bth's documented wrapped form needs
   it.** `bth run … -- uv run --no-sync python3 script.py` fails with `exit_code: 1` and **no
   script error at all**, because `uv` is never found and the script never executes. `bth`
   itself has the same problem (`/home/solab/.local/bin/bth`). Fix:
   `PATH=/home/solab/.local/bin:$PATH`. Three runs failed this way before the cause was
   isolated by comparing a direct `.venv/bin/python3` run (which succeeded) against
   `uv run` (which could not exec).
2. **A fast-failing bth run leaves its cool-tier record stuck at
   `status=running, outcome='', exit_code=-1, duration_s=0`** — indistinguishable from an
   in-flight run, *permanently*, after the wrapper has exited. The older pilot `03d2ef37`
   recorded `failed/error/exit 1` correctly, so bth can do this; it appears specific to
   failures that occur almost immediately. **This undercuts "verify by record" precisely where
   it is most needed**, so cross-check with `ps` that no wrapper is alive before reading a
   `running` row as in-flight.

   **It is not a one-off: the aminx catalog holds NINE such phantom rows**, spanning
   2026-10-01 to 2026-10-06 across five vehicles — `laser_decode_e2e` (`068b027c`),
   `laser_proofread_parity` (`4ae3a093`, `86fb61a3`), `potts_ar_refine_exact` (`fd0bafb7`,
   `c823350b`), `run_gate.py` (`e0537910`), and tonight's three
   `laser_sample_dist_pilot` PATH failures (`3cab0e8c`, `ced279d9`, `48fa3fac`). So anyone
   querying the catalog for in-flight work sees nine runs that died days ago. Worth knowing
   before the re-wave, whose own gate run could add a tenth.

   **Do not reach for `bth repair`.** `bth repair --dry-run --tier cool` on titanix reports
   `action_count: 0` — bathos does not consider these repairable, because they are *valid*
   records of a terminal state that was never written, not corruption. There is no tooling fix;
   the remedy is procedural (check `ps`). Worth stating because the instinct is to try a repair,
   and `--force-rebuild` has wiped a warm table before.

   Separately, three `potts_energy_parity` runs from 2026-09-30 (`35f9a257`, `6b2b3110`,
   `755271fb`) show `completed / exit 0 / output_paths=None` and **no outcome** — they
   succeeded but never graded, because nothing was registered for bth to grade from (the
   `--out`-without-`--output-paths` trap). They are harmless only because `06f329d4` later
   graded that slug `pass`; the lesson is that an ungraded success looks identical to an
   ungraded nothing.
3. **Debt #2483, reproduced in practice.** `aminx-confirm-git`'s venv lacks the `laser` extra,
   and the failure is a bare `ModuleNotFoundError: No module named 'prody'` naming neither the
   family nor the extra. Exactly the defect #2483 describes — previously documented from
   reading `pyproject.toml`, now observed.

## What this note does NOT say

It does not argue for or against launching. The cost (~60 h CPU-only, because the titanix
oracle venv is CPU-only torch), whether a CUDA jaxlib / GPU oracle should be provisioned
first, and the fact that this is **not a gate slug** — so it blocks nothing in the redsox
gate — are all inputs to a decision that remains open. It also does not re-verify the
constants: those were filled at `cf2ea530`, which also refused `min_p0.05@0.3` because
upstream NaNs there.

## Where the ~223 h actually goes: one hypothesis falsified, one real knob found

Asked 2026-10-06: *why does it take so long, have we profiled it, and can Engaging do it
instead?* The honest answer to the middle question was **no** — the split was profiled at the
*unit* level (~12% compile / ~88% sampling, §"Measured per-arm cost") but nothing had ever
looked inside sampling. The doc's own explanation, "LASEr sampling is intrinsically far
heavier (rotamer and chi sampling, a larger model, ligand features — all plausible)", is a
hypothesis dressed as a conclusion. Two probes follow.

### Falsified: the planner does NOT serialise the sample axis on CPU

`host/plan.py:281-284` reads the memory budget from `jax.devices()[0].memory_stats()` and
falls back to **4 GiB** on exception. CPU devices return `None` there, and
`N_SAMPLES.default_batch_size` is **1** (`tiling/axes.py:64-70`), so a demotion from `Vmap` to
`ChunkedMap` would have meant **one sample at a time** — which would explain the whole 113x
gap against Potts. `plan.py:306-320` records that this exact bug was found once before on the
legacy dispatch path (EPIC #1541 T-PLANNER.2), which made it a live suspicion rather than a
guess.

Probed (`.praxia/spikes/261006_sample_axis_plan_probe.py`, local CPU, throwaway):

```
memory_stats() RAISED TypeError: 'NoneType' object is not subscriptable
  -> plan.py falls back to limit = 4 GiB

 seq_len  n_samples     strategy  batch_size  safe_map
      76       1000         Vmap           1         0
     120       1000         Vmap           1         0
     240       1000   ChunkedMap           1         1

reasoning @ seq_len=120, n_samples=1000:
  joint-budget: Vmap retained (final estimate 2022000000 B <= budget 3435973836 B)
```

So the CPU fallback is **real** (4 GiB, confirmed) but **not harmful at these sizes**: `Vmap`
is retained all the way to n=1000 for seq_len ≤ 120, and only seq_len 240 at n=1000 demotes.
**The hypothesis is dead.** Worth recording precisely because it was the attractive
explanation — a planner artifact would have meant the ~223 h was nearly free to fix.

### The real cap is the confirm script's own `samples_chunk_size=8`

`laser_sample_dist_confirm.py:557` sets `samples_chunk_size=8`. Per
`resolve_chunk_size` (`plan.py:450-454`), that value wins outright, and **without** it the
chunk defaults to the full `total_num_samples`. Since the planner would retain `Vmap` at 1000
for these structures, the script is choosing a vectorisation width of **8** where the budget
permits ~1000 — i.e. **125 sequential dispatches** at n=1000 instead of one.

**What this does and does not establish.** It establishes that the width is a script-level
constant, not an irreducible property of the model — which is what "have we identified the
bottleneck" was really asking. It does **not** establish that widening it is faster: CPU `vmap`
is memory-bandwidth-bound and a 125x wider batch at seq_len 120 is estimated at ~2.0 GB of
activations, close enough to the 3.4 GB CPU budget that the planner's own margin is thin. The
chunk also bounds the blast radius of a failure, which is a deliberate design property per the
preemption-safety rule, so widening it trades recoverability for throughput.

**Unmeasured, and the next thing to measure:** wall time per sample as a function of
`samples_chunk_size` (8 / 32 / 128) at small n on one structure. If per-sample cost falls with
width, the cost is dispatch-bound and the fix is a knob; if it is flat, the cost is real
per-sample compute and only different hardware helps. That is a ~1 h experiment that decides
whether to provision anything at all, and it must be run under a sidecar before any number
from it is cited.

### Engaging: the right instinct, but it buys parallelism more than speed

The per-arm cost is ~88% sampling on a **CPU-only jaxlib** — so GPU is exactly the lever the
estimate is begging for, and the question was well aimed. Two caveats and four prerequisites.

The caveats: GPU helps only to the extent the work is actually GPU-shaped, and at
`samples_chunk_size=8` the device would sit mostly idle — so **the chunk-width measurement
above should come first**, or a GPU run reproduces the CPU run's serialisation on more
expensive hardware. And the upstream arms are the *cheap* 1/17th of the bill; moving them
buys almost nothing.

The bigger structural win is not speed but **shape**: 40 units that are independent by
construction, and `graded_resume` already persists and reuses per unit (proven — run
`9d621aee` reused 24 of 24). A 40-way job array turns ~223 h serial into ~1 unit of wall
clock, which also makes the **12 h MIT partition cap** a non-issue per unit (currently each
aminx unit is projected at ~10.5 h, uncomfortably close to it on CPU).

Prerequisites, none of which are verified yet:

1. **A CUDA jaxlib** in an aminx venv on Engaging. This whole estimate exists because neither
   titanix nor the oracle venv has one.
2. **Pre-staged weights.** Compute nodes have no outbound internet, and aminx resolves
   checkpoints from the Hub (`HF_REVISION`). The fetch belongs on `mit_data_transfer`, not a
   login node, and is a separate job from the run.
3. **The upstream oracle** needs `torch` + `prody` (debt #2483). Engaging's aminx checkout is
   also known to receive only hand-transferred files, not a full sync.
4. **`XLA_FLAGS=--xla_gpu_shard_autotuning=false`** is mandatory on `node4007`/`node4008`
   (Blackwell SM120) — a 1170x difference, keyed on hostname.

So: Engaging is plausibly the right venue, and the array shape is a genuine improvement over
anything titanix can offer. But provisioning it is several hours of work resting on an
unmeasured assumption, and the chunk-width probe costs ~1 h and could change the target.
**Measure first.**

### A cheaper GPU path than Engaging: titanix already has the GPUs

Measured 2026-10-06, from jax's own startup warning in the new pin-compliant venv:

```
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed.
Falling back to cpu.
...
jax 0.10.2  [CpuDevice(id=0)]
```

So the CPU-only constraint behind the ~223 h is **purely a missing wheel**, not absent
hardware. titanix has NVIDIA GPUs; GPUs 0 and 1 hold vLLM, which leaves 2 and 3. That makes
"install a CUDA jaxlib on titanix" a materially cheaper experiment than provisioning Engaging:
it needs no DTN fetch job, no pre-staged weights, no second checkout, no SM120 flag, and no
12 h walltime chunking — and it can be tried in one venv without touching the one the vehicles
use.

**It is not a substitute for Engaging, and it is not free.** titanix gives at most two free
GPUs against Engaging's 40-way array, so the array shape remains the better answer for the
full confirm. And a CUDA jaxlib changes the numerical environment, which matters for a
*parity* project: any ledger row re-measured on GPU is re-measured on a different backend, so
this belongs on a measurement venv first, never on the vehicles' venv, and never during the
freeze.

Ordering, then, is: chunk-width measurement (~2.2 h, decides whether width is the lever) ->
CUDA jaxlib on a titanix measurement venv (cheap, bounds the GPU speedup on 2 GPUs) ->
Engaging provisioning only if the confirm is still the bottleneck. Each step is cheap enough
to refute the next one's premise.

### REFUTED: max_length padding is inert on the LASEr path, so the ~10x framing was wrong

I claimed, confidently, that the ~223 h was inflated ~10x because the confirm never sets
`max_length` and so inherits the 512 default against a 154-residue structure, citing
`host/runner.py:127-131`'s own cost model ("Autoregressive decode cost grows roughly with the
square of the padded length"). **That is wrong for LASEr.** Three direct probes, 2026-10-06:

1. **The featurizer emits real-length arrays.** `families/laser_mpnn/featurize.featurize` on
   `103m_1.pdb` returns `chain_mask (154,)` and `sequence_indices (154,)` — not `(512,)`.
2. **`l_pad` is a Potts-only mechanism.** It appears solely in
   `families/potts_mpnn/featurize.py:505-544` and `potts_mpnn/sample_host.py:217-235`. Nothing
   under `families/laser_mpnn/` references `l_pad` or `max_length` at all.
3. **The LASEr batch takes its length from the features**, not from a padding target:
   `sample_host.py:165`, `length = int(features.sequence_indices.shape[0])`.

So `max_length` does not govern this path, and the `(512/160)^2 = 10.24` figure describes a
configuration LASEr never enters.

**The absent warning was the clue I misread.** The first completed unit recorded
`padding_warning_seen=False`. `_PADDING_WARN_RATIO` is 2.0 and 512 > 2x154, so I initially
read the silence as a gap in aminx's warning coverage. It is the opposite: `_PaddingCheck`
measures the span of the mask it is *given*, that mask is length 154, so padded == real,
ratio 1.0, and the check correctly says nothing. **aminx was right and my inference was
wrong** — the warning's silence was evidence against my hypothesis, not evidence of a missing
warning.

**Why this still matters for #2358.** That debt (~86 s/sample for a 93-residue chain at the
default) is real, but it is about the **host loader's** padded path used by the
ProteinMPNN/LigandMPNN/Potts families — not LASEr. Sizing `max_length` remains a genuine
lever for *those* runs and is simply not available here.

**What this leaves.** Four candidate explanations for LASEr's ~50 s/sample at L=154 against
Potts' 0.36 s/sample, two now eliminated by direct probe:

| explanation | status |
| :-- | :-- |
| the planner demotes the sample axis to width 1 on a CPU 4 GiB budget | **falsified** — `Vmap` retained to n=1000 |
| the confirm pads 154 residues to `max_length=512` | **falsified** — `max_length` inert on this path |
| `samples_chunk_size=8` caps vectorisation at 8 of a permitted ~1000 | **measuring** (run at `a4bfe86c`) |
| LASEr sampling is genuinely heavier: rotamer and chi sampling, larger model, ligand features | surviving, and now better supported by elimination |

The surviving explanation is the readiness doc's original one. It is in a stronger position
than when it was written, not because new evidence arrived for it, but because two rivals were
tested and failed. The padding pair is being left in the running measurement so the refutation
is a **recorded null** rather than my assertion.

### Incidental, and it corroborates debt #2321

Every completed unit's captured warnings include, twice:

```
Explicitly requested dtype float64 requested in astype is not available, and will be
truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option
```

Debt #2321 says `categorical_draw`'s f64 CDF "is inert in production" and "emits a JAX
UserWarning on every trace of the sampler". This is that warning, observed on the **LASEr**
sampling path with `jax_enable_x64` explicitly off, which is what production uses. So #2321 is
live here and not only on the path it was filed against. It is not a cost finding — a truncated
`astype` is cheap — but it is independent confirmation that the f64 CDF the parity waves
validate is not the code production runs.

### #2481 diagnosed from the pilot's own artifact: the chi1 control cannot fail, and why

The item said the chi1 negative control "cannot fail at any pre-registered m, so that channel
is unvalidated", and flagged it as something to decide *before* spending oracle time. It is
now quantified, from the pilot's own record rather than a new run — run `9d621aee`
(`completed / derived / exit 0`, `git_hash 0e642d44`),
`outputs/laser_sample_dist_pilot/0e642d44/result.json`, `cells['min_p0@0.3']['chi1']`.
252 positions, n = 1000, `chosen_m = 1.25`:

| quantity | value | Δ against the U2/U1 baseline |
| :-- | --: | --: |
| `delta_chi` (threshold) | **0.05** | — |
| `mean_d_u2_u1` (baseline) | 0.013714 | — |
| control, m = 1.10 | 0.014297 | 0.00058 |
| control, m = 1.25 | 0.023939 | 0.01023 |
| control, m = 1.50 | 0.041489 | **0.02778** |

A control grades `fail` only when the **90% CI lower bound** on Δ exceeds `delta_chi`. The
largest pre-registered m yields a *point* Δ of 0.0278 — **56% of the threshold** — and even its
raw distance, 0.0415, sits below 0.05. So no m in {1.10, 1.25, 1.50} can make this control
fail. At the m actually chosen, 1.25, it sits at **20% of threshold**. **#2481 is confirmed
exactly as written.**

**The cause is NOT an unread knob, and I checked that first because it would have been tidy.**
`LaserOptions.chi_temp` is what the confirm scales for `CTRL_m`, and debt #2435 listed
`chi_temp` among eight fields "declared but never read" — which would have made the control
structurally inert. Both halves of that are wrong: `chi_temp` **is** read
(`laser_mpnn/sample_host.py:499,575` → `model/laser/tied.py:243`,
`softmax(stored / chi_temperature)`), and #2435's dead-field claim is stale for all eight
fields (plan §2f). So this item must **not** be attributed to #2435.

**The cause is channel insensitivity against an asymmetric threshold.** Scaling temperature by
1.5× does move the chi1 distance — 0.0137 → 0.0415, about 3× — so the knob is live; it just
does not move it far enough. Meanwhile `delta_chi = 0.05` is **five times** the sequence
channel's `delta = 0.01`, while the chi1 distance responds *less* to temperature than the
sequence distance does. Two compounding reasons it is hard to shift, both in
`scripts/parity/sample_dist_stats.py:366-405`: each position is restricted to samples whose
amino acid equals upstream's **modal** amino acid, and chi1 is binned into **36 bins of 10°**,
so any sub-bin angular shift is absorbed entirely.

**What would fix it — a spec decision, not a code fix.** The measured trend is roughly a
doubling of the control distance per +0.25 in m (0.0143, 0.0239, 0.0415), so m ≈ 1.75–2.0
extrapolates to a distance of ~0.07–0.09 and a Δ of ~0.056–0.076, which **would** clear 0.05.
In the order that best preserves the pre-registration:

1. **Add a chi1-specific control m of about 2.0**, leaving the sequence channel's m untouched.
2. Use a channel-specific m throughout.
3. Revisit `delta_chi` — but that changes the **pass** criterion for the real arm too, so it is
   not a free move and should be last.

**The extrapolation is from three points and must be confirmed before it is relied on.** The
cost of confirming it is small: one extra control arm is **one aminx unit per structure**, not a
re-run, and `graded_resume` reuses everything else — `9d621aee` reused 24 of 24 units in 32 s.
So this converts "decide before spending the oracle time" into "run one cheap arm, then
decide", which is a strictly better position than the item was filed in.

### Chunk-width run halted at 5/16 units: the host became unmeasurable, and the warm-up numbers are confounded

**Halted deliberately 2026-10-06, not failed.** titanix's 1-minute load average went
15 → 32 → 72 → 107 on 20 cores while the run was in flight. Three heavy jobs had converged on
the box: another session's `staging_rooting_invariance_stage2.py` (125 threads, 1.5 h elapsed),
a Rust `sccache` build (118 threads), and my own worker plus its `bth` wrapper (~210 threads
between them). At 3.6x oversubscription and rising, this measurement **cannot** satisfy its own
load-stability gate (max/min of the 1-minute load within a pair instance <= 1.50), so
continuing would have spent ~3 h to earn an `instrument_unverified` verdict while degrading two
other sessions' work. Only my own processes were killed; the staging job and `sccache` were left
alone, and the load that remains is theirs.

**What the five completed units support, and what they do NOT.**

| unit | n | width | pad | s/sample |
| :-- | --: | --: | --: | --: |
| `ref_n8_w8_p512__warm` | 8 | 8 | 512 | 50.53 |
| `fit_n8_w8_p160__warm` | 8 | 8 | 160 | 48.74 |
| `wide_n32_w32_p512__warm` | 32 | 32 | 512 | 50.15 |
| `ctrl_n8_w1_p512__warm` | 1 | 1 | 512 | 89.36 |
| `ref_n8_w8_p512__paddingr0` | 8 | 8 | 512 | 55.92 |

**VALID: the padding null.** `ref_n8_w8_p512__warm` against `fit_n8_w8_p160__warm` is a clean
comparison — same n, same width, both cold — and gives **50.53 vs 48.74, ratio 0.965**. That
independently confirms by measurement what §"REFUTED" established by code reading: `max_length`
is inert on the LASEr path.

**NOT VALID: any width conclusion from these rows.** Every one is a **warm-up**, which the
design deliberately excludes from grading *because warm-ups carry the cold compile* — and they
carry it over different sample counts. `ctrl_n8_w1_p512__warm` ran at **n = 1**, so its single
sample absorbs an entire cold compile, while the w8 row amortises its compile over 8. Solving
`8p + C = 404.2` against `1p' + C' = 89.36` is underdetermined without C. So the apparent
"width 1 is 1.77x worse than width 8" is **confounded by compile amortisation and must not be
quoted.** I stated it as a finding in an earlier status; that was wrong.

The valid width comparison is the **graded** `control` pair, which runs *both* members at
n = 8 after a per-width warm-up has already paid each shape's compile. That pair had not run
when the host was halted.

**Re-run conditions, so the next attempt is not wasted.** A fresh `--work-dir` is required
rather than resuming: the completed units have their observed loads baked into their bodies, and
reusing a unit measured at load 15 beside one measured at load 70 is precisely what the
stability gate exists to reject. Check `/proc/loadavg` and `ps --sort=-nlwp` first; the run
wants a box near idle, and it is ~3.5 h of wall clock at the ~50 s/sample that every
configuration here agrees on.

**The one durable number.** Per-sample cost is **~50 s at L = 154** on CPU, consistent across
three independent configurations (50.53, 50.15, 48.74) and a fourth under heavier load (55.92).
Against Potts' 0.36 s/sample that is the ~113x gap the whole investigation is about, and it is
now the only cost figure here not resting on a refuted mechanism.

### The load-stability gate measures my own worker's startup, which is an instrument flaw

Observed 2026-10-06 on the relaunched run, at graded padding instance r0:

```
ref_n8_w8_p512__paddingr0   52.68 s/sample   load 12.6 -> 22.6
fit_n8_w8_p160__paddingr0   57.61 s/sample   load 22.7 -> 19.1
```

max/min over that instance's four load samples = **22.7 / 12.6 = 1.80**, against the
pre-registered ceiling of 1.50. So this pair instance fails the stability gate, and because
`_pair_ratio` ANDs stability across a pair's instances, one bad instance forces
`instrument_unverified` for the whole run.

**The drift is mostly mine.** A unit is a fresh subprocess that spins up ~100 JAX threads. The
`before` sample is taken at the very start of the timed call — while the *previous* unit's
process is still exiting and this one's threads are still spinning up — and the `after` sample
once it is fully warm. So the gate is partly measuring **my own process lifecycle**, not
external contention. The external load was steady at 15-19 throughout (`/proc/loadavg` was
15.94 at the time of reading, and the box's other tenant has been stable for an hour).

**I am not touching the threshold.** Loosening a pre-registered band because it is inconvenient
is the exact failure this project's rules exist to prevent, and the band was written down
before any graded number existed. The run continues to completion and will report whatever it
reports.

**But the instrument needs a revision before the next attempt**, and the fix is not the
threshold:

1. **Sample load as a median over the unit's duration**, not two endpoints — a background
   sampler thread, or simply read `/proc/loadavg` at intervals and keep the median. Two
   endpoints on a 7-minute unit are a tiny, badly-timed sample.
2. **Or exclude self-load**: compare against the load attributable to *other* processes (total
   minus this process's own runnable threads), which is what the gate is actually trying to
   control for.
3. **Or warm the process before the first load sample** — take `before` after the model has
   loaded and threads have settled, immediately prior to the timed region, rather than at its
   start.

Option 3 is the smallest change and addresses the observed mechanism directly. Option 1 is the
most honest about what "load during this unit" means. Either way this is a revision to **how
the covariate is sampled**, not to the criterion it feeds, so it does not touch the verdict
bands.

**What survives regardless.** The ratios are still recorded per unit, so the padding and width
numbers remain readable with the load caveat attached — clearly labelled as *not having cleared
the pre-registered gate*. And the padding null now has three independent readings pointing the
same way (0.965 and 0.962 from warm-up pairs on two different hosts' load regimes, plus
whatever the graded instances give), which is a much stronger position than the single
measurement the question started from.

### The padding pair is COMPLETE and grades `padding_irrelevant`, with a gate-clearing instance

Both graded instances finished 2026-10-06:

| instance | `ref` p512 | `fit` p160 | ratio fit/ref | load spread | stability gate |
| :-- | --: | --: | --: | --: | :-- |
| r0 | 52.68 | 57.61 | 1.094 | 1.80 | **fails** (>1.50) |
| **r1** (order reversed) | **44.43** | **45.05** | **1.014** | **1.17** | **passes** |

Median ratio **1.054**, which is >= 0.90, so the pre-registered band grades
**`padding_irrelevant`**. The 160-padded configuration is, if anything, marginally *slower* —
exactly the noise you expect around a true null.

**And r1 clears the stability gate on its own**, at a load spread of 1.17 against the 1.50
ceiling. That matters: it means the self-load problem recorded above is **not** a permanent bar
to a valid instance — r1 is a fully valid measurement by this run's own criteria, taken with
the member order reversed, which is also the arm where a monotone drift would have shown up
with the opposite sign. It did not.

**So the padding question is settled, four ways:**

1. Code reading — `max_length` never reaches the LASEr path (`l_pad` is Potts-only;
   `sample_host.py:165` takes length from the features).
2. Shape probe — `featurize` returns `(154,)`, not `(512,)`.
3. Warm-up pairs — ratio 0.965 and 0.962, under two different load regimes on two hosts.
4. **Graded pairs — median 1.054, with a gate-clearing instance at 1.014.**

The ~10x padding hypothesis I asserted earlier today is dead by measurement as well as by
reading, and `_PaddingCheck`'s silence was correct all along.

**Still open: the lever.** The `control` and `lever` pairs had not run when the host went to
load 69 again (another session started a 87-thread spike on top of the long-running staging
job). The run continues; if the lever's instances land under drifting load they will fail the
same gate r0 failed, and the ratios will be reported with that caveat rather than as a verdict.

### The CUDA-jaxlib probe, specified concretely (measured 2026-10-06, not yet run)

Since the chunk-width lever is looking like a ~25% effect and padding is a null, **hardware is
the only remaining lever** on the ~223 h. The cheap probe is titanix itself, and the hardware
facts are now measured rather than assumed:

```
index, name,              memory.total, memory.used, driver_version
0,     NVIDIA TITAN RTX,  24576 MiB,    21580 MiB,   595.84     <- vLLM
1,     NVIDIA TITAN RTX,  24576 MiB,    21580 MiB,   595.84     <- vLLM
2,     NVIDIA TITAN RTX,  24576 MiB,        5 MiB,   595.84     <- FREE
3,     NVIDIA TITAN RTX,  24576 MiB,       38 MiB,   595.84     <- FREE
CUDA Version: 13.2
```

**Two full 24 GB cards are idle.** 24 GB is ample for L = 154, and the driver (595.84, CUDA
13.2) is newer than any CUDA 12 wheel needs, so forward compatibility covers
`jax[cuda12]` at the pinned `jax>=0.10.2,<0.11`.

**Four constraints, each of which would invalidate the probe if ignored.**

1. **Pin to GPUs 2 and 3.** `CUDA_VISIBLE_DEVICES=2,3`. GPUs 0 and 1 hold vLLM at 21.5 GB
   each — a run that lands there competes with another service, and that service is not mine
   to disturb.
2. **A separate venv, never the vehicles'.** A CUDA jaxlib changes the numerical backend, and
   this is a parity project under a freeze. The measurement venv gets it; nothing that writes
   a ledger row does.
3. **TITAN RTX is Turing, compute capability 7.5, and its f64 throughput is 1/32 of f32.** So
   this hardware is a reasonable probe for **f32 production sampling** — which is what the
   distributional confirm does — and a *poor* choice for the **f64 parity tiers**, which would
   likely run slower than on CPU. Do not generalise a GPU speedup measured here to the f64
   waves. This is the detail most likely to be missed, because "we have GPUs now" invites
   exactly that generalisation.
4. **It measures a ceiling, not the plan.** Two cards against Engaging's 40-way array: even a
   large per-unit speedup leaves the array the better shape for the full 40-unit confirm. The
   probe's job is to size the speedup cheaply, so the Engaging provisioning decision rests on
   a number instead of a hope.

**The probe itself:** one LASEr unit at n = 8, width 8, the same structure and checkpoint this
script already pins, timed the same way — directly comparable against the 44–60 s/sample this
run has measured across ten units on CPU. Roughly 30 minutes including the venv build, against
the ~223 h it is deciding about.

**Do it after the current run finishes, not alongside it** — a GPU run still consumes host CPU
for dispatch, and the timing measurement in flight has already lost one attempt to contention.

### THE CONTROL ARM FAILS, so the chunk-width lever is NOT established — retracting the ~25%

The validity gate has landed and it does not clear. This retracts the width number I reported
twice today.

| pair | members | ratio | pre-registered band | verdict |
| :-- | :-- | --: | :-- | :-- |
| **control** | `ctrl_n8_w1` 53.04 vs `ctrl_n8_w8` 52.33 | **1.014** | **≥ 1.20** | **FAILS** |
| lever r0 | `wide_n32_w32` 45.12 vs `base_n32_w8` 60.25 | 0.749 | ≤0.70 bound / ≥0.90 compute | — |
| lever r1 (reversed) | `wide_n32_w32` 43.65 vs `base_n32_w8` 50.94 | 0.857 | — | — |

**What the control was for.** Width 1 at n=8 runs eight single-sample chunks where width 8 runs
one chunk of eight. If per-chunk dispatch were an appreciable share of the cost, width 1 had to
be visibly worse — the band asked only for 1.20×, far less than the 1.77× I once wrongly quoted
off a warm-up row. It came in at **1.014: no detectable difference at all.** Dispatch overhead on
this path is negligible, which is a clean negative result and refutes the mechanism the whole
lever hypothesis rested on.

**And the lever's effect is the same size as its own replicate noise.** The two instances of the
*identical* configuration `base_n32_w8_p512` measured **60.25 and 50.94 s/sample — an 18%
spread within one configuration.** The lever's apparent effect is ~20% (median ratio 0.803). An
effect indistinguishable in magnitude from the within-configuration spread is not a measured
effect. The `wide` arm, by contrast, reproduced tightly (45.12 / 43.65, 3.3%), so the instability
is concentrated in the base arm — consistent with the load-drift mechanism recorded two sections
above, and not with width.

**So both candidate levers are now closed by measurement:**

| lever | status | evidence |
| :-- | :-- | :-- |
| planner serialisation | falsified | code reading; `samples_chunk_size` never reaches the path |
| `max_length` padding | falsified | four independent lines, median ratio 1.054 |
| chunk width | **not established** | control fails at 1.014 vs ≥1.20; effect ≈ replicate noise |

**The surviving explanation is the original one, now measured rather than assumed: LASEr sampling
is simply heavy — ~44–60 s/sample at L=154 on CPU-only jaxlib, across fourteen units.** The
~223 h estimate stands with no software lever against it, which makes **hardware the only
remaining lever** and the CUDA probe specified above the next thing worth running.

**I am reporting the control as a failure rather than reinterpreting it.** The honest reading of
a failed validity gate is that the lever arm is uninterpretable, not that the lever arm is the
real result and the control is noise. Had the control been the one at 0.80 and the lever at 1.01
I would have had to say the same thing, which is the test of whether the band was a real
commitment.

**Final verdict still comes from the instrument, not from this arithmetic.** The linearity pair
was still running when this was written; the run writes its own `results.json` and grades itself
against the bands in `laser_sample_chunk_width_cost.bth.toml`. Expect `instrument_unverified`
overall, because padding r0's load spread of 1.80 exceeded the pre-registered 1.50 ceiling and
`_pair_ratio` ANDs stability across a pair's instances. The ratios above are per-unit facts and
stand on their own; the graded verdict is the record's to issue.
