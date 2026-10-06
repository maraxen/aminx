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

**This is not confined to LASEr.** Tonight's Potts confirmatory run (`096d0847`, 11.63 h) also
ran its 16 units as separate subprocesses with no persistent cache. So the measured 11.63 h
re-run price in `plans/261005_rewave-composition-scoped-batch.md` §3 may be materially
inflated by the same repeated compilation.

**What is NOT measured, and must be before anyone quotes a saving:** the compile-versus-sampling
split. The case for compile dominance is indirect (this stack's 15–20 min cold-compile figure,
plus ~11-core parallelism consistent with XLA compilation). The clean experiment is two runs of
a single aminx unit with `jax_compilation_cache_dir` set — the first pays compile, the second
reuses it, and the difference *is* the compile cost. That is a ~1 h measurement that could
retire several hours from both estimates, and it should be run before committing to either the
60 h LASEr confirm or the 11.63 h re-wave.

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

## What this note does NOT say

It does not argue for or against launching. The cost (~60 h CPU-only, because the titanix
oracle venv is CPU-only torch), whether a CUDA jaxlib / GPU oracle should be provisioned
first, and the fact that this is **not a gate slug** — so it blocks nothing in the redsox
gate — are all inputs to a decision that remains open. It also does not re-verify the
constants: those were filled at `cf2ea530`, which also refused `min_p0.05@0.3` because
upstream NaNs there.
