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

## What this note does NOT say

It does not argue for or against launching. The cost (~60 h CPU-only, because the titanix
oracle venv is CPU-only torch), whether a CUDA jaxlib / GPU oracle should be provisioned
first, and the fact that this is **not a gate slug** — so it blocks nothing in the redsox
gate — are all inputs to a decision that remains open. It also does not re-verify the
constants: those were filled at `cf2ea530`, which also refused `min_p0.05@0.3` because
upstream NaNs there.
