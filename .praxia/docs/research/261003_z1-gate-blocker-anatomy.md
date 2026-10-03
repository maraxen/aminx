---
title: 'Z1 gate blockers: what is measured, what is decision debt'
description: The redsox step-2 knob blocker reduces to 3 fields of decision debt and 0 of test debt; every other condition of test_parity_ids_passed is measured empty
status: draft
task_id: 260929_potts-laser-xtrax-compose
date: '261003'
confidence: measured except where marked
sources: 'bathos fa2fac73-0a38-4e72-a2b3-5e596221d55b; titanix b7i pytest run'
---
# Z1 gate blockers: what is measured, what is decision debt

`step2_passed = False` came from `test_parity_ids_passed`
(`tests/knob_gate/test_knob_superset.py:114-156`), which fails on any of three
independent conditions. The assertion messages name the offending rows but not
*which* condition fired nor why, and the three have incompatible remedies. This
doc resolves all three so the next gate run is not a guess.

**Bottom line: the step-2 knob blocker is 3 fields of decision debt and 0 of
test debt.** There is no test left to write for it. It is waiting on a scope
call that is not mine to make.

## The three conditions, resolved

| Condition | Result | Basis |
| :-- | :-- | :-- |
| `unwired` — live rows naming no test | **0 of 212** | computed off `alias_map.toml` |
| `not_passed` — a named test that did not pass | **0 of 68** | see below |
| `uncovered` — new fields no prefixed test reaches | **3 of 51** | bathos `fa2fac73` |

`test_superset`'s own `unmapped` check (live rows with empty `targets`) is also
**0 of 212**.

### `not_passed` is empty, in two parts

The 212 live rows name 68 distinct nodeids. Both halves of "did it pass" check out:

- **They exist.** Every one of the 68 resolves to a real `def` in a real file —
  no stale or renamed id. None is parametrized, so no bracketed id can be
  missing a param that no longer exists.
- **They pass.** The 68 live in exactly 5 files under `tests/knob_semantics/`.
  Running all 5 on titanix b7i (CPU, `JAX_PLATFORMS=cpu`, `-o addopts=""`):
  **87 passed, 0 failed, 659.90s**. 87 is a superset of the 68, and nothing in
  those files failed or was skipped, so all 68 passed.
- **They will be collected.** This is the trap the assertion message warns
  about — an id absent from the outcomes file counts as *not passed*, so a test
  that passes standalone but is never collected still fails the gate.
  `gate_ids.py:26` sets `_COLLECT_ROOTS = ("tests/knob_semantics", "tests/port",
  "tests/golden")` and collects with `-o addopts=""` specifically so the
  pyproject `parity_heavy` deselect cannot shrink the list. All 68 are under
  `tests/knob_semantics/`, so all 68 are collected.

*Caveat, stated because it is not measured:* the authority is the gate's own
outcomes file, and the gate runs these in one session where fixture or ordering
interaction could in principle differ from this run. Collection is assured;
passing is measured at this tree, not inside a gate session.

### `uncovered` is 3 fields, each structural

Tracked run `fa2fac73-0a38-4e72-a2b3-5e596221d55b`, `outcome = pass`, read off
the cool-tier parquet (`status` completed, `outcome_is_residual` false,
`outcome_error_reason` empty, `adversarial_check_status` present). `git_hash`
`bf67a559` is the commit carrying the sidecar, so the pre-registration provably
precedes the run. Script and sidecar: `scripts/redsox/knob_coverage_audit.py`,
`knob_coverage_audit.bth.toml`. Results: `scripts/redsox/results/`.

**48 of 51 `NEW_FIELDS` are reached** by a `test_knob_semantics_*` nodeid named
on a live row. Three are not:

| Field | Mechanism | Rows naming it |
| :-- | :-- | :-- |
| `emit_dense_hJ` | `NO_REFERENCE_FIELD` | 0, live or exclusion |
| `emit_etab` | `NO_REFERENCE_FIELD` | 0, live or exclusion |
| `tied_beta` | `PREFIX_MISMATCH` | 1 live |

#### Why the first two cannot be closed by writing a test

`test_rows_bijective` pins `{row.ref} == REF_F`, the set of **upstream**
reference-surface field names. A row can therefore only exist where an upstream
analogue exists, and neither `emit_etab` nor `emit_dense_hJ` has one — there is
no hit for `etab` or `dense_hJ` anywhere in `reference_surfaces.py`. So no row
can name them, so `covered` can never contain them, however many tests exist.

`emit_etab` makes the point sharply: **it is already implemented and already
tested, and the gate still cannot see it.** This is the gate's construction, not
a coverage hole. Reading the uncovered list as a test backlog would have
generated work that could not have closed the blocker.

#### Why the third is the one to be careful about

`tied_beta` is **one rename from green.** Its only naming test is
`test_fixed_mask_and_tied_beta_are_still_inert`, in `tests/knob_semantics/` — a
non-scoped path — so prefixing it with `test_knob_semantics_` would turn the
gate green in a single edit that was available to me.

I did not make it. The test asserts the knob is **inert**, and the prefix is the
gate's marker for a test that pins knob *semantics*. Renaming would have the
gate certify semantics for a knob that has none. That is the substance of the
check, not its spelling, so it is a scope decision.

### The control is why the zero means anything

`n_missing_test = 0` is the expected answer, and a classifier that can only ever
return zero produces it whether or not it works. The run injects a synthetic
field that *is* test debt (live row, no tests) and requires the label
`MISSING_TEST`, then names a prefixed test on that same row and requires
`COVERABLE`. **Both fired.** The sidecar grades a control failure as
`inconclusive` and discards the counts, so this was not optional.

## What is still open on Z1, and what each needs

1. **3 fields of decision debt** (above). Needs a scope call per field. The
   remedies are: narrow `NEW_FIELDS`; add an aminx-only row kind to the alias
   map so fields without an upstream analogue are expressible; or, for
   `tied_beta` only, accept the rename and with it the claim that an inertness
   test counts as a knob-semantics test. All three change pre-registered gate
   scope.
2. **`rc = 1` from step 1** — `laser_decode_step` + `laser_score` tier-3 f32,
   debt #2445. The tolerance basis is self-declared unmeasured. **Not widened,**
   per the standing rule that a pre-registered band is never loosened to make a
   run pass. Closing this needs a measured floor, the same way #2432 was closed.

## What this supersedes

Task #18 recorded "four measured blockers, two need user decisions" without
distinguishing mechanism. Two of those four (`unwired`, `not_passed`) are now
measured empty and are not blockers. What remains is one decision cluster (3
fields) and one measurement (#2445).
