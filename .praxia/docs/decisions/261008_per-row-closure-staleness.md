---
title: Per-row import-closure staleness for ledger rows
description: Ledger rows go stale only when a changed path is in that row's import closure, not on any scoped-prefix change; fail-safe rules, declared edges, and a run-time loaded-files check.
status: accepted
task_id: 261008_closure-staleness
date: '261008'
supersedes: ''
backlog_ids: ''
---
# Per-row import-closure staleness for ledger rows

## Decision

Accepted by the user 261008 ("i want the per-vehicle closure rule pursued before the final rewave").

Step 1c condition "no scoped path since the run commit" changes from a global file-prefix test to a
per-row test: a row is stale when a changed path is in **that row's import closure**
(`tests/knob_gate/_closure.py`; declarations in `tests/knob_gate/closure_edges.toml`). This amends the
pre-registered staleness criterion of the redsox gate (spec `260929_pottsmpnn-lasermpnn-xtrax-composition`
§6.6 step 1c). The change was made before any wave that it could affect, and no recorded result was
re-graded under it.

## Why

The global rule invalidated every row on any scoped commit. Measured 261008 with
`scripts/redsox/stale_rows.py --run-commit dab58900 --head origin/main`: PR #205 touched nine files, all under
`src/aminx/families/potts_mpnn/`; the old rule marks 8 of 8 rows stale, the new rule marks the 4 Potts rows
stale and leaves the 4 LASEr rows fresh (7.75 h of the ~9.1 h graded compute, and the 4.4 h critical path).

## Rule

- Closure = static import closure (every import statement, any nesting depth, plus package `__init__`) of
  `scripts/parity/<slug>.py` over `src/` and `scripts/parity/`, plus the slug's declared `soft` edges.
- **Fails safe.** Unresolvable, undeclared or unparseable input makes a row stale, never fresh:
  - a dynamic import / package-data read in a closure file is unsound unless declared in `dynamic_ok`;
  - non-`.py` files under `src/aminx/` or `scripts/parity/` stay global unless declared as `data`;
  - `pyproject.toml`, `uv.lock`, `scripts/recapture/`, `aminx-oracles/` stay global;
  - no closure computed (missing vehicle, bad TOML) falls back to the global rule.
- `tests/port/` is no longer scoped for rows: no graded vehicle imports from it (grep, 261008) and the port
  suite is re-run by gate step 2 on every gate run.

## Evidence and its limits

- Verified: static closures of the 8 rows are sound after the declarations; LASEr closures contain 0 Potts
  files and Potts closures 0 LASEr files (130 `src` files each, shared core included). 83 tests pass on
  titanix (`test_closure_selftest.py`, `test_coverage_selftest.py` including two end-to-end fixtures
  `viii_closure_outside` / `viii_closure_inside` that differ only in the changed path). A mutant that
  disables closure membership is killed by 9 of them.
- **Not verified:** that a real vehicle run loads nothing outside its closure. `dynamic_ok` entries are
  reviewed claims; the one prior dynamic probe (import-time only, e56aa398) found 0 Potts modules for the
  LASEr vehicles. `scripts/redsox/closure_hook` records the files a run really loads and `verify_wave.py`
  fails the row if one is outside the closure, but runs recorded before this change have no such file, so
  the first wave after it is the first real-execution check. If that check ever fails, the closure or its
  declarations are wrong and every row that relied on them must re-run.
- Not done: gate step 1c itself does not require `loaded_files.json` (legacy rows lack it).

## Consequences

`launch_wave.sh --stale-only` runs only stale slugs per group; `stale_rows.py` lists them. Shared core code is
in every closure, so changing it still invalidates every row (pinned as a negative control).

