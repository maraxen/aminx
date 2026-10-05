---
title: 'Re-wave composition: what the single scoped batch should contain, and the one item that costs a Potts re-run'
description: Decision-ready menu of every scoped-pending item with its blast radius, which ones change measured numbers, and the sequencing constraint that makes one of them expensive
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261005'
sprint: ''
backlog_ids: ''
---
# Re-wave composition: what the single scoped batch should contain, and the one item that costs a Potts re-run

**This document decides nothing.** Composition is the user's call; it is on the open-decision
list. What it does is price each candidate, so the choice is cheap to make.

## 0. The invariant that forces a single batch

`tests/knob_gate/_coverage.py:21-28` defines `_SCOPED_PREFIXES = (src/aminx/, scripts/parity/,
scripts/recapture/, tests/port/, aminx-oracles/)` plus `_SCOPED_FILES = {pyproject.toml,
uv.lock}`. Section 1c rejects any ledger row whose run's commit touches one of those paths —
**globally, not per-slug**. So one scoped commit invalidates all eight rows, and the eight rows
then have to be re-measured together. That is why scoped work batches into one wave instead of
trickling.

Verified this session as **outside** the freeze, and therefore not re-wave material (several
were landed tonight without touching a row): `tests/knob_gate/`, `tests/lint/`, `tests/potts/`,
`tests/audit/`, `tests/benchmarks/`, `scripts/benchmarks/`, `scripts/redsox/`, `docs/`,
`.praxia/docs/`.

## 1. What the gate is actually waiting on

Gate run `fb898d24` @ `feade020`, from its own `outcomes.jsonl`: **586 passed, 63 failed, 6
skipped, and all 63 failures are `tests/port/test_laser_score.py` clean-arm failures.** Nothing
else in step 1 is red, and `step2_passed` was true.

So exactly one candidate below clears the gate. Everything else is a **passenger** — worth
including because the wave's cost is paid once either way, but not the reason to run it.

## 2. The menu

Tier A — no measured number moves. Free passengers: they cannot change a ledger row's value,
only its commit, which the wave re-measures anyway.

| # | item | files | size | notes |
| :-- | :-- | :-- | :-- | :-- |
| A1 | **laser_score option 3** | `tests/port/test_laser_score.py`, `tests/port/targets/laser_score.toml` (+ `tests/lint/test_port_tolerances_match_targets.py`, unscoped) | small | **The only gate-clearing item.** Band already measured: 4e-4 of field scale, 100x margin, run `3af3be02`. Gated on the user picking option 3. |
| A2 | #2483 missing-`laser`-extra message | `src/aminx/host/runner.py` | one block | Wrap the `importlib.import_module` so a missing extra names the family and the extra instead of raising a bare `prody` ImportError. |
| A3 | #2484 unknown-`model_family` validation | `src/aminx/run/specs.py`, `src/aminx/host/runner.py` | small | Makes `None` mean "stock handles this" rather than "nothing matched". Changes no number **for a valid family** — but see §4. |
| A4 | #2417 `--double` flag | `scripts/parity/potts_refine.py` | one line + argparse | Diagnostic only. Unlocks the run that decides whether the 0.9539 refine gap is f32 precision or a logic defect. |

Tier B — changes measured numbers. Each needs its effect attributable.

| # | item | files | what moves | re-measurement |
| :-- | :-- | :-- | :-- | :-- |
| B1 | #2475 refine key split | `src/aminx/families/potts_mpnn/sample_host.py` | `optimize_pdb` / `optimize_fasta`, and the sample path **only when a chain suffix is set** | No gate slug exercises those paths, so plausibly zero rows move. Confirm before assuming. |
| B2 | #2371 stage 2 step 3 (dispatched sample loop) | `src/aminx/families/potts_mpnn/sample_host.py`, `src/aminx/host/family_runner.py` | the sample chunk path — **which includes refine@0.3** | See §3. This is the expensive one. |
| B3 | #2480(b) padded-vs-real length | likely `src/aminx/` | unknown | Not yet diagnosed. `(128,1)` vs `(76,21)`; same shape as #2135 / #2081. Size unknown, so do not commit to it blind. |

Tier C — larger, and each wants its own decision first, so listed for completeness rather than
proposed: #2459 (`tied_positions` inert), #2443 (Potts optional dicts), #2435 (eight unread
Options fields), #2433 (three unimplemented Potts fields — note its own impact line says
`test_superset` cannot pass while two alias rows have no mappable target), #2311 (LASEr RBF
`D_mu` dtype), #2309 (`laser_layers` pure-rtol tolerance), #2321 (`categorical_draw` f64 CDF),
#2444 (`AMINX_PORT_WAVE` loads the wrong oracle), #2371 steps 1/2/4/5.

## 3. The load-bearing constraint: B2 costs a Potts re-run

The refine@0.3 confirmatory run is in flight **pinned to `5c625b00`**, a tree that does not
contain B2. B2 changes the sample chunk path, and refine@0.3 goes through that path. So:

- **Land B2 → tonight's refine@0.3 result no longer describes the final tree**, and it has to be
  re-run. Cost measured from this very run, from its own stamp mtimes: **~8–8.5 h, serial**.
- **Leave B2 out → the run stands**, and stage 2 of #2371 waits for a later wave.

That is the single biggest lever in the composition, and it is not a correctness question —
both answers are defensible. It is a question of whether stage 2 is worth ~8 h of re-measurement
now or later.

**Where the ~8 h comes from** (read off the in-flight run's own stamp mtimes in
`/tmp/potts_sample_dist_confirm_r2/units`, so it is measured, not projected from a sidecar):
twelve stamps at 10:34 10:47 10:53 10:59 11:36 12:19 13:16 14:09 14:46 15:23 15:54 16:24,
i.e. gaps of 13 6 6 37 43 57 53 37 37 31 30 min. **Per-unit cost is bimodal** — four fast
upstream-arm units at ~6–14 min and eight aminx-arm units averaging **40.6 min** — and the
work is **serial**: exactly one `--aminx-worker` process exists at a time and no two stamps
overlap. Sixteen units therefore cost ~8–8.5 h end to end; this run started 10:20 and projects
to finish ~18:35.

An earlier revision of this section said "~5–6 h at the current 4-way concurrency". **Both
halves were wrong**: there is no 4-way concurrency, and averaging the fast upstream units into
the per-unit figure understated a re-run by roughly 3 h. The corrected number is the one above.

**B1 and B2 are separable in effect even though they edit the same file.** B1 only reaches
`fresh_refine_order`, which is called when `not stored_orders_present or chain_suffix`; the
confirm path passes `stored_orders_present=True` with no suffix and `num_samples=1`, so it takes
the stored-AR-order branch and never reads the key B1 changes (verified by reading
`sample_host.py:72-92` and `:581-588` against `potts_sample_dist_confirm.py:450`). B2 reaches the
chunk loop. So if both land, a refine@0.3 delta is attributable to B2 alone and an
`optimize_pdb` delta to B1 alone — no confounding. **Do not infer from this that B1 is free to
land with B2 unverified**; it means only that the attribution survives.

## 4. Two traps for whoever implements

- **A3 can start raising on existing callers.** It turns an accepted-but-unknown family into an
  error. Before landing it, grep every vehicle, fixture and sidecar for a `model_family` string
  outside the four valid values — anything relying on the silent fall-through would begin to
  fail. The CLI accepts a free string today (`cli.py:485-493`, `:1095-1103`), so a stale value in
  a committed spec JSON is the realistic case.
- **A4 changes a sidecar's script, not just a script.** `potts_refine.py` has a committed
  `.bth.toml`; section 1c compares a run's `sidecar_sha256` against the sidecar file's *current*
  digest, so if the sidecar is edited the run must be redone. Editing the script alone is fine;
  editing the sidecar means re-running.

## 5. The cheapest wave that clears the gate

If the goal is "gate green, minimum risk": **A1 alone**, plus A2–A4 as free passengers. No Tier B,
no Potts re-run, one re-measurement cycle of the eight rows, and the gate's only red is gone.

If the goal is "spend the re-measurement cycle well": add B1 (likely zero row movement) and B3
only if §3's diagnosis has happened first. B2 is the deliberate ~6 h decision.

Either way the wave is followed by one re-measurement of all eight rows, and **redsox has to be
re-provisioned first** or step 2 cannot run at all — see debt #2319, which also records that
the install must survive a `uv sync` rather than being repeated by hand.
