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

## 0a. Sequencing: run the wave AFTER the main merge, or pay for it twice

The eight rows are **currently live**, and the frame you evaluate that in decides the answer —
which is worth stating because evaluating it in the wrong frame gives the opposite result.

| frame | scoped commits since the measurement | rows |
| :-- | :-- | :-- |
| `origin/wt/260929-potts-laser-main` (the sprint branch — **the gate's frame**) | **0** | **live** |
| `wt/260929-laser-confirm` (this confirmatory branch) | **41** | would fail 1c |

**Anchor correctly: the rows do not sit at `feade020`.** `feade020` (2026-10-04 01:34,
*"redsox: all eight ledger rows re-measured and grading"*) is where the ledger *file* was
written. 1c compares `<the run's own git_hash>..HEAD`, and reading the eight records shows
**three** distinct trees, all earlier than `feade020`:

| run tree | rows |
| :-- | :-- |
| `719c33e0` | `laser_proofread_parity`, `laser_proofread_unconditional_parity`, `laser_decode_e2e` |
| `e9ef452f` | `potts_ar_decode`, `potts_energy_parity`, `laser_score_parity`, `potts_ar_refine_exact` |
| `124f6e9e` | `potts_ddg_megascale` |

Because those trees precede `feade020`, each one's diff to HEAD is a *superset* of
`feade020..HEAD` — so checking `feade020` alone would have been too weak. Re-checked per
anchor against the sprint branch: all three are ancestors of it, and all three have **zero**
scoped paths since. The conclusion survives the stricter test.

All eight runs also read `status=completed, outcome=pass, exit_code=0, git_dirty=False`, so no
row is weak at the record level either — the gate's "all waves pass except `laser_score`" holds
as recorded, not merely as remembered.
Those 41 scoped commits are **only on this branch** — verified with `git branch --contains`:
both `bc3f1950` (*merge origin/main into the sprint lineage*, 10-05 08:48, bringing xtrax
0.4.0a11 + the ChunkedMap dispatch fix) and `5c625b00` (*Potts confirmatory round 2*,
`scripts/parity/`, 10-05 07:20) exist on `wt/260929-laser-confirm` and its remote, and nowhere
else. The sprint branch has had no scoped commit since the measurement.

**"Live" was checked against every tree-dependent 1c condition, not just the scoped one.**
A scoped-path check alone would be a weak basis for this section, because several other
conditions read files that are *not* scoped and could therefore break silently without
tripping `_touches_scoped` — notably `tests/knob_gate/checkpoint_registry.json` (the
weights-key condition), the manifests under `tests/port/manifests/` (argv mutants must equal
the row ids) and each vehicle's `.bth.toml` (`sidecar_sha256` is compared to the file's
*current* digest). `git diff --name-only feade020..origin/wt/260929-potts-laser-main --
tests/knob_gate/ scripts/redsox/` returns exactly one path: `scripts/redsox/README.md`. The
registry, the manifests and the sidecars are all untouched, and the run records themselves are
historical facts that the ledger header says were checked against 1c before being written. So
the rows are live on every condition, not merely on the one that is easy to check.

**The consequence is a sequencing constraint, not a composition one.** The main merge is
wanted — it is the user's standing directive to use xtrax natively, and it carries xtrax
0.4.0a11 plus the ChunkedMap dispatch fix. The moment that merge (or this branch) reaches the
sprint lineage, its 41 scoped commits invalidate all eight rows at once by §0's rule. So:

- **Wave before the merge** → eight rows re-measured, then the merge kills them again, and you
  run a **second** wave. Two re-measurement cycles.
- **Wave after the merge** → one cycle, which is also the cycle that picks up whatever scoped
  items are chosen from §2.

So the re-wave is **elective in composition but not in timing**: once the merge lands it is
forced regardless of which Tier A/B items ride along. Deferring a scoped item to "protect" the
current rows only works while the sprint branch stays free of this branch and of main.

A corollary worth stating because it is easy to get backwards: the rows being live right now is
*not* an argument for keeping them. They were measured against a pre-merge tree, so after the
merge they would describe code that no longer exists even if 1c somehow still accepted them.

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
| A1 | **laser_score option 3** | `tests/port/test_laser_score.py`, `tests/port/targets/laser_score.toml` (+ `tests/lint/test_port_tolerances_match_targets.py`, unscoped) | small | **The only gate-clearing item.** Band measured: `proposed_tol` 4e-4 over a floor of 3.63e-5, **10x headroom** — see §2a. Gated on the user picking option 3. |

### 2a. A1's band, verified by its own record rather than carried forward

Run `3af3be02` reads `status=completed, outcome=pass, exit_code=0, git_dirty=False`, from
`scripts/analysis/laser_score_scale_relative_floor.py` at `0e642d44`, 458 s, sidecar
`c40240fe…`. Its claimed artifact resolves
(`outputs/laser_score_scale_relative_floor/0e642d44/result.json`) and contains:

```
proposed_tol             = 0.0004        floor_scale_relative = 3.6297e-05
headroom                 = 10.0          control_scale_margin = 10.0
control_rejected         = true          below_control_scale  = true
f32_violated_elementwise = true          f64_passes           = true
n_pairs = 63   n_fields = 9   loo_all_admitted = true   loo_worst_ratio = 0.121
floor_field = seq_log_prob               floor_pair = soluble_65000__107m_1
```

**Correction:** earlier revisions of this doc (and `92bdab20`'s message) said "100x margin".
The record says **10x** — `headroom = 10.0`, and 4e-4 ÷ 3.63e-5 ≈ 11. The band itself, 4e-4,
was right.

Three things in that artifact strengthen the case for option 3 beyond the headline:

- **`n_pairs = 63`** — the band was measured over exactly as many pairs as there are failing
  clean-arm cases in `tests/port/test_laser_score.py` (§1). The measurement covers the
  population it would license, not a sample of it.
- **`control_rejected = true`** with `control_scale_ratio = 0.04` — the negative control fails,
  so the instrument is tested rather than merely agreeable.
- **`loo_all_admitted = true`, `n_loo_not_admitted = 0`, `loo_worst_ratio = 0.121`** —
  leave-one-out: no single pair props the floor up. A band that only holds because of one
  outlier would show here, and does not.

And `f32_violated_elementwise = true` with `f64_passes = true` is the attribution restated in
the artifact's own terms: this is f32 precision, not a logic defect. Note the vehicle lives in
`scripts/analysis/`, which is **not** under `_SCOPED_PREFIXES`, so this evidence is not exposed
to the freeze and does not need re-measuring with the wave.
| A2 | #2483 missing-`laser`-extra message | `src/aminx/host/runner.py` | one block | Wrap the `importlib.import_module` so a missing extra names the family and the extra instead of raising a bare `prody` ImportError. |
| A3 | #2484 unknown-`model_family` validation | `src/aminx/run/specs.py`, `src/aminx/host/runner.py` | small | Makes `None` mean "stock handles this" rather than "nothing matched". Changes no number **for a valid family** — but see §4. |
| A4 | #2417 `--double` flag | `scripts/parity/potts_refine.py` | one line + argparse | Diagnostic only. Unlocks the run that decides whether the 0.9539 refine gap is f32 precision or a logic defect. |

Tier B — changes measured numbers. Each needs its effect attributable.

| # | item | files | what moves | re-measurement |
| :-- | :-- | :-- | :-- | :-- |
| B1 | #2475 refine key split | `src/aminx/families/potts_mpnn/sample_host.py` | `optimize_pdb` / `optimize_fasta`, and the sample path **only when a chain suffix is set** | No gate slug exercises those paths, so plausibly zero rows move. Confirm before assuming. |
| B2 | #2371 stage 2 step 3 (dispatched sample loop) | `src/aminx/families/potts_mpnn/sample_host.py`, `src/aminx/host/family_runner.py` | the sample chunk path — **which includes refine@0.3** | See §3. This is the expensive one. |
| B3 | #2480(b) `state_position_map` not bucket-padded | `src/aminx/inference/bundle_builder.py` (one field) | **small, now diagnosed** | See §3a. One conditioning field escapes bucket padding; everything else in the bundle is padded correctly. |

Tier C — larger, and each wants its own decision first, so listed for completeness rather than
proposed: #2459 (`tied_positions` inert), #2443 (Potts optional dicts), #2435 (eight unread
Options fields), #2433 (three unimplemented Potts fields — note its own impact line says
`test_superset` cannot pass while two alias rows have no mappable target), #2311 (LASEr RBF
`D_mu` dtype), #2309 (`laser_layers` pure-rtol tolerance), #2321 (`categorical_draw` f64 CDF),
#2444 (`AMINX_PORT_WAVE` loads the wrong oracle), #2371 steps 1/2/4/5.

## 3a. B3 is now diagnosed: one conditioning field escapes bucket padding

Measured by instrumenting the benchmark smoke on titanix (`--smoke --hardware CPU`, L=76
fixture, `BucketingConfig()` whose default buckets are `(64, 128, 256, 512)`, so 76 buckets
up to **128**). Printing the bundle at the failing call site:

```
state_position_map = (1, 76)      <- REAL length, not padded
bias               = (128, 21)    <- bucket-padded
sequence_oh        = (128, 21)    <- bucket-padded
tie_group_map      = (1, 128)     <- bucket-padded
geometry.coords    = (1, 128, 4, 3)
geometry.mask      = (1, 128)
```

The padding is otherwise uniform and correct. `state_position_map` alone keeps the real
length. `_apply_logit_transform` (`inference/decode/_base.py:122-123`) then calls
`_realign_states_to_reference`, whose signature is `("S L V", "S L") -> "S L V"`
(`inference/decode/_kernel.py:173-176`): it **gathers the logits along L into the map's
frame**, silently turning `(S, 128, 21)` into `(S, 76, 21)`. The next line adds the still-padded
`bias`, giving the reported

```
TypeError: add got incompatible shapes for broadcasting: (76, 21), (128, 21)
```

Traceback, for whoever implements: `bench_aminx_jax.py:539` -> `host/plan.py:791` (score) ->
`:709` (decode) -> `inference/decode/conditional.py:154` -> `_base.py:126` ->
`inference/logits.py:104` (`result = result + bias`).

**Two reasons this is larger than a benchmark artifact.** (1) The map is an *identity* map by
default -- the realignment docstring says so explicitly -- so it is non-None on an ordinary
single-structure score, and therefore **every bucket-padded conditional score takes this
path**, not just multi-state designs. (2) A second mismatch sits immediately behind it:
`conditional.py:159` passes `cond.tie_group_map[0]` (length 128) to `_apply_tie_group_fuse`
alongside fused logits that are now length 76.

**Not fixed here**, because `src/aminx/` is scoped and a commit there invalidates all eight
ledger rows (§0). The fix is one field in the bundle's padding step, and the convention
question -- pad `state_position_map` to the bucket like every sibling field, or slice `bias`
and `tie_group_map` down to the real length -- should be answered the way the rest of the
bundle already answers it: pad.

**Corrects two earlier mischaracterisations of mine.** `c18062d6`'s message called 128 "a
padded length" and guessed the failure was in `ar_sample`; and a later reading of mine called
128 a sample-axis cardinality. Both wrong. The failing cell is `score_conditional`, which goes
through `make_inference_plan` normally, and 128 is a *length bucket ceiling* --
`max_length` defaults to 512 and `N_SAMPLES.cardinality` merely happens to also be 128
(`tiling/axes.py:66`), which is a coincidence that made the wrong reading look plausible.

## 3. The load-bearing constraint: B2 costs a Potts re-run

The refine@0.3 confirmatory run is in flight **pinned to `5c625b00`**, a tree that does not
contain B2. B2 changes the sample chunk path, and refine@0.3 goes through that path. So:

- **Land B2 → tonight's refine@0.3 result no longer describes the final tree**, and it has to be
  re-run. Cost measured from this very run, from its own stamp mtimes: **~8.5–9 h, serial**.
- **Leave B2 out → the run stands**, and stage 2 of #2371 waits for a later wave.

That is the single biggest lever in the composition, and it is not a correctness question —
both answers are defensible. It is a question of whether stage 2 is worth ~8.75 h of re-measurement
now or later.

**Where the ~8.75 h comes from** (read off the in-flight run's own stamp mtimes in
`/tmp/potts_sample_dist_confirm_r2/units`, so it is measured, not projected from a sidecar):
twelve stamps at 10:34 10:47 10:53 10:59 11:36 12:19 13:16 14:09 14:46 15:23 15:54 16:24,
i.e. gaps of 13 6 6 37 43 57 53 37 37 31 30 min. **Per-unit cost is bimodal** — four fast
upstream-arm units at ~6–14 min and eight aminx-arm units averaging **40.6 min** — and the
work is **serial**: exactly one `--aminx-worker` process exists at a time and no two stamps
overlap. A re-run pays twelve *slow* units, not twelve average ones, so the arithmetic is
12 × 40.6 + 4 × 9.75 ≈ **8.75 h**; call it **~8.5–9 h**. This run started 10:20, has three
aminx-arm units left behind the one in flight, and projects to finish ~19:05 — consistent.

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

### 5a. Redsox re-provisioning splits into a cheap half and a scoped half

Checked on titanix 261005: the **source is already there** — `/home/solab/projects/redsox`,
a normal installable package (`name = "redsox"`, `version = "0.1.0a1"`, with `src/` and
`pyproject.toml`). It is simply not installed: `find_spec("redsox")` is `None` in the confirm
venv and nothing named `redsox` is on `PATH`. So this is a local install, not a fetch or a
clone, and that splits the blocker in two:

| | what it buys | scope |
| :-- | :-- | :-- |
| (a) hand-install from the local checkout | step 2 can run **now**; no repo file changes | **not scoped** — no ledger row touched |
| (b) declare it so it survives `uv sync` | the durable fix #2319 actually asks for | **scoped** (`pyproject.toml` is in `_SCOPED_FILES`) → rides the wave |

Useful consequence: **(a) unblocks a gate run without a scoped commit.** The gate can be
re-run to confirm the remaining red is only `laser_score` before anyone commits to the wave.
(b) is the real fix and must ride the wave, because touching `pyproject.toml` invalidates all
eight rows by itself.

**Do not do (a) into `aminx-confirm-git/.venv` while the Potts confirmatory run is in
flight** — that run uses that interpreter, and installing into a venv underneath a live
measured process is not worth the risk for a step that can wait an hour.
