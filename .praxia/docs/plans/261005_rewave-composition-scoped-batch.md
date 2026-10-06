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

## 0b. Z1 status against the spec's own criteria — one unmet item, and the wave is self-inflicted

Spec line 1246 states Z1's acceptance criteria. Each was checked rather than assumed:

| criterion | status |
| :-- | :-- |
| `sidecar_sha256 == sha256(run_gate.bth.toml)`, non-empty | **met** — gate run `fb898d24` recorded `39babc8b…`, and `git show origin/wt/260929-potts-laser-main:scripts/redsox/run_gate.bth.toml \| sha256sum` is the same digest. The gate's own sidecar has not drifted (worth checking separately, because `scripts/redsox/` is **not** scoped, so it could have). |
| `git_dirty` false | **met** — `fb898d24` recorded `False`. |
| *"re-runs every vehicle sidecar whose ledger record fails the ancestor or diff-scope check"* | **met, vacuously: the re-run set is EMPTY.** All eight rows pass both checks today (§0a). |
| `status == 'completed'` | **met.** |
| `outcome == 'pass'` | **NOT met** — `fb898d24` is `outcome=fail` (at `exit_code=0`, incidentally). The cause is step 1's 63 `laser_score` clean-arm failures and nothing else (§1). |

**So Z1 has exactly one unmet criterion, and the re-wave exists only because the fix for it is
itself scoped.** Read Z1's own wording: the operator re-runs the vehicles whose records *fail*
1c. Today that set is empty, so if `laser_score` could be cleared without touching a scoped
path, Z1 would complete with **no re-measurement at all**. It cannot: option 3 moves a
tolerance, the tolerance lives in `tests/port/targets/laser_score.toml`, and
`tests/port/` is scoped — and it has to live there, because the unscoped
`tests/lint/test_port_tolerances_match_targets.py` exists precisely to keep the test and the
target file in agreement.

The eight-row re-measurement is therefore **self-inflicted by the fix's location**, not the
repair of pre-existing drift. That does not make it avoidable; it does mean the wave's cost is
attributable to A1 alone, which is the cleanest possible argument for letting A2–A4 ride along
free (§5).

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
| A1 | **laser_score option 3** | `tests/port/test_laser_score.py`, `tests/port/targets/laser_score.toml` (+ `tests/lint/test_port_tolerances_match_targets.py`, unscoped) | small | **The only gate-clearing item.** Band measured: `proposed_tol` 4e-4 over a floor of 3.63e-5, **10x headroom** — see §2a. **DECIDED 2026-10-06: the user chose option 3.** No longer gated. |

**A1 is decided.** The user selected option 3 on 2026-10-06, so the single unmet Z1 criterion in
§0b is now met and the last gate-blocking design question is closed. What A1 still needs is
mechanical, not a decision: the band applied to the target TOML, the `f32_atol_basis` prose
rewritten to cite run `3af3be02` (its current text ends "Do not move it without a new
measurement that clears the margin" — that measurement now exists, so leaving the sentence
would make the file argue against its own number), and the unscoped lint test kept in step.

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
| A5 | #2480(b) `state_position_map` not bucket-padded (**was B3**) | `src/aminx/tiling/pad.py:44-54`, one field in `pad_bundle`'s tuple | small | Reclassified from Tier B — **no measured value can move**, see §2b. Fixes a path that currently crashes. |

### 2b. Why #2480(b) is a free passenger, not a Tier B risk

It was filed under Tier B as "unknown" size, and the honest reason was that nobody had diagnosed
it. Now that it is located (§3a), its blast radius is checkable, and it is empty:

- **No `tests/port/` wave uses bucketing.** `grep -rln 'bucket_config\|BucketingConfig'
  tests/port/` returns nothing. The only users are `tests/tiling/test_bucketing.py`,
  `tests/host/test_bucketed_plan.py`, `tests/utils/test_autoregression.py`,
  `tests/parity/browser_validation_paths.json` and `scripts/benchmarks/*` — **all unscoped**.
  So none of the eight gate vehicles reaches `pad_bundle` at all, and padding one more field
  cannot change a ledger row's value.
- **No test asserts the map's padded width.** `state_position_map` appears in none of those
  three bucketing tests. The single `pad_bundle` mention
  (`tests/utils/test_autoregression.py:154`) is about zero-padded *waves* and the mask, not the
  map. So the fix needs no test updates.

The commit is still **scoped** (`src/aminx/tiling/pad.py`), so it still invalidates all eight
rows by the path rule — which is exactly what Tier A means: *"they cannot change a ledger row's
value, only its commit, which the wave re-measures anyway."* It changes behaviour only on a path
that currently **raises**, so there is no prior number to preserve.

One thing to look at rather than assume: `tests/parity/browser_validation_paths.json:716` maps
`src/aminx/tiling/pad.py::pad_bundle` to coverage id `P27`. That file is unscoped, but whoever
lands this should confirm P27's claim still holds.

### 2c. #2475 confirmed inert on every gate row — the doc's own TODO, discharged

The B1 row used to end *"plausibly zero rows move. Confirm before assuming."* Confirmed, and
the check was worth running because the first result looked **bad**: grepping the eight gate
vehicles for `optimize_pdb|optimize_fasta|chain_suffix` returns 0 for six of them and **3 for
`potts_ar_refine_exact`** — a *passing* ledger row. Nothing in `tests/port/` matches.

Reading those three settles it:

- `:119` and `:122` are the **n-to-c mutant**, `_ntoc`, whose body is
  `del ar_order, randn, num_samples, chain_suffix, stored_orders_present` — it ignores every
  one of them.
- `:344-351` is the real call, and it passes **`chain_suffix=""` with
  `stored_orders_present=True`**, `num_samples=1`.

`upstream_refine_order` takes `fresh_refine_order(chain_mask, randn)` only when
`not stored_orders_present or chain_suffix` — false here — then with `num_samples == 1` returns
`ar_order` directly. **So `randn` is never read**, and the key B1 changes has no consumer on
this vehicle.

B1 therefore cannot move any ledger row's value, which makes it a free passenger by this doc's
own definition. **It is still a real fix** — the key reuse is a genuine defect on
`optimize_pdb`/`optimize_fasta` and the chain-suffix path — so it needs verifying on its own
terms; it just needs no attribution against a ledger row.

**Consequence: Tier B is now B2 alone.** The wave is six free passengers plus the one
gate-clearing item, with a single genuinely expensive optional decision (§3).

Tier B — changes measured numbers. Each needs its effect attributable.

| # | item | files | what moves | re-measurement |
| :-- | :-- | :-- | :-- | :-- |
*(B1 was here. **Confirmed and reclassified to Tier A as A6** — see §2c. Tier B is now **B2
alone**.)*
| B2 | #2371 stage 2 step 3 (dispatched sample loop) | `src/aminx/families/potts_mpnn/sample_host.py`, `src/aminx/host/family_runner.py` | the sample chunk path — **which includes refine@0.3** | See §3. This is the expensive one. |
*(B3 was here. It has been **reclassified to Tier A as A5** — see below. Tier B is now B1 and
B2 only.)*

### 2d. #2444 cannot have contaminated any recorded evidence

Worth checking before anything else in Tier C, because "`AMINX_PORT_WAVE` loads the wrong
oracle" reads like it could undermine the eight passing rows. It cannot. Blast radius, bounded
by reading the resolution order:

`tests/port/conftest.py:71-81` resolves the target as: env var set and not `__nonport__` →
`targets/<wave>.toml`; otherwise fall back to `pyproject.toml`'s `[tool.port] target`;
otherwise raise. That fallback is configured (`pyproject.toml:279-280`) to
**`tests/port/targets/port_selftest.toml`** — the self-test target, not a sibling wave's.

| path | affected? | why |
| :-- | :-- | :-- |
| the redsox gate | **no** | `run_gate.py:147,216` sets the var per wave, and `:258-260` pops it for step 0 precisely so a leak cannot deselect the rest |
| the eight ledger rows | **no** | they are `scripts/parity/` vehicle runs, not pytest port waves — the variable is not in their path |
| `tests/port/run_waves.sh` | **no** | `:16` sets it explicitly |
| a bare `pytest tests/port/test_<wave>.py` | **yes** | resolves to `port_selftest.toml` |

So #2444 is a **developer footgun, not an evidence problem** — no recorded measurement could
have been graded against the wrong target. That lowers its urgency considerably: it does not
need to ride this wave to protect anything, though `tests/port/` being scoped means it is cheap
to include once the wave is running anyway.

### 2e. #2433 does not block the gate — its impact line is stale

This doc previously repeated #2433's own impact text, which says `test_superset` cannot pass
while two alias rows have no mappable target. **Measured on titanix against this branch, it
passes:**

```
tests/knob_gate/test_knob_superset.py
  test_rows_bijective PASSED   test_superset PASSED   test_exclusions PASSED
  test_row_markers_are_declared PASSED   test_u1_reachability_present PASSED
  test_parity_ids_passed SKIPPED (structural: needs AMINX_REDSOX_OUTCOMES_READ)
  => 5 passed, 1 skipped
```

The alias-map target work resolved it. So #2433 is **not** a gate blocker — the three
unimplemented Potts fields are a completeness matter. The stale line made it look like it gated
step 2, which would have mis-prioritised it: it is Tier C and elective. Corrected on the debt
item too, so the ledger does not keep the wrong story.

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

**The exact fix site, located by reading this branch's tree** (not the one the probe ran in —
see the caveat below): `pad_bundle` in `src/aminx/tiling/pad.py` pads exactly six fields —
`fixed_mask, fixed_tokens, bias, tie_group_map, sequence_oh, ar_mask` (`:44-46`, tuple at
`:52-54`) — and `state_position_map` is **not among them**. Meanwhile
`bundle_builder.py:198-203` defaults the map to `jnp.arange(seq_len)` *before*
`pad_bundle(bundle, target_length)` runs at `:364-367`. So the map is created at the real
length and then never padded, while its six siblings are. Adding it to that tuple is the fix.

Note `bundle_builder.py:208` already contains an F004 guard whose comment reads *"a
reference-frame width different from the padded chain length"* — i.e. the intended invariant is
that the map matches the **padded** length. The default path violates the invariant the guard
was written to protect, which is why the guard does not fire: it only checks a *user-supplied*
map.

**Not fixed here**, because `src/aminx/` is scoped and a commit there invalidates all eight
ledger rows (§0). The convention question -- pad `state_position_map` like every sibling field,
or slice `bias` and `tie_group_map` down to the real length -- is already answered by that
F004 comment and by the other six fields: pad.

**Provenance caveat.** The shapes above were observed by instrumenting the benchmark in the
`aminx-confirm-git` tree, whose venv pins the **pre-0.4.0a11** xtrax and whose `src/aminx` is
`5c625b00` — this branch cannot run there at all (`ImportError: cannot import name
'ChunkedMap' from 'xtrax.tiling'`, because the main merge `bc3f1950` moved the floor to
0.4.0a11). The defect's presence on *this* branch was therefore established by reading
`pad.py` and `bundle_builder.py` directly, as cited, rather than by re-running. Upgrading
xtrax in that venv to re-run was deliberately not done: the Potts confirmatory run was using
that interpreter at the time.

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
  re-run. Cost, now **measured end to end rather than projected**: `duration_s = 41863.66`
  on run `096d0847`, i.e. **11.63 h, serial**.
- **Leave B2 out → the run stands**, and stage 2 of #2371 waits for a later wave.

That is the single biggest lever in the composition, and it is not a correctness question —
both answers are defensible. It is a question of whether stage 2 is worth **~11.6 h** of
re-measurement now or later.

**The measurement.** Run `096d0847` @ `5c625b00` completed at 21:58 with `outcome=pass`,
`duration_s = 41863.66` (11.63 h), `n_computed = 16`, `n_reused = 0`. Per-unit, from the stamp
mtimes (run started 10:20):

| unit | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| min | 14 | 13 | 6 | 6 | 37 | 43 | 57 | 53 | 37 | 37 | 31 | 30 | 66 | 66 | 100 | ~102 |

The structure is **4 targets × 4 arms** (U1, U2 upstream; A, CTRL_m aminx), units grouped by
target in four consecutive blocks:

| target block | U1 | U2 | A | CTRL_m |
| :-- | --: | --: | --: | --: |
| `3gg7`-tier (1–4) | 14 | 13 | **6** | **6** |
| (5–8) | 37 | 43 | 57 | 53 |
| (9–12) | 37 | 37 | 31 | 30 |
| `swe1_ligand`-tier (13–16) | 66 | 66 | 100 | ~102 |

**Cost is dominated by the TARGET, not the arm.** A re-run pays the same 16-unit mix, so
11.63 h is the price. The work is serial — exactly one `--aminx-worker` at a time, no two
stamps overlapping.

**Three earlier figures in this section were wrong, and all three understated the cost.** They
are recorded rather than quietly replaced, because the pattern matters: every projection
assumed the observed rate would continue, and each was made before the expensive target ran.

1. *"~5–6 h at the current 4-way concurrency"* — both halves wrong. There is no concurrency.
2. *"~8.5–9 h"*, from `12 × 40.6 + 4 × 9.75` — wrong because it called the spread **bimodal**
   and attributed it to **arm**: "four fast upstream-arm units … eight aminx-arm units
   averaging 40.6 min". The four fast units are in fact **all four arms of target 1, aminx
   included**, and there are four target tiers (~10, ~45, ~34, ~83 min), not two modes.
3. *"aminx arms cost 2–3× their upstream counterparts"* — wrong. Measured A/U1 ratios are
   **0.43, 1.54, 0.84, 1.52**; on target 1 aminx is **faster** than upstream.

The lesson for the next estimate: with per-target tiers this wide, a partial run supports no
total. Quote a duration only once the run has paid for its most expensive target.

**Is the 11.63 h inflated by repeated cold compiles? No — and the run's own data settles it.**
Each of the 16 units ran in its own `--aminx-worker` subprocess and **no persistent XLA
compilation cache is configured** (not in the parity drivers; no `JAX_*`/`XLA_*` variable in
the titanix environment), so every unit does pay a fresh compile. That raised the question of
whether this figure is mostly compilation, since this stack has recorded 15–20 min cold
compiles elsewhere.

It is not. `_runs("refine@0.3")` returns `("U1", "U2", "A", "CTRL_m")`
(`potts_sample_dist_confirm.py:157-159`), so within each target block the last two units are
the aminx arms — and on target 1 those took **6 minutes each**, total, including process start,
weight load, compile *and* sampling at n=1000. A 15–20 min compile is therefore impossible
here. Compile is bounded above by 6 min per unit, i.e. **≤48 min across all eight aminx units,
≤7% of 11.63 h** — and that bound is generous, because it counts target 1's entire sampling
cost as compile.

**So 11.63 h stands as the re-run price.** An earlier revision of this section claimed it
"may itself be materially inflated" by compilation; that claim is withdrawn, falsified by the
run's own per-arm timings at no cost. The open compile question is **LASEr-specific** — see
`reference/261005_laser-confirm-run-readiness.md`, where the per-sample gap is large enough
that it still matters there.

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

### 5b. Step 2 is verified green on this branch, and its two skips are expected

`tests/knob_gate` — the gate's step 2 — run against this branch on titanix: **35 passed, 2
skipped**. Both skips are structural, not failures:
`test_branch_coverage.py:33` and `test_knob_superset.py:250` require
`AMINX_REDSOX_OUTCOMES_READ`, which only `scripts/redsox/run_gate.py` exports, so outside a
gate run they have nothing to grade — and they *do* execute inside one. **Do not treat them as
regressions when reading a standalone run.**

The drift guard is **not** among the skips: `test_reference_surfaces_drift.py` now runs and
passes, which is the point of `9f7208aa` (it previously skipped on any machine that was not one
user's laptop — including the gate host — hiding a stale `EXTRACTOR_SHA256`).

**Environment note, which is a trap in its own right.** This could not be run in
`aminx-confirm-git`'s venv: that one pins the **pre-0.4.0a11** xtrax, and this branch carries
the main merge that moved the floor, so it dies at import with `cannot import name 'ChunkedMap'
from 'xtrax.tiling'`. It was run instead with `aminx-b7i-git`'s interpreter (xtrax 0.4.0a11 +
pyarrow 25.0.1) and `PYTHONPATH` pointed at this branch's `src/`. `aminx-ci`'s venv has the
right xtrax but **no pyarrow**, so it cannot collect `test_coverage_selftest.py`. Whoever runs
the wave needs an environment with **both**.

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

## 5c. The b7i venv is xtrax 0.4.0a10, not a11 — it reflects main's pin, not this branch's

Found 2026-10-06 while provisioning the chunk-width measurement, and it corrects a claim this
sprint had been carrying: `aminx-b7i-git`'s venv was recorded as "xtrax 0.4.0a11 + pyarrow".
**It is 0.4.0a10.** Measured side by side:

```
local worktree venv   xtrax 0.4.0a11   ['ChunkedMap', 'ChunkedMapIterator', 'MapIterator', 'Vmap', 'VmapIterator']
titanix b7i venv      xtrax 0.4.0a10   ['MapIterator', 'SafeMap', 'SafeMapIterator', 'Vmap', 'VmapIterator']
```

**a11 renamed `SafeMap` to `ChunkedMap`.** `src/aminx/tiling/planner.py:8` does
`from xtrax.tiling import ChunkedMap as XtraxChunkedMap`, so on a10 the import fails outright:

```
ImportError: cannot import name 'ChunkedMap' from 'xtrax.tiling'
  via src/aminx/sampling/conditional_logits.py:34 -> src/aminx/tiling/planner.py:8
```

This is **not a mystery, and not a broken venv** — it is a pin difference. `pyproject.toml:26`
on this branch pins `xtrax[io,export]==0.4.0a11` exactly, while **main pins a10**. b7i is
synced to main's pin. So the venv is correct for main and wrong for this branch.

**Three consequences.**

1. **Any run of THIS branch's code on b7i that reaches the tiling planner fails at import.**
   That covers every sampling path, including the LASEr and Potts distributional vehicles. A
   measurement must use a venv built from this branch's own lock.
2. **It does not invalidate the step-2 result from earlier today.** The
   `tests/knob_gate/test_knob_superset.py` run (5 passed, 1 structural skip) does not import
   `conditional_logits`, so it never reaches `planner.py`. That result stands — stated
   explicitly so it is not discarded by association.
3. **It sharpens the main-merge work.** The `SafeMap` -> `ChunkedMap` rename is exactly the
   kind of cross-version API change the merge has to reconcile, and it is load-bearing for the
   standing directive that aminx use xtrax natively with Vmap/ChunkedMap dispatch happening
   under the hood. Whoever lands the merge should expect the pin bump to be part of it, not a
   follow-up.

A pin-compliant venv now exists at `titanix:~/aminx-chunkwidth-261006/.venv`, built by
`uv sync --extra laser` against this branch's `uv.lock`: xtrax 0.4.0a11 with `ChunkedMap`
present, prody importable, jax 0.10.2. It also **independently re-confirms the CPU-only
constraint** behind the ~223 h estimate — `jax.devices()` returns `[CpuDevice(id=0)]`, so the
cost figure is not resting on a stale note about the environment.
