# Z1 runbook — closing the redsox gate

The gate grades `tests/knob_gate/branch_manifest.toml` against the gate's
outcomes and `tests/knob_gate/sidecar_ledger.toml`. Every sidecar-backed row
needs a bathos run that satisfies all **14** step-1c conditions
(`tests/knob_gate/_coverage.py:287-340`).

This file used to say **nine**, and that number was stale rather than wrong at
the time: the weights and branch-controls conditions were added after it was
written, and nothing re-counted. Do not trust a count in prose here — run
`verify_wave.py`, which imports the gate's own predicates from
`tests/knob_gate` so its report cannot drift from what the gate actually
checks, and which names every condition separately.

## The ordering constraint, which is the thing that bites

**Since 261008 staleness is per row, not global.** Before that, one commit under
any scoped path invalidated every row, whatever it contained and whichever
vehicle it could not possibly affect: a one-line docstring change to
`src/aminx/cli.py` (04d689b3) discarded three hours of parity evidence, and
PR #205's edits under `src/aminx/families/potts_mpnn/` would have discarded the
four LASEr rows (7.75 h of compute) as well.

A row is now stale when a path changed since its run commit is in **that row's
import closure** (`tests/knob_gate/_closure.py`): the static import closure of
`scripts/parity/<slug>.py` over `src/` and `scripts/parity/`, plus the edges
declared for the slug in `tests/knob_gate/closure_edges.toml`. The gate uses
`row_is_stale` (`_coverage.py:_validate_sidecar`), and so do `verify_wave.py` and
`stale_rows.py`.

    uv run python3 scripts/redsox/stale_rows.py     # which rows must re-run, and why
    scripts/redsox/launch_wave.sh --stale-only 1    # launch only the stale slugs of a group

It over-approximates and **fails safe**: anything the scanner cannot prove
irrelevant makes the row stale, never fresh.

- `pyproject.toml`, `uv.lock`, `scripts/recapture/`, `aminx-oracles/` stay
  **global** (any change invalidates every row).
- A non-`.py` file under `src/aminx/` or `scripts/parity/` (packaged weights,
  fixtures) is global unless a slug declares it as `data`. A vehicle's own
  `.bth.toml` is the exception: `sidecar_sha256` already pins it per row.
- A dynamic import (`import_module(<variable>)`, `spec_from_file_location`,
  `importlib.resources.files`, entry points) inside a closure makes the closure
  unsound, and the row falls back to the global rule, unless the slug lists that
  file under `dynamic_ok` with its reason.
- `tests/port/` is **no longer scoped for rows**: no graded vehicle imports from
  it, and the port suite is re-run by the gate's step 2 on every gate run.

The declarations are reviewed claims. `closure_hook/` is how a real run checks
them: `launch_wave.sh` records every repo file each vehicle process actually
loads (`loaded_files.json` beside `branch_controls.json`), and `verify_wave.py`
fails the row if a loaded file is outside its closure. It edits no vehicle,
because a vehicle is in its own closure. **A run from before the hook has no
such file, and `verify_wave.py` reports that without failing it**, so the first
wave after this change is the first one that validates the rule against a real
execution rather than a static argument.

Shared core code (`src/aminx/__init__.py`, `host/runner.py`, the model
packages every vehicle imports) is in every closure, so changing it still
invalidates every row. That is the intended behaviour, and
`test_closure_selftest.py` pins it as a negative control.

`tests/knob_gate/` and `scripts/redsox/` are **not** scoped, so manifests,
ledger ids and this tooling can be edited freely before or after the wave.

## Environment the gate box needs, which is not in `pyproject.toml`

Two prerequisites live only in the venv and the launcher. Neither is declared as
a dependency, and that is deliberate: `pyproject.toml` and `uv.lock` are both
**scoped**, so declaring either would invalidate all seven ledger rows at once.
They are installed by hand and survive because everything here runs
`uv run --no-sync` — a bare `uv run` re-syncs and drops them.

1. **`redsox` must be importable in the venv the gate runs in.**
   `tests/knob_gate/test_knob_superset.py::test_u1_reachability_present` imports
   `redsox.checkers`. Measured 261002: the b7i checkout had it and the **sprint**
   checkout — the one the gate runs in — did not, so that test would have failed
   under the gate while passing everywhere it was tried.

       ssh titanix 'cd ~/projects/aminx-sprint-git && uv pip install -e ~/projects/redsox'

   Venv-only; `git status` stays empty afterwards, so the freeze survives.
   Check it with `uv run --no-sync python3 -c 'import redsox'` before a gate run.

2. **The gate needs a GPU, and must not use GPUs 0 or 1.**
   Run it with `CUDA_VISIBLE_DEVICES=2,3`.

   Two separate facts, both measured 261002:

   * It needs the GPU. Forcing `JAX_PLATFORMS=cpu` fails `declayer_f64` — the
     wave compares against a SEALED f64 dump and misses `rtol=1e-12` by
     `1.13e-12` on one element of 512. Same wave, same ids, default backend:
     3 passed. Do **not** copy `JAX_PLATFORMS=cpu` from `launch_wave.sh`; that
     is there because the parity *vehicles* compare against a CPU torch oracle,
     which is a different job.
   * It must avoid GPUs 0 and 1. They are held by the vLLM backend
     (`VLLM::Worker_TP0/TP1`, ~21.5 GB each), leaving GPU 0 at 24022/24576 MiB.
     JAX defaults to device 0 and preallocates, so an unpinned run dies with
     `CUDA_ERROR_OUT_OF_MEMORY` across the whole `__nonport__` wave — fifteen
     tests, one cause, none of them a numeric failure. GPUs 2 and 3 are the
     compute pair on this box.

3. **`$POTTS_ORACLE_PYTHON` must point at the oracles venv.**
   `launch_wave.sh:100-102` exports it (and `LASER_ORACLE_PYTHON`,
   `AMINX_POTTS_ROOT`), so vehicles launched through the script are fine. A
   hand-rolled `bth run` is **not**: `potts_ddg_megascale`'s oracle arm imports
   upstream `run_utils`, which imports `seaborn` at module scope, and
   `_oracle_python` falls back to `sys.executable` when the variable is unset.
   Launched by hand it died in under a minute; through the launcher it just
   works. Prefer the launcher, and export these three if you must not.

## Running ONE port wave by hand

Do not `pytest tests/port/<file>.py` bare. `_resolve_port_target_path`
(`tests/port/conftest.py:71-81`) uses `AMINX_PORT_WAVE` when set and otherwise
falls back to `[tool.port] target` in `pyproject.toml` — a single default — so a
bare run silently gives every test the **wrong oracle**. It then fails as

    AttributeError: module 'port_reference_reference_port_selftest'
                    has no attribute 'OracleAbsentError'

which names neither the cause nor the fix (`a1_compare.open_dump` references
`oracle.OracleAbsentError` in its `except` clause, and the selftest reference
exports neither that nor `load()`). Filed as debt 2444.

Reproduce a wave exactly as the gate runs it, reusing its emitted lists:

    OUT=outputs/gate/<timestamp>        # or any dir gate_ids.py wrote
    files=$(cat $OUT/files_<wave>.txt | tr '\n' ' ')
    AMINX_PORT_WAVE=<wave> AMINX_REDSOX_SELECT=$OUT/ids_<wave>.txt \
      uv run --no-sync python3 -m pytest $files -o addopts= -q

Measured 261002: `test_declayer_f64.py` bare gives an `AttributeError`; the same
file under those two variables gives 3 passed. A comparison run bare is not a
comparison of anything the gate does.

## Sequence

1. **Freeze.** Land every scoped change. Confirm local == remote and nothing
   uncommitted under a scoped path.

2. **Run the wave**, in two groups so the box is not oversubscribed (five
   concurrent arms put load at 32-37 on 20 cores and slowed every one):

       ssh titanix 'cd ~/projects/aminx-sprint-git && bash scripts/redsox/launch_wave.sh 1'
       # when group 1 has exited:
       ssh titanix 'cd ~/projects/aminx-sprint-git && bash scripts/redsox/launch_wave.sh 2'
       # when group 2 has exited:
       ssh titanix 'cd ~/projects/aminx-sprint-git && bash scripts/redsox/launch_wave.sh 3'

   Group 3 is `potts_ddg_megascale` alone — the largest vehicle, 202804 rows
   over 371 per-PDB resumable units per arm, ~25 minutes at `e3bc540e`. It is
   isolated for CPU contention, not for duration.

   The launcher fetches and checks out the branch itself, refuses to start on a
   dirty tree, and checks every `--mutants` string against the manifest before
   anything launches. It must be run from a checkout that already has it.

   Group 4 is `protonpotts_parity` alone (spec 261007 section 50): the seven ProtonPottsMPNN waves behind one row,
   47 controls named `<wave>.<control>`. It needs the sealed P4 dumps (`AMINX_PROTONPOTTS_ORACLES`, default
   `~/projects/aminx-oracles-protonpotts`) and the converted state (`AMINX_PROTONPOTTS_STATE`, default
   `~/scratch/v6_state.npz`); the launcher exports both:

       ssh titanix 'cd ~/projects/aminx-sprint-git && bash scripts/redsox/launch_wave.sh 4'

   The positive control is a separate, ungraded invocation:

       ssh titanix 'cd ~/projects/aminx-sprint-git && bash scripts/redsox/launch_wave.sh positive'

3. **Collect run ids.** Each vehicle's log ends with a JSON object carrying
   `run_id`:

       ssh titanix 'for v in laser_proofread_parity laser_decode_e2e potts_ar_refine_exact \
         potts_ar_decode potts_energy_parity laser_score_parity potts_ddg_megascale; do
           echo "$v $(grep -o "\"run_id\": \"[^\"]*\"" ~/${v}_<sha8>.log | tail -1)"; done'

4. **Verify BEFORE writing a row.** This is the step the ledger header asks for
   and that nothing made easy until now:

       uv run python3 scripts/redsox/verify_wave.py --run <slug>=<run_id> ...

   It reports all 14 conditions separately with observed vs expected, and
   prints argv-only / rows-only on a mutant mismatch. `check_branch_coverage`
   only ever says `pass` or `instrument_invalid`, which cannot distinguish
   "the runs are stale" from "the manifest is malformed".

5. **Write the ledger**, one `[sidecar.<slug>] bth_run_id = "..."` per slug, and
   run the gate:

       ssh titanix 'cd ~/projects/aminx-sprint-git && \
         git checkout -- .praxia/audits.jsonl && \
         CUDA_VISIBLE_DEVICES=2,3 bth run --project-slug aminx \
           -- uv run --no-sync python3 scripts/redsox/run_gate.py'

   Through `bth run … -- uv run …`, never `bash -c` (that records `script_path`
   as bash and leaves the outcome empty). `CUDA_VISIBLE_DEVICES=2,3` and the
   `audits.jsonl` restore are both load-bearing — see the prerequisites and
   trap 2 above.

   **`run_gate.py` exits 0 whenever it GRADED, fail included.** The verdict is
   the bathos `outcome` plus `{rc, step2_passed}`; exit status says only that
   the harness did not crash.

## What the gate measured, 261008 (run `e4bc0904` at `e56aa398`, GPUs 2,3) -- PASS

    {'n_ids': 1048, 'n_mutant_runs': 0, 'rc': 0, 'step2_passed': True}

Read from the cool-tier record: `status=completed`, **`outcome=pass`**, exit 0,
`git_dirty=False` before and after, adversarial check present, 50.8 min,
`sidecar_sha256` = the committed `run_gate.bth.toml`. **The first green gate.**

`e56aa398` is main `dab58900` (PR #165 merged, which carried the `laser_score`
tier-3 option the user chose on 261006) plus this ledger only, so no scoped path
separates the eight rows from the gated tree. The post-merge re-wave (backlog
#5780) re-ran all eight vehicles at `dab58900`: all `pass` by record, clean,
`verify_wave.py` grading all eight on all 14 conditions, and every clean-arm
number **bit-identical** to the 261004 rows -- merging #165 moved nothing on any
vehicle. The ungraded positive control (`89b9572d`) read 4.218847493575595e-15,
the instrument floor, as before.

Any later commit to a scoped path (P7's generic-alphabet refactor, the Potts
ONNX export fixes) invalidates every row again: land those together, then run
one re-wave and one gate.

## What the gate measured, 261004 (run `fb898d24` at `feade020`, GPUs 2,3)

    {'n_ids': 728, 'n_mutant_runs': 0, 'rc': 1, 'step2_passed': True}

Read from the cool-tier record: `status=completed`, `outcome=fail`,
`git_dirty=False`, adversarial check fired; tree still clean afterwards.

**Every wave exits 0 except `laser_score`, and its failures are exactly the 63
`test_tier_3_f32` cases (one per checkpoint x fixture pair) and nothing else** --
its other 189 tests pass, including the f64 tier. New since the 261003 gate:

- `laser_decode_step` now exits 0: its f32 band was measured (#2445, run
  `e47bfaab`) and applied to BOTH the target TOML and the test's own
  `_RTOL`/`_ATOL` (`2a4442d1`; see "where the bands actually live" below).
- `potts_order` (3 tests) and `laser_order` (1) -- the last three spec-named
  oracle tests, new waves in `719c33e0` -- pass inside the gate.
- All eight ledger rows were re-measured after the scoped batch and grade
  together (`feade020`); every pre-existing slug came back bit-identical, and
  the new `laser_proofread_unconditional_parity` (#2424) passed on its first
  graded run.

**The only thing between this gate and a PASS is a decision that is the user's**:
how to grade `laser_score`'s tier-3 f32 test. aminx and upstream make the same
f32 error against f64 (run `18b97f4f`), so no element-wise pairwise band can
separate that noise from a 1% defect. Options and a recommendation:
`.praxia/docs/decisions/261003_laser-score-tier3-f32-design.md`.

**Overnight incident, recorded so a later reader does not misread a ledger
id.** Titanix hit a global OOM at 20:18-20:29 on 261003 (wave group 1 plus
another session's AF3 memory harness). It killed `laser_proofread_parity`'s
clean arm, so run `f7887e2f` recorded `fail` with no cause; the re-run
`182ab6d4` passes bit-identically. It also killed a vLLM worker (the
`titanix-vllm-primary` server went down). Before a multi-arm wave, check
`free -g` and other sessions' RSS, and do not stack wave groups.

## What the gate measured, 261003 (run `fdd64720` at `b880882e`, GPUs 2,3)

    {'n_ids': 723, 'n_mutant_runs': 0, 'rc': 1, 'step2_passed': True}

Read from the cool-tier record, not the console: `status=completed`,
`outcome=fail`, `git_dirty=False`, sidecar resolved, adversarial check fired.

**Step 2 now passes.** Every knob test the alias map names was collected and
passed. The three `uncovered` fields from the previous run are closed:
`emit_etab` / `emit_dense_hJ` by explicit row kinds, and `tied_beta` first by an
`inert` row kind and then by deleting the field outright (`6a024d56`: upstream
derives it from the tied-positions weights and never takes it as config).

**`rc = 1` is exactly debt #2445**, the two LASEr tier-3 **f32** waves
(`laser_decode_step`, `laser_score`); every other wave exits 0. The
pre-registered floor measurement (run `e47bfaab`, `64274bcf`) split them:

| wave | f64 tier | f32 floor | verdict | action |
| :--- | :--- | :--- | :--- | :--- |
| `laser_decode_step` | passes 1e-8 | 1.08e-4 (904 ulp) | **pass** | atol 1e-7 -> **2e-3** (`2a4442d1`) |
| `laser_score` | passes 1e-8 | 5.36e-4 (4496 ulp) | `noise_reaches_defect_scale` | **held**, not widened |

Both waves compute the same function as upstream (f64 agrees to <8e-13). For
`laser_score`, a band wide enough to admit the f32 disagreement would be 0.61 of
an injected 1e-2 defect, so no band can separate them, and the pre-registered
decision was "escalate, do not widen". Whether that means aminx's f32 path is
less stable than upstream's, or that two f32 implementations simply disagree
this much, is the pre-registered attribution run
`scripts/analysis/laser_score_f32_attribution.py` (`6a24c7a7`).

**WHERE THE BANDS ACTUALLY LIVE -- the previous version of this section was
wrong.** It said the bands live in `tests/port/targets/<wave>.toml`. The
ASSERTED band is the test module's own `_RTOL` / `_ATOL`; `conftest.py` reads the
TOML policy only to stamp it into emitted tier verdicts. An amendment to the
TOML alone changes no assertion -- a CPU run of `laser_decode_step`'s
`test_tier_3_f32` after a TOML-only edit still reported `atol=1e-07`. Edit
both, and `tests/lint/test_port_tolerances_match_targets.py` now fails if they
disagree. Both are under `tests/port/`, a **scoped** path: one commit there
invalidates every ledger row.

### The previous run, at `9e2ba84f`

    {'n_ids': 694, 'n_mutant_runs': 0, 'rc': 1, 'step2_passed': False}

Same `rc = 1` cause. `step2_passed` failed on `uncovered` 3 of 51 (`emit_etab`,
`emit_dense_hJ`, `tied_beta`), none of them a missing test; full anatomy in
`.praxia/docs/research/261003_z1-gate-blocker-anatomy.md`.
`n_mutant_runs = 0` is correct in both runs: all 16 manifest rows are
`kind = "sidecar"`, and step 1b only iterates `kind = "pytest"` rows.

## Killing a gate run: kill the GROUP, not the script

`run_gate.py` spawns one `uv run pytest` per wave, and those children survive a
`pkill -f run_gate.py`. Measured 261003: killing the parent that way left

    uv run --no-sync pytest -o addopts= tests/port/test_laser_decode_step.py

orphaned for **10.5 hours**, holding a **48 GB deleted file** on `/tmp` (tmpfs,
so 48 GB of RAM) plus 2.4 GB of GPU 0. `df` showed `/tmp` at 50G used while `du`
saw 2.3G — that gap IS the signature, since a deleted-but-open file has no path
left to walk. Everything on the box then failed with
`OSError: [Errno 122] Disk quota exceeded`, which looks nothing like its cause
and reads like a full disk needing someone's checkouts deleted.

The launcher uses `setsid`, so each run is its own process group. Kill that:

    ssh titanix "ps -eo pgid,args | grep '[r]un_gate.py'"   # get the PGID
    ssh titanix "kill -- -<PGID>"                            # whole group

To find a leak after the fact, the deleted file is only visible through
`/proc`:

    for p in $(ls /proc | grep -E '^[0-9]+$'); do
      ls -l /proc/$p/fd 2>/dev/null | grep -q deleted && echo "$p"; done

SIGTERM did not end it — it took `kill -9`. Check the process is actually gone
rather than trusting the kill, and do not kill PIDs you did not start: the vLLM
workers on GPUs 0 and 1 share this box.

## Traps, each of which yields a run that looks perfect and is discarded

1. **`--mutants` must be in argv.** `_argv_mutants` returns `None` when the flag
   is absent, so a run relying on the script's default mutant list fails.
2. **The tree must be clean**, or every run records `git_dirty = true`.
   `.praxia/audits.jsonl` is TRACKED and **the port tier appends to it**.

   Measured 261002, after getting it wrong in both directions, so the scope is
   worth stating exactly. `tests/knob_gate` + `tests/lint` leave the file
   byte-identical — sha256 `44642583f94f5d94` before and after, 206 lines both
   times — which is what made "anything running pytest appends to it" look
   false. But `tests/port/` appends heavily: a gate run plus two `tests/port/`
   invocations added **259** lines, each a
   `{"domain": "port", "rule_id": "parity_tier_1", …}` finding written by the
   port harness itself.

   So neither blanket claim holds. The operative rule: **any run of
   `tests/port/` dirties a tracked file**, and the gate runs exactly that. This
   also means a gate run leaves the tree dirty for whatever comes next —
   `launch_wave.sh` restores before launching for this reason, and a gate
   re-run needs `git checkout -- .praxia/audits.jsonl` first.
3. **The weights key must be the stable identifier**, not an absolute path;
   `scripts/parity/artifact_key.py` derives it. Controls written before 6cb81d49
   carry `/home/solab/repos/...` and never validate.
4. **A run id must be the FULL uuid.** `default_resolve_run` stats
   `run_<id>.parquet` by exact filename, but every place you read an id from —
   a vehicle log, a `bth` listing — shows the 8-char prefix. `verify_wave.py`
   now globs the prefix and prints the full id; before that it reported
   "no cool-tier parquet (try: bth compact)", which blames a subsystem that
   was never involved.
5. **A positive control must not ride along on a graded run.** `_argv_mutants`
   reads the raw string, so `scalar_both_off` lands in `listed` even though
   `laser_proofread_parity` excludes it from `n_listed`, `n_failed` and the
   controls dict. It has no manifest row (it must PASS; every manifest row must
   report `failed`), so `listed != row_ids` and the run is thrown out. The
   launcher now refuses this before anything starts.

## What the verifier measured, 261002

Run against real runs before the wave landed, so every result is a negative
control (nothing can be eligible until the freeze holds):

| slug | sha | weights key | verdict |
| :--- | :--- | :--- | :--- |
| `laser_score_parity` | `f11f2bf1` | absolute path | **FAIL** weights key resolves |
| `potts_energy_parity` | `6cb81d49` | `PottsMPNN/...pottsmpnn_20.pt` | PASS weights key resolves |
| `potts_ar_decode` | `6cb81d49` | `PottsMPNN/...pottsmpnn_20.pt` | PASS weights key resolves |

The same condition passing for one family and failing for the other is what
shows it discriminates. The three potts runs fail **exactly one** condition --
`no scoped path since` -- and pass every other, which is the freeze and is
exactly what the wave re-run is for. (This read "eight of nine" when written;
the substance is unchanged, but the count was stale, so it is stated as
"exactly one failure" rather than re-pinned to a number that drifts.)

Consequence: **`laser_score_parity`'s only passing run is ineligible.**
`f11f2bf1` is an ancestor of `6cb81d49` ("key oracle weights by a stable
identifier, not an absolute path"), so it predates the fix. Its run id must NOT
be written to the ledger as "already passed" — group 2 re-runs it. The stale
`laser_decode_e2e` id already sitting in `sidecar_ledger.toml` is ineligible for
the same two reasons and must be replaced, not kept.

All six vehicles do route weights through `stable_artifact_key`, directly or via
`potts_graded_common.py`, so every run at the frozen tree emits a resolvable
key. Verified by reading the sources, not inferred from the two that pass.

### The weights requirement is inert

`_weights_required(slug)` is `slug.startswith(("pottsmpnn_", "lasermpnn_"))`.
**No vehicle slug matches it** — all six are `potts_*` or `laser_*`; the only
`pottsmpnn_*` slug in the repo is `pottsmpnn_cfg`, a knob slug from a different
namespace. So the gate never *requires* a weights map; a run declaring none
passes step 1c. The per-key check is separate and unconditional, so weights
that are present must still resolve — which is the trap that actually fires.

Not changed here. Tightening a pre-registered pass criterion days before
writing ledger rows is the mirror image of loosening a tolerance to make a run
pass, and belongs in a deliberate decision, not a drive-by edit.

## Known coverage gaps

`tests/knob_gate/branch_manifest.toml` documents these inline. **One**
spec-listed sidecar vehicle is deliberately unlisted:

- `potts_refine` — clean arm fails at 0.9539 (debt #2417), and it sits in the
  spec's wave table rather than the sidecar table, so whether it owes a row is
  a real question rather than an oversight.

`potts_ddg_megascale` was the other, and is now **listed**: first run 261002,
`90f945fc-49fc-4474-af22-124f817188db` at `e3bc540e`, clean `max_abs_delta`
1.2207e-4 against a `<= 5e-4` band, control `skip_transpose_merge_pair` failing
at 11.6458 against a 5e-3 threshold. Listing it needed **three** edits, not one:
the manifest row, the `MUT` entry, and a group — `check_coverage` refuses to
launch if a manifest slug is reachable from no group, so the row alone would
have made every later `launch_wave.sh` exit 1.

`test_manifest_can_pass.py` asserts the manifest *can* grade to `pass` under
flawless stubbed runs, with a negative control that a mutant-set mismatch is
caught. It does not assert the manifest is *complete* — that is the gap above.
