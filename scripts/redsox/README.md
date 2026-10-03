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

`_touches_scoped` is **global, not per-slug** (`_coverage.py:242-250`). One
commit under any of

    src/aminx/  scripts/parity/  scripts/recapture/  tests/port/
    aminx-oracles/  pyproject.toml  uv.lock

invalidates **every** row in the ledger at once, whatever the commit contained.
It is path-based and does not inspect the diff: a one-line docstring change to
`src/aminx/cli.py` (04d689b3) discarded three hours of parity evidence.

So ledger rows **cannot be accumulated incrementally**. The vehicle wave runs
*after* the last scoped commit, and nothing scoped may land afterwards.

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

## What the gate measured, 261003 (run at `9e2ba84f`, GPUs 2,3)

    {'n_ids': 694, 'n_mutant_runs': 0, 'rc': 1, 'step2_passed': False}

`n_mutant_runs = 0` is correct, not a miss: all 16 manifest rows are
`kind = "sidecar"`, and step 1b only iterates `kind = "pytest"` rows.

**`check_branch_coverage` returns `pass`.** Verified directly against the run's
own `outcomes.jsonl`, not inferred from a green test. All 16 rows graded, all
seven sidecar vehicles accepted — every clean arm passed and every negative
control failed. The ledger work is done and the gate agrees.

Ten of twelve waves exit 0, including `__nonport__` at 87 passed / 1 skipped
(the golden device-skip) and all five potts waves. Two things stand between
this and a gate PASS, both filed, neither environmental:

| what | where | status |
| :--- | :--- | :--- |
| `rc = 1` | `laser_decode_step` and `laser_score` tier-3 **f32** | debt 2445 |
| `step2_passed = False` | `test_parity_ids_passed`, `uncovered` list | 3 scope decisions |

`test_branch_coverage` and 30 other `knob_gate` tests pass.

**The step-2 row above is CORRECTED as of 261003.** It used to read "6 unwired
alias rows, debt 2434", and both halves of that are now wrong: 2434 is resolved
(the LASEr noise is implemented), and `test_parity_ids_passed` fails three
independent conditions, of which `unwired` is no longer one. Measured on the
current tree — `unwired` 0 of 212, `unmapped` 0 of 212, `not_passed` 0 of 68,
`uncovered` **3 of 51**. So the surviving step-2 failure is `uncovered`, and it
is not a test backlog: zero of the three is a missing test.

  `emit_dense_hJ`, `emit_etab` — no alias row can ever name them.
  `test_rows_bijective` pins `{row.ref} == REF_F`, the **upstream**
  reference-surface field names, and `reference_surfaces.py` has no `etab` or
  `dense_hJ`. `emit_etab` is already implemented and already tested and the
  gate still cannot see it.

  `tied_beta` — one rename from green, and deliberately not renamed. Its only
  naming test is `test_fixed_mask_and_tied_beta_are_still_inert`; the prefix
  `test_knob_semantics_` is the gate's marker for a test that pins knob
  *semantics*, and that test asserts the knob is **inert**.

All three remedies (narrow `NEW_FIELDS`, add an aminx-only row kind, or accept
the `tied_beta` rename) change pre-registered gate scope, so they are the
user's call rather than an edit to make here. Full anatomy, with the tracked
run id and its negative control:
`.praxia/docs/research/261003_z1-gate-blocker-anatomy.md`.

The f32 failures miss `rtol=1e-4, atol=1e-7` by ~4.2e-5 and are
**device-independent** — identical on CPU and on GPUs 2,3 — so they are not the
device-numerics artefact that explained two earlier gate failures. The target
file declares `f32_atol_basis = "house-paired and NOT measured"`, so the band
itself was never derived from a measurement. Do not widen it to make the gate
pass; measure the floor as debt 2432 did elsewhere.

The bands live in `tests/port/targets/laser_decode_step.toml:29` and
`laser_score.toml:29` (`laser_encoder.toml:29` carries the same unmeasured
basis). `tests/port/` is a **scoped** path, so the amendment cannot be
committed while a ledger wave is in flight — one commit there invalidates every
ledger row. Measuring the floor does not touch those files, so the measurement
can run first and the amendment can land after the gate.

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
