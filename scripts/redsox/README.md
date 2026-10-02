# Z1 runbook — closing the redsox gate

The gate grades `tests/knob_gate/branch_manifest.toml` against the gate's
outcomes and `tests/knob_gate/sidecar_ledger.toml`. Every sidecar-backed row
needs a bathos run that satisfies all nine step-1c conditions
(`tests/knob_gate/_coverage.py:287-340`).

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

## Sequence

1. **Freeze.** Land every scoped change. Confirm local == remote and nothing
   uncommitted under a scoped path.

2. **Run the wave**, in two groups so the box is not oversubscribed (five
   concurrent arms put load at 32-37 on 20 cores and slowed every one):

       ssh titanix 'cd ~/projects/aminx-sprint-git && bash scripts/redsox/launch_wave.sh 1'
       # when group 1 has exited:
       ssh titanix 'cd ~/projects/aminx-sprint-git && bash scripts/redsox/launch_wave.sh 2'

   The launcher fetches and checks out the branch itself, refuses to start on a
   dirty tree, and checks every `--mutants` string against the manifest before
   anything launches. It must be run from a checkout that already has it.

   The positive control is a separate, ungraded invocation:

       ssh titanix 'cd ~/projects/aminx-sprint-git && bash scripts/redsox/launch_wave.sh positive'

3. **Collect run ids.** Each vehicle's log ends with a JSON object carrying
   `run_id`:

       ssh titanix 'for v in laser_proofread_parity laser_decode_e2e potts_ar_refine_exact \
         potts_ar_decode potts_energy_parity laser_score_parity; do
           echo "$v $(grep -o "\"run_id\": \"[^\"]*\"" ~/${v}_<sha8>.log | tail -1)"; done'

4. **Verify BEFORE writing a row.** This is the step the ledger header asks for
   and that nothing made easy until now:

       uv run python3 scripts/redsox/verify_wave.py --run <slug>=<run_id> ...

   It reports all nine conditions separately with observed vs expected, and
   prints argv-only / rows-only on a mutant mismatch. `check_branch_coverage`
   only ever says `pass` or `instrument_invalid`, which cannot distinguish
   "the runs are stale" from "the manifest is malformed".

5. **Write the ledger**, one `[sidecar.<slug>] bth_run_id = "..."` per slug, and
   run the gate.

## Traps, each of which yields a run that looks perfect and is discarded

1. **`--mutants` must be in argv.** `_argv_mutants` returns `None` when the flag
   is absent, so a run relying on the script's default mutant list fails.
2. **The tree must be clean.** `.praxia/audits.jsonl` is TRACKED and anything
   running pytest appends to it, so a gate run leaves later vehicle runs
   recording `git_dirty = true`.
3. **The weights key must be the stable identifier**, not an absolute path;
   `scripts/parity/artifact_key.py` derives it. Controls written before 6cb81d49
   carry `/home/solab/repos/...` and never validate.
4. **A positive control must not ride along on a graded run.** `_argv_mutants`
   reads the raw string, so `scalar_both_off` lands in `listed` even though
   `laser_proofread_parity` excludes it from `n_listed`, `n_failed` and the
   controls dict. It has no manifest row (it must PASS; every manifest row must
   report `failed`), so `listed != row_ids` and the run is thrown out. The
   launcher now refuses this before anything starts.

## Known coverage gaps

`tests/knob_gate/branch_manifest.toml` documents these inline. Two spec-listed
sidecar vehicles are deliberately unlisted:

- `potts_ddg_megascale` — never run at any sha; run it before listing it.
- `potts_refine` — clean arm fails at 0.9539 (debt #2417), and it sits in the
  spec's wave table rather than the sidecar table, so whether it owes a row is
  a real question rather than an oversight.

`test_manifest_can_pass.py` asserts the manifest *can* grade to `pass` under
flawless stubbed runs, with a negative control that a mutant-set mismatch is
caught. It does not assert the manifest is *complete* — that is the gap above.
