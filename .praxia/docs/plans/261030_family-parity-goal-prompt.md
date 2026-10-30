---
title: Goal prompt for MPNN family parity coverage and the stock-MPNN fix
description: Paste-ready goal prompt for a continuous orchestrated run that executes spec 261030_family-redsox-parity-coverage.
task_id: 261030_family-parity-coverage
status: active
created: 261030
---

# Goal prompt

Paste the block below as the session goal (or as the `/loop` prompt). The spec it executes is
`.praxia/docs/specs/261030_family-redsox-parity-coverage.md`; its section 6 holds the settled decisions.

```text
GOAL (task_id 261030_family-parity-coverage): give every aminx MPNN family redsox knob coverage and
a sealed bathos parity vehicle, and close the stock ProteinMPNN/LigandMPNN parity defects, as scoped
in .praxia/docs/specs/261030_family-redsox-parity-coverage.md. Section 6 of that spec is settled;
do not reopen it.

ROLE: you are the orchestrator. Load praxia:orchestration and praxia:praxia-agent-dispatch. Delegate
the work to subagents and keep only conclusions in your own context. Set the model on every dispatch
(Haiku for recon and implementation, Sonnet for review, Opus for phase-exit audits). Use one worktree
and one branch per track (wt/261030-<track>), each based on a fresh origin/main. Put the task_id in
every dispatch and every transduction_log call.

STATE: keep a state file at an absolute path under the main worktree's .praxia/loop/261030_family-parity/.
Per work item record status, branch, PR, bth run id, verdict, blocker. Read it at the start of each
iteration and update it at the end. Each iteration: pick the highest-priority unblocked item,
dispatch it, verify the result yourself, record it, and report one line on what moved.

PHASES
P0  Ledger sharding. One gate engine, one ledger per family: tests/knob_gate/ledgers/<family>.toml,
    with run_gate.py and verify_wave.py gaining --family and an aggregate mode. The existing ledger
    becomes the `potts` shard unchanged. First spike where the single-ledger assumption lives
    (unverified). Scope the global-staleness set (uv.lock, aminx-oracles/) down to what each row's
    closure actually reads. Make extract_upstream_knobs.py table-driven.
P1  Stock fixes. Critical path; run in parallel with P2.
    a. Re-run the adjudication probes for defects 7, 8, 9 on current main, under new sidecars.
       Decide from the reference's mechanism in ~/repos/ProteinMPNN whether 7 is an aminx defect or
       a reference property.
    b. Compare the bias slots in the HF_REVISION-pinned weights against zero for defects 2 and 3.
       Report only; an HF re-upload needs the user.
    c. Retire the generate_ar_mask rank/order hazard and pin export/wrappers.py with a test.
    d. Implement, or formally waive under a filed debt, the four MISSING clauses, each with a
       pre-registered probe.
    One PR per defect. Then e: one parity_validate_mpnn re-grade under a new sidecar. The expected
    grade is still FAIL because the O1 sampling prerequisite is missing; report it that way.
P2  ThermoMPNN/-D adoption: its controls become manifest rows, add --controls-out, alias rows, a
    decision on the float16 band, and a `thermo` ledger shard. No stock dependency.
P3  After P1 merges: Tier 1/2 vehicles (AbMPNN, HyperMPNN, CyclicMPNN, CA-only). A missing
    checkpoint is a hard failure under bth run, never a skip.
P4  FAMPNN, Caliby, AntiBMPNN. Spike first whether the upstream samplers accept injected noise
    without patching. Then sealed upstream oracle dumps with pinned hashes, graded stages, alias
    rows, and a filed debt before any deferred:<id> exclusion.
P5  Per-family waves, gate runs and ledger rewrites. Ask the user before launching.

RULES
- Pre-register every finding: a tracked script plus a committed .bth.toml with [outcomes] before
  the run. Use `bth run`, never `uv run bth` or `-- bash -c`. Verify from the run record, not the
  exit code.
- Do not run test suites locally, even narrow ones. Tests run on titanix. Before a launch, check
  free memory and RSS, pin GPU 2 or 3, set AMINX_LASER_ORACLE_DIR, and confirm the remote checkout
  is at the pushed SHA. Sync branches with git, not myxcel push. Use ruff --no-fix there.
- Long runs persist a completion stamp per unit, reuse only on matching sha256, and use one
  process and timeout per chunk.
- Read the code at file:line rather than trusting docstrings or subagent summaries, and tag
  each assumption verified or unverified in specs. Spot-check any claim that something is
  absent or dead.
- A change to shared model code makes redsox rows stale; batch such changes before any wave.
- A row graded green is not evidence until its ledger record resolves.

AUTHORITY: you may commit and push to wt/* branches, open draft PRs, and launch pre-registered
titanix runs on GPUs 2 and 3. You may not merge, push to main, run a full redsox wave, or upload to
HF without asking the user first (use AskUserQuestion).

STOP AND REPORT when: P1 and P2 are merged-ready, a spike falsifies a spec assumption, a probe
contradicts the spec's status table, or you need a user decision. Finish each report with the
branch, the PR, the run ids, and the next command.
```
