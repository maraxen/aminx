---
title: Triage of the 11 late-committed sidecars
description: 'Run 83935786: 9 of 11 late sidecars graded against byte-identical committed criteria; 2 flagged cases resolve to feature-branch commits and an error-run schema fix — no post-hoc registration found.'
status: active
task_id: 261124_prereg-violation-triage
date: '261124'
verdict: 'no post-hoc registration found'
base_sha: 'd0fb83e4'
---
# Triage of the 11 late-committed sidecars

## Question

The v2 prereg-order audit (run `a5275269`, `prereg_order_violated`) found 11 sidecars committed
to main after a run that cites them. That settles "committed before running" as written. It does
not settle whether the criteria were chosen with the numbers in view. This triage settles that
second question where the recorded bytes allow it.

## Method

Pre-registered at `8ec75d2a` (`scripts/analysis/prereg_violation_triage.{py,bth.toml}`), then run
as bathos run `83935786`. The outcome evaluated to `criteria_edited_found`. The self-test passed
8 of 8 checks, covering must-fire, must-not-fire and never-guess cases. 49 runs were checked:
40 from the local catalog and 9 from titanix.

bathos stores `sidecar_sha256` as the SHA-256 of the sidecar's raw bytes
(`bathos/prereg.py:128`, `runner.py:1374`). For each run, those bytes were recovered from either
of two places:

- any version of the sidecar on any ref;
- the run's own dirty-run snapshot commit.

The recovered bytes were then compared with the version first committed to main. "Criteria"
means the parsed `[outcomes]` and `[result_schema]` tables. All 49 runs were recoverable.

## Result

| Sidecar | Late (h) | Pre-commit runs graded against | Clean run after commit? |
|---|---|---|---|
| ebm/collect_synthetic_parity_evidence | 60.9 | identical bytes (unknown) | no |
| ebm/lpla_biasing_check | 60.9 | identical bytes (error, pass) | no |
| ebm/real_decoy_ranking_benchmark | 60.1 | identical bytes (marginal) | yes (unknown ×2) |
| ebm/real_ddg_stability_benchmark | 60.1 | identical bytes (pass) | yes (unknown) |
| benchmarks/poe_fused_scaling | 28.5 | identical bytes (pass) | no |
| parity/e2e_run_api_parity | 22.8 | **differ** for 2 runs; see below | yes (fail, unknown, pass) |
| analysis/verify_done_markers | 1.1 | **differ** for 1 run; see below | no |
| browser_validation/export_safety_census | 1.0 | identical bytes (untraceable ×2) | yes (pass) |
| parity/thermo_mpnn_parity | 0.6 | identical bytes (error, pass) | no |
| benchmarks/bench_xtrax_vs_aminx_tiling | 0.06 | identical bytes (unknown ×2) | yes (inconclusive, pass) |
| benchmarks/bench_xtrax_vs_aminx_dispatch_gpu | 0.02 | identical bytes (unknown) | yes (15 runs) |

**The 9 sidecars marked "identical bytes" are commit lag.** Each pre-commit run graded against
exactly the bytes later committed. The criteria did not move.

**The 2 flagged sidecars are not post-hoc registrations either.** This was established by a
follow-up inspection of commit dates. That inspection is not part of the graded outcome.

- **`e2e_run_api_parity`.** The audit's "add" commit is the squash onto main (`f32c70b7`,
  2026-10-02T01:57Z). The sidecar had already been committed on feature branches
  `wt/e2e-parity-confirm` and `wt/e2e-parity-gpu`:
  - Version `22931213` was committed in `3a31203b` at 2026-10-01T00:19Z. Runs `0f282f6a`
    (03:07Z, fail) and `55ce2877` (06:28Z, pass) graded against it after that commit.
  - The criteria were then revised in a commit: `outcomes.marginal` and
    `result_schema.provenance` changed in `b10dee7a`, at 15:59Z.
  - Run `239505fd` (16:52Z, fail) ran at exactly that commit.

  So every run was pre-registered on its branch. The two early results belong to the superseded
  criteria. The citable result is a run under the committed criteria; the latest is `fce94f47`,
  pass, 2026-11-18.
- **`verify_done_markers`.** The first run (`878bde95`) graded against a schema without the
  `negative_control_mode` field and recorded `error`, so no graded number existed. The field was
  added, and the re-run `6588d9e9` passed against the exact bytes committed about 4 h later.
  That is commit lag after an error run.

## Limits

- Identical bytes prove the criteria a run graded against equal the committed criteria. They
  cannot prove the criteria pre-date an earlier, untracked look at the numbers.
- "First committed version" follows the audit's own add-commit definition: author date, on
  main's history. For a squash-merged sidecar this can name a later commit than the real
  registration, as `e2e_run_api_parity` shows. The audit (`prereg_order_audit_v2`) would give
  fewer false positives if it also checked the run's recorded sidecar hash against branch
  versions.

## Follow-up

- The procedural fix stands: commit the sidecar before `bth run`, on whatever branch is current.
- Outcomes recorded before commit with no clean run afterwards: `poe_fused_scaling` (pass),
  `lpla_biasing_check` (pass), `thermo_mpnn_parity` (pass, later superseded by `_v2`),
  `collect_synthetic_parity_evidence`, `verify_done_markers` (pass). By this triage their
  criteria are unchanged, so they need no correction. Under the audit's strict rule they remain
  pre-commit results.

