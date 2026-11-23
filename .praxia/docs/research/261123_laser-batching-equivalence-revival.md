---
title: LASEr GPU batching equivalence and pre-registration-order audit (revived from laser-confirm)
description: Records the 261007 LASEr batching-equivalence (sequence + chi1) and prereg-order audit runs whose scripts were stranded on wt/260929-laser-confirm; what the catalog does and does not hold.
status: active
task_id: 261123_aminx-branch-archive-review
date: '261123'
confidence: 'outcome labels verified against the bathos catalog; numeric values partly unrecorded (see below)'
sources: 'bathos runs 04e881a5, 294f93f6, 0a525d15, 7c07b6cd, bb4da3c6; tag archive/wt/260929-laser-confirm; commits 78e4f30c, 37736e15'
---
# LASEr GPU batching equivalence and pre-registration-order audit (revived from laser-confirm)

## Why this exists

The runs below were made on 2026-10-06/07 from branch `wt/260929-laser-confirm`. No PR was ever opened for
that branch, so their scripts and sidecars never reached main. The 2026-11-23 branch review found them stranded,
and this PR brings them back unchanged.

## Provenance check

Source for this section: bathos catalog queries on 2026-11-23 (`runs` table: `git_hash`, `git_dirty`,
`script_sha256`, `sidecar_sha256`, `outcome`, `output_paths`). None of it comes from commit messages.

All runs used commit 2eb50edb with a dirty tree. 2eb50edb is reachable from tag
`archive/wt/260929-laser-confirm`. For each run, the catalog's recorded `script_sha256` and `sidecar_sha256` equal
the sha256 of the file committed here, so these are the exact bytes that ran:

| Script | Run(s) | Outcome | script sha256 | sidecar sha256 |
|---|---|---|---|---|
| `scripts/analysis/laser_batch_equivalence.py` | 04e881a5, 294f93f6 | `batched_equivalent` | `5946b2fa…` | `03e4a407…` |
| `scripts/analysis/laser_batch_chi1_equivalence.py` | 0a525d15, 7c07b6cd | `chi1_batched_equivalent` | `e54f3067…` | `4847176c…` |
| `scripts/analysis/prereg_order_audit.py` | bb4da3c6 | `prereg_order_violated` | `57e84e3f…` | `e9181b9a…` |

Superseded attempts:
- 2b056576 graded `unknown`, the result of a null-handling bug, and ran against an earlier sidecar, `96cd1834…`.
- 698bf97a and 746a6cef ended in `error`.
- The audit runs d2034c13 and 5656a3ab used earlier sidecar versions.

## Findings

- **Sequence marginal: GPU `num_copies` batching matches the CPU sampler.** Run 04e881a5 measured mean cross-arm
  TV 0.0613 against a floor edge of 0.0650, with the control at 0.1621. These numbers come from the 78e4f30c commit message. The
  catalog stores the outcome label only, and the run registered no output files. 294f93f6 repeats the result at the
  same script and sidecar.
- **chi1 conditional on the modal amino acid: equivalent under batching.** Runs 0a525d15 and 7c07b6cd graded
  `chi1_batched_equivalent` against the pre-registered criteria. **The numeric values were never persisted**: they
  appear in no output file, no commit message and no doc. Re-run the script under this sidecar before citing a
  chi1 number.
- **Pre-registration order.** Run bb4da3c6 graded `prereg_order_violated`. 10 sidecars had been committed after
  their first run, and 2 of those first runs graded pass (`real_ddg_stability_benchmark`, 60.1 h late;
  `poe_fused_scaling`, 28.5 h late). The audit counts runs from every catalog it finds, and an earlier local catalog
  confirmed the same list. Source: the 37736e15 commit message.

## Scope and caveats

- The equivalence results cover the LASEr checkpoint and fixtures as of 2026-10-07. A sampler change since then
  could invalidate them. Re-run before relying on batching for a new campaign.
- The lint tests revived alongside these scripts are `test_laser_fixture_manifest.py`, `test_port_tolerance_coverage.py`
  and the hardened `test_port_tolerances_match_targets.py`. They were updated in this PR to the current
  `tests/port/targets`: `declayer_f64` is now banded, and `port_selftest` is found under `tests/port/selftest/`.
  The revival also tightened them: a pinned fixture count, recursive module discovery, and missing-component
  mismatches.
- `test_sidecar_structure.py` was deliberately not revived, because its registry of known-bad sidecars dates
  from 2026-10-06. That is a judgement made at revival time, not something the branch recorded.

## Known limitations of the revived scripts (left byte-identical on purpose)

The Cursor review of #302 found these. Fixing any of them changes bytes the cited runs graded, so each fix needs a
re-run under a new sidecar version. Until then the runs above stand as recorded, with these caveats:

- **`prereg_order_audit.py`: the date self-test is hollow.** It compares `git log --format=%aI` with itself and
  never calls `add_commit`, so it cannot detect `add_commit` regressing to committer dates. The audit itself is
  unaffected: `add_commit` does read `%aI`. Only the guard around it is vacuous.
- **`prereg_order_audit.py`: the earliest-run merge sorts timestamps as strings.** A space-separated timestamp
  sorts before an earlier `T`-separated one. That matters only if catalogs mix the two formats, which was not
  checked for run bb4da3c6.
- **`laser_batch_equivalence.py`: `first_batch_independent` is recorded but never graded**, and the chi1 script
  drops it. A sampler that returned identical copies within a batch could still grade `batched_equivalent`. The
  module header also says `mean_tv_chi1` is graded, but only `mean_tv` is.

