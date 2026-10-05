---
title: "Merge readiness: browser split export branch onto main"
description: "What actually blocks merging feat/t11d-session-release, measured rather than assumed — including an export-safety fix the branch has and main does not."
task_id: 260926_browser-export-loop
status: active
created: 260930
---

# Merge readiness — `feat/t11d-session-release`

Measured 2026-09-30. Every claim below comes from a command, not a recollection.

## Divergence

- Merge base: `a5ced9c7`
- Branch ahead: **153 commits**, 332 files (69 commits from 2026-09-29 onward)
- **Main ahead: 2 commits** — `79956bd0` ("Debt sweep + random decoding order by default;
  parm7 trajectory loading, release 0.2.0a2") and `626722e3` (bathos project id)

`git merge-tree --write-tree main HEAD` returns a single tree OID with **no conflicts**.
The merge is textually clean. That is not the same as safe — see below.

## Only two source files genuinely overlap

`src/aminx/inference/sample_autoregressive.py` and `src/aminx/utils/autoregression.py`.

Three things that *look* like branch deletions in a `main..HEAD` diff are not:
`host/schema_versions.py` and `utils/decoding_order.py` were **added by main** after the
merge base, and the branch never touched either. Anyone reading that diff cold will
misread them as removals.

## The one thing that actually matters

`generate_ar_mask` differs between the two, and **Graph W bakes it into the exported
ONNX**. The direction is the opposite of what "main is newer" suggests:

| | `generate_ar_mask` group ordering |
| :--- | :--- |
| main | `jnp.argsort(jnp.where(group_present, group_first_occurrence, N + 1))` |
| **branch** | `jax.lax.sort((key, index), num_keys=2, is_stable=False)` — index tiebreak |

**The branch carries the export-safety fix; main does not.** Every absent group shares the
sentinel key `N + 1`, so ties are the common case rather than an edge case, and
`jnp.argsort` defaults to `stable=True`, whose tie order IREE does not honour (xtrax
export-safety rule `sort-stability`; same fix as `features.top_k` in PR #156). The
branch's version is also the one every validation run measured.

Main additionally introduces `_position_wave_index`, `ar_mask_from_decoding_order` and
`decoding_order_from_wave`, which the branch predates and does not have.

## What this implies for the merge

A clean textual merge would take main's new functions *and* the branch's
`generate_ar_mask`. That combination has never been run. Two consequences:

1. **Keep the branch's `generate_ar_mask`.** Taking main's would silently reintroduce the
   tie-order dependence the export was fixed to remove, and no conflict marker would
   appear to warn anyone.
2. **Re-export and re-validate — do not merely re-validate.** The `.onnx` files embed this
   function. If the merged source differs at all from what was exported, the existing
   graphs are stale and the manifest hashes no longer describe the code.

Recommended sequence (cherry-pick, per the project's no-merges rule):

1. Cherry-pick the branch onto current main.
2. Re-run `scripts/browser_validation/p07_split_export.py --buckets 128 256` — new
   manifest, new hashes.
3. Re-run **G1** (`p07_split_parity.py`), ~25 min. It compares tokens EXACTLY, so it is
   the gate that would catch a decoding-order semantic change.
4. Re-run **G1c** (`p07_split_browser_parity.py`), ~5 min, as browser confirmation.
5. Only then treat the handoff's numbers as still current.

A G1 failure after the merge would be informative, not alarming: it would mean main's
decoding-order default changed AR behaviour, and the export needs re-validating against
the new default — not that the split is wrong.

## Not blocking the advisor handoff

`docs/browser_integration.md` and `.html` are self-contained, every figure cites a bathos
run id, and those runs are pinned to commits on this branch. They can be shared before any
merge happens.

## Open questions for the user

- **`.onnx` binaries**: measured 11.9 MB (split, both buckets) vs 12.1 MB (both monoliths)
  as git objects — nearly identical, because the monolith's redundant structure compresses
  to 24% of raw while the split's dense weights only reach 89%. The repo already routes
  `*.eqx`, `*.npz`, `*.array_record`, `*.tar.gz` through LFS and carries 110 MB of model
  params, so `*.onnx` in `.gitattributes` follows existing convention.
- **99 of the 332 changed files are `outputs/`** run records. Whether those belong on main
  as provenance, or should be reduced to manifests plus passing records, is a judgement
  call nobody has made yet.
