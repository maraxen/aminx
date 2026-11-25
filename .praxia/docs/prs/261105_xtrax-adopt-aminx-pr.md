---
title: "PR Specification: Adopt xtrax.pack and xtrax.eda in aminx"
task_id: "261105_xtrax-adopt-aminx"
epic_id: 6497
branch: "deps/xtrax-b7-261105"
target_branch: "main"
pr: "https://github.com/maraxen/aminx/pull/270"
specification: ".praxia/docs/specs/261105_aminx-xtrax-adoption-spec.md"
date: "2026-11-06"
head: "2f9654a4"
status: "draft_pr"
---

# feat(pack,eda): adopt xtrax.pack and xtrax.eda in aminx

## Summary

This PR moves aminx to xtrax 0.4.1b7 and adopts two of its modules:

- **`xtrax.pack`**, opt-in via `aminx campaign worker|run --pack`. A finished campaign row becomes one xtrax unit: a `unit.json` record plus `members.tar.zst` and `members.idx.json`, in place of a loose zarr tree. This targets the file-count cap on shared scratch.
- **`xtrax.eda`** (`src/aminx/host/plan_eda.py`). Plan explanation, bucket analysis, panel rendering and export of plan artefacts, with sha256 verification of the export manifest.

`--no-pack`, the default, still runs the reference runner (`campaign.py: run_manifest_row` and `execute_manifest`) unchanged.

## Design

### Pack runner (`src/aminx/host/pack_shim.py`)
`run_packed_manifest_row` follows the reference runner's contract and changes only how a finished row is stored.

- **Locking:**
  - takes the per-row lock through `_campaign_lock_context`, with a heartbeat;
  - passes `--lock-backend`, `--lock-lease-seconds` and `--heartbeat-interval-seconds` through;
  - refuses a distributed lock backend under `--pack`.
- **Commit sequence:** the worker writes a `.partial.<attempt_id>` store. Before that store is renamed into place, the runner:
  - refuses it if it holds no array node (R-U1);
  - fsyncs it;
  - takes its `zarr_content_digest`.

  It then commits the record through xtrax's write-once `write_unit_record`, collects the sink arrays, and packs with `pack_unit(check=aminx_unit_check)`. `aminx_unit_check` keeps xtrax's record and content-digest checks.
- **Resume:**
  - a valid packed unit is reused;
  - a row finished by the reference runner (v3 done marker) is reused;
  - a unit whose inputs changed is quarantined and recomputed (`InputDigestMismatchError`);
  - a unit committed but not yet packed, or packed but not swept, is finished;
  - existing output with neither a record nor a marker is refused, not deleted;
  - pack corruption is a hard error, not a recompute.
- **Failure:** an exception writes `<unit>.unit.failed.json` beside the unit, never inside it. The stamp is cleared on the next success.
- **Legacy markers:** a `.done.json` beside a unit is copied into `meta.legacy_done_marker` when the record is written. It is renamed to `.done.json.packed` only after a successful pack, and only when the record holds identical content.
- **Summary:** `execute_packed_manifest` returns the reference summary shape:
  - counts every non-success status as failed;
  - reports reused and recomputed counts;
  - refuses an unknown `--row-hash`;
  - reduces each row result to hash, status, output path and reuse, so `campaign run --pack` prints JSON.

### Gates hook (`src/aminx/host/campaign.py`, one function)
`_collect_manifest_gate_state` now recognises a packed row: an output directory containing `unit.json`. It reads that row's digest through `packed_unit_gate_digest`, which:

- validates the unit;
- checks `meta.manifest_row_hash`;
- returns the `zarr_content_digest` recorded at commit.

That is the same algorithm the done marker records. It excludes per-run provenance attrs (`created_at`, `run_id`, `git_*`), so determinism gates compare packed reruns correctly. All other gate logic is unchanged.

### Adapter resolution
`resolve_unit_adapter` returns xtrax's `UnitJsonAdapter` for a packed unit and `AminxDoneMarkerAdapter` for a legacy marker. A custom `record_name` raises `ValueError`, because neither adapter would find the record.

### EDA (`src/aminx/host/plan_eda.py`)
- **Checkpoint verification:** the checkpoint is resolved and verified through `resolve_family_checkpoint` before the plan is explained.
- **Export manifest verification:** refuses several candidate manifests when it has to choose one itself, entries without sha256, and filenames that are not plain names. `export_and_verify` checks the `{prefix}_manifest.json` it wrote.

## Dependencies
- `xtrax[io,pack,export]==0.4.1b7` (exact pin, as before).
- The direct `flax` and `zarr` lines are removed; they come through xtrax.
- New optional extra `eda = ["xtrax[eda]==0.4.1b7"]` for the plotting stack. CI installs it with `--extra eda`; the eda tests skip without it.
- `onnx = ["xtrax[onnx]==0.4.1b7"]`.
- `uv.lock` is regenerated.

## Files

| Path | Change |
| :--- | :--- |
| `src/aminx/host/pack_shim.py` | New: pack runner, unit check, adapter resolver, failure stamp, gate digest |
| `src/aminx/host/plan_eda.py` | New: xtrax.eda adapter with export-manifest verification |
| `src/aminx/host/campaign.py` | Gates read packed rows (one function) |
| `src/aminx/cli.py` | `--pack/--no-pack` on `campaign worker` and `campaign run`; lock options passed through |
| `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml` | Pins, the `eda` extra, CI install |
| `tests/host/test_pack_shim.py` | 15 unit-level tests, including tamper, edited-record, empty-store and fill-value controls |
| `tests/host/test_pack_shim_runner.py` | 16 runner/CLI/gates tests: rerun, gates, provenance-invariant digest, quarantine, legacy reuse, lock contention, pack failure and resume, interrupted sweep, CLI JSON summary, refusals |
| `tests/host/test_plan_eda.py` | 11 tests, including corrupted export, wrong prefix, missing sha256, path traversal |
| `scripts/analysis/verify_xtrax_adoption.py` + `.bth.toml` | Pre-registered adoption check: round trips and negative controls |

## Verification
- **Tests (titanix):** `tests/host` and `tests/cli` at `2f9654a4` gave **740 passed, 10 skipped, 0 failed**. The venv had the `cpu`, `dev`, `tests`, `eda` and `laser` extras.
- **Bathos:**
  - run `341c3456` recorded outcome **pass** at `2f9654a4` with the sidecar bound (`stage_name = "validation"`);
  - run `e4e3cf5a` crashed on a missing import, since fixed in `7222fe8e`, and is kept in the catalog as a failed run.
- **Review:** three rounds.
  1. Self-review found 4 blocking defects, fixed in `a398ab30` and `c9021fb6`.
  2. An independent reviewer confirmed 4 more, fixed in `6e266e32`:
     - `campaign run --pack` crashed on non-JSON results;
     - the gate digest included per-run provenance, so packed reruns never matched;
     - a pack interrupted before its sweep was never finished;
     - an unknown `--row-hash` was silently skipped.
  3. The verdict was MERGE. Its two remaining low items were fixed in `2f9654a4`: digest before rename, and arrays holding only fill values.

## Known limits
- `--pack` supports only the `local_fs` lock backend; a distributed backend is refused.
- Custom unit record names cannot be indexed, so the adapter resolver refuses them.
- The bathos verify script covers the shim, eda and the negative controls, not the runner, CLI or gates. Those are covered by `tests/host/test_pack_shim_runner.py`.
- `ty` is advisory in this repo. The remaining diagnostics in `pack_shim.py` are in code copied verbatim from the reference runner.

## Related
- Epic #6497 (`261105_xtrax-adopt-aminx`).
- xtrax debts #3257 (adopt `xtrax.pack`) and #3143 (adopt `xtrax.eda`). #3142 (campaign tooling) is only partly addressed: the runner's packed storage and gates path, not the scheduler.
- aminx debt #3147 (file-count pressure from campaign stores).
