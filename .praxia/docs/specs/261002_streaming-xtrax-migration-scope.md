---
title: 'Scope: replace aminx''s custom streaming/campaign sampling path with xtrax-managed, preemption-safe execution'
description: 'Opus-tier scoping report (task 261002_streaming-xtrax-migration, debt #2421): premise check, current-state map, xtrax capability map, gap analysis, target architecture, preemption/resume design, staged plan, asks of xtrax, decisions. Includes the verified reuse-safety defect in the campaign row hash.'
status: draft
task_id: 261002_streaming-xtrax-migration
date: '261002'
backlog_ids: '2421'
adversarial_review: ''
---
# Scope: replace aminx's custom streaming/campaign sampling path with xtrax-managed, preemption-safe execution

## Provenance and what has been independently checked

This is a scoping report written by a read-only subagent (Opus tier; the `opus` model alias, exact model version not confirmed) from a brief that stated the user's decision (eliminate the custom streaming path in favour of xtrax, with preemption handled properly) and the standing rules for long-running work. **It is model output.** Every claim below carries a `path:line` from the agent; the table says which ones the parent session re-read.

| Claim | Status |
|---|---|
| Campaign row hash omits inputs, `checkpoint_id`, `random_seed`, the actual fixed mask/tokens and bias (`campaign.py:93-110`) | **Re-read by the parent: confirmed.** The payload covers `campaign_id`, `job_id`, chunk/sample ranges, `fixed_policy` (a name), conditioning flags, `multi_state_strategy`, `temperature`, `backbone_noise`, `state_weights` only. |
| Stale-schema done markers are deleted and recomputed, so debt #2148's "hard-errors" claim looks out of date | **Partly re-read: confirmed the helper.** `StaleDoneMarkerSchemaError` is documented "safe to invalidate and recompute" and `_invalidate_stale_output` removes the output and marker (`campaign.py:546-562`). That it is wired into `run_manifest_row` (agent cites `:1218-1224`) was not re-read. |
| xtrax has no resume/preemption layer for inference; `run/resume/sweep` verbs are training-only | Agent only. Not re-read by the parent. |
| The output path is `campaign_id/row_hash.h5`, so a changed run with the same row hash is reported `already_done` | Agent only (cites `campaign.py:315-316`). The hash payload gap is confirmed; the path derivation and the absence of any other guard were not re-read. |
| Seed/key dependence on batch composition and on PoE chunking | Agent only. |
| The sink at `runner.py:1584` is in `jacobian()`, not `inspect()` | Agent only; contradicts the parent's earlier statement, which was made from a grep without function context. |

Not run by anyone: any probe in section 10 below. Treat everything marked "agent only" as a lead to confirm before building on it.

---

## 0. The premise is partly wrong
1. **xtrax has no resume or preemption layer for inference work.** Its `run`, `resume` and `sweep` verbs only do training. `resume_verb.run_resume` reads an Orbax checkpoint and calls `Engine.fit_sync`, and mints a new run id (`xtrax/cli/resume_verb.py:57-120`). `sweep` re-runs every combination under a fresh `sweep_id` and skips nothing (`cli/sweep_verb.py:94-136`). `ResumableState` is optimizer state (`training/types.py:41-55`). The `stages` executor works inside a JAX trace, per axis, and persists nothing (`stages/executor.py:143-248`). "Migrate to xtrax" therefore means **building a work-unit layer in xtrax first**, not adopting one that exists.
2. **The campaign path already persists each unit as it completes.** One manifest row is one (arm x profile x ligand x sidechain x sample-chunk) unit (`host/campaign.py:970-1070`). Each row writes to `.partial.<attempt>`, is fsynced and content-digested, promoted atomically, then given a done marker (`campaign.py:1282-1296`). The real gaps are listed in section 1.
3. **Brief corrections:**
   - The sink at `runner.py:1584` is in **`jacobian()`**, not `inspect()`. `inspect()` and `score()` both refuse `output_h5_path` (`runner.py:710-712, 1245-1247`).
   - campaign's digest, fsync and canonical-JSON calls already come from `xtrax.run` (`campaign.py:24-30`). Only the lock and the marker schema are aminx's own.
   - `multistate_poe` materialises exactly one batch (the states) and asserts `len==1` (`multistate_poe.py:322-331`). Its custom loop is the noise x temperature cell loop, which buffers every cell and writes once at the end (`:645-774`).

Code read: the installed **xtrax 0.4.0a11** (`.venv/.../xtrax-0.4.0a11.dist-info`). The checkout at `/home/marielle/projects/xtrax` is **older**: it is on branch `skills/activation-parity-260929`, its newest release commit is 0.4.0a7, and it has no `export/` and no `_renamed.py`. `diff -rq` also shows `zarr_integrity.py` without the provenance exclusion. Every xtrax claim below is from the installed copy.

## 1. Current state map

| Piece | What it does / callers | Resume and preemption today |
|---|---|---|
| `host/streaming.py::_sample_streaming` (`:64-324`) | Called from `runner.sample` when `output_h5_path` is set (`runner.py:298-299`). Builds root attrs (schema, provenance, bias, grid lineage, seed hash; `:98-177`). Non-campaign mode stages one group per structure per batch (`:188-230`). Campaign mode loops over sample chunks and concatenates in memory per batch (`:231-297`). Calls `drain()` but **never `finalize()`** (`:299`). | No stamp. A preempted direct `aminx sample` leaves a partial store that looks valid. Re-running on the same path fails loudly, because `ZarrStagingSink` refuses a store holding another `run_id` (`zarr_sink.py:240-248`) and aminx never sets `run_id` (`derive_sink_spec` mints a new one, `run/sink.py:100`). |
| `host/campaign.py` | `plan_campaign_manifest` (`:876-1073`); `run_manifest_row` (`:1136-1314`), called by `aminx campaign worker` (`cli.py:1829`); `execute_manifest` runs **all rows sequentially in one process** (`:1332-1405`), called by `aminx campaign run` (`cli.py:1886`). | **Unit = row**, persisted per row. Done = marker schema is v2, `manifest_row_hash` matches, and `zarr_content_digest` matches (`:565-594`). A v1 or other stale schema is **deleted and recomputed** (`:546-562, 1218-1224`), so #2148's "hard-errors" title looks stale. A digest or hash mismatch raises a hard error. On SIGKILL the `finally` never runs: the `.partial.*` dir is orphaned and the lock blocks re-runs until the 1800 s lease expires (`:49, 383-389`). `campaign run` puts no timeout around a row. |
| `sampling/multistate_poe.py::sample_multistate_poe_campaign_row` (`:472-799`) | `run_manifest_row` routes rows with `len(inputs)>1` here (`campaign.py:1267-1275`). It duplicates `_sample_streaming`'s root-attr and lineage code (`:693-748`) and writes one `poe_fused` group. | Inherits the row marker. Within a row, all work is lost on preemption (acceptable, since a row is the unit). |
| `runner.py::jacobian` sink (`:1635-1783`) | Stages each structure, then fsyncs and digests at the end. | Digest is returned but no stamp is written. Out of scope for campaigns; worth noting. |
| Done marker (`campaign.py:597-619`) | `{schema_version, manifest_row_hash, attempt_id, output_h5_path, content_digest_sha256, lock_backend, completed_at}` | Records **no input hash and no digest-algorithm version.** |

**Reuse-safety defects:**
- The row hash payload (`campaign.py:93-110`) leaves out `inputs`, `checkpoint_id`, `random_seed`, `fixed_mask`/`fixed_tokens`, `bias`, and every other `sampling_spec` field. *(Payload confirmed by the parent session.)* The output path is `campaign_id/row_hash.h5` (`:315-316`). Consumers patch `sampling_spec` after planning (docstring `:196-199`). A re-planned or patched manifest with a different mask, seed or inputs therefore resolves to the **same path and is reported `already_done`.** This breaks the requirement that reuse be allowed only when inputs and artifact hashes match.
- **Two different hashes share the name `manifest_row_hash`.** The store root attr is `_grid_manifest_row_hash` (`streaming.py:149-151`; payload at `_sampling_grid_lineage.py:121-133`). The marker uses the campaign row hash (`campaign.py:111`). Their payloads differ.

**Seed derivation:**
- Base key = `key(random_seed)` folded with `_grid_job_seed_hash(job_id, ...)` (`_sampling_grid_lineage.py:169-180`). `job_id` comes from planning order (`campaign.py:1042`).
- Per-sample keys use `fold_in(base, absolute_sample_index)` (`plan.py:347-379`), so the single-structure path does not depend on chunking.
- The encode key uses `fold_in(base, structure_idx)` with a **batch-local** index (`kernel_dispatch.py:321, 345`), so it depends on batch composition.
- PoE folds `sample_start` into the base key and then draws `n_samples` keys (`multistate_poe.py:595, 653`), so **PoE output depends on chunking.** Changing chunk size or batch membership changes the numbers.

## 2. What xtrax offers (0.4.0a11)

| Need | xtrax symbol | Coverage |
|---|---|---|
| Zarr writer | `xtrax.run.ZarrStagingSink`, `SinkSpec`, `derive_sink_spec`, `make_sink` | Stage/drain/finalize. `create_array(..., overwrite=True)` (`zarr_sink.py:408-416`). **No append**, no reopen of a run. |
| Integrity | `xtrax.run.zarr_content_digest`, `fsync_tree/file/directory`, `canonical_json_bytes` | Covered. The digest skips provenance attrs (`zarr_integrity.py:110-127`). **No version constant**, and the digest semantics already changed between the checkout and the pin. |
| Run config | `xtrax.run.RunSpec` (has `run_id`; `run/spec.py:13-22`), `InputResolver`, `RuntimeBundle` | Config only. aminx subclasses `RunSpec` (`aminx/run/spec.py:21,138`). |
| Preemption | `xtrax.safety.preemption.PreemptionHandler` | Catches SIGUSR1/SIGTERM and calls `save_fn` once. It **overwrites, does not chain**, existing handlers (`preemption.py:42-43`). |
| Ledger | `xtrax.telemetry.ledger.RunLedger` | Records telemetry status, not unit completion. Kinds are limited to train/eval/export (`telemetry/record.py:81-84`). |
| Work-unit lock, stamp, verify-or-recompute, reuse report | - | **Not covered.** |
| CLI run/resume/sweep | `xtrax.cli.*` | Training only; not applicable. |

## 3. Gap analysis

| Behaviour | Status |
|---|---|
| Grid lineage, seed hash, `_base_sampling_key` | **Stays in aminx** (domain-specific and load-bearing for the numbers) |
| Campaign planner (arms, profiles, chunking into rows) | Stays in aminx |
| Provenance attrs (bias semantics, fixed provenance, aminx_version) | Stays in aminx; unify the two writers into one |
| Per-structure groups, `poe_fused` layout | Stays in aminx (Zarr layout unchanged) |
| Sample chunking inside a row | Stays in aminx. Append mode is only needed for memory, not resume. |
| Lease lock, heartbeat, CAS steal | **Needs xtrax change** (move the generic code from `campaign.py:333-533`) |
| Atomic partial-to-final promote, completion stamp, verify-or-recompute, reuse report | **Needs xtrax change** |
| Digest-algorithm versioning | Needs xtrax change |
| Per-unit process and timeout | Stays in aminx and SLURM (one subprocess or array task per row) |
| Preemption signal -> release lock | Covered in part (`PreemptionHandler`; needs chaining) |

## 4. Target architecture
- **xtrax gets a new domain-agnostic module, `xtrax.run.units`:**
  - `UnitLock` (lease, heartbeat, CAS steal; lifted from campaign.py)
  - `promote_unit(partial, final)` (fsync_tree, digest, rename, fsync dir)
  - `CompletionStamp` (JSON schema `unit_completion_v3`)
  - `check_reusable(stamp, unit_id, input_hash, path) -> Reuse | Recompute(reason)`
  - `ReuseReport`
- **aminx keeps the planner, the seeds, and one writer.** A single `write_sampling_unit(...)` replaces the duplicated root-attr blocks in `streaming.py:98-177` and `multistate_poe.py:693-748`, and `run_manifest_row` becomes a thin adapter over `xtrax.run.units`. `_sample_batch` and the key derivation are untouched.
- **Execution:** one unit per process. `campaign worker` stays the SLURM array element. `campaign run` becomes a driver that spawns `aminx campaign worker` per row with `subprocess.run(timeout=unit_timeout)`; the in-process loop is kept only behind `--in-process` for tests.

**Rejected alternatives:**
- **(a) Model sampling as an xtrax `TrainConfig` run and use `xtrax resume`.** The verbs are hard-wired to Engine, Orbax and an optimizer, and resume continues one run rather than skipping verified units.
- **(b) Append-mode sink with resume inside one store per job.** The store would stay mutable after completion, which breaks the "digest of a finished store" invariant. It would also need `run_id` reuse, which the sink refuses (`zarr_sink.py:240-248`). PoE keys depend on chunking, so resuming in the middle of a chunk risks changing the numbers. Rows already give per-chunk granularity.
- **(c) Keep everything in aminx and just patch it.** This is viable and is stages 1-2 below, but it leaves the parallel path that #2421 is about.

## 5. Preemption and resume design
- **Unit:** one manifest row (`manifest_row_hash`), unchanged.
- **Input hash (new):** sha256 over the canonical JSON of:
  - the full coerced worker `sampling_spec` minus `output_h5_path`;
  - the sha256 of each input structure file;
  - the sha256 of the checkpoint weights (plus `checkpoint_id`);
  - `_SEED_HASH_SCHEMA_PIN` and the sampling/grid schema versions;
  - a manually bumped `SAMPLING_NUMERICS_EPOCH`.

  It does not include the git SHA (see decision 1).
- **Stamp v3** (same file, `<path>.done.json`): v2's fields plus `input_hash`, `input_hash_components`, `artifact_digest`, `digest_algo` (xtrax version plus a digest-version constant), and `producer {aminx, xtrax, git_sha}`.
- **Reuse rule:** reuse only if all of these hold, else recompute:
  - the stamp schema is known;
  - `unit_id` matches;
  - `input_hash` matches the current run;
  - the recomputed `zarr_content_digest` equals the stamp's digest.

  On recompute, move the old store to `<path>.superseded.<ts>`; do not `rmtree` it as `campaign.py:558-562` does today. The execute report (v2) lists each reused unit as `{unit_id, path, input_hash, artifact_digest, stamp attempt_id, completed_at}`, and each recomputed unit with its reason.
- **Timeouts:** a per-unit wall-clock timeout in the driver plus the SLURM `--time` per array task. There is never a timeout around the whole campaign.
- **Preemption:**
  - A chained `PreemptionHandler` on SIGTERM releases the lock and marks the partial as abandoned.
  - On start, the worker garbage-collects `.partial.*` dirs whose lock is stale.
  - The lease drops from 1800 s to a few heartbeats (for example 300 s with 60 s beats), so a requeued task is not blocked for 30 min.
- **bathos:**
  - The sidecar must pin the manifest's sha256 and the full unit set before launch.
  - Resume re-checks the manifest against that pin and refuses if rows were added, removed or re-hashed.
  - Outcome criteria read only stamped units.
  - The reuse report is attached as a run artifact, and a resumed run is recorded as a child of the original.

  bathos itself needs nothing new beyond attaching the artifact. The agent did not verify the sidecar fields.

## 6. Staged migration plan
- **S0 Probes and golden fixtures** (no code change).
  - Capture golden stores for a tiny single-structure row and a tiny PoE row on current code. Compare arrays and attrs **excluding `aminx_version`**: that attr is digested (it is not a provenance-excluded name), so whole-store digests differ across aminx versions.
  - Count existing v1/v2 markers.
  - Risk: none.
- **S1 Input hash, stamp v3 and reuse report, inside aminx.** No deletion yet.
  - Compatibility: the reader accepts v2 according to decision 2; v1 keeps its recompute path.
  - Must-fail controls (each must be **rejected and recomputed**):
    - a store with one chunk file deleted, under a valid v3 stamp;
    - the same `manifest_row_hash` with `fixed_mask` changed in `sampling_spec`;
    - the same row with a different checkpoint weights file;
    - a stamp with an unknown schema;
    - a stamp whose `artifact_digest` has one byte flipped.
  - Must-pass control: an untouched unit is reused and appears in the report with its hashes.
  - Risk: hashing structure files changes nothing numerically, but a wrong component list makes reuse either too eager or never happen.
- **S2 Preemption hygiene and per-unit timeouts.**
  - Replace `campaign run`'s in-process loop with per-row subprocesses.
  - Add the SIGTERM handler, partial GC and a shorter lease.
  - Controls:
    - `kill -9` a worker mid-row, re-run within the lease window: it must proceed after the lease expires, must not reuse the partial, and must produce the golden store.
    - A unit exceeding its timeout is killed and reported failed while other units complete.
    - SIGTERM releases the lock.
  - Risk: stealing a lock from a live slow worker if the lease is too short.
- **S3 Move the generic code into `xtrax.run.units`.**
  - Release xtrax, bump the pin, delete `campaign.py`'s lock and marker code (about `:321-619`).
  - Compatibility: the xtrax stamp schema *is* v3, so stamps written in S1 stay valid.
  - Controls: the S1 and S2 suite passes unchanged against the xtrax-backed code, and a stamp written before S3 is reused after it.
  - Risk: cross-repo release coupling.
- **S4 Unify the writers.**
  - Delete the duplicate root-attr and lineage code in `multistate_poe.py:693-799` and `streaming.py:98-177, 301-323`.
  - Control: S0 golden equality (arrays and attrs minus `aminx_version`) for both row types.
  - Risk: silent attr drift, which is the history the code's own comments describe.
- **S5 Direct `aminx sample --output-h5-path`.** Run it as a degenerate one-row unit (stamp, `finalize()`). Control: a preempted direct run must not be accepted as complete.
- **S6 (optional) `output_h5_path` -> `output_path`** (#2137). Write both keys into rows and stamps for at least one minor version, and keep reading the old one. Coordinate with the downstream repo first.

## 7. Asks of xtrax
1. `xtrax.run.units` (lock, promote, stamp, verify-or-recompute, reuse report): every preemptible consumer needs it, and xtrax already owns the primitives underneath.
2. An exported `ZARR_CONTENT_DIGEST_VERSION`: digest semantics have already changed once, and without a version a semantic change looks like corruption.
3. A `ZarrStagingSink(fresh_only=True)` that refuses a non-empty `output_dir`: guards against writing into a stale partial (it currently opens with `mode="a"`).
4. `PreemptionHandler` should chain to earlier handlers and accept several callbacks: today it clobbers other handlers.
5. A `RunLedger` kind for sampling/inference: units cannot be ledgered with the current kinds.
6. An input-file sha256 helper: one shared definition for input hashing.
7. (Low priority) append mode on the sink: per-chunk writes inside a unit, needed only if a row's memory footprint becomes a problem.

## 8. Effort and risk
- **Sizes:** S0 0.5-1 d | S1 2-3 d | S2 2 d | S3 3-5 d (includes xtrax release) | S4 2-3 d | S5 1-2 d | S6 1 d plus coordination.
- **Hardest part:** choosing the input-hash components (decision 1) and the v2 policy (decision 2).
- **Could silently change results:**
  - S4 attr or array drift (caught by the golden test);
  - any change to `batch_size`, chunk size or `job_id` ordering, which changes keys (encode key and PoE keys);
  - stale-lock stealing that lets two writers share one partial (each attempt has its own partial path, so this is low risk).

## 9. Decisions only the user can make
1. **Should code identity be part of the input hash?** Recommendation: no git SHA. Use a manually bumped numerics epoch plus the weights hash, and record the git SHA in the stamp for audit.
2. **What happens to legacy v2 markers?** Recommendation: a one-shot `campaign adopt-legacy` command that verifies the digest, computes the input hash from the current manifest, and upgrades the marker to v3 marked `adopted_from_v2: true`. The default stays recompute.
3. **Digest mismatch: keep the hard error, or quarantine and recompute?** Recommendation: quarantine, recompute, and report it.
4. **Keep in-process `campaign run`?** Recommendation: only behind `--in-process`.
5. **Upstream to xtrax (S3), or stop after S2?** Recommendation: upstream, but only once S1 and S2 have been validated on real preemptions.
6. **Make PoE keys independent of chunking?** This would change the numbers. Recommendation: not now; document it.
7. **When to rename `output_h5_path`?** Recommendation: last, and dual-keyed.

## 10. Unverified, or where the brief was wrong
- **Brief errors:** as listed in section 0.3, plus "xtrax offers resume" (training only) and "campaign persists only at the end" (it persists per row).
- **UNVERIFIED, with the cheap probe that would settle each:**
  - (a) A row takes "more than a few minutes": time one real row, or read bathos run durations.
  - (b) mit_preemptable's grace window: `scontrol show partition mit_preemptable` and `scontrol show config | grep -iE 'killwait|preempt'`.
  - (c) Which downstream scripts read `done.json` fields or `output_h5_path`: grep the downstream repo for `done.json`, `content_digest_sha256` and `output_h5_path`.
  - (d) Whether existing v2 markers still verify under the 0.4.0a11 digest: for a sample of completed rows, compare `zarr_content_digest(path)` to the marker (read-only). If they fail, every resume will hard-error today.
  - (e) Whether the downstream SLURM jobs use `campaign worker` per array task or `campaign run`: there is no campaign submit script in aminx's `scripts/`.
  - (f) bathos sidecar fields for a manifest pin and child runs.
  - (g) Whether the store is identical regardless of `flush_every`: check with the S0 golden comparison.
- **#2148:** the current code recomputes stale markers rather than hard-erroring, so the debt item may be outdated.

### Critical files for implementation
- `src/aminx/host/campaign.py`
- `src/aminx/host/streaming.py`
- `src/aminx/sampling/multistate_poe.py`
- `src/aminx/host/_sampling_grid_lineage.py`
- xtrax (installed 0.4.0a11): `run/zarr_sink.py`, `run/zarr_integrity.py`, `safety/preemption.py`
