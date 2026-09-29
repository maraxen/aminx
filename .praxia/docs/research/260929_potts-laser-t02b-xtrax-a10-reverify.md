---
title: T0.2b — xtrax 0.4.0a10 re-verification of the PottsMPNN/LASErMPNN composition spec
description: Claim-by-claim re-check of every xtrax-derived and aminx-anchor fact in the spec against installed xtrax 0.4.0a10 and main's 9e6c340a/#160, after the r13 rebase
task_id: 260929_potts-laser-xtrax-compose
created: 260929
---

# T0.2b report

Method: read-only inspection (no experiment, no finding beyond code facts). xtrax = installed
`.venv/lib/python3.14/site-packages/xtrax` (`xtrax-0.4.0a10.dist-info` confirmed; `pyproject.toml:26`
pins `xtrax[io,export]==0.4.0a10`, holds). Abbreviation `X/` = that site-packages path. `port/` = 
`/home/marielle/projects/xtrax/port` (checkout HEAD `5a6ab77`; `git diff 35c5100 HEAD -- port` is empty,
so port/ is unchanged since the spec's `35c5100` anchor).

Verdicts: holds / changed / false.

| # | Spec statement (r13) | Verdict | Evidence |
|---|---|---|---|
| 1 | `SinkSpec.run_id` is required (`run/sink.py:26`) | holds | `X/run/sink.py:26` `run_id: str` (no default), `:36-38` `__post_init__` TypeErrors on non-str |
| 2 | SinkSpec other fields / validation | holds (detail) | `sink.py:27-30` `output_dir=None, format="jsonl", flush_every=1, extension_schema=None`. Blank/whitespace run_id is rejected at ZarrStagingSink construction (`zarr_sink.py:151-157`, ValueError), not in SinkSpec |
| 3 | `derive_sink_spec` `run/sink.py:61-105`; precedence explicit run_id, then `run_spec.run_id`, then fresh id | holds | `sink.py:61-69` signature `(run_spec, *, run_id=None, output_dir, format="zarr", flush_every=1, extension_schema=None)`; `:100` `run_id or run_spec.run_id or new_run_id()` |
| 4 | `xtrax.run.RunSpec.run_id` is a static field (setting it retraces) | holds | `X/run/spec.py:21-22` `run_id: str \| None = eqx.field(default=None, static=True)` |
| 5 | `ZarrStagingSink.stage(key, attrs, **arrays)` + `take(key)` at `zarr_sink.py:254-313`; `_CORE_PROVENANCE_FIELDS` `:34` | holds | `zarr_sink.py:254-297` stage, `:299-313` take, `:34` frozenset {git_sha, git_branch, git_dirty, run_id, created_at}; attrs colliding with it raise `:231-238` |
| 6 | Key structure: key components stringified, joined by `/` into group path; `()` is the root | holds | `zarr_sink.py:331-332` (`"/".join(str(p) ...)`; empty path uses root) so `stage((), attrs=...)` writes root attrs and `("structure_0","0")` nests |
| 7 | (implicit) staged arrays are written on drain; `finalize()` after all drained | holds | `stage` drains when `staged_since_drain >= flush_every` `:296`; `finalize` raises if pending `:372-379`, or twice `:369-371`; `stage`/`drain` after finalize raise RuntimeError `:283-288,324-329` |
| 8 | (new fact, not in spec) `take()` only sees undrained keys | added | With `flush_every=1` every `stage` drains immediately (`:296`), so `take(key)` after `stage` is KeyError (`:309-313`). Drivers do not use `take` (spec §2.1); note added |
| 9 | Reopen with different run_id | changed | Spec T0.0 row assumed a deterministic spec-hash id and a *aminx* ValueError. xtrax itself raises `ValueError` when the root already carries a different `run_id` (`zarr_sink.py:214-222`); same id reopens (`mode="a"`, arrays `overwrite=True` `:334-340`). Main mints a fresh id per call, so re-running into an existing output dir now raises the xtrax error (see 11) |
| 10 | Provenance stamped on root and each drained key group | holds | `zarr_sink.py:227,347-352` (root full record; per-key `run_id`+`git_sha`) |
| 11 | T0.0 helper is `aminx.host.sink_ids.sink_spec_for(...)`/`spec_run_id` (sha256 of spec JSON); re-run policy ValueError with aminx text; `run_spec.run_id` never set | **false** | `src/aminx/host/sink_ids.py` does not exist; `grep sink_spec_for\|spec_run_id` over `src scripts` = 0 hits. Main (`9e6c340a`) calls `xtrax.run.derive_sink_spec(spec.run_spec, output_dir=..., format="zarr", flush_every=...)` directly at `host/streaming.py:88-91`, `host/runner.py:1280,1336-1341` (jacobian), `sampling/multistate_poe.py:48,689`. `RunSpec.run_id` is never populated, so precedence falls to `new_run_id()`: each call mints an UNLINKED id (commit message of `9e6c340a` says so explicitly). Re-run policy = xtrax's own ValueError (row 9), no aminx-level check |
| 12 | Spec-less sites: `DesignsWriter(..., run_id=None)` defaults to sha256 of resolved path; `jacobian_profile.py` uses sha256 of argv | **false** | `io/designs.py:74,93-95,107-109`: `DesignZarrWriter(..., run_id=None)` defaults to `xtrax.run.new_run_id()` (fresh, not path-derived), forwarded from `from_multistate_shapes`; builds `SinkSpec(run_id=self.run_id, ...)` directly |
| 13 | "All five call sites migrated" | **false** | Four migrated (`streaming.py`, `runner.py`, `designs.py`, `multistate_poe.py`). `scripts/analysis/jacobian_profile.py:187` still builds `SinkSpec(output_dir=..., format="zarr", flush_every=1)` with no `run_id` and raises TypeError at runtime (only remaining `SinkSpec(` without run_id repo-wide, `grep` excluding `.venv`). Not on the runner path; does not block T0.5 |
| 14 | Old anchors `host/streaming.py:80`, `host/runner.py:1225`, `io/designs.py:81`, `multistate_poe.py:605` | changed | Now `streaming.py:88-91`, `runner.py:1336-1341` (`jacobian` at `:1220`), `designs.py:107-109`, `multistate_poe.py:689` |
| 15 | Root attrs via `sink.stage((), attrs={...})` "as `streaming.py:104`"; root at `streaming.py:79` | changed | Root stage is `streaming.py:173` (`sink.stage((), attrs=root_attrs, **root_arrays)`); sink built `:88-91`; `model_family` attr `:102` |
| 16 | `make_axis_dispatch_via_xtrax` `tiling/dispatch.py:161`; docstring "not wired" stale (~15 call sites) | holds (count) | `src/aminx/tiling/dispatch.py:161` def; docstring still says "Not yet wired" at `:172`; 24 call-site lines under `src` outside dispatch.py. xtrax side `X/tiling/dispatch.py:31` `make_axis_dispatch(strategy, *, axis="", heterogeneous_axes=None)` |
| 17 | `AxisSpec` fields (`bucket_boundaries`) | holds | `X/tiling/plan.py:31-54`: `name, cardinality, default_batch_size, tile_granularity=1, heterogeneous=False, dedup_eligible=False, bucket_boundaries=None, role=AxisRole.KNOWN`; boundaries validated non-empty/positive/strictly ascending `:64-75`; old `batch_size`/`granularity` names are deprecation shims `:77-92`. Frozen dataclass. NEW since a5 docs: `role` field (AxisRole; `AmbiguousAxisError` on UNKNOWN role at plan time `:209`) |
| 18 | `BatchPlanner`; ragged L via `AxisSpec(bucket_boundaries=...)` | holds | `X/tiling/plan.py:146-186` ctor `(memory_estimator, carry_specs, dedup_specs, heterogeneous_axes, budget)`; rule 1 = bucket_boundaries selects Bucket `:127-128`; `plan(specs) -> BatchPlan` `:188`; `MemoryBudget` `X/tiling/budget.py:34` |
| 19 | `AxisBoundary` `Fuse` (reductions) | holds | `X/stages/boundaries.py:84` `AxisBoundary(eqx.Module)` with static `fuse/tap/sink/materialize`; `Fuse` Protocol `:33-45` `__call__(stacked) -> Out`, pure, no io_callback; topology rules in docstring `:89-96` |
| 20 | `xtrax.stages` io_callback shim; "v1: no io_callback in drivers" | holds | `X/stages/_callback.py:28-113` import-time jax-version (`PINNED_JAX_RANGE=((0,10,2),(0,12,0))`) and signature check; Tap/Sink must import from it. Drivers do not use it (spec §2.1). Note: `AxisBoundary.sink` on a Vmap axis with `ordered=True` is a `PlanTopologyError`; irrelevant to drivers (Fuse only) |
| 21 | xtrax port contract anchors (`port_target.toml`, `conftest.py:64-71,103-114,159-171,214,232,277-286`, `test_parity_safe_map.py:53`) | holds | `port/tests/conftest.py`: `:64-71` `_resolve_port_target_path`, `:103-114` `verify_manifest_hash`, `:159-171` `_import_reference_algo`, `:214` timeout message, `:232` `add_marker(timeout)`, `:277-286` `emit_tier_verdict`; `test_parity_safe_map.py:53` `jax_enable_x64`. `port/` unchanged since `35c5100` |
| 22 | Runner entry points `host/runner.py:60,441,820,1143` | changed | `sample :79`, `score :480`, `inspect :881`, `jacobian :1220` |
| 23 | `score` `output_h5_path` NotImplemented `:489-491`; `make_score_fn` `:498-506`; inspect same | changed (drift) | score `:532-534`, `make_score_fn` import `:554`/call `:567`; inspect NotImplemented `:940-942`; (r13(e) drift class) |
| 24 | `_canonical_structure_ids_for_spec` exists and reads only `spec.inputs` | holds | `host/_sampling_helper.py:36-45` (also `_canonical_structure_id :22`, `_structure_ids_for_batch :48`) |
| 25 | `resolve_target_samples` | holds | `host/plan.py:382`; `resolve_chunk_size :425` |
| 26 | "MPNN AR ignores `decoding_order_fn`; order from wave schedule" (§1 row, `bundle_builder.py:76,187-188,237-243`, `kernel_dispatch.py:222-258`) | changed | main #160: `host/kernel_dispatch.py:253-268` draws a per-sample order (fixed-first, uniform over groups) and applies it via `with_decoding_order` (`inference/bundle_builder.py:381-413`, `WaveScheduleBundle.from_decoding_order(order, tie)` `:413`), honouring `run_spec.sampling.decoding_order_fn` when set. Only `schedule="fixed_n_to_c"` remains N->C without it. The FamilyDriver rule "setting `decoding_order_fn` with a driver family raises" is still a driver-side rule and stands |
| 27 | `WaveScheduleBundle.from_decoding_order` / `ar_mask_from_decoding_order` (new on main) | new | `types/bundles.py:357-397` `from_decoding_order(decoding_order: Int[Array,"L"], tie_group_map=None)`, jit/vmap-safe, W=L waves, tie groups decoded at earliest member; `utils/autoregression.py:318` `ar_mask_from_decoding_order(decoding_order, tie_group_map=None)` (ORDER array, not rank; built through the same wave schedule); `utils/decoding_order.py:111` `random_design_order`. Driver design does not depend on these |
| 28 | `model_family` Literal `run/specs.py:236`; derivation `:476-490` | changed (drift) | Literal at `run/specs.py:238` (`Literal["proteinmpnn","ligandmpnn"] \| None`); derivation `:485` |
| 29 | `decoding_order_fn` callable `run/specs.py:260`; `utils/decoding_order.py:21-24` | changed (drift) | field at `run/specs.py:262`; utils functions at `decoding_order.py:28,111,158` |

Not re-verified (outside T0.2b scope; r13(e) rule "code at HEAD wins" applies): remaining
`file:line` anchors into `src/aminx/**` in §1 rows not listed above.

Consequences for T0.5

- No blocker. T0.0 is functionally done on runner paths; drivers construct their sink with
  `derive_sink_spec(spec.run_spec, output_dir=Path(spec.run_spec.io.output_h5_path), format="zarr",
  flush_every=1)` like the existing sites.
- Design consequence: run ids are unlinked and fresh; re-running into an existing output dir raises
  xtrax's ValueError (drivers must not swallow it). T0.5b Zarr goldens must compare arrays plus
  non-provenance attrs only (already the spec's rule) and must write into a fresh dir per capture.
- Follow-up (small, not gating): `scripts/analysis/jacobian_profile.py:187` still TypeErrors.
