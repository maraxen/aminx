---
title: "S1: aminx spec-system unification (one model, flat vocabulary as a codec)"
description: "Collapse the RunSpec pytree and the RunSpecification dataclass facade (synced via _sync_run_spec) into one nested RunSpec model with a field registry and generated codecs, migrated in PR-sized slices behind byte-identical goldens."
task_id: 261001_aminx-hub-ecosystem-specs
status: draft
created: 261001
owner_repos:
  - aminx
  - mistypotts
  - mpnn_ext
  - asr
  - hautespout
  - tev_design
related_specs:
  - S2
  - S3
  - S4
  - S5
  - S6
---

# S1: aminx spec-system unification

Shared inputs: `.praxia/docs/research/261001_ecosystem-hub-recon-brief.md` (decisions D1-D7; D4 is this
spec's mandate). This spec is SPEC ONLY; nothing in it executes yet (D7). Paths are relative to the aminx
repo root unless prefixed by a repo name. **Path convention after round 1:** paths are as read in the current
tree. The S3 rename (S3-07, which every S1 code item now follows, see D7) maps `src/aminx/` to
`src/molxmpnn/`; the S1 items apply by that mapping and S1-01 re-pins the anchors it records on the renamed tree.

## Goal and non-goals

**Goal (one sentence).** After this work aminx has exactly one in-memory run-configuration model (a nested
`RunSpec`, subclass of `xtrax.run.RunSpec`), one field registry that every codec and conformance check is
generated from, and no state-sync between two objects; every existing flat-keyword construction, flat JSON
file, campaign manifest and manifest-row hash keeps working byte-for-byte.

**Non-goals.**
- No change to model numerics, sampling behaviour, CLI flags, CLI names, or the `aminx` package name
  (renaming is S3; extraction of `src/aminx/ebm/` is S2).
- No JSON-Schema generator, no JS code generation, no browser knob document and no letter-keyed design-constraint
  lowering (S4 owns generation and, per the C3 resolution, S4-20 owns the knob document and its lowering; S1 only
  provides the registry plus the field-metadata hook they read, and a drift test until then).
- No new user-facing spec file format (JSON stays the only one; see Current state, "Spec files").
- No folding of `PottsRunSpec` / `PottsTRWRunSpec` into `RunSpec` (justified in Design D8; open question Q5).
- No deduplication of the Typer option lists in `src/aminx/cli.py` (a schema-driven CLI is S4/S5 follow-on).
- No behaviour fixes to the `or`-coalescing quirks (Design D9); they are preserved and surfaced as Q3.

## Current state

Every anchor below was read in this worktree (HEAD cb8ea2c6, a docs-only commit on top of origin/main e9a005b3;
the brief's d1210e4a is an ancestor of it) unless
tagged *unverified*. Counts marked *exploratory* come from unrecorded `rg`/AST probes and must not be cited as
findings; item S1-02 re-measures them under a bathos sidecar.

### Two systems plus two more vocabularies

1. **`RunSpec`** (`src/aminx/run/spec.py:138`) is an `eqx.Module` extending `xtrax.run.RunSpec`
   (`../xtrax/src/xtrax/run/spec.py:13-22`: `seed`, `axes`, `carry_specs`, `boundaries`, static `run_id`).
   It has sub-configs `io, resource, multistate, ligand, grid, precision, plan, sampling` (`spec.py:141-148`)
   and `encoding_fusion`/`decoding_fusion` (`spec.py:149-150`). Every sub-config defaults to `None`
   (`field(default_factory=lambda: None)`), so a bare `RunSpec` is not usable. It is **only ever built as a
   derived view**: `build_run_spec(spec)` (`spec.py:297-393`) reads a flat spec with `getattr(..., default)`;
   the only other construction site is the portable-JSON placeholder
   (`src/aminx/run/run_spec_portable_json.py:127`).
2. **`RunSpecification` dataclass facade** (`src/aminx/run/specs.py:190`) with subclasses
   `ScoringSpecification` (:527), `SamplingSpecification` (:582), `JacobianSpecification` (:718),
   `InspectionSpecification` (:749) and `TrainingSpecification` (`src/aminx/training/specs.py:18`). Roughly 88
   init fields on `SamplingSpecification` (documented at `src/aminx/host/spec_partition.py:6`; exploratory AST
   count 51 + 39 declared minus two `init=False`). Each `__post_init__` ends in `self._sync_run_spec()`
   (`specs.py:329`; calls at :522, :577, :713, :741) which **snapshots** the flat values into a `RunSpec`
   once (`_run_spec_synced` guard). There is no `__setattr__` hook, so any later attribute write leaves
   `spec.run_spec` stale (hazard; no production writer found by exploratory `rg`, see A19).
3. **`PottsRunSpec`** (`src/aminx/potts/spec.py:36`, frozen dataclass, own JSON) and **`PottsTRWRunSpec`**
   (`src/aminx/potts/_trw_spec.py:14`, frozen slots dataclass used as `eqx.field(static=True)`,
   `src/aminx/potts/model.py:98`). Neither touches the host planner or `RunSpec`.
4. **Browser "runspec"** (`browser/aminx-sampler/runspec_core.mjs:249-314`): a *third vocabulary* with keys
   `seed, temperature, bias_AA, bias_AA_per_residue, omit_AA, omit_AA_per_residue, chains_to_design,
   fixed_positions, tied_positions, decoding_order`. It is **not** a field-for-field mirror of `RunSpec`:
   letter-keyed, structure-relative maps are lowered to arrays inside the JS. `chains_to_design` fixes the
   other chains to native tokens, whereas the Python `chain_id` is a loader filter ("chain(s) to keep",
   `src/aminx/cli.py:1733`). The only Python twin of the lowering is a *script*
   (`scripts/browser_validation/p07_knobs_gate.py:297`, docstring "Python twin of buildP07Inputs"), not
   library code. The brief's description of this file as "a hand-written JS mirror of RunSpec" is therefore
   imprecise (correction recorded in A15).

### Evidence that the two-object design is the drift mechanism
- `RunSpec.seed` is read by `build_run_spec` as `getattr(spec, "seed", 0)` (`spec.py:381`) but no flat spec has
  a `seed` field (the knob is `random_seed`, `specs.py:258`), so `run_spec.seed` is always 0 while the real
  seed lives at `sampling.random_seed`. `derive_sink_spec` (`../xtrax/src/xtrax/run/sink.py:61`) reads
  `run_spec.run_id`, which no façade ever sets.
- `RunSpec.encoding_fusion` / `decoding_fusion` (`spec.py:149-150`) are declared but never passed in the
  constructor call at `spec.py:380-393`; the host reads the flat attributes instead
  (`src/aminx/host/plan.py:966,978`).
- The combine strategy has two homes inside `RunSpec` itself: `MultistateConfig.combine_strategy`
  (`spec.py:50,321`) and `SamplingConfig.multi_state_strategy` (`spec.py:124,364`).
- Host code mixes both vocabularies in one function: `spec.run_spec.sampling.*` for the fields RunSpec has and
  `getattr(spec, "average_node_features")` / `getattr(spec, "decoding_fusion")` for the ones it lacks
  (`src/aminx/host/plan.py:941-978`). Exploratory counts: ~85 `.run_spec` reads in src (sampling, io
  output/cache, precision.compute, plan.use_unified_driver) versus several hundred flat reads.
- A prior audit (`.praxia/docs/specs/260827_runspec-scaffolding-remediation-migration-map-re-authoring-and-xtrax-transforms-adoption.md:60-130`)
  already shows `GridLineageConfig` and `LigandConfig` have no production read site; the "27 dead fields"
  figure in the brief is stale (that spec re-baselines it to 11 after commit dd0e952 deleted three
  sub-configs). Both configs still exist (`spec.py:53-71`), so that spec's WS-A deletion was not executed.
  Exploratory probes found no alias reads (`rs = spec.run_spec` pattern) in src.

### Everything that depends on the flat vocabulary (the compat surface)
- **Runtime class dispatch**: `src/aminx/run/_exports.py:34-37` calls `isinstance(args[0], spec_type)` with
  `SamplingSpecification` / `ScoringSpecification`, so those names must remain classes.
- **Flat dataclass idioms**: `dataclasses.replace(spec, ...)` at `src/aminx/host/campaign.py:989` and
  `src/aminx/sampling/multistate_poe.py:650`; `dataclasses.fields(SamplingSpecification)` at
  `src/aminx/host/campaign.py:1857`, `src/aminx/host/spec_partition.py:108`, `scripts/audit/knob_matrix.py:50`,
  `tests/host/test_knob_differential.py:164`.
- **Duck-typed specs**: `build_run_spec(obj)` is documented as accepting any object
  (`scripts/benchmarks/bench_aminx_jax.py:144`, `tests/inference/test_plan_reuse_compiles_once.py:60`,
  `tests/inference/test_product_sharpness.py:143-149`). Many tests build specs as `MagicMock`/`SimpleNamespace`
  or hand-rolled classes rather than facades, e.g. `tests/scoring/test_averaged_parity.py:404-440`
  (`_MinimalScoringSpec`: flat attributes plus a hand-built nested `run_spec` namespace) and
  `tests/host/test_comp_unified_encoder_fusion.py` (~24 duck specs); exploratory count 113 constructs in 22 test
  files. These are not "construct-only": host readers rewritten to nested reads fail on a duck spec that lacks
  the nested attributes (round-1 objection C1). Handled by the duck-spec seam (D2, S1-23).
- **Post-construction writes** (exploratory): `spec.temperature = ...` at `scripts/benchmarks/bench_aminx_jax.py:169`
  and `scripts/benchmarks/bench_mixed_length.py:103`; `spec.run_spec = build_run_spec(spec)` at
  `scripts/benchmarks/bench_dedup_hetero.py:337,791`; `_BenchmarkSpec.run_spec = build_run_spec(...)` on duck
  classes at `bench_aminx_jax.py:148`, `bench_mixed_length.py:93`, `bench_inference_plan_latency.py:153`.
  Whether the `spec.x =` writers target a real facade or a duck class is settled by S1-02.
- **Persisted wire format**: campaign manifests carry a flat `sampling_spec` dict
  (`src/aminx/host/campaign.py:1055,1154,1259`) rebuilt through the plain constructor
  (`SamplingSpecification(**worker_payload)`, which rejects unknown keys by design). The manifest row hash is
  computed from *semantic values only* (`campaign.py:100-117`: model_family, ligand/sidechain conditioning,
  multi_state_strategy, temperature, backbone_noise, ...), so it survives any restructuring that preserves
  those values; done markers are keyed by that hash.
- **Five hand-maintained field classifications that must agree** (each is a historical source of silent
  knob loss, see `spec_partition.py:1-25`): `_NON_JSON_ROOT_FIELDS` / `_DERIVED_FIELDS` /
  `_coerce_field_value` whitelists (`src/aminx/run/spec_json.py:99,121,164`), `CAMPAIGN_OWNED_KEYS` /
  `EXCLUDED_WITH_REASON` / `DERIVED_FIELDS` / `_CALLABLE_FIELDS` (`src/aminx/host/spec_partition.py:46,68,96,166`),
  the portable-v2 hand parser (`run_spec_portable_json.py:142-261`), the CLI option lists
  (`src/aminx/cli.py:448-640,1029-1100`, and each `emit-*`/`run *` command), and the JS key reads.
- **Spec files**: JSON only. `run_specification_from_json` (`spec_json.py:249`) requires a `_spec_class`
  discriminator and iterates `fields(cls)` (`:231`), so unknown keys are silently ignored on this path
  (the campaign path deliberately differs, `campaign.py:1253-1259`). CLI surface: `aminx spec
  emit-sample|emit-score|emit-jacobian|emit-inspect|emit-potts|validate|roundtrip|portable-roundtrip`
  (`cli.py:1172-1623`) and `aminx campaign plan|worker|run|gates|ramp-*`. There is no `aminx run --spec FILE`.
  No TOML or YAML spec parser exists in the CLI (`rg "toml|yaml" src/aminx/cli.py` is empty).
- **Portable v2 JSON** (`run_spec_portable_json.py:35`) has exactly one production caller, `aminx spec
  portable-roundtrip` (`cli.py:1623-1638`), and no browser or hub consumer (exploratory `rg` over `browser/`,
  `site/`, `tools/`).
- **Downstream users the brief missed** (the brief says no sibling imports aminx; that is false for four
  repos outside its list): `../mpnn_ext` depends on aminx as a uv workspace member
  (`../mpnn_ext/pyproject.toml:12-15`: `aminx = { workspace = true }`, `members = ["external/aminx"]`, a vendored
  git submodule pinned to a SHA; exploratory: 57 files outside `external/` reference a `*Specification` class);
  `../asr` has the same shape (`../asr/pyproject.toml:23,94`: `aminx = { workspace = true }`; S3 A1 records its
  submodule); `../hautespout` is a third vendoring consumer per S3 A1 (*unverified* by S1: only S3's evidence;
  S1-28 re-reads it); `../tev_design` pins a built wheel (`../tev_design/pyproject.toml:66`,
  `wheels/aminx-0.1.0a24`, far behind; exploratory: 46 files reference a `*Specification` class; its
  `AMINX_PIN.md` names two gate scripts to run before any re-pin). Vendored consumers break **at their submodule
  SHA bump**, not when upstream shims are removed; `tev_design` breaks at its next re-pin, which from 0.1.0a24 is
  a re-pin project, not a note.
- **Blast radius** (exploratory `rg -l`): 96 files reference a `*Specification` class: 26 in `src/aminx`
  (cli.py 30 refs, host/runner.py 25, run/spec_json.py 17, run/specs.py 15, host/plan.py 12,
  host/campaign.py 12, host/spec_partition.py 11 ...), 37 in `tests/host`, 9 in `tests/run`, 5 in
  `tests/cli`, the rest in other test dirs, scripts, `examples/`, `docs/COMPOSITION_GUIDE.md`, README,
  CHANGELOG.
- **TRW fork**: `src/aminx/potts/_trw_spec.py` (77 lines) and
  `../mistypotts/src/mistypotts/potts_trw_spec.py` (77 lines) are byte-identical over the whole file
  (`diff` exit 0; the brief said first 40 lines). mistypotts uses it in seven other src modules plus tests
  and scripts (exploratory), and aminx does not depend on mistypotts (`rg mistypotts pyproject.toml` empty;
  it appears only in comments and recapture scripts).
- **EBM independence from this spec**: nothing under `src/aminx/ebm/`, `tests/ebm/` or `scripts/ebm/`
  references the spec system (one docstring mention, `src/aminx/ebm/langevin_schedule.py:353`).
- **xtrax contract stability**: `xtrax.run.RunSpec` and `derive_sink_spec` are textually identical at
  `v0.4.0a10`, `v0.4.0a11` and `origin/main` (empty `git diff` on `run/spec.py`, `run/sink.py`), so S1 does
  not depend on draft PR #174 (the a11 pin).

### Assumption Ledger

Spike harness note: the protocol's `scripts/loop/adversarial_metrics.py spike-run` does not exist in this
worktree, in `~/projects/aminx`, or in `~/projects/praxia` main (`ls` verified); copies exist only inside
unrelated praxia worktrees and running them would write outside this tree. The task also forbids heavy
commands, so no spike could be recorded. Rows that would need one are `UNVERIFIED` with an explicit `deferred:`
and are turned into the gate of item S1-03 / S1-02, which is sequenced before any item that depends on them.

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | `RunSpec` is only constructed as a derived view (`build_run_spec` and the portable placeholder), never directly by users or src | none (unified model still replaces both sites) | VERIFIED | read: src/aminx/run/spec.py:380; read: src/aminx/run/run_spec_portable_json.py:127 |
| A2 | `RunSpec.seed` is always 0 for facade-built specs because no flat field is named `seed` | Registry maps `random_seed` to the inherited `seed` anyway; only the motivation text changes | VERIFIED | read: src/aminx/run/spec.py:381; read: src/aminx/run/specs.py:258 |
| A3 | `RunSpec.encoding_fusion` / `decoding_fusion` are never populated by `build_run_spec` | none | VERIFIED | read: src/aminx/run/spec.py:149-150; read: src/aminx/run/spec.py:380-393 |
| A4 | The combine strategy has two homes inside `RunSpec` (`MultistateConfig.combine_strategy` and `SamplingConfig.multi_state_strategy`) | none | VERIFIED | read: src/aminx/run/spec.py:50; read: src/aminx/run/spec.py:124; read: src/aminx/run/spec.py:364 |
| A5 | `GridLineageConfig`, `LigandConfig`, `ResourceConfig`, `MultistateConfig` and most of `IOConfig` have no production read site outside the portable-JSON codec; no `run_spec` aliasing exists | none (unified model keeps these sub-configs as field homes; only the S1-16 prune list changes) | UNVERIFIED | deferred: exploratory rg only and S1-02 re-measures by AST under a pre-registered bathos sidecar (the 260827 spec section 0.3 makes the same claim for grid and ligand) |
| A6 | The campaign manifest row hash depends only on semantic values, not on spec class structure | If false, goldens must also pin the hash input and the wire format cannot change | VERIFIED | read: src/aminx/host/campaign.py:100-117 |
| A7 | `run_specification_from_json_dict` ignores unknown keys, while the campaign worker path rejects them; both behaviours must survive | none | VERIFIED | read: src/aminx/run/spec_json.py:231; read: src/aminx/host/campaign.py:1253-1259 |
| A8 | `PottsTRWRunSpec` is byte-identical across aminx and mistypotts for the whole file | If false, the conformance test in S1-17 must compare behaviour not bytes and the owner choice (Q4) needs a diff review | VERIFIED | read: src/aminx/potts/_trw_spec.py:1-77; read: ../../../../mistypotts/src/mistypotts/potts_trw_spec.py:1-77 |
| A9 | Runtime `isinstance` dispatch on the spec classes exists, so they must stay classes | none | VERIFIED | read: src/aminx/run/_exports.py:34-37 |
| A10 | Equinox 0.13.8 (installed) supports a custom `__init__` on a Module subclass, runs `__check_init__` for every class in the MRO after init, and defines `__eq__` as `tree_equal` | If the lower bound `equinox>=0.13.2` differs, the supported-floor test in S1-03 pins the real minimum | VERIFIED | read: .venv/lib/python3.14/site-packages/equinox/_module/_module.py:303-332; read: .venv/lib/python3.14/site-packages/equinox/_module/_module.py:426-434; read: .venv/lib/python3.14/site-packages/equinox/_module/_module.py:558-559 |
| A11 | Generated read-only compat properties plus a custom flat `__init__` on a `RunSpec` subclass do not break `tree_flatten`/`unflatten`, `eqx.tree_at`, `eqx.partition`, pickling or `copy` | Fall back to design B (facade survives as the one model; RunSpec becomes a derived xtrax adapter) and raise to the user | UNVERIFIED | deferred: spike-run harness absent and heavy commands forbidden so this is the gate of S1-03 which must pass before S1-04 or any later item |
| A12 | No `RunSpec`/spec instance is passed as an argument into a jitted function, so host-only static fields (TextIO, FoldCompDatabase, callables) need not be hashable | Host-only fields move into a separate non-pytree holder and `eqx.partition` is used at the jit seam | UNVERIFIED | deferred: needs an AST pass over every jit/filter_jit signature, produced by S1-02 (tracked script); S1-04 now depends on both S1-02 and S1-03 so ratification cannot precede the result |
| A13 | `xtrax.run.RunSpec` and `derive_sink_spec` are unchanged from v0.4.0a10 through v0.4.0a11 and origin/main, so S1 is independent of PR #174 | If a later xtrax changes the base fields, S1-06 must re-align the inherited-field mapping | VERIFIED | read: ../../../../xtrax/src/xtrax/run/spec.py:13-22; read: ../../../../xtrax/src/xtrax/run/sink.py:61-96 |
| A14 | `mpnn_ext` and `asr` consume aminx as a uv workspace member that is a vendored submodule pinned to a SHA (break at SHA bump, not at upstream shim removal); `tev_design` pins wheel 0.1.0a24 (re-pin project). Corrected in round 1 (C7): the earlier "editable, breaks when shims are removed" model was wrong | If a consumer tracks aminx HEAD rather than a SHA, its migration item must land before S1-07 instead of before S1-20 | VERIFIED | read: ../mpnn_ext/pyproject.toml:12-15; read: ../asr/pyproject.toml:94; read: ../tev_design/pyproject.toml:66; changed: consumer-migration items S1-26..S1-29 and canary rehearsal S1-21 added; risk row corrected |
| A15 | The browser runspec vocabulary is letter-keyed and structure-relative, differs from the Python facade, and has its Python twin only in a script | If the twin existed in library code, S4-20's lowering work (which now owns it) would reduce to a move | VERIFIED | read: browser/aminx-sampler/runspec_core.mjs:249-314; read: scripts/browser_validation/p07_knobs_gate.py:297-320 |
| A16 | Portable v2 JSON has no production consumer besides the `aminx spec portable-roundtrip` command | If a hub or browser consumer exists, v2 cannot be frozen and must be versioned with it | VERIFIED | read: src/aminx/cli.py:1623-1638; read: src/aminx/run/__init__.py:10-11 |
| A17 | EBM code does not depend on the spec system (so S1 and S2 share no files) | If it did, S1 and S2 would need an ordering edge | VERIFIED | read: src/aminx/ebm/langevin_schedule.py:353 |
| A18 | `or`-coalescing at construction maps falsy-but-valid values (random_seed 0, multi_state_temperature 0.0, num_samples 0) to defaults | none (behaviour is preserved either way; only the Q3 framing changes) | VERIFIED | read: src/aminx/run/spec.py:351-352; read: src/aminx/run/spec.py:367; read: src/aminx/host/runner.py:743 |
| A19 | No production code mutates a spec attribute after construction, so a frozen `RunSpec` is safe (D2 defines setattr as a pointed error regardless; this row decides only which callers need `.replace()` rewrites) | If a writer exists, `.replace()` is required at that site; scripts/benchmarks writers are listed in S1-13 | UNVERIFIED | deferred: exploratory rg found none in src but scripts/benchmarks write `spec.temperature` and `spec.run_spec` (bench_aminx_jax.py:169, bench_dedup_hetero.py:337); the S1-02 AST pass settles src, tests and scripts and gates S1-06 through S1-05 |
| A20 | `dataclasses.replace(spec, ...)` is used on specs at exactly two production sites | If more exist, S1-07 must keep a `dataclasses.replace`-compatible shim | VERIFIED | read: src/aminx/host/campaign.py:989; read: src/aminx/sampling/multistate_poe.py:650 |
| A21 | Flat `TrainingSpecification.precision` mirrors `run_spec.precision.compute` and the trainer reads only the latter | If other readers use the flat value, the `precision` name collision needs a read-through | VERIFIED | read: src/aminx/training/specs.py:98-99; read: src/aminx/training/trainer.py:53-59 |
| A22 | `SamplingSpecification` has 88 init fields | none (only a count; the registry is built from reflection, not from this number) | UNVERIFIED | deferred: documented in the spec_partition.py docstring and consistent with an exploratory AST count but a citable count is produced by S1-02 under a sidecar |
| A23 | mistypotts can be added as a dev-only dependency of aminx (needs `proxide>=0.1.0a3`, `../mistypotts/pyproject.toml:15`; resolvable under `uv lock --check`) | none: S1-17 defaults to a pinned sha256 comparison with no import; the dev dependency is an option adopted only if `uv lock --check` proves it | UNVERIFIED | deferred: resolvability needs the owner's distribution choice and a lock resolution the spec-only constraint forbids; the default path does not depend on it |
| A24 | `xtrax.run.RunSpec.from_spec` is an identity hook that aminx's `RunSpec` does not override | If aminx overrode it, `from_flat` would have to chain to it | VERIFIED | read: ../../../../xtrax/src/xtrax/run/spec.py:24-33; read: src/aminx/run/spec.py:138-150 |
| A25 | The prior WS-A decision to delete `GridLineageConfig`/`LigandConfig` was not executed | If it had been, the sub-configs would be re-introduced by S1-06 instead of kept | VERIFIED | read: src/aminx/run/spec.py:53-71 |
| A26 | The flat attribute holds the RAW constructed value while `run_spec.sampling.*` holds the COALESCED one (`random_seed` flat default 42, but an explicit 0 stays 0 flat and becomes 42 nested; `num_samples`, `multi_state_temperature` likewise), and `runner.py` coalesces again on its flat read | If the two views were already equal, D9's raw-storage-plus-accessors rule is merely defensive | VERIFIED | read: src/aminx/run/specs.py:258; read: src/aminx/run/spec.py:351-352; read: src/aminx/run/spec.py:367; read: src/aminx/host/runner.py:743 |
| A27 | The same flat name differs in type/default/semantics across facade classes: `temperature` float 1.0 (scoring) vs `Sequence[float] \| float` 0.1 tuple-wrapped (sampling); `ligand_conditioning` bool False (scoring) vs `bool \| None` None (sampling); `output_h5_path` is redeclared in four classes | A single default per name would have sufficed (it does not) | VERIFIED | read: src/aminx/run/specs.py:549; read: src/aminx/run/specs.py:560; read: src/aminx/run/specs.py:587; read: src/aminx/run/specs.py:636; read: src/aminx/run/specs.py:597 |
| A28 | `SamplingConfig.multi_state_strategy` and `MultistateConfig.combine_strategy` are both `eqx.field(static=True)`, so merging them keeps that field static; other `SamplingConfig` fields mix static and dynamic and `FieldDef` must record which | If the merged home were dynamic the jit cache key would change | VERIFIED | read: src/aminx/run/spec.py:50; read: src/aminx/run/spec.py:123-135 |
| A29 | `SamplingSpecification.__post_init__` mutates stored values (noise-bundle merge, `backbone_noise` back-population to a tuple, `use_electrostatics`/`use_vdw` None to bool, tuple-ization of `estat_noise`/`vdw_noise`, scalar `temperature` to tuple) and `dataclasses.replace` re-runs it on the normalised values | If it did not, `to_flat(from_flat(x)) == x` would hold trivially | VERIFIED | read: src/aminx/run/specs.py:420-443; read: src/aminx/run/specs.py:650-652 |
| A30 | S4's schemagen walks `dataclass`/`eqx.Module` fields and reads opt-in per-field metadata; S4 expects S1 to provide the surviving spec class plus a field-metadata hook and S4-20 owns the knob projection | If S4 changed to consume `run_spec_fields.json`, the hook becomes snapshot-only and no S1 item changes | VERIFIED | read: .praxia/docs/specs/261001_xtrax-model-contract.md:486-493; read: .praxia/docs/specs/261001_xtrax-model-contract.md:689-697 |
| A31 | S3 recommends rename-first (S3-07 before S1/S2 code items; `molxmpnn` has the API of `aminx 0.2.0a3` before S1 changes) and is the DAG root | If the user chooses S1-first, the S1-01/02/03/17 edges to S3-07 are dropped and S3-07 gains edges to S1-13 and S1-26..S1-28 | VERIFIED | read: .praxia/docs/specs/261001_molxmpnn-rename.md:55; read: .praxia/docs/specs/261001_molxmpnn-rename.md:352; read: .praxia/docs/specs/261001_molxmpnn-rename.md:408 |
| A32 | The old nested path set is closed and countable: the eight old sub-config classes declare 47 fields (io 5, resource 4, multistate 3, ligand 5, grid 6, precision 1, plan 1, sampling 22) and `RunSpec` adds 5 inherited plus 2 fusion names (54 paths); sub-configs are plain `eqx.Module` children with no parent reference, so a sub-config cannot serve `random_seed` (home: top-level `seed`) or `multi_state_*` (home: `multistate`) | If the set were larger or open-ended (dynamic attributes), the P_old reflection in S1-01 records the real set and the view/row count changes; the view-object decision is unaffected | VERIFIED | read: src/aminx/run/spec.py:26-135; read: src/aminx/run/spec.py:138-150 |
| A33 | `xtrax.run.derive_sink_spec` reads only `run_spec.run_id` from its argument (xtrax v0.4.0a11, read with `git show`), so a view that passes `run_id` through satisfies the three production call sites that pass `spec.run_spec` | If it read more, the three call sites (`host/runner.py:1551`, `host/streaming.py:91`, `sampling/multistate_poe.py:690`) would pass `as_run_spec(spec)` instead and the view needs no xtrax-facing surface | VERIFIED | read: ../../../../xtrax/src/xtrax/run/sink.py:100; read: src/aminx/host/streaming.py:91 |
| A34 | No code in src, tests or scripts treats a `spec.run_spec` or `build_run_spec(...)` value as a `RunSpec` instance (`isinstance`, pytree or `eqx` operation, `dataclasses`), apart from the type assertion `isinstance(rs.plan, PlannerTopology)` at `tests/run/test_run_spec.py:141` (satisfied because the view returns the stored `PlannerTopology`); `resolve_decode_mode` takes `run_spec: Any`; tests do alias (`rs = build_run_spec(...)`), which the view serves | If a site exists, the view must become a pytree or a `RunSpec` subclass (rejected as it recreates the two-homes problem) or that site is migrated before S1-06 | UNVERIFIED | read: src/aminx/host/plan.py:818; read: tests/run/test_run_spec.py:141; deferred: the exploratory grep for `isinstance`/tree operations on `run_spec` found none, but a negative grep is not a recorded result and the spike-run harness (`scripts/loop/adversarial_metrics.py`) is absent from this worktree, so S1-02 `view_incompatible_uses` records it and gates S1-06 |
| A35 | The old `build_run_spec` body reads the spec only through `getattr(spec, <name>, default)`, so, moved verbatim to `legacy_layout.py` and fed the S1-01 recorded flat facade attributes as a `SimpleNamespace`, it reproduces the pre-flip projection and can serve as the S1-07 differential oracle. It must NOT be fed the flipped spec: `seed` (`spec.py:381`) and `precision` (`spec.py:199`) become real unified fields and would be read with new meanings | The oracle falls back to the S1-01 recorded baseline values, which remain a sufficient gate (D3a) | VERIFIED (corrected convergence-check-3) | read: src/aminx/run/spec.py:197-203, 297-393 |

## Design

### D1. Which system survives (decision, with the alternative rejected)

**Chosen (design A): nested `RunSpec` is the single model; the flat keyword vocabulary survives as a codec
and as constructor signatures, not as a second stateful object.**

Why A over keeping the facade (design B):
1. `RunSpec` is the object the xtrax contract is written against (`derive_sink_spec`, `from_spec`,
   A13/A24), and the one S4 (schema generation) and S5/S6 (forms, editor) need as a typed source. B would
   leave that contract on a derived copy.
2. The measured failure mode is the two-object sync (A2, A3, A4 and the dead sub-configs): fields the
   facade has but RunSpec lacks are read through `getattr` fallbacks; fields RunSpec has are never fed.
   A removes the sync by construction; B only removes the guard flag.
3. The facade's real value is the flat vocabulary (ergonomics, campaign wire format, JSON files, ~96
   referencing files), not its class hierarchy. D2 keeps that value.

Cost honestly stated: today's `RunSpec` is *not* a complete model (it lacks inputs, model identity, noise
bundles, tie/state maps and all task-specific fields), so "RunSpec survives" means RunSpec grows to hold
every field (S1-06). Design B is the fallback if the S1-03 probes refute A11 or A12; it needs far fewer file
edits but keeps both vocabularies. This is Q1.

**Go/no-go decision rule (C13), applied at S1-04 by the user after S1-02 and S1-03 are both recorded.**
Refuted S1-03 probe for custom flat init with `__check_init__`, generated read-only properties, pytree ops
(flatten/unflatten, `tree_at`, `partition`, pickle, `copy`) or static-key/jit-cache-key equality, **or** an S1-02
finding that spec instances are jit arguments with unhashable host fields that cannot be moved to a holder
(A12) -> propose fallback B and ask. Refuted probes for `dataclasses.replace` failure mode, equality or hash
semantics -> adjust the D2 compat contract, not B. A refuted probe is an informative record, not a failed item.

### D2. Flat vocabulary becomes a permanent codec (this is what keeps the migration small)

- `RunSpecification`, `ScoringSpecification`, `SamplingSpecification`, `JacobianSpecification`,
  `InspectionSpecification` (and `TrainingSpecification`) **remain importable classes** that are thin
  subclasses of `RunSpec` with a flat-keyword `__init__` generated from the registry (A9, A10). Each
  `__init__` applies the same normalisation as today's `__post_init__` (noise-bundle merge, tied-position
  validation, `model_family` derivation, grid/campaign validation, deprecated-kwarg stripping via the
  existing `register_spec` / `pop_deprecated_spec_kwargs` behaviour, `specs.py:149-185`), then builds the
  nested sub-configs. Validation that is a property of a sub-config (tied-position/pass_mode,
  straight-through needing iterations, grid constraints) moves into that sub-config's `__check_init__`
  (A10) so direct nested construction is validated identically.
- `RunSpec.from_flat(task, **flat)` and `spec.to_flat()` are the public codec pair; `spec_json`, the
  campaign worker and the CLI call them. `spec.replace(**flat_overrides)` replaces `dataclasses.replace`
  and re-runs normalisation exactly as `replace` re-ran `__post_init__` (A20; call sites
  `campaign.py:989`, `multistate_poe.py:650`). `dataclasses.replace` on a spec raises a clear error naming
  `.replace`.
- Because constructors keep their flat signatures, **tests and scripts that construct real facade specs need no
  edit**; only code that reads flat attributes, uses `dataclasses.fields/replace`, or touches
  `run_spec`/`_sync_run_spec` is migrated (S1-10..S1-13). **Duck-typed specs are a separate case (C1)**: they are
  not covered by that sentence.
- **Duck-spec seam (S1-23).** Host and src readers receive specs through one function,
  `as_run_spec(obj)`: a `RunSpec` is returned as is; an object whose `.run_spec` is a `RunSpec` yields that; any
  other object (SimpleNamespace, hand-rolled class) is lifted through `from_flat` using only the attributes it
  has. Slices S1-10/S1-11 therefore read nested fields through the seam and their gates can pass while
  unmigrated duck fixtures still exist. Rule for the residue: `MagicMock` specs (whose `getattr` fabricates
  values) cannot be lifted, so **the slice PR whose gate breaks a fixture migrates that fixture** to the shared
  factory `tests/_support/spec_factory.py` (`make_spec(task, **flat)` returning a real facade, and
  `make_duck_spec(task, **flat)` returning the lifted-equivalent namespace for tests that deliberately pass
  partial objects). S1-12/S1-13 migrate the remainder. S1-02 classifies every duck construct as
  liftable / needs-factory / MagicMock-fabricating, per directory, and that table sizes the slices.
- `spec.run_spec` returns the **legacy nested view** (D3a), not `self`: a plain read-only object that closes over the
  whole spec and answers every old `run_spec.<sub>.<attr>` path (R2-1 convergence; the earlier "returns `self`"
  plus per-sub-config compat properties is withdrawn because a sub-config cannot read its parent or siblings and a
  stored field cannot share its name with a coalescing property). `build_run_spec(obj)` stays as a deprecated
  adapter until S1-20: pre-flip it is today's function; post-flip it returns `as_run_spec(obj).run_spec`, i.e. the
  same view. **`as_run_spec` is built before the flip (S1-23, `depends_on` S1-06
  and S1-31, and itself a dependency of S1-07)**: pre-flip it returns `build_run_spec(facade)` for today's
  facades, `from_flat` for duck objects, and refuses `MagicMock`; post-flip the same function returns a `RunSpec`
  as is. S1-07's shim is therefore a one-line delegation, and the S1-07 gate probes it with a real spec, a
  `SimpleNamespace` and a `_MinimalScoringSpec`-style class (R2-4).
- **Nested-path compat reads (R2-1) are specified in D3a** (mechanism, the complete old-path map, the completeness
  gate and the pre-flip coexistence rule). In one line: the view is built in S1-06/S1-31 (additive, new module,
  `aminx/run/spec.py` and `specs.py` untouched), value-checked over the whole corpus in S1-32, differentially
  checked against the moved-verbatim old builder in S1-07, and removed in S1-20. Migration of the readers to
  `resolved_*` accessors stays in S1-10..S1-13, so S1-07 needs no read migration in the same PR.
- **Compat contract for the unified class (C8).** (1) *Attribute writes*: `spec.x = v` and
  `spec.run_spec = v` on a unified spec raise `AttributeError` whose message names `.replace(**flat)`; the
  shim window does not tolerate writes (writes onto plain duck classes, e.g. `_BenchmarkSpec.run_spec = ...`,
  are unaffected because those are not `RunSpec`). Known writers are listed in Current state and migrated in
  S1-13. (2) *Positional init order*: the generated `__init__` keeps the exact parameter names, order and
  defaults of each current facade class (registry `order` column); S1-07 gate compares `inspect.signature` of
  every facade class against a signature fixture recorded by S1-01 from the unchanged code. (3) *Equality*:
  `==` becomes eqx `tree_equal` (A10), which differs from dataclass field-wise `==` for array-valued fields;
  the contract is `a == b` iff `a.to_flat() == b.to_flat()` under the codec's array comparison, enforced by a
  test over the ~13 `spec ==` style assertions (7 test files, exploratory) migrated in S1-12/13. (4) *Hash*:
  the facade today is a non-frozen dataclass with `eq=True`, hence unhashable (*unverified*; S1-03 probes it);
  the unified class is not required to be hashable and `hash(spec)` behaviour is pinned by a test, not left to
  eqx defaults. (5) *repr*: not contractual; goldens never use it. (6) *`dataclasses.replace`/`fields`*:
  pointed error, `RunSpec.field_names()` replaces `fields`.
- Flat attribute reads (`spec.inputs`, `spec.temperature`, ...) are served by **explicit generated
  read-only properties**, not `__getattr__` (avoids pytree/pickle recursion hazards). They emit
  `DeprecationWarning` once per (class, name). Name collisions between a flat name and a sub-config
  attribute are computed from the registry; the known one is flat `precision` (training) versus
  `precision.compute`, handled by the codemod, not by a property.

### D3. Nested layout (authoritative assignment is the registry produced by S1-05)

| Sub-config (attribute) | Holds (flat names) | Status |
|---|---|---|
| inherited | `random_seed` -> `seed` (stored RAW, see D9); `carry_specs`; `axes`/`boundaries`/`run_id` (no flat name) | `random_seed` single home is `seed` (removes `sampling.random_seed`) |
| `data: DataConfig` (new) | `inputs, topology, chain_id, model, altloc, foldcomp_database, conformational_states, max_length, truncation_strategy, use_preprocessed, preprocessed_index_path, split` | new |
| `model_ref: ModelRef` (new) | `model_weights, model_version, model_family, checkpoint_id, model_local_path, checkpoint_registry_path, ligand_mpnn_use_side_chain_context` | new; absorbs the `model_family` and `use_side_chain_context` fields of the old `ligand` (view paths in D3a) |
| `perturb: PerturbationConfig` (new) | `noise` bundles plus the six deprecated noise/mode fields, `use_electrostatics, use_vdw` | new; `noise` is both a public input kwarg and a normalised stored value (D9 rows N1-N4); estat/vdw bundles have no flat representation beyond the tuple fields, which the codec carries explicitly |
| `io: IOConfig` | `cache_path, output_dir, output_h5_path, overwrite_cache` | exists (`spec.py:26`); legacy `io.sink_kind` and `io.manifest_path` are not stored here (view-derived, D3a) |
| `resource: ResourceConfig` | `batch_size, samples_batch_size, max_buffer_size, host_resource_allocation_strategy, ram_budget_mb, max_workers, n_devices` | exists (`spec.py:36`), extended; legacy `sample_batch_size`/`structure_batch_size` are view-derived from `samples_batch_size`/`batch_size` (D3a) |
| `multistate: MultistateConfig` | `multi_state_strategy` (merges the duplicate home, A4), `multi_state_temperature, multi_state_sharpness, state_weights, tied_positions, pass_mode, tie_group_map, state_position_map, structure_mapping, use_rolling_state` | exists (`spec.py:45`), absorbs fields now on `SamplingConfig`; legacy `mode` and `n_states` are view-derived from `data.conformational_states`, not stored (D3a) |
| `sampling: SamplingConfig` | `num_samples, sampling_strategy, temperature, backbone_noise (derived view), bias, fixed_mask, fixed_positions, fixed_tokens, return_*, compute_pseudo_perplexity, decoding_order_fn, iterations, learning_rate, concrete_*, samples_chunk_size, noise_batch_size, temperature_batch_size, average_node_features, dedup_specs` | exists (`spec.py:97`) |
| `grid: GridLineageConfig` | `grid_mode, campaign_mode, job_id, chunk_id, sample_start, sample_count, allow_logits_in_campaign, logits_memory_budget_mb, weight_profile, fixed_group` | exists (`spec.py:63`); becomes live (supersedes the 260827 WS-A delete, A25) |
| `ligand: LigandConfig` | `ligand_conditioning, sidechain_conditioning, ligand_context_path` | exists; `model_family`/`use_side_chain_context` move to `model_ref` |
| `precision: PrecisionConfig` | `precision` (training) | exists; flat `precision` maps to `precision.compute` (A21) |
| `plan: PlannerTopology` | `use_unified_driver` | exists |
| fusion (top level) | `encoding_fusion, decoding_fusion` | exists; finally populated (A3) |
| `score` / `jacobian` / `inspect` / `train` (new, `None` unless the task) | task-specific fields of each facade subclass | new; `task` (static) derived from the subclass |

Rule enforced by the registry test: **every flat field has exactly one home per task**; no field is stored
twice. Moving a field between sub-configs must preserve its `static` classification (A28 shows the one merge in
this table, `multi_state_strategy`, is static on both sides); S1-05 gates the set of static field names against
the current dataclasses and S1-03 probes jit cache-key equality on a toy (C4).

### D3a. Legacy nested view: how old `run_spec.<sub>.<attr>` reads keep working until S1-20 (R2-1)

**Why not properties on the sub-configs.** (1) A stored field and a coalescing property cannot share a name:
D9 stores `sampling.num_samples` and `sampling.sampling_strategy` RAW, while the old nested read returned the
coalesced value. (2) A sub-config is a plain `eqx.Module` child with no parent reference (`spec.py:141-148`), so it
cannot compute `sampling.random_seed` (home: top-level `seed`) or `sampling.multi_state_strategy` /
`multi_state_temperature` (home: `multistate`); storing a copy there would violate "no field is stored twice"
(D3). Both are properties of the old layout, so the old layout is reproduced by an object that sees the whole spec.

**Mechanism (one, decided).**
1. After the flip, `spec.run_spec` returns `LegacyRunSpecView(spec)`: a plain immutable Python object
   (`__slots__`; not an `eqx.Module`, not a pytree; built on each access, nothing cached) defined in
   `src/aminx/run/legacy_view.py`. It holds a reference to the whole unified spec. Its attributes `io, resource,
   multistate, ligand, grid, precision, plan, sampling` are plain sub-views exposing exactly the old field names
   with the old values; it also passes through `seed, axes, carry_specs, boundaries, run_id, encoding_fusion,
   decoding_fusion` (`derive_sink_spec(spec.run_spec, ...)` reads only `.run_id`, A33). Exception that keeps old type
   assertions true: `plan` is returned as the stored `PlannerTopology` itself (its field set is unchanged).
2. **Name collision resolved by construction.** Unified sub-configs store RAW values under their natural names and
   carry no compat properties. Old names exist only on the view's sub-views. The coalescing logic is the D9
   `resolved_*` accessors (plus the old coercions `_optional_path`, `_as_float_tuple`, `_coerce_max_buffer_size`,
   `_infer_*`, referenced from the registry as `view_fn`); the view calls the same functions the migrated readers
   will call, so there is one implementation.
3. **The view exposes exactly the old path set P_old and nothing else**: the 47 field names of the eight old
   sub-config classes (io 5, resource 4, multistate 3, ligand 5, grid 6, precision 1, plan 1, sampling 22; read at
   `spec.py:26-135`, A32) plus 7 top-level names (5 inherited from xtrax, 2 fusion) = 54. Each path is a registry
   `compat_nested` row: `old_path`, `kind` (`same_home`, `coerced`, `coalesced`, `relocated`, `derived`),
   `new_home`, `view_fn`. The first read of a (view class, name) emits `DeprecationWarning` once.
4. Writes on the view raise `AttributeError` naming `.replace(**flat)`. The view is not a `RunSpec`: `isinstance`,
   pytree operations and `dataclasses` helpers on it are unsupported. S1-02 records every such use of a
   `spec.run_spec` or `build_run_spec` value as `view_incompatible_uses`; the list must be empty or migrated before
   S1-06 starts (A34). Known type assertion: `tests/run/test_run_spec.py:141` is satisfied by the `plan` exception.
5. `build_run_spec(obj)` post-flip returns the view, so alias reads such as `rs = build_run_spec(spec); rs.sampling.x`
   (`tests/run/test_run_spec_sampling_config.py`) and the portable codec's `run_spec_portable_to_dict(rs)` keep
   working until S1-08 regenerates the codec.

**Old-path map (seed for the S1-05 `compat_nested` rows; the completeness gate below checks it against reflection,
not against this table).** Home names are the D3 homes.

| Old path(s) on `run_spec` | kind | New home | View value |
|---|---|---|---|
| `io.output_h5_path`, `io.cache_path` | coerced | `io.*` | `_optional_path(x)` |
| `io.output_dir` | derived | `io.output_dir`, `io.output_h5_path`, `io.cache_path` | `_infer_output_dir` (explicit, else h5 parent, else cache) |
| `io.sink_kind` | derived | `io.output_h5_path` | `_infer_sink_kind` (`"zarr"` or `"none"`); no flat name, never stored |
| `io.manifest_path` | relocated | `data.preprocessed_index_path` | `_optional_path(x)` |
| `resource.n_devices` | derived | `resource.n_devices` | `_infer_n_devices` (falls back to `jax.local_device_count()`; compared on the recorded device only) |
| `resource.structure_batch_size` | relocated | `resource.batch_size` | `int(x)` |
| `resource.sample_batch_size` | derived | `resource.samples_batch_size`, `resource.batch_size` | samples batch if not `None`, else batch |
| `resource.max_buffer_size` | coerced | `resource.max_buffer_size` | `_coerce_max_buffer_size` |
| `multistate.combine_strategy` | coalesced, relocated | `multistate.multi_state_strategy` | `str(x)` of the old `getattr` default path (S1-01 pins) |
| `multistate.mode`, `multistate.n_states` | derived | `data.conformational_states` | `_infer_multistate_mode`, `_coerce_n_states`; never stored |
| `ligand.model_family`, `ligand.use_side_chain_context` | relocated | `model_ref.model_family`, `model_ref.ligand_mpnn_use_side_chain_context` | `str(x)`; passthrough |
| `ligand.ligand_conditioning`, `.sidechain_conditioning`, `.context_path` | coerced | `ligand.*` | `_optional_bool`, `bool`, `_optional_path` |
| `grid.grid_mode`, `.campaign_mode`, `.job_id`, `.chunk_id`, `.sample_start`, `.sample_count` | coerced | `grid.*` | `bool` for the two flags, passthrough otherwise |
| `precision.compute` | derived | `precision.compute` | `_run_spec_precision_compute` (non-training default `fp32`) |
| `plan.use_unified_driver` | same_home | `plan.use_unified_driver` | the stored `PlannerTopology` is returned as is |
| `sampling.num_samples`, `sampling.sampling_strategy` | coalesced | `sampling.*` (RAW) | `resolved_num_samples()`, `resolved_sampling_strategy()` |
| `sampling.random_seed` | coalesced, relocated | top-level `seed` (RAW) | `resolved_seed()` (0 becomes 42) |
| `sampling.multi_state_strategy`, `sampling.multi_state_temperature` | coalesced, relocated | `multistate.*` (RAW) | `resolved_multi_state_strategy()`, `resolved_multi_state_temperature()` |
| `sampling.multi_state_sharpness`, `.use_rolling_state`, `.tie_group_map`, `.state_position_map`, `.state_weights`, `.structure_mapping` | relocated | `multistate.*` | `_coerce_sharpness`, `bool`, passthrough |
| `sampling.temperature`, `sampling.backbone_noise` | coerced | `sampling.*` | `_as_float_tuple` |
| `sampling.return_logits`, `.compute_pseudo_perplexity`, `.return_decoding_orders`, `.return_logit_fingerprint` | coerced | `sampling.*` | `bool` |
| `sampling.bias`, `.fixed_mask`, `.fixed_positions`, `.fixed_tokens`, `.decoding_order_fn` | same_home | `sampling.*` | passthrough |
| `seed` | relocated, **expected change (EC)** | raw `random_seed` | the stored raw value; old builder always produced 0 (D9, A2) |
| `encoding_fusion`, `decoding_fusion` | same_home, **expected change (EC)** | fusion (top level) | passthrough of the stored flat value; old builder never populated them, so the old value was always `None` (A3) |
| `axes`, `carry_specs`, `boundaries`, `run_id` | same_home | inherited | passthrough |

**Completeness gate (one test file, `tests/run/test_legacy_view_paths.py`, re-run by every item that touches it).**
- *Path-set equality, both directions.* The `compat_nested` `old_path` set equals P_old, which S1-01 records as
  `tests/golden/spec_system/legacy_nested_paths.json` by reflection over the fields of the unchanged old classes
  (`spec.py` at baseline; expected 54 entries), and equals the attribute set the view exposes (reflection over the
  view and sub-view slots). Negative control: deleting one `compat_nested` row fails the test.
- *Read-site coverage.* Every `<expr>.run_spec.<sub>.<attr>` and aliased `<alias>.<sub>.<attr>` read found by S1-02
  across `src`, `tests`, `scripts` and `examples` (aliases traced one assignment deep, including
  `rs = build_run_spec(...)`; string-form `getattr(<sub>, "<name>")` listed separately) is a member of P_old and has a
  row. Exploratory count from this tree: 28 distinct paths in `src` and 4 in `tests`, plus alias reads in
  `tests/run/test_run_spec_sampling_config.py` and `tests/run/test_run_spec.py`; S1-02 re-measures.
- *Expected-change list (EC), closed at three paths: `seed`, `encoding_fusion`, `decoding_fusion`.* These are the
  only paths where the design intentionally differs from the old builder (D9 for `seed`, D3/A3 for the two fusion
  fields; the old builder produced `seed = 0` and `None` fusion for every facade-built spec). S1-01 records, per
  corpus case and for each EC path, **two** values in `legacy_nested_paths.json`: `baseline` (read through the
  unchanged `build_run_spec(facade)`) and `expected_post_flip`. `expected_post_flip` is derived from the documented
  rule applied to the unchanged facade's flat attributes, never from running new code: `seed` = the raw flat
  `random_seed` as constructed (an explicit 0 stays 0, not 42), `encoding_fusion` / `decoding_fusion` = the flat
  attribute of the same name after the unchanged facade's normalisation (canonical encoding for callables: type
  qualified name plus the leaf bytes of the module). The corpus must contain, per task class that has a
  `random_seed` flat field, at least one case with a non-default `random_seed`, the `random_seed=0` case, and at
  least one case with non-`None` `encoding_fusion` and one with non-`None` `decoding_fusion`, so that
  `expected_post_flip != baseline` somewhere for each EC path (otherwise the exception would be untested). Any change
  in a path outside EC is a failure and nothing may be added to EC without editing this spec. S1-02 additionally
  confirms that no reader in `src`, `tests`, `scripts` or `examples` reads `run_spec.seed`,
  `run_spec.encoding_fusion` or `run_spec.decoding_fusion` and depends on the old value (expected none: A2, A3; the
  host reads the flat attributes); a reader found is migrated before S1-06.
- *Value equality over the corpus (EC-aware).* Over every S1-01 corpus case (all five task classes), each of the 51
  non-EC paths resolves to its S1-01 `baseline` value by exact equality (canonical encoding; arrays by bytes;
  `n_devices` on the recorded device) and warns once on first read; each of the 3 EC paths resolves to the recorded
  `expected_post_flip` value by exact equality. Includes the quirk cases: `random_seed=0` (`seed` is 0 and
  `sampling.random_seed` is 42, both asserted), `num_samples=0`, `multi_state_temperature=0.0`,
  `multi_state_sharpness=None`. Negative controls: perturbing one recorded `baseline` or one recorded
  `expected_post_flip` value fails the test, and a view that returns the old value (`seed` 0, fusion `None`) on a
  case where `expected_post_flip` differs fails.
- *Differential oracle at the flip (S1-07).* The old `build_run_spec` and old sub-config classes are moved
  verbatim to `src/aminx/run/legacy_layout.py`. The oracle input is **not** the flipped spec: it is
  `flat_case = SimpleNamespace(**S1-01 recorded flat facade attributes for the case)`, i.e. exactly the flat names
  and values the old builder saw before the flip. (Feeding it the flipped spec is unpassable: after the flip `seed`
  is a real stored field, so `getattr(spec, "seed", 0)` at `spec.py:381` returns the raw seed instead of 0, and
  `precision` is the `PrecisionConfig` sub-config, so `getattr(spec, "precision", "fp32")` at `spec.py:199` yields
  `"fp32"` instead of a training spec's `"bf16"`.) The test asserts `legacy_layout.build_run_spec(flat_case)` equals
  `flipped_spec.run_spec` on the 51 non-EC paths for the whole corpus. On the 3 EC paths the two are required to
  differ exactly as documented: the moved-verbatim builder's value equals the S1-01 `baseline` (proving the old code
  was moved unchanged) and the view's value equals `expected_post_flip`; any other discrepancy on any path fails.
  The flat read properties are exercised separately by the S1-07 compat contract tests, not by this oracle.

**Coexistence before the flip.** S1-05 adds `fields.py`; S1-06, S1-31 and S1-32 add the new model as a
parallel hierarchy in new modules `src/aminx/run/nested.py` (`RunSpec`, `DataConfig`, `ModelRef`,
`PerturbationConfig`, the extended sub-configs, `from_flat`/`to_flat`) and `src/aminx/run/legacy_view.py`. The
existing `aminx.run.spec` and `aminx.run.specs` are **byte-unchanged** until S1-07 (gate: empty `git diff` over both
files at S1-06, S1-31 and S1-32), `build_run_spec` keeps constructing `SamplingConfig(random_seed=...)` of the old
layout, and the shadow test (S1-32) imports both hierarchies by module path. S1-07 then (i) moves the old sub-config
classes, old `RunSpec` and old `build_run_spec` verbatim into `src/aminx/run/legacy_layout.py`, (ii) turns `spec.py`
into re-exports of the `nested.py` names plus `topology_hash`, and (iii) retargets the imports of the files that
construct old-layout objects directly to `legacy_layout` (S1-02 `legacy_layout_constructors`; exploratory:
`run/run_spec_portable_json.py` and construct sites in `tests/run/test_run_spec_portable_json.py` and
`tests/cli/test_inputs_integration.py`). S1-08 regenerates the portable codec on the unified model and stops
importing `legacy_layout`; S1-20 deletes `legacy_layout.py` and `legacy_view.py`.

### D4. The field registry (the one source of truth for codecs)

`src/aminx/run/fields.py` defines a tuple of `FieldDef`, **one row per `(task, flat_name)`** (C4: the same flat
name has different type, default and semantics per class, A27): `FieldDef(task, flat_name, order, home_path,
type_ann, default, static, normalise, denormalise, kind, wire, portable, browser_key, reason)`:
- `order` reproduces each facade's positional init order (D2 compat contract); `type_ann` and `default` are the
  per-task values (e.g. `temperature`: `float`=1.0 for scoring, `Sequence[float] | float`=0.1 for sampling;
  `ligand_conditioning`: `bool`=False vs `bool | None`=None where `False` means ablate).
- `static: bool` records whether the nested home is `eqx.field(static=True)` (A28); the generated model must
  match, otherwise jit cache keys and hashability change silently.
- `normalise`/`denormalise` name the pair that converts flat value to stored value and back (scalar-to-tuple,
  noise-bundle merge, tied-pair coercion); both are pure functions listed in one module so D9's quirks are
  data, not scattered code.
- **Source of truth and field-metadata hook (C3).** The registry is the single source. Each nested leaf field
  of the unified model is declared with `eqx.field(metadata={"aminx_field": "<flat_name>"})`-style metadata
  generated from, and tested against, the registry row (portable/browser_key/wire/reason), so S4's schemagen,
  which walks `eqx.Module` fields and reads opt-in metadata (A30), sees the same facts as the JSON snapshot.
  **Metadata shape (C10 coherence).** Leaf metadata is `{"xtrax": {expose, browser_key, ...}}`, the documented
  S4 FieldMeta shape (S4 section 4.7), built by a local pure function in `run/fields.py` that returns a plain
  dict (no `xtrax_contract` import before S4-35). Registry `portable` maps to `expose`; `browser_key` is copied;
  any aminx-specific key (`flat_name`, `wire`, `reason`) lives beside `xtrax`, not inside it. Each field keeps
  its own `static` classification (A28): the metadata is passed as `eqx.field(static=<registry static>,
  metadata=...)`, never forcing static. **Round 2 (CH2-02):** the authoritative key list is the published
  `field_meta.v1.json` (twelve keys, S4-07), not S4 section 4.7 prose, so S1-06 `depends_on` S4-07 (S1-31
  inherits it); the S1-06/S1-31 gates assert the dict's keys are a subset of that file. S4-32 still validates
  the dict against the snapshot's portable rows downstream.
  `run_spec_fields.json` is a derived snapshot for non-Python consumers only; there is no second authored source.
- `wire` in `{serialized, campaign_owned, excluded, derived}` replaces the hand lists in
  `spec_partition.py` (the `EXCLUDED_WITH_REASON` reason text moves into `reason`, with the existing
  "must name a mechanism" check).
- `kind` drives `_to_json_value` / `_coerce_field_value` (path, ndarray, float tuple, tied-pair list,
  callable, handle), replacing the whitelists at `spec_json.py:164-217`.
- `portable` and `browser_key` flag the subset a browser/hub request may carry; this is the contract S4 turns
  into JSON Schema and JS validators (Provides P2). S1 does not generate either, and does not build the browser
  knob document or lowering (S4-20 owns them, D10).
- Codecs `spec_json`, `campaign_sampling_spec_payload`, `run_spec_portable_{to,from}_dict`,
  `scripts/audit/knob_matrix.py`, the flat read properties and the legacy nested view (D3a `compat_nested` rows) are
  generated from it. The import-time
  `assert_partition_is_exhaustive()` keeps its semantics (every field classified, stale entries rejected,
  vague reasons rejected) but is computed from the registry plus reflection over `RunSpec` fields, so a new
  field cannot exist without a row. A committed snapshot `src/aminx/run/run_spec_fields.json`
  (deterministic, tested for freshness) is what S4/S5 consume.

### D5. Wire formats (frozen, versioned, two readers)

- **flat-v1** (`{_spec_class: ..., <flat keys>}`): unchanged bytes on write; same two reader policies
  (lenient `spec_json`, strict campaign constructor, A7). `_spec_class` strings and
  `MANIFEST_ROW_SCHEMA_VERSION` stay; the row-hash input stays (A6). Goldens (S1-01) pin all of this.
- **portable-v2** (`io, multistate, resource, precision`): frozen as is, generated from the registry,
  documented as deprecated-pending-S4 because it has no consumer (A16, Q8). Its two RS-8 guards are kept;
  the missing import-direction guard noted in the 260827 spec (section 0.4) is implemented here because the
  portable codec is being re-generated anyway.
- No new format is introduced by S1. The schema-driven nested/portable format for the hub is S4's.

### D6. Migration order (strangler, byte-identical at every step)

1. Pre-register and record goldens against the *unchanged* code (S1-01), inventory (S1-02), probes (S1-03),
   ADR (S1-04).
2. Registry (S1-05) with a both-directions conformance test against the facade as it stands.
3. Additive nested model (S1-06, S1-31) plus codec with a shadow-equivalence test against the facade (S1-32), in new
   modules `run/nested.py` and `run/legacy_view.py` beside the byte-unchanged `run/spec.py` and `run/specs.py`
   (D3a, coexistence). Preceded by the S1-02 per-read-site raw-versus-coalesced table (D9) and the nested-path
   inventory, which are hard preconditions.
4. The flip (S1-07), one PR: old layout classes and old `build_run_spec` move verbatim to `run/legacy_layout.py`;
   `spec.py` re-exports the unified model; facade classes become `RunSpec` subclasses; `spec.run_spec` returns the
   legacy view; sync machinery becomes shims. Half-flipped states are not mergeable, hence size L. Its gate runs
   the reader-test selection recorded by S1-02, not a keyword selector.
5. Codecs on the registry (S1-08), CLI constructor swap (S1-09), duck-spec seam and factory (S1-23), codemod
   tool with its own fixtures (S1-24).
6. Read migration in four slices (S1-10..S1-13) driven by the S1-24 codemod, plus docs (S1-14). Each slice is
   gated on goldens plus "the slice's own paths emit zero DeprecationWarning, measured by the fixed script
   `scripts/spec/deprecation_paths.py`" (no global-count clause: parallel slices cannot each claim a strict
   decrease of a shared count).
7. Behavioural equivalence on small real runs (S1-15), prune (S1-16, after all migrations), release cut that
   carries the deprecation window (S1-25), consumer migrations (S1-26..S1-29), shim-removal rehearsal (S1-21),
   then removal of shims (S1-20).

Slices are grouped by directory to stay PR-sized: A `src/aminx/host/` (13 files); B the other 13 src files
(`io`, `sampling`, `tiling`, `training`, `utils`, `run/_exports`, `__init__`); C `tests/host` (37 files; mostly
duck-spec and idiom edits only; the seam S1-23 keeps most unchanged); D remaining tests (run, cli, audit, sampling, scoring, utils,
parity, integration, inference), `scripts/`, `examples/`.

### D7. Ordering relative to S2 and S3 (no double churn, acyclic)

**Round-1 change (C2, C14).** Round 0 recommended S1 before the S3 rename; S3 (the DAG root, A31) recommends the
opposite and says S1/S2 code items should depend on S3-07. Applying both is a cycle. This spec now **adopts
rename-first as the default edge** because (a) S3 is the root and its rename is output-equivalent mechanical work
done once against the `aminx 0.2.0a3` API, so S1 then lands entirely in the final names with no rebase of
in-flight slices across a rename; (b) the S3 compat shim (`aminx` re-export, S3-10) keeps vendored consumers
importable while S1 proceeds. The choice is still the user's (Q10) and is encoded only through `depends_on`.
**Q10, S3 Q4 and S2 Q7 are the same question**, answered once by the user at S3-01; S1 and S3 already encode
rename-first, and S2 must align to the answer:
- S1-01, S1-02, S1-03 and S1-17 `depends_on` **S3-07** (the code rename). Everything else follows transitively.
- **Pinned-tag protection (round 2, CH2-03).** S3-13 cuts its tag from the merge of the last of S3-07..S3-12 and
  gates that the diff from the S3-07 merge lists only S3-owned paths. Any S1 commit merged inside that window
  would break the gate, so S1-05 (the first item to add files under `src/.../run`) and S1-17 (potts file) also
  `depends_on` **S3-13**; S1-06, S1-31, S1-32, S1-07 and S1-18 follow transitively. S1-01, S1-02 and S1-03 are
  additive-only (tests/golden, scripts, records) and need no S3-13 edge. S1-17's optional mistypotts dev
  dependency (a uv.lock change) lands on main and does not touch S2's pinned F' checkout, so no S2 edge is
  needed (CH2-04).
- The deprecation window is a real item: **S1-25 (release cut)** `depends_on` S1-12, S1-13, S1-14, S1-15, S3-13
  and **S5-52** (CH2-01: the browser-assets tag is cut first, so the two releases are comparable in the union
  DAG as S3-28 requires; S1-25 is a tail item, so S5-52 and the first hub deployment never wait on the S1 flip
  chain), publishing an alpha that carries the flip plus `DeprecationWarning`s. **S1-20 depends on
  S1-25** (and on the consumer migrations), so shim removal cannot merge before one alpha has shipped. S1-20 then
  ships in a later alpha, not the S3 rename release; downstream takes the S3 break (name) and the S1 break
  (reads) separately but each is announced and windowed.
- If the user chooses S1-first instead: drop the S3-07 edge from S1-01/02/03/17, and S3-07 gains
  `depends_on = ["S1-13", "S1-26", "S1-27", "S1-28"]`; S1-25 then depends on the S3 release item instead of S3-13
  (and keeps S5-52).
- S2 (EBM extraction): no spec-field dependency (A17), but S2's A0 freeze window (`git diff F' <sha>` empty over
  `src/aminx/ebm`, `utils`, `io`, `parity`, `scripts/ebm` at S2-32/33/34 and S2-06) must not be disturbed by S1
  edits to those paths. S1-02 therefore records whether any file in S2's consumed-file list reads spec fields
  (expected none, A17); S1-11 and S1-24 **exclude that list** from rewrites. If the S1-02 answer is non-empty,
  S1-11 gains `depends_on` S2-06 (so the clone at G precedes the rewrite). Otherwise no S1-to-S2 edge exists.
  Ordering of S3-07 against S2-03..S2-06/S2-32..34 is S2/S3's to encode, not S1's.
- Consumer inventory is reconciled with S3 A1: `mpnn_ext`, `asr`, `hautespout` (vendored submodules) and
  `tev_design` (wheel). File counts differ because S1 counts files naming a `*Specification` class (96 in aminx)
  and S3 counts files mentioning the package name (S3 reports 812); the measures are different, not in conflict.
- Identifiers frozen by S1 and **not** touched by the rename: `_spec_class` values, manifest schema version
  strings, flat field names. Class names do not contain the package name.

### D8. Potts specs

Keep `PottsRunSpec` and `PottsTRWRunSpec` out of `RunSpec`. Reasons: `PottsTRWRunSpec` is a hashable static
config used as a compile-cache key on modules (`potts/model.py:98`, `potts/_trw.py:168`) and folding it into an
`eqx.Module` sub-config changes hashing and equality semantics for no consumer; `PottsRunSpec` does not
use the host planner. To satisfy "one system" at the boundary, Potts conforms to a tiny `SpecFamily`
protocol (`_spec_class` discriminator, `to_json_dict`/`from_json_dict`, registry rows with `portable` flags)
so S4's schema generation treats it uniformly. Revisit the fold when a second Potts consumer or S4 needs it (Q5).

**TRW single owner (alphex D4 pattern, `../alphex/.praxia/docs/specs/260814_alphabet-contract.md:283-300`):**
`mistypotts` owns `PottsTRWRunSpec` (it is the TRW numerics home and has the tests); aminx keeps a vendored copy
with a header recording the source file's sha256. **All deliverables of the conformance work live in aminx**
(S1-17): the vendored header and a conformance test that compares the vendored file's sha256 with a pinned
value of the owner's file, with **no import of mistypotts** (default). The negative control is a pytest fixture
that copies the vendored file to a temp path, mutates it, and asserts the check fails. A dev-only dependency on
mistypotts (which would also force aminx's universal lock to resolve `proxide>=0.1.0a3` and any index/source pin,
A23) is an **option adopted only if `uv lock --check` proves it resolvable**; then the test additionally asserts
equal dataclass fields/defaults/`to_json_dict` for all three `default_*` constructors. mistypotts gets at most a
docs/ownership note (S1-30). No runtime edge is added in either direction. Alternative rejected: a new
stdlib-only leaf package for 77 lines (Q4).

### D9. Storage semantics and quirk register (preserved in S1, surfaced as Q3)

**Decision (C5): the single home stores the RAW constructed value; coalescing happens at named read accessors.**
Reason (A26): today the flat attribute is raw (`random_seed=0` stays 0, flat-v1 bytes contain 0) while
`run_spec.sampling.*` is coalesced (0 becomes 42). One home cannot be both, so flat-v1 bytes (D5) fix the stored
value as raw, and every site that read the coalesced nested value, or coalesced inline (`host/runner.py:743,1037,
1194`), reads through accessors `resolved_seed()`, `resolved_num_samples()`, `resolved_sampling_strategy()`,
`resolved_multi_state_strategy()`, `resolved_multi_state_temperature()`. The codemod (S1-24) rewrites
`run_spec.sampling.<f>` for these five fields to the accessor, and rewrites inline `x or default` on flat reads to
the accessor. The **per-read-site table** (site, view read today: raw or coalesced, accessor required) is an
output of S1-02 and a precondition of S1-06 (S1-06 `depends_on` S1-02 explicitly). The **expected-change list
(EC) of the legacy view is exactly three paths** (D3a): `run_spec.seed` (0 today, now raw `random_seed`; nothing in
src reads it, A2) and `run_spec.encoding_fusion` / `run_spec.decoding_fusion` (always `None` today, A3, now the
stored flat value). All three are visible changes listed in the S1-07 changelog entry, and the D3a value gates and
the S1-32 / S1-07 gates compare them with the S1-01 `expected_post_flip` values, not with the old baseline;
`resolved_seed()` returns the coalesced 42-for-0 value that sampling uses.

Behaviours the codec must reproduce exactly, each pinned by an S1-01 golden:
- Q-rows: `num_samples or 1` (`spec.py:351`); `random_seed or 42` (`spec.py:352`); `sampling_strategy or
  "temperature"`, `multi_state_strategy or "arithmetic_mean"`, `multi_state_temperature or 1.0`
  (`spec.py:363-367`; 0.0 becomes 1.0); `carry_specs` flat default `None` fed into a list-typed field
  (`spec.py:383`); `model_family` derived once at construction so `replace(checkpoint_id=<ligand>)` does not
  re-derive it (`specs.py:485-486`).
- N-rows (normalisation performed by today's `__post_init__`, A29): **N1** noise-bundle merge is idempotent
  (merging the already-merged `noise` plus the six deprecated fields again yields the same bundles); **N2**
  `backbone_noise` is back-populated from bundles and scalar-wrapped to a tuple; **N3** `use_electrostatics` and
  `use_vdw` go from `None` to `bool` derived from bundles; **N4** `estat_noise`/`vdw_noise` and a scalar
  sampling `temperature` are tuple-ized; **N5** `replace` after normalisation feeds the normalised values back
  through normalisation (`campaign.py:989`, `multistate_poe.py:650` depend on this), including `noise` bundles
  combined with a newly supplied `backbone_noise`.
- **Round-trip contract (weakened from round 0):** `to_flat(from_flat(x))` equals the flat view of the *facade
  built from x by the unchanged code* (so normalisation is included), and `from_flat(to_flat(s)) == s` is a
  fixpoint. `to_flat(from_flat(x)) == x` is **not** claimed for un-normalised `x`.

### D10. Browser / S4 coordination (S1 provides, S4 owns the knob document)

**Resolution of C3.** One source of truth: the S1 registry (D4), exposed to S4 as nested-field metadata plus a
derived JSON snapshot. One owner of the portable MPNN knob document, its defaults/validators, and the pure
lowering from letter-keyed structure-relative constraints (`bias`, `fixed_mask`, `fixed_tokens`,
`tie_group_map`, the `chains_to_design` vs `chain_id` mismatch): **S4-20**, which `depends_on` S1-05 and S1-07.
The round-0 `DesignConstraints` sub-config and library lowering (old S1-19) is **removed from S1**; building
it here would duplicate S4-20 and reintroduce the third spec system D4 forbids. S1 keeps only the **key-set
conformance test** (S1-18): keys read from `spec.` in `runspec_core.mjs` must equal registry rows with a
`browser_key`, with a negative control; it fails on drift until S4-20's generated schema replaces the hand
copy, at which point S4 deletes it. Q6 records the remaining user call (confirm S4-20 as owner).

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Behaviour drift during the flip (silent knob loss was the historical failure, `spec_partition.py:1-25`) | Goldens recorded on unchanged code before any edit (S1-01); registry exhaustiveness at import; shadow-equivalence test before the flip; flat-v1 bytes and row hashes pinned |
| Equinox constraints (A11/A12): custom init, generated properties, frozen instances, `tree_equal` equality replacing dataclass equality | S1-03 probes gate everything; fallback design B named; goldens compare via `to_flat()` not `==`; arrays in `bias`/`fixed_mask` already live in `SamplingConfig` so no new array-equality surface |
| Campaign resume breakage (done markers keyed by row hash, `campaign.py:100-117`) | Hash input untouched (A6); golden manifest plan on a fixed config; `campaign_done_marker_v2` untouched |
| Downstream break: `mpnn_ext`/`asr`/`hautespout` break at their submodule SHA bump, `tev_design` (wheel 0.1.0a24) at a re-pin project (A14, corrected) | Flat constructors permanent; read shims kept through one published alpha (S1-25) with `DeprecationWarning`; consumer migration items S1-26..S1-29 precede shim removal; S1-21 rehearses removal on a throwaway aminx tree against consumer scratch checkouts with no writes to their tracked state; removal (S1-20) user-gated |
| Duck-typed spec fixtures break when host reads go nested (C1) | `as_run_spec` seam plus shared factory (S1-23); S1-02 classifies every duck construct; the slice whose gate breaks a fixture migrates it in the same PR |
| Two read views (raw flat, coalesced nested) collapse into one home (C5) | Raw storage plus named accessors (D9); per-read-site table from S1-02 gates S1-06; goldens cover each coalescing case through both views |
| FieldDef cannot express per-class defaults/static-ness (C4) | One row per `(task, flat_name)` with `static`, normaliser pair; S1-05 asserts generated defaults and static set equal the current dataclasses |
| Rename/spec ordering cycle with S3 (C2) | Single encoded edge S1-01/02/03/17 -> S3-07 (D7); flip path documented; Q10 |
| Knob document split across S1 and S4 (C3) | One source (registry), one owner (S4-20); old S1-19 removed |
| `dataclasses.replace`/`fields` idioms stop working on eqx modules (A20) | `.replace(**flat)` and `RunSpec.field_names()` provided; two production sites plus four introspection sites migrated in S1-08/S1-10; `dataclasses.replace` raises a pointed error |
| Double churn with S3 rename | D7: S1 follows S3-07; identifiers frozen; S1-25/S1-20 windowing |
| Large S1-07 PR | It is deliberately mechanical (class bodies replaced by generated inits) behind the S1-06/S1-31/S1-32 shadow test and S1-01 goldens; no read migration in the same PR, because every one of the 54 old nested paths is served by the legacy view (D3a, R2-1) and the gate runs the S1-02 reader-test selection, so a flip that breaks a reader fails in this PR, not in S1-10 or S1-12 |
| Legacy view diverges from the old nested layout, or something treats `spec.run_spec` as a `RunSpec` (pytree op, `isinstance`) | Three-way path-set equality, corpus value equality and a differential oracle against the moved-verbatim old builder (D3a completeness gate); S1-02 `view_incompatible_uses` must be empty before S1-06; the view is deleted with the other shims in S1-20 |
| Old and new hierarchies share class names during S1-06..S1-32 | Separate modules (`nested.py` versus untouched `spec.py`); empty `git diff` gate over `spec.py` and `specs.py`; shadow test imports by module path; old classes move verbatim to `legacy_layout.py` at the flip |
| Heavy test suites on this machine | Every gate runs on titanix (rsync worktree + ssh), narrow selectors only; nothing runs locally (local compute limits) |
| Prior spec conflict: the 260611 three-layer model and 260827 WS-A deletes | S1-04 ADR supersedes the relevant clauses explicitly; backlog items for WS-A A1/A2 must be closed as superseded |
| `random_seed` single-home move changes `run_spec.seed` from 0, and populating the fusion fields changes `run_spec.encoding_fusion` / `decoding_fusion` from `None` | Nothing reads them (A2, A3; S1-02 re-checks all of src, tests, scripts, examples); changelog line; S1-01 records `expected_post_flip` per case and the D3a gates assert it for exactly these three paths (EC), exact baseline equality on the other 51 |
| Rollback | Each slice is an independent PR; before S1-07 the change is additive (revert S1-06); after S1-07 revert that single PR (shims make S1-08..S1-14 individually revertable); S1-20 is the only irreversible-for-downstream step and is user-gated and preceded by a published alpha |

## Interfaces

### Provides

| ID | Contract | Detail | Consumers |
|---|---|---|---|
| P1 | `RunSpec` unified model, **stable surface** (the one surviving spec class, with field-metadata hook) | Nested eqx.Module extending `xtrax.run.RunSpec`; `from_flat`, `to_flat`, `replace`, `resolved_*` accessors, `as_run_spec` for real `RunSpec` inputs only; each leaf field carries registry metadata; facade subclasses as thin constructors. (`topology_hash` removed from the contract in round 2: S1 defines no such thing.) | S4 (S4-20), S5, S6 |
| P1b | Compat shims (**removed at S1-20, do not build on them**) | flat read properties, `.run_spec` alias (the legacy nested view, D3a), `legacy_layout.py`, `build_run_spec`, `as_run_spec` duck lifting | in-repo code and migrated consumers during the window only |
| P2 | `RunSpecFieldRegistry` (single source) + derived `run_spec_fields.json` snapshot | One row per (task, flat_name): home path, type, default, static, normaliser pair, kind, wire class, `portable`, `browser_key`, reason; deterministic, versioned (`registry_version`) | S4 (schemagen reads the same facts through field metadata; snapshot for JS), S5 (forms, catalog), S6 (parameter panels) |
| P3 | flat-v1 codec | `_spec_class` + flat keys; lenient reader and strict campaign constructor; manifest `sampling_spec` payload | campaign tooling, `mpnn_ext`, `tev_design` |
| P4 | Conformance kit | Registry-vs-codec exhaustiveness, JS key-set drift test, TRW sha256/field conformance, golden corpus | S4, S3 (rename regression), `mistypotts` |
| P5 | `PottsTRWRunSpec` single owner | `mistypotts` owns; aminx vendors with sha256 conformance | `mistypotts`, aminx potts |
| P6 | Consumer migration record | Migrated `mpnn_ext`, `asr`, `hautespout` submodule bumps and the `tev_design` re-pin handoff | S3 (consumer inventory), downstream owners |

Edges (the S1 TOML now carries its own foreign edges to S3; the edges below must be declared by the other specs):
- **S3 has no edge to S1** under rename-first; S1 carries S1-01/02/03/17 -> S3-07 and S1-25 -> S3-13 itself.
  If the user flips to S1-first (Q10), S3-07 gains `depends_on` S1-13, S1-26..S1-28.
- S4-20 (knob document, lowering, schema generation over the surviving model) `depends_on` S1-05 and S1-07; to delete
  the drift test it also `depends_on` S1-18.
- S5/S6 forms and parameter panels `depends_on` S1-05 for the registry snapshot.
- **Edges owed by other specs (assembly input, R2-8):** S4-20 -> S1-05, S1-07, S1-18 (S4's current `depends_on`
  is `[S4-02, S4-09, S4-19]` and lacks all three; S4 defers this to DAG assembly). S1-26/27/28 share the consumer
  branches with S3-14/15/17 and carry those edges themselves. The assembly checklist must verify both.
- **`repo` convention:** S1 items use `repo = "molxmpnn"` (every one runs after S3-07 renames the repo; consumer
  items name the consumer). `mistypotts` stays `mistypotts`.
- S2 has no edge to S1.

### Consumes

| Contract | Providing spec | What S1 needs |
|---|---|---|
| xtrax `RunSpec` base and `derive_sink_spec` stable (`seed, axes, carry_specs, boundaries, run_id`, `from_spec`) | S4 (xtrax model contract) | S4 must keep these fields and the identity `from_spec` backward compatible, or announce a change before S1-06; verified unchanged a10 -> main (A13) |
| `molxmpnn` namespace with the `aminx 0.2.0a3` API (S3-07), the `aminx` compat shim (S3-10) and the first published release (S3-13) | S3 (rename) | S1 code items run on the renamed tree (S3-07); consumers stay importable via the shim during S1; the S1-25 alpha release needs S3-13 (and follows S5-52) |
| FieldMeta key list (`field_meta.v1.json`, ParamProfile@1) | S4 (S4-07) | Authoritative keys for the registry metadata dict; S1-06 `depends_on` S4-07 and its gate asserts keys are a subset of the file (CH2-02) |
| Browser-assets release tag (S5-52) | S5 | S1-25 `depends_on` S5-52 so the flip-carrying alpha is the next unused alpha after it (CH2-01) |
| Knob document, lowering and schema generation over the surviving model (S4-20) | S4 | Replaces S1-18's drift test and owns what old S1-19 would have built; S4 reads S1's field-metadata hook (P1/P2) |
| EBM extraction | S2 | None (A17); stated explicitly so the DAG records the absence of an edge |

## Verification gates

Global rules for every item: tests run on **titanix** (rsync the worktree, ssh), narrow selectors only,
never a whole suite locally; every number or pass/fail that will be cited is produced by a tracked script with a
`.bth.toml` sidecar committed **before** the run, launched as `bth run --project-slug aminx -- uv run
--no-sync python3 scripts/<...>.py` (not `uv run bth`), and verified by its **record**
(`bth compact`, then `bth sql "SELECT id,status,outcome,exit_code,command FROM runs WHERE id LIKE '<id>%'"`,
outcome evaluated) not by exit code. No new derived-data directory is introduced; goldens are committed fixtures
under `tests/golden/spec_system/`. If a scratch or cache directory becomes necessary it resolves explicit
argument > documented env var (`none`/empty disables) > `[tool.<name>]` in `pyproject.toml` >
`${XDG_CONFIG_HOME:-~/.config}/<name>/config.toml` > off, with a `*_source()` reporter, never a hardcoded path
and never inside another project's data directory.

| Item | Gate (how we prove done) |
|---|---|
| S1-01 | Sidecar committed first. Outcomes: (1) goldens regenerated twice are byte-identical (determinism); (2) negative control: mutating one field of a probe spec changes flat JSON, row hash and projection (instrument can fail); (3) positive control: a corpus case per D9 row (Q, N1-N5) is present, including `replace()` on an already-normalised spec with `noise` bundles plus a new `backbone_noise`, and each coalescing case read through both the flat and the nested view; (4) a signature fixture of every facade class (`inspect.signature`) is recorded; (5) `legacy_nested_paths.json` (D3a): P_old by reflection over the fields of the unchanged old sub-config classes and `RunSpec` (expected 54 paths), and for every corpus case the baseline value of each path read through the unchanged `build_run_spec(facade)` (canonical encoding, arrays by bytes, `n_devices` on the recorded device); the RunSpec projection golden covers the 51 non-EC paths only (the three expected-change paths are carried solely by `legacy_nested_paths.json`, so byte-identity of the projection at S1-07 cannot contradict the documented changes); for the three expected-change paths (`seed`, `encoding_fusion`, `decoding_fusion`; D3a EC) each case also records `expected_post_flip`, derived from the documented rule applied to the unchanged facade's flat attributes (raw `random_seed`; flat fusion attributes), never from new code, with the corpus required to include non-default seed, `random_seed=0` and non-`None` fusion cases so expected differs from baseline somewhere per EC path; negative control: perturbing one baseline value or one expected value fails the value check. Pinned in the sidecar: device (CPU for the numeric cases unless a recorded titanix GPU is named), XLA flags, and the weights revision (LFS availability on titanix checked first); S1-07 and S1-15 compare on the same device only. Reuse of S3-05's golden fixtures for model-family outputs is allowed when its recorded SHA and device match, otherwise S1-01 records its own (overlap *unverified*; S1-01 reads S3-05's record first). Covers all five task classes, ligand, multi-state PoE, grid/campaign, training, deprecated kwargs, tied positions, portable v2, and small-L fixed-seed numeric outputs. Chunked/resumable: one process and one completion stamp per case, resume skips cases whose input and artifact sha256 match and records which were reused |
| S1-02 | AST script emits `spec_field_inventory.json` (per flat field and per task: defining class, flat reads, `run_spec` reads, wire class, name collisions, jit-argument check for A12, post-construction writes for A19, **per-read-site raw-versus-coalesced table**, **duck-construct classification per directory**, **every `<expr>.run_spec.<sub>.<attr>` and aliased `<alias>.<sub>.<attr>` nested read across src, tests, scripts and examples checked to be a member of the S1-01 P_old set, with string-form `getattr` reads listed (D3a), and any reader of `run_spec.seed` / `encoding_fusion` / `decoding_fusion` that depends on the old value (expected none)**, **`view_incompatible_uses`: every `isinstance`, pytree, `dataclasses` or `eqx` use of a `spec.run_spec` or `build_run_spec` value, which must be empty or migrated before S1-06**, **`legacy_layout_constructors`: every direct construction of an old-layout sub-config or `RunSpec`**, and **`reader_tests.json`: every test file that imports, transitively through the src import graph, a module holding a nested read or a `run_spec` parameter, with a recorded reason for any exclusion**). Positive control: a seeded dead field in a fixture is reported; negative control: a live field is not. Replaces the "27 dead" claim with a recorded number |
| S1-03 | Go/no-go record committed in which **every probe outcome is evaluated and recorded** (a refuted probe is a valid pass of the item): custom flat init plus `__check_init__`; generated read-only properties; flatten/unflatten, `eqx.tree_at`, `eqx.partition`, pickle, `copy`; static-field jit cache-key equality on a toy; `dataclasses.replace` failure mode; equality; hashability of the current facade and the unified class; on the installed equinox and the declared floor. Sidecar outcome: "record exists and each probe outcome evaluated". The decision rule in D1 maps refuted probes to fallback B; the user decides at S1-04 |
| S1-04 | ADR committed under `.praxia/docs/decisions/` naming the S1-02/S1-03 results it relied on; supersession notes added to the 260611 and 260827 specs; `docs(action="check")` clean; user ratified Q1 |
| S1-05 | `pytest tests/run/test_field_registry.py` on titanix: every facade field has exactly one row per task and every row names a real field (both directions); **generated flat default and type for every (class, field) equal the current dataclass's** (checked against the unchanged facade); **static-field set equals the current dataclasses'**; generated `__init__` signature equals the S1-01 signature fixture; snapshot freshness; negative control: deleting a row, and altering one default, each fail the test |
| S1-06 | (a: shared and sampling sub-configs) `DataConfig`, `ModelRef`, `PerturbationConfig` and sampling additions with real defaults, raw storage, `resolved_*` accessors, in the new module `run/nested.py`, plus `run/legacy_view.py` (the `LegacyRunSpecView` mechanism and the `compat_nested` rows for the shared and sampling paths, D3a); unit tests on titanix over hand-built nested specs (the codec arrives in S1-32), including `random_seed=0`, `num_samples=0`, `multi_state_temperature=0.0`: the view's attribute set equals the `compat_nested` rows and the S1-01 `legacy_nested_paths.json` entries for those sub-configs (both directions; negative control: deleting a row fails); `sampling.random_seed` is served from the top-level `seed` and `sampling.multi_state_strategy`/`multi_state_temperature` from `multistate`, with no stored field sharing a name with a view attribute (reflection test); `seed`, `encoding_fusion` and `decoding_fusion` are served from the stored values (EC paths, asserted against the stored value, not against the old baseline of 0 / `None`); writes raise naming `.replace`; the S1-02 `view_incompatible_uses` list is empty; the diff over `src/aminx/run/spec.py` and `src/aminx/run/specs.py` is empty; registry metadata dict keys are a subset of the published `field_meta.v1.json` (S4-07) |
| S1-31 | (b: score/jacobian/inspect/train task configs) same checks for those task classes, completing the `compat_nested` rows so the three-way path-set equality of D3a (rows == S1-01 P_old == view attribute set, 54 paths) holds in full from this item on; registry metadata on every leaf; the diff over `spec.py` and `specs.py` is empty |
| S1-32 | (c: codec and shadow test) Shadow-equivalence test over the S1-01 corpus: `RunSpec.from_flat(...)` projection equals `build_run_spec(facade)` on every field both have except the three expected-change fields `seed`, `encoding_fusion`, `decoding_fusion` (coalesced fields compared through `resolved_*`; the EC fields are checked against S1-01 `expected_post_flip` instead); `to_flat(from_flat(x))` equals the unchanged facade's flat view of `x`; `from_flat(to_flat(s)) == s` fixpoint; leaf-field registry metadata equals registry rows; **legacy view value gate (D3a, EC-aware): over every S1-01 corpus case and all five tasks, each of the 51 non-EC old nested paths read through the view of `from_flat(...)` equals its S1-01 `baseline` value exactly and warns once, and each of the 3 expected-change paths (`seed`, `encoding_fusion`, `decoding_fusion`) equals its recorded S1-01 `expected_post_flip` value; any path outside that list that differs fails; includes the quirk cases and `sampling.random_seed`/`multi_state_*` served across parent and sibling; negative controls: perturbing one baseline or expected value fails, and a view returning the old `seed` 0 / `None` fusion on a case where expected differs fails**; facade untouched (no diff under `spec.py` or `specs.py`) |
| S1-07 | Goldens byte-identical (flat JSON, row hash, projection of the 51 non-EC paths); **S1-01 small-L fixed-seed numeric cases (sample, score, jacobian, inspect) sha-equal to baseline, chunked per case**; compat contract tests (setattr error message names `.replace`, `inspect.signature` equal to the fixture, `==` and `hash` behaviour pinned); `pytest tests/run tests/cli` plus the **S1-02 reader-test selection** (`reader_tests.json`: every test file that reaches a nested read in `host/runner`, `streaming`, `plan`, `kernel_dispatch`, `_sampling_helper`, `_sampling_grid_lineage`, `campaign`, `prep`, `sampling/multistate_poe`, `io/sink_provenance`; no `-k` keyword selector), run on titanix one directory per process with its own timeout and completion stamp, exclusions only as recorded in the file with a reason and covered by S1-15; `_sync_run_spec`/`_run_spec_synced` absent from non-shim code; `isinstance` dispatch test; `DeprecationWarning` baseline recorded; changelog entry lists the `run_spec.seed` and `run_spec.encoding_fusion` / `decoding_fusion` changes; **legacy view completeness (D3a)**: `tests/run/test_legacy_view_paths.py` re-run on the flipped tree: path-set equality (rows == S1-01 P_old == view attribute set, 54 paths, both directions), the 51 non-EC paths equal to their S1-01 `baseline` value and the 3 expected-change paths (`seed`, `encoding_fusion`, `decoding_fusion`) equal to their S1-01 `expected_post_flip` value over the whole corpus (including `ligand.model_family`, `multistate.mode/n_states`, `io.manifest_path`, `io.sink_kind`, `resource.sample_batch_size/structure_batch_size`, `grid.grid_mode`, the `or 42` seed case with `seed` 0 and `sampling.random_seed` 42), warn-once, and the differential oracle `legacy_layout.build_run_spec(flat_case)` (flat_case = S1-01 recorded flat facade attributes, D3a) equals `flipped_spec.run_spec` on the 51 non-EC paths while on the 3 EC paths the moved-verbatim builder equals the `baseline` and the view equals `expected_post_flip` (any other difference fails); the old classes and builder in `legacy_layout.py` are content-hash-equal to the baseline definitions (moved verbatim); the S1-02 `legacy_layout_constructors` files import from `legacy_layout` and their tests pass; `derive_sink_spec(spec.run_spec, ...)` yields the explicit or minted `run_id`; `isinstance(rs.plan, PlannerTopology)` holds; `build_run_spec(SimpleNamespace)` and a `_MinimalScoringSpec`-style object route through `as_run_spec` correctly |
| S1-08 | flat-v1 golden unchanged; import-time exhaustiveness now derived from the registry (test: add an unregistered field in a fixture subclass, import fails); portable v2 goldens unchanged plus the new import-direction guard test; the three hand lists it regenerates are deleted (spec_json whitelists, spec_partition lists, the portable-v2 hand parser); the regenerated portable codec builds a unified `RunSpec` through `from_flat` and `run/run_spec_portable_json.py` no longer imports `legacy_layout` (import test); CLI option lists (`cli.py`) and JS key reads (`runspec_core.mjs`) stay hand-maintained (non-goals; JS covered by S1-18 and S4-20); `dataclasses.replace`/introspection sites in host spec_json users are migrated here |
| S1-09 | `tests/cli` green; golden `emit-*`, `spec validate`, `spec roundtrip`, `spec portable-roundtrip` outputs byte-identical |
| S1-23 | (built before the flip, against the S1-06 model) `as_run_spec` unit tests: real spec, spec with `.run_spec`, `SimpleNamespace`, hand-rolled class lift correctly; `MagicMock` raises a pointed error; the factory returns specs whose `to_flat` equals the facade's; negative control: a namespace missing a required attribute fails loudly, not silently defaulted |
| S1-24 | Codemod fixtures: positive (each rewrite class: `run_spec.sampling.<f>`, flat read, `dataclasses.replace`, `fields`, accessor rewrite for the five coalesced fields) and negative (look-alike attribute on an unrelated class is not rewritten); idempotent on second run; `scripts/spec/deprecation_paths.py` reports per-path warning counts and is the single measuring script for S1-10..S1-13 |
| S1-10..S1-13 | After each slice: goldens byte-identical; the slice's directories green on titanix; **the slice's own paths report zero deprecation warnings via `scripts/spec/deprecation_paths.py`** and `-W error::DeprecationWarning` passes for migrated paths; no global-count clause. S1-13 additionally lists and migrates the scripts/benchmarks attribute writers |
| S1-14 | Docs and examples render; `rg` finds no mention of `_sync_run_spec` or `build_run_spec` in README, `docs/` or `examples/` outside the migration note (the repo-wide check is S1-20's) |
| S1-15 | Pre-registered bathos sidecar: for each case (sample, score, jacobian, inspect, campaign row) new-path outputs equal the S1-01 baseline sha256 (tokens bitwise, logits within the S1-01 recorded tolerance, on the S1-01 recorded device); negative control: a different seed must differ; chunked, one process and timeout per case, per-case result file with completion stamp, resume reuses only cases whose input and artifact hashes match, reuse recorded in the result |
| S1-16 | Re-run of S1-02 inventory on the fully migrated tree, **excluding legacy-view reads and counting only direct nested reads and wire roles**; the prune list is the recorded set of fields with zero reads and no wire role; user sign-off item by item; tests green |
| S1-17 | In aminx: conformance test green on the real vendored file and red on a temp copy mutated by a pytest fixture (recorded control); sha256 header present; if the dev-dependency option is taken, `uv lock --check` passes |
| S1-18 | Drift test fails when a key is added to `runspec_core.mjs` without a registry row (negative control) and passes otherwise; Node test suite `browser/aminx-sampler` unchanged |
| S1-20 | Shims removed, including `run/legacy_view.py` and `run/legacy_layout.py` (files absent, no import of either); repo-wide `rg` for removed names returns only the allowlist (`CHANGELOG.md` and the migration note); goldens byte-identical; the S1-21 rehearsal re-run on the merged tree is green |
| S1-21 | Rehearsal on a throwaway aminx worktree with shims deleted (never merged): each migrated consumer's narrow spec-constructing tests run on titanix in a scratch checkout of the consumer whose aminx submodule path is pointed at the rehearsal tree by a local-only override; scratch root resolves explicit arg > `AMINX_CANARY_ROOT` (`none` disables) > `[tool.aminx.canary]` > `~/.config/aminx/config.toml` > off, with a `canary_root_source()` reporter; nothing is written to the consumer's tracked state or to another project's data directory; negative control: an unmigrated consumer commit fails the rehearsal; a consumer recorded as moot by the S3-17 decision (S1-28) is excluded and named in the record |
| S1-22 | Decision recorded; if "fix", a separate golden diff shows exactly the seed-0 and zero-valued-knob cases change and nothing else |
| S1-25 | (S1-15 equivalence record must be a pass before the cut) A published alpha (`molxmpnn`) containing the S1-07 flip and deprecation warnings exists on the index; changelog names the warning schedule and S1-20 timing; install smoke on titanix imports a flat-constructed spec and sees the warning |
| S1-26..S1-28 | Per consumer: spec-constructing narrow tests green on titanix after the submodule bump to a SHA containing S1-13 and the migrated reads; work is stacked on the S3-14 / S3-15 / S3-17 consumer branches (the same branches, not new ones off main), and S1-28 is also satisfied by a recorded "moot" outcome when the S3-17 decision leaves hautespout pinned; consumer branch only, no push to main without owner; `hautespout` item first re-reads its pyproject and `.gitmodules` (S1's A-row is S3's evidence only) |
| S1-29 | `tev_design` re-pin handoff note committed in aminx docs (wheel 0.1.0a24 to a post-S1 release: gap list, the two `AMINX_PIN.md` gate scripts, checklist); execution is the downstream owner's |
| S1-30 | mistypotts ownership note committed; mistypotts spec tests green on titanix |

## Open questions for the user

1. **Q1 Which survives.** Recommend design A (nested `RunSpec` as the model, flat vocabulary as a permanent
   codec and constructor signature). Fallback B (facade survives, `RunSpec` becomes a derived xtrax adapter) is
   cheaper but keeps two vocabularies; the D1 decision rule maps refuted S1-03 probes or an unresolvable A12 to
   it. Confirm A at S1-04, after both S1-02 and S1-03 have recorded results.
2. **Q2 Break timing for flat attribute reads and `.run_spec`.** Recommend: shims survive through at least one
   published alpha (S1-25, an item, so the DAG enforces it) before S1-20 removes them, and consumers are migrated
   first (S1-26..S1-29). Alternative: keep read shims indefinitely (zero break, permanent second vocabulary).
3. **Q3 `or`-coalescing quirks** (`random_seed=0` becomes 42, `multi_state_temperature=0.0` becomes 1.0,
   `num_samples=0` becomes 1). Recommend preserve in S1 (raw storage, `resolved_*` accessors, D9) and fix
   separately (S1-22) with an explicit changelog, because fixing changes sampled outputs for seed 0 and could
   invalidate pre-registered runs.
4. **Q4 TRW owner and conformance mechanism.** Recommend `mistypotts` as owner, aminx vendors a copy and tests a
   pinned sha256 with no import (alphex D4 pattern). The dev-dependency variant is optional and only if
   `uv lock --check` resolves. Alternatives: aminx owns and mistypotts vendors; or a 77-line stdlib-only leaf
   package (not recommended).
5. **Q5 Fold Potts into `RunSpec`?** Recommend no for now; shared `SpecFamily` protocol only; revisit with S4.
6. **Q6 Owner of the browser knob document and the letter-keyed lowering.** Recommend S4-20 (it depends on S1-05
   and S1-07), with the S1 registry as the single source and S1-18 as the interim drift test; S1 builds no
   `DesignConstraints` sub-config. Alternative: S1 builds it (round-0 S1-19), which duplicates S4-20. The
   `chains_to_design` vs `chain_id` semantic difference is S4-20's to settle.
7. **Q7 Pruning knobs accepted but ignored.** Recommend: scaffolding fields not in the flat surface are deleted
   outright; public flat kwargs that are provably inert go through the existing warn-and-ignore mechanism
   (`_DEPRECATED_SPEC_KWARGS`, `specs.py:19`) for one window, then are removed (S1-16 asks item by item).
8. **Q8 Portable v2 fate.** No consumer (A16). Recommend freeze and mark deprecated until S4 ships the
   schema-generated portable format, then delete.
9. **Q9 `RunSpec.run_id`.** Never populated today, so every `derive_sink_spec` call mints a fresh id.
   Recommend a follow-on (outside S1) deriving it deterministically from the canonical flat payload plus code
   SHA; S1 only keeps the field wired.
10. **Q10 Rename ordering (reversed from round 0).** The DAG encodes **rename-first**: S1-01/02/03/17 depend on
    S3-07, matching S3's own recommendation and avoiding a cycle (D7). Recommend keeping it. If you want S1
    first, drop those four edges and add `depends_on = ["S1-13", "S1-26", "S1-27", "S1-28"]` to S3-07; S1-25
    then depends on the S3 release item instead of S3-13. This is your call because it fixes which break
    (name or spec reads) downstream sees first.
11. **Q11 Consumer migration scope.** Recommend S1 owns migration branches for `mpnn_ext`, `asr`, `hautespout`
    (S1-26..S1-28) and only a handoff for `tev_design` (S1-29, wheel 0.1.0a24 is a re-pin project that belongs to
    its owner). Alternative: leave all four to their owners, in which case S1-20 depends on an acknowledgement
    instead. S1 and S3 **share the same consumer branches**: S3-14/15/17 rename the submodule URL, path and
    workspace member, then S1-26/27/28 bump the SHA on that branch, so each S1 item depends on its S3 twin. If
    S3-17 chooses "leave hautespout pinned", S1-28 is moot, hence S1-28 is `user_decision = true`.

## Backlog items

```toml
[[item]]
id = "S1-01"
title = "Pre-register and record characterization goldens (flat JSON, manifest row hash, RunSpec projection, portable v2, facade signature fixtures, the 54-path legacy nested-path baseline, D9 quirk and normalisation cases, small-L fixed-seed outputs) on the renamed but otherwise unchanged code"
repo = "molxmpnn"
size = "M"
depends_on = ["S3-07"]
gate = "bathos record shows determinism outcome pass (two generations byte-identical) and negative-control outcome pass; goldens including legacy_nested_paths.json (reflected path set, per-case baseline values, and per-case expected_post_flip values for the three expected-change paths seed, encoding_fusion, decoding_fusion derived from the documented rule) committed under tests/golden/spec_system/"
user_decision = false

[[item]]
id = "S1-02"
title = "Tracked AST script emitting spec_field_inventory.json (per-task reads, dead fields, name collisions, jit-argument and post-construction-write checks, per-read-site raw-vs-coalesced table, duck-construct classification, every nested run_spec path read in src/tests/scripts/examples checked against the S1-01 path set, view-incompatible uses, legacy-layout constructors, reader-test selection) with positive and negative controls"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-07", "S1-01"]
gate = "bathos record outcome pass; inventory JSON and reader_tests.json committed; seeded dead field reported and live field not reported; a seeded nested read of a path outside the S1-01 set is reported and an in-set read is not"
user_decision = false

[[item]]
id = "S1-03"
title = "Feasibility probes for design A on installed equinox (custom flat init, generated properties, pytree ops, static-key jit cache equality, replace/eq/hash) producing a go/no-go record"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-07"]
gate = "bathos record exists with every probe outcome evaluated and recorded (a refuted probe is a valid result); decision rule from D1 applied in the record"
user_decision = false

[[item]]
id = "S1-04"
title = "ADR recording the unification decision from the S1-02 and S1-03 records; mark 260611 three-layer clause amended and 260827 WS-A deletions superseded; close superseded backlog items"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-02", "S1-03"]
gate = "ADR committed under .praxia/docs/decisions and docs check clean; user ratified Q1"
user_decision = true

[[item]]
id = "S1-05"
title = "Field registry (src/aminx/run/fields.py, one row per task and flat name with default, type, static, normaliser pair) plus derived run_spec_fields.json and both-direction conformance test against the current facade"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-02", "S1-04", "S3-13"]
gate = "pytest tests/run/test_field_registry.py green on titanix: rows both directions, generated defaults, static set and init signatures equal the current facade; negative controls and snapshot-freshness test"
user_decision = false

[[item]]
id = "S1-06"
title = "Additive nested RunSpec layout, part a: new modules run/nested.py and run/legacy_view.py with DataConfig, ModelRef, PerturbationConfig and sampling/shared sub-configs (real defaults, raw storage, resolved_* accessors), the LegacyRunSpecView mechanism with compat_nested rows for the shared and sampling paths, and registry field metadata; spec.py and specs.py unchanged"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-01", "S1-02", "S1-05", "S4-07"]
gate = "unit tests green on titanix over hand-built nested specs: view attribute set equals compat_nested rows and the S1-01 path set for these sub-configs (both directions, negative control); random_seed served from top-level seed and multi_state_* from multistate; seed/encoding_fusion/decoding_fusion served from stored values; no stored field shares a name with a view attribute; registry metadata dict keys are a subset of the published field_meta.v1.json; empty diff over spec.py and specs.py"
user_decision = false

[[item]]
id = "S1-31"
title = "Additive nested RunSpec layout, part b: score, jacobian, inspect and train task configs with real defaults, raw storage, accessors, the remaining compat_nested rows (full 54-path set) and registry field metadata"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-06"]
gate = "unit tests green on titanix: task-config defaults equal the unchanged facade's; compat_nested rows equal the S1-01 54-path set and the view attribute set (three-way, both directions); empty diff over spec.py and specs.py"
user_decision = false

[[item]]
id = "S1-32"
title = "Additive nested RunSpec layout, part c: from_flat/to_flat codec for all task classes, the shadow-equivalence test against the unchanged facade, and the legacy-view value gate over all 54 old nested paths"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-06", "S1-31"]
gate = "shadow-equivalence, normalised round-trip and fixpoint tests green over the S1-01 corpus on titanix; on every corpus case the 51 non-EC old nested paths read through the view equal their S1-01 baseline value and the three expected-change paths (seed, encoding_fusion, decoding_fusion) equal their S1-01 expected_post_flip value, any other difference fails; empty diff over spec.py and specs.py"
user_decision = false

[[item]]
id = "S1-07"
title = "Flip: old layout classes and old build_run_spec move verbatim to run/legacy_layout.py, spec.py re-exports the unified model; facade classes become thin RunSpec subclasses with generated flat init, replace(), flat read properties and setattr error; run_spec returns the legacy nested view; sync machinery becomes a shim and build_run_spec delegates to as_run_spec"
repo = "molxmpnn"
size = "L"
depends_on = ["S1-06", "S1-31", "S1-32", "S1-23"]
gate = "goldens and S1-01 numeric small-L cases byte-identical on the recorded device; compat contract tests pass; test_legacy_view_paths.py green (54-path set equality, 51 non-EC paths equal their S1-01 baseline and seed/encoding_fusion/decoding_fusion equal their S1-01 expected_post_flip, differential oracle: the moved-verbatim legacy_layout.build_run_spec applied to the S1-01 recorded flat facade attributes (not to the flipped spec) equal on the 51 non-EC paths and differing on the 3 EC paths exactly as recorded); build_run_spec shim probed for duck objects; tests/run, tests/cli and the S1-02 reader-test selection (no keyword selector) green on titanix, chunked per directory; warning baseline recorded"
user_decision = false

[[item]]
id = "S1-08"
title = "Regenerate spec_json whitelists, spec_partition classification and the portable v2 hand parser (with import-direction guard) and knob_matrix from the registry; delete those three hand lists (CLI option lists and JS key reads stay hand-maintained)"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-07"]
gate = "flat-v1 and portable-v2 goldens unchanged; import-time exhaustiveness fails for an unregistered fixture field; the three generated lists are deleted; the portable codec no longer imports legacy_layout"
user_decision = false

[[item]]
id = "S1-09"
title = "CLI: build specs through the registry-backed constructors and keep spec validate/roundtrip/portable-roundtrip/emit-* output unchanged"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-07"]
gate = "tests/cli green on titanix and CLI golden outputs byte-identical"
user_decision = false

[[item]]
id = "S1-23"
title = "Duck-spec seam as_run_spec(obj) and shared test factory tests/_support/spec_factory.py (make_spec, make_duck_spec), with controls"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-06", "S1-31"]
gate = "seam and factory unit tests green on titanix; MagicMock raises a pointed error; namespace missing a required attribute fails loudly"
user_decision = false

[[item]]
id = "S1-24"
title = "Registry-driven codemod tool (nested-read, flat-read, replace/fields and resolved_* accessor rewrites) plus scripts/spec/deprecation_paths.py measuring per-path warning counts, with positive and negative fixtures"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-05", "S1-07"]
gate = "codemod fixtures green (positive, look-alike negative, idempotent second run); deprecation_paths.py reports counts on a fixture tree"
user_decision = false

[[item]]
id = "S1-10"
title = "Read-migration slice A (via seam and S1-24 codemod): src/aminx/host/ spec.run_spec.X and flat reads to nested reads, dataclasses.replace to spec.replace; migrates the fixtures its own gate breaks"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-08", "S1-23", "S1-24"]
gate = "goldens byte-identical; host tests green on titanix; deprecation_paths.py reports zero warnings for src/aminx/host"
user_decision = false

[[item]]
id = "S1-11"
title = "Read-migration slice B (via seam and S1-24 codemod): src io, sampling, tiling, training, utils, run/_exports, package __init__; migrates the fixtures its own gate breaks"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-23", "S1-24"]
gate = "goldens byte-identical; affected test dirs green on titanix; deprecation_paths.py reports zero warnings for the slice paths"
user_decision = false

[[item]]
id = "S1-12"
title = "Read-migration slice C: remaining tests/host fixtures and idioms (duck specs to factory, fields/replace, run_spec access, spec equality assertions)"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-10", "S1-23", "S1-24"]
gate = "tests/host green on titanix with -W error::DeprecationWarning; goldens byte-identical"
user_decision = false

[[item]]
id = "S1-13"
title = "Read-migration slice D: remaining tests, scripts/ (including the benchmark attribute writers and build_run_spec users), examples/"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-09", "S1-10", "S1-11", "S1-23", "S1-24"]
gate = "narrow test dirs green on titanix with -W error::DeprecationWarning; deprecation_paths.py reports zero warnings for tests (excluding tests/host), scripts and examples"
user_decision = false

[[item]]
id = "S1-14"
title = "Docs: README, CHANGELOG (incl. run_spec.seed and run_spec.encoding_fusion/decoding_fusion changes), COMPOSITION_GUIDE, examples text and the downstream migration note"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-10", "S1-11", "S1-13"]
gate = "docs and examples render; rg finds no _sync_run_spec or build_run_spec in README, docs/ or examples/ outside the migration note"
user_decision = false

[[item]]
id = "S1-15"
title = "Pre-registered bathos behavioural-equivalence run (sample, score, jacobian, inspect, campaign row) new path vs S1-01 baseline, chunked per case and resumable"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-01", "S1-10", "S1-11"]
gate = "bathos record outcome pass: every case sha256-equal to baseline, negative control differs, reuse of prior cases recorded"
user_decision = false

[[item]]
id = "S1-16"
title = "Prune dead fields from the re-measured inventory on the fully migrated tree (scaffolding deleted; inert public kwargs through warn-and-ignore)"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-12", "S1-13", "S1-15"]
gate = "S1-02 inventory re-run recorded excluding legacy-view reads; prune list signed off item by item; tests green on titanix"
user_decision = true

[[item]]
id = "S1-17"
title = "PottsTRWRunSpec single owner in aminx: vendored copy with sha256 header and a no-import conformance test with a mutate-a-copy negative control (dev-dependency variant only if uv lock --check passes)"
repo = "molxmpnn"
size = "S"
depends_on = ["S3-07", "S3-13"]
gate = "conformance test green on the real vendored file and red on a mutated temp copy via pytest fixture; sha256 header present"
user_decision = true

[[item]]
id = "S1-30"
title = "mistypotts ownership note for potts_trw_spec.py (owner; aminx vendors; changes bump the aminx pinned sha256)"
repo = "mistypotts"
size = "S"
depends_on = ["S1-17"]
gate = "note committed in mistypotts; its spec tests green on titanix"
user_decision = false

[[item]]
id = "S1-18"
title = "Key-set drift test between runspec_core.mjs spec reads and registry browser_key rows, with a negative control (deleted by S4-20 once the generated schema replaces the hand copy)"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-05"]
gate = "drift test fails when a JS key is added without a registry row and passes otherwise; browser/aminx-sampler Node tests unchanged"
user_decision = false

[[item]]
id = "S1-25"
title = "Release cut: publish a molxmpnn alpha carrying the flip, deprecation warnings and the changelog warning schedule (the deprecation window as an item)"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-12", "S1-13", "S1-14", "S1-15", "S3-13", "S5-52"]
gate = "alpha present on the index at the version the S3 release table assigns to the flip-carrying alpha (the next unused molxmpnn alpha after S5-52's tag, 0.2.0a6 under S3's table; S3-13 is 0.2.0a4 and S5-52 takes 0.2.0a5), whose tag contains S5-52's work, and the S1-15 equivalence pass precedes the cut; titanix install smoke imports a flat-constructed spec and observes the DeprecationWarning"
user_decision = false

[[item]]
id = "S1-26"
title = "Migrate mpnn_ext to the post-S1 surface: on the S3-14 consumer branch (stacked on it, not a new branch off main), bump its external/aminx submodule SHA and migrate spec reads"
repo = "mpnn_ext"
size = "S"
depends_on = ["S1-13", "S3-10", "S3-14"]
gate = "mpnn_ext narrow spec-constructing tests green on titanix on the S3-14 consumer branch (stacked on its commits); no push to main without the owner"
user_decision = false

[[item]]
id = "S1-27"
title = "Migrate asr to the post-S1 surface: on the S3-15 consumer branch (stacked on it, not a new branch off main), bump its aminx submodule SHA and migrate spec reads"
repo = "asr"
size = "S"
depends_on = ["S1-13", "S3-10", "S3-15"]
gate = "asr narrow spec-constructing tests green on titanix on the S3-15 consumer branch (stacked on its commits); no push to main without the owner"
user_decision = false

[[item]]
id = "S1-28"
title = "Migrate hautespout (re-read its pyproject and .gitmodules first): on the S3-17 consumer branch (stacked on it, not a new branch off main), bump its aminx submodule SHA and migrate spec reads, or record the item as moot if S3-17 leaves hautespout pinned"
repo = "hautespout"
size = "S"
depends_on = ["S1-13", "S3-10", "S3-17"]
gate = "hautespout narrow spec-constructing tests green on titanix on the S3-17 consumer branch (stacked on its commits), or the item is recorded as moot by the S3-17 decision (hautespout left pinned); no push to main without the owner"
user_decision = true

[[item]]
id = "S1-29"
title = "tev_design re-pin handoff note (wheel 0.1.0a24 to a post-S1 release: gap list, the two AMINX_PIN.md gate scripts, checklist)"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-25"]
gate = "handoff note committed under .praxia/docs/handoffs and docs check clean"
user_decision = false

[[item]]
id = "S1-21"
title = "Shim-removal rehearsal: on a throwaway aminx worktree with shims deleted, run migrated consumers' narrow tests in scratch checkouts via a local-only submodule override (config-resolved scratch root, no writes to consumer tracked state)"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-13", "S1-26", "S1-27", "S1-28"]
gate = "all migrated consumers' narrow tests green on titanix against the shim-less tree (a consumer recorded as moot by the S3-17 decision is excluded and named in the record); negative control (an unmigrated consumer commit) fails; canary_root_source() reports the deciding layer"
user_decision = false

[[item]]
id = "S1-20"
title = "Remove compat shims: flat read properties, run_spec alias and legacy_view.py, legacy_layout.py, build_run_spec, as_run_spec duck lifting, _sync_run_spec, dataclasses.replace guard, facade remnants"
repo = "molxmpnn"
size = "M"
depends_on = ["S1-12", "S1-13", "S1-14", "S1-15", "S1-21", "S1-25", "S1-29"]
gate = "legacy_view.py and legacy_layout.py absent and not imported; repo-wide rg for removed names returns only the allowlist (CHANGELOG.md, migration note); goldens byte-identical; S1-21 rehearsal re-run green on the merged tree"
user_decision = true

[[item]]
id = "S1-22"
title = "Decide and (if approved) fix the or-coalescing quirks (random_seed 0, zero-valued knobs) as an isolated behaviour change"
repo = "molxmpnn"
size = "S"
depends_on = ["S1-15"]
gate = "decision recorded; if fix approved, golden diff shows only the seed-0 and zero-valued-knob cases change"
user_decision = true
```

## Revision log

### Coherence round 1 (cross-spec fixes C1, C3, C10, C14)

- **C1** Already rename-first. D7/Q10 now state that Q10, S3 Q4 and S2 Q7 are one question answered once at S3-01;
  S2 must align (its TOML has no S3 edge today).
- **C3** D7 now protects S2's A0 freeze window: S1-02 records whether S2's consumed-file list reads spec fields
  (expected none, A17); S1-11 and S1-24 exclude that list; if non-empty, S1-11 gains `depends_on` S2-06.
- **C10** D4 metadata hook changed from `aminx_field` to `{"xtrax": {expose, browser_key, ...}}` (S4 FieldMeta
  shape) built by a local plain-dict function; aminx-specific keys beside it; per-field static unchanged (A28);
  no S1-06 to S4 edge (S4-32 validates the dict).
- **C14** S1-25 gate states its version from the S3 table. *Superseded in round 2 (CH2-01):* S1-25 now follows
  S5-52 (version 0.2.0a6) and contains its work.

### Round 1 (adversarial objections C1-C14)

Applied (conceded):
- **C1** Duck fixtures: new D2 seam `as_run_spec` plus shared factory (S1-23); S1-10/11/12/13 depend on it; the
  slice whose gate breaks a fixture migrates it; S1-02 classifies the 113 constructs. Construct-only claim
  narrowed to real facades.
- **C2** Rename ordering: reversed to rename-first (S3 is the root and recommends it); S1-01/02/03/17 depend on
  S3-07; D7 rewritten, flip path documented; Q10 reversed; consumer inventory reconciled with S3 A1 (asr,
  hautespout added); file-count measures explained. Added A31.
- **C3** One source and one owner: registry is the single source, exposed as nested-field metadata plus derived
  snapshot (P1/P2, D4); knob document and lowering owned by S4-20 (depends on S1-05, S1-07). Old S1-19 **removed**
  (id not reused); Q6 reworded. Added A30.
- **C4** FieldDef is one row per (task, flat_name) with order, type, default, static, normaliser pair; S1-05 asserts
  generated defaults, static set and signatures; S1-03 probes jit cache-key equality. Added A27, A28.
- **C5** D9 rewritten: raw storage, `resolved_*` accessors, per-read-site table from S1-02 as S1-06 precondition
  (explicit edge), rows N1-N5, weakened round-trip contract, S1-01 cases. Added A26, A29.
- **C6** S1-14 gate limited to README/docs/examples and now depends on S1-13; repo-wide rg moved to S1-20 with an
  allowlist.
- **C7** Consumer items S1-26..S1-29 added; S1-20 depends on them through S1-21 (rehearsal on a throwaway tree,
  config-resolved scratch root, no writes to consumer state); A14 and the risk row corrected (vendored
  submodule breaks at SHA bump).
- **C9** S1-07 gate gains the S1-01 numeric small-L cases (chunked).
- **C10** Codemod is item S1-24 with fixtures; S1-10/11/12/13 depend on it; gate is per-slice-path zero warnings
  via a fixed script, global-decrease clause dropped.
- **C11** S1-16 depends on S1-12, S1-13, S1-15; inventory excludes compat-property reads.
- **C12** S1-17 repo set to aminx; default sha256 comparison with no import; dev dependency optional behind
  `uv lock --check`; negative control is a pytest fixture; A23 `If false` now `none`; mistypotts note split to S1-30.
- **C14** Release-cut item S1-25 (depends on S3-13) and S1-20 depends on it.

Partially applied:
- **C8** Sequencing claim declined: S1-06 already depends on S1-05, which depends on S1-02, so the write-site
  result gates S1-06 transitively (A19 now says so). The compat gap was real and is applied: D2 compat contract
  (setattr error, positional order via signature fixture, equality, hash, repr, `fields`/`replace`), scripts/
  benchmarks writers listed in Current state and S1-13, S1-07 compat tests.
- **C13** Gate wording and decision rule applied (S1-03 gate, D1 rule); S1-04 now depends on S1-02 and S1-03.
  Declined: the claim that the A12 AST result has no edge to the flip, since S1-05, S1-06 and S1-07 already chain
  from S1-02 and S1-04.

Declined: none outright. Not re-verified in this round: hautespout's own pyproject and `.gitmodules` (S3's evidence is
cited and S1-28 re-reads it); the 113/22 duck counts and the 57/46 consumer file counts remain exploratory until S1-02.

### Round 2 (adversarial objections R2-1 to R2-10)

Applied (conceded):
- **R2-1** D2 gains generated, coalescing, warning compat properties for every existing nested path
  (`sampling.random_seed`, `sampling.multi_state_strategy`, `multistate.combine_strategy`, coalesced `num_samples`,
  `sampling_strategy`, `multi_state_temperature`); built in S1-06/S1-31, probed in the S1-07 gate; removed at S1-20.
- **R2-2** S1-26/27/28 now depend on S3-14/15/17; S1-28 is `user_decision = true`; Q11 states the shared branches.
- **R2-3** S1-08 narrowed to the three lists it regenerates; CLI lists and JS reads stay hand-maintained.
- **R2-4** `as_run_spec` (S1-23) now precedes the flip (S1-23 depends on S1-06/S1-31; S1-07 depends on S1-23); the
  S1-07 shim behaviour for duck objects is specified and gated.
- **R2-5** S1-25 depends on S1-15. Declined adding S1-22: it is a user-gated optional behaviour change and the
  changelog states the preserve-quirks default (Q3).
- **R2-6** S1-06 split into S1-06 (a), S1-31 (b), S1-32 (c codec plus shadow test); S1-07 depends on all three.
- **R2-7** `topology_hash` removed; P1 split into stable surface and P1b shims.
- **R2-9** Device, XLA flags and weights revision pinned in the S1-01 sidecar; S1-07/S1-15 compare on the same
  device; S3-05 fixture reuse stated as conditional (overlap *unverified*, S1-01 reads S3-05's record first).

Partially applied:
- **R2-8** Added an "edges owed by other specs" list (S4-20 -> S1-05/S1-07/S1-18) and the `repo = "molxmpnn"`
  convention (applied to all S1 items). The S4 edge itself remains S4's to declare at assembly.
- **R2-10** S1-09 dropped from S1-10 and S1-11 (no shared file named; S1-13 keeps it). S1-08 dropped from S1-11 and
  kept on S1-10, which migrates host spec_json users and `dataclasses.replace` sites that S1-08 regenerates.

### Coherence round 2 (cross-spec fixes CH2-01 to CH2-04)

- **CH2-01** S1-25 `depends_on` gains S5-52, making it comparable with S5-52 in the union DAG (S3-28 requirement).
  S1-25 gate now names the next unused alpha after S5-52's tag (0.2.0a6 under S3's table), requires the tag to
  contain S5-52's work, and keeps the S1-15-before-cut rule. D7 sentence and the Round-1 C14 line are superseded
  accordingly; S5-52 is not delayed by the S1 flip chain because S1-25 is a tail item.
- **CH2-02** S1-06 `depends_on` gains S4-07 (S1-31 inherits). D4 points the metadata shape at `field_meta.v1.json`;
  the S1-06 gate asserts the key subset; the "no S1-06 to S4 edge" sentence is removed; new Consumes row added.
  No cycle: S4-07 depends only on S4-01 and S4-02.
- **CH2-03** S1-05 and S1-17 `depends_on` gain S3-13 so no S1 code lands inside the S3-13 tag window; S1-06, S1-31,
  S1-32, S1-07, S1-18 follow transitively; S1-01/02/03 are additive-only and need no edge (D7).
- **CH2-04** Stated in D7 that S1-17's optional mistypotts dev dependency (uv.lock change) lands on main without
  affecting S2's pinned F' checkout, so no S2 edge is added. Lock churn from S4 items is S2/S4's to order.

### Convergence check (R2-1 residue)

R2-1 was only partly resolved by round 2: the D2 compat properties could not exist as specified. Resolved here:
- **Mechanism replaced (a).** Per-sub-config compat properties are withdrawn. A stored field cannot share its name
  with a coalescing property (D9 stores `sampling.num_samples` and `sampling_strategy` RAW), and a sub-config cannot
  reach the top-level `seed` or the `multistate` sibling (A32). `spec.run_spec` now returns a plain read-only
  `LegacyRunSpecView` that closes over the whole spec (new D3a); `run_spec` no longer returns `self`;
  `build_run_spec` post-flip returns the same view. D2 and D6 updated.
- **Complete path set (b).** D3a lists all 54 old paths (47 sub-config fields plus 7 top-level names, A32) with a kind,
  a new home and a view value, including `ligand.model_family`, `multistate.mode/n_states`, `io.manifest_path`,
  `io.sink_kind`, `resource.sample_batch_size/structure_batch_size`, `grid.grid_mode`. The D3 io, resource,
  multistate and model_ref rows now say which legacy fields are derived rather than stored (the stray `sink_kind`
  in the io row is removed). The completeness gate is reflection-based, not table-based: S1-01 records P_old and
  per-case baseline values (`legacy_nested_paths.json`), S1-02 checks every nested read in src/tests/scripts/
  examples against it, and S1-06/S1-31/S1-32/S1-07 assert three-way path-set equality and value equality.
  The D2 "~14 nested reads" figure is replaced by the exploratory 28 (src) plus 4 (tests) distinct paths.
- **Gate widened (c).** S1-07 no longer runs `tests/host -k "spec or campaign"`; it runs the S1-02 `reader_tests.json`
  selection (every test reaching a nested read in runner, streaming, plan, kernel_dispatch, multistate_poe,
  sink_provenance and the other reader modules), chunked per directory, plus a differential oracle against the
  moved-verbatim old builder. Migration of the reads stays in S1-10..S1-13.
- **Coexistence (d).** New model and view live in `run/nested.py` and `run/legacy_view.py`; `spec.py` and `specs.py`
  are byte-unchanged through S1-06/S1-31/S1-32 (empty-diff gate); S1-07 moves the old classes and builder verbatim to
  `run/legacy_layout.py` and re-exports the unified names; S1-08 stops the portable codec importing it; S1-20
  deletes `legacy_layout.py` and `legacy_view.py`.
- **Ledger.** Added A32 (closed path set, no parent reference), A33 (`derive_sink_spec` reads only `run_id`,
  xtrax v0.4.0a11), A34 (nothing treats the view as a `RunSpec`, deferred to S1-02 because the spike harness is
  absent), A35 (old builder runs through `getattr`, so it is a usable oracle).
- **Graph.** No `depends_on` edge changed, so the S1 subgraph is still acyclic; only titles and gates changed.
  S1-01 and S1-02 gained outputs; S1-06 now consumes S1-02's `view_incompatible_uses` through its existing S1-02 edge.

### Convergence check 2 (convergence-check-2: value-gate contradiction)

The final recheck found that the legacy-view value gate (D3a "Value equality over the corpus", S1-32, S1-07 and their
toml `gate` strings) demanded old == new for every one of the 54 paths, while D9/D3/A2/A3 intentionally change three of
them (`seed` 0 -> raw `random_seed`; `encoding_fusion` / `decoding_fusion` `None` -> the stored flat value). As written
the gate could never pass without reverting the design. Resolved by the suggested direction, which was checked and
kept (an alternative, making the view return the old 0 / `None`, would defeat D9 and the A3 "finally populated" row):
- **Expected-change list (EC) = {`seed`, `encoding_fusion`, `decoding_fusion`}**, closed, in D3a (new bullet), the old-path
  map (rows split out and marked EC), D9 and the risk table. Nothing joins EC without editing the spec.
- **S1-01 records `expected_post_flip` per corpus case** for the three EC paths, next to `baseline`, derived from the
  documented rule applied to the unchanged facade's flat attributes (raw `random_seed`; flat fusion attributes after the
  old normalisation), never from running new code. The corpus must include non-default seed, `random_seed=0` and
  non-`None` fusion cases so expected differs from baseline for each EC path. The RunSpec projection golden covers the
  51 non-EC paths only, so its S1-07 byte-identity does not contradict the changes.
- **Gates (D3a, S1-32, S1-07, S1-06, and the toml strings)**: exact equality to `baseline` on the other 51 paths,
  exact equality to `expected_post_flip` on the 3 EC paths, any other difference fails; negative controls (perturbed
  baseline or expected value fails; a view returning old 0 / `None` where expected differs fails). The S1-07 differential
  oracle requires equality on 51 paths and, on the 3 EC paths, old-builder == `baseline` (proves verbatim move) and view
  == `expected_post_flip`. The S1-32 shadow projection excludes the 3 EC fields from the `build_run_spec` comparison.
- **S1-02** also confirms no reader depends on the old `run_spec.seed` / fusion values (expected none, A2/A3).
  Changelog (S1-07, S1-14) now names the fusion changes as well as the seed change.
- **Non-blocking hardening**: S1-26/27/28 titles and gates say they stack on the S3-14/15/17 consumer branches; the S1-28
  gate and the S1-21 gate (and table rows) accept "recorded as moot by the S3-17 decision" (hautespout left pinned).
- **Graph.** No `depends_on` edge, id or toml block structure changed; the graph is unchanged and acyclic.

### Convergence check 3 (orchestrator, after an independent skeptical recheck)

- **B1 (blocking, fixed).** The S1-07 differential oracle fed the moved-verbatim `build_run_spec` the *flipped*
  spec, which cannot pass: after the flip `seed` is a stored field (`spec.py:381` would return the raw seed, not 0)
  and `precision` is the `PrecisionConfig` sub-config (`spec.py:199` would yield `"fp32"`, not a training spec's
  `"bf16"`). The oracle now runs on `flat_case`, a `SimpleNamespace` of the S1-01 recorded flat facade attributes.
  Updated D3a, the S1-07 verification row, the S1-07 toml gate and A35.
- **B2 (blocking, fixed).** S1-02 checks nested reads against the S1-01 path set but did not depend on S1-01;
  added `S1-01` to its `depends_on` (S1-01 depends only on S3-07, so no cycle).
- Not applied (minor, recorded for implementation): fusion identity should be encoded as qualname plus module or a
  fixed corpus sentinel (two lambdas share a qualname); "after the facade's normalisation" is vacuous for fusion
  (no normalisation exists); the unified model needs explicit `axes=[]` / `carry_specs` defaults, since
  `xtrax.run.RunSpec.axes` and `seed` are required fields.
