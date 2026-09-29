---
title: PottsMPNN + LASErMPNN as xtrax-composed model families on the central runner
description: Port KeatingLab PottsMPNN and polizzilab LASErMPNN into aminx as xtrax-composed FamilyDrivers dispatched from aminx.host.runner, with redsox knob-superset, xtrax-tier parity, and bathos-preregistered gates
task_id: 260929_potts-laser-xtrax-compose
status: draft-r5
created: 260929
amends: decisions/260605_potts-parallel-not-stageset.md (scope-narrowing, see §3)
adversarial_log: audits/260929_potts-laser-spec-adversarial-log.md
---

# PottsMPNN + LASErMPNN as xtrax-composed model families

Revision history:
- r0 (8ea7604e) → r1 (3ee01410): FamilyDriver seam; removed `plan_builder`, `post_decode` slot,
  `InterpolatedTieFuse`, LASEr `sc_coords` carry.
- r1 → r2: **all PottsMPNN sampling in the driver** (`PottsARDecode` reusing aminx `DecoderLayer`s;
  removed `mpnn_randn`, hybrid routing predicate, `emits_potts`); driver outputs via xtrax
  `ZarrStagingSink` (removed staging-sink reuse claim); runner golden gate; concrete L-DRV lint;
  `nodes`=Gibbs; padding/K convention; LASEr alphabet boundary; LASEr step order; oracle shims;
  distributional protocol with pilot; per-sidecar negative controls; junit-backed redsox harness;
  quirk + divergence tables; explicit fallback raise; family temperature default.
- r2 → r3: T0.0 sink `run_id` prerequisite (existing aminx Zarr paths TypeError at pinned xtrax);
  normative row kernel + f64-capable `DecoderLayer` accumulation; `edge_valid` as decoder
  `attention_mask`; goldens split T0.5a/T0.5b minus provenance attrs; UNSET resolution hook;
  device-side tie-group order; collision-free Zarr keys; corrected pilot rule + pinned structures +
  shim check; inverse-CDF clamp; L-DRV allowances; conftest outcome plugin; spec_json Options.
- r3 → r4: tied `rank_flat` from flattened group order (static `(L_pad,M_max)` groups); LASEr χ
  offset head + χ-angle gate; pre-refine energy ranking + `refined_sequence`; PottsMPNN host
  featurizer A0 with upstream gap rows and `present`/`pad_valid` mask table (retires r3
  `edge_valid`); per-wave redsox gate loop; tied last-member select; `nodes` fw recompute; sealed
  DecLayer fixture.
- r4 → r5: AR masks use `present` only (refine/`nodes` keep `present·chain_M_pos`) + fixed-position
  fixtures/controls; `knn_boundary_tie` detection/exclusion (stock features untouched); port
  self-test wave; order-injection padding + tied oracle `randn` injection; gapped `sequences_to_score`
  alignment.

## 0. Goal, non-goals, assumed decisions

**Goal.** Through the one central runner (same CLI, same `RunSpecification`, same
`host/runner.sample`/`score`/`jacobian`/`inspect` entry points, same xtrax tiling/sinks):

```bash
aminx run --model-family pottsmpnn --checkpoint-id pottsmpnn_vanilla_20 sample --inputs x.pdb --potts-options-json opts.json
aminx run --model-family pottsmpnn --checkpoint-id pottsmpnn_vanilla_20 score  --inputs x.pdb --output-kind ddg --potts-options-json opts.json
aminx run --model-family lasermpnn --checkpoint-id lasermpnn_0p1A_nothing_heldout sample --inputs lig.pdb --laser-options-json opts.json
```

every input knob of the pinned upstream implementations is reachable from `RunSpecification`
(or is a justified exclusion/divergence row), and numerical/behavioural parity against pinned
upstream PyTorch is demonstrated by pre-registered bathos runs over an xtrax-contract tier ladder.

**"xtrax-composed" (normative, enforced by L-DRV §2.4).** Per-structure computation is `eqx.Module`
stages; batch/sample/temperature/candidate/partition/proofread-order/dropout-seed loops are xtrax
axes (`AxisSpec` → `BatchPlanner` → `make_axis_dispatch_via_xtrax`); cross-axis reductions are
`AxisBoundary` `Fuse`s; host export is xtrax `ZarrStagingSink`. Genuinely sequential dependencies
(AR steps, tied members, refine sweeps) are `lax.scan`/`while_loop` at allowlisted sites.

**Non-goals (v1)**, each a recorded exclusion row (§6.4), never silent: training; LASEr
`entropy_decoder`; PottsMPNN `mutation_search.py`; CA-only PottsMPNN; MSA vocab-22 checkpoints;
plotting; migrating TRW `aminx.potts.PottsModel`; LASEr partial-charge entry point; portable-JSON v3;
gradient (T4) consumers.

**Assumed decisions (defaults applied; user may override — log U1–U5, U-a–U-c).**
- U1: LASEr is a FamilyDriver; MPNN contracts (`StageSet`, `EncoderOutput`, `ModelProtocol`,
  `DecodeMode`, `make_inference_plan`, `AutoregressiveDecode`, `kernel_dispatch`) unchanged.
- U2/U-b: **bit-parity** for all numerical and sampling semantics incl. quirks (§6.5); pure I/O
  plumbing bugs are fixed and recorded as divergences (§6.5b). Family defaults follow upstream.
- U3: MSA vocab-22 deferred. U4: local harness over redsox library `check_superset` (§6.6).
- U5: LASEr proofreading + two-structure tied in v1.
- U-a: all PottsMPNN sampling via the driver.
- U-c: LASEr default temperature `None` (argmax) via a family-resolved default (§2.3).
- C3-04(b): stock MPNN decoder unchanged; PottsMPNN nll/logits fallback parity asserted only on
  unpadded runs; pre-existing padded-neighbour behaviour recorded (§6.5b).
- C3-02: sink `run_id` fix is prerequisite task T0.0 in this epic; deterministic spec-hash id.

## 1. Recon evidence base

| Fact | Anchor |
|---|---|
| Runner entry points | `cli.py` `run` callback (`:448-461`, `--model-family` before subcommand) → `run_sample`/`run_score`/`run_jacobian`/`run_inspect` → `host/runner.py` `sample/score/jacobian/inspect` |
| score bypasses InferencePlan unless averaging | `host/runner.py:498-506` (`make_score_fn`), `one_hot(seq,21)` `:626`, `output_h5_path` NotImplemented `:489-491`, NLL result `:661-670`; `sequences_to_score` required `run/specs.py:556-561`, `cli.py:816` |
| MPNN AR ignores `decoding_order_fn`; order from wave schedule | `inference/decode/autoregressive.py:191-420`; `inference/bundle_builder.py:76,187-188,237-243`; `host/kernel_dispatch.py:222-258`; `decoding_order_fn` callable `run/specs.py:260`, `utils/decoding_order.py:21-24` |
| MPNN tied fuse = plain mean | `autoregressive.py:326-331` |
| Staging sink carries only seq/logits | `host/output_sinks.py:51-153`; `types/protocols.py:102-120` |
| xtrax generic array sink | `xtrax/run/zarr_sink.py:254-313` `ZarrStagingSink.stage(key, attrs, **arrays)` + `take(key)`; provenance attrs `_CORE_PROVENANCE_FIELDS` `:34` stamped on root/groups; **at pinned 56a9f551 `SinkSpec.run_id` is required (`run/sink.py:26`) and every aminx call site omits it (`host/streaming.py:80`, `host/runner.py:1225`, `io/designs.py:81`, `sampling/multistate_poe.py:605`, `scripts/analysis/jacobian_profile.py:187`) → all existing aminx Zarr paths TypeError today (fixed by T0.0)**; `derive_sink_spec` `run/sink.py:61-105` |
| Structure ids positional | `host/_sampling_helper.py:36-61`; prep forces inference mode `host/prep.py:200` |
| `DecoderLayer` reusable per row | `model/decoder.py:281` |
| Family Literal + consumers | `run/specs.py:236`; derivation `:476-490`; `_sampling_helper.py:255`, `prep.py:47-49`, `run_spec_portable_json.py:12-15,153`, `multistate_poe.py:609`, `streaming.py:84`, `_sampling_grid_lineage.py:94,111`, `campaign.py:95`, `run/spec.py:321` |
| `load_model` id normalisation | `io/weights.py:386-392`; `VOCAB_SIZE` `:453` |
| aminx sampling knobs | `temperature` default 0.1 non-Optional (`run/specs.py:574`), `bias`, `fixed_positions`, `fixed_tokens`, `num_samples`; `fixed_mask`, `tied_positions`, `noise`, `random_seed`; no omit |
| Encoder output | `types/bundles.py:457-486` `EncoderOutput(node_features, edge_features (S,L,K,D), neighbor_indices, mask)` |
| No CA featurizer / no B-factors | `rg` 0 hits each |
| TRW Potts model (distinct) | `potts/model.py:71`; ADR 260605; `tests/lint/test_potts_import_boundary.py` |
| RS-6b lint pattern | `tests/lint/test_rs6b_flat_field_gate.py:54-80`; `.ast-grep/rules/rs6b-host-flat-field-ban.yml` |
| Upstream PottsMPNN | KeatingLab/PottsMPNN @ `0cb0a58`; class `potts_mpnn_utils.py:1225-1261` (ProteinMPNN layers + `etab_out`); forward `:1267-1290`; K=min(48,L) `:1150`; CLI decode path `decoder :1415-1488` (order `:1419-1421`, masks `:1422-1428`, masked rows `:1448`, fixed select `:1483`) and `tied_decoder :1599-` (body as `:1546-1595`, groups `:1515-1523`, last-member leak `:1575-1589`); called from `sample_seqs.py:226,235` |
| PottsMPNN energy/refine | `run_utils.get_etab :816-835`; `etab_utils.functionalize_etab` denom 4 excl. self `:177-178,253-271`; `calc_eners :299-309`; alphabets `potts_mpnn_utils.py:69` vs `etab_utils.py:362-366`; `optimize_sequence :113-198` (binding `:135-148`, accumulate `:148,176`, cap `:116-120`), `nodes :180-270`; T floor `sample_seqs.py:38-39`; energy sort `:268-287`; refine order keying `:274-279,321-345`, `run_utils.py:110-111`; input list format `sample_seqs.py:81-94`; `optimize_fasta` `:133-141`; `strict=False` `:60` after xavier `potts_mpnn_utils.py:1263-1265` |
| Upstream LASErMPNN | polizzilab/LASErMPNN @ `e70f2c6d765416f7e29d51bfd6d4e08496438878` (MIT); `utils/model.py:107`; alphabet `utils/constants.py:34`; `tied_sample :460-627` (λ mix `:626`, T None→1e-6 `:617-618`, disabled mask `:606`); `sample :727-941` (carry `:773`, prior visibility `:815-818,916-917`, step `:833-865`, χ GVP `:931-934`); rotamers post-sampling `run_inference.py:747-750`; argparse `:773-792`; `sample_model` `:511-517` |
| redsox | `checkers/superset.py:27-60` (`alias_map: dict[str,str]` `:31,43`); CLI `target_dataclasses[0]` only `cli/main.py:120-121,161-165,205-207`; U1 presence-only `reachability.py:18-40` |
| xtrax port contract | `xtrax/port/port_target.toml`, `port/tests/conftest.py` (`:64-71` target, `:103-114` manifest, `:159-171` oracle import, `:214,232` timeout, `:277-286` emit), `test_parity_safe_map.py:53` (global x64); aminx pins wheel `pyproject.toml:26` → `port/` not importable |
| aminx parity/bathos | `parity_targeted` tests `tests/parity/test_full_model_parity.py:273`, `tests/parity/test_packer_parity.py:120`, `tests/model/test_ligandmpnn_equivalence.py:348` (none via runner); `parity_heavy` deselected `pyproject.toml:229`; `.bth.toml` slug `aminx` |

## 2. Architecture

### 2.1 FamilyDriver seam

```python
# src/aminx/host/family_driver.py
class FamilyDriver(Protocol):
  name: str
  options_type: type
  mpnn_fallback_purposes: frozenset[str]   # e.g. {"jacobian","inspect","score:nll","score:logits"}
  def handles(self, spec, purpose: str) -> bool
  def load(self, spec) -> eqx.Module                        # never applies inference_mode
  def mpnn_core(self, model) -> ModelProtocol | None        # Potts: model.mpnn ; LASEr: None
  def batches(self, spec) -> Iterator[FamilyBatch]
  def axes(self, spec, purpose, batch) -> list[AxisSpec]
  def stages(self, spec, purpose, model) -> FamilyStages
  def result_schema(self, spec, purpose) -> Mapping[str, SinkArraySpec]   # depends on purpose + output_kind
class FamilyBatch(NamedTuple):
  input_indices: tuple[int, ...]           # positions in spec.inputs; strictly increasing across the stream
  arrays: PyTree                           # fixed-shape padded device inputs
  skipped: tuple[tuple[int, str], ...]     # (input_index, reason)
class SinkArraySpec(NamedTuple):
  dims: tuple[str, ...]; dtype: str; attrs: Mapping[str, str | int | float]
FAMILY_DRIVERS: Registry[FamilyDriver]     # aminx.registry.Registry
```

**Dispatch** (first statement after spec sync in `runner.sample/score/jacobian/inspect`; purpose key
is `"score:<output_kind>"` for score):
```python
if (d := FAMILY_DRIVERS.get(spec.model_family)) is not None:
  if d.handles(spec, purpose): return run_family_driver(d, spec, purpose)
  if purpose not in d.mpnn_fallback_purposes: raise ValueError(f"{d.name} does not support {purpose} in v1")
  # fallback: prep loads d.mpnn_core(d.load(spec)); None -> ValueError; MPNN code path otherwise unchanged
```
`PottsMPNNDriver`: `handles` True for `sample` (every configuration) and `score:energy`,
`score:ddg`; `mpnn_fallback_purposes={"jacobian","inspect","score:nll","score:logits"}` run on the
embedded stock `Aminx`. `LaserDriver`: `handles` True for `sample`, `score:{nll,logits,
proofread_unconditional,proofread_conditional}`; `mpnn_fallback_purposes=∅`. Setting
`decoding_order_fn` with a driver family → `ValueError`.

**Output channel (v1: no `io_callback` in drivers).** Each jitted dispatch over one `FamilyBatch` ×
one `samples` chunk **returns** `dict[str, Array]` whose keys/dims equal `result_schema(spec,
purpose)` (mismatch → `RuntimeError`). `run_family_driver` then: (i) if `output_h5_path` set →
on one `sink = ZarrStagingSink(sink_spec_for(spec, spec.run_spec.io.output_h5_path))` (T0.0; same
root as `streaming.py:79`, not `RunSpecification.output_dir`):
`sink.stage((f"structure_{input_index}", str(chunk_start)), **chunk)`; per structure once
`sink.stage((f"structure_{input_index}",), attrs={"structure_index": input_index, "structure_id":
structure_id})`; root attrs via `sink.stage((), attrs={…})` (as `streaming.py:104`):
`schema_version="<family>_v1"`, `model_family`, `purpose`, `output_kind`,
`alphabet="ACDEFGHIKLMNPQRSTVWYX"`, `skipped_inputs`; `finalize()` at end; else (ii) concatenates
chunks into the result dict. Memory bound = one batch × one chunk (as `host/streaming.py:62-69`).
MPNN `StreamingTensorStagingSink`/`take_staging_sequences_logits`/`_structure_ids_for_batch` unused
by drivers and unchanged.

**Ids/skips.** `structure_id = _canonical_structure_ids_for_spec(spec)[input_index]`; result
carries `skipped_inputs: list[{input_index, structure_id, reason}]`. Test: 3 inputs, middle
unparsable → ids + order asserted. Test: inputs `a/model.pdb`, `b/model.pdb` → two distinct groups
whose `structure_id` attrs are both `model`.

**Inference mode/dropout.** `run_family_driver` never calls `prep_protein_stream_and_model`. Stages
take static `inference: bool` + explicit `dropout_key`; True for all purposes except LASEr
`proofread_conditional` (built on `eqx.nn.inference_mode(model, value=False)`, keys
`fold_in(fold_in(key, order_idx), dropout_idx)`, or injected masks in parity tiers §7.1).

**Reused as-is:** `_canonical_structure_ids_for_spec`, `resolve_target_samples`,
`make_axis_dispatch_via_xtrax`, xtrax `ZarrStagingSink`, `metadata` key set of `sample()` (T0.2
confirms names/signatures; any extraction refactor keeps MPNN outputs byte-identical, gated by T0.5a).

### 2.2 Family plumbing

- `run/specs.py:236` Literal adds `"pottsmpnn"`, `"lasermpnn"`; `__post_init__` derives them from
  `checkpoint_id` prefixes `pottsmpnn_`/`lasermpnn_` before the `model_type` derivation; explicit
  family never overridden.
- Each `model_family` consumer (§1; T0.2 re-greps): `_prepare_ligand_context` never reached for
  driver families (test); checkpoint registry gains `pottsmpnn`/`lasermpnn` sections; portable JSON
  v2 raises `ValueError`; multistate PoE / grid lineage / campaign / streaming raise `ValueError` for
  driver families unless T0.2 shows them family-agnostic. Fallback purposes see `model_family=
  "pottsmpnn"` but an `Aminx` model: consumers reached on the fallback path (T0.2 lists) treat it
  as `proteinmpnn`.
- Unsupported `(family, purpose, output_kind)` → `ValueError` at `__post_init__`.

### 2.3 Spec/CLI surface

- `ScoringSpecification.output_kind: Literal["nll","logits","energy","ddg","proofread_unconditional",
  "proofread_conditional"] = "nll"`; `sequences_to_score` required only for `nll`/`logits`.
- `RunSpecification.potts_mpnn: PottsMPNNOptions | None`, `laser: LaserOptions | None`
  (`None` → family defaults); `build_run_spec` threads to `run_spec.potts_mpnn`/`run_spec.laser`;
  options for a non-matching family → `ValueError`.
- `SamplingSpecification.omit_aa: Sequence[str] = ()`, `omit_aa_per_position: Mapping[int,str] |
  None = None`, threaded into `run_spec.sampling`; added to the RS-6b ast-grep sampling-field list
  with a planted-violation case. Compiled host-side to bias `-1e8` at designed positions (upstream
  constant).
- **Temperature family default (U-c).** `SamplingSpecification.temperature: Sequence[float|None] |
  float | None | Unset = UNSET` (`UNSET` = module singleton in `run/specs.py`, `__reduce__` returns
  itself). **Resolution point:** base `RunSpecification.__post_init__` calls new hook
  `self._resolve_family_defaults()` immediately after the `model_family` derivation block and before
  `self._sync_run_spec()` (currently `specs.py:519-520`); base hook no-op; `SamplingSpecification`
  overrides: `UNSET` → `(0.1,)` for proteinmpnn/ligandmpnn/membrane/pottsmpnn, `(None,)` for
  lasermpnn; scalar/`None` → 1-tuple; sequence → tuple; any `None` with non-lasermpnn →
  `ValueError`. The existing float→tuple at `specs.py:633-634` moves into the hook.
  `ScoringSpecification.temperature: float = 1.0` untouched. **Types:** `SamplingConfig.temperature:
  tuple[float|None, ...]`; new `_as_temperature_tuple` keeps `None` (`_as_float_tuple` unchanged, used
  for noise). MPNN consumers (`kernel_dispatch.py:134`, `_sampling_grid_lineage.py:98,115`,
  `campaign.py:84`, `multistate_poe.py:378,560`) read via `mpnn_temperatures(run_spec) ->
  tuple[float,...]` which raises on `None`. **replace():** post-init value is concrete →
  `dataclasses.replace` treats it as explicit; `replace(s, model_family=…, temperature=UNSET)`
  re-resolves. **JSON:** `(None,)` ↔ `[null]`; `_coerce_field_value` temperature branch accepts
  `int|float|None` elements. **CLI:** `--temperature` default unset → kwarg omitted. In a
  `temperatures` axis `None` is a static per-element branch (argmax, no min-p). Tests: UNSET never
  reaches `build_run_spec`; per-family defaults; None+MPNN ValueError; both replace cases; `[null]`
  round-trip; T0.5a goldens exact.
- CLI (T0.6): `run score --output-kind`; `_RunBase` `--potts-options-json`, `--laser-options-json`
  via `_base_spec_kwargs`. **spec_json:** `_OPTIONS_FIELDS = {"potts_mpnn": PottsMPNNOptions,
  "laser": LaserOptions}`; encode = `{f.name: _to_json_value(v)}` per field; decode = dict →
  `cls(**…)` (lists → tuples where the field default is a tuple; unknown keys →
  `SpecJSONDecodeError`). The partition probe (`spec_partition.py:128-135`) cannot carry Options, so
  dedicated round-trip tests with `pottsmpnn_*`/`lasermpnn_*` ids cover spec_json and campaign.

### 2.4 L-DRV lint (`tests/lint/test_l_drv.py`, stdlib `ast`)

Scope: S_F = `src/aminx/families/**`, S_M = `src/aminx/model/{potts_mpnn,laser}/**`,
S_H = `src/aminx/host/family_*.py`.
- **R1** in S_F ∪ S_H: no call to `jax.vmap`/`vmap`/`eqx.filter_vmap`/`jax.pmap`/`shard_map` (named
  axes go through `BatchPlanner`/`make_axis_dispatch_via_xtrax`). Intra-layer neighbour/atom vmaps
  allowed in S_M only.
- **R2** in S_F ∪ S_M: no `For`/`While`/comprehension inside (a) `__call__` of any `eqx.Module`
  subclass or (b) any function decorated with, or passed as first argument to, `jax.jit`/
  `eqx.filter_jit`/`lax.scan`/`lax.while_loop`/`lax.fori_loop`/`lax.cond`; except iterables
  `self.<name>`, `enumerate(self.<name>)`, `zip(self.<name>, <any>)`, `range(<int literal ≤ 4>)`,
  `range(len(self.<name>))`.
- **R3** `lax.scan`/`while_loop`/`fori_loop` in S_F ∪ S_M only at `(path, enclosing qualname)` in
  `tests/lint/l_drv_allowlist.toml` with `reason="sequential_dependency"`; expected entries:
  `PottsARDecode` (steps, tied groups, tied members), `PottsRefine` (`potts` sweep,
  `potts_converge` while, `nodes` sweep), `LaserJointDecode` (steps). Stale entries fail.
- **R4** inside R2 bodies: no `.item()`, `.tolist()`, `np.asarray`, `jax.device_get`;
  `int()`/`float()`/`len()` only on literals, `<expr>.shape[<int literal>]`/`.ndim`/`.size`, or
  `self.<name>` declared `eqx.field(static=True)` or as a tuple of modules in the class body.
- **R5** no import of `aminx.potts` in S_F ∪ S_M ∪ S_H.
One planted-violation test per rule (temp file in scope, as `test_rs6b_flat_field_gate.py:54-80`)
+ clean-tree test + allowed-pattern fixture (`enumerate(self.layers)`, `zip(self.layers, keys)`,
`int(x.shape[0])`, `len(self.layers)`) that must pass.

## 3. ADR amendment

New `decisions/260929_pottsmpnn-lasermpnn-family-drivers.md` (`amends: 260605`):
- 260605 governs the TRW structure model (`aminx.potts.*`); unchanged, lint unchanged.
- PottsMPNN sampling and energy purposes run in a FamilyDriver whose decode stage reuses aminx
  `DecoderLayer` modules; its nll/logits scoring and jacobian/inspect run the unmodified MPNN paths
  on the embedded `Aminx` core. 260605 Option III (EncoderOutput widening) stays rejected.
- LASErMPNN is a FamilyDriver with its own encoder-output/result types; MPNN contracts not widened.
- New modules: `aminx.model.potts_mpnn`, `aminx.families.potts_mpnn`, `aminx.model.laser`,
  `aminx.families.laser`, `aminx.host.family_driver`, `aminx.host.family_runner`; L-DRV R5.
- `potts/model.py` docstring: one-line disambiguation pointer.

## 4. PottsMPNN

### 4.1 Model and weights

`PottsMPNN(eqx.Module)` = **composition**: `mpnn: Aminx` (stock aminx ProteinMPNN, shared
weights converted) + `potts_head: PottsHead`. `ModelProtocol` (incl. `__call__`, `stage_schema`)
satisfied by `mpnn` by construction; no subclassing, no capability field changes.

- Conversion `scripts/recapture/pottsmpnn_model_to_eqx.py` (bathos-tracked) reuses
  `scripts/convert_weights.py`'s ProteinMPNN key map; only `etab_out.*` is new. T0.2 records
  `load_state_dict(strict=False)` missing/unexpected keys per checkpoint; non-empty missing →
  blocking finding. Registered with source SHA-256 + upstream commit.
- In scope: `vanilla/pottsmpnn_{20,30}`, `soluble/sol_pottsmpnn_{20,30}`, `ft/potts_ft` (T0.2
  confirms ft config); `proteinmpnn_compatible_model_weights/` → `duplicate` exclusion if
  byte-equivalent after mapping (T0.2), else in scope.
- `mpnn_ext/external/aminx/weights/pottsmpnn/*.eqx.zst` not reused (TRW-model artefacts).
- Self-edge invariant test: `neighbor_indices[...,0]==arange(L)` on present rows.

### 4.1a Host featurizer (A0)

PottsMPNN driver purposes do not use proxide (`prep.py:102-122`: no upstream gap rows,
`residue_idx`, or chain order). `aminx.families.potts_mpnn.featurize` (host numpy, called from
`batches()`) ports:
- `parse_pdb_upstream(path, chains, skip_gaps)` = `parse_PDB`/`parse_PDB_biounits`
  (`potts_mpnn_utils.py:73-205`): ATOM + HETATM MSE→MET; first occurrence per atom per
  (resnum, icode) wins (altloc, multi-MODEL); rows `min_resn..max_resn`, one per icode, sorted;
  unknown names → `-`; missing resnum → gap row (`-`, N/CA/C/O NaN) unless `skip_gaps`
  (`:143-156`); absent chains dropped; non-`.pdb` inputs → `skipped_inputs` reason
  `pottsmpnn_requires_pdb`.
- `tied_featurize_port` (`:293-512`): chain order = sorted designed chains + sorted fixed chains;
  `-`→`X`; `S_true` (model alphabet), `chain_M`, `chain_M_pos` (per-chain 1-based p → row
  `global_idx_start[chain]+p−1`, `:415`), `chain_encoding`, `residue_idx = 100(c−1)+row`
  (`:373,408`), omit/pssm/`bias_by_res` arrays, tied groups + `tied_beta` (`:445-460`);
  `present = isfinite(sum X over atoms,xyz)`, then `X[isnan]=0` (`:510-512`).
Output length `L_total` (gap rows included), padded to `L_pad` with
`pad_valid = arange(L_pad) < L_total`; pad rows X=0, S=X, present=0.

| Site | Mask |
|---|---|
| kNN (`mpnn.features(mask=present)`) | invalid pairs → +inf; `lax.top_k` lower-index-first tie-break ⇒ gap rows before pad rows; `K_static=min(48,L_pad)`; first `min(48,L_total)` slots of real rows = upstream set (`:1143-1150`) on non-tie rows (upstream invalid pairs get row D_max, `:1148`, never selected when #present > K). Structures with #present ≤ K_eff < L_total are `knn_boundary_tie` (host-detected): upstream topk breaks the D_max tie arbitrarily, so no fill reproduces it; aminx keeps +inf; stock `ProteinFeatures` untouched |
| encoder `mask_V`/`mask_attend` | `present` (`:1789-1792`) |
| `PottsHead` | `present[i]·pad_valid[E_idx[i,k]]`, then slot-0×eye (`:1795-1797`) |
| `merge_pair` | reverse edge exists ∧ `pad_valid` both ends; no valid reverse → undivided (`etab_utils.py:176-183`); `exclude_self` ≡ slot `k==0` |
| `potts_energy`, `positional_potts_energy` | `pad_valid` both ends (gap rows contribute 0 via row mask + zero `-`/`X` slots, as upstream) |
| row kernel `attention_mask` (AR + `nodes`) | `nbr_valid[i,k] = pad_valid[E_idx[i,k]]` |
| AR `mask_bw/mask_fw` (decoder, tied_decoder), `S_true` branch | `present` only (raw mask, `:1426-1428`, `:1620-1622`, `:1448`); `chain_M_pos` enters AR only via `cm` (order key + fixed/sampled select, `:1419`/`:1603`) |
| `nodes`/refine `mask_bw/mask_fw`, refine skip | `present·chain_M_pos` (`sample_seqs.py:346,355`) |

**Index space.** All driver per-position inputs (`fixed_positions`, `fixed_tokens`,
`tied_positions`, `bias` rows, `omit_aa_per_position`) are 0-based over A0 rows; `mutant_csv` `pos`
is 0-based into the gap-filled chain (`run_utils.py:737`). Fallback purposes keep proxide; their
fixtures are gap-free.

**Gate (A0).** Field-level equality vs upstream `parse_PDB`+`tied_featurize` on: 2 example_pdbs;
a numbering-gap chain with `L_total ≤ 48`; a gapped chain with `L_total > 48` and #present < 48; an
insertion-code chain; a partial-backbone residue; each with `skip_gaps ∈ {False, True}`. `E_idx`
compared as per-row sets. Gap rows (all-tie): record §6.5b `gap_row_knn_tiebreak` (only
observable: `potts_etab` on edges incident to gap rows; excluded, listed by index).
`knn_boundary_tie` structures (#present ≤ K_eff < L_total): A0 compares only the deterministic
subset `{j : d_ij < D_max_i} ⊆ E_idx[i]` on present rows; §6.5b `knn_boundary_tie` scope = all
outputs of that structure; runtime `logger.warning`; excluded from exact waves and sidecars. Exact
tiers use the gapped `L_total ≤ 48` fixture.
`test_knob_semantics_skip_gaps` on the gapped fixture.

### 4.2 Potts head, etab conventions, padding

- `PottsHead(edge_features (L,K,H), E_idx, present, pad_valid) -> etab_raw (L,K,20,20)`: linear →
  `present[i]·pad_valid[E_idx[i,k]]` mask → slot-0 × eye(20) (upstream masks rows only,
  `potts_mpnn_utils.py:1284,1795`).
- `merge_pair(etab, E_idx, pad_valid, *, denom, exclude_self)`: reverse-edge gather; an entry merges
  iff its reverse edge exists and both ends are `pad_valid`, else stays undivided
  (`etab_utils.py:176-183`); `exclude_self` ≡ slot `k==0`; ports `merge_duplicate_pairE` for both
  call sites.
- `etab_forward = merge_pair(etab_raw, denom=2, exclude_self=False)` — consumed by `PottsRefine`,
  emitted by `potts_etab` sink (`etab_convention="forward_denom2"`).
- `etab_energy = merge_pair(etab_forward, denom=4, exclude_self=True)` — consumed by
  `potts_energy`/ddG/`PottsSampleEnergy`.
- `potts_energy(etab_energy, E_idx, pad_valid, seq_etab_alphabet)` ports `calc_eners` (no ×0.5).
  Sequences are encoded in the etab_utils alphabet (`-`=20, `X`=21, both zero-padded slots —
  replicated, §6.5); explicit model↔etab alphabet conversion with round-trip test.
- **Padding/K convention.** Parity convention = upstream **unpadded over A0 rows**
  (K=min(48,L_total), gap rows included as real nodes). Masks per the §4.1a table (`present` vs
  `pad_valid`; the r3 `edge_valid` rule is retired — upstream applies no node mask on edges at
  etab/merge/energy/decoder). Tests: `test_energy_invariant_to_padding` (`L_pad∈{L_total,128,512}`,
  f64 exact on `pad_valid` edges); `test_valid_neighbour_set_matches_upstream`
  (`L_total∈{30,48,49}`); oracle fixtures include an L=30 chain and a binding partition with
  `L_p<48`. (d) In the PottsARDecode/`nodes` row kernel, `nbr_valid[t:t+1]` is passed as
  `DecoderLayer(attention_mask=…)` (`decoder.py:333-336`) — masking **messages** from pad rows;
  zeroing context alone is insufficient since `message_mlp([h_i,0]) ≠ 0`. Test
  `test_decode_invariant_to_padding`: `L_total∈{30,49}`, `L_pad∈{L_total,128}`, f64, injected
  order/uniforms; tokens exact, `pad_valid` rows of `h_V_stack` rtol 1e-12. **Fallback (stock MPNN)
  unchanged:** `score:nll`/`score:logits` fallback parity asserted only on unpadded, gap-free
  single-input runs (`max_length=L_total`); padded `L_total<48<L_pad` divergence recorded in §6.5b.

### 4.3 Purposes and stages

| purpose / output_kind | Path | Stages / axes |
|---|---|---|
| `sample` (any mode/PSSM/tied/bias_by_res) | driver | `MPNNEncode → PottsARDecode → PottsSampleEnergy → [PottsRefine iff mode≠none] → sinks`; host ranking after last chunk; axes `samples`, `temperatures` |
| `score:energy` | driver | `MPNNEncode → PottsHead → etab_energy → potts_energy`; axis `candidates` = `sequences_to_score` if non-empty, else the `score:ddg` resolution (mutant_fasta / mutant_csv / DMS), always plus WT; absolute energies; partitions never evaluated (upstream `ddG=False`). Each `sequences_to_score` entry must have length L_total in A0 row order (sorted designed chains then sorted fixed chains, `tied_featurize :327`) with gap rows as `-`/`X`, else `ValueError` naming expected length + chain order; test `test_score_energy_gapped_alignment` |
| `score:ddg` | driver | as energy; candidates = mutant_fasta / mutant_csv else single-mutant DMS (respecting `exclude_chains`); `ddg=E(mut)−E(wt)`; binding: axis `partition` (§4.4); `mean_norm` per PDB after ddG |
| `score:nll`, `score:logits`, `jacobian`, `inspect` | MPNN fallback on `model.mpnn` | unchanged |

- **`PottsARDecode(eqx.Module)`**: fields `layers: tuple[DecoderLayer,...]`, `w_s_embed`, `w_out`
  (references to `mpnn`'s submodules). Ports upstream `decoder` (`:1415-1488`) and `tied_decoder`
  (`:1599-`) as incremental per-position decode: `lax.scan` over order, carry `(S (L,), h_S (L,H),
  h_V_stack (n_dec+1,L,H))`; step t gathers row t, applies each layer on the `(1,K,·)` slice with
  `mask_bw/mask_fw` from the order exactly as `:1422-1428`, scatters row t; O(n_dec·K)/step.
  **Row kernel (normative).** Once per structure, outside the scan: `rank_flat` (Tied, below;
  equals `argsort(order)` when every group is a singleton); `mask_attend[i,k] =
  rank_flat[E_idx[i,k]] < rank_flat[i]` (= upstream `order_mask_backward` gathered by `E_idx`,
  `:1423-1426`/`:1616-1619`, O(L·K)); with `m = present` (AR; upstream raw mask, `:1426-1428`):
  `mask_bw = m[:,None]·mask_attend`; `mask_fw = m[:,None]·(1−mask_attend)`; `h_EXV_fw = mask_fw[...,None]·concatenate_neighbor_nodes(
  h_V, concatenate_neighbor_nodes(zeros_like(h_V), h_E, E_idx), E_idx)`. Step t:
  `h_ES_t = concatenate_neighbor_nodes(h_S, h_E[t:t+1], E_idx[t:t+1])`; per layer l:
  `ctx = mask_bw[t:t+1,:,None]·concatenate_neighbor_nodes(h_V_stack[l], h_ES_t, E_idx[t:t+1]) +
  h_EXV_fw[t:t+1]`; `h_V_stack[l+1] = h_V_stack[l+1].at[t].set(layer(h_V_stack[l][t:t+1], ctx,
  present[t:t+1], attention_mask=nbr_valid[t:t+1], inference=True)[0])` (mirrors `:1455-1462`; aminx
  `concatenate_neighbor_nodes` = upstream `cat_neighbors_nodes` [edge, neighbour] order).
  **Precision prerequisite (T0.5):** `DecoderLayer.__call__` accumulates in
  `jnp.promote_types(message.dtype, jnp.float32)` instead of hardcoded float32
  (`decoder.py:338-341`) — byte-identical for f32/bf16/f16 (T0.5a goldens); test `DecoderLayer`
  against sealed f64 fixture `declayer_f64.npz` (weights, `h_V`, `h_E`, `mask_V`, torch `DecLayer`
  output) produced by `dump_potts_oracles.py` (A1; SHA-256 in `oracle_manifest.toml`), rtol 1e-12,
  no torch in the aminx env; T0.2 greps `src/aminx/model/**` for other
  hardcoded float32 on encoder/decoder paths.
  Order `argsort((chain_mask·chain_M_pos·mask + 1e-4)·|randn|)` (`:1419-1421`),
  `randn = jax.random.normal(key_order,(L,))`; injected `decoding_order` has length L_total;
  device order = `concatenate([injected, arange(L_total, L_pad)])`. Untied oracle: passed to
  `decoder(decoding_order=)`. Tied oracle: inject `randn`, then pass the oracle's returned flattened
  `decoding_order` to aminx (regroup-invariant; test asserts `rank_flat` equality). Masked rows
  take `S_true` (`:1448`); fixed rows `S_t·cm + S_true·(1−cm)` (`:1483`). Per-step probabilities via
  PSSMMix (below).
  **Tied (also covers untied: all singletons, `M_max=1`).** Host builds `tie_groups: int32 (L_pad,
  M_max)`, −1-padded (shape depends only on static buckets `L_pad` and `M_max` = next power of 2 ≥
  largest group, min 1): rows = `tied_positions` groups in listing order (members in listing order),
  then one singleton `[i]` for each row `i < L_total` not in any group, ascending, **including
  `present==0` rows**, then all-−1 rows. Host asserts groups disjoint with union `range(L_total)`;
  overlapping groups → `ValueError` (§6.5b). On device:
  ```python
  key   = jnp.where(pad_valid, (chain_mask*cmp*present + 1e-4)*jnp.abs(randn), jnp.inf)  # pad rows last
  order = decoding_order if injected else jnp.argsort(key)
  rank  = jnp.argsort(order)
  BIG   = jnp.iinfo(jnp.int32).max
  vm    = tie_groups >= 0
  safe  = jnp.where(vm, tie_groups, 0)
  group_key   = jnp.where(vm, rank[safe], BIG).min(axis=1)             # (L_pad,)
  group_order = jnp.argsort(group_key, stable=True)                    # empty rows last
  size   = vm.sum(axis=1).astype(jnp.int32)
  s_sorted = size[group_order]
  offset = jnp.zeros_like(size).at[group_order].set(jnp.cumsum(s_sorted) - s_sorted)
  m_idx  = jnp.cumsum(vm, axis=1) - 1                                  # valid-member index, listing order
  rank_flat = (jnp.full((L_pad,), BIG, jnp.int32)
               .at[jnp.where(vm, tie_groups, L_pad)].set(offset[:, None] + m_idx, mode="drop"))
  rank_flat = jnp.where(pad_valid, rank_flat, L_pad + jnp.arange(L_pad))
  ```
  `rank_flat` = inverse permutation of upstream flattened `new_decoding_order` (`:1606-1614`) and
  drives `mask_attend`/`mask_bw`/`mask_fw`/`h_EXV_fw`. Outer `lax.scan` over `g_step ∈ range(L_pad)`,
  `g = group_order[g_step]`, `lax.cond(size[g]==0, skip, body)`; inner scan over `m ∈ range(M_max)`
  skipping `tie_groups[g,m] < 0`; member m sees rows written by earlier members; group `h_S` = 0
  until sampled; masked singletons and groups with any `present==0` member take the `S_true` branch
  (§6.5); logits `Σ_m tied_beta[t_m]·(W_out(h_{t_m})/T)/size[g]`; `bias_by_res`/pssm/omit **and the
  fixed/sampled select** are read at `t_last = tie_groups[g, size[g]−1]`:
  `tok = cm[t_last]·draw + (1−cm[t_last])·S_true[t_last]` (`cm = chain_mask·chain_M_pos·present`),
  written to all members (§6.5). Tests `test_tied_rank_flat_matches_upstream`: `rank_flat[:L_total]
  == argsort(oracle decoding_order)` and `mask_attend == gather(oracle order_mask_backward, E_idx)`
  for 3 seeds, fixture with a group whose listing order ≠ raw-rank order and a non-member whose raw
  rank lies between two members; negative control raw `rank` must fail; overlap `ValueError` test.
- **PSSMMix** (inside PottsARDecode step, upstream order; T0.2 transcribes `:1391-1404`/decoder
  equivalents verbatim into the docstring): `p = softmax(logits/T − omit·1e8 + bias/T +
  bias_by_res/T)`; if `(pssm_bias_flag and coef.numel()>0) or pssm_bias.numel()>0`:
  `p = (1−coef·multi)·p + coef·multi·pssm_bias`; if `pssm_log_odds_flag`: `p=p·(mask+0.001)`,
  renormalise; `omit_AA_mask` renormalise; draw (§7.1 sampling rule).
- **PottsRefine** (bit-parity port): `potts` (one sweep), `potts_converge` (`lax.while_loop`, stop
  test literally `(ener_delta != 0) & (iters < max_iters)`, `max_iters=1000`), `nodes`. Temperature
  floor per upstream (`sample_seqs.py:38-39`) then draw (never argmin). Refine order per upstream
  keying quirk (§6.5). Binding: at interface positions `predicted_E=(E_c−E_c[cur])−(E_p−E_p[cur])`
  (`:148`); `ener_delta += predicted_E[chosen]` (`:176`); under `both` non-interface positions add
  absolute energies; `only` skips non-interface (`:135-136`); "≈1000 sweeps" holds only for
  `binding_energy_optimization="none"`. **`nodes`** = one sequential Gibbs sweep: init `S=seq`,
  `h_S=W_s(S)` everywhere, `h_V_stack[0]=h_V`, `h_V_stack[1..]=0`, `mask_bw=1` except self slot
  (`:188-190`); per visited position (skip `mask==0`/`chain_mask==0`): update row t layer by layer
  reading `h_V_stack[l]` (unvisited rows 0 for l≥1, `:183,221-241`), `W_out/T_opt` → PSSMMix → draw
  → write `h_S[t]`, `S[t]` (`:263-268`); reuses the row kernel with `mask_attend := (k≠0)`, i.e.
  `mask_bw = m[:,None]·(k≠0)`, `mask_fw = m[:,None]·(k==0)`, `m = present·chain_M_pos`
  (`sample_seqs.py:346`, `run_utils.py:187-191`); `h_EXV_fw` is **recomputed** from this `mask_fw`
  (self slot only; never the AR-order `h_EXV_fw`); `h_S` at t not masked (`h_S_masked` is a plain
  clone, `:203`); `attention_mask = nbr_valid`. `test_knob_semantics_nodes_gibbs` negative control:
  AR-order `h_EXV_fw` must fail.
  Tied refine (`tied_optimize_sequence`, `tied_epistasis`) transcribed in T0.2. Tests
  `test_knob_semantics_nodes_gibbs`, `test_knob_semantics_binding_converge_stop`.
- **PottsSampleEnergy + host ranking.** Device: `sample_energy[n] = potts_energy(etab_energy,
  S_AR[n])` on the **pre-refine** AR sequence (`sample_seqs.py:259`); every sample refined
  independently (`:321-366`). After a structure's last `samples` chunk, `run_family_driver` computes
  `sample_rank = argsort(all sample_energy, stable=True)` over all N (= Python `sorted`),
  `best = sample_rank[0]`. PDB (iff `write_pdb`): `sequence[best]` if mode none, else
  `refined_sequence[best]` (not re-ranked, `:368-373`). Sampled FASTA = `sequence` in rank order named
  `<pdb><suffix>_<sidx>` (no `_sidx` when N=1) (`:268-272,392-396`); optimized FASTA =
  `refined_sequence` in the same order (`:378-382`). `optimize_fasta` path: only `refined_sequence`;
  PDB only when N=1 (`:368`). Test `knob_semantics_rank_before_refine`: seed where AR-energy argmin ≠
  refined-energy argmin; PDB follows AR energy and matches oracle.

### 4.4 Binding partitions

Each partition's chains re-featurised by A0 (own L, kNN, encode, head; `get_etab :816-833`). Axis
`partition` (ragged L → `AxisSpec(bucket_boundaries=…)`; §4.2 padding convention makes energies
bucket-invariant); `AxisBoundary` Fuse on scalar energies `E_bind = E_complex − Σ_p E_p`; partitions
scored only when `partition_flag` (`energy_prediction.py:57-75`).

### 4.5 Result schema (`result_schema(spec, purpose)`)

`sample`: `sequence (N,L) i32` (AR, pre-refine), `sample_energy (N,) f32` (of `sequence`),
`refined_sequence (N,L) i32` iff mode≠none, `sample_rank (N,) i32` (structure-level:
`attrs={"level":"structure"}`, staged once via `sink.stage((f"structure_{i}",), sample_rank=…)`,
excluded from the per-chunk key check), opt-in `potts_etab
(L,K,20,20)`, `potts_E_idx`, `refine_energy_trace`. `score:energy`: `energy (N_cand,) f32`,
`candidate_ids`. `score:ddg`: `ddg (N_mut,) f32`, `mutant_ids`, `ddg_expt (N_mut,) f32|nan`.
`emit_dense_hJ` converts host-side only.

## 5. LASErMPNN

### 5.1 Types

```python
class LaserEncoderOutput(eqx.Module):
  prot_scalars: Float[Array,"L Hs"];  prot_vectors: Float[Array,"L V 3"]
  lig_scalars:  Float[Array,"A Hl"];  lig_vectors:  Float[Array,"A Vl 3"]
  pr_pr_eattr:  Float[Array,"L K E"]; pr_pr_idx: Int[Array,"L K"]
  lig_pr_eattr: Float[Array,"L Kl E"]; lig_pr_idx: Int[Array,"L Kl"]
  prot_mask, lig_mask, pr_pr_mask, lig_pr_mask
class LaserSampleResult(eqx.Module):
  sequence: Int[Array,"L"]; seq_logits: Float[Array,"L 21"]
  chi_deg: Float[Array,"L 4"]; chi_logits: Float[Array,"L 4 Nbins"]; chi_mask: Bool[Array,"L 4"]
```
Dims from checkpoint `model_params` (T0.2).

### 5.1a Alphabet boundary

`aminx.families.laser.alphabet.LASER_ALPHABET = "ARNDCEQGHILKMFPSTWYVX"` (`utils/constants.py:34`;
T0.2 verifies exact string) with `to_laser`/`to_mpnn` int32 permutations. Converted exactly once:
inbound in `batches()` (`sequences_to_score`, `fixed_tokens`, `bias`/`omit_aa*` columns,
`disabled_residues`, budget/charged letters by letter); outbound in result assembly (`sequence`,
`seq_logits`, `seq_log_prob`, `proofread_mean/std` columns). Sinks/results in aminx order; stages in
LASEr order. Oracle outputs mapped to aminx order before comparison. Tests
`test_laser_alphabet_roundtrip`, `test_knob_semantics_laser_bias_single_aa`.

### 5.2 Model port

| Upstream | aminx | Notes |
|---|---|---|
| `LigandFeaturizer`, `LigandEncoderModule` (`model.py:1150,1312`) | `LaserLigandEncoder` | weights from main checkpoint (T0.2 confirms keys) |
| `HomoGATv2`, `HeteroGATv2`, `GVP`, `DenseGVP`, `EquivariantLayerNorm` | `aminx.model.laser.layers` | edge-list → dense padded neighbour axis + mask; `scatter_softmax` → masked softmax returning 0 on empty K |
| encoder/decoder layers | `LaserEncoder`, `LaserDecoderLayer` | |
| χ heads | `LaserChiHead` | bins from checkpoint `chi_angle_rbf_bin_width` |
| `chi_offset_prediction_layers[k]` (`model.py:902`) | `LaserChiOffsetHead` | input `cat[chi_logits_k, one_hot(bin_k, Nbins)]`, `chi_logits_k` post-`minp_warp` when χT set, raw under argmax (`:897`) |
| `RotamerBuilder` | `aminx.model.laser.rotamers` — post-decode only (PDB writer / `sidechain_coords` sink) | ideal geometry `files/*.pt` → hashed `.npz` |

Graphs host-side, fixed shapes (protein kNN on Cα `k_pp`, ligand kNN `k_ll`, ligand→protein kNN
`k_lp` + cutoff mask; from checkpoint `graph_structure`); ligand atoms padded to `ligand_atoms`
buckets. T0.2 enumerates every `radius_graph`/`scatter_*` site with its dense replacement.

### 5.3 Joint sequence+χ AR decode (`LaserJointDecode`)

`lax.scan` over decoding order (length L, masked); carry `(seq, seq_emb, chi_enc (L,4,R),
node_stack_s (n_dec+1,L,Hs), node_stack_v (n_dec+1,L,V,3), ala_count, gly_count)`. Step t
(`utils/model.py:833-865`):
1. decoder layers → `sequence_output_layer`;
2. disabled residues := `finfo.min` on rows `~chain_mask` (LASEr `chain_mask=1` = fixed), or on rows
   `chain_mask` under `ignore_chain_mask_zeros` (`:836`);
3. ALA/GLY over budget := `finfo.min` (`:840-841`);
4. `disable_charged_fs` K/R/D/E := `-inf` (`:843-848`);
5. if T is None: argmax; else `minp_warp_logits(logits, seq_min_p)` on **untempered** logits → `/T`
   (per-residue vector under fs temp) → softmax → draw;
6. fixed/sampled select (`:861-865`);
7. χ1..χ4 sequential (`:889-934`): `chi_logits_k = head_k(cat[s, seq_emb[t], chi_prev])`; argmax if
   χT None else `minp_warp(·, chi_min_p)` → `/chi_temp` → draw; `offset_k =
   LaserChiOffsetHead_k(cat[chi_logits_k, one_hot(bin_k)])`; `angle_k = remainder(bin_deg[bin_k] +
   offset_k + 180, 360) − 180` (`jnp.remainder` = torch floor-mod); unless `ignore_chain_mask_zeros`
   or `repack_all`: `angle_k = input_chi` where `chain_mask & ~isnan(input_chi)` (`:907-914`);
   `enc_k = RBF(angle_k)` appended to `chi_prev` for **every** k; `chi_enc[t,k]`, `chi_deg[t,k]`,
   `chi_logits[t,k]` written only where `chi_mask(aa_{X→G})[k]` (`:922-925`) — this `chi_enc` is what
   later residues see (`:816`); χ-GVP update after k ∈ {0,1,2} (`:931-934`).
Prior residues visible only via seq embedding + binned-χ RBF. Per-step O(n_dec·K) + 4 χ heads; no
rotamer build in-loop; jaxpr loop-body guard asserts no O(L²) op.

#### 5.3.1 Edge semantics (one test each)
1. Budget counts every decoded residue incl. fixed (`:782-788`); region from
   `budget_residue_selection` (ProDy host) or exposed-non-SS heuristic (B0).
2. Disabled `finfo.min` vs charged `-inf` (step 2/4).
3. min-p only when T set; T None → argmax.
4. Input χ kept only where fixed and non-NaN (`:907-914`).
5. `ignore_chain_mask_zeros` writes `X` at unsampled positions (`:936-937`).
6. `fs_sequence_temp` set → global seq temp 1e-6 (`run_inference.py:541-544`).
7. `ignore_ligand` → empty lig-prot K; masked softmax 0; equals upstream empty-ligand output.
8. `repack_only` ⇒ `chain_mask=1` + repack-all (`run_inference.py:729-740`).
9. Deterministic E2E: injected order + argmax seq/χ exact vs upstream.
10. `ignore_chain_mask_zeros` inverts the disabled-row set.

### 5.4 Purposes

| purpose | Stages / axes |
|---|---|
| `sample` | encode → `LaserJointDecode` → RotamerBuilder (post) → sinks; axes `samples`, `temperatures` |
| `sample` tied (`tied_second_input`) | encode both (equal L else `ValueError`), shared order, per-step `λ·P1+(1−λ)·P2` (§6.5); tied semantics: disabled mask on `~chain_mask` only (`:606`), no min-p, no charged mask, T None→1e-6 sampled (`:617-618`); setting `seq_min_p`/`chi_min_p`/`fs_sequence_temp`/`disable_charged_fs` with tied → `ValueError` (T0.2 verifies list) |
| `score:nll`, `score:logits` | teacher-forced seq+χ log-probs (`get_logits_for_score :261` port) — primary parity surface |
| `score:proofread_unconditional` | one forward, `return_unconditional_probabilities`, softmax; residues = first shell or `selection_string` |
| `score:proofread_conditional` | axes `focus_residue` × `decoding_order` (`n_decoding_orders`) × `dropout_seed` (`n_dropouts`); each cell = single-residue-designable sample at T=1, χT=1, dropout on, graph `num_adjacent_residues_to_drop`; Fuse mean over orders then dropouts + `std` |
| `jacobian`, `inspect` | `ValueError` (no fallback) |

### 5.5 Host featurizer (B0)

B-factors (`fix_from_bfactor` requires {0,1}), side-chain coords + χ (NaN → mask), ligand
elements/H/water (`use_water`)/ncAA-as-ligand, φ/ψ, SS/exposure, alpha-hull first shell. Proxide
(`prep.py:102-122`) lacks B-factors → B0 extends the proxide call or parses via ProDy in `batches()`.
Gate: field-level equality vs upstream `BatchData` on 5 fixtures (`example_pdbs/4jnj-1_prot.pdb` + 4
from `databases/`, pinned path+SHA-256).

### 5.6 Result schema

`sample`: `sequence`, `seq_log_prob`, `chi_deg (N,L,4)`, `chi_mask`, opt-in `sidechain_coords`;
PDB (hydrogens per checkpoint `build_hydrogens`); FASTA (`output_fasta`, `output_fasta_only`).
`score:nll/logits`: `seq_log_prob`, `chi_log_prob`, opt-in logits. `proofread_*`: `proofread_mean
(R,21)`, `proofread_std (R,21)` (conditional only), `residue_ids`.

## 6. Knob surface and redsox coverage

### 6.1 Options (flat dataclasses)

`PottsMPNNOptions`: `optimization_mode: Literal["none","potts","potts_converge","nodes"]="potts"`,
`optimization_temperature=0.0`, `binding_energy_optimization: Literal["none","both","only"]="none"`,
`binding_energy_json`, `binding_energy_cutoff=8.0`, `mean_norm=False`,
`filter_nan=False` (no-op quirk), `mutant_fasta`, `mutant_csv`, `exclude_chains`, `pssm_json`,
`pssm_threshold`, `pssm_multi`, `pssm_log_odds_flag`, `pssm_bias_flag`, `bias_by_res_json`,
`tied_beta`, `tied_epistasis`, `skip_gaps`, `optimize_pdb`, `optimize_fasta`, `write_pdb=True`,
`emit_etab=False`, `emit_dense_hJ=False`, `chain_design_mask_json`. Path-valued fields are
`str | None`. (Upstream `ddG` maps to `output_kind`, §6.3.)

`LaserOptions`: `fs_sequence_temp`, `chi_temp`, `seq_min_p`, `chi_min_p`,
`disabled_residues=("X","C")`, `disable_charged_fs`, `repack_only`, `fix_from_bfactor`,
`ignore_ligand`, `use_water`, `noncanonical_aa_ligand`, `fs_calc_ca_distance=10.0`,
`fs_calc_burial_hull_alpha_value=9.0`, `fs_no_calc_burial`, `ala_budget=4`, `gly_budget=0`,
`constrain_ala_gly_to_exposed_non_ss`, `budget_residue_selection`, `ignore_chain_mask_zeros`,
`tied_second_input`, `tied_interpolation_lambda=0.0`, `selection_string`, `n_decoding_orders=10`,
`n_dropouts=10`, `num_adjacent_residues_to_drop=6`, `strict_load=True`, `output_fasta`,
`output_fasta_only`. (Defaults transcribed by the extractor; generated alias table is authoritative.)

### 6.2 Reference surfaces (generated)

`scripts/redsox/extract_upstream_knobs.py` (bathos-tracked, pure AST): every upstream `*.py` with
argparse (glob) → dests; `sample_model`/entry-point parameters; PottsMPNN `cfg.inference.X`/
`cfg.model.X` reads and `'X' in cfg.inference` tests ∪ example-YAML keys. Emits
`tests/redsox/reference_surfaces.py`: one frozen dataclass per entry point, fields namespaced
`<entrypoint>__<dest>`, header with upstream SHA; regeneration diff test. Plus a manual dataclass
`pottsmpnn_input_list` (`pdb`, `designed_chains`, `fixed_chains`; line format
`pdb|designed:chains|fixed:chains`, `sample_seqs.py:81-94`), marked `# MANUAL`, skipped by the diff.

### 6.3 Alias table

`tests/redsox/alias_map.toml`, one row per reference field: `{ref, targets:[..], equivalence ∈
{identical, semantic, exclusion, divergence}, reason?, parity_test_ids:[..], note}`. Namespacing ⇒
no implicit name matches. Known non-identical rows: `*__chain_dict_json` and
`pottsmpnn_input_list__{designed,fixed}_chains` → `[fixed_mask, chain_design_mask_json]` semantic;
`fix_decoding_order`+`decoding_order_offset` → `[random_seed]` semantic (seeded-order test; order
source is the driver, not `decoding_order_fn`); Potts `noise` (eval `augment_eps`, all backbone
atoms) → `[noise]` semantic (knob test: iid Gaussian N/CA/C/O); LASEr `bb_noise` → `[noise]`;
`repack_all` → `[repack_only]` semantic; `disable_inference_dropout` → exclusion `internal`;
`laser_*__sequence_temp` → `[temperature]` semantic (family default None, §2.3);
`designs_per_batch`/`max_tokens`/`inputs_processed_simultaneously` → `[batch_size]` semantic
(output-invariance test); `model_weights`/`check_path` → `[checkpoint_id, model_local_path]`;
`device` → exclusion `device`; `verbose`/`disable_pbar`/`silent` → `io_only`; `model.*`,
`graph_structure.*`, `build_hydrogens` → `checkpoint_derived`; `filter` → `[filter_nan]` (no-op);
`optimize_fasta` → `[optimize_fasta]` divergence (§6.5b); `*__ddG` → `[output_kind]` semantic
(True ≡ `ddg`, False ≡ `energy`).

### 6.4 Exclusions

`reason ∈ {io_only, device, visualization, training_only, checkpoint_derived, no_op, internal,
duplicate, deferred:<backlog#>}`. Deferred (v1): `entropy_decoder`, `mutation_search.*`, CA-only,
MSA vocab-22, `run_predict_partial_charges.*`; `run_inference_ligandmpnn.py`,
`run_batch_inference_ligandmpnn.py` → `duplicate` if T0.2 confirms they drive LigandMPNN (aminx
already implements it). `DEFERRED_IDS` is a checked-in list refreshed by T0.4.

### 6.5 Upstream quirks preserved (bit-parity)

| Quirk | Anchor | Handling / test |
|---|---|---|
| PSSM mixing fires whenever `pssm_bias` non-empty (precedence) | `potts_mpnn_utils.py:1392` | PSSMMix; `knob_semantics_pssm_precedence` |
| `potts_converge` accumulates absolute energy (non-binding) → ~1000 sweeps; binding sums ΔE | `run_utils.py:116-120,148,176` | PottsRefine; `knob_semantics_potts_converge`, `..._binding_converge_stop` |
| T=0 → floor + draw | `sample_seqs.py:38-39` | `knob_semantics_t0_floor` |
| `filter` no-op | `etab_utils.py:312` | `no_op` inverse differential |
| Refine order lookup key mismatch (`str(i)` vs `"_i"`) → N→C for `num_samples>1`; chain-suffix miss → fresh randn order | `sample_seqs.py:274-279,321-345`; `run_utils.py:110-111` | host `upstream_refine_order(...)`; `knob_semantics_refine_order` |
| Ranked by pre-refine AR energy; PDB = refined seq of that best | `sample_seqs.py:257-287,368-373` | host ranking; `knob_semantics_rank_before_refine` |
| Tied bias/PSSM/omit **and fixed/sampled select** read at last listed member; designed-last group overwrites fixed members with the draw, fixed-last writes `S_true[t_last]` to all | `potts_mpnn_utils.py:1665-1684` (`tied_sample :1575-1594`) | PottsARDecode tied; `knob_semantics_tied_last_member` covers fixed-last and designed-last mixed groups |
| `X`/`-` score as zero-padded slots 21/20 | `etab_utils.py:362-366`; `sample_seqs.py:211` | potts_energy; `knob_semantics_x_gap_energy` |
| LASEr tied λ weights structure 1; default 0.0 = structure 2 only (help text says opposite) | `utils/model.py:626` vs `run_inference_tied.py:875` | code semantics, default 0.0; `knob_semantics_tied_lambda` |
| LASEr fs temp forces seq temp 1e-6 | `run_inference.py:541-544` | §5.3.1-6 |
| Tied group with any `mask==0` member: all members take that member's `S_true`, no draw; rows already written by earlier members stay | `potts_mpnn_utils.py:1641-1647` (and `:1551-1557`) | PottsARDecode tied `lax.cond` at member scan; `knob_semantics_tied_masked_member` |

### 6.5b Divergences (fixed upstream I/O bugs)

| Upstream bug | Anchor | aminx | Test |
|---|---|---|---|
| `optimize_fasta` asserts its own path, then reads `out_dir/out_name.fasta` | `sample_seqs.py:133-137` | reads `optimize_fasta` | `test_divergence_optimize_fasta_path` (oracle fixture uses path == `filename` so both agree) |
| `knn_boundary_tie`: #present ≤ K_eff < L_total → upstream torch.topk breaks the row-D_max tie arbitrarily | `potts_mpnn_utils.py:1144-1150` | aminx +inf fill, lower-index-first; runtime warning; scope = all outputs of that structure; excluded from exact waves/sidecars | A0 deterministic-subset test |
| Overlapping tied groups double-decode shared positions | `potts_mpnn_utils.py:1606-1614` | `ValueError` | `test_tied_overlap_raises` |
| (pre-existing aminx, not upstream) stock MPNN decoder passes invalid-neighbour messages when `L_total<48<L_pad` | `decoder.py:144-147`; `features.py:165-184` | unchanged; PottsMPNN fallback parity only unpadded | backlog id filed in T0.2 |

### 6.6 Gate

redsox added as a dev dependency pinned to a SHA. Harness `tests/redsox/test_knob_superset.py`:

```python
ROWS = tomllib.load(open(ALIAS, "rb"))["row"]
REFS = [c for c in vars(reference_surfaces).values() if is_dataclass(c)]
REF_F = {f.name for c in REFS for f in fields(c)}
TGT = (RunSpecification, SamplingSpecification, ScoringSpecification, PottsMPNNOptions, LaserOptions)
TGT_F = {f.name for c in TGT for f in fields(c)}
LIVE = [r for r in ROWS if r["equivalence"] != "exclusion"]

def test_rows_bijective():
    refs = [r["ref"] for r in ROWS]; assert len(refs) == len(set(refs)) and set(refs) == REF_F
def test_superset():
    alias = {r["ref"]: r["targets"][0] for r in LIVE}
    filt = [make_dataclass(c.__name__, [(f.name, object) for f in fields(c) if f.name in alias]) for c in REFS]
    res = check_superset(TGT, filt, alias); assert res["verdict"] == "SUPERSET-HYPOTHESIS-PASS", res
    for r in LIVE: assert r["targets"] and set(r["targets"]) <= TGT_F, r
def test_exclusions():
    for r in ROWS:
        if r["equivalence"] == "exclusion":
            assert not r.get("targets") and (r["reason"] in REASONS or r["reason"].startswith("deferred:"))
            if r["reason"].startswith("deferred:"): assert r["reason"][9:] in DEFERRED_IDS
def test_parity_ids_passed():
    passed = passed_nodeids(os.environ["AMINX_REDSOX_OUTCOMES"])   # call passed, no failed phase, not xfail
    for r in LIVE: assert r["parity_test_ids"] and set(r["parity_test_ids"]) <= passed, r
    sem = {t for r in LIVE for t in r["targets"]
           if any(i.split("::")[-1].startswith("test_knob_semantics_") and i in passed for i in r["parity_test_ids"])}
    assert NEW_FIELDS <= sem   # all fields of both Options + omit_aa, omit_aa_per_position, output_kind
```
Gate script `scripts/redsox/run_gate.sh`, run as `bth run --project-slug aminx -- bash
scripts/redsox/run_gate.sh` on titanix (sidecar `scripts/redsox/run_gate.bth.toml`; pass = all
tests pass; fail = any):
0. `uv run --no-sync python3 scripts/redsox/gate_ids.py --out $OUT` → per-wave `ids_<W>.txt` /
   `files_<W>.txt` (union of alias `parity_test_ids` + every `tests/knob_semantics`, `tests/port`,
   `tests/golden` test id, partitioned by wave; `__nonport__` holds all ids outside `tests/port/**` so
   `tests/port/conftest.py` is never loaded in that wave; the port self-test is its own wave
   `port_selftest` with `tests/port/targets/port_selftest.toml` + manifest).
1. `rc=0; for W in __nonport__ $(ls tests/port/targets/*.toml | xargs -n1 basename -s .toml | sort); do AMINX_PORT_WAVE=$W AMINX_REDSOX_SELECT=$OUT/ids_$W.txt AMINX_REDSOX_OUTCOMES=$OUT/outcomes.jsonl uv run --no-sync pytest -o addopts="" $(cat $OUT/files_$W.txt) || rc=1; done`
   (no `-m`). Every module under `tests/port/` (incl. selftest, `port_wave("port_selftest")`) declares
   `pytestmark = pytest.mark.port_wave("<wave>")` (missing marker = collection error); its ids go only
   to that wave's list. Hooks in `tests/conftest.py`, active only when those env vars are set:
   deselect (never skip) items whose `port_wave` ≠ `AMINX_PORT_WAVE` or whose nodeid ∉ ids file;
   `pytest.UsageError` if a listed id is not collected; `pytest_runtest_logreport` appends
   `{nodeid, wave, when, outcome, wasxfail}`. `passed_nodeids` = ids whose call phase passed in a
   record with `wave == declared wave`, no failed phase, no `wasxfail`; `skipped` never counts. Gate
   passes iff `rc==0` and step 2 passes. `tests/port/conftest.py` hooks return early if
   `AMINX_PORT_WAVE == "__nonport__"` (defensive).
2. `AMINX_REDSOX_OUTCOMES=$OUT/outcomes.jsonl uv run --no-sync pytest -o addopts="" tests/redsox -q`
redsox U1 reachability runs only as a smoke check (presence-only). T0.4 runs the gate on current
aminx first (expected FAIL = implementation checklist). Follow-up filed: redsox CLI multi-target +
recursion + per-class aliases.

## 7. Parity

### 7.1 Oracles and shims

- Upstreams vendored read-only at pinned SHAs: `/home/marielle/repos/PottsMPNN` @ `0cb0a58`,
  `/home/marielle/repos/LASErMPNN` @ `e70f2c6d`.
- Oracle env: separate project `aminx-oracles/` (torch, torch_scatter, torch_cluster, prody) on
  titanix; never in aminx's env.
- `scripts/parity/dump_{potts,laser}_oracles.py`: `torch.manual_seed(0)` before model construction;
  record missing/unexpected keys; each fixture in f64 and f32; sealed `.npz` (inputs, per-layer
  intermediates, outputs) + `oracle_manifest.toml` (upstream SHA, weights SHA-256, dump SHA-256,
  precision, `shimmed`, `shim_sha256`, `shim_sites`). Oracle generation is a bathos run.
- **Shims** (`aminx-oracles/shims/`, monkeypatched; vendored sources never edited):
  1. Draws (`torch.multinomial`/`Categorical.sample` at `run_utils.py:174,263`,
     `potts_mpnn_utils.py:1480,1590`, LASEr `model.py:627,859` + χ sites listed in T0.2) →
     `c = cumsum(p.double(), -1)`; `i = searchsorted(c, u·c[-1], right=True)`;
     `i = min(i, last index with p>0)`, with an injected f64 uniform stream indexed by
     (sample, step[, χ]). aminx implements the identical rule in **all** driver draws, in f64 under
     x64 (parity tiers) and in `p.dtype` otherwise (injected-uniform parity is structural, not a
     test-only path). Tests: zero-probability tail token never drawn; `u = 1−2⁻⁵³` (f64) and
     `1−2⁻²⁴` (f32) return the last positive-probability index.
  2. Order: PottsMPNN native `decoder(decoding_order=)` / `optimize_sequence(decoding_order=)`
     (no patch); LASEr order draw in `sample` patched (T0.2 cites line).
  3. Dropout: LASEr `nn.Dropout.forward` → `x*mask/(1-p)` with injected masks keyed by (module path,
     call idx); aminx proofread stage accepts the same masks.
- Exact-token tiers are **f64 only**; f32 reports match rate and permits a mismatch only where
  `min_k|cdf_k − u| < 1e-6` (logged), else fail.

### 7.2 Tier contract (derived from xtrax `port/` @ xtrax `35c5100`; extensions listed)

Per wave `tests/port/targets/<wave>.toml`: `[port] wave_id, symbol_qualname, oracle_id
("ref:tests/port/reference/<wave>:v1:sha256:<hash>"), reference_subtree`; `[capabilities]
stochastic, dynamic_shape`; `[parity] ad_critical=false, ad_critical_justification="",
tolerance_policy_f64, tolerance_policy_f32, max_traces`; `[access] reference_write_identities,
fixer_read_only_on_reference=true`. `tests/port/manifests/<wave>.toml` with `manifest_hash`
verified by conftest. Sealed `tests/port/reference/<wave>/algo.py` (`# REFERENCE: DO NOT MODIFY`)
loads the sealed `.npz` by hash. Markers `tier_1..tier_5` in aminx `pyproject.toml`; conftest
enforces T1→T2→T3→(T4 iff ad_critical)→T5; `domain=port` records to `.praxia/audits.jsonl` in xtrax
record shape. **Extensions vs xtrax:** (1) two tolerance keys (xtrax has one, `port_target.toml:17`);
the audit record carries the policy of the tier that ran; (2) `AMINX_PORT_WAVE` overrides `[tool.port]
target` (xtrax reads pyproject only, `conftest.py:64-71`); CI iterates waves; (3) conftest reads
`stochastic`: a stochastic wave without an injected-uniform oracle key is a T2/T3 error; (4) T2 runs
in a scoped `jax.experimental.enable_x64()` fixture; (5) `pytest-timeout` added to dev deps
(`conftest.py:214,232`). T5 uses `max_traces`. **Self-test first (T0.3):** sign-flipped kernel must
fail T2; retrace-per-call kernel must fail T5. Follow-up: `xtrax.port` pytest plugin in the wheel.

| Wave | Symbol | stochastic | tol f64 / f32 | max_traces |
|---|---|---|---|---|
| `potts_head` | `PottsHead` | no | rtol=1e-10,atol=1e-12 / rtol=1e-5,atol=1e-6 | 1 |
| `potts_merge_pair_d2` | `merge_pair(denom=2)` | no | exact / atol=1e-7 | 1 |
| `potts_merge_pair_d4` | `merge_pair(denom=4,exclude_self)` | no | exact / atol=1e-7 | 1 |
| `potts_energy` | `potts_energy` | no | rtol=1e-10 / rtol=1e-5 | 1 |
| `potts_ar_decode` | `PottsARDecode` (+tied, PSSMMix) with injected uniforms/order; fixture with `fixed_positions` (`chain_M_pos=0` rows); negative control AR `m = present·chain_M_pos` must fail | yes | exact tokens / match rate | 1 |
| `potts_refine` | `PottsRefine` (all modes) injected uniforms | yes | exact tokens / match rate | 1 |
| `pottsmpnn_full` | etab_forward + teacher-forced log-probs | no | atol 1e-8 / log-prob 1e-4 | 1 per bucket |
| `laser_layers` | GATv2/GVP/LN | no | rtol=1e-9 / rtol=1e-5 | 1 |
| `laser_encoder` | encoders | no | rtol=1e-8 / rtol=1e-4 | 1 per bucket |
| `laser_score` | teacher-forced seq+χ log-probs | no | atol 1e-8 / 1e-4 nats | 1 per bucket |
| `laser_decode_step` | one AR step, injected uniforms | yes | exact / match rate | 1 |
| `laser_rotamers` | RotamerBuilder | no | atol 1e-9 Å / 1e-4 Å | 1 |

Tolerances fixed in the wave TOML before first run; loosening only via a new bathos run citing the
measured deviation.

### 7.3 Bathos sidecars (committed before running)

Invocation `bth run --project-slug aminx -- uv run --no-sync python3 scripts/parity/<name>.py …`
(titanix; L1/L2 local gates first); verification by record (`bth compact`; `bth sql "SELECT
id,status,outcome,exit_code,command FROM runs WHERE id LIKE '<p>%'"`) + spot-check output paths.
Every sidecar declares `pass`/`inconclusive`/`fail`, a measured-path negative control that must FAIL
(else `instrument_invalid`), and a synthetic ground truth where applicable.

| Sidecar | Data (path+SHA-256 in TOML) | pass | inconclusive | fail | Negative control / ground truth |
|---|---|---|---|---|---|
| `runner_goldens` (T0.5a in-memory at pre-refactor SHA; T0.5b Zarr at T0.0 merge SHA — Zarr paths TypeError before T0.0) | matrix {sample in-memory, sample Zarr, score nll, jacobian in-memory, jacobian Zarr, inspect} × 2 MPNN ids (1 proteinmpnn, 1 ligandmpnn) × 2 fixtures, pinned; `uv.lock` SHA-256, jax/jaxlib versions, titanix GPU index recorded; capture + compare same host/GPU | arrays byte-equal + dtype + shape; attrs equal on root and every group after dropping `_CORE_PROVENANCE_FIELDS` (`zarr_sink.py:34`); `metadata["specification"]` compared only on field names present at capture SHA; other metadata keys exact | — | any diff | `random_seed=1` must differ; planted change to one non-provenance attr must fail |
| `potts_energy_parity` | `PottsMPNN/inputs/example_pdbs/*` + L=30 chain + 200 random seqs/structure | \|ΔE\| ≤ 1e-4 + 1e-5·\|E_up\| | ≤ 10× bound | > 10× | permute `etab_out` rows → fail; hand-built 3-residue etab with analytic E → 1e-12 (f64) |
| `potts_ddg_megascale` | `energy_benchmark_datasets/megascale_test_subset.csv` (all rows; n in TOML) | max \|Δddg\| ≤ 1e-4 | (1e-4,1e-3] | > 1e-3 | skip transpose in `merge_pair` → fail |
| `potts_ar_refine_exact` | example_pdbs, 50 seeds, injected uniforms/order, f64 | exact match = 1.0 | [0.99,1.0) | < 0.99 | wrong partition sign; N→C order; AR `m = present·chain_M_pos` on a `fixed_positions` fixture → fail |
| `potts_sample_dist` | distributional protocol, Potts conditions | see below | | | T×m; N→C (plain, T=1.0) |
| `laser_score_parity` | `4jnj-1_prot.pdb` + 20 complexes from `databases/` (list pinned) | max \|Δ log-prob\| ≤ 1e-4 | (1e-4,1e-3] | > 1e-3 | permute one decoder layer → fail |
| `laser_decode_e2e` | same 21, injected order, argmax | exact seq + χ-bin = 1.0 AND max circular \|Δchi_deg\| ≤ 1e-6° (f64) on chi_mask | — | otherwise | reversed order → < 1.0; aminx χ bin +1 mod Nbins → fail; offset := 0 → fail |
| `laser_sample_dist` | distributional protocol, LASEr conditions | see below | | | T×m |
| `laser_proofread_parity` | 5 complexes, injected masks | max \|Δ mean\| ≤ 1e-4 | (1e-4,1e-3] | > 1e-3 | dropout off (std≡0, \|Δmean\|>1e-3 somewhere) → fail; ground truth: n_orders=n_dropouts=1 equals the single injected-mask sample (f64 exact) |
| `knob_semantics_<knob>` (pytest, collected by gate) | per-knob fixture where the knob binds | set ≠ default on a pre-registered observable AND set case matches oracle/analytic | — | no difference or mismatch | built-in (differential); `no_op` rows invert: set == unset byte-identical in aminx and oracle |

**Distributional protocol.** Runs per (structure s, condition c, temperature T): aminx A, upstream
U1 (shimmed, iid u), U2 (unshimmed), n=1000 each, disjoint seeds. Per designed position i
`TV_i(X,Y)=½Σ_a|p̂_X(a)−p̂_Y(a)|`, `D(X,Y)=mean_i TV_i`, `Δ_s = D(A,U1) − D(U2,U1)` (equal n cancels
plug-in bias to first order). Bootstrap B=2000 resampling sequences within A, U1, U2 independently.
Per (c,T), pooled `Δ̄ = mean_s Δ_s`: **pass** = 95% one-sided upper bound < δ AND `max_s Δ_s < 2δ`;
**fail** = 90% CI lower bound > δ; else **inconclusive**. Sidecar pass iff all (c,T) pass; fail if
any fail. Default δ = 0.02 (TV units), pilot may set within [0.01, 0.05]. LASEr χ1: positions where
upstream's modal AA has χ1, restricted to samples with that AA, TV over 36×10° bins, δ_χ = 0.05.
Potts conditions: `plain` (mode none, no PSSM/ties) T∈{0.1,0.3,1.0}; `pssm` (fixture PSSM,
`pssm_multi=0.5`, bias flag on, mode none) T=0.3; `refine` (`potts`, `optimization_temperature=0.5`,
AR T=0.3); `tied` (2-member fixture, β=1) T=0.3; default T_opt=0 covered only by the exact tier.
LASEr conditions: T∈{0.1,0.3,1.0} min_p=0; T=0.3 min_p=0.05. Negative controls (must fail):
near-margin `T_A = m·T` (m from pilot, default 1.25); N→C order (Potts plain T=1.0); r1 gross
controls secondary.

**Runs.** U3 = second shimmed upstream run with fresh iid uniforms, seeds disjoint from U1 (a null
replicate of U1). **Structures (pinned path+SHA-256).** Potts pilot `inputs/example_pdbs/{2yc3,3dkm}.pdb`;
Potts confirmatory `{3gg7,4jox,6w25,swe1_ligand}.pdb`; `tied` uses the pinned 2-member tied fixture.
LASEr pilot: entries 1–2 of the `laser_score_parity` list; LASEr confirmatory: `4jnj-1_prot.pdb` +
entries 3–6. **Pilot** (`*_pilot.bth.toml`, exploratory, cites nothing), per (c,T) at n=1000: U1,
U2, U3, control `C_m` = shimmed upstream at `m·T`, m∈{1.1,1.25,1.5}. Null statistic
`Δ⁰_s = D(U3,U1) − D(U2,U1)`; `ĥ` = half-width of the two-sided 90% bootstrap CI (B=2000) of
`mean_s Δ⁰_s` (unscaled; conservative for S_conf ≥ S_pilot); `q̂_s` = max over pilot structures of
the 97.5th bootstrap percentile of `Δ⁰_s`. `δ = clip(max(2ĥ, q̂_s/2), 0.01, 0.05)`;
`Δ_neg(m) = mean_s[D(C_m,U1) − D(U2,U1)]`; choose smallest m with `Δ_neg(m) ≥ δ + 2ĥ`. Escalation:
if `max(2ĥ, q̂_s/2) > 0.05` or no m qualifies → n=4000, re-pilot once; still failing →
`instrument_invalid`. Confirmatory: the U1 bootstrap resample is shared by both terms of `Δ_s`.
**Shim check** (every (c,T)): `mean_s[D(U2,U1) − D(U3,U1)] > ĥ` → `instrument_invalid` (shim
distorts sampling). δ, m, n, ĥ committed into the confirmatory sidecar before it runs; its negative
control is aminx at `m·T`.

## 8. Fixer tasks

| ID | Task | Depends | Gate |
|---|---|---|---|
| T0.0 | **Sink run_id fix (prerequisite).** `aminx.host.sink_ids.sink_spec_for(spec, output_dir, *, flush_every=1, run_id=None) -> SinkSpec` calls pinned `xtrax.run.sink.derive_sink_spec(spec.run_spec, run_id=run_id or spec_run_id(spec), output_dir=Path(output_dir), format="zarr", flush_every=flush_every)` (`sink.py:61-105`). `spec_run_id(spec) = sha256(json.dumps(run_specification_to_json_dict(spec), sort_keys=True, separators=(",",":")).encode()).hexdigest()[:16]`; on `SpecJSONEncodeError` fall back to `xtrax.run.ident.new_run_id()` + `logger.warning`. `run_spec.run_id` never set (static field → retrace). Re-run policy: if `output_dir` holds a Zarr root whose `attrs["run_id"]` ≠ derived id → `ValueError(f"{output_dir} holds outputs of a different specification (run_id {old}); use a new output_dir or pass run_id=")`; same spec reopens (mode `a`, arrays overwrite). Spec-less sites: `DesignsWriter(..., run_id=None)` defaults to `sha256(str(Path(path).resolve()))[:16]`; `jacobian_profile.py` uses sha256 of canonical argv JSON. All five call sites (`host/streaming.py:80`, `host/runner.py:1225`, `io/designs.py:81`, `sampling/multistate_poe.py:605`, `scripts/analysis/jacobian_profile.py:187`) migrated | T0.5a | per-site tests: fresh dir writes expected root `run_id`; same spec reopens; different spec raises the aminx ValueError; in-memory goldens exact |
| T0.1 | Vendor upstreams at pinned SHAs; `aminx-oracles/` env on titanix | — | files + manifest |
| T0.2 | Probe report (§10) | T0.1 | appended to §10 |
| T0.3 | `tests/port/` contract + self-test | — | selftest on titanix |
| T0.4 | Extractor, reference surfaces, alias skeleton, exclusions, harness, gate script; run vs current aminx (FAIL baseline); file redsox follow-up + deferred items | T0.1 | bathos run FAIL recorded |
| T0.5a | Runner goldens, in-memory rows, at pre-refactor SHA | — | `runner_goldens` sidecar (capture) |
| T0.5b | Runner goldens, Zarr rows, at T0.0 merge SHA | T0.0 | `runner_goldens` sidecar (capture) |
| T0.5 | FamilyDriver protocol, registry, runner dispatch + raise, `run_family_driver` (ZarrStagingSink channel, keys, ids/skips, inference mode), family Literal/derivation/consumer branches, `DecoderLayer` promote_types accumulation, L-DRV lint | T0.0, T0.2, T0.5a, T0.5b | unit tests; lint; goldens exact on titanix |
| T0.6 | `output_kind`, Options, `omit_aa*` (+RS-6b list), temperature family default, CLI flags, spec_json/campaign round-trip, fail-loud validation | T0.5 | unit tests; goldens exact |
| A0 | PottsMPNN host featurizer (§4.1a) | T0.1, T0.2 | A0 gate |
| A1 | PottsMPNN oracle dumps (f64+f32, shims, `declayer_f64.npz`) | T0.1, T0.2 | manifest |
| A2 | `PottsMPNN` composition + conversion + self-edge test | T0.3, T0.5, A0, A1 | wave `pottsmpnn_full`; `DecoderLayer` f64 fixture test |
| A3 | `PottsHead`, `merge_pair`, `potts_energy`, §4.1a masks, alphabet conversion | T0.3, A0, A1 | waves `potts_head`, `potts_merge_pair_d2/d4`, `potts_energy` |
| A4 | Driver `score:energy|ddg`, mutants/DMS, partitions axis, sinks | A2, A3, T0.6 | `potts_energy_parity`, `potts_ddg_megascale` |
| A5 | `PottsARDecode` (+tied, PSSMMix), `PottsRefine` (all modes, binding, order quirk), `PottsSampleEnergy` + host ranking, all Potts Options | A4 | waves `potts_ar_decode`, `potts_refine`; `potts_ar_refine_exact`; pilot → `potts_sample_dist`; knob tests |
| A6 | ADR + lint updates | A2 | lint green |
| B0 | LASEr host featurizer | T0.1, T0.2 | field-level parity (5 fixtures) |
| B1 | LASEr oracle dumps | T0.1, T0.2 | manifest |
| B1.5 | LASEr weight conversion | B1 | 0 unmapped keys |
| B2 | layers | T0.3, B1.5 | wave `laser_layers` |
| B3 | encoders + ligand bucket axis | T0.3, B0, B2 | wave `laser_encoder` |
| B4 | LaserDriver `score:nll|logits`, alphabet boundary | T0.3, B3, T0.6 | wave `laser_score`; `laser_score_parity` |
| B5 | `LaserJointDecode` + §5.3.1 | T0.3, B4 | wave `laser_decode_step`; `laser_decode_e2e`; pilot → `laser_sample_dist` |
| B6 | RotamerBuilder (post) + PDB/FASTA sinks | T0.3, B5 | wave `laser_rotamers` |
| B7 | proofreading (both), two-structure tied, remaining Laser Options | B5 | `laser_proofread_parity`; knob tests |
| Z1 | redsox gate PASS on final tree | A5, A6, B6, B7 | `run_gate` sidecar PASS |
| Z2 | CLI docs + `using-aminx` skill | Z1 | — |

## 9. Risks

| Risk | L | Mitigation |
|---|---|---|
| `run_family_driver`/temperature-default refactor perturbs MPNN path | M | T0.5a exact goldens |
| merge_pair / padding edge cases | H | exact d2/d4 waves; padding-invariance test; L=30 & short-partition fixtures |
| PottsARDecode row-kernel diverges from full-sequence DecoderLayer | M | wave `potts_ar_decode` f64 exact on injected path; `pottsmpnn_full` teacher-forced |
| LASEr scan compile time | M | O(n_dec·K) step; jaxpr guard; bucketed L |
| Ragged ligand / partition recompiles | M | bucket axes; T5 `max_traces` |
| Oracle env (torch_scatter/cluster) | M | `aminx-oracles/` on titanix; sealed dumps |
| Semantic mismatch behind aliases | H | namespaced refs; parity ids must PASS (junit); differential knob tests for all new fields |
| Stochastic parity false pass/fail | M | shims + f64 exact tiers; TOST-style protocol with piloted δ and near-margin controls |
| Upstream `strict=False` random weights | M | T0.2 key audit blocks |
| Oracle fixture size | L | dumps on titanix, hashes in repo |

## 10. Probe report

Resolved during authoring (260929): `EncoderOutput` carries `edge_features`/`neighbor_indices`
(`types/bundles.py:457-473`); no CA featurizer; aminx `bias`/`fixed_positions`/`fixed_tokens`/
`fixed_mask`, no omit; `temperature` is `Sequence[float] | float` default 0.1.

Open for T0.2: runner helper names/signatures (`_canonical_structure_ids_for_spec`,
`resolve_target_samples`, `make_axis_dispatch_via_xtrax`, `sample()` metadata keys); full
`model_family` consumer grep incl. fallback-path consumers; PottsMPNN missing/unexpected keys per
checkpoint; ft config; `proteinmpnn_compatible` equivalence; T floor value; verbatim transcription
of `decoder`/`tied_decoder` PSSM block, `optimize_sequence` incl. tied variants, `nodes`, refine
order keying `sample_seqs.py:321-345`; LASEr `model_params` dims, ligand-encoder keys,
`LASER_ALPHABET` string, every `radius_graph`/`scatter_*` site, χ draw sites, order-draw line,
`tied_sample` ignored-knob list; LigandMPNN entry points in LASEr repo.
