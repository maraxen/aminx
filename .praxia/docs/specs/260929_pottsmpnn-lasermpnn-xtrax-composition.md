---
title: PottsMPNN + LASErMPNN as xtrax-composed model families on the central runner
description: Port KeatingLab PottsMPNN and polizzilab LASErMPNN into aminx as xtrax-composed FamilyDrivers dispatched from aminx.host.runner, with redsox knob-superset, xtrax-tier parity, and bathos-preregistered gates
task_id: 260929_potts-laser-xtrax-compose
status: draft-r1
created: 260929
amends: decisions/260605_potts-parallel-not-stageset.md (scope-narrowing, see §3)
adversarial_log: audits/260929_potts-laser-spec-adversarial-log.md
---

# PottsMPNN + LASErMPNN as xtrax-composed model families

Revision history: r0 (8ea7604e) → r1 after round-1 challenge (25 objections) + defense
(22 concede / 3 partial / 0 rebut). r1 **removes** four r0 designs rather than patching them:
`ModelFamily.plan_builder` bypass of `make_inference_plan`, the `StageSet.post_decode` slot,
`InterpolatedTieFuse`, and the LASEr `sc_coords` decode carry.

## 0. Goal, non-goals, assumed decisions

**Goal.** Through the one central runner (same CLI, same `RunSpecification`, same
`host/runner.sample`/`score` entry points, same prep/sinks/xtrax tiling):

```bash
aminx run --model-family pottsmpnn --checkpoint-id pottsmpnn_vanilla_20 sample --inputs x.pdb ...
aminx run --model-family pottsmpnn --checkpoint-id pottsmpnn_vanilla_20 score  --inputs x.pdb --output-kind ddg --potts-options-json opts.json
aminx run --model-family lasermpnn --checkpoint-id lasermpnn_0p1A_nothing_heldout sample --inputs lig.pdb --laser-options-json opts.json
```

every input knob of the pinned upstream implementations is reachable from `RunSpecification`
(or recorded as a justified exclusion), and numerical/behavioural parity against pinned upstream
PyTorch is demonstrated by pre-registered bathos runs over an xtrax-contract tier ladder.

**"xtrax-composed" (normative).** Every per-structure computation in a driver is an `eqx.Module`
stage; every batch/sample/candidate/partition/order/dropout loop is an xtrax axis
(`AxisSpec` → `BatchPlanner` → `make_axis_dispatch` / aminx `make_axis_dispatch_via_xtrax`);
every cross-axis reduction is an `AxisBoundary` `Fuse`; every host export is a xtrax
`Sink`/`ZarrStagingSink`. No Python `for` loop over a data axis inside a driver's jitted region; no
bare `jax.vmap` over a named axis outside the planner. Enforced by lint test L-DRV (§8 T0.5).

**Non-goals (v1).** Training; LASEr entropy-based decoding (`entropy_decoder`/`--ebd`);
PottsMPNN `mutation_search.py`; CA-only PottsMPNN (`ca_model_weights/`, aminx has no CA
featurizer); MSA vocab-22 checkpoints (`*_msa_20`); plotting/visualisation; migrating the TRW
`aminx.potts.PottsModel`; LASEr partial-charge prediction entry point; portable-JSON v3. Each is a
recorded `deferred:<backlog#>` or `visualization`/`training_only` exclusion (§6.4), never silent.

**Assumed decisions (defaults applied; user may override — see adversarial log U1–U5).**
- U1: LASEr gets its own FamilyDriver; aminx MPNN contracts (`StageSet`, `EncoderOutput`,
  `ModelProtocol`, `DecodeMode`, `make_inference_plan`) are **not** widened.
- U2: **bit-parity** with pinned upstream, including quirks (PSSM precedence at
  `potts_mpnn_utils.py:1392`, `potts_converge` absolute-energy accumulation, T=0 floor, `filter`
  no-op). Each quirk is listed in §6.5 with its knob. Family-level defaults follow upstream example
  configs (e.g. `optimization_mode="potts"`, `example_config_sample_seqs.yaml:22`).
- U3: MSA vocab-22 checkpoints deferred.
- U4: redsox gate via a local pytest harness over the redsox **library** `check_superset`
  (multi-target capable); redsox CLI multi-class support filed as a follow-up.
- U5: LASEr proofreading and two-structure tied sampling are **in** v1 (task B7).

## 1. Recon evidence base

| Fact | Anchor |
|---|---|
| Central runner entry points | `src/aminx/cli.py` `run` callback (`:448-461`, holds `--model-family`) + `run_sample`/`run_score`/`run_jacobian`/`run_inspect` → `src/aminx/host/runner.py` `sample()`/`score()`/`jacobian()`/`inspect()` |
| `score()` does **not** build an InferencePlan unless `average_node_features` | `host/runner.py:498-506` → `aminx.scoring.score.make_score_fn`; `one_hot(seq,21)` `:626`; `output_h5_path` NotImplemented `:489-491`; NLL result `:661-670` |
| `ScoringSpecification.sequences_to_score` required | `run/specs.py:556-561`; CLI `--sequences-to-score` required `cli.py:816` |
| Family Literal + consumers | `run/specs.py:236`; derivation `:476-490`; consumers `host/_sampling_helper.py:255`, `host/prep.py:47-49`, `run/run_spec_portable_json.py:12-15,153`, `multistate_poe.py:609`, `streaming.py:84`, `_sampling_grid_lineage.py:94,111`, `campaign.py:95`, `run/spec.py:321` (T0.2 re-greps) |
| `load_model` checkpoint-id normalisation | `io/weights.py:386-392` nulls ids lacking `_v_`/`mpnn`; substring topology `:77-103`; `VOCAB_SIZE` `:453` |
| MPNN contracts (unchanged by this spec) | `types/stages.py:389` `StageSet`; `types/bundles.py:457-486` `EncoderOutput(node_features, edge_features (S,L,K,D), neighbor_indices, mask)`; `types/protocols.py:124` `ModelProtocol`; `host/plan.py:836` `make_inference_plan`; `plan.py:796-803` `resolve_decode_mode` single decision point; `inference/decode/factory.py:141` sealed `DecodeMode` |
| aminx sampling knobs | `SamplingSpecification.temperature (Sequence|float)`, `bias (L,V)`, `fixed_positions`, `fixed_tokens`, `num_samples`; `RunSpecification.fixed_mask`, `tied_positions`, `noise`, `decoding_order_fn`, `random_seed`; **no omit field** (`run/specs.py:~570-590`) |
| aminx has no CA featurizer, no B-factor handling | `rg ca_only\|CAProteinFeatures src/aminx` 0; `rg b_factor\|bfactor src/aminx` 0 (only type aliases) |
| Existing Potts code = TRW structure model (not PottsMPNN) | `potts/model.py:71` `PottsModel` (`node_lin`/`project_h`/`project_j` + `DifferentiableTRW`); ADR `decisions/260605_potts-parallel-not-stageset.md`; lint `tests/lint/test_potts_import_boundary.py` |
| Upstream PottsMPNN | KeatingLab/PottsMPNN @ `0cb0a58`; `potts_mpnn_utils.py:1225` class; `:1237-1261` = ProteinMPNN layers + `etab_out=nn.Linear(hidden,potts_dim)`; forward `:1267-1290` etab from post-encoder `h_E`, masked, slot-0 diagonalised, `merge_duplicate_pairE` (denom 2) |
| PottsMPNN energy path uses a second merge | `run_utils.get_etab` `:816-835` → `etab_utils.functionalize_etab` merge denom 4 excluding self (`etab_utils.py:177-178,253-271`), pad to 22; `calc_eners` `:299-309` (no ×0.5); alphabets: model `X`=20 (`potts_mpnn_utils.py:69`), etab_utils `-`=20 `X`=21 (`etab_utils.py:362-366`) |
| PottsMPNN sampling/refine | AR `sample` `:1321` (PSSM prob-space mix `:1391-1404`, precedence quirk `:1392`, decoding order `argsort((chain_mask+1e-4)*|randn|)`), `tied_sample` `:1490`; `run_utils.optimize_sequence` `:113-198` (modes potts/potts_converge, binding `:135-148`, accumulation `:176`, cap `:116-120`), `nodes` `:180-270`; T floor `sample_seqs.py:38-39`; `strict=False` load `sample_seqs.py:60`, `energy_prediction.py:35` after xavier init `potts_mpnn_utils.py:1263-1265` |
| PottsMPNN config | `inputs/example_config_{sample_seqs,energy_prediction}.yaml` (hidden 128, edge 128, potts_dim 400, layers 3, num_edges 48, vocab 21); vocab 22 iff `'msa' in check_path` (`sample_seqs.py:37`) |
| Upstream LASErMPNN | polizzilab/LASErMPNN @ `e70f2c6d765416f7e29d51bfd6d4e08496438878` (MIT); `utils/model.py:107` `class LASErMPNN(...)`; GATv2/GVP (`model_generics.py`), `torch_scatter`/`torch_cluster` (`model.py:4-5`); encode `:216-259`; score `:261`; `tied_sample` `:460-627` (two structures, `λP1+(1−λ)P2` `:626`, equal-length assert `:477`); `sample` `:727-941`; carry = per-layer `prot_node_stack` `:773`; prior residues seen via seq embedding + binned-χ RBF only `:815-818,916-917`; χ GVP update `:931-934`; rotamers built post-sampling `run_inference.py:747-750` |
| LASEr knob entry points | `run_inference.py:773-792` (+ `sample_model` kwargs `:511-517`), `run_batch_inference.py`, `run_inference_tied.py:873-875`, `run_proofreading.py`, `run_inference_ligandmpnn.py`, `run_batch_inference_ligandmpnn.py`, `run_predict_partial_charges.py` |
| redsox | `checkers/superset.py:27-60` `check_superset(target_dataclasses, reference_dataclasses, alias_map)` (library accepts many targets; no recursion `:37-39`); CLI uses only `target_dataclasses[0]` (`cli/main.py:120-121,161-165,205-207`); U1 `checkers/reachability.py:18-40,63-68` is attribute/kw-name presence anywhere in src |
| xtrax port contract | `/home/marielle/projects/xtrax/port/` (dev-only, **not in wheel**; aminx pins wheel `pyproject.toml:26` @56a9f551): `port_target.toml` (`[port] wave_id, oracle_id(sha256), symbol_qualname, reference_subtree`; `[capabilities] stochastic, dynamic_shape`; `[parity] ad_critical, ad_critical_justification, tolerance_policy="rtol=…,atol=…,matmul_precision=highest", max_traces`; `[access]`), `tests/conftest.py` (markers `tier_1..tier_5`, blocking T1→T2→T3→(T4 iff ad_critical)→T5, target resolution `:64-76`, sealed `reference_subtree/algo.py` import `:159-171`, `manifest_hash` verify `:103-114`, audits.jsonl emit `:277-286`) |
| aminx parity/bathos | `src/aminx/parity/`; `tests/parity/test_full_model_parity.py:32-36`; `parity_heavy` excluded by addopts (`pyproject.toml:229`); `.bth.toml` slug `aminx`; template `scripts/ebm/collect_synthetic_parity_evidence.bth.toml` |

## 2. Architecture

### 2.1 FamilyDriver seam (the only runner change)

```python
# src/aminx/host/family_driver.py
class FamilyDriver(Protocol):
  name: str                                             # "pottsmpnn" | "lasermpnn"
  options_type: type                                    # PottsMPNNOptions | LaserOptions
  def handles(self, spec: RunSpecification, purpose: Literal["sample","score","jacobian","inspect"]) -> bool
  def load(self, spec) -> eqx.Module                    # checkpoint registry / model_local_path; never io.weights.load_model
  def batches(self, spec) -> Iterable[FamilyBatch]      # host featurisation → fixed-shape padded pytrees
  def axes(self, spec, batch) -> list[AxisSpec]         # named xtrax axes for this purpose
  def stages(self, spec, model) -> FamilyStages         # eqx stage modules + AxisBoundary map
  result_schema: Mapping[str, SinkArraySpec]            # name -> (dims, dtype, attrs); drives sink + result dict
FAMILY_DRIVERS: Registry[FamilyDriver]                  # aminx.registry.Registry (like SAMPLERS)
```

- **Dispatch.** `runner.sample()` and `runner.score()` gain one statement immediately after the
  spec is synced: `if (d := FAMILY_DRIVERS.get(spec.model_family)) and d.handles(spec, purpose):
  return run_family_driver(d, spec, purpose=purpose)`. `jacobian()`/`inspect()` gain the same
  check; `LaserDriver.handles` returns False for them and the runner then raises
  `ValueError("lasermpnn does not support <purpose> in v1")` (fail-loud, never silent fallthrough).
- **`run_family_driver`** (`host/family_runner.py`) reuses, not reimplements:
  `_canonical_structure_ids_for_spec`, `_structure_ids_for_batch`,
  `streaming_tensor_sink_session`/`StreamingBatchHost`, `resolve_target_samples`,
  `make_axis_dispatch_via_xtrax`, `ZarrStagingSink` (fed by `result_schema`), result-dict keys
  `schema_version="<family>_v1"` + the same `metadata` keys as `sample()`. T0.2 confirms each
  helper's current name/signature; any helper that is not reusable as-is gets a thin extraction
  refactor with the MPNN path byte-identical (guarded by `parity_targeted`).
- **PottsMPNN plain AR sampling rides the existing MPNN path.** `PottsMPNN` satisfies
  `ModelProtocol` (same `features/encoder/decoder/w_out/w_s_embed`, plus `potts_head`).
  `PottsMPNNDriver.handles(spec,"sample")` is False iff `optimization_mode=="none"` and no PSSM
  knobs are set; that case goes through `make_inference_plan` unchanged, with the family's
  `decoding_order_fn` default `"mpnn_randn"` (§4.3). All other PottsMPNN purposes and all LASEr
  purposes go through the driver. `prep_protein_stream_and_model` calls `d.load(spec)` when a
  driver is registered for the family (so `load_model`'s id normalisation at `weights.py:386-392`
  is never reached for `laser_*`/`pottsmpnn_*`).
- **Unchanged:** `StageSet`, `EncoderOutput`, `ModelProtocol`, `DecodeMode`, `make_inference_plan`,
  `resolve_decode_mode`, `InferencePlan.decode`. No `plan_builder`, no new StageSet slot.

### 2.2 Family plumbing

- `run/specs.py:236` Literal widens to add `"pottsmpnn"`, `"lasermpnn"`; `__post_init__` derives
  them from `checkpoint_id` prefixes `pottsmpnn_`/`lasermpnn_` (checked **before** the existing
  `model_type`-based derivation); explicit family never overridden.
- **Every `model_family` consumer** (list in §1; T0.2 re-greps and appends) gets an explicit branch:
  `_prepare_ligand_context` — not reached for driver families (driver owns ligand featurisation),
  asserted by test; checkpoint registry (`prep.py:47-49`) — gains `pottsmpnn`/`lasermpnn`
  sections; portable JSON v2 (`run_spec_portable_json.py:153`) — raises `ValueError` for both new
  families (v3 deferred); multistate PoE / grid lineage / campaign / streaming — reject driver
  families with `ValueError` unless T0.2 shows they are family-agnostic.
- Unsupported `(family, purpose, output_kind)` → `ValueError` at `__post_init__`.

### 2.3 Spec/CLI surface

- `ScoringSpecification.output_kind: Literal["nll","logits","energy","ddg"] = "nll"`;
  `sequences_to_score` required only for `nll`/`logits`. MPNN families reject `energy`/`ddg`.
- `RunSpecification.potts_mpnn: PottsMPNNOptions | None`, `laser: LaserOptions | None`;
  `build_run_spec` threads them to `run_spec.potts_mpnn` / `run_spec.laser`. Setting options for a
  family other than `model_family` → `ValueError`.
- `SamplingSpecification.omit_aa: Sequence[str] = ()`, `omit_aa_per_position: Mapping[int,str] |
  None = None` (family-agnostic; compiled host-side into bias `-1e8` at designed positions to match
  upstream's finite constant — note `autoregressive.py` MPNN path semantics unchanged when empty).
- CLI (task T0.6): `run score --output-kind`; `_RunBase` `--potts-options-json PATH`,
  `--laser-options-json PATH` threaded through `_base_spec_kwargs`; `spec_json`/campaign
  serialisation of both Options with round-trip tests.

## 3. ADR amendment

New `decisions/260929_pottsmpnn-lasermpnn-family-drivers.md` (`amends: 260605_potts-parallel-not-stageset`):
- 260605 governs the **TRW structure model** (`aminx.potts.*`); unchanged, lint unchanged.
- PottsMPNN is a different model: ProteinMPNN encoder/decoder + edge Potts head. Its plain AR
  sampling reuses the MPNN StageSet path **verbatim except PSSM probability-space mixing** (§4.3);
  its Potts-specific purposes run in a FamilyDriver. 260605's Option III (EncoderOutput widening)
  rejection is honoured: the Potts head consumes the existing `EncoderOutput.edge_features`.
- LASErMPNN is a FamilyDriver with its own encoder-output and result types; MPNN contracts are
  not widened.
- New code: `aminx.model.potts_mpnn`, `aminx.families.potts_mpnn`, `aminx.model.laser`,
  `aminx.families.laser`, `aminx.host.family_driver`, `aminx.host.family_runner`. A lint asserts
  none of them import `aminx.potts` (energy conventions differ; C1-07).
- `potts/model.py` `PottsModel` docstring gains a one-line pointer disambiguating it from `PottsMPNN`.

## 4. PottsMPNN

### 4.1 Model and weights

`PottsMPNN(eqx.Module)` satisfies `ModelProtocol`: `features: ProteinFeatures`, `encoder`,
`decoder`, `w_e`, `w_s_embed`, `w_out` (same names/roles as `Aminx`) + `potts_head: PottsHead`;
`capabilities = ModelCapabilities(..., emits_potts=True)` (new static field, default False).

- Conversion `scripts/recapture/pottsmpnn_model_to_eqx.py` (bathos-tracked): reuses
  `scripts/convert_weights.py`'s ProteinMPNN key map for shared layers; only `etab_out.*` is new.
  T0.2 records `load_state_dict(strict=False)` **missing/unexpected keys** for every checkpoint
  loaded into upstream `PottsMPNN`; a non-empty missing set is a blocking finding (upstream would
  be running xavier-random weights there). Output registered with source SHA-256 + upstream commit.
- In-scope checkpoints: `vanilla_model_weights/pottsmpnn_{20,30}.pt`,
  `soluble_model_weights/sol_pottsmpnn_{20,30}.pt`, `ft_model_weights/potts_ft.pt` (T0.2 confirms
  ft config). `proteinmpnn_compatible_model_weights/` are recorded as an alternate encoding of the
  same models; T0.2 determines whether they are byte-equivalent after key mapping (if so,
  excluded as `duplicate`).
- The 7 `.eqx.zst` under `mpnn_ext/external/aminx/weights/pottsmpnn/` are **not reused** (produced
  for the TRW model; provenance/meaning differ).
- Self-edge invariant: test `neighbor_indices[..., 0] == arange(L)` on valid rows for aminx
  `ProteinFeatures` (slot-0 diagonalisation `:1286` and `positional_potts_energy`
  `etab_utils.py:240` assume it).

### 4.2 Potts head and the two etab conventions

`aminx.model.potts_mpnn.head`:
- `PottsHead.__call__(edge_features (L,K,H), E_idx (L,K), mask (L,)) -> etab_raw (L,K,20,20)`:
  linear → row mask → slot-0 × eye(20).
- `merge_pair(etab, E_idx, *, denom, exclude_self)` — pure gather over reverse edges with a
  validity mask for j∉kNN(i); ports `merge_duplicate_pairE` semantics exactly for both call sites.
- `etab_forward = merge_pair(etab_raw, denom=2, exclude_self=False)` — upstream forward output;
  consumed by PottsRefine (`optimize_sequence`, `positional_potts_energy`) and emitted by the
  `potts_etab` sink with attr `etab_convention="forward_denom2"`.
- `etab_energy = merge_pair(etab_forward, denom=4, exclude_self=True)` — upstream
  `functionalize_etab`; consumed by `potts_energy`/ddG only.
- `potts_energy(etab_energy, E_idx, seq_etab_alphabet, mask)` ports `calc_eners` (no ×0.5).
  Scored sequences are encoded in the **etab_utils alphabet** (`-`=20 zero-padded energy, `X`=21):
  `X` in a scored/mutant sequence → `ValueError`; gaps allowed. Conversion from the model alphabet
  (X=20) is an explicit function with a round-trip test.
- Energy sign/scale reported exactly as upstream; `output_kind="nll"` stays MPNN AR NLL.

### 4.3 Purposes

| purpose / output_kind | Path | Stages / axes |
|---|---|---|
| `sample`, `optimization_mode="none"`, no PSSM | existing MPNN `make_inference_plan` | unchanged; `decoding_order_fn="mpnn_randn"` |
| `sample`, PSSM set | driver | MPNN encode → AR decode stage with `PSSMMix` (below) |
| `sample`, `optimization_mode ∈ {potts, potts_converge, nodes}` | driver | AR decode → `PottsRefine` stage (etab_forward) |
| `score` / `energy` | driver | encode → head → `etab_energy` → `potts_energy`; axis `candidates` |
| `score` / `ddg` | driver | as energy, `candidates` = mutants (mutant_fasta/mutant_csv else single-mutant DMS respecting `exclude_chains`), `ddg = E(mut) − E(wt)`; binding mode: axis `partition` over P sub-structure inputs (§4.4); `mean_norm` per PDB after ddG |
| `score` / `nll`, `logits` | existing MPNN score path | unchanged |
| `jacobian`, `inspect` | existing MPNN paths | Potts head not involved |

- **`decoding_order_fn="mpnn_randn"`**: `argsort((chain_mask + 1e-4) * |randn|)` (fixed positions
  first). T0.2 checks whether aminx's existing default already equals this; if so, reuse.
- **`PSSMMix`** (driver-internal AR decode stage): at each step, upstream order —
  `p = softmax(logits/T − omit·1e8 + bias/T + bias_by_res/T)`; if
  `(pssm_bias_flag and coef.numel()>0) or pssm_bias.numel()>0` (quirk preserved, §6.5):
  `p = (1−coef·multi)·p + coef·multi·pssm_bias`; if `pssm_log_odds_flag`: `p = p·(mask+0.001)`,
  renormalise; `omit_AA_mask` renormalise; `categorical`. T0.2 transcribes exact upstream lines
  `:1391-1404` into the stage docstring before implementation.
- **`PottsRefine`** (bit-parity port of `optimize_sequence` + `nodes`): modes `potts` (one sweep),
  `potts_converge` (`lax.while_loop`, upstream stop condition incl. absolute-energy accumulation,
  cap `max_iters=1000`; documented as effectively 1000 sweeps), `nodes` (teacher-forced MPNN
  decoder pseudo-likelihood pass per upstream `:180-270`); temperature floor per upstream
  (`sample_seqs.py:38-39`, T0.2 confirms value) then multinomial (never argmin); binding modes
  `both`/`only` use ΔE vs current residue against per-partition `etab_forward`s, `only` skips
  non-interface positions (`binding_energy_cutoff` Å). Tied variants `tied_optimize_sequence` /
  `tied_epistasis` in scope via `tied_positions` + `tied_epistasis` knob (T0.2 transcribes).
  Inputs: `etab_forward`, `E_idx`, decoding order, design/fixed/omit/bias/PSSM masks, optional
  partition etabs + interface mask.

### 4.4 Binding partitions

Partitions are **separate sub-structure inputs**: each partition's chains are re-featurised (own
L, kNN graph, encode, head) — upstream `get_etab` `:816-833`. Axis `partition` (ragged L → bucket
via `AxisSpec(bucket_boundaries=…)`); `AxisBoundary` Fuse on scalar energies
`E_bind = E_complex − Σ_p E_p`. Partitions scored only when `partition_flag` (ddG on and partition
contains a mutated chain, `energy_prediction.py:57-75`).

### 4.5 Sinks / result schema

`energy (N_struct, N_cand) f32`, `ddg (N_struct, N_mut) f32`, `mutant_ids`, `ddg_expt (N_mut,)
f32|nan` (from mutant_csv when present), `potts_etab (N,L,K,20,20)` opt-in (`emit_etab`),
`potts_E_idx`, `sequence`, `refine_energy_trace` opt-in. `emit_dense_hJ` converts host-side only.
Score Zarr sink exists for driver families only (MPNN score sink remains NotImplemented).

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
  chi_deg: Float[Array,"L 4"]; chi_logits: Float[Array,"L 4 72"]; chi_mask: Bool[Array,"L 4"]
```
Exact dims (`Hs, V, Hl, Vl, E, K, Kl`) come from checkpoint `model_params` (T0.2).

### 5.2 Model port

| Upstream | aminx | Notes |
|---|---|---|
| `LigandFeaturizer`, `LigandEncoderModule` (`model.py:1150,1312`) | `LaserLigandEncoder` | weights from the main checkpoint's ligand-encoder keys (T0.2 confirms present) |
| `HomoGATv2`, `HeteroGATv2`, `GVP`, `DenseGVP`, `EquivariantLayerNorm` | `aminx.model.laser.layers` | edge-list → dense padded neighbour axis + mask; `scatter_softmax` → masked softmax returning **0 (not NaN) on empty K** |
| encoder/decoder layers | `LaserEncoder`, `LaserDecoderLayer` | |
| χ heads | `LaserChiHead` | bins from `chi_angle_rbf_bin_width` (checkpoint), angle = wrap(bin_center + offset) |
| `RotamerBuilder` | `aminx.model.laser.rotamers` — **post-decode** stage (PDB writer / `sidechain_coords` sink only) | ideal-geometry buffers from `files/*.pt` → hashed `.npz` |

Graphs are built host-side into fixed shapes (protein kNN on Cα `k_pp`, ligand kNN `k_ll`,
ligand→protein kNN `k_lp` with distance cutoff mask; all from checkpoint `graph_structure`),
ligand atoms padded to axis `ligand_atoms` buckets. Every `radius_graph`/`scatter_*` site is
enumerated in T0.2 with its dense replacement.

### 5.3 Joint sequence+χ AR decode stage (driver-internal)

`lax.scan` over the decoding order (length L, masked). Carry:
`(seq (L,), seq_emb (L,H), chi_enc (L,4,R), node_stack_s (n_dec+1,L,Hs), node_stack_v (n_dec+1,L,V,3), ala_count, gly_count)`.
Step t: decoder layers over the carry (prior residues visible via seq embedding + binned-χ RBF
only) → seq logits → temperature → min-p → masks → sample → χ1..χ4 sequential with χ-GVP vector
update of node t → carry update. Per-step cost O(n_dec·K) + 4 χ heads; no rotamer build in-loop.
Jaxpr loop-body complexity guard asserts no O(L²) op in the step.

#### 5.3.1 Edge semantics (one unit test each)

1. ALA/GLY budget counts **every** decoded residue incl. fixed (`model.py:782-788`); region mask
   from `budget_residue_selection` (ProDy, host) or exposed-non-SS heuristic (B0).
2. Disabled residues masked only at designed positions with `finfo.min` (`:835-841`);
   `disable_charged_fs` uses `-inf` (`:843-848`).
3. min-p applied only when a temperature is set (`:851-859`); no temperature → argmax.
4. Input χ kept only where fixed **and** non-NaN (`:907-914`), else sampled.
5. `ignore_chain_mask_zeros` writes `X` at unsampled positions (`:936-937`).
6. `fs_sequence_temp` set → global sequence temp defaults to 1e-6 (`run_inference.py:541-544`).
7. `ignore_ligand` → empty lig-prot neighbourhood; masked softmax returns 0; result equals
   upstream empty-ligand output.
8. `repack_only` ⇒ `chain_mask=1` and repack-all (`run_inference.py:729-740`) — single knob.
9. Deterministic E2E: injected decoding order + argmax seq/χ must match upstream exactly.

### 5.4 Purposes

| purpose / mode | Stages / axes |
|---|---|
| `sample` | encode → joint AR decode → RotamerBuilder (post) → sinks; axis `samples` |
| `sample` tied (`tied_second_input` set) | encode both structures (equal L else `ValueError`), shared decoding order, sample from `λ·P1+(1−λ)·P2` (probability space) per step; knobs ignored by upstream `tied_sample` (T0.2 verifies list: min-p, fs temp, disable_charged) → `ValueError` if set together |
| `score` / `nll`,`logits` | teacher-forced seq+χ log-probs (`get_logits_for_score` port) — **primary parity surface** |
| `score` / `proofread_unconditional` | one forward, `return_unconditional_probabilities`, softmax; residues = first shell or `selection_string` |
| `score` / `proofread_conditional` | axes `focus_residue` (first shell / selection) × `decoding_order` (`n_decoding_orders`) × `dropout_seed` (`n_dropouts`); each cell = single-residue-designable `sample` at T=1, χT=1, dropout **on**, graph with `num_adjacent_residues_to_drop=6`; Fuse: mean over orders then dropouts, plus `std` output |
| `jacobian`, `inspect` | `ValueError` in v1 |

`output_kind` for LASEr score: `nll | logits | proofread_unconditional | proofread_conditional`.

### 5.5 Host featurizer (task B0)

Per-residue B-factors (`fix_from_bfactor` requires {0,1}), side-chain coords + χ (NaN → mask),
ligand elements/H/water (`use_water`)/ncAA-as-ligand (`noncanonical_aa_ligand`), φ/ψ, SS/exposure
for budget, alpha-hull first shell (`fs_calc_ca_distance`, `fs_calc_burial_hull_alpha_value`,
`fs_no_calc_burial`). Proxide loader (`prep.py:102-122`) lacks B-factors → B0 either extends the
proxide call or parses via the same upstream-equivalent path (ProDy) in the driver's `batches()`.
Gate: field-level equality vs upstream `BatchData` on 5 fixtures (`example_pdbs/4jnj-1_prot.pdb` +
4 from `databases/` pinned by path+SHA-256).

### 5.6 Sinks / result schema

`sequence`, `seq_log_prob`, `chi_deg (N,L,4)`, `chi_mask`, `sidechain_coords (N,L,A,3)` opt-in,
`proofread_mean/std (R,21)`, PDB writer (side chains; hydrogens per checkpoint `build_hydrogens`),
FASTA (`output_fasta`, `output_fasta_only`).

## 6. Knob surface and redsox coverage

### 6.1 Options (flat — no nested dataclasses, so `check_superset` sees every name)

`PottsMPNNOptions`: `optimization_mode: Literal["none","potts","potts_converge","nodes"]="potts"`,
`optimization_temperature: float=0.0`, `binding_energy_optimization: Literal["none","both","only"]`,
`binding_energy_json: str|None`, `binding_energy_cutoff: float=8.0`, `ddg: bool=True`,
`mean_norm: bool=False`, `filter_nan: bool=False` (no-op, §6.5), `mutant_fasta: str|None`,
`mutant_csv: str|None`, `exclude_chains: Sequence[str]`, `pssm_json: str|None`,
`pssm_threshold: float`, `pssm_multi: float`, `pssm_log_odds_flag: bool`, `pssm_bias_flag: bool`,
`bias_by_res_json: str|None`, `tied_beta: Sequence[float]|None`, `tied_epistasis: bool`,
`skip_gaps: bool`, `optimize_pdb: bool`, `optimize_fasta: str|None`, `write_pdb: bool=True`,
`emit_etab: bool=False`, `emit_dense_hJ: bool=False`, `chain_design_mask_json: str|None`.

`LaserOptions`: `fs_sequence_temp`, `chi_temp`, `seq_min_p`, `chi_min_p`,
`disabled_residues=("X","C")`, `disable_charged_fs`, `repack_only`, `fix_from_bfactor`,
`ignore_ligand`, `use_water`, `noncanonical_aa_ligand`, `fs_calc_ca_distance=10.0`,
`fs_calc_burial_hull_alpha_value=9.0`, `fs_no_calc_burial`, `ala_budget=4`, `gly_budget=0`,
`constrain_ala_gly_to_exposed_non_ss`, `budget_residue_selection`, `ignore_chain_mask_zeros`,
`tied_second_input`, `tied_interpolation_lambda=0.0`, `selection_string`,
`n_decoding_orders=10`, `n_dropouts=10`, `num_adjacent_residues_to_drop=6`, `strict_load=True`,
`output_fasta`, `output_fasta_only`. (Exact defaults transcribed from upstream by the extractor;
the list above is indicative and is overridden by the generated alias table.)

### 6.2 Reference surfaces (generated)

`scripts/redsox/extract_upstream_knobs.py` (bathos-tracked, pure AST, no torch):
- every upstream `*.py` with `argparse` (glob, not a hand list) → `add_argument` dests;
- `sample_model` / public sampling-entrypoint parameters;
- PottsMPNN: every `cfg.inference.X` / `cfg.model.X` attribute read and `'X' in cfg.inference`
  test (AST), unioned with the example-YAML keys (catches `tied_epistasis`, `exclude_chains`).
Emits `tests/redsox/reference_surfaces.py`: one frozen dataclass per entry point with fields
**namespaced** `<entrypoint>__<dest>` (e.g. `laser_run_inference__sequence_temp`), header with
upstream SHA. A test regenerates and diffs (drift → fail).

### 6.3 Alias table (the coverage matrix)

`tests/redsox/alias_map.toml`, one row per reference field:
`{ref, target, equivalence ∈ {identical, semantic, exclusion}, parity_test_id, note}`.
Namespacing guarantees **no implicit name matches** — every upstream knob needs a row.
Known non-identical mappings: `chain_dict_json` → designed/visible chain masks
(`fixed_mask` + `chain_design_mask_json`), **not** `chain_id`; `fix_decoding_order` +
`decoding_order_offset` → `decoding_order_fn` + `random_seed` (**semantic**, seeded-order test);
Potts `noise` (eval-time `augment_eps` on all backbone atoms) → `noise` direct mode (**semantic**,
knob test verifies iid Gaussian on N/CA/C/O); LASEr `bb_noise` → `noise`; `repack_all` →
implied by `repack_only` (single knob); `disable_inference_dropout` → exclusion `internal`
(proofreading requires dropout on); `sequence_temp`/`temp`/`temperature` → `temperature`;
`designs_per_batch`/`max_tokens`/`inputs_processed_simultaneously` → `batch_size` (**semantic**,
tiling-only, parity test = output invariance to batch size); `model_weights`/`check_path` →
`checkpoint_id`|`model_local_path`; `device` → exclusion `device`; `verbose`/`disable_pbar`/
`silent` → exclusion `io_only`; `model.*` arch keys, `graph_structure.*`, `build_hydrogens` →
exclusion `checkpoint_derived`; `filter` → `filter_nan` (identical name-mapped, no-op quirk).

### 6.4 Exclusions

Rows with `equivalence="exclusion"` carry `reason ∈ {io_only, device, visualization,
training_only, checkpoint_derived, no_op, internal, duplicate, deferred:<backlog#>}`. Tests:
every exclusion names a knob present in the regenerated reference surface (no stale exclusions);
every `deferred` cites a real backlog id (validated via `praxia backlog` MCP in the harness setup,
or a checked-in id list refreshed by T0.4). Deferred set (v1): `entropy_decoder`, `mutation_search.*`,
CA-only, MSA vocab-22, `run_predict_partial_charges.*`, LigandMPNN-in-LASEr-repo entry points
(`run_inference_ligandmpnn.py`, `run_batch_inference_ligandmpnn.py` → reason `duplicate`: aminx
already implements LigandMPNN; T0.2 confirms they drive a LigandMPNN not LASEr model).

### 6.5 Upstream quirks preserved (U2 bit-parity)

| Quirk | Anchor | Knob | Test |
|---|---|---|---|
| PSSM mixing fires whenever `pssm_bias` non-empty (precedence) | `potts_mpnn_utils.py:1392` | `pssm_*` | `knob_semantics_pssm_precedence` |
| `potts_converge` accumulates absolute energy → ~1000 sweeps | `run_utils.py:116-120,176` | `optimization_mode` | `knob_semantics_potts_converge` |
| T=0 → floor + multinomial | `sample_seqs.py:38-39` | `optimization_temperature`, `temperature` | `knob_semantics_t0_floor` |
| `filter` no-op (`nrgs != nan` always True) | `etab_utils.py:312` | `filter_nan` | `knob_semantics_filter_noop` |
| fs temp forces seq temp 1e-6 | `run_inference.py:541-544` | `fs_sequence_temp` | §5.3.1-6 |

### 6.6 Gate

Harness `tests/redsox/test_knob_superset.py`:
`check_superset(flatten(RunSpecification, SamplingSpecification, ScoringSpecification,
PottsMPNNOptions, LaserOptions), reference_surfaces.*, alias_map_as_redsox_dict)` must return
`SUPERSET-HYPOTHESIS-PASS`; plus: every non-exclusion alias row has a `parity_test_id` that
resolves to a collected pytest id (`pytest --collect-only`); every **new** Options field appears as a
`target` in ≥1 row whose `parity_test_id` is a `knob_semantics_*` test (the behavioural gate —
redsox U1 reachability is run only as a smoke check, it is presence-only). Command:
`bth run --project-slug aminx -- uv run --no-sync pytest tests/redsox -q` with sidecar
`tests/redsox/knob_superset.bth.toml` (pass: all three assertions; fail: any). T0.4 runs it against
**current** aminx first (expected FAIL; the missing list is the implementation checklist).
Follow-up (filed in T0.4): redsox CLI multi-target + recursion + per-class alias maps.

## 7. Parity

### 7.1 Oracles

- Upstreams vendored read-only at pinned SHAs: `/home/marielle/repos/PottsMPNN` @ `0cb0a58`,
  `/home/marielle/repos/LASErMPNN` @ `e70f2c6d` (not `mistypotts/.tmp`, not `/tmp`).
- Oracle env: separate project `aminx-oracles/` (own `pyproject.toml`, torch + torch_scatter +
  torch_cluster + prody), run on titanix; never in aminx's env.
- `scripts/parity/dump_{potts,laser}_oracles.py` call `torch.manual_seed(0)` before model
  construction, record missing/unexpected keys, run each fixture in **float64 and float32**, and
  write sealed `.npz` (inputs, per-layer intermediates, outputs; teacher-forced and injected-order
  paths) + `oracle_manifest.toml` (upstream SHA, weights SHA-256, dump SHA-256, precision).
  Oracle generation is a bathos run (provenance; no finding).

### 7.2 Tier contract (mirror of xtrax `port/`, field-for-field)

- Per wave: `tests/port/targets/<wave>.toml` with `[port] wave_id, symbol_qualname, oracle_id
  ("ref:tests/port/reference/<wave>:v1:sha256:<hash>"), reference_subtree`, `[capabilities]
  stochastic, dynamic_shape`, `[parity] ad_critical=false, ad_critical_justification="",
  tolerance_policy="rtol=…,atol=…,matmul_precision=highest", max_traces`, `[access]
  reference_write_identities, fixer_read_only_on_reference=true`; `tests/port/manifests/<wave>.toml`
  with `manifest_hash` verified by conftest.
- Sealed reference: `tests/port/reference/<wave>/algo.py` (`# REFERENCE: DO NOT MODIFY`) loads the
  sealed `.npz` by hash (keeps torch out of aminx) and exposes the oracle callable the conftest
  imports, exactly like xtrax `_import_reference_algo`.
- Active wave via `AMINX_PORT_WAVE` (xtrax uses one active target; aminx iterates waves in CI by
  re-invoking pytest per wave). Markers `tier_1..tier_5` registered in aminx `pyproject.toml`;
  conftest enforces T1→T2→T3→(T4 iff ad_critical)→T5 and emits `domain=port` verdict records to
  `.praxia/audits.jsonl` using the xtrax record shape.
- T5 jit invariance uses `max_traces` (recompile counter).
- Follow-up: propose `xtrax.port` pytest plugin in the wheel; aminx then drops its copy.
- **Self-test first (T0.3):** a deliberately wrong kernel (sign-flipped) must fail T2 and a
  retrace-per-call kernel must fail T5 — the contract is proven to fire before any real wave.

| Wave | Symbol | stochastic | tolerance_policy (f64 T2 / f32 T3) | max_traces |
|---|---|---|---|---|
| `potts_head` | `PottsHead` | no | rtol=1e-10,atol=1e-12 / rtol=1e-5,atol=1e-6 | 1 |
| `potts_merge_pair_d2` | `merge_pair(denom=2)` | no | rtol=0,atol=0 / rtol=0,atol=1e-7 | 1 |
| `potts_merge_pair_d4` | `merge_pair(denom=4, exclude_self)` | no | same | 1 |
| `potts_energy` | `potts_energy` | no | rtol=1e-10 / rtol=1e-5 | 1 |
| `potts_pssm_mix` | `PSSMMix` (fixed probs in → probs out) | no | rtol=1e-10 / 1e-6 | 1 |
| `potts_refine` | `PottsRefine` with injected uniforms (RNG-free) | no | exact tokens | 1 |
| `pottsmpnn_full` | teacher-forced log-probs + etab_forward | no | atol 1e-8 / log-prob max\|Δ\| 1e-4 | 1 per bucket |
| `laser_layers` | GATv2/GVP/LN | no | rtol=1e-9 / rtol=1e-5 | 1 |
| `laser_encoder` | encoders | no | rtol=1e-8 / rtol=1e-4 | 1 per bucket |
| `laser_score` | teacher-forced seq+χ log-probs | no | atol 1e-8 / 1e-4 nats | 1 per bucket |
| `laser_decode_step` | one AR step, injected uniforms | **yes** (T2/T3 on injected-uniform path) | exact token/bin | 1 |
| `laser_rotamers` | RotamerBuilder | no | atol 1e-9 Å / 1e-4 Å | 1 |

`ad_critical=false` for all v1 waves (no v1 gradient consumer; follow-up filed). Tolerances are
fixed in the wave TOML **before** the first run; loosening requires a new bathos run citing the
measured deviation.

### 7.3 Behavioural parity (bathos sidecars, committed before running)

Invocation `bth run --project-slug aminx -- uv run --no-sync python3 scripts/parity/<name>.py …`
(titanix for heavy runs; L1/L2 local gates first); verification by record
(`bth compact`; `bth sql "SELECT id,status,outcome,exit_code,command FROM runs WHERE id LIKE '<p>%'"`)
and spot-checking output paths. Every sidecar declares `pass`, `fail` (residual), and
`inconclusive` (the band between) outcomes, a **measured-path negative control** that must FAIL
(else outcome `instrument_invalid`), and where applicable a **synthetic ground truth**.

| Sidecar | Data (pinned path + SHA-256 in TOML) | Metric / n | pass | inconclusive | fail | Negative control / ground truth |
|---|---|---|---|---|---|---|
| `potts_energy_parity` | `PottsMPNN/inputs/example_pdbs/*` + 200 random seqs/structure | max rel \|ΔE\| | ≤1e-5 | (1e-5,1e-4] | >1e-4 | permute `etab_out` rows → must fail; hand-built 3-residue etab with analytic E(seq) must match to 1e-12 (f64) |
| `potts_ddg_megascale` | `energy_benchmark_datasets/megascale_test_subset.csv` (n stated in TOML, all rows) | max \|Δddg\| vs upstream-run-by-us | ≤1e-4 | (1e-4,1e-3] | >1e-3 | skip transpose in `merge_pair` → must fail |
| `potts_refine_parity` | example_pdbs, 50 seeds, injected uniforms | exact sequence match rate | =1.0 | [0.99,1.0) | <0.99 | wrong partition sign → fail |
| `potts_sample_ar` | example_pdbs, n=1000 samples/condition × 3 temps | TOST on per-position TV distance vs upstream-vs-upstream null (margin in TOML) | equivalent at α=0.05 | neither | non-equivalent | swap two alphabet columns → must fail |
| `laser_score_parity` | `4jnj-1_prot.pdb` + 20 complexes from `databases/` (list pinned) | max \|Δ log-prob\| | ≤1e-4 | (1e-4,1e-3] | >1e-3 | permute one decoder layer → fail |
| `laser_decode_e2e` | same 21, injected order, argmax | exact seq + χ-bin match | =1.0 | — | <1.0 | — |
| `laser_sample_dist` | same 21, n=1000/structure × 3 temps | TOST on seq-recovery and χ1 circular MAE vs upstream-vs-upstream null | equivalent | neither | non-equivalent | disable χ feedback → fail |
| `laser_proofread_parity` | 5 complexes | max \|Δ mean prob\| (dropout via injected masks) | ≤1e-4 | (1e-4,1e-3] | >1e-3 | — |
| `knob_semantics_<knob>` | per knob fixture | per test | per test | — | — | per test |

## 8. Fixer task decomposition

| ID | Task | Depends | Gate |
|---|---|---|---|
| T0.1 | Vendor upstreams at pinned SHAs; `aminx-oracles/` env on titanix | — | files + manifest |
| T0.2 | Probe report (§10 open items): helper reuse list for `run_family_driver`; full `model_family` consumer grep; PottsMPNN missing/unexpected keys per checkpoint; ft config; proteinmpnn_compatible equivalence; T floor value; `optimize_sequence`/`nodes`/tied transcription; `:1391-1404` transcription; aminx default decoding order vs `mpnn_randn`; LASEr model_params dims, ligand-encoder keys, every `radius_graph`/`scatter_*` site; `tied_sample` ignored knobs; LigandMPNN entry points in LASEr repo | T0.1 | report appended to §10 |
| T0.3 | `tests/port/` contract (targets/manifests/reference/conftest/markers) + self-test (wrong kernel fails T2; retracing kernel fails T5) | — | `uv run pytest tests/port/selftest` on titanix |
| T0.4 | Extractor + reference surfaces + alias table skeleton + exclusions + harness; run vs current aminx (expected FAIL recorded); file redsox follow-up + deferred backlog items | T0.1 | bathos run FAIL baseline |
| T0.5 | `FamilyDriver` protocol + `FAMILY_DRIVERS` + runner dispatch + `run_family_driver` + family Literal/derivation + consumer branches + lint L-DRV (no Python loops over data axes / bare vmap in `aminx.families.*`) + `aminx.potts` import ban | T0.2 | unit tests; `uv run pytest -m parity_targeted` (titanix) byte-identical MPNN outputs |
| T0.6 | `output_kind`, Options fields, `omit_aa*`, CLI flags, `spec_json`/campaign round-trip, fail-loud validation | T0.5 | unit tests |
| A1 | PottsMPNN oracle dumps (f64+f32) | T0.1, T0.2 | manifest |
| A2 | `PottsMPNN` model + conversion (bathos) + self-edge test | T0.5, A1 | wave `pottsmpnn_full` |
| A3 | `PottsHead`, `merge_pair`, `potts_energy`, alphabet conversion | A1 | waves `potts_head`, `potts_merge_pair_d2/d4`, `potts_energy` |
| A4 | `PottsMPNNDriver` score energy/ddg + mutants/DMS + partitions axis + score sink | A2, A3, T0.6 | `potts_energy_parity`, `potts_ddg_megascale` |
| A5 | `PSSMMix`, `mpnn_randn`, `PottsRefine` (potts/converge/nodes/binding/tied) + all Potts Options | A4 | waves `potts_pssm_mix`, `potts_refine`; `potts_refine_parity`, `potts_sample_ar`; knob tests |
| A6 | ADR + lint updates | A2 | lint green |
| B0 | LASEr host featurizer | T0.1, T0.2 | field-level parity vs `BatchData` (5 fixtures) |
| B1 | LASEr oracle dumps | T0.1, T0.2 | manifest |
| B1.5 | LASEr weight conversion (bathos) | B1 | key-coverage report (0 unmapped) |
| B2 | layers | B1.5 | wave `laser_layers` |
| B3 | encoders + ligand bucket axis | B0, B2 | wave `laser_encoder` |
| B4 | `LaserDriver` score (teacher-forced) | B3, T0.6 | wave `laser_score`; `laser_score_parity` |
| B5 | joint AR decode stage + §5.3.1 edge semantics | B4 | wave `laser_decode_step`; `laser_decode_e2e`; `laser_sample_dist` |
| B6 | RotamerBuilder (post) + PDB/FASTA sinks | B5 | wave `laser_rotamers` |
| B7 | proofreading (both modes) + two-structure tied + remaining Laser Options | B5 | `laser_proofread_parity`; knob tests |
| Z1 | redsox gate PASS on final tree | A5, B7 | bathos run PASS |
| Z2 | CLI docs + `using-aminx` skill | Z1 | — |

## 9. Risks

| Risk | L | Mitigation |
|---|---|---|
| `run_family_driver` helper extraction perturbs MPNN path | M | byte-identical `parity_targeted` gate in T0.5 |
| merge_pair asymmetric-kNN edge cases (two conventions) | H | exact-match waves d2/d4 with adversarial kNN fixtures |
| LASEr scan compile time (L × n_dec·K) | M | O(K) step, jaxpr guard; bucketed L |
| Ragged ligand atoms / partition L recompiles | M | xtrax bucket axes; T5 `max_traces` |
| torch_scatter/cluster oracle env | M | separate `aminx-oracles/` on titanix; sealed dumps |
| Semantic mismatch behind name aliases | H | namespaced refs + mandatory parity id per row + `knob_semantics_*` for all new fields |
| Stochastic parity false pass | M | TOST vs upstream-vs-upstream null; measured-path negative controls; injected-uniform exact paths |
| Upstream `strict=False` hides random weights | M | T0.2 key audit blocks |
| Oracle fixtures large | L | store dumps on titanix + hash in repo; fetch in CI by hash |

## 10. Probe report

Resolved during authoring (260929):

| Item | Answer | Anchor |
|---|---|---|
| `EncoderOutput` has post-encoder edge embeddings | yes: `edge_features (S,L,K,D)`, `neighbor_indices` | `types/bundles.py:457-473` |
| aminx CA featurizer | no → CA-only deferred | — |
| aminx bias/omit/fixed names | `bias`, `fixed_positions`, `fixed_tokens`, `fixed_mask`; no omit → `omit_aa*` added | `run/specs.py:~570-590` |
| Temperature shape | `Sequence[float] | float` | same |

Open for T0.2: see T0.2 row.
