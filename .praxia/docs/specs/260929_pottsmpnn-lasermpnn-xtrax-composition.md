---
title: PottsMPNN + LASErMPNN as xtrax-composed model families on the central runner
description: Port KeatingLab PottsMPNN and polizzilab LASErMPNN into aminx as StageSet-composed families invokable via aminx.host.runner, with redsox knob-superset, xtrax-tier parity, and bathos-preregistered gates
task_id: 260929_potts-laser-xtrax-compose
status: draft-r0
created: 260929
supersedes_partially: decisions/260605_potts-parallel-not-stageset.md (scope-narrowing amendment, see §3)
---

# PottsMPNN + LASErMPNN as xtrax-composed model families

## 0. Goal and non-goals

**Goal.** A user can run

```bash
aminx run sample --inputs x.pdb --model-family pottsmpnn --checkpoint-id pottsmpnn_vanilla_20 ...
aminx run score  --inputs x.pdb --model-family pottsmpnn --output-kind energy ...
aminx run sample --inputs lig.pdb --model-family lasermpnn --checkpoint-id laser_0p1A_nothing_heldout ...
```

through the **one** central runner (`cli.py:run_sample/run_score` → `host/runner.py:sample/score`
→ `host/prep.py:prep_protein_stream_and_model` → `io/weights.py:load_model` →
`host/plan.py:make_inference_plan` → `StageSet` → xtrax-dispatched axes → sinks), with every
input knob of the upstream implementations reachable from `RunSpecification`, and with numerical
parity against the pinned upstream PyTorch code demonstrated by pre-registered bathos runs.

**Non-goals (v1).**
- Training / fine-tuning either model in aminx (inference + scoring only; weights are converted).
- LASErMPNN entropy-based decoding (`--ebd`) — O(L²) re-decode per step; deferred to a follow-up
  item with a jaxpr complexity guard (see memory: complexity guards). Listed as an explicit
  redsox exclusion (§6.3) so the gap is visible, not silent.
- PottsMPNN `mutation_search.py` (recursive beam search over mutants) — host-side combinatorial
  driver over `score(output_kind="energy")`; deferred, exclusion recorded.
- Replacing or migrating the existing TRW `aminx.potts.PottsModel` (mistypotts structure model).
  It stays a parallel family (§3).
- Visualization / plotting knobs (`chain_ranges`, `plot_dir`, proofreading heatmaps).

## 1. Recon summary (evidence base)

| Fact | Anchor |
|---|---|
| Central runner entry points | `src/aminx/cli.py` `run_sample`/`run_score`/`run_jacobian`/`run_inspect` → `src/aminx/host/runner.py` `sample()`/`score()`/`jacobian()`/`inspect()` |
| Family selector is a closed Literal | `src/aminx/run/specs.py:236` `model_family: Literal["proteinmpnn","ligandmpnn"] \| None`; auto-derived in `__post_init__` |
| Model class dispatch | `src/aminx/io/weights.py` `get_topology_for_checkpoint` (`model_type` ∈ protein/ligand/packer) and `load_model` |
| Plan factory | `src/aminx/host/plan.py:836` `make_inference_plan(model, spec, packer, *, purpose)` → `make_stage_set` + `make_encode_fn` + `make_decode_fn` |
| Stage bag | `src/aminx/types/stages.py:389` `StageSet` (`logit_transform`, `ar_logit_transform`, `decode_step`, `sample_step`, `tie_group_fuse`, `encoder_sink`, `axis_boundaries`, `encoding_fusion`, `decoding_fusion`) |
| Model protocol | `src/aminx/types/protocols.py:124` `ModelProtocol` (`features`, `encoder`, `decoder`, `w_out`, `w_s_embed`, `capabilities`, `stage_schema()`) |
| Capabilities | `src/aminx/model/capabilities.py` `ModelCapabilities(is_ligand_model, encode_fn_supports_structure_mapping)` |
| Packer is a post-decode vmap, not interleaved | `src/aminx/host/plan.py:697-719` `_run_packer_vmap` |
| Existing Potts code is the **TRW structure model**, not PottsMPNN | `src/aminx/potts/model.py:71` `PottsModel` = `node_lin`/`project_h`/`project_j` + `DifferentiableTRW`; no MPNN encoder, no AR decode |
| Potts isolation ADR + lint | `.praxia/docs/decisions/260605_potts-parallel-not-stageset.md`; `tests/lint/test_potts_import_boundary.py` |
| Upstream PottsMPNN | KeatingLab/PottsMPNN @ `0cb0a58`, checkout at `/home/marielle/projects/mistypotts/.tmp/PottsMPNN`; `potts_mpnn_utils.py:1225` `class PottsMPNN` |
| PottsMPNN = ProteinMPNN + edge head | `potts_mpnn_utils.py:1237-1261`: `ProteinFeatures`/`CA_ProteinFeatures`, `W_e`, `W_s`, `EncLayer`×3, `DecLayer`×3, `W_out`, **`etab_out = nn.Linear(hidden_dim, potts_dim)`**; `forward` 1267-1290: `etab = etab_out(h_E)`, masked, slot-0 diagonalised, `merge_duplicate_pairE` |
| PottsMPNN config | `inputs/example_config_sample_seqs.yaml`: hidden 128, edge 128, potts_dim 400, layers 3, num_edges 48, vocab 21 — identical shape to aminx `v_48_*` ProteinMPNN |
| PottsMPNN alphabet | `potts_mpnn_utils.py:69` `'ACDEFGHIKLMNPQRSTVWYX'` (etab is 20×20, X excluded) |
| PottsMPNN sampling | AR `sample()` 1321 / `tied_sample()` 1490 (with `pssm_*`, `bias_by_res`, `omit_AA_mask`) + Potts refinement `run_utils.optimize_sequence` (`optimization_mode` ∈ `""`/`potts`/`potts_converge`) |
| PottsMPNN weights | `vanilla_model_weights/` (20, 30, msa_20), `soluble_model_weights/`, `ft_model_weights/potts_ft.pt`, `ca_model_weights/v_48_{002,010,020}.pt`, `proteinmpnn_compatible_model_weights/` |
| Upstream LASErMPNN | polizzilab/LASErMPNN @ `e70f2c6d765416f7e29d51bfd6d4e08496438878` (MIT); ephemeral recon copy at `/tmp/claude-1000/ref/LASErMPNN-main` |
| LASErMPNN architecture | `utils/model.py:107` `class LASErMPNN(ligand_encoder_params, node_embedding_dim, protein_edge_embedding_dim, chi_angle_rbf_bin_width, prot_prot_edge_rbf_params, lig_prot_edge_rbf_params, num_encoder_layers, num_decoder_layers, additional_ligand_mlp, num_laser_vectors)`; GATv2 (`HomoGATv2`/`HeteroGATv2`) + GVP equivariant layers; `torch_scatter`, `torch_cluster` deps (`model.py:4-5`) |
| LASErMPNN decode | `model.py:727 sample`, `460 tied_sample`, `943 sample_by_lowest_entropy`, `261 get_logits_for_score`; joint AR over (aa, χ1..χ4) with RotamerBuilder feeding designed side-chains back into later steps |
| LASErMPNN knobs | `run_inference.py:773-792` argparse (+ `run_batch_inference.py`, `run_inference_tied.py`, `run_proofreading.py`) and `sample_model(...)` kwargs |
| redsox U5 | `/home/marielle/projects/redsox/src/redsox/checkers/superset.py` `check_superset(target_dataclasses, reference_dataclasses, alias_map)` — field-name superset over dataclass/NamedTuple classes; config keys `vendor_source_root`, `vendor_reference_classes`, `superset_alias_map` (`core/config.py:42-44`) |
| xtrax port tiers | `/home/marielle/projects/xtrax/port/` (dev-only, **not in the wheel**): `port_target.toml`, sealed `reference/`, `tests/conftest.py` markers `tier_1..tier_5` (T1 dtype/shape, T2 float64, T3 float32, T4 gradient/AD iff `ad_critical`, T5 jit invariance), blocking order T1→T2→T3→(T4)→T5, `emit/port_emit.py` |
| aminx pins xtrax as a git wheel | `pyproject.toml:26` `xtrax[io] @ git+…@56a9f551` → `port/` is **not importable** from aminx |
| aminx parity infra | `src/aminx/parity/` (`evidence.py` metrics, `matrix.py`), `tests/parity/test_full_model_parity.py:32-36` tolerance style, markers `parity_targeted`/`parity_heavy` |
| bathos | `.bth.toml` slug `aminx`; parity sidecar template `scripts/ebm/collect_synthetic_parity_evidence.bth.toml`; existing `scripts/recapture/pottsmpnn_to_eqx.bth.toml` |

## 2. Architecture overview

Both models become **model families** selected by `RunSpecification.model_family`, loaded by
`io/weights.load_model`, and executed by the existing `make_inference_plan` / `StageSet`
machinery. No second runner, no second spec type.

```
                       ┌──────────── RunSpecification (model_family, checkpoint_id, knobs) ───────────┐
                       │   run_spec.sampling / run_spec.potts_mpnn / run_spec.laser (new subconfigs)  │
                       └───────────────────────────────┬──────────────────────────────────────────────┘
                                                       ▼
 prep_protein_stream_and_model ──► load_model ──► {Aminx | PrxteinLigandMPNN | PottsMPNN | LaserMPNN}
                                                       ▼
                     make_inference_plan(model, spec, purpose) — family-dispatched via FAMILY_PLANNERS
                                                       ▼
   encode_fn ─► [encoding_fusion] ─► decode_fn(decode_mode) ─► [post_decode stages] ─► sinks (Zarr)
   PottsMPNN:  MPNN encoder (reused)     AR decode (reused)     PottsRefine (optional)   potts_etab, energy
   LASEr:      LaserEncoder (new)        JointSeqChiAR (new)    —                        chi, sc_coords
```

### 2.1 Family registry (replaces string-sniffing growth)

Add `src/aminx/model/families.py`:

```python
class ModelFamily(eqx.Module):          # static-only
  name: str                              # "proteinmpnn" | "ligandmpnn" | "pottsmpnn" | "lasermpnn"
  model_type: str                        # topology key consumed by load_model
  build_skeleton: Callable[..., ModelProtocol]
  plan_builder: Callable[[ModelProtocol, Any, Any, str], InferencePlan]
  supported_purposes: frozenset[str]     # {"sample","score","jacobian","inspect"}
  supported_output_kinds: frozenset[str] # {"logits","sequence","energy","chi"}
  options_type: type | None              # PottsMPNNOptions | LaserOptions | None
MODEL_FAMILIES: Registry[ModelFamily]    # aminx.registry.Registry, like SAMPLERS
```

- `run/specs.py:236` Literal widens to `"proteinmpnn","ligandmpnn","pottsmpnn","lasermpnn"`.
  `__post_init__` auto-derivation gains `checkpoint_id` prefixes `pottsmpnn_`, `laser_`; an explicit
  family is never overridden (existing invariant preserved).
- `io/weights.get_topology_for_checkpoint` / `load_model` dispatch through `MODEL_FAMILIES[...]`
  for the two new families; the existing three `model_type` branches are **not** refactored in this
  epic (scope control; follow-up item filed).
- `make_inference_plan` keeps its signature; first line becomes
  `if family.plan_builder is not None and family.name in NEW_FAMILIES: return family.plan_builder(...)`
  — existing ProteinMPNN/LigandMPNN code path is byte-identical (guarded by existing parity tests).
- Unsupported `(family, purpose)` or `(family, output_kind)` → `ValueError` at spec construction
  (fail-loud; never a silent no-op — cf. the 2026-07-14 `model_family` silent-no-op incident).

### 2.2 `ModelCapabilities` additions

```python
emits_potts: bool = eqx.field(static=True, default=False)
emits_chi: bool = eqx.field(static=True, default=False)
joint_seq_chi_decode: bool = eqx.field(static=True, default=False)
```
Defaults keep existing constants unchanged.

## 3. ADR amendment (260605_potts-parallel-not-stageset)

New decision doc `decisions/260929_pottsmpnn-is-a-stageset-family.md`, status Accepted on spec
convergence, `amends: 260605_potts-parallel-not-stageset`:

- The 260605 decision governs **the TRW structure model** (`aminx.potts.*`). It stays parallel,
  and `tests/lint/test_potts_import_boundary.py` is unchanged.
- **PottsMPNN is a different model**: its encoder *is* the ProteinMPNN encoder and it has a
  ProteinMPNN `W_out` AR head. The 260605 rejection of Option II ("decode-step adapter") does not
  apply because PottsMPNN's AR decode is the MPNN decode verbatim. The rejection of Option III
  ("EncoderOutput widening") is **honoured**: etab is *not* added to `EncoderOutput`; it is computed
  by a head stage from `EncoderOutput`'s edge features at the point of use (§4.2).
- New code lives in `aminx.model.potts_mpnn` and `aminx.inference.potts_head`, **not** in
  `aminx.potts`. A new lint test asserts `aminx.model.potts_mpnn` does not import `aminx.potts`
  except the pure energy helper `aminx.potts.sampling.log_energy` (to share one energy convention);
  if that helper's convention differs (×2 directed-slot), the new module owns its own `potts_energy`
  and the parity test cross-checks the two (§7, T2).
- Naming: user-facing family is `pottsmpnn`; class `PottsMPNN`. The existing `PottsModel` docstring
  gets a one-line disambiguation pointer.

## 4. PottsMPNN design

### 4.1 Model

```python
class PottsMPNN(eqx.Module):            # satisfies ModelProtocol
  features: ProteinFeatures | CAProteinFeatures
  encoder: Encoder
  decoder: Decoder
  w_e, w_s_embed, w_out                 # identical roles/names to Aminx
  potts_head: PottsHead                 # eqx.nn.Linear(hidden_dim, 400)
  capabilities = ModelCapabilities(is_ligand_model=False, encode_fn_supports_structure_mapping=False,
                                   emits_potts=True)
```

- **Weights.** Encoder/decoder/`W_out`/`W_s`/`W_e`/features map onto aminx's existing ProteinMPNN
  eqx layout via the existing converter (`scripts/convert_weights.py`); only `etab_out.{weight,bias}`
  is new. T0 task verifies key-for-key that `vanilla_model_weights/pottsmpnn_20.pt` minus `etab_out.*`
  converts with the ProteinMPNN converter unchanged; if any key/shape differs, the converter gains a
  `family="pottsmpnn"` branch rather than a fork.
- **CA-only.** `ca_model_weights/v_48_*.pt` require `CA_ProteinFeatures`. aminx already ships CA
  ProteinMPNN? — T0 determines. If absent, CA-only PottsMPNN is **deferred** (exclusion recorded;
  it is not required for the vanilla/soluble/ft checkpoints).
- **vocab 22 (msa checkpoints).** `pottsmpnn_msa_20` uses vocab 22 per upstream config comments.
  `W_s`/`W_out` shape follows the checkpoint; alphabet mapping for index 21 is resolved in T0 and
  guarded by a T1 alphabet-conformance test extending `tests/test_alphabet_conformance.py`.
- **Checkpoint registry.** 7 converted `.eqx.zst` already exist at
  `mpnn_ext/external/aminx/weights/pottsmpnn/` — **provenance unknown (produced for the TRW
  model's `pottsmpnn_to_eqx.py`, which exports *etab-derived h/J*, not model weights)**. They are NOT
  reused. New conversions are produced by a bathos-tracked `scripts/recapture/pottsmpnn_model_to_eqx.py`
  and registered in the checkpoint registry with source SHA-256 + upstream commit.

### 4.2 Potts head and energy

`aminx.inference.potts_head`:

```python
class PottsHead(eqx.Module):
  linear: eqx.nn.Linear                                # hidden → 400
  def __call__(self, h_E: Float[Array,"L K H"], E_idx: Int[Array,"L K"], mask: Float[Array,"L"]
               ) -> Float[Array,"L K 20 20"]:
    # 1) linear; 2) mask rows; 3) slot-0 (self edge) × eye(20); 4) merge_duplicate_pairE
```

- `merge_duplicate_pairE` is ported as a pure gather over `E_idx` (reverse-edge lookup with a
  validity mask when j∉kNN(i)), matching upstream `etab_utils.merge_duplicate_pairE` semantics
  including the "only one direction present" case. This is the single highest-risk numeric piece
  → dedicated port wave `potts_merge_pairE` (§7).
- `potts_energy(etab, E_idx, seq, mask) -> Float[""]` reproduces upstream `etab_utils.calc_eners`
  (T2 oracle), including its directed-slot ×0.5 bookkeeping. Energy sign convention: upstream
  energies are returned **as upstream reports them**; `score(output_kind="energy")` documents the
  sign and `output_kind="log_prob"` remains the MPNN AR log-prob, never conflated.
- **No EncoderOutput widening.** The head consumes the edge features already present in
  `EncoderOutput` (verify field name in T0: aminx's encoder must return `h_E` post-encoder; if the
  encoder currently discards `h_E`, the minimal change is exposing it in the existing output —
  this is the one place §3's Option-III honour is tested and must be raised as a blocking finding
  if widening is unavoidable).

### 4.3 Stages and purposes

| purpose / output_kind | Stages | Notes |
|---|---|---|
| `sample` / `sequence` | reused MPNN encode → reused AR decode (`W_out`) | knobs: temperature, bias/omit, pssm_*, bias_by_res, fixed, tied (existing `tie_group_fuse`) |
| `sample` + `optimization_mode ∈ {potts, potts_converge}` | + `PottsRefine` post-decode stage | fixed-shape `lax.fori_loop` (`potts`, one sweep in decoding order) / `lax.while_loop` with `max_sweeps` cap (`potts_converge`) |
| `score` / `energy` | encode → `PottsHead` → `potts_energy` | no decode; deterministic |
| `score` / `ddg` | energy(mut) − energy(wt) per `(pdb, mutant)` row; `mean_norm`; binding mode = E(complex) − Σ E(partitions) | partitions → an xtrax axis `partition` with a `difference_fuse`-style Fuse (precedent: `ebm/dispatch.py` `difference_fuse`) |
| `score` / `logits` | reused conditional/unconditional score | identical to ProteinMPNN path |
| `jacobian` | reused | Potts head not involved |

`PottsRefine` is a new optional `StageSet` slot `post_decode: tuple[PostDecodeFn, ...]`
(default `()`), analogous to `encoder_sink`; the existing packer vmap is **not** migrated into it in
this epic. `optimize_sequence` semantics to port exactly: per-position energy
`positional_potts_energy`, sampling from `exp(-E_i/T)` at `optimization_temperature` (T=0 → argmin),
respect fixed/omit/bias/pssm masks, binding mode subtracts unbound-partition energies.

### 4.4 Sinks

Zarr sink arrays (xtrax `ZarrStagingSink`): `potts_etab (N,L,K,20,20) float16|float32` (opt-in,
`potts_mpnn.emit_etab`, default False — 48×400 floats/residue), `potts_E_idx`, `potts_energy (N,)`,
`ddg (N,)`. `emit_dense_hJ: bool` converts to dense `(L,L,20,20)` host-side only (never in-graph).

## 5. LASErMPNN design

### 5.1 Model (new family, no weight sharing)

`aminx.model.laser` package:

| Upstream | aminx | Port notes |
|---|---|---|
| `LigandFeaturizer`, `LigandEncoderModule` (`model.py:1150,1312`) | `LaserLigandEncoder` | pretrained weights load from the main checkpoint's `ligand_encoder.*` keys (T0 confirms the main `.pt` already contains them; `pretrained_ligand_encoder_weights.pt` is a training-time init only) |
| `HomoGATv2`, `HeteroGATv2`, `GVP`, `DenseGVP`, `EquivariantLayerNorm` (`model_generics.py`) | `aminx.model.laser.layers` | edge-list attention → **dense padded neighbour axis** `(L, K)` with mask; `scatter_softmax` → masked softmax over K; vectors kept as `(…, V, 3)` |
| `LASErMPNN_Encoder`/`_Decoder` | `LaserEncoder`, `LaserDecoderLayer` | |
| `RotamerBuilder` (`build_rotamers.py`) | `aminx.model.laser.rotamers` | ideal geometry buffers become static arrays loaded from `files/rotamer_alignment.pt`, `new_ideal_bond_lengths.pt` (converted to `.npz`, hashed); NaN-as-missing → explicit `chi_mask (L,4)` |
| chi heads (`chi_prediction_layers`, `chi_offset_prediction_layers`, `chi_vector_update_layers`) | `LaserChiHead` | 72 bins @5°; angle = wrap(bin_center + offset) |

Graphs (host-side prep, numpy, then fixed-shape arrays):
- protein–protein kNN on Cα, `k_pp` from checkpoint `model_params` (default 16);
- ligand–ligand kNN `k_ll` (default 5); ligand–protein kNN `k_lp` (16) **with** 15 Å cutoff → mask;
- ligand atoms padded to a bucket `A_max` from an xtrax `AxisSpec(name="ligand_atoms", bucket_boundaries=…)`
  so ragged ligand counts do not recompile per input (xtrax Bucket strategy).
- `torch_cluster.radius_graph` usages → dense distance + top-k/cutoff mask; T0 enumerates every call.

### 5.2 Joint sequence+χ autoregressive decode

New decode mode `JointSeqChiAutoregressive` in `aminx.inference.decode` (registered with
`make_decode_fn`), selected when `capabilities.joint_seq_chi_decode`:

- `lax.scan` over decoding order (fixed length L, masked); carry =
  `(seq (L,), chi (L,4), chi_mask (L,4), sc_coords (L,A_sc,3), node_state)`.
- Per step t: decoder layers on current carry → seq logits → temperature (global or first-shell
  `fs_sequence_temp` via `first_shell_mask`) → min-p warp (`seq_min_p`) → disabled residues /
  `disable_charged_fs` / ALA-GLY budget masks → categorical sample (or argmax when temp absent) →
  χ1..χ4 sequential (χ_k conditioned on χ_<k RBF) with `chi_temp`, `chi_min_p` → RotamerBuilder
  places side-chain atoms for residue t → carry update so later steps see them.
- **ALA/GLY budget** is a running count in the carry (upstream uses scatter over batch); budget
  region mask is a host-computed input from `budget_residue_sele_string` (ProDy selection evaluated
  host-side) or the exposed-non-SS heuristic.
- Fixed residues (`fix_beta`, `chain_mask`) and `repack_only`/`repack_all` switch per-position
  between "sample aa", "keep aa, sample χ", "keep both".
- `score` purpose: `get_logits_for_score` port → teacher-forced seq + χ log-probs (deterministic
  given decoding order) — **this is the primary parity surface** (§7).
- Proofreading (`run_proofreading.py`) = `score` with two xtrax axes `decoding_order (n=10)` and
  `dropout_seed (n=10)`, dropout **enabled** via an `inference_dropout: bool` knob, reduced by a
  mean Fuse over both axes → per-position probability + entropy outputs. Plots are out of scope.
- Tied sampling (`run_inference_tied.py`, `tied_probability_interpolation_lambda`) maps onto the
  existing `tie_group_fuse` slot with a new `InterpolatedTieFuse(lambda_)`; T0 must confirm the
  upstream tie semantics (probability interpolation between two tied positions) fit the
  `(L,V),mask → (V,)` signature; if not, raise a blocking finding.

First-shell determination (`fs_calc_ca_distance`, `fs_calc_burial_hull_alpha_value`,
`fs_no_calc_burial`) is host-side prep (scipy/alphashape as upstream) producing `first_shell_mask`.

### 5.3 Sinks

`sequence`, `seq_log_prob`, `chi (N,L,4)`, `chi_mask`, `sidechain_coords (N,L,A,3)` (opt-in),
PDB writer reuses aminx designs IO with side-chain atoms (hydrogens only when `build_hydrogens`).

## 6. RunSpec knob surface and redsox coverage

### 6.1 New sub-configs (frozen dataclasses, `run/spec.py`, surfaced on `RunSpecification`)

`RunSpecification` gains two optional fields, `potts_mpnn: PottsMPNNOptions | None = None` and
`laser: LaserOptions | None = None`; `build_run_spec` threads them into `run_spec.potts_mpnn` /
`run_spec.laser`. Knobs with an existing aminx equivalent **map onto the existing field** (no
duplicates) — the alias map (§6.3) records the mapping.

`PottsMPNNOptions`: `optimization_mode: Literal["none","potts","potts_converge"]`,
`optimization_temperature: float`, `max_refine_sweeps: int`, `binding_energy_mode:
Literal["none","both","only"]`, `binding_partitions: Sequence[Sequence[str]] | None`,
`binding_energy_cutoff: float`, `ddg: bool`, `mean_norm: bool`, `filter_nan: bool`,
`pssm: PSSMOptions | None` (`coef/bias/threshold/multi/log_odds_flag/bias_flag`),
`bias_by_res: ArrayLike | None`, `emit_etab: bool`, `emit_dense_hJ: bool`, `skip_gaps: bool`,
`optimize_input_sequences: bool`.

`LaserOptions`: `fs_sequence_temp: float | None`, `chi_temp: float | None`, `seq_min_p: float`,
`chi_min_p: float`, `disabled_residues: Sequence[str] = ("X","C")`, `disable_charged_fs: bool`,
`repack_only: bool`, `repack_all: bool`, `fix_from_bfactor: bool`, `ignore_ligand: bool`,
`use_water: bool`, `noncanonical_aa_ligand: bool`, `fs_calc_ca_distance: float`,
`fs_calc_burial_hull_alpha_value: float`, `fs_no_calc_burial: bool`, `ala_budget: int`,
`gly_budget: int`, `constrain_ala_gly_to_exposed_non_ss: bool`, `budget_residue_selection: str | None`,
`tied_interpolation_lambda: float`, `inference_dropout: bool`, `n_proofread_orders: int`,
`n_proofread_dropouts: int`, `build_hydrogens: bool`, `ligand_atom_bucket_boundaries: tuple[int,...]`.

Mapped to existing fields: temperature → `run_spec.sampling.temperature`; `num_samples`/
`designs_per_input` → `num_samples`; `noise`/`bb_noise` → `noise` (`FeatureNoiseBundle`,
direct mode); `fixed_positions_json` → `fixed_mask`; `tied_positions_json` → `tied_positions`;
`omit_AAs`/`omit_AA_json`/`bias_AA_json` → existing bias/omit fields (T0 names them);
`fix_decoding_order`+`decoding_order_offset` → `decoding_order_fn` + `random_seed`;
`chain_dict_json` → `chain_id`; `model_weights`/`check_path` → `checkpoint_id`/`model_local_path`;
`output_path`/`out_dir` → `output_dir`; `designs_per_batch`/`max_tokens`/
`inputs_processed_simultaneously` → `batch_size` (xtrax planner decides tiling; recorded as
*semantic-equivalent*, not identical).

Each new field must be reachable (redsox **U1**) — a declared-but-unread knob is a defect.

### 6.2 Reference surfaces (generated, not hand-written)

`scripts/redsox/extract_upstream_knobs.py` (bathos-tracked, pure AST — no torch import):
- LASErMPNN: every `parser.add_argument` in `run_inference.py`, `run_batch_inference.py`,
  `run_inference_tied.py`, `run_proofreading.py` → dest names; plus parameters of `sample_model`.
- PottsMPNN: keys of `inputs/example_config_sample_seqs.yaml`, `example_config_energy_prediction.yaml`
  (flattened `model.*`/`inference.*`), plus argparse dests of `mutation_search.py`.
- Emits `tests/redsox/reference_surfaces.py` with one frozen dataclass per upstream entry point
  (`LaserRunInferenceKnobs`, `PottsSampleSeqsKnobs`, …) and a header recording upstream SHA; a test
  regenerates and diffs (drift → fail).

### 6.3 Coverage config

`.redsox/potts_laser.yaml`:
```yaml
target_package: aminx
target_source_root: src
target_dataclasses: [aminx.run.specs.RunSpecification, aminx.run.spec.PottsMPNNOptions,
                     aminx.run.spec.LaserOptions, aminx.run.spec.SamplingConfig]   # T0 confirms names
vendor_source_root: tests/redsox
vendor_reference_classes: [reference_surfaces.LaserRunInferenceKnobs, …]
superset_alias_map: { sequence_temp: temperature, check_path: checkpoint_id, … }
```
U5 has no exclusion mechanism, so exclusions live in `tests/redsox/knob_exclusions.toml`
(`knob`, `entry_point`, `reason ∈ {io_only, device, visualization, deferred:<backlog#>, training_only}`);
the generator **omits** excluded knobs from the reference dataclasses, and a test asserts every
exclusion names a knob that actually exists upstream (no stale exclusions) and every `deferred`
reason carries a real backlog id. Gate: `SUPERSET-HYPOTHESIS-PASS` + U1 zero unreachable new fields.

Alias caveat: U5 compares **field names only**. The alias map proves *presence*, not *semantics*;
semantic equivalence of each mapped knob is covered by a behavioural parity case (§7.3) — every
alias entry must cite the parity test id that exercises it (checked by a test over the alias map).

## 7. Parity plan (xtrax tiers + bathos)

### 7.1 Oracles

- Upstream sources vendored read-only at pinned SHAs to `/home/marielle/repos/PottsMPNN` (`0cb0a58`)
  and `/home/marielle/repos/LASErMPNN` (`e70f2c6d`) — **not** `mistypotts/.tmp` or `/tmp`.
- `scripts/parity/dump_{potts,laser}_oracles.py` run in a torch env (separate uv group
  `oracle-torch`, needs `torch_scatter`/`torch_cluster` for LASEr) and write sealed `.npz` oracle
  dumps (inputs, intermediates per layer, outputs, RNG-free teacher-forced paths) +
  `oracle_manifest.toml` (upstream SHA, weights SHA-256, dump SHA-256). Oracle dumps are
  provenance artifacts; their generation runs under bathos but produce no finding.

### 7.2 Tier ladder (xtrax contract adopted in aminx)

aminx cannot import xtrax `port/` (wheel pin). It adopts the **same contract**: pytest markers
`tier_1..tier_5` registered in aminx `pyproject.toml`, a `tests/port/conftest.py` enforcing blocking
order T1→T2→T3→(T4 iff `ad_critical`)→T5, and per-wave `tests/port/waves/<wave>.toml` mirroring
`port_target.toml` fields (`oracle_id` with SHA-256, `symbol_qualname`, `tolerance_policy`,
`ad_critical` + justification). A follow-up xtrax item proposes shipping the tier conftest as a
pytest plugin in the wheel (`xtrax.port`), at which point aminx switches imports (no semantics change).

| Wave | Symbol | ad_critical | tolerance (T2 f64 / T3 f32) |
|---|---|---|---|
| `potts_head` | `PottsHead` | no | 1e-10 / 1e-5 |
| `potts_merge_pairE` | `merge_duplicate_pairE` | no | exact / exact |
| `potts_energy` | `potts_energy` | **yes** (energy gradients used by `jacobian`/STE-style design) | 1e-10 / 1e-5 rel |
| `potts_refine` | `PottsRefine` (T=0 argmin path) | no | exact sequence match |
| `pottsmpnn_full` | `PottsMPNN` teacher-forced log-probs + etab | no | log-prob max\|Δ\| ≤ 1e-4 (existing `LOG_PROB_MAX_ABS_DEVIATION` style) |
| `laser_layers` | GATv2, GVP, LayerNorm | no | 1e-9 / 1e-5 |
| `laser_rotamers` | RotamerBuilder | **yes** (coords differentiable wrt χ) | 1e-9 / 1e-4 Å |
| `laser_encoder` | ligand + protein encoder | no | 1e-8 / 1e-4 |
| `laser_score` | teacher-forced seq+χ log-probs | no | max\|Δ\| ≤ 1e-4 nats |
| `laser_decode_step` | one AR step given carry | no | exact token/bin at argmax |

Tolerances are **pre-registered in the wave TOML before the first run** and may only be loosened
through a new bathos run citing the measured deviation (never edited post-hoc).

### 7.3 Behavioural / distributional parity (bathos sidecars, pre-registered)

Each is `scripts/parity/<name>.py` + `<name>.bth.toml`, committed before running, invoked as
`bth run --project-slug aminx -- uv run --no-sync python3 scripts/parity/<name>.py …`, verified by
record (`bth compact` then `bth sql "SELECT id,status,outcome,exit_code FROM runs WHERE id LIKE '<p>%'"`).

| Sidecar | Hypothesis | pass | fail (residual) |
|---|---|---|---|
| `potts_energy_parity` | aminx `score(energy)` equals upstream `calc_eners` on `inputs/example_pdbs` + 200 random sequences/structure | max rel err ≤ 1e-5 (f32) | > 1e-4 |
| `potts_ddg_megascale` | aminx ddG matches **upstream ddG run by us** (same weights, same subset) | Pearson(aminx, upstream) ≥ 0.9999 AND \|Δr_expt\| ≤ 0.005 | Pearson < 0.999 |
| `potts_sample_ar` | AR sampling marginals match upstream | per-position TV distance of 1000-sample empirical marginals ≤ bootstrap null 99th pct (upstream-vs-upstream two seeds) | > null 99.9th pct at >1% positions |
| `laser_score_parity` | teacher-forced seq/χ log-probs on `example_pdbs/4jnj-1_prot.pdb` + 20 ligand complexes | max\|Δ\| ≤ 1e-4 | > 1e-3 |
| `laser_sample_dist` | sampled seq recovery + χ1 MAE distributions match upstream at 3 temps | two-sample KS p > 0.01 per metric and \|Δmean\| within upstream seed-to-seed range | KS p < 1e-3 |
| `knob_semantics_<knob>` | one per alias-mapped knob with non-trivial semantics (temperature, fs temp, min-p, budgets, bias, omit, fixed, tied, noise) | behaviour matches upstream on a targeted fixture | — |

**Controls (mandatory, per `rules/BATHOS.md`).** Each numeric parity script runs a **positive
control** (upstream vs itself, different process → must pass) and a **negative control**
(aminx with one decoder layer's weights permuted → must FAIL). A script whose negative control
passes is a broken instrument; its sidecar outcome is `instrument_invalid`, not pass.

Heavy runs go to titanix / Engaging (never whole suites locally); L1/L2 local gates precede L3.

## 8. Fixer task decomposition

Phase A (PottsMPNN) precedes Phase B (LASEr); Phase 0 is shared.

| ID | Task | Depends | Gate |
|---|---|---|---|
| T0.1 | Vendor upstreams at pinned SHAs; oracle-torch uv group; record SHAs | — | files + manifest |
| T0.2 | Verification probes (answers the "T0 confirms" items in §4–§6): converter key map for pottsmpnn .pt; `h_E` in EncoderOutput; CA features presence; vocab-22 alphabet; LASEr checkpoint contains ligand encoder keys; every `radius_graph`/`scatter_*` call site; upstream tie semantics; existing bias/omit field names | T0.1 | written probe report appended to this spec (§10) |
| T0.3 | `tests/port/` tier contract (markers, conftest, wave TOML schema) + synthetic self-test (a deliberately wrong kernel fails T2) | — | pytest on titanix |
| T0.4 | redsox extractor + reference surfaces + exclusions + config; run U5 on **current** aminx (expected FAIL listing all missing knobs = the implementation checklist) | T0.1 | bathos run, FAIL recorded as baseline |
| T0.5 | `ModelFamily` registry + Literal widening + fail-loud unsupported purpose/output_kind | — | unit tests; existing parity_targeted green |
| A1 | Oracle dumps PottsMPNN | T0.1 | manifest |
| A2 | `PottsMPNN` model + weight conversion (bathos) | T0.2, T0.5 | wave `pottsmpnn_full` T1–T3,T5 |
| A3 | `PottsHead`, `merge_duplicate_pairE`, `potts_energy` | A1 | waves `potts_head`/`potts_merge_pairE`/`potts_energy` T1–T5 |
| A4 | `score(energy|ddg)` runner wiring + binding partitions axis + sinks | A2, A3 | `potts_energy_parity`, `potts_ddg_megascale` |
| A5 | `post_decode` slot + `PottsRefine` + `PottsMPNNOptions` knobs incl. pssm/bias_by_res | A4 | wave `potts_refine`; `potts_sample_ar`; knob sidecars |
| A6 | ADR amendment + new lint test | A2 | lint green |
| B1 | Oracle dumps LASEr | T0.1 | manifest |
| B2 | LASEr layers (GATv2/GVP/LN) | B1 | wave `laser_layers` |
| B3 | RotamerBuilder port | B1 | wave `laser_rotamers` (incl. T4) |
| B4 | Encoders + host graph prep + ligand bucket axis | B2 | wave `laser_encoder` |
| B5 | `score` (teacher-forced) + weight conversion | B3, B4 | wave `laser_score`; `laser_score_parity` |
| B6 | `JointSeqChiAutoregressive` decode + budgets/first-shell/fixed/repack | B5 | wave `laser_decode_step`; `laser_sample_dist` |
| B7 | Proofreading axes + tied interpolation + `LaserOptions` wiring + sinks/PDB | B6 | knob sidecars |
| Z1 | redsox U5 PASS + U1 clean on final tree | A5, B7 | bathos run PASS |
| Z2 | CLI docs + `using-aminx` skill update | Z1 | — |

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `h_E` not in `EncoderOutput` → Option-III widening pressure | M | T0.2 probe; minimal exposure only; escalate if bundle schema must change |
| merge_duplicate_pairE asymmetric-kNN edge cases | H | dedicated exact-match wave with adversarial kNN fixtures |
| LASEr joint AR scan compile time/memory (L steps × encoder-lite + rotamer build) | H | per-step work bounded to decoder layers + one residue rotamer build; jaxpr loop-body complexity guard (O(K) not O(L²) per step) |
| Ragged ligand atoms recompiles | M | xtrax bucket axis; recompile counter asserted in T5 |
| torch_scatter/cluster oracle env hard to install | M | oracle env on titanix only; dumps committed as fixtures (LFS/remote per size) |
| U5 name-only superset hides semantic mismatch | H | alias→parity-test citation check (§6.3) |
| Existing `mpnn_ext` pottsmpnn `.eqx.zst` mistaken for model weights | M | explicitly not reused (§4.1) |
| Stochastic sampling parity false-fail/false-pass | M | bootstrap null from upstream-vs-upstream; negative control |

## 10. Open questions / T0 probe report

Resolved during spec authoring (orchestrator probes, 260929):

| Item | Answer | Anchor |
|---|---|---|
| Does `EncoderOutput` carry post-encoder edge embeddings? | **Yes** — `edge_features (S,L,K,D)` + `neighbor_indices (S,L,K)`; PottsHead consumes these, no widening | `src/aminx/types/bundles.py:457-473` |
| aminx CA-only featurizer? | **No** (`rg ca_only\|CAProteinFeatures src/aminx` → 0 hits) → CA-only PottsMPNN deferred, exclusion `deferred:<filed in T0.4>` | — |
| aminx bias/omit/fixed field names | `SamplingSpecification.bias (L,V)`, `fixed_positions`, `fixed_tokens`; `RunSpecification.fixed_mask`; **no omit field** → add `omit_aa: Sequence[str]` + `omit_aa_per_position: Mapping[int, str] \| None` to `SamplingSpecification`, compiled host-side into a −inf-equivalent bias (upstream uses `-1e8·mask`; aminx uses the same finite constant for bit-parity) | `src/aminx/run/specs.py:~570-590` |
| Temperature shape | `SamplingSpecification.temperature: Sequence[float] \| float` (already a temperature axis) | same |

Still open for T0.2: converter key map for PottsMPNN `.pt`; vocab-22 index-21 semantics;
LASEr checkpoint ligand-encoder keys; enumerated `radius_graph`/`scatter_*` call sites; upstream
tied-interpolation semantics.
