---
title: PottsMPNN + LASErMPNN T0.2 probe report
description: Resolution of every T0.2 open item in the potts/laser xtrax-composition spec (runner helpers, model_family consumers, torch checkpoint key audits, verbatim upstream transcriptions, LASEr graph/scatter/draw/dropout enumerations, aminx float32 and self-slot behaviour) plus spec corrections found while probing
task_id: 260929_potts-laser-xtrax-compose
created: 260929
---

# T0.2 probe report

Spec: `.praxia/docs/specs/260929_pottsmpnn-lasermpnn-xtrax-composition.md` (status converged-r11).
Pinned upstreams: PottsMPNN `0cb0a58874e373d664114f1735deab77614e4ab5`, LASErMPNN
`e70f2c6d765416f7e29d51bfd6d4e08496438878` (local vendored copies at `/home/marielle/repos/{PottsMPNN,LASErMPNN}`,
copies on titanix at `/home/solab/repos/`: PottsMPNN checkout HEAD is `0cb0a58`; the LASEr copy has no `.git`, only `VENDOR_PIN.toml`). Every weights file the probes touched has a SHA-256 equal to the
`VENDOR_PIN.toml` entry (16/16 PottsMPNN, 4/4 LASEr; re-hashed on titanix).

Conventions. `file:line` anchors are against the pinned trees (upstream) or the worktree
`src/aminx` (aminx). Code blocks marked "verbatim" are extracted mechanically from the source files with the
source line number as a left gutter (`1466| ...`); the gutter is not part of the code.

Provenance / pre-registration status. This is a T0.2 static-read plus probe task, not a claim-tier experiment.
The torch probes below were run inline on titanix (`~/projects/aminx-oracles`, torch 2.4.1+cpu,
`torch.manual_seed(0)`, CPU) as ephemeral inspection scripts with no bathos sidecar. They are
**exploratory and cite nothing as a finding**: every number here that could be re-derived is either a key/shape/bit-equality
fact reproducible by the commands in section 11, or is flagged "exploratory, single fixture". Anything that a later
gate wants to cite must be re-run under a sidecar. The dump scripts in A1/B1 (`dump_{potts,laser}_oracles.py`) are the
tracked replacements for the key audits.

## 0. Summary: BLOCKING findings and spec corrections

BLOCKING (a non-empty `missing_keys` set, per spec section 4.1 rule, or a gate that cannot be built as specified):

- **B1. `proteinmpnn_compatible_model_weights/*` (6 files) have a non-empty missing set** (`etab_out.weight`,
  `etab_out.bias`) when loaded into upstream `PottsMPNN` with `strict=False`. They are ProteinMPNN-only weights. The
  spec's `duplicate` exclusion is correct (118/118 shared tensors bit-equal to the corresponding vanilla/soluble
  file) but they must stay excluded from the `pottsmpnn` family; loading one as a Potts model would silently run
  Xavier-random `etab_out` (`potts_mpnn_utils.py:1263-1265`). Section 3.
- **B2. LASEr B0/B1/`laser_score_parity` fixtures are not obtainable from the pinned repo.** `databases/` contains only
  `README.md` and `dataset_split_info.zip` (split JSON, no structures); `example_pdbs/` has only
  `4jnj-1_prot.pdb`. The "4 from `databases/`" (B0) and "20 complexes from `databases/`" (`laser_score_parity`)
  structures must come from Zenodo (README: 50 GB reassembled zip, two chunks; `download_protonated_pdb_training_dataset.sh`
  fetches `zenodo.org/records/15035128/files/all_data.zip`). Someone must pick and pin them (path + SHA-256) before B0. Section 6.11.
- **B3. Order-tier semantics (spec 5.4a) contradict the upstream inference path.** In upstream inference
  (`ProteinComplexData.output_batch_data`) `extra_atom_contact_mask` is all-False (`run_inference.py:350,371`), so the
  tier-2 (contact) set is always empty and `sample`/proofread orders are `fixed first, then all designable, random`.
  The spec says the same rule with `contact = extra_atom_contact_mask` (dataset definition, `pdb_dataset.py:1233-1239`)
  drives `sample`; implementing that would break `sample` order parity. Also the order vehicle fixture "fixed / non-contact /
  contact rows" cannot be produced through the upstream inference featurizer (must set `batch.extra_atom_contact_mask`
  by hand). Section 6.7 and Spec corrections C1.

Spec corrections (details and evidence in the last section): C1 order tiers (above); C2 first-shell knobs
`fs_calc_ca_distance`/`fs_calc_burial_hull_alpha_value`/`fs_no_calc_burial` are inert for the consumed mask;
C3 stored `seq_logits` are post-min-p upstream (spec says pre); C4 `bb_noise` semantics differ from Potts `noise`;
C5 LASEr T=0 handling differs between CLIs; C6 `disabled_residues` default provenance; C7 anchor
`LigandFeaturizer (model.py:1150,1312)` wrong; C8 dead knobs on the tied CLI; C9 tied-refine quirks the spec's quirk table
omits (binding reference subtraction, leaked `pos`, ZeroDivisionError, no per-member `chain_mask` skip in tied nodes);
C10 proofread `resindex` identity is assumed upstream, not guaranteed; C11 `make_axis_dispatch_via_xtrax` docstring stale;
C12 minor anchor notes.

Everything else in the T0.2 list resolved as recorded in the sections below.

## 1. Runner helper names and signatures (spec 2.1 "Reused as-is")

All confirmed present; no extraction refactor is needed to call them from a driver.

| Helper | Signature and anchor |
|---|---|
| `_canonical_structure_id` | `_canonical_structure_id(input_item: Any, index: int) -> str` at `src/aminx/host/_sampling_helper.py:22` (Path/str -> stem, else `.name` stem, else `f"structure_{index}"`) |
| `_canonical_structure_ids_for_spec` | `_canonical_structure_ids_for_spec(spec: SamplingSpecification) -> list[str]` at `_sampling_helper.py:36`; reads only `spec.inputs`. Called with Scoring/Inspection/Jacobian specs too (`host/runner.py:176,539,712,897,1199`, `host/streaming.py:72`, `sampling/multistate_poe.py:536`) so a driver may pass any spec. Distinct-directory inputs `a/model.pdb`,`b/model.pdb` both map to `model` (stem only), as the spec test expects |
| `_structure_ids_for_batch` | `_structure_ids_for_batch(canonical_ids: list[str], *, structure_offset: int, batch_size: int) -> list[str]` at `_sampling_helper.py:48` (falls back to `structure_{global_idx}` beyond the list) |
| `resolve_target_samples` | `resolve_target_samples(spec: SamplingSpecification, chunk_sample_count: int \| None = None, grid_lineage: dict[str, int \| str] \| None = None) -> int` at `src/aminx/host/plan.py:382` (chunk count, else `grid_lineage["sample_count"]`, else `spec.run_spec.sampling.num_samples`; non-positive -> ValueError). `resolve_chunk_size` at `plan.py:425` |
| `make_axis_dispatch_via_xtrax` | `make_axis_dispatch_via_xtrax(strategy: AxisStrategy, *, axis: str = "state") -> object` at `src/aminx/tiling/dispatch.py:161`. Rejects `DedupGather` up front; wraps xtrax `DispatchRejected` into aminx `DispatchRejected`; passes `heterogeneous_axes=set(_DISPATCH_HETEROGENEOUS_AXES)`. **Its docstring says "Not yet wired into any call site"; that is stale**: wired at `host/runner.py:531,612,774,780`, `sampling/multistate_poe.py:80,227`, `sampling/mbr_consensus.py:30,198`, `sampling/conditional_logits.py:33,365,401`, `utils/forward_jac.py:42,176`, `inference/decode/factory.py:33,102-110` |
| runner entry points | `sample(spec: SamplingSpecification \| None = None, **kwargs) -> dict[str, Any]` `host/runner.py:60`; `score(spec: ScoringSpecification \| None = None, **kwargs)` `:441`; `inspect(spec: InspectionSpecification \| None = None, **kwargs)` `:820`; `jacobian(spec: JacobianSpecification \| None = None, **kwargs)` `:1143`. There is no separate "spec sync" statement: `sample` builds the spec from kwargs when `spec is None` (`:155-158`) then immediately calls `prep_protein_stream_and_model(spec)` (`:160`), then `make_inference_plan(model, spec, purpose="sample")` (`:166`); the driver dispatch goes between the spec construction and `:160` (`score`: `:484`/`:496`) |
| `prep_protein_stream_and_model` | `prep_protein_stream_and_model(spec: Specs) -> tuple[IterDataset, Model]` `host/prep.py:68` (called by every runner entry point; registry lookup by `(model_family, checkpoint_id)` at `prep.py:48,55`; `sha256` is optional at `prep.py:60`) |

`sample()` in-memory result. Top-level keys: `sequences`, `mask`, `schema_version` (`GRID_SCHEMA_VERSION` if `spec.grid_mode` else
`SAMPLING_SCHEMA_VERSION`), `metadata`; optional `logits`, `logit_fingerprint`, `pseudo_perplexity`, and (grid) `sample_indices`.
**`metadata` key set**: `specification` (the spec object), `skipped_inputs` (`getattr(protein_iterator, "skipped_frames", [])`),
`structure_ids`, plus `lineage` when a grid lineage is present (`runner.py:233,256`, `manifest_row_hash`,
`sample_indices`, `grid_iteration_ids`, `grid_iteration_sample_start`, `grid_iteration_sample_count`). Streaming
root attrs (`host/streaming.py:82-93` and `sampling/multistate_poe.py:607-619`): `schema_version`, `model_family`,
`ligand_conditioning`, `sidechain_conditioning`, `samples_chunk_size`. Both Zarr sites build
`ZarrStagingSink(SinkSpec(output_dir=..., format="zarr", flush_every=1))` with no `run_id`, confirming the T0.0 premise.

## 2. `model_family` consumer grep (spec 2.2)

Command: `rg -n model_family src/aminx` (whole tree, worktree at HEAD). Complete hit list, classified. "Fallback path"
= reached by PottsMPNN `jacobian`/`inspect`/`score:nll`/`score:logits` where the spec has `model_family="pottsmpnn"` but the
model is a stock `Aminx`.

| Site | What it does | Driver-family verdict | Fallback path (Potts on `Aminx`) |
|---|---|---|---|
| `run/specs.py:236` `model_family: Literal["proteinmpnn","ligandmpnn"] \| None = None` | field/Literal (T0.6 adds the two families) | extend Literal | field itself |
| `run/specs.py:483-516` (`__post_init__` derivation; comment block `:459-481`) | derives `ligandmpnn` from `get_topology_for_checkpoint(checkpoint_id)["model_type"] in ("ligand","packer")`; warns on explicit disagreement; ends with `self._sync_run_spec()` at `:520` | needs prefix branch `pottsmpnn_`/`lasermpnn_` **before** `get_topology_for_checkpoint` result is used; `get_topology_for_checkpoint("pottsmpnn_vanilla_20")` returns model_type `protein`, `atom_context_num=20` (last `_`-part digit not in {32,48,30}), `lasermpnn_0p1A_nothing_heldout` returns defaults (`io/weights.py:65-105`); neither contains `ligandmpnn` or `_sc_`, so neither misderives | same |
| `run/specs.py:633-634` float->tuple temperature coercion sits in `SamplingSpecification.__post_init__` *after* `super().__post_init__()` (which calls `_sync_run_spec()` at `:520` first) | the spec's `_resolve_family_defaults()` insertion point (`:519-520`) is confirmed; the float->tuple move must go into the hook or the first `_sync_run_spec()` still sees a bare float | as spec | as spec |
| `run/spec.py:56` `LigandConfig.model_family: str = eqx.field(static=True)`; `:321` `model_family=str(getattr(spec, "model_family", "proteinmpnn"))` | copies the label into `RunSpecification.ligand.model_family`. Only reader of `.ligand.model_family` anywhere: `run_spec_portable_json.py:153` | label only | label `pottsmpnn` flows into the InferencePlan run-spec; no reader other than the portable-JSON guard |
| `run/run_spec_portable_json.py:14,112,153` | v2 wire format: default `ligand.model_family="proteinmpnn"` on decode (`:112`); **serialize guard is `== "ligandmpnn"` only** (`:153`) | **guard must become "not proteinmpnn"** or a `pottsmpnn`/`lasermpnn` spec would serialize as v2 and silently round-trip as `proteinmpnn` (the v2 dict carries no family) | not reached by runner paths; reachable via `run_spec_portable_to_dict` |
| `host/_sampling_helper.py:255` `_prepare_ligand_context`: `if spec.model_family != "ligandmpnn": return {Y: None, ...}` | ligand/sidechain injection gate | driver families never reach it; on the fallback path `pottsmpnn != "ligandmpnn"` so it already behaves as `proteinmpnn` (no change needed, keep the explicit test) | **consumer reached on fallback, already correct** |
| `host/prep.py:48,55,84-86` `_resolve_local_checkpoint_from_registry` | registry entry match `entry["model_family"] == spec.model_family and entry["checkpoint_id"] == spec.checkpoint_id`; `sha256` verified only `if "sha256" in found` (`:60-63`) | registry needs `pottsmpnn`/`lasermpnn` sections; make `sha256` required for those (spec 4.1) | **consumer reached on fallback**: the registry family key stays `pottsmpnn`; `load_model` must return the `Aminx` inside the composed artifact (`mpnn_core`), not the composite |
| `host/streaming.py:84` (root attrs), `host/_sampling_grid_lineage.py:94,111` (grid hash payloads), `host/campaign.py:76,95` (manifest row hash), `sampling/multistate_poe.py:609` (root attrs) | write the family *label* into attrs / hash payloads; mechanically family-agnostic | all four sit on the MPNN `sample` / grid / campaign paths that driver families must not enter; raise `ValueError` for `grid_mode`/campaign/multistate-PoE with a driver family, per spec (the label is harmless, the *lineage semantics* — chunked `sample_start`, grid iteration ids — are not implemented by the driver) | not reached (no fallback purpose uses streaming sample/grid/campaign/PoE) |
| `cli.py:410,454,550,599,1026,1106` | `--model-family` option (help text says "proteinmpnn or ligandmpnn"), `_RunBase.model_family`, `_base_spec_kwargs` | update help text/validation | pass-through |

No consumer of `model_family` exists in `host/runner.py`, `host/kernel_dispatch.py`, `inference/**`, `io/**` (grep is clean),
so the runner dispatch (spec 2.1) is the only new reader that needs to be added there. Other family-adjacent
selectors not matched by the grep but relevant: `get_topology_for_checkpoint` (`io/weights.py:65`, keyed on
checkpoint-id substrings) and the legacy-arg guard in `load_model` (`io/weights.py:~386-392`: `"_v_" in checkpoint_id or "mpnn" in checkpoint_id`;
both `pottsmpnn_*` and `lasermpnn_*` ids contain `mpnn` and pass).

## 3. PottsMPNN checkpoint key audit (spec 4.1)

Method (titanix, oracle env): `torch.load(..., weights_only=False)`; upstream `PottsMPNN(ca_only=False,
num_letters=vocab, vocab=vocab, node_features=128, edge_features=128, hidden_dim=128, potts_dim=400,
num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48, augment_eps=0.0)` exactly as
`sample_seqs.py:46-58` builds it from `inputs/example_config_sample_seqs.yaml` (`hidden_dim 128, edge_features 128,
potts_dim 400, num_layers 3, num_edges 48`), `vocab = 22 if 'msa' in check_path else 21`, then
`model.load_state_dict(checkpoint['model_state_dict'], strict=False)` (`sample_seqs.py:60`). The constructor's
`nn.init.xavier_uniform_` (`potts_mpnn_utils.py:1263-1265`) runs before the load, so any missing key stays random.
`torch.manual_seed(0)` set before construction. No shape-mismatch error was raised for any file.

| Checkpoint (SHA-256 = pin) | epoch / step | `noise_level` / `num_edges` / `noise_type` | n state keys | `W_out` / `W_s` / `etab_out.weight` | missing | unexpected | Scope |
|---|---|---|---|---|---|---|---|
| `vanilla_model_weights/pottsmpnn_20.pt` | 100 / 144514 | 0.2 / 48 / atomic | 120 | (21,128) / (21,128) / (400,128) | `[]` | `[]` | in scope (`vanilla/pottsmpnn_20`) |
| `vanilla_model_weights/pottsmpnn_30.pt` | 100 / 144554 | 0.3 / 48 / atomic | 120 | same | `[]` | `[]` | in scope |
| `vanilla_model_weights/pottsmpnn_msa_20.pt` | 100 / 2066175 | 0.2 / 48 / atomic | 120 | (22,128) x2 / (400,128) | `[]` (vocab 22) | `[]` | out of scope (U3 MSA vocab-22) |
| `soluble_model_weights/sol_pottsmpnn_20.pt` | 100 / 125704 | 0.2 / 48 / atomic | 120 | (21,128) x2 / (400,128) | `[]` | `[]` | in scope (`soluble/sol_pottsmpnn_20`) |
| `soluble_model_weights/sol_pottsmpnn_30.pt` | 100 / 125730 | **0.0** / 48 / atomic | 120 | same | `[]` | `[]` | in scope. Note `noise_level` 0.0 although named `_30`; both the soluble and the compat copy carry 0.0 (metadata only; inference noise comes from `cfg.inference.noise`, not the checkpoint) |
| `soluble_model_weights/sol_pottsmpnn_msa_20.pt` | 100 / 1909820 | 0.2 / 48 / atomic | 120 | (22,128) x2 / (400,128) | `[]` (vocab 22) | `[]` | out of scope |
| `ft_model_weights/potts_ft.pt` | **200 / 4168657** | 0.2 / 48 / atomic | 120 | (21,128) x2 / (400,128) | `[]` | `[]` | in scope (`ft/potts_ft`) |
| `proteinmpnn_compatible_model_weights/pottsmpnn_20.pt` | 100 / 144514 | 0.2 / 48 / atomic | **118** | (21,128) x2 / **absent** | **`['etab_out.weight','etab_out.bias']`** | `[]` | excluded (`duplicate`) — **BLOCKING if loaded as Potts (B1)** |
| `proteinmpnn_compatible_model_weights/pottsmpnn_30.pt` | 100 / 144554 | 0.3 / 48 / atomic | 118 | absent | **`['etab_out.weight','etab_out.bias']`** | `[]` | excluded |
| `proteinmpnn_compatible_model_weights/pottsmpnn_msa_20.pt` | 100 / 2066175 | 0.2 / 48 / atomic | 118 | (22,128) x2 / absent | **`['etab_out.weight','etab_out.bias']`** | `[]` | excluded |
| `proteinmpnn_compatible_model_weights/sol_pottsmpnn_20.pt` | 100 / 125704 | 0.2 / 48 / atomic | 118 | absent | **`['etab_out.weight','etab_out.bias']`** | `[]` | excluded |
| `proteinmpnn_compatible_model_weights/sol_pottsmpnn_30.pt` | 100 / 125730 | 0.0 / 48 / atomic | 118 | absent | **`['etab_out.weight','etab_out.bias']`** | `[]` | excluded |
| `proteinmpnn_compatible_model_weights/sol_pottsmpnn_msa_20.pt` | 100 / 1909820 | 0.2 / 48 / atomic | 118 | (22,128) x2 / absent | **`['etab_out.weight','etab_out.bias']`** | `[]` | excluded |
| `ca_model_weights/v_48_{002,010,020}.pt` | n/a (top keys `model_state_dict`,`num_edges`=48,`noise_level`=0.02/0.10/0.20) | see left | 123 | (21,128) x2 / **absent** | not constructed (CA-only, `ca_only=True` constructor differs; spec non-goal `deferred:2068`) | not constructed | out of scope |

Top-level checkpoint keys (non-CA): `epoch, step, num_edges, noise_level, noise_type, model_state_dict, optimizer_state_dict`.
State-dict key set of an in-scope checkpoint (120 keys): `features.embeddings.linear.{weight,bias}`, `features.edge_embedding.weight`,
`features.norm_edges.{weight,bias}`, `W_e.*`, `W_s.weight`, `encoder_layers.{0,1,2}.*`, `decoder_layers.{0,1,2}.*`, `W_out.*`, `etab_out.{weight,bias}`
(the last two are the only keys not in ProteinMPNN's map, matching spec 4.1).

**Verdict for the in-scope set (5 checkpoints: vanilla 20/30, soluble 20/30, ft): missing and unexpected are empty for all five. No blocking finding for them.**

### 3.1 `ft/potts_ft` configuration (spec 4.1 "T0.2 confirms ft config")

`potts_ft.pt` loads with the identical constructor arguments as the other checkpoints (hidden/edge 128, potts_dim 400,
3+3 layers, k_neighbors 48, vocab 21, `W_out` (21,128)); its metadata is `epoch=200, step=4168657, num_edges=48,
noise_level=0.2, noise_type=atomic`. It is key-identical to `vanilla_model_weights/pottsmpnn_20.pt` (120/120 keys, identical shapes) but
**120 of 120 tensors differ** from it, so it is a separately trained/fine-tuned model, not a relabel. No extra config keys are stored in the
checkpoint. The example YAML configs need no `ft`-specific overrides. Registry id `ft/potts_ft` is valid as a normal in-scope entry.

### 3.2 `proteinmpnn_compatible_model_weights` vs vanilla/soluble (spec 4.1 exclusion)

For each of the six compat files the key set equals the corresponding vanilla/soluble file minus `{etab_out.weight, etab_out.bias}`,
and **all 118 shared tensors are bit-identical (`torch.equal`, max |d| = 0.0)**:
`compat/pottsmpnn_{20,30,msa_20}` vs `vanilla/pottsmpnn_{20,30,msa_20}`; `compat/sol_pottsmpnn_{20,30,msa_20}` vs `soluble/sol_pottsmpnn_{20,30,msa_20}`.
Key names are already identical to the PottsMPNN class names (no rename needed). Verdict: the `duplicate` exclusion is justified for the
trunk weights (the trunk is loaded from the in-scope vanilla/soluble file anyway); the files are **not** loadable as Potts models (B1).

### 3.3 Temperature floor value (spec 4.3, 6.5)

`if cfg.inference.temperature == 0: cfg.inference.temperature = 1e-6` (`sample_seqs.py:38`);
`if cfg.inference.optimization_temperature == 0: cfg.inference.optimization_temperature = 1e-6` (`sample_seqs.py:39`).
**Floor = 1e-6, applied to exact 0 only** (not a `max(T, 1e-6)` clamp; `T=1e-9` is left alone). Neighbouring config normalisations:
`:40` `if optimize_pdb or optimize_fasta: num_samples = 1` (silent), `:41` `if optimization_mode == "none": optimization_mode = ""`,
`:37` `vocab = 22 if 'msa' in check_path else 21`. `energy_prediction.py` has no temperature.

Verbatim `sample_seqs.py:36-42` (pinned 0cb0a58; left gutter = source line number):

```python
36|     cfg = OmegaConf.load(args.config)
37|     cfg.model.vocab = 22 if 'msa' in cfg.model.check_path else 21
38|     if cfg.inference.temperature == 0: cfg.inference.temperature = 1e-6
39|     if cfg.inference.optimization_temperature == 0: cfg.inference.optimization_temperature = 1e-6
40|     if cfg.inference.optimize_pdb or cfg.inference.optimize_fasta: cfg.inference.num_samples = 1
41|     if cfg.inference.optimization_mode == "none": cfg.inference.optimization_mode = ""
42|
```

## 4. PottsMPNN verbatim transcriptions

### 4.1 Decoder order/mask prelude and PSSMMix in `decoder` (spec 4.3 "PSSMMix ... `:1391-1404`/decoder equivalents")

`sample()` (unused by the CLI path, kept because the spec cites `:1391-1404`), the PSSM block:

Verbatim `potts_mpnn_utils.py:1390-1405` (pinned 0cb0a58; left gutter = source line number):

```python
1390|                 logits = self.W_out(h_V_t) / temperature
1391|                 probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
1392|                 if pssm_bias_flag and (pssm_coef.numel()>0) or (pssm_bias.numel()>0):
1393|                     pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
1394|                     pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
1395|                     probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
1396|                 if pssm_log_odds_flag and pssm_log_odds_mask.numel()>0:
1397|                     pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, self.vocab]
1398|                     probs_masked = probs*pssm_log_odds_mask_gathered
1399|                     probs_masked += probs * 0.001
1400|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
1401|                 if omit_AA_mask_flag and omit_AA_mask.numel()>0:
1402|                     omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, self.vocab]
1403|                     probs_masked = probs*(1.0-omit_AA_mask_gathered)
1404|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
1405|                 S_t = torch.multinomial(probs, 1)
```

`decoder` (`potts_mpnn_utils.py:1415-1488`, the untied CLI path, called from `sample_seqs.py:226`), verbatim in full. The order key is
`:1419-1421`, `mask_bw/mask_fw` `:1422-1428`, masked-row branch `:1448-1449`, PSSM/omit block `:1465-1479`, draw `:1480`,
fixed select `:1483`:

Verbatim `potts_mpnn_utils.py:1415-1488` (pinned 0cb0a58; left gutter = source line number):

```python
1415|     def decoder(self, h_V, E_idx, h_E, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, bias_by_res=None, decoding_order=None):
1416|         device = h_V.device
1417|
1418|         # Decoder uses masked self-attention
1419|         chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
1420|         if decoding_order is None:
1421|             decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
1422|         mask_size = E_idx.shape[1]
1423|         permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
1424|         order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
1425|         mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
1426|         mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
1427|         mask_bw = mask_1D * mask_attend
1428|         mask_fw = mask_1D * (1. - mask_attend)
1429|
1430|         N_batch, N_nodes = h_V.size(0), h_V.size(1)
1431|         all_probs = torch.zeros((N_batch, N_nodes, self.vocab), device=device, dtype=torch.float32)
1432|         h_S = torch.zeros_like(h_V, device=device)
1433|         S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
1434|         h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
1435|         constant = torch.tensor(omit_AAs_np, device=device)
1436|         constant_bias = torch.tensor(bias_AAs_np, device=device)
1437|         #chain_mask_combined = chain_mask*chain_M_pos
1438|         omit_AA_mask_flag = omit_AA_mask != None
1439|
1440|         h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
1441|         h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
1442|         h_EXV_encoder_fw = mask_fw * h_EXV_encoder
1443|         for t_ in range(N_nodes):
1444|             t = decoding_order[:,t_] #[B]
1445|             chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
1446|             mask_gathered = torch.gather(mask, 1, t[:,None]) #[B]
1447|             bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,self.vocab))[:,0,:] #[B, self.vocab]
1448|             if (mask_gathered==0).all(): # for padded or missing regions only
1449|                 S_t = torch.gather(S_true, 1, t[:,None])
1450|             else:
1451|                 # Hidden layers
1452|                 E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
1453|                 h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
1454|                 h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
1455|                 h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
1456|                 mask_t = torch.gather(mask, 1, t[:,None])
1457|                 for l, layer in enumerate(self.decoder_layers):
1458|                     # Updated relational features for future states
1459|                     h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
1460|                     h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
1461|                     h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
1462|                     h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
1463|                 # Sampling step
1464|                 h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
1465|                 logits = self.W_out(h_V_t) / temperature
1466|                 probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
1467|                 if pssm_bias_flag and (pssm_coef.numel()>0) or (pssm_bias.numel()>0):
1468|                     pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
1469|                     pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
1470|                     probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
1471|                 if pssm_log_odds_flag and pssm_log_odds_mask.numel()>0:
1472|                     pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, self.vocab]
1473|                     probs_masked = probs*pssm_log_odds_mask_gathered
1474|                     probs_masked += probs * 0.001
1475|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
1476|                 if omit_AA_mask_flag and omit_AA_mask.numel()>0:
1477|                     omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, self.vocab]
1478|                     probs_masked = probs*(1.0-omit_AA_mask_gathered)
1479|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
1480|                 S_t = torch.multinomial(probs, 1)
1481|                 all_probs.scatter_(1, t[:,None,None].repeat(1,1,self.vocab), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
1482|             S_true_gathered = torch.gather(S_true, 1, t[:,None])
1483|             S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
1484|             temp1 = self.W_s(S_t)
1485|             h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
1486|             S.scatter_(1, t[:,None], S_t)
1487|         output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
1488|         return output_dict, all_probs
```

### 4.2 `tied_decoder` (spec 4.3 Tied; groups `:1606-1614`, masked member `:1641-1647`, last-member leak `:1665-1684`)

Note the spec's row-153 anchor "`tied_decoder :1599-` (body as `:1546-1595`...)" cites the near-identical `tied_sample` copy
(`:1490-1597`); the CLI calls `tied_decoder` (`sample_seqs.py:235`), whose line numbers are below. Both copies are byte-equivalent
in the decode body apart from the encoder prelude.

Verbatim `potts_mpnn_utils.py:1599-1687` (pinned 0cb0a58; left gutter = source line number):

```python
1599|     def tied_decoder(self, h_V, E_idx, h_E, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, tied_pos=None, tied_beta=None, bias_by_res=None):
1600|         device = h_V.device
1601|
1602|         # Decoder uses masked self-attention
1603|         chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
1604|         decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
1605|
1606|         new_decoding_order = []
1607|         for t_dec in list(decoding_order[0,].cpu().data.numpy()):
1608|             if t_dec not in list(itertools.chain(*new_decoding_order)):
1609|                 list_a = [item for item in tied_pos if t_dec in item]
1610|                 if list_a:
1611|                     new_decoding_order.append(list_a[0])
1612|                 else:
1613|                     new_decoding_order.append([t_dec])
1614|         decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=device)[None,].repeat(h_V.shape[0],1)
1615|
1616|         mask_size = E_idx.shape[1]
1617|         permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
1618|         order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
1619|         mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
1620|         mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
1621|         mask_bw = mask_1D * mask_attend
1622|         mask_fw = mask_1D * (1. - mask_attend)
1623|
1624|         N_batch, N_nodes = h_V.size(0), h_V.size(1)
1625|         all_probs = torch.zeros((N_batch, N_nodes, self.vocab), device=device, dtype=torch.float32)
1626|         h_S = torch.zeros_like(h_V, device=device)
1627|         S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
1628|         h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
1629|         constant = torch.tensor(omit_AAs_np, device=device)
1630|         constant_bias = torch.tensor(bias_AAs_np, device=device)
1631|         omit_AA_mask_flag = omit_AA_mask != None
1632|
1633|         h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
1634|         h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
1635|         h_EXV_encoder_fw = mask_fw * h_EXV_encoder
1636|         for t_list in new_decoding_order:
1637|             logits = 0.0
1638|             logit_list = []
1639|             done_flag = False
1640|             for t in t_list:
1641|                 if (mask[:,t]==0).all():
1642|                     S_t = S_true[:,t]
1643|                     for t in t_list:
1644|                         h_S[:,t,:] = self.W_s(S_t)
1645|                         S[:,t] = S_t
1646|                     done_flag = True
1647|                     break
1648|                 else:
1649|                     E_idx_t = E_idx[:,t:t+1,:]
1650|                     h_E_t = h_E[:,t:t+1,:,:]
1651|                     h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
1652|                     h_EXV_encoder_t = h_EXV_encoder_fw[:,t:t+1,:,:]
1653|                     mask_t = mask[:,t:t+1]
1654|                     for l, layer in enumerate(self.decoder_layers):
1655|                         h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
1656|                         h_V_t = h_V_stack[l][:,t:t+1,:]
1657|                         h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_EXV_encoder_t
1658|                         h_V_stack[l+1][:,t,:] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
1659|                     h_V_t = h_V_stack[-1][:,t,:]
1660|                     logit_list.append((self.W_out(h_V_t) / temperature)/len(t_list))
1661|                     logits += tied_beta[t]*(self.W_out(h_V_t) / temperature)/len(t_list)
1662|             if done_flag:
1663|                 pass
1664|             else:
1665|                 bias_by_res_gathered = bias_by_res[:,t,:] #[B, self.vocab]
1666|                 probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
1667|                 if pssm_bias_flag and (pssm_coef.numel()>0) or (pssm_bias.numel()>0):
1668|                     pssm_coef_gathered = pssm_coef[:,t]
1669|                     pssm_bias_gathered = pssm_bias[:,t]
1670|                     probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
1671|                 if pssm_log_odds_flag and pssm_log_odds_mask.numel()>0:
1672|                     pssm_log_odds_mask_gathered = pssm_log_odds_mask[:,t]
1673|                     probs_masked = probs*pssm_log_odds_mask_gathered
1674|                     probs_masked += probs * 0.001
1675|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
1676|                 if omit_AA_mask_flag and omit_AA_mask.numel()>0:
1677|                     omit_AA_mask_gathered = omit_AA_mask[:,t]
1678|                     probs_masked = probs*(1.0-omit_AA_mask_gathered)
1679|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
1680|                 S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
1681|                 S_t_repeat = (chain_mask[:,t]*S_t_repeat + (1-chain_mask[:,t])*S_true[:,t]).long() #hard pick fixed positions
1682|                 for t in t_list:
1683|                     h_S[:,t,:] = self.W_s(S_t_repeat)
1684|                     S[:,t] = S_t_repeat
1685|                     all_probs[:,t,:] = probs.float()
1686|         output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
1687|         return output_dict, all_probs
```

Confirmed properties (each maps to a `conditional_ids.txt` row): group order built from the singleton order by first-member rule
(`:1606-1614`; overlapping groups double-decode); untied positions also multiply logits by `tied_beta[t]` (`:1661`, default ones
`potts_mpnn_utils.py:444`); the loop variable `t` after `for t in t_list` is the **last member** and is what `bias_by_res[:,t]`, `pssm_*[:,t]`,
`omit_AA_mask[:,t]`, `chain_mask[:,t]`, `S_true[:,t]` read (`:1665-1681`); a masked (`mask==0`) member aborts the group: members processed
earlier keep their layer updates, all members get `S_true[m*]` (`:1641-1647`), and `all_probs` is not written.

### 4.3 `optimize_sequence` (untied refine, `run_utils.py:75-271`), including `nodes` (`:180-270`)

Signature (`:75-81`) and body. Binding branch `:135-148`, accumulate `:148,176`, converge cap `:116-120`, PSSMMix `:161-173`, draw `:174`,
`nodes` `:180-270`:

Verbatim `run_utils.py:75-81` (pinned 0cb0a58; left gutter = source line number):

```python
75| def optimize_sequence(seq, etab, E_idx, mask, chain_mask, opt_type, seq_encoder, optimization_temp=0.0001,
76|                       constant=None, constant_bias=None, bias_by_res=None,
77|                       pssm_bias_flag=False, pssm_coef=None, pssm_bias=None, pssm_multi=None,
78|                       pssm_log_odds_flag=False, pssm_log_odds_mask=None, omit_AA_mask=None,
79|                       model=None, h_E=None, h_EXV_encoder=None, h_V=None,
80|                       decoding_order=None, partition_etabs=None,
81|                       partition_index=None, inter_mask=None, binding_optimization=None, vocab=21):
```
Verbatim `run_utils.py:105-271` (pinned 0cb0a58; left gutter = source line number):

```python
105|     omit_AA_mask_flag = omit_AA_mask != None
106|     etab = etab.clone().view(etab.shape[0], etab.shape[1], etab.shape[2], int(np.sqrt(etab.shape[3])), int(np.sqrt(etab.shape[3])))
107|     etab = torch.nn.functional.pad(etab, (0, 2, 0, 2), "constant", 0)
108|     seq = torch.Tensor(seq_encoder(seq)).unsqueeze(0).to(dtype=torch.int64, device=E_idx.device)
109|
110|     if decoding_order is None:
111|         decoding_order = np.arange(seq.shape[1])
112|
113|     if 'nodes' not in opt_type:
114|         ener_delta = 1
115|         iters_done = 0
116|         if 'converge' not in opt_type:
117|             max_iters = 1
118|         else:
119|             max_iters = 1000
120|         while (ener_delta != 0 and iters_done < max_iters):
121|             ener_delta = 0
122|             for pos in decoding_order:
123|                 if not mask[0,pos] or not chain_mask[0,pos]:
124|                     continue
125|                 sort_seqs = []
126|
127|                 for mut_ind in range(20):
128|                     mut_seq = copy.deepcopy(seq)
129|                     mut_seq[0, pos] = mut_ind
130|                     sort_seqs.append(mut_seq)
131|
132|                 sort_seqs = torch.stack(sort_seqs, dim=1).to(etab.device)
133|
134|                 # Perform standard stability prediction by default, binding energy if requested
135|                 if binding_optimization == 'only' and not inter_mask[0, pos]:
136|                     continue
137|
138|                 predicted_E = etab_utils.positional_potts_energy(etab, E_idx, seq, pos)
139|                 if binding_optimization in ['only', 'both'] and inter_mask[0, pos]:
140|                     partition_mask = partition_index == partition_index[0,pos]
141|                     partition_seq = seq[:, partition_mask[0]]
142|                     partition_pos = partition_mask[:, :pos].sum(dim=1).cpu().item()
143|                     partition_etab, partition_E_idx, _ = partition_etabs[partition_index[0, pos].cpu().item()]
144|                     unbound_predicted_E = etab_utils.positional_potts_energy(
145|                         partition_etab, partition_E_idx, partition_seq, partition_pos
146|                     )
147|                     # predicted_E = (predicted_E / etab.shape[2]) - (unbound_predicted_E / partition_etab.shape[2]) # Bound - unbound
148|                     predicted_E = (predicted_E - predicted_E[seq[0,pos].cpu().item()]) - (unbound_predicted_E - unbound_predicted_E[seq[0,pos].cpu().item()]) # Bound - unbound
149|
150|                 # Sample from predicted energies
151|                 predicted_E = predicted_E[:vocab]
152|                 t = torch.tensor([pos], dtype=torch.long, device=E_idx.device)
153|                 bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,predicted_E.shape[-1]))[:,0,:20] #[B, 20]
154|                 logits = -predicted_E / optimization_temp
155|                 logits = logits[:20] # Gap and X should never be chosen
156|                 constant = constant[:20]
157|                 constant_bias = constant_bias[:20]
158|                 probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/optimization_temp+bias_by_res_gathered/optimization_temp, dim=-1)
159|                 pad = (0, vocab-20)
160|                 probs = F.pad(probs, pad, "constant", 0) # Reshape to match other tensor shapes
161|                 if pssm_bias_flag and (pssm_coef.numel()>0) or (pssm_bias.numel()>0):
162|                     pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
163|                     pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
164|                     probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
165|                 if pssm_log_odds_flag and pssm_log_odds_mask.numel()>0:
166|                     pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, self.vocab]
167|                     probs_masked = probs*pssm_log_odds_mask_gathered
168|                     probs_masked += probs * 0.001
169|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
170|                 if omit_AA_mask_flag and omit_AA_mask.numel()>0:
171|                     omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, self.vocab]
172|                     probs_masked = probs*(1.0-omit_AA_mask_gathered)
173|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
174|                 mut_res = torch.multinomial(probs, num_samples=1).squeeze(-1)
175|                 seq = sort_seqs[0, mut_res]
176|                 ener_delta += predicted_E[mut_res.cpu().item()]
177|
178|             iters_done += 1
179|         best_seq = seq[0]
180|     else:
181|         S = seq.clone() # [B, L]
182|         h_S = model.W_s(S)             # [B, L, D]
183|         h_V_stack = [h_V] + [torch.zeros_like(h_V, device=h_V.device) for _ in range(len(model.decoder_layers))]
184|
185|         # Ensure decoding order has batch dimension
186|         decoding_order = torch.as_tensor(decoding_order, dtype=torch.long, device=h_V.device).unsqueeze(0)
187|         mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
188|         mask_bw = torch.ones((mask.size(0), mask.size(1), E_idx.size(2), 1)).to(device=h_V.device) * mask_1D
189|         mask_bw[:,:,0,0] = 0
190|         mask_fw = mask_1D * (1 - mask_bw)
191|         h_EXV_encoder_fw = h_EXV_encoder * mask_fw
192|
193|         for t_ in range(seq.shape[1]):
194|             t = decoding_order[:, t_]  # [B]
195|             if not mask[0,t[0].cpu().item()] or not chain_mask[0,t[0].cpu().item()]:
196|                 continue
197|             mask_gathered = torch.gather(mask, 1, t[:, None])  # [B, 1]
198|
199|             if (mask_gathered == 0).all():
200|                 continue
201|
202|             # --- MASK the current position ---
203|             h_S_masked = h_S.clone()
204|             # Expand t to match h_S shape for scatter
205|             # index = t[:, None, None].expand(-1, 1, h_S.shape[-1])  # [B, 1, D]
206|             h_EXV_encoder_t = torch.gather(
207|                 h_EXV_encoder_fw,
208|                 1,
209|                 t[:, None, None, None].expand(-1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]),
210|             )
211|
212|             # Hidden layers
213|             E_idx_t = torch.gather(E_idx, 1, t[:, None, None].expand(-1, 1, E_idx.shape[-1]))
214|             h_E_t = torch.gather(
215|                 h_E, 1, t[:, None, None, None].expand(-1, 1, h_E.shape[-2], h_E.shape[-1])
216|             )
217|             h_ES_t = cat_neighbors_nodes(h_S_masked, h_E_t, E_idx_t)
218|
219|             mask_t = torch.gather(mask, 1, t[:, None])
220|
221|             for l, layer in enumerate(model.decoder_layers):
222|                 h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
223|                 # h_ESV_decoder_t[:,:,0] = h_EXV_encoder_t[:,:,0]
224|                 h_V_t = torch.gather(
225|                     h_V_stack[l], 1, t[:, None, None].expand(-1, 1, h_V_stack[l].shape[-1])
226|                 )
227|
228|                 h_ESV_t = (
229|                     torch.gather(
230|                         mask_bw,
231|                         1,
232|                         t[:, None, None, None].expand(-1, 1, mask_bw.shape[-2], mask_bw.shape[-1]),
233|                     )
234|                     * h_ESV_decoder_t
235|                     + h_EXV_encoder_t
236|                 )
237|                 h_V_stack[l + 1].scatter_(
238|                     1,
239|                     t[:, None, None].expand(-1, 1, h_V.shape[-1]),
240|                     layer(h_V_t, h_ESV_t, mask_V=mask_t),
241|                 )
242|
243|             # Compute residue probabilities and sample new residue
244|             h_V_t = torch.gather(
245|                 h_V_stack[-1], 1, t[:, None, None].expand(-1, 1, h_V_stack[-1].shape[-1])
246|             )[:, 0]
247|             bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,vocab))[:,0,:] #[B, self.vocab]
248|             logits = model.W_out(h_V_t) / optimization_temp
249|             probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/optimization_temp+bias_by_res_gathered/optimization_temp, dim=-1)
250|             if pssm_bias_flag and (pssm_coef.numel()>0) or (pssm_bias.numel()>0):
251|                 pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
252|                 pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
253|                 probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
254|             if pssm_log_odds_flag and pssm_log_odds_mask.numel()>0:
255|                 pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, self.vocab]
256|                 probs_masked = probs*pssm_log_odds_mask_gathered
257|                 probs_masked += probs * 0.001
258|                 probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
259|             if omit_AA_mask_flag and omit_AA_mask.numel()>0:
260|                 omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, self.vocab]
261|                 probs_masked = probs*(1.0-omit_AA_mask_gathered)
262|                 probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
263|             S_t = torch.multinomial(probs, num_samples=1) # [B, 1]
264|
265|             # Update sequence embedding at this position
266|             temp1 = model.W_s(S_t)  # [B, 1, D]
267|             h_S.scatter_(1, t[:, None, None].expand(-1, 1, temp1.shape[-1]), temp1)
268|             S.scatter_(1, t[:, None], S_t)
269|
270|         best_seq = S[0]
271|     return best_seq
```

### 4.4 `tied_optimize_sequence` (tied refine incl. `tied_epistasis`, `run_utils.py:273-519`)

Signature (`:273-280`) and body. Non-nodes `:321-431`, tied `nodes` `:432-518`; `tied_epistasis` branches at `:344-373` (joint) vs `:374-401`
(per member) and `:422-427`, `nodes` same-group mask `:453-463`:

Verbatim `run_utils.py:273-280` (pinned 0cb0a58; left gutter = source line number):

```python
273| def tied_optimize_sequence(seq, etab, E_idx, mask, chain_mask, opt_type, seq_encoder, optimization_temp=0.0001,
274|                       constant=None, constant_bias=None, bias_by_res=None,
275|                       pssm_bias_flag=False, pssm_coef=None, pssm_bias=None, pssm_multi=None,
276|                       pssm_log_odds_flag=False, pssm_log_odds_mask=None, omit_AA_mask=None,
277|                       model=None, h_E=None, h_EXV_encoder=None, h_V=None,
278|                       decoding_order=None, partition_etabs=None,
279|                       partition_index=None, inter_mask=None, binding_optimization=None, vocab=21,
280|                       tied_pos=None, tied_beta=None, tied_epistasis=True):
```
Verbatim `run_utils.py:304-519` (pinned 0cb0a58; left gutter = source line number):

```python
304|     omit_AA_mask_flag = omit_AA_mask != None
305|     etab = etab.clone().view(etab.shape[0], etab.shape[1], etab.shape[2], int(np.sqrt(etab.shape[3])), int(np.sqrt(etab.shape[3])))
306|     etab = torch.nn.functional.pad(etab, (0, 2, 0, 2), "constant", 0)
307|     seq = torch.Tensor(seq_encoder(seq)).unsqueeze(0).to(dtype=torch.int64, device=E_idx.device)
308|
309|     if decoding_order is None:
310|         decoding_order = np.arange(seq.shape[1])
311|
312|     new_decoding_order = []
313|     for t_dec in decoding_order:
314|         if t_dec not in list(itertools.chain(*new_decoding_order)):
315|             list_a = [item for item in tied_pos if t_dec in item]
316|             if list_a:
317|                 new_decoding_order.append(list_a[0])
318|             else:
319|                 new_decoding_order.append([t_dec])
320|
321|     if 'nodes' not in opt_type:
322|         ener_delta = 1
323|         iters_done = 0
324|         if 'converge' not in opt_type:
325|             max_iters = 1
326|         else:
327|             max_iters = 1000
328|         while (ener_delta != 0 and iters_done < max_iters):
329|             ener_delta = 0
330|             for pos_list in new_decoding_order:
331|                 # If any of the positions are masked, set all other residues to that position and skip
332|                 skip_pos = False
333|                 for pos in pos_list:
334|                     if not mask[0,pos] or not chain_mask[0,pos]:
335|                         skip_pos = True
336|                         for pos_inner in pos_list:
337|                             seq[0, pos_inner] = seq[0, pos]
338|                         break
339|                 if skip_pos:
340|                     continue
341|
342|                 predicted_E = 0.0
343|                 num_pos = 0
344|                 if tied_epistasis:
345|                     sort_seqs = []
346|                     skip_pos = True
347|                     for mut_ind in range(20):
348|                         mut_seq = copy.deepcopy(seq)
349|                         for pos in pos_list:
350|                             if binding_optimization == 'only' and not inter_mask[0, pos]:
351|                                     continue
352|                             else:
353|                                 skip_pos = False
354|                             mut_seq[0, pos] = mut_ind
355|                         sort_seqs.append(mut_seq)
356|                     if skip_pos: # Skip if all positions in tied set are non-binding and optimizing only binding energy
357|                         continue
358|                     sort_seqs = torch.stack(sort_seqs, dim=1).to(etab.device)
359|
360|                     # Perform standard stability prediction by default, binding energy if requested
361|                     predicted_E_pos = etab_utils.positional_potts_energy(etab, E_idx, seq, pos)
362|                     if binding_optimization in ['only', 'both'] and inter_mask[0, pos]:
363|                         partition_mask = partition_index == partition_index[0,pos]
364|                         partition_seq = seq[:, partition_mask[0]]
365|                         partition_pos = partition_mask[:, :pos].sum(dim=1).cpu().item()
366|                         partition_etab, partition_E_idx, _ = partition_etabs[partition_index[0, pos].cpu().item()]
367|                         unbound_predicted_E_pos = etab_utils.positional_potts_energy(
368|                             partition_etab, partition_E_idx, partition_seq, partition_pos
369|                         )
370|
371|                         predicted_E_pos = predicted_E_pos - unbound_predicted_E_pos # Bound - unbound
372|                     predicted_E += predicted_E_pos
373|                     num_pos = 1
374|                 else:
375|                     for pos in pos_list:
376|
377|                         # Perform standard stability prediction by default, binding energy if requested
378|                         if binding_optimization == 'only' and not inter_mask[0, pos]:
379|                             continue
380|
381|                         sort_seqs = []
382|                         for mut_ind in range(20):
383|                             mut_seq = copy.deepcopy(seq)
384|                             mut_seq[0, pos] = mut_ind
385|                             sort_seqs.append(mut_seq)
386|                         sort_seqs = torch.stack(sort_seqs, dim=1).to(etab.device)
387|
388|                         predicted_E_pos = etab_utils.positional_potts_energy(etab, E_idx, seq, pos)
389|                         if binding_optimization in ['only', 'both'] and inter_mask[0, pos]:
390|                             partition_mask = partition_index == partition_index[0,pos]
391|                             partition_seq = seq[:, partition_mask[0]]
392|                             partition_pos = partition_mask[:, :pos].sum(dim=1).cpu().item()
393|                             partition_etab, partition_E_idx, _ = partition_etabs[partition_index[0, pos].cpu().item()]
394|                             unbound_predicted_E_pos = etab_utils.positional_potts_energy(
395|                                 partition_etab, partition_E_idx, partition_seq, partition_pos
396|                             )
397|
398|                             predicted_E_pos = predicted_E_pos - unbound_predicted_E_pos # Bound - unbound
399|                         predicted_E += predicted_E_pos
400|                         num_pos += 1
401|                 predicted_E /= num_pos
402|                 # Sample from predicted energies
403|                 predicted_E = predicted_E[:vocab] # Gap should never be chosen if not in model vocab
404|                 t = torch.tensor([pos], dtype=torch.long, device=E_idx.device)
405|                 bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,predicted_E.shape[-1]))[:,0,:] #[B, self.vocab]
406|                 logits = -predicted_E / optimization_temp
407|                 probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/optimization_temp+bias_by_res_gathered/optimization_temp, dim=-1)
408|                 if pssm_bias_flag and (pssm_coef.numel()>0) or (pssm_bias.numel()>0):
409|                     pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
410|                     pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
411|                     probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
412|                 if pssm_log_odds_flag and pssm_log_odds_mask.numel()>0:
413|                     pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, self.vocab]
414|                     probs_masked = probs*pssm_log_odds_mask_gathered
415|                     probs_masked += probs * 0.001
416|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
417|                 if omit_AA_mask_flag and omit_AA_mask.numel()>0:
418|                     omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, self.vocab]
419|                     probs_masked = probs*(1.0-omit_AA_mask_gathered)
420|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
421|                 mut_seq = torch.multinomial(probs, num_samples=1).squeeze(-1)
422|                 if tied_epistasis:
423|                     seq = sort_seqs[0, mut_seq]
424|                 else:
425|                     mut_res = sort_seqs[0, mut_seq][0, pos]
426|                     for pos in pos_list:
427|                         seq[0, pos] = mut_res
428|                 ener_delta += predicted_E[mut_seq.cpu().item()]
429|
430|             iters_done += 1
431|         best_seq = seq[0]
432|     else:
433|         S = seq.clone() # [B, L]
434|         h_S = model.W_s(S)             # [B, L, D]
435|         h_V_stack = [h_V] + [torch.zeros_like(h_V, device=h_V.device) for _ in range(len(model.decoder_layers))]
436|
437|         # Ensure decoding order has batch dimension
438|         decoding_order = torch.as_tensor(decoding_order, dtype=torch.long, device=h_V.device).unsqueeze(0)
439|         new_decoding_order = []
440|         for t_dec in list(decoding_order[0,].cpu().data.numpy()):
441|             if t_dec not in list(itertools.chain(*new_decoding_order)):
442|                 list_a = [item for item in tied_pos if t_dec in item]
443|                 if list_a:
444|                     new_decoding_order.append(list_a[0])
445|                 else:
446|                     new_decoding_order.append([t_dec])
447|         decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=h_V.device)[None,].repeat(h_V.shape[0],1)
448|
449|         mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
450|         mask_bw = torch.ones((mask.size(0), mask.size(1), E_idx.size(2), 1)).to(device=h_V.device) * mask_1D
451|         mask_bw[:,:,0,0] = 0
452|
453|         # Mask paired positions to model epistasis
454|         if tied_epistasis:
455|             N = mask.shape[1]
456|             group_map = torch.full((N,), -1, device=h_V.device, dtype=torch.long)
457|             for group_id, group_indices in enumerate(new_decoding_order):
458|                 # Assign a unique integer (group_id) to all indices in this group
459|                 group_map[group_indices] = group_id
460|             self_groups = group_map.view(1, N, 1)
461|             neighbor_groups = group_map[E_idx]
462|             is_same_group = (neighbor_groups == self_groups) & (neighbor_groups != -1)
463|             mask_bw.masked_fill_(is_same_group.unsqueeze(-1), 0.0)
464|
465|         mask_fw = mask_1D * (1 - mask_bw)
466|         h_EXV_encoder_fw = h_EXV_encoder * mask_fw
467|
468|         for t_list in new_decoding_order:
469|             logits = 0.0
470|             logit_list = []
471|             done_flag = False
472|             for t in t_list:
473|                 if (mask[:,t]==0).all():
474|                     S_t = S[:,t]
475|                     for t in t_list:
476|                         h_S[:,t,:] = model.W_s(S_t)
477|                         S[:,t] = S_t
478|                     done_flag = True
479|                     break
480|                 else:
481|                     E_idx_t = E_idx[:,t:t+1,:]
482|                     h_E_t = h_E[:,t:t+1,:,:]
483|                     h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
484|                     h_EXV_encoder_t = h_EXV_encoder_fw[:,t:t+1,:,:]
485|                     mask_t = mask[:,t:t+1]
486|                     for l, layer in enumerate(model.decoder_layers):
487|                         h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
488|                         h_V_t = h_V_stack[l][:,t:t+1,:]
489|                         h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_EXV_encoder_t
490|                         h_V_stack[l+1][:,t,:] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
491|                     h_V_t = h_V_stack[-1][:,t,:]
492|                     logit_list.append((model.W_out(h_V_t) / optimization_temp)/len(t_list))
493|                     logits += tied_beta[t]*(model.W_out(h_V_t) / optimization_temp)/len(t_list)
494|             if done_flag:
495|                 pass
496|             else:
497|                 bias_by_res_gathered = bias_by_res[:,t,:] #[B, self.vocab]
498|                 probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/optimization_temp+bias_by_res_gathered/optimization_temp, dim=-1)
499|                 if pssm_bias_flag and (pssm_coef.numel()>0) or (pssm_bias.numel()>0):
500|                     pssm_coef_gathered = pssm_coef[:,t]
501|                     pssm_bias_gathered = pssm_bias[:,t]
502|                     probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
503|                 if pssm_log_odds_flag and pssm_log_odds_mask.numel()>0:
504|                     pssm_log_odds_mask_gathered = pssm_log_odds_mask[:,t]
505|                     probs_masked = probs*pssm_log_odds_mask_gathered
506|                     probs_masked += probs * 0.001
507|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
508|                 if omit_AA_mask_flag and omit_AA_mask.numel()>0:
509|                     omit_AA_mask_gathered = omit_AA_mask[:,t]
510|                     probs_masked = probs*(1.0-omit_AA_mask_gathered)
511|                     probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, self.vocab]
512|                 S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
513|                 S_t_repeat = (chain_mask[:,t]*S_t_repeat + (1-chain_mask[:,t])*S[:,t]).long() #hard pick fixed positions
514|                 for t in t_list:
515|                     h_S[:,t,:] = model.W_s(S_t_repeat)
516|                     S[:,t] = S_t_repeat
517|
518|         best_seq = S[0]
519|     return best_seq
```

Quirks visible in the transcription that the spec's quirk table (6.5) does not list (each is a bit-parity obligation; see C9):
(a) `tied_epistasis=True` computes the positional energy at the **leaked loop variable `pos`** = last member (`:349` loop, `:361`), while the mutant
sequences set *all* members jointly (`:349-355`); `t = tensor([pos])` (`:404`) also reads the leaked `pos`. (b) the tied binding branch subtracts the
unbound energy **without** the current-identity reference subtraction used untied (`:371`,`:398` vs `:148`), so `ener_delta` accumulates
`E_bound - E_unbound` (absolute), not a delta-of-deltas; probabilities are identical (softmax shift invariance) but the `potts_converge` stop test is not.
(c) non-epistasis path divides by `num_pos` (`:401`); if every member is non-interface under `binding_energy_optimization="only"` then `num_pos == 0` and
`predicted_E /= 0` raises `ZeroDivisionError` (float `0.0`), an upstream crash. (d) tied `nodes` has **no per-member `chain_mask` skip** (only `mask==0`,
`:473`); `chain_mask` enters only through the last-member hard pick (`:513`) and the `mask` argument (`mask*chain_M_pos` from `sample_seqs.py:355`). (e) the
tied `nodes` masked-member fallback copies `S[:,t]` (the current sequence, `:474`), not `S_true`. (f) tied non-nodes skip test at `:331-340` propagates the residue of the *first member that is masked (`mask==0`) **or fixed (`chain_mask==0`)*** to all members (`seq[0,pos_inner] = seq[0,pos]`) and skips the group, so a group containing any fixed member is frozen to that member's residue.

### 4.5 Refine order keying and the sampling/refine driver (spec 4.3, 6.5; `sample_seqs.py:321-345`)

Order storage during sampling (`:274-279`) and the refine loop (`:316-373`), verbatim:

Verbatim `sample_seqs.py:267-287` (pinned 0cb0a58; left gutter = source line number):

```python
267|             # Sort and Store
268|             sample_records = sorted(sample_records, key=lambda x: x['energy'])
269|             for k, rec in enumerate(sample_records):
270|                 sidx = rec['sample_idx']
271|                 sample_suffix = f"_{sidx}" if cfg.inference.num_samples != 1 else ''
272|                 out_seqs[pdb_with_chain_suffix + sample_suffix] = ':'.join(rec['seq'][a:b] for a, b in zip(chain_cuts, chain_cuts[1:]))
273|
274|                 if pdb not in decoding_orders:
275|                     decoding_orders[pdb_with_chain_suffix] = {}
276|                 if cfg.inference.num_samples == 1:
277|                     decoding_orders[pdb_with_chain_suffix] = rec['decoding_order'].squeeze().cpu().numpy().tolist()
278|                 else:
279|                     decoding_orders[pdb_with_chain_suffix][sample_suffix.split('_')[1]] = rec['decoding_order'].squeeze().cpu().numpy().tolist()
280|
281|                 av_losses['pdb'].append(pdb_with_chain_suffix + sample_suffix)
282|                 av_losses['seq_loss'].append(sample_seq_loss[sidx])
283|                 av_losses['nsr'].append(sample_nsr[sidx])
284|                 av_losses['potts_loss'].append(sample_nlcpl[sidx])
285|
286|                 if k == 0: # Save best sequence and sample number
287|                     best_seqs[pdb_with_chain_suffix] = (out_seqs[pdb_with_chain_suffix + sample_suffix], sidx)
```
Verbatim `sample_seqs.py:316-373` (pinned 0cb0a58; left gutter = source line number):

```python
316|             # Optimize sequences associated with this PDB
317|             source_seqs = existing_seqs if skip_calc else out_seqs
318|
319|             current_pdb_keys = [k for k in source_seqs.keys() if k.startswith(pdb_with_chain_suffix)]
320|
321|             for key in current_pdb_keys:
322|                 seq_to_opt = source_seqs[key].replace(':', '')
323|                 suffix_key = key[len(pdb_with_chain_suffix):] if len(key) > len(pdb_with_chain_suffix) else ''
324|
325|                 stored_decoding = None
326|                 if pdb in decoding_orders:
327|                     if cfg.inference.num_samples == 1:
328|                         decoding_order = decoding_orders[pdb_with_chain_suffix]
329|                     else:
330|                         decoding_order = decoding_orders[pdb_with_chain_suffix].get(suffix_key, None)
331|                 else:
332|                     if cfg.inference.fix_decoding_order:
333|                         if cfg.inference.num_samples != 1:
334|                             suffix_add = int(suffix_key.split('_')[1])
335|                         else:
336|                             suffix_add = 0
337|                         torch.manual_seed(string_to_int(pdb) + cfg.inference.decoding_order_offset + suffix_add)
338|                     randn = torch.randn(chain_mask.shape, device=X.device)
339|                     decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))).squeeze().cpu().numpy().tolist()
340|                     if cfg.inference.num_samples == 1:
341|                         decoding_orders[pdb_with_chain_suffix] = decoding_order
342|                     else:
343|                         decoding_orders[pdb_with_chain_suffix][suffix_key.split('_')[1]] = decoding_order
344|                 if tied_positions_dict is None or not tied_pos_list_of_lists_list[0]:
345|                     opt_seq = optimize_sequence(
346|                         seq_to_opt, etab, E_idx, mask*chain_M_pos, chain_mask, cfg.inference.optimization_mode,
347|                         etab_utils.seq_to_ints, cfg.inference.optimization_temperature, constant, constant_bias,
348|                         bias_by_res, cfg.inference.pssm_bias_flag, pssm_coef, pssm_bias, cfg.inference.pssm_multi,
349|                         cfg.inference.pssm_log_odds_flag, pssm_log_odds_mask, omit_AA_mask, model, h_E, h_EXV_encoder, h_V,
350|                         decoding_order=decoding_order, partition_etabs=partition_etabs, partition_index=partition_index,
351|                         inter_mask=inter_mask, binding_optimization=cfg.inference.binding_energy_optimization, vocab=cfg.model.vocab
352|                     )
353|                 else:
354|                     opt_seq = tied_optimize_sequence(
355|                         seq_to_opt, etab, E_idx, mask*chain_M_pos, chain_mask, cfg.inference.optimization_mode,
356|                         etab_utils.seq_to_ints, cfg.inference.optimization_temperature, constant, constant_bias,
357|                         bias_by_res, cfg.inference.pssm_bias_flag, pssm_coef, pssm_bias, cfg.inference.pssm_multi,
358|                         cfg.inference.pssm_log_odds_flag, pssm_log_odds_mask, omit_AA_mask, model, h_E, h_EXV_encoder, h_V,
359|                         decoding_order=decoding_order, partition_etabs=partition_etabs, partition_index=partition_index,
360|                         inter_mask=inter_mask, binding_optimization=cfg.inference.binding_energy_optimization, vocab=cfg.model.vocab,
361|                         tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, tied_epistasis=cfg.inference.tied_epistasis
362|                     )
363|
364|                 opt_seq = etab_utils.ints_to_seq_torch(opt_seq)
365|                 opt_seq = ':'.join(opt_seq[a:b] for a, b in zip(chain_cuts, chain_cuts[1:]))
366|                 opt_seqs[key] = opt_seq
367|
368|                 if cfg.inference.num_samples == 1 or (not skip_calc and int(suffix_key.split('_')[1]) == best_seqs[pdb_with_chain_suffix][1]): # Overwrite best sequence if on appropriate sample
369|                     if pdb_with_chain_suffix in best_seqs:
370|                         suffix = best_seqs[pdb_with_chain_suffix][1]
371|                     else:
372|                         suffix = 0
373|                     best_seqs[pdb_with_chain_suffix] = (opt_seq, suffix)
```

Resolved keying behaviour (this is what `upstream_refine_order(...)` must reproduce):

1. Stored orders: for `num_samples == 1`, `decoding_orders[pdb_with_chain_suffix]` is the flat AR order list (`:276-277`); for `num_samples > 1` it is a dict keyed
   by `str(sidx)` (`sample_suffix.split('_')[1]`, i.e. `"0"`, `"1"`, ..., `:279`).
2. The presence test at `:274` and `:326` is `if pdb not in decoding_orders` / `if pdb in decoding_orders`, keyed by **`pdb`, not `pdb_with_chain_suffix`**.
   For an unsuffixed entry the two keys coincide. For a suffixed entry (`name|A|B`) `pdb` is never a key, so (i) at `:274` the dict is **re-initialised on
   every rank iteration `k`** (only the last-written sample's order survives for `N > 1`) and (ii) at `:326` the refine loop always falls to the fresh-order branch.
3. `N == 1`, unsuffixed: reuse the stored AR order (`:327-328`). `N > 1`, unsuffixed: lookup `.get(suffix_key)` with `suffix_key = "_k"` (`:330`) but stored keys are `"k"`:
   **always `None`**, so `optimize_sequence` defaults to `np.arange(L)` (N->C, `run_utils.py:110-111`).
4. Suffixed entries (any `N`): fresh order `argsort((chain_mask + 1e-4) * |randn|)` where `chain_mask` is the featurizer's designed mask with **no `chain_M_pos`
   and no `present`** (`:338-339`); seeded with `string_to_int(pdb) + decoding_order_offset + suffix_add` only if `fix_decoding_order`
   (`suffix_add = int(suffix_key.split('_')[1])` for `N != 1`, else 0, `:332-337`); otherwise the global torch RNG state after the AR draws. Storage `:340-343`.
5. `skip_calc` (optimize_fasta/optimize_pdb): `decoding_orders` is loaded from the implicit prior-run file `out_dir/out_name_decoding_order.json` if it exists and
   `optimize_fasta` is set (`:178-180`); otherwise empty, so unsuffixed entries take the fresh-order branch as well (spec divergence row `test_divergence_refine_order_json`).

### 4.6 `skip_calc` stage graph anchors (spec 4.3 row 432)

Verbatim `sample_seqs.py:126-159` (pinned 0cb0a58; left gutter = source line number):

```python
126|
127|     # Optimization check logic
128|     skip_calc = False
129|
130|     if cfg.inference.optimization_mode:
131|         optimized_filename = os.path.join(cfg.out_dir, cfg.out_name + f'_optimized_{cfg.inference.optimization_mode}.fasta')
132|
133|         if cfg.inference.optimize_fasta:
134|             assert os.path.exists(cfg.inference.optimize_fasta), f"Tried to optimize sequences in {cfg.inference.optimize_fasta}, but the file does not exist."
135|             print(f"Found existing sequences at {filename}. Loading for optimization...")
136|             with open(filename, 'r') as f:
137|                 seqs_raw = f.readlines()
138|             # Parse .fasta
139|             existing_seqs = {pdb.strip('>').strip(): seq.strip() for pdb, seq in zip(seqs_raw[::2], seqs_raw[1::2])}
140|             print(f'Saving optimized sequences to filename {optimized_filename}.')
141|             skip_calc = True
142|         elif cfg.inference.optimize_pdb:
143|             print(f"Optimizing existing sequences in pdb files in {cfg.input_dir}. Loading for optimization...")
144|             existing_seqs = {}
145|             for pdb, chain_info in zip(pdb_list, chain_suffixes):
146|                 wt_info = parse_PDB_seq_only(os.path.join(cfg.input_dir, pdb + '.pdb'), skip_gaps=cfg.inference.skip_gaps) # Parse .pdb files
147|                 if chain_info:
148|                     _, hidden_chains, vis_chains = chain_info.split('|')
149|                     chain_order = hidden_chains.split(':') + vis_chains.split(':')
150|                     wt_seq = ""
151|                     for chain in chain_order:
152|                         if chain: wt_seq += wt_info[f'seq_chain_{chain}']
153|                     existing_seqs[pdb + chain_info] = wt_seq
154|                 else:
155|                     existing_seqs[pdb ] = wt_info['seq']
156|             print(f'Saving optimized sequences to filename {optimized_filename}.')
157|             skip_calc = True
158|         else:
159|             print(f'Saving sequences to filename {filename} and saving optimized sequences to filename {optimized_filename}.')
```

Confirmed: `optimize_fasta` asserts `cfg.inference.optimize_fasta` exists but reads `filename` (= `out_dir/out_name.fasta`, `:133-137`); key
match is `k.startswith(pdb_with_chain_suffix)` (`:319`, prefix quirk: `2yc3` matches `2yc3x_0`); `':'` stripped (`:322`); `optimize_pdb` builds the sequence from
`parse_PDB_seq_only` (`potts_mpnn_utils.py:268-291`, chain alphabet order A..Z,a..z,0..299 unless a list is passed) and, when a chain suffix exists, concatenates
`seq_chain_{c}` in **listing** order hidden+visible (`sample_seqs.py:148-153`) whereas featurization sorts chains (`potts_mpnn_utils.py:327-329`). Both quirks are the spec's
6.5b rows. PDB/`best_seqs` rule `:368-373` confirmed: for `N == 1` every key overwrites `best_seqs[...]` (last key wins under `skip_calc`); for `N > 1` only the sample equal to
the rank-0 `sidx` is replaced by its refined sequence.

### 4.7 Featurizer branches (A0 anchors, `potts_mpnn_utils.py`)

Verified against the spec's 4.1a bullet list: MSE->MET `:107`; first-occurrence-wins per (resn, icode, atom) `:128-139`; icode split `:119-121`; gap rows `:143-156`
(`skip_gaps` `:147-149`); `-`->`X` `:357,391`; sorted designed then sorted visible chains `:327-329`; `residue_idx = 100*(c-1)+arange` `:373,408`; fixed positions per chain
1-based `fixed_position_mask[fixed_pos_list-1] = 0` `:411-415`; omit-AA per position `:417-424`; PSSM dict `:425-435`; `bias_by_res` `:436-439`; tied groups and `tied_beta`
(listed-beta only for the `isinstance(v[0], list)` form, else ones) `:443-461`; `present = isfinite(sum X)` then `X[nan]=0` `:510-512`. `tied_featurize` default `tied_beta = np.ones(L_max)` `:444`.
`_dist` (K = `min(top_k, L)` `:1150`, invalid pairs get row `D_max` `:1147-1148`, `eps=1e-6` `:1143`).

## 5. aminx-side probes (spec 4.2, 4.1)

### 5.1 Hardcoded float32 on encoder/decoder paths (spec 4.3 precision prerequisite)

Command: `rg -n "float32|float16|bfloat16|jnp\.float_" src/aminx/model src/aminx/utils src/aminx/inference/decode`.
Result for `src/aminx/model/**`:

| File:line | Site | Effect on the f64 parity tier |
|---|---|---|
| `model/decoder.py:338-341` | `message_f32 = message.astype(jnp.float32); aggregated_message_f32 = jnp.sum(message_f32, -2) / scale; aggregated_message = aggregated_message_f32.astype(message.dtype)` in `DecoderLayer.__call__` | **the only hardcoded float32 accumulation on the encoder/decoder core**. Downcasts f64 messages, so `DecoderLayer` cannot reach rtol 1e-12 (spec prerequisite confirmed; `jnp.promote_types(message.dtype, jnp.float32)` fixes it) |
| `model/ligand_features.py:118,359,361` | `same_chain.astype(jnp.float32)`, `jnp.zeros_like(mask, dtype=jnp.float32)`, `chain_mask.astype(jnp.float32)` | LigandMPNN feature masks only; not on the Potts/MPNN encoder/decoder path |
| `model/diffusion_mpnn.py:37,40` | timestep embedding `jnp.float32` | diffusion model only |
| `model/encoder.py`, `model/features.py`, `model/mpnn.py`, `model/mpnn_core.py`, `model/dropout.py` | no float32/bfloat16/float16 literals | clean (`dropout.py:49` is a `jnp.bool_`) |

Outside `model/`, on paths a Potts driver would reuse: `utils/autoregression.py:268,317,319` (`.astype(jnp.float32)` AR/wave masks and `ones-eye`),
`inference/decode/_kernel.py:159`, `inference/decode/autoregressive.py:403` (`position_selects_group.astype(jnp.float32)`),
`inference/decode/ste.py:194-235`. The driver builds its own masks (spec 4.3 row kernel) and must not call these `float32` helpers under x64.
`utils/coordinates.py` (`compute_backbone_distance`) has no dtype literal (`1e-6` python float promotes with the input dtype).

### 5.2 `ProteinFeatures` `neighbor_indices[..., 0] == self` behaviour (spec 4.1 self-edge invariant; static read of `model/features.py`)

- Computed path (`features.py:143-186`): `distances = compute_backbone_distance(...)` = `sqrt(1e-6 + sum((CA_i - CA_j)^2))` on the **CA** atom (`utils/coordinates.py:158-166`), so the
  self distance is `sqrt(1e-6) = 1e-3`, strictly below every other pair unless another present residue's CA is within ~1e-3 A of residue i. Then
  `distances_masked = where(mask[:,None]*mask[None,:], distances, inf)` (`:165-171`), optional cross-structure `inf` (`:173-181`),
  `k = min(self.k_neighbors, structure_coordinates.shape[0])` (`:183`; static, from the padded length), `top_k(-distances_masked, k)` (`:184`, `jax.lax.top_k`, lower index wins ties).
- Therefore **for every row with `mask[i] == 1` the self index is at slot 0**, except exact/near-exact CA coincidence with a lower-indexed present residue (then the lower index wins the tie or the smaller distance).
- For rows with `mask[i] == 0` (gap or pad) every entry of the row is `inf`, all tie, and `top_k` returns indices `0..k-1` ascending: **self is at slot 0 only if `i == 0`**, and self appears in the row only if `i < k`.
  This is the spec's "gap rows before pad rows" tie-break (4.1a table), and it means the self-edge invariant test must be restricted to `present` rows exactly as the spec says.
- Precomputed path (`rbf_features`/`neighbor_indices` from proxide, `features.py:~130-134,192`): `neighbor_indices` is taken as given; self-at-slot-0 is then a property of the proxide producer, not of this module. PottsMPNN `batches()` (A0) does not use it.
- Upstream parity: `_dist` uses the same `eps=1e-6` and `D_adjust = D + (1 - mask_2D) * D_max` (`potts_mpnn_utils.py:1143-1150`); `torch.topk(largest=False)` is not tie-stable, so tie behaviour (spec `knn_boundary_tie`, `gap_row_knn_tiebreak`) is the only observable difference.

## 6. LASErMPNN

### 6.1 Checkpoints, `model_params` dims, ligand-encoder keys (spec 5.1, 5.2)

Loader: `load_model_from_parameter_dict` (`run_inference.py:488-508`): reads `loaded_dict['params']`, `['model_state_dict']`, **`['ligand_encoder_params']`** from the same file,
builds `LASErMPNN(ligand_encoder_params=..., **params['model_params'])`, `load_state_dict(strict=strict)`, `.eval()`.

| File (SHA-256 = pin) | top keys | n state keys | `strict=False` missing / unexpected | `strict=True` |
|---|---|---|---|---|
| `laser_weights_0p1A_nothing_heldout.pt` (`304fe02a...`; **default**, `run_inference.py:769`) | `params, model_state_dict, optimizer_num_steps, optimizer_rate, ligand_encoder_params, resume_epoch` | 492 | `[]` / `[]` | loads |
| `laser_weights_0p1A_noise_ligandmpnn_split.pt` (`4c76f879...`) | `params, model_state_dict, ligand_encoder_params, resume_epoch` | 492 | `[]` / `[]` | loads |
| `soluble_weights_no_heldout_drop_clusters_optstep_65000.pt` (`5046dcfd...`) | same as nothing_heldout | 492 | `[]` / `[]` | loads |
| `pretrained_ligand_encoder_weights.pt` (`3f62f5a3...`) | `params, model_state_dict` only (no `ligand_encoder_params`, not a LASEr checkpoint; ligand-encoder pretraining artifact, 128 keys incl. SPICE heads `predict_dipoles_layer`, `partial_charge_pred_layer`, ...) | 128 | not constructible as `LASErMPNN` | n/a |

The three LASEr checkpoints have **identical `params['model_params']` and identical `ligand_encoder_params`** and identical key sets. No blocking finding (all missing sets empty).

**`params['model_params']` (all three checkpoints):**
`node_embedding_dim=256` (Hs), `protein_edge_embedding_dim=128` (E), `ligand_edge_embedding_dim=128`, `num_laser_vectors=10` (V),
`num_encoder_layers=3`, `num_decoder_layers=3`, `num_attention_heads=1`, `atten_head_aggr_layers=0`, `atten_dimension_upscale_factor=4`,
`additional_ligand_mlp=True`, `dropout=0.1`, `build_hydrogens=True`, `chi_angle_rbf_bin_width=5`,
`graph_structure = {pr_pr_knn_graph_k: 48, lig_pr_knn_graph_k: 48, lig_lig_knn_graph_k: 5, lig_pr_distance_cutoff: 20.0}`,
`prot_prot_edge_rbf_params = {bin_min: 2, bin_max: 22, num_bins: 16}`, `lig_prot_edge_rbf_params = {bin_min: 0.0, bin_max: 15.0, num_bins: 75}`,
`lig_lig_edge_rbf_params = {bin_min: 0.0, bin_max: 15.0, num_bins: 75}`.
**`ligand_encoder_params['model_params']`:** `node_embedding_dim=256` (Hl), `ligand_edge_embedding_dim=128`, `num_ligand_encoder_vectors=15` (Vl), `num_encoder_layers=3`,
`num_attention_heads=3`, `atten_head_aggr_layers=0`, `atten_dimension_upscale_factor=None`, `dropout=0.1`, `graph_structure={lig_lig_knn_graph_k: 5}`,
`lig_lig_edge_rbf_params` as above; (training-only keys `batch_size, device, learning_rate, path_to_dataset, ...` present, ignore).
**Derived dims (instantiated model, 20,599,412 parameters):** `num_chi_bins = 360/5 = 72` (`build_rotamers.py:87`), `chi_embedding_dim = 72*4 = 288`;
`ligand_featurizer.output_dim = 26` (`ligand_input_layer` (256,26)); `prot_prot_edge_input_layer` (128,500) = `16*25 + 10^2`; `lig_prot_edge_input_layer` (128,385) = `75*5 + 10`;
`sequence_label_embedding` (22,256) (20 AA + X + NotDecoded), `sequence_output_layer` (21,256); `chi_prediction_layers.k` `Linear(2*256 + k*72 -> 256 -> 256 -> 72)`;
`chi_offset_prediction_layers.k` `Linear(144 -> 256 -> 256 -> 1)`; `chi_vector_update_layers.{0,1,2}` GVP; `ligand_encoder_output_gvp` `DenseGVP((256,15) -> (256,15) -> (256,0))`
(present because `additional_ligand_mlp=True`); backbone frame GVP `wh (10,4)`; `backbone_coords` are `(L,5,3)` (N, CA, CB, C, O) so 25 atom-pair distances.
Spec 5.1 `Nbins = 72`, `Hs=256`, `V=10`, `E=128`, `Hl=256`, `Vl=15`, `K=48`, `Kl=48`, `k_ll=5`, cutoff 20.0.

**State-dict key groups (nothing_heldout; identical for the other two):** `hbond_network_detector` 9 (buffers), `rotamer_builder` 35 (buffers), `ligand_featurizer` 6 (buffers),
`ligand_encoder` 57, `ligand_encoder_output_gvp` 20, `protein_encoder_layers` 150, `protein_decoder_layers` 138, `chi_prediction_layers` 24, `chi_offset_prediction_layers` 24,
`chi_vector_update_layers` 18, `prot_prot_rbf_encoding` 1, `lig_prot_rbf_encoding` 1, `prot_prot_edge_input_layer` 2, `lig_prot_edge_input_layer` 2, `backbone_frame_vec_input_layer` 2,
`sequence_label_embedding` 1, `sequence_output_layer` 2. Dtypes: 458 float32, 25 int64, 9 bool. `rotamer_builder.*` and `hbond_network_detector.*` (44 keys) are
buffers persisted **in the checkpoint** (e.g. `rotamer_builder.ideal_aa_coords (21,14,3)`, `ideal_bond_lengths (21,4)`, `ideal_bond_angles (21,4)`, `aa_to_chi_angle_atom_index (21,4,4)`,
`index_to_degree_bin (72,)`); B1.5 must classify buffers vs parameters. One odd tensor: `ligand_encoder_output_gvp.dropout.vdropout.dummy_param`, shape `(0,)` (an empty `Parameter` in `_VDropout`), must be
mapped or explicitly dropped for "0 unmapped keys".

**Ligand-encoder keys are in the main checkpoint (spec 5.2 "T0.2 confirms keys"): yes.** The main checkpoint carries `ligand_featurizer.*` (6), `ligand_encoder.*` (57) and `ligand_encoder_output_gvp.*` (20) = 83 keys
containing `ligand`; the `pretrained_ligand_encoder_weights.pt` file is **not needed**. The 57 `ligand_encoder.*` tensors are bit-identical to the same-named tensors in the pretrained file
(57/57 `torch.equal` under the `ligand_encoder.` prefix); `ligand_featurizer.atomic_number_idx_to_period_idx`, `..._group_idx` and `hydrogen_encoding` are equal too, whereas
`sequence_index_to_atomic_number_index`, `amino_acid_index_to_featurization`, `amino_acid_index_to_is_hydrogen_mask` differ between the files; those three feed only
`generate_ligand_nodes_from_amino_acid_labels` (pseudo-ligand training path), which inference never calls (`sampled_pseudoligands is None`, section 6.9), so take all buffers from the main checkpoint.
The class for the spec's "LigandFeaturizer" is `utils/ligand_featurization.py:7` (see C7).

### 6.2 `LASER_ALPHABET` (spec 5.1a)

Derived on titanix from `LASErMPNN.utils.constants.aa_idx_to_short`: `"".join(aa_idx_to_short[i] for i in range(21))` =
**`'ARNDCEQGHILKMFPSTWYVX'`** (exactly the spec string). Source: `aa_long_to_idx` (`utils/constants.py:34`: ALA 0, ARG 1, ASN 2, ASP 3, CYS 4, GLU 5, GLN 6, GLY 7, HIS 8, ILE 9, LEU 10, LYS 11,
MET 12, PHE 13, PRO 14, SER 15, THR 16, TRP 17, TYR 18, VAL 19, XAA 20), `aa_short_to_idx` `:35`, `aa_idx_to_short` `:37`. Index 21 is the model-only "NotDecoded" embedding row
(no letter). Note E/Q order: **`E`(5) before `Q`(6)**, i.e. NOT the aminx/ProteinMPNN order `ACDEFGHIKLMNPQRSTVWYX`, so the permutation is non-trivial.

Verbatim `utils/constants.py:31-37` (pinned e70f2c6d; left gutter = source line number):

```python
31| aa_short_to_long = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET', 'X': 'XAA'}
32|
33| aa_long_to_short = {x: y for y, x in aa_short_to_long.items()}
34| aa_long_to_idx = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'XAA': 20}
35| aa_short_to_idx = {x: aa_long_to_idx[y] for x, y in aa_short_to_long.items()}
36| aa_idx_to_long = {x: y for y, x in aa_long_to_idx.items()}
37| aa_idx_to_short = {x: aa_long_to_short[y] for x, y in aa_idx_to_long.items()}
```

### 6.3 Entry-point defaults relevant to `LaserOptions` (spec 6.1)

- Single `run_inference.py`: `--temp` is **`type=str, default=''`** (`:777`), converted by `float(x) if x else None` (`:796`); `--fs_sequence_temp` float; `--disable_charged_fs`; `--repack_only`
  (`:785`); `--ignore_ligand`; `--noncanonical_aa_ligand`; `--fs_calc_ca_distance` 10.0, `--fs_calc_burial_hull_alpha_value` 9.0, `--fs_no_calc_burial`; `--fix_beta`; `--ebd`
  (entropy decoder); `--ignore_statedict_mismatch` -> `strict_load` (`store_false`, `:783`). No `disabled_residues` flag: `sample_model(disabled_residues=['X'])` default (`:514`), so **single-input default disables only `X`**.
  No `chi_temp`, `seq_min_p`, `chi_min_p`, `ala/gly_budget` CLI flags (budget mask is `None`, so budgets are inert).
- `run_batch_inference.py`: `--sequence_temp` **`type=float, default=None`** (`:370`), `float(x) if x else None` (`:245`) so `0.0 -> None -> argmax`; `--chi_temp` same (`:246`); `--seq_min_p`, `--chi_min_p`
  default 0.0; `--disabled_residues` default `'X,C'`; `--ala_budget 4`, `--gly_budget 0` (only active with `-c` or `--budget_residue_sele_string`); `--repack_only_input_sequence` and `--repack_all`
  (`:203,217`); `--ignore_key_mismatch` (`store_false`, `:379`, passed as `strict=`, `:250`); `--output_fasta`, `--output_fasta_only`; `--designs_per_batch` default 30, `--inputs_processed_simultaneously` default 5. `ignore_chain_mask_zeros` and
  `bb_noise` exist on `_run_inference` (`:139-153`) but **no CLI flag and `run_inference()` never passes them**, so via the CLIs they are always `False`/`0.0`.
- `run_inference_tied.py`: `disabled_residues` default `['X','C']` in its `sample_model` (`:568`); see section 6.8 for dead flags.

The spec's `LaserOptions.disabled_residues=("X","C")` therefore matches the batch and tied CLIs but not the single-input CLI (C6).

### 6.4 χ binning: `bin` and the encoding (spec 5.4a "T0.2 transcribes `bin`")

`chi_log_prob[i,k] = log_softmax(chi_logits_k)[bin(chi[i,k])]` where, in upstream teacher forcing (`forward` `:383-385`), the bin is
`chi_bin_center = argmax(compute_binned_degree_basis_function(chi).nan_to_num(), dim=-1)`. Bin centres `index_to_degree_bin = arange(-180, 180, 5)` (72 values,
`CHI_BIN_MIN, CHI_BIN_MAX = -180, 180`, `utils/constants.py:12`). The RBF is a circular Gaussian with sigma = bin_width/2 = 2.5 deg normalised to sum 1, so the argmax equals the
nearest bin centre by circular distance **with ties (exactly half-way angles, e.g. -177.5) resolved by `torch.argmax` first-max**, i.e. the lower bin index. NaN angles: the RBF row is NaN,
`.nan_to_num()` gives all-zero, `argmax` = 0 (bin index 0), which is why `forward` masks with `~isnan(chi)` (`:425`, `:433`) and the spec sets `chi_log_prob` NaN where chi is NaN.

Verbatim `utils/build_rotamers.py:87-89` (pinned e70f2c6d; left gutter = source line number):

```python
87|         self.num_chi_bins = int((180 * 2) / self.chi_angle_rbf_bin_width)
88|         self.chi_angle_embed_dim = self.num_chi_bins * 4
89|         self.register_buffer('index_to_degree_bin', torch.arange(CHI_BIN_MIN, CHI_BIN_MAX, self.chi_angle_rbf_bin_width).float())
```
Verbatim `utils/build_rotamers.py:94-124` (pinned e70f2c6d; left gutter = source line number):

```python
 94|     def compute_binned_degree_basis_function(self, degrees: torch.Tensor, std_dev: Optional[float] = None) -> torch.Tensor:
 95|         """
 96|         Given degrees tensor (N, 4) of degrees between -180 and 180, computes the density of that value in circularly symmetric bin space.
 97|             Degree values can be NaN.
 98|
 99|         Basically implements an RBF for degrees between [-180, 180) with a standard deviation of half the bin width in degrees by
100|             default in modulus 360 degrees so -180 and 179 are 1 degree apart.
101|
102|         Each bin is a gaussian centered at the bin center (increments of 5 degrees) with a standard deviation of 2.5 degrees by default.
103|         """
104|         if degrees.shape[0] == 0:
105|             return torch.empty(0, 4, self.chi_angle_embed_dim, device=degrees.device)
106|
107|         # Default standard deviation to half the bin width in degrees.
108|         if std_dev is None:
109|             std_dev = self.chi_angle_rbf_bin_width / 2
110|
111|         # Reshape the input to allow subtraction broadcasting with the bin centers.
112|         degree_input_shape = degrees.shape
113|         if degrees.dim() != 1:
114|             degrees = degrees.flatten().unsqueeze(-1)
115|
116|         # Compute offset in degrees relative to the current angle in circular bin space.
117|         bin_degrees_exp = self.index_to_degree_bin.expand(degrees.shape[0], -1) # type: ignore
118|         circular_bin_distance = torch.minimum(torch.remainder(bin_degrees_exp - degrees, 360), torch.remainder(degrees - bin_degrees_exp, 360))
119|
120|         # Encode offset in a gaussian with standard deviation of 5 degrees by default meaning the width of 1 bin is within +/- 1 of std_dev
121|         A = torch.exp((-1/2) * (circular_bin_distance / std_dev) ** 2)
122|
123|         # Returns encoding normalized to sum to 1
124|         out = A / A.sum(dim=1).unsqueeze(-1)
```

`forward` bin/offset targets (for reference; the score port only needs the bin):

Verbatim `utils/model.py:382-390` (pinned e70f2c6d; left gutter = source line number):

```python
382|         # Use ground-truth chi angles discretized into bins with an associated continuous bin offset which we can learn jointly.
383|         chi_angle_encoding = self.rotamer_builder.compute_binned_degree_basis_function(batch.chi_angles).nan_to_num()
384|         chi_bin_center = chi_angle_encoding.argmax(dim=-1)
385|         discrete_chi_angle_encoding = F.one_hot(chi_bin_center, num_classes=self.rotamer_builder.num_chi_bins).float()
386|         chi_angle_discrete_degrees = self.rotamer_builder.index_to_degree_bin[chi_bin_center]
387|         true_chi_offset_pos = torch.remainder(batch.chi_angles - chi_angle_discrete_degrees, 360)
388|         true_chi_offset_neg = -torch.remainder(chi_angle_discrete_degrees - batch.chi_angles, 360)
389|         abs_min_indices = torch.stack([true_chi_offset_pos.abs(), true_chi_offset_neg.abs()], dim=-1).argmin(dim=-1)
390|         target_chi_offsets = torch.where(abs_min_indices == 0, true_chi_offset_pos, true_chi_offset_neg)
```

### 6.5 Draw sites, order-draw line, RNG consumers (spec 7.1 shim 1 and 2)

Every stochastic draw in the pinned inference path (LASEr):

| Site | Line | Draw | Shim key (sample, step[, chi, structure]) |
|---|---|---|---|
| `sample` sequence | `utils/model.py:859` `torch.distributions.Categorical(probs=curr_out_probs).sample()` | one draw per decoded row | (sample, step) |
| `sample` chi | `utils/model.py:899` `Categorical(probs=chi_probs).sample()` (only when `chi_angle_sample_temperature` is not None; else argmax `:895`) | one per (row, chi index 0..3) | (sample, step, chi) |
| `tied_sample` sequence | `utils/model.py:627` `Categorical(probs=interpolated_probs).sample()` (single draw from the λ-mix, shared) | one per row | (sample, step) |
| `tied_sample` chi | `utils/model.py:676` (structure 1) then `:677` (structure 2), **two independent draws in that order per chi index** (only when `chi_angle_sample_temperature` set) | 2 per (row, chi) | (sample, step, chi, structure) |
| decoding order | `utils/pdb_dataset.py:1641` `rand_urns = torch.rand((submask.sum(),), device=...) + mask_idx` inside `_masked_sort_for_decoding_order` (`:1618-1650`), called per batch element from `BatchData.generate_decoding_order(stack_tensors=True)` (`:610-635`), which `run_inference.py:538`, `run_inference_tied.py` (`generate_decoding_order(stack_tensors=True)` then copies to batch 2), and `run_proofreading.py:53` (flat order, unconditional pass) and `:136` (stacked order, conditional pass) call | tier-0, tier-1, tier-2 uniforms in that order, each in ascending row index; order = `all_indices[sort_keys.argsort()]` | order stream |
| entropy decoder (`sample_by_lowest_entropy`) | `utils/model.py:1065,1104` | excluded (spec non-goal) | n/a |

**Order-draw line for the shim (spec 7.1 shim 2, "T0.2 cites line"): `utils/pdb_dataset.py:1641`.** The shim replaces `torch.rand` in that function (module-local), leaving all other
`torch.rand` uses alone.

RNG consumers that are *not* draws of the parity surface but interleave the global torch stream (relevant only to unshimmed U2 runs): `pdb_dataset.py:414` `torch.randn((N,1,3))` bb noise (only if `bb_noise > 0`);
`:593` `torch.randn_like(all_lig_coords)` **always executed when ligands exist** (multiplied by `ligand_training_noise = 0.0`, so no value effect but it advances the stream); `:533` `torch.rand(1)` per ligand-bearing complex;
`build_rotamers.py:211,213` `torch.rand_like(phi_psi)` (random phi/psi for NaN angles at chain breaks, post-decode rotamer/hydrogen placement; the `laser_rotamers` wave must avoid NaN phi/psi or shim this);
`compute_fast_ligand_burial_mask(..., num_rays=5)` in `output_batch_data` (`run_inference.py:~327`) may consume RNG (not instrumented).

Verbatim `utils/pdb_dataset.py:610-635` (pinned e70f2c6d; left gutter = source line number):

```python
610|     def generate_decoding_order(self, stack_tensors: bool = False) -> None:
611|         """
612|         Inputs:
613|             stack_tensors (bool):
614|                 If False, decoding order is a (N, ) tensor of indices into the (N, ) dimensional tensors.
615|                 If True, stacks decoding orders into a batch dimension for parallel decoding.
616|                     Used for autoregressive sampling in model.sample function.
617|                     Results in (B, sub_N_max) tensor of indices into the (N, ) dimensional tensors.
618|         """
619|         if not stack_tensors:
620|             self.decoding_order = _masked_sort_for_decoding_order(self.chain_mask, self.extra_atom_contact_mask)
621|         else:
622|
623|             batched_decoding_orders = []
624|             for batch_idx in range(self.batch_indices.max() + 1):
625|                 curr_batch_mask = self.batch_indices == batch_idx
626|                 output = _masked_sort_for_decoding_order(self.chain_mask, self.extra_atom_contact_mask, curr_batch_mask)
627|                 batched_decoding_orders.append(output)
628|
629|             output = []
630|             max_size = max([x.shape[0] for x in batched_decoding_orders])
631|             for tensor in batched_decoding_orders:
632|                 output.append(torch.cat([tensor, torch.full((max_size - tensor.shape[0],), torch.nan, device=self.device)]))
633|             self.decoding_order = torch.stack(output)
634|
635|     def _apply_mask_to_residue_metadata_tensors(self, mask: torch.Tensor) -> None:
```
Verbatim `utils/pdb_dataset.py:1618-1650` (pinned e70f2c6d; left gutter = source line number):

```python
1618| def _masked_sort_for_decoding_order(chain_mask, extra_atom_contact_mask, curr_batch_mask: Optional[torch.Tensor] = None):
1619|     """
1620|     Generates a permutation order for the indices of (N,) dim tensors that are True in curr_batch_mask.
1621|     Randomly permutes indices in the order masks are added to ordered_mask_list.
1622|     """
1623|     ordered_mask_list = []
1624|
1625|     if curr_batch_mask is None:
1626|         curr_batch_mask = torch.ones_like(chain_mask, dtype=torch.bool)
1627|
1628|     # First decode the provided residues in batch.order_mask (1 in chain_mask).
1629|     ordered_mask_list.append(chain_mask & curr_batch_mask)
1630|     # Decode residues that are 0 in chain chain mask but not in contact with random atoms.
1631|     ordered_mask_list.append((~chain_mask) & (~extra_atom_contact_mask) & curr_batch_mask)
1632|     # Decode residues in contact with random atoms not provided as sequence context last.
1633|     ordered_mask_list.append(extra_atom_contact_mask & (~chain_mask) & curr_batch_mask)
1634|
1635|     all_indices = []
1636|     sort_keys = []
1637|     for mask_idx, submask in enumerate(ordered_mask_list):
1638|         # Get indices of residues in the current mask.
1639|         indices = submask.nonzero().flatten()
1640|         # Generate URNs offset by mask index to ensure sort preserves order in ordered_mask_list.
1641|         rand_urns = torch.rand((submask.sum(),), device=chain_mask.device) + mask_idx
1642|         all_indices.append(indices)
1643|         sort_keys.append(rand_urns)
1644|     all_indices = torch.cat(all_indices, dim=0)
1645|     sort_keys = torch.cat(sort_keys, dim=0)
1646|
1647|     # Sort the indices by the random keys.
1648|     decoding_order = all_indices[sort_keys.argsort()]
1649|
1650|     return decoding_order
```

### 6.6 Graph construction: every `radius_graph` / `knn_graph` / `compute_ligand_protein_knn_graph` / `scatter_*` site and its dense replacement (spec 5.2 "T0.2 enumerates")

Static grep over `utils/*.py` and `run_*.py` (excluding the LigandMPNN copies): `rg -n "radius_graph|knn_graph|scatter|torch_scatter|torch_cluster" ...`.
**`radius_graph` has zero call sites**: `utils/model.py:5` imports it and nothing calls it (dead import; no replacement needed). `scatter_log_softmax` and `scatter_max`,
`scatter_min` (except the excluded entropy decoder) are also imported but unused in the ported path. Reachable-at-inference sites:

| # | Site | Upstream operation | Dense replacement (fixed shapes) |
|---|---|---|---|
| G1 | `pdb_dataset.py:450` (no-ligand complex) and `:518` (ligand complex) `knn_graph(CA, k=48, loop=True)` | k nearest CA neighbours **including self**, edge_index = [source(neighbour), sink(centre)] (torch_cluster `flow='source_to_target'`), `k` clipped to L when L < 48 | `d = cdist(CA, CA)` with `+inf` for pad/invalid rows and cols, `lax.top_k(-d, K_static)`, `K_static = min(48, L_pad)`; slot valid iff `slot < min(48, L_total)` and neighbour is a real row; self edge at slot 0 by strict-min. Ties (exact duplicates) are measure-zero (torch_cluster tie order is unspecified) |
| G2 | `pdb_dataset.py:519` `knn_graph(lig_coords, k=5, loop=True, batch=curr_subbatch_idces)` | per-ligand (subbatch id) kNN incl. self, `k` clipped to atoms in that ligand | pad ligand atoms to `A_pad`, `d` with `+inf` across different subbatch ids and pad rows, `top_k(-d, k_ll=5)`; slot valid iff `slot < min(5, n_atoms_in_that_ligand)`. Every atom has >= 1 edge (itself) |
| G3 | `pdb_dataset.py:520` -> `compute_ligand_protein_knn_graph` (`:797-828`) | for each protein residue: `ca_distances = cdist(CA, lig)` in float64; **connected residues** = those with any ligand atom `< 20.0` A (strict `<`, `:817`); for each connected residue the `min(48, A)` **nearest ligand atoms** (`argsort` on the row, `:825`), **not limited by the cutoff**; edge = [ligand atom, residue] (`:824-827`). The aliphatic-hydrogen branch (`:805-814`, incl. `scatter(..., 'max')` at `:809`) runs only if `use_aliphatic_ligand_hydrogens=False`; `construct_graphs` defaults it to **True** (`:401`) and the inference callers never pass it (`**params['model_params']['graph_structure']` carries only the four graph keys), so **`:805-814` is unreachable at inference** | `(L, Kl)` table, `Kl_static = 48` (`lig_pr_knn_graph_k`): `d = cdist(CA (L,3), lig (A,3))` in f64; `connected = (d < 20.0).any(axis=1)`; `idx = top_k(-d, Kl)`; slot valid iff `connected[i] and slot < min(48, A_real)`; whole row invalid if not connected. Observed on `4jnj-1_prot.pdb` (exploratory, single fixture): 101 of 115 residues connected, each with exactly 31 = `min(48, 31)` edges, 14 residues with 0 edges |
| G4 | `model_generics.py:156` `HeteroGATv2.aggregate_node_update`: `scatter_softmax(all_presoftmax_atten, all_sink_edges, dim=1)` | softmax **jointly over the concatenation** of protein-protein and ligand-protein edges per sink residue (`all_sink_edges = cat(...)`, `:195`; the decoder layers use `preexpanded_edges_forward`, same concatenation at `:228`) | concatenate the two dense neighbour axes to `(L, K + Kl)` with mask; `where(mask, softmax(where(mask, x, -inf)), 0)` (fully-masked row -> 0, guard the max-subtraction against `-inf`); attention dropout `:157` is identity at eval |
| G5 | `model_generics.py:159` `scatter(node_update, all_sink_edges, dim=1, reduce='sum', dim_size=N)` | per-sink weighted sum, empty sink -> 0 | `sum(mask * atten * update, axis=K)`, empty -> 0 |
| G6 | `model_generics.py:349` `HomoGATv2.forward`: `scatter_softmax(atten, e_idx[1], dim=1)` and `:352` `scatter(..., reduce='sum')` | same, single edge type (ligand encoder GAT layers over the ligand-ligand graph) | same as G4/G5 on `(A, k_ll)` |
| G7 | `model.py:1183` `LigandEncoderModule.forward`: `scatter(lig_coords[edge_index[0]], edge_index[1], dim=0, reduce='mean')` | mean neighbour coordinate per ligand atom (self included, loop=True) | `sum(mask * coords[idx], K) / sum(mask, K)`; every atom has its self edge so the denominator is >= 1; then `input_vecs = (avg - coord) / _norm_no_nan(...)` |
| G8 | `model.py:527,528` (tied) and `:782,:786` (sample): `scatter((seq_idx == A/G).float(), batch_indices, reduce='sum', dim_size=n_batch)` | per-structure ALA/GLY counts over **all decoded rows, fixed rows included** (uses `output_tensors.sampled_sequence_indices`, initial 21 = not decoded), `>=` budget, `& budget_residue_mask` | scalar carry counters `ala_count`, `gly_count` per structure (spec carry); note `gly_budget = 0` makes `>= 0` always true (glycine always disabled on budget rows); tied counts use **structure 1's** decoded sequence only |
| G9 | `model.py:76` `minp_warp_logits`: `sorted_indices_to_remove.scatter(1, sorted_indices, ...)` (torch `Tensor.scatter`, inverse permutation) | mask tokens with `probs < min_p * max_prob`, always keep the top-1 index | `remove = probs < min_p*max(probs); remove = remove.at[argmax].set(False)`; `min_p == 0.0` short-circuits to the identity (`:61-62`); ties for the max are all kept since `p_max >= min_p * p_max` for `min_p <= 1` |
| G10 | `torch.isin(edge_index[1], node_idces)` edge selection in `sample` (`:799,810`) and `tied_sample` (`:538,542,560,564`) | select edges whose sink is the row(s) decoded this step | the dense `(L,K)` layout makes this a row gather `E[node_idx]`; no analogue needed |

Not reachable at inference (documented so the "every site" list is complete): `pdb_dataset.py:1301` `knn_graph(k=48, loop=False)` and `hbond_network.py:321,322,446` `scatter` (dataset preprocessing
`compute_hydrogen_bond_contact_number`, `pdb_dataset.py:1248-1310`; `compute_first_shell_hbonding_ligand_mask` has no callers), `helper_functions.py:143` `scatter_mean` (training metric),
`pdb_dataset.py:1691` `scatter_add_` (k-means training clusters), `spice_dataset.py:107` `knn_graph` (ligand pretraining), `model.py:1034` `scatter_min` (entropy decoder, non-goal).

Additional ground truths from the exploratory `4jnj-1_prot.pdb` run (single fixture, not a finding): `pr_pr_edge_index` has exactly 48 edges per node (5520 = 115 x 48) with the self edge present for every node;
`lig_lig_edge_index` has exactly 5 edges per atom. The construction is per-`batch_index` loop (`pdb_dataset.py:440-574`), so multi-copy batches are independent graphs.

Verbatim `utils/pdb_dataset.py:797-828` (pinned e70f2c6d; left gutter = source line number):

```python
797| def compute_ligand_protein_knn_graph(
798|         curr_lig_coords, curr_complex_ca_coords, lig_pr_k, distance_cutoff,
799|         lig_index_offset, prot_index_offset, lig_lig_eidx, lig_atomic_number_idces, use_aliphatic_ligand_hydrogens
800| ) -> torch.Tensor:
801|     # Identify the residues with CA coordinates within connection radius to ligand.
802|     ca_distances = torch.cdist(curr_complex_ca_coords.double(), curr_lig_coords.unsqueeze(0).double()).squeeze(0)
803|
804|     num_ligand_atoms = curr_lig_coords.shape[0]
805|     if not use_aliphatic_ligand_hydrogens:
806|         # Use the ligand-ligand graph to identify the ligand atoms that are hydrogens covalently bonded to a carbon with a mask for curr_lig_coords.
807|         lig_lig_eidx_ = lig_lig_eidx[:, lig_lig_eidx[0] != lig_lig_eidx[1]]
808|         is_within_covalent_hydrogen_distance = (torch.cdist(curr_lig_coords[lig_lig_eidx_[0]].unsqueeze(1), curr_lig_coords[lig_lig_eidx_[1]].unsqueeze(1)) < COVALENT_HYDROGEN_BOND_MAX_DISTANCE).flatten().float()
809|         sink_is_within_covalent_hydrogen_distance = scatter(is_within_covalent_hydrogen_distance, lig_lig_eidx_[1], reduce='max', dim_size=curr_lig_coords.shape[0]).bool()
810|         is_aliphatic_hydrogen_mask = (lig_atomic_number_idces == 0) & sink_is_within_covalent_hydrogen_distance
811|
812|         # Set the distances to the aliphatic hydrogens to infinity.
813|         ca_distances[:, is_aliphatic_hydrogen_mask] = torch.finfo(ca_distances.dtype).max
814|         num_ligand_atoms = (~is_aliphatic_hydrogen_mask).sum().item()
815|
816|     # Identify the protein atoms within the connection radius to the ligand.
817|     connected_bb_indices = (ca_distances < distance_cutoff).sum(dim=1).nonzero().flatten()
818|
819|     # Update KNN-k parameter to handle case where we have fewer than k ligand atoms.
820|     curr_lig_pr_k = min(lig_pr_k, num_ligand_atoms)
821|
822|     # Construct a KNN graph between the connected protein atoms and the ligand atoms.
823|     #   Ordered as lig -> prot.
824|     curr_lig_pr_edge_index = torch.stack([
825|         ca_distances[connected_bb_indices].argsort(dim=1)[:, :curr_lig_pr_k].flatten() + lig_index_offset, # Keep indices for K nearest cg atoms by sorting along column dimension.
826|         connected_bb_indices.unsqueeze(-1).expand(-1, curr_lig_pr_k).flatten() + prot_index_offset # Duplicates backbone indices K times.
827|     ])
828|     return curr_lig_pr_edge_index
```

### 6.7 Order semantics, first-shell masks, noise (spec 5.4a, 5.5, 6.3; feeds C1-C4)

**Inference featurization sets `extra_atom_contact_mask` to all-False**: `run_inference.py:350` (single copy) and `:371` (multi-copy) `'extra_atom_contact_mask': zeros_bool`.
Consequently tier 2 of `_masked_sort_for_decoding_order` (`extra_atom_contact_mask & ~chain_mask`) is empty in every inference/proofreading call; the order is tier 0 (fixed, `chain_mask`) then tier 1 (all designable).
Exploratory check on `4jnj-1_prot.pdb`: `extra_atom_contact_mask.sum() == 0`.

**`first_shell_ligand_contact_mask` is overwritten by `construct_graphs`.** `output_batch_data` computes it as "CA within `first_shell_ca_distance` (10 A) of a ligand heavy atom, AND buried by an
alpha-hull if `first_shell_buried_only`" (`run_inference.py:321-328`), but `construct_graphs` unconditionally reassigns `self.first_shell_ligand_contact_mask = first_shell_masks` (`pdb_dataset.py:590`),
built from `compute_first_shell_node_idces` (`:756-786`, heavy-atom contact `< HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD = 5.0` A, `+0.3` for Gly/X CA, restricted to residues present in the ligand-protein
edge set) at `:528,555`. Every consumer (`fs_sequence_temp` vector `run_inference.py:544`, `disable_charged_fs` `:552`, tied `run_inference_tied.py`, proofreading `run_proofreading.py:55`, budget) reads the mask **after**
`construct_graphs`. Ligand-free complexes get an all-False mask (`pdb_dataset.py:449-454`). Exploratory check (`4jnj-1_prot.pdb`, three knob settings): pre-graph mask sizes 18 / 115 / 0 for
(default 10 A + burial) / (100 A, no burial) / (3 A + burial); **post-graph mask identical (17 residues: 11, 13, 15, 31, 33, 35, 38, 39, 40, 62, 69, 71, 73, 75, 91, 93, 108) in all three**, so
`fs_calc_ca_distance`, `fs_calc_burial_hull_alpha_value`, `fs_no_calc_burial` do not affect any consumed output.

Verbatim `utils/pdb_dataset.py:756-786` (pinned e70f2c6d; left gutter = source line number):

```python
756| def compute_first_shell_node_idces(
757|     fa_coords: torch.Tensor, seq_indices: torch.Tensor, ligand_coords: torch.Tensor, ligand_atomic_number_idces: torch.Tensor,
758|     curr_lig_pr_edge_index: torch.Tensor, prot_index_offset: int
759| ) -> torch.Tensor:
760|     """
761|     Identifies residues within the first shell of a pseudoligand and returns the indices of those residues.
762|     """
763|
764|     # Get the protein indices of the ligand-protein edge index.
765|     protein_eidx = (curr_lig_pr_edge_index[1, :] - prot_index_offset).unique()
766|     putative_contacting_residues = fa_coords[protein_eidx]
767|     seq_indices_ = seq_indices[protein_eidx]
768|
769|     is_gly_mask = (seq_indices_ == aa_short_to_idx['G']) | (seq_indices_ == aa_short_to_idx['X'])
770|     hydrogen_mask = (ligand_atomic_number_idces == 0)
771|
772|     # Compute the distance between the ligand heavy atoms and the putative contacting residue heavy atoms.
773|     contact_mask = torch.zeros_like(protein_eidx, dtype=torch.bool)
774|     not_gly_contact_mask = (torch.cdist(ligand_coords, putative_contacting_residues[~is_gly_mask][:, 4:]) < HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD).any(dim=-1)[:, ~hydrogen_mask].any(dim=-1)
775|     gly_contact_mask = (torch.cdist(ligand_coords, putative_contacting_residues[is_gly_mask][:, 1].unsqueeze(1)) < HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD + 0.3).any(dim=-1)[:, ~hydrogen_mask].any(dim=-1)
776|
777|     contact_mask[~is_gly_mask] = not_gly_contact_mask
778|     contact_mask[is_gly_mask] = gly_contact_mask
779|
780|     return protein_eidx[contact_mask]
781|
782|
783| def get_list_of_all_paths(path: str) -> list:
784|     """
785|     Recursively get a list of all paths of pytorch files in a directory.
786|     """
```
Verbatim `utils/pdb_dataset.py:586-591` (pinned e70f2c6d; left gutter = source line number):

```python
586|
587|         # Mask out dropped residue metadata and update the first shell mask.
588|         self._apply_mask_to_residue_metadata_tensors(full_batch_node_mask)
589|         first_shell_masks = torch.cat(first_shell_masks, dim=0) if len(first_shell_masks) > 0 else torch.empty((0,), dtype=torch.bool, device=self.device)
590|         self.first_shell_ligand_contact_mask = first_shell_masks
591|
```

`bb_noise`: `pdb_dataset.py:412-415`, applied only if `protein_training_noise > 0`: `noised = torch.round(backbone_coords, decimals=2) + noise * torch.randn((N, 1, 3))` (**one 3-vector per residue broadcast to all 5 backbone atoms,
i.e. a rigid per-residue translation, after rounding coordinates to 0.01 A**).

Verbatim `utils/pdb_dataset.py:412-415` (pinned e70f2c6d; left gutter = source line number):

```python
412|         # Noise the backbone coordinates with x,y,z noise for each frame independently.
413|         if protein_training_noise > 0.0:
414|             noised_backbone_coords = torch.round(self.backbone_coords, decimals=2) + (protein_training_noise * torch.randn((self.backbone_coords.shape[0], 1, 3), device=self.device))
415|             self.backbone_coords = noised_backbone_coords
```

### 6.8 `tied_sample`: parameters, ignored knobs, dead CLI flags (spec 5.4 "T0.2 verifies list")

`tied_sample` signature (`utils/model.py:460-465`), verbatim:

Verbatim `utils/model.py:460-465` (pinned e70f2c6d; left gutter = source line number):

```python
460|     def tied_sample(
461|             self, batch1: BatchData, batch2: BatchData, lambda_: float = 0.5, sequence_sample_temperature: Optional[Union[float, torch.Tensor]] = None,
462|             budget_residue_mask: Optional[torch.Tensor] = None, ala_budget: int = 4, gly_budget: int = 0,
463|             chi_angle_sample_temperature: Optional[float] = None, disabled_residues: Optional[list] = ['X'],
464|             disable_pbar: bool = False, repack_all: bool = False
465|     ) -> Tuple[Sampled_Output, Sampled_Output]:
```

**Parameters accepted:** `batch1, batch2, lambda_, sequence_sample_temperature, budget_residue_mask, ala_budget, gly_budget, chi_angle_sample_temperature, disabled_residues, disable_pbar, repack_all`.
**Not parameters of `tied_sample`** (the spec's list is confirmed exactly): `seq_min_p`, `chi_min_p`, `ignore_chain_mask_zeros`, `disable_charged_residue_mask`/`disable_charged_fs`, `fs_sequence_temp` (the wrapper turns it into a per-residue
tensor temperature, which then hits the `NameError` below), `return_encoder_embeddings`. Semantics visible in the body: disabled mask on `~chain_mask` rows only (`:605-609`); budget uses structure 1 counts (`:527-528`); T None -> `1e-6`
(`:617-618`) then a **sampled** draw (never argmax) from the λ-mix `λ P1 + (1-λ) P2` (`:626-627`); χ argmax if `chi_angle_sample_temperature is None` (`:669-671`) else plain softmax (no min-p, `:672-677`); χ offset heads see the raw
(un-warped) χ logits; input χ kept wherever `chain_mask` (all rows, no NaN test because `nan_to_num` at `:491-492`) unless `repack_all` (`:690-692`); `sequence`/`chi_mask` shared, `output_tensors_2.sampled_sequence_indices`
gets structure 1's fixed labels (`:647,650`).

`NameError` confirmed (spec divergence row `test_divergence_tied_fs_temp`): `:484-485`

Verbatim `utils/model.py:484-485` (pinned e70f2c6d; left gutter = source line number):

```python
484|         if sequence_sample_temperature is not None:
485|             assert (isinstance(sequence_sample_temperature, torch.Tensor) and (sequence_sample_temperature.shape[0] == batch.num_residues or sequence_sample_temperature.numel() == 1)) or isinstance(sequence_sample_temperature, (int, float)), f"Sequence sample temperature must be a scalar or a tensor of shape (num_residues,). Got {sequence_sample_temperature}."
```

For a `float` temperature the first `and` operand is `False` and `isinstance(T,(int,float))` short-circuits, so no error; for a `Tensor` temperature (what `fs_sequence_temp` produces,
`run_inference_tied.py:606-610`) `batch.num_residues` is evaluated and `batch` is undefined -> `NameError`.

Dead / silently ignored flags on the tied CLI (`run_inference_tied.py`), relevant to the redsox alias rows:
- `--disable_charged_fs`: parsed (`:899`) and passed to `sample_model` (`run_inference_tied.py:837-840`), but `tied_sample`'s `disable_charged_residue_mask=` argument is **commented out** (`run_inference_tied.py:620`), so the flag has no effect.
- `--disabled_residues`: parsed (`:884`, default `'X,C'`) but `run_inference(...)` has no such parameter and never passes it; `sample_model` uses its own default `['X','C']` (`:568`). Dead.
- `--ebd/--entropy_decoder` -> `use_edo`: accepted by `sample_model` but unused in the tied path (`:565-620` never branches on it). Dead.
- `ignore_chain_mask_zeros` commented out at `:618`.
- `--temp` `float(x) if x else None` (`:903`); `--bb_noise` applied independently to each of the two structures (`:576-599`); `--budget_residue_sele_string` mask is built from structure 1 only (`:821-835`).

`sample_model` tied call, verbatim:

Verbatim `run_inference_tied.py:606-620` (pinned e70f2c6d; left gutter = source line number):

```python
606|
607|     if fs_sequence_temp is not None:
608|         sequence_temp = 1e-6 if sequence_temp is None else sequence_temp
609|         sample_temperature_vector = torch.full((batch_data1.num_residues,), sequence_temp, dtype=torch.float, device=model.device)
610|         sample_temperature_vector[batch_data1.first_shell_ligand_contact_mask] = fs_sequence_temp
611|
612|     sampling_output_1, sampling_output_2 = model.tied_sample(
613|         batch_data1, batch_data2,
614|         lambda_=interpolation_lambda,
615|         budget_residue_mask=budget_residue_mask, ala_budget=ala_budget, gly_budget=gly_budget,
616|         sequence_sample_temperature=sample_temperature_vector if sample_temperature_vector is not None else sequence_temp,
617|         chi_angle_sample_temperature=chi_temp, disable_pbar=disable_pbar,
618|         # ignore_chain_mask_zeros=ignore_chain_mask_zeros,
619|         disabled_residues=disabled_residues, repack_all=repack_all,
620|         # disable_charged_residue_mask=None if not disable_charged_fs else batch_data1.first_shell_ligand_contact_mask
```

### 6.9 `sample` step transcriptions and single-structure invariants (spec 5.3)

Steps 2-6 (`:833-865`) and χ loop (`:884-934`), verbatim, for the stage docstrings:

Verbatim `utils/model.py:833-866` (pinned e70f2c6d; left gutter = source line number):

```python
833|             # Convert node embeddings to logits for sequence prediction.
834|             curr_out_logits = self.sequence_output_layer(prot_node_stack[-1].scalars[node_idces])
835|             if disabled_residues is not None:
836|                 sampling_residue_mask = ~(curr_chain_mask.bool()) if not ignore_chain_mask_zeros else curr_chain_mask.bool()
837|                 for res_short in disabled_residues:
838|                     curr_out_logits[sampling_residue_mask, aa_short_to_idx[res_short]] = torch.finfo(curr_out_logits.dtype).min
839|
840|             curr_out_logits[curr_res_ala_over_budget, aa_short_to_idx['A']] = torch.finfo(curr_out_logits.dtype).min
841|             curr_out_logits[curr_res_gly_over_budget, aa_short_to_idx['G']] = torch.finfo(curr_out_logits.dtype).min
842|
843|             if disable_charged_residue_mask is not None:
844|                 curr_disable_charged_mask = disable_charged_residue_mask[node_idces]
845|                 curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['K']] = float('-Inf')
846|                 curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['R']] = float('-Inf')
847|                 curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['D']] = float('-Inf')
848|                 curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['E']] = float('-Inf')
849|
850|             # Sample sequence indices if temperature is specified, otherwise take argmax.
851|             if sequence_sample_temperature is None:
852|                 curr_out_sample = curr_out_logits.argmax(dim=-1)
853|             else:
854|                 curr_out_logits = minp_warp_logits(curr_out_logits, seq_min_p)
855|                 if isinstance(sequence_sample_temperature, torch.Tensor) and (sequence_sample_temperature.numel() > 1):
856|                     curr_out_probs = torch.softmax(curr_out_logits / sequence_sample_temperature[node_idces].unsqueeze(-1), dim=-1)
857|                 else:
858|                     curr_out_probs = torch.softmax(curr_out_logits / sequence_sample_temperature, dim=-1)
859|                 curr_out_sample = torch.distributions.Categorical(probs=curr_out_probs).sample()
860|
861|             # Use chain_mask to select from input sequence for partial-sequence design as needed.
862|             if not ignore_chain_mask_zeros:
863|                 sampled_or_fixed_sequence_idx = (curr_chain_mask * batch.sequence_indices[node_idces]) + ((1 - curr_chain_mask) * curr_out_sample)
864|             else:
865|                 sampled_or_fixed_sequence_idx = curr_out_sample
866|
```
Verbatim `utils/model.py:884-934` (pinned e70f2c6d; left gutter = source line number):

```python
884|             # Decode chi angles from final protein node embeddings.
885|             #   pull previously decoded chi angles from output tensors.
886|             curr_nodes_decoder_embeddings = prot_node_stack[-1].get_indices(node_idces)
887|             prot_scalars = curr_nodes_decoder_embeddings.scalars
888|             chi_prev = torch.empty((node_idces.shape[0], 0), device=self.device)
889|             for chi_idx, chi_layer in enumerate(self.chi_prediction_layers):
890|                 # Predict chi angles from final protein node embeddings.
891|                 chi_logits = chi_layer(torch.cat([prot_scalars, sequence_embeddings[node_idces], chi_prev], dim=1))
892|
893|                 # Sample chi angles if temperature is specified, otherwise take argmax.
894|                 if chi_angle_sample_temperature is None:
895|                     chi_sample = chi_logits.argmax(dim=-1)
896|                 else:
897|                     chi_logits = minp_warp_logits(chi_logits, chi_min_p)
898|                     chi_probs = torch.softmax(chi_logits / chi_angle_sample_temperature, dim=-1)
899|                     chi_sample = torch.distributions.Categorical(probs=chi_probs).sample()
900|
901|                 chi_sample_one_hot = F.one_hot(chi_sample, num_classes=self.rotamer_builder.num_chi_bins).float()
902|                 chi_sample_offset = self.chi_offset_prediction_layers[chi_idx](torch.cat([chi_logits, chi_sample_one_hot], dim=1)).squeeze()
903|
904|                 # Convert sampled index to angle, then to RBF encoding.
905|                 #   1 in chain mask tells us to sample chi angle, a 0 tells us to use the input chi angle.
906|                 sampled_angles = torch.remainder(self.rotamer_builder.index_to_degree_bin[chi_sample] + chi_sample_offset + 180, 360) - 180 # type: ignore
907|                 if not ignore_chain_mask_zeros and not repack_all:
908|                     nan_mask = ~input_chi_angles[node_idces, chi_idx].isnan()
909|
910|                     # This is only 1 if we are both trying to fix the identity of the residue and the residue is not NaN
911|                     sample_mask = (curr_chain_mask.bool() & nan_mask).long()
912|
913|                     # Keep sampled angles according to whether sample mask is 1 or 0.
914|                     sampled_angles = (sample_mask * input_chi_angles[node_idces, chi_idx].nan_to_num()) + ((1 - sample_mask) * sampled_angles)
915|
916|                 curr_chi_encoding = self.rotamer_builder.compute_binned_degree_basis_function(sampled_angles.unsqueeze(-1)).squeeze(1)
917|                 chi_prev = torch.cat([chi_prev, curr_chi_encoding], dim=1)
918|                 # curr_chi_encoding = torch.stack([torch.sin(sampled_angles.deg2rad()), torch.cos(sampled_angles.deg2rad())], dim=-1)
919|                 # chi_prev = torch.cat([chi_prev, curr_chi_encoding], dim=1)
920|
921|                 # Convert chi angle encoding to logits and store in output tensors.
922|                 curr_chi_masks = sampled_sequence_chi_masks[:, chi_idx]
923|                 output_tensors.chi_logits[node_idces[curr_chi_masks], chi_idx] = chi_logits[curr_chi_masks]
924|                 output_tensors.sampled_chi_encoding[node_idces[curr_chi_masks], chi_idx] = curr_chi_encoding[curr_chi_masks]
925|                 output_tensors.sampled_chi_degrees[node_idces[curr_chi_masks], chi_idx] = sampled_angles[curr_chi_masks]
926|
927|                 if chi_idx == 3:
928|                     # Don't need to update node representations for the last chi angle since there is no next chi angle to predict.
929|                     break
930|
931|                 gvp_layer, gvp_norm = self.chi_vector_update_layers[chi_idx], self.chi_vector_layer_norms[chi_idx]
932|                 concat_features = EquivariantData(torch.cat([prot_scalars, chi_prev], dim=1), curr_nodes_decoder_embeddings.vectors)
933|                 curr_nodes_decoder_embeddings = gvp_norm(gvp_layer(concat_features))
934|                 prot_scalars = curr_nodes_decoder_embeddings.scalars
```

Confirmed against the spec's 5.3 list: disabled mask sets `finfo.min` on `~chain_mask` rows (or `chain_mask` rows under `ignore_chain_mask_zeros`) `:835-838`; ALA/GLY budget `finfo.min` `:840-841`; charged residues `-inf`
(`:843-848`, applied to *all* rows in the mask, fixed rows included, so stored logits of fixed first-shell rows have `-inf` at K/R/D/E); T None -> argmax else `minp_warp` then `/T` then softmax then draw (`:850-859`);
fixed/sampled select (`:861-865`); `chi_logits` is **reassigned** by `minp_warp_logits` at `:897` before the offset head and before storage (`:923`) (post-warp), matching the spec's step 7.

**Upstream latent multi-structure bug under `ignore_chain_mask_zeros`** (documentation only; single-structure batches, the only shape aminx drives, are unaffected): `curr_chain_mask` (`:791`) and the budget vectors are computed *before* the
`node_idces = node_idces[curr_chain_mask.bool()]` filter (`:793-796`), so `sampling_residue_mask` at `:836` (`curr_chain_mask.bool()`) can have a different length from the filtered `curr_out_logits` when a batch holds more than one structure.

`num_adjacent_residues_to_drop` is inert at inference (spec 5.5 "T0.2 confirms"): it is only read inside `if self.sampled_pseudoligands is not None` blocks (`pdb_dataset.py:425,470,481-488,512`);
`sampled_pseudoligands` is set only by `BatchData.sample_pseudoligands` (`:207,303`), whose callers are the training scripts (`train_lasermpnn*.py:245,348,503,...`) and `tests/test_equivariance.py:168,220`. Inference callers pass
`0` (`run_inference.py:532`, `run_inference_tied.py:586,598`) or `6` (`run_proofreading.py:50,133`); no inference path calls `sample_pseudoligands`, so both are equivalent (confirmed statically; B0 gate should still assert graph equality for 0 vs 6).

### 6.10 Dropout: every `nn.Dropout` module and every `self.training` / `F.dropout` read (spec 2.1)

Enumerated on titanix from `load_model_from_parameter_dict(laser_weights_0p1A_nothing_heldout.pt)`, `model.named_modules()`: **74 dropout-bearing modules = 72 `nn.Dropout` + 1 `EquivariantDropout` (wrapper, no own state) + 1 `_VDropout`.**
Source reads: **`self.training` is read exactly once in the package, `_VDropout.forward` (`utils/model_generics.py:631`)**; there is no `F.dropout` / `torch.nn.functional.dropout` / `training=` call anywhere in the non-LigandMPNN tree (`rg` clean; the only other
hit is the comment `# if self.training:` at `utils/model.py:253`). The 72 `nn.Dropout` modules read `self.training` inside torch. State after `model.eval()` (which `load_model_from_parameter_dict` does, `:505`): **all 74 modules `training=False`**.
Under the proofreading rule (`run_proofreading.py:21-30`: `model.eval()` then `.train()` on every `isinstance(module, torch.nn.Dropout)` when dropout is enabled), the 72 `nn.Dropout` (including `ligand_encoder_output_gvp.dropout.sdropout`, an `nn.Dropout` child of the wrapper)
become `training=True`, while `EquivariantDropout` (the wrapper) and `_VDropout` stay `training=False` (`_VDropout` is not an `nn.Dropout` subclass), so vector-channel dropout is never active (spec confirmed).

Module paths, grouped (`i` = layer index):

| Path pattern | Count | `p` | Effective under proofread dropout (`p > 0` and `train()`) | Notes |
|---|---|---|---|---|
| `ligand_encoder.gat_layers.{0,1,2}.dropout` | 3 | 0.1 | yes | attention dropout `model_generics.py:350`; edge-update dropout `:328` uses the same module (layers 0,1 only) |
| `ligand_encoder.gat_layers.{0,1}.linear_edge_updates.layers.{1,4}` | 4 | **0.0** | inert | `DenseMLP` default `mlp_dropout=0.0` (`model_generics.py:403-410`) |
| `ligand_encoder_output_gvp.dropout.sdropout` | 1 | 0.1 | yes | inside `EquivariantDropout` (`model_generics.py:640-656`) |
| `ligand_encoder_output_gvp.dropout` (`EquivariantDropout`) | 1 | wrapper | stays eval | no own tensor state |
| `ligand_encoder_output_gvp.dropout.vdropout` (`_VDropout`) | 1 | 0.1 | **no** (stays eval, `:631`) | never active |
| `protein_{encoder,decoder}_layers.{i}.hetgat.dropout` | 6 | 0.1 | yes | joint attention dropout `:157` |
| `protein_{encoder,decoder}_layers.{i}.hetgat.subgats.{0,1}.dropout` | 12 | 0.1 | yes | per-edge-type `HomoGATv2.dropout` |
| `protein_{encoder,decoder}_layers.{i}.hetgat.dense_residual_node_update.dropout` | 6 | 0.1 | yes | `DenseResidualNodeUpdate` `:433-439` (used twice per call) |
| `protein_{encoder,decoder}_layers.{i}.hetgat.subgats.{0,1}.dense_node_update_layers.layers.{1,4}` | 24 | **0.0** | inert | |
| `chi_prediction_layers.{0..3}.layers.{1,4}` | 8 | 0.1 | yes (conditional pass / repack) | do not affect sequence logits in a single `forward`; they feed later steps only through sampled chi in `sample` when a chi is drawn (`repack_all` or NaN input chi) |
| `chi_offset_prediction_layers.{0..3}.layers.{1,4}` | 8 | 0.1 | yes (same caveat) | |

Counts check: p>0 `nn.Dropout` = 3 + 1 + 6 + 12 + 6 + 8 + 8 = **44**; p=0 (inert) = 4 + 24 = **28**; 44 + 28 = 72. For the **unconditional** proofreading pass (one `forward`, `run_proofreading.py:58`) the sequence logits depend on the
28 non-chi `p>0` modules (3+1+6+12+6); the 16 chi-head dropouts only matter for the conditional pass. The `nn.Dropout.forward` shim (spec 7.1 shim 3: injected masks keyed by (module path, call idx)) must cover all 44 `p>0` modules; a
`_VDropout` assertion (`training is False` for the one instance) is the spec's existing check. Decoder modules are called once per decode step (`sample`), so call indices run to `n_steps` per decoder-layer module.

### 6.11 LigandMPNN scripts (spec 6.4 exclusion) and fixtures

`run_inference_ligandmpnn.py` and `run_batch_inference_ligandmpnn.py` drive **LigandMPNN, not LASEr**: `run_inference_ligandmpnn.py:23-25` imports `LASErMPNN.utils.model_ligandmpnn.LigandMPNN` and `pdb_dataset_ligandmpnn`,
`:490-506` `load_model_from_parameter_dict -> LigandMPNN(**params['model_params'])`, `:512` `sample_model(model: LigandMPNN, ...)`; `run_batch_inference_ligandmpnn.py:19-21` imports `Sampled_Output`/`BatchData` from the `_ligandmpnn` modules and
`load_model_from_parameter_dict, sample_model` from `run_inference_ligandmpnn`. Also `train_ligandmpnn*.py`. **`duplicate` exclusion confirmed** (aminx already implements LigandMPNN).

Fixtures (B2): `databases/` = `README.md` + `dataset_split_info.zip` (six split JSON files: `ligandmpnn_train_data_{30pct_main_clusters,70pct_subclusters}.json`, `train_streptavidin_heldout_split_*`, `val_streptavidin_heldout_split_*`); `example_pdbs/` = `4jnj-1_prot.pdb` (73592 bytes; 115 protein rows in chains A, B; ligand BTN chain B, 31 heavy+H atoms, 31 HETATM lines).
PottsMPNN fixtures exist as specified: `inputs/example_pdbs/{2yc3,3dkm,3gg7,4jox,6w25,swe1_ligand}.pdb`; `energy_benchmark_datasets/megascale_test_subset.csv` has 202805 lines (all rows requested by the `potts_ddg_megascale` sidecar).

### 6.12 Proofreading (spec 5.4): `resindex` <-> row map, reduction, focus set

Transcribed core (`run_proofreading.py:108-160`):

Verbatim `run_proofreading.py:108-152` (pinned e70f2c6d; left gutter = source line number):

```python
108| def compute_conditional_probs(model, training_parameter_dict, pdb_file: Path, output_dir: Path, fs_mask, ylabels, n_decoding_orders: int, n_dropouts: int, repack_all: bool):
109|     stacked_probs = []
110|     stacked_probs_mean_plus_stdv = []
111|     for unfixed_index in tqdm(fs_mask.nonzero().flatten().tolist(), total=len(fs_mask.nonzero().flatten().tolist())):
112|         subbatch = []
113|         subbatch_stddv = []
114|         for _ in range(n_dropouts):
115|             # Get the protein hierview
116|             protein_hv = get_protein_hierview(str(pdb_file))
117|
118|             # Set all betas to 1.0 except for the unfixed residue
119|             protein_hv.getAtoms().setBetas(1.0)
120|             protein_hv.getAtoms().select(f'resindex {unfixed_index}').setBetas(0.0)
121|
122|             data = ProteinComplexData(protein_hv, str(pdb_file), verbose=False)
123|             batch_data = data.output_batch_data(fix_beta=True, num_copies=n_decoding_orders)
124|
125|             batch_data.to_device(model.device)
126|             batch_data.construct_graphs(
127|                 model.rotamer_builder,
128|                 model.ligand_featurizer,
129|                 **training_parameter_dict['model_params']['graph_structure'],
130|                 protein_training_noise = 0.0,
131|                 ligand_training_noise = 0.0,
132|                 subgraph_only_dropout_rate = 0.0,
133|                 num_adjacent_residues_to_drop = 6,
134|                 build_hydrogens = training_parameter_dict['model_params']['build_hydrogens'],
135|             )
136|             batch_data.generate_decoding_order(True)
137|
138|             sampling_output = model.sample(batch_data, sequence_sample_temperature=1.0, chi_angle_sample_temperature=1.0, disabled_residues=['X'], disable_pbar=True, repack_all=repack_all)
139|
140|             logits_reshaped = sampling_output.sequence_logits.reshape(n_decoding_orders, -1, 21).softmax(dim=-1).mean(dim=0)
141|             stddev_reshaped = sampling_output.sequence_logits.reshape(n_decoding_orders, -1, 21).softmax(dim=-1).std(dim=0)
142|             subbatch.append(logits_reshaped[unfixed_index])
143|             subbatch_stddv.append(stddev_reshaped[unfixed_index])
144|         stacked_probs.append(torch.stack(subbatch).mean(dim=0))
145|         stacked_probs_mean_plus_stdv.append(torch.stack(subbatch).mean(dim=0) + torch.stack(subbatch_stddv).mean(dim=0))
146|
147|     stacked_probs = torch.stack(stacked_probs)
148|     stacked_probs_mean_plus_stdv = torch.stack(stacked_probs_mean_plus_stdv)
149|     stacked_probs_mean_plus_stdv = stacked_probs_mean_plus_stdv.cpu().numpy()
150|
151|     probs_renormed = stacked_probs.cpu().numpy()
152|     probs_renormed = probs_renormed / probs_renormed.sum(axis=-1, keepdims=True)
```

Confirmed: focus set `fs_mask = batch_data.first_shell_ligand_contact_mask` (post-`construct_graphs`, `:55`) or, with `selection_string`, `resindices` of `(same residue as (sel)) and name CA` (`:69-79`; `None` -> `ValueError`, `:79`);
`residue_ids` = `f'{resname}-{resnum}'` for the mask rows (`ylabels`); per (focus, dropout rep) one batch of `n_decoding_orders` copies with **every residue except the focus set to B-factor 1.0** (`:119-120`, so `chain_mask=1` for all others via
`fixed_rotamers = isclose(max(bfac), 1.0)`, `run_inference.py:213`); `sample(T=1.0, chi_T=1.0, disabled_residues=['X'], repack_all=...)` (`:138`); `softmax` of stored logits, `mean(dim=0)`, `std(dim=0)` (torch default **unbiased**, `n=1 -> NaN`) (`:140-141`),
`stack(subbatch).mean(0)` for the mean and `mean + mean(std)` for `mean_plus_std` (`:145`); the saved `conditional_probs.pt` is the un-renormalised mean (`:154`, plus `..._mean_plus_stdv_no_norm.pt` `:155`), renormalisation is for the plots only. The focus residue is
decoded **last** in every order (it is the only non-fixed row, tier 1 vs tier 0). `ala_budget`/`gly_budget` defaults are inert (`budget_residue_mask=None`).

**`resindex` <-> B0 row map (spec 5.4 "T0.2 verifies"):** upstream *assumes* identity. It selects the focus row with `resindex {unfixed_index}` (`:120`) and, with a selection string, writes `new_mask[resindices] = True` from a fresh
`pr.parsePDB` (`:70-77`), treating ProDy `resindex` (counts **every** residue of the parsed `AtomGroup`: ligand, water, non-amino, dropped residues, in file order) as the batch row index. Rows are built by iterating
`HierView` chains then residues, keeping amino-acid residues with N/CA/C present (`run_inference.py:180-215`); waters skipped, ncAA-as-ligand skipped, incomplete backbone dropped. Identity therefore holds iff no non-row residue precedes
the focus residue in file order and chain iteration order equals file order. **Verified on `4jnj-1_prot.pdb` (exploratory): 115 rows, 116 residues in the AtomGroup (chains A and B), `resindex` of the 115 protein CA atoms == `arange(115)`** (BTN is last).
B0 must therefore emit an explicit `row_to_resindex` array and the proofread stage must map focus rows through it (upstream is wrong for inputs where identity fails; decide whether that is a divergence row or bit-parity; recommend a
divergence with a warning, since upstream would fix the wrong residue or raise).

## 7. `conditional_ids.txt` (spec 0 branch-coverage clause; content to commit at `tests/port/conditional_ids.txt`)

The instruction for this task forbids editing any other file, so the file content is reproduced here. Format: `id<TAB>anchor<TAB>note`. One id per documented upstream conditional that changes an output; owning task in the
last column of the note (A0/A3/A5/B0/B4/B5/B7). Each id needs a `[[branch]]` row (fixture + mutant) before its stage's gate.

```text
# PottsMPNN featurizer (A0)  -- potts_mpnn_utils.py @ 0cb0a58
potts.parse.mse_as_met	potts_mpnn_utils.py:107	HETATM MSE treated as ATOM MET (A0)
potts.parse.altloc_first_wins	potts_mpnn_utils.py:128-139	first occurrence per (resn,icode,atom) wins; also multi-MODEL (A0)
potts.parse.icode_split	potts_mpnn_utils.py:119-121	resnum with trailing alpha = insertion code (A0)
potts.parse.gap_row_fill	potts_mpnn_utils.py:143-156	missing resnum -> '-' row with NaN N/CA/C/O (A0)
potts.parse.skip_gaps	potts_mpnn_utils.py:147-149,261	skip_gaps drops gap rows (A0)
potts.parse.absent_chain_dropped	potts_mpnn_utils.py:183	chain not in file returns str -> dropped (A0)
potts.parse.unknown_resname_dash	potts_mpnn_utils.py:145,259	aa_3_N.get(name,20) unknown -> '-'/X slot (A0)
potts.feat.designed_chains_from_dict	potts_mpnn_utils.py:322-326	chain_dict entry non-empty -> designed/visible split; else all designed (A0)
potts.feat.chain_order_sorted	potts_mpnn_utils.py:327-329	sorted designed + sorted visible (A0)
potts.feat.visible_chain	potts_mpnn_utils.py:353-386	visible chain: chain_M=0 (A0)
potts.feat.masked_chain	potts_mpnn_utils.py:387-439	designed chain (A0)
potts.feat.dash_to_x	potts_mpnn_utils.py:357,391	'-' in chain seq -> 'X' (A0)
potts.feat.fixed_positions	potts_mpnn_utils.py:411-416	chain_M_pos=0 rows for fixed_positions (A0)
potts.feat.omit_aa_per_position	potts_mpnn_utils.py:417-424	omit_AA_mask (A0)
potts.feat.pssm_dict_present	potts_mpnn_utils.py:425-435	pssm coef/bias/log_odds vs defaults (A0)
potts.feat.bias_by_res_present	potts_mpnn_utils.py:436-439	bias_by_res (A0)
potts.feat.tied_groups_listed_beta	potts_mpnn_utils.py:449-456	listed [positions, betas] form (A0/A5)
potts.feat.tied_groups_plain	potts_mpnn_utils.py:457-459	plain positions form, beta=1 (A0/A5)
potts.feat.present_mask_nan_zero	potts_mpnn_utils.py:510-512	present=isfinite, X[nan]=0 (A0)
potts.knn.invalid_pair_dmax	potts_mpnn_utils.py:1146-1150	invalid pairs get row D_max; K=min(top_k,L) (A0/A3)
potts.knn.boundary_tie	potts_mpnn_utils.py:1144-1150	#present <= K_eff < L_total tie (excluded; A0)
# PottsMPNN energy / head (A3)
potts.head.row_mask_and_slot0_eye	potts_mpnn_utils.py:1795-1797	etab*mask, slot0 x eye(20) (A3)
potts.merge.denom2_all	etab_utils.py:172-183	denom==2: no self exclusion (A3)
potts.merge.denom4_exclude_self	etab_utils.py:177-178	denom!=2: reverse slot%k==0 excluded (A3)
potts.merge.no_reverse_edge_undivided	etab_utils.py:175-176	reverse_idx<0 -> entry stays undivided (A3)
potts.energy.x_gap_zero_slots	etab_utils.py:362-366; sample_seqs.py:211	'-'=20,'X'=21 zero-padded etab slots (A3)
potts.energy.no_half	etab_utils.py:299-309	calc_eners sums k and l, no 0.5 (A3)
# PottsMPNN AR decode (A5)
potts.sample.temp_zero_floor	sample_seqs.py:38	T==0 -> 1e-6 (A5)
potts.sample.opt_temp_zero_floor	sample_seqs.py:39	T_opt==0 -> 1e-6 (A5)
potts.sample.optimize_forces_n1	sample_seqs.py:40	optimize_pdb/fasta -> num_samples=1 (A5)
potts.sample.mode_none_empty	sample_seqs.py:41	mode 'none' -> '' (no refine) (A5)
potts.sample.fix_decoding_order_seed	sample_seqs.py:201-202,219-220	seed=string_to_int(pdb)+offset(+sidx) (A5)
potts.sample.tied_vs_untied_dispatch	sample_seqs.py:225-242	tied_decoder iff non-empty tie list (A5)
potts.decoder.order_key_chainM_pos	potts_mpnn_utils.py:1419-1421	argsort((cm*cmp*present+1e-4)|randn|) (A5)
potts.decoder.masked_row_true_branch	potts_mpnn_utils.py:1448-1449	present==0 row takes S_true, no layers (A5)
potts.decoder.fixed_select	potts_mpnn_utils.py:1483	S_t*cm+S_true*(1-cm) (A5)
potts.decoder.omit_bias_constants	potts_mpnn_utils.py:1466	-omit*1e8 + bias/T + bias_by_res/T (A5)
potts.decoder.pssm_precedence	potts_mpnn_utils.py:1467	(flag and coef.numel>0) or bias.numel>0 (A5)
potts.decoder.pssm_bias_mix	potts_mpnn_utils.py:1468-1470	(1-coef*multi)p+coef*multi*bias (A5)
potts.decoder.pssm_log_odds	potts_mpnn_utils.py:1471-1475	p*(mask+0.001) renorm (A5)
potts.decoder.omit_aa_mask_renorm	potts_mpnn_utils.py:1476-1479	omit_AA_mask renorm (A5)
potts.decoder.multinomial_draw	potts_mpnn_utils.py:1480	torch.multinomial (shim key (sample,step)) (A5)
potts.tied.group_order_first_member	potts_mpnn_utils.py:1606-1614	group placed at first member's position (A5)
potts.tied.overlap_double_decode	potts_mpnn_utils.py:1606-1614	overlap -> ValueError divergence (A5)
potts.tied.masked_member_all_S_true	potts_mpnn_utils.py:1641-1647	group with present==0 member: S_true[m*] to all (A5)
potts.tied.beta_weighted_logits	potts_mpnn_utils.py:1660-1661	sum tied_beta[t]*W_out/T/len (also singletons) (A5)
potts.tied.last_member_bias_pssm_omit	potts_mpnn_utils.py:1665-1679	all reads at last member (A5)
potts.tied.last_member_fixed_select	potts_mpnn_utils.py:1681	cm[t_last] select, written to all (A5)
potts.rank.pre_refine_energy_sort	sample_seqs.py:268	sorted by AR energy (A5)
potts.rank.n1_no_suffix	sample_seqs.py:271	no _sidx when N==1 (A5)
potts.rank.pdb_refined_of_best	sample_seqs.py:368-373	PDB = refined seq of rank-0 sample (A5)
# PottsMPNN refine (A5)
potts.refine.mode_potts_single	run_utils.py:113-118	one sweep (A5)
potts.refine.mode_converge_1000	run_utils.py:116-120	while (delta!=0 and iters<1000) (A5)
potts.refine.skip_masked_or_fixed	run_utils.py:123-124	skip mask==0 or chain_mask==0 (A5)
potts.refine.binding_only_skip_noninterface	run_utils.py:135-136	only: skip non-interface (A5)
potts.refine.binding_unbound_delta	run_utils.py:139-148	(E_c-E_c[cur])-(E_p-E_p[cur]) at interface (A5)
potts.refine.binding_both_noninterface_abs	run_utils.py:138	both: non-interface uses absolute E (A5)
potts.refine.logits_first20_only	run_utils.py:151-160	gap/X never chosen; pad probs (A5)
potts.refine.pssm_omit_same_block	run_utils.py:161-173	PSSMMix in refine (A5)
potts.refine.draw_not_argmin	run_utils.py:174	multinomial draw (A5)
potts.refine.ener_delta_accumulate	run_utils.py:176	+= predicted_E[chosen] (A5)
potts.refine.order_default_arange	run_utils.py:110-111	decoding_order None -> arange (A5)
potts.refine.nodes_init	run_utils.py:181-191	Gibbs init, mask_bw self-slot only (A5)
potts.refine.nodes_skip	run_utils.py:195-200	skip mask==0/chain_mask==0 (A5)
potts.refine.nodes_hS_unmasked	run_utils.py:203	h_S_masked plain clone (A5)
potts.refine.nodes_draw	run_utils.py:263	multinomial (A5)
potts.refine.order_lookup_str_vs_suffix	sample_seqs.py:325-330	N>1 unsuffixed -> None -> arange (A5)
potts.refine.order_fresh_key	sample_seqs.py:331-343	fresh (chain_mask+1e-4)|randn| no cmp/present (A5)
potts.refine.order_fix_seed	sample_seqs.py:332-337	fix_decoding_order seed (A5)
potts.refine.order_stored_reuse_n1	sample_seqs.py:327-328	N==1 unsuffixed reuses AR order (A5)
potts.refine.order_suffix_dict_reset	sample_seqs.py:274-279	suffixed entries: dict reset per rank iteration (A5)
potts.tied_refine.masked_member_propagate	run_utils.py:331-340	first masked member's residue to all (A5)
potts.tied_refine.epistasis_joint	run_utils.py:344-373	joint 20-way mutation, energy at leaked last pos (A5)
potts.tied_refine.per_member	run_utils.py:374-401	per-member energies averaged (A5)
potts.tied_refine.zero_members_div	run_utils.py:401	num_pos==0 -> ZeroDivisionError (upstream crash; divergence) (A5)
potts.tied_refine.binding_no_reference	run_utils.py:371,398	E_b-E_u without current-identity subtraction (A5)
potts.tied_refine.apply_epistasis	run_utils.py:422-423	seq = sort_seqs[mut] (A5)
potts.tied_refine.apply_per_member	run_utils.py:424-427	mut_res at leaked pos to all members (A5)
potts.tied_refine.nodes_group_order	run_utils.py:439-447	groups from order (A5)
potts.tied_refine.nodes_epistasis_group_mask	run_utils.py:453-463	same-group neighbours forward-masked (A5)
potts.tied_refine.nodes_masked_member	run_utils.py:473-479	masked member copies S (A5)
potts.tied_refine.nodes_last_member_select	run_utils.py:512-516	chain_mask[t_last] hard pick (A5)
potts.skip_calc.optimize_fasta	sample_seqs.py:133-141	loads out_dir/out_name.fasta (divergence) (A5)
potts.skip_calc.optimize_pdb	sample_seqs.py:142-157	native sequences, listing-order concat (divergence) (A5)
potts.skip_calc.key_prefix_match	sample_seqs.py:319	k.startswith(pdb_with_chain_suffix) (A5)
potts.skip_calc.colon_strip	sample_seqs.py:322	ints from ':'-stripped (A5)
potts.skip_calc.stored_orders_file	sample_seqs.py:178-180	implicit decoding_order.json (divergence) (A5)
# LASEr featurizer (B0)  -- run_inference.py / pdb_dataset.py @ e70f2c6d
laser.feat.water_ignored	run_inference.py:187-194	HOH ignored unless use_water (B0)
laser.feat.ncaa_as_ligand	run_inference.py:201-215	noncanonical_aa_ligand -> ligand atoms (B0)
laser.feat.missing_backbone_dropped	run_inference.py:203-204	residue missing N/CA/C dropped (B0)
laser.feat.fix_beta_chain_mask	run_inference.py:213,331	chain_mask=isclose(max(bfac),1) if fix_beta else all-0 (B0)
laser.feat.extra_atom_contact_zero	run_inference.py:350,371	extra_atom_contact_mask all-False (B0; see C1)
laser.feat.first_shell_overwritten	pdb_dataset.py:528-555,590	post-graph heavy-atom mask consumed (B0; see C2)
laser.graph.no_ligand_pr_only	pdb_dataset.py:449-454	ligand-free complex: pr-pr graph only, first-shell all-False (B0)
laser.graph.knn_self_loop	pdb_dataset.py:450,518	knn_graph loop=True (B0)
laser.graph.lig_prot_cutoff_rows	pdb_dataset.py:817-827	connected rows only, K=min(48,A) unbounded by cutoff (B0)
laser.graph.aliphatic_h_unreachable	pdb_dataset.py:805-814	use_aliphatic_ligand_hydrogens default True (B0; assert unreachable)
laser.graph.bb_noise_round_translate	pdb_dataset.py:413-415	round 0.01 + per-residue translation (B0/B5)
laser.ignore_ligand	run_inference.py:721-726	empty ligand tensors (B0)
laser.repack_only_chain_mask	run_inference.py:729-733	chain_mask=1 (B0)
# LASEr decode (B4/B5)
laser.sample.first_shell_temp_vector	run_inference.py:541-544	fs_sequence_temp: T None->1e-6 elsewhere (B5)
laser.sample.charged_mask	model.py:843-848	K/R/D/E -inf on first-shell rows (B5)
laser.sample.disabled_mask_sampled_rows	model.py:835-838	finfo.min on ~chain_mask rows (B5)
laser.sample.disabled_mask_ignore_zeros	model.py:836	inverted under ignore_chain_mask_zeros (B5)
laser.sample.budget_ala_gly	model.py:782-788,840-841	counts incl fixed; finfo.min (B5)
laser.sample.argmax_vs_sampled	model.py:851-859	T None -> argmax (B5)
laser.sample.minp_warp	model.py:854,48-79	min-p on untempered logits (B5)
laser.sample.stored_logits_post_minp	model.py:854,881	stored seq_logits are post-min-p (B5; see C3)
laser.sample.fixed_select	model.py:861-865	chain_mask picks input label (B5)
laser.sample.ignore_zeros_noop	model.py:793-796	chain_mask==0 step skipped (B5)
laser.sample.ignore_zeros_final_x	model.py:936-937	unsampled rows -> X (B5)
laser.chi.argmax_vs_sampled	model.py:894-899	chi_T None -> argmax (B5)
laser.chi.minp_post_warp_offset	model.py:897-902	offset head sees post-warp logits (B5)
laser.chi.input_chi_retained	model.py:907-914	fixed & non-NaN input chi kept unless repack_all/ignore_zeros (B5)
laser.chi.nan_input_sampled	model.py:908-914	NaN input chi sampled (B5)
laser.chi.mask_write_x_to_g	model.py:870-878,922-925	chi written only where chi_mask(aa X->G) (B5)
laser.chi.gvp_update_k012	model.py:927-934	no update after k=3 (B5)
laser.tied.lambda_mix	model.py:626-627	lambda*P1+(1-lambda)*P2 (B7)
laser.tied.disabled_fixed_only	model.py:605-609	~chain_mask rows only (B7)
laser.tied.temp_none_1e6	model.py:617-618	T None -> 1e-6 sampled (B7)
laser.tied.chi_argmax_vs_sampled	model.py:669-677	chi T None argmax; else 2 draws (B7)
laser.tied.chi_input_all_fixed	model.py:690-692	chain_mask rows keep input chi (nan->0) unless repack_all (B7)
laser.tied.fs_temp_nameerror	model.py:484-485	tensor T -> NameError (divergence ValueError) (B7)
laser.tied.budget_struct1_counts	model.py:527-528	counts from structure 1 (B7)
laser.score.unconditional_edge_mask	model.py:370-372	return_unconditional_probabilities: no visible predecessors (B7)
laser.score.teacher_forced_chi_mask	model.py:321-333	chi logits only where ~isnan(chi) (B4)
laser.score.bin_argmax_nan_zero	model.py:383-385	nan chi -> zero RBF -> bin 0 (masked) (B4)
laser.proofread.dropout_train_on_nn_dropout	run_proofreading.py:21-30	only nn.Dropout .train() (B7)
laser.proofread.selection_string_override	run_proofreading.py:69-79	selection replaces first shell; None->ValueError (B7)
laser.proofread.repack_all	run_proofreading.py:130	repack_all passthrough (B7)
laser.proofread.n_orders_one_std_nan	run_proofreading.py:133	std of 1 sample NaN (B7)
laser.proofread.reduction_mean_plus_std	run_proofreading.py:138-141	mean_d m + mean_d s (B7)
```

## 8. Pre-existing aminx debt to file (spec 6.5b last row, "backlog id filed in T0.2")

Not filed by this task: no tracker write was authorised beyond the single report file. Draft entry for the orchestrator to file (praxia `debt`, task `260929_potts-laser-xtrax-compose`):

> **aminx stock MPNN decoder passes invalid-neighbour messages when `L_total < 48 < L_pad`.** `model/features.py:183-184` uses `k = min(k_neighbors, L_pad)`;
> for present rows with fewer than `k` present neighbours the remaining slots are `inf`-distance pad rows (lower index first). `model/decoder.py:144-147` builds `mask_bw/mask_fw` from `mask[:,None]` and the AR
> ordering only, never from neighbour validity, and `DecoderLayer` (`decoder.py:281-357`) only masks messages when an `attention_mask` is supplied by the caller (the stock path does not). Pad-row messages therefore
> enter the aggregation (`message_mlp([h_i, 0]) != 0`). Upstream ProteinMPNN/PottsMPNN never see pad rows because they featurize unpadded. Impact: padded runs differ from unpadded; parity with upstream is guaranteed only for `max_length == L_total` gap-free runs.

## 9. Open-item disposition (spec section 10 and every "T0.2 ..." phrase)

| Spec locus | Item | Status | Section |
|---|---|---|---|
| 2.1 (line 233) | every `nn.Dropout` path and `self.training` read | done | 6.10 |
| 2.1 (line 237) | helper names/signatures, `sample()` metadata keys | done | 1 |
| 2.2 (line 245, 248) | `model_family` re-grep, fallback consumers | done | 2 |
| 4.1 (line 340) | strict=False missing/unexpected per checkpoint | done, B1 flagged | 3 |
| 4.1 (line 345) | ft config | done | 3.1 |
| 4.1 (line 347) | `proteinmpnn_compatible` equivalence | done | 3.2 |
| 4.3 (line 461) | float32 grep in `src/aminx/model/**` | done | 5.1 |
| 4.3 (line 510) | PSSMMix transcription (`:1391-1404` + decoder equivalents) | done | 4.1, 4.2 |
| 4.3 (line 531) | tied refine (`tied_optimize_sequence`, `tied_epistasis`) | done | 4.4 |
| 5.1 (line 579) | `model_params` dims | done | 6.1 |
| 5.1a (line 584) | `LASER_ALPHABET` exact string | done, matches | 6.2 |
| 5.2 (line 595) | ligand-encoder keys in main checkpoint | done, yes | 6.1 |
| 5.2 (line 608) | every `radius_graph`/`scatter_*` site + dense replacement | done | 6.6 |
| 5.4 (line 678) | `tied_sample` ignored-knob list | done, confirmed | 6.8 |
| 5.4 (line 681) | resindex <-> row map | done, identity not guaranteed | 6.12 |
| 5.4a (line 701) | transcribe `bin` | done | 6.4 |
| 5.5 (line 717) | `num_adjacent_residues_to_drop` inert | confirmed statically | 6.9 |
| 6.4 (line 792) | LigandMPNN scripts drive LigandMPNN | confirmed | 6.11 |
| 6.5b (line 821) | backlog id for aminx decoder invalid-neighbour behaviour | draft written, **not filed** | 8 |
| 0 (line 117) | `conditional_ids.txt` | content provided (file not created) | 7 |
| 7.1 (line 947, 955) | χ draw sites, order-draw line | done | 6.5 |
| 9 (line 1140) | key audit blocks | B1 (excluded files only) | 3 |
| 10 | T floor value | 1e-6, exact-zero only | 3.3 |
| 10 | verbatim `decoder`/`tied_decoder` PSSM, `optimize_sequence`, `nodes`, refine order keying | done | 4 |
| task brief | aminx `neighbor_indices[...,0]==self` behaviour | done | 5.2 |

## 10. Spec corrections

Each entry: the spec claim, the evidence, and the suggested amendment.

**C1. Order tiers / `extra_atom_contact_mask` (spec 5.4a, 8 stage table "Order generation", 7.3 order vehicle).** Claim: `contact = upstream extra_atom_contact_mask (utils/pdb_dataset.py:1233-1239)`, "the same rule drives `sample` and proofread orders",
vehicle fixture with "fixed/non-contact/contact rows". Evidence: `:1233-1239` is the *training-dataset* definition (heavy atoms of `all_extra_coords_data`). The inference featurizer sets the mask to all-False (`run_inference.py:350,371`); exploratory
`extra_atom_contact_mask.sum() == 0` on `4jnj`. Amendment: for `sample` and proofread, tier 2 is empty (orders = fixed then all designable); for `score:nll|logits` (no upstream caller) the aminx-defined contact mask is a free choice and must be
named separately from the `sample` rule; the order vehicle must inject `extra_atom_contact_mask` on the oracle batch by hand to exercise tier 2.

**C2. First-shell knobs are inert (spec 6.1 `fs_calc_ca_distance`, `fs_calc_burial_hull_alpha_value`, `fs_no_calc_burial`; 5.5 "alpha-hull first shell"; 6.3 alias rows).** Evidence: `pdb_dataset.py:590` overwrites the mask computed with those knobs
(section 6.7); exploratory 3-setting check gives an identical post-graph mask. Amendment: B0 need not implement the alpha-hull burial or the 10 A CA test for the consumed mask; the three knobs are `no_op`-style rows (inverse differential:
set == unset), not live knobs; the first-shell rule to port is the heavy-atom contact rule (`pdb_dataset.py:756-786,528-555`, 5.0 A threshold, +0.3 A for Gly/X CA, restricted to residues in the ligand-protein edge set).

**C3. Stored `seq_logits` (spec 5.3 steps 1b/5).** Claim: stored logits = "the step-5 input" (pre-min-p) and "with zero bias both equal upstream `utils/model.py:881`". Evidence: `:854` reassigns `curr_out_logits = minp_warp_logits(...)` (only
when `T` is set) and `:881` stores the reassigned tensor, so stored logits are **post-min-p** (`-inf` for removed tokens when `seq_min_p > 0`), and `seq_log_prob`
(`run_inference.py:744`, `softmax(stored)[sampled]`) is renormalised over the kept tokens. Equality holds only for `seq_min_p == 0` or `T is None`. Amendment: either store post-warp logits (bit-parity, matches the spec's own χ statement) or add a divergence row; add a min-p>0 stored-logits check.

**C4. `noise` alias (spec 6.3: "LASEr `bb_noise` -> `[noise]`" alongside "Potts `noise` (iid Gaussian N/CA/C/O)").** Evidence: Potts noise is `X + augment_eps * randn_like(X)` per atom (`potts_mpnn_utils.py:1170-1171`, applied whenever `augment_eps > 0`,
i.e. in eval too). LASEr noise is `round(coords, 2 decimals) + noise * randn((N,1,3))`: one translation per residue shared by the 5 backbone atoms, only when `noise > 0` (`pdb_dataset.py:412-415`). The knob test "iid Gaussian N/CA/C/O" is Potts-only; the
LASEr driver needs its own noise stage and a `knob_semantics_laser_noise` test (per-residue rigid translation, rounding).

**C5. LASEr temperature 0 (spec 2.3 U-c: `None` = argmax; 5.3 step 5).** Evidence: `run_batch_inference.py:245` `float(x) if x else None` maps 0.0 to argmax, but single-input `run_inference.py:777,796` (`--temp` is a *string*; `'0'` is truthy) passes `0.0` to
`softmax(logits / 0.0)` -> NaN. The spec is silent on `temperature == 0` for lasermpnn. Amendment: state the mapping (recommend 0 -> argmax, matching the batch CLI; PottsMPNN's 0 -> 1e-6 floor is a different rule) and add it to `laser_*__sequence_temp` alias notes.

**C6. `disabled_residues` default (spec 6.1 `disabled_residues=("X","C")`).** Provenance: batch CLI default `'X,C'` (`run_batch_inference.py:380`) and tied (`run_inference_tied.py:568,884`); single-input CLI and `sample()` default `['X']`
(`run_inference.py:514`, `utils/model.py:731`). Amendment: record the default as the batch-CLI value and note the single-input divergence; the alias rows for `run_inference__*` need `disabled_residues` marked "no such flag; fixed `['X']`".

**C7. Anchor error (spec 5.2 table row 1).** `LigandFeaturizer, LigandEncoderModule (model.py:1150,1312)`: `LigandEncoderModule` is `model.py:1146` (init `:1150`); `model.py:1312` is `SpiceDatasetPretrainingModule.__init__`
(ligand-encoder pretraining wrapper, not used at inference); `LigandFeaturizer` is `utils/ligand_featurization.py:7` (its buffers are the 6 `ligand_featurizer.*` keys).

**C8. Dead knobs on the tied CLI (spec 6.3 alias table is silent).** `run_inference_tied__disable_charged_fs`, `__disabled_residues`, `__entropy_decoder` are parsed but have no effect (section 6.8; `run_inference_tied.py:620,618`). They must be `no_op` rows (or
`exclusion: no_op`), not mapped onto live Options (which would make `tied_second_input` + `disable_charged_fs` a ValueError while upstream silently accepts it; the spec's ValueError policy is fine, but the alias row must say so).

**C9. Tied-refine quirks missing from the 6.5 quirk table** (spec 4.3 says only "transcribed in T0.2"): (a) `tied_epistasis` energy at leaked last `pos`; (b) tied binding lacks the current-identity reference subtraction (`run_utils.py:371,398` vs `:148`) so the spec's
"binding sums ΔE" holds only untied and tied `potts_converge` binding runs to the 1000-sweep cap like the non-binding case; (c) `num_pos == 0` `ZeroDivisionError` (`:401`); (d) tied `nodes` lacks the per-member `chain_mask` skip; (e) tied `nodes` masked member copies `S`, not `S_true`.
Amendment: add rows to 6.5/6.5b (ZeroDivisionError as a divergence with a ValueError) and branch ids (already in section 7).

**C10. Proofread `resindex` identity (spec 5.4 "T0.2 verifies resindex <-> B0 row map").** Not guaranteed; upstream assumes identity (`run_proofreading.py:120,69-79`); holds on `4jnj-1_prot.pdb`; breaks when any ligand/water/non-amino/dropped residue precedes the focus
residue in file order. B0 must carry `row_to_resindex`; decide divergence vs parity (section 6.12).

**C11. `make_axis_dispatch_via_xtrax` docstring says "Not yet wired into any call site"** (`tiling/dispatch.py:161-200`): stale; ~15 call sites (section 1). No spec change; note for T0.5 authors so they do not conclude the function is unused.

**C12. Minor anchor / wording notes (no behaviour change).** (i) spec 4 row 153 `tied_decoder :1599- (body as :1546-1595 ...)`: the CLI body is `:1636-1685`; `:1546-1595` is the `tied_sample` copy. (ii) spec 4.1a "`chain_M_pos` ... `:415`": `:415` is chain-local
(`fixed_position_mask[np.array(fixed_pos_list)-1] = 0`); global row = concatenation offset, as stated. (iii) spec 2.3 "`_resolve_family_defaults` after the model_family block and before `_sync_run_spec()` (currently `specs.py:519-520`)": confirmed, `:520` is the base `_sync_run_spec()`;
note `SamplingSpecification.__post_init__` runs the float->tuple coercion (`:633-634`) *after* `super().__post_init__()`, so the hook must own that coercion (as the spec says). (iv) spec 4.3 row 432 "`:41`" for mode-none: `:41` is the `"none" -> ""` normalisation; `N` is forced by `:40`. (v) spec 1 row 154 `get_etab :816-835`: function starts at `run_utils.py:791`; `:816-835` is its body. (vi) LASEr `run_inference.py` `sample_model` default `chi_temp=None` and CLI has no chi flag, so single-input χ is always argmax.

## 11. Commands run (reproducibility; exploratory, not sidecar-tracked)

Torch probes (titanix, unsandboxed ssh, `cd ~/projects/aminx-oracles && ~/.local/bin/uv run --no-sync python - <<"PY" ... PY`, `torch 2.4.1+cpu`, CPU, `torch.manual_seed(0)`):
(1) load every `*_model_weights/*.pt`, build upstream `PottsMPNN` as in section 3, `load_state_dict(strict=False)`, print missing/unexpected, top-level keys, SHA-256; (2) tensor-by-tensor `torch.equal` of the six compat files vs vanilla/soluble, and `potts_ft` vs `vanilla_20`;
(3) `sha256sum` of all 16 PottsMPNN + 4 LASEr weights vs `VENDOR_PIN.toml`; (4) LASEr: `torch.load` each file, print `params['model_params']`, `ligand_encoder_params`, key-prefix counts, `strict=False` missing/unexpected and `load_model_from_parameter_dict(strict=True)`, pretrained-encoder tensor equality;
(5) `LASER_ALPHABET` from `aa_idx_to_short`; module enumeration for dropout (section 6.10); (6) `4jnj-1_prot.pdb`: `ProteinComplexData` rows vs ProDy `resindex`, `extra_atom_contact_mask`, first-shell mask before/after `construct_graphs` for three knob settings, graph degree facts.
Static reads: `rg`/`sed` over the pinned trees and the worktree. No file other than this report was written; nothing was committed. The three exploratory numbers labelled "single fixture" (first-shell mask sizes, connected-residue counts, `resindex` identity) must be re-derived under a
sidecar (B0 gate) before they are cited outside this report.
