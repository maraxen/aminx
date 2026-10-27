---
title: Scope and Specification for Supporting AbMPNN, ThermoMPNN, and Other MPNN Variant Weights
task_id: 261026_mpnn-variant-weights-scope-spec
date: 261026
status: draft
brainstorm_session: true
invest_overrides: ["Small"]
---

# Scope & Specification: Supporting AbMPNN, ThermoMPNN, and Other MPNN Variant Weights in `aminx`

## 0. Executive Summary & Difficulty Ranking (Easiest → Hardest)

The broader ProteinMPNN ecosystem consists of five distinct structural classes of "variant weights," ranging from pure fine-tuned state dicts that drop straight into [`Aminx`](file:///home/marielle/projects/aminx/src/aminx/model/mpnn.py#L27-L172) to transfer-learning heads, coarse-grained featurizers, and non-MPNN backbones that carry the "MPNN" or "ProteinDPO" name in literature.

| Rank | Tier | Variant / Model | Upstream Source & Weights | Relationship to `aminx` | Effort | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 (Easiest)** | **Tier 1A: Zero-Code Drop-In Fine-Tunes** | **AbMPNN**<br>**HyperMPNN**<br>**CyclicMPNN** | • **AbMPNN**: Zenodo `8164693` (`abmpnn.pt`)<br>• **HyperMPNN**: `meilerlab/HyperMPNN` (`retrained_models/*.pt`)<br>• **CyclicMPNN**: `ParisaH-Lab/CyclicMPNN` (`cyclicmpnn_48_010.pt`) | **100% identical architecture** to [`Aminx`](file:///home/marielle/projects/aminx/src/aminx/model/mpnn.py#L27-L172) (`node=128, edge=128, enc=3, dec=3, k=48, vocab=21`). Zero kernel changes; requires weight conversion via [`scripts/convert_weights.py`](file:///home/marielle/projects/aminx/scripts/convert_weights.py) + alias/topology registration in [`weights.py`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py) and [`versions.py`](file:///home/marielle/projects/aminx/src/aminx/model/versions.py). | **~0.5–1 day** | **PHASE 1 (Immediate)** |
| **2** | **Tier 1B: Drop-In Weights + Host/Spec Policy** | **AntiBMPNN** | • `zeysun/AntiBMPNN` | Uses vanilla ProteinMPNN backbone ($0.0\text{ \AA}$ noise) paired with CDR-specific frequency scoring / sampling masks. Core weights are Tier 1A; CDR frequency reweighting maps to a [`StageSet`](file:///home/marielle/projects/aminx/src/aminx/types/stages.py) `logit_transform`. | **~1–2 days** | **PHASE 1B (Optional StageSet helper)** |
| **3** | **Tier 2: Alternate Featurizer + Same Trunk** | **CA-Only ProteinMPNN** (`ca_model_weights`) | • `dauparas/ProteinMPNN` (`ca_model_weights/v_48_{002,010,020,030}.pt`) | Encoder (`EncLayer` ×3) and Decoder (`DecLayer` ×3) are identical to [`Aminx`](file:///home/marielle/projects/aminx/src/aminx/model/mpnn.py#L27-L172), **but** [`ProteinFeatures`](file:///home/marielle/projects/aminx/src/aminx/model/features.py#L246-L455) (`edge_in = 416`, 25-pair full-backbone RBFs) is replaced by `CA_ProteinFeatures` (`edge_in = num_pos + 9*16 + 7 = 167`, virtual $\text{C}_\alpha$ frames, coarse orientations, quaternions, 9 shifted $\text{C}_\alpha$ RBFs). | **~2–3 days** | **PHASE 2** |
| **4** | **Tier 3A: Frozen Trunk + Transfer Head (Single Mutant)** | **ThermoMPNN** (`TransferModel`) | • `Kuhlman-Lab/ThermoMPNN` (`models/thermoMPNN_default.pt`) | Runs a frozen ProteinMPNN trunk (`v_48_020`), extracts intermediate decoder hidden states (`all_mpnn_hid[:2]`, $2 \times 128$) + sequence/node embeddings (`mpnn_embed`, $128$) $\rightarrow 384$-d vector $\rightarrow$ `LightAttention` (parallel 1D convs + softmax attention) $\rightarrow$ `both_out` / `ddg_out` MLP ($[64, 32] \rightarrow 21$) predicting per-mutation $\Delta\Delta G^\circ$. | **~3–5 days** | **PHASE 3A** |
| **5** | **Tier 3B: Frozen Trunk + Siamese Epistatic Head** | **ThermoMPNN-D** (`TransferModelPLv2Siamese`) | • `Kuhlman-Lab/ThermoMPNN-D` | Extends ThermoMPNN with a **Siamese network** over pairs of mutated positions $(i, j)$ to guarantee order-invariance ($\Delta\Delta G(A,B) == \Delta\Delta G(B,A)$) and capture epistatic double-mutant couplings. | **~2–3 days** (after 3A) | **PHASE 3B (`FamilyDriver`)** |
| **6** | **Tier 3C / 4A: Ensemble-Conditioned Potts / MPNN** | **Caliby** (`ProteinDesignLab/caliby`) | • `ProteinDesignLab/caliby` (Po-Ssu Huang Lab, Stanford; Zenodo checkpoints) | **Ensemble-conditioned Potts / inverse-folding model** trained on structural conformational ensembles (MD / AF2 ensembles) rather than single static structures. (**Note**: Corrects backlog `#5595` misattribution in [`.praxia/docs/research/260605_caliby-nature-and-dataset.md`](file:///home/marielle/projects/aminx/.praxia/docs/research/260605_caliby-nature-and-dataset.md), which confused Caliby with a TRW marginal calibration sidecar.) Requires multi-conformation ensemble featurization $(N_{\text{conf}}, L, \dots)$ + Potts/MPNN trunk. | **~1–2 weeks** | **DEBT / SEPARATE EPIC (`FamilyDriver`)** |
| **7** | **Tier 4B: Joint Sequence + Continuous Sidechain Diffusion** | **FAMPNN** (Full-Atom MPNN) | • `richardshuai/fampnn` (`weights/fampnn_0_0.pt`, `fampnn_0_3.pt`) | Substantially modifies both featurization and decoding to jointly model categorical sequence identity (cross-entropy) and continuous sidechain atom coordinates (diffusion loss), superseding the 4-torsion mixture [`Packer`](file:///home/marielle/projects/aminx/src/aminx/model/packer.py). | **~2–3 weeks** | **DEBT / SEPARATE EPIC** |
| **8 (Hardest / Out-of-Family)** | **Tier 5: Non-MPNN Backbones** | **ProteinDPO**<br>**Frame2seq**<br>**CarbonDesign**<br>**ProstT5** | • `evo-design/protein-dpo` (ESM-IF1 GVP-GNN, 142M)<br>• `dakpinaroglu/Frame2seq` (IPA masked LM)<br>• `Rostlab/ProstT5` (3Di T5) | **Not ProteinMPNN architectures.** ProteinDPO fine-tunes ESM-IF1 (`esm_if1_gvp4_t16_142M_UR50.pt`); Frame2seq uses Invariant Point Attention. Zero module or weight reuse with `aminx.model`. | **N/A (New repo/family)** | **REJECT / OUT OF SCOPE** |

---

## 1. Deep-Dive Architectural Assessment by Tier

### 1.1 Tier 1: Drop-In Fine-Tuned Weights (AbMPNN, HyperMPNN, CyclicMPNN)

#### Why They Are Easiest
All three models were trained by taking the upstream `dauparas/ProteinMPNN` training loop (`vanilla_model_weights`, $K=48$, `node_features=128`, `edge_features=128`, `hidden_dim=128`, 3 encoder layers, 3 decoder layers, `vocab=21`) and fine-tuning or retraining on domain-specific datasets:
1. **AbMPNN** (*Dreyer et al., 2023*, Zenodo `8164693`, `abmpnn.pt`): Fine-tuned sequentially on OAS and SAbDab antibody/nanobody structures to improve CDR (especially CDR-H3) sequence recovery and structural consistency.
2. **HyperMPNN** (*Meiler Lab*, `meilerlab/HyperMPNN`, `retrained_models/*.pt`): Retrained on AlphaFold-predicted structures from hyperthermophilic organisms ($T_{\text{opt}}$ up to $105^\circ\text{C}$) across multiple backbone noise levels ($0.02, 0.10, 0.20, 0.30\text{ \AA}$) to bias sequence design toward thermostability.
3. **CyclicMPNN** (*Hosseinzadeh Lab*, `ParisaH-Lab/CyclicMPNN`, `cyclicmpnn_48_010.pt`): Fine-tuned on PDB + GenKIC *in silico* cyclic peptide backbones ($K=48$, $0.10\text{ \AA}$ noise). Note: for head-to-tail cyclic peptides, callers either supply contiguous `residue_index` or cyclic offset handling; the `.pt` state dict itself has identical layer shapes to `proteinmpnn_v_48_010.pt`.

#### Current Friction in `aminx`
Even though [`Aminx`](file:///home/marielle/projects/aminx/src/aminx/model/mpnn.py#L27-L172) and [`scripts/convert_weights.py`](file:///home/marielle/projects/aminx/scripts/convert_weights.py) already support the exact tensor layout of these checkpoints, four host-layer bottlenecks currently block first-class usage:
1. **Literal Type Restrictions**: [`MODEL_WEIGHTS`](file:///home/marielle/projects/aminx/src/aminx/model/versions.py#L7) in [`src/aminx/model/versions.py`](file:///home/marielle/projects/aminx/src/aminx/model/versions.py) is restricted to `Literal["original", "soluble", "ligand", "sc", "membrane"]`.
2. **Filename & Alias Resolution**: [`load_model`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L361-L471) in [`src/aminx/io/weights.py`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py) checks `if checkpoint_id and ("_v_" in checkpoint_id or "mpnn" in checkpoint_id)` and constructs `{model_weights}_{model_version}.eqx.zst` via [`LEGACY_ALIAS_MAP`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L57-L66). A checkpoint named `abmpnn_v_48_020.eqx.zst` or `abmpnn.eqx.zst` or `cyclicmpnn_v_48_010.eqx.zst` needs explicit entries in [`LEGACY_ALIAS_MAP`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L57-L66) and safe parsing in [`get_topology_for_checkpoint`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L69-L110).
3. **Topology Parser Suffix Bug (Defect `checkpoint_topology_source`)**: As documented in [`.praxia/docs/audits/260926_mpnn-reference-parity-verdict.md`](file:///home/marielle/projects/aminx/.praxia/docs/audits/260926_mpnn-reference-parity-verdict.md#L80), [`get_topology_for_checkpoint`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L100-L108) splits on `_` and parses any trailing integer not in `[32, 48, 30]` as `atom_context_num` (so `_020` becomes `atom_context_num = 20`!). While `Aminx` (protein branch) ignores `atom_context_num`, fixing [`get_topology_for_checkpoint`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L69-L110) to only parse `atom_context_num` on `ligand`/`packer` checkpoints prevents metadata corruption across all new variants.
4. **Positional Embedding Auto-Detection**: [`convert_weights.py`](file:///home/marielle/projects/aminx/scripts/convert_weights.py#L819-L827) dynamically detects `num_pos = (pos_weight.shape[1] - 2) // 2` (`16` vs `32`), whereas [`get_topology_for_checkpoint`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L73) hardcodes `num_positional_embeddings = 32` for `v_48` and `16` otherwise. Vanilla `dauparas/ProteinMPNN` checkpoints use `num_positional_embeddings = 16` (`pos_weight.shape == (16, 66)` where `MAX_REL = 32`, i.e., `2*16` in `ProteinFeatures` where `num_positional_embeddings` is `MAX_REL=32`). Any converted variant checkpoint must match [`get_topology_for_checkpoint`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L69-L110)'s `num_positional_embeddings` contract.

---

### 1.2 Tier 2: CA-Only ProteinMPNN (`ca_model_weights`)

#### Architectural Delta vs. [`ProteinFeatures`](file:///home/marielle/projects/aminx/src/aminx/model/features.py#L246-L455)
Upstream `dauparas/ProteinMPNN` ships four `ca_model_weights` checkpoints (`v_48_002.pt`, `v_48_010.pt`, `v_48_020.pt`, `v_48_030.pt`).
- **Trunk**: Identical `W_e`, `W_s`, `encoder_layers` (3), `decoder_layers` (3), `W_out` as [`Aminx`](file:///home/marielle/projects/aminx/src/aminx/model/mpnn.py#L27-L172).
- **Featurizer (`CA_ProteinFeatures`)**:
  - Takes only $\text{C}_\alpha$ coordinates `(L, 3)` (or slices index 1 from `(L, 4, 3)` backbone coordinates).
  - `_orientations_coarse(Ca, E_idx)`: Computes sequential $\Delta X$ vectors, masks chain breaks outside $3.6\text{ \AA} < \|\Delta X\| < 4.0\text{ \AA}$, derives virtual backbone normals/angles (`AD_features`, dim 6 — though `node_embedding` is unused in upstream forward), constructs local $3 \times 3$ coordinate frames per residue, and computes relative rotation quaternions `_quaternions(R)` + local direction features (`O_features`, dim 7).
  - Computes 9 shifted $\text{C}_\alpha$ RBF distance matrices (`Ca_0`, `Ca_1`, `Ca_2` representing $i-1, i, i+1$ shifted coordinates: $9 \times 16 = 144$ dims).
  - Concatenates `[E_positional (16), RBF_all (144), O_features (7)]` $\rightarrow$ `edge_in = 167` (vs. `416` in full-backbone [`ProteinFeatures`](file:///home/marielle/projects/aminx/src/aminx/model/features.py#L282)).
  - Projects via `edge_embedding` (`Linear(167, 128, bias=False)`) $\rightarrow$ `norm_edges` (`LayerNorm(128)`).

#### Why Tier 2 is Medium-Low Effort
- No new inference topology or `StageSet` changes: once `ProteinFeaturesCA` emits `(edge_features, neighbor_indices, None, prng_key)`, the entire [`Aminx`](file:///home/marielle/projects/aminx/src/aminx/model/mpnn.py#L27-L172) encoder, decoder, [`InferencePlan`](file:///home/marielle/projects/aminx/src/aminx/host/plan.py), and ONNX/IREE browser export pipeline work unchanged (provided `ProteinFeaturesCA` uses [`aminx.model.features.top_k`](file:///home/marielle/projects/aminx/src/aminx/model/features.py#L48-L134)).

---

### 1.3 Tier 3: ThermoMPNN & ThermoMPNN-D (Transfer-Learning $\Delta\Delta G$ Heads)

#### How ThermoMPNN Works Under the Hood
ThermoMPNN (*Dieckhaus et al., PNAS 2024*) and ThermoMPNN-D (*Dieckhaus et al., 2024*) do **not** sample sequences autoregressively. Instead, they predict scalar thermodynamic stability changes ($\Delta\Delta G^\circ$ in kcal/mol) for single or double point mutations:
1. **Frozen ProteinMPNN Trunk**: Uses vanilla `v_48_020` (`Aminx` with `k_neighbors=48`).
2. **All-at-Once Representation Extraction**:
   - Runs the encoder (`h_V`, `h_E`, `E_idx`) and unconditional/conditional decoder layers on the wild-type backbone and wild-type sequence $S_{\text{WT}}$.
   - Extracts intermediate decoder hidden states `all_mpnn_hid` from the first $M$ decoder layers (default: first 2 decoder layers, $2 \times 128 = 256$ dims) and concatenates them with the node/sequence embedding `mpnn_embed` ($128$ dims) to form a per-residue environment vector $z \in \mathbb{R}^{L \times 384}$.
3. **`LightAttention` Module**:
   - Applies two parallel 1D convolutions (`feature_convolution` and `attention_convolution`, `in_channels=384, out_channels=384, kernel_size=1` or `9`) across the residue representation:
     $$\tilde{z} = \text{Dropout}(\text{Conv}_{\text{feat}}(z)) \odot \text{Softmax}(\text{Conv}_{\text{attn}}(z))$$
4. **Prediction Heads (`both_out` and `ddg_out`)**:
   - Slices the reweighted representation $\tilde{z}_i$ at the mutated position $i$, concatenates wild-type one-hot / embedding context, passes through `both_out` (MLP with `hidden_dims=[64, 32]`) and `ddg_out` (`Linear(32, 21)`), and subtracts the wild-type logit/energy from the mutant energy:
     $$\Delta\Delta G(i: \text{WT} \to \text{Mut}) = \hat{G}_{\text{Mut}} - \hat{G}_{\text{WT}}$$
5. **ThermoMPNN-D (`TransferModelPLv2Siamese`)**:
   - Evaluates residue pairs $(i, j)$ with a shared Siamese edge/pair MLP over $[\tilde{z}_i, \tilde{z}_j, e_{ij}]$ and $[\tilde{z}_j, \tilde{z}_i, e_{ji}]$, averaging or symmetrically pooling both directions so $\Delta\Delta G(i, j) \equiv \Delta\Delta G(j, i)$.

#### Architectural Fit in `aminx` (`FamilyDriver` Pattern on `origin/main`)
- On `origin/main` (`v0.3.0+`), non-vanilla MPNN families (`potts_mpnn`, `laser_mpnn`, `protonpotts_mpnn`) live under `src/aminx/families/<family>/`, implement the [`FamilyDriver`](file:///home/marielle/projects/aminx/src/aminx/host/family_driver.py) protocol, and register their converted weights in `src/aminx/io/family_weights.json` (built via `scripts/weights/build_family_weights.py`).
- **ThermoMPNN's currency is a site-saturation $\Delta\Delta G$ matrix `(L, 21)` (single-mutant) or pair tensor `(N_pairs,)` (ThermoMPNN-D)** from a single forward pass per structure.
- Therefore, `ThermoMPNN` and `ThermoMPNN-D` fit cleanly as a `thermo_mpnn` family under `src/aminx/families/thermo_mpnn/` (exposing both direct Equinox modules `ThermoMPNN` / `ThermoMPNNDouble` and a `ThermoMPNNDriver` / `build_family_weights.py` entry).

---

### 1.4 Tier 3C / 4A: Caliby (`ProteinDesignLab/caliby`, Po-Ssu Huang Lab)

#### Correcting the Earlier Misattribution (Backlog `#5595`)
- An earlier internal research note ([`.praxia/docs/research/260605_caliby-nature-and-dataset.md`](file:///home/marielle/projects/aminx/.praxia/docs/research/260605_caliby-nature-and-dataset.md)) and recapture stub ([`scripts/recapture/caliby_recapture.py`](file:///home/marielle/projects/aminx/scripts/recapture/caliby_recapture.py)) misattributed **Caliby** as an internal "learned post-hoc marginal calibration sidecar (`log_b_offset` + `temperature`) for `mistypotts` TRW".
- In reality, **Caliby** (`https://github.com/ProteinDesignLab/caliby`, Po-Ssu Huang Lab, Stanford University; *Shuai, Huang et al.*) is an external **ensemble-conditioned Potts / inverse-folding model** trained on structural conformational ensembles (e.g., AlphaFold2 / MD structural ensembles) to improve sequence design and energy landscape conditioning across backbone flexibility.
- Supporting real `ProteinDesignLab/caliby` requires:
  1. Correcting/superseding [`.praxia/docs/research/260605_caliby-nature-and-dataset.md`](file:///home/marielle/projects/aminx/.praxia/docs/research/260605_caliby-nature-and-dataset.md) and renaming/deprecating the synthetic TRW sidecar in [`scripts/recapture/caliby_recapture.py`](file:///home/marielle/projects/aminx/scripts/recapture/caliby_recapture.py).
  2. Multi-conformation ensemble featurization (ingesting $N_{\text{conf}}$ conformer backbones per target and pooling/conditioning across conformers) paired with a dedicated `src/aminx/families/caliby/` `FamilyDriver` and checkpoint converter.
  3. Because multi-conformation ensemble featurization introduces an ensemble axis $N_{\text{conf}}$ prior to graph message passing, it is tracked as a dedicated technical debt / follow-up epic alongside Tier 4 (FAMPNN).

---

## 2. Proposed Architecture & Phased Implementation Spec

### Phase 1: Tier 1 Drop-In Variant Weights & Open Registry (AbMPNN, HyperMPNN, CyclicMPNN)

#### 2.1.1 Widen Weight Families & Fix Topology Inference
1. **[`src/aminx/model/versions.py`](file:///home/marielle/projects/aminx/src/aminx/model/versions.py)**:
   - Expand `MODEL_WEIGHTS`:
     ```python
     MODEL_WEIGHTS = Literal[
       "original",
       "soluble",
       "ligand",
       "sc",
       "membrane",
       "abmpnn",
       "hypermpnn",
       "cyclicmpnn",
       "ca",
     ]
     ```
2. **[`src/aminx/io/weights.py`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py)**:
   - Expand [`LEGACY_ALIAS_MAP`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L57-L66) with canonical aliases:
     - `("abmpnn", "v_48_020"): "abmpnn_v_48_020.eqx.zst"` (plus bare `"abmpnn"` default resolution in `load_model`)
     - `("hypermpnn", "v_48_002")`, `("hypermpnn", "v_48_010")`, `("hypermpnn", "v_48_020")`, `("hypermpnn", "v_48_030")`
     - `("cyclicmpnn", "v_48_010"): "cyclicmpnn_v_48_010.eqx.zst"`
   - Fix [`get_topology_for_checkpoint`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L69-L110):
     - Only parse trailing integer tokens as `atom_context_num` when `model_type in ("ligand", "packer")`, eliminating the `_020` $\rightarrow$ `atom_context_num=20` bug (`checkpoint_topology_source`).
     - Support bare checkpoint IDs (`"abmpnn"`, `"cyclicmpnn"`) without requiring `_v_` in `load_model`'s legacy heuristic (`line 390`).

#### 2.1.2 Conversion & Verification Pipeline
- [`scripts/convert_weights.py`](file:///home/marielle/projects/aminx/scripts/convert_weights.py) already converts any vanilla-topology `.pt` checkpoint to `.eqx` and validates bias presence (`convert_linear_layer`).
- Add a manifest-driven batch converter / fetcher target in [`scripts/regenerate_packaged_eqx_zst.py`](file:///home/marielle/projects/aminx/scripts/regenerate_packaged_eqx_zst.py) and [`tests/parity/parity_assets.json`](file:///home/marielle/projects/aminx/tests/parity/parity_assets.json) for:
  - `abmpnn_v_48_020.eqx.zst` (converted from Zenodo `8164693/files/abmpnn.pt`)
  - `hypermpnn_v_48_{002,010,020,030}.eqx.zst` (converted from `meilerlab/HyperMPNN`)
  - `cyclicmpnn_v_48_010.eqx.zst` (converted from `ParisaH-Lab/CyclicMPNN`)
- Upload converted `.eqx.zst` artifacts to `maraxen/aminx` on Hugging Face Hub and bump `HF_REVISION` in [`src/aminx/io/weights.py`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L34).

---

### Phase 2: Tier 2 CA-Only ProteinMPNN (`ca_model_weights`)

1. **`ProteinFeaturesCA(eqx.Module)` in [`src/aminx/model/features.py`](file:///home/marielle/projects/aminx/src/aminx/model/features.py)**:
   - Implements coarse $\text{C}_\alpha$-only featurization matching `dauparas/ProteinMPNN`'s `CA_ProteinFeatures`:
     - Uses [`select_neighbors`](file:///home/marielle/projects/aminx/src/aminx/model/features.py#L172-L230) (backed by IREE-safe [`top_k`](file:///home/marielle/projects/aminx/src/aminx/model/features.py#L48-L134)) on $\text{C}_\alpha$ distances.
     - Computes `_orientations_coarse` (7-d orientation + quaternion features) and 9 shifted $\text{C}_\alpha$ RBF blocks ($9 \times 16 = 144$-d) + 16-d positional encoding = 167-d edge input.
     - Projects with `w_e: Linear(167, 128, use_bias=False)`, `norm_edges: LayerNorm(128)`, and `w_e_proj: Linear(128, 128, use_bias=True)`.
2. **Wiring into [`Aminx`](file:///home/marielle/projects/aminx/src/aminx/model/mpnn.py#L27-L172) & [`convert_weights.py`](file:///home/marielle/projects/aminx/scripts/convert_weights.py)**:
   - Add `ca_only: bool = False` to `Aminx.__init__`, selecting `ProteinFeaturesCA` vs `ProteinFeatures`.
   - In [`convert_weights.py`](file:///home/marielle/projects/aminx/scripts/convert_weights.py), detect `CA_ProteinFeatures` when `pt_state_dict["features.edge_embedding.weight"].shape[1] == 167`.
   - In [`get_topology_for_checkpoint`](file:///home/marielle/projects/aminx/src/aminx/io/weights.py#L69-L110), set `model_type = "ca"` when `checkpoint_id` starts with `"ca_"` or `"campnn_"`.

---

### Phase 3: Tier 3 ThermoMPNN & ThermoMPNN-D (`aminx.thermompnn`)

1. **Decoder Hidden-State Tap in [`Decoder`](file:///home/marielle/projects/aminx/src/aminx/model/decoder.py)**:
   - Add a side-effect-free helper `decode_with_intermediates(model, ...)` (or optional return on `Decoder`) that returns the per-layer hidden states `(h_V_layer_0, h_V_layer_1, ..., h_V_final)` alongside the initial sequence/node embedding, without altering existing `Decoder.__call__` signatures or JIT traces.
2. **Equinox Modules (`src/aminx/thermompnn/model.py`)**:
   - `LightAttention(eqx.Module)`:
     - `feature_convolution: eqx.nn.Conv1d`
     - `attention_convolution: eqx.nn.Conv1d`
     - `dropout: Dropout`
   - `ThermoMPNN(eqx.Module)`:
     - `mpnn: Aminx` (frozen `proteinmpnn_v_48_020` trunk)
     - `light_attention: LightAttention`
     - `both_out: eqx.nn.MLP` (`in_size -> 64 -> 32`)
     - `ddg_out: eqx.nn.Linear` (`32 -> 21` or scalar per-AA readout)
     - `__call__(coords, mask, residue_index, chain_index, wt_sequence) -> Float[Array, "L 21"]`: returns full site-saturation mutagenesis (SSM) $\Delta\Delta G^\circ$ matrix in one compiled pass.
   - `ThermoMPNNDouble(eqx.Module)` (Phase 3B):
     - Siamese pair readout over `LightAttention` features for mutation pairs `(i, mut_a, j, mut_b) -> Float[Array, "N_pairs"]`, strictly invariant to `(i, a) <-> (j, b)` swap.
3. **Weight Recapture Script (`scripts/recapture/thermompnn_to_eqx.py`)**:
   - Following [`scripts/recapture/pottsmpnn_to_eqx.py`](file:///home/marielle/projects/aminx/scripts/recapture/pottsmpnn_to_eqx.py), converts `thermoMPNN_default.pt` (and ThermoMPNN-D checkpoints) into `.eqx.zst` artifacts and verifies numerical parity against `Kuhlman-Lab/ThermoMPNN` on a reference PDB.

---

## Acceptance Criteria

- Given a PyTorch checkpoint from AbMPNN (`abmpnn.pt`), HyperMPNN (`retrained_models/*.pt`), or CyclicMPNN (`cyclicmpnn_48_010.pt`), when converted via `scripts/convert_weights.py` and loaded via `load_model`, then unconditional and conditional log-probabilities match the upstream PyTorch reference within `1e-4` nats max absolute difference.
- Given a `SamplingSpecification` or `ScoringSpecification` with `model_weights="abmpnn"`, `"hypermpnn"`, or `"cyclicmpnn"`, when `prep_protein_stream_and_model` executes, then the corresponding `.eqx.zst` checkpoint is resolved with a valid `WeightProvenance` SHA-256 record.
- Given any checkpoint ID passed to `get_topology_for_checkpoint`, when `model_type` is not `"ligand"` or `"packer"`, then `atom_context_num` is not overwritten by the noise-level suffix (e.g. `_020`).
- Given a `ca_model_weights` checkpoint (`ca_proteinmpnn_v_48_020.pt`), when converted and loaded into `Aminx(ca_only=True)`, then `ProteinFeaturesCA` produces edge features of shape `(L, K, 128)` from `(L, 3)` $\text{C}_\alpha$ coordinates and matches PyTorch `CA_ProteinFeatures` output within `1e-4` max absolute difference.
- Given a wild-type backbone and sequence loaded into `ThermoMPNN`, when site-saturation mutagenesis inference runs on a single structure, then the resulting `(L, 21)` $\Delta\Delta G$ matrix matches `Kuhlman-Lab/ThermoMPNN` (`thermoMPNN_default.pt`) within `1e-4` kcal/mol max absolute difference and `0.0` kcal/mol on wild-type self-mutations.
- Given a batch of double mutations `[(i, a, j, b)]` and their reversed order `[(j, b, i, a)]`, when scored by `ThermoMPNNDouble`, then the predicted $\Delta\Delta G$ values are bit-identical (`max_abs_diff == 0.0`).

## Decision Log

| Option | Verdict | Rationale |
| :--- | :--- | :--- |
| **Phased Tier 1 → Tier 2 → Tier 3 rollout (Drop-in fine-tunes first, then `ProteinFeaturesCA`, then `aminx.thermompnn`)** | **Selected** | Delivers immediate user value for AbMPNN/HyperMPNN/CyclicMPNN with zero kernel blast radius (~1 day), isolates `CA_ProteinFeatures` to a clean featurizer swap (~2 days), and scopes ThermoMPNN as a pure Equinox transfer module reusing the validated `Aminx` trunk. |
| **Force ThermoMPNN through `StageSet` autoregressive decoding** | **Rejected** | Currency and phase mismatch (same class as ADR [`260605_potts-parallel-not-stageset`](file:///home/marielle/projects/aminx/.praxia/docs/decisions/260605_potts-parallel-not-stageset.md)): ThermoMPNN computes one-shot $\Delta\Delta G$ over wild-type decoder hidden states from layers 0–1, not autoregressive per-token categorical logits. |
| **Include FAMPNN (Tier 4) in the same implementation wave** | **Deferred** | FAMPNN replaces categorical-only decoding with joint discrete sequence + continuous 3D sidechain diffusion; viable only as a dedicated epic after standard MPNN variant weights and ThermoMPNN land. |
| **Include ProteinDPO / Frame2seq / ProstT5 (Tier 5) as MPNN variants** | **Rejected** | Despite naming similarity in literature reviews, ProteinDPO is an ESM-IF1 (GVP-GNN, 142M params) fine-tune, Frame2seq is an IPA transformer, and ProstT5 is a T5 language model; none share layers or weights with ProteinMPNN. |

## Assumptions

| Assumption | Owner | Verification method |
| :--- | :--- | :--- |
| Upstream `abmpnn.pt`, `HyperMPNN`, and `CyclicMPNN` `.pt` files use standard `dauparas/ProteinMPNN` state_dict keys (`features.embeddings.linear.weight`, `W_e`, `encoder_layers.{0..2}`, `decoder_layers.{0..2}`, `W_out`, `W_s`). | `aminx` maintainer | Run `scripts/convert_weights.py` on each downloaded `.pt` file and assert zero unmapped/extra keys and zero bias mismatches. |
| `thermoMPNN_default.pt` embeds or pairs with standard `v_48_020.pt` trunk weights without modifying the `EncLayer`/`DecLayer` equations. | `aminx` maintainer | Inspect `thermoMPNN_default.pt` state_dict keys in `scripts/recapture/thermompnn_to_eqx.py` and run layer-by-layer parity against `Kuhlman-Lab/ThermoMPNN`. |
| Redistributing converted `.eqx.zst` checkpoints on `maraxen/aminx` HF Hub complies with each upstream license (AbMPNN: CC-BY 4.0; HyperMPNN: MIT; CyclicMPNN: MIT; ThermoMPNN: MIT). | `aminx` maintainer | Verify `LICENSE` files in each upstream repo/Zenodo record before pushing to `maraxen/aminx`. |

## TBDs

| Item | Owner | Resolution deadline |
| :--- | :--- | :--- |
| Whether cyclic sequence offset wrapping ($|i - j|$ across N- to C-terminus) should be added as an optional flag on `compute_neighbor_offsets` for CyclicMPNN, or handled via caller `residue_index`. | `aminx` maintainer | Before Phase 1 PR merge |
| Whether `ThermoMPNN` single-mutant SSM should be exposed via `aminx run thermompnn` CLI subcommand or integrated as a mode under `aminx run score`. | `aminx` maintainer | Before Phase 3A kickoff |

## Pre-mortem Record

- **Failure scenario (user / domain)**: A user loads `abmpnn` or `cyclicmpnn` weights expecting upstream recovery performance, but `get_topology_for_checkpoint` initializes `num_positional_embeddings=32` while the checkpoint was trained with `num_positional_embeddings=16` (or vice versa), causing silent tensor shape errors at deserialization or subtle positional encoding corruption.
- **Failure scenario (AI — edge-case & provenance lens)**: In ThermoMPNN, `LightAttention` uses `Dropout(0.25)` and 1D convolutions over the sequence length axis $L$; if padding residues (`mask == 0`) are not masked before the `attention_convolution` `Softmax(dim=-1)`, length-bucket padding in `aminx` (e.g., padding $L=76$ to bucket $128$) will leak into the softmax denominator and shift $\Delta\Delta G$ predictions across buckets.
- **Mitigation addressed in spec**:
  1. `convert_weights.py` and `get_topology_for_checkpoint` explicitly verify `w_pos.weight.shape` against the skeleton before serializing/deserializing, and parity tests enforce `< 1e-4` nats agreement against upstream PyTorch.
  2. `ThermoMPNN`'s `LightAttention` implementation must apply `-inf` masking to `attention_convolution` logits at `mask == 0` positions prior to `jax.nn.softmax`, with a bucket-invariance test (`L=76` unpadded vs padded to `128` must be bit-identical within `1e-6`).

---

## Filed Technical Debt Registry (Praxia)

| Debt ID | Priority | Category | Title & Scope |
| :--- | :--- | :--- | :--- |
| **`#3072`** | `P2` | `feature` | **`[caliby] Correct Caliby misattribution (#5595) and port ProteinDesignLab/caliby ensemble-conditioned Potts model family`** — Corrects `#5595` (which misclassified Caliby as a sequence-recovery test bench) and tracks porting `ProteinDesignLab/caliby` (`weights/caliby/caliby.ckpt`, `weights/soluble_caliby/soluble_caliby.ckpt`, $M=32$ conformer-ensemble Protelle MPNN encoder + Potts decoder) under `src/aminx/families/caliby/`. |
| **`#3073`** | `P2` | `feature` | **`[fampnn] Support Tier 4 Full-Atom MPNN (richardshuai/fampnn) joint categorical sequence and continuous sidechain diffusion weights`** — Tracks Tier 4 joint sequence + 3D sidechain diffusion (`fampnn_0_2.pt`, `fampnn_0_3.pt`) once `StageSet` supports mixed discrete-categorical + continuous-coordinate denoising. |
| **`#3074`** | `P3` | `feature` | **`[antibmpnn] Add Tier 1B AntiBMPNN CDR frequency logit_transform and antibody CDR region masking helpers`** — Tracks optional Chothia/IMGT CDR region frequency bias helpers on top of `StageSet` `logit_transform`. |
| **`#3075`** | `P3` | `feature` | **`[cyclicmpnn] Optional head-to-tail cyclic sequence offset wrapping in compute_neighbor_offsets for CyclicMPNN`** — Tracks optional `cyclic_mask` / `cyclic: bool = False` in `compute_neighbor_offsets` (`src/aminx/model/features.py`) for macrocyclic head-to-tail boundary offsets. |
| **`#3076`** | `P3` | `documentation` | **`[docs] Document Tier 5 non-MPNN inverse folding backbones (ProteinDPO/ESM-IF1, Frame2seq, ProstT5) as out-of-family in model docs`** — Documents why non-MPNN inverse folding architectures are out-of-scope for `aminx` weight loading. |
| **`#3078`** | `P1` | `architecture` | **`[campaign/sinks] Support multi-backend campaign arms with minimal JIT recompilations and unified, examinable sink storage`** — Lifts `refuse_driver_family(..., surface="campaign")` in `src/aminx/host/campaign.py` and `_refuse_driver_modes` in `src/aminx/host/family_runner.py` so campaigns can declare arbitrary numbers of backends (`proteinmpnn`, `soluble`, `ca`, `abmpnn`, `hypermpnn`, `cyclicmpnn`, `ligand`, `pottsmpnn`, `protonpottsmpnn`, `lasermpnn`, `thermompnn`, `caliby`), schedules execution across topology classes and shared length buckets with dynamic weight PyTree hoisting to eliminate XLA recompilations, and unifies `StreamingTensorStagingSink` + `ZarrStagingSink` into a common, self-describing Zarr/Parquet sink structure. |

