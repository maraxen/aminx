# MPNN Variant Taxonomy (Tiers 1–5)

This document catalogs the ProteinMPNN variant hierarchy supported in `aminx` (Tiers 1–4) and documents why Tier 5 non-MPNN inverse folding backbones are out-of-family for `aminx` checkpoint loading.

## Overview of Tiers 1–5

| Tier | Family / Checkpoints | Architecture & Graph Topology | `aminx` Support Surface |
| :--- | :--- | :--- | :--- |
| **Tier 1A** | `original` (`proteinmpnn_v_48_*`), `soluble` (`solublempnn_v_48_*`), `abmpnn` (`abmpnn_v_48_020`), `hypermpnn` (`hypermpnn_v_48_*`), `cyclicmpnn` (`cyclicmpnn_v_48_010`) | Vanilla 3-encoder + 3-decoder `Aminx` (`hidden_dim=128`, `k_neighbors=48`, `num_positional_embeddings=32`, `edge_embed_in_dim=416`) | Native `load_model(model_weights=...)` (`src/aminx/model/mpnn.py`) |
| **Tier 1B** | `antibmpnn` (`antibmpnn_v_48_{000,010,020,030}`) | Vanilla `Aminx` (`k_neighbors=48`, `num_positional_embeddings=32`) + pure-JAX CDR region masks, CDR amino-acid frequency log-odds `logit_transform`, and Barley/Lehninger physicochemical descriptors (`PD1_volume`, `PD2_hydro`, `PD3_charge`) | Native `load_model(model_weights="antibmpnn")` + `src/aminx/model/cdr_bias.py` (`#3074`) |
| **Tier 1C** | `cyclicmpnn` with `cyclic=True` | Vanilla `Aminx` / `ProteinFeatures` / `ProteinFeaturesCA` with intra-chain cyclic sequence offset wrapping ($d_{\text{cyclic}} \in [-\lfloor L_c/2 \rfloor, +\lfloor L_c/2 \rfloor]$) | `compute_neighbor_offsets(..., cyclic=True)` & `Aminx(..., cyclic=True)` (`#3075`) |
| **Tier 2** | `ca` / `campnn` (`ca_v_48_{002,010,020}`) | CA-only coarse backbone geometry (`ProteinFeaturesCA`, `edge_embed_in_dim=167`: 16 pos + $9 \times 16$ CA-triplet RBFs + 7 local-frame orientation/quaternion features) | Native `load_model(model_weights="ca")` (`src/aminx/model/features.py`) |
| **Tier 3** | `thermompnn` (`thermo_mpnn_v1_default`, `thermo_mpnn_v2_ens{1,2,3}`, `thermo_mpnn_d_ens{1,2,3}`) | Frozen all-residues-visible `Aminx` trunk + `LightAttention1D` (`K=9` in v1, `K=1` in v2) + MLP $\Delta\Delta G^\circ$ head (`ThermoMPNN`) or Siamese double-mutant epistatic head (`ThermoMPNNDouble`) | `src/aminx/families/thermo_mpnn/` & `ThermoMPNNDriver` (`#3078`) |
| **Tier 4** | `pottsmpnn`, `protonpottsmpnn`, `lasermpnn`, `caliby`, `fampnn` | Custom graph/attention encoders and specialized heads: Potts $(h, J)$ energy tables (`PottsMPNN`, `ProtonPottsMPNN`), conformer-ensemble `GraphPotts` + EDM sidechain diffusion (`Caliby`, `#3072`), and `FAEncoder` + `GVPEncoder` + `SidechainDiffusionModule` + MAR/EDM joint sampling (`FAMPNN`, `#3073`) | Dedicated family packages under `src/aminx/families/<family>/` and `FamilyDriver` registry |
| **Tier 5** | `ProteinDPO` / `ESM-IF1`, `Frame2seq`, `ProstT5` | Non-MPNN inverse folding backbones (Geometric Vector Perceptron Transformer, Invariant Point Attention / framed edge-node network, 3Di bilingual T5 encoder-decoder) | **Out-of-family** (documented below; `#3076`) |

---

## Tier 1B: AntiBMPNN (`#3074`)

**Upstream Reference**: Sun et al. (2024), `zeysun/AntiBMPNN`, Zenodo DOI `10.5281/zenodo.13387792`.

### Checkpoints
- `antibmpnn_v_48_000.eqx.zst` ($0.00\text{ \AA}$ backbone noise; default for `model_weights="antibmpnn"`)
- `antibmpnn_v_48_010.eqx.zst` ($0.10\text{ \AA}$ backbone noise)
- `antibmpnn_v_48_020.eqx.zst` ($0.20\text{ \AA}$ backbone noise)
- `antibmpnn_v_48_030.eqx.zst` ($0.30\text{ \AA}$ backbone noise)

### Antibody CDR Helpers (`aminx.model.cdr_bias`)
- `build_cdr_region_mask(residue_index, chain_index, *, heavy_chain_id=0, light_chain_id=1, scheme="chothia", mask=None)`: Builds per-loop (`h1`, `h2`, `h3`, `l1`, `l2`, `l3`) and combined (`heavy_cdr`, `light_cdr`, `any_cdr`) boolean masks under Chothia or IMGT numbering while isolating heavy and light chains and respecting padding (`mask > 0`).
- `make_cdr_frequency_logit_transform(cdr_mask, *, target_frequencies=None, background_frequencies=None, weight=1.0, base_transform=None)`: Constructs a `StageSet`-compatible `CDRFrequencyLogitTransform` (`BatchLogitFn`) that adds log-odds amino-acid frequency bias on active CDR positions while keeping token `20` (`'X'`) unbiased.
- `compute_cdr_physicochemical_descriptors(sequences, cdr_mask=None, *, ph=7.4, include_termini=True)`: Pure-JAX computation of AntiBMPNN's three physicochemical evaluation descriptors (`pd1_volume` and `pd2_hydro` from Barley et al. 2018, and `pd3_charge` net Henderson-Hasselbalch charge at pH 7.4 using Lehninger $\text{pK}_a$ values).

---

## Tier 1C: CyclicMPNN & Cyclic Sequence Offset Wrapping (`#3075`)

**Upstream Reference**: `ParisaH-Lab/CyclicMPNN`.

When designing head-to-tail macrocyclic peptides (`cyclic=True`), intra-chain sequence offsets $d_{ij} = r_i - r_j$ wrap across the per-chain valid cycle length $L_c = \max(\max_{k \in c_{\text{valid}}} r_k - \min_{k \in c_{\text{valid}}} r_k + 1, 1)$ via shortest signed modular distance:
$$d_{\text{cyclic}}(i, j) = \bigl((d_{ij} + \lfloor L_c / 2 \rfloor) \bmod L_c\bigr) - \lfloor L_c / 2 \rfloor$$
with antisymmetric tie-breaking ($d_{\text{cyclic}}(j, i) = -d_{\text{cyclic}}(i, j)$) at the even-length antipode $d_{ij} = +L_c / 2$. Padded slots (`mask <= 0`) and cross-chain residue pairs are excluded from wrapping.

---

## Tier 4: Caliby (`#3072`) and FAMPNN (`#3073`)

### Caliby (`src/aminx/families/caliby/`)
- **Reference**: Shuai, Sidhu, & Huang (2025), *"Ensemble-conditioned protein sequence design with Caliby"*, *bioRxiv* `2025.09.30.679633` (`ProteinDesignLab/caliby`, Zenodo DOI `10.5281/zenodo.18644985`).
- **Architecture**: All-atom `CalibyTokenFeatures` (`model_version=0` Atom23 $23 \times 23$ RBFs `edge_in=8481`; `model_version=1` Atom14+virtual-$C_\beta$ `edge_in=1057`) + `CalibyMPNN` + score-parameterized `CalibyGraphPotts` (`num_factors=32`, `k_neighbors_potts=30`) with conformer-ensemble $(h, J)$ aggregation (`aggregate_ensemble_potts_params`) and `CalibySidechainMLP` + Karras `EDM` sidechain diffusion (`caliby_packer_{000,010,030}`).
- **Checkpoints**: `caliby`, `soluble_caliby`, `soluble_caliby_v1`, `caliby_distill`, `soluble_caliby_distill`, `caliby_packer_000`, `caliby_packer_010`, `caliby_packer_030`.

### FAMPNN (`src/aminx/families/fampnn/`)
- **Reference**: Shuai et al. (2025), *"Full-Atom Protein Sequence Design with Continuous Sidechain Diffusion"*, `richardshuai/fampnn`.
- **Architecture**: 5-encoder + 5-decoder `FAEncoder` (with 656-d `SidechainProteinFeatures` and edge-updating `DecLayer`) + 3-layer `GVPEncoder` (PyG `source_to_target` scatter-mean into `E_idx` and local-frame vector output rotation) + `mlp_seq` + 4-block AdaLN-Zero `SidechainDiffusionModule` (EDM ODE in local backbone frame) + optional 3-layer `SidechainConfidenceModule` (33-bin expected pSCE) + Masked Autoregressive (`MAR`) joint sequence-and-sidechain unmasking loop.
- **Checkpoints**: `fampnn_0_0`, `fampnn_0_3`, `fampnn_0_3_cath`.

---

## Tier 5: Out-of-Family Non-MPNN Inverse Folding Backbones (`#3076`)

The following inverse folding models are sometimes grouped alongside ProteinMPNN in benchmark suites, but share **zero module definitions, graph featurizers, or weight tensor layouts** with `Aminx` and are therefore **out-of-family** for `aminx`:

1. **`ESM-IF1` (`facebookresearch/esm`, Hsu et al., *ICML* 2022) and `ProteinDPO` (`wendao/ProteinDPO`, Widatalla et al., 2024)**:
   - **Architecture**: `GVPTransformerEncoder` (4 Geometric Vector Perceptron graph convolution layers with tuple scalar/vector channels + 8 autoregressive Transformer encoder-decoder layers with causal self-attention and cross-attention over backbone GVP embeddings, ~142M parameters).
   - **Why Out-of-Family**: `ProteinDPO` is a Direct Preference Optimization (DPO) fine-tune of `ESM-IF1` (`esm_if1_gvp4_t16_142M_UR50`), not a ProteinMPNN checkpoint. It has no message-passing `EncLayer`/`DecLayer` edge updates or `ProteinFeatures` RBF projections.
2. **`Frame2seq` (`kortemmelab/frame2seq`, Akpinaroglu et al., 2023)**:
   - **Architecture**: SE(3)-invariant framed edge-node iterative update blocks operating on explicit backbone rigid frames $(R_i, t_i)$ and $\psi/\phi$ torsion embeddings with parallel single-pass sequence logit prediction.
   - **Why Out-of-Family**: Uses custom frame-pair orientation/translation update blocks rather than Dauparas k-NN 25-pair Gaussian RBFs or autoregressive `Aminx` decoding.
3. **`ProstT5` (`mheinzinger/ProstT5`, Heinzinger et al., 2023)**:
   - **Architecture**: 3B-parameter T5 encoder-decoder language model translating between 1D amino-acid sequences and Foldseek 20-state 3Di structural tokens (`<AA2fold>` / `<fold2AA>`).
   - **Why Out-of-Family**: Operates on discrete 1D Foldseek 3Di strings via a 24-layer T5 Transformer with relative attention biases rather than 3D atomic coordinates or k-NN protein graphs.
