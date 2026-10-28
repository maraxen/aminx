---
title: "Caliby Family Port & Misattribution Correction Specification"
description: "Corrects the #5595 TRW marginal-calibration misattribution of Caliby and specifies the JAX/Equinox port of ProteinDesignLab/caliby (Shuai, Sidhu, & Huang, bioRxiv 2025.09.30.679633) as src/aminx/families/caliby/"
task_id: 261027_caliby-family-port-spec
date: '261027'
status: draft
backlog_ids: "debt #3072, corrects #5595"
adversarial_review: "Spec Swarm Agent B (Author + Challenger + Defender, 7 hazards resolved)"
---

# Caliby Family Port & Misattribution Correction Specification

## Goal

Correct the architectural misattribution of **Caliby** introduced under `#5595` (which falsely documented and tested "Caliby" as a 2-parameter tree-reweighted marginal temperature/field calibration wrapper around `src/aminx/potts/trw.py`) and specify the complete, numerically verified JAX/Equinox port of **`ProteinDesignLab/caliby`** (Shuai, Sidhu, & Huang, *"Ensemble-conditioned protein sequence design with Caliby"*, *bioRxiv* 2025.09.30.679633; DOI `10.1101/2025.09.30.679633`) as a first-class model family at `src/aminx/families/caliby/`.

Upstream `ProteinDesignLab/caliby` is a PyTorch Lightning protein sequence design and sidechain packing package (`caliby/model/seq_denoiser/lit_sd_model.py:13-154` [`V`]) consisting of:
1. **`CalibyMPNN` (`atom_mpnn.py:24-158` [`V`]) + `TokenFeatures` (`atom_mpnn.py:161-303` [`V`])**: An all-atom / Atom14 / Atom23 3-encoder + 3-decoder message-passing network (`hidden_dim = 128`, `k_neighbors = 48`, `n_tokens = 32` AlphaFold3 token state space) supporting both `model_version = 0` (`edge_in = 8481`, $23 \times 23$ all-atom RBFs) and `model_version = 1` (`edge_in = 1057`, $5 \times 5$ backbone+$C_\beta$ plus $4 \times 10$ backbone-sidechain RBFs).
2. **`GraphPotts` (`potts.py:40-289` [`V`]) + Conformer-Ensemble Aggregation (`atom_mpnn_denoiser.py:451-516` [`V`])**: A score-parameterized (`parameterization = "score"`, `symmetric_J = True`, `num_factors = 32`, `k_neighbors_potts = 30`, learned `log_scale`) Potts decoder head `decoder_S_potts` whose site fields $h$ and pairwise couplings $J$ are sliced from the 32-token AF3 state space to the 22-token Potts state space (`potts_state_space.py:69-88, 234-249` [`V`]) and averaged across $N_{\text{conf}}$ structural conformers in dense $(N, N, C, C)$ edge space before MCMC sampling.
3. **`SidechainDiffusionModule` (`scn_diffusion_mlp.py:21-333` [`V`]) + `EDM` (`edm_interpolant.py:10-217` [`V`])**: An AdaLN-modulated 4-block DiT-style residual MLP (`hidden_size = 256`, `in_channels = 230`, `out_channels = 99`) conditioned on frozen `CalibyMPNN` node embeddings `h_V` that denoises 33 non-backbone Atom37 sidechain coordinates in each residue's local $(C, C_\alpha, N)$ rigid frame via Karras et al. (2022) EDM preconditioning and Euler SDE/ODE integration.

This specification defines:
- **Part 1**: The historical correction to `.praxia/docs/research/260605_caliby-nature-and-dataset.md` and `scripts/recapture/caliby_recapture.py` (while preserving `src/aminx/potts/calibration.py` as general Potts marginal calibration utilities).
- **Part 2**: The complete module architecture (`src/aminx/families/caliby/{__init__.py,features.py,potts.py,sidechain_diffusion.py,model.py,convert.py,driver.py}`), weight conversion for all 8 official Zenodo checkpoints (`doi:10.5281/zenodo.18644985`), host registration (`family_driver.py`, `specs.py`, `options.py`, `prep.py`, `weights.py`), and L-DRV static AST compliance (`tests/lint/test_l_drv.py:1-533` [`V`]).
- **Part 3**: The internal Adversarial Spec Challenge (Challenger vs. Defender across 7 concrete technical hazards) and the graded numerical parity verification plan (`tests/families/test_caliby_parity.py`).

## Non-goals

- **Deleting or breaking `src/aminx/potts/calibration.py` (`calibration.py:1-159` [`V`])**: The functions `calibrate_temperature`, `calibrate_fields`, `apply_calibration`, and `CalibrationResult` remain valid general-purpose tree-reweighted Potts marginal calibration utilities used by `tests/potts/test_calibration.py`; they are already free of any "Caliby" naming (`calibration.py:1-21` [`V`]) and will not be removed.
- **Porting upstream training loops, DeepSpeed, or WandB callbacks (`lit_sd_model.py:30-153` [`V`])**: Only inference, ensemble Potts parameterization, Potts MCMC / annealed sampling, sidechain EDM diffusion sampling, and checkpoint conversion are in scope.
- **Porting `atomworks` or ` biotite` CIF parser pipelines (`caliby/data/` [`V`])**: `aminx` already parses PDB/mmCIF structures into `BackboneFrame` and Atom14/Atom37 coordinates via `src/aminx/io/` and `src/aminx/host/prep.py`.
- **Porting Protenix / AlphaFold3 conformer generation or Protpardelle-1c partial diffusion (`scripts/` in upstream `caliby`)**: Conformer ensembles are passed into `aminx` either as multi-model/multi-PDB conformer lists or generated via the Gaussian coordinate perturbation ensemble (`generate_gaussian_ensemble`) matching upstream `caliby/eval/eval_utils/seq_des_utils.py`.

## Functional requirements

- **FR-1**: Prepend an explicit `[!WARNING] HISTORICAL MISATTRIBUTION CORRECTION (Debt #3072)` banner to `.praxia/docs/research/260605_caliby-nature-and-dataset.md` (`260605_caliby-nature-and-dataset.md:1-34` [`V`]) and update `scripts/recapture/caliby_recapture.py` (`caliby_recapture.py:1-22` [`V`]) so that neither file claims "Caliby" is an internal TRW marginal-calibration sidecar; document that `calibration.py` implements generic TRW temperature/field scaling whereas Caliby is Shuai, Sidhu, & Huang (*bioRxiv* 2025.09.30.679633, `ProteinDesignLab/caliby`).
- **FR-2**: Implement `CalibyPositionalEncodings` (`atom_mpnn.py:319-332` [`V`]) and `CalibyTokenFeatures` (`atom_mpnn.py:161-303` [`V`]) in `src/aminx/families/caliby/features.py` supporting both `model_version = 0` (`edge_in = 8481`, Atom23 padded with 9 dummy $C_\alpha$ slots, $23 \times 23 = 529$ pairwise 16-bin RBFs + 16 pos + 1 `token_bonds`, `atom_mpnn.py:180-181, 204-213` [`V`]) and `model_version = 1` (`edge_in = 1057`, Atom14 + virtual $C_\beta$, $5 \times 5 = 25$ backbone+$C_\beta$ pairs + $4 \times 10 = 40$ backbone-sidechain pairs = 65 pairwise 16-bin RBFs + 16 pos + 1 `token_bonds`, `atom_mpnn.py:182-185, 214-215, 278-303` [`V`]), using purely vectorized array broadcasting with zero Python loops over atom pairs so `tests/lint/test_l_drv.py` R1/R2/R4 pass without allowlist exemptions.
- **FR-3**: Implement `CalibyEncLayer` (`atom_mpnn.py:393-447` [`V`]), `CalibyDecLayer` (`atom_mpnn.py:335-390` [`V`]), and `CalibyMPNN` (`atom_mpnn.py:24-158` [`V`]) in `src/aminx/families/caliby/model.py`, faithfully reproducing: (a) masking `restype` with `<G>` (AF3 index 31) where `seq_cond_mask == 0` (`atom_mpnn.py:103-106` [`V`]); (b) pre-encoder initialization `h_V = W_s(restype)` (`atom_mpnn.py:99, 107, 113` [`V`]); (c) `W_e = eqx.nn.Linear(128, 128, use_bias=False)` and `W_s = eqx.nn.Linear(32, 128, use_bias=False)` (`atom_mpnn.py:38, 41` [`V`]); (d) parallel all-at-once decoder message passing with edge updates (`update_edges = True` on all 3 decoder layers for `seq_des`/`student-seq_des`, and `update_edges = False` on `decoder_layers[2]` when `task == "scn_pack"`, `atom_mpnn.py:54-65` [`V`]); and (e) `W_out = None` when `task == "student-seq_des"` (`atom_mpnn.py:88-89` [`V`]).
- **FR-4**: Implement `CalibyGraphPotts` (`potts.py:40-289` [`V`]), `slice_potts_params` (`potts_state_space.py:69-88, 234-249` [`V`]), and exact alphabet permutation tables (`AF3_TO_POTTS22_IDXS`, `POTTS22_TO_AMINX21_PERM`, `AMINX21_TO_POTTS22_PERM`) in `src/aminx/families/caliby/potts.py`, reproducing `parameterization = "score"`, `num_factors = 32`, `scale = jnp.exp(log_scale)`, zero-sum gauge projection on `h` and `J`, background bias heads `W_h_bg` and `W_J_bg`, reciprocal-edge coupling symmetrization `_symmetrize_J` (`potts.py:178-198, 247-248, 276-288` [`V`]), and mapping between upstream `const.STANDARD_AA` 3-letter alphabetical order (`ALA, ARG, ASN, ...`, `const.py:28-37, 138-139` [`V`]) and `aminx` 1-letter alphabetical order (`ACDEFGHIKLMNPQRSTVWYX`, `specs.py:22` [`V`]).
- **FR-5**: Implement conformer-ensemble Potts parameter aggregation `aggregate_ensemble_potts_params` (`atom_mpnn_denoiser.py:451-516` [`V`]) and Gaussian coordinate perturbation ensemble generation `generate_gaussian_ensemble` in `src/aminx/families/caliby/potts.py`, scattering per-conformer sparse $(N, K_{\text{potts}}, C, C)$ couplings onto a shared dense $(N, N, C, C)$ graph with intersection node masking (`mask_i_new = (node_counts == N_conf)`), union edge masking (`mask_ij_new = (edge_counts > 0) * mask_i_new[:, None] * mask_i_new[None, :]`), and arithmetic mean scaling (`h_new / N_conf`, `J_new / N_conf`).
- **FR-6**: Implement `CalibySidechainMLP`, `DenoisingMLPBlock`, `TimestepEmbedder`, `FinalLayer`, local backbone rigid frame projection `global_to_local_scn` / `local_to_global_scn`, and the Karras `EDM` Euler SDE/ODE sampler in `src/aminx/families/caliby/sidechain_diffusion.py` (`scn_diffusion_mlp.py:21-333`, `edm_interpolant.py:10-217`, `dit_utils.py:18-134` [`V`]), operating on the 33 non-backbone Atom37 sidechain coordinates (`AF2_SCN_IDXS`, `99` channels) in the `rigid_from_3_points(C, CA, N)` local frame.
- **FR-7**: Implement top-level `CalibyModel(eqx.Module)` in `src/aminx/families/caliby/model.py` combining `features: CalibyTokenFeatures`, `mpnn: CalibyMPNN`, `potts_decoder: CalibyGraphPotts | None`, and `scn_diffusion: CalibySidechainMLP | None`, exposing `encode_conformer`, `score_ensemble`, and `pack_sidechains`.
- **FR-8**: Implement PyTorch Lightning `.ckpt` $\rightarrow$ `.eqx.zst` weight conversion in `src/aminx/families/caliby/convert.py` supporting all 8 official Zenodo checkpoints (`caliby`, `soluble_caliby`, `soluble_caliby_v1`, `caliby_distill`, `soluble_caliby_distill`, `caliby_packer_000`, `caliby_packer_010`, `caliby_packer_030`), automatically applying `repair_state_dict` (`_orig_mod.` stripping, `checkpoint_utils.py:139-156` [`V`]) and `migrate_legacy_state_dict` (`denoiser.atom_mpnn.` $\rightarrow$ `denoiser.mpnn.`, `checkpoint_utils.py:20-39` [`V`]), and register the 8 weight descriptors in `src/aminx/io/family_weights.json` (`family_weights.json:1-78` [`V`]) and `src/aminx/io/weights.py`.
- **FR-9**: Implement `CalibyDriver(FamilyDriver)` in `src/aminx/families/caliby/driver.py` and register `"caliby"` across `src/aminx/host/family_driver.py` (`family_driver.py:173-185` [`V`]), `src/aminx/run/options.py` (`CalibyOptions`, `options.py:113-145` [`V`]), `src/aminx/run/specs.py` (`FamilyName`, `DesignSpec`, `specs.py:28-36, 420-520` [`V`]), and `src/aminx/host/prep.py` (`prep.py:155-249` [`V`]), while passing all `tests/lint/test_l_drv.py` static AST checks (`R1`–`R5`).
- **FR-10**: Implement graded numerical parity and negative-control tests in `tests/families/test_caliby_parity.py` covering `CalibyTokenFeatures` (`model_version=0` and `model_version=1`), `CalibyMPNN`, `CalibyGraphPotts` (`slice_potts_params` + `_symmetrize_J`), `aggregate_ensemble_potts_params`, `CalibySidechainMLP` + `EDM.euler_step`, and `convert.py` round-trip verification (`float64` max abs diff $< 10^{-6}$, `float32` max abs diff $< 10^{-4}$).

---

## Part 1: Historical Misattribution Audit & Correction Plan (`#5595` $\rightarrow$ `#3072`)

### 1.1 Root Cause of the `#5595` Misattribution

1. **`.praxia/docs/research/260605_caliby-nature-and-dataset.md` (`260605_caliby-nature-and-dataset.md:1-178` [`V`])**:
   - Authored on `2026-06-05` to investigate `scripts/recapture/caliby_recapture.py` (`#5595`).
   - Lines `9-34` [`V`] concluded that *"Caliby is an internal aminx methodology — not an external published package"*, claiming it stood for *"Calibration of Potts parameters via tree-reweighted marginal matching"* (`src/aminx/potts/calibration.py:1-159` [`V`]).
   - This conclusion was reached because `caliby_recapture.py` had been stubbed against `calibrate_temperature` / `calibrate_fields` in `src/aminx/potts/calibration.py`, and the author of `260605_caliby-nature-and-dataset.md` searched only the local repository rather than the protein design literature.
2. **`scripts/recapture/caliby_recapture.py` (`caliby_recapture.py:1-148` [`V`])**:
   - Lines `1-19` [`V`] state: *"Caliby recapture script: post-hoc temperature & field calibration for Potts models."*
   - Lines `27-105` [`V`] generate synthetic random Gaussian Potts parameters $(h, J)$ on a 12-residue ring graph and test `calibrate_temperature` and `calibrate_fields` from `aminx.potts.calibration`.
3. **`src/aminx/potts/calibration.py` (`calibration.py:1-159` [`V`])**:
   - Contains `CalibrationResult`, `calibrate_temperature`, `calibrate_fields`, and `apply_calibration`.
   - Crucially, `src/aminx/potts/calibration.py` itself **never uses the word "Caliby"** (`calibration.py:1-21` [`V`]). It is a clean, general-purpose mathematical module for tree-reweighted (TRW) marginal temperature and site-field calibration.

### 1.2 What Caliby Actually Is (`ProteinDesignLab/caliby`)

**Caliby** is an external published model family from the Possu Huang lab at Stanford University (`ProteinDesignLab/caliby`):
- **Citation**: Richard W. Shuai,inderjit S. Sidhu, & Possu Huang (2025). *"Ensemble-conditioned protein sequence design with Caliby."* *bioRxiv* 2025.09.30.679633; DOI: `10.1101/2025.09.30.679633`.
- **Repository & Weights**: `https://github.com/ProteinDesignLab/caliby`, weights hosted at Zenodo (`doi:10.5281/zenodo.18644985`).
- **Core Mechanism**:
  - Caliby performs **conformer-ensemble-conditioned protein sequence design** and **diffusion-based all-atom sidechain packing**.
  - Rather than conditioning an autoregressive decoder on a single static backbone, Caliby runs an AtomMPNN encoder-decoder (`CalibyMPNN`, `atom_mpnn.py:24-158` [`V`]) + score-parameterized `GraphPotts` head (`potts.py:40-289` [`V`]) independently on $N_{\text{conf}}$ structural conformers of a backbone ensemble (generated via MD, AlphaFold3/Protenix sampling, Protpardelle-1c partial diffusion, or Gaussian coordinate noise) and **averages the resulting Potts fields $h$ and couplings $J$ across the ensemble** (`_aggregate_potts_params`, `atom_mpnn_denoiser.py:451-516` [`V`]) before sampling sequences via annealed Potts MCMC (`potts.py:292-620` [`V`]).
  - For sidechain packing (`task = "scn_pack"`), Caliby runs `CalibyMPNN` on the backbone + sequence to produce node embeddings `h_V` (`(B, N, 128)`) that condition an AdaLN-modulated `SidechainMLP` (`scn_diffusion_mlp.py:160-276` [`V`]) trained with an `EDM` diffusion interpolant (`edm_interpolant.py:10-217` [`V`]) over the 33 non-backbone Atom37 coordinates in each residue's local $(C, C_\alpha, N)$ frame.

### 1.3 Required Corrections (FR-1)

1. **Update `.praxia/docs/research/260605_caliby-nature-and-dataset.md`**:
   - Set YAML frontmatter `status: superseded_by_261027_caliby-family-port-spec` and add a top-level `> [!WARNING] HISTORICAL MISATTRIBUTION CORRECTION (Debt #3072)` block immediately after the YAML frontmatter explaining:
     - The `2026-06-05` note falsely concluded that "Caliby" was an internal `aminx` portmanteau for tree-reweighted Potts marginal calibration (`src/aminx/potts/calibration.py`).
     - In reality, **Caliby** is the ensemble-conditioned Potts sequence design and EDM sidechain packing model by Shuai, Sidhu, & Huang (*bioRxiv* 2025.09.30.679633; `https://github.com/ProteinDesignLab/caliby`), ported in `src/aminx/families/caliby/`.
     - `src/aminx/potts/calibration.py` remains in `aminx.potts` as general TRW marginal calibration infrastructure and is unrelated to Caliby.
2. **Update `scripts/recapture/caliby_recapture.py`**:
   - Update module docstring (`caliby_recapture.py:1-19` [`V`]) and JSON summary metadata (`caliby_recapture.py:114-125` [`V`]) to explicitly distinguish:
     - `trw_marginal_calibration`: the synthetic TRW temperature/field recovery check exercising `aminx.potts.calibration` (historically mislabeled `#5595`).
     - `caliby_family_recapture`: forward-pass and ensemble-Potts parity check exercising `aminx.families.caliby` (`CalibyModel`, `CalibyGraphPotts`, `aggregate_ensemble_potts_params`).

---

## Part 2: Upstream `ProteinDesignLab/caliby` Architecture & `src/aminx/families/caliby/` Port Design

### 2.1 Official Checkpoint Inventory (`doi:10.5281/zenodo.18644985`)

All 8 official PyTorch Lightning checkpoints (`ckpt["state_dict"]` + `ckpt["hyper_parameters"]["cfg"]`, `checkpoint_utils.py:82-96` [`V`]) must be supported by `src/aminx/families/caliby/convert.py`:

| Checkpoint ID (`family_weights.json`) | Upstream Filename | Task (`cfg.model.denoiser.task`) | `model_version` | `edge_in` (`edge_embedding.weight`) | Key Structural Differences |
|---|---|---|---|---|---|
| `caliby` | `caliby.ckpt` | `"seq_des"` | `0` | `8481` (`(128, 8481)`) | Atom23 ($23\times 23$ RBFs), `W_out` present (`(32, 128)`), `decoder_S_potts` present (`k_neighbors_potts=30`), legacy prefix `denoiser.atom_mpnn.*` + `_orig_mod.` |
| `soluble_caliby` | `soluble_caliby.ckpt` | `"seq_des"` | `0` | `8481` (`(128, 8481)`) | Soluble-trained `model_version=0` weights; identical tensor shapes to `caliby.ckpt` |
| `soluble_caliby_v1` | `soluble_caliby_v1.ckpt` | `"seq_des"` | `1` | `1057` (`(128, 1057)`) | Atom14 + virtual $C_\beta$ ($5\times 5 + 4\times 10 = 65$ RBFs), prefix `denoiser.mpnn.*` |
| `caliby_distill` | `caliby_distill.ckpt` | `"student-seq_des"` | `0` | `8481` (`(128, 8481)`) |Ensemble-distilled student model; **no `W_out` weights** (`atom_mpnn.py:88-89` [`V`]); sidechain conditioning forbidden (`atom_mpnn_denoiser.py:321-336` [`V`]) |
| `soluble_caliby_distill` | `soluble_caliby_distill.ckpt` | `"student-seq_des"` | `0` | `8481` (`(128, 8481)`) | Soluble ensemble-distilled student model; **no `W_out` weights** |
| `caliby_packer_000` | `caliby_packer_000.ckpt` | `"scn_pack"` | `0` | `8481` (`(128, 8481)`) | Sidechain packer ($\sigma_{\text{noise}} = 0.0\text{ \AA}$); **no `decoder_S_potts`**; `decoder_layers.2` has `update_edges=False` (no `W11, W12, W13, norm3`, `atom_mpnn.py:54-65` [`V`]); includes `sidechain_diffusion_module.net.*` (`scn_diffusion_mlp.py:21-60` [`V`]) |
| `caliby_packer_010` | `caliby_packer_010.ckpt` | `"scn_pack"` | `0` | `8481` (`(128, 8481)`) | Sidechain packer trained with $0.10\text{ \AA}$ backbone noise; same architecture as `caliby_packer_000` |
| `caliby_packer_030` | `caliby_packer_030.ckpt` | `"scn_pack"` | `0` | `8481` (`(128, 8481)`) | Sidechain packer trained with $0.30\text{ \AA}$ backbone noise; same architecture as `caliby_packer_000` |

### 2.2 Alphabet & Token State Space Mapping (`const.py` & `potts_state_space.py` [`V`])

Upstream `ProteinDesignLab/caliby` uses two token alphabets defined in `caliby/data/const.py:28-37, 138-139` [`V`]:
1. **32-Token AlphaFold3 Alphabet (`AF3_TOKENS`, `n_tokens = 32`)**:
   - Indices `0..19`: `atomworks.constants.STANDARD_AA`, sorted **alphabetically by 3-letter CCD code**:
     ```
     0: ALA (A),  1: ARG (R),  2: ASN (N),  3: ASP (D),  4: CYS (C),
     5: GLN (Q),  6: GLU (E),  7: GLY (G),  8: HIS (H),  9: ILE (I),
     10: LEU (L), 11: LYS (K), 12: MET (M), 13: PHE (F), 14: PRO (P),
     15: SER (S), 16: THR (T), 17: TRP (W), 18: TYR (Y), 19: VAL (V)
     ```
   - Index `20`: `UNK` (`X`, unknown amino acid).
   - Indices `21..30`: 10 RNA/DNA tokens (`A, C, G, U, N, DA, DC, DG, DT, DN`).
   - Index `31`: `<G>` (gap / mask token, `const.GAP = "<G>"`).
2. **22-Token Potts Alphabet (`POTTS_TOKENS`, `const.py:138-139` [`V`])**:
   - Sliced from `AF3_TOKENS` via `POTTS_AF3_TOKEN_IDXS = [0, 1, ..., 20, 31]` (`potts_state_space.py:69-88` [`V`]).
   - Indices `0..19` are `STANDARD_AA` (`A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V`), index `20` is `UNK` (`X`), and index `21` is `<G>` (gap).
3. **Bijection to `aminx` 21-Token Alphabet (`specs.py:22` [`V`])**:
   - `aminx` uses 1-letter alphabetical order: `ACDEFGHIKLMNPQRSTVWYX` (indices `0..19` standard AAs, index `20` = `X`).
   - In `src/aminx/families/caliby/potts.py`, we define static tuple/array constants:
     - `CALIBY_AA20_1LETTER = ("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")`
     - `POTTS_AF3_TOKEN_IDXS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 31)`
     - `AMINX21_TO_CALIBY21_IDXS = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20)` (maps `aminx` index $a \in [0..20]$ to Caliby index $c \in [0..20]$).
     - `CALIBY21_TO_AMINX21_IDXS = (0, 14, 11, 2, 1, 13, 3, 5, 6, 7, 9, 8, 10, 4, 12, 15, 16, 18, 19, 17, 20)` (maps Caliby index $c \in [0..20]$ to `aminx` index $a \in [0..20]$).

### 2.3 Module Breakdown under `src/aminx/families/caliby/`

#### 2.3.1 `src/aminx/families/caliby/features.py` (FR-2)

Ports `PositionalEncodings` (`atom_mpnn.py:319-332` [`V`]) and `TokenFeatures` (`atom_mpnn.py:161-303` [`V`]):
- **`CalibyPositionalEncodings(eqx.Module)`**:
  - Fields: `max_relative_feature: int = eqx.field(static=True)` (`32`), `linear: eqx.nn.Linear` (`in_features = 2 * 32 + 2 = 66`, `out_features = 16`, `use_bias = True`).
  - Note on `use_multichain_encoding = False` (`atom_mpnn.py:173, 218-233` [`V`]): All 8 released Caliby checkpoints set `use_multichain_encoding = False`. Before computing `offset = r[:, None] - r[None, :]`, upstream shifts `residue_index` by `asym_id * (jnp.max(residue_index) + 100 + 1)` (`_RESIDUE_INDEX_GAP = 100`, `atom_mpnn.py:18, 224-228` [`V`]) and sets `d_chains = jnp.ones_like(E_idx)` (`atom_mpnn.py:231` [`V`]).
- **`CalibyTokenFeatures(eqx.Module)`**:
  - Static fields: `edge_features: int` (`128`), `node_features: int` (`128`), `num_positional_embeddings: int` (`16`), `num_rbf: int` (`16`), `top_k: int` (`48`), `model_version: int` (`0` or `1`), `use_multichain_encoding: bool` (`False`).
  - Learned layers: `embeddings: CalibyPositionalEncodings`, `edge_embedding: eqx.nn.Linear` (`8481 -> 128` for `v0`, `1057 -> 128` for `v1`, `use_bias=True`), `norm_edges: eqx.nn.LayerNorm(128)`.
  - **Input masking (`atom_mpnn.py:194-196` [`V`])**:
    - Given Atom14 coordinates `X_atom14: Array[N, 14, 3]`, atom mask `atom_mask: Array[N, 14]`, and conditioning mask `X_cond_mask: Array[N, 14]` (where backbone atoms `0..3` are always `1` and sidechain atoms `4..13` are `1` only for conditioned/fixed sidechain residues), replace any atom with `X_cond_mask == 0` by that residue's $C_\alpha$ coordinate `X_atom14[:, 1:2, :]`.
  - **KNN Graph (`_dist`, `atom_mpnn.py:251-260` [`V`])**:
    - Pairwise Euclidean distance on $C_\alpha$ (`X[:, 1, :]`): `D = jnp.sqrt(jnp.sum((ca[:, None, :] - ca[None, :, :]) ** 2, axis=-1) + 1e-6)`.
    - Masked residues get distance offset `(1.0 - mask_2d) * 10000.0`.
    - `k = min(self.top_k, N)`; select smallest `k` distances via `jax.lax.top_k(-D, k)` (giving `D_neighbors: Array[N, K]`, `E_idx: Array[N, K]`).
  - **Vectorized RBF Computation (L-DRV R2 Compliance)**:
    - Upstream `model_version == 0` loops `for i in range(23): for j in range(23):` (`atom_mpnn.py:209-211` [`V`]) and `model_version == 1` loops `for i in range(5): for j in range(5):` and `for i in range(4): for j in range(10):` (`atom_mpnn.py:293-301` [`V`]).
    - Because `tests/lint/test_l_drv.py` R2 (`test_l_drv.py:76-98` [`V`]) forbids `for` loops over `range(n)` with `n > 4` inside `eqx.Module.__call__`, `CalibyTokenFeatures.__call__` computes pairwise distances via **pure tensor broadcasting**:
      - **`model_version == 0`**: Pad `X_masked: Array[N, 14, 3]` with 9 copies of $C_\alpha$ (`jnp.repeat(X_masked[:, 1:2, :], 9, axis=1)`) to form `X_23: Array[N, 23, 3]` (`atom_mpnn.py:205-206` [`V`]). Gather neighbor coordinates `X_23_nbr = X_23[E_idx]` (`(N, K, 23, 3)`). Compute all $23 \times 23$ distances simultaneously:
        `D_23 = jnp.sqrt(jnp.sum((X_23[:, None, :, None, :] - X_23_nbr[:, :, None, :, :]) ** 2, axis=-1) + 1e-6)` (`(N, K, 23, 23)`), evaluate the 16-bin Gaussian RBFs (`D_min=2.0`, `D_max=22.0`, `D_count=16`, `atom_mpnn.py:262-276` [`V`]) to `(N, K, 23, 23, 16)`, and reshape to `(N, K, 8464)`.
      - **`model_version == 1`**: Compute virtual $C_\beta$ (`b = CA - N`, `c = C - CA`, `a = jnp.cross(b, c)`, `Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA`, `atom_mpnn.py:286-289` [`V`]). Form `X_bb5 = jnp.concatenate([X_masked[:, :4, :], Cb[:, None, :]], axis=1)` (`(N, 5, 3)`) and `X_sc10 = X_masked[:, 4:14, :]` (`(N, 10, 3)`). Broadcast `X_bb5` against `X_bb5[E_idx]` $\rightarrow$ `(N, K, 5, 5, 16)` $\rightarrow$ `(N, K, 400)` and `X_bb4 = X_masked[:, :4, :]` against `X_sc10[E_idx]` $\rightarrow$ `(N, K, 4, 10, 16)` $\rightarrow$ `(N, K, 640)`. Concatenate `[RBF_bb5, RBF_bb4_sc10]` $\rightarrow$ `(N, K, 1040)`.
  - **Final Edge Tensor (`atom_mpnn.py:235-248` [`V`])**:
    - Concatenate `[E_positional (16), RBF_all (8464 or 1040), E_token_bonds (1)]` along axis `-1`, apply `edge_embedding` + `norm_edges`, giving `E: Array[N, K, 128]`, `E_idx: Array[N, K]`, `D_neighbors: Array[N, K]`.

#### 2.3.2 `src/aminx/families/caliby/model.py` (FR-3, FR-7)

Ports `PositionWiseFeedForward` (`atom_mpnn.py:306-316` [`V`]), `EncLayer` (`atom_mpnn.py:393-447` [`V`]), `DecLayer` (`atom_mpnn.py:335-390` [`V`]), `CalibyMPNN` (`atom_mpnn.py:24-158` [`V`]), and top-level `CalibyModel`:
- **`CalibyEncLayer(eqx.Module)`**:
  - `W1(384->128)`, `W2(128->128)`, `W3(128->128)`, `W11(384->128)`, `W12(128->128)`, `W13(128->128)`, `norm1(128)`, `norm2(128)`, `norm3(128)`, `dense: PositionWiseFeedForward(128, 512)`, `act = jax.nn.gelu` (exact `nn.GELU()`, `atom_mpnn.py:416` [`V`]).
  - Note: In `CalibyMPNN.__init__` (`atom_mpnn.py:45-50` [`V`]), all 3 encoder layers are constructed with default `is_last_layer = False`, so every `CalibyEncLayer` updates both `h_V` and `h_E`.
- **`CalibyDecLayer(eqx.Module)`**:
  - `W1(384->128)`, `W2(128->128)`, `W3(128->128)`, `norm1(128)`, `norm2(128)`, `dense: PositionWiseFeedForward(128, 512)`, plus optional `W11(384->128)`, `W12(128->128)`, `W13(128->128)`, `norm3(128)` when `update_edges = True` (`atom_mpnn.py:54-65, 340-390` [`V`]).
  - For `task in ("seq_des", "student-seq_des")`, all 3 decoder layers have `update_edges = True`. For `task == "scn_pack"`, `decoder_layers[0]` and `decoder_layers[1]` have `update_edges = True` while `decoder_layers[2]` has `update_edges = False` (`W11 = W12 = W13 = norm3 = None`).
- **`CalibyMPNN(eqx.Module)`**:
  - Forward pass (`atom_mpnn.py:96-158` [`V`]):
    1. Extract `(E, E_idx, D_neighbors) = self.features(...)`.
    2. Initialize `h_E = self.W_e(E)` (`W_e` has `use_bias = False`).
    3. Mask sequence conditioning: where `seq_cond_mask == 0`, replace `restype` (32-dim one-hot) with one-hot of `<G>` (index 31) (`atom_mpnn.py:103-106` [`V`]).
    4. Compute `h_S = self.W_s(restype_masked)` (`W_s` has `use_bias = False`) and set **`h_V = h_S`** BEFORE the encoder layers (`atom_mpnn.py:99, 107, 113` [`V`]).
    5. Run 3 `encoder_layers` with `mask_V = token_mask` and `mask_attend = token_mask[E_idx]`.
    6. Run 3 `decoder_layers` in parallel: at each layer, form `h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)` (`(N, K, 256)`), then `h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)` (`(N, K, 384)`), and update `(h_V, h_E) = layer(h_V, h_ESV, h_E, token_mask, token_mask_2d)`.
    7. If `task in ("seq_des", "student-seq_des")`, form final `h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)` and `h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)`, truncate `(h_ESV, E_idx, token_mask_2d)` to `:k_neighbors_potts` (`30`), and invoke `self.decoder_S_potts(h_V, h_ESV, E_idx, token_mask, token_mask_2d)`.

#### 2.3.3 `src/aminx/families/caliby/potts.py` (FR-4, FR-5)

Ports `GraphPotts` (`potts.py:40-289` [`V`]), `slice_potts_params` (`potts_state_space.py:69-88, 234-249` [`V`]), `_aggregate_potts_params` (`atom_mpnn_denoiser.py:451-516` [`V`]), and Potts MCMC / energy evaluation (`potts.py:292-620` [`V`]):
- **`CalibyGraphPotts(eqx.Module)`**:
  - Layers (`parameterization = "score"`, `potts.py:115-124` [`V`]):
    - `W_h: eqx.nn.Linear(128, 32, use_bias=True)`
    - `W_h_bg: eqx.nn.Linear(128, 1, use_bias=True)`
    - `W_J_left: eqx.nn.Linear(384, 1024, use_bias=True)` (`32 * num_factors` where `num_factors = 32`)
    - `W_J_right: eqx.nn.Linear(384, 1024, use_bias=True)`
    - `W_J_bg: eqx.nn.Linear(384, 1, use_bias=True)`
    - `log_scale: Array` (shape `(1,)`)
  - Forward pass (`_get_potts_params`, `potts.py:178-198, 240-258` [`V`]):
    1. `scale = jnp.exp(self.log_scale)`
    2. `h = scale * self.W_h(h_V)` (`(N, 32)`)
    3. `J_left = self.W_J_left(h_E).reshape(N, K, 32, 32)` and `J_right = self.W_J_right(h_E).reshape(N, K, 32, 32)`
    4. `J = scale * jnp.einsum("nkcf,nkdf->nkcd", J_left, J_right)` (`potts.py:187` [`V`]: `torch.matmul(J_left, J_right.transpose(-1, -2))` over `num_factors` axis!)
    5. Zero-sum gauge (`potts.py:193-194` [`V`]):
       - `h = h - jnp.mean(h, axis=-1, keepdims=True)`
       - `J = J - jnp.mean(J, axis=-1, keepdims=True) - jnp.mean(J, axis=-2, keepdims=True) + jnp.mean(J, axis=(-2, -1), keepdims=True)`
    6. Add background bias (`potts.py:197-198` [`V`]):
       - `h = h + scale * self.W_h_bg(h_V)`
       - `J = J + (scale * self.W_J_bg(h_E))[..., None]`
    7. Symmetrize couplings (`_symmetrize_J`, `potts.py:247-248, 276-288` [`V`]):
       - For every directed edge $(i, k)$ with neighbor $j = \text{E\_idx}[i, k]$, look up whether $i$ appears in $\text{E\_idx}[j, :]$; if so, fetch $J_{ji}^{\top}$ (transposing the two state-space axes `(-2, -1)`) and set $J_{ij} \leftarrow \frac{1}{2} m_{ji} (J_{ij} + J_{ji}^{\top})$ where $m_{ji} = 1$ iff the reverse edge exists in $j$'s top-$K_{\text{potts}}$ list.
    8. Mask invalid nodes/edges (`potts.py:251-252` [`V`]): `h = mask_i[:, None] * h`, `J = mask_ij[:, :, None, None] * J`.
    9. Slice 32-token AF3 state space to 22-token Potts state space (`slice_potts_params`, `potts_state_space.py:69-88` [`V`]):
       - `h_22 = -h[:, POTTS_AF3_TOKEN_IDXS]` (note: `CalibyMPNN` returns `-h, -J` when returning energy parameters or keeps sign convention consistent with `potts.energy`: in `potts.py:303-312` [`V`], `U_ij = -J`, `U_i = -h`, so higher `h, J` means lower energy / higher probability!).
- **`aggregate_ensemble_potts_params` (`atom_mpnn_denoiser.py:451-516` [`V`])**:
  - Given a list/stack of $N_{\text{conf}}$ conformer Potts outputs `[(h_c, J_c, E_idx_c, mask_i_c, mask_ij_c)]`:
    1. `mask_i_ens = jnp.all(mask_i_stack > 0.5, axis=0).astype(jnp.float32)` (node $i$ is valid iff present in **all** conformers, `atom_mpnn_denoiser.py:488` [`V`]).
    2. Sum `h_ens = jnp.sum(h_stack, axis=0) * mask_i_ens[:, None] / N_conf`.
    3. Scatter each conformer's sparse `J_c: Array[N, K, C, C]` and `mask_ij_c: Array[N, K]` into a dense `(N, N, C, C)` coupling tensor and `(N, N)` edge count matrix using `E_idx_c`, sum across all $N_{\text{conf}}$ conformers, set `mask_ij_ens = (edge_counts > 0).astype(jnp.float32) * (mask_i_ens[:, None] * mask_i_ens[None, :])`, and divide `J_ens` by `N_conf` (`use_mean = True`, `atom_mpnn_denoiser.py:506-514` [`V`]).

#### 2.3.4 `src/aminx/families/caliby/sidechain_diffusion.py` (FR-6)

Ports `SidechainDiffusionModule` & `SidechainMLP` (`scn_diffusion_mlp.py:21-333` [`V`]), `DenoisingMLPBlock`, `TimestepEmbedder`, `FinalLayer` (`dit_utils.py:18-134` [`V`]), and `EDM` (`edm_interpolant.py:10-217` [`V`]):
- **Local Backbone Frame (`scn_diffusion_mlp.py:279-333` [`V`])**:
  - Extracts `AF2_SCN_IDXS` (the 33 Atom37 indices excluding backbone `N=0, CA=1, C=2, O=4`, `const.py:149-150` [`V`]).
  - Builds local rigid frame from `(x1=C, x2=CA, x3=N)` via Gram-Schmidt (`rigid_from_3_points`, `scn_diffusion_mlp.py:285-293` [`V`]).
  - Maps global sidechain coordinates to local frame `X_local = R^T (X_global - CA)`, zeroing out non-existent atoms (`atom37_mask == 0`), and flattens `(N, 33, 3) -> (N, 99)`.
- **`CalibySidechainMLP(eqx.Module)` (`scn_diffusion_mlp.py:160-276` [`V`])**:
  - `in_channels = 230` (`99` noised local sidechain coords + `99` self-conditioning coords + `32` one-hot `aatype`), `out_channels = 99`, `hidden_size = 256`, `cond_C = 128`, `depth = 4`, `mlp_ratio = 4.0`.
  - `x_embedder: eqx.nn.Linear(230, 256)`, `t_embedder: TimestepEmbedder(256, 256)`, `h_V_embedder: eqx.nn.Linear(128, 256)`.
  - Conditioning vector `c = t_embedder(c_noise) + h_V_embedder(h_V)` (`(N, 256)`).
  - 4 `DenoisingMLPBlock`s: each has `norm1` (`LayerNorm` without affine params, `eps=1e-6`), `mlp` (`Linear(256, 1024) -> jax.nn.gelu(..., approximate=True) -> Linear(1024, 256)`), and `adaLN_modulation` (`jax.nn.silu -> Linear(256, 768)` split into `shift_mlp, scale_mlp, gate_mlp`).
  - `final_layer: FinalLayer(256, 99)`: `norm_final` (non-affine `LayerNorm`, `eps=1e-6`), `adaLN_modulation` (`silu -> Linear(256, 512)` split into `shift, scale`), `linear: Linear(256, 99)`.
- **`EDM` Interpolant & Sampler (`edm_interpolant.py:10-217` [`V`])**:
  - Parameters: `sigma_data = 1.0`, `sigma_min = 0.002`, `sigma_max = 80.0`, `rho = 7.0`, `s_churn = 50.0`, `s_noise = 1.003`.
  - Preconditioned denoiser $D_\theta(x_t, t) = c_{\text{skip}}(\sigma) x_t + c_{\text{out}}(\sigma) F_\theta(c_{\text{in}}(\sigma) x_t, c_{\text{noise}}(\sigma))$ (`edm_interpolant.py:160-198` [`V`]).
  - Time grid `t = jnp.linspace(0.0, 1.0, num_timesteps + 1)` integrated via `jax.lax.scan` over `euler_step` (`edm_interpolant.py:92-158` [`V`]).

---

## Part 3: Adversarial Spec Challenge (Challenger vs. Defender)

| # | Challenger Attack (Concrete Technical Hazard) | Defender Resolution & Spec Mandate |
|---|---|---|
| **H-1** | **Versioned Checkpoint Shape & Prefix Split (`model_version=0` vs `model_version=1`)**: `caliby.ckpt` and `soluble_caliby.ckpt` have `edge_embedding.weight` of shape `(128, 8481)` under prefix `_orig_mod.denoiser.atom_mpnn.*`, whereas `soluble_caliby_v1.ckpt` has shape `(128, 1057)` under prefix `denoiser.mpnn.*`. A single static `edge_in` will crash weight conversion. | `CalibyTokenFeatures` parameterizes `model_version: int = eqx.field(static=True)` (`0` $\rightarrow$ `edge_in = 8481`; `1` $\rightarrow$ `edge_in = 1057`, `atom_mpnn.py:180-185` [`V`]). `convert.py` runs `repair_state_dict` (`_orig_mod.` strip, `checkpoint_utils.py:139-156` [`V`]) and `migrate_legacy_state_dict` (`denoiser.atom_mpnn.` $\rightarrow$ `denoiser.mpnn.`, `checkpoint_utils.py:20-39` [`V`]) and inspects `cfg.model.denoiser.get("model_version", 0)` (verified against `edge_embedding.weight.shape[1]`) before instantiating `CalibyModel` (FR-2, FR-8). |
| **H-2** | **Silent Alphabet Permutation Corruption**: Upstream `const.STANDARD_AA` (`const.py:28-37` [`V`]) sorts amino acids alphabetically by **3-letter code** (`ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL` $\rightarrow$ `ARNDCQEGHILKMFPSTWYV`), whereas `aminx` (`specs.py:22` [`V`]) sorts by **1-letter code** (`ACDEFGHIKLMNPQRSTVWYX`). Passing `aminx` indices into `W_s` or returning `GraphPotts` $(h, J)$ without permutation scrambles 16 of 20 amino acids! | `src/aminx/families/caliby/potts.py` defines explicit constant permutation tables `AMINX21_TO_CALIBY21_IDXS` and `CALIBY21_TO_AMINX21_IDXS`. Inside `CalibyModel`, internal MPNN and `GraphPotts` layers operate in upstream Caliby token order so converted weights are unmodified, and `CalibyDriver` / `score_ensemble` permute input conditioning sequences via `AMINX21_TO_CALIBY21_IDXS` and output $(h, J)$ fields/couplings via `CALIBY21_TO_AMINX21_IDXS` before returning `PottsParams` to `aminx` (FR-4). |
| **H-3** | **L-DRV Static AST Linter Failures (`tests/lint/test_l_drv.py`)**: Upstream `TokenFeatures` uses nested Python loops `for i in range(23): for j in range(23):` (`atom_mpnn.py:209-211` [`V`]) and `for i in range(5): ... for j in range(10):` (`atom_mpnn.py:293-301` [`V`]). In `aminx`, `test_l_drv.py` R2 (`test_l_drv.py:76-98, 307-326` [`V`]) bans `for` loops over `range(N)` with `N > 4` inside `eqx.Module.__call__`, R1 bans `vmap`, R3 requires allowlist entries for `lax.scan`, and R5 bans `import aminx.potts` inside `src/aminx/families/`. | 1. All pairwise RBF distances in `CalibyTokenFeatures` (`v0` $23\times 23$ and `v1` $5\times 5 + 4\times 10$) are computed via 5D array broadcasting `(N, K, A_1, A_2, 3)` with zero Python loops (FR-2).<br>2. Layer iteration in `CalibyMPNN` and `CalibySidechainMLP` iterates directly over `self.encoder_layers`, `self.decoder_layers`, and `self.blocks` (permitted by R2, `test_l_drv.py:85-87` [`V`]).<br>3. The `lax.scan` in `sidechain_diffusion.py::edm_sample` is registered in `tests/lint/l_drv_allowlist.toml` with `reason = "sequential_dependency"` (R3).<br>4. Zero imports from `aminx.potts` in `src/aminx/families/caliby/` (R5). |
| **H-4** | **`CalibyMPNN` Architectural Divergences from `ProteinMPNN`**: (a) Upstream initializes `h_V = W_s(restype_masked)` BEFORE the encoder (`atom_mpnn.py:107, 113` [`V`]), not zeros; (b) `W_e` and `W_s` have `bias=False` (`atom_mpnn.py:38, 41` [`V`]); (c) Decoder runs all-at-once without causal masks and updates edges (`W11, W12, W13, norm3`), EXCEPT `decoder_layers[2]` when `task == "scn_pack"` (`atom_mpnn.py:54-65` [`V`]); (d) `caliby_distill` (`task == "student-seq_des"`) has no `W_out` (`atom_mpnn.py:88-89` [`V`]). | `CalibyMPNN` is implemented as a dedicated Equinox module in `src/aminx/families/caliby/model.py` (never reusing `src/aminx/families/mpnn/`), with `CalibyDecLayer(update_edges=...)` controlled per layer by `task` and `W_out: eqx.nn.Linear | None` set to `None` when `task == "student-seq_des"` (FR-3). |
| **H-5** | **Ensemble Potts Sparse-to-Dense Graph Mismatch (`_aggregate_potts_params`)**: Different conformers $c \in \{1,\dots,N_{\text{conf}}\}$ have different $C_\alpha$ coordinates and therefore **different KNN neighbor index matrices `E_idx_c` (`(N, 30)`)**! Naively averaging sparse `J_c` arrays of shape `(N, 30, C, C)` averages couplings between completely different residue pairs $(i, j)$ across conformers. | `aggregate_ensemble_potts_params` (`atom_mpnn_denoiser.py:451-516` [`V`]) scatters each conformer's sparse `(N, K_potts, C, C)` tensor into a dense `(N, N, C, C)` matrix via `.at[jnp.arange(N)[:, None], E_idx_c].add(J_c * mask_ij_c[..., None, None])`, computes intersection node mask `mask_i_ens = (node_counts == N_conf)` and union edge mask `mask_ij_ens = (edge_counts > 0) * mask_i_ens[:, None] * mask_i_ens[None, :]`, divides by `N_conf`, and returns dense `E_idx_ens = jnp.broadcast_to(jnp.arange(N)[None, :], (N, N))` (FR-5). |
| **H-6** | **KNN Neighbor Truncation & Symmetrization Order**: `CalibyMPNN` runs encoder/decoder layers at `k_neighbors = 48`, then truncates `h_ESV, E_idx, token_mask_2d` to `k_neighbors_potts = 30` BEFORE calling `decoder_S_potts` (`atom_mpnn.py:134-138` [`V`]), and `GraphPotts._symmetrize_J` zeros out any directed edge $(i, j)$ whose reverse edge $(j, i)$ is not within $j$'s top-30 neighbors (`potts.py:247-248, 276-288` [`V`]). | `CalibyMPNN` slices `h_ESV[:, :self.k_neighbors_potts]`, `E_idx[:, :self.k_neighbors_potts]`, and `token_mask_2d[:, :self.k_neighbors_potts]` prior to `decoder_S_potts`, and `CalibyGraphPotts._symmetrize_J` computes the exact reciprocal-edge indicator `mask_ji` on the truncated `(N, 30)` neighbor graph before multiplying `0.5 * mask_ji * (J_ij + J_ji.swapaxes(-1, -2))` (FR-4). |
| **H-7** | **Sidechain Diffusion Local Frame & Non-Affine LayerNorm**: In `scn_diffusion_mlp.py:285-293` [`V`], `rigid_from_3_points` is called with `(x1=C, x2=CA, x3=N)` (reverse of standard `(N, CA, C)` order!), ghost atoms (`atom37_mask == 0`) are set to `0.0` in local space and `CA` in global space (`scn_diffusion_mlp.py:317-331` [`V`]), and `DenoisingMLPBlock` / `FinalLayer` use `LayerNorm(256, elementwise_affine=False, eps=1e-6)` and `GELU(approximate="tanh")` (`dit_utils.py:48-50, 113` [`V`]). | `sidechain_diffusion.py` passes `(C, CA, N)` into `rigid_from_3_points`, masks absent Atom37 slots to `0.0` in local frame (`CA` in global frame), uses `eqx.nn.LayerNorm(256, eps=1e-6, use_weight=False, use_bias=False)`, and uses `jax.nn.gelu(x, approximate=True)` (FR-6). |

---

## Part 4: Host Integration & Registration (FR-8, FR-9)

1. **`src/aminx/run/options.py` & `src/aminx/run/specs.py`**:
   - Add `"caliby"` to `FamilyName` literal in `src/aminx/run/specs.py` (`specs.py:28-36` [`V`]).
   - Define `CalibyOptions( FrozenDataclass )` in `src/aminx/run/options.py` with fields:
     - `checkpoint: Literal["caliby", "soluble_caliby", "soluble_caliby_v1", "caliby_distill", "soluble_caliby_distill", "caliby_packer_000", "caliby_packer_010", "caliby_packer_030"] = "soluble_caliby_v1"`
     - `ensemble_size: int = 1` (number of conformers; if `> 1` and a single structure is provided, generates $N_{\text{conf}}$ Gaussian-perturbed conformers with `ensemble_noise_std`)
     - `ensemble_noise_std: float = 0.2` (coordinate noise standard deviation in $\text{\AA}$ when synthesizing a conformer ensemble from a single structure)
     - `pack_sidechains: bool = False` (whether to run `caliby_packer_*` sidechain EDM diffusion after sequence design)
     - `packer_checkpoint: Literal["caliby_packer_000", "caliby_packer_010", "caliby_packer_030"] = "caliby_packer_010"`
     - `packer_num_steps: int = 100`
2. **`src/aminx/families/caliby/driver.py`**:
   - Implement `CalibyDriver(FamilyDriver)` satisfying the 5-method `FamilyDriver` protocol (`family_driver.py:57-149` [`V`]):
     - `capabilities`: `FamilyCapabilities(family="caliby", outputs_potts=True, outputs_packed_sidechains=True, supports_ensemble=True)`
     - `featurize(structure, spec) -> CalibyInputs`
     - `forward(model, inputs, key) -> CalibyRawOutputs` (evaluates each conformer through `CalibyModel.encode_conformer` and aggregates $(h, J)$ via `aggregate_ensemble_potts_params`)
     - `to_potts_or_scores(raw_outputs, inputs, spec)` (permutes 21-state AA+X fields/couplings from Caliby 3-letter alphabetical order to `aminx` 1-letter order via `CALIBY21_TO_AMINX21_IDXS`)

---

## Acceptance

- [FR-1] `rg -n "2025.09.30.679633" .praxia/docs/research/260605_caliby-nature-and-dataset.md scripts/recapture/caliby_recapture.py`
- [FR-2] `uv run pytest tests/families/test_caliby_parity.py -k "test_token_features"`
- [FR-3] `uv run pytest tests/families/test_caliby_parity.py -k "test_caliby_mpnn"`
- [FR-4] `uv run pytest tests/families/test_caliby_parity.py -k "test_graph_potts"`
- [FR-5] `uv run pytest tests/families/test_caliby_parity.py -k "test_ensemble_aggregation"`
- [FR-6] `uv run pytest tests/families/test_caliby_parity.py -k "test_sidechain_diffusion"`
- [FR-7] `uv run pytest tests/families/test_caliby_parity.py -k "test_caliby_model_end_to_end"`
- [FR-8] `uv run pytest tests/families/test_caliby_parity.py -k "test_checkpoint_conversion"`
- [FR-9] `uv run pytest tests/lint/test_l_drv.py`
- [FR-10] `uv run pytest tests/families/test_caliby_parity.py`

## Assumptions

- **A-1**: All 8 official `ProteinDesignLab/caliby` checkpoints (`doi:10.5281/zenodo.18644985`) use `use_multichain_encoding = False`, `parameterization = "score"`, `symmetric_J = True`, `num_factors = 32`, `hidden_dim = 128`, and `k_neighbors = 48` (`atom_mpnn.py:24-95`, `potts.py:40-124` [`V`]).
- **A-2**: Upstream `atomworks.constants.STANDARD_AA` is fixed to the 20 standard amino acids sorted alphabetically by 3-letter CCD code (`ALA` through `VAL`, `const.py:28-37` [`V`]).
- **A-3**: `src/aminx/potts/calibration.py` (`calibration.py:1-159` [`V`]) remains in place as general-purpose tree-reweighted Potts marginal calibration infrastructure and requires no API changes.

## Prior art

- **Upstream `ProteinDesignLab/caliby` (Shuai, Sidhu, & Huang, *bioRxiv* 2025.09.30.679633)**: Reference PyTorch Lightning implementation of `CalibyMPNN` (`atom_mpnn.py:24-158` [`V`]), `TokenFeatures` (`atom_mpnn.py:161-303` [`V`]), `GraphPotts` (`potts.py:40-289` [`V`]), `_aggregate_potts_params` (`atom_mpnn_denoiser.py:451-516` [`V`]), and `SidechainDiffusionModule` (`scn_diffusion_mlp.py:21-333` [`V`]).
- **`src/aminx/families/potts_mpnn/` (`driver.py:1-165`, `model.py:1-210` [`V`])**: Existing Potts-outputting MPNN family in `aminx`; demonstrates how a family returns Potts $(h, J)$ parameters to the host runner while complying with `tests/lint/test_l_drv.py` (`R1`–`R5`).
- **`src/aminx/families/thermo_mpnn/convert.py` (`convert.py:1-140` [`V`])**: Reference PyTorch Lightning `.ckpt` $\rightarrow$ `.eqx.zst` converter pattern in `aminx`.
- **`.praxia/docs/research/260605_caliby-nature-and-dataset.md` (`260605_caliby-nature-and-dataset.md:1-178` [`V`])**: The historical research note containing the `#5595` misattribution corrected by FR-1 of this specification.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dense $(N, N, 22, 22)$ ensemble Potts coupling tensor memory scaling for very large complexes ($N > 1500$) | Medium | Medium | At $N = 1000$, a `float32` tensor of shape `(1000, 1000, 22, 22)` uses $1.93\text{ GB}$, matching upstream `_aggregate_potts_params` (`atom_mpnn_denoiser.py:491` [`V`]). For $N = 1$ conformer (`ensemble_size == 1`), `score_ensemble` bypasses dense $(N, N)$ expansion and keeps the sparse $(N, 30, 22, 22)$ graph directly. |
| Tie-breaking discrepancies in $C_\alpha$ KNN `top_k` when two atoms have identical distances | Low | Low | In `test_caliby_parity.py`, test coordinates use realistic protein backbones with non-degenerate pairwise $C_\alpha$ distances, and layer parity tests also verify exact equivalence when `E_idx` is passed directly. |
| Sidechain packer checkpoint (`caliby_packer_*`) invoked on sequence design or student checkpoint (`caliby_distill`) invoked with sidechain conditioning | Medium | High | `CalibyDriver` and `CalibyModel` validate checkpoint `task` metadata at load/call time: `task == "student-seq_des"` raises `ValueError` if sidechain conditioning is requested (`atom_mpnn_denoiser.py:321-336` [`V`]), and `pack_sidechains` requires a `task == "scn_pack"` checkpoint. |
