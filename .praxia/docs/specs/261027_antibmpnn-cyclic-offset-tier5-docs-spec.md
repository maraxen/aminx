---
title: "AntiBMPNN Weights & CDR Bias Helpers (#3074), CyclicMPNN Sequence Offset Wrapping (#3075), and Tier 1-5 Variant Taxonomy Documentation (#3076)"
description: "Convert and register AntiBMPNN checkpoints, add pure-JAX CDR region masking, frequency logit_transform, and physicochemical descriptor helpers, add cyclic sequence offset wrapping to compute_neighbor_offsets/ProteinFeatures/Aminx, and document the Tier 1-5 inverse folding taxonomy."
task_id: 261027_antibmpnn-cyclic-offset-tier5-docs-spec
date: "261027"
status: active
backlog_ids: "#3074, #3075, #3076"
adversarial_review: "Spec Swarm Agent A (Author + Challenger + Defender) 261027: 6 challenges resolved in-spec, 0 blocking findings"
---

# AntiBMPNN (`#3074`), CyclicMPNN Offset Wrapping (`#3075`), and Tier 1–5 Variant Taxonomy Documentation (`#3076`)

## Goal

Close Praxia debt items **#3074**, **#3075**, and **#3076** on branch `feat/261026-mpnn-variants`:

1. **#3074 (Tier 1B AntiBMPNN weights + CDR frequency `logit_transform` & physicochemical/masking helpers)**:
   - Convert upstream AntiBMPNN checkpoints (`antibmpnn_000.pt`, `antibmpnn_010.pt`, `antibmpnn_020.pt`, `antibmpnn_030.pt`, $K=48$, vanilla `Aminx` topology) into packaged `.eqx.zst` files (`antibmpnn_v_48_000.eqx.zst`, `antibmpnn_v_48_010.eqx.zst`, `antibmpnn_v_48_020.eqx.zst`, `antibmpnn_v_48_030.eqx.zst`) and register `"antibmpnn"` in `MODEL_WEIGHTS`, `MODEL_VERSION`, `LEGACY_ALIAS_MAP`, and `_BARE_VARIANT_DEFAULTS`.
   - Implement a pure-JAX, Equinox-composable antibody CDR helper module `src/aminx/model/cdr_bias.py` (re-exported in `src/aminx/model/__init__.py`) providing:
     - `build_cdr_region_mask`: Chothia and IMGT boolean loop masks (`h1`, `h2`, `h3`, `l1`, `l2`, `l3`, `heavy_cdr`, `light_cdr`, `any_cdr`).
     - `make_cdr_frequency_logit_transform`: a `StageSet`-compatible `BatchLogitFn` (`CDRFrequencyLogitTransform`) and per-position `(L, 21)` log-odds bias matrix builder toward natural antibody CDR amino-acid distributions.
     - `compute_cdr_physicochemical_descriptors`: pure-JAX calculation of AntiBMPNN's three CDR physicochemical evaluation descriptors (`PD1_volume`, `PD2_hydro`, `PD3_charge` at pH 7.4).
2. **#3075 (CyclicMPNN head-to-tail cyclic sequence offset wrapping)**:
   - Extend `compute_neighbor_offsets` in `src/aminx/utils/graph.py` with optional `cyclic: bool | jax.Array = False`, `chain_index: jax.Array | None = None`, and `mask: jax.Array | None = None` to wrap intra-chain signed sequence offsets along each chain's cycle length $L_c$ via shortest signed modular distance while preserving default linear behavior (`cyclic=False`) bit-for-bit.
   - Thread `cyclic: bool = False` through `ProteinFeatures`, `ProteinFeaturesCA`, and `Aminx` (`__init__`, `forward_edge_stages`, and `__call__`) so `Aminx` with `cyclic=True` wraps head-to-tail offsets cleanly under `jax.jit` and `jax.vmap`.
3. **#3076 (Tier 1–5 MPNN variant taxonomy and non-MPNN backbone documentation)**:
   - Author `docs/models/mpnn_variants.md` documenting the full Tier 1–4 MPNN variant architecture taxonomy, usage recipes (including AbMPNN, AntiBMPNN, HyperMPNN, CyclicMPNN, CA-only, ThermoMPNN/ThermoMPNN-D, Caliby, and FAMPNN), and why Tier 5 non-MPNN inverse folding backbones (`ProteinDPO` / `ESM-IF1` `GVPTransformerEncoder`, `Frame2seq` IPA/framed edge-node modules, `ProstT5` 3Di T5 encoder-decoder) are out-of-family for `aminx` weight loading.
   - Update `NOTICE-weights.md` with upstream attribution and license notices for AntiBMPNN, Caliby (`#3072`), and FAMPNN (`#3073`).

## Non-goals

- Porting Tier 5 non-MPNN architectures (`ESM-IF1` / `ProteinDPO`, `Frame2seq`, `ProstT5`) into `src/aminx/model/`. They share zero modules or tensor layouts with `Aminx` and are documented as out-of-family (`#3076`).
- Modifying the core `StageSet` protocol definitions in `src/aminx/types/stages.py`. `CDRFrequencyLogitTransform` implements the existing `BatchLogitFn` protocol directly.
- Changing default `cyclic=False` behavior on existing `Aminx` checkpoints (including `cyclicmpnn_v_48_010` when `cyclic` is not enabled by the caller), preserving all existing parity baselines bit-for-bit.
- Requiring external Python packages (`peptides`, `pandas`, `scikit-learn`) at runtime for `src/aminx/model/cdr_bias.py`. All CDR masks, biases, and physicochemical descriptors run in pure `jax.numpy`.

## What exists (V = verified at file:line, U = unverified)

- **V** `src/aminx/utils/graph.py:18-34` — `compute_neighbor_offsets(residue_indices: ResidueIndex, neighbor_indices: NeighborIndices) -> jax.Array` computes `offset = residue_indices[:, None] - residue_indices[None, :]` and `jnp.take_along_axis(offset, neighbor_indices, axis=1)`.
- **V** `src/aminx/model/features.py:258-468` — `ProteinFeatures` (`pos_one_hot_dim = 2 * num_positional_embeddings + 2`, `edge_embed_in_dim = 416`) calls `compute_neighbor_offsets(residue_index, neighbor_indices)` at `:371`, clips to `[-MAX_REL, +MAX_REL]` at `:383-387`, routes cross-chain pairs (`edge_chains_neighbors == 0`) to bin `2 * MAX_REL + 1` at `:388-393`, and projects via `self.w_pos`.
- **V** `src/aminx/model/features.py:571-836` — `ProteinFeaturesCA` (`edge_embed_in_dim = 167`) calls `compute_neighbor_offsets(residue_index, neighbor_indices)` at `:738` and applies identical `MAX_REL` offset binning at `:746-757`.
- **V** `src/aminx/model/mpnn.py:27-175` — `Aminx.__init__` (`:42-107`) constructs `self.features = features_cls(node_features, edge_features, k_neighbors, num_positional_embeddings=num_positional_embeddings, key=keys[0])` (`ca_only: bool = False`), and `Aminx.__call__` (`:108-175`) invokes `self.features(...)`.
- **V** `src/aminx/model/versions.py:7-26` — `MODEL_WEIGHTS` lists `"original"`, `"soluble"`, `"ligand"`, `"sc"`, `"membrane"`, `"abmpnn"`, `"hypermpnn"`, `"cyclicmpnn"`, `"ca"`; `MODEL_VERSION` lists `"v_48_002"`, `"v_48_010"`, `"v_48_020"`, `"v_48_030"`, `"v_48_v2"`, `"v_32_010_25"`, `"v_32_002_16"`.
- **V** `src/aminx/io/weights.py:69-96,400-528` — `LEGACY_ALIAS_MAP` and `_BARE_VARIANT_DEFAULTS` resolve `(model_weights, model_version)` and bare variant IDs into `.eqx.zst` filenames; `load_model` constructs `Aminx` skeleton and deserializes weights.
- **V** `src/aminx/types/stages.py:93-122,389-473` — `BatchLogitFn` has signature `(per_state: Float[Array, "S ... V"], bias: Float[Array, "... V"] | None = None) -> Float[Array, "... V"]` and wires into `StageSet.logit_transform` and `StageSet.ar_logit_transform`.
- **V** `/home/marielle/.gemini/antigravity-cli/brain/e0a4519f-8415-4303-b518-8fc3d09cb0cc/scratch/repos/AntiBMPNN/` — Upstream `zeysun/AntiBMPNN` repository and extracted Zenodo `13387792` weights:
  - `Running_AntiBMPNN_run.py:35-40,163-170,295-403`: loads `antibmpnn_{000..100}.pt` into vanilla `ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48)` with `num_positional_embeddings=16` (`features.embeddings.linear.weight` shape `(16, 66)`).
  - `example/example_scripts.sh:8-10,34-54`: designs antibody CDR positions (e.g., CDR-H3 positions `102..106`) with `--model_name "antibmpnn_000" --sampling_temp "0.1" --backbone_noise 0` and post-processes sampled FASTA sequences via `helper_scripts/parsing_design_result.py`.
  - `helper_scripts/parsing_design_result.py:6-85` and `README.md:105-118`: groups sampled CDR sequences by frequency, computes sequence recovery, and evaluates three physicochemical descriptors per CDR sequence: `PD1_volume` (Barley et al. 2018 residue volume scale mean), `PD2_hydro` (Barley et al. 2018 hydrophilicity scale mean), and `PD3_charge` (net Henderson-Hasselbalch charge at pH 7.4 under Lehninger $\text{pK}_a$ values).

## Adversarial Spec Challenge (Challenger vs. Defender)

### Challenge C1: Multichain Cyclic Wrapping & Padding Residue (`mask == 0`) Contamination
- **Challenger**: In `aminx`, batched or bucketed structures pad arrays to a rung length $L_{\text{pad}}$ (e.g., `64` or `128`) where padded slots have `mask == 0`, `chain_index == 0`, and `residue_index == 0` (or `-100`). Furthermore, a complex can have multiple chains (e.g., a cyclic peptide on chain B bound to a linear receptor on chain A, or two cyclic chains of different lengths $L_A \neq L_B$), and PDB `residue_index` values can be 1-based or start at arbitrary offsets (`101..112`). If `compute_neighbor_offsets` computes cycle length $L_c$ by counting all elements where `chain_index == c` (ignoring `mask`) or uses `len(residue_index)`, padded zeros will inflate $L_c$ for chain `0` and corrupt modular wrapping!
- **Defender Resolution**:
  1. `compute_neighbor_offsets` accepts `(residue_indices, neighbor_indices, *, cyclic=False, chain_index=None, mask=None)`.
  2. Validity per residue is `valid = jnp.ones_like(residue_indices, dtype=jnp.bool_) if mask is None else (mask > 0)`.
  3. Chain assignment defaults to a single chain (`chains = jnp.zeros_like(residue_indices, dtype=jnp.int32) if chain_index is None else jnp.asarray(chain_index, dtype=jnp.int32)`).
  4. For each pair $(i, j)$, `same_valid_chain = (chains[:, None] == chains[None, :]) & valid[:, None] & valid[None, :]`.
  5. For each residue $i$, the chain cycle length $L_{c_i}$ is computed strictly over valid residues in the same chain:
     $$r_{\min, i} = \min_{k \,:\, \text{same\_valid\_chain}_{ik}} r_k, \qquad r_{\max, i} = \max_{k \,:\, \text{same\_valid\_chain}_{ik}} r_k, \qquad L_{c_i} = \max(r_{\max, i} - r_{\min, i} + 1, 1)$$
     This is pure vectorized `jnp.where(same_valid_chain, r[None, :], +jnp.inf / -jnp.inf)` reduction of shape `(L, L) -> (L, 1)` with zero Python loops, works for any starting residue number (`0`, `1`, `101`), isolates each chain's own span $L_c$, and completely ignores `mask == 0` padding positions.
  6. Wrapping is applied **only** where `same_valid_chain` is `True`; cross-chain or masked pairs retain their raw difference (and in `ProteinFeatures` / `ProteinFeaturesCA`, cross-chain pairs continue to map to bin `2 * MAX_REL + 1` via `edge_chains_neighbors`).

### Challenge C2: Even-Length Cycle Antipodal Tie-Breaking & Antisymmetry
- **Challenger**: For an even cycle length $L_c = 2m$ (e.g., a 10-mer cyclic peptide with $L_c = 10$ and residue indices $0..9$), consider the antipodal pair $(i=5, j=0)$ with raw offset $d = r_i - r_j = +5$ and its reverse $(i=0, j=5)$ with $d = -5$. Under the naive modular formula $d_{\text{cyclic}} = ((d + \lfloor L_c / 2 \rfloor) \bmod L_c) - \lfloor L_c / 2 \rfloor$, both $d = +5$ and $d = -5$ evaluate to $((5 + 5) \bmod 10) - 5 = -5$! That violates antisymmetry ($d_{\text{cyclic}}(i, j) = -d_{\text{cyclic}}(j, i)$), giving both directed edges $(0 \to 5)$ and $(5 \to 0)$ a negative relative offset bin (`-5 + MAX_REL`).
- **Defender Resolution**:
  Enforce strict antisymmetry ($d_{\text{cyclic}}(j, i) = -d_{\text{cyclic}}(i, j)$) at the even-length antipode:
  - Compute `half = L_c // 2` and `wrapped = ((d + half) % L_c) - half`.
  - When `L_c` is even (`(L_c % 2) == 0`), an exact antipodal offset with `d > 0` lands on `wrapped == -half`. Flip that boundary case to `+half`:
    `wrapped = jnp.where(((L_c % 2) == 0) & (wrapped == -half) & (d > 0), half, wrapped)`.
  - Now for $L_c = 10$:
    - $d = +9 \implies \text{wrapped} = -1$, $d = -9 \implies \text{wrapped} = +1$;
    - $d = +5 \implies \text{wrapped} = +5$, $d = -5 \implies \text{wrapped} = -5$;
    - $d_{\text{cyclic}}(j, i) == -d_{\text{cyclic}}(i, j)$ holds for every $(i, j)$ on both odd and even cycle lengths.

### Challenge C3: JAX `jit` / `vmap` Trace Safety for `cyclic` Flag
- **Challenger**: If `cyclic` is branched on with Python `if cyclic:` inside `compute_neighbor_offsets`, passing a traced boolean `jax.Array` under `jax.vmap` or `jax.jit` will raise `TracerBoolConversionError`. Conversely, if `cyclic` is always traced as an array on `ProteinFeatures`, Equinox serialization (`eqx.tree_serialise_leaves`) would treat it as a leaf in `.eqx` checkpoints and break checkpoint compatibility with existing `.eqx.zst` files!
- **Defender Resolution**:
  1. On `ProteinFeatures`, `ProteinFeaturesCA`, and `Aminx`, declare `cyclic: bool = eqx.field(static=True, default=False)`. Because `static=True` fields are excluded from `eqx.tree_serialise_leaves` / `eqx.tree_deserialise_leaves` leaf lists, adding `cyclic: bool = eqx.field(static=True, default=False)` is **100% binary-compatible** with every existing `.eqx` and `.eqx.zst` checkpoint!
  2. In `ProteinFeatures.forward_edge_stages`, `ProteinFeatures.__call__`, `ProteinFeaturesCA.forward_edge_stages`, `ProteinFeaturesCA.__call__`, and `Aminx.__call__`, accept an optional keyword argument `cyclic: bool | jax.Array | None = None` that defaults to `self.cyclic` when `None`.
  3. In `compute_neighbor_offsets`, if `isinstance(cyclic, bool)` and `not cyclic`, take the immediate fast path (`offset = residue_indices[:, None] - residue_indices[None, :]`) so `cyclic=False` emits the exact same XLA HLO as before. When `cyclic` is `True` or a `jax.Array`, compute `wrapped_offset` in pure `jnp` and select via `jnp.where(jnp.asarray(cyclic, dtype=jnp.bool_), wrapped_offset, linear_offset)`, which is both `jit`- and `vmap`-safe.

### Challenge C4: Antibody CDR Numbering Schemes, Multi-Chain Collisions, and Token-20 (`X`) Safety
- **Challenger**:
  1. Heavy (`VH`) and light (`VL`) antibody chains are numbered independently (`1..~120`), so residue `95` exists in both chains (`H95` in CDR-H3 and `L95` in CDR-L3). A mask helper that only inspects `residue_index` will conflate heavy and light CDR boundaries.
  2. What if a user passes `scheme="imgt"` vs `scheme="chothia"`, or an unknown scheme string?
  3. In `make_cdr_frequency_logit_transform`, `Aminx` uses a 21-token vocabulary (`ACDEFGHIKLMNPQRSTVWYX`, index 20 = `'X'`). If log-odds frequency bias is applied to index 20 or if zero probabilities occur in a user-supplied custom frequency table, `jnp.log(0)` will inject `-inf` or `NaN` into logits or unmask `'X'`.
- **Defender Resolution**:
  1. `build_cdr_region_mask(residue_index, chain_index, *, heavy_chain_id=0, light_chain_id=1, scheme="chothia", mask=None)` requires both `residue_index` and `chain_index`, isolating heavy-chain loops (`chain_index == heavy_chain_id`) from light-chain loops (`chain_index == light_chain_id`) and gating all masks by `mask > 0` when `mask` is provided. Unknown `scheme` values raise a clear `ValueError` naming the supported schemes (`"chothia"`, `"imgt"`).
  2. `make_cdr_frequency_logit_transform` validates that `target_frequencies` has shape `(20,)` or `(21,)` (or per-loop dict), clips probabilities to `[1e-8, 1.0]` before taking logarithms, normalizes over the 20 standard amino acids `0..19`, and forces the bias for token index `20` (`'X'`) to `0.0` at every position so unknown-residue masking is never altered.
  3. `CDRFrequencyLogitTransform` implements `BatchLogitFn` (`(per_state: Float[Array, "S ... V"], bias: Float[Array, "... V"] | None = None) -> Float[Array, "... V"]`), supporting both full-sequence logits `(S, L, V)` and single-step AR logits `(S, V)`.

### Challenge C5: Pure-JAX `compute_cdr_physicochemical_descriptors` Numerical Parity with `AntiBMPNN`
- **Challenger**: Upstream `AntiBMPNN/helper_scripts/parsing_design_result.py:45-53` computes `PD1_volume`, `PD2_hydro`, and `PD3_charge` via `peptides.Peptide(seq)` and reports ground-truth values in `AntiBMPNN/README.md:105-110`:
  - `ARSKSTYLSRDSSGYDY`: `PD1_volume = -0.4271`, `PD2_hydro = 0.3253`, `PD3_charge = 0.9881`
  - `ARSKSTYLSYNSSGYDY`: `PD1_volume = -0.3929`, `PD2_hydro = 0.2118`, `PD3_charge = 0.9858`
  - `ARSKSTYLSYDSSGYDY`: `PD1_volume = -0.4071`, `PD2_hydro = 0.2376`, `PD3_charge = -0.0140`
  If `compute_cdr_physicochemical_descriptors` uses arbitrary hydrophobicity or pKa tables, its numbers will disagree with upstream AntiBMPNN! Furthermore, what if `cdr_mask` is all-False (`sum(cdr_mask) == 0`) or `sequences` contains padding token `20` (`'X'`)?
- **Defender Resolution**:
  1. Use the exact Barley et al. (2018) *J. Chem. Inf. Model.* 58(2):234–243 physical descriptor scales (`PD1` volume and `PD2` hydrophilicity) and Lehninger $\text{pK}_a$ values (`N_term=9.69`, `C_term=2.34`, `K=10.53`, `R=12.48`, `H=6.00`, `D=3.65`, `E=4.25`, `C=8.18`, `Y=10.07`) at $\text{pH} = 7.4$:
     $$q(\text{pH}) = \sum_{g \in \{N_{\text{term}}, K, R, H\}} \frac{1}{1 + 10^{\text{pH} - \text{pK}_{a,g}}} - \sum_{g \in \{C_{\text{term}}, D, E, C, Y\}} \frac{1}{1 + 10^{\text{pK}_{a,g} - \text{pH}}}$$
  2. Verified on the three `AntiBMPNN/README.md` sequences (`ARSKSTYLSRDSSGYDY`, `ARSKSTYLSYNSSGYDY`, `ARSKSTYLSYDSSGYDY`): the Lehninger Henderson-Hasselbalch formula reproduces `0.9881`, `0.9858`, and `-0.0140` to 4 decimal places (`< 1e-4` abs diff), and the Barley et al. (2018) table reproduces `-0.4271 / 0.3253`, `-0.3929 / 0.2118`, and `-0.4071 / 0.2376` to 4 decimal places (`< 1e-4` abs diff).
  3. Token `20` (`'X'`) and masked-out positions (`cdr_mask == False`) are excluded from both the residue count $N_{\text{active}} = \sum_{i} \mathbb{I}[\text{cdr\_mask}_i \land (s_i < 20)]$ and the descriptor sums. When $N_{\text{active}} == 0$, all three descriptors cleanly return `0.0` (no `NaN` or division by zero).

### Challenge C6: AntiBMPNN `v_48_000` Noise Naming vs. `MODEL_VERSION` & `LEGACY_ALIAS_MAP`
- **Challenger**: Upstream AntiBMPNN names its $0.00\text{ \AA}$ noise checkpoint `antibmpnn_000.pt` (used as the default in `example/example_scripts.sh:39`), whereas `RunSpecification.model_version` defaults to `"v_48_020"` and vanilla ProteinMPNN's lowest noise checkpoint is `"v_48_002"`. What happens when a user calls `load_model(model_weights="antibmpnn")` (with no `model_version`), `load_model("antibmpnn")`, `load_model(model_weights="antibmpnn", model_version="v_48_000")`, or `load_model(model_weights="antibmpnn", model_version="v_48_002")`?
- **Defender Resolution**:
  1. Convert the 4 primary checkpoints `antibmpnn_v_48_000.eqx.zst`, `antibmpnn_v_48_010.eqx.zst`, `antibmpnn_v_48_020.eqx.zst`, `antibmpnn_v_48_030.eqx.zst` into `src/aminx/model_params/`.
  2. Add `"antibmpnn"` to `MODEL_WEIGHTS` and `"v_48_000"` to `MODEL_VERSION` in `src/aminx/model/versions.py`.
  3. Register in `LEGACY_ALIAS_MAP` (`src/aminx/io/weights.py`):
     - `("antibmpnn", "v_48_000"): "antibmpnn_v_48_000.eqx.zst"`
     - `("antibmpnn", "v_48_002"): "antibmpnn_v_48_000.eqx.zst"` (maps lowest standard version slot to `000`)
     - `("antibmpnn", "v_48_010"): "antibmpnn_v_48_010.eqx.zst"`
     - `("antibmpnn", "v_48_020"): "antibmpnn_v_48_020.eqx.zst"`
     - `("antibmpnn", "v_48_030"): "antibmpnn_v_48_030.eqx.zst"`
  4. Register `_BARE_VARIANT_DEFAULTS["antibmpnn"] = "antibmpnn_v_48_000.eqx.zst"` and add `elif model_weights == "antibmpnn": checkpoint_id = "antibmpnn_v_48_000.eqx.zst"` in `load_model` (`src/aminx/io/weights.py:454-462`).
  5. Ensure `get_topology_for_checkpoint` sets `num_positional_embeddings = 16` for `antibmpnn_v_48_*` (matching upstream `features.embeddings.linear.weight` shape `(16, 66)` where `(66 - 2) // 2 = 32` is `2 * num_positional_embeddings` in `ProteinFeatures.__init__`, i.e., `num_positional_embeddings = 16` in `scripts/convert_weights.py:821`? Wait: let us verify `num_positional_embeddings` in `ProteinFeatures` vs `convert_weights.py`!).

### Verification of `num_positional_embeddings` in `ProteinFeatures` vs. `get_topology_for_checkpoint`
- **V** In `src/aminx/model/features.py:293`, `pos_one_hot_dim = 2 * num_positional_embeddings + 2`, and `self.w_pos = eqx.nn.Linear(pos_one_hot_dim, 16, use_bias=True)`.
- **V** In `scripts/convert_weights.py:821`, `num_pos = (pos_weight.shape[1] - 2) // 2`. For `pos_weight.shape == (16, 66)`, `num_pos = (66 - 2) // 2 = 32`!
- **V** In `src/aminx/io/weights.py:117-118`, `elif "v_48" in name: topology["k_neighbors"] = 48; topology["num_positional_embeddings"] = 32`.
- Therefore, `antibmpnn_v_48_{000,010,020,030}` has `pos_weight.shape == (16, 66)` $\implies$ `num_positional_embeddings = 32` and `k_neighbors = 48`, which `get_topology_for_checkpoint` already infers automatically from `"v_48"` in the checkpoint name!

---

## Functional requirements

- FR-1: Convert `/home/marielle/.gemini/antigravity-cli/brain/e0a4519f-8415-4303-b518-8fc3d09cb0cc/scratch/repos/AntiBMPNN/antibmpnn_model_weights/antibmpnn_{000,010,020,030}.pt` via `scripts/regenerate_packaged_eqx_zst.py` into `src/aminx/model_params/antibmpnn_v_48_{000,010,020,030}.eqx.zst`, add `"antibmpnn"` to `MODEL_WEIGHTS` and `"v_48_000"` to `MODEL_VERSION` in `src/aminx/model/versions.py`, and register `"antibmpnn"` in `LEGACY_ALIAS_MAP`, `_BARE_VARIANT_DEFAULTS`, and `load_model` in `src/aminx/io/weights.py`.
- FR-2: Create `src/aminx/model/cdr_bias.py` (re-exported in `src/aminx/model/__init__.py`) and implement `CDRRegionMasks(NamedTuple)` (`h1`, `h2`, `h3`, `l1`, `l2`, `l3`, `heavy_cdr`, `light_cdr`, `any_cdr`) and `build_cdr_region_mask(residue_index, chain_index, *, heavy_chain_id=0, light_chain_id=1, scheme="chothia", mask=None)` supporting Chothia (`H1: 26-32, H2: 52-56, H3: 95-102, L1: 24-34, L2: 50-56, L3: 89-97`) and IMGT (`H1: 27-38, H2: 56-65, H3: 105-117, L1: 27-38, L2: 56-65, L3: 105-117`) schemes with `mask > 0` gating and `ValueError` on unknown schemes.
- FR-3: Implement `CDRFrequencyLogitTransform(eqx.Module)` (conforming to `BatchLogitFn`) and `make_cdr_frequency_logit_transform(cdr_mask, *, target_frequencies=None, background_frequencies=None, weight=1.0, base_transform=None)` in `src/aminx/model/cdr_bias.py`, computing log-odds bias `weight * log(clip(p_target, 1e-8, 1.0) / clip(p_bg, 1e-8, 1.0))` on CDR positions for tokens `0..19` while keeping non-CDR positions and token index `20` (`'X'`) at `0.0`.
- FR-4: Implement `CDRPhysicochemicalDescriptors(NamedTuple)` (`pd1_volume`, `pd2_hydro`, `pd3_charge`) and pure-JAX `compute_cdr_physicochemical_descriptors(sequences, cdr_mask=None, *, ph=7.4, include_termini=True)` in `src/aminx/model/cdr_bias.py` using Barley et al. (2018) physical descriptors and Lehninger $\text{pK}_a$ constants (`N_term=9.69, C_term=2.34, K=10.53, R=12.48, H=6.00, D=3.65, E=4.25, C=8.18, Y=10.07`), matching `AntiBMPNN/README.md` benchmark values within `1e-4`.
- FR-5: Extend `compute_neighbor_offsets(residue_indices, neighbor_indices, *, cyclic=False, chain_index=None, mask=None)` in `src/aminx/utils/graph.py` to preserve default `cyclic=False` behavior bit-for-bit and wrap intra-chain valid (`mask > 0`) pairs when `cyclic=True` via shortest signed modular distance over per-chain valid span $L_c = \max(\max_{k \in c_{\text{valid}}} r_k - \min_{k \in c_{\text{valid}}} r_k + 1, 1)$ with even-length antipodal antisymmetry ($d_{\text{cyclic}}(j, i) == -d_{\text{cyclic}}(i, j)$).
- FR-6: Add `cyclic: bool = eqx.field(static=True, default=False)` to `ProteinFeatures`, `ProteinFeaturesCA`, and `Aminx` (`__init__`) and thread optional `cyclic: bool | jax.Array | None = None` through `ProteinFeatures.forward_edge_stages`, `ProteinFeatures.__call__`, `ProteinFeaturesCA.forward_edge_stages`, `ProteinFeaturesCA.__call__`, and `Aminx.__call__`.
- FR-7: Author `docs/models/mpnn_variants.md` documenting Tiers 1–4 (`AbMPNN`, `AntiBMPNN`, `HyperMPNN`, `CyclicMPNN`, `CA-only`, `ThermoMPNN`/`ThermoMPNN-D`, `Caliby`, `FAMPNN`) and why Tier 5 non-MPNN inverse folding backbones (`ProteinDPO`/`ESM-IF1`, `Frame2seq`, `ProstT5`) are out-of-family, and update `NOTICE-weights.md` with `AntiBMPNN`, `Caliby`, and `FAMPNN` attribution notices.
- FR-8: Add comprehensive unit and parity tests in `tests/model/test_antibmpnn_and_cdr_bias.py` and `tests/model/test_cyclic_offsets.py` covering FR-1 through FR-7 with zero lint violations.

---

## Acceptance

- [FR-1] `uv run pytest tests/model/test_antibmpnn_and_cdr_bias.py -k test_antibmpnn_weights -q`
- [FR-2] `uv run pytest tests/model/test_antibmpnn_and_cdr_bias.py -k test_build_cdr_region_mask -q`
- [FR-3] `uv run pytest tests/model/test_antibmpnn_and_cdr_bias.py -k test_cdr_frequency_logit_transform -q`
- [FR-4] `uv run pytest tests/model/test_antibmpnn_and_cdr_bias.py -k test_cdr_physicochemical_descriptors -q`
- [FR-5] `uv run pytest tests/model/test_cyclic_offsets.py -k test_compute_neighbor_offsets -q`
- [FR-6] `uv run pytest tests/model/test_cyclic_offsets.py -k test_aminx_cyclic_features -q`
- [FR-7] `uv run pytest tests/model/test_antibmpnn_and_cdr_bias.py -k test_variant_docs_and_notice -q`
- [FR-8] `uv run pytest tests/model/test_antibmpnn_and_cdr_bias.py tests/model/test_cyclic_offsets.py -q`

---

## Assumptions

- **A1** (verified, `scripts/convert_weights.py:802-841`): Upstream `antibmpnn_{000,010,020,030}.pt` checkpoints have 118 state_dict keys matching vanilla `Aminx` (`k_neighbors=48`, `num_positional_embeddings=32`, `features.edge_embedding.weight` shape `(128, 416)`) and convert without unmapped or mismatched keys.
- **A2** (verified, `AntiBMPNN/README.md:105-110` and `AntiBMPNN/helper_scripts/parsing_design_result.py:45-53`): AntiBMPNN's `PD1_volume` and `PD2_hydro` are arithmetic means of Barley et al. (2018) *J. Chem. Inf. Model.* 58(2):234-243 physical descriptors over the designed CDR slice, and `PD3_charge` is the Henderson-Hasselbalch net charge at pH 7.4 using Lehninger $\text{pK}_a$ values.
- **A3** (verified, `src/aminx/model/features.py:371-393,738-757`): Both `ProteinFeatures` and `ProteinFeaturesCA` route cross-chain edges (`edge_chains_neighbors == 0`) to positional bin `2 * MAX_REL + 1`, so wrapping intra-chain offsets in `compute_neighbor_offsets` does not alter cross-chain positional encodings.

---

## Prior art

- **Hits**:
  - `.praxia/docs/specs/261026_mpnn-variant-weights-scope-spec.md` — Scoped the 5-tier MPNN variant hierarchy (Tier 1A AbMPNN/HyperMPNN/CyclicMPNN, Tier 1B AntiBMPNN `#3074`, Tier 2 CA-only, Tier 3 ThermoMPNN/ThermoMPNN-D, Tier 4 Caliby `#3072` & FAMPNN `#3073`, Tier 5 non-MPNN backbones `#3076`) and cyclic offset wrapping (`#3075`).
  - `src/aminx/inference/logits.py` — Defines `BatchLogitFn` implementations (`ArithmeticMeanLogits`, `GeometricMeanLogits`, `ProductOfProbabilities`) that `CDRFrequencyLogitTransform` composes with.
  - `src/aminx/model/features.py` & `src/aminx/utils/graph.py` — Implement `ProteinFeatures`, `ProteinFeaturesCA`, and `compute_neighbor_offsets`.
- **Queries**:
  - `rg "compute_neighbor_offsets" src/aminx/`
  - `rg "LEGACY_ALIAS_MAP|_BARE_VARIANT_DEFAULTS" src/aminx/`
  - `rg "BatchLogitFn|logit_transform" src/aminx/`
  - `ls .praxia/docs/specs/*mpnn*`

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Padding positions (`mask == 0`) with `chain_index == 0` and `residue_index == 0` distort per-chain cycle length $L_c$ when `cyclic=True` | High (if unmasked) | High (wrong positional encodings on bucketed batches) | Compute $L_c = \max_{k \in c_{\text{valid}}} r_k - \min_{k \in c_{\text{valid}}} r_k + 1$ strictly over valid residues (`mask > 0`) in chain $c$, and test explicitly on padded batches (`L=12` padded to `32`). |
| Adding `cyclic` attribute to `ProteinFeatures` / `Aminx` changes Equinox PyTree leaf count and breaks `.eqx.zst` deserialization | Medium | High (all checkpoint loads fail) | Declare `cyclic: bool = eqx.field(static=True, default=False)` so it is part of the static treedef, not a serialized array leaf; verify existing `.eqx.zst` checkpoints deserialize unchanged. |
| Even-length cycle ($L_c = 2m$) maps both $+m$ and $-m$ to $-m$, breaking antisymmetry $d(i, j) = -d(j, i)$ | Medium | Low-Medium (directional edge asymmetry at antipode) | Explicitly preserve the sign of $d = r_i - r_j$ when `(L_c % 2 == 0) & (wrapped == -half) & (d > 0)` and assert $d_{\text{cyclic}}(j, i) == -d_{\text{cyclic}}(i, j)$ in unit tests. |
| `compute_cdr_physicochemical_descriptors` divides by zero when `cdr_mask` has 0 active positions or sequence is all `'X'` | Low | Medium (`NaN` propagation in downstream filtering) | Clamp active residue count denominator to `jnp.maximum(count, 1.0)` and zero out results via `jnp.where(count > 0, val, 0.0)`. |
