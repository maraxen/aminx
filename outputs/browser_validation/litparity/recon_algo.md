# Blind Reconstruction — ALGORITHM lens

## Lens

Algorithmic detail: control flow, loop structure, data flow, order of operations,
per-step state updates, and what is computed once vs. per decoding step. Pseudocode
is given where it clarifies control flow. Every clause cites `reference_file:line`.

## Sources read

- `model_utils.py` — `ProteinMPNN` (encode/sample/single_aa_score/score),
  `ProteinFeaturesLigand`, `ProteinFeatures`, `ProteinFeaturesMembrane`, `EncLayer`,
  `DecLayer`, `DecLayerJ`, `PositionalEncodings`, gather helpers. This file's
  `ProteinMPNN` class supports `model_type in {protein_mpnn, ligand_mpnn,
  soluble_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn}`.
- `protein_mpnn_utils.py` — an older/independent `ProteinMPNN` implementation
  (`forward`, `sample`, `tied_sample`, `conditional_probs`, `unconditional_probs`),
  plus `tied_featurize`, `parse_PDB`, `CA_ProteinFeatures`/`ProteinFeatures`. This
  file's sampling/tying math differs from `model_utils.py` in several places (see
  Ambiguities).
- `data_utils.py` — `parse_PDB`, `featurize`, `get_nearest_neighbours`, `get_score`,
  `get_seq_rec`, `write_full_PDB`.
- `sc_utils.py` — `Packer`, `ProteinFeatures` (packer-specific), `pack_side_chains`,
  `make_torsion_features`.
- `run.py` — CLI driving `model.sample()` (design/generation).
- `score.py` — CLI driving `model.score()` / `model.single_aa_score()` (scoring only).
- `proteinmpnn_2022.pdf` (Dauparas et al., *Science* 2022, HHMI author manuscript,
  18 pp.) — read in full; this copy has no separate written Methods section (points
  to PMC supplement), but Fig. 1 (p.12) and its legend, and Table 1 (p.18), are used.
- `ligandmpnn.pdf` (Dauparas et al., *Nature Methods* 2025, 14 pp.) — read in full,
  including the numbered pseudocode "Algorithm 1–14" block in the online Methods
  (pp. 8–9), which is the paper's own formal algorithm description and is cited
  directly below alongside code line numbers.

## Clauses

### 1. Default decoding order construction (noise, chain/fixed mask, argsort; fixed-first rule)

- `model_utils.py:217-220` (`ProteinMPNN.sample`): `chain_mask = mask * chain_mask`
  (folds the missing-residue mask into the design mask), then
  `decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))`.
  `argsort` is ascending, so positions with `chain_mask == 0` (fixed **or** masked/
  missing — the multiplication does not distinguish the two) get a key of
  `0.0001*|randn|` (small) and sort near the front of `decoding_order`; positions
  with `chain_mask == 1` get a key of `~1.0001*|randn|` (larger) and sort later.
  Because `decoding_order[:, t_]` is consumed in ascending step order
  (`for t_ in range(L): t = decoding_order[:, t_]`, line 262), **fixed/masked
  positions are decoded first** — this is the "fixed-first" rule.
- The same formula, `argsort((chain_M+0.0001)*abs(randn))`, recurs in
  `protein_mpnn_utils.py:1082` (`forward`, teacher-forced training path),
  `protein_mpnn_utils.py:1119` (`sample`, additionally gated by `chain_M_pos` —
  line 1118: `chain_mask = chain_mask*chain_M_pos*mask`, i.e. redesign-position
  mask is folded in as a second multiplicative gate before the same argsort), and
  `protein_mpnn_utils.py:1205` (`tied_sample`, same `chain_M_pos` gating).
- `randn` is supplied externally as `feature_dict["randn"] = torch.randn([batch_size,
  L])` (`run.py:420-423`, `score.py:281-284`) — freshly drawn **once per
  `number_of_batches` iteration**, not per decoding step.
- Paper: Fig. 1B/legend (`proteinmpnn_2022.pdf` p.12) — "decoding order... can be
  chosen such that the fixed context is decoded first" — matches the code exactly
  for this default (non-tied) case.
- `single_aa_score`/`conditional_probs` construct a *different*, per-residue
  `order_mask` instead of using `chain_mask` directly — see clause 10.

### 2. Sampling loop: temperature placement, softmax, multinomial draw, state update

Pseudocode for `model_utils.py:ProteinMPNN.sample`, trivial-symmetry branch
(`model_utils.py:236-343`):
```
h_V, h_E, E_idx = encode(...)                      # once, before loop (line 215)
mask_bw, mask_fw = build_from(decoding_order)       # once (lines 223-235)
h_EXV_encoder_fw = mask_fw * h_EXV_encoder          # once (line 260)
h_S = zeros; S = 20*ones                            # running caches
for t_ in range(L):
    t = decoding_order[:, t_]
    gather bias_t, E_idx_t, h_E_t, h_EXV_encoder_t, mask_bw_t  (per-position gather)
    for layer in decoder_layers:                    # full stack, this position only
        h_ESV_t = mask_bw_t * cat(h_V_stack[l], h_ES_t, E_idx_t) + h_EXV_encoder_t
        h_V_stack[l+1][t] = layer(h_V_stack[l][t], h_ESV_t)
    logits = W_out(h_V_stack[-1][t])                # [B,21], line 314
    log_probs = log_softmax(logits, dim=-1)         # UNBIASED, T=1  (line 315)
    probs = softmax((logits + bias_t) / temperature, dim=-1)   # (line 317-319)
    probs_sample = probs[:, :20] / probs[:, :20].sum(-1, keepdim=True)  # (320-322)
    S_t = multinomial(probs_sample, 1)              # (line 323)
    S_t = S_t*chain_mask_t + S_true_t*(1-chain_mask_t)   # hard-pick fixed (336)
    h_S[t] = W_s(S_t)                                # scatter (337-341)
    S[t] = S_t                                       # scatter (342)
```
- Temperature is applied to `logits + bias_t` **jointly**, i.e. bias is added
  *before* dividing by temperature (`model_utils.py:317-319`).
- `log_probs` (returned for scoring/confidence, `output_dict["log_probs"]`) is
  computed from the **raw, unbiased, untempered** logits (line 315) — it is *not*
  the same distribution the token was drawn from (`probs`, biased+tempered, line
  317-319). This is a real distinction: the reported log-likelihood used downstream
  for `get_score` (`data_utils.py:218-232`, called from `run.py:427-442`) does not
  include the user's amino-acid bias/omission or temperature.
- Contrast: `protein_mpnn_utils.py:ProteinMPNN.sample` (`protein_mpnn_utils.py:
  1165-1179`) divides temperature into the raw logits **first** (`logits =
  W_out(h_V_t) / temperature`), then adds `-constant*1e8` (global omit),
  `+constant_bias/temperature` (global AA bias) and `+bias_by_res_gathered/
  temperature` (per-residue bias) — bias terms are scaled by temperature
  *separately* from the omit term, which is not. This is a materially different
  formula from `model_utils.py`'s single `(logits+bias)/temperature` — see
  Ambiguities.

### 3. X (index 20) at sampling: omitted or samplable, by which mechanism

- `model_utils.py:320-322`: `probs_sample = probs[:, :20] / sum(probs[:, :20])`,
  commented `# hard omit X`. The 21st class (index 20, "X"/unknown) is
  unconditionally sliced off and the remaining 20-way distribution renormalized
  before `torch.multinomial` — X can **never** be sampled by `ProteinMPNN.sample`
  in `model_utils.py`, regardless of any bias setting.
- `protein_mpnn_utils.py:ProteinMPNN.sample` (`protein_mpnn_utils.py:1166-1180`)
  instead runs `torch.multinomial(probs, 1)` over the **full 21-way** `probs`
  (softmax over all 21 logits with global/per-residue omit and bias folded in) —
  X is sampled unless the caller explicitly adds `'X'` to `omit_AAs_np`/
  `omit_AA_mask`. Same for `tied_sample` (`protein_mpnn_utils.py:1268-1282`).
- So the two files differ on whether X-exclusion is hard-coded (`model_utils.py`)
  or an opt-in per the omit mechanism (`protein_mpnn_utils.py`) — see Ambiguities.
- `S` is initialized to `20*torch.ones(...)` in `model_utils.py:252` (i.e. "X"/
  unknown is the fill value for not-yet-decoded positions during the loop, purely
  as a placeholder — every position is overwritten by the end of the `for t_`
  loop since `decoding_order` is a full permutation of `range(L)`).

### 4. omit_AA / per-residue omit as a bias: value and where it enters

- `run.py:404-408` (CLI, feeds `model_utils.py`'s `ProteinMPNN.sample`):
  ```
  feature_dict["bias"] = (-1e8 * omit_AA[None,None,:] + bias_AA).repeat([1,L,1])
                          + bias_AA_per_residue[None] - 1e8 * omit_AA_per_residue[None]
  ```
  i.e. global omit (`--omit_AA`), global bias (`--bias_AA`), per-residue bias
  (`--bias_AA_per_residue[_multi]`) and per-residue omit
  (`--omit_AA_per_residue[_multi]`) are all folded into **one** `[B,L,21]` additive
  bias tensor at construction time, entering the sampling loop as `bias_t` added
  to `logits` before the temperature division (clause 2). An omitted class gets
  `-1e8`, driving its post-softmax probability to ~0.
- `protein_mpnn_utils.py`'s `sample`/`tied_sample` instead use **two separate**
  mechanisms at two different points: (a) a *global* omit vector `omit_AAs_np`
  subtracted as `-constant*1e8` directly in logit space
  (`protein_mpnn_utils.py:1166`, after the temperature division — see clause 2);
  (b) a *per-residue* `omit_AA_mask` `[B,L,21]` applied **multiplicatively in
  probability space, after softmax**, with renormalization:
  `probs_masked = probs*(1-omit_AA_mask_gathered); probs = probs_masked/sum(...)`
  (`protein_mpnn_utils.py:1176-1179`). This is a different mathematical mechanism
  (post-softmax multiplicative mask + renormalize) from the single pre-softmax
  additive `-1e8` bias used uniformly in `model_utils.py`/`run.py`.

### 5. augment_eps application (when, to which coordinates)

- `model_utils.py:CA_ProteinFeatures.forward` / `ProteinFeatures.forward` /
  `ProteinFeaturesMembrane.forward` (e.g. `model_utils.py:963-965`,
  `model_utils.py:1392-1393 [1-indexed dup for the top-level ProteinFeatures class],
  model_utils.py:1518-1519`): `if self.augment_eps > 0: X = X + self.augment_eps *
  torch.randn_like(X)` — applied **unconditionally whenever `augment_eps>0`**,
  with **no `self.training` guard**, i.e. it fires identically at inference time
  if a nonzero `augment_eps` is configured. It is applied to the raw backbone
  coordinate tensor `X` **before** any derived geometry (virtual Cβ construction,
  k-NN distance computation, RBF expansion) is computed, so the noise propagates
  into every downstream feature.
- `model_utils.py:ProteinFeaturesLigand.forward` (`model_utils.py:1188-1190`)
  additionally noises the ligand coordinates: `Y = Y + self.augment_eps *
  torch.randn_like(Y)`, same unconditional (no train/eval guard) behavior.
- Contrast: `sc_utils.py:ProteinFeatures.features_encode` (`sc_utils.py:938-939`)
  **does** guard with `if self.training and self.augment_eps > 0:` — the packer's
  own feature class only augments during training, unlike the main
  sequence-design feature classes above. (Inference call sites set
  `augment_eps=0.0` explicitly for both the design model and the packer in
  `run.py:77-118`, so this discrepancy is masked at the CLI level but is a real
  difference in the underlying class behavior.)

### 6. k-NN: k clamp min(k, L), topk call, tie behaviour

- Every feature class's `_dist` (`model_utils.py:1361-1364` for `ProteinFeatures`;
  analogous code in `CA_ProteinFeatures._dist`, `ProteinFeaturesLigand._dist`,
  `ProteinFeaturesMembrane._dist`, and `sc_utils.py:ProteinFeatures._dist:845-855`):
  ```
  D_adjust = D + (1. - mask_2D) * D_max        # invalid pairs pushed to D_max, not removed
  D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]),
                                    dim=-1, largest=False)
  ```
  The neighbor count is `min(top_k, L)` — clamped to sequence length (so a chain
  shorter than `top_k` still returns a full row of L neighbors, not a ragged one).
  Invalid/missing-residue pairs are not excluded from the candidate set; they are
  reassigned the per-row max valid distance (`D_max`) so they sort to the *back*
  of the ranking and are only selected if there are fewer than `top_k` genuinely
  valid partners (this is why `mask_neighbors`/`Y_m` gating is still needed
  downstream).
- `CA_ProteinFeatures._dist` (`model_utils.py:846`, comment) explicitly documents
  "Identify k nearest neighbors (**including self**)" — the zero (or near-zero,
  `+eps`) self-distance ranks first and is not excluded, so a node's own index
  can appear in its own `E_idx`.
- Tie behaviour at exactly equal distances is **not specified by this code** — it
  delegates entirely to `torch.topk`'s tie-breaking, which is not documented as
  stable/deterministic by PyTorch. Recorded as an ambiguity, not resolved.

### 7. membrane label encoding path from input to features

- `run.py:252-277` (CLI): `buried_positions`/`interface_positions` built from
  `--transmembrane_buried`/`--transmembrane_interface` residue-name lists, then
  ```
  protein_dict["membrane_per_residue_labels"] = 2*buried*(1-interface) + 1*interface*(1-buried)
  ```
  — encodes {0: neither, 1: interface, 2: buried} **except** that a residue
  flagged as *both* buried and interface evaluates to `2*1*0 + 1*1*0 = 0`
  (silently falls back to "neither") — see Ambiguities.
  For `model_type == "global_label_membrane_mpnn"`, this per-residue tensor is
  instead entirely overwritten with one scalar broadcast to all positions:
  `protein_dict["membrane_per_residue_labels"] = args.global_transmembrane_label
  + 0*fixed_positions` (`run.py:274-277`).
- `data_utils.py:featurize:961-963`: for either membrane model type, the tensor is
  just given a batch dimension (`[None,]`) — no other transform.
- `model_utils.py:ProteinFeaturesMembrane.forward:1572-1576`:
  `C_1hot = one_hot(membrane_per_residue_labels, num_classes=3)` →
  `V = norm_nodes(node_embedding(C_1hot))` — this becomes the **node** feature
  `V`; edges `E` for this model type are computed identically to the plain
  `ProteinFeatures` class (no membrane information in edges at all).
- `model_utils.py:ProteinMPNN.encode:168-170` (membrane branch): `h_V =
  self.W_v(V)` — the membrane-derived node feature enters only at the very first
  node embedding, prior to the encoder stack; it is never reintroduced later.

### 8. ligand context: atom-context number, cutoff, featurisation path

- `atom_context_num` is read from the **model checkpoint**, not a CLI flag, when
  `model_type=="ligand_mpnn"` (`run.py:69`: `atom_context_num =
  checkpoint["atom_context_num"]`); otherwise defaults to `1` (`run.py:73`).
- `--ligand_mpnn_cutoff_for_score` (default 8.0 Å) is used **only** to decide
  which residues count toward a reported "ligand confidence" score
  (`data_utils.py:featurize:948`: `mask_XY = (cutoff_for_score > D_XY) * mask *
  Y_m[:,0]`) — it does not truncate which ligand atoms are fed to the network.
- `data_utils.py:get_nearest_neighbours:889-922`: for every residue's virtual Cβ,
  computes squared distance to every ligand atom, masks by
  `mask[:,None]*Y_m[None,:]`, `argsort`s ascending, and keeps the first
  `number_of_ligand_atoms` (`==atom_context_num`) — **each residue gets its own
  independent nearest-atom set**, not a single global top-K; padded with zeros if
  fewer atoms exist than requested (lines 907-919).
- `--ligand_mpnn_use_atom_context` (`data_utils.py:featurize:955-956`): if false,
  `Y_m` (the ligand-atom mask) is zeroed *after* the nearest-neighbour selection
  above still runs — the ligand-context network branch still executes with the
  same compute, its contribution is just multiplicatively masked to (ideally)
  zero, rather than being skipped.
- `model_utils.py:ProteinFeaturesLigand.forward:1296-1328`: five sets of
  RBF-expanded protein-atom↔ligand-atom distances (N,Cα,C,O,Cβ vs. Y) + a 147-dim
  one-hot chemical (element/group/period) type feature projected to 64 dims via
  `type_linear` + a 4-dim local-frame angle feature (`_make_angle_features`) are
  concatenated and projected via `node_project_down` into the ligand-conditioned
  node feature `V`. Separately, ligand-atom–to–ligand-atom RBF distances become
  `Y_edges` and ligand atom types become `Y_nodes` (`model_utils.py:1316-1326`).
  In `ProteinMPNN.encode` (`model_utils.py:140-154`), `Y_nodes`/`Y_edges` are
  message-passed among themselves via `y_context_encoder_layers` (`DecLayerJ`),
  then cross-attended into the protein node stream `h_V_C` via
  `context_encoder_layers` (`DecLayer`), added back into `h_V` after a
  `LayerNorm`+`Dropout` (`V_C_norm`, line 154) — a residual-style fusion, not a
  concatenation into the main encoder stack.

### 9. side-chain-context toggle and what it changes

- `ligand_mpnn_use_side_chain_context` → `ProteinFeaturesLigand(use_side_chains=...)`
  (`model_utils.py:42`, `679-684`). When true, `forward`
  (`model_utils.py:1246-1281`):
  - Pulls fixed (non-designed) residues' side-chain heavy atoms
    (`xyz_37[:,:,5:,:]`, the 32 atoms beyond N/CA/C/CB/O) as extra context,
    restricted to the top-16 protein-graph neighbors per residue
    (`E_idx_sub = E_idx[:,:,:16]`, line 1249).
  - Masked to only the **fixed** positions:
    `xyz_37_m * (1 - mask_residues[:,:,None])` where `mask_residues =
    input_features["chain_mask"]` (line 1250-1251) — a **designed** residue's own
    side chain is explicitly excluded from being used as context for itself/
    others (it doesn't exist yet at the semantic level of the design problem).
  - These side-chain atoms are `torch.cat`-appended onto the true ligand-atom
    tensors `Y`/`Y_t`/`Y_m` (lines 1268-1270) and then **re-subjected to the same
    nearest-neighbour re-selection down to `atom_context_num`** (via the
    Cβ-distance top-k immediately below, lines 1272-1281) — side chains are not a
    separate encoder branch; they compete with true ligand atoms for the same
    fixed-size context slots.
  - `run.py:195-197`: the CLI must parse all 37 atoms (`parse_all_atoms_flag =
    ligand_mpnn_use_side_chain_context or (pack_side_chains and not
    repack_everything)`) whenever this flag (or non-`repack_everything` packing)
    is on, since `xyz_37`/`xyz_37_m` are otherwise absent from `protein_dict`.

### 10. score_only: number of decoding orders, averaging procedure

- `score.py:280-299`: loops `for _ in range(args.number_of_batches)`, drawing a
  **fresh** `feature_dict["randn"]` (hence a fresh decoding order) each iteration,
  calling either `model.score()` (`--autoregressive_score`) or
  `model.single_aa_score()` (`--single_aa_score`, the CLI default,
  `score.py:541-545`). Results from each iteration are `torch.cat`-stacked along
  batch dim (`score.py:296-299`) — **not** averaged inside the model.
- Averaging happens only downstream over the stacked batch of independent runs:
  `mean_probs = np.mean(out_dict["probs"], 0)`, `std_probs = np.std(...)`
  (`score.py:314-315`) — an unweighted arithmetic mean/std over the
  `number_of_batches` draws, per amino acid per residue.
- `model_utils.py:single_aa_score` (`model_utils.py:470-556`): for **each**
  residue `idx` in `range(L)` **independently**, builds a per-position
  `order_mask` (`model_utils.py:501-506`):
  ```
  if not use_sequence: order_mask = zeros(L); order_mask[idx] = 1.
  else:                 order_mask = ones(L);  order_mask[idx] = 0.
  decoding_order = argsort((order_mask + 0.0001) * abs(randn))
  ```
  and runs the **full** decoder stack once for that `idx` (no incremental
  caching across `idx` iterations — see clause 13), extracting only
  `log_probs[:, idx, :]` / `logits[:, idx, :]` from the result
  (`model_utils.py:546-548`). Because `h_S = W_s(S_true)` is built once from the
  **true, full** sequence (`model_utils.py:529`, not incrementally revealed), the
  `order_mask` only controls whether the *other* L−1 positions are placed
  "before" or "after" `idx` in the mask_bw/mask_fw partition — see clause 14 and
  the corresponding entry in Ambiguities (the direction of `use_sequence`'s
  effect here needs care and is recorded as an open question, not resolved).
- `model.score()` (`model_utils.py:559-665`) is a *different* function with its
  own, more literal `use_sequence` semantics: if false, the decoder loop entirely
  skips the `h_ES`-based branch (`for layer: h_V = layer(h_V, h_EXV_encoder_fw,
  mask)`, `model_utils.py:647-648`, i.e. no sequence information anywhere,
  matching `unconditional_probs`); if true, the normal
  `mask_bw*h_ESV + h_EXV_encoder_fw` branch runs (`model_utils.py:651-654`), using
  the single decoding order derived from `chain_mask`/`randn` exactly as in
  `sample()` (clause 1), with symmetry-tying handled the same way as `sample()`
  (`model_utils.py:600-631`, mirroring the symmetric branch of clause 12).

### 11. packer chi sampling procedure

- `sc_utils.py:make_torsion_features` (`sc_utils.py:147-229`): converts the MPNN
  21-letter alphabet to AF2 `aatype` via a fixed permutation matmul
  (`map_mpnn_to_af2_seq`, lines 156-161); assembles `xyz37` from the model's 4
  backbone atoms (N/CA/C at slots 0:3, O placed at slot 4, line 153-155); calls
  OpenFold's `atom37_to_torsion_angles` to get ground-truth torsion sin/cos
  (`torsion_angles_sin_cos`, of which `[...,3:]` are the 4 **chi** angles — the
  first 3 slots are backbone dihedrals, not touched by the packer at all); builds
  rigid frames from N/CA/C via `Rigid.make_transform_from_reference`.
  - If `repack_everything=False`: uses the **true** crystal `xyz_37` to compute
    `torsions_true`, and `mask_fix_sc = chain_mask[:,:,None,None]` — only
    `chain_mask==1` (designed) positions get their chi angles randomized/denoised;
    fixed positions keep their true chi throughout.
  - If `repack_everything=True`: `torsions_true = zeros(...)` and
    `mask_fix_sc = ones(...)` — every position is repacked, no reference retained.
  - Initialization: chi torsions (`[...,3:]`) are replaced with a **uniform
    random angle in [0, 2π)** (`sc_utils.py:193-198`), blended against
    `torsions_true` by `mask_fix_sc`. Backbone torsion slots (`[...,0:3]`) are
    left exactly as computed from the **observed/current** backbone — never
    randomized or predicted by the packer.
- `sc_utils.py:pack_side_chains` (`sc_utils.py:56-144`):
  - `h_V, h_E, E_idx = model_sc.encode(feature_dict)` is computed **once**, before
    the denoising loop, and cached in `feature_dict` (lines 86-89) — see clause 13.
  - For `step in range(num_denoising_steps)` (lines 90-124):
    1. `model_sc.decode(feature_dict)` (recomputed fresh each step from the
       *current* `feature_dict["X"]`) returns `(mean, concentration, mix_logits)`,
       each shaped `[B, L, 4 chi, num_mix=3]` (`sc_utils.py:Packer.decode:
       383-388`) — a per-residue, per-chi-angle, 3-component mixture-of-von-Mises
       distribution.
    2. Build `D.MixtureSameFamily(D.Categorical(mix_logits), D.VonMises(mean,
       concentration))`, draw `num_samples` candidate angle sets
       (`pred_dist.sample([num_samples])`), score each with `pred_dist.log_prob`,
       and — independently per (residue, chi-angle) — keep the single sample with
       the **highest log-probability** among the `num_samples` draws
       (`torch.argmax(log_probs_of_samples, 0)` then `torch.gather`,
       `sc_utils.py:96-99`). This is best-of-N / mode-seeking selection, not a
       plain single stochastic draw.
    3. The chosen angles overwrite the noised chi slots **only where
       `mask_fix_sc==1`**; fixed positions keep `torsions_true`
       (`sc_utils.py:103-105`).
    4. The updated torsion set is converted back to atom14 xyz via OpenFold's
       `torsion_angles_to_frames` + `frames_and_literature_positions_to_atom14_pos`,
       masked by `X_m`, and written into `feature_dict["X"]`
       (`sc_utils.py:106-121`) — this becomes the input to the **next** step's
       `model_sc.decode()` call (not `encode()` — see clause 13).
  - After the loop: `log_prob = pred_dist.log_prob(sample) * mask_fix_sc + 2.0 *
    (1 - mask_fix_sc)` (`sc_utils.py:126-128`) — fixed positions are assigned an
    arbitrary constant "confidence" of `2.0` rather than a real log-likelihood;
    this value is later mapped to a per-atom b-factor proxy via a
    `restype_atom14_to_rigid_group` chi-group lookup (`sc_utils.py:130-136`).

### 12. Tied/symmetry positions in the sampling loop

(This clause covers `model_utils.py:ProteinMPNN.sample`'s non-trivial-symmetry
branch, `model_utils.py:350-467`, entered whenever `symmetry_residues` is not the
sentinel `[[]]`.)

- **(a) Fixed members' effect on the group:** `chain_mask_t = chain_mask[:, t]` is
  read **individually per member** inside the first per-`t` loop
  (`model_utils.py:415`) — there is no group-level reduction of fixedness before
  fusion. After the single joint sample `S_t_repeat` is drawn for the whole
  group, a **second** per-member loop applies each member's own gate:
  `S_t_repeat = (chain_mask[:,t]*S_t_repeat + (1-chain_mask[:,t])*S_true[:,t])`
  (`model_utils.py:457-458`, inside `for t in t_list:` at line 452). So a
  **fixed** member of a tied group is forced back to its own true residue even
  though the whole group shared one sampled value — "tying" is only actually
  enforced across the **designed** members of a group; a fixed member can end up
  different from its designed, nominally-tied partners.
- **(b) Which member's bias is added:** `bias_t = bias[:, t]`
  (`model_utils.py:417`) is (re)assigned inside the same first per-`t` loop
  **without accumulation or reset**. After that loop exits, `bias_t` holds the
  value from the **last** `t` processed in `t_list` — a Python loop-variable
  leak — and it is that single, last-member bias that is added in the shared
  softmax call `probs = softmax((total_logits + bias_t)/temperature)`
  (`model_utils.py:445-447`). Every other member's own per-residue bias is
  silently discarded for this group's joint sampling step.
- **(c) Logit fusion formula and weights:** `total_logits = 0.0`
  (`model_utils.py:413`) accumulates `total_logits += symmetry_weights[t] *
  logits` for each `t` in the group (`model_utils.py:443`), where `logits =
  self.W_out(h_V_t)` is that member's own raw (untempered) decoder output and
  `symmetry_weights` comes from `feature_dict["symmetry_weights"]`
  (CLI: `--symmetry_weights`, or `1/num_chains` per position when
  `--homo_oligomer` is set — `run.py:352-358`). This is a **weighted sum**, not
  divided by group size — it is a true average only when the caller's weights
  already sum to 1 per group (as `--homo_oligomer`'s `1/len(chain_letters_set)`
  weights do). `probs = softmax((total_logits + bias_t)/temperature, dim=-1)`
  (line 446) is computed **once** for the whole group.
- **(d) Sequential processing / does a later member see an earlier member's
  updated state:** The members of `t_list` **are** processed strictly
  sequentially, in the **same outer decoding step**, via `for t in t_list:`
  (`model_utils.py:414-443`), and each member runs the **full** decoder-layer
  stack, mutating `h_V_stack` in place: `h_V_stack[l+1][:, t, :] = layer(...)`
  (`model_utils.py:431-433`). Because `h_V_stack[l]` is read via
  `cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)` (line 424) for **every**
  member's forward pass, a later member `t'` in the same `t_list` **would** see
  an earlier member `t`'s just-written `h_V_stack` activations **if** `t'`
  happens to be a k-NN graph-neighbor of `t` (since `h_V_stack` is a single
  shared tensor mutated across the whole batch of positions, not member-scoped).
  However, `h_S` (the sequence/token embedding) for **any** member of the current
  group is **not** written until the second per-member loop, after the joint
  sample is drawn (`model_utils.py:459`, inside `for t in t_list:` at line 452)
  — so within the first loop, no group member's *identity* embedding is visible
  to any other group member yet, only whatever partial hidden-state leakage
  occurs via `h_V_stack`. Whether this `h_V_stack` cross-member leakage is an
  intended part of the tying design or an incidental side effect of reusing
  shared tensors across the whole sequence is not stated anywhere in the code or
  paper — recorded as an ambiguity below, not resolved.
- Paper (`ligandmpnn.pdf` Fig. 1 legend, and `proteinmpnn_2022.pdf` Fig. 1C
  legend, p.12): "Predicted logits for tied positions are **averaged** to get a
  single probability distribution." The code in `model_utils.py` (weighted sum,
  not renormalized by group size in the general case) only literally averages
  when the supplied weights already sum to 1 — see Ambiguities for the
  cross-file disagreement with `protein_mpnn_utils.py:tied_sample`, which *does*
  divide by `len(t_list)` in addition to a `tied_beta` weight
  (`protein_mpnn_utils.py:1262-1263`).

### 13. What is cached between decoding steps vs. recomputed

- `ProteinMPNN.sample` trivial-symmetry branch (`model_utils.py:215-343`):
  - `h_V, h_E, E_idx` (encoder outputs) — computed **once** via `self.encode()`
    before the loop (line 215); never recomputed.
  - `h_EXV_encoder_fw = mask_fw * h_EXV_encoder` — computed **once** (line 260);
    only gathered (indexed) per position per step (line 279-285), never
    recomputed.
  - `mask_bw`, `mask_fw`, `order_mask_backward`, `permutation_matrix_reverse` —
    computed **once** from `decoding_order` before the loop (lines 223-235);
    gathered per position per step (lines 287-293).
  - `h_V_stack` (one tensor per decoder-layer depth, `len(decoder_layers)+1`
    entries) — initialized once (`[h_V] + [zeros]*num_layers`, lines 253-256),
    then **incrementally** updated: each step writes only the single position
    `t`'s entry via `scatter_`/index-assign (lines 295-307); all other positions'
    entries persist from earlier steps (or remain zero if not yet decoded).
  - `h_S` — starts at all-zeros (line 251), filled in **one position per step**
    via `scatter_` (lines 337-341) — a running cache of decoded-token embeddings;
    non-decoded positions read as zero embedding until their own turn.
- Contrast, `single_aa_score` (`model_utils.py:470-556`): **no** cross-iteration
  caching at all. The outer Python loop is `for idx in range(L)`, and **each**
  iteration independently re-derives a fresh `decoding_order`
  (`model_utils.py:507-509`), rebuilds `mask_bw`/`mask_fw` from scratch
  (lines 511-523), and re-runs the **entire** decoder-layer stack over the
  **whole** sequence (`for layer in self.decoder_layers: h_ESV = ...`, lines
  537-541) — i.e. `O(L)` full, non-incremental decoder passes, one per residue,
  versus `sample()`'s `O(L)` incremental single-position updates that are
  layer-parallel across the whole sequence but position-serial in decoding order.
- `sc_utils.py:pack_side_chains` (clause 11): `model_sc.encode()` (backbone
  encoder + protein–ligand context fusion) is computed **once**, before the
  denoising loop, and its output cached directly in `feature_dict["h_V"/"h_E"/
  "E_idx"]` (`sc_utils.py:86-89`); only `model_sc.decode()` (which reads the
  *current* side-chain coordinates for its explicit atom-pair RBF features) is
  recomputed every denoising step.

### 14. The autoregressive visibility mask construction

- Common core, appearing (with identical structure) in `model_utils.py:sample`
  (lines 221-231), `model_utils.py:score` (lines 585-596, and again after
  `repeat` at 623-631 for the symmetric branch), and mirrored in
  `protein_mpnn_utils.py:forward`/`sample`/`tied_sample`
  (lines 1084-1089 / 1121-1126 / 1218-1223):
  ```
  P = one_hot(decoding_order, num_classes=L)              # [B,L,L] rank-one-hot
  order_mask_backward = einsum('ij,biq,bjp->bqp',
                                (1 - triu(ones(L,L))), P, P)   # [B,L,L]
  mask_attend = gather(order_mask_backward, 2, E_idx)       # narrow to k-NN graph
  mask_1D = mask.view([B,L,1,1])
  mask_bw = mask_1D * mask_attend          # "p decoded strictly before q"
  mask_fw = mask_1D * (1 - mask_attend)    # "p decoded at-or-after q" (incl. p==q)
  ```
  `1 - triu(ones(L,L))` is strictly lower-triangular (all-ones where
  `row_rank > col_rank`). Contracting through two copies of the one-hot rank
  matrix `P` re-expresses "later rank than" back in terms of the **original**
  sequence positions: `order_mask_backward[b,q,p] = 1` iff `p` was decoded
  strictly before `q` under `decoding_order`. This dense `[B,L,L]` relation is
  then narrowed to just the k-NN graph edges via `torch.gather(..., E_idx)` (only
  the K graph-neighbor columns per query position survive) and combined with the
  plain residue-validity mask (`mask_1D`, exclusion of missing/padded residues).
- Self-position (`p==q`) always lands in `mask_fw` (the identity/zero-`P`
  diagonal of `1-triu(...)` has 0 there, so `q` is never counted as "before
  itself"): every position always sees **itself** only through the
  backbone/ligand encoder branch (`h_EXV_encoder_fw`, zero sequence embedding),
  never through its own true/sampled identity — this holds uniformly across
  `sample`, `score`, `single_aa_score`, `conditional_probs`, and the tied/
  symmetric branches.
- `mask_bw`-gated features use `h_ES` (built from `h_S`, which carries true or
  already-sampled sequence identity depending on the function); `mask_fw`-gated
  features use `h_EX_encoder`/`h_EXV_encoder` (built with `torch.zeros_like(h_S)`
  in place of any sequence embedding — i.e. backbone/ligand-only, no identity
  information at all, `model_utils.py:258-259`, `640-643`, `1140-1141`,
  `protein_mpnn_utils.py:1076-1077`, `1311-1312`, `1367-1368`).

## Ambiguities

1. **Temperature/bias/omit formula differs between the two `ProteinMPNN.sample`
   implementations.** `model_utils.py:317-319` computes
   `softmax((logits+bias)/temperature)` (bias inside the temperature division,
   with X hard-omitted afterward — clause 2/3). `protein_mpnn_utils.py:1165-1179`
   instead computes `logits/temperature` first, then adds an **un-tempered**
   `-constant*1e8` (global omit) plus **tempered** `constant_bias/temperature`
   and `bias_by_res/temperature` terms, and additionally applies per-residue
   `omit_AA_mask` **multiplicatively after softmax** with renormalization
   (clause 4). These are not algebraically equivalent formulas (temperature
   scales the omit penalty differently, and the per-residue-omit mechanism is
   pre- vs. post-softmax). Which is authoritative depends entirely on which of
   the two code paths is treated as canonical; the packet gives no basis to
   prefer one over the other as "the" algorithm.
2. **X (index-20 "unknown") sampling is hard-disabled in one file, opt-in in the
   other** (clause 3) — `model_utils.py` always slices `probs[:, :20]` before
   `multinomial`; `protein_mpnn_utils.py` samples over the full 21-way `probs`
   unless the caller adds `'X'` to the omit list.
3. **Symmetric/tied logit fusion is a plain weighted sum in `model_utils.py`
   but a size-normalized, temperature-pre-divided, `tied_beta`-weighted average
   in `protein_mpnn_utils.py:tied_sample`** (clause 12c):
   `model_utils.py:443`: `total_logits += symmetry_weights[t] * logits` (no
   `/len(t_list)`, `logits` untempered at accumulation time) vs.
   `protein_mpnn_utils.py:1262-1263`: `logits += tied_beta[t] *
   (self.W_out(h_V_t)/temperature) / len(t_list)`. The paper's own wording
   ("logits ... are averaged", Fig. 1C legend / Fig. 1 legend in both PDFs)
   matches the *older* file's normalized-average formula more closely than the
   newer generalized-symmetry branch, which only reduces to a true average when
   the caller's `symmetry_weights` happen to sum to 1 per group (true for the
   `--homo_oligomer` convenience path, not guaranteed for arbitrary
   `--symmetry_weights`). Not resolved from the packet which is "the" intended
   behavior for hand-specified, non-normalized weights.
4. **Loop-variable leak in the tied-sampling bias** (clause 12b):
   `bias_t = bias[:, t]` is overwritten on every iteration of the inner
   `for t in t_list:` loop and used *after* the loop with whatever value it last
   held (the last member's bias). Cannot tell from the code whether this is
   deliberate (e.g. "use the last/reference member's bias for the group") or an
   unintentional artifact of reusing a per-member loop variable outside its
   natural per-member scope — no comment addresses it either way.
5. **`h_V_stack` cross-member visibility within a tied step** (clause 12d): a
   later member of a symmetry group can see an earlier member's freshly written
   decoder hidden state (via the shared `h_V_stack` tensor and k-NN graph
   adjacency) even though the group's sequence identity (`h_S`) is not yet
   shared. Whether this partial, graph-adjacency-gated leakage is an intentional
   part of how "tying" propagates information or simply falls out of using
   shared, whole-sequence tensors for a per-position update loop is not stated
   anywhere in the code or paper text read.
6. **`single_aa_score`'s `use_sequence` flag direction is unclear/possibly
   inverted relative to its own docstring** (clause 10). The function docstring
   says `use_sequence - False using backbone info only`, but tracing
   `order_mask` construction (`model_utils.py:501-506`) shows: when
   `use_sequence=False`, `idx` is the **only** position with a nonzero
   `order_mask` entry, which — given `argsort` is ascending and all other
   positions get the tiny `0.0001*|randn|` key — places `idx` **last** in
   decoding order, meaning **all other true residues are visible** to `idx`
   when it is finally scored (they are all "before" it, i.e. reachable via the
   `h_ES`/true-sequence branch). Conversely `use_sequence=True` places `idx`
   **first** (`order_mask[idx]=0`, others `=1`), meaning `idx` is scored with
   **no** other true residues visible (backbone/context only via `h_EXV_encoder`).
   This traced behavior is the reverse of what the docstring states in plain
   English. This has not been re-derived independently or confirmed against the
   paper (which does not describe `single_aa_score` at this level of detail);
   recorded as an open question rather than asserted as a bug, since a
   misreading of the `mask_bw`/`mask_fw` convention on my part cannot be ruled
   out from static reading alone.
7. **`torch.topk` tie-breaking for k-NN selection** (clause 6) is unspecified by
   the reference code and not asserted by PyTorch's documentation to be stable;
   no behavior is claimed here beyond "delegates to `topk`".
8. **Membrane label collision (buried AND interface both set)** (clause 7):
   `2*buried*(1-interface) + 1*interface*(1-buried)` evaluates to `0` (i.e.
   "neither") if a residue is listed in both `--transmembrane_buried` and
   `--transmembrane_interface`. Cannot tell whether this is an intentional "such
   inputs are invalid/mutually exclusive by contract" assumption (unstated
   anywhere) or an unhandled edge case.
9. Whether `augment_eps`'s lack of a `self.training` guard in the main
   sequence-design feature classes (clause 5) is deliberate (e.g., to allow
   noised backbones to be scored/sampled at inference too, as a user-controlled
   robustness knob) or simply because those classes were never intended to be
   instantiated with nonzero `augment_eps` outside training is not stated; the
   `sc_utils.py` packer's feature class *does* guard on `self.training`, which
   is the only textual signal that the omission elsewhere might be worth
   flagging rather than assuming intentional.

## Assumptions

- Treated `model_utils.py`'s `ProteinMPNN` (the file bundled with `run.py`/
  `score.py`/`sc_utils.py`, supporting `model_type` dispatch across
  protein/ligand/membrane/soluble variants) as the more complete, more current
  description of the sampling algorithm's control flow for elements not present
  at all in `protein_mpnn_utils.py` (ligand context, membrane labels, side-chain
  context, the generalized `symmetry_residues`/`symmetry_weights` mechanism,
  `single_aa_score`). Where `protein_mpnn_utils.py` implements an overlapping but
  numerically different mechanism (temperature/bias math, tying-average formula,
  X-sampling), both are reported side by side as a disagreement (see
  Ambiguities) rather than one being treated as superseding the other.
- Assumed `decoding_order[:, t_]`/`torch.argsort` ascending-order semantics
  throughout (verified directly against `torch.argsort`'s documented default
  behavior, not assumed from context) — i.e. "smaller sort key ⇒ earlier decode
  step" is a direct reading of the `argsort` call, not an inference.
- Did not attempt to independently re-derive or verify the einsum-based
  `order_mask_backward` construction (clause 14) numerically (e.g. by hand-
  tracing a small L); the interpretation given is a direct symbolic reading of
  the einsum contraction and the `1-triu` mask, treated as reliable since the
  indices and their roles are named explicitly in the surrounding code.
- Where the ProteinMPNN 2022 paper (`proteinmpnn_2022.pdf`) is cited, only the
  main text and figure legends were available in this packet (no separate
  written Methods section is present in this copy — it refers readers to a PMC
  supplement not included here); paper citations for that work are therefore
  limited to what Fig. 1/Fig. 2/Table 1 and the main narrative state.
- Read the LigandMPNN paper's own "Algorithm 1–14" pseudocode block (online
  Methods) as the paper's formal algorithm description and treated it as
  consistent with the code's `ProteinMPNN`/`ProteinFeaturesLigand`/`Packer`
  classes in `model_utils.py` (variable names and structure line up directly,
  e.g. Algorithm 12 `LigandMPNN_decode` ≈ the decoder loop body inside
  `model_utils.py:sample`/`score`); did not find any further disagreement
  between that pseudocode and the code beyond what is already listed under
  Ambiguities (the paper's pseudocode does not cover the tied/symmetric
  branch's exact fusion arithmetic, which is where the code-level ambiguities
  above arise).
