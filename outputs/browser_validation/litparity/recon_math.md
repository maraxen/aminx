# Blind Reconstruction — MATHEMATICAL-FORMULATION Lens

## Lens
Mathematical formulation: equations, tensor shapes, normalisations, constants, masks, and
exact numeric conventions, as implemented by the reference code (not as a generic MPNN
might do it). Every clause cites `file:line` in the reference packet; paper section cited where
the paper covers it.

## Sources read
- `run.py` (LigandMPNN inference CLI, commit 26ec57ac)
- `model_utils.py` (`ProteinMPNN`, `ProteinFeatures`, `ProteinFeaturesLigand`,
  `ProteinFeaturesMembrane`, `EncLayer`, `DecLayer`, `DecLayerJ`, gather/cat helpers; 26ec57ac)
- `data_utils.py` (`parse_PDB`, `featurize`, `get_nearest_neighbours`, `get_score`,
  `get_seq_rec`; 26ec57ac)
- `sc_utils.py` (`Packer`, `pack_side_chains`, `make_torsion_features`,
  `ProteinFeatures` decoder-side; 26ec57ac)
- `score.py` (scoring CLI: `single_aa_score`/`score`/`autoregressive_score` dispatch; 26ec57ac)
- `protein_mpnn_utils.py` (ProteinMPNN reference, commit 8907e667) — read in full; where its
  math is identical to the LigandMPNN packet copies (RBF, positional encodings, Enc/DecLayer,
  `/scale` constant, argsort decoding order) it is not re-cited separately below since the
  LigandMPNN packet's `model_utils.py` is the shared descendant actually exercised by `run.py`.
- ProteinMPNN 2022 (Dauparas et al., *Science* 378:49–56), full 10-page main-text PDF. Its
  Methods/equations are **not included** — the PDF text itself says "Supplementary Material:
  Refer to Web version on PubMed Central" (page 9); only prose description of decoding order,
  positional-encoding cap, and symmetric-logit averaging is available from this document.
- LigandMPNN (Dauparas et al., *Nature Methods* 22:717–723, 2025), full 14-page PDF **including
  an in-article Methods section with explicit pseudocode**, Algorithms 1–14 (pages 8–10 of the
  PDF). This is the primary paper source for equations below; cited as "LigandMPNN Alg. N".

---

## Clauses

**1. Default decoding order** — `model_utils.py:217-220` (`ProteinMPNN.sample`, non-symmetric
branch) and `:581-584` (`.score`):
```
chain_mask = mask * chain_mask
decoding_order = argsort((chain_mask + 0.0001) * abs(randn))
```
`randn` is standard-normal noise (`[B,L]`, only the first batch row's values are used — see
`run.py:420-423` allocating `[batch_size, L]` and the sample docstring at
`model_utils.py:199-201` noting only the first entry is meaningful for symmetric decoding).
Argsort is ascending, so index 0 of `decoding_order` is decoded first. Because chain_mask∈{0,1},
fixed positions (chain_mask=0) get keys ~`0.0001·|randn|` and designed positions (chain_mask=1)
get keys ~`1.0001·|randn|`; the 1e4 scale gap makes fixed positions sort before designed ones
with overwhelming probability (not a hard guarantee — see Ambiguities), i.e. "fixed-first,
then random order among designed positions by |randn| magnitude." `single_aa_score`
(`:507-509`) uses the identical `argsort((order_mask+0.0001)*abs(randn))` formula but replaces
`chain_mask` with a task-specific `order_mask` (see clause 15). Paper: *Science* p.3, "the
decoding order is randomly sampled from the set of all possible permutations" — no formula
given (order-agnostic training via random permutations, consistent with the code's `argsort`
trick for producing a valid random permutation while forcing fixed positions first).

**2. Temperature: applied to (logits + bias), not logits alone** — `model_utils.py:317-319`:
```
probs = softmax((logits + bias_t) / temperature, dim=-1)   # [B,21]
```
`bias_t` is gathered per-position bias (clause 4) which already contains the `-1e8` omission
penalty, so temperature also divides that penalty (still effectively −∞ for any typical
T∈[0.05,2]). This matches LigandMPNN Alg. 13 exactly: `p = softmax((logits+bias)/T)`. Symmetric
branch (`:445-447`) applies the same formula to `total_logits` (fused logits, clause 13) plus
`bias_t`.

**3. X (index 20) handling: samplable in the raw distribution, hard-omitted at the draw** —
`model_utils.py:314-322`:
```
logits = self.W_out(h_V_t)                          # [B,21]  -- includes class 20 (X)
log_probs = log_softmax(logits, dim=-1)             # [B,21]  -- X-inclusive, stored in full
probs = softmax((logits + bias_t) / temperature)    # [B,21]
probs_sample = probs[:, :20] / sum(probs[:, :20])   # [B,20]  -- "hard omit X" (code comment)
S_t = multinomial(probs_sample, 1)[:, 0]
```
So `log_probs` (21-way, returned to callers, used by `get_score`/CCE in `data_utils.py:218-232`)
faithfully reports whatever probability mass the network assigns to X, but the actual
categorical draw renormalizes over the first 20 classes only — X can never be sampled. `S` is
pre-initialized to the X sentinel (`20 * ones(...)`, `:252`) before any position is filled.
Not paper-documented (X/UNK handling is not in the packet's Algorithm 13 pseudocode, which
samples over all 21 without an explicit renormalization step — code-only detail).

**4. omit_AA / per-residue omit as a `-1e8` additive bias, folded into `bias` before decode** —
`run.py:404-408`:
```
feature_dict["bias"] = (-1e8 * omit_AA[None,None,:] + bias_AA).repeat([1,L,1])
                        + bias_AA_per_residue[None]
                        - 1e8 * omit_AA_per_residue[None]
```
i.e. global `omit_AA` and per-residue `omit_AA_per_residue` both use exactly `-1e8` (not `-inf`,
not scaled by anything else), summed together with the (unbounded-magnitude, user-supplied)
`bias_AA` / `bias_AA_per_residue` terms into one `[1,L,21]` tensor. This whole tensor is added
to raw `logits` before the temperature division (clause 2), at `model_utils.py:266-268,318`
(gathered per-timestep as `bias_t`).

**5. augment_eps (backbone/ligand noise): isotropic Gaussian added to raw coordinates,
default 0.0 (off at inference)** — `model_utils.py:1188-1190` (`ProteinFeaturesLigand.forward`):
```
if self.augment_eps > 0:
    X = X + self.augment_eps * torch.randn_like(X)
    Y = Y + self.augment_eps * torch.randn_like(Y)
```
Identical `if self.augment_eps > 0: X = X + augment_eps*randn_like(X)` pattern in the
protein-only `ProteinFeatures.forward` (`:1392-1393`) and `ProteinFeaturesMembrane.forward`
(`:1518-1519`) — ligand coordinates `Y` are only noised in the ligand-aware feature class.
`ProteinMPNN.__init__` default is `augment_eps=0.0` (`:20`) and `run.py`'s model construction
(`:77-88`) never overrides it, so **inference always runs with augment_eps=0** regardless of
what the checkpoint was trained with. Paper: LigandMPNN Alg. 10 lists `noise_level ∈ ℝ` as a
parameter with stated default 0.1 (training-time value, page 8: "We augmented the training
dataset by ... adding 0.1 Å standard deviation Gaussian noise"); ProteinMPNN 2022 paper
reports training variants at std=0.02 Å and 0.3 Å (page 4-5) — both papers describe this as a
*training* augmentation; the packet code confirms it is architecturally also available at
inference (same `forward`) but disabled by the CLI's default construction.

**6. k-NN neighbour count: clamped to min(k, L); Cα–Cα distance; masked pairs pushed to
row-max, not zero** — identical `_dist` in `ProteinFeatures` (`model_utils.py:1355-1364`),
`ProteinFeaturesLigand` (`:1148-1157`), `ProteinFeaturesMembrane` (`:1480-1489`), and the
packer's `ProteinFeatures` (`sc_utils.py:845-855`):
```
mask_2D = mask[:,:,None] * mask[:,None,:]
D = mask_2D * sqrt(sum((X_i - X_j)**2, -1) + eps)
D_max = max(D, -1, keepdim=True)
D_adjust = D + (1 - mask_2D) * D_max            # masked pairs pushed to the row max, not 0
D_neighbors, E_idx = topk(D_adjust, min(top_k, L), largest=False)
```
`eps=1e-6`. All coordinates are Cα (`X[:,:,1,:]`, or `X[:,:,self.CA_idx,:]` in the packer). The
`np.minimum(top_k, L)` clamp means a chain shorter than `top_k` (48 for `protein_mpnn`, `k`
read from `checkpoint["num_edges"]` — `run.py:71,75`) uses `k=L`, i.e. a fully-connected graph.

**7. top-k tie-break: unspecified / implementation-defined, not controlled by the code** —
`torch.topk(..., largest=False)` (clause 6) and, separately, `torch.argsort(L2_AB, -1)` in
`get_nearest_neighbours` (`data_utils.py:895`) — neither call passes `stable=True` or any
secondary sort key (e.g. index). No manual tie-break logic exists anywhere in the packet.
Record as ambiguity (below) rather than assumed behavior.

**8. Membrane label encoding: per-residue ∈ {0,1,2} one-hot into a 3-class node embedding;
global label is a scalar broadcast into the same 3-class encoding** — construction in
`run.py:270-277`:
```
protein_dict["membrane_per_residue_labels"] = 2*buried*(1-interface) + 1*interface*(1-buried)
```
so 0 = neither flag (default/soluble), 1 = interface-only, 2 = buried-only (both flags set
simultaneously cancels back to 0 — see Ambiguities). For `global_label_membrane_mpnn`
(`:274-277`): `membrane_per_residue_labels = global_transmembrane_label + 0*fixed_positions`
(broadcasts a single 0/1 scalar to every residue; class 2 is structurally never produced).
Consumed in `ProteinFeaturesMembrane.forward` (`model_utils.py:1572-1576`):
```
C_1hot = one_hot(membrane_per_residue_labels, num_classes=3).float()
V = norm_nodes(node_embedding(C_1hot))         # node_embedding: Linear(3, node_features, bias=False)
```
i.e. this is a **node** feature (not an edge feature) — `node_embedding` weights defined at
`:1475-1477`.

**9. Ligand atom context: two-stage nearest-neighbour selection, count and "cutoff" have
different roles** — Stage 1, `data_utils.get_nearest_neighbours` (`:889-922`), run from
`featurize()` (`:945-947`) using virtual Cβ (formula clause below) and the model's
`atom_context_num` (from `checkpoint["atom_context_num"]`, `run.py:69`) as
`number_of_ligand_atoms`:
```
L2_AB = sum((Cb_i - Y_j)^2, -1)
L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0      # sentinel, NOT a hard cutoff
nn_idx = argsort(L2_AB, -1)[:, :number_of_ligand_atoms]  # nearest-by-count, no distance gate
```
There is **no distance cutoff at atom-selection time** — only a fixed count of nearest atoms.
The separately-named `cutoff_for_score` (`featurize()` default arg `8.0`, exposed as CLI
`--ligand_mpnn_cutoff_for_score`, `run.py:842-846`) instead gates `mask_XY`
(`data_utils.py:948`): `mask_XY = (cutoff_for_score > D_XY) * mask * Y_m[:,0]` — used only to
decide which residues count toward the *reported* ligand-conditioned score (`get_score` with
`combined_mask`, `run.py:432-442`), not which atoms the encoder sees. Stage 2 (only when
`ligand_mpnn_use_side_chain_context=1`), `ProteinFeaturesLigand.forward`
(`model_utils.py:1246-1281`): fixed-chain sidechain atoms (last 32 of the 37-atom
representation, masked to only *fixed* residues via `1 - chain_mask`, `:1250-1252`) are
concatenated onto the ligand atom list, then re-selected down to `self.atom_context_num` via
a **second** top-k on squared Cβ-distance with sentinel `10000.0` (`:1272-1277`) — a stricter,
larger sentinel than stage 1's `1000.0`. Atom-type features: `Y_t` (element index from
`data_utils.element_dict`, 1..118, 0=unknown/pad) is expanded to one-hot(120) plus periodic
`group` (one-hot 19) and `period` (one-hot 8) lookups via the `periodic_table_features` table
(`:752-1120`, duplicated verbatim in `sc_utils.py:475-843`), concatenated to 147 dims and
projected by `type_linear: Linear(147,64)` (`:703,1294`) for the node-distance feature path,
while a *separate* `y_nodes: Linear(147, node_features, bias=False)` (`:705`) consumes the
full 147-dim one-hot directly for the ligand-node encoder path (`:1323`).

**10. Side-chain-context toggle** — `ligand_mpnn_use_side_chain_context`
(`ProteinMPNN.__init__` → `ProteinFeaturesLigand(..., use_side_chains=...)`,
`model_utils.py:34-43`) gates exactly the Stage-2 block in clause 9: when off, `Y`/`Y_t`/`Y_m`
are only the ligand atoms already selected in `featurize()`; when on, the last 32 atoms of the
37-atom fixed-residue sidechain representation (`xyz_37[:,:,5:,:]`, atom types hard-coded in
`self.side_chain_atom_types`, `:714-750`) are added as additional context atoms, restricted to
non-designed (`chain_mask==0`) residues only (`xyz_37_m * (1 - chain_mask[...,None])`,
`:1251`), then the combined atom set is re-truncated to `atom_context_num` by distance. `run.py`
separately forces `parse_all_atoms=True` in `parse_PDB` whenever this flag (or side-chain
packing without full repack) is set (`:195-197`), since the 37-atom representation is only
populated by `parse_PDB` under `parse_all_atoms=True` (`data_utils.py:735-776`).

**11. score_only: multi-order averaging is over `number_of_batches` independent randomized
decoding orders, averaged in probability space (not log-space)** — `score.py:280-314`: each of
`args.number_of_batches` iterations draws a fresh `feature_dict["randn"]` (`:281-284`), calls
`model.score`/`model.single_aa_score`, and appends `torch.exp(score_dict["log_probs"])`
(`:294`) to `probs_list`. After stacking over the batch/order axis, `mean_probs =
np.mean(out_dict["probs"], 0)` and `std_probs = np.std(out_dict["probs"], 0)` (`:314-315`) —
i.e. the reported "averaged" quantity is the **arithmetic mean of per-order probabilities**
(mean of `exp(log_probs)`), not the mean of log-probs and not a log-sum-exp. `logits`/
`log_probs` themselves are stacked but never averaged in the script (only `probs` is
summarized). No default count is enforced beyond the CLI's `--number_of_batches` (default 1),
and orders differ only in `randn`; `--autoregressive_score` runs the same-shaped `.score()`
path used for generation-style scoring while `--single_aa_score` (script default: on) runs the
per-position independent path (clause 15).

**12. Packer chi sampling: mixture-of-3 von Mises per chi angle (all 4 chis jointly per
step), sampled `num_samples` times, MAP-selected, iterated over denoising steps** —
`sc_utils.py:90-99` (`pack_side_chains`), matching LigandMPNN Alg. 14:
```
mean, concentration, mix_logits = model_sc.decode(feature_dict)      # each [B,L,4,num_mix]
mix  = Categorical(logits=mix_logits)
comp = VonMises(mean, concentration)
pred_dist = MixtureSameFamily(mix, comp)
predicted_samples = pred_dist.sample([num_samples])                  # [num_samples,B,L,4]
log_probs_of_samples = pred_dist.log_prob(predicted_samples)
sample = gather(predicted_samples, 0, argmax(log_probs_of_samples,0)[None])[0]  # per-angle MAP-of-draws
```
`concentration = 0.1 + softplus(torsions[...,1])` (`sc_utils.py:386`, matches LigandMPNN Alg.
14 line 14 exactly) — the `0.1` floor prevents zero concentration. `num_mix=3` is set by
`run.py:112` at Packer construction (not a `Packer` class default, which has no default for
`num_mix` other than the constructor arg default `3` — same value). There is **no temperature
parameter** exposed for packer sampling (unlike sequence sampling's explicit `/temperature`);
"temperature" here is implicit only in how many `num_samples` are drawn and MAP-selected
(`--sc_num_samples`, default 16, `run.py:922-927`) — sharper/more confident angles are favored
by drawing more candidates and keeping the highest joint log-prob one, not by scaling
`concentration`/`mean` directly. All 4 chi torsions are predicted and refined **simultaneously**
each of `num_denoising_steps` recycling iterations (`sc_utils.py:90-124`) — the reshape
`torsions.reshape(B,L,4,num_mix,3)` (`sc_utils.py:384`) has no autoregressive structure across
the 4 chi indices within a step (see Ambiguities re: paper text implying chi1→chi2→chi3→chi4
sequential decoding).

**13. Tied/symmetry positions** — `model_utils.py:350-467` (`ProteinMPNN.sample`, else-branch).
  - **(a) Effect of a fixed member on the tie group — order-dependent `S_t` aliasing.**
    After the joint amino acid is sampled (`S_t = multinomial(probs_sample,1)[:,0]`, `:451`),
    each member `t` in the group's list is processed in a **second** loop that reuses and
    reassigns the *same* scalar variable `S_t` (`:452-461`):
    ```
    for t in t_list:
        chain_mask_t = chain_mask[:, t]
        S_true_t = S_true[:, t]
        S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()   # OVERWRITES S_t
        h_S[:, t] = self.W_s(S_t)
        S[:, t] = S_t
    ```
    Because `S_t` is overwritten in place across iterations, a **fixed** member
    (`chain_mask_t==0`) earlier in `t_list` replaces `S_t` with its own `S_true_t`; every
    *subsequent* member in the same group — even a designed one with `chain_mask_t==1` —
    then computes `S_t = S_t(now = prior fixed member's native AA)*1 + S_true_t*0`, i.e. it
    inherits the earlier fixed member's native identity instead of the originally-sampled
    joint amino acid. If the fixed member is last in the user-supplied group order (or no
    member is fixed), this does not occur. `t_list` order is exactly the order given in
    `--symmetry_residues` (or, for `--homo_oligomer`, the order of `chain_letters_set`,
    `:342-358`), so this is order-of-input-dependent, not randomized.
  - **(b) Which member's bias is added.** `bias_t = bias[:, t]` (`:417`) is reassigned inside
    the *first* per-member loop and never re-gathered afterward; the fused sampling step
    (`:445-447`) uses whatever `bias_t` value survives from the **last** `t` iterated in
    `t_list` — i.e. only the last group member's `bias_AA_per_residue`/`omit_AA_per_residue`
    contributes to the shared draw, not a sum or average over the group.
  - **(c) How member logits are fused.** `total_logits = 0.0; for t in t_list: total_logits +=
    symmetry_weights[t] * logits` (`:413-443`) — a **weighted sum of raw logits** (not
    log-softmax outputs, not probabilities), then `softmax((total_logits + bias_t)/temperature)`
    once for the whole group (`:445-447`). Weights default to `1.0` per position
    (`symmetry_weights = ones([L])`, `:352`) unless overridden by `--symmetry_weights`, or set
    to `1/num_chains` uniformly under `--homo_oligomer` (`run.py:356`). Separately,
    `all_log_probs[:, t] = chain_mask_t[:,None] * log_probs` (`:440-442`) records **each
    member's own unfused** `log_softmax(logits)` — the returned `log_probs` for a tied
    position therefore does *not* match the distribution actually sampled from (which used
    `total_logits`, weighted-summed across the group).
  - **(d) Sequential decoding within a group / visibility of earlier members.** Members of a
    `t_list` are processed by a Python `for t in t_list` loop that calls the decoder layers
    once per member and writes into `h_V_stack[l+1][:, t:t+1, :]` incrementally (`:423-433`);
    each member's node hidden state (`h_V_stack`) is thus updated before the next member's
    decoder pass runs, so later group members' *node* computation can see earlier members'
    updated node state at that layer/position. However `h_S` (the sequence-token embedding
    used by `h_ES_t`/`cat_neighbors_nodes`) is **not** updated for any group member until after
    the whole group's joint `S_t` is resolved (the `h_S[:, t] = self.W_s(S_t)` writes happen
    only in the second, post-sampling loop, `:459`) — so no group member's decoder pass can see
    another member's *sampled identity* while computing logits, only the group's shared
    pre-group `h_S` state. Cross-group visibility (whether an already-decided *other* group
    precedes this one) is governed by the same `mask_bw`/`mask_fw` machinery as clause 15,
    built once from the full `new_decoding_order` (`:412` outer loop, masks built at
    `:370-382`).
  - Paper coverage: *Science* p.3 describes only the *intent* ("predicting logits for A1 and
    B1 first and then combine... to construct a normalized probability distribution from which
    a joint amino acid is sampled" and "averaging logits" for homooligomers) — the paper text
    does not specify weighting, bias handling, or per-member log-prob bookkeeping; (a)-(d)
    above are code-only findings not stated in either paper.
  - `model.score()`'s own symmetry handling (`:600-631`) only builds `mask_bw`/`mask_fw` from
    the grouped `new_decoding_order` to control the encoder/decoder visibility mask; it does
    **not** perform any per-group logit fusion or joint sampling — `h_S` there is built
    directly from ground-truth `S_true` (`:638`), never from a jointly-sampled `S_t`. So "tied
    positions" mean something structurally different in `.score()` (shared visibility only)
    versus `.sample()` (visibility + logit fusion + joint draw).

**14. Encoder/decoder message-passing equations** — `EncLayer.forward`
(`model_utils.py:1716-1739`), `DecLayer.forward` (`:1670-1691`), `DecLayerJ.forward`
(`:1598-1621`), all matching LigandMPNN Alg. 7 (`EncLayer`), Alg. 8 (`DecLayer`), Alg. 9
(`DecLayerJ`/`DecLayerJ`≈"Context Decoder Layer") exactly:
```
# node update (Enc/Dec/DecLayerJ all share this form)
h_EV = cat([h_V.expand(...), h_E], -1)                       # [B,L,K,2H] or [B,L,M,2H] for DecLayerJ
h_message = W3(GELU(W2(GELU(W1(h_EV)))))                     # 3-layer MLP, GELU nonlinearity
if mask_attend is not None: h_message = mask_attend[...,None] * h_message
dh = sum(h_message, dim=-2) / scale                          # scale = 30.0 (default arg), FIXED constant
h_V = LayerNorm(h_V + Dropout(dh))                            # post-norm (residual THEN norm)
dh = PositionWiseFeedForward(h_V)                              # Linear(H,4H) -> GELU -> Linear(4H,H)
h_V = LayerNorm(h_V + Dropout(dh))
if mask_V is not None: h_V = mask_V[...,None] * h_V
```
Edge update (`EncLayer` only, `:1734-1739`), structurally identical MLP shape but **not**
divided by `scale`:
```
h_EV = cat([h_V.expand(...), h_E], -1)     # h_V now the POST-update node state
h_message = W13(GELU(W12(GELU(W11(h_EV)))))
h_E = LayerNorm(h_E + Dropout(h_message))   # no /scale here
```
`scale` defaults to `30.0` in all three layer classes' constructors (`:1582,1654,1695`) and is
**not** rescaled by the true neighbour count (even when the clause-6 `min(top_k, L)` clamp
means fewer than `top_k` neighbours are actually summed) — division is always by the fixed
constant 30, not by the number of unmasked neighbours. `Dropout` is instantiated with whatever
`dropout` is passed to `ProteinMPNN.__init__`; `run.py:77-88` never passes `dropout=`, so the
class default `dropout=0.0` (`model_utils.py:21`) is used at inference — dropout layers exist
in the graph but are no-ops (`nn.Dropout(0.0)`), and `model.eval()` (`run.py:92`) additionally
disables any train-mode-only behavior (none is train-mode-conditional here except this).
`ProteinFeatures`/`ProteinFeaturesLigand`/`ProteinFeaturesMembrane` all use `LayerNorm` after
every linear embedding (`norm_edges`, `norm_nodes`, `norm_y_edges`, `norm_y_nodes`, `V_C_norm`)
— consistently post-projection, pre-residual-consumption.

**15. Autoregressive visibility mask** — construction (`model_utils.py:223-235`, non-symmetric
branch; `:370-382` symmetric branch — identical formula):
```
P = one_hot(decoding_order, num_classes=L)                                  # [B,L,L] permutation
order_mask_backward = einsum("ij,biq,bjp->bqp", 1 - triu(ones(L,L)), P, P)
# order_mask_backward[q,p] = 1 iff position p is decoded strictly BEFORE position q
mask_attend = gather(order_mask_backward, 2, E_idx)          # per-neighbour lookup, [B,L,K]
mask_bw = mask[...,None,None] * mask_attend                  # visible: neighbour already decoded
mask_fw = mask[...,None,None] * (1 - mask_attend)            # not-yet-decoded: structure only
```
At each generation step the decoder receives `h_ESV_t = mask_bw_t * h_ESV_decoder_t +
h_EXV_encoder_t` (`:302`, and per-layer inside the `t_`-loop `:296-307`): the `mask_bw` branch
carries `h_ES` built from the *actual, currently-scattered* `h_S` (sequence embedding, zero
until a position is decoded, `:251,337-341`) for already-decided neighbours, while the
`mask_fw` branch (`h_EXV_encoder_fw`, precomputed once from the encoder output with `h_S`
replaced by `zeros_like(h_S)`, `:258-260`) supplies the not-yet-decided neighbours' purely
structural (identity-blind) representation. `single_aa_score` (`:501-506`) reuses this same
mask machinery but derives `decoding_order` from a **position-specific** `order_mask` instead
of `chain_mask`:
```
if not use_sequence:
    order_mask = zeros(L); order_mask[idx] = 1.
else:
    order_mask = ones(L);  order_mask[idx] = 0.
```
Given the "smaller key ⇒ decoded earlier ⇒ others see it, it sees nothing yet" semantics
established above (and empirically consistent with the `chain_mask` convention, where
chain_mask=0/fixed gets the *small* key and is decoded first specifically so everyone else can
condition on it): setting `order_mask[idx]=1` (large key ⇒ `idx` decoded **last** ⇒ `idx` sees
*every* other position, since all of them then satisfy "decoded before idx") happens in the
`not use_sequence` branch, and setting `order_mask[idx]=0` (small key ⇒ `idx` decoded
**first** ⇒ `idx` sees *nothing*, mask_bw≈0 for its row) happens in the `use_sequence` branch.
That is, by this derivation, **`use_sequence=True` yields the backbone-only score and
`use_sequence=False` yields the full-context (all-other-true-residues) score** — the reverse
of the parameter's own docstring ("`use_sequence` - False using backbone info only",
`model_utils.py:471-474`) and CLI help (`score.py:526-531`: "1 - get scores using amino acid
sequence info; 0 - get scores using backbone info only"). This is a direct reading of the
`order_mask`/`argsort`/`einsum` chain, not an assumption; it has not been verified by running
the code. Contrast with `model.score()`'s *own*, differently-implemented `use_sequence` switch
(`:646-654`), which is unambiguous because it is a plain `if/else` on which term feeds the
decoder (`h_EXV_encoder_fw` alone vs. `mask_bw*h_ESV + h_EXV_encoder_fw`) rather than a
decoding-order trick, and is not subject to the same inversion. Paper: LigandMPNN Alg. 12
(`LigandMPNN_decode`) encodes the same idea as a single explicit `causal_mask =
upper_triangular[decoding_order](L,L)` applied once (matching `.score()`'s one-shot style,
not the per-step `t_`-loop gather/scatter used by `.sample()`/generation) — the paper does not
describe `single_aa_score` at all (it is not one of Algorithms 1–14).

---

## Ambiguities

1. **Tie-break at equal top-k distances is genuinely unspecified.** `torch.topk`
   (`largest=False`, no `sorted=` control relevant to ties) and `torch.argsort` (no
   `stable=True`) are both used without any documented tie-break contract; the packet code
   never sorts by a secondary key (e.g. residue index) to make ties deterministic. Recorded as
   ambiguity, not assumed to be "stable" or "arbitrary" — PyTorch does not guarantee either
   without `stable=True`.

2. **`single_aa_score`'s `use_sequence` polarity (clause 15) reads as inverted relative to its
   own docstring/CLI text, and relative to `model.score()`'s same-named, differently-coded
   flag.** This is derived from static reading of the `order_mask`/`argsort`/`einsum` chain,
   corroborated by the `chain_mask` convention used everywhere else in the file (fixed=0 sorts
   first). It has **not** been confirmed by executing the code (out of scope for this
   reconstruction) or against the paper (which does not cover `single_aa_score`). Recorded as
   a specific, cited candidate discrepancy rather than a guess about intent.

3. **The `S_t`-aliasing behavior for tied groups with a fixed member (clause 13a) is
   order-of-input dependent** and only manifests when a fixed member precedes a designed
   member in the user-supplied `--symmetry_residues` group ordering. Whether this is intended
   (e.g. "fixed members should always be listed last") or an oversight is not stated anywhere
   in the packet; no docstring or help text in `run.py` mentions ordering constraints on
   `--symmetry_residues` groups relative to `--fixed_residues`.

4. **Membrane label collision (buried=1 AND interface=1 simultaneously) silently produces
   class 0** (`run.py:270-272`: `2*buried*(1-interface) + 1*interface*(1-buried)` evaluates to
   0 when both are 1), which is indistinguishable from "neither flag set." Nothing in `run.py`
   validates that `--transmembrane_buried` and `--transmembrane_interface` are disjoint sets.

5. **What "temperature" means for packer sampling** is not a literal scalar anywhere in
   `sc_utils.py`/`model_utils.py`'s Packer path — only `--sc_num_samples` (draw-and-MAP-select
   count) and `--sc_num_denoising_steps` (recycling count) are exposed. Whether the
   reconstructed method should treat these as a temperature-equivalent, or note the packer has
   no temperature control at all, is left as recorded ambiguity rather than resolved by
   analogy to sequence sampling's explicit `/temperature`.

6. **Chi-angle decoding: paper text vs. code structure (clause 12).** LigandMPNN's Methods
   prose states the model "autoregressively decompose[s] the joint chi angle distribution by
   decoding all chi1 angles first, then all chi2 angles, and finally all chi4 angles (after the
   model decodes one of the chi angles, its angular value and the associated three-dimensional
   atom coordinates are used for further decoding)" — describing an autoregressive-by-chi-index
   scheme. The packet's `pack_side_chains`/`Packer.decode` instead predicts and updates all
   four chi angles **simultaneously** each denoising/recycling step (`torsions.reshape(...,4,
   num_mix,3)` covers all 4 chis in one `W_torsions` call per step), with autoregression only
   across the `num_denoising_steps` recycling loop (structure-conditioned refinement), not
   across chi indices within a step. Recording both descriptions rather than resolving which
   is authoritative — the packet does not include a lower-level decode function that iterates
   per-chi-index within a single step.

7. **`randn`'s "only the first entry is used" comment vs. actual code path.** The docstring at
   `model_utils.py:199-201` says "only the first entry is used since decoding within a batch
   needs to match for symmetry," but the non-symmetric branch's `decoding_order =
   argsort((chain_mask+0.0001)*abs(randn))` (`:218-220`) is computed once, before any
   `B_decoder`-repeat (`:222,238-243`), so the *whole* `[B=1, L]` `randn` row is used for that
   single decoding order shared across the repeated `B_decoder` batch — consistent with the
   comment for the between-samples-in-a-batch case, but note `feature_dict["randn"]` itself is
   freshly redrawn once per `--number_of_batches` outer iteration (`run.py:420-423`), so
   different *batches* (in the `--number_of_batches` sense) do get different decoding orders,
   while all samples *within* one `--batch_size` batch share the same order. Left as a
   terminology note rather than folded into clause 1 to avoid overstating certainty about
   intended vs. incidental behavior.

## Assumptions

- Where `protein_mpnn_utils.py` (8907e667) and the LigandMPNN packet's `model_utils.py`
  (26ec57ac) implement the same primitive (RBF, `_dist` clamp-and-sentinel, `PositionalEncodings`
  cap of 32, `EncLayer`/`DecLayer`/`DecLayerJ` `/scale=30` form, `argsort`-based decoding order),
  the reconstruction cites the LigandMPNN packet copy as canonical, since `run.py` (the only
  driver in the packet) imports from `model_utils.py`, not from `protein_mpnn_utils.py`
  directly — `protein_mpnn_utils.py` is treated as corroborating/ancestor evidence, not a
  separately-exercised code path.
- "Default" values quoted throughout (e.g. `augment_eps=0.0`, `dropout=0.0`, `scale=30.0`,
  `num_rbf`/RBF bin ranges `[2.0,22.0]`) are the values that are actually reached by `run.py`'s
  call graph at inference (i.e. accounting for which constructor arguments `run.py` does and
  does not override), not merely the class-definition defaults in isolation — these happened to
  coincide for every parameter checked in this reconstruction.
- `checkpoint["atom_context_num"]` and `checkpoint["num_edges"]` (`run.py:69,71,75`) are
  treated as opaque checkpoint-supplied integers (their concrete values, e.g. 16/25/32, are not
  stated anywhere in the packet code or papers read) — clauses citing `atom_context_num`/`k`
  describe the *formula*, not a specific numeric default, except where a literal is hard-coded
  in a call site (e.g. `run.py:496`'s literal `16` for side-chain packing context, or
  `sc_utils.py`'s `Packer(..., atom_context_num=16, ...)` at `run.py:104`).
