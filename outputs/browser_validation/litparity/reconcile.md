# Literature parity, Phase 2 (reconcile)

task_id: `260923_aminx-browser-validation`, Mode A
Reference: dauparas/LigandMPNN@26ec57ac (+ ProteinMPNN@8907e667)
Inputs: `recon_math.md`, `recon_algo.md`, `recon_protocol.md`, `reference_packet/`, `impl_packet/` (byte-identical to `src/aminx/` at this worktree's HEAD), plus `src/aminx/**` outside the impl paths.
Output: `clauses.json` (40 clauses). Every verdict was checked against the aminx source at the cited `path:line`. Nothing was executed. Where a consequence depends on numerical magnitude, the note says it is a static reading.

## 1. Cross-reconstruction comparison

Each row lists where the three blind reconstructions agree or disagree, which one diverged, and how the reference code settles it.

| # | Topic | Agreement | Disagreement and resolution |
|---|---|---|---|
| 1 | Default decoding order | All three give `argsort((mask*chain_mask + 1e-4)*|randn|)`, fixed-first, with a fresh `randn` per `number_of_batches` draw. | **Strength of fixed-first.** math says it is "not a hard guarantee". protocol says fixed residues are "always" decoded first. algo adds that masked (missing) residues sort first too. **math is right:** the order of a pair flips when `|randn_designed| < ~1e-4 |randn_fixed|`, which has a per-pair probability of order 1e-4. algo's point about masked residues is correct (`chain_mask = mask*chain_mask`, `model_utils.py:217`). |
| 2 | Temperature and bias | All three give `softmax((logits + bias)/T)` for `model_utils.py`. | algo and protocol record that ProteinMPNN 2022 (`protein_mpnn_utils.py:1165-1179`) applies T first, does not temper the -1e8 omit, and applies the per-residue omit after softmax. **Resolved:** `run.py` drives `model_utils.py`, so the LigandMPNN formula is canonical for this pin. The two formulas differ literally but not numerically at sane T. |
| 3 | X at sampling | All three: hard-omitted via `probs[:, :20]` renormalisation. | algo and protocol note that the ProteinMPNN ancestor samples all 21 classes unless X is omitted. protocol adds that `score()`/`single_aa_score()` keep X in the 21-way output. **Resolved:** the LigandMPNN pin hard-omits. The ancestor difference matters only for the P00 lane. |
| 4 | Omit as bias | All three: a single additive bias of -1e8. | No disagreement. |
| 5 | augment_eps | All three: 0 at inference (never passed). | algo notes there is no `self.training` guard in the design feature classes, but there is one in the packer. protocol notes the 2022 class default is 0.05. Both are consistent with an inference value of 0. |
| 6 | k clamp | All three: `min(k, L)`. | protocol adds that k is read from `checkpoint["num_edges"]` and that the per-checkpoint values are only inferred from filenames. |
| 7 | top-k ties | All three: unspecified (no stable flag, no secondary key). | No disagreement. The reference cannot settle it, so the verdict is **AMBIGUOUS**. |
| 8 | Membrane labels | All three give {0 neither, 1 interface, 2 buried}. A residue with both flags collapses to 0. The global label is a broadcast scalar. The labels enter only as a node feature. | No disagreement. |
| 9 | Ligand atom context | All three: selection is by count, not by distance cutoff. `cutoff_for_score` gates only the reported ligand-proximal score. | Only protocol pins the number: 25, from the filename and the paper, noted as not code-confirmed. The packer's hard-coded 16 is noted by protocol. |
| 10 | Side-chain context | All three: side chains of fixed residues from the 16-nearest sub-graph are concatenated and then re-selected. Default off. | algo adds the explicit `E_idx[:, :16]` detail. No disagreement. |
| 11 | score_only | All three: mean of `exp(log_probs)` in probability space over `number_of_batches` draws. The CLI default is `single_aa_score`. | **Polarity of `use_sequence` in `single_aa_score`.** math asserts it is inverted relative to the docstring. algo and protocol hedge and leave it as an open question. **Resolved by the reference code, not left AMBIGUOUS.** The einsum `M[q,p] = 1 iff rank(q) > rank(p)` together with ascending argsort puts `idx` first when `use_sequence=True`, so `idx` sees no sequence and the result is backbone-only. ProteinMPNN `conditional_probs(backbone_only=True)` (`protein_mpnn_utils.py:1323-1326`) builds the identical `order_mask[idx]=0`, which confirms math's reading. See clause `unconditional_scoring_backbone_only`. |
| 12 | Packer | All three: 3-component von Mises mixture, 16 draws, per-angle argmax of log-prob, 3 recycling steps, no temperature. | math and protocol flag the paper's chi1→chi4 autoregression against the code's joint per-step update. algo is silent. The **code is the parity target**. |
| 13a | Tied: fixed member | — | **Three-way divergence.** math reports **S_t aliasing**: the second loop reassigns `S_t`, so a designed member listed after a fixed member inherits that fixed member's native (`model_utils.py:457-461`). algo says fixed members keep their own native and "tying is enforced only across designed members". It names the variable `S_t_repeat` and so misses that it is overwritten. protocol says every member is "assigned identically" and misses the fixed override entirely. **math verified correct.** |
| 13b | Tied: bias | All three: only the last member's bias survives (loop-variable leak). | No disagreement. |
| 13c | Tied: fusion | All three: weighted sum of raw logits, with one softmax per group. | **Default weights.** math says weights "default to 1.0". protocol says a missing `--symmetry_weights` raises `IndexError`. **protocol is right** (`run.py:331-337` gives `[[]]`, which is indexed at `model_utils.py:352-355`). math's `ones([L])` is only the initial value for untied positions. algo flags that the paper's "averaged" wording matches ProteinMPNN `tied_sample` (`/len(t_list)`), not LigandMPNN. |
| 13d | Tied: intra-group visibility | math and algo: members get sequential decoder passes. A later member sees an earlier member's updated `h_V_stack` through `mask_bw`. `h_S` is not written until after the joint draw. | protocol says members are "decoded simultaneously, not sequentially". That is true of the single joint draw but not of the decoder passes. math and algo are correct. |
| 14 | Message passing / AR mask | math and algo agree: divisor fixed at 30, post-norm, no scale on the edge update, and self always falls in `mask_fw`. | No disagreement. |
| — | Seeds (protocol only) | `--seed 0` is falsy and means a random seed. | Not a parity clause: aminx uses explicit PRNG keys and no token-level agreement is claimed. Recorded here only. |
| — | `--homo_oligomer` order (reconciler) | — | `run.py:342` uses `list(set(chain_letters_list))`. With Python string-hash randomisation, the `t_list` member order varies from run to run, so behaviours 13a and 13b are hash-seed dependent under `--homo_oligomer`. |

## 2. Findings not covered by any reconstruction

These came from mapping to aminx. All of them are core DEVIATIONs.

- **`ligand_atom_context`.** `get_topology_for_checkpoint` parses `atom_context_num=25`, but `load_model` never passes it to `PrxteinLigandMPNN`. `ProteinFeaturesLigand` therefore keeps its default of 16 (`ligand_mpnn.py:170-178`, `ligand_features.py:220`).
- **`positional_encoding_ligand`.** The ligand `PositionalEncodings` has no bias, and conversion drops the checkpoint's bias. Inter-chain edges are encoded two-hot instead of as the reference's single index 65.
- **`conditional_score_context_mask`.** aminx's single-pass `1 - I` conditional score lets a residue's own identity return to it through 2-hop decoder paths. The reference's per-residue, idx-last passes cannot leak this way.
- **`membrane_label_encoding` and `weight_conversion_bias_handling`.** The membrane projection keeps a random-initialisation bias because the reference layer has no bias. `convert_linear_layer` silently reconciles bias mismatches in both directions.
- **`tied_scoring_logit_fusion`.** aminx scoring applies product-of-experts (PoE) fusion to tied logits. The reference `score()` only regroups the decoding order.

## 3. Verdict summary

| Verdict | n | Elements |
|---|---|---|
| MATCH | 19 | temperature_on_logits_plus_bias, omit_as_bias, augment_eps, knn_k_clamp, side_chain_context_toggle, ligand_context_fusion, unconditional_scoring_backbone_only, score_nll_full_vocab, packer_mixture_parameterisation, packer_graph_and_context_size, tied_group_step_placement, ar_visibility_mask, undrawn_sequence_embedding, message_passing_encoder, message_passing_decoder, dropout_off_at_inference, rbf_edge_features, positional_encoding_protein, temperature_default |
| DEVIATION | 16 | 6 designed (core=false): decoding_order_default, x_omit_at_sampling, tied_fixed_override, tied_last_member_bias, tied_fusion_weights, tied_intra_group_visibility. 10 unexpected (core=true): membrane_label_encoding, ligand_atom_context, conditional_score_context_mask, tied_scoring_logit_fusion, sample_log_probs_output, random_order_fixed_first, positional_encoding_ligand, knn_masked_pair_sentinel, checkpoint_topology_source, weight_conversion_bias_handling |
| MISSING | 4 | membrane_label_construction, ligand_cutoff_for_score, score_only_multi_order_average, packer_chi_sampling |
| AMBIGUOUS | 1 | topk_tie_break |

- **Core.** 34 of 40 clauses are core: every clause lists an in-scope P-ID, and the six designed deviations are excluded. The `core` flag was checked mechanically against the rule.
- **Parity numerator and denominator.** MATCH ∩ core = 19. There are 33 core clauses that are not AMBIGUOUS, so `clause_parity_pct` = 19/33 = 0.576, which is 0.076 above the 0.5 ceiling.
- **Ambiguity.** `n_ambiguous` = 1. `ambiguity_load` = **load_bearing**, because `topk_tie_break` is core and lists P03, P11 and P14.

## 4. Where a reference-compatible mode exists

- **Decoding order.** Pass a WaveScheduleBundle built on the host from the reference argsort order (`build_inference_bundle(wave=...)`).
- **X omission.** Set `bias[:, 20] = -1e8`. This is exact.
- **Omit and per-residue bias.** Build the reference bias tensor on the host.
- **Tied scoring.** Pass no tie groups.

There is no compatible mode for the three tied-sampling semantics (fixed override, last-member bias, weighted raw-logit fusion) or for multi-order averaging. Both the conditional estimand and packer sampling need an external harness.

## 5. Flags for Phase 3

Phase 3 should adversarially confirm the following with narrow local probes:

1. The atom_context 16 vs 25 wiring: `load_model("ligandmpnn_v_32_010_25").features.atom_context_num`.
2. The ligand positional-encoding bias, by comparing `features.embeddings.linear.bias` in the `.pt` against the `.eqx` leaves.
3. The 2-hop self-identity leak in `score_conditional`: perturb `s_i` only and check whether `logits[i]` changes under `full_context_ar_mask`.
4. The membrane random bias.
5. `knn_masked_pair_sentinel` on a gapped chain with fewer than k valid partners per row.
