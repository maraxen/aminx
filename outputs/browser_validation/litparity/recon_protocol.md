# Blind Reconstruction — Lens: EXPERIMENTAL PROTOCOL

task_id: 260923_aminx-browser-validation — literature-parity Phase 1 (blind reconstruction)

## Lens

This reconstruction focuses on the experimental-protocol surface: CLI defaults and
hyperparameters relevant to inference, model variants/checkpoints, input-parsing
conventions, output conventions, and the defaults a user gets without flags. It does
not attempt to reconstruct the neural architecture's internal math beyond what is
needed to explain a protocol-relevant default (e.g. why a value is divided by
temperature before or after a bias term).

## Sources read

All paths below are relative to the reference packet directory (no other files, in
or out of that directory, were opened):

- `run.py` (full, 959 lines) — main sequence-design/inference CLI, commit 26ec57ac
- `score.py` (full, 549 lines) — scoring CLI, commit 26ec57ac
- `model_utils.py` (full, 1772 lines) — `ProteinMPNN` class (encode/sample/score/
  single_aa_score), `ProteinFeatures`, `ProteinFeaturesLigand`,
  `ProteinFeaturesMembrane`, layer classes, commit 26ec57ac
- `data_utils.py` (full, 988 lines) — `parse_PDB`, `featurize`, `get_score`,
  `get_seq_rec`, `write_full_PDB`, alphabet/element tables, commit 26ec57ac
- `sc_utils.py` (full, 1157 lines) — `Packer`, `pack_side_chains`,
  `make_torsion_features`, side-chain `ProteinFeatures`, commit 26ec57ac
- `protein_mpnn_utils.py` (targeted: `tied_featurize` signature, `ProteinFeatures`/
  `CA_ProteinFeatures`, `ProteinMPNN.__init__`, `.sample`, `.tied_sample`,
  `.conditional_probs`, alphabet string) — original 2022 ProteinMPNN, commit
  8907e667
- `proteinmpnn_2022.pdf`, pages 2–4 (Dauparas et al., *Science* 2022, PMC
  manuscript pagination)
- `ligandmpnn.pdf`, pages 2–4 (Dauparas et al., *Nature Methods* 22, 717–723,
  2025)

No `run.py`/inference-script equivalent for the original 2022 ProteinMPNN codebase
was in the packet, so its own CLI-level defaults (omit_AAs_np, bias_AAs_np
construction, etc.) could not be reconstructed — only what is visible from
`protein_mpnn_utils.py`'s function signatures and body.

## Clauses

**1. Default decoding order and any flag changing it; fixed-first rule.**
No dedicated `--decoding_order` flag exists. The order is derived per forward pass
from `chain_mask` and a fresh random vector:
`decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))`
(`model_utils.py:218-220` in `sample()`; identical formula in `score()` at
`model_utils.py:582-584`, and in `single_aa_score()` at `model_utils.py:507-509`
with `chain_mask` replaced by a per-call `order_mask`). Because `chain_mask` is 0.0
for fixed residues and 1.0 for designed ones, fixed positions get a weight near
0.0001·|randn| while designed positions get weight ≈|randn|; `argsort` (ascending)
therefore places fixed residues first in the decode sequence, i.e. **fixed
residues are always decoded before designed ones** ("fixed-first"), matching the
paper's statement that order-agnostic decoding "skips the fixed regions but
includes them in the sequence context for the remaining positions" (paper:
`proteinmpnn_2022.pdf` p.3, "To enable application to a broad range... order
agnostic autoregressive model"). `--symmetry_residues` (run.py:774-779) and
`--homo_oligomer` (run.py:786-791) do not change this base ordering but *regroup*
it: `model_utils.py:357-368` walks the base `decoding_order` and merges any tied
residues into one joint decode step at the position of the first-encountered
member, so symmetric residues that would otherwise decode at different times are
forced to decode simultaneously.

**2. Temperature default(s) and whether bias is added before temperature.**
`--temperature` default `0.1` (`run.py:826-830`). `score.py` has no temperature
flag (scoring is not sampled). In `model_utils.py` `sample()` the combined
bias/omit tensor is added to raw logits **before** dividing by temperature:
`probs = softmax((logits + bias_t) / temperature)` (`model_utils.py:317-319`,
non-symmetric branch; `model_utils.py:445-447`, symmetric branch uses
`total_logits` in place of `logits`). `bias_t` itself is
`feature_dict["bias"] = (-1e8*omit_AA + bias_AA).repeat([1,L,1]) + bias_AA_per_residue - 1e8*omit_AA_per_residue`
(`run.py:404-408`), i.e. the `-1e8` omission penalty is *also* divided by
temperature under this formula.
**Ambiguity/contrast with the original 2022 code:** `protein_mpnn_utils.py:1165-1166`
computes `logits = self.W_out(h_V_t) / temperature` first, then
`probs = softmax(logits - constant*1e8 + constant_bias/temperature + bias_by_res_gathered/temperature)`
— here the `-1e8` omission term is **not** divided by temperature (only the raw
logits and the two *bias* terms are divided by T). Both formulations are numerically
indistinguishable in practice (constant·1e8 dominates regardless of T), but they are
literally different formulas; a reimplementation that "adds bias then divides by T
everywhere" is faithful to the LigandMPNN repo's `sample()` but not literally
identical to the original 2022 `sample()`'s arithmetic. Recorded as ambiguity, not
resolved.

**3. X at sampling: default omit list, hard-coded omissions.**
`--omit_AA` default `""` (`run.py:756-760`) — no amino acids omitted by the
user-facing flag by default. Independent of that flag, `model_utils.py`'s
`sample()` **always** truncates the 21-way softmax to the first 20 classes before
building the multinomial:
`probs_sample = probs[:, :20] / probs[:, :20].sum(...)  # hard omit X` (comment in
source, `model_utils.py:320-322`, and symmetric branch `model_utils.py:448-450`).
Since `restype_str_to_int["X"] == 20` (`data_utils.py:32-54`), this hard-slices out
exactly the unknown-residue class in *both* decoding branches, unconditionally, for
every model type. This makes explicitly adding `"X"` to `--omit_AA` a no-op for
`sample()` (it is already excluded by the slice) — flagged as an ambiguity, not
verified against any test. This truncation is **absent** from `score()` and
`single_aa_score()` (`model_utils.py:656-657`, `543-544`): those report
`log_softmax` over the full 21-way distribution, so `probs`/`log_probs` returned by
`score.py` *do* include an X column, unlike `run.py`'s generated sequences which
can never contain X.

**4. omit_AA / omit_AA_per_residue / bias_AA flags, formats, and the numeric omit value.**
- `--omit_AA` (str, default `""`): concatenated letters, e.g. `"ACG"`
  (`run.py:755-760`); converted to a boolean mask over `alphabet`
  (`run.py:177-181`).
- `--omit_AA_per_residue` (path to JSON, default `""`): `{"A12": "PQR", "A13": "QS"}`
  (`run.py:761-766`); `--omit_AA_per_residue_multi` is the per-PDB dict version,
  `{"pdb_path": {"A12": "APQ", ...}}` (`run.py:767-772`). Encoded residue keys are
  `chain_letter + R_idx + insertion_code` strings (`run.py:205-215`).
- `--bias_AA` (str, default `""`): colon/comma format `"A:-1.024,P:2.34,C:-12.34"`
  (`run.py:736-741`), parsed into a length-21 float tensor indexed by
  `restype_str_to_int` (`run.py:146-152`).
- `--bias_AA_per_residue` / `--bias_AA_per_residue_multi` (path to JSON, default
  `""`): `{"A12": {"G": -0.3, "C": -2.0, "H": 0.8}}` (`run.py:742-753`).
- The numeric omission penalty is a hard-coded constant `1e8` subtracted from the
  logit before softmax (`run.py:404-408`, `model_utils.py:317-319`), not a CLI
  flag — same magnitude used for both `omit_AA` and `omit_AA_per_residue`.
  `score.py` has none of the omit/bias flags at all — only `run.py`'s `sample()`
  path supports biasing/omission.

**5. augment_eps / backbone noise defaults per model.**
The `ProteinMPNN` class in `model_utils.py` (`model_utils.py:20`) declares
`augment_eps=0.0` as its constructor default, and neither `run.py:77-88` nor
`score.py:60-71` ever pass an `augment_eps` argument when instantiating the
model — so **at inference, backbone/ligand coordinate noise is always exactly
0.0**, for every `model_type` (`ProteinFeatures.forward`, `model_utils.py:1392-1393`;
`ProteinFeaturesLigand.forward`, `model_utils.py:1188-1191`; both gated on
`if self.augment_eps > 0`). The side-chain `Packer` is likewise instantiated with
`augment_eps=0.0` explicitly (`run.py:109`).
**Contrast:** the original 2022 `ProteinMPNN.__init__` (`protein_mpnn_utils.py:1022`)
defaults to `augment_eps=0.05`, and the LigandMPNN paper states its released
checkpoint was *trained* with 0.1 Å Gaussian coordinate noise ("The baseline model
was trained with 0.1 Å standard deviation noise... training with 0.05 Å and 0.2 Å
noise instead increased and decreased sequence recovery by about 2%,"
`ligandmpnn.pdf` p.719); the ProteinMPNN paper similarly reports training with
0.02 Å noise for its AlphaFold-robust variant (`proteinmpnn_2022.pdf` p.4,
"Training with backbone noise improves model performance for protein design").
These are training-time noise levels baked into the checkpoint weights, not
reproduced at inference — none of the provided scripts re-apply noise at
inference time; `augment_eps` is purely a training-era hyperparameter for the
released checkpoints as far as this packet shows.

**6. k (number of neighbours) per checkpoint and the min(k, L) clamp.**
`k_neighbors` is not chosen by the user; both `run.py:71,75` and `score.py:54,58`
read it from the checkpoint file itself: `k_neighbors = checkpoint["num_edges"]`,
then forwarded into `ProteinMPNN(..., k_neighbors=k_neighbors, ...)`
(`run.py:77-88`). The class-level defaults (`ProteinFeatures` top_k=48,
`model_utils.py:1338`; `ProteinFeaturesLigand` top_k=30, `model_utils.py:675`;
`ProteinFeaturesMembrane` top_k=48, `model_utils.py:1456`) are therefore
overridden at runtime by whatever the checkpoint stores and are never hit for
`run.py`/`score.py` usage. Checkpoint *filenames* suggest (not code-confirmed,
since the number lives inside the `.pt` file) k=48 for
`proteinmpnn_v_48_020.pt`, `per_residue_label_membrane_mpnn_v_48_020.pt`,
`global_label_membrane_mpnn_v_48_020.pt`, `solublempnn_v_48_020.pt`
(`run.py:662-689`), and k=32 for `ligandmpnn_v_32_010_25.pt` — corroborated by
the paper's Fig. 1 caption: "32 closest protein residues" for LigandMPNN's
protein graph (`ligandmpnn.pdf` p.717 figure). The side-chain `Packer` is the one
place `top_k` is set directly in `run.py` rather than from a checkpoint:
`top_k=32` explicit (`run.py:107`), overriding the `Packer`/`ProteinFeatures`
class default of 30 (`sc_utils.py:240,399`). Every neighbour-selection routine
clamps k to sequence length: `torch.topk(D_adjust, np.minimum(self.top_k,
X.shape[1]), dim=-1, largest=False)` — present in `model_utils.py:1155-1156`
(ligand), `model_utils.py:1361-1363` (protein/membrane, shared code), and
`sc_utils.py:852-854` (packer), and identically in the original code at
`protein_mpnn_utils.py:849` and `:943`.

**7. Top-k tie handling; determinism flags/seeds.**
No explicit tie-breaking is coded anywhere in the packet — `torch.topk(...,
largest=False)` is used as-is for neighbour selection (`model_utils.py:1154-1156`
etc.) with no `sorted=` argument override and no secondary sort key (e.g. by
residue index) to break distance ties deterministically. No call to
`torch.use_deterministic_algorithms` or `torch.backends.cudnn.deterministic`
appears in any of the five source files. **Ambiguity:** whether tied distances
resolve identically across CPU/GPU or across PyTorch versions is not
knowable from source alone; likewise `torch.multinomial` (used for sampling,
`model_utils.py:323`, `:451`) has known CPU/GPU algorithm differences, so
"same seed ⇒ same output" is not guaranteed across devices — see clause 13.

**8. Membrane labels: flags, per-residue vs global, encoding values.**
Two membrane model types share one 3-class per-residue label scheme
(`ProteinFeaturesMembrane`, `num_classes=3`, `model_utils.py:1449-1478`,
one-hot embedded via `self.node_embedding`, `model_utils.py:1572-1576`):
- `per_residue_label_membrane_mpnn`: per-residue labels built from two
  space-separated residue-list flags, `--transmembrane_buried` and
  `--transmembrane_interface` (`run.py:868-878`), encoded as
  `membrane_per_residue_labels = 2*buried*(1-interface) + 1*interface*(1-buried)`
  (`run.py:270-272`): **0 = neither (exposed/soluble), 1 = interface, 2 =
  buried**. Default (no flags given) is all-zeros (`run.py:259-260,268-269`).
- `global_label_membrane_mpnn`: a single scalar `--global_transmembrane_label`
  (int, default `0`) is broadcast to every residue via
  `args.global_transmembrane_label + 0*fixed_positions` (`run.py:274-277`); help
  text: "1 - transmembrane, 0 - soluble" (`run.py:880-885`). Default is
  soluble (0) for every residue.
Both flags/paths feed the same `ProteinFeaturesMembrane` node embedding at
inference — the distinction between "per-residue" and "global" is entirely in how
`protein_dict["membrane_per_residue_labels"]` is populated in `run.py`/`score.py`,
not in the model class itself.

**9. ligand_mpnn: atom context num default, cutoff, use_side_chain_context flag default.**
- Atom-context count (`atom_context_num`) is **not** a CLI flag for `run.py`'s
  main model — it is read from the checkpoint dict, `checkpoint["atom_context_num"]`
  (`run.py:69`), and passed into `featurize(..., number_of_ligand_atoms=atom_context_num, ...)`
  (`run.py:397`). Checkpoint filename `ligandmpnn_v_32_010_25.pt` suggests 25
  (not code-confirmed — see clause 6 caveat), and this is corroborated by the
  paper: "we obtained the best performance by selecting for the protein–ligand
  and individual residue intraligand graphs the 25 closest ligand atoms"
  (`ligandmpnn.pdf` p.718). The side-chain packing feature build in `run.py`
  hardcodes a *different* value, `number_of_ligand_atoms=16`
  (`run.py:492-498`), independent of whatever the main ligand_mpnn checkpoint
  uses.
- `--ligand_mpnn_cutoff_for_score` default `8.0` (Å) (`run.py:842-846`,
  `score.py:479-483`) — used only to build `mask_XY` for the *reported*
  ligand-proximal score (`data_utils.py:948`), not to select which atoms are fed
  into the network (that selection is purely nearest-`atom_context_num` via
  `get_nearest_neighbours`, `data_utils.py:889-922`).
- `--ligand_mpnn_use_side_chain_context` default `0` (False)
  (`run.py:847-852`, `score.py:471-476`) — consistent with the paper's ablation
  that this did not significantly change performance (`ligandmpnn.pdf` p.719,
  "Providing sidechain atoms as additional context did not significantly
  increase sequence recovery... Supplementary Fig. 1b").
- `--ligand_mpnn_use_atom_context` default `1` (True) (`run.py:836-840`) — set
  to 0 zeroes out `Y_m` post-hoc (`data_utils.py:955-956`) rather than skipping
  the ligand branch of the network.

**10. score_only / single_aa_score: number of batches/orders, averaging.**
`score.py` exposes `--autoregressive_score` (default `0`) and `--single_aa_score`
(default `1`) (`score.py:533-545`) — **the CLI default is single-AA scoring, not
autoregressive scoring**, and if neither flag is set the script prints an error
and exits (`score.py:289-291`). `--number_of_batches` default `1`,
`--batch_size` default `1` (`score.py:451-462`) — each "batch" draws a fresh
`feature_dict["randn"]` (hence a fresh decoding order for `autoregressive_score`,
or a fresh set of per-residue orders for `single_aa_score`) and results are
concatenated along dim 0 (`score.py:280-299`). Averaging for the reported
`mean_of_probs`/`std_of_probs` is **over `probs` (i.e. `exp(log_probs)`), not over
`log_probs`**: `mean_probs = np.mean(out_dict["probs"], 0)` /
`std_probs = np.std(out_dict["probs"], 0)` (`score.py:314-315`), averaged across
the `number_of_batches` axis. `--use_sequence` (default `1`) toggles whether
`single_aa_score`/`score` conditions on the rest of the true sequence
(`score.py:526-531`); it is passed through to `model.score`/`model.single_aa_score`
verbatim.
**Ambiguity (not resolved from source alone):** in `single_aa_score`
(`model_utils.py:470-556`), for each residue `idx` a fresh `order_mask` is built
— `use_sequence=True` sets `order_mask = ones; order_mask[idx] = 0` and
`use_sequence=False` sets `order_mask = zeros; order_mask[idx] = 1`
(`model_utils.py:501-506`) — then the same
`argsort((order_mask+0.0001)*abs(randn))` formula from clause 1 is applied. Given
that formula's ascending-sort behaviour (small weight ⇒ decoded early, and "early"
neighbours populate `mask_bw`, i.e. are visible to later positions per the
`order_mask_backward` einsum, `model_utils.py:514-519`), a literal derivation
suggests `use_sequence=True` places `idx` at rank 0 (nothing precedes it, so `idx`
sees only encoder/structural context, no other positions' identities) — which
reads as the *opposite* of the docstring ("1 - get scores using amino acid
sequence info," `score.py:526-531`). I was not able to fully resolve this
contradiction by static reading alone (it may hinge on a subtlety of how
`mask_bw`/`mask_fw` are consumed for the *query* row versus in the autoregressive
`sample()` loop, which processes one position per step rather than all positions
in one masked pass). Recording as an explicit ambiguity rather than asserting
either direction; would need a numerical check (e.g. perturb `S_true` at another
position and confirm whether `log_probs_out[:, idx, :]` changes) to resolve.

**11. packer: number of samples, chi sampling temperature/defaults, repacking flags.**
`--pack_side_chains` default `0` (off) (`run.py:894-899`). When enabled:
`--number_of_packs_per_design` default `4` (`run.py:908-913`),
`--sc_num_denoising_steps` default `3` (`run.py:915-920`),
`--sc_num_samples` default `16` (`run.py:922-927`, help text: "draw from a
mixture distribution and then take a sample with the highest likelihood"),
`--repack_everything` default `0` (keep fixed residues' side chains fixed,
`run.py:929-934`), `--pack_with_ligand_context` default `1`
(`run.py:950-955`), `--force_hetatm` default `0` (`run.py:936-941`). There is
**no user-exposed chi-sampling "temperature"** flag: `pack_side_chains`
(`sc_utils.py:56-144`) draws `num_samples` i.i.d. draws from a
`MixtureSameFamily` of `VonMises` components (`sc_utils.py:92-99`) and
deterministically keeps the single highest-log-probability draw per
residue/chi-angle (`torch.argmax(log_probs_of_samples, 0)`,
`sc_utils.py:97-99`) — repeated for `--sc_num_denoising_steps` recycling
iterations. The `Packer` model itself is constructed with a hard-coded
`num_mix=3` (`run.py:112`), matching the paper's "mixture (three components) of
circular normal distributions for torsion angles... three means and three
variances per chi angle" (`ligandmpnn.pdf` p.718). The paper additionally
states the four chi angles are decoded autoregressively, "chi1... then chi2,
chi3 and finally chi4" (`ligandmpnn.pdf` p.718); the provided `sc_utils.py`
snippet updates all four torsions' sin/cos jointly per denoising step
(`torsion_dict["torsions_noised"][:, :, 3:] = ...`, `sc_utils.py:100-105`)
rather than showing an explicit per-chi-index loop, so the chi1→chi4
autoregressive detail is taken from the paper, not directly visible in the
`pack_side_chains` function body in this packet — flagged as
paper-vs-code-visibility gap, not a contradiction.

**12. symmetry_residues / symmetry_weights: format, defaults, interaction with ties, bias combination, decode order.**
Format: `--symmetry_residues "A12,A13,A14|C2,C3|A5,B6"` (pipe-separated groups,
comma-separated encoded residues per group, `run.py:774-779`); matching
`--symmetry_weights "1.01,1.0,1.0|-1.0,2.0|2.0,2.3"` (`run.py:780-785`).
`--homo_oligomer` (default `0`) auto-derives both from detected chain identity,
assigning equal weight `1/num_chains` to every tied position
(`run.py:339-358`). If `symmetry_residues` is set but `symmetry_weights` is left
at its default empty value, `model_utils.py:352-355` indexes
`symmetry_weights_list_of_lists[i1][i2]` against an empty inner list for every
group — **there is no silent default of 1.0 per member; an unmatched
`symmetry_weights` for a supplied `symmetry_residues` will raise an
`IndexError`** unless `--homo_oligomer` populated both together. `score.py` only
exposes `--symmetry_residues` (no `--symmetry_weights` flag at all,
`score.py:417-422`) — its use there only affects the *decoding-order grouping*
(tied residues are merged into one step in `new_decoding_order`,
`model_utils.py:601-612`, shared code path with `sample()`); `model.score()`
(`model_utils.py:559-665`) never references any per-group weight or combines
logits across tied members — no equivalent of `sample()`'s `total_logits`
weighting exists in the scoring path.
Decoding: tied members are merged into a single step at the position of the
first-encountered member in the base per-residue order (`model_utils.py:357-368`);
within that step, each member's own encoder/decoder path is run individually
inside an inner `for t in t_list:` loop, and a single `total_logits` is
accumulated as `total_logits += symmetry_weights[t] * logits`
(`model_utils.py:412-443`) — i.e. **members' logits are combined by a weighted
sum, matching the paper's "averaged logits" scheme** (`proteinmpnn_2022.pdf`
p.4: "averaged probabilities, and averaged logits resulted in 52%, 53%, and 55%
median sequence recoveries respectively" — the paper explicitly compares
probability-averaging vs. logit-averaging and reports logit-averaging as best;
the code implements weighted logit summation, consistent with that choice). A
single amino acid `S_t` is then sampled once from the combined
`softmax((total_logits + bias_t)/temperature)` and assigned identically to every
member of the tied group (`model_utils.py:451-460`) — **tied members are
decoded simultaneously as one joint sample, not sequentially.**
**Ambiguity/possible defect:** inside the inner `for t in t_list:` loop,
`bias_t = bias[:, t]` is reassigned on every iteration (`model_utils.py:417`) and
then read *once*, after the loop, to build `probs`
(`model_utils.py:445-447`). This means **only the last member's per-residue
bias survives** to affect the shared softmax for the whole tied group — earlier
members' `bias_AA_per_residue`/`omit_AA_per_residue` contributions are silently
discarded rather than summed/averaged across the group. This reads as an
unintentional artifact of variable reuse in a loop, but it is exactly what the
source does; recorded as an ambiguity/finding rather than an assumption about
intended behaviour.

**13. Seeds and RNG usage; what makes two runs identical.**
`--seed` (int, default `0`) (`run.py:807-812`, `score.py:445-450`). Both scripts
use `if args.seed: seed = args.seed else: seed = np.random.randint(0, 99999, ...)`
(`run.py:31-34`, `score.py:23-26`) — **because `0` is falsy in Python, passing
`--seed 0` explicitly is indistinguishable from not passing `--seed` at all; it
always falls through to a fresh random seed.** The chosen/derived seed then seeds
all three RNG sources used anywhere in the scripts:
`torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)`
(`run.py:35-37`, `score.py:27-29`) — note Python's own `random` module is seeded
even though neither script visibly calls into it directly in the reconstructed
excerpts. `feature_dict["randn"]` (the per-decode-order noise vector consumed by
`torch.argsort` in clause 1) is drawn fresh via `torch.randn(...)` once per
`--number_of_batches` iteration (`run.py:419-423`, `score.py:281-284`), so
reproducibility depends on: same seed, same device (CPU vs GPU RNG streams are
not required to agree — see clause 7), same `--batch_size`/`--number_of_batches`
(these determine how many `torch.randn`/`torch.multinomial` calls occur and in
what order, since PyTorch's global generator state is consumed sequentially),
and — if `--pack_side_chains` is used — the same number of `torch.rand` calls
inside `make_torsion_features` (`sc_utils.py:193-195`) and the same
`MixtureSameFamily.sample()` calls inside `pack_side_chains`
(`sc_utils.py:95`), since those also consume the seeded global RNG state in
sequence. **What makes two runs identical, concretely:** same effective seed
(non-zero, or both left at the falsy default and coincidentally drawing the
same random seed — practically never), same model/checkpoint, same input PDB
and flags (which determine `L`, `chain_mask`, `bias`, etc.), same device, and
the same total ordered sequence of RNG-consuming calls (a change in
`batch_size`, `number_of_batches`, or enabling `pack_side_chains` changes that
sequence and will generally change every subsequent draw even with the same
seed).

**14. Checkpoint names, hashes, weight-loading conventions.**
Default checkpoint paths (all under `./model_params/`, `run.py:661-690,901-906`,
identical in `score.py:348-377`):
- `protein_mpnn` → `proteinmpnn_v_48_020.pt`
- `ligand_mpnn` → `ligandmpnn_v_32_010_25.pt`
- `per_residue_label_membrane_mpnn` → `per_residue_label_membrane_mpnn_v_48_020.pt`
- `global_label_membrane_mpnn` → `global_label_membrane_mpnn_v_48_020.pt`
- `soluble_mpnn` → `solublempnn_v_48_020.pt`
- side-chain packer → `ligandmpnn_sc_v_32_002_16.pt` (`run.py:901-906`)
No hashes/checksums are referenced anywhere in the five source files. Loading
convention: `checkpoint = torch.load(checkpoint_path, map_location=device)`
followed by `model.load_state_dict(checkpoint["model_state_dict"])`
(`run.py:67,90`; `score.py:50,73`) — the checkpoint dict additionally carries
`num_edges` and (for `ligand_mpnn`) `atom_context_num` as plain metadata keys
read out before model construction (`run.py:69-75`; see clauses 6, 9); no other
metadata keys (e.g. training noise level, git commit, config hash) are read from
the checkpoint in either script. `node_features`, `edge_features`, `hidden_dim`
(all `128`), `num_encoder_layers`/`num_decoder_layers` (both `3`) are hard-coded
identically in both scripts' `ProteinMPNN(...)` call (`run.py:77-88`,
`score.py:60-71`) rather than sourced from the checkpoint — a mismatched
checkpoint architecture would fail at `load_state_dict`, not be silently
adapted to.

## Ambiguities

- Clause 2: whether the omission penalty (`1e8`) is or is not divided by
  temperature differs between the LigandMPNN-repo `sample()` formula and the
  original 2022 `protein_mpnn_utils.py` `sample()` formula. Both are practically
  equivalent at any sane temperature but are literally different expressions.
- Clause 3: adding `"X"` to `--omit_AA` appears to be a no-op for `sample()`
  because of the unconditional `probs[:, :20]` truncation — not verified
  end-to-end, inferred from the slice location relative to where `omit_AA`
  enters the `bias` tensor.
- Clause 7: no evidence either way on `torch.topk`/`torch.multinomial`
  determinism across devices/versions; recorded as unresolved rather than
  assumed benign.
- Clause 9/6: exact numeric `k_neighbors`/`atom_context_num` per checkpoint are
  inferred from filename conventions and paper text, not read directly from any
  `.pt` file (none were in the packet) — code only shows *that* these values are
  read from the checkpoint at runtime, not *what* they are.
- Clause 10: the direction of information flow in `single_aa_score`'s
  `use_sequence` branch (does `True` give idx full sequence context or
  structure-only context?) could not be resolved with confidence from static
  reading; my literal trace of the `argsort`/`order_mask_backward` formula
  suggests the opposite of what the docstring/CLI help implies, but I flag this
  as something I could easily have mis-traced rather than asserting a bug.
- Clause 11: the paper describes explicit chi1→chi2→chi3→chi4 autoregressive
  decoding for the packer; the `pack_side_chains` function body provided in
  this packet updates all four torsions' sin/cos representations together per
  denoising step rather than showing a per-chi-index loop, so I cannot confirm
  from code alone whether the per-chi autoregressive detail lives in a part of
  `sc_utils.py`/`model_utils.py` outside what was excerpted/legible, or is
  implemented differently than the paper's prose suggests.
- Clause 12: the "only the last tied member's bias survives" behaviour reads
  like an unintentional consequence of variable shadowing in a loop rather than
  a documented design choice — I record the code's literal behaviour without
  asserting whether it is a bug or intended.
- Clause 13: `--seed 0` silently means "no seed" due to Python truthiness; not
  documented in the `--seed` help text (`run.py:807-812` / `score.py:445-450`
  just say "Set seed for torch, numpy, and python random").
- General (not code-derived): `score.py`'s `--parse_these_chains_only` is passed
  to `parse_PDB`'s `chains` parameter as a raw, unsplit string
  (`score.py:107-113`), whereas `run.py` explicitly splits the same flag's value
  on commas into a list first (`run.py:183-186,198-204`). Because `parse_PDB`
  iterates `chains` character-by-character to build a selection string
  (`data_utils.py:780-784`), `score.py`'s own help text uses concatenated
  single letters (`"ABCF"`, `score.py:492-497`) while `run.py`'s help text uses
  comma-separated letters (`"A,B,C,F"`, `run.py:860-865`) — these are two
  genuinely different accepted input conventions for what looks like the same
  flag name across the two scripts, not a typo in one of the help strings.
- General: `--zero_indexed` is declared `type=str` but defaulted to the int `0`
  in both scripts (`run.py:801-806`, `score.py:439-444`); the default (int `0`)
  is falsy, but if a user explicitly passes `--zero_indexed 0` on the CLI,
  argparse stores the *string* `"0"`, which is truthy in Python — `if not
  args.zero_indexed:` (`run.py:549-550`) would then behave as though
  `zero_indexed` were `True`, the opposite of the value the user typed. This is
  a flag-type inconsistency visible directly in the argparse declarations, not
  an inference.
- General: `score.py`'s `--chains_to_design` defaults to `None` (not `""`) and
  is checked with `if type(args.chains_to_design) == str:` (`score.py:161-164,
  485-490`); passing an explicit empty string versus leaving the flag unset
  therefore produce different `chains_to_design_list` values (`[""]` vs. "all
  chains"), unlike `run.py` where the flag always defaults to `""` and is
  always split (`run.py:278-281`).

## Assumptions

- Where a checkpoint-embedded value (num_edges, atom_context_num) could only be
  inferred from filename convention or from the paper's prose rather than read
  directly from a `.pt` file, I have explicitly labelled it as inferred/not
  code-confirmed in the clause text rather than silently treating it as fact.
- I assume `alphabet = list(restype_str_to_int)` (`data_utils.py:78`) preserves
  Python 3.7+ dict insertion order, so `alphabet[i] == restype_int_to_str[i]`
  for all `i`; this is standard CPython behaviour, not something I verified by
  execution (no code was run for this reconstruction, per the blind-protocol
  instructions — everything above is from static reading only).
- I assume the two PDF page ranges I fetched (pp.2–4 of each paper) are
  representative of the papers' Methods framing for decoding order, noise, and
  ligand-atom-context choices; I did not read either paper's full Methods
  section or Supplementary Information, so paper-side numeric details beyond
  what appears on those pages (e.g. any Table S1 hyperparameter grid) are not
  reflected here.
