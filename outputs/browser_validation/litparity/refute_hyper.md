# Phase 3: Adversarial Refutation — Hyperparameter Fidelity

task_id: `260923_aminx-browser-validation`

## Lens

Hyperparameter fidelity: are the numeric constants and defaults in aminx exactly those of
`dauparas/LigandMPNN@26ec57ac` (+ `ProteinMPNN@8907e667`), and where they are not, do the
mismatches change model output?

## Assumption(s) hunted

1. I will test whether `atom_context_num` is actually wired from the checkpoint filename (25 for
   `ligandmpnn_v_32_*_25`) into the instantiated `PrxteinLigandMPNN`/`ProteinFeaturesLigand`, or
   silently stays at the code default (16) — clauses.json's `ligand_atom_context` claims the
   latter from a static read; I will confirm it by actually building the model.
2. I will test whether the shipped `.eqx.zst` weights contain any **invented bias** (a bias
   parameter the reference layer does not have, left at its random skeleton-initialisation value
   because `convert_linear_layer` only overwrites a bias when the `.pt` supplies one) beyond the
   one instance (`membrane_label_encoding`) clauses.json already names — by systematically
   diffing every reference `bias=False` `nn.Linear` against its aminx counterpart, not just the
   two clauses.json flagged.
3. I will quantify, not just assert, the two known "random bias baked into shipped weights"
   claims (`membrane_label_encoding`, and whatever new instance #2 turns up) by loading the real
   shipped checkpoint and comparing the loaded bias to a bias from a freshly-initialised skeleton
   with the same PRNG key.
4. I will check whether the checkpoint metadata itself (the `.pt` file's `num_edges` /
   `atom_context_num` keys, present in this packet's checkpoints, contra reconcile.md's caveat
   that "the .pt value itself is not in the packet") actually equals the filename-parsed value
   aminx uses, closing the "inferred, not verified" gap `checkpoint_topology_source` left open.

**Honesty tax**: default verdict is DEVIATION/inconclusive unless a probe gives a clean
runtime confirmation either way. Every mismatch below except the LayerNorm-eps and dropout
checks was confirmed by loading the real shipped `.eqx.zst` and/or the real reference `.pt`
locally (no network; `AMINX_WEIGHTS_DIR` unset, packaged resources under
`src/aminx/model_params/` and `reference_ligandmpnn_clone/model_params/` used directly). Nothing
was mocked or synthetic.

## Probe environment

`JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`, run via
`uv run --frozen --extra dev --extra benchmark python3 -`. All scratch scripts were inline heredocs
under the sandboxed tmp dir, nothing written to the repo, nothing committed.

## Constants table

Every numeric constant/default on the in-scope paths (`aminx/io/weights.py`, model constructors,
inference defaults), cross-checked against the reference packet. "Confirmed" = I loaded real
bytes and checked the number this session; "Static" = read from source, not re-executed (already
verified at `path:line` by Phase 1/2, no new evidence added this session).

| Constant | Reference value (file:line) | aminx value (file:line) | Match? | Output impact |
|---|---|---|---|---|
| `k_neighbors` (protein, `v_48`) | `checkpoint['num_edges']` = 48; `model_utils.py:20` default `top_k=48` | `get_topology_for_checkpoint`: `k_neighbors=48` for `v_48` (`weights.py:83-84`) | MATCH | none |
| `k_neighbors` (ligand, `v_32`) | `checkpoint['num_edges']` = 32 — **CONFIRMED this session**: `ligandmpnn_v_32_010_25.pt` meta `{'num_edges': 32, 'noise_level': 0.1, 'atom_context_num': 25}` | `get_topology_for_checkpoint`: `k_neighbors=32` (`weights.py:78-79`); runtime `load_model(...).features.k_neighbors == 32` — **CONFIRMED** | MATCH | none |
| `k_neighbors` (packer, `sc_v_32`) | `run.py:95-110` hard-codes `top_k=32` for the packer | `weights.py`: `v_32` branch gives 32, model_type="packer" (`weights.py:87-89`) | MATCH | none |
| `atom_context_num` (ligand) | `checkpoint['atom_context_num']` = 25 — **CONFIRMED this session** from the real `.pt` metadata (paper: "25 closest ligand atoms", filename `_25`) | `get_topology_for_checkpoint("ligandmpnn_v_32_010_25")` returns `atom_context_num: 25` (`weights.py:96-104`) — **BUT** `load_model` never forwards it: `PrxteinLigandMPNN.__init__` (`ligand_mpnn.py:117-146`) has no `atom_context_num` parameter at all, so `ProteinFeaturesLigand` keeps its own default `atom_context_num: int = 16` (`ligand_features.py:220`). **Runtime-confirmed this session**: `load_model(checkpoint_id="ligandmpnn_v_32_010_25").features.atom_context_num == 16` | **DEVIATION** | **CORE.** Every ligand-conditioned forward pass conditions on the 16 nearest ligand atoms per residue instead of 25 whenever a residue has >16 ligand atoms within reach — changes `ProteinFeaturesLigand` edge/node inputs and therefore logits on any multi-atom ligand. This is the shipped checkpoint's own declared value, silently discarded, not an ambiguous inference. |
| `atom_context_num` (packer) | `run.py:95-110` hard-codes 16 for the packer | `weights.py` parses 16 from `_16` suffix and `load_model` **does** pass `atom_context_num=topo["atom_context_num"]` to `Packer(...)` (`weights.py:434-444`) | MATCH | none — packer wiring is correct; only the ligand (`PrxteinLigandMPNN`) constructor drops the parameter. |
| `num_positional_embeddings` (protein `v_48`) | reference `PositionalEncodings(num_embeddings=32)` size implied by `v_48`/32-edge checkpoints (`max_relative_feature=32` in `model_utils.py:1638`) | `weights.py:83-84`: 32 | MATCH | none |
| `num_positional_embeddings` (ligand `v_32`) | ligand checkpoints use 32-wide relative encoding regardless of `k_neighbors=32` label (paper/protocol) | `weights.py:90-92`: explicitly overridden to 32 for `"ligandmpnn" in name` (post the `v_32`→16 branch) | MATCH | none |
| `num_positional_embeddings` (packer) | `run.py` packer path, 16 | `weights.py:88-89`: 16 for `ligandmpnn_sc`/`_sc_` | MATCH | none |
| max relative index / chain offset | `max_relative_feature=32`, inter-chain edges collapse to a single index `2*32+1=65` with the offset discarded (`model_utils.py:1645-1648`) | Protein path: `positional_encoding_protein` — same clamp+single-hot, MATCH (Static, verified Phase 2 at `features.py:148,254-265`). **Ligand path differs**: `ligand_features.py:106-122` is two-hot (clipped offset one-hot **concatenated with** a separate same/different-chain bit), so inter-chain ligand edges still vary with the raw offset instead of collapsing to one fixed index | protein: MATCH; ligand: **DEVIATION** (`positional_encoding_ligand`) | Core on the ligand path for any multi-chain input; no effect on single-chain inputs (offset clamp range is otherwise identical). |
| RBF count / range (protein) | `[2, 22]`, 16 bins (Assumptions in recon_math, corroborated by code shape) | `radial_basis.py:16-18`: `RADIAL_BASE_MINIMUM=2.0, RADIAL_BASE_MAXIMUM=22.0, RADIAL_BASES=16` | MATCH | none |
| RBF count / range (ligand) | same | `ligand_features.py:184-188`: `D_min=2.0, D_max=22.0, D_count=16` | MATCH | none |
| Virtual-Cb coefficients | `-0.58273431, 0.56802827, -0.54067466` (Static, Phase 2) | `ligand_features.py:294`: identical constants — **spot-checked this session**, byte-identical literals | MATCH | none |
| Message-passing scale constant | fixed divisor 30 (not true neighbour count), node update only, no `/scale` on edge update (Static, Phase 2, `model_utils.py:1716-1739`) | `encoder.py`: `scale: float = 30.0` default (line ~484), used in node update; edge update has no `/scale` (Static, confirmed at `encoder.py:216-249`) | MATCH | none |
| LayerNorm eps | `torch.nn.LayerNorm(dim)` → PyTorch default `eps=1e-5` everywhere aminx's reference calls it (grep of `model_utils.py`/`sc_utils.py` finds no explicit `eps=` override) | aminx uses `eqx.nn.LayerNorm(shape)` with no explicit `eps=` anywhere in `src/aminx/model/*.py` → equinox default `eps=1e-5` (confirmed from the installed equinox signature this session) | MATCH | none |
| Hidden / node / edge feature dims | `128` (paper + `run.py` architecture hard-code) | `weights.py:42-44`: `NODE_FEATURES=EDGE_FEATURES=HIDDEN_FEATURES=128` | MATCH | none |
| Encoder / decoder layer counts | `3` / `3` (paper + `run.py`) | `weights.py:45-46`: `NUM_ENCODER_LAYERS=NUM_DECODER_LAYERS=3` | MATCH | none |
| Ligand context layers | Dauparas reference LigandMPNN uses 2 alternating context blocks (Static; not independently re-derived this session) | `weights.py:50`: `NUM_LIGAND_CONTEXT_LAYERS=2` | MATCH (assumed, not re-derived) | none |
| Dropout at inference | reference `model.eval()` / `Dropout(p)` with training-time `p`, but eval mode zeroes it | aminx modules keep `dropout_rate=0.1` topology but call with `inference=True`/`key=None` everywhere in the inference paths, making dropout identity (Static, Phase 2) | MATCH | none |
| `augment_eps` / backbone noise default | `0.0` — never passed by `run.py`/`score.py` (Static, Phase 2) | `backbone_noise` default `0.0` (Static, Phase 2, `model/mpnn.py`) | MATCH | none — divergence exists only outside the reference-reachable config (`backbone_noise>0`), already recorded, not core. |
| Temperature default | `run.py:826-830`: `--temperature` default `0.1` | `run/specs.py:576`: `SamplingSpecification.temperature = 0.1` | MATCH | none |
| Omit-AA bias constant | `-1e8`, additive, inside the temperature division (`run.py:404-408`) | No built-in constant — `bias` is a caller-supplied `(L,21)` array; `-1e8` reproduces the reference exactly if the caller supplies it (Static, Phase 2, `omit_as_bias`) | MATCH-if-constructed | none when the harness builds the bias correctly; aminx ships no `-1e8` constant of its own, so a caller who does not know the magic number gets silently-wrong (near-zero, not hard-zero) suppression. Not a code defect per se — an API-surface gap. |
| Ligand-proximal score cutoff | `8.0 Å` (`run.py:432-442`, `cutoff_for_score`) | **absent** — grep of `src/aminx` finds no `cutoff_for_score`/ligand-distance-masked score anywhere (Static, Phase 2, `ligand_cutoff_for_score`) | MISSING | Reported-number-only: does not change which atoms the network sees or the logits, only a derived confidence metric a harness must compute host-side. |
| Packer `num_samples` (best-of-N draws) | `16` (`sc_utils.py`/`run.py:908-927`) | **absent** — `packer.py:904-969` returns mixture parameters only, no draw/argmax loop (Static, Phase 2, `packer_chi_sampling`) | MISSING | No packer sampling exists natively; not a "wrong constant", a whole absent stage. |
| Packer mixture components | `3` (von Mises mixture, `sc_utils.py:383-387`) | `packer.py:749`: `num_mix=3`, `:806` `w_torsions` sized `4*3*num_mix` | MATCH | none |
| Packer concentration floor | `0.1 + softplus(...)` | `packer.py:962-969`: identical `0.1` floor + softplus (Static, Phase 2) | MATCH | none |
| Membrane label values | `{0: neither, 1: interface, 2: buried}`, both-flags collapse to 0 (Static, Phase 2, `membrane_label_construction`) | Not constructed in aminx at all — caller must supply the one-hot; no label encoder present | MISSING (construction) | Caller responsibility; the label *values themselves* are not a code constant to mismatch, only their construction. Separate from the bias defect below. |
| `physics_feature_dim` (membrane) | `3` (buried/interface/global one-hot width) | `weights.py:93-94`: `physics_feature_dim=3` for `"membrane" in name` | MATCH | none |
| KNN masked-pair sentinel | reference pushes masked pairs to `D_max` (row max), so they **tie** the farthest valid neighbour (`sc_utils.py:852-854`) | aminx uses a strictly-larger sentinel: `+inf` (protein `features.py:208-225`), `1e4` (ligand `ligand_features.py:299`), `1e8` squared-distance (packer `packer.py:417`) | DEVIATION (`knn_masked_pair_sentinel`) | Core only on rows with fewer than `k` valid partners (short/gapped chains); identical neighbour sets otherwise. Algorithmic-structure-adjacent; sentinel *values* are the hyperparameter-fidelity angle, already well evidenced in Phase 2, not re-probed this session (would need a synthetic gapped-chain fixture; out of this session's time budget). |
| `x_omit_at_sampling` (renormalisation vs 21-way) | probs renormalised over `[:20]`, X unreachable (Static, Phase 2) | full 21-way categorical, X reachable unless caller sets `bias[:,20]=-1e8` (Static, Phase 2) | DEVIATION, designed (`core:false` per clauses.json) | Not re-probed; this is a *default-arm* behavioural choice more than a stray numeric constant, and clauses.json already marks it non-core (exact reference-compatible mode exists). |

## New findings this session (beyond clauses.json)

### 1. `ligand_atom_context` — runtime-confirmed, upgraded from static to executed evidence

```
>>> get_topology_for_checkpoint("ligandmpnn_v_32_010_25")
{'k_neighbors': 32, 'num_positional_embeddings': 32, 'atom_context_num': 25,
 'physics_feature_dim': 0, 'model_type': 'ligand'}
>>> load_model(checkpoint_id="ligandmpnn_v_32_010_25", key=jax.random.PRNGKey(0)).features.atom_context_num
16
```
The checkpoint's own `.pt` metadata (`atom_context_num: 25`, `num_edges: 32`) matches the
filename-derived topology exactly, closing reconcile.md's "not code-confirmed" gap — but the
constructed model still gets 16, because `PrxteinLigandMPNN.__init__` (`ligand_mpnn.py:117-146`)
simply has no `atom_context_num` parameter to receive it. This is a wiring bug, not an inference
ambiguity: the correct number is computed and then dropped on the floor.

### 2. NEW instance of `weight_conversion_bias_handling`: `v_c` (ligand context fusion) carries an invented, untrained bias

Systematically diffed every reference `nn.Linear(..., bias=False)` site in `model_utils.py` +
`sc_utils.py` (14 sites) against aminx's `use_bias=False` sites (12 sites) in
`src/aminx/model/*.py`. Two reference `bias=False` layers have **no** matching `use_bias=False`
in aminx:

- `model_utils.py:1475-1477`: membrane `node_embedding` — already known (`membrane_label_encoding`).
- `model_utils.py:50` / `sc_utils.py:316`-analogue: `self.V_C = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)`,
  the protein←ligand context-fusion projection quoted in clauses.json's own `ligand_context_fusion`
  note ("`h_V + V_C_norm(dropout(V_C(h_V_C)))`"). aminx's `PrxteinLigandMPNN.v_c` is built at
  `ligand_mpnn.py:224` as `eqx.nn.Linear(node_features, node_features, key=proj_keys[4])` — **no**
  `use_bias=False`. `ligand_context_fusion` was marked MATCH in clauses.json; that verdict covered
  the *op order* correctly but missed this bias.

Confirmed against the real checkpoint (`ligandmpnn_v_32_010_25.pt`, `model_state_dict` has
`V_C.weight` only, no `V_C.bias` key) and the real shipped `.eqx.zst`:

```
skeleton.v_c.bias[:8]  (fresh PRNGKey(0) init) = [-0.00984, -0.04649, -0.05454, -0.04275, -0.06967, 0.01342, 0.03485, 0.04101]
loaded.v_c.bias[:8]    (from shipped ligandmpnn_v_32_010_25.eqx.zst) = IDENTICAL, bit-for-bit
loaded.v_c.bias mean/std/absmax = -0.00033 / 0.0497 / 0.0866
loaded.v_c.weight[:2,:3] = real, trained-looking values (differs from random init)
```
The weight matrix loaded correctly; the bias is untouched random initialisation, because
`convert_linear_layer` (`scripts/convert_weights.py:25-44`) only overwrites `jax_layer.bias` when
`pt_bias is not None`, and here it is `None`.

**This is a broader blast radius than the membrane bias.** `v_c` sits on the *main* LigandMPNN
path — `h_V = h_V + v_c_norm(dropout(v_c(h_V_C)))` fires on every ligand-conditioned forward pass
regardless of membrane/side-chain-context flags, whereas the membrane bias only affects the
separate `physics_feature_dim=3` checkpoint family. Every existing `ligandmpnn_v_32_*` checkpoint
in `src/aminx/model_params/` ships this same random 128-d bias, added to every residue's node
feature identically before `v_c_norm` (LayerNorm), which does not cancel a non-uniform per-channel
additive constant. `ligand_context_fusion`'s own PASS note in clauses.json needs a bias-scoped
caveat, and `weight_conversion_bias_handling`'s "known instances" list needs updating from 2 to a
minimum of 3.

### 3. Quantified the known membrane bias (previously "unmeasured" per clauses.json)

```
skeleton.encoder.physics_projection.bias[:8]  (fresh PRNGKey(0) init) = [0.244, 0.0289, 0.565, -0.249, -0.121, -0.349, -0.0426, 0.499]
loaded.encoder.physics_projection.bias[:8]    (from shipped global_label_membrane_mpnn_v_48_020.eqx.zst) = IDENTICAL, bit-for-bit
loaded bias mean/std/absmax = 0.0197 / 0.345 / 0.571
```
Confirms clauses.json's `membrane_label_encoding` exactly and gives it the magnitude reconcile.md
said was "unmeasured, static reading, not executed": a ~0.57-magnitude untrained per-channel shift
is large relative to a post-LayerNorm(128) unit-variance activation, applied identically to every
residue before `physics_norm`. The reference `.pt` (`global_label_membrane_mpnn_v_48_020.pt` and
`per_residue_label_membrane_mpnn_v_48_020.pt`, both checked) has `features.node_embedding.weight`
only, confirming `bias=False` on the reference side for both membrane variants.

### 4. Positional-encoding-ligand bias, quantified on the reference side

```
pt features.embeddings.linear.bias  mean/std/absmax = 0.0189 / 0.0263 / 0.0639
pt features.embeddings.linear.weight abs-mean (scale reference) = 0.119
```
The reference bias is small but non-negligible relative to the weight's own per-feature
contribution (roughly 10-50% of a single active one-hot term), and it is a per-position-constant
(same 16-d addition on every residue pair) that is completely absent from the aminx ligand edge
embedding — not merely mis-set, since `PositionalEncodings.w_pos` is built with `use_bias=False`
(`ligand_features.py:106`), a structural absence with no leaf to overwrite. Not re-probed on the
aminx side beyond confirming the field is `None`, which is unambiguous from the constructor alone.

### 5. Inert filename-parsing artifact (informational, not a defect)

`get_topology_for_checkpoint("global_label_membrane_mpnn_v_48_020.eqx.zst")` returns
`atom_context_num: 20` — the `"_020"` noise-level suffix (0.20 Å training noise, same convention
as `proteinmpnn_v_48_020`) is misread by the generic "last numeric part, if not in {32,48,30}"
heuristic as an atom-context count. This has **zero output impact**: `atom_context_num` is not
read anywhere on the protein/membrane construction path (only `ligand`/`packer` types consume it).
Flagged only as corroborating evidence for `checkpoint_topology_source`'s existing point that the
filename convention is a heuristic, not a real parse of checkpoint metadata, and can silently
produce a plausible-looking wrong number for unrelated fields.

## Side note, out of lens (not scored as a defect)

While loading checkpoints for the bias probes, `global_label_membrane_mpnn_v_48_020.eqx.zst` and
`per_residue_label_membrane_mpnn_v_48_020.eqx.zst` in this worktree's
`src/aminx/model_params/` are **not actually zstd-compressed** (`file` reports "NumPy array",
magic bytes `\x93NUM` not `\x28\xb5\x2f\xfd`), unlike every other `.eqx.zst` in the same
directory. `load_weights()`'s `checkpoint_id` route (`weights.py:349-354`) calls
`zstd.ZstdDecompressor().decompress(data)` unconditionally and raises `ZstdError` on these two
files; only the separate `local_path` route (`weights.py:341-348`) checks the zstd magic bytes
first and falls back to a raw `eqx.tree_deserialise_leaves`. I bypassed this to reach the actual
tensors (direct `eqx.tree_deserialise_leaves(path, skeleton)`, no zstd). This is plausibly a
local packaging artifact of this dev worktree rather than a defect in a real install (the shipped
Hub/wheel resource may be correctly compressed), so I am not scoring it as a hyperparameter-fidelity
defect — it is orthogonal to the reference-parity lens and I did not verify which is true of a
real install. Flagging so it is not lost: if it does reproduce against the packaged Hub revision
(`HF_REVISION` in `weights.py:30`), `load_model("global_label_membrane_mpnn_v_48_020")` and its
per-residue sibling are currently **unloadable** via the normal checkpoint-id API entirely.

## Verdicts

| Element | Verdict | Severity | Confidence |
|---|---|---|---|
| `ligand_atom_context` | Defect found, runtime-confirmed | core | high — direct execution, checkpoint metadata cross-checked |
| `ligand_context_fusion_bias` (new) | Defect found, runtime-confirmed | core | high — direct execution against real shipped weights and real `.pt` |
| `membrane_label_encoding` | Defect found, runtime-confirmed and quantified | core | high |
| `positional_encoding_ligand` | Defect found (structural absence + reference bias quantified) | core | high on structure; magnitude-vs-output not executed (would need a forward-pass ablation, out of this session's scope) |
| `weight_conversion_bias_handling` | Defect found — mechanism confirmed, instance count raised from 2 to ≥3 | core | high |
| `checkpoint_topology_source` | Defect found — "inferred not verified" gap closed (now verified-correct for k/atom_context_num on this checkpoint), still fragile by construction | core (unchanged) | high on this checkpoint; open for any renamed/local checkpoint |
| `knn_masked_pair_sentinel` | Not independently re-probed this session | core (per Phase 2) | inconclusive (would need a gapped-chain fixture) — deferring to Phase 2's static finding, no new evidence either way |
| Other MATCH constants in the table (k_neighbors, num_positional_embeddings, RBF, scale/30, LayerNorm eps, dims, layer counts, dropout, augment_eps, temperature) | Defect not found | n/a | high — either directly executed or unambiguous from default-vs-default source comparison |
| `x_omit_at_sampling`, `topk_tie_break` | Not re-probed; Phase 2 verdicts stand | designed-deviation / ambiguous (per clauses.json) | deferring, no new evidence |

## Recommendations

1. **Fix `ligand_atom_context` and add a regression test.** Add `atom_context_num` to
   `PrxteinLigandMPNN.__init__` and thread it from `load_model`'s `topo["atom_context_num"]`
   (`weights.py:420-432`) the same way `Packer` already receives it. Add an assertion in
   `load_model` (or a loader-level test) that `model.features.atom_context_num ==
   topo["atom_context_num"]` for every shipped ligand checkpoint, so this class of drop cannot
   recur silently.
2. **Fix the bias-drop family in one pass, not two.** `convert_linear_layer`'s silent
   bidirectional reconciliation is the root cause of at least 3 known instances
   (`membrane_label_encoding`, `v_c`/`ligand_context_fusion_bias`, and the inverse case
   `positional_encoding_ligand`). Recommend: (a) fix `PrxteinLigandMPNN.v_c` and
   `Encoder.physics_projection` to `use_bias=False` to match the reference exactly (cheapest,
   correctness-preserving fix — zero the random bias rather than trying to recover a value that
   never existed); (b) add `use_bias=True`/explicit bias plumbing to `PositionalEncodings.w_pos`
   (ligand) so its reference bias can actually be loaded; (c) change `convert_linear_layer` to
   **raise** on any bias-presence mismatch instead of silently reconciling it, per
   `weight_conversion_bias_handling`'s own recommendation — this converts future instances of the
   same defect class from silent to loud.
3. **Run the strict leaf-by-leaf `.pt` vs `.eqx` comparison reconcile.md called for**, now that a
   working local recipe exists (`torch.load(..., weights_only=False)` + direct
   `eqx.tree_deserialise_leaves(path, skeleton)`, bypassing zstd where needed) — this session's
   grep-based sweep covered every literal `bias=False`/`use_bias=False` call site, which should
   catch anything structurally similar, but a full leaf-by-leaf pass would also catch a
   shape/dtype mismatch this sweep could not (e.g. a `nn.Parameter` built by hand rather than via
   `nn.Linear`).
4. **Resolve the two non-zstd `.eqx.zst` files** (side note above) before treating them as a
   working parity target — confirm whether this reproduces against the Hub-pinned revision
   (`HF_REVISION` in `weights.py:30`) or is worktree-local drift.

```json
{"defects": [
  {"id": "ligand_atom_context", "paths": ["P11"], "severity": "core", "evidence": "load_model('ligandmpnn_v_32_010_25').features.atom_context_num == 16, not the checkpoint's own metadata value 25 (verified via .pt: atom_context_num=25, num_edges=32)", "runnable": true},
  {"id": "ligand_context_fusion_bias", "paths": ["P11"], "severity": "core", "evidence": "PrxteinLigandMPNN.v_c (ligand_mpnn.py:224) lacks use_bias=False; shipped ligandmpnn_v_32_010_25.eqx.zst v_c.bias is bit-identical to a fresh PRNGKey(0) skeleton init (mean -0.00033, std 0.0497), while the .pt has V_C.weight only and no V_C.bias", "runnable": true},
  {"id": "membrane_label_encoding", "paths": ["P13", "P01"], "severity": "core", "evidence": "shipped global_label_membrane_mpnn_v_48_020.eqx.zst encoder.physics_projection.bias is bit-identical to a fresh PRNGKey(0) skeleton init (mean 0.0197, std 0.345, absmax 0.571); reference .pt features.node_embedding has weight only, no bias", "runnable": true},
  {"id": "positional_encoding_ligand", "paths": ["P11", "P01"], "severity": "core", "evidence": "aminx ligand PositionalEncodings.w_pos built with use_bias=False (structural absence); reference features.embeddings.linear.bias exists in every ligand .pt with nonzero magnitude (mean 0.0189, std 0.0263, absmax 0.0639)", "runnable": true},
  {"id": "weight_conversion_bias_handling", "paths": ["P01", "P11", "P13"], "severity": "core", "evidence": "convert_linear_layer only overwrites jax_layer.bias when pt_bias is not None; systematic sweep of all 14 reference bias=False Linear sites vs aminx's 12 use_bias=False sites found the same 2 invented-bias instances above plus the 1 dropped-bias instance, no others", "runnable": true},
  {"id": "checkpoint_topology_source", "paths": ["P01", "P11"], "severity": "core", "evidence": "get_topology_for_checkpoint is filename-heuristic-only; verified correct for ligandmpnn_v_32_010_25 against real .pt metadata (k=32, atom_context_num=25), but misparses membrane checkpoint's noise-level suffix '_020' as atom_context_num=20 (inert for that model type, but shows the heuristic has no real validation)", "runnable": true}
]}
```
