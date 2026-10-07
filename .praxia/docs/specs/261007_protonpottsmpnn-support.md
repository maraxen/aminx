---
title: 'ProtonPottsMPNN support: a different upstream lineage, a familiar feature contract, and where the oracle boundary belongs'
description: V1 scope (inference + design) for porting ProtonPottsMPNN into aminx, with the atomworks featurization re-scoped as a front end rather than a feature algebra, and a decision rule for when tensor-level parity is required versus whole-pipeline agreement
status: proposed
task_id: 261007_protonpottsmpnn-support
date: '261007'
sprint: ''
backlog_ids: ''
---
# ProtonPottsMPNN support

Upstream: <https://github.com/christian-creator/ProtonPottsMPNN> — *"A PottsMPNN with explicit
protonation-state alphabet for designing pH-switchable binders."*

**This document proposes; it decides nothing.** §11 lists what is the user's.

## 0. Verdict

Feasible. The expensive-looking part is not expensive, and the cheap-looking part carries the
only unbounded risk.

- **It is not an increment on our PottsMPNN port.** Different upstream lineage (§1). Almost no
  aminx *model* code transfers; almost all of the sprint *scaffolding* does.
- **The atomworks featurization is a front end, not a feature algebra** (§4). It emits the nine
  tensors we already build. An earlier scoping pass called this the schedule-dominating risk;
  that was wrong, and it is corrected here rather than quietly dropped.
- **The one genuinely unbounded item is encoder weight compatibility** (P0/P6, §8) — whether
  foundry's `ProteinMPNN` is the same network as the one aminx already implements. Everything
  else is bounded once that is answered, which is why P0 runs before anything is designed on
  top of it.

## 1. Lineage: the thing that must not be misread

aminx ported **KeatingLab/PottsMPNN @ `0cb0a58`** (`specs/260929_pottsmpnn-lasermpnn-xtrax-composition.md:212`,
`:1078`) — a single `potts_mpnn_utils.py` built on the original ProteinMPNN.

ProtonPottsMPNN is built on **IPD's `rc-foundry` `mpnn` package**, vendored into the repo under
`foundry/models/mpnn/`. Verified by reading
`foundry/models/mpnn/src/mpnn/model/pottsmpnn.py`:

```python
from atomworks.constants import UNKNOWN_AA
from mpnn.model.layers.graph_embeddings import (
    ProteinFeatures, ProteinFeaturesLigand, ProteinFeaturesMembrane, ProteinFeaturesPSSM,
)
from mpnn.model.layers.message_passing import DecLayer, EncLayer, cat_neighbors_nodes, gather_nodes
from mpnn.model.mpnn import ProteinMPNN
...
class PottsMPNN(ProteinMPNN):
```

Two different codebases that share a name and a paper lineage. **Anyone who reads this as
"PottsMPNN plus nine tokens" will build on sand.** That framing is what P0 exists to refuse.

What *does* transfer, and is worth a great deal: the FamilyDriver seam
(`host/family_driver.py`, `host/runner.py`), the `tests/port/` tier contract and
`targets/*.toml` tolerance-target shape, the oracle dump + seal discipline, the
`scripts/parity/` vehicle pattern, and the knob gate. The last sprint's machinery is reusable
even though its model code is not.

## 1a. P0 HAS RUN. The checkpoint settles three open questions, and the biggest risk shrank

Spike executed 2026-10-07 against the vendored clone at
`/home/marielle/repos/ProtonPottsMPNN` @ **`09682abfa7d20e0abcdeea0490b7a4b1c190aee3`**
(2026-09-30, 640 MB, weights committed in-repo). Throwaway probe, no sidecar — it produced an
answer, not a finding.

**How it was loaded, because it matters.** `torch.load(..., weights_only=False)`, which the
upstream inference script uses, is arbitrary code execution on a third-party file. The probe
instead ran `weights_only=True` with a **bounded allowlist** of omegaconf's data containers plus
`collections.defaultdict` — every other global still refuses. Anyone repeating this should do
the same rather than disabling the check.

### (a) The vocab is 30. The 30-vs-32 conflict is resolved, by the weights

| tensor | shape |
| :-- | :-- |
| `W_s.weight` | `(30, 128)` |
| `W_out.weight` / `W_out.bias` | `(30, 128)` / `(30,)` |
| **`etab_out.weight`** | **`(900, 128)`** |

900 = 30², so the pair table is V×V at V=30 — the etab confirms the alphabet independently of
the embedding. `train_cfg.extended_vocab = 'v6'`. The 32-token claim in
`transforms/extended_vocab.py` does not describe this checkpoint. **§5's refactor target is
V=30, PAIR_DIM=900.**

### (b) The architecture is ProteinMPNN, by parameter name and by shape

120 tensors, hidden dim **128**, **3 encoder + 3 decoder layers**, `graph_featurization_module`
(5 tensors), `W_e`, `W_s`, `W_out`, `etab_out`. Each layer carries
`norm1/norm2/norm3`, `W1/W2/W3` (node update), `W11/W12/W13` (edge update) and
`dense.W_in/W_out` — the ProteinMPNN `EncLayer`/`DecLayer` names aminx already implements, at
the shapes it already uses (`W1: (128, 384)` = 3×128 concat, `dense: 128→512→128`).

**This is the single most important P0 result: the top risk in §9 got much smaller.** P6 was
sized as "2–4 d, unbounded if foundry's ProteinMPNN diverges". On names and shapes it does not
diverge — encoder reuse now looks like a weight-conversion job rather than a second port.
It is **not yet proven numerically**; that is P6's gate (0 unmapped keys plus
`protonpotts_encoder` parity), and this evidence is structural only.

### (c) Everything is float32

`dtypes = {'torch.float32': 120}`. No f64 anywhere in the checkpoint, consistent with the
standing constraint that f64 is for parity assertions and never for production.

### Re-running the probe

The probe script itself was throwaway (a spike produces an answer, not a finding), but the
*loading recipe* is the non-obvious part and P2 needs it, so it is recorded here rather than
rediscovered. The project venv has no `omegaconf`, so an ephemeral environment is the cheapest
route:

```bash
uv run --no-project --with omegaconf --with torch python -I <probe.py> \
  /home/marielle/repos/ProtonPottsMPNN/checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt
```

and inside it, before `torch.load(..., weights_only=True)`:

```python
torch.serialization.add_safe_globals([
    DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode, ValueNode,
    Any, dict, list, tuple, set, int, float, str, bool, bytes,
    collections.defaultdict, collections.OrderedDict,
    pathlib.PosixPath, pathlib.PurePosixPath,
])
```

Two traps worth keeping: `train_cfg` is an `omegaconf.DictConfig`, **not** a `dict`, so a plain
`isinstance(cfg, dict)` check silently skips it — use `OmegaConf.to_container`. And run the
probe with `python -I`, since the checkpoint is untrusted third-party data.

### Still open after P0

`etab_source` / `field_source` are not in `train_cfg` — upstream infers them from the weights
via `PottsMPNN.infer_etab_source()`, and the checkpoint directory name
(`potts_v6_afdb_edge_his0.3_acid0.06`) says `edge`, consistent with `etab_out` mapping 128 → 900
rather than 3×128 → 900. Treat that as strongly indicated, not measured. P2 resolves it.

## 2. What the model is

| Fact | Evidence | Status |
| :-- | :-- | :-- |
| `class PottsMPNN(ProteinMPNN)`, foundry lineage | `pottsmpnn.py` (read) | **verified** |
| `self.potts_vocab_size = self.vocab_size if vocab_size is None else vocab_size` — a **constructor parameter**, not a module constant | `pottsmpnn.py` (read) | **verified** |
| etab is `[B, L, K, V, V]` with `V = potts_vocab_size`; head mode `"edge"` (`h_E`) or `"node_edge_node"` (`[h_i, h_E, h_j]`, 3H); field from `field_source ∈ {"self_edge", "node"}` | `pottsmpnn.py` | unverified (second-hand) |
| Merge rule `0.5·(etab + reverseᵀ)` applied **only to reciprocal edges**; double-counting of reciprocal pairs retained deliberately; slot 0 masked to its diagonal = single-site field | `model/POTTS_CONDITIONAL_ENERGY.md` | unverified (second-hand) |
| v6 alphabet = 30 tokens: 0–19 canonical, 20 `UNK`, 21–23 `HIS-P/S/A`, 24–26 `ASP-P/D/A`, 27–29 `GLU-P/D/A` | `transforms/feature_aggregation/token_encodings.py` | unverified (second-hand) |
| **Conflict:** `transforms/extended_vocab.py` reportedly says 32. Since `potts_vocab_size` is a parameter read from the checkpoint's `train_cfg`, **the ckpt settles this, not the source.** | — | unverified; P0 resolves |
| Checkpoint committed in-repo (`checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt`, ~21.5 MB); `torch.load` dict with `"model"` (loaded `strict=True`) and `"train_cfg"` | `potts_inference.py` | unverified (second-hand) |
| Design engine `PottsMPNNPHEngine` + `PHDesignCriteria`: block-descent redesign over pinned protonated centres, Pareto over stability vs selectivity gap `E_P − E_D` | `inference/design_ph.py` (imports read), README | partly verified |

Repository layout (verified): root holds `benchmarks/`, `checkpoints/`, `figures/`, `foundry/`,
`inference/`, `labeller/`, `scoring/`, `training/`, plus `install.sh` and
`requirements-extra.txt`. `inference/` holds `design_ph.py`, `design_ph.ipynb`,
`design_placement_scan.py`, `fold_rf3.py`.

## 2a. The etab head's conventions, now measured

Read from the vendored source 2026-10-07 and **spot-checked against the file by hand**, not
taken from a report. These were three of §12's unverified rows.

**`etab_source` is `edge` for this checkpoint.** `get_pottshead_input()` returns `h_E`
unchanged, so the head maps 128 → V². `node_edge_node` would consume `[h_i, h_E, h_j]` at 384.
`etab_out.weight`'s measured `(900, 128)` therefore settles the mode, independently of
`infer_etab_source()`.

**`field_source` writes the single-site field into slot 0.** Two branches, both verified:

```python
if self.field_source == "node":
    field = self.node_field(encoder_features["h_V"])          # [B, L, Vp]
    etab_out[:, :, 0, :, :] = torch.diag_embed(field)
else:                                                          # "self_edge"
    eye = torch.eye(self.potts_vocab_size, ...)
    etab_out[:, :, 0, :, :] = etab_out[:, :, 0, :, :] * eye
```

So `self_edge` zeroes the off-diagonal of the self-edge table and keeps its diagonal as the
field; `node` overwrites the whole slot with a dedicated head's diagonal.

**The merge is reciprocal-only, and non-reciprocal edges are untouched.** Verbatim:

```python
merged_etab = 0.5 * (etab_out + reverse_etab.transpose(-1, -2))
...
valid_merge = has_reverse & E_mask.bool() & reverse_mask.bool()
etab_out = torch.where(valid_merge[..., None, None], merged_etab, etab_out)
```

**The energy double-counts reciprocal pairs, deliberately.** `calc_potts_eners` is
`H(s) = Σ_i Σ_k etab[i, k, s_i, s_neighbour]`, implemented as
`etab[L_idx, K_idx, s_i, E_aa_j].sum(dim=(-1, -2))` — one scalar per *directed* edge. The
upstream note states the consequence rather than hiding it
(`POTTS_CONDITIONAL_ENERGY.md:44-48`): *"a **reciprocal** pair … contributes **two** terms →
reciprocal pairs are double-counted; a **non-reciprocal** pair contributes **one** term; the
matrices are directed and are **not** required to be symmetric."* The merge's 0.5 makes the
tables symmetric; it does **not** undo the double counting in the Hamiltonian.

**Consequence for the port, and it is a real difference rather than a restatement.** aminx's
`etab.py` exposes `merge_pair(denom, exclude_self)` with a d2/d4 + `exclude_self` convention.
That is *not* this rule. `protonpotts_merge` and `protonpotts_energy` (§7) must implement
reciprocal-only averaging and directed double-counting explicitly, and must not be satisfied by
reusing the existing helper.

## 3. The featurization is a front end, not a feature algebra

**This section corrects an earlier estimate.** A first scoping pass called the atomworks
featurizer a 2–3 day port that "may dominate the schedule" and named it the thing most likely to
make the project not worth doing. Reading it says otherwise.

`foundry/models/mpnn/src/mpnn/transforms/feature_aggregation/mpnn.py` holds three `Transform`
classes whose entire output surface is **nine keys** (verified):

| Transform | Keys |
| :-- | :-- |
| `EncodeMPNNNonAtomizedTokens` | `X` (coords), `X_m` (occupancy mask), `S` (sequence, int64) |
| `FeaturizeNonAtomizedTokens` | `R_idx` (int32), `chain_labels` (via `KeyToIntMapper`), `residue_mask` |
| `FeaturizeAtomizedTokens` | `Y` (coords, f32), `Y_t` (atomic numbers, int32), `Y_m` (mask) |

Its atomworks imports are `KeyToIntMapper`, `check_atom_array_annotation`, `Transform`,
`atom_array_to_encoding`, `get_token_starts` — a parsing-and-tokenizing surface, not a feature
algebra.

**That is the LigandMPNN / LASEr contract aminx already produces.** `PottsFeatures`
(`families/potts_mpnn/featurize.py:86-106`) carries `x`, `s`, `present`, `chain_m`,
`chain_m_pos`, `chain_encoding`, `residue_idx`; the `Y` / `Y_t` / `Y_m` ligand triple is already
built and consumed (`model/ligand_features.py`, `model/ligand_mpnn.py`, used at
`sampling/multistate_poe.py:442-444`).

So atomworks is doing *structure file → `AtomArray` → tokens → the same tensors*. The genuinely
new work above that is the thin **protonation-annotation layer**: `pka_annotation.py`,
`charge_network.py`, `bond_annotation.py`, `vocab_annotation.py`, `extended_vocab_v6.py` — plus
the `ev6` and `feature_aggregation` subpackages. That is a far smaller surface than "port the
featurizer", and it is the surface §6 says we should *not* try to bit-match anyway.

## 4. V1 scope: inference and design

V1 validates **inference + design**. The labeller and the training path are out.

Verified from `inference/design_ph.py`'s imports: the design path pulls `mpnn`,
`PottsMPNNPHEngine`, `PHDesignCriteria`, biotite's `PDBFile`, and local helpers
(`design_placement_scan`, `annotate_charge_clash`, `fold_rf3`, `annotate_folds`). It imports
**none of** `flaml`, `xgboost`, `lightgbm`, `sklearn`, or `hbplus`.

### CORRECTION 2026-10-07: that import list proves less than it looks, and two bullets here were wrong

An earlier revision of this section read the import list above and concluded "FLAML is not on
the V1 path" and "HBPLUS is not required". **Both were wrong**, and wrong in a way this project
has a standing lesson about: *audit transitive imports, not the file you read*. `design_ph.py`
genuinely imports none of them — but it reaches them through the featurization pipeline.

Measured in the vendored tree:

- **Protonation state is assigned by a LEARNED model, not a rule.**
  `classify_titratable_residues()` (`transforms/extended_vocab_v6.py`) runs the EV6 **5-fold
  FLAML ensemble** via `transforms/ev6/predictor.py`, and discretises with
  `if metal_adjacent or sd > sd_cut → "-A"` (ambiguous) `elif p >= prob_thr → "-P"` else
  `"-S"`/`"-D"`, where `p` and `sd` are the ensemble mean and spread.
- **HBPLUS is a real subprocess on that path.** `bond_annotation.py:198` defines
  `_run_hbplus_cmd(hbplus_cmd, pdb_path, …)`, and `extended_vocab_v6.py:100` says in terms:
  *"None makes EV6 run HBPLUS itself — correct, but two extra [calls]"*. `precomputed.py:3`
  names the whole cost: *"Protonation labelling (HBPLUS + biotite geometry + the EV6 fold
  ensemble) is the dominant per-structure [cost]"*.

**The escape hatch is real and is the actual decision.** `transforms/precomputed.py` plus
`ApplyProtonationThreshold` let a run consume *persisted* per-residue scores (`p`, `sd`,
`metal`, threshold-independent) and skip HBPLUS and FLAML entirely. So V1 has two genuinely
different shapes:

| V1 shape | needs | what aminx would have to reproduce |
| :-- | :-- | :-- |
| **consume pre-labelled structures** | nothing beyond torch + atomworks + foundry | the threshold rule only (`-P`/`-S`/`-D`/`-A` from stored `p`/`sd`/`metal`) |
| **label end to end** | HBPLUS binary + FLAML + sklearn/xgboost/lightgbm pins | the full EV6 ensemble, i.e. porting or shelling out to a 5-fold AutoML model |

This is **§11a**, and it is no longer "already answered" — it is the single biggest scope fork
in the project. It also revives the pins (`sklearn<1.9`, FLAML, xgboost/lightgbm) that this
section previously dismissed as labeller-only, *if* the second shape is chosen.

Unchanged and still true:

- **RF3 (~3 GB) is optional** — it folds candidate sequences *after* design for structural
  metrics. Out of V1.
- The design *engine* itself (`PottsMPNNPHEngine`) pulls none of these; the dependency enters
  upstream of it, at featurization.

What V1 must reproduce: featurize a structure → build the etab → compute conditional energies →
run `PottsMPNNPHEngine`'s block descent → emit designs and the stability/selectivity-gap Pareto
front.

## 5. The alphabet boundary

Protonation states change the token alphabet (21 → 30, pending P0). This is the one place the
existing Potts port is genuinely load-bearing and brittle:

- `families/potts_mpnn/potts_head.py:26-27` — `N_AA = 20`, `PAIR_DIM = N_AA * N_AA`, **module
  constants**.
- `families/potts_mpnn/etab.py:20-29` — `ETAB_ALPHABET = f"{MODEL_ALPHABET[:-1]}-X"`,
  `N_ETAB = len(ETAB_ALPHABET)` (22), `_MODEL_TO_ETAB` / `_ETAB_TO_MODEL`.
- `.praxia/docs/decisions/260605_potts-alphabet-alignment.md`'s `POTTS_TO_MPNN_ALPHABET_MAP`:
  identity on 0–19 still holds, but token 20 is `UNK` upstream where aminx's is `X`, and 21–29
  have no aminx counterpart at all.

Making `V` generic is therefore a refactor of **shipped, frozen, measured** code. It needs
goldens captured before it starts (P7's gate), and the existing Potts waves must still be green
afterwards.

A projection "canonical AA ← protonation variant" is needed wherever aminx assumes a 21-letter
sequence (FASTA sinks, `sequences_to_score`, DMS mutants). Whether that projection is the
*public* alphabet is a user decision (§11b).

## 6. Where the oracle boundary belongs

This is the design question the user raised: at what point do we need element-level parity
against an oracle, versus accepting that we are close enough and judging whole-pipeline
agreement with ProtonPottsMPNN?

**Put the seam at the feature dict.** Upstream's pipeline ends at the nine tensors of §3. Dump
them, and feed *those exact tensors* into aminx. That one move decouples the two halves and
makes each answerable by the method that suits it.

### Below the seam — numeric parity, exact in f64

Encoder, etab head, the reciprocal-only merge, conditional energy, the design engine's
arithmetic. Given identical input tensors these are deterministic functions, so they get the
`tests/port/` tier treatment: f64 exact, f32 to a **pre-registered measured floor** (never a
guessed band).

This is cheap and it is where defects hide. Our own evidence: the e2e parity wave is what
surfaced the real LigandMPNN O=CB defect (#2326) — a whole-pipeline comparison would have
absorbed it.

### Above the seam — a discrete agreement rate, not a tolerance

Parsing and protonation assignment. Trying to bit-match a parser is exactly where the earlier
estimate went overboard. The honest question is not "do the floats agree to 1e-8" but **"does
the annotator assign the same protonation token to the same residue?"** — a match rate over
fixtures, reported as a count, with disagreements enumerated rather than averaged away.

### The condition that makes "close enough" legitimate

Whole-pipeline agreement is an acceptable standard for a component **if and only if the
pipeline comparison carries a negative control that actually fires.**

We have a hard, recent demonstration of the failure mode in this project, not a hypothetical.
In the LASEr distributional confirm (`reference/261007_laser-distributional-confirm-result.md`)
both cells passed on χ1 — **and so did the negative control**, because `delta_chi = 0.05` was
five times the sequence band and the control's χ1 displacement was only ≈0.0120. A pass whose
control also passes measures nothing. That result is cited as "within the pre-registered band",
never as "control-verified", for exactly this reason.

So, as a rule for this port:

> A component may be accepted on whole-pipeline agreement when a deliberate perturbation of
> that component — flip one protonation assignment; scale the selectivity weight; permute one
> annotation — is **rejected** by the same pipeline-level comparison. If the perturbation is not
> rejected, the comparison is not an instrument, and that component moves below the seam and
> gets tensor-level parity instead.

This turns "where do we need an oracle" into a per-component measurement rather than a
judgement call, and it is pre-registerable: each wave's sidecar declares its perturbation
control up front.

## 7. Proposed parity waves

No tolerances are invented here. Every f32 band is marked for a pre-registered floor run, in the
shape of `scripts/analysis/laser_score_scale_relative_floor.py`.

| Wave | Compares | Tier shape |
| :-- | :-- | :-- |
| `protonpotts_vocab` | token encoding, aa↔token maps, aminx-alphabet projection round-trip | exact |
| `protonpotts_features` | the nine tensors vs upstream's post-pipeline dump | exact on ints/masks; floor run on coords |
| `protonpotts_protonation` | **above the seam**: protonation token assigned per residue | discrete match rate + enumerated disagreements |
| `protonpotts_encoder` | foundry `ProteinMPNN` encoder `h_V` / `h_E` | f64 exact; f32 floor run |
| `protonpotts_head` | etab `[L, K, V, V]`, both `etab_source` modes, both `field_source` | f64 exact; f32 floor run |
| `protonpotts_merge` | reciprocal-only `0.5(e + eᵀ)`; non-reciprocal untouched | exact |
| `protonpotts_energy` | `calc_potts_eners`, double-counting retained | f64 exact; f32 floor run |
| `protonpotts_conditional` | conditional energy per `POTTS_CONDITIONAL_ENERGY.md` | f64 exact; f32 floor run |
| `protonpotts_ph_design` | `PottsMPNNPHEngine` block descent on injected uniforms | exact tokens in f64; match rate in f32 |

Each wave's sidecar declares its **perturbation control** per §6. A wave whose control does not
fire grades `instrument_unverified`, not `pass`.

## 8. Tasks

| ID | Task | Depends | Gate |
| :-- | :-- | :-- | :-- |
| ~~P0~~ | ~~**Spike.** Load the ckpt; settle 30-vs-32; check encoder compatibility~~ **DONE 2026-10-07 — see §1a.** V=30, architecture is ProteinMPNN by name and shape, all f32 | — | answer recorded |
| P1 | ~~Vendor upstream at a pinned SHA~~ **done: `/home/marielle/repos/ProtonPottsMPNN` @ `09682abf`**; stand up `aminx-oracles-protonpotts/` on titanix (torch + atomworks + foundry, Py3.12) | P0 | env resolves; manifest |
| P2 | Probe report: key audit, `strict=True` behaviour, featurizer surface, engine entry points | P1 | doc |
| P3 | Vocab module, aa↔token maps, aminx-alphabet projection | P2 | `protonpotts_vocab` |
| P4 | Oracle dumps — the nine tensors + per-stage activations, f64 and f32, with draw shims | P1, P2 | sealed `.npz` + `oracle_manifest.toml` |
| P5 | Host featurizer: adapt the existing aminx path to the atomworks token contract (**not** a from-scratch port — see §3) | P2, P4 | `protonpotts_features`, `protonpotts_protonation` |
| P6 | Encoder: reuse-or-port decision from P0, plus weight conversion | P0, P4 | 0 unmapped keys; `protonpotts_encoder` |
| P7 | Generic-`V` `PottsHead` / merge / energy; refactor the existing Potts path to parameterised `V` | P3, P6 | 3 waves; **existing Potts waves still green against pre-captured goldens** |
| P8 | `ProtonPottsDriver`: family literal, options, `score:energy\|ddg`, sinks | P5, P7 | energy sidecar |
| P9 | pH design engine + selectivity-gap purposes | P8 | `protonpotts_ph_design` + sidecar |
| P10 | Knob surface, `_WEIGHT_PREFIXES` arm, ledger re-freeze, ADR, docs | P9 | full gate |

## 9. Cost and risk

P5's estimate is **reduced** from the first pass (2–3 d → ~1 d) on the strength of §3, and P6's
is reduced again by §1a — it was "2–4 d, unbounded"; with the architecture matching on names and
shapes it is a weight-conversion job plus a parity wave.

**Top risks, re-ranked after P0.**

1. **The generic-`V` refactor regresses the shipped Potts port.** Now the top risk. `N_AA = 20`
   and `PAIR_DIM = 400` are load-bearing in two modules of frozen, measured code, and the target
   is V=30 / PAIR_DIM=900. Capture goldens *before* P7 starts; re-run the existing Potts waves
   after.
2. **The atomworks front end.** Reduced but not eliminated: the tensor contract matches (§3), so
   what remains is fidelity of parsing and protonation assignment — which §6 deliberately places
   *above* the seam, as a discrete match rate rather than a float comparison.
3. **Encoder divergence — downgraded, not closed.** §1a found name- and shape-compatibility with
   the ProteinMPNN aminx implements, so this is no longer unbounded. It stays a risk until P6
   shows *numerical* agreement: structural agreement is not numerical agreement.

*(The original top risk, "wrong-lineage assumption", is retired — P0 ran and §1 is measured.)*

## 10. Freeze impact — budget it explicitly

`tests/knob_gate/_coverage.py:21-28` scopes `src/aminx/`, `scripts/parity/`,
`scripts/recapture/`, `tests/port/`, `aminx-oracles/`, plus `pyproject.toml` and `uv.lock`
(`_SCOPED_FILES`).

**Every task above except P0 and P2 writes into one of those.** So this sprint invalidates all
eight ledger rows and requires a full re-measurement plus a gate re-run. `_WEIGHT_PREFIXES`
(`_coverage.py:36`, currently `("pottsmpnn_", "lasermpnn_")`) needs a `protonpottsmpnn_` arm.

This is not free and should not be discovered halfway. It is also a sequencing argument: this
sprint should start **after** the current Potts/LASEr re-wave has re-measured those rows, so the
cost is paid once rather than twice.

## 11. Open decisions (the user's)

*(Until 2026-10-07 this list was mis-numbered `a, b, b, c, d` — two items carried the label
`b`, so "decision (c)" meant the second env in the doc and the alphabet in every summary written
from it. Renumbered `a`–`e` below. If an older note cites a letter past `b`, re-resolve it
against this list rather than trusting the letter.)*

a. ~~**THE BIG ONE — does aminx label protonation states, or consume them pre-labelled?**~~
   **DECIDED 2026-10-07 by the user: PRE-LABELLED.** See §18.3 for the evidence the decision was
   taken on, and §19 for what it binds. Pre-labelled means reproducing only the threshold rule
   and needs no new dependency. End-to-end labelling would have meant a HBPLUS binary plus the
   EV6 5-fold FLAML ensemble, reviving the `sklearn<1.9` / xgboost / lightgbm pins; §18.3
   measured that its default path is a hardcoded absolute path to the upstream author's laptop,
   so it does not run anywhere else without a separately-built binary. An earlier revision of
   this spec declared this closed on the strength of `design_ph.py`'s import list; that was
   wrong, and it was the largest scope fork in the project.
b. **OPEN — Design engine in V1?** This spec assumes yes (P9). The alternative is energy/scoring
   only, stopping at P8's `ProtonPottsDriver` with `score:energy|ddg` and deferring the engine —
   smaller, but then V1 does not validate the thing the repo exists for.
c. **OPEN — Public alphabet.** 30 tokens end-to-end, or a 21-token public alphabet with
   protonation as a side channel? Affects every sink and every `sequences_to_score` caller.
   §1a measured V=30 off the checkpoint, so 30 is what the model speaks; the question is whether
   that width reaches aminx's public surface or is projected down at the boundary (P3).
d. ~~**Second oracle environment on titanix.**~~ **ANSWERED 2026-10-07 by §17.4, not a choice.**
   `aminx-oracles/` pins `requires-python >=3.11,<3.12` and `rc-foundry` pins `>=3.12,<3.13`, so
   the two cannot share an interpreter — the second env was FORCED. Built, and it cost one
   `uv sync`, 2.0 GB and about a minute.
e. **OPEN — Where §6's seam sits for `protonpotts_features`.** Exact on coordinates, or a floor
   run? This spec proposes a floor run because the upstream path is f32 throughout.

*(An earlier revision closed the labeller/FLAML question here, claiming §4 showed it off the V1
path. That claim is retracted — it is now decision (a) above, and it is the biggest one.)*

## 12. Assumption register

**Verified by reading the source or the live repo:** the lineage and class declaration;
`potts_vocab_size` as a constructor parameter; the nine feature keys and their atomworks
imports; the `transforms/` and `feature_aggregation/` file listings; the repo root layout;
`design_ph.py`'s complete import list and the absence of flaml/xgboost/lightgbm/sklearn/hbplus;
aminx's `PottsFeatures` fields, `N_AA`/`PAIR_DIM`, and `ETAB_ALPHABET`; the KeatingLab pin;
`_SCOPED_PREFIXES`.

**Converted from lead to measured by P0 (§1a), 2026-10-07:** the alphabet is **30**, settled by
`W_s`/`W_out` at 30 and `etab_out` at 900 = 30², with `train_cfg.extended_vocab = 'v6'`; the
checkpoint's internal structure (120 tensors, hidden 128, 3+3 layers, all float32); and the
architecture's ProteinMPNN lineage by parameter name and shape. The pinned SHA is
`09682abfa7d20e0abcdeea0490b7a4b1c190aee3`.

**Still unverified, carried from a scoping subagent and flagged as leads, not facts:** the etab
merge rule (reciprocal-only `0.5(e+eᵀ)`) and the conditional-energy convention; `etab_source` /
`field_source`, which are inferred from weights rather than stored (the directory name says
`edge` and `etab_out`'s 128→900 shape agrees, so this is strongly indicated but not measured);
the design engine's internals and file size; the paper citation. **P2 exists to convert these.**
Nothing in §7–§8 that depends on them should be treated as settled until it does.

## 13. Where this work is tracked

Checked 2026-10-07, because it was not obvious and looking in the wrong place says "untracked".

**The sprint's remaining work lives in `praxia debt`, not the backlog.** Roughly 30 open debt
rows cover it — `#2417` (the 0.9539 refine gap), `#2433`/`#2443`/`#2459`/`#2435` (unimplemented
and inert Options fields), `#2445`, `#2481` (the χ1 control that cannot fail), `#2494`, `#2506`,
`#2507`, `#2309`, `#2319` (the redsox gate checklist), `#2067`–`#2070` (unported features).
The backlog's open rows are, by contrast, entirely re-homed cross-repo items from tev_design and
xtrax — none of them is Potts/LASEr sprint work.

Three things were tracked in neither and are now filed:

| row | covers |
| :-- | :-- |
| backlog **#5780** | the re-wave as a unit: the single scoped batch, A1's mechanical application, redsox, and the eight-row re-measurement |
| backlog **#5781** | this spec — ProtonPottsMPNN V1, P1–P10 |
| backlog **#5782** | the three unmeasured Potts distributional cells, and the decision they wait on |


## 14. P6's evidence: the encoder is reusable, and there is exactly one trap

Comparison run 2026-10-07 against the vendored upstream. Every claim reproduced below was
re-read from the source by hand before being written here; claims I could not reproduce are
marked as such rather than carried over.

### The verdict

**aminx's `encoder.py` and `decoder.py` need no change.** The message-passing layers are
operation-for-operation the same. P6 is therefore a weight conversion plus three changes
*outside* the encoder — and one input permutation that must not be forgotten.

Independent corroboration of the structure: 3 enc × 22 + 3 dec × 14 + `W_e`(2) + `W_s`(1) +
`W_out`(2) + `etab_out`(2) + featurizer(5) = **exactly 120**, matching the measured tensor
count in §1a. Two counts derived from different directions agreeing is worth more than either.

### Verified identical

| step | evidence |
| :-- | :-- |
| neighbour aggregation | upstream `dh = torch.sum(h_message, -2) / self.scale` with `scale=30` (`message_passing.py:107,172,316`); aminx `jnp.sum(message, -2) / scale` with `scale: float = 30.0` (`encoder.py:170,225`). **Sum, not mean** — a mean would have been a silent divergence. |
| residual / norm | both post-norm: dropout → add → norm |
| message MLP | `W3(act(W2(act(W1))))`, exact (non-approximate) GELU on both |
| dense block | 128 → 512 → 128 |
| LayerNorm eps | 1e-5 both |

### ⚠ The one that would silently destroy parity: RBF atom-pair order

**Verified directly, and it is real.** Upstream builds the 25 atom-pair blocks as a full outer
product — `X[:, :, None, :, None, :] - X_g[:, :, :, None, :, :]` then `.view(B, L, K, -1)`
(`graph_embeddings.py:450-480`) — so the flat slot is `5*i + j`, self-atom major. aminx uses an
explicit hand-ordered list, `BACKBONE_PAIRS` starting `[1, 1]` (Ca–Ca), then `[0,0]`, `[2,2]`…
(`utils/radial_basis.py:32-60`).

Both are 400 wide. **There is no shape error — only wrong numbers.** The fix is a permutation of
`edge_embedding.weight`'s input columns in 16-wide blocks, mapping aminx slot *s* with pair
`(p0, p1)` to upstream column block `5*p0 + p1`.

Also verified and not currently reproduced in aminx: foundry zeroes the RBF block by the atom
mask when any atom is absent (`RBF_all * X_m[...] * X_m_gathered[...]`, `graph_embeddings.py`).

#### This is NOT a latent defect in shipped aminx, and the distinction matters

Read carelessly, the two paragraphs above look like a bug report against aminx. They are not,
and nobody should file one. **aminx matches its own upstream exactly on both points** — checked
against the vendored `/home/marielle/repos/PottsMPNN` @ `0cb0a58`:

- **Ordering.** KeatingLab builds `RBF_all` by explicit `append`, and from index 5 its order is
  `Ca-N, Ca-C, Ca-O, Ca-Cb, N-C, N-O, N-Cb, Cb-C, Cb-O, O-C, N-Ca, C-Ca, O-Ca, Cb-Ca, C-N, O-N,
  Cb-N, C-Cb, O-Cb, C-O` (`potts_mpnn_utils.py:1190-1209`). That is **exactly** aminx's
  `BACKBONE_PAIRS` under 0=N, 1=CA, 2=C, 3=O, 4=CB, including the five self-pairs first and the
  `[1,1]` Ca–Ca lead.
- **Atom mask.** `_get_rbf(self, A, B, E_idx)` (`:1070`) takes no atom mask and applies none.

So **both differences are foundry-fork innovations**, not aminx omissions. The outer-product
flattening and the `X_m` zeroing were introduced downstream of the code aminx ports.

The positive evidence agrees: `potts_energy_parity` passes at `max_abs_delta` 7.3e-4 and
`potts_ar_decode` reaches exact token match 1.0, both end to end through this feature embedding
against sealed upstream oracles. A pair ordering wrong *relative to its own upstream* could not
produce either number.

**What this means operationally:** the permutation is a *porting requirement for
ProtonPottsMPNN*, to be applied when converting foundry weights — not a repair to aminx's
featurizer, which must keep its current order for the Potts and ProteinMPNN families it already
serves. If the two families ever share one featurizer, the ordering becomes a per-family
parameter rather than a constant.

### An open question about the decoder mask, stated precisely rather than called a bug

Upstream applies `mask_E` to the decoder **message**, after the MLP. aminx's decoder layer
*has* an `attention_mask` parameter and applies it (`decoder.py:287,334`) — but **both call
sites pass no such argument** (`decoder.py:648-653, 733-738`), so it is always `None`. The
conditional site carries the comment *"masking already applied to layer_edge_features"*, i.e.
aminx masks the edge features **before** the MLP instead.

Those placements are not equivalent in general: an MLP with biases maps a zeroed input to a
nonzero output, so a masked neighbour can still contribute. Whether it matters here depends on
the surrounding construction and only bites when padding is present. **UNRESOLVED** — resolve it
in P6 with a padded-input differential, not by reasoning.

### Changes needed outside the encoder

- `vocab_size` / `num_amino_acids` = **30** (not the default).
- `num_positional_embeddings` = **32**. It is the max *relative* feature, not an output width;
  `io/weights.py` already passes 32 for ProteinMPNN topologies, and the wrong value is a loud
  shape error rather than a silent one.
- A new 30×30 Potts head: aminx's `PottsHead` is hard-coded `N_AA = 20` → 400
  (`potts_head.py:26-27`) against the measured `etab_out` 900.

### Not checked, and named so nobody assumes otherwise

Causal/anti-causal mask equivalence beyond structure, the AR sampling path, symmetry groups,
noise injection, ligand and packer paths, `construct_X_atoms` token-indexing edge cases, and the
`_mlp`-headed (`etab_hidden`) variants. The C-beta construction was reported as algebraically
identical through a double sign flip; that is plausible but was **not** re-derived here.

## 15. Foundry normalizes BOTH orderings at weight-conversion time — this simplifies P6 and P7

Found 2026-10-07 while auditing the featurizer boundary
(`.praxia/docs/decisions/261007_featurizer-robustness-and-the-proxide-boundary.md` §10.1). It
changes the task graph, so it belongs here and not only there.

`foundry/models/mpnn/src/mpnn/utils/weights.py:~228-280` rewrites
`graph_featurization_module.edge_embedding.weight` on load: split off the first
`num_positional_embeddings` columns, `view(out_dim, num_atoms * num_atoms, num_rbf)`, reorder
the pair axis by `[legacy_order[name] for name in new_order]`, reshape, concatenate back. The
same file then permutes the amino-acid token order coming out of the model.

### 15.1 §14's RBF trap is absorbed by P6, not by a featurizer change

§14 recorded that aminx's `BACKBONE_PAIRS` order and foundry's flattened `5*i + j` order differ
with no shape error, and framed the remedy as a permutation of `edge_embedding.weight`'s input
columns. That is exactly what foundry already does — so **P6's weight conversion is the right
and only home for it**, and no runtime `pair_order` parameter is required for this port.

The alternative (a `FeatureSpec.pair_order` field consulted at featurization time) is only
needed if one featurizer must serve two live conventions *simultaneously*. ProtonPottsMPNN does
not require that: its weights are converted once, on the way in. **Prefer the conversion-time
permutation.** It costs nothing at runtime, adds no per-family branch, and keeps aminx's
featurizer bit-identical for the Potts and ProteinMPNN families it already serves — which is
what §14 said must not change.

### 15.2 The same applies to §5's alphabet boundary, and it narrows P3/P7

Foundry permutes the token order at conversion too. So the 30-token question splits cleanly:

- **Token ORDER** (which index means which protonated residue) — settle at conversion, like the
  pair order. No runtime machinery.
- **Token COUNT** (V=30 vs aminx's hard-coded 20/21) — genuinely runtime, and genuinely a
  refactor of frozen code. `potts_head.py:26-27`'s `N_AA = 20` → `PAIR_DIM` has to become
  parameterised, and §5's warning about goldens-before-refactor stands unchanged.

P3 therefore shrinks to the maps plus the aminx-alphabet projection; it does not also have to
negotiate ordering at runtime. P7 keeps its full scope — generic `V` is the real work.

### 15.3 Upstream already parameterizes what aminx hard-codes

`num_atoms = num_backbone_atoms + num_virtual_atoms`, and the pair count is `num_atoms²` — so
foundry's "25" is derived and generalizes to virtual atoms. Where aminx writes `16 + 16 * 25`
as a literal (`model/features.py:294`), foundry computes it. Two consequences:

1. The `FeatureSpec` idea in the boundary doc is not speculative — the fork has implemented its
   core, and it is a reference to copy rather than a design to invent.
2. If ProtonPottsMPNN ever uses virtual atoms, aminx's literal `25` becomes wrong in a way that
   **is** a loud shape error (the edge-embedding input width changes), unlike the ordering trap.
   Worth confirming during P6 whether this checkpoint has `num_virtual_atoms > 0`; §1a measured
   the vocab and the layer counts but not this.

### 15.4 Status of the task graph after this

P0 done (§1a). P1 half done — upstream vendored at `09682abf`; the `aminx-oracles-protonpotts/`
environment is **not** stood up, and that is now the critical path, because P4's oracle dumps
gate P5, P6 and P7. It is also the subject of open decision §11c. P2 done (§2a, §4's
correction). P6's structural half done (§14). Everything from P3 on touches `src/aminx/` and is
therefore **blocked by THE FREEZE until PR #165 merges**.

`aminx-oracles-protonpotts/` itself is **not** scoped: `_coverage.py:247` matches by literal
`path.startswith(prefix)` and `_SCOPED_PREFIXES` contains `aminx-oracles/`, which
`aminx-oracles-protonpotts/` does not start with. So P1 and P4 can proceed during the freeze —
they are the only ProtonPottsMPNN tasks that can.

## 16. P1: what the V1 oracle environment actually needs (traced, not assumed)

Traced 2026-10-07 while PR #165's CI ran. This sizes open decision §11c and sharpens §11a. It
is **static import tracing, not a built env** — every line below is a read of source at
file:line, and the authoritative check is still to build it and import the V1 entry point. Do
not treat this as P1's gate being met.

### 16.1 Upstream's own recipe

`install.sh` is `uv venv --clear --python 3.12 .venv` then a **single** resolution:
`uv pip install --python .venv/bin/python -e ./foundry -r requirements-extra.txt`. Its comment
says why it is one command: *"the resolver keeps BOTH sets — installing them separately can
prune the extras."*

`foundry/pyproject.toml`'s base dependencies are the heavy half and are not optional: `torch`,
`lightning`, `wandb`, **`atomworks[ml]>=2.1.1`**, `hydra-core`, `rootutils`, `environs`,
`jaxtyping`, `beartype`, `typer`, `loralib`, `einops`, `einx`, `opt_einsum`, `dm-tree`,
`zstandard`, `pandas`, `ipykernel`, `assertpy`, `toolz`.

### 16.2 `requirements-extra.txt` is mostly NOT V1 — with one exception that is

The extras file is the labeller, the notebook and training. Traced per entry:

| Dep | On the V1 path? | Evidence |
| :-- | :-- | :-- |
| `propka>=3.5` | **NO** | `import propka.run` is module-level at `transforms/pka_annotation.py:10`, and `pka_annotation` is imported by exactly one file, `pipelines/potts_mpnn.py:40`. `inference/` and `scoring/` import that pipeline **nowhere** (grep: zero hits). |
| `ipdb` | **YES — mandatory** | see §16.3 |
| `flaml`, `xgboost`, `lightgbm`, `scikit-learn<1.9`, `shap` | **only under §11a's labelling fork** | these are the EV6 labeller; needed iff aminx labels protonation rather than consuming it |
| `matplotlib`, `nbconvert`, `nbformat`, `jupyterlab` | no | notebook only |
| `pandas>=2.1,<3` | yes, but foundry already requires `pandas` | the pin exists because *"pandas 3.0 defaults to Arrow-backed strings, which FLAML 2.6 can't index"* — i.e. it is a FLAML constraint, so it relaxes if the labeller leaves V1 |

### 16.3 `ipdb` is required by a lazy import that V1 is guaranteed to make

This is the §4 transitive-import trap again, and it is worth stating precisely because the
"it's only a dev dependency" reading is wrong.

- `transforms/bond_annotation.py:20` — `import ipdb`, **module level**.
- `transforms/ev6/features.py:32` — `from mpnn.transforms.bond_annotation import calculate_hbonds`,
  **module level**. So importing `ev6` imports `bond_annotation` imports `ipdb`.
- `transforms/extended_vocab_v6.py:77,91` and `transforms/vocab_annotation.py:192` import
  `mpnn.transforms.ev6` **function-locally** — lazy, so a bare `import mpnn` does not pull it.

A lazy import is eager once its caller runs. §1a measured `train_cfg.extended_vocab='v6'` off
the checkpoint, so the V1 path *will* enter EV6, and `ipdb` becomes a hard runtime requirement.
`requirements-extra.txt` says as much in a comment — *"foundry's bond_annotation.py has a
top-level `import ipdb` (a dev dependency)"* — and it would be easy to drop while trimming
"dev" deps.

### 16.4 The minimal V1 environment, and what it means for §11c

**`-e ./foundry` plus `ipdb`.** Nothing else from `requirements-extra.txt`, *unless* §11a is
decided toward end-to-end labelling, which adds FLAML + xgboost + lightgbm + `sklearn<1.9` +
the `pandas<3` pin.

So §11c's "a second environment" is real but is **one `uv` resolution over foundry's own
dependency set**, not foundry-plus-the-extras. The heavy items are `torch` and
`atomworks[ml]` — both unavoidable, since atomworks defines the token contract §3 depends on.

Two things still unverified, and they are what P1's gate actually requires:
1. **Does `atomworks[ml]>=2.1.1` resolve from PyPI on titanix's Python 3.12?** If it is not
   published, the whole env plan changes and §11c becomes a much larger question. Not checked.
2. **Does HBPLUS need to be on PATH for V1?** `ev6/features.py:7` says it uses
   `bond_annotation.calculate_hbonds` *"instead of a private HBPLUS invocation"*, and
   `bond_annotation.py:198` is `_run_hbplus_cmd`. Whether a V1 inference run actually shells out
   depends on whether annotations are precomputed (`transforms/precomputed.py`, §4). This is the
   same question as §11a from the other side.

### 16.5 Recommendation

Build the env with foundry + `ipdb` only and try to import the V1 entry point. That both
answers (1) above and tests §11a's "pre-labelled" branch directly: if the minimal env can run
inference on a pre-annotated structure, the labelling fork is genuinely optional and §11a can be
decided on merits rather than on necessity.

## 17. P1 BUILT. The env resolves on titanix — with one pin I chose, and should not have

Built 2026-10-07 at `titanix:~/projects/aminx-oracles-protonpotts/`, following §16's
recommendation. **P1's gate ("env resolves; manifest") is met.**

### 17.1 What is there

| | |
| :-- | :-- |
| Upstream | `ProtonPottsMPNN/` cloned at `09682abfa7d20e0abcdeea0490b7a4b1c190aee3` (verified; no LFS — `.gitattributes` absent and `checkpoints/` is 21 MB of plain objects, so the clone is complete) |
| Manifest | `pyproject.toml` + `uv.lock`, mirroring `aminx-oracles/`'s shape |
| Env | `.venv`, Python 3.12, **120 packages, 2.0 GB** |
| Key pins | `torch==2.14.1+cpu`, `atomworks==3.0.0`, `numpy==2.5.3`, `lightning==2.6.6`, `biotite==1.6.0`, `ipdb==0.13.13`, `pandas==3.0.6` |

Deps are exactly §16.4's minimum: `rc-foundry` (editable, from the vendored tree) + `ipdb` +
an explicit `torch`. None of `requirements-extra.txt`'s labeller or notebook deps.

### 17.2 Verified by execution, not by reading

- `import mpnn, foundry` resolves to the vendored tree — upstream's own `install.sh` assertion
  (`'ProtonPottsMPNN' in inspect.getfile(mpnn)`) passes.
- `torch 2.14.1+cpu`, `torch.version.cuda is None`.
- `from mpnn.transforms import ev6` succeeds, and **`'ipdb' in sys.modules` is `True`
  afterwards.** §16.3 predicted this statically; it is now measured. The "dev dependency" really
  is a hard runtime requirement.
- All **13** `atomworks.*` modules foundry imports resolve under atomworks **3.0.0**, against
  foundry's `>=2.1.1` — a major version ahead. Import surface intact; *behaviour* under 3.0 is
  **not** tested by this.

### 17.3 A pin I got wrong, recorded because it was my own reasoning

`requirements-extra.txt` pins `pandas>=2.1,<3`, commenting: *"pandas 3.0 defaults to
Arrow-backed strings, which FLAML 2.6 can't index (ArrowStringArray has no .iloc); pin to the
2.x series **the model was developed with**."*

§16.2 read that as a FLAML-only constraint and concluded it "relaxes if the labeller leaves V1".
Excluding the extras therefore let pandas resolve to **3.0.6**. That inference is only half
supported: the stated *reason* is FLAML, but the stated *scope* is "the series the model was
developed with", which is broader. Any foundry path that indexes a pandas string column could
hit the same `ArrowStringArray` edge with no FLAML involved.

**I should have kept the pin and dropped it later with evidence, rather than dropping it on an
inference.** It is not a measured failure — nothing has exercised a dataframe path yet — so it
is recorded here rather than silently "fixed": the honest state is an unvalidated relaxation.
Next action before any P4 dump: either pin `pandas<3` and re-sync (cheap, reversible, matches
upstream), or exercise a dataframe-bearing path and show 3.0 is fine. Prefer the pin.

Two further version gaps worth stating now rather than discovering mid-dump: `numpy==2.5.3`
here against `numpy<2` in `aminx-oracles/`, and `torch 2.14` here against `torch==2.4.1` there.
Both are expected — the envs are deliberately separate — but it means **no oracle tensor from
one env should ever be compared bit-for-bit against the other's** without saying so.

### 17.4 §11c is now answered on evidence

"A second oracle environment on titanix" costs **one `uv sync`, 2.0 GB, and ~1 minute**, and it
was forced rather than chosen: `aminx-oracles/` pins `requires-python >=3.11,<3.12` and
`rc-foundry` pins `>=3.12,<3.13`. They cannot share an interpreter, so this was never really a
decision about preference.

### 17.5 What P1 did NOT establish

The env imports; it has not run the model. Specifically still open: loading the checkpoint
through foundry's own loader (which is where §15's conversion-time pair-order permutation
lives), whether HBPLUS must be on PATH for a V1 run (§16.5), and the two `atomworks` data
mirrors it warns about at import — `CCD_MIRROR_PATH` and `PDB_MIRROR_PATH` are unset, which
atomworks reports as "will not be able to use function requiring this variable". That is likely
to matter for featurization (P5) and is not yet scoped.

## 18. Running the env corrected three of my own claims, and answered §11a with evidence

Everything here is **measured on titanix**, by building the engine and featurizing upstream's
own example. It supersedes §16's static tracing wherever the two disagree — and they disagree
three times, which is the point of building the thing.

### 18.1 I was WRONG that propka is off the V1 path

§16.2 said propka was unreachable. The engine build failed immediately on it. The real chain is
two hops deeper than I traced:

```
mpnn/inference_engines/potts_mpnn_ph.py:56 -> mpnn.potts_inference
mpnn/potts_inference.py:22                 -> mpnn.pipelines.potts_mpnn
mpnn/pipelines/potts_mpnn.py:40            -> mpnn.transforms.pka_annotation
mpnn/transforms/pka_annotation.py:10       -> import propka.run
```

**Why I got it wrong:** I grepped the repo's top-level `inference/` and `scoring/` directories
for importers of `pipelines.potts_mpnn` and found none — true, and irrelevant, because the V1
engine lives *inside* the foundry package at `mpnn/inference_engines/`. The right question was
"what does the entry point import, transitively", not "what does this directory import".
Third instance of this trap in this project; the first two are §4's FLAML correction and
§16.3's ipdb chain.

### 18.2 atomworks 3.0 BREAKS V1 — measured, and foundry's floor has no ceiling

`rc-foundry` asks `atomworks[ml]>=2.1.1` with **no upper bound**, so 3.0.0 resolved. Under it,
`_build_context` dies inside atomworks itself:

```
mpnn/utils/inference.py:713 -> atomworks.io.parser.parse_atom_array
atomworks/io/_pipeline.py:148 -> template.add_missing_atoms
ValueError: Input atom_array is missing 'charge' annotation.
  Ensure the structure was loaded via parse() or get_structure(), which always sets charge.
```

The irony is exact: the structure *was* loaded via `get_structure()` — upstream's own notebook
does `PDBFile.read(str(PDB)).get_structure(model=1)` at `design_ph.py:58`. biotite's
`get_structure` does not set `charge`; atomworks 3.0 began requiring it. Pinned
`atomworks[ml]>=2.1.1,<3`, which resolves 2.2.1, and the parse then succeeds.

§17.2 recorded that all 13 atomworks modules *imported* fine under 3.0 and explicitly said
behaviour was untested. It was right to say so: the import surface was intact and the
behaviour was not.

### 18.3 §11a is answered on evidence: HBPLUS is on the default V1 path, and its default is unrunnable

With atomworks pinned, featurization proceeds all the way to:

```
mpnn/transforms/bond_annotation.py:339 calculate_hbonds
mpnn/transforms/bond_annotation.py:199 _run_hbplus_cmd -> subprocess.run
FileNotFoundError: '/Users/chrjac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/PHD/tools/hbplus/hbplus'
```

Two things follow, and the second is an upstream defect worth reporting:

1. **HBPLUS really is on the default V1 featurization path** — not post-hoc, not optional. This
   confirms §4's CORRECTION by execution and closes the last doubt about it.
2. **The default is a hardcoded path to the upstream author's own laptop.**
   `bond_annotation.py:289-297` reads `HBPLUS_PATH` from the environment, then at `:292`
   assigns that absolute Mac path when it is unset — which makes the helpful error at `:296`
   (*"HBPLUS_PATH environment variable not set. Please set it..."*) **unreachable dead code**,
   because the `:294` emptiness check can never be true after `:292`. So a clean checkout gets
   a raw `FileNotFoundError` naming a stranger's OneDrive instead of the actionable message the
   author wrote.

**Consequence for decision §11a.** The "end-to-end labelling" branch does not merely add
FLAML + xgboost + lightgbm + `sklearn<1.9`; it requires an **HBPLUS binary**, which is not on
PyPI, must be obtained and built separately, and which upstream cannot even locate portably.
The "consume pre-labelled structures" branch (`transforms/precomputed.py` +
`ApplyProtonationThreshold`, §4) is therefore not just the lighter option — on this evidence it
is the only one that runs anywhere but the author's machine without extra tooling.
**Recommend (a) = pre-labelled**, and treat end-to-end labelling as a later, separately-scoped
capability.

### 18.4 The env as it now stands

`rc-foundry` (editable) + `torch>=2.2,<3` (CPU) + `ipdb` + `pandas>=2.1,<3` + `propka>=3.5` +
`atomworks[ml]>=2.1.1,<3`. **Every pin beyond `rc-foundry` was added because something failed**,
and the manifest says which failure next to each one. Still absent, and now known not to block
engine construction: `flaml`, `xgboost`, `lightgbm`, `scikit-learn`, `shap`.

Verified: the engine **builds** from the v6 checkpoint. Featurization gets as far as HBPLUS and
stops there for an environmental reason, not a dependency one.

### 18.5 Next, in order

1. ~~Decide §11a (recommendation above).~~ **DECIDED: pre-labelled — see §19.** The follow-on
   instruction here said "find the `precomputed.py` entry point"; §19.1 shows that was pointed at
   the wrong module.
2. If end-to-end is wanted instead, obtain HBPLUS and set `HBPLUS_PATH`; the hardcoded fallback
   should be reported upstream regardless.
3. Only then P4's oracle dumps. Nothing should be dumped from an env whose featurization path
   has never completed once.

## 19. §11a DECIDED: pre-labelled. What that binds, read off the pipeline rather than assumed

**The user chose pre-labelled on 2026-10-07**, on the §18.3 evidence: HBPLUS is on the default V1
featurization path and its default resolves to a hardcoded absolute path on the upstream author's
laptop, so the end-to-end branch does not run anywhere else without a separately-built binary that
is not on PyPI.

### 19.1 `precomputed.py` is the TRAINING snapshot cache, not an inference entry

§18.5 step 1 told the next session to re-run the context build "through the `precomputed.py` entry
point". That module is `mpnn/transforms/precomputed.py`, and its own docstring says the opposite of
what I implied: *"Training / validation are PRECOMPUTED-ONLY … EV6 runs live ONLY in the PKAD /
MegaScale benchmark callbacks."* It is a `save_snapshot` / `load_snapshot` pair plus a dataset
`loader`, and it exists because EV6 is a FLAML/tree ensemble whose OpenMP deadlocks inside a forked
DataLoader worker. It is a training-throughput device. Following that instruction literally would
have sent the next session into the training path looking for an inference smoke test.

### 19.2 The real lever is a pipeline flag, and it skips exactly the right block

`build_mpnn_transform_pipeline(..., precomputed=True)` at `pipelines/potts_mpnn.py:221`. Its own
comment states the contract: when true, the cleaned+annotated FULL structure is supplied by the
loader, so the pipeline *"SKIPS `get_cleanup_transforms` and the whole HBPLUS/annotation block — it
runs only the chain crop + featurization tail."* `precomputed_dir` is informational there; the
loader owns the lookup.

At `is_inference=True` the chain crop is `Identity()` anyway (`:248`), so the inference +
precomputed combination reduces to the featurization tail alone.

> **CORRECTION, same day, before this paragraph was acted on.** The sentence that stood here —
> *"pre-labelled is a supported upstream configuration, not a bypass we invent"* — is true of the
> PIPELINE layer and **false of the INFERENCE layer**, and the distinction is the whole
> difference between "already works" and "we must build it."
>
> `potts_inference.py:113-121` builds the V1 pipeline with an explicit argument list that **does
> not include `precomputed`**, so it takes the default `False` and the inference path always runs
> the full HBPLUS/annotation block. Grepping every caller that sets the flag in the whole
> repository returns exactly two, both in the trainer: `train.py:306` and `train.py:323`. There
> are **zero** inference callers, and `grep -rn precomputed inference_engines/ potts_inference.py`
> returns nothing at all.
>
> So `precomputed=True` is, in practice, a training-only flag. Reaching it from inference means
> calling `build_mpnn_transform_pipeline` directly and bypassing `prepare_potts_input`, or
> patching the engine. That is a small job, but it is a job, and it was not visible from the
> pipeline's docstring alone. **I read the flag's definition and inferred its reachability; the
> call sites say otherwise.** Same shape as §18.1 — the fourth time on this spec that reading a
> definition rather than its callers produced a wrong claim.
>
> One more thing the call site shows: `_get_potts_pipeline` defaults `extended_vocab="v4"`, but
> §1a measured the checkpoint at v6. Any caller we write must pass `extended_vocab="v6"`
> explicitly or it will silently featurize under the wrong vocabulary — and the vocabulary owns
> the H-bond cutoffs (§19.3), so this is not a cosmetic default.

### 19.3 What aminx must therefore supply — a block, not a threshold

§11a's old wording said pre-labelled "means reproducing only the threshold rule". That understates
it, and the understatement should not survive into P5. The skipped block is four transforms
(`get_protonation_state_transforms`, `:111-157`):

| Transform | Why it matters to the contract |
| :-- | :-- |
| `RemoveHydrogens` | H is stripped FIRST, deliberately: with deposited H the label would be read off the observed answer, and EV6 trained H-stripped |
| `CalculateHbondsPlus` | the HBPLUS call; cutoffs come from the **vocabulary**, not from free knobs |
| `AnnotateSaltBridges` | `dist_max=5.5`, `min_dist=0.5`, PLIP criteria |
| `AnnotateProtonationStates` | the actual labelling; `deterministic=True` at inference |

The input contract is the block's **output on the atom array**, not a scalar rule. One consequence
is load-bearing and belongs in P5: the docstring records that **v6 consumes this pass's bonds
directly** via `data["hbond_records"]`, so a pre-labelled structure must carry the bond records
too, not only the per-residue protonation labels. A snapshot with labels but no `hbond_records`
would be silently under-specified for exactly the checkpoint we are porting.

### 19.4 What this decision does NOT settle

It does not make V1 runnable yet, and per §19.2's correction it does not even name a reachable
configuration — the flag it relies on has no inference caller. The featurization path still has
never completed once in this env, so §18.5's rule stands unchanged: nothing is dumped for P4 until
it does. §20 states what still has to be chosen to get there. The open decisions for the user are
now §11b (design engine in V1) and §11c (public alphabet width), plus §20's dump-input question.

## 20. The decision governs AMINX's runtime; it does not by itself unblock the ORACLE DUMPS

These are two different questions and §19 ran them together. Separating them:

- **What must aminx do at runtime?** Consume labels. Settled by §11a.
- **What must happen once, to produce P4's reference tensors?** Upstream has to featurize a real
  structure. Under `precomputed=False` that needs HBPLUS. Under `precomputed=True` it needs an
  annotated structure — which something has to have annotated.

So the labels have to come from somewhere even in the pre-labelled world, and **HBPLUS is absent
from titanix**: `which hbplus` is empty, no `hbplus*` anywhere under a depth-4 filesystem sweep,
and there is no conda/mamba/micromamba on the box to fetch one from a bioconda channel.

Three ways out, with the one I would not pick named as such:

1. **Build HBPLUS once** on titanix, set `HBPLUS_PATH`, run the default path. Cleanest reference
   tensors — they are the labels the model was trained to expect. Cost: obtaining and compiling
   third-party C source, which is a user-facing step, not one to take unasked.
2. **Synthesise labels** and drive `precomputed=True` directly. Legitimate *for a parity test*,
   because parity asks whether two implementations agree on the same input — the labels are an
   input, and they do not need to be chemically correct, only identical on both sides and
   well-formed. **Caveat that decides whether this works at all:** §19.3 established that v6 also
   consumes `data["hbond_records"]` directly, so synthetic per-residue labels are not sufficient
   on their own; the bond records must be synthesised too, and it is unverified whether EV6's
   feature path tolerates an empty or hand-built bond list. Spike that before committing to it.
3. **Reuse a PKAD benchmark structure** — `benchmarks/data/PKAD/pdb_cache/` ships real PDBs
   (3EVQ, 3RUZ, 1NLX, …). This does NOT dodge the problem: they are inputs, not annotations, so
   they still need HBPLUS. Named only so nobody mistakes the cache for a shipped snapshot. No
   annotated snapshot is vendored anywhere in the repo.

**Not recommended: deciding this silently inside P4.** It changes what the reference tensors mean
— option 1 dumps the model's real operating point, option 2 dumps an arbitrary but reproducible
one — and that belongs in the pre-registration, not in a dump script.

Worth recording while the tree is in front of us: the FLAML labeller pickles **are** vendored
(`labeller/models/automl_feature_{acid,HIS}/{full,fold1..5}.pkl`, present in the clone, not LFS
pointers). So the end-to-end branch was never blocked on the *models* — only ever on the HBPLUS
binary that feeds them.

## 21. MEASURED: V1 featurization completes with NO annotation and NO HBPLUS. §18.5's gate is met

Spiked on titanix, 2026-10-07. This is the first time the featurization path has completed in this
environment, which was the standing precondition on P4.

### 21.1 The spike, and why it is faithful

One intervention only: `mpnn.pipelines.potts_mpnn.get_protonation_state_transforms` — the sole
HBPLUS consumer — replaced by `lambda **kw: []`. Everything else, **including the structure
parse**, is upstream's own: the spike calls `prepare_potts_input(pdb, extended_vocab="v6")`, which
is the V1 entry point. Input was upstream's shipped example,
`inference/examples/pdl1_seed_binder.pdb` (1689 atoms).

```
network_input keys: ['atom_array', 'input_features']
X    (1, 229, 37, 3) torch.float32
X_m  (1, 229, 37)    torch.bool
S    (1, 229)        torch.int64
S    min=0 max=19 n_distinct=20
```

`input_features` carries the full 19-key surface (`R_idx`, `bias`, `pair_bias`,
`designed_residue_mask`, `chain_labels`, `residue_mask`, `symmetry_*`, …), i.e. the real network
input, not a stub.

A first attempt failed differently and the distinction matters: calling
`build_mpnn_transform_pipeline` on a bare `PDBFile.get_structure()` died at
`RemoveUnresolvedPNUnits` for missing `pn_unit_iid` / `occupancy`. That is **input preparation**,
not HBPLUS — `MPNNInferenceInput.from_atom_array_and_dict` supplies those annotations. Routing
through `prepare_potts_input` fixed it.

### 21.2 This retires §20's expensive caveat

§20 said synthesising labels was gated on an unverified question — whether EV6 tolerates a
hand-built `hbond_records`. **That question does not need answering**, for two independent
reasons, both now confirmed:

1. `hbond_records` is consumed by **EV6 the labeller** (`ev6/predictor.py:113-121`,
   `vocab_annotation.py:120,135`), which lives inside the annotation block. Skip the block and EV6
   never runs. (Also worth noting from `predictor.py:117`: passing `hbond_records` *"skips two
   HBPLUS subprocess calls"* — so EV6 given `None` would invoke HBPLUS itself. A further reason
   not to route through it.)
2. `BuildBondEdgeLabels` — the only tail consumer of bond labels — is gated on
   `build_bond_labels`, which **defaults `False`** (`potts_mpnn.py:186`). The default featurization
   never asks for them.

And the labels themselves turn out to be optional, not merely synthesisable:
`EncodePottsMPNNNonAtomizedTokens.check_input` requires only `atomize`, `res_name`, `occupancy`,
and `_build_protonation_aware_seq` says so outright — *"`protonation_label` is optional; when
absent the function behaves identically to the standard encoder"* — gating on
`has_protonation = "protonation_label" in atom_array.get_annotation_categories()`. So §20 option 2
is cheaper than written: **no synthesis at all** is needed for a canonical-token cell.

### 21.3 ⚠ But this cell is VACUOUS with respect to the capability being ported

`S max=19`, `n_distinct=20` — the twenty standard amino acids and nothing else. Not even `UNK`
(index 20) appeared, and **no protonation token did**, because with no labels every residue falls
back to its canonical `res_name`.

So a parity suite built only on this cell would compare aminx against upstream on a path where
ProtonPottsMPNN is indistinguishable from PottsMPNN. That is precisely the vacuity failure this
project already audits for (T0 found exactly one genuinely vacuous comparison field and it was
treated as a defect). **Stated as a pre-registration constraint: no `protonpotts_*` wave may
consist solely of cells whose `S` never exceeds 20.** At least one cell must carry populated
protonation labels, which is now easy — the label is just an atom-array annotation, and the
fallback is keyed on its presence.

### 21.4 The v6 alphabet, read off the source — this makes §11c concrete

`token_encodings.py:135-162`. **30 = 21 standard + 9 protonation**, where 21 is
`STANDARD_AA + (UNKNOWN_AA,)` at indices 0–20, and the 9 occupy **21–29**:

| residue | tokens |
| :-- | :-- |
| HIS | `HIS-P` (protonated), `HIS-S` (neutral), `HIS-A` (ambiguous) |
| ASP | `ASP-P`, `ASP-D` (deprotonated), `ASP-A` |
| GLU | `GLU-P`, `GLU-D`, `GLU-A` |

Two facts that bear directly on the public-alphabet decision:

- The extension touches **exactly three residues**, each getting exactly three states. A 21-token
  public alphabet plus a side channel therefore needs only a small per-position enum that is
  non-null at H/D/E and null everywhere else — far lighter than threading width 30 through every
  sink and every `sequences_to_score` caller.
- v6 **predicts charge state, not tautomer**: the source comment records that neutral His is a
  *single* token `HIS-S` because the HID/HIE tautomers and the rare imidazolate `HIS-D` are
  deliberately dropped. So the side channel does not need to carry tautomer information — a
  3-valued charge state per titratable residue is complete. This also confirms §2a's reading that
  v6 is deliberately weight-incompatible with the 32-token v3/v4 encodings.

I am not deciding §11c here — it stays the user's — but the option space is narrower and cheaper
than the spec implied.

### 21.5 Where this leaves the task graph

P4's precondition is now **met**: featurization has completed once. What is still unestablished is
whether the *engine* runs forward on these features to produce energies — the spike stopped at
featurization and did not load the checkpoint through `potts_mpnn_ph`. That is the next probe, and
it needs no user decision.

Unchanged: everything from P3 on touches `src/aminx/` and is blocked by THE FREEZE until #165
merges. `aminx-oracles-protonpotts/` remains unscoped, so P1/P4 stay the available work.

## 22. MEASURED: the V1 engine runs forward. P4 is fully reachable, and a silent v4/v6 hazard

Second spike, same day, same single intervention (HBPLUS block neutralised, nothing else).
Loads the shipped v6 checkpoint and calls upstream's own `load_model` + `run_forward`.

```
model loaded: PottsMPNN
extended_vocab_name = 'v6'            <- auto-detected from the checkpoint
etab_out   (1, 229, 48, 30, 30) torch.float32   finite, absmax 36.9262
E_idx      (1, 229, 48)         torch.int64
S_argmax   (1, 229)
S_sampled  (1, 229)
log_probs  (1, 229, 30)
```

Checkpoint: `checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt`.

### 22.1 What this settles

- **P4 is fully reachable.** `etab_out` is the tensor the oracle dumps exist to capture, and it
  comes out finite on upstream's own example with no HBPLUS anywhere in the path. The P1→P4
  critical path is open.
- **V = 30 confirmed by EXECUTION**, not only by weight-shape inference. §1a read 30 off the
  weights; the running engine emits `[1, L, K, 30, 30]`. Independent confirmation of §1a's
  central measurement.
- **K = 48** neighbours in the Potts graph — recorded because it is a featurizer/geometry
  constant the port must match and it had not been measured before.
- `load_model` **auto-detects the vocabulary** from `train_cfg.extended_vocab` and reports
  `extended_vocab_name='v6'` with no explicit argument, exactly as its docstring claims.

### 22.2 ⚠ A silent v4/v6 mismatch, and it hides in precisely the cell we can already run

§19.2 flagged that `_get_potts_pipeline` defaults `extended_vocab="v4"`. §22.1 shows the *model*
auto-detects v6. Those two facts together are worse than either alone:

**The model self-corrects to v6; the featurizer does not.** A caller who uses
`prepare_potts_input(pdb)` without passing `extended_vocab="v6"` gets a v4-built `S` fed into a
v6-sized model — and nothing raises, because `S` is `int64 [1, L]` either way.

Why it cannot be caught by the obvious test: `potts_token_order = token_order + PROTONATION_TOKENS`
and `potts_v6_token_order = token_order + POTTS_MPNN_V6_PROTONATION_TOKENS` share the **same
21-token `token_order` prefix** (`token_encodings.py:140,158`). So for any structure with no
protonation labels, v4 and v6 produce **bit-identical `S`**. The vocabularies diverge only at
indices ≥ 21 — the 11 v4 protonation tokens versus the 9 v6 ones.

The consequence is sharp, and it compounds §21.3: **the annotation-free cell cannot detect this
bug, and the labelled cell is the only one that can.** A port validated solely on the cell that
runs today would pass while carrying a wrong-vocabulary featurizer. So §21.3's constraint is not
merely about avoiding a vacuous comparison — it is the only thing standing between this defect and
a green parity suite.

Recorded as a hard requirement for P5/P6: **pass `extended_vocab` explicitly at every featurizer
construction site; never rely on the default.** And the port's own featurizer must take the
vocabulary as a parameter rather than a constant — which is the same conclusion §15.3 reached from
`num_atoms`, arrived at from a second direction.

### 22.3 Still open after this

- §15.3's `num_virtual_atoms` question is **not** answered by `etab_out`'s shape — that axis is
  the vocabulary, not the atom-pair count. Still an open measurement, and it is the one that
  decides whether aminx's literal `25` is wrong.
- Nothing here exercises protonation tokens 21–29; `S` is still canonical-only.
- The engine ran on CPU torch. No GPU/f32 determinism claim is made or implied.

## 23. A non-vacuous cell is constructible — and `etab_out` is NOT the observable that makes it so

Two more spikes. The first builds the labelled cell §21.3 demanded; the second checks whether it
actually bites, and the answer corrects §22.1.

### 23.1 The labelled cell runs, and it confirms the alphabet indexing by execution

Instead of deleting the annotation block, **substitute** a deterministic synthetic labeller for it
(`HIS→HIS-S`, `ASP→ASP-D`, `GLU→GLU-P` by `res_name`); everything downstream stays upstream's own.

```
labelled 183/1689 atoms across ['ASP', 'GLU', 'HIS']
S min=0 max=27 distinct=[0,1,2,4,5,7,9,10,11,12,13,14,15,16,17,18,19, 22,25,27]
etab_out (1,229,48,30,30) finite
PASS: non-vacuous labelled cell runs end-to-end with NO HBPLUS
```

The three protonation indices are **22, 25, 27**. Under §21.4's ordering (21 standard, then
`HIS-{P,S,A}`, `ASP-{P,D,A}`, `GLU-{P,D,A}` at 21–29) those are exactly `HIS-S`, `ASP-D`, `GLU-P`
— precisely the three labels injected. So **§21.4's index arithmetic is confirmed by execution**,
not merely read off a tuple. Corroborating detail: indices 3, 6 and 8 (`ASP`, `GLU`, `HIS`) have
*vanished* from the distinct set, as they must if every titratable residue was relabelled.

So the pre-labelled design needs no HBPLUS even for a cell that exercises the new tokens.

### 23.2 ⚠ But `etab_out` is invariant to `S` — measured, bit-exact

Both cells reported `absmax=36.9262`, identical to six figures. Running them in one process and
differencing the tensors:

```
S differs at 21/229 positions   (max 19 unlabelled -> 27 labelled)
etab_out   identical=True   max|delta|=0
E_idx      identical=True
log_probs  identical=False  max|delta|=48.2565
```

`etab_out` is **bit-identical** while `S` changes at 21 positions. That is by construction, not a
bug: a Potts model emits an energy *function* over the vocabulary from the structure encoder, and
the sequence selects entries from it afterwards. `E_idx` is likewise structure-only.

**This corrects §22.1.** I wrote there that `etab_out` "is the tensor the oracle dumps exist to
capture". It is *a* tensor worth capturing, but it cannot be the one that validates protonation
handling, because protonation labels do not reach it.

### 23.3 The precise statement, replacing §21.3's rule

Being exact about what `etab_out` does and does not cover, because "vacuous" is too blunt:

| observable | validates | does NOT validate |
| :-- | :-- | :-- |
| `etab_out` (1,L,K,30,30) | encoder + Potts head at V=30; the architecture port; a wrong *vocabulary width* | anything about whether labels reach `S` |
| `E_idx` (1,L,K) | the neighbour graph, K=48 | same |
| `log_probs` (1,L,30) | the label→`S`→score path; sequence dependence | — |

The consequence for §22.2's silent v4/v6 featurizer mismatch is sharp: **`etab_out` would not
catch it either**, since that defect corrupts `S` and `etab_out` ignores `S`. Only a
sequence-dependent observable can.

**Superseding §21.3's wording.** The constraint is not merely "at least one cell must carry
protonation labels" — it is:

> A `protonpotts_*` wave must compare at least one **sequence-dependent** observable
> (`log_probs`, or an energy evaluated at `S`) on a cell whose `S` contains tokens ≥ 21.
> A wave comparing only `etab_out`/`E_idx` is PottsMPNN structure parity, however many
> protonation tokens the cell contains.

`max|delta| = 48.2565` on `log_probs` between the two cells is the margin showing that observable
genuinely moves — it is a positive control for the instrument, available before any parity run.

### 23.4 One caching trap worth keeping

`_get_potts_pipeline` memoises on `(device, build_bond_labels, hbond_scope, extended_vocab,
use_salt_bridge, deterministic, protonation_seed)` — **not** on the annotation block. Patching
`get_protonation_state_transforms` between two cells in one process therefore has no effect unless
`PI._PIPELINE_CACHE` is cleared first. The §23.2 comparison clears it explicitly; without that it
would have silently compared the first cell against itself and "proved" invariance trivially.

## 24. §15.3's open measurement is CLOSED: `num_virtual_atoms = 1`, so aminx's literal 25 is right

Read off the loaded checkpoint, not off a fixture.

```
graph_featurization_module: PottsProteinFeatures
num_backbone_atoms = 4
num_virtual_atoms  = 1          ->  num_atoms = 5,  pair count = 25
num_rbf = 16
num_positional_embeddings = 16
edge_embedding.weight (128, 416)
```

§15.3 flagged `num_virtual_atoms > 0` as the open question that would make aminx's hard-coded `25`
**wrong** — loudly, as an edge-embedding width mismatch. It is `1`, but the backbone count is `4`,
so `num_atoms = 5` and the pair count is `5² = 25`. **aminx's literal 25 is correct for this
checkpoint.** The risk §15.3 named does not materialise, and P6 shrinks accordingly.

Worth separating the two things that happen to coincide: `4 + 1 = 5` is *not* the same fact as
"aminx uses the 5 backbone slots". Upstream reaches 5 as 4 real backbone atoms plus one virtual
atom (the CB pseudo-atom, by the usual convention); aminx reaches 25 as a 5×5 loop over
`backbone_coords`. They agree on the number today. They would stop agreeing the moment a
checkpoint ships `num_virtual_atoms = 2`. So §5's recommendation to parameterise rather than
hard-code stands on robustness grounds — it is just no longer a *correctness* fix for this port.

### 24.1 ⚠ This does NOT discriminate between the two `edge_in_dim` formulas

The measured `edge_in_dim = 416` reproduces the corrected formula exactly:

```
pos_embed_dim + rbf_count * n_pairs  =  16 + 16 * 25  =  416   ✓
```

But the formula I originally wrote and later corrected also lands on 416:

```
rbf_count * (len(pair_order) + 1)    =  16 * (25 + 1) =  416   ✗ (still wrong, still 416)
```

They coincide **precisely because `num_rbf == num_positional_embeddings == 16` on this
checkpoint** — which is the degeneracy that let the wrong formula survive unnoticed in the first
place. So this measurement confirms the *value* and does **not** confirm the *formula*. The
correction rests on foundry's own arithmetic (`weights.py:242-243`, splitting off the first
`num_positional_embeddings` columns before viewing the remainder as
`(out_dim, num_atoms*num_atoms, num_rbf)`), not on this number.

Recorded so nobody later cites 416 as evidence the formula is right. A checkpoint with
`num_rbf != num_positional_embeddings` is the only thing that would discriminate them empirically,
and none is in hand.

Incidental confirmation: 416 is also the literal at aminx's `ligand_features.py:250` (the real
constant — `:249` is a comment, per §8's audit correction), so that site is consistent with this
checkpoint too.


---

## §25 — alphex and the variable alphabet: not used, and not usable as it stands

Asked directly by the user 2026-10-07: *"are we using alphex to manage the variable
alphabet?"* The short answer is **no**, on both halves — we are not using it, and the library
cannot currently express the alphabet in question. But the question lands on exactly the right
seam, and §25.3 below records a defect it makes visible that no earlier section had.

### §25.1 What alphex is to aminx today

`alphex` is the ecosystem's single-declaration alphabet library
(`/home/marielle/projects/alphex`, PyPI). aminx already depends on it — but **dev group only**,
`pyproject.toml:306-312`, under decision D4:

> `# Phase 0 alphabet conformance (decision D4). DEV GROUP ONLY -- nothing under src/`
> `# imports this, so no runtime dependency edge is created.`

Its entire use is `tests/test_alphabet_conformance.py`, which *pins* aminx's alphabet constants
against alphex's declarations rather than *sourcing* them from it. That file's own docstring is
explicit about the status:

> *"Phase 0 (decision D4): **dev-dependency-only**. Nothing under `src/` imports the library."*

It asserts three things, all at width 21: `MPNN_ALPHABET`/`AF_ALPHABET` match
`known.MPNN_X_21`/`known.AF_X_21`, the two orderings differ at 17 of 20 positions, and
`aminx.potts.model.POTTS_ALPHABET` matches the MPNN declaration with `X` at 20.

**So the existing Potts alphabet IS covered by alphex — at 21 tokens.** Nothing above index 20
is.

This spec, all 1327 lines of it before this section, **never mentions alphex**. That is the
honest state: the ProtonPotts alphabet work was designed without reference to the library that
exists for exactly this problem.

### §25.2 Why it cannot express the v6 alphabet as it stands

`Alphabet.__post_init__` rejects multi-character symbols outright
(`alphex/src/alphex/alphabet.py:89-91`):

```python
if any(len(c) != 1 for c in self.symbols):
    msg = f"symbols must be single characters, got {self.symbols!r}"
    raise AlphabetDeclarationError(msg)
```

Every protonation token is multi-character — `HIS-P`, `ASP-D`, `GLU-A`. And this is not an
oversight to be patched casually: alphex's own contract spec
(`alphex/.praxia/docs/specs/260814_alphabet-contract.md:479`) lists multi-character tokens as a
surveyed case with the resolution **"declare unrepresentable"**, and the API surface spec
(`260814_alphabet-api-surface.md:562`) gives them a dedicated error, `MultiCharTokenError`.
Refusing them is a *decided* behaviour, not a gap.

The two escape hatches both fail on meaning rather than on mechanism:

| route | why it does not work |
|---|---|
| declare the 9 tokens as `specials` | `SpecialKind` has 8 members (`UNKNOWN`, `GAP`, `STOP`, `MASK`, `BOS`, `EOS`, `PAD`, `CHAIN_BREAK`) and none is a residue variant. A protonation state is not a sentinel. |
| declare indices 21–29 as `unclaimed` | `unclaimed` means *this index carries no meaning*. alphex's own declaration error calls an unaccounted index "how a lookup table silently clamps". Declaring nine meaningful residue variants as meaningless is worse than not declaring them. |
| assign single-char surrogates | Possible, but invents a symbol vocabulary that exists nowhere upstream, so the declaration would no longer be checkable against the source it is meant to pin. |

So using alphex here is **a change to alphex**, not a change to aminx. That is a real option —
alphex is ours — but it is a scoped piece of work on another repo, not a wiring task.

### §25.3 THE DEFECT THIS QUESTION EXPOSES: v4 and v6 collide above index 20

Re-reading `token_encodings.py` to answer the question surfaced something §22.2 recorded only
as a risk. There are **two** extended vocabularies upstream, not one, and they are both live:

- **v3/v4 → 32 tokens** = 21 standard + `PROTONATION_TOKENS` (11), `token_encodings.py:124-139`.
  This is the **default** (`potts_inference.py:94-121`, `extended_vocab: str = "v4"`).
- **v6 → 30 tokens** = 21 standard + `POTTS_MPNN_V6_PROTONATION_TOKENS` (9), `:138-146`.

They share the same 21-token prefix, so `S` is bit-identical when no protonation labels are
present — which is what makes the mismatch silent. Above index 20 they disagree **completely**:

| index | v4 (32-token) | v6 (30-token) |
|---|---|---|
| 21 | `HID` | `HIS-P` |
| 22 | `HIE` | `HIS-S` |
| 23 | `HIS-P` | `HIS-A` |
| 24 | `HIS-D` | `ASP-P` |
| 25 | `HIS-A` | **`ASP-D`** |
| 26 | `ASP-P` | `ASP-A` |
| 27 | **`ASP-D`** | **`GLU-P`** |
| 28 | `ASP-A` | `GLU-D` |
| 29 | `GLU-P` | `GLU-A` |
| 30 | `GLU-D` | — |
| 31 | `GLU-A` | — |

Not one index above 20 agrees. The v6 column is **measured**, not read — §23's spike injected
`HIS-S`/`ASP-D`/`GLU-P` and observed `S` taking 22/25/27, which is exactly what this table
predicts. The v4 column is arithmetic over the tuple at `:124-139` and is **unverified by
execution** (no labelled v4 cell has been run).

Read index 27 across the row: **v4 says `ASP-D`, v6 says `GLU-P`.** Feed a v6-produced `S` to a
v4-expecting consumer and an aspartate silently becomes a protonated glutamate. Different
residue, different charge, different chemistry — and shape-valid, dtype-valid, in range, and
silent. It is the same failure mode as alphex's founding example ("build a lookup table from
one and label it with the other, and 17 of 20 residues are silently permuted... nothing
raises"), reproduced one level up in the index space.

Worth noting since it will confuse the next reader: the comment above `PROTONATION_TOKENS`
says *"8 protonation-state variants"* while the tuple holds **11** and the comment four lines
below correctly says 11. Upstream doc slip; the tuple is the truth.

### §25.4 What this means for §11c

§11c is open: **public alphabet of 30 tokens, or 21 + a protonation side channel?** §25 does
not decide it, but it supplies an argument neither option had:

- **21 + side channel** keeps aminx's public alphabet inside what alphex can already declare
  and what `test_alphabet_conformance.py` already pins, and moves protonation into a channel
  where a width/meaning mismatch is a *type* error rather than an *index* error. The v4/v6
  collision above becomes unexpressible rather than merely detectable.
- **30 tokens** is closer to upstream and avoids a translation seam, but leaves the widest,
  most collision-prone part of the index space outside any declaration — unless alphex grows
  multi-character support first, which is a decided-against behaviour in its own contract.

I am not treating this as settling §11c. It is one input, and the user's call.

### §25.5 What is actionable now, regardless of §11c

Independent of the alphabet decision, the v4/v6 collision deserves a guard, because the default
is `v4` and every cell run so far has had to pass `extended_vocab="v6"` explicitly:

1. **Assert the vocabulary, don't inherit it.** Any aminx-side ProtonPotts entry point should
   require the vocabulary name explicitly and fail on absence, rather than defaulting. The
   upstream default being `v4` while the checkpoint in hand is v6 is a trap the current spikes
   avoid only by always passing the flag.
2. **Pin the index tables.** Whatever §11c decides, the table in §25.3 should become a test
   asserting both orderings by index, so an upstream reorder is loud. This needs no alphex.
3. **If alphex is to own this**, the work is in alphex: either multi-character token support
   (contradicting `260814_alphabet-contract.md:479`) or a distinct residue-variant concept
   layered above `Alphabet`. Either is a spec'd change to another repo and should be filed
   there, not assumed here.

Item 2 is unblocked and cheap. Items 1 and 3 wait on §11c.
