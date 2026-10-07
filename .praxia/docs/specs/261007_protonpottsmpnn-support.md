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
predicts. The v4 column was arithmetic over the tuple at `:124-139` when first written.

**UPDATE, same day — the v4 row is no longer unverified.** `tests/protonpotts/`
`test_vocab_index_tables.py` now pins both extension tuples as literals and asserts them
against upstream, and it was run in the `aminx-oracles-protonpotts` env on titanix where
upstream *is* importable: **8 passed, 0 skipped** — so the three conformance assertions
genuinely executed rather than skipping. That also confirmed `len(token_order) == 21` and both
full widths (32 / 30) by execution.

Be precise about what that buys. The v4 row is now verified at the **declaration** level — the
tuple's content and order, and the 21-token prefix it sits behind. It is still unverified at
the **inference** level: no labelled v4 cell has been run, so "a v4 checkpoint actually emits
`ASP-D` at index 27 in a forward pass" rests on the shared `token_order + <extension>`
construction rather than on an observation. For v6 that step *was* observed (§23). The
distinction matters only if upstream ever gives the two vocabularies different assembly paths.

The instrument is control-verified: mutating a single pinned token (`ASP-D` → `ASP-X` at index
27) fails 3 of the 8 tests, including the upstream conformance one. A pinning test that cannot
fail is not a pin.

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

1. ~~**Pin the index tables.**~~ **DONE** — `tests/protonpotts/test_vocab_index_tables.py`, 8 passed on titanix, negative control fires.
2. **Assert the vocabulary, don't inherit it.** Any aminx-side ProtonPotts entry point should
   require the vocabulary name explicitly and fail on absence, rather than defaulting. The
   upstream default being `v4` while the checkpoint in hand is v6 is a trap the current spikes
   avoid only by always passing the flag.
3. **If alphex is to own this**, the work is in alphex: either multi-character token support
   (contradicting `260814_alphabet-contract.md:479`) or a distinct residue-variant concept
   layered above `Alphabet`. Either is a spec'd change to another repo and should be filed
   there, not assumed here.

Item 1 is done — it needed no alphex, no checkpoint and no decision, which is why it went
first. Items 2 and 3 wait on §11c.

---

## §26 — the §23 labelled cell covers 3 of 9 tokens, and that is a vacuity hole

§23 established that a labelled cell is constructible with no HBPLUS, and §23.3 ruled that a
`protonpotts_*` wave must compare a sequence-dependent observable on a cell whose `S` contains
tokens ≥ 21. Both still hold. But §23's labeller assigned a **fixed label per residue type**
(every `HIS` → `HIS-S`, every `ASP` → `ASP-D`, every `GLU` → `GLU-P`), so the cell reached
exactly **3 of the 9** protonation indices.

That is not sufficient, and the gap is the interesting kind. A wave built on that cell never
exercises **two different protonation states of the same residue type** — which is precisely
the axis the v6 vocabulary exists to represent. Such a wave could pass in full while the
P/S/A (and P/D/A) discrimination is completely broken, because nothing ever asks the model to
tell those apart. §23.3's rule is necessary but not sufficient; §26.4 tightens it.

### §26.1 Cycling by `res_id % 3` is coverage-fragile — measured

First attempt: cycle the label deterministically on `res_id % 3`. On `5o45_cropped.pdb`:

```
protonation indices present: [21, 24, 25, 26, 27, 28, 29]
covered 7/9   missing=[22, 23]        (HIS-S, HIS-A)
log_probs max|delta| vs unlabelled = 52.8716
```

Both missing tokens are HIS. The cause is that coverage under this scheme depends on where
residues happen to sit in the numbering — every HIS in this structure landed on the same value
mod 3, so one HIS token appeared and the other two could not. A labelling scheme whose coverage
is an accident of the author's residue numbering is not a scheme you want behind a parity gate.

### §26.2 Cycling by ordinal-within-type fixes the scheme but not the cell

Second attempt: cycle on the **ordinal of each titratable residue within its own type**, so
coverage depends only on *how many* residues of that type exist. Same structure:

```
titratable residue counts: {'HIS': 2, 'ASP': 8, 'GLU': 7}
covered 8/9   missing=[23]            (HIS-A)
log_probs max|delta| vs unlabelled = 38.3736
```

Now the only missing token is explained by the cell rather than by the scheme: **this structure
has 2 histidines**, so at most 2 of the 3 HIS states can appear. That is a hard property of the
input, and the per-type counts the labeller now prints make it diagnosable before the run
rather than inferable afterwards.

### §26.3 A cell that covers all nine — measured

`benchmarks/data/PKAD/pdb_cache/1BVC.pdb` (HIS 12 / ASP 7 / GLU 14):

```
protonation indices present: [21, 22, 23, 24, 25, 26, 27, 28, 29]
covered 9/9   missing=[]
log_probs max|delta| vs unlabelled = 37.1461
PASS: all nine v6 protonation tokens exercised in ONE cell, non-vacuously
```

So **one cell can exercise the entire v6 extension region**, and it remains non-vacuous by
§23.3 (`log_probs` moves by 37.15 against the unlabelled cell, while `etab_out` would not move
at all — §23.2).

The resulting cell-selection criterion is cheap and checkable without running anything:

> A `protonpotts_*` cell intended to cover the v6 extension region needs **≥ 3 HIS, ≥ 3 ASP and
> ≥ 3 GLU**. Count them from the structure before the run; do not discover a shortfall from a
> coverage report afterwards.

A scan of the structures vendored with upstream found 17 satisfying it, including several in
the PKAD cache.

### §26.4 Tightening §23.3

§23.3 said *tokens ≥ 21*. Superseded by:

> A `protonpotts_*` wave must compare at least one **sequence-dependent** observable
> (`log_probs`, or an energy evaluated at `S`) on a cell whose `S` covers **all nine** v6
> protonation indices (21–29), with the per-type counts recorded in the pre-registration. A
> wave comparing only `etab_out`/`E_idx` is PottsMPNN structure parity; a wave whose `S` covers
> only some protonation indices is partial, and must say which it covered.

### §26.5 THESE LABELS ARE NOT CHEMISTRY

Stated here because it is the obvious way for this to be misread later. The cycling labeller
assigns protonation states **by position in an enumeration**, not by any physical reasoning.
Its output is wrong as chemistry and is not a prediction of anything.

That is fine for its purpose and only for its purpose: a parity wave compares aminx against
upstream **on identical inputs**, so the labels need only be *valid tokens that exercise the
vocabulary*, not correct ones. Any pre-registration using this labeller must say so in those
words, and no number produced from such a cell may be presented as a protonation-state result.

### §26.6 An opening for §20

§20 asked where P4's dump labels come from and recorded three routes, one of which was that
the PKAD cache does not help. That still holds for the *parity* dumps — they need no chemistry.
But the scan above is a reminder that upstream vendors `benchmarks/data/PKAD/`, a pKa benchmark
carrying **experimental** values. If a later wave wants labels that are chemically grounded
rather than merely valid, that is where they would come from, and it needs no HBPLUS either.
Not required for P4; recorded so §20's route list is not read as exhaustive.

---

## §27 — the upstream v6 inference path is NONDETERMINISTIC, and that invalidates §23.2

The P4 dumper ran, covered 9/9, and both controls behaved. Then the same command on the same
cell produced a different number, so I checked instead of moving on. **Every output of the v6
inference path varies run to run on bit-identical inputs.** This is the most consequential
finding in this document so far, because several earlier sections rest on single draws from
what turns out to be a distribution.

### §27.1 The measurement

Same cell, same checkpoint, two `run_forward` calls in one process, pipeline cache cleared
between them, no protonation labels involved at all (so nothing here is about labelling):

```
X           identical=True
S           identical=True
etab_out    identical=False  max|delta|=1.56884
E_idx       identical=False
log_probs   identical=False  max|delta|=45.7627
S_sampled   identical=False
S_argmax    identical=False
```

**The inputs are bit-identical and every output differs.** Corroborating, from four runs of
the P4 dumper on the identical cell, the "positive control" `log_probs` delta came out
`37.1461 / 41.9841 / 48.0462 / 45.9002`, and two successful runs wrote `oracle_f32.npz` files
with different sha256 (`01584ded…` vs `7caa4d61…`).

### §27.2 What this invalidates

**§23.2's central claim — "`etab_out` is invariant to `S`, measured bit-exact, max|delta| = 0"
— must not be cited.** It is not disproven, but the instrument that produced it is unstable:
the same quantity now differs by 1.57 on inputs that do not differ at all. A single
observation of exact equality from a nondeterministic process is a coincidence until repeated,
and it was not repeated. Everything §23.2 and §22.1 concluded about which observables can
validate protonation handling is **suspended**, not settled.

**§26's log_probs deltas are draws, not measurements.** The numbers in §26.1–§26.3 (52.8716,
38.3736, 37.1461) were each one sample. They are still evidence that labels *reach* the model —
a nonzero delta is nonzero — but no comparison *between* those numbers means anything, and
none of them should be quoted as "the" effect size.

**The P4 dump artifacts are unusable and must not be sealed.** An oracle whose sha256 changes
per run cannot anchor a parity wave at any tolerance tighter than the run-to-run spread, and
that spread is currently unmeasured.

§24, §25 and the §26 *coverage* results are untouched: checkpoint geometry, the alphabet
tables, and which token indices appear in `S` are all deterministic (`S` is identical above).

### §27.3 Attribution — partial, and two obvious explanations ruled out by reading

`model/pottsmpnn.py:500` uses `torch.randperm` and `:544` `torch.multinomial`. That is the
Gibbs/sampling path, and it straightforwardly explains `S_sampled`, `S_argmax` and `log_probs`.

**It does not explain `etab_out` and `E_idx`**, which should be pure functions of the
structure. That is the open question, and it is the more serious half: if the energy table
itself is not reproducible, the Potts port's own notion of structure parity is in question for
this model.

Two candidate causes checked and **excluded**:

| candidate | why it is not the cause |
|---|---|
| dropout active at inference (cf. aminx debt #2051) | `load_model` ends `return model.eval().to(device)` (`potts_inference.py:82`) — eval mode *is* set |
| coordinate augmentation | `structure_noise` defaults to `0.0` (`potts_inference.py:36`) and the only `randn_like` sites multiply by it (`graph_embeddings.py:689, 2387, 2390`), so they contribute exactly zero |

Both were plausible and both are wrong, which is why they are recorded — the next reader
should not spend the same time on them.

### §27.4 The control earned its place

The negative control pre-registered in `dump_protonpotts_oracles.bth.toml` — *`etab_out` must
be bit-identical between the two cells* — **fired on one of four runs**. That is the control
doing precisely the job it was added for: distinguishing "the labels changed the output" from
"the two runs differ for an unrelated reason". Without it, the dumper would have written four
mutually inconsistent oracles, all reporting a healthy positive control, and the defect would
have surfaced much later as an unreproducible parity failure.

Worth stating plainly because it cuts against the temptation to simplify: the positive control
alone passed on all four runs. A one-sided instrument would have seen nothing wrong.

### §27.5 What has to happen before P4 can produce an oracle

1. **Find the source of `etab_out`/`E_idx` nondeterminism.** Not the sampler — those two are
   upstream of it. Candidates not yet checked: tie-breaking in the k-NN that builds `E_idx`,
   a non-deterministic reduction/scatter kernel, or TF32/cuDNN autotuning if this ran on GPU.
2. **Then make the dump reproducible**, by seeding (`torch.manual_seed` plus
   `torch.use_deterministic_algorithms(True)`) or by eliminating the cause, and *prove* it by
   dumping twice and comparing sha256 — not by asserting a seed was set.
3. **Re-establish §23.2 under that determinism.** Whether `etab_out` depends on `S` is a real
   and load-bearing question, and it currently has no trustworthy answer.
4. Only then seal an oracle.

Until step 2 passes, the dumper's self-checks are the right behaviour but its output must not
be used, and `.praxia/spikes` / `oracles*/` artifacts from today are throwaway.

---

## §28 — with the sampler seeded, the labels have no measurable effect: the "positive control" was noise

§27 said the v6 path is nondeterministic and left `etab_out`/`E_idx` unexplained. Chasing that
produced two corrections, one of them to §27 itself, and then a third result that undercuts
the foundation §23 and §26 were built on. Taking them in order.

### §28.1 CORRECTION to §27 — `etab_out`/`E_idx` nondeterminism was overstated

§27 reported `etab_out identical=False max|delta|=1.56884` and `E_idx identical=False` from a
single pair of runs, and generalised that to "every output varies". That generalisation does
not hold. Measured over **32 pairs** (8 pairs × {default 10 threads, 1 thread} × {seeded,
unseeded}):

```
UNSEEDED threads=10  etab differs 0/8  E_idx differs 0/8
UNSEEDED threads= 1  etab differs 0/8  E_idx differs 0/8
seeded   threads=10  etab differs 0/8  E_idx differs 0/8
seeded   threads= 1  etab differs 0/8  E_idx differs 0/8
```

Zero disagreements in 32 pairs, seeded *or not*. A separate bisect run also showed
`etab_out=SAME  E_idx=SAME` on its baseline. So §27's observation stands as **one anomaly in
roughly 35 pairs, unexplained and not reproduced** — not as an established property. The
thread-count hypothesis (a multi-threaded reduction flipping a k-NN near-tie) is **not
supported**: 1 thread and 10 threads behave identically.

What *is* established: the run is on **CPU** (`cuda available = False`, TF32 off,
`cudnn.benchmark` off), which independently kills the GPU-kernel and TF32 candidates §27
listed as unchecked.

### §28.2 The real nondeterminism is the sampler, and seeding fixes it

```
baseline       etab_out=SAME  E_idx=SAME  log_probs=diff(39.64)
seeded         etab_out=SAME  E_idx=SAME  log_probs=SAME
deterministic  etab_out=SAME  E_idx=SAME  log_probs=SAME
cpu            etab_out=SAME  E_idx=SAME  log_probs=SAME
```

`torch.manual_seed` alone makes `log_probs` reproducible. That narrows the cause to **RNG
consumption by the Gibbs sampler** (`pottsmpnn.py:500` `randperm`, `:544` `multinomial`), not
to kernel nondeterminism — `torch.use_deterministic_algorithms(True)` adds nothing beyond the
seed. The dumper now seeds identically before each cell.

### §28.3 AND THEN THE RESULT THAT MATTERS: the label effect vanishes

With the seed in place, the dumper's positive control — `log_probs` must differ between the
unlabelled and labelled cells — collapses:

| run | `log_probs` max&#124;delta&#124;, labelled vs unlabelled |
|---|---|
| unseeded ×4 | 37.1461 / 41.9841 / 48.0462 / 45.9002 |
| **seeded ×3** | **0.000164986 / 0.000161171 / exactly 0** |

Three to five orders of magnitude smaller, and on the third seeded run **exactly zero**, which
tripped the positive control and correctly refused to write.

**So the 37–52 deltas reported in §23.2, §26.1, §26.2 and §26.3 were sampler noise, not an
effect of the protonation labels.** Every one of those numbers was a difference between two
independent draws from a stochastic decoder, and the labels contributed nothing detectable to
them. §23.2's claim that `max|delta| = 48.2565` on `log_probs` is "a positive control for the
instrument, available before any parity run" is **withdrawn**: it was a measurement of the
sampler's variance.

The residual 1.6e-4 is at f32 round-off for quantities of this magnitude and must not be read
as a small real effect until something demonstrates it is one.

### §28.4 What this leaves standing, and what it blocks

Still good: §24 (checkpoint geometry), §25 (the alphabet tables and the v4/v6 collision),
§26's **coverage** results. `S` is deterministic, and that the labelled cell's `S` reaches all
nine indices 21–29 is unaffected — the labels demonstrably reach **featurization**.

Blocked: **we currently have no observable that demonstrably responds to protonation labels.**
`etab_out`/`E_idx` do not (structure-only). `log_probs` does not, once the sampler is
controlled. `S` does, but `S` is the *input* — comparing it validates the featurizer, not the
model. A `protonpotts_*` parity wave cannot be built until one exists, because otherwise the
wave validates nothing about protonation however many tokens its cell carries.

This is the same vacuity trap §23.3 and §26.4 were written to prevent, one level deeper: both
rules assumed `log_probs` was a working protonation-sensitive observable. It is not.

### §28.5 The most likely resolution, not yet tested

§23.3's own wording offered an alternative I never implemented: *"`log_probs`, **or an energy
evaluated at `S`**"*. That clause is probably the answer. A Potts model's protonation
sensitivity should live in

```
E(S) = sum over edges (i,j) of etab_out[i, j, S_i, S_j]   (plus the self/field terms)
```

which is **not** any tensor the engine returns — it must be computed from `etab_out`, `E_idx`
and `S`. It is sequence-dependent by construction, it is deterministic (both inputs are), and
it is exactly where a 30-token vocabulary would express a protonation preference.

Next step for P4: compute `E(S)` for the labelled and unlabelled cells and check it moves.
If it does, that is the wave's observable and the dumper should capture it. If it does **not**,
then this checkpoint's protonation tokens have no effect on any model output at inference, and
that is a much larger finding about the upstream model than about our port.

Do not seal any oracle, and do not write a `protonpotts_*` pre-registration naming `log_probs`
as its protonation observable, until that question is answered.

---

## §29 — the extension region DOES carry signal, but almost none of it is P-vs-D

§28 left P4 blocked with no observable responding to protonation labels. The untested
candidate was §23.3's "an energy evaluated at `S`". It works, the blocker lifts, and the
measurement immediately constrains how the wave's tolerance may be set.

### §29.1 Two tests, because the obvious one is nearly vacuous

**A — `H(S)`.** Upstream has its own scorer, `PottsMPNN.calc_potts_eners`
(`model/pottsmpnn.py:300-314`), `H(s) = Σ_i Σ_k etab[i,k,s_i,s_{E_idx[i,k]}]`. Used rather
than reimplemented.

Note what A alone does *not* prove. Evaluated on **one** `etab` with two different `S`, it will
differ whenever `S` differs — indexing a table at different indices gives different numbers by
arithmetic, regardless of whether the model learned anything. So A is reported for magnitude,
not as evidence.

**B — is the extension region of the table non-degenerate?** The single-site field is
`etab[:, 0].diagonal()` → `[L, V]` (convention confirmed at `pottsmpnn.py:401`). At a
titratable position, compare the field entries for the parent token and its three protonation
tokens. This needs **no labels at all**, is fully deterministic, and is the decisive one.

### §29.2 Measured (1BVC, v6 checkpoint, seeded)

```
etab_out (1, 153, 48, 30, 30)   single-site field (153, 30)
field abs-max over the whole table = 12.4132

HIS @pos11: parent=-4.39852  HIS-P=-4.43930  HIS-S=-4.31135  HIS-A=-4.48335
  spread among the 3 states = 0.171997   max spread over all 12 HIS positions = 0.278798
ASP @pos19: parent=-4.70759  ASP-P=-4.50254  ASP-D=-4.71181  ASP-A=-10.94776
  spread among the 3 states = 6.44522    max spread over all  7 ASP positions = 6.75074
GLU @pos3 : parent=-4.40029  GLU-P=-4.42511  GLU-D=-4.38911  GLU-A=-8.88752
  spread among the 3 states = 4.49841    max spread over all 14 GLU positions = 5.06311

H(S) plain=-34724.792969  relabelled=-34448.917969  delta=+275.875  (S changed at 33 positions)
```

**The extension region carries signal.** The 30-token vocabulary is not inert, `H(S)` is a
valid sequence-dependent and deterministic observable, and P4 is unblocked.

### §29.3 THE PART THAT CONSTRAINS THE WAVE: the signal is an ambiguity penalty

Read the three numbers in each row rather than their spread. For ASP and GLU, the large spread
comes almost entirely from the **`-A` (ambiguous) state being heavily penalised** — `ASP-A` at
−10.95 against `ASP-P` −4.50 and `ASP-D` −4.71; `GLU-A` at −8.89 against −4.43 and −4.39. The
chemically interesting contrast is much smaller:

| contrast | magnitude |
|---|---|
| `ASP-A` vs its P/D siblings | ≈ 6.3 |
| `GLU-A` vs its P/D siblings | ≈ 4.5 |
| **`ASP-P` vs `ASP-D`** | **0.209** |
| **`GLU-P` vs `GLU-D`** | **0.036** |
| `HIS-P`/`HIS-S`/`HIS-A` spread | 0.172 |

So the model mostly encodes *"I have no evidence here"* versus *"I have evidence"*. The
protonated-versus-deprotonated discrimination — the thing a protonation-aware model exists
for — moves the field by **0.04 to 0.21**, around 150× smaller than the ambiguity penalty and
roughly 350× smaller than the table's abs-max of 12.41.

**Consequence for P4, and it is the main point of this section:** a tolerance band derived
from the overall spread (6.75) or from the table's scale (12.41) would be one to two orders of
magnitude too loose to notice a completely broken P/D discrimination. A wave that passes under
such a band validates the ambiguity penalty and nothing else. **The band must be set against
the 0.036 `GLU-P`/`GLU-D` scale**, which is the smallest contrast the vocabulary is supposed to
express, and the pre-registration must say so.

HIS is the other way round: its three states sit within 0.17 of each other with no large
ambiguity penalty, so HIS carries the least separable signal of the three despite the
checkpoint being named `…his0.3_acid0.06`. Whether that is expected is a question for the
upstream author, not something to infer here.

### §29.4 What P4 captures now

The dumper's protonation observable changes from `log_probs` to:

- **`H(S)`** on the labelled and unlabelled cells, via upstream's `calc_potts_eners`.
- **the single-site field restricted to columns 21–29**, which is the targeted quantity and
  the one whose per-contrast scale §29.3 just pinned.

`etab_out`/`E_idx` stay in the dump as PottsMPNN structure parity. `log_probs` stays too, but
**demoted to a recorded quantity rather than a control** — §28 showed it does not respond to
labels once seeded.

This supersedes the original pre-registration's hypothesis 2, whose positive control named
`log_probs`. That hypothesis was **falsified**, not relaxed; the amended sidecar records the
falsification and the replacement rather than quietly editing the criterion.

---

## §30 — the observables are now deterministic; the instability is in the FIRST forward of a process

Three runs of the amended dumper, same cell, same seed:

```
control A: H(S) plain=-34724.792969 labelled=-33881.527344 delta=+843.265625   [all 3 runs]
control B: ASP_P_vs_D=0.2092742919921875  GLU_P_vs_D=0.12173986434936523
           HIS_P_vs_S=0.17555570602416992                                      [all 3 runs]
negative : etab_out differs, max|delta| = 4.76804 / 0.941982 / 3.84095         [all 3 differ]
```

**P4's observables are deterministic.** `H(S)` and the field contrasts are bit-identical to
the last digit across all three runs, so §28's seeding fix did its job. Control B also now
agrees with §29.3 measured independently: `ASP_P_vs_D` 0.2093 against 0.209, `HIS_P_vs_S`
0.1756 against 0.172. (`GLU_P_vs_D` reads 0.122 here against 0.036 there because this is the
max over all 14 GLU positions and §29.3 quoted one position — different statistics of the same
quantity, not a disagreement.)

### §30.1 A bug in control B, caught before it was relied on

The first version of `_check_field_non_degenerate` took the max over **all L positions**,
reporting the contrast at positions where the token is meaningless — an ALA position has a
`GLU-P` field entry and it signifies nothing. That returned `0.354 / 0.403 / 0.412`. Restricted
to positions actually holding the parent residue it returns `0.209 / 0.122 / 0.176`.

Worth naming because of what it would have cost: the inflated GLU figure is **~3.3× too large**,
and this function's entire job is to supply the tolerance scale. A band set from `0.403` would
have been loose enough to pass a wave with a broken GLU P/D discrimination — the precise
failure §29.3 was written to prevent, reintroduced by the check meant to enforce it.

### §30.2 The remaining instability is narrow and now locatable

The negative control fires on **every** run, but with a **different** magnitude each time
(4.77 / 0.94 / 3.84) while every other number is bit-stable. That pattern locates it:

- `H(S)` and the field both derive from the **labelled** cell's `etab_out`, and both are
  bit-stable across runs → **the labelled cell is deterministic**.
- The only unstable quantity is the *difference* against the **unlabelled** cell's `etab_out`
  → **the unlabelled cell's `etab_out` is what varies**.
- The unlabelled cell is the one built **first** in each process.

So the hypothesis is **first-forward instability** — lazy initialisation, a one-time allocator
or kernel-selection effect that perturbs the first forward pass in a process and not later
ones. This also retro-explains §27 (whose spike compared call 1 against call 2 and saw a
difference) and §28's 32 clean pairs (which seeded before *each* call, but still compared
call-1-of-pair against call-2-of-pair... so that part does **not** fit, and I am flagging the
inconsistency rather than smoothing it over).

### §30.3 §23.2 is now more likely wrong than merely unproven

`etab_out` differs between the labelled and unlabelled cells on **every** run. §23.2 claimed
it invariant to `S`. The varying magnitude means the two candidate explanations cannot yet be
separated:

1. `etab_out` genuinely depends on `S`, and the varying delta is first-forward noise on top.
2. `etab_out` is `S`-invariant and the entire delta is first-forward noise.

Under (1) the dumper's negative control is simply the wrong check and should be deleted;
under (2) it is right and the warm-up must be fixed. **Do not delete it to make the dumper
pass** — that is the failure mode the control exists to prevent, and the distinction is one
experiment away.

### §30.4 The next experiment, which separates them

Add a throwaway warm-up forward before both cells, and independently swap the cell order
(labelled first, unlabelled second):

- delta → **0** ⇒ `etab_out` is `S`-invariant, §23.2 was right, all of §27/§30's instability
  was warm-up, and the negative control stays.
- delta → **stable and nonzero** ⇒ `etab_out` depends on `S`, §23.2 is wrong, and the
  negative control must be replaced by an equality-to-a-pinned-value check.
- delta → **still varying** ⇒ the warm-up hypothesis is wrong too, and the cause is elsewhere.

Each outcome is distinguishable and each dictates a different, specific change. Until it runs,
no oracle is sealed and `etab_out`'s dependence on `S` has no trustworthy answer.

---

## §31 — §23.2 is RESTORED: `etab_out` is `S`-invariant; the anomaly is per-process and ~1/3

§30.4's experiment ran, and all four arms came back zero:

```
A  no warm-up   plain#1 vs plain#2    delta=0
B  warmed       plain#1 vs plain#2    delta=0
C  warmed       plain   vs labelled   delta=0
C' warmed       plain   vs labelled   delta=0
plain reproducible across C and C' : 0     labelled reproducible : 0
```

Arm A is the informative one: **even with no warm-up, the first forward is stable.** So §30.2's
warm-up hypothesis is **wrong**, and C/C' answer §23.2 directly.

**§23.2 is restored.** `etab_out` is invariant to `S`. §27 suspended it and §30.3 called it
"more likely wrong than merely unproven" — both of those were reacting to the dumper's firing
control, and the dumper turns out not to be measuring what the model does.

### §31.1 The anomaly is per-process, and the control is a correct filter

Within one process the comparison is zero every time, across many arms. Across **fresh
processes** it is not. Six runs of the dumper, default environment:

```
run1 bit-identical   run2 max|delta|=1.56884   run3 bit-identical
run4 bit-identical   run5 bit-identical        run6 max|delta|=4.76804
```

Two of six fire. And critically — **the four clean runs produced exactly ONE distinct
sha256.** So when the control passes, the oracle is bit-reproducible.

That is a usable state, and it is worth saying plainly rather than treating the anomaly as a
total blocker: **the negative control is behaving as a correct filter.** Bad runs are rejected
before anything is written; clean runs agree with each other exactly. P4 can proceed on a
retry-until-clean basis while the anomaly is tracked separately.

### §31.2 What the anomaly is not

- **Not warm-up** — arm A above.
- **Not the seed** — seeding is already in place and the derived observables are bit-stable
  across all runs, firing or not (`H(S)` and the field contrasts are identical to the last
  digit in every run of §30 and §31).
- **Not `PYTHONHASHSEED`.** Tested directly: with `PYTHONHASHSEED=0` fixed, 5 of 6 runs fired
  (`1.56884, 1.56884, bit-identical, 0.543993, 4.76804, 4.76804`). Pinning the hash seed does
  not remove it. This was a good candidate — a dict/set iteration order varying per process
  would explain "constant within a process, varying across them" — and it is **excluded**.
- **Not continuous float noise.** The deltas recur exactly: `1.56884` twice, `4.76804` twice,
  across different runs. Something discrete is being selected, not perturbed.

### §31.3 The actual open question, which is narrower than it was

The **spike never fires and the dumper fires ~1/3 of the time**, and both do plain-then-
labelled in a fresh process against the same cell. So the difference is in the *dumper*, not
in the model — which is a much smaller search space than "upstream is nondeterministic".

The leading candidate is the labeller object. The dumper's `CyclingProtonationLabels` declares
a `counts` attribute and defines `__init__` calling `super().__init__()`; the spike's
equivalent defines neither. If `Transform.__init__` carries state, the two are not the same
transform even though their `forward` is identical.

Next step is a direct diff of the two code paths rather than more sampling: make the spike
progressively more like the dumper until it starts firing. Each step is one edit and one run,
and the first step that flips the behaviour names the cause.

### §31.4 Status of the controls

Keep the negative control. §30.3 set out the rule that it must not be deleted to make the
dumper pass, and the experiment has now come down on the side where **it is the right check** —
`etab_out` really is `S`-invariant, so a nonzero delta really does mean the two cells differ by
more than their labels. Deleting it would have silently admitted the ~1/3 of runs that are
wrong for an unknown reason.

---

## §32 — found it: multi-threaded reduction flips k-NN ties, and §31's conclusion was my own confound

### §32.1 CORRECTION to §31 — the dumper was innocent

§31 concluded "the difference is in the *dumper*, not the model", because the spike never fired
and the dumper fired ~1/3. That conclusion was wrong, and the reason is a confound in the arms
I designed in §30.4:

| | first two cells of the process? | labelling differs between them? |
|---|---|---|
| arm A | **yes** | no (plain vs plain) |
| arm C | no (warmed first) | **yes** |
| **dumper** | **yes** | **yes** |

**Neither arm tested both at once**, so "the spike never fires" was never evidence that the
dumper's configuration is safe — the spike had never run the dumper's configuration. Running
exactly it (two cells per process, plain then labelled, then exit):

```
etab_delta=0         E_idx_same=True   E_idx_diffs=0
etab_delta=0         E_idx_same=True   E_idx_diffs=0
etab_delta=0.941982  E_idx_same=False  E_idx_diffs=6
etab_delta=0.941982  E_idx_same=False  E_idx_diffs=6
etab_delta=4.76804   E_idx_same=False  E_idx_diffs=6
etab_delta=0         E_idx_same=True   E_idx_diffs=0
```

Reproduced in the spike, 3/6, with the same recurring discrete values. The dumper was never
special.

### §32.2 The mechanism: it is `E_idx`, every time

The new column is what matters. **`E_idx` differs at exactly 6 entries whenever `etab_out`
moves, and at 0 entries whenever it does not** — perfect correlation across all six runs, and
the same 6 whichever delta appears. Out of `L × K = 153 × 48 = 7344` neighbour slots.

So nothing about the *energies* is unstable. The **neighbour graph** changes, and `etab_out`
simply reports a different set of pairs. That reframes the whole investigation: §27 spent its
effort on "which outputs are nondeterministic" when the live question was "why does the k-NN
pick different neighbours".

### §32.3 It is thread count — and §28 tested that, in the wrong configuration

Twelve fresh processes in each arm, identical in every other respect:

```
1-thread     (OMP_NUM_THREADS=1 MKL_NUM_THREADS=1):   0 fired / 12
multi-thread (default, 10 torch threads):             9 fired / 12
```

**0/12 against 9/12.** Fisher exact ≈ 2e-5. This is the cause.

The reading: a multi-threaded float reduction in the pairwise-distance computation produces
last-bit differences depending on how work is partitioned across threads, which varies per
process. Where two neighbours are near-tied, that flips the `topk` ordering, six slots change,
and `etab_out` moves by O(1) because it is reporting different pairs — not because any energy
changed.

**§28 explicitly tested thread count and reported it "NOT supported".** That test ran the
*plain vs plain* configuration, which never fires at any thread count, so it could only ever
return 0/8 in both arms. Same confound as §32.1, one section earlier. A negative result from a
configuration that cannot produce a positive is not a negative result.

### §32.4 Consequences

1. **Pin threads for oracle dumps.** `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` (or
   `torch.set_num_threads(1)`) for any run whose artifact is meant to be reproducible. Slower,
   and worth it for a dump that is generated once and compared forever.
2. **This is not a ProtonPotts bug and probably not ours either** — near-tied neighbours are a
   property of the structure, and any k-NN over float distances has this exposure. But it does
   mean **`E_idx` is an unsafe parity observable near ties** for *any* MPNN-family port,
   including the Potts/LASEr work. Whether those oracles pinned threads is worth checking; I
   have not.
3. **Keep the negative control, now understood.** §30.3 forbade deleting it to make the dumper
   pass; §31 kept it on the evidence then available. It turns out to be detecting a real
   defect in how the dump is *run*, which is exactly what a control is for.
4. §23.2 stands as restored in §31 — `etab_out` is `S`-invariant. Nothing here disturbs that;
   the firing runs differ by neighbour graph, not by `S`.

### §32.5 Verified: the dump is now reproducible

`torch.set_num_threads(1)` added to the dumper, then four fresh runs:

```
bit-identical / bit-identical / bit-identical / bit-identical
sha256 (all four): bbb6828335d28e7427b18d7bdc1c39138e8716022fb54c1a875e4fcd3d6a0d6e
```

Four clean runs, **one** distinct hash. Proven by comparing artifacts, not by asserting the
fix was applied — which is what §27.5 step 2 demanded and is now satisfied. **P4's oracle is
reproducible.**

### §32.6 Method note worth carrying

Two separate wrong conclusions in this document (§28's "thread count not supported", §31's
"the anomaly is in the dumper") came from the same mistake: **testing a factor in a
configuration that cannot exhibit the effect, and reading the resulting null as informative.**
Before accepting a negative, check that the configuration reproduces the phenomenon at all —
a positive control on the *setup*, not just on the measurement.
