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

Consequences, and they are all simplifications:

- **FLAML is not on the V1 path.** It belongs to the labeller/training side. The user's
  position — "if flaml is needed for that it is covered" — is satisfied without needing it.
- **HBPLUS is not required.** It is env-gated inside `annotate_folds`, i.e. post-hoc fold
  annotation, not design.
- **RF3 (~3 GB) is optional** — it folds candidate sequences *after* design for structural
  metrics. Out of V1.
- The heavy pins that made a second oracle environment look costly (`sklearn<1.9`, FLAML,
  xgboost/lightgbm) are labeller pins. The V1 oracle env is torch + atomworks + foundry on
  Python 3.12.

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
| **P0** | **Spike.** Load the ckpt; print `train_cfg` and every tensor shape; settle 30-vs-32 from the weights; confirm or kill foundry-`ProteinMPNN` ↔ aminx encoder weight compatibility | — | answer recorded; throwaway, no sidecar |
| P1 | Vendor upstream at a pinned SHA; stand up `aminx-oracles-protonpotts/` on titanix (torch + atomworks + foundry, Py3.12) | P0 | env resolves; manifest |
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

P5's estimate is **reduced** from the first pass (2–3 d → ~1 d) on the strength of §3. P6 remains
the unbounded one and is why P0 is first.

**Top risks.**

1. **Wrong-lineage assumption** (§1). Mitigated by making P0 blocking.
2. **Encoder divergence.** If foundry's `ProteinMPNN` is not weight-compatible with aminx's, P6
   becomes a second encoder port. P0 answers this in ~2 h, before anything depends on it.
3. **The generic-`V` refactor regresses the shipped Potts port.** `N_AA = 20` is load-bearing in
   two modules of frozen, measured code. Goldens captured *before* P7 starts; existing waves
   re-run after.

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

a. **Design engine in V1?** This spec assumes yes (P9). The alternative is energy/scoring only,
   deferring the engine — smaller, but then V1 does not validate the thing the repo exists for.
b. **Public alphabet.** 30 tokens end-to-end, or a 21-token public alphabet with protonation as
   a side channel? Affects every sink and every `sequences_to_score` caller.
c. **Second oracle environment on titanix.** atomworks + foundry pin incompatibly against the
   existing `aminx-oracles/`. §4 shows the V1 env is lighter than first thought, but it is still
   a second env.
d. **Where §6's seam sits for `protonpotts_features`.** Exact on coordinates, or a floor run?
   This spec proposes a floor run because the upstream path is f32 throughout.

*(The labeller/FLAML question is no longer open: §4 shows it is off the V1 path.)*

## 12. Assumption register

**Verified by reading the source or the live repo:** the lineage and class declaration;
`potts_vocab_size` as a constructor parameter; the nine feature keys and their atomworks
imports; the `transforms/` and `feature_aggregation/` file listings; the repo root layout;
`design_ph.py`'s complete import list and the absence of flaml/xgboost/lightgbm/sklearn/hbplus;
aminx's `PottsFeatures` fields, `N_AA`/`PAIR_DIM`, and `ETAB_ALPHABET`; the KeatingLab pin;
`_SCOPED_PREFIXES`.

**Unverified, carried from a scoping subagent and flagged as leads, not facts:** the 30-token v6
listing and the 30-vs-32 conflict; the checkpoint's size and internal structure; the etab merge
rule and conditional-energy convention; the design engine's internals and file size; the paper
citation. **P0 and P2 exist to convert these.** Nothing in §7–§8 should be treated as settled
until they do.
