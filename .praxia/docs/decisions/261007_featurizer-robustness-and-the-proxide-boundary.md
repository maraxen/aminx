---
title: 'The featurizer is four forks and a filename regex: making it resolved rather than hardcoded, and where the proxide line goes'
description: What the aminx featurizer actually hardcodes (with file:line), a FeatureSpec design that derives every width from the weights instead of a literal, and the aminx/proxide split — proxide owns the mechanism, aminx owns the convention
status: proposed
task_id: 261007_protonpottsmpnn-support
date: '261007'
sprint: ''
backlog_ids: ''
---
# The featurizer is four forks and a filename regex

Prompted by the user asking, 2026-10-07: *"how can we make our featurizer more robust and
less hardcoded? what belongs in aminx and what belongs in proxide?"* The RBF pair-order
finding in `261007_protonpottsmpnn-support.md` §14 is the forcing function — that trap exists
*because* a convention is a module-level global rather than a resolved parameter.

Everything below is read from source, not reported. Line numbers are at `67d7bfcc`.

## 1. What is actually hardcoded

### 1a. `25` is a literal in three places, and it is `BACKBONE_PAIRS.shape[0]`

- `model/features.py:294` — `edge_embed_in_dim = 16 + 16 * 25  # Matches POS_EMBED_DIM + RBF_DIM * 25`
- `model/ligand_features.py:249` — `# edge_in = 16 + 16 * 25 = 416`
- `model/packer.py:385` — `enc_edge_in = num_positional_embeddings + num_rbf * 25`

The packer line is the interesting one: it *parameterizes* `num_rbf` and still hardcodes the
pair count beside it. Both numbers have the same status — they are properties of the feature
convention — and only one of them was given a name.

### 1b. The pair table is a global closed over by a jitted function

`utils/radial_basis.py:32-60` defines `BACKBONE_PAIRS` at module scope; `compute_radial_basis`
at `:65` is `@jax.jit` and closes over it. There is no argument by which a caller could supply
a different order. A second family needing a different order therefore cannot reuse the
function at all — which is exactly the situation ProtonPottsMPNN puts us in.

### 1c. Widths assigned as literals inside `__init__` next to a parameter that isn't

`model/features.py:286-288`:

```python
self.rbf_dim = 16
self.pos_embed_dim = 16  # Fixed output dim
self.num_positional_embeddings = num_positional_embeddings
```

Two of these three are constructor-invisible. `rbf_dim` in particular is `RADIAL_BASES` from
`radial_basis.py` restated as a digit, so the two can disagree without anything noticing.

### 1d. Alphabet size

- `families/potts_mpnn/potts_head.py:26-27` — `N_AA = 20`, `PAIR_DIM = N_AA * N_AA`, with `400`
  written into the docstrings and jaxtyping annotations at `:7,:40,:45,:46,:50`.
- `num_amino_acids: int = 21` / `vocab_size: int = 21` repeated as defaults in `model/mpnn.py:52-53`,
  `model/ligand_mpnn.py:121-122`, `model/diffusion_mpnn.py:94-95`.

ProtonPottsMPNN is V=30 (measured, spec §1a), so `N_AA` is the one that has to become a
parameter; the `21`s are at least already parameters with a default.

### 1e. Architecture is inferred by substring-matching a filename

`io/weights.py:69-100`, `get_topology_for_checkpoint`:

```python
if "v_32" in name: topology["k_neighbors"] = 32 ...
elif "v_30" in name: ...
if "ligandmpnn_sc" in name or "_sc_" in name: topology["model_type"] = "packer"
elif "membrane" in name: topology["physics_feature_dim"] = 3
```

This is the most load-bearing hardcoding in the tree, because it is the thing that decides what
every downstream width will be. A checkpoint renamed on the way in silently gets a different
architecture. A new family must either adopt the naming convention or add an `elif`.

And it is unnecessary for at least one real checkpoint: the ProtonPottsMPNN `.pt` **carries its
own `train_cfg` DictConfig** stating `extended_vocab='v6'` (spec §1a). The authoritative answer
is inside the file we are already opening.

### 1f. Four featurizers, one per family

| Implementation | What it is |
| --- | --- |
| `model/features.py` `ProteinFeatures` | stock MPNN path |
| `model/ligand_features.py` `ProteinFeaturesLigand` | ligand path, same `16 + 16*25` shape |
| `families/potts_mpnn/features.py` `potts_edge_features` | reuses the stock *weights*, reimplements the *graph* |
| `model/laser/graphs.py` | numpy kNN, different edge packing entirely |

The Potts one is the tell. Its module docstring (`features.py:1-23`) is a careful four-item
catalog of how the stock featurizer differs from what the family needs — masked-vs-`+inf` kNN,
`D_max` padding, Ca-Ca RBF on `D_adjust`, tie-break order. That catalog is a *configuration
table written in prose*, and the resolution chosen was to fork the module. It is good work and
it is the right call given the current shape; it is also the fourth time we have made it.

## 2. The good precedents are already in the tree

The fix is not a new idea here, it is the generalization of three things we already do.

1. **Derive the width from the weight.** `model/features.py:383` and again at
   `families/potts_mpnn/features.py:103`:
   `max_relative = (features.w_pos.weight.shape[1] - 2) // 2`. The one-hot width is *read off
   the loaded tensor*. This is exactly right and should be the rule, not the exception.
2. **Resolve dtype at call time, not import time.** `utils/radial_basis.py:20-28` `rbf_centers(dtype)`,
   with a docstring explaining the `jax_enable_x64`-after-import trap it fixes.
3. **A caller may supply the features.** `ProteinFeatures.__call__` already accepts
   `rbf_features` and `neighbor_indices` (`features.py:313-314, 445-448`) and skips graph
   construction when given them. **The proxide seam already exists, at the right place.**

## 3. The design: a resolved `FeatureSpec`

One frozen dataclass, static on the module, carrying everything that is currently a literal:

```python
@dataclass(frozen=True)
class FeatureSpec:
  pair_order: tuple[tuple[int, int], ...]   # replaces BACKBONE_PAIRS
  atom_slots: tuple[str, ...]               # ("N","CA","C","O","CB") -- names, not indices
  rbf_count: int                            # 16
  rbf_range: tuple[float, float]            # (2.0, 22.0)
  k_neighbors: int
  max_relative: int
  alphabet_size: int                        # 20 / 21 / 30
  knn_invalid: Literal["inf", "row_max"]    # the Potts docstring item 1
  ca_rbf_source: Literal["raw", "adjusted"] # item 2
  rbf_atom_mask: bool                       # the foundry innovation
  @property
  def edge_in_dim(self) -> int:
    return self.rbf_count * (len(self.pair_order) + 1)  # +1 slot for positional
```

Three rules make it robust rather than merely indirected:

- **Every width is computed from the spec, never written as a digit.** `25` becomes
  `len(spec.pair_order)`; `400` becomes `spec.alphabet_size ** 2`; `16 + 16*25` becomes
  `spec.edge_in_dim`.
- **The spec is resolved once, at load, from the checkpoint plus a registry keyed by family —
  never from a filename substring.** Where the checkpoint states a fact (`train_cfg`), the
  checkpoint wins; the registry supplies only what the file does not carry. `get_topology_for_checkpoint`
  shrinks to a legacy fallback for the eight `LEGACY_ALIAS_MAP` names that genuinely have no
  embedded config.
- **A conformance check at load asserts the derived widths equal the loaded tensors' actual
  shapes.** This is the part that converts "less hardcoded" into "robust": a wrong spec becomes
  a loud failure at load instead of silently wrong numbers on every input. It is precisely the
  failure mode §14 describes — two 400-wide tensors in different orders produce no shape error
  — and a width check alone would not catch an *ordering* difference, so the spec's
  `pair_order` must also be stamped into the converted-weight artifact and compared, not
  inferred.

The three Potts-docstring behaviours become spec fields rather than a forked module, which is
what collapses four featurizers back toward one.

## 4. Where the proxide line goes

The useful cut is **mechanism versus convention**.

| | proxide | aminx |
| --- | --- | --- |
| owns | structure → tensors | weights → predictions |
| parsing, atom37 ordering, chemistry, trajectories | yes | no |
| kNN, RBF, dihedrals, NeRF, alignment — the kernels | yes | no |
| dataset/prefetch/batching of structures | yes | no |
| *which* pair order / mask policy / alphabet a checkpoint wants | **no** | **yes** |
| learned embeddings, encoder/decoder, heads | no | yes |
| sampling, design, parity harness, oracles | no | yes |

**proxide owns the mechanism; aminx owns the convention.** The seam is: a `FeatureSpec` goes
down, a feature dict comes back. The RBF loop is a protein-geometry kernel and belongs in
proxide. The *table saying which 25 pairs and in what order* is a property of a particular
trained network and belongs in aminx.

The corollary is the actionable part: **proxide must become parameterized, not opinionated.**
Today `proxide/geometry/radial_basis.py:20` has its own `BACKBONE_PAIRS` global — proxide is
currently *asserting* a convention, which is aminx's job. Its `compute_radial_basis` should take
the pair order as an argument and ship the common tables as named constants callers may pick
from.

### 4a. The fork has already drifted, and in the direction of a real bug

`diff proxide/src/proxide/geometry/radial_basis.py aminx/src/aminx/utils/radial_basis.py` is
four hunks: the module docstring (proxide's still reads `prxteinmpnn.utils.radial_basis`), the
type imports, and **`rbf_centers(dtype)` — present in aminx, absent in proxide.** aminx's copy
carries the f64 centres fix from debt #2310; proxide's does not.

So the duplication is not hypothetical future drift; it has already happened, and the stale side
is the one a caller reaching proxide directly would get. Any f64 work routed through
`proxide.geometry` re-finds a bug aminx already fixed. Unify on one implementation and delete
the other.

### 4b. What does *not* move into aminx

Protonation-state labelling (HBPLUS + the 5-fold FLAML ensemble, spec §4 CORRECTION) is
structure annotation, not model inference. By the rule above it sits on proxide's side of the
line, or stays an upstream preprocessing step consumed through
`transforms/precomputed.py`. Pulling an sklearn/xgboost pin into aminx to label residues would
put a chemistry oracle inside the inference library. This is open user decision (a) in the
ProtonPottsMPNN spec and the boundary argument is one input to it, not the decision.

## 5. Sequencing, and what this costs

This touches `src/aminx/`, so under THE FREEZE it is scoped work that cannot land before #165
merges. It is **not** a prerequisite for the ProtonPottsMPNN port — but doing the port first
without it means writing the fifth featurizer fork, and the fifth fork is the one that makes
the consolidation twice as expensive.

Proposed order, each independently useful:

1. **Derive, don't declare** (small, mechanical, zero behaviour change): replace the three `25`
   literals with `len(BACKBONE_PAIRS)`, `rbf_dim` with `RADIAL_BASES`, `PAIR_DIM` with
   `N_AA ** 2` where it is spelled `400`. Pure refactor; the parity waves are the regression test.
2. **`FeatureSpec` + load-time conformance check**, with every existing family resolving to a
   spec that reproduces today's behaviour exactly. Waves must stay green with zero band changes.
3. **Checkpoint-stated config beats filename**, `get_topology_for_checkpoint` demoted to legacy
   fallback.
4. **Unify the two `radial_basis.py`**, parameterizing proxide's on pair order.
5. **Collapse the forks**: Potts' three documented deviations become spec fields; `potts_edge_features`
   becomes a spec, not a module.

Steps 1 and 2 are what the ProtonPottsMPNN port actually needs. 3–5 can follow it.

## 6. Assumption register

| Assumption | Status |
| --- | --- |
| `25` appears as a literal in exactly the three sites listed | verified, grep over `src/aminx/model`, `src/aminx/potts`, `src/aminx/families` |
| `BACKBONE_PAIRS` is closed over by a jitted function with no override path | verified, `utils/radial_basis.py:32,65,88` |
| topology is decided by substring match on the checkpoint id | verified, `io/weights.py:69-100` |
| proxide's `radial_basis.py` is a fork missing the dtype fix | verified by `diff`; four hunks, one is `rbf_centers` |
| the ProtonPottsMPNN checkpoint carries `train_cfg` with the vocab | verified, spec §1a P0 probe |
| a `FeatureSpec` can reproduce all four current featurizers | **unverified** — the Potts and ligand paths are read and look reducible; `laser/graphs.py` packs edges differently enough that it may stay separate. Spike before committing to step 5. |
| widths-only conformance would not have caught the §14 ordering trap | verified by construction — both tensors are 400 wide; hence the `pair_order` stamp requirement in §3 |
