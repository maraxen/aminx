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
  pos_embed_dim: int                        # 16 -- NOT the same number as rbf_count
  @property
  def edge_in_dim(self) -> int:
    return self.pos_embed_dim + self.rbf_count * len(self.pair_order)
```

(An earlier draft wrote this as `rbf_count * (len(pair_order) + 1)`, which is right only by
the coincidence that both widths are 16 on the stock path. `packer.py:385` is
`num_positional_embeddings + num_rbf * 25` with the two genuinely independent — see §8.2.)

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

1. **Derive, don't declare** (small, mechanical, zero behaviour change) — **but not the way an
   earlier draft of this line said; see §8.2 before doing it.** Each site derives from its *own*
   quantity: `features.py:294` from `len(BACKBONE_PAIRS)`; `packer.py:385` from
   `len(backbone_coords) ** 2`, because packer builds its pairs with its own 5×5 nested loop
   (`packer.py:502-505`) and never imports `BACKBONE_PAIRS`; `ligand_features.py:250`'s literal
   `416` from its own inline pair list (`:331-358`). **Do not rewrite `400` globally**: it means
   25×16 at `families/potts_mpnn/features.py:44` and 20×20 at `potts_head.py:26-27`. Pure
   refactor; the parity waves are the regression test.
2. **`FeatureSpec` + load-time conformance check**, with every existing family resolving to a
   spec that reproduces today's behaviour exactly. Waves must stay green with zero band changes.
3. **Checkpoint-stated config beats filename**, `get_topology_for_checkpoint` demoted to legacy
   fallback.
4. **Unify the RBF kernels** — targeting the **Rust** crate, not the dead Python fork. See §8.5:
   `proxide/crates/proxide-geometry/src/geometry/radial_basis.rs` is the shipping implementation
   and it carries its own `const BACKBONE_PAIRS` under a *different slot convention*.
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

## 7. The third side: what belongs to xtrax

Added 2026-10-07 after the user asked *"is there anything that belongs xtrax side?"* Yes, and
it is narrower than the proxide share — but one piece of it is a **gap in the #198 separation
audit**.

### 7a. The audit never entered `src/aminx/model/`

`.praxia/docs/audits/261007_aminx-xtrax-separation-audit.md` (63 findings, at `d3a58a42`) scoped
itself to run/IO/execution/config/training. Grepping it for `model/`, `features.py` and `top_k`
returns **zero hits**. So everything below is uncovered by that sprint, not a restatement of it.

### 7b. `_top_k_row_chunked` is a hand-rolled `ChunkedMap`, and it pads where xtrax does not

`model/features.py:155-181` flattens the leading batch axes, pads up to a multiple of
`row_chunk` with `-inf` rows, reshapes to `(n_chunks, row_chunk, sort_len)`, runs `jax.lax.map`,
and slices the padding away.

`xtrax.transforms.chunked_map` (`transforms/map.py:8-33`) is that primitive — and strictly
better. It dispatches to `jax.vmap` when `n <= batch_size`, else to
`jax.lax.map(..., batch_size=)`, which per its own docstring (xtrax #5565) "runs the
`n // batch_size` full chunks as a scan and the `n % batch_size` remainder as one smaller
vmapped chunk, so peak memory stays bounded by `batch_size` and **nothing is padded**."

aminx computes on synthetic rows that xtrax would never materialize. Filed as debt **#2561**,
with the three reasons it is not a drop-in (axis-0-only iteration needs a flatten wrapper;
xtrax's `chunked_map` still lacks the #2391 size-1 guard, so the swap is blocked on xtrax
#2520; and the IREE stack-allocation budget must be **re-measured**, because a ragged remainder
is a different trace than a padded one).

### 7c. `top_k` itself is an export primitive, not protein code

`model/features.py:60-145` exists entirely for backend portability: `jax.lax.top_k` lowers to a
`stablehlo.composite` that IREE's importer marks illegal, and IREE does not honour stable-sort
tie order (measured 260911 — an integer `lax.sort_key_val` over 64 slots disagreed with XLA at
**45 of 64 positions**, carried entirely by the indices while the gathered values stayed
bit-identical, so a float parity check reported `max_abs_diff 0.0` and passed). The remedy —
folding the index into the sort key with `num_keys=2` for a strict total order — is a
miscompile guard and an export primitive. It contains no protein logic, and the #198 audit's own
line 20 puts miscompile guards and export tooling in xtrax. It should move, with aminx importing it.

### 7d. The one distinction worth stating carefully: pad mechanism vs pad semantics

Generic axis padding, trimming and the bucket ladder are xtrax's (already debt #2522/#2537).
But **what a pad row means to a protein is not generic**, and the Potts featurizer docstring is
the proof: a zeroed gap row puts N/CA/C/O at the origin and derives Cb from that frame, so
`N-N` into the gap is a real finite distance to the origin, and the Ca-Ca RBF of a
present→absent edge is the RBF of `D_max_i` (`families/potts_mpnn/features.py:9-20`). The
padding *mechanism* is xtrax; the padding *semantics* belong with the FeatureSpec in aminx, or
with the geometry in proxide. Conflating them is how a bucket boundary silently changes a
model's numbers.

### 7e. The split, stated once

| Library | Owns | Never knows |
| --- | --- | --- |
| **xtrax** | how work is *shaped* — vmap/chunk/scan/bucket, pad mechanics, memory estimates, miscompile guards, export-legal primitives | that it is a protein |
| **proxide** | what a protein *is* — parsing, atom37, chemistry, geometry kernels (kNN, RBF, dihedrals, NeRF) | that there is a model |
| **aminx** | what *this checkpoint* wants — FeatureSpec conventions, embeddings, model, sampling, parity | — |

aminx is the only one of the three that knows both. That is the whole point of it, and it is
also the test for any new code: if it would still make sense without proteins, it is xtrax's; if
it would still make sense without weights, it is proxide's; if it needs both, it is ours.

### 7f. What is already tracked, so this does not get re-filed

`utils/safe_map.py` / `safe_scan.py` duplication (#2533), three live copies of axis dispatch
(#2533), the size-1 guard gap (xtrax #2520), `BUCKET_LADDER` stranded in `export/rings.py`
(xtrax #2522), the generic half of `host/bucketing.py` (#2537), and dead `tiling/bucketing.py`
/ `tiling/pad.py` (#2532) are all already filed from the #198 audit. **#2561 is the only new
one**, and it is new precisely because the audit stopped at the model boundary.

## 8. Audit corrections (2026-10-07)

An adversarial audit was run against §§1-7 at the user's request. It returned `needs_work`. The
corrections below are the ones **I re-verified by hand** — a subagent report is a lead, not a
fact, and I did not take any of these on report alone. Items the audit raised that I did not
re-verify are listed in §8.7 as unconfirmed leads.

### 8.1 `ligand_features.py:249` is a COMMENT, not a literal

`:249` reads `# edge_in = 16 + 16 * 25 = 416`. The actual code is `:250`,
`eqx.nn.Linear(416, edge_features, ...)` — a bare `416`, which is *worse* than the `25` I
reported, because the derivation exists only in prose beside it. §1a's "three places" is also
wrong as a count: `model/laser/graphs.py` and `model/laser/encoders.py` carry further 25s, and
`laser/` is under `model/`, so §6's "verified over `src/aminx/model`" was not the exhaustive
sweep I claimed.

### 8.2 The three `25`s are NOT the same quantity — and §5 step 1 was wrong because of it

`packer.py:385` is `num_positional_embeddings + num_rbf * 25`, and packer builds its pairs with
its own 5×5 nested loop over `[n, ca, c, o, cb]` (`packer.py:502-505`). It never imports
`BACKBONE_PAIRS`. Rewriting its `25` as `len(BACKBONE_PAIRS)` — which §5 step 1 originally
instructed — would have introduced a dependency on a table packer does not use, coupling two
things that only happen to share a number. Corrected in §5.

Same class of error on `400`: it is 25×16 at `families/potts_mpnn/features.py:44` and 20×20 at
`potts_head.py:26-27`. A global rewrite to `N_AA ** 2` would corrupt the first. Also corrected.

**This is the most valuable thing the audit found**, because §5 step 1 was phrased as a
mechanical instruction someone could have followed.

### 8.3 `rbf_dim` and `pos_embed_dim` are dead, so §1c overstated the risk

Both are declared `eqx.field(static=True)` at `features.py:270-271` and assigned at `:289-290`,
and **read nowhere** in `src/`, `tests/` or `scripts/` (grep). They are write-only. They
therefore cannot "silently disagree" with `RADIAL_BASES` — nothing consumes them. My line
citation was also off by three (`:286-288` → `:289-291`).

`POS_EMBED_DIM` (`:57`) and `MAXIMUM_RELATIVE_FEATURES` (`:56`) are likewise unused in `src/`;
the comment at `:294` cites `POS_EMBED_DIM` while the code beside it writes `16`.

**The sharper point the audit made, which I should have made myself:** a wrong *width* is a
loud failure — a matmul or deserialisation shape error. Only a wrong *order* is silent. §1c
blurred the two. That does not weaken §3's conformance-check requirement; it sharpens it, since
it means the `pair_order` stamp is doing essentially all the work and the width check almost
none.

### 8.4 Nothing imports proxide's Python RBF — downgrade §4a's consequence

`grep` over `src/`, `tests/`, `scripts/` for `proxide.geometry` returns **zero hits**. So "any
f64 work routed through `proxide.geometry` re-finds a bug aminx already fixed" is speculative,
not a live hazard. **Downgraded.** The *drift* claim itself stands — aminx has `rbf_centers(dtype)`
and proxide does not, verified by diff, and nothing in proxide has a fix aminx lacks.

### 8.5 The shipping proxide kernel is Rust, and it asserts a DIFFERENT slot convention

This is the finding that most changes the picture, and it strengthens §4's thesis while
invalidating §4a's target.

`proxide/crates/proxide-geometry/src/geometry/radial_basis.rs` holds the live implementation:
`pub const BACKBONE_PAIRS: [[usize; 2]; 25]` at `:25`, with `RBF_MIN`/`RBF_MAX`/`RBF_SIGMA`
declared **`f32`** at `:14-18`. Its own doc comment reads `/// Backbone atom indices: N=0, CA=1,
C=2, CB=3, O=4` — the **atom37** layout, with CB at 3 and O at 4 — whereas aminx's
`BACKBONE_PAIRS` is written against the **PDB** layout (`utils/atom_ordering.py:33-41`: O at 3,
CB at 4). Two tables of the same 25 pairs under two different slot conventions, feeding the
same `rbf_features` seam at `ProteinFeatures.__call__`.

So: proxide does not merely *have* an opinion about pair order, it has a **different** one, in
the half that actually ships. §4's "proxide must become parameterized, not opinionated" is more
urgent than written; §4a's "unify the two `radial_basis.py` files" is aimed at dead code.

### 8.6 §7c is STRONGER than I wrote: xtrax already documents both hazards

`xtrax/export/safety.py:11-24` already names `"unlegalizable-op" — currently jax.lax.top_k,
which lowers to a stablehlo.composite wrapping chlo.top_k that the importer marks explicitly
illegal`, and `"sort-stability" — ... silently produces different index results on ties,
invisible to a float-tolerance parity check because the divergence lives entirely in integer
indices.` That is aminx's `top_k` docstring reasoning, already in xtrax, with `_TOP_K_DETAIL`
at `:298` saying the obvious replacement "inherits the sort-stability problem and needs the same
tiebreak fix."

xtrax already **diagnoses** this and does not **supply** the cure; aminx has the cure. Moving
`top_k` is not a proposal to relocate protein code — it is completing something xtrax started.
I did not cite this and should have.

### 8.7 Raised by the audit, NOT re-verified by me — treat as leads

- That `families/laser_mpnn/featurize.py` (≈1505 lines) is the real LASEr featurizer, making
  the featurizer count ≥6 rather than 4.
- That foundry already permutes weight columns at conversion time
  (`mpnn/utils/weights.py:225-278`), which would be an alternative to a runtime `pair_order`
  field that §3 does not consider. If true this is a genuine design fork worth weighing.
- That `row_chunk` has three call sites (`export/wrappers.py:193,256,431`) plus two scripts, not
  one, and that the other four `top_k` sites cannot receive it at all — so the IREE fix covers
  only `select_neighbors`.
- That proxide's Rust kernel zero-fills a missing Gly CB while aminx builds a virtual CB, a
  substantive numerical divergence across the `rbf_features` seam.
- That padding to a multiple of `row_chunk` *incidentally* avoids the #2391 size-1 remainder
  hazard that xtrax's `chunked_map` lacks a guard for — i.e. aminx's version may be safer on
  that axis, not merely more wasteful. If so, "strictly better" in §7b is wrong.
- That `families/potts_mpnn/model.py:30-38` hardcodes `48/32/21`, bypassing
  `get_topology_for_checkpoint` entirely, so §1e's "load-bearing" holds for stock/ligand/packer
  but not for Potts.
- That §3's "legacy fallback for the eight `LEGACY_ALIAS_MAP` names" undercounts: all 15
  packaged `.eqx.zst` carry no config.

The first, fourth and fifth of these would each change a recommendation if true. They are the
next things to probe, and none should be cited until they are.

### 8.8 What the audit did NOT overturn

§2 (the three good precedents), §3's core design, §4's mechanism-vs-convention rule, §7a (the
separation audit contains zero hits for `model/`, `features.py` or `top_k` — I re-ran that
grep), §7b's identification of `_top_k_row_chunked` as a duplicated primitive, and §11's point
that a widths-only check cannot catch an ordering divergence. Debt #2561 stands, with the
"strictly better" wording flagged for correction pending §8.7.

## 9. Three of the §8.7 leads, settled by hand

Probed while CI ran. All three were read-only; nothing scoped was touched.

### 9.1 The missing featurizer is real, and it is the LARGEST one

`families/laser_mpnn/featurize.py` is **1505 lines**, and its own docstring says: *"Host-numpy
LASErMPNN featurizer. Ports inference parsing in `run_inference.py` (`ProteinComplexData`) and
the post-`construct_graphs` first-shell mask... Geometry is NumPy; nothing here is traced by
JAX."* §1f's count of four is wrong; it is at least six once this, `model/packer.py` and
`model/laser/graphs.py` are counted separately.

**And it is a better example for §4 than anything I used.** It is host-side parsing and
geometry with no weights in it — by this document's own rule (*"if it would still make sense
without weights, it is proxide's"*) a 1505-line module sitting in aminx is the single largest
live violation of the boundary I proposed. I argued the rule from small cases and missed the
biggest one.

### 9.2 proxide's Rust kernel really does zero-fill — but the divergence is LATENT, not live

Confirmed at `radial_basis.rs:122-125`: `if coord_a[0].is_nan() || coord_b[0].is_nan() { continue; }`,
over a buffer pre-zeroed by `vec![0.0f32; ...]` at `:106`. A missing atom leaves that pair's
whole 16-wide block at exactly `0.0`.

aminx cannot reach that state at all: `utils/coordinates.py:123-129` **unconditionally** computes
a virtual C-beta from the N→CA and CA→C bond vectors (`compute_c_beta`) and never reads a CB
column from the input. So aminx always has a finite CB, for Gly and non-Gly alike.

That makes the divergence **wider than the audit claimed** — it is not only Gly. Because aminx
always uses the *idealized* CB while proxide uses whatever its caller put in the CB slot, the
two disagree on every residue whose crystallographic CB deviates from ideal geometry, not just
on the ones that have none. (proxide's kernel does not build CB itself; `radial_basis.rs:82,88`
documents `backbone_coords` as an *input*, `(N_res, 5, 3)` = `[N, CA, C, CB, O]`.)

**But it is not a live defect.** Every caller that passes `rbf_features=` — `export/wrappers.py:203,266,441`
and `scripts/browser_validation/p07_split_{export,feasibility}.py` — computes that tensor with
**aminx's own** `compute_radial_basis`. Nothing in the repo feeds a proxide-computed RBF into
the seam. So this is the same verdict shape as §14 of the ProtonPottsMPNN spec: a real
convention divergence, latent because the two sides are not currently wired together, and a
trap for whoever wires them. **Downgrade from "defect" to "latent"** — and note the seam's own
comment at `features.py:330` ("assuming they match proxide's implicit indices") is an unchecked
contract exactly here.

### 9.3 "Strictly better" in §7b is WRONG — aminx's padding is incidentally the safer behaviour

`utils/safe_map.py:34-40` states it outright: *"It never lets XLA compile a vmapped chunk of
size exactly 1 (aminx #2391). On one GPU stack (TITAN RTX, jax 0.10.2, CUDA 12.9) a jitted
size-1 batch around a matmul whose intermediate is square returned wrong activations silently.
A chunk of 1 arises three ways... a remainder of 1 (`n % batch_size == 1`; lax.map vmaps the
remainder)."*

Padding to a multiple of `row_chunk` **structurally cannot produce a remainder**, so
`_top_k_row_chunked` is immune to that hazard by construction. xtrax's `chunked_map` produces a
remainder by design — that is precisely the property §7b praised — and lacks the guard. So the
swap trades wasted work for exposure to a known silent-miscompute. The *sequencing* in debt
#2561 was right (blocked on xtrax #2520); the word **"strictly better" is not**, and #2561's
wording is corrected accordingly.

The same docstring also confirms the intended direction independently of my argument: *"the
same guard belongs in xtrax's `chunked_map`, after which this helper is a re-export and then
removed (aminx debt #2371)."*

### 9.4 Still unverified

The foundry conversion-time column permutation (`mpnn/utils/weights.py:225-278`) as an
alternative to a runtime `pair_order` field, the three-vs-one `row_chunk` call sites, and the
`potts_mpnn/model.py:30-38` hardcodes bypassing `get_topology_for_checkpoint`. Not cited
anywhere until probed.

## 10. The last three leads, settled

### 10.1 Foundry already solved the ordering problem at CONVERSION time — and §3 should have considered it

`ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/utils/weights.py:~228-280` does exactly this to
`graph_featurization_module.edge_embedding.weight`:

1. split the loaded weight into `[:, :num_positional_embeddings]` and the rest;
2. `view(out_dim, num_atoms * num_atoms, num_rbf)`;
3. reorder with `[legacy_order[name] for name in new_order]`;
4. reshape back and `torch.cat` the two halves.

Three things follow, and they matter more than the lead itself.

**(a) There is a route with no runtime parameter at all.** §3 proposed `pair_order` as a
`FeatureSpec` field consulted at featurization time. Foundry instead normalizes the *weights*
once, at conversion, and ships a featurizer with a single fixed order. That is cheaper at
runtime and removes the per-family branch entirely. It is only available when you control
weight conversion — which for a port is exactly the case. A runtime field is needed only if one
featurizer must serve two live conventions *simultaneously*. **§3 should present these as two
options with that as the deciding question**, not assume the runtime field.

**(b) Foundry already parameterizes the pair count.** It is `num_atoms * num_atoms` where
`num_atoms = num_backbone_atoms + num_virtual_atoms` — so upstream's "25" is derived, and it
generalizes past 5 atoms to include *virtual* atoms. The FeatureSpec idea is not speculative;
the fork this document is about has already implemented its core.

**(c) It independently confirms §8's corrected width formula.** Foundry's split implies
`edge_in = num_positional_embeddings + num_atoms² × num_rbf` — the same shape as the corrected
`pos_embed_dim + rbf_count * len(pair_order)`, and *not* the `rbf_count * (len(pair_order) + 1)`
I first wrote. Two independent derivations agreeing is worth more than either alone.

The same file goes on to permute the amino-acid token order coming out of the model, which is
the alphabet-boundary question (spec §5) appearing in the same place for the same reason.

### 10.2 `row_chunk` has FIVE call sites, not one

I wrote "set in exactly one place." Wrong — I conflated the constant's *definition* with its
*uses*, and my grep was truncated before reaching the rest:

- `export/wrappers.py:92` defines `EXPORT_TOP_K_ROW_CHUNK = 32`
- used at `export/wrappers.py:193`, `:256`, `:431` — three separate export wrappers
- plus `scripts/browser_validation/p07_split_export.py:124` and `p07_split_feasibility.py:154`

"Runtime is unaffected" still holds — all five are export-path — but "one place" was wrong and
is corrected here and in debt #2561.

**A new observation that refines §9.3.** `tests/export/test_top_k_export.py:287` parametrizes
`row_chunk` over `[1, 8, 32]`. At `row_chunk=1` every chunk is size 1 — which is `safe_map`'s
case (b), `batch_size == 1`, one of the three #2391 shapes it exists to forbid. So
`_top_k_row_chunked` is immune to the *remainder*-of-1 case by construction (§9.3) but has **no
guard against the other two**, and the test suite exercises one of them. Production uses 32, so
this is latent, not live — but "aminx's version is the safer one" is narrower than §9.3 implies:
it is safer on exactly one of three hazards, and unguarded on the rest.

### 10.3 Potts bypasses `get_topology_for_checkpoint` entirely — so §1e has TWO sources, not one

`families/potts_mpnn/model.py:25-31` hardcodes its own topology:

```python
_EDGE_FEATURES = 128      _HIDDEN_FEATURES = 128
_ENCODER_LAYERS = 3       _DECODER_LAYERS = 3
_K_NEIGHBORS = 48         _POSITIONAL_EMBEDDINGS = 32
_VOCAB = 21
```

So §1e's "architecture is inferred by substring-matching a filename" is true for the stock,
ligand and packer paths and **false for Potts**, which never consults `io/weights.py` at all.
That makes the problem *worse* than §1e stated rather than better: there are at least two
independent, unreconciled sources of model topology, and nothing cross-checks them.

**This is directly on the ProtonPottsMPNN critical path.** `_VOCAB = 21` and the `etab_raw`
annotation `Float[Array, "L K 20 20"]` (`model.py:~45`) are both wrong for V=30, and they live
in a different file from the `io/weights.py` ladder §1e pointed at. A port that fixed only the
ladder would leave these untouched — and `_VOCAB` is not a tensor shape, so it would not fail
loudly.
