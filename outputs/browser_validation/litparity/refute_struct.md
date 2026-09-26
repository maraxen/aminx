# Phase 3: Adversarial Refutation — lens: ALGORITHMIC STRUCTURE

task_id: `260923_aminx-browser-validation`

## Lens

Trace aminx's actual execution (file:line on both sides) against the reconstructed
algorithm (recon_algo.md) and the reconciled clauses (clauses.json / reconcile.md).
Assume a hidden defect exists; try to prove it. Honesty tax: assumptions stated up
front, default to DEVIATION/inconclusive under weak evidence.

## Assumption(s) hunted

1. That the full-decoder-recompute-per-wave design in `AutoregressiveDecode`
   (`inference/decode/autoregressive.py`) is **not actually** informationally
   equivalent to the reference's incremental per-position `h_V_stack` caching —
   i.e. that `ar_visibility_mask` (MATCH, core=true) is wrong, and a later wave's
   full recompute silently perturbs an earlier, already-decided position's
   hidden state or lets it see something it shouldn't.
2. That the wave-schedule / AR-mask construction (`utils/autoregression.py`)
   contains a convention bug (order-vs-rank, off-by-one, or similar) not caught
   by the reconciler, specifically in code paths *other* than the one the
   reconciler traced for the main sampling pipeline.
3. That the tied-group mechanisms (fixed override, last-member bias, weighted
   fusion, intra-group visibility) documented in clauses.json don't actually
   match the code as I re-derive it independently, or hide a further
   undocumented divergence (e.g. the k-NN-disjoint-groups "no visible
   difference" claim in `tied_intra_group_visibility`).

## Per-target findings

### Target 1 — AR decode loop: per-step state update / full recompute vs. incremental caching

**Trace.** `AutoregressiveDecode.__call__` (`inference/decode/autoregressive.py:155-480`)
calls `model.decoder.call_conditional` **once per wave**, over **all L positions**,
via `_decode_one_step` (`inference/decode/_kernel.py:16-83`). `Decoder.call_conditional`
(`model/decoder.py:655-739`) builds `sequence_edge_features`/`mask_bw`/
`masked_node_edge_features` **once** from the *fixed* `ar_mask` and the *current*
one-hot sequence carry (`pack_conditional_decoder_static_edges`,
`model/decoder.py:80-148`), then re-derives `layer_edge_features` from the
**evolving** `loop_node_features` at every layer
(`conditional_decoder_layer_edge_features`, `model/decoder.py:151-189`). Critically:
- The **mask_fw** (not-yet-visible) branch is built from `node_features`, the
  function's *encoder-level input parameter* — never touched by the per-layer loop
  (`model/decoder.py:126-136,147`). This is exactly the reference's
  `h_EXV_encoder_fw` (fixed, precomputed once, `model_utils.py:258-260`).
- The **mask_bw** (visible) branch gathers `loop_node_features[neighbor]`, i.e. the
  full-batch recompute's *own* current-layer state for that neighbour
  (`model/decoder.py:184-189`), not a frozen per-step cache.

By induction on layer depth: a position's `loop_node_features` at any layer is a
pure function of (a) the fixed `ar_mask`, (b) the frozen encoder features, and
(c) the **final, frozen tokens of its own mask_bw-visible predecessors** — never of
anything at or after its own rank. Recomputing the whole decoder fresh every wave
therefore cannot perturb an already-decided position's value, and produces bit
values identical to what the reference's incremental cache would have frozen at
that position's own decode step. Tie groups are the sole exception (cyclic
mutual visibility within one wave), already flagged in `tied_intra_group_visibility`.

**Evidence (executed).** `/tmp/probe_ar_equivalence.py` (ephemeral, not committed):
built a `Decoder` (3 layers, random-init weights, no checkpoint needed), a random
6-position k-NN graph, and a random rank array; compared (i) a "full" call with the
entire final sequence revealed vs. (ii) six "partial" calls, each revealing only
the positions each target position can causally see. Run via
`JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 uv run --frozen --extra dev --extra benchmark python3`.

```
=== POSITIVE TEST ===
step=0 position=4 max_abs_diff=0.000e+00
step=1 position=5 max_abs_diff=0.000e+00
step=2 position=1 max_abs_diff=0.000e+00
step=3 position=2 max_abs_diff=0.000e+00
step=4 position=0 max_abs_diff=0.000e+00
step=5 position=3 max_abs_diff=0.000e+00
overall max_abs_diff=0.0
[control a] perturbing a DECIDED (mask_bw-visible) neighbour's token: diff=3.3e-04 / 4.9e-04 / 6.6e-04 (nonzero -> test is sensitive, not vacuous)
[control b] perturbing a NOT-YET-decided (mask_fw) position: diff=0.0 (exactly invariant, as required)
```

Positive and negative controls both behave as required: the full-batch recompute
is **bit-exact** equivalent to incremental reveal, and the test is demonstrably
sensitive (control a) rather than trivially passing (no dead mechanism).

**A first attempt at this probe (before correcting for Target 2 below) showed a
false positive of 7.96e-04 at one position** — traced to my own script passing a
plain `jax.random.permutation` as if `generate_ar_mask` consumed step→position
"order," when it actually consumes position→step "rank" (see Target 2). Once
`rank` was passed directly, the diff went to exactly 0.0 across all positions and
both controls. This confusion is the *same* bug documented in Target 2, encountered
independently while building this probe — which is itself corroborating evidence.

**Verdict:** `ar_visibility_mask` (MATCH, core) — **not found**, high confidence.
The full-recompute design is a mathematically exact substitute for the
reference's incremental cache, for the non-tied portion of any schedule with a
fixed, cycle-free `ar_mask`. `message_passing_decoder` (MATCH) likewise holds up:
the layer structure (message MLP, mask_bw/mask_fw split, static mask_fw branch)
matches the reference's algorithm exactly at the code level.

**Recommendation:** none — treat as confirmed; worth a permanent regression test
mirroring this probe's construction (full-vs-incremental-reveal on a tiny random
`Decoder`) since the equivalence is currently only argued in the clause's prose,
not tested directly (the existing test suite tests self-exclusion and tie
visibility, not this specific full-batch/incremental equivalence).

### Target 2 — AR mask construction from a wave schedule: order vs. rank convention (NEW, UNDESIGNED)

**Trace.** `generate_ar_mask` (`utils/autoregression.py:141-225`) computes, for the
untied branch: `ar_mask[i,j] = decoding_order[i] >= decoding_order[j]`. This
directly compares `decoding_order[i]` and `decoding_order[j]` as if `decoding_order`
were a **rank array** (`rank[position] = step`). This is explicit and pinned by
`tests/inference/test_fixed_position_self_visibility.py:143-200`
(`test_matches_colabdesign_bit_for_bit_on_the_untied_path`,
`test_generate_ar_mask_consumes_a_rank_array_not_an_order_array`), which documents:
"`generate_ar_mask` compares `decoding_order` values directly, so it expects a
RANK array" and flags this as "a live inconsistency worth pinning."

Two of `generate_ar_mask`'s three non-test call sites feed it an **order array**
(step→position), not rank, whenever the schedule is non-trivial:
- `sampling/sample.py:177-185`: `decoding_order, _ = decoding_order_fn(k_order, L, None, None)` then
  `ar_mask_single = generate_ar_mask(decoding_order, ...)`.
- `inference/decode/ste.py:377-386`: identical pattern
  (`final_decoding_order, _ = self.decoding_order_fn(...)` → `generate_ar_mask(final_decoding_order, ...)`).

`decoding_order_fn` is a **user-injectable, documented public parameter**
(`sample.py:27`, `ste.py:109-135`, `factory.py:44-98`, all typed `DecodingOrderFn`).
Its built-in implementations, `random_decoding_order` and `single_decoding_order`
(`utils/decoding_order.py:27-133`), both return **order-convention** arrays
(`jax.random.permutation(key, arange(L))` — read exactly like the reference's own
`decoding_order[:, t_] = position`, and exactly like `WaveScheduleBundle.from_tie_groups`
consumes it correctly — see below). Feeding this straight into `generate_ar_mask`
silently builds the mask for the **algebraic inverse permutation** of the one the
caller asked for.

**By contrast, the *other* consumer of an order array, `WaveScheduleBundle.from_tie_groups`
(`types/bundles.py:212-252`), gets this right**: `for i in decoding_order.tolist(): g = tie_group_map[i]`
uses each element of `decoding_order` as a **position** to index into `tie_group_map`
— i.e. it correctly treats `decoding_order` as order-convention. This is the path
`build_inference_bundle`'s `random_ar`/`frozen_random_sigma`/coloring schedules go
through (`inference/bundle_builder.py:246-262` → `generate_wave_ar_mask`, not
`generate_ar_mask`), so **the main `AutoregressiveDecode` production pipeline is not
affected** by this bug — only the separate `sample.py`/`ste.py` entry points that
call `generate_ar_mask` directly are.

**Evidence (executed).** Ran, CPU-only, no model:

```python
order_from_fn = random_decoding_order(key, L=8)  # [1 3 5 0 2 6 7 4]
ar_mask_actual   = generate_ar_mask(order_from_fn)          # what sample.py/ste.py build
ar_mask_intended = ar_mask_from_order(order_from_fn)         # reference-convention mask
```
Result: **24 of 64 cells differ** between `ar_mask_actual` and `ar_mask_intended`;
both are still valid, self-excluding, non-degenerate causal masks (diag sum 0,
nnz 28 both sides) — i.e. this is not a leak or a broken invariant, it is a
**silently different, but still internally valid, decoding order** than the one
the caller specified.

**Why this is invisible today.** `random_decoding_order`'s output is a uniformly
random permutation, and the inverse of a uniform random permutation is *also*
uniformly random — so for the *built-in* random arm alone, this bug produces no
statistically detectable difference and would not be caught by any
order-marginal-uniformity test (e.g. spec P19). It only bites, per the pinned
test's own words, "code that specifies a decoding order deliberately (a
counterfactual schedule, or reproducing a named order)" — which is exactly the
scenario clauses.json repeatedly recommends as the reference-compatible fix for
`decoding_order_default`/`random_order_fixed_first` ("pass a host-built
WaveScheduleBundle... carrying the reference argsort order"). That specific
recommended fix goes through `build_inference_bundle(wave=...)` →
`WaveScheduleBundle.from_tie_groups`, which is **not** affected. But a caller who
instead reaches for the more obviously-named, public `decoding_order_fn` hook on
`make_sample_sequences`/`AutoregressiveSTE` to inject the reference's exact
argsort order will get silently the wrong mask, with no error and no warning.

**Verdict:** **DEVIATION — new, undesigned, not present in clauses.json.**
Severity **major**: it is a real, executable output difference (24/64 mask cells)
in a supported, documented customization path (`decoding_order_fn`), outside the
already-validated exact-tier harness (which uses `build_inference_bundle(wave=...)`,
not this path). Not core to the currently-audited harness configuration, but a
live landmine for exactly the workaround the audit recommends elsewhere.

**Recommendation:** Either (a) change `generate_ar_mask` to accept an order array
and invert it internally (matching `WaveScheduleBundle.from_tie_groups`'s and the
reference's convention, which is what every *other* order-producer in this
codebase already assumes), or (b) rename/document the parameter as `rank` and fix
`sample.py:177-185` and `ste.py:377-386` to invert `decoding_order_fn`'s output
before calling `generate_ar_mask`. Add a regression test asserting
`generate_ar_mask(rank-of(order)) == ar_mask_from_order(order)` for a
non-involutory permutation (the existing pinned tests use only `rank` arrays
directly, so they cannot catch a caller passing the wrong convention).

### Target 3 — Fixed-position pre-seeding and visibility

**Trace.** `init_sequence` is pre-seeded with `fixed_tokens` at `fixed_mask`
positions *before* any wave runs (`inference/decode/autoregressive.py:238-245`).
Visibility is still gated purely by `cond.ar_mask` (mask_bw/mask_fw), which is
built from the *decoding order*, not from fixed/designed status, for the default
`fixed_n_to_c` schedule: `select_decoding_order` returns
`jnp.arange(seq_len)` unconditionally for `"fixed_n_to_c"`
(`inference/schedule_selector.py:365-366`), and `bundle_builder.py:272-278` builds
`generate_ar_mask(jnp.arange(seq_len), tie_group_map=...)` for that arm — i.e.
**strict N→C order regardless of fixed/designed status**, matching
`decoding_order_default`'s note exactly ("fixed tokens are pre-seeded... but are
only VISIBLE to a designed position if they precede it in N→C order"). Confirmed
directly in code, not just by the reconciler's prose: the "fixed_n_to_c" name
promises nothing about reordering fixed residues to the front; it is a plain
positional arange plus pre-seeding.

**Verdict:** confirms `decoding_order_default` / `random_order_fixed_first`
exactly as already recorded (DEVIATION, designed for the default arm, core=true
undesigned for `random_ar`/`frozen_random_sigma` in the "no fixed-first" sense).
No further, new divergence found here beyond what Target 2 already adds for the
`sample.py`/`ste.py` entry points specifically. **Not found** (no additional
defect), high confidence.

### Target 4 — Tied groups: fixed override, last-member bias, fusion weights, intra-group visibility

**Trace.** Re-derived all four mechanisms independently from
`inference/decode/autoregressive.py:386-412` and `inference/logits.py:395-435`,
without relying on clauses.json's prose:
- **Fixed override** (`autoregressive.py:399-406`): `group_fixed_token = max(fixed_tokens
  over fixed members in the group)`; applied to **every** member if **any** is
  fixed. Matches `tied_fixed_override`'s description of aminx's behaviour exactly
  (all members forced, including overwriting a non-max fixed member's own native
  with the max-index one — reproduced independently by hand-tracing the `jnp.max`
  call, not just re-reading the clause note).
- **Last-member bias vs. per-member bias**: aminx's `sampling_logits` is built
  by `vmap`ping `ar_logit_transform` over **each position's own** `cond_bias`
  row (`autoregressive.py:353-383`), so every member's own bias is baked into its
  own `log_softmax` term before the group PoE sum
  (`TieGroupProductOfExperts.__call__`, `logits.py:414-435`) — confirmed the
  formula is `sum_t log_softmax(logits_t + bias_t)`, matching `tied_last_member_bias`
  and `tied_fusion_weights`'s notes precisely, including the "unweighted, no
  `symmetry_weights` input" observation (no such field exists anywhere in
  `ConditioningBundle`, `bundle_builder.py`, or `logits.py`).
- **Default tie-group fusion strategy**: confirmed `make_stage_set` wires
  `tie_group_fuse=TieGroupProductOfExperts()` by default (`logits.py:531`), i.e.
  the PoE formula is genuinely what production code uses, not merely one of two
  registered strategies (`TieGroupLogsumexpMean` also exists,
  `logits.py:355-392`, but is not the default and would need an explicit
  `stage_set.tie_group_fuse=...` override to engage).
- **Intra-group visibility**: `generate_wave_ar_mask`'s `same_wave & same_group`
  term (`utils/autoregression.py:290-294`) is symmetric — both directions of a
  same-group pair are visible, unlike the reference's one-directional
  (`t_list`-order-dependent) sequential visibility. Confirmed by direct reading,
  matching `tied_intra_group_visibility`'s claim.

**Hunting for the k-NN-disjoint "no visible difference" claim.** The claim that
disjoint tie members produce identical logits to the reference *when they are not
each other's k-NN neighbours* is structurally guaranteed, not merely likely: the
decoder only ever reads `ar_mask` through `jnp.take_along_axis(ar_mask,
neighbor_indices, axis=1)` (`model/decoder.py:144`), i.e. a same-group mutual-
visibility bit at `ar_mask[i,j]` is **inert** unless `j` literally appears in
`i`'s `neighbor_indices` row (or vice versa). No indirect (multi-hop, through a
shared third neighbour) leak specific to tying was found: any such multi-hop
propagation is the *ordinary*, intended behaviour of a multi-layer causal decoder
(a later-decided position legitimately sees an earlier one's influence through a
k-NN chain), and both the reference and aminx exhibit it identically for
non-tied positions — it is not a tie-specific artifact and does not compound
the already-recorded `tied_intra_group_visibility` deviation.

**Verdict:** all four tied-group clauses (`tied_fixed_override`,
`tied_last_member_bias`, `tied_fusion_weights`, `tied_intra_group_visibility`) —
**confirmed as stated, not found to be broken.** High confidence; each mechanism
was re-derived from code independently rather than trusted from the clause text.

### Target 5 — Any MATCH verdict breakable?

Attempted specifically on `ar_visibility_mask`, `message_passing_decoder`, and
`tied_group_step_placement` (all core=true). None broke under trace + execution.
The one genuine break found (Target 2) is **not itself a clauses.json MATCH
entry** — `generate_ar_mask`'s order/rank convention is not covered by any
existing clause at all, which is why it survived three independent
reconstructions and a reconciliation pass: none of math/algo/protocol
reconstruction was reading `sample.py`/`ste.py`'s calling convention, only
`model_utils.py`'s.

## Recommendation summary

1. Fix or clearly document the order/rank convention split between
   `generate_ar_mask` (rank-in) and `WaveScheduleBundle.from_tie_groups`/every
   `DecodingOrderFn` implementation (order-out) — Target 2. This is the one
   actionable, novel defect from this pass.
2. Add a permanent test for the full-recompute/incremental-equivalence property
   demonstrated in Target 1's probe (currently only argued in prose).
3. No changes needed for Targets 3/4 — existing clauses.json verdicts hold up
   under independent re-derivation and (for Target 1) execution.

```json
{"defects": [{"id": "ar_mask_order_rank_convention_mismatch", "paths": ["P07", "P19"], "severity": "major", "evidence": "generate_ar_mask expects rank[position]=step but sample.py:181/ste.py:383 feed it decoding_order_fn's order[step]=position output directly, producing a silently different (24/64 cells on an 8-residue test) but still-valid causal mask for any non-identity custom decoding_order_fn", "runnable": true}, {"id": "ar_visibility_mask", "paths": ["P07", "P05"], "severity": "accepted", "evidence": "full-batch per-wave decoder recompute verified bit-exact (max_abs_diff=0.0) vs incremental reveal on a random 6-position Decoder with positive+negative controls; MATCH confirmed, not broken", "runnable": true}]}
```
