---
title: "Complexity guards for the scan/while/fori hot paths beyond AR decode (debt #1981)"
description: "Shared jaxpr loop-scaling guard helper and per-loop red-checked guards for encode scan, STE x2, kernel_dispatch chunked/Scan, ligand tiling and scoring score_pairs; Potts/ProtonPotts/Langevin deferred."
status: draft
task_id: 261124_aminx-loop-complexity-guards
date: '261124'
---

# Complexity guards for loop hot paths beyond AR decode (debt #1981)

## 0. Overview

The AR sampler once ran the full decoder over all L rows each wave (O(L^2 k)); parity tests passed and it
surfaced only as a projected 435-483 h calibrate. `tests/inference/decode/test_incremental_autoregressive.py`
now guards AR decode structurally with `xtrax.profiling.loop_scaling`. This spec applies the same idea to
the other loops, with a second criterion because most of them have a fixed trip count. No implementation
here; every number below came from throwaway probes (section 1, tag P) that the fixer re-runs as tracked,
pre-registered tests.

## 1. Assumption Ledger

Tags: V(path:line) = read the code; P = measured by a throwaway probe in `/tmp/claude-1000/p/p1..p5.py` on
aminx main 23ce7456 with `.venv` (xtrax 0.4.1b7 installed, pyproject pins 0.4.1b9, jax 0.11.2, CPU). P rows are
UNVERIFIED in ledger terms because aminx has no `scripts/loop/adversarial_metrics.py` spike-run harness; Task 0
re-runs them under tracked tests.

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | `loop_bodies` traces only and reports per-iteration "work" = 2MNK per `dot_general`, output numel for other eqns, nested loop bodies x trips, `cond` = max branch, `dynamic_update_slice`/scatter costed by the update operand | helper thresholds change | VERIFIED | read: /home/marielle/projects/xtrax/src/xtrax/profiling/loop_scaling.py:157-206 (installed 0.4.1b7 differs only in pallas fields: diff checked) |
| A2 | A `scan` has a static trip count; a `while` has `None` and `extent_scaling_report` measures it by running eagerly under `disable_jit` with `jax.lax.while_loop` monkeypatched, treating unmeasured as "scales" | the fixed-trip/while classification in D2 | VERIFIED | read: loop_scaling.py:244-322 |
| A3 | `extent_scaling_report` flags only when work ratio >= 1.5 AND trips scale; a fixed-trip loop is never flagged, whatever its per-iteration work | STE guard needs the new criterion | VERIFIED | read: loop_scaling.py:374-385 |
| A4 | The existing test's helpers are module-local and use that file's `model` fixture, `_bundle`, `_random_wave` | lifting changes the test | VERIFIED | read: tests/inference/decode/test_incremental_autoregressive.py:21,37-55,104-122,233-285 |
| A5 | `STEDecode` (decode/ste.py:373) lowers to ONE scan with fixed trips = `iterations`; per-iteration work ratio is ~2.0 per L doubling on clean code and ~4.0 with the inner decode replicated L times | criterion B threshold | UNVERIFIED | deferred: probe p1/p4: clean 2.007, 2.014 (L 32/64/128, iters 3 and 5); mutated 3.995 (L 32->64); re-run in Task 0 |
| A6 | `make_optimize_sequence_fn` jitted with traced `iterations` lowers to a `while` (trips None) whose work equals STEDecode's | optimize_ste needs its own extents | UNVERIFIED | deferred: probe p4: path `/2:jit[0]/13:while`, ratios 2.007, 2.014 |
| A7 | `make_encode_fn(use_rolling_state=True)` is one scan over S whose per-iteration work is independent of S (1.000 at S=2/4/8) and linear in L (2.008); a fold-over-S mutation gives 1.845-1.916 over S | encode guard extent | UNVERIFIED | deferred: probe p2/p5; mutation = wrapped `model.encoder` replicating itself S times |
| A8 | `_chunked_sample_decode` (kernel_dispatch.py:118-144) is one scan over N/batch chunks with per-iteration work 1.000 over N on a stub decode; re-running `f` over all `xs` per chunk gives 1.733-1.846 | decode seam guard; stub is a valid probe | UNVERIFIED | deferred: probe p5 (fresh N=16/32/64 and 48/96/192 to dodge the jit cache) |
| A9 | `_dispatch_axis`'s `Scan` branch (kernel_dispatch.py:187-211) is a pure carry-passthrough scan over xs with trips = N and per-iteration work independent of N | scan guard | UNVERIFIED | read: kernel_dispatch.py:187-211; deferred: probe p3 gave 1.000 at N=32/64/128. Whether the planner ever emits `Scan` for the sample axis is not checked (host/plan.py:71,619 only mention it) |
| A10 | `map_chunks_axis0` / `_multi` lower to a `while` (fori with traced trip count) whose body works on a `(cs, ...)` slab; the pad slack is the constant `cs-1`, so per-iteration work is independent of L | tiling guard extent | VERIFIED | read: src/aminx/model/ligand_tiling.py:57-87,160-200 (constant slack, `dynamic_slice` of `cs` rows); probe p2/p4: 1.000 for L 32/64/128 (real `_y_edges_coords_to_embed`, M=12, cs=8); full-input closure mutation 1.771, 1.871 |
| A11 | `score_pairs` (`_score_pairs_kernel`, scoring/score.py:431-492) is ONE scan over P=K*C pairs / chunk; per-iteration work is independent of K (1.000) and linear in L (2.005, 2.010); the loop body is ONE decoder pass per pair | D6 | VERIFIED | read: score.py:480-492 (`chunked_map(_one, arange(K*C))`), 509-536 (one `score_from_encoding` per pair, teacher-forced); probe p3 |
| A12 | A weak mutation does not move the ratio: injecting `einsum(masks, ones(K,L,4))` per pair changed work by 0.1%; only re-running the whole body over the extent (the AR-bug shape) gives 1.83-1.91 | every must-fire control must re-run the body | UNVERIFIED | deferred: probe p3 (weak, ratio 1.000/1.001) vs p5 (fold, 1.831/1.908) |
| A13 | `chunked_map` with `batch_size=None` is a vmap (no loop), so Vmap strategy paths contain no loop to guard | scope of D3 | VERIFIED | read: scoring/score.py:462-465 docstring is a claim; kernel_dispatch.py:183-184 calls it with `None`; spec relies on A11 probe where `chunk_size=2` produced a scan. Vmap-no-loop itself: deferred (not probed) |
| A14 | The decoder's dense `ar_mask` makes true per-iteration work O(L^2) at large L, so criterion B can drift above 2.0 as L grows | thresholds/extents | UNVERIFIED | deferred: ratios were 2.007 -> 2.014 at L 32 -> 128, still dominated by k-neighbour work; not probed above L=128 |
| A15 | There is no ADR/spec stating the scoring estimand; D6 is derived from code | D6 | UNVERIFIED | deferred: `.praxia/docs/decisions` lists only 261003_laser-score-tier3-f32-design.md and 260618_rs7b-averaged-scoring-design.md by name; neither was read |

## 2. Decisions

### D1. Shared helper module

New `tests/helpers/loop_scaling.py` (`tests/helpers/` exists: `__init__.py`, `multistate.py`; tests is a
package, `tests/__init__.py` exists). It contains only generic machinery, no aminx model fixtures:

- `loop_work_by_extent(fn, make_args, extents) -> list[LoopRow]` where `LoopRow(path, primitive, trips: list[int|None], work: list[int])`
  uses `loop_bodies` only (traces, never executes, never needs `disable_jit` trip measurement), asserts the
  loop `path` lists are identical across extents (same check as `extent_scaling_report`), and returns all
  extents in one pass so criterion B can use three points.
- `work_ratios(row) -> list[float]` between consecutive extents.
- `assert_work_independent(rows, *, only_trips_scaling=True, threshold=1.5)` (criterion A).
- `assert_work_at_most_linear(rows, *, threshold=3.0)` (criterion B).
- `scaling_summary(rows) -> str` for assertion messages.
- `fold_over(fn, n, arg_index)` helper: `jax.vmap` of `fn` over `arange(n)` perturbation, returns element 0,
  used by every must-fire control (shape of the AR bug: whole body recomputed over the extent). Documented
  in its docstring with the A12 lesson.

Lifting the existing test is NOT part of Task 1's behaviour change: the AR file keeps its module-level
`_decode_scaling`, `_trips_scale`, `_quadratic_loops`, `_unconditional_quadratic_loops` untouched (they use
`extent_scaling_report` plus a second trace for trip counts and are tied to its fixtures). Task 6 (optional,
last) re-expresses `_trips_scale`/`TRIP_SCALING_RATIO` through the helper only if both versions give the
identical flagged-loop list on force/off/sampling-vmap cases (a differential check, below); otherwise leave
both. Default recommendation: leave the AR file alone.

### D2. The two criteria

Let `w(e)` be a loop's per-iteration work at extent e, from `loop_bodies`. Extents are always compared on the
same loop path.

- **Criterion A, trip-scaling loops** (the loop iterates once per element of axis X: AR waves over L, encode
  scan over S, chunk loops over N or P, tiling over L/cs): per-iteration work must be independent of X.
  Extent = X, at least three points (X, 2X, 4X). Assert `w(2X)/w(X) < 1.5` and `w(4X)/w(2X) < 1.5` for every
  loop whose trips scale with X (`trips[i+1]/trips[i] >= 1.5`, or trips `None`). Clean measurements: 1.000
  (encode/S, decode seam/N, score_pairs/P, Scan/N, tiling/L). Fold-mutant: 1.73-1.92. Threshold 1.5 is the
  one xtrax and the AR guard already use.
- **Criterion B, fixed-trip loops over L-sized state** (STE optimise: trips = `iterations`, each iteration
  decodes the whole sequence): per-iteration work may scale at most linearly in L. Extent = L in
  {32, 64, 128}. Assert every consecutive ratio `< 3.0`. Clean 2.007/2.014; fold mutant 3.995. 3.0 sits
  between 2.0 (linear) and 4.0 (quadratic) with margin for dense `ar_mask` L^2 terms (A14) and for padding.
  Trips are asserted equal across extents (a fixed-trip loop whose trips start to scale becomes A's business
  and is a failure of its own: "trips must not depend on L").
- Criterion B is also applied to the L extent of loops guarded by A (encode scan, score_pairs, decode seam is
  length-free), because per-iteration work must still be linear in L: ratio < 3.0 at L.
- **Noise and padding.** Work is a deterministic jaxpr proxy, so there is no noise; the only sources are
  padding and bookkeeping ops. (i) Use extents that are multiples of any chunk size; add ONE non-multiple
  extent for tiling (L=37/74/148, cs=8) asserting the same bound (slack is constant, A10). (ii) Hold every
  non-extent dimension fixed (K, C, S, M, cs). (iii) Never compare a clean loop to a different jaxpr path;
  path mismatch across extents is an error, not a skip. (iv) A `jnp.searchsorted` scan (log2 L trips, body
  carries O(L)) has trips ratio < 1.5 and is exempt from A by the trips test (existing TRIP_SCALING_RATIO
  rationale, test lines 228-233).
- **Vacuity check** (like the AR test's "no loop whose trip count scales"): every guard asserts at least one
  loop of the expected primitive and path fragment exists, and that A-guards see at least one trips-scaling
  loop, so a refactor that removes the loop cannot silently pass.

### D3. Scope, ranked by risk

In scope now (each: entry point, section D4):

1. Ligand tiling `map_chunks_axis0` and `map_chunks_axis0_multi` (ligand_tiling.py). Highest risk: callers
   pass `fn` closures (`ligand_features.py:453-466`, `ligand_mpnn.py:406-443`); a closure that captures the
   full `y` makes every tile O(L). Cheapest to guard (no model).
2. STE decode (`decode/ste.py:373`) and legacy `optimize_ste.py:256`, criterion B. Both run a full decode
   per iteration; fixed trips defeated the existing criterion entirely.
3. `score_pairs` scan (`scoring/score.py:431-492`), criterion A over P and B over L. The loop that scores
   every (mask, sequence) pair; this is where a K-fold recompute would hide.
4. kernel_dispatch `_chunked_sample_decode` (118-144) and the `Scan` branch (187-211), criterion A over N.
5. Encode scan over S (`encode.py:258`), criterion A over S, B over L.

Deferred, with reasons:

- Potts / ProtonPotts / Langevin loops: not read or probed for this spec (ledger honesty). Each has its own
  state shape (Potts couplings are intrinsically O(L^2) per sweep, so neither criterion fits without per-model
  expected exponents); needs its own spec. File one debt item per family at implementation time.
- `conditional_logits.py` candidate/replicate dispatch (`make_axis_dispatch` over `xtrax.tiling`): a
  different dispatcher than `chunked_map`; same helper applies, but not probed. Follow-up.
- Real-decoder scan inside `_chunked_sample_decode` (the per-sample decode): guarded by the AR file and by
  STE/score guards; the seam guard uses a stub decode on purpose (A8) so it tests only the loop plumbing.
- GPU-only fusions (pallas/megakernel): not traceable on CPU; out of scope.

### D4. Per-loop entry point, inputs, budget

All CPU, trace-only (`jax.make_jaxpr`), no execution, no whole-suite runs. Budget per guard test: <= 20 s wall
on this box with `OMP_NUM_THREADS=4`, <= 90 s per file; the three-extent STE trace was the slowest probe
(minutes were not needed; trace at L=128 with a 64-d, 2-layer model). If a trace exceeds budget, drop L=128
and use extents {24, 48, 96}.

| Guard | File (new) | Entry traced | Inputs (reuse) | Extents | Criterion |
|---|---|---|---|---|---|
| tiling | tests/model/test_ligand_tiling_complexity.py | `map_chunks_axis0(y, chunk_size=8, fn=feat._y_edges_coords_to_embed)`; `_multi` with a two-array synthetic fn | `ProteinFeaturesLigand(node_features=128, edge_features=128, k_neighbors=8, ligand_l_chunk=-1)` as in tests/model/test_ligand_feature_tiling.py:44-52; y = `ones((L, 12, 3))` | L 32/64/128 plus 37/74/148 | A |
| STEDecode | tests/inference/decode/test_ste_complexity.py | `STEDecode(inner=ConditionalDecode(model, VmapIterator()), iterations=3, optimizer=adam)` | fixture of tests/inference/decode/test_ste.py:33-105 generalised over L (S=1, untied and tied) | L 32/64/128 | B, trips equal |
| optimize_ste | same file | `make_optimize_sequence_fn(model, stage_set)` with `iterations=3` | same bundle builder | L 32/64/128 | B (while, trips None: use `loop_bodies`, not trip measurement) |
| score_pairs | tests/scoring/test_score_pairs_complexity.py | `make_score_split_fns(model).score_pairs(..., chunk_size=2)` | `_structure` + AR-test `model` fixture; keys (C,), seqs (C,L), masks tiled `1-eye(L)` (K,L,L); `encode_structure` outside the traced fn | P axis: K 5/10/20 (C=2, chunk 2); L 32/64/128 (K=C=2) | A over K, B over L |
| decode seam | tests/host/test_chunked_sample_decode_complexity.py | `_chunked_sample_decode(stub, enc, bundle, None, keys, ORDER_CFG, batch_size=8)` | `_StubDecode`, `ORDER_CFG` from tests/host/test_chunked_sample_decode.py:23-60 | N 16/32/64 | A |
| Scan dispatch | same file | `_dispatch_axis(Scan(), body, xs)` | `body = x @ ones((4,4))` | N 32/64/128 | A |
| encode scan | tests/inference/test_encode_scan_complexity.py | `make_encode_fn(model, use_rolling_state=True)(bundle, key, config)` | `_bundle(length, n_states=S)` lifted via helper or copied from AR test | S 2/4/8 (L=24); L 32/64/128 (S=2) | A over S, B over L |

The AR test's `model`, `_structure`, `_bundle` fixtures (lines 37-122) are needed by three guards. Put them in
`tests/helpers/aminx_small.py` (new) with the SAME definitions, and have the AR test keep its copies (D1:
no behaviour change); fixers must not make the AR test import them in this change.

### D5. Must-fire and must-not-fire per guard

Each guard has a positive-on-clean test and a red-check (negative control) in the same file. The mutation
re-runs the original body over the full extent (`fold_over`) so its cost is the same order as the body (A12:
a small injected einsum is NOT enough and gave ratio 1.001). The mutated function must be marked by a
sentinel counter (trace-time `+= 1`, as test_chunked_sample_decode.py does) asserted > 0 so the mutation
provably ran.

| Guard | Must-not-fire (current code) | Must-fire control | Measured (probe) |
|---|---|---|---|
| tiling | ratios 1.000 at all extents, both fns | same `map_chunks_axis0` with `fn=lambda s: g(s) + 0.0*sum(g(y))` (closure over full y) | clean 1.000; mutant 1.771, 1.871 |
| STEDecode | ratios < 3.0 (2.007, 2.014) | `ConditionalDecode` wrapped so each call is vmapped over L perturbed copies, element 0 returned (subclass of `eqx.Module` delegating `.model`) | mutant 3.995 |
| optimize_ste | ratios < 3.0 (2.007, 2.014) | same wrapper applied through a monkeypatched `score_conditional` in the module namespace | not probed (p4 mutated only STEDecode); fixer must confirm >3.0 or choose a heavier fold |
| score_pairs | A over K: 1.000; B over L: 2.005, 2.010 | `monkeypatch.setattr(score_module, "_score_candidate_kernel", bad)` where `bad` re-runs the original over every mask (`vmap` over masks) | mutant 1.831, 1.908 over K |
| decode seam | 1.000 | monkeypatch `kernel_dispatch.chunked_map` with a wrapper whose per-chunk body also runs `f` over all `xs` | mutant 1.733, 1.846 |
| Scan dispatch | 1.000 | `Scan` body that closes over all `xs` (`x + sum(xs @ w)*0`) | not probed |
| encode scan | A over S 1.000; B over L 2.008, 2.015 | `eqx.tree_at(lambda m: m.encoder, model, FoldEncoder(model.encoder, S))` | mutant 1.845, 1.916 over S |

Threshold sanity: the red check must exceed the threshold with margin (A: >= 1.7 vs 1.5; B: > 3.5 vs 3.0).
If a mutant's measured ratio is within 10% of its threshold, the fixer strengthens the mutation, never lowers
the threshold. Each red-check also asserts the clean/mutant ratio gap, so a guard that stops being able to
fail (a refactor making the mutation a no-op) fails loudly ("the guard cannot fail", as the AR negative control).

### D6. Scoring `chunked_map`: estimand and decision

Reading `_score_pairs_kernel` and `_score_candidate_kernel` (score.py:431-536, A11): pair p = (mask k, sequence
n) maps to ONE `score_from_encoding` call that runs the decoder once over all L positions with the supplied
`(L, L)` ar_mask and the teacher-forced sequence, then takes one NLL. The per-pair work is O(L k), one decoder
pass, not O(L) passes. "Per-element full decode" is therefore the intended estimand: each (mask, sequence)
pair needs its own conditional logits, and the loop over pairs is O(K C) by definition. It is not the AR
bug's shape (that recomputed L rows to keep one per wave). Decision: GUARD (not defer), with criterion A over
the pair axis (work independent of K and C, 1.000) and criterion B over L (linear, 2.005). What the guard
protects against is a regression where a pair re-decodes all masks or candidates (K x C^2 / K^2 C cost),
mutant 1.83-1.91. Caveat A15: no ADR was read; the owner confirms the estimand in section 6 Q2.

## 3. Acceptance criteria

- `tests/helpers/loop_scaling.py` exists; `tests/inference/decode/test_incremental_autoregressive.py` is
  byte-identical to main (git diff empty) and its two guard tests still pass.
- Seven guards (tiling, STEDecode, optimize_ste, score_pairs, decode seam, Scan dispatch, encode scan), each
  with a clean pass and a red-check that fails the guard when the mutation is injected (the assert on
  the mutant must be the guard's own assertion helper, not a try/except).
- Each red-check mutant demonstrably executed (sentinel > 0) and its ratio exceeds its threshold with the
  margin of D5.
- Each guard file runs in <= 90 s on CPU with `OMP_NUM_THREADS=4`.
- A bathos sidecar `.bth.toml` (committed before first run) with `[outcomes]` for the measured clean and
  mutant ratios (Task 0), so the numbers in D5 are re-derivable and the thresholds are not post-hoc.
- `uv run --group lint ruff check` and `ruff format --check` pass on the new files; `ty` is advisory.
- Debt #1981 closed; follow-up debt items filed for each deferred loop in D3.

## 4. Fixer tasks

Each task is one fixer session. All gates are narrow named-file runs, never the whole suite; run them with
`OMP_NUM_THREADS=4 uv run --group test pytest <file>` (heavy runs go to titanix, per the repo's memory).

### Task 0: Pre-register and re-run the probes (all later tasks depend on it)
Create `scripts/spikes/loop_complexity_probe.py` (argparse + logging; trace-only) reproducing the probe
tables of A5-A12 and `.bth.toml` with `[outcomes]` fixed in advance: clean ratios `< 1.5` (A loops) /
`< 3.0` (B loops), mutant ratios `>= 1.7` (A) / `> 3.5` (B), and the optimize_ste and Scan mutants measured
for the first time. Run with `bth`. Outcome: the numbers of D5 (or revised ones) recorded. **Gate**: `bth`
record shows `outcome` evaluated; no outcome fails (a failing outcome changes the threshold in D2 via a
`changed:` ledger edit, not by editing the test). Scope ~200 LOC.

### Task 1: Shared helper
Create `tests/helpers/loop_scaling.py` and `tests/helpers/aminx_small.py` (D1, D4) plus
`tests/helpers/test_loop_scaling_helpers.py`: unit tests on toy jaxprs (a scan with O(1) body: ratio 1.0; a scan
whose body recomputes an (L, d) tensor: ratio ~2; a fixed-trip fori with a quadratic body: ratio 4; path
mismatch across extents raises; a `searchsorted` scan is exempt from A). **Gate**: `uv run --group test pytest
tests/helpers/test_loop_scaling_helpers.py`; AR test file unchanged (`git diff --stat` empty for it). ~250 LOC.

### Task 2: Ligand tiling guard
`tests/model/test_ligand_tiling_complexity.py` per D4/D5, including the non-multiple-of-cs extents. **Gate**:
`pytest tests/model/test_ligand_tiling_complexity.py`. ~120 LOC.

### Task 3: STE guards (STEDecode + optimize_ste)
`tests/inference/decode/test_ste_complexity.py`, criterion B, untied and tied, wrapper mutant per D5; confirm
the optimize_ste mutant exceeds 3.5 (Task 0 measures it first). **Gate**: `pytest
tests/inference/decode/test_ste_complexity.py`. ~180 LOC.

### Task 4: score_pairs guard
`tests/scoring/test_score_pairs_complexity.py`, A over K and C, B over L, fold-over-masks mutant by
monkeypatching `_score_candidate_kernel` (restore via `monkeypatch`, not manual try/finally). Import the
module with `importlib.import_module("aminx.scoring.score")` (the package re-exports a `score` function that
shadows the submodule; see tests/host/test_compile_shape_classes.py:347-350). **Gate**:
`pytest tests/scoring/test_score_pairs_complexity.py`. ~150 LOC.

### Task 5: kernel_dispatch seam + Scan
`tests/host/test_chunked_sample_decode_complexity.py`, both loops, monkeypatched `chunked_map` mutant; use
fresh N values per case (the filter_jit cache would otherwise hide retracing, though `make_jaxpr` itself
retraces; a plain jit-cache hit does not affect `loop_bodies`). **Gate**: `pytest
tests/host/test_chunked_sample_decode_complexity.py`. ~130 LOC.

### Task 6: Encode scan guard
`tests/inference/test_encode_scan_complexity.py`, A over S and B over L, `FoldEncoder` mutant. **Gate**:
`pytest tests/inference/test_encode_scan_complexity.py`. ~120 LOC.

### Task 7 (optional): AR file differential
Only if the owner wants one implementation: show helper-based flagged-loop list == existing
`_quadratic_loops` for force/off/sampling-vmap, then switch. **Gate**: both AR tests plus the
`test_auto_under_sample_vmap_keeps_only_the_incremental_body` cases pass unchanged. Default: skip.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Work proxy is a jaxpr count, not wall time; a fused kernel can change real cost with no jaxpr change | Guards catch structure (the AR bug shape), not constants; one-line statement in each test docstring; keep the AR calibrate profile as the wall-time check |
| xtrax version drift (installed 0.4.1b7 vs pinned 0.4.1b9; cost model constants may change) | Use ratios only; red-checks fail loudly if the proxy changes enough to blind the guard; record xtrax version in Task 0 sidecar |
| Dense `ar_mask` L^2 term pushes criterion-B ratio above 2 at large L (A14) | Extents capped at 128; threshold 3.0; if clean ratio exceeds 2.5 at an extent, document the L^2 term rather than raise the threshold |
| Weak mutation yields a false sense of coverage (A12: ratio 1.001) | D5 mandates fold-shaped mutants, sentinel counters, and the margin rule |
| `while` trips not measurable for `map_chunks_axis0` (fori with traced n_chunks) and `extent_scaling_report` would call it "scaling" | The helper uses `loop_bodies` only and never measures trips; A applies by declared axis |
| Monkeypatching `jax.lax.while_loop` (xtrax trip measurement) is process-global | Helper avoids it entirely (D1) |
| Stub-decode seam guard misses a regression inside the real decode | Covered by the AR, STE and score guards; stated limitation |
| Trace time at L=128 on STE (probes ran in minutes in total, but unsplit) | Budget and fallback extents in D4; time Task 0 and put it in the sidecar |
| Rollback | Test-only change plus one script; revert the files; no production code touched |

## 6. Open questions for the owner

1. Should the AR test file stay untouched (Task 7 skipped)? Recommended: yes, leave it; two implementations
   agree by construction on one shared threshold constant, and the existing file is the reviewed reference.
2. Is one decoder pass per (mask, sequence) pair the intended scoring estimand (D6)? Recommended: yes; guard
   as specified. If an ADR says otherwise, the guard still holds for K and C independence.
3. Criterion B threshold 3.0 and extents 32/64/128: accept, or tighten to 2.5 after Task 0? Recommended:
   3.0 now (clean 2.007-2.014, mutant 3.995); tighten only after measuring at L=256 on titanix.
4. Should `Scan` dispatch be guarded when the planner may never emit it for the sample axis (A9)? Recommended:
   yes, it is ~30 LOC and the branch exists; drop the guard only if the branch is deleted.
5. Defer Potts / ProtonPotts / Langevin as separate debt items? Recommended: yes, one item per family; they
   need per-family expected exponents, not these two criteria.
6. Add `conditional_logits.py` `make_axis_dispatch` loops to scope? Recommended: next ticket; same helper.
7. Where should the pre-registered sidecar live (this repo has no `scripts/loop/`)? Recommended:
   `scripts/spikes/` with a committed `.bth.toml`, run through `bth` per the global rule.
