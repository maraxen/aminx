---
title: 'Stage 2 of debt #2371: the tiling plan is computed three times and used zero times'
description: Measured evidence that BatchPlanner says Vmap+Bucket for the sample axes while execution is a Python loop, plus the T2.GATE blocker on deleting aminx's duplicate strategy types
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261005'
supersedes: ''
backlog_ids: ''
---
# Stage 2 of debt #2371: the tiling plan is computed three times and used zero times

**User direction (261005), verbatim:** "aminx should not have a separte tiling strategy, we should
be using xtrax natively. we might need to rebase on current main." And, on sequencing:
"type-swap, dispatch to Vmap or ChunkedMap should be automatically happen under the hood, but
vectorization should be native."

The merge half of that direction is done (`bc3f1950`, xtrax 0.4.0a11 + main's dual-name
`ChunkedMap` dispatch fix). This document is about the other half, and it exists because the
work turned out to be a different shape than "swap the types".

## 1. What is actually happening (measured, not inferred)

There are **three** places a tiling decision is made on the sample path, and the decision reaches
execution at **none** of them.

| # | Site | What it does | What reaches execution |
| --- | --- | --- | --- |
| 1 | `families/potts_mpnn/sample_host.py:386` | `BatchPlanner().plan(specs)` — return value **not assigned** | nothing; `return specs` on the next line |
| 2 | `host/family_runner.py:90-94` `_bind_declared_axes` | hardcodes `SafeMap(tile=...)` for **every** axis, ignoring the plan; builds an iterator and **discards it** (`-> None`) | nothing; one caller at `:296` ignores it too |
| 3 | `families/potts_mpnn/sample_host.py:545` | `for offset in range(chunk_count)` | **this** — a Python loop |

So the plan is computed, overridden by a hardcoded strategy, and then neither is used; the real
execution is sequential Python. `_bind_declared_axes`'s docstring ("Bind each declared axis
through xtrax. Drivers must not vmap these.") describes a binding that does not happen — the call
is effectively an assertion that dispatch would not raise.

### What the planner would say

Measured by running the real `xtrax.tiling.BatchPlanner` (0.4.0a11) on the exact specs
`sample_axes` builds. This is a throwaway spike, not a graded run — do not cite these as
measured performance, they are the planner's *decisions*, which are deterministic:

| num_samples | n_temp | n_struct | `structures` | `samples` | `temperatures` |
| --- | --- | --- | --- | --- | --- |
| 8 | 1 | 1 | **Bucket** | **Vmap** | **Vmap** |
| 50 | 1 | 1 | **Bucket** | **Vmap** | **Vmap** |
| 1000 | 1 | 1 | **Bucket** | **Vmap** | **Vmap** |
| 8 | 3 | 4 | **Bucket** | **Vmap** | **Vmap** |

Planner's own reasoning strings: `samples`/`temperatures` → `cardinality <= batch_size → Vmap`;
`structures` → `bucket_boundaries=(100,200,400,800,1200) → Bucket (host-side padding)`.

**This answers the user's earlier question — "are we leveraging vmapping (+ bucketing, padding,
etc.)?" — with a no, and shows the intent was already there.** The plan asks for full
vectorization on samples and host-side bucketing on structures. `_bind_declared_axes` converts
both to `SafeMap`, and the Python loop then discards even that. The measured linear ~2.04 s per
sample for Potts is exactly what a sequential loop predicts.

## 2. The catch: `default_batch_size` is set to `cardinality`

`sample_axes` declares `AxisSpec(name="samples", cardinality=n_samples,
default_batch_size=n_samples)`. The planner's rule is `cardinality <= batch_size → Vmap`, so with
the two set equal **the planner can never choose `ChunkedMap` for this axis** — it returns `Vmap`
at every size, including 1000.

That matters for the user's "dispatch to Vmap or ChunkedMap should automatically happen under the
hood": as declared today, the automatic choice is not a choice. Simply honouring the plan would
vmap 1000 samples at once and is a plausible OOM, which is the hazard the planner exists to
prevent. The real memory bounding currently lives somewhere else entirely — the chunk windows
from `resolve_chunk_size` in `_chunk_windows`.

So `default_batch_size` has to come from the memory budget (or from the same source
`resolve_chunk_size` uses) rather than from `cardinality`, **before** honouring the plan is safe.
This is the load-bearing finding: the fix is not "stop discarding the plan", it is "make the plan
answerable first, then stop discarding it".

## 3. The blocker on deleting aminx's duplicate types

The literal reading of the direction — delete `aminx/tiling/strategy.py`'s and `iterator.py`'s
`Vmap`/`SafeMap`/`Scan`/`DedupGather` and use xtrax's natively — collides with a **standing** gate.

`tiling/dispatch.py`'s module docstring states that the aminx-native `make_axis_dispatch` has zero
production callers but is kept deliberately as the reference implementation that T2.GATE's Gate
Measurement Protocol compares `make_axis_dispatch_via_xtrax` against, that the protocol is
explicitly standing ("re-run on production shapes, not a one-shot at flip", per
`.praxia/docs/specs/260611_aminx-xtrax-refactor.md`), and closes with: "Do not delete without
retiring that gate first."

Verified — all four named artifacts exist, and six files reference the aminx-native entry point:

- `tests/tiling/test_dispatch_via_xtrax_parity.py`
- `tests/tiling/test_t2_4_xtrax_dispatch_compat.py`
- `tests/tiling/test_t2_gate_bitforbit_golden.py`
- `tests/tiling/test_dispatch.py`
- `tests/tiling/test_safemap_deprecation.py`
- `scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py`

**You cannot delete one arm of an A/B gate while the gate is standing.** Retiring T2.GATE is a
decision with a pre-registered protocol behind it, so it is the user's, not something to do
silently. Note main already landed stage 1 — `tiling/_deprecation.py`, caller-attributed
`DeprecationWarning`s on aminx's own `SafeMap`/`SafeMapIterator` — so the deprecation path is
open; only the deletion is gated.

## 4. Proposed sequencing

1. **Re-declare `default_batch_size`** from the memory budget / `resolve_chunk_size`, not from
   `cardinality`, so Vmap-vs-ChunkedMap becomes a real decision. No behaviour change yet.
2. **Honour the plan**: `_bind_declared_axes` takes the `BatchPlan` and returns the per-axis
   iterators instead of hardcoding `SafeMap` and discarding them. `sampling/multistate_poe.py:255`
   is already the correct pattern in-tree (`iterator = make_axis_dispatch_via_xtrax(strategy,
   axis=N_SAMPLES.name)`) — copy it rather than inventing one.
3. **Make vectorization native**: replace the `for offset in range(chunk_count)` loop with the
   dispatched iterator. This is the step that changes numerics (vmap reassociates), so it
   re-measures every affected slug and must not ride along with anything else.
4. **Let `structures` be `Bucket`**, which the planner already asks for — that is the bucketing
   and padding half of the user's question.
5. **Only then**, and only with the user's go-ahead on retiring T2.GATE, delete the duplicate
   types and `_strategy_to_xtrax`.

L-DRV R1 bans `vmap`/`pmap`/`shard_map` in `families/**` and `host/family_*.py`, so step 3's
vectorization has to arrive *through* xtrax's dispatch rather than a hand-rolled `jax.vmap` —
which is the same thing the user asked for ("vectorization should be native"). Check whether the
R1 lint text names the aminx wrapper by name; if it does, it needs updating with step 5.

## 5. Why none of this landed tonight

Steps 1-4 all touch `src/aminx/`, which under THE FREEZE invalidates every ledger row globally and
has to land in the single scoped re-wave. Step 3 additionally changes sampling numerics while a
Potts distributional confirmatory run is in flight pinned to `5c625b00`. Doing a
numerics-changing refactor of the sample path unattended, against a live run and a frozen ledger,
trades a day of gate re-measurement for an overnight head start. The measurement above is the part
that was worth doing now, because it is what makes steps 1-4 designable at all — and it corrects
the assumption that stage 2 is a type-swap.
