---
title: "Debt: expand direct reference parity (G3a) to tied lanes, fixed positions, and the full knob grid"
description: "The direct split-vs-reference measurement is the narrowest link in the browser validation chain; what it skips, why, and the cheapest order to close it."
task_id: 260926_browser-export-loop
status: open
created: 260930
---

# Debt — direct reference parity coverage

**Filed as a document because the praxia debt store was unreachable** (`db_unavailable`,
`profile: postgres`, "no connect attempt has been recorded yet" — the MCP server has no
`DATABASE_URL`). This needs re-filing into the tracker once that is fixed; the text below
is written to be pasted in.

**Priority:** P2 · **Category:** testing · **Task:** `260926_browser-export-loop`

**Paths:** `scripts/browser_validation/p07_split_reference_tf.py` (+ `.bth.toml`),
`scripts/browser_validation/p07_knobs_gate.py`,
`tests/parity/test_full_model_parity.py`

## What

G3a (`p07_split_reference_tf.py`, run `ed4f0b77`, **3.822e-05 nats** against a 1e-4 bound)
is the only link in the browser validation chain measured **without an intermediary** —
the exported graphs compared directly to reference ProteinMPNN. It is also by far the
narrowest: **2 graded cases**, untied lanes (`P08@1.0`) on fully-designed structures
(5L33, 6MRR), L=128, ORT-CPU.

## What it skips, and why

Every skip is recorded in the run's `skipped` field rather than hidden.

1. **Every P07 lane, plus 1BC8's P08 lane — non-zero `fixed_mask`.** Graph D takes no
   `chain_mask` input: the split handles fixed positions through Graph F's override during
   *sampling*, not inside the conditional decode, whereas `aminx_conditional_logits` feeds
   `chain_mask` into the inference bundle. Where positions are fixed, these are different
   functions, so the comparison is **ill-posed rather than merely unrun** — the first run
   of this gate compared them anyway and correctly disagreed by whole nats. Closing this
   needs a decision about what the right comparison *is*; likely comparing only at
   `comparison_positions` with `chain_mask` applied upstream of Graph D, but that must be
   derived, not guessed.

2. **Tied lanes (`P09-s`).** Needs the reference-side tie fusion B1 applies via
   `tests.parity.test_full_model_parity._combine_reference_tied_log_probs`. Reproducing a
   second piece of reference machinery was judged out of proportion at the time; it is the
   obvious next increment.

3. **`tie_lattice_L96`.** Excluded because the knobs gate itself excludes it from B1
   (`p07_knobs_gate.py:1934-1943`, which prefers 5L33 and skips this fixture by name). It
   is a *synthetic* lattice built to produce tied k-NN distances, so **aminx JAX versus
   reference has never been measured on it either**. Two earlier G3a runs scored ~2.4 nats
   there; that supports no split-specific conclusion, and reading it as an export defect
   would have been wrong. Establishing JAX-vs-reference on deliberately-tied geometry is a
   prerequisite — and independently interesting given the sort-stability work.

4. **Coverage breadth.** 2 of 7 knobs, one seed each, one bucket. The knobs gate covers
   the full 288-cell grid for JAX-vs-ONNX; the *reference* link does not.

## Why it matters

This is a **strength-of-evidence gap, not a correctness hole.** The composed claim already
covers tied and fixed cases end to end:

| Link | Measured | Run |
| :--- | :--- | :--- |
| reference PyTorch → aminx JAX, teacher-forced | 4.482e-05 nats | `fd80f81d` (B1) |
| aminx JAX → split, tokens exact | 56/56, 32/32, 8/8, 4/4 | `d2a06073`, `ed2a2617`, `d6be02ea`, `8a3bb1d6` |

But a composed claim is only as tight as its weaker link, and the direct measurement is
the one an external reviewer trusts most. Right now it is the thinnest part of an
otherwise broad chain. The intended reader is the advisor.

## Proposed increments, cheapest first

1. Widen to **all untied knobs and seeds** on the existing fully-designed fixtures — no
   new machinery required.
2. Add **tied lanes** by importing `_combine_reference_tied_log_probs`.
3. Derive and implement the **fixed-position** comparison.
4. Extend to **L256**.
5. Settle **JAX-vs-reference on `tie_lattice_L96`** as a separate question.

Each increment is cheap to *run*: teacher-forced scoring needs **one conditional decoder
pass**, not the autoregressive loop, because the sequence is known up front — the existing
gate grades its two cases in seconds.

**Keep the 1e-4 bound inherited from B1** rather than choosing a new one. Inheriting it is
what makes the figure comparable to aminx's own JAX-vs-reference number; a bound chosen
for this gate would not be.
