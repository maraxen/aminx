---
title: "Overnight autonomous decisions — browser export/validation"
description: "Decisions taken without a human gate during the 260929 overnight autonomous loop, with enough context to backtrack each one."
task_id: 260926_browser-export-loop
status: active
created: 260929
---

# Overnight decisions — 260929

Autonomy granted by the user: *"15 minutes please keep driving the browser inference and
validation forward overnight autnonomusly."* Cron `cfca917b`, every 15 min.

Each entry records what was decided, why, and what would reverse it.

---

## D1 — Split the export into FOUR graphs, not two graphs plus a hand-written JS port

**Decided:** 260929, ~18:45 local. Question was put to the user but they moved to
autonomous operation without answering, so it is taken here.

**Context.** The original split design (spec `260929_p07-split-export.md` §2) cut P07 into
Graph E (encoder) + Graph D (decoder step) and moved everything else into JavaScript:
wave-schedule construction, `can_increment`, `ar_mask`, one-hot, the fuse-and-sample step
(`_fuse_and_sample`, including `TieGroupProductOfExperts`), and the sequence scatter.

Reading `_fuse_and_sample` (`inference/decode/autoregressive.py:99-189`) closely showed it
is **pure arithmetic with no scan and no cond** — the `if gumbel_noise is None` branch
resolves at trace time when noise is supplied, which is exactly how P07 runs it. So it is
exportable, and so is the wave schedule.

**Decision.** Four graphs:

| Graph | Runs | Contains |
| :--- | :--- | :--- |
| E encoder | once / structure | k-NN sort → the one int64 `TopK` |
| W wave schedule + ar_mask | once / sample | the other sort, kept in validated ONNX |
| D decoder step | per wave | no sort, no TopK |
| F fuse-and-sample | per wave | PoE fusion, tied averaging, Gumbel argmax, fixed override |

JavaScript keeps orchestration only: hold the sequence, one-hot it, slice per-wave inputs,
call graphs in order, scatter the result.

**Why.** The deliverable's purpose is evidence an external reviewer trusts. Hand-porting
product-of-experts fusion and a multi-key sort into JS creates new numerics that must then
be validated from scratch — and the xtrax route research
(`260914_browser-inference-routes-jaxjs-jax2onnx.md` §5) names exactly that multi-key sort
as "the single most likely way this route produces a green check over wrong output", and
notes it is *invisible to float parity*. Keeping the arithmetic in graphs already covered
by the JAX and reference comparisons shrinks the new hand-written surface to index
bookkeeping.

**Cost accepted.** Graph W retains an int64 `TopK`, which ORT-Web's WebGPU EP cannot take,
so it partitions to CPU. That is **once per sample**, not once per wave, and Graph D — the
hot loop — stays sort-free either way. Judged a good trade against hand-porting the
riskiest function in the design.

**What reverses this.** If G2 measures the Graph W partition as a real cost on WebGPU,
moving W into JS is a contained change, and G1a is already the guard that would catch a
tie-order divergence introduced by doing so. Revisit then, with a number rather than a
prediction.

---

## D2 — G0b filed as a NEW gate rather than extending the passed G0

**Decided:** 260929. Graph W and Graph F feasibility gets its own script and sidecar
instead of widening `p07_split_feasibility.{py,bth.toml}`.

**Why.** G0 has a recorded `pass` (run `8daaa978`) under a specific sidecar hash. Editing
that sidecar to cover two more graphs would retroactively change what that pass meant.
A separate gate leaves the existing record immutable and independently citable.

---
