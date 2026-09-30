---
title: "P07 split export: encoder graph + per-step decoder graph, loop in JS"
description: "Split the monolithic P07 ONNX export into an encoder graph and a per-step decoder graph so the AR loop and its branch decision move to JavaScript, enabling one implementation across wasm (CPU) and WebGPU."
task_id: 260926_browser-export-loop
status: draft
created: 260929
---

# P07 split export (T13)

**Status: DRAFT / PRE-REGISTRATION. No numbers in this document. Nothing here has run.**

Goal: a browser ProteinMPNN sampler whose implementation is the *same* on CPU (wasm) and
GPU (WebGPU), with parity evidence strong enough for an external reviewer to trust and
adapt. Motivated by the advisor handoff (`docs/browser_integration.md`).

## 1. Why the monolith blocks the GPU path

All claims in this section were read directly out of the code at
`feat/t11d-session-release`; each carries its anchor.

The exported P07 graph keeps the entire autoregressive loop inside ONNX, and it keeps
**two complete decoder implementations** plus a runtime branch selecting between them:

- `src/aminx/inference/decode/autoregressive.py:809` —
  `jax.lax.cond(can_increment, run_incremental, run_full)`.
- `src/aminx/inference/decode/mode.py:80` — `incremental: Literal["auto","off","force"] = "auto"`.
- `src/aminx/export/wrappers.py:402` — comment *"Default config: incremental cache when it
  is exact, full recompute otherwise"*; `make_p07_sample` constructs
  `make_decode_fn(model, mode=AutoregressiveMode(), strategy=Vmap())` and sets **no**
  `autoregressive_config`, so the `"auto"` default applies.
- `mode.py` docstring states the cost of each arm: `"off"` re-runs the decoder over all L
  positions every wave — **O(L²k) per sequence**; the incremental path decodes only the
  wave's positions against a per-layer cache — **O(Lk) per sequence**, "with identical
  tokens and logits whenever its preconditions hold".
- The wave axis is `lax.scan` (`autoregressive.py:618` via `wave_iterator`);
  `inference_only` defaults `False` (`mode.py`) and P07 does not override it, so the
  `lax.while_loop` arm at `autoregressive.py:612-617` is not what gets exported.

This is observable in the produced artifact, not only in the source. ORT's graph-load
warnings on the current `.onnx` name initializers such as:

```
cond_then_0/scan_loop_0/cond_then_0/cond_then_0/cond_then_0/scan_loop_0/cond_then_0/...
```

— nested `If` inside `Scan` inside `If`, several levels deep.

ORT generally runs control-flow nodes (`If`, `Scan`, `Loop`) on CPU. On the WebGPU EP that
forces a device transfer at every step, which is why the WebGPU path was deferred rather
than measured (ODQ-B3).

### 1.1 The load-bearing observation

`can_increment` is computed at `autoregressive.py:800-806`:

```
consistent     = ~any(reads & (wave_nbr > wave_row))
identity_frame = all(state_position_map == arange(L))
fits_slab      = all(diff(wave_start) <= slab)
can_increment  = consistent & identity_frame & fits_slab
```

Every term derives from the **wave bundle**, which is built from `decoding_order` and
`tie_group_map` — both already runtime inputs, and both already constructed in JavaScript
by `browser/aminx-sampler/runspec_core.mjs`. The in-code comment says so: *"The predicate
depends on the bundle only."*

**Therefore JS can evaluate `can_increment` before calling the model at all.** The split is
not merely "move the loop out"; it is "move the branch decision out", so we export *one*
decoder path instead of two-plus-an-`If`. That removes the dominant control-flow obstacle
to the WebGPU EP.

### 1.2 Why jax2onnx, and not jax-js or xtrax/IREE

Settled by prior research —
`xtrax:.praxia/docs/research/260914_browser-inference-routes-jaxjs-jax2onnx.md` (branch
`docs/browser-inference-routes`) — recorded here because this spec is otherwise silent on
route choice and the question recurs.

- **jax-js is not a converter.** It is a from-scratch reimplementation of the JAX/NumPy API
  in JavaScript. It cannot ingest a Python JAX model: no StableHLO, no `jax.export`, no
  serialized-program path. The only two ways in are rewriting the model in JS or loading an
  **ONNX** file via `@jax-js/onnx`. So jax-js is a *runtime* downstream of jax2onnx, not an
  alternative to it — but it IS a second candidate WebGPU consumer of the very files this
  spec produces (see G2).
- **xtrax/IREE cannot reach WebGPU today.** aminx already uses `xtrax.export`
  (`check_export_safety`, `compile_for_target`, `NATIVE`/`WASM32`) on the layer-b IREE
  route. IREE's target list is `cuda llvm-cpu metal-spirv rocm vmvx vmvx-inline
  vulkan-spirv` — no `webgpu`, blocked on iree#24650 (open since 2026-06-29). This is why
  the spec designates IREE the *fallback* route (ODQ-1), and it is unchanged.
- **jax2onnx takes Equinox directly**, so there is no reimplementation step, and it ships
  an explicit `export_mode="web"`.

### 1.3 The split isolates the project's single biggest index risk

This is an argument FOR the split that §1.1 does not make, and it may matter more.

Per that research, `lax.sort` is the **sole** index-producing primitive in the traced graph
(`jax.lax.top_k` is banned repo-wide; the surviving calls are eager host code). ONNX
*mandates* `tensor(int64)` for `TopK`'s index output — not the exporter's choice — and
jax2onnx lowers `lax.sort` to `TopK` + `GatherElements`, stamping indices INT64. **ORT-Web's
WebGPU EP does not support int64.** The consequence is not a load failure but per-node EP
fallback: index nodes execute on CPU while matmuls run on GPU, with device copies at each
boundary — and, in that research's words, *"the WebGPU run and the wasm run may not agree
on tie order even for the same file."*

There are two sort sites, and the split treats them very differently:

| Site | Purpose | Under the split |
| :--- | :--- | :--- |
| `model/features.py:90` | k-NN neighbour selection | stays in **Graph E** — once per structure |
| `wrappers.py:316-321` (`wave_from_decoding_order`) | wave schedule from decoding order | **leaves the graph entirely** → JS |

The research calls the decode-order site "the more consequential of the two: it determines
the sequence in which residues are generated, so a permutation there changes the whole
design, not one neighbour list." The split removes it from ONNX altogether and confines the
remaining one to a graph that runs **once per structure**. **Graph D — the per-step hot
loop — contains no sort and no TopK at all**, so the per-step path carries no int64 index
tensor and nothing for the WebGPU EP to reject.

That also means the JS gains responsibility for wave-schedule tie order, which is why G1a
compares **tokens exactly** and not merely log-probs within a bound: the research flags
jax2onnx's multi-key sort (an LSD radix pass emitting one `TopK` per key, correct only if
each pass is stable) as "the single most likely way this route produces a green check over
wrong output", and notes it is *invisible to float parity*. Exact token comparison is the
instrument that sees it.

## 2. The cut

The encoder/decoder boundary in `wrappers.py` is already a clean line at 425-455.

**Graph E (encoder), exported once per bucket.**
Inputs: `coords [L,4,3] f32`, `mask [L] f32`, `residue_index [L] i32`, `chain_index [L] i32`.
Body: `compute_backbone_coordinates` → `compute_backbone_distance` → `select_neighbors`
(k=48, `row_chunk=EXPORT_TOP_K_ROW_CHUNK`) → `compute_radial_basis` →
`features.forward_edge_stages` → `model.encoder`.
Outputs: `node_features [L,H]`, `edge_features [L,K,H]`, `neighbor_indices [L,K] i32`,
`mask [L]` — i.e. exactly the `EncoderOutput` built at `wrappers.py:450-455`, minus the
`[None, ...]` batch axis.
Properties: no RNG, no scan, no cond. Runs once per structure.

**Graph D (decoder step), exported once per bucket, called once per wave.**
From `_kernel.py:16-83` (`_decode_one_step` → `decoder.call_conditional`) and
`_kernel.py:86-110` (`_project_logits`).
Inputs: the four Graph-E outputs, plus `ar_mask`, plus `sequence_oh [L,21] f32`.
Outputs: `logits [L,21] f32`.
Properties: no RNG at inference (`inference=True` disables dropout), no scan, no cond.

**`ar_mask` shape — resolved.** `_kernel.py`'s docstring says `(L,)` and
`decoder.call_conditional`'s says `(L, L)`; they disagree, and `_decode_one_step` passes the
array straight through, so `call_conditional` is authoritative. Confirmed against the code
rather than the docstrings: `autoregressive.py:680` does
`jnp.take_along_axis(cond.ar_mask, nbr_all, axis=2) -> (S, L, K)`, so the stored array is
`(S, L, L)`. **`_kernel.py`'s `(L,)` docstring is wrong** and should be fixed separately.

**`ar_mask` is constant across waves — this is load-bearing.** The decoder reads
`cond.ar_mask` (`autoregressive.py:516`), and `cond` is the conditioning bundle built once
by `p07_bundle` (`wrappers.py:456-467`) *outside* the scan. It encodes the decoding order,
which does not change during sampling. So of Graph D's six inputs, **five are loop
invariants** (`node_features`, `edge_features`, `neighbor_indices`, `mask`, `ar_mask`) and
only `sequence_oh [L,21]` changes per step.

Two consequences:

1. JS builds the `(L,L)` mask **once per sample**, not once per wave — O(L²) total, not
   O(L³).
2. Per-step the only tensor crossing the boundary is `sequence_oh` (256×21 f32 ≈ 21 KB at
   L=256), with `logits [L,21]` coming back. Under ORT IO-binding the five invariants can
   be bound once and left resident on the device. This is what makes Split-A plausible on
   the WebGPU EP despite W dispatches, and it should be measured before assuming Split-B's
   cache is needed at all.

A further reduction may be available: `autoregressive.py:219` documents
`ar_neighbors : (L, K) ar_mask gathered at each row's neighbours`, suggesting the decoder
only ever consumes the gathered form. If `pack_conditional_decoder_static_edges` does that
gather internally, Graph D could take the pre-gathered `[L,K]` (K=48) instead of `[L,L]`.
**Unverified** — it needs a read of `pack_conditional_decoder_static_edges`, and it is an
optimisation, not a prerequisite.

**What JavaScript owns after the split.** Already present in `runspec_core.mjs`: PRNG
(SplitMix64), Gumbel noise, decoding-order construction (`shuffle`/`fixedFirstShuffle`/
`biasedShuffle`), `bias_AA`/`omit_AA` and per-residue variants, `fixedMask`/`fixedTokens`,
`tieGroupMap`, temperature. **New JS work required:**

1. Wave-schedule construction — currently `wave_from_decoding_order` (`wrappers.py:298-337`)
   producing `{group_ids, group_positions, group_valid, position_valid}`.
2. `can_increment` evaluation (§1.1), and with it the choice of variant (§3).
3. Per-wave `ar_mask` construction.
4. One-hot of the running sequence (currently `_one_hot_tokens`, `autoregressive.py:504`).
5. The fuse-and-sample step — `_fuse_and_sample` (`autoregressive.py:99-189`): tied-group
   logsumexp averaging, `argmax(avg/temperature + gumbel_noise[group_id])`, fixed-position
   override, stored-vs-sampling logits distinction.
6. Sequence scatter update (currently `autoregressive.py:584-586`).

Items 5 and 6 are where numerical parity is most at risk, because they are the only new JS
that does arithmetic on model outputs rather than on RunSpec inputs. They get their own
parity gate (§4, G1a).

## 3. Two variants, and the one real performance question

**Split-A (stateless).** Graph D recomputes over all L positions each wave; no cache
crosses the boundary. Total cost O(L²k) — matches the `"off"` arm. Simplest to export and
to reason about; slowest.

**Split-B (cached).** Graph D additionally takes and returns the per-layer cache
(`(n_cache, L, H)`, with `n_cache = max(len(layers)-1, 1)`, `autoregressive.py:639,794`).
Total cost O(Lk). For L=256 the cache is roughly `2 × 256 × 128` f32 ≈ 256 KB crossing the
boundary **per wave**, which on WebGPU is a device round-trip per step unless the tensor is
kept on-device via ORT's IO-binding / GPU-resident tensors.

Split-A is the correctness baseline and ships first. Split-B is the performance variant and
is only worth building if G4 (§4) shows Split-A is too slow *and* a device-resident cache
measurably fixes it. **We do not assume Split-B is faster in the browser** — the O(Lk) vs
O(L²k) arithmetic advantage is a JAX-side measurement and says nothing about per-call ORT
dispatch overhead across W calls, which may dominate at these sizes.

## 4. Pre-registered validation

### 4.0 The chain of custody

**Agreement with aminx's JAX is necessary but not sufficient.** On its own it proves only
that our ONNX matches our JAX — self-consistency. If the JAX port itself carried a defect,
every downstream gate would agree with it perfectly and still be wrong, and an external
reviewer has no reason to accept it. The evidence must therefore terminate at the
**reference implementation**, not at us:

```
reference ProteinMPNN (dauparas/LigandMPNN @ 26ec57ac, PyTorch)
      |  R1  deterministic, teacher-forced per-position log-probs
      v
aminx JAX  (make_p07_sample, checkpoint proteinmpnn_v_48_020)
      |  R2  same graph, converted
      v
ONNX monolith  (ORT-CPU, ORT-Web wasm)
      |  R3  the split
      v
Graph E + JS loop + Graph D
      |  R4  same graphs, different EP
      v
wasm (CPU)  and  WebGPU
```

Every link needs its own evidence, and **R1 is the one that anchors the rest**. R1 is
already the layer-(a) exact tier's job, and the knobs gate already carries a teacher-forced
reference comparison (its `b1_max_abs_nats <= 1e-4` check). The split adds R3; it must not
be allowed to silently replace R1.

Two different kinds of reference comparison, and the difference matters:

- **Deterministic (R1, the strong one).** Teacher-forced: feed both implementations the
  *same* sequence and decoding order and compare per-position log-probs. No sampling, no
  statistics, tight bound. This is what actually demonstrates "the same model".
- **Distributional (G3b).** Sequence recovery and perplexity from free sampling. Necessary
  because the deterministic check cannot see a sampling-path defect (Gumbel, temperature,
  tie fusion, fixed-position override), but it is a weaker instrument and needs a sized
  control.

The reportable claim is the *composition* of the chain, and it is only as strong as its
weakest link. Both must be on the same checkpoint (`proteinmpnn_v_48_020`) with the
reference pinned at `26ec57ac976ade5379920dbd43c7f97a91cf82de`.

### 4.1 Gates

Each gate below gets a tracked script and a `.bth.toml` sidecar **committed before the run**,
with a negative control. No gate's threshold may be relaxed after seeing a number. Verify
every run by its record (cool-tier parquet), never by exit code.

**G0 — feasibility (binary, blocks everything).** Does `jax2onnx` convert Graph E and
Graph D standalone, outside the enclosing `lax.scan`? Pass = both `.onnx` files load in
onnxruntime and produce finite outputs on one fixture. Fail = the split is not available by
this route and we say so. *This is the first thing to run and it is cheap.*

**G1a — JS step arithmetic vs JAX.** The new JS from §2 items 1-6, fed fixed synthetic
logits, against the JAX functions it replaces. Tokens exact; fused log-probs within the
pre-registered bound. Negative control: perturb the tied-group averaging and confirm the
comparison fires.

**G1b — split vs monolith vs JAX, ORT-CPU.** Full sampler: Graph E + JS loop + Graph D,
against the existing monolithic P07 and against JAX, on held-out fixtures, across the
RunSpec knob grid already used by the knobs gate. Tokens exact; log-probs within the
pre-registered bound (the knobs gate's re-preregistered 2e-4 is the reference point; the
split's own bound is set in its sidecar before running, not inherited silently).

**G1c — ORT-Web wasm.** G1b repeated in Chromium. Negative control: the existing
`--no-isolation` arm.

**G2 — WebGPU, two candidate runtimes.** Same fixtures, tokens exact vs wasm, log-probs
within bound. A capability probe runs first and its failure is a legitimate recorded
outcome, not a retry trigger. Per ODQ-B3, no WebGPU number is quoted anywhere until this
gate passes on real GPU hardware.

Two runtimes consume the *same* Graph E / Graph D files and both are in scope:

- **ORT-Web WebGPU EP** (`executionProviders:["webgpu"]`). Expected to partition: Graph E's
  k-NN `TopK` emits int64 indices the EP cannot take, so those nodes fall back to CPU with
  device copies. Graph D should partition cleanly, having no sort (§1.3).
- **`@jax-js/onnx`** — jax-js's own WebGPU runtime, reading the same ONNX. Worth measuring
  precisely because it is not bound to ORT's int64 gap, and because Graph D's shape (five
  loop-invariant inputs, one changing, §2) suits a runtime that can keep tensors resident.

Report both, or report one and say plainly that the other was not run. **Tokens must be
exact against the wasm reference on either runtime**; a tie-order difference between
backends on the same file is the specific failure §1.3 predicts, and it is invisible to a
log-prob bound.

**G3a — deterministic vs the reference implementation (link R1, the anchor).**
Teacher-forced per-position log-probs from the **split** pipeline against reference
ProteinMPNN/LigandMPNN@26ec57ac, same checkpoint, same fed sequence and decoding order, on
the held-out set. Tight pre-registered bound in nats (the knobs gate's existing
teacher-forced reference check runs at 1e-4; the split's bound is set in its own sidecar,
not inherited silently). Negative control: a planted perturbation of the reference-facing
comparison, sized during calibration, that must trip it.

This gate is what makes the whole chain mean anything, and it is deliberately **not**
routed through aminx's JAX: it compares the artifact we ship to the implementation the
field trusts. Running it on the split pipeline rather than only on the monolith is the
point — otherwise R3 is unanchored.

**G3b — distributional vs the reference.** Sequence recovery and perplexity from free
sampling, against the same reference. Needed because G3a is teacher-forced and therefore
structurally blind to a defect in the sampling path itself — Gumbel draw, temperature,
tied-group fusion, fixed-position override — which is exactly the code §2 moves into new
JavaScript. Statistical, not bitwise; needs its own sized control, and a margin <= 0 is
recorded as unsized rather than as a pass.

**G4 — performance.** Wall time per design, session-create time, peak memory; Split-A on
wasm and (if G2 passes) WebGPU. **The timer must first pass a planted-delay control** — the
earlier benchmark failed exactly this and so produced no trustworthy timing number. No
timing is reported before that control passes.

## 5. Risks

- **`lax.cond` conversion.** Even for Graph D alone the surrounding code uses `lax.cond`
  pervasively for masking/gating (`autoregressive.py:594, 778-782, 791`). Graph D as cut in
  §2 should sit *below* those, but this is unverified until G0.
- **Sort ops.** `wave_from_decoding_order` uses `lax.sort`/`argsort`
  (`wrappers.py:316-321`); under the split this moves to JS entirely, which removes the
  risk from the graph rather than mitigating it.
- **`select_neighbors` / custom `top_k`.** Stays in Graph E, runs once. `jax.lax.top_k` is
  banned repo-wide; the custom implementation lives at `src/aminx/model/features.py:48-73`.
  The int64 `TopK` WebGPU-EP fallback risk (ODQ-B3) applies to Graph E only, once per
  structure, not per step.
- **Scatter/gather on the WebGPU EP.** `GatherND`/`ScatterND` commonly fall back to CPU.
  Under Split-A the per-step scatter moves to JS; under Split-B the cache update does not.
- **Per-call dispatch overhead.** W ORT calls instead of 1. Unmeasured, and plausibly the
  dominant cost at L≤256. G4 is what decides this; §3 must not be read as a prediction.
- **Two implementations to keep in parity.** Shipping the split alongside the monolith means
  two things that can drift. G1b is the standing guard, and should run in CI on the fixture
  subset rather than only once.

## 6. Sequencing

G0 first — it is cheap, it is binary, and a failure changes the whole plan. Then Graph E +
Graph D export with a sha256 manifest, then G1a (pure JS, no model), then G1b/G1c, then
**G3a** (the reference anchor — before any performance work, because a fast wrong sampler
is worthless), then G3b, then G4, then G2 last since it needs GPU hardware and gates only
its own claims.

## 7. Reporting

The deliverable is a **parity table organised by the §4.0 chain**, one row per link, each
row carrying: what was compared, the pre-registered bound, the measured value, n, whether
its negative control fired, and the bathos run id. A reviewer must be able to see which
link is weakest without reading prose.

| Link | Comparison | Bound | Measured | n | Control | Run id |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1/G3a | reference PyTorch ↔ split, teacher-forced log-probs | prereg | — | — | — | — |
| R2 | aminx JAX ↔ ONNX monolith | 2e-4 | — | — | — | — |
| R3/G1b | monolith ↔ split (ORT-CPU) | prereg | — | — | — | — |
| R3/G1c | split ORT-CPU ↔ ORT-Web wasm | prereg | — | — | — | — |
| R4/G2 | wasm ↔ WebGPU | prereg | — | — | — | — |
| G3b | reference ↔ split, recovery/perplexity | prereg | — | — | — | — |

Rules for the table, all of which exist to keep it honest:

1. **Every cell is filled from a run record** (cool-tier parquet), never from console text
   or an exit code. A cell whose run cannot be resolved stays empty and is reported empty.
2. **A gate that did not run is named as not-run**, not omitted. `docs/browser_integration.md`
   already uses explicit **pending** markers; keep that convention.
3. **A `ctrl_blind` outcome is not a pass and is not a failure** — it means the instrument
   was insensitive, and the row reports that rather than a number.
4. **No WebGPU number appears anywhere until G2 passes** (ODQ-B3).
5. **No timing number appears until G4's planted-delay timer control passes.** The earlier
   benchmark failed exactly this, which is why there is currently no trustworthy timing
   figure anywhere in the handoff doc.
6. The composed claim is only as strong as its weakest link, and is stated that way rather
   than by quoting the best row.

Findings are promoted to a bathos **ledger claim** (`bth claim`) rather than cited from a
dated document, because dated docs are an append-only log and go stale. The advisor-facing
artifacts — `docs/browser_integration.md` and its shareable HTML page — then cite the claim
and carry the table, replacing their current **pending** rows.

Nothing in this document is measured. Every number arrives through a sidecar committed
before its run.
