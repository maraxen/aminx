---
title: "Advisor handoff message — browser ProteinMPNN sampling (final, post-release)"
description: "The Slack message handed to the advisor for v0.2.0a3, with the claim-to-run-id mapping behind every figure it quotes."
task_id: 260926_browser-export-loop
status: sent
created: 260930
---

# Advisor handoff message — final

Reflects the state after integration: work is on `main` at `d1210e4a`, tagged `v0.2.0a3`,
release published 2026-09-30T17:13:24Z with the eight split graphs + `MANIFEST.json` attached.

## The message

> Hey Sergey — ProteinMPNN sampling now runs fully in the browser, validated end to end against
> your implementation. It's on `main` in `maraxen/aminx`, tagged `v0.2.0a3`:
> https://github.com/maraxen/aminx/releases/tag/v0.2.0a3
>
> **Start here:** `docs/browser_integration.html` — self-contained, opens in a browser, and every
> number in it cites the run that produced it.
>
> The part I most wanted to show you is that it's anchored to *your* code, not just
> self-consistent with mine. Teacher-forced per-position log-probs from the exported graphs vs
> reference ProteinMPNN (LigandMPNN `26ec57ac`, `proteinmpnn_v_48_020`) agree to **3.8e-05 nats**
> — essentially where our own JAX port sits against you (4.5e-05), so the export adds no
> measurable error on top of the port. Downstream of that, exported tokens match JAX **exactly**:
> 288/288 cells across the full RunSpec knob grid, and again through the shipping JS in headless
> Chromium. I compare tokens exactly rather than within a tolerance, because a changed token is a
> changed design and a log-prob bound can't see one.
>
> **~6.8 s per 128-residue design** in-page with COOP/COEP and 4 threads; 15.8 s single-threaded,
> with log-probs bit-identical between the two. L=256 is ~27 s — cost grows roughly quadratically
> in length, which is worth knowing before offering the bigger bucket to anyone. Those timings
> come from a harness that first had to detect a planted 250 ms delay to within 0.1%; an earlier
> benchmark here never established that control and I threw its numbers away.
>
> Export is four small ONNX graphs — encoder / wave schedule / decoder step / fuse-and-sample —
> with the autoregressive loop driven from JavaScript. That's 6.7 MB against 31.5 MB for the
> single-graph version at L=256, and it keeps all control flow out of the per-step path, which
> should help anyone taking it to WebGPU.
>
> **Two things I haven't done.** WebGPU is untested — not deferred on principle, I just couldn't
> get an adapter on my machine (WSL2), so ONNX Runtime was never actually asked for one. The known
> obstacle is that ONNX mandates int64 `TopK` indices for the k-NN sort and the WebGPU EP doesn't
> support them; in the split that lives only in the encoder, which runs once per structure rather
> than once per step. And free-sampling distributional agreement (recovery, perplexity) against
> the reference isn't measured yet — the exact-token results above are teacher-forced and
> greedy-path, not distributional. The direct reference comparison is also scoped to untied lanes
> on fully-designed structures at L=128; tied and fixed-position cases are covered by the
> exact-token rows instead, which is a two-link argument rather than a direct one. Both gaps are
> written up in the repo rather than left implicit.
>
> Plain ES modules, no bundler, copy-paste examples for both the single-graph and split paths in
> the guide.
>
> **For your agents — you have my explicit permission to merge to `main` and push.** The graphs
> and `MANIFEST.json` are attached to the release above and also tracked in git-LFS at
> `release/browser/`. To confirm the bytes you serve are the bytes that were validated:
>
> ```
> python scripts/browser_validation/p07_split_export.py --verify-manifest MANIFEST.json
> ```
>
> `MANIFEST.json` records each file's sha256, byte size, input shapes and dtypes alongside the
> commit and checkpoint id. Rebuild with the same script, `--buckets 128 256`.

## Claim → run id

Every figure quoted above, and where it came from. Full table in
`docs/browser_integration.md`.

| Claim in the message | Value | Run |
| :--- | :--- | :--- |
| exported graphs vs reference ProteinMPNN, teacher-forced | 3.822e-05 nats (bound 1e-4) | `ed4f0b77` |
| aminx JAX vs reference, teacher-forced (the comparison anchor) | 4.482e-05 nats | `fd80f81d` (B1) |
| monolith vs JAX + ORT-Web, full knob grid | tokens exact, 288/288 | knobs gate |
| split vs monolith + JAX, ORT-CPU | exact, 56/56 (L128), 32/32 (L256) | `d2a06073` |
| shipping JS loop, ORT-Web wasm under Node | exact, 56/56 | `ed2a2617` |
| shipping JS loop, headless Chromium | exact, 8/8 (L128), 4/4 (L256) | `d6be02ea` |
| 4 threads + COOP/COEP, log-probs bit-identical | exact | `8a3bb1d6` |
| 6.80 s / 15.77 s / 27.18 s medians, timer control detected | — | browser bench |
| 6.7 MB split vs 31.5 MB monolith at L256 | — | export manifest |

## What the message deliberately does not claim

- It does not call the reference agreement broad. Two graded cases, untied lanes, L=128 — the
  narrowest link in the chain, and named as such. Coverage debt is filed in
  `260930_reference-parity-coverage-debt.md`.
- It does not present exact-token agreement as distributional agreement.
- It does not say WebGPU is blocked, only untested, and says why.
