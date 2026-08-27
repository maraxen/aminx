---
title: xtrax Engine.fit Feasibility vs aminx Trainer
description: Feasibility assessment evaluating xtrax.engine.Engine.fit() as a replacement for aminx training/trainer.py, identifying 5 architectural capability gaps and recommending retaining the local trainer.
status: accepted
task_id: 260827_runspec-scaffolding-remediation-spec
date: '260827'
confidence: high
sources: src/aminx/training/trainer.py, xtrax/engine/engine.py
---

# `xtrax.engine.Engine.fit()` Feasibility vs `aminx.training.trainer.py`

## 1. Executive Summary

This assessment formally evaluates whether `xtrax.engine.Engine.fit()` can replace `aminx.training.trainer.train()`, resolving the unassessed finding from `.praxia/docs/specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md:216-223`.

**Verdict: `Engine.fit()` is NOT a drop-in replacement for `trainer.train()`.**

There are five major architectural capability gaps between `trainer.py` and `Engine.fit()`. Three of these gaps require upstream modifications to `xtrax` before any migration can be considered viable. Because two gaps directly impact checkpoint durability and training convergence/early termination, adopting `Engine.fit()` today would introduce severe correctness regressions.

**Recommendation:** Retain `src/aminx/training/trainer.py` locally. File upstream feature requests with `xtrax` for synchronous step callbacks and training termination protocols. Re-evaluate if and when those upstream capabilities land.

---

## 2. Detailed Capability Gap Analysis

| # | Requirement | `src/aminx/training/trainer.py` | `xtrax.engine.Engine` (`xtrax/engine/engine.py`) | Bridgeable in `aminx` Alone? | Upstream `xtrax` Prerequisite |
|---|---|---|---|---|---|
| **1** | **Step Function Contract** | `train_step` accepts **19 positional arguments** (`trainer.py:772-792`). | `trainer.step(state, batch) -> (state, metrics)` (`engine.py:34,132`). | **Yes** — Mechanical adapter (`TrainStepLike`) closing over `optimizer`, `spec`, `noise_schedule`, `compute_dtype`. | None. |
| **2** | **Checkpoint Cadence** | Saves every `spec.checkpoint_every` **steps** (`trainer.py:852-853`), plus a dedicated `permanent_manager` for `spec.save_at_epochs` (`trainer.py:856-858`). | Single `checkpoint_dir` saved **per-epoch only** (`engine.py:150-151`). | **No.** Step-level checkpointing would require `on_step_end` hook, but `Engine` executes callbacks asynchronously via `BoundedCallbackHandler(max_concurrent=4)` (`engine.py:107,135-140`), providing no write-ordering or durability guarantees. Dual-manager checkpointing has no representation. | Synchronous callback execution mode or explicit step-based checkpoint cadence parameter in `Engine`. |
| **3** | **Mid-Epoch Evaluation** | Evaluates every `spec.eval_every` steps inside the active epoch batch loop (`trainer.py:808-838`). | `Engine.eval()` is a separate top-level method; `fit()` never invokes mid-epoch evaluation (`engine.py:160`). | **Partially** — A callback could close over `val_loader`, but suffers from the same async race conditions as #2. | Native mid-epoch evaluation cadence option in `Engine.fit()`. |
| **4** | **Early Stopping** | Integrated patience counter and early-exit `break` (`trainer.py:840-850`). | No early-stop protocol: callback return values are ignored, and no termination exceptions are caught (`engine.py:126-156`). | **No.** Only expressible by throwing an unhandled exception to abort execution; while `fit()`'s `finally` block runs (`engine.py:153-156`), this is control-flow abuse, not a supported protocol. | Explicit `StopTraining` exception or boolean return value from `on_epoch_end`/`on_step_end` callbacks. |
| **5** | **Data Contract** | `create_protein_dataset` iterables yielding attribute-accessible batches (`batch.coordinates`, `trainer.py:776`). | `train_iter()` / `eval_iter()` (`engine.py:41-43`); `eval()` assumes mapping-style dict batches (`batch["inputs"]`, `engine.py:204-205`) when `loss_fn` is provided. | **Yes** — Thin `DataIterLike` wrapper adapter; mapping assumption is bypassed by omitting `loss_fn`. | None. |

---

## 3. Non-Issues and Shared Foundations

The following aspects were verified to be non-blocking and compatible between both systems:
- **Mixed Precision:** `setup_mixed_precision` (`trainer.py:728`) is configured once before the training loop begins and does not conflict with `Engine`.
- **State Currency:** `ResumableState` (`trainer.py:795-801`) matches the pytree state abstraction expected by `Engine` (`engine.py:65`).
- **Post-Training Test Loop:** The standalone test evaluation loop (`trainer.py:862-934`) operates outside `fit()`'s scope in either architecture.

---

## 4. Incidental Finding: JIT Cache Invalidation inside Epoch Loop

During this assessment, a potential optimization defect in `src/aminx/training/trainer.py` was identified:
- Lines `760-762` construct `eqx.filter_jit(train_step)` and `eqx.filter_jit(eval_step)` **inside** the `for epoch in range(...)` loop (`trainer.py:755`).
- Constructing fresh `filter_jit` wrapper instances per epoch creates new function objects, potentially triggering unnecessary recompilations at the start of each epoch instead of reusing the compilation cache across epochs.
- Hoisting these `filter_jit` instantiations above line `755` is a clean, low-risk fix, but must be verified empirically with `JAX_LOG_COMPILES=1` before filing as a separate backlog item.

---

## 5. Conclusion & Action Items

1. **Retain Local Trainer:** `src/aminx/training/trainer.py` remains the active training orchestration module for `aminx`.
2. **Upstream Requests:** Track two capability requests for `xtrax.engine.Engine`:
   - Feature: Synchronous step callback handling and configurable step-based checkpoint cadence.
   - Feature: Standardized early stopping / training termination protocol.
3. **Closure:** Finding E / unassessed Engine feasibility from `260707_xtrax-migration-gap-audit-runspec-scaffolding.md` is fully resolved and closed.


