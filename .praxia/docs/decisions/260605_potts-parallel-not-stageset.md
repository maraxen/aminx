---
title: PottsModel is a parallel model family, NOT a StageSet consumer
decision_id: 260605_potts-parallel-not-stageset
date: 2026-06-05
status: Accepted
decision_type: architectural
relates_to: 260605_integration-architecture-for-mistypotts
---

## Status: Accepted 2026-06-05

**Amended by:** [`260929_pottsmpnn-lasermpnn-family-drivers`](260929_pottsmpnn-lasermpnn-family-drivers.md) — scope-narrowing: this decision governs `aminx.potts.*` only.

This ADR documents the architectural decision to implement PottsModel as a standalone parallel model family, independent from the aminx.inference StageSet pipeline.

## Context: Integration of PottsTRWStructureModel into aminx.potts.PottsModel

The aminx project integrates multiple sequence design paradigms:
- **MPNN family**: Autoregressive decoding via StageSet (decode_step → per-position logits)
- **Potts model**: Global probabilistic inference via tree-reweighted message passing (TRW)

When integrating Potts inference into aminx, three architectural options emerged:

1. **Standalone parallel model** ← CHOSEN
2. Decode-step adapter pattern (Option II)
3. EncoderOutput widening (Option III)
4. Product-of-Probabilities currency unification (Option IV)

## Decision: Parallel Model Family, NOT StageSet Consumer

**PottsModel is a parallel model family.** It:
- Defines its own inference pathway: `infer_params(coords, mask, ...) → PottsParams(marginals, h, J, rho, W)`
- Lives in `aminx.potts.{model, poe, sampling, designer}` with no imports from `aminx.inference.decode`, `host.plan`, `types.stages`, or `inference.logits`
- Operates on its own energy currency: Potts unary (h) and pairwise (J) potentials with x2 scale factor
- Supports ensemble via Product-of-Experts (PoeModel) for multi-backbone combination
- Owns its own calibration, sampling, and design orchestration (designer.py, future)

**Enforcement:** The following are **forbidden**:
- `aminx.potts.model` importing from `aminx.inference.decode`, `host.plan`, `types.stages`, `inference.logits`
- `aminx.potts.poe` importing from the above
- `aminx.potts.sampling` importing from the above
- Exception: `aminx.potts.designer` is permitted cross-pipeline imports for orchestration (future)

This is enforced via static analysis (#1304) and documented in module docstrings.

## Rejected Alternatives

### Option II: Decode-Step Adapter Pattern

**Concept:** Adapt `decode_step(node_f, edge_f, nei, mask) → logits(L, V)` to output Potts parameters.

**Why rejected:**
- **Currency mismatch:** `decode_step` is designed for autoregressive per-position decoding. It outputs logits (L, V) where L is the number of positions and V is vocabulary size (20 amino acids).
  - Potts inference is **one-shot global**, not per-position sequential.
  - Potts energy operates on the full structure at once, not position-by-position.

- **Parameter shape mismatch:** `decode_step` produces (L, V) logits; Potts requires:
  - h: unary potentials (N, 20) — structure-aware
  - J: pairwise potentials (N, N, 20, 20) — node-pair dependent, not one-hot logits
  - rho: tree-reweighting parameters (N, N) — TRW-specific, meaningless in autoregressive context

- **Phase mismatch:** TRW is pre-computation (once per structure); autoregressive decode is per-token streaming. Forcing TRW through the per-position decode interface breaks the inference model.

- **Integration complexity:** Wrapping TRW in a decode_step shape would require fake token-at-a-time framing, breaking TRW's global structure-awareness.

### Option III: EncoderOutput Widening

**Concept:** Extend `EncoderOutput` (bundles.py:359) to carry both MPNN logits and Potts parameters, making Potts visible downstream via the unified pipeline.

**Why rejected:**
- **Blast radius:** The blast radius of widening `EncoderOutput` (bundles.py:359) spans the full StageSet pipeline — every downstream consumer would require changes:
  - Inference pipeline (host.plan, types.stages)
  - Result aggregation and sinks
  - Serialization and benchmarking
  - Cross-project parity validation
  
  Adding Potts fields would force all downstream code to handle unused fields or conditional logic.

- **Coupling:** Every StageSet consumer (compose, compress, aggregate, sink) would need to know about Potts-specific fields, even when Potts is not in use.

- **Versioning and schema drift:** EncoderOutput is load-bearing for checkpointing and parity. Adding fields breaks checkpoint compatibility and requires schema migration tooling.

- **Separation of concerns:** Potts is a **different inference paradigm**, not an enhancement of the MPNN pipeline. Mixing them in a shared output type violates single responsibility.

### Option IV: ProductOfProbabilities for PoE

**Concept:** Use a unified currency (logits or probabilities) to combine multiple inference models via `p_ensemble = ∏ p_backbone`.

**Why rejected:**
- **Currency incompatibility:** 
  - MPNN pipeline output: logits (S, L, V) where S is samples, L is length, V is vocabulary size (20 amino acids)
  - Potts inference output: J tensor (N, N, 20, 20) and marginal probabilities
  
  There is no direct conversion: J is structure-dependent coupling strength; logits are position-wise preferences. They do not compose via simple element-wise product.

- **Graph structure dependency:** Potts J depends on the k-NN graph (edges, edge features). MPNN logits are graph-agnostic. Combining them requires explicit graph awareness downstream, breaking modularity.

- **Energy vs. logits:** Potts parameters are energies (log-unnormalized); MPNN logits are model outputs. Energy-space products (`exp(e1 + e2)`) and logit-space products differ fundamentally.

- **No backward compatibility:** Existing PoE code (poe.py) operates on identical Potts backbones via energy summation. Widening to MPNN would require new PoE logic and break existing semantics.

## Enforcement

### Static Analysis (#1304)

A forbidden-import checker will flag:
```
ERROR: aminx/potts/model.py imports from aminx.inference.decode (forbidden)
ERROR: aminx/potts/poe.py imports from aminx.inference.logits (forbidden)
```

Allowed only in `aminx.potts.designer` for orchestration-layer imports.

### Module Docstrings

Both `model.py` and `designer.py` reference this ADR in their headers:
```python
"""Potts model with differentiable TRW inference.

Architecture: PottsModel is a parallel model family (NOT a StageSet consumer).
See ADR 260605_potts-parallel-not-stageset for design rationale.
"""
```

## Design Invariants & Risks

### Risk 1: Alphabet Index Collision (X / Gap)

**Issue:** Potts uses canonical MPNN alphabet (20 standard + X for gap). Index 20 is ambiguous: gap or unknown?

**Mitigation:** Static `POTTS_TO_MPNN_ALPHABET_MAP` (identity) and explicit docstring in model.py. Synthetic validation test (test_potts_correctness.py) verifies alphabet round-trip.

### Risk 2: h/J Scale Factor (x2 from Directed-Slot Convention)

**Issue:** PottsMPNN convention counts each pairwise edge twice (directed slot framing). h and J carry implicit x2 scale.

**Mitigation:**
- Documented in model.py:6–9 and PottsParams docstring
- Reference function `etab_to_dense_h_j_w` (mistypotts) for scale derivation
- Synthetic test: verify `log_prob` energy matches mistypotts reference under x2 scale

### Risk 3: k_neighbors Baked Into Checkpoint

**Issue:** Graph connectivity is part of the model checkpoint metadata, not a constructor argument. User cannot change k after load.

**Mitigation:** k_neighbors is read-only (eqx.field(static=True)). Attempting to override raises ValueError. Documented in model.py:72–74.

### Risk 4: Fori Loop OOM in Training

**Issue:** TRW with trw_loop='fori' materializes all intermediate states in reverse-mode autodiff, causing OOM on large graphs.

**Mitigation:**
- PottsRunSpec.__post_init__ enforces: `if training=True and trw_loop='fori': raise ValueError`
- training_default() constructor uses trw_loop='scan' with checkpoint_trw_step=True
- Documented in spec.py:85–90

### Risk 5: caliby_path=None Default

**Issue:** None is a valid value (identity calibration), not missing-value sentinel.

**Mitigation:** Explicit None handling in __post_init__. Documented in spec.py:70. No auto-fill; None is identity, required for inference.

### Risk 6: Method Naming: infer_params / sample / score

**Issue:** MPNN pipeline uses encode/decode; Potts uses infer_params/sample/score.

**Mitigation:** Deliberate naming distinction (not encode/decode) to signal different inference paradigm. Method names enforce architectural separation.

## Consequences

**Design Integrity:**
- Potts and MPNN are separately deployable inference families
- No cross-contamination of inference assumptions
- Easier to test and verify each paradigm independently

**Integration:**
- Multi-state design: PoeModel orchestrates N independent Potts backbones
- Designer layer (future) will coordinate Potts + MPNN for hybrid workflows
- Result aggregation (sink) handles both families in parallel, not sequentially

**Maintenance:**
- Potts module evolves independently (TRW improvements, calibration updates)
- MPNN pipeline unaffected by Potts changes
- Clear forbidden boundary prevents architectural creep

---

**Decision made:** 2026-06-05 by multistate-potts architecture review
**Enforced:** Static import checker (#1304), module docstrings, test suite
