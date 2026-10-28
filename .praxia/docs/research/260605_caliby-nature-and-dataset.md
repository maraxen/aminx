---
title: Caliby nature and dataset investigation
research_id: 260605_caliby-nature-and-dataset
task_id: 260605_multistate-potts
date: 260605
status: superseded_by_261027_caliby-family-port-spec
backlog_item: 1298
---

# Caliby Nature and Dataset Investigation

> [!WARNING] **HISTORICAL MISATTRIBUTION CORRECTION (Debt #3072)**
> This 2026-06-05 document misidentified **"Caliby"** as an internal shorthand ("calibration y") for a post-hoc TRW marginal calibration sidecar.
> **Upstream Caliby** (`ProteinDesignLab/caliby`, Shuai, Widatalla, Huang et al., *bioRxiv* `10.1101/2025.09.30.679633` / Zenodo `10.5281/zenodo.18644985`) is a **3D-coordinate Potts model + flow-matching / EDM sidechain diffusion packer** created by the Protein Design Lab (Po-Ssu Huang Lab, Stanford).
> The internal Potts marginal calibration utility remains in `src/aminx/potts/calibration.py`, whereas the true upstream Caliby model family is implemented in `src/aminx/families/caliby/` per `.praxia/docs/specs/261027_caliby-family-port-spec.md`.


## Summary

Caliby (calibration y) is a learned correction sidecar for TRW marginals in Potts inference. This document investigates its nature, computational requirement, dataset dependencies, and implementation plan for aminx integration.

## Findings

### 1. Caliby Definition and Purpose

**What is caliby?**
- Post-hoc correction applied to TRW marginal probabilities after full TRW inference completes
- Learned module (eqx.Module subclass) that maps marginals → corrected marginals
- Captures systematic biases in TRW marginals relative to target distributions (e.g., native sequences, experimental folding)
- NOT a modification to the underlying Potts energy — marginals are post-processed at inference time

**Why needed?**
- TRW computes exact marginals of the Potts energy, but the energy itself may have systematic biases
- Re-tuning the Potts energy (h, J coupling terms) is expensive and risks overfitting
- Post-hoc marginal correction is cheaper and more interpretable: identifies which marginals are systematically over/under-confident

**Design decision from ADR 260605_integration-architecture-for-mistypotts.md:**
- Caliby is a **learned sidecar** — stored separately from potts_<id>.eqx.zst
- Versioned independently (caliby_<id>.eqx.zst)
- Can be updated without retraining PottsModel
- Default behavior: caliby_path=None → IdentityCalibration (no-op), enabling safe rollout

### 2. Computational Nature

**Inference-time cost:**
- Marginals passed through eqx.Module.__call__ (one or two array operations)
- Adds <1% overhead to inference (negligible compared to TRW backprop through TRW itself)
- Must be JAX-compatible and eqx.filter_jit-safe

**Training-time cost:**
- Full recapture requires:
  1. Load PottsModel (weights_path)
  2. Forward pass on calibration dataset (TRW inference for each structure) — expensive
  3. Compute gap between native marginals and model marginals
  4. Learn correction as a simple additive/multiplicative adjustment (likely 1-2 gradient steps)
  5. Save LearnedCalibration checkpoint

**Recapture frequency:**
- One-shot per PottsModel version + calibration dataset version
- Can run offline; not part of main inference loop
- Good candidate for bathos-tracked cluster job with pre-flight local gates

### 3. Dataset Requirement

**What data is needed?**
- A calibration dataset containing:
  - Protein structures (PDB files or coordinate arrays)
  - Native sequences (ground truth)
  - Optionally: experimental folding data, stability measurements, etc.

**Dataset location in mistypotts:**
- **Investigation status:** NOT FOUND in current mistypotts codebase
- Grep for "caliby\|calibration" in mistypotts/src yields no matches
- ROADMAP mentions "calibration diagnostics" (c_i metrics) but not a full calibration dataset
- F4 implementation summary covers per-residue diagnostics, not caliby recapture pipeline
- **Conclusion:** Caliby dataset does not yet exist in mistypotts; specification is a prerequisite

**Candidate sources:**
- Could be derived from training split of ProteinMPNN data (if available)
- Could be CASP-format structures with native sequences
- Could be experimental folding datasets (e.g., ProTherm, FoldX benchmarks)
- **TBD:** Awaiting decision from mistypotts project on which dataset to use

### 4. Implementation Plan

#### Phase A: Specification (Blocked)
- **Owner:** mistypotts project
- **Task:** Define caliby training dataset (format, location, size)
- **Blockers:** None for aminx side; waiting for mistypotts upstream
- **Deliverable:** mistypotts dataset location + format doc

#### Phase B: Scaffold (Complete)
- **Status:** ✓ DONE (Task 260605, Track J)
- **Deliverables:**
  - `src/aminx/potts/calibration.py`: CalibrationModule protocol + IdentityCalibration + LearnedCalibration
  - `scripts/recapture/caliby_recapture.py`: Argparse scaffold + --dry-run gate + NotImplementedError on real run
  - `scripts/recapture/caliby_recapture.bth.toml`: Bathos sidecar template
  - Module supports three correction modes: additive, multiplicative, learned_scale
  - Type selection at load time (no None-branch inside jit)

#### Phase C: Local Gates (Deferred)
- **Required:** caliby dataset available locally or remotely accessible
- **Task:** Implement L1 (dry-run local validation) + L2 (smoke test on tiny subset)
- **Owner:** aminx fixer after dataset available

#### Phase D: Cluster Submission (Deferred)
- **Required:** Caliby dataset on cluster storage
- **Task:** Implement L3 (sbatch submission with reduced budget, verify completion)
- **Owner:** aminx fixer (follows CLUSTER.md rules)

#### Phase E: Integration (Deferred)
- **Task:** Hardwire caliby_recapture into Track K (runner.py loads caliby via load_calibration)
- **Blocker:** Phase A (dataset spec) + Phase C (gates pass)

### 5. Risk Assessment

**P1 — Blocked on dataset:**
- Caliby training cannot proceed without upstream mistypotts dataset specification
- Mitigation: IdentityCalibration is safe default; can rollout aminx.potts without caliby
- Timeline: Awaiting mistypotts decision (external dependency)

**P2 — Dataset availability on cluster:**
- Recapture is cluster-only if dataset is large (>10GB) or requires HPC compute
- Mitigation: Design for both local testing (tiny synthetic subset) and cluster (full dataset)
- Timeline: Dependent on Phase A outcome

**P3 — Caliby generalization:**
- Caliby trained on one dataset may not transfer to another (overfitting risk)
- Mitigation: Train on diverse dataset (multiple backbones, multiple datasets); validate via cross-dataset testing
- Timeline: Phase A (dataset specification) should address this

### 6. Integration with Track J (1298)

**Completed:**
1. ✓ Investigated caliby nature (learned sidecar, not energy term)
2. ✓ Designed CalibrationModule protocol and two implementations
3. ✓ Scaffolded recapture script with --dry-run and dataset-absent guard
4. ✓ Created research note (this document)

**Deferred:**
- Full recapture training loop (awaits dataset)
- Cluster gates L2/L3 (awaits dataset availability)
- Integration into Track K runner (dependent on caliby checkpoint availability)

## Cluster Gates (Placeholder)

If caliby recapture runs on cluster, follow `/using-myxcel` and `CLUSTER.md`:

### L1 — Local Dry-Run
```bash
uv run python scripts/recapture/caliby_recapture.py --dry-run --out /dev/null
```
Expected: Exit 0, validates argparse + output path writability.

### L2 — Local Smoke Test
```bash
# Requires tiny synthetic caliby dataset (TBD format)
uv run python scripts/recapture/caliby_recapture.py \
  --dataset-path /tmp/caliby_synthetic.parquet \
  --potts-model path/to/potts_model.eqx.zst \
  --out /tmp/caliby_synthetic.eqx.zst
```
Expected: Completes in <60s on CPU, produces valid eqx.zst.

### L3 — Cluster Smoke Test
```bash
sbatch --array=0-0 scripts/cluster/caliby_recapture_smoke.sbatch
```
Expected: Single-job array completes, verifies cluster path mapping + cluster-side eqx.zst save.

## Next Steps

1. **mistypotts project:** Specify caliby training dataset (location, format, size)
2. **aminx fixer:** Once dataset available, implement Phase C (L2 local smoke test) + Phase E (integrate into Track K)
3. **Optional:** Add cross-dataset validation test to verify caliby generalization

## References

- ADR 260605_potts-parallel-not-stageset.md
- Integration spec 260605_integration-architecture-for-mistypotts.md (Approach A+D, caliby as sidecar)
- Task 260605_multistate-potts Track J (#1298)
- Sprint plan `.praxia/sprint_plans/260605_potts-integration.toml`
- `/using-myxcel` skill for cluster submission protocol
- `CLUSTER.md` for mandatory local gates before cluster submission
