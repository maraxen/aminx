---
session_id: runspec-xtrax-unified
topic: RunSpec unification bundled with aminx→xtrax refactor — three-layer config model, planner topology, and full inference surface coverage
task_type: architectural
parent_epic: 1541
related_spec: 260611_aminx-xtrax-refactor.md
winner: Three-layer model (dataclass façade + RunSpec PyTree + PlannerTopology builder) parallel to xtrax T0–T1; blocks T4.1 only at RS-2; ExecutionProfile feeds T2.5b axis injection rather than duplicating BatchPlanner
created_at: 2026-06-11T00:00:00+00:00
adversarial_review: spec-challenger + spec-defender cycle 2026-06-11
---

# RunSpec unification (addendum to xtrax refactor epic #1541)

This spec **extends** [`260611_aminx-xtrax-refactor.md`](260611_aminx-xtrax-refactor.md). It does not replace the xtrax vertical-slice spine (T0→T5). It defines how aminx's run entry point becomes a **single, testable configuration contract** that survives the T4.1 MOVE/SPLIT of generic host/plan/sinks into xtrax.

**Coordinated program, not one sprint:** foundations sprint covers T0–T1.4 + RS-1/RS-4/RS-5 (Cursor subagent orchestration). G1 waived; cluster smoke deferred. T4 and RS-6+ land after G3.

---

## Problem frame

### Fixed (from recon 2026-06-11)

- Six specification families exist today (`RunSpecification` + 4 task subclasses + `TrainingSpecification` + parallel `PottsRunSpec`).
- `build_run_spec()` builds an `RunSpec` PyTree on every spec init, but host code **almost never reads it** (2 call sites: `streaming.py`, `training/trainer.py`).
- `aminx.run` re-exports **tensor-level** `sample`/`score` while README documents **spec-driven** usage — a pre-existing bug.
- `BatchPlanner` accepts `carries` and `dedup_specs` but `make_sampling_planner` never passes them.
- `use_unified_driver` defaults differ: `SamplingSpecification=True`, `kernel_dispatch getattr=False`.
- Campaign workers use **full** `spec_json` manifest rows, not portable RunSpec JSON.

### Goal

One authoritative composed config (`RunSpec` + `PlannerTopology`) that:

1. Covers all inference surfaces (sample, score, inspect, jacobian, campaign/grid).
2. Feeds the inference planner and axis primitives without unnecessary JIT recompiles.
3. Partitions cleanly for T4.1: **generic** sub-configs → xtrax; **protein** sub-configs stay in aminx.

---

## Target architecture

```
User / CLI / spec_json (full round-trip)
       ↓
*Specification dataclass façade  (protein + task fields; one minor version)
       ↓ build_run_spec()  [single builder authority]
       ↓
RunSpec (eqx.Module)
  ├── generic (future xtrax.run.*): IOConfig, ResourceConfig, BatchingConfig, PrecisionConfig
  ├── protein (aminx): MultistateConfig, LigandConfig, TiedPositionsConfig, GridLineageConfig, AveragingConfig
  └── planner: PlannerTopology  [NEW — aminx builder, feeds T2.5b injection]
       ↓
prep (loader kwargs) → make_inference_plan → make_sampling_planner → host.runner
```

### PlannerTopology (renamed from ExecutionProfile)

**Not a second BatchPlanner.** PlannerTopology is the aminx-side **builder** that derives axis cardinalities, `unified_driver`, `use_rolling_state`, optional `carries[]` / `dedup_specs[]`, `bucket_policy`, and **`_HETEROGENEOUS_AXIS_NAMES`** (the heterogeneous axis-name set from `tiling/carry.py:21`, today `{"n_states","n_structures"}`) from `SamplingSpecification` (and shared base fields).

- **T2.5b (xtrax):** library seam — caller supplies `AxisSpec[]` + heterogeneous set as parameters.
- **PlannerTopology (aminx):** maps spec fields → those injection parameters for `make_sampling_planner` (and, post-T2.5b, the injected planner).

Data flow: `spec → build_run_spec() → run_spec.planner → make_sampling_planner → BatchPlanner.plan()`.

`topology_hash`: SHA256 of canonical JSON (sorted keys) over the planner's static fields (`axis cardinalities`, `unified_driver`, `use_rolling_state`, `carries[]`, `dedup_specs[]`, `bucket_policy`, `_HETEROGENEOUS_AXIS_NAMES`); used as a monitoring tripwire alongside the parent's JIT recompile fixture (not a substitute for `eqx.field(static=True)` at filter_jit boundaries).

---

## Surface coverage matrix

| Surface | Spec class | Runner | Full JSON | Portable JSON | RunSpec slice |
|---------|------------|--------|-----------|---------------|---------------|
| Sample | `SamplingSpecification` | `host.runner.sample` | ✓ | subset only | io, batching, grid, ligand, averaging, planner |
| Score | `ScoringSpecification` | `host.runner.score` | ✓ | subset | batching, averaging, planner (partial) |
| Inspect | `InspectionSpecification` | `host.runner.inspect` | ✓ | subset (export broken today; RS-8) | io |
| Jacobian | `JacobianSpecification` | `host.runner.jacobian` | ✓ | subset (portable coverage **DEFERRED**) | batching |
| Campaign/grid | `SamplingSpecification` grid fields | `host.campaign` | ✓ manifest | **not used** | grid |
| Train | `TrainingSpecification` | `training.train` | partial | precision | precision |
| Potts | `PottsRunSpec` | `potts.runner` | partial | N/A | optional `IOConfig` only |
| Tensor API | — | `aminx.sampling` / `aminx.scoring` | — | — | bypasses RunSpec |

### Wire-format contract (post challenger review)

| Format | Use case | grid/ligand |
|--------|----------|-------------|
| `spec_json` (full dataclass) | CLI emit, **campaign manifest rows** | Round-trip |
| `portable` v2 | Generic cluster slice (io/resource/multistate/precision) | Placeholder defaults — **not for campaign** |
| `portable` v3 (optional, RS-8) | Scheduler handoff if needed | Explicit fields |

**Guard (RS-8):** `run_spec_portable_to_dict` must **raise** `ValueError` when serializing a spec with `grid.grid_mode=True` or `ligand.model_family='ligandmpnn'` until v3 exists (the lossy boundary — v2 silently drops these fields today). `run_spec_portable_from_dict` must reject unknown/lossy top-level keys instead of ignoring them. *(Superseded 2026-08-27 by `260827_runspec-subconfig-membership-rule.md` / WS-A: `RunSpec.grid`/`.ligand` deleted as dead scaffolding; export guard retired in favor of strict unknown-key rejection on import boundary).*

---

## Sub-config partition (T4.1 MOVE boundary)

| Sub-config | Owner today | T4.1 destination |
|------------|-------------|------------------|
| `IOConfig`, `ResourceConfig`, `BatchingConfig`, `PrecisionConfig` | aminx `run/spec.py` | xtrax.run (generic) |
| `MultistateConfig`, `LigandConfig`, `TiedPositionsConfig`, `GridLineageConfig`, `AveragingConfig` | aminx | **stay in aminx** |
| `PlannerTopology` | aminx (new) | aminx protein axis names + injection; calls xtrax BatchPlanner |

Potts may optionally import generic `IOConfig` type; it does **not** subclass `RunSpec` (ADR: parallel model family). After T4.1, generic sub-config types are **re-exported from `aminx.run`** so Potts and RS track code never `import xtrax` directly.

---

## Backlog track RS (DAG)

| ID | Task | Blocks | Sprint |
|----|------|--------|--------|
| **1620** RS-1 | Inventory flat-field reads in host/* | RS-2 | 1 (foundations) |
| **1621** RS-2 | PlannerTopology + topology_hash + tests | **T4.1 (#1561)**; feeds **T2.5b** | 2 |
| **1622** RS-3 | Wire carries/dedup into make_sampling_planner | — | 2+ (after RS-2, T2.2) |
| **1623** RS-4 | Fix aminx.run exports + deprecation shim | — | 1 |
| **1624** RS-5 | Unify use_unified_driver default | — | 1 |
| **1625** RS-6 | Phased host migration + lint | T4.1 | 2+ (after RS-2) |
| **1626** RS-7 | Scoring InferencePlan + gap triage | — | 2+ |
| **1627** RS-8 | Export Inspection + portable guard | — | 2+ (optional) |

**Hard rule:** zero `import xtrax` in RS track items.

**T4.1 (#1561) deps:** G3 (#1559) **and** RS-2 (#1621).

---

## Acceptance criteria

### AC-RS-1a (RS-1) — Inventory

**Given** the host inference path, **When** RS-1 completes, **Then** a committed table maps every `getattr(spec, …)` / `spec.<field>` read in `host/{plan,prep,runner,streaming,kernel_dispatch}.py` to either `run_spec.<subconfig>` or "protein-only façade".

### AC-RS-2 (RS-2) — PlannerTopology

**Given** a `SamplingSpecification` with `batch_size=4`, `num_samples=100`, `temperature=[0.1, 0.5]`, `backbone_noise=[0.0, 0.1]`, **When** `build_run_spec(spec)` runs, **Then** `spec.run_spec.planner` exposes axis cardinalities matching `make_sampling_planner` today, includes `_HETEROGENEOUS_AXIS_NAMES` sourced from `tiling/carry.py:21` (not hardcoded in xtrax), `unified_driver` mirrors `spec.use_unified_driver` (default `True`), and `topology_hash` (SHA256 of canonical JSON over the static planner fields listed above) is stable across two builds with identical spec. **Post-T2.5b:** parity test asserts injected planner receives the same injection parameters PlannerTopology produces.

### AC-RS-3 (RS-3) — Carries/dedup wiring

**Given** RS-2 and xtrax T2.2 landed, **When** spec declares carry/dedup fields (future), **Then** `make_sampling_planner` passes `carries` / `dedup_specs` to `BatchPlanner`. Until fields exist, empty lists are explicit (no silent skip).

### AC-RS-4 (RS-4) — Export fix

**Given** `from aminx.run import sample`, **When** called with `SamplingSpecification(...)`, **Then** execution routes to `host.runner.sample` (dict result). **Given** raw array call, **Then** `aminx.sampling.sample` remains available; deprecated `aminx.run` tensor path emits `DeprecationWarning` for one minor version.

### AC-RS-5 (RS-5) — Unified driver default

**Given** default `SamplingSpecification()`, **When** `_sample_batch` / `kernel_dispatch` reads unified driver, **Then** value matches `spec.use_unified_driver` (default `True`); `kernel_dispatch.py` `getattr(..., 'use_unified_driver', …)` default is corrected from `False` to `True`.

### AC-RS-6 (RS-6) — Phased migration

**Given** RS-2, **When** migrating host hot path, **Then** `host/plan.py`, `host/streaming.py`, `host/prep.py`, `host/runner.py` read generic caps from `spec.run_spec.*`; ruff `banned-api` blocks **new** flat-field reads in `host/*`.

### AC-RS-7 (RS-7) — Scoring parity

**Given** `ScoringSpecification(average_node_features=True)`, **When** `score(spec)` runs, **Then** InferencePlan averaging path is used (same topology as sampling). Temperature scaling applied or explicitly documented as N/A with test.

### AC-RS-8 (RS-8) — Exports + portable guard

**Given** `from aminx.run import InspectionSpecification`, **Then** import succeeds. **Given** a `RunSpec` with `grid.grid_mode=True` or `ligand.model_family='ligandmpnn'`, **When** `run_spec_portable_to_dict` runs, **Then** `ValueError` until v3 wire format. **Given** portable JSON with lossy/unknown grid or ligand keys, **When** `run_spec_portable_from_dict` runs, **Then** `ValueError` (no silent ignore).

---

## Decision log (adversarial cycle)

| Decision | Rationale |
|----------|-----------|
| **ACCEPT** PlannerTopology as aminx builder feeding T2.5b | Avoids duplicating BatchPlanner; maps today's inlined `make_sampling_planner` logic |
| **REVISE** AC-RS-1 from big-bang to lint-first + phased migration | Challenger #2 valid; list hot-path files that must migrate before T4.1 |
| **REVISE** portable JSON wire-format matrix + campaign guard | Challenger #5 valid latent risk |
| **REJECT** Re-export swap without deprecation shim | RS-4 includes warning period |
| **REJECT** Wire CarrySpec/DedupSpec before spec fields exist | Follow 260603 defer precedent; RS-3 gated on T2.2 |
| **ACCEPT** Bundle with epic #1541 as coordinated program | Separate sprint tracks; T4 never in foundations sprint |
| **DEFER** RS-8 to sprint 2+ | Not blocking T0–T1 |
| **REVISE** (R2) AC-RS-5 reads `spec.use_unified_driver`, not `run_spec.planner` | RS-C2: PlannerTopology is sprint-2; RS-5 is sprint-1 |
| **REVISE** (R2) portable guard on `to_dict`, not `from_dict` alone | RS-C10: v2 silently drops grid/ligand on serialize |
| **REVISE** (R2) PlannerTopology includes `_HETEROGENEOUS_AXIS_NAMES` | RS-C3/C6: parent T2.5b D5 mandate |
| **REVISE** (R2) parent DAG T4.1 deps include RS-2 (#1621) | RS-C1: machine-readable DAG sync |

---

## Assumptions

| # | Assumption | Falsification |
|---|------------|---------------|
| AS-RS1 | `build_run_spec()` remains single builder authority | Second builder introduced without test |
| AS-RS2 | Campaign manifest rows stay on full `spec_json` | Portable JSON used in campaign.py |
| AS-RS3 | PlannerTopology can be built without xtrax import | RS track imports xtrax |

---

## TBDs

| # | Question | Default |
|---|----------|---------|
| T-RS1 | Exact `topology_hash` algorithm (which fields, ordering) | SHA256 of canonical JSON of planner static fields |
| T-RS2 | Scoring HDF5 streaming | **SHELVED (2026-06-11 user):** #1444 permanently out of scope; no HDF5 scoring sink work |

---

## Pre-mortem record

**PM-RS-a:** PlannerTopology added fields not mirrored in `make_sampling_planner` → batch size mismatch and extra recompiles. **Countermeasure:** RS-2 AC requires cardinality parity test against legacy planner output.

**PM-RS-b:** Portable JSON used for campaign workers → silent wrong grid defaults. **Countermeasure:** AC-RS-8 guard; document wire-format matrix.

---

## References

- Parent: [`260611_aminx-xtrax-refactor.md`](260611_aminx-xtrax-refactor.md)
- Codebase model: [`../research/260611_aminx-xtrax-refactor-codebase-model.md`](../research/260611_aminx-xtrax-refactor-codebase-model.md)
- Architecture sequencing: [`260611_architecture-sequencing-testing-and-pack.md`](260611_architecture-sequencing-testing-and-pack.md)
- Code: `src/aminx/run/spec.py`, `src/aminx/run/specs.py`, `src/aminx/host/plan.py`, `src/aminx/host/runner.py`
