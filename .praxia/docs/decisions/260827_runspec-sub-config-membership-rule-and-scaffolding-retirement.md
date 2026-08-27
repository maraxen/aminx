---
title: RunSpec Sub-Config Membership Rule and Scaffolding Retirement
description: Decision record defining the two-clause membership rule for RunSpec sub-configs and documenting the retirement of write-only scaffolding.
status: accepted
task_id: 260827_runspec-scaffolding-remediation-spec
date: '260827'
supersedes: 260614_runspec-migration-map.md
backlog_ids: ''
---
# RunSpec Sub-Config Membership Rule and Scaffolding Retirement

## 1. Context & Motivation

The `RunSpec` track (RS-1 through RS-8, `.praxia/docs/specs/260611_runspec-unification.md`) was designed to provide a pytree-shaped configuration representation. `RunSpec` subclasses `xtrax.run.RunSpec` and contains sub-configs implemented as `equinox.Module` dataclasses with static leaves (`eqx.field(static=True)`), enabling them to cross JAX tracing and `jit` compilation boundaries. The flat facade `SamplingSpecification` cannot cross trace boundaries because it holds filesystem paths, loader handles, and unhashable state.

During early RS-1 scaffolding, 11 sub-configs were provisioned. However, many sub-configs became write-only scaffolding: populated during `build_run_spec()` but never consumed downstream because host runner and kernel dispatch paths continued reading the flat facade. An initial migration map (`.praxia/docs/plans/260614_runspec-migration-map.md`) documented line-level migrations, but quickly drifted and rotted into obsolescence.

This decision record establishes the project's canonical membership rule for `RunSpec` sub-configs and records the retirement of dead scaffolding.

---

## 2. The Two-Clause Membership Rule

A sub-config earns a seat on `RunSpec` if and only if **either**:

1. **Clause 1 (Trace-Adjacent Consumer):** At least one of its fields is read at or near a JAX traced/jitted dispatch boundary (e.g., `kernel_dispatch.py` or `plan.py` dispatch paths); **or**
2. **Clause 2 (Wire Boundary):** At least one of its fields is required to cross the portable-JSON wire boundary (`run_spec_portable_to_dict` / `run_spec_portable_from_dict`).

A sub-config satisfying neither clause is dead scaffolding and must not be retained. Fields read only during host-side orchestration, metadata dictionary construction, or dataset loading do not earn a seat on `RunSpec`.

---

## 3. Retained Sub-Config Roster

Following the WS-A remediation (2026-08-27), `RunSpec` retains exactly 6 sub-configs and 2 top-level callable slots:

| Sub-Config | Satisfied Clause | Justification |
|---|---|---|
| `sampling: SamplingConfig` | **Clause 1** | Consumed directly in traced kernel dispatch (`kernel_dispatch.py:500,533`) and memory plan budgeting (`plan.py:941-950`). |
| `plan: PlannerTopology` | **Clause 1** | Drives JAX kernel dispatch topology selection and cache-key generation. |
| `io: IOConfig` | **Clause 2** | Serialized across portable-JSON wire boundary (`run_spec_portable_json.py:163-165`). Host-side readers also read `output_h5_path` / `cache_path`. |
| `resource: ResourceConfig` | **Clause 2** | Serialized across portable-JSON wire boundary (`run_spec_portable_json.py:172-177`). |
| `multistate: MultistateConfig` | **Clause 2** | Serialized across portable-JSON wire boundary (`run_spec_portable_json.py:167-171`). |
| `precision: PrecisionConfig` | **Clause 2** | Serialized across portable-JSON wire boundary (`run_spec_portable_json.py:178`). |
| `encoding_fusion` / `decoding_fusion` | Top-level callable slots | Static callable leaves (`eqx.field(static=True)`) injected into stagesets for multistate fusion. |

---

## 4. Scaffolding Retirement History

The evolution of `RunSpec` sub-configs is summarized as follows:

1. **RS-1 Inception:** 11 sub-configs were created in `src/aminx/run/spec.py`.
2. **Scaffolding Stall:** 5 sub-configs became write-only scaffolding (populated by `build_run_spec`, never read).
3. **First Cleanup (PR #91 / commit `16d7e0d9`):** `TiedPositionsConfig`, `BatchingConfig`, and `AveragingConfig` (18 fields total) were deleted.
4. **Remediation WS-A (2026-08-27):** `GridLineageConfig` (6 fields) and `LigandConfig` (5 fields) were deleted after structural tracing proved their apparent "guards" were completely unreachable dead code in production.
5. **Net State:** 29 write-only fields across 5 dead sub-configs have been deleted.

Future contributors must **not** resurrect or re-add dead sub-configs like `BatchingConfig` or `GridLineageConfig` without satisfying the two-clause membership rule.

---

## 5. Disposition of RS-8 Export Guard

The original RS-8 specification (`.praxia/docs/specs/260611_runspec-unification.md:91`) required `run_spec_portable_to_dict` to raise `ValueError` on specs with `grid.grid_mode=True` or `ligand.model_family="ligandmpnn"`.

With the deletion of `RunSpec.grid` and `RunSpec.ligand` in WS-A:
- The **export-direction guard** (`to_dict`) is retired: `RunSpec` instances no longer hold grid or ligand fields, and `to_dict`'s sole caller (`aminx spec portable-roundtrip`) is fed directly by `from_dict`.
- The **import-direction guard** (`from_dict`) is hardened: `run_spec_portable_from_dict` now strictly rejects any top-level key outside `_PORTABLE_RUN_SPEC_V2_KEYS = {"version", "io", "multistate", "resource", "precision"}`, ensuring no payload carrying arbitrary unhandled blocks is silently accepted.

---

## 6. Field-Level Inventory and Anti-Drift Enforcement

Human-maintained `file:line` inventory tables (like `260614_runspec-migration-map.md`) are inherently brittle. Field-level classification is now enforced programmatically:

- `src/aminx/run/_runspec_coverage.py` derives the `SERIALIZATION_ONLY` bucket dynamically by executing `run_spec_portable_to_dict` against a probe spec.
- All remaining sub-config fields must be explicitly classified (`MIGRATED` or `MIRRORED` with a substantial reason).
- An import-time exhaustiveness assertion (`assert_runspec_coverage_is_exhaustive()`) guarantees that adding or removing fields without updating the classification fails immediately at import, mirroring `src/aminx/host/spec_partition.py`.

---

## 7. Open Questions

1. **Portable-JSON Wire Format Value:** Four sub-configs (`IOConfig`, `ResourceConfig`, `MultistateConfig`, `PrecisionConfig`) exist solely to satisfy Clause 2 (the portable-JSON wire format). If `portable` v2 is ever retired or simplified, those sub-configs must be re-evaluated under Clause 1.


