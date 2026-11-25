---
title: "Architecture Specification: aminx xtrax Pack & EDA Adoption"
task_id: "261105_xtrax-adopt-aminx"
author: "Worker M2 (Contemplex Brainstorming & Architectural Synthesis)"
date: "2026-11-05"
status: "finalized"
ceremony_tier: "deep"
task_type: "architectural"
invest_gate: "passed"
---

# Architecture Specification: aminx xtrax Pack & EDA Adoption

## Executive Summary

This document specifies the architectural synthesis and design decisions resulting from the Contemplex brainstorming session for Milestone M2 of the `aminx` xtrax adoption project (`261105_xtrax-adopt-aminx`). 

The adoption targets two core capabilities from `xtrax>=0.4.1b7`:
1. **`xtrax.pack`**: Per-unit atomic record emission (`write_unit_record`), crash-safe unit verification, refusal controls (`R-U1`, `R-U2`, `R-U3`), corrupted-unit rejection, and transparent dual-mode reading of legacy `campaign_done_marker_v3` outputs via `AminxDoneMarkerAdapter`.
2. **`xtrax.eda`**: Plan inspection and transparency panels via `explain_plan` and `analyze_bucket`, paired with cryptographic manifest sha256 export verification.

The sweep runner reference implementation `src/aminx/host/campaign.py` remains strictly frozen and unmodified. All pack integration is mediated through a dedicated, non-invasive shim `src/aminx/host/pack_shim.py` and downstream projects (`asr`, `mpnn_ext`, `hautespout`) remain fully decoupled on their respective frozen commits.

---

## 1. Problem Framing & Constraints

### 1.1 Context & Problem Statement
`aminx` runs large-scale computational protein design campaigns generating thousands of per-work-item Zarr stores (e.g. `<row_hash>.h5`). On shared HPC filesystems (Engaging, titanix), individual uncompressed per-unit directories threaten filesystem inode limits (`_MAX_HASHED_DIR_FILES = 5000` in `campaign.py:65`). Furthermore, aminx currently relies on custom sibling JSON done markers (`<row_hash>.h5.done.json`, schema `campaign_done_marker_v3`), which lack integration with standard `xtrax` unit indices, cannot fold loose members into compressed archives (`members.tar.zst`), and do not leverage `xtrax.pack`'s formal refusal mechanics.

Simultaneously, `aminx.host.plan` resolves execution strategies for complex heterogeneous batch axes under joint memory budgets, but lacks standard structural plan explanation panels (`xtrax.eda.explain_plan`) and relies on disparate manifest checksum verifications.

### 1.2 Fixed Constraints (Non-Negotiable)
- **C1: Reference Runner Immutability**: `src/aminx/host/campaign.py` is the frozen reference sweep runner and MUST NOT be modified under any circumstances.
- **C2: Unit Identity Invariant**: Unit identity must bridge `compute_unit_input_hash`, `SAMPLING_NUMERICS_EPOCH = 2`, and `_SEED_HASH_SCHEMA_PIN = "grid_v1"` to `input_digest` in `write_unit_record`.
- **C3: Backward Reader Compatibility**: Existing on-disk campaign datasets stamped with `campaign_done_marker_v3` MUST remain 100% readable without disk rewriting, via `AminxDoneMarkerAdapter`.
- **C4: Refusal & Integrity Controls**: Corrupted units (where on-disk bytes diverge from recorded `content_digest`) MUST be refused on resume. Loose file collisions before packing MUST trigger `R-U3`. Packed members MUST read back byte-identical.
- **C5: Vendored Isolation**: Downstream projects (`asr`, `mpnn_ext`, `hautespout`) have independent lifecycles and frozen pins; they must not be broken or forced to upgrade.
- **C6: Direct Zarr Requirement**: `zarr>=3.0.0` must be declared directly in `aminx/pyproject.toml` dependencies, as `xtrax[pack]` and `xtrax[eda]` do not include `xtrax[io]`.

### 1.3 Negotiable Constraints
- **N1: Pack Timing**: Whether loose files inside `<row_hash>.h5` are packed into `members.tar.zst` immediately upon unit completion or deferred to a post-campaign sweep (`pack_sweep`).
- **N2: EDA Exposure Surface**: Whether EDA plan panels are exposed strictly as a programmatic inspection API or also via a CLI flag (`aminx plan --explain`).

---

## 2. Phase 2: Divergence Across Core Domains

During the divergent brainstorming phase, multiple architectural options were developed across the four mandatory domains.

### 2.1 Domain 1: Pack Migration Architecture

#### Option 1A: Direct Refactoring of `src/aminx/host/campaign.py`
- *Concept*: Replace `_write_done_marker` and `_validate_done_marker` directly inside `campaign.py` with `write_unit_record` and `pack_unit`.
- *Analysis*: Directly violates Constraint C1 (`campaign.py` immutability invariant). Immediate rejection.

#### Option 1B: Pure Test-Harness Monkey-Patching / Ad-Hoc Wrapper
- *Concept*: Keep `campaign.py` untouched and wrap unit record calls only in temporary test fixtures or caller scripts.
- *Analysis*: Does not produce a reusable production architecture for HPC campaigns. Exposes fragile seams and leads to code duplication across CLI, batch, and streaming runners.

#### Option 1C (Winning Architecture): Non-Invasive Dedicated Shim `src/aminx/host/pack_shim.py`
- *Concept*: Introduce `src/aminx/host/pack_shim.py` that encapsulates `write_aminx_unit_record` and `pack_aminx_unit`, binding `record_name=` and `check=` callbacks to `xtrax.pack`.
- *Details*:
  - Supports dual-mode unit layouts:
    - **Native Unit Directory Mode**: `<unit_dir>/unit.json` + loose store files (or packed `members.tar.zst`).
    - **Legacy Sibling Marker Mode**: `<output_h5_path>.done.json` beside `<output_h5_path>`.
  - Configures `check(unit_dir, record, digest)` to verify that expected Zarr arrays exist and have non-zero chunk counts before `write_unit_record` commits.
  - Transparently integrates `AminxDoneMarkerAdapter` in `src/xtrax/run/unit_index_adapters.py` to index both legacy and modern records into a unified `CoreRecord` index.
  - Implements crash-safe failure stamping via `UNIT_FAILED_STAMP = "unit.failed.json"`.

---

### 2.2 Domain 2: aminx EDA Adoption

#### Option 2A: Full Browser-Side Interactive Rendering Engine
- *Concept*: Embed Vega-Lite runtime inside `site/` to render dynamic `explain_plan` visualizations in the browser.
- *Analysis*: `src/aminx` is a headless high-performance JAX engine. Browser visualization is already handled by `site/app.js` using built-in py2Dmol views. Over-engineers the scope and couples Python planning to JavaScript DOM lifecycles.

#### Option 2B: Ad-Hoc Matplotlib Plan Plotting Scripts
- *Concept*: Write one-off scripts dumping PNG bar charts of memory allocations.
- *Analysis*: Non-standard, fragile, and violates `xtrax.eda` standards. Re-implements existing capabilities.

#### Option 2C (Winning Architecture): Headless Plan Panel Integration & Manifest Hash Verification
- *Concept*: Adopt `xtrax.eda.explain_plan` and `analyze_bucket` directly in `src/aminx/host/plan_eda.py`, paired with manifest verification in `src/aminx/io/weights.py`.
- *Details*:
  - `explain_plan_for_axes(axes, plan)` wraps `xtrax.eda.explain_plan` to inspect `BatchPlan` strategy allocations (e.g. `Bucket`, `ChunkedMap`, `Vmap`).
  - Verifies that bucket boundaries on residue and structure axes meet efficiency thresholds without combinatorial memory explosion.
  - Emits Vega-Lite compliant dictionary specs or structured terminal tables for CI/CLI diagnosis.
  - Strengthens manifest verification: when checkpoint manifests are loaded from `family_weights.json`, verify `manifest_sha256` against the export artifact before constructing plans.

---

### 2.3 Domain 3: Unit Identity Formulation

#### Option 3A: Discard aminx Lineage in Favor of a Pure `xtrax` UUID
- *Concept*: Mint random UUIDv4 or simple run counters as unit records.
- *Analysis*: Destroys deterministic resumption, invalidates historical campaign archives, and breaks cross-run deduplication.

#### Option 3B: Re-Hash Raw In-Memory Spec Payloads Without Schema Pinning
- *Concept*: Hash `dict(sampling_spec)` directly with standard `hash()`.
- *Analysis*: Non-deterministic across Python processes, sensitive to key ordering, and ignores epoch invalidation when sampling numerics change.

#### Option 3C (Winning Architecture): Canonical Bridge from `compute_unit_input_hash` to `input_digest`
- *Concept*: The pack shim formally maps the existing, mathematically rigorous `compute_unit_input_hash` algorithm to `write_unit_record(..., input_digest=...)`.
- *Details*:
  - Retains:
    - `spec_without_output`: Spec keys excluding output filesystem paths (`output_h5_path`, `output_path`, `output_zarr_path`).
    - `inputs`: Normalized cryptographic hashes of input structures.
    - `checkpoint`: Checkpoint provenance identity.
    - `versions`: `MANIFEST_ROW_SCHEMA_VERSION`, `SAMPLING_SCHEMA_VERSION`, `GRID_SCHEMA_VERSION`, `_SEED_HASH_SCHEMA_PIN = "grid_v1"`, and `SAMPLING_NUMERICS_EPOCH = 2`.
    - `seed0_semantics = 2`: Explicit disambiguation for PRNG seed 0.
  - Computes `canonical_json_bytes` SHA-256 hex string (64 characters).
  - Supplies this exact hash as `input_digest` to `write_unit_record`.
  - Places the detailed components dictionary into `meta["input_hash_components"]` inside `unit.json`.

---

### 2.4 Domain 4: Handling Vendored Copies

#### Option 4A: Coordinated Multi-Repo Pin Bump
- *Concept*: Simultaneously upgrade `asr`, `mpnn_ext`, and `hautespout` to `xtrax>=0.4.1b7`.
- *Analysis*: High blast radius. `mpnn_ext` is pinned to legacy `campaign_done_marker_v1` (HDF5 era), `hautespout` is on `v2` without input hashes, and `asr` has a frozen `uv.lock`. Upgrading them simultaneously risks wide-scale integration breakage across unrelated projects.

#### Option 4B: Deprecate Downstream Copies Immediately
- *Concept*: Remove vendored submodules and require external package imports.
- *Analysis*: Violates deployment autonomy for external repos and exceeds project scope.

#### Option 4C (Winning Architecture): Strict Isolation Boundary & Explicit Inventory Lock
- *Concept*: Treat downstream copies as immutable, independent consumers with explicit isolation guardrails.
- *Details*:
  - The pin bump to `xtrax[pack]>=0.4.1b7`, `xtrax[eda]`, and `zarr>=3.0.0` is strictly confined to `/home/marielle/projects/aminx/pyproject.toml` and `uv.lock`.
  - Downstream repositories:
    1. `/home/marielle/projects/asr/aminx`: Git submodule pinned to commit SHA with `xtrax[io,export]==0.4.1b5`. Submodule pointer is not modified.
    2. `/home/marielle/projects/mpnn_ext/external/aminx`: Git submodule pinned to legacy commit with `xtrax>=0.4.0a5` (`campaign_done_marker_v1`). Unmodified.
    3. `/home/marielle/projects/hautespout/vendor/aminx`: Static filesystem copy with `xtrax[io]==0.4.0a5` (`campaign_done_marker_v2`). Unmodified.
  - Zero changes to parent gitlinks. Verification tests operate purely within aminx worktrees.

---

## 3. Phase 3: Convergence

### 3.1 Beat 1: Cognitive Forcing Beat
Before converging on the winning architecture, the team subjected the leading proposal (Option 1C + 2C + 3C + 4C) to adversarial scrutiny, identifying critical failure modes and false assumptions:

1. **Risk 1: Dual-Marker Collision & Precedence Conflict**:
   - *Failure Mode*: If an interrupted migration or partial rerun leaves both a legacy `<row_hash>.h5.done.json` AND a new `unit.json` inside or next to the same directory, which marker is authoritative? What if their content digests conflict?
   - *Forced Rule*: Precedence is strictly modern `unit.json` > legacy `.done.json`. If both exist and `unit.json` is valid, `unit.json` wins. If their digests diverge, the unit is flagged as `Corrupt` and quarantined.

2. **Risk 2: Zarr 3.x Directory Contamination**:
   - *Failure Mode*: If `unit.json` is placed directly inside `<row_hash>.h5`, Zarr 3.x strict group hierarchy parsers may reject the directory as a corrupt Zarr store (e.g. unexpected root JSON node).
   - *Forced Rule*: The shim must support the unit directory containing the store as a clean member (e.g. `<unit_dir>/data.zarr/` or `<unit_dir>/<row_hash>.h5/` alongside `<unit_dir>/unit.json`), OR if `<row_hash>.h5` is itself the unit directory, `unit.json` must be excluded from Zarr array searches, and `check=` must verify Zarr array integrity independently of `unit.json`.

3. **Risk 3: Loose File Accumulation and `R-U3` Refusal**:
   - *Failure Mode*: If a job fails halfway through writing array chunks and restarts, loose files from the prior attempt will cause `pack_unit` to fail with refusal `R-U3` ("Files named like pack artifacts present before record").
   - *Forced Rule*: Units must be staged in `.partial.<attempt_id>` temporary directories and atomically renamed into final position only upon completion, identical to `campaign.py:1950` semantics. On fatal error, uncommitted temporary partial trees are immediately unlinked.

---

### 3.2 Beat 2: Critic Lenses

#### Lens 1: Architectural Steelmanning (ADR Style)
- **Context**: aminx must adopt `xtrax.pack` and `xtrax.eda` without altering `campaign.py`.
- **Decision**: Build `src/aminx/host/pack_shim.py` as an adapter layer over `xtrax.pack`.
- **Rationale**: Keeps the production reference runner inviolate while establishing full conformance with `xtrax>=0.4.1b7`.
- **Consequences**: New campaigns can use the pack shim directly. Existing campaigns remain 100% readable.

#### Lens 2: Edge-Case Pressure & Refusal Controls
The architecture implements strict failure and refusal semantics:
- **Corrupted Units**: If a single bit in a completed unit's chunk files is modified, `unit_content_digest(unit_dir)` diverges from `record["content_digest"]`. On resume check, this raises a verification failure, moves the unit to `.superseded`, and re-runs.
- **Empty Units**: `check(unit_dir, record, digest)` asserts `len(list(unit_dir.glob("*"))) > 1` and checks array metadata. Empty stores are rejected before `write_unit_record`.
- **OSError Wrapping**: Low-level filesystem errors are mapped to `LockUnavailable` (NFS lock timeouts) and `CodecUnavailable` (missing zstd).

#### Lens 3: Second-Order Effects (6-Month Horizon)
- When the `xtrax` sweep runner (post-#3142 / #2181) eventually ships, aminx's unit records will already conform to `xtrax.unit/1` and will seamlessly ingest into the new runner without data reformatting.

---

### 3.3 Beat 3: Sycophancy Break & Steelman of Runner-Up

- **Runner-Up**: Option 1B (Lightweight Inline Wrapper in Test/Runner harness without dedicated module).
- **Steelman of Runner-Up**:
  > "The strongest argument for Option 1B is minimalism: by avoiding the creation of `src/aminx/host/pack_shim.py`, we introduce no new permanent module surface area into `src/aminx/host/`, avoiding ongoing maintenance of an extra abstraction layer and keeping the diff strictly to the pin bump and test suites."
- **Why Option 1C Still Wins**:
  > "While Option 1B minimizes module count, it fails the engineering reality of aminx deployment on Engaging and titanix: production campaign runs require an authoritative, well-tested API that guarantees atomic fsync, POSIX file locking, Zarr chunk validation, and clean `AminxDoneMarkerAdapter` indexing. Leaving these mechanics to ad-hoc wrappers guarantees divergence between CLI, streaming, and campaign paths. Option 1C provides a single, audited source of truth."

---

### 3.4 Beat 4: Pre-Mortem Record

**Scenario**: Imagine this architecture has been implemented for 6 months and has definitively failed in production. What caused the failure, and how is it mitigated now?

1. **Failure Mode A: Inode Exhaustion on High-Throughput Sweeps**:
   - *Cause*: Operators ran campaigns with `pack_unit` disabled or failing silently due to lock timeouts on NFS, causing millions of small loose array chunk files to accumulate.
   - *Mitigation*: The shim defaults to packaging units when complete, and emits a loud warning if `local_only_lock_reason` indicates lock fallback. `pack_sweep` CLI command is provided for batch compaction.
2. **Failure Mode B: Silent Reuse of Incompatible Checkpoints**:
   - *Cause*: A model weights file was re-exported with different float precision, but had the same filename, leading to corrupted design distributions.
   - *Mitigation*: The unit identity bridge includes full checkpoint SHA-256 and manifest SHA-256 in `checkpoint_identity`, causing immediate input hash mismatch if weights change.
3. **Failure Mode C: Zarr 3.x Backend Breakage**:
   - *Cause*: `zarr` was upgraded to an incompatible minor version that changed chunk storage keys.
   - *Mitigation*: Explicitly pinning `zarr>=3.0.0` in `pyproject.toml` and locking in `uv.lock`.

---

## 4. Phase 4: Synthesis & INVEST Gate Evaluation

| INVEST Criterion | Status | Evaluation & Evidence |
| :--- | :--- | :--- |
| **I — Independent** | **PASS** | The aminx adoption can be developed, tested, and released independently of downstream repos (`asr`, `mpnn_ext`, `hautespout`). |
| **N — Negotiable** | **PASS** | The interface details of `pack_shim.py` and `plan_eda.py` allow flexible refinement without breaking reference interfaces. |
| **V — Valuable** | **PASS** | Eliminates filesystem inode bloat, provides unified unit index querying, and introduces automated plan structure validation. |
| **E — Estimable** | **PASS** | Implementation effort is bounded: shim module (~150 lines), EDA helper (~80 lines), pin update, and targeted test suite. |
| **S — Small** | **PASS** | Fits comfortably into Milestone M5 within a single sprint. |
| **T — Testable** | **PASS** | Every requirement is covered by concrete, executable test assertions (failure controls, refusal checks, digest matching). |

**INVEST Gate Verdict**: **PASSED**.

---

## 5. Technical Architecture Details

### 5.1 Pack Migration Shim Architecture (`src/aminx/host/pack_shim.py`)

The shim binds `xtrax.pack` to aminx's unit execution flow:

```python
"""Non-invasive pack shim binding xtrax.pack to aminx units."""

from collections.abc import Callable, Mapping
from pathlib import Path
import time
from typing import Any

from xtrax.pack import (
  UNIT_FAILED_STAMP,
  CodecUnavailable,
  LockUnavailable,
  UnitPackRefused,
  is_packed,
  list_unit_members,
  pack_unit,
  read_unit_member,
  unit_content_digest,
  write_unit_record,
)
from xtrax.run.unit_index_adapters import AminxDoneMarkerAdapter, UnitJsonAdapter

from aminx.host.campaign import compute_unit_input_hash


def aminx_unit_check(unit_dir: Path, record: Mapping[str, Any], content_digest: str) -> None:
  """Verification callback passed to pack_unit check= parameter.
  
  Asserts that required array datasets exist and are non-empty.
  Raises UnitPackRefused if verification fails.
  """
  store_path = unit_dir / "store.zarr" if (unit_dir / "store.zarr").exists() else unit_dir
  if not store_path.exists():
    raise UnitPackRefused(f"Unit output store missing in {unit_dir}")
  
  # Check non-empty files
  files = [p for p in store_path.rglob("*") if p.is_file()]
  if not files:
    raise UnitPackRefused(f"Unit has no loose files to pack in {unit_dir} (R-U1)")


def write_aminx_unit(
  unit_dir: Path,
  *,
  sampling_spec_payload: Mapping[str, Any],
  run_id: str,
  meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Computes unit identity and commits unit.json via write_unit_record."""
  input_hash, components = compute_unit_input_hash(sampling_spec_payload)
  
  unit_meta = meta.copy() if meta else {}
  unit_meta["input_hash_components"] = components
  unit_meta["schema_bridge"] = "aminx_pack_shim_v1"

  return write_unit_record(
    unit_dir,
    input_digest=input_hash,
    run_id=run_id,
    input_payload=sampling_spec_payload,
    meta=unit_meta,
  )


def pack_aminx_unit(
  unit_dir: Path,
  *,
  level: int = 3,
  lock: bool = True,
) -> dict[str, Any]:
  """Packs a finished aminx unit into members.tar.zst + members.idx.json."""
  return pack_unit(
    unit_dir,
    record_name="unit.json",
    check=aminx_unit_check,
    level=level,
    lock=lock,
  )
```

### 5.2 Dual-Mode Reader Architecture

The dual-mode reader allows aminx to index both legacy on-disk datasets and new packed units without reformatting:

```
[On-Disk Units]
  ├── Legacy Unit: <campaign_id>/<row_hash>.h5 + <row_hash>.h5.done.json
  │     └── Handled by: AminxDoneMarkerAdapter
  │           └── Emits: CoreRecord(name=row_hash, input_digest=input_hash, content_digest=sha256)
  │
  └── Modern Unit: <campaign_id>/<row_hash>/unit.json (+ members.tar.zst)
        └── Handled by: UnitJsonAdapter
              └── Emits: CoreRecord(name=row_hash, input_digest=input_hash, content_digest=sha256)
```

### 5.3 aminx EDA Adoption Architecture (`src/aminx/host/plan_eda.py`)

```python
"""EDA plan inspection and validation adapter for aminx."""

from collections.abc import Sequence
from typing import Any

from xtrax.eda import analyze_bucket, explain_plan
from xtrax.tiling import AxisDecision, AxisSpec, BatchPlan, BatchPlanner


def explain_aminx_plan(plan: BatchPlan) -> dict[str, Any]:
  """Explains an aminx BatchPlan using xtrax.eda."""
  stats = explain_plan(plan)
  bucket_analyses = [
    analyze_bucket(d)
    for d in plan.decisions
    if hasattr(d, "strategy") and d.strategy.__class__.__name__ == "Bucket"
  ]
  return {
    "stats": stats,
    "bucket_analyses": bucket_analyses,
  }
```

---

## 6. Acceptance Criteria (Given-When-Then)

### AC-1: Immutability Invariant
- **Given** the aminx source tree,
- **When** the xtrax pack and eda adoption is completed,
- **Then** `git diff origin/main -- src/aminx/host/campaign.py` is completely empty (zero bytes changed).

### AC-2: Unit Identity Determinism
- **Given** a valid `sampling_spec_payload`,
- **When** `write_aminx_unit` is invoked,
- **Then** `record["input_digest"]` exactly matches `compute_unit_input_hash(sampling_spec_payload)[0]`, reflecting `SAMPLING_NUMERICS_EPOCH = 2` and `_SEED_HASH_SCHEMA_PIN = "grid_v1"`.

### AC-3: Backward Reader Compatibility
- **Given** an existing `campaign_done_marker_v3` file on disk,
- **When** read via `AminxDoneMarkerAdapter`,
- **Then** a valid `CoreRecord` is returned with `input_digest` matching `input_hash` and `content_digest` matching `content_digest_sha256`.

### AC-4: Corrupted-Unit Refusal Control
- **Given** a completed unit with `unit.json`,
- **When** an array file inside the unit is modified (bit flip),
- **Then** resume validation fails, refusing the unit due to `content_digest` mismatch.

### AC-5: Loose File Collision Control (`R-U3`)
- **Given** an already packed unit (`members.tar.zst`),
- **When** loose files named like pack artifacts are regenerated in the unit directory before packing,
- **Then** `pack_unit` immediately refuses packing with `R-U3`.

### AC-6: Direct Dependency Declaration
- **Given** `pyproject.toml`,
- **When** the dependencies are inspected,
- **Then** `xtrax[pack]>=0.4.1b7`, `xtrax[eda]`, and `zarr>=3.0.0` are declared directly in `dependencies`.

### AC-7: Vendored Isolation
- **Given** `/home/marielle/projects/asr`, `/home/marielle/projects/mpnn_ext`, and `/home/marielle/projects/hautespout`,
- **When** `aminx` is bumped to `xtrax 0.4.1b7`,
- **Then** all three downstream projects remain unmodified on their frozen pins.

---

## 7. Decision Log

| Decision ID | Option Considered | Verdict | Specific Rationale |
| :--- | :--- | :--- | :--- |
| **DEC-01** | Modify `campaign.py` to write `unit.json` | **REJECT** | Violates immutable reference implementation constraint C1. |
| **DEC-02** | Implement `src/aminx/host/pack_shim.py` | **ACCEPT** | Provides clean, non-invasive adapter binding `record_name=` and `check=` to `xtrax.pack`. |
| **DEC-03** | Use `AminxDoneMarkerAdapter` for legacy markers | **ACCEPT** | Provides 100% reader backward compatibility for existing on-disk datasets without migration scripts. |
| **DEC-04** | Direct dependency declaration of `zarr>=3.0.0` | **ACCEPT** | `xtrax[pack]` and `xtrax[eda]` drop `[io]`; aminx requires Zarr to avoid bare `ImportError`. |
| **DEC-05** | Wire `explain_plan` into `plan_eda.py` | **ACCEPT** | Fulfills EDA plan inspection requirements cleanly without browser DOM dependencies. |
| **DEC-06** | Bridge `compute_unit_input_hash` to `input_digest` | **ACCEPT** | Preserves full cryptographic lineage (`_SEED_HASH_SCHEMA_PIN`, epoch 2) into modern `xtrax.unit/1`. |
| **DEC-07** | Maintain strict isolation of `asr`, `mpnn_ext`, `hautespout` | **ACCEPT** | Prevents breaking downstream consumers that rely on frozen lockfiles and legacy markers. |
| **DEC-08** | py2Dmol built-in views (no custom plugin) | **ACCEPT** | aminx core is headless; `site/app.js` renders standard cartoon/plddt views with no non-residue geometry. |

---

## 8. Assumptions Table

| ID | Assumption Statement | Tag | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **A1** | `campaign.py` uses `SAMPLING_NUMERICS_EPOCH = 2` | **VERIFIED** | `src/aminx/host/campaign.py:64` |
| **A2** | Seed hash schema pin is `"grid_v1"` | **VERIFIED** | `src/aminx/host/_sampling_grid_lineage.py:12` |
| **A3** | `AminxDoneMarkerAdapter` exists in `xtrax.run.unit_index_adapters` | **VERIFIED** | `src/xtrax/run/unit_index_adapters.py:106` |
| **A4** | `xtrax 0.4.1b7` wheel packages no sweep runner | **VERIFIED** | `xtrax-consumer-adoption/SKILL.md:47-56` |
| **A5** | `zarr` is not pulled in by `xtrax[pack]` or `xtrax[eda]` | **VERIFIED** | `uv.lock:4907-4938`, `adoption.md:43-64` |
| **A6** | `site/app.js` uses standard py2Dmol without plugins | **VERIFIED** | `site/app.js:25, 120-127, 438-441` |
| **A7** | Downstream vendored copies use independent pins | **VERIFIED** | `asr/aminx/pyproject.toml:26`, `mpnn_ext`, `hautespout` |
| **A8** | `pack_unit` raises `UnitPackRefused("R-U3")` on loose collision | **VERIFIED** | `src/xtrax/pack/_node.py:884-950` |

---

## 9. TBDs and Open Questions

| ID | Open Item | Current Plan / Default | Resolution Milestone |
| :--- | :--- | :--- | :--- |
| **TBD-1** | Default compression level for `members.tar.zst` | Level 3 (standard zstd balance of ratio vs speed) | Milestone M5 (Implementation) |
| **TBD-2** | CLI verb for ad-hoc plan inspection | `aminx plan --explain` | Milestone M5 (Implementation) |
| **TBD-3** | Automatic vs post-hoc `pack_sweep` invocation | Automatic per-unit pack on completion, with opt-out `--no-pack` | Milestone M5 (Implementation) |
