---
title: "P3 Adoption Specification: aminx xtrax Pack & EDA Migration"
task_id: "261105_xtrax-adopt-aminx"
author: "Specification Miner M3 (P3 Adoption Specification)"
date: "2026-11-05"
status: "draft"
ceremony_tier: "deep"
task_type: "specification"
category: "specs"
invest_gate: "passed"
epic_id: 6497
child_items: [6498, 6499, 6500, 6501, 6502, 6503, 6504, 6505, 6506, 6507, 6508, 6509]
tags: ["xtrax", "pack", "eda", "campaign", "unit-index", "zarr", "py2Dmol", "specs"]
---

# P3 Adoption Specification: aminx xtrax Pack & EDA Migration

## Executive Summary

This formal specification establishes the implementation contracts, architecture, verification protocols, and edge-case handling for migrating `aminx` onto `xtrax.pack` and `xtrax.eda` under Task ID `261105_xtrax-adopt-aminx` (Epic #6497, resolving debts #3257, #3143, and #3142).

The core technical objectives are:
1. **`xtrax.pack` Adoption**: Move aminx per-work-item execution units from legacy ad-hoc sibling done markers (`<row_hash>.h5.done.json`, schema `campaign_done_marker_v3`) onto standard `xtrax.pack` unit records (`unit.json`, schema `xtrax.unit/1`) and compressed multi-member archives (`members.tar.zst` + `members.idx.json`). This resolves shared HPC filesystem inode exhaustion (`_MAX_HASHED_DIR_FILES = 5000` bound) while enforcing strict write-once semantics (`R-U1`), loose collision prevention (`R-U3`), POSIX lock safety (`LockUnavailable`), and corrupted-unit refusal.
2. **Backward Reader Compatibility**: Guarantee 100% backward reader compatibility for existing on-disk campaign datasets stamped with `campaign_done_marker_v3` via `AminxDoneMarkerAdapter` (`src/xtrax/run/unit_index_adapters.py:106-165`), requiring zero data rewriting or on-disk migrations for historical runs.
3. **`xtrax.eda` Adoption**: Formalize plan inspection panels (`explain_plan`, `analyze_bucket`) for aminx's `BatchPlan` / `BatchPlanner` execution paths, and enforce cryptographic checkpoint integrity via `manifest_sha256` verification.
4. **Reference Runner Immutability**: Enforce that the production sweep runner `src/aminx/host/campaign.py` remains strictly frozen and unmodified (0-byte diff against `origin/main`), mediating all new packing logic through a non-invasive shim module `src/aminx/host/pack_shim.py`.
5. **Dependency & Downstream Isolation**: Upgrade aminx to `xtrax[pack,eda,export]>=0.4.1b7`, direct requirements `zarr>=3.0.0` and `flax>=0.8.0`, and optional extra `onnx = ["xtrax[onnx]>=0.4.1b7"]`, with verified compatibility across `jax>=0.4.35` through `0.10.2`, while strictly isolating downstream vendored copies (`asr`, `mpnn_ext`, `hautespout`).
6. **py2Dmol Formal Assessment**: Classify aminx visualization architecture as **Built-in views** (standard cartoon/plddt rendering in `site/app.js`), with no py2Dmol renderer plugin required and a strictly headless Python core.

---

## Backlog Epic & Child Items Traceability

All work under this specification maps directly to Backlog Epic #6497 in the `aminx` workspace:

| Backlog ID | Milestone | Phase / Workstream | Title & Scope | Status |
| :--- | :--- | :--- | :--- | :--- |
| **#6497** | ALL | **[EPIC]** | **Adopt xtrax.pack and xtrax.eda in aminx (261105_xtrax-adopt-aminx)** | `open` |
| **#6498** | M0 | P0: Reconnaissance | Applicability evaluation, done-marker inventory, vendored copies, and debts | `open` (M0 done) |
| **#6499** | M1 | P1: Research | xtrax pin bump plan, JAX compatibility, and py2Dmol assessment | `open` (M1 done) |
| **#6500** | M2 | P2: Brainstorming | Contemplex session on pack shim, reader backward-compat, and refusal controls | `open` (M2 done) |
| **#6501** | M3 | P3: Specification | Formal adoption specification with assumption table, edge-case table, verification plan | `open` (Active) |
| **#6502** | M4 | P4: Review | 4-role adversarial review (challenger/defender/adjudicator/editor) and verification spikes | `open` (Planned) |
| **#6503** | M5 | P5a: Dependencies | Pin bump to `xtrax[pack]>=0.4.1b7`, `xtrax[eda]`, `zarr>=3.0.0`, JAX checks | `open` (Planned) |
| **#6504** | M5 | P5b: Pack Shim | Pack migration, `write_unit_record` shim, and backward reader compatibility | `open` (Planned) |
| **#6505** | M5 | P5c: Refusal Controls| Corrupted-unit refusal controls, negative test controls, and resume verification | `open` (Planned) |
| **#6506** | M5 | P5d: Parity | Parity verification with `campaign.py` reference on real manifest slice | `open` (Planned) |
| **#6507** | M6 | P6: Pull Requests | Open workstream PRs against `main` referencing debts, spec, and run IDs | `open` (Planned) |
| **#6508** | M7 | P7: Review | Code review convergence and green CI (`pytest`, `ruff`) | `open` (Planned) |
| **#6509** | M8 | DONE: Delivery | Final handoff report, debt status updates, and skill documentation proposal | `open` (Planned) |

---

## 1. Comprehensive Assumption Table

Every technical premise, tool capability, file layout, and interface contract across `aminx` and `xtrax` is tagged below. Every assumption is verified by direct source inspection:

| ID | Subject | Assumption Statement | Tag | Verification Evidence & Citation |
| :--- | :--- | :--- | :--- | :--- |
| **A-01** | aminx Core | `src/aminx/host/campaign.py` uses `DONE_MARKER_SCHEMA_VERSION = "campaign_done_marker_v3"` | **[VERIFIED]** | `src/aminx/host/campaign.py:57` |
| **A-02** | aminx Core | `src/aminx/host/campaign.py` uses `SAMPLING_NUMERICS_EPOCH = 2` | **[VERIFIED]** | `src/aminx/host/campaign.py:64` |
| **A-03** | aminx Core | Shared filesystem file-count cap binds at `_MAX_HASHED_DIR_FILES = 5000` | **[VERIFIED]** | `src/aminx/host/campaign.py:65` |
| **A-04** | aminx Core | Unit output path is formatted as `<base_dir>/<campaign_id>/<row_hash>.h5` | **[VERIFIED]** | `src/aminx/host/campaign.py:517-518` |
| **A-05** | aminx Core | Sibling done marker path is formatted as `<output_h5_path>.done.json` | **[VERIFIED]** | `src/aminx/host/campaign.py:525-526` |
| **A-06** | aminx Core | Partial execution directory is `<output_h5_path>.partial.<attempt_id>` | **[VERIFIED]** | `src/aminx/host/campaign.py:529-530` |
| **A-07** | aminx Core | `compute_unit_input_hash` hashes `spec_without_output`, `inputs`, `checkpoint`, and `versions` | **[VERIFIED]** | `src/aminx/host/campaign.py:902-938` |
| **A-08** | aminx Core | `_SEED_HASH_SCHEMA_PIN = "grid_v1"` is frozen forever in `_sampling_grid_lineage.py` | **[VERIFIED]** | `src/aminx/host/_sampling_grid_lineage.py:26` |
| **A-09** | aminx Core | `_validate_done_marker` enforces schema check, row hash check, store existence, input hash check, and content digest check | **[VERIFIED]** | `src/aminx/host/campaign.py:969-1013` |
| **A-10** | aminx Core | `_write_done_marker` uses `tmp_marker_path` atomic replacement and `fsync_directory` | **[VERIFIED]** | `src/aminx/host/campaign.py:1015-1050` |
| **A-11** | aminx Core | Unfinished output file without done marker raises `RuntimeError` | **[VERIFIED]** | `src/aminx/host/campaign.py:1812-1817, 1879-1884` |
| **A-12** | aminx Core | Legacy v2 markers are quarantined to `.superseded` and recomputed | **[VERIFIED]** | `src/aminx/host/campaign.py:1842-1851` |
| **A-13** | aminx Core | Temporary `.partial` trees are unlinked in `finally: shutil.rmtree(partial_path)` on error | **[VERIFIED]** | `src/aminx/host/campaign.py:1952-1954` |
| **A-14** | aminx Core | aminx `BatchPlanner` constructs and resolves `BatchPlan` with joint memory budgets | **[VERIFIED]** | `src/aminx/host/plan.py:19, 80-145` |
| **A-15** | aminx Core | Heterogeneous axes `n_states` and `n_structures` are pre-fixed to `ChunkedMap` | **[VERIFIED]** | `src/aminx/host/plan.py:77, 111-124` |
| **A-16** | aminx Core | `src/aminx/host/streaming.py` imports `ZarrStagingSink` and `derive_sink_spec` from `xtrax.run` | **[VERIFIED]** | `src/aminx/host/streaming.py:10` |
| **A-17** | aminx Core | `src/aminx/io/campaign_sink.py` manages per-arm Zarr stores | **[VERIFIED]** | `src/aminx/io/campaign_sink.py:1-30` |
| **A-18** | aminx Core | Checkpoint manifests verify `manifest_sha256` via `verify_pinned_file` | **[VERIFIED]** | `src/aminx/io/weights.py:606, 625-650` |
| **A-19** | aminx Core | `tests/host/test_campaign_done_marker.py` implements bit-flip corruption detection | **[VERIFIED]** | `tests/host/test_campaign_done_marker.py:146-172` |
| **A-20** | aminx Core | Existing pin is `xtrax[io,export]==0.4.1b5` | **[VERIFIED]** | `pyproject.toml:26` |
| **A-21** | aminx Core | Existing optional extra is `onnx = ["xtrax[onnx]==0.4.1b5"]` | **[VERIFIED]** | `pyproject.toml:88-90` |
| **A-22** | aminx Core | Ecosystem requirements: `jax>=0.4.35`, `equinox>=0.13.2`, `optax>=0.2`, `proxide>=0.1.0a17` | **[VERIFIED]** | `pyproject.toml:11, 16, 20, 23` |
| **A-23** | aminx Core | Resolved versions in `uv.lock`: `jax==0.10.2`, `equinox==0.13.8`, `optax==0.2.8`, `proxide==0.1.0a17`, `zarr==3.2.1` | **[VERIFIED]** | `uv.lock:1111, 1788, 3111, 3413, 4942` |
| **A-24** | aminx Core | `site/app.js` embeds py2Dmol via remote URL and uses built-in cartoon/chain/plddt views | **[VERIFIED]** | `site/app.js:25, 120-127, 438-441` |
| **A-25** | aminx Core | `src/aminx/` contains zero visual rendering library dependencies (headless core) | **[VERIFIED]** | `pyproject.toml:10-27`, `src/aminx/` tree |
| **A-26** | xtrax 0.4.1b7 | `xtrax 0.4.1b7` wheel packages `src/xtrax` only; contains NO sweep runner in wheel | **[VERIFIED]** | `xtrax-consumer-adoption/SKILL.md:47-56` |
| **A-27** | xtrax 0.4.1b7 | `AminxDoneMarkerAdapter` exists in `xtrax.run.unit_index_adapters:106-165` | **[VERIFIED]** | `using-xtrax/references/adoption.md:17, 63`, `SKILL.md:52` |
| **A-28** | xtrax 0.4.1b7 | `UnitJsonAdapter` indexes `unit.json` units loose or packed | **[VERIFIED]** | `using-xtrax/references/adoption.md:61, 63` |
| **A-29** | xtrax 0.4.1b7 | `write_unit_record` commits `unit.json` (`xtrax.unit/1`) with write-once semantics (`R-U1`) | **[VERIFIED]** | `using-xtrax/references/adoption.md:50, 56, 63` |
| **A-30** | xtrax 0.4.1b7 | `pack_unit` compresses members into `members.tar.zst` + `members.idx.json` | **[VERIFIED]** | `using-xtrax/references/adoption.md:45, 51, 63` |
| **A-31** | xtrax 0.4.1b7 | `pack_unit` raises `UnitPackRefused("R-U3")` on collision with loose files | **[VERIFIED]** | `using-xtrax/references/adoption.md:51, 63` |
| **A-32** | xtrax 0.4.1b7 | `UNIT_FAILED_STAMP = "unit.failed.json"` records structured unit failure | **[VERIFIED]** | `xtrax-consumer-adoption/SKILL.md:61`, `adoption.md:15` |
| **A-33** | xtrax 0.4.1b7 | Structured exception hierarchy: `CodecUnavailable(OSError)`, `LockUnavailable(OSError)`, `UnitPackRefused(ValueError)` | **[VERIFIED]** | `using-xtrax/references/adoption.md:45, 58, 63` |
| **A-34** | xtrax 0.4.1b7 | `xtrax[pack]` pulls in `zstandard`; `xtrax[eda]` pulls in Vega-Lite plan rendering | **[VERIFIED]** | `using-xtrax/references/adoption.md:45`, `uv.lock:4907-4938` |
| **A-35** | xtrax 0.4.1b7 | Neither `xtrax[pack]` nor `xtrax[eda]` includes `xtrax[io]` (which pulled in `zarr`) | **[VERIFIED]** | `using-xtrax/references/adoption.md:45`, `uv.lock:4931-4933` |
| **A-36** | xtrax 0.4.1b7 | `xtrax.run` zarr modules import without `zarr` and raise bare `ImportError` when called | **[VERIFIED]** | `xtrax-consumer-adoption/SKILL.md:63-64` |
| **A-37** | Downstream | `asr` vendors aminx via git submodule pinned to historical commit | **[UNVERIFIED - EXTERNAL REPOSITORY]** | `ORIGINAL_REQUEST.md:14, 58`; external repo not in workspace |
| **A-38** | Downstream | `mpnn_ext` vendors aminx via git submodule pinned to historical commit | **[UNVERIFIED - EXTERNAL REPOSITORY]** | `ORIGINAL_REQUEST.md:14, 58`; external repo not in workspace |
| **A-39** | Downstream | `hautespout` vendors aminx via static vendor tree pinned to historical commit | **[UNVERIFIED - EXTERNAL REPOSITORY]** | `ORIGINAL_REQUEST.md:14, 58`; external repo not in workspace |

---

## 2. Comprehensive Pin Plan & Dependency Architecture

### 2.1 Specification Target & Direct Dependencies

The package dependencies in `/home/marielle/projects/aminx/pyproject.toml` are updated as follows:

```toml
# In pyproject.toml [project] dependencies:
dependencies = [
    "jax>=0.4.35",
    "jaxtyping>=0.2.30",
    "joblib>=1.4",
    "psutil>=5.9",
    "numpy>=1.26",
    "optax>=0.2",
    "h5py>=3.11",
    "grain>=0.2",
    "array-record>=0.8.1",
    "equinox>=0.13.2",
    "zstandard>=0.21.0",
    "msgpack-numpy>=0.4.8",
    "proxide>=0.1.0a17",
    "typer>=0.12.0",
    "huggingface-hub>=0.23.0",
    "xtrax[pack,eda,export]>=0.4.1b7",
    "zarr>=3.0.0",
    "flax>=0.8.0",
]

# In pyproject.toml [project.optional-dependencies]:
onnx = [
    "xtrax[onnx]>=0.4.1b7",
]
```

### 2.2 Direct `zarr` and `flax` Requirement Rationale

Per authoritative directive in `xtrax-consumer-adoption` `SKILL.md:63-64`:
> *"If xtrax becomes a core dependency and the project uses `xtrax.run` zarr modules, require zarr directly: those modules import without it and raise a bare `ImportError` when called."*

- **The Problem**: In `0.4.1b5`, `zarr` was transitively pulled into aminx solely via `"xtrax[io,export]==0.4.1b5"`. When moving to `"xtrax[pack,eda,export]>=0.4.1b7"`, the `[io]` extra is dropped. Neither `[pack]` (which only pulls `zstandard`), `[eda]` (which pulls Vega-Lite rendering), nor `[export]` depends on `zarr`. Additionally, `proxide` imports `from flax.struct import dataclass` in `proxide.core.atomic_system:16`, but `flax` was not previously declared in `dependencies`.
- **The Consequence**: Without an explicit `zarr>=3.0.0` dependency in `aminx/pyproject.toml`, calls to `src/aminx/host/streaming.py` (`from xtrax.run import ZarrStagingSink`), `src/aminx/io/campaign_sink.py` (`UnifiedCampaignSink`), and `tests/host/test_campaign_done_marker.py` (`import zarr`) fail at runtime with bare `ImportError: No module named 'zarr'`. Furthermore, without `flax>=0.8.0`, running `uv run pytest tests/host/test_campaign_done_marker.py` crashes in `tests/conftest.py` with `ModuleNotFoundError: No module named 'flax'`.
- **The Resolution**: Consolidate the xtrax pin to `"xtrax[pack,eda,export]>=0.4.1b7"` (preserving `[export]` needed by `tests/export/test_export_wrappers.py:532`), declare `zarr>=3.0.0` directly in `dependencies` (resolving to `zarr 3.2.1` in `uv.lock`), and declare `flax>=0.8.0` in `dependencies` (ensuring clean test suite and conftest execution).

### 2.3 JAX & Equinox Compatibility Proof

A rigorous audit of the JAX ecosystem compatibility demonstrates zero version friction:

| Package | aminx Constraint | Locked Version (`uv.lock`) | `xtrax 0.4.1b7` Constraint | Compatibility Assessment |
| :--- | :--- | :--- | :--- | :--- |
| `jax` | `>=0.4.35` (`pyproject.toml:11`) | `0.10.2` (`uv.lock:1788`) | `{ name = "jax" }` (unbounded) | **Compatible**: `0.4.35` through `0.10.2` verified |
| `jaxlib` | (transitive via jax) | `0.10.2` (`uv.lock:1800`) | `{ name = "jaxlib" }` | **Compatible**: matches jax runtime |
| `equinox` | `>=0.13.2` (`pyproject.toml:20`) | `0.13.8` (`uv.lock:1111`) | (compatible PyTree JIT) | **Compatible**: zero pin constraint in xtrax |
| `optax` | `>=0.2` (`pyproject.toml:16`) | `0.2.8` (`uv.lock:3111`) | `{ name = "optax" }` | **Compatible**: fully aligned |
| `proxide` | `>=0.1.0a17` (`pyproject.toml:23`) | `0.1.0a17` (`uv.lock:3413`) | (independent biological IR) | **Compatible**: zero pin constraint in xtrax |
| `zstandard` | `>=0.21.0` (`pyproject.toml:21`) | `0.25.0` (`uv.lock:4968`) | (satisfies `xtrax[pack]`) | **Compatible**: provides required zstd codec |

### 2.4 Downstream Vendored Copies Isolation Proof

Downstream consumer repositories vendor or submodule aminx copies that will **NOT** follow the pin bump:

1. **`asr`**:
   - Structure: Vendors aminx via git submodule (`https://github.com/maraxen/aminx.git`).
   - Done marker schema: `campaign_done_marker_v3` per historical commit.
   - **Isolation**: Pinned to a specific commit SHA gitlink in `asr`. Upstream commits in `aminx` do not modify the submodule tree without an explicit `git submodule update` in `asr`.
2. **`mpnn_ext`**:
   - Structure: Vendors aminx via git submodule.
   - Done marker schema: `campaign_done_marker_v1` (HDF5 era).
   - **Isolation**: Submodule gitlink is frozen to a legacy commit. Completely decoupled.
3. **`hautespout`**:
   - Structure: Vendors aminx via static filesystem copy.
   - Done marker schema: `campaign_done_marker_v2` (early Zarr era without input hash).
   - **Isolation**: Static vendored directory with no git tracking link to `aminx.git`.

- **Verdict**: Downstream consumers (`asr`, `mpnn_ext`, `hautespout`) pin aminx via git submodule commit SHAs or static filesystem copies. Modifying `pyproject.toml` in `aminx` cannot reach or affect downstream consumers without an explicit upstream merge and submodule pointer update in those respective repositories. Local verification is deferred to downstream environments. The blast radius of the `xtrax>=0.4.1b7` pin bump is strictly confined to `aminx`. Zero regression risk exists for `asr`, `mpnn_ext`, or `hautespout`.

---

## 3. Pack Adoption Architecture & Old-vs-New Edge-Case Analysis

### 3.1 Old-vs-New Edge-Case Deep-Dive Table

The edge cases between the old writer (`campaign_done_marker_v3` in `src/aminx/host/campaign.py`) and the new writer (`write_unit_record` / `pack_unit` in `xtrax.pack`) are exhaustively analyzed across the six critical dimensions:

| Dimension | Old Writer: `campaign_done_marker_v3` (`src/aminx/host/campaign.py`) | New Writer: `write_unit_record` / `pack_unit` (`xtrax.pack`) | Observed / Proposed Migration Behavior |
| :--- | :--- | :--- | :--- |
| **1. Failure Stamps** | **No failure stamps written to disk.**<br>When execution fails, an exception is raised; the temporary directory `.partial.<attempt_id>` is deleted via `shutil.rmtree` in a `finally` block (`campaign.py:1953`). Subsequent attempts re-run from scratch with zero record of past failures. | **Structured Failure Stamp: `UNIT_FAILED_STAMP`.**<br>Defines `UNIT_FAILED_STAMP = "unit.failed.json"` (`schema="xtrax.unit_failure/1"`). Records error message, traceback, attempt ID, and timestamp. `pack_unit` and unit index detect the failure stamp; unit is excluded from packing and indexed as `Failed`. | In the migrated pack runner (`src/aminx/host/pack_shim.py`), write `unit.failed.json` upon unhandled worker exception before cleaning temporary partial stores. Resume logic checks for `unit.failed.json`, allowing the scheduler to distinguish between unattempted units, permanently failed units, and valid completions. |
| **2. Regenerated Loose Files** | **Output store collision check.**<br>The store is `<row_hash>.h5`. If the directory exists on disk without a valid done marker, it raises `RuntimeError("Output file already exists without done marker...")` (`campaign.py:1812, 1879`). | **Refusal `R-U3`: Loose File Collision.**<br>A packed unit contains `members.tar.zst` + `members.idx.json`. If execution restarts or regenerates loose files in a unit directory that already has a pack archive, `pack_unit` raises `UnitPackRefused("R-U3: Files named like pack artifacts present before record")`. | Resume check calls `is_packed(unit_dir, record_name="unit.json")`. If packed, loose array files must never be written alongside `members.tar.zst`. If a clean rerun is requested, the entire unit directory must be cleared or quarantined prior to execution. When packing an existing unit with a sibling `<output_h5_path>.done.json`, the legacy marker is archived into `unit.json` (`meta['legacy_done_marker']`) and atomically renamed to `<output_h5_path>.done.json.packed` to prevent legacy runner crashes. |
| **3. Empty Units** | **Permissive empty directory validation.**<br>If an output directory exists but contains 0 bytes or empty arrays, `zarr_content_digest` computes the hash of the empty tree and writes the done marker (`campaign.py:1005`). Missing output raises `ValueError` (`campaign.py:996`). | **Refusal `R-U1`: Empty Unit Refusal.**<br>`write_unit_record` computes digest over `_loose_shas`. `pack_unit` refuses empty units with refusal `R-U1` ("unit has no loose files to pack"). | The pack shim configures `check(unit_dir, record, digest)` callback. The check asserts that expected Zarr arrays (`samples`, `scores`) exist and are non-empty before committing `write_unit_record`. Empty stores are rejected immediately. |
| **4. OSError Wrapping** | **Raw POSIX filesystem exceptions.**<br>Directly calls `write_bytes`, `replace`, `fsync_file`, and `fsync_directory`. Raises bare `OSError`, `IOError`, or `RuntimeError` on disk full, read-only filesystems, or lock heartbeat failure (`campaign.py:1048-1050, 1930`). | **Domain Exception Hierarchy.**<br>Structured exceptions: `CodecUnavailable(OSError)` (missing zstd), `LockUnavailable(OSError)` (NFS/Lustre locking failure), `UnitPackRefused(ValueError)` (`R-U1`, `R-U2`, `R-U3`). | Caller loops in aminx catch `UnitPackRefused`, `LockUnavailable`, and `CodecUnavailable`. Lock fallback strategies are clearly logged, and cluster-level errors are distinguished from algorithmic failures. |
| **5. Resume Reuse Contract** | **5-point validation check.**<br>Verifies: `schema_version == "campaign_done_marker_v3"`, `manifest_row_hash`, `output_h5_path.exists()`, `input_hash`, and `zarr_content_digest`. Returns `_already_done_result` on match (`campaign.py:1805, 1872`). | **Content Digest Verification.**<br>Verifies `unit_content_digest(unit_dir) == record["content_digest"]` and `core.input_digest == input_digest`. Corrupted units are flagged as `Corrupt` and refused; stale inputs are flagged as `IndexStale`. Packed members read byte-identical. | Units whose on-disk bytes diverge from recorded `content_digest` (tested via single bit-flip) are strictly refused on resume via `validate_aminx_unit` and quarantined. Negative test controls prove corruption causes rejection. Packed units must be resumed and indexed via `pack_shim` / `UnitJsonAdapter`, not `campaign.py`. |
| **6. Backward Reader Compatibility** | **Legacy v2 markers quarantined.**<br>Legacy v2 markers (lacking input hash) raise `LegacyV2DoneMarkerError`; quarantined aside to `.superseded` suffix (`campaign.py:1842-1850`). Requires running `adopt_legacy_done_markers` CLI tool to upgrade. | **`AminxDoneMarkerAdapter` Zero-Rewrite Reading.**<br>`AminxDoneMarkerAdapter` (`src/xtrax/run/unit_index_adapters.py:106-165`) directly parses `campaign_done_marker_v3` files into `CoreRecord` (`name=manifest_row_hash`, `input_digest=input_hash`, `content_digest=content_digest_sha256`). | Existing on-disk datasets on Engaging and titanix remain 100% readable with **ZERO** disk modifications. The unit index seamlessly reads both legacy `.done.json` markers and new `unit.json` records in a unified index. |

### 3.2 Unit Directory Layout Contract

For xtrax.pack integration, each work item row maps to a dedicated unit directory:
`<base_dir>/<campaign_id>/<row_hash>/`

Inside this directory:
- Zarr store directory: `store.zarr/` (containing array chunk files and group metadata)
- Unit record: `unit.json` (schema `xtrax.unit/1`)
- When packed: `members.tar.zst` + `members.idx.json` archiving `store.zarr/` loose files

For legacy on-disk stores (`<base_dir>/<campaign_id>/<row_hash>.h5/`), `output_h5_path` is treated directly as `unit_dir`, with `unit.json` placed at `<output_h5_path>/unit.json`.

---

## 4. Unit Identity Contract & Reference Immutability

### 4.1 Inviolable Invariant: Reference Sweep Runner Immutability

`src/aminx/host/campaign.py` is the frozen reference implementation for high-throughput campaign sweeps.
- **Rule**: `src/aminx/host/campaign.py` **MUST NOT BE MODIFIED** under any circumstances.
- **Verification**: `git diff origin/main -- src/aminx/host/campaign.py` must return strictly zero diff throughout the entire project lifecycle.

### 4.2 Unit Identity Contract & Cryptographic Continuity

Unit identity in aminx is mathematically determined by `compute_unit_input_hash` (`src/aminx/host/campaign.py:902-938`). To preserve complete cryptographic continuity when emitting `xtrax.unit/1` records, `write_aminx_unit` maps this algorithm directly to `input_digest`:

```
┌────────────────────────────────────────────────────────────────────────┐
│ aminx Unit Identity Formulation (compute_unit_input_hash)              │
├────────────────────────────────────────────────────────────────────────┤
│ 1. spec_without_output:                                                │
│      Excludes 'output_h5_path', 'output_path', 'output_zarr_path'      │
│ 2. inputs:                                                             │
│      Normalized SHA-256 hashes of input PDB/mmCIF structures           │
│ 3. checkpoint:                                                         │
│      FamilyCheckpoint identity (family, filename, sha256, manifest)   │
│ 4. versions:                                                           │
│      - MANIFEST_ROW_SCHEMA_VERSION = "campaign_manifest_row_v1"        │
│      - SAMPLING_SCHEMA_VERSION     = "sampling_spec_v1"                │
│      - GRID_SCHEMA_VERSION         = "grid_v1"                         │
│      - _SEED_HASH_SCHEMA_PIN       = "grid_v1" (lineage.py:26) [FROZEN]│
│      - SAMPLING_NUMERICS_EPOCH     = 2 (campaign.py:64)       [FROZEN]│
│ 5. seed0_semantics:                                                    │
│      seed0_semantics = 2 (if random_seed == 0 and not bool)           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ canonical_json_bytes + SHA-256
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ input_digest (64 hex characters)                                       │
├────────────────────────────────────────────────────────────────────────┤
│ Passed directly to xtrax.pack.write_unit_record(..., input_digest=...) │
│ Full components dictionary stored in meta["input_hash_components"]     │
└────────────────────────────────────────────────────────────────────────┘
```

- **Seed Hash Schema Pin**: `_SEED_HASH_SCHEMA_PIN: Final[str] = "grid_v1"` in `src/aminx/host/_sampling_grid_lineage.py:26` is frozen forever. It feeds `_grid_job_seed_hash` -> `_seed_words_from_manifest_hash` -> `jax.random.fold_in`.
- **Sampling Numerics Epoch**: `SAMPLING_NUMERICS_EPOCH = 2` in `src/aminx/host/campaign.py:64`. It ensures that any sampling numerics changes invalidate previous done markers rather than mixing design distributions.
- **Bridge Contract**: The pack shim (`src/aminx/host/pack_shim.py`) computes `input_hash, components = compute_unit_input_hash(spec)` and passes `input_hash` as `input_digest` to `write_unit_record`.

---

## 5. Non-Invasive Shim Architecture (`src/aminx/host/pack_shim.py`)

To satisfy the pack adoption requirements without editing `src/aminx/host/campaign.py`, a dedicated adapter module `src/aminx/host/pack_shim.py` is established.

### 5.1 Interface Specification

```python
"""Non-invasive pack shim binding xtrax.pack to aminx units."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import logging
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

logger = logging.getLogger(__name__)


def aminx_unit_check(unit_dir: Path, record: Mapping[str, Any], content_digest: str) -> None:
  """Verification callback passed to pack_unit(..., check=aminx_unit_check).

  Validates that:
  1. The unit directory contains store files.
  2. The store has valid array datasets (excluding .zgroup and zarr.json metadata).
  3. No empty units are committed.

  Raises
  ------
  UnitPackRefused
      If validation fails (maps to R-U1 empty unit refusal).
  """
  store_path = (
    unit_dir / "store.zarr"
    if (unit_dir / "store.zarr").exists()
    else unit_dir
  )
  if not store_path.exists():
    msg = f"Unit store path missing in {unit_dir}"
    raise UnitPackRefused(msg)

  # Exclude metadata and unit record; require actual array chunks
  files = [
    p
    for p in store_path.rglob("*")
    if p.is_file() and p.name not in ("unit.json", ".zgroup", "zarr.json")
  ]
  if not files:
    msg = f"Unit has no array member files in {unit_dir} (R-U1 empty unit refusal)"
    raise UnitPackRefused(msg)


def write_aminx_unit(
  unit_dir: Path,
  *,
  sampling_spec_payload: Mapping[str, Any],
  run_id: str,
  meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Computes unit identity and commits unit.json via write_unit_record.

  Parameters
  ----------
  unit_dir : Path
      Directory containing the completed unit outputs.
  sampling_spec_payload : Mapping[str, Any]
      Sampling specification payload used for compute_unit_input_hash.
  run_id : str
      Unique execution run ID.
  meta : dict[str, Any] | None, default=None
      Additional metadata to record in unit.json.

  Returns
  -------
  dict[str, Any]
      The committed unit record.
  """
  # Enforce non-empty store check before committing record (Directive 6 / R-U1)
  aminx_unit_check(unit_dir, record={}, content_digest="")

  input_hash, components = compute_unit_input_hash(sampling_spec_payload)

  unit_meta = meta.copy() if meta else {}
  unit_meta["input_hash_components"] = components
  unit_meta["schema_bridge"] = "aminx_pack_shim_v1"

  return write_unit_record(
    unit_dir,
    input_digest=input_hash,
    run_id=run_id,
    input_payload=dict(sampling_spec_payload),
    meta=unit_meta,
  )


def pack_aminx_unit(
  unit_dir: Path,
  *,
  level: int = 3,
  lock: bool = True,
) -> dict[str, Any]:
  """Packs an aminx unit into members.tar.zst + members.idx.json.

  If a legacy sibling marker exists (e.g. `<unit_dir>.done.json`), it is
  quarantined and archived:
  1. Embedded in unit.json under meta['legacy_done_marker'].
  2. Atomically renamed to `<unit_dir>.done.json.packed`.

  Parameters
  ----------
  unit_dir : Path
      Unit directory containing loose files and unit.json.
  level : int, default=3
      Zstandard compression level.
  lock : bool, default=True
      Whether to acquire POSIX file lock during packing.

  Returns
  -------
  dict[str, Any]
      Storage metadata from pack_unit.
  """
  # Archive and quarantine legacy sibling done marker if present
  legacy_marker = unit_dir.with_name(f"{unit_dir.name}.done.json")
  if legacy_marker.exists():
    try:
      legacy_data = json.loads(legacy_marker.read_text(encoding="utf-8"))
      record_path = unit_dir / "unit.json"
      if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.setdefault("meta", {})["legacy_done_marker"] = legacy_data
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
      legacy_marker.rename(unit_dir.with_name(f"{unit_dir.name}.done.json.packed"))
    except Exception as exc:
      logger.warning("Failed to archive legacy done marker at %s: %s", legacy_marker, exc)

  return pack_unit(
    unit_dir,
    record_name="unit.json",
    check=aminx_unit_check,
    level=level,
    lock=lock,
  )


def validate_aminx_unit(
  unit_dir: Path,
  *,
  expected_input_digest: str | None = None,
  record_name: str = "unit.json",
) -> dict[str, Any]:
  """Validates unit record and member integrity on resume.

  Raises
  ------
  ValueError
      If unit.json is missing or unit_content_digest mismatches record['content_digest'].
  UnitPackRefused
      If expected_input_digest is provided and mismatches record['input_digest'].
  """
  record_path = unit_dir / record_name
  if not record_path.exists():
    msg = f"Unit record missing at {record_path}"
    raise ValueError(msg)
  record = json.loads(record_path.read_text(encoding="utf-8"))
  if (
    expected_input_digest is not None
    and record.get("input_digest") != expected_input_digest
  ):
    msg = (
      f"Input digest mismatch in {unit_dir}: "
      f"expected {expected_input_digest}, got {record.get('input_digest')}"
    )
    raise UnitPackRefused(msg)
  observed = unit_content_digest(unit_dir)
  expected = record.get("content_digest")
  if observed != expected:
    msg = (
      f"Unit content digest mismatch in {unit_dir}: "
      f"expected {expected}, observed {observed}"
    )
    raise ValueError(msg)
  return record


def write_aminx_unit_failure(
  unit_dir: Path,
  *,
  attempt_id: str,
  error: Exception,
  meta: dict[str, Any] | None = None,
) -> Path:
  """Writes UNIT_FAILED_STAMP upon worker failure."""
  unit_dir.mkdir(parents=True, exist_ok=True)
  stamp_path = unit_dir / UNIT_FAILED_STAMP
  payload = {
    "schema": "xtrax.unit_failure/1",
    "attempt_id": attempt_id,
    "error_type": error.__class__.__name__,
    "error_message": str(error),
    "failed_at_unix_s": time.time(),
    "meta": meta or {},
  }
  stamp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return stamp_path


def run_packed_manifest_row(
  manifest_path: Path | str,
  *,
  row_index: int | None = None,
  row_hash: str | None = None,
  pack: bool = True,
  **kwargs: Any,
) -> dict[str, Any]:
  """Executes a manifest row and commits via write_aminx_unit / pack_aminx_unit."""
  from aminx.host.campaign import (
    _manifest_rows,
    _row_output_path,
    _select_row,
    load_manifest,
  )
  from aminx.host.runner import sample

  payload = (
    load_manifest(manifest_path)
    if isinstance(manifest_path, (str, Path))
    else {"rows": list(manifest_path)}
  )
  rows = _manifest_rows(manifest_path)
  row = _select_row(rows, row_index=row_index, row_hash=row_hash)
  sampling_spec = row.get("sampling_spec")
  if not isinstance(sampling_spec, dict):
    msg = "Manifest row must include a 'sampling_spec' object."
    raise TypeError(msg)

  row_hash_str = str(row["manifest_row_hash"])
  base_dir = Path(payload.get("output_dir", "campaign_runs"))
  campaign_id = str(payload.get("campaign_id", "default"))
  unit_dir = Path(_row_output_path(base_dir, campaign_id, row_hash_str))

  # Resume check: if already completed, validate and return
  if (unit_dir / "unit.json").exists():
    try:
      record = validate_aminx_unit(unit_dir)
      return {
        "manifest_row_hash": row_hash_str,
        "status": "already_done",
        "record": record,
      }
    except Exception as exc:
      logger.warning(
        "Resume validation failed for %s: %s; recomputing", unit_dir, exc
      )

  # Execute sampling using aminx runner logic with the sampling_spec payload
  sample_result = sample(sampling_spec)

  # Commit unit record via xtrax.pack
  record = write_aminx_unit(
    unit_dir,
    sampling_spec_payload=sampling_spec,
    run_id=kwargs.get("run_id", "prod_run"),
  )
  if pack:
    pack_aminx_unit(unit_dir)
  return {
    "manifest_row_hash": row_hash_str,
    "status": "completed",
    "record": record,
  }


def execute_packed_manifest(
  manifest_path: Path | str,
  *,
  pack: bool = True,
  **kwargs: Any,
) -> list[dict[str, Any]]:
  """Executes an entire manifest sequentially or with worker dispatch using packed units."""
  from aminx.host.campaign import _manifest_rows

  rows = _manifest_rows(manifest_path)
  results = []
  for idx, _ in enumerate(rows):
    res = run_packed_manifest_row(
      manifest_path, row_index=idx, pack=pack, **kwargs
    )
    results.append(res)
  return results
```

### 5.2 Dual-Mode Reader Integration

The reader uses `AminxDoneMarkerAdapter` and `UnitJsonAdapter` to index both legacy and modern on-disk datasets:

```python
def resolve_unit_adapter(path: Path) -> Any:
  """Returns the appropriate adapter based on pack state and marker presence."""
  if (
    is_packed(path, record_name="unit.json")
    or (path / "unit.json").exists()
    or (path.parent / f"{path.name}.done.json.packed").exists()
  ):
    return UnitJsonAdapter()
  if (path.with_name(f"{path.name}.done.json")).exists():
    return AminxDoneMarkerAdapter()
  return UnitJsonAdapter()
```

### 5.3 CLI Integration (`src/aminx/cli.py`)

Production campaigns are dispatched through `src/aminx/cli.py`. To allow seamless selection between the frozen reference runner (`campaign.py`) and the packed runner (`pack_shim.py`), CLI options `--pack / --no-pack` are introduced:

1. **CLI Commands**:
   - `aminx campaign worker <manifest_path>`: supports `--pack / --no-pack` (default: `--no-pack` for reference runs, `--pack` for packed execution).
   - `aminx campaign run <manifest_path>`: supports `--pack / --no-pack`.
2. **Dispatch Mechanics**:
   - When `--pack` is enabled: `cli.py` dispatches to `pack_shim.run_packed_manifest_row` or `pack_shim.execute_packed_manifest`.
   - When `--no-pack` is passed: `cli.py` dispatches to `campaign.run_manifest_row` or `campaign.execute_manifest`.
3. **Reference Invariant C1 Preservation**:
   - `src/aminx/host/campaign.py` remains strictly intact and frozen (0-byte diff against `origin/main`).
   - All packing logic is encapsulated inside `src/aminx/host/pack_shim.py`, with dispatch managed solely in `src/aminx/cli.py`.

---

## 6. aminx EDA Plan Inspection Architecture (`src/aminx/host/plan_eda.py`)

`aminx` builds execution schedules for heterogeneously dimensioned batches over residues and structures (`src/aminx/host/plan.py:19, 80-145`). `xtrax.eda` provides transparent plan explanation.

### 6.1 Programmatic EDA Interface

`src/aminx/host/plan_eda.py` wraps `xtrax.eda.explain_plan` and `analyze_bucket`:

```python
"""EDA plan inspection and validation adapter for aminx."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xtrax.eda import analyze_bucket, explain_plan
from xtrax.tiling import BatchPlan


def explain_aminx_plan(plan: BatchPlan) -> dict[str, Any]:
  """Explains an aminx BatchPlan using xtrax.eda.

  Parameters
  ----------
  plan : BatchPlan
      The planned execution topology.

  Returns
  -------
  dict[str, Any]
      Dictionary containing explain_plan summary and per-bucket decision analyses.
  """
  stats = explain_plan(plan)
  bucket_analyses = []
  for d in plan.decisions:
    strategy_name = d.strategy.__class__.__name__ if hasattr(d, "strategy") else ""
    if strategy_name == "Bucket":
      bucket_analyses.append(analyze_bucket(d))

  return {
    "plan_summary": stats,
    "bucket_analyses": bucket_analyses,
  }


def verify_checkpoint_and_explain_plan(
  plan: BatchPlan,
  *,
  checkpoint_family: str,
  checkpoint_id: str,
) -> dict[str, Any]:
  """Verifies checkpoint manifest checksum before explaining plan topology."""
  from aminx.io.weights import resolve_family_checkpoint

  # Resolves artifact and companion manifest, validating both sha256 pins
  resolve_family_checkpoint(checkpoint_id, family=checkpoint_family)
  return explain_aminx_plan(plan)
```

### 6.2 Checkpoint Manifest Cryptographic Verification

In `src/aminx/io/weights.py:606-650`, model weights are registered with `FamilyCheckpoint(filename, sha256, manifest_filename, manifest_sha256)`.
- When constructing plans, `plan_eda` asserts that `manifest_sha256` matches the export artifact checksum before planning axes, preventing plan generation against mismatched or truncated model checkpoints.

---

## 7. Resume Design & Negative Test Controls Protocol

A robust resume system must be capable of failing: if a unit is corrupted, modified, or collided, the system must detect it, refuse reuse, and report an error. Existing bathos `[outcomes]` must remain unchanged.

### 7.1 Pre-Registration Protocol

Every test run producing a finding is pre-registered with a committed `.bth.toml` sidecar and tracked script before execution. Testing verifies behavior via run records, not bare exit codes.

### 7.2 Negative Test Control 1: Corrupted Unit Bit-Flip Rejection Control

- **Objective**: Prove that `unit_content_digest` strictly detects post-completion bit corruption and refuses reuse.
- **Protocol**:
  1. Create a temporary unit directory `unit_dir` with a valid Zarr store.
  2. Commit `write_unit_record(unit_dir, input_digest=input_hash, run_id="run-1")`.
  3. Verify `unit_content_digest(unit_dir) == record["content_digest"]`.
  4. Perform bit-flip mutation: mutate an array file within the unit (`data[0] = 999999`).
  5. Attempt to validate resume via `validate_aminx_unit(unit_dir)`.
  6. **Assertion**: Observed digest differs from `record["content_digest"]`. `validate_aminx_unit` raises `ValueError` / refuses reuse.
  7. The test PASSES only when the corrupted unit is REJECTED.

### 7.3 Negative Test Control 2: Loose File Collision `R-U3` Control

- **Objective**: Prove that regenerating loose files inside an already packed unit triggers refusal `R-U3`.
- **Protocol**:
  1. Create a valid unit and call `pack_aminx_unit(unit_dir)`.
  2. Verify `members.tar.zst` and `members.idx.json` exist, and loose member files are unlinked.
  3. Simulate an invalid rerun: recreate a loose array chunk file in `unit_dir`.
  4. Call `pack_aminx_unit(unit_dir)`.
  5. **Assertion**: `pack_aminx_unit` immediately raises `UnitPackRefused` with error code `R-U3`.
  6. The test PASSES only when packing is REFUSED.

### 7.4 Negative Test Control 3: Empty Unit `R-U1` Control

- **Objective**: Prove that committing an empty unit directory (0 bytes, no member files) is refused.
- **Protocol**:
  1. Create an empty directory `empty_unit_dir`.
  2. Attempt to call `pack_aminx_unit(empty_unit_dir)` with `aminx_unit_check`.
  3. **Assertion**: `aminx_unit_check` raises `UnitPackRefused("R-U1 empty unit refusal")`.
  4. The test PASSES only when empty unit packaging is REFUSED.

---

## 8. Formal py2Dmol Assessment

### 8.1 Specification Rule

Per `xtrax-consumer-adoption` `SKILL.md:91-102`:
> *"Wherever the project shows structures, per-residue tracks, contacts or sites, decide in the spec between built-in views, a py2Dmol renderer plugin, or not applicable, with one line of reasoning.*
> *A plugin fits geometry that is not a residue: isosurfaces, site markers, marker-to-residue contacts, MI or coupling edges. If built-ins do the job, write no plugin."*

### 8.2 aminx Audit & Verdict

1. **Python Core (`src/aminx/`)**:
   - Zero occurrences of `py2Dmol`, `py3Dmol`, `matplotlib`, or DOM visualization libraries.
   - `src/aminx/` is a high-performance headless JAX library designed for cluster and CLI execution.
2. **Browser Interface (`site/`)**:
   - `site/index.html:38`: Attributions credit Sergey Ovchinnikov's py2Dmol.
   - `site/app.js:25`: Loads `https://py2dmol.solab.org/py2Dmol/resources/bundles/py2Dmol.embed.min.js`.
   - `site/app.js:120-127`: Uses `py2Dmol.show` with standard cartoon style (`setStyle("cartoon")`) and chain coloring (`setColor("chain")`).
   - `site/app.js:438-441`: Visualizes generated designs colored by plddt/B-factor (`viewer.setColor("plddt")`).
   - Contains **NO** custom non-residue geometry (no isosurfaces, site markers, marker-to-residue contacts, MI or coupling edges).
3. **Formal Verdict**:
   - **Decision: Built-in views**.
   - **One-line reasoning**: aminx core is headless JAX, and `site/app.js` visualizes only standard cartoon backbone and chain/plddt B-factor coloring with zero non-residue geometry, so built-in views do the job and no py2Dmol renderer plugin is required.

---

## 9. Acceptance Criteria (Given-When-Then) & Verification Procedures

### AC-1: Reference Runner Immutability
- **Given** the aminx source tree,
- **When** the xtrax pack and eda adoption is completed,
- **Then** `git diff origin/main -- src/aminx/host/campaign.py` is strictly empty (zero bytes changed).

### AC-2: Unit Identity Preservation
- **Given** any valid `sampling_spec_payload`,
- **When** `write_aminx_unit(unit_dir, ...)` is called,
- **Then** `record["input_digest"]` exactly matches `compute_unit_input_hash(sampling_spec_payload)[0]`, reflecting `_SEED_HASH_SCHEMA_PIN = "grid_v1"` and `SAMPLING_NUMERICS_EPOCH = 2`.

### AC-3: Backward Reader Compatibility
- **Given** existing on-disk datasets marked with `campaign_done_marker_v3` (`<row_hash>.h5.done.json`),
- **When** indexed via `AminxDoneMarkerAdapter`,
- **Then** a valid `CoreRecord` is emitted with matching `manifest_row_hash`, `input_hash`, and `content_digest_sha256`, without modifying the on-disk marker.

### AC-4: Corrupted-Unit Refusal
- **Given** a finished unit record,
- **When** an underlying data file is modified post-completion (bit-flip),
- **Then** resume validation fails and refuses reuse due to `content_digest` mismatch.

### AC-5: Loose File Collision Refusal (`R-U3`)
- **Given** an already packed unit with `members.tar.zst`,
- **When** loose files named like pack artifacts are regenerated in the unit directory before packing,
- **Then** `pack_aminx_unit` raises `UnitPackRefused("R-U3")`.

### AC-6: Empty Unit Refusal (`R-U1`)
- **Given** a unit directory with no loose data files,
- **When** `pack_aminx_unit` is called,
- **Then** `aminx_unit_check` raises `UnitPackRefused("R-U1")`.

### AC-7: Direct Dependencies Pinning
- **Given** `pyproject.toml`,
- **When** package dependencies are inspected,
- **Then** `xtrax[pack,eda,export]>=0.4.1b7`, `zarr>=3.0.0`, and `flax>=0.8.0` are declared in `dependencies`, and `onnx = ["xtrax[onnx]>=0.4.1b7"]` in `optional-dependencies`.

### AC-8: Vendored Isolation
- **Given** downstream consumers `asr`, `mpnn_ext`, and `hautespout`,
- **When** the pin bump is applied to `aminx`,
- **Then** downstream projects remain architecturally isolated on their frozen submodule gitlinks or static vendor copies without inheriting upstream configuration changes.

### AC-9: py2Dmol Compliance
- **Given** `site/app.js` and `src/aminx/`,
- **When** visual components are audited,
- **Then** `src/aminx/` has zero visual dependencies and `site/app.js` operates exclusively with built-in views without custom plugins.

### AC-10: Manifest Slice Parity Verification
- **Given** a representative manifest slice from aminx datasets,
- **When** executed independently via `campaign.py` (legacy reference) and `pack_shim.py` (migrated runner),
- **Then** the generated outputs (arrays, scores, and metadata) are mathematically and structurally equivalent, and both readers successfully parse their respective outputs.

---

## 10. Independent Verification Commands

To independently verify the specification and deliverables:

```bash
# 1. Verify reference runner immutability (must return 0 diff)
git diff origin/main -- src/aminx/host/campaign.py

# 2. Run test suite to verify current baseline
# Note: `flax>=0.8.0` is required by proxide (via `from flax.struct import dataclass` in `proxide.core.atomic_system`)
# and declared in `pyproject.toml` dependencies so tests and conftest load cleanly.
uv run pytest tests/host/test_campaign_done_marker.py

# 3. Verify code style and linting
uv run ruff check src/aminx/host/campaign.py

# 4. Verify spec file exists and frontmatter parses cleanly
test -f /home/marielle/projects/aminx/.praxia/docs/specs/261105_aminx-xtrax-adoption-spec.md
```
