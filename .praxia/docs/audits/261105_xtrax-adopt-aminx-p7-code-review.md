---
title: "P7 Code Review Audit: aminx xtrax Pack & EDA Implementation"
task_id: "261105_xtrax-adopt-aminx"
author: "P7 Code Review Panel"
date: "2026-11-05"
status: "approved"
ceremony_tier: "deep"
category: "audits"
epic_id: 6497
child_items: [6498, 6499, 6500, 6501, 6502, 6503, 6504, 6505, 6506, 6507, 6508, 6509]
tags: ["xtrax", "pack", "eda", "code-review", "p7", "audits", "invariant-c1"]
---

# P7 Code Review Audit: aminx xtrax Pack & EDA Implementation

## 1. Executive Summary

This audit records the Phase 7 (`P7`) code review verification for the aminx adoption of `xtrax.pack` and `xtrax.eda` under Task ID `261105_xtrax-adopt-aminx` (Epic #6497, addressing debts #3257, #3143, #3142, #3147, #2421, #2148, #2149, #2540, #2135, #2136, #2493).

All implementation artifacts across `src/aminx/host/pack_shim.py`, `src/aminx/host/plan_eda.py`, `src/aminx/cli.py`, `pyproject.toml`, `tests/host/test_pack_shim.py`, `tests/host/test_plan_eda.py`, and `scripts/analysis/verify_xtrax_adoption.py` were audited for correctness, architectural integrity, invariant compliance, test coverage, and code hygiene.

The implementation is **APPROVED** with zero outstanding blocking or major issues.

---

## 2. Invariant C1 Compliance Verification

- **Invariant C1 Statement**: The production reference runner `src/aminx/host/campaign.py` must remain completely intact and unmodified (0 bytes diff against `origin/main`).
- **Audit Command**:
  ```bash
  git diff origin/main -- src/aminx/host/campaign.py
  ```
- **Observed Result**: Exactly **0 bytes diff**; command exited 0 with no stdout/stderr.
- **Verdict**: **VERIFIED — INVARIANT C1 FULLY PRESERVED**.

---

## 3. Code Review Findings & Verified Solutions

### 3.1 `record_name` Binding & Atomic Relocation in `src/aminx/host/pack_shim.py`
- **Issue Audited**: `xtrax.pack.write_unit_record` hardcodes output record name to `"unit.json"` and does not accept `record_name`.
- **Verified Solution**:
  - `write_aminx_unit` accepts `record_name: str = "unit.json"`.
  - Prior to writing, verifies `(unit_path / record_name).exists()`; raises `UnitPackRefused("R-U1", ...)` if target record already exists (preserving record immutability).
  - Calls `write_unit_record(unit_path, ...)`.
  - If `record_name != "unit.json"`, atomically moves `default_record_path.replace(target_record_path)`.
  - `run_packed_manifest_row` and `resolve_unit_adapter` accept `record_name: str = "unit.json"` and propagate it consistently across resume checks, writer, packer, and adapter classification.
- **Coverage**: Verified via `tests/host/test_pack_shim.py::test_custom_record_name_roundtrip`.

### 3.2 Plan Panel Rendering & Manifest SHA-256 Verification in `src/aminx/host/plan_eda.py`
- **Capabilities Verified**:
  - `render_aminx_plan_panels(plan, fmt="svg", ...)` delegates to `xtrax.eda.render` producing valid SVG bytes containing `<svg` root element.
  - `generate_aminx_plan_spec(plan, ...)` emits declarative Vega-Lite v5 specification dictionary containing `$schema` and `description`.
  - `verify_export_manifest_sha256(export_dir, ...)` verifies byte length and SHA-256 digests for all declared artifacts against `*_manifest.json`.
  - `export_and_verify_plan_artifacts(plan, output_dir, prefix="plan")` exports byte-deterministic plan artifacts and cryptographically verifies them in one atomic operation.
- **Coverage**: Verified via `test_render_aminx_plan_panels`, `test_generate_aminx_plan_spec`, `test_export_and_verify_plan_artifacts`, and negative control `test_negative_control_corrupted_export_artifact_raises`.

### 3.3 Negative Refusal Controls
- Tested negative controls confirm that:
  - Bit corruption in `members.tar.zst` triggers `ValueError` during resume validation.
  - Duplicate record creation on an existing unit raises `UnitPackRefused("R-U1")`.
  - Loose file collision in an already-packed unit raises `UnitPackRefused("R-U3")`.
  - Empty Zarr store without array chunks raises `UnitPackRefused("R-U1")`.
  - Tampering with any exported plan artifact triggers `ValueError` (digest mismatch) during manifest verification.

---

## 4. Test Suite and Linter Status

### 4.1 Pytest Execution
- **Command**:
  ```bash
  uv run pytest tests/host/test_pack_shim.py tests/host/test_plan_eda.py -vv
  ```
- **Results**:
  - `tests/host/test_pack_shim.py`: 9 passed
    1. `test_unit_creation_packing_and_member_readback` PASSED
    2. `test_reader_backward_compatibility_aminx_done_marker_adapter` PASSED
    3. `test_negative_control_bit_corruption_raises_value_error` PASSED
    4. `test_negative_control_loose_file_collision_refuses_pack` PASSED
    5. `test_negative_control_empty_unit_refusal` PASSED
    6. `test_quarantining_legacy_done_marker_upon_packing` PASSED
    7. `test_write_aminx_unit_failure_stamp` PASSED
    8. `test_ac10_manifest_slice_parity` PASSED
    9. `test_custom_record_name_roundtrip` PASSED
  - `tests/host/test_plan_eda.py`: 8 passed
    1. `test_explain_aminx_plan` PASSED
    2. `test_analyze_aminx_bucket` PASSED
    3. `test_verify_checkpoint_and_explain_plan` PASSED
    4. `test_verify_checkpoint_and_explain_plan_rejects_unknown_checkpoint` PASSED
    5. `test_render_aminx_plan_panels` PASSED
    6. `test_generate_aminx_plan_spec` PASSED
    7. `test_export_and_verify_plan_artifacts` PASSED
    8. `test_negative_control_corrupted_export_artifact_raises` PASSED
  - **Summary**: **17 passed** in 11.03s, 0 failures.

### 4.2 Ruff Linter Execution
- **Command**:
  ```bash
  uv run ruff check src/aminx/host/pack_shim.py src/aminx/host/plan_eda.py src/aminx/cli.py tests/host/test_pack_shim.py tests/host/test_plan_eda.py
  ```
- **Result**: `All checks passed!` (0 violations).

---

## 5. Pre-Registered Bathos (`bth`) Verification Run

- **Run ID**: `51098fa2-f84c-480a-893c-bf881c7e10f2`
- **Project**: `aminx`
- **Status**: `completed`
- **Outcome**: `pass` (residual = False)
- **Exit Code**: `0`
- **Duration**: `3.48s`
- **Git Hash**: `bcbd966588e29cea080b8d003d30761a8d3a100f`
- **Git Branch**: `deps/xtrax-b7-261105`
- **Command**:
  `/tmp/py312_venv/bin/python scripts/analysis/verify_xtrax_adoption.py --out outputs/analysis/verify_xtrax_adoption.json`
- **Output Artifact**: `outputs/analysis/verify_xtrax_adoption.json`
- **Emitted Metrics**:
  ```json
  {
    "pack_ok": true,
    "eda_ok": true,
    "controls_ok": true,
    "custom_record_name_ok": true,
    "eda_render_svg_ok": true,
    "eda_manifest_sha256_ok": true,
    "negative_controls_ok": true,
    "xtrax_version": "0.4.1b7"
  }
  ```

---

## 6. Review Conclusion

The code implementation fulfills all requirements of Task `261105_xtrax-adopt-aminx`, maintains full backward reader compatibility for existing `campaign_done_marker_v3` units, leaves `src/aminx/host/campaign.py` strictly frozen and unmodified (Invariant C1), establishes genuine cryptographic manifest verification and visual plan inspection, and records a verified PASS run in the Bathos catalog.
