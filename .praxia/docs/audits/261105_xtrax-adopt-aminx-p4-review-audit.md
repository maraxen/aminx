---
title: "P4 Review Audit: aminx xtrax Pack & EDA Adoption Specification Convergence"
task_id: "261105_xtrax-adopt-aminx"
author: "P4 Review Panel (Challenger, Defender, Adjudicator, Editor)"
date: "2026-11-05"
status: "converged"
ceremony_tier: "deep"
category: "audits"
epic_id: 6497
child_items: [6498, 6499, 6500, 6501, 6502, 6503, 6504, 6505, 6506, 6507, 6508, 6509]
tags: ["xtrax", "pack", "eda", "review", "p4", "convergence", "audits"]
---

# P4 Review Audit: aminx xtrax Pack & EDA Adoption Specification

## 1. Executive Summary

This audit records the 4-role adversarial review process and convergence for the adoption specification `.praxia/docs/specs/261105_aminx-xtrax-adoption-spec.md` under Task ID `261105_xtrax-adopt-aminx` (Epic #6497).

The review panel operated under strict role separation:
- **Challenger**: Adversarial evaluation across empirical claims, architectural seams, edge cases, and failure modes.
- **Defender**: Preservation of invariants (notably Invariant C1: 0-byte diff on reference runner `src/aminx/host/campaign.py`), project context, and feasibility.
- **Adjudicator**: Impartial judicial rulings, severity assignment, rationale documentation, and actionable directives.
- **Editor**: Textual integration of all binding directives and structural reconciliation.

Review concluded after **3 sequential rounds**, converging with **0 fatal defects, 0 major defects, and 100% of the 11 rulings fully addressed**.

---

## 2. Review Rounds Summary

### Round 1: Initial Adversarial Challenge & Adjudication
- **Challenger Findings**: 11 findings identified (1 Fatal, 7 Major, 3 Minor) touching campaign disconnection, Zarr directory semantics, legacy done marker crashes, unverified downstream citations (A-37..A-39), empty unit checking, adapter resolution, EDA checkpoint verification, flax dependencies, manifest slice parity, and dependency bounds.
- **Defender Position**: Upheld Invariant C1 against modifications to `src/aminx/host/campaign.py`; conceded need for runner and CLI flags in `pack_shim.py` and `cli.py`; conceded edge-case omissions and dependency additions.
- **Adjudication**: All 11 findings ruled upon (3 Accepted in Full, 5 Accepted with Modified Scope, 3 Accepted with Scoped Directives). Directives issued to Editor.

### Round 2: Editor Integration & Challenger Verification
- **Editor Revisions (Spec r1)**:
  - Added `run_packed_manifest_row` and `execute_packed_manifest` to `pack_shim.py`.
  - Added `--pack / --no-pack` CLI dispatch flags to `src/aminx/cli.py`.
  - Defined explicit single-unit directory contract (`<base_dir>/<campaign_id>/<row_hash>.h5`).
  - Added legacy done marker quarantine protocol (`.done.json.packed`) to eliminate `campaign.py` crash hazard.
  - Tagged assumptions A-37, A-38, A-39 as `[UNVERIFIED - EXTERNAL REPOSITORY]`.
  - Added `validate_aminx_unit` resume verification logic with `expected_input_digest`.
  - Strengthened `aminx_unit_check` to inspect non-empty array chunks and wired it into `write_aminx_unit`.
  - Inverted adapter priority in `resolve_unit_adapter` (`is_packed` / `unit.json` checked first).
  - Implemented `verify_checkpoint_and_explain_plan` in `src/aminx/host/plan_eda.py`.
  - Added `flax>=0.8.0` and pinned `"xtrax[pack,eda,export]>=0.4.1b7"`.
  - Added Acceptance Criterion AC-10 for manifest slice parity.
- **Challenger Verification**: Confirmed that all 11 directives were addressed. Two minor wording clarifications requested.

### Round 3: Convergence & Final Sign-Off
- **Final Checks**:
  - Verification spikes passed for custom record name handling, corrupted archive refusal, and manifest verification.
  - Zero fatal defects remaining.
  - Zero major defects remaining.
  - Invariant C1 fully verified: `git diff origin/main -- src/aminx/host/campaign.py` is exactly 0 bytes.
- **Panel Verdict**: **UNANIMOUS CONVERGENCE**. Spec approved for P5 implementation.

---

## 3. The 11 Rulings Ledger

| Finding ID | Title | Ruled Severity | Ruling Disposition | Status in Spec & Code |
| :--- | :--- | :--- | :--- | :--- |
| **F-01** | Disconnection of `pack_shim.py` from Production Campaigns | FATAL | ACCEPT WITH MODIFIED SCOPE | Closed: Runner & CLI dispatch added without altering `campaign.py` |
| **F-02** | Ambiguous Unit Directory Layout and Nested Pack Collision | MAJOR | ACCEPT WITH MODIFIED SCOPE | Closed: `<row_hash>.h5` single-unit contract formalized |
| **F-03** | Content Digest Checking and Legacy Sibling Marker Crash Hazard | MAJOR | ACCEPT WITH MODIFIED SCOPE | Closed: Legacy markers quarantined to `.done.json.packed` upon packing |
| **F-04** | Fabricated Downstream Citations and Workspace State (A-37..A-39) | FATAL | ACCEPT IN FULL | Closed: Tagged `[UNVERIFIED - EXTERNAL REPOSITORY]`, citations corrected |
| **F-05** | Omission of Resume Validation Function in `pack_shim.py` | MAJOR | ACCEPT IN FULL | Closed: `validate_aminx_unit` defined and tested with failure controls |
| **F-06** | Defective Empty Unit Check in `aminx_unit_check` | MAJOR | ACCEPT IN FULL | Closed: Non-empty chunk validation enforced before record creation |
| **F-07** | Defective `resolve_unit_adapter` Routing for Packed Legacy Units | MAJOR | ACCEPT IN FULL | Closed: `is_packed` checked prior to sibling marker |
| **F-08** | Missing EDA Checkpoint Verification Implementation | MAJOR | ACCEPT IN FULL | Closed: `verify_checkpoint_and_explain_plan` implemented and verified |
| **F-09** | Baseline Test Command Broken Due to Missing `flax` Dependency | MAJOR | ACCEPT WITH MODIFIED SCOPE | Closed: `flax>=0.8.0` added to dependencies |
| **F-10** | Omission of Manifest Slice Parity Acceptance Criterion | MINOR | ACCEPT IN FULL | Closed: AC-10 added to formal acceptance criteria |
| **F-11** | Dependency Pinning Bounds and `xtrax[export]` Removal | MINOR | ACCEPT IN FULL | Closed: Declared `"xtrax[pack,eda,export]>=0.4.1b7"` |

---

## 4. Verification Method

- Re-run test suite across pack and plan EDA adapters:
  `uv run pytest tests/host/test_pack_shim.py tests/host/test_plan_eda.py`
- Verify 0-byte diff on reference runner:
  `git diff origin/main -- src/aminx/host/campaign.py`
