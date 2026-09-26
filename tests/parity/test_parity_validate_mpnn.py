"""Tests for `parity_validate_mpnn.py`'s pure helpers (T11 step 2).

Only the pure logic (SQL construction, the O1 mapping, clause-parity/adversarial-
survival arithmetic, junit parsing) is exercised here. The script itself is never
invoked end to end (it runs `bth sync`, the heavy invariant suite, and calls
`bathos.parity.compute_grade` -- the orchestrator runs it for real, per this
task's brief).
"""

from __future__ import annotations

from pathlib import Path

from scripts.browser_validation.parity_validate_mpnn import (
  compute_adversarial_survived,
  compute_clause_parity,
  determine_prereq_status,
  display_outcome,
  identity_query_sql,
  outcome_query_sql,
  parse_invariant_junit,
)

# --------------------------------------------------------------------------------------
# Identity-form SQL construction
# --------------------------------------------------------------------------------------


def test_identity_query_sql_is_a_count_query_on_native_columns() -> None:
  sql = identity_query_sql("layer_a_exact_validate", "abc123", "sidecar-sha", "lock-sha")
  assert sql.startswith("SELECT count(*) FROM runs WHERE")
  assert "command LIKE '%layer_a_exact_validate.py%'" in sql
  assert "git_hash = 'abc123'" in sql
  assert "git_dirty = false" in sql
  assert "dependency_lock_sha256 = 'lock-sha'" in sql
  assert "sidecar_sha256 = 'sidecar-sha'" in sql
  assert "differential_status = 'passed'" in sql
  assert "exit_code = 0" in sql
  assert "status = 'completed'" in sql
  assert "metadata" not in sql  # never json_extract on the dropped metadata column


def test_outcome_query_sql_reuses_the_identity_filter() -> None:
  sql = outcome_query_sql("layer_a_sampling_validate", "def456", "sc", "lk")
  assert sql.startswith("SELECT outcome FROM runs WHERE")
  assert "git_hash = 'def456'" in sql
  assert sql.endswith("ORDER BY timestamp DESC LIMIT 1")


# --------------------------------------------------------------------------------------
# O1 mapping
# --------------------------------------------------------------------------------------


def test_mapping_both_pass_is_ok_rung_r1() -> None:
  status, rung = determine_prereq_status(
    exact_ready=True, exact_outcome="pass", sampling_ready=True, sampling_outcome="pass"
  )
  assert (status, rung) == ("ok", "R1")


def test_mapping_exact_headroom_sampling_pass_is_headroom_rung_r2() -> None:
  status, rung = determine_prereq_status(
    exact_ready=True,
    exact_outcome="partial_headroom",
    sampling_ready=True,
    sampling_outcome="pass",
  )
  assert (status, rung) == ("headroom", "R2")


def test_mapping_missing_sampling_entry_is_missing() -> None:
  """The live state of this branch (260925): no `layer_a_sampling_validate` ledger
  entry exists at all (sampling calibrate ended `budget_exceeded`)."""
  status, rung = determine_prereq_status(
    exact_ready=True, exact_outcome="partial_headroom", sampling_ready=False, sampling_outcome=None
  )
  assert (status, rung) == ("missing", "R4")


def test_mapping_both_ready_but_unmapped_outcomes_is_failed() -> None:
  status, rung = determine_prereq_status(
    exact_ready=True, exact_outcome="fail", sampling_ready=True, sampling_outcome="fail"
  )
  assert (status, rung) == ("failed", "R4")


def test_mapping_exact_not_ready_is_missing_even_if_sampling_ready() -> None:
  status, rung = determine_prereq_status(
    exact_ready=False, exact_outcome=None, sampling_ready=True, sampling_outcome="pass"
  )
  assert (status, rung) == ("missing", "R4")


def test_display_outcome_prefers_identity_verified_outcome() -> None:
  assert display_outcome({"outcome": "pass"}, "partial_headroom") == "partial_headroom"


def test_display_outcome_falls_back_to_ledger_outcome() -> None:
  assert display_outcome({"outcome": "budget_exceeded"}, None) == "budget_exceeded"


def test_display_outcome_missing_when_no_entry_at_all() -> None:
  assert display_outcome(None, None) == "missing"


# --------------------------------------------------------------------------------------
# Clause parity / adversarial survival (spec "Grade inputs")
# --------------------------------------------------------------------------------------


def test_clause_parity_is_core_restricted_in_both_terms() -> None:
  clauses = [
    {"core": True, "verdict": "MATCH"},
    {"core": True, "verdict": "MATCH"},
    {"core": True, "verdict": "MISSING"},
    {"core": False, "verdict": "MATCH"},  # designed deviation: excluded from denominator too
  ]
  pct, numerator, denominator = compute_clause_parity(clauses)
  assert (numerator, denominator) == (2, 3)
  assert pct == 2 / 3


def test_clause_parity_excludes_ambiguous_from_both_terms() -> None:
  clauses = [
    {"core": True, "verdict": "MATCH"},
    {"core": True, "verdict": "AMBIGUOUS"},
  ]
  pct, numerator, denominator = compute_clause_parity(clauses)
  assert (numerator, denominator) == (1, 1)
  assert pct == 1.0


def test_clause_parity_zero_denominator_is_zero_not_error() -> None:
  pct, numerator, denominator = compute_clause_parity([])
  assert (pct, numerator, denominator) == (0.0, 0, 0)


def test_adversarial_survived_false_only_on_confirmed_core_defect() -> None:
  assert compute_adversarial_survived([{"severity": "minor"}, {"severity": "major"}]) is True
  assert compute_adversarial_survived([{"severity": "core"}]) is False
  assert compute_adversarial_survived([]) is True


# --------------------------------------------------------------------------------------
# Invariant junit parsing
# --------------------------------------------------------------------------------------

_JUNIT_GREEN = """<?xml version="1.0"?>
<testsuites>
  <testsuite name="invariants" tests="8" failures="0" errors="0" skipped="0"></testsuite>
</testsuites>
"""

_JUNIT_ONE_FAILED = """<?xml version="1.0"?>
<testsuites>
  <testsuite name="invariants" tests="8" failures="1" errors="0" skipped="0"></testsuite>
</testsuites>
"""

_JUNIT_TOO_FEW = """<?xml version="1.0"?>
<testsuites>
  <testsuite name="invariants" tests="5" failures="0" errors="0" skipped="0"></testsuite>
</testsuites>
"""


def test_parse_invariant_junit_green(tmp_path: Path) -> None:
  junit_path = tmp_path / "junit.xml"
  junit_path.write_text(_JUNIT_GREEN)
  info = parse_invariant_junit(junit_path)
  assert info == {"n_invariant_tests": 8, "n_invariant_skipped": 0, "invariant_pass": True}


def test_parse_invariant_junit_failure_is_not_pass(tmp_path: Path) -> None:
  junit_path = tmp_path / "junit.xml"
  junit_path.write_text(_JUNIT_ONE_FAILED)
  info = parse_invariant_junit(junit_path)
  assert info["invariant_pass"] is False
  assert info["n_invariant_tests"] == 8


def test_parse_invariant_junit_too_few_tests_is_not_pass(tmp_path: Path) -> None:
  junit_path = tmp_path / "junit.xml"
  junit_path.write_text(_JUNIT_TOO_FEW)
  info = parse_invariant_junit(junit_path)
  assert info["invariant_pass"] is False
  assert info["n_invariant_tests"] == 5
