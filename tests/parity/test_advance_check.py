"""Tests for the Phase-2 advance gate (`advance_check.py`, T11 step 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.browser_validation.advance_check import (
  NO_ROWS_REASON,
  build_advance_table,
  compute_path_verdict,
  gather_context,
  group_defects_by_pid,
  group_lanes_by_pid,
  group_rows_by_pid,
  main,
)

# --------------------------------------------------------------------------------------
# Grouping helpers
# --------------------------------------------------------------------------------------


def test_group_rows_by_pid_splits_on_first_dot() -> None:
  rows = [
    {"path": "P00.weights", "status": "validated"},
    {"path": "P00.log_probs", "status": "validated"},
    {"path": "P05.log_probs", "status": "not_advanced"},
  ]
  by_pid = group_rows_by_pid(rows)
  assert set(by_pid) == {"P00", "P05"}
  assert len(by_pid["P00"]) == 2
  assert len(by_pid["P05"]) == 1


def test_group_rows_by_pid_ignores_malformed_paths() -> None:
  rows = [{"path": "no-dot-here", "status": "validated"}, {"no_path_key": True}]
  assert group_rows_by_pid(rows) == {}


def test_group_lanes_by_pid_maps_lane_names() -> None:
  lanes = [
    {"lane": "P07@T0.1", "equiv": True},
    {"lane": "P07@T1.0", "equiv": True},
    {"lane": "P08", "equiv": False},
    {"lane": "unknown-lane", "equiv": True},
  ]
  by_pid = group_lanes_by_pid(lanes)
  assert len(by_pid["P07"]) == 2
  assert len(by_pid["P08"]) == 1
  assert "unknown-lane" not in {lane.get("lane") for lane in by_pid.get("P99", [])}
  assert all(pid in ("P07", "P08") for pid in by_pid)


def test_group_defects_by_pid_collects_ids_per_path() -> None:
  defects = [
    {"id": "D1", "paths": ["P05", "P06"], "severity": "core"},
    {"id": "D2", "paths": ["P05"], "severity": "minor"},
    {"id": "not-a-str-safe", "paths": []},
  ]
  by_pid = group_defects_by_pid(defects)
  assert sorted(by_pid["P05"]) == ["D1", "D2"]
  assert by_pid["P06"] == ["D1"]


# --------------------------------------------------------------------------------------
# compute_path_verdict
# --------------------------------------------------------------------------------------


def _base_kwargs(**overrides: object) -> dict:
  kwargs = {
    "exact_rows_by_pid": {},
    "sampling_status": "ok",
    "sampling_missing_reason": None,
    "sampling_lanes_by_pid": {},
    "defects_by_pid": {},
  }
  kwargs.update(overrides)
  return kwargs


def test_advances_when_exact_rows_validated_and_no_sampling_required() -> None:
  kwargs = _base_kwargs(
    exact_rows_by_pid={
      "P00": [
        {"path": "P00.weights", "status": "validated", "ratio": 0.0},
        {"path": "P00.log_probs", "status": "validated", "ratio": 0.5},
      ]
    }
  )
  advances, reasons = compute_path_verdict("P00", **kwargs)
  assert advances is True
  assert reasons == []


def test_blocked_when_no_rows_at_all() -> None:
  advances, reasons = compute_path_verdict("P00", **_base_kwargs())
  assert advances is False
  assert reasons == [NO_ROWS_REASON]


def test_blocked_when_a_row_is_not_validated() -> None:
  kwargs = _base_kwargs(
    exact_rows_by_pid={"P05": [{"path": "P05.log_probs", "status": "not_advanced", "ratio": None}]}
  )
  advances, reasons = compute_path_verdict("P05", **kwargs)
  assert advances is False
  assert any("not all validated" in r for r in reasons)


def test_blocked_when_a_row_is_over_bar() -> None:
  kwargs = _base_kwargs(
    exact_rows_by_pid={
      "P05": [{"path": "P05.log_probs", "status": "validated", "ratio": 1.7976931348623157e308}]
    }
  )
  advances, reasons = compute_path_verdict("P05", **kwargs)
  assert advances is False
  assert any("over bar" in r for r in reasons)


def test_sampling_required_path_blocked_when_sampling_missing() -> None:
  kwargs = _base_kwargs(
    exact_rows_by_pid={"P07": [{"path": "P07.log_probs", "status": "validated", "ratio": 0.1}]},
    sampling_status="missing",
    sampling_missing_reason="sampling tier not validated (budget_exceeded)",
  )
  advances, reasons = compute_path_verdict("P07", **kwargs)
  assert advances is False
  assert reasons == ["sampling tier not validated (budget_exceeded)"]


def test_sampling_required_path_blocked_when_lane_not_equivalent() -> None:
  kwargs = _base_kwargs(
    exact_rows_by_pid={"P08": [{"path": "P08.log_probs", "status": "validated", "ratio": 0.1}]},
    sampling_status="ok",
    sampling_lanes_by_pid={"P08": [{"lane": "P08", "equiv": False}]},
  )
  advances, reasons = compute_path_verdict("P08", **kwargs)
  assert advances is False
  assert any("not equivalent" in r for r in reasons)


def test_sampling_required_path_advances_when_lanes_all_equivalent() -> None:
  kwargs = _base_kwargs(
    exact_rows_by_pid={"P08": [{"path": "P08.log_probs", "status": "validated", "ratio": 0.1}]},
    sampling_status="ok",
    sampling_lanes_by_pid={"P08": [{"lane": "P08", "equiv": True}]},
  )
  advances, reasons = compute_path_verdict("P08", **kwargs)
  assert advances is True
  assert reasons == []


def test_blocked_by_confirmed_defect_naming_the_path() -> None:
  kwargs = _base_kwargs(
    exact_rows_by_pid={"P05": [{"path": "P05.log_probs", "status": "validated", "ratio": 0.1}]},
    defects_by_pid={"P05": ["D1"]},
  )
  advances, reasons = compute_path_verdict("P05", **kwargs)
  assert advances is False
  assert any("confirmed defect" in r for r in reasons)


# --------------------------------------------------------------------------------------
# build_advance_table: one advancing path, one blocked path (spec's fixture requirement)
# --------------------------------------------------------------------------------------


def test_build_advance_table_one_advancing_one_blocked() -> None:
  table = build_advance_table(
    exact_rows_by_pid={
      "P00": [{"path": "P00.weights", "status": "validated", "ratio": 0.0}],
      "P05": [{"path": "P05.log_probs", "status": "not_advanced", "ratio": None}],
    },
    exact_result_sha256="deadbeef",
    sampling_status="ok",
    sampling_missing_reason=None,
    sampling_lanes_by_pid={},
    sampling_result_sha256=None,
    defects_by_pid={},
    in_scope_pids={"P00", "P05"},
  )
  assert table["paths"]["P00"]["advances"] is True
  assert table["paths"]["P00"]["reasons"] == []
  assert table["paths"]["P05"]["advances"] is False
  assert table["paths"]["P05"]["reasons"]
  assert table["exact_result_sha256"] == "deadbeef"


# --------------------------------------------------------------------------------------
# End-to-end fixture: real filesystem layout under tmp_path (gather_context + CLI)
# --------------------------------------------------------------------------------------


def _write_json(path: Path, obj: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj))


@pytest.fixture
def fake_worktree(tmp_path: Path) -> Path:
  """Build a minimal worktree: one advancing P-ID (P00), one blocked P-ID (P05).

  P05 is blocked two ways at once (not-validated row AND a confirmed defect) so the
  fixture also exercises `group_defects_by_pid` end to end. No
  `layer_a_sampling_validate` ledger entry exists, matching the real state of this
  branch (260925): the sampling calibrate ended `budget_exceeded`.
  """
  root = tmp_path
  h_v = "4c3a9c3b13e3c3b62b317045a2f0f80160796178"

  _write_json(
    root / "outputs" / "browser_validation" / "layer_a" / "run_ledger.json",
    [
      {
        "stem": "layer_a_sampling_calibrate",
        "H_v": "799a1c4a0816878abac95233c632932914582226",
        "outcome": "budget_exceeded",
      },
      {"stem": "layer_a_exact_validate", "H_v": h_v, "outcome": "partial_headroom"},
    ],
  )
  _write_json(
    root
    / "outputs"
    / "browser_validation"
    / "titanix"
    / f"layer_a_exact_validate-{h_v[:12]}"
    / "layer_a"
    / "layer_a_exact_validate.json",
    {
      "rows": [
        {"path": "P00.weights", "status": "validated", "ratio": 0.0},
        {"path": "P00.log_probs", "status": "validated", "ratio": 0.3},
        {"path": "P05.log_probs", "status": "not_advanced", "ratio": None},
      ]
    },
  )
  _write_json(
    root / "outputs" / "browser_validation" / "litparity" / "adjudication.json",
    {"defects_confirmed": [{"id": "D1", "paths": ["P05"], "severity": "minor"}]},
  )
  return root


def test_gather_context_reads_ledger_titanix_and_adjudication(fake_worktree: Path) -> None:
  ctx = gather_context(fake_worktree)
  assert ctx["sampling_status"] == "missing"
  assert ctx["sampling_missing_reason"] == "sampling tier not validated (budget_exceeded)"
  assert set(ctx["exact_rows_by_pid"]) == {"P00", "P05"}
  assert ctx["defects_by_pid"] == {"P05": ["D1"]}
  assert ctx["exact_result_sha256"] is not None


def test_cli_build_writes_table_with_advancing_and_blocked_paths(
  fake_worktree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  out_path = fake_worktree / "advance_table.json"
  exit_code = main(["--build", "--worktree-root", str(fake_worktree), "--out", str(out_path)])
  assert exit_code == 0
  table = json.loads(out_path.read_text())
  assert table["paths"]["P00"]["advances"] is True
  assert table["paths"]["P05"]["advances"] is False
  assert "no layer-(a) rows" in table["paths"]["P19"]["reasons"]


def test_cli_single_path_exit_code_advancing(fake_worktree: Path) -> None:
  assert main(["P00", "--worktree-root", str(fake_worktree)]) == 0


def test_cli_single_path_exit_code_blocked(fake_worktree: Path) -> None:
  assert main(["P05", "--worktree-root", str(fake_worktree)]) == 1


def test_cli_unknown_path_id_exits_2(fake_worktree: Path) -> None:
  assert main(["P97", "--worktree-root", str(fake_worktree)]) == 2


def test_cli_requires_path_id_or_build(fake_worktree: Path) -> None:
  with pytest.raises(SystemExit):
    main(["--worktree-root", str(fake_worktree)])
