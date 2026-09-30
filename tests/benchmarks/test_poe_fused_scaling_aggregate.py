"""The verdict logic of scripts/benchmarks/poe_fused_scaling.py, checked on synthetic records.

The experiment's verdict is computed from files, so the aggregator is the measuring instrument and
must be tested with a case that can ONLY pass (all cells fine) AND cases that MUST fail: a planned
cell that errored, a cell that crashed leaving no record, and a control that never started. A
control that fires only on recorded evidence of failure is what stops "the job was cancelled"
from being read as "the pre-fix configuration crashed".
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "poe_fused_scaling.py"


@pytest.fixture(scope="module")
def mod():
  name = "poe_fused_scaling_under_test"
  spec = importlib.util.spec_from_file_location(name, _SCRIPT)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  yield module
  sys.modules.pop(name, None)


def _ok_record(cell: str, n: int) -> dict:
  return {
    "cell": cell, "status": "ok", "n_samples": n, "shape_ok": True, "tokens_in_vocab": True,
    "logits_finite": True, "n_unique_sequences": n, "first_call_seconds": 1.0,
    "device_kind": "H200", "code_commit": "abc123",
  }


def _error_record(cell: str, n: int) -> dict:
  return {"cell": cell, "status": "error", "n_samples": n, "error_type": "XlaRuntimeError",
          "error_head": "Autotuning failed", "code_commit": "abc123"}


def _write(out: Path, cell: str, record: dict | None, *, started: bool = True) -> None:
  if started:
    (out / f"{cell}.started").write_text("{}")
  if record is not None:
    (out / f"{cell}.json").write_text(json.dumps(record))


def _populate(out: Path, mod, *, planned: str = "ok", control: str = "error") -> None:
  for cell, (kind, n) in mod.CELLS.items():
    want = planned if kind == "planned" else control
    if want == "ok":
      _write(out, cell, _ok_record(cell, n))
    elif want == "error":
      _write(out, cell, _error_record(cell, n))
    elif want == "crashed":
      _write(out, cell, None)  # started, no record
    elif want == "absent":
      _write(out, cell, None, started=False)


def test_fixed_and_instrument_valid(mod, tmp_path: Path) -> None:
  """Positive case: planned cells fine, control fails as it must."""
  _populate(tmp_path, mod, planned="ok", control="error")
  result = mod.aggregate(tmp_path)
  assert result["planned_cells_ok"] is True
  assert result["n_planned_ok"] == 3
  assert result["negative_control_failed"] is True


def test_planned_error_is_not_ok(mod, tmp_path: Path) -> None:
  """Negative case: a recorded error in a planned cell must not count as ok."""
  _populate(tmp_path, mod, planned="error", control="error")
  result = mod.aggregate(tmp_path)
  assert result["planned_cells_ok"] is False
  assert result["n_planned_ok"] == 0


def test_planned_crash_with_no_record_is_not_ok(mod, tmp_path: Path) -> None:
  """A process killed mid-compile leaves only a .started stamp; that is a failure."""
  _populate(tmp_path, mod, planned="crashed", control="error")
  result = mod.aggregate(tmp_path)
  assert result["planned_cells_ok"] is False
  assert result["cells"]["planned_n128"]["state"] == "crashed_or_killed"


def test_missing_planned_cell_is_not_ok(mod, tmp_path: Path) -> None:
  _populate(tmp_path, mod, planned="ok", control="error")
  (tmp_path / "planned_n2048.json").unlink()
  (tmp_path / "planned_n2048.started").unlink()
  result = mod.aggregate(tmp_path)
  assert result["planned_cells_ok"] is False
  assert result["cells"]["planned_n2048"]["state"] == "absent"


def test_control_that_passes_means_instrument_cannot_discriminate(mod, tmp_path: Path) -> None:
  """If the pre-fix configuration works on this hardware, the verdict must say so."""
  _populate(tmp_path, mod, planned="ok", control="ok")
  result = mod.aggregate(tmp_path)
  assert result["planned_cells_ok"] is True
  assert result["negative_control_failed"] is False


def test_control_that_never_started_is_not_evidence_of_failure(mod, tmp_path: Path) -> None:
  """A cancelled or still-queued control says nothing about the pre-fix behaviour."""
  _populate(tmp_path, mod, planned="ok", control="absent")
  result = mod.aggregate(tmp_path)
  assert result["negative_control_failed"] is False
  assert result["negative_control_state"] == "absent"


def test_control_killed_mid_run_counts_as_failure(mod, tmp_path: Path) -> None:
  """Started, no record: an OOM-kill or crash, the very failure class being reproduced."""
  _populate(tmp_path, mod, planned="ok", control="crashed")
  assert mod.aggregate(tmp_path)["negative_control_failed"] is True


def test_invalid_output_is_not_ok(mod, tmp_path: Path) -> None:
  """status=ok is not enough: NaN logits or a wrong shape must not pass."""
  _populate(tmp_path, mod, planned="ok", control="error")
  bad = _ok_record("planned_n512", 512)
  bad["logits_finite"] = False
  (tmp_path / "planned_n512.json").write_text(json.dumps(bad))
  result = mod.aggregate(tmp_path)
  assert result["planned_cells_ok"] is False
  assert result["cells"]["planned_n512"]["state"] == "invalid_output"


def test_mixed_code_commits_are_flagged(mod, tmp_path: Path) -> None:
  _populate(tmp_path, mod, planned="ok", control="error")
  other = _ok_record("planned_n128", 128)
  other["code_commit"] = "different"
  (tmp_path / "planned_n128.json").write_text(json.dumps(other))
  assert mod.aggregate(tmp_path)["single_code_commit"] is False
