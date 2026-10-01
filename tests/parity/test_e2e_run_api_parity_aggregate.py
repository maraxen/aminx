"""The e2e run-API parity verdict is computed from files; check it on synthetic records.

A verdict computed from records must fire on a pass AND refuse each way a run can silently look fine:
a failed lane, a control that was not rejected, an absent cell, a started-but-unrecorded cell, and
mixed code commits.  Also checks the statistics the distributional lane relies on, against ground truth.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "e2e_run_api_parity.py"
_spec = importlib.util.spec_from_file_location("e2e_run_api_parity", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["e2e_run_api_parity"] = mod
_spec.loader.exec_module(mod)


def _write(out: Path, cell: str, *, stamp: bool = True, **fields) -> None:
  """Write a cell record the way run_cell does: result file plus a .done stamp carrying its sha256."""
  record = {"cell": cell, "status": "ok", "code_commit": "abc123", "smoke": False, "protocol_version": 2, **fields}
  path = out / f"{cell}.json"
  path.write_text(json.dumps(record))
  if stamp:
    (out / f"{cell}.done").write_text(json.dumps({
      "cell": cell, "inputs_hash": "h", "result_sha256": mod._sha256_file(path), "status": record["status"]}))


def _full_pass(out: Path) -> None:
  for cell in (*mod.KNOB_CELLS, *mod.SCORE_CELLS):
    _write(out, cell, passed=True, lanes={"T": {"passed": True}})
  for cell in mod.CONTROL_CELLS:
    _write(out, cell, rejected=True, lanes={"control": {}})
  _write(out, "cli_equivalence", cli={"passed": True})
  _write(out, "instrument", instrument={"passed": True})


def test_all_pass_verdict(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  agg = mod.aggregate(tmp_path)
  assert agg["cells_ok"] and agg["controls_rejected"] and agg["instrument_ok"] and agg["cli_ok"]
  assert agg["single_code_commit"]
  assert agg["n_cells_ok"] == agg["n_cells_expected"] == len(mod.KNOB_CELLS) + len(mod.SCORE_CELLS)
  assert agg["knobs_declared_unvaried"], "unvaried knobs must be stated, not omitted"


def test_failed_lane_is_named(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  _write(tmp_path, "pmpnn__omit_CW", passed=False, lanes={"D": {"passed": False, "p_value": 1e-9, "tv": 0.4}})
  agg = mod.aggregate(tmp_path)
  assert not agg["cells_ok"]
  assert agg["failing_cells"]["pmpnn__omit_CW"]["D"]["p_value"] == 1e-9


def test_unrejected_control_blocks_pass(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  _write(tmp_path, "ctl_D_wrong_temperature", rejected=False, lanes={"control": {}})
  agg = mod.aggregate(tmp_path)
  assert agg["cells_ok"] and not agg["controls_rejected"]
  assert agg["n_controls_rejected"] == len(mod.CONTROL_CELLS) - 1


@pytest.mark.parametrize("drop", ["pmpnn__base", "ctl_T_wrong_order", "cli_equivalence", "instrument"])
def test_absent_cell_counts_against(tmp_path: Path, drop: str) -> None:
  _full_pass(tmp_path)
  (tmp_path / f"{drop}.json").unlink()
  agg = mod.aggregate(tmp_path)
  assert not (agg["cells_ok"] and agg["controls_rejected"] and agg["instrument_ok"] and agg["cli_ok"])


def test_started_but_unrecorded_is_crash(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  (tmp_path / "lmpnn__bias_AE.json").unlink()
  (tmp_path / "lmpnn__bias_AE.started").write_text("{}")
  agg = mod.aggregate(tmp_path)
  assert not agg["cells_ok"]
  assert agg["failing_cells"]["lmpnn__bias_AE"]["state"] == "crashed_or_killed"


def test_errored_cell_is_failure_not_pass(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  _write(tmp_path, "lmpnn__score", status="error", error_type="RuntimeError", error_head="boom")
  assert not mod.aggregate(tmp_path)["cells_ok"]


def test_mixed_code_commits_refused(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  _write(tmp_path, "pmpnn__base", passed=True, code_commit="other", lanes={})
  assert not mod.aggregate(tmp_path)["single_code_commit"]


def test_control_needs_rejected_true_not_just_present(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  _write(tmp_path, "ctl_S_stock_single_aa", lanes={"control": {}})  # no 'rejected' key at all
  assert not mod.aggregate(tmp_path)["controls_rejected"]


# ---- the chi-square instrument against ground truth ---------------------------------------------
def test_chisq_accepts_exact_draws_and_rejects_a_shift() -> None:
  rng = np.random.default_rng(1)
  p = rng.dirichlet(np.full(20, 0.5))
  null = mod.chisq_test(rng.multinomial(mod.N_DIST, p).astype(float), p)
  assert null["p_value"] > mod.ALPHA
  shifted = p**0.5 / (p**0.5).sum()  # a temperature-like change of the distribution
  alt = mod.chisq_test(rng.multinomial(mod.N_DIST, shifted).astype(float), p)
  assert alt["tv"] > 0.05
  assert alt["p_value"] <= mod.ALPHA


def test_instrument_cell_has_power_and_controlled_false_rejects() -> None:
  result = mod.lane_instrument(seed=0)
  assert result["false_reject_rate"] <= mod.INSTRUMENT_FALSE_REJECT_MAX
  assert result["power_at_tv"] >= mod.INSTRUMENT_POWER_MIN
  assert result["passed"]


def test_cell_table_is_consistent() -> None:
  assert len(set(mod.ALL_CELLS)) == len(mod.ALL_CELLS)
  assert set(mod.CONTROL_CELLS) == {"ctl_T_wrong_order", "ctl_D_wrong_temperature", "ctl_C_wrong_bias",
                                    "ctl_S_stock_single_aa", "ctl_CLI_dropped_knob"}
  for model, knob in mod.KNOB_CELLS.values():
    assert model in mod.MODELS and knob in mod.KNOBS


# ---- review hardening: a verdict must rest on stamped, non-smoke, verified records -----------------------------
def test_smoke_records_cannot_produce_a_pass(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  _write(tmp_path, "pmpnn__base", passed=True, smoke=True, lanes={})
  agg = mod.aggregate(tmp_path)
  assert not agg["cells_ok"]
  assert agg["failing_cells"]["pmpnn__base"]["state"] == "invalid_record"


def test_a_record_without_a_stamp_is_not_evidence(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  (tmp_path / "pmpnn__score.done").unlink()
  assert not mod.aggregate(tmp_path)["cells_ok"]


def test_a_hand_edited_record_fails_its_stamp(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  path = tmp_path / "lmpnn__score.json"
  record = json.loads(path.read_text())
  record["lanes"] = {"S": {"passed": True, "max_abs_logit": 0.0}}  # edited after the stamp was written
  path.write_text(json.dumps(record))
  assert not mod.aggregate(tmp_path)["cells_ok"]


def _restamp(out: Path, cell: str, inputs_hash: str) -> None:
  stamp = json.loads((out / f"{cell}.done").read_text())
  stamp["inputs_hash"] = inputs_hash
  (out / f"{cell}.done").write_text(json.dumps(stamp))


def test_an_errored_cell_is_not_reused_on_rerun(tmp_path: Path) -> None:
  """A harness failure (OOM, import error) must be retried, not frozen in by its own stamp."""
  _write(tmp_path, "pmpnn__base", status="error", error_type="MemoryError", error_head="x")
  _restamp(tmp_path, "pmpnn__base", "same")
  assert not mod._can_reuse(tmp_path / "pmpnn__base.done", tmp_path / "pmpnn__base.json", "same")


def test_a_clean_cell_is_reused_only_on_a_matching_hash(tmp_path: Path) -> None:
  _write(tmp_path, "pmpnn__temp05", passed=True, lanes={})
  _restamp(tmp_path, "pmpnn__temp05", "same")
  done, result = tmp_path / "pmpnn__temp05.done", tmp_path / "pmpnn__temp05.json"
  assert mod._can_reuse(done, result, "same")
  assert not mod._can_reuse(done, result, "different")


def test_draws_on_a_token_the_reference_forbids_are_rejected_even_when_no_tail_mass_is_expected() -> None:
  """omit_AA gives the reference probability exactly 0 on some tokens; aminx emitting them must fail lane D."""
  probs = np.zeros(20)
  probs[3] = 1.0  # all expected mass on one token, exactly zero elsewhere
  counts = np.zeros(20)
  counts[3], counts[1] = 500, 12  # token 1 is forbidden by the reference
  assert mod.chisq_test(counts, probs)["p_value"] <= mod.ALPHA
  counts_ok = np.zeros(20)
  counts_ok[3] = 512
  assert mod.chisq_test(counts_ok, probs)["p_value"] > mod.ALPHA


# ---- second review round ------------------------------------------------------------------------------------
@pytest.mark.parametrize("top", [0.9999, 0.99974])
def test_lane_d_test_holds_its_level_on_peaked_distributions(top: float) -> None:
  """The registered Pearson chi-square false-rejected ~5% (nominal 0.1%) when almost all mass is on one token."""
  rng = np.random.default_rng(0)
  p = np.r_[top, np.full(19, (1 - top) / 19)]
  rejects = sum(mod.chisq_test(rng.multinomial(mod.N_DIST, p).astype(float), p, n_sim=3000)["p_value"] <= mod.ALPHA
                for _ in range(1000))
  assert rejects / 1000 <= mod.INSTRUMENT_FALSE_REJECT_MAX


def test_instrument_measures_the_peaked_regime_it_is_used_in() -> None:
  result = mod.lane_instrument(seed=0, reps=1000, n_sim=3000)
  assert {"dirichlet", "peaked"} <= set(result["by_class"])
  assert result["false_reject_rate"] <= mod.INSTRUMENT_FALSE_REJECT_MAX
  assert result["passed"]


def test_a_draw_of_the_X_token_is_counted_not_dropped() -> None:
  """aminx emitting token 20 (X) at the tested position must register as a forbidden-token draw."""
  seqs = np.full((512, 4), 3)
  seqs[:12, 2] = 20  # twelve draws of X at position 2
  probs = np.zeros(20)
  probs[3] = 1.0
  result = mod.first_step_test(seqs[:, 2], probs)
  assert result["p_value"] <= mod.ALPHA
  clean = mod.first_step_test(np.full(512, 3), probs)
  assert clean["p_value"] > mod.ALPHA


def test_first_designed_rank_is_read_from_the_order_not_assumed() -> None:
  # The reference orders by argsort((chain_mask + 1e-4) * |randn|), which makes "fixed first" likely, not certain: here
  # fixed position 0 is decoded AFTER designed position 5.
  order = np.array([1, 5, 0, 2, 3, 4])
  designed = np.array([False, False, True, True, True, True])  # positions 0 and 1 are fixed
  n_fixed = int((~designed).sum())
  assert not designed[order[n_fixed]], "the old 'order[n_fixed]' shortcut lands on a FIXED position here"
  assert mod.first_designed_rank(order, designed) == 1
  assert mod.first_designed_rank(np.array([0, 1, 2, 3, 4, 5]), designed) == 2  # the usual fixed-first case


def test_cli_check_reads_top_level_keys_only() -> None:
  expected = {"chain_id": "C", "temperature": 0.3}
  ok = mod.check_cli_knobs({"chain_id": "C", "temperature": [0.3]}, expected)
  assert all(v["ok"] for v in ok.values())
  nested_only = mod.check_cli_knobs({"chain_id": None, "meta": {"chain_id": "C"}, "temperature": [0.3]}, expected)
  assert not nested_only["chain_id"]["ok"], "a knob that only reaches an unrelated sub-object must not count"


def test_truncated_stamp_or_record_is_invalid_not_a_crash(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  (tmp_path / "pmpnn__base.done").write_text('{"cell": "pmpnn__base", "inputs_h')  # killed mid-write
  assert not mod._can_reuse(tmp_path / "pmpnn__base.done", tmp_path / "pmpnn__base.json", "h")
  (tmp_path / "pmpnn__temp05.json").write_text('{"cell": "pmpnn__temp05", "stat')
  agg = mod.aggregate(tmp_path)  # must not raise
  assert not agg["cells_ok"]
  assert agg["failing_cells"]["pmpnn__temp05"]["state"] == "invalid_record"


def test_different_aminx_source_trees_are_not_one_code_state(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  _write(tmp_path, "pmpnn__base", passed=True, lanes={}, src_tree_sha256="aaa")
  _write(tmp_path, "pmpnn__temp05", passed=True, lanes={}, src_tree_sha256="bbb")
  assert not mod.aggregate(tmp_path)["single_code_commit"]
  _write(tmp_path, "pmpnn__temp05", passed=True, lanes={}, src_tree_sha256="aaa")
  assert mod.aggregate(tmp_path)["single_code_commit"]


def test_a_cpu_run_cannot_pass_as_a_gpu_run() -> None:
  """GPU mode must fail loudly rather than fall back to CPU and produce a plausible, wrongly-labelled record."""
  mod.check_platform("gpu", "gpu")
  mod.check_platform("cpu", "cpu")
  with pytest.raises(RuntimeError, match="expected gpu"):
    mod.check_platform("gpu", "cpu")
  with pytest.raises(RuntimeError, match="expected cpu"):
    mod.check_platform("cpu", "gpu")


def test_mixed_hardware_in_one_directory_is_reported_and_refused(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  assert mod.aggregate(tmp_path)["single_platform"]
  _write(tmp_path, "pmpnn__base", passed=True, lanes={}, platform="cpu")
  _write(tmp_path, "pmpnn__temp05", passed=True, lanes={}, platform="gpu")
  agg = mod.aggregate(tmp_path)
  assert not agg["single_platform"]
  assert not agg["single_code_commit"], "mixed hardware must not be able to produce a pass"
  assert agg["platforms"] == ["cpu", "gpu"]


def test_a_directory_from_before_the_amendment_is_judged_by_its_own_four_controls(tmp_path: Path) -> None:
  """The registered runs (protocol 1) have no CLI control; they must still re-aggregate from their own records."""
  _full_pass(tmp_path)
  for path in tmp_path.glob("*.json"):  # rewrite every record as protocol 1 and drop the fifth control
    rec = json.loads(path.read_text())
    rec.pop("protocol_version", None)
    path.write_text(json.dumps(rec))
    stamp = json.loads(path.with_suffix(".done").read_text())
    stamp["result_sha256"] = mod._sha256_file(path)
    path.with_suffix(".done").write_text(json.dumps(stamp))
  for name in ("ctl_CLI_dropped_knob.json", "ctl_CLI_dropped_knob.done"):
    (tmp_path / name).unlink()
  agg = mod.aggregate(tmp_path)
  assert agg["controls_rejected"] and agg["n_controls_rejected"] == 4


def test_an_amended_run_missing_the_cli_control_is_not_a_pass(tmp_path: Path) -> None:
  _full_pass(tmp_path)
  for name in ("ctl_CLI_dropped_knob.json", "ctl_CLI_dropped_knob.done"):
    (tmp_path / name).unlink()
  assert not mod.aggregate(tmp_path)["controls_rejected"]
