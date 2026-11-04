# ruff: noqa: S101
"""The one shard-resolution rule (``_ledger.py``), and the family-scoped verdicts built on it.

Each test names the pre-fix behaviour it guards against. These were written without being run
on this machine (local runs are capped to narrow selectors; the suite runs remotely), so the
first remote run is their first execution.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from knob_gate._coverage import check_branch_coverage
from knob_gate._ledger import LedgerError, ledger_ids, rows_for_family, sidecar_slugs

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]


def _shard(ledgers: Path, name: str, body: str) -> None:
  ledgers.mkdir(parents=True, exist_ok=True)
  (ledgers / f"{name}.toml").write_text(body, encoding="utf-8")


_POTTS_SHARD = """
[sidecar.potts_energy_parity]
bth_run_id = "run-potts-1"

[sidecar.laser_score_parity]
bth_run_id = "run-laser-misfiled"
"""


def test_missing_family_shard_is_empty_and_never_falls_back(tmp_path: Path) -> None:
  # Pre-fix: a laser read with no ledgers/laser.toml fell back to the potts ledger and returned
  # potts run ids under the laser label.
  ledgers = tmp_path / "ledgers"
  _shard(ledgers, "potts", _POTTS_SHARD)
  assert ledger_ids(ledgers, "laser") == {}
  assert ledger_ids(ledgers, "potts") == {"potts_energy_parity": "run-potts-1", "laser_score_parity": "run-laser-misfiled"}
  assert ledger_ids(ledgers, None) == ledger_ids(ledgers, "all")


def test_malformed_run_id_fails_loudly(tmp_path: Path) -> None:
  ledgers = tmp_path / "ledgers"
  _shard(ledgers, "potts", '[sidecar.potts_x]\nbth_run_id = ""\n')
  with pytest.raises(LedgerError):
    ledger_ids(ledgers, "potts")


def test_row_without_family_cannot_be_attributed_in_a_family_scope() -> None:
  rows = [
    {"id": "m1", "vehicle": {"kind": "sidecar", "slug": "potts_x"}, "family": "potts"},
    {"id": "m2", "vehicle": {"kind": "sidecar", "slug": "potts_y"}},
  ]
  # Pre-fix: the family-less row was silently skipped in a family scope.
  with pytest.raises(LedgerError):
    rows_for_family(rows, "potts")
  # Unscoped reads keep every row, family-less or not.
  assert len(rows_for_family(rows, None)) == 2


def test_sidecar_slugs_keep_every_row_of_a_scope() -> None:
  rows = [
    {"id": "a", "family": "potts", "vehicle": {"kind": "sidecar", "slug": "potts_x"}},
    {"id": "b", "family": "potts", "vehicle": {"kind": "sidecar", "slug": "potts_x"}},
    {"id": "c", "family": "laser", "vehicle": {"kind": "sidecar", "slug": "laser_y"}},
  ]
  assert sidecar_slugs(rows_for_family(rows, "potts")) == {"potts_x": {"a", "b"}}


_PYTEST_ROW = (
  '[[branch]]\nid = "m1"\nfamily = "potts"\n[branch.vehicle]\nkind = "pytest"\nnodeids = ["t::a"]\n'
)


def _outcomes(path: Path) -> Path:
  """A clean pass for t::a and an AssertionError kill of mutant m1: the pytest row grades 'pass'."""
  records = [
    {"nodeid": "t::a", "when": "call", "outcome": "passed", "mutant": None},
    {"nodeid": "t::a", "when": "call", "outcome": "failed", "exc_type": "AssertionError", "mutant": "m1"},
  ]
  path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
  return path


def test_family_scoped_coverage_refuses_an_unattributed_row(tmp_path: Path) -> None:
  # Pre-fix: a family-less sidecar row whose slug was absent from the ledger was skipped in a family
  # scope, the potts pytest row alone matched, and the verdict was 'pass'. The fixture is first
  # shown to be passable on its own, so the only reason for instrument_invalid is the unattributed row.
  outcomes = _outcomes(tmp_path / "outcomes.jsonl")
  ledgers = tmp_path / "ledgers"
  _shard(ledgers, "potts", _POTTS_SHARD)
  alone = tmp_path / "alone.toml"
  alone.write_text(_PYTEST_ROW, encoding="utf-8")
  assert check_branch_coverage(alone, outcomes, ledgers, family="potts") == "pass"

  unattributed = tmp_path / "unattributed.toml"
  unattributed.write_text(
    _PYTEST_ROW + '\n[[branch]]\nid = "ghost"\n[branch.vehicle]\nkind = "sidecar"\nslug = "potts_absent"\n',
    encoding="utf-8",
  )
  assert check_branch_coverage(unattributed, outcomes, ledgers, family="potts") == "instrument_invalid"


def test_family_row_with_no_ledger_entry_is_instrument_invalid(tmp_path: Path) -> None:
  outcomes = _outcomes(tmp_path / "outcomes.jsonl")
  ledgers = tmp_path / "ledgers"
  _shard(ledgers, "potts", _POTTS_SHARD)
  manifest = tmp_path / "manifest.toml"
  manifest.write_text(
    _PYTEST_ROW
    + '\n[[branch]]\nid = "ghost2"\nfamily = "potts"\n[branch.vehicle]\nkind = "sidecar"\nslug = "potts_absent"\n',
    encoding="utf-8",
  )
  assert check_branch_coverage(manifest, outcomes, ledgers, family="potts") == "instrument_invalid"


def _load_verify_wave() -> ModuleType:
  path = _REPO / "scripts" / "redsox" / "verify_wave.py"
  spec = importlib.util.spec_from_file_location("verify_wave_under_test", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


_ENERGY_ROW = (
  '[[branch]]\nid = "m1"\nfamily = "potts"\n[branch.vehicle]\nkind = "sidecar"\nslug = "potts_energy_parity"\n'
)


def _verify_wave_at(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  manifest_text: str,
) -> ModuleType:
  knob_gate = tmp_path / "tests" / "knob_gate"
  _shard(knob_gate / "ledgers", "potts", _POTTS_SHARD)
  (knob_gate / "branch_manifest.toml").write_text(manifest_text, encoding="utf-8")
  verify_wave = _load_verify_wave()
  monkeypatch.setattr(verify_wave, "REPO", tmp_path)
  monkeypatch.setattr(verify_wave, "verify", lambda _slug, _rid, _ids: True)
  return verify_wave


def test_familyless_row_under_a_family_read_is_refused(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Pre-fix: the family-less potts_absent row was dropped by the allowed-slug filter, nothing was
  # reported missing, and the run exited 0 with an unverified manifest slug.
  verify_wave = _verify_wave_at(
    tmp_path,
    monkeypatch,
    _ENERGY_ROW + '\n[[branch]]\nid = "m2"\n[branch.vehicle]\nkind = "sidecar"\nslug = "potts_absent"\n',
  )
  assert verify_wave.main(["--family", "potts"]) == 2


def test_family_slug_without_a_ledger_entry_is_the_exact_missing_set(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  caplog.set_level("INFO")
  verify_wave = _verify_wave_at(
    tmp_path,
    monkeypatch,
    _ENERGY_ROW
    + '\n[[branch]]\nid = "m2"\nfamily = "potts"\n[branch.vehicle]\nkind = "sidecar"\nslug = "potts_absent"\n',
  )
  assert verify_wave.main(["--family", "potts"]) == 1
  missing = [r.getMessage() for r in caplog.records if r.getMessage().startswith("no run id yet for:")]
  assert missing == ["no run id yet for: potts_absent"]


def test_run_override_for_a_slug_outside_the_family_is_refused(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Pre-fix: the override was logged 'ignored' and the run could still exit 0. Now it is an error.
  verify_wave = _verify_wave_at(tmp_path, monkeypatch, _ENERGY_ROW)
  assert verify_wave.main(["--family", "potts", "--run", "laser_score_parity=zzz"]) == 2


def test_unknown_family_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  verify_wave = _verify_wave_at(tmp_path, monkeypatch, _ENERGY_ROW)
  assert verify_wave.main(["--family", "nosuch"]) == 2
