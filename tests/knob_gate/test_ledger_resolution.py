# ruff: noqa: S101
"""The one shard-resolution rule (``_ledger.py``), and the family-scoped verdicts built on it.

Each test names the pre-fix behaviour it guards against. These were written without being run
on this machine (local runs are capped to narrow selectors; the suite runs remotely), so the
first remote run is their first execution.
"""

from __future__ import annotations

import importlib.util
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


def test_family_scoped_coverage_is_invalid_for_an_unattributed_row(tmp_path: Path) -> None:
  # Pre-fix: the family-scoped verdict skipped a family-less row whose slug was absent from the
  # ledger, so the verdict could come out 'pass'. Unscoped it was already instrument_invalid.
  manifest = tmp_path / "manifest.toml"
  manifest.write_text(
    '[[branch]]\nid = "m1"\n[branch.vehicle]\nkind = "sidecar"\nslug = "potts_absent"\n',
    encoding="utf-8",
  )
  outcomes = tmp_path / "outcomes.jsonl"
  outcomes.write_text("", encoding="utf-8")
  ledgers = tmp_path / "ledgers"
  _shard(ledgers, "potts", _POTTS_SHARD)
  assert check_branch_coverage(manifest, outcomes, ledgers, family="potts") == "instrument_invalid"
  assert check_branch_coverage(manifest, outcomes, ledgers, family=None) == "instrument_invalid"


def _load_verify_wave() -> ModuleType:
  path = _REPO / "scripts" / "redsox" / "verify_wave.py"
  spec = importlib.util.spec_from_file_location("verify_wave_under_test", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_verify_wave_exits_nonzero_for_a_manifest_slug_without_a_ledger_entry(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Pre-fix: --family potts filtered the manifest by the ledger, so potts_absent vanished and the
  # run exited 0 with an unverified slug. potts_energy_parity is recorded and "verifies" here.
  knob_gate = tmp_path / "tests" / "knob_gate"
  _shard(knob_gate / "ledgers", "potts", _POTTS_SHARD)
  (knob_gate / "branch_manifest.toml").write_text(
    '[[branch]]\nid = "m1"\nfamily = "potts"\n[branch.vehicle]\nkind = "sidecar"\nslug = "potts_energy_parity"\n\n'
    '[[branch]]\nid = "m2"\nfamily = "potts"\n[branch.vehicle]\nkind = "sidecar"\nslug = "potts_absent"\n',
    encoding="utf-8",
  )
  verify_wave = _load_verify_wave()
  monkeypatch.setattr(verify_wave, "REPO", tmp_path)
  monkeypatch.setattr(verify_wave, "verify", lambda _slug, _rid, _ids: True)
  assert verify_wave.main(["--family", "potts"]) == 1
