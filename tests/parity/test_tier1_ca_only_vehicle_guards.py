"""Guards on the Tier 1 and CA-only parity vehicles.

* A missing upstream checkpoint is a hard failure (``FileNotFoundError`` from the suite, exit 1 from
  ``main``), never a skip and never a pass.
* ``--synthetic`` is a harness smoke test. It must never grade ``clean == "pass"``, so a synthetic
  run cannot promote a row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts.parity import ca_only_parity, tier1_variants_parity

_VEHICLES: list[tuple[str, ModuleType, str]] = [
  ("tier1_variants_parity", tier1_variants_parity, "run_tier1_parity_suite"),
  ("ca_only_parity", ca_only_parity, "run_ca_only_parity_suite"),
]


@pytest.mark.parametrize(("name", "module", "suite"), _VEHICLES, ids=[v[0] for v in _VEHICLES])
def test_missing_checkpoint_is_hard_failure(
  name: str,
  module: ModuleType,
  suite: str,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(module, "find_parity_repo_file", lambda _rel: None)
  with pytest.raises(FileNotFoundError, match="Missing required upstream checkpoints"):
    getattr(module, suite)(synthetic=False)


@pytest.mark.parametrize(("name", "module", "suite"), _VEHICLES, ids=[v[0] for v in _VEHICLES])
def test_main_exits_nonzero_and_writes_nothing_on_missing_checkpoint(
  name: str,
  module: ModuleType,
  suite: str,
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  out = tmp_path / "out.json"
  controls = tmp_path / "controls.json"
  monkeypatch.setattr(module, "find_parity_repo_file", lambda _rel: None)
  monkeypatch.setattr(sys, "argv", [name, "--out", str(out), "--controls-out", str(controls)])
  with pytest.raises(SystemExit) as excinfo:
    module.main()
  assert excinfo.value.code == 1
  assert not out.exists()
  assert not controls.exists()


@pytest.mark.slow
@pytest.mark.parametrize(("name", "module", "suite"), _VEHICLES, ids=[v[0] for v in _VEHICLES])
def test_synthetic_run_never_grades_pass(
  name: str,
  module: ModuleType,
  suite: str,
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  # Synthetic mode must not look for checkpoints at all: make any lookup fail loudly.
  def _no_lookup(_rel: str) -> None:
    msg = "--synthetic must not resolve upstream checkpoints"
    raise AssertionError(msg)

  monkeypatch.setattr(module, "find_parity_repo_file", _no_lookup)

  result = getattr(module, suite)(synthetic=True)
  assert result["clean"] != "pass"
  assert result["weight_source"] == "synthetic"

  # Same through the CLI: both the results file and branch_controls.json carry the non-pass grade.
  out = tmp_path / "out.json"
  controls = tmp_path / "controls.json"
  monkeypatch.setattr(sys, "argv", [name, "--synthetic", "--out", str(out), "--controls-out", str(controls)])
  module.main()
  assert json.loads(out.read_text(encoding="utf-8"))["clean"] != "pass"
  assert json.loads(controls.read_text(encoding="utf-8"))["clean"] != "pass"
