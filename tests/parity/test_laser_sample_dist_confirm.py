# ruff: noqa: S101
"""CPU checks for the LASEr confirmatory distributional script.

No oracle interpreter and no checkpoint: refusal until the pilot constants
exist, cell refusal, unit seeds, and result assembly on synthetic tokens.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "parity"
  / "laser_sample_dist_confirm.py"
)

_TOP_LEVEL = {
  "sidecar_verdict",
  "controls_all_fail",
  "n_controls",
  "cells",
  "cells_selected",
  "structures",
  "pilot_run_id",
  "n_reused",
  "n_computed",
  "script_sha256",
  "smoke",
}
_CELL_KEYS = {
  "verdict",
  "sequence",
  "chi1",
  "m",
  "h_hat",
  "n",
  "controls",
  "per_structure_delta",
}
_DISTANCE_KEYS = {"verdict", "mean_delta", "upper_95", "lower_90", "max_delta", "delta"}
_CHI_KEYS = _DISTANCE_KEYS | {"delta_chi"}
_CONTROL_KEYS = {"verdict", "sequence", "chi1"}


def _load() -> Any:
  spec = importlib.util.spec_from_file_location("laser_sample_dist_confirm_under_test", _SCRIPT)
  if spec is None or spec.loader is None:
    msg = f"cannot load {_SCRIPT}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


confirm = _load()


def test_refuses_until_the_pilot_constants_are_committed() -> None:
  assert confirm.PILOT_RUN_ID is None
  assert confirm.CELLS == {}
  assert confirm.LASER_CELLS == ("min_p0@0.3", "min_p0@1.0", "min_p0.05@0.3")
  with pytest.raises(SystemExit, match="PILOT_RUN_ID"):
    confirm.selected_cells(None, smoke=False)
  with pytest.raises(SystemExit, match="PILOT_RUN_ID"):
    confirm.main([])
  assert confirm.selected_cells(None, smoke=True) == ["min_p0@1.0"]


def test_cells_rejects_min_p0_at_0_1_and_unknown() -> None:
  with pytest.raises(SystemExit, match="min_p0@0.1"):
    confirm.selected_cells("min_p0@0.1", smoke=False)
  with pytest.raises(SystemExit, match="min_p0@0.1"):
    confirm.selected_cells("min_p0@0.1", smoke=True)
  with pytest.raises(SystemExit, match="unknown cell"):
    confirm.selected_cells("plain@1.0", smoke=False)
  with pytest.raises(SystemExit, match="unknown cell"):
    confirm.selected_cells("min_p0@0.2", smoke=False)


def _registered() -> dict[str, dict[str, float | int]]:
  return {
    "min_p0@0.3": {
      "delta": 0.02,
      "delta_chi": 0.05,
      "m": 1.25,
      "h_hat": 0.01,
      "n": 1000,
    },
    "min_p0@1.0": {
      "delta": 0.02,
      "delta_chi": 0.05,
      "m": 1.1,
      "h_hat": 0.011,
      "n": 1000,
    },
    "min_p0.05@0.3": {
      "delta": 0.03,
      "delta_chi": 0.05,
      "m": 1.5,
      "h_hat": 0.012,
      "n": 1000,
    },
  }


def _structures() -> dict[str, dict[str, str]]:
  wanted = {"4jnj-1_prot", "103m_1"}
  return {
    stem: {
      "pdb": f"/pdbs/{stem}.pdb",
      "relative": relative,
      "sha256": digest,
    }
    for relative, digest in confirm.STRUCTURES.items()
    if (stem := Path(relative).stem) in wanted
  }


def test_unit_specs_runs_and_seeds_are_disjoint_from_the_pilot() -> None:
  previous_cells = confirm.CELLS
  confirm.CELLS = _registered()
  try:
    used: set[int] = set()
    specs = confirm.unit_specs(
      _structures(),
      ["min_p0@0.3", "min_p0@1.0", "min_p0.05@0.3"],
      1000,
      used,
    )
  finally:
    confirm.CELLS = previous_cells
  pilot = confirm.pilot
  by_cell_structure: dict[tuple[str, str], set[str]] = {}
  seeds: list[int] = []
  pilot_seeds = {pilot._hash_seed(spec["unit_id"]) for spec in specs}
  for spec in specs:
    by_cell_structure.setdefault((spec["cell"], spec["structure"]), set()).add(spec["run"])
    seeds.append(spec["seed"])
    assert spec["seed"] != pilot._hash_seed(spec["unit_id"])
  assert len(seeds) == len(set(seeds))
  assert set(seeds).isdisjoint(pilot_seeds)
  for structure in ("4jnj-1_prot", "103m_1"):
    for cell in confirm.LASER_CELLS:
      assert by_cell_structure[(cell, structure)] == {"U1", "U2", "A", "CTRL_m"}
  hot = next(
    spec
    for spec in specs
    if spec["cell"] == "min_p0@1.0" and spec["run"] == "CTRL_m" and spec["structure"] == "103m_1"
  )
  warm = next(
    spec
    for spec in specs
    if spec["cell"] == "min_p0@0.3" and spec["run"] == "CTRL_m" and spec["structure"] == "103m_1"
  )
  assert hot["sample_temperature"] == 1.1
  assert warm["sample_temperature"] == 1.25 * 0.3
  assert all(spec["shim"] is (spec["run"] == "U1") for spec in specs)
  assert all(
    spec["min_p"] == (0.05 if spec["cell"].startswith("min_p0.05") else 0.0) for spec in specs
  )
  aminx = next(spec for spec in specs if spec["run"] == "A" and spec["cell"] == "min_p0@1.0")
  assert aminx["sample_temperature"] == 1.0


def _rows(
  token: int,
  n: int,
  length: int,
  *,
  flip_at: int | None = None,
  flip_to: int = 3,
) -> list[list[int]]:
  rows = [[token] * length for _index in range(n)]
  if flip_at is not None:
    for row in rows:
      row[flip_at] = flip_to
  return rows


def _chi(bin_id: int, n: int, length: int) -> list[list[float]]:
  value = -180.0 + 10.0 * bin_id
  return [[value] * length for _index in range(n)]


def _run(
  sequences: list[list[int]],
  degrees: list[list[float]],
  mask: list[bool],
) -> dict[str, Any]:
  return {"sequences": sequences, "chi1_degrees": degrees, "mask": mask}


def _samples_mixed() -> dict[str, dict[str, dict[str, Any]]]:
  """min_p0@1.0 fails on χ1; min_p0@0.3 is sequence-inconclusive with a passing control."""
  n = 8
  short = 4
  # Length 2 with one substituted column is total variation 1/2, so a cell
  # whose δ is exactly 0.5 is inconclusive rather than pass or fail.
  pair = 2
  mask_short = [True] * short
  mask_pair = [True] * pair
  matched_short = _rows(1, n, short)
  chi_match = _chi(0, n, short)
  chi_shift = _chi(1, n, short)
  half = _rows(1, n, short)
  for row in half:
    row[2] = 3
    row[3] = 3
  matched_pair = _rows(1, n, pair)
  flipped = _rows(1, n, pair, flip_at=0, flip_to=2)
  chi_pair = _chi(0, n, pair)
  failing = {
    "U1": _run(matched_short, chi_match, mask_short),
    "U2": _run(matched_short, chi_match, mask_short),
    "A": _run(matched_short, chi_shift, mask_short),
    "CTRL_m": _run(half, chi_match, mask_short),
  }
  borderline = {
    "U1": _run(matched_pair, chi_pair, mask_pair),
    "U2": _run(matched_pair, chi_pair, mask_pair),
    "A": _run(flipped, chi_pair, mask_pair),
    "CTRL_m": _run(matched_pair, chi_pair, mask_pair),
  }
  return {"min_p0@1.0": {"103m_1": failing}, "min_p0@0.3": {"103m_1": borderline}}


def _samples_controls_fail() -> dict[str, dict[str, dict[str, Any]]]:
  n = 8
  length = 4
  mask = [True] * length
  sequences = _rows(1, n, length)
  matched = _chi(0, n, length)
  shifted = _chi(1, n, length)
  body = {
    "U1": _run(sequences, matched, mask),
    "U2": _run(sequences, matched, mask),
    "A": _run(sequences, matched, mask),
    "CTRL_m": _run(sequences, shifted, mask),
  }
  return {"min_p0@1.0": {"103m_1": body}}


def _assemble(
  samples: dict[str, dict[str, dict[str, Any]]],
  cells: list[str],
  tables: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
  structures = {
    "103m_1": {
      "pdb": "/pdbs/103m_1.pdb",
      "relative": "tests/fixtures/laser/103m_1.pdb",
      "sha256": confirm.STRUCTURES["tests/fixtures/laser/103m_1.pdb"],
    },
  }
  previous_cells = confirm.CELLS
  previous_id = confirm.PILOT_RUN_ID
  confirm.CELLS = tables
  confirm.PILOT_RUN_ID = "pilot-not-yet-committed"
  try:
    return confirm.assemble_result(
      samples,
      structures,
      cells,
      n_boot=20,
      bootstrap_seeds=dict.fromkeys(cells, 11),
      n_reused=2,
      n_computed=7,
      script_sha256="abc",
      smoke=False,
    )
  finally:
    confirm.CELLS = previous_cells
    confirm.PILOT_RUN_ID = previous_id


def test_assemble_result_keys_worst_of_and_control_failure() -> None:
  tables = _registered()
  for row in tables.values():
    row["n"] = 8
  tables["min_p0@0.3"]["delta"] = 0.5
  mixed = _assemble(_samples_mixed(), ["min_p0@1.0", "min_p0@0.3"], tables)
  assert set(mixed) == _TOP_LEVEL
  assert mixed["sidecar_verdict"] == "fail"
  assert mixed["controls_all_fail"] is False
  assert mixed["n_controls"] == 2
  assert mixed["pilot_run_id"] == "pilot-not-yet-committed"
  assert mixed["n_reused"] == 2
  assert mixed["n_computed"] == 7
  assert mixed["script_sha256"] == "abc"
  assert mixed["smoke"] is False
  assert mixed["cells_selected"] == ["min_p0@1.0", "min_p0@0.3"]
  chi_fail = mixed["cells"]["min_p0@1.0"]
  assert set(chi_fail) == _CELL_KEYS
  assert set(chi_fail["sequence"]) == _DISTANCE_KEYS
  assert set(chi_fail["chi1"]) == _CHI_KEYS
  assert chi_fail["sequence"]["verdict"] == "pass"
  assert chi_fail["chi1"]["verdict"] == "fail"
  assert chi_fail["chi1"]["delta_chi"] == 0.05
  assert chi_fail["verdict"] == "fail"
  assert chi_fail["m"] == 1.1
  assert chi_fail["h_hat"] == 0.011
  assert chi_fail["n"] == 8
  assert chi_fail["per_structure_delta"] == {"103m_1": 0.0}
  control = chi_fail["controls"]["CTRL_m"]
  assert set(control) == _CONTROL_KEYS
  assert control["sequence"]["verdict"] == "fail"
  assert control["chi1"]["verdict"] == "pass"
  assert control["verdict"] == "fail"
  borderline = mixed["cells"]["min_p0@0.3"]
  assert borderline["sequence"]["verdict"] == "inconclusive"
  assert borderline["chi1"]["verdict"] == "pass"
  assert borderline["verdict"] == "inconclusive"
  assert borderline["controls"]["CTRL_m"]["verdict"] == "pass"
  assert borderline["per_structure_delta"]["103m_1"] == 0.5
  assert borderline["sequence"]["delta"] == 0.5
  passed = _assemble(_samples_controls_fail(), ["min_p0@1.0"], tables)
  assert passed["sidecar_verdict"] == "pass"
  assert passed["controls_all_fail"] is True
  assert passed["n_controls"] == 1
  body = passed["cells"]["min_p0@1.0"]
  assert body["verdict"] == "pass"
  assert body["sequence"]["verdict"] == "pass"
  assert body["chi1"]["verdict"] == "pass"
  assert body["controls"]["CTRL_m"]["sequence"]["verdict"] == "pass"
  assert body["controls"]["CTRL_m"]["chi1"]["verdict"] == "fail"
  assert body["controls"]["CTRL_m"]["verdict"] == "fail"
