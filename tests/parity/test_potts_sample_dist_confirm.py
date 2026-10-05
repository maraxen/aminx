# ruff: noqa: S101
"""CPU checks for the Potts confirmatory distributional script.

No oracle interpreter and no checkpoint: constants, cell refusal, unit seeds,
and result assembly on synthetic tokens.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "parity"
  / "potts_sample_dist_confirm.py"
)


def _load() -> Any:
  spec = importlib.util.spec_from_file_location("potts_sample_dist_confirm_under_test", _SCRIPT)
  if spec is None or spec.loader is None:
    msg = f"cannot load {_SCRIPT}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


confirm = _load()

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
  "confirm_round",
}
_CELL_KEYS = {
  "verdict",
  "mean_delta",
  "upper_95",
  "lower_90",
  "max_delta",
  "delta",
  "m",
  "h_hat",
  "n",
  "controls",
  "per_structure_delta",
}
_CONTROL_KEYS = {"verdict", "mean_delta", "upper_95", "lower_90", "max_delta"}


def test_constants_match_the_preregistered_pilot() -> None:
  assert confirm.PILOT_RUN_ID == "6b47d40c-32e4-434b-b94f-fa2f8c077f2d"
  assert confirm.N_BOOT == 2000
  assert confirm.CELLS == {
    "plain@1.0": {
      "delta": 0.01,
      "m": 1.1,
      "h_hat": 0.0022991985350076035,
      "n": 1000,
    },
    "refine@0.3": {
      "delta": 0.01,
      "m": 1.5,
      "h_hat": 0.0020500332952815914,
      "n": 1000,
    },
  }
  assert "plain@0.3" not in confirm.CELLS
  assert confirm.STRUCTURES == {
    "inputs/example_pdbs/3gg7.pdb": (
      "2203e4a69287a5b968a78df5fb80b29f10fbdb37f73f64363cefbee8f1899712"
    ),
    "inputs/example_pdbs/4jox.pdb": (
      "c16543717793ced9e9475df6213a52054d05b70dd02069d41fac82e164f09ee8"
    ),
    "inputs/example_pdbs/6w25.pdb": (
      "4a9a6dc228bf3953a746d09f29e6116f07c1f2cfc152030fe878728c65cca086"
    ),
    "inputs/example_pdbs/swe1_ligand.pdb": (
      "493352b8c64c02c133f1143c1705e7b5217e0f67127bd22e6b3964319b6445c6"
    ),
  }


def test_cells_rejects_plain_0_3_and_unknown() -> None:
  with pytest.raises(SystemExit, match="plain@0.3"):
    confirm.selected_cells("plain@0.3", smoke=False)
  with pytest.raises(SystemExit, match="unknown cell"):
    confirm.selected_cells("plain@0.1", smoke=False)
  with pytest.raises(SystemExit, match="unknown cell"):
    confirm.selected_cells("tied@0.3", smoke=False)
  assert confirm.selected_cells(None, smoke=False) == ["plain@1.0", "refine@0.3"]
  assert confirm.selected_cells("refine@0.3,plain@1.0", smoke=False) == [
    "plain@1.0",
    "refine@0.3",
  ]


def _structures() -> dict[str, dict[str, str]]:
  return {
    stem: {
      "pdb": f"/pdbs/{stem}.pdb",
      "relative": relative,
      "sha256": digest,
    }
    for relative, digest in confirm.STRUCTURES.items()
    if (stem := Path(relative).stem) in {"3gg7", "4jox"}
  }


def test_unit_specs_runs_and_seeds_are_disjoint_from_the_pilot() -> None:
  used: set[int] = set()
  specs = confirm.unit_specs(
    _structures(),
    ["plain@1.0", "refine@0.3"],
    1000,
    used,
  )
  pilot = confirm.pilot
  by_cell_structure: dict[tuple[str, str], set[str]] = {}
  seeds: list[int] = []
  for spec in specs:
    by_cell_structure.setdefault((spec["cell"], spec["structure"]), set()).add(spec["run"])
    seeds.append(spec["seed"])
    assert spec["seed"] != pilot._hash_seed(spec["unit_id"])
    assert spec["seed"] not in {pilot._hash_seed(other["unit_id"]) for other in specs}
  assert len(seeds) == len(set(seeds))
  for structure in ("3gg7", "4jox"):
    assert by_cell_structure[("refine@0.3", structure)] == {"U1", "U2", "A", "CTRL_m"}
    assert by_cell_structure[("plain@1.0", structure)] == {
      "U1",
      "U2",
      "A",
      "CTRL_m",
      "CTRL_ntoc",
    }
  plain_control = next(
    spec
    for spec in specs
    if spec["cell"] == "plain@1.0" and spec["run"] == "CTRL_m" and spec["structure"] == "3gg7"
  )
  refine_control = next(
    spec
    for spec in specs
    if spec["cell"] == "refine@0.3" and spec["run"] == "CTRL_m" and spec["structure"] == "3gg7"
  )
  assert plain_control["sample_temperature"] == 1.1
  assert refine_control["sample_temperature"] == confirm.CELLS["refine@0.3"]["m"] * 0.3
  assert all(spec["shim"] is (spec["run"] == "U1") for spec in specs)
  assert all(spec["ntoc"] is (spec["run"] == "CTRL_ntoc") for spec in specs)


def _block(token: int, n: int, length: int) -> np.ndarray:
  return np.full((n, length), token, dtype=np.int64)


def _samples(
  *,
  n: int,
  length: int,
  control_token: int,
  ntoc_token: int,
) -> dict[str, dict[str, dict[str, Any]]]:
  mask = np.ones(length, dtype=bool)
  u1 = _block(0, n, length)
  u2 = _block(0, n, length)

  def _run(token: int) -> dict[str, Any]:
    return {"sequences": _block(token, n, length), "mask": mask}

  plain = {
    "U1": _run(0),
    "U2": {"sequences": u2, "mask": mask},
    "A": {"sequences": u1, "mask": mask},
    "CTRL_m": _run(control_token),
    "CTRL_ntoc": _run(ntoc_token),
  }
  refine = {
    "U1": _run(0),
    "U2": {"sequences": u2, "mask": mask},
    "A": {"sequences": u1, "mask": mask},
    "CTRL_m": _run(control_token),
  }
  return {
    "plain@1.0": {"3gg7": plain},
    "refine@0.3": {"3gg7": refine},
  }


def _assemble(
  samples: dict[str, dict[str, dict[str, Any]]],
  *,
  n: int,
) -> dict[str, Any]:
  structures = {
    "3gg7": {
      "pdb": "/pdbs/3gg7.pdb",
      "relative": "inputs/example_pdbs/3gg7.pdb",
      "sha256": confirm.STRUCTURES["inputs/example_pdbs/3gg7.pdb"],
    },
  }
  return confirm.assemble_result(
    samples,
    structures,
    ["plain@1.0", "refine@0.3"],
    n=n,
    n_boot=20,
    bootstrap_seeds={"plain@1.0": 11, "refine@0.3": 13},
    n_reused=2,
    n_computed=7,
    script_sha256="abc",
    smoke=False,
  )


def test_assemble_result_keys_and_control_failure() -> None:
  n = 8
  shifted = _assemble(_samples(n=n, length=4, control_token=3, ntoc_token=4), n=n)
  assert set(shifted) == _TOP_LEVEL
  assert shifted["sidecar_verdict"] == "pass"
  assert shifted["controls_all_fail"] is True
  assert shifted["n_controls"] == 3
  assert shifted["pilot_run_id"] == confirm.PILOT_RUN_ID
  assert shifted["n_reused"] == 2
  assert shifted["n_computed"] == 7
  assert shifted["script_sha256"] == "abc"
  assert shifted["smoke"] is False
  assert shifted["cells_selected"] == ["plain@1.0", "refine@0.3"]
  for cell, expected_controls in (
    ("plain@1.0", {"CTRL_m", "CTRL_ntoc"}),
    ("refine@0.3", {"CTRL_m"}),
  ):
    body = shifted["cells"][cell]
    assert set(body) == _CELL_KEYS
    assert body["verdict"] == "pass"
    assert body["delta"] == 0.01
    assert body["m"] == confirm.CELLS[cell]["m"]
    assert body["h_hat"] == confirm.CELLS[cell]["h_hat"]
    assert body["n"] == n
    assert set(body["controls"]) == expected_controls
    assert body["per_structure_delta"] == {"3gg7": 0.0}
    for control in body["controls"].values():
      assert set(control) == _CONTROL_KEYS
      assert control["verdict"] == "fail"
  matched = _assemble(_samples(n=n, length=4, control_token=3, ntoc_token=0), n=n)
  assert matched["controls_all_fail"] is False
  assert matched["cells"]["plain@1.0"]["controls"]["CTRL_ntoc"]["verdict"] != "fail"
  assert matched["cells"]["plain@1.0"]["controls"]["CTRL_m"]["verdict"] == "fail"


def _unit(kind: str, n: int, *, run: str) -> dict[str, Any]:
  temperature = 0.3 if kind == "refine" else 1.0
  return {
    "unit_id": f"{kind}|{temperature:.1f}|3gg7|{run}|{n}",
    "cell": f"{kind}@{temperature:.1f}",
    "condition": kind,
    "cell_temperature": temperature,
    "sample_temperature": temperature,
    "structure": "3gg7",
    "run": run,
    "m": None,
    "shim": False,
    "kind": kind,
    "n": n,
    "seed": 17,
    "pdb": "/pdbs/3gg7.pdb",
    "pdb_sha256": "abc",
    "ntoc": False,
  }


def test_refine_draws_one_sample_per_call_and_plain_draws_once(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import aminx.host.runner as runner

  length = 3
  calls: list[Any] = []

  def _fake(sampling: Any) -> dict[str, Any]:
    calls.append(sampling)
    n = int(sampling.num_samples)
    token = len(calls)
    key = "refined_sequence" if n == 1 else "sequence"
    return {
      "structures": {
        "0": {
          "arrays": {
            key: np.full((n, length), token, dtype=np.int32),
            "score": np.zeros((n, length), dtype=np.float32),
          },
        },
      },
    }

  monkeypatch.setattr(runner, "sample", _fake)
  monkeypatch.setattr(confirm, "_design_mask_from_pdb", lambda _pdb, _options: [True] * length)

  n = 4
  for run in ("A", "CTRL_m"):
    calls.clear()
    drawn = confirm._sample_aminx(_unit("refine", n, run=run), Path("/ckpt.pt"))
    assert len(calls) == n
    assert [int(call.num_samples) for call in calls] == [1] * n
    assert [int(call.samples_chunk_size) for call in calls] == [1] * n
    seeds = [int(call.random_seed) for call in calls]
    assert seeds == [confirm._prng_seed(17, index) for index in range(n)]
    assert len(set(seeds)) == n
    assert len(drawn["sequences"]) == n
    assert [row[0] for row in drawn["sequences"]] == list(range(1, n + 1))

  calls.clear()
  plain = confirm._sample_aminx(_unit("plain", n, run="A"), Path("/ckpt.pt"))
  assert len(calls) == 1
  assert int(calls[0].num_samples) == n
  assert int(calls[0].samples_chunk_size) == 8
  assert int(calls[0].random_seed) == confirm._prng_seed(17)
  assert len(plain["sequences"]) == n


def _claimed(labels: list[str], prefix: str) -> list[int]:
  used: set[int] = set()
  for label in labels:
    used.add(confirm.pilot._hash_seed(label))
  return [
    confirm.pilot._claim(confirm.pilot._hash_seed(f"{prefix}{label}"), used)
    for label in labels
  ]


def test_round2_unit_seeds_are_disjoint_from_round1_and_the_pilot() -> None:
  assert confirm.CONFIRM_ROUND == 2
  assert confirm._ROUND1_PREFIX == "confirm|"
  assert confirm._WORK == "potts_sample_dist_confirm_r2"
  used: set[int] = set()
  specs = confirm.unit_specs(_structures(), ["plain@1.0", "refine@0.3"], 1000, used)
  labels = [spec["unit_id"] for spec in specs]
  round2 = [spec["seed"] for spec in specs]
  round1 = _claimed(labels, confirm._ROUND1_PREFIX)
  pilot_seeds = [confirm.pilot._hash_seed(label) for label in labels]
  assert round2 == _claimed(labels, f"confirm{confirm.CONFIRM_ROUND}|")
  assert set(round2).isdisjoint(round1)
  assert set(round2).isdisjoint(pilot_seeds)


def test_result_json_carries_confirm_round() -> None:
  import json

  n = 8
  payload = _assemble(_samples(n=n, length=4, control_token=3, ntoc_token=4), n=n)
  decoded = json.loads(json.dumps(payload))
  assert decoded["confirm_round"] == 2
  assert isinstance(decoded["confirm_round"], int)
