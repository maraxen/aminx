"""CPU checks for the P07 sampling-knobs gate. No Chromium and no reference weights."""

# ruff: noqa: S101

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from scripts.browser_validation.p07_knobs_gate import (
  OUTCOMES,
  P07_KEYS,
  SplitMix64,
  _json_default,
  _node_bin,
  bad_gumbel_from_uniform,
  build_p07_inputs,
  evaluate_outcome,
  gumbel_from_uniform,
  js_build,
  order_counts,
  passing_result,
  result_template,
  summarize_gumbel,
  summarize_orders,
  tie_group_map,
  uniform_from_u32,
)

_ROOT = Path(__file__).resolve().parents[2]
_SIDECAR = _ROOT / "scripts" / "browser_validation" / "p07_knobs_gate.bth.toml"
_CASES = _ROOT / "scripts" / "browser_validation" / "p07_knobs_gate.cases.json"


def _noise(transform: str) -> Callable[[int, int, int], np.ndarray]:
  fn = gumbel_from_uniform if transform == "gumbel" else bad_gumbel_from_uniform

  def noise(seed: int, rows: int, cols: int) -> np.ndarray:
    rng = SplitMix64(seed)
    out = np.empty((rows, cols), dtype=np.float32)
    flat = out.reshape(-1)
    for index in range(flat.shape[0]):
      flat[index] = fn(uniform_from_u32(rng.next_u32()))
    return out

  return noise


def test_tie_ids_stay_in_range() -> None:
  length = 12
  ids = tie_group_map(length, [[4, 1, 9], [2, 7]])
  assert int(ids.min()) >= 0
  assert int(ids.max()) < length
  assert int(ids[4]) == 1
  assert int(ids[1]) == 1
  assert int(ids[9]) == 1
  assert int(ids[2]) == 2
  assert int(ids[7]) == 2
  assert int(ids[0]) == 0


def test_runspec_matches_python_for_each_field() -> None:
  length = 6
  coords = np.arange(length * 12, dtype=np.float32).reshape(length, 4, 3)
  structure = {
    "coords": coords,
    "coords_shape": [length, 4, 3],
    "mask": np.ones((length,), dtype=np.float32),
    "residue_index": np.arange(length, dtype=np.int32),
    "chain_index": np.zeros((length,), dtype=np.int32),
    "chain_ids": ["A", "A", "A", "B", "B", "B"],
    "native_tokens": (np.arange(length, dtype=np.int32) % 20),
  }
  specs: dict[str, dict[str, object]] = {
    "temperature": {"seed": 3, "decoding_order": [5, 4, 3, 2, 1, 0]},
    "bias_AA": {"seed": 4, "bias_AA": {"A": 1.5}, "decoding_order": "random"},
    "omit": {
      "seed": 5,
      "omit_AA": "CX",
      "omit_AA_per_residue": {"1": "W"},
      "decoding_order": [0, 1, 2, 3, 4, 5],
    },
    "fixed_positions": {"seed": 6, "fixed_positions": {"0": "A"}, "decoding_order": "random"},
    "chains_to_design": {"seed": 7, "chains_to_design": ["A"], "decoding_order": "random"},
    "tied_positions": {"seed": 8, "tied_positions": [[4, 1, 3]], "decoding_order": "random"},
    "bias_AA_per_residue": {
      "seed": 9,
      "bias_AA_per_residue": {"2": {"G": 0.25}},
      "decoding_order": [0, 1, 2, 3, 4, 5],
    },
  }
  node = _node_bin(None)
  for name, spec in specs.items():
    blob = json.loads(json.dumps({"structure": structure, "runspec": spec}, default=_json_default))
    py = build_p07_inputs(blob["structure"], blob["runspec"])
    js = js_build(blob["structure"], blob["runspec"], node)
    for key in P07_KEYS:
      assert np.array_equal(py[key], js[key]), (name, key)


def test_b2_statistic_passes_on_the_real_transform_and_fails_on_controls() -> None:
  good = summarize_gumbel(_noise("gumbel"), 20_000, 50_000)
  bad = summarize_gumbel(_noise("bad"), 20_000, 50_000)
  assert good["ok"], good
  assert not bad["ok"], bad
  fair = summarize_orders(order_counts(20_000, 4, 7, kind="fisher"))
  biased = summarize_orders(order_counts(20_000, 4, 7, kind="biased"))
  assert fair["ok"], fair
  assert not biased["ok"], biased


def test_outcome_pass_and_each_failure_and_sidecar_schema() -> None:
  doc = tomllib.loads(_SIDECAR.read_text())
  for name, expr in OUTCOMES:
    assert doc["outcomes"][name]["condition"] == expr
  assert set(doc["result_schema"]) == set(result_template())
  assert evaluate_outcome(passing_result()) == "pass"
  for row in json.loads(_CASES.read_text()):
    assert evaluate_outcome(row["result"]) == row["expect"], row["name"]


def test_cases_grade_with_the_pinned_bathos_evaluator() -> None:
  """The local evaluator above mirrors the sidecar; this grades it with bathos itself."""
  sidecar_mod = pytest.importorskip("bathos.sidecar")
  sidecar = sidecar_mod.parse_sidecar(_SIDECAR)
  for row in json.loads(_CASES.read_text()):
    got = sidecar_mod.evaluate_outcome(sidecar, row["result"])
    got = getattr(got, "outcome", got)
    assert got == row["expect"], row["name"]
  passing = sidecar_mod.evaluate_outcome(sidecar, passing_result())
  assert getattr(passing, "outcome", passing) == "pass"
