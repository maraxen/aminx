# ruff: noqa: S101, SLF001
"""Finiteness rule of ``scripts/parity/antibmpnn_parity_v3.py`` (grading functions only; no JAX, no sealed inputs).

A non-finite or shape-mismatched clean cell must FAIL the clean verdict; a mutant trips only on a FINITE over-band
diff or a discrete flip; every aggregate max is NaN-safe. The v2 vehicle passed a NaN cell (``nan > tol`` is False).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "antibmpnn_parity_v3.py"


@pytest.fixture(scope="module")
def v3() -> ModuleType:
  sys.path[:0] = [str(_PATH.parents[1]), str(_PATH.parent)]
  spec = importlib.util.spec_from_file_location("antibmpnn_parity_v3_under_test", _PATH)
  assert spec is not None
  assert spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _cell(ckpt: str, stage: str, gap: float = 1e-6, floor: float = 1e-4) -> dict:
  return {"ckpt": ckpt, "stage": stage, "gap_same": gap, "gap_f64": gap, "floor": floor}


def _clean(**over: object) -> dict:
  stages = ("features", "enc_node", "enc_edge", "logits")
  payload = {
    "f32": {"cells": [_cell("antibmpnn_000", s) for s in stages], "nbr_mismatch": 0},
    "f64": {"cells": [_cell("antibmpnn_000", s, 1e-13, 1e-4) for s in stages], "nbr_mismatch": 0},
    "entry": {"cells": [_cell("antibmpnn_000", "entry_tf")]},
    "sampler": 3e-5, "sampler_token_mismatch": 0, "parse": 0.0,
    "descriptors": {"csv": 5e-5, "direct": 1e-6, "ph_sweep": 1e-7},
    "issues": {"nonfinite": [], "shape_mismatch": []},
  }  # fmt: skip
  payload.update(over)
  return payload


def test_clean_baseline_passes(v3: ModuleType) -> None:
  verdict, flat, _ = v3._summarize_clean(_clean())
  assert verdict == "pass"
  assert flat["nonfinite"] == 0
  assert flat["n_shape_mismatch"] == 0


@pytest.mark.parametrize("where", ["f32", "f64", "entry"])
def test_nan_cell_fails_clean_even_without_a_recorded_issue(v3: ModuleType, where: str) -> None:
  payload = _clean()
  payload[where]["cells"][0]["gap_same"] = math.nan
  payload[where]["cells"][0]["gap_f64"] = math.nan
  verdict, flat, _ = v3._summarize_clean(payload)
  assert verdict != "pass"
  assert flat["nonfinite"] >= 1


def test_nan_floor_and_inf_shape_fail_clean(v3: ModuleType) -> None:
  payload = _clean()
  payload["f32"]["cells"][1]["floor"] = math.nan
  assert v3._summarize_clean(payload)[0] != "pass"
  payload = _clean()
  payload["entry"]["cells"][0]["gap_f64"] = math.inf
  verdict, flat, _ = v3._summarize_clean(payload)
  assert verdict != "pass"
  assert flat["n_shape_mismatch"] >= 1


@pytest.mark.parametrize("key", ["sampler", "parse"])
def test_nan_scalar_fails_clean(v3: ModuleType, key: str) -> None:
  payload = _clean(**{key: math.nan, "issues": {"nonfinite": [key], "shape_mismatch": []}})
  assert v3._summarize_clean(payload)[0] != "pass"


def test_nan_descriptor_fails_clean(v3: ModuleType) -> None:
  payload = _clean(descriptors={"csv": math.nan, "direct": 0.0, "ph_sweep": 0.0}, issues={"nonfinite": ["d"], "shape_mismatch": []})
  assert v3._summarize_clean(payload)[0] != "pass"


def test_recorded_shape_mismatch_fails_clean(v3: ModuleType) -> None:
  assert v3._summarize_clean(_clean(issues={"nonfinite": [], "shape_mismatch": ["nbr"]}))[0] != "pass"


def test_maxabs_records_and_pmax_is_order_independent(v3: ModuleType) -> None:
  v3._ISSUES["nonfinite"].clear()
  v3._ISSUES["shape_mismatch"].clear()
  assert math.isnan(v3._maxabs([1.0, math.nan], [1.0, 1.0], "a"))
  assert v3._maxabs([1.0, 2.0], [1.0], "b") == math.inf
  assert v3._ISSUES == {"nonfinite": ["a"], "shape_mismatch": ["b"]}
  assert math.isnan(v3._pmax(1.0, math.nan))
  assert math.isnan(v3._pmax(math.nan, 1.0))
  assert v3._fmax([1.0, math.nan, math.inf, 3.0]) == 3.0


def _mutant_payload(**over: object) -> dict:
  payload = _clean()
  payload.update(over)
  return payload


def test_mutant_nan_only_does_not_trip(v3: ModuleType) -> None:
  nan_cells = [_cell("antibmpnn_000", "logits", math.nan, math.nan)]
  payload = _mutant_payload(f32={"cells": nan_cells, "nbr_mismatch": 0})
  status, value = v3._mutant_status("reversed_ar_mask", payload)
  assert status == "passed"
  assert value == 0.0
  inf_cells = [_cell("antibmpnn_000", "logits", math.inf)]
  assert v3._mutant_status("reversed_ar_mask", _mutant_payload(f32={"cells": inf_cells, "nbr_mismatch": 0}))[0] == "passed"


def test_mutant_finite_over_band_trips_even_with_a_nan_sibling(v3: ModuleType) -> None:
  cells = [_cell("antibmpnn_000", "logits", math.nan), _cell("antibmpnn_010", "logits", 3.0)]
  status, value = v3._mutant_status("reversed_ar_mask", _mutant_payload(f32={"cells": cells, "nbr_mismatch": 0}))
  assert status == "failed"
  assert value == 3.0
  # order independence: NaN last
  status, _ = v3._mutant_status("reversed_ar_mask", _mutant_payload(f32={"cells": cells[::-1], "nbr_mismatch": 0}))
  assert status == "failed"


def test_sampler_token_flip_counts_only_with_a_finite_distribution(v3: ModuleType) -> None:
  assert v3._mutant_status("omit_penalty_dropped", _mutant_payload(sampler=1e-5, sampler_token_mismatch=4))[0] == "failed"
  assert v3._mutant_status("omit_penalty_dropped", _mutant_payload(sampler=math.nan, sampler_token_mismatch=4))[0] == "passed"
  assert v3._mutant_status("omit_penalty_dropped", _mutant_payload(sampler=math.inf, sampler_token_mismatch=4))[0] == "passed"


def test_crashed_arm_is_error_not_trip(v3: ModuleType) -> None:
  assert v3._mutant_status("offset_shift", {"error": "boom"})[0] == "error"
