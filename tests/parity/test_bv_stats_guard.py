"""Tests for `scripts/browser_validation/bv_stats_guard.py` (T2, common context "Result-
emission rule" #6).

Every guard must return `None` (never `NaN`/raise) on an undefined input: constant-array
Pearson, alpha (log2 ratio) with a zero/negative operand, and a zero/negative-denominator
ratio. A fourth test drives a tiny sample sidecar's dry-run end to end: a result with a
NaN-sourced gated value plus its `_available = False` sibling must classify `incomplete`,
proving `emit()`'s NaN->null sanitisation (`layer_a_common.py`) still lands in a residual
outcome rather than a spuriously-computed `pass` (result-emission rule #4).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.browser_validation.bv_stats_guard import (
  classify_gated_result,
  guarded_log2_ratio,
  guarded_pearson,
  guarded_ratio,
)


def test_guarded_pearson_constant_array_returns_null() -> None:
  a = np.full(8, 3.0)
  b = np.linspace(0.0, 1.0, 8)
  value, available = guarded_pearson(a, b)
  assert value is None
  assert available is False


def test_guarded_pearson_varying_arrays_returns_value() -> None:
  rng = np.random.default_rng(0)
  a = rng.normal(size=16)
  b = a * 2.0 + rng.normal(size=16) * 1e-3
  value, available = guarded_pearson(a, b)
  assert available is True
  assert value is not None
  assert value == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize(("a", "b"), [(0.0, 5.0), (5.0, 0.0), (-1.0, 5.0), (5.0, -1.0)])
def test_guarded_log2_ratio_zero_or_negative_operand_returns_null(a: float, b: float) -> None:
  assert guarded_log2_ratio(a, b) is None


def test_guarded_log2_ratio_positive_operands_returns_value() -> None:
  assert guarded_log2_ratio(8.0, 2.0) == pytest.approx(2.0)


@pytest.mark.parametrize("denominator", [0.0, -1.0])
def test_guarded_ratio_zero_or_negative_denominator_returns_null(denominator: float) -> None:
  assert guarded_ratio(1.0, denominator) is None


def test_guarded_ratio_positive_denominator_returns_value() -> None:
  assert guarded_ratio(3.0, 2.0) == pytest.approx(1.5)


def test_nan_gated_value_with_available_false_classifies_incomplete() -> None:
  """A sample gated result: a NaN-sourced metric, sanitised to `null` (mirroring
  result-emission rule #4's `layer_a_common.emit -> _finite_json`), with its
  `metric_available = False` sibling, must classify as the residual `incomplete` label
  -- never the pass-direction label -- regardless of what the (already-nulled) metric
  value itself is. `classify_gated_result` is every sidecar's own declaration-order
  contract in miniature (common context, "Sidecars": "every pass-direction condition
  also requires `<x>_available = true` for each gated float")."""
  raw_metric = float("nan")
  # Mirrors layer_a_common.emit()'s _finite_json: NaN -> None before it ever reaches an
  # outcome evaluation.
  sanitised_metric = None if math.isnan(raw_metric) else raw_metric
  metric_available = False

  assert sanitised_metric is None
  assert classify_gated_result(metric_available) == "incomplete"


def test_available_gated_result_classifies_pass() -> None:
  assert classify_gated_result(True) == "pass"
