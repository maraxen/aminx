"""Guarded statistics for browser-validation scripts (T2, common context "Result-emission
rule" #6).

Every guard here returns `None` (never `NaN`/`inf`/a raised exception) on an undefined
input, so a caller can set the matching `<x>_available = False` and land the case in
`incomplete`/`fail` rather than a spuriously-computed `pass` (result-emission rule #4).

- `guarded_pearson(a, b)` -- `(None, False)` when either array has zero variance,
  else `(r, True)`.
- `guarded_log2_ratio(a, b)` -- `alpha = log2(a / b)`, `None` when `a <= 0 or b <= 0`.
- `guarded_ratio(numerator, denominator)` -- `None` when `denominator <= 0`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from collections.abc import Sequence


def guarded_pearson(
  a: Sequence[float] | np.ndarray,
  b: Sequence[float] | np.ndarray,
) -> tuple[float | None, bool]:
  """Pearson correlation of `a`, `b`; `(None, False)` if either has zero variance.

  A zero-variance array makes the standard Pearson formula divide by zero (or, in
  `numpy.corrcoef`, silently return `nan`); this guard makes that case an explicit,
  gated `None` instead of a `NaN` a caller might forget to check.
  """
  arr_a = np.asarray(a, dtype=np.float64)
  arr_b = np.asarray(b, dtype=np.float64)
  if arr_a.size < 2 or arr_b.size < 2 or float(np.std(arr_a)) == 0.0 or float(np.std(arr_b)) == 0.0:
    return None, False
  return float(np.corrcoef(arr_a, arr_b)[0, 1]), True


def guarded_log2_ratio(a: float, b: float) -> float | None:
  """`log2(a / b)`, or `None` when `a <= 0` or `b <= 0` (log of a non-positive number)."""
  if a <= 0 or b <= 0:
    return None
  return float(math.log2(a / b))


def guarded_ratio(numerator: float, denominator: float) -> float | None:
  """`numerator / denominator`, or `None` when `denominator <= 0`."""
  if denominator <= 0:
    return None
  return float(numerator / denominator)


def classify_gated_result(
  is_available: bool,
  *,
  pass_label: str = "pass",
  incomplete_label: str = "incomplete",
) -> str:
  """First-match-wins classification for a single `<x>_available`-gated pass condition.

  Mirrors every sidecar's own declaration-order contract (common context, "Sidecars":
  "every pass-direction condition also requires `<x>_available = true` for each gated
  float"): a gated quantity that is unavailable (a sanitised `null`/non-finite source,
  result-emission rule #4) can never classify as the pass-direction label, regardless of
  its numeric value -- it always falls through to the residual/incomplete branch.
  """
  return pass_label if is_available else incomplete_label
