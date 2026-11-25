"""Torch-free numeric helpers shared by the ThermoMPNN parity script and knob tests."""

from __future__ import annotations

import numpy as np


def pearson_r(x: object, y: object) -> float:
  """Return the Pearson correlation of the flattened arrays ``x`` and ``y``.

  Parameters
  ----------
  x : object
    Array-like of predictions; flattened before use.
  y : object
    Array-like of reference values with the same number of elements.

  Returns
  -------
  float
    Correlation in ``[-1, 1]``, or ``NaN`` when either input has zero variance or
    contains NaN. ``NaN`` (not 1.0) keeps a constant output from passing an
    ``r > threshold`` gate.
  """
  x_flat = np.asarray(x, dtype=np.float64).ravel()
  y_flat = np.asarray(y, dtype=np.float64).ravel()
  x_c = x_flat - np.mean(x_flat)
  y_c = y_flat - np.mean(y_flat)
  denom = float(np.sqrt(np.sum(x_c**2) * np.sum(y_c**2)))
  if not denom > 0.0:
    return float("nan")
  return float(np.sum(x_c * y_c) / denom)
