"""Tests for `scripts/browser_validation/layer_b_ort_calibrate.py`'s `_max_abs_dist_err`.

Regression coverage for a real bug found while running this task's own gate (see the
commit body): `compute_backbone_distance` adds `1e-6` inside its `sqrt`, so a row's
self-distance (the diagonal) evaluates to `sqrt(1e-6) = 1e-3` against the host
comparator's exact `0` -- swamping the genuine (~ulp-scale) off-diagonal signal this
proxy is meant to measure unless the diagonal is excluded.
"""

from __future__ import annotations

import numpy as np

import pytest

# Skip loudly, with the fix, where the ONNX toolchain is absent. CI installs it (--extra onnx) so these RUN there
# (aminx debt #2392); without this guard a missing package is a collection ERROR that reds the whole job.
for _mod in ('onnx',):
  pytest.importorskip(_mod, reason="needs the ONNX toolchain: uv sync --extra onnx (aminx debt #2392)")

from scripts.browser_validation.layer_b_ort_calibrate import _max_abs_dist_err


def test_max_abs_dist_err_excludes_the_diagonal_self_distance() -> None:
  """The diagonal's own `sqrt(1e-6) - 0 = 1e-3` artifact must not dominate the result;
  the real off-diagonal float32-vs-float64 evaluation-order error is orders of
  magnitude smaller for well-separated points."""
  n = 5
  ca = np.stack([np.arange(n, dtype=np.float32) * 3.8, np.zeros(n), np.zeros(n)], axis=-1)
  x4 = np.zeros((n, 4, 3), dtype=np.float32)
  x4[:, 1, :] = ca

  worst = _max_abs_dist_err([{"x4": x4, "ca_real": ca}])
  assert worst < 1e-4  # far below the diagonal's own 1e-3 artifact (would fail pre-fix)
