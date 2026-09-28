"""Tests for `scripts/browser_validation/layer_b_build.py`'s ORT execution gate
(260926_browser-export-loop, T2b).

`ort_execution_check` is the check T2's own build gate never had: conversion success
through `jax2onnx.to_onnx` is not evidence an artifact is usable. Every P03/P04
artifact T2 recorded `pass` on in fact failed `onnxruntime.InferenceSession`
construction outright (`ShapeInferenceError` at node `node_Squeeze_24`, root-caused to
`aminx.utils.coordinates.compute_backbone_coordinates` reading an atom-axis index of 4
(`atom_order["O"]`) on the export wrappers' compact `(L, 4, 3)` backbone-only input,
whose valid range is 0..3).

Builds tiny synthetic ONNX models with `onnx.helper` (no aminx model, no jax2onnx) that
reproduce the EXACT failure shape: a `Slice` on a 4-wide axis followed by a `Squeeze`
of the sliced (size-1-expected) dimension -- `node_Squeeze_24`'s own shape. The
negative control (`start=4`, out of range) must be flagged `ort_ok=False`; the positive
control (`start=3`, in range) must be flagged `ort_ok=True` -- a check that can only
fail, or only pass, is not a check (BATHOS.md).

Needs the `$ORTW` overlay (`onnx`/`onnxruntime` are not base project dependencies).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from scripts.browser_validation.layer_b_build import ort_execution_check

#: jax2onnx's own real exports use IR version 10 (confirmed against a real built
#: artifact); onnxruntime 1.30.0 rejects onnx==1.23.0's default IR version (14)
#: outright with an unrelated "Unsupported model IR version" load error, which would
#: mask the ShapeInferenceError this test exists to exercise if left unpinned.
_IR_VERSION = 10


def _single_atom_slice_model(
  start: int, atom_axis_size: int = 4, length: int = 6
) -> onnx.ModelProto:
  """A minimal ``in_0[:, start, :]``-style read: `Slice` (axis 1) then `Squeeze` of
  that axis -- the same two-node shape as the real defect's `node_Squeeze_24`, fed by
  a dynamic-slice-lowered read of a 4-wide atom axis at an out-of-range index."""
  x = helper.make_tensor_value_info("in_0", TensorProto.FLOAT, [length, atom_axis_size, 3])
  y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [length, 3])
  starts = helper.make_tensor("starts", TensorProto.INT64, [1], [start])
  ends = helper.make_tensor("ends", TensorProto.INT64, [1], [start + 1])
  axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
  slice_node = helper.make_node("Slice", ["in_0", "starts", "ends", "axes"], ["sliced"])
  squeeze_node = helper.make_node("Squeeze", ["sliced", "axes"], ["y"])
  graph = helper.make_graph(
    [slice_node, squeeze_node],
    f"single_atom_slice_start{start}",
    [x],
    [y],
    initializer=[starts, ends, axes],
  )
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
  model.ir_version = _IR_VERSION
  return model


def test_out_of_range_slice_artifact_is_flagged_not_ok(tmp_path: Path) -> None:
  """Negative control: `start=4` on a 4-wide axis (valid range 0..3) is exactly the
  real defect's shape -- ONNX Runtime's static shape inference computes an empty
  `[length, 0, 3]` slice and the downstream `Squeeze` rejects it at
  `InferenceSession` construction. Must be flagged `ort_ok=False`."""
  onnx_path = tmp_path / "out_of_range.onnx"
  onnx.save(_single_atom_slice_model(start=4), str(onnx_path))
  inputs = (np.zeros((6, 4, 3), dtype=np.float32),)

  result = ort_execution_check(onnx_path, inputs)

  assert result["ort_ok"] is False
  assert result["error"] is not None
  assert "Squeeze" in result["error"] or "ShapeInferenceError" in result["error"]


def test_in_range_slice_artifact_is_flagged_ok(tmp_path: Path) -> None:
  """Positive control: `start=3` is the last in-range column of a 4-wide axis (the
  compact export layout's own O column) -- an otherwise-identical graph must build an
  `InferenceSession`, `.run()` successfully, and produce finite output. Paired with the
  negative control above: a check that can only fail (or only pass) is not a check."""
  onnx_path = tmp_path / "in_range.onnx"
  onnx.save(_single_atom_slice_model(start=3), str(onnx_path))
  inputs = (np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3),)

  result = ort_execution_check(onnx_path, inputs)

  assert result == {"ort_ok": True, "error": None}


def test_missing_artifact_is_flagged_not_ok(tmp_path: Path) -> None:
  """A nonexistent path must not raise out of `ort_execution_check` -- it is recorded
  as a failed check, same as any other `InferenceSession` construction error."""
  onnx_path = tmp_path / "does_not_exist.onnx"
  inputs = (np.zeros((6, 4, 3), dtype=np.float32),)

  result = ort_execution_check(onnx_path, inputs)

  assert result["ort_ok"] is False
  assert result["error"] is not None
