"""Tests for `scripts/browser_validation/onnx_audit.py`'s ONNX-graph RNG walker (T2).

Builds tiny synthetic ONNX models with `onnx.helper` (no aminx model, no jax2onnx) to
exercise every branch of `find_onnx_rng_ops`: a clean graph, an RNG op planted inside an
`If` branch, inside a nested `Loop` body, inside a local `FunctionProto` called from the
main graph, and a `com.microsoft`-domain op.

Needs the `$ORTW` overlay (`onnx` is not a base project dependency).
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from scripts.browser_validation.onnx_audit import find_onnx_rng_ops


def _identity_model() -> onnx.ModelProto:
  x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
  y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
  node = helper.make_node("Identity", ["x"], ["y"])
  graph = helper.make_graph([node], "clean", [x], [y])
  return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_clean_graph_returns_empty() -> None:
  assert find_onnx_rng_ops(_identity_model()) == []


def test_random_uniform_in_if_branch_is_flagged() -> None:
  x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
  y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
  cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])

  then_out = helper.make_tensor_value_info("then_y", TensorProto.FLOAT, [4])
  rng_node = helper.make_node("RandomUniform", [], ["then_y"], shape=[4], dtype=TensorProto.FLOAT)
  then_graph = helper.make_graph([rng_node], "then", [], [then_out])

  else_out = helper.make_tensor_value_info("else_y", TensorProto.FLOAT, [4])
  identity_node = helper.make_node("Identity", ["x"], ["else_y"])
  else_graph = helper.make_graph([identity_node], "else", [], [else_out])

  if_node = helper.make_node("If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph)
  graph = helper.make_graph([if_node], "with_if", [x, cond], [y])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

  findings = find_onnx_rng_ops(model)
  assert len(findings) == 1
  assert findings[0].startswith("rng:")
  assert "RandomUniform" in findings[0]


def test_random_uniform_in_nested_loop_body_is_flagged() -> None:
  iter_num = helper.make_tensor_value_info("iter", TensorProto.INT64, [])
  cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
  cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
  acc_in = helper.make_tensor_value_info("acc_in", TensorProto.FLOAT, [4])
  acc_out = helper.make_tensor_value_info("acc_out", TensorProto.FLOAT, [4])

  rng_node = helper.make_node("RandomUniformLike", ["acc_in"], ["acc_out"])
  cond_identity = helper.make_node("Identity", ["cond_in"], ["cond_out"])
  body = helper.make_graph(
    [rng_node, cond_identity],
    "loop_body",
    [iter_num, cond_in, acc_in],
    [cond_out, acc_out],
  )

  x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
  trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
  cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
  y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
  loop_node = helper.make_node("Loop", ["trip_count", "cond", "x"], ["y"], body=body)
  graph = helper.make_graph([loop_node], "with_loop", [x, trip_count, cond], [y])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

  findings = find_onnx_rng_ops(model)
  assert len(findings) == 1
  assert findings[0].startswith("rng:")
  assert "RandomUniformLike" in findings[0]


def _model_with_local_function(rng_inside_function: bool) -> onnx.ModelProto:
  domain = "bv.test"
  fn_name = "PlantedFn"

  if rng_inside_function:
    fn_node = helper.make_node("RandomUniform", [], ["fn_y"], shape=[4], dtype=TensorProto.FLOAT)
  else:
    fn_node = helper.make_node("Identity", ["fn_x"], ["fn_y"])

  function = helper.make_function(
    domain,
    fn_name,
    inputs=["fn_x"],
    outputs=["fn_y"],
    nodes=[fn_node],
    opset_imports=[helper.make_opsetid("", 18)],
  )

  x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
  y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
  call_node = helper.make_node(fn_name, ["x"], ["y"], domain=domain)
  graph = helper.make_graph([call_node], "with_function", [x], [y])
  model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(domain, 1)],
    functions=[function],
  )
  return model


def test_random_uniform_in_local_function_is_flagged() -> None:
  model = _model_with_local_function(rng_inside_function=True)
  findings = find_onnx_rng_ops(model)
  assert len(findings) == 1
  assert findings[0].startswith("rng:")
  assert "RandomUniform" in findings[0]


def test_clean_local_function_is_not_flagged_unknown_domain() -> None:
  # A custom-domain call that RESOLVES to a clean function must not be misreported as
  # unknown-domain: -- function resolution happens before any domain check on the call
  # node (module docstring).
  model = _model_with_local_function(rng_inside_function=False)
  assert find_onnx_rng_ops(model) == []


def test_com_microsoft_domain_op_is_flagged() -> None:
  x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
  y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
  node = helper.make_node("FusedMatMul", ["x"], ["y"], domain="com.microsoft")
  graph = helper.make_graph([node], "with_ms_op", [x], [y])
  model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
  )
  findings = find_onnx_rng_ops(model)
  assert findings == ["ms-domain:FusedMatMul"]


def test_unknown_domain_op_is_flagged() -> None:
  x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
  y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
  node = helper.make_node("MysteryOp", ["x"], ["y"], domain="some.other.vendor")
  graph = helper.make_graph([node], "with_unknown_domain", [x], [y])
  model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("some.other.vendor", 1)],
  )
  findings = find_onnx_rng_ops(model)
  assert findings == ["unknown-domain:some.other.vendor/MysteryOp"]
