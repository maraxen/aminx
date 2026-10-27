"""declayer_f64 wave: DecoderLayer against the sealed f64 fixture (rtol=1e-12).

Skipped until A2 lands ``aminx.families.potts_mpnn.model.PottsMPNN``. The f32
tier skips because the dump is f64 only.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import assert_close, assert_shape_dtype, open_dump, x64_context

from aminx.model.decoder import DecoderLayer

pytestmark = [pytest.mark.port_wave("declayer_f64"), pytest.mark.parity_heavy]

_RTOL = {"f64": 1e-12, "f32": 1e-4}
_ATOL = {"f64": 1e-12, "f32": 1e-5}


def _require_model() -> None:
  module = pytest.importorskip("aminx.families.potts_mpnn.model")
  if getattr(module, "PottsMPNN", None) is None:
    pytest.skip("PottsMPNN pending A2")


def _with_linear(
  linear: eqx.nn.Linear, weight: np.ndarray, bias: np.ndarray, precision: str = "f64"
) -> eqx.nn.Linear:
  dtype = jnp.float64 if precision == "f64" else jnp.float32
  linear = eqx.tree_at(lambda layer: layer.weight, linear, jnp.asarray(weight, dtype=dtype))
  return eqx.tree_at(lambda layer: layer.bias, linear, jnp.asarray(bias, dtype=dtype))


def _with_norm(
  norm: eqx.nn.LayerNorm, weight: np.ndarray, bias: np.ndarray, precision: str = "f64"
) -> eqx.nn.LayerNorm:
  dtype = jnp.float64 if precision == "f64" else jnp.float32
  norm = eqx.tree_at(lambda layer: layer.weight, norm, jnp.asarray(weight, dtype=dtype))
  return eqx.tree_at(lambda layer: layer.bias, norm, jnp.asarray(bias, dtype=dtype))


def _decoder(data: np.lib.npyio.NpzFile, precision: str = "f64") -> DecoderLayer:
  h_v = np.asarray(data["h_V"])
  h_e = np.asarray(data["h_E"])
  layer = DecoderLayer(
    int(h_v.shape[-1]),
    int(h_e.shape[-1]),
    int(h_v.shape[-1]) * 4,
    key=jax.random.key(0),
  )
  message_weights = (
    (data["weight__W1__weight"], data["weight__W1__bias"]),
    (data["weight__W2__weight"], data["weight__W2__bias"]),
    (data["weight__W3__weight"], data["weight__W3__bias"]),
  )
  message_layers = tuple(
    _with_linear(linear, weight, bias, precision)
    for linear, (weight, bias) in zip(layer.message_mlp.layers, message_weights, strict=True)
  )
  message = eqx.tree_at(lambda mlp: mlp.layers, layer.message_mlp, message_layers)
  dense_weights = (
    (data["weight__dense__W_in__weight"], data["weight__dense__W_in__bias"]),
    (data["weight__dense__W_out__weight"], data["weight__dense__W_out__bias"]),
  )
  dense_layers = tuple(
    _with_linear(linear, weight, bias, precision)
    for linear, (weight, bias) in zip(layer.dense.layers, dense_weights, strict=True)
  )
  dense = eqx.tree_at(lambda mlp: mlp.layers, layer.dense, dense_layers)
  norm1 = _with_norm(layer.norm1, data["weight__norm1__weight"], data["weight__norm1__bias"], precision)
  norm2 = _with_norm(layer.norm2, data["weight__norm2__weight"], data["weight__norm2__bias"], precision)
  layer = eqx.tree_at(lambda dec: dec.message_mlp, layer, message)
  layer = eqx.tree_at(lambda dec: dec.dense, layer, dense)
  layer = eqx.tree_at(lambda dec: dec.norm1, layer, norm1)
  return eqx.tree_at(lambda dec: dec.norm2, layer, norm2)


def _inputs(
  data: np.lib.npyio.NpzFile,
  precision: str = "f64",
) -> tuple[jax.Array, jax.Array, jax.Array, float, np.ndarray]:
  dtype = jnp.float64 if precision == "f64" else jnp.float32
  np_dtype = np.float64 if precision == "f64" else np.float32
  nodes = jnp.asarray(data["h_V"][0], dtype=dtype)
  edges = jnp.asarray(data["h_E"][0], dtype=dtype)
  mask = jnp.asarray(data["mask_V"][0], dtype=dtype)
  scale = float(np.asarray(data["scale"]))
  ref = np.asarray(data["output"][0], dtype=np_dtype)
  return nodes, edges, mask, scale, ref


def _predict(data: np.lib.npyio.NpzFile, precision: str = "f64") -> tuple[np.ndarray, np.ndarray]:
  layer = _decoder(data, precision)
  nodes, edges, mask, scale, ref = _inputs(data, precision)
  with x64_context(precision), jax.default_matmul_precision("highest"):
    pred = np.asarray(layer(nodes, edges, mask, scale=scale, inference=True))
  return pred, ref


@pytest.mark.tier_1
def test_tier_1_dtype_shape(oracle: object) -> None:
  """DecoderLayer output matches the sealed f64 dtype and shape."""
  _require_model()
  data = open_dump(oracle, "f64")
  try:
    # The fixture is f64 only, so the dtype check needs x64 like tier 2.
    with x64_context("f64"):
      pred, ref = _predict(data, "f64")
  finally:
    data.close()
  assert_shape_dtype(pred, ref, "declayer_f64")


@pytest.mark.tier_2
def test_tier_2_f64(oracle: object) -> None:
  """f64 DecoderLayer matches the sealed output at rtol=1e-12, atol=1e-12."""
  _require_model()
  data = open_dump(oracle, "f64")
  try:
    pred, ref = _predict(data, "f64")
  finally:
    data.close()
  assert_close(pred, ref, rtol=_RTOL["f64"], atol=_ATOL["f64"], label="declayer_f64")


@pytest.mark.tier_3
def test_tier_3_f32(oracle: object) -> None:
  """f32 DecoderLayer matches at rtol=1e-4, atol=1e-5."""
  _require_model()
  data = open_dump(oracle, "f64")
  try:
    pred, ref = _predict(data, "f32")
  finally:
    data.close()
  assert_close(pred, ref, rtol=_RTOL["f32"], atol=_ATOL["f32"], label="declayer_f64")


@pytest.mark.tier_5
def test_tier_5_trace_budget(oracle: object, max_traces: int) -> None:
  """One static shape traces at most ``max_traces`` times."""
  import chex

  _require_model()
  data = open_dump(oracle, "f64")
  try:
    layer = _decoder(data)
    nodes, edges, mask, scale, _ref = _inputs(data)
  finally:
    data.close()
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(
    module: DecoderLayer,
    h_v: jax.Array,
    h_e: jax.Array,
    row_mask: jax.Array,
  ) -> jax.Array:
    return module(h_v, h_e, row_mask, scale=scale, inference=True)

  compiled = eqx.filter_jit(kernel)
  with x64_context("f64"):
    compiled(layer, nodes, edges, mask)
    compiled(layer, nodes, edges, mask)
