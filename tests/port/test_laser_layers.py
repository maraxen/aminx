"""laser_layers wave.

f64 uses rtol=1e-9. f32 uses rtol=1e-5 and reports every key whose rows exceed
that tolerance, rather than stopping at the first mismatch (debt #2299).

Each tier is one item per (checkpoint, fixture) pair. Those ids are read from
the dump at collection time; an absent oracle yields no pairs and skips.
Tier 1 is also split by precision.

The sealed dump must contain both ``out__`` leaves and the ``in__`` leaves
written by ``scripts/parity/dump_laser_oracles.py``. A dump that only has
outputs cannot replay a layer. Weights come from the LASEr checkpoint under
``AMINX_LASER_ROOT``; a missing oracle directory or a missing torch/checkpoint
skips.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import assert_shape_dtype, names, open_dump, x64_context
from port.reference.laser_layers.algo import OracleAbsentError, load as load_laser_dump

from aminx.model.laser.layers import EquivariantLayerNorm, GVP, HeteroGATv2, HomoGATv2

pytestmark = [pytest.mark.port_wave("laser_layers"), pytest.mark.parity_heavy]

_RTOL = {"f64": 1e-9, "f32": 1e-5}
_CHECKPOINTS = {
  "nothing_heldout": "model_weights/laser_weights_0p1A_nothing_heldout.pt",
  "noise_ligandmpnn_split": "model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt",
  "soluble_65000": "model_weights/soluble_weights_no_heldout_drop_clusters_optstep_65000.pt",
}
_NODE = 256
_EDGE = 128
_HEADS_LIG = 3
_HEADS_PROT = 1
_V_LIG = 15
_V_PROT = 10
_UPSCALE = 4
_DROPOUT = 0.1


def _laser_root() -> Path:
  return Path(os.environ.get("AMINX_LASER_ROOT", "~/repos/LASErMPNN")).expanduser()


_STATE_CACHE: dict[str, dict[str, Any]] = {}


def _state_dict(checkpoint: str) -> dict[str, Any]:
  """Load one checkpoint at most once per session."""
  cached = _STATE_CACHE.get(checkpoint)
  if cached is not None:
    return cached
  torch = pytest.importorskip("torch")
  root = _laser_root()
  if not root.is_dir():
    pytest.skip(f"LASErMPNN checkout absent: {root}")
  path = root / _CHECKPOINTS[checkpoint]
  if not path.is_file():
    pytest.skip(f"checkpoint absent: {path}")
  blob = torch.load(path, map_location="cpu", weights_only=False)
  state = blob["model_state_dict"]
  if not isinstance(state, dict):
    msg = f"{path} model_state_dict is not a dict"
    raise TypeError(msg)
  _STATE_CACHE[checkpoint] = state
  return state


def _set_path(tree: Any, parts: list[str], value: jax.Array) -> Any:
  head, *rest = parts
  if head.isdigit():
    index = int(head)
    seq = list(tree)
    seq[index] = _set_path(seq[index], rest, value) if rest else value
    return tuple(seq)
  child = _set_path(getattr(tree, head), rest, value) if rest else value
  return eqx.tree_at(lambda obj, name=head: getattr(obj, name), tree, child)


def _load_prefix(module: Any, state: dict[str, Any], prefix: str, dtype: np.dtype) -> Any:
  for key, value in state.items():
    text = str(key)
    if not text.startswith(prefix):
      continue
    array = np.asarray(value.detach().cpu().numpy())
    if np.issubdtype(array.dtype, np.floating):
      array = array.astype(dtype, copy=False)
    module = _set_path(module, text[len(prefix) :].split("."), jnp.asarray(array))
  return module


def _pack(
  n_nodes: int,
  edge_index: np.ndarray,
  edge_attr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Scatter an edge list into a dense ``(N, K)`` neighbourhood, sink = row."""
  source = np.asarray(edge_index[0], dtype=np.int32)
  sink = np.asarray(edge_index[1], dtype=np.int32)
  n_edges = int(source.shape[0])
  edge_dim = int(edge_attr.shape[-1]) if edge_attr.ndim > 1 else 1
  attr = np.asarray(edge_attr)
  if attr.ndim == 1:
    attr = attr[:, None]
  if n_nodes == 0 or n_edges == 0:
    empty_i = np.zeros((n_nodes, 0), np.int32)
    empty_a = np.zeros((n_nodes, 0, edge_dim), attr.dtype)
    empty_m = np.zeros((n_nodes, 0), np.bool_)
    return empty_i, empty_a, empty_m, np.zeros((0,), np.int32), np.zeros((0,), np.int32)
  counts = np.zeros((n_nodes,), np.int32)
  slots = np.empty((n_edges,), np.int32)
  for edge in range(n_edges):
    row = int(sink[edge])
    slots[edge] = counts[row]
    counts[row] += 1
  width = int(counts.max())
  neighbours = np.zeros((n_nodes, width), np.int32)
  dense = np.zeros((n_nodes, width, edge_dim), attr.dtype)
  mask = np.zeros((n_nodes, width), np.bool_)
  for edge in range(n_edges):
    row = int(sink[edge])
    slot = int(slots[edge])
    neighbours[row, slot] = source[edge]
    dense[row, slot] = attr[edge]
    mask[row, slot] = True
  return neighbours, dense, mask, slots, sink


def _unpack(dense: np.ndarray, slots: np.ndarray, sink: np.ndarray) -> np.ndarray:
  if slots.shape[0] == 0:
    tail = dense.shape[2:]
    return np.zeros((0, *tail), dtype=dense.dtype)
  out = np.empty((slots.shape[0], *dense.shape[2:]), dtype=dense.dtype)
  for edge in range(int(slots.shape[0])):
    out[edge] = dense[int(sink[edge]), int(slots[edge])]
  return out


def _require(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
  if key not in data.files:
    present = [name for name in data.files if "__in__" in name]
    msg = (
      f"laser_layers dump has no {key}. Layer replay needs the in__ leaves "
      f"recorded by dump_laser_oracles (found {len(present)} in__ keys)."
    )
    raise AssertionError(msg)
  return np.asarray(data[key])


def _gvp_forward(module: GVP, data: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
  scalars = jnp.asarray(_require(data, prefix + "in__0__scalars"))
  vectors = jnp.asarray(_require(data, prefix + "in__0__vectors"))
  out_s, out_v = module(scalars, vectors)
  return {"scalars": np.asarray(out_s), "vectors": np.asarray(out_v)}


def _norm_forward(
  module: EquivariantLayerNorm,
  data: np.lib.npyio.NpzFile,
  prefix: str,
) -> dict[str, np.ndarray]:
  scalars = jnp.asarray(_require(data, prefix + "in__0__scalars"))
  vectors = jnp.asarray(_require(data, prefix + "in__0__vectors"))
  out_s, out_v = module(scalars, vectors)
  return {"scalars": np.asarray(out_s), "vectors": np.asarray(out_v)}


def _homo_forward(
  module: HomoGATv2,
  data: np.lib.npyio.NpzFile,
  prefix: str,
) -> dict[str, np.ndarray]:
  scalars_np = _require(data, prefix + "in__0__scalars")
  vectors = jnp.asarray(_require(data, prefix + "in__0__vectors"))
  edge_index = _require(data, prefix + "in__2")
  edge_attr = _require(data, prefix + "in__1")
  neighbours, dense, mask, slots, sink = _pack(int(scalars_np.shape[0]), edge_index, edge_attr)
  out_s, out_v, out_e = module(
    jnp.asarray(scalars_np),
    vectors,
    jnp.asarray(dense),
    jnp.asarray(neighbours),
    jnp.asarray(mask),
  )
  return {
    "0__scalars": np.asarray(out_s),
    "0__vectors": np.asarray(out_v),
    "1": _unpack(np.asarray(out_e), slots, sink),
  }


def _hetero_forward(
  module: HeteroGATv2,
  data: np.lib.npyio.NpzFile,
  prefix: str,
) -> dict[str, np.ndarray]:
  sink_s = _require(data, prefix + "in__0__scalars")
  sink_v = jnp.asarray(_require(data, prefix + "in__0__vectors"))
  n_prot = int(sink_s.shape[0])
  src_prot = _require(data, prefix + "in__1__0__0__scalars")
  src_lig = _require(data, prefix + "in__1__1__0__scalars")
  packed = []
  for attr_key, index_key in (("in__2__0", "in__3__0"), ("in__2__1", "in__3__1")):
    packed.append(_pack(n_prot, _require(data, prefix + index_key), _require(data, prefix + attr_key)))
  (n0, a0, m0, slots0, sink0), (n1, a1, m1, slots1, sink1) = packed
  out_s, out_v, edges = module(
    jnp.asarray(sink_s),
    sink_v,
    (jnp.asarray(src_prot), jnp.asarray(src_lig)),
    (jnp.asarray(n0), jnp.asarray(n1)),
    (jnp.asarray(a0), jnp.asarray(a1)),
    (jnp.asarray(m0), jnp.asarray(m1)),
    (True, False),
  )
  return {
    "0__scalars": np.asarray(out_s),
    "0__vectors": np.asarray(out_v),
    "1__0": _unpack(np.asarray(edges[0]), slots0, sink0),
    "1__1": _unpack(np.asarray(edges[1]), slots1, sink1),
  }


def _build(layer_id: str, *, key: jax.Array) -> Any:
  if layer_id == "ligand_encoder_input_gvp":
    return GVP((_NODE, 1), (_NODE, _V_LIG), vector_gate=True, key=key)
  if layer_id == "ligand_encoder_gat_layers_0":
    return HomoGATv2(
      _NODE,
      _EDGE,
      _HEADS_LIG,
      _DROPOUT,
      update_edges=True,
      use_mlp_node_update=False,
      atten_head_aggr_layers=0,
      num_vectors=_V_LIG,
      atten_dimension_upscale_factor=None,
      key=key,
    )
  if layer_id == "backbone_frame_vec_input_layer":
    return GVP((0, 4), (0, _V_PROT), vector_gate=False, key=key)
  if layer_id == "backbone_frame_vec_norm":
    return EquivariantLayerNorm((0, _V_PROT), vector_only=True)
  if layer_id == "protein_encoder_layers_0_hetgat":
    return HeteroGATv2(
      2,
      _NODE,
      _EDGE,
      0,
      _HEADS_PROT,
      _DROPOUT,
      num_vectors=_V_PROT,
      use_mlp_node_update=True,
      use_residual_node_update=True,
      compute_edge_updates=True,
      atten_dimension_upscale_factor=_UPSCALE,
      key=key,
    )
  msg = f"unknown laser layer {layer_id}"
  raise KeyError(msg)


_WEIGHT_PREFIX = {
  "ligand_encoder_input_gvp": "ligand_encoder.input_gvp.",
  "ligand_encoder_gat_layers_0": "ligand_encoder.gat_layers.0.",
  "backbone_frame_vec_input_layer": "backbone_frame_vec_input_layer.",
  "backbone_frame_vec_norm": None,
  "protein_encoder_layers_0_hetgat": "protein_encoder_layers.0.hetgat.",
}


def _module_for(
  layer_id: str,
  state: dict[str, Any],
  dtype: np.dtype,
) -> Any:
  key = jax.random.key(0)
  module = _build(layer_id, key=key)
  prefix = _WEIGHT_PREFIX[layer_id]
  if prefix is None:
    return module
  return _load_prefix(module, state, prefix, dtype)


def _predict(layer_id: str, module: Any, data: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
  if layer_id in {"ligand_encoder_input_gvp", "backbone_frame_vec_input_layer"}:
    if not isinstance(module, GVP):
      msg = f"{layer_id} is not a GVP"
      raise TypeError(msg)
    return _gvp_forward(module, data, prefix)
  if layer_id == "backbone_frame_vec_norm":
    if not isinstance(module, EquivariantLayerNorm):
      msg = "backbone norm is not EquivariantLayerNorm"
      raise TypeError(msg)
    return _norm_forward(module, data, prefix)
  if layer_id == "ligand_encoder_gat_layers_0":
    if not isinstance(module, HomoGATv2):
      msg = "ligand GAT is not HomoGATv2"
      raise TypeError(msg)
    return _homo_forward(module, data, prefix)
  if not isinstance(module, HeteroGATv2):
    msg = "protein hetgat is not HeteroGATv2"
    raise TypeError(msg)
  return _hetero_forward(module, data, prefix)


def _rows_over(got: np.ndarray, ref: np.ndarray, rtol: float) -> tuple[int, int, float]:
  error = np.abs(got.astype(np.float64) - ref.astype(np.float64))
  limit = rtol * np.abs(ref.astype(np.float64))
  bad = error > limit
  if bad.ndim == 0:
    return int(bad), 1, float(error)
  row_bad = np.any(bad, axis=tuple(range(1, bad.ndim)))
  return int(np.sum(row_bad)), int(row_bad.shape[0]), float(np.max(error))


def _run(
  data: np.lib.npyio.NpzFile,
  precision: str,
  checkpoint: str,
  fixture: str,
  *,
  numeric: bool,
) -> int:
  compared = 0
  failures: list[str] = []
  worst = 0.0
  dtype = np.float64 if precision == "f64" else np.float32
  rtol = _RTOL[precision]
  base = f"{checkpoint}__{fixture}__layer__"
  layer_ids = sorted(
    {
      key[len(base) :].split("__out__", 1)[0]
      for key in data.files
      if key.startswith(base) and "__out__" in key
    },
  )
  with x64_context(precision), jax.default_matmul_precision("highest"):
    if layer_ids:
      state = _state_dict(checkpoint)
      for layer_id in layer_ids:
        module = _module_for(layer_id, state, dtype)
        prefix = base + layer_id + "__"
        predicted = _predict(layer_id, module, data, prefix)
        out_keys = [key for key in data.files if key.startswith(prefix + "out__")]
        for key in out_keys:
          suffix = key.split("__out__", 1)[1]
          if suffix not in predicted:
            msg = f"{key}: aminx produced no leaf {suffix!r} (have {sorted(predicted)})"
            raise AssertionError(msg)
          got = predicted[suffix]
          ref = np.asarray(data[key])
          label = f"laser_layers {precision} {key}"
          if not numeric:
            assert_shape_dtype(got, ref, label)
          else:
            if got.shape != ref.shape or got.dtype != ref.dtype:
              assert_shape_dtype(got, ref, label)
            n_bad, n_rows, max_abs = _rows_over(got, ref, rtol)
            worst = max(worst, max_abs)
            if n_bad:
              failures.append(f"{key}: {n_bad}/{n_rows} rows exceed rtol={rtol:g}")
          compared += 1
  if numeric and failures:
    report = "\n".join(failures)
    msg = f"Max absolute difference: {worst:.6e}\n{report}"
    raise AssertionError(msg)
  return compared


def _oracle_pairs() -> list[tuple[str, str]]:
  """Read checkpoint and fixture ids from the dump at collection time.

  Returns ``[]`` when the oracle is absent so parametrization collects no
  cases and the wave skips instead of erroring. Only those two arrays are
  read; the archive is closed before return.
  """
  data: np.lib.npyio.NpzFile | None = None
  for precision in ("f64", "f32"):
    try:
      data = load_laser_dump(precision)
    except OracleAbsentError:
      continue
    break
  if data is None:
    return []
  try:
    checkpoints = names(data["checkpoint_ids"])
    fixtures = names(data["fixture_names"])
  finally:
    data.close()
  return [(checkpoint, fixture) for checkpoint in checkpoints for fixture in fixtures]


_PAIRS = _oracle_pairs()
_PAIR_IDS = [f"{checkpoint}-{fixture}" for checkpoint, fixture in _PAIRS]


def _skip_absent_oracle() -> None:
  if not _PAIRS:
    pytest.skip("laser_layers oracle absent")


@pytest.mark.tier_1
@pytest.mark.parametrize("precision", ("f64", "f32"))
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_1_dtype_shape(
  oracle: object,
  precision: str,
  checkpoint: str,
  fixture: str,
) -> None:
  """Hooked layer outputs match the sealed dtype and shape."""
  _skip_absent_oracle()
  data = open_dump(oracle, precision)
  try:
    assert _run(data, precision, checkpoint, fixture, numeric=False) > 0
  finally:
    data.close()


@pytest.mark.tier_2
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_2_f64(oracle: object, checkpoint: str, fixture: str) -> None:
  """f64 layer outputs match at rtol=1e-9 under x64."""
  _skip_absent_oracle()
  data = open_dump(oracle, "f64")
  try:
    assert _run(data, "f64", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_3_f32(oracle: object, checkpoint: str, fixture: str) -> None:
  """f32 layer outputs match at rtol=1e-5, reporting every exceeding key."""
  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_5
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_5_trace_budget(
  oracle: object,
  max_traces: int,
  checkpoint: str,
  fixture: str,
) -> None:
  """One static GVP shape traces at most ``max_traces`` times."""
  import chex

  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    base = f"{checkpoint}__{fixture}__layer__"
    assert any(key.startswith(base) and "__out__" in key for key in data.files)
  finally:
    data.close()
  module = GVP((_NODE, 1), (_NODE, _V_LIG), vector_gate=True, key=jax.random.key(0))
  scalars = jnp.zeros((_NODE, _NODE), dtype=jnp.float32)
  vectors = jnp.zeros((_NODE, 1, 3), dtype=jnp.float32)
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(layer: GVP, nodes: jax.Array, frames: jax.Array) -> jax.Array:
    out_s, _out_v = layer(nodes, frames)
    return out_s

  compiled = eqx.filter_jit(kernel)
  compiled(module, scalars, vectors)
  compiled(module, scalars, vectors)
