# ruff: noqa: S101
"""PottsMPNN composition, self-edge invariant, and DecoderLayer f64 fixture."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.model.decoder import DecoderLayer

_REPO = Path(__file__).resolve().parents[3]
_REFERENCE = _REPO / "tests/port/reference/a1_potts"
_CONVERTER = _REPO / "scripts/recapture/pottsmpnn_model_to_eqx.py"


def _converter():
  spec = importlib.util.spec_from_file_location("pottsmpnn_model_to_eqx", _CONVERTER)
  if spec is None or spec.loader is None:
    msg = f"cannot load {_CONVERTER}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _oracle_dir() -> Path:
  raw = os.environ.get("AMINX_A1_ORACLE_DIR")
  if raw:
    return Path(raw)
  return Path.home() / "projects/aminx-oracles/dumps/a1_potts"


def _expected_sha(suffix: str) -> str:
  for line in (_REFERENCE / "oracles.sha256").read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    digest, path = line.split(maxsplit=1)
    if Path(path.strip()).as_posix().endswith(suffix):
      return digest
  msg = f"{suffix} is not listed in oracles.sha256"
  raise AssertionError(msg)


def _backbone(length: int, dtype: jnp.dtype) -> jax.Array:
  index = jnp.arange(length, dtype=dtype)
  ca = jnp.stack((index * jnp.asarray(3.8, dtype=dtype), jnp.zeros(length, dtype), jnp.zeros(length, dtype)), axis=-1)
  nitrogen = ca + jnp.asarray([-0.5, 1.0, 0.0], dtype=dtype)
  carbon = ca + jnp.asarray([1.5, 0.0, 0.0], dtype=dtype)
  oxygen = carbon + jnp.asarray([0.0, 1.2, 0.0], dtype=dtype)
  return jnp.stack((nitrogen, ca, carbon, oxygen), axis=1)


def _call(model: PottsMPNN, coords: jax.Array, mask: jax.Array) -> object:
  length = coords.shape[0]
  residue_index = jnp.arange(length, dtype=jnp.int32)
  chain_index = jnp.zeros((length,), dtype=jnp.int32)
  sequence = jnp.zeros((length,), dtype=jnp.int32)
  decoding_order = jnp.arange(length, dtype=jnp.int32)
  return model(coords, mask, residue_index, chain_index, sequence, decoding_order)


def test_self_edge_on_present_rows() -> None:
  length = 6
  coords = _backbone(length, jnp.float32)
  mask = jnp.ones((length,), dtype=jnp.float32).at[2].set(0.0)
  model = PottsMPNN(key=jax.random.key(0))
  output = _call(model, coords, mask)
  present = np.asarray(mask) > 0
  indices = np.asarray(output.neighbor_indices)
  expected = np.arange(length)
  np.testing.assert_array_equal(indices[present, 0], expected[present])


def test_dtype_follows_coords() -> None:
  previous = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", True)
  try:
    model = PottsMPNN(key=jax.random.key(1))
    for dtype in (jnp.float32, jnp.float64):
      coords = _backbone(5, dtype)
      mask = jnp.ones((5,), dtype=dtype)
      output = _call(model, coords, mask)
      assert output.log_probs.dtype == dtype
      assert output.etab_raw.dtype == dtype
      assert output.enc_h_v[-1].dtype == dtype
      assert output.dec_h_v[-1].dtype == dtype
  finally:
    jax.config.update("jax_enable_x64", previous)


def test_jit_matches_eager() -> None:
  coords = _backbone(5, jnp.float32)
  mask = jnp.ones((5,), dtype=jnp.float32)
  model = PottsMPNN(key=jax.random.key(2))

  @eqx.filter_jit
  def _run(module: PottsMPNN, xyz: jax.Array, residue_mask: jax.Array):
    return _call(module, xyz, residue_mask)

  eager = _call(model, coords, mask)
  compiled = _run(model, coords, mask)
  np.testing.assert_allclose(np.asarray(compiled.log_probs), np.asarray(eager.log_probs), atol=1e-6, rtol=1e-6)
  np.testing.assert_allclose(np.asarray(compiled.etab_raw), np.asarray(eager.etab_raw), atol=1e-6, rtol=1e-6)


def test_registry_excludes_proteinmpnn_compatible() -> None:
  converter = _converter()
  registered = [checkpoint_id for checkpoint_id, _path in converter.IN_SCOPE_CHECKPOINTS]
  assert len(registered) == 5
  for checkpoint_id in registered:
    assert "proteinmpnn_compatible" not in checkpoint_id
  for checkpoint_id in converter.PROTEINMPNN_COMPATIBLE_IDS:
    assert checkpoint_id not in registered
  bad = {
    "model_family": "pottsmpnn",
    "checkpoint_id": "proteinmpnn_compatible_pottsmpnn_20",
    "artifact_path": "proteinmpnn_compatible_model_weights/pottsmpnn_20.pt",
    "sha256": "ab" * 32,
    "source_sha256": "cd" * 32,
    "upstream_commit": converter.UPSTREAM_COMMIT,
  }
  with pytest.raises(ValueError, match="proteinmpnn_compatible"):
    converter.validate_pottsmpnn_registry([bad])
  good = converter.registry_fragment(
    checkpoint_id="vanilla_20",
    artifact_path="vanilla_20.eqx",
    artifact_bytes=b"artifact",
    source_bytes=b"source",
  )
  for field in ("sha256", "source_sha256", "upstream_commit"):
    assert good[field]


def test_converter_raises_without_etab_out() -> None:
  torch = pytest.importorskip("torch")
  converter = _converter()
  state = {
    "W_out.weight": torch.zeros(21, 128),
    "W_out.bias": torch.zeros(21),
  }
  with pytest.raises(ValueError, match="etab_out"):
    converter.convert_state_dict(state)


def _set_linear(layer: DecoderLayer, path: str, weight: np.ndarray, bias: np.ndarray) -> DecoderLayer:
  parts = path.split(".")

  def _where(module: DecoderLayer):
    cursor = module
    for part in parts[:-1]:
      cursor = cursor[int(part)] if part.isdigit() else getattr(cursor, part)
    return getattr(cursor, parts[-1])

  # eqx.tree_at needs a lambda over the root. Rebuild via successive attribute walks.
  del _where
  updated = layer
  weight_array = jnp.asarray(weight)
  bias_array = jnp.asarray(bias)
  if path == "message_mlp.layers.0":
    updated = eqx.tree_at(lambda module: module.message_mlp.layers[0].weight, updated, weight_array)
    updated = eqx.tree_at(lambda module: module.message_mlp.layers[0].bias, updated, bias_array)
  elif path == "message_mlp.layers.1":
    updated = eqx.tree_at(lambda module: module.message_mlp.layers[1].weight, updated, weight_array)
    updated = eqx.tree_at(lambda module: module.message_mlp.layers[1].bias, updated, bias_array)
  elif path == "message_mlp.layers.2":
    updated = eqx.tree_at(lambda module: module.message_mlp.layers[2].weight, updated, weight_array)
    updated = eqx.tree_at(lambda module: module.message_mlp.layers[2].bias, updated, bias_array)
  elif path == "dense.layers.0":
    updated = eqx.tree_at(lambda module: module.dense.layers[0].weight, updated, weight_array)
    updated = eqx.tree_at(lambda module: module.dense.layers[0].bias, updated, bias_array)
  elif path == "dense.layers.1":
    updated = eqx.tree_at(lambda module: module.dense.layers[1].weight, updated, weight_array)
    updated = eqx.tree_at(lambda module: module.dense.layers[1].bias, updated, bias_array)
  elif path == "norm1":
    updated = eqx.tree_at(lambda module: module.norm1.weight, updated, weight_array)
    updated = eqx.tree_at(lambda module: module.norm1.bias, updated, bias_array)
  elif path == "norm2":
    updated = eqx.tree_at(lambda module: module.norm2.weight, updated, weight_array)
    updated = eqx.tree_at(lambda module: module.norm2.bias, updated, bias_array)
  else:
    msg = f"unmapped decoder weight path {path}"
    raise AssertionError(msg)
  return updated


def test_decoder_layer_f64_fixture() -> None:
  oracle_dir = _oracle_dir()
  payload_path = oracle_dir / "declayer_f64.npz"
  if not payload_path.is_file():
    pytest.skip(f"declayer_f64.npz absent under {oracle_dir}")
  digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
  expected = _expected_sha("declayer_f64.npz")
  if digest != expected:
    pytest.fail(f"declayer_f64.npz sha256 {digest} != {expected}")
  keys_text = (_REFERENCE / "keys.txt").read_text(encoding="utf-8")
  for name in ("h_V", "h_E", "mask_V", "output", "weight__W1__weight"):
    assert name in keys_text
  payload = np.load(payload_path)
  previous = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", True)
  try:
    layer = DecoderLayer(128, 384, 128, key=jax.random.key(0))
    mapping = {
      "message_mlp.layers.0": ("weight__W1__weight", "weight__W1__bias"),
      "message_mlp.layers.1": ("weight__W2__weight", "weight__W2__bias"),
      "message_mlp.layers.2": ("weight__W3__weight", "weight__W3__bias"),
      "dense.layers.0": ("weight__dense__W_in__weight", "weight__dense__W_in__bias"),
      "dense.layers.1": ("weight__dense__W_out__weight", "weight__dense__W_out__bias"),
      "norm1": ("weight__norm1__weight", "weight__norm1__bias"),
      "norm2": ("weight__norm2__weight", "weight__norm2__bias"),
    }
    for path, (weight_key, bias_key) in mapping.items():
      layer = _set_linear(layer, path, payload[weight_key], payload[bias_key])
    h_v = jnp.asarray(payload["h_V"][0])
    h_e = jnp.asarray(payload["h_E"][0])
    mask = jnp.asarray(payload["mask_V"][0])
    scale = float(np.asarray(payload["scale"]))
    predicted = layer(h_v, h_e, mask, scale=scale, inference=True)
    np.testing.assert_allclose(np.asarray(predicted), payload["output"][0], rtol=1e-12, atol=1e-12)
  finally:
    jax.config.update("jax_enable_x64", previous)
