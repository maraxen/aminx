"""PyTorch Lightning and PyTorch checkpoint converter for FrustraMPNN.

Loads and converts weights from:
- Upstream PyTorch Lightning checkpoint (e.g. ``megascale_default_la-true_esm-False_split1.ckpt``)
- Pretrained ProteinMPNN trunk (``v_48_020.pt``)
into a pure Equinox ``FrustraMPNN`` PyTree with strict zero-unmapped-leaf validation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.frustra_mpnn.model import (
  BothOutMLP,
  FrustraMPNN,
  LightAttention1D,
  cast_floating,
)
from aminx.io.torch_checkpoint import load_restricted_torch_checkpoint

if TYPE_CHECKING:
  from aminx.model.decoder import DecoderLayer
  from aminx.model.encoder import EncoderLayer
  from aminx.model.mpnn import Aminx

log = logging.getLogger(__name__)

# Expected 12 head parameter keys and shapes
FRUSTRA_HEAD_SHAPES: dict[str, tuple[int, ...]] = {
  "light_attention_mpnn.feature_convolution.weight": (512, 512, 9),
  "light_attention_mpnn.feature_convolution.bias": (512,),
  "light_attention_mpnn.attention_convolution.weight": (512, 512, 9),
  "light_attention_mpnn.attention_convolution.bias": (512,),
  "both_out.1.weight": (64, 512),
  "both_out.1.bias": (64,),
  "both_out.3.weight": (32, 64),
  "both_out.3.bias": (32,),
  "both_out.5.weight": (21, 32),
  "both_out.5.bias": (21,),
  "frustration_out.weight": (1, 1),
  "frustration_out.bias": (1,),
}


def _to_numpy(value: object) -> np.ndarray:
  """Convert tensor-like value to NumPy array."""
  if hasattr(value, "detach"):
    return np.asarray(value.detach().cpu().numpy())
  return np.asarray(value)


def _normalize_state_dict(raw_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
  """Unwrap state_dict / model_state_dict, strip model. prefix, and drop metric keys."""
  inner: Mapping[str, Any] = raw_state
  for wrapper_key in ("state_dict", "model_state_dict", "model"):
    candidate = raw_state.get(wrapper_key)
    if isinstance(candidate, Mapping):
      inner = candidate
      break

  normalized: dict[str, np.ndarray] = {}
  for raw_key, value in inner.items():
    if raw_key.startswith(("metrics.", "train_metrics.", "val_metrics.")):
      continue
    key = raw_key.removeprefix("model.")
    if key.startswith(("metrics.", "train_metrics.", "val_metrics.")):
      continue
    # Normalize light_attention alias if needed
    if key.startswith("light_attention."):
      key = "light_attention_mpnn." + key.removeprefix("light_attention.")
    normalized[key] = _to_numpy(value)
  return normalized


def _take(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  key: str,
) -> np.ndarray:
  """Retrieve required key and record it in consumed set."""
  if key not in state:
    msg = f"missing required checkpoint key {key!r}"
    raise KeyError(msg)
  consumed.add(key)
  return state[key]


def _check_shape(what: str, array: np.ndarray, expected_shape: tuple[int, ...]) -> None:
  """Assert exact tensor shape matching."""
  if tuple(array.shape) != expected_shape:
    msg = f"{what}: checkpoint shape {tuple(array.shape)} != expected {expected_shape}"
    raise ValueError(msg)


def _convert_linear(
  weight: np.ndarray,
  bias: np.ndarray | None,
  target: eqx.nn.Linear,
  *,
  name: str,
) -> eqx.nn.Linear:
  """Map weight and bias into eqx.nn.Linear."""
  if (bias is None) != (target.bias is None):
    msg = (
      f"{name}: bias presence mismatch "
      f"(checkpoint bias={bias is not None}, target={target.bias is not None})"
    )
    raise ValueError(msg)
  _check_shape(f"{name}.weight", weight, tuple(target.weight.shape))
  updated = eqx.tree_at(lambda layer: layer.weight, target, jnp.asarray(weight))
  if bias is not None and target.bias is not None:
    _check_shape(f"{name}.bias", bias, tuple(target.bias.shape))
    updated = eqx.tree_at(lambda layer: layer.bias, updated, jnp.asarray(bias))
  return updated


def _convert_layer_norm(
  weight: np.ndarray,
  bias: np.ndarray,
  target: eqx.nn.LayerNorm,
  *,
  name: str,
) -> eqx.nn.LayerNorm:
  """Map weight and bias into eqx.nn.LayerNorm."""
  if target.weight is None or target.bias is None:
    msg = f"{name}: target LayerNorm must have weight and bias"
    raise ValueError(msg)
  _check_shape(f"{name}.weight", weight, tuple(target.weight.shape))
  _check_shape(f"{name}.bias", bias, tuple(target.bias.shape))
  updated = eqx.tree_at(lambda norm: norm.weight, target, jnp.asarray(weight))
  return eqx.tree_at(lambda norm: norm.bias, updated, jnp.asarray(bias))


def _convert_mlp(
  wb_pairs: list[tuple[np.ndarray, np.ndarray]],
  target: eqx.nn.MLP,
  *,
  name: str,
) -> eqx.nn.MLP:
  """Map linear weight-bias pairs into eqx.nn.MLP."""
  new_layers = list(target.layers)
  pair_idx = 0
  for idx, layer in enumerate(new_layers):
    if isinstance(layer, eqx.nn.Linear):
      w, b = wb_pairs[pair_idx]
      new_layers[idx] = _convert_linear(w, b, layer, name=f"{name}.layers[{idx}]")
      pair_idx += 1
  if pair_idx != len(wb_pairs):
    msg = f"{name}: expected {len(wb_pairs)} linear layers, found {pair_idx}"
    raise ValueError(msg)
  return eqx.tree_at(lambda mlp: mlp.layers, target, tuple(new_layers))


def _convert_encoder_layer(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  prefix: str,
  target: EncoderLayer,
) -> EncoderLayer:
  """Map PyTorch encoder layer weights into Aminx EncoderLayer."""
  msg_pairs = [
    (_take(state, consumed, f"{prefix}W1.weight"), _take(state, consumed, f"{prefix}W1.bias")),
    (_take(state, consumed, f"{prefix}W2.weight"), _take(state, consumed, f"{prefix}W2.bias")),
    (_take(state, consumed, f"{prefix}W3.weight"), _take(state, consumed, f"{prefix}W3.bias")),
  ]
  upd_pairs = [
    (_take(state, consumed, f"{prefix}W11.weight"), _take(state, consumed, f"{prefix}W11.bias")),
    (_take(state, consumed, f"{prefix}W12.weight"), _take(state, consumed, f"{prefix}W12.bias")),
    (_take(state, consumed, f"{prefix}W13.weight"), _take(state, consumed, f"{prefix}W13.bias")),
  ]
  dense_pairs = [
    (
      _take(state, consumed, f"{prefix}dense.W_in.weight"),
      _take(state, consumed, f"{prefix}dense.W_in.bias"),
    ),
    (
      _take(state, consumed, f"{prefix}dense.W_out.weight"),
      _take(state, consumed, f"{prefix}dense.W_out.bias"),
    ),
  ]
  layer = eqx.tree_at(
    lambda mod: mod.edge_message_mlp,
    target,
    _convert_mlp(msg_pairs, target.edge_message_mlp, name=f"{prefix}edge_message_mlp"),
  )
  layer = eqx.tree_at(
    lambda mod: mod.edge_update_mlp,
    layer,
    _convert_mlp(upd_pairs, layer.edge_update_mlp, name=f"{prefix}edge_update_mlp"),
  )
  layer = eqx.tree_at(
    lambda mod: mod.dense,
    layer,
    _convert_mlp(dense_pairs, layer.dense, name=f"{prefix}dense"),
  )
  layer = eqx.tree_at(
    lambda mod: mod.norm1,
    layer,
    _convert_layer_norm(
      _take(state, consumed, f"{prefix}norm1.weight"),
      _take(state, consumed, f"{prefix}norm1.bias"),
      layer.norm1,
      name=f"{prefix}norm1",
    ),
  )
  layer = eqx.tree_at(
    lambda mod: mod.norm2,
    layer,
    _convert_layer_norm(
      _take(state, consumed, f"{prefix}norm2.weight"),
      _take(state, consumed, f"{prefix}norm2.bias"),
      layer.norm2,
      name=f"{prefix}norm2",
    ),
  )
  return eqx.tree_at(
    lambda mod: mod.norm3,
    layer,
    _convert_layer_norm(
      _take(state, consumed, f"{prefix}norm3.weight"),
      _take(state, consumed, f"{prefix}norm3.bias"),
      layer.norm3,
      name=f"{prefix}norm3",
    ),
  )


def _convert_decoder_layer(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  prefix: str,
  target: DecoderLayer,
) -> DecoderLayer:
  """Map PyTorch decoder layer weights into Aminx DecoderLayer."""
  msg_pairs = [
    (_take(state, consumed, f"{prefix}W1.weight"), _take(state, consumed, f"{prefix}W1.bias")),
    (_take(state, consumed, f"{prefix}W2.weight"), _take(state, consumed, f"{prefix}W2.bias")),
    (_take(state, consumed, f"{prefix}W3.weight"), _take(state, consumed, f"{prefix}W3.bias")),
  ]
  dense_pairs = [
    (
      _take(state, consumed, f"{prefix}dense.W_in.weight"),
      _take(state, consumed, f"{prefix}dense.W_in.bias"),
    ),
    (
      _take(state, consumed, f"{prefix}dense.W_out.weight"),
      _take(state, consumed, f"{prefix}dense.W_out.bias"),
    ),
  ]
  layer = eqx.tree_at(
    lambda mod: mod.message_mlp,
    target,
    _convert_mlp(msg_pairs, target.message_mlp, name=f"{prefix}message_mlp"),
  )
  layer = eqx.tree_at(
    lambda mod: mod.dense,
    layer,
    _convert_mlp(dense_pairs, layer.dense, name=f"{prefix}dense"),
  )
  layer = eqx.tree_at(
    lambda mod: mod.norm1,
    layer,
    _convert_layer_norm(
      _take(state, consumed, f"{prefix}norm1.weight"),
      _take(state, consumed, f"{prefix}norm1.bias"),
      layer.norm1,
      name=f"{prefix}norm1",
    ),
  )
  return eqx.tree_at(
    lambda mod: mod.norm2,
    layer,
    _convert_layer_norm(
      _take(state, consumed, f"{prefix}norm2.weight"),
      _take(state, consumed, f"{prefix}norm2.bias"),
      layer.norm2,
      name=f"{prefix}norm2",
    ),
  )


def _convert_mpnn_trunk(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  target: Aminx,
  *,
  prefix: str = "prot_mpnn.",
) -> Aminx:
  """Convert 118 ProteinMPNN trunk parameters from PyTorch state dict."""
  w_pos = _convert_linear(
    _take(state, consumed, f"{prefix}features.embeddings.linear.weight"),
    _take(state, consumed, f"{prefix}features.embeddings.linear.bias"),
    target.features.w_pos,
    name=f"{prefix}features.embeddings.linear",
  )
  w_e = _convert_linear(
    _take(state, consumed, f"{prefix}features.edge_embedding.weight"),
    None,
    target.features.w_e,
    name=f"{prefix}features.edge_embedding",
  )
  norm_edges = _convert_layer_norm(
    _take(state, consumed, f"{prefix}features.norm_edges.weight"),
    _take(state, consumed, f"{prefix}features.norm_edges.bias"),
    target.features.norm_edges,
    name=f"{prefix}features.norm_edges",
  )
  w_e_proj = _convert_linear(
    _take(state, consumed, f"{prefix}W_e.weight"),
    _take(state, consumed, f"{prefix}W_e.bias"),
    target.features.w_e_proj,
    name=f"{prefix}W_e",
  )
  features = eqx.tree_at(lambda feat: feat.w_pos, target.features, w_pos)
  features = eqx.tree_at(lambda feat: feat.w_e, features, w_e)
  features = eqx.tree_at(lambda feat: feat.norm_edges, features, norm_edges)
  features = eqx.tree_at(lambda feat: feat.w_e_proj, features, w_e_proj)
  mpnn = eqx.tree_at(lambda mod: mod.features, target, features)

  enc_layers = tuple(
    _convert_encoder_layer(state, consumed, f"{prefix}encoder_layers.{idx}.", layer)
    for idx, layer in enumerate(mpnn.encoder.layers)
  )
  mpnn = eqx.tree_at(lambda mod: mod.encoder.layers, mpnn, enc_layers)

  dec_layers = tuple(
    _convert_decoder_layer(state, consumed, f"{prefix}decoder_layers.{idx}.", layer)
    for idx, layer in enumerate(mpnn.decoder.layers)
  )
  mpnn = eqx.tree_at(lambda mod: mod.decoder.layers, mpnn, dec_layers)

  w_s_weight = _take(state, consumed, f"{prefix}W_s.weight")
  _check_shape(f"{prefix}W_s.weight", w_s_weight, tuple(mpnn.w_s_embed.weight.shape))
  w_s_embed = eqx.tree_at(lambda emb: emb.weight, mpnn.w_s_embed, jnp.asarray(w_s_weight))
  mpnn = eqx.tree_at(lambda mod: mod.w_s_embed, mpnn, w_s_embed)

  w_out = _convert_linear(
    _take(state, consumed, f"{prefix}W_out.weight"),
    _take(state, consumed, f"{prefix}W_out.bias"),
    mpnn.w_out,
    name=f"{prefix}W_out",
  )
  return eqx.tree_at(lambda mod: mod.w_out, mpnn, w_out)


def _convert_light_attention(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  target: LightAttention1D,
) -> LightAttention1D:
  """Map 4 LightAttention parameter tensors."""
  prefix = "light_attention_mpnn."
  fw = _take(state, consumed, f"{prefix}feature_convolution.weight")
  fb = _take(state, consumed, f"{prefix}feature_convolution.bias")
  aw = _take(state, consumed, f"{prefix}attention_convolution.weight")
  ab = _take(state, consumed, f"{prefix}attention_convolution.bias")

  _check_shape(f"{prefix}feature_convolution.weight", fw, tuple(target.feature_weight.shape))
  _check_shape(f"{prefix}feature_convolution.bias", fb, tuple(target.feature_bias.shape))
  _check_shape(f"{prefix}attention_convolution.weight", aw, tuple(target.attention_weight.shape))
  _check_shape(f"{prefix}attention_convolution.bias", ab, tuple(target.attention_bias.shape))

  la = eqx.tree_at(lambda mod: mod.feature_weight, target, jnp.asarray(fw))
  la = eqx.tree_at(lambda mod: mod.feature_bias, la, jnp.asarray(fb))
  la = eqx.tree_at(lambda mod: mod.attention_weight, la, jnp.asarray(aw))
  return eqx.tree_at(lambda mod: mod.attention_bias, la, jnp.asarray(ab))


def _convert_both_out(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  target: BothOutMLP,
) -> BothOutMLP:
  """Map 6 both_out MLP parameter tensors (Linear layers 1, 3, 5)."""
  w1 = _take(state, consumed, "both_out.1.weight")
  b1 = _take(state, consumed, "both_out.1.bias")
  w3 = _take(state, consumed, "both_out.3.weight")
  b3 = _take(state, consumed, "both_out.3.bias")
  w5 = _take(state, consumed, "both_out.5.weight")
  b5 = _take(state, consumed, "both_out.5.bias")

  l1 = _convert_linear(w1, b1, target.layers[0], name="both_out.1")
  l2 = _convert_linear(w3, b3, target.layers[1], name="both_out.3")
  l3 = _convert_linear(w5, b5, target.layers[2], name="both_out.5")

  return eqx.tree_at(lambda mod: mod.layers, target, (l1, l2, l3))


def _convert_frustration_out(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  target: eqx.nn.Linear,
) -> eqx.nn.Linear:
  """Map 2 frustration_out Linear(1, 1) parameter tensors."""
  w = _take(state, consumed, "frustration_out.weight")
  b = _take(state, consumed, "frustration_out.bias")
  return _convert_linear(w, b, target, name="frustration_out")


def convert_frustra_mpnn_state_dict(
  state_dict: Mapping[str, Any],
  *,
  mpnn_state_dict: Mapping[str, Any] | None = None,
  subtract_mut: bool = False,
  key: jax.Array | None = None,
) -> FrustraMPNN:
  """Convert FrustraMPNN state dict into pure Equinox PyTree with strict key validation.

  Parameters
  ----------
  state_dict : Mapping[str, Any]
    State dictionary containing 12 head parameters (and optionally embedded trunk).
  mpnn_state_dict : Mapping[str, Any] | None
    Fallback state dictionary containing ProteinMPNN trunk weights (e.g. from v_48_020.pt).
  subtract_mut : bool
    Whether to subtract wild-type predictions (default False).
  key : jax.Array | None
    Optional PRNG key for initial skeleton.

  Returns
  -------
  FrustraMPNN
    Equinox module populated with exact weights and zero unmapped leaves.
  """
  if key is None:
    key = jax.random.PRNGKey(0)

  norm_head = _normalize_state_dict(state_dict)
  consumed_head: set[str] = set()

  skeleton = cast_floating(
    FrustraMPNN(key=key, subtract_mut=subtract_mut),
    jnp.float32,
  )

  # 1. Convert LightAttention (4 tensors)
  la = _convert_light_attention(norm_head, consumed_head, skeleton.light_attention)
  model = eqx.tree_at(lambda mod: mod.light_attention, skeleton, la)

  # 2. Convert both_out MLP (6 tensors)
  both = _convert_both_out(norm_head, consumed_head, model.both_out)
  model = eqx.tree_at(lambda mod: mod.both_out, model, both)

  # 3. Convert frustration_out Linear(1, 1) (2 tensors)
  frust = _convert_frustration_out(norm_head, consumed_head, model.frustration_out)
  model = eqx.tree_at(lambda mod: mod.frustration_out, model, frust)

  # 4. Convert ProteinMPNN Trunk (118 tensors)
  has_embedded = any(k.startswith("prot_mpnn.") for k in norm_head)
  if has_embedded:
    mpnn = _convert_mpnn_trunk(norm_head, consumed_head, model.mpnn, prefix="prot_mpnn.")
  elif mpnn_state_dict is not None:
    norm_mpnn = _normalize_state_dict(mpnn_state_dict)
    consumed_mpnn: set[str] = set()
    mpnn_prefix = "prot_mpnn." if "prot_mpnn.W_s.weight" in norm_mpnn else ""
    mpnn = _convert_mpnn_trunk(norm_mpnn, consumed_mpnn, model.mpnn, prefix=mpnn_prefix)
    unconsumed_mpnn = set(norm_mpnn) - consumed_mpnn
    if unconsumed_mpnn:
      msg = f"unconsumed keys in mpnn_state_dict: {sorted(unconsumed_mpnn)}"
      raise ValueError(msg)
  else:
    msg = (
      "No ProteinMPNN trunk weights found in state_dict (no 'prot_mpnn.*' keys) "
      "and mpnn_state_dict was not provided."
    )
    raise KeyError(msg)

  model = eqx.tree_at(lambda mod: mod.mpnn, model, mpnn)

  # 5. Verify zero unconsumed keys in head state dict
  unconsumed_head = set(norm_head) - consumed_head
  if unconsumed_head:
    msg = f"unconsumed keys in FrustraMPNN state_dict: {sorted(unconsumed_head)}"
    raise ValueError(msg)

  return model


def load_frustra_mpnn_checkpoint(
  checkpoint_path: str | Path,
  *,
  trunk_checkpoint_path: str | Path | None = None,
  subtract_mut: bool = False,
  dtype: str = "float32",
) -> FrustraMPNN:
  """Load and convert FrustraMPNN checkpoint into Equinox module.

  Parameters
  ----------
  checkpoint_path : str | Path
    Path to PyTorch Lightning .ckpt file (e.g. megascale_default_la-true_esm-False_split1.ckpt).
  trunk_checkpoint_path : str | Path | None
    Optional path to ProteinMPNN v_48_020.pt weights file.
  subtract_mut : bool
    Whether to subtract wild-type score.
  dtype : str
    Precision to cast model parameters ("float32" or "float64").

  Returns
  -------
  FrustraMPNN
    Loaded Equinox model.
  """
  ckpt_path = Path(checkpoint_path)
  log.info("Loading FrustraMPNN head checkpoint from %s...", ckpt_path)
  head_ckpt = load_restricted_torch_checkpoint(ckpt_path)

  trunk_sd: dict[str, Any] | None = None
  if trunk_checkpoint_path is not None:
    trunk_path = Path(trunk_checkpoint_path)
    log.info("Loading ProteinMPNN trunk checkpoint from %s...", trunk_path)
    trunk_data = load_restricted_torch_checkpoint(trunk_path)
    trunk_sd = (
      trunk_data.get("model_state_dict", trunk_data) if isinstance(trunk_data, dict) else trunk_data
    )

  model = convert_frustra_mpnn_state_dict(
    head_ckpt,
    mpnn_state_dict=trunk_sd,
    subtract_mut=subtract_mut,
  )

  target_dtype = jnp.float64 if dtype == "float64" else jnp.float32
  return cast_floating(model, target_dtype)
