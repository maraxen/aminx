"""Weight conversion for ThermoMPNN (v1/v2 single-mutant) and ThermoMPNN-D (v2 double-mutant).

Supports:
- ``thermoMPNN_default.pt`` (ThermoMPNN v1: ``both_out.{1,3,5}``, ``ddg_out``, ``kernel_size=9``)
- ``ThermoMPNN-ens{1,2,3}.ckpt`` (ThermoMPNN v2 single: ``ddg_out.{1,3,5}``, ``kernel_size=1``)
- ``ThermoMPNN-D-ens{1,2,3}.ckpt`` (ThermoMPNN-D v2 double Siamese: ``light_attention.{0,1}``,
  ``ddg_out.{0,3,6}``)
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.thermo_mpnn.model import (
  LightAttention1D,
  ThermoMPNN,
  ThermoMPNNDouble,
  cast_floating,
)
from aminx.io.weights import load_weights, resolve_family_checkpoint

if TYPE_CHECKING:
  from aminx.model.decoder import DecoderLayer
  from aminx.model.encoder import EncoderLayer
  from aminx.model.mpnn import Aminx

log = logging.getLogger(__name__)

_ALLOWED_UNPICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
  {
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "str"),
    ("builtins", "bytes"),
    ("builtins", "bool"),
    ("builtins", "complex"),
    ("builtins", "slice"),
    ("builtins", "object"),
    ("builtins", "getattr"),
    ("__builtin__", "dict"),
    ("__builtin__", "list"),
    ("__builtin__", "tuple"),
    ("__builtin__", "set"),
    ("__builtin__", "object"),
    ("copy_reg", "_reconstructor"),
    ("copyreg", "_reconstructor"),
    ("typing", "Any"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_device_tensor_from_numpy"),
    ("torch.storage", "_load_from_bytes"),
    ("torch", "FloatStorage"),
    ("torch", "DoubleStorage"),
    ("torch", "HalfStorage"),
    ("torch", "BFloat16Storage"),
    ("torch", "ByteStorage"),
    ("torch", "CharStorage"),
    ("torch", "ShortStorage"),
    ("torch", "IntStorage"),
    ("torch", "LongStorage"),
    ("torch", "BoolStorage"),
    ("torch", "Size"),
    ("torch", "device"),
    ("torch", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy", "float32"),
    ("numpy", "float64"),
    ("numpy", "int64"),
    ("numpy", "int32"),
  },
)

_PLACEHOLDER_UNPICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
  {
    ("argparse", "Namespace"),
  },
)

_PLACEHOLDER_UNPICKLE_MODULE_PREFIXES: tuple[str, ...] = (
  "pytorch_lightning.",
  "lightning_fabric.",
  "omegaconf.",
)


class _UnpicklePlaceholder:
  """Fallback placeholder for non-tensor objects in PyTorch Lightning ``.ckpt`` files."""

  def __init__(self, *args: object, **kwargs: object) -> None:
    """Ignore arbitrary constructor arguments.

    Parameters
    ----------
    args : object
      Ignored positional arguments.
    kwargs : object
      Ignored keyword arguments.

    Returns
    -------
    None
      Initializes placeholder instance.
    """

  def __setstate__(self, state: object) -> None:
    """Ignore arbitrary pickle state.

    Parameters
    ----------
    state : object
      Ignored pickled state payload.

    Returns
    -------
    None
      Restores placeholder instance.
    """


class _SafeUnpickler(pickle.Unpickler):
  """Allowlist-enforcing unpickler for ThermoMPNN PyTorch / PyTorch Lightning checkpoints."""

  def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
    """Resolve an allowlisted ``(module, name)``, stub Lightning metadata, or refuse.

    Parameters
    ----------
    module : str
      Module path requested by pickle.
    name : str
      Class or callable name requested by pickle.

    Returns
    -------
    Any
      Resolved allowlisted class or :class:`_UnpicklePlaceholder`.

    Raises
    ------
    pickle.UnpicklingError
      If ``(module, name)`` is not on the explicit checkpoint allowlist.
    """
    if (module, name) in _ALLOWED_UNPICKLE_GLOBALS:
      return super().find_class(module, name)
    if (module, name) in _PLACEHOLDER_UNPICKLE_GLOBALS or (
      module in ("pytorch_lightning", "lightning_fabric", "omegaconf")
      or module.startswith(_PLACEHOLDER_UNPICKLE_MODULE_PREFIXES)
    ):
      return _UnpicklePlaceholder
    msg = (
      f"Refusing to unpickle disallowed global '{module}.{name}'; "
      "only allowlisted torch/numpy/collections/builtins/pytorch_lightning checkpoint globals are permitted."
    )
    raise pickle.UnpicklingError(msg)


class _SafePickleModule:
  """Minimal ``pickle_module`` shim for ``torch.load``."""

  Unpickler = _SafeUnpickler
  load = staticmethod(pickle.load)
  loads = staticmethod(pickle.loads)
  dump = staticmethod(pickle.dump)
  dumps = staticmethod(pickle.dumps)
  HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _to_numpy(value: object) -> np.ndarray:
  if hasattr(value, "detach"):
    return np.asarray(value.detach().cpu().numpy())  # type: ignore[union-attr]
  return np.asarray(value)


def _normalize_state_dict(raw_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
  """Unwrap ``state_dict`` / ``model_state_dict``, strip ``model.`` prefixes, and drop ``metrics.*``."""
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
    normalized[key] = _to_numpy(value)
  return normalized


def _take(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  key: str,
) -> np.ndarray:
  if key not in state:
    msg = f"missing required checkpoint key {key!r}"
    raise KeyError(msg)
  consumed.add(key)
  return state[key]


def _check_shape(what: str, array: np.ndarray, expected_shape: tuple[int, ...]) -> None:
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


def _convert_light_attention_1d(
  state: Mapping[str, np.ndarray],
  consumed: set[str],
  target: LightAttention1D,
) -> LightAttention1D:
  fw = _take(state, consumed, "light_attention.feature_convolution.weight")
  fb = _take(state, consumed, "light_attention.feature_convolution.bias")
  aw = _take(state, consumed, "light_attention.attention_convolution.weight")
  ab = _take(state, consumed, "light_attention.attention_convolution.bias")
  _check_shape(
    "light_attention.feature_convolution.weight",
    fw,
    tuple(target.feature_weight.shape),
  )
  _check_shape(
    "light_attention.feature_convolution.bias",
    fb,
    tuple(target.feature_bias.shape),
  )
  _check_shape(
    "light_attention.attention_convolution.weight",
    aw,
    tuple(target.attention_weight.shape),
  )
  _check_shape(
    "light_attention.attention_convolution.bias",
    ab,
    tuple(target.attention_bias.shape),
  )
  la = eqx.tree_at(lambda mod: mod.feature_weight, target, jnp.asarray(fw))
  la = eqx.tree_at(lambda mod: mod.feature_bias, la, jnp.asarray(fb))
  la = eqx.tree_at(lambda mod: mod.attention_weight, la, jnp.asarray(aw))
  return eqx.tree_at(lambda mod: mod.attention_bias, la, jnp.asarray(ab))


def _sequential_linear_indices(state: Mapping[str, np.ndarray], prefix: str) -> list[int]:
  """Find sorted integer layer indices ``<prefix>.<idx>.{weight,bias}`` in ``state``."""
  indices: set[int] = set()
  head = f"{prefix}."
  for key in state:
    for suffix in (".weight", ".bias"):
      if key.startswith(head) and key.endswith(suffix):
        middle = key[len(head) : -len(suffix)]
        if middle.isdigit():
          indices.add(int(middle))
  return sorted(indices)


def _has_embedded_mpnn_trunk(norm_state: Mapping[str, Any]) -> bool:
  """Return True when normalized ``norm_state`` contains ``prot_mpnn.*`` trunk keys."""
  return any(key.startswith("prot_mpnn.") for key in norm_state)


def _resolve_and_convert_mpnn_trunk(
  norm_state: Mapping[str, np.ndarray],
  consumed: set[str],
  target_mpnn: Aminx,
  mpnn_state_dict: Mapping[str, Any] | None,
  *,
  override_embedded_mpnn: bool,
  owner_name: str,
) -> Aminx:
  """Convert the ProteinMPNN trunk from ``norm_state`` or ``mpnn_state_dict``."""
  has_embedded = _has_embedded_mpnn_trunk(norm_state)
  if mpnn_state_dict is None:
    return _convert_mpnn_trunk(norm_state, consumed, target_mpnn, prefix="prot_mpnn.")

  if has_embedded and not override_embedded_mpnn:
    msg = (
      f"{owner_name} state_dict already embeds a 'prot_mpnn.*' trunk, and a separate "
      "mpnn_state_dict was also supplied. Omit mpnn_state_dict to use the embedded trunk, "
      "or pass override_embedded_mpnn=True to explicitly replace it."
    )
    raise ValueError(msg)

  mpnn_norm = _normalize_state_dict(mpnn_state_dict)
  mpnn_consumed: set[str] = set()
  mpnn_prefix = "prot_mpnn." if "prot_mpnn.W_s.weight" in mpnn_norm else ""
  mpnn = _convert_mpnn_trunk(mpnn_norm, mpnn_consumed, target_mpnn, prefix=mpnn_prefix)
  mpnn_unconsumed = set(mpnn_norm) - mpnn_consumed
  if mpnn_unconsumed:
    msg = f"unconsumed keys in fallback mpnn_state_dict: {sorted(mpnn_unconsumed)}"
    raise ValueError(msg)
  if has_embedded and override_embedded_mpnn:
    for k in norm_state:
      if k.startswith("prot_mpnn."):
        consumed.add(k)
  return mpnn


def convert_thermo_mpnn_state_dict(
  state_dict: Mapping[str, Any],
  *,
  mpnn_state_dict: Mapping[str, Any] | None = None,
  override_embedded_mpnn: bool = False,
  apply_chain_mask_to_order: bool | None = None,
  subtract_mut: bool = True,
  key: jax.Array | None = None,
) -> ThermoMPNN:
  """Convert a ThermoMPNN (v1 or v2 single-mutant) state dict into :class:`ThermoMPNN`.

  Verifies every tensor shape and ensures no unconsumed keys remain.

  Parameters
  ----------
  state_dict : Mapping[str, Any]
    Raw or normalized ThermoMPNN state dictionary.
  mpnn_state_dict : Mapping[str, Any] | None
    Optional fallback ProteinMPNN trunk state dictionary when not embedded in ``state_dict``.
  override_embedded_mpnn : bool
    When ``True``, allow ``mpnn_state_dict`` to replace an embedded ``prot_mpnn.*`` trunk.
    Defaults to ``False`` (raises :class:`ValueError` if both are present).
  apply_chain_mask_to_order : bool | None
    Optional override for ``apply_chain_mask_to_order`` (defaults to ``False`` for v1, ``True`` for v2).
  subtract_mut : bool
    Whether to subtract wild-type self-mutation predictions.
  key : jax.Array | None
    Optional PRNG key for skeleton initialization.

  Returns
  -------
  ThermoMPNN
    Converted :class:`ThermoMPNN` module with verified weights.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  norm_state = _normalize_state_dict(state_dict)
  consumed: set[str] = set()

  if "light_attention.feature_convolution.weight" not in norm_state:
    msg = (
      "state_dict is missing 'light_attention.feature_convolution.weight'; "
      "for ThermoMPNN-D double-mutant checkpoints use convert_thermo_mpnn_double_state_dict"
    )
    raise ValueError(msg)

  feat_conv_w = norm_state["light_attention.feature_convolution.weight"]
  if feat_conv_w.ndim != 3:
    msg = f"light_attention.feature_convolution.weight must be 3D, got {feat_conv_w.shape}"
    raise ValueError(msg)
  kernel_size = int(feat_conv_w.shape[2])

  is_v1 = "both_out.1.weight" in norm_state
  mlp_prefix = "both_out" if is_v1 else "ddg_out"
  has_ddg_out_scale = is_v1 and ("ddg_out.weight" in norm_state)
  resolved_chain_mask_order = (
    (not is_v1) if apply_chain_mask_to_order is None else apply_chain_mask_to_order
  )

  mlp_indices = _sequential_linear_indices(norm_state, mlp_prefix)
  if len(mlp_indices) < 2:
    msg = f"expected at least 2 linear layers under {mlp_prefix}.*, found indices {mlp_indices}"
    raise ValueError(msg)
  hidden_dims = tuple(
    int(norm_state[f"{mlp_prefix}.{idx}.weight"].shape[0]) for idx in mlp_indices[:-1]
  )

  model = cast_floating(
    ThermoMPNN(
      key=key,
      hidden_dims=hidden_dims,
      num_final_layers=2,
      subtract_mut=subtract_mut,
      light_attention_kernel_size=kernel_size,
      has_ddg_out_scale=has_ddg_out_scale,
      apply_chain_mask_to_order=resolved_chain_mask_order,
    ),
    jnp.float32,
  )

  mpnn = _resolve_and_convert_mpnn_trunk(
    norm_state,
    consumed,
    model.mpnn,
    mpnn_state_dict,
    override_embedded_mpnn=override_embedded_mpnn,
    owner_name="ThermoMPNN",
  )
  model = eqx.tree_at(lambda mod: mod.mpnn, model, mpnn)

  la = _convert_light_attention_1d(norm_state, consumed, model.light_attention)
  model = eqx.tree_at(lambda mod: mod.light_attention, model, la)

  new_mlp_layers = tuple(
    _convert_linear(
      _take(norm_state, consumed, f"{mlp_prefix}.{layer_idx}.weight"),
      _take(norm_state, consumed, f"{mlp_prefix}.{layer_idx}.bias"),
      target_layer,
      name=f"{mlp_prefix}.{layer_idx}",
    )
    for layer_idx, target_layer in zip(mlp_indices, model.mlp_layers, strict=True)
  )
  model = eqx.tree_at(lambda mod: mod.mlp_layers, model, new_mlp_layers)

  if has_ddg_out_scale and model.ddg_out is not None:
    ddg_out = _convert_linear(
      _take(norm_state, consumed, "ddg_out.weight"),
      _take(norm_state, consumed, "ddg_out.bias"),
      model.ddg_out,
      name="ddg_out",
    )
    model = eqx.tree_at(lambda mod: mod.ddg_out, model, ddg_out)

  unconsumed = set(norm_state) - consumed
  if unconsumed:
    msg = f"unconsumed keys in ThermoMPNN state_dict: {sorted(unconsumed)}"
    raise ValueError(msg)

  return model


def convert_thermo_mpnn_double_state_dict(
  state_dict: Mapping[str, Any],
  *,
  mpnn_state_dict: Mapping[str, Any] | None = None,
  override_embedded_mpnn: bool = False,
  key: jax.Array | None = None,
) -> ThermoMPNNDouble:
  """Convert a ThermoMPNN-D Siamese state dict into :class:`ThermoMPNNDouble`.

  Verifies every tensor shape and ensures no unconsumed keys remain.

  Parameters
  ----------
  state_dict : Mapping[str, Any]
    Raw or normalized ThermoMPNN-D state dictionary.
  mpnn_state_dict : Mapping[str, Any] | None
    Optional fallback ProteinMPNN trunk state dictionary when not embedded in ``state_dict``.
  override_embedded_mpnn : bool
    When ``True``, allow ``mpnn_state_dict`` to replace an embedded ``prot_mpnn.*`` trunk.
    Defaults to ``False`` (raises :class:`ValueError` if both are present).
  key : jax.Array | None
    Optional PRNG key for skeleton initialization.

  Returns
  -------
  ThermoMPNNDouble
    Converted :class:`ThermoMPNNDouble` module with verified weights.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  norm_state = _normalize_state_dict(state_dict)
  consumed: set[str] = set()

  if "light_attention.0.weight" not in norm_state or "light_attention.1.weight" not in norm_state:
    msg = (
      "state_dict is missing 'light_attention.0.weight' / 'light_attention.1.weight' "
      "expected for ThermoMPNN-D (TransferModelv2Siamese)"
    )
    raise ValueError(msg)

  mlp_indices = _sequential_linear_indices(norm_state, "ddg_out")
  if len(mlp_indices) < 2:
    msg = f"expected at least 2 linear layers under ddg_out.*, found indices {mlp_indices}"
    raise ValueError(msg)
  hidden_dims = tuple(
    int(norm_state[f"ddg_out.{idx}.weight"].shape[0]) for idx in mlp_indices[:-1]
  )

  model = cast_floating(
    ThermoMPNNDouble(
      key=key,
      hidden_dims=hidden_dims,
      num_final_layers=2,
    ),
    jnp.float32,
  )

  mpnn = _resolve_and_convert_mpnn_trunk(
    norm_state,
    consumed,
    model.mpnn,
    mpnn_state_dict,
    override_embedded_mpnn=override_embedded_mpnn,
    owner_name="ThermoMPNNDouble",
  )
  model = eqx.tree_at(lambda mod: mod.mpnn, model, mpnn)

  proj_norm = _convert_layer_norm(
    _take(norm_state, consumed, "light_attention.0.weight"),
    _take(norm_state, consumed, "light_attention.0.bias"),
    model.proj_norm,
    name="light_attention.0",
  )
  proj_linear = _convert_linear(
    _take(norm_state, consumed, "light_attention.1.weight"),
    _take(norm_state, consumed, "light_attention.1.bias"),
    model.proj_linear,
    name="light_attention.1",
  )
  model = eqx.tree_at(lambda mod: mod.proj_norm, model, proj_norm)
  model = eqx.tree_at(lambda mod: mod.proj_linear, model, proj_linear)

  new_mlp_layers = tuple(
    _convert_linear(
      _take(norm_state, consumed, f"ddg_out.{layer_idx}.weight"),
      _take(norm_state, consumed, f"ddg_out.{layer_idx}.bias"),
      target_layer,
      name=f"ddg_out.{layer_idx}",
    )
    for layer_idx, target_layer in zip(mlp_indices, model.mlp_layers, strict=True)
  )
  model = eqx.tree_at(lambda mod: mod.mlp_layers, model, new_mlp_layers)

  unconsumed = set(norm_state) - consumed
  if unconsumed:
    msg = f"unconsumed keys in ThermoMPNNDouble state_dict: {sorted(unconsumed)}"
    raise ValueError(msg)

  return model


def _load_raw_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
  ckpt_path = Path(path)
  if ckpt_path.suffix.lower() == ".npz":
    with np.load(ckpt_path) as data:
      raw_npz = {k: np.asarray(data[k]) for k in data.files}
    return _normalize_state_dict(raw_npz)

  import torch  # noqa: PLC0415

  blob = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False,
    pickle_module=_SafePickleModule,
  )
  if not isinstance(blob, Mapping):
    msg = f"expected Mapping in checkpoint {ckpt_path}, got {type(blob)}"
    raise TypeError(msg)
  return _normalize_state_dict(blob)


def _resolve_fallback_mpnn_state(
  ckpt_path: Path,
  state: Mapping[str, Any],
  mpnn_path: str | Path | None,
  *,
  override_embedded_mpnn: bool = False,
) -> dict[str, np.ndarray] | None:
  norm_state = _normalize_state_dict(state)
  has_embedded = _has_embedded_mpnn_trunk(norm_state)
  if has_embedded:
    if mpnn_path is not None and not override_embedded_mpnn:
      msg = (
        f"Checkpoint {ckpt_path} already embeds a 'prot_mpnn.*' trunk, and mpnn_path={mpnn_path!r} "
        "was also supplied. Omit mpnn_path to use the embedded trunk, or pass "
        "override_embedded_mpnn=True to explicitly replace it."
      )
      raise ValueError(msg)
    if mpnn_path is not None and override_embedded_mpnn:
      return _load_raw_checkpoint(mpnn_path)
    return None
  if mpnn_path is not None:
    return _load_raw_checkpoint(mpnn_path)
  candidate = ckpt_path.resolve().parent.parent / "vanilla_model_weights" / "v_48_020.pt"
  if candidate.is_file():
    log.info(
      "Using directory-layout fallback ProteinMPNN trunk weights at %s for checkpoint %s",
      candidate,
      ckpt_path,
    )
    return _load_raw_checkpoint(candidate)
  return None


def load_thermo_mpnn_pt(
  path: str | Path,
  *,
  mpnn_path: str | Path | None = None,
  override_embedded_mpnn: bool = False,
  apply_chain_mask_to_order: bool | None = None,
  subtract_mut: bool = True,
  key: jax.Array | None = None,
) -> ThermoMPNN:
  """Load a ThermoMPNN single-mutant model from a ``.pt``, ``.ckpt``, or ``.npz`` file.

  Parameters
  ----------
  path : str | Path
    Path to the ThermoMPNN checkpoint file.
  mpnn_path : str | Path | None
    Optional path to vanilla ``v_48_020.pt`` when not embedded in ``path``.
  override_embedded_mpnn : bool
    When ``True``, allow ``mpnn_path`` to replace an embedded ``prot_mpnn.*`` trunk.
  apply_chain_mask_to_order : bool | None
    Optional override for ``apply_chain_mask_to_order``.
  subtract_mut : bool
    Whether to subtract wild-type self-mutation predictions.
  key : jax.Array | None
    Optional PRNG key for skeleton initialization.

  Returns
  -------
  ThermoMPNN
    Loaded :class:`ThermoMPNN` model.
  """
  ckpt_path = Path(path)
  state = _load_raw_checkpoint(ckpt_path)
  mpnn_state = _resolve_fallback_mpnn_state(
    ckpt_path,
    state,
    mpnn_path,
    override_embedded_mpnn=override_embedded_mpnn,
  )
  return convert_thermo_mpnn_state_dict(
    state,
    mpnn_state_dict=mpnn_state,
    override_embedded_mpnn=override_embedded_mpnn,
    apply_chain_mask_to_order=apply_chain_mask_to_order,
    subtract_mut=subtract_mut,
    key=key,
  )


def load_thermo_mpnn_double_pt(
  path: str | Path,
  *,
  mpnn_path: str | Path | None = None,
  override_embedded_mpnn: bool = False,
  key: jax.Array | None = None,
) -> ThermoMPNNDouble:
  """Load a ThermoMPNN-D double-mutant model from a ``.pt``, ``.ckpt``, or ``.npz`` file.

  Parameters
  ----------
  path : str | Path
    Path to the ThermoMPNN-D checkpoint file.
  mpnn_path : str | Path | None
    Optional path to vanilla ``v_48_020.pt`` when not embedded in ``path``.
  override_embedded_mpnn : bool
    When ``True``, allow ``mpnn_path`` to replace an embedded ``prot_mpnn.*`` trunk.
  key : jax.Array | None
    Optional PRNG key for skeleton initialization.

  Returns
  -------
  ThermoMPNNDouble
    Loaded :class:`ThermoMPNNDouble` model.
  """
  ckpt_path = Path(path)
  state = _load_raw_checkpoint(ckpt_path)
  mpnn_state = _resolve_fallback_mpnn_state(
    ckpt_path,
    state,
    mpnn_path,
    override_embedded_mpnn=override_embedded_mpnn,
  )
  return convert_thermo_mpnn_double_state_dict(
    state,
    mpnn_state_dict=mpnn_state,
    override_embedded_mpnn=override_embedded_mpnn,
    key=key,
  )


def load_thermo_mpnn_checkpoint(
  checkpoint_id: str = "thermo_mpnn_v1_default",
  *,
  local_path: str | Path | None = None,
  has_ddg_out_scale: bool | None = None,
  light_attention_kernel_size: int | None = None,
  apply_chain_mask_to_order: bool | None = None,
  subtract_mut: bool = True,
  key: jax.Array | None = None,
) -> ThermoMPNN:
  """Load a serialized ``.eqx.zst`` / ``.eqx`` ThermoMPNN checkpoint from Hub or local path.

  When ``local_path`` is ``None``, resolves ``checkpoint_id`` through
  :func:`aminx.io.weights.resolve_family_checkpoint` (enforcing ``family_weights.json`` sha256
  pinning).

  Parameters
  ----------
  checkpoint_id : str
    Logical checkpoint ID (e.g. ``"thermo_mpnn_v1_default"``, ``"thermo_mpnn_v2_ens1"``).
  local_path : str | Path | None
    Optional explicit file path overriding Hub / ``AMINX_WEIGHTS_DIR`` resolution.
  has_ddg_out_scale : bool | None
    Optional override for the v1 ``ddg_out`` linear scale layer.
  light_attention_kernel_size : int | None
    Optional override for the LightAttention 1D kernel size (9 for v1, 1 for v2).
  apply_chain_mask_to_order : bool | None
    Optional override for ``apply_chain_mask_to_order``.
  subtract_mut : bool
    Whether to subtract wild-type self-mutation predictions.
  key : jax.Array | None
    Optional PRNG key for skeleton construction.

  Returns
  -------
  ThermoMPNN
    Deserialized :class:`ThermoMPNN` model.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  name = checkpoint_id.lower()
  is_v1 = "v1" in name or "default" in name
  ddg_out_scale = is_v1 if has_ddg_out_scale is None else has_ddg_out_scale
  kernel_size = (
    light_attention_kernel_size
    if light_attention_kernel_size is not None
    else (9 if is_v1 else 1)
  )
  chain_mask_order = (
    apply_chain_mask_to_order
    if apply_chain_mask_to_order is not None
    else (not is_v1)
  )
  skeleton = cast_floating(
    ThermoMPNN(
      light_attention_kernel_size=kernel_size,
      has_ddg_out_scale=ddg_out_scale,
      apply_chain_mask_to_order=chain_mask_order,
      subtract_mut=subtract_mut,
      key=key,
    ),
    jnp.float32,
  )
  if local_path is None:
    bare_id = checkpoint_id.removesuffix(".zst").removesuffix(".eqx")
    provenance = resolve_family_checkpoint(bare_id, family="thermompnn")
    resolved_path = provenance.path
    loaded = load_weights(
      checkpoint_id=None,
      local_path=resolved_path,
      skeleton=skeleton,
    )
  else:
    loaded = load_weights(
      checkpoint_id=checkpoint_id,
      local_path=str(local_path),
      skeleton=skeleton,
    )
  return cast("ThermoMPNN", loaded)


def load_thermo_mpnn_double_checkpoint(
  checkpoint_id: str = "thermo_mpnn_d_ens1",
  *,
  local_path: str | Path | None = None,
  key: jax.Array | None = None,
) -> ThermoMPNNDouble:
  """Load a serialized ``.eqx.zst`` / ``.eqx`` ThermoMPNN-D checkpoint from Hub or local path.

  When ``local_path`` is ``None``, resolves ``checkpoint_id`` through
  :func:`aminx.io.weights.resolve_family_checkpoint` (enforcing ``family_weights.json`` sha256
  pinning).

  Parameters
  ----------
  checkpoint_id : str
    Logical checkpoint ID (e.g. ``"thermo_mpnn_d_ens1"``, ``"thermo_mpnn_d_ens2"``).
  local_path : str | Path | None
    Optional explicit file path overriding Hub / ``AMINX_WEIGHTS_DIR`` resolution.
  key : jax.Array | None
    Optional PRNG key for skeleton construction.

  Returns
  -------
  ThermoMPNNDouble
    Deserialized :class:`ThermoMPNNDouble` model.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  skeleton = cast_floating(ThermoMPNNDouble(key=key), jnp.float32)
  if local_path is None:
    bare_id = checkpoint_id.removesuffix(".zst").removesuffix(".eqx")
    provenance = resolve_family_checkpoint(bare_id, family="thermompnn")
    resolved_path = provenance.path
    loaded = load_weights(
      checkpoint_id=None,
      local_path=resolved_path,
      skeleton=skeleton,
    )
  else:
    loaded = load_weights(
      checkpoint_id=checkpoint_id,
      local_path=str(local_path),
      skeleton=skeleton,
    )
  return cast("ThermoMPNNDouble", loaded)
