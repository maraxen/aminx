"""Weight converter for Caliby Potts and sidechain diffusion checkpoints.

Converts upstream PyTorch Lightning ``.ckpt`` state dicts across all 8 released
Caliby checkpoints:
- ``caliby``, ``soluble_caliby``, ``caliby_distill``, ``soluble_caliby_distill`` (``model_version=0``, ``edge_in=8481``)
- ``soluble_caliby_v1`` (``model_version=1``, ``edge_in=1057``)
- ``caliby_packer_000``, ``caliby_packer_010``, ``caliby_packer_030`` (sidechain diffusion packer)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.caliby.model import CalibyModel
from aminx.families.caliby.sidechain_diffusion import CalibySidechainMLP

if TYPE_CHECKING:
  from aminx.families.caliby.features import CalibyPositionalEncodings, CalibyTokenFeatures
  from aminx.families.caliby.model import (
    CalibyDecLayer,
    CalibyEncLayer,
    CalibyMPNN,
    CalibyPositionWiseFFN,
  )
  from aminx.families.caliby.potts import CalibyGraphPotts
  from aminx.families.caliby.sidechain_diffusion import (
    DenoisingMLPBlock,
    FinalLayer,
    TimestepEmbedder,
  )

CALIBY_POTTS_CHECKPOINTS: frozenset[str] = frozenset({
  "caliby",
  "soluble_caliby",
  "soluble_caliby_v1",
  "caliby_distill",
  "soluble_caliby_distill",
})

CALIBY_PACKER_CHECKPOINTS: frozenset[str] = frozenset({
  "caliby_packer_000",
  "caliby_packer_010",
  "caliby_packer_030",
})

CALIBY_ALL_CHECKPOINTS: frozenset[str] = CALIBY_POTTS_CHECKPOINTS | CALIBY_PACKER_CHECKPOINTS


def normalize_caliby_state_dict(
  raw_state_dict: Mapping[str, Any],
) -> dict[str, np.ndarray]:
  """Normalize upstream PyTorch Lightning state dict keys to canonical Caliby paths.

  Strips leading ``model.`` and ``_orig_mod.`` prefixes (from ``torch.compile``),
  normalizes ``denoiser.atom_mpnn.`` to ``mpnn.``, ``token_features.`` to
  ``features.``, ``decoder_S_potts.`` to ``potts_decoder.``, and
  ``scn_diffusion_module.scn_denoiser.`` to the root denoiser namespace.

  Parameters
  ----------
  raw_state_dict : Mapping[str, Any]
    Upstream PyTorch state dictionary (or checkpoint dict containing ``"state_dict"``).

  Returns
  -------
  dict[str, np.ndarray]
    Normalized mapping from canonical parameter key to NumPy float32 array.
  """
  sd = raw_state_dict.get("state_dict", raw_state_dict)
  if not isinstance(sd, Mapping):
    msg = "Expected a mapping for Caliby state_dict"
    raise TypeError(msg)

  out: dict[str, np.ndarray] = {}
  for raw_key, val in sd.items():
    k = str(raw_key)
    while k.startswith(("model.", "_orig_mod.")):
      k = k.removeprefix("model.")
      k = k.removeprefix("_orig_mod.")
    k = k.replace("._orig_mod.", ".")
    if k.startswith("denoiser.atom_mpnn."):
      k = "denoiser.mpnn." + k[len("denoiser.atom_mpnn.") :]
    k = k.removeprefix("denoiser.")
    if k.startswith("mpnn.token_features."):
      k = "mpnn.features." + k[len("mpnn.token_features.") :]
    if k.startswith("mpnn.decoder_S_potts."):
      k = "potts_decoder." + k[len("mpnn.decoder_S_potts.") :]
    if k.startswith("scn_diffusion_module.scn_denoiser.timestep_embedder."):
      k = "t_embedder." + k[len("scn_diffusion_module.scn_denoiser.timestep_embedder.") :]
    elif k.startswith("scn_diffusion_module.scn_denoiser.h_V_embedder."):
      k = "c_embedder." + k[len("scn_diffusion_module.scn_denoiser.h_V_embedder.") :]
    elif k.startswith("scn_diffusion_module.scn_denoiser."):
      k = k[len("scn_diffusion_module.scn_denoiser.") :]
    elif k == "scn_diffusion_module.scn_interpolant.sigma_data":
      k = "sigma_data"
    arr = val.detach().cpu().numpy() if hasattr(val, "detach") else np.asarray(val)
    out[k] = np.asarray(arr, dtype=np.float32)
  return out


def infer_caliby_checkpoint_kind(
  sd: Mapping[str, np.ndarray],
) -> tuple[str, int]:
  """Infer ``(kind, model_version)`` from a normalized Caliby state dict.

  Parameters
  ----------
  sd : Mapping[str, np.ndarray]
    Normalized state dictionary from :func:`normalize_caliby_state_dict`.

  Returns
  -------
  tuple[str, int]
    ``("packer" | "potts", 0 | 1)`` based on denoiser keys and ``edge_embedding.weight`` shape.
  """
  edge_w = sd.get("mpnn.features.edge_embedding.weight")
  model_version = 1 if (edge_w is not None and edge_w.shape[1] == 1057) else 0
  if "x_embedder.weight" in sd or "final_layer.linear.weight" in sd:
    return "packer", model_version
  return "potts", model_version


def _load_linear(
  linear: eqx.nn.Linear,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> eqx.nn.Linear:
  w = jnp.asarray(sd[f"{prefix}.weight"], dtype=jnp.float32)
  if w.shape != linear.weight.shape:
    msg = f"Shape mismatch at {prefix}.weight: {w.shape} vs {linear.weight.shape}"
    raise ValueError(msg)
  linear = eqx.tree_at(lambda m: m.weight, linear, w)
  if linear.bias is not None:
    if f"{prefix}.bias" in sd:
      b = jnp.asarray(sd[f"{prefix}.bias"], dtype=jnp.float32)
    else:
      b = jnp.zeros_like(linear.bias)
    linear = eqx.tree_at(lambda m: m.bias, linear, b)
  return linear


def _load_layernorm(
  norm: eqx.nn.LayerNorm,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> eqx.nn.LayerNorm:
  if norm.weight is not None and f"{prefix}.weight" in sd:
    w = jnp.asarray(sd[f"{prefix}.weight"], dtype=jnp.float32)
    norm = eqx.tree_at(lambda m: m.weight, norm, w)
  if norm.bias is not None and f"{prefix}.bias" in sd:
    b = jnp.asarray(sd[f"{prefix}.bias"], dtype=jnp.float32)
    norm = eqx.tree_at(lambda m: m.bias, norm, b)
  return norm


def _load_ffn(
  ffn: CalibyPositionWiseFFN,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> CalibyPositionWiseFFN:
  w_in = _load_linear(ffn.W_in, sd, f"{prefix}.W_in")
  w_out = _load_linear(ffn.W_out, sd, f"{prefix}.W_out")
  ffn = eqx.tree_at(lambda m: m.W_in, ffn, w_in)
  return eqx.tree_at(lambda m: m.W_out, ffn, w_out)


def _load_enc_layer(
  layer: CalibyEncLayer,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> CalibyEncLayer:
  for name in ("W1", "W2", "W3", "W11", "W12", "W13"):
    lin = _load_linear(getattr(layer, name), sd, f"{prefix}.{name}")
    layer = eqx.tree_at(lambda m, n=name: getattr(m, n), layer, lin)
  for n_name in ("norm1", "norm2", "norm3"):
    ln = _load_layernorm(getattr(layer, n_name), sd, f"{prefix}.{n_name}")
    layer = eqx.tree_at(lambda m, n=n_name: getattr(m, n), layer, ln)
  dense = _load_ffn(layer.dense, sd, f"{prefix}.dense")
  return eqx.tree_at(lambda m: m.dense, layer, dense)


def _load_dec_layer(
  layer: CalibyDecLayer,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> CalibyDecLayer:
  for name in ("W1", "W2", "W3"):
    lin = _load_linear(getattr(layer, name), sd, f"{prefix}.{name}")
    layer = eqx.tree_at(lambda m, n=name: getattr(m, n), layer, lin)
  for n_name in ("norm1", "norm2"):
    ln = _load_layernorm(getattr(layer, n_name), sd, f"{prefix}.{n_name}")
    layer = eqx.tree_at(lambda m, n=n_name: getattr(m, n), layer, ln)
  if layer.update_edges:
    for edge_w in ("W11", "W12", "W13"):
      cur_lin = getattr(layer, edge_w)
      if cur_lin is not None and f"{prefix}.{edge_w}.weight" in sd:
        lin = _load_linear(cur_lin, sd, f"{prefix}.{edge_w}")
        layer = eqx.tree_at(lambda m, n=edge_w: getattr(m, n), layer, lin)
    if layer.norm3 is not None and f"{prefix}.norm3.weight" in sd:
      ln3 = _load_layernorm(layer.norm3, sd, f"{prefix}.norm3")
      layer = eqx.tree_at(lambda m: m.norm3, layer, ln3)
  dense = _load_ffn(layer.dense, sd, f"{prefix}.dense")
  return eqx.tree_at(lambda m: m.dense, layer, dense)


def _load_token_features(
  feat: CalibyTokenFeatures,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> CalibyTokenFeatures:
  pos_lin = _load_linear(feat.embeddings.linear, sd, f"{prefix}.embeddings.linear")
  emb: CalibyPositionalEncodings = eqx.tree_at(lambda m: m.linear, feat.embeddings, pos_lin)
  edge_emb = _load_linear(feat.edge_embedding, sd, f"{prefix}.edge_embedding")
  norm_e = _load_layernorm(feat.norm_edges, sd, f"{prefix}.norm_edges")
  feat = eqx.tree_at(lambda m: m.embeddings, feat, emb)
  feat = eqx.tree_at(lambda m: m.edge_embedding, feat, edge_emb)
  feat = eqx.tree_at(lambda m: m.norm_edges, feat, norm_e)
  if f"{prefix}.node_embedding.weight" in sd:
    node_emb = _load_linear(feat.node_embedding, sd, f"{prefix}.node_embedding")
    norm_n = _load_layernorm(feat.norm_nodes, sd, f"{prefix}.norm_nodes")
  else:
    # Upstream Caliby initializes h_V to zeros before adding W_s(restype)
    zero_w = jnp.zeros_like(feat.node_embedding.weight)
    zero_b = (
      jnp.zeros_like(feat.node_embedding.bias)
      if feat.node_embedding.bias is not None
      else None
    )
    node_emb = eqx.tree_at(lambda m: m.weight, feat.node_embedding, zero_w)
    if zero_b is not None:
      node_emb = eqx.tree_at(lambda m: m.bias, node_emb, zero_b)
    norm_n = feat.norm_nodes
    if norm_n.bias is not None:
      norm_n = eqx.tree_at(lambda m: m.bias, norm_n, jnp.zeros_like(norm_n.bias))
  feat = eqx.tree_at(lambda m: m.node_embedding, feat, node_emb)
  return eqx.tree_at(lambda m: m.norm_nodes, feat, norm_n)


def _load_mpnn(
  mpnn: CalibyMPNN,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> CalibyMPNN:
  feat = _load_token_features(mpnn.features, sd, f"{prefix}.features")
  w_e = _load_linear(mpnn.W_e, sd, f"{prefix}.W_e")
  mpnn = eqx.tree_at(lambda m: m.features, mpnn, feat)
  mpnn = eqx.tree_at(lambda m: m.W_e, mpnn, w_e)
  if mpnn.W_s is not None and f"{prefix}.W_s.weight" in sd:
    ws_w = jnp.asarray(sd[f"{prefix}.W_s.weight"], dtype=jnp.float32)
    if ws_w.shape != mpnn.W_s.weight.shape and ws_w.T.shape == mpnn.W_s.weight.shape:
      ws_w = ws_w.T
    w_s = eqx.tree_at(lambda m: m.weight, mpnn.W_s, ws_w)
    mpnn = eqx.tree_at(lambda m: m.W_s, mpnn, w_s)
  if mpnn.W_out is not None and f"{prefix}.W_out.weight" in sd:
    w_out = _load_linear(mpnn.W_out, sd, f"{prefix}.W_out")
    mpnn = eqx.tree_at(lambda m: m.W_out, mpnn, w_out)
  enc_layers = tuple(
    _load_enc_layer(layer, sd, f"{prefix}.encoder_layers.{i}")
    for i, layer in enumerate(mpnn.encoder_layers)
  )
  dec_layers = tuple(
    _load_dec_layer(layer, sd, f"{prefix}.decoder_layers.{i}")
    for i, layer in enumerate(mpnn.decoder_layers)
  )
  mpnn = eqx.tree_at(lambda m: m.encoder_layers, mpnn, enc_layers)
  return eqx.tree_at(lambda m: m.decoder_layers, mpnn, dec_layers)


def convert_caliby_potts_state_dict(
  raw_state_dict: Mapping[str, Any],
  *,
  key: jax.Array | None = None,
) -> CalibyModel:
  """Convert a PyTorch Caliby Potts state dict into a :class:`CalibyModel`.

  Supports both the synthetic test state dict layout (``parameterization="factorized"``)
  and the upstream checkpoint layout (``parameterization="factor"`` with
  ``W_J_left``, ``W_J_right``, ``W_h``, ``log_scale``, and edge-updating decoder layers).

  Parameters
  ----------
  raw_state_dict : Mapping[str, Any]
    Upstream PyTorch state dict or checkpoint dictionary.
  key : jax.Array | None, optional
    Optional PRNG key for skeleton initialization.

  Returns
  -------
  CalibyModel
    Populated Equinox ``CalibyModel`` instance.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  sd = normalize_caliby_state_dict(raw_state_dict)
  kind, model_version = infer_caliby_checkpoint_kind(sd)
  if kind != "potts":
    msg = f"Expected a Caliby Potts state dict, got kind={kind!r}"
    raise ValueError(msg)

  is_factor = "potts_decoder.W_J_left.weight" in sd
  num_enc = sum(1 for k in sd if k.startswith("mpnn.encoder_layers.") and k.endswith(".W1.weight"))
  num_dec = sum(1 for k in sd if k.startswith("mpnn.decoder_layers.") and k.endswith(".W1.weight"))
  has_w_out = "mpnn.W_out.weight" in sd
  has_bg = "potts_decoder.W_h_bg.weight" in sd or "potts_decoder.W_J_bg.weight" in sd
  if "potts_decoder.W_h_bg.weight" in sd:
    bg_dim = int(sd["potts_decoder.W_h_bg.weight"].shape[0])
  elif "potts_decoder.W_J_bg.weight" in sd:
    bg_dim = int(sd["potts_decoder.W_J_bg.weight"].shape[0])
  else:
    bg_dim = 1
  model = CalibyModel(
    num_encoder_layers=max(num_enc, 3),
    num_decoder_layers=max(num_dec, 3),
    model_version=model_version,
    parameterization="factor" if is_factor else "factorized",
    use_sequence=is_factor or ("mpnn.W_s.weight" in sd),
    update_decoder_edges=is_factor or ("mpnn.decoder_layers.0.W11.weight" in sd),
    has_w_out=has_w_out,
    has_bg=has_bg,
    bg_dim=bg_dim,
    key=key,
  )
  mpnn = _load_mpnn(model.mpnn, sd, "mpnn")

  potts: CalibyGraphPotts = model.potts_decoder
  if is_factor:
    if potts.log_scale is not None and "potts_decoder.log_scale" in sd:
      ls = jnp.asarray(sd["potts_decoder.log_scale"], dtype=jnp.float32)
      potts = eqx.tree_at(lambda m: m.log_scale, potts, ls)
    for fn_name in ("W_h", "W_J_left", "W_J_right", "W_h_bg", "W_J_bg"):
      cur_l = getattr(potts, fn_name)
      if cur_l is not None and f"potts_decoder.{fn_name}.weight" in sd:
        loaded_l = _load_linear(cur_l, sd, f"potts_decoder.{fn_name}")
        potts = eqx.tree_at(lambda m, n=fn_name: getattr(m, n), potts, loaded_l)
  else:
    for lin_name in ("node_Wait_i", "node_Wait_j", "edge_Wait", "node_to_h"):
      lin = _load_linear(getattr(potts, lin_name), sd, f"potts_decoder.{lin_name}")
      potts = eqx.tree_at(lambda m, n=lin_name: getattr(m, n), potts, lin)
    for norm_name in ("norm_J_i", "norm_J_j", "norm_h"):
      ln = _load_layernorm(getattr(potts, norm_name), sd, f"potts_decoder.{norm_name}")
      potts = eqx.tree_at(lambda m, n=norm_name: getattr(m, n), potts, ln)

  model = eqx.tree_at(lambda m: m.mpnn, model, mpnn)
  return eqx.tree_at(lambda m: m.potts_decoder, model, potts)


def convert_caliby_packer_state_dict(
  raw_state_dict: Mapping[str, Any],
  *,
  key: jax.Array | None = None,
) -> CalibySidechainMLP:
  """Convert a PyTorch Caliby sidechain packer state dict into a :class:`CalibySidechainMLP`.

  Parameters
  ----------
  raw_state_dict : Mapping[str, Any]
    Upstream PyTorch state dict or checkpoint dictionary.
  key : jax.Array | None, optional
    Optional PRNG key for skeleton initialization.

  Returns
  -------
  CalibySidechainMLP
    Populated Equinox ``CalibySidechainMLP`` instance.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  sd = normalize_caliby_state_dict(raw_state_dict)
  kind, model_version = infer_caliby_checkpoint_kind(sd)
  if kind != "packer":
    msg = f"Expected a Caliby packer state dict, got kind={kind!r}"
    raise ValueError(msg)

  num_enc = sum(1 for k in sd if k.startswith("mpnn.encoder_layers.") and k.endswith(".W1.weight"))
  num_dec = sum(1 for k in sd if k.startswith("mpnn.decoder_layers.") and k.endswith(".W1.weight"))
  depth = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".mlp.fc1.weight"))
  in_ch = int(sd["x_embedder.weight"].shape[1]) if "x_embedder.weight" in sd else 254
  out_ch = int(sd["final_layer.linear.weight"].shape[0]) if "final_layer.linear.weight" in sd else 54
  sigma_data = float(sd["sigma_data"].item()) if "sigma_data" in sd else 1.0

  packer = CalibySidechainMLP(
    in_channels=in_ch,
    out_channels=out_ch,
    depth=max(depth, 1),
    num_encoder_layers=max(num_enc, 3),
    num_decoder_layers=max(num_dec, 3),
    model_version=model_version,
    update_decoder_edges="mpnn.decoder_layers.0.W11.weight" in sd,
    has_w_out="mpnn.W_out.weight" in sd,
    sigma_data=sigma_data,
    key=key,
  )
  mpnn = _load_mpnn(packer.mpnn, sd, "mpnn")
  x_emb = _load_linear(packer.x_embedder, sd, "x_embedder")
  c_emb = _load_linear(packer.c_embedder, sd, "c_embedder")

  t_mlp0 = _load_linear(packer.t_embedder.mlp_0, sd, "t_embedder.mlp.0")
  t_mlp2 = _load_linear(packer.t_embedder.mlp_2, sd, "t_embedder.mlp.2")
  t_emb: TimestepEmbedder = eqx.tree_at(lambda m: m.mlp_0, packer.t_embedder, t_mlp0)
  t_emb = eqx.tree_at(lambda m: m.mlp_2, t_emb, t_mlp2)

  blocks: list[DenoisingMLPBlock] = []
  for i, blk in enumerate(packer.blocks):
    fc1 = _load_linear(blk.mlp_fc1, sd, f"blocks.{i}.mlp.fc1")
    fc2 = _load_linear(blk.mlp_fc2, sd, f"blocks.{i}.mlp.fc2")
    adaln = _load_linear(blk.adaLN_modulation, sd, f"blocks.{i}.adaLN_modulation.1")
    b_new = eqx.tree_at(lambda m: m.mlp_fc1, blk, fc1)
    b_new = eqx.tree_at(lambda m: m.mlp_fc2, b_new, fc2)
    b_new = eqx.tree_at(lambda m: m.adaLN_modulation, b_new, adaln)
    blocks.append(b_new)

  fl_lin = _load_linear(packer.final_layer.linear, sd, "final_layer.linear")
  fl_adaln = _load_linear(packer.final_layer.adaLN_modulation, sd, "final_layer.adaLN_modulation.1")
  final_layer: FinalLayer = eqx.tree_at(lambda m: m.linear, packer.final_layer, fl_lin)
  final_layer = eqx.tree_at(lambda m: m.adaLN_modulation, final_layer, fl_adaln)

  packer = eqx.tree_at(lambda m: m.mpnn, packer, mpnn)
  packer = eqx.tree_at(lambda m: m.x_embedder, packer, x_emb)
  packer = eqx.tree_at(lambda m: m.c_embedder, packer, c_emb)
  packer = eqx.tree_at(lambda m: m.t_embedder, packer, t_emb)
  packer = eqx.tree_at(lambda m: m.blocks, packer, tuple(blocks))
  return eqx.tree_at(lambda m: m.final_layer, packer, final_layer)


def convert_caliby_checkpoint(
  raw_state_dict: Mapping[str, Any],
  output_path: str | Path,
  *,
  checkpoint_id: str = "caliby",
  model_version: int | None = None,
  key: jax.Array | None = None,
) -> tuple[eqx.Module, dict[str, Any]]:
  """Convert a PyTorch Caliby checkpoint, serialize ``.eqx``, and build its manifest.

  Parameters
  ----------
  raw_state_dict : Mapping[str, Any]
    Upstream PyTorch state dict or checkpoint dictionary.
  output_path : str | Path
    Destination file path for the serialized ``.eqx`` weights.
  checkpoint_id : str, default="caliby"
    Canonical checkpoint identifier.
  model_version : int | None, optional
    Explicit feature layout version override.
  key : jax.Array | None, optional
    Optional PRNG key for skeleton initialization.

  Returns
  -------
  tuple[eqx.Module, dict[str, Any]]
    Tuple ``(model, manifest_dict)``.
  """
  import hashlib  # noqa: PLC0415

  sd = normalize_caliby_state_dict(raw_state_dict)
  kind, inferred_ver = infer_caliby_checkpoint_kind(sd)
  eff_ver = inferred_ver if model_version is None else model_version
  if kind == "packer":
    model: eqx.Module = convert_caliby_packer_state_dict(sd, key=key)
    packer_m = model
    arch_meta: dict[str, Any] = {
      "in_channels": int(packer_m.x_embedder.weight.shape[1]),  # type: ignore[attr-defined]
      "out_channels": int(packer_m.final_layer.linear.weight.shape[0]),  # type: ignore[attr-defined]
      "depth": len(packer_m.blocks),  # type: ignore[attr-defined]
      "num_encoder_layers": len(packer_m.mpnn.encoder_layers),  # type: ignore[attr-defined]
      "num_decoder_layers": len(packer_m.mpnn.decoder_layers),  # type: ignore[attr-defined]
      "update_decoder_edges": packer_m.mpnn.decoder_layers[0].update_edges,  # type: ignore[attr-defined]
      "has_w_out": packer_m.mpnn.W_out is not None,  # type: ignore[attr-defined]
      "sigma_data": float(packer_m.sigma_data),  # type: ignore[attr-defined]
    }
  else:
    model = convert_caliby_potts_state_dict(sd, key=key)
    potts_m = model
    arch_meta = {
      "num_encoder_layers": len(potts_m.mpnn.encoder_layers),  # type: ignore[attr-defined]
      "num_decoder_layers": len(potts_m.mpnn.decoder_layers),  # type: ignore[attr-defined]
      "parameterization": potts_m.potts_decoder.parameterization,  # type: ignore[attr-defined]
      "use_sequence": potts_m.mpnn.W_s is not None,  # type: ignore[attr-defined]
      "update_decoder_edges": potts_m.mpnn.decoder_layers[0].update_edges,  # type: ignore[attr-defined]
      "has_w_out": potts_m.mpnn.W_out is not None,  # type: ignore[attr-defined]
      "has_bg": bool(potts_m.potts_decoder.has_bg),  # type: ignore[attr-defined]
      "bg_dim": int(potts_m.potts_decoder.bg_dim),  # type: ignore[attr-defined]
    }
  out_p = Path(output_path)
  out_p.parent.mkdir(parents=True, exist_ok=True)
  eqx.tree_serialise_leaves(out_p, model)
  output_sha256 = hashlib.sha256(out_p.read_bytes()).hexdigest()
  manifest: dict[str, Any] = {
    "checkpoint_id": checkpoint_id,
    "family": "caliby",
    "kind": kind,
    "model_version": eff_ver,
    "num_tensors": len(sd),
    "output_sha256": output_sha256,
    **arch_meta,
  }
  return model, manifest

