"""Weight converter for FAMPNN PyTorch checkpoints (``fampnn_0_0``, ``fampnn_0_3``, ``fampnn_0_3_cath``).

Converts upstream PyTorch state dicts (with or without ``scn_confidence`` / ``scn_Wait``
keys) into Equinox :class:`FAMPNNModel` instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.fampnn.model import FAMPNNModel

if TYPE_CHECKING:
  from aminx.families.fampnn.features import (
    FAMPNNPositionalEncodings,
    FAMPNNProteinFeatures,
    SidechainProteinFeatures,
  )
  from aminx.families.fampnn.gvp import (
    GVP,
    GVPConvLayer,
    GVPEncoder,
    GVPGraphEmbedding,
    GVPLayerNorm,
  )
  from aminx.families.fampnn.model import (
    FAEncoder,
    FAMPNNDecLayer,
    FAMPNNEdgeUpdatingDecLayer,
    FAMPNNEncLayer,
    FAMPNNPositionWiseFFN,
  )
  from aminx.families.fampnn.sidechain_diffusion import (
    FAMPNNDenoisingMLPBlock,
    FAMPNNFinalLayer,
    FAMPNNTimestepEmbedder,
    SidechainConfidenceModule,
    SidechainDiffusionModule,
  )

FAMPNN_ALL_CHECKPOINTS: frozenset[str] = frozenset({
  "fampnn_0_0",
  "fampnn_0_3",
  "fampnn_0_3_cath",
})


def normalize_fampnn_state_dict(
  raw_state_dict: Mapping[str, Any],
) -> dict[str, np.ndarray]:
  """Normalize upstream FAMPNN PyTorch state dict keys.

  Strips leading ``model.`` and ``_orig_mod.`` prefixes and normalizes
  ``denoiser.seq_design_module.`` to ``fa_encoder.``, ``vector_encoder.`` to
  ``gvp_encoder.``, ``.wv.`` to ``.wu.``, ``.wsv.`` to ``.wg.``, and
  ``denoiser.scn_diffusion_module.`` to ``scn_diffusion.`` / ``scn_confidence.``.

  Parameters
  ----------
  raw_state_dict : Mapping[str, Any]
    Upstream PyTorch state dict or checkpoint dictionary.

  Returns
  -------
  dict[str, np.ndarray]
    Normalized mapping from parameter key to float32 NumPy array.
  """
  sd = raw_state_dict.get("state_dict", raw_state_dict)
  if not isinstance(sd, Mapping):
    msg = "Expected a mapping for FAMPNN state_dict"
    raise TypeError(msg)

  out: dict[str, np.ndarray] = {}
  for raw_key, val in sd.items():
    k = str(raw_key)
    while k.startswith(("model.", "_orig_mod.")):
      k = k.removeprefix("model.")
      k = k.removeprefix("_orig_mod.")
    k = k.replace("._orig_mod.", ".")
    if k.startswith("denoiser."):
      k = k.removeprefix("denoiser.")
    if k.startswith("seq_design_module.W_out."):
      k = "W_out." + k[len("seq_design_module.W_out.") :]
    elif k.startswith("seq_design_module."):
      k = "fa_encoder." + k[len("seq_design_module.") :]
    k = k.replace("fa_encoder.vector_encoder.", "fa_encoder.gvp_encoder.")
    k = k.replace(".embed_graph.embed_node.0.", ".embed_graph.W_v.1.")
    k = k.replace(".embed_graph.embed_node.1.", ".embed_graph.W_v.2.")
    k = k.replace(".embed_graph.embed_edge.0.", ".embed_graph.W_e.1.")
    k = k.replace(".embed_graph.embed_edge.1.", ".embed_graph.W_e.2.")
    k = k.replace(".wv.", ".wu.")
    k = k.replace(".wsv.", ".wg.")
    if k.startswith("scn_diffusion_module.scn_denoiser.timestep_embedder."):
      k = (
        "scn_diffusion.denoiser.t_embedder."
        + k[len("scn_diffusion_module.scn_denoiser.timestep_embedder.") :]
      )
    elif k.startswith("scn_diffusion_module.scn_denoiser.h_V_embedder."):
      k = (
        "scn_diffusion.denoiser.c_embedder."
        + k[len("scn_diffusion_module.scn_denoiser.h_V_embedder.") :]
      )
    elif k.startswith("scn_diffusion_module.scn_denoiser."):
      k = (
        "scn_diffusion.denoiser."
        + k[len("scn_diffusion_module.scn_denoiser.") :]
      )
    elif k.startswith("scn_diffusion_module.confidence_module.structure_encoder.decoder_layers."):
      k = (
        "scn_confidence.decoder_layers."
        + k[len("scn_diffusion_module.confidence_module.structure_encoder.decoder_layers.") :]
      )
    elif k.startswith("scn_diffusion_module.confidence_module."):
      k = (
        "scn_confidence."
        + k[len("scn_diffusion_module.confidence_module.") :]
      )
    if k.startswith("scn_Wait."):
      k = "scn_confidence." + k[len("scn_Wait.") :]
    arr = val.detach().cpu().numpy() if hasattr(val, "detach") else np.asarray(val)
    out[k] = np.asarray(arr, dtype=np.float32)
  return out


def has_fampnn_confidence_head(sd: Mapping[str, np.ndarray]) -> bool:
  """Return ``True`` if ``sd`` contains ``scn_confidence`` parameters.

  Parameters
  ----------
  sd : Mapping[str, np.ndarray]
    Normalized FAMPNN state dictionary.

  Returns
  -------
  bool
    Whether ``scn_confidence`` keys are present.
  """
  return any(k.startswith("scn_confidence.") for k in sd)


def _load_linear(
  linear: eqx.nn.Linear,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> eqx.nn.Linear:
  if f"{prefix}.weight" in sd:
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


def _load_gvp_norm(
  gnorm: GVPLayerNorm,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> GVPLayerNorm:
  sn = _load_layernorm(gnorm.scalar_norm, sd, f"{prefix}.scalar_norm")
  return eqx.tree_at(lambda m: m.scalar_norm, gnorm, sn)


def _load_gvp(
  gvp: GVP,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> GVP:
  if gvp.wh is not None and f"{prefix}.wh.weight" in sd:
    wh = _load_linear(gvp.wh, sd, f"{prefix}.wh")
    gvp = eqx.tree_at(lambda m: m.wh, gvp, wh)
  ws = _load_linear(gvp.ws, sd, f"{prefix}.ws")
  gvp = eqx.tree_at(lambda m: m.ws, gvp, ws)
  if gvp.wu is not None and f"{prefix}.wu.weight" in sd:
    wu = _load_linear(gvp.wu, sd, f"{prefix}.wu")
    gvp = eqx.tree_at(lambda m: m.wu, gvp, wu)
  if gvp.wg is not None and f"{prefix}.wg.weight" in sd:
    wg = _load_linear(gvp.wg, sd, f"{prefix}.wg")
    gvp = eqx.tree_at(lambda m: m.wg, gvp, wg)
  return gvp


def _load_gvp_encoder(
  enc: GVPEncoder,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> GVPEncoder:
  eg: GVPGraphEmbedding = enc.embed_graph
  # Upstream embed_node/embed_edge are Sequential(GVP, LayerNorm), renamed to
  # W_v.1 / W_v.2 and W_e.1 / W_e.2. There is no W_*.0 pre-norm in the checkpoint.
  eg = eqx.tree_at(
    lambda m: m.W_v_gvp, eg, _load_gvp(eg.W_v_gvp, sd, f"{prefix}.embed_graph.W_v.1"),
  )
  eg = eqx.tree_at(
    lambda m: m.W_v_norm1, eg, _load_gvp_norm(eg.W_v_norm1, sd, f"{prefix}.embed_graph.W_v.2"),
  )
  eg = eqx.tree_at(
    lambda m: m.W_e_gvp, eg, _load_gvp(eg.W_e_gvp, sd, f"{prefix}.embed_graph.W_e.1"),
  )
  eg = eqx.tree_at(
    lambda m: m.W_e_norm1, eg, _load_gvp_norm(eg.W_e_norm1, sd, f"{prefix}.embed_graph.W_e.2"),
  )
  if eg.embed_mpnn_node is not None and f"{prefix}.embed_graph.embed_mpnn_node.weight" in sd:
    mn = _load_linear(eg.embed_mpnn_node, sd, f"{prefix}.embed_graph.embed_mpnn_node")
    eg = eqx.tree_at(lambda m: m.embed_mpnn_node, eg, mn)
  if eg.embed_mpnn_edge is not None and f"{prefix}.embed_graph.embed_mpnn_edge.weight" in sd:
    me = _load_linear(eg.embed_mpnn_edge, sd, f"{prefix}.embed_graph.embed_mpnn_edge")
    eg = eqx.tree_at(lambda m: m.embed_mpnn_edge, eg, me)

  new_layers: list[GVPConvLayer] = []
  for i, layer in enumerate(enc.encoder_layers):
    lp = f"{prefix}.encoder_layers.{i}"
    l_new = eqx.tree_at(
      lambda m: m.conv_gvp0, layer, _load_gvp(layer.conv_gvp0, sd, f"{lp}.conv.message_func.0"),
    )
    l_new = eqx.tree_at(
      lambda m: m.conv_gvp1, l_new, _load_gvp(l_new.conv_gvp1, sd, f"{lp}.conv.message_func.1"),
    )
    l_new = eqx.tree_at(
      lambda m: m.conv_gvp2, l_new, _load_gvp(l_new.conv_gvp2, sd, f"{lp}.conv.message_func.2"),
    )
    l_new = eqx.tree_at(
      lambda m: m.norm0, l_new, _load_gvp_norm(l_new.norm0, sd, f"{lp}.norm.0"),
    )
    l_new = eqx.tree_at(
      lambda m: m.ff_gvp0, l_new, _load_gvp(l_new.ff_gvp0, sd, f"{lp}.ff_func.0"),
    )
    l_new = eqx.tree_at(
      lambda m: m.ff_gvp1, l_new, _load_gvp(l_new.ff_gvp1, sd, f"{lp}.ff_func.1"),
    )
    l_new = eqx.tree_at(
      lambda m: m.norm1, l_new, _load_gvp_norm(l_new.norm1, sd, f"{lp}.norm.1"),
    )
    new_layers.append(l_new)

  norm_out = _load_layernorm(enc.norm_gvp_out, sd, f"{prefix}.norm_gvp_out")
  enc = eqx.tree_at(lambda m: m.embed_graph, enc, eg)
  enc = eqx.tree_at(lambda m: m.encoder_layers, enc, tuple(new_layers))
  enc = eqx.tree_at(lambda m: m.norm_gvp_out, enc, norm_out)
  if enc.w_out_norm is not None and enc.w_out_gvp is not None:
    wn = _load_gvp_norm(enc.w_out_norm, sd, f"{prefix}.W_out.0")
    wg = _load_gvp(enc.w_out_gvp, sd, f"{prefix}.W_out.1")
    enc = eqx.tree_at(lambda m: m.w_out_norm, enc, wn)
    enc = eqx.tree_at(lambda m: m.w_out_gvp, enc, wg)
  elif enc.w_out_linear is not None and f"{prefix}.W_out.weight" in sd:
    wl = _load_linear(enc.w_out_linear, sd, f"{prefix}.W_out")
    enc = eqx.tree_at(lambda m: m.w_out_linear, enc, wl)
  return enc


def _load_ffn(
  ffn: FAMPNNPositionWiseFFN,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> FAMPNNPositionWiseFFN:
  w_in = _load_linear(ffn.W_in, sd, f"{prefix}.W_in")
  w_out = _load_linear(ffn.W_out, sd, f"{prefix}.W_out")
  ffn = eqx.tree_at(lambda m: m.W_in, ffn, w_in)
  return eqx.tree_at(lambda m: m.W_out, ffn, w_out)


def _load_enc_or_edge_dec(
  layer: FAMPNNEncLayer | FAMPNNEdgeUpdatingDecLayer,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> Any:  # noqa: ANN401
  for name in ("W1", "W2", "W3", "W11", "W12", "W13"):
    lin = _load_linear(getattr(layer, name), sd, f"{prefix}.{name}")
    layer = eqx.tree_at(lambda m, n=name: getattr(m, n), layer, lin)
  for n_name in ("norm1", "norm2", "norm3"):
    ln = _load_layernorm(getattr(layer, n_name), sd, f"{prefix}.{n_name}")
    layer = eqx.tree_at(lambda m, n=n_name: getattr(m, n), layer, ln)
  dense = _load_ffn(layer.dense, sd, f"{prefix}.dense")
  return eqx.tree_at(lambda m: m.dense, layer, dense)


def _load_dec_layer(
  layer: FAMPNNDecLayer,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> FAMPNNDecLayer:
  for name in ("W1", "W2", "W3"):
    lin = _load_linear(getattr(layer, name), sd, f"{prefix}.{name}")
    layer = eqx.tree_at(lambda m, n=name: getattr(m, n), layer, lin)
  for n_name in ("norm1", "norm2"):
    ln = _load_layernorm(getattr(layer, n_name), sd, f"{prefix}.{n_name}")
    layer = eqx.tree_at(lambda m, n=n_name: getattr(m, n), layer, ln)
  dense = _load_ffn(layer.dense, sd, f"{prefix}.dense")
  return eqx.tree_at(lambda m: m.dense, layer, dense)


def _load_features(
  feat: FAMPNNProteinFeatures | SidechainProteinFeatures,
  sd: Mapping[str, np.ndarray],
  prefix: str,
) -> Any:  # noqa: ANN401
  pos_lin = _load_linear(feat.embeddings.linear, sd, f"{prefix}.embeddings.linear")
  emb: FAMPNNPositionalEncodings = eqx.tree_at(lambda m: m.linear, feat.embeddings, pos_lin)
  edge_emb = _load_linear(feat.edge_embedding, sd, f"{prefix}.edge_embedding")
  norm_e = _load_layernorm(feat.norm_edges, sd, f"{prefix}.norm_edges")
  feat = eqx.tree_at(lambda m: m.embeddings, feat, emb)
  feat = eqx.tree_at(lambda m: m.edge_embedding, feat, edge_emb)
  return eqx.tree_at(lambda m: m.norm_edges, feat, norm_e)


def convert_fampnn_state_dict(  # noqa: PLR0915
  raw_state_dict: Mapping[str, Any],
  *,
  key: jax.Array | None = None,
) -> FAMPNNModel:
  """Convert an upstream FAMPNN PyTorch state dict into an Equinox :class:`FAMPNNModel`.

  Parameters
  ----------
  raw_state_dict : Mapping[str, Any]
    Parameter ``raw_state_dict``.
  key : jax.Array | None
    Parameter ``key``.

  Returns
  -------
  FAMPNNModel
    Result of ``convert_fampnn_state_dict``.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  sd = normalize_fampnn_state_dict(raw_state_dict)
  has_conf = has_fampnn_confidence_head(sd)
  num_enc = sum(
    1 for k in sd if k.startswith("fa_encoder.encoder_layers.") and k.endswith(".W1.weight")
  )
  num_dec = sum(
    1 for k in sd if k.startswith("fa_encoder.decoder_layers.") and k.endswith(".W1.weight")
  )
  num_gvp = sum(
    1
    for k in sd
    if k.startswith("fa_encoder.gvp_encoder.encoder_layers.")
    and k.endswith(".conv.message_func.0.ws.weight")
  )
  scn_depth = sum(
    1 for k in sd if k.startswith("scn_diffusion.denoiser.blocks.") and k.endswith(".mlp.fc1.weight")
  )
  use_sc_feat = "fa_encoder.sidechain_features.edge_embedding.weight" in sd
  has_w_out = "W_out.weight" in sd
  has_f_emb = "scn_diffusion.denoiser.f_embedder.weight" in sd
  if "fa_encoder.gvp_encoder.W_out.1.ws.weight" in sd:
    gvp_w_out_mode = "gvp"
    gvp_rotate_out = False
  elif "fa_encoder.gvp_encoder.W_out.weight" in sd:
    gvp_w_out_mode = "linear"
    gvp_rotate_out = True
  else:
    gvp_w_out_mode = "norm"
    gvp_rotate_out = True

  conf_has_upstream = "scn_confidence.mlp.2.weight" in sd
  conf_num_bins = (
    int(sd["scn_confidence.mlp.2.weight"].shape[0]) // 33 if conf_has_upstream else 64
  )

  model = FAMPNNModel(
    has_confidence=has_conf,
    num_encoder_layers=max(num_enc, 3) if num_enc > 0 else 5,
    num_decoder_layers=max(num_dec, 3) if num_dec > 0 else 5,
    use_sidechain_features=use_sc_feat,
    num_gvp_layers=max(num_gvp, 3),
    gvp_w_out_mode=gvp_w_out_mode,
    gvp_rotate_out=gvp_rotate_out,
    has_w_out=has_w_out,
    scn_depth=max(scn_depth, 3),
    has_f_embedder=has_f_emb,
    conf_num_bins=conf_num_bins,
    conf_has_upstream_heads=conf_has_upstream,
    key=key,
  )

  # 1. Load FAEncoder
  fa: FAEncoder = model.fa_encoder
  fa_feat = _load_features(fa.features, sd, "fa_encoder.features")
  fa_gvp = _load_gvp_encoder(fa.gvp_encoder, sd, "fa_encoder.gvp_encoder")
  fa_wv = _load_linear(fa.W_v, sd, "fa_encoder.W_v")
  fa_we = _load_linear(fa.W_e, sd, "fa_encoder.W_e")
  fa_ws = fa.W_s
  if "fa_encoder.W_s.weight" in sd:
    fa_ws = eqx.tree_at(
      lambda m: m.weight, fa.W_s, jnp.asarray(sd["fa_encoder.W_s.weight"], dtype=jnp.float32),
    )
  if fa.sidechain_features is not None and fa.W_e2 is not None:
    sc_feat = _load_features(fa.sidechain_features, sd, "fa_encoder.sidechain_features")
    w_e2 = _load_linear(fa.W_e2, sd, "fa_encoder.W_e2")
    fa = eqx.tree_at(lambda m: m.sidechain_features, fa, sc_feat)
    fa = eqx.tree_at(lambda m: m.W_e2, fa, w_e2)

  fa_enc = tuple(
    _load_enc_or_edge_dec(lyr, sd, f"fa_encoder.encoder_layers.{i}")
    for i, lyr in enumerate(fa.encoder_layers)
  )
  fa_dec = tuple(
    _load_enc_or_edge_dec(lyr, sd, f"fa_encoder.decoder_layers.{i}")
    for i, lyr in enumerate(fa.decoder_layers)
  )
  fa = eqx.tree_at(lambda m: m.features, fa, fa_feat)
  fa = eqx.tree_at(lambda m: m.gvp_encoder, fa, fa_gvp)
  fa = eqx.tree_at(lambda m: m.W_v, fa, fa_wv)
  fa = eqx.tree_at(lambda m: m.W_e, fa, fa_we)
  fa = eqx.tree_at(lambda m: m.W_s, fa, fa_ws)
  fa = eqx.tree_at(lambda m: m.encoder_layers, fa, fa_enc)
  fa = eqx.tree_at(lambda m: m.decoder_layers, fa, fa_dec)

  # 2. Load sequence output head
  seq0 = _load_linear(model.mlp_seq_0, sd, "mlp_seq.0")
  seq2 = _load_linear(model.mlp_seq_2, sd, "mlp_seq.2")
  if model.W_out is not None and "W_out.weight" in sd:
    w_out = _load_linear(model.W_out, sd, "W_out")
    model = eqx.tree_at(lambda m: m.W_out, model, w_out)

  # 3. Load SidechainDiffusionModule
  scn: SidechainDiffusionModule = model.scn_diffusion
  scn_lin = _load_linear(scn.mlp_s_c_lin, sd, "scn_diffusion.mlp_s_c.0")
  scn_norm = _load_layernorm(scn.mlp_s_c_norm, sd, "scn_diffusion.mlp_s_c.1")
  x_emb = _load_linear(scn.x_embedder, sd, "scn_diffusion.denoiser.x_embedder")
  c_emb = _load_linear(scn.c_embedder, sd, "scn_diffusion.denoiser.c_embedder")
  if scn.f_embedder is not None and "scn_diffusion.denoiser.f_embedder.weight" in sd:
    f_emb = _load_linear(scn.f_embedder, sd, "scn_diffusion.denoiser.f_embedder")
    scn = eqx.tree_at(lambda m: m.f_embedder, scn, f_emb)

  t_mlp0 = _load_linear(scn.t_embedder.mlp_0, sd, "scn_diffusion.denoiser.t_embedder.mlp.0")
  t_mlp2 = _load_linear(scn.t_embedder.mlp_2, sd, "scn_diffusion.denoiser.t_embedder.mlp.2")
  t_emb: FAMPNNTimestepEmbedder = eqx.tree_at(lambda m: m.mlp_0, scn.t_embedder, t_mlp0)
  t_emb = eqx.tree_at(lambda m: m.mlp_2, t_emb, t_mlp2)

  blocks: list[FAMPNNDenoisingMLPBlock] = []
  for i, blk in enumerate(scn.blocks):
    bp = f"scn_diffusion.denoiser.blocks.{i}"
    b_new = eqx.tree_at(lambda m: m.mlp_fc1, blk, _load_linear(blk.mlp_fc1, sd, f"{bp}.mlp.fc1"))
    b_new = eqx.tree_at(lambda m: m.mlp_fc2, b_new, _load_linear(b_new.mlp_fc2, sd, f"{bp}.mlp.fc2"))
    b_new = eqx.tree_at(
      lambda m: m.adaLN_modulation,
      b_new,
      _load_linear(b_new.adaLN_modulation, sd, f"{bp}.adaLN_modulation.1"),
    )
    blocks.append(b_new)

  fl_lin = _load_linear(scn.final_layer.linear, sd, "scn_diffusion.denoiser.final_layer.linear")
  fl_mod = _load_linear(
    scn.final_layer.adaLN_modulation, sd, "scn_diffusion.denoiser.final_layer.adaLN_modulation.1",
  )
  fl: FAMPNNFinalLayer = eqx.tree_at(lambda m: m.linear, scn.final_layer, fl_lin)
  fl = eqx.tree_at(lambda m: m.adaLN_modulation, fl, fl_mod)

  scn = eqx.tree_at(lambda m: m.mlp_s_c_lin, scn, scn_lin)
  scn = eqx.tree_at(lambda m: m.mlp_s_c_norm, scn, scn_norm)
  scn = eqx.tree_at(lambda m: m.x_embedder, scn, x_emb)
  scn = eqx.tree_at(lambda m: m.c_embedder, scn, c_emb)
  scn = eqx.tree_at(lambda m: m.t_embedder, scn, t_emb)
  scn = eqx.tree_at(lambda m: m.blocks, scn, tuple(blocks))
  scn = eqx.tree_at(lambda m: m.final_layer, scn, fl)

  model = eqx.tree_at(lambda m: m.fa_encoder, model, fa)
  model = eqx.tree_at(lambda m: m.mlp_seq_0, model, seq0)
  model = eqx.tree_at(lambda m: m.mlp_seq_2, model, seq2)
  model = eqx.tree_at(lambda m: m.scn_diffusion, model, scn)

  # 4. Load optional SidechainConfidenceModule
  if has_conf and model.scn_confidence is not None:
    conf: SidechainConfidenceModule = model.scn_confidence
    c_feat = _load_features(conf.features, sd, "scn_confidence.features")
    c_gvp = _load_gvp_encoder(conf.gvp_encoder, sd, "scn_confidence.gvp_encoder")
    c_wv = _load_linear(conf.W_v, sd, "scn_confidence.W_v")
    c_we = _load_linear(conf.W_e, sd, "scn_confidence.W_e")
    c_ws = conf.W_s
    if "scn_confidence.W_s.weight" in sd:
      c_ws = eqx.tree_at(
        lambda m: m.weight,
        conf.W_s,
        jnp.asarray(sd["scn_confidence.W_s.weight"], dtype=jnp.float32),
      )
    c_enc = tuple(
      _load_enc_or_edge_dec(lyr, sd, f"scn_confidence.encoder_layers.{i}")  # type: ignore[arg-type]
      for i, lyr in enumerate(conf.encoder_layers)
    )
    if conf_has_upstream:
      c_dec: tuple[eqx.Module, ...] = tuple(
        _load_enc_or_edge_dec(lyr, sd, f"scn_confidence.decoder_layers.{i}")  # type: ignore[arg-type]
        for i, lyr in enumerate(conf.decoder_layers)
      )
    else:
      c_dec = tuple(
        _load_dec_layer(lyr, sd, f"scn_confidence.decoder_layers.{i}")  # type: ignore[arg-type]
        for i, lyr in enumerate(conf.decoder_layers)
      )
    w0 = _load_linear(conf.W_out_0, sd, "scn_confidence.W_out.0")
    w2 = _load_linear(conf.W_out_2, sd, "scn_confidence.W_out.2")
    w4 = _load_linear(conf.W_out_4, sd, "scn_confidence.W_out.4")
    conf = eqx.tree_at(lambda m: m.features, conf, c_feat)
    conf = eqx.tree_at(lambda m: m.gvp_encoder, conf, c_gvp)
    conf = eqx.tree_at(lambda m: m.W_v, conf, c_wv)
    conf = eqx.tree_at(lambda m: m.W_e, conf, c_we)
    conf = eqx.tree_at(lambda m: m.W_s, conf, c_ws)
    conf = eqx.tree_at(lambda m: m.encoder_layers, conf, c_enc)
    conf = eqx.tree_at(lambda m: m.decoder_layers, conf, c_dec)
    conf = eqx.tree_at(lambda m: m.W_out_0, conf, w0)
    conf = eqx.tree_at(lambda m: m.W_out_2, conf, w2)
    conf = eqx.tree_at(lambda m: m.W_out_4, conf, w4)
    if conf.aatype_embedder is not None and "scn_confidence.aatype_embedder.weight" in sd:
      ae = _load_linear(conf.aatype_embedder, sd, "scn_confidence.aatype_embedder")
      conf = eqx.tree_at(lambda m: m.aatype_embedder, conf, ae)
    if conf.sidechain_encoder is not None and "scn_confidence.sidechain_encoder.weight" in sd:
      se = _load_linear(conf.sidechain_encoder, sd, "scn_confidence.sidechain_encoder")
      conf = eqx.tree_at(lambda m: m.sidechain_encoder, conf, se)
    if conf.mlp_0 is not None and "scn_confidence.mlp.0.weight" in sd:
      m0 = _load_linear(conf.mlp_0, sd, "scn_confidence.mlp.0")
      conf = eqx.tree_at(lambda m: m.mlp_0, conf, m0)
    if conf.mlp_2 is not None and "scn_confidence.mlp.2.weight" in sd:
      m2 = _load_linear(conf.mlp_2, sd, "scn_confidence.mlp.2")
      conf = eqx.tree_at(lambda m: m.mlp_2, conf, m2)
    model = eqx.tree_at(lambda m: m.scn_confidence, model, conf)

  return model


def convert_fampnn_checkpoint(
  raw_state_dict: Mapping[str, Any],
  output_path: str | Path,
  *,
  checkpoint_id: str = "fampnn_0_3",
  key: jax.Array | None = None,
) -> tuple[FAMPNNModel, dict[str, Any]]:
  """Convert a PyTorch FAMPNN checkpoint, serialize ``.eqx``, and build its manifest.

  Parameters
  ----------
  raw_state_dict : Mapping[str, Any]
    Parameter ``raw_state_dict``.
  output_path : str | Path
    Parameter ``output_path``.
  checkpoint_id : str
    Parameter ``checkpoint_id``.
  key : jax.Array | None
    Parameter ``key``.

  Returns
  -------
  tuple[FAMPNNModel, dict[str, Any]]
    Result of ``convert_fampnn_checkpoint``.
  """
  import hashlib  # noqa: PLC0415

  sd = normalize_fampnn_state_dict(raw_state_dict)
  model = convert_fampnn_state_dict(sd, key=key)
  out_p = Path(output_path)
  out_p.parent.mkdir(parents=True, exist_ok=True)
  eqx.tree_serialise_leaves(out_p, model)
  output_sha256 = hashlib.sha256(out_p.read_bytes()).hexdigest()
  has_conf = model.scn_confidence is not None
  gvp_enc = model.fa_encoder.gvp_encoder
  if gvp_enc.w_out_gvp is not None:
    gvp_w_out_mode = "gvp"
  elif gvp_enc.w_out_linear is not None:
    gvp_w_out_mode = "linear"
  else:
    gvp_w_out_mode = "norm"
  manifest: dict[str, Any] = {
    "checkpoint_id": checkpoint_id,
    "family": "fampnn",
    "has_confidence": has_conf,
    "num_encoder_layers": len(model.fa_encoder.encoder_layers),
    "num_decoder_layers": len(model.fa_encoder.decoder_layers),
    "use_sidechain_features": model.fa_encoder.sidechain_features is not None,
    "num_gvp_layers": len(gvp_enc.encoder_layers),
    "gvp_w_out_mode": gvp_w_out_mode,
    "gvp_rotate_out": gvp_enc.rotate_out,
    "has_w_out": model.W_out is not None,
    "scn_depth": len(model.scn_diffusion.blocks),
    "has_f_embedder": model.scn_diffusion.f_embedder is not None,
    "conf_num_bins": model.scn_confidence.num_bins if model.scn_confidence is not None else 64,
    "conf_has_upstream_heads": (
      model.scn_confidence.mlp_2 is not None if model.scn_confidence is not None else False
    ),
    "num_tensors": len(sd),
    "output_sha256": output_sha256,
  }
  return model, manifest
