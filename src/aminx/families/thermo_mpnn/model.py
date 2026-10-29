"""ThermoMPNN (single-mutant) and ThermoMPNN-D (double-mutant epistatic) models.

Ports the all-residues-visible ProteinMPNN trunk and transfer heads from:
- ``ThermoMPNN/transfer_model.py`` and ``ThermoMPNN/protein_mpnn_utils.py`` (v1)
- ``ThermoMPNN-D/thermompnn/model/v2_model.py`` and ``ThermoMPNN-D/v2_ssm.py`` (v2)
"""

from __future__ import annotations

import itertools
from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.families.potts_mpnn.features import _layer_norm, _linear, potts_edge_features
from aminx.model.mpnn import Aminx

THERMO_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

_NODE_FEATURES = 128
_EDGE_FEATURES = 128
_HIDDEN_FEATURES = 128
_ENCODER_LAYERS = 3
_DECODER_LAYERS = 3
_K_NEIGHBORS = 48
_POSITIONAL_EMBEDDINGS = 32
_VOCAB = 21


class ThermoTrunkOutput(NamedTuple):
  """All-residues-visible ProteinMPNN trunk output for ThermoMPNN and ThermoMPNN-D.

  ``all_mpnn_hid`` is in reversed decoder-layer order (index 0 is decoder layer 2,
  index 1 is decoder layer 1, index 2 is decoder layer 0), matching upstream
  ``return list(reversed(all_hidden)), h_S, log_probs[, h_E]``.
  """

  all_mpnn_hid: tuple[Float[Array, "L 128"], ...]
  mpnn_embed: Float[Array, "L 128"]
  log_probs: Float[Array, "L 21"]
  mpnn_edges: Float[Array, "L K 128"]
  neighbor_indices: Int[Array, "L K"]


def cast_floating[ModuleT: eqx.Module](model: ModuleT, dtype: jnp.dtype) -> ModuleT:
  """Cast floating parameter arrays of ``model`` to ``dtype``.

  Parameters
  ----------
  model : _ModuleT
    Equinox module whose floating leaves will be cast.
  dtype : jnp.dtype
    Target floating-point dtype.

  Returns
  -------
  _ModuleT
    Module with all floating-point array leaves cast to ``dtype``.
  """

  def _cast(leaf: object) -> object:
    if eqx.is_array(leaf):
      array = cast("jax.Array", leaf)
      if jnp.issubdtype(array.dtype, jnp.floating):
        return array.astype(dtype)
    return leaf

  return cast("ModuleT", jax.tree.map(_cast, model))


def _linear_cast(
  layer: eqx.nn.Linear,
  x: Float[Array, "... I"],
  dtype: jnp.dtype,
) -> Float[Array, "... O"]:
  """Apply ``eqx.nn.Linear`` cast to ``dtype`` over the trailing axis of ``x``."""
  out = x @ layer.weight.astype(dtype).T
  if layer.bias is not None:
    out = out + layer.bias.astype(dtype)
  return out


def _cat_neighbors_nodes(
  nodes: Float[Array, "L C"],
  edges: Float[Array, "L K E"],
  neighbor_indices: Int[Array, "L K"],
) -> Float[Array, "L K E_plus_C"]:
  """Upstream ``cat_neighbors_nodes``: ``concat([edges, nodes[E_idx]], axis=-1)``."""
  return jnp.concatenate([edges, nodes[neighbor_indices]], axis=-1)


def thermo_mpnn_trunk(
  mpnn: Aminx,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  sequence: Int[Array, " L"],
  chain_mask: Float[Array, " L"] | None = None,
  *,
  apply_chain_mask_to_order: bool = False,
  inference: bool = True,
) -> ThermoTrunkOutput:
  """Run the all-residues-visible ProteinMPNN trunk used by ThermoMPNN.

  Matches ``ThermoMPNN/protein_mpnn_utils.py`` (lines 1222-1277, where
  ``apply_chain_mask_to_order=False``) and
  ``ThermoMPNN-D/thermompnn/protein_mpnn_utils.py`` (lines 1250-1333, where
  ``apply_chain_mask_to_order=True``).

  Parameters
  ----------
  mpnn : Aminx
    Stock ProteinMPNN trunk module.
  coords : Float[Array, "L 4 3"]
    Backbone ``(N, CA, C, O)`` coordinates.
  mask : Float[Array, " L"]
    Residue presence mask.
  residue_index : Int[Array, " L"]
    Residue sequence numbers.
  chain_index : Int[Array, " L"]
    Chain integer indices.
  sequence : Int[Array, " L"]
    Wild-type amino acid indices in ``0..20``.
  chain_mask : Float[Array, " L"] | None
    Optional designable-chain mask.
  apply_chain_mask_to_order : bool
    Whether to multiply ``order_mask_backward`` by ``chain_mask * mask`` (v2 behavior).
  inference : bool
    Whether to run in inference mode (disables dropout).

  Returns
  -------
  ThermoTrunkOutput
    Reversed decoder hidden states, sequence embeddings, log-probabilities, edges, and kNN indices.
  """
  dtype = coords.dtype
  mpnn_cast = cast_floating(mpnn, dtype)
  present = mask.astype(dtype)
  coords_clean = jnp.nan_to_num(coords.astype(dtype), nan=0.0)

  graph = potts_edge_features(
    mpnn_cast.features,
    coords_clean,
    present,
    residue_index.astype(jnp.int32),
    chain_index.astype(jnp.int32),
  )
  h_e = graph.edge_features.astype(dtype)
  neighbor_indices = graph.neighbor_indices.astype(jnp.int32)
  num_residues = h_e.shape[0]

  h_v = jnp.zeros((num_residues, mpnn_cast.encoder.node_feature_dim), dtype=dtype)
  mask_2d = present[:, None] * present[None, :]
  enc_mask_attend = jnp.take_along_axis(mask_2d, neighbor_indices, axis=1)
  for enc_layer in mpnn_cast.encoder.layers:
    updated_v, updated_e = enc_layer(
      h_v,
      h_e,
      neighbor_indices,
      present,
      mask_attend=enc_mask_attend,
      inference=inference,
      key=None,
    )
    h_v = cast("Array", updated_v)
    h_e = cast("Array", updated_e)

  h_s = mpnn_cast.w_s_embed.weight[sequence.astype(jnp.int32)]
  h_es = _cat_neighbors_nodes(h_s, h_e, neighbor_indices)
  h_ex_encoder = _cat_neighbors_nodes(jnp.zeros_like(h_s), h_e, neighbor_indices)
  h_exv_encoder = _cat_neighbors_nodes(h_v, h_ex_encoder, neighbor_indices)

  eff_chain_m = present if chain_mask is None else chain_mask.astype(dtype) * present
  if apply_chain_mask_to_order:
    ones_ll = jnp.ones((num_residues, num_residues), dtype=dtype)
    order_mask_backward = eff_chain_m[:, None] * ones_ll
  else:
    order_mask_backward = jnp.ones((num_residues, num_residues), dtype=dtype)

  dec_mask_attend = jnp.take_along_axis(order_mask_backward, neighbor_indices, axis=1)[..., None]
  mask_1d = present[:, None, None]
  mask_bw = mask_1d * dec_mask_attend
  mask_fw = mask_1d * (1.0 - dec_mask_attend)
  h_exv_encoder_fw = mask_fw * h_exv_encoder

  all_hidden: list[Array] = []
  for dec_layer in mpnn_cast.decoder.layers:
    h_esv = _cat_neighbors_nodes(h_v, h_es, neighbor_indices)
    h_esv = mask_bw * h_esv + h_exv_encoder_fw
    h_v = cast(
      "Array",
      dec_layer(
        h_v,
        h_esv,
        present,
        inference=inference,
        key=None,
      ),
    )
    all_hidden.append(h_v)

  logits = _linear(mpnn_cast.w_out, h_v)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  return ThermoTrunkOutput(
    all_mpnn_hid=tuple(reversed(all_hidden)),
    mpnn_embed=h_s,
    log_probs=log_probs,
    mpnn_edges=h_e,
    neighbor_indices=neighbor_indices,
  )


def _concat_final_hidden(
  all_mpnn_hid: tuple[Float[Array, "L 128"], ...],
  mpnn_embed: Float[Array, "L 128"],
  num_final_layers: int,
) -> Float[Array, "L C"]:
  """Concatenate ``all_mpnn_hid[:num_final_layers]`` and ``mpnn_embed`` along ``-1``."""
  if num_final_layers > 0:
    return jnp.concatenate([*all_mpnn_hid[:num_final_layers], mpnn_embed], axis=-1)
  return mpnn_embed


class LightAttention1D(eqx.Module):
  """1D light-attention block used by single-mutant ThermoMPNN (v1 ``K=9`` and v2 ``K=1``).

  Upstream ``TransferModel`` and ``TransferModelv2`` pass per-residue vectors
  with trailing length ``1`` (shape ``(B, C, 1)``) into ``LightAttention``,
  where ``Conv1d(C, C, kernel_size=K, padding=K // 2)`` evaluates the center
  kernel slice ``K // 2`` and ``Softmax(dim=-1)`` over length ``1`` is ``1.0``.
  """

  feature_weight: Float[Array, "C C K"]
  feature_bias: Float[Array, " C"]
  attention_weight: Float[Array, "C C K"]
  attention_bias: Float[Array, " C"]
  embeddings_dim: int = eqx.field(static=True)
  kernel_size: int = eqx.field(static=True)

  def __init__(
    self,
    embeddings_dim: int = 384,
    kernel_size: int = 9,
    *,
    key: jax.Array,
  ) -> None:
    """Initialize 1D light-attention weights and biases.

    Parameters
    ----------
    embeddings_dim : int
      Channel dimension ``C`` of input embeddings.
    kernel_size : int
      1D convolution kernel width ``K``.
    key : jax.Array
      PRNG key for parameter initialization.

    Returns
    -------
    None
      Initializes module attributes in place.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    scale = 1.0 / jnp.sqrt(float(embeddings_dim * kernel_size))
    self.embeddings_dim = embeddings_dim
    self.kernel_size = kernel_size
    self.feature_weight = jax.random.uniform(
      k1,
      (embeddings_dim, embeddings_dim, kernel_size),
      minval=-scale,
      maxval=scale,
    )
    self.feature_bias = jax.random.uniform(
      k2,
      (embeddings_dim,),
      minval=-scale,
      maxval=scale,
    )
    self.attention_weight = jax.random.uniform(
      k3,
      (embeddings_dim, embeddings_dim, kernel_size),
      minval=-scale,
      maxval=scale,
    )
    self.attention_bias = jax.random.uniform(
      k4,
      (embeddings_dim,),
      minval=-scale,
      maxval=scale,
    )

  def __call__(self, x: Float[Array, "... C"]) -> Float[Array, "... C"]:
    """Apply upstream per-residue length-1 LightAttention to ``x``.

    Parameters
    ----------
    x : Float[Array, "... C"]
      Input embedding tensor with trailing channel axis ``C``.

    Returns
    -------
    Float[Array, "... C"]
      Attended feature tensor of identical shape.
    """
    center = self.kernel_size // 2
    feat = x @ self.feature_weight[:, :, center].T + self.feature_bias
    attn = x @ self.attention_weight[:, :, center].T + self.attention_bias
    weights = jax.nn.softmax(attn[..., None], axis=-1)[..., 0]
    return feat * weights


class ThermoMPNN(eqx.Module):
  """Single-mutant $\\Delta\\Delta G^\\circ$ predictor (ThermoMPNN v1 and v2).

  Supports both ``thermoMPNN_default.pt`` (v1: ``kernel_size=9``,
  ``ddg_out=Linear(1, 1)``, ``apply_chain_mask_to_order=False``) and
  ``ThermoMPNN-ens{1,2,3}.ckpt`` (v2: ``kernel_size=1``, ``ddg_out=None``,
  ``apply_chain_mask_to_order=True``).
  """

  mpnn: Aminx
  light_attention: LightAttention1D
  mlp_layers: tuple[eqx.nn.Linear, ...]
  ddg_out: eqx.nn.Linear | None
  num_final_layers: int = eqx.field(static=True, default=2)
  subtract_mut: bool = eqx.field(static=True, default=True)
  apply_chain_mask_to_order: bool = eqx.field(static=True, default=False)

  def __init__(
    self,
    *,
    key: jax.Array,
    hidden_dims: tuple[int, ...] = (64, 32),
    num_final_layers: int = 2,
    subtract_mut: bool = True,
    light_attention_kernel_size: int = 9,
    has_ddg_out_scale: bool = True,
    apply_chain_mask_to_order: bool = False,
  ) -> None:
    """Initialize a single-mutant :class:`ThermoMPNN` module.

    Parameters
    ----------
    key : jax.Array
      PRNG key for parameter initialization.
    hidden_dims : tuple[int, ...]
      Hidden layer widths of the readout MLP.
    num_final_layers : int
      Number of reversed decoder hidden layers concatenated into the readout input.
    subtract_mut : bool
      Whether to subtract the wild-type self-mutation prediction at each position.
    light_attention_kernel_size : int
      Kernel width of :class:`LightAttention1D` (``9`` for v1, ``1`` for v2).
    has_ddg_out_scale : bool
      Whether to include the scalar ``1 -> 1`` ``ddg_out`` linear layer (v1).
    apply_chain_mask_to_order : bool
      Whether to multiply ``order_mask_backward`` by ``chain_mask * mask`` (v2).

    Returns
    -------
    None
      Initializes module attributes in place.
    """
    keys = jax.random.split(key, 3 + len(hidden_dims) + 1)
    self.num_final_layers = num_final_layers
    self.subtract_mut = subtract_mut
    self.apply_chain_mask_to_order = apply_chain_mask_to_order
    self.mpnn = Aminx(
      node_features=_NODE_FEATURES,
      edge_features=_EDGE_FEATURES,
      hidden_features=_HIDDEN_FEATURES,
      num_encoder_layers=_ENCODER_LAYERS,
      num_decoder_layers=_DECODER_LAYERS,
      k_neighbors=_K_NEIGHBORS,
      num_positional_embeddings=_POSITIONAL_EMBEDDINGS,
      num_amino_acids=_VOCAB,
      vocab_size=_VOCAB,
      key=keys[0],
    )
    embed_dim = _HIDDEN_FEATURES * num_final_layers + _NODE_FEATURES
    self.light_attention = LightAttention1D(
      embeddings_dim=embed_dim,
      kernel_size=light_attention_kernel_size,
      key=keys[1],
    )
    sizes = (embed_dim, *hidden_dims, _VOCAB)
    self.mlp_layers = tuple(
      eqx.nn.Linear(sz_in, sz_out, key=keys[2 + idx])
      for idx, (sz_in, sz_out) in enumerate(itertools.pairwise(sizes))
    )
    self.ddg_out = eqx.nn.Linear(1, 1, key=keys[-1]) if has_ddg_out_scale else None

  def __call__(
    self,
    coords: Float[Array, "L 4 3"],
    mask: Float[Array, " L"],
    residue_index: Int[Array, " L"],
    chain_index: Int[Array, " L"],
    sequence: Int[Array, " L"],
    chain_mask: Float[Array, " L"] | None = None,
    *,
    inference: bool = True,
  ) -> Float[Array, "L 21"]:
    """Predict ``(L, 21)`` single-mutant $\\Delta\\Delta G^\\circ$ values.

    Parameters
    ----------
    coords : Float[Array, "L 4 3"]
      Backbone ``(N, CA, C, O)`` coordinates.
    mask : Float[Array, " L"]
      Residue presence mask.
    residue_index : Int[Array, " L"]
      Residue sequence numbers.
    chain_index : Int[Array, " L"]
      Chain integer indices.
    sequence : Int[Array, " L"]
      Wild-type amino acid indices in ``0..20``.
    chain_mask : Float[Array, " L"] | None
      Optional designable-chain mask.
    inference : bool
      Whether to disable dropout.

    Returns
    -------
    Float[Array, "L 21"]
      Predicted $\\Delta\\Delta G^\\circ$ matrix across all 21 alphabet tokens.
    """
    dtype = coords.dtype
    model = cast_floating(self, dtype)
    trunk = thermo_mpnn_trunk(
      model.mpnn,
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      chain_mask=chain_mask,
      apply_chain_mask_to_order=self.apply_chain_mask_to_order,
      inference=inference,
    )
    z = _concat_final_hidden(trunk.all_mpnn_hid, trunk.mpnn_embed, self.num_final_layers)
    h = model.light_attention(z)
    for layer in self.mlp_layers:
      h = jax.nn.relu(h)
      h = _linear_cast(layer, h, dtype)
    raw_ddg = h
    if model.ddg_out is not None:
      raw_ddg = _linear(model.ddg_out, raw_ddg[..., None])[..., 0]
    if self.subtract_mut:
      wt_ddg = jnp.take_along_axis(raw_ddg, sequence[:, None].astype(jnp.int32), axis=-1)
      raw_ddg = raw_ddg - wt_ddg
    return raw_ddg


def _siamese_mlp(
  layers: tuple[eqx.nn.Linear, ...],
  x: Float[Array, "M 256"],
) -> Float[Array, "M 1"]:
  """Apply Siamese ``ddg_out`` MLP (``Linear -> ReLU`` on hidden layers, ``Linear`` on output)."""
  h = x
  for layer in layers[:-1]:
    h = jax.nn.relu(_linear(layer, h))
  return _linear(layers[-1], h)


class ThermoMPNNDouble(eqx.Module):
  """Double-mutant epistatic Siamese predictor (``TransferModelv2Siamese`` / ThermoMPNN-D).

  Matches ``ThermoMPNN-D/thermompnn/model/v2_model.py`` (lines 359-527) and
  ``ThermoMPNN-D/v2_ssm.py`` (lines 78-221).
  """

  mpnn: Aminx
  proj_norm: eqx.nn.LayerNorm
  proj_linear: eqx.nn.Linear
  mlp_layers: tuple[eqx.nn.Linear, ...]
  num_final_layers: int = eqx.field(static=True, default=2)

  def __init__(
    self,
    *,
    key: jax.Array,
    hidden_dims: tuple[int, ...] = (128, 128),
    num_final_layers: int = 2,
  ) -> None:
    """Initialize a double-mutant Siamese :class:`ThermoMPNNDouble` module.

    Parameters
    ----------
    key : jax.Array
      PRNG key for parameter initialization.
    hidden_dims : tuple[int, ...]
      Hidden layer widths of the Siamese ``ddg_out`` MLP.
    num_final_layers : int
      Number of reversed decoder hidden layers concatenated into each slot representation.

    Returns
    -------
    None
      Initializes module attributes in place.
    """
    keys = jax.random.split(key, 2 + len(hidden_dims) + 1)
    self.num_final_layers = num_final_layers
    self.mpnn = Aminx(
      node_features=_NODE_FEATURES,
      edge_features=_EDGE_FEATURES,
      hidden_features=_HIDDEN_FEATURES,
      num_encoder_layers=_ENCODER_LAYERS,
      num_decoder_layers=_DECODER_LAYERS,
      k_neighbors=_K_NEIGHBORS,
      num_positional_embeddings=_POSITIONAL_EMBEDDINGS,
      num_amino_acids=_VOCAB,
      vocab_size=_VOCAB,
      key=keys[0],
    )
    proj_in_dim = _HIDDEN_FEATURES * num_final_layers + _NODE_FEATURES + _EDGE_FEATURES
    self.proj_norm = eqx.nn.LayerNorm(proj_in_dim)
    self.proj_linear = eqx.nn.Linear(proj_in_dim, _HIDDEN_FEATURES, key=keys[1])
    sizes = (_HIDDEN_FEATURES * 2, *hidden_dims, 1)
    self.mlp_layers = tuple(
      eqx.nn.Linear(sz_in, sz_out, key=keys[2 + idx])
      for idx, (sz_in, sz_out) in enumerate(itertools.pairwise(sizes))
    )

  def __call__(
    self,
    coords: Float[Array, "L 4 3"],
    mask: Float[Array, " L"],
    residue_index: Int[Array, " L"],
    chain_index: Int[Array, " L"],
    sequence: Int[Array, " L"],
    mut_positions: Int[Array, "M 2"],
    mut_aas: Int[Array, "M 2"],
    chain_mask: Float[Array, " L"] | None = None,
    *,
    inference: bool = True,
  ) -> Float[Array, " M"]:
    """Predict ``(M,)`` symmetric double-mutant $\\Delta\\Delta G^\\circ$ values.

    Parameters
    ----------
    coords : Float[Array, "L 4 3"]
      Backbone ``(N, CA, C, O)`` coordinates.
    mask : Float[Array, " L"]
      Residue presence mask.
    residue_index : Int[Array, " L"]
      Residue sequence numbers.
    chain_index : Int[Array, " L"]
      Chain integer indices.
    sequence : Int[Array, " L"]
      Wild-type amino acid indices in ``0..20``.
    mut_positions : Int[Array, "M 2"]
      Zero-based residue index pairs ``(pos_1, pos_2)`` for ``M`` double mutations.
    mut_aas : Int[Array, "M 2"]
      Mutant amino acid index pairs ``(aa_1, aa_2)`` in ``0..20`` for ``M`` double mutations.
    chain_mask : Float[Array, " L"] | None
      Optional designable-chain mask.
    inference : bool
      Whether to disable dropout.

    Returns
    -------
    Float[Array, " M"]
      Order-invariant double-mutant $\\Delta\\Delta G^\\circ$ predictions for each pair.
    """
    dtype = coords.dtype
    model = cast_floating(self, dtype)
    trunk = thermo_mpnn_trunk(
      model.mpnn,
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      chain_mask=chain_mask,
      apply_chain_mask_to_order=True,
      inference=inference,
    )
    struct_embed = jnp.concatenate(trunk.all_mpnn_hid[: self.num_final_layers], axis=-1)
    wt_embed = trunk.mpnn_embed
    mpnn_edges = trunk.mpnn_edges
    neighbor_indices = trunk.neighbor_indices

    slot_embeds: list[Array] = []
    for m in range(2):
      pos_m = mut_positions[:, m].astype(jnp.int32)
      pos_other = mut_positions[:, 1 - m].astype(jnp.int32)
      aa_m = mut_aas[:, m].astype(jnp.int32)

      e_idx_m = neighbor_indices[pos_m]
      edges_m = mpnn_edges[pos_m]
      matches = e_idx_m == pos_other[:, None]
      has_match = jnp.any(matches, axis=-1, keepdims=True)
      first_k = jnp.argmax(matches, axis=-1)
      selected_edge = jnp.take_along_axis(edges_m, first_k[:, None, None], axis=1)[:, 0, :]
      edge_vec = jnp.where(has_match, selected_edge, jnp.zeros_like(selected_edge))

      g_struct = struct_embed[pos_m]
      g_seq = wt_embed[pos_m] - model.mpnn.w_s_embed.weight[aa_m]
      g_embed = jnp.concatenate([g_struct, g_seq, edge_vec], axis=-1)
      h_m = jax.nn.relu(_linear(model.proj_linear, _layer_norm(model.proj_norm, g_embed)))
      slot_embeds.append(h_m)

    h_0, h_1 = slot_embeds[0], slot_embeds[1]
    embed_ab = jnp.concatenate([h_0, h_1], axis=-1)
    embed_ba = jnp.concatenate([h_1, h_0], axis=-1)
    ddg_a = _siamese_mlp(model.mlp_layers, embed_ab)[..., 0]
    ddg_b = _siamese_mlp(model.mlp_layers, embed_ba)[..., 0]
    return 0.5 * (ddg_a + ddg_b)
