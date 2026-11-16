"""Equinox PyTree model definition for FrustraMPNN.

Ports the all-residues-visible ProteinMPNN trunk and FrustraMPNN transfer heads from:
- ``frustrampnn/model/transfer_model.py``
- ``frustrampnn/model/protein_mpnn.py``
"""

from __future__ import annotations

from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.model.features import ProteinFeatures, top_k
from aminx.model.mpnn import Aminx
from aminx.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance
from aminx.utils.graph import compute_neighbor_offsets
from aminx.utils.radial_basis import (
  RADIAL_BASES,
  RBF_SIGMA,
  compute_radial_basis,
  rbf_centers,
)

FRUSTRA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

_NODE_FEATURES = 128
_EDGE_FEATURES = 128
_HIDDEN_FEATURES = 128
_ENCODER_LAYERS = 3
_DECODER_LAYERS = 3
_K_NEIGHBORS = 48
_POSITIONAL_EMBEDDINGS = 32
_VOCAB = 21
_NUM_FINAL_LAYERS = 3


class FrustraTrunkOutput(NamedTuple):
  """All-residues-visible ProteinMPNN trunk output for FrustraMPNN.

  ``all_mpnn_hid`` is in reversed decoder-layer order (index 0 is decoder layer 2,
  index 1 is decoder layer 1, index 2 is decoder layer 0), matching upstream
  ``return list(reversed(all_hidden)), h_S, log_probs``.
  """

  all_mpnn_hid: tuple[Float[Array, "L 128"], ...]
  mpnn_embed: Float[Array, "L 128"]
  log_probs: Float[Array, "L 21"]
  encoder_out: Float[Array, "L 128"]
  raw_features: Float[Array, "L K 128"]
  edge_features: Float[Array, "L K 128"]
  neighbor_indices: Int[Array, "L K"]


class FrustraStageOutputs(NamedTuple):
  """Intermediate stage outputs for stage-by-stage parity verification."""

  features: Float[Array, "L K 128"]
  encoder_out: Float[Array, "L 128"]
  decoder_hidden: Float[Array, "3 L 128"]
  lin_input: Float[Array, "L 512"]
  light_attention_out: Float[Array, "L 512"]
  both_out: Float[Array, "L 21"]
  frustration_scores: Float[Array, "L 21"]


def cast_floating[ModuleT: eqx.Module](model: ModuleT, dtype: jnp.dtype) -> ModuleT:
  """Cast floating parameter arrays of ``model`` to ``dtype``."""

  def _cast(leaf: object) -> object:
    if eqx.is_array(leaf):
      array = cast("jax.Array", leaf)
      if jnp.issubdtype(array.dtype, jnp.floating):
        return array.astype(dtype)
    return leaf

  return cast("ModuleT", jax.tree.map(_cast, model))


def _rbf(distance: Float[Array, "L K"]) -> Float[Array, "L K 16"]:
  """Upstream ``_rbf``: 16 Gaussians on ``[2, 22]``."""
  scaled = (distance[..., None] - rbf_centers(distance.dtype)) / RBF_SIGMA
  return jnp.exp(-jnp.square(scaled))


def _linear(layer: eqx.nn.Linear, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
  """Apply ``eqx.nn.Linear`` over trailing axis of ``x``."""
  out = x @ layer.weight.T
  if layer.bias is not None:
    out = out + layer.bias
  return out


def _layer_norm(norm: eqx.nn.LayerNorm, x: Float[Array, "... H"]) -> Float[Array, "... H"]:
  """Apply ``eqx.nn.LayerNorm`` over trailing axis, matching standard ops."""
  mean = jnp.mean(x, axis=-1, keepdims=True)
  variance = jnp.var(x, axis=-1, keepdims=True)
  out = (x - mean) * jax.lax.rsqrt(variance + norm.eps)
  if norm.weight is not None:
    out = out * norm.weight
  if norm.bias is not None:
    out = out + norm.bias
  return out


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


def frustra_edge_features(
  features: ProteinFeatures,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
) -> tuple[Float[Array, "L K 128"], Float[Array, "L K 128"], Int[Array, "L K"]]:
  """Build the FrustraMPNN edge graph.

  Returns
  -------
  tuple
    (raw_features, projected_edge_features, neighbor_indices)
    raw_features: E (normalized edge embedding before W_e projection, matching oracle['features']).
    projected_edge_features: h_E (W_e(E) edge embedding fed to encoder/decoder).
    neighbor_indices: kNN neighbour indices.
  """
  backbone = compute_backbone_coordinates(coords)
  distances = compute_backbone_distance(backbone)
  mask_2d = mask[:, None] * mask[None, :]
  masked = mask_2d * distances
  distance_max = jnp.max(masked, axis=-1, keepdims=True)
  adjusted = masked + (1.0 - mask_2d) * distance_max

  k = min(features.k_neighbors, coords.shape[0])
  _values, neighbor_indices = top_k(-adjusted, k)
  neighbor_indices = jnp.asarray(neighbor_indices, dtype=jnp.int32)

  rbf = compute_radial_basis(backbone, neighbor_indices)
  ca_neighbors = jnp.take_along_axis(adjusted, neighbor_indices, axis=1)
  rbf = rbf.at[..., :RADIAL_BASES].set(_rbf(ca_neighbors).astype(rbf.dtype))

  neighbor_offsets = compute_neighbor_offsets(residue_index, neighbor_indices)
  edge_chains = (chain_index[:, None] == chain_index[None, :]).astype(jnp.int32)
  edge_chains_neighbors = jnp.take_along_axis(edge_chains, neighbor_indices, axis=1)
  max_relative = (features.w_pos.weight.shape[1] - 2) // 2
  neighbor_offset_factor = jnp.minimum(
    jnp.maximum(neighbor_offsets + max_relative, 0),
    2 * max_relative,
  )
  edge_chain_factor = (1 - edge_chains_neighbors) * (2 * max_relative + 1)
  encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
  encoded_offset_one_hot = jax.nn.one_hot(encoded_offset, 2 * max_relative + 2)
  encoded_positions = _linear(features.w_pos, encoded_offset_one_hot.astype(rbf.dtype))
  edges_concat = jnp.concatenate([encoded_positions, rbf], axis=-1)
  embedded = _linear(features.w_e, edges_concat)
  normalized = _layer_norm(features.norm_edges, embedded)
  projected = _linear(features.w_e_proj, normalized)
  return normalized, projected, neighbor_indices


def frustra_mpnn_trunk(
  mpnn: Aminx,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  sequence: Int[Array, " L"],
  chain_mask: Float[Array, " L"] | None = None,
  *,
  inference: bool = True,
) -> FrustraTrunkOutput:
  """Run all-residues-visible ProteinMPNN trunk (k=48, order_mask_backward=1.0).

  Parameters
  ----------
  mpnn : Aminx
    Stock ProteinMPNN trunk module.
  coords : Float[Array, "L 4 3"]
    Backbone (N, CA, C, O) coordinates.
  mask : Float[Array, " L"]
    Residue presence mask.
  residue_index : Int[Array, " L"]
    Residue sequence numbers.
  chain_index : Int[Array, " L"]
    Chain integer indices.
  sequence : Int[Array, " L"]
    Amino acid token indices in 0..20.
  chain_mask : Float[Array, " L"] | None
    Optional chain mask.
  inference : bool
    Whether to run in inference mode (disables dropout).

  Returns
  -------
  FrustraTrunkOutput
    Reversed decoder states, sequence embedding, log-probs, encoder out, and graph features.
  """
  del chain_mask  # Upstream FrustraMPNN does not apply chain_mask to order_mask_backward
  dtype = coords.dtype
  mpnn_cast = cast_floating(mpnn, dtype)
  present = mask.astype(dtype)
  coords_clean = jnp.nan_to_num(coords.astype(dtype), nan=0.0)

  raw_features, h_e, neighbor_indices = frustra_edge_features(
    mpnn_cast.features,
    coords_clean,
    present,
    residue_index.astype(jnp.int32),
    chain_index.astype(jnp.int32),
  )
  h_e = h_e.astype(dtype)
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

  encoder_out = h_v

  h_s = mpnn_cast.w_s_embed.weight[sequence.astype(jnp.int32)]
  h_es = _cat_neighbors_nodes(h_s, h_e, neighbor_indices)
  h_ex_encoder = _cat_neighbors_nodes(jnp.zeros_like(h_s), h_e, neighbor_indices)
  h_exv_encoder = _cat_neighbors_nodes(h_v, h_ex_encoder, neighbor_indices)

  # All residues visible conditioning: order_mask_backward = 1.0 everywhere
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
  return FrustraTrunkOutput(
    all_mpnn_hid=tuple(reversed(all_hidden)),
    mpnn_embed=h_s,
    log_probs=log_probs,
    encoder_out=encoder_out,
    raw_features=raw_features.astype(dtype),
    edge_features=h_e.astype(dtype),
    neighbor_indices=neighbor_indices,
  )


class LightAttention1D(eqx.Module):
  """1D light-attention refinement block for FrustraMPNN (Finding F-01).

  Evaluates in per-residue mode matching upstream native inference:
  Softmax collapses to 1.0, Conv1d evaluates to center filter slice
  W[:, :, 4] @ x + b.
  """

  feature_weight: Float[Array, "512 512 9"]
  feature_bias: Float[Array, " 512"]
  attention_weight: Float[Array, "512 512 9"]
  attention_bias: Float[Array, " 512"]
  embeddings_dim: int = eqx.field(static=True, default=512)
  kernel_size: int = eqx.field(static=True, default=9)

  def __init__(
    self,
    embeddings_dim: int = 512,
    kernel_size: int = 9,
    *,
    key: jax.Array,
  ) -> None:
    """Initialize LightAttention parameters."""
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

  def __call__(self, x: Float[Array, "... 512"]) -> Float[Array, "... 512"]:
    """Evaluate per-residue LightAttention.

    In singleton slice evaluation (L=1 per residue), attention Softmax reduces
    identically to 1.0 and Conv1d reduces to the center kernel weight slice.
    """
    center = self.kernel_size // 2
    feat = x @ self.feature_weight[:, :, center].T + self.feature_bias
    attn = x @ self.attention_weight[:, :, center].T + self.attention_bias
    weights = jax.nn.softmax(attn[..., None], axis=-1)[..., 0]
    return feat * weights


class BothOutMLP(eqx.Module):
  """2-layer MLP head projecting from 512-d to 21 vocabulary values.

  Architecture:
  - Layer 0: ReLU
  - Layer 1: Linear(512 -> 64)
  - Layer 2: ReLU
  - Layer 3: Linear(64 -> 32)
  - Layer 4: ReLU
  - Layer 5: Linear(32 -> 21)
  """

  layers: tuple[eqx.nn.Linear, eqx.nn.Linear, eqx.nn.Linear]

  def __init__(self, *, key: jax.Array) -> None:
    """Initialize BothOutMLP linear layers."""
    k1, k2, k3 = jax.random.split(key, 3)
    self.layers = (
      eqx.nn.Linear(512, 64, key=k1),
      eqx.nn.Linear(64, 32, key=k2),
      eqx.nn.Linear(32, 21, key=k3),
    )

  def __call__(self, x: Float[Array, "... 512"]) -> Float[Array, "... 21"]:
    """Forward pass through ReLU -> Linear(512->64) -> ReLU -> Linear(64->32) -> ReLU -> Linear(32->21)."""
    dtype = x.dtype
    h = jax.nn.relu(x)
    h = _linear_cast(self.layers[0], h, dtype)
    h = jax.nn.relu(h)
    h = _linear_cast(self.layers[1], h, dtype)
    h = jax.nn.relu(h)
    return _linear_cast(self.layers[2], h, dtype)


class FrustraMPNN(eqx.Module):
  """Equinox PyTree for FrustraMPNN local energetic frustration index prediction."""

  mpnn: Aminx
  light_attention: LightAttention1D
  both_out: BothOutMLP
  frustration_out: eqx.nn.Linear
  subtract_mut: bool = eqx.field(static=True, default=False)

  def __init__(
    self,
    *,
    key: jax.Array,
    subtract_mut: bool = False,
  ) -> None:
    """Initialize FrustraMPNN module with random weights."""
    k_mpnn, k_la, k_both, k_frust = jax.random.split(key, 4)
    self.subtract_mut = subtract_mut
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
      key=k_mpnn,
    )
    self.light_attention = LightAttention1D(
      embeddings_dim=512,
      kernel_size=9,
      key=k_la,
    )
    self.both_out = BothOutMLP(key=k_both)
    self.frustration_out = eqx.nn.Linear(1, 1, key=k_frust)

  def forward_stages(
    self,
    coords: Float[Array, "L 4 3"],
    mask: Float[Array, " L"],
    residue_index: Int[Array, " L"],
    chain_index: Int[Array, " L"],
    sequence: Int[Array, " L"],
    chain_mask: Float[Array, " L"] | None = None,
    *,
    inference: bool = True,
  ) -> FrustraStageOutputs:
    """Run full forward pass returning all intermediate stage tensors."""
    dtype = coords.dtype
    model = cast_floating(self, dtype)
    trunk = frustra_mpnn_trunk(
      model.mpnn,
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      chain_mask=chain_mask,
      inference=inference,
    )

    # 1. Latent concatenation: reversed decoder outputs (3x128 = 384-d) + sequence embedding h_S (128-d) = 512-d
    mpnn_hid = jnp.concatenate(trunk.all_mpnn_hid[:_NUM_FINAL_LAYERS], axis=-1)
    lin_input = jnp.concatenate([mpnn_hid, trunk.mpnn_embed], axis=-1)

    # 2. LightAttention1D in per-residue mode
    refined = model.light_attention(lin_input)

    # 3. Readout head: both_out (512 -> 64 -> 32 -> 21)
    both_out_scores = model.both_out(refined)

    # 4. Auxiliary scalar projection: frustration_out (Linear 1 -> 1)
    raw_frust = _linear_cast(model.frustration_out, both_out_scores[..., None], dtype)[..., 0]

    # 5. Wild-type subtraction (if subtract_mut=True)
    if self.subtract_mut:
      wt_scores = jnp.take_along_axis(raw_frust, sequence[:, None].astype(jnp.int32), axis=-1)
      raw_frust = raw_frust - wt_scores

    decoder_hidden = jnp.stack(trunk.all_mpnn_hid[:_NUM_FINAL_LAYERS], axis=0)

    return FrustraStageOutputs(
      features=trunk.raw_features,
      encoder_out=trunk.encoder_out,
      decoder_hidden=decoder_hidden,
      lin_input=lin_input,
      light_attention_out=refined,
      both_out=both_out_scores,
      frustration_scores=raw_frust,
    )

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
    """Predict (L, 21) single-residue local energetic frustration indices."""
    return self.forward_stages(
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      chain_mask=chain_mask,
      inference=inference,
    ).frustration_scores
