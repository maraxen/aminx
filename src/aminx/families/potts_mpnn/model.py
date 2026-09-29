"""PottsMPNN composition: stock Aminx plus a Potts head.

``ModelProtocol`` stays on ``mpnn``. This module does not register a family driver.
"""

from __future__ import annotations

from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.families.potts_mpnn.potts_head import PottsHead
from aminx.model.decoder import (
  conditional_decoder_layer_edge_features,
  pack_conditional_decoder_static_edges,
)
from aminx.model.mpnn import Aminx
from aminx.utils.autoregression import ar_mask_from_decoding_order

_NODE_FEATURES = 128
_EDGE_FEATURES = 128
_HIDDEN_FEATURES = 128
_ENCODER_LAYERS = 3
_DECODER_LAYERS = 3
_K_NEIGHBORS = 48
_POSITIONAL_EMBEDDINGS = 32
_VOCAB = 21


class PottsMPNNOutput(NamedTuple):
  """Teacher-forced PottsMPNN forward.

  Encoder stacks keep the layer-0 input, then each layer output (the oracle
  hook convention). Decoder node stacks do the same; decoder edge stacks are
  the packed context fed to each layer.
  """

  enc_h_v: tuple[Array, ...]
  enc_h_e: tuple[Array, ...]
  neighbor_indices: Int[Array, "L K"]
  etab_raw: Float[Array, "L K 20 20"]
  log_probs: Float[Array, "L 21"]
  dec_h_v: tuple[Array, ...]
  dec_h_e: tuple[Array, ...]


class PottsMPNN(eqx.Module):
  """Stock ProteinMPNN (``mpnn``) plus ``potts_head``.

  ``__call__`` follows the floating dtype of ``coords``: parameters are cast
  for the forward, and every returned array uses that dtype.
  """

  mpnn: Aminx
  potts_head: PottsHead

  def __init__(self, *, key: jax.Array) -> None:
    mpnn_key, head_key = jax.random.split(key)
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
      key=mpnn_key,
    )
    self.potts_head = PottsHead(_EDGE_FEATURES, key=head_key)

  def __call__(
    self,
    coords: Float[Array, "L 4 3"],
    mask: Float[Array, " L"],
    residue_index: Int[Array, " L"],
    chain_index: Int[Array, " L"],
    sequence: Int[Array, " L"],
    decoding_order: Int[Array, " L"],
    pad_valid: Float[Array, " L"] | None = None,
    *,
    inference: bool = True,
  ) -> PottsMPNNOutput:
    """Encode, read ``etab_raw``, and teacher-force MPNN log-probabilities.

    ``decoding_order`` is an ORDER array (``decoding_order[t]`` is the position
    decoded at step ``t``), matching upstream ``use_input_decoding_order``.
    """
    dtype = coords.dtype
    model = cast_floating(self, dtype)
    present = mask.astype(dtype)
    valid = jnp.ones_like(present) if pad_valid is None else pad_valid.astype(dtype)
    enc_h_v, enc_h_e, neighbor_indices = _encoder_states(
      model.mpnn,
      coords.astype(dtype),
      present,
      residue_index,
      chain_index,
      inference=inference,
    )
    etab_raw = model.potts_head(enc_h_e[-1], neighbor_indices, present, valid)
    log_probs, dec_h_v, dec_h_e = _teacher_forced(
      model.mpnn,
      enc_h_v[-1],
      enc_h_e[-1],
      neighbor_indices,
      present,
      sequence,
      decoding_order,
      inference=inference,
    )
    return PottsMPNNOutput(
      enc_h_v=enc_h_v,
      enc_h_e=enc_h_e,
      neighbor_indices=neighbor_indices,
      etab_raw=etab_raw,
      log_probs=log_probs,
      dec_h_v=dec_h_v,
      dec_h_e=dec_h_e,
    )


def cast_floating(model: PottsMPNN, dtype: jnp.dtype) -> PottsMPNN:
  """Cast floating parameter arrays. Integer and static fields stay put."""

  def _cast(leaf: object) -> object:
    if eqx.is_array(leaf):
      array = cast("jax.Array", leaf)
      if jnp.issubdtype(array.dtype, jnp.floating):
        return array.astype(dtype)
    return leaf

  return cast("PottsMPNN", jax.tree.map(_cast, model))


def _encoder_states(
  mpnn: Aminx,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  *,
  inference: bool,
) -> tuple[tuple[Array, ...], tuple[Array, ...], Int[Array, "L K"]]:
  """Feature graph plus per-layer encoder states.

  Index 0 is the input of encoder layer 0 (node features are zeros, matching
  upstream ``h_V = 0``). Later indices are layer outputs.
  """
  raw_edges, raw_neighbors, _node_features, _key = mpnn.features(
    jax.random.key(0),
    coords,
    mask,
    residue_index,
    chain_index,
    jnp.zeros((), dtype=coords.dtype),
  )
  # mpnn.features promotes to float64 under jax_enable_x64 even for float32
  # coords; pin the edge graph to the input dtype so the forward follows it.
  edge_features = cast("Array", raw_edges).astype(coords.dtype)
  neighbor_indices = cast('Int[Array, "L K"]', raw_neighbors)
  node_features = jnp.zeros(
    (edge_features.shape[0], mpnn.encoder.node_feature_dim),
    dtype=edge_features.dtype,
  )
  mask_2d = mask[:, None] * mask[None, :]
  mask_attend = jnp.take_along_axis(
    mask_2d,
    neighbor_indices.astype(jnp.int32),
    axis=1,
  )
  node_states = [node_features]
  edge_states = [edge_features]
  for layer in mpnn.encoder.layers:
    updated_nodes, updated_edges = layer(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      mask_attend=mask_attend,
      inference=inference,
      key=None,
    )
    node_features = cast("Array", updated_nodes)
    edge_features = cast("Array", updated_edges)
    node_states.append(node_features)
    edge_states.append(edge_features)
  return tuple(node_states), tuple(edge_states), neighbor_indices


def _teacher_forced(
  mpnn: Aminx,
  node_features: Float[Array, "L H"],
  edge_features: Float[Array, "L K H"],
  neighbor_indices: Int[Array, "L K"],
  mask: Float[Array, " L"],
  sequence: Int[Array, " L"],
  decoding_order: Int[Array, " L"],
  *,
  inference: bool,
) -> tuple[Float[Array, "L 21"], tuple[Array, ...], tuple[Array, ...]]:
  """Full-sequence conditional decoder used by the ``pottsmpnn_full`` wave."""
  one_hot = jax.nn.one_hot(sequence, mpnn.w_s_embed.num_embeddings, dtype=node_features.dtype)
  ar_mask = cast("Array", ar_mask_from_decoding_order(decoding_order.astype(jnp.int32)))
  sequence_edge_features, mask_bw, masked_node_edge_features = (
    pack_conditional_decoder_static_edges(
      node_features,
      edge_features,
      neighbor_indices,
      one_hot,
      mpnn.w_s_embed.weight,
      ar_mask,
      mask,
    )
  )
  node_states = [node_features]
  edge_states: list[Array] = []
  for layer in mpnn.decoder.layers:
    layer_edge_features = conditional_decoder_layer_edge_features(
      node_features,
      sequence_edge_features,
      neighbor_indices,
      mask_bw,
      masked_node_edge_features,
    )
    edge_states.append(layer_edge_features)
    node_features = cast(
      "Array",
      layer(
        node_features,
        layer_edge_features,
        mask,
        inference=inference,
        key=None,
      ),
    )
    node_states.append(node_features)
  # Row-wise readout as a matmul: L-DRV R1 bans vmap in families/.
  w_out = mpnn.w_out
  logits = node_features @ w_out.weight.T
  if w_out.bias is not None:
    logits = logits + w_out.bias
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  return log_probs, tuple(node_states), tuple(edge_states)
