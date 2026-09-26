"""Decoder module for Aminx.

Contains ``DecoderLayer``, ``DecoderLayerJ``, ``Decoder``, and **orchestration helpers**
(``pack_decoder_unconditional_layer_edge_features``, conditional packing helpers) split out in
Phase **5b** so ``mpnn.py`` stays focused on model wiring.
"""

# NOTE: Public API surface audited Sprint 3 Wave 5. Docstrings complete as of refactor-full.

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from aminx.model.dropout import Dropout
from aminx.utils.concatenate import concatenate_neighbor_nodes

if TYPE_CHECKING:
  from aminx.types.arrays import (
    AlphaCarbonMask,
    Array,
    AutoRegressiveMask,
    EdgeFeatures,
    NeighborIndices,
    NodeFeatures,
    OneHotProteinSequence,
  )

# Layer normalization with a standard epsilon
LayerNorm = eqx.nn.LayerNorm
_gelu = partial(jax.nn.gelu, approximate=False)


def pack_decoder_unconditional_layer_edge_features(
  node_features: jax.Array,
  edge_features: jax.Array,
  neighbor_indices: jax.Array,
) -> jax.Array:
  """Pack unconditional decoder edge features from node and edge tensors.

  Builds neighbor context as ``[h_i, 0, e_ij, h_j]`` for each position-neighbor pair.

  Parameters
  ----------
  node_features : jax.Array
    Node feature matrix. Shape ``(L, D)``.
  edge_features : jax.Array
    Edge feature matrix. Shape ``(L, K, D)``.
  neighbor_indices : jax.Array
    Neighbor indices for gather. Shape ``(L, K)``.

  Returns
  -------
  jax.Array
    Packed neighbor tensor matching unconditional decoder input format.
    Shape ``(L, K, 3D)``.

  Notes
  -----
  This matches the legacy ProteinMPNN unconditional decode convention.

  """
  zeros_with_edges = concatenate_neighbor_nodes(
    jnp.zeros_like(node_features),
    edge_features,
    neighbor_indices,
  )
  return concatenate_neighbor_nodes(
    node_features,
    zeros_with_edges,
    neighbor_indices,
  )


def pack_conditional_decoder_static_edges(
  node_features: jax.Array,
  edge_features: jax.Array,
  neighbor_indices: jax.Array,
  one_hot_sequence: jax.Array,
  w_s_weight: jax.Array,
  ar_mask: jax.Array,
  mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Build pre-computed tensors for conditional decoder layers.

  Prepares edge contexts, attention masks, and position masks that remain
  constant across all conditional decoder layer passes.

  Parameters
  ----------
  node_features : jax.Array
    Node feature matrix. Shape ``(L, D)``.
  edge_features : jax.Array
    Edge feature matrix. Shape ``(L, K, D)``.
  neighbor_indices : jax.Array
    Neighbor indices for gather. Shape ``(L, K)``.
  one_hot_sequence : jax.Array
    One-hot encoded protein sequence. Shape ``(L, 21)``.
  w_s_weight : jax.Array
    Sequence embedding weight matrix. Shape ``(21, D)``.
  ar_mask : jax.Array
    Autoregressive mask for conditional decoding. Shape ``(L, L)``.
  mask : jax.Array
    Alpha carbon mask. Shape ``(L,)``.

  Returns
  -------
  tuple[jax.Array, jax.Array, jax.Array]
    - ``sequence_edge_features``: Packed sequence+edge context. Shape ``(L, K, D)``.
    - ``mask_bw``: Backward (already-decoded) attention mask. Shape ``(L, K)``.
    - ``masked_node_edge_features``: Forward (future-token) edge context. Shape ``(L, K, D)``.

  Notes
  -----
  These tensors are computed once and reused across all decoder layer iterations
  to avoid redundant computation.

  """
  embedded_sequence = jnp.atleast_2d(one_hot_sequence) @ w_s_weight

  temp_node_edge = concatenate_neighbor_nodes(
    jnp.zeros_like(node_features),
    edge_features,
    neighbor_indices,
  )

  node_edge_features = concatenate_neighbor_nodes(
    node_features,
    temp_node_edge,
    neighbor_indices,
  )

  sequence_edge_features = concatenate_neighbor_nodes(
    embedded_sequence,
    edge_features,
    neighbor_indices,
  )

  attention_mask = jnp.take_along_axis(ar_mask, neighbor_indices, axis=1)
  mask_bw = mask[:, None] * attention_mask
  mask_fw = mask[:, None] * (1 - attention_mask)
  masked_node_edge_features = mask_fw[..., None] * node_edge_features
  return sequence_edge_features, mask_bw, masked_node_edge_features


def conditional_decoder_layer_edge_features(
  loop_node_features: jax.Array,
  sequence_edge_features: jax.Array,
  neighbor_indices: jax.Array,
  mask_bw: jax.Array,
  masked_node_edge_features: jax.Array,
) -> jax.Array:
  """Assemble per-layer edge context for conditional (scoring) decoding.

  Gathers neighbor node features and concatenates with sequence edge context,
  then applies backward mask and adds pre-computed masked edge features.
  Called once per layer in :meth:`Decoder.call_conditional`.

  Parameters
  ----------
  loop_node_features : jax.Array
      Updated node features from current layer. Shape ``(L, D)``.
  sequence_edge_features : jax.Array
      Sequence-to-sequence edge context. Shape ``(L, K, D_edge)``.
  neighbor_indices : jax.Array
      Neighbor indices for gathering. Shape ``(L, K)``.
  mask_bw : jax.Array
      Backward (causal) mask for conditional decoding. Shape ``(L,)``.
  masked_node_edge_features : jax.Array
      Pre-masked node-edge features. Shape ``(L, K, D)``.

  Returns
  -------
  jax.Array
      Per-layer edge context with mask and pre-computed terms applied.
      Shape ``(L, K, D_combined)``.

  """
  current_features = concatenate_neighbor_nodes(
    loop_node_features,
    sequence_edge_features,
    neighbor_indices,
  )
  return (mask_bw[..., None] * current_features) + masked_node_edge_features


class DecoderLayer(eqx.Module):
  """Single message-passing decoder layer for ProteinMPNN.

  Reads from pre-packed neighbor context (node features + edge context),
  applies 2-layer MLP message update, layer norm, dense feedforward,
  residual connections, and dropout.

  Parameters
  ----------
  message_mlp : eqx.nn.MLP
    Message network combining node + edge context.
  norm1 : LayerNorm
    Layer normalization after message aggregation.
  dense : eqx.nn.MLP
    Feedforward network.
  norm2 : LayerNorm
    Layer normalization after feedforward.
  dropout1 : Dropout
    Dropout after message aggregation.
  dropout2 : Dropout
    Dropout after feedforward.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  message_mlp: eqx.nn.MLP
  norm1: LayerNorm
  dense: eqx.nn.MLP
  norm2: LayerNorm
  dropout1: Dropout
  dropout2: Dropout

  def __init__(
    self,
    node_features: int,
    edge_context_features: int,
    _hidden_features: int,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the decoder layer.

    Parameters
    ----------
    node_features : int
        Dimension of node features (e.g., 128).
    edge_context_features : int
        Dimension of edge context (e.g., 384).
    _hidden_features : int
        Dimension of hidden layer in dense MLP.
    dropout_rate : float
        Dropout rate. Default: 0.1.
    key : PRNGKeyArray
        PRNG key for weight initialization.

    """
    keys = jax.random.split(key, 4)

    self.dropout1 = Dropout(dropout_rate)
    self.dropout2 = Dropout(dropout_rate)

    # Input dim is [h_i, e_context]
    mlp_input_dim = node_features + edge_context_features

    self.message_mlp = eqx.nn.MLP(
      in_size=mlp_input_dim,
      out_size=node_features,
      width_size=node_features,
      depth=2,
      activation=_gelu,
      key=keys[2],
    )
    self.norm1 = LayerNorm(node_features)
    self.dense = eqx.nn.MLP(
      in_size=node_features,
      out_size=node_features,
      width_size=node_features * 4,
      depth=1,
      activation=_gelu,
      key=keys[3],
    )
    self.norm2 = LayerNorm(node_features)

  def __call__(
    self,
    node_features: NodeFeatures,
    layer_edge_features: EdgeFeatures,  # This is the (N, K, 384) context
    mask: AlphaCarbonMask,
    scale: float = 30.0,
    attention_mask: Array | None = None,
    *,
    inference: bool = False,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for the decoder layer.

    Parameters
    ----------
    node_features : NodeFeatures
        Node features from encoder. Shape ``(L, D)``.
    layer_edge_features : EdgeFeatures
        Pre-packed edge context. Shape ``(L, K, D_edge)``.
    mask : AlphaCarbonMask
        Alpha carbon mask. Shape ``(L,)``.
    scale : float
        Message aggregation scale factor. Default: 30.0.
    attention_mask : Array | None
        Optional attention mask for conditional decoding. Shape ``(L, K)``.
    inference : bool
        If True, disable dropout. Default: False.
    key : PRNGKeyArray | None
        PRNG key for dropout (optional).

    Returns
    -------
    NodeFeatures
        Updated node features. Shape ``(L, D)``.

    """
    # Pass the key to jax.random.split for potential dropout use
    if key is None:
      inference = True
    keys = jax.random.split(key, 2) if key is not None else (None, None)

    node_features_expand = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, layer_edge_features.shape[1], 1],
    )

    # Concat with context [h_i, e_context]
    mlp_input = jnp.concatenate([node_features_expand, layer_edge_features], -1)

    # Apply MLP to each (atom, neighbor) pair: vmap over atoms, then over neighbors
    message = jax.vmap(jax.vmap(self.message_mlp))(mlp_input)

    # Apply attention mask if provided (for conditional decoding)
    if attention_mask is not None:
      mask_cast = attention_mask.astype(message.dtype)
      message = jnp.expand_dims(mask_cast, -1) * message

    # Stability fix: Accumulate message sums in float32
    message_f32 = message.astype(jnp.float32)
    aggregated_message_f32 = jnp.sum(message_f32, -2) / scale
    aggregated_message = aggregated_message_f32.astype(message.dtype)

    # Aggregate messages and apply dropout
    h_V = node_features + self.dropout1(aggregated_message, key=keys[0], inference=inference)
    h_V = jax.vmap(self.norm1)(h_V)

    # Dense layer and residue connection
    h_dense = jax.vmap(self.dense)(h_V)
    h_V = h_V + self.dropout2(h_dense, key=keys[1], inference=inference)
    h_V = jax.vmap(self.norm2)(h_V)

    # Handle both batched (N,) mask and scalar mask
    if jnp.ndim(mask) == 0:
      return mask * h_V
    return mask[:, None] * h_V


class DecoderLayerJ(eqx.Module):
  """Gated attention decoder for ligand-atom context encoding.

  Named after the upstream LigandMPNN ``DecLayerJ`` reference implementation.
  Used within ``Packer`` as ``y_context_encoder_layers`` to encode ligand atom
  context via multi-head message passing. Operates on 3-D node tensors
  ``[L, M, D]`` and 4-D edge tensors ``[L, M, M, D]``, where ``L`` is sequence
  length, ``M`` is the ligand context neighbourhood size, and ``D`` is the
  hidden dimension.

  Parameters
  ----------
  w1 : eqx.nn.Linear
      First message projection. Input: concatenated node-edge context.
  w2 : eqx.nn.Linear
      Second message projection (gating layer).
  w3 : eqx.nn.Linear
      Third message projection (gating layer).
  dense : eqx.nn.MLP
      Feedforward network for residual update.
  norm1 : eqx.nn.LayerNorm
      Layer normalization after message aggregation.
  norm2 : eqx.nn.LayerNorm
      Layer normalization after feedforward.
  dropout1 : Dropout
      Dropout after message aggregation.
  dropout2 : Dropout
      Dropout after feedforward.
  scale : float
      Message aggregation scale factor. Static (not a JAX array).

  References
  ----------
  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
     https://github.com/dauparas/LigandMPNN

  """

  w1: eqx.nn.Linear
  w2: eqx.nn.Linear
  w3: eqx.nn.Linear
  dense: eqx.nn.MLP
  norm1: eqx.nn.LayerNorm
  norm2: eqx.nn.LayerNorm
  dropout1: Dropout
  dropout2: Dropout
  scale: float = eqx.field(static=True)

  def __init__(
    self,
    hidden_dim: int,
    in_dim: int,
    dropout: float = 0.1,
    scale: float = 30.0,
    *,
    key: PRNGKeyArray,
  ):
    """Initialize the gated attention decoder layer.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for all linear projections and layer norms.
    in_dim : int
        Input dimension for edge context (concatenated with node features).
    dropout : float
        Dropout rate. Default: 0.1.
    scale : float
        Message aggregation scale factor. Default: 30.0.
    key : PRNGKeyArray
        PRNG key for weight initialization.

    """
    keys = jax.random.split(key, 5)
    self.w1 = eqx.nn.Linear(hidden_dim + in_dim, hidden_dim, key=keys[0])
    self.w2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1])
    self.w3 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[2])
    self.dense = eqx.nn.MLP(
      hidden_dim, hidden_dim, hidden_dim * 4, depth=1, activation=_gelu, key=keys[3],
    )
    self.norm1 = eqx.nn.LayerNorm(hidden_dim)
    self.norm2 = eqx.nn.LayerNorm(hidden_dim)
    self.dropout1 = Dropout(dropout)
    self.dropout2 = Dropout(dropout)
    self.scale = scale

  def __call__(
    self,
    h_v: NodeFeatures,
    h_e: EdgeFeatures,
    mask_v: AlphaCarbonMask | None = None,
    mask_attend: Array | None = None,
    *,
    inference: bool = False,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for gated attention message passing over ligand context.

    Parameters
    ----------
    h_v : NodeFeatures
        Node features from ligand context. Shape ``(L, M, D)``.
    h_e : EdgeFeatures
        Edge features (local ligand context). Shape ``(L, M, M, D)``.
    mask_v : AlphaCarbonMask | None
        Optional node mask. Shape ``(L, M)``. Default: None.
    mask_attend : Array | None
        Optional attention mask for message gating. Shape ``(L, M, M)``.
        Default: None.
    inference : bool
        If True, disable dropout. Default: False.
    key : PRNGKeyArray | None
        PRNG key for dropout (optional).

    Returns
    -------
    NodeFeatures
        Updated node features. Shape ``(L, M, D)``.

    """
    if key is None:
      inference = True
    keys = jax.random.split(key, 2) if key is not None else (None, None)

    # h_v: [L, M, D]
    # h_e: [L, M, M, D]

    # Expand h_v to match h_e for local context
    h_v_expand = jnp.expand_dims(h_v, axis=-2)
    h_v_expand = jnp.broadcast_to(
      h_v_expand, (*h_v_expand.shape[:-2], h_e.shape[-2], h_v.shape[-1]),
    )

    h_ev = jnp.concatenate([h_v_expand, h_e], axis=-1)

    # Message passing
    h_message = jax.vmap(jax.vmap(jax.vmap(self.w1)))(h_ev)
    h_message = _gelu(h_message)
    h_message = jax.vmap(jax.vmap(jax.vmap(self.w2)))(h_message)
    h_message = _gelu(h_message)
    h_message = jax.vmap(jax.vmap(jax.vmap(self.w3)))(h_message)

    if mask_attend is not None:
      h_message = jnp.expand_dims(mask_attend, axis=-1) * h_message

    dh = jnp.sum(h_message, axis=-2) / self.scale

    h_v = jax.vmap(jax.vmap(self.norm1))(h_v + self.dropout1(dh, key=keys[0], inference=inference))

    # MLP
    dh_dense = jax.vmap(jax.vmap(self.dense))(h_v)
    h_v = jax.vmap(jax.vmap(self.norm2))(
      h_v + self.dropout2(dh_dense, key=keys[1], inference=inference),
    )

    if mask_v is not None:
      h_v = jnp.expand_dims(mask_v, axis=-1) * h_v

    return h_v


class Decoder(eqx.Module):
  """Full decoder stacking multiple DecoderLayer passes.

  Parameters
  ----------
  layers : tuple[DecoderLayer, ...]
    Tuple of stacked decoder layers.
  node_features_dim : int
    Dimension of node features. Static (not a JAX array).
  edge_features_dim : int
    Dimension of raw edge features. Static (not a JAX array).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  layers: tuple[DecoderLayer, ...]
  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,  # This is the raw edge_features dim (128)
    hidden_features: int,
    num_layers: int = 3,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the decoder.

    Parameters
    ----------
    node_features : int
        Dimension of node features (e.g., 128).
    edge_features : int
        Dimension of raw edge features (e.g., 128).
    hidden_features : int
        Dimension of hidden layer in decoder layers.
    num_layers : int
        Number of stacked decoder layers. Default: 3.
    dropout_rate : float
        Dropout rate. Default: 0.1.
    key : PRNGKeyArray
        PRNG key for weight initialization.

    """
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features

    keys = jax.random.split(key, num_layers)

    # The context dim is [h_i, e_ij, h_j]
    edge_context_features = 2 * node_features + edge_features

    self.layers = tuple(
      DecoderLayer(
        node_features,
        edge_context_features,
        hidden_features,
        dropout_rate=dropout_rate,
        key=k,
      )
      for k in keys
    )

  def __call__(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,  # Raw 128-dim edges
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    *,
    inference: bool = False,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for unconditional decoding.

    Parameters
    ----------
    node_features : NodeFeatures
        Node features from encoder. Shape ``(L, D)``.
    edge_features : EdgeFeatures
        Raw edge features from encoder. Shape ``(L, K, D)``.
    neighbor_indices : NeighborIndices
        Neighbor indices for each node. Shape ``(L, K)``.
    mask : AlphaCarbonMask
        Alpha carbon mask. Shape ``(L,)``.
    inference : bool
        If True, disable dropout. Default: False. Passing ``key=None`` also
        forces inference. Mirrors ``call_conditional`` so callers can request
        deterministic decoding while still supplying a key.
    key : PRNGKeyArray | None
        PRNG key for dropout (optional).

    Returns
    -------
    NodeFeatures
        Decoded node features. Shape ``(L, D)``.

    Notes
    -----
    Unconditional decoding does not condition on any sequence.
    This is used for sequence generation / sampling.

    """
    if key is None:
      inference = True
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    layer_edge_features = pack_decoder_unconditional_layer_edge_features(
      node_features,
      edge_features,
      neighbor_indices,
    )

    loop_node_features = node_features
    for i, layer in enumerate(self.layers):
      loop_node_features = layer(
        loop_node_features,
        layer_edge_features,
        mask,
        inference=inference,
        key=keys[i],
      )
    return loop_node_features

  def call_conditional(
    self,
    node_features: NodeFeatures,  # h_i from encoder
    edge_features: EdgeFeatures,  # e_ij from encoder
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    one_hot_sequence: OneHotProteinSequence,
    w_s_weight: jnp.ndarray,  # Sequence embedding weight
    *,
    inference: bool = False,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for conditional decoding (scoring).

    Used during sequence scoring / log-probability computation.

    Parameters
    ----------
    node_features : NodeFeatures
        Node features from encoder. Shape ``(L, D)``.
    edge_features : EdgeFeatures
        Edge features from encoder. Shape ``(L, K, D)``.
    neighbor_indices : NeighborIndices
        Neighbor indices for each node. Shape ``(L, K)``.
    mask : AlphaCarbonMask
        Alpha carbon mask. Shape ``(L,)``.
    ar_mask : AutoRegressiveMask
        Autoregressive mask for conditional decoding. Shape ``(L, L)``.
    one_hot_sequence : OneHotProteinSequence
        One-hot encoded protein sequence. Shape ``(L, 21)``.
    w_s_weight : jnp.ndarray
        Sequence embedding weight matrix. Shape ``(21, D)``.
    inference : bool
        If True, disable dropout. Default: False.
    key : PRNGKeyArray | None
        PRNG key for dropout (optional).

    Returns
    -------
    NodeFeatures
        Decoded node features. Shape ``(L, D)``.

    Notes
    -----
    Conditional decoding conditions on the given sequence and uses
    autoregressive masking to prevent attending to future positions.

    """
    if key is None:
      inference = True
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    sequence_edge_features, mask_bw, masked_node_edge_features = (
      pack_conditional_decoder_static_edges(
        node_features,
        edge_features,
        neighbor_indices,
        one_hot_sequence,
        w_s_weight,
        ar_mask,
        mask,
      )
    )

    loop_node_features = node_features
    for i, layer in enumerate(self.layers):
      layer_edge_features = conditional_decoder_layer_edge_features(
        loop_node_features,
        sequence_edge_features,
        neighbor_indices,
        mask_bw,
        masked_node_edge_features,
      )

      # Run the layer (masking already applied to layer_edge_features)
      loop_node_features = layer(
        loop_node_features,
        layer_edge_features,
        mask,
        inference=inference,
        key=keys[i],
      )

    return loop_node_features
