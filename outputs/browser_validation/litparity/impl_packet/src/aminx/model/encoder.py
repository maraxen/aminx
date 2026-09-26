"""Encoder module for Aminx.

Contains ``EncoderLayer``, ``Encoder``, ``PhysicsEncoder``, and small **orchestration
helpers** (``pack_encoder_context``, ``encoder_forward_with_int_neighbors``) shared from
``mpnn.Aminx`` without importing the full model.
"""

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
    EdgeFeatures,
    NeighborIndices,
    NodeFeatures,
  )

# Type alias for PRNG keys
PRNGKeyArray = jax.Array

# Layer normalization
LayerNorm = eqx.nn.LayerNorm
_gelu = partial(jax.nn.gelu, approximate=False)


class EncoderLayer(eqx.Module):
  """Single message-passing encoder layer for ProteinMPNN.

  Processes node and edge features via [h_i, e_ij, h_j] concatenation,
  updates node features via message aggregation and feedforward network,
  then updates edge features. Used in both protein-only and ligand-conditioned models.

  Parameters
  ----------
  edge_message_mlp : eqx.nn.MLP
      Edge message computation MLP: (embed_input_size) -> edge_features.
  norm1 : LayerNorm
      Layer normalization for aggregated node messages. Shape: (node_features,).
  dense : eqx.nn.MLP
      Node feature feedforward MLP: (node_features) -> (node_features).
  norm2 : LayerNorm
      Layer normalization after dense update. Shape: (node_features,).
  edge_update_mlp : eqx.nn.MLP
      Edge update MLP: (embed_input_size) -> edge_features.
  norm3 : LayerNorm
      Layer normalization for edge updates. Shape: (edge_features,).
  dropout1 : Dropout
      Dropout after message aggregation.
  dropout2 : Dropout
      Dropout after dense feedforward.
  dropout3 : Dropout
      Dropout after edge update.
  node_features_dim : int, static
      Dimension of node features.
  edge_features_dim : int, static
      Dimension of edge features.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  edge_message_mlp: eqx.nn.MLP
  norm1: LayerNorm
  dense: eqx.nn.MLP
  norm2: LayerNorm
  edge_update_mlp: eqx.nn.MLP
  norm3: LayerNorm
  dropout1: Dropout
  dropout2: Dropout
  dropout3: Dropout
  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the encoder layer.

    Parameters
    ----------
    node_features : int
        Dimension of node features.
    edge_features : int
        Dimension of edge features.
    hidden_features : int
        Dimension of hidden features in feedforward networks.
    dropout_rate : float, optional
        Dropout rate (default: 0.1).
    key : PRNGKeyArray
        PRNG key for weight initialization.

    """
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features

    keys = jax.random.split(key, 7)
    embed_input_size = edge_features + node_features * 2

    self.dropout1 = Dropout(dropout_rate)
    self.dropout2 = Dropout(dropout_rate)
    self.dropout3 = Dropout(dropout_rate)

    self.edge_message_mlp = eqx.nn.MLP(
      in_size=embed_input_size,
      out_size=edge_features,
      width_size=hidden_features,
      depth=2,
      activation=_gelu,
      key=keys[3],
    )
    self.norm1 = LayerNorm(node_features)
    self.dense = eqx.nn.MLP(
      in_size=node_features,
      out_size=node_features,
      width_size=embed_input_size + hidden_features,
      depth=1,
      activation=_gelu,
      key=keys[4],
    )
    self.norm2 = LayerNorm(node_features)
    self.edge_update_mlp = eqx.nn.MLP(
      in_size=embed_input_size,
      out_size=edge_features,
      width_size=edge_features,
      depth=2,
      activation=_gelu,
      key=keys[5],
    )
    self.norm3 = LayerNorm(edge_features)

  def _get_mlp_input(
    self,
    h: NodeFeatures,
    e: EdgeFeatures,
    neighbor_indices: NeighborIndices,
  ) -> jax.Array:
    """Return the input tensor [h_i, e_ij, h_j] for edge_message_mlp."""
    e_with_neighbors = concatenate_neighbor_nodes(h, e, neighbor_indices)
    node_expanded = jnp.tile(jnp.expand_dims(h, -2), [1, e_with_neighbors.shape[-2], 1])
    return jnp.concatenate([node_expanded, e_with_neighbors], -1)

  def __call__(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    mask_attend: jnp.ndarray | None = None,
    scale: float = 30.0,
    *,
    inference: bool = False,
    key: PRNGKeyArray | None = None,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Forward pass for the encoder layer.

    Computes edge messages via [h_i, e_ij, h_j] concatenation and MLP,
    aggregates to node features, updates via feedforward, then updates edges.

    Parameters
    ----------
    node_features : NodeFeatures
        Node features. Shape: (L, H_n).
    edge_features : EdgeFeatures
        Edge features (neighbor-aggregated). Shape: (L, K, H_e) where K is neighbor count.
    neighbor_indices : NeighborIndices
        Neighbor indices for gather operations. Shape: (L, K).
    mask : AlphaCarbonMask
        Alpha carbon mask (1.0 for valid, 0.0 for invalid). Shape: (L,).
    mask_attend : jnp.ndarray, optional
        Attention mask for neighbor validity. Shape: (L, K).
    scale : float, optional
        Divisor applied to summed neighbor messages before the residual update,
        controlling the effective temperature of message aggregation. Set to
        30.0 to match the ProteinMPNN reference implementation (Dauparas et al.
        2022); smaller values sharpen aggregation, larger values smooth it.
    inference : bool, optional
        Whether in inference mode (no dropout). Default: False.
    key : PRNGKeyArray, optional
        PRNG key for dropout. If None, inference mode is enabled.

    Returns
    -------
    tuple[NodeFeatures, EdgeFeatures]
        Updated node and edge features with same shapes as inputs.

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187

    """
    if key is None:
      inference = True
    keys = jax.random.split(key, 3) if key is not None else (None, None, None)

    mlp_input = self._get_mlp_input(node_features, edge_features, neighbor_indices)
    message = jax.vmap(jax.vmap(self.edge_message_mlp))(mlp_input)

    # Apply attention mask to zero out messages from invalid neighbors
    if mask_attend is not None:
      message = message * mask_attend[..., None]

    aggregated_message = jnp.sum(message, -2) / scale

    aggregated_message = self.dropout1(aggregated_message, key=keys[0], inference=inference)

    node_features = node_features + aggregated_message
    node_features = jax.vmap(self.norm1)(node_features)

    dense_out = jax.vmap(self.dense)(node_features)
    dense_out = self.dropout2(dense_out, key=keys[1], inference=inference)

    node_features = node_features + dense_out
    node_features = jax.vmap(self.norm2)(node_features)
    node_features = mask[:, None] * node_features

    edge_features_cat = concatenate_neighbor_nodes(node_features, edge_features, neighbor_indices)
    node_features_expand = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, edge_features_cat.shape[-2], 1],
    )
    mlp_input_edge_update = jnp.concatenate([node_features_expand, edge_features_cat], -1)
    edge_message = jax.vmap(jax.vmap(self.edge_update_mlp))(mlp_input_edge_update)
    edge_message = self.dropout3(edge_message, key=keys[2], inference=inference)

    edge_features = edge_features + edge_message
    edge_features = jax.vmap(jax.vmap(self.norm3))(edge_features)

    return node_features, edge_features


class Encoder(eqx.Module):
  """Multi-layer encoder stacking EncoderLayer message-passing passes.

  Iteratively refines node and edge features through a sequence of encoder layers.
  Each layer updates features via message passing and is used in both protein-only
  and ligand-conditioned models.

  Parameters
  ----------
  layers : tuple[EncoderLayer, ...]
      Tuple of encoder layers applied sequentially.
  node_feature_dim : int, static
      Dimension of node features (cached from initialization).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  layers: tuple[EncoderLayer, ...]

  node_feature_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_layers: int = 3,
    dropout_rate: float = 0.1,
    _physics_feature_dim: int | None = None,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the encoder.

    Parameters
    ----------
    node_features : int
        Dimension of node features.
    edge_features : int
        Dimension of edge features.
    hidden_features : int
        Dimension of hidden features in feedforward networks.
    num_layers : int, optional
        Number of stacked encoder layers (default: 3).
    dropout_rate : float, optional
        Dropout rate (default: 0.1).
    _physics_feature_dim : int, optional
        Unused; reserved for subclass compatibility.
    key : PRNGKeyArray
        PRNG key for weight initialization.

    """
    self.node_feature_dim = node_features
    keys = jax.random.split(key, num_layers)
    self.layers = tuple(
      EncoderLayer(
        node_features,
        edge_features,
        hidden_features,
        dropout_rate=dropout_rate,
        key=k,
      )
      for k in keys
    )

  def __call__(
    self,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    initial_node_features: jnp.ndarray | None = None,
    scale: float = 30.0,
    *,
    inference: bool = False,
    key: PRNGKeyArray | None = None,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Forward pass through all encoder layers.

    Processes edge and initial node features through stacked layers, updating
    both node and edge representations iteratively.

    Parameters
    ----------
    edge_features : EdgeFeatures
        Input edge features (neighbor-aggregated). Shape: (L, K, H_e).
    neighbor_indices : NeighborIndices
        Neighbor indices for gather operations. Shape: (L, K).
    mask : AlphaCarbonMask
        Alpha carbon mask. Shape: (L,).
    initial_node_features : jnp.ndarray, optional
        Initial node features. If None, zero-initialized. Shape: (L, H_n).
    scale : float, optional
        Attention scaling constant (default: 30.0).
    inference : bool, optional
        Whether in inference mode. Default: False.
    key : PRNGKeyArray, optional
        PRNG key for dropout.

    Returns
    -------
    tuple[NodeFeatures, EdgeFeatures]
        Final node and edge features. Shapes: (L, H_n), (L, K, H_e).

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187

    """
    if key is None:
      inference = True
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    mask_2d = mask[:, None] * mask[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, neighbor_indices.astype(jnp.int32), axis=1)

    node_features = (
      jnp.zeros((edge_features.shape[0], self.node_feature_dim))
      if initial_node_features is None
      else initial_node_features
    )

    for i, layer in enumerate(self.layers):
      node_features, edge_features = layer(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        mask_attend=mask_attend,
        scale=scale,
        inference=inference,
        key=keys[i],
      )
    return node_features, edge_features


class PhysicsEncoder(eqx.Module):
  """Physics-conditioned encoder extension for membrane protein models.

  Extends the standard encoder with a projection from physics features
  (e.g., membrane accessibility) to node feature space, enabling conditioning
  on physical properties during message passing.

  Parameters
  ----------
  layers : tuple[EncoderLayer, ...]
      Tuple of encoder layers applied sequentially.
  physics_projection : eqx.nn.Linear
      Projects physics features to node feature space. Shape: (P,) -> (H_n).
  physics_norm : eqx.nn.LayerNorm
      Layer normalization of projected physics features. Shape: (H_n,).
  physics_w_v : eqx.nn.Linear
      Value projection of normalized physics (W_v in membrane paper). Shape: (H_n,) -> (H_n,).
  node_feature_dim : int, static
      Dimension of node features.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  layers: tuple[EncoderLayer, ...]
  physics_projection: eqx.nn.Linear
  physics_norm: eqx.nn.LayerNorm
  physics_w_v: eqx.nn.Linear
  node_feature_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_layers: int = 3,
    dropout_rate: float = 0.1,
    physics_feature_dim: int = 0,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the physics-conditioned encoder.

    Parameters
    ----------
    node_features : int
        Dimension of node features.
    edge_features : int
        Dimension of edge features.
    hidden_features : int
        Dimension of hidden features in feedforward networks.
    num_layers : int, optional
        Number of stacked encoder layers (default: 3).
    dropout_rate : float, optional
        Dropout rate (default: 0.1).
    physics_feature_dim : int, optional
        Dimension of physics features (e.g., membrane properties). Default: 0.
    key : PRNGKeyArray
        PRNG key for weight initialization.

    """
    self.node_feature_dim = node_features
    # keys[0] → physics_projection, keys[1] → physics_w_v, keys[2:] → encoder layers
    keys = jax.random.split(key, num_layers + 2)

    self.physics_projection = eqx.nn.Linear(
      physics_feature_dim,
      node_features,
      key=keys[0],
    )
    self.physics_norm = eqx.nn.LayerNorm(node_features)
    self.physics_w_v = eqx.nn.Linear(node_features, node_features, key=keys[1])

    self.layers = tuple(
      EncoderLayer(
        node_features,
        edge_features,
        hidden_features,
        dropout_rate=dropout_rate,
        key=k,
      )
      for k in keys[2:]
    )

  def __call__(
    self,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    initial_node_features: jnp.ndarray | None = None,
    scale: float = 30.0,
    *,
    inference: bool = False,
    key: PRNGKeyArray | None = None,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Forward pass through the physics-conditioned encoder.

    Projects physics features to node space (W_v(LayerNorm(projection(physics)))),
    then processes through stacked encoder layers.

    Parameters
    ----------
    edge_features : EdgeFeatures
        Input edge features (neighbor-aggregated). Shape: (L, K, H_e).
    neighbor_indices : NeighborIndices
        Neighbor indices for gather operations. Shape: (L, K).
    mask : AlphaCarbonMask
        Alpha carbon mask. Shape: (L,).
    initial_node_features : jnp.ndarray, optional
        Physics features to project. Shape: (L, P) where P is physics_feature_dim.
        If None, zero-initialized node features used instead.
    scale : float, optional
        Attention scaling constant (default: 30.0).
    inference : bool, optional
        Whether in inference mode. Default: False.
    key : PRNGKeyArray, optional
        PRNG key for dropout.

    Returns
    -------
    tuple[NodeFeatures, EdgeFeatures]
        Final node and edge features. Shapes: (L, H_n), (L, K, H_e).

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187

    """
    if key is None:
      inference = True
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    mask_2d = mask[:, None] * mask[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, neighbor_indices.astype(jnp.int32), axis=1)

    # Membrane init: W_v(LayerNorm(node_embedding(one_hot(labels))))
    # All three layers operate per-residue → vmap over L.
    node_features = (
      jnp.zeros((edge_features.shape[0], self.node_feature_dim))
      if initial_node_features is None
      else jax.vmap(self.physics_w_v)(
        jax.vmap(self.physics_norm)(
          jax.vmap(self.physics_projection)(initial_node_features),
        ),
      )
    )

    for i, layer in enumerate(self.layers):
      node_features, edge_features = layer(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        mask_attend=mask_attend,
        scale=scale,
        inference=inference,
        key=keys[i],
      )
    return node_features, edge_features


def pack_encoder_context(
  node_features: jax.Array,
  edge_features: jax.Array,
  neighbor_indices: jax.Array,
  mask_fw: jax.Array,
) -> jax.Array:
  """Pack neighbor context tensor for encoder MLP input during AR decoding.

  Gathers edge features from neighbors, concatenates with node features,
  and applies forward mask. Used in AR sampling loops to construct the
  encoder context (packed neighbors) for each position.

  Parameters
  ----------
  node_features : jax.Array
      Node features. Shape: (L, H_n).
  edge_features : jax.Array
      Neighbor-aggregated edge features. Shape: (L, K, H_e).
  neighbor_indices : jax.Array
      Neighbor indices for gather. Shape: (L, K).
  mask_fw : jax.Array
      Forward (feasible position) mask. Shape: (L,).

  Returns
  -------
  jax.Array
      Packed encoder context: [zeros‖edge] gathered, [node‖that] gathered,
      then masked. Shape: (L, K, H_e + H_n).

  Notes
  -----
  Matches the legacy sequence: zeros gather, node gather, then masking.
  Batched graphs: vmap(pack_encoder_context) over batch dimension.
  Used in AR scan and ligand state_vmap_exact call sites.

  """
  encoder_edge_neighbors = concatenate_neighbor_nodes(
    jnp.zeros_like(node_features),
    edge_features,
    neighbor_indices,
  )
  encoder_context = concatenate_neighbor_nodes(
    node_features,
    encoder_edge_neighbors,
    neighbor_indices,
  )
  return encoder_context * mask_fw[..., None]


def encoder_forward_with_int_neighbors(
  encoder: Encoder | PhysicsEncoder,
  edge_features: jax.Array,
  neighbor_indices: jax.Array,
  mask: jax.Array,
  initial_node_features: jax.Array,
  *,
  inference: bool,
  key: PRNGKeyArray | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Run encoder and return int32 neighbor indices for downstream gathers.

  Executes the encoder forward pass and casts neighbor indices to int32
  for compatibility with downstream gather operations (e.g., pack_encoder_context).

  Parameters
  ----------
  encoder : Encoder | PhysicsEncoder
      Encoder module to apply.
  edge_features : jax.Array
      Neighbor-aggregated edge features. Shape: (L, K, H_e).
  neighbor_indices : jax.Array
      Neighbor indices. Shape: (L, K).
  mask : jax.Array
      Alpha carbon mask. Shape: (L,).
  initial_node_features : jax.Array
      Initial node features (or physics features for PhysicsEncoder).
      Shape: (L, H_n) or (L, P).
  inference : bool
      Whether in inference mode (no dropout).
  key : PRNGKeyArray, optional
      PRNG key for dropout. If None, inference mode enabled.

  Returns
  -------
  tuple[jax.Array, jax.Array, jax.Array]
      (updated_node_features, updated_edge_features, int32_neighbor_indices).
      Shapes: (L, H_n), (L, K, H_e), (L, K).

  Notes
  -----
  Callers must pass the **same** ``key`` / ``inference`` pairing as the inlined path.
  Example: wave ``encode_one`` uses one ``k`` for both ``features`` and encoder;
  scoring paths use ``k_feat`` vs ``k_enc`` separately — only the encoder tail
  is unified here.

  """
  nf2, ef2 = encoder(
    edge_features,
    neighbor_indices,
    mask,
    initial_node_features=initial_node_features,
    inference=inference,
    key=key,
  )
  return nf2, ef2, neighbor_indices.astype(jnp.int32)
