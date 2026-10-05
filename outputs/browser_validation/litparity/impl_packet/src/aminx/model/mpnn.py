"""Main ProteinMPNN model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

from aminx.model.capabilities import (
  PRXTEIN_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from aminx.model.decoder import Decoder
from aminx.model.encoder import (
  Encoder,
  PhysicsEncoder,
)
from aminx.model.features import ProteinFeatures

if TYPE_CHECKING:
  from aminx.types.arrays import PRNGKeyArray
else:
  PRNGKeyArray = Any


class Aminx(eqx.Module):
  """The complete end-to-end ProteinMPNN model."""

  features: ProteinFeatures
  encoder: Encoder | PhysicsEncoder
  decoder: Decoder

  w_s_embed: eqx.nn.Embedding
  w_out: eqx.nn.Linear

  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)
  num_decoder_layers: int = eqx.field(static=True)
  capabilities: ModelCapabilities = eqx.field(static=True, default=PRXTEIN_MPNN_CAPABILITIES)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    k_neighbors: int,
    num_positional_embeddings: int = 16,
    physics_feature_dim: int | None = None,
    num_amino_acids: int = 21,
    vocab_size: int = 21,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features
    self.num_decoder_layers = num_decoder_layers

    keys = jax.random.split(key, 5)

    self.features = ProteinFeatures(
      node_features,
      edge_features,
      k_neighbors,
      num_positional_embeddings=num_positional_embeddings,
      key=keys[0],
    )
    self.encoder = (
      Encoder(
        node_features,
        edge_features,
        hidden_features,
        num_encoder_layers,
        dropout_rate=dropout_rate,
        key=keys[1],
      )
      if physics_feature_dim is None
      else PhysicsEncoder(
        node_features,
        edge_features,
        hidden_features,
        num_encoder_layers,
        dropout_rate,
        physics_feature_dim,
        key=keys[1],
      )
    )
    self.decoder = Decoder(
      node_features,
      edge_features,
      hidden_features,
      num_decoder_layers,
      dropout_rate=dropout_rate,
      key=keys[2],
    )
    self.w_s_embed = eqx.nn.Embedding(
      num_embeddings=vocab_size,
      embedding_size=node_features,
      key=keys[3],
    )
    self.w_out = eqx.nn.Linear(node_features, num_amino_acids, key=keys[4])

  def __call__(
    self,
    coords: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    *,
    prng_key: PRNGKeyArray | None = None,
    backbone_noise: float | jax.Array = 0.0,
    backbone_noise_mode: str = "direct",
    structure_mapping: jax.Array | None = None,
    y: jax.Array | None = None,
    y_t: jax.Array | None = None,
    y_m: jax.Array | None = None,
    inference: bool = True,
    initial_node_features: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-state encoder: features + encoder for one conformational state.

    This is the low-level building block called by make_encode_fn (inference/encode.py).
    It is NOT an inference entry point — use the Sprint 2 inference API:
    build_inference_bundle + score_conditional/score_unconditional/sample_autoregressive kernels.

    Args:
        coords: (L, 4, 3) atom coordinates for one state.
        mask: (L,) residue mask.
        residue_index: (L,) residue indices.
        chain_index: (L,) chain indices.
        prng_key: PRNG key for encoder dropout.
        backbone_noise: scalar backbone noise level.
        backbone_noise_mode: mode for backbone noise ("direct" or other).
        structure_mapping: optional structure isolation mapping.
        y: accepted but ignored (for API compatibility with LigandMPNN).
        y_t: accepted but ignored (for API compatibility with LigandMPNN).
        y_m: accepted but ignored (for API compatibility with LigandMPNN).
        inference: accepted but ignored (kept for symmetry with downstream calls).

    Returns:
        (node_features, edge_features, neighbor_indices) for this state.

    """
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)

    edge_features, neighbor_indices, feat_node_features, _ = self.features(
      prng_key,
      coords,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      backbone_noise_mode=backbone_noise_mode,
      structure_mapping=structure_mapping,
    )
    # initial_node_features overrides ProteinFeatures output (always None for soluble);
    # used by membrane models to pass one-hot physics labels through PhysicsEncoder.
    effective_node_features = (
      initial_node_features if initial_node_features is not None else feat_node_features
    )
    node_features, edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
      initial_node_features=effective_node_features,
      key=prng_key,
    )
    return node_features, edge_features, neighbor_indices
