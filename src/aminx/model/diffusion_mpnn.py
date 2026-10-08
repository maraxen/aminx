"""Diffusion-aware ProteinMPNN subclass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.model.mpnn import Aminx

if TYPE_CHECKING:
  from aminx.types.arrays import (
    PRNGKeyArray,
  )


class SinusoidalEmbedding(eqx.Module):
  """Sinusoidal positional embeddings for time."""

  embedding_dim: int
  max_period: float = 10000.0

  def __call__(self, timesteps: jax.Array) -> jax.Array:
    """Compute sinusoidal embeddings.

    Args:
        timesteps: [B] array of timesteps

    Returns:
        [B, embedding_dim] embeddings

    """
    half_dim = self.embedding_dim // 2
    freqs = jnp.exp(
      -jnp.log(self.max_period) * jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim,
    )
    # Expect timesteps as 1D array [B]
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    if self.embedding_dim % 2 == 1:
      embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)

    return embedding


class SwiGLU(eqx.Module):
  """SwiGLU activation layer."""

  w_gate: eqx.nn.Linear
  w_val: eqx.nn.Linear
  w_out: eqx.nn.Linear

  def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, key: PRNGKeyArray) -> None:
    """Initialize SwiGLU layer.

    Args:
        in_dim: Input dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        key: PRNG key for initialization

    """
    k1, k2, k3 = jax.random.split(key, 3)
    self.w_gate = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
    self.w_val = eqx.nn.Linear(in_dim, hidden_dim, key=k2)
    self.w_out = eqx.nn.Linear(hidden_dim, out_dim, key=k3)

  def __call__(self, x: jax.Array) -> jax.Array:
    """Apply SwiGLU activation."""
    gate = jax.nn.silu(self.w_gate(x))
    val = self.w_val(x)
    return self.w_out(gate * val)


class DiffusionAminx(Aminx):
  """ProteinMPNN extended for diffusion training."""

  t_embed_sin: SinusoidalEmbedding
  t_embed_mlp: SwiGLU

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
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize DiffusionAminx.

    Args:
        node_features: Dimension of node features.
        edge_features: Dimension of edge features.
        hidden_features: Dimension of hidden layers.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        k_neighbors: Number of neighbors for graph construction.
        num_positional_embeddings: Number of positional embeddings.
        physics_feature_dim: Dimension of additional physics features.
        num_amino_acids: Number of amino acid types.
        vocab_size: Vocabulary size.
        key: PRNG key for initialization.

    """
    key, t_key = jax.random.split(key)

    super().__init__(
      node_features=node_features,
      edge_features=edge_features,
      hidden_features=hidden_features,
      num_encoder_layers=num_encoder_layers,
      num_decoder_layers=num_decoder_layers,
      k_neighbors=k_neighbors,
      num_positional_embeddings=num_positional_embeddings,
      physics_feature_dim=physics_feature_dim,
      num_amino_acids=num_amino_acids,
      vocab_size=vocab_size,
      key=key,
    )

    # Timestep embedding: Sinusoidal -> SwiGLU MLP -> Node Features
    # We project to node_features_dim so we can add it to node features
    self.t_embed_sin = SinusoidalEmbedding(node_features)
    self.t_embed_mlp = SwiGLU(node_features, node_features * 4, node_features, key=t_key)

  @classmethod
  def stage_schema(cls) -> dict[str, type | None]:
    """Returns {stage_name: type_alias} for this Diffusion MPNN variant."""
    from aminx.types.stages import (
      ARLogitTransformFn,
      ConditionalDecodeFn,
      FeaturizeFn,
      LogitTransformFn,
      ProteinEncodeFn,
      UnconditionalDecodeFn,
    )

    return {
      "featurize": FeaturizeFn,
      "encode": ProteinEncodeFn,
      "decode": ConditionalDecodeFn | UnconditionalDecodeFn,
      "logit_transform": LogitTransformFn,
      "ar_logit_transform": ARLogitTransformFn,
      "encoder_state_fn": None,
    }

  def __call__(  # type: ignore[override]
    self,
    prng_key: PRNGKeyArray,
    coords: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    backbone_noise: float | jax.Array = 0.0,
    *,
    backbone_noise_mode: str = "direct",
    structure_mapping: jax.Array | None = None,
    initial_node_features: jax.Array | None = None,
    timestep: jax.Array | None = None,
    inference: bool = True,
    **kwargs: Any,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Encoder-only entry point with diffusion timestep support.

    ``inference`` is forwarded to ``Aminx`` so encoder dropout follows the caller.

    Returns:
        3-tuple of (node_features, edge_features, edge_indices).

    """
    # 1. Base Encode
    node_features, edge_features, edge_indices = super().__call__(
      prng_key,
      coords,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      backbone_noise_mode=backbone_noise_mode,
      structure_mapping=structure_mapping,
      initial_node_features=initial_node_features,
      inference=inference,
      **kwargs,
    )

    # 2. Inject Timestep Embedding (if provided)
    if timestep is not None:
      t_embed = self.t_embed_sin(timestep)
      t_embed = self.t_embed_mlp(t_embed)  # [B, C]

      # Match node_features shape [B, L, C]
      t_embed = t_embed[:, None, :]  # [B, 1, C]

      node_features = node_features + t_embed

    return node_features, edge_features, edge_indices
